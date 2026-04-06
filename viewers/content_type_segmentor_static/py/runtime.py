from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


BYTE_VOCAB_SIZE = 256
PAD_BYTE_ID = 256
NUM_TOKEN_EMBEDDINGS = BYTE_VOCAB_SIZE + 1

_VISIBLE_ASCII_BYTES = tuple(range(0x20, 0x7F))
_WHITESPACE_BYTES = (0x09, 0x0A, 0x0D)
_CURRENCY_BYTE_ID = np.int32(0xA4)
_ALLOWED_MODEL_BYTE_VALUES = np.array(
    sorted(set(_VISIBLE_ASCII_BYTES) | set(_WHITESPACE_BYTES) | {int(_CURRENCY_BYTE_ID)}),
    dtype=np.int32,
)
_ALLOWED_MODEL_TOKEN_VALUES = np.array(
    sorted(set(_ALLOWED_MODEL_BYTE_VALUES.tolist()) | {int(PAD_BYTE_ID)}),
    dtype=np.int32,
)

_PLACEHOLDER_CHAR = "\u00A4"
_ALLOWED_TEXT_CHARS = {chr(b) for b in _VISIBLE_ASCII_BYTES}
_ALLOWED_TEXT_CHARS.update({" ", "\n", "\t", _PLACEHOLDER_CHAR})
_VISUAL_WHITESPACE_SET = frozenset((" ", "\t", "\n"))
_INLINE_WHITESPACE_SET = frozenset((" ", "\t"))
_BOUNDARY_SNAP_DELIMITER_CHARS = frozenset(
    ("<", ">", "/", "\\", '"', "'", "`", "(", ")", "[", "]", "{", "}", ",", ";", ":", "=")
)
_BOUNDARY_SNAP_ADJACENT_CHARS = _BOUNDARY_SNAP_DELIMITER_CHARS | frozenset((" ", "\t"))
_BOUNDARY_SNAP_PROB_MARGIN = 0.1
_BOUNDARY_SNAP_DELIMITER_PROB_MARGIN = 0.15
_BOUNDARY_SNAP_WHITESPACE_PROB_MARGIN = 0.05
_BOUNDARY_SNAP_MIN_IMPROVEMENT = 0.75
_BOUNDARY_WRAP_OPEN_TO_CLOSE = {
    '"': '"',
    "'": "'",
    "`": "`",
    "(": ")",
    "[": "]",
    "{": "}",
}
_BOUNDARY_WRAP_QUOTE_CHARS = frozenset(('"', "'", "`"))
_BOUNDARY_WRAP_PROB_MARGIN = 0.20
_BOUNDARY_WRAP_PAIR_BONUS = 1.35
_BOUNDARY_WRAP_QUOTE_BONUS = 0.35
_BOUNDARY_WRAP_SHELL_DELIMITER_EJECT_BONUS = 0.80
_BOUNDARY_WRAP_SHELL_DELIMITER_SWALLOW_PENALTY = 0.45
_BOUNDARY_WRAP_SHELL_QUOTE_EJECT_BONUS = 0.35
_BOUNDARY_WRAP_MIN_IMPROVEMENT = 0.70
_LOCAL_HOST_POSTPROCESS_RULE_NAMES: Tuple[Tuple[str, str, str], ...] = (
    ("json", "javascript_typescript", "json"),
)
_LOCAL_HOST_SINGLE_SIDE_MIN_CHARS = 8
_LAYER_NORM_EPS = 1e-6


_MODEL: Optional["NumpyMambaSegmentor"] = None
_NUMERIC_RE = re.compile(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?")


def _normalize_input_text(text: Optional[str]) -> str:
    if not text:
        return ""
    out_chars: List[str] = []
    for ch in text:
        if ch == "\r":
            ch = "\n"
        if ch in _ALLOWED_TEXT_CHARS:
            out_chars.append(ch)
        else:
            out_chars.append(_PLACEHOLDER_CHAR)
    return "".join(out_chars)


def _sanitize_model_bytes(arr: np.ndarray) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.uint8)
    if arr_np.size == 0:
        return arr_np
    invalid = ~np.isin(arr_np.astype(np.int32), _ALLOWED_MODEL_BYTE_VALUES)
    if np.any(invalid):
        arr_np = arr_np.copy()
        arr_np[invalid] = np.uint8(_CURRENCY_BYTE_ID)
    return arr_np


def _sanitize_model_tokens(arr: np.ndarray) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.int32)
    if arr_np.size == 0:
        return arr_np
    invalid = ~np.isin(arr_np, _ALLOWED_MODEL_TOKEN_VALUES)
    if np.any(invalid):
        arr_np = arr_np.copy()
        arr_np[invalid] = int(_CURRENCY_BYTE_ID)
    return arr_np


def _window_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(length, 0),), dtype=np.float32)
    positions = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
    sigma = 0.5
    weights = np.exp(-0.5 * (positions / sigma) ** 2)
    return weights.astype(np.float32)


def _build_window_spans(length: int, chunk_size: int, stride: Optional[int] = None) -> List[Tuple[int, int]]:
    n = int(length)
    if n <= 0:
        return []
    win = max(64, int(chunk_size))
    if n <= win:
        return [(0, n)]
    step = int(stride) if stride is not None and int(stride) > 0 else max(1, win // 2)
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + win, n)
        spans.append((int(start), int(end)))
        if end >= n:
            break
        start += step
    return spans


def _resolve_local_host_postprocess_rules(label_order: Sequence[str]) -> Tuple[Tuple[int, int, str], ...]:
    index_by_name = {str(name): idx for idx, name in enumerate(label_order)}
    resolved: List[Tuple[int, int, str]] = []
    for inner_name, host_name, rule_kind in _LOCAL_HOST_POSTPROCESS_RULE_NAMES:
        inner_idx = index_by_name.get(inner_name)
        host_idx = index_by_name.get(host_name)
        if inner_idx is None or host_idx is None:
            continue
        if int(inner_idx) == int(host_idx):
            continue
        resolved.append((int(inner_idx), int(host_idx), str(rule_kind)))
    return tuple(resolved)


def _resolve_label_id(label_order: Sequence[str], name: str) -> Optional[int]:
    lookup = {str(label): idx for idx, label in enumerate(label_order)}
    value = lookup.get(str(name))
    return int(value) if value is not None else None


def _new_lock_mask(length: int) -> np.ndarray:
    return np.zeros((max(0, int(length)),), dtype=bool)


def _mark_locked_range(mask: np.ndarray, start: int, end: int) -> None:
    lo = max(0, int(start))
    hi = min(int(mask.shape[0]), int(end))
    if lo < hi:
        mask[lo:hi] = True


def _merge_lock_masks(base: np.ndarray, update: np.ndarray) -> np.ndarray:
    if int(base.shape[0]) != int(update.shape[0]):
        return base
    np.logical_or(base, update, out=base)
    return base


def _find_next_backtick_run(text: str, start: int, end: int, *, min_len: int) -> Tuple[int, int]:
    pos = max(0, int(start))
    line_end = min(len(text), int(end))
    while pos < line_end:
        if text[pos] != "`":
            pos += 1
            continue
        run_end = pos
        while run_end < line_end and text[run_end] == "`":
            run_end += 1
        if (run_end - pos) >= int(min_len):
            return pos, run_end
        pos = run_end
    return -1, -1


def _infer_uniform_body_label(
    labels: np.ndarray,
    char_probs: np.ndarray,
    start: int,
    end: int,
    *,
    markdown_label: int,
) -> int:
    lo = max(0, int(start))
    hi = min(int(labels.shape[0]), int(end))
    if lo >= hi:
        return int(markdown_label)
    window_labels = np.asarray(labels[lo:hi], dtype=np.int32)
    if window_labels.size == 0:
        return int(markdown_label)
    counts: Dict[int, int] = {}
    supports: Dict[int, float] = {}
    for offset, label in enumerate(window_labels):
        label_int = int(label)
        counts[label_int] = counts.get(label_int, 0) + 1
        if 0 <= label_int < int(char_probs.shape[1]):
            supports[label_int] = supports.get(label_int, 0.0) + float(char_probs[lo + offset, label_int])
        else:
            supports.setdefault(label_int, 0.0)
    best_count = max(counts.values())
    candidates = [label for label, count in counts.items() if count == best_count]
    if len(candidates) == 1:
        return int(candidates[0])
    best_support = max(supports.get(label, 0.0) for label in candidates)
    support_candidates = [label for label in candidates if supports.get(label, 0.0) >= (best_support - 1e-9)]
    if int(markdown_label) in support_candidates:
        return int(markdown_label)
    return int(min(support_candidates))


def _layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    normalized = centered / np.sqrt(var + np.float32(_LAYER_NORM_EPS))
    return normalized * scale[None, :] + bias[None, :]


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-abs_x))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    denom = np.maximum(np.sum(exp_x, axis=-1, keepdims=True), 1e-9)
    return exp_x / denom


def _depthwise_conv_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    k = int(kernel.shape[0])
    pad_left = max(0, (k - 1) // 2)
    pad_right = max(0, (k - 1) - pad_left)
    padded = np.pad(x, ((pad_left, pad_right), (0, 0)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=k, axis=0)
    if windows.ndim != 3:
        raise ValueError(f"Unexpected depthwise conv window shape: {windows.shape}")
    if windows.shape[1] != k:
        windows = np.moveaxis(windows, -1, 1)
    return np.einsum("tkc,kc->tc", windows, kernel, optimize=True) + bias[None, :]


def _selective_scan(
    x_in: np.ndarray,
    dt_in: np.ndarray,
    b_in: np.ndarray,
    c_in: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    x_f32 = np.asarray(x_in, dtype=np.float32)
    dt_f32 = np.asarray(dt_in, dtype=np.float32)
    b_f32 = np.asarray(b_in, dtype=np.float32)
    c_f32 = np.asarray(c_in, dtype=np.float32)
    a_f32 = np.asarray(a, dtype=np.float32)
    d_f32 = np.asarray(d, dtype=np.float32)

    length = int(x_f32.shape[0])
    d_inner = int(x_f32.shape[1])
    d_state = int(b_f32.shape[1])
    state = np.zeros((d_inner, d_state), dtype=np.float32)
    out = np.zeros((length, d_inner), dtype=np.float32)
    for idx in range(length):
        dt_t = dt_f32[idx][:, None]
        a_t = np.exp(dt_t * a_f32)
        state = a_t * state + x_f32[idx][:, None] * (dt_t * b_f32[idx][None, :])
        out[idx] = np.sum(state * c_f32[idx][None, :], axis=-1) + x_f32[idx] * d_f32
    return out


def _threshold_predictions(probs: np.ndarray, other_threshold: Optional[float], other_id: int) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float32)
    pred = np.argmax(arr, axis=-1).astype(np.int32)
    thr = float(other_threshold) if other_threshold is not None else 0.0
    if thr > 0.0 and pred.size > 0:
        conf = np.max(arr, axis=-1)
        pred[conf < thr] = int(other_id)
    return pred


def _resolve_threshold(value: Optional[float], default: float) -> float:
    threshold = default if value is None else float(value)
    if not np.isfinite(threshold):
        threshold = default
    return float(np.clip(threshold, 0.0, 1.0))


def _relabel_whitespace_from_neighbors(text: str, labels: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(text)
    if n == 0 or labels.size == 0 or labels.shape[0] != n:
        return labels, probs
    if not any(ch in _VISUAL_WHITESPACE_SET for ch in text):
        return labels, probs

    new_labels = np.asarray(labels, dtype=np.int32).copy()
    new_probs = np.asarray(probs, dtype=np.float32).copy()

    line_starts: List[int] = [0]
    for idx, ch in enumerate(text):
        if ch == "\n" and idx + 1 < n:
            line_starts.append(idx + 1)
    line_starts = sorted(set(line_starts))
    line_segments: List[Tuple[int, int]] = []
    for idx, start in enumerate(line_starts):
        end = line_starts[idx + 1] if idx + 1 < len(line_starts) else n
        if start < end:
            line_segments.append((start, end))

    left_same_line = [-1] * n
    right_same_line = [-1] * n
    for start, end in line_segments:
        last_non_ws = -1
        for pos in range(start, end):
            if text[pos] not in _VISUAL_WHITESPACE_SET:
                last_non_ws = pos
            left_same_line[pos] = last_non_ws
        last_non_ws = -1
        for pos in range(end - 1, start - 1, -1):
            if text[pos] not in _VISUAL_WHITESPACE_SET:
                last_non_ws = pos
            right_same_line[pos] = last_non_ws

    left_any = [-1] * n
    right_any = [-1] * n
    last_non_ws = -1
    for pos in range(n):
        if text[pos] not in _VISUAL_WHITESPACE_SET:
            last_non_ws = pos
        left_any[pos] = last_non_ws
    last_non_ws = -1
    for pos in range(n - 1, -1, -1):
        if text[pos] not in _VISUAL_WHITESPACE_SET:
            last_non_ws = pos
        right_any[pos] = last_non_ws

    for idx, ch in enumerate(text):
        if ch not in _VISUAL_WHITESPACE_SET:
            continue
        src = -1
        ls = left_same_line[idx]
        rs = right_same_line[idx]
        if ls != -1 or rs != -1:
            if ls == -1:
                src = rs
            elif rs == -1:
                src = ls
            else:
                dist_l = idx - ls
                dist_r = rs - idx
                src = ls if dist_l <= dist_r else rs
        else:
            la = left_any[idx]
            ra = right_any[idx]
            if la != -1 or ra != -1:
                if la == -1:
                    src = ra
                elif ra == -1:
                    src = la
                else:
                    dist_l = idx - la
                    dist_r = ra - idx
                    src = la if dist_l <= dist_r else ra
        if src == -1:
            continue
        new_labels[idx] = int(labels[src])
        new_probs[idx] = probs[src]
    return new_labels, new_probs


def _build_label_runs(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    if arr.size == 0:
        return []
    runs: List[Tuple[int, int, int]] = []
    current = int(arr[0])
    start = 0
    for idx in range(1, int(arr.shape[0])):
        label = int(arr[idx])
        if label != current:
            runs.append((start, idx, current))
            start = idx
            current = label
    runs.append((start, int(arr.shape[0]), current))
    return runs


def _is_identifier_like_char(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch in ("_", "$", "-"))


def _dominant_label_from_prob_rows(
    char_probs: np.ndarray,
    start: int,
    end: int,
    *,
    exclude_labels: Sequence[int] = (),
) -> Tuple[Optional[int], float, float]:
    if start >= end or char_probs.size == 0:
        return None, 0.0, 0.0
    excluded = {int(label) for label in exclude_labels}
    n_classes = int(char_probs.shape[1]) if char_probs.ndim == 2 else 0
    if n_classes <= 0:
        return None, 0.0, 0.0
    valid_labels = [idx for idx in range(n_classes) if idx not in excluded]
    if not valid_labels:
        return None, 0.0, 0.0
    window = np.asarray(char_probs[max(0, int(start)) : min(int(end), int(char_probs.shape[0]))], dtype=np.float32)
    if window.size == 0:
        return None, 0.0, 0.0
    means = np.mean(window[:, valid_labels], axis=0, dtype=np.float32)
    best_offset = int(np.argmax(means))
    best_label = int(valid_labels[best_offset])
    argmax_window = np.argmax(window, axis=-1)
    argmax_fraction = float(np.mean(argmax_window == best_label, dtype=np.float32))
    return best_label, float(means[best_offset]), argmax_fraction


def _matching_wrap_delimiter(left: str, right: str) -> bool:
    return bool(left) and _BOUNDARY_WRAP_OPEN_TO_CLOSE.get(left) == right


def _wrapped_pair_bonus(left: str, right: str) -> float:
    if not _matching_wrap_delimiter(left, right):
        return 0.0
    bonus = _BOUNDARY_WRAP_PAIR_BONUS
    if left in _BOUNDARY_WRAP_QUOTE_CHARS:
        bonus += _BOUNDARY_WRAP_QUOTE_BONUS
    return bonus


def _is_codeish_wrapped_content(text: str, *, wrapper_char: str) -> bool:
    if not text:
        return False
    has_identifier = any(_is_identifier_like_char(ch) for ch in text)
    has_nonwrapper_delimiter = any(
        ch in _BOUNDARY_SNAP_DELIMITER_CHARS and ch != wrapper_char
        for ch in text
    )
    return has_identifier and has_nonwrapper_delimiter


def _boundary_local_score(text: str, boundary: int) -> float:
    if boundary < 0 or boundary > len(text):
        return float("-inf")
    left = text[boundary - 1] if boundary > 0 else ""
    right = text[boundary] if boundary < len(text) else ""
    score = 0.0
    if left in _BOUNDARY_SNAP_DELIMITER_CHARS:
        score += 1.25
    elif left in _INLINE_WHITESPACE_SET:
        score += 0.20
    if right in _BOUNDARY_SNAP_DELIMITER_CHARS:
        score += 1.25
    elif right in _INLINE_WHITESPACE_SET:
        score += 0.20
    if left in _BOUNDARY_SNAP_DELIMITER_CHARS and right in _BOUNDARY_SNAP_DELIMITER_CHARS:
        score += 0.35
    if left in _INLINE_WHITESPACE_SET and right in _INLINE_WHITESPACE_SET:
        score -= 0.25
    if left in ("<", "(", "[", "{") and _is_identifier_like_char(right):
        score += 0.35
    if right in (">", ")", "]", "}") and _is_identifier_like_char(left):
        score += 0.35
    if _is_identifier_like_char(left) and _is_identifier_like_char(right):
        score -= 1.5
    return score


def _is_jsonish_content(text: str) -> bool:
    trimmed = text.strip()
    if not trimmed:
        return False
    if len(trimmed) >= 2 and (
        (trimmed[0] == "{" and trimmed[-1] == "}")
        or (trimmed[0] == "[" and trimmed[-1] == "]")
        or (trimmed[0] == '"' and trimmed[-1] == '"')
    ):
        return True
    lowered = trimmed.lower()
    if lowered in {"true", "false", "null"}:
        return True
    if _NUMERIC_RE.fullmatch(trimmed) is not None:
        return True
    if ":" in trimmed and any(ch in trimmed for ch in ('"', "{", "[")):
        return True
    if "," in trimmed and any(ch in trimmed for ch in ('"', "{", "}", "[", "]")):
        return True
    return False


def _local_host_rule_matches_text(rule_kind: str, text: str) -> bool:
    if rule_kind == "json":
        return _is_jsonish_content(text)
    return bool(text)


def _apply_local_host_postprocess_rules(
    text: str,
    labels: np.ndarray,
    *,
    local_host_rules: Sequence[Tuple[int, int, str]],
    min_run_chars: int,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).copy()
    if arr.size == 0 or not text or not local_host_rules:
        return arr
    single_side_min_chars = max(int(min_run_chars), int(_LOCAL_HOST_SINGLE_SIDE_MIN_CHARS))
    max_passes = max(1, int(arr.shape[0]))
    for _ in range(max_passes):
        runs = _build_label_runs(arr)
        changed = False
        for run_idx, (start, end, label) in enumerate(runs):
            for inner_label, host_label, rule_kind in local_host_rules:
                if int(label) != int(inner_label):
                    continue
                left_host_len = 0
                right_host_len = 0
                if run_idx > 0 and int(runs[run_idx - 1][2]) == int(host_label):
                    left_host_len = int(runs[run_idx - 1][1] - runs[run_idx - 1][0])
                if run_idx + 1 < len(runs) and int(runs[run_idx + 1][2]) == int(host_label):
                    right_host_len = int(runs[run_idx + 1][1] - runs[run_idx + 1][0])
                if left_host_len <= 0 and right_host_len <= 0:
                    continue
                if not _local_host_rule_matches_text(str(rule_kind), text[int(start):int(end)]):
                    continue
                if left_host_len > 0 and right_host_len > 0:
                    should_relabel = True
                else:
                    should_relabel = (left_host_len + right_host_len) >= single_side_min_chars
                if not should_relabel:
                    continue
                arr[int(start):int(end)] = int(host_label)
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    return arr


def _score_boundary_candidate(
    text: str,
    labels: np.ndarray,
    char_probs: np.ndarray,
    left_run: Tuple[int, int, int],
    right_run: Tuple[int, int, int],
    boundary: int,
) -> Optional[float]:
    current = int(left_run[1])
    left_start, _, left_label = left_run
    _, right_end, right_label = right_run
    if boundary < left_start or boundary > right_end:
        return None
    left_adjacent = text[boundary - 1] if boundary > 0 else ""
    right_adjacent = text[boundary] if boundary < len(text) else ""
    if left_adjacent == "\n" or right_adjacent == "\n":
        return None
    if boundary != current and (
        left_adjacent not in _BOUNDARY_SNAP_ADJACENT_CHARS
        and right_adjacent not in _BOUNDARY_SNAP_ADJACENT_CHARS
    ):
        return None

    score = _boundary_local_score(text, boundary)
    if boundary < current:
        moved_positions = range(boundary, current)
        src_label = int(left_label)
        dest_label = int(right_label)
    else:
        moved_positions = range(current, boundary)
        src_label = int(right_label)
        dest_label = int(left_label)
    for pos in moved_positions:
        ch = text[pos]
        if ch == "\n":
            return None
        src_prob = float(char_probs[pos, src_label]) if 0 <= src_label < int(char_probs.shape[1]) else 0.0
        dest_prob = float(char_probs[pos, dest_label]) if 0 <= dest_label < int(char_probs.shape[1]) else 0.0
        if ch in _BOUNDARY_SNAP_DELIMITER_CHARS:
            if dest_prob < (src_prob - _BOUNDARY_SNAP_DELIMITER_PROB_MARGIN):
                return None
        elif ch in _INLINE_WHITESPACE_SET:
            if dest_prob < (src_prob - _BOUNDARY_SNAP_WHITESPACE_PROB_MARGIN):
                return None
        elif ch not in _BOUNDARY_SNAP_ADJACENT_CHARS and dest_prob < (src_prob - _BOUNDARY_SNAP_PROB_MARGIN):
            return None
        score += 0.5 * (dest_prob - src_prob)
        if ch in _BOUNDARY_SNAP_DELIMITER_CHARS:
            score += 0.15
        elif ch in _INLINE_WHITESPACE_SET:
            score += 0.02
    return score


def _score_wrapped_run_candidate(
    text: str,
    char_probs: np.ndarray,
    left_run: Tuple[int, int, int],
    middle_run: Tuple[int, int, int],
    right_run: Tuple[int, int, int],
    start: int,
    end: int,
) -> Optional[float]:
    left_start, _, left_label = left_run
    current_start, current_end, middle_label = middle_run
    _, right_end, right_label = right_run
    if int(left_label) != int(right_label) or int(middle_label) == int(left_label):
        return None
    if start < int(left_start) or end > int(right_end) or start >= end:
        return None
    if start <= 0 or end >= len(text):
        return None
    left_delim = text[start - 1]
    right_delim = text[end]
    score = _boundary_local_score(text, start) + _boundary_local_score(text, end)
    score += _wrapped_pair_bonus(left_delim, right_delim)

    host_label = int(left_label)
    inner_label = int(middle_label)
    changed = False
    union_start = min(int(current_start), int(start))
    union_end = max(int(current_end), int(end))
    shift_penalty = 0
    for pos in range(union_start, union_end):
        current_assign = inner_label if int(current_start) <= pos < int(current_end) else host_label
        candidate_assign = inner_label if int(start) <= pos < int(end) else host_label
        if candidate_assign == current_assign:
            continue
        ch = text[pos]
        if ch == "\n":
            return None
        changed = True
        current_prob = float(char_probs[pos, current_assign]) if 0 <= current_assign < int(char_probs.shape[1]) else 0.0
        candidate_prob = float(char_probs[pos, candidate_assign]) if 0 <= candidate_assign < int(char_probs.shape[1]) else 0.0
        if ch not in _BOUNDARY_SNAP_ADJACENT_CHARS and candidate_prob < (current_prob - _BOUNDARY_WRAP_PROB_MARGIN):
            return None
        score += 0.8 * (candidate_prob - current_prob)
        if ch in _BOUNDARY_SNAP_ADJACENT_CHARS:
            score += 0.10
        moving_out = candidate_assign == host_label and current_assign == inner_label
        moving_in = candidate_assign == inner_label and current_assign == host_label
        if ch in _BOUNDARY_SNAP_DELIMITER_CHARS:
            if moving_out:
                score += _BOUNDARY_WRAP_SHELL_DELIMITER_EJECT_BONUS
            elif moving_in:
                score -= _BOUNDARY_WRAP_SHELL_DELIMITER_SWALLOW_PENALTY
        if ch in _BOUNDARY_WRAP_QUOTE_CHARS:
            if moving_out:
                score += _BOUNDARY_WRAP_SHELL_QUOTE_EJECT_BONUS
            elif moving_in:
                score -= 0.15
        shift_penalty += 1
    if changed:
        score += 0.60 * (
            _mean_label_support(char_probs, start, end, inner_label)
            - _mean_label_support(char_probs, start, end, host_label)
        )
        score -= 0.05 * max(0, shift_penalty - 2)
    return score


def _apply_boundary_shift(
    labels: np.ndarray,
    current: int,
    boundary: int,
    left_label: int,
    right_label: int,
) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    if boundary < current:
        out[boundary:current] = int(right_label)
    elif boundary > current:
        out[current:boundary] = int(left_label)
    return out


def _snap_boundaries_to_delimiters(
    text: str,
    labels: np.ndarray,
    char_probs: np.ndarray,
    *,
    max_shift: int = 2,
    min_run_chars: int = 1,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).copy()
    if max_shift <= 0 or arr.size <= 1:
        return arr
    max_passes = max(1, int(arr.shape[0]) * 2)
    for _ in range(max_passes):
        runs = _build_label_runs(arr)
        if len(runs) <= 1:
            break
        changed = False
        for idx in range(len(runs) - 1):
            left_run = runs[idx]
            right_run = runs[idx + 1]
            current = int(left_run[1])
            window_start = int(runs[idx - 1][0]) if idx > 0 else int(left_run[0])
            window_end = int(runs[idx + 2][1]) if (idx + 2) < len(runs) else int(right_run[1])
            current_short_count, current_min_len = _count_local_submin_interior_runs(
                arr,
                min_run_chars=int(min_run_chars),
                window_start=window_start,
                window_end=window_end,
            )
            current_score = _score_boundary_candidate(text, arr, char_probs, left_run, right_run, current)
            if current_score is None:
                current_score = _boundary_local_score(text, current)
            best_boundary = current
            best_score = current_score
            best_short_count = current_short_count
            best_min_len = current_min_len
            for shift in range(-int(max_shift), int(max_shift) + 1):
                if shift == 0:
                    continue
                candidate = current + shift
                score = _score_boundary_candidate(text, arr, char_probs, left_run, right_run, candidate)
                if score is None:
                    continue
                candidate_labels = _apply_boundary_shift(arr, current, candidate, left_run[2], right_run[2])
                candidate_short_count, candidate_min_len = _count_local_submin_interior_runs(
                    candidate_labels,
                    min_run_chars=int(min_run_chars),
                    window_start=window_start,
                    window_end=window_end,
                )
                better_structure = (
                    candidate_short_count < best_short_count
                    or (
                        candidate_short_count == best_short_count
                        and candidate_min_len > best_min_len
                    )
                )
                same_structure = (
                    candidate_short_count == best_short_count
                    and candidate_min_len == best_min_len
                )
                if better_structure or (same_structure and score > (best_score + 1e-6)):
                    best_boundary = candidate
                    best_score = score
                    best_short_count = candidate_short_count
                    best_min_len = candidate_min_len
            if best_boundary != current and best_score >= (current_score + _BOUNDARY_SNAP_MIN_IMPROVEMENT):
                arr = _apply_boundary_shift(arr, current, best_boundary, left_run[2], right_run[2])
                changed = True
                break
        if not changed:
            break
    return arr


def _apply_wrapped_run_shift(
    labels: np.ndarray,
    current_start: int,
    current_end: int,
    start: int,
    end: int,
    *,
    host_label: int,
    inner_label: int,
) -> np.ndarray:
    out = np.asarray(labels, dtype=np.int32).copy()
    union_start = min(int(current_start), int(start))
    union_end = max(int(current_end), int(end))
    out[union_start:union_end] = int(host_label)
    out[int(start):int(end)] = int(inner_label)
    return out


def _fill_markdown_structure_regions(
    text: str,
    labels: np.ndarray,
    char_probs: np.ndarray,
    *,
    markdown_label: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(labels, dtype=np.int32).copy()
    locked = _new_lock_mask(arr.shape[0])
    if markdown_label is None or arr.size == 0 or not text:
        return arr, locked
    n = len(text)
    line_start = 0
    while line_start < n:
        line_end = text.find("\n", line_start)
        if line_end == -1:
            line_end = n
        pos = line_start
        while pos < line_end:
            if text[pos] != "`":
                pos += 1
                continue
            run_end = pos
            while run_end < line_end and text[run_end] == "`":
                run_end += 1
            run_len = run_end - pos
            if run_len < 3:
                pos = run_end
                continue
            next_start, next_end = _find_next_backtick_run(text, run_end, line_end, min_len=run_len)
            if next_start != -1 and next_start > run_end:
                arr[pos:run_end] = int(markdown_label)
                body_label = _infer_uniform_body_label(
                    arr,
                    char_probs,
                    run_end,
                    next_start,
                    markdown_label=int(markdown_label),
                )
                arr[run_end:next_start] = int(body_label)
                arr[next_start:next_end] = int(markdown_label)
                _mark_locked_range(locked, pos, next_end)
                pos = next_end
                continue
            token_end = run_end
            while token_end < line_end and text[token_end] not in (" ", "\t"):
                token_end += 1
            arr[pos:token_end] = int(markdown_label)
            _mark_locked_range(locked, pos, token_end)
            pos = token_end
        line_start = line_end + 1
    return arr, locked


def _refine_wrapped_runs(
    text: str,
    labels: np.ndarray,
    char_probs: np.ndarray,
    *,
    max_shift: int = 2,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).copy()
    if max_shift <= 0 or arr.size <= 2:
        return arr
    max_passes = max(1, int(arr.shape[0]))
    for _ in range(max_passes):
        runs = _build_label_runs(arr)
        if len(runs) <= 2:
            break
        changed = False
        for idx in range(1, len(runs) - 1):
            left_run = runs[idx - 1]
            middle_run = runs[idx]
            right_run = runs[idx + 1]
            if int(left_run[2]) != int(right_run[2]) or int(middle_run[2]) == int(left_run[2]):
                continue
            current_start = int(middle_run[0])
            current_end = int(middle_run[1])
            current_score = _score_wrapped_run_candidate(
                text,
                char_probs,
                left_run,
                middle_run,
                right_run,
                current_start,
                current_end,
            )
            if current_score is None:
                current_score = _boundary_local_score(text, current_start) + _boundary_local_score(text, current_end)
            best_start = current_start
            best_end = current_end
            best_score = current_score
            for left_shift in range(-int(max_shift), int(max_shift) + 1):
                cand_start = current_start + left_shift
                if cand_start < int(left_run[0]) or cand_start >= current_end:
                    continue
                for right_shift in range(-int(max_shift), int(max_shift) + 1):
                    cand_end = current_end + right_shift
                    if cand_end <= cand_start or cand_end > int(right_run[1]):
                        continue
                    if cand_start == current_start and cand_end == current_end:
                        continue
                    if not _matching_wrap_delimiter(
                        text[cand_start - 1] if cand_start > 0 else "",
                        text[cand_end] if cand_end < len(text) else "",
                    ):
                        continue
                    score = _score_wrapped_run_candidate(
                        text,
                        char_probs,
                        left_run,
                        middle_run,
                        right_run,
                        cand_start,
                        cand_end,
                    )
                    if score is None:
                        continue
                    if score > (best_score + 1e-6):
                        best_start = cand_start
                        best_end = cand_end
                        best_score = score
            if (
                (best_start != current_start or best_end != current_end)
                and best_score >= (current_score + _BOUNDARY_WRAP_MIN_IMPROVEMENT)
            ):
                arr = _apply_wrapped_run_shift(
                    arr,
                    current_start,
                    current_end,
                    best_start,
                    best_end,
                    host_label=int(left_run[2]),
                    inner_label=int(middle_run[2]),
                )
                changed = True
                break
        if not changed:
            break
    return arr


def _mean_label_support(char_probs: np.ndarray, start: int, end: int, label: int) -> float:
    if start >= end or label < 0 or label >= int(char_probs.shape[1]):
        return 0.0
    window = np.asarray(char_probs[start:end, label], dtype=np.float32)
    if window.size == 0:
        return 0.0
    return float(np.mean(window, dtype=np.float32))


def _count_local_submin_interior_runs(
    labels: np.ndarray,
    *,
    min_run_chars: int,
    window_start: int,
    window_end: int,
) -> Tuple[int, int]:
    runs = _build_label_runs(labels)
    count = 0
    min_len: Optional[int] = None
    for idx, (start, end, _label) in enumerate(runs):
        if end <= int(window_start) or start >= int(window_end):
            continue
        run_len = int(end - start)
        min_len = run_len if min_len is None else min(min_len, run_len)
        if 0 < idx < (len(runs) - 1) and run_len < int(min_run_chars):
            count += 1
    return count, (int(min_len) if min_len is not None else 0)


def _normalize_short_runs(
    labels: np.ndarray,
    char_probs: np.ndarray,
    *,
    min_run_chars: int,
    locked_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).copy()
    if min_run_chars <= 1 or arr.size <= 2:
        return arr
    if locked_mask is not None and int(locked_mask.shape[0]) == int(arr.shape[0]):
        locks = np.asarray(locked_mask, dtype=bool)
    else:
        locks = _new_lock_mask(arr.shape[0])
    max_passes = max(1, int(arr.shape[0]))
    for _ in range(max_passes):
        runs = _build_label_runs(arr)
        changed = False
        for idx in range(1, len(runs) - 1):
            start, end, _label = runs[idx]
            if (end - start) >= int(min_run_chars):
                continue
            if bool(np.any(locks[start:end])):
                continue
            left_run = runs[idx - 1]
            right_run = runs[idx + 1]
            left_label = int(left_run[2])
            right_label = int(right_run[2])
            left_len = int(left_run[1] - left_run[0])
            right_len = int(right_run[1] - right_run[0])
            candidates: List[Tuple[Tuple[float, ...], int]] = []
            seen_targets: set[int] = set()
            for direction, target, neighbor_len in (
                ("left", left_label, left_len),
                ("right", right_label, right_len),
            ):
                if int(target) in seen_targets:
                    continue
                seen_targets.add(int(target))
                candidate = arr.copy()
                candidate[start:end] = int(target)
                submin_count, min_run_len = _count_local_submin_interior_runs(
                    candidate,
                    min_run_chars=min_run_chars,
                    window_start=int(left_run[0]),
                    window_end=int(right_run[1]),
                )
                support = _mean_label_support(char_probs, start, end, int(target))
                sandwich = 1.0 if left_label == right_label == int(target) else 0.0
                direction_tiebreak = 1.0 if direction == "left" else 0.0
                key = (
                    -float(submin_count),
                    float(min_run_len),
                    sandwich,
                    float(neighbor_len),
                    float(support),
                    direction_tiebreak,
                )
                candidates.append((key, int(target)))
            if not candidates:
                continue
            target = max(candidates, key=lambda item: item[0])[1]
            arr[start:end] = int(target)
            changed = True
            break
        if not changed:
            break
    return arr


def _postprocess_char_labels(
    text: str,
    labels: np.ndarray,
    char_probs: np.ndarray,
    *,
    min_run_chars: int,
    boundary_snap_max_shift: int = 2,
    local_host_rules: Sequence[Tuple[int, int, str]] = (),
    markdown_label: Optional[int] = None,
    html_label: Optional[int] = None,
) -> np.ndarray:
    markdown_filled, markdown_locked = _fill_markdown_structure_regions(
        text,
        labels,
        char_probs,
        markdown_label=markdown_label,
    )
    snapped = _snap_boundaries_to_delimiters(
        text,
        markdown_filled,
        char_probs,
        max_shift=boundary_snap_max_shift,
        min_run_chars=min_run_chars,
    )
    wrapped = _refine_wrapped_runs(
        text,
        snapped,
        char_probs,
        max_shift=boundary_snap_max_shift,
    )
    local_host_filled = _apply_local_host_postprocess_rules(
        text,
        wrapped,
        local_host_rules=local_host_rules,
        min_run_chars=min_run_chars,
    )
    locked_mask = _new_lock_mask(labels.shape[0])
    _merge_lock_masks(locked_mask, markdown_locked)
    normalized = _normalize_short_runs(
        local_host_filled,
        char_probs,
        min_run_chars=min_run_chars,
        locked_mask=locked_mask,
    )
    snapped_relit = _snap_boundaries_to_delimiters(
        text,
        normalized,
        char_probs,
        max_shift=boundary_snap_max_shift,
        min_run_chars=min_run_chars,
    )
    wrapped_relit = _refine_wrapped_runs(
        text,
        snapped_relit,
        char_probs,
        max_shift=boundary_snap_max_shift,
    )
    markdown_relit, _markdown_relock = _fill_markdown_structure_regions(
        text,
        wrapped_relit,
        char_probs,
        markdown_label=markdown_label,
    )
    return _apply_local_host_postprocess_rules(
        text,
        markdown_relit,
        local_host_rules=local_host_rules,
        min_run_chars=min_run_chars,
    )


class NumpyMambaSegmentor:
    def __init__(self, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
        self.manifest = dict(manifest)
        self.model_id = str(self.manifest.get("model_id", "sfullfiles4"))
        self.label_order = [str(label) for label in self.manifest.get("label_order", [])]
        self.display_labels = [str(label) for label in self.manifest.get("display_labels", self.label_order)]
        self.num_classes = int(self.manifest.get("num_classes", len(self.label_order)))
        self.window_bytes = int(self.manifest.get("window_bytes", 1536))
        self.window_stride_bytes = int(self.manifest.get("window_stride_bytes", self.window_bytes // 2))
        self.other_threshold = _resolve_threshold(self.manifest.get("other_threshold", 0.3), 0.3)
        self.max_input_bytes = max(0, int(self.manifest.get("max_input_bytes", 6144)))
        self.postprocess_min_run_chars = max(1, int(self.manifest.get("postprocess_min_run_chars", 5)))
        self.postprocess_boundary_snap_max_shift = max(
            0,
            int(self.manifest.get("postprocess_boundary_snap_max_shift", 2)),
        )
        self.other_label_id = int(self.num_classes)
        self.local_host_postprocess_rules = _resolve_local_host_postprocess_rules(self.label_order)
        self.markdown_label_id = _resolve_label_id(self.label_order, "markdown")
        self.html_label_id = _resolve_label_id(self.label_order, "html")

        self.embed = np.asarray(arrays["embed/embedding"], dtype=np.float32)
        self.final_ln_scale = np.asarray(arrays["final/ln_scale"], dtype=np.float32)
        self.final_ln_bias = np.asarray(arrays["final/ln_bias"], dtype=np.float32)
        self.final_dense_kernel = np.asarray(arrays["final/dense_kernel"], dtype=np.float32)
        self.final_dense_bias = np.asarray(arrays["final/dense_bias"], dtype=np.float32)

        self.blocks: List[Dict[str, np.ndarray]] = []
        n_layers = int(self.manifest.get("model", {}).get("n_layers", 0))
        for idx in range(n_layers):
            prefix = f"blocks/{idx}/"
            block = {
                "ln_scale": np.asarray(arrays[prefix + "ln_scale"], dtype=np.float32),
                "ln_bias": np.asarray(arrays[prefix + "ln_bias"], dtype=np.float32),
                "in_proj_kernel": np.asarray(arrays[prefix + "in_proj_kernel"], dtype=np.float32),
                "in_proj_bias": np.asarray(arrays[prefix + "in_proj_bias"], dtype=np.float32),
                "conv_kernel": np.asarray(arrays[prefix + "conv_kernel"], dtype=np.float32),
                "conv_bias": np.asarray(arrays[prefix + "conv_bias"], dtype=np.float32),
                "x_proj_kernel": np.asarray(arrays[prefix + "x_proj_kernel"], dtype=np.float32),
                "x_proj_bias": np.asarray(arrays[prefix + "x_proj_bias"], dtype=np.float32),
                "dt_proj_kernel": np.asarray(arrays[prefix + "dt_proj_kernel"], dtype=np.float32),
                "dt_proj_bias": np.asarray(arrays[prefix + "dt_proj_bias"], dtype=np.float32),
                "out_proj_kernel": np.asarray(arrays[prefix + "out_proj_kernel"], dtype=np.float32),
                "out_proj_bias": np.asarray(arrays[prefix + "out_proj_bias"], dtype=np.float32),
                "a": -np.exp(np.asarray(arrays[prefix + "a_log"], dtype=np.float32)),
                "d": np.asarray(arrays[prefix + "d"], dtype=np.float32),
            }
            self.blocks.append(block)

    @classmethod
    def from_files(cls, manifest_path: str | Path, weights_path: str | Path) -> "NumpyMambaSegmentor":
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        with np.load(Path(weights_path), allow_pickle=False) as data:
            arrays = {str(key): np.asarray(data[key], dtype=np.float32) for key in data.files}
        return cls(manifest, arrays)

    @classmethod
    def from_bytes(cls, manifest_bytes: bytes, weights_bytes: bytes) -> "NumpyMambaSegmentor":
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        with np.load(io.BytesIO(weights_bytes), allow_pickle=False) as data:
            arrays = {str(key): np.asarray(data[key], dtype=np.float32) for key in data.files}
        return cls(manifest, arrays)

    def _block_forward(self, x: np.ndarray, block: Mapping[str, np.ndarray]) -> np.ndarray:
        h = _layer_norm(x, block["ln_scale"], block["ln_bias"])
        xz = h @ block["in_proj_kernel"] + block["in_proj_bias"][None, :]
        split = xz.shape[1] // 2
        u = xz[:, :split]
        gate = xz[:, split:]

        u = _depthwise_conv_same(u, block["conv_kernel"], block["conv_bias"])
        u = _silu(u)

        x_dbl = u @ block["x_proj_kernel"] + block["x_proj_bias"][None, :]
        dt_rank = int(block["dt_proj_kernel"].shape[0])
        d_state = int(block["a"].shape[1])
        dt_raw = x_dbl[:, :dt_rank]
        b_in = x_dbl[:, dt_rank : dt_rank + d_state]
        c_in = x_dbl[:, dt_rank + d_state : dt_rank + (2 * d_state)]
        dt = _softplus(dt_raw @ block["dt_proj_kernel"] + block["dt_proj_bias"][None, :]) + 1e-4

        y = _selective_scan(u, dt, b_in, c_in, block["a"], block["d"])
        if bool(self.manifest.get("model", {}).get("bidirectional", True)):
            y_rev = _selective_scan(
                u[::-1],
                dt[::-1],
                b_in[::-1],
                c_in[::-1],
                block["a"],
                block["d"],
            )[::-1]
            y = y + y_rev

        y = y * _silu(gate)
        y = y @ block["out_proj_kernel"] + block["out_proj_bias"][None, :]
        return x + y

    def predict_window_probs(self, tokens: np.ndarray) -> np.ndarray:
        tok = _sanitize_model_tokens(np.asarray(tokens, dtype=np.int32).reshape(-1))
        length = int(tok.shape[0])
        if length <= 0:
            return np.zeros((0, self.num_classes), dtype=np.float32)
        h = np.asarray(self.embed[tok], dtype=np.float32)
        for block in self.blocks:
            h = self._block_forward(h, block)
        h = _layer_norm(h, self.final_ln_scale, self.final_ln_bias)
        logits = h @ self.final_dense_kernel + self.final_dense_bias[None, :]
        return _softmax(logits).astype(np.float32)

    def segment_byte_probs(self, byte_arr: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        arr = _sanitize_model_bytes(np.asarray(byte_arr, dtype=np.uint8).reshape(-1))
        length = int(arr.shape[0])
        if length == 0:
            return np.zeros((0, self.num_classes), dtype=np.float32), []
        probs = self.predict_window_probs(arr.astype(np.int32, copy=False))
        return probs, [(0, length)]

    def _byte_probs_to_char_payload(
        self,
        text: str,
        byte_probs: np.ndarray,
        *,
        top_k: int,
        other_threshold: float,
    ) -> Tuple[List[Dict[str, int]], List[List[Dict[str, float]]], List[float]]:
        n_chars = len(text)
        if n_chars == 0:
            return [], [], []

        labels = np.zeros((n_chars,), dtype=np.int32)
        char_probs = np.zeros((n_chars, self.num_classes), dtype=np.float32)
        byte_pos = 0
        for idx, ch in enumerate(text):
            byte_len = len(ch.encode("utf-8", "ignore"))
            if byte_len <= 0:
                continue
            avg_probs = np.mean(byte_probs[byte_pos : byte_pos + byte_len], axis=0, dtype=np.float32)
            labels[idx] = int(np.argmax(avg_probs))
            char_probs[idx] = avg_probs
            byte_pos += byte_len

        labels, char_probs = _relabel_whitespace_from_neighbors(text, labels, char_probs)
        labels = _threshold_predictions(char_probs, other_threshold, self.other_label_id)
        labels = _postprocess_char_labels(
            text,
            labels,
            char_probs,
            min_run_chars=self.postprocess_min_run_chars,
            boundary_snap_max_shift=self.postprocess_boundary_snap_max_shift,
            local_host_rules=self.local_host_postprocess_rules,
            markdown_label=self.markdown_label_id,
            html_label=self.html_label_id,
        )

        segments: List[Dict[str, int]] = []
        start = 0
        current = int(labels[0])
        for idx in range(1, n_chars):
            nxt = int(labels[idx])
            if nxt != current:
                segments.append({"start": int(start), "end": int(idx), "label_id": int(current)})
                start = idx
                current = nxt
        segments.append({"start": int(start), "end": int(n_chars), "label_id": int(current)})

        top_payload: List[List[Dict[str, float]]] = []
        confidences = np.max(char_probs, axis=-1).astype(np.float32)
        top_k_eff = max(1, min(int(top_k), self.num_classes))
        for idx in range(n_chars):
            order = np.argsort(char_probs[idx])[::-1][:top_k_eff]
            top_payload.append(
                [
                    {"id": int(label_id), "prob": float(char_probs[idx, label_id])}
                    for label_id in order
                ]
            )
        return segments, top_payload, [float(value) for value in confidences]

    def segment_text_payload(
        self,
        text: str,
        *,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        effective_threshold = _resolve_threshold(threshold, self.other_threshold)
        normalized = _normalize_input_text(text)
        raw_bytes = np.frombuffer(normalized.encode("utf-8", "ignore"), dtype=np.uint8)
        sanitized_bytes = _sanitize_model_bytes(raw_bytes)
        input_bytes = int(sanitized_bytes.shape[0])

        if self.max_input_bytes > 0 and input_bytes > self.max_input_bytes:
            raise ValueError(
                f"Input exceeds the public demo limit of {self.max_input_bytes} bytes after sanitization."
            )

        if input_bytes == 0:
            return {
                "text": normalized,
                "segments": [],
                "char_top_probs": [],
                "char_confidences": [],
                "stats": [],
                "input_bytes": 0,
                "window_count": 0,
                "other_threshold": effective_threshold,
                "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
            }

        byte_probs, spans = self.segment_byte_probs(sanitized_bytes)
        segments, char_top_probs, char_confidences = self._byte_probs_to_char_payload(
            normalized,
            byte_probs,
            top_k=top_k,
            other_threshold=effective_threshold,
        )

        counts: Dict[int, int] = {}
        for segment in segments:
            label_id = int(segment["label_id"])
            counts[label_id] = counts.get(label_id, 0) + int(segment["end"] - segment["start"])
        total_chars = max(1, len(normalized))
        stats = [
            {
                "id": int(label_id),
                "count": int(count),
                "pct": float(count / total_chars * 100.0),
            }
            for label_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

        return {
            "text": normalized,
            "segments": segments,
            "char_top_probs": char_top_probs,
            "char_confidences": char_confidences,
            "stats": stats,
            "input_bytes": input_bytes,
            "window_count": len(spans),
            "other_threshold": effective_threshold,
            "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
        }


def load_model_json(manifest_path: str, weights_path: str) -> str:
    global _MODEL
    _MODEL = NumpyMambaSegmentor.from_files(manifest_path, weights_path)
    payload = {
        "loaded": True,
        "model_id": _MODEL.model_id,
        "num_classes": _MODEL.num_classes,
        "window_bytes": _MODEL.window_bytes,
        "window_stride_bytes": _MODEL.window_stride_bytes,
        "other_threshold": _MODEL.other_threshold,
        "max_input_bytes": _MODEL.max_input_bytes,
        "postprocess_min_run_chars": _MODEL.postprocess_min_run_chars,
        "postprocess_boundary_snap_max_shift": _MODEL.postprocess_boundary_snap_max_shift,
        "label_order": list(_MODEL.label_order),
        "display_labels": list(_MODEL.display_labels),
        "runtime": "pyodide-wasm",
    }
    return json.dumps(payload, separators=(",", ":"))


def segment_text_json(text: str, top_k: int = 5, threshold: Optional[float] = None) -> str:
    if _MODEL is None:
        raise RuntimeError("Model has not been loaded yet.")
    payload = _MODEL.segment_text_payload(
        text,
        top_k=max(1, int(top_k)),
        threshold=threshold,
    )
    return json.dumps(payload, separators=(",", ":"))
