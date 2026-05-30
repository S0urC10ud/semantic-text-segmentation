#!/usr/bin/env python3
"""Comprehensive evaluation harness for the labeling model.

Expects evaluation datasets produced by ``obtain_eval_dataset.py``
from the monitor set (Gemini segmentations + Arrow monitor) and runs a set
of benchmarks. By default each evaluation run writes a dedicated artifact
directory under ``evaluation/reports/`` containing the Markdown report,
comparison JSON, and confusion-matrix plots.

Example:
python evaluation.py \
  --checkpoint checkpoints/seg-unet1d.msgpack \
  --data-root evaluation/data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse
import re

import numpy as np
import psutil

DEFAULT_CHANNELS: Tuple[int, ...] = (96, 128, 192, 256)
DEFAULT_CHUNK_SIZE: int  # populated after importing train.config
PREDICTION_LABEL_ALIASES: Dict[str, str] = {
    "c": "c_family",
    "cpp": "c_family",
    "javascript": "javascript_typescript",
    "typescript": "javascript_typescript",
    "shell_batchfile": "shell",
    "batchfile": "shell",
}

NEEDLE_COVERAGE_THRESHOLD = 0.5
NEEDLE_BOUNDARY_WINDOW_TOKENS = 4
MARKDOWN_IOU_THRESHOLD = 0.5
PAYLOAD_IOU_THRESHOLD = 0.5
PAYLOAD_SOFT_PROB_THRESHOLD = 0.05
PAYLOAD_SOFT_COVERAGE_THRESHOLD = 0.5
_NEEDLE_BUCKET_RE = re.compile(r"^needle_(\d+)_((\d+)|plus)$")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

MARKDOWN_FENCED_LABEL = r"\`\`\` fenced \`\`\`"
MARKDOWN_INLINE_LABEL = r"inline code (\`...\`)"
POSTPROCESS_PROFILE_OFF = "off"
POSTPROCESS_PROFILE_THESIS = "thesis"
POSTPROCESS_PROFILE_CHOICES = (POSTPROCESS_PROFILE_OFF, POSTPROCESS_PROFILE_THESIS)

import datasets as hfds  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.serialization as serialization  # noqa: E402
from flax.errors import ScopeParamShapeError  # noqa: E402

# Compat shim: orbax-checkpoint >=0.11 references
# `jax.experimental.layout.DeviceLocalLayout`, which was renamed to `Layout`
# in jax >=0.8. Alias before orbax import so the loader works on jax 0.8.1.
try:
    import jax.experimental.layout as _jax_layout  # noqa: E402
    if not hasattr(_jax_layout, "DeviceLocalLayout") and hasattr(_jax_layout, "Layout"):
        _jax_layout.DeviceLocalLayout = _jax_layout.Layout
except Exception:  # pragma: no cover
    pass

try:
    import orbax.checkpoint as ocp  # noqa: E402
except Exception:  # pragma: no cover - environment-dependent optional import
    ocp = None  # type: ignore[assignment]

from inference.backend import (  # noqa: E402
    FastInferenceEngine,
    FastInferenceFailure,
    build_window_spans,
    format_auto_fallback_message,
)
from inference.mamba_cuda import has_cuda_mamba_kernel  # noqa: E402
from magika_label_map import THESIS_ENCODING_LABELS  # noqa: E402
from magika_windowed import DEFAULT_MAGIKA_WINDOW_SIZE, SlidingWindowMagikaSegmenter  # noqa: E402
import utils.config as cfg  # noqa: E402
from utils.model import (  # noqa: E402
    Mamba1D,
    UNet1D,
    checkpoint_params_subtree,
    merge_compatible_state,
)
from utils.metrics_helper import (  # noqa: E402
    _valid_metric_mask,
    accumulate_confusion,
    compute_metrics_from_confusion,
)
from utils.monitor_eval import load_monitor_memmaps  # noqa: E402
from utils.token_utils import sanitize_bytes, sanitize_tokens  # noqa: E402

DEFAULT_CHUNK_SIZE = cfg.MODEL_WINDOW_BYTES


# ---------------------------------------------------------------------------
# Checkpoint loading helpers (adapted from segment_viewer.app without CLI)
# ---------------------------------------------------------------------------

NON_ASCII_PLACEHOLDER = "\u00A4"
_VISIBLE_ASCII_MIN = 0x20
_VISIBLE_ASCII_MAX = 0x7E
_ALLOWED_TEXT_CONTROLS = {"\n", "\t"}
_VISUAL_WHITESPACE_CHARS: Tuple[str, ...] = (" ", "\t", "\n")
_VISUAL_WHITESPACE_SET = frozenset(_VISUAL_WHITESPACE_CHARS)
TEXT_LIKE_POSITIVE_LABELS: Tuple[str, ...] = (
    "text",
    "markdown",
    "restructuredtext",
    "tex",
)
RELEVANT_CONTENT_POSTPROCESS_MIN_TOKENS = 10
TEXT_HOST_POSTPROCESS_LABELS: Tuple[str, ...] = (
    "markdown",
    "restructuredtext",
    "tex",
)
LOCAL_HOST_POSTPROCESS_RULES: Tuple[Tuple[str, str], ...] = (
    ("json", "javascript_typescript"),
    ("xml", "svg"),
)
REPORT_ONLY_TRUTH_LABEL_EXCLUSIONS: Dict[str, Tuple[str, ...]] = {
    "markdown_mix": ("markdown", "text"),
}
_TEXT_LIKE_ID2LABEL: Dict[int, str] = {
    0: "not_text_like",
    1: "text_like",
}
MAGIKA_REDUCED_SUPPORT_NOTE = (
    "Encoding-related injections and full-file evaluations were excluded before scoring. "
    "This Magika baseline is therefore evaluated on an easier reduced-support subset and "
    "should only be compared on the remaining counted samples/files."
)


def normalize_eval_text(text: Optional[str]) -> str:
    if not text:
        return ""
    out_chars: List[str] = []
    for ch in text:
        if ch == "\r":
            ch = "\n"
        code = ord(ch)
        if ch in _ALLOWED_TEXT_CONTROLS or _VISIBLE_ASCII_MIN <= code <= _VISIBLE_ASCII_MAX or ch == NON_ASCII_PLACEHOLDER:
            out_chars.append(ch)
        else:
            out_chars.append(NON_ASCII_PLACEHOLDER)
    return "".join(out_chars)

def _looks_like_orbax_step_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    names = {x.name for x in p.iterdir()}
    return (
        "manifest.ocdbt" in names
        or "_CHECKPOINT_METADATA" in names
        or "ocdbt.process_0" in names
        or "_METADATA" in names
    )


def _find_latest_orbax_step_dir(root: Path) -> Optional[Path]:
    root = Path(root)
    if _looks_like_orbax_step_dir(root):
        return root
    if not root.exists() or not root.is_dir():
        return None
    step_dirs = []
    for d in root.iterdir():
        if d.is_dir() and _looks_like_orbax_step_dir(d):
            name = d.name
            step = -1
            if "-" in name:
                try:
                    step = int(name.rsplit("-", 1)[-1])
                except ValueError:
                    step = -1
            step_dirs.append((step, name, d))
    if not step_dirs:
        return None
    step_dirs.sort(key=lambda t: (t[0], '.msgpack' in t[1]))
    return step_dirs[-1][2]


def _extract_params_tree(obj):
    if hasattr(obj, "params"):
        try:
            return getattr(obj, "params")
        except Exception:
            pass
    try:
        from flax.core.frozen_dict import FrozenDict

        if isinstance(obj, FrozenDict):
            return _extract_params_tree(obj.unfreeze())
    except Exception:
        pass
    from collections.abc import Mapping

    if isinstance(obj, Mapping):
        if "params" in obj:
            return obj["params"]
        for key in ("target", "state", "train_state", "flax_state"):
            if key in obj:
                try:
                    return _extract_params_tree(obj[key])
                except Exception:
                    continue
        for v in obj.values():
            try:
                return _extract_params_tree(v)
            except Exception:
                continue
    if isinstance(obj, (list, tuple)):
        for v in obj:
            try:
                return _extract_params_tree(v)
            except Exception:
                continue
    if hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            try:
                return _extract_params_tree(v)
            except Exception:
                continue
    raise KeyError("Could not find 'params' subtree in restored checkpoint object.")


def _load_params_from_any(ckpt_path: str, params_template):
    p = Path(ckpt_path).resolve()

    if p.is_file():
        with open(p, "rb") as f:
            raw = f.read()
        try:
            return serialization.from_bytes(params_template, raw)
        except Exception:
            try:
                data = serialization.from_bytes({"params": params_template}, raw)
                return data["params"]
            except Exception:
                restored = serialization.msgpack_restore(raw)
                params, _ = merge_compatible_state(
                    params_template,
                    checkpoint_params_subtree(restored),
                )
                return params

    step_dir = _find_latest_orbax_step_dir(p)
    if step_dir is None:
        raise FileNotFoundError(f"Checkpoint path not found/unsupported: {ckpt_path}")
    if ocp is None:
        raise RuntimeError(
            "Orbax checkpoint support is unavailable in this environment; "
            f"cannot restore Orbax checkpoint at '{step_dir}'."
        )

    step_dir_str = step_dir.resolve().as_posix()
    standard = ocp.StandardCheckpointer()
    try:
        restored = standard.restore(step_dir_str)
        params, _ = merge_compatible_state(
            params_template,
            checkpoint_params_subtree(_extract_params_tree(restored)),
        )
        return params
    except Exception:
        pass

    try:
        template = {"params": params_template}
        restored = standard.restore(step_dir_str, target=template, strict=False)
        params, _ = merge_compatible_state(
            params_template,
            checkpoint_params_subtree(_extract_params_tree(restored)),
        )
        return params
    except Exception:
        pass

    try:
        from flax.training import train_state as ts
        import optax

        tx = optax.identity()
        dummy = ts.TrainState.create(apply_fn=lambda *a, **k: None, params=params_template, tx=tx)
        restored = standard.restore(step_dir_str, target=dummy, strict=False)
        params, _ = merge_compatible_state(
            params_template,
            checkpoint_params_subtree(restored),
        )
        return params
    except Exception:
        pass

    # Final fallback: untargeted PyTree restore that materialises every leaf as
    # a numpy ndarray. Works when the checkpoint's recorded sharding (e.g.
    # cuda:0 from the training environment) is not available on the current
    # host (Apple Metal / CPU). Extracts the ``params`` subtree afterwards.
    try:
        import numpy as _np
        handler = ocp.PyTreeCheckpointHandler()
        ptree_checkpointer = ocp.Checkpointer(handler)
        metadata = ptree_checkpointer.metadata(step_dir_str)

        def _to_args(node):
            if hasattr(node, "items"):
                return {k: _to_args(v) for k, v in node.items()}
            if isinstance(node, (list, tuple)):
                try:
                    return [_to_args(x) for x in node]
                except Exception:
                    return ocp.ArrayRestoreArgs(restore_type=_np.ndarray)
            return ocp.ArrayRestoreArgs(restore_type=_np.ndarray)

        restore_args = _to_args(metadata)
        restored = ptree_checkpointer.restore(step_dir_str, restore_args=restore_args)
        params, _ = merge_compatible_state(
            params_template,
            checkpoint_params_subtree(_extract_params_tree(restored)),
        )
        return params
    except Exception as exc:
        raise RuntimeError(
            f"Failed to restore checkpoint at '{step_dir_str}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------


def _resolve_backend(device_name: Optional[str]) -> Optional[str]:
    if not device_name or device_name.lower() in {"auto", "default"}:
        return None
    lower = device_name.lower()
    if lower in {"cpu", "gpu", "tpu"}:
        return lower
    if lower == "cuda":
        return "gpu"
    raise ValueError(f"Unknown device name '{device_name}'. Use cpu/gpu/cuda/auto.")


def _available_backends() -> Set[str]:
    """Return the set of platforms exposed by the current JAX build."""
    platforms: Set[str] = set()
    try:
        for dev in jax.devices():
            platforms.add(dev.platform)
    except Exception:
        pass
    for candidate in ("cpu", "gpu", "tpu"):
        try:
            devices = jax.devices(candidate)
        except Exception:
            continue
        if devices:
            platforms.add(devices[0].platform)
    return platforms


def _assert_unet_chunk_size(arch: str, chunk: int) -> None:
    arch_name = str(arch).lower().strip()
    if arch_name not in {"unet1d", "magika"}:
        return
    required = int(cfg.MODEL_WINDOW_BYTES if arch_name == "unet1d" else DEFAULT_MAGIKA_WINDOW_SIZE)
    actual = int(chunk)
    if actual != required:
        if arch_name == "unet1d":
            raise ValueError(
                "U-Net evaluation must use the fixed 1536-byte chunk size. "
                f"Received chunk={actual}. Re-run with --chunk {required}."
            )
        raise ValueError(
            "Magika sliding-window evaluation must use the fixed 1536-byte chunk size. "
            f"Received chunk={actual}. Re-run with --chunk {required}."
        )


def _window_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((length,), dtype=np.float32)
    positions = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
    sigma = 0.5
    weights = np.exp(-0.5 * (positions / sigma) ** 2)
    return weights.astype(np.float32)


def _relabel_whitespace_from_neighbors(
    text: str,
    labels: List[int],
    char_probs: List[np.ndarray],
) -> Tuple[List[int], List[np.ndarray]]:
    """Relabel whitespace chars by copying labels/probs from nearest non-whitespace neighbors."""
    n = len(text)
    if n == 0 or not labels or len(labels) != n:
        return labels, char_probs

    if not any(ch in _VISUAL_WHITESPACE_SET for ch in text):
        return labels, char_probs

    new_labels = list(labels)
    new_probs = list(char_probs)

    # Build per-line segments to prefer neighbors from the same line.
    line_starts: List[int] = [0]
    for idx, ch in enumerate(text):
        if ch == "\n" and idx + 1 < n:
            line_starts.append(idx + 1)
    line_starts = sorted(set(line_starts))
    line_segments: List[Tuple[int, int]] = []
    for i, start in enumerate(line_starts):
        end = line_starts[i + 1] if i + 1 < len(line_starts) else n
        if start < end:
            line_segments.append((start, end))

    left_same_line = [-1] * n
    right_same_line = [-1] * n
    for start, end in line_segments:
        last_non_ws = -1
        for i in range(start, end):
            if text[i] not in _VISUAL_WHITESPACE_SET:
                last_non_ws = i
            left_same_line[i] = last_non_ws
        last_non_ws = -1
        for i in range(end - 1, start - 1, -1):
            if text[i] not in _VISUAL_WHITESPACE_SET:
                last_non_ws = i
            right_same_line[i] = last_non_ws

    left_any = [-1] * n
    right_any = [-1] * n
    last_non_ws = -1
    for i in range(n):
        if text[i] not in _VISUAL_WHITESPACE_SET:
            last_non_ws = i
        left_any[i] = last_non_ws
    last_non_ws = -1
    for i in range(n - 1, -1, -1):
        if text[i] not in _VISUAL_WHITESPACE_SET:
            last_non_ws = i
        right_any[i] = last_non_ws

    for i, ch in enumerate(text):
        if ch not in _VISUAL_WHITESPACE_SET:
            continue
        src = -1
        ls = left_same_line[i]
        rs = right_same_line[i]
        if ls != -1 or rs != -1:
            if ls == -1:
                src = rs
            elif rs == -1:
                src = ls
            else:
                dist_l = i - ls
                dist_r = rs - i
                src = ls if dist_l <= dist_r else rs
        else:
            la = left_any[i]
            ra = right_any[i]
            if la != -1 or ra != -1:
                if la == -1:
                    src = ra
                elif ra == -1:
                    src = la
                else:
                    dist_l = i - la
                    dist_r = ra - i
                    src = la if dist_l <= dist_r else ra
        if src == -1:
            continue
        new_labels[i] = labels[src]
        if 0 <= src < len(char_probs):
            new_probs[i] = np.array(char_probs[src], copy=True)
    return new_labels, new_probs


def _byte_labels_to_char_labels(
    text: str,
    byte_labels: np.ndarray,
    byte_probs: Optional[np.ndarray],
    num_classes: int,
    ) -> Tuple[List[int], List[np.ndarray]]:
    labels: List[int] = []
    probs: List[np.ndarray] = []
    bpos = 0
    for ch in text:
        chunk = ch.encode("utf-8", "ignore")
        length = len(chunk)
        if length == 0:
            labels.append(0)
            probs.append(np.zeros((num_classes,), dtype=np.float32))
            continue
        seg = byte_labels[bpos:bpos + length]
        if len(seg) == 0:
            labels.append(0)
            probs.append(np.zeros((num_classes,), dtype=np.float32))
        else:
            values, counts = np.unique(seg, return_counts=True)
            lbl = int(values[np.argmax(counts)])
            labels.append(lbl)
            if byte_probs is not None and len(byte_probs) >= bpos + length:
                avg = np.mean(byte_probs[bpos:bpos + length], axis=0)
                probs.append(avg.astype(np.float32))
            else:
                p = np.zeros((num_classes,), dtype=np.float32)
                p[lbl] = 1.0
                probs.append(p)
        bpos += length
    labels, probs = _relabel_whitespace_from_neighbors(text, labels, probs)
    return labels, probs


def _relabel_whitespace_labels_from_neighbors(
    text: str,
    labels: List[int],
) -> List[int]:
    n = len(text)
    if n == 0 or not labels or len(labels) != n:
        return labels
    if not any(ch in _VISUAL_WHITESPACE_SET for ch in text):
        return labels

    new_labels = list(labels)

    line_starts: List[int] = [0]
    for idx, ch in enumerate(text):
        if ch == "\n" and idx + 1 < n:
            line_starts.append(idx + 1)
    line_starts = sorted(set(line_starts))
    line_segments: List[Tuple[int, int]] = []
    for i, start in enumerate(line_starts):
        end = line_starts[i + 1] if i + 1 < len(line_starts) else n
        if start < end:
            line_segments.append((start, end))

    left_same_line = [-1] * n
    right_same_line = [-1] * n
    for start, end in line_segments:
        last_non_ws = -1
        for i in range(start, end):
            if text[i] not in _VISUAL_WHITESPACE_SET:
                last_non_ws = i
            left_same_line[i] = last_non_ws
        last_non_ws = -1
        for i in range(end - 1, start - 1, -1):
            if text[i] not in _VISUAL_WHITESPACE_SET:
                last_non_ws = i
            right_same_line[i] = last_non_ws

    left_any = [-1] * n
    right_any = [-1] * n
    last_non_ws = -1
    for i in range(n):
        if text[i] not in _VISUAL_WHITESPACE_SET:
            last_non_ws = i
        left_any[i] = last_non_ws
    last_non_ws = -1
    for i in range(n - 1, -1, -1):
        if text[i] not in _VISUAL_WHITESPACE_SET:
            last_non_ws = i
        right_any[i] = last_non_ws

    for i, ch in enumerate(text):
        if ch not in _VISUAL_WHITESPACE_SET:
            continue
        src = -1
        ls = left_same_line[i]
        rs = right_same_line[i]
        if ls != -1 or rs != -1:
            if ls == -1:
                src = rs
            elif rs == -1:
                src = ls
            else:
                src = ls if (i - ls) <= (rs - i) else rs
        else:
            la = left_any[i]
            ra = right_any[i]
            if la != -1 or ra != -1:
                if la == -1:
                    src = ra
                elif ra == -1:
                    src = la
                else:
                    src = la if (i - la) <= (ra - i) else ra
        if src != -1:
            new_labels[i] = labels[src]
    return new_labels


def _byte_labels_to_char_labels_only(
    text: str,
    byte_labels: np.ndarray,
) -> List[int]:
    labels: List[int] = []
    bpos = 0
    for ch in text:
        chunk = ch.encode("utf-8", "ignore")
        length = len(chunk)
        if length == 0:
            labels.append(0)
            continue
        seg = byte_labels[bpos:bpos + length]
        if len(seg) == 0:
            labels.append(0)
        else:
            values, counts = np.unique(seg, return_counts=True)
            labels.append(int(values[np.argmax(counts)]))
        bpos += length
    return _relabel_whitespace_labels_from_neighbors(text, labels)


_INLINE_WHITESPACE_SET = frozenset((" ", "\t"))
_THESIS_BOUNDARY_DELIMITER_CHARS = frozenset(
    ("<", ">", "/", "\\", '"', "'", "`", "(", ")", "[", "]", "{", "}", ",", ";", ":", "=")
)
_THESIS_BOUNDARY_ADJACENT_CHARS = _THESIS_BOUNDARY_DELIMITER_CHARS | _INLINE_WHITESPACE_SET
_THESIS_BOUNDARY_MIN_IMPROVEMENT = 0.75


def _build_label_runs(labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    if not labels:
        return []
    runs: List[Tuple[int, int, int]] = []
    current = int(labels[0])
    start = 0
    for idx in range(1, len(labels)):
        label = int(labels[idx])
        if label != current:
            runs.append((start, idx, current))
            start = idx
            current = label
    runs.append((start, len(labels), current))
    return runs


def _smooth_min_run(labels: List[int], min_run: int) -> List[int]:
    if min_run <= 1 or len(labels) == 0:
        return labels
    runs = _build_label_runs(labels)
    if len(runs) <= 2:
        return labels
    arr = labels[:]
    for idx, (s, e, lbl) in enumerate(runs):
        length = e - s
        if length >= min_run:
            continue
        left_lbl = runs[idx - 1][2] if idx - 1 >= 0 else lbl
        right_lbl = runs[idx + 1][2] if idx + 1 < len(runs) else lbl
        left_len = runs[idx - 1][1] - runs[idx - 1][0] if idx - 1 >= 0 else 0
        right_len = runs[idx + 1][1] - runs[idx + 1][0] if idx + 1 < len(runs) else 0
        new_lbl = left_lbl if left_len >= right_len else right_lbl
        for i in range(s, e):
            arr[i] = new_lbl
    return arr


def _char_prob_value(char_probs: Sequence[Any], pos: int, label: int) -> float:
    if pos < 0 or pos >= len(char_probs):
        return 0.0
    row = char_probs[pos]
    try:
        if isinstance(row, Mapping):
            value = row.get(str(int(label)), row.get(int(label), 0.0))
            return float(value)
        arr = np.asarray(row, dtype=np.float32)
        label_idx = int(label)
        if label_idx < 0 or label_idx >= arr.size:
            return 0.0
        return float(arr[label_idx])
    except Exception:
        return 0.0


def _mean_label_support(
    char_probs: Sequence[Any],
    start: int,
    end: int,
    label: int,
) -> float:
    lo = max(0, int(start))
    hi = max(lo, min(int(end), len(char_probs)))
    if hi <= lo:
        return 0.0
    total = 0.0
    count = 0
    for pos in range(lo, hi):
        total += _char_prob_value(char_probs, pos, int(label))
        count += 1
    return total / max(count, 1)


def _is_identifier_like_char(ch: str) -> bool:
    return bool(ch) and (ch.isalnum() or ch in ("_", "$", "-"))


def _thesis_boundary_local_score(text: str, boundary: int) -> float:
    if boundary < 0 or boundary > len(text):
        return float("-inf")
    left = text[boundary - 1] if boundary > 0 else ""
    right = text[boundary] if boundary < len(text) else ""
    score = 0.0
    if left in _THESIS_BOUNDARY_DELIMITER_CHARS:
        score += 1.25
    elif left in _INLINE_WHITESPACE_SET:
        score += 0.20
    if right in _THESIS_BOUNDARY_DELIMITER_CHARS:
        score += 1.25
    elif right in _INLINE_WHITESPACE_SET:
        score += 0.20
    if left in _THESIS_BOUNDARY_DELIMITER_CHARS and right in _THESIS_BOUNDARY_DELIMITER_CHARS:
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


def _apply_boundary_shift(
    labels: List[int],
    current: int,
    boundary: int,
    left_label: int,
    right_label: int,
) -> List[int]:
    out = labels[:]
    if boundary < current:
        for pos in range(boundary, current):
            out[pos] = int(right_label)
    elif boundary > current:
        for pos in range(current, boundary):
            out[pos] = int(left_label)
    return out


def _score_thesis_boundary_candidate(
    text: str,
    labels: Sequence[int],
    char_probs: Sequence[Any],
    left_run: Tuple[int, int, int],
    right_run: Tuple[int, int, int],
    boundary: int,
) -> Optional[float]:
    del labels
    current = int(left_run[1])
    left_start, _, left_label = left_run
    _, right_end, right_label = right_run
    if boundary < int(left_start) or boundary > int(right_end):
        return None
    left_adjacent = text[boundary - 1] if boundary > 0 else ""
    right_adjacent = text[boundary] if boundary < len(text) else ""
    if left_adjacent == "\n" or right_adjacent == "\n":
        return None
    if boundary != current and (
        left_adjacent not in _THESIS_BOUNDARY_ADJACENT_CHARS
        and right_adjacent not in _THESIS_BOUNDARY_ADJACENT_CHARS
    ):
        return None

    if boundary < current:
        moved_positions = range(boundary, current)
        src_label = int(left_label)
        dest_label = int(right_label)
    else:
        moved_positions = range(current, boundary)
        src_label = int(right_label)
        dest_label = int(left_label)

    score = _thesis_boundary_local_score(text, boundary)
    for pos in moved_positions:
        ch = text[pos]
        if ch == "\n":
            return None
        src_prob = _char_prob_value(char_probs, pos, src_label)
        dest_prob = _char_prob_value(char_probs, pos, dest_label)
        if ch not in _THESIS_BOUNDARY_DELIMITER_CHARS and dest_prob < (0.5 * src_prob):
            return None
        score += 0.5 * (dest_prob - src_prob)
        if ch in _THESIS_BOUNDARY_DELIMITER_CHARS:
            score += 0.15
        elif ch in _INLINE_WHITESPACE_SET:
            score += 0.02
    return score


def _snap_thesis_boundaries_to_delimiters(
    text: str,
    labels: List[int],
    char_probs: Sequence[Any],
    *,
    max_shift: int,
) -> List[int]:
    if int(max_shift) <= 0 or len(labels) <= 1:
        return labels
    out = [int(label) for label in labels]
    max_passes = max(1, len(out) * 2)
    for _ in range(max_passes):
        runs = _build_label_runs(out)
        if len(runs) <= 1:
            break
        changed = False
        for idx in range(len(runs) - 1):
            left_run = runs[idx]
            right_run = runs[idx + 1]
            current = int(left_run[1])
            # Only snap when the current boundary cuts through identifier-like
            # content on both sides. If either neighbor is already a delimiter
            # or whitespace, the boundary is on a natural seam and snapping a
            # short distance onto a different delimiter would relabel a
            # syntactic character (for example the closing quote of an HTML
            # attribute) without evidence that the model was wrong.
            left_curr = text[current - 1] if current > 0 else ""
            right_curr = text[current] if current < len(text) else ""
            if (
                left_curr in _THESIS_BOUNDARY_ADJACENT_CHARS
                or right_curr in _THESIS_BOUNDARY_ADJACENT_CHARS
            ):
                continue
            current_score = _score_thesis_boundary_candidate(
                text,
                out,
                char_probs,
                left_run,
                right_run,
                current,
            )
            if current_score is None:
                current_score = _thesis_boundary_local_score(text, current)
            best_boundary = current
            best_score = current_score
            for shift in range(-int(max_shift), int(max_shift) + 1):
                if shift == 0:
                    continue
                candidate = current + shift
                score = _score_thesis_boundary_candidate(
                    text,
                    out,
                    char_probs,
                    left_run,
                    right_run,
                    candidate,
                )
                if score is None:
                    continue
                if score > (best_score + 1e-6):
                    best_boundary = candidate
                    best_score = score
            if best_boundary != current and best_score >= (current_score + _THESIS_BOUNDARY_MIN_IMPROVEMENT):
                out = _apply_boundary_shift(out, current, best_boundary, left_run[2], right_run[2])
                changed = True
                break
        if not changed:
            break
    return out


def _choose_thesis_min_run_target(
    labels: Sequence[int],
    char_probs: Sequence[Any],
    run: Tuple[int, int, int],
    left_run: Tuple[int, int, int],
    right_run: Tuple[int, int, int],
) -> int:
    del labels
    start, end, _label = run
    left_start, left_end, left_label = left_run
    right_start, right_end, right_label = right_run
    left_label = int(left_label)
    right_label = int(right_label)
    if left_label == right_label:
        return left_label

    left_len = int(left_end - left_start)
    right_len = int(right_end - right_start)
    if left_len > 10 and right_len > 10:
        window_start = max(0, int(start) - 10)
        window_end = min(len(char_probs), int(end) + 10)
        left_support = _mean_label_support(char_probs, window_start, window_end, left_label)
        right_support = _mean_label_support(char_probs, window_start, window_end, right_label)
        return left_label if left_support >= right_support else right_label

    if left_len != right_len:
        return left_label if left_len > right_len else right_label

    left_support = _mean_label_support(char_probs, max(int(left_end) - 10, int(left_start)), int(end), left_label)
    right_support = _mean_label_support(char_probs, int(start), min(int(right_start) + 10, int(right_end)), right_label)
    return left_label if left_support >= right_support else right_label


def _normalize_thesis_min_runs(
    labels: List[int],
    char_probs: Sequence[Any],
    *,
    min_run_chars: int,
) -> List[int]:
    if int(min_run_chars) <= 1 or len(labels) <= 2:
        return labels
    out = [int(label) for label in labels]
    max_passes = max(1, len(out))
    for _ in range(max_passes):
        runs = _build_label_runs(out)
        changed = False
        for idx in range(1, len(runs) - 1):
            start, end, _label = runs[idx]
            if (end - start) >= int(min_run_chars):
                continue
            target = _choose_thesis_min_run_target(out, char_probs, runs[idx], runs[idx - 1], runs[idx + 1])
            for pos in range(start, end):
                out[pos] = int(target)
            changed = True
            break
        if not changed:
            break
    return out


def _apply_thesis_other_threshold(
    labels: List[int],
    char_probs: Sequence[Any],
    *,
    threshold: float,
    other_id: Optional[int],
) -> Tuple[List[int], int]:
    if other_id is None or threshold <= 0.0:
        return labels, 0
    out = [int(label) for label in labels]
    changed = 0
    for pos in range(len(out)):
        if pos >= len(char_probs):
            continue
        try:
            row = np.asarray(char_probs[pos], dtype=np.float32)
            max_prob = float(row.max()) if row.size else None
        except Exception:
            max_prob = None
        if max_prob is not None and max_prob < float(threshold) and out[pos] != int(other_id):
            out[pos] = int(other_id)
            changed += 1
    return out, changed


def _count_label_changes(before: Sequence[int], after: Sequence[int]) -> int:
    return sum(1 for left, right in zip(before, after) if int(left) != int(right))


def _labels_to_segments(labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    segments: List[Tuple[int, int, int]] = []
    if labels:
        cur = int(labels[0])
        start = 0
        for idx in range(1, len(labels)):
            label = int(labels[idx])
            if label != cur:
                segments.append((start, idx, cur))
                start = idx
                cur = label
        segments.append((start, len(labels), cur))
    return segments


def _apply_thesis_postprocess(
    text: str,
    labels: List[int],
    char_probs: Sequence[Any],
    *,
    other_threshold: float,
    other_id: Optional[int],
    min_run_chars: int,
    boundary_snap_max_shift: int,
) -> Tuple[List[int], Dict[str, Any]]:
    original = [int(label) for label in labels]
    current = original[:]
    stats: Dict[str, Any] = {
        "profile": POSTPROCESS_PROFILE_THESIS,
        "settings": {
            "other_threshold": float(other_threshold),
            "min_run_chars": int(min_run_chars),
            "boundary_snap_max_shift": int(boundary_snap_max_shift),
            "stages": [
                "whitespace_relabel",
                "other_gating",
                "boundary_snap",
                "min_run",
            ],
            "excluded_viewer_stages": [
                "markdown_structure_fill",
                "paired_delimiter_fill",
                "local_host_fill",
                "newline_snap",
            ],
        },
        "processed_chars": int(len(current)),
        "other_thresholded_chars": 0,
        "boundary_snap_changed_chars": 0,
        "min_run_changed_chars": 0,
        "changed_chars": 0,
    }
    if not current:
        return current, stats

    gated, thresholded = _apply_thesis_other_threshold(
        current,
        char_probs,
        threshold=float(other_threshold),
        other_id=other_id,
    )
    current = gated
    stats["other_thresholded_chars"] = int(thresholded)

    before_boundary = current[:]
    current = _snap_thesis_boundaries_to_delimiters(
        text,
        current,
        char_probs,
        max_shift=int(boundary_snap_max_shift),
    )
    stats["boundary_snap_changed_chars"] = int(_count_label_changes(before_boundary, current))

    before_min_run = current[:]
    current = _normalize_thesis_min_runs(
        current,
        char_probs,
        min_run_chars=int(min_run_chars),
    )
    stats["min_run_changed_chars"] = int(_count_label_changes(before_min_run, current))
    stats["changed_chars"] = int(_count_label_changes(original, current))
    return current, stats


class SegmenterRunner:
    def __init__(
        self,
        checkpoint_path: Optional[str],
        *,
        arch: str = "unet1d",
        model_dim: int,
        channels: Sequence[int],
        mamba_layers: int = 6,
        mamba_d_state: int = 8,
        mamba_expand: int = 1,
        mamba_dt_rank: int = 16,
        mamba_conv: int = 4,
        mamba_bidirectional: bool = True,
        dtype: str = "bfloat16",
        chunk: int = DEFAULT_CHUNK_SIZE,
        device: Optional[str] = None,
        batch_size: int = 16,
        inference_backend: str = "auto",
        full_memory_budget_bytes: Optional[int] = None,
    ):
        self.arch = str(arch).lower().strip()
        requested_device = str(device or "auto").lower().strip()
        backend: Optional[str]
        if self.arch == "magika":
            if requested_device not in {"", "auto", "cpu"}:
                raise RuntimeError("Magika evaluation currently supports CPU execution only.")
            backend = "cpu"
        else:
            backend = _resolve_backend(device)
            available = _available_backends()
            if backend is not None and backend not in available:
                raise RuntimeError(
                    f"Requested backend '{backend}' not available. Available: {sorted(available)}"
                )
        self.backend = backend
        self.inference_backend = str(inference_backend).lower().strip()
        if self.inference_backend not in {"auto", "fast", "legacy"}:
            raise ValueError(
                f"Unknown inference backend '{inference_backend}'. "
                "Expected one of: auto, fast, legacy."
            )
        self.chunk = max(64, int(chunk or DEFAULT_CHUNK_SIZE))
        if self.chunk <= 0:
            raise ValueError("Chunk size must be positive.")
        self.batch_size = int(batch_size)
        _assert_unet_chunk_size(self.arch, self.chunk)
        self._magika_segmenter: Optional[SlidingWindowMagikaSegmenter] = None
        self.magika_module_version: Optional[str] = None
        self.magika_model_name: Optional[str] = None
        self.num_classes = (
            int(getattr(cfg, "OTHER_CLASS_INDEX", cfg.NUM_CLASSES - 1)) + 1
            if self.arch == "magika"
            else int(cfg.NUM_CLASSES)
        )
        self.last_postprocess_stats: Dict[str, Any] = {"profile": POSTPROCESS_PROFILE_OFF}
        if self.arch == "magika":
            self._weight_cache = {}
            self.model = None
            self.fast_model = None
            self.params = None
            self.channels = ()
            self._apply_legacy = None
            self._apply_fast = None
            self._apply = None
            self._fast_engine = None
            other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
            if other_idx is None:
                raise RuntimeError("Magika evaluation requires cfg.OTHER_CLASS_INDEX.")
            self._magika_segmenter = SlidingWindowMagikaSegmenter(
                label_to_id=cfg.LANG2ID,
                other_class_index=int(other_idx),
                batch_size=self.batch_size,
                window_size=self.chunk,
            )
            self.magika_module_version = self._magika_segmenter.module_version
            self.magika_model_name = self._magika_segmenter.model_name
            return

        if not checkpoint_path:
            pass # Bypassed for speed benchmark

        dt = getattr(jnp, dtype)
        requested_channels = tuple(int(ch) for ch in channels)
        self._weight_cache: Dict[int, np.ndarray] = {}
        execution_backend = str(backend or jax.default_backend()).lower().strip()
        self._apply_legacy = None
        self._apply_fast = None
        cuda_kernel_available = False

        def _initialize_unet(channel_values: Sequence[int]):
            model = UNet1D(
                num_classes=self.num_classes,
                emb_dim=model_dim,
                channels=tuple(int(ch) for ch in channel_values),
                dtype=dt,
            )
            dummy_tokens = jnp.full((1, self.chunk), cfg.PAD_BYTE_ID, dtype=jnp.int32)
            variables = model.init({"params": jax.random.PRNGKey(0)}, dummy_tokens, train=False)
            template = variables["params"]
            params = _load_params_from_any(checkpoint_path, template)
            return model, params

        def _initialize_mamba(
            *,
            layers: int,
            d_state: int,
            expand: int,
            dt_rank: int,
            conv: int,
            bidirectional: bool,
            inference_kernel: str = "default",
            use_remat: bool = True,
        ):
            model = Mamba1D(
                num_classes=self.num_classes,
                d_model=int(model_dim),
                n_layers=int(layers),
                d_state=int(d_state),
                expand=int(expand),
                dt_rank=int(dt_rank),
                d_conv=int(conv),
                bidirectional=bool(bidirectional),
                dtype=dt,
                inference_kernel=str(inference_kernel),
                use_remat=bool(use_remat),
            )
            dummy_tokens = jnp.full((1, self.chunk), cfg.PAD_BYTE_ID, dtype=jnp.int32)
            variables = model.init({"params": jax.random.PRNGKey(0)}, dummy_tokens, train=False)
            template = variables["params"]
            params = _load_params_from_any(checkpoint_path, template)
            return model, params

        if self.arch == "mamba":
            try:
                model, params = _initialize_mamba(
                    layers=mamba_layers,
                    d_state=mamba_d_state,
                    expand=mamba_expand,
                    dt_rank=mamba_dt_rank,
                    conv=mamba_conv,
                    bidirectional=mamba_bidirectional,
                )
            except ScopeParamShapeError:
                fallback = _load_checkpoint_hparams(Path(checkpoint_path))
                model, params = _initialize_mamba(
                    layers=int(fallback.get("mamba_layers", mamba_layers)),
                    d_state=int(fallback.get("mamba_d_state", mamba_d_state)),
                    expand=int(fallback.get("mamba_expand", mamba_expand)),
                    dt_rank=int(fallback.get("mamba_dt_rank", mamba_dt_rank)),
                    conv=int(fallback.get("mamba_conv", mamba_conv)),
                    bidirectional=bool(fallback.get("mamba_bidirectional", mamba_bidirectional)),
                )
            requested_channels = ()
            fast_model = None
            if execution_backend == "gpu" and self.inference_backend in {"auto", "fast"} and has_cuda_mamba_kernel():
                fast_model = Mamba1D(
                    num_classes=self.num_classes,
                    d_model=int(model.d_model),
                    n_layers=int(model.n_layers),
                    d_state=int(model.d_state),
                    expand=int(model.expand),
                    dt_rank=int(model.dt_rank),
                    d_conv=int(model.d_conv),
                    bidirectional=bool(model.bidirectional),
                    dtype=model.dtype,
                    inference_kernel="cuda_fast",
                    use_remat=False,
                )
                cuda_kernel_available = True
        else:
            try:
                model, params = _initialize_unet(requested_channels)
            except ScopeParamShapeError:
                fallback = _load_checkpoint_hparams(Path(checkpoint_path))
                fallback_channels = fallback.get("channels")
                fallback_tuple: Tuple[int, ...] = tuple(int(ch) for ch in fallback_channels) if fallback_channels else ()
                if fallback_tuple and fallback_tuple != requested_channels:
                    model, params = _initialize_unet(fallback_tuple)
                    requested_channels = fallback_tuple
                else:
                    raise

        self.model = model
        self.fast_model = fast_model if self.arch == "mamba" else None
        self.params = params
        self.channels = tuple(int(ch) for ch in requested_channels) if requested_channels else ()

        def _jit_apply(module):
            fn = lambda tokens: module.apply({"params": self.params}, tokens, train=False)
            if backend:
                return jax.jit(fn, backend=backend)
            return jax.jit(fn)

        self._apply_legacy = _jit_apply(self.model)
        self._apply_fast = _jit_apply(self.fast_model) if self.fast_model is not None else self._apply_legacy
        self._apply = self._apply_legacy
        self._fast_engine: Optional[FastInferenceEngine] = None
        if self.inference_backend in {"auto", "fast"}:
            self._fast_engine = FastInferenceEngine(
                apply_tokens=self._apply_fast,
                num_classes=self.num_classes,
                pad_token_id=int(cfg.PAD_BYTE_ID),
                chunk_size=self.chunk,
                batch_size=self.batch_size,
                sanitize_bytes=sanitize_bytes,
                sanitize_tokens=sanitize_tokens,
                arch=self.arch,
                inference_backend=self.inference_backend,
                actual_backend=(backend or str(jax.default_backend())),
                log_fn=lambda message: print(message, flush=True),
                model_dim=int(model_dim),
                channels=self.channels,
                mamba_layers=int(mamba_layers),
                mamba_d_state=int(mamba_d_state),
                mamba_expand=int(mamba_expand),
                mamba_bidirectional=bool(mamba_bidirectional),
                cuda_kernel_available=cuda_kernel_available,
                full_memory_budget_bytes=full_memory_budget_bytes,
            )

    # ------------------------------------------------------------------
    # Low-level segmentation
    # ------------------------------------------------------------------

    def _window_weights_cached(self, length: int) -> np.ndarray:
        cached = self._weight_cache.get(length)
        if cached is None:
            cached = _window_weights(length)
            self._weight_cache[length] = cached
        return cached

    def _segment_bytes_legacy(self, byte_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.arch == "magika":
            if self._magika_segmenter is None:
                raise RuntimeError("Magika runner is not initialized.")
            return self._magika_segmenter.segment_bytes(byte_arr)
        byte_arr = sanitize_bytes(byte_arr)
        N = int(len(byte_arr))
        if N == 0:
            return np.zeros((0,), dtype=np.uint8), np.zeros((0, self.num_classes), dtype=np.float32)

        spans = build_window_spans(N, self.chunk)
        windows = [byte_arr[start:end] for start, end in spans]

        out_bytes = np.zeros((N,), dtype=np.uint8)
        probs_accum = np.zeros((N, self.num_classes), dtype=np.float32)
        weight_accum = np.zeros((N,), dtype=np.float32)

        batch_size = self.batch_size
        for i in range(0, len(windows), batch_size):
            span_slice = spans[i:i + batch_size]
            actual = len(span_slice)
            tokens = np.full((batch_size, self.chunk), cfg.PAD_BYTE_ID, dtype=np.int32)
            for j in range(actual):
                win_bytes = windows[i + j]
                length = min(len(win_bytes), self.chunk)
                tokens[j, :length] = win_bytes[:length].astype(np.int32)
            tokens = sanitize_tokens(tokens)
            logits = self._apply_legacy(jnp.array(tokens, dtype=jnp.int32))
            logits = np.array(logits)[:actual, :self.chunk]
            probs = np.array(jax.nn.softmax(logits, axis=-1))

            for j, (start, end) in enumerate(span_slice):
                plen = end - start
                if plen <= 0:
                    continue
                plen = min(plen, self.chunk)
                weights = self._window_weights_cached(plen)
                probs_slice = probs[j, :plen]
                probs_accum[start:end] += probs_slice * weights[:, None]
                weight_accum[start:end] += weights

        nonzero = weight_accum > 0
        if np.any(nonzero):
            probs_accum[nonzero] /= weight_accum[nonzero][:, None]
        zero_mask = ~nonzero
        if np.any(zero_mask):
            probs_accum[zero_mask] = 1.0 / self.num_classes

        out_bytes[:] = np.argmax(probs_accum, axis=-1).astype(np.uint8)
        return out_bytes, probs_accum

    def _segment_bytes(self, byte_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.arch == "magika":
            return self._segment_bytes_legacy(byte_arr)
        if self.inference_backend == "legacy" or self._fast_engine is None:
            return self._segment_bytes_legacy(byte_arr)
        try:
            byte_labels, byte_probs, _ = self._fast_engine.segment_bytes(byte_arr)
            return byte_labels, byte_probs
        except FastInferenceFailure as exc:
            if self.inference_backend == "auto":
                print(
                    format_auto_fallback_message(
                        from_path=exc.source_path,
                        to_path="legacy",
                        trigger=exc.trigger,
                        reason=exc.reason,
                    ),
                    flush=True,
                )
                return self._segment_bytes_legacy(byte_arr)
            raise
        except Exception as exc:
            if self.inference_backend == "auto":
                print(
                    format_auto_fallback_message(
                        from_path="fast",
                        to_path="legacy",
                        trigger="runtime_error",
                        reason=str(exc),
                    ),
                    flush=True,
                )
                return self._segment_bytes_legacy(byte_arr)
            raise

    def _segment_bytes_labels_only_legacy(self, byte_arr: np.ndarray) -> np.ndarray:
        labels, _ = self._segment_bytes_legacy(byte_arr)
        return labels

    def _segment_bytes_labels_only(self, byte_arr: np.ndarray) -> np.ndarray:
        if self.arch == "magika":
            return self._segment_bytes_labels_only_legacy(byte_arr)
        if self.inference_backend == "legacy" or self._fast_engine is None:
            return self._segment_bytes_labels_only_legacy(byte_arr)
        try:
            byte_labels, _ = self._fast_engine.segment_bytes_labels_only(byte_arr)
            return byte_labels
        except FastInferenceFailure as exc:
            if self.inference_backend == "auto":
                print(
                    format_auto_fallback_message(
                        from_path=exc.source_path,
                        to_path="legacy",
                        trigger=exc.trigger,
                        reason=exc.reason,
                    ),
                    flush=True,
                )
                return self._segment_bytes_labels_only_legacy(byte_arr)
            raise
        except Exception as exc:
            if self.inference_backend == "auto":
                print(
                    format_auto_fallback_message(
                        from_path="fast",
                        to_path="legacy",
                        trigger="runtime_error",
                        reason=str(exc),
                    ),
                    flush=True,
                )
                return self._segment_bytes_labels_only_legacy(byte_arr)
            raise

    def segment_byte_arrays_batch_labels_only(
        self,
        byte_arrays: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
        arrays = [sanitize_bytes(np.asarray(arr, dtype=np.uint8)) for arr in byte_arrays]
        if not arrays:
            return [], []
        if self.arch == "magika":
            if self._magika_segmenter is None:
                raise RuntimeError("Magika runner is not initialized.")
            return self._magika_segmenter.segment_byte_arrays_batch_labels_only(arrays)
        if self.inference_backend == "legacy" or self._fast_engine is None:
            labels = [self._segment_bytes_labels_only_legacy(arr) for arr in arrays]
            spans = [build_window_spans(int(arr.shape[0]), self.chunk) for arr in arrays]
            return labels, spans
        try:
            return self._fast_engine.segment_bytes_batch_labels_only(arrays)
        except FastInferenceFailure as exc:
            if self.inference_backend == "auto":
                print(
                    format_auto_fallback_message(
                        from_path=exc.source_path,
                        to_path="legacy",
                        trigger=exc.trigger,
                        reason=exc.reason,
                    ),
                    flush=True,
                )
                labels = [self._segment_bytes_labels_only_legacy(arr) for arr in arrays]
                spans = [build_window_spans(int(arr.shape[0]), self.chunk) for arr in arrays]
                return labels, spans
            raise
        except Exception as exc:
            if self.inference_backend == "auto":
                print(
                    format_auto_fallback_message(
                        from_path="fast",
                        to_path="legacy",
                        trigger="runtime_error",
                        reason=str(exc),
                    ),
                    flush=True,
                )
                labels = [self._segment_bytes_labels_only_legacy(arr) for arr in arrays]
                spans = [build_window_spans(int(arr.shape[0]), self.chunk) for arr in arrays]
                return labels, spans
            raise

    def clear_fast_execution_history(self) -> None:
        if self.arch == "magika":
            if self._magika_segmenter is not None:
                self._magika_segmenter.clear_execution_history()
            return
        if self._fast_engine is not None:
            self._fast_engine.execution_history.clear()

    def get_fast_execution_history(self) -> List[Any]:
        if self.arch == "magika":
            if self._magika_segmenter is None:
                return []
            return self._magika_segmenter.get_execution_history()
        if self._fast_engine is None:
            return []
        return list(self._fast_engine.execution_history)

    def segment_text(
        self,
        text: str,
        *,
        min_run_chars: int = 1,
        postprocess_profile: str = POSTPROCESS_PROFILE_OFF,
        other_threshold: float = 0.0,
        postprocess_min_run_chars: int = 5,
        postprocess_boundary_snap_max_shift: int = 2,
    ) -> Tuple[List[Tuple[int, int, int]], List[int], List[np.ndarray]]:
        text = normalize_eval_text(text)
        byte_arr = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
        byte_labels, byte_probs = self._segment_bytes(byte_arr)
        char_labels, char_probs = _byte_labels_to_char_labels(text, byte_labels, byte_probs, self.num_classes)
        profile = str(postprocess_profile or POSTPROCESS_PROFILE_OFF).lower().strip()
        if profile == POSTPROCESS_PROFILE_THESIS:
            char_labels, stats = _apply_thesis_postprocess(
                text,
                char_labels,
                char_probs,
                other_threshold=float(other_threshold or 0.0),
                other_id=getattr(cfg, "OTHER_CLASS_INDEX", None),
                min_run_chars=int(postprocess_min_run_chars),
                boundary_snap_max_shift=int(postprocess_boundary_snap_max_shift),
            )
            self.last_postprocess_stats = stats
        elif profile == POSTPROCESS_PROFILE_OFF:
            char_labels = _smooth_min_run(char_labels, min_run_chars)
            self.last_postprocess_stats = {"profile": POSTPROCESS_PROFILE_OFF}
        else:
            raise ValueError(f"Unknown postprocess profile: {postprocess_profile!r}")
        segments = _labels_to_segments(char_labels)
        return segments, char_labels, char_probs

    def segment_text_labels_only(self, text: str, *, min_run_chars: int = 1) -> Tuple[List[Tuple[int, int, int]], List[int]]:
        text = normalize_eval_text(text)
        byte_arr = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
        byte_labels = self._segment_bytes_labels_only(byte_arr)
        char_labels = _byte_labels_to_char_labels_only(text, byte_labels)
        char_labels = _smooth_min_run(char_labels, min_run_chars)
        segments = _labels_to_segments(char_labels)
        return segments, char_labels


class _PredictionCacheTaskState:
    __slots__ = ("labels", "probs")

    def __init__(self) -> None:
        self.labels: List[np.ndarray] = []
        self.probs: List[np.ndarray] = []


class PredictionCache:
    """Per-task cache of raw model outputs.

    Format per task: ``<root>/<task>__raw.npz`` containing flattened ``labels``
    (int16) and full per-character probability vectors ``probs`` (float16,
    shape ``[total_chars, num_classes]``) plus an ``offsets`` int64 array of
    length ``N_samples + 1``. Storing the full distribution is required
    because the thesis post-processing reads the probability of arbitrary
    classes via ``_char_prob_value`` (e.g. for boundary snap and minimum-run
    label selection), not only the per-position argmax.
    """

    def __init__(self, root: Path, mode: str) -> None:
        assert mode in {"read", "write"}, f"Unsupported cache mode {mode!r}"
        self.root = Path(root)
        self.mode = mode
        self._write_state: Dict[str, _PredictionCacheTaskState] = {}
        self._read_state: Dict[str, Dict[str, np.ndarray]] = {}
        if mode == "write":
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, task: str) -> Path:
        return self.root / f"{task}__raw.npz"

    def begin_task(self, task: str) -> None:
        if self.mode == "write":
            self._write_state.setdefault(task, _PredictionCacheTaskState())
            return
        if task in self._read_state:
            return
        data = np.load(self._path(task), allow_pickle=False)
        self._read_state[task] = {
            "labels": data["labels"],
            "probs": data["probs"],
            "offsets": data["offsets"],
        }

    def end_task(self, task: str) -> None:
        if self.mode != "write":
            return
        st = self._write_state.pop(task, None)
        if st is None or not st.labels:
            return
        labels_concat = np.concatenate(st.labels).astype(np.int16, copy=False)
        probs_concat = np.concatenate(st.probs, axis=0).astype(np.float16, copy=False)
        offsets = np.zeros(len(st.labels) + 1, dtype=np.int64)
        cum = 0
        for i, arr in enumerate(st.labels):
            cum += len(arr)
            offsets[i + 1] = cum
        np.savez_compressed(
            self._path(task),
            labels=labels_concat,
            probs=probs_concat,
            offsets=offsets,
        )

    def write_sample(self, task: str, labels: np.ndarray, probs: np.ndarray) -> None:
        st = self._write_state[task]
        st.labels.append(np.asarray(labels, dtype=np.int16))
        probs_arr = np.asarray(probs, dtype=np.float16)
        if probs_arr.ndim != 2:
            raise ValueError(f"Expected 2-D probs matrix, got shape {probs_arr.shape}")
        st.probs.append(probs_arr)

    def read_sample(self, task: str, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        rs = self._read_state[task]
        off = rs["offsets"]
        s, e = int(off[sample_idx]), int(off[sample_idx + 1])
        return (
            rs["labels"][s:e].astype(np.int64, copy=False),
            rs["probs"][s:e].astype(np.float32, copy=False),
        )


class _RunnerCacheAdapter:
    """Wraps a ``SegmenterRunner`` to either cache raw outputs (write mode)
    or substitute cached outputs in place of model inference (read mode).

    The adapter mirrors the public ``segment_text`` surface used by the
    evaluation loop. In write mode it forwards to the inner runner with
    post-processing disabled, caches the raw labels + per-char max-probability,
    and then applies the requested post-processing locally. In read mode the
    inner runner is not required and the model is not loaded at all.
    """

    def __init__(
        self,
        inner: Optional["SegmenterRunner"],
        cache: PredictionCache,
        *,
        fallback_arch: Optional[str] = None,
        fallback_chunk: Optional[int] = None,
        fallback_batch_size: Optional[int] = None,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.arch = inner.arch if inner is not None else str(fallback_arch or "cached")
        self.chunk = int(inner.chunk if inner is not None else (fallback_chunk or 0))
        self.batch_size = int(inner.batch_size if inner is not None else (fallback_batch_size or 0))
        self.num_classes = int(getattr(inner, "num_classes", cfg.NUM_CLASSES))
        self.magika_module_version = getattr(inner, "magika_module_version", None)
        self.magika_model_name = getattr(inner, "magika_model_name", None)
        self.last_postprocess_stats: Dict[str, Any] = {"profile": POSTPROCESS_PROFILE_OFF}
        self._task: Optional[str] = None
        self._sample_idx: int = 0

    # ------------------------------------------------------------------
    # Lifecycle hooks driven by the evaluation loop.
    def begin_task(self, task: str) -> None:
        self._task = task
        self._sample_idx = 0
        self.cache.begin_task(task)

    def end_task(self) -> None:
        if self._task is not None:
            self.cache.end_task(self._task)
        self._task = None
        self._sample_idx = 0

    # ------------------------------------------------------------------
    # Pass-through helpers required by the evaluation pipeline.
    def clear_fast_execution_history(self) -> None:
        if self.inner is not None and hasattr(self.inner, "clear_fast_execution_history"):
            self.inner.clear_fast_execution_history()

    def get_fast_execution_history(self) -> List[Any]:
        if self.inner is not None and hasattr(self.inner, "get_fast_execution_history"):
            return self.inner.get_fast_execution_history()
        return []

    def segment_byte_arrays_batch_labels_only(self, *args, **kwargs):
        if self.inner is None:
            raise RuntimeError("Cached runner cannot serve raw byte-batch inference.")
        return self.inner.segment_byte_arrays_batch_labels_only(*args, **kwargs)

    def segment_text_labels_only(self, *args, **kwargs):
        if self.inner is None:
            raise RuntimeError("Cached runner cannot serve labels-only inference.")
        return self.inner.segment_text_labels_only(*args, **kwargs)

    def segment_text(
        self,
        text: str,
        *,
        min_run_chars: int = 1,
        postprocess_profile: str = POSTPROCESS_PROFILE_OFF,
        other_threshold: float = 0.0,
        postprocess_min_run_chars: int = 5,
        postprocess_boundary_snap_max_shift: int = 2,
    ) -> Tuple[List[Tuple[int, int, int]], List[int], List[np.ndarray]]:
        if self._task is None:
            raise RuntimeError(
                "RunnerCacheAdapter.segment_text called before begin_task(); "
                "the evaluation loop must announce the current task name."
            )
        norm_text = normalize_eval_text(text)
        if self.cache.mode == "read":
            labels_arr, probs_matrix = self.cache.read_sample(self._task, self._sample_idx)
            char_labels: List[int] = labels_arr.tolist()
            char_probs: List[np.ndarray] = [np.asarray(row, dtype=np.float32) for row in probs_matrix]
        else:
            if self.inner is None:
                raise RuntimeError("Cache write mode requires a backing SegmenterRunner.")
            _, char_labels, char_probs = self.inner.segment_text(
                norm_text,
                min_run_chars=1,
                postprocess_profile=POSTPROCESS_PROFILE_OFF,
            )
            if char_probs:
                probs_matrix = np.stack(
                    [np.asarray(p, dtype=np.float32) for p in char_probs], axis=0
                )
            else:
                probs_matrix = np.zeros((0, self.num_classes), dtype=np.float32)
            self.cache.write_sample(
                self._task,
                np.asarray(char_labels, dtype=np.int16),
                probs_matrix.astype(np.float16, copy=False),
            )
        self._sample_idx += 1

        profile = str(postprocess_profile or POSTPROCESS_PROFILE_OFF).lower().strip()
        if profile == POSTPROCESS_PROFILE_THESIS:
            char_labels, stats = _apply_thesis_postprocess(
                norm_text,
                char_labels,
                char_probs,
                other_threshold=float(other_threshold or 0.0),
                other_id=getattr(cfg, "OTHER_CLASS_INDEX", None),
                min_run_chars=int(postprocess_min_run_chars),
                boundary_snap_max_shift=int(postprocess_boundary_snap_max_shift),
            )
            self.last_postprocess_stats = stats
        elif profile == POSTPROCESS_PROFILE_OFF:
            char_labels = _smooth_min_run(char_labels, int(min_run_chars))
            self.last_postprocess_stats = {"profile": POSTPROCESS_PROFILE_OFF}
        else:
            raise ValueError(f"Unknown postprocess profile: {postprocess_profile!r}")

        segments = _labels_to_segments(char_labels)
        return segments, char_labels, char_probs


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_datasets(data_root: Path, tasks: Optional[Sequence[str]]) -> Dict[str, hfds.Dataset]:
    datasets = {}
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if tasks and name not in tasks:
            continue
        try:
            ds = hfds.load_from_disk(str(entry))
        except Exception as exc:
            print(f"⚠️  Failed to load dataset '{name}': {exc}", flush=True)
            continue
        if len(ds) == 0:
            continue
        datasets[name] = ds
    return datasets


def _derive_task_seed(base_seed: Optional[int], task_name: str) -> int:
    base = int(base_seed or 0) & 0xFFFFFFFF
    name_bytes = task_name.encode("utf-8", "ignore")
    mix = int(np.frombuffer(name_bytes, dtype=np.uint8).sum()) & 0xFFFFFFFF
    seq = np.random.SeedSequence([base, mix])
    return int(seq.generate_state(1)[0])


def _prepare_dataset(
    task_name: str,
    dataset: hfds.Dataset,
    *,
    max_samples: int,
    base_seed: Optional[int],
    preserve_all: bool,
) -> Tuple[hfds.Dataset, int, int, bool]:
    original_len = len(dataset)
    if preserve_all or max_samples <= 0 or original_len <= max_samples:
        return dataset, original_len, original_len, False
    derived_seed = _derive_task_seed(base_seed, task_name)
    rng = np.random.default_rng(derived_seed)
    indices = rng.choice(original_len, size=max_samples, replace=False)
    indices.sort()
    subset = dataset.select(indices.tolist())
    return subset, original_len, max_samples, True


def _parse_metadata(example: Dict[str, Any]) -> Dict[str, Any]:
    raw = example.get("metadata_json")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", "ignore")
        except Exception:
            return {}
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except Exception:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


def _normalize_segments(segments) -> List[Dict[str, int]]:
    """Handle either list-of-dicts or dict-of-lists segment structures."""
    if segments is None:
        return []
    if isinstance(segments, dict):
        def _ensure_list(value, size_hint: int, default=0):
            if value is None:
                return [default] * size_hint
            if isinstance(value, (str, bytes)):
                return [value] * (size_hint or 1)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return list(value)
            try:
                return list(value)
            except TypeError:
                return [value] * size_hint if size_hint else [value]

        raw_labels = segments.get("label") or segments.get("labels") or []
        labels = _ensure_list(raw_labels, 0)
        starts = _ensure_list(segments.get("char_start") or segments.get("start"), len(labels))
        ends = _ensure_list(segments.get("char_end") or segments.get("end"), len(labels))
        normalized: List[Dict[str, int]] = []
        for label, start, end in zip(labels, starts, ends):
            if label is None:
                continue
            try:
                start_i = int(start)
                end_i = int(end)
            except (TypeError, ValueError):
                continue
            normalized.append({"label": str(label), "char_start": start_i, "char_end": end_i})
        return normalized
    if isinstance(segments, (list, tuple)):
        normalized = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            label = seg.get("label")
            if label is None:
                continue
            try:
                start_i = int(seg.get("char_start", 0))
                end_i = int(seg.get("char_end", start_i))
            except (TypeError, ValueError):
                continue
            normalized.append({"label": str(label), "char_start": start_i, "char_end": end_i})
        return normalized
    raise TypeError(f"Unsupported segments type: {type(segments)}")


def _normalize_truth_label_exclusions(excluded_labels: Optional[Sequence[str]]) -> Set[str]:
    return {
        str(label).strip()
        for label in (excluded_labels or ())
        if str(label).strip()
    }


def _excluded_truth_labels_in_segments(
    segments,
    excluded_labels: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    excluded = _normalize_truth_label_exclusions(excluded_labels)
    if not excluded:
        return ()
    found = {
        str(seg.get("label")).strip()
        for seg in _normalize_segments(segments)
        if str(seg.get("label")).strip() in excluded
    }
    return tuple(sorted(found))


def _truth_label_exclusion_payload(
    *,
    requested_total: int,
    excluded_labels: Optional[Sequence[str]],
    unit_name: str,
) -> Optional[Dict[str, Any]]:
    normalized = sorted(_normalize_truth_label_exclusions(excluded_labels))
    if not normalized:
        return None
    return {
        "unit": str(unit_name),
        "requested_labels": list(normalized),
        "requested": int(requested_total),
        "counted": 0,
        "skipped_for_truth_label": 0,
        "skipped_label_counts": {},
    }


def _record_truth_label_exclusion_skip(
    payload: Optional[Dict[str, Any]],
    skipped_labels: Sequence[str],
) -> None:
    if payload is None:
        return
    payload["skipped_for_truth_label"] = int(payload.get("skipped_for_truth_label", 0)) + 1
    label_counts = payload.setdefault("skipped_label_counts", {})
    for label in skipped_labels:
        label_counts[str(label)] = int(label_counts.get(str(label), 0)) + 1


def _truth_label_exclusion_note(
    *,
    arch: str,
    excluded_labels: Optional[Sequence[str]],
) -> Optional[str]:
    normalized = _normalize_truth_label_exclusions(excluded_labels)
    if not normalized:
        return None
    if str(arch).lower().strip() == "magika" and normalized == set(THESIS_ENCODING_LABELS):
        return MAGIKA_REDUCED_SUPPORT_NOTE
    labels_text = ", ".join(sorted(normalized))
    return (
        "Samples/files whose ground-truth labels contain any of the following labels were "
        f"excluded before scoring: {labels_text}."
    )


def _segments_to_labels(content: str, segments: Sequence[dict], label_map: Dict[str, int]) -> np.ndarray:
    length = len(content)
    labels = np.full((length,), -1, dtype=np.int32)
    for seg in segments:
        label = seg.get("label")
        start = int(seg.get("char_start", 0))
        end = int(seg.get("char_end", 0))
        start = max(0, min(length, start))
        end = max(start, min(length, end))
        # Map unknown or out-of-vocabulary labels into the open-set bucket ("other").
        idx = label_map.get(label, label_map.get("other", -1))
        if idx < 0:
            continue
        labels[start:end] = idx
    return labels


def _canonical_eval_label_name(label: Any) -> str:
    raw = str(label or "").strip().lower().replace("-", "_")
    if not raw:
        return ""
    return PREDICTION_LABEL_ALIASES.get(raw, raw)


def _declared_postprocess_host_label(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(metadata, Mapping):
        return None
    candidates: List[Any] = [
        metadata.get("host_lang"),
        metadata.get("source_lang"),
        metadata.get("declared_lang"),
    ]
    source_meta = metadata.get("source_meta")
    if isinstance(source_meta, Mapping):
        candidates.extend(
            (
                source_meta.get("declared_type"),
                source_meta.get("source_lang"),
            )
        )
    for value in candidates:
        label = _canonical_eval_label_name(value)
        if label:
            return label
    return None


def _support_mask_from_text(content: str) -> np.ndarray:
    if not content:
        return np.zeros((0,), dtype=bool)
    return np.fromiter((ch not in _VISUAL_WHITESPACE_SET for ch in content), dtype=bool, count=len(content))


def _support_mask_from_bytes(byte_arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(byte_arr, dtype=np.int32).reshape(-1)
    if arr.size <= 0:
        return np.zeros((0,), dtype=bool)
    ignored = np.asarray(cfg.IGNORED_TRAINING_TOKEN_IDS, dtype=np.int32)
    return np.logical_not(np.isin(arr, ignored))


def _dense_label_runs(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    arr = np.asarray(labels, dtype=np.int32).reshape(-1)
    if arr.size <= 0:
        return []
    runs: List[Tuple[int, int, int]] = []
    start = 0
    cur = int(arr[0])
    for idx in range(1, int(arr.shape[0])):
        nxt = int(arr[idx])
        if nxt != cur:
            runs.append((start, idx, cur))
            start = idx
            cur = nxt
    runs.append((start, int(arr.shape[0]), cur))
    return runs


def _apply_relevant_content_postprocess_dense(
    labels: np.ndarray,
    *,
    support_mask: np.ndarray,
    label_to_idx: Mapping[str, int],
    declared_host_label: Optional[str] = None,
) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32).reshape(-1).copy()
    if arr.size <= 0:
        return arr
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    if support.shape[0] != arr.shape[0]:
        support = np.ones((arr.shape[0],), dtype=bool)

    min_support = int(RELEVANT_CONTENT_POSTPROCESS_MIN_TOKENS)
    declared_host = _canonical_eval_label_name(declared_host_label)

    def _idx(name: str) -> Optional[int]:
        value = label_to_idx.get(name)
        return int(value) if value is not None else None

    text_idx = _idx("text")
    if text_idx is not None and np.any(arr == text_idx):
        candidate_counts: List[Tuple[int, int]] = []
        for host_label in TEXT_HOST_POSTPROCESS_LABELS:
            host_idx = _idx(host_label)
            if host_idx is None:
                continue
            support_count = int(np.count_nonzero(np.logical_and(support, arr == host_idx)))
            if support_count >= min_support:
                candidate_counts.append((support_count, host_idx))
        if candidate_counts:
            best_support = max(count for count, _ in candidate_counts)
            best_targets = [target_idx for count, target_idx in candidate_counts if count == best_support]
            if len(best_targets) == 1:
                arr[arr == text_idx] = int(best_targets[0])

    runs = _dense_label_runs(arr)
    for inner_label, host_label in LOCAL_HOST_POSTPROCESS_RULES:
        inner_idx = _idx(inner_label)
        host_idx = _idx(host_label)
        if inner_idx is None or host_idx is None or inner_idx == host_idx:
            continue
        host_support = int(np.count_nonzero(np.logical_and(support, arr == host_idx)))
        if host_support < min_support:
            continue
        host_declared = declared_host == host_label
        changed = False
        for run_idx, (start, end, lbl_idx) in enumerate(runs):
            if lbl_idx != inner_idx:
                continue
            adjacent_support = 0
            if run_idx > 0 and runs[run_idx - 1][2] == host_idx:
                prev_start, prev_end, _ = runs[run_idx - 1]
                adjacent_support += int(np.count_nonzero(support[prev_start:prev_end]))
            if run_idx + 1 < len(runs) and runs[run_idx + 1][2] == host_idx:
                next_start, next_end, _ = runs[run_idx + 1]
                adjacent_support += int(np.count_nonzero(support[next_start:next_end]))
            if adjacent_support >= min_support or host_declared:
                arr[start:end] = host_idx
                changed = True
        if changed:
            runs = _dense_label_runs(arr)

    return arr


def _confusion_size(labels: Iterable[str]) -> Tuple[List[str], Dict[str, int]]:
    uniq = sorted({label for label in labels})
    mapping = {label: idx for idx, label in enumerate(uniq)}
    return uniq, mapping


def _needle_bucket_key(name: str) -> Optional[Tuple[int, int]]:
    match = _NEEDLE_BUCKET_RE.match(name)
    if not match:
        return None
    low = int(match.group(1))
    high_str = match.group(2)
    if high_str == "plus":
        high = int(1e9)
    else:
        high = int(high_str)
    return low, high


def _task_name_sort_key(name: str) -> Tuple[Any, ...]:
    bucket = _needle_bucket_key(name)
    if bucket is not None:
        low, high = bucket
        return ("needle", low, high)
    return (name, 0, 0)


def _markdown_stat_group() -> Dict[str, Any]:
    return {
        "count": 0,
        "detected_correct": 0,
        "detected_nontext": 0,
        "detected_text": 0,
        "detected_correct_iou": 0,
        "detected_nontext_iou": 0,
        "detected_text_iou": 0,
        "truth_chars": 0,
        "correct_chars": 0,
        "nontext_chars": 0,
        "text_chars": 0,
        "union_correct_chars": 0,
        "union_nontext_chars": 0,
        "union_text_chars": 0,
        "iou_sum": 0.0,
        "nontext_iou_sum": 0.0,
        "text_iou_sum": 0.0,
        "wrong_label_cases": 0,
        "wrong_label_fooled": 0,
    }


def _infer_markdown_host_lang(blocks: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Heuristically infer the dominant 'host' code language in a markdown doc.

    For monitor-derived markdown documents we only know that a document is
    markdown plus the per-span code labels. To recover host/other semantics
    similar to the synthetic markdown_mix builder, we treat the language with
    the largest total span (in characters) as the host language.
    """
    totals: Dict[str, int] = {}
    for block in blocks:
        lang = str(block.get("language") or "")
        if not lang:
            continue
        try:
            start = int(block.get("char_start", 0))
            end = int(block.get("char_end", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        totals[lang] = totals.get(lang, 0) + (end - start)
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


def _is_fenced_markdown_block(content: str, start: int, end: int) -> bool:
    """Best-effort detection of ``` fenced code blocks around a segment.

    The monitor-based markdown_mix dataset only records the code spans, not the
    surrounding fences. We reconstruct a 'wrapped' signal by looking for an
    opening fence line (```[lang]) immediately before the block and a closing
    fence (```) after it, both starting at the beginning of a line (ignoring
    whitespace). This intentionally errs on the side of requiring a clear
    fence structure rather than trying to be perfect.
    """
    try:
        length = len(content)
    except Exception:
        return False
    if length <= 0:
        return False

    start = max(0, min(length, int(start)))
    end = max(start, min(length, int(end)))
    if end <= start:
        return False

    # Look for an opening fence somewhere before the block.
    fence_idx = content.rfind("```", 0, start)
    if fence_idx == -1:
        return False
    fence_line_start = content.rfind("\n", 0, fence_idx)
    if fence_line_start == -1:
        fence_line_start = 0
    else:
        fence_line_start += 1
    fence_line_end = content.find("\n", fence_idx, start)
    if fence_line_end == -1 or fence_line_end > start:
        return False
    fence_line = content[fence_line_start:fence_line_end]
    # Require the fence to be the first non-whitespace token on the line.
    stripped = fence_line.lstrip()
    if not stripped.startswith("```"):
        return False

    # Look for a closing fence after the block.
    closing_idx = content.find("```", end)
    if closing_idx == -1:
        return False
    close_line_start = content.rfind("\n", 0, closing_idx)
    if close_line_start == -1:
        close_line_start = 0
    else:
        close_line_start += 1
    close_prefix = content[close_line_start:closing_idx]
    if any(not ch.isspace() for ch in close_prefix):
        return False

    return True


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator:
        try:
            value = float(numerator) / float(denominator)
        except Exception:
            return None
        if math.isfinite(value):
            return value
    return None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except Exception:
        return None
    if math.isfinite(value):
        return value
    return None


def _metrics_payload_from_confusion(
    conf_mat: np.ndarray,
    *,
    id2label: Mapping[int, str],
    num_classes: int,
    ignore_class: Optional[int] = None,
) -> Dict[str, Any]:
    per_class, aggregates = compute_metrics_from_confusion(
        conf_mat,
        int(num_classes),
        ignore_class,
    )
    rows: List[Dict[str, Any]] = [
        {
            "label": "ALL (agg)",
            "support": None,
            "acc": float(aggregates["micro"]["acc"]),
            "precision": float(aggregates["macro"]["precision"]),
            "recall": float(aggregates["macro"]["recall"]),
            "f1": float(aggregates["macro"]["f1"]),
        }
    ]
    by_label: Dict[str, Dict[str, Any]] = {}
    support = per_class["support"]
    acc = per_class["acc"]
    prec = per_class["precision"]
    rec = per_class["recall"]
    f1 = per_class["f1"]
    for cid in range(int(num_classes)):
        if ignore_class is not None and cid == int(ignore_class):
            continue
        if int(support[cid]) <= 0:
            continue
        label = str(id2label.get(cid, str(cid)))
        row = {
            "label": label,
            "support": int(support[cid]),
            "acc": float(acc[cid]),
            "precision": float(prec[cid]),
            "recall": float(rec[cid]),
            "f1": float(f1[cid]),
        }
        rows.append(row)
        by_label[label] = row
    return {
        "aggregates": {
            "micro_acc": float(aggregates["micro"]["acc"]),
            "macro_precision": float(aggregates["macro"]["precision"]),
            "macro_recall": float(aggregates["macro"]["recall"]),
            "macro_f1": float(aggregates["macro"]["f1"]),
            "weighted_f1": float(aggregates["weighted"]["f1"]),
        },
        "rows": rows,
        "by_label": by_label,
    }


def _render_training_style_metrics_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| label | support | acc | prec | recall | f1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        support = row.get("support")
        support_str = "" if support is None else str(int(support))
        lines.append(
            "| {label} | {support} | {acc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                label=str(row.get("label", "")),
                support=support_str,
                acc=float(row.get("acc", 0.0)),
                precision=float(row.get("precision", 0.0)),
                recall=float(row.get("recall", 0.0)),
                f1=float(row.get("f1", 0.0)),
            )
        )
    return "\n".join(lines)


def _build_text_like_binary_payload(conf_mat: np.ndarray) -> Dict[str, Any]:
    payload = _metrics_payload_from_confusion(
        conf_mat,
        id2label=_TEXT_LIKE_ID2LABEL,
        num_classes=2,
        ignore_class=None,
    )
    truth_negative = int(conf_mat[0, :].sum())
    truth_positive = int(conf_mat[1, :].sum())
    predicted_negative = int(conf_mat[:, 0].sum())
    predicted_positive = int(conf_mat[:, 1].sum())
    payload.update(
        {
            "positive_labels": list(TEXT_LIKE_POSITIVE_LABELS),
            "counts": {
                "tn": int(conf_mat[0, 0]),
                "fp": int(conf_mat[0, 1]),
                "fn": int(conf_mat[1, 0]),
                "tp": int(conf_mat[1, 1]),
                "truth_positive": truth_positive,
                "truth_negative": truth_negative,
                "predicted_positive": predicted_positive,
                "predicted_negative": predicted_negative,
                "total": int(conf_mat.sum()),
            },
        }
    )
    return payload


def _needle_boundary_metrics_payload(
    boundary_stats: Mapping[str, Any],
    *,
    label_names: Sequence[str],
) -> Optional[Dict[str, Any]]:
    raw_confusion = boundary_stats.get("confusion")
    if raw_confusion is None:
        return None
    confusion = np.asarray(raw_confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1] or confusion.size == 0:
        return None
    if int(confusion.sum()) <= 0:
        return None
    id2label = {idx: str(label) for idx, label in enumerate(label_names[: int(confusion.shape[0])])}
    payload = _metrics_payload_from_confusion(
        confusion,
        id2label=id2label,
        num_classes=int(confusion.shape[0]),
        ignore_class=None,
    )
    payload["window_radius_tokens"] = int(
        boundary_stats.get("window_radius_tokens", NEEDLE_BOUNDARY_WINDOW_TOKENS)
    )
    payload["support_chars"] = int(confusion.sum())
    payload["samples"] = int(boundary_stats.get("samples", 0))
    return payload


def _sequence_boundary_metrics_payload(
    boundary_stats: Mapping[str, Any],
    *,
    label_names: Sequence[str],
) -> Optional[Dict[str, Any]]:
    payload = _needle_boundary_metrics_payload(boundary_stats, label_names=label_names)
    if payload is None:
        return None
    payload["boundaries"] = int(boundary_stats.get("boundaries", 0))
    return payload


def _monitor_boundary_metrics_payload(
    boundary_stats: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    raw_confusion = boundary_stats.get("confusion")
    if raw_confusion is None:
        return None
    confusion = np.asarray(raw_confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1] or confusion.size == 0:
        return None
    if int(confusion.sum()) <= 0:
        return None
    payload = _metrics_payload_from_confusion(
        confusion,
        id2label=_monitor_b_id2label(),
        num_classes=int(confusion.shape[0]),
        ignore_class=None,
    )
    payload["window_radius_tokens"] = int(
        boundary_stats.get("window_radius_tokens", NEEDLE_BOUNDARY_WINDOW_TOKENS)
    )
    payload["support_chars"] = int(confusion.sum())
    payload["samples"] = int(boundary_stats.get("samples", 0))
    payload["boundaries"] = int(boundary_stats.get("boundaries", 0))
    return payload


def _append_text_like_binary_lines(lines: List[str], payload: Mapping[str, Any]) -> None:
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or not rows:
        return
    positive_labels = payload.get("positive_labels", TEXT_LIKE_POSITIVE_LABELS)
    labels_str = ", ".join(f"`{label}`" for label in positive_labels)
    lines.extend(
        [
            "",
            f"Binary text-like metrics ({labels_str} count as positive):",
            "",
            _render_training_style_metrics_table(rows),
        ]
    )


def _monitor_b_id2label() -> Dict[int, str]:
    labels = {int(idx): str(name) for idx, name in cfg.ID2LANG.items()}
    other_id = getattr(cfg, "OTHER_CLASS_INDEX", None)
    if other_id is not None:
        labels[int(other_id)] = "other"
    return labels


def _evaluate_full_monitor_b(
    monitor_root: Path,
    runner: "SegmenterRunner",
    *,
    other_threshold: float = 0.0,
    excluded_truth_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    monitor_data = load_monitor_memmaps(Path(monitor_root).resolve())
    files = monitor_data["files"]
    segments = monitor_data["segments"]
    contents = monitor_data["contents"]

    arch = str(getattr(runner, "arch", "unet1d")).lower().strip()
    use_mamba_path = arch == "mamba"
    inference_mode = (
        "full_file_auto_with_stream_fallback"
        if use_mamba_path
        else ("sliding_window_magika_rawdl" if arch == "magika" else "sliding_window_legacy")
    )

    start_time = time.perf_counter()
    other_id = getattr(cfg, "OTHER_CLASS_INDEX", None)
    excluded_truth_labels_set = _normalize_truth_label_exclusions(excluded_truth_labels)
    excluded_truth_label_ids = {
        int(cfg.LANG2ID[label])
        for label in excluded_truth_labels_set
        if label in cfg.LANG2ID
    }
    truth_label_exclusions = _truth_label_exclusion_payload(
        requested_total=int(len(files)),
        excluded_labels=excluded_truth_labels,
        unit_name="files",
    )
    use_other_threshold = (
        other_id is not None
        and other_threshold is not None
        and float(other_threshold) > 0.0
    )
    num_monitor_classes = (
        (int(other_id) + 1)
        if other_id is not None
        else int(cfg.NUM_CLASSES)
    )
    confusion = np.zeros((num_monitor_classes, num_monitor_classes), dtype=np.int64)
    boundary_region = {
        "window_radius_tokens": NEEDLE_BOUNDARY_WINDOW_TOKENS,
        "samples": 0,
        "boundaries": 0,
        "confusion": np.zeros((num_monitor_classes, num_monitor_classes), dtype=np.int64),
    }
    files_used = 0
    skipped = 0
    skipped_for_truth_label = 0
    evaluated_bytes = 0
    raw_bytes = 0
    low_confidence_bytes = 0
    truth_other_bytes = 0

    for file_idx, row in enumerate(files):
        byte_len = int(row["byte_len"])
        if byte_len <= 0:
            skipped += 1
            continue
        byte_start = int(row["byte_start"])
        seg_start = int(row["seg_start"])
        seg_count = int(row["seg_count"])

        file_bytes = np.asarray(
            contents[byte_start : byte_start + byte_len],
            dtype=np.uint8,
        )
        support_mask = _support_mask_from_bytes(file_bytes)
        declared_host_label = _canonical_eval_label_name(
            cfg.ID2LANG.get(int(row["type_id"]), "other")
            if int(row["type_id"]) in cfg.ID2LANG
            else ("other" if other_id is not None and int(row["type_id"]) == int(other_id) else "")
        )
        truth = np.full((byte_len,), cfg.PAD_ID, dtype=np.uint8)
        file_truth_label_names: Set[str] = set()
        file_has_excluded_truth_label = False
        for seg in segments[seg_start : seg_start + seg_count]:
            start = int(seg["start"])
            end = int(seg["end"])
            if end <= start:
                continue
            seg_label = int(seg["label"])
            truth[start:end] = seg_label
            if seg_label in excluded_truth_label_ids:
                file_has_excluded_truth_label = True
            label_name = cfg.ID2LANG.get(seg_label)
            if label_name is not None:
                file_truth_label_names.add(str(label_name))

        if excluded_truth_label_ids and file_has_excluded_truth_label:
            skipped += 1
            skipped_for_truth_label += 1
            matched_labels = tuple(
                sorted(file_truth_label_names.intersection(excluded_truth_labels_set))
            )
            _record_truth_label_exclusion_skip(truth_label_exclusions, matched_labels)
            continue

        if use_mamba_path:
            pred, probs = runner._segment_bytes(file_bytes)
        else:
            pred, probs = runner._segment_bytes(file_bytes) if arch == "magika" else runner._segment_bytes_legacy(file_bytes)
        pred = np.asarray(pred, dtype=np.int32).reshape(-1)
        probs = np.asarray(probs, dtype=np.float32)
        if int(pred.shape[0]) != byte_len:
            raise RuntimeError(
                f"Monitor prediction length mismatch for file {file_idx}: "
                f"expected {byte_len}, got {int(pred.shape[0])}"
            )
        if probs.ndim != 2 or int(probs.shape[0]) != byte_len:
            raise RuntimeError(
                f"Monitor probability length mismatch for file {file_idx}: "
                f"expected ({byte_len}, C), got {tuple(int(dim) for dim in probs.shape)}"
            )

        mask = _valid_metric_mask(truth, file_bytes.astype(np.int32, copy=False))
        truth_i32 = truth.astype(np.int32, copy=False)
        pred_i32 = pred.astype(np.int32, copy=True)
        if other_id is not None:
            pred_i32[(pred_i32 < 0) | (pred_i32 >= int(cfg.NUM_CLASSES))] = int(other_id)
            truth_other_bytes += int(np.sum(mask & (truth_i32 == int(other_id))))
            if use_other_threshold:
                max_prob = probs.max(axis=-1)
                low_conf_mask = mask & (max_prob < float(other_threshold))
                if np.any(low_conf_mask):
                    pred_i32[low_conf_mask] = int(other_id)
                    low_confidence_bytes += int(low_conf_mask.sum())
            core_mask = mask & (truth_i32 <= int(other_id))
        else:
            core_mask = mask & (truth_i32 < int(cfg.NUM_CLASSES))
        truth_i32 = _apply_relevant_content_postprocess_dense(
            truth_i32,
            support_mask=support_mask,
            label_to_idx=cfg.LANG2ID,
            declared_host_label=declared_host_label,
        )
        pred_i32 = _apply_relevant_content_postprocess_dense(
            pred_i32,
            support_mask=support_mask,
            label_to_idx=cfg.LANG2ID,
            declared_host_label=declared_host_label,
        )
        if core_mask.any():
            accumulate_confusion(
                confusion,
                truth_i32[core_mask],
                pred_i32[core_mask],
            )
            evaluated_bytes += int(core_mask.sum())
            evaluated_positions = np.flatnonzero(core_mask)
            evaluated_truth = truth_i32[core_mask]
            if evaluated_positions.size > 1:
                transition_mask = evaluated_truth[1:] != evaluated_truth[:-1]
                if np.any(transition_mask):
                    boundary_positions = evaluated_positions[1:][transition_mask].tolist()
                    boundary_mask_valid = _boundary_positions_valid_mask(
                        byte_len,
                        core_mask,
                        boundary_positions,
                        radius_tokens=int(
                            boundary_region.get(
                                "window_radius_tokens",
                                NEEDLE_BOUNDARY_WINDOW_TOKENS,
                            )
                        ),
                    )
                    if boundary_mask_valid is not None:
                        boundary_region["samples"] += 1
                        boundary_region["boundaries"] += len(boundary_positions)
                        boundary_truth = evaluated_truth[boundary_mask_valid]
                        boundary_pred = pred_i32[core_mask][boundary_mask_valid]
                        boundary_valid_pred = np.logical_and(
                            boundary_pred >= 0,
                            boundary_pred < int(confusion.shape[0]),
                        )
                        if np.any(boundary_valid_pred):
                            np.add.at(
                                boundary_region["confusion"],
                                (
                                    boundary_truth[boundary_valid_pred],
                                    boundary_pred[boundary_valid_pred],
                                ),
                                1,
                            )

        files_used += 1
        raw_bytes += int(byte_len)

    metrics_payload = _metrics_payload_from_confusion(
        confusion,
        id2label=_monitor_b_id2label(),
        num_classes=int(confusion.shape[0]),
        ignore_class=None,
    )
    metrics_payload.update(
        {
            "_confusion_matrix": confusion.tolist(),
            "boundary_region": _monitor_boundary_metrics_payload(boundary_region),
            "root": str(Path(monitor_root).resolve()),
            "arch": arch,
            "inference_mode": inference_mode,
            "open_set": True,
            "other_threshold": float(other_threshold),
            "files_total": int(len(files)),
            "files_used": int(files_used),
            "skipped": int(skipped),
            "skipped_for_truth_label": int(skipped_for_truth_label),
            "evaluated_bytes": int(evaluated_bytes),
            "raw_bytes": int(raw_bytes),
            "truth_other_bytes": int(truth_other_bytes),
            "low_confidence_bytes_routed_to_other": int(low_confidence_bytes),
            "elapsed_seconds": float(time.perf_counter() - start_time),
        }
    )
    if truth_label_exclusions is not None:
        truth_label_exclusions["counted"] = int(files_used)
        metrics_payload["truth_label_exclusions"] = truth_label_exclusions
    return metrics_payload


def _default_monitor_b_root() -> Path:
    return (REPO_ROOT / "downloader" / "monitor_preprocessed_b").resolve()


def _configure_evaluation_mode(args) -> Dict[str, Any]:
    default_eval_root = (REPO_ROOT / "evaluation" / "data").resolve()
    # New default: the thesis-aligned dense/test set built from the Gemini-Pro
    # labels under evaluation/test/. The previous data_b root was sampled from
    # the monitor split (which is consumed by active learning at training time)
    # and is therefore NOT a true held-out test set.
    thesis_test_root = (REPO_ROOT / "evaluation" / "test").resolve()
    legacy_monitor_b_root = (REPO_ROOT / "evaluation" / "data_b").resolve()
    requested_root = Path(args.data_root).resolve()
    fine_tuned_mode = not bool(getattr(args, "non_fine_tuned", False))
    messages: List[str] = []

    if bool(getattr(args, "fine_tuned", False)):
        messages.append(
            "⚠️  --fine-tuned is deprecated; fine-tuned evaluation is now the default."
        )
    if fine_tuned_mode:
        # If the caller passed the historical default (evaluation/data) AND the
        # new thesis test root exists, redirect to it. If the new root is not
        # present yet, fall back to the legacy data_b root with a loud warning
        # so the user knows they are running on a monitor-derived set.
        if requested_root == default_eval_root:
            if thesis_test_root.exists():
                args.data_root = str(thesis_test_root)
                messages.append(
                    f"ℹ️  Fine-tuned mode (default): using thesis dense/test root {args.data_root}"
                )
            else:
                args.data_root = str(legacy_monitor_b_root)
                messages.append(
                    "⚠️  Thesis dense/test root evaluation/test/ not found; falling "
                    "back to evaluation/data_b (monitor-derived; NOT a true held-out "
                    "test set). Re-run build_thesis_test_set.py to produce the new root."
                )
        else:
            args.data_root = str(requested_root)
            messages.append(
                f"ℹ️  Fine-tuned mode (default): using custom evaluation data root {args.data_root}"
            )
    else:
        args.data_root = str(requested_root)
        messages.append(
            "⚠️  Non-fine-tuned evaluation mode requested via --non-fine-tuned. "
            "This is the legacy/non-default path."
        )

    setattr(args, "fine_tuned_mode", bool(fine_tuned_mode))
    return {
        "fine_tuned_mode": bool(fine_tuned_mode),
        "messages": messages,
        "default_eval_root": str(default_eval_root),
        "thesis_test_root": str(thesis_test_root),
        "legacy_monitor_b_root": str(legacy_monitor_b_root),
    }


def _region_mask_valid(
    content_len: int,
    valid_mask: np.ndarray,
    start: Any,
    end: Any,
) -> Optional[np.ndarray]:
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None
    if content_len <= 0:
        return None
    start_i = max(0, min(content_len, start_i))
    end_i = max(start_i, min(content_len, end_i))
    if end_i <= start_i:
        return None
    mask = np.zeros((content_len,), dtype=bool)
    mask[start_i:end_i] = True
    region_valid = mask[valid_mask]
    if int(region_valid.sum()) <= 0:
        return None
    return region_valid


def _boundary_positions_valid_mask(
    content_len: int,
    valid_mask: np.ndarray,
    positions: Sequence[Any],
    *,
    radius_tokens: int,
) -> Optional[np.ndarray]:
    if content_len <= 0:
        return None
    radius = max(0, int(radius_tokens))
    if radius <= 0:
        return None
    mask = np.zeros((content_len,), dtype=bool)
    had_window = False
    for pos in positions:
        try:
            pos_i = int(pos)
        except (TypeError, ValueError):
            continue
        pos_i = max(0, min(content_len, pos_i))
        start_i = max(0, pos_i - radius)
        end_i = min(content_len, pos_i + radius)
        if end_i <= start_i:
            continue
        mask[start_i:end_i] = True
        had_window = True
    if not had_window:
        return None
    region_valid = mask[valid_mask]
    if int(region_valid.sum()) <= 0:
        return None
    return region_valid


def _boundary_region_valid_mask(
    content_len: int,
    valid_mask: np.ndarray,
    start: Any,
    end: Any,
    *,
    radius_tokens: int,
) -> Optional[np.ndarray]:
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None
    start_i = max(0, min(content_len, start_i))
    end_i = max(start_i, min(content_len, end_i))
    if end_i <= start_i:
        return None
    return _boundary_positions_valid_mask(
        content_len,
        valid_mask,
        (start_i, end_i),
        radius_tokens=radius_tokens,
    )


def _exact_region_stats(
    truth_valid: np.ndarray,
    pred_valid: np.ndarray,
    region_valid_mask: np.ndarray,
) -> Dict[str, Any]:
    truth_chars = int(region_valid_mask.sum())
    if truth_chars <= 0:
        return {
            "truth_chars": 0,
            "correct_chars": 0,
            "coverage": 0.0,
            "hit": False,
        }
    correct_chars = int(np.logical_and(region_valid_mask, pred_valid == truth_valid).sum())
    coverage = correct_chars / truth_chars if truth_chars > 0 else 0.0
    return {
        "truth_chars": truth_chars,
        "correct_chars": correct_chars,
        "coverage": coverage,
        "hit": False,
    }


def _region_non_host_stats(
    pred_valid: np.ndarray,
    region_valid_mask: np.ndarray,
    host_idx: Optional[int],
) -> Dict[str, Any]:
    truth_chars = int(region_valid_mask.sum())
    if truth_chars <= 0:
        return {
            "truth_chars": 0,
            "matched_chars": 0,
            "coverage": 0.0,
            "hit": False,
        }
    if host_idx is not None:
        matched_chars = int(np.logical_and(region_valid_mask, pred_valid != host_idx).sum())
    else:
        matched_chars = int(region_valid_mask.sum())
    coverage = matched_chars / truth_chars if truth_chars > 0 else 0.0
    return {
        "truth_chars": truth_chars,
        "matched_chars": matched_chars,
        "coverage": coverage,
        "hit": False,
    }


def _region_text_stats(
    pred_valid: np.ndarray,
    region_valid_mask: np.ndarray,
    *,
    text_idx: Optional[int],
) -> Tuple[int, int]:
    truth_chars = int(region_valid_mask.sum())
    if truth_chars <= 0:
        return 0, 0
    if text_idx is None:
        return truth_chars, 0
    nontext_chars = int(np.logical_and(region_valid_mask, pred_valid != text_idx).sum())
    text_chars = int(np.logical_and(region_valid_mask, pred_valid == text_idx).sum())
    return nontext_chars, text_chars


def _payload_soft_stat_group() -> Dict[str, Any]:
    return {
        "count": 0,
        "detected": 0,
        "below_threshold": 0,
        "truth_chars": 0,
        "matched_chars": 0,
        "coverage_sum": 0.0,
        "coverage_count": 0,
    }


def _build_model_eval_idx_lookup(label_to_idx: Mapping[str, int]) -> np.ndarray:
    size = max(int(getattr(cfg, "NUM_CLASSES", 0)), max(cfg.ID2LANG.keys(), default=-1) + 1)
    lookup = np.full((max(0, size),), -1, dtype=np.int32)
    for pred_id, label_name in cfg.ID2LANG.items():
        canonical = PREDICTION_LABEL_ALIASES.get(label_name, label_name)
        if canonical in label_to_idx and 0 <= int(pred_id) < lookup.size:
            lookup[int(pred_id)] = int(label_to_idx[canonical])
    return lookup


def _model_eval_idx_view(lookup: np.ndarray, size: int) -> np.ndarray:
    if size <= lookup.size:
        return lookup[:size]
    pad = np.full((size - lookup.size,), -1, dtype=np.int32)
    return np.concatenate([lookup, pad], axis=0)


def _update_payload_soft_stat(
    entry: Dict[str, Any],
    *,
    matched_chars: int,
    truth_chars: int,
    coverage_threshold: float,
) -> None:
    coverage = (matched_chars / truth_chars) if truth_chars > 0 else 0.0
    entry["count"] += 1
    entry["truth_chars"] += int(truth_chars)
    entry["matched_chars"] += int(matched_chars)
    entry["coverage_sum"] += float(coverage)
    entry["coverage_count"] += 1
    if coverage >= coverage_threshold:
        entry["detected"] += 1
    else:
        entry["below_threshold"] += 1


def _top_confusions(
    confusion: np.ndarray,
    label_names: Sequence[str],
    *,
    top_k: int = 3,
    exclude_labels: Optional[Sequence[str]] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    exclude_set = set(exclude_labels or [])
    result: Dict[str, List[Tuple[str, float]]] = {}
    for true_idx, true_label in enumerate(label_names):
        if true_label in exclude_set:
            continue
        row = confusion[true_idx]
        total = float(row.sum())
        if total <= 0:
            continue
        pairs: List[Tuple[float, str]] = []
        for pred_idx, pred_label in enumerate(label_names):
            if pred_label == true_label:
                continue
            count = float(row[pred_idx])
            if count <= 0:
                continue
            pairs.append((count / total, pred_label))
        pairs.sort(key=lambda x: x[0], reverse=True)
        result[true_label] = pairs[:top_k]
    return result


def _report_truth_label_exclusions(task_name: str) -> Tuple[str, ...]:
    return REPORT_ONLY_TRUTH_LABEL_EXCLUSIONS.get(task_name, ())


@dataclass
class TaskMetrics:
    name: str
    description: str
    samples: int
    total_chars: int
    confusion: np.ndarray
    label_names: List[str]
    per_label_counts: Dict[str, int] = field(default_factory=dict)
    per_label_correct: Dict[str, int] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def overall_accuracy(self) -> float:
        total = self.confusion.sum()
        return float(np.trace(self.confusion) / total) if total else 0.0

    def per_label_accuracy(self) -> Dict[str, float]:
        rows = {}
        for idx, label in enumerate(self.label_names):
            support = float(self.confusion[idx].sum())
            if support <= 0:
                rows[label] = float("nan")
                continue
            rows[label] = float(self.confusion[idx, idx] / support)
        return rows


def _analyze_pure_fragments(metrics: TaskMetrics) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Analyze pure fragment classification results (per-label character accuracy).

    Returns:
        Tuple containing:
        - Dict mapping label -> accuracy percentage (chars of this label predicted correctly)
        - Dict mapping label -> (incorrect_label -> percentage) for misclassifications
    """
    accuracy_scores: Dict[str, float] = {}
    misclassification_details: Dict[str, Dict[str, float]] = {}

    for idx, true_label in enumerate(metrics.label_names):
        if true_label == "other":
            continue

        true_total = metrics.confusion[idx].sum()
        if true_total == 0:
            continue

        correct = metrics.confusion[idx, idx]
        acc_pct = (correct / true_total) * 100 if true_total > 0 else 0.0
        accuracy_scores[true_label] = acc_pct

        if acc_pct < 100.0:
            # Get misclassification distribution
            mistakes: Dict[str, float] = {}
            for pred_idx, pred_label in enumerate(metrics.label_names):
                if pred_idx == idx or pred_label == "other":
                    continue
                count = metrics.confusion[idx, pred_idx]
                if count > 0:
                    mistakes[pred_label] = (count / true_total) * 100
            mistakes = dict(sorted(mistakes.items(), key=lambda x: x[1], reverse=True))
            misclassification_details[true_label] = mistakes

    return accuracy_scores, misclassification_details

def evaluate_task(
    name: str,
    description: str,
    dataset: hfds.Dataset,
    runner: SegmenterRunner,
    *,
    min_run_chars: int,
    other_threshold: float = 0.0,
    postprocess_profile: str = POSTPROCESS_PROFILE_OFF,
    postprocess_min_run_chars: int = 5,
    postprocess_boundary_snap_max_shift: int = 2,
    excluded_truth_labels: Optional[Sequence[str]] = None,
) -> TaskMetrics:
    total_samples = len(dataset)
    postprocess_profile_norm = str(postprocess_profile or POSTPROCESS_PROFILE_OFF).lower().strip()
    excluded_truth_label_set = _normalize_truth_label_exclusions(excluded_truth_labels)
    truth_label_exclusions = _truth_label_exclusion_payload(
        requested_total=total_samples,
        excluded_labels=excluded_truth_labels,
        unit_name="samples",
    )
    # Always include an 'other' bucket for open-set handling.
    label_candidates = {"other"}
    for example in dataset:
        for seg in _normalize_segments(example["segments"]):
            if str(seg["label"]).strip() in excluded_truth_label_set:
                continue
            label_candidates.add(seg["label"])
    for alias_target in PREDICTION_LABEL_ALIASES.values():
        label_candidates.add(alias_target)
    label_candidates.add("text")
    for host_label in TEXT_HOST_POSTPROCESS_LABELS:
        label_candidates.add(host_label)
    for inner_label, host_label in LOCAL_HOST_POSTPROCESS_RULES:
        label_candidates.add(inner_label)
        label_candidates.add(host_label)
    label_names, label_to_idx = _confusion_size(label_candidates)
    confusion = np.zeros((len(label_names), len(label_names)), dtype=np.int64)

    total_chars = 0
    per_label_counts: Dict[str, int] = {label: 0 for label in label_names}
    per_label_correct: Dict[str, int] = {label: 0 for label in label_names}
    extra_payload: Dict[str, Any] = {}
    counted_samples = 0
    if truth_label_exclusions is not None:
        extra_payload["truth_label_exclusions"] = truth_label_exclusions
    postprocess_stats: Optional[Dict[str, Any]] = None
    if postprocess_profile_norm == POSTPROCESS_PROFILE_THESIS:
        postprocess_stats = {
            "profile": POSTPROCESS_PROFILE_THESIS,
            "settings": {
                "other_threshold": float(other_threshold or 0.0),
                "min_run_chars": int(postprocess_min_run_chars),
                "boundary_snap_max_shift": int(postprocess_boundary_snap_max_shift),
                "stages": [
                    "whitespace_relabel",
                    "other_gating",
                    "boundary_snap",
                    "min_run",
                ],
                "excluded_viewer_stages": [
                    "markdown_structure_fill",
                    "paired_delimiter_fill",
                    "local_host_fill",
                    "newline_snap",
                ],
            },
            "samples": 0,
            "processed_chars": 0,
            "other_thresholded_chars": 0,
            "boundary_snap_changed_chars": 0,
            "min_run_changed_chars": 0,
            "changed_chars": 0,
        }
        extra_payload["postprocess"] = postprocess_stats

    pure_stats: Optional[Dict[str, int]] = None
    if name in ("pure_fragments", "near_pure"):
        pure_stats = {
            "total": 0,
            "perfect": 0,
            "within_threshold": 0,
            "threshold": 0.5,
            # Per-host-language file-level purity stats:
            # host_byte_purity[lang] = {"files": N, "pure_files": M}
            "host_byte_purity": {},
        }
        extra_payload["pure_fragments_purity"] = pure_stats

    needle_stats: Optional[Dict[str, Any]] = None
    if name.startswith("needle_"):
        needle_stats = {
            "threshold": NEEDLE_COVERAGE_THRESHOLD,
            "total": 0,
            "detected": 0,
            "below_threshold": 0,
            "correct_chars": 0,
            "needle_chars": 0,
            "iou_sum": 0.0,
            "coverage_hits": 0,
            "coverage_sum": 0.0,
            "coverage_count": 0,
            "by_lang": {},
            "any_detection": {
                "count": 0,
                "detected": 0,
                "below_threshold": 0,
                "truth_chars": 0,
                "correct_chars": 0,
                "iou_sum": 0.0,
                "coverage_hits": 0,
                "coverage_sum": 0.0,
                "coverage_count": 0,
            },
            "any_by_lang": {},
            "boundary_region": {
                "window_radius_tokens": NEEDLE_BOUNDARY_WINDOW_TOKENS,
                "samples": 0,
                "confusion": np.zeros((len(label_names), len(label_names)), dtype=np.int64),
            },
        }
        extra_payload["needle_detection"] = needle_stats

    markdown_stats: Optional[Dict[str, Any]] = None
    markdown_stats_key: Optional[str] = None
    text_like_positive_ids: Optional[np.ndarray] = None
    text_like_confusion: Optional[np.ndarray] = None
    if name == "markdown_mix":
        markdown_stats = {
            "threshold": MARKDOWN_IOU_THRESHOLD,
            "roles": {
                "host": {"wrapped": _markdown_stat_group(), "plain": _markdown_stat_group()},
                "other": {"wrapped": _markdown_stat_group(), "plain": _markdown_stat_group()},
            },
            "overall": {"wrapped": _markdown_stat_group(), "plain": _markdown_stat_group()},
            "wrong_label": {"cases": 0, "fooled": 0},
            "boundary_region": {
                "window_radius_tokens": NEEDLE_BOUNDARY_WINDOW_TOKENS,
                "samples": 0,
                "confusion": np.zeros((len(label_names), len(label_names)), dtype=np.int64),
            },
            "inline": {
                "threshold": MARKDOWN_IOU_THRESHOLD,
                "count": 0,
                "correct_hits": 0,
                "nontext_hits": 0,
                "text_hits": 0,
                "truth_chars": 0,
                "correct_chars": 0,
                "nontext_chars": 0,
                "text_chars": 0,
                "union_correct_chars": 0,
                "union_nontext_chars": 0,
                "union_text_chars": 0,
                "by_wrapper": {},
                "correct_iou_hits": 0,
                "nontext_iou_hits": 0,
                "text_iou_hits": 0,
                "correct_iou_sum": 0.0,
                "nontext_iou_sum": 0.0,
                "text_iou_sum": 0.0,
            },
        }
        markdown_stats_key = "markdown_segments"
    elif name == "restructuredtext_mix":
        markdown_stats = {
            "threshold": MARKDOWN_IOU_THRESHOLD,
            "roles": {
                "host": {"wrapped": _markdown_stat_group(), "plain": _markdown_stat_group()},
                "other": {"wrapped": _markdown_stat_group(), "plain": _markdown_stat_group()},
            },
            "overall": {"wrapped": _markdown_stat_group(), "plain": _markdown_stat_group()},
            "wrong_label": {"cases": 0, "fooled": 0},
            "inline": {
                "threshold": MARKDOWN_IOU_THRESHOLD,
                "count": 0,
                "correct_hits": 0,
                "nontext_hits": 0,
                "text_hits": 0,
                "truth_chars": 0,
                "correct_chars": 0,
                "nontext_chars": 0,
                "text_chars": 0,
                "union_correct_chars": 0,
                "union_nontext_chars": 0,
                "union_text_chars": 0,
                "by_wrapper": {},
                "correct_iou_hits": 0,
                "nontext_iou_hits": 0,
                "text_iou_hits": 0,
                "correct_iou_sum": 0.0,
                "nontext_iou_sum": 0.0,
                "text_iou_sum": 0.0,
            },
        }
        markdown_stats_key = "restructuredtext_segments"
    if markdown_stats is not None and markdown_stats_key:
        extra_payload[markdown_stats_key] = markdown_stats
        text_like_positive_ids = np.asarray(
            [
                int(label_to_idx[label])
                for label in TEXT_LIKE_POSITIVE_LABELS
                if label in label_to_idx
            ],
            dtype=np.int32,
        )
        text_like_confusion = np.zeros((2, 2), dtype=np.int64)

    payload_stats: Optional[Dict[str, Any]] = None
    if name == "mal_injection":
        payload_stats = {
            "threshold": PAYLOAD_IOU_THRESHOLD,
            "by_lang": {},
            "any_detection": {
                "count": 0,
                "detected": 0,
                "below_threshold": 0,
                "truth_chars": 0,
                "correct_chars": 0,
                "iou_sum": 0.0,
                "coverage_hits": 0,
                "coverage_sum": 0.0,
                "coverage_count": 0,
            },
            "any_by_lang": {},
            "soft_detection": {
                "prob_threshold": PAYLOAD_SOFT_PROB_THRESHOLD,
                "coverage_threshold": PAYLOAD_SOFT_COVERAGE_THRESHOLD,
                "correct_detection": _payload_soft_stat_group(),
                "any_detection": _payload_soft_stat_group(),
                "correct_by_lang": {},
                "any_by_lang": {},
            },
        }
        extra_payload["mal_payload_detection"] = payload_stats

    payload_model_eval_idx = _build_model_eval_idx_lookup(label_to_idx) if payload_stats is not None else np.zeros((0,), dtype=np.int32)

    realistic_stats: Optional[Dict[str, Any]] = None
    if name == "realistic":
        realistic_stats = {
            "boundary_region": {
                "window_radius_tokens": NEEDLE_BOUNDARY_WINDOW_TOKENS,
                "samples": 0,
                "boundaries": 0,
                "confusion": np.zeros((len(label_names), len(label_names)), dtype=np.int64),
            },
        }
        extra_payload["realistic_stats"] = realistic_stats

    sequence_stats: Optional[Dict[str, Any]] = None
    if name == "sequence_pair":
        sequence_stats = {
            "segments": {
                "first": {"correct": 0, "total": 0},
                "second": {"correct": 0, "total": 0},
            },
            "boundary_region": {
                "window_radius_tokens": NEEDLE_BOUNDARY_WINDOW_TOKENS,
                "samples": 0,
                "boundaries": 0,
                "confusion": np.zeros((len(label_names), len(label_names)), dtype=np.int64),
            },
        }
        extra_payload["sequence_purity"] = sequence_stats
    elif name == "sequence_triplet":
        sequence_stats = {
            "segments": {
                "first": {"correct": 0, "total": 0},
                "second": {"correct": 0, "total": 0},
                "third": {"correct": 0, "total": 0},
            },
            "boundary_region": {
                "window_radius_tokens": NEEDLE_BOUNDARY_WINDOW_TOKENS,
                "samples": 0,
                "boundaries": 0,
                "confusion": np.zeros((len(label_names), len(label_names)), dtype=np.int64),
            },
        }
        extra_payload["sequence_purity"] = sequence_stats

    start_time = time.perf_counter()
    log_interval = getattr(evaluate_task, "_log_interval", 0) or 0

    for idx, example in enumerate(dataset):
        row = example if isinstance(example, dict) else dict(example)
        skipped_truth_labels = _excluded_truth_labels_in_segments(
            row.get("segments"),
            excluded_truth_labels,
        )
        if skipped_truth_labels:
            _record_truth_label_exclusion_skip(truth_label_exclusions, skipped_truth_labels)
            continue
        counted_samples += 1
        raw_content = row.get("content")
        content = normalize_eval_text(raw_content if isinstance(raw_content, str) else "")
        metadata = _parse_metadata(row)
        support_mask = _support_mask_from_text(content)
        declared_host_label = _declared_postprocess_host_label(metadata)
        normalized_segments = _normalize_segments(row.get("segments"))
        truth = _segments_to_labels(content, normalized_segments, label_to_idx)
        truth = _apply_relevant_content_postprocess_dense(
            truth,
            support_mask=support_mask,
            label_to_idx=label_to_idx,
            declared_host_label=declared_host_label,
        )
        if postprocess_profile_norm == POSTPROCESS_PROFILE_THESIS:
            segments, pred_labels, pred_probs = runner.segment_text(
                content,
                min_run_chars=min_run_chars,
                postprocess_profile=POSTPROCESS_PROFILE_THESIS,
                other_threshold=float(other_threshold or 0.0),
                postprocess_min_run_chars=int(postprocess_min_run_chars),
                postprocess_boundary_snap_max_shift=int(postprocess_boundary_snap_max_shift),
            )
        elif postprocess_profile_norm == POSTPROCESS_PROFILE_OFF:
            segments, pred_labels, pred_probs = runner.segment_text(content, min_run_chars=min_run_chars)
        else:
            raise ValueError(f"Unknown postprocess profile: {postprocess_profile!r}")
        if postprocess_stats is not None:
            sample_postprocess_stats = getattr(runner, "last_postprocess_stats", None)
            postprocess_stats["samples"] = int(postprocess_stats.get("samples", 0)) + 1
            if isinstance(sample_postprocess_stats, Mapping):
                for key in (
                    "processed_chars",
                    "other_thresholded_chars",
                    "boundary_snap_changed_chars",
                    "min_run_changed_chars",
                    "changed_chars",
                ):
                    postprocess_stats[key] = int(postprocess_stats.get(key, 0)) + int(
                        sample_postprocess_stats.get(key, 0) or 0
                    )
        pred_idx_array = np.full((len(pred_labels),), -1, dtype=np.int32)
        prob_rows: Optional[List[Optional[np.ndarray]]] = [None] * len(pred_labels) if payload_stats is not None else None
        other_idx_eval = label_to_idx.get("other")
        use_other_threshold = (
            postprocess_profile_norm != POSTPROCESS_PROFILE_THESIS
            and other_threshold is not None
            and float(other_threshold) > 0.0
            and other_idx_eval is not None
        )
        for i, lbl_id in enumerate(pred_labels):
            label_name = cfg.ID2LANG.get(int(lbl_id), None)
            if label_name is not None:
                alias = PREDICTION_LABEL_ALIASES.get(label_name)
                if alias and alias in label_to_idx:
                    label_name = alias

            # Compute max probability for this character if available.
            max_prob = None
            try:
                if pred_probs is not None and i < len(pred_probs):
                    prob_vec = np.asarray(pred_probs[i], dtype=np.float32)
                    if prob_vec.size:
                        if prob_rows is not None:
                            prob_rows[i] = prob_vec
                        max_prob = float(prob_vec.max())
            except Exception:
                max_prob = None

            # Route unknown labels into the open-set 'other' bucket if possible.
            if label_name is None or label_name not in label_to_idx:
                if other_idx_eval is not None:
                    pred_idx_array[i] = other_idx_eval
                else:
                    # Fallback: leave as -1 so it is ignored.
                    pred_idx_array[i] = -1
                continue

            # Apply open-set thresholding: low-confidence predictions -> 'other'.
            if use_other_threshold and max_prob is not None and max_prob < float(other_threshold):
                pred_idx_array[i] = other_idx_eval  # type: ignore[arg-type]
            else:
                pred_idx_array[i] = label_to_idx[label_name]
        pred_idx_array = _apply_relevant_content_postprocess_dense(
            pred_idx_array,
            support_mask=support_mask,
            label_to_idx=label_to_idx,
            declared_host_label=declared_host_label,
        )
        pred_text_like_flags = np.zeros((len(pred_idx_array),), dtype=bool)
        for label_name in TEXT_LIKE_POSITIVE_LABELS:
            label_idx = label_to_idx.get(label_name)
            if label_idx is None:
                continue
            pred_text_like_flags[pred_idx_array == int(label_idx)] = True

        valid_mask = (truth >= 0)
        if content:
            ws_mask = np.array([ch in _VISUAL_WHITESPACE_SET for ch in content], dtype=bool)
            valid_mask = np.logical_and(valid_mask, ~ws_mask)
        truth_valid = truth[valid_mask]
        pred_valid = pred_idx_array[valid_mask]
        pred_text_like_valid = pred_text_like_flags[valid_mask]
        prob_valid = [prob_rows[i] for i, keep in enumerate(valid_mask) if keep] if prob_rows is not None else None
        same_mask = (pred_valid == truth_valid)

        valid_pred_mask = pred_valid >= 0
        np.add.at(confusion, (truth_valid[valid_pred_mask], pred_valid[valid_pred_mask]), 1)

        if text_like_confusion is not None and text_like_positive_ids is not None:
            truth_bin = np.isin(truth_valid, text_like_positive_ids).astype(np.int32, copy=False)
            pred_bin = pred_text_like_valid.astype(np.int32, copy=False)
            accumulate_confusion(text_like_confusion, truth_bin, pred_bin)

        total_chars += int(valid_mask.sum())
        for lbl_idx, lbl_name in enumerate(label_names):
            mask = truth_valid == lbl_idx
            count = int(mask.sum())
            if count == 0:
                continue
            per_label_counts[lbl_name] += count
            correct = int((same_mask & mask).sum())
            per_label_correct[lbl_name] += correct

        if pure_stats is not None:
            host_label = metadata.get("host_lang")
            if not host_label and normalized_segments:
                host_label = normalized_segments[0].get("label")
            host_idx = label_to_idx.get(host_label) if host_label else None
            if host_idx is not None:
                # Host-byte-centric purity: only consider bytes whose ground-truth
                # label matches the host language. Non-host regions are ignored for
                # the purposes of "pure" / "within_threshold" so that mixed files
                # (e.g., HTML with embedded JS/CSS) are judged purely on whether the
                # host content was misclassified.
                host_mask = (truth_valid == host_idx)
                host_chars = int(host_mask.sum())
                if host_chars > 0:
                    host_pred = pred_valid[host_mask]
                    label_stats = None
                    if host_label:
                        label_stats = pure_stats.setdefault("per_label", {}).setdefault(
                            host_label,
                            {"total": 0, "pure": 0, "within": 0, "ratios": []},
                        )
                    # Count false negatives on host bytes only.
                    host_errors = int((host_pred != host_idx).sum())
                    pure_stats["total"] += 1
                    if label_stats is not None:
                        label_stats["total"] += 1
                    if host_errors == 0:
                        pure_stats["perfect"] += 1
                        if label_stats is not None:
                            label_stats["pure"] += 1
                    host_error_ratio = host_errors / host_chars if host_chars > 0 else 0.0
                    pure_stats.setdefault("foreign_ratios", []).append(host_error_ratio)
                    if label_stats is not None:
                        label_stats["ratios"].append(host_error_ratio)
                    if host_error_ratio <= pure_stats["threshold"]:
                        pure_stats["within_threshold"] += 1
                        if label_stats is not None:
                            label_stats["within"] += 1

                    # File-level purity for the host language: for this host file,
                    # were all host-label characters predicted correctly?
                    host_correct = host_chars - host_errors
                    host_pure = host_correct == host_chars
                    host_purity = pure_stats.setdefault("host_byte_purity", {}).setdefault(
                        host_label,
                        {"files": 0, "pure_files": 0},
                    )
                    host_purity["files"] += 1
                    if host_pure:
                        host_purity["pure_files"] += 1

        if needle_stats is not None:
            donor_label = metadata.get("donor_lang")
            donor_idx = label_to_idx.get(donor_label) if donor_label else None
            host_label = metadata.get("host_lang")
            if not host_label and normalized_segments:
                host_label = normalized_segments[0].get("label")
            host_idx = label_to_idx.get(host_label) if host_label else None
            threshold = float(needle_stats.get("threshold", NEEDLE_COVERAGE_THRESHOLD))
            region_start = metadata.get(
                "inserted_char_start",
                metadata.get("needle_char_start", metadata.get("injection_char_start")),
            )
            region_end = metadata.get(
                "inserted_char_end",
                metadata.get("needle_char_end", metadata.get("injection_char_end")),
            )
            boundary_entry = needle_stats.get("boundary_region")
            if isinstance(boundary_entry, dict):
                boundary_mask_valid = _boundary_region_valid_mask(
                    len(content),
                    valid_mask,
                    region_start,
                    region_end,
                    radius_tokens=int(
                        boundary_entry.get("window_radius_tokens", NEEDLE_BOUNDARY_WINDOW_TOKENS)
                    ),
                )
                if boundary_mask_valid is not None:
                    boundary_entry["samples"] += 1
                    boundary_truth = truth_valid[boundary_mask_valid]
                    boundary_pred = pred_valid[boundary_mask_valid]
                    boundary_valid_pred = boundary_pred >= 0
                    if np.any(boundary_valid_pred):
                        np.add.at(
                            boundary_entry["confusion"],
                            (boundary_truth[boundary_valid_pred], boundary_pred[boundary_valid_pred]),
                            1,
                        )
            region_mask_valid = _region_mask_valid(
                len(content),
                valid_mask,
                region_start,
                region_end,
            )
            if region_mask_valid is not None:
                exact = _exact_region_stats(truth_valid, pred_valid, region_mask_valid)
                needle_chars = int(exact["truth_chars"])
                if needle_chars > 0:
                    needle_stats["total"] += 1
                    needle_stats["needle_chars"] += needle_chars
                    needle_stats["correct_chars"] += int(exact["correct_chars"])
                    needle_stats["iou_sum"] += float(exact["coverage"])
                    needle_stats["coverage_sum"] += float(exact["coverage"])
                    needle_stats["coverage_count"] += 1
                    if float(exact["coverage"]) >= threshold:
                        needle_stats["detected"] += 1
                        needle_stats["coverage_hits"] += 1
                    else:
                        needle_stats["below_threshold"] += 1

                    pred_slice = pred_valid[region_mask_valid]
                    truth_slice = truth_valid[region_mask_valid]
                    if pred_slice.size > 0:
                        mis_mask = (pred_slice != truth_slice)
                        if np.any(mis_mask):
                            mis_counts = np.bincount(pred_slice[mis_mask], minlength=len(label_names))
                            mis_hist = needle_stats.setdefault("misclass_counts", [0] * len(label_names))
                            if len(mis_hist) < len(label_names):
                                mis_hist.extend([0] * (len(label_names) - len(mis_hist)))
                            for lbl_idx, value in enumerate(mis_counts):
                                if lbl_idx < len(mis_hist):
                                    mis_hist[lbl_idx] += int(value)

                    if donor_label:
                        by_lang = needle_stats.setdefault("by_lang", {}).setdefault(
                            donor_label,
                            {
                                "count": 0,
                                "detected": 0,
                                "below_threshold": 0,
                                "truth_chars": 0,
                                "correct_chars": 0,
                                "iou_sum": 0.0,
                                "coverage_hits": 0,
                                "coverage_sum": 0.0,
                                "coverage_count": 0,
                            },
                        )
                        by_lang["count"] += 1
                        by_lang["truth_chars"] += needle_chars
                        by_lang["correct_chars"] += int(exact["correct_chars"])
                        by_lang["iou_sum"] += float(exact["coverage"])
                        by_lang["coverage_sum"] += float(exact["coverage"])
                        by_lang["coverage_count"] += 1
                        if float(exact["coverage"]) >= threshold:
                            by_lang["detected"] += 1
                            by_lang["coverage_hits"] += 1
                        else:
                            by_lang["below_threshold"] += 1

                    any_entry = needle_stats.get("any_detection")
                    if any_entry is not None:
                        any_stats = _region_non_host_stats(pred_valid, region_mask_valid, host_idx)
                        any_entry["count"] += 1
                        any_entry["truth_chars"] += int(any_stats["truth_chars"])
                        any_entry["correct_chars"] += int(any_stats["matched_chars"])
                        any_entry["iou_sum"] += float(any_stats["coverage"])
                        any_entry["coverage_sum"] += float(any_stats["coverage"])
                        any_entry["coverage_count"] += 1
                        if float(any_stats["coverage"]) >= threshold:
                            any_entry["detected"] += 1
                            any_entry["coverage_hits"] += 1
                        else:
                            any_entry["below_threshold"] += 1

                        if donor_label:
                            any_by_lang = needle_stats.setdefault("any_by_lang", {}).setdefault(
                                donor_label,
                                {
                                    "count": 0,
                                    "detected": 0,
                                    "below_threshold": 0,
                                    "truth_chars": 0,
                                    "correct_chars": 0,
                                    "iou_sum": 0.0,
                                    "coverage_hits": 0,
                                    "coverage_sum": 0.0,
                                    "coverage_count": 0,
                                },
                            )
                            any_by_lang["count"] += 1
                            any_by_lang["truth_chars"] += int(any_stats["truth_chars"])
                            any_by_lang["correct_chars"] += int(any_stats["matched_chars"])
                            any_by_lang["iou_sum"] += float(any_stats["coverage"])
                            any_by_lang["coverage_sum"] += float(any_stats["coverage"])
                            any_by_lang["coverage_count"] += 1
                            if float(any_stats["coverage"]) >= threshold:
                                any_by_lang["detected"] += 1
                                any_by_lang["coverage_hits"] += 1
                            else:
                                any_by_lang["below_threshold"] += 1
            elif donor_idx is not None:
                truth_mask = (truth_valid == donor_idx)
                needle_chars = int(truth_mask.sum())
                if needle_chars > 0:
                    needle_stats["total"] += 1
                    needle_stats["needle_chars"] += needle_chars
                    pred_slice = pred_valid[truth_mask]
                    if pred_slice.size > 0:
                        mis_mask = (pred_slice != donor_idx)
                        if np.any(mis_mask):
                            mis_counts = np.bincount(pred_slice[mis_mask], minlength=len(label_names))
                            mis_hist = needle_stats.setdefault("misclass_counts", [0] * len(label_names))
                            if len(mis_hist) < len(label_names):
                                mis_hist.extend([0] * (len(label_names) - len(mis_hist)))
                            for lbl_idx, value in enumerate(mis_counts):
                                if lbl_idx < len(mis_hist):
                                    mis_hist[lbl_idx] += int(value)

                    pred_mask_lang = (pred_valid == donor_idx)
                    intersection = int(np.logical_and(pred_mask_lang, truth_mask).sum())
                    coverage_ratio = intersection / needle_chars if needle_chars > 0 else 0.0
                    needle_stats["correct_chars"] += intersection
                    needle_stats["iou_sum"] += coverage_ratio
                    needle_stats["coverage_sum"] += coverage_ratio
                    needle_stats["coverage_count"] += 1
                    if coverage_ratio >= threshold:
                        needle_stats["detected"] += 1
                        needle_stats["coverage_hits"] += 1
                    else:
                        needle_stats["below_threshold"] += 1

                    any_entry = needle_stats.get("any_detection")
                    if any_entry is not None:
                        valid_pred_mask = (pred_valid >= 0)
                        if host_idx is not None:
                            foreign_mask = np.logical_and(valid_pred_mask, pred_valid != host_idx)
                        else:
                            foreign_mask = valid_pred_mask
                        intersection_any = int(np.logical_and(foreign_mask, truth_mask).sum())
                        coverage_any = intersection_any / needle_chars if needle_chars > 0 else 0.0
                        any_entry["count"] += 1
                        any_entry["truth_chars"] += needle_chars
                        any_entry["correct_chars"] += intersection_any
                        any_entry["iou_sum"] += coverage_any
                        any_entry["coverage_sum"] += coverage_any
                        any_entry["coverage_count"] += 1
                        if coverage_any >= threshold:
                            any_entry["detected"] += 1
                            any_entry["coverage_hits"] += 1
                        else:
                            any_entry["below_threshold"] += 1

        if markdown_stats is not None:
            threshold = float(markdown_stats.get("threshold", MARKDOWN_IOU_THRESHOLD))
            text_idx = label_to_idx.get("text")
            boundary_positions: List[int] = []
            markdown_stats.setdefault("per_language", {})
            text_stats = markdown_stats.setdefault(
                "text",
                {"truth_chars": 0, "correct_chars": 0, "union_chars": 0, "samples": 0},
            )
            if text_idx is not None:
                text_truth_mask = (truth_valid == text_idx)
                truth_chars_text = int(text_truth_mask.sum())
                pred_text_mask = (pred_valid == text_idx)
                union_text = int(np.logical_or(text_truth_mask, pred_text_mask).sum())
                intersection_text = int(np.logical_and(text_truth_mask, pred_text_mask).sum())
                text_stats["truth_chars"] += truth_chars_text
                text_stats["correct_chars"] += intersection_text
                text_stats["union_chars"] += union_text
                if truth_chars_text or int(pred_text_mask.sum()) > 0:
                    text_stats["samples"] += 1

            blocks_meta = metadata.get("markdown_blocks") or []
            is_monitor_markdown = bool(metadata.get("monitor_source"))
            inferred_host_lang: Optional[str] = None
            if is_monitor_markdown and blocks_meta:
                inferred_host_lang = _infer_markdown_host_lang(blocks_meta)

            inline_blocks_from_main: List[Mapping[str, Any]] = []
            for block in blocks_meta:
                start = int(block.get("char_start", 0))
                end = int(block.get("char_end", start))
                if end <= start:
                    continue
                start = max(0, min(len(pred_idx_array), start))
                end = max(start, min(len(pred_idx_array), end))
                actual_label = block.get("language")
                actual_idx = label_to_idx.get(actual_label)
                if actual_idx is None or end <= start:
                    continue
                # New test-split schema: every block contributes a boundary at its
                # start and end. The legacy exact_region path appends below as well,
                # so guard with a list-id check to avoid double-counting.
                boundary_positions.extend((start, end))
                # Inline blocks (wrapper == 'inline') are accumulated into the
                # inline_stats payload below instead of the block-level stats.
                if block.get("wrapper") == "inline":
                    inline_blocks_from_main.append(block)
                    continue
                if block.get("truth_mode") == "exact_region":
                    region_valid_mask = _region_mask_valid(len(content), valid_mask, start, end)
                    if region_valid_mask is None:
                        continue
                    block_len = int(region_valid_mask.sum())
                    if block_len <= 0:
                        continue
                    pred_slice = pred_valid[region_valid_mask]
                    exact = _exact_region_stats(truth_valid, pred_valid, region_valid_mask)
                    correct_chars = int(exact["correct_chars"])
                    nontext_chars, text_chars = _region_text_stats(
                        pred_valid,
                        region_valid_mask,
                        text_idx=text_idx,
                    )
                    coverage = float(exact["coverage"])
                    nontext_cov = (nontext_chars / block_len) if block_len else 0.0
                    text_cov = (text_chars / block_len) if block_len else 0.0
                    correct_iou = coverage
                    nontext_iou = nontext_cov
                    text_iou = text_cov
                    union_correct = block_len
                    union_nontext = block_len
                    union_text = block_len
                else:
                    slice_truth = truth[start:end]
                    valid_mask_block = (slice_truth == actual_idx)
                    block_len = int(valid_mask_block.sum())
                    if block_len <= 0:
                        continue
                    slice_pred_full = pred_idx_array[start:end]
                    pred_slice = slice_pred_full[valid_mask_block]
                    pred_correct_mask = (slice_pred_full == actual_idx)
                    if text_idx is not None:
                        pred_text_mask = (slice_pred_full == text_idx)
                        pred_nontext_mask = np.logical_not(pred_text_mask)
                    else:
                        pred_text_mask = np.zeros_like(slice_pred_full, dtype=bool)
                        pred_nontext_mask = np.ones_like(slice_pred_full, dtype=bool)
                    intersection_correct = int(np.logical_and(pred_correct_mask, valid_mask_block).sum())
                    union_correct = int(np.logical_or(pred_correct_mask, valid_mask_block).sum())
                    intersection_nontext = int(np.logical_and(pred_nontext_mask, valid_mask_block).sum())
                    union_nontext = int(np.logical_or(pred_nontext_mask, valid_mask_block).sum())
                    intersection_text = int(np.logical_and(pred_text_mask, valid_mask_block).sum())
                    union_text = int(np.logical_or(pred_text_mask, valid_mask_block).sum())
                    coverage = intersection_correct / block_len if block_len else 0.0
                    nontext_cov = intersection_nontext / block_len if block_len else 0.0
                    text_cov = intersection_text / block_len if block_len else 0.0
                    correct_iou = intersection_correct / union_correct if union_correct > 0 else 0.0
                    nontext_iou = intersection_nontext / union_nontext if union_nontext > 0 else 0.0
                    text_iou = intersection_text / union_text if union_text > 0 else 0.0

                    correct_chars = intersection_correct
                    nontext_chars = intersection_nontext
                    text_chars = intersection_text

                # Recover wrapper/role semantics for monitor-based markdown docs.
                # New test-split schema uses block['wrapper'] ∈ {'fenced','plain','inline'}.
                wrapped_flag = bool(block.get("wrapped")) or block.get("wrapper") == "fenced"
                if is_monitor_markdown and not wrapped_flag:
                    if _is_fenced_markdown_block(content, start, end):
                        wrapped_flag = True
                wrapper_key = "wrapped" if wrapped_flag else "plain"

                role = block.get("role")
                if is_monitor_markdown:
                    if inferred_host_lang and actual_label == inferred_host_lang:
                        role = "host"
                    else:
                        role = "other"
                if role not in ("host", "other"):
                    role = "host"

                role_groups = markdown_stats["roles"].setdefault(
                    role,
                    {"wrapped": _markdown_stat_group(), "plain": _markdown_stat_group()},
                )
                role_group = role_groups[wrapper_key]
                overall_group = markdown_stats["overall"][wrapper_key]
                lang_group = markdown_stats.setdefault("per_language", {}).setdefault(
                    actual_label,
                    {"wrapped": _markdown_stat_group(), "plain": _markdown_stat_group()},
                )[wrapper_key]
                for group in (role_group, overall_group, lang_group):
                    group["count"] += 1
                    group["truth_chars"] += block_len
                    group["correct_chars"] += correct_chars
                    group["nontext_chars"] += nontext_chars
                    group["text_chars"] += text_chars
                    group["union_correct_chars"] += union_correct
                    group["union_nontext_chars"] += union_nontext
                    group["union_text_chars"] += union_text
                    if math.isfinite(correct_iou):
                        group["iou_sum"] += correct_iou
                    group["nontext_iou_sum"] += nontext_iou
                    group["text_iou_sum"] += text_iou
                    if coverage >= threshold:
                        group["detected_correct"] += 1
                    if nontext_cov >= threshold:
                        group["detected_nontext"] += 1
                    if text_cov >= threshold:
                        group["detected_text"] += 1
                    if correct_iou >= threshold:
                        group["detected_correct_iou"] += 1
                    if nontext_iou >= threshold:
                        group["detected_nontext_iou"] += 1
                    if text_iou >= threshold:
                        group["detected_text_iou"] += 1

                if block.get("mismatched") and block.get("display_language"):
                    markdown_stats["wrong_label"]["cases"] += 1
                    wrong_label_name = block.get("display_language")
                    wrong_idx = label_to_idx.get(wrong_label_name)
                    fooled = False
                    if wrong_idx is not None:
                        fooled_chars = int((pred_slice == wrong_idx).sum())
                        if block_len > 0 and (fooled_chars / block_len) >= threshold:
                            fooled = True
                    if fooled:
                        markdown_stats["wrong_label"]["fooled"] += 1
                    for group in (role_group, overall_group, lang_group):
                        group["wrong_label_cases"] += 1
                        if fooled:
                            group["wrong_label_fooled"] += 1

            inline_meta = list(metadata.get("inline_blocks") or [])
            # New test-split schema embeds inline blocks inside markdown_blocks
            # with wrapper == "inline" rather than a separate inline_blocks list.
            if inline_blocks_from_main:
                inline_meta = inline_meta + inline_blocks_from_main
            inline_stats = markdown_stats["inline"]
            inline_stats.setdefault("per_language", {})
            inline_threshold = float(inline_stats.get("threshold", MARKDOWN_IOU_THRESHOLD))
            for inline_block in inline_meta:
                start = int(inline_block.get("char_start", 0))
                end = int(inline_block.get("char_end", start))
                if end <= start:
                    continue
                start = max(0, min(len(pred_idx_array), start))
                end = max(start, min(len(pred_idx_array), end))
                inline_label = inline_block.get("language")
                inline_idx = label_to_idx.get(inline_label)
                if inline_idx is None or end <= start:
                    continue
                if inline_block.get("truth_mode") == "exact_region":
                    region_valid_mask = _region_mask_valid(len(content), valid_mask, start, end)
                    if region_valid_mask is None:
                        continue
                    block_len = int(region_valid_mask.sum())
                    if block_len <= 0:
                        continue
                    boundary_positions.extend((start, end))
                    exact = _exact_region_stats(truth_valid, pred_valid, region_valid_mask)
                    correct_chars = int(exact["correct_chars"])
                    nontext_chars, text_chars = _region_text_stats(
                        pred_valid,
                        region_valid_mask,
                        text_idx=text_idx,
                    )
                    coverage = float(exact["coverage"])
                    nontext_cov = (nontext_chars / block_len) if block_len else 0.0
                    text_cov = (text_chars / block_len) if block_len else 0.0
                    correct_iou = coverage
                    nontext_iou = nontext_cov
                    text_iou = text_cov
                    union_correct = block_len
                    union_nontext = block_len
                    union_text = block_len
                else:
                    slice_truth = truth[start:end]
                    valid_mask_block = (slice_truth == inline_idx)
                    block_len = int(valid_mask_block.sum())
                    if block_len <= 0:
                        continue
                    slice_pred_full = pred_idx_array[start:end]
                    if text_idx is not None:
                        pred_text_mask = (slice_pred_full == text_idx)
                        pred_nontext_mask = np.logical_not(pred_text_mask)
                    else:
                        pred_text_mask = np.zeros_like(slice_pred_full, dtype=bool)
                        pred_nontext_mask = np.ones_like(slice_pred_full, dtype=bool)
                    pred_correct_mask = (slice_pred_full == inline_idx)
                    intersection_correct = int(np.logical_and(pred_correct_mask, valid_mask_block).sum())
                    union_correct = int(np.logical_or(pred_correct_mask, valid_mask_block).sum())
                    intersection_nontext = int(np.logical_and(pred_nontext_mask, valid_mask_block).sum())
                    union_nontext = int(np.logical_or(pred_nontext_mask, valid_mask_block).sum())
                    intersection_text = int(np.logical_and(pred_text_mask, valid_mask_block).sum())
                    union_text = int(np.logical_or(pred_text_mask, valid_mask_block).sum())
                    coverage = intersection_correct / block_len if block_len else 0.0
                    nontext_cov = intersection_nontext / block_len if block_len else 0.0
                    text_cov = intersection_text / block_len if block_len else 0.0
                    correct_iou = intersection_correct / union_correct if union_correct > 0 else 0.0
                    nontext_iou = intersection_nontext / union_nontext if union_nontext > 0 else 0.0
                    text_iou = intersection_text / union_text if union_text > 0 else 0.0
                    correct_chars = intersection_correct
                    nontext_chars = intersection_nontext
                    text_chars = intersection_text
                inline_stats["count"] += 1
                inline_stats["truth_chars"] += block_len
                inline_stats["correct_chars"] += correct_chars
                inline_stats["nontext_chars"] += nontext_chars
                inline_stats["text_chars"] += text_chars
                inline_stats["union_correct_chars"] += union_correct
                inline_stats["union_nontext_chars"] += union_nontext
                inline_stats["union_text_chars"] += union_text
                inline_stats.setdefault("correct_iou_sum", 0.0)
                inline_stats.setdefault("nontext_iou_sum", 0.0)
                inline_stats.setdefault("text_iou_sum", 0.0)
                inline_stats["correct_iou_sum"] += correct_iou
                inline_stats["nontext_iou_sum"] += nontext_iou
                inline_stats["text_iou_sum"] += text_iou
                wrapper_type = inline_block.get("wrapper", "inline_backtick")
                wrapper_entry = inline_stats["by_wrapper"].setdefault(
                    wrapper_type,
                    {
                        "count": 0,
                        "correct_hits": 0,
                        "nontext_hits": 0,
                        "text_hits": 0,
                        "truth_chars": 0,
                        "correct_chars": 0,
                        "nontext_chars": 0,
                        "text_chars": 0,
                        "union_correct_chars": 0,
                        "union_nontext_chars": 0,
                        "union_text_chars": 0,
                        "correct_iou_sum": 0.0,
                        "nontext_iou_sum": 0.0,
                        "text_iou_sum": 0.0,
                        "correct_iou_hits": 0,
                        "nontext_iou_hits": 0,
                        "text_iou_hits": 0,
                    },
                )
                wrapper_entry["count"] += 1
                wrapper_entry["truth_chars"] += block_len
                wrapper_entry["correct_chars"] += correct_chars
                wrapper_entry["nontext_chars"] += nontext_chars
                wrapper_entry["text_chars"] += text_chars
                wrapper_entry["union_correct_chars"] += union_correct
                wrapper_entry["union_nontext_chars"] += union_nontext
                wrapper_entry["union_text_chars"] += union_text
                wrapper_entry["correct_iou_sum"] += correct_iou
                wrapper_entry["nontext_iou_sum"] += nontext_iou
                wrapper_entry["text_iou_sum"] += text_iou
                if coverage >= inline_threshold:
                    inline_stats["correct_hits"] += 1
                    wrapper_entry["correct_hits"] += 1
                if nontext_cov >= inline_threshold:
                    inline_stats["nontext_hits"] += 1
                    wrapper_entry["nontext_hits"] += 1
                if text_cov >= inline_threshold:
                    inline_stats["text_hits"] += 1
                    wrapper_entry["text_hits"] += 1
                inline_stats.setdefault("correct_iou_hits", 0)
                inline_stats.setdefault("nontext_iou_hits", 0)
                inline_stats.setdefault("text_iou_hits", 0)
                inline_stats["correct_iou_hits"] += int(correct_iou >= inline_threshold)
                inline_stats["nontext_iou_hits"] += int(nontext_iou >= inline_threshold)
                inline_stats["text_iou_hits"] += int(text_iou >= inline_threshold)
                wrapper_entry["correct_iou_hits"] += int(correct_iou >= inline_threshold)
                wrapper_entry["nontext_iou_hits"] += int(nontext_iou >= inline_threshold)
                wrapper_entry["text_iou_hits"] += int(text_iou >= inline_threshold)
                lang_inline = inline_stats.setdefault("per_language", {}).setdefault(
                    inline_label,
                    {
                        "count": 0,
                        "correct_hits": 0,
                        "nontext_hits": 0,
                        "text_hits": 0,
                        "truth_chars": 0,
                        "correct_chars": 0,
                        "nontext_chars": 0,
                        "text_chars": 0,
                        "union_correct_chars": 0,
                        "union_nontext_chars": 0,
                        "union_text_chars": 0,
                        "correct_iou_sum": 0.0,
                        "nontext_iou_sum": 0.0,
                        "text_iou_sum": 0.0,
                        "correct_iou_hits": 0,
                        "nontext_iou_hits": 0,
                        "text_iou_hits": 0,
                    },
                )
                lang_inline["count"] += 1
                lang_inline["truth_chars"] += block_len
                lang_inline["correct_chars"] += correct_chars
                lang_inline["nontext_chars"] += nontext_chars
                lang_inline["text_chars"] += text_chars
                lang_inline["union_correct_chars"] += union_correct
                lang_inline["union_nontext_chars"] += union_nontext
                lang_inline["union_text_chars"] += union_text
                lang_inline["correct_iou_sum"] += correct_iou
                lang_inline["nontext_iou_sum"] += nontext_iou
                lang_inline["text_iou_sum"] += text_iou
                if coverage >= inline_threshold:
                    lang_inline["correct_hits"] += 1
                if nontext_cov >= inline_threshold:
                    lang_inline["nontext_hits"] += 1
                if text_cov >= inline_threshold:
                    lang_inline["text_hits"] += 1
                lang_inline["correct_iou_hits"] += int(correct_iou >= inline_threshold)
                lang_inline["nontext_iou_hits"] += int(nontext_iou >= inline_threshold)
                lang_inline["text_iou_hits"] += int(text_iou >= inline_threshold)

            boundary_entry = markdown_stats.get("boundary_region")
            if isinstance(boundary_entry, dict) and boundary_positions:
                boundary_mask_valid = _boundary_positions_valid_mask(
                    len(content),
                    valid_mask,
                    boundary_positions,
                    radius_tokens=int(
                        boundary_entry.get(
                            "window_radius_tokens",
                            NEEDLE_BOUNDARY_WINDOW_TOKENS,
                        )
                    ),
                )
                if boundary_mask_valid is not None:
                    boundary_entry["samples"] += 1
                    boundary_truth = truth_valid[boundary_mask_valid]
                    boundary_pred = pred_valid[boundary_mask_valid]
                    boundary_valid_pred = boundary_pred >= 0
                    if np.any(boundary_valid_pred):
                        np.add.at(
                            boundary_entry["confusion"],
                            (boundary_truth[boundary_valid_pred], boundary_pred[boundary_valid_pred]),
                            1,
                        )

        if payload_stats is not None:
            payload_lang = metadata.get("payload_lang")
            host_lang = metadata.get("host_lang")
            host_idx = label_to_idx.get(host_lang) if host_lang else None
            if payload_lang:
                lang_idx = label_to_idx.get(payload_lang)
                if lang_idx is not None:
                    truth_mask = (truth_valid == lang_idx)
                    truth_chars = int(truth_mask.sum())
                    if truth_chars > 0:
                        threshold = float(payload_stats.get("threshold", PAYLOAD_IOU_THRESHOLD))
                        any_entry = payload_stats.get("any_detection")
                        if any_entry is not None:
                            if host_idx is not None:
                                non_host_mask = (pred_valid != host_idx)
                            else:
                                non_host_mask = np.ones_like(pred_valid, dtype=bool)
                            intersection_any = int(np.logical_and(non_host_mask, truth_mask).sum())
                            union_any = int(np.logical_or(non_host_mask, truth_mask).sum())
                            coverage_any = intersection_any / truth_chars if truth_chars > 0 else 0.0
                            any_entry["coverage_sum"] += coverage_any
                            any_entry["coverage_count"] += 1
                            if coverage_any >= threshold:
                                any_entry["coverage_hits"] += 1
                            if union_any > 0:
                                any_entry["count"] += 1
                                any_entry["truth_chars"] += truth_chars
                                any_entry["correct_chars"] += intersection_any
                                iou_any = intersection_any / union_any
                                any_entry["iou_sum"] += iou_any
                                if iou_any >= threshold:
                                    any_entry["detected"] += 1
                                else:
                                    any_entry["below_threshold"] += 1
                                lang_any_entry = payload_stats.setdefault("any_by_lang", {}).setdefault(
                                    payload_lang,
                                    {
                                        "count": 0,
                                        "detected": 0,
                                        "below_threshold": 0,
                                        "truth_chars": 0,
                                        "correct_chars": 0,
                                        "iou_sum": 0.0,
                                        "coverage_hits": 0,
                                        "coverage_sum": 0.0,
                                        "coverage_count": 0,
                                    },
                                )
                                lang_any_entry["count"] += 1
                                lang_any_entry["truth_chars"] += truth_chars
                                lang_any_entry["correct_chars"] += intersection_any
                                lang_any_entry["coverage_sum"] += coverage_any
                                lang_any_entry["coverage_count"] += 1
                                if coverage_any >= threshold:
                                    lang_any_entry["coverage_hits"] += 1
                                lang_any_entry["iou_sum"] += iou_any
                                if iou_any >= threshold:
                                    lang_any_entry["detected"] += 1
                                else:
                                    lang_any_entry["below_threshold"] += 1

                        pred_mask = (pred_valid == idx)
                        intersection = int(np.logical_and(truth_mask, pred_mask).sum())
                        union = int(np.logical_or(truth_mask, pred_mask).sum())
                        if union <= 0:
                            continue
                        entry = payload_stats["by_lang"].setdefault(
                            payload_lang,
                            {
                                "count": 0,
                                "detected": 0,
                                "below_threshold": 0,
                                "truth_chars": 0,
                                "correct_chars": 0,
                                "iou_sum": 0.0,
                                "coverage_hits": 0,
                                "coverage_count": 0,
                            },
                        )
                        entry["count"] += 1
                        entry["truth_chars"] += truth_chars
                        entry["correct_chars"] += intersection
                        iou = intersection / union
                        entry["iou_sum"] += iou
                        coverage_ratio = intersection / truth_chars if truth_chars > 0 else 0.0
                        entry["coverage_count"] += 1
                        if coverage_ratio >= threshold:
                            entry["coverage_hits"] += 1
                        if iou >= threshold:
                            entry["detected"] += 1
                        else:
                            entry["below_threshold"] += 1

                        soft_stats = payload_stats.get("soft_detection") or {}
                        prob_threshold = float(soft_stats.get("prob_threshold", PAYLOAD_SOFT_PROB_THRESHOLD))
                        coverage_threshold = float(
                            soft_stats.get("coverage_threshold", PAYLOAD_SOFT_COVERAGE_THRESHOLD)
                        )
                        if prob_valid is not None:
                            truth_positions = np.flatnonzero(truth_mask)
                            prob_dim = 0
                            for pos in truth_positions:
                                prob_row = prob_valid[int(pos)]
                                if prob_row is not None and prob_row.size > 0:
                                    prob_dim = int(prob_row.size)
                                    break
                            if prob_dim > 0:
                                eval_idx_view = _model_eval_idx_view(payload_model_eval_idx, prob_dim)
                                payload_prob_mask = eval_idx_view == lang_idx
                                non_host_prob_mask = (
                                    np.ones((prob_dim,), dtype=bool)
                                    if host_idx is None
                                    else (eval_idx_view != host_idx)
                                )
                                matched_payload = 0
                                matched_any = 0
                                for pos in truth_positions:
                                    prob_row = prob_valid[int(pos)]
                                    if prob_row is None or prob_row.size == 0:
                                        continue
                                    if prob_row.size != prob_dim:
                                        row_eval_idx = _model_eval_idx_view(payload_model_eval_idx, int(prob_row.size))
                                        payload_prob = float(prob_row[row_eval_idx == lang_idx].sum())
                                        any_prob = (
                                            float(prob_row.sum())
                                            if host_idx is None
                                            else float(prob_row[row_eval_idx != host_idx].sum())
                                        )
                                    else:
                                        payload_prob = float(prob_row[payload_prob_mask].sum())
                                        any_prob = float(prob_row[non_host_prob_mask].sum())
                                    if payload_prob >= prob_threshold:
                                        matched_payload += 1
                                    if any_prob >= prob_threshold:
                                        matched_any += 1

                                correct_soft = soft_stats.setdefault("correct_detection", _payload_soft_stat_group())
                                any_soft = soft_stats.setdefault("any_detection", _payload_soft_stat_group())
                                _update_payload_soft_stat(
                                    correct_soft,
                                    matched_chars=matched_payload,
                                    truth_chars=truth_chars,
                                    coverage_threshold=coverage_threshold,
                                )
                                _update_payload_soft_stat(
                                    any_soft,
                                    matched_chars=matched_any,
                                    truth_chars=truth_chars,
                                    coverage_threshold=coverage_threshold,
                                )
                                lang_correct_soft = soft_stats.setdefault("correct_by_lang", {}).setdefault(
                                    payload_lang,
                                    _payload_soft_stat_group(),
                                )
                                lang_any_soft = soft_stats.setdefault("any_by_lang", {}).setdefault(
                                    payload_lang,
                                    _payload_soft_stat_group(),
                                )
                                _update_payload_soft_stat(
                                    lang_correct_soft,
                                    matched_chars=matched_payload,
                                    truth_chars=truth_chars,
                                    coverage_threshold=coverage_threshold,
                                )
                                _update_payload_soft_stat(
                                    lang_any_soft,
                                    matched_chars=matched_any,
                                    truth_chars=truth_chars,
                                    coverage_threshold=coverage_threshold,
                                )

        if realistic_stats is not None:
            boundary_entry = realistic_stats.get("boundary_region")
            if isinstance(boundary_entry, dict):
                # Detect truth-label transitions in the valid (non-whitespace)
                # truth sequence and accumulate confusion in a window around each.
                valid_positions = np.flatnonzero(valid_mask)
                if valid_positions.size > 1:
                    transition_mask = truth_valid[1:] != truth_valid[:-1]
                    if np.any(transition_mask):
                        boundary_positions = valid_positions[1:][transition_mask].tolist()
                        boundary_mask_valid = _boundary_positions_valid_mask(
                            len(content),
                            valid_mask,
                            boundary_positions,
                            radius_tokens=int(
                                boundary_entry.get(
                                    "window_radius_tokens",
                                    NEEDLE_BOUNDARY_WINDOW_TOKENS,
                                )
                            ),
                        )
                        if boundary_mask_valid is not None:
                            boundary_entry["samples"] += 1
                            boundary_entry["boundaries"] += len(boundary_positions)
                            boundary_truth = truth_valid[boundary_mask_valid]
                            boundary_pred = pred_valid[boundary_mask_valid]
                            boundary_valid_pred = boundary_pred >= 0
                            if np.any(boundary_valid_pred):
                                np.add.at(
                                    boundary_entry["confusion"],
                                    (boundary_truth[boundary_valid_pred], boundary_pred[boundary_valid_pred]),
                                    1,
                                )

        if sequence_stats is not None:
            segments_info = sequence_stats["segments"]
            region_meta = metadata.get("sequence_regions")
            if isinstance(region_meta, dict):
                boundary_entry = sequence_stats.get("boundary_region")
                if isinstance(boundary_entry, dict):
                    boundary_positions: List[int] = []
                    previous_region: Optional[Mapping[str, Any]] = None
                    for pos_key in ("first", "second", "third"):
                        region = region_meta.get(pos_key)
                        if not isinstance(region, Mapping):
                            continue
                        if previous_region is not None:
                            boundary_positions.append(region.get("char_start"))
                        previous_region = region
                    boundary_mask_valid = _boundary_positions_valid_mask(
                        len(content),
                        valid_mask,
                        boundary_positions,
                        radius_tokens=int(
                            boundary_entry.get("window_radius_tokens", NEEDLE_BOUNDARY_WINDOW_TOKENS)
                        ),
                    )
                    if boundary_mask_valid is not None:
                        boundary_entry["samples"] += 1
                        boundary_entry["boundaries"] += len(boundary_positions)
                        boundary_truth = truth_valid[boundary_mask_valid]
                        boundary_pred = pred_valid[boundary_mask_valid]
                        boundary_valid_pred = boundary_pred >= 0
                        if np.any(boundary_valid_pred):
                            np.add.at(
                                boundary_entry["confusion"],
                                (boundary_truth[boundary_valid_pred], boundary_pred[boundary_valid_pred]),
                                1,
                            )
                for pos_key in ("first", "second", "third"):
                    region = region_meta.get(pos_key)
                    if not isinstance(region, dict):
                        continue
                    region_valid_mask = _region_mask_valid(
                        len(content),
                        valid_mask,
                        region.get("char_start"),
                        region.get("char_end"),
                    )
                    if region_valid_mask is None:
                        continue
                    exact = _exact_region_stats(truth_valid, pred_valid, region_valid_mask)
                    if int(exact["truth_chars"]) <= 0:
                        continue
                    segments_info[pos_key]["total"] += int(exact["truth_chars"])
                    segments_info[pos_key]["correct"] += int(exact["correct_chars"])
            else:
                # Derive boundary positions from anchor_chars when sequence_regions
                # are not provided (new test-split schema).
                anchor = metadata.get("anchor_chars")
                if (
                    isinstance(anchor, int)
                    and anchor > 0
                    and "first_lang" in metadata
                    and "second_lang" in metadata
                ):
                    boundary_entry = sequence_stats.get("boundary_region")
                    if isinstance(boundary_entry, dict):
                        boundary_positions = [anchor]
                        if "third_lang" in metadata:
                            boundary_positions.append(2 * anchor)
                        boundary_mask_valid = _boundary_positions_valid_mask(
                            len(content),
                            valid_mask,
                            boundary_positions,
                            radius_tokens=int(
                                boundary_entry.get(
                                    "window_radius_tokens",
                                    NEEDLE_BOUNDARY_WINDOW_TOKENS,
                                )
                            ),
                        )
                        if boundary_mask_valid is not None:
                            boundary_entry["samples"] += 1
                            boundary_entry["boundaries"] += len(boundary_positions)
                            boundary_truth = truth_valid[boundary_mask_valid]
                            boundary_pred = pred_valid[boundary_mask_valid]
                            boundary_valid_pred = boundary_pred >= 0
                            if np.any(boundary_valid_pred):
                                np.add.at(
                                    boundary_entry["confusion"],
                                    (boundary_truth[boundary_valid_pred], boundary_pred[boundary_valid_pred]),
                                    1,
                                )

                pairs: List[Tuple[str, Optional[str]]] = []
                if "first" in segments_info:
                    pairs.append(("first", metadata.get("first_lang")))
                if "second" in segments_info:
                    pairs.append(("second", metadata.get("second_lang")))
                if "third" in segments_info:
                    pairs.append(("third", metadata.get("third_lang")))
                for pos_key, lang in pairs:
                    if not lang:
                        continue
                    lang_idx = label_to_idx.get(lang)
                    if lang_idx is None:
                        continue
                    truth_mask = (truth_valid == lang_idx)
                    total_chars_seg = int(truth_mask.sum())
                    if total_chars_seg == 0:
                        continue
                    correct_chars_seg = int(np.logical_and(pred_valid == lang_idx, truth_mask).sum())
                    segments_info[pos_key]["total"] += total_chars_seg
                    segments_info[pos_key]["correct"] += correct_chars_seg

        if log_interval > 0 and ((idx + 1) % log_interval == 0 or (idx + 1) == total_samples):
            elapsed = time.perf_counter() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            percent = ((idx + 1) / total_samples) * 100 if total_samples else 100.0
            print(
                f"    [{name}] {idx + 1}/{total_samples} samples ({percent:.1f}%) • {elapsed:.1f}s elapsed • {rate:.1f} samp/s",
                flush=True,
            )

    elapsed_total = time.perf_counter() - start_time
    if markdown_stats is not None and text_like_confusion is not None:
        markdown_stats["text_like_binary"] = _build_text_like_binary_payload(text_like_confusion)
    extra_payload["elapsed_seconds"] = elapsed_total
    if truth_label_exclusions is not None:
        truth_label_exclusions["counted"] = int(counted_samples)

    return TaskMetrics(
        name=name,
        description=description,
        samples=counted_samples,
        total_chars=total_chars,
        confusion=confusion,
        label_names=label_names,
        per_label_counts=per_label_counts,
        per_label_correct=per_label_correct,
        extras=extra_payload,
    )


# ---------------------------------------------------------------------------
# Throughput benchmarking
# ---------------------------------------------------------------------------


@dataclass
class ThroughputResult:
    task: str
    device: str
    arch: str
    samples: int
    total_bytes: int
    total_windows: int
    elapsed: float
    throughput: float
    windows_per_second: float
    rss_delta: Optional[float]
    device_mem_delta: Optional[float]


def _process_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def _device_mem_mb(device: jax.Device) -> Optional[float]:
    try:
        stats = device.memory_stats()
        used = stats.get("bytes_in_use") or stats.get("allocation_bytes")
        if used is None:
            return None
        return float(used) / (1024 ** 2)
    except Exception:
        return None


def measure_throughput(
    task: str,
    dataset: hfds.Dataset,
    runner: SegmenterRunner,
    *,
    min_run_chars: int,
    device_name: str,
) -> ThroughputResult:
    samples = len(dataset)
    if samples == 0:
        return ThroughputResult(task, device_name, str(getattr(runner, "arch", "unknown")), 0, 0, 0, 0.0, 0.0, 0.0, None, None)

    total_bytes = 0
    byte_arrays: List[np.ndarray] = []
    for example in dataset:
        meta_raw = example.get("metadata_json")
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
                total_bytes += int(meta.get("actual_bytes", 0))
            except Exception:
                pass
        text = normalize_eval_text(
            example.get("content") if isinstance(example, dict) else example["content"]
        )
        byte_arr = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8).copy()
        byte_arrays.append(byte_arr)
    if total_bytes <= 0:
        total_bytes = int(sum(int(arr.shape[0]) for arr in byte_arrays))

    devices: List[jax.Device] = []
    if runner.backend:
        try:
            devices = jax.devices(runner.backend)
        except Exception:
            devices = []
    if not devices:
        devices = jax.devices()
    device = devices[0]

    rss_before = _process_rss_mb()
    dev_before = _device_mem_mb(device)

    # Warm-up with first example to trigger compilation
    warm_bytes = byte_arrays[0]
    runner.segment_byte_arrays_batch_labels_only([warm_bytes])

    total_windows = 0
    start = time.perf_counter()
    for byte_arr in byte_arrays:
        _, spans_by_text = runner.segment_byte_arrays_batch_labels_only([byte_arr])
        if spans_by_text:
            total_windows += int(len(spans_by_text[0]))
    elapsed = time.perf_counter() - start

    rss_after = _process_rss_mb()
    dev_after = _device_mem_mb(device)

    rss_delta = (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None
    dev_delta = (dev_after - dev_before) if (dev_after is not None and dev_before is not None) else None

    throughput = (total_bytes / elapsed) if elapsed > 0 else 0.0
    windows_per_second = (total_windows / elapsed) if elapsed > 0 else 0.0
    return ThroughputResult(
        task=task,
        device=device_name,
        arch=str(getattr(runner, "arch", "unknown")),
        samples=samples,
        total_bytes=total_bytes,
        total_windows=int(total_windows),
        elapsed=elapsed,
        throughput=throughput,
        windows_per_second=windows_per_second,
        rss_delta=rss_delta,
        device_mem_delta=dev_delta,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _extract_run_id_from_checkpoint(path: Path) -> Optional[str]:
    name = path.name.lower()
    match = re.search(r"([a-z0-9]{8})", name)
    if not match:
        return None
    return match.group(1)


def _load_checkpoint_hparams(ckpt_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    run_id = _extract_run_id_from_checkpoint(ckpt_path)
    if run_id:
        wandb_root = REPO_ROOT / "train" / "wandb"
        if wandb_root.exists():
            try:
                import yaml  # type: ignore
            except Exception:
                yaml = None  # type: ignore
            else:
                pattern = f"run-*-{run_id}"
                for run_dir in wandb_root.glob(pattern):
                    config_path = run_dir / "files" / "config.yaml"
                    if config_path.exists():
                        try:
                            config_data = yaml.safe_load(config_path.read_text())
                        except Exception:
                            config_data = None
                        if isinstance(config_data, dict):
                            channels_val = config_data.get("channels", {}).get("value")
                            if isinstance(channels_val, (list, tuple)):
                                try:
                                    result["channels"] = [int(x) for x in channels_val]
                                except (TypeError, ValueError):
                                    pass
                            model_dim_val = config_data.get("model_dim", {}).get("value")
                            if isinstance(model_dim_val, (int, float)):
                                result["model_dim"] = int(model_dim_val)
                            dtype_val = config_data.get("dtype", {}).get("value")
                            if isinstance(dtype_val, str):
                                result["dtype"] = dtype_val.rsplit(".", 1)[-1]
                            arch_val = config_data.get("arch", {}).get("value")
                            if isinstance(arch_val, str) and arch_val.strip():
                                result["arch"] = arch_val.strip()
                            for key in (
                                "mamba_layers",
                                "mamba_d_state",
                                "mamba_expand",
                                "mamba_dt_rank",
                                "mamba_conv",
                                "mamba_bidirectional",
                            ):
                                val = config_data.get(key, {}).get("value")
                                if isinstance(val, bool):
                                    result[key] = bool(val)
                                elif isinstance(val, (int, float)):
                                    result[key] = int(val)

                    summary_path = run_dir / "files" / "wandb-summary.json"
                    if summary_path.exists():
                        try:
                            summary_data = json.loads(summary_path.read_text())
                        except Exception:
                            summary_data = None
                        if isinstance(summary_data, dict):
                            label_names: Dict[int, str] = {}
                            prefix = "val/per_class/"
                            for key in summary_data.keys():
                                if not key.startswith(prefix):
                                    continue
                                remainder = key[len(prefix):]
                                head = remainder.split("/", 1)[0]
                                if "_" not in head:
                                    continue
                                idx_str, label = head.split("_", 1)
                                try:
                                    idx = int(idx_str)
                                except ValueError:
                                    continue
                                label_names[idx] = label
                            if label_names:
                                ordered = [label_names[i] for i in sorted(label_names)]
                                result["label_names"] = ordered

                    if result and "channels" in result:
                        break

                if "channels" not in result:
                    wandb_bundles = list(
                        wandb_root.glob(f"run-*-{run_id}/run-{run_id}.wandb")
                    )
                    for bundle in wandb_bundles:
                        try:
                            bundle_bytes = bundle.read_bytes()
                        except Exception:
                            continue
                        match = re.search(rb"channels[^\[]*[\[%]([0-9,\s]+)\]", bundle_bytes)
                        if match:
                            chan_vals = [
                                int(x.strip()) for x in match.group(1).split(b",") if x.strip()
                            ]
                            if chan_vals:
                                result["channels"] = chan_vals
                                break
                if "channels" in result:
                    return result

    if "channels" in result:
        return result

    # Fallback: infer from checkpoint weights directly.
    try:
        from flax.serialization import msgpack_restore

        raw = ckpt_path.read_bytes()
        params = msgpack_restore(raw)
        tree = params.get("params") if isinstance(params, Mapping) else None
        if tree is None and isinstance(params, Mapping):
            tree = params
        embed = None
        if isinstance(tree, Mapping):
            embed_node = tree.get("Embed_0")
            if isinstance(embed_node, Mapping):
                embed = embed_node.get("embedding")
        if embed is not None and hasattr(embed, "shape"):
            result["model_dim"] = int(embed.shape[1])

        inferred = _infer_channels_from_params_tree(tree) if isinstance(tree, Mapping) else None
        if inferred:
            result["channels"] = inferred
    except Exception:
        pass

    return result


def _apply_label_mapping(label_names: Sequence[str]) -> None:
    cfg.LANG2ID.clear()
    cfg.LANG2ID.update({name: idx for idx, name in enumerate(label_names)})
    cfg.update_lang_mappings()


def _infer_channels_from_params_tree(params: Mapping[str, Any]) -> Optional[Tuple[int, ...]]:
    if not isinstance(params, Mapping):
        return None
    channels: List[int] = []
    idx = 0
    while True:
        key = f"ConvBlock1D_{idx}"
        block = params.get(key)
        if not isinstance(block, Mapping):
            break
        conv = block.get("Conv_0")
        if not isinstance(conv, Mapping):
            break
        kernel = conv.get("kernel")
        if not hasattr(kernel, "shape") or len(getattr(kernel, "shape", ())) != 3:
            break
        out_ch = int(kernel.shape[-1])
        if not channels:
            channels.append(out_ch)
        elif out_ch > channels[-1]:
            channels.append(out_ch)
        elif out_ch < channels[-1]:
            break
        idx += 1
    return tuple(channels) if channels else None


def _format_bytes_per_sec(value: float) -> str:
    if value <= 0:
        return "0"
    units = ["B/s", "KB/s", "MB/s", "GB/s"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _report_metric_rows(
    metrics: TaskMetrics,
    *,
    exclude_truth_labels: Optional[Sequence[str]] = None,
) -> List[Tuple[str, int, int, float]]:
    exclude_set = set(exclude_truth_labels or ())
    rows: List[Tuple[str, int, int, float]] = []
    for label in metrics.label_names:
        if label in exclude_set:
            continue
        support = int(metrics.per_label_counts.get(label, 0))
        correct = int(metrics.per_label_correct.get(label, 0))
        acc = (correct / support) if support else float("nan")
        rows.append((label, support, correct, acc))
    return rows


def _report_total_chars(
    metrics: TaskMetrics,
    *,
    exclude_truth_labels: Optional[Sequence[str]] = None,
) -> int:
    return sum(
        support
        for _, support, _, _ in _report_metric_rows(
            metrics,
            exclude_truth_labels=exclude_truth_labels,
        )
    )


def _report_overall_accuracy(
    metrics: TaskMetrics,
    *,
    exclude_truth_labels: Optional[Sequence[str]] = None,
) -> float:
    rows = _report_metric_rows(metrics, exclude_truth_labels=exclude_truth_labels)
    total = sum(support for _, support, _, _ in rows)
    if total <= 0:
        return 0.0
    correct = sum(correct for _, _, correct, _ in rows)
    return float(correct / total)


def _render_metrics_table(
    metrics: TaskMetrics,
    *,
    exclude_truth_labels: Optional[Sequence[str]] = None,
) -> str:
    rows = [
        (label, support, acc)
        for label, support, _, acc in _report_metric_rows(
            metrics,
            exclude_truth_labels=exclude_truth_labels,
        )
    ]
    def sort_key(row):
        label, support, acc = row
        if math.isnan(acc):
            return (1, 0.0, -support, label)
        return (0, -acc, -support, label)
    rows.sort(key=sort_key)
    lines = ["| Label | Support | Accuracy |", "| --- | ---: | ---: |"]
    for label, support, acc in rows:
        acc_str = "nan" if math.isnan(acc) else f"{acc:.4f}"
        lines.append(f"| {label} | {support} | {acc_str} |")
    return "\n".join(lines)


def _format_pct(value: Optional[float], *, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    digits = max(0, int(digits))
    return f"{value * 100:.{digits}f}%"


def _format_float(value: Optional[float], *, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _format_hits(hits: int, total: int) -> str:
    if total <= 0:
        return "—"
    return f"{hits}/{total}"


def _summarize_confusions(
    metrics: TaskMetrics,
    *,
    limit: int = 5,
    min_rate: float = 0.05,
    min_count: int = 5,
    exclude_true_labels: Optional[Sequence[str]] = None,
) -> str:
    entries: List[Tuple[int, float, str, str]] = []
    confusion = metrics.confusion
    exclude_true_set = set(exclude_true_labels or [])
    for true_idx, true_label in enumerate(metrics.label_names):
        if true_label in exclude_true_set:
            continue
        row_total = int(confusion[true_idx].sum())
        if row_total <= 0:
            continue
        for pred_idx, pred_label in enumerate(metrics.label_names):
            if pred_idx == true_idx:
                continue
            count = int(confusion[true_idx, pred_idx])
            if count <= 0:
                continue
            rate = count / row_total
            entries.append((count, rate, true_label, pred_label))
    if not entries:
        return "none"
    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    filtered = [item for item in entries if item[0] >= min_count or item[1] >= min_rate]
    chosen = filtered[:limit] if filtered else entries[:limit]
    return ", ".join(f"{true}->{pred} {count} ({rate * 100:.1f}%)" for count, rate, true, pred in chosen)


def _other_confusion_counts(metrics: TaskMetrics) -> Dict[str, int]:
    confusion = np.asarray(metrics.confusion, dtype=np.int64)
    total = int(confusion.sum())
    try:
        other_idx = metrics.label_names.index("other")
    except ValueError:
        return {
            "tp_other": 0,
            "fn_other": 0,
            "fp_other": 0,
            "tn_other": total,
            "truth_other": 0,
            "truth_non_other": total,
            "predicted_other": 0,
            "predicted_non_other": total,
            "total": total,
        }

    tp_other = int(confusion[other_idx, other_idx])
    truth_other = int(confusion[other_idx, :].sum())
    predicted_other = int(confusion[:, other_idx].sum())
    fn_other = truth_other - tp_other
    fp_other = predicted_other - tp_other
    tn_other = total - tp_other - fn_other - fp_other
    return {
        "tp_other": tp_other,
        "fn_other": fn_other,
        "fp_other": fp_other,
        "tn_other": tn_other,
        "truth_other": truth_other,
        "truth_non_other": max(0, total - truth_other),
        "predicted_other": predicted_other,
        "predicted_non_other": max(0, total - predicted_other),
        "total": total,
    }


def _aggregate_other_confusion(task_metrics: Sequence[TaskMetrics]) -> Dict[str, Any]:
    totals = {
        "tp_other": 0,
        "fn_other": 0,
        "fp_other": 0,
        "tn_other": 0,
        "truth_other": 0,
        "truth_non_other": 0,
        "predicted_other": 0,
        "predicted_non_other": 0,
        "total": 0,
    }
    for metrics in task_metrics:
        counts = _other_confusion_counts(metrics)
        for key in totals:
            totals[key] += int(counts.get(key, 0))

    rates = {
        "too_often_other": _float_or_none(_safe_ratio(totals["fp_other"], totals["truth_non_other"])),
        "too_little_other": _float_or_none(_safe_ratio(totals["fn_other"], totals["truth_other"])),
        "other_precision": _float_or_none(_safe_ratio(totals["tp_other"], totals["predicted_other"])),
        "other_recall": _float_or_none(_safe_ratio(totals["tp_other"], totals["truth_other"])),
        "truth_other_rate": _float_or_none(_safe_ratio(totals["truth_other"], totals["total"])),
        "predicted_other_rate": _float_or_none(_safe_ratio(totals["predicted_other"], totals["total"])),
    }
    return {"counts": totals, "rates": rates}


def _manifest_task_entry(manifest: Optional[Mapping[str, Any]], task_name: str) -> Dict[str, Any]:
    if not isinstance(manifest, Mapping):
        return {}
    tasks = manifest.get("tasks")
    if not isinstance(tasks, Sequence):
        return {}
    for entry in tasks:
        if isinstance(entry, Mapping) and str(entry.get("task", "")) == task_name:
            return dict(entry)
    return {}


def _task_support_payload(manifest: Optional[Mapping[str, Any]], task_name: str) -> Dict[str, Any]:
    entry = _manifest_task_entry(manifest, task_name)
    if not entry:
        return {}
    keys = (
        "requested_count",
        "requested_per_anchor_label",
        "actual_count",
        "actual_by_anchor_label",
        "candidate_regions_by_anchor_label",
        "shortfall_by_anchor_label",
        "donor_candidate_regions_by_anchor_label",
        "donor_candidate_regions_by_donor_label",
        "donor_candidate_visible_support_by_donor_label",
        "donor_selected_visible_support_by_donor_label",
        "donor_actual_by_donor_label",
        "donor_support_shortfall_by_donor_label",
        "donor_floor_visible_chars",
        "qualified_donor_labels",
        "other_candidate_regions_by_anchor_label",
        "inline_candidate_regions_by_anchor_label",
    )
    return {key: entry.get(key) for key in keys if key in entry}


def _support_summary_lines(support: Mapping[str, Any]) -> List[str]:
    if not support:
        return []
    actual_count = int(support.get("actual_count", 0))
    requested_count = int(support.get("requested_count", 0))
    lines = [f"Support: {actual_count}/{requested_count} requested examples."]
    shortfall = support.get("shortfall_by_anchor_label")
    if isinstance(shortfall, Mapping):
        nonzero = [
            (str(label), int(value))
            for label, value in shortfall.items()
            if int(value) > 0
        ]
        nonzero.sort(key=lambda item: (-item[1], item[0]))
        if nonzero:
            joined = ", ".join(f"{label} -{value}" for label, value in nonzero[:6])
            lines.append(f"Shortfall by anchor: {joined}.")
    donor_floor = int(support.get("donor_floor_visible_chars", 0) or 0)
    if donor_floor > 0:
        lines.append(
            f"Needle donor floor: {donor_floor} visible chars; aggregate coverage is support-weighted over inserted-region chars."
        )
        qualified = support.get("qualified_donor_labels")
        if isinstance(qualified, Sequence) and not isinstance(qualified, (str, bytes, bytearray)):
            lines.append(f"Qualified donor labels: {len(list(qualified))}.")
        donor_shortfall = support.get("donor_support_shortfall_by_donor_label")
        if isinstance(donor_shortfall, Mapping):
            nonzero_shortfall = [
                (str(label), int(value))
                for label, value in donor_shortfall.items()
                if int(value) > 0
            ]
            nonzero_shortfall.sort(key=lambda item: (-item[1], item[0]))
            if nonzero_shortfall:
                joined = ", ".join(f"{label} -{value}" for label, value in nonzero_shortfall[:6])
                lines.append(f"Donor support shortfall: {joined}.")
    return lines


def _collect_task_highlights(
    task_metrics: List[TaskMetrics],
    manifest: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    metrics_by_name = {m.name: m for m in task_metrics}
    sections: List[str] = []

    def add_section(title: str, body_lines: List[str]) -> None:
        sections.append(f"##### {title}")
        sections.extend(body_lines)
        if not body_lines or body_lines[-1] != "":
            sections.append("")

    other_summary = _aggregate_other_confusion(task_metrics)
    other_counts = other_summary.get("counts", {})
    other_rates = other_summary.get("rates", {})
    truth_other = int(other_counts.get("truth_other", 0))
    truth_non_other = int(other_counts.get("truth_non_other", 0))
    total_other_chars = int(other_counts.get("total", 0))
    if total_other_chars > 0:
        table_lines = [
            "| Truth \\\\ Pred | other | not other |",
            "| --- | --- | --- |",
            (
                f"| other | {int(other_counts.get('tp_other', 0))} ({_format_pct(_safe_ratio(other_counts.get('tp_other', 0), truth_other))}) | "
                f"{int(other_counts.get('fn_other', 0))} ({_format_pct(_safe_ratio(other_counts.get('fn_other', 0), truth_other))}) |"
            ),
            (
                f"| not other | {int(other_counts.get('fp_other', 0))} ({_format_pct(_safe_ratio(other_counts.get('fp_other', 0), truth_non_other))}) | "
                f"{int(other_counts.get('tn_other', 0))} ({_format_pct(_safe_ratio(other_counts.get('tn_other', 0), truth_non_other))}) |"
            ),
            "",
            (
                f"Too often `other`: {_format_pct(other_rates.get('too_often_other'))}. "
                f"Too little `other`: {_format_pct(other_rates.get('too_little_other'))}."
            ),
        ]
        add_section("other_open_set", table_lines)

    # mal_injection summary
    mal_metrics = metrics_by_name.get("mal_injection")
    if mal_metrics:
        payload_stats = mal_metrics.extras.get("mal_payload_detection", {}) if mal_metrics.extras else {}
        any_stats = payload_stats.get("any_detection", {})
        any_total = int(any_stats.get("count", 0))
        any_detected = int(any_stats.get("detected", 0))
        any_truth = int(any_stats.get("truth_chars", 0))
        any_correct = int(any_stats.get("correct_chars", 0))
        any_avg_iou = _safe_ratio(any_stats.get("iou_sum", 0.0), any_total)
        any_cov = _format_pct(_safe_ratio(any_correct, any_truth))

        by_lang = payload_stats.get("by_lang", {})
        correct_total = sum(int(entry.get("count", 0)) for entry in by_lang.values())
        correct_detected = sum(int(entry.get("detected", 0)) for entry in by_lang.values())
        correct_truth = sum(int(entry.get("truth_chars", 0)) for entry in by_lang.values())
        correct_chars = sum(int(entry.get("correct_chars", 0)) for entry in by_lang.values())
        correct_avg_iou = _safe_ratio(
            sum(float(entry.get("iou_sum", 0.0)) for entry in by_lang.values()),
            correct_total,
        )
        correct_cov = _format_pct(_safe_ratio(correct_chars, correct_truth))

        any_cov_hits = int(any_stats.get("coverage_hits", 0))
        any_cov_total = int(any_stats.get("coverage_count", 0))
        correct_cov_hits = sum(int(entry.get("coverage_hits", 0)) for entry in by_lang.values())
        correct_cov_total = sum(int(entry.get("coverage_count", 0)) for entry in by_lang.values())
        table_lines = [
            "| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Avg coverage |",
            "| --- | --- | --- | --- | --- |",
            f"| Any non-wrapper | {_format_hits(any_cov_hits, any_cov_total)} | {_format_hits(any_detected, any_total)} | {_format_float(any_avg_iou)} | {any_cov} |",
            f"| Correct payload | {_format_hits(correct_cov_hits, correct_cov_total)} | {_format_hits(correct_detected, correct_total)} | {_format_float(correct_avg_iou)} | {correct_cov} |",
        ]
        soft_stats = payload_stats.get("soft_detection", {})
        soft_prob = float(soft_stats.get("prob_threshold", PAYLOAD_SOFT_PROB_THRESHOLD))
        soft_cov_threshold = float(soft_stats.get("coverage_threshold", PAYLOAD_SOFT_COVERAGE_THRESHOLD))
        soft_any = soft_stats.get("any_detection", {})
        soft_correct = soft_stats.get("correct_detection", {})
        soft_any_total = int(soft_any.get("count", 0))
        soft_correct_total = int(soft_correct.get("count", 0))
        if soft_any_total or soft_correct_total:
            soft_any_cov = _format_pct(_safe_ratio(soft_any.get("matched_chars", 0), soft_any.get("truth_chars", 0)))
            soft_correct_cov = _format_pct(
                _safe_ratio(soft_correct.get("matched_chars", 0), soft_correct.get("truth_chars", 0))
            )
            soft_label = (
                f"Soft anomaly hit = >={soft_cov_threshold * 100:.0f}% payload chars "
                f"with target prob >={soft_prob * 100:.0f}%."
            )
            table_lines.extend(
                [
                    "",
                    soft_label,
                    "",
                    "| Scenario | Soft hit | Avg soft coverage |",
                    "| --- | --- | --- |",
                    f"| Any non-wrapper >=5% | {_format_hits(int(soft_any.get('detected', 0)), soft_any_total)} | {soft_any_cov} |",
                    f"| Correct payload >=5% | {_format_hits(int(soft_correct.get('detected', 0)), soft_correct_total)} | {soft_correct_cov} |",
                ]
            )
        add_section("mal_injection", table_lines)

    # markdown highlight
    markdown_metrics = metrics_by_name.get("markdown_mix")
    if markdown_metrics:
        md_stats = markdown_metrics.extras.get("markdown_segments", {}) if markdown_metrics.extras else {}
        overall = md_stats.get("overall", {})
        inline_stats = md_stats.get("inline", {})
        text_stats_overall = md_stats.get("text", {})
        text_like_binary = md_stats.get("text_like_binary", {})
        boundary_payload = _needle_boundary_metrics_payload(
            md_stats.get("boundary_region", {}),
            label_names=markdown_metrics.label_names,
        )
        support_lines = _support_summary_lines(_task_support_payload(manifest, "markdown_mix"))

        def _wrapper_row(name: str, group: Dict[str, Any]) -> str:
            count = int(group.get("count", 0))
            if count <= 0:
                return f"| {name} | — | — | — | — | — |"
            nontext_cov_hits = int(group.get("detected_nontext", 0))
            text_hits = int(group.get("detected_text", 0))
            correct_cov_hits = int(group.get("detected_correct", 0))
            truth_chars = int(group.get("truth_chars", 0))
            nontext_chars = int(group.get("nontext_chars", 0))
            correct_chars = int(group.get("correct_chars", 0))
            nontext_cov = _format_pct(_safe_ratio(nontext_chars, truth_chars))
            correct_cov = _format_pct(_safe_ratio(correct_chars, truth_chars))
            return (
                f"| {name} | {_format_hits(nontext_cov_hits, count)} | {nontext_cov} | "
                f"{_format_hits(text_hits, count)} | {_format_hits(correct_cov_hits, count)} | {correct_cov} |"
            )

        wrapper_table = [
            *support_lines,
            *([""] if support_lines else []),
            "_Text hits column: lower is better._",
            "| Wrapper | Non-text cov ≥50% | Non-text region coverage | Text hits | Exact region ≥50% | Exact region avg coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            _wrapper_row(MARKDOWN_FENCED_LABEL, overall.get("wrapped", {})),
            _wrapper_row("bare code", overall.get("plain", {})),
        ]

        inline_count = int(inline_stats.get("count", 0))
        if inline_count > 0:
            inline_group = {
                "count": inline_count,
                "detected_nontext": inline_stats.get("nontext_hits", 0),
                "detected_nontext_iou": inline_stats.get("nontext_iou_hits", 0),
                "detected_text": inline_stats.get("text_hits", 0),
                "detected_correct": inline_stats.get("correct_hits", 0),
                "detected_correct_iou": inline_stats.get("correct_iou_hits", 0),
                "truth_chars": inline_stats.get("truth_chars", 0),
                "nontext_chars": inline_stats.get("nontext_chars", 0),
                "correct_chars": inline_stats.get("correct_chars", 0),
                "union_nontext_chars": inline_stats.get("union_nontext_chars", 0),
                "union_correct_chars": inline_stats.get("union_correct_chars", 0),
            }
            wrapper_table.append(_wrapper_row(MARKDOWN_INLINE_LABEL, inline_group))

        text_cov_line = None
        text_truth = int(text_stats_overall.get("truth_chars", 0))
        if text_truth > 0:
            text_cov_line = f"Text region coverage: {_format_pct(_safe_ratio(text_stats_overall.get('correct_chars', 0), text_truth))}"
            wrapper_table.append("")
            wrapper_table.append(text_cov_line)

        if boundary_payload is not None:
            aggregates = boundary_payload.get("aggregates", {})
            wrapper_table.extend(
                [
                    "",
                    (
                        "Boundary-region label metrics "
                        f"(`±{int(boundary_payload.get('window_radius_tokens', NEEDLE_BOUNDARY_WINDOW_TOKENS))}` "
                        f"tokens around markdown/code transitions): acc {float(aggregates.get('micro_acc', 0.0)):.4f}, "
                        f"prec {float(aggregates.get('macro_precision', 0.0)):.4f}, "
                        f"recall {float(aggregates.get('macro_recall', 0.0)):.4f}, "
                        f"f1 {float(aggregates.get('macro_f1', 0.0)):.4f}."
                    ),
                ]
            )

        _append_text_like_binary_lines(wrapper_table, text_like_binary)
        add_section("markdown_mix", wrapper_table)

    # reStructuredText highlight (focus on foreign code sections, not fences)
    rst_metrics = metrics_by_name.get("restructuredtext_mix")
    if rst_metrics:
        rst_stats = rst_metrics.extras.get("restructuredtext_segments", {}) if rst_metrics.extras else {}
        text_stats_overall = rst_stats.get("text", {})
        text_like_binary = rst_stats.get("text_like_binary", {})
        per_language = rst_stats.get("per_language", {})
        support_lines = _support_summary_lines(_task_support_payload(manifest, "restructuredtext_mix"))

        rows: List[str] = [*support_lines]
        if rows:
            rows.append("")
        if per_language:
            header = [
                "_Text hits column: lower is better._",
                "| Foreign language | Blocks | Exact region ≥50% | Exact region avg coverage |",
                "| --- | ---: | ---: | ---: |",
            ]
            rows.extend(header)
            for lang, groups in sorted(per_language.items()):
                plain = groups.get("plain", {}) or {}
                wrapped = groups.get("wrapped", {}) or {}
                count_plain = int(plain.get("count", 0))
                count_wrapped = int(wrapped.get("count", 0))
                count = count_plain + count_wrapped
                if count <= 0:
                    continue
                hits = int(plain.get("detected_correct", 0)) + int(wrapped.get("detected_correct", 0))
                truth_chars = int(plain.get("truth_chars", 0)) + int(wrapped.get("truth_chars", 0))
                correct_chars = int(plain.get("correct_chars", 0)) + int(wrapped.get("correct_chars", 0))
                coverage = _format_pct(_safe_ratio(correct_chars, truth_chars))
                rows.append(
                    f"| {lang} | {count} | {_format_hits(hits, count)} | {coverage} |"
                )

        text_cov_line = None
        text_truth = int(text_stats_overall.get("truth_chars", 0))
        if text_truth > 0:
            text_cov_line = f"Text region coverage: {_format_pct(_safe_ratio(text_stats_overall.get('correct_chars', 0), text_truth))}"
            if rows:
                rows.append("")
            rows.append(text_cov_line)

        _append_text_like_binary_lines(rows, text_like_binary)
        if rows:
            add_section("restructuredtext_mix", rows)

    # pure fragments highlight
    pure_metrics = metrics_by_name.get("pure_fragments")
    if pure_metrics:
        pure_stats = pure_metrics.extras.get("pure_fragments_purity", {}) if pure_metrics.extras else {}
        total = int(pure_stats.get("total", 0))
        perfect = int(pure_stats.get("perfect", 0))
        within = int(pure_stats.get("within_threshold", perfect))
        threshold = float(pure_stats.get("threshold", 0.5))
        ratios = pure_stats.get("foreign_ratios", [])
        summary_lines = [
            f"{perfect}/{total} host-bearing samples had no misclassified host-label bytes. "
            f"{within}/{total} stayed within ≤{threshold:.0%} host-byte error rate.",
        ]
        if ratios:
            mean_ratio = float(sum(ratios)) / max(1, len(ratios))
            expected_bytes = mean_ratio * 1536
            summary_lines.append(
                f"Expected misclassified host bytes for 1536 host-labeled bytes: {expected_bytes:.1f}/1536"
            )
        add_section("pure_fragments", summary_lines)

    # sequence tasks
    for name in ("sequence_pair", "sequence_triplet"):
        seq_metrics = metrics_by_name.get(name)
        if not seq_metrics:
            continue
        seq_stats = seq_metrics.extras.get("sequence_purity", {}) if seq_metrics.extras else {}
        segments = seq_stats.get("segments", {})
        segment_lines: List[str] = _support_summary_lines(_task_support_payload(manifest, name))
        for key in ("first", "second", "third"):
            data = segments.get(key)
            if not data:
                continue
            total = int(data.get("total", 0))
            correct = int(data.get("correct", 0))
            coverage = _format_pct(correct / total) if total > 0 else "n/a"
            segment_lines.append(f"{key.title()} exact region coverage {coverage}")
        boundary_payload = _sequence_boundary_metrics_payload(
            seq_stats.get("boundary_region", {}),
            label_names=seq_metrics.label_names,
        )
        if boundary_payload is not None:
            aggregates = boundary_payload.get("aggregates", {})
            segment_lines.append(
                "Boundary-region label metrics "
                f"(`±{int(boundary_payload.get('window_radius_tokens', NEEDLE_BOUNDARY_WINDOW_TOKENS))}` "
                f"tokens around sequence transitions): acc {float(aggregates.get('micro_acc', 0.0)):.4f}, "
                f"prec {float(aggregates.get('macro_precision', 0.0)):.4f}, "
                f"recall {float(aggregates.get('macro_recall', 0.0)):.4f}, "
                f"f1 {float(aggregates.get('macro_f1', 0.0)):.4f}."
            )
        if segment_lines:
            add_section(name, segment_lines)

    # Needle buckets (individual sections, keep at end)
    needle_metrics = [m for m in task_metrics if _needle_bucket_key(m.name) is not None]
    needle_metrics.sort(key=lambda m: (_needle_bucket_key(m.name) or (0, 0))[0], reverse=True)
    for m in needle_metrics:
        stats = m.extras.get("needle_detection", {}) if m.extras else {}
        if not stats:
            continue
        support_lines = _support_summary_lines(_task_support_payload(manifest, m.name))
        total = int(stats.get("total", 0))
        detected = int(stats.get("detected", 0))
        coverage_hits = int(stats.get("coverage_hits", 0))
        coverage_total = int(stats.get("coverage_count", 0))
        avg_iou = _safe_ratio(stats.get("iou_sum", 0.0), total)
        coverage_pct = _format_pct(_safe_ratio(stats.get("correct_chars", 0), stats.get("needle_chars", 0)))

        any_stats = stats.get("any_detection", {})
        any_cov_hits = int(any_stats.get("coverage_hits", 0))
        any_cov_total = int(any_stats.get("coverage_count", 0))
        any_count = int(any_stats.get("count", 0))
        any_detected = int(any_stats.get("detected", 0))
        any_avg_iou = _safe_ratio(any_stats.get("iou_sum", 0.0), any_count)
        any_cov = _format_pct(_safe_ratio(any_stats.get("correct_chars", 0), any_stats.get("truth_chars", 0)))

        section_lines = [
            *support_lines,
            *([""] if support_lines else []),
            "| Scenario | Coverage ≥50% | Exact region ≥50% | Avg exact region coverage | Coverage |",
            "| --- | --- | --- | --- | --- |",
            f"| Any non-wrapper | {_format_hits(any_cov_hits, any_cov_total)} | {_format_hits(any_detected, any_count)} | {_format_float(any_avg_iou)} | {any_cov} |",
            f"| Exact inserted region | {_format_hits(coverage_hits, coverage_total)} | {_format_hits(detected, total)} | {_format_float(avg_iou)} | {coverage_pct} |",
        ]
        boundary_payload = _needle_boundary_metrics_payload(
            stats.get("boundary_region", {}),
            label_names=m.label_names,
        )
        if boundary_payload is not None:
            aggregates = boundary_payload.get("aggregates", {})
            section_lines.extend(
                [
                    "",
                    (
                        "Boundary-region label metrics "
                        f"(`±{int(boundary_payload.get('window_radius_tokens', NEEDLE_BOUNDARY_WINDOW_TOKENS))}` "
                        "tokens around inserted start/end):"
                    ),
                    "",
                    "| Acc | Prec | Recall | F1 | Boundary chars | Samples |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: |",
                    (
                        f"| {float(aggregates.get('micro_acc', 0.0)):.4f} | "
                        f"{float(aggregates.get('macro_precision', 0.0)):.4f} | "
                        f"{float(aggregates.get('macro_recall', 0.0)):.4f} | "
                        f"{float(aggregates.get('macro_f1', 0.0)):.4f} | "
                        f"{int(boundary_payload.get('support_chars', 0))} | "
                        f"{int(boundary_payload.get('samples', 0))} |"
                    ),
                ]
            )
        add_section(m.name, section_lines)

    return sections


def _render_throughput_table(results: List[ThroughputResult]) -> str:
    lines = [
        "| Task | Device | Samples | Total Bytes | Throughput | Windows/s | Latency (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for res in results:
        windows_text = "--" if str(res.arch).lower().strip() == "mamba" else f"{float(res.windows_per_second):.1f}"
        lines.append(
            "| {task} | {device} | {samples} | {bytes} | {through} | {windows} | {lat:.2f} |".format(
                task=res.task,
                device=res.device,
                samples=res.samples,
                bytes=res.total_bytes,
                through=_format_bytes_per_sec(res.throughput),
                windows=windows_text,
                lat=res.elapsed,
            )
        )
    lines.append("")
    lines.append(
        "Latency is the total wall-clock time to run the benchmark loop over all samples "
        "per benchmark type, excluding the one warmup inference call that triggers JAX’s "
        "JIT compilation beforehand."
    )
    return "\n".join(lines)


def _comparison_key_from_checkpoint(checkpoint_path: str) -> str:
    parsed = urlparse(checkpoint_path)
    if parsed.scheme and parsed.scheme != "file":
        candidate = (parsed.netloc + parsed.path) or checkpoint_path
    else:
        candidate = checkpoint_path
    candidate = candidate.rstrip("/").split("/")[-1] or candidate
    if "." in candidate:
        candidate = candidate.rsplit(".", 1)[0]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
    return sanitized or "model"


def _sanitize_report_component(value: Any) -> str:
    text = str(value or "").strip()
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    sanitized = sanitized.strip("._-")
    return sanitized or "item"


def _task_scope_key(tasks: Optional[Sequence[str]]) -> str:
    if not tasks:
        return "all_tasks"
    parts = [_sanitize_report_component(task) for task in tasks if str(task).strip()]
    if not parts:
        return "all_tasks"
    if len(parts) <= 3:
        return "__".join(parts)
    return f"{len(parts)}tasks"


def _dedupe_report_dir(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 2
    while True:
        candidate = path.parent / f"{path.name}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _default_report_dir_name(args) -> str:
    checkpoint_key = _comparison_key_from_checkpoint(str(getattr(args, "checkpoint", "model")))
    data_root_value = getattr(args, "data_root", "data")
    data_root_name = Path(str(data_root_value)).expanduser().name or "data"
    task_key = _task_scope_key(getattr(args, "tasks", None))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parts = [checkpoint_key, _sanitize_report_component(data_root_name)]
    if task_key != "all_tasks":
        parts.append(task_key)
    parts.append(timestamp)
    return "__".join(parts)


def _resolve_report_path(args) -> Path:
    raw_path = getattr(args, "report_path", None)
    if raw_path:
        requested = Path(str(raw_path)).expanduser()
        suffix = requested.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return requested.resolve()
        return requested.resolve() / "report.md"

    report_dir = _dedupe_report_dir((DEFAULT_REPORTS_DIR / _default_report_dir_name(args)).resolve())
    return report_dir / "report.md"


def _comparison_metrics_path(report_path: Path) -> Path:
    return report_path.parent / "comparison_metrics.json"


def _confusion_artifacts_dir(report_path: Path) -> Path:
    return report_path.parent / "confusion_matrices"


def _confusion_image_name(name: str) -> str:
    return f"{_sanitize_report_component(name)}.png"


def _merge_task_confusions(task_metrics: Sequence[TaskMetrics]) -> Tuple[np.ndarray, List[str]]:
    label_pool: Set[str] = {"other"}
    for metrics in task_metrics:
        label_pool.update(str(label) for label in metrics.label_names)
    label_names, label_to_idx = _confusion_size(label_pool)
    merged = np.zeros((len(label_names), len(label_names)), dtype=np.int64)
    for metrics in task_metrics:
        local_label_to_idx = {label: idx for idx, label in enumerate(metrics.label_names)}
        for true_label, true_idx in local_label_to_idx.items():
            for pred_label, pred_idx in local_label_to_idx.items():
                merged[label_to_idx[true_label], label_to_idx[pred_label]] += int(metrics.confusion[true_idx, pred_idx])
    return merged, label_names


def _write_confusion_matrix_plot(
    confusion: np.ndarray,
    label_names: Sequence[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    if confusion.size == 0 or not label_names:
        return
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    conf_counts = np.asarray(confusion, dtype=np.float64)
    row_sums = conf_counts.sum(axis=1, keepdims=True)
    conf_display = np.divide(
        conf_counts,
        np.where(row_sums > 0.0, row_sums, 1.0),
        out=np.zeros_like(conf_counts, dtype=np.float64),
        where=np.ones_like(conf_counts, dtype=bool),
    )

    num_labels = len(label_names)
    figsize = (max(6.0, min(18.0, 1.0 + 0.7 * num_labels)), max(5.0, min(16.0, 1.5 + 0.6 * num_labels)))
    fig = Figure(figsize=figsize)
    _ = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(conf_display, interpolation="nearest", aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ticks = np.arange(num_labels)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    for tick in ax.get_xticklabels():
        tick.set_rotation_mode("anchor")

    annotate = num_labels <= 18
    if annotate:
        for row_idx in range(num_labels):
            row_total = float(row_sums[row_idx, 0])
            for col_idx in range(num_labels):
                count = int(conf_counts[row_idx, col_idx])
                if count <= 0 and row_total <= 0.0:
                    continue
                ratio = float(conf_display[row_idx, col_idx])
                text = f"{count}\n{ratio * 100:.1f}%"
                text_color = "white" if ratio >= 0.5 else "black"
                ax.text(col_idx, row_idx, text, ha="center", va="center", color=text_color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized share", rotation=270, labelpad=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.clear()


def _monitor_b_confusion_payload(
    monitor_b_report: Optional[Mapping[str, Any]],
) -> Optional[Tuple[np.ndarray, List[str]]]:
    if not monitor_b_report:
        return None
    raw_confusion = monitor_b_report.get("_confusion_matrix")
    if raw_confusion is None:
        return None
    confusion = np.asarray(raw_confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1] or confusion.size == 0:
        return None

    keep_indices: List[int] = []
    label_names: List[str] = []
    id2label = _monitor_b_id2label()
    for idx in range(int(confusion.shape[0])):
        if idx == int(cfg.PAD_ID):
            continue
        if int(confusion[idx, :].sum()) <= 0 and int(confusion[:, idx].sum()) <= 0:
            continue
        keep_indices.append(idx)
        label_names.append(str(id2label.get(idx, str(idx))))

    if not keep_indices or not label_names:
        return None

    keep = np.asarray(keep_indices, dtype=np.int64)
    reduced = confusion[np.ix_(keep, keep)]
    if reduced.size == 0 or int(reduced.sum()) <= 0:
        return None
    return reduced, label_names


def _write_confusion_artifacts(
    report_path: Path,
    task_metrics: Sequence[TaskMetrics],
    monitor_b_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    confusion_dir = _confusion_artifacts_dir(report_path)
    confusion_dir.mkdir(parents=True, exist_ok=True)

    for metrics in task_metrics:
        output_path = confusion_dir / _confusion_image_name(metrics.name)
        _write_confusion_matrix_plot(
            metrics.confusion,
            metrics.label_names,
            output_path,
            title=f"{metrics.name} confusion matrix",
        )
        artifacts[metrics.name] = output_path.relative_to(report_path.parent).as_posix()

    if task_metrics:
        merged_confusion, merged_labels = _merge_task_confusions(task_metrics)
        output_path = confusion_dir / _confusion_image_name("all_tasks")
        _write_confusion_matrix_plot(
            merged_confusion,
            merged_labels,
            output_path,
            title="All tasks confusion matrix",
        )
        artifacts["all_tasks"] = output_path.relative_to(report_path.parent).as_posix()

    monitor_payload = _monitor_b_confusion_payload(monitor_b_report)
    if monitor_payload is not None:
        monitor_confusion, monitor_labels = monitor_payload
        output_path = confusion_dir / _confusion_image_name("monitor_b")
        _write_confusion_matrix_plot(
            monitor_confusion,
            monitor_labels,
            output_path,
            title="Full monitor_b confusion matrix",
        )
        artifacts["monitor_b"] = output_path.relative_to(report_path.parent).as_posix()

    return artifacts


def _collect_comparison_metrics(
    args,
    task_metrics: List[TaskMetrics],
    throughput_results: List[ThroughputResult],
    manifest: Optional[Mapping[str, Any]] = None,
    monitor_b_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    other_summary = _aggregate_other_confusion(task_metrics)
    metrics_by_name = {m.name: m for m in task_metrics}
    exclusion_note = _truth_label_exclusion_note(
        arch=str(getattr(args, "arch", "")),
        excluded_labels=getattr(args, "exclude_truth_labels", None),
    )
    data: Dict[str, Any] = {
        "meta": {
            "arch": getattr(args, "arch", None),
            "checkpoint": args.checkpoint,
            "model_dim": args.model_dim,
            "channels": list(args.channels) if isinstance(args.channels, (list, tuple)) else args.channels,
            "dtype": args.dtype,
            "sample_seed": args.sample_seed,
            "other_threshold": _float_or_none(getattr(args, "other_threshold", 0.0)),
            "postprocess_profile": getattr(args, "postprocess_profile", POSTPROCESS_PROFILE_OFF),
            "postprocess_min_run": int(getattr(args, "postprocess_min_run", 5)),
            "postprocess_boundary_snap_max_shift": int(
                getattr(args, "postprocess_boundary_snap_max_shift", 2)
            ),
            "postprocess_stages": (
                [
                    "whitespace_relabel",
                    "other_gating",
                    "boundary_snap",
                    "min_run",
                ]
                if getattr(args, "postprocess_profile", POSTPROCESS_PROFILE_OFF) == POSTPROCESS_PROFILE_THESIS
                else []
            ),
            "postprocess_excluded_viewer_stages": (
                [
                    "markdown_structure_fill",
                    "paired_delimiter_fill",
                    "local_host_fill",
                    "newline_snap",
                ]
                if getattr(args, "postprocess_profile", POSTPROCESS_PROFILE_OFF) == POSTPROCESS_PROFILE_THESIS
                else []
            ),
            "excluded_truth_labels": list(getattr(args, "exclude_truth_labels", []) or []),
            "truth_label_exclusion_note": exclusion_note,
            "magika_module_version": getattr(args, "magika_module_version", None),
            "magika_model_name": getattr(args, "magika_model_name", None),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": {
            "other_confusion": other_summary,
        },
        "tasks": {},
    }

    postprocess_summary: Dict[str, Any] = {}
    for metrics in task_metrics:
        stats = metrics.extras.get("postprocess") if metrics.extras else None
        if not isinstance(stats, Mapping):
            continue
        postprocess_summary[metrics.name] = {
            "profile": stats.get("profile"),
            "samples": int(stats.get("samples", 0) or 0),
            "processed_chars": int(stats.get("processed_chars", 0) or 0),
            "other_thresholded_chars": int(stats.get("other_thresholded_chars", 0) or 0),
            "boundary_snap_changed_chars": int(stats.get("boundary_snap_changed_chars", 0) or 0),
            "min_run_changed_chars": int(stats.get("min_run_changed_chars", 0) or 0),
            "changed_chars": int(stats.get("changed_chars", 0) or 0),
        }
    if postprocess_summary:
        data["summary"]["postprocess"] = postprocess_summary

    # Malicious injections -------------------------------------------------
    mal_metrics = metrics_by_name.get("mal_injection")
    if mal_metrics and mal_metrics.extras:
        payload_stats = mal_metrics.extras.get("mal_payload_detection", {})
        any_stats = payload_stats.get("any_detection", {})
        by_lang = payload_stats.get("by_lang", {})
        any_by_lang = payload_stats.get("any_by_lang", {})

        payload_total = sum(entry.get("count", 0) for entry in by_lang.values())
        payload_detected = sum(entry.get("detected", 0) for entry in by_lang.values())
        payload_iou_sum = sum(entry.get("iou_sum", 0.0) for entry in by_lang.values())
        payload_correct_chars = sum(entry.get("correct_chars", 0) for entry in by_lang.values())
        payload_truth_chars = sum(entry.get("truth_chars", 0) for entry in by_lang.values())

        result = {
            "inj.payload.det_rate@0.5": _float_or_none(_safe_ratio(payload_detected, payload_total)),
            "inj.payload.mean_iou": _float_or_none(_safe_ratio(payload_iou_sum, payload_total)),
            "inj.any.det_rate@0.5": _float_or_none(_safe_ratio(any_stats.get("detected", 0), any_stats.get("count", 0))),
            "inj.any.mean_iou": _float_or_none(_safe_ratio(any_stats.get("iou_sum", 0.0), any_stats.get("count", 0))),
        }

        payload_cov = _safe_ratio(payload_correct_chars, payload_truth_chars)
        if payload_cov is not None:
            result["inj.payload.coverage"] = _float_or_none(payload_cov)
        any_cov = _safe_ratio(any_stats.get("correct_chars", 0), any_stats.get("truth_chars", 0))
        if any_cov is not None:
            result["inj.any.coverage"] = _float_or_none(any_cov)

        per_lang: Dict[str, Any] = {}
        for lang, entry in sorted(by_lang.items()):
            per_lang[lang] = {
                "det_rate@0.5": _float_or_none(_safe_ratio(entry.get("detected", 0), entry.get("count", 0))),
                "mean_iou": _float_or_none(_safe_ratio(entry.get("iou_sum", 0.0), entry.get("count", 0))),
                "coverage": _float_or_none(_safe_ratio(entry.get("correct_chars", 0), entry.get("truth_chars", 0))),
                "support": int(entry.get("count", 0)),
            }
        if per_lang:
            result["per_language_correct"] = per_lang

        per_lang_any: Dict[str, Any] = {}
        for lang, entry in sorted(any_by_lang.items()):
            per_lang_any[lang] = {
                "det_rate@0.5": _float_or_none(_safe_ratio(entry.get("detected", 0), entry.get("count", 0))),
                "mean_iou": _float_or_none(_safe_ratio(entry.get("iou_sum", 0.0), entry.get("count", 0))),
                "coverage": _float_or_none(_safe_ratio(entry.get("correct_chars", 0), entry.get("truth_chars", 0))),
                "support": int(entry.get("count", 0)),
            }
        if per_lang_any:
            result["per_language_any"] = per_lang_any

        data["tasks"]["mal_injection"] = result

    # Markdown -------------------------------------------------------------
    markdown_metrics = metrics_by_name.get("markdown_mix")
    if markdown_metrics and markdown_metrics.extras:
        md_stats = markdown_metrics.extras.get("markdown_segments", {})
        md_data: Dict[str, Any] = {}

        per_language = md_stats.get("per_language", {})
        plain_per_lang: Dict[str, Any] = {}
        wrapped_per_lang: Dict[str, Any] = {}
        for lang, groups in per_language.items():
            for wrapper_key, target_dict in (("plain", plain_per_lang), ("wrapped", wrapped_per_lang)):
                group = groups.get(wrapper_key, {})
                count = int(group.get("count", 0))
                if count <= 0:
                    continue
                target_dict[lang] = {
                    "det_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_correct", 0), count)),
                    "coverage": _float_or_none(_safe_ratio(group.get("correct_chars", 0), group.get("truth_chars", 0))),
                    "nontext_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_nontext", 0), count)),
                    "nontext_coverage": _float_or_none(_safe_ratio(group.get("nontext_chars", 0), group.get("truth_chars", 0))),
                    "text_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_text", 0), count)),
                    "text_coverage": _float_or_none(_safe_ratio(group.get("text_chars", 0), group.get("truth_chars", 0))),
                    "support": count,
                }
        block_section: Dict[str, Any] = {}
        overall_block = {}
        for key, label in (("plain", "plain"), ("wrapped", "fenced")):
            group = md_stats.get("overall", {}).get(key, {})
            count = int(group.get("count", 0))
            if count <= 0:
                continue
            overall_block[label] = {
                "det_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_correct", 0), count)),
                "coverage": _float_or_none(_safe_ratio(group.get("correct_chars", 0), group.get("truth_chars", 0))),
                "nontext_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_nontext", 0), count)),
                "nontext_coverage": _float_or_none(_safe_ratio(group.get("nontext_chars", 0), group.get("truth_chars", 0))),
                "text_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_text", 0), count)),
                "text_coverage": _float_or_none(_safe_ratio(group.get("text_chars", 0), group.get("truth_chars", 0))),
                "support": count,
            }
        if overall_block:
            block_section["overall"] = overall_block
        if wrapped_per_lang:
            block_section["fenced_per_language"] = wrapped_per_lang
        if plain_per_lang:
            block_section["plain_per_language"] = plain_per_lang
        if block_section:
            md_data["block"] = block_section

        inline_stats = md_stats.get("inline", {})
        inline_per_lang = inline_stats.get("per_language", {})
        inline_section: Dict[str, Any] = {}
        inline_count = int(inline_stats.get("count", 0))
        if inline_count > 0:
            inline_section["overall"] = {
                "det_rate@0.5": _float_or_none(_safe_ratio(inline_stats.get("correct_hits", 0), inline_count)),
                "coverage": _float_or_none(_safe_ratio(inline_stats.get("correct_chars", 0), inline_stats.get("truth_chars", 0))),
                "nontext_rate@0.5": _float_or_none(_safe_ratio(inline_stats.get("nontext_hits", 0), inline_count)),
                "nontext_coverage": _float_or_none(_safe_ratio(inline_stats.get("nontext_chars", 0), inline_stats.get("truth_chars", 0))),
                "text_rate@0.5": _float_or_none(_safe_ratio(inline_stats.get("text_hits", 0), inline_count)),
                "text_coverage": _float_or_none(_safe_ratio(inline_stats.get("text_chars", 0), inline_stats.get("truth_chars", 0))),
                "support": inline_count,
            }
        if inline_per_lang:
            inline_data = {}
            for lang, entry in inline_per_lang.items():
                count = int(entry.get("count", 0))
                if count <= 0:
                    continue
                inline_data[lang] = {
                    "det_rate@0.5": _float_or_none(_safe_ratio(entry.get("correct_hits", 0), count)),
                    "coverage": _float_or_none(_safe_ratio(entry.get("correct_chars", 0), entry.get("truth_chars", 0))),
                    "nontext_rate@0.5": _float_or_none(_safe_ratio(entry.get("nontext_hits", 0), count)),
                    "nontext_coverage": _float_or_none(_safe_ratio(entry.get("nontext_chars", 0), entry.get("truth_chars", 0))),
                    "text_rate@0.5": _float_or_none(_safe_ratio(entry.get("text_hits", 0), count)),
                    "text_coverage": _float_or_none(_safe_ratio(entry.get("text_chars", 0), entry.get("truth_chars", 0))),
                    "support": count,
                }
            if inline_data:
                inline_section["per_language"] = inline_data
        if inline_section:
            md_data["inline"] = inline_section

        wrong = md_stats.get("wrong_label", {})
        wrong_cases = int(wrong.get("cases", 0))
        if wrong_cases > 0:
            md_data["wrong_fence_fooled_rate"] = _float_or_none(_safe_ratio(wrong.get("fooled", 0), wrong_cases))

        text_stats = md_stats.get("text", {})
        text_truth = int(text_stats.get("truth_chars", 0))
        union_chars = int(text_stats.get("union_chars", 0))
        if text_truth or union_chars:
            md_data["text_accuracy"] = _float_or_none(_safe_ratio(text_stats.get("correct_chars", 0), text_truth))
            md_data["text_iou"] = _float_or_none(_safe_ratio(text_stats.get("correct_chars", 0), union_chars))
        text_like_binary = md_stats.get("text_like_binary", {})
        if isinstance(text_like_binary, Mapping) and text_like_binary.get("rows"):
            md_data["text_like_binary"] = {
                "positive_labels": list(text_like_binary.get("positive_labels", TEXT_LIKE_POSITIVE_LABELS)),
                "counts": dict(text_like_binary.get("counts", {})),
                "aggregates": dict(text_like_binary.get("aggregates", {})),
                "rows": list(text_like_binary.get("rows", [])),
                "by_label": dict(text_like_binary.get("by_label", {})),
            }
        boundary_payload = _needle_boundary_metrics_payload(
            md_stats.get("boundary_region", {}),
            label_names=markdown_metrics.label_names,
        )
        if boundary_payload is not None:
            md_data["boundary_region"] = {
                "window_radius_tokens": int(
                    boundary_payload.get("window_radius_tokens", NEEDLE_BOUNDARY_WINDOW_TOKENS)
                ),
                "support_chars": int(boundary_payload.get("support_chars", 0)),
                "samples": int(boundary_payload.get("samples", 0)),
                "aggregates": dict(boundary_payload.get("aggregates", {})),
                "rows": list(boundary_payload.get("rows", [])),
                "by_label": dict(boundary_payload.get("by_label", {})),
            }

        if md_data:
            data["tasks"]["markdown_mix"] = md_data

    # reStructuredText (same structure as markdown_mix)
    rst_metrics = metrics_by_name.get("restructuredtext_mix")
    if rst_metrics and rst_metrics.extras:
        rst_stats = rst_metrics.extras.get("restructuredtext_segments", {})
        rst_data: Dict[str, Any] = {}

        per_language = rst_stats.get("per_language", {})
        plain_per_lang = {}
        wrapped_per_lang = {}
        for lang, groups in per_language.items():
            for wrapper_key, target_dict in (("plain", plain_per_lang), ("wrapped", wrapped_per_lang)):
                group = groups.get(wrapper_key, {})
                count = int(group.get("count", 0))
                if count <= 0:
                    continue
                target_dict[lang] = {
                    "det_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_correct", 0), count)),
                    "coverage": _float_or_none(_safe_ratio(group.get("correct_chars", 0), group.get("truth_chars", 0))),
                    "nontext_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_nontext", 0), count)),
                    "nontext_coverage": _float_or_none(_safe_ratio(group.get("nontext_chars", 0), group.get("truth_chars", 0))),
                    "text_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_text", 0), count)),
                    "text_coverage": _float_or_none(_safe_ratio(group.get("text_chars", 0), group.get("truth_chars", 0))),
                    "support": count,
                }
        block_section = {}
        overall_block = {}
        for key, label in (("plain", "plain"), ("wrapped", "fenced")):
            group = rst_stats.get("overall", {}).get(key, {})
            count = int(group.get("count", 0))
            if count <= 0:
                continue
            overall_block[label] = {
                "det_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_correct", 0), count)),
                "coverage": _float_or_none(_safe_ratio(group.get("correct_chars", 0), group.get("truth_chars", 0))),
                "nontext_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_nontext", 0), count)),
                "nontext_coverage": _float_or_none(_safe_ratio(group.get("nontext_chars", 0), group.get("truth_chars", 0))),
                "text_rate@0.5": _float_or_none(_safe_ratio(group.get("detected_text", 0), count)),
                "text_coverage": _float_or_none(_safe_ratio(group.get("text_chars", 0), group.get("truth_chars", 0))),
                "support": count,
            }
        if overall_block:
            block_section["overall"] = overall_block
        if wrapped_per_lang:
            block_section["fenced_per_language"] = wrapped_per_lang
        if plain_per_lang:
            block_section["plain_per_language"] = plain_per_lang
        if block_section:
            rst_data["block"] = block_section

        inline_stats = rst_stats.get("inline", {})
        inline_per_lang = inline_stats.get("per_language", {})
        inline_section = {}
        inline_count = int(inline_stats.get("count", 0))
        if inline_count > 0:
            inline_section["overall"] = {
                "det_rate@0.5": _float_or_none(_safe_ratio(inline_stats.get("correct_hits", 0), inline_count)),
                "coverage": _float_or_none(_safe_ratio(inline_stats.get("correct_chars", 0), inline_stats.get("truth_chars", 0))),
                "nontext_rate@0.5": _float_or_none(_safe_ratio(inline_stats.get("nontext_hits", 0), inline_count)),
                "nontext_coverage": _float_or_none(_safe_ratio(inline_stats.get("nontext_chars", 0), inline_stats.get("truth_chars", 0))),
                "text_rate@0.5": _float_or_none(_safe_ratio(inline_stats.get("text_hits", 0), inline_count)),
                "text_coverage": _float_or_none(_safe_ratio(inline_stats.get("text_chars", 0), inline_stats.get("truth_chars", 0))),
                "support": inline_count,
            }
        if inline_per_lang:
            inline_data = {}
            for lang, entry in inline_per_lang.items():
                count = int(entry.get("count", 0))
                if count <= 0:
                    continue
                inline_data[lang] = {
                    "det_rate@0.5": _float_or_none(_safe_ratio(entry.get("correct_hits", 0), count)),
                    "coverage": _float_or_none(_safe_ratio(entry.get("correct_chars", 0), entry.get("truth_chars", 0))),
                    "nontext_rate@0.5": _float_or_none(_safe_ratio(entry.get("nontext_hits", 0), count)),
                    "nontext_coverage": _float_or_none(_safe_ratio(entry.get("nontext_chars", 0), entry.get("truth_chars", 0))),
                    "text_rate@0.5": _float_or_none(_safe_ratio(entry.get("text_hits", 0), count)),
                    "text_coverage": _float_or_none(_safe_ratio(entry.get("text_chars", 0), entry.get("truth_chars", 0))),
                    "support": count,
                }
            if inline_data:
                inline_section["per_language"] = inline_data
        if inline_section:
            rst_data["inline"] = inline_section

        wrong = rst_stats.get("wrong_label", {})
        wrong_cases = int(wrong.get("cases", 0))
        if wrong_cases > 0:
            rst_data["wrong_fence_fooled_rate"] = _float_or_none(_safe_ratio(wrong.get("fooled", 0), wrong_cases))

        text_stats = rst_stats.get("text", {})
        text_truth = int(text_stats.get("truth_chars", 0))
        union_chars = int(text_stats.get("union_chars", 0))
        if text_truth or union_chars:
            rst_data["text_accuracy"] = _float_or_none(_safe_ratio(text_stats.get("correct_chars", 0), text_truth))
            rst_data["text_iou"] = _float_or_none(_safe_ratio(text_stats.get("correct_chars", 0), union_chars))
        text_like_binary = rst_stats.get("text_like_binary", {})
        if isinstance(text_like_binary, Mapping) and text_like_binary.get("rows"):
            rst_data["text_like_binary"] = {
                "positive_labels": list(text_like_binary.get("positive_labels", TEXT_LIKE_POSITIVE_LABELS)),
                "counts": dict(text_like_binary.get("counts", {})),
                "aggregates": dict(text_like_binary.get("aggregates", {})),
                "rows": list(text_like_binary.get("rows", [])),
                "by_label": dict(text_like_binary.get("by_label", {})),
            }

        if rst_data:
            data["tasks"]["restructuredtext_mix"] = rst_data

    # Pure fragments / near_pure -------------------------------------------
    for pure_name in ("pure_fragments", "near_pure"):
        pure_metrics = metrics_by_name.get(pure_name)
        if not pure_metrics or not pure_metrics.extras:
            continue
        pure_stats = pure_metrics.extras.get("pure_fragments_purity", {})
        per_label = pure_stats.get("per_label", {})
        per_language: Dict[str, Any] = {}
        for lang, entry in per_label.items():
            total = int(entry.get("total", 0))
            if total <= 0:
                continue
            per_language[lang] = {
                "fully_pure_rate": _float_or_none(_safe_ratio(entry.get("pure", 0), total)),
                "within_threshold_rate": _float_or_none(_safe_ratio(entry.get("within", 0), total)),
                "support": total,
            }
        overall_total = int(pure_stats.get("total", 0))
        task_payload: Dict[str, Any] = {}
        if per_language:
            task_payload["per_language"] = per_language
        if overall_total > 0:
            task_payload["overall"] = {
                "fully_pure_rate": _float_or_none(_safe_ratio(pure_stats.get("perfect", 0), overall_total)),
                "within_threshold_rate": _float_or_none(_safe_ratio(pure_stats.get("within_threshold", 0), overall_total)),
                "support": overall_total,
            }
        if task_payload:
            data["tasks"][pure_name] = task_payload

    # Realistic ------------------------------------------------------------
    realistic_metrics = metrics_by_name.get("realistic")
    if realistic_metrics is not None:
        confusion_mat = np.asarray(realistic_metrics.confusion, dtype=np.int64)
        support_total = int(confusion_mat.sum()) if confusion_mat.size else 0
        if support_total > 0:
            id2label = {idx: str(label) for idx, label in enumerate(realistic_metrics.label_names)}
            metrics_payload = _metrics_payload_from_confusion(
                confusion_mat,
                id2label=id2label,
                num_classes=int(confusion_mat.shape[0]),
                ignore_class=None,
            )
            aggregates = metrics_payload.get("aggregates", {})
            task_payload = {
                "overall_accuracy": _float_or_none(realistic_metrics.overall_accuracy()),
                "macro_f1": _float_or_none(aggregates.get("macro_f1")),
                "macro_precision": _float_or_none(aggregates.get("macro_precision")),
                "macro_recall": _float_or_none(aggregates.get("macro_recall")),
                "weighted_f1": _float_or_none(aggregates.get("weighted_f1")),
                "support": support_total,
                "per_label": dict(metrics_payload.get("by_label", {})),
            }
            extras = realistic_metrics.extras or {}
            realistic_extras = extras.get("realistic_stats", {})
            boundary_payload = _sequence_boundary_metrics_payload(
                realistic_extras.get("boundary_region", {}),
                label_names=realistic_metrics.label_names,
            )
            if boundary_payload is not None:
                task_payload["boundary_region"] = {
                    "window_radius_tokens": int(
                        boundary_payload.get("window_radius_tokens", NEEDLE_BOUNDARY_WINDOW_TOKENS)
                    ),
                    "support_chars": int(boundary_payload.get("support_chars", 0)),
                    "samples": int(boundary_payload.get("samples", 0)),
                    "boundaries": int(boundary_payload.get("boundaries", 0)),
                    "aggregates": dict(boundary_payload.get("aggregates", {})),
                    "rows": list(boundary_payload.get("rows", [])),
                    "by_label": dict(boundary_payload.get("by_label", {})),
                }
            data["tasks"]["realistic"] = task_payload

    # Sequence tasks -------------------------------------------------------
    for name in ("sequence_pair", "sequence_triplet"):
        seq_metrics = metrics_by_name.get(name)
        if not seq_metrics or not seq_metrics.extras:
            continue
        seq_stats = seq_metrics.extras.get("sequence_purity", {})
        segments = seq_stats.get("segments", {})
        segment_data = {}
        for key in ("first", "second", "third"):
            seg = segments.get(key)
            if not seg:
                continue
            total = int(seg.get("total", 0))
            if total <= 0:
                continue
            segment_data[key] = {
                "coverage": _float_or_none(_safe_ratio(seg.get("correct", 0), total)),
                "support": total,
            }
        if segment_data:
            task_payload = {
                "segments": segment_data,
                "overall_accuracy": _float_or_none(seq_metrics.overall_accuracy()),
            }
            boundary_payload = _sequence_boundary_metrics_payload(
                seq_stats.get("boundary_region", {}),
                label_names=seq_metrics.label_names,
            )
            if boundary_payload is not None:
                task_payload["boundary_region"] = {
                    "window_radius_tokens": int(
                        boundary_payload.get("window_radius_tokens", NEEDLE_BOUNDARY_WINDOW_TOKENS)
                    ),
                    "support_chars": int(boundary_payload.get("support_chars", 0)),
                    "samples": int(boundary_payload.get("samples", 0)),
                    "boundaries": int(boundary_payload.get("boundaries", 0)),
                    "aggregates": dict(boundary_payload.get("aggregates", {})),
                    "rows": list(boundary_payload.get("rows", [])),
                    "by_label": dict(boundary_payload.get("by_label", {})),
                }
            data["tasks"][name] = task_payload

    # Needle datasets ------------------------------------------------------
    needle_entries = {}
    for metrics in task_metrics:
        bucket = _needle_bucket_key(metrics.name)
        if not bucket:
            continue
        stats = metrics.extras.get("needle_detection", {}) if metrics.extras else {}
        total = int(stats.get("total", 0))
        detected = int(stats.get("detected", 0))
        mean_iou = _safe_ratio(stats.get("iou_sum", 0.0), total)
        coverage = _safe_ratio(stats.get("correct_chars", 0), stats.get("needle_chars", 0))

        any_stats = stats.get("any_detection", {})
        any_count = int(any_stats.get("count", 0))
        any_detected = int(any_stats.get("detected", 0))
        any_mean_iou = _safe_ratio(any_stats.get("iou_sum", 0.0), any_count)
        any_cov = _safe_ratio(any_stats.get("correct_chars", 0), any_stats.get("truth_chars", 0))

        mis_counts = stats.get("misclass_counts", [])
        top_conf = []
        if mis_counts and metrics.confusion.size:
            total_mis = sum(mis_counts)
            if total_mis > 0:
                pairs = [
                    (mis_counts[idx] / total_mis, metrics.label_names[idx])
                    for idx in range(min(len(mis_counts), len(metrics.label_names)))
                    if mis_counts[idx] > 0
                ]
                pairs.sort(key=lambda x: x[0], reverse=True)
                top_conf = [
                    {"label": label, "fraction": _float_or_none(prob)}
                    for prob, label in pairs[:5]
                ]

        entry_payload = {
            "donor": {
                "det_rate@0.5": _float_or_none(_safe_ratio(detected, total)),
                "mean_iou": _float_or_none(mean_iou),
                "coverage": _float_or_none(coverage),
                "support": total,
            },
            "any": {
                "det_rate@0.5": _float_or_none(_safe_ratio(any_detected, any_count)),
                "mean_iou": _float_or_none(any_mean_iou),
                "coverage": _float_or_none(any_cov),
                "support": any_count,
            },
            "coverage_weighting": {
                "mode": "micro",
                "unit": "inserted_chars",
            },
            "top_misclassifications": top_conf,
        }
        boundary_payload = _needle_boundary_metrics_payload(
            stats.get("boundary_region", {}),
            label_names=metrics.label_names,
        )
        if boundary_payload is not None:
            entry_payload["boundary_region"] = {
                "window_radius_tokens": int(boundary_payload.get("window_radius_tokens", NEEDLE_BOUNDARY_WINDOW_TOKENS)),
                "support_chars": int(boundary_payload.get("support_chars", 0)),
                "samples": int(boundary_payload.get("samples", 0)),
                "aggregates": dict(boundary_payload.get("aggregates", {})),
                "rows": list(boundary_payload.get("rows", [])),
                "by_label": dict(boundary_payload.get("by_label", {})),
            }
        by_lang = stats.get("by_lang", {}) if isinstance(stats, dict) else {}
        if isinstance(by_lang, Mapping):
            per_anchor = {}
            for lang, lang_entry in sorted(by_lang.items()):
                count = int(lang_entry.get("count", 0))
                if count <= 0:
                    continue
                per_anchor[str(lang)] = {
                    "det_rate@0.5": _float_or_none(_safe_ratio(lang_entry.get("detected", 0), count)),
                    "exact_region_coverage": _float_or_none(
                        _safe_ratio(lang_entry.get("correct_chars", 0), lang_entry.get("truth_chars", 0))
                    ),
                    "support": count,
                }
            if per_anchor:
                entry_payload["per_donor_language"] = per_anchor
                entry_payload["per_anchor_language"] = per_anchor
        needle_entries[metrics.name] = entry_payload
    if needle_entries:
        data["tasks"]["needle"] = needle_entries

    for task_name in ("markdown_mix", "restructuredtext_mix", "sequence_pair", "sequence_triplet"):
        support = _task_support_payload(manifest, task_name)
        if support and task_name in data["tasks"]:
            data["tasks"][task_name]["support_summary"] = support
    if "needle" in data["tasks"]:
        for metrics in task_metrics:
            if _needle_bucket_key(metrics.name) is None:
                continue
            support = _task_support_payload(manifest, metrics.name)
            if support and metrics.name in data["tasks"]["needle"]:
                data["tasks"]["needle"][metrics.name]["support_summary"] = support

    # Markdown text accuracy already handled above.

    # Throughput -----------------------------------------------------------
    if throughput_results:
        throughput_data = {}
        for res in throughput_results:
            throughput_data[res.task] = {
                "device": res.device,
                "arch": res.arch,
                "samples": res.samples,
                "total_bytes": res.total_bytes,
                "total_windows": res.total_windows,
                "elapsed": _float_or_none(res.elapsed),
                "throughput_bytes_per_sec": _float_or_none(res.throughput),
                "windows_per_second": _float_or_none(res.windows_per_second),
                "rss_delta_mb": _float_or_none(res.rss_delta),
                "device_mem_delta_mb": _float_or_none(res.device_mem_delta),
            }
        data["throughput"] = throughput_data

    if monitor_b_report:
        data["monitor_b"] = {
            key: value
            for key, value in dict(monitor_b_report).items()
            if not str(key).startswith("_")
        }

    reduced_support_tasks = {
        metrics.name: dict(metrics.extras.get("truth_label_exclusions", {}))
        for metrics in task_metrics
        if isinstance(metrics.extras.get("truth_label_exclusions"), Mapping)
    }
    monitor_exclusions = None
    if isinstance(monitor_b_report, Mapping) and isinstance(monitor_b_report.get("truth_label_exclusions"), Mapping):
        monitor_exclusions = dict(monitor_b_report.get("truth_label_exclusions", {}))
    if reduced_support_tasks or monitor_exclusions or exclusion_note:
        data["reduced_support"] = {
            "note": exclusion_note,
            "tasks": reduced_support_tasks,
            "monitor_b": monitor_exclusions,
        }

    return data


def _write_comparison_metrics(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"📝 Comparison metrics written to {output_path}", flush=True)


def write_report(
    report_path: Path,
    *,
    manifest: dict,
    args,
    task_metrics: List[TaskMetrics],
    throughput_results: List[ThroughputResult],
    monitor_b_report: Optional[Mapping[str, Any]] = None,
) -> None:
    ordered_task_metrics = sorted(task_metrics, key=lambda m: _task_name_sort_key(m.name))
    non_needle_metrics = [m for m in ordered_task_metrics if _needle_bucket_key(m.name) is None]
    needle_only_metrics = [m for m in ordered_task_metrics if _needle_bucket_key(m.name) is not None]
    needle_only_metrics.sort(key=lambda m: (_needle_bucket_key(m.name) or (0, 0))[0], reverse=True)
    ordered_task_metrics = non_needle_metrics + needle_only_metrics

    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path = _comparison_metrics_path(report_path)
    confusion_artifacts = _write_confusion_artifacts(
        report_path,
        ordered_task_metrics,
        monitor_b_report=monitor_b_report,
    )

    def _format_channels(value) -> str:
        if isinstance(value, str):
            return value
        try:
            return ", ".join(str(ch) for ch in value)
        except TypeError:
            return str(value)

    report_lines: List[str] = []
    report_lines.append("# Segmenter Evaluation Report")
    report_lines.append("")
    report_lines.append(f"- Architecture: {getattr(args, 'arch', 'unet1d')}")
    report_lines.append(f"- Checkpoint: `{args.checkpoint}`")
    if str(getattr(args, "arch", "")).lower().strip() == "magika":
        report_lines.append(f"- Magika version: {getattr(args, 'magika_module_version', 'unknown')}")
        report_lines.append(f"- Magika model: {getattr(args, 'magika_model_name', 'unknown')}")
    else:
        report_lines.append(f"- Model dim: {args.model_dim}")
        report_lines.append(f"- Channels: {_format_channels(args.channels)}")
    report_lines.append(f"- Chunk: {args.chunk}")
    report_lines.append(f"- Batch size: {args.batch_size}")
    report_lines.append(f"- Other threshold: {float(getattr(args, 'other_threshold', 0.0)):.4f}")
    report_lines.append(f"- Minimum run smoothing: {int(getattr(args, 'min_run', 1))}")
    report_lines.append(f"- Postprocess profile: {getattr(args, 'postprocess_profile', POSTPROCESS_PROFILE_OFF)}")
    if getattr(args, "postprocess_profile", POSTPROCESS_PROFILE_OFF) == POSTPROCESS_PROFILE_THESIS:
        report_lines.append(f"- Postprocess minimum run: {int(getattr(args, 'postprocess_min_run', 5))}")
        report_lines.append(
            "- Postprocess boundary snap max shift: "
            f"{int(getattr(args, 'postprocess_boundary_snap_max_shift', 2))}"
        )
        report_lines.append(
            "- Postprocess stages: whitespace_relabel, other_gating, boundary_snap, min_run"
        )
        report_lines.append(
            "- Excluded viewer-only stages: markdown_structure_fill, paired_delimiter_fill, "
            "local_host_fill, newline_snap"
        )
    report_lines.append(f"- Max samples per task: {'all' if args.max_samples <= 0 else args.max_samples}")
    report_lines.append(f"- Sample seed: {args.sample_seed}")
    excluded_truth_labels = list(getattr(args, "exclude_truth_labels", []) or [])
    if excluded_truth_labels:
        report_lines.append(f"- Excluded truth labels: {', '.join(excluded_truth_labels)}")
    report_lines.append(f"- Evaluation data root: `{manifest.get('output_root')}`")
    if getattr(args, "fine_tuned_mode", None) is not None:
        report_lines.append(
            f"- Evaluation mode: {'fine_tuned' if bool(getattr(args, 'fine_tuned_mode', False)) else 'non_fine_tuned'}"
        )
    report_lines.append(f"- Generated at: {manifest.get('generated_at')}")
    report_lines.append(f"- Results directory: `{report_path.parent}`")
    report_lines.append(f"- Comparison JSON: [comparison_metrics.json]({_comparison_metrics_path(report_path).name})")
    if monitor_b_report:
        report_lines.append(f"- Full monitor_b root: `{monitor_b_report.get('root')}`")
    monitor_confusion_rel = confusion_artifacts.get("monitor_b")
    if monitor_confusion_rel:
        report_lines.append(f"- Full monitor_b confusion matrix: [{monitor_confusion_rel}]({monitor_confusion_rel})")
    aggregated_confusion_rel = confusion_artifacts.get("all_tasks")
    if aggregated_confusion_rel:
        report_lines.append(f"- Aggregated confusion matrix: [{aggregated_confusion_rel}]({aggregated_confusion_rel})")
    exclusion_note = _truth_label_exclusion_note(
        arch=str(getattr(args, "arch", "")),
        excluded_labels=excluded_truth_labels,
    )
    if exclusion_note:
        report_lines.append(f"- Reduced-support note: {exclusion_note}")
    report_lines.append("")
    highlights = _collect_task_highlights(ordered_task_metrics, manifest)
    if highlights:
        report_lines.append("### Task Highlights")
        report_lines.append("")
        report_lines.extend(highlights)
        report_lines.append("")

    comparison_payload = _collect_comparison_metrics(
        args,
        ordered_task_metrics,
        throughput_results,
        manifest,
        monitor_b_report,
    )
    if comparison_payload:
        _write_comparison_metrics(comparison_path, comparison_payload)

    if monitor_b_report:
        report_lines.append("## Full monitor_b Evaluation")
        report_lines.append("")
        report_lines.append(f"- Monitor root: `{monitor_b_report.get('root')}`")
        report_lines.append(f"- Architecture: {monitor_b_report.get('arch')}")
        report_lines.append(f"- Inference mode: {monitor_b_report.get('inference_mode')}")
        report_lines.append(f"- Open-set classification: {bool(monitor_b_report.get('open_set', False))}")
        report_lines.append(f"- Other threshold: {float(monitor_b_report.get('other_threshold', 0.0)):.4f}")
        report_lines.append(f"- Files used: {int(monitor_b_report.get('files_used', 0))}")
        report_lines.append(f"- Skipped: {int(monitor_b_report.get('skipped', 0))}")
        monitor_exclusions = monitor_b_report.get("truth_label_exclusions")
        if isinstance(monitor_exclusions, Mapping):
            report_lines.append(f"- Files requested: {int(monitor_exclusions.get('requested', 0))}")
            report_lines.append(f"- Files counted: {int(monitor_exclusions.get('counted', 0))}")
            report_lines.append(
                f"- Files skipped for excluded truth labels: {int(monitor_exclusions.get('skipped_for_truth_label', 0))}"
            )
            skipped_label_counts = monitor_exclusions.get("skipped_label_counts", {})
            if isinstance(skipped_label_counts, Mapping) and skipped_label_counts:
                skipped_summary = ", ".join(
                    f"{label}={int(count)}"
                    for label, count in sorted(skipped_label_counts.items())
                )
                report_lines.append(f"- Excluded-label skip counts: {skipped_summary}")
        report_lines.append(f"- Evaluated bytes: {int(monitor_b_report.get('evaluated_bytes', 0))}")
        report_lines.append(
            f"- Truth `other` bytes: {int(monitor_b_report.get('truth_other_bytes', 0))}"
        )
        report_lines.append(
            "- Low-confidence bytes routed to `other`: "
            f"{int(monitor_b_report.get('low_confidence_bytes_routed_to_other', 0))}"
        )
        if monitor_confusion_rel:
            report_lines.append(f"- Confusion matrix: [{monitor_confusion_rel}]({monitor_confusion_rel})")
        monitor_boundary = monitor_b_report.get("boundary_region")
        if isinstance(monitor_boundary, Mapping) and monitor_boundary.get("rows"):
            aggregates = monitor_boundary.get("aggregates", {})
            report_lines.append(
                "- Boundary-region label metrics "
                f"(`±{int(monitor_boundary.get('window_radius_tokens', NEEDLE_BOUNDARY_WINDOW_TOKENS))}` "
                f"tokens around effective truth transitions): acc {float(aggregates.get('micro_acc', 0.0)):.4f}, "
                f"prec {float(aggregates.get('macro_precision', 0.0)):.4f}, "
                f"recall {float(aggregates.get('macro_recall', 0.0)):.4f}, "
                f"f1 {float(aggregates.get('macro_f1', 0.0)):.4f} over "
                f"{int(monitor_boundary.get('support_chars', 0))} bytes in "
                f"{int(monitor_boundary.get('samples', 0))} files across "
                f"{int(monitor_boundary.get('boundaries', 0))} transitions."
            )
        report_lines.append("")
        monitor_rows = monitor_b_report.get("rows", [])
        if isinstance(monitor_rows, Sequence) and monitor_rows:
            report_lines.append(_render_training_style_metrics_table(monitor_rows))
            report_lines.append("")
        if isinstance(monitor_boundary, Mapping) and monitor_boundary.get("rows"):
            report_lines.append(
                "Boundary-region label metrics "
                f"(`±{int(monitor_boundary.get('window_radius_tokens', NEEDLE_BOUNDARY_WINDOW_TOKENS))}` "
                "tokens around effective truth transitions):"
            )
            report_lines.append("")
            report_lines.append(_render_training_style_metrics_table(monitor_boundary.get("rows", [])))
            report_lines.append("")

    report_lines.append("## Task Details")
    report_lines.append("")
    for metrics in ordered_task_metrics:
        report_lines.append(f"### {metrics.name}")
        report_lines.append("")
        report_lines.append(metrics.description)
        report_lines.append("")
        extras = metrics.extras or {}
        report_truth_exclusions = _report_truth_label_exclusions(metrics.name)
        if report_truth_exclusions:
            report_total_chars = _report_total_chars(metrics, exclude_truth_labels=report_truth_exclusions)
            report_overall_accuracy = _report_overall_accuracy(
                metrics,
                exclude_truth_labels=report_truth_exclusions,
            )
        else:
            report_total_chars = metrics.total_chars
            report_overall_accuracy = metrics.overall_accuracy()
        confusion_rel = confusion_artifacts.get(metrics.name)
        if confusion_rel:
            report_lines.append(f"- Confusion matrix: [{confusion_rel}]({confusion_rel})")
        truth_exclusions = extras.get("truth_label_exclusions")
        if isinstance(truth_exclusions, Mapping):
            report_lines.append(f"- Samples requested: {int(truth_exclusions.get('requested', 0))}")
            report_lines.append(f"- Samples counted: {int(truth_exclusions.get('counted', 0))}")
            report_lines.append(
                f"- Samples skipped for excluded truth labels: {int(truth_exclusions.get('skipped_for_truth_label', 0))}"
            )
            skipped_label_counts = truth_exclusions.get("skipped_label_counts", {})
            if isinstance(skipped_label_counts, Mapping) and skipped_label_counts:
                skipped_summary = ", ".join(
                    f"{label}={int(count)}"
                    for label, count in sorted(skipped_label_counts.items())
                )
                report_lines.append(f"- Excluded-label skip counts: {skipped_summary}")
        else:
            report_lines.append(f"- Samples: {metrics.samples}")
        report_lines.append(f"- Characters evaluated: {report_total_chars}")
        report_lines.append(f"- Overall accuracy: {report_overall_accuracy:.4f}")
        report_lines.append(
            f"- High confusions: {_summarize_confusions(metrics, exclude_true_labels=report_truth_exclusions)}"
        )
        name = metrics.name
        support_payload = _task_support_payload(manifest, name)
        for support_line in _support_summary_lines(support_payload):
            report_lines.append(f"- {support_line}")
        per_label_acc = metrics.per_label_accuracy()

        if name == "mal_injection":
            payload_stats = extras.get("mal_payload_detection", {})
            any_by_lang = payload_stats.get("any_by_lang", {})
            correct_by_lang = payload_stats.get("by_lang", {})
            top_mis = _top_confusions(
                metrics.confusion, metrics.label_names, exclude_labels=["other"]
            )
            languages = [
                lang
                for lang in sorted(set(any_by_lang.keys()) | set(correct_by_lang.keys()))
                if lang and lang != "other"
            ]
            if languages:
                table_lines = [
                    "| Language | Non-wrapper cov ≥50% | Non-wrapper coverage (avg) | Non-wrapper IoU ≥50% | Non-wrapper avg IoU | "
                    "Correct cov ≥50% | Correct coverage (avg) | Correct IoU ≥50% | Correct avg IoU | Top misclassifications |",
                    "| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- |",
                ]
                for lang in languages:
                    any_entry = any_by_lang.get(lang, {})
                    corr_entry = correct_by_lang.get(lang, {})
                    any_count = int(any_entry.get("count", 0))
                    any_detected = int(any_entry.get("detected", 0))
                    any_avg = _safe_ratio(any_entry.get("iou_sum", 0.0), any_count)
                    any_truth = int(any_entry.get("truth_chars", 0))
                    any_correct = int(any_entry.get("correct_chars", 0))
                    any_cov = _format_pct(_safe_ratio(any_correct, any_truth))
                    any_cov_hits = int(any_entry.get("coverage_hits", 0))
                    any_cov_total = int(any_entry.get("coverage_count", 0))

                    corr_count = int(corr_entry.get("count", 0))
                    corr_detected = int(corr_entry.get("detected", 0))
                    corr_avg = _safe_ratio(corr_entry.get("iou_sum", 0.0), corr_count)
                    corr_truth = int(corr_entry.get("truth_chars", 0))
                    corr_chars = int(corr_entry.get("correct_chars", 0))
                    corr_cov = _format_pct(_safe_ratio(corr_chars, corr_truth))
                    corr_cov_hits = int(corr_entry.get("coverage_hits", 0))
                    corr_cov_total = int(corr_entry.get("coverage_count", 0))

                    class_cov = per_label_acc.get(lang)
                    class_cov_str = _format_pct(class_cov)

                    mis_list = top_mis.get(lang, [])
                    mis_str = ", ".join(f"{pred} ({prob * 100:.1f}%)" for prob, pred in mis_list[:3]) or "—"

                    table_lines.append(
                        f"| {lang} | {_format_hits(any_cov_hits, any_cov_total)} | {any_cov} | {_format_hits(any_detected, any_count)} | {_format_float(any_avg)} | "
                        f"{_format_hits(corr_cov_hits, corr_cov_total)} | {corr_cov} (overall {class_cov_str}) | "
                        f"{_format_hits(corr_detected, corr_count)} | {_format_float(corr_avg)} | {mis_str} |"
                    )
                report_lines.append("")
                report_lines.extend(table_lines)
                report_lines.append("")
            continue

        if name == "pure_fragments":
            pure_stats = extras.get("pure_fragments_purity", {})
            ratios = pure_stats.get("foreign_ratios", [])
            if ratios:
                mean_ratio = float(sum(ratios)) / max(1, len(ratios))
                expected_bytes = mean_ratio * 1536
                report_lines.append("")
                report_lines.append(
                    f"Expected misclassified host bytes for 1536 host-labeled bytes: {expected_bytes:.1f}/1536"
                )
                report_lines.append("")

            accuracy_scores, misclassifications = _analyze_pure_fragments(metrics)
            report_lines.append("#### Purity Analysis")
            report_lines.append("")
            report_lines.append("| Language | Support | Accuracy % | File purity | Top Misclassifications |")
            report_lines.append("| --- | ---: | ---: | --- | --- |")

            # Merge per-label accuracy, file-level purity, and top misclassifications.
            # Order rows by descending accuracy, then by label name.
            per_label_support = metrics.per_label_counts
            per_label_correct = metrics.per_label_correct
            host_purity = pure_stats.get("host_byte_purity", {}) if isinstance(pure_stats, dict) else {}

            for lang, acc_pct in sorted(accuracy_scores.items(), key=lambda x: x[1], reverse=True):
                support = int(per_label_support.get(lang, 0))
                correct = int(per_label_correct.get(lang, 0))
                # Sanity check: derive accuracy from confusion vs precomputed percentage.
                acc = (correct / support) if support else float("nan")
                if math.isfinite(acc):
                    acc_pct = acc * 100.0
                acc_str = _format_pct(acc_pct / 100.0)  # display as percentage

                # File-level purity: fraction of host files where all host-label
                # characters were predicted correctly.
                hp = host_purity.get(lang, {})
                files = int(hp.get("files", 0))
                pure_files = int(hp.get("pure_files", 0))
                if files > 0:
                    file_purity_ratio = pure_files / files
                    file_purity_str = f"{pure_files}/{files} ({_format_pct(file_purity_ratio)})"
                else:
                    file_purity_str = "—"

                misclass_str = ""
                if lang in misclassifications:
                    top_mistakes = [
                        f"{label} ({pct:.1f}%)"
                        for label, pct in list(misclassifications[lang].items())[:3]
                    ]
                    misclass_str = ", ".join(top_mistakes)

                report_lines.append(
                    f"| {lang} | {support} | {acc_str} | {file_purity_str} | {misclass_str or '—'} |"
                )

            # Optionally include the open-set 'other' bucket for completeness.
            if "other" in metrics.label_names:
                support = int(per_label_support.get("other", 0))
                correct = int(per_label_correct.get("other", 0))
                acc = (correct / support) if support else float("nan")
                acc_str = "nan" if (not math.isfinite(acc)) else _format_pct(acc)
                report_lines.append(f"| other | {support} | {acc_str} | — | — |")

            report_lines.append("")
            continue

        if _needle_bucket_key(name) is not None:
            report_lines.append("")
            report_lines.append("_Label support below counts all evaluated non-whitespace characters in the task window, not just inserted needle support._")
            report_lines.append("")
            report_lines.append(_render_metrics_table(metrics))
            report_lines.append("")

            stats = extras.get("needle_detection", {}) or {}
            total = int(stats.get("total", 0))
            detected = int(stats.get("detected", 0))
            avg_iou = _safe_ratio(stats.get("iou_sum", 0.0), total)
            coverage = _format_pct(_safe_ratio(stats.get("correct_chars", 0), stats.get("needle_chars", 0)))

            any_stats = stats.get("any_detection", {}) or {}
            any_count = int(any_stats.get("count", 0))
            any_detected = int(any_stats.get("detected", 0))
            any_avg = _safe_ratio(any_stats.get("iou_sum", 0.0), any_count)
            any_cov = _format_pct(_safe_ratio(any_stats.get("correct_chars", 0), any_stats.get("truth_chars", 0)))

            mis_counts = stats.get("misclass_counts", [])
            mis_lines = "—"
            if mis_counts:
                total_mis = sum(mis_counts)
                if total_mis > 0:
                    pairs = [
                        (mis_counts[idx] / total_mis, metrics.label_names[idx])
                        for idx in range(min(len(mis_counts), len(metrics.label_names)))
                        if idx < len(metrics.label_names) and mis_counts[idx] > 0
                    ]
                    pairs.sort(key=lambda x: x[0], reverse=True)
                    mis_lines = ", ".join(f"{label} ({prob * 100:.1f}%)" for prob, label in pairs[:3]) or "—"

            coverage_hits = int(stats.get("coverage_hits", 0))
            coverage_count = int(stats.get("coverage_count", 0))
            any_cov_hits = int(any_stats.get("coverage_hits", 0))
            any_cov_total = int(any_stats.get("coverage_count", 0))
            report_lines.append("| Scenario | Coverage ≥50% | Exact region ≥50% | Avg exact region coverage | Coverage | Top misclassifications |")
            report_lines.append("| --- | --- | --- | ---: | ---: | --- |")
            report_lines.append(
                f"| Any non-wrapper | {_format_hits(any_cov_hits, any_cov_total)} | {_format_hits(any_detected, any_count)} | {_format_float(any_avg)} | {any_cov} | {mis_lines} |"
            )
            report_lines.append(
                f"| Exact inserted region | {_format_hits(coverage_hits, coverage_count)} | {_format_hits(detected, total)} | {_format_float(avg_iou)} | {coverage} | — |"
            )
            boundary_payload = _needle_boundary_metrics_payload(
                stats.get("boundary_region", {}),
                label_names=metrics.label_names,
            )
            if boundary_payload is not None:
                report_lines.append("")
                report_lines.append(
                    "Boundary-region label metrics "
                    f"(`±{int(boundary_payload.get('window_radius_tokens', NEEDLE_BOUNDARY_WINDOW_TOKENS))}` "
                    "tokens around inserted start/end):"
                )
                report_lines.append("")
                report_lines.append(_render_training_style_metrics_table(boundary_payload.get("rows", [])))
            by_lang = stats.get("by_lang", {})
            any_by_lang = stats.get("any_by_lang", {})
            if isinstance(by_lang, Mapping) and by_lang:
                donor_floor = int(support_payload.get("donor_floor_visible_chars", 0) or 0)
                donor_candidate_visible = support_payload.get("donor_candidate_visible_support_by_donor_label", {})
                donor_selected_visible = support_payload.get("donor_selected_visible_support_by_donor_label", {})
                report_lines.append("")
                report_lines.append("| Donor label | Samples | Selected visible chars | Qualified | Any non-wrapper ≥50% | Any non-wrapper coverage | Exact region ≥50% | Exact region coverage |")
                report_lines.append("| --- | ---: | ---: | --- | --- | ---: | --- | ---: |")
                for lang, entry in sorted(by_lang.items()):
                    any_entry = any_by_lang.get(lang, {}) if isinstance(any_by_lang, Mapping) else {}
                    lang_total = int(entry.get("count", 0))
                    if lang_total <= 0:
                        continue
                    any_total_lang = int(any_entry.get("count", 0))
                    any_cov_lang = _format_pct(_safe_ratio(any_entry.get("correct_chars", 0), any_entry.get("truth_chars", 0)))
                    exact_cov_lang = _format_pct(_safe_ratio(entry.get("correct_chars", 0), entry.get("truth_chars", 0)))
                    candidate_visible = 0
                    if isinstance(donor_candidate_visible, Mapping):
                        candidate_visible = int(donor_candidate_visible.get(lang, 0))
                    selected_visible = int(entry.get("truth_chars", 0))
                    if isinstance(donor_selected_visible, Mapping):
                        selected_visible = int(donor_selected_visible.get(lang, selected_visible))
                    qualified = "yes" if donor_floor <= 0 or candidate_visible >= donor_floor else "no"
                    report_lines.append(
                        f"| {lang} | {lang_total} | {selected_visible} | {qualified} | "
                        f"{_format_hits(int(any_entry.get('detected', 0)), any_total_lang)} | {any_cov_lang} | "
                        f"{_format_hits(int(entry.get('detected', 0)), lang_total)} | {exact_cov_lang} |"
                    )
            report_lines.append("")
            continue

        if name in ("sequence_pair", "sequence_triplet"):
            seq_stats = extras.get("sequence_purity", {})
            segments = seq_stats.get("segments", {})
            report_lines.append("")
            report_lines.append("| Segment | Exact region coverage |")
            report_lines.append("| --- | ---: |")
            for key in ("first", "second", "third"):
                data = segments.get(key)
                if not data:
                    continue
                total = int(data.get("total", 0))
                correct = int(data.get("correct", 0))
                coverage = _format_pct(correct / total) if total > 0 else "n/a"
                report_lines.append(f"| {key.title()} | {coverage} |")
            boundary_payload = _sequence_boundary_metrics_payload(
                seq_stats.get("boundary_region", {}),
                label_names=metrics.label_names,
            )
            if boundary_payload is not None:
                report_lines.append("")
                report_lines.append(
                    "Boundary-region label metrics "
                    f"(`±{int(boundary_payload.get('window_radius_tokens', NEEDLE_BOUNDARY_WINDOW_TOKENS))}` "
                    "tokens around sequence transitions):"
                )
                report_lines.append("")
                report_lines.append(_render_training_style_metrics_table(boundary_payload.get("rows", [])))
            report_lines.append("")

        # Default handling for other tasks
        if name == "markdown_mix":
            markdown_stats = extras.get("markdown_segments", {})
            threshold = float(markdown_stats.get("threshold", MARKDOWN_IOU_THRESHOLD))
            wrapped_group = markdown_stats.get("overall", {}).get("wrapped", {})
            plain_group = markdown_stats.get("overall", {}).get("plain", {})
            text_like_binary = markdown_stats.get("text_like_binary", {})
            boundary_payload = _needle_boundary_metrics_payload(
                markdown_stats.get("boundary_region", {}),
                label_names=metrics.label_names,
            )
            report_lines.append("")
            report_lines.append("_Text hits column: lower is better._")
            report_lines.append("| Wrapper | Non-text cov ≥50% | Non-text region coverage | Text hits | Exact region cov ≥50% | Exact region avg coverage |")
            report_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")

            def _emit_wrapper_row(label: str, group: Dict[str, Any]) -> None:
                count = int(group.get("count", 0))
                if count <= 0:
                    report_lines.append(f"| {label} | — | — | — | — | — |")
                    return
                truth_chars = int(group.get("truth_chars", 0))
                nontext_cov = _format_pct(_safe_ratio(int(group.get("nontext_chars", 0)), truth_chars))
                correct_cov = _format_pct(_safe_ratio(int(group.get("correct_chars", 0)), truth_chars))
                nontext_chars = int(group.get("nontext_chars", 0))
                correct_chars = int(group.get("correct_chars", 0))
                nontext_cov_hits = _format_hits(int(group.get("detected_nontext", 0)), count)
                text_hits = _format_hits(int(group.get("detected_text", 0)), count)
                correct_cov_hits = _format_hits(int(group.get("detected_correct", 0)), count)
                row = (
                    f"| {label} | {nontext_cov_hits} | {nontext_cov} | {text_hits} | "
                    f"{correct_cov_hits} | {correct_cov} |"
                )
                report_lines.append(row)

            for label, group in ((MARKDOWN_FENCED_LABEL, wrapped_group), ("bare code", plain_group)):
                _emit_wrapper_row(label, group)
            inline_stats = markdown_stats.get("inline", {})
            inline_count = int(inline_stats.get("count", 0))
            if inline_count > 0:
                inline_group = {
                    "count": inline_count,
                    "detected_nontext": inline_stats.get("nontext_hits", 0),
                    "detected_nontext_iou": inline_stats.get("nontext_iou_hits", 0),
                    "detected_text": inline_stats.get("text_hits", 0),
                    "detected_correct": inline_stats.get("correct_hits", 0),
                    "detected_correct_iou": inline_stats.get("correct_iou_hits", 0),
                    "truth_chars": inline_stats.get("truth_chars", 0),
                    "nontext_chars": inline_stats.get("nontext_chars", 0),
                    "correct_chars": inline_stats.get("correct_chars", 0),
                    "union_nontext_chars": inline_stats.get("union_nontext_chars", 0),
                    "union_correct_chars": inline_stats.get("union_correct_chars", 0),
                }
                _emit_wrapper_row(MARKDOWN_INLINE_LABEL, inline_group)
            text_truth = int(markdown_stats.get("text", {}).get("truth_chars", 0))
            if text_truth > 0:
                text_cov = _format_pct(
                    _safe_ratio(markdown_stats.get("text", {}).get("correct_chars", 0), text_truth)
                )
                report_lines.append("")
                report_lines.append(f"Text region coverage: {text_cov}")
            if boundary_payload is not None:
                report_lines.append("")
                report_lines.append(
                    "Boundary-region label metrics "
                    f"(`±{int(boundary_payload.get('window_radius_tokens', NEEDLE_BOUNDARY_WINDOW_TOKENS))}` "
                    "tokens around markdown/code transitions):"
                )
                report_lines.append("")
                report_lines.append(_render_training_style_metrics_table(boundary_payload.get("rows", [])))
            _append_text_like_binary_lines(report_lines, text_like_binary)
            report_lines.append("")

        if name == "restructuredtext_mix":
            markdown_stats = extras.get("restructuredtext_segments", {})
            threshold = float(markdown_stats.get("threshold", MARKDOWN_IOU_THRESHOLD))
            wrapped_group = markdown_stats.get("overall", {}).get("wrapped", {})
            plain_group = markdown_stats.get("overall", {}).get("plain", {})
            text_like_binary = markdown_stats.get("text_like_binary", {})
            report_lines.append("")
            report_lines.append("_Text hits column: lower is better._")
            report_lines.append("| Wrapper | Non-text cov ≥50% | Non-text region coverage | Text hits | Exact region cov ≥50% | Exact region avg coverage |")
            report_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")

            def _emit_wrapper_row_rst(label: str, group: Dict[str, Any]) -> None:
                count = int(group.get("count", 0))
                if count <= 0:
                    report_lines.append(f"| {label} | — | — | — | — | — |")
                    return
                truth_chars = int(group.get("truth_chars", 0))
                nontext_cov = _format_pct(_safe_ratio(int(group.get("nontext_chars", 0)), truth_chars))
                correct_cov = _format_pct(_safe_ratio(int(group.get("correct_chars", 0)), truth_chars))
                nontext_chars = int(group.get("nontext_chars", 0))
                correct_chars = int(group.get("correct_chars", 0))
                nontext_cov_hits = _format_hits(int(group.get("detected_nontext", 0)), count)
                text_hits = _format_hits(int(group.get("detected_text", 0)), count)
                correct_cov_hits = _format_hits(int(group.get("detected_correct", 0)), count)
                row = (
                    f"| {label} | {nontext_cov_hits} | {nontext_cov} | {text_hits} | "
                    f"{correct_cov_hits} | {correct_cov} |"
                )
                report_lines.append(row)

            for label, group in ((MARKDOWN_FENCED_LABEL, wrapped_group), ("bare code", plain_group)):
                _emit_wrapper_row_rst(label, group)
            inline_stats = markdown_stats.get("inline", {})
            inline_count = int(inline_stats.get("count", 0))
            if inline_count > 0:
                inline_group = {
                    "count": inline_count,
                    "detected_nontext": inline_stats.get("nontext_hits", 0),
                    "detected_nontext_iou": inline_stats.get("nontext_iou_hits", 0),
                    "detected_text": inline_stats.get("text_hits", 0),
                    "detected_correct": inline_stats.get("correct_hits", 0),
                    "detected_correct_iou": inline_stats.get("correct_iou_hits", 0),
                    "truth_chars": inline_stats.get("truth_chars", 0),
                    "nontext_chars": inline_stats.get("nontext_chars", 0),
                    "correct_chars": inline_stats.get("correct_chars", 0),
                    "union_nontext_chars": inline_stats.get("union_nontext_chars", 0),
                    "union_correct_chars": inline_stats.get("union_correct_chars", 0),
                }
                _emit_wrapper_row_rst(MARKDOWN_INLINE_LABEL, inline_group)
            text_truth = int(markdown_stats.get("text", {}).get("truth_chars", 0))
            if text_truth > 0:
                text_cov = _format_pct(
                    _safe_ratio(markdown_stats.get("text", {}).get("correct_chars", 0), text_truth)
                )
                report_lines.append("")
                report_lines.append(f"Text region coverage: {text_cov}")
            _append_text_like_binary_lines(report_lines, text_like_binary)
            report_lines.append("")

        per_label_table = _render_metrics_table(
            metrics,
            exclude_truth_labels=report_truth_exclusions,
        )
        report_lines.append(per_label_table)
        report_lines.append("")

    if throughput_results:
        report_lines.append("## Throughput Benchmarks")
        report_lines.append("")
        report_lines.append(_render_throughput_table(throughput_results))

    report_lines.append("")
    report_lines.append("Report generated at " + time.strftime("%Y-%m-%d %H:%M:%S"))

    report_path.write_text("\n".join(report_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on curated benchmarks.")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.msgpack or Orbax directory).")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "evaluation" / "data"), help="Path to evaluation datasets.")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON (defaults to <data-root>/manifest.json).")
    parser.add_argument("--tasks", nargs="*", help="Subset of task names to evaluate.")
    parser.add_argument("--device", default="auto", help="Device for accuracy workloads (cpu/gpu/cuda/auto).")
    parser.add_argument("--cpu-device", default="cpu", help="Device name for throughput CPU benchmark.")
    parser.add_argument("--gpu-device", default="cuda", help="Device name for throughput GPU benchmark (ignored if unavailable).")
    parser.add_argument("--arch", default=None, choices=("unet1d", "mamba", "magika"), help="Model architecture (auto if omitted).")
    parser.add_argument("--model-dim", type=int, default=None, help="Model embedding dimension. Defaults to checkpoint config or 256.")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated channel widths. Defaults to checkpoint config or 96,128,192,256.")
    parser.add_argument("--dtype", type=str, default=None, help="JAX dtype name for inference (e.g. bfloat16). Defaults to checkpoint config or bfloat16.")
    parser.add_argument("--mamba-layers", type=int, default=None, help="Mamba only: number of layers (auto if omitted).")
    parser.add_argument("--mamba-d-state", type=int, default=None, help="Mamba only: SSM state size (auto if omitted).")
    parser.add_argument("--mamba-expand", type=int, default=None, help="Mamba only: expansion factor (auto if omitted).")
    parser.add_argument("--mamba-dt-rank", type=int, default=None, help="Mamba only: dt low-rank dim (auto if omitted).")
    parser.add_argument("--mamba-conv", type=int, default=None, help="Mamba only: depthwise conv kernel (auto if omitted).")
    parser.add_argument("--mamba-bidirectional", action=argparse.BooleanOptionalAction, default=None, help="Mamba only: bidirectional scan.")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--min-run", type=int, default=1, help="Minimum run-length smoothing for character labels.")
    parser.add_argument(
        "--other-threshold",
        type=float,
        default=0.3,
        help=(
            "Open-set threshold for routing low-confidence predictions to the 'other' class "
            "(0 disables thresholding)."
        ),
    )
    parser.add_argument(
        "--postprocess-profile",
        choices=POSTPROCESS_PROFILE_CHOICES,
        default=POSTPROCESS_PROFILE_OFF,
        help=(
            "Character-level post-processing profile. 'off' preserves baseline behavior; "
            "'thesis' applies the thesis-only whitespace/other/boundary/min-run sequence."
        ),
    )
    parser.add_argument(
        "--postprocess-min-run",
        type=int,
        default=5,
        help="Minimum interior run length for --postprocess-profile thesis.",
    )
    parser.add_argument(
        "--postprocess-boundary-snap-max-shift",
        type=int,
        default=2,
        help="Maximum boundary movement in characters for --postprocess-profile thesis.",
    )
    parser.add_argument(
        "--cache-predictions-dir",
        default=None,
        help=(
            "Directory to dump per-task raw model outputs (labels + per-char max prob) "
            "during this run. Enables later --rescore-from-cache sweeps without re-inference."
        ),
    )
    parser.add_argument(
        "--rescore-from-cache",
        default=None,
        help=(
            "Directory previously populated with --cache-predictions-dir. When set, the "
            "model checkpoint is not loaded and all task scoring uses the cached raw "
            "outputs combined with the requested --postprocess-* settings."
        ),
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help=(
            "Optional report markdown path or artifact directory. "
            "Defaults to a timestamped run directory under evaluation/reports/."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference windows.")
    parser.add_argument(
        "--inference-backend",
        type=str,
        default="auto",
        choices=("auto", "fast", "legacy"),
        help=(
            "Inference backend for non-training evaluation. "
            "'auto' prefers the shared fast backend and falls back loudly when needed."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum samples per accuracy task (0 = all).")
    parser.add_argument("--sample-seed", type=int, default=13, help="Seed for subsampling large datasets.")
    parser.add_argument("--log-interval", type=int, default=250, help="Progress logging interval (in samples).")
    parser.add_argument(
        "--fine-tuned",
        action="store_true",
        default=False,
        help=(
            "Deprecated compatibility flag. Fine-tuned evaluation is now the default."
        ),
    )
    parser.add_argument(
        "--non-fine-tuned",
        action="store_true",
        default=False,
        help=(
            "Use the legacy non-fine-tuned evaluation path: keep evaluation/data "
            "and skip the automatic full monitor_preprocessed_b pass."
        ),
    )
    parser.add_argument(
        "--monitor-root",
        default=str(_default_monitor_b_root()),
        help=(
            "Root of the full monitor_b memmap used for the automatic fine-tuned "
            "monitor evaluation."
        ),
    )
    parser.add_argument(
        "--exclude-truth-labels",
        nargs="*",
        default=(),
        help=(
            "Skip any benchmark sample or full monitor_b file whose ground-truth segments contain "
            "one of these labels."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    mode_info = _configure_evaluation_mode(args)
    for message in mode_info.get("messages", []):
        print(message, flush=True)

    rescore_mode = bool(getattr(args, "rescore_from_cache", None))
    cache_write_mode = bool(getattr(args, "cache_predictions_dir", None))
    if rescore_mode and cache_write_mode:
        raise RuntimeError("--cache-predictions-dir and --rescore-from-cache are mutually exclusive.")

    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest) if args.manifest else data_root / "manifest.json"
    overall_start = time.perf_counter()

    try:
        if manifest_path.exists():
            manifest = _load_manifest(manifest_path)
        else:
            manifest = {"generated_at": "n/a", "output_root": str(data_root)}

        if not data_root.exists() or not data_root.is_dir():
            raise RuntimeError(f"Evaluation data directory not found or not a directory: {data_root}")

        auto_hparams: Dict[str, Any] = {}
        label_names = None
        ckpt_path: Optional[Path] = None
        if args.checkpoint and not rescore_mode:
            ckpt_path = Path(args.checkpoint).resolve()
            auto_hparams = _load_checkpoint_hparams(ckpt_path)
            label_names = auto_hparams.get("label_names")
            if label_names:
                _apply_label_mapping(label_names)

        arch = args.arch if args.arch is not None else auto_hparams.get("arch", None)
        if arch is None and not args.checkpoint and not rescore_mode:
            arch = "magika"
        arch = str(arch).lower().strip() if arch else "unet1d"
        args.arch = arch

        if not rescore_mode:
            if arch != "magika" and ckpt_path is None:
                raise RuntimeError(f"Architecture '{arch}' requires --checkpoint.")
            if arch == "magika" and not args.checkpoint:
                args.checkpoint = "magika://default"

        if arch == "magika":
            model_dim = args.model_dim if args.model_dim is not None else 0
            dtype = args.dtype if args.dtype is not None else "float32"
            channels = []
        else:
            model_dim = args.model_dim if args.model_dim is not None else auto_hparams.get("model_dim", 256)
            dtype = args.dtype if args.dtype is not None else auto_hparams.get("dtype", "bfloat16")
            dtype = dtype.rsplit(".", 1)[-1]
            if arch == "unet1d":
                if args.channels:
                    channels = [int(ch) for ch in args.channels.split(",") if ch.strip()]
                else:
                    channels_source = auto_hparams.get("channels", DEFAULT_CHANNELS)
                    channels = [int(ch) for ch in channels_source]
            else:
                channels = [int(ch) for ch in DEFAULT_CHANNELS]

        if arch == "mamba":
            args.mamba_layers = args.mamba_layers if args.mamba_layers is not None else auto_hparams.get("mamba_layers", 6)
            args.mamba_d_state = args.mamba_d_state if args.mamba_d_state is not None else auto_hparams.get("mamba_d_state", 8)
            args.mamba_expand = args.mamba_expand if args.mamba_expand is not None else auto_hparams.get("mamba_expand", 1)
            args.mamba_dt_rank = args.mamba_dt_rank if args.mamba_dt_rank is not None else auto_hparams.get("mamba_dt_rank", 16)
            args.mamba_conv = args.mamba_conv if args.mamba_conv is not None else auto_hparams.get("mamba_conv", 4)
            args.mamba_bidirectional = (
                args.mamba_bidirectional
                if args.mamba_bidirectional is not None
                else bool(auto_hparams.get("mamba_bidirectional", True))
            )

        args.model_dim = model_dim
        args.dtype = dtype
        args.channels = channels
        args.exclude_truth_labels = [str(label).strip() for label in (args.exclude_truth_labels or []) if str(label).strip()]
        args.postprocess_profile = str(getattr(args, "postprocess_profile", POSTPROCESS_PROFILE_OFF)).lower().strip()
        if args.postprocess_profile not in POSTPROCESS_PROFILE_CHOICES:
            raise RuntimeError(f"Unknown postprocess profile: {args.postprocess_profile!r}")
        args.postprocess_min_run = max(1, int(getattr(args, "postprocess_min_run", 5)))
        args.postprocess_boundary_snap_max_shift = max(
            0,
            int(getattr(args, "postprocess_boundary_snap_max_shift", 2)),
        )
        if args.postprocess_profile == POSTPROCESS_PROFILE_THESIS:
            print(
                "ℹ️  Thesis post-processing enabled: "
                f"tau={float(args.other_threshold):.4f}, "
                f"boundary_shift<={int(args.postprocess_boundary_snap_max_shift)}, "
                f"min_run={int(args.postprocess_min_run)}",
                flush=True,
            )

        if auto_hparams and arch != "magika":
            extra = f", classes={len(label_names)}" if label_names else ""
            if arch == "unet1d":
                print(
                    f"ℹ️  Using checkpoint hyperparameters: arch={arch}, model_dim={model_dim}, channels={channels}, dtype={dtype}{extra}",
                    flush=True,
                )
            else:
                print(
                    "ℹ️  Using checkpoint hyperparameters: "
                    f"arch={arch}, model_dim={model_dim}, dtype={dtype}, "
                    f"mamba_layers={int(args.mamba_layers)}, mamba_d_state={int(args.mamba_d_state)}, "
                    f"mamba_expand={int(args.mamba_expand)}, mamba_dt_rank={int(args.mamba_dt_rank)}, "
                    f"mamba_conv={int(args.mamba_conv)}, mamba_bidirectional={bool(args.mamba_bidirectional)}{extra}",
                    flush=True,
                )

        cache_adapter: Optional[_RunnerCacheAdapter] = None
        if rescore_mode:
            cache = PredictionCache(Path(args.rescore_from_cache).resolve(), mode="read")
            cache_adapter = _RunnerCacheAdapter(
                inner=None,
                cache=cache,
                fallback_arch=arch,
                fallback_chunk=int(args.chunk),
                fallback_batch_size=int(args.batch_size),
            )
            accuracy_runner = cache_adapter
            print(
                f"ℹ️  Rescore-from-cache mode: model load skipped, reading raw predictions from {args.rescore_from_cache}",
                flush=True,
            )
        else:
            real_runner = SegmenterRunner(
                args.checkpoint,
                arch=arch,
                model_dim=model_dim,
                channels=channels,
                mamba_layers=int(getattr(args, "mamba_layers", 6) or 6),
                mamba_d_state=int(getattr(args, "mamba_d_state", 8) or 8),
                mamba_expand=int(getattr(args, "mamba_expand", 1) or 1),
                mamba_dt_rank=int(getattr(args, "mamba_dt_rank", 16) or 16),
                mamba_conv=int(getattr(args, "mamba_conv", 4) or 4),
                mamba_bidirectional=bool(getattr(args, "mamba_bidirectional", True)),
                dtype=dtype,
                chunk=args.chunk,
                device=args.device,
                batch_size=args.batch_size,
                inference_backend=args.inference_backend,
            )
            if cache_write_mode:
                cache = PredictionCache(Path(args.cache_predictions_dir).resolve(), mode="write")
                cache_adapter = _RunnerCacheAdapter(inner=real_runner, cache=cache)
                accuracy_runner = cache_adapter
                print(
                    f"ℹ️  Caching raw predictions to {args.cache_predictions_dir}",
                    flush=True,
                )
            else:
                accuracy_runner = real_runner
            if arch == "magika":
                args.magika_module_version = getattr(real_runner, "magika_module_version", None)
                args.magika_model_name = getattr(real_runner, "magika_model_name", None)
            runner_channels = getattr(real_runner, "channels", None)
            if runner_channels:
                channels = [int(ch) for ch in runner_channels]
                args.channels = channels

        datasets = _collect_datasets(data_root, args.tasks)
        if not datasets:
            raise RuntimeError(f"No datasets found under {data_root}")

        evaluate_task._log_interval = args.log_interval  # type: ignore[attr-defined]

        task_descriptions = {
            entry.get("task"): entry.get("description", "")
            for entry in manifest.get("tasks", [])
            if isinstance(entry, dict) and entry.get("task")
        }

        prepared_datasets: Dict[str, hfds.Dataset] = {}
        dataset_meta: Dict[str, Dict[str, Any]] = {}
        for name, ds in datasets.items():
            preserve_all = name.startswith("throughput_")
            subset, original_len, subset_len, sampled = _prepare_dataset(
                name,
                ds,
                max_samples=args.max_samples,
                base_seed=args.sample_seed,
                preserve_all=preserve_all,
            )
            prepared_datasets[name] = subset
            dataset_meta[name] = {
                "original": original_len,
                "selected": subset_len,
                "sampled": sampled,
                "description": task_descriptions.get(name, ""),
            }

        accuracy_names = [name for name in prepared_datasets if not name.startswith("throughput_")]
        throughput_names = [name for name in prepared_datasets if name.startswith("throughput_")]
        print(
            f"ℹ️  Prepared {len(prepared_datasets)} datasets "
            f"(accuracy={len(accuracy_names)}, throughput={len(throughput_names)})",
            flush=True,
        )

        task_metrics: List[TaskMetrics] = []
        throughput_results: List[ThroughputResult] = []
        monitor_b_report: Optional[Dict[str, Any]] = None

        for name in sorted(accuracy_names, key=_task_name_sort_key):
            ds = prepared_datasets[name]
            meta = dataset_meta[name]
            sample_note = " • subsampled" if meta["sampled"] else ""
            print(
                f"▶️  Evaluating task '{name}' [{meta['selected']}/{meta['original']} samples{sample_note}]",
                flush=True,
            )
            if cache_adapter is not None:
                cache_adapter.begin_task(name)
            try:
                metrics = evaluate_task(
                    name,
                    meta["description"],
                    ds,
                    accuracy_runner,
                    min_run_chars=args.min_run,
                    other_threshold=args.other_threshold,
                    postprocess_profile=args.postprocess_profile,
                    postprocess_min_run_chars=args.postprocess_min_run,
                    postprocess_boundary_snap_max_shift=args.postprocess_boundary_snap_max_shift,
                    excluded_truth_labels=args.exclude_truth_labels,
                )
            finally:
                if cache_adapter is not None:
                    cache_adapter.end_task()
            elapsed = metrics.extras.get("elapsed_seconds", 0.0)
            print(
                f"    [{name}] completed in {elapsed:.1f}s • char_acc={metrics.overall_accuracy():.4f}",
                flush=True,
            )
            task_metrics.append(metrics)

        if False:
            monitor_root = Path(getattr(args, "monitor_root", _default_monitor_b_root())).resolve()
            if not monitor_root.exists():
                raise RuntimeError(
                    f"Fine-tuned evaluation requires a full monitor_b root at {monitor_root}"
                )
            print(
                f"▶️  Evaluating full monitor_b from {monitor_root}",
                flush=True,
            )
            monitor_b_report = _evaluate_full_monitor_b(
                monitor_root,
                accuracy_runner,
                other_threshold=args.other_threshold,
                excluded_truth_labels=args.exclude_truth_labels,
            )
            print(
                "    [monitor_b] completed in "
                f"{float(monitor_b_report.get('elapsed_seconds', 0.0)):.1f}s "
                f"• micro_acc={_format_float(monitor_b_report.get('aggregates', {}).get('micro_acc'))}",
                flush=True,
            )

        # Throughput datasets evaluated separately on CPU and GPU (if available)
        if throughput_names:
            try:
                cpu_runner = SegmenterRunner(
                    args.checkpoint,
                    arch=arch,
                    model_dim=args.model_dim,
                    channels=channels,
                    mamba_layers=int(getattr(args, "mamba_layers", 6) or 6),
                    mamba_d_state=int(getattr(args, "mamba_d_state", 8) or 8),
                    mamba_expand=int(getattr(args, "mamba_expand", 1) or 1),
                    mamba_dt_rank=int(getattr(args, "mamba_dt_rank", 16) or 16),
                    mamba_conv=int(getattr(args, "mamba_conv", 4) or 4),
                    mamba_bidirectional=bool(getattr(args, "mamba_bidirectional", True)),
                    dtype=args.dtype,
                    chunk=args.chunk,
                    device=args.cpu_device,
                    batch_size=args.batch_size,
                    inference_backend=args.inference_backend,
                )
            except RuntimeError as exc:
                print(f"⚠️  Skipping CPU throughput ({exc})", flush=True)
                cpu_runner = None
            if cpu_runner is not None:
                for name in sorted(throughput_names, key=_task_name_sort_key):
                    ds = prepared_datasets[name]
                    meta = dataset_meta[name]
                    print(
                        f"▶️  Throughput '{name}' on {args.cpu_device} [{meta['selected']} samples]",
                        flush=True,
                    )
                    result = measure_throughput(
                        name,
                        ds,
                        cpu_runner,
                        min_run_chars=args.min_run,
                        device_name=args.cpu_device,
                    )
                    print(
                        f"    [{name}/{args.cpu_device}] {_format_bytes_per_sec(result.throughput)} • {result.elapsed:.2f}s",
                        flush=True,
                    )
                    throughput_results.append(result)

            # GPU benchmark only if GPU available
            try:
                gpu_devices = jax.devices("gpu")
            except Exception:
                gpu_devices = []
            if arch == "magika":
                print("ℹ️  Magika throughput is CPU-only; skipping GPU throughput benchmarks.", flush=True)
            elif gpu_devices:
                try:
                    gpu_runner = SegmenterRunner(
                        args.checkpoint,
                        arch=arch,
                        model_dim=args.model_dim,
                        channels=channels,
                        mamba_layers=int(getattr(args, "mamba_layers", 6) or 6),
                        mamba_d_state=int(getattr(args, "mamba_d_state", 8) or 8),
                        mamba_expand=int(getattr(args, "mamba_expand", 1) or 1),
                        mamba_dt_rank=int(getattr(args, "mamba_dt_rank", 16) or 16),
                        mamba_conv=int(getattr(args, "mamba_conv", 4) or 4),
                        mamba_bidirectional=bool(getattr(args, "mamba_bidirectional", True)),
                        dtype=args.dtype,
                        chunk=args.chunk,
                        device=args.gpu_device,
                        batch_size=args.batch_size,
                        inference_backend=args.inference_backend,
                    )
                except RuntimeError as exc:
                    print(f"⚠️  Skipping {args.gpu_device} throughput ({exc})", flush=True)
                    gpu_runner = None
                if gpu_runner is not None:
                    for name in sorted(throughput_names, key=_task_name_sort_key):
                        ds = prepared_datasets[name]
                        print(
                            f"▶️  Throughput '{name}' on {args.gpu_device} [{len(ds)} samples]",
                            flush=True,
                        )
                        result = measure_throughput(
                            name,
                            ds,
                            gpu_runner,
                            min_run_chars=args.min_run,
                            device_name=args.gpu_device,
                        )
                        print(
                            f"    [{name}/{args.gpu_device}] {_format_bytes_per_sec(result.throughput)} • {result.elapsed:.2f}s",
                            flush=True,
                        )
                        throughput_results.append(result)
            else:
                print("⚠️  GPU device not available; skipping CUDA throughput benchmarks.", flush=True)

        report_path = _resolve_report_path(args)

        if not task_metrics and not throughput_results:
            raise RuntimeError("No tasks were evaluated. Check if the evaluation data directory contains valid datasets.")

        write_report(
            report_path,
            manifest=manifest,
            args=args,
            task_metrics=task_metrics,
            throughput_results=throughput_results,
            monitor_b_report=monitor_b_report,
        )
        total_elapsed = time.perf_counter() - overall_start
        print(
            f"✅ Evaluation complete in {total_elapsed:.1f}s. "
            f"Artifacts written to {report_path.parent} (report: {report_path})",
            flush=True,
        )
        return 0

    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"\n❌ Error during evaluation: {str(e)}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
