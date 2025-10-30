"""
Functions for generating training windows from datasets, including pure, mixed,
and line-injected augmentation. Also includes the multi-threaded prefetcher.
"""
import math
import random
import re
import threading
import queue
import time
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING, Any

import numpy as np
import datasets as hfds

import config as cfg

if TYPE_CHECKING:
    from config import DataConfig
from data_utils import bytes_from_text
from token_utils import sanitize_tokens

# ---------------------------
# Helpers
# ---------------------------

def _random_example(ds) -> Optional[dict]:
    """Return a random example from a (non-streaming) HF Dataset.
    Falls back to a best-effort sample for streaming/unsized datasets.
    """
    n = None
    try:
        n = len(ds)  # Works for non-streaming Arrow datasets
    except Exception:
        pass

    if n is not None and n > 0:
        idx = random.randrange(0, n)
        return ds[idx]


def _sample_nonempty_example(
    dsets_by_lang: Dict[int, hfds.Dataset],
    lids: List[int],
    *,
    prefer_lid: Optional[int] = None,
    per_lid_attempts: int = 4,
    total_attempts: int = 32,
) -> Tuple[Optional[int], Optional[dict], Optional[np.ndarray]]:
    """Best-effort draw of an example that yields at least one UTF-8 byte."""
    if not lids:
        return None, None, None
    attempts = 0
    while attempts < total_attempts:
        if prefer_lid is not None and attempts == 0:
            lid = prefer_lid
        else:
            lid = random.choice(lids)
        ds = dsets_by_lang.get(lid)
        if ds is None:
            attempts += 1
            continue
        for _ in range(per_lid_attempts):
            ex = _random_example(ds)
            if not ex:
                continue
            content = ex.get("content", "")
            if not content:
                continue
            byte_content = bytes_from_text(content)
            if byte_content.size > 0:
                return lid, ex, byte_content
        attempts += 1
    return None, None, None


def _sample_mixed_segment_count(max_segments: int = 3, continue_prob: float = 0.2) -> int:
    """Geometric-like sampler favouring fewer segments when mixing windows."""
    count = 1
    while count < max_segments and random.random() < continue_prob:
        count += 1
    return count


def _choose_segment_slice(
    byte_content: np.ndarray,
    seg_len: int,
    min_letters: int = 4,
    guard_len: int = 64,
    max_attempts: int = 6,
    return_start: bool = False,
) -> np.ndarray:
    """
    Pick a slice of ``byte_content`` up to ``seg_len`` bytes. For shorter slices we
    bias toward samples that contain at least ``min_letters`` ASCII alphabetic chars.
    """
    total = int(len(byte_content))
    if seg_len <= 0 or total == 0:
        return np.empty((0,), dtype=np.uint8)

    seg_len = min(seg_len, total)
    chosen_start = 0
    if seg_len > guard_len:
        if total == seg_len:
            segment = byte_content[:seg_len]
            return (segment, chosen_start) if return_start else segment
        start = random.randint(0, total - seg_len)
        segment = byte_content[start:start + seg_len]
        return (segment, start) if return_start else segment

    best_slice = byte_content[:seg_len]
    best_start = 0
    best_letters = -1
    for _ in range(max_attempts):
        if total == seg_len:
            start = 0
        else:
            start = random.randint(0, total - seg_len)
        segment = byte_content[start:start + seg_len]
        text = segment.tobytes().decode("utf-8", "ignore")
        letters = sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text)
        if letters >= min_letters:
            return (segment, start) if return_start else segment
        if letters > best_letters:
            best_slice = segment
            best_start = start
            best_letters = letters
    return (best_slice, best_start) if return_start else best_slice


def _lang_name(lid: int) -> str:
    return cfg.ID2LANG.get(int(lid), f"id_{int(lid)}")


def _extract_example_source(example: Optional[dict]) -> str:
    if not example:
        return "unknown"

    def _stringify(val) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            for item in val:
                s = _stringify(item)
                if s:
                    return s
            return None
        text = str(val)
        return text.strip() or None

    primary_keys = [
        "source_path",
        "file_path",
        "filepath",
        "path",
        "filename",
        "source",
        "source_file",
        "doc_path",
        "doc_id",
        "uid",
        "id",
        "url",
    ]
    for key in primary_keys:
        if key in example:
            candidate = _stringify(example.get(key))
            if candidate:
                return candidate

    for meta_key in ("meta", "metadata", "_metadata"):
        meta_val = example.get(meta_key)
        if isinstance(meta_val, dict):
            for key in primary_keys:
                if key in meta_val:
                    candidate = _stringify(meta_val.get(key))
                    if candidate:
                        return candidate
            for key, value in meta_val.items():
                candidate = _stringify(value)
                if candidate:
                    return f"{key}:{candidate}"

    content_preview = _stringify(example.get("content"))
    if content_preview:
        return f"content:{content_preview[:40]}..."

    return "unknown"


def _compute_final_segments(labels_u8: np.ndarray) -> List[Dict[str, int]]:
    segments = []
    pad_id = cfg.PAD_ID
    L = int(labels_u8.shape[0])
    i = 0
    while i < L:
        lid = int(labels_u8[i])
        j = i
        while j < L and int(labels_u8[j]) == lid:
            j += 1
        if lid != pad_id and i != j:
            segments.append(
                {
                    "language_id": lid,
                    "start": i,
                    "end": j,
                    "length": j - i,
                }
            )
        i = j
    return segments


def _apply_final_byte_contributions(metadata: Optional[Dict], source_idx_bytes: Optional[np.ndarray]):
    if not metadata or source_idx_bytes is None or source_idx_bytes.size == 0:
        return
    samples = metadata.get("samples") or []
    if not samples:
        return

    counts: Dict[int, Dict[str, Any]] = {}
    current_idx: Optional[int] = None
    run_start = 0

    for pos, raw_idx in enumerate(source_idx_bytes.tolist()):
        src_idx = int(raw_idx)
        if src_idx == current_idx:
            continue
        if current_idx is not None and current_idx >= 0:
            span_len = pos - run_start
            if span_len > 0:
                entry = counts.setdefault(current_idx, {"bytes": 0, "spans": []})
                entry["bytes"] += span_len
                entry["spans"].append((run_start, pos))
        run_start = pos
        current_idx = src_idx

    if current_idx is not None and current_idx >= 0:
        span_len = len(source_idx_bytes) - run_start
        if span_len > 0:
            entry = counts.setdefault(current_idx, {"bytes": 0, "spans": []})
            entry["bytes"] += span_len
            entry["spans"].append((run_start, len(source_idx_bytes)))

    for idx, data in counts.items():
        if idx < 0 or idx >= len(samples):
            continue
        sample = samples[idx]
        sample["final_bytes"] = int(data["bytes"])
        sample["final_spans"] = [
            {"start": int(start), "end": int(end)}
            for start, end in data["spans"]
            if end > start
        ]

    for sample in samples:
        sample.setdefault("final_bytes", 0)
        sample.setdefault("final_spans", [])
# ---------------------------
# Window builders (supports partial fill + masking)
# ---------------------------

def _make_pure_window_impl(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    collect_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    meta = {"samples": []} if collect_meta else None
    lids = list(dsets_by_lang.keys())
    x = np.full((target_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    y = np.full((target_len,), cfg.PAD_ID, dtype=np.uint8)
    if not lids:
        return sanitize_tokens(x), y, meta
    lid, ex, byte_content = _sample_nonempty_example(dsets_by_lang, lids)
    if ex is None or byte_content is None or byte_content.size == 0:
        if collect_meta:
            meta["samples"].append(
                {
                    "origin": "base",
                    "language_id": -1,
                    "language": "unknown",
                    "source": "unknown",
                    "status": "failed_to_sample_nonempty",
                }
            )
        return sanitize_tokens(x), y, meta

    total_len = int(byte_content.shape[0])
    start = 0
    slice_bytes = byte_content
    if total_len >= target_len:
        start = random.randint(0, total_len - target_len)
        slice_bytes = byte_content[start:start + target_len]

    L = int(slice_bytes.shape[0])
    if L <= 0:
        if collect_meta:
            meta["samples"].append(
                {
                    "origin": "base",
                    "language_id": int(lid),
                    "language": _lang_name(lid),
                    "source": _extract_example_source(ex),
                    "status": "zero_bytes",
                }
            )
        return sanitize_tokens(x), y, meta

    x[:L] = slice_bytes.astype(np.int32)
    y[:L] = lid
    if collect_meta:
        meta["samples"].append(
            {
                "origin": "base",
                "language_id": int(lid),
                "language": _lang_name(lid),
                "source": _extract_example_source(ex),
                "bytes": int(L),
                "start": 0,
                "end": int(L),
                "source_offset": int(start),
            }
        )
    return sanitize_tokens(x), y, meta


def make_pure_window(dsets_by_lang: Dict[int, hfds.Dataset],
                     target_len: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y, _ = _make_pure_window_impl(dsets_by_lang, target_len, collect_meta=False)
    return sanitize_tokens(x), y


def _make_mixed_window_impl(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    min_seg: int,
    collect_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    # inputs are int32 to allow PAD_BYTE_ID=256
    x_buf = np.full((target_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    y_buf = np.full((target_len,), cfg.PAD_ID, dtype=np.uint8)
    meta = {"samples": []} if collect_meta else None

    lids = list(dsets_by_lang.keys())
    if not lids:
        return sanitize_tokens(x_buf), y_buf, meta

    nsegs = _sample_mixed_segment_count()
    left, pos = target_len, 0
    for i in range(nsegs):
        seg_len = random.randint(min_seg, left) if left > min_seg else left
        if i == nsegs - 1:
            seg_len = left
        if seg_len <= 0:
            break

        preferred = random.choice(lids)
        lid, ex, byte_content = _sample_nonempty_example(
            dsets_by_lang, lids, prefer_lid=preferred
        )
        if ex is None or byte_content is None or byte_content.size == 0:
            if collect_meta:
                meta["samples"].append(
                    {
                        "origin": "base",
                        "language_id": int(preferred),
                        "language": _lang_name(preferred),
                        "source": _extract_example_source(ex),
                        "start": int(pos),
                        "end": int(pos),
                        "bytes": 0,
                        "requested_bytes": int(seg_len),
                        "status": "failed_to_sample_nonempty",
                    }
                )
            pos += seg_len
            left -= seg_len
            continue

        if collect_meta:
            segment, src_start = _choose_segment_slice(
                byte_content, seg_len, return_start=True
            )
        else:
            segment = _choose_segment_slice(byte_content, seg_len)
            src_start = None
        L = int(segment.shape[0])
        if L > 0:
            x_buf[pos:pos + L] = segment[:L].astype(np.int32)
            y_buf[pos:pos + L] = lid
        if collect_meta:
            entry = {
                "origin": "base",
                "language_id": int(lid),
                "language": _lang_name(lid),
                "source": _extract_example_source(ex),
                "start": int(pos),
                "end": int(pos + L),
                "bytes": int(L),
                "requested_bytes": int(seg_len),
            }
            if src_start is not None:
                entry["source_offset"] = int(src_start)
            if L == 0:
                entry["status"] = "zero_bytes"
            meta["samples"].append(entry)

        pos += seg_len
        left -= seg_len

    return sanitize_tokens(x_buf), y_buf, meta


def make_mixed_window(dsets_by_lang: Dict[int, hfds.Dataset],
                      target_len: int,
                      min_seg: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y, _ = _make_mixed_window_impl(dsets_by_lang, target_len, min_seg, collect_meta=False)
    return sanitize_tokens(x), y


# ---------------------------
# Shared sampler for train/eval/preview
# ---------------------------

def _line_inject_mode_prob(data_cfg: "DataConfig") -> float:
    return max(0.0, getattr(data_cfg, "line_inject_prob", 0.0))


def _resolve_mix_probability(data_cfg: "DataConfig") -> float:
    mix_prob = getattr(data_cfg, "mix_prob", None)
    pure_prob = max(0.0, getattr(data_cfg, "pure_prob", 0.0))
    line_mode_prob = _line_inject_mode_prob(data_cfg)
    if mix_prob is None:
        mix_prob = 1.0 - pure_prob - line_mode_prob
    return max(0.0, mix_prob)


def _choose_window_mode(data_cfg: "DataConfig") -> str:
    pure_prob = max(0.0, getattr(data_cfg, "pure_prob", 0.0))
    line_mode_prob = _line_inject_mode_prob(data_cfg)
    mix_prob = _resolve_mix_probability(data_cfg)
    total = pure_prob + line_mode_prob + mix_prob
    if total <= 0.0:
        return "mixed"

    r = random.random() * total
    if r < pure_prob:
        return "pure"
    if r < pure_prob + line_mode_prob:
        return "line_inject"
    return "mixed"


def _make_training_window_internal(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    data_cfg: "DataConfig",
    collect_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Draw a window using the same probability distribution as the training
    prefetchers, including mixed windows, line injection, optional overlays,
    and tail padding variation.
    """
    metadata = None
    if collect_meta:
        metadata = {
            "mode": "",
            "requested_mode": "",
            "samples": [],
            "line_injections": [],
            "overlays": [],
            "host": None,
            "final_segments": [],
            "length": int(target_len),
        }

    mode = _choose_window_mode(data_cfg)
    resolved_mode = mode
    if mode == "pure":
        x, y, partial = _make_pure_window_impl(dsets_by_lang, target_len, collect_meta)
        if collect_meta and metadata is not None:
            metadata["mode"] = "pure"
            metadata["requested_mode"] = "pure"
            if partial:
                metadata["samples"].extend(partial.get("samples", []))
    elif mode == "line_inject":
        x, y, partial = _make_line_injected_window_impl(dsets_by_lang, target_len, data_cfg, collect_meta)
        if collect_meta and metadata is not None:
            metadata["mode"] = "line_inject"
            metadata["requested_mode"] = "line_inject"
            if partial:
                metadata["samples"].extend(partial.get("samples", []))
                metadata["line_injections"].extend(partial.get("line_injections", []))
                metadata["host"] = partial.get("host", metadata.get("host"))
    else:
        x, y, partial = _make_mixed_window_impl(dsets_by_lang, target_len, data_cfg.min_seg_len, collect_meta)
        if collect_meta and metadata is not None:
            metadata["mode"] = "mixed"
            metadata["requested_mode"] = "mixed"
            if partial:
                metadata["samples"].extend(partial.get("samples", []))

    # Optionally overlay mixed slices on top of base window
    both_prob = getattr(data_cfg, "both_prob", 0.2)
    if both_prob > 0.0 and random.random() < both_prob:
        xm, ym, overlay_meta = _make_mixed_window_impl(
            dsets_by_lang,
            target_len,
            data_cfg.min_seg_len,
            collect_meta,
        )
        nslices = random.randint(1, 3)
        for _ in range(nslices):
            max_len = max(data_cfg.min_seg_len, target_len // 4)
            seg_len = random.randint(data_cfg.min_seg_len, max_len)
            if seg_len >= target_len:
                seg_len = target_len - 1
            start = random.randint(0, target_len - seg_len)
            end = start + seg_len
            x[start:end] = xm[start:end]
            y[start:end] = ym[start:end]
            if collect_meta and metadata is not None:
                overlay_entry = {
                    "start": int(start),
                    "end": int(end),
                    "mode": "mixed",
                    "segments": [],
                }
                if overlay_meta:
                    for seg in overlay_meta.get("samples", []):
                        seg_start = int(seg.get("start", 0))
                        seg_end = int(seg.get("end", 0))
                        overlap_start = max(start, seg_start)
                        overlap_end = min(end, seg_end)
                        if overlap_start >= overlap_end:
                            continue
                        new_seg = dict(seg)
                        new_seg["origin"] = "overlay"
                        new_seg["start"] = int(overlap_start)
                        new_seg["end"] = int(overlap_end)
                        new_seg["bytes"] = int(overlap_end - overlap_start)
                        overlay_entry["segments"].append(new_seg)
                        metadata["samples"].append(new_seg)
                metadata["overlays"].append(overlay_entry)
                resolved_mode = "mixed" if resolved_mode != "line_inject" else resolved_mode

    # Tail padding variation mirrors training augmentation
    pad_tail_prob = getattr(data_cfg, "pad_tail_prob", 0.1)
    if pad_tail_prob > 0.0 and random.random() < pad_tail_prob:
        max_frac = getattr(data_cfg, "pad_tail_max_frac", 0.9)
        max_pad = max(1, int(target_len * max_frac))
        pad_len = random.randint(0, max_pad)
        if pad_len > 0:
            content_end = max(1, target_len - pad_len)
            # Keep the longest contiguous non-pad block by sliding the window if possible
            last_content = np.where(y != cfg.PAD_ID)[0]
            if last_content.size and last_content[-1] + 1 < content_end:
                shift = min(content_end - (int(last_content[-1]) + 1), pad_len)
                x = np.roll(x, -shift)
                y = np.roll(y, -shift)
                x[-shift:] = cfg.PAD_BYTE_ID
                y[-shift:] = cfg.PAD_ID
            else:
                x[content_end:] = cfg.PAD_BYTE_ID
                y[content_end:] = cfg.PAD_ID

    if not np.any(y != cfg.PAD_ID):
        fallback_x, fallback_y, fallback_meta = _make_pure_window_impl(
            dsets_by_lang, target_len, collect_meta
        )
        if not np.any(fallback_y != cfg.PAD_ID):
            raise RuntimeError("Failed to sample non-empty training window")
        x, y = fallback_x, fallback_y
        resolved_mode = "pure"
        if collect_meta and metadata is not None:
            prev_requested = metadata.get("requested_mode") or metadata.get("mode") or mode
            metadata["mode"] = "pure"
            metadata["requested_mode"] = prev_requested
            metadata["line_injections"] = []
            metadata["overlays"] = []
            metadata["host"] = None
            metadata["samples"] = fallback_meta.get("samples", []) if fallback_meta else []
            metadata["fallback"] = "pure_due_to_empty"

    if collect_meta and metadata is not None:
        final_segments = []
        for seg in _compute_final_segments(y):
            final_segments.append(
                {
                    **seg,
                    "language": _lang_name(seg["language_id"]),
                }
            )
        metadata["final_segments"] = final_segments
        unique_langs = {seg["language_id"] for seg in final_segments}
        if metadata["line_injections"]:
            actual_mode = "line_inject"
        elif len(unique_langs) <= 1:
            actual_mode = resolved_mode if resolved_mode == "line_inject" else "pure"
        else:
            actual_mode = "mixed"
        metadata["actual_mode"] = actual_mode
        metadata["mode"] = actual_mode
    return sanitize_tokens(x), y, metadata


def make_training_window(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    data_cfg: "DataConfig",
) -> Tuple[np.ndarray, np.ndarray]:
    x, y, _ = _make_training_window_internal(dsets_by_lang, target_len, data_cfg, collect_meta=False)
    return sanitize_tokens(x), y


def make_training_window_with_metadata(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    data_cfg: "DataConfig",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    x, y, meta = _make_training_window_internal(dsets_by_lang, target_len, data_cfg, collect_meta=True)
    return sanitize_tokens(x), y, meta or {}


# ---------------------------
# LINE-LEVEL INJECTION
# ---------------------------

def _split_keepends_lines(text: str) -> List[str]:
    return text.splitlines(keepends=True) if text else []


def _count_letters(text: str) -> int:
    return sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text)


def _sample_truncated_exp_lines(lam: float, max_lines: int) -> int:
    k = int(math.ceil(random.expovariate(lam))) if lam > 0 else 1
    return max(1, min(k, max_lines))


def _choose_injection_boundaries(host_text: str, skip_top_min: int, skip_top_max: int, max_inj: int) -> List[int]:
    lines = _split_keepends_lines(host_text)
    if not lines:
        return []
    boundaries = np.cumsum([len(ln) for ln in lines]).tolist()
    skip = random.randint(skip_top_min, max(skip_top_min, skip_top_max))
    valid = [b for b in boundaries if b > boundaries[skip - 1]] if skip < len(boundaries) else []
    if not valid:
        return []
    n = 1
    # More injections per file
    while n < max_inj and random.random() < 0.25:
        n += 1
    n = min(n, len(valid))
    picks = random.sample(valid, k=n)
    return sorted(picks, reverse=True)


def _leading_indent_of_line(chars: List[str], at_char_idx: int) -> str:
    i = at_char_idx - 1
    while i >= 0 and chars[i] != "\n":
        i -= 1
    line_start = i + 1
    j = line_start
    indent = []
    while j < len(chars) and chars[j] in (" ", "\t"):
        indent.append(chars[j]); j += 1
    return "".join(indent)


def _prepare_donor_block(donor_text: str, data_cfg: "DataConfig", insertion_indent: str) -> str:
    lines = _split_keepends_lines(donor_text)
    if not lines:
        return ""

    has_interior = len(lines) > 2
    if has_interior:
        interior_start = 1
        interior_end = len(lines) - 2
        if interior_start > interior_end:
            has_interior = False
    if not has_interior:
        interior_start = 0
        interior_end = len(lines) - 1

    if has_interior:
        skip_min = max(interior_start, min(data_cfg.donor_skip_top_min, interior_end))
        skip_max = min(interior_end, max(data_cfg.donor_skip_top_min, data_cfg.donor_skip_top_max))
        if skip_min > skip_max:
            skip_min = skip_max = interior_end
        start_idx = random.randint(skip_min, skip_max)
        end_limit = len(lines) - 1  # keep the final line untouched when interior exists
    else:
        start_idx = 0
        end_limit = len(lines)

    L = _sample_truncated_exp_lines(data_cfg.line_inject_exp_rate, data_cfg.line_inject_max_lines)
    end_idx = min(end_limit, start_idx + L)
    if end_idx <= start_idx:
        end_idx = min(len(lines), start_idx + 1)
    pick = lines[start_idx:end_idx]
    if not pick:
        if has_interior:
            return ""
        pick = lines[:]  # fall back to whole donor when no interior exists

    if len(pick) == 1 and len(pick[0]) < data_cfg.line_inject_min_single_len:
        if end_idx < end_limit:
            pick.append(lines[end_idx])
            end_idx += 1
        elif start_idx > interior_start:
            start_idx -= 1
            pick.insert(0, lines[start_idx])

    if has_interior:
        max_total_lines = min(len(lines) - 2, data_cfg.line_inject_max_lines)
    else:
        max_total_lines = min(len(lines), data_cfg.line_inject_max_lines)
    min_letters = getattr(data_cfg, "line_inject_min_letters", 4)

    def pick_letter_count() -> int:
        return sum(_count_letters(ln) for ln in pick)

    letter_count = pick_letter_count()
    while letter_count < min_letters and len(pick) < max_total_lines:
        expanded = False
        if end_idx < end_limit and len(pick) < max_total_lines:
            pick.append(lines[end_idx])
            end_idx += 1
            expanded = True
            letter_count = pick_letter_count()
        if letter_count < min_letters and len(pick) < max_total_lines and start_idx > interior_start:
            start_idx -= 1
            pick.insert(0, lines[start_idx])
            expanded = True
            letter_count = pick_letter_count()
        if not expanded:
            break

    strip_prob = getattr(data_cfg, "line_inject_strip_prob", 0.8)
    force_strip = random.random() < strip_prob
    w_none, w_l, w_r, w_b = data_cfg.strip_weights
    mode = random.choices(["none", "lstrip", "rstrip", "strip"], weights=[w_none, w_l, w_r, w_b], k=1)[0]

    processed = []
    for ln in pick:
        core = ln.rstrip('\n')
        if force_strip:
            core = core.strip()
        elif mode == "lstrip":
            core = core.lstrip()
        elif mode == "rstrip":
            core = core.rstrip()
        elif mode == "strip":
            core = core.strip()

        if random.random() < data_cfg.reindent_prob:
            core = insertion_indent + core.lstrip()

        processed.append(core + ('\n' if ln.endswith('\n') else ''))

    block = "".join(processed)
    if _count_letters(block) < min_letters:
        return ""
    return block


def _insert_block_at(
    chars: List[str],
    labels: List[int],
    sources: Optional[List[int]],
    idx: int,
    block: str,
    lid: int,
    source_idx: Optional[int],
):
    if not block:
        return
    ins_chars = list(block)
    ins_labs = [lid] * len(ins_chars)
    chars[idx:idx] = ins_chars
    labels[idx:idx] = ins_labs
    if sources is not None:
        assign_idx = -1 if source_idx is None else int(source_idx)
        sources[idx:idx] = [assign_idx] * len(ins_chars)


def _ensure_merge_without_newline(
    chars: List[str],
    labels: List[int],
    sources: Optional[List[int]],
    idx: int,
) -> int:
    if idx > 0 and chars[idx - 1] == "\n":
        del chars[idx - 1]
        del labels[idx - 1]
        if sources is not None:
            del sources[idx - 1]
        return idx - 1
    return idx


def _to_bytes_with_byte_labels(
    chars: List[str],
    char_labels: List[int],
    char_sources: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    xb, yb, sb = [], [], []
    for idx, (ch, lid) in enumerate(zip(chars, char_labels)):
        bs = ch.encode("utf-8", "ignore")
        xb.extend(bs)
        yb.extend([lid] * len(bs))
        if char_sources is not None:
            src_idx = char_sources[idx] if idx < len(char_sources) else -1
            sb.extend([src_idx] * len(bs))
    xb_arr = np.array(xb, dtype=np.uint8)
    yb_arr = np.array(yb, dtype=np.uint8)
    if char_sources is not None:
        sb_arr = np.array(sb, dtype=np.int32) if sb else np.empty((0,), dtype=np.int32)
        return xb_arr, yb_arr, sb_arr
    return xb_arr, yb_arr, None


def _make_line_injected_window_impl(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    data_cfg: "DataConfig",
    collect_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    meta = {"samples": [], "line_injections": [], "host": None} if collect_meta else None
    lids = list(dsets_by_lang.keys())
    text_lid = None
    if hasattr(cfg, "LANG2ID"):
        text_lid = cfg.LANG2ID.get("text")
    if text_lid is None and hasattr(cfg, "ID2LANG"):
        text_lid = next((lid for lid, name in cfg.ID2LANG.items() if name == "text"), None)

    x = np.full((target_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    y = np.full((target_len,), cfg.PAD_ID, dtype=np.uint8)
    if not lids:
        return sanitize_tokens(x), y, meta

    preferred_host = random.choice(lids)
    host_lid, host_ex, _ = _sample_nonempty_example(
        dsets_by_lang, lids, prefer_lid=preferred_host
    )
    if host_ex is None:
        if collect_meta and meta is not None:
            meta["samples"].append(
                {
                    "origin": "host",
                    "language_id": int(preferred_host),
                    "language": _lang_name(preferred_host),
                    "source": _extract_example_source(host_ex),
                    "status": "failed_to_sample_nonempty",
                }
            )
        return _make_pure_window_impl(dsets_by_lang, target_len, collect_meta)

    host_text = host_ex.get("content", "") or ""
    if not host_text:
        return _make_pure_window_impl(dsets_by_lang, target_len, collect_meta)

    donor_lids = [lid for lid in lids if lid != text_lid] if text_lid is not None else list(lids)
    chars, labs = list(host_text), [host_lid] * len(host_text)
    char_sources: Optional[List[int]] = None
    host_idx: Optional[int] = None
    host_source = _extract_example_source(host_ex)
    if collect_meta and meta is not None:
        host_entry = {
            "origin": "host",
            "language_id": int(host_lid),
            "language": _lang_name(host_lid),
            "source": host_source,
            "chars": len(chars),
            "requested_chars": len(chars),
        }
        meta["host"] = {
            "language_id": int(host_lid),
            "language": _lang_name(host_lid),
            "source": host_source,
            "characters": len(chars),
            "sample_index": len(meta["samples"]),
        }
        meta["samples"].append(host_entry)
        host_idx = len(meta["samples"]) - 1
        char_sources = [host_idx] * len(chars)

    boundaries = _choose_injection_boundaries(
        host_text, data_cfg.host_skip_top_min, data_cfg.host_skip_top_max, data_cfg.line_inject_max_injections
    )

    for bidx in boundaries:
        candidates = donor_lids
        if not candidates:
            continue
        if not data_cfg.allow_same_lang_injection:
            alts = [lid for lid in candidates if lid != host_lid]
            if alts:
                candidates = alts
            elif host_lid in candidates:
                continue
        donor_choice = random.choice(candidates)
        donor_lid, donor_ex, _ = _sample_nonempty_example(
            dsets_by_lang, candidates, prefer_lid=donor_choice
        )
        if donor_ex is None:
            continue
        donor_text = donor_ex.get("content", "") or ""

        insertion_indent = _leading_indent_of_line(chars, bidx)
        donor_block = _prepare_donor_block(donor_text, data_cfg, insertion_indent)
        if not donor_block:
            continue

        newline_max = max(
            0,
            getattr(
                data_cfg,
                "inject_extra_newlines_max",
                getattr(data_cfg, "line_inject_extra_newlines_max", 4),
            ),
        )
        pre_pad_count = random.randint(0, newline_max) if newline_max > 0 else 0
        post_pad_count = random.randint(0, newline_max) if newline_max > 0 else 0
        start_with_newline_prob = getattr(data_cfg, "start_with_newline_prob", 0.5)
        if random.random() > start_with_newline_prob:
            bidx = _ensure_merge_without_newline(chars, labs, char_sources, bidx)
            pre_pad_count = 0

        if pre_pad_count > 0:
            pre_block = "\n" * pre_pad_count
            _insert_block_at(chars, labs, char_sources, bidx, pre_block, host_lid, host_idx)
            bidx += len(pre_block)
        donor_insert_idx = bidx

        sample_idx = None
        donor_source = _extract_example_source(donor_ex)

        if collect_meta and meta is not None:
            sample_entry = {
                "origin": "injection",
                "language_id": int(donor_lid),
                "language": _lang_name(donor_lid),
                "source": donor_source,
                "chars": len(donor_block),
                "requested_chars": len(donor_block),
            }
            meta["samples"].append(sample_entry)
            sample_idx = len(meta["samples"]) - 1
        _insert_block_at(chars, labs, char_sources, donor_insert_idx, donor_block, donor_lid, sample_idx)
        bidx = donor_insert_idx + len(donor_block)

        if post_pad_count > 0:
            post_block = "\n" * post_pad_count
            _insert_block_at(chars, labs, char_sources, bidx, post_block, host_lid, host_idx)
            bidx += len(post_block)

        if collect_meta and meta is not None:
            meta["line_injections"].append(
                {
                    "language_id": int(donor_lid),
                    "language": _lang_name(donor_lid),
                    "source": donor_source,
                    "insert_char_index": int(donor_insert_idx),
                    "chars_inserted": len(donor_block),
                    "newlines_before": int(pre_pad_count),
                    "newlines_after": int(post_pad_count),
                    "preview": donor_block[:120],
                    "sample_index": sample_idx,
                }
            )

    xb_u8, yb_u8, source_idx_bytes = _to_bytes_with_byte_labels(chars, labs, char_sources)

    # Windowing/padding
    if len(xb_u8) >= target_len:
        start = random.randint(0, len(xb_u8) - target_len)
        x = xb_u8[start:start + target_len].astype(np.int32)
        y = yb_u8[start:start + target_len]
        if source_idx_bytes is not None and source_idx_bytes.size:
            source_idx_bytes = source_idx_bytes[start:start + target_len]
    else:
        x[:len(xb_u8)] = xb_u8.astype(np.int32)
        y[:len(yb_u8)] = yb_u8
        if source_idx_bytes is not None and source_idx_bytes.size:
            pad_len = target_len - len(xb_u8)
            if pad_len > 0:
                source_idx_bytes = np.concatenate(
                    [
                        source_idx_bytes,
                        np.full((pad_len,), -1, dtype=np.int32),
                    ]
                )
    if collect_meta and meta is not None:
        if source_idx_bytes is not None and source_idx_bytes.size:
            _apply_final_byte_contributions(meta, source_idx_bytes)
            samples_list = meta.get("samples", [])
            for inj in meta.get("line_injections", []):
                idx = inj.get("sample_index")
                if isinstance(idx, int) and 0 <= idx < len(samples_list):
                    inj["final_bytes"] = samples_list[idx].get("final_bytes", 0)
                    inj["final_spans"] = samples_list[idx].get("final_spans", [])
            host_info = meta.get("host")
            if isinstance(host_info, dict):
                idx = host_info.get("sample_index")
                if isinstance(idx, int) and 0 <= idx < len(samples_list):
                    host_info["final_bytes"] = samples_list[idx].get("final_bytes", 0)
                    host_info["final_spans"] = samples_list[idx].get("final_spans", [])
        for inj in meta.get("line_injections", []):
            inj.pop("sample_index", None)
        host_info = meta.get("host")
        if isinstance(host_info, dict):
            host_info.pop("sample_index", None)
    return sanitize_tokens(x), y, meta


def make_line_injected_window(dsets_by_lang: Dict[int, hfds.Dataset],
                              target_len: int,
                              data_cfg: "DataConfig") -> Tuple[np.ndarray, np.ndarray]:
    x, y, _ = _make_line_injected_window_impl(dsets_by_lang, target_len, data_cfg, collect_meta=False)
    return sanitize_tokens(x), y


# ---------------------------
# Prefetcher (multi-thread, variable-length buckets)
# ---------------------------

class PrefetchBatcher:
    def __init__(self, dsets_by_lang, data_cfg: "DataConfig"):
        self.dsets_by_lang = dsets_by_lang
        self.cfg = data_cfg
        # Smaller queue size to prevent memory buildup
        self.q = queue.Queue(maxsize=max(2, data_cfg.prefetch_batches))
        self.stop_flag = threading.Event()
        self.buckets = data_cfg.buckets()
        self.threads: List[threading.Thread] = []
        self._gc_counter = 0
        # Use fewer worker threads
        for wid in range(max(1, data_cfg.num_workers // 2)):
            t = threading.Thread(target=self._worker, args=(wid,), daemon=True)
            t.start()
            self.threads.append(t)

    def _worker(self, wid: int):
        random.seed(self.cfg.seed ^ wid ^ int(time.time()))
        hold = max(1, self.cfg.bucket_hold_steps)
        L = random.choice(self.buckets)
        k = 0
        while not self.stop_flag.is_set():
            if k % hold == 0:
                L = random.choice(self.buckets)
            k += 1

            xb = np.full((self.cfg.batch_size, L), cfg.PAD_BYTE_ID, dtype=np.int32)
            yb = np.full((self.cfg.batch_size, L), cfg.PAD_ID, dtype=np.uint8)

            for i in range(self.cfg.batch_size):
                x, y = make_training_window(self.dsets_by_lang, L, self.cfg)
                xb[i], yb[i] = x, y

            try:
                self.q.put((xb, yb), timeout=1.0)
            except queue.Full:
                continue

    def get(self):
        return self.q.get()

    def close(self):
        self.stop_flag.set()
        # Drain queue to unblock workers
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        for t in self.threads:
            t.join(timeout=2.0)
