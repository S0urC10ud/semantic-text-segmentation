#!/usr/bin/env python3
"""
Interactive confusion-matrix explorer that pairs each cell with real
validation windows rendered in the familiar segment viewer style.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import collections
import importlib.util
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Ensure the repo root is importable as a package when invoked from within segment_viewer
if str(REPO_ROOT / "train") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "train"))

import utils.config as cfg
from utils.data import prepare_dsets_by_lang_with_splits
from utils.metrics_helper import accumulate_confusion, compute_metrics_from_confusion
from utils.monitor_eval import load_monitor_memmaps
from utils.window_generator import make_training_window_with_metadata

try:
    from .core import (
        DEFAULT_CHANNELS,
        DEFAULT_CHUNK_SIZE,
        Predictor,
        _apply_label_mapping,
        _hex_to_rgba,
        _hex_to_rgba_confidence,
        _infer_checkpoint_architecture,
        _load_checkpoint_hparams,
        _normalize_input_text,
        _resolve_colors,
        _resolve_hparam,
        _resolve_langs_and_display,
        _sanitize_model_bytes,
        make_slug,
    )
except ImportError:  # pragma: no cover
    from core import (
        DEFAULT_CHANNELS,
        DEFAULT_CHUNK_SIZE,
        Predictor,
        _apply_label_mapping,
        _hex_to_rgba,
        _hex_to_rgba_confidence,
        _infer_checkpoint_architecture,
        _load_checkpoint_hparams,
        _normalize_input_text,
        _resolve_colors,
        _resolve_hparam,
        _resolve_langs_and_display,
        _sanitize_model_bytes,
        make_slug,
    )

MAX_EXAMPLES_PER_CELL = 64
DEFAULT_DATA_ROOT = (REPO_ROOT / "downloader" / "arrow_out").resolve()


def _load_training_label_order() -> List[str]:
    candidates = [
        REPO_ROOT / "train" / "utils" / "config.py",
        REPO_ROOT / "train" / "config.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(
            "confusion_viewer_config_snapshot", path
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mapping = getattr(module, "LANG2ID", {})
        if not isinstance(mapping, dict):
            continue
        return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    return []


TRAINING_LABEL_ORDER: List[str] = _load_training_label_order()


def _sanitize_label_order(raw: Iterable[Any]) -> List[str]:
    cleaned: List[str] = []
    if not raw:
        return cleaned
    allowed = set(cfg.LANG2ID.keys())
    seen = set()
    dropped: List[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        candidate = ANSI_ESCAPE_RE.sub("", entry).strip()
        if not candidate:
            continue
        if allowed and candidate not in allowed:
            dropped.append(candidate)
            continue
        if candidate in seen:
            continue
        cleaned.append(candidate)
        seen.add(candidate)
    if dropped:
        sample = ", ".join(sorted(set(dropped))[:5])
        print(
            f"⚠️  Ignoring {len(dropped)} unknown labels from checkpoint metadata: {sample}",
            flush=True,
        )
    return cleaned


def _sync_config_with_labels(label_order: List[str]) -> None:
    cfg.LANG2ID.clear()
    for idx, name in enumerate(label_order):
        cfg.LANG2ID[name] = idx
    update_fn = getattr(cfg, "update_lang_mappings", None)
    if callable(update_fn):
        update_fn()


def _resolve_data_root(raw: Optional[str]) -> Path:
    if not raw:
        return DEFAULT_DATA_ROOT
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        (Path.cwd() / path).resolve(),
        (REPO_ROOT / path).resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    # Fall back to first candidate even if it does not exist; downstream code will raise a helpful error.
    return candidates[0]


@dataclass
class SampleRecord:
    tokens: np.ndarray
    labels: np.ndarray
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)
    length: int = 0


@dataclass
class MonitorExamplePointer:
    file_idx: int
    byte_start: int
    byte_end: int
    meta: Dict[str, Any] = field(default_factory=dict)


IGNORED_WHITESPACE_CHARS: Tuple[str, ...] = (" ", "\t", "\n")
IGNORED_WHITESPACE_CHAR_SET = frozenset(IGNORED_WHITESPACE_CHARS)
IGNORED_WHITESPACE_BYTE_VALUES = np.array([ord(ch) for ch in IGNORED_WHITESPACE_CHARS], dtype=np.int32)
SOURCE_NAMES = {0: "segmented", 1: "pure"}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[@-~]")


def _tokens_to_text(tokens: np.ndarray) -> str:
    arr = np.asarray(tokens, dtype=np.int32)
    if arr.size == 0:
        return ""
    mask = arr != cfg.PAD_BYTE_ID
    if not mask.any():
        return ""
    byte_arr = arr[mask].astype(np.uint8)
    # Use a 1:1 byte-to-char mapping so label alignment cannot drift when
    # sanitised bytes (e.g., 0xA4) are present. Non-ASCII bytes become '?'.
    chars: List[str] = []
    for val in byte_arr.tolist():
        if (0x20 <= val <= 0x7E) or val in (0x09, 0x0A, 0x0D):
            chars.append(chr(val))
        else:
            chars.append("?")
    return "".join(chars)


def _summarize_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    summary: Dict[str, Any] = {}
    for key in ("mode", "requested_mode", "actual_mode", "fallback"):
        if key in meta and meta[key]:
            summary[key] = meta[key]
    host = meta.get("host")
    if isinstance(host, dict):
        summary["host_language"] = host.get("language")
    trimmed: List[Dict[str, Any]] = []
    final_segments = meta.get("final_segments") or []
    if isinstance(final_segments, list) and final_segments:
        totals: Dict[str, int] = {}
        for segment in final_segments:
            language = segment.get("language")
            if not language:
                continue
            length = segment.get("length")
            if length is None:
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                try:
                    length = int(end) - int(start)
                except Exception:
                    length = None
            try:
                length_int = int(length) if length is not None else None
            except Exception:
                length_int = None
            if length_int is None or length_int <= 0:
                continue
            totals[language] = totals.get(language, 0) + length_int
        if totals:
            for lang, total in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])):
                trimmed.append({"language": lang, "bytes": total})
    if not trimmed:
        samples = meta.get("samples") or []
        for entry in samples[:3]:
            language = entry.get("language") or entry.get("lang")
            length = entry.get("final_bytes")
            if not length:
                length = entry.get("bytes")
            if not length:
                length = entry.get("length")
            if not length:
                spans = entry.get("final_spans") or entry.get("spans")
                if spans:
                    total = 0
                    for span in spans:
                        try:
                            start = int(span.get("start", 0))
                            end = int(span.get("end", 0))
                        except Exception:
                            continue
                        if end > start:
                            total += end - start
                    if total > 0:
                        length = total
            if language is None and length is None:
                continue
            try:
                bytes_val = int(length)
            except (TypeError, ValueError):
                bytes_val = None
            trimmed.append(
                {
                    "language": language,
                    "bytes": bytes_val,
                }
            )
    if trimmed:
        summary["sources"] = trimmed
    if meta.get("line_injections"):
        summary["line_injections"] = len(meta["line_injections"])
    lp_mode = meta.get("language_pair_mode")
    if isinstance(lp_mode, dict) and lp_mode.get("selected"):
        summary["language_pair_mode"] = {
            "requested": [int(i) for i in lp_mode.get("ids", [])],
            "used": [int(i) for i in lp_mode.get("used_language_ids", [])],
        }
    return summary


def _aggregate_probs(probs: Dict[str, float]) -> Dict[str, float]:
    filtered = {k: float(v) for k, v in probs.items() if isinstance(v, (int, float))}
    if not filtered:
        return {}
    sorted_items = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)
    top = dict(sorted_items[:5])
    remainder = sum(prob for _, prob in sorted_items[5:])
    if remainder > 1e-6:
        top["others"] = remainder
    return top


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _build_segments_html(
    text: str,
    char_labels: List[int],
    char_probs: List[Dict[str, float]],
    true_labels: List[int],
    highlight_mask: List[bool],
    id2name: Dict[int, str],
    id2canon: Dict[int, str],
    id2color: Dict[int, str],
) -> Tuple[str, List[Dict[str, Any]]]:
    if not text:
        return "", []
    segments: List[Tuple[int, int, int]] = []
    if char_labels:
        start = 0
        cur = char_labels[0]
        for idx in range(1, len(char_labels)):
            lbl = char_labels[idx]
            if lbl != cur:
                segments.append((start, idx, cur))
                start = idx
                cur = lbl
        segments.append((start, len(char_labels), cur))

    stats = []
    counts = collections.Counter(char_labels)
    total = max(1, sum(counts.values()))
    for cid, count in counts.items():
        stats.append(
            {
                "id": cid,
                "name": id2name.get(cid, str(cid)),
                "count": int(count),
                "pct": float(100.0 * count / total),
                "color": id2color.get(cid, "#888888"),
            }
        )

    out_html: List[str] = []
    for (start, end, lbl) in segments:
        raw = text[start:end]
        cls = id2canon.get(lbl, f"class-{lbl}")
        color = id2color.get(lbl, "#888888")
        border = _hex_to_rgba(color, 0.35)
        chunk: List[str] = []
        for offset, ch in enumerate(raw):
            char_idx = start + offset
            if ch == "\n":
                chunk.append("<br/>")
                continue
            raw_probs = char_probs[char_idx]
            conf = 1.0
            if isinstance(raw_probs, dict) and raw_probs:
                key = str(int(lbl)) if isinstance(lbl, (int, np.integer)) else str(lbl)
                try:
                    conf = float(raw_probs.get(key, 0.0))
                except Exception:
                    conf = 0.0
            char_bg = _hex_to_rgba_confidence(color, 0.22, conf)
            agg: Dict[str, float] = {}
            for key, value in raw_probs.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                canonical = id2canon.get(idx)
                if canonical is None:
                    continue
                display = id2name.get(idx, canonical)
                agg[display] = agg.get(display, 0.0) + float(value)
            probs = _aggregate_probs(agg)
            payload = _escape_html(json.dumps(probs))
            display_label = _escape_html(id2name.get(lbl, str(lbl)))
            base_class = "char"
            if highlight_mask[char_idx]:
                base_class += " confused"
            true_lbl = id2name.get(true_labels[char_idx], str(true_labels[char_idx]))
            chunk.append(
                f'<span class="{base_class}" style="background-color:{char_bg};" data-probs="{payload}" '
                f'data-label="{display_label}" data-true="{_escape_html(true_lbl)}">'
                f"{_escape_html(ch)}</span>"
            )
        out_html.append(
            f'<span class="seg {cls}" data-label="{_escape_html(id2name.get(lbl, str(lbl)))}" '
            f'style="--seg-color:{color}; background-color: transparent !important; box-shadow: inset 0 -1px 0 {border};">'
            f'{"".join(chunk)}</span>'
        )
    return "".join(out_html), stats


def _remap_colors_for_active_classes(
    matrix: np.ndarray,
    canonical_label_names: List[str],
    colors_arg: Optional[str],
) -> Dict[int, str]:
    total_labels = len(canonical_label_names)
    if total_labels == 0:
        return {}
    if matrix.size == 0:
        return {idx: "#9ba3b4" for idx in range(total_labels)}
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    active_ids = sorted(
        set(np.nonzero(row_totals)[0].tolist()) | set(np.nonzero(col_totals)[0].tolist())
    )
    if not active_ids:
        return {idx: "#9ba3b4" for idx in range(total_labels)}
    subset_names = [canonical_label_names[idx] for idx in active_ids]
    subset_colors = _resolve_colors(subset_names, colors_arg, allow_unknown_named=True)
    color_map = {idx: "#d0d5dd" for idx in range(total_labels)}
    for palette_idx, label_id in enumerate(active_ids):
        color_map[label_id] = subset_colors[palette_idx]
    return color_map


def _generate_window_palette(count: int) -> List[str]:
    if count <= 0:
        return []
    import colorsys

    hues = [idx / max(count, 1) for idx in range(count)]
    palette = []
    for hue in hues:
        r, g, b = colorsys.hls_to_rgb(hue % 1.0, 0.55, 0.7)
        palette.append("#{0:02x}{1:02x}{2:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _build_window_color_map(
    pred_char_labels: List[int],
    true_char_labels: List[int],
    id2canon: Dict[int, str],
    colors_arg: Optional[str],
) -> Tuple[List[int], Dict[int, str]]:
    counts: collections.Counter = collections.Counter()
    for seq in (pred_char_labels, true_char_labels):
        for lbl in seq or []:
            if not isinstance(lbl, (int, np.integer)):
                continue
            label_id = int(lbl)
            if label_id < 0 or label_id not in id2canon:
                continue
            counts[label_id] += 1
    if not counts:
        return [], {}
    ordered_ids = [
        label_id for label_id, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    subset_names = [id2canon[label_id] for label_id in ordered_ids]
    if colors_arg:
        subset_colors = _resolve_colors(subset_names, colors_arg, allow_unknown_named=True)
    else:
        subset_colors = _generate_window_palette(len(ordered_ids))
    color_map = {label_id: subset_colors[idx] for idx, label_id in enumerate(ordered_ids)}
    return ordered_ids, color_map


def _build_true_char_labels(text: str, byte_labels: np.ndarray, predictor: Predictor) -> List[int]:
    labels, _ = predictor._byte_labels_to_char_labels(text, byte_labels, None)
    return labels


def _slice_valid(arr: np.ndarray, length: int) -> np.ndarray:
    return np.asarray(arr[:length], dtype=arr.dtype).copy()


def _collect_confusion_samples(
    predictor: Predictor,
    val_dsets: Dict[int, Any],
    chunk: int,
    batch_size: int,
    batches: int,
    data_cfg: cfg.DataConfig,
    examples_per_cell: int,
    num_classes: int,
    other_threshold: float,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[int]], List[SampleRecord]]:
    """
    Build a confusion matrix over validation windows, optionally routing
    low-confidence predictions into a derived 'other' bucket at index
    num_classes - 1 when other_threshold > 0.
    """
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    cell_examples: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
    samples: List[SampleRecord] = []
    other_id = num_classes - 1
    use_threshold = other_threshold is not None and other_threshold > 0.0
    total = batch_size * batches
    print(f"📊 Sampling {total} windows ({batches} batches × {batch_size}) for validation confusion...", flush=True)
    for batch_idx in range(batches):
        xb = np.full((batch_size, chunk), cfg.PAD_BYTE_ID, dtype=np.int32)
        yb = np.full((batch_size, chunk), cfg.PAD_ID, dtype=np.int32)
        metas: List[Dict[str, Any]] = []
        for i in range(batch_size):
            tokens, labels, meta = make_training_window_with_metadata(
                val_dsets, chunk, data_cfg
            )
            xb[i] = tokens
            yb[i] = labels
            metas.append(meta or {})
        logits = predictor.predict_logits(xb)
        logits_np = np.asarray(logits, dtype=np.float32)
        preds_core = np.argmax(logits_np, axis=-1).astype(np.int32)
        if use_threshold:
            logits_shift = logits_np - logits_np.max(axis=-1, keepdims=True)
            probs = np.exp(logits_shift)
            probs /= np.maximum(probs.sum(axis=-1, keepdims=True), 1e-9)
            max_prob = probs.max(axis=-1)
        else:
            max_prob = None
        mask = (yb != cfg.PAD_ID)
        for i in range(batch_size):
            valid_mask = mask[i]
            if not valid_mask.any():
                continue
            ignored_mask = np.isin(xb[i], IGNORED_WHITESPACE_BYTE_VALUES)
            effective_mask = valid_mask & (~ignored_mask)
            if not effective_mask.any():
                continue
            true_flat = yb[i][effective_mask].astype(np.int32)
            pred_flat = preds_core[i][effective_mask]
            if use_threshold and max_prob is not None:
                max_flat = max_prob[i][effective_mask]
                pred_thresh = pred_flat.copy()
                pred_thresh[max_flat < float(other_threshold)] = int(other_id)
            else:
                pred_thresh = pred_flat
            core_mask = (
                (true_flat >= 0)
                & (true_flat < num_classes)
                & (pred_thresh >= 0)
                & (pred_thresh < num_classes)
            )
            if not core_mask.any():
                continue
            true_core = true_flat[core_mask]
            pred_core = pred_thresh[core_mask]
            accumulate_confusion(conf_mat, true_core, pred_core)
            combos = np.stack([true_core, pred_core], axis=1)
            unique_pairs = np.unique(combos, axis=0)
            if unique_pairs.size == 0:
                continue
            length = int(valid_mask.sum())
            tokens_trim = _slice_valid(xb[i], length)
            labels_trim = _slice_valid(yb[i], length)
            text = _tokens_to_text(tokens_trim)
            record = SampleRecord(
                tokens=tokens_trim,
                labels=labels_trim,
                text=text,
                meta=_summarize_meta(metas[i]),
                length=length,
            )
            sample_id = len(samples)
            samples.append(record)
            for t_id, p_id in unique_pairs:
                key = (int(t_id), int(p_id))
                bucket = cell_examples.setdefault(key, [])
                if len(bucket) < examples_per_cell:
                    bucket.append(sample_id)
        print(f"   ✔ Processed batch {batch_idx + 1}/{batches}", flush=True)
    return conf_mat, cell_examples, samples


def _row_normalize(conf_mat: np.ndarray) -> np.ndarray:
    row_sums = conf_mat.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.divide(
            conf_mat.astype(np.float64),
            row_sums,
            out=np.zeros_like(conf_mat, dtype=np.float64),
            where=row_sums > 0,
        )
    return norm


def _format_metrics(per_class: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
    return {key: [float(x) for x in values.tolist()] for key, values in per_class.items()}


def _make_data_config(chunk: int, batch_size: int, data_root: str) -> cfg.DataConfig:
    cfg_obj = cfg.DataConfig()
    cfg_obj.data_root = data_root
    cfg_obj.window_min_bytes = chunk
    cfg_obj.window_max_bytes = chunk
    cfg_obj.batch_size = batch_size
    return cfg_obj


def _build_sample_payload(
    sample: SampleRecord,
    true_id: int,
    pred_id: int,
    predictor: Predictor,
    id2name: Dict[int, str],
    id2canon: Dict[int, str],
    id2color: Dict[int, str],
    colors_arg: Optional[str],
) -> Dict[str, Any]:
    logits = predictor.predict_logits(sample.tokens)
    preds = np.argmax(logits[0, : sample.length], axis=-1).astype(np.int32)
    probs = jax.nn.softmax(jnp.array(logits[0, : sample.length]), axis=-1)
    probs_np = np.array(probs, dtype=np.float32)
    byte_true = sample.labels[: sample.length]
    text = _normalize_input_text(sample.text)
    pred_char_labels, char_probs = predictor._byte_labels_to_char_labels(text, preds, probs_np)
    true_char_labels, _ = predictor._byte_labels_to_char_labels(text, byte_true, None)
    # Apply the same low-confidence 'other' behavior as the main viewer.
    pred_char_labels, char_probs = predictor._apply_other_threshold(pred_char_labels, char_probs)
    highlight: List[bool] = []
    text_len = len(text)
    true_len = len(true_char_labels)
    for idx, pred_label in enumerate(pred_char_labels):
        true_label = true_char_labels[idx] if idx < true_len else None
        ch = text[idx] if idx < text_len else ""
        should_highlight = (
            ch not in IGNORED_WHITESPACE_CHAR_SET
            and true_label == true_id
            and pred_label == pred_id
        )
        highlight.append(should_highlight)
    palette_ids, window_colors = _build_window_color_map(
        pred_char_labels=pred_char_labels,
        true_char_labels=true_char_labels,
        id2canon=id2canon,
        colors_arg=colors_arg,
    )
    effective_colors = window_colors or id2color
    pred_html, stats = _build_segments_html(
        text=text,
        char_labels=pred_char_labels,
        char_probs=char_probs,
        true_labels=true_char_labels,
        highlight_mask=highlight,
        id2name=id2name,
        id2canon=id2canon,
        id2color=effective_colors,
    )
    truth_html, _ = _build_segments_html(
        text=text,
        char_labels=true_char_labels,
        char_probs=[{} for _ in true_char_labels],
        true_labels=true_char_labels,
        highlight_mask=[False] * len(true_char_labels),
        id2name=id2name,
        id2canon=id2canon,
        id2color=effective_colors,
    )
    palette = [
        {
            "id": label_id,
            "name": id2name.get(label_id, str(label_id)),
            "color": effective_colors.get(label_id, "#888888"),
        }
        for label_id in palette_ids
    ]
    return {
        "html": pred_html,
        "truth_html": truth_html,
        "stats": stats,
        "palette": palette,
        "text_preview": text[:240],
        "meta": sample.meta or {},
        "char_count": len(text),
        "highlighted_chars": int(sum(1 for flag in highlight if flag)),
    }


def _format_file_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _monitor_row_meta(row: Any, file_idx: int) -> Dict[str, Any]:
    source_id = int(row.get("source", 0)) if isinstance(row, dict) else int(row["source"])
    type_id = int(row.get("type_id", 0)) if isinstance(row, dict) else int(row["type_id"])
    return {
        "file_index": int(file_idx),
        "file_bytes": int(row.get("byte_len", 0)) if isinstance(row, dict) else int(row["byte_len"]),
        "source": SOURCE_NAMES.get(source_id, str(source_id)),
        "declared": ID2CANONICAL.get(type_id, str(type_id)),
    }


def _build_monitor_true_labels(byte_len: int, seg_slice: Iterable[Any]) -> np.ndarray:
    labels = np.full(int(byte_len), cfg.PAD_ID, dtype=np.int32)
    for seg in seg_slice:
        start = int(seg["start"])
        end = int(seg["end"])
        label = int(seg["label"])
        start = max(0, min(byte_len, start))
        end = max(start, min(byte_len, end))
        if end <= start:
            continue
        labels[start:end] = label
    return labels


def _maybe_collect_monitor_example(
    *,
    file_idx: int,
    run_start: int,
    run_end: int,
    true_id: int,
    pred_id: int,
    examples_per_cell: int,
    byte_length: int,
    window: int,
    row_meta: Dict[str, Any],
    pointers: List[MonitorExamplePointer],
    cell_examples: Dict[Tuple[int, int], List[int]],
):
    if byte_length <= 0 or true_id < 0 or pred_id < 0:
        return
    bucket = cell_examples.setdefault((true_id, pred_id), [])
    if len(bucket) >= examples_per_cell:
        return
    center_idx = (int(run_start) + int(run_end)) // 2
    half_window = max(1, int(window)) // 2
    start = max(0, center_idx - half_window)
    end = min(byte_length, start + max(1, int(window)))
    if end - start < max(1, int(window)):
        start = max(0, end - max(1, int(window)))
    pointer = MonitorExamplePointer(
        file_idx=int(file_idx),
        byte_start=int(start),
        byte_end=int(end),
        meta=dict(row_meta),
    )
    pointer_id = len(pointers)
    pointers.append(pointer)
    bucket.append(pointer_id)


def _collect_monitor_confusion(
    *,
    predictor: Predictor,
    monitor_data: Dict[str, Any],
    chunk: int,
    examples_per_cell: int,
    num_classes: int,
    other_threshold: float,
    monitor_limit: int = 0,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[int]], List[MonitorExamplePointer], Dict[str, Any]]:
    files = monitor_data["files"]
    segments = monitor_data["segments"]
    contents = monitor_data["contents"]
    total_available = len(files)
    limit = total_available if monitor_limit <= 0 else min(total_available, int(monitor_limit))
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    cell_examples: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
    pointers: List[MonitorExamplePointer] = []
    processed = 0
    total_bytes = 0
    source_counts: collections.Counter = collections.Counter()
    # Treat the last index as a derived 'other' bucket when thresholding.
    other_id = num_classes - 1
    use_threshold = other_threshold is not None and other_threshold > 0.0
    window = max(1, int(chunk))
    for file_idx in range(limit):
        row = files[file_idx]
        byte_len = int(row["byte_len"])
        if byte_len <= 0:
            continue
        byte_start = int(row["byte_start"])
        raw = np.asarray(contents[byte_start : byte_start + byte_len], dtype=np.uint8)
        sanitized = _sanitize_model_bytes(raw)
        seg_start = int(row["seg_start"])
        seg_count = int(row["seg_count"])
        seg_slice = segments[seg_start : seg_start + seg_count]
        true_labels = _build_monitor_true_labels(len(sanitized), seg_slice)
        pred_bytes, probs = predictor._segment_bytes(sanitized)
        preds_core = pred_bytes.astype(np.int32)
        preds_thresh = preds_core.copy()
        if use_threshold:
            max_prob_full = np.max(probs, axis=-1)
            low_conf_mask = max_prob_full < float(other_threshold)
            preds_thresh[low_conf_mask] = int(other_id)
        valid_mask = (true_labels != cfg.PAD_ID)
        if not valid_mask.any():
            continue
        ignored_mask = np.isin(sanitized, IGNORED_WHITESPACE_BYTE_VALUES)
        effective_mask = valid_mask & (~ignored_mask)
        if not effective_mask.any():
            continue
        y_true = true_labels[effective_mask].astype(np.int32)
        y_pred = preds_thresh[effective_mask]
        core_mask = (
            (y_true >= 0)
            & (y_true < num_classes)
            & (y_pred >= 0)
            & (y_pred < num_classes)
        )
        if not core_mask.any():
            continue
        y_true_core = y_true[core_mask]
        y_pred_core = y_pred[core_mask]
        accumulate_confusion(conf_mat, y_true_core, y_pred_core)
        # Use the same thresholded labels when bucketing examples.
        indices = np.nonzero(effective_mask)[0]
        if indices.size > 0:
            row_meta = _monitor_row_meta(row, file_idx)
            run_start = int(indices[0])
            run_end = run_start
            run_true = int(true_labels[run_start])
            run_pred = int(preds_thresh[run_start])
            for pos in indices[1:]:
                pos = int(pos)
                t_val = int(true_labels[pos])
                p_val = int(preds_thresh[pos])
                if t_val == run_true and p_val == run_pred:
                    run_end = pos
                    continue
                _maybe_collect_monitor_example(
                    file_idx=file_idx,
                    run_start=run_start,
                    run_end=run_end,
                    true_id=run_true,
                    pred_id=run_pred,
                    examples_per_cell=examples_per_cell,
                    byte_length=len(sanitized),
                    window=window,
                    row_meta=row_meta,
                    pointers=pointers,
                    cell_examples=cell_examples,
                )
                run_start = pos
                run_end = pos
                run_true = t_val
                run_pred = p_val
            _maybe_collect_monitor_example(
                file_idx=file_idx,
                run_start=run_start,
                run_end=run_end,
                true_id=run_true,
                pred_id=run_pred,
                examples_per_cell=examples_per_cell,
                byte_length=len(sanitized),
                window=window,
                row_meta=row_meta,
                pointers=pointers,
                cell_examples=cell_examples,
            )
        processed += 1
        total_bytes += len(sanitized)
        source_counts[int(row["source"])] += 1
        if processed % 250 == 0 or processed == limit:
            print(f"   ✔ Processed {processed}/{limit} monitor files", flush=True)
    meta = {
        "files": processed,
        "bytes": total_bytes,
        "sources": {SOURCE_NAMES.get(k, str(k)): int(v) for k, v in source_counts.items()},
        "limit": limit,
        "available": total_available,
    }
    return conf_mat, cell_examples, pointers, meta


def _ensure_monitor_prediction(
    file_idx: int,
    monitor_data: Dict[str, Any],
    predictor: Predictor,
) -> Dict[str, Any]:
    global _MONITOR_PRED_CACHE
    cache = _MONITOR_PRED_CACHE
    if cache.get("file_idx") == file_idx:
        return cache
    files = monitor_data["files"]
    if file_idx < 0 or file_idx >= len(files):
        raise IndexError(f"Monitor file index {file_idx} out of range")
    row = files[file_idx]
    byte_len = int(row["byte_len"])
    if byte_len <= 0:
        raise ValueError("Monitor file has no bytes to render.")
    byte_start = int(row["byte_start"])
    contents = monitor_data["contents"]
    raw = np.asarray(contents[byte_start : byte_start + byte_len], dtype=np.uint8)
    sanitized = _sanitize_model_bytes(raw)
    seg_start = int(row["seg_start"])
    seg_count = int(row["seg_count"])
    seg_slice = monitor_data["segments"][seg_start : seg_start + seg_count]
    true_labels = _build_monitor_true_labels(len(sanitized), seg_slice)
    pred_bytes, probs = predictor._segment_bytes(sanitized)
    cache = {
        "file_idx": file_idx,
        "tokens": sanitized,
        "true": true_labels,
        "pred": pred_bytes.astype(np.int32),
        "probs": np.asarray(probs, dtype=np.float32),
        "row_meta": _monitor_row_meta(row, file_idx),
    }
    _MONITOR_PRED_CACHE = cache
    return cache


def _build_monitor_sample_payload(
    pointer: MonitorExamplePointer,
    true_id: int,
    pred_id: int,
    predictor: Predictor,
    monitor_data: Dict[str, Any],
    id2name: Dict[int, str],
    id2canon: Dict[int, str],
    id2color: Dict[int, str],
    colors_arg: Optional[str],
) -> Dict[str, Any]:
    cache = _ensure_monitor_prediction(pointer.file_idx, monitor_data, predictor)
    tokens = cache["tokens"]
    truth = cache["true"]
    preds = cache["pred"]
    probs = cache["probs"]
    start = max(0, min(len(tokens), int(pointer.byte_start)))
    end = max(start, min(len(tokens), int(pointer.byte_end)))
    if end <= start:
        end = min(len(tokens), start + predictor.chunk)
    snippet_tokens = tokens[start:end]
    snippet_truth = truth[start:end]
    snippet_preds = preds[start:end]
    snippet_probs = probs[start:end]
    text = _normalize_input_text(_tokens_to_text(snippet_tokens))
    pred_char_labels, char_probs = predictor._byte_labels_to_char_labels(text, snippet_preds, snippet_probs)
    true_char_labels, _ = predictor._byte_labels_to_char_labels(text, snippet_truth, None)
    # Apply the same low-confidence 'other' behavior as the main viewer.
    pred_char_labels, char_probs = predictor._apply_other_threshold(pred_char_labels, char_probs)
    highlight: List[bool] = []
    for idx, pred_label in enumerate(pred_char_labels):
        true_label = true_char_labels[idx] if idx < len(true_char_labels) else None
        ch = text[idx] if idx < len(text) else ""
        highlight.append(
            ch not in IGNORED_WHITESPACE_CHAR_SET and true_label == true_id and pred_label == pred_id
        )
    palette_ids, window_colors = _build_window_color_map(
        pred_char_labels=pred_char_labels,
        true_char_labels=true_char_labels,
        id2canon=id2canon,
        colors_arg=colors_arg,
    )
    effective_colors = window_colors or id2color
    pred_html, stats = _build_segments_html(
        text=text,
        char_labels=pred_char_labels,
        char_probs=char_probs,
        true_labels=true_char_labels,
        highlight_mask=highlight,
        id2name=id2name,
        id2canon=id2canon,
        id2color=effective_colors,
    )
    truth_html, _ = _build_segments_html(
        text=text,
        char_labels=true_char_labels,
        char_probs=[{} for _ in true_char_labels],
        true_labels=true_char_labels,
        highlight_mask=[False] * len(true_char_labels),
        id2name=id2name,
        id2canon=id2canon,
        id2color=effective_colors,
    )
    palette = [
        {
            "id": label_id,
            "name": id2name.get(label_id, str(label_id)),
            "color": effective_colors.get(label_id, "#888888"),
        }
        for label_id in palette_ids
    ]
    merged_meta = dict(cache.get("row_meta", {}))
    merged_meta.update(pointer.meta or {})
    merged_meta["slice"] = f"{start}-{end}"
    return {
        "html": pred_html,
        "truth_html": truth_html,
        "stats": stats,
        "palette": palette,
        "text_preview": text[:240],
        "meta": merged_meta,
        "char_count": len(text),
        "highlighted_chars": int(sum(1 for flag in highlight if flag)),
    }


def _dataset_entries_payload(active: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for key, state in DATASET_STATES.items():
        entries.append(
            {
                "id": key,
                "name": DATASET_LABELS.get(key, key.title()),
                "summary": state.get("summary"),
                "available": state.get("matrix") is not None,
                "active": key == active,
            }
        )
    entries.sort(key=lambda item: (item["id"] != "val", item["id"]))
    return entries


def _get_dataset_state(dataset: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    ds_id = (dataset or "").strip() or DEFAULT_DATASET
    if ds_id not in DATASET_STATES:
        raise KeyError(ds_id)
    state = DATASET_STATES[ds_id]
    if state.get("matrix") is None:
        raise KeyError(ds_id)
    return ds_id, state


# ---------------------------
# FastAPI wiring + CLI
# ---------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    required=True,
    help="Path to a .msgpack file OR an Orbax checkpoint dir/root",
)
parser.add_argument(
    "--arch",
    type=str,
    default=None,
    choices=("unet1d", "mamba"),
    help="Model architecture (auto if omitted).",
)
parser.add_argument("--model-dim", type=int, default=None, help="Model embedding dimension (auto if omitted).")
parser.add_argument("--channels", type=str, default=None, help="Comma-separated channel sizes (auto if omitted).")
parser.add_argument("--dtype", type=str, default=None, help="Model dtype name (auto if omitted).")
# Mamba-only knobs (ignored for unet1d); auto if omitted.
parser.add_argument("--mamba-layers", type=int, default=None)
parser.add_argument("--mamba-d-state", type=int, default=None)
parser.add_argument("--mamba-expand", type=int, default=None)
parser.add_argument("--mamba-dt-rank", type=int, default=None)
parser.add_argument("--mamba-conv", type=int, default=None)
parser.add_argument("--mamba-bidirectional", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument(
    "--chunk",
    type=int,
    default=DEFAULT_CHUNK_SIZE,
    help="Inference window size (must match training).",
)
parser.add_argument(
    "--lang",
    type=str,
    default=None,
    help="Optional comma-separated subset of languages (with optional display overrides via =).",
)
parser.add_argument("--colors", type=str, default=None, help="Optional colors; positional or lang=#hex mapping.")
parser.add_argument("--data-root", type=str, default=None, help="Path to Arrow data root (defaults to downloader/arrow_out).")
parser.add_argument("--batch-size", type=int, default=16, help="Batch size for confusion sampling.")
parser.add_argument(
    "--eval-batches",
    type=int,
    default=25,
    help="Base number of batches to sample (internally multiplied by 10).",
)
parser.add_argument(
    "--examples-per-cell",
    type=int,
    default=MAX_EXAMPLES_PER_CELL,
    help="Maximum cached examples per confusion cell.",
)
parser.add_argument("--seed", type=int, default=42, help="Seed controlling sampling order.")
parser.add_argument(
    "--use-train-windows",
    action="store_true",
    default=False,
    help="Use augmented training windows instead of raw language directories.",
)
parser.add_argument(
    "--monitor-root",
    type=str,
    default=str(REPO_ROOT / "downloader" / "monitor_preprocessed"),
    help="Path to downloader/monitor_preprocessed (set empty to disable monitor dataset).",
)
parser.add_argument(
    "--monitor-limit",
    type=int,
    default=0,
    help="Optional cap on monitor files to scan (0 = use all available).",
)
parser.add_argument(
    "--other-threshold",
    type=float,
    default=0.2,
    help=(
        "If >0, route low-confidence predictions (max softmax below this) "
        "into a virtual 'other' class in both the confusion matrix and sample views."
    ),
)
parser.add_argument(
    "--fine-tuned",
    action="store_true",
    default=False,
    help=(
        "When set, use downloader/monitor_preprocessed_b as the monitor memmap "
        "source (matching fine-tuned runs)."
    ),
)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--openapi", action="store_true")
args, _ = parser.parse_known_args()

# For fine-tuned runs, default the monitor memmap to the 'B' split that
# serves as the held-out monitor validation set, unless the user has
# explicitly overridden --monitor-root.
default_monitor_root = str(REPO_ROOT / "downloader" / "monitor_preprocessed")
monitor_b_root = str(REPO_ROOT / "downloader" / "monitor_preprocessed_b")
if getattr(args, "fine_tuned", False):
    if not args.monitor_root or args.monitor_root.strip() == default_monitor_root:
        args.monitor_root = monitor_b_root

ckpt_path = Path(args.ckpt).resolve()
auto_hparams = _load_checkpoint_hparams(ckpt_path)
ckpt_inferred = _infer_checkpoint_architecture(ckpt_path)

resolved_data_root = _resolve_data_root(args.data_root)
if not resolved_data_root.exists():
    raise SystemExit(
        f"❌ DATA ROOT NOT FOUND: {resolved_data_root}\n"
        f"Set --data-root to the directory containing train/val/test Arrow shards."
    )
args.data_root = str(resolved_data_root)
effective_eval_batches = max(1, int(args.eval_batches) * 10)

label_names = auto_hparams.get("label_names")
if label_names:
    label_names = _sanitize_label_order(label_names)
if label_names:
    _apply_label_mapping(label_names)
    TRAINING_LABEL_ORDER = list(label_names)

arch_value, arch_source = _resolve_hparam(
    args.arch,
    auto_hparams.get("arch"),
    ckpt_inferred.get("arch"),
    "unet1d",
)
arch = str(arch_value).lower().strip()
args.arch = arch

model_dim_value, model_dim_source = _resolve_hparam(
    args.model_dim,
    auto_hparams.get("model_dim"),
    ckpt_inferred.get("model_dim"),
    256,
)
model_dim = int(model_dim_value)

dtype_value, dtype_source = _resolve_hparam(
    args.dtype,
    auto_hparams.get("dtype"),
    ckpt_inferred.get("dtype"),
    "bfloat16",
)
dtype = dtype_value.rsplit(".", 1)[-1] if isinstance(dtype_value, str) else str(dtype_value)

channels: Tuple[int, ...]
channels_source = "unused"
if arch == "unet1d":
    channels_cli = None
    if args.channels:
        channels_cli = [int(x) for x in args.channels.split(",") if x.strip()]
    channels_value, channels_source = _resolve_hparam(
        channels_cli,
        auto_hparams.get("channels"),
        ckpt_inferred.get("channels"),
        list(DEFAULT_CHANNELS),
    )
    channels = tuple(int(x) for x in channels_value)
else:
    channels = tuple(DEFAULT_CHANNELS)

args.model_dim = model_dim
args.dtype = dtype
args.channels = ",".join(str(ch) for ch in channels)

if arch == "mamba":
    args.mamba_layers, _ = _resolve_hparam(
        getattr(args, "mamba_layers", None),
        auto_hparams.get("mamba_layers"),
        ckpt_inferred.get("mamba_layers"),
        6,
    )
    args.mamba_d_state, _ = _resolve_hparam(
        getattr(args, "mamba_d_state", None),
        auto_hparams.get("mamba_d_state"),
        ckpt_inferred.get("mamba_d_state"),
        8,
    )
    args.mamba_expand, _ = _resolve_hparam(
        getattr(args, "mamba_expand", None),
        auto_hparams.get("mamba_expand"),
        ckpt_inferred.get("mamba_expand"),
        1,
    )
    args.mamba_dt_rank, _ = _resolve_hparam(
        getattr(args, "mamba_dt_rank", None),
        auto_hparams.get("mamba_dt_rank"),
        ckpt_inferred.get("mamba_dt_rank"),
        16,
    )
    args.mamba_conv, _ = _resolve_hparam(
        getattr(args, "mamba_conv", None),
        auto_hparams.get("mamba_conv"),
        ckpt_inferred.get("mamba_conv"),
        4,
    )
    args.mamba_bidirectional, _ = _resolve_hparam(
        getattr(args, "mamba_bidirectional", None),
        auto_hparams.get("mamba_bidirectional"),
        ckpt_inferred.get("mamba_bidirectional"),
        True,
    )

if auto_hparams or ckpt_inferred:
    details = [
        f"arch={arch} ({arch_source})",
        f"model_dim={model_dim} ({model_dim_source})",
        f"dtype={dtype} ({dtype_source})",
    ]
    if arch == "unet1d":
        details.insert(2, f"channels={list(channels)} ({channels_source})")
    else:
        details.append(
            "mamba="
            + f"layers={int(args.mamba_layers)},d_state={int(args.mamba_d_state)},expand={int(args.mamba_expand)},"
            + f"dt_rank={int(args.mamba_dt_rank)},conv={int(args.mamba_conv)},bi={bool(args.mamba_bidirectional)}"
        )
    if label_names:
        details.append(f"classes={len(label_names)}")
    elif ckpt_inferred.get("num_classes"):
        details.append(f"classes={ckpt_inferred['num_classes']} (checkpoint)")
    print(f"ℹ️  Resolved hyperparameters: {', '.join(details)}", flush=True)

try:
    canonical_label_names, display_label_names = _resolve_langs_and_display(args.lang)
    display_overrides = {
        canonical: display
        for canonical, display in zip(canonical_label_names, display_label_names)
        if display != canonical
    }
    if not args.lang:
        canonical_label_names = list(TRAINING_LABEL_ORDER or canonical_label_names)
        display_label_names = [display_overrides.get(name, name) for name in canonical_label_names]
    colors = _resolve_colors(canonical_label_names, args.colors)
except (RuntimeError, ValueError) as exc:
    parser.error(str(exc))

_sync_config_with_labels(canonical_label_names)

num_classes = len(canonical_label_names)
ckpt_classes = ckpt_inferred.get("num_classes")
if ckpt_classes is not None and ckpt_classes != num_classes:
    print(
        f"⚠️  Checkpoint expects {ckpt_classes} classes but resolved {num_classes}.",
        flush=True,
    )

ID2CANONICAL = {i: canonical_label_names[i] for i in range(num_classes)}
ID2NAME = {i: display_label_names[i] for i in range(num_classes)}
ID2COLOR = {i: colors[i] for i in range(num_classes)}
ID2SLUG = {i: make_slug(ID2CANONICAL[i]) for i in range(num_classes)}

# Virtual 'other' bucket for low-confidence characters in sample views.
OTHER_LABEL_ID = num_classes
ID2CANONICAL[OTHER_LABEL_ID] = "other"
ID2NAME[OTHER_LABEL_ID] = "other"
ID2COLOR.setdefault(OTHER_LABEL_ID, "#7f8c8d")
ID2SLUG[OTHER_LABEL_ID] = make_slug("other")

DATASET_LABELS = {
    "val": "Validation",
    "monitor": "Monitor",
}

DATASET_STATES: Dict[str, Dict[str, Any]] = {}
DEFAULT_DATASET: str = "val"
MONITOR_DATA: Optional[Dict[str, Any]] = None
_MONITOR_PRED_CACHE: Dict[str, Any] = {}

load_error: Optional[str] = None
predictor: Optional[Predictor] = None
try:
    predictor = Predictor(
        ckpt_path=args.ckpt,
        num_classes=num_classes,
        model_dim=args.model_dim,
        channels=channels,
        arch=args.arch,
        mamba_layers=int(getattr(args, "mamba_layers", 6) or 6),
        mamba_d_state=int(getattr(args, "mamba_d_state", 8) or 8),
        mamba_expand=int(getattr(args, "mamba_expand", 1) or 1),
        mamba_dt_rank=int(getattr(args, "mamba_dt_rank", 16) or 16),
        mamba_conv=int(getattr(args, "mamba_conv", 4) or 4),
        mamba_bidirectional=bool(getattr(args, "mamba_bidirectional", True)),
        dtype_str=args.dtype,
        chunk=args.chunk,
        other_threshold=args.other_threshold,
    )
except Exception as exc:
    load_error = str(exc)
    print(f"❌ Failed to load checkpoint: {exc}", flush=True)

if load_error is None and predictor is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_cfg = _make_data_config(args.chunk, args.batch_size, args.data_root)
    include_langs = canonical_label_names
    splits = prepare_dsets_by_lang_with_splits(
        data_root=args.data_root,
        use_train_windows=args.use_train_windows,
        include_languages=include_langs,
        preserve_lang_order=True,
        preferred_order=canonical_label_names,
    )
    source_lang2id = dict(cfg.LANG2ID)
    cfg.update_lang_mappings()
    canonical_label_names = [name for _, name in sorted(cfg.ID2LANG.items(), key=lambda kv: kv[0])]
    display_label_names = [display_overrides.get(name, name) for name in canonical_label_names]
    colors = _resolve_colors(canonical_label_names, args.colors)
    _sync_config_with_labels(canonical_label_names)
    num_classes = len(canonical_label_names)
    ckpt_classes = ckpt_inferred.get("num_classes")
    if ckpt_classes is not None and ckpt_classes != num_classes:
        print(
            f"⚠️  Checkpoint expects {ckpt_classes} classes but resolved {num_classes}.",
            flush=True,
        )
    ID2CANONICAL = {i: canonical_label_names[i] for i in range(num_classes)}
    ID2NAME = {i: display_label_names[i] for i in range(num_classes)}
    ID2COLOR = {i: colors[i] for i in range(num_classes)}
    ID2SLUG = {i: make_slug(ID2CANONICAL[i]) for i in range(num_classes)}
    # Ensure the virtual 'other' bucket stays in sync after label remapping.
    OTHER_LABEL_ID = num_classes
    ID2CANONICAL[OTHER_LABEL_ID] = "other"
    ID2NAME[OTHER_LABEL_ID] = "other"
    ID2COLOR.setdefault(OTHER_LABEL_ID, "#7f8c8d")
    ID2SLUG[OTHER_LABEL_ID] = make_slug("other")
    # Confusion/metrics operate over the explicit classes plus a derived
    # 'other' bucket at the final index.
    eval_num_classes = OTHER_LABEL_ID + 1
    label_names_for_colors = list(canonical_label_names) + ["other"]
    print(f"✅ Final label order: {canonical_label_names}", flush=True)

    inv_source = {idx: name for name, idx in source_lang2id.items()}
    def _remap_split_dict(original: Dict[int, Any]) -> Dict[int, Any]:
        remapped: Dict[int, Any] = {}
        for old_id, ds in original.items():
            lang_name = inv_source.get(old_id)
            if lang_name is None:
                continue
            new_id = cfg.LANG2ID.get(lang_name)
            if new_id is None:
                continue
            remapped[new_id] = ds
        return remapped
    splits = {split_name: _remap_split_dict(mapping) for split_name, mapping in splits.items()}

    val_dsets = splits["val"]
    if not val_dsets:
        load_error = "Validation datasets not found for the requested languages."
        print(f"❌ {load_error}", flush=True)
    else:
        max_examples = max(1, int(args.examples_per_cell or MAX_EXAMPLES_PER_CELL))
        conf_mat, cell_examples, samples = _collect_confusion_samples(
            predictor=predictor,
            val_dsets=val_dsets,
            chunk=args.chunk,
            batch_size=args.batch_size,
            batches=effective_eval_batches,
            data_cfg=data_cfg,
            examples_per_cell=max_examples,
            num_classes=eval_num_classes,
            other_threshold=float(args.other_threshold or 0.0),
        )
        per_class, aggregates = compute_metrics_from_confusion(
            conf_mat, eval_num_classes, cfg.PAD_ID
        )
        DATASET_STATES["val"] = {
            "matrix": conf_mat,
            "row_norm": _row_normalize(conf_mat),
            "per_class": per_class,
            "aggregates": aggregates,
            "cell_examples": cell_examples,
            "samples": samples,
            "row_totals": conf_mat.sum(axis=1),
            "total_windows": len(samples),
            "cell_counts": {
                (t, p): int(conf_mat[t, p])
                for t in range(conf_mat.shape[0])
                for p in range(conf_mat.shape[1])
            },
            "colors": _remap_colors_for_active_classes(
                conf_mat, label_names_for_colors, args.colors
            ),
            "summary": f"{len(samples)} windows",
            "sample_type": "val",
            "meta": {
                "batch_size": args.batch_size,
                "eval_batches": effective_eval_batches,
            },
        }
        monitor_root_arg = (args.monitor_root or "").strip()
        if monitor_root_arg:
            monitor_root = Path(monitor_root_arg).expanduser()
            if monitor_root.exists():
                try:
                    print(
                        f"📈 Building monitor confusion from {monitor_root} (limit={args.monitor_limit or 0})",
                        flush=True,
                    )
                    MONITOR_DATA = load_monitor_memmaps(monitor_root)
                    _MONITOR_PRED_CACHE.clear()
                    monitor_conf, monitor_examples, monitor_pointers, monitor_meta = _collect_monitor_confusion(
                        predictor=predictor,
                        monitor_data=MONITOR_DATA,
                        chunk=args.chunk,
                        examples_per_cell=max_examples,
                        num_classes=eval_num_classes,
                        other_threshold=float(args.other_threshold or 0.0),
                        monitor_limit=int(args.monitor_limit or 0),
                    )
                    per_class_m, aggregates_m = compute_metrics_from_confusion(
                        monitor_conf, eval_num_classes, cfg.PAD_ID
                    )
                    summary_label = (
                        f"{monitor_meta['files']} files ({_format_file_size(monitor_meta['bytes'])})"
                        if monitor_meta.get("files")
                        else "0 files"
                    )
                    monitor_meta.update({"root": str(monitor_root)})
                    DATASET_STATES["monitor"] = {
                        "matrix": monitor_conf,
                        "row_norm": _row_normalize(monitor_conf),
                        "per_class": per_class_m,
                        "aggregates": aggregates_m,
                        "cell_examples": monitor_examples,
                        "samples": monitor_pointers,
                        "row_totals": monitor_conf.sum(axis=1),
                        "total_windows": int(monitor_meta.get("files", 0)),
                        "cell_counts": {
                            (t, p): int(monitor_conf[t, p])
                            for t in range(monitor_conf.shape[0])
                            for p in range(monitor_conf.shape[1])
                        },
                        "colors": _remap_colors_for_active_classes(
                            monitor_conf, label_names_for_colors, args.colors
                        ),
                        "summary": summary_label,
                        "sample_type": "monitor",
                        "meta": monitor_meta,
                    }
                    print(
                        f"✅ Monitor confusion built over {monitor_meta['files']} files (examples cached for {len(monitor_pointers)} snippets)",
                        flush=True,
                    )
                except Exception as exc:
                    MONITOR_DATA = None
                    _MONITOR_PRED_CACHE.clear()
                    print(f"⚠️  Failed to build monitor confusion: {exc}", flush=True)
            else:
                print(f"⚠️  Monitor root not found at {monitor_root}, skipping monitor dataset.", flush=True)
        else:
            print("ℹ️  Monitor dataset disabled (--monitor-root empty).", flush=True)

if "val" in DATASET_STATES:
    DEFAULT_DATASET = "val"
elif DATASET_STATES:
    DEFAULT_DATASET = next(iter(DATASET_STATES.keys()))

app = FastAPI(title="Segmenter Confusion Viewer", docs_url="/docs" if args.openapi else None)

static_path = os.path.join(os.path.dirname(__file__), "confusion_static")
if os.path.isdir(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    if not os.path.isdir(static_path):
        return HTMLResponse("<h1>UI not found</h1>", status_code=500)
    return FileResponse(os.path.join(static_path, "index.html"))


@app.get("/api/confusion")
def api_confusion(dataset: Optional[str] = None):
    if load_error is not None:
        raise HTTPException(status_code=500, detail=load_error)
    if not DATASET_STATES:
        raise HTTPException(status_code=503, detail="Confusion matrix not ready.")
    try:
        dataset_id, state = _get_dataset_state(dataset)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' is unavailable.")
    conf_mat = state["matrix"]
    row_norm = state["row_norm"]
    per_class = state["per_class"]
    aggregates = state["aggregates"]
    cell_examples = state["cell_examples"]
    color_map = state.get("colors") or ID2COLOR
    label_ids = sorted(ID2NAME.keys())
    payload = {
        "dataset": dataset_id,
        "datasets": _dataset_entries_payload(dataset_id),
        "labels": [
            {"id": i, "name": ID2NAME[i], "slug": ID2SLUG[i], "color": color_map.get(i)}
            for i in label_ids
        ],
        "matrix": conf_mat.astype(int).tolist(),
        "row_normalized": row_norm.tolist(),
        "per_class": _format_metrics(per_class),
        "aggregates": {
            key: {metric: float(value) for metric, value in metrics.items()}
            for key, metrics in aggregates.items()
        },
        "cell_examples": {
            f"{true}_{pred}": len(ids) for (true, pred), ids in cell_examples.items()
        },
        "row_totals": state["row_totals"].astype(int).tolist(),
        "total_windows": state.get("total_windows", 0),
        "summary": state.get("summary"),
        "meta": state.get("meta", {}),
        "device": str(jax.devices()),
    }
    return payload


@app.get("/api/example")
def api_example(true_id: int, pred_id: int, dataset: Optional[str] = None):
    if load_error is not None or predictor is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {load_error}")
    if not DATASET_STATES:
        raise HTTPException(status_code=503, detail="Confusion matrix not ready.")
    try:
        dataset_id, state = _get_dataset_state(dataset)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' is unavailable.")
    cell_examples = state["cell_examples"]
    samples = state["samples"]
    conf_mat = state["matrix"]
    row_totals = state["row_totals"]
    row_norm = state["row_norm"]
    key = (int(true_id), int(pred_id))
    ids = cell_examples.get(key)
    if not ids:
        raise HTTPException(status_code=404, detail="No cached examples for this cell.")
    sample_id = random.choice(ids)
    sample = samples[sample_id]
    color_map = state.get("colors") or ID2COLOR
    if state.get("sample_type") == "monitor":
        if MONITOR_DATA is None:
            raise HTTPException(status_code=503, detail="Monitor dataset not loaded.")
        payload = _build_monitor_sample_payload(
            pointer=sample,
            true_id=true_id,
            pred_id=pred_id,
            predictor=predictor,
            monitor_data=MONITOR_DATA,
            id2name=ID2NAME,
            id2canon=ID2CANONICAL,
            id2color=color_map,
            colors_arg=args.colors,
        )
    else:
        payload = _build_sample_payload(
            sample=sample,
            true_id=true_id,
            pred_id=pred_id,
            predictor=predictor,
            id2name=ID2NAME,
            id2canon=ID2CANONICAL,
            id2color=color_map,
            colors_arg=args.colors,
        )
    payload.update(
        {
            "dataset": {"id": dataset_id, "name": DATASET_LABELS.get(dataset_id, dataset_id.title())},
            "cell": {
                "true": {"id": true_id, "name": ID2NAME.get(true_id, str(true_id))},
                "pred": {"id": pred_id, "name": ID2NAME.get(pred_id, str(pred_id))},
                "count": int(conf_mat[true_id, pred_id]),
                "row_total": int(row_totals[true_id]),
                "row_pct": float(row_norm[true_id, pred_id]) if row_totals[true_id] > 0 else 0.0,
                "pool": len(ids),
            },
        }
    )
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("confusion_viewer:app", host=args.host, port=args.port, reload=False)
