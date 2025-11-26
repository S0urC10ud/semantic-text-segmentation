#!/usr/bin/env python3
"""
Interactive confusion-matrix explorer that pairs each cell with real
validation windows rendered in the familiar segment viewer style.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import sys
import argparse
import bisect
import collections
import json
import random
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Ensure the repo root is importable as a package when invoked from within segment_viewer
if str(REPO_ROOT / "train") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "train"))

import train.config as cfg
from train.data_utils import prepare_dsets_by_lang_with_splits
from train.window_generator import make_training_window_with_metadata
from train.metrics_helper import compute_metrics_from_confusion, accumulate_confusion

try:
    from .core import (
        DEFAULT_CHANNELS,
        DEFAULT_CHUNK_SIZE,
        Predictor,
        _apply_label_mapping,
        _resolve_langs_and_display,
        _load_checkpoint_hparams,
        _infer_checkpoint_architecture,
        _resolve_hparam,
        _resolve_colors,
        _hex_to_rgba,
        _normalize_input_text,
        make_slug,
    )
except ImportError:  # pragma: no cover
    from core import (
        DEFAULT_CHANNELS,
        DEFAULT_CHUNK_SIZE,
        Predictor,
        _apply_label_mapping,
        _resolve_langs_and_display,
        _load_checkpoint_hparams,
        _infer_checkpoint_architecture,
        _resolve_hparam,
        _resolve_colors,
        _hex_to_rgba,
        _normalize_input_text,
        make_slug,
    )

MAX_EXAMPLES_PER_CELL = 64
DEFAULT_DATA_ROOT = (REPO_ROOT / "downloader" / "arrow_out").resolve()


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


IGNORED_WHITESPACE_CHARS: Tuple[str, ...] = (" ", "\t", "\n")
IGNORED_WHITESPACE_CHAR_SET = frozenset(IGNORED_WHITESPACE_CHARS)
IGNORED_WHITESPACE_BYTE_VALUES = np.array([ord(ch) for ch in IGNORED_WHITESPACE_CHARS], dtype=np.int32)


def _tokens_to_text(tokens: np.ndarray) -> str:
    arr = np.asarray(tokens, dtype=np.int32)
    if arr.size == 0:
        return ""
    mask = arr != cfg.PAD_BYTE_ID
    if not mask.any():
        return ""
    byte_arr = arr[mask].astype(np.uint8)
    try:
        return byte_arr.tobytes().decode("utf-8", "ignore")
    except Exception:
        return byte_arr.tobytes().decode("latin-1", "ignore")


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
        bg = _hex_to_rgba(color, 0.22)
        border = _hex_to_rgba(color, 0.35)
        chunk: List[str] = []
        for offset, ch in enumerate(raw):
            char_idx = start + offset
            raw_probs = char_probs[char_idx]
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
                f'<span class="{base_class}" data-probs="{payload}" '
                f'data-label="{display_label}" data-true="{_escape_html(true_lbl)}">'
                f"{_escape_html(ch)}</span>"
            )
        out_html.append(
            f'<span class="seg {cls}" data-label="{_escape_html(id2name.get(lbl, str(lbl)))}" '
            f'style="--seg-color:{color}; background-color:{bg}; box-shadow: inset 0 -1px 0 {border};">'
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
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[int]], List[SampleRecord]]:
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    cell_examples: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
    samples: List[SampleRecord] = []
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
        preds = np.argmax(logits, axis=-1).astype(np.int32)
        mask = (yb != cfg.PAD_ID)
        for i in range(batch_size):
            valid_mask = mask[i]
            if not valid_mask.any():
                continue
            ignored_mask = np.isin(xb[i], IGNORED_WHITESPACE_BYTE_VALUES)
            effective_mask = valid_mask & (~ignored_mask)
            if not effective_mask.any():
                continue
            true_flat = yb[i][effective_mask]
            pred_flat = preds[i][effective_mask]
            accumulate_confusion(conf_mat, true_flat, pred_flat)
            combos = np.stack([true_flat, pred_flat], axis=1)
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
parser.add_argument("--model-dim", type=int, default=None, help="Model embedding dimension (auto if omitted).")
parser.add_argument("--channels", type=str, default=None, help="Comma-separated channel sizes (auto if omitted).")
parser.add_argument("--dtype", type=str, default=None, help="Model dtype name (auto if omitted).")
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
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--openapi", action="store_true")
args, _ = parser.parse_known_args()

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
    _apply_label_mapping(label_names)

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

args.model_dim = model_dim
args.dtype = dtype
args.channels = ",".join(str(ch) for ch in channels)

if auto_hparams or ckpt_inferred:
    details = [
        f"model_dim={model_dim} ({model_dim_source})",
        f"channels={list(channels)} ({channels_source})",
        f"dtype={dtype} ({dtype_source})",
    ]
    if label_names:
        details.append(f"classes={len(label_names)}")
    elif ckpt_inferred.get("num_classes"):
        details.append(f"classes={ckpt_inferred['num_classes']} (checkpoint)")
    print(f"ℹ️  Resolved hyperparameters: {', '.join(details)}", flush=True)

try:
    canonical_label_names, display_label_names = _resolve_langs_and_display(args.lang)
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

load_error: Optional[str] = None
predictor: Optional[Predictor] = None
try:
    predictor = Predictor(
        ckpt_path=args.ckpt,
        num_classes=num_classes,
        model_dim=args.model_dim,
        channels=channels,
        dtype_str=args.dtype,
        chunk=args.chunk,
    )
except Exception as exc:
    load_error = str(exc)
    print(f"❌ Failed to load checkpoint: {exc}", flush=True)

CONFUSION_STATE: Dict[str, Any] = {
    "matrix": None,
    "row_norm": None,
    "per_class": None,
    "aggregates": None,
    "cell_examples": {},
    "samples": [],
    "row_totals": None,
    "total_windows": 0,
    "cell_counts": {},
}

if load_error is None and predictor is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_cfg = _make_data_config(args.chunk, args.batch_size, args.data_root)
    include_langs = canonical_label_names
    splits = prepare_dsets_by_lang_with_splits(
        data_root=args.data_root,
        use_train_windows=args.use_train_windows,
        include_languages=include_langs,
    )
    val_dsets = splits["val"]
    if not val_dsets:
        load_error = "Validation datasets not found for the requested languages."
        print(f"❌ {load_error}", flush=True)
    else:
        conf_mat, cell_examples, samples = _collect_confusion_samples(
            predictor=predictor,
            val_dsets=val_dsets,
            chunk=args.chunk,
            batch_size=args.batch_size,
            batches=effective_eval_batches,
            data_cfg=data_cfg,
            examples_per_cell=max(1, int(args.examples_per_cell or MAX_EXAMPLES_PER_CELL)),
            num_classes=num_classes,
        )
        per_class, aggregates = compute_metrics_from_confusion(conf_mat, num_classes, cfg.PAD_ID)
        CONFUSION_STATE.update(
            {
                "matrix": conf_mat,
                "row_norm": _row_normalize(conf_mat),
                "per_class": per_class,
                "aggregates": aggregates,
                "cell_examples": cell_examples,
                "samples": samples,
                "row_totals": conf_mat.sum(axis=1),
                "total_windows": len(samples),
                "cell_counts": {(t, p): int(conf_mat[t, p]) for t in range(num_classes) for p in range(num_classes)},
            }
        )
        ID2COLOR = _remap_colors_for_active_classes(conf_mat, canonical_label_names, args.colors)

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
def api_confusion():
    if load_error is not None:
        raise HTTPException(status_code=500, detail=load_error)
    if CONFUSION_STATE["matrix"] is None:
        raise HTTPException(status_code=503, detail="Confusion matrix not ready.")
    conf_mat = CONFUSION_STATE["matrix"]
    row_norm = CONFUSION_STATE["row_norm"]
    per_class = CONFUSION_STATE["per_class"]
    aggregates = CONFUSION_STATE["aggregates"]
    cell_examples = CONFUSION_STATE["cell_examples"]
    payload = {
        "labels": [
            {"id": i, "name": ID2NAME[i], "slug": ID2SLUG[i], "color": ID2COLOR.get(i)}
            for i in range(num_classes)
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
        "row_totals": CONFUSION_STATE["row_totals"].astype(int).tolist(),
        "total_windows": CONFUSION_STATE["total_windows"],
        "device": str(jax.devices()),
    }
    return payload


@app.get("/api/example")
def api_example(true_id: int, pred_id: int):
    if load_error is not None or predictor is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {load_error}")
    cell_examples = CONFUSION_STATE["cell_examples"]
    samples = CONFUSION_STATE["samples"]
    conf_mat = CONFUSION_STATE["matrix"]
    row_totals = CONFUSION_STATE["row_totals"]
    row_norm = CONFUSION_STATE["row_norm"]
    key = (int(true_id), int(pred_id))
    ids = cell_examples.get(key)
    if not ids:
        raise HTTPException(status_code=404, detail="No cached examples for this cell.")
    sample_id = random.choice(ids)
    sample = samples[sample_id]
    payload = _build_sample_payload(
        sample=sample,
        true_id=true_id,
        pred_id=pred_id,
        predictor=predictor,
        id2name=ID2NAME,
        id2canon=ID2CANONICAL,
        id2color=ID2COLOR,
        colors_arg=args.colors,
    )
    payload.update(
        {
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
