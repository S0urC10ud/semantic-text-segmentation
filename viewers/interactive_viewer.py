#!/usr/bin/env python3
"""
Segmenter Viewer — FastAPI backend + beautiful frontend

Run:
  pip install fastapi uvicorn jax jaxlib flax optax numpy orbax-checkpoint
  # (Install the right jax/jaxlib for your CUDA setup if using GPU.)
  python app.py --ckpt ./seg-unet1d.msgpack --model-dim 256 --channels 96,128,192,256 --dtype bfloat16 --chunk 1536 --lang html,css,javascript_typescript,php

Then open http://127.0.0.1:8000
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from .core import (
        DEFAULT_CHANNELS,
        DEFAULT_CHUNK_SIZE,
        Predictor,
        auto_color,
        _apply_label_mapping,
        _hex_to_rgba,
        _hex_to_rgba_confidence,
        _infer_checkpoint_architecture,
        _load_checkpoint_hparams,
        _normalize_input_text,
        _resolve_colors,
        _resolve_hparam,
        _resolve_langs_and_display,
        make_slug,
    )
except ImportError:  # pragma: no cover
    from core import (
        DEFAULT_CHANNELS,
        DEFAULT_CHUNK_SIZE,
        Predictor,
        auto_color,
        _apply_label_mapping,
        _hex_to_rgba,
        _hex_to_rgba_confidence,
        _infer_checkpoint_architecture,
        _load_checkpoint_hparams,
        _normalize_input_text,
        _resolve_colors,
        _resolve_hparam,
        _resolve_langs_and_display,
        make_slug,
    )

# ---------------------------
# FastAPI wiring
# ---------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    required=True,
    help="Path to a .msgpack file OR an Orbax checkpoint dir/root"
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
    help="Inference window size (model expects padded segments of this length).",
)
parser.add_argument(
    "--lang",
    type=str,
    default=None,
    help=(
        "Comma-separated subset of languages. "
        "Use entries like 'php=PHP (Server)' to override display names."
    ),
)
parser.add_argument(
    "--colors",
    type=str,
    default=None,
    help="Optional colors; either positional list or 'lang=#hex' mappings.",
)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--openapi", action="store_true")
parser.add_argument(
    "--other-threshold",
    type=float,
    default=0.2,
    help="If >0, treat characters whose max softmax is below this as a virtual 'other' class.",
)
args, _ = parser.parse_known_args()

ckpt_path = Path(args.ckpt).resolve()
auto_hparams = _load_checkpoint_hparams(ckpt_path)
ckpt_inferred = _infer_checkpoint_architecture(ckpt_path)

label_names = auto_hparams.get("label_names")
if label_names:
    _apply_label_mapping(label_names)

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
if isinstance(dtype_value, str):
    dtype = dtype_value.rsplit(".", 1)[-1]
else:
    dtype = str(dtype_value)

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
    channel_values = [int(x) for x in channels_value]
    channels = tuple(channel_values)
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
    label_source = auto_hparams.pop("_label_source", None)
    if label_names:
        origin = label_source if label_source else "wandb"
        details.append(f"classes={len(label_names)} ({origin})")
    elif ckpt_inferred.get("num_classes"):
        details.append(f"classes={ckpt_inferred['num_classes']} (checkpoint)")
    print(f"ℹ️  Resolved hyperparameters: {', '.join(details)}", flush=True)

try:
    canonical_label_names, display_label_names = _resolve_langs_and_display(args.lang)
    cols = _resolve_colors(canonical_label_names, args.colors)
except (RuntimeError, ValueError) as exc:
    parser.error(str(exc))

num_classes = len(canonical_label_names)
ckpt_classes = ckpt_inferred.get("num_classes")
if ckpt_classes is not None and ckpt_classes != num_classes:
    print(
        f"⚠️  Checkpoint expects {ckpt_classes} classes but label config resolved {num_classes}.",
        flush=True,
    )
print(
    f"Label order resolved from training config: {canonical_label_names}",
    flush=True,
)

ID2CANONICAL = {i: canonical_label_names[i] for i in range(num_classes)}
ID2NAME = {i: display_label_names[i] for i in range(num_classes)}
ID2COLOR = {i: cols[i] for i in range(num_classes)}
ID2SLUG  = {i: make_slug(ID2CANONICAL[i]) for i in range(num_classes)}

# Virtual 'other' bucket used when the predictor applies a low-confidence
# threshold per character. Its index is one past the highest trained class.
OTHER_LABEL_ID = num_classes
ID2CANONICAL[OTHER_LABEL_ID] = "other"
ID2NAME[OTHER_LABEL_ID] = "other"
ID2COLOR.setdefault(OTHER_LABEL_ID, "#7f8c8d")
ID2SLUG[OTHER_LABEL_ID] = make_slug("other")

predictor = None
load_error = None
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
except Exception as e:
    load_error = str(e)

class SegmentRequest(BaseModel):
    text: str
    min_run: int = 4
    chunk: Optional[int] = None

app = FastAPI(title="Segmenter Viewer", docs_url="/docs" if args.openapi else None)

# Serve static UI
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    if not os.path.isdir(static_path):
        return HTMLResponse("<h1>UI not found</h1>", status_code=500)
    return FileResponse(os.path.join(static_path, "index.html"))

@app.get("/api/labels")
def get_labels():
    label_ids = sorted(ID2NAME.keys())
    return {
        "num_classes": len(label_ids),
        "labels": [
            {
                "id": i,
                "name": ID2NAME[i],
                "slug": ID2SLUG[i],
                "color": ID2COLOR.get(i),
            }
            for i in label_ids
            if ID2COLOR.get(i)
        ],
        "device": str(jax.devices()),
        "loaded": load_error is None,
        "error": load_error,
    }

@app.post("/api/segment")
def api_segment(req: SegmentRequest):
    if load_error is not None or predictor is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {load_error}")
    if not isinstance(req.text, str):
        return {"segments": [], "stats": {}, "html": ""}
    text = _normalize_input_text(req.text)
    if len(text) == 0:
        return {"segments": [], "stats": {}, "html": ""}

    if req.chunk and int(req.chunk) != predictor.chunk:
        raise HTTPException(
            status_code=400,
            detail=f"Chunk size override ({req.chunk}) does not match model window ({predictor.chunk}).",
        )

    import time as _time
    t0 = _time.perf_counter()
    try:
        segs, char_labels, char_probs, window_info = predictor.segment_text(
            text, min_run_chars=int(req.min_run)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = float((_time.perf_counter() - t0) * 1000.0)

    import collections
    counts = collections.Counter(char_labels)
    total = max(1, sum(counts.values()))
    used_ids = [int(lbl) for (lbl, _count) in counts.most_common()]
    used_non_other = [lbl for lbl in used_ids if lbl != OTHER_LABEL_ID]
    id2color_run: Dict[int, str] = {}
    for idx, lbl in enumerate(used_non_other):
        id2color_run[lbl] = auto_color(idx, len(used_non_other))
    if OTHER_LABEL_ID in counts:
        id2color_run[OTHER_LABEL_ID] = ID2COLOR.get(OTHER_LABEL_ID, "#7f8c8d")

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;").replace("'", "&#39;"))

    window_lookup = {int(w["index"]): w for w in window_info}
    window_start_map: Dict[int, List[int]] = {}
    window_end_map: Dict[int, List[int]] = {}
    for entry in window_info:
        idx = int(entry["index"])
        window_start_map.setdefault(int(entry["start_char"]), []).append(idx)
        window_end_map.setdefault(int(entry["end_char"]), []).append(idx)
    for mapping in (window_start_map, window_end_map):
        for key in mapping:
            mapping[key].sort()

    def marker_html(idx: int, position: str) -> str:
        info = window_lookup.get(idx)
        char_pos = info.get(f"{position}_char") if info else None
        byte_pos = info.get(f"{position}_byte") if info else None
        details = []
        if isinstance(char_pos, int):
            details.append(f"char {char_pos}")
        if isinstance(byte_pos, int):
            details.append(f"byte {byte_pos}")
        label = f"Window {idx + 1} {position}"
        if details:
            label += " (" + ", ".join(details) + ")"
        safe_label = esc(label)
        return (
            f'<span class="window-marker window-marker-{position}" '
            f'data-window="{idx + 1}" data-position="{position}" '
            f'title="{safe_label}" aria-label="{safe_label}">|</span>'
        )

    out_html: List[str] = []
    for (s, e, lbl) in segs:
        raw = text[s:e]
        cls = ID2SLUG.get(lbl, f"class-{lbl}")
        color = id2color_run.get(lbl, "#888888")
        border_color = _hex_to_rgba(color, 0.35)
        chars_html = []
        for i, ch in enumerate(raw):
            char_idx = s + i
            for win_idx in window_start_map.get(char_idx, []):
                chars_html.append(marker_html(win_idx, "start"))
            for win_idx in window_end_map.get(char_idx, []):
                chars_html.append(marker_html(win_idx, "end"))
            probs = char_probs[char_idx]
            conf = 1.0
            if isinstance(probs, dict) and probs:
                try:
                    key = str(int(lbl))
                except Exception:
                    key = str(lbl)
                try:
                    conf = float(probs.get(key, 0.0))
                except Exception:
                    conf = 0.0
            char_bg = _hex_to_rgba_confidence(color, 0.22, conf)
            # Only keep top probabilities to keep HTML light.
            sorted_items = sorted(
                ((key, float(value)) for key, value in probs.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )
            top_items = sorted_items[:5]
            remaining = sum(prob for _, prob in sorted_items[5:])
            payload = {key: prob for key, prob in top_items}
            if remaining > 1e-6:
                payload["others"] = remaining
            probs_attr = esc(json.dumps(payload))
            display_label = esc(ID2NAME.get(lbl, str(lbl)))
            chars_html.append(
                f'<span class="char" style="background-color:{char_bg};" '
                f'data-probs="{probs_attr}" data-label="{display_label}">{esc(ch)}</span>'
            )
        out_html.append(
            f'<span class="seg {cls}" '
            f'data-label="{esc(ID2NAME.get(lbl, str(lbl)))}" '
            f'style="--seg-color:{color}; background-color: transparent !important; '
            f'box-shadow: inset 0 -1px 0 {border_color};">'
            f'{"".join(chars_html)}</span>'
        )
    total_chars = len(char_labels)
    trailing_markers: List[str] = []
    for win_idx in window_end_map.get(total_chars, []):
        trailing_markers.append(marker_html(win_idx, "end"))
    if trailing_markers:
        out_html.append("".join(trailing_markers))
    html_joined = "".join(out_html)

    stats = [
        {
            "id": i,
            "name": ID2NAME.get(i, str(i)),
            "count": int(counts.get(i, 0)),
            "pct": float(100.0 * counts.get(i, 0) / total),
            "color": id2color_run.get(i, "#888888"),
        }
        for i in used_ids
    ]

    return {
        "segments": [{"start": int(s), "end": int(e), "label": int(lbl)} for (s, e, lbl) in segs],
        "stats": stats,
        "html": html_joined,
        "elapsed_ms": elapsed_ms,
        "window_info": window_info,
    }

if __name__ == "__main__":
    import uvicorn
    # When executing this file directly, `app` is defined in this module.
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
