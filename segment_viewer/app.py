#!/usr/bin/env python3
"""
Segmenter Viewer — FastAPI backend + beautiful frontend

Run:
  pip install fastapi uvicorn jax jaxlib flax optax numpy orbax-checkpoint
  # (Install the right jax/jaxlib for your CUDA setup if using GPU.)
  python app.py --ckpt ./seg-unet1d.msgpack --num-classes 3 --model-dim 128 --channels 128,256,384,512 --dtype bfloat16 --chunk 1024

Then open http://127.0.0.1:8000
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from typing import List, Tuple, Optional, Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from collections.abc import Mapping
import dataclasses

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
import flax.serialization as serialization

from collections.abc import Mapping
import dataclasses
from pathlib import Path
import re
import orbax.checkpoint as ocp
import flax.serialization as serialization

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

def _find_latest_orbax_step_dir(root: Path):
    root = Path(root)
    if _looks_like_orbax_step_dir(root):
        return root
    if not root.exists() or not root.is_dir():
        return None
    step_dirs = []
    for d in root.iterdir():
        if d.is_dir() and _looks_like_orbax_step_dir(d):
            m = re.search(r"-(\d+)$", d.name)
            step = int(m.group(1)) if m else -1
            step_dirs.append((step, d.name, d))
    if not step_dirs:
        return None
    step_dirs.sort(key=lambda t: (t[0], ".msgpack" in t[1]))
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
    if isinstance(obj, Mapping):
        if "params" in obj:
            return obj["params"]
        for key in ("target", "state", "train_state", "flax_state"):
            if key in obj:
                try:
                    return _extract_params_tree(obj[key])
                except Exception:
                    pass
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
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            try:
                return _extract_params_tree(getattr(obj, f.name))
            except Exception:
                continue
    elif hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            try:
                return _extract_params_tree(v)
            except Exception:
                continue
    raise KeyError("Could not find 'params' subtree in restored checkpoint object.")

def _load_params_from_any(ckpt_path: str, params_template_for_msgpack):
    p = Path(ckpt_path).resolve()

    # --- Case 1: a raw Flax .msgpack ---
    if p.is_file():
        with open(p, "rb") as f:
            raw = f.read()
        try:
            return serialization.from_bytes(params_template_for_msgpack, raw)
        except Exception:
            dct = serialization.from_bytes({"params": params_template_for_msgpack}, raw)
            return dct["params"]

    # --- Case 2: an Orbax directory (step dir or its parent) ---
    step_dir = _find_latest_orbax_step_dir(p)
    if step_dir is None:
        raise FileNotFoundError(f"Checkpoint path not found/unsupported: {ckpt_path}")
    step_dir_abs = step_dir.resolve().as_posix()
    ckptr = ocp.StandardCheckpointer()

    last_err = None

    # 2a) Untyped restore (often returns a TrainState)
    try:
        restored = ckptr.restore(step_dir_abs)
        return _extract_params_tree(restored)
    except Exception as e:
        last_err = e
        print(f"[orbax] untyped restore failed: {e}", flush=True)

    # 2b) Typed restore: provide only the params subtree as the target
    try:
        tmpl = {"params": params_template_for_msgpack}
        restored = ckptr.restore(step_dir_abs, target=tmpl, strict=False)
        return _extract_params_tree(restored)
    except Exception as e:
        last_err = e
        print(f"[orbax] target={{'params': ...}} restore failed: {e}", flush=True)

    # 2c) Typed restore: provide a dummy TrainState structure as the target
    try:
        from flax.training import train_state as ts
        import optax
        # Any tx works; we only need structure. Identity keeps it light.
        tx = optax.identity()
        dummy_state = ts.TrainState.create(
            apply_fn=lambda *a, **k: None,
            params=params_template_for_msgpack,
            tx=tx,
        )
        restored = ckptr.restore(step_dir_abs, target=dummy_state, strict=False)
        return restored.params
    except Exception as e:
        last_err = e
        print(f"[orbax] target=TrainState restore failed: {e}", flush=True)

    # If we get here, everything failed.
    raise RuntimeError(
        f"Orbax restore failed for '{step_dir_abs}'. "
        f"Tried untyped, target={{'params': ...}}, and target=TrainState. "
        f"Last error: {last_err}"
    )
# ---------------------------
# Input tokenization constants (match training)
# ---------------------------
BYTE_VOCAB_SIZE = 256
PAD_BYTE_ID = 256
NUM_TOKEN_EMBEDDINGS = BYTE_VOCAB_SIZE + 1  # 257

# ---------------------------
# Model (must mirror training EXACTLY)
# ---------------------------

class ConvBlock1D(nn.Module):
    features: int
    kernel_size: int = 3
    groups: int = 8
    dropout_rate: float = 0.0  # retained for parity; inactive at eval
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, train: bool):
        h = nn.Conv(self.features, (self.kernel_size,), padding="SAME",
                    dtype=self.dtype, param_dtype=self.dtype)(x)  # use_bias=True, like training
        h = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(h)
        h = nn.gelu(h)
        if self.dropout_rate > 0.:
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)
        return h

def upsample_nn_1d(x, factor: int):
    return jnp.repeat(x, repeats=factor, axis=1)

class UNet1D(nn.Module):
    num_classes: int
    emb_dim: int = 128
    channels: Tuple[int, ...] = (128, 256, 384, 512)
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = False):
        # tokens are int32 with possible PAD_BYTE_ID=256
        tok_i32 = tokens.astype(jnp.int32)
        h = nn.Embed(num_embeddings=NUM_TOKEN_EMBEDDINGS, features=self.emb_dim,
                     embedding_init=nn.initializers.normal(stddev=0.02),
                     dtype=self.dtype, param_dtype=self.dtype)(tok_i32)

        skips = []
        # Down-sampling path: two ConvBlocks per level then max-pool (except last)
        for i, ch in enumerate(self.channels):
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)
            skips.append(h)
            if i < len(self.channels) - 1:
                h = nn.max_pool(h, (2,), strides=(2,), padding="SAME")

        # Up-sampling path: NN upsample, pad/crop to match, concat skip, two ConvBlocks
        for i, ch in enumerate(reversed(self.channels[:-1])):
            h = upsample_nn_1d(h, factor=2)
            skip = skips[-(i + 2)]
            if h.shape[1] != skip.shape[1]:
                if h.shape[1] < skip.shape[1]:
                    h = jnp.pad(h, ((0, 0), (0, skip.shape[1] - h.shape[1]), (0, 0)))
                else:
                    h = h[:, :skip.shape[1], :]
            h = jnp.concatenate([h, skip], axis=-1)
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)

        logits_bf16 = nn.Conv(self.num_classes, (1,), padding="SAME",
                              dtype=self.dtype, param_dtype=self.dtype)(h)
        return logits_bf16.astype(jnp.float32)

# ---------------------------
# Predictor
# ---------------------------

def make_slug(name: str) -> str:
    import re
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

class Predictor:
    def __init__(self, ckpt_path: str, num_classes: int, model_dim: int,
                 channels: Tuple[int, ...], dtype_str: str = "bfloat16", chunk: int = 1024):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        self.num_classes = int(num_classes)
        self.chunk = int(chunk)
        self.dtype = getattr(jnp, dtype_str)
        self.model = UNet1D(num_classes=self.num_classes, emb_dim=int(model_dim),
                            channels=tuple(channels), dtype=self.dtype)
        # Init with dummy to create param structure (int32 tokens to allow PAD_BYTE_ID=256)
        dummy_tokens = jnp.full((1, 512), PAD_BYTE_ID, dtype=jnp.int32)
        variables = self.model.init({"params": jax.random.PRNGKey(0)}, dummy_tokens, train=False)
        params_template_for_msgpack = variables["params"]

        # Load params from either a .msgpack file or an Orbax directory/root (untyped restore).
        self.params = _load_params_from_any(ckpt_path, params_template_for_msgpack)

        # Precompile apply fn; JIT caches per-seq-length (shape-polymorphic)
        self._apply = jax.jit(lambda tok: self.model.apply({"params": self.params}, tok, train=False))

    def _segment_bytes(self, byte_arr: np.ndarray, chunk: int = None) -> tuple[np.ndarray, np.ndarray]:
        chunk = int(chunk or self.chunk)
        N = int(len(byte_arr))
        out = np.zeros((N,), dtype=np.uint8)
        probs = np.zeros((N, self.num_classes), dtype=np.float32)
        if N == 0:
            return out, probs
        win = max(64, int(chunk))
        stride = max(1, win // 2)
        xs, idxs = [], []
        for start in range(0, max(1, N - win + 1), stride):
            xs.append(byte_arr[start:start+win])
            idxs.append((start, min(start + win, N)))
        if not xs:
            xs, idxs = [byte_arr], [(0, N)]
        bs = 16
        for i in range(0, len(xs), bs):
            batch = xs[i:i+bs]
            maxL = max(len(b) for b in batch)
            tokens = np.full((len(batch), maxL), PAD_BYTE_ID, dtype=np.int32)
            for j, b in enumerate(batch):
                tokens[j, :len(b)] = b.astype(np.int32)
            logits = self._apply(jnp.array(tokens, dtype=jnp.int32))
            # Convert logits to probabilities using softmax
            probs_batch = jax.nn.softmax(logits, axis=-1)
            pred = np.argmax(np.array(logits), axis=-1).astype(np.uint8)
            for j, (s, e) in enumerate(idxs[i:i+bs]):
                plen = e - s
                out[s:e] = pred[j, :plen]
                probs[s:e] = np.array(probs_batch[j, :plen])
        return out, probs

    def _byte_labels_to_char_labels(self, text: str, byte_labels: np.ndarray, byte_probs: np.ndarray = None) -> tuple[List[int], List[Dict[str, float]]]:
        labels = []
        char_probs = []
        bpos = 0
        for ch in text:
            cb = ch.encode("utf-8")
            L = len(cb)
            if L == 0:
                labels.append(0)
                char_probs.append({str(i): 0.0 for i in range(self.num_classes)})
                continue
            seg = byte_labels[bpos:bpos+L]
            if len(seg) == 0:
                lbl = 0
                probs = {str(i): 0.0 for i in range(self.num_classes)}
            else:
                vals, counts = np.unique(seg, return_counts=True)
                lbl = int(vals[np.argmax(counts)])
                if byte_probs is not None:
                    # Average probabilities across bytes in the character
                    avg_probs = np.mean(byte_probs[bpos:bpos+L], axis=0)
                    probs = {str(i): float(avg_probs[i]) for i in range(self.num_classes)}
                else:
                    probs = {str(i): 1.0 if i == lbl else 0.0 for i in range(self.num_classes)}
            labels.append(lbl)
            char_probs.append(probs)
            bpos += L
        return labels, char_probs

    def _smooth_min_run(self, labels: List[int], min_run: int) -> List[int]:
        if min_run <= 1 or len(labels) == 0:
            return labels
        runs = []
        cur = labels[0]; start = 0
        for i in range(1, len(labels)):
            if labels[i] != cur:
                runs.append((start, i, cur))
                start = i; cur = labels[i]
        runs.append((start, len(labels), cur))
        if len(runs) <= 2:
            return labels
        arr = labels[:]
        for k, (s, e, lbl) in enumerate(runs):
            length = e - s
            if length >= min_run:
                continue
            left_lbl = runs[k-1][2] if k-1 >= 0 else lbl
            right_lbl = runs[k+1][2] if k+1 < len(runs) else lbl
            left_len = runs[k-1][1] - runs[k-1][0] if k-1 >= 0 else 0
            right_len = runs[k+1][1] - runs[k+1][0] if k+1 < len(runs) else 0
            new_lbl = left_lbl if left_len >= right_len else right_lbl
            for i in range(s, e):
                arr[i] = new_lbl
        return arr

    def segment_text(self, text: str, min_run_chars: int = 6, chunk: int = None):
        b = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
        byte_labels, byte_probs = self._segment_bytes(b, chunk=chunk)
        char_labels, char_probs = self._byte_labels_to_char_labels(text, byte_labels, byte_probs)
        char_labels = self._smooth_min_run(char_labels, int(min_run_chars))
        segs = []
        if len(char_labels) > 0:
            cur = char_labels[0]; start = 0
            for i in range(1, len(char_labels)):
                if char_labels[i] != cur:
                    segs.append((start, i, cur))
                    start = i; cur = char_labels[i]
            segs.append((start, len(char_labels), cur))
        return segs, char_labels, char_probs

# ---------------------------
# FastAPI wiring
# ---------------------------

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt",
    type=str,
    required=True,
    help="Path to a .msgpack file OR an Orbax checkpoint dir/root"
)
parser.add_argument("--model-dim", type=int, default=128)
parser.add_argument("--channels", type=str, default="128,256,384,512")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16","float32","float16"])
parser.add_argument("--chunk", type=int, default=1024, help="Inference window")
parser.add_argument("--labels", type=str, default="html,css,javascript,c,cpp,csv,java,json,python,text", help="Comma-separated class names")
parser.add_argument("--colors", type=str, default="#e67e22,#3498db,#f1c40f,#9b59b6,#2ecc71,#1abc9c,#e74c3c,#8e44ad,#16a085,#95a5a6", help="Optional comma-separated hex colors per class")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--openapi", action="store_true")
args, _ = parser.parse_known_args()

channels = tuple(int(x) for x in args.channels.split(",") if x.strip())
label_names = [x.strip() for x in args.labels.split(",") if x.strip()]
num_classes = len(label_names)
if len(label_names) != num_classes:
    while len(label_names) < num_classes:
        label_names.append(f"class-{len(label_names)}")
    label_names = label_names[:num_classes]

def auto_color(k: int, n: int) -> str:
    import colorsys
    h = (k / max(n,1)) % 1.0
    s, l = 0.65, 0.55
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

if args.colors:
    cols = [c.strip() for c in args.colors.split(",") if c.strip()]
else:
    defaults = ["#e67e22", "#3498db", "#f1c40f"]
    cols = [defaults[i] if i < len(defaults) else auto_color(i, num_classes) for i in range(num_classes)]
if len(cols) != num_classes:
    while len(cols) < num_classes:
        cols.append(auto_color(len(cols), num_classes))
    cols = cols[:num_classes]

ID2NAME = {i: label_names[i] for i in range(num_classes)}
ID2COLOR = {i: cols[i] for i in range(num_classes)}
ID2SLUG  = {i: make_slug(ID2NAME[i]) for i in range(num_classes)}

predictor = None
load_error = None
try:
    predictor = Predictor(ckpt_path=args.ckpt, num_classes=num_classes,
                          model_dim=args.model_dim, channels=channels,
                          dtype_str=args.dtype, chunk=args.chunk)
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
    return {
        "num_classes": num_classes,
        "labels": [{"id": i, "name": ID2NAME[i], "slug": ID2SLUG[i], "color": ID2COLOR[i]}
                   for i in range(num_classes)],
        "device": str(jax.devices()),
        "loaded": load_error is None,
        "error": load_error,
    }

@app.post("/api/segment")
def api_segment(req: SegmentRequest):
    if load_error is not None or predictor is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {load_error}")
    if not isinstance(req.text, str) or len(req.text) == 0:
        return {"segments": [], "stats": {}, "html": ""}

    segs, char_labels, char_probs = predictor.segment_text(
        req.text, min_run_chars=int(req.min_run),
        chunk=int(req.chunk) if req.chunk else args.chunk
    )

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;").replace("'", "&#39;"))

    import json
    out_html = []
    for (s, e, lbl) in segs:
        raw = req.text[s:e]
        cls = ID2SLUG.get(lbl, f"class-{lbl}")
        color = ID2COLOR.get(lbl, "#888888")
        # Create character spans with probability data
        chars_html = []
        for i, ch in enumerate(raw):
            char_idx = s + i
            probs = char_probs[char_idx]
            # Properly format as JSON string
            probs_attr = esc(json.dumps(probs))
            chars_html.append(f'<span class="char" data-probs="{probs_attr}">{esc(ch)}</span>')
        # Add single span that combines coloring and character-level probabilities
        out_html.append(
            f'<span class="seg {cls}" data-label="{esc(ID2NAME.get(lbl, str(lbl)))}" '
            f'style="--seg-color:{color}; background: linear-gradient(0deg, {color}22, {color}22), transparent;">'
            f'{"".join(chars_html)}</span>'
        )
    html_joined = "".join(out_html)

    import collections
    counts = collections.Counter(char_labels)
    total = max(1, sum(counts.values()))
    stats = [{"id": i, "name": ID2NAME[i], "count": int(counts.get(i, 0)),
              "pct": float(100.0 * counts.get(i, 0) / total), "color": ID2COLOR[i]}
             for i in range(num_classes)]

    return {
        "segments": [{"start": int(s), "end": int(e), "label": int(lbl)} for (s, e, lbl) in segs],
        "stats": stats,
        "html": html_joined,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
