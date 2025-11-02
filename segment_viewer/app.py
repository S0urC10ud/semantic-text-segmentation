#!/usr/bin/env python3
"""
Segmenter Viewer — FastAPI backend + beautiful frontend

Run:
  pip install fastapi uvicorn jax jaxlib flax optax numpy orbax-checkpoint
  # (Install the right jax/jaxlib for your CUDA setup if using GPU.)
  python app.py --ckpt ./seg-unet1d.msgpack --model-dim 256 --channels 96,128,192,256 --dtype bfloat16 --chunk 1024 --lang html,css,javascript_typescript,php

Then open http://127.0.0.1:8000
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from typing import List, Tuple, Optional, Any, Dict, Sequence
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
import json
import importlib.util
import orbax.checkpoint as ocp
import flax.serialization as serialization

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_train_module(module_name: str):
    module_path = REPO_ROOT / "train" / f"{module_name}.py"
    if not module_path.exists():
        raise ImportError(
            f"Expected to find train/{module_name}.py next to segment_viewer, "
            f"but {module_path} does not exist."
        )
    spec = importlib.util.spec_from_file_location(
        f"segment_viewer.train_{module_name}", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for train/{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    TRAIN_CONFIG = _load_train_module("config")
except ImportError:
    TRAIN_CONFIG = None

DEFAULT_CHANNELS: Tuple[int, ...] = (96, 128, 192, 256)


def _apply_label_mapping(label_names: Sequence[str]) -> None:
    if not label_names or TRAIN_CONFIG is None:
        return
    LANG2ID = getattr(TRAIN_CONFIG, "LANG2ID", None)
    update_fn = getattr(TRAIN_CONFIG, "update_lang_mappings", None)
    if not isinstance(LANG2ID, dict):
        return
    LANG2ID.clear()
    for idx, name in enumerate(label_names):
        LANG2ID[name] = idx
    if callable(update_fn):
        update_fn()


def _configured_languages() -> List[str]:
    if TRAIN_CONFIG is None:
        raise RuntimeError(
            "train/config.py could not be loaded; unable to resolve label ordering automatically."
        )
    mapping = getattr(TRAIN_CONFIG, "LANG2ID", None)
    if not isinstance(mapping, dict):
        raise RuntimeError("train/config.py does not define LANG2ID mapping")
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def _resolve_langs_and_display(
    lang_arg: Optional[str],
) -> Tuple[List[str], List[str]]:
    configured_langs = _configured_languages()
    canonical_map = {lang.lower(): lang for lang in configured_langs}

    if not lang_arg:
        return configured_langs, configured_langs[:]

    requested: List[str] = []
    display_map: Dict[str, str] = {}
    for raw in lang_arg.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
        else:
            key = item
            value = ""
        key_lower = key.lower()
        if key_lower not in canonical_map:
            raise ValueError(
                f"Unknown language '{key}'. Available: {sorted(canonical_map.values())}"
            )
        canonical = canonical_map[key_lower]
        requested.append(canonical)
        if value:
            display_map[canonical] = value

    deduped = list(dict.fromkeys(requested))
    if not deduped:
        raise ValueError("No languages resolved from --lang.")

    selection_set = set(deduped)
    ordered = [lang for lang in configured_langs if lang in selection_set]
    if not ordered:
        raise ValueError("No overlap between --lang selection and training labels.")

    if ordered != deduped:
        print(
            f"Subset derived from --lang reordered to match training: {ordered}",
            flush=True,
        )

    display_names = [display_map.get(name, name) for name in ordered]
    return ordered, display_names


def _extract_run_id_from_checkpoint(path: Path) -> Optional[str]:
    name = path.name.lower()
    match = re.search(r"([a-z0-9]{8})", name)
    if not match:
        return None
    return match.group(1)

def _parse_label_names_from_output(log_text: str) -> List[str]:
    labels: List[str] = []
    seen = set()
    collecting = False
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if line.startswith("[Validation] Per-class metrics"):
            collecting = False
            continue
        if line.startswith("label ") and "support" in line:
            collecting = True
            continue
        if not collecting:
            continue
        if line.startswith("-"):
            continue
        if not line or line.startswith("Step "):
            collecting = False
            continue
        if line.startswith("ALL"):
            continue
        if line.startswith("WARNING") or line.startswith("INFO"):
            collecting = False
            continue
        parts = line.split()
        if not parts:
            continue
        label = parts[0]
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _load_checkpoint_hparams(ckpt_path: Path) -> Dict[str, Any]:
    run_id = _extract_run_id_from_checkpoint(ckpt_path)
    if not run_id:
        return {}
    wandb_root = REPO_ROOT / "train" / "wandb"
    if not wandb_root.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    pattern = f"run-*-{run_id}"
    for run_dir in wandb_root.glob(pattern):
        config_path = run_dir / "files" / "config.yaml"
        if not config_path.exists():
            continue
        try:
            config_data = yaml.safe_load(config_path.read_text())
        except Exception:
            config_data = None
        if not isinstance(config_data, dict):
            config_data = {}
        result: Dict[str, Any] = {}
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
                    result["_label_source"] = "wandb"
        # Fallback: parse label order from output.log
        output_log = run_dir / "files" / "output.log"
        if output_log.exists():
            parsed_labels = _parse_label_names_from_output(output_log.read_text())
            if parsed_labels:
                result.setdefault("label_names", parsed_labels)
                result.setdefault("_label_source", "output_log")
        if result:
            return result
    return {}

def _infer_checkpoint_architecture(ckpt_path: Path) -> Dict[str, Any]:
    """Best-effort inference of model hyperparameters from a Flax msgpack checkpoint."""
    result: Dict[str, Any] = {}
    p = ckpt_path.resolve()
    if not p.is_file():
        return result
    try:
        params = serialization.msgpack_restore(p.read_bytes())
    except Exception:
        return result

    # Embedding dimension -> model_dim
    try:
        embedding = params["Embed_0"]["embedding"]
        result["model_dim"] = int(embedding.shape[1])
        result["dtype"] = getattr(embedding.dtype, "name", str(embedding.dtype))
    except Exception:
        pass

    # Down path channels: walk ConvBlock1D_{0,2,4,...} until we hit the up path
    channels: List[int] = []
    current_in = result.get("model_dim")
    block_idx = 0
    while True:
        name = f"ConvBlock1D_{block_idx}"
        block = params.get(name)
        if block is None:
            break
        conv = block.get("Conv_0")
        if conv is None:
            break
        kernel = conv.get("kernel")
        if kernel is None:
            break
        in_ch = int(kernel.shape[-2])
        out_ch = int(kernel.shape[-1])
        if block_idx == 0 and current_in is None:
            current_in = in_ch
            result["model_dim"] = in_ch
        if channels and current_in is not None and in_ch != current_in:
            break
        channels.append(out_ch)
        current_in = out_ch
        block_idx += 2  # skip the paired block belonging to the same stage
    if channels:
        result["channels"] = channels

    # Output layer -> num_classes
    try:
        result["num_classes"] = int(params["Conv_0"]["kernel"].shape[-1])
    except Exception:
        pass

    return result

def _resolve_hparam(cli_value, wandb_value, inferred_value, default_value):
    """Pick a hyperparameter value while recording its source."""
    if cli_value not in (None, "", []):
        return cli_value, "cli"
    if wandb_value is not None:
        return wandb_value, "wandb"
    if inferred_value is not None:
        return inferred_value, "checkpoint"
    return default_value, "default"


def auto_color(k: int, n: int) -> str:
    import colorsys

    h = (k / max(n, 1)) % 1.0
    s, l = 0.65, 0.55
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


DEFAULT_COLOR_BY_LABEL = {
    "html": "#f2994a",
    "css": "#3498db",
    "javascript_typescript": "#f1c40f",
    "php": "#9b59b6",
    "python": "#2ecc71",
    "json": "#1abc9c",
    "sql": "#e74c3c",
    "java": "#8e44ad",
    "go": "#16a085",
    "c_family": "#2ecc71",
    "csharp": "#1abc9c",
    "csv": "#e74c3c",
    "ruby": "#8e44ad",
    "rust": "#16a085",
    "text": "#95a5a6",
    "yaml": "#d35400",
    "powershell": "#8e44ad",
    "shell": "#636e72",
}


def _default_color_for_label(name: str, index: int, total: int) -> str:
    return DEFAULT_COLOR_BY_LABEL.get(name.lower(), auto_color(index, total))


def _resolve_colors(base_names: List[str], colors_arg: Optional[str]) -> List[str]:
    total = len(base_names)
    if total == 0:
        return []

    if not colors_arg:
        return [
            _default_color_for_label(name, idx, total)
            for idx, name in enumerate(base_names)
        ]

    entries = [item.strip() for item in colors_arg.split(",") if item.strip()]
    if not entries:
        return [
            _default_color_for_label(name, idx, total)
            for idx, name in enumerate(base_names)
        ]

    named: Dict[str, str] = {}
    positional: List[str] = []
    for entry in entries:
        if "=" in entry:
            key, value = entry.split("=", 1)
            key = key.strip()
            value = value.strip()
            match = next((name for name in base_names if name.lower() == key.lower()), None)
            if match is None:
                raise ValueError(f"Color override references unknown class '{key}'.")
            if value:
                named[match] = value
        else:
            positional.append(entry)

    if named:
        if positional:
            raise ValueError("Mixing named and positional colors in --colors is not supported.")
        return [
            named.get(name, _default_color_for_label(name, idx, total))
            for idx, name in enumerate(base_names)
        ]

    # Pure positional overrides
    colors = positional[:total]
    while len(colors) < total:
        idx = len(colors)
        colors.append(_default_color_for_label(base_names[idx], idx, total))
    return colors[:total]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = (hex_color or "").strip().lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    if len(hex_color) != 6:
        return f"rgba(136, 136, 136, {max(0.0, min(alpha, 1.0)):.2f})"
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError:
        return f"rgba(136, 136, 136, {max(0.0, min(alpha, 1.0)):.2f})"
    return f"rgba({r}, {g}, {b}, {max(0.0, min(alpha, 1.0)):.2f})"

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
        arr_np[invalid] = _CURRENCY_BYTE_ID
    return arr_np

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

    @staticmethod
    def _window_weights(length: int) -> np.ndarray:
        """Return center-weighted coefficients for a window of given length."""
        if length <= 1:
            return np.ones((length,), dtype=np.float32)
        positions = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
        sigma = 0.5
        weights = np.exp(-0.5 * (positions / sigma) ** 2)
        return weights.astype(np.float32)

    def _segment_bytes(self, byte_arr: np.ndarray, chunk: int = None) -> tuple[np.ndarray, np.ndarray]:
        chunk = int(chunk or self.chunk)
        N = int(len(byte_arr))
        out = np.zeros((N,), dtype=np.uint8)
        probs_accum = np.zeros((N, self.num_classes), dtype=np.float32)
        weight_accum = np.zeros((N,), dtype=np.float32)
        if N == 0:
            return out, probs_accum
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
            tokens = _sanitize_model_tokens(tokens)
            logits = self._apply(jnp.array(tokens, dtype=jnp.int32))
            # Convert logits to probabilities using softmax
            probs_batch = np.array(jax.nn.softmax(logits, axis=-1))
            for j, (s, e) in enumerate(idxs[i:i+bs]):
                plen = e - s
                if plen <= 0:
                    continue
                weights = self._window_weights(plen)
                window_probs = probs_batch[j, :plen]
                probs_accum[s:e] += window_probs * weights[:, None]
                weight_accum[s:e] += weights
        if np.any(weight_accum > 0):
            nonzero = weight_accum > 0
            probs_accum[nonzero] /= weight_accum[nonzero, None]
            zero_mask = ~nonzero
            if np.any(zero_mask):
                probs_accum[zero_mask] = 1.0 / self.num_classes
        else:
            probs_accum[:] = 1.0 / self.num_classes
        out = np.argmax(probs_accum, axis=-1).astype(np.uint8)
        return out, probs_accum

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
        b = _sanitize_model_bytes(np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8))
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
parser.add_argument("--model-dim", type=int, default=None, help="Model embedding dimension (auto if omitted).")
parser.add_argument("--channels", type=str, default=None, help="Comma-separated channel sizes (auto if omitted).")
parser.add_argument("--dtype", type=str, default=None, help="Model dtype name (auto if omitted).")
parser.add_argument("--chunk", type=int, default=1024, help="Inference window")
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
args, _ = parser.parse_known_args()

ckpt_path = Path(args.ckpt).resolve()
auto_hparams = _load_checkpoint_hparams(ckpt_path)
ckpt_inferred = _infer_checkpoint_architecture(ckpt_path)

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
if isinstance(dtype_value, str):
    dtype = dtype_value.rsplit(".", 1)[-1]
else:
    dtype = str(dtype_value)

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

args.model_dim = model_dim
args.dtype = dtype
args.channels = ",".join(str(ch) for ch in channels)

if auto_hparams or ckpt_inferred:
    details = [
        f"model_dim={model_dim} ({model_dim_source})",
        f"channels={list(channels)} ({channels_source})",
        f"dtype={dtype} ({dtype_source})",
    ]
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
        "labels": [{"id": i, "name": ID2NAME[i], "slug": ID2SLUG[i], "color": ID2COLOR.get(i)}
                   for i in range(num_classes) if ID2COLOR.get(i)],
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

    import time as _time
    t0 = _time.perf_counter()
    try:
        segs, char_labels, char_probs = predictor.segment_text(
            text, min_run_chars=int(req.min_run),
            chunk=int(req.chunk) if req.chunk else args.chunk
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = float((_time.perf_counter() - t0) * 1000.0)

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;").replace("'", "&#39;"))

    out_html = []
    for (s, e, lbl) in segs:
        raw = text[s:e]
        cls = ID2SLUG.get(lbl, f"class-{lbl}")
        color = ID2COLOR.get(lbl, "#888888")
        bg_color = _hex_to_rgba(color, 0.22)
        border_color = _hex_to_rgba(color, 0.35)
        chars_html = []
        for i, ch in enumerate(raw):
            char_idx = s + i
            probs = char_probs[char_idx]
            # Aggregate by canonical label
            agg: Dict[str, float] = {}
            for key, value in probs.items():
                try:
                    label_idx = int(key)
                except (TypeError, ValueError):
                    continue
                base_label = ID2CANONICAL.get(label_idx)
                if base_label is None:
                    continue
                agg[base_label] = agg.get(base_label, 0.0) + float(value)
            sorted_items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
            top_items = sorted_items[:5]
            remaining = sum(prob for _, prob in sorted_items[5:])
            payload = {label: prob for label, prob in top_items}
            if remaining > 1e-6:
                payload["others"] = remaining
            probs_attr = esc(json.dumps(payload))
            display_label = esc(ID2NAME.get(lbl, str(lbl)))
            chars_html.append(
                f'<span class="char" data-probs="{probs_attr}" data-label="{display_label}">{esc(ch)}</span>'
            )
        out_html.append(
            f'<span class="seg {cls}" '
            f'data-label="{esc(ID2NAME.get(lbl, str(lbl)))}" '
            f'style="--seg-color:{color}; background-color:{bg_color}; '
            f'box-shadow: inset 0 -1px 0 {border_color};">'
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
        "elapsed_ms": elapsed_ms,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
