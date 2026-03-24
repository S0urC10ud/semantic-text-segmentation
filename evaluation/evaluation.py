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

import datasets as hfds  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.serialization as serialization  # noqa: E402
from flax.errors import ScopeParamShapeError  # noqa: E402
try:
    import orbax.checkpoint as ocp  # noqa: E402
except Exception:  # pragma: no cover - environment-dependent optional import
    ocp = None  # type: ignore[assignment]

from inference.backend import (  # noqa: E402
    FastInferenceEngine,
    FastInferenceFailure,
    format_auto_fallback_message,
)
from inference.mamba_cuda import has_cuda_mamba_kernel  # noqa: E402
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
_TEXT_LIKE_ID2LABEL: Dict[int, str] = {
    0: "not_text_like",
    1: "text_like",
}


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
    checkpointer = ocp.StandardCheckpointer()
    try:
        restored = checkpointer.restore(step_dir_str)
        params, _ = merge_compatible_state(
            params_template,
            checkpoint_params_subtree(_extract_params_tree(restored)),
        )
        return params
    except Exception:
        pass

    try:
        template = {"params": params_template}
        restored = checkpointer.restore(step_dir_str, target=template, strict=False)
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
        restored = checkpointer.restore(step_dir_str, target=dummy, strict=False)
        params, _ = merge_compatible_state(
            params_template,
            checkpoint_params_subtree(restored),
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


def _smooth_min_run(labels: List[int], min_run: int) -> List[int]:
    if min_run <= 1 or len(labels) == 0:
        return labels
    runs = []
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            runs.append((start, i, current))
            start = i
            current = labels[i]
    runs.append((start, len(labels), current))
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


class SegmenterRunner:
    def __init__(
        self,
        checkpoint_path: str,
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
    ):
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
        self.num_classes = cfg.NUM_CLASSES
        self.arch = str(arch).lower().strip()
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
        byte_arr = sanitize_bytes(byte_arr)
        N = int(len(byte_arr))
        if N == 0:
            return np.zeros((0,), dtype=np.uint8), np.zeros((0, self.num_classes), dtype=np.float32)

        win = max(64, int(self.chunk))
        stride = max(1, win // 2)
        windows: List[np.ndarray] = []
        spans: List[Tuple[int, int]] = []
        for start in range(0, max(1, N - win + 1), stride):
            end = min(start + win, N)
            windows.append(byte_arr[start:end])
            spans.append((start, end))
        if not windows:
            windows = [byte_arr]
            spans = [(0, N)]

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
        byte_arr = sanitize_bytes(byte_arr)
        N = int(len(byte_arr))
        if N == 0:
            return np.zeros((0,), dtype=np.uint8)

        win = max(64, int(self.chunk))
        stride = max(1, win // 2)
        windows: List[np.ndarray] = []
        spans: List[Tuple[int, int]] = []
        for start in range(0, max(1, N - win + 1), stride):
            end = min(start + win, N)
            windows.append(byte_arr[start:end])
            spans.append((start, end))
        if not windows:
            windows = [byte_arr]
            spans = [(0, N)]

        votes = np.zeros((N, self.num_classes), dtype=np.float32)
        batch_size = self.batch_size
        for i in range(0, len(windows), batch_size):
            span_slice = spans[i:i + batch_size]
            actual = len(span_slice)
            tokens = np.full((batch_size, self.chunk), cfg.PAD_BYTE_ID, dtype=np.int32)
            for j in range(actual):
                win_bytes = windows[i + j]
                length = min(len(win_bytes), self.chunk)
                if length > 0:
                    tokens[j, :length] = win_bytes[:length].astype(np.int32)
            tokens = sanitize_tokens(tokens)
            logits = self._apply_legacy(jnp.array(tokens, dtype=jnp.int32))
            labels = np.asarray(jax.device_get(jnp.argmax(logits, axis=-1).astype(jnp.uint8)))[:actual, :self.chunk]

            for j, (start, end) in enumerate(span_slice):
                plen = end - start
                if plen <= 0:
                    continue
                plen = min(plen, self.chunk)
                weights = self._window_weights_cached(plen)
                window_labels = np.asarray(labels[j, :plen], dtype=np.int64)
                positions = np.arange(int(start), int(end), dtype=np.int64)
                np.add.at(votes, (positions, window_labels), weights)

        return np.argmax(votes, axis=-1).astype(np.uint8)

    def _segment_bytes_labels_only(self, byte_arr: np.ndarray) -> np.ndarray:
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

    def segment_text(self, text: str, *, min_run_chars: int = 1) -> Tuple[List[Tuple[int, int, int]], List[int], List[np.ndarray]]:
        text = normalize_eval_text(text)
        byte_arr = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
        byte_labels, byte_probs = self._segment_bytes(byte_arr)
        char_labels, char_probs = _byte_labels_to_char_labels(text, byte_labels, byte_probs, self.num_classes)
        char_labels = _smooth_min_run(char_labels, min_run_chars)
        segments: List[Tuple[int, int, int]] = []
        if char_labels:
            cur = char_labels[0]
            start = 0
            for idx in range(1, len(char_labels)):
                if char_labels[idx] != cur:
                    segments.append((start, idx, cur))
                    start = idx
                    cur = char_labels[idx]
            segments.append((start, len(char_labels), cur))
        return segments, char_labels, char_probs

    def segment_text_labels_only(self, text: str, *, min_run_chars: int = 1) -> Tuple[List[Tuple[int, int, int]], List[int]]:
        text = normalize_eval_text(text)
        byte_arr = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
        byte_labels = self._segment_bytes_labels_only(byte_arr)
        char_labels = _byte_labels_to_char_labels_only(text, byte_labels)
        char_labels = _smooth_min_run(char_labels, min_run_chars)
        segments: List[Tuple[int, int, int]] = []
        if char_labels:
            cur = char_labels[0]
            start = 0
            for idx in range(1, len(char_labels)):
                if char_labels[idx] != cur:
                    segments.append((start, idx, cur))
                    start = idx
                    cur = char_labels[idx]
            segments.append((start, len(char_labels), cur))
        return segments, char_labels


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


def _evaluate_full_monitor_b(
    monitor_root: Path,
    runner: "SegmenterRunner",
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
        else "sliding_window_legacy"
    )

    start_time = time.perf_counter()
    confusion = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    files_used = 0
    skipped = 0
    evaluated_bytes = 0
    raw_bytes = 0

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
        truth = np.full((byte_len,), cfg.PAD_ID, dtype=np.uint8)
        for seg in segments[seg_start : seg_start + seg_count]:
            start = int(seg["start"])
            end = int(seg["end"])
            if end <= start:
                continue
            truth[start:end] = int(seg["label"])

        if use_mamba_path:
            pred = runner._segment_bytes_labels_only(file_bytes)
        else:
            pred = runner._segment_bytes_labels_only_legacy(file_bytes)
        pred = np.asarray(pred, dtype=np.int32).reshape(-1)
        if int(pred.shape[0]) != byte_len:
            raise RuntimeError(
                f"Monitor prediction length mismatch for file {file_idx}: "
                f"expected {byte_len}, got {int(pred.shape[0])}"
            )

        mask = _valid_metric_mask(truth, file_bytes.astype(np.int32, copy=False))
        core_mask = mask & (truth < cfg.NUM_CLASSES)
        if core_mask.any():
            accumulate_confusion(
                confusion,
                truth[core_mask].astype(np.int32, copy=False),
                pred[core_mask].astype(np.int32, copy=False),
            )
            evaluated_bytes += int(core_mask.sum())

        files_used += 1
        raw_bytes += int(byte_len)

    metrics_payload = _metrics_payload_from_confusion(
        confusion,
        id2label=cfg.ID2LANG,
        num_classes=cfg.NUM_CLASSES,
        ignore_class=cfg.PAD_ID,
    )
    metrics_payload.update(
        {
            "root": str(Path(monitor_root).resolve()),
            "arch": arch,
            "inference_mode": inference_mode,
            "files_total": int(len(files)),
            "files_used": int(files_used),
            "skipped": int(skipped),
            "evaluated_bytes": int(evaluated_bytes),
            "raw_bytes": int(raw_bytes),
            "elapsed_seconds": float(time.perf_counter() - start_time),
        }
    )
    return metrics_payload


def _default_monitor_b_root() -> Path:
    return (REPO_ROOT / "downloader" / "monitor_preprocessed_b").resolve()


def _configure_evaluation_mode(args) -> Dict[str, Any]:
    default_eval_root = (REPO_ROOT / "evaluation" / "data").resolve()
    fine_tune_eval_root = (REPO_ROOT / "evaluation" / "data_b").resolve()
    requested_root = Path(args.data_root).resolve()
    fine_tuned_mode = not bool(getattr(args, "non_fine_tuned", False))
    messages: List[str] = []

    if bool(getattr(args, "fine_tuned", False)):
        messages.append(
            "⚠️  --fine-tuned is deprecated; fine-tuned evaluation is now the default."
        )
    if fine_tuned_mode:
        if requested_root == default_eval_root:
            args.data_root = str(fine_tune_eval_root)
            messages.append(
                f"ℹ️  Fine-tuned mode (default): using evaluation data root {args.data_root}"
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
        "fine_tune_eval_root": str(fine_tune_eval_root),
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
) -> TaskMetrics:
    total_samples = len(dataset)
    # Always include an 'other' bucket for open-set handling.
    label_candidates = {"other"}
    for example in dataset:
        for seg in _normalize_segments(example["segments"]):
            label_candidates.add(seg["label"])
    for alias_target in PREDICTION_LABEL_ALIASES.values():
        label_candidates.add(alias_target)
    label_names, label_to_idx = _confusion_size(label_candidates)
    confusion = np.zeros((len(label_names), len(label_names)), dtype=np.int64)

    total_chars = 0
    per_label_counts: Dict[str, int] = {label: 0 for label in label_names}
    per_label_correct: Dict[str, int] = {label: 0 for label in label_names}
    extra_payload: Dict[str, Any] = {}

    pure_stats: Optional[Dict[str, int]] = None
    if name == "pure_fragments":
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

    sequence_stats: Optional[Dict[str, Any]] = None
    if name == "sequence_pair":
        sequence_stats = {
            "segments": {
                "first": {"correct": 0, "total": 0},
                "second": {"correct": 0, "total": 0},
            }
        }
        extra_payload["sequence_purity"] = sequence_stats
    elif name == "sequence_triplet":
        sequence_stats = {
            "segments": {
                "first": {"correct": 0, "total": 0},
                "second": {"correct": 0, "total": 0},
                "third": {"correct": 0, "total": 0},
            }
        }
        extra_payload["sequence_purity"] = sequence_stats

    start_time = time.perf_counter()
    log_interval = getattr(evaluate_task, "_log_interval", 0) or 0

    for idx, example in enumerate(dataset):
        row = example if isinstance(example, dict) else dict(example)
        raw_content = row.get("content")
        content = normalize_eval_text(raw_content if isinstance(raw_content, str) else "")
        normalized_segments = _normalize_segments(row.get("segments"))
        truth = _segments_to_labels(content, normalized_segments, label_to_idx)
        segments, pred_labels, pred_probs = runner.segment_text(content, min_run_chars=min_run_chars)
        pred_idx_array = np.full((len(pred_labels),), -1, dtype=np.int32)
        pred_text_like_flags = np.zeros((len(pred_labels),), dtype=bool)
        prob_rows: Optional[List[Optional[np.ndarray]]] = [None] * len(pred_labels) if payload_stats is not None else None
        other_idx_eval = label_to_idx.get("other")
        use_other_threshold = other_threshold is not None and float(other_threshold) > 0.0 and other_idx_eval is not None
        for i, lbl_id in enumerate(pred_labels):
            label_name = cfg.ID2LANG.get(int(lbl_id), None)
            if label_name is not None:
                alias = PREDICTION_LABEL_ALIASES.get(label_name)
                if alias and alias in label_to_idx:
                    label_name = alias
                pred_text_like_flags[i] = label_name in TEXT_LIKE_POSITIVE_LABELS

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

        valid_mask = (truth >= 0)
        if content:
            ws_mask = np.array([ch in _VISUAL_WHITESPACE_SET for ch in content], dtype=bool)
            valid_mask = np.logical_and(valid_mask, ~ws_mask)
        truth_valid = truth[valid_mask]
        pred_valid = pred_idx_array[valid_mask]
        pred_text_like_valid = pred_text_like_flags[valid_mask]
        prob_valid = [prob_rows[i] for i, keep in enumerate(valid_mask) if keep] if prob_rows is not None else None
        same_mask = (pred_valid == truth_valid)

        metadata = _parse_metadata(row)

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
            region_mask_valid = _region_mask_valid(
                len(content),
                valid_mask,
                metadata.get("inserted_char_start", metadata.get("needle_char_start")),
                metadata.get("inserted_char_end", metadata.get("needle_char_end")),
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
                wrapped_flag = bool(block.get("wrapped"))
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

            inline_meta = metadata.get("inline_blocks") or []
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

        if payload_stats is not None:
            payload_lang = metadata.get("payload_lang")
            host_lang = metadata.get("host_lang")
            host_idx = label_to_idx.get(host_lang) if host_lang else None
            if payload_lang:
                idx = label_to_idx.get(payload_lang)
                if idx is not None:
                    truth_mask = (truth_valid == idx)
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
                                payload_prob_mask = eval_idx_view == idx
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
                                        payload_prob = float(prob_row[row_eval_idx == idx].sum())
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

        if sequence_stats is not None:
            segments_info = sequence_stats["segments"]
            region_meta = metadata.get("sequence_regions")
            if isinstance(region_meta, dict):
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
                    idx = label_to_idx.get(lang)
                    if idx is None:
                        continue
                    truth_mask = (truth_valid == idx)
                    total_chars_seg = int(truth_mask.sum())
                    if total_chars_seg == 0:
                        continue
                    correct_chars_seg = int(np.logical_and(pred_valid == idx, truth_mask).sum())
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

    return TaskMetrics(
        name=name,
        description=description,
        samples=total_samples,
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
    samples: int
    total_bytes: int
    elapsed: float
    throughput: float
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
        return ThroughputResult(task, device_name, 0, 0, 0.0, 0.0, None, None)

    total_bytes = 0
    for example in dataset:
        meta_raw = example.get("metadata_json")
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
                total_bytes += int(meta.get("actual_bytes", 0))
            except Exception:
                pass
    if total_bytes <= 0:
        total_bytes = int(
            sum(
                len(
                    normalize_eval_text(
                        row.get("content") if isinstance(row, dict) else row["content"]
                    ).encode("utf-8", "ignore")
                )
                for row in dataset
            )
        )

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
    warm_example = dataset[0]
    warm_text = normalize_eval_text(
        warm_example.get("content") if isinstance(warm_example, dict) else warm_example["content"]
    )
    runner.segment_text_labels_only(warm_text, min_run_chars=min_run_chars)

    start = time.perf_counter()
    for example in dataset:
        text = normalize_eval_text(example.get("content") if isinstance(example, dict) else example["content"])
        runner.segment_text_labels_only(text, min_run_chars=min_run_chars)
    elapsed = time.perf_counter() - start

    rss_after = _process_rss_mb()
    dev_after = _device_mem_mb(device)

    rss_delta = (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None
    dev_delta = (dev_after - dev_before) if (dev_after is not None and dev_before is not None) else None

    throughput = (total_bytes / elapsed) if elapsed > 0 else 0.0
    return ThroughputResult(
        task=task,
        device=device_name,
        samples=samples,
        total_bytes=total_bytes,
        elapsed=elapsed,
        throughput=throughput,
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


def _render_metrics_table(metrics: TaskMetrics) -> str:
    rows = []
    for label in metrics.label_names:
        support = metrics.per_label_counts.get(label, 0)
        correct = metrics.per_label_correct.get(label, 0)
        acc = (correct / support) if support else float("nan")
        rows.append((label, support, acc))
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
) -> str:
    entries: List[Tuple[int, float, str, str]] = []
    confusion = metrics.confusion
    for true_idx, true_label in enumerate(metrics.label_names):
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
        add_section(m.name, section_lines)

    return sections


def _render_throughput_table(results: List[ThroughputResult]) -> str:
    lines = [
        "| Task | Device | Samples | Total Bytes | Throughput | Latency (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for res in results:
        lines.append(
            "| {task} | {device} | {samples} | {bytes} | {through} | {lat:.2f} |".format(
                task=res.task,
                device=res.device,
                samples=res.samples,
                bytes=res.total_bytes,
                through=_format_bytes_per_sec(res.throughput),
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


def _write_confusion_artifacts(report_path: Path, task_metrics: Sequence[TaskMetrics]) -> Dict[str, str]:
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
    data: Dict[str, Any] = {
        "meta": {
            "checkpoint": args.checkpoint,
            "model_dim": args.model_dim,
            "channels": list(args.channels) if isinstance(args.channels, (list, tuple)) else args.channels,
            "dtype": args.dtype,
            "sample_seed": args.sample_seed,
            "other_threshold": _float_or_none(getattr(args, "other_threshold", 0.0)),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "summary": {
            "other_confusion": other_summary,
        },
        "tasks": {},
    }

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

    # Pure fragments -------------------------------------------------------
    pure_metrics = metrics_by_name.get("pure_fragments")
    if pure_metrics and pure_metrics.extras:
        pure_stats = pure_metrics.extras.get("pure_fragments_purity", {})
        per_label = pure_stats.get("per_label", {})
        per_language = {}
        for lang, entry in per_label.items():
            total = int(entry.get("total", 0))
            if total <= 0:
                continue
            per_language[lang] = {
                "fully_pure_rate": _float_or_none(_safe_ratio(entry.get("pure", 0), total)),
                "within_threshold_rate": _float_or_none(_safe_ratio(entry.get("within", 0), total)),
                "support": total,
            }
        if per_language:
            data["tasks"]["pure_fragments"] = {"per_language": per_language}

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
            data["tasks"][name] = {
                "segments": segment_data,
                "overall_accuracy": _float_or_none(seq_metrics.overall_accuracy()),
            }

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
            "top_misclassifications": top_conf,
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
                "samples": res.samples,
                "total_bytes": res.total_bytes,
                "elapsed": _float_or_none(res.elapsed),
                "throughput_bytes_per_sec": _float_or_none(res.throughput),
                "rss_delta_mb": _float_or_none(res.rss_delta),
                "device_mem_delta_mb": _float_or_none(res.device_mem_delta),
            }
        data["throughput"] = throughput_data

    if monitor_b_report:
        data["monitor_b"] = dict(monitor_b_report)

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
    confusion_artifacts = _write_confusion_artifacts(report_path, ordered_task_metrics)

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
    report_lines.append(f"- Checkpoint: `{args.checkpoint}`")
    report_lines.append(f"- Model dim: {args.model_dim}")
    report_lines.append(f"- Channels: {_format_channels(args.channels)}")
    report_lines.append(f"- Chunk: {args.chunk}")
    report_lines.append(f"- Batch size: {args.batch_size}")
    report_lines.append(f"- Other threshold: {float(getattr(args, 'other_threshold', 0.0)):.4f}")
    report_lines.append(f"- Max samples per task: {'all' if args.max_samples <= 0 else args.max_samples}")
    report_lines.append(f"- Sample seed: {args.sample_seed}")
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
    aggregated_confusion_rel = confusion_artifacts.get("all_tasks")
    if aggregated_confusion_rel:
        report_lines.append(f"- Aggregated confusion matrix: [{aggregated_confusion_rel}]({aggregated_confusion_rel})")
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
        report_lines.append(f"- Files used: {int(monitor_b_report.get('files_used', 0))}")
        report_lines.append(f"- Skipped: {int(monitor_b_report.get('skipped', 0))}")
        report_lines.append(f"- Evaluated bytes: {int(monitor_b_report.get('evaluated_bytes', 0))}")
        report_lines.append("")
        monitor_rows = monitor_b_report.get("rows", [])
        if isinstance(monitor_rows, Sequence) and monitor_rows:
            report_lines.append(_render_training_style_metrics_table(monitor_rows))
            report_lines.append("")

    report_lines.append("## Task Details")
    report_lines.append("")
    for metrics in ordered_task_metrics:
        report_lines.append(f"### {metrics.name}")
        report_lines.append("")
        report_lines.append(metrics.description)
        report_lines.append("")
        extras = metrics.extras or {}
        confusion_rel = confusion_artifacts.get(metrics.name)
        if confusion_rel:
            report_lines.append(f"- Confusion matrix: [{confusion_rel}]({confusion_rel})")
        report_lines.append(f"- Samples: {metrics.samples}")
        report_lines.append(f"- Characters evaluated: {metrics.total_chars}")
        report_lines.append(f"- Overall accuracy: {metrics.overall_accuracy():.4f}")
        report_lines.append(f"- High confusions: {_summarize_confusions(metrics)}")
        name = metrics.name
        for support_line in _support_summary_lines(_task_support_payload(manifest, name)):
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
            by_lang = stats.get("by_lang", {})
            any_by_lang = stats.get("any_by_lang", {})
            if isinstance(by_lang, Mapping) and by_lang:
                report_lines.append("")
                report_lines.append("| Anchor label | Any non-wrapper ≥50% | Any non-wrapper coverage | Exact region ≥50% | Exact region coverage |")
                report_lines.append("| --- | --- | ---: | --- | ---: |")
                for lang, entry in sorted(by_lang.items()):
                    any_entry = any_by_lang.get(lang, {}) if isinstance(any_by_lang, Mapping) else {}
                    lang_total = int(entry.get("count", 0))
                    if lang_total <= 0:
                        continue
                    any_total_lang = int(any_entry.get("count", 0))
                    any_cov_lang = _format_pct(_safe_ratio(any_entry.get("correct_chars", 0), any_entry.get("truth_chars", 0)))
                    exact_cov_lang = _format_pct(_safe_ratio(entry.get("correct_chars", 0), entry.get("truth_chars", 0)))
                    report_lines.append(
                        f"| {lang} | {_format_hits(int(any_entry.get('detected', 0)), any_total_lang)} | {any_cov_lang} | "
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
            report_lines.append("")

        # Default handling for other tasks
        if name == "markdown_mix":
            markdown_stats = extras.get("markdown_segments", {})
            threshold = float(markdown_stats.get("threshold", MARKDOWN_IOU_THRESHOLD))
            wrapped_group = markdown_stats.get("overall", {}).get("wrapped", {})
            plain_group = markdown_stats.get("overall", {}).get("plain", {})
            text_like_binary = markdown_stats.get("text_like_binary", {})
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

        per_label_table = _render_metrics_table(metrics)
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
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.msgpack or Orbax directory).")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "evaluation" / "data"), help="Path to evaluation datasets.")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON (defaults to <data-root>/manifest.json).")
    parser.add_argument("--tasks", nargs="*", help="Subset of task names to evaluate.")
    parser.add_argument("--device", default="auto", help="Device for accuracy workloads (cpu/gpu/cuda/auto).")
    parser.add_argument("--cpu-device", default="cpu", help="Device name for throughput CPU benchmark.")
    parser.add_argument("--gpu-device", default="cuda", help="Device name for throughput GPU benchmark (ignored if unavailable).")
    parser.add_argument("--arch", default=None, choices=("unet1d", "mamba"), help="Model architecture (auto if omitted).")
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    mode_info = _configure_evaluation_mode(args)
    for message in mode_info.get("messages", []):
        print(message, flush=True)

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

        ckpt_path = Path(args.checkpoint).resolve()
        auto_hparams = _load_checkpoint_hparams(ckpt_path)

        label_names = auto_hparams.get("label_names")
        if label_names:
            _apply_label_mapping(label_names)

        arch = args.arch if args.arch is not None else auto_hparams.get("arch", None)
        arch = str(arch).lower().strip() if arch else "unet1d"
        args.arch = arch

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

        if auto_hparams:
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

        accuracy_runner = SegmenterRunner(
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
        runner_channels = getattr(accuracy_runner, "channels", None)
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
            metrics = evaluate_task(
                name,
                meta["description"],
                ds,
                accuracy_runner,
                min_run_chars=args.min_run,
                other_threshold=args.other_threshold,
            )
            elapsed = metrics.extras.get("elapsed_seconds", 0.0)
            print(
                f"    [{name}] completed in {elapsed:.1f}s • char_acc={metrics.overall_accuracy():.4f}",
                flush=True,
            )
            task_metrics.append(metrics)

        if bool(getattr(args, "fine_tuned_mode", False)):
            monitor_root = Path(getattr(args, "monitor_root", _default_monitor_b_root())).resolve()
            if not monitor_root.exists():
                raise RuntimeError(
                    f"Fine-tuned evaluation requires a full monitor_b root at {monitor_root}"
                )
            print(
                f"▶️  Evaluating full monitor_b from {monitor_root}",
                flush=True,
            )
            monitor_b_report = _evaluate_full_monitor_b(monitor_root, accuracy_runner)
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
            if gpu_devices:
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
        print(f"\n❌ Error during evaluation: {str(e)}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
