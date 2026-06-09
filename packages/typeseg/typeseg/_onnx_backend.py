"""Optional ONNX Runtime backend (GPU-capable, dynamic sequence length).

Used automatically when ``onnxruntime`` (or ``onnxruntime-gpu``) is installed and
the bundled ``*.onnx`` graphs are present; otherwise the pure-numpy backend runs.
Set ``TYPESEG_BACKEND=numpy`` to force the numpy path.

The graphs consume *compact* token ids (0..129); the compact remap is applied
here before the session runs. Math is verified bit-close to ``_numpy_backend``.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import numpy as np

from ._numpy_backend import _COMPACT_TABLE, NUM_TOKEN_EMBEDDINGS_LEGACY

try:  # Python 3.9+
    from importlib.resources import files as _files
except ImportError:  # pragma: no cover
    from importlib_resources import files as _files  # type: ignore


def _data(name: str):
    return _files("typeseg") / "data" / name


def _compact_ids(tokens: np.ndarray) -> np.ndarray:
    tok = np.clip(np.asarray(tokens, dtype=np.int64), 0, NUM_TOKEN_EMBEDDINGS_LEGACY - 1)
    return _COMPACT_TABLE[tok].astype(np.int64)


def _mode() -> str:
    """TYPESEG_BACKEND: '' / 'onnx' / 'cpu' (auto), 'numpy' (force numpy),
    'gpu' / 'cuda' (force CUDA, fail fast if it cannot initialise)."""
    return os.environ.get("TYPESEG_BACKEND", "").strip().lower()


def _require_gpu() -> bool:
    return _mode() in ("gpu", "cuda")


def _providers(model: str = "unet"):
    import onnxruntime as ort

    avail = set(ort.get_available_providers())
    if _require_gpu():
        if "CUDAExecutionProvider" not in avail:
            raise RuntimeError(
                f"TYPESEG_BACKEND={_mode()} requested but CUDAExecutionProvider is not "
                "available. Install the GPU backend: pip install \"typeseg[gpu]\"."
            )
        return ["CUDAExecutionProvider"]  # no CPU fallback -> session creation fails fast
    # Auto: U-Net (conv/matmul) benefits from CUDA, but Mamba's selective scan is
    # the ONNX `Scan` op, which is markedly *slower* on CUDA than on CPU (per-step
    # kernel launches). GPU acceleration for Mamba is provided by `_cupy_backend`
    # (a parallel-prefix scan) instead; the ONNX Mamba path always stays on CPU.
    if model == "mamba":
        return ["CPUExecutionProvider"]
    out = []
    if "CUDAExecutionProvider" in avail:
        out.append("CUDAExecutionProvider")
    out.append("CPUExecutionProvider")
    return out


def available() -> bool:
    """True if the ONNX backend should be used (onnxruntime imports + graphs bundled).

    With ``TYPESEG_BACKEND=gpu``/``cuda`` a missing onnxruntime or missing graphs is a
    hard error (fail fast) rather than a silent numpy fallback.
    """
    mode = _mode()
    if mode == "numpy":
        return False
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:
        if _require_gpu():
            raise RuntimeError(
                f"TYPESEG_BACKEND={mode} requires the ONNX GPU backend, but onnxruntime "
                "is not installed. Install with: pip install \"typeseg[gpu]\"."
            ) from exc
        return False
    try:
        ok = _data("unet_al.onnx").is_file() and _data("mamba_al.onnx").is_file()
    except Exception:
        ok = False
    if not ok and _require_gpu():
        raise RuntimeError(f"TYPESEG_BACKEND={mode} requested but the bundled ONNX graphs are missing.")
    return ok


def active_providers(name: str = "unet_al.onnx") -> list:
    """Providers a real session *actually* loaded for ``name``.

    ``onnxruntime.get_available_providers()`` lists everything compiled in, even a
    CUDA provider that fails to initialise (missing CUDA/cuDNN libs) and silently
    falls back to CPU. A loaded session's ``get_providers()`` reflects the truth.
    """
    if not available():
        return []
    try:
        return list(_session(name).get_providers())
    except Exception:
        return []


def using_gpu() -> bool:
    # Headline: whether the U-Net (the GPU-benefiting path) runs on CUDA.
    return "CUDAExecutionProvider" in active_providers("unet_al.onnx")


@lru_cache(maxsize=4)
def _session(name: str):
    import onnxruntime as ort

    so = ort.SessionOptions()
    with _data(name).open("rb") as fh:
        blob = fh.read()
    providers = _providers("mamba" if "mamba" in name else "unet")
    try:
        sess = ort.InferenceSession(blob, sess_options=so, providers=providers)
    except Exception as exc:
        if _require_gpu():
            raise RuntimeError(
                f"TYPESEG_BACKEND={_mode()}: the CUDA execution provider failed to "
                f"initialise ({exc}). Ensure CUDA 12.x + cuDNN 9.x are installed and on "
                "the library path (LD_LIBRARY_PATH)."
            ) from exc
        raise
    if _require_gpu() and "CUDAExecutionProvider" not in sess.get_providers():
        # Provider was requested but silently dropped to CPU at init.
        raise RuntimeError(
            f"TYPESEG_BACKEND={_mode()}: CUDA was requested but the session loaded only "
            f"{sess.get_providers()}. Check CUDA 12.x / cuDNN 9.x install and your GPU's "
            "compute-capability support in this onnxruntime build."
        )
    return sess


def unet_window_logits(token_windows: np.ndarray, batch: int = 64) -> np.ndarray:
    """token_windows: (N, 1536) raw byte ids -> logits (N, 1536, num_classes)."""
    sess = _session("unet_al.onnx")
    ids = _compact_ids(token_windows)
    outs = []
    for i in range(0, ids.shape[0], batch):
        outs.append(sess.run(None, {"ids": ids[i:i + batch]})[0])
    return np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]


def mamba_logits(tokens: np.ndarray) -> np.ndarray:
    """tokens: (T,) raw byte ids -> logits (T, num_classes)."""
    sess = _session("mamba_al.onnx")
    ids = _compact_ids(tokens)
    return sess.run(None, {"ids": ids})[0]
