"""Optional CuPy GPU backend for the Mamba (``precise``) model.

The Mamba selective-scan is a parallel-prefix scan; on GPU it is dramatically
faster run as ~log2(T) large vectorised steps than as the ONNX ``Scan`` op's
per-timestep recurrence (which is launch-bound and actually slower on GPU than
CPU). This backend runs the shared ``_mamba_kernel`` with ``xp=cupy`` and the
parallel scan, loading the bundled ``mamba_al.npz`` weights onto the device once.

Used automatically when ``cupy`` imports and a CUDA device is present; otherwise
the ONNX (CPU) or pure-numpy backend handles ``precise()``. ``TYPESEG_BACKEND``
follows the same contract as ``_onnx_backend``: ``numpy`` forces it off,
``gpu``/``cuda`` force it on and fail fast if CuPy or a device is missing.
"""
from __future__ import annotations

import json
import warnings
from functools import lru_cache
from typing import Optional

import numpy as np

from ._mamba_kernel import mamba_forward as _kernel_forward
from ._onnx_backend import _mode, _require_gpu

try:  # Python 3.9+
    from importlib.resources import files as _files
except ImportError:  # pragma: no cover
    from importlib_resources import files as _files  # type: ignore


def _data(name: str):
    return _files("typeseg") / "data" / name


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads(_data("manifest.json").read_text())


def _import_cupy():
    import cupy as cp  # may raise ImportError / CUDA init errors
    return cp


# CUDA kernel for the bidirectional selective scan. One thread per inner channel
# `d`; each thread carries its own `d_state`-vector state and sweeps the sequence
# once (O(T) work, a single kernel launch). This reads every element of the big
# (T, d_inner, d_state) state exactly once -- vastly less memory traffic than a
# log-step parallel scan, and no per-timestep kernel launches. Math is identical
# to ``_mamba_kernel._selective_scan_seq``:
#   a = exp(dt*A); s = a*s + u*(dt*B); y = sum_n s_n*C_n + u*D
_SCAN_SRC = r"""
extern "C" __global__
void selective_scan(const float* __restrict__ u,
                    const float* __restrict__ dt,
                    const float* __restrict__ B,
                    const float* __restrict__ C,
                    const float* __restrict__ A,
                    const float* __restrict__ D,
                    float* __restrict__ y,
                    const int T, const int di, const int ds) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;   // inner channel
    if (d >= di) return;
    float s[64];                                     // d_state <= 64
    for (int n = 0; n < ds; ++n) s[n] = 0.0f;
    const float Dd = D[d];
    const float* Arow = A + d * ds;
    for (int t = 0; t < T; ++t) {
        const float dtt = dt[t * di + d];
        const float ut  = u[t * di + d];
        const float* Brow = B + t * ds;
        const float* Crow = C + t * ds;
        float yt = ut * Dd;
        for (int n = 0; n < ds; ++n) {
            float sn = __expf(dtt * Arow[n]) * s[n] + ut * (dtt * Brow[n]);
            s[n] = sn;
            yt += sn * Crow[n];
        }
        y[t * di + d] = yt;
    }
}
"""


@lru_cache(maxsize=1)
def _scan_kernel():
    cp = _import_cupy()
    return cp.RawKernel(_SCAN_SRC, "selective_scan")


def _cupy_scan(xp, u, dt, B, C, A, D):
    """Single-direction selective scan on the GPU via the RawKernel.

    Reversed inputs (for the backward pass) arrive as non-contiguous views; we
    make them contiguous -- those are only (T, d_inner)/(T, d_state) copies, cheap
    next to the scan itself.
    """
    cp = xp
    u = cp.ascontiguousarray(u, dtype=cp.float32)
    dt = cp.ascontiguousarray(dt, dtype=cp.float32)
    B = cp.ascontiguousarray(B, dtype=cp.float32)
    C = cp.ascontiguousarray(C, dtype=cp.float32)
    A = cp.ascontiguousarray(A, dtype=cp.float32)
    D = cp.ascontiguousarray(D, dtype=cp.float32)
    T, di = int(u.shape[0]), int(u.shape[1])
    ds = int(A.shape[1])
    if ds > 64:
        raise ValueError(f"d_state={ds} exceeds the kernel's 64-state register budget")
    y = cp.empty((T, di), dtype=cp.float32)
    if T == 0:
        return y
    threads = 128
    blocks = (di + threads - 1) // threads
    _scan_kernel()((blocks,), (threads,),
                   (u, dt, B, C, A, D, y, np.int32(T), np.int32(di), np.int32(ds)))
    return y


def _has_device() -> bool:
    cp = _import_cupy()
    return int(cp.cuda.runtime.getDeviceCount()) > 0


@lru_cache(maxsize=1)
def _probe_compile():
    """Confirm CuPy can JIT-compile a kernel. Returns ``(ok, error_or_None)``.

    CuPy compiles every elementwise/raw kernel at runtime via nvrtc, which needs the
    CUDA toolkit headers. The pip ``cupy-cuda12x`` wheel ships nvrtc but NOT those
    headers, so on a machine without a system CUDA toolkit (or the ``[ctk]`` header
    wheels) compilation raises at first use -- e.g. ``RuntimeError: Failed to find
    CUDA headers``. A bare device check passes there, so without this probe the auto
    router picks CuPy and then crashes mid-inference instead of falling back. The
    ``astype`` forces a real nvrtc compile; the result is cached so a broken box pays
    it once. Never raises -- callers branch on the returned flag.
    """
    try:
        cp = _import_cupy()
        cp.arange(4, dtype=cp.int32).astype(cp.float32).sum().item()  # forces nvrtc compile
        return True, None
    except Exception as exc:  # nvrtc/header/driver init failure
        return False, exc


_warned_compile_fail = False


def _warn_compile_fail_once(exc: Exception) -> None:
    global _warned_compile_fail
    if _warned_compile_fail:
        return
    _warned_compile_fail = True
    warnings.warn(
        "typeseg: a CUDA device was found but CuPy could not compile its GPU kernels "
        f"({type(exc).__name__}: {exc}); falling back to the CPU (ONNX) backend for "
        "precise(). CuPy JIT-compiles kernels and needs the CUDA toolkit headers -- "
        "install them with: pip install \"cupy-cuda12x[ctk]\" (or set the CUDA_PATH "
        "environment variable to a system CUDA 12.x install). Silence with "
        "TYPESEG_BACKEND=numpy or Python's warnings filters.",
        RuntimeWarning,
        stacklevel=3,
    )


def available() -> bool:
    """True if the CuPy GPU Mamba path should be used.

    Auto mode: True when cupy imports, a CUDA device is present, AND CuPy can
    actually compile a kernel (headers available). Any of those failing falls back
    to ONNX/numpy (a one-time warning if a device was present but kernels won't
    compile). With ``TYPESEG_BACKEND=gpu``/``cuda`` any failure is a hard error;
    with ``TYPESEG_BACKEND=numpy`` this is always off.
    """
    mode = _mode()
    if mode == "numpy":
        return False
    stage = "device"
    try:
        if not _has_device():
            raise RuntimeError("no CUDA device visible to CuPy")
        if not _data("mamba_al.npz").is_file():
            raise RuntimeError("bundled Mamba weights are missing")
        stage = "compile"
        ok, perr = _probe_compile()
        if not ok:
            raise perr if perr is not None else RuntimeError("CuPy kernel compilation failed")
    except Exception as exc:
        if _require_gpu():
            raise RuntimeError(
                f"TYPESEG_BACKEND={mode} requires the CuPy GPU backend, but it could not "
                f"initialise ({exc}). Install with: pip install \"typeseg[gpu]\" (which bundles "
                "the CUDA toolkit headers CuPy needs to JIT its kernels); with a system CUDA "
                "install, set CUDA_PATH."
            ) from exc
        if stage == "compile":  # device present but headers missing -> the actionable case
            _warn_compile_fail_once(exc)
        return False
    return True


@lru_cache(maxsize=1)
def _weights_gpu():
    """Load the slimmed Mamba weights and push them onto the GPU once."""
    cp = _import_cupy()
    with _data(_manifest()["mamba"]["file"]).open("rb") as fh:
        data = np.load(fh)
        flat = {k.replace("__", "/"): cp.asarray(np.asarray(data[k], dtype=np.float32))
                for k in data.files}
    return flat


def device_name() -> str:
    try:
        cp = _import_cupy()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"]
        return name.decode() if isinstance(name, (bytes, bytearray)) else str(name)
    except Exception:
        return "cuda"


def active_providers() -> list:
    if not available():
        return []
    return [f"CuPyCUDA:{device_name()}"]


def mamba_logits(tokens: np.ndarray) -> np.ndarray:
    """tokens: (T,) raw byte ids -> logits (T, num_classes) as a host ndarray.

    Runs the parallel selective-scan on the GPU. Raw (non-compacted) tokens: the
    kernel applies the compact remap internally, matching the numpy path exactly.
    """
    cp = _import_cupy()
    cfg = _manifest()["mamba"]
    w = _weights_gpu()
    tok = cp.asarray(np.asarray(tokens, dtype=np.int64))
    logits = _kernel_forward(
        cp, w, tok,
        n_layers=cfg["n_layers"], d_state=cfg["d_state"],
        dt_rank=cfg["dt_rank"], d_conv=cfg["d_conv"],
        scan=_cupy_scan,
    )
    return cp.asnumpy(logits).astype(np.float32)
