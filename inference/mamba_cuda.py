from __future__ import annotations

import functools
from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp

try:  # pragma: no cover - availability depends on local JAX build
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
    _PALLAS_OK = True
except Exception:  # pragma: no cover - availability depends on local JAX build
    pl = None  # type: ignore[assignment]
    plgpu = None  # type: ignore[assignment]
    _PALLAS_OK = False


def has_cuda_mamba_kernel() -> bool:
    if not _PALLAS_OK:
        return False
    try:
        return any(dev.platform == "gpu" for dev in jax.devices("gpu"))
    except Exception:
        return False


def _pick_block_channels(d_inner: int) -> int:
    if d_inner <= 32:
        return 32
    if d_inner <= 64:
        return 64
    return 64


def _pick_num_warps(block_channels: int) -> int:
    if block_channels <= 32:
        return 4
    return 8


def _selective_scan_associative(
    x_in: jnp.ndarray,
    dt_in: jnp.ndarray,
    B_in: jnp.ndarray,
    C_in: jnp.ndarray,
    A: jnp.ndarray,
    D: jnp.ndarray,
) -> jnp.ndarray:
    x_f32 = x_in.astype(jnp.float32)
    dt_f32 = dt_in.astype(jnp.float32)
    B_f32 = B_in.astype(jnp.float32)
    C_f32 = C_in.astype(jnp.float32)

    x_tm = jnp.swapaxes(x_f32, 0, 1)
    dt_tm = jnp.swapaxes(dt_f32, 0, 1)
    B_tm = jnp.swapaxes(B_f32, 0, 1)
    C_tm = jnp.swapaxes(C_f32, 0, 1)

    a = jnp.exp(dt_tm[:, :, :, None] * A[None, None, :, :])
    b = x_tm[:, :, :, None] * (dt_tm[:, :, :, None] * B_tm[:, :, None, :])

    def combine(left, right):
        a1, b1 = left
        a2, b2 = right
        return a2 * a1, b2 + a2 * b1

    _, state_tm = jax.lax.associative_scan(combine, (a, b), axis=0)
    y_tm = jnp.sum(state_tm * C_tm[:, :, None, :], axis=-1) + x_tm * D[None, None, :]
    return jnp.swapaxes(y_tm, 0, 1)


def _selective_scan_kernel(
    x_ref,
    dt_ref,
    B_ref,
    C_ref,
    A_ref,
    D_ref,
    y_ref,
    *,
    block_channels: int,
    d_state: int,
):
    batch_idx = pl.program_id(0)
    channel_block = pl.program_id(1)
    chan_idx = channel_block * block_channels + jnp.arange(block_channels)
    chan_mask = chan_idx < x_ref.shape[2]
    state_idx = jnp.arange(d_state)

    A_block = plgpu.load(
        A_ref.at[chan_idx[:, None], state_idx[None, :]],
        mask=chan_mask[:, None],
        other=0.0,
    ).astype(jnp.float32)
    D_block = plgpu.load(
        D_ref.at[chan_idx],
        mask=chan_mask,
        other=0.0,
    ).astype(jnp.float32)

    def body(t: int, state: jnp.ndarray) -> jnp.ndarray:
        x_t = plgpu.load(
            x_ref.at[batch_idx, t, chan_idx],
            mask=chan_mask,
            other=0.0,
        ).astype(jnp.float32)
        dt_t = plgpu.load(
            dt_ref.at[batch_idx, t, chan_idx],
            mask=chan_mask,
            other=0.0,
        ).astype(jnp.float32)
        B_t = plgpu.load(
            B_ref.at[batch_idx, t, state_idx],
            mask=state_idx < B_ref.shape[2],
            other=0.0,
        ).astype(jnp.float32)
        C_t = plgpu.load(
            C_ref.at[batch_idx, t, state_idx],
            mask=state_idx < C_ref.shape[2],
            other=0.0,
        ).astype(jnp.float32)
        a_t = jnp.exp(dt_t[:, None] * A_block)
        state = a_t * state + x_t[:, None] * (dt_t[:, None] * B_t[None, :])
        y_t = jnp.sum(state * C_t[None, :], axis=-1) + x_t * D_block
        plgpu.store(
            y_ref.at[batch_idx, t, chan_idx],
            y_t.astype(y_ref.dtype),
            mask=chan_mask,
        )
        return state

    init_state = jnp.zeros((block_channels, d_state), dtype=jnp.float32)
    lax.fori_loop(0, x_ref.shape[1], body, init_state)


def selective_scan_cuda(
    x_in: jnp.ndarray,
    dt_in: jnp.ndarray,
    B_in: jnp.ndarray,
    C_in: jnp.ndarray,
    A: jnp.ndarray,
    D: jnp.ndarray,
    *,
    block_channels: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: int = 1,
    interpret: bool = False,
) -> jnp.ndarray:
    if not has_cuda_mamba_kernel():
        raise RuntimeError("CUDA Mamba kernel requested but Pallas/Triton GPU support is unavailable.")
    if x_in.ndim != 3 or dt_in.ndim != 3 or B_in.ndim != 3 or C_in.ndim != 3:
        raise ValueError("selective_scan_cuda expects rank-3 inputs.")
    if x_in.shape != dt_in.shape:
        raise ValueError(f"x and dt must share shape, got {x_in.shape} vs {dt_in.shape}")
    if B_in.shape[:2] != x_in.shape[:2] or C_in.shape[:2] != x_in.shape[:2]:
        raise ValueError("B/C must share the batch/sequence dimensions of x.")
    if A.ndim != 2 or D.ndim != 1:
        raise ValueError("A must be rank-2 and D must be rank-1.")

    d_inner = int(x_in.shape[-1])
    d_state = int(B_in.shape[-1])
    if int(A.shape[0]) != d_inner or int(A.shape[1]) != d_state:
        raise ValueError(f"A shape {A.shape} is incompatible with d_inner={d_inner}, d_state={d_state}")
    if int(D.shape[0]) != d_inner:
        raise ValueError(f"D shape {D.shape} is incompatible with d_inner={d_inner}")

    block_channels = int(block_channels or _pick_block_channels(d_inner))
    num_warps = int(num_warps or _pick_num_warps(block_channels))
    out_shape = jax.ShapeDtypeStruct(shape=x_in.shape, dtype=jnp.float32)
    kernel = functools.partial(
        _selective_scan_kernel,
        block_channels=block_channels,
        d_state=d_state,
    )
    fn = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(int(x_in.shape[0]), pl.cdiv(d_inner, block_channels)),
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps,
            num_stages=int(num_stages),
        ),
        debug=False,
        interpret=interpret,
        name="mamba_selective_scan_cuda",
    )
    return fn(
        x_in.astype(jnp.float32),
        dt_in.astype(jnp.float32),
        B_in.astype(jnp.float32),
        C_in.astype(jnp.float32),
        A.astype(jnp.float32),
        D.astype(jnp.float32),
    )


def selective_scan_inference(
    x_in: jnp.ndarray,
    dt_in: jnp.ndarray,
    B_in: jnp.ndarray,
    C_in: jnp.ndarray,
    A: jnp.ndarray,
    D: jnp.ndarray,
    *,
    backend: str = "default",
) -> jnp.ndarray:
    mode = str(backend).lower().strip()
    if mode == "cuda_fast":
        return selective_scan_cuda(x_in, dt_in, B_in, C_in, A, D)
    return _selective_scan_associative(x_in, dt_in, B_in, C_in, A, D)
