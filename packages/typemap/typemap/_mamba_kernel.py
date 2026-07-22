"""Array-module-agnostic Mamba forward (numpy on CPU, cupy on GPU).

The math mirrors the JAX/Flax reference (``train/utils/model.py`` and
``inference/mamba_cuda.py``) exactly. Every function takes the array module
``xp`` (``numpy`` or ``cupy``) as its first argument, so the *same* source runs
on CPU and GPU with no behavioural drift -- the pure-numpy backend and the CuPy
GPU backend both call into here.

The only nontrivial op is the selective scan, a first-order linear recurrence
``s_t = a_t * s_{t-1} + b_t`` with ``a_t = exp(dt_t * A) in (0, 1]``. Two
implementations are provided:

* ``_selective_scan_seq`` -- the sequential per-timestep loop (CPU default).
* ``_selective_scan_parallel`` -- a chunked Hillis-Steele inclusive prefix scan
  (~log2(chunk) vectorised steps per chunk), which is what makes the GPU path
  fast: it replaces O(T) kernel launches with O(log T) large vectorised ops.
  The combine is identical to ``jax.lax.associative_scan`` in the reference, so
  the parallel and sequential results agree to float precision.

Weights are supplied as a mapping ``"Module/sub/param" -> xp.ndarray`` already in
the target module (the CuPy backend pushes them to the device once).
"""
from __future__ import annotations

import numpy as np

# Compact 257 -> 130 token remap (see train/utils/token_utils.COMPACT_TOKEN_TABLE).
NUM_TOKEN_EMBEDDINGS_LEGACY = 257


def _compact_token_table() -> np.ndarray:
    table = np.empty(NUM_TOKEN_EMBEDDINGS_LEGACY, dtype=np.int64)
    table[:128] = np.arange(128)
    table[128:256] = 128
    table[256] = 129
    return table


_COMPACT_TABLE = _compact_token_table()


# --------------------------------------------------------------------------
# Elementwise ops (match jax.nn / flax defaults)
# --------------------------------------------------------------------------
def _sigmoid(xp, x):
    return 1.0 / (1.0 + xp.exp(-x))


def _silu(xp, x):
    return x * _sigmoid(xp, x)


def _softplus(xp, x):
    # numerically stable log(1 + exp(x)) == logaddexp(0, x)
    return xp.logaddexp(0.0, x)


def _layernorm(xp, x, scale, bias, eps: float = 1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / xp.sqrt(var + eps) * scale + bias


def _depthwise_conv1d_same(xp, x, kernel, bias):
    # x: (L, C); kernel: (k, 1, C) depthwise (feature_group_count=C). SAME padding.
    k, _one, c = kernel.shape
    total = k - 1
    low = total // 2
    high = total - low
    xp_pad = xp.pad(x, ((low, high), (0, 0)))
    L = x.shape[0]
    out = xp.zeros((L, c), dtype=xp.float32)
    for j in range(k):
        out = out + xp_pad[j:j + L] * kernel[j, 0, :]
    return out + bias


# --------------------------------------------------------------------------
# Embedding (with compact remap)
# --------------------------------------------------------------------------
def _embed(xp, w, tokens):
    table = w["Embed_0/embedding"]  # (vocab, d), xp array
    tok = xp.asarray(tokens).astype(xp.int64)
    if int(table.shape[0]) != NUM_TOKEN_EMBEDDINGS_LEGACY:
        compact = xp.asarray(_COMPACT_TABLE)
        tok = compact[xp.clip(tok, 0, NUM_TOKEN_EMBEDDINGS_LEGACY - 1)]
    return table[tok]


# --------------------------------------------------------------------------
# Selective scan
# --------------------------------------------------------------------------
def _selective_scan_seq(xp, u, dt, B, C, A, D):
    """Sequential recurrence. u,dt: (L, d_inner); B,C: (L, d_state);
    A: (d_inner, d_state); D: (d_inner,). Returns y: (L, d_inner)."""
    L, d_inner = u.shape
    s = xp.zeros((d_inner, A.shape[1]), dtype=xp.float32)
    y = xp.empty((L, d_inner), dtype=xp.float32)
    for t in range(L):
        dt_t = dt[t][:, None]                       # (d_inner, 1)
        a_t = xp.exp(dt_t * A)                       # (d_inner, d_state)
        b_t = u[t][:, None] * (dt_t * B[t][None, :])
        s = a_t * s + b_t
        y[t] = (s * C[t][None, :]).sum(axis=1) + u[t] * D
    return y


def _selective_scan_parallel(xp, u, dt, B, C, A, D, chunk: int = 4096):
    """Chunked Hillis-Steele inclusive prefix scan (parallel over time).

    Identical result to ``_selective_scan_seq`` (combine matches the reference
    ``jax.lax.associative_scan``), but built from O(log2(chunk)) vectorised steps
    per chunk instead of an O(L) Python loop. The sequence is processed in chunks
    of ``chunk`` carrying the final state across chunk boundaries, which bounds
    memory and supports arbitrary length. Stable in float32: all ``a <= 1`` keeps
    the running product in (0, 1] and the state bounded.
    """
    L, d_inner = u.shape
    d_state = A.shape[1]
    if L == 0:
        return xp.empty((0, d_inner), dtype=xp.float32)

    A_bc = A[None, :, :]                              # (1, d_inner, d_state)
    y = xp.empty((L, d_inner), dtype=xp.float32)
    carry = xp.zeros((d_inner, d_state), dtype=xp.float32)

    for c0 in range(0, L, chunk):
        c1 = min(c0 + chunk, L)
        u_c = u[c0:c1]; dt_c = dt[c0:c1]
        B_c = B[c0:c1]; C_c = C[c0:c1]
        Lc = c1 - c0

        dtc = dt_c[:, :, None]                        # (Lc, d_inner, 1)
        a = xp.exp(dtc * A_bc)                         # (Lc, d_inner, d_state), in (0,1]
        b = u_c[:, :, None] * (dtc * B_c[:, None, :])  # (Lc, d_inner, d_state)

        # Inclusive Hillis-Steele scan over the chunk (axis 0):
        #   combine(left, right) = (a2*a1, b2 + a2*b1)
        d = 1
        while d < Lc:
            a_prev = xp.concatenate(
                [xp.ones((d, d_inner, d_state), dtype=xp.float32), a[:Lc - d]], axis=0)
            b_prev = xp.concatenate(
                [xp.zeros((d, d_inner, d_state), dtype=xp.float32), b[:Lc - d]], axis=0)
            b = b + a * b_prev
            a = a * a_prev
            d <<= 1
        # a[i] = prod_{j<=i} a_j (decay from chunk start); b[i] = state with zero entering state.
        s = b + a * carry[None, :, :]                 # fold in the carried state
        y[c0:c1] = (s * C_c[:, None, :]).sum(axis=-1) + u_c * D
        carry = s[-1]
    return y


# --------------------------------------------------------------------------
# Mamba block + forward
# --------------------------------------------------------------------------
def _resolve_scan(parallel: bool, scan):
    if scan is not None:
        return scan
    if parallel:
        return lambda xp, u, dt, B, C, A, D: _selective_scan_parallel(xp, u, dt, B, C, A, D)
    return _selective_scan_seq


def _mamba_block(xp, w, idx: int, x, d_state: int, dt_rank: int, d_conv: int, scan):
    p = f"CheckpointMambaBlock1D_{idx}/"
    h = _layernorm(xp, x, w[p + "LayerNorm_0/scale"], w[p + "LayerNorm_0/bias"], eps=1e-6)
    xz = h @ w[p + "Dense_0/kernel"] + w[p + "Dense_0/bias"]   # (L, 2*d_inner)
    d_inner = xz.shape[1] // 2
    u, gate = xz[:, :d_inner], xz[:, d_inner:]
    u = _depthwise_conv1d_same(xp, u, w[p + "Conv_0/kernel"], w[p + "Conv_0/bias"])
    u = _silu(xp, u)
    x_dbl = u @ w[p + "Dense_1/kernel"] + w[p + "Dense_1/bias"]  # (L, dt_rank+2*d_state)
    dt_raw = x_dbl[:, :dt_rank]
    B = x_dbl[:, dt_rank:dt_rank + d_state]
    C = x_dbl[:, dt_rank + d_state:dt_rank + 2 * d_state]
    dt = dt_raw @ w[p + "Dense_2/kernel"] + w[p + "Dense_2/bias"]  # (L, d_inner)
    dt = _softplus(xp, dt) + 1e-4
    A = -xp.exp(w[p + "A_log"])           # (d_inner, d_state)
    D = w[p + "D"]                        # (d_inner,)

    # bidirectional: forward scan + reverse pass on reversed inputs, then reverse output
    y = scan(xp, u, dt, B, C, A, D)
    y_rev = scan(xp, u[::-1], dt[::-1], B[::-1], C[::-1], A, D)[::-1]
    y = y + y_rev
    y = y * _silu(xp, gate)
    y = y @ w[p + "Dense_3/kernel"] + w[p + "Dense_3/bias"]   # (L, d_model)
    return x + y


def mamba_forward(xp, w, tokens, n_layers: int = 6, d_state: int = 16,
                  dt_rank: int = 16, d_conv: int = 4, parallel: bool = False,
                  chunk: int = 4096, scan=None):
    """tokens: (L,) raw byte ids -> logits (L, num_classes).

    ``scan(xp, u, dt, B, C, A, D) -> y`` is the single-direction selective scan;
    when omitted, the sequential loop (``parallel=False``) or the chunked
    parallel scan (``parallel=True``) is used. The CuPy backend injects a custom
    RawKernel scan here.
    """
    scan = _resolve_scan(parallel, scan)
    h = _embed(xp, w, tokens).astype(xp.float32)
    for i in range(n_layers):
        h = _mamba_block(xp, w, i, h, d_state=d_state, dt_rank=dt_rank, d_conv=d_conv, scan=scan)
    h = _layernorm(xp, h, w["LayerNorm_0/scale"], w["LayerNorm_0/bias"], eps=1e-6)
    logits = h @ w["Dense_0/kernel"] + w["Dense_0/bias"]
    return logits
