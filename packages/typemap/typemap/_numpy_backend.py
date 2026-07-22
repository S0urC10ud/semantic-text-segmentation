"""Pure-numpy inference for the slimmed U-Net and Mamba segmenters.

The math mirrors the JAX/Flax reference (``train/utils/model.py`` and
``inference/mamba_cuda.py``) exactly: the only dependency is numpy, so the
published package carries no JAX/Flax/TF runtime. Weights are supplied as a flat
mapping ``"Module/sub/param" -> np.ndarray`` (see ``_load_npz``).
"""
from __future__ import annotations

from typing import Dict, Mapping

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
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid(x)


def _softplus(x: np.ndarray) -> np.ndarray:
    # numerically stable log(1+exp(x)) == logaddexp(0, x)
    return np.logaddexp(0.0, x)


def _gelu(x: np.ndarray) -> np.ndarray:
    # jax.nn.gelu default: tanh approximation
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * scale + bias


def _groupnorm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray, groups: int = 8, eps: float = 1e-5) -> np.ndarray:
    # x: (L, C). Flax GroupNorm reduces over (spatial, channels-in-group) per group.
    L, C = x.shape
    g = groups
    xg = x.reshape(L, g, C // g)
    mean = xg.mean(axis=(0, 2), keepdims=True)
    var = xg.var(axis=(0, 2), keepdims=True)
    xg = (xg - mean) / np.sqrt(var + eps)
    return xg.reshape(L, C) * scale + bias


def _same_pad(k: int) -> tuple:
    total = k - 1
    low = total // 2
    return low, total - low


def _conv1d_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # x: (L, Cin); kernel: (k, Cin, Cout) ; full (dense) convolution, SAME padding.
    k, cin, cout = kernel.shape
    low, high = _same_pad(k)
    xp = np.pad(x, ((low, high), (0, 0)))
    L = x.shape[0]
    # im2col -> (L, k*Cin) @ (k*Cin, Cout)
    cols = np.empty((L, k, cin), dtype=np.float32)
    for j in range(k):
        cols[:, j, :] = xp[j:j + L]
    out = cols.reshape(L, k * cin) @ kernel.reshape(k * cin, cout)
    return out + bias


def _depthwise_conv1d_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # x: (L, C); kernel: (k, 1, C) depthwise (feature_group_count=C). SAME padding.
    k, _one, c = kernel.shape
    low, high = _same_pad(k)
    xp = np.pad(x, ((low, high), (0, 0)))
    L = x.shape[0]
    out = np.zeros((L, c), dtype=np.float32)
    for j in range(k):
        out += xp[j:j + L] * kernel[j, 0, :]
    return out + bias


# --------------------------------------------------------------------------
# Weight access
# --------------------------------------------------------------------------
class Weights:
    """Flat ``path -> ndarray`` view with ``w["A/B/c"]`` access."""

    def __init__(self, flat: Mapping[str, np.ndarray]):
        self._d = {k: np.asarray(v, dtype=np.float32) for k, v in flat.items()}

    def __getitem__(self, key: str) -> np.ndarray:
        return self._d[key]

    def __contains__(self, key: str) -> bool:
        return key in self._d


def flatten_params(tree: Mapping, prefix: str = "") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in tree.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten_params(v, key + "/"))
        else:
            out[key] = np.asarray(v, dtype=np.float32)
    return out


# --------------------------------------------------------------------------
# Embedding (with compact remap)
# --------------------------------------------------------------------------
def _embed(w: Weights, tokens: np.ndarray) -> np.ndarray:
    table = w["Embed_0/embedding"]  # (vocab, d)
    tok = np.asarray(tokens, dtype=np.int64)
    if table.shape[0] != NUM_TOKEN_EMBEDDINGS_LEGACY:
        tok = _COMPACT_TABLE[np.clip(tok, 0, NUM_TOKEN_EMBEDDINGS_LEGACY - 1)]
    return table[tok]


# --------------------------------------------------------------------------
# U-Net
# --------------------------------------------------------------------------
def unet_forward(w: Weights, tokens: np.ndarray, channels=(32, 64, 64, 128, 128, 128, 128, 256)) -> np.ndarray:
    """tokens: (L,) int -> logits (L, num_classes)."""
    h = _embed(w, tokens).astype(np.float32)  # (L, emb)
    cb = 0

    def convblock(idx: int, x: np.ndarray) -> np.ndarray:
        p = f"ConvBlock1D_{idx}/"
        x = _conv1d_same(x, w[p + "Conv_0/kernel"], w[p + "Conv_0/bias"])
        x = _groupnorm(x, w[p + "GroupNorm_0/scale"], w[p + "GroupNorm_0/bias"], groups=8, eps=1e-5)
        return _gelu(x)

    skips = []
    for i, _ch in enumerate(channels):
        h = convblock(cb, h); cb += 1
        h = convblock(cb, h); cb += 1
        skips.append(h)
        if i < len(channels) - 1:
            # max pool size 2 stride 2
            L = (h.shape[0] // 2) * 2
            h = h[:L].reshape(L // 2, 2, h.shape[1]).max(axis=1)

    for i, _ch in enumerate(reversed(channels[:-1])):
        h = np.repeat(h, 2, axis=0)  # nearest-neighbour upsample x2
        skip = skips[-(i + 2)]
        if h.shape[0] != skip.shape[0]:
            if h.shape[0] < skip.shape[0]:
                h = np.pad(h, ((0, skip.shape[0] - h.shape[0]), (0, 0)))
            else:
                h = h[: skip.shape[0]]
        h = np.concatenate([h, skip], axis=-1)
        h = convblock(cb, h); cb += 1
        h = convblock(cb, h); cb += 1

    logits = _conv1d_same(h, w["Conv_0/kernel"], w["Conv_0/bias"])  # (L, num_classes)
    return logits


# --------------------------------------------------------------------------
# Mamba (delegates to the shared array-module-agnostic kernel; xp=numpy)
# --------------------------------------------------------------------------
def mamba_forward(w: Weights, tokens: np.ndarray, n_layers: int = 6,
                  d_state: int = 16, dt_rank: int = 16, d_conv: int = 4) -> np.ndarray:
    """tokens: (L,) int -> logits (L, num_classes). CPU path: sequential scan."""
    from . import _mamba_kernel as mk

    return mk.mamba_forward(np, w, tokens, n_layers=n_layers, d_state=d_state,
                            dt_rank=dt_rank, d_conv=d_conv, parallel=False)
