"""
The 1D U-Net model architecture and training-related utilities like the
TrainState, loss functions, and train/eval steps.
"""
from typing import TYPE_CHECKING, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import utils.config as cfg
from flax import linen as nn
from flax.training import train_state

if TYPE_CHECKING:
    from utils.config import TrainConfig

# ---------------------------
# Model: 1D U-Net
# ---------------------------

def decay_mask(params):
    def _mask(tree, path=()):
        if isinstance(tree, dict):
            return {k: _mask(v, path + (k,)) for k, v in tree.items()}
        name = "/".join(path)
        # No decay for layer norms, biases, or embeddings
        if (
            "GroupNorm" in name
            or "LayerNorm" in name
            or name.endswith("bias")
            or "Embed_0/embedding" in name
        ):
            return False
        return True
    return _mask(params)

class ConvBlock1D(nn.Module):
    features: int
    kernel_size: int = 3
    groups: int = 8
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, train: bool):
        h = nn.Conv(self.features, (self.kernel_size,), padding="SAME",
                    dtype=self.dtype, param_dtype=jnp.float32)(x)
        h = nn.GroupNorm(num_groups=self.groups, epsilon=1e-5)(h)
        h = nn.gelu(h)
        if self.dropout_rate > 0.:
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)
        return h

def upsample_nn_1d(x, factor: int):
    return jnp.repeat(x, repeats=factor, axis=1)

class UNet1D(nn.Module):
    num_classes: int = cfg.NUM_CLASSES
    emb_dim: int = 128
    channels: Tuple[int, ...] = (128, 256, 384, 512)
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = True):
        h = nn.Embed(num_embeddings=cfg.NUM_TOKEN_EMBEDDINGS, features=self.emb_dim,
                     embedding_init=nn.initializers.normal(stddev=0.02),
                     dtype=self.dtype, param_dtype=jnp.float32)(tokens)

        skips = []
        # Down-sampling path
        for i, ch in enumerate(self.channels):
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)
            skips.append(h)
            if i < len(self.channels) - 1:
                h = nn.max_pool(h, (2,), strides=(2,))

        # Up-sampling path
        for i, ch in enumerate(reversed(self.channels[:-1])):
            h = upsample_nn_1d(h, factor=2)
            skip = skips[-(i + 2)]
            if h.shape[1] != skip.shape[1]:
                h = jnp.pad(h, ((0, 0), (0, skip.shape[1] - h.shape[1]), (0, 0)))
            h = jnp.concatenate([h, skip], axis=-1)
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)
            h = ConvBlock1D(ch, dropout_rate=self.dropout_rate, dtype=self.dtype)(h, train)

        logits_bf16 = nn.Conv(self.num_classes, (1,), padding="SAME",
                              dtype=self.dtype, param_dtype=jnp.float32)(h)
        return logits_bf16.astype(jnp.float32)

# ---------------------------
# Model: Mamba (simple Flax port)
# ---------------------------

class MambaBlock1D(nn.Module):
    """A small, self-contained Mamba-style block for sequence mixing.

    Notes:
      - This is intentionally minimal and pure-JAX/Flax (no custom kernels).
      - The SSM scan uses `lax.scan` (sequential). It is meant for quick
        architecture comparisons, not peak throughput.
      - Set bidirectional=True to make it usable for per-position segmentation.
    """

    d_model: int
    d_state: int = 8
    expand: int = 1
    dt_rank: int = 16
    d_conv: int = 4
    dropout_rate: float = 0.0
    bidirectional: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        # Pre-norm
        h = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(x)

        d_inner = int(self.d_model) * int(self.expand)

        # Input projection: (x, gate)
        xz = nn.Dense(
            2 * d_inner,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            use_bias=True,
        )(h)
        u, gate = jnp.split(xz, 2, axis=-1)

        # Local mixing (depthwise conv)
        if self.d_conv and int(self.d_conv) > 1:
            u = nn.Conv(
                features=d_inner,
                kernel_size=(int(self.d_conv),),
                padding="SAME",
                feature_group_count=d_inner,
                dtype=self.dtype,
                param_dtype=jnp.float32,
                use_bias=True,
            )(u)
        u = jax.nn.silu(u)

        # Token-dependent SSM parameters
        dt_rank = int(self.dt_rank) if int(self.dt_rank) > 0 else max(4, d_inner // 16)
        x_dbl = nn.Dense(
            dt_rank + 2 * int(self.d_state),
            dtype=self.dtype,
            param_dtype=jnp.float32,
            use_bias=True,
        )(u)
        dt_raw, B, C = jnp.split(
            x_dbl, [dt_rank, dt_rank + int(self.d_state)], axis=-1
        )
        dt = nn.Dense(
            d_inner,
            dtype=self.dtype,
            param_dtype=jnp.float32,
            use_bias=True,
        )(dt_raw)
        dt = jax.nn.softplus(dt).astype(jnp.float32) + 1e-4

        # Learned continuous-time parameters (diagonal A per channel)
        A_log = self.param(
            "A_log",
            nn.initializers.normal(stddev=0.02),
            (d_inner, int(self.d_state)),
            jnp.float32,
        )
        A = -jnp.exp(A_log)  # (d_inner, d_state), negative for stability
        D = self.param("D", nn.initializers.ones, (d_inner,), jnp.float32)

        def selective_scan(
            x_in: jnp.ndarray,
            dt_in: jnp.ndarray,
            B_in: jnp.ndarray,
            C_in: jnp.ndarray,
        ) -> jnp.ndarray:
            # x_in: (B, L, d_inner)
            # dt_in: (B, L, d_inner)
            # B_in/C_in: (B, L, d_state)
            #
            # We avoid a sequential per-token lax.scan (very slow on GPU at long L)
            # by using an associative scan over affine transforms:
            #   s_t = a_t * s_{t-1} + b_t
            # where a_t = exp(dt_t * A), b_t = x_t * dt_t * B_t.
            x_f32 = x_in.astype(jnp.float32)
            dt_f32 = dt_in.astype(jnp.float32)
            B_f32 = B_in.astype(jnp.float32)
            C_f32 = C_in.astype(jnp.float32)

            # Time-major: (L, B, ...)
            x_tm = jnp.swapaxes(x_f32, 0, 1)   # (L,B,d_inner)
            dt_tm = jnp.swapaxes(dt_f32, 0, 1)  # (L,B,d_inner)
            B_tm = jnp.swapaxes(B_f32, 0, 1)   # (L,B,d_state)
            C_tm = jnp.swapaxes(C_f32, 0, 1)   # (L,B,d_state)

            a = jnp.exp(dt_tm[:, :, :, None] * A[None, None, :, :])  # (L,B,d_inner,d_state)
            b = (
                x_tm[:, :, :, None]
                * (dt_tm[:, :, :, None] * B_tm[:, :, None, :])
            )  # (L,B,d_inner,d_state)

            def combine(left, right):
                a1, b1 = left
                a2, b2 = right
                # Compose affine transforms: (a2,b2) ∘ (a1,b1)
                return a2 * a1, b2 + a2 * b1

            _, state_tm = jax.lax.associative_scan(combine, (a, b), axis=0)

            y_tm = (
                jnp.sum(state_tm * C_tm[:, :, None, :], axis=-1)
                + x_tm * D[None, None, :]
            )  # (L,B,d_inner)
            return jnp.swapaxes(y_tm, 0, 1)  # (B,L,d_inner)

        y = selective_scan(u, dt, B, C)
        if self.bidirectional:
            y_rev = selective_scan(u[:, ::-1, :], dt[:, ::-1, :], B[:, ::-1, :], C[:, ::-1, :])
            y = y + y_rev[:, ::-1, :]

        y = y.astype(self.dtype)

        # Gating + output projection back to d_model
        y = y * jax.nn.silu(gate)
        y = nn.Dense(
            int(self.d_model),
            dtype=self.dtype,
            param_dtype=jnp.float32,
            use_bias=True,
        )(y)
        if self.dropout_rate and float(self.dropout_rate) > 0.0:
            y = nn.Dropout(rate=float(self.dropout_rate), deterministic=not train)(y)
        return x + y


class Mamba1D(nn.Module):
    num_classes: int = cfg.NUM_CLASSES
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 8
    expand: int = 1
    dt_rank: int = 16
    d_conv: int = 4
    bidirectional: bool = True
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, train: bool = True):
        h = nn.Embed(
            num_embeddings=cfg.NUM_TOKEN_EMBEDDINGS,
            features=int(self.d_model),
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(tokens)

        if self.dropout_rate and float(self.dropout_rate) > 0.0:
            h = nn.Dropout(rate=float(self.dropout_rate), deterministic=not train)(h)

        for _ in range(int(self.n_layers)):
            h = MambaBlock1D(
                d_model=int(self.d_model),
                d_state=int(self.d_state),
                expand=int(self.expand),
                dt_rank=int(self.dt_rank),
                d_conv=int(self.d_conv),
                dropout_rate=float(self.dropout_rate),
                bidirectional=bool(self.bidirectional),
                dtype=self.dtype,
            )(h, train=train)

        h = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(h)
        logits = nn.Dense(
            int(self.num_classes),
            dtype=self.dtype,
            param_dtype=jnp.float32,
            use_bias=True,
        )(h)
        return logits.astype(jnp.float32)


def build_model(train_cfg: "TrainConfig", num_classes: int) -> nn.Module:
    arch = str(getattr(train_cfg, "arch", "unet1d")).lower().strip()
    if arch in {"unet", "unet1d", "u-net", "u_net"}:
        return UNet1D(
            num_classes=num_classes,
            emb_dim=train_cfg.model_dim,
            channels=train_cfg.channels,
            dropout_rate=train_cfg.dropout_rate,
            dtype=train_cfg.dtype,
        )
    if arch in {"mamba", "mamba1d", "bimamba", "ssm"}:
        return Mamba1D(
            num_classes=num_classes,
            d_model=train_cfg.model_dim,
            n_layers=getattr(train_cfg, "mamba_layers", 6),
            d_state=getattr(train_cfg, "mamba_d_state", 8),
            expand=getattr(train_cfg, "mamba_expand", 1),
            dt_rank=getattr(train_cfg, "mamba_dt_rank", 16),
            d_conv=getattr(train_cfg, "mamba_conv", 4),
            bidirectional=getattr(train_cfg, "mamba_bidirectional", True),
            dropout_rate=train_cfg.dropout_rate,
            dtype=train_cfg.dtype,
        )
    raise ValueError(f"Unknown arch '{arch}'. Expected 'unet1d' or 'mamba'.")

# ---------------------------
# Training utilities
# ---------------------------

class TrainState(train_state.TrainState):
    batch_stats: Optional[dict] = None

def count_params(params) -> int:
    return sum([np.prod(x.shape) for x in jtu.tree_leaves(params)])

def create_train_state(rng, cfg: "TrainConfig", num_classes: int):
    model = build_model(cfg, num_classes)
    dummy_tokens = jnp.zeros((1, cfg.MODEL_WINDOW_BYTES), dtype=jnp.int32)
    variables = model.init({"params": rng, "dropout": rng}, dummy_tokens, train=True)
    params = variables["params"]

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=cfg.lr, warmup_steps=cfg.warmup,
        decay_steps=cfg.steps, end_value=cfg.lr * 0.1
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule, b1=0.9, b2=0.95, eps=1e-8,
            weight_decay=cfg.weight_decay,
            mask=decay_mask(params)
        ),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def _ignored_token_mask(tokens: Optional[jnp.ndarray]) -> Optional[jnp.ndarray]:
    if tokens is None:
        return None
    ignore_vals = getattr(cfg, "IGNORED_TRAINING_TOKEN_IDS", ())
    if not ignore_vals:
        return jnp.zeros_like(tokens, dtype=jnp.bool_)
    mask = jnp.zeros_like(tokens, dtype=jnp.bool_)
    for val in ignore_vals:
        mask = jnp.logical_or(mask, tokens == int(val))
    return mask


def _supervision_mask(
    labels: jnp.ndarray,
    tokens: Optional[jnp.ndarray],
    pad_id: Optional[int] = None,
) -> jnp.ndarray:
    if pad_id is None:
        pad_id = cfg.PAD_ID
    # Treat any labels mapped to the derived "other" bucket as padding so
    # they do not contribute to loss/accuracy during training. This keeps
    # supervision aligned with the model's explicit logits and avoids NaNs
    # when fine-tuning on monitor sets that include an 'other' label.
    other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
    if other_idx is not None:
        labels = jnp.where(labels == int(other_idx), pad_id, labels)
    mask = labels != pad_id
    ignore_mask = _ignored_token_mask(tokens) if tokens is not None else None
    if ignore_mask is not None:
        mask = jnp.logical_and(mask, jnp.logical_not(ignore_mask))
    return mask


@jax.jit
def cross_entropy_masked(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    tokens: Optional[jnp.ndarray] = None,
    pad_id: Optional[int] = None,
) -> jnp.ndarray:
    mask = _supervision_mask(labels, tokens, pad_id)
    safe_labels = jnp.where(mask, labels, 0)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, safe_labels.astype(jnp.int32)
    )
    loss = loss * mask.astype(loss.dtype)
    denom = jnp.maximum(1, jnp.sum(mask))
    return jnp.sum(loss) / denom


@jax.jit
def accuracy_masked(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    tokens: Optional[jnp.ndarray] = None,
    pad_id: Optional[int] = None,
) -> jnp.ndarray:
    mask = _supervision_mask(labels, tokens, pad_id)
    pred = jnp.argmax(logits, axis=-1).astype(labels.dtype)
    correct = jnp.sum(jnp.logical_and(pred == labels, mask).astype(jnp.int32))
    total = jnp.maximum(1, jnp.sum(mask.astype(jnp.int32)))
    return (correct / total).astype(jnp.float32)

@jax.jit
def train_step(state: TrainState, batch_tokens: jnp.ndarray, batch_labels: jnp.ndarray, rng):
    """Perform a single training step."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            rngs={"dropout": dropout_rng}
        )
        loss = cross_entropy_masked(logits, batch_labels, tokens=batch_tokens)
        return loss, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    return state, loss, acc

def train_step_no_jit(state: TrainState, batch_tokens: jnp.ndarray, batch_labels: jnp.ndarray, rng):
    """Debug-friendly, non-JITted training step."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            rngs={"dropout": dropout_rng}
        )
        loss = cross_entropy_masked(logits, batch_labels, tokens=batch_tokens)
        return loss, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    return state, loss, acc

@jax.jit
def microbatch_grad_step(state: TrainState, batch_tokens: jnp.ndarray, batch_labels: jnp.ndarray, rng):
    """Compute gradients, loss, and accuracy for a single microbatch without applying updates."""
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            rngs={"dropout": dropout_rng},
        )
        loss = cross_entropy_masked(logits, batch_labels, tokens=batch_tokens)
        return loss, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    return grads, loss, acc


def microbatch_grad_step_no_jit(state: TrainState, batch_tokens: jnp.ndarray, batch_labels: jnp.ndarray, rng):
    """Non-JIT version of microbatch_grad_step."""
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            rngs={"dropout": dropout_rng},
        )
        loss = cross_entropy_masked(logits, batch_labels, tokens=batch_tokens)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    return grads, loss, acc


def grad_global_norm(grads) -> jnp.ndarray:
    """Compute global norm of a gradient PyTree."""
    return jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jtu.tree_leaves(grads)]))

@jax.jit
def eval_step(state: TrainState, batch_tokens: jnp.ndarray, batch_labels: jnp.ndarray, rng):
    """Perform a single evaluation step."""
    logits = state.apply_fn(
        {"params": state.params},
        batch_tokens,
        train=False,
        rngs={"dropout": rng}
    )
    loss = cross_entropy_masked(logits, batch_labels, tokens=batch_tokens)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    return loss, acc
