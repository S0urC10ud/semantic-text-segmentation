"""
The 1D U-Net model architecture and training-related utilities like the
TrainState, loss functions, and train/eval steps.
"""
from typing import Any, Mapping, TYPE_CHECKING, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from inference.mamba_cuda import selective_scan_inference
import utils.config as cfg
from utils.token_utils import COMPACT_TOKEN_TABLE
from flax import linen as nn
from flax import serialization
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
    aux_offsets: Tuple[int, ...] = cfg.AUX_NEIGHBOR_OFFSETS
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16
    num_token_embeddings: int = cfg.NUM_TOKEN_EMBEDDINGS

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        train: bool = True,
        return_auxiliary: bool = False,
    ):
        tok = tokens.astype(jnp.int32)
        if int(self.num_token_embeddings) != cfg.NUM_TOKEN_EMBEDDINGS:
            tok = jnp.asarray(COMPACT_TOKEN_TABLE)[jnp.clip(tok, 0, cfg.NUM_TOKEN_EMBEDDINGS - 1)]
        h = nn.Embed(num_embeddings=self.num_token_embeddings, features=self.emb_dim,
                     embedding_init=nn.initializers.normal(stddev=0.02),
                     dtype=self.dtype, param_dtype=jnp.float32)(tok)

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
        logits = logits_bf16.astype(jnp.float32)
        if not return_auxiliary or not self.aux_offsets:
            return logits

        aux_logits_bf16 = nn.Conv(
            self.num_classes * len(self.aux_offsets),
            (1,),
            padding="SAME",
            dtype=self.dtype,
            param_dtype=jnp.float32,
            name="aux_logits_head",
        )(h)
        aux_logits = aux_logits_bf16.reshape(
            aux_logits_bf16.shape[:2] + (len(self.aux_offsets), self.num_classes)
        )
        return logits, aux_logits.astype(jnp.float32)

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
    inference_kernel: str = "default"

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

        y = selective_scan_inference(
            u,
            dt,
            B,
            C,
            A,
            D,
            backend=self.inference_kernel if not train else "default",
        )
        if self.bidirectional:
            y_rev = selective_scan_inference(
                u[:, ::-1, :],
                dt[:, ::-1, :],
                B[:, ::-1, :],
                C[:, ::-1, :],
                A,
                D,
                backend=self.inference_kernel if not train else "default",
            )
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
    aux_offsets: Tuple[int, ...] = cfg.AUX_NEIGHBOR_OFFSETS
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.bfloat16
    inference_kernel: str = "default"
    use_remat: bool = True
    num_token_embeddings: int = cfg.NUM_TOKEN_EMBEDDINGS

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,
        train: bool = True,
        return_auxiliary: bool = False,
    ):
        tok = tokens.astype(jnp.int32)
        if int(self.num_token_embeddings) != cfg.NUM_TOKEN_EMBEDDINGS:
            tok = jnp.asarray(COMPACT_TOKEN_TABLE)[jnp.clip(tok, 0, cfg.NUM_TOKEN_EMBEDDINGS - 1)]
        h = nn.Embed(
            num_embeddings=self.num_token_embeddings,
            features=int(self.d_model),
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(tok)

        if self.dropout_rate and float(self.dropout_rate) > 0.0:
            h = nn.Dropout(rate=float(self.dropout_rate), deterministic=not train)(h)

        if self.use_remat:
            block_cls = nn.remat(MambaBlock1D, static_argnums=(2,))
            for _ in range(int(self.n_layers)):
                h = block_cls(
                    d_model=int(self.d_model),
                    d_state=int(self.d_state),
                    expand=int(self.expand),
                    dt_rank=int(self.dt_rank),
                    d_conv=int(self.d_conv),
                    dropout_rate=float(self.dropout_rate),
                    bidirectional=bool(self.bidirectional),
                    dtype=self.dtype,
                    inference_kernel=str(self.inference_kernel),
                )(h, train)
        else:
            for layer_idx in range(int(self.n_layers)):
                h = MambaBlock1D(
                    d_model=int(self.d_model),
                    d_state=int(self.d_state),
                    expand=int(self.expand),
                    dt_rank=int(self.dt_rank),
                    d_conv=int(self.d_conv),
                    dropout_rate=float(self.dropout_rate),
                    bidirectional=bool(self.bidirectional),
                    dtype=self.dtype,
                    inference_kernel=str(self.inference_kernel),
                    name=f"CheckpointMambaBlock1D_{layer_idx}",
                )(h, train)

        h = nn.LayerNorm(dtype=self.dtype, param_dtype=jnp.float32)(h)
        logits = nn.Dense(
            int(self.num_classes),
            dtype=self.dtype,
            param_dtype=jnp.float32,
            use_bias=True,
        )(h)
        logits = logits.astype(jnp.float32)
        if not return_auxiliary or not self.aux_offsets:
            return logits

        aux_logits = nn.Dense(
            int(self.num_classes) * len(self.aux_offsets),
            dtype=self.dtype,
            param_dtype=jnp.float32,
            use_bias=True,
            name="aux_logits_head",
        )(h)
        aux_logits = aux_logits.reshape(
            aux_logits.shape[:2] + (len(self.aux_offsets), int(self.num_classes))
        )
        return logits, aux_logits.astype(jnp.float32)


def build_model(train_cfg: "TrainConfig", num_classes: int) -> nn.Module:
    arch = str(getattr(train_cfg, "arch", "unet1d")).lower().strip()
    if arch in {"unet", "unet1d", "u-net", "u_net"}:
        return UNet1D(
            num_classes=num_classes,
            emb_dim=train_cfg.model_dim,
            channels=train_cfg.channels,
            dropout_rate=train_cfg.dropout_rate,
            dtype=train_cfg.dtype,
            num_token_embeddings=int(getattr(train_cfg, "num_token_embeddings", cfg.NUM_TOKEN_EMBEDDINGS)),
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
            num_token_embeddings=int(getattr(train_cfg, "num_token_embeddings", cfg.NUM_TOKEN_EMBEDDINGS)),
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
    variables = model.init(
        {"params": rng, "dropout": rng},
        dummy_tokens,
        train=True,
        return_auxiliary=True,
    )
    params = variables["params"]
    decay_steps = int(getattr(cfg, "schedule_steps", 0) or cfg.steps)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=cfg.lr, warmup_steps=cfg.warmup,
        decay_steps=decay_steps, end_value=cfg.lr * 0.1
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


def split_model_outputs(outputs) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    if (
        isinstance(outputs, tuple)
        and len(outputs) == 2
        and outputs[0] is not None
    ):
        return outputs[0], outputs[1]
    return outputs, None


def checkpoint_params_subtree(obj: Any):
    if hasattr(obj, "params"):
        return getattr(obj, "params")
    if isinstance(obj, Mapping):
        return obj.get("params", obj)
    state_dict = serialization.to_state_dict(obj)
    if isinstance(state_dict, Mapping):
        return state_dict.get("params", state_dict)
    return state_dict


def _restore_leaf_is_compatible(target_leaf: Any, source_leaf: Any) -> bool:
    try:
        target_shape = tuple(np.shape(np.asarray(target_leaf)))
        source_shape = tuple(np.shape(np.asarray(source_leaf)))
        return target_shape == source_shape
    except Exception:
        return type(target_leaf) is type(source_leaf)


def merge_compatible_state(target_obj: Any, source_obj: Any):
    target_state = serialization.to_state_dict(target_obj)
    source_state = serialization.to_state_dict(source_obj)

    loaded = []
    missing = []
    mismatched = []
    extra = []

    def _collect_paths(node: Any, path: Tuple[str, ...]) -> list[Tuple[str, ...]]:
        if isinstance(node, Mapping):
            if not node:
                return [path]
            out: list[Tuple[str, ...]] = []
            for key, value in node.items():
                out.extend(_collect_paths(value, path + (str(key),)))
            return out
        return [path]

    def _merge_nodes(target_node: Any, source_node: Any, path: Tuple[str, ...]) -> Any:
        if isinstance(target_node, Mapping):
            if not isinstance(source_node, Mapping):
                mismatched.extend(_collect_paths(target_node, path))
                return target_node
            merged_node = {}
            for key, target_child in target_node.items():
                key_str = str(key)
                child_path = path + (key_str,)
                if key not in source_node:
                    missing.extend(_collect_paths(target_child, child_path))
                    merged_node[key] = target_child
                    continue
                merged_node[key] = _merge_nodes(target_child, source_node[key], child_path)
            for key, source_child in source_node.items():
                if key not in target_node:
                    extra.extend(_collect_paths(source_child, path + (str(key),)))
            return merged_node

        if isinstance(source_node, Mapping):
            mismatched.append(path)
            return target_node
        if _restore_leaf_is_compatible(target_node, source_node):
            loaded.append(path)
            return source_node
        mismatched.append(path)
        return target_node

    merged_state = _merge_nodes(target_state, source_state, ())
    restored = serialization.from_state_dict(target_obj, merged_state)
    return restored, {
        "loaded": tuple(loaded),
        "missing": tuple(missing),
        "mismatched": tuple(mismatched),
        "extra": tuple(extra),
    }


def seed_missing_auxiliary_heads_from_main(
    target_params: Any,
    source_params: Optional[Any] = None,
) -> Tuple[Any, Optional[str]]:
    """Initialize missing aux heads by tiling the main prediction head."""
    offsets = tuple(getattr(cfg, "AUX_NEIGHBOR_OFFSETS", ()))
    if not offsets:
        return target_params, None

    target_state = serialization.to_state_dict(target_params)
    source_state = (
        serialization.to_state_dict(source_params)
        if source_params is not None
        else {}
    )
    aux_head = target_state.get("aux_logits_head")
    if not isinstance(aux_head, Mapping):
        return target_params, None
    if isinstance(source_state, Mapping) and "aux_logits_head" in source_state:
        return target_params, None

    aux_kernel = aux_head.get("kernel")
    if aux_kernel is None:
        return target_params, None

    repeats = len(offsets)
    for main_head_name in ("Conv_0", "Dense_0"):
        main_head = target_state.get(main_head_name)
        if not isinstance(main_head, Mapping):
            continue

        main_kernel = main_head.get("kernel")
        if main_kernel is None:
            continue

        main_kernel_np = np.asarray(main_kernel)
        aux_kernel_np = np.asarray(aux_kernel)
        if (
            main_kernel_np.ndim < 1
            or aux_kernel_np.ndim < 1
            or main_kernel_np.shape[:-1] != aux_kernel_np.shape[:-1]
            or main_kernel_np.shape[-1] * repeats != aux_kernel_np.shape[-1]
        ):
            continue

        updated_aux = dict(aux_head)
        updated_aux["kernel"] = np.concatenate(
            [main_kernel_np] * repeats,
            axis=-1,
        ).astype(aux_kernel_np.dtype, copy=False)

        main_bias = main_head.get("bias")
        aux_bias = aux_head.get("bias")
        if main_bias is not None and aux_bias is not None:
            main_bias_np = np.asarray(main_bias)
            aux_bias_np = np.asarray(aux_bias)
            if (
                main_bias_np.ndim >= 1
                and aux_bias_np.ndim >= 1
                and main_bias_np.shape[:-1] == aux_bias_np.shape[:-1]
                and main_bias_np.shape[-1] * repeats == aux_bias_np.shape[-1]
            ):
                updated_aux["bias"] = np.concatenate(
                    [main_bias_np] * repeats,
                    axis=-1,
                ).astype(aux_bias_np.dtype, copy=False)

        updated_state = dict(target_state)
        updated_state["aux_logits_head"] = updated_aux
        restored = serialization.from_state_dict(target_params, updated_state)
        offset_desc = ", ".join(f"{offset:+d}" for offset in offsets)
        note = (
            "Auxiliary neighbor heads were missing in the checkpoint; "
            f"initialized from {main_head_name} for offsets [{offset_desc}]."
        )
        return restored, note

    return target_params, None

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
    weights = mask.astype(loss.dtype)
    boundary_extra = float(max(0.0, getattr(cfg, "BOUNDARY_LOSS_WEIGHT", 0.0)))
    boundary_radius = int(max(0, getattr(cfg, "BOUNDARY_LOSS_RADIUS", 0)))
    if boundary_extra > 0.0 and boundary_radius >= 0 and labels.ndim >= 2:
        transition = jnp.zeros_like(mask, dtype=jnp.bool_)
        adjacent_valid = jnp.logical_and(mask[:, 1:], mask[:, :-1])
        changed = jnp.logical_and(labels[:, 1:] != labels[:, :-1], adjacent_valid)
        transition = transition.at[:, 1:].set(changed)
        vicinity = jnp.zeros_like(transition, dtype=jnp.bool_)
        for offset in range(-boundary_radius, boundary_radius + 1):
            vicinity = jnp.logical_or(
                vicinity,
                _shift_sequence(transition, offset, False),
            )
        weights = weights * (
            1.0 + jnp.asarray(boundary_extra, dtype=loss.dtype) * vicinity.astype(loss.dtype)
        )
    loss = loss * weights
    denom = jnp.maximum(1.0, jnp.sum(weights))
    return jnp.sum(loss) / denom


def _shift_sequence(values: jnp.ndarray, offset: int, fill_value) -> jnp.ndarray:
    shift = abs(int(offset))
    if shift == 0:
        return values
    seq_len = values.shape[1]
    if shift >= seq_len:
        return jnp.full_like(values, fill_value)
    fill = jnp.full(
        values.shape[:1] + (shift,) + values.shape[2:],
        fill_value,
        dtype=values.dtype,
    )
    if offset > 0:
        return jnp.concatenate([values[:, shift:, ...], fill], axis=1)
    return jnp.concatenate([fill, values[:, : seq_len - shift, ...]], axis=1)


def auxiliary_neighbor_cross_entropy(
    aux_logits: Optional[jnp.ndarray],
    labels: jnp.ndarray,
    tokens: Optional[jnp.ndarray] = None,
    pad_id: Optional[int] = None,
) -> jnp.ndarray:
    if aux_logits is None:
        return jnp.zeros((), dtype=jnp.float32)
    if pad_id is None:
        pad_id = cfg.PAD_ID

    offsets = tuple(getattr(cfg, "AUX_NEIGHBOR_OFFSETS", ()))
    if not offsets:
        return jnp.zeros((), dtype=aux_logits.dtype)

    target_mask = _supervision_mask(labels, tokens, pad_id)
    source_mask = labels != pad_id
    total_loss = jnp.zeros((), dtype=aux_logits.dtype)
    total_count = jnp.zeros((), dtype=jnp.int32)

    for head_idx, offset in enumerate(offsets):
        shifted_labels = _shift_sequence(labels, offset, 0)
        shifted_mask = _shift_sequence(target_mask, offset, False)
        shifted_mask = jnp.logical_and(shifted_mask, source_mask)
        safe_labels = jnp.where(shifted_mask, shifted_labels, 0)
        head_loss = optax.softmax_cross_entropy_with_integer_labels(
            aux_logits[:, :, head_idx, :],
            safe_labels.astype(jnp.int32),
        )
        total_loss = total_loss + jnp.sum(head_loss * shifted_mask.astype(head_loss.dtype))
        total_count = total_count + jnp.sum(shifted_mask.astype(jnp.int32))

    denom = jnp.maximum(1, total_count)
    return total_loss / denom.astype(total_loss.dtype)


def supervised_training_loss(
    model_outputs,
    labels: jnp.ndarray,
    tokens: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    logits, aux_logits = split_model_outputs(model_outputs)
    loss = cross_entropy_masked(logits, labels, tokens=tokens)
    aux_weight = jnp.asarray(cfg.AUX_NEIGHBOR_LOSS_WEIGHT, dtype=logits.dtype)
    aux_loss = auxiliary_neighbor_cross_entropy(aux_logits, labels, tokens=tokens)
    return loss + aux_weight * aux_loss, logits


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
def outlier_uniform_cross_entropy(logits: jnp.ndarray) -> jnp.ndarray:
    """
    OE loss for outlier batches: H(U; p) where U is uniform over classes.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -(log_probs.mean(axis=-1)).mean()


@jax.jit
def intrinsic_oe_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    tokens: Optional[jnp.ndarray] = None,
    lambda_weight: float = 0.1,
) -> jnp.ndarray:
    """
    Apply OE uniform distribution penalty to any tokens inherently labeled 
    as OTHER_CLASS_INDEX within the main training batch.
    """
    other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
    if other_idx is None:
        return jnp.zeros((), dtype=logits.dtype)

    other_mask = (labels == int(other_idx))
    ignore_mask = _ignored_token_mask(tokens) if tokens is not None else None
    if ignore_mask is not None:
        other_mask = jnp.logical_and(other_mask, jnp.logical_not(ignore_mask))

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_uniform_ce = -(log_probs.mean(axis=-1))

    masked_ce = token_uniform_ce * other_mask.astype(token_uniform_ce.dtype)
    denom = jnp.maximum(1, jnp.sum(other_mask))
    loss_val = jnp.sum(masked_ce) / denom

    has_other = jnp.any(other_mask)
    weight = jnp.asarray(lambda_weight, dtype=logits.dtype)
    return weight * has_other.astype(logits.dtype) * loss_val


@jax.jit
def train_step(state: TrainState, batch_tokens: jnp.ndarray, batch_labels: jnp.ndarray, rng):
    """Perform a single training step."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    def loss_fn(params):
        model_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng}
        )
        loss, logits = supervised_training_loss(model_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_oe = intrinsic_oe_loss(logits, batch_labels, tokens=batch_tokens)
        return loss + intrinsic_oe, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    return state, loss, acc

def train_step_no_jit(state: TrainState, batch_tokens: jnp.ndarray, batch_labels: jnp.ndarray, rng):
    """Debug-friendly, non-JITted training step."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    def loss_fn(params):
        model_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng}
        )
        loss, logits = supervised_training_loss(model_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_oe = intrinsic_oe_loss(logits, batch_labels, tokens=batch_tokens)
        return loss + intrinsic_oe, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    return state, loss, acc


@jax.jit
def train_step_with_oe(
    state: TrainState,
    batch_tokens: jnp.ndarray,
    batch_labels: jnp.ndarray,
    outlier_tokens: jnp.ndarray,
    oe_lambda: float,
    rng,
):
    """Single training step with OE regularization."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    dropout_rng_id, dropout_rng_oe = jax.random.split(dropout_rng)
    oe_lambda_f = jnp.asarray(oe_lambda, dtype=jnp.float32)

    def loss_fn(params):
        id_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng_id},
        )
        ce_id, logits_id = supervised_training_loss(id_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_lambda = jnp.where(oe_lambda > 0.0, oe_lambda, 0.1)
        intrinsic_oe = intrinsic_oe_loss(logits_id, batch_labels, tokens=batch_tokens, lambda_weight=intrinsic_lambda)

        logits_out = state.apply_fn(
            {"params": params},
            outlier_tokens,
            train=True,
            rngs={"dropout": dropout_rng_oe},
        )
        ce_uniform = outlier_uniform_cross_entropy(logits_out)
        total = ce_id + intrinsic_oe + oe_lambda_f * ce_uniform
        return total, (logits_id, ce_uniform)

    (loss, (logits_id, ce_uniform)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = accuracy_masked(logits_id, batch_labels, tokens=batch_tokens)
    return state, loss, acc, ce_uniform


def train_step_with_oe_no_jit(
    state: TrainState,
    batch_tokens: jnp.ndarray,
    batch_labels: jnp.ndarray,
    outlier_tokens: jnp.ndarray,
    oe_lambda: float,
    rng,
):
    """Non-JIT version of train_step_with_oe."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    dropout_rng_id, dropout_rng_oe = jax.random.split(dropout_rng)
    oe_lambda_f = jnp.asarray(oe_lambda, dtype=jnp.float32)

    def loss_fn(params):
        id_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng_id},
        )
        ce_id, logits_id = supervised_training_loss(id_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_lambda = jnp.where(oe_lambda > 0.0, oe_lambda, 0.1)
        intrinsic_oe = intrinsic_oe_loss(logits_id, batch_labels, tokens=batch_tokens, lambda_weight=intrinsic_lambda)

        logits_out = state.apply_fn(
            {"params": params},
            outlier_tokens,
            train=True,
            rngs={"dropout": dropout_rng_oe},
        )
        ce_uniform = outlier_uniform_cross_entropy(logits_out)
        total = ce_id + intrinsic_oe + oe_lambda_f * ce_uniform
        return total, (logits_id, ce_uniform)

    (loss, (logits_id, ce_uniform)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    acc = accuracy_masked(logits_id, batch_labels, tokens=batch_tokens)
    return state, loss, acc, ce_uniform


@jax.jit
def microbatch_grad_step(state: TrainState, batch_tokens: jnp.ndarray, batch_labels: jnp.ndarray, rng):
    """Compute gradients, loss, and accuracy for a single microbatch without applying updates."""
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        model_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng},
        )
        loss, logits = supervised_training_loss(model_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_oe = intrinsic_oe_loss(logits, batch_labels, tokens=batch_tokens)
        return loss + intrinsic_oe, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    return grads, loss, acc


@jax.jit
def microbatch_grad_step_with_oe(
    state: TrainState,
    batch_tokens: jnp.ndarray,
    batch_labels: jnp.ndarray,
    outlier_tokens: jnp.ndarray,
    oe_lambda: float,
    rng,
):
    """Microbatch gradient step with OE regularization."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    dropout_rng_id, dropout_rng_oe = jax.random.split(dropout_rng)
    oe_lambda_f = jnp.asarray(oe_lambda, dtype=jnp.float32)

    def loss_fn(params):
        id_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng_id},
        )
        ce_id, logits_id = supervised_training_loss(id_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_lambda = jnp.where(oe_lambda > 0.0, oe_lambda, 0.1)
        intrinsic_oe = intrinsic_oe_loss(logits_id, batch_labels, tokens=batch_tokens, lambda_weight=intrinsic_lambda)

        logits_out = state.apply_fn(
            {"params": params},
            outlier_tokens,
            train=True,
            rngs={"dropout": dropout_rng_oe},
        )
        ce_uniform = outlier_uniform_cross_entropy(logits_out)
        total = ce_id + intrinsic_oe + oe_lambda_f * ce_uniform
        return total, (logits_id, ce_uniform)

    (loss, (logits_id, ce_uniform)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    acc = accuracy_masked(logits_id, batch_labels, tokens=batch_tokens)
    return grads, loss, acc, ce_uniform


def microbatch_grad_step_no_jit(state: TrainState, batch_tokens: jnp.ndarray, batch_labels: jnp.ndarray, rng):
    """Non-JIT version of microbatch_grad_step."""
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        model_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng},
        )
        loss, logits = supervised_training_loss(model_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_oe = intrinsic_oe_loss(logits, batch_labels, tokens=batch_tokens)
        return loss + intrinsic_oe, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    return grads, loss, acc


def microbatch_grad_step_with_oe_no_jit(
    state: TrainState,
    batch_tokens: jnp.ndarray,
    batch_labels: jnp.ndarray,
    outlier_tokens: jnp.ndarray,
    oe_lambda: float,
    rng,
):
    """Non-JIT version of microbatch_grad_step_with_oe."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    dropout_rng_id, dropout_rng_oe = jax.random.split(dropout_rng)
    oe_lambda_f = jnp.asarray(oe_lambda, dtype=jnp.float32)

    def loss_fn(params):
        id_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng_id},
        )
        ce_id, logits_id = supervised_training_loss(id_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_lambda = jnp.where(oe_lambda > 0.0, oe_lambda, 0.1)
        intrinsic_oe = intrinsic_oe_loss(logits_id, batch_labels, tokens=batch_tokens, lambda_weight=intrinsic_lambda)

        logits_out = state.apply_fn(
            {"params": params},
            outlier_tokens,
            train=True,
            rngs={"dropout": dropout_rng_oe},
        )
        ce_uniform = outlier_uniform_cross_entropy(logits_out)
        total = ce_id + intrinsic_oe + oe_lambda_f * ce_uniform
        return total, (logits_id, ce_uniform)

    (loss, (logits_id, ce_uniform)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    acc = accuracy_masked(logits_id, batch_labels, tokens=batch_tokens)
    return grads, loss, acc, ce_uniform


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


@jax.jit
def eval_step_with_logits(
    state: TrainState,
    batch_tokens: jnp.ndarray,
    batch_labels: jnp.ndarray,
    rng,
):
    """Perform a single evaluation step and return logits from the same forward pass."""
    logits = state.apply_fn(
        {"params": state.params},
        batch_tokens,
        train=False,
        rngs={"dropout": rng}
    )
    loss = cross_entropy_masked(logits, batch_labels, tokens=batch_tokens)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    return loss, acc, logits


# ---------------------------
# Multi-GPU (pmap) utilities
# ---------------------------

def replicate_state(state: TrainState, num_devices: int) -> TrainState:
    """Replicate a TrainState across `num_devices` for use with jax.pmap."""
    devices = jax.local_devices()[:num_devices]
    return jax.device_put_replicated(state, devices)


def unreplicate_state(state: TrainState) -> TrainState:
    """Extract single-device copy from a replicated TrainState (take device 0)."""
    return jax.tree.map(lambda x: x[0], state)


@jax.pmap
def p_apply_gradients(state: TrainState, grads):
    """Apply gradients to a replicated TrainState under pmap."""
    return state.apply_gradients(grads=grads)


# --- pmap-ed train steps ---

def _p_train_step_impl(state, batch_tokens, batch_labels, rng):
    """Inner logic for pmap-ed single training step."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    def loss_fn(params):
        model_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng}
        )
        loss, logits = supervised_training_loss(model_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_oe = intrinsic_oe_loss(logits, batch_labels, tokens=batch_tokens)
        return loss + intrinsic_oe, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="devices")
    loss = jax.lax.pmean(loss, axis_name="devices")
    state = state.apply_gradients(grads=grads)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    acc = jax.lax.pmean(acc, axis_name="devices")
    return state, loss, acc

p_train_step = jax.pmap(_p_train_step_impl, axis_name="devices")


def _p_train_step_with_oe_impl(state, batch_tokens, batch_labels, outlier_tokens, oe_lambda, rng):
    """Inner logic for pmap-ed training step with OE regularization."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    dropout_rng_id, dropout_rng_oe = jax.random.split(dropout_rng)
    oe_lambda_f = jnp.asarray(oe_lambda, dtype=jnp.float32)

    def loss_fn(params):
        id_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng_id},
        )
        ce_id, logits_id = supervised_training_loss(id_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_lambda = jnp.where(oe_lambda_f > 0.0, oe_lambda_f, 0.1)
        intrinsic_oe = intrinsic_oe_loss(logits_id, batch_labels, tokens=batch_tokens, lambda_weight=intrinsic_lambda)

        logits_out = state.apply_fn(
            {"params": params},
            outlier_tokens,
            train=True,
            rngs={"dropout": dropout_rng_oe},
        )
        ce_uniform = outlier_uniform_cross_entropy(logits_out)
        total = ce_id + intrinsic_oe + oe_lambda_f * ce_uniform
        return total, (logits_id, ce_uniform)

    (loss, (logits_id, ce_uniform)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="devices")
    loss = jax.lax.pmean(loss, axis_name="devices")
    ce_uniform = jax.lax.pmean(ce_uniform, axis_name="devices")
    state = state.apply_gradients(grads=grads)
    acc = accuracy_masked(logits_id, batch_labels, tokens=batch_tokens)
    acc = jax.lax.pmean(acc, axis_name="devices")
    return state, loss, acc, ce_uniform

p_train_step_with_oe = jax.pmap(
    _p_train_step_with_oe_impl,
    axis_name="devices",
    in_axes=(0, 0, 0, 0, None, 0),  # oe_lambda is a scalar, not sharded
)


# --- pmap-ed microbatch grad steps ---

def _p_microbatch_grad_step_impl(state, batch_tokens, batch_labels, rng):
    """Inner logic for pmap-ed microbatch gradient step (no optimizer update)."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    def loss_fn(params):
        model_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng},
        )
        loss, logits = supervised_training_loss(model_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_oe = intrinsic_oe_loss(logits, batch_labels, tokens=batch_tokens)
        return loss + intrinsic_oe, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="devices")
    loss = jax.lax.pmean(loss, axis_name="devices")
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    acc = jax.lax.pmean(acc, axis_name="devices")
    return grads, loss, acc

p_microbatch_grad_step = jax.pmap(_p_microbatch_grad_step_impl, axis_name="devices")


def _p_microbatch_grad_step_with_oe_impl(state, batch_tokens, batch_labels, outlier_tokens, oe_lambda, rng):
    """Inner logic for pmap-ed microbatch gradient step with OE."""
    dropout_rng = jax.random.fold_in(rng, state.step)
    dropout_rng_id, dropout_rng_oe = jax.random.split(dropout_rng)
    oe_lambda_f = jnp.asarray(oe_lambda, dtype=jnp.float32)

    def loss_fn(params):
        id_outputs = state.apply_fn(
            {"params": params},
            batch_tokens,
            train=True,
            return_auxiliary=True,
            rngs={"dropout": dropout_rng_id},
        )
        ce_id, logits_id = supervised_training_loss(id_outputs, batch_labels, tokens=batch_tokens)
        intrinsic_lambda = jnp.where(oe_lambda_f > 0.0, oe_lambda_f, 0.1)
        intrinsic_oe = intrinsic_oe_loss(logits_id, batch_labels, tokens=batch_tokens, lambda_weight=intrinsic_lambda)

        logits_out = state.apply_fn(
            {"params": params},
            outlier_tokens,
            train=True,
            rngs={"dropout": dropout_rng_oe},
        )
        ce_uniform = outlier_uniform_cross_entropy(logits_out)
        total = ce_id + intrinsic_oe + oe_lambda_f * ce_uniform
        return total, (logits_id, ce_uniform)

    (loss, (logits_id, ce_uniform)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="devices")
    loss = jax.lax.pmean(loss, axis_name="devices")
    ce_uniform = jax.lax.pmean(ce_uniform, axis_name="devices")
    acc = accuracy_masked(logits_id, batch_labels, tokens=batch_tokens)
    acc = jax.lax.pmean(acc, axis_name="devices")
    return grads, loss, acc, ce_uniform

p_microbatch_grad_step_with_oe = jax.pmap(
    _p_microbatch_grad_step_with_oe_impl,
    axis_name="devices",
    in_axes=(0, 0, 0, 0, None, 0),  # oe_lambda is a scalar, not sharded
)


# --- pmap-ed eval step ---

def _p_eval_step_impl(state, batch_tokens, batch_labels, rng):
    """Inner logic for pmap-ed evaluation step."""
    logits = state.apply_fn(
        {"params": state.params},
        batch_tokens,
        train=False,
        rngs={"dropout": rng}
    )
    loss = cross_entropy_masked(logits, batch_labels, tokens=batch_tokens)
    acc = accuracy_masked(logits, batch_labels, tokens=batch_tokens)
    loss = jax.lax.pmean(loss, axis_name="devices")
    acc = jax.lax.pmean(acc, axis_name="devices")
    return loss, acc

p_eval_step = jax.pmap(_p_eval_step_impl, axis_name="devices")
