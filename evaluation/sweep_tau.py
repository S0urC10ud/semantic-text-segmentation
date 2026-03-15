#!/usr/bin/env python3
"""
Sweep open-set OTHER threshold (tau) on the monitor validation memmap.

The script reuses training-time monitor evaluation logic:
  - monitor memmap loader: utils.monitor_eval.load_monitor_memmaps
  - thresholding + confusion accumulation: utils.monitor_eval.evaluate_monitor_set
  - metric computation: utils.metrics_helper.compute_metrics_from_confusion
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training import checkpoints

import utils.config as cfg
from utils.config import TrainConfig
from utils.metrics_helper import (
    _forward_logits,
    _valid_metric_mask,
    compute_metrics_from_confusion,
)
from utils.model import (
    checkpoint_params_subtree,
    create_train_state,
    eval_step,
    merge_compatible_state,
)
from utils.monitor_eval import evaluate_monitor_set, load_monitor_memmaps
from utils.token_utils import sanitize_tokens

try:
    from datasets import load_from_disk  # type: ignore
except Exception:
    load_from_disk = None  # type: ignore


def _extract_run_id_from_checkpoint(path: Path) -> Optional[str]:
    m = re.search(r"([a-z0-9]{8})", path.name.lower())
    return m.group(1) if m else None


def _load_wandb_hparams(ckpt_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    run_id = _extract_run_id_from_checkpoint(ckpt_path)
    if not run_id:
        return result
    wandb_root = REPO_ROOT / "train" / "wandb"
    if not wandb_root.exists():
        return result
    try:
        import yaml  # type: ignore
    except Exception:
        return result

    pattern = f"run-*-{run_id}"
    for run_dir in sorted(wandb_root.glob(pattern)):
        cfg_path = run_dir / "files" / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            config_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(config_data, dict):
            continue

        channels_val = config_data.get("channels", {}).get("value")
        if isinstance(channels_val, (list, tuple)):
            try:
                result["channels"] = [int(x) for x in channels_val]
            except Exception:
                pass

        arch_val = config_data.get("arch", {}).get("value")
        if isinstance(arch_val, str) and arch_val.strip():
            result["arch"] = arch_val.strip().lower()

        model_dim_val = config_data.get("model_dim", {}).get("value")
        if isinstance(model_dim_val, (int, float)):
            result["model_dim"] = int(model_dim_val)

        dtype_val = config_data.get("dtype", {}).get("value")
        if isinstance(dtype_val, str) and dtype_val.strip():
            result["dtype"] = dtype_val.rsplit(".", 1)[-1]

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
        if result:
            break
    return result


def _load_msgpack_params_tree(ckpt_path: Path) -> Optional[Mapping[str, Any]]:
    if not ckpt_path.exists() or not ckpt_path.is_file():
        return None
    try:
        restored = serialization.msgpack_restore(ckpt_path.read_bytes())
    except Exception:
        return None
    if isinstance(restored, Mapping) and "params" in restored and isinstance(restored["params"], Mapping):
        return restored["params"]
    if isinstance(restored, Mapping):
        return restored
    return None


def _infer_hparams_from_checkpoint(ckpt_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    params = _load_msgpack_params_tree(ckpt_path)
    if params is None:
        return result

    top_keys = {str(k) for k in params.keys()}
    is_mamba = any(k.startswith("MambaBlock1D_") for k in top_keys)
    result["arch"] = "mamba" if is_mamba else "unet1d"

    embed = params.get("Embed_0") if isinstance(params, Mapping) else None
    if isinstance(embed, Mapping):
        emb_arr = embed.get("embedding")
        if hasattr(emb_arr, "shape") and len(getattr(emb_arr, "shape", ())) >= 2:
            result["model_dim"] = int(emb_arr.shape[1])
            dtype_name = getattr(getattr(emb_arr, "dtype", None), "name", None)
            if isinstance(dtype_name, str):
                result["dtype"] = dtype_name

    if is_mamba:
        indices: List[int] = []
        for key in top_keys:
            m = re.match(r"^MambaBlock1D_(\d+)$", key)
            if m:
                indices.append(int(m.group(1)))
        if indices:
            result["mamba_layers"] = int(max(indices) + 1)

        block0 = params.get("MambaBlock1D_0")
        if isinstance(block0, Mapping):
            a_log = block0.get("A_log")
            if hasattr(a_log, "shape") and len(getattr(a_log, "shape", ())) >= 2:
                result["mamba_d_state"] = int(a_log.shape[-1])

            dense0 = block0.get("Dense_0")
            kernel0 = dense0.get("kernel") if isinstance(dense0, Mapping) else None
            d_model = result.get("model_dim")
            if (
                hasattr(kernel0, "shape")
                and len(getattr(kernel0, "shape", ())) >= 2
                and isinstance(d_model, int)
                and d_model > 0
            ):
                d_inner = int(kernel0.shape[-1]) // 2
                result["mamba_expand"] = int(max(1, d_inner // d_model))

            dense1 = block0.get("Dense_1")
            kernel1 = dense1.get("kernel") if isinstance(dense1, Mapping) else None
            d_state = result.get("mamba_d_state")
            if (
                hasattr(kernel1, "shape")
                and len(getattr(kernel1, "shape", ())) >= 2
                and isinstance(d_state, int)
            ):
                result["mamba_dt_rank"] = int(kernel1.shape[-1]) - 2 * d_state

            conv0 = block0.get("Conv_0")
            conv_kernel = conv0.get("kernel") if isinstance(conv0, Mapping) else None
            if hasattr(conv_kernel, "shape") and len(getattr(conv_kernel, "shape", ())) >= 1:
                result["mamba_conv"] = int(conv_kernel.shape[0])
        return result

    channels: List[int] = []
    current_in = result.get("model_dim")
    block_idx = 0
    while True:
        name = f"ConvBlock1D_{block_idx}"
        block = params.get(name)
        if not isinstance(block, Mapping):
            break
        conv = block.get("Conv_0")
        if not isinstance(conv, Mapping):
            break
        kernel = conv.get("kernel")
        if not hasattr(kernel, "shape") or len(getattr(kernel, "shape", ())) < 3:
            break
        in_ch = int(kernel.shape[-2])
        out_ch = int(kernel.shape[-1])
        if block_idx == 0 and current_in is None:
            current_in = in_ch
            result["model_dim"] = in_ch
        if channels and isinstance(current_in, int) and in_ch != current_in:
            break
        channels.append(out_ch)
        current_in = out_ch
        block_idx += 2
    if channels:
        result["channels"] = channels
    return result


def _resolve_hparam(
    cli_value: Any,
    wandb_value: Any,
    inferred_value: Any,
    default_value: Any,
) -> Tuple[Any, str]:
    if cli_value is not None:
        return cli_value, "cli"
    if wandb_value is not None:
        return wandb_value, "wandb"
    if inferred_value is not None:
        return inferred_value, "checkpoint"
    return default_value, "default"


def _parse_dtype(dtype_name: str) -> jnp.dtype:
    key = str(dtype_name or "bfloat16").strip().lower().rsplit(".", 1)[-1]
    if key in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if key in {"float16", "f16", "half"}:
        return jnp.float16
    if key in {"float32", "f32", "single"}:
        return jnp.float32
    if key in {"float64", "f64", "double"}:
        return jnp.float64
    raise ValueError(f"Unsupported dtype '{dtype_name}'.")


def _parse_channels(channels: Optional[str]) -> Optional[List[int]]:
    if channels is None:
        return None
    parsed = [int(x.strip()) for x in str(channels).split(",") if x.strip()]
    if not parsed:
        raise ValueError("--channels must contain at least one integer.")
    return parsed


def _resolve_ckpt_paths(path: str) -> Tuple[str, str, str]:
    blob_abs = os.path.abspath(path)
    ckpt_dir_abs = os.path.dirname(blob_abs) or os.getcwd()
    base = os.path.basename(blob_abs)
    prefix = base + "-"
    return ckpt_dir_abs, prefix, blob_abs


def _restore_state_from_checkpoint(state, checkpoint_path: str):
    ckpt_dir, ckpt_prefix, ckpt_blob = _resolve_ckpt_paths(checkpoint_path)
    if os.path.exists(ckpt_blob):
        with open(ckpt_blob, "rb") as f:
            raw = f.read()
        restored = serialization.msgpack_restore(raw)
        params, stats = merge_compatible_state(
            state.params,
            checkpoint_params_subtree(restored),
        )
        print(
            "Checkpoint param merge: "
            f"loaded={len(stats['loaded'])} missing={len(stats['missing'])} "
            f"mismatched={len(stats['mismatched'])} extra={len(stats['extra'])}",
            flush=True,
        )
        return state.replace(params=params), f"raw-msgpack:{ckpt_blob}"

    restored = checkpoints.restore_checkpoint(ckpt_dir, target=None, prefix=ckpt_prefix)
    if restored is None:
        restored = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    if restored is None:
        raise FileNotFoundError(
            f"Checkpoint not found from '{checkpoint_path}'. "
            f"Tried raw file '{ckpt_blob}' and Flax checkpoints with prefix '{ckpt_prefix}'."
        )
    params, stats = merge_compatible_state(
        state.params,
        checkpoint_params_subtree(restored),
    )
    print(
        "Checkpoint param merge: "
        f"loaded={len(stats['loaded'])} missing={len(stats['missing'])} "
        f"mismatched={len(stats['mismatched'])} extra={len(stats['extra'])}",
        flush=True,
    )
    return state.replace(params=params), f"flax:{ckpt_dir} (prefix={ckpt_prefix})"


def _build_tau_grid(args: argparse.Namespace) -> List[float]:
    vals: List[float] = []
    if args.taus:
        for token in str(args.taus).split(","):
            token = token.strip()
            if not token:
                continue
            vals.append(float(token))
    else:
        if args.tau_step <= 0:
            raise ValueError("--tau-step must be > 0.")
        if args.tau_end < args.tau_start:
            raise ValueError("--tau-end must be >= --tau-start.")
        cur = float(args.tau_start)
        end = float(args.tau_end)
        step = float(args.tau_step)
        while cur <= end + 1e-12:
            vals.append(cur)
            cur += step

    cleaned: List[float] = []
    for v in vals:
        vf = float(v)
        if not math.isfinite(vf):
            continue
        if vf < 0.0 or vf > 1.0:
            raise ValueError(f"Tau values must be in [0, 1], got {vf}.")
        cleaned.append(round(vf, 6))
    if not cleaned:
        raise ValueError("Tau grid is empty. Provide --taus or a valid start/end/step.")
    return sorted(set(cleaned))


def _safe_metric(v: Any) -> float:
    if v is None:
        return float("-inf")
    vf = float(v)
    if not math.isfinite(vf):
        return float("-inf")
    return vf


def _looks_like_oom_error(err: Exception) -> bool:
    txt = str(err).lower()
    return (
        "out of memory" in txt
        or "resource_exhausted" in txt
        or "cuda_error_out_of_memory" in txt
        or "allocator" in txt and "ran out of memory" in txt
    )


def _select_score(row: Dict[str, Any], select_by: str) -> float:
    if select_by == "micro_acc":
        return _safe_metric(row.get("micro_acc"))
    if select_by == "macro_f1":
        return _safe_metric(row.get("macro_f1"))
    if select_by == "weighted_f1":
        return _safe_metric(row.get("weighted_f1"))
    if select_by == "other_precision":
        return _safe_metric(row.get("other_precision"))
    if select_by == "other_recall":
        return _safe_metric(row.get("other_recall"))
    if select_by == "other_f1":
        return _safe_metric(row.get("other_f1"))
    raise ValueError(f"Unknown select metric '{select_by}'.")


def _format_float(v: Any, ndigits: int = 4) -> str:
    if v is None:
        return "n/a"
    vf = float(v)
    if not math.isfinite(vf):
        return "n/a"
    return f"{vf:.{ndigits}f}"


def _load_other_dataset(
    *,
    root: Path,
    split: str,
    label: str,
):
    if load_from_disk is None:
        return None, None
    ds_path = root / split / label / "dataset"
    if not ds_path.exists():
        return None, ds_path
    try:
        ds = load_from_disk(str(ds_path))
    except Exception:
        return None, ds_path
    return ds, ds_path


def _collect_other_predictions(
    *,
    state,
    dataset,
    chunk: int,
    batch_size: int,
    limit_rows: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    n_rows_total = int(len(dataset))
    if n_rows_total <= 0:
        return {"argmax": np.zeros((0,), dtype=np.int32), "max_prob": np.zeros((0,), dtype=np.float32), "rows": 0}

    row_indices = np.arange(n_rows_total, dtype=np.int64)
    if limit_rows is not None and limit_rows > 0 and limit_rows < n_rows_total:
        rng = np.random.default_rng(int(seed) ^ 0x0A117E)
        row_indices = rng.choice(row_indices, size=int(limit_rows), replace=False)

    windows: List[np.ndarray] = []
    lengths: List[int] = []
    rows_used = 0

    for ridx in row_indices:
        try:
            row = dataset[int(ridx)]
        except Exception:
            continue
        text = row.get("content") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text:
            continue
        raw = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8).astype(
            np.int32, copy=False
        )
        if raw.size <= 0:
            continue
        rows_used += 1
        if raw.size <= chunk:
            windows.append(raw)
            lengths.append(int(raw.size))
            continue
        start = 0
        while start < int(raw.size):
            seg = raw[start : start + chunk]
            if seg.size <= 0:
                break
            windows.append(seg)
            lengths.append(int(seg.size))
            start += chunk

    if not windows:
        return {"argmax": np.zeros((0,), dtype=np.int32), "max_prob": np.zeros((0,), dtype=np.float32), "rows": 0}

    other_id = int(getattr(cfg, "OTHER_CLASS_INDEX", cfg.NUM_CLASSES))
    pad_id = int(cfg.PAD_ID)
    pad_tok = int(cfg.PAD_BYTE_ID)

    all_argmax: List[np.ndarray] = []
    all_max_prob: List[np.ndarray] = []
    oom_skipped_windows = 0
    global_max_micro = max(1, int(batch_size))

    rng = jax.random.PRNGKey(int(seed) ^ 0x07A11E)
    for i in range(0, len(windows), batch_size):
        win_slice = windows[i : i + batch_size]
        len_slice = lengths[i : i + batch_size]
        b_total = len(win_slice)
        if b_total <= 0:
            continue
        start = 0
        max_micro = min(int(global_max_micro), int(b_total))
        while start < b_total:
            micro = min(max_micro, b_total - start)
            done = False
            while not done:
                cur_win = win_slice[start : start + micro]
                cur_len = len_slice[start : start + micro]
                xb = np.full((micro, chunk), pad_tok, dtype=np.int32)
                yb = np.full((micro, chunk), pad_id, dtype=np.int32)
                for j, (arr, n) in enumerate(zip(cur_win, cur_len)):
                    n_use = min(int(n), chunk)
                    if n_use <= 0:
                        continue
                    xb[j, :n_use] = arr[:n_use]
                    yb[j, :n_use] = other_id
                xb = sanitize_tokens(xb)
                rng, logits_rng = jax.random.split(rng)
                try:
                    logits = _forward_logits(
                        state,
                        jnp.array(xb, dtype=jnp.int32),
                        logits_rng,
                    )
                except Exception as err:
                    if not _looks_like_oom_error(err):
                        raise
                    if micro <= 1:
                        oom_skipped_windows += 1
                        print(
                            "WARNING: OOM on OTHER cache at microbatch=1; skipping one window "
                            f"(global_window_index={i + start}).",
                            flush=True,
                        )
                        start += 1
                        done = True
                        del xb, yb
                        try:
                            jax.clear_caches()
                        except Exception:
                            pass
                        gc.collect()
                        continue
                    old_micro = micro
                    new_micro = max(1, micro // 2)
                    if new_micro == micro:
                        new_micro = micro - 1
                    max_micro = min(max_micro, new_micro)
                    global_max_micro = min(global_max_micro, new_micro)
                    micro = new_micro
                    print(
                        "WARNING: OOM during OTHER cache forward; reducing microbatch "
                        f"{old_micro} -> {micro} and retrying.",
                        flush=True,
                    )
                    del xb, yb
                    try:
                        jax.clear_caches()
                    except Exception:
                        pass
                    gc.collect()
                    continue

                logits_np = np.asarray(logits, dtype=np.float32)
                logits_shift = logits_np - logits_np.max(axis=-1, keepdims=True)
                probs = np.exp(logits_shift)
                probs /= np.maximum(probs.sum(axis=-1, keepdims=True), 1e-9)

                argmax = np.argmax(probs, axis=-1).astype(np.int32)
                max_prob = probs.max(axis=-1).astype(np.float32)
                valid_mask = _valid_metric_mask(yb, xb)

                for j in range(micro):
                    m = valid_mask[j]
                    if not m.any():
                        continue
                    all_argmax.append(argmax[j][m])
                    all_max_prob.append(max_prob[j][m])

                start += micro
                done = True
                del xb, yb, logits, logits_np, logits_shift, probs, argmax, max_prob, valid_mask

    if not all_argmax:
        return {
            "argmax": np.zeros((0,), dtype=np.int32),
            "max_prob": np.zeros((0,), dtype=np.float32),
            "rows": rows_used,
            "oom_skipped_windows": int(oom_skipped_windows),
        }

    return {
        "argmax": np.concatenate(all_argmax, axis=0).astype(np.int32, copy=False),
        "max_prob": np.concatenate(all_max_prob, axis=0).astype(np.float32, copy=False),
        "rows": int(rows_used),
        "oom_skipped_windows": int(oom_skipped_windows),
    }


def _augment_confusion_with_other(
    *,
    conf: np.ndarray,
    other_cache: Dict[str, Any],
    tau: float,
) -> np.ndarray:
    num_classes = int(cfg.NUM_CLASSES)
    other_id = int(getattr(cfg, "OTHER_CLASS_INDEX", num_classes))
    out = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    if conf.shape[0] == num_classes + 1:
        out[:, :] = conf.astype(np.int64, copy=False)
    else:
        out[:num_classes, :num_classes] = conf.astype(np.int64, copy=False)

    argmax = np.asarray(other_cache.get("argmax"), dtype=np.int32)
    max_prob = np.asarray(other_cache.get("max_prob"), dtype=np.float32)
    if argmax.size <= 0 or max_prob.size <= 0:
        return out
    preds = argmax.copy()
    preds[max_prob < float(tau)] = other_id
    counts = np.bincount(preds, minlength=num_classes + 1).astype(np.int64, copy=False)
    out[other_id, : num_classes + 1] += counts[: num_classes + 1]
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep open-set tau on monitor validation memmap and pick the best threshold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Model checkpoint path (.msgpack or Flax checkpoint prefix/dir).",
    )
    parser.add_argument(
        "--monitor-root",
        type=str,
        default=str(REPO_ROOT / "downloader" / "monitor_preprocessed_b"),
        help="Path to monitor memmap root (default: downloader/monitor_preprocessed_b).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4096,
        help="Max monitor files to sample (<=0 uses all).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=int(cfg.MODEL_WINDOW_BYTES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--other-root",
        type=str,
        default=str(REPO_ROOT / "downloader" / "arrow_out_other"),
        help="Root for extracted OTHER Arrow data (<root>/<split>/<label>/dataset).",
    )
    parser.add_argument(
        "--other-split",
        type=str,
        default="train",
        help="Split under --other-root for OTHER augmentation.",
    )
    parser.add_argument(
        "--other-label",
        type=str,
        default="other",
        help="Label directory under --other-root for OTHER augmentation.",
    )
    parser.add_argument(
        "--other-limit",
        type=int,
        default=1024,
        help="Max OTHER rows to sample from --other-root (<=0 disables OTHER augmentation).",
    )

    parser.add_argument(
        "--taus",
        type=str,
        default="",
        help="Explicit comma-separated tau values (overrides start/end/step).",
    )
    parser.add_argument("--tau-start", type=float, default=0.05)
    parser.add_argument("--tau-end", type=float, default=0.95)
    parser.add_argument("--tau-step", type=float, default=0.05)
    parser.add_argument(
        "--select-by",
        choices=(
            "macro_f1",
            "micro_acc",
            "weighted_f1",
            "other_precision",
            "other_recall",
            "other_f1",
        ),
        default="macro_f1",
        help="Metric used to pick best tau.",
    )

    parser.add_argument("--arch", type=str, choices=("unet1d", "mamba"), default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)

    parser.add_argument("--mamba-layers", type=int, default=None)
    parser.add_argument("--mamba-d-state", type=int, default=None)
    parser.add_argument("--mamba-expand", type=int, default=None)
    parser.add_argument("--mamba-dt-rank", type=int, default=None)
    parser.add_argument("--mamba-conv", type=int, default=None)
    parser.add_argument(
        "--mamba-bidirectional",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional CSV output path for per-tau results.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional JSON output path for full report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many best tau rows to print in summary.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    if args.chunk <= 0:
        raise ValueError("--chunk must be > 0.")

    tau_grid = _build_tau_grid(args)
    ckpt_path = Path(args.checkpoint).resolve()

    wandb_hp = _load_wandb_hparams(ckpt_path)
    inferred_hp = _infer_hparams_from_checkpoint(ckpt_path)

    arch, arch_src = _resolve_hparam(
        args.arch,
        wandb_hp.get("arch"),
        inferred_hp.get("arch"),
        "unet1d",
    )
    arch = str(arch).lower().strip()

    model_dim, model_dim_src = _resolve_hparam(
        args.model_dim,
        wandb_hp.get("model_dim"),
        inferred_hp.get("model_dim"),
        256,
    )
    model_dim = int(model_dim)

    dtype_name, dtype_src = _resolve_hparam(
        args.dtype,
        wandb_hp.get("dtype"),
        inferred_hp.get("dtype"),
        "bfloat16",
    )
    dtype = _parse_dtype(str(dtype_name))

    channels_cli = _parse_channels(args.channels)
    channels_val, channels_src = _resolve_hparam(
        channels_cli,
        wandb_hp.get("channels"),
        inferred_hp.get("channels"),
        [128, 256, 384, 512],
    )
    channels = tuple(int(x) for x in channels_val)

    mamba_layers, _ = _resolve_hparam(
        args.mamba_layers,
        wandb_hp.get("mamba_layers"),
        inferred_hp.get("mamba_layers"),
        6,
    )
    mamba_d_state, _ = _resolve_hparam(
        args.mamba_d_state,
        wandb_hp.get("mamba_d_state"),
        inferred_hp.get("mamba_d_state"),
        8,
    )
    mamba_expand, _ = _resolve_hparam(
        args.mamba_expand,
        wandb_hp.get("mamba_expand"),
        inferred_hp.get("mamba_expand"),
        1,
    )
    mamba_dt_rank, _ = _resolve_hparam(
        args.mamba_dt_rank,
        wandb_hp.get("mamba_dt_rank"),
        inferred_hp.get("mamba_dt_rank"),
        16,
    )
    mamba_conv, _ = _resolve_hparam(
        args.mamba_conv,
        wandb_hp.get("mamba_conv"),
        inferred_hp.get("mamba_conv"),
        4,
    )
    mamba_bidirectional, _ = _resolve_hparam(
        args.mamba_bidirectional,
        wandb_hp.get("mamba_bidirectional"),
        inferred_hp.get("mamba_bidirectional"),
        True,
    )

    train_cfg = TrainConfig(
        steps=2,
        lr=1e-3,
        weight_decay=0.01,
        warmup=1,
        accum_steps=1,
        dtype=dtype,
        arch=arch,
        model_dim=int(model_dim),
        channels=tuple(channels),
        dropout_rate=0.0,
        mamba_layers=int(mamba_layers),
        mamba_d_state=int(mamba_d_state),
        mamba_expand=int(mamba_expand),
        mamba_dt_rank=int(mamba_dt_rank),
        mamba_conv=int(mamba_conv),
        mamba_bidirectional=bool(mamba_bidirectional),
    )

    print(
        "Resolved model config: "
        f"arch={arch}({arch_src}) model_dim={model_dim}({model_dim_src}) "
        f"dtype={str(dtype_name).rsplit('.', 1)[-1]}({dtype_src}) "
        f"channels={list(channels)}({channels_src})",
        flush=True,
    )
    if arch == "mamba":
        print(
            "Mamba config: "
            f"layers={int(mamba_layers)} d_state={int(mamba_d_state)} "
            f"expand={int(mamba_expand)} dt_rank={int(mamba_dt_rank)} "
            f"conv={int(mamba_conv)} bidirectional={bool(mamba_bidirectional)}",
            flush=True,
        )

    print("Creating model state...", flush=True)
    init_rng = jax.random.PRNGKey(0)
    state = create_train_state(init_rng, train_cfg, cfg.NUM_CLASSES)

    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    state, restore_src = _restore_state_from_checkpoint(state, str(ckpt_path))
    print(f"Restored parameters via {restore_src}", flush=True)

    monitor_root = Path(args.monitor_root).resolve()
    print(f"Loading monitor memmap: {monitor_root}", flush=True)
    monitor_data = load_monitor_memmaps(monitor_root)
    total_files = int(monitor_data.get("meta", {}).get("num_files", len(monitor_data["files"])))
    limit = None if int(args.limit) <= 0 else int(args.limit)
    print(
        f"Monitor files available: {total_files}, evaluation limit: {limit if limit is not None else 'all'}",
        flush=True,
    )
    print(f"Tau grid ({len(tau_grid)}): {tau_grid}", flush=True)

    other_cache: Optional[Dict[str, Any]] = None
    other_ds = None
    other_ds_path = None
    if int(args.other_limit) > 0:
        other_root = Path(args.other_root).resolve()
        other_ds, other_ds_path = _load_other_dataset(
            root=other_root,
            split=str(args.other_split),
            label=str(args.other_label),
        )
        if other_ds is None:
            print(
                f"⚠️  OTHER augmentation disabled: dataset not found/loadable at {other_ds_path}",
                flush=True,
            )
        else:
            print(
                f"Loading OTHER augmentation dataset: {other_ds_path} (rows={len(other_ds)})",
                flush=True,
            )
            other_cache = _collect_other_predictions(
                state=state,
                dataset=other_ds,
                chunk=int(args.chunk),
                batch_size=int(args.batch_size),
                limit_rows=int(args.other_limit),
                seed=int(args.seed),
            )
            print(
                "Prepared OTHER augmentation cache: "
                f"rows_used={int(other_cache.get('rows', 0))}, "
                f"valid_tokens={int(np.asarray(other_cache.get('argmax')).size)}, "
                f"oom_skipped_windows={int(other_cache.get('oom_skipped_windows', 0))}",
                flush=True,
            )
    else:
        print("OTHER augmentation disabled (--other-limit <= 0).", flush=True)

    base_rng = jax.random.PRNGKey(int(args.seed))
    results: List[Dict[str, Any]] = []
    current_eval_batch_size = max(1, int(args.batch_size))

    for tau in tau_grid:
        while True:
            try:
                stats = evaluate_monitor_set(
                    state=state,
                    monitor_data=monitor_data,
                    L=int(args.chunk),
                    batch_size=int(current_eval_batch_size),
                    rng=base_rng,
                    limit=limit,
                    eval_step_fn=eval_step,
                    other_threshold=float(tau),
                )
                break
            except Exception as err:
                if not _looks_like_oom_error(err):
                    raise
                if current_eval_batch_size <= 1:
                    raise
                next_bs = max(1, current_eval_batch_size // 2)
                print(
                    "WARNING: OOM during monitor eval; reducing eval batch size "
                    f"{current_eval_batch_size} -> {next_bs} and retrying tau={tau:.4f}.",
                    flush=True,
                )
                current_eval_batch_size = next_bs
                try:
                    jax.clear_caches()
                except Exception:
                    pass
                gc.collect()
                continue
        cm_thresh = stats.get("conf_thresh")
        if cm_thresh is not None:
            conf = cm_thresh
            conf_classes = cfg.NUM_CLASSES + 1
            thresholded = True
        else:
            conf = stats["conf_mat"]
            conf_classes = cfg.NUM_CLASSES
            thresholded = False

        if other_cache is not None:
            conf = _augment_confusion_with_other(
                conf=conf,
                other_cache=other_cache,
                tau=float(tau),
            )
            conf_classes = cfg.NUM_CLASSES + 1

        per_class, aggregates = compute_metrics_from_confusion(
            conf, conf_classes, cfg.PAD_ID
        )
        other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
        other_prec = None
        other_rec = None
        other_f1 = None
        other_support = None
        if (
            other_idx is not None
            and isinstance(other_idx, int)
            and 0 <= other_idx < conf_classes
        ):
            other_prec = float(per_class["precision"][other_idx])
            other_rec = float(per_class["recall"][other_idx])
            other_f1 = float(per_class["f1"][other_idx])
            other_support = int(per_class["support"][other_idx])

        row = {
            "tau": float(tau),
            "thresholded": bool(thresholded),
            "eval_batch_size": int(current_eval_batch_size),
            "loss_mean": float(stats.get("loss_mean", 0.0)),
            "acc_mean": float(stats.get("acc_mean", 0.0)),
            "acc_thresh_mean": (
                float(stats["acc_thresh_mean"])
                if stats.get("acc_thresh_mean") is not None
                else None
            ),
            "micro_acc": float(aggregates["micro"]["acc"]),
            "macro_precision": float(aggregates["macro"]["precision"]),
            "macro_recall": float(aggregates["macro"]["recall"]),
            "macro_f1": float(aggregates["macro"]["f1"]),
            "weighted_f1": float(aggregates["weighted"]["f1"]),
            "other_precision": other_prec,
            "other_recall": other_rec,
            "other_f1": other_f1,
            "other_support": other_support,
            "files_used": int(stats.get("files_used", 0)),
            "windows": int(stats.get("windows", 0)),
            "skipped": int(stats.get("skipped", 0)),
            "score": None,
        }
        row["score"] = _select_score(row, args.select_by)
        results.append(row)
        print(
            f"tau={tau:.4f} score={_format_float(row['score'])} "
            f"eval_bs={int(row['eval_batch_size'])} "
            f"micro_acc={_format_float(row['micro_acc'])} "
            f"macro_f1={_format_float(row['macro_f1'])} "
            f"other_f1={_format_float(row['other_f1'])}",
            flush=True,
        )

    ranked = sorted(
        results,
        key=lambda r: (_select_score(r, args.select_by), -float(r["tau"])),
        reverse=True,
    )
    best = ranked[0]

    print("\nTop tau candidates:", flush=True)
    print(
        f"{'rank':>4} {'tau':>8} {'score':>10} {'micro_acc':>10} {'macro_f1':>10} "
        f"{'other_f1':>10} {'other_rec':>10} {'other_prec':>10}",
        flush=True,
    )
    for idx, row in enumerate(ranked[: max(1, int(args.top_k))], start=1):
        print(
            f"{idx:>4d} "
            f"{row['tau']:>8.4f} "
            f"{_format_float(row['score']):>10} "
            f"{_format_float(row['micro_acc']):>10} "
            f"{_format_float(row['macro_f1']):>10} "
            f"{_format_float(row['other_f1']):>10} "
            f"{_format_float(row['other_recall']):>10} "
            f"{_format_float(row['other_precision']):>10}",
            flush=True,
        )

    print("\nSelected tau:", flush=True)
    print(
        f"tau={best['tau']:.4f} by {args.select_by}={_format_float(best['score'])} "
        f"(micro_acc={_format_float(best['micro_acc'])}, "
        f"macro_f1={_format_float(best['macro_f1'])}, "
        f"other_f1={_format_float(best['other_f1'])})",
        flush=True,
    )

    if args.out_csv:
        out_csv = Path(args.out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "tau",
            "score",
            "thresholded",
            "eval_batch_size",
            "micro_acc",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1",
            "other_precision",
            "other_recall",
            "other_f1",
            "other_support",
            "loss_mean",
            "acc_mean",
            "acc_thresh_mean",
            "files_used",
            "windows",
            "skipped",
        ]
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(results, key=lambda r: float(r["tau"])):
                writer.writerow({k: row.get(k) for k in fieldnames})
        print(f"Wrote CSV: {out_csv}", flush=True)

    if args.out_json:
        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(ckpt_path),
            "monitor_root": str(monitor_root),
            "select_by": args.select_by,
            "best": best,
            "taus": tau_grid,
            "results": sorted(results, key=lambda r: float(r["tau"])),
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {out_json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
