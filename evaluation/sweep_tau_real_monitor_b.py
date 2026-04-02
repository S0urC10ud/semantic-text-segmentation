#!/usr/bin/env python3
"""
Sweep open-set tau on real monitor_b data using a deterministic full-file subset.

This differs from evaluation/sweep_tau.py in two important ways:
  1) it uses only the real labels already present in monitor_preprocessed_b
  2) it reports binary open-set counts for OTHER-vs-known tokens:
     TP, TN, FP, FN in absolute token counts

The binary framing is:
  - positive: truth label == "other"
  - predicted positive: max softmax < tau (or an out-of-range prediction)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from flax import serialization

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg
from utils.metrics_helper import _valid_metric_mask
from utils.monitor_eval import load_monitor_memmaps

DEFAULT_MAMBA_CHUNK = 10000


def _load_repo_evaluation_module():
    module_path = REPO_ROOT / "evaluation" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("_repo_evaluation_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluation module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_EVALMOD = _load_repo_evaluation_module()
DEFAULT_CHANNELS = _EVALMOD.DEFAULT_CHANNELS
SegmenterRunner = _EVALMOD.SegmenterRunner
_apply_label_mapping = _EVALMOD._apply_label_mapping
_load_checkpoint_hparams = _EVALMOD._load_checkpoint_hparams


def _parse_channels(channels: Optional[str]) -> Optional[List[int]]:
    if channels is None:
        return None
    parsed = [int(part.strip()) for part in str(channels).split(",") if part.strip()]
    if not parsed:
        raise ValueError("--channels must contain at least one integer.")
    return parsed


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

    top_keys = {str(key) for key in params.keys()}
    is_mamba = any(
        key.startswith("MambaBlock1D_") or key.startswith("CheckpointMambaBlock1D_")
        for key in top_keys
    )
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
            match = re.match(r"^(?:Checkpoint)?MambaBlock1D_(\d+)$", key)
            if match:
                indices.append(int(match.group(1)))
        if indices:
            result["mamba_layers"] = int(max(indices) + 1)

        block0 = params.get("MambaBlock1D_0")
        if not isinstance(block0, Mapping):
            block0 = params.get("CheckpointMambaBlock1D_0")
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


def _resolve_hparam(cli_value: Any, auto_value: Any, inferred_value: Any, default_value: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if auto_value is not None:
        return auto_value
    if inferred_value is not None:
        return inferred_value
    return default_value


def _build_tau_grid(args: argparse.Namespace) -> List[float]:
    values: List[float] = []
    if args.taus:
        for token in str(args.taus).split(","):
            token = token.strip()
            if token:
                values.append(float(token))
    else:
        if float(args.tau_step) <= 0.0:
            raise ValueError("--tau-step must be > 0.")
        if float(args.tau_end) < float(args.tau_start):
            raise ValueError("--tau-end must be >= --tau-start.")
        current = float(args.tau_start)
        end = float(args.tau_end)
        step = float(args.tau_step)
        while current <= end + 1e-12:
            values.append(current)
            current += step

    cleaned: List[float] = []
    for value in values:
        value_f = round(float(value), 6)
        if not math.isfinite(value_f):
            continue
        if not (0.0 <= value_f <= 1.0):
            raise ValueError(f"Tau values must be in [0, 1], got {value_f}.")
        cleaned.append(value_f)
    if not cleaned:
        raise ValueError("Tau grid is empty. Provide --taus or a valid start/end/step.")
    return sorted(set(cleaned))


def select_monitor_file_indices(
    num_files: int,
    limit_files: int,
    subset_seed: int,
) -> np.ndarray:
    if num_files <= 0:
        return np.zeros((0,), dtype=np.int64)
    indices = np.arange(int(num_files), dtype=np.int64)
    if int(limit_files) <= 0 or int(limit_files) >= int(num_files):
        return indices
    rng = np.random.default_rng(int(subset_seed))
    chosen = rng.choice(indices, size=int(limit_files), replace=False)
    chosen.sort()
    return chosen.astype(np.int64, copy=False)


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    if int(denominator) <= 0:
        return None
    value = float(numerator) / float(denominator)
    if math.isfinite(value):
        return value
    return None


def _safe_f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom <= 0.0:
        return None
    value = 2.0 * precision * recall / denom
    if math.isfinite(value):
        return value
    return None


def _format_float(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value_f = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(value_f):
        return "n/a"
    return f"{value_f:.{ndigits}f}"


def _select_score(row: Mapping[str, Any], select_by: str) -> float:
    value = row.get(select_by)
    if value is None:
        return float("-inf")
    value_f = float(value)
    if math.isfinite(value_f):
        return value_f
    return float("-inf")


def _resolve_runner_args(
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ckpt_path = Path(args.checkpoint).resolve()
    auto_hparams = _load_checkpoint_hparams(ckpt_path)
    inferred_hparams = _infer_hparams_from_checkpoint(ckpt_path)

    label_names = auto_hparams.get("label_names")
    if label_names:
        _apply_label_mapping(label_names)

    arch = str(
        _resolve_hparam(
            args.arch,
            auto_hparams.get("arch"),
            inferred_hparams.get("arch"),
            "unet1d",
        )
    ).lower().strip()
    model_dim = int(
        _resolve_hparam(
            args.model_dim,
            auto_hparams.get("model_dim"),
            inferred_hparams.get("model_dim"),
            256,
        )
    )
    dtype = str(
        _resolve_hparam(
            args.dtype,
            auto_hparams.get("dtype"),
            inferred_hparams.get("dtype"),
            "bfloat16",
        )
    ).rsplit(".", 1)[-1]
    chunk = int(
        args.chunk
        if args.chunk is not None
        else (DEFAULT_MAMBA_CHUNK if arch == "mamba" else int(cfg.MODEL_WINDOW_BYTES))
    )

    channels_cli = _parse_channels(args.channels)
    if channels_cli is not None:
        channels = [int(ch) for ch in channels_cli]
    else:
        channels = [
            int(ch)
            for ch in _resolve_hparam(
                None,
                auto_hparams.get("channels"),
                inferred_hparams.get("channels"),
                DEFAULT_CHANNELS,
            )
        ]

    resolved: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "arch": arch,
        "model_dim": model_dim,
        "dtype": dtype,
        "chunk": chunk,
        "channels": [int(ch) for ch in channels],
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "inference_backend": str(args.inference_backend),
        "mamba_layers": int(
            _resolve_hparam(
                args.mamba_layers,
                auto_hparams.get("mamba_layers"),
                inferred_hparams.get("mamba_layers"),
                6,
            )
        ),
        "mamba_d_state": int(
            _resolve_hparam(
                args.mamba_d_state,
                auto_hparams.get("mamba_d_state"),
                inferred_hparams.get("mamba_d_state"),
                8,
            )
        ),
        "mamba_expand": int(
            _resolve_hparam(
                args.mamba_expand,
                auto_hparams.get("mamba_expand"),
                inferred_hparams.get("mamba_expand"),
                1,
            )
        ),
        "mamba_dt_rank": int(
            _resolve_hparam(
                args.mamba_dt_rank,
                auto_hparams.get("mamba_dt_rank"),
                inferred_hparams.get("mamba_dt_rank"),
                16,
            )
        ),
        "mamba_conv": int(
            _resolve_hparam(
                args.mamba_conv,
                auto_hparams.get("mamba_conv"),
                inferred_hparams.get("mamba_conv"),
                4,
            )
        ),
        "mamba_bidirectional": bool(
            _resolve_hparam(
                args.mamba_bidirectional,
                auto_hparams.get("mamba_bidirectional"),
                inferred_hparams.get("mamba_bidirectional"),
                True,
            )
        ),
    }

    runner_kwargs = dict(resolved)
    runner_kwargs.pop("checkpoint")
    return resolved, runner_kwargs


def accumulate_open_set_binary_counts(
    counts: Dict[str, np.ndarray],
    *,
    truth_is_other: np.ndarray,
    max_prob: np.ndarray,
    tau_grid: Sequence[float],
    base_pred_is_other: Optional[np.ndarray] = None,
) -> None:
    truth_is_other = np.asarray(truth_is_other, dtype=np.bool_)
    max_prob = np.asarray(max_prob, dtype=np.float32)
    if truth_is_other.ndim != 1 or max_prob.ndim != 1:
        raise ValueError("truth_is_other and max_prob must be 1D arrays.")
    if truth_is_other.shape[0] != max_prob.shape[0]:
        raise ValueError("truth_is_other and max_prob must have the same length.")

    if base_pred_is_other is None:
        base_pred_is_other = np.zeros_like(truth_is_other, dtype=np.bool_)
    else:
        base_pred_is_other = np.asarray(base_pred_is_other, dtype=np.bool_)
        if base_pred_is_other.shape != truth_is_other.shape:
            raise ValueError("base_pred_is_other must match truth_is_other shape.")

    truth_other = int(truth_is_other.sum())
    truth_non_other = int(truth_is_other.shape[0] - truth_other)

    for idx, tau in enumerate(tau_grid):
        pred_is_other = base_pred_is_other | (max_prob < float(tau))
        tp = int(np.sum(truth_is_other & pred_is_other))
        fp = int(np.sum((~truth_is_other) & pred_is_other))
        fn = truth_other - tp
        tn = truth_non_other - fp
        counts["tp_other"][idx] += tp
        counts["fp_other"][idx] += fp
        counts["fn_other"][idx] += fn
        counts["tn_other"][idx] += tn


def _build_rows_from_counts(
    *,
    tau_grid: Sequence[float],
    counts: Dict[str, np.ndarray],
    select_by: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, tau in enumerate(tau_grid):
        tp = int(counts["tp_other"][idx])
        fn = int(counts["fn_other"][idx])
        fp = int(counts["fp_other"][idx])
        tn = int(counts["tn_other"][idx])
        truth_other = tp + fn
        truth_non_other = fp + tn
        predicted_other = tp + fp
        predicted_non_other = fn + tn
        total = truth_other + truth_non_other
        precision = _safe_ratio(tp, predicted_other)
        recall = _safe_ratio(tp, truth_other)
        specificity = _safe_ratio(tn, truth_non_other)
        f1 = _safe_f1(precision, recall)
        balanced_acc = (
            None
            if recall is None or specificity is None
            else (float(recall) + float(specificity)) / 2.0
        )
        row = {
            "tau": float(tau),
            "tp_other": tp,
            "fn_other": fn,
            "fp_other": fp,
            "tn_other": tn,
            "truth_other": truth_other,
            "truth_non_other": truth_non_other,
            "predicted_other": predicted_other,
            "predicted_non_other": predicted_non_other,
            "total_tokens": total,
            "other_precision": precision,
            "other_recall": recall,
            "other_f1": f1,
            "specificity": specificity,
            "balanced_acc": balanced_acc,
            "predicted_other_rate": _safe_ratio(predicted_other, total),
            "truth_other_rate": _safe_ratio(truth_other, total),
            "score": None,
        }
        row["score"] = _select_score(row, select_by)
        rows.append(row)
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep open-set tau on a deterministic full-file subset of real monitor_b "
            "and report OTHER-vs-known TP/TN/FP/FN token counts."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "sweeps" / "sfullfiles3.msgpack"),
        help="Checkpoint path (.msgpack or Orbax directory).",
    )
    parser.add_argument(
        "--monitor-root",
        type=str,
        default=str(REPO_ROOT / "downloader" / "monitor_preprocessed_b"),
        help="Path to downloader/monitor_preprocessed_b.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=1024,
        help="How many monitor_b files to use (<=0 uses all files).",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=123,
        help="Seed used to choose the deterministic file subset.",
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
        type=str,
        choices=("other_f1", "other_precision", "other_recall", "balanced_acc", "specificity"),
        default="other_f1",
        help="Metric used to rank tau values.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top tau rows to print in the summary table.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Progress log interval in processed files.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--inference-backend",
        type=str,
        default="auto",
        choices=("auto", "fast", "legacy"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Inference chunk size. Defaults to 10000 for Mamba and 1536 for U-Net.",
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
    parser.add_argument("--out-csv", type=str, default="")
    parser.add_argument("--out-json", type=str, default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tau_grid = _build_tau_grid(args)
    monitor_root = Path(args.monitor_root).resolve()

    resolved_runner, runner_kwargs = _resolve_runner_args(args)
    ckpt_path = Path(resolved_runner["checkpoint"]).resolve()
    print(
        "Resolved model config: "
        f"checkpoint={ckpt_path} "
        f"arch={resolved_runner['arch']} "
        f"model_dim={resolved_runner['model_dim']} "
        f"dtype={resolved_runner['dtype']} "
        f"chunk={resolved_runner['chunk']} "
        f"batch_size={resolved_runner['batch_size']}",
        flush=True,
    )
    if str(resolved_runner["arch"]) == "mamba":
        print(
            "Resolved Mamba config: "
            f"layers={resolved_runner['mamba_layers']} "
            f"d_state={resolved_runner['mamba_d_state']} "
            f"expand={resolved_runner['mamba_expand']} "
            f"dt_rank={resolved_runner['mamba_dt_rank']} "
            f"conv={resolved_runner['mamba_conv']} "
            f"bidirectional={bool(resolved_runner['mamba_bidirectional'])}",
            flush=True,
        )

    print(f"Loading monitor memmap: {monitor_root}", flush=True)
    monitor_data = load_monitor_memmaps(monitor_root)
    files = monitor_data["files"]
    segments = monitor_data["segments"]
    contents = monitor_data["contents"]
    total_files_available = int(len(files))
    subset_indices = select_monitor_file_indices(
        total_files_available,
        int(args.limit_files),
        int(args.subset_seed),
    )
    subset_preview = subset_indices[: min(12, int(subset_indices.shape[0]))].tolist()
    print(
        f"Using monitor_b subset: files_selected={int(subset_indices.shape[0])}/{total_files_available} "
        f"subset_seed={int(args.subset_seed)} preview_indices={subset_preview}",
        flush=True,
    )
    print(f"Tau grid ({len(tau_grid)}): {tau_grid}", flush=True)

    other_id = getattr(cfg, "OTHER_CLASS_INDEX", None)
    if other_id is None:
        raise RuntimeError("cfg.OTHER_CLASS_INDEX is not configured.")

    print("Creating inference runner...", flush=True)
    runner = SegmenterRunner(
        resolved_runner["checkpoint"],
        **runner_kwargs,
    )
    use_fast_path = str(getattr(runner, "arch", "unet1d")).lower().strip() == "mamba"
    print(
        f"Inference path: {'full_file_auto_with_stream_fallback' if use_fast_path else 'sliding_window_legacy'}",
        flush=True,
    )

    counts = {
        "tp_other": np.zeros((len(tau_grid),), dtype=np.int64),
        "fn_other": np.zeros((len(tau_grid),), dtype=np.int64),
        "fp_other": np.zeros((len(tau_grid),), dtype=np.int64),
        "tn_other": np.zeros((len(tau_grid),), dtype=np.int64),
    }
    files_used = 0
    skipped = 0
    raw_bytes = 0
    valid_tokens = 0
    truth_other_tokens = 0
    started_at = time.perf_counter()

    for position, file_idx in enumerate(subset_indices, start=1):
        row = files[int(file_idx)]
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
            seg_s = int(seg["start"])
            seg_e = int(seg["end"])
            if seg_e <= seg_s:
                continue
            truth[seg_s:seg_e] = int(seg["label"])

        if use_fast_path:
            pred, probs = runner._segment_bytes(file_bytes)
        else:
            pred, probs = runner._segment_bytes_legacy(file_bytes)

        pred = np.asarray(pred, dtype=np.int32).reshape(-1)
        probs = np.asarray(probs, dtype=np.float32)
        if int(pred.shape[0]) != byte_len:
            raise RuntimeError(
                f"Prediction length mismatch for file {int(file_idx)}: expected {byte_len}, got {int(pred.shape[0])}."
            )
        if probs.ndim != 2 or int(probs.shape[0]) != byte_len:
            raise RuntimeError(
                f"Probability shape mismatch for file {int(file_idx)}: expected ({byte_len}, C), got {tuple(int(x) for x in probs.shape)}."
            )

        metric_mask = _valid_metric_mask(truth, file_bytes.astype(np.int32, copy=False))
        truth_i32 = truth.astype(np.int32, copy=False)
        pred_i32 = pred.astype(np.int32, copy=False)
        core_mask = metric_mask & (truth_i32 <= int(other_id))
        if not np.any(core_mask):
            files_used += 1
            raw_bytes += int(byte_len)
            continue

        truth_valid = truth_i32[core_mask]
        pred_valid = pred_i32[core_mask]
        max_prob_valid = probs.max(axis=-1)[core_mask]
        truth_is_other = truth_valid == int(other_id)
        base_pred_is_other = (pred_valid < 0) | (pred_valid >= int(cfg.NUM_CLASSES))

        accumulate_open_set_binary_counts(
            counts,
            truth_is_other=truth_is_other,
            max_prob=max_prob_valid,
            tau_grid=tau_grid,
            base_pred_is_other=base_pred_is_other,
        )

        files_used += 1
        raw_bytes += int(byte_len)
        valid_tokens += int(truth_valid.shape[0])
        truth_other_tokens += int(truth_is_other.sum())

        if int(args.log_every) > 0 and (
            position == 1
            or position % int(args.log_every) == 0
            or position == int(subset_indices.shape[0])
        ):
            elapsed = time.perf_counter() - started_at
            rate = float(position) / elapsed if elapsed > 0.0 else 0.0
            print(
                f"[progress] {position}/{int(subset_indices.shape[0])} files "
                f"processed={files_used} skipped={skipped} valid_tokens={valid_tokens} "
                f"truth_other_tokens={truth_other_tokens} files_per_sec={rate:.2f}",
                flush=True,
            )

    rows = _build_rows_from_counts(
        tau_grid=tau_grid,
        counts=counts,
        select_by=str(args.select_by),
    )
    ranked = sorted(
        rows,
        key=lambda row: (_select_score(row, str(args.select_by)), -float(row["tau"])),
        reverse=True,
    )
    best = ranked[0]

    print("\nTop tau candidates:", flush=True)
    print(
        f"{'rank':>4} {'tau':>8} {'score':>10} {'f1':>10} {'recall':>10} {'prec':>10} "
        f"{'TP':>10} {'FN':>10} {'FP':>10} {'TN':>10}",
        flush=True,
    )
    for rank, row in enumerate(ranked[: max(1, int(args.top_k))], start=1):
        print(
            f"{rank:>4d} "
            f"{float(row['tau']):>8.4f} "
            f"{_format_float(row['score']):>10} "
            f"{_format_float(row['other_f1']):>10} "
            f"{_format_float(row['other_recall']):>10} "
            f"{_format_float(row['other_precision']):>10} "
            f"{int(row['tp_other']):>10d} "
            f"{int(row['fn_other']):>10d} "
            f"{int(row['fp_other']):>10d} "
            f"{int(row['tn_other']):>10d}",
            flush=True,
        )

    print("\nSelected tau:", flush=True)
    print(
        f"tau={float(best['tau']):.4f} by {args.select_by}={_format_float(best['score'])} "
        f"(TP={int(best['tp_other'])}, FN={int(best['fn_other'])}, "
        f"FP={int(best['fp_other'])}, TN={int(best['tn_other'])}, "
        f"precision={_format_float(best['other_precision'])}, "
        f"recall={_format_float(best['other_recall'])}, "
        f"f1={_format_float(best['other_f1'])})",
        flush=True,
    )

    payload = {
        "checkpoint": str(ckpt_path),
        "monitor_root": str(monitor_root),
        "subset_seed": int(args.subset_seed),
        "limit_files": int(args.limit_files),
        "files_total_available": int(total_files_available),
        "files_selected": int(subset_indices.shape[0]),
        "files_used": int(files_used),
        "skipped": int(skipped),
        "raw_bytes": int(raw_bytes),
        "valid_tokens": int(valid_tokens),
        "truth_other_tokens": int(truth_other_tokens),
        "truth_other_rate": _safe_ratio(int(truth_other_tokens), int(valid_tokens)),
        "select_by": str(args.select_by),
        "taus": [float(tau) for tau in tau_grid],
        "subset_file_indices": [int(idx) for idx in subset_indices.tolist()],
        "runner": {
            "arch": str(resolved_runner["arch"]),
            "model_dim": int(resolved_runner["model_dim"]),
            "dtype": str(resolved_runner["dtype"]),
            "chunk": int(resolved_runner["chunk"]),
            "batch_size": int(resolved_runner["batch_size"]),
            "device": str(resolved_runner["device"]),
            "inference_backend": str(resolved_runner["inference_backend"]),
            "channels": [int(ch) for ch in resolved_runner["channels"]],
            "mamba_layers": int(resolved_runner["mamba_layers"]),
            "mamba_d_state": int(resolved_runner["mamba_d_state"]),
            "mamba_expand": int(resolved_runner["mamba_expand"]),
            "mamba_dt_rank": int(resolved_runner["mamba_dt_rank"]),
            "mamba_conv": int(resolved_runner["mamba_conv"]),
            "mamba_bidirectional": bool(resolved_runner["mamba_bidirectional"]),
        },
        "best": best,
        "results": sorted(rows, key=lambda row: float(row["tau"])),
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "counts_note": (
            "TP/TN/FP/FN are token counts for binary open-set detection on valid monitor tokens: "
            "positive means truth label == 'other'; predicted positive means max softmax < tau."
        ),
    }

    if args.out_csv:
        out_csv = Path(args.out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "tau",
            "score",
            "other_precision",
            "other_recall",
            "other_f1",
            "specificity",
            "balanced_acc",
            "tp_other",
            "fn_other",
            "fp_other",
            "tn_other",
            "truth_other",
            "truth_non_other",
            "predicted_other",
            "predicted_non_other",
            "total_tokens",
            "predicted_other_rate",
            "truth_other_rate",
        ]
        with out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows, key=lambda item: float(item["tau"])):
                writer.writerow({name: row.get(name) for name in fieldnames})
        print(f"Wrote CSV: {out_csv}", flush=True)

    if args.out_json:
        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {out_json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
