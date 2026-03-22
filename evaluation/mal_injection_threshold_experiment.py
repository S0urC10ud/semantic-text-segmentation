#!/usr/bin/env python3
"""Standalone threshold sweeps for the mal_injection benchmark.

This script intentionally lives outside the main evaluator so we can experiment
with open-set threshold policies for payload-like classes without changing the
default evaluation flow.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import datasets as hfds
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluation as evalmod  # noqa: E402


DEFAULT_GLOBAL_THRESHOLDS = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 0.9)
DEFAULT_LABEL_THRESHOLDS = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85)


@dataclass
class CachedExample:
    payload_lang: str
    host_lang: str
    payload_idx: int
    host_idx: int
    truth_chars: int
    truth_mask: np.ndarray
    raw_pred_idx: np.ndarray
    max_prob: np.ndarray


def _parse_threshold_grid(raw: Optional[str], default: Sequence[float]) -> List[float]:
    if raw is None or not str(raw).strip():
        values = list(default)
    else:
        values = []
        for piece in str(raw).split(","):
            piece = piece.strip()
            if not piece:
                continue
            value = float(piece)
            if value < 0.0:
                raise ValueError(f"Thresholds must be >= 0, got {value}.")
            values.append(value)
    uniq = sorted({round(float(v), 6) for v in values})
    return [float(v) for v in uniq]


def _parse_channels(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    parts = [piece.strip() for piece in str(raw).split(",")]
    values = [int(piece) for piece in parts if piece]
    return values or None


def _format_pct(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_float(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator:
        value = float(numerator) / float(denominator)
        if math.isfinite(value):
            return value
    return None


def _format_hits(hits: int, total: int) -> str:
    return f"{int(hits)}/{int(total)}"


def _score_tuple(summary: Mapping[str, Any]) -> Tuple[float, float, float, float, float, float]:
    payload = summary["payload"]
    any_payload = summary["any"]
    return (
        float(payload.get("det_rate@0.5") or 0.0),
        float(payload.get("mean_iou") or 0.0),
        float(payload.get("coverage") or 0.0),
        float(any_payload.get("det_rate@0.5") or 0.0),
        float(any_payload.get("mean_iou") or 0.0),
        float(any_payload.get("coverage") or 0.0),
    )


def _normalize_policy(
    base_threshold: float,
    overrides: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    cleaned = {
        str(label): float(value)
        for label, value in sorted((overrides or {}).items())
        if float(value) != float(base_threshold)
    }
    return {"base_threshold": float(base_threshold), "overrides": cleaned}


def _policy_name(policy: Mapping[str, Any]) -> str:
    overrides = policy.get("overrides", {})
    if not overrides:
        return f"global={float(policy['base_threshold']):.2f}"
    parts = [f"{label}={float(value):.2f}" for label, value in sorted(overrides.items())]
    return f"base={float(policy['base_threshold']):.2f}; " + ", ".join(parts)


def _label_name_from_pred_id(pred_id: int, label_to_idx: Mapping[str, int]) -> Optional[str]:
    label_name = evalmod.cfg.ID2LANG.get(int(pred_id))
    if label_name is None:
        return None
    alias = evalmod.PREDICTION_LABEL_ALIASES.get(label_name)
    if alias and alias in label_to_idx:
        label_name = alias
    return label_name


def _resolve_model_args(args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_path = Path(args.checkpoint).resolve()
    auto_hparams = evalmod._load_checkpoint_hparams(ckpt_path)

    label_names = auto_hparams.get("label_names")
    if label_names:
        evalmod._apply_label_mapping(label_names)

    arch = str(args.arch or auto_hparams.get("arch") or "unet1d").lower().strip()
    model_dim = int(args.model_dim if args.model_dim is not None else auto_hparams.get("model_dim", 256))
    dtype = str(args.dtype or auto_hparams.get("dtype", "bfloat16")).rsplit(".", 1)[-1]

    if arch == "unet1d":
        channels = _parse_channels(args.channels)
        if channels is None:
            channels = [int(ch) for ch in auto_hparams.get("channels", evalmod.DEFAULT_CHANNELS)]
    else:
        channels = [int(ch) for ch in evalmod.DEFAULT_CHANNELS]

    resolved = {
        "arch": arch,
        "model_dim": model_dim,
        "dtype": dtype,
        "channels": [int(ch) for ch in channels],
        "mamba_layers": int(args.mamba_layers if args.mamba_layers is not None else auto_hparams.get("mamba_layers", 6)),
        "mamba_d_state": int(args.mamba_d_state if args.mamba_d_state is not None else auto_hparams.get("mamba_d_state", 8)),
        "mamba_expand": int(args.mamba_expand if args.mamba_expand is not None else auto_hparams.get("mamba_expand", 1)),
        "mamba_dt_rank": int(args.mamba_dt_rank if args.mamba_dt_rank is not None else auto_hparams.get("mamba_dt_rank", 16)),
        "mamba_conv": int(args.mamba_conv if args.mamba_conv is not None else auto_hparams.get("mamba_conv", 4)),
        "mamba_bidirectional": bool(
            args.mamba_bidirectional
            if args.mamba_bidirectional is not None
            else auto_hparams.get("mamba_bidirectional", True)
        ),
        "checkpoint_hparams": auto_hparams,
    }
    return resolved


def _load_dataset(args: argparse.Namespace) -> hfds.Dataset:
    data_root = Path(args.data_root).resolve()
    dataset_path = data_root / "mal_injection"
    if not dataset_path.exists():
        raise FileNotFoundError(f"mal_injection dataset not found: {dataset_path}")
    dataset = hfds.load_from_disk(str(dataset_path))
    subset, _, _, _ = evalmod._prepare_dataset(
        "mal_injection",
        dataset,
        max_samples=int(args.max_samples),
        base_seed=args.sample_seed,
        preserve_all=False,
    )
    return subset


def _build_eval_label_map(dataset: hfds.Dataset) -> Tuple[List[str], Dict[str, int]]:
    label_candidates = {"other"}
    for example in dataset:
        for seg in evalmod._normalize_segments(example.get("segments")):
            label_candidates.add(seg["label"])
    for alias_target in evalmod.PREDICTION_LABEL_ALIASES.values():
        label_candidates.add(alias_target)
    return evalmod._confusion_size(label_candidates)


def _cache_examples(
    dataset: hfds.Dataset,
    runner: evalmod.SegmenterRunner,
    *,
    label_to_idx: Mapping[str, int],
    min_run_chars: int,
    log_interval: int,
) -> List[CachedExample]:
    other_idx = int(label_to_idx["other"])
    cached: List[CachedExample] = []
    total = len(dataset)
    start_time = time.perf_counter()

    for idx, example in enumerate(dataset):
        row = example if isinstance(example, dict) else dict(example)
        raw_content = row.get("content")
        content = evalmod.normalize_eval_text(raw_content if isinstance(raw_content, str) else "")
        segments = evalmod._normalize_segments(row.get("segments"))
        truth = evalmod._segments_to_labels(content, segments, dict(label_to_idx))
        _, pred_labels, pred_probs = runner.segment_text(content, min_run_chars=min_run_chars)

        pred_idx = np.full((len(pred_labels),), other_idx, dtype=np.int16)
        max_prob = np.zeros((len(pred_labels),), dtype=np.float32)
        for pos, pred_id in enumerate(pred_labels):
            label_name = _label_name_from_pred_id(int(pred_id), label_to_idx)
            if label_name is not None and label_name in label_to_idx:
                pred_idx[pos] = np.int16(label_to_idx[label_name])
            if pred_probs is not None and pos < len(pred_probs):
                prob_vec = np.asarray(pred_probs[pos], dtype=np.float32)
                if prob_vec.size:
                    max_prob[pos] = float(prob_vec.max())

        valid_mask = truth >= 0
        if content:
            ws_mask = np.array([ch in evalmod._VISUAL_WHITESPACE_SET for ch in content], dtype=bool)
            valid_mask = np.logical_and(valid_mask, ~ws_mask)

        truth_valid = truth[valid_mask]
        pred_valid = pred_idx[valid_mask]
        max_prob_valid = max_prob[valid_mask]
        metadata = evalmod._parse_metadata(row)

        payload_lang = str(metadata.get("payload_lang") or "")
        host_lang = str(metadata.get("host_lang") or "")
        payload_idx = int(label_to_idx.get(payload_lang, -1))
        host_idx = int(label_to_idx.get(host_lang, -1))
        if payload_idx < 0:
            continue

        truth_mask = truth_valid == payload_idx
        truth_chars = int(truth_mask.sum())
        if truth_chars <= 0:
            continue

        cached.append(
            CachedExample(
                payload_lang=payload_lang,
                host_lang=host_lang,
                payload_idx=payload_idx,
                host_idx=host_idx,
                truth_chars=truth_chars,
                truth_mask=truth_mask.astype(bool, copy=False),
                raw_pred_idx=pred_valid.astype(np.int16, copy=False),
                max_prob=max_prob_valid.astype(np.float32, copy=False),
            )
        )

        if log_interval > 0 and ((idx + 1) % log_interval == 0 or idx + 1 == total):
            elapsed = time.perf_counter() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"[cache] {idx + 1}/{total} examples cached in {elapsed:.1f}s ({rate:.2f} ex/s)",
                flush=True,
            )

    return cached


def _empty_stat() -> Dict[str, Any]:
    return {
        "count": 0,
        "detected": 0,
        "below_threshold": 0,
        "truth_chars": 0,
        "correct_chars": 0,
        "iou_sum": 0.0,
        "coverage_hits": 0,
        "coverage_sum": 0.0,
        "coverage_count": 0,
    }


def _finalize_stat(entry: Mapping[str, Any]) -> Dict[str, Any]:
    count = int(entry.get("count", 0))
    truth_chars = int(entry.get("truth_chars", 0))
    return {
        "support": count,
        "det_rate@0.5": _safe_ratio(int(entry.get("detected", 0)), count),
        "mean_iou": _safe_ratio(float(entry.get("iou_sum", 0.0)), count),
        "coverage": _safe_ratio(int(entry.get("correct_chars", 0)), truth_chars),
        "coverage_hits": int(entry.get("coverage_hits", 0)),
        "coverage_count": int(entry.get("coverage_count", 0)),
        "iou_hits": int(entry.get("detected", 0)),
        "iou_count": count,
    }


def _evaluate_policy(
    examples: Sequence[CachedExample],
    label_names: Sequence[str],
    label_to_idx: Mapping[str, int],
    *,
    base_threshold: float,
    overrides: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    other_idx = int(label_to_idx["other"])
    thresholds = np.full((len(label_names),), float(base_threshold), dtype=np.float32)
    for label, value in (overrides or {}).items():
        idx = label_to_idx.get(label)
        if idx is not None:
            thresholds[int(idx)] = float(value)

    any_stats = _empty_stat()
    payload_stats = _empty_stat()
    any_by_lang: Dict[str, Dict[str, Any]] = {}
    payload_by_lang: Dict[str, Dict[str, Any]] = {}

    for example in examples:
        raw_pred = example.raw_pred_idx.astype(np.int32, copy=False)
        tau = thresholds[raw_pred]
        final_pred = np.where(example.max_prob < tau, other_idx, raw_pred)

        payload_lang = example.payload_lang
        truth_mask = example.truth_mask
        truth_chars = int(example.truth_chars)
        payload_idx = int(example.payload_idx)
        host_idx = int(example.host_idx)

        payload_entry = payload_by_lang.setdefault(payload_lang, _empty_stat())
        any_entry = any_by_lang.setdefault(payload_lang, _empty_stat())

        pred_payload_mask = final_pred == payload_idx
        inter_payload = int(np.count_nonzero(np.logical_and(truth_mask, pred_payload_mask)))
        pred_payload_total = int(np.count_nonzero(pred_payload_mask))
        union_payload = truth_chars + pred_payload_total - inter_payload
        coverage_payload = inter_payload / truth_chars if truth_chars > 0 else 0.0

        if union_payload > 0:
            iou_payload = inter_payload / union_payload
            payload_stats["count"] += 1
            payload_stats["truth_chars"] += truth_chars
            payload_stats["correct_chars"] += inter_payload
            payload_stats["iou_sum"] += iou_payload
            payload_stats["coverage_sum"] += coverage_payload
            payload_stats["coverage_count"] += 1
            payload_entry["count"] += 1
            payload_entry["truth_chars"] += truth_chars
            payload_entry["correct_chars"] += inter_payload
            payload_entry["iou_sum"] += iou_payload
            payload_entry["coverage_sum"] += coverage_payload
            payload_entry["coverage_count"] += 1
            if coverage_payload >= evalmod.PAYLOAD_IOU_THRESHOLD:
                payload_stats["coverage_hits"] += 1
                payload_entry["coverage_hits"] += 1
            if iou_payload >= evalmod.PAYLOAD_IOU_THRESHOLD:
                payload_stats["detected"] += 1
                payload_entry["detected"] += 1
            else:
                payload_stats["below_threshold"] += 1
                payload_entry["below_threshold"] += 1

        if host_idx >= 0:
            pred_any_mask = final_pred != host_idx
            inter_any = int(np.count_nonzero(np.logical_and(truth_mask, pred_any_mask)))
            pred_any_total = int(np.count_nonzero(pred_any_mask))
        else:
            inter_any = truth_chars
            pred_any_total = int(final_pred.size)
        union_any = truth_chars + pred_any_total - inter_any
        coverage_any = inter_any / truth_chars if truth_chars > 0 else 0.0

        if union_any > 0:
            iou_any = inter_any / union_any
            any_stats["count"] += 1
            any_stats["truth_chars"] += truth_chars
            any_stats["correct_chars"] += inter_any
            any_stats["iou_sum"] += iou_any
            any_stats["coverage_sum"] += coverage_any
            any_stats["coverage_count"] += 1
            any_entry["count"] += 1
            any_entry["truth_chars"] += truth_chars
            any_entry["correct_chars"] += inter_any
            any_entry["iou_sum"] += iou_any
            any_entry["coverage_sum"] += coverage_any
            any_entry["coverage_count"] += 1
            if coverage_any >= evalmod.PAYLOAD_IOU_THRESHOLD:
                any_stats["coverage_hits"] += 1
                any_entry["coverage_hits"] += 1
            if iou_any >= evalmod.PAYLOAD_IOU_THRESHOLD:
                any_stats["detected"] += 1
                any_entry["detected"] += 1
            else:
                any_stats["below_threshold"] += 1
                any_entry["below_threshold"] += 1

    summary = {
        "policy": _normalize_policy(base_threshold, overrides),
        "payload": _finalize_stat(payload_stats),
        "any": _finalize_stat(any_stats),
        "per_language_correct": {
            label: _finalize_stat(entry)
            for label, entry in sorted(payload_by_lang.items())
        },
        "per_language_any": {
            label: _finalize_stat(entry)
            for label, entry in sorted(any_by_lang.items())
        },
    }
    summary["score_tuple"] = list(_score_tuple(summary))
    return summary


def _best_of(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return max(results, key=_score_tuple)


def _run_global_sweep(
    examples: Sequence[CachedExample],
    label_names: Sequence[str],
    label_to_idx: Mapping[str, int],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    results = []
    for threshold in thresholds:
        results.append(
            _evaluate_policy(
                examples,
                label_names,
                label_to_idx,
                base_threshold=float(threshold),
                overrides=None,
            )
        )
    return results


def _run_single_label_sweep(
    examples: Sequence[CachedExample],
    label_names: Sequence[str],
    label_to_idx: Mapping[str, int],
    *,
    base_threshold: float,
    focus_labels: Sequence[str],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    results = []
    for label in focus_labels:
        for threshold in thresholds:
            results.append(
                _evaluate_policy(
                    examples,
                    label_names,
                    label_to_idx,
                    base_threshold=float(base_threshold),
                    overrides={label: float(threshold)},
                )
            )
    return results


def _greedy_search(
    examples: Sequence[CachedExample],
    label_names: Sequence[str],
    label_to_idx: Mapping[str, int],
    *,
    base_threshold: float,
    focus_labels: Sequence[str],
    thresholds: Sequence[float],
    max_steps: int,
) -> Dict[str, Any]:
    current = _evaluate_policy(
        examples,
        label_names,
        label_to_idx,
        base_threshold=float(base_threshold),
        overrides=None,
    )
    history = [current]
    used: Dict[str, float] = {}

    for _ in range(max(0, int(max_steps))):
        best_candidate = None
        best_score = _score_tuple(current)
        for label in focus_labels:
            for threshold in thresholds:
                if used.get(label) == float(threshold):
                    continue
                overrides = dict(used)
                overrides[label] = float(threshold)
                candidate = _evaluate_policy(
                    examples,
                    label_names,
                    label_to_idx,
                    base_threshold=float(base_threshold),
                    overrides=overrides,
                )
                candidate_score = _score_tuple(candidate)
                if candidate_score > best_score:
                    best_candidate = candidate
                    best_score = candidate_score
        if best_candidate is None:
            break
        current = best_candidate
        used = dict(current["policy"]["overrides"])
        history.append(current)

    return {"best": current, "history": history}


def _language_comparison_rows(
    baseline: Mapping[str, Any],
    best_global: Mapping[str, Any],
    best_greedy: Mapping[str, Any],
) -> List[str]:
    labels = sorted(
        set(baseline.get("per_language_correct", {}).keys())
        | set(best_global.get("per_language_correct", {}).keys())
        | set(best_greedy.get("per_language_correct", {}).keys())
    )
    lines = [
        "| Payload | Baseline IoU >=50% | Best global IoU >=50% | Greedy IoU >=50% | Baseline coverage | Best global coverage | Greedy coverage |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for label in labels:
        base = baseline["per_language_correct"].get(label, {})
        glob = best_global["per_language_correct"].get(label, {})
        greedy = best_greedy["per_language_correct"].get(label, {})
        lines.append(
            f"| {label} | {_format_hits(base.get('iou_hits', 0), base.get('iou_count', 0))} | "
            f"{_format_hits(glob.get('iou_hits', 0), glob.get('iou_count', 0))} | "
            f"{_format_hits(greedy.get('iou_hits', 0), greedy.get('iou_count', 0))} | "
            f"{_format_pct(base.get('coverage'))} | {_format_pct(glob.get('coverage'))} | {_format_pct(greedy.get('coverage'))} |"
        )
    return lines


def _summary_table_row(title: str, result: Mapping[str, Any]) -> str:
    payload = result["payload"]
    any_payload = result["any"]
    return (
        f"| {title} | {_policy_name(result['policy'])} | "
        f"{_format_hits(payload['coverage_hits'], payload['coverage_count'])} | "
        f"{_format_hits(payload['iou_hits'], payload['iou_count'])} | "
        f"{_format_float(payload['mean_iou'])} | {_format_pct(payload['coverage'])} | "
        f"{_format_hits(any_payload['coverage_hits'], any_payload['coverage_count'])} | "
        f"{_format_hits(any_payload['iou_hits'], any_payload['iou_count'])} | "
        f"{_format_float(any_payload['mean_iou'])} | {_format_pct(any_payload['coverage'])} |"
    )


def _render_report(
    args: argparse.Namespace,
    resolved_model: Mapping[str, Any],
    baseline: Mapping[str, Any],
    global_results: Sequence[Mapping[str, Any]],
    single_label_results: Sequence[Mapping[str, Any]],
    greedy_result: Mapping[str, Any],
) -> str:
    best_global = _best_of(global_results)
    best_single_by_label: Dict[str, Dict[str, Any]] = {}
    for result in single_label_results:
        overrides = result["policy"].get("overrides", {})
        if len(overrides) != 1:
            continue
        label, _ = next(iter(overrides.items()))
        prev = best_single_by_label.get(label)
        if prev is None or _score_tuple(result) > _score_tuple(prev):
            best_single_by_label[label] = dict(result)

    lines = [
        "# Mal Injection Threshold Experiment",
        "",
        "Separate threshold sweep for `mal_injection` only. This does not change `evaluation.py`.",
        "",
        "## Config",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Data root: `{args.data_root}`",
        f"- Samples: `{baseline['payload']['support']}`",
        f"- Base threshold: `{float(args.base_threshold):.2f}`",
        f"- Global grid: `{', '.join(f'{v:.2f}' for v in _parse_threshold_grid(args.global_thresholds, DEFAULT_GLOBAL_THRESHOLDS))}`",
        f"- Label grid: `{', '.join(f'{v:.2f}' for v in _parse_threshold_grid(args.label_thresholds, DEFAULT_LABEL_THRESHOLDS))}`",
        f"- Focus labels: `{', '.join(args.focus_labels) if args.focus_labels else 'auto'}`",
        (
            "- Model: "
            f"`arch={resolved_model['arch']}`, "
            f"`model_dim={resolved_model['model_dim']}`, "
            f"`dtype={resolved_model['dtype']}`, "
            f"`chunk={int(args.chunk)}`, "
            f"`batch_size={int(args.batch_size)}`"
        ),
        "",
        "## Summary",
        "",
        "| Candidate | Policy | Correct cov >=50% | Correct IoU >=50% | Correct avg IoU | Correct avg coverage | Any cov >=50% | Any IoU >=50% | Any avg IoU | Any avg coverage |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |",
        _summary_table_row("Baseline", baseline),
        _summary_table_row("Best global", best_global),
        _summary_table_row("Best greedy", greedy_result["best"]),
        "",
        "## Best Global Thresholds",
        "",
        "| Policy | Correct cov >=50% | Correct IoU >=50% | Correct avg IoU | Correct avg coverage | Any IoU >=50% | Any avg IoU |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]

    for result in sorted(global_results, key=_score_tuple, reverse=True):
        payload = result["payload"]
        any_payload = result["any"]
        lines.append(
            f"| {_policy_name(result['policy'])} | "
            f"{_format_hits(payload['coverage_hits'], payload['coverage_count'])} | "
            f"{_format_hits(payload['iou_hits'], payload['iou_count'])} | "
            f"{_format_float(payload['mean_iou'])} | {_format_pct(payload['coverage'])} | "
            f"{_format_hits(any_payload['iou_hits'], any_payload['iou_count'])} | "
            f"{_format_float(any_payload['mean_iou'])} |"
        )

    lines.extend(
        [
            "",
            "## Best Single-Label Overrides",
            "",
            "| Label | Policy | Correct IoU >=50% | Correct avg IoU | Correct avg coverage |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for label in sorted(best_single_by_label):
        result = best_single_by_label[label]
        payload = result["payload"]
        lines.append(
            f"| {label} | {_policy_name(result['policy'])} | "
            f"{_format_hits(payload['iou_hits'], payload['iou_count'])} | "
            f"{_format_float(payload['mean_iou'])} | {_format_pct(payload['coverage'])} |"
        )

    lines.extend(
        [
            "",
            "## Greedy Search",
            "",
            "| Step | Policy | Correct IoU >=50% | Correct avg IoU | Correct avg coverage |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for idx, result in enumerate(greedy_result["history"]):
        payload = result["payload"]
        lines.append(
            f"| {idx} | {_policy_name(result['policy'])} | "
            f"{_format_hits(payload['iou_hits'], payload['iou_count'])} | "
            f"{_format_float(payload['mean_iou'])} | {_format_pct(payload['coverage'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Payload Comparison",
            "",
        ]
    )
    lines.extend(_language_comparison_rows(baseline, best_global, greedy_result["best"]))
    lines.extend(
        [
            "",
            "Report generated at " + time.strftime("%Y-%m-%d %H:%M:%S"),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold experiments for mal_injection.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "evaluation" / "data"),
        help="Evaluation data root containing mal_injection.",
    )
    parser.add_argument("--device", default="auto", help="cpu/gpu/cuda/auto")
    parser.add_argument("--arch", choices=("unet1d", "mamba"), default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--channels", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--mamba-layers", type=int, default=None)
    parser.add_argument("--mamba-d-state", type=int, default=None)
    parser.add_argument("--mamba-expand", type=int, default=None)
    parser.add_argument("--mamba-dt-rank", type=int, default=None)
    parser.add_argument("--mamba-conv", type=int, default=None)
    parser.add_argument("--mamba-bidirectional", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--chunk", type=int, default=evalmod.DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inference-backend", choices=("auto", "fast", "legacy"), default="auto")
    parser.add_argument("--min-run", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 keeps all mal_injection samples.")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=0.85,
        help="Baseline open-set threshold to compare against.",
    )
    parser.add_argument(
        "--global-thresholds",
        type=str,
        default=None,
        help="Comma-separated global thresholds to test.",
    )
    parser.add_argument(
        "--label-thresholds",
        type=str,
        default=None,
        help="Comma-separated per-label override thresholds to test.",
    )
    parser.add_argument(
        "--focus-labels",
        nargs="*",
        default=None,
        help="Payload labels to treat as threshold-override candidates. Defaults to all payload labels in mal_injection.",
    )
    parser.add_argument("--greedy-steps", type=int, default=6)
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "evaluation" / "mal_injection_threshold_experiment.md"),
    )
    parser.add_argument(
        "--json-path",
        default=str(REPO_ROOT / "evaluation" / "mal_injection_threshold_experiment.json"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    start_time = time.perf_counter()

    resolved_model = _resolve_model_args(args)
    dataset = _load_dataset(args)
    label_names, label_to_idx = _build_eval_label_map(dataset)
    focus_labels = sorted({label for label in (args.focus_labels or []) if label in label_to_idx})
    if not focus_labels:
        payload_labels = set()
        for row in dataset:
            metadata = evalmod._parse_metadata(row)
            payload_lang = metadata.get("payload_lang")
            if payload_lang:
                payload_labels.add(str(payload_lang))
        focus_labels = sorted(label for label in payload_labels if label in label_to_idx)
    args.focus_labels = focus_labels

    print(
        f"Preparing mal_injection threshold experiment on {len(dataset)} samples "
        f"with {len(focus_labels)} focus labels.",
        flush=True,
    )

    runner = evalmod.SegmenterRunner(
        args.checkpoint,
        arch=resolved_model["arch"],
        model_dim=int(resolved_model["model_dim"]),
        channels=resolved_model["channels"],
        mamba_layers=int(resolved_model["mamba_layers"]),
        mamba_d_state=int(resolved_model["mamba_d_state"]),
        mamba_expand=int(resolved_model["mamba_expand"]),
        mamba_dt_rank=int(resolved_model["mamba_dt_rank"]),
        mamba_conv=int(resolved_model["mamba_conv"]),
        mamba_bidirectional=bool(resolved_model["mamba_bidirectional"]),
        dtype=str(resolved_model["dtype"]),
        chunk=int(args.chunk),
        device=args.device,
        batch_size=int(args.batch_size),
        inference_backend=str(args.inference_backend),
    )

    cached_examples = _cache_examples(
        dataset,
        runner,
        label_to_idx=label_to_idx,
        min_run_chars=int(args.min_run),
        log_interval=int(args.log_interval),
    )
    print(f"Cached {len(cached_examples)} usable examples for threshold sweeps.", flush=True)

    global_thresholds = _parse_threshold_grid(args.global_thresholds, DEFAULT_GLOBAL_THRESHOLDS)
    if float(args.base_threshold) not in global_thresholds:
        global_thresholds = sorted(global_thresholds + [float(args.base_threshold)])
    label_thresholds = _parse_threshold_grid(args.label_thresholds, DEFAULT_LABEL_THRESHOLDS)
    if float(args.base_threshold) not in label_thresholds:
        label_thresholds = sorted(label_thresholds + [float(args.base_threshold)])

    baseline = _evaluate_policy(
        cached_examples,
        label_names,
        label_to_idx,
        base_threshold=float(args.base_threshold),
        overrides=None,
    )
    global_results = _run_global_sweep(
        cached_examples,
        label_names,
        label_to_idx,
        global_thresholds,
    )
    single_label_results = _run_single_label_sweep(
        cached_examples,
        label_names,
        label_to_idx,
        base_threshold=float(args.base_threshold),
        focus_labels=focus_labels,
        thresholds=label_thresholds,
    )
    greedy_result = _greedy_search(
        cached_examples,
        label_names,
        label_to_idx,
        base_threshold=float(args.base_threshold),
        focus_labels=focus_labels,
        thresholds=label_thresholds,
        max_steps=int(args.greedy_steps),
    )

    best_global = _best_of(global_results)
    print(
        "Best global policy: "
        f"{_policy_name(best_global['policy'])} "
        f"payload_det@0.5={_format_pct(best_global['payload']['det_rate@0.5'])} "
        f"payload_cov={_format_pct(best_global['payload']['coverage'])}",
        flush=True,
    )
    print(
        "Best greedy policy: "
        f"{_policy_name(greedy_result['best']['policy'])} "
        f"payload_det@0.5={_format_pct(greedy_result['best']['payload']['det_rate@0.5'])} "
        f"payload_cov={_format_pct(greedy_result['best']['payload']['coverage'])}",
        flush=True,
    )

    report = _render_report(
        args,
        resolved_model,
        baseline,
        global_results,
        single_label_results,
        greedy_result,
    )
    report_path = Path(args.report_path).resolve()
    json_path = Path(args.json_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    payload = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "data_root": str(args.data_root),
            "base_threshold": float(args.base_threshold),
            "global_thresholds": global_thresholds,
            "label_thresholds": label_thresholds,
            "focus_labels": focus_labels,
            "greedy_steps": int(args.greedy_steps),
            "chunk": int(args.chunk),
            "batch_size": int(args.batch_size),
            "arch": resolved_model["arch"],
            "model_dim": int(resolved_model["model_dim"]),
            "dtype": str(resolved_model["dtype"]),
            "channels": [int(ch) for ch in resolved_model["channels"]],
            "mamba_layers": int(resolved_model["mamba_layers"]),
            "mamba_d_state": int(resolved_model["mamba_d_state"]),
            "mamba_expand": int(resolved_model["mamba_expand"]),
            "mamba_dt_rank": int(resolved_model["mamba_dt_rank"]),
            "mamba_conv": int(resolved_model["mamba_conv"]),
            "mamba_bidirectional": bool(resolved_model["mamba_bidirectional"]),
        },
        "baseline": baseline,
        "global_results": global_results,
        "single_label_results": single_label_results,
        "greedy_result": greedy_result,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote report to {report_path}", flush=True)
    print(f"Wrote JSON to {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
