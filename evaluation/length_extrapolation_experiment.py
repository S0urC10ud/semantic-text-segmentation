#!/usr/bin/env python3
"""Run a repeated-length extrapolation experiment on a mixed monitor document.

This experiment:
1. Finds a near-target-size mixed-content monitor file with ground-truth labels.
2. Scores candidates with a chosen checkpoint and selects the strongest seed file
   (unless a specific file index is pinned).
3. Repeats that file 1x/2x/4x/8x/16x/32x and measures how labeling quality
   changes with length.

The script reuses the evaluation runtime so the scoring path matches the normal
benchmark harness closely.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg  # noqa: E402
from utils.monitor_eval import load_monitor_memmaps  # noqa: E402

_EVAL_MODULE_PATH = REPO_ROOT / "evaluation" / "evaluation.py"
_EVAL_SPEC = importlib.util.spec_from_file_location("_length_experiment_eval_module", _EVAL_MODULE_PATH)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:
    raise RuntimeError(f"Failed to load evaluation module from {_EVAL_MODULE_PATH}")
_EVAL_MODULE = importlib.util.module_from_spec(_EVAL_SPEC)
sys.modules[_EVAL_SPEC.name] = _EVAL_MODULE
_EVAL_SPEC.loader.exec_module(_EVAL_MODULE)

SegmenterRunner = _EVAL_MODULE.SegmenterRunner
_VISUAL_WHITESPACE_SET = _EVAL_MODULE._VISUAL_WHITESPACE_SET
_load_checkpoint_hparams = _EVAL_MODULE._load_checkpoint_hparams
_relabel_whitespace_labels_from_neighbors = _EVAL_MODULE._relabel_whitespace_labels_from_neighbors
_smooth_min_run = _EVAL_MODULE._smooth_min_run
normalize_eval_text = _EVAL_MODULE.normalize_eval_text


DEFAULT_REPEAT_FACTORS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
DEFAULT_SFULLFILES3_CHECKPOINT = REPO_ROOT / "checkpoints" / "sweeps" / "sfullfiles3.msgpack"


@dataclass(frozen=True)
class MonitorCandidate:
    file_idx: int
    content: str
    truth_labels: np.ndarray
    truth_segments: Tuple[Tuple[int, int, str], ...]
    byte_len: int
    label_bytes: Dict[str, int]
    seg_count: int
    distinct_labels: Tuple[str, ...]

    @property
    def minority_share(self) -> float:
        total = sum(int(v) for v in self.label_bytes.values())
        if total <= 0 or not self.label_bytes:
            return 0.0
        return float(min(self.label_bytes.values()) / total)

    def summary(self) -> Dict[str, Any]:
        total = sum(int(v) for v in self.label_bytes.values())
        shares = {
            label: {
                "chars": int(chars),
                "share": (float(chars) / float(total)) if total > 0 else 0.0,
            }
            for label, chars in sorted(self.label_bytes.items())
        }
        return {
            "file_idx": int(self.file_idx),
            "byte_len": int(self.byte_len),
            "char_len": int(len(self.content)),
            "seg_count": int(self.seg_count),
            "distinct_labels": list(self.distinct_labels),
            "minority_share": float(self.minority_share),
            "label_bytes": {k: int(v) for k, v in sorted(self.label_bytes.items())},
            "label_shares": shares,
        }


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: MonitorCandidate
    accuracy: float
    macro_accuracy: float
    evaluated_chars: int
    elapsed_seconds: float
    per_label_accuracy: Dict[str, float]

    def summary(self) -> Dict[str, Any]:
        out = self.candidate.summary()
        out.update(
            {
                "accuracy": float(self.accuracy),
                "macro_accuracy": float(self.macro_accuracy),
                "evaluated_chars": int(self.evaluated_chars),
                "elapsed_seconds": float(self.elapsed_seconds),
                "per_label_accuracy": {
                    k: float(v) for k, v in sorted(self.per_label_accuracy.items())
                },
            }
        )
        return out


@dataclass(frozen=True)
class LengthSweepResult:
    repeat_factor: int
    total_chars: int
    total_bytes_utf8: int
    accuracy: float
    macro_accuracy: float
    evaluated_chars: int
    elapsed_seconds: float
    per_label_accuracy: Dict[str, float]

    def summary(self) -> Dict[str, Any]:
        return {
            "repeat_factor": int(self.repeat_factor),
            "total_chars": int(self.total_chars),
            "total_bytes_utf8": int(self.total_bytes_utf8),
            "accuracy": float(self.accuracy),
            "macro_accuracy": float(self.macro_accuracy),
            "evaluated_chars": int(self.evaluated_chars),
            "elapsed_seconds": float(self.elapsed_seconds),
            "per_label_accuracy": {
                k: float(v) for k, v in sorted(self.per_label_accuracy.items())
            },
        }


def _parse_repeat_factors(raw_values: Sequence[str]) -> Tuple[int, ...]:
    factors: List[int] = []
    for raw in raw_values:
        for piece in str(raw).split(","):
            piece = piece.strip()
            if not piece:
                continue
            value = int(piece)
            if value <= 0:
                raise ValueError("Repeat factors must be positive integers.")
            factors.append(value)
    uniq = sorted(set(factors))
    if not uniq:
        raise ValueError("At least one repeat factor is required.")
    return tuple(int(v) for v in uniq)


def _label_mappings() -> Tuple[Dict[int, str], Optional[int]]:
    cfg.update_lang_mappings()
    id2lang = dict(cfg.ID2LANG)
    other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
    if other_idx is not None:
        id2lang[int(other_idx)] = "other"
    return id2lang, (int(other_idx) if other_idx is not None else None)


def _candidate_sort_key(candidate: MonitorCandidate, target_bytes: int) -> Tuple[Any, ...]:
    return (
        abs(int(candidate.byte_len) - int(target_bytes)),
        len(candidate.distinct_labels),
        int(candidate.seg_count),
        -float(candidate.minority_share),
        int(candidate.file_idx),
    )


def _load_mixed_candidates(
    monitor_root: Path,
    *,
    target_bytes: int,
    size_tolerance: int,
    min_minority_share: float,
    min_distinct_labels: int,
    max_segments: Optional[int],
    required_labels: Sequence[str],
    excluded_labels: Sequence[str],
) -> List[MonitorCandidate]:
    data = load_monitor_memmaps(monitor_root)
    files = data["files"]
    segments = data["segments"]
    contents = data["contents"]
    id2lang, _ = _label_mappings()

    required = {str(label).strip() for label in required_labels if str(label).strip()}
    excluded = {str(label).strip() for label in excluded_labels if str(label).strip()}

    out: List[MonitorCandidate] = []
    lower = max(1, int(target_bytes) - int(size_tolerance))
    upper = int(target_bytes) + int(size_tolerance)

    for file_idx, row in enumerate(files):
        byte_len = int(row["byte_len"])
        if byte_len < lower or byte_len > upper:
            continue

        byte_start = int(row["byte_start"])
        seg_start = int(row["seg_start"])
        seg_count = int(row["seg_count"])
        if seg_count <= 0:
            continue

        file_slice = contents[byte_start : byte_start + byte_len]
        raw_chars = "".join(chr(int(b)) for b in np.asarray(file_slice, dtype=np.uint8))
        content = normalize_eval_text(raw_chars)
        if not content:
            continue

        truth = np.full((len(content),), -1, dtype=np.int32)
        label_bytes: Dict[str, int] = {}
        truth_segments: List[Tuple[int, int, str]] = []

        for seg in segments[seg_start : seg_start + seg_count]:
            start = int(seg["start"])
            end = int(seg["end"])
            if end <= start:
                continue
            label_id = int(seg["label"])
            label_name = id2lang.get(label_id)
            if label_name is None:
                continue
            truth[start:end] = label_id
            truth_segments.append((start, end, label_name))
            label_bytes[label_name] = label_bytes.get(label_name, 0) + (end - start)

        if not truth_segments:
            continue

        distinct_labels = tuple(sorted(label_bytes))
        if len(distinct_labels) < int(min_distinct_labels):
            continue
        if required and not required.issubset(set(distinct_labels)):
            continue
        if excluded and any(label in excluded for label in distinct_labels):
            continue
        if max_segments is not None and len(truth_segments) > int(max_segments):
            continue

        total = sum(int(v) for v in label_bytes.values())
        if total <= 0:
            continue
        minority_share = float(min(label_bytes.values()) / float(total))
        if minority_share < float(min_minority_share):
            continue

        out.append(
            MonitorCandidate(
                file_idx=int(file_idx),
                content=content,
                truth_labels=truth,
                truth_segments=tuple(truth_segments),
                byte_len=int(byte_len),
                label_bytes={k: int(v) for k, v in label_bytes.items()},
                seg_count=int(len(truth_segments)),
                distinct_labels=distinct_labels,
            )
        )

    out.sort(key=lambda cand: _candidate_sort_key(cand, target_bytes))
    return out


def _repeat_candidate(candidate: MonitorCandidate, repeat_factor: int) -> MonitorCandidate:
    factor = int(repeat_factor)
    if factor <= 0:
        raise ValueError("repeat_factor must be positive.")
    if factor == 1:
        return candidate

    base_len = len(candidate.content)
    repeated_segments: List[Tuple[int, int, str]] = []
    for idx in range(factor):
        offset = idx * base_len
        for start, end, label in candidate.truth_segments:
            repeated_segments.append((int(start + offset), int(end + offset), label))

    repeated_label_bytes = {
        label: int(chars * factor) for label, chars in candidate.label_bytes.items()
    }
    return MonitorCandidate(
        file_idx=int(candidate.file_idx),
        content=(candidate.content * factor),
        truth_labels=np.tile(candidate.truth_labels, factor),
        truth_segments=tuple(repeated_segments),
        byte_len=int(candidate.byte_len * factor),
        label_bytes=repeated_label_bytes,
        seg_count=int(candidate.seg_count * factor),
        distinct_labels=tuple(candidate.distinct_labels),
    )


def _predict_char_labels(
    runner: SegmenterRunner,
    text: str,
    *,
    min_run_chars: int,
    other_threshold: float,
    other_label_idx: Optional[int],
) -> np.ndarray:
    normalized = normalize_eval_text(text)
    if not normalized:
        return np.zeros((0,), dtype=np.int32)

    byte_arr = np.frombuffer(normalized.encode("utf-8", "ignore"), dtype=np.uint8)
    if other_threshold > 0.0:
        byte_labels, byte_probs = runner._segment_bytes(byte_arr)
        char_labels: List[int] = []
        bpos = 0
        for ch in normalized:
            encoded = ch.encode("utf-8", "ignore")
            length = len(encoded)
            if length <= 0:
                char_labels.append(int(other_label_idx or 0))
                continue
            seg = byte_labels[bpos : bpos + length]
            if len(seg) == 0:
                label_id = int(other_label_idx or 0)
            else:
                values, counts = np.unique(seg, return_counts=True)
                label_id = int(values[np.argmax(counts)])
            if (
                other_label_idx is not None
                and byte_probs is not None
                and len(byte_probs) >= bpos + length
            ):
                avg_probs = np.mean(byte_probs[bpos : bpos + length], axis=0)
                if float(np.max(avg_probs)) < float(other_threshold):
                    label_id = int(other_label_idx)
            char_labels.append(int(label_id))
            bpos += length
        char_labels = _relabel_whitespace_labels_from_neighbors(normalized, char_labels)
        char_labels = _smooth_min_run(char_labels, int(min_run_chars))
        return np.asarray(char_labels, dtype=np.int32)

    _, labels = runner.segment_text_labels_only(normalized, min_run_chars=int(min_run_chars))
    return np.asarray(labels, dtype=np.int32)


def _score_prediction(
    candidate: MonitorCandidate,
    pred_labels: np.ndarray,
    *,
    id2lang: Dict[int, str],
) -> Tuple[float, float, int, Dict[str, float]]:
    truth = np.asarray(candidate.truth_labels, dtype=np.int32)
    pred = np.asarray(pred_labels, dtype=np.int32)
    if len(pred) != len(truth):
        raise ValueError(
            f"Prediction length mismatch for file_idx={candidate.file_idx}: "
            f"pred={len(pred)} truth={len(truth)}"
        )

    valid = truth >= 0
    if candidate.content:
        ws_mask = np.array([ch in _VISUAL_WHITESPACE_SET for ch in candidate.content], dtype=bool)
        valid = np.logical_and(valid, ~ws_mask)

    total = int(valid.sum())
    if total <= 0:
        return 0.0, 0.0, 0, {}

    correct_mask = np.logical_and(valid, pred == truth)
    accuracy = float(correct_mask.sum() / total)

    present = sorted(int(v) for v in np.unique(truth[valid]))
    per_label: Dict[str, float] = {}
    macro_values: List[float] = []
    for label_id in present:
        label_mask = np.logical_and(valid, truth == label_id)
        support = int(label_mask.sum())
        if support <= 0:
            continue
        label_acc = float(np.logical_and(correct_mask, label_mask).sum() / support)
        label_name = id2lang.get(label_id, str(label_id))
        per_label[label_name] = label_acc
        macro_values.append(label_acc)
    macro_accuracy = float(np.mean(macro_values)) if macro_values else 0.0
    return accuracy, macro_accuracy, total, per_label


def _score_candidate(
    runner: SegmenterRunner,
    candidate: MonitorCandidate,
    *,
    min_run_chars: int,
    other_threshold: float,
    other_label_idx: Optional[int],
    id2lang: Dict[int, str],
) -> ScoredCandidate:
    started = time.perf_counter()
    pred = _predict_char_labels(
        runner,
        candidate.content,
        min_run_chars=min_run_chars,
        other_threshold=other_threshold,
        other_label_idx=other_label_idx,
    )
    elapsed = time.perf_counter() - started
    accuracy, macro_accuracy, total, per_label = _score_prediction(
        candidate,
        pred,
        id2lang=id2lang,
    )
    return ScoredCandidate(
        candidate=candidate,
        accuracy=accuracy,
        macro_accuracy=macro_accuracy,
        evaluated_chars=total,
        elapsed_seconds=float(elapsed),
        per_label_accuracy=per_label,
    )


def _run_length_sweep(
    runner: SegmenterRunner,
    seed: MonitorCandidate,
    *,
    repeat_factors: Sequence[int],
    min_run_chars: int,
    other_threshold: float,
    other_label_idx: Optional[int],
    id2lang: Dict[int, str],
) -> List[LengthSweepResult]:
    results: List[LengthSweepResult] = []
    for factor in repeat_factors:
        repeated = _repeat_candidate(seed, int(factor))
        started = time.perf_counter()
        pred = _predict_char_labels(
            runner,
            repeated.content,
            min_run_chars=min_run_chars,
            other_threshold=other_threshold,
            other_label_idx=other_label_idx,
        )
        elapsed = time.perf_counter() - started
        accuracy, macro_accuracy, total, per_label = _score_prediction(
            repeated,
            pred,
            id2lang=id2lang,
        )
        results.append(
            LengthSweepResult(
                repeat_factor=int(factor),
                total_chars=int(len(repeated.content)),
                total_bytes_utf8=int(len(repeated.content.encode("utf-8", "ignore"))),
                accuracy=float(accuracy),
                macro_accuracy=float(macro_accuracy),
                evaluated_chars=int(total),
                elapsed_seconds=float(elapsed),
                per_label_accuracy=per_label,
            )
        )
    return results


def _build_runner(args: argparse.Namespace) -> SegmenterRunner:
    ckpt_path = Path(args.checkpoint).resolve()
    auto = _load_checkpoint_hparams(ckpt_path)

    arch = str(args.arch or auto.get("arch", "mamba")).lower().strip() or "mamba"
    model_dim = int(args.model_dim if args.model_dim is not None else auto.get("model_dim", 256))
    channels_source = args.channels if args.channels else auto.get("channels", (96, 128, 192, 256))
    if isinstance(channels_source, str):
        channels = tuple(int(part.strip()) for part in channels_source.split(",") if part.strip())
    else:
        channels = tuple(int(v) for v in channels_source)

    return SegmenterRunner(
        str(ckpt_path),
        arch=arch,
        model_dim=model_dim,
        channels=channels,
        mamba_layers=int(args.mamba_layers if args.mamba_layers is not None else auto.get("mamba_layers", 6)),
        mamba_d_state=int(args.mamba_d_state if args.mamba_d_state is not None else auto.get("mamba_d_state", 16)),
        mamba_expand=int(args.mamba_expand if args.mamba_expand is not None else auto.get("mamba_expand", 1)),
        mamba_dt_rank=int(args.mamba_dt_rank if args.mamba_dt_rank is not None else auto.get("mamba_dt_rank", 16)),
        mamba_conv=int(args.mamba_conv if args.mamba_conv is not None else auto.get("mamba_conv", 4)),
        mamba_bidirectional=bool(
            args.mamba_bidirectional
            if args.mamba_bidirectional is not None
            else auto.get("mamba_bidirectional", True)
        ),
        dtype=str(args.dtype or auto.get("dtype", "bfloat16")).rsplit(".", 1)[-1],
        chunk=int(args.chunk),
        device=args.device,
        batch_size=int(args.batch_size),
        inference_backend=str(args.inference_backend),
    )


def _default_output_paths(checkpoint: Path) -> Tuple[Path, Path]:
    stem = checkpoint.stem
    return (
        REPO_ROOT / "evaluation" / f"length_extrapolation_{stem}.json",
        REPO_ROOT / "evaluation" / f"length_extrapolation_{stem}.md",
    )


def _render_markdown(
    *,
    args: argparse.Namespace,
    seed: ScoredCandidate,
    sweep_results: Sequence[LengthSweepResult],
    top_candidates: Sequence[ScoredCandidate],
) -> str:
    lines: List[str] = []
    lines.append(f"# Length Extrapolation: `{Path(args.checkpoint).name}`")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- Monitor root: `{Path(args.monitor_root).resolve()}`")
    lines.append(f"- Checkpoint: `{Path(args.checkpoint).resolve()}`")
    lines.append(f"- Target bytes: `{int(args.target_bytes)}`")
    lines.append(f"- Size tolerance: `{int(args.size_tolerance)}`")
    lines.append(f"- Min minority share: `{float(args.min_minority_share):.4f}`")
    lines.append(f"- Min distinct labels: `{int(args.min_distinct_labels)}`")
    lines.append(f"- Max candidate segments: `{args.max_candidate_segments}`")
    lines.append(f"- Device: `{args.device}`")
    lines.append(f"- Inference backend: `{args.inference_backend}`")
    lines.append(f"- Chunk: `{int(args.chunk)}`")
    lines.append(f"- Batch size: `{int(args.batch_size)}`")
    lines.append(f"- Min run chars: `{int(args.min_run)}`")
    lines.append(f"- Other threshold: `{float(args.other_threshold):.4f}`")
    lines.append(f"- Repeat factors: `{','.join(str(v) for v in args.repeat_factors)}`")
    if args.seed_file_idx is not None:
        lines.append(f"- Seed file index pinned: `{int(args.seed_file_idx)}`")
    if args.required_labels:
        lines.append(f"- Required labels: `{','.join(args.required_labels)}`")
    if args.excluded_labels:
        lines.append(f"- Excluded labels: `{','.join(args.excluded_labels)}`")
    lines.append("")
    lines.append("## Seed")
    lines.append("")
    seed_summary = seed.summary()
    lines.append(f"- File index: `{seed_summary['file_idx']}`")
    lines.append(f"- Base bytes: `{seed_summary['byte_len']}`")
    lines.append(f"- Base chars: `{seed_summary['char_len']}`")
    lines.append(f"- Labels: `{', '.join(seed_summary['distinct_labels'])}`")
    lines.append(f"- Segments: `{seed_summary['seg_count']}`")
    lines.append(f"- Minority share: `{seed_summary['minority_share']:.4f}`")
    lines.append(f"- Base accuracy: `{seed_summary['accuracy']:.6f}`")
    lines.append(f"- Base macro accuracy: `{seed_summary['macro_accuracy']:.6f}`")
    lines.append("")
    lines.append("## Sweep")
    lines.append("")
    lines.append("| Repeat | Bytes | Accuracy | Macro Accuracy | Evaluated Chars | Elapsed (s) |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in sweep_results:
        lines.append(
            f"| {row.repeat_factor} | {row.total_bytes_utf8} | {row.accuracy:.6f} | "
            f"{row.macro_accuracy:.6f} | {row.evaluated_chars} | {row.elapsed_seconds:.2f} |"
        )
    if top_candidates:
        lines.append("")
        lines.append("## Top Seed Candidates")
        lines.append("")
        lines.append("| Rank | File Idx | Labels | Segments | Minority Share | Accuracy | Macro Accuracy |")
        lines.append("|---:|---:|---|---:|---:|---:|---:|")
        for rank, cand in enumerate(top_candidates[:10], start=1):
            summary = cand.summary()
            lines.append(
                f"| {rank} | {summary['file_idx']} | {', '.join(summary['distinct_labels'])} | "
                f"{summary['seg_count']} | {summary['minority_share']:.4f} | "
                f"{summary['accuracy']:.6f} | {summary['macro_accuracy']:.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _score_candidates(
    runner: SegmenterRunner,
    candidates: Sequence[MonitorCandidate],
    *,
    min_run_chars: int,
    other_threshold: float,
    other_label_idx: Optional[int],
    id2lang: Dict[int, str],
) -> List[ScoredCandidate]:
    scored: List[ScoredCandidate] = []
    total = len(candidates)
    started = time.perf_counter()
    for idx, candidate in enumerate(candidates, start=1):
        result = _score_candidate(
            runner,
            candidate,
            min_run_chars=min_run_chars,
            other_threshold=other_threshold,
            other_label_idx=other_label_idx,
            id2lang=id2lang,
        )
        scored.append(result)
        if idx % 10 == 0 or idx == total:
            elapsed = time.perf_counter() - started
            print(f"[seed-scan] scored {idx}/{total} candidates in {elapsed:.1f}s", flush=True)
    scored.sort(
        key=lambda item: (
            -float(item.accuracy),
            -float(item.macro_accuracy),
            -len(item.candidate.distinct_labels),
            -int(item.candidate.seg_count),
            -float(item.candidate.minority_share),
            int(item.candidate.file_idx),
        )
    )
    return scored


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_json, default_md = _default_output_paths(DEFAULT_SFULLFILES3_CHECKPOINT)

    parser = argparse.ArgumentParser(
        description="Repeat one mixed near-10k monitor file and measure quality drift with length."
    )
    parser.add_argument(
        "--monitor-root",
        default=str(REPO_ROOT / "downloader" / "monitor_preprocessed_b"),
        help="Preprocessed monitor memmap root.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_SFULLFILES3_CHECKPOINT),
        help="Checkpoint to evaluate.",
    )
    parser.add_argument("--arch", default="mamba", help="Model architecture (default tuned for sfullfiles3).")
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--channels", default="96,128,192,256")
    parser.add_argument("--mamba-layers", type=int, default=6)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-expand", type=int, default=1)
    parser.add_argument("--mamba-dt-rank", type=int, default=16)
    parser.add_argument("--mamba-conv", type=int, default=4)
    parser.add_argument("--mamba-bidirectional", action="store_true", default=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chunk", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-backend", default="auto")
    parser.add_argument("--min-run", type=int, default=1)
    parser.add_argument("--other-threshold", type=float, default=0.65)

    parser.add_argument("--target-bytes", type=int, default=10000)
    parser.add_argument("--size-tolerance", type=int, default=512)
    parser.add_argument("--min-minority-share", type=float, default=0.02)
    parser.add_argument("--min-distinct-labels", type=int, default=2)
    parser.add_argument(
        "--max-candidate-segments",
        type=int,
        default=None,
        help="Optional upper bound on truth segment count for seed candidates.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="If > 0, only score the nearest N candidates after filtering.",
    )
    parser.add_argument(
        "--required-labels",
        nargs="*",
        default=[],
        help="Optional labels that must all be present in the seed candidate.",
    )
    parser.add_argument(
        "--excluded-labels",
        nargs="*",
        default=[],
        help="Optional labels to exclude from seed candidates.",
    )
    parser.add_argument(
        "--seed-file-idx",
        type=int,
        default=None,
        help="Pin a specific monitor file index instead of auto-selecting a seed.",
    )
    parser.add_argument(
        "--repeat-factors",
        nargs="*",
        default=[",".join(str(v) for v in DEFAULT_REPEAT_FACTORS)],
        help="Repeat factors, e.g. 1,2,4,8,16,32.",
    )
    parser.add_argument("--output-json", default=str(default_json))
    parser.add_argument("--output-md", default=str(default_md))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.repeat_factors = _parse_repeat_factors(args.repeat_factors)

    monitor_root = Path(args.monitor_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not monitor_root.exists():
        raise FileNotFoundError(f"Monitor root not found: {monitor_root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    id2lang, other_idx = _label_mappings()
    candidates = _load_mixed_candidates(
        monitor_root,
        target_bytes=int(args.target_bytes),
        size_tolerance=int(args.size_tolerance),
        min_minority_share=float(args.min_minority_share),
        min_distinct_labels=int(args.min_distinct_labels),
        max_segments=args.max_candidate_segments,
        required_labels=args.required_labels,
        excluded_labels=args.excluded_labels,
    )
    if not candidates:
        raise RuntimeError("No mixed candidates matched the requested constraints.")

    if int(args.max_candidates) > 0 and len(candidates) > int(args.max_candidates):
        print(
            f"[seed-scan] limiting candidates from {len(candidates)} to {int(args.max_candidates)}",
            flush=True,
        )
        candidates = candidates[: int(args.max_candidates)]
    else:
        print(f"[seed-scan] scoring {len(candidates)} candidates", flush=True)

    runner = _build_runner(args)

    scored_candidates: List[ScoredCandidate] = []
    if args.seed_file_idx is not None:
        match = next((cand for cand in candidates if int(cand.file_idx) == int(args.seed_file_idx)), None)
        if match is None:
            raise RuntimeError(
                f"seed_file_idx={int(args.seed_file_idx)} did not match any filtered candidate."
            )
        seed = _score_candidate(
            runner,
            match,
            min_run_chars=int(args.min_run),
            other_threshold=float(args.other_threshold),
            other_label_idx=other_idx,
            id2lang=id2lang,
        )
        scored_candidates = [seed]
    else:
        scored_candidates = _score_candidates(
            runner,
            candidates,
            min_run_chars=int(args.min_run),
            other_threshold=float(args.other_threshold),
            other_label_idx=other_idx,
            id2lang=id2lang,
        )
        seed = scored_candidates[0]

    print(
        "[seed] selected "
        f"file_idx={seed.candidate.file_idx} labels={list(seed.candidate.distinct_labels)} "
        f"acc={seed.accuracy:.6f} minority={seed.candidate.minority_share:.4f}",
        flush=True,
    )

    sweep_results = _run_length_sweep(
        runner,
        seed.candidate,
        repeat_factors=args.repeat_factors,
        min_run_chars=int(args.min_run),
        other_threshold=float(args.other_threshold),
        other_label_idx=other_idx,
        id2lang=id2lang,
    )
    for row in sweep_results:
        print(
            f"[repeat={row.repeat_factor}] bytes={row.total_bytes_utf8} "
            f"acc={row.accuracy:.6f} macro={row.macro_accuracy:.6f} "
            f"elapsed={row.elapsed_seconds:.2f}s",
            flush=True,
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "monitor_root": str(monitor_root),
            "checkpoint": str(checkpoint),
            "arch": str(args.arch),
            "model_dim": int(args.model_dim),
            "channels": str(args.channels),
            "mamba_layers": int(args.mamba_layers),
            "mamba_d_state": int(args.mamba_d_state),
            "mamba_expand": int(args.mamba_expand),
            "mamba_dt_rank": int(args.mamba_dt_rank),
            "mamba_conv": int(args.mamba_conv),
            "mamba_bidirectional": bool(args.mamba_bidirectional),
            "dtype": str(args.dtype),
            "chunk": int(args.chunk),
            "batch_size": int(args.batch_size),
            "device": str(args.device),
            "inference_backend": str(args.inference_backend),
            "min_run": int(args.min_run),
            "other_threshold": float(args.other_threshold),
            "target_bytes": int(args.target_bytes),
            "size_tolerance": int(args.size_tolerance),
            "min_minority_share": float(args.min_minority_share),
            "min_distinct_labels": int(args.min_distinct_labels),
            "max_candidate_segments": args.max_candidate_segments,
            "max_candidates": int(args.max_candidates),
            "required_labels": list(args.required_labels),
            "excluded_labels": list(args.excluded_labels),
            "seed_file_idx": args.seed_file_idx,
            "repeat_factors": list(args.repeat_factors),
        },
        "seed": seed.summary(),
        "results": [row.summary() for row in sweep_results],
        "top_candidates": [cand.summary() for cand in scored_candidates[:10]],
    }

    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(
        _render_markdown(
            args=args,
            seed=seed,
            sweep_results=sweep_results,
            top_candidates=scored_candidates[:10],
        ),
        encoding="utf-8",
    )
    print(f"[write] json -> {output_json}", flush=True)
    print(f"[write] md   -> {output_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
