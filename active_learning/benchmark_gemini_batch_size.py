from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from datasets import concatenate_datasets, load_from_disk  # type: ignore
try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    _tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from .oracle import (  # noqa: E402
    BoundarySnippet,
    GeminiBoundaryOracle,
    OracleSegment,
    StubOracle,
    normalize_label_to_allowed,
)


DEFAULT_TASKS: tuple[str, ...] = (
    "pure_fragments",
    "sequence_pair",
    "sequence_triplet",
    "markdown_mix",
    "restructuredtext_mix",
    "mal_injection",
)
DEFAULT_MIXED_WEIGHTS: Mapping[str, float] = {
    "sequence_pair": 0.30,
    "sequence_triplet": 0.30,
    "mal_injection": 0.25,
    "markdown_mix": 0.08,
    "restructuredtext_mix": 0.07,
}
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "active_learning" / "benchmark_results"
DEFAULT_CURATED_BENCHMARK_DATASET = (
    REPO_ROOT / "active_learning" / "benchmark_data" / "curated_oracle_segments_v1.json"
)


@dataclass(frozen=True)
class GoldSegment:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class BenchmarkSample:
    snippet_id: str
    text: str
    boundary: int
    truth_labels: tuple[str, ...]
    truth_segments: tuple[GoldSegment, ...]
    predicted_labels: tuple[str, ...]
    task: str
    example_id: str
    mixed_truth: bool
    source_langs: tuple[str, ...]
    metadata: Dict[str, object]

    def to_boundary_snippet(self) -> BoundarySnippet:
        source_lang = ""
        if self.source_langs:
            source_lang = str(self.source_langs[0])
        elif self.truth_segments:
            source_lang = str(self.truth_segments[0].label)
        source_lang = _canonical_label(source_lang)
        metadata = dict(self.metadata)
        metadata["source_lang"] = source_lang
        metadata["benchmark_task"] = self.task
        metadata["benchmark_mixed_truth"] = bool(self.mixed_truth)
        metadata["benchmark_example_id"] = self.example_id
        metadata["benchmark_source_langs"] = list(self.source_langs)
        return BoundarySnippet(
            snippet_id=self.snippet_id,
            text=self.text,
            global_start=0,
            global_end=len(self.text),
            boundary=int(self.boundary),
            predicted_labels=list(self.predicted_labels),
            metadata=metadata,
        )

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "snippet_id": self.snippet_id,
            "text": self.text,
            "boundary": int(self.boundary),
            "task": self.task,
            "example_id": self.example_id,
            "mixed_truth": bool(self.mixed_truth),
            "source_langs": list(self.source_langs),
            "truth_segments": [segment.__dict__ for segment in self.truth_segments],
            "predicted_segments": [segment.__dict__ for segment in _segments_from_labels(self.predicted_labels)],
        }


@dataclass
class ScoreAccumulator:
    sample_count: int = 0
    chars_total: int = 0
    chars_correct: int = 0
    exact_match_samples: int = 0
    boundary_true_total: int = 0
    boundary_pred_total: int = 0
    boundary_exact_tp: int = 0
    boundary_tol_tp: int = 0
    tp_by_label: Counter[str] = field(default_factory=Counter)
    fp_by_label: Counter[str] = field(default_factory=Counter)
    fn_by_label: Counter[str] = field(default_factory=Counter)
    support_by_label: Counter[str] = field(default_factory=Counter)

    def add(self, truth_labels: Sequence[str], pred_labels: Sequence[str], tolerance: int) -> None:
        if not truth_labels or not pred_labels or len(truth_labels) != len(pred_labels):
            return
        n = len(truth_labels)
        self.sample_count += 1
        self.chars_total += n
        sample_correct = 0
        labels_seen = set(truth_labels).union(pred_labels)
        for idx in range(n):
            truth = str(truth_labels[idx])
            pred = str(pred_labels[idx])
            self.support_by_label[truth] += 1
            if truth == pred:
                sample_correct += 1
                self.tp_by_label[truth] += 1
            else:
                self.fp_by_label[pred] += 1
                self.fn_by_label[truth] += 1
        self.chars_correct += sample_correct
        if sample_correct == n:
            self.exact_match_samples += 1
        for label in labels_seen:
            # Ensure labels with only FN/FP still appear in macro averaging.
            _ = self.tp_by_label[label]

        truth_boundaries = _boundaries_from_labels(truth_labels)
        pred_boundaries = _boundaries_from_labels(pred_labels)
        self.boundary_true_total += len(truth_boundaries)
        self.boundary_pred_total += len(pred_boundaries)
        self.boundary_exact_tp += len(set(truth_boundaries).intersection(pred_boundaries))
        self.boundary_tol_tp += _tolerant_boundary_tp(
            truth_boundaries=truth_boundaries,
            pred_boundaries=pred_boundaries,
            tolerance=max(0, int(tolerance)),
        )

    def finalize(self) -> Dict[str, object]:
        macro_values: List[float] = []
        weighted_num = 0.0
        weighted_den = 0.0
        for label, support in sorted(self.support_by_label.items()):
            tp = int(self.tp_by_label.get(label, 0))
            fp = int(self.fp_by_label.get(label, 0))
            fn = int(self.fn_by_label.get(label, 0))
            f1 = _f1(tp, fp, fn)
            macro_values.append(f1)
            weighted_num += float(support) * float(f1)
            weighted_den += float(support)

        boundary_exact_precision = _safe_ratio(self.boundary_exact_tp, self.boundary_pred_total)
        boundary_exact_recall = _safe_ratio(self.boundary_exact_tp, self.boundary_true_total)
        boundary_tol_precision = _safe_ratio(self.boundary_tol_tp, self.boundary_pred_total)
        boundary_tol_recall = _safe_ratio(self.boundary_tol_tp, self.boundary_true_total)

        return {
            "samples": int(self.sample_count),
            "chars_total": int(self.chars_total),
            "char_accuracy": _safe_ratio(self.chars_correct, self.chars_total),
            "exact_match_rate": _safe_ratio(self.exact_match_samples, self.sample_count),
            "macro_f1": float(np.mean(macro_values)) if macro_values else 0.0,
            "weighted_f1": _safe_ratio(weighted_num, weighted_den),
            "boundary_precision_exact": boundary_exact_precision,
            "boundary_recall_exact": boundary_exact_recall,
            "boundary_f1_exact": _f1_from_pr(boundary_exact_precision, boundary_exact_recall),
            "boundary_precision_tolerant": boundary_tol_precision,
            "boundary_recall_tolerant": boundary_tol_recall,
            "boundary_f1_tolerant": _f1_from_pr(boundary_tol_precision, boundary_tol_recall),
            "boundary_true_total": int(self.boundary_true_total),
            "boundary_pred_total": int(self.boundary_pred_total),
            "boundary_exact_tp": int(self.boundary_exact_tp),
            "boundary_tolerant_tp": int(self.boundary_tol_tp),
        }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _canonical_label(label: str) -> str:
    normalized, _ = normalize_label_to_allowed(str(label))
    return str(normalized)


def _merge_segments(segments: Iterable[GoldSegment]) -> List[GoldSegment]:
    ordered = sorted(segments, key=lambda seg: (int(seg.start), int(seg.end)))
    merged: List[GoldSegment] = []
    for segment in ordered:
        start = int(segment.start)
        end = int(segment.end)
        if end <= start:
            continue
        label = _canonical_label(segment.label)
        if not merged:
            merged.append(GoldSegment(start=start, end=end, label=label))
            continue
        last = merged[-1]
        if last.label == label and start <= last.end:
            merged[-1] = GoldSegment(start=last.start, end=max(last.end, end), label=last.label)
            continue
        if last.label == label and start == last.end:
            merged[-1] = GoldSegment(start=last.start, end=end, label=last.label)
            continue
        merged.append(GoldSegment(start=start, end=end, label=label))
    return merged


def _segments_from_row(row: Mapping[str, object]) -> List[GoldSegment]:
    seg_obj = row.get("segments")
    if not isinstance(seg_obj, dict):
        return []
    labels = seg_obj.get("label")
    starts = seg_obj.get("char_start")
    ends = seg_obj.get("char_end")
    if not isinstance(labels, list) or not isinstance(starts, list) or not isinstance(ends, list):
        return []
    segments: List[GoldSegment] = []
    for label, start, end in zip(labels, starts, ends):
        try:
            s = int(start)
            e = int(end)
        except (TypeError, ValueError):
            continue
        if e <= s:
            continue
        segments.append(GoldSegment(start=s, end=e, label=_canonical_label(str(label))))
    return _merge_segments(segments)


def _slice_segments(
    segments: Sequence[GoldSegment],
    window_start: int,
    window_end: int,
) -> List[GoldSegment]:
    out: List[GoldSegment] = []
    for segment in segments:
        if segment.end <= window_start:
            continue
        if segment.start >= window_end:
            continue
        start = max(window_start, segment.start) - window_start
        end = min(window_end, segment.end) - window_start
        if end <= start:
            continue
        out.append(GoldSegment(start=start, end=end, label=segment.label))
    return _merge_segments(out)


def _segments_from_labels(
    labels: Sequence[str],
) -> List[GoldSegment]:
    if not labels:
        return []
    out: List[GoldSegment] = []
    start = 0
    cur = _canonical_label(str(labels[0]))
    for idx in range(1, len(labels)):
        nxt = _canonical_label(str(labels[idx]))
        if nxt != cur:
            out.append(GoldSegment(start=start, end=idx, label=cur))
            start = idx
            cur = nxt
    out.append(GoldSegment(start=start, end=len(labels), label=cur))
    return out


def _dense_labels_for_window(window_len: int, segments: Sequence[GoldSegment]) -> List[str]:
    if window_len <= 0:
        return []
    dense = ["other"] * int(window_len)
    for segment in segments:
        start = max(0, min(window_len, int(segment.start)))
        end = max(start, min(window_len, int(segment.end)))
        if end <= start:
            continue
        label = _canonical_label(str(segment.label))
        for idx in range(start, end):
            dense[idx] = label
    return dense


def _boundaries_from_labels(labels: Sequence[str]) -> List[int]:
    out: List[int] = []
    for idx in range(1, len(labels)):
        if labels[idx] != labels[idx - 1]:
            out.append(idx)
    return out


def _tolerant_boundary_tp(
    truth_boundaries: Sequence[int],
    pred_boundaries: Sequence[int],
    tolerance: int,
) -> int:
    if not truth_boundaries or not pred_boundaries:
        return 0
    truth_used = [False] * len(truth_boundaries)
    tp = 0
    for pred in pred_boundaries:
        best_idx = -1
        best_dist = None
        for idx, truth in enumerate(truth_boundaries):
            if truth_used[idx]:
                continue
            dist = abs(int(pred) - int(truth))
            if dist > tolerance:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0:
            truth_used[best_idx] = True
            tp += 1
    return tp


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * int(tp)) + int(fp) + int(fn)
    if denom <= 0:
        return 0.0
    return (2.0 * float(tp)) / float(denom)


def _f1_from_pr(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return (2.0 * float(precision) * float(recall)) / float(precision + recall)


def _parse_csv_str(raw: str) -> List[str]:
    parts = [part.strip() for part in str(raw or "").split(",")]
    return [part for part in parts if part]


def _parse_batch_sizes(raw: str) -> List[int]:
    out: List[int] = []
    for token in _parse_csv_str(raw):
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0 and value not in out:
            out.append(value)
    return sorted(out)


def _round_weighted_targets(total: int, weights: Mapping[str, float]) -> Dict[str, int]:
    if total <= 0 or not weights:
        return {key: 0 for key in weights}
    positive = {k: float(v) for k, v in weights.items() if float(v) > 0.0}
    if not positive:
        uniform = int(math.floor(float(total) / float(max(1, len(weights)))))
        out = {key: uniform for key in weights}
        rem = max(0, int(total) - sum(out.values()))
        for key in weights:
            if rem <= 0:
                break
            out[key] += 1
            rem -= 1
        return out

    weight_sum = sum(positive.values())
    raw_targets = {key: (float(total) * weight / weight_sum) for key, weight in positive.items()}
    floors = {key: int(math.floor(value)) for key, value in raw_targets.items()}
    remainder = int(total) - sum(floors.values())
    ranked_remainders = sorted(
        ((raw_targets[key] - floors[key], key) for key in floors),
        reverse=True,
    )
    out = {key: floors.get(key, 0) for key in weights}
    for _, key in ranked_remainders:
        if remainder <= 0:
            break
        out[key] += 1
        remainder -= 1
    return out


def _build_predicted_labels(
    truth_labels: Sequence[str],
    *,
    rng: np.random.Generator,
    max_boundary_shift: int,
) -> List[str]:
    pred = [_canonical_label(label) for label in truth_labels]
    if max_boundary_shift <= 0:
        return pred
    boundaries = _boundaries_from_labels(pred)
    if not boundaries:
        return pred
    n = len(pred)
    truth = list(pred)
    for boundary in boundaries:
        shift = int(rng.integers(-max_boundary_shift, max_boundary_shift + 1))
        if shift < 0:
            fill = truth[boundary]
            start = max(0, boundary + shift)
            for idx in range(start, boundary):
                pred[idx] = fill
        elif shift > 0:
            fill = truth[boundary - 1]
            end = min(n, boundary + shift)
            for idx in range(boundary, end):
                pred[idx] = fill
    return pred


def _choose_pure_window(
    *,
    content_len: int,
    segments: Sequence[GoldSegment],
    sample_length: int,
    rng: np.random.Generator,
) -> Optional[tuple[int, int, List[GoldSegment], int]]:
    candidates = [segment for segment in segments if int(segment.end) - int(segment.start) >= sample_length]
    if not candidates or content_len < sample_length:
        return None
    segment = candidates[int(rng.integers(0, len(candidates)))]
    start_low = int(segment.start)
    start_high = int(segment.end) - int(sample_length)
    if start_high < start_low:
        return None
    start = int(rng.integers(start_low, start_high + 1))
    end = start + int(sample_length)
    sliced = _slice_segments(segments, start, end)
    if len(sliced) != 1:
        return None
    return start, end, sliced, sample_length // 2


def _choose_mixed_window(
    *,
    content_len: int,
    segments: Sequence[GoldSegment],
    sample_length: int,
    boundary_center_jitter: int,
    min_boundary_margin: int,
    rng: np.random.Generator,
) -> Optional[tuple[int, int, List[GoldSegment], int]]:
    if content_len < sample_length:
        return None
    boundaries = [int(segment.end) for segment in segments[:-1] if 0 < int(segment.end) < content_len]
    if not boundaries:
        return None
    boundaries = list(boundaries)
    rng.shuffle(boundaries)
    max_start = int(content_len) - int(sample_length)
    for boundary in boundaries:
        for _ in range(8):
            jitter = int(rng.integers(-boundary_center_jitter, boundary_center_jitter + 1))
            start = int(boundary) - (sample_length // 2) + jitter
            start = max(0, min(max_start, start))
            end = start + int(sample_length)
            sliced = _slice_segments(segments, start, end)
            if len(sliced) < 2:
                continue
            local_boundaries = [int(segment.end) for segment in sliced[:-1]]
            if not local_boundaries:
                continue
            chosen_boundary = min(local_boundaries, key=lambda value: abs(value - (sample_length // 2)))
            if chosen_boundary <= int(min_boundary_margin):
                continue
            if chosen_boundary >= int(sample_length - min_boundary_margin):
                continue
            return start, end, sliced, int(chosen_boundary)
    return None


def _load_datasets(
    *,
    roots: Sequence[Path],
    tasks: Sequence[str],
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for task in tasks:
        datasets_for_task = []
        for root in roots:
            path = root / task
            if not path.exists():
                continue
            datasets_for_task.append(load_from_disk(str(path)))
        if not datasets_for_task:
            continue
        if len(datasets_for_task) == 1:
            out[task] = datasets_for_task[0]
        else:
            out[task] = concatenate_datasets(datasets_for_task)
    return out


def _sample_task_rows(
    *,
    dataset: object,
    task: str,
    target_count: int,
    sample_length: int,
    mixed: bool,
    seen_hashes: set[str],
    boundary_center_jitter: int,
    min_boundary_margin: int,
    max_prior_boundary_shift: int,
    rng: np.random.Generator,
    start_index: int,
) -> List[BenchmarkSample]:
    out: List[BenchmarkSample] = []
    total = int(len(dataset))
    if total <= 0 or target_count <= 0:
        return out
    order = rng.permutation(total).tolist()
    for row_idx in order:
        if len(out) >= int(target_count):
            break
        row = dataset[int(row_idx)]
        if not isinstance(row, dict):
            continue
        content = row.get("content")
        if not isinstance(content, str):
            continue
        if len(content) < int(sample_length):
            continue
        full_segments = _segments_from_row(row)
        if not full_segments:
            continue
        selected = None
        if mixed:
            selected = _choose_mixed_window(
                content_len=len(content),
                segments=full_segments,
                sample_length=int(sample_length),
                boundary_center_jitter=int(boundary_center_jitter),
                min_boundary_margin=int(min_boundary_margin),
                rng=rng,
            )
        else:
            selected = _choose_pure_window(
                content_len=len(content),
                segments=full_segments,
                sample_length=int(sample_length),
                rng=rng,
            )
        if selected is None:
            continue
        window_start, window_end, window_segments, boundary = selected
        truth_labels = _dense_labels_for_window(int(sample_length), window_segments)
        if len(truth_labels) != int(sample_length):
            continue
        unique_truth = sorted(set(truth_labels))
        if mixed and len(unique_truth) < 2:
            continue
        if (not mixed) and len(unique_truth) != 1:
            continue
        text = content[int(window_start) : int(window_end)]
        if len(text) != int(sample_length):
            continue
        text_hash = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        predicted_labels = _build_predicted_labels(
            truth_labels,
            rng=rng,
            max_boundary_shift=int(max_prior_boundary_shift),
        )
        snippet_index = int(start_index + len(out))
        snippet_id = f"bench-{snippet_index:04d}-{text_hash[:10]}"
        source_langs = tuple(str(lang) for lang in (row.get("source_langs") or []))
        metadata = {
            "window_start": int(window_start),
            "window_end": int(window_end),
            "window_text_sha256": text_hash,
            "row_index": int(row_idx),
        }
        out.append(
            BenchmarkSample(
                snippet_id=snippet_id,
                text=text,
                boundary=int(boundary),
                truth_labels=tuple(truth_labels),
                truth_segments=tuple(window_segments),
                predicted_labels=tuple(predicted_labels),
                task=str(task),
                example_id=str(row.get("example_id", f"{task}-{row_idx}")),
                mixed_truth=bool(mixed),
                source_langs=source_langs,
                metadata=metadata,
            )
        )
    return out


def _build_benchmark_samples(
    *,
    datasets_by_task: Mapping[str, object],
    tasks: Sequence[str],
    total_samples: int,
    pure_samples: int,
    sample_length: int,
    boundary_center_jitter: int,
    min_boundary_margin: int,
    max_prior_boundary_shift: int,
    rng: np.random.Generator,
) -> List[BenchmarkSample]:
    requested_total = max(1, int(total_samples))
    pure_target = max(0, min(int(pure_samples), requested_total))
    mixed_target = max(0, requested_total - pure_target)
    samples: List[BenchmarkSample] = []
    seen_hashes: set[str] = set()

    if pure_target > 0:
        pure_dataset = datasets_by_task.get("pure_fragments")
        if pure_dataset is None:
            raise RuntimeError(
                "Requested non-mixed samples but task 'pure_fragments' is not available in --data-roots."
            )
        pure_rows = _sample_task_rows(
            dataset=pure_dataset,
            task="pure_fragments",
            target_count=int(pure_target),
            sample_length=int(sample_length),
            mixed=False,
            seen_hashes=seen_hashes,
            boundary_center_jitter=int(boundary_center_jitter),
            min_boundary_margin=int(min_boundary_margin),
            max_prior_boundary_shift=0,
            rng=rng,
            start_index=len(samples),
        )
        samples.extend(pure_rows)

    mixed_tasks = [task for task in tasks if task != "pure_fragments" and task in datasets_by_task]
    if mixed_target > 0 and not mixed_tasks:
        raise RuntimeError("Requested mixed samples but no mixed tasks are available in --data-roots.")
    weights = {
        task: float(DEFAULT_MIXED_WEIGHTS.get(task, 1.0))
        for task in mixed_tasks
    }
    mixed_targets = _round_weighted_targets(mixed_target, weights)
    for task in mixed_tasks:
        task_target = int(mixed_targets.get(task, 0))
        if task_target <= 0:
            continue
        mixed_rows = _sample_task_rows(
            dataset=datasets_by_task[task],
            task=task,
            target_count=task_target,
            sample_length=int(sample_length),
            mixed=True,
            seen_hashes=seen_hashes,
            boundary_center_jitter=int(boundary_center_jitter),
            min_boundary_margin=int(min_boundary_margin),
            max_prior_boundary_shift=int(max_prior_boundary_shift),
            rng=rng,
            start_index=len(samples),
        )
        samples.extend(mixed_rows)

    if len(samples) < requested_total:
        missing = requested_total - len(samples)
        print(
            f"Sampling fallback: still missing {missing} snippets; filling from any available task.",
            flush=True,
        )
        fallback_plan: List[tuple[str, bool]] = []
        for task in tasks:
            if task not in datasets_by_task:
                continue
            if task == "pure_fragments":
                fallback_plan.append((task, False))
            else:
                fallback_plan.append((task, True))
        for task, mixed in fallback_plan:
            if len(samples) >= requested_total:
                break
            additional = _sample_task_rows(
                dataset=datasets_by_task[task],
                task=task,
                target_count=requested_total - len(samples),
                sample_length=int(sample_length),
                mixed=bool(mixed),
                seen_hashes=seen_hashes,
                boundary_center_jitter=int(boundary_center_jitter),
                min_boundary_margin=int(min_boundary_margin),
                max_prior_boundary_shift=int(max_prior_boundary_shift if mixed else 0),
                rng=rng,
                start_index=len(samples),
            )
            samples.extend(additional)

    if len(samples) < requested_total:
        raise RuntimeError(
            f"Could not build enough benchmark snippets: requested={requested_total}, built={len(samples)}."
        )

    return samples[:requested_total]


def _segments_from_payload_rows(rows: Sequence[Mapping[str, object]], text_len: int) -> List[GoldSegment]:
    out: List[GoldSegment] = []
    for row in rows:
        try:
            start = int(row.get("start", 0))
            end = int(row.get("end", 0))
            label = _canonical_label(str(row.get("label", "other")))
        except Exception:
            continue
        start = max(0, min(text_len, start))
        end = max(start, min(text_len, end))
        if end <= start:
            continue
        out.append(GoldSegment(start=start, end=end, label=label))
    return _merge_segments(out)


def _load_curated_benchmark_pool(dataset_path: Path) -> List[BenchmarkSample]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    sample_rows = raw.get("samples")
    if not isinstance(sample_rows, list) or not sample_rows:
        raise RuntimeError(f"Curated benchmark dataset has no samples: {dataset_path}")

    out: List[BenchmarkSample] = []
    seen_ids: set[str] = set()
    for idx, sample in enumerate(sample_rows):
        if not isinstance(sample, dict):
            continue
        text = sample.get("text")
        if not isinstance(text, str) or not text:
            continue
        text_len = len(text)
        truth_rows = sample.get("truth_segments")
        if not isinstance(truth_rows, list):
            continue
        truth_segments = _segments_from_payload_rows(
            [row for row in truth_rows if isinstance(row, dict)],
            text_len=text_len,
        )
        if not truth_segments:
            continue
        truth_labels = tuple(_dense_labels_for_window(text_len, truth_segments))

        predicted_rows = sample.get("predicted_segments")
        if isinstance(predicted_rows, list):
            predicted_segments = _segments_from_payload_rows(
                [row for row in predicted_rows if isinstance(row, dict)],
                text_len=text_len,
            )
            predicted_labels = tuple(_dense_labels_for_window(text_len, predicted_segments))
        else:
            predicted_labels = tuple(truth_labels)

        snippet_id = str(sample.get("snippet_id", "")).strip() or f"curated-{idx:04d}"
        if snippet_id in seen_ids:
            snippet_id = f"{snippet_id}-{idx}"
        seen_ids.add(snippet_id)

        boundary_val = sample.get("boundary")
        try:
            boundary = int(boundary_val)
        except (TypeError, ValueError):
            boundary = text_len // 2
        if boundary <= 0 or boundary >= text_len:
            boundaries = [seg.end for seg in truth_segments[:-1]]
            if boundaries:
                center = text_len // 2
                boundary = min(boundaries, key=lambda x: abs(int(x) - int(center)))
            else:
                boundary = text_len // 2

        mixed_truth = bool(sample.get("mixed_truth", len(set(truth_labels)) > 1))
        source_langs_obj = sample.get("source_langs")
        source_langs: List[str] = []
        if isinstance(source_langs_obj, list):
            source_langs = [_canonical_label(str(label)) for label in source_langs_obj if str(label).strip()]
        if not source_langs:
            source_langs = sorted({segment.label for segment in truth_segments})
        metadata_obj = sample.get("metadata")
        metadata = dict(metadata_obj) if isinstance(metadata_obj, dict) else {}
        metadata["benchmark_origin"] = "curated_dataset"
        metadata["benchmark_dataset_path"] = str(dataset_path)
        out.append(
            BenchmarkSample(
                snippet_id=snippet_id,
                text=text,
                boundary=int(boundary),
                truth_labels=truth_labels,
                truth_segments=tuple(truth_segments),
                predicted_labels=predicted_labels,
                task=str(sample.get("task", "curated")),
                example_id=str(sample.get("example_id", snippet_id)),
                mixed_truth=bool(mixed_truth),
                source_langs=tuple(source_langs),
                metadata=metadata,
            )
        )
    if not out:
        raise RuntimeError(f"Curated benchmark dataset had no valid parsed samples: {dataset_path}")
    return out


def _select_curated_samples(
    *,
    pool: Sequence[BenchmarkSample],
    total_samples: int,
    pure_samples: int,
    rng: np.random.Generator,
) -> List[BenchmarkSample]:
    requested_total = max(1, int(total_samples))
    if requested_total > len(pool):
        raise RuntimeError(
            f"Requested {requested_total} snippets but curated pool only has {len(pool)}."
        )
    pure_target = max(0, min(int(pure_samples), requested_total))
    mixed_target = max(0, requested_total - pure_target)

    pure_pool = [sample for sample in pool if not sample.mixed_truth]
    mixed_pool = [sample for sample in pool if sample.mixed_truth]
    if pure_target > len(pure_pool):
        raise RuntimeError(
            f"Requested {pure_target} non-mixed snippets, but curated pool has only {len(pure_pool)}."
        )
    if mixed_target > len(mixed_pool):
        raise RuntimeError(
            f"Requested {mixed_target} mixed snippets, but curated pool has only {len(mixed_pool)}."
        )

    by_id = {sample.snippet_id: sample for sample in pool}
    selected_ids: set[str] = set()
    pure_used = 0
    mixed_used = 0

    def _try_add(sample: BenchmarkSample) -> bool:
        nonlocal pure_used, mixed_used
        if sample.snippet_id in selected_ids:
            return False
        if sample.mixed_truth:
            if mixed_used >= mixed_target:
                return False
            mixed_used += 1
        else:
            if pure_used >= pure_target:
                return False
            pure_used += 1
        selected_ids.add(sample.snippet_id)
        return True

    # Pass 1: guarantee broad label coverage where possible.
    labels = sorted({label for sample in pool for label in sample.source_langs})
    for label in labels:
        candidates = [sample for sample in pool if label in sample.source_langs and sample.snippet_id not in selected_ids]
        if not candidates:
            continue
        perm = rng.permutation(len(candidates)).tolist()
        for idx in perm:
            if _try_add(candidates[idx]):
                break

    # Pass 2: guarantee broad task coverage where possible.
    tasks = sorted({sample.task for sample in pool})
    for task in tasks:
        candidates = [sample for sample in pool if sample.task == task and sample.snippet_id not in selected_ids]
        if not candidates:
            continue
        perm = rng.permutation(len(candidates)).tolist()
        for idx in perm:
            if _try_add(candidates[idx]):
                break

    # Pass 3: fill pure quota randomly.
    pure_remaining = pure_target - pure_used
    if pure_remaining > 0:
        pure_candidates = [sample for sample in pure_pool if sample.snippet_id not in selected_ids]
        if pure_remaining > len(pure_candidates):
            raise RuntimeError(
                f"Unable to fill pure quota after diversity passes: need={pure_remaining}, available={len(pure_candidates)}."
            )
        perm = rng.permutation(len(pure_candidates))[:pure_remaining].tolist()
        for idx in perm:
            added = _try_add(pure_candidates[idx])
            if not added:
                continue

    # Pass 4: fill mixed quota randomly.
    mixed_remaining = mixed_target - mixed_used
    if mixed_remaining > 0:
        mixed_candidates = [sample for sample in mixed_pool if sample.snippet_id not in selected_ids]
        if mixed_remaining > len(mixed_candidates):
            raise RuntimeError(
                f"Unable to fill mixed quota after diversity passes: need={mixed_remaining}, available={len(mixed_candidates)}."
            )
        perm = rng.permutation(len(mixed_candidates))[:mixed_remaining].tolist()
        for idx in perm:
            added = _try_add(mixed_candidates[idx])
            if not added:
                continue

    selected = [by_id[sid] for sid in selected_ids]
    if len(selected) < requested_total:
        remainder = [sample for sample in pool if sample.snippet_id not in selected_ids]
        needed = requested_total - len(selected)
        perm = rng.permutation(len(remainder))[:needed].tolist()
        for idx in perm:
            selected.append(remainder[idx])

    if len(selected) > requested_total:
        perm = rng.permutation(len(selected))[:requested_total].tolist()
        selected = [selected[idx] for idx in perm]

    rng.shuffle(selected)
    return selected[:requested_total]


def _oracle_segments_to_dense_labels(
    *,
    sample: BenchmarkSample,
    segments: Sequence[OracleSegment],
) -> Optional[List[str]]:
    n = len(sample.text)
    if n <= 0:
        return []
    if not segments:
        return None
    labels = ["other"] * n
    cursor = 0
    for segment in segments:
        start = max(0, min(n, int(segment.start)))
        end = max(start, min(n, int(segment.end)))
        if end <= start:
            return None
        if int(start) != int(cursor):
            return None
        label_source = segment.raw_label if segment.raw_label is not None else segment.label
        label = _canonical_label(str(label_source))
        for idx in range(start, end):
            labels[idx] = label
        cursor = int(end)
    if int(cursor) != int(n):
        return None
    return labels


def _score_subset(
    *,
    samples: Sequence[BenchmarkSample],
    predicted_labels_by_id: Mapping[str, Sequence[str]],
    boundary_tolerance: int,
    include_ids: Optional[set[str]] = None,
) -> Dict[str, object]:
    accumulator = ScoreAccumulator()
    missing_predictions = 0
    for sample in samples:
        if include_ids is not None and sample.snippet_id not in include_ids:
            continue
        pred = predicted_labels_by_id.get(sample.snippet_id)
        if pred is None:
            missing_predictions += 1
            continue
        pred_list = [_canonical_label(str(label)) for label in pred]
        truth_list = [_canonical_label(str(label)) for label in sample.truth_labels]
        if len(pred_list) != len(truth_list):
            missing_predictions += 1
            continue
        accumulator.add(
            truth_labels=truth_list,
            pred_labels=pred_list,
            tolerance=max(0, int(boundary_tolerance)),
        )
    metrics = accumulator.finalize()
    metrics["missing_predictions"] = int(missing_predictions)
    return metrics


def _run_oracle_for_batch_size(
    *,
    batch_size: int,
    snippets: Sequence[BoundarySnippet],
    samples_by_id: Mapping[str, BenchmarkSample],
    oracle_mode: str,
    model: str,
    api_key: Optional[str],
    proxy: Optional[str],
    rate_limit_sleep_seconds: float,
    rate_limit_max_retries: int,
    missing_snippet_retries: int,
    boundary_tolerance: int,
    progress: bool,
) -> Dict[str, object]:
    run_started = time.perf_counter()
    runtime_error: Optional[str] = None
    predictions: Dict[str, List[str]] = {}
    source_map: Dict[str, str] = {}
    batch_diagnostics: List[Dict[str, object]] = []
    failed_source_states = {
        "fallback_failed_segmentation",
        "fallback_oracle_error",
        "fallback_runtime_error",
    }

    if oracle_mode == "stub":
        oracle = StubOracle()
    else:
        oracle = GeminiBoundaryOracle(
            model=model,
            api_key=api_key,
            batch_size=int(batch_size),
            proxy=proxy,
            rate_limit_sleep_seconds=float(rate_limit_sleep_seconds),
            rate_limit_max_retries=int(rate_limit_max_retries),
            missing_snippet_retries=int(missing_snippet_retries),
            show_progress=bool(progress),
            progress_desc=f"oracle batches (size={int(batch_size)})",
            progress_leave=False,
        )

    try:
        refined = oracle.annotate(snippets)
        if oracle_mode == "stub":
            source_map = {snippet.snippet_id: "model" for snippet in snippets}
            requests = int(math.ceil(float(len(snippets)) / float(max(1, int(batch_size)))))
            batch_diagnostics = [
                {
                    "run_id": "stub",
                    "status": "al_oracle_ok",
                    "requested": int(len(snippets)),
                    "parsed": int(len(snippets)),
                    "model_output_count": int(len(snippets)),
                    "rate_limit_retries": 0,
                    "initial_missing_snippets": 0,
                    "initial_parse_failed_snippets": 0,
                    "recovered_missing_snippets": 0,
                    "recovered_parse_failed_snippets": 0,
                    "missing_retry_requests": 0,
                    "parse_failed_retry_requests": 0,
                    "final_parse_failed_snippets": 0,
                    "final_missing_snippets": 0,
                    "failed_segmentation_snippets": 0,
                    "skipped_segmentation_snippets": 0,
                    "parse_failed_snippet_ids": [],
                    "missing_snippet_ids": [],
                    "parse_failed_reasons": {},
                    "error": "",
                    "requests": requests,
                }
            ]
        else:
            source_map_obj = getattr(oracle, "last_snippet_sources", {})
            if isinstance(source_map_obj, dict):
                source_map = {str(k): str(v) for k, v in source_map_obj.items()}
            batches_obj = getattr(oracle, "last_batches", [])
            if isinstance(batches_obj, list):
                batch_diagnostics = [dict(item) for item in batches_obj if isinstance(item, dict)]
        for snippet in snippets:
            sid = str(snippet.snippet_id)
            source_map.setdefault(sid, "unknown")
            source_state = str(source_map.get(sid, "unknown"))
            if source_state != "model":
                continue
            sample = samples_by_id.get(sid)
            if sample is None:
                continue
            segs = refined.get(sid, [])
            dense_labels = _oracle_segments_to_dense_labels(
                sample=sample,
                segments=segs,
            )
            if dense_labels is None:
                source_map[sid] = "fallback_failed_segmentation"
                continue
            predictions[sid] = dense_labels
    except Exception as exc:
        runtime_error = str(exc)
        for snippet in snippets:
            source_map[snippet.snippet_id] = "fallback_runtime_error"

    for snippet in snippets:
        source_map.setdefault(snippet.snippet_id, "unknown")

    elapsed_seconds = time.perf_counter() - run_started
    model_ids = {sid for sid, state in source_map.items() if state == "model"}
    mixed_ids = {sample.snippet_id for sample in samples_by_id.values() if sample.mixed_truth}
    non_mixed_ids = {sample.snippet_id for sample in samples_by_id.values() if not sample.mixed_truth}

    scores_all = _score_subset(
        samples=list(samples_by_id.values()),
        predicted_labels_by_id=predictions,
        boundary_tolerance=int(boundary_tolerance),
        include_ids=None,
    )
    scores_model_only = _score_subset(
        samples=list(samples_by_id.values()),
        predicted_labels_by_id=predictions,
        boundary_tolerance=int(boundary_tolerance),
        include_ids=model_ids,
    )
    scores_mixed = _score_subset(
        samples=list(samples_by_id.values()),
        predicted_labels_by_id=predictions,
        boundary_tolerance=int(boundary_tolerance),
        include_ids=mixed_ids,
    )
    scores_non_mixed = _score_subset(
        samples=list(samples_by_id.values()),
        predicted_labels_by_id=predictions,
        boundary_tolerance=int(boundary_tolerance),
        include_ids=non_mixed_ids,
    )

    status_counts = Counter(source_map.values())
    failed_batches = sum(
        1 for row in batch_diagnostics if str(row.get("status", "")).endswith("failed")
    )
    requested = int(sum(int(row.get("requested", 0)) for row in batch_diagnostics))
    parsed = int(sum(int(row.get("parsed", 0)) for row in batch_diagnostics))
    model_outputs = int(sum(int(row.get("model_output_count", 0)) for row in batch_diagnostics))
    if requested <= 0:
        requested = len(snippets)
    if parsed <= 0:
        parsed = len(predictions)
    if model_outputs <= 0 and "model" in status_counts:
        model_outputs = int(status_counts["model"])
    failed_segmentations = int(
        sum(1 for state in source_map.values() if str(state) in failed_source_states)
    )
    skipped_segmentations = int(
        sum(1 for state in source_map.values() if str(state) != "model")
    )
    scored_snippets = int(len(predictions))

    run_summary = {
        "batch_size": int(batch_size),
        "elapsed_seconds": float(elapsed_seconds),
        "snippets_per_second": _safe_ratio(len(snippets), elapsed_seconds),
        "runtime_error": runtime_error,
        "requested_snippets": int(requested),
        "parsed_snippets": int(parsed),
        "model_output_snippets": int(model_outputs),
        "fallback_snippets": int(max(0, len(snippets) - model_outputs)),
        "failed_segmentations": int(failed_segmentations),
        "skipped_segmentations": int(skipped_segmentations),
        "scored_snippets": int(scored_snippets),
        "model_output_rate": _safe_ratio(model_outputs, len(snippets)),
        "fallback_rate": _safe_ratio(max(0, len(snippets) - model_outputs), len(snippets)),
        "failed_batches": int(failed_batches),
        "batch_diagnostics": batch_diagnostics,
        "source_state_counts": dict(status_counts),
        "snippet_sources": dict(source_map),
        "scores": {
            "all": scores_all,
            "model_only": scores_model_only,
            "mixed": scores_mixed,
            "non_mixed": scores_non_mixed,
        },
        "predictions": predictions,
    }
    return run_summary


def _recommend_max_batch_size(
    *,
    runs: Sequence[Mapping[str, object]],
    max_char_accuracy_drop: float,
    max_boundary_f1_drop: float,
    max_fallback_rate: float,
    max_failed_batches: int,
    min_model_output_rate: float,
) -> Dict[str, object]:
    successful: List[Mapping[str, object]] = []
    for run in runs:
        err = run.get("runtime_error")
        if err is None:
            successful.append(run)
            continue
        if not str(err).strip():
            successful.append(run)
    if not successful:
        return {
            "recommended_batch_size": None,
            "reason": "No successful runs.",
            "best_char_accuracy": None,
            "best_boundary_f1_tolerant": None,
            "eligibility": [],
        }
    best_char = max(
        float(run.get("scores", {}).get("all", {}).get("char_accuracy", 0.0))
        for run in successful
    )
    best_boundary = max(
        float(run.get("scores", {}).get("all", {}).get("boundary_f1_tolerant", 0.0))
        for run in successful
    )
    eligibility: List[Dict[str, object]] = []
    eligible_runs: List[Mapping[str, object]] = []
    for run in successful:
        batch_size = int(run.get("batch_size", 0))
        reasons: List[str] = []
        fallback_rate = float(run.get("fallback_rate", 0.0))
        failed_batches = int(run.get("failed_batches", 0))
        model_output_rate = float(run.get("model_output_rate", 0.0))
        char_accuracy = float(run.get("scores", {}).get("all", {}).get("char_accuracy", 0.0))
        boundary_f1 = float(run.get("scores", {}).get("all", {}).get("boundary_f1_tolerant", 0.0))
        if failed_batches > int(max_failed_batches):
            reasons.append(
                f"failed_batches={failed_batches} > allowed={int(max_failed_batches)}"
            )
        if fallback_rate > float(max_fallback_rate):
            reasons.append(
                f"fallback_rate={fallback_rate:.4f} > allowed={float(max_fallback_rate):.4f}"
            )
        if model_output_rate < float(min_model_output_rate):
            reasons.append(
                f"model_output_rate={model_output_rate:.4f} < required={float(min_model_output_rate):.4f}"
            )
        if char_accuracy < (best_char - float(max_char_accuracy_drop)):
            reasons.append(
                "char_accuracy_drop="
                f"{(best_char - char_accuracy):.4f} > allowed={float(max_char_accuracy_drop):.4f}"
            )
        if boundary_f1 < (best_boundary - float(max_boundary_f1_drop)):
            reasons.append(
                "boundary_f1_drop="
                f"{(best_boundary - boundary_f1):.4f} > allowed={float(max_boundary_f1_drop):.4f}"
            )
        is_eligible = not reasons
        if is_eligible:
            eligible_runs.append(run)
        eligibility.append(
            {
                "batch_size": int(batch_size),
                "eligible": bool(is_eligible),
                "reasons": reasons,
            }
        )

    if not eligible_runs:
        return {
            "recommended_batch_size": None,
            "reason": "No batch size met reliability + quality thresholds.",
            "best_char_accuracy": float(best_char),
            "best_boundary_f1_tolerant": float(best_boundary),
            "eligibility": eligibility,
        }
    recommended = max(eligible_runs, key=lambda run: int(run.get("batch_size", 0)))
    return {
        "recommended_batch_size": int(recommended.get("batch_size", 0)),
        "reason": "Highest batch size meeting reliability + quality thresholds.",
        "best_char_accuracy": float(best_char),
        "best_boundary_f1_tolerant": float(best_boundary),
        "eligibility": eligibility,
    }


def _build_markdown_report(
    *,
    args: argparse.Namespace,
    samples: Sequence[BenchmarkSample],
    prior_scores: Mapping[str, object],
    runs: Sequence[Mapping[str, object]],
    recommendation: Mapping[str, object],
) -> str:
    lines: List[str] = []
    lines.append("# Active-Learning Oracle Batch Benchmark")
    lines.append("")
    lines.append(f"- Generated at: `{_now_utc_iso()}`")
    lines.append(f"- Model: `{args.model}`")
    lines.append(f"- Oracle backend: `{args.oracle}`")
    lines.append(f"- Benchmark dataset: `{args.benchmark_dataset}`")
    lines.append(f"- Ignore benchmark dataset: `{bool(args.ignore_benchmark_dataset)}`")
    lines.append(f"- Snippets: `{len(samples)}`")
    lines.append(f"- Snippet length: `{args.sample_length}`")
    lines.append(f"- Batch sizes: `{','.join(str(x) for x in _parse_batch_sizes(args.batch_sizes))}`")
    lines.append(f"- Boundary tolerance: `±{args.boundary_tolerance}` chars")
    lines.append("")

    task_counts = Counter(sample.task for sample in samples)
    mixed_count = sum(1 for sample in samples if sample.mixed_truth)
    non_mixed_count = len(samples) - mixed_count
    unique_truth_labels = sorted({label for sample in samples for label in sample.truth_labels})
    lines.append("## Benchmark Composition")
    lines.append("")
    lines.append(f"- Mixed samples: `{mixed_count}`")
    lines.append(f"- Non-mixed samples: `{non_mixed_count}`")
    lines.append(f"- Unique truth labels: `{len(unique_truth_labels)}`")
    lines.append(f"- Label set: `{', '.join(unique_truth_labels)}`")
    lines.append("")
    lines.append("| task | samples |")
    lines.append("|---|---:|")
    for task, count in sorted(task_counts.items()):
        lines.append(f"| `{task}` | {count} |")
    lines.append("")

    lines.append("## Prior Baseline (Predicted Segments in Prompt)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| char_accuracy | {float(prior_scores.get('char_accuracy', 0.0)):.4f} |")
    lines.append(f"| exact_match_rate | {float(prior_scores.get('exact_match_rate', 0.0)):.4f} |")
    lines.append(
        f"| boundary_f1_tolerant | {float(prior_scores.get('boundary_f1_tolerant', 0.0)):.4f} |"
    )
    lines.append("")

    lines.append("## Batch-Size Results")
    lines.append("")
    lines.append(
        "| batch | failed_batches | failed_seg | skipped_seg | scored | fallback_rate | "
        "char_acc | Δchar vs prior | boundary_f1_tol | exact_match | snippets/s |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    prior_char = float(prior_scores.get("char_accuracy", 0.0))
    for run in sorted(runs, key=lambda row: int(row.get("batch_size", 0))):
        all_scores = run.get("scores", {}).get("all", {})
        char_acc = float(all_scores.get("char_accuracy", 0.0))
        lines.append(
            f"| {int(run.get('batch_size', 0))} | "
            f"{int(run.get('failed_batches', 0))} | "
            f"{int(run.get('failed_segmentations', 0))} | "
            f"{int(run.get('skipped_segmentations', 0))} | "
            f"{int(run.get('scored_snippets', 0))} | "
            f"{float(run.get('fallback_rate', 0.0)):.4f} | "
            f"{char_acc:.4f} | "
            f"{(char_acc - prior_char):+.4f} | "
            f"{float(all_scores.get('boundary_f1_tolerant', 0.0)):.4f} | "
            f"{float(all_scores.get('exact_match_rate', 0.0)):.4f} | "
            f"{float(run.get('snippets_per_second', 0.0)):.3f} |"
        )
    lines.append("")

    lines.append("## Recommended Max Batch Size")
    lines.append("")
    rec = recommendation.get("recommended_batch_size")
    if rec is None:
        lines.append(f"- Recommendation: none (`{recommendation.get('reason', 'n/a')}`)")
    else:
        lines.append(
            f"- Recommendation: `{int(rec)}` ({recommendation.get('reason', 'n/a')})"
        )
    lines.append(
        f"- Best char accuracy observed: `{float(recommendation.get('best_char_accuracy') or 0.0):.4f}`"
    )
    lines.append(
        f"- Best tolerant boundary F1 observed: "
        f"`{float(recommendation.get('best_boundary_f1_tolerant') or 0.0):.4f}`"
    )
    lines.append("")

    lines.append("### Eligibility By Batch Size")
    lines.append("")
    lines.append("| batch | eligible | reasons |")
    lines.append("|---:|:---:|---|")
    for row in recommendation.get("eligibility", []):
        reasons = "; ".join(str(item) for item in row.get("reasons", [])) or "ok"
        lines.append(
            f"| {int(row.get('batch_size', 0))} | "
            f"{'yes' if bool(row.get('eligible')) else 'no'} | "
            f"{reasons} |"
        )
    lines.append("")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score Gemini active-learning boundary refinement quality across different batch sizes "
            "using a diverse 256-char benchmark. By default it uses a fixed curated dataset."
        )
    )
    parser.add_argument(
        "--benchmark-dataset",
        type=str,
        default=str(DEFAULT_CURATED_BENCHMARK_DATASET),
        help=(
            "Path to a curated benchmark JSON file built by "
            "active_learning.build_curated_benchmark_set. "
            "This is required; fallback sampling from evaluation/data is disabled."
        ),
    )
    parser.add_argument(
        "--ignore-benchmark-dataset",
        action="store_true",
        help="Deprecated/unsupported. This script now requires --benchmark-dataset and fails fast otherwise.",
    )
    parser.add_argument(
        "--data-roots",
        type=str,
        default="evaluation/data",
        help=(
            "Comma-separated dataset roots containing tasks like pure_fragments/sequence_pair. "
            "Unused by default; kept for backward compatibility only."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated subset of benchmark tasks.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Total benchmark snippets to use. 0 means use the full curated benchmark pool.",
    )
    parser.add_argument(
        "--pure-samples",
        type=int,
        default=-1,
        help=(
            "Number of non-mixed snippets in the selected subset. "
            "-1 auto-selects based on pool composition "
            "(and uses full pure count when --samples=0)."
        ),
    )
    parser.add_argument("--sample-length", type=int, default=256, help="Snippet length in characters.")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(value) for value in DEFAULT_BATCH_SIZES),
        help="Comma-separated oracle batch sizes to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for benchmark sampling.")
    parser.add_argument(
        "--boundary-center-jitter",
        type=int,
        default=48,
        help="How far mixed windows may shift around the selected boundary.",
    )
    parser.add_argument(
        "--min-boundary-margin",
        type=int,
        default=12,
        help="Minimum distance from snippet edges for selected mixed boundaries.",
    )
    parser.add_argument(
        "--max-prior-boundary-shift",
        type=int,
        default=6,
        help="Synthetic shift applied to prior boundaries before oracle refinement.",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=int,
        default=2,
        help="Tolerance (in chars) for tolerant boundary precision/recall/F1.",
    )

    parser.add_argument("--oracle", choices=("gemini", "stub"), default="gemini")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--proxy", type=str, default=None)
    parser.add_argument("--rate-limit-sleep-seconds", type=float, default=65.0)
    parser.add_argument("--rate-limit-max-retries", type=int, default=8)
    parser.add_argument("--missing-snippet-retries", type=int, default=2)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars for batch-size and oracle-batch processing.",
    )
    parser.add_argument(
        "--run-live",
        action="store_true",
        help=(
            "Run oracle calls for each batch size. Without this flag the script only builds "
            "the benchmark and reports the no-cost prior baseline."
        ),
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only build benchmark + baseline scores; skip oracle calls.",
    )

    parser.add_argument("--max-char-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--max-boundary-f1-drop", type=float, default=0.03)
    parser.add_argument("--max-fallback-rate", type=float, default=0.0)
    parser.add_argument("--max-failed-batches", type=int, default=0)
    parser.add_argument("--min-model-output-rate", type=float, default=1.0)

    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output JSON path (default: active_learning/benchmark_results/<timestamp>.json).",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default="",
        help="Optional output Markdown path (default: active_learning/benchmark_results/<timestamp>.md).",
    )
    parser.add_argument(
        "--save-benchmark-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store sampled benchmark snippets in output JSON.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    tasks = _parse_csv_str(args.tasks)
    if not tasks:
        raise ValueError("No --tasks were provided.")
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    if not batch_sizes:
        raise ValueError("No valid --batch-sizes were provided.")
    if args.sample_length <= 16:
        raise ValueError("--sample-length must be > 16.")
    if args.samples < 0:
        raise ValueError("--samples must be >= 0 (0 means full curated benchmark).")
    if args.pure_samples < -1:
        raise ValueError("--pure-samples must be >= -1 (-1 means auto).")
    if bool(args.ignore_benchmark_dataset):
        raise ValueError(
            "--ignore-benchmark-dataset is no longer supported. "
            "This benchmark now requires a curated dataset file and fails fast otherwise."
        )

    rng = np.random.default_rng(int(args.seed))
    benchmark_source = "curated_dataset"
    roots: List[Path] = []
    if not str(args.benchmark_dataset).strip():
        raise ValueError("--benchmark-dataset must be provided and non-empty.")
    dataset_path = Path(args.benchmark_dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Curated benchmark dataset not found. "
            "Fallback to evaluation/data is disabled. "
            f"Expected file: {dataset_path}"
        )

    print(f"Loading curated benchmark dataset: {dataset_path}", flush=True)
    pool = _load_curated_benchmark_pool(dataset_path)
    pool = [sample for sample in pool if len(sample.text) == int(args.sample_length)]
    if not pool:
        raise RuntimeError(
            f"Curated benchmark dataset has no samples with length={int(args.sample_length)}."
        )
    pool_total = int(len(pool))
    pool_pure = int(sum(1 for sample in pool if not sample.mixed_truth))
    requested_samples = int(args.samples)
    requested_pure_samples = int(args.pure_samples)

    if requested_samples == 0:
        selected_samples = pool_total
    else:
        selected_samples = requested_samples
        if selected_samples > pool_total:
            raise ValueError(
                f"--samples={selected_samples} exceeds curated pool size={pool_total} "
                f"for length={int(args.sample_length)}."
            )

    if requested_pure_samples == -1:
        if requested_samples == 0:
            selected_pure_samples = pool_pure
        else:
            pure_ratio = float(pool_pure) / float(pool_total) if pool_total > 0 else 0.0
            selected_pure_samples = int(round(float(selected_samples) * pure_ratio))
            selected_pure_samples = max(0, min(selected_samples, selected_pure_samples))
    else:
        selected_pure_samples = requested_pure_samples

    if selected_pure_samples > selected_samples:
        raise ValueError(
            f"--pure-samples={selected_pure_samples} must be <= selected samples {selected_samples}."
        )

    print(
        f"Selecting benchmark subset from curated pool "
        f"(pool={pool_total}, total={selected_samples}, pure={selected_pure_samples}, "
        f"mixed={selected_samples - selected_pure_samples}) ...",
        flush=True,
    )
    samples = _select_curated_samples(
        pool=pool,
        total_samples=int(selected_samples),
        pure_samples=int(selected_pure_samples),
        rng=rng,
    )

    mixed_count = sum(1 for sample in samples if sample.mixed_truth)
    non_mixed_count = len(samples) - mixed_count
    unique_labels = sorted({label for sample in samples for label in sample.truth_labels})
    print(
        "Benchmark ready: "
        f"samples={len(samples)}, mixed={mixed_count}, non_mixed={non_mixed_count}, "
        f"unique_labels={len(unique_labels)}",
        flush=True,
    )

    snippets = [sample.to_boundary_snippet() for sample in samples]
    samples_by_id = {sample.snippet_id: sample for sample in samples}
    prior_predictions = {sample.snippet_id: list(sample.predicted_labels) for sample in samples}
    prior_all = _score_subset(
        samples=samples,
        predicted_labels_by_id=prior_predictions,
        boundary_tolerance=int(args.boundary_tolerance),
        include_ids=None,
    )
    prior_mixed = _score_subset(
        samples=samples,
        predicted_labels_by_id=prior_predictions,
        boundary_tolerance=int(args.boundary_tolerance),
        include_ids={sample.snippet_id for sample in samples if sample.mixed_truth},
    )
    prior_non_mixed = _score_subset(
        samples=samples,
        predicted_labels_by_id=prior_predictions,
        boundary_tolerance=int(args.boundary_tolerance),
        include_ids={sample.snippet_id for sample in samples if not sample.mixed_truth},
    )
    prior_scores = {
        "all": prior_all,
        "mixed": prior_mixed,
        "non_mixed": prior_non_mixed,
    }

    runs: List[Dict[str, object]] = []
    should_run_live = bool(args.run_live) and not bool(args.skip_run)
    if args.skip_run:
        print("Skipping oracle runs (--skip-run).", flush=True)
    elif not should_run_live:
        print(
            "Skipping oracle runs: pass --run-live to execute batch-size calls.",
            flush=True,
        )
    else:
        batch_iter: Iterable[int] = list(batch_sizes)
        progress_bar = None
        if bool(args.progress) and _tqdm is not None and len(batch_sizes) > 1:
            progress_bar = _tqdm(
                batch_sizes,
                total=len(batch_sizes),
                desc="batch sizes",
                unit="size",
                dynamic_ncols=True,
                leave=False,
            )
            batch_iter = progress_bar
        for batch_size in batch_iter:
            print(f"\n=== Benchmark batch_size={batch_size} ===", flush=True)
            run_summary = _run_oracle_for_batch_size(
                batch_size=int(batch_size),
                snippets=snippets,
                samples_by_id=samples_by_id,
                oracle_mode=str(args.oracle),
                model=str(args.model),
                api_key=args.api_key,
                proxy=args.proxy,
                rate_limit_sleep_seconds=float(args.rate_limit_sleep_seconds),
                rate_limit_max_retries=int(args.rate_limit_max_retries),
                missing_snippet_retries=int(args.missing_snippet_retries),
                boundary_tolerance=int(args.boundary_tolerance),
                progress=bool(args.progress),
            )
            all_scores = run_summary["scores"]["all"]
            prior_char = float(prior_all.get("char_accuracy", 0.0))
            char_acc = float(all_scores.get("char_accuracy", 0.0))
            failed_seg = int(run_summary.get("failed_segmentations", 0))
            skipped_seg = int(run_summary.get("skipped_segmentations", 0))
            scored_seg = int(run_summary.get("scored_snippets", 0))
            print(
                "Result: "
                f"char_acc={char_acc:.4f} (delta_vs_prior={char_acc - prior_char:+.4f}), "
                f"boundary_f1_tol={float(all_scores.get('boundary_f1_tolerant', 0.0)):.4f}, "
                f"exact_match={float(all_scores.get('exact_match_rate', 0.0)):.4f}, "
                f"failed_batches={int(run_summary.get('failed_batches', 0))}, "
                f"failed_segmentations={failed_seg}, "
                f"skipped_segmentations={skipped_seg}, "
                f"scored_snippets={scored_seg}, "
                f"fallback_rate={float(run_summary.get('fallback_rate', 0.0)):.4f}, "
                f"snippets_per_second={float(run_summary.get('snippets_per_second', 0.0)):.3f}",
                flush=True,
            )
            runs.append(run_summary)
            if progress_bar is not None:
                progress_bar.set_postfix(
                    failed=int(failed_seg),
                    skipped=int(skipped_seg),
                    scored=int(scored_seg),
                    refresh=False,
                )
        if progress_bar is not None:
            progress_bar.close()

    recommendation = _recommend_max_batch_size(
        runs=runs,
        max_char_accuracy_drop=float(args.max_char_accuracy_drop),
        max_boundary_f1_drop=float(args.max_boundary_f1_drop),
        max_fallback_rate=float(args.max_fallback_rate),
        max_failed_batches=int(args.max_failed_batches),
        min_model_output_rate=float(args.min_model_output_rate),
    )

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    default_stem = f"al_oracle_batch_benchmark_{timestamp}"
    output_json = Path(args.output_json).resolve() if args.output_json else (output_dir / f"{default_stem}.json")
    output_markdown = (
        Path(args.output_markdown).resolve()
        if args.output_markdown
        else (output_dir / f"{default_stem}.md")
    )

    result_payload: Dict[str, object] = {
        "created_at": _now_utc_iso(),
        "config": {
            "benchmark_source": str(benchmark_source),
            "benchmark_dataset": str(dataset_path) if dataset_path else "",
            "ignore_benchmark_dataset": bool(args.ignore_benchmark_dataset),
            "data_roots": [str(root) for root in roots],
            "tasks": list(tasks),
            "samples": int(selected_samples),
            "pure_samples": int(selected_pure_samples),
            "samples_requested": int(requested_samples),
            "pure_samples_requested": int(requested_pure_samples),
            "sample_length": int(args.sample_length),
            "batch_sizes": list(batch_sizes),
            "seed": int(args.seed),
            "oracle": str(args.oracle),
            "model": str(args.model),
            "run_live_requested": bool(args.run_live),
            "skip_run_requested": bool(args.skip_run),
            "ran_live_calls": bool(should_run_live),
            "boundary_tolerance": int(args.boundary_tolerance),
            "max_char_accuracy_drop": float(args.max_char_accuracy_drop),
            "max_boundary_f1_drop": float(args.max_boundary_f1_drop),
            "max_fallback_rate": float(args.max_fallback_rate),
            "max_failed_batches": int(args.max_failed_batches),
            "min_model_output_rate": float(args.min_model_output_rate),
        },
        "benchmark": {
            "samples_total": int(len(samples)),
            "mixed_samples": int(mixed_count),
            "non_mixed_samples": int(non_mixed_count),
            "unique_truth_labels": list(unique_labels),
            "task_counts": dict(Counter(sample.task for sample in samples)),
        },
        "prior_scores": prior_scores,
        "runs": [],
        "recommendation": recommendation,
    }
    if args.save_benchmark_snapshot:
        result_payload["samples"] = [sample.to_json_dict() for sample in samples]
    for run in runs:
        run_copy = dict(run)
        run_copy.pop("predictions", None)
        run_copy["deltas_vs_prior"] = {
            "char_accuracy": float(run_copy.get("scores", {}).get("all", {}).get("char_accuracy", 0.0))
            - float(prior_all.get("char_accuracy", 0.0)),
            "boundary_f1_tolerant": float(
                run_copy.get("scores", {}).get("all", {}).get("boundary_f1_tolerant", 0.0)
            )
            - float(prior_all.get("boundary_f1_tolerant", 0.0)),
        }
        result_payload["runs"].append(run_copy)

    markdown = _build_markdown_report(
        args=args,
        samples=samples,
        prior_scores=prior_all,
        runs=result_payload["runs"],
        recommendation=recommendation,
    )

    output_json.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_markdown.write_text(markdown, encoding="utf-8")

    rec_batch = recommendation.get("recommended_batch_size")
    if rec_batch is None:
        print(
            "\nRecommendation: none "
            f"({recommendation.get('reason', 'no reason')}).",
            flush=True,
        )
    else:
        print(
            f"\nRecommendation: max batch size {int(rec_batch)} "
            f"({recommendation.get('reason', 'n/a')}).",
            flush=True,
        )
    print(f"Wrote JSON report: {output_json}", flush=True)
    print(f"Wrote Markdown report: {output_markdown}", flush=True)


if __name__ == "__main__":
    main()
