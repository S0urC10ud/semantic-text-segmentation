from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateSpan:
    """A high-uncertainty boundary context selected for oracle labeling."""

    start: int
    end: int
    boundary: int
    score: float
    entropy_mean: float
    flip_rate: float
    left_label: int
    right_label: int


def normalized_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-position entropy normalized to [0, 1]."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected probs shape [T, C>=2], got {arr.shape}")
    arr = np.clip(arr, eps, 1.0)
    arr /= arr.sum(axis=-1, keepdims=True)
    raw = -np.sum(arr * np.log(arr), axis=-1)
    return (raw / np.log(arr.shape[1])).astype(np.float32)


def _flip_rate_around(labels: np.ndarray, center: int, radius: int) -> float:
    """Fraction of adjacent positions with class flips in a local neighborhood."""
    n = int(labels.shape[0])
    left = max(0, int(center) - int(radius))
    right = min(n, int(center) + int(radius) + 1)
    if right - left <= 1:
        return 0.0
    local = labels[left:right]
    flips = np.count_nonzero(local[1:] != local[:-1])
    return float(flips) / float(max(1, local.shape[0] - 1))


def _non_overlapping_topk(
    candidates: Sequence[CandidateSpan],
    *,
    top_k: int,
    min_gap: int,
) -> List[CandidateSpan]:
    if not candidates:
        return []
    selected: List[CandidateSpan] = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        overlap = False
        for keep in selected:
            if cand.end + min_gap <= keep.start:
                continue
            if keep.end + min_gap <= cand.start:
                continue
            overlap = True
            break
        if overlap:
            continue
        selected.append(cand)
        if len(selected) >= top_k:
            break
    return sorted(selected, key=lambda c: c.start)


def select_candidate_spans(
    probs: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    context_chars: int = 250,
    boundary_radius: int = 12,
    entropy_weight: float = 1.0,
    flip_weight: float = 0.6,
    min_score: float = 0.2,
    top_k: int = 16,
    min_gap_chars: int = 24,
) -> List[CandidateSpan]:
    """
    Select boundary-centric candidate spans from model probabilities.

    Score per boundary = entropy_weight * mean_entropy(local) +
                         flip_weight * flip_rate(local).
    """
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []

    pred = np.asarray(labels, dtype=np.int32) if labels is not None else np.argmax(arr, axis=-1).astype(np.int32)
    if pred.shape[0] != arr.shape[0]:
        raise ValueError(
            f"labels length {pred.shape[0]} does not match probs length {arr.shape[0]}"
        )
    if pred.shape[0] < 2:
        return []

    ent = normalized_entropy(arr)
    boundary_positions = np.flatnonzero(pred[1:] != pred[:-1]) + 1
    if boundary_positions.size == 0:
        return []

    candidates: List[CandidateSpan] = []
    n = int(pred.shape[0])
    for boundary in boundary_positions.tolist():
        local_left = max(0, boundary - int(boundary_radius))
        local_right = min(n, boundary + int(boundary_radius) + 1)
        ent_mean = float(np.mean(ent[local_left:local_right])) if local_right > local_left else 0.0
        flip = _flip_rate_around(pred, boundary, int(boundary_radius))
        score = float(entropy_weight) * ent_mean + float(flip_weight) * flip
        if score < float(min_score):
            continue

        left_label = int(pred[max(0, boundary - 1)])
        right_label = int(pred[min(n - 1, boundary)])
        start = max(0, int(boundary) - int(context_chars))
        end = min(n, int(boundary) + int(context_chars))
        if end <= start:
            continue

        candidates.append(
            CandidateSpan(
                start=start,
                end=end,
                boundary=int(boundary),
                score=score,
                entropy_mean=ent_mean,
                flip_rate=flip,
                left_label=left_label,
                right_label=right_label,
            )
        )

    return _non_overlapping_topk(
        candidates,
        top_k=max(1, int(top_k)),
        min_gap=max(0, int(min_gap_chars)),
    )

