#!/usr/bin/env python3
"""
Interactive monitor-set error viewer.

Runs a specified checkpoint over a large subset of monitor_preprocessed_b and
serves a browser UI for inspecting mislabelings side-by-side:

  - choose a focal content type / label
  - sort windows by focal-label F1, precision, recall, diff ratio, etc.
  - inspect ground truth vs prediction with mismatch highlighting
  - hover any rendered byte to see the model's top probabilities
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg  # noqa: E402
from utils.metrics_helper import _valid_metric_mask  # noqa: E402
from utils.monitor_eval import load_monitor_memmaps  # noqa: E402
from viewers.core import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    Predictor,
    _apply_label_mapping,
    _infer_checkpoint_architecture,
    _load_checkpoint_hparams,
    auto_color,
)


DEFAULT_MONITOR_ROOT = (REPO_ROOT / "downloader" / "monitor_preprocessed_b").resolve()
STATIC_ROOT = Path(__file__).resolve().parent / "monitor_error_static"
VISIBLE_ASCII_BYTES = tuple(range(0x20, 0x7F))
VISIBLE_BYTE_SET = set(VISIBLE_ASCII_BYTES)
SORT_OPTIONS: Dict[str, Tuple[str, bool]] = {
    "f1_asc": ("focus_f1", False),
    "f1_desc": ("focus_f1", True),
    "precision_asc": ("focus_precision", False),
    "precision_desc": ("focus_precision", True),
    "recall_asc": ("focus_recall", False),
    "recall_desc": ("focus_recall", True),
    "diff_desc": ("diff_ratio", True),
    "diff_asc": ("diff_ratio", False),
    "accuracy_asc": ("overall_accuracy", False),
    "accuracy_desc": ("overall_accuracy", True),
    "support_desc": ("focus_support", True),
    "support_asc": ("focus_support", False),
}
SORT_OPTION_LABELS: Dict[str, str] = {
    "f1_asc": "F1: worst first",
    "f1_desc": "F1: best first",
    "precision_asc": "Precision: lowest first",
    "precision_desc": "Precision: highest first",
    "recall_asc": "Recall: lowest first",
    "recall_desc": "Recall: highest first",
    "diff_desc": "Diff ratio: highest first",
    "diff_asc": "Diff ratio: lowest first",
    "accuracy_asc": "Accuracy: lowest first",
    "accuracy_desc": "Accuracy: highest first",
    "support_desc": "Support: largest first",
    "support_asc": "Support: smallest first",
}


@dataclass(frozen=True)
class FocusCandidate:
    file_idx: int
    file_type_id: int
    byte_len: int
    focus_label_id: int
    focus_total_bytes: int
    focus_largest_start: int
    focus_largest_end: int


@dataclass
class MonitorWindowRecord:
    sample_id: int
    file_idx: int
    file_type_id: int
    file_type: str
    byte_len: int
    focus_label_id: int
    focus_label: str
    focus_total_bytes: int
    focus_support: int
    window_start: int
    window_end: int
    overall_accuracy: float
    diff_ratio: float
    focus_precision: float
    focus_recall: float
    focus_f1: float
    tp: int
    fp: int
    fn: int
    preview: str
    window_bytes: bytes
    truth_labels: np.ndarray


def _label_name(label_id: int) -> str:
    lid = int(label_id)
    other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
    if other_idx is not None and lid == int(other_idx):
        return "other"
    if lid == int(cfg.PAD_ID):
        return "pad"
    return str(cfg.ID2LANG.get(lid, f"id_{lid}"))


def _palette_color(index: int, total: int) -> str:
    return auto_color(int(index), max(1, int(total)))


def _global_label_palette() -> Dict[str, str]:
    total = max(1, len(cfg.LANG2ID))
    mapping: Dict[str, str] = {}
    for label, idx in sorted(cfg.LANG2ID.items(), key=lambda kv: kv[1]):
        mapping[str(label)] = _palette_color(int(idx), total)
    mapping["other"] = "#9aa4b2"
    mapping["pad"] = "#e5e7eb"
    mapping["unlabeled"] = "#cbd5e1"
    return mapping


def _bytes_to_display_text(byte_values: Sequence[int]) -> str:
    out_chars: List[str] = []
    for raw in byte_values:
        val = int(raw)
        if val == 0x0D:
            out_chars.append("\n")
            continue
        if val in VISIBLE_BYTE_SET or val in (0x09, 0x0A):
            out_chars.append(chr(val))
        else:
            out_chars.append("?")
    return "".join(out_chars)


def _preview_text(text: str, limit: int = 140) -> str:
    compact = " ".join(text.replace("\n", " ").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)] + "..."


def _centered_window_start(byte_len: int, focus_start: int, focus_end: int, window_len: int) -> int:
    if byte_len <= window_len:
        return 0
    midpoint = (int(focus_start) + int(focus_end)) // 2
    start = midpoint - window_len // 2
    start = max(0, min(start, byte_len - window_len))
    return int(start)


def _build_window(
    row: np.void,
    segments: np.ndarray,
    contents: np.memmap,
    *,
    focus_start: int,
    focus_end: int,
    window_len: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    byte_len = int(row["byte_len"])
    byte_start = int(row["byte_start"])
    win_start = _centered_window_start(byte_len, focus_start, focus_end, window_len)
    win_end = min(byte_len, win_start + window_len)
    full = np.asarray(contents[byte_start : byte_start + byte_len], dtype=np.uint8)
    x = np.full((window_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    used = max(0, int(win_end) - int(win_start))
    if used > 0:
        x[:used] = full[win_start:win_end].astype(np.int32)

    other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
    y_fill = int(other_idx) if other_idx is not None else int(cfg.PAD_ID)
    y = np.full((window_len,), y_fill, dtype=np.int32)
    seg_slice = segments[int(row["seg_start"]) : int(row["seg_start"]) + int(row["seg_count"])]
    for seg in seg_slice:
        seg_start = int(seg["start"])
        seg_end = int(seg["end"])
        label = int(seg["label"])
        overlap_s = max(seg_start, win_start)
        overlap_e = min(seg_end, win_start + window_len)
        if overlap_e <= overlap_s:
            continue
        y[overlap_s - win_start : overlap_e - win_start] = label
    return x, y, int(win_start), int(win_end)


def _focus_metrics(
    tokens: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    focus_label_id: int,
) -> Dict[str, float | int]:
    mask = _valid_metric_mask(truth.reshape(1, -1), tokens.reshape(1, -1))[0]
    valid = int(mask.sum())
    if valid <= 0:
        return {
            "overall_accuracy": 0.0,
            "diff_ratio": 0.0,
            "focus_support": 0,
            "focus_precision": 0.0,
            "focus_recall": 0.0,
            "focus_f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    valid_truth = truth[mask]
    valid_pred = pred[mask]
    overall_accuracy = float(np.mean(valid_truth == valid_pred))
    diff_ratio = float(np.mean(valid_truth != valid_pred))

    focus_truth = valid_truth == int(focus_label_id)
    focus_pred = valid_pred == int(focus_label_id)
    tp = int(np.sum(focus_truth & focus_pred))
    fp = int(np.sum((~focus_truth) & focus_pred))
    fn = int(np.sum(focus_truth & (~focus_pred)))
    support = int(np.sum(focus_truth))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    denom = 2 * tp + fp + fn
    f1 = float((2 * tp) / denom) if denom > 0 else 0.0
    return {
        "overall_accuracy": overall_accuracy,
        "diff_ratio": diff_ratio,
        "focus_support": support,
        "focus_precision": precision,
        "focus_recall": recall,
        "focus_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _compress_label_runs(label_ids: Sequence[int]) -> List[Dict[str, Any]]:
    if not label_ids:
        return []
    out: List[Dict[str, Any]] = []
    start = 0
    current = int(label_ids[0])
    for idx in range(1, len(label_ids)):
        lid = int(label_ids[idx])
        if lid == current:
            continue
        out.append({"start": int(start), "end": int(idx), "label": _label_name(current)})
        start = idx
        current = lid
    out.append({"start": int(start), "end": int(len(label_ids)), "label": _label_name(current)})
    return out


def _build_palette_for_detail(
    truth_ids: Sequence[int],
    pred_ids: Sequence[int],
    base_palette: Dict[str, str],
) -> Tuple[Dict[str, str], List[str]]:
    order: List[str] = []
    seen = set()
    for lid in list(truth_ids) + list(pred_ids):
        name = _label_name(int(lid))
        if name in {"pad", "unlabeled"}:
            continue
        if name in seen:
            continue
        seen.add(name)
        order.append(name)
    palette = dict(base_palette)
    for idx, name in enumerate(order):
        palette.setdefault(name, _palette_color(idx, max(1, len(order))))
    if "other" not in palette:
        palette["other"] = "#9aa4b2"
    return palette, order


def _top_prob_entries(
    prob_row: np.ndarray,
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    if prob_row.size <= 0:
        return []
    pairs: List[Tuple[str, float]] = []
    for idx, prob in enumerate(prob_row.tolist()):
        value = float(prob)
        if value <= 0.0:
            continue
        pairs.append((_label_name(int(idx)), value))
    pairs.sort(key=lambda item: item[1], reverse=True)
    out: List[Dict[str, Any]] = []
    running = 0.0
    for pos, (label, prob) in enumerate(pairs):
        if pos < limit:
            out.append({"label": label, "prob": float(prob)})
            running += float(prob)
    remainder = float(max(0.0, 1.0 - running))
    if remainder > 1e-6 and len(pairs) > limit:
        out.append({"label": "others", "prob": remainder})
    return out


def _detail_confusions(
    truth: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    *,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for t, p, use in zip(truth.tolist(), pred.tolist(), mask.tolist()):
        if not bool(use):
            continue
        if int(t) == int(p):
            continue
        counts[(_label_name(int(t)), _label_name(int(p)))] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [
        {"truth": truth_name, "pred": pred_name, "count": int(count)}
        for (truth_name, pred_name), count in ordered[:limit]
    ]


class MonitorErrorStore:
    def __init__(
        self,
        *,
        predictor: Predictor,
        monitor_root: Path,
        per_label_limit: int,
        min_label_bytes: int,
        other_threshold: float,
    ) -> None:
        self.predictor = predictor
        self.monitor_root = Path(monitor_root).resolve()
        self.per_label_limit = max(1, int(per_label_limit))
        self.min_label_bytes = max(1, int(min_label_bytes))
        self.other_threshold = float(other_threshold)
        self.palette = _global_label_palette()
        self.monitor_data = load_monitor_memmaps(self.monitor_root)
        self.records: List[MonitorWindowRecord] = []
        self.by_id: Dict[int, MonitorWindowRecord] = {}
        self.label_stats: List[Dict[str, Any]] = []
        self._build_index()

    def _collect_candidates(self) -> List[FocusCandidate]:
        files = self.monitor_data["files"]
        segments = self.monitor_data["segments"]
        grouped: Dict[int, List[FocusCandidate]] = defaultdict(list)
        for file_idx, row in enumerate(files):
            seg_slice = segments[int(row["seg_start"]) : int(row["seg_start"]) + int(row["seg_count"])]
            totals: Dict[int, int] = defaultdict(int)
            largest: Dict[int, Tuple[int, int]] = {}
            for seg in seg_slice:
                label = int(seg["label"])
                if label < 0 or label == int(cfg.PAD_ID):
                    continue
                start = int(seg["start"])
                end = int(seg["end"])
                if end <= start:
                    continue
                if label >= int(cfg.NUM_CLASSES):
                    continue
                totals[label] += end - start
                current = largest.get(label)
                if current is None or (end - start) > (current[1] - current[0]):
                    largest[label] = (start, end)
            for label, total_bytes in totals.items():
                if int(total_bytes) < self.min_label_bytes:
                    continue
                largest_span = largest.get(label)
                if largest_span is None:
                    continue
                grouped[label].append(
                    FocusCandidate(
                        file_idx=int(file_idx),
                        file_type_id=int(row["type_id"]),
                        byte_len=int(row["byte_len"]),
                        focus_label_id=int(label),
                        focus_total_bytes=int(total_bytes),
                        focus_largest_start=int(largest_span[0]),
                        focus_largest_end=int(largest_span[1]),
                    )
                )

        selected: List[FocusCandidate] = []
        for label_id, candidates in grouped.items():
            candidates.sort(
                key=lambda item: (
                    -int(item.focus_total_bytes),
                    -(int(item.focus_largest_end) - int(item.focus_largest_start)),
                    int(item.file_idx),
                )
            )
            selected.extend(candidates[: self.per_label_limit])
            print(
                f"[monitor_error_viewer] selected {min(len(candidates), self.per_label_limit)} "
                f"windows for focus label {_label_name(label_id)}",
                flush=True,
            )
        selected.sort(key=lambda item: (item.focus_label_id, -item.focus_total_bytes, item.file_idx))
        return selected

    def _predict_labels(
        self,
        token_windows: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        byte_arrays = [
            np.asarray(arr[arr != cfg.PAD_BYTE_ID], dtype=np.uint8)
            for arr in token_windows
        ]
        pred_by_text, prob_by_text, _, _ = self.predictor._segment_bytes_batch(byte_arrays)
        out_labels: List[np.ndarray] = []
        out_probs: List[np.ndarray] = []
        other_idx = int(getattr(cfg, "OTHER_CLASS_INDEX", cfg.NUM_CLASSES))
        for probs in prob_by_text:
            pred = np.argmax(probs, axis=-1).astype(np.int32)
            if self.other_threshold > 0.0 and probs.size > 0:
                pred = Predictor.threshold_predictions(
                    probs,
                    other_threshold=self.other_threshold,
                    other_id=other_idx,
                ).astype(np.int32)
            out_labels.append(pred)
            out_probs.append(np.asarray(probs, dtype=np.float32))
        return out_labels, out_probs

    def _build_index(self) -> None:
        files = self.monitor_data["files"]
        segments = self.monitor_data["segments"]
        contents = self.monitor_data["contents"]
        candidates = self._collect_candidates()
        if not candidates:
            raise RuntimeError(
                f"No monitor windows matched min_label_bytes={self.min_label_bytes} in {self.monitor_root}"
            )

        next_id = 1
        outer_batch = 256
        for offset in range(0, len(candidates), outer_batch):
            chunk = candidates[offset : offset + outer_batch]
            token_windows: List[np.ndarray] = []
            truth_windows: List[np.ndarray] = []
            metas: List[Tuple[FocusCandidate, int, int]] = []
            for cand in chunk:
                row = files[int(cand.file_idx)]
                x, y, win_start, win_end = _build_window(
                    row,
                    segments,
                    contents,
                    focus_start=int(cand.focus_largest_start),
                    focus_end=int(cand.focus_largest_end),
                    window_len=int(self.predictor.chunk),
                )
                token_windows.append(x)
                truth_windows.append(y.astype(np.int32))
                metas.append((cand, win_start, win_end))

            pred_windows, _ = self._predict_labels(token_windows)
            for idx, (cand, win_start, win_end) in enumerate(metas):
                tokens = token_windows[idx]
                truth = truth_windows[idx]
                pred = pred_windows[idx]
                used = min(len(pred), int((tokens != cfg.PAD_BYTE_ID).sum()))
                if used <= 0:
                    continue
                truth_used = truth[:used].astype(np.int32, copy=False)
                pred_used = pred[:used].astype(np.int32, copy=False)
                token_used = tokens[:used].astype(np.int32, copy=False)
                metrics = _focus_metrics(
                    token_used,
                    truth_used,
                    pred_used,
                    focus_label_id=int(cand.focus_label_id),
                )
                display_text = _bytes_to_display_text(token_used.tolist())
                record = MonitorWindowRecord(
                    sample_id=int(next_id),
                    file_idx=int(cand.file_idx),
                    file_type_id=int(cand.file_type_id),
                    file_type=_label_name(int(cand.file_type_id)),
                    byte_len=int(cand.byte_len),
                    focus_label_id=int(cand.focus_label_id),
                    focus_label=_label_name(int(cand.focus_label_id)),
                    focus_total_bytes=int(cand.focus_total_bytes),
                    focus_support=int(metrics["focus_support"]),
                    window_start=int(win_start),
                    window_end=int(win_end),
                    overall_accuracy=float(metrics["overall_accuracy"]),
                    diff_ratio=float(metrics["diff_ratio"]),
                    focus_precision=float(metrics["focus_precision"]),
                    focus_recall=float(metrics["focus_recall"]),
                    focus_f1=float(metrics["focus_f1"]),
                    tp=int(metrics["tp"]),
                    fp=int(metrics["fp"]),
                    fn=int(metrics["fn"]),
                    preview=_preview_text(display_text),
                    window_bytes=bytes(token_used.astype(np.uint8).tolist()),
                    truth_labels=truth_used.copy(),
                )
                self.records.append(record)
                self.by_id[record.sample_id] = record
                next_id += 1
            print(
                f"[monitor_error_viewer] indexed {min(offset + outer_batch, len(candidates))}/{len(candidates)} windows",
                flush=True,
            )

        by_label: Dict[str, List[MonitorWindowRecord]] = defaultdict(list)
        for record in self.records:
            by_label[record.focus_label].append(record)
        stats: List[Dict[str, Any]] = []
        for label, items in by_label.items():
            stats.append(
                {
                    "label": label,
                    "color": self.palette.get(label, "#9aa4b2"),
                    "count": len(items),
                    "avg_f1": float(np.mean([item.focus_f1 for item in items])),
                    "avg_precision": float(np.mean([item.focus_precision for item in items])),
                    "avg_recall": float(np.mean([item.focus_recall for item in items])),
                    "avg_diff_ratio": float(np.mean([item.diff_ratio for item in items])),
                }
            )
        stats.sort(key=lambda row: (row["avg_f1"], -row["count"], row["label"]))
        self.label_stats = stats

    def config_payload(self) -> Dict[str, Any]:
        return {
            "checkpoint_chunk": int(self.predictor.chunk),
            "monitor_root": str(self.monitor_root),
            "total_samples": len(self.records),
            "per_label_limit": int(self.per_label_limit),
            "min_label_bytes": int(self.min_label_bytes),
            "other_threshold": float(self.other_threshold),
            "label_stats": list(self.label_stats),
            "sort_options": [
                {"value": key, "label": SORT_OPTION_LABELS.get(key, key.replace("_", " "))}
                for key in SORT_OPTIONS.keys()
            ],
            "focus_labels": [row["label"] for row in self.label_stats],
            "palette": dict(self.palette),
        }

    def _filtered_records(
        self,
        *,
        focus_label: str,
        search: str,
        mistakes_only: bool,
    ) -> List[MonitorWindowRecord]:
        focus = str(focus_label or "").strip().lower()
        search_text = str(search or "").strip().lower()
        out: List[MonitorWindowRecord] = []
        for record in self.records:
            if focus and record.focus_label.lower() != focus:
                continue
            if mistakes_only and record.diff_ratio <= 0.0:
                continue
            if search_text:
                hay = " ".join(
                    [
                        record.focus_label,
                        record.file_type,
                        str(record.file_idx),
                        record.preview,
                    ]
                ).lower()
                if search_text not in hay:
                    continue
            out.append(record)
        return out

    def list_samples(
        self,
        *,
        page: int,
        page_size: int,
        focus_label: str,
        sort_key: str,
        search: str,
        mistakes_only: bool,
    ) -> Dict[str, Any]:
        items = self._filtered_records(
            focus_label=focus_label,
            search=search,
            mistakes_only=mistakes_only,
        )
        field_name, descending = SORT_OPTIONS.get(sort_key, SORT_OPTIONS["f1_asc"])
        items.sort(
            key=lambda row: (
                float(getattr(row, field_name)),
                -int(row.focus_support),
                int(row.sample_id),
            ),
            reverse=bool(descending),
        )
        total = len(items)
        total_pages = max(1, int(math.ceil(total / max(1, page_size))))
        page = max(1, min(int(page), total_pages))
        start = (page - 1) * page_size
        page_items = items[start : start + page_size]

        avg_f1 = float(np.mean([row.focus_f1 for row in items])) if items else 0.0
        avg_diff = float(np.mean([row.diff_ratio for row in items])) if items else 0.0
        avg_acc = float(np.mean([row.overall_accuracy for row in items])) if items else 0.0

        return {
            "page": int(page),
            "page_size": int(page_size),
            "total": int(total),
            "total_pages": int(total_pages),
            "filtered_stats": {
                "avg_focus_f1": avg_f1,
                "avg_diff_ratio": avg_diff,
                "avg_overall_accuracy": avg_acc,
            },
            "items": [
                {
                    "id": int(row.sample_id),
                    "file_idx": int(row.file_idx),
                    "file_type": row.file_type,
                    "focus_label": row.focus_label,
                    "focus_support": int(row.focus_support),
                    "focus_total_bytes": int(row.focus_total_bytes),
                    "focus_f1": float(row.focus_f1),
                    "focus_precision": float(row.focus_precision),
                    "focus_recall": float(row.focus_recall),
                    "overall_accuracy": float(row.overall_accuracy),
                    "diff_ratio": float(row.diff_ratio),
                    "preview": row.preview,
                }
                for row in page_items
            ],
        }

    def sample_detail(self, sample_id: int) -> Dict[str, Any]:
        record = self.by_id.get(int(sample_id))
        if record is None:
            raise KeyError(f"Unknown sample id {sample_id}")

        raw_bytes = np.frombuffer(record.window_bytes, dtype=np.uint8)
        pred_windows, prob_windows = self._predict_labels([raw_bytes.astype(np.int32)])
        pred = pred_windows[0].astype(np.int32, copy=False)
        probs = prob_windows[0]
        truth = np.asarray(record.truth_labels, dtype=np.int32)
        used = min(len(raw_bytes), len(pred), len(truth))
        raw_bytes = raw_bytes[:used]
        pred = pred[:used]
        truth = truth[:used]
        probs = probs[:used]

        mask = _valid_metric_mask(truth.reshape(1, -1), raw_bytes.reshape(1, -1))[0]
        display_text = _bytes_to_display_text(raw_bytes.tolist())
        palette, order = _build_palette_for_detail(truth.tolist(), pred.tolist(), self.palette)

        char_data: List[Dict[str, Any]] = []
        for idx, ch in enumerate(display_text):
            truth_name = _label_name(int(truth[idx]))
            pred_name = _label_name(int(pred[idx]))
            char_data.append(
                {
                    "char": ch,
                    "truth": truth_name,
                    "pred": pred_name,
                    "match": bool(int(truth[idx]) == int(pred[idx])),
                    "valid": bool(mask[idx]),
                    "probs": _top_prob_entries(probs[idx]),
                }
            )

        return {
            "id": int(record.sample_id),
            "file_idx": int(record.file_idx),
            "file_type": record.file_type,
            "byte_len": int(record.byte_len),
            "window_start": int(record.window_start),
            "window_end": int(record.window_end),
            "focus_label": record.focus_label,
            "focus_support": int(record.focus_support),
            "focus_total_bytes": int(record.focus_total_bytes),
            "metrics": {
                "focus_f1": float(record.focus_f1),
                "focus_precision": float(record.focus_precision),
                "focus_recall": float(record.focus_recall),
                "overall_accuracy": float(record.overall_accuracy),
                "diff_ratio": float(record.diff_ratio),
                "tp": int(record.tp),
                "fp": int(record.fp),
                "fn": int(record.fn),
                "valid_bytes": int(mask.sum()),
            },
            "display_text": display_text,
            "char_data": char_data,
            "label_colors": palette,
            "label_order": order,
            "truth_runs": _compress_label_runs(truth.tolist()),
            "pred_runs": _compress_label_runs(pred.tolist()),
            "top_confusions": _detail_confusions(truth, pred, mask),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor-B error viewer")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.msgpack or Orbax dir).")
    parser.add_argument("--monitor-root", type=str, default=str(DEFAULT_MONITOR_ROOT))
    parser.add_argument("--device", type=str, default="auto", help="Inference device (auto/cpu/gpu/cuda).")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--min-run", type=int, default=1, help="Prediction smoothing run length.")
    parser.add_argument("--other-threshold", type=float, default=0.0, help="Optional confidence threshold for virtual 'other'.")
    parser.add_argument("--per-label-limit", type=int, default=128, help="Max windows to index per focal label.")
    parser.add_argument("--min-label-bytes", type=int, default=64, help="Minimum bytes of the focal label within a file to index it.")
    parser.add_argument("--inference-batch-size", type=int, default=16, help="Batch size for model window inference.")
    parser.add_argument(
        "--inference-backend",
        type=str,
        default="auto",
        choices=("auto", "fast", "legacy"),
        help="Inference backend for viewer predictions. 'auto' prefers the shared fast backend.",
    )
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float32", "float16"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8067)
    parser.add_argument("--openapi", action="store_true", help="Expose OpenAPI docs.")
    return parser


def _resolve_predictor(args: argparse.Namespace) -> Predictor:
    ckpt_path = Path(args.checkpoint).resolve()
    auto_hparams = _load_checkpoint_hparams(ckpt_path)
    label_names = auto_hparams.get("label_names")
    if label_names:
        _apply_label_mapping(label_names)  # type: ignore[arg-type]

    inferred = _infer_checkpoint_architecture(ckpt_path)
    model_dim = args.model_dim if args.model_dim is not None else int(auto_hparams.get("model_dim", inferred.get("model_dim", 256)))
    dtype = args.dtype or str(auto_hparams.get("dtype", inferred.get("dtype", "bfloat16"))).rsplit(".", 1)[-1]
    arch = str(auto_hparams.get("arch", inferred.get("arch", "unet1d"))).strip().lower()
    predictor = Predictor(
        str(ckpt_path),
        num_classes=int(cfg.NUM_CLASSES),
        model_dim=int(model_dim),
        channels=tuple(int(x) for x in auto_hparams.get("channels", inferred.get("channels", ()) or (96, 128, 192, 256))),
        arch=arch,
        mamba_layers=int(auto_hparams.get("mamba_layers", inferred.get("mamba_layers", 6))),
        mamba_d_state=int(auto_hparams.get("mamba_d_state", inferred.get("mamba_d_state", 8))),
        mamba_expand=int(auto_hparams.get("mamba_expand", inferred.get("mamba_expand", 1))),
        mamba_dt_rank=int(auto_hparams.get("mamba_dt_rank", inferred.get("mamba_dt_rank", 16))),
        mamba_conv=int(auto_hparams.get("mamba_conv", inferred.get("mamba_conv", 4))),
        mamba_bidirectional=bool(auto_hparams.get("mamba_bidirectional", inferred.get("mamba_bidirectional", True))),
        dtype_str=str(dtype),
        chunk=int(args.chunk),
        other_threshold=float(args.other_threshold),
        inference_batch_size=int(args.inference_batch_size),
        device=args.device,
        inference_backend=args.inference_backend,
    )
    return predictor


def create_app(args: argparse.Namespace) -> FastAPI:
    load_error: Optional[str] = None
    store: Optional[MonitorErrorStore] = None
    try:
        predictor = _resolve_predictor(args)
        store = MonitorErrorStore(
            predictor=predictor,
            monitor_root=Path(args.monitor_root),
            per_label_limit=int(args.per_label_limit),
            min_label_bytes=int(args.min_label_bytes),
            other_threshold=float(args.other_threshold),
        )
    except Exception as exc:
        load_error = str(exc)

    app = FastAPI(title="Monitor Error Viewer", docs_url="/docs" if args.openapi else None)
    if STATIC_ROOT.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        if not STATIC_ROOT.is_dir():
            return HTMLResponse("<h1>UI not found</h1>", status_code=500)
        return FileResponse(str(STATIC_ROOT / "index.html"))

    @app.get("/api/config")
    def api_config() -> Dict[str, Any]:
        if load_error:
            raise HTTPException(status_code=500, detail=load_error)
        assert store is not None
        return store.config_payload()

    @app.get("/api/samples")
    def api_samples(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
        focus_label: str = Query(default=""),
        sort: str = Query(default="f1_asc"),
        search: str = Query(default=""),
        mistakes_only: bool = Query(default=False),
    ) -> Dict[str, Any]:
        if load_error:
            raise HTTPException(status_code=500, detail=load_error)
        assert store is not None
        return store.list_samples(
            page=int(page),
            page_size=int(page_size),
            focus_label=str(focus_label),
            sort_key=str(sort),
            search=str(search),
            mistakes_only=bool(mistakes_only),
        )

    @app.get("/api/sample/{sample_id}")
    def api_sample(sample_id: int) -> Dict[str, Any]:
        if load_error:
            raise HTTPException(status_code=500, detail=load_error)
        assert store is not None
        try:
            return store.sample_detail(int(sample_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    return app


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    app = create_app(args)
    import uvicorn

    uvicorn.run(app, host=args.host, port=int(args.port), reload=False)


if __name__ == "__main__":
    main()
