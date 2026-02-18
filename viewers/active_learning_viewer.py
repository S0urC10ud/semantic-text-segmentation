#!/usr/bin/env python3
"""
Active-learning refinement viewer.

Explore refinement samples stored in active_learning/label_store.sqlite:
  - summary stats (status, rounds, label distributions)
  - coarse refined-vs-predicted confusion
  - per-sample inference coverage and query-trigger highlighting
"""

from __future__ import annotations

import argparse
import colorsys
import html
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from active_learning.oracle import OracleSegment, canonicalize_label, merge_adjacent_segments  # noqa: E402


DEFAULT_STORE_PATH = (REPO_ROOT / "active_learning" / "label_store.sqlite").resolve()
TOP_LABELS_MATRIX = 14


def _auto_color(label: str) -> str:
    base = sum(ord(c) for c in label) + len(label) * 131
    hue = (base % 360) / 360.0
    sat = 0.63
    lig = 0.53
    r, g, b = colorsys.hls_to_rgb(hue, lig, sat)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _palette_color(index: int, total: int) -> str:
    # Use an evenly spaced hue wheel for labels visible in one sample so
    # neighboring classes are easy to distinguish.
    n = max(1, int(total))
    i = max(0, int(index))
    hue = ((float(i) / float(n)) + 0.17) % 1.0
    sat = 0.76
    lig = 0.50 if (i % 2 == 0) else 0.40
    r, g, b = colorsys.hls_to_rgb(hue, lig, sat)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(160,160,160,{alpha})"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _color_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for label in cfg.LANG2ID.keys():
        idx = int(cfg.LANG2ID[label])
        mapping[label] = _palette_color(idx, len(cfg.LANG2ID))
    mapping["other"] = mapping.get("other", "#9a6bff")
    mapping["unlabeled"] = "#9aa4b2"
    return mapping


def _sample_color_map(
    segment_groups: Sequence[Sequence[OracleSegment]],
    fallback_colors: Dict[str, str],
) -> Dict[str, str]:
    ordered_labels: List[str] = []
    seen: set[str] = set()
    for group in segment_groups:
        for seg in sorted(group, key=lambda s: (int(s.start), int(s.end))):
            label = canonicalize_label(seg.label)
            if label in seen:
                continue
            seen.add(label)
            ordered_labels.append(label)

    if not ordered_labels:
        return dict(fallback_colors)

    mapping = dict(fallback_colors)
    total = len(ordered_labels)
    for idx, label in enumerate(ordered_labels):
        mapping[label] = _palette_color(idx, total)
    mapping["unlabeled"] = "#9aa4b2"
    return mapping


def _parse_segments(raw_json: str, text_len: int) -> List[OracleSegment]:
    try:
        payload = json.loads(raw_json) if raw_json else []
    except json.JSONDecodeError:
        payload = []
    if not isinstance(payload, list):
        payload = []
    parsed: List[OracleSegment] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start", 0))
            end = int(item.get("end", 0))
            label = canonicalize_label(str(item.get("label", "other")))
        except Exception:
            continue
        start = max(0, min(text_len, start))
        end = max(start, min(text_len, end))
        if end <= start:
            continue
        parsed.append(OracleSegment(start, end, label))
    parsed = merge_adjacent_segments(parsed)
    if not parsed:
        return [OracleSegment(0, max(0, text_len), "unlabeled")] if text_len > 0 else []

    filled: List[OracleSegment] = []
    cursor = 0
    for seg in parsed:
        if seg.start > cursor:
            filled.append(OracleSegment(cursor, seg.start, "unlabeled"))
        filled.append(seg)
        cursor = seg.end
    if cursor < text_len:
        filled.append(OracleSegment(cursor, text_len, "unlabeled"))
    return merge_adjacent_segments(filled)


def _labels_from_segments(text_len: int, segments: Sequence[OracleSegment]) -> List[str]:
    labels = ["unlabeled"] * max(0, text_len)
    for seg in segments:
        for idx in range(max(0, seg.start), min(text_len, seg.end)):
            labels[idx] = seg.label
    return labels


def _segments_to_html(
    text: str,
    segments: Sequence[OracleSegment],
    colors: Dict[str, str],
    *,
    diff_mask: Optional[Sequence[bool]] = None,
) -> str:
    if not text:
        return "<em>Empty snippet.</em>"
    chunks: List[str] = []
    n = len(text)

    def _is_diff(idx: int) -> bool:
        if diff_mask is None:
            return False
        if idx < 0 or idx >= len(diff_mask):
            return False
        return bool(diff_mask[idx])

    def _emit_span(start_idx: int, end_idx: int, label: str, color: str, *, diff: bool) -> None:
        if end_idx <= start_idx:
            return
        span_text = html.escape(text[start_idx:end_idx])
        class_name = "tok"
        if diff_mask is not None:
            class_name += " diff" if diff else " same"
        chunks.append(
            f'<span class="{class_name}" style="background:{_hex_to_rgba(color, 0.20)};" '
            f'title="{html.escape(label)}" '
            f'data-label="{html.escape(label)}">{span_text}</span>'
        )

    for seg in segments:
        start = max(0, min(n, int(seg.start)))
        end = max(start, min(n, int(seg.end)))
        if end <= start:
            continue
        label = canonicalize_label(seg.label)
        color = colors.get(label, _auto_color(label))
        if diff_mask is None:
            _emit_span(start, end, label, color, diff=False)
            continue

        run_start = start
        run_diff = _is_diff(start)
        for idx in range(start + 1, end):
            cur_diff = _is_diff(idx)
            if cur_diff == run_diff:
                continue
            _emit_span(run_start, idx, label, color, diff=run_diff)
            run_start = idx
            run_diff = cur_diff
        _emit_span(run_start, end, label, color, diff=run_diff)
    return "".join(chunks) if chunks else "<em>No renderable segments.</em>"


def _parse_trigger_ranges(raw_json: str, text_len: int) -> List[Dict[str, object]]:
    try:
        payload = json.loads(raw_json) if raw_json else []
    except json.JSONDecodeError:
        payload = []
    if not isinstance(payload, list):
        return []
    out: List[Dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start", 0))
            end = int(item.get("end", 0))
        except Exception:
            continue
        start = max(0, min(text_len, start))
        end = max(start, min(text_len, end))
        if end <= start:
            continue
        out.append(
            {
                "start": start,
                "end": end,
                "boundary": int(item.get("boundary", start)),
                "score": float(item.get("score", 0.0)),
                "entropy_mean": float(item.get("entropy_mean", 0.0)),
                "flip_rate": float(item.get("flip_rate", 0.0)),
                "left_label": str(item.get("left_label", "")),
                "right_label": str(item.get("right_label", "")),
            }
        )
    out.sort(key=lambda row: (int(row["start"]), int(row["end"])))
    return out


def _render_text_with_triggers(text: str, triggers: Sequence[Dict[str, object]]) -> str:
    if not text:
        return "<em>Empty sample.</em>"
    n = len(text)
    ranges = []
    for row in triggers:
        try:
            start = max(0, min(n, int(row.get("start", 0))))
            end = max(start, min(n, int(row.get("end", 0))))
        except Exception:
            continue
        if end <= start:
            continue
        ranges.append((start, end, row))
    if not ranges:
        return f'<span class="src-plain">{html.escape(text)}</span>'

    chunks: List[str] = []
    cursor = 0
    for start, end, row in ranges:
        if start > cursor:
            chunks.append(f'<span class="src-plain">{html.escape(text[cursor:start])}</span>')
        title = (
            f"score={float(row.get('score', 0.0)):.3f} | "
            f"entropy={float(row.get('entropy_mean', 0.0)):.3f} | "
            f"flip={float(row.get('flip_rate', 0.0)):.3f} | "
            f"left={str(row.get('left_label', ''))} | right={str(row.get('right_label', ''))}"
        )
        chunks.append(
            f'<span class="trigger-span" title="{html.escape(title)}">{html.escape(text[start:end])}</span>'
        )
        cursor = end
    if cursor < n:
        chunks.append(f'<span class="src-plain">{html.escape(text[cursor:])}</span>')
    return "".join(chunks)


@dataclass
class RefinementRow:
    row_id: int
    created_at: str
    round_id: str
    status: str
    source_split: str
    source_lang: str
    sample_index: int
    sample_hash: str
    boundary_index: int
    snippet_start: int
    snippet_end: int
    snippet_text: str
    oracle_name: str
    oracle_model: str
    oracle_run_id: str
    acquisition_score: float
    predicted_segments_json: str
    refined_segments_json: str
    metadata_json: str


@dataclass
class InferenceSampleRow:
    row_id: int
    created_at: str
    round_id: str
    source_split: str
    source_lang: str
    sample_index: int
    sample_hash: str
    sample_text: str
    char_count: int
    queried_for_oracle: int
    candidate_count: int
    trigger_ranges_json: str
    predicted_segments_json: str
    metadata_json: str


class ActiveLearningStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path.resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Active-learning store not found at '{self.db_path}'.")
        self.colors = _color_map()
        self.has_inference_samples_table = self._table_exists("inference_samples")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    def _table_exists(self, table_name: str) -> bool:
        with self._connect() as con:
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
                (str(table_name),),
            ).fetchone()
        return row is not None

    def _table_row_count(self, table_name: str) -> int:
        if not self._table_exists(table_name):
            return 0
        with self._connect() as con:
            row = con.execute(f"SELECT COUNT(*) AS c FROM {table_name}").fetchone()
        return int(row["c"]) if row is not None else 0

    def _has_inference_rows(self) -> bool:
        return self.has_inference_samples_table and self._table_row_count("inference_samples") > 0

    @staticmethod
    def _row_to_dataclass(row: sqlite3.Row) -> RefinementRow:
        return RefinementRow(
            row_id=int(row["id"]),
            created_at=str(row["created_at"]),
            round_id=str(row["round_id"]),
            status=str(row["status"]),
            source_split=str(row["source_split"]),
            source_lang=str(row["source_lang"]),
            sample_index=int(row["sample_index"]),
            sample_hash=str(row["sample_hash"]),
            boundary_index=int(row["boundary_index"]),
            snippet_start=int(row["snippet_start"]),
            snippet_end=int(row["snippet_end"]),
            snippet_text=str(row["snippet_text"]),
            oracle_name=str(row["oracle_name"]),
            oracle_model=str(row["oracle_model"]),
            oracle_run_id=str(row["oracle_run_id"]),
            acquisition_score=float(row["acquisition_score"]),
            predicted_segments_json=str(row["predicted_segments_json"]),
            refined_segments_json=str(row["refined_segments_json"]),
            metadata_json=str(row["metadata_json"]),
        )

    @staticmethod
    def _row_to_inference_dataclass(row: sqlite3.Row) -> InferenceSampleRow:
        keys = set(row.keys())
        return InferenceSampleRow(
            row_id=int(row["id"]),
            created_at=str(row["created_at"]),
            round_id=str(row["round_id"]),
            source_split=str(row["source_split"]),
            source_lang=str(row["source_lang"]),
            sample_index=int(row["sample_index"]),
            sample_hash=str(row["sample_hash"]),
            sample_text=str(row["sample_text"]),
            char_count=int(row["char_count"]),
            queried_for_oracle=int(row["queried_for_oracle"]),
            candidate_count=int(row["candidate_count"]),
            trigger_ranges_json=str(row["trigger_ranges_json"]),
            predicted_segments_json=str(row["predicted_segments_json"]) if "predicted_segments_json" in keys else "[]",
            metadata_json=str(row["metadata_json"]),
        )

    def _fetch_rows(
        self,
        *,
        status: Optional[str] = "ok",
        round_id: Optional[str] = None,
        limit: Optional[int] = None,
        order_desc: bool = True,
    ) -> List[RefinementRow]:
        query = "SELECT * FROM refinements"
        params: List[object] = []
        where: List[str] = []
        if status and status != "all":
            where.append("status = ?")
            params.append(status)
        if round_id:
            where.append("round_id = ?")
            params.append(round_id)
        if where:
            query += " WHERE " + " AND ".join(where)
        query += f" ORDER BY id {'DESC' if order_desc else 'ASC'}"
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [self._row_to_dataclass(row) for row in rows]

    def _fetch_inference_rows(
        self,
        *,
        round_id: Optional[str] = None,
        queried: str = "all",
        limit: Optional[int] = None,
        order_desc: bool = True,
    ) -> List[InferenceSampleRow]:
        if not self.has_inference_samples_table:
            return []
        query = "SELECT * FROM inference_samples"
        params: List[object] = []
        where: List[str] = []
        if round_id:
            where.append("round_id = ?")
            params.append(round_id)
        if where:
            query += " WHERE " + " AND ".join(where)
        query += f" ORDER BY id {'DESC' if order_desc else 'ASC'}"
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
            if round_id:
                refined_rows = con.execute(
                    """
                    SELECT DISTINCT round_id, sample_hash, sample_index
                    FROM refinements
                    WHERE round_id = ?
                    """,
                    (round_id,),
                ).fetchall()
            else:
                refined_rows = con.execute(
                    """
                    SELECT DISTINCT round_id, sample_hash, sample_index
                    FROM refinements
                    """
                ).fetchall()
        refined_keys = {
            (str(row["round_id"]), str(row["sample_hash"]), int(row["sample_index"]))
            for row in refined_rows
        }
        entries = [self._row_to_inference_dataclass(row) for row in rows]
        for entry in entries:
            entry.queried_for_oracle = 1 if (
                entry.round_id,
                entry.sample_hash,
                int(entry.sample_index),
            ) in refined_keys else 0

        q = str(queried or "all").lower().strip()
        if q in {"yes", "queried", "1", "true"}:
            entries = [row for row in entries if int(row.queried_for_oracle) == 1]
        elif q in {"no", "unqueried", "0", "false"}:
            entries = [row for row in entries if int(row.queried_for_oracle) == 0]
        return entries

    def list_rounds(self) -> List[Dict[str, object]]:
        table_name = "inference_samples" if self._has_inference_rows() else "refinements"
        with self._connect() as con:
            rows = con.execute(
                f"""
                SELECT round_id, COUNT(*) AS cnt
                FROM {table_name}
                GROUP BY round_id
                ORDER BY MAX(id) DESC
                """
            ).fetchall()
        return [{"round_id": str(row["round_id"]), "count": int(row["cnt"])} for row in rows]

    def summary(self, *, status: Optional[str], round_id: Optional[str], queried: str = "all") -> Dict[str, object]:
        inference_rows_all = self._fetch_inference_rows(round_id=round_id, queried="all", limit=None, order_desc=False)
        inference_rows_filtered = self._fetch_inference_rows(
            round_id=round_id,
            queried=queried,
            limit=None,
            order_desc=False,
        )
        inference_total = len(inference_rows_all)
        inference_queried = sum(1 for row in inference_rows_all if int(row.queried_for_oracle) == 1)
        inference_unqueried = inference_total - inference_queried
        inference_filtered_total = len(inference_rows_filtered)
        inference_avg_candidates = (
            float(sum(int(row.candidate_count) for row in inference_rows_filtered)) / float(inference_filtered_total)
            if inference_filtered_total > 0
            else 0.0
        )
        rows = self._fetch_rows(status=status, round_id=round_id, limit=None, order_desc=False)
        if not rows:
            return {
                "total_rows": 0,
                "store_path": str(self.db_path),
                "status_counts": {},
                "rounds": self.list_rounds(),
                "label_counts_predicted": [],
                "label_counts_refined": [],
                "diff_ratio_mean": 0.0,
                "diff_ratio_p95": 0.0,
                "confusion": {"labels": [], "matrix": []},
                "inference_total": int(inference_total),
                "inference_filtered_total": int(inference_filtered_total),
                "inference_queried": int(inference_queried),
                "inference_unqueried": int(inference_unqueried),
                "inference_query_rate": (
                    float(inference_queried) / float(inference_total) if inference_total > 0 else 0.0
                ),
                "inference_avg_candidates": float(inference_avg_candidates),
            }

        status_counts: Dict[str, int] = {}
        label_pred_counts: Dict[str, int] = {}
        label_ref_counts: Dict[str, int] = {}
        confusion_counts: Dict[Tuple[str, str], int] = {}
        diff_ratios: List[float] = []

        for row in rows:
            status_counts[row.status] = status_counts.get(row.status, 0) + 1
            text_len = len(row.snippet_text or "")
            pred_segments = _parse_segments(row.predicted_segments_json, text_len)
            ref_segments = _parse_segments(row.refined_segments_json, text_len)
            pred_labels = _labels_from_segments(text_len, pred_segments)
            ref_labels = _labels_from_segments(text_len, ref_segments)
            if text_len > 0:
                diff = sum(1 for i in range(text_len) if pred_labels[i] != ref_labels[i])
                diff_ratios.append(float(diff) / float(text_len))
            for lab in pred_labels:
                label_pred_counts[lab] = label_pred_counts.get(lab, 0) + 1
            for lab in ref_labels:
                label_ref_counts[lab] = label_ref_counts.get(lab, 0) + 1
            for i in range(min(len(pred_labels), len(ref_labels))):
                key = (ref_labels[i], pred_labels[i])
                confusion_counts[key] = confusion_counts.get(key, 0) + 1

        def _sorted_counts(raw: Dict[str, int]) -> List[Dict[str, object]]:
            return [
                {"label": lab, "count": int(cnt), "color": self.colors.get(lab, _auto_color(lab))}
                for lab, cnt in sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))
            ]

        top_labels = [item["label"] for item in _sorted_counts(label_ref_counts)[:TOP_LABELS_MATRIX]]
        if "unlabeled" in label_ref_counts and "unlabeled" not in top_labels:
            top_labels.append("unlabeled")
        matrix = []
        for true_lab in top_labels:
            row_vals = []
            for pred_lab in top_labels:
                row_vals.append(int(confusion_counts.get((true_lab, pred_lab), 0)))
            matrix.append(row_vals)

        diff_array = sorted(diff_ratios)
        p95 = diff_array[int(0.95 * (len(diff_array) - 1))] if diff_array else 0.0
        return {
            "total_rows": len(rows),
            "store_path": str(self.db_path),
            "status_counts": status_counts,
            "rounds": self.list_rounds(),
            "label_counts_predicted": _sorted_counts(label_pred_counts),
            "label_counts_refined": _sorted_counts(label_ref_counts),
            "diff_ratio_mean": float(sum(diff_ratios) / len(diff_ratios)) if diff_ratios else 0.0,
            "diff_ratio_p95": float(p95),
            "confusion": {"labels": top_labels, "matrix": matrix},
            "inference_total": int(inference_total),
            "inference_filtered_total": int(inference_filtered_total),
            "inference_queried": int(inference_queried),
            "inference_unqueried": int(inference_unqueried),
            "inference_query_rate": float(inference_queried) / float(inference_total) if inference_total > 0 else 0.0,
            "inference_avg_candidates": float(inference_avg_candidates),
        }

    def list_samples(
        self,
        *,
        page: int,
        page_size: int,
        status: Optional[str],
        round_id: Optional[str],
        search: Optional[str],
        queried: str = "all",
    ) -> Dict[str, object]:
        if self._has_inference_rows():
            rows = self._fetch_inference_rows(round_id=round_id, queried=queried, limit=None, order_desc=True)
            token = (search or "").strip().lower()
            if token:
                filtered: List[InferenceSampleRow] = []
                for row in rows:
                    hay = " ".join(
                        [
                            row.source_lang,
                            row.round_id,
                            row.sample_text[:400],
                        ]
                    ).lower()
                    if token in hay:
                        filtered.append(row)
                rows = filtered

            total = len(rows)
            p = max(1, int(page))
            ps = max(1, min(100, int(page_size)))
            start = (p - 1) * ps
            end = start + ps
            sliced = rows[start:end]

            items: List[Dict[str, object]] = []
            for row in sliced:
                trigger_ranges = _parse_trigger_ranges(row.trigger_ranges_json, int(row.char_count))
                items.append(
                    {
                        "id": row.row_id,
                        "created_at": row.created_at,
                        "round_id": row.round_id,
                        "source_split": row.source_split,
                        "source_lang": row.source_lang,
                        "sample_index": row.sample_index,
                        "char_count": int(row.char_count),
                        "queried_for_oracle": bool(row.queried_for_oracle),
                        "candidate_count": int(row.candidate_count),
                        "trigger_count": int(len(trigger_ranges)),
                        "preview": (row.sample_text or "")[:180].replace("\n", "⏎ "),
                    }
                )
            return {
                "items": items,
                "total": total,
                "page": p,
                "page_size": ps,
                "total_pages": max(1, (total + ps - 1) // ps),
            }

        rows = self._fetch_rows(status=status, round_id=round_id, limit=None, order_desc=True)
        token = (search or "").strip().lower()
        if token:
            filtered: List[RefinementRow] = []
            for row in rows:
                hay = " ".join(
                    [
                        row.source_lang,
                        row.round_id,
                        row.oracle_name,
                        row.oracle_model,
                        row.snippet_text[:400],
                    ]
                ).lower()
                if token in hay:
                    filtered.append(row)
            rows = filtered

        total = len(rows)
        p = max(1, int(page))
        ps = max(1, min(100, int(page_size)))
        start = (p - 1) * ps
        end = start + ps
        sliced = rows[start:end]

        items: List[Dict[str, object]] = []
        for row in sliced:
            text_len = len(row.snippet_text or "")
            pred_segments = _parse_segments(row.predicted_segments_json, text_len)
            ref_segments = _parse_segments(row.refined_segments_json, text_len)
            pred_labels = _labels_from_segments(text_len, pred_segments)
            ref_labels = _labels_from_segments(text_len, ref_segments)
            diff_chars = sum(1 for i in range(text_len) if pred_labels[i] != ref_labels[i]) if text_len > 0 else 0
            diff_ratio = (float(diff_chars) / float(text_len)) if text_len > 0 else 0.0
            items.append(
                {
                    "id": row.row_id,
                    "created_at": row.created_at,
                    "round_id": row.round_id,
                    "status": row.status,
                    "source_split": row.source_split,
                    "source_lang": row.source_lang,
                    "sample_index": row.sample_index,
                    "boundary_index": row.boundary_index,
                    "acquisition_score": row.acquisition_score,
                    "diff_chars": int(diff_chars),
                    "diff_ratio": float(diff_ratio),
                    "char_count": int(text_len),
                    "oracle_name": row.oracle_name,
                    "oracle_model": row.oracle_model,
                    "preview": (row.snippet_text or "")[:180].replace("\n", "⏎ "),
                }
            )
        return {
            "items": items,
            "total": total,
            "page": p,
            "page_size": ps,
            "total_pages": max(1, (total + ps - 1) // ps),
        }

    def sample_detail(self, row_id: int) -> Dict[str, object]:
        if self._has_inference_rows():
            with self._connect() as con:
                row = con.execute("SELECT * FROM inference_samples WHERE id = ?", (int(row_id),)).fetchone()
            if row is not None:
                entry = self._row_to_inference_dataclass(row)
                text = entry.sample_text or ""
                text_len = len(text)
                trigger_ranges = _parse_trigger_ranges(entry.trigger_ranges_json, text_len)
                source_html = _render_text_with_triggers(text, trigger_ranges)
                pred_segments = _parse_segments(entry.predicted_segments_json, text_len)
                sample_colors = _sample_color_map([pred_segments], self.colors)
                predicted_html = _segments_to_html(text, pred_segments, sample_colors, diff_mask=None)
                prediction_has_model_segments = False
                try:
                    payload = json.loads(entry.predicted_segments_json) if entry.predicted_segments_json else []
                    prediction_has_model_segments = isinstance(payload, list) and len(payload) > 0
                except json.JSONDecodeError:
                    prediction_has_model_segments = False
                try:
                    metadata = json.loads(entry.metadata_json) if entry.metadata_json else {}
                except json.JSONDecodeError:
                    metadata = {}

                linked_refinements: List[Dict[str, object]] = []
                refinement_pairs: List[Dict[str, object]] = []
                with self._connect() as con:
                    ref_rows = con.execute(
                        """
                        SELECT * FROM refinements
                        WHERE round_id = ? AND sample_hash = ? AND sample_index = ?
                        ORDER BY id ASC
                        """,
                        (entry.round_id, entry.sample_hash, int(entry.sample_index)),
                    ).fetchall()
                for ref_row in ref_rows:
                    ref = self._row_to_dataclass(ref_row)
                    snippet_text = ref.snippet_text or ""
                    snippet_len = len(snippet_text)
                    pred_segments = _parse_segments(ref.predicted_segments_json, snippet_len)
                    ref_segments = _parse_segments(ref.refined_segments_json, snippet_len)
                    pred_labels = _labels_from_segments(snippet_len, pred_segments)
                    ref_labels = _labels_from_segments(snippet_len, ref_segments)
                    diff_chars = (
                        sum(1 for idx in range(snippet_len) if pred_labels[idx] != ref_labels[idx])
                        if snippet_len > 0
                        else 0
                    )
                    sample_colors = _sample_color_map([pred_segments, ref_segments], self.colors)
                    linked_refinements.append(
                        {
                            "id": int(ref.row_id),
                            "boundary_index": int(ref.boundary_index),
                            "snippet_start": int(ref.snippet_start),
                            "snippet_end": int(ref.snippet_end),
                            "acquisition_score": float(ref.acquisition_score),
                            "diff_chars": int(diff_chars),
                            "diff_ratio": float(diff_chars) / float(snippet_len) if snippet_len > 0 else 0.0,
                            "preview": snippet_text[:180].replace("\n", "⏎ "),
                        }
                    )
                    refinement_pairs.append(
                        {
                            "id": int(ref.row_id),
                            "boundary_index": int(ref.boundary_index),
                            "snippet_start": int(ref.snippet_start),
                            "snippet_end": int(ref.snippet_end),
                            "acquisition_score": float(ref.acquisition_score),
                            "diff_chars": int(diff_chars),
                            "diff_ratio": float(diff_chars) / float(snippet_len) if snippet_len > 0 else 0.0,
                            "predicted_html": _segments_to_html(snippet_text, pred_segments, sample_colors, diff_mask=None),
                            "refined_html": _segments_to_html(snippet_text, ref_segments, sample_colors, diff_mask=None),
                        }
                    )

                return {
                    "id": entry.row_id,
                    "created_at": entry.created_at,
                    "round_id": entry.round_id,
                    "source_split": entry.source_split,
                    "source_lang": entry.source_lang,
                    "sample_index": entry.sample_index,
                    "sample_hash": entry.sample_hash,
                    "char_count": text_len,
                    "queried_for_oracle": bool(ref_rows),
                    "candidate_count": int(entry.candidate_count),
                    "trigger_ranges": trigger_ranges,
                    "trigger_count": int(len(trigger_ranges)),
                    "source_html": source_html,
                    "predicted_html": predicted_html,
                    "prediction_has_model_segments": bool(prediction_has_model_segments),
                    "linked_refinements": linked_refinements,
                    "refinement_pairs": refinement_pairs,
                    "metadata": metadata,
                }

        with self._connect() as con:
            row = con.execute("SELECT * FROM refinements WHERE id = ?", (int(row_id),)).fetchone()
        if row is None:
            raise KeyError(f"Unknown refinement row id={row_id}")
        entry = self._row_to_dataclass(row)
        text = entry.snippet_text or ""
        text_len = len(text)
        pred_segments = _parse_segments(entry.predicted_segments_json, text_len)
        ref_segments = _parse_segments(entry.refined_segments_json, text_len)
        pred_labels = _labels_from_segments(text_len, pred_segments)
        ref_labels = _labels_from_segments(text_len, ref_segments)
        diff_mask = [pred_labels[i] != ref_labels[i] for i in range(text_len)]
        diff_chars = int(sum(1 for v in diff_mask if v))
        diff_ratio = float(diff_chars) / float(text_len) if text_len > 0 else 0.0
        sample_colors = _sample_color_map([pred_segments, ref_segments], self.colors)

        try:
            metadata = json.loads(entry.metadata_json) if entry.metadata_json else {}
        except json.JSONDecodeError:
            metadata = {}

        return {
            "id": entry.row_id,
            "created_at": entry.created_at,
            "round_id": entry.round_id,
            "status": entry.status,
            "source_split": entry.source_split,
            "source_lang": entry.source_lang,
            "sample_index": entry.sample_index,
            "sample_hash": entry.sample_hash,
            "boundary_index": entry.boundary_index,
            "snippet_start": entry.snippet_start,
            "snippet_end": entry.snippet_end,
            "acquisition_score": entry.acquisition_score,
            "oracle_name": entry.oracle_name,
            "oracle_model": entry.oracle_model,
            "oracle_run_id": entry.oracle_run_id,
            "char_count": text_len,
            "diff_chars": diff_chars,
            "diff_ratio": diff_ratio,
            "metadata": metadata,
            "predicted_segments": [{"start": s.start, "end": s.end, "label": s.label} for s in pred_segments],
            "refined_segments": [{"start": s.start, "end": s.end, "label": s.label} for s in ref_segments],
            "predicted_html": _segments_to_html(text, pred_segments, sample_colors, diff_mask=None),
            "refined_html": _segments_to_html(text, ref_segments, sample_colors, diff_mask=None),
        }

    def labels_palette(self) -> List[Dict[str, str]]:
        labels = sorted(self.colors.keys())
        return [{"label": label, "color": self.colors[label]} for label in labels]

    def delete_sample(self, row_id: int) -> Dict[str, object]:
        target_id = int(row_id)
        if self._has_inference_rows():
            with self._connect() as con:
                row = con.execute(
                    """
                    SELECT id, round_id, source_lang, sample_hash, sample_index
                    FROM inference_samples
                    WHERE id = ?
                    """,
                    (target_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Unknown inference sample row id={target_id}")
                round_id = str(row["round_id"])
                sample_hash = str(row["sample_hash"])
                sample_index = int(row["sample_index"])
                source_lang = str(row["source_lang"])

                ref_cur = con.execute(
                    """
                    DELETE FROM refinements
                    WHERE round_id = ? AND sample_hash = ? AND sample_index = ?
                    """,
                    (round_id, sample_hash, sample_index),
                )
                inf_cur = con.execute(
                    "DELETE FROM inference_samples WHERE id = ?",
                    (target_id,),
                )
            return {
                "mode": "inference_sample",
                "deleted_sample_id": int(target_id),
                "round_id": round_id,
                "source_lang": source_lang,
                "sample_hash": sample_hash,
                "sample_index": int(sample_index),
                "deleted_inference_samples": max(0, int(inf_cur.rowcount)),
                "deleted_refinements": max(0, int(ref_cur.rowcount)),
            }

        with self._connect() as con:
            row = con.execute(
                """
                SELECT id, round_id, source_lang, sample_hash, sample_index, boundary_index
                FROM refinements
                WHERE id = ?
                """,
                (target_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown refinement row id={target_id}")
            cur = con.execute("DELETE FROM refinements WHERE id = ?", (target_id,))
        return {
            "mode": "refinement_row",
            "deleted_sample_id": int(target_id),
            "round_id": str(row["round_id"]),
            "source_lang": str(row["source_lang"]),
            "sample_hash": str(row["sample_hash"]),
            "sample_index": int(row["sample_index"]),
            "boundary_index": int(row["boundary_index"]),
            "deleted_inference_samples": 0,
            "deleted_refinements": max(0, int(cur.rowcount)),
        }


parser = argparse.ArgumentParser(description="Browse active-learning refinement samples.")
parser.add_argument("--store", type=str, default=str(DEFAULT_STORE_PATH), help="Path to AL SQLite store.")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8061)
parser.add_argument("--openapi", action="store_true", help="Expose OpenAPI docs.")
args, _ = parser.parse_known_args()

load_error: Optional[str] = None
store: Optional[ActiveLearningStore] = None
try:
    store = ActiveLearningStore(Path(args.store))
except Exception as exc:
    load_error = str(exc)

app = FastAPI(title="Active Learning Viewer", docs_url="/docs" if args.openapi else None)

static_path = Path(__file__).resolve().parent / "active_learning_static"
if static_path.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    if not static_path.is_dir():
        return HTMLResponse("<h1>UI not found</h1>", status_code=500)
    return FileResponse(str(static_path / "index.html"))


@app.get("/api/palette")
def api_palette():
    if load_error:
        raise HTTPException(status_code=500, detail=load_error)
    assert store is not None
    return {"palette": store.labels_palette()}


@app.get("/api/summary")
def api_summary(
    status: str = Query(default="ok"),
    round_id: Optional[str] = Query(default=None),
    queried: str = Query(default="all"),
):
    if load_error:
        raise HTTPException(status_code=500, detail=load_error)
    assert store is not None
    return store.summary(status=status, round_id=round_id, queried=queried)


@app.get("/api/samples")
def api_samples(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: str = Query(default="ok"),
    round_id: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    queried: str = Query(default="all"),
):
    if load_error:
        raise HTTPException(status_code=500, detail=load_error)
    assert store is not None
    return store.list_samples(
        page=page,
        page_size=page_size,
        status=status,
        round_id=round_id,
        search=search,
        queried=queried,
    )


@app.get("/api/sample/{row_id}")
def api_sample_detail(row_id: int):
    if load_error:
        raise HTTPException(status_code=500, detail=load_error)
    assert store is not None
    try:
        return store.sample_detail(row_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/api/sample/{row_id}")
def api_delete_sample(row_id: int):
    if load_error:
        raise HTTPException(status_code=500, detail=load_error)
    assert store is not None
    try:
        return store.delete_sample(row_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("active_learning_viewer:app", host=args.host, port=args.port, reload=False)
