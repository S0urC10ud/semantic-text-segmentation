#!/usr/bin/env python3
"""
Review viewer for curated benchmark JSONL files.

Highlights schema/segment issues and renders truth vs predicted segments.
"""

from __future__ import annotations

import argparse
import colorsys
import html
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg  # noqa: E402

try:
    from active_learning.oracle import GeminiBoundaryOracle  # noqa: E402
except Exception:  # pragma: no cover - optional dependency path
    GeminiBoundaryOracle = None


DEFAULT_JSONL_PATH = (
    REPO_ROOT / "active_learning" / "benchmark_data" / "curated_oracle_segments_v1.jsonl"
)
DEFAULT_STATIC_DIR = (REPO_ROOT / "viewers" / "curated_benchmark_static").resolve()
DEFAULT_BENCHMARK_RESULTS_DIR = (REPO_ROOT / "active_learning" / "benchmark_results").resolve()

LABEL_ALIASES = {
    "javascript": "javascript_typescript",
    "typescript": "javascript_typescript",
    "js": "javascript_typescript",
    "ts": "javascript_typescript",
    "c": "c_family",
    "cpp": "c_family",
    "c++": "c_family",
    "vb": "visual_basic",
    "gettext-catalog": "gettext_catalog",
}
ALLOWED_LABELS = set(cfg.LANG2ID.keys()) | {"other"}


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Issue:
    severity: str  # error|warn|info
    code: str
    message: str


@dataclass
class SampleReview:
    index: int
    line_number: int
    snippet_id: str
    task: str
    mixed_truth: bool
    text: str
    boundary: int
    source_langs: List[str]
    truth_segments: List[Segment]
    predicted_segments: List[Segment]
    issues: List[Issue] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    raw: Dict[str, object] = field(default_factory=dict)

    def add_issue(self, severity: str, code: str, message: str) -> None:
        self.issues.append(Issue(severity=str(severity), code=str(code), message=str(message)))

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warn_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warn")

    @property
    def info_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "info")

    @property
    def problem_count(self) -> int:
        return self.error_count + self.warn_count


@dataclass
class ParseError:
    line_number: int
    message: str


@dataclass(frozen=True)
class LiveRunManifest:
    run_id: str
    file_path: Path
    created_at: str
    ran_live_calls: bool
    batch_sizes: Tuple[int, ...]
    samples_total: int
    model: str
    oracle: str
    snippet_ids: frozenset[str]
    batch_diagnostics_by_size: Dict[int, List[Dict[str, object]]]
    snippet_sources_by_size: Dict[int, Dict[str, str]]
    failure_reasons_by_size: Dict[int, Dict[str, str]]


def _canonical_label(label: str) -> str:
    raw = str(label or "").strip().lower().replace("-", "_")
    raw = LABEL_ALIASES.get(raw, raw)
    if raw in ALLOWED_LABELS:
        return raw
    return "other"


def _auto_color(label: str) -> str:
    base = sum(ord(c) for c in label) + len(label) * 131
    hue = (base % 360) / 360.0
    sat = 0.63
    lig = 0.53
    r, g, b = colorsys.hls_to_rgb(hue, lig, sat)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def _palette() -> Dict[str, str]:
    out: Dict[str, str] = {}
    labels = [name for name, _ in sorted(cfg.LANG2ID.items(), key=lambda kv: kv[1])]
    n = max(1, len(labels))
    for idx, label in enumerate(labels):
        hue = ((idx / n) + 0.17) % 1.0
        sat = 0.76
        lig = 0.50 if (idx % 2 == 0) else 0.40
        r, g, b = colorsys.hls_to_rgb(hue, lig, sat)
        out[label] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    out["other"] = out.get("other", "#9a6bff")
    out["unlabeled"] = "#9aa4b2"
    return out


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = str(hex_color or "").lstrip("#")
    if len(h) != 6:
        return f"rgba(160,160,160,{alpha})"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _segments_to_html(
    text: str,
    segments: Sequence[Segment],
    *,
    colors: Mapping[str, str],
    diff_mask: Optional[Sequence[bool]] = None,
) -> str:
    if not text:
        return "<em>Empty sample.</em>"
    n = len(text)
    chunks: List[str] = []

    def _is_diff(idx: int) -> bool:
        if diff_mask is None:
            return False
        if idx < 0 or idx >= len(diff_mask):
            return False
        return bool(diff_mask[idx])

    def _emit(start: int, end: int, label: str, color: str, diff: bool) -> None:
        if end <= start:
            return
        class_name = "tok"
        if diff_mask is not None:
            class_name += " diff" if diff else " same"
        span_text = html.escape(text[start:end])
        chunks.append(
            f'<span class="{class_name}" style="background:{_hex_to_rgba(color, 0.22)};" '
            f'title="{html.escape(label)}">{span_text}</span>'
        )

    for seg in segments:
        start = max(0, min(n, int(seg.start)))
        end = max(start, min(n, int(seg.end)))
        if end <= start:
            continue
        label = _canonical_label(seg.label)
        color = colors.get(label, _auto_color(label))
        if diff_mask is None:
            _emit(start, end, label, color, diff=False)
            continue
        run_start = start
        run_diff = _is_diff(start)
        for i in range(start + 1, end):
            cur = _is_diff(i)
            if cur == run_diff:
                continue
            _emit(run_start, i, label, color, run_diff)
            run_start = i
            run_diff = cur
        _emit(run_start, end, label, color, run_diff)
    return "".join(chunks) if chunks else "<em>No renderable segments.</em>"


def _labels_from_segments(text_len: int, segments: Sequence[Segment]) -> List[str]:
    labels = ["unlabeled"] * max(0, int(text_len))
    for seg in segments:
        start = max(0, min(text_len, int(seg.start)))
        end = max(start, min(text_len, int(seg.end)))
        label = _canonical_label(seg.label)
        for idx in range(start, end):
            labels[idx] = label
    return labels


def _segments_from_labels(labels: Sequence[str]) -> List[Segment]:
    if not labels:
        return []
    out: List[Segment] = []
    start = 0
    cur = _canonical_label(str(labels[0]))
    for idx in range(1, len(labels)):
        nxt = _canonical_label(str(labels[idx]))
        if nxt != cur:
            out.append(Segment(start=start, end=idx, label=cur))
            start = idx
            cur = nxt
    out.append(Segment(start=start, end=len(labels), label=cur))
    return out


def _segments_from_oracle_rows(rows: Any) -> List[Segment]:
    if not isinstance(rows, list):
        return []
    out: List[Segment] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            start = int(row.get("start", 0))
            end = int(row.get("end", 0))
        except Exception:
            continue
        if end <= start:
            continue
        label_raw = row.get("raw_label")
        if label_raw is None:
            label_raw = row.get("label", "other")
        label = _canonical_label(str(label_raw))
        out.append(Segment(start=start, end=end, label=label))
    out.sort(key=lambda seg: (int(seg.start), int(seg.end)))
    return out


def _segments_from_oracle_objects(rows: Any) -> List[Segment]:
    if not isinstance(rows, list):
        return []
    out: List[Segment] = []
    for row in rows:
        try:
            start = int(getattr(row, "start", 0))
            end = int(getattr(row, "end", 0))
            raw_label = getattr(row, "raw_label", None)
            if raw_label is None:
                raw_label = getattr(row, "label", "other")
            label = _canonical_label(str(raw_label))
        except Exception:
            continue
        if end <= start:
            continue
        out.append(Segment(start=start, end=end, label=label))
    out.sort(key=lambda seg: (int(seg.start), int(seg.end)))
    return out


def _overlay_segments_on_labels(
    *,
    base_labels: Sequence[str],
    overlay_segments: Sequence[Segment],
) -> List[str]:
    out = [_canonical_label(str(label)) for label in base_labels]
    n = len(out)
    for seg in overlay_segments:
        start = max(0, min(n, int(seg.start)))
        end = max(start, min(n, int(seg.end)))
        if end <= start:
            continue
        label = _canonical_label(seg.label)
        for idx in range(start, end):
            out[idx] = label
    return out


def _segment_rows_to_segments(
    rows: Any,
    *,
    field_name: str,
    text_len: int,
    review: SampleReview,
    require_full_coverage: bool,
) -> List[Segment]:
    out: List[Segment] = []
    if not isinstance(rows, list):
        review.add_issue(
            "error",
            f"{field_name}_not_list",
            f"{field_name} must be a list of segment objects.",
        )
        return out

    raw_rows: List[Tuple[int, int, str, int]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            review.add_issue(
                "error",
                f"{field_name}_item_not_object",
                f"{field_name}[{idx}] is not an object.",
            )
            continue
        start_obj = row.get("start")
        end_obj = row.get("end")
        label_obj = row.get("label")
        try:
            start = int(start_obj)
            end = int(end_obj)
        except Exception:
            review.add_issue(
                "error",
                f"{field_name}_invalid_offsets",
                f"{field_name}[{idx}] has invalid start/end offsets.",
            )
            continue
        label_raw = str(label_obj or "").strip()
        label = _canonical_label(label_raw)
        if label_raw and label == "other" and _canonical_label(label_raw) == "other" and label_raw.lower() not in {
            "other",
            "text",
        }:
            review.add_issue(
                "warn",
                f"{field_name}_unknown_label",
                f"{field_name}[{idx}] unknown label '{label_raw}', canonicalized to 'other'.",
            )
        if start < 0 or end < 0:
            review.add_issue(
                "error",
                f"{field_name}_negative_offset",
                f"{field_name}[{idx}] has negative offsets.",
            )
        if end <= start:
            review.add_issue(
                "error",
                f"{field_name}_non_positive_length",
                f"{field_name}[{idx}] end must be > start.",
            )
            continue
        if start > text_len or end > text_len:
            review.add_issue(
                "error",
                f"{field_name}_offset_out_of_range",
                f"{field_name}[{idx}] offset exceeds text length {text_len}.",
            )
        start = max(0, min(text_len, start))
        end = max(start, min(text_len, end))
        if end <= start:
            continue
        raw_rows.append((start, end, label, idx))

    if not raw_rows:
        return out

    sorted_rows = sorted(raw_rows, key=lambda item: (item[0], item[1], item[3]))
    if [idx for _, _, _, idx in sorted_rows] != list(range(len(sorted_rows))):
        review.add_issue(
            "warn",
            f"{field_name}_unsorted",
            f"{field_name} is not sorted by start offset.",
        )

    cursor = 0
    prev_label = None
    for start, end, label, raw_idx in sorted_rows:
        if start > cursor:
            review.add_issue(
                "error" if require_full_coverage else "warn",
                f"{field_name}_gap",
                f"{field_name} gap before segment index {raw_idx}: [{cursor}, {start}).",
            )
        if start < cursor:
            review.add_issue(
                "error",
                f"{field_name}_overlap",
                f"{field_name} overlap at segment index {raw_idx}.",
            )
        if prev_label == label and start == cursor:
            review.add_issue(
                "warn",
                f"{field_name}_adjacent_same_label",
                f"{field_name} has adjacent same-label segments ('{label}').",
            )
        clipped_start = max(cursor, start)
        if end > clipped_start:
            out.append(Segment(start=clipped_start, end=end, label=label))
        cursor = max(cursor, end)
        prev_label = label
    if require_full_coverage:
        if out and out[0].start != 0:
            review.add_issue(
                "error",
                f"{field_name}_no_head_coverage",
                f"{field_name} does not start at 0.",
            )
        if cursor < text_len:
            review.add_issue(
                "error",
                f"{field_name}_tail_uncovered",
                f"{field_name} leaves tail [{cursor}, {text_len}) uncovered.",
            )
    return out


class ReviewStore:
    def __init__(self, jsonl_path: Path, expected_length: int, benchmark_results_dir: Path) -> None:
        self.jsonl_path = jsonl_path.resolve()
        self.expected_length = int(expected_length)
        self.benchmark_results_dir = benchmark_results_dir.resolve()
        self.palette = _palette()
        self.entries: List[SampleReview] = []
        self.entries_by_snippet_id: Dict[str, SampleReview] = {}
        self.parse_errors: List[ParseError] = []
        self.duplicate_ids: set[str] = set()
        self.live_manifests: Dict[str, LiveRunManifest] = {}
        self.live_run_items: List[Dict[str, object]] = []
        self.live_overlay_cache: Dict[Tuple[str, int], Dict[str, List[Segment]]] = {}
        self._reload()

    def _reload(self) -> None:
        self.entries = []
        self.entries_by_snippet_id = {}
        self.parse_errors = []
        self.duplicate_ids = set()
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {self.jsonl_path}")
        lines = self.jsonl_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            text = str(line or "").strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                self.parse_errors.append(ParseError(line_number=line_number, message=str(exc)))
                continue
            if not isinstance(row, dict):
                self.parse_errors.append(
                    ParseError(line_number=line_number, message="Top-level JSON must be an object.")
                )
                continue
            entry = self._validate_row(row=row, line_number=line_number, index=len(self.entries))
            self.entries.append(entry)
            self.entries_by_snippet_id[entry.snippet_id] = entry

        by_id: Dict[str, List[int]] = defaultdict(list)
        for idx, entry in enumerate(self.entries):
            by_id[entry.snippet_id].append(idx)
        for sid, idxs in by_id.items():
            if len(idxs) <= 1:
                continue
            self.duplicate_ids.add(sid)
            for idx in idxs:
                self.entries[idx].add_issue(
                    "error",
                    "duplicate_snippet_id",
                    f"snippet_id '{sid}' appears {len(idxs)} times.",
                )
        self._refresh_live_runs()

    def _refresh_live_runs(self) -> None:
        self.live_manifests = {}
        self.live_run_items = []
        self.live_overlay_cache = {}
        if not self.benchmark_results_dir.exists() or not self.benchmark_results_dir.is_dir():
            return

        files = sorted(
            self.benchmark_results_dir.glob("al_oracle_batch_benchmark_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            config_obj = payload.get("config")
            config = dict(config_obj) if isinstance(config_obj, dict) else {}
            runs_obj = payload.get("runs")
            runs = list(runs_obj) if isinstance(runs_obj, list) else []
            batch_sizes: List[int] = []
            by_batch: Dict[int, List[Dict[str, object]]] = defaultdict(list)
            source_by_batch: Dict[int, Dict[str, str]] = defaultdict(dict)
            reason_by_batch: Dict[int, Dict[str, str]] = defaultdict(dict)
            for run in runs:
                if not isinstance(run, dict):
                    continue
                try:
                    batch_size = int(run.get("batch_size", 0))
                except Exception:
                    continue
                if batch_size <= 0:
                    continue
                if batch_size not in batch_sizes:
                    batch_sizes.append(batch_size)
                source_obj = run.get("snippet_sources")
                if isinstance(source_obj, dict):
                    for sid_obj, state_obj in source_obj.items():
                        sid = str(sid_obj).strip()
                        if not sid:
                            continue
                        source_by_batch[batch_size][sid] = str(state_obj)
                diag_obj = run.get("batch_diagnostics")
                diagnostics = list(diag_obj) if isinstance(diag_obj, list) else []
                normalized_diags = [dict(item) for item in diagnostics if isinstance(item, dict)]
                by_batch[batch_size].extend(normalized_diags)
                for diag in normalized_diags:
                    parse_failed_obj = diag.get("parse_failed_snippet_ids")
                    parse_failed_ids = list(parse_failed_obj) if isinstance(parse_failed_obj, list) else []
                    for sid_obj in parse_failed_ids:
                        sid = str(sid_obj).strip()
                        if not sid:
                            continue
                        source_by_batch[batch_size].setdefault(
                            sid,
                            "fallback_failed_segmentation",
                        )
                    missing_obj = diag.get("missing_snippet_ids")
                    missing_ids = list(missing_obj) if isinstance(missing_obj, list) else []
                    for sid_obj in missing_ids:
                        sid = str(sid_obj).strip()
                        if not sid:
                            continue
                        source_by_batch[batch_size].setdefault(
                            sid,
                            "fallback_missing_snippet",
                        )
                    reasons_obj = diag.get("parse_failed_reasons")
                    if isinstance(reasons_obj, dict):
                        for sid_obj, reason_obj in reasons_obj.items():
                            sid = str(sid_obj).strip()
                            if not sid:
                                continue
                            reason_by_batch[batch_size][sid] = str(reason_obj)

            samples_obj = payload.get("samples")
            sample_rows = list(samples_obj) if isinstance(samples_obj, list) else []
            snippet_ids: set[str] = set()
            for row in sample_rows:
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("snippet_id", "")).strip()
                if sid:
                    snippet_ids.add(sid)

            run_id = str(path.stem)
            created_at = str(payload.get("created_at", "")).strip()
            ran_live_calls = bool(config.get("ran_live_calls", False))
            manifest = LiveRunManifest(
                run_id=run_id,
                file_path=path.resolve(),
                created_at=created_at,
                ran_live_calls=ran_live_calls,
                batch_sizes=tuple(sorted(batch_sizes)),
                samples_total=int(len(sample_rows)),
                model=str(config.get("model", "")),
                oracle=str(config.get("oracle", "")),
                snippet_ids=frozenset(snippet_ids),
                batch_diagnostics_by_size={int(k): list(v) for k, v in by_batch.items()},
                snippet_sources_by_size={int(k): dict(v) for k, v in source_by_batch.items()},
                failure_reasons_by_size={int(k): dict(v) for k, v in reason_by_batch.items()},
            )
            self.live_manifests[run_id] = manifest
            self.live_run_items.append(
                {
                    "run_id": run_id,
                    "file_name": path.name,
                    "file_path": str(path.resolve()),
                    "created_at": created_at,
                    "ran_live_calls": bool(ran_live_calls),
                    "batch_sizes": list(manifest.batch_sizes),
                    "samples_total": int(manifest.samples_total),
                    "model": manifest.model,
                    "oracle": manifest.oracle,
                    "snippet_count": int(len(manifest.snippet_ids)),
                }
            )

    def list_live_runs(self) -> Dict[str, object]:
        return {
            "results_dir": str(self.benchmark_results_dir),
            "runs": list(self.live_run_items),
        }

    def _manifest_for_run(self, run_id: str) -> Optional[LiveRunManifest]:
        key = str(run_id or "").strip()
        if not key:
            return None
        return self.live_manifests.get(key)

    def _resolve_batch_size(self, run_id: str, batch_size: Optional[int]) -> Optional[int]:
        manifest = self._manifest_for_run(run_id)
        if manifest is None:
            return None
        if batch_size is not None and int(batch_size) > 0 and int(batch_size) in manifest.batch_sizes:
            return int(batch_size)
        if not manifest.batch_sizes:
            return None
        return int(max(manifest.batch_sizes))

    def _resolve_log_path(self, raw_path: object) -> Optional[Path]:
        value = str(raw_path or "").strip()
        if not value:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / value).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    def _live_overlay_segments(self, *, run_id: str, batch_size: int) -> Dict[str, List[Segment]]:
        key = (str(run_id), int(batch_size))
        cached = self.live_overlay_cache.get(key)
        if cached is not None:
            return cached

        manifest = self._manifest_for_run(run_id)
        if manifest is None:
            self.live_overlay_cache[key] = {}
            return {}
        diagnostics = list(manifest.batch_diagnostics_by_size.get(int(batch_size), []))
        overlay: Dict[str, List[Segment]] = {}
        for diag in diagnostics:
            log_path = self._resolve_log_path(diag.get("log_path"))
            if log_path is None:
                continue
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            seg_rows_obj = payload.get("segments")
            seg_rows = list(seg_rows_obj) if isinstance(seg_rows_obj, list) else []
            for row in seg_rows:
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("snippet_id", "")).strip()
                if not sid:
                    continue
                oracle_rows = row.get("segments")
                parsed = _segments_from_oracle_rows(oracle_rows)
                overlay[sid] = parsed
            if GeminiBoundaryOracle is not None:
                metadata_obj = payload.get("metadata")
                metadata = dict(metadata_obj) if isinstance(metadata_obj, dict) else {}
                snippet_ids_obj = metadata.get("snippet_ids")
                snippet_ids = list(snippet_ids_obj) if isinstance(snippet_ids_obj, list) else []
                snippet_text_by_id: Dict[str, str] = {}
                for sid_obj in snippet_ids:
                    sid = str(sid_obj).strip()
                    if not sid:
                        continue
                    entry = self.entries_by_snippet_id.get(sid)
                    if entry is None:
                        continue
                    snippet_text_by_id[sid] = str(entry.text)
                raw_generated = payload.get("generated_code")
                generated_code = str(raw_generated) if isinstance(raw_generated, str) else ""
                if generated_code and snippet_text_by_id:
                    try:
                        reparsed_map, _failures, _returned = GeminiBoundaryOracle._parse_segments_payload(
                            generated_code,
                            snippet_text_by_id=snippet_text_by_id,
                        )
                    except Exception:
                        reparsed_map = {}
                    for sid, segs in reparsed_map.items():
                        if sid in overlay and overlay[sid]:
                            continue
                        parsed = _segments_from_oracle_objects(segs)
                        if parsed:
                            overlay[sid] = parsed
        self.live_overlay_cache[key] = overlay
        return overlay

    def _prediction_segments_for_entry(
        self,
        entry: SampleReview,
        *,
        live_run_id: str = "",
        live_batch_size: Optional[int] = None,
    ) -> Tuple[List[Segment], str, Optional[int], bool, str]:
        pred_segments = list(entry.predicted_segments)
        prediction_source = "dataset_prior"
        prediction_failure_reason = ""
        resolved_run_id = str(live_run_id or "").strip()
        resolved_batch = self._resolve_batch_size(resolved_run_id, live_batch_size)
        live_overlay_applied = False
        if resolved_run_id and resolved_batch is not None:
            manifest = self._manifest_for_run(resolved_run_id)
            in_selected_run = bool(
                manifest is not None and entry.snippet_id in manifest.snippet_ids
            )
            source_map = (
                manifest.snippet_sources_by_size.get(int(resolved_batch), {})
                if manifest is not None
                else {}
            )
            has_source_map = bool(source_map)
            reason_map = (
                manifest.failure_reasons_by_size.get(int(resolved_batch), {})
                if manifest is not None
                else {}
            )
            source_state = str(source_map.get(entry.snippet_id, "")).strip()
            prediction_failure_reason = str(reason_map.get(entry.snippet_id, "")).strip()
            overlay = self._live_overlay_segments(run_id=resolved_run_id, batch_size=int(resolved_batch))
            oracle_segments = overlay.get(entry.snippet_id, [])
            if oracle_segments:
                pred_segments = list(oracle_segments)
                live_overlay_applied = True
                if source_state.startswith("fallback_"):
                    prediction_source = "gemini_live_recovered"
                elif source_state and source_state != "model":
                    prediction_source = source_state
                else:
                    prediction_source = "gemini_live"
                prediction_failure_reason = ""
            else:
                if source_state and source_state != "model":
                    prediction_source = source_state
                elif source_state == "model":
                    prediction_source = "gemini_live_unavailable"
                elif has_source_map:
                    prediction_source = (
                        "dataset_prior_not_in_run" if not in_selected_run else "gemini_live_unavailable"
                    )
                else:
                    prediction_source = "dataset_prior_live_untracked"
                if source_state.startswith("fallback_") or source_state == "model":
                    pred_segments = []
                elif has_source_map and in_selected_run:
                    pred_segments = []
        return (
            pred_segments,
            prediction_source,
            resolved_batch,
            live_overlay_applied,
            prediction_failure_reason,
        )

    @staticmethod
    def _mismatch_stats(text_len: int, truth_segments: Sequence[Segment], pred_segments: Sequence[Segment]) -> Tuple[int, float]:
        if text_len <= 0:
            return 0, 0.0
        truth_labels = _labels_from_segments(text_len, truth_segments)
        pred_labels = _labels_from_segments(text_len, pred_segments)
        n = min(len(truth_labels), len(pred_labels), int(text_len))
        mismatches = 0
        for idx in range(n):
            if truth_labels[idx] != pred_labels[idx]:
                mismatches += 1
        mismatch_rate = float(mismatches) / float(max(1, text_len))
        return int(mismatches), float(mismatch_rate)

    def _validate_row(self, *, row: Dict[str, object], line_number: int, index: int) -> SampleReview:
        snippet_id = str(row.get("snippet_id", "")).strip() or f"line-{line_number}"
        task = str(row.get("task", "")).strip()
        text = row.get("text")
        if not isinstance(text, str):
            text = ""
        text_len = len(text)
        mixed_truth = bool(row.get("mixed_truth", False))

        boundary_obj = row.get("boundary", text_len // 2)
        try:
            boundary = int(boundary_obj)
        except Exception:
            boundary = text_len // 2

        source_langs_obj = row.get("source_langs")
        source_langs: List[str] = []
        if isinstance(source_langs_obj, list):
            source_langs = [_canonical_label(str(item)) for item in source_langs_obj if str(item).strip()]
        metadata_obj = row.get("metadata")
        metadata = dict(metadata_obj) if isinstance(metadata_obj, dict) else {}

        review = SampleReview(
            index=index,
            line_number=line_number,
            snippet_id=snippet_id,
            task=task,
            mixed_truth=mixed_truth,
            text=text,
            boundary=boundary,
            source_langs=source_langs,
            truth_segments=[],
            predicted_segments=[],
            metadata=metadata,
            raw=dict(row),
        )

        if not snippet_id:
            review.add_issue("error", "missing_snippet_id", "snippet_id is missing/empty.")
        if self.expected_length > 0 and text_len != self.expected_length:
            review.add_issue(
                "warn",
                "text_length_mismatch",
                f"text length is {text_len}, expected {self.expected_length}.",
            )
        if text_len <= 0:
            review.add_issue("error", "empty_text", "text is empty.")

        review.truth_segments = _segment_rows_to_segments(
            row.get("truth_segments"),
            field_name="truth_segments",
            text_len=text_len,
            review=review,
            require_full_coverage=True,
        )
        review.predicted_segments = _segment_rows_to_segments(
            row.get("predicted_segments"),
            field_name="predicted_segments",
            text_len=text_len,
            review=review,
            require_full_coverage=True,
        )

        truth_labels = set(segment.label for segment in review.truth_segments)
        if mixed_truth and len(truth_labels) < 2:
            review.add_issue(
                "warn",
                "mixed_truth_inconsistent",
                "mixed_truth=true but truth segments contain only one label.",
            )
        if (not mixed_truth) and len(truth_labels) > 1:
            review.add_issue(
                "warn",
                "mixed_truth_inconsistent",
                "mixed_truth=false but truth segments contain multiple labels.",
            )
        if source_langs:
            truth_non_empty = {label for label in truth_labels if label}
            src = set(source_langs)
            missing = sorted(truth_non_empty - src)
            extra = sorted(src - truth_non_empty)
            if missing:
                review.add_issue(
                    "warn",
                    "source_langs_missing_truth_labels",
                    f"source_langs missing truth labels: {', '.join(missing)}",
                )
            if extra:
                review.add_issue(
                    "warn",
                    "source_langs_extra_labels",
                    f"source_langs has labels not in truth segments: {', '.join(extra)}",
                )

        if boundary <= 0 or boundary >= text_len:
            review.add_issue(
                "warn",
                "boundary_out_of_range",
                f"boundary={boundary} is outside (0, {text_len}).",
            )
        else:
            truth_dense = _labels_from_segments(text_len, review.truth_segments)
            if truth_dense and truth_dense[boundary - 1] == truth_dense[boundary]:
                review.add_issue(
                    "info",
                    "boundary_not_on_truth_transition",
                    f"boundary={boundary} is not at a truth label transition.",
                )
        return review

    def reload(self) -> None:
        self._reload()

    def summary(self) -> Dict[str, object]:
        severity_counts = Counter()
        issue_code_counts = Counter()
        task_counts = Counter()
        labels_sample_counts = Counter()
        labels_char_counts = Counter()
        text_lengths: List[int] = []
        mixed_count = 0
        samples_with_issues = 0
        samples_with_problems = 0

        for entry in self.entries:
            task_counts[entry.task or "unknown"] += 1
            if entry.mixed_truth:
                mixed_count += 1
            if entry.issue_count > 0:
                samples_with_issues += 1
            if entry.problem_count > 0:
                samples_with_problems += 1
            text_lengths.append(len(entry.text))
            for issue in entry.issues:
                severity_counts[issue.severity] += 1
                issue_code_counts[issue.code] += 1
            for label in set(entry.source_langs):
                labels_sample_counts[label] += 1
            for seg in entry.truth_segments:
                labels_char_counts[seg.label] += int(seg.end) - int(seg.start)

        expected = self.expected_length if self.expected_length > 0 else None
        avg_len = float(sum(text_lengths)) / float(len(text_lengths)) if text_lengths else 0.0
        return {
            "path": str(self.jsonl_path),
            "expected_length": expected,
            "total_rows": len(self.entries),
            "parse_errors": len(self.parse_errors),
            "samples_with_issues": int(samples_with_issues),
            "samples_without_issues": int(max(0, len(self.entries) - samples_with_issues)),
            "samples_with_problems": int(samples_with_problems),
            "samples_without_problems": int(max(0, len(self.entries) - samples_with_problems)),
            "mixed_rows": int(mixed_count),
            "non_mixed_rows": int(max(0, len(self.entries) - mixed_count)),
            "avg_text_length": avg_len,
            "min_text_length": min(text_lengths) if text_lengths else 0,
            "max_text_length": max(text_lengths) if text_lengths else 0,
            "severity_counts": dict(severity_counts),
            "issue_code_counts": dict(issue_code_counts),
            "task_counts": dict(task_counts),
            "label_sample_counts": dict(labels_sample_counts),
            "label_char_counts": dict(labels_char_counts),
        }

    def list_samples(
        self,
        *,
        page: int,
        page_size: int,
        status: str,
        q: str,
        run_id: str = "",
        run_only: bool = False,
        live_batch_size: Optional[int] = None,
    ) -> Dict[str, object]:
        page = max(1, int(page))
        page_size = max(1, min(200, int(page_size)))
        status_key = str(status or "all").strip().lower()
        needle = str(q or "").strip().lower()
        run_key = str(run_id or "").strip()
        manifest = self._manifest_for_run(run_key) if run_key else None
        run_snippet_ids = set(manifest.snippet_ids) if manifest is not None else set()
        restrict_to_run = bool(run_only and manifest is not None and run_snippet_ids)

        filtered: List[SampleReview] = []
        computed_mismatch: Dict[str, Tuple[int, float, str, str]] = {}
        for entry in self.entries:
            if restrict_to_run and entry.snippet_id not in run_snippet_ids:
                continue
            if status_key == "ok" and entry.problem_count > 0:
                continue
            if status_key == "issues" and entry.problem_count == 0:
                continue
            if status_key == "errors" and entry.error_count == 0:
                continue
            (
                pred_segments,
                prediction_source,
                _resolved_batch,
                _live_overlay_applied,
                prediction_failure_reason,
            ) = self._prediction_segments_for_entry(
                entry,
                live_run_id=run_key,
                live_batch_size=live_batch_size,
            )
            mismatch_chars, mismatch_rate = self._mismatch_stats(
                len(entry.text),
                entry.truth_segments,
                pred_segments,
            )
            computed_mismatch[entry.snippet_id] = (
                mismatch_chars,
                mismatch_rate,
                prediction_source,
                prediction_failure_reason,
            )
            failed_states = {
                "fallback_failed_segmentation",
                "fallback_missing_snippet",
                "fallback_oracle_error",
                "fallback_runtime_error",
                "gemini_live_unavailable",
            }
            is_failed_seg = prediction_source in failed_states
            if status_key == "incorrect" and (mismatch_chars <= 0 or is_failed_seg):
                continue
            if status_key == "correct" and (mismatch_chars > 0 or is_failed_seg):
                continue
            if status_key == "failed_segmentation":
                if not is_failed_seg:
                    continue
            if needle:
                hay = " ".join(
                    [
                        entry.snippet_id,
                        entry.task,
                        " ".join(entry.source_langs),
                        str(entry.metadata.get("note", "")),
                        entry.text[:220],
                    ]
                ).lower()
                if needle not in hay:
                    continue
            filtered.append(entry)

        total = len(filtered)
        total_pages = max(1, int(math.ceil(float(total) / float(page_size)))) if total > 0 else 1
        page = min(page, total_pages)
        start = (page - 1) * page_size
        end = start + page_size
        items = []
        for entry in filtered[start:end]:
            issue_codes = sorted({issue.code for issue in entry.issues})
            preview = entry.text.replace("\n", "\\n")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            mismatch_chars, mismatch_rate, prediction_source, prediction_failure_reason = computed_mismatch.get(
                entry.snippet_id,
                (0, 0.0, "dataset_prior", ""),
            )
            items.append(
                {
                    "index": int(entry.index),
                    "line_number": int(entry.line_number),
                    "snippet_id": entry.snippet_id,
                    "task": entry.task,
                    "mixed_truth": bool(entry.mixed_truth),
                    "text_len": len(entry.text),
                    "boundary": int(entry.boundary),
                    "issue_count": int(entry.issue_count),
                    "problem_count": int(entry.problem_count),
                    "error_count": int(entry.error_count),
                    "warn_count": int(entry.warn_count),
                    "info_count": int(entry.info_count),
                    "issue_codes": issue_codes,
                    "source_langs": list(entry.source_langs),
                    "preview": preview,
                    "in_selected_run": bool(manifest is not None and entry.snippet_id in run_snippet_ids),
                    "mismatch_chars": int(mismatch_chars),
                    "mismatch_rate": float(mismatch_rate),
                    "prediction_source": str(prediction_source),
                    "prediction_failure_reason": str(prediction_failure_reason),
                }
            )
        return {
            "items": items,
            "total": int(total),
            "page": int(page),
            "page_size": int(page_size),
            "total_pages": int(total_pages),
            "run_id": run_key,
            "run_only": bool(restrict_to_run),
            "run_samples_total": int(len(run_snippet_ids)),
        }

    def sample_detail(
        self,
        index: int,
        *,
        live_run_id: str = "",
        live_batch_size: Optional[int] = None,
    ) -> Dict[str, object]:
        if index < 0 or index >= len(self.entries):
            raise KeyError(f"sample index {index} out of range")
        entry = self.entries[index]
        text_len = len(entry.text)
        truth_labels = _labels_from_segments(text_len, entry.truth_segments)

        (
            pred_segments,
            prediction_source,
            resolved_batch,
            live_overlay_applied,
            prediction_failure_reason,
        ) = self._prediction_segments_for_entry(
            entry,
            live_run_id=live_run_id,
            live_batch_size=live_batch_size,
        )
        resolved_run_id = str(live_run_id or "").strip()

        pred_labels = _labels_from_segments(text_len, pred_segments)
        mismatch_chars, mismatch_rate = self._mismatch_stats(
            text_len,
            entry.truth_segments,
            pred_segments,
        )
        diff_mask = [
            bool(truth_labels[i] != pred_labels[i]) if i < len(truth_labels) and i < len(pred_labels) else False
            for i in range(text_len)
        ]
        return {
            "index": int(entry.index),
            "line_number": int(entry.line_number),
            "snippet_id": entry.snippet_id,
            "task": entry.task,
            "mixed_truth": bool(entry.mixed_truth),
            "text_len": text_len,
            "boundary": int(entry.boundary),
            "source_langs": list(entry.source_langs),
            "issues": [
                {"severity": issue.severity, "code": issue.code, "message": issue.message}
                for issue in entry.issues
            ],
            "truth_segments": [seg.__dict__ for seg in entry.truth_segments],
            "predicted_segments": [seg.__dict__ for seg in pred_segments],
            "text": entry.text,
            "truth_html": _segments_to_html(
                entry.text,
                entry.truth_segments,
                colors=self.palette,
            ),
            "predicted_html": _segments_to_html(
                entry.text,
                pred_segments,
                colors=self.palette,
            ),
            "diff_html": _segments_to_html(
                entry.text,
                entry.truth_segments,
                colors=self.palette,
                diff_mask=diff_mask,
            ),
            "prediction_source": prediction_source,
            "prediction_failure_reason": str(prediction_failure_reason),
            "live_run_id": resolved_run_id if resolved_run_id else "",
            "live_batch_size": int(resolved_batch) if resolved_batch is not None else None,
            "live_prediction_applied": bool(live_overlay_applied),
            "mismatch_chars": int(mismatch_chars),
            "mismatch_rate": float(mismatch_rate),
            "metadata": entry.metadata,
            "raw": entry.raw,
        }

    def label_palette(self) -> Dict[str, str]:
        return dict(self.palette)


def create_app(store: ReviewStore, static_dir: Path) -> FastAPI:
    app = FastAPI(title="Curated Benchmark JSONL Viewer")
    if not static_dir.exists():
        raise FileNotFoundError(f"Static dir not found: {static_dir}")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(str(static_dir / "index.html"))

    @app.get("/api/summary")
    def api_summary() -> Dict[str, object]:
        summary = store.summary()
        summary["palette"] = store.label_palette()
        summary["parse_error_rows"] = [
            {"line_number": err.line_number, "message": err.message}
            for err in store.parse_errors[:100]
        ]
        summary["duplicate_ids"] = sorted(store.duplicate_ids)
        summary["live_runs_count"] = int(len(store.live_run_items))
        return summary

    @app.get("/api/live-runs")
    def api_live_runs() -> Dict[str, object]:
        return store.list_live_runs()

    @app.get("/api/samples")
    def api_samples(
        page: int = Query(1, ge=1),
        page_size: int = Query(25, ge=1, le=200),
        status: str = Query("all"),
        q: str = Query(""),
        run_id: str = Query(""),
        run_only: bool = Query(False),
        batch_size: Optional[int] = Query(None),
    ) -> Dict[str, object]:
        return store.list_samples(
            page=page,
            page_size=page_size,
            status=status,
            q=q,
            run_id=run_id,
            run_only=bool(run_only),
            live_batch_size=batch_size,
        )

    @app.get("/api/sample/{index}")
    def api_sample(
        index: int,
        run_id: str = Query(""),
        batch_size: Optional[int] = Query(None),
    ) -> Dict[str, object]:
        try:
            return store.sample_detail(
                int(index),
                live_run_id=run_id,
                live_batch_size=batch_size,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/reload")
    def api_reload() -> Dict[str, object]:
        store.reload()
        return {"status": "ok", "total_rows": len(store.entries), "parse_errors": len(store.parse_errors)}

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review curated benchmark JSONL for mistakes.")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL_PATH)
    parser.add_argument("--benchmark-results-dir", type=Path, default=DEFAULT_BENCHMARK_RESULTS_DIR)
    parser.add_argument("--expected-length", type=int, default=256)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    store = ReviewStore(
        jsonl_path=Path(args.jsonl),
        expected_length=int(args.expected_length),
        benchmark_results_dir=Path(args.benchmark_results_dir),
    )
    app = create_app(store=store, static_dir=DEFAULT_STATIC_DIR)
    print(
        f"Curated benchmark viewer: rows={len(store.entries)}, parse_errors={len(store.parse_errors)} "
        f"file={Path(args.jsonl).resolve()}, live_runs={len(store.live_run_items)}",
        flush=True,
    )
    uvicorn.run(
        app,
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
    )


if __name__ == "__main__":
    main()
