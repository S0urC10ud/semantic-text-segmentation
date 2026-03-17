#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

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
    _resolve_hparam,
)

from active_learning.label_store import LabelStore, StoredRefinement  # noqa: E402
from active_learning.oracle import (  # noqa: E402
    BoundarySnippet,
    GeminiBoundaryOracle,
    OracleSegment,
    StubOracle,
)


DEFAULT_STORE_PATH = (REPO_ROOT / "active_learning" / "label_store.sqlite").resolve()
DEFAULT_PROGRESS_DB_PATH = (REPO_ROOT / "active_learning" / "monitor_relabel_progress.sqlite").resolve()
DEFAULT_MONITOR_ROOTS: Dict[str, Path] = {
    "monitor_a": (REPO_ROOT / "downloader" / "monitor_preprocessed_a").resolve(),
    "monitor_b": (REPO_ROOT / "downloader" / "monitor_preprocessed_b").resolve(),
}
VISIBLE_ASCII_BYTES = tuple(range(0x20, 0x7F))
VISIBLE_BYTE_SET = set(VISIBLE_ASCII_BYTES)


class ByteBatchPredictor(Protocol):
    def _segment_bytes_batch(
        self,
        byte_arrays: Sequence[np.ndarray],
        chunk: int | None = None,
    ) -> tuple[List[np.ndarray], List[np.ndarray], List[List[Tuple[int, int]]]]:
        ...


@dataclass(frozen=True)
class MonitorSource:
    split_name: str
    root: Path
    data: Dict[str, Any]


@dataclass
class AffectedMonitorFile:
    source_split: str
    source_root: str
    file_idx: int
    file_type_id: int
    file_type: str
    byte_len: int
    sample_hash: str
    text: str
    truth_ids: np.ndarray
    pred_ids: np.ndarray
    diff_count: int
    valid_count: int
    diff_ratio: float
    first_diff_index: int


@dataclass
class MonitorRelabelChunk:
    affected_file: AffectedMonitorFile
    chunk_index: int
    chunk_count: int
    start: int
    end: int
    diff_count: int
    valid_count: int
    diff_ratio: float
    first_diff_global: int
    predicted_segments: List[Dict[str, object]]
    snippet: BoundarySnippet


@dataclass(frozen=True)
class PreparedRelabelFile:
    affected_file: AffectedMonitorFile
    chunks: List[MonitorRelabelChunk]


@dataclass(frozen=True)
class ScanSummary:
    total_files_scanned: int
    affected_files_total: int
    affected_files_by_split: Dict[str, int]
    skipped_existing_files: int
    skipped_encoding_files: int
    queued_files: List[AffectedMonitorFile]


class MonitorRelabelProgress:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.path))
        con.row_factory = sqlite3.Row
        return con

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_files (
                  sample_hash TEXT PRIMARY KEY,
                  source_split TEXT NOT NULL,
                  file_idx INTEGER NOT NULL,
                  file_type TEXT NOT NULL,
                  status TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  chunk_count INTEGER NOT NULL,
                  row_count INTEGER NOT NULL,
                  updated_at TEXT NOT NULL,
                  metadata_json TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processed_files_status
                ON processed_files(status)
                """
            )

    def processed_hashes(self) -> set[str]:
        with self._connect() as con:
            rows = con.execute("SELECT sample_hash FROM processed_files").fetchall()
        return {str(row["sample_hash"]) for row in rows if row["sample_hash"]}

    def record(
        self,
        *,
        affected_file: AffectedMonitorFile,
        status: str,
        reason: str,
        chunk_count: int,
        row_count: int,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        payload = dict(metadata or {})
        payload.setdefault("source_split", affected_file.source_split)
        payload.setdefault("file_idx", int(affected_file.file_idx))
        payload.setdefault("file_type", affected_file.file_type)
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO processed_files(
                  sample_hash, source_split, file_idx, file_type,
                  status, reason, chunk_count, row_count, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_hash) DO UPDATE SET
                  source_split=excluded.source_split,
                  file_idx=excluded.file_idx,
                  file_type=excluded.file_type,
                  status=excluded.status,
                  reason=excluded.reason,
                  chunk_count=excluded.chunk_count,
                  row_count=excluded.row_count,
                  updated_at=excluded.updated_at,
                  metadata_json=excluded.metadata_json
                """,
                (
                    str(affected_file.sample_hash),
                    str(affected_file.source_split),
                    int(affected_file.file_idx),
                    str(affected_file.file_type),
                    str(status),
                    str(reason),
                    int(chunk_count),
                    int(row_count),
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _label_name(label_id: int) -> str:
    lid = int(label_id)
    other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
    if other_idx is not None and lid == int(other_idx):
        return "other"
    if lid == int(cfg.PAD_ID):
        return "pad"
    return str(cfg.ID2LANG.get(lid, f"id_{lid}"))


def _segments_from_label_names(labels: Sequence[str]) -> List[Dict[str, object]]:
    if not labels:
        return []
    out: List[Dict[str, object]] = []
    start = 0
    current = str(labels[0])
    for idx in range(1, len(labels)):
        nxt = str(labels[idx])
        if nxt == current:
            continue
        out.append({"start": int(start), "end": int(idx), "label": current})
        start = idx
        current = nxt
    out.append({"start": int(start), "end": int(len(labels)), "label": current})
    return out


def _bytes_to_ascii_text(byte_values: Sequence[int]) -> str:
    chars: List[str] = []
    for raw in byte_values:
        value = int(raw)
        if value == 0x0D:
            chars.append("\n")
            continue
        if value in VISIBLE_BYTE_SET or value in (0x09, 0x0A):
            chars.append(chr(value))
        else:
            chars.append("?")
    return "".join(chars)


def _hash_monitor_file(split_name: str, file_idx: int, raw_bytes: np.ndarray) -> str:
    h = hashlib.blake2s(digest_size=16)
    h.update(str(split_name).encode("utf-8", "ignore"))
    h.update(b":")
    h.update(str(int(file_idx)).encode("utf-8", "ignore"))
    h.update(b":")
    h.update(np.asarray(raw_bytes, dtype=np.uint8).tobytes())
    return h.hexdigest()


def _make_snippet_id(sample_hash: str, start: int, end: int, boundary: int) -> str:
    raw = f"{sample_hash}:{int(start)}:{int(end)}:{int(boundary)}"
    return hashlib.blake2s(raw.encode("utf-8", "ignore"), digest_size=8).hexdigest()


def _build_truth_labels(
    row: np.void,
    segments: np.ndarray,
) -> np.ndarray:
    byte_len = int(row["byte_len"])
    other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
    fill_value = int(other_idx) if other_idx is not None else int(cfg.PAD_ID)
    truth = np.full((byte_len,), fill_value, dtype=np.int32)
    seg_slice = segments[int(row["seg_start"]) : int(row["seg_start"]) + int(row["seg_count"])]
    for seg in seg_slice:
        start = max(0, min(byte_len, int(seg["start"])))
        end = max(start, min(byte_len, int(seg["end"])))
        if end <= start:
            continue
        truth[start:end] = int(seg["label"])
    return truth


def _build_predictor(args: argparse.Namespace) -> Predictor:
    ckpt_path = Path(args.checkpoint).resolve()
    auto_hparams = _load_checkpoint_hparams(ckpt_path)
    label_names = auto_hparams.get("label_names")
    if label_names:
        _apply_label_mapping(label_names)  # type: ignore[arg-type]

    inferred = _infer_checkpoint_architecture(ckpt_path)
    arch, _ = _resolve_hparam(
        args.arch,
        auto_hparams.get("arch"),
        inferred.get("arch"),
        "unet1d",
    )
    model_dim, _ = _resolve_hparam(
        args.model_dim,
        auto_hparams.get("model_dim"),
        inferred.get("model_dim"),
        256,
    )
    channels, _ = _resolve_hparam(
        None,
        auto_hparams.get("channels"),
        inferred.get("channels"),
        (96, 128, 192, 256),
    )
    mamba_layers, _ = _resolve_hparam(
        args.mamba_layers,
        auto_hparams.get("mamba_layers"),
        inferred.get("mamba_layers"),
        6,
    )
    mamba_d_state, _ = _resolve_hparam(
        args.mamba_d_state,
        auto_hparams.get("mamba_d_state"),
        inferred.get("mamba_d_state"),
        8,
    )
    mamba_expand, _ = _resolve_hparam(
        args.mamba_expand,
        auto_hparams.get("mamba_expand"),
        inferred.get("mamba_expand"),
        1,
    )
    mamba_dt_rank, _ = _resolve_hparam(
        args.mamba_dt_rank,
        auto_hparams.get("mamba_dt_rank"),
        inferred.get("mamba_dt_rank"),
        16,
    )
    mamba_conv, _ = _resolve_hparam(
        args.mamba_conv,
        auto_hparams.get("mamba_conv"),
        inferred.get("mamba_conv"),
        4,
    )
    mamba_bidirectional, _ = _resolve_hparam(
        args.mamba_bidirectional,
        auto_hparams.get("mamba_bidirectional"),
        inferred.get("mamba_bidirectional"),
        True,
    )
    dtype_value = str(
        args.dtype
        or auto_hparams.get("dtype")
        or inferred.get("dtype")
        or "bfloat16"
    ).rsplit(".", 1)[-1]

    return Predictor(
        ckpt_path=str(ckpt_path),
        num_classes=int(cfg.NUM_CLASSES),
        model_dim=int(model_dim),
        channels=tuple(int(x) for x in channels),
        arch=str(arch).strip().lower(),
        mamba_layers=int(mamba_layers),
        mamba_d_state=int(mamba_d_state),
        mamba_expand=int(mamba_expand),
        mamba_dt_rank=int(mamba_dt_rank),
        mamba_conv=int(mamba_conv),
        mamba_bidirectional=bool(mamba_bidirectional),
        dtype_str=dtype_value,
        chunk=int(args.chunk),
        other_threshold=float(args.other_threshold),
        inference_batch_size=int(args.predict_batch_size),
    )


def _predict_label_ids(
    predictor: ByteBatchPredictor,
    raw_files: Sequence[np.ndarray],
    *,
    other_threshold: float,
) -> List[np.ndarray]:
    byte_labels, prob_by_text, _ = predictor._segment_bytes_batch(raw_files)
    out: List[np.ndarray] = []
    other_id = int(getattr(cfg, "OTHER_CLASS_INDEX", cfg.NUM_CLASSES))
    for labels, probs in zip(byte_labels, prob_by_text):
        pred = np.asarray(labels, dtype=np.int32)
        if other_threshold > 0.0 and np.asarray(probs).size > 0:
            pred = Predictor.threshold_predictions(
                np.asarray(probs, dtype=np.float32),
                other_threshold=float(other_threshold),
                other_id=other_id,
            ).astype(np.int32)
        out.append(pred)
    return out


def _load_monitor_sources(split_names: Sequence[str]) -> List[MonitorSource]:
    sources: List[MonitorSource] = []
    for split_name in split_names:
        root = DEFAULT_MONITOR_ROOTS.get(split_name)
        if root is None:
            raise ValueError(f"Unknown monitor split '{split_name}'. Expected one of: {sorted(DEFAULT_MONITOR_ROOTS)}")
        sources.append(
            MonitorSource(
                split_name=str(split_name),
                root=root,
                data=load_monitor_memmaps(root),
            )
        )
    return sources


def scan_monitor_sources(
    *,
    sources: Sequence[MonitorSource],
    predictor: ByteBatchPredictor,
    min_diff_chars: int,
    other_threshold: float,
    allow_repeat_hashes: bool,
    existing_hashes: Optional[set[str]] = None,
    limit_files_per_split: Optional[int] = None,
    progress_every: int = 256,
    batch_size: int = 8,
) -> ScanSummary:
    existing = existing_hashes or set()
    queued_files: List[AffectedMonitorFile] = []
    affected_by_split: Counter[str] = Counter()
    total_files_scanned = 0
    affected_files_total = 0
    skipped_existing = 0
    skipped_encoding = 0

    for source in sources:
        files = source.data["files"]
        segments = source.data["segments"]
        contents = source.data["contents"]
        total_in_split = len(files)
        limit = total_in_split
        if limit_files_per_split is not None and limit_files_per_split > 0:
            limit = min(limit, int(limit_files_per_split))

        print(
            f"[monitor_relabel] scanning {source.split_name}: files={limit}/{total_in_split}",
            flush=True,
        )

        for batch_start in range(0, limit, max(1, int(batch_size))):
            batch_stop = min(limit, batch_start + max(1, int(batch_size)))
            raw_batch: List[np.ndarray] = []
            truth_batch: List[np.ndarray] = []
            meta_batch: List[Tuple[int, np.void]] = []

            for file_idx in range(batch_start, batch_stop):
                row = files[file_idx]
                file_type = _label_name(int(row["type_id"]))
                if str(file_type).startswith("encoding_"):
                    skipped_encoding += 1
                    continue
                byte_len = int(row["byte_len"])
                byte_start = int(row["byte_start"])
                raw = np.asarray(contents[byte_start : byte_start + byte_len], dtype=np.uint8).copy()
                truth = _build_truth_labels(row, segments)
                raw_batch.append(raw)
                truth_batch.append(truth)
                meta_batch.append((int(file_idx), row))

            if not raw_batch:
                if progress_every > 0 and (batch_stop >= limit or batch_stop % progress_every == 0):
                    print(
                        f"[monitor_relabel] {source.split_name}: scanned={batch_stop}/{limit}, "
                        f"affected={affected_by_split[source.split_name]}, queued={len(queued_files)}",
                        flush=True,
                    )
                continue

            pred_batch = _predict_label_ids(
                predictor,
                raw_batch,
                other_threshold=float(other_threshold),
            )

            for raw, truth, pred, (file_idx, row) in zip(raw_batch, truth_batch, pred_batch, meta_batch):
                total_files_scanned += 1
                mask = _valid_metric_mask(truth.reshape(1, -1), raw.astype(np.int32, copy=False).reshape(1, -1))[0]
                valid_count = int(mask.sum())
                if valid_count <= 0:
                    continue
                diff_mask = (pred[: truth.shape[0]] != truth) & mask
                diff_count = int(diff_mask.sum())
                if diff_count <= int(min_diff_chars):
                    continue

                affected_files_total += 1
                affected_by_split[source.split_name] += 1
                sample_hash = _hash_monitor_file(source.split_name, file_idx, raw)
                if not allow_repeat_hashes and sample_hash in existing:
                    skipped_existing += 1
                    continue

                diff_positions = np.flatnonzero(diff_mask)
                first_diff = int(diff_positions[0]) if diff_positions.size > 0 else -1
                text = _bytes_to_ascii_text(raw.tolist())
                queued_files.append(
                    AffectedMonitorFile(
                        source_split=str(source.split_name),
                        source_root=str(source.root),
                        file_idx=int(file_idx),
                        file_type_id=int(row["type_id"]),
                        file_type=_label_name(int(row["type_id"])),
                        byte_len=int(row["byte_len"]),
                        sample_hash=sample_hash,
                        text=text,
                        truth_ids=truth.copy(),
                        pred_ids=np.asarray(pred, dtype=np.int32).copy(),
                        diff_count=diff_count,
                        valid_count=valid_count,
                        diff_ratio=float(diff_count) / float(valid_count),
                        first_diff_index=first_diff,
                    )
                )

            if progress_every > 0 and (batch_stop >= limit or batch_stop % progress_every == 0):
                print(
                    f"[monitor_relabel] {source.split_name}: scanned={batch_stop}/{limit}, "
                    f"affected={affected_by_split[source.split_name]}, queued={len(queued_files)}",
                    flush=True,
                )

    return ScanSummary(
        total_files_scanned=int(total_files_scanned),
        affected_files_total=int(affected_files_total),
        affected_files_by_split=dict(affected_by_split),
        skipped_existing_files=int(skipped_existing),
        skipped_encoding_files=int(skipped_encoding),
        queued_files=queued_files,
    )


def _chunk_bounds(text: str, max_chars: int) -> List[Tuple[int, int]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    n = len(text)
    if n <= 0:
        return []
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + int(max_chars))
        if end < n:
            lower = min(n, start + max(1, int(max_chars) // 2))
            newline_cut = text.rfind("\n", lower, end)
            if newline_cut > start:
                end = newline_cut + 1
        if end <= start:
            end = min(n, start + int(max_chars))
        bounds.append((int(start), int(end)))
        start = end
    return bounds


def build_relabel_chunks(
    files: Sequence[AffectedMonitorFile],
    *,
    oracle_max_chars: int,
) -> List[MonitorRelabelChunk]:
    chunks: List[MonitorRelabelChunk] = []
    for affected in files:
        bounds = _chunk_bounds(affected.text, int(oracle_max_chars))
        chunk_count = len(bounds)
        pred_names = [_label_name(int(label_id)) for label_id in affected.pred_ids.tolist()]
        for chunk_index, (start, end) in enumerate(bounds):
            truth_slice = affected.truth_ids[start:end]
            pred_slice = affected.pred_ids[start:end]
            valid_count = int(end - start)
            diff_mask = pred_slice != truth_slice
            diff_count = int(diff_mask.sum())
            diff_positions = np.flatnonzero(diff_mask)
            first_diff_global = int(start + diff_positions[0]) if diff_positions.size > 0 else -1
            boundary_rel = int(diff_positions[0]) if diff_positions.size > 0 else 0
            predicted_labels = pred_names[start:end]
            snippet = BoundarySnippet(
                snippet_id=_make_snippet_id(
                    affected.sample_hash,
                    int(start),
                    int(end),
                    int(first_diff_global),
                ),
                text=affected.text[start:end],
                global_start=int(start),
                global_end=int(end),
                boundary=int(boundary_rel),
                predicted_labels=list(predicted_labels),
                metadata={
                    "source_lang": affected.file_type,
                    "sample_index": int(affected.file_idx),
                    "sample_hash": affected.sample_hash,
                    "boundary_global": int(first_diff_global),
                    "monitor_split": affected.source_split,
                    "monitor_root": affected.source_root,
                    "monitor_file_idx": int(affected.file_idx),
                    "monitor_file_type_id": int(affected.file_type_id),
                    "monitor_file_type": affected.file_type,
                    "monitor_file_diff_count": int(affected.diff_count),
                    "monitor_file_valid_count": int(affected.valid_count),
                    "monitor_file_diff_ratio": float(affected.diff_ratio),
                    "oracle_chunk_index": int(chunk_index),
                    "oracle_chunk_count": int(chunk_count),
                    "oracle_chunk_diff_count": int(diff_count),
                    "oracle_chunk_valid_count": int(valid_count),
                    "oracle_chunk_diff_ratio": float(diff_count) / float(max(1, valid_count)),
                    "relabel_kind": "monitor_file_relabel",
                },
            )
            chunks.append(
                MonitorRelabelChunk(
                    affected_file=affected,
                    chunk_index=int(chunk_index),
                    chunk_count=int(chunk_count),
                    start=int(start),
                    end=int(end),
                    diff_count=int(diff_count),
                    valid_count=int(valid_count),
                    diff_ratio=float(diff_count) / float(max(1, valid_count)),
                    first_diff_global=int(first_diff_global),
                    predicted_segments=_segments_from_label_names(predicted_labels),
                    snippet=snippet,
                )
            )
    return chunks


def prepare_relabel_files(
    files: Sequence[AffectedMonitorFile],
    *,
    oracle_max_chars: int,
) -> List[PreparedRelabelFile]:
    prepared: List[PreparedRelabelFile] = []
    for affected in files:
        prepared.append(
            PreparedRelabelFile(
                affected_file=affected,
                chunks=build_relabel_chunks([affected], oracle_max_chars=int(oracle_max_chars)),
            )
        )
    return prepared


def group_relabel_requests(
    prepared_files: Sequence[PreparedRelabelFile],
    *,
    max_snippets_per_request: int,
) -> List[List[PreparedRelabelFile]]:
    groups: List[List[PreparedRelabelFile]] = []
    current: List[PreparedRelabelFile] = []
    current_snippets = 0
    limit = max(1, int(max_snippets_per_request))

    for prepared in prepared_files:
        file_snippets = max(1, len(prepared.chunks))
        if current and current_snippets + file_snippets > limit:
            groups.append(current)
            current = []
            current_snippets = 0
        current.append(prepared)
        current_snippets += file_snippets
        if current_snippets >= limit:
            groups.append(current)
            current = []
            current_snippets = 0

    if current:
        groups.append(current)
    return groups


def _file_ignore_reason(
    *,
    snippets: Sequence[BoundarySnippet],
    refined: Dict[str, List[OracleSegment]],
    snippet_sources: Dict[str, str],
) -> Optional[Tuple[str, Dict[str, object]]]:
    missing_ids = [
        str(snippet.snippet_id)
        for snippet in snippets
        if str(snippet.snippet_id) not in refined and str(snippet.snippet_id) not in snippet_sources
    ]
    parse_failed_ids = [
        str(snippet.snippet_id)
        for snippet in snippets
        if str(snippet_sources.get(snippet.snippet_id, "")) == "skipped_parse_failed"
    ]
    unresolved_missing_ids = [
        str(snippet.snippet_id)
        for snippet in snippets
        if str(snippet_sources.get(snippet.snippet_id, "")) == "skipped_missing"
    ]
    unexpected_ids = [
        (str(snippet.snippet_id), str(snippet_sources.get(snippet.snippet_id, "")))
        for snippet in snippets
        if str(snippet.snippet_id) in snippet_sources
        and str(snippet_sources.get(snippet.snippet_id, "")) not in {"model", "skipped_parse_failed", "skipped_missing"}
    ]

    if parse_failed_ids:
        return (
            "parse_failed",
            {
                "parse_failed_snippet_ids": list(parse_failed_ids),
                "missing_snippet_ids": list(unresolved_missing_ids),
            },
        )
    if unresolved_missing_ids or missing_ids:
        return (
            "missing",
            {
                "missing_snippet_ids": list(unresolved_missing_ids) + list(missing_ids),
            },
        )
    if unexpected_ids:
        return (
            "unexpected_non_model",
            {
                "unexpected_snippets": [
                    {"snippet_id": sid, "state": state}
                    for sid, state in unexpected_ids
                ],
            },
        )
    for snippet in snippets:
        if str(snippet.snippet_id) not in refined:
            return (
                "missing",
                {"missing_snippet_ids": [str(snippet.snippet_id)]},
            )
    return None


def build_store_rows(
    *,
    chunks: Sequence[MonitorRelabelChunk],
    refined: Dict[str, List[OracleSegment]],
    round_id: str,
    oracle_name: str,
    oracle_model: str,
) -> List[StoredRefinement]:
    rows: List[StoredRefinement] = []
    for chunk in chunks:
        snippet = chunk.snippet
        refined_segments = []
        open_set_labels = []
        for seg in refined.get(snippet.snippet_id, []):
            segment_row: Dict[str, object] = {
                "start": int(seg.start),
                "end": int(seg.end),
                "label": str(seg.label),
            }
            raw_label = getattr(seg, "raw_label", None)
            if isinstance(raw_label, str) and raw_label:
                segment_row["open_set_label"] = raw_label
                open_set_labels.append(raw_label)
            refined_segments.append(segment_row)

        metadata = dict(snippet.metadata)
        metadata["monitor_source_split"] = chunk.affected_file.source_split
        metadata["monitor_source_root"] = chunk.affected_file.source_root
        metadata["monitor_write_round_id"] = str(round_id)
        if open_set_labels:
            metadata["oracle_open_set_labels"] = sorted(set(open_set_labels))
            metadata["oracle_open_set_segments"] = int(len(open_set_labels))

        rows.append(
            StoredRefinement(
                round_id=str(round_id),
                source_split=str(chunk.affected_file.source_split),
                source_lang=str(chunk.affected_file.file_type),
                sample_index=int(chunk.affected_file.file_idx),
                sample_hash=str(chunk.affected_file.sample_hash),
                boundary_index=int(chunk.first_diff_global),
                snippet_start=int(chunk.start),
                snippet_end=int(chunk.end),
                snippet_text=str(snippet.text),
                oracle_name=str(oracle_name),
                oracle_model=str(oracle_model),
                oracle_run_id="",
                status="ok",
                acquisition_score=float(chunk.diff_ratio),
                predicted_segments=list(chunk.predicted_segments),
                refined_segments=refined_segments,
                metadata=metadata,
            )
        )
    return rows


def _build_oracle(args: argparse.Namespace) -> object:
    if args.oracle == "stub":
        return StubOracle()
    requested_retries = int(getattr(args, "gemini_missing_snippet_retries", 0))
    if requested_retries != 0:
        print(
            "[monitor_relabel] forcing gemini missing-snippet retries to 0 so each file is attempted only once.",
            flush=True,
        )
    return GeminiBoundaryOracle(
        model=args.gemini_model,
        thinking_level=args.gemini_thinking_level,
        api_key=args.api_key,
        batch_size=int(args.gemini_batch_size),
        proxy=args.proxy,
        rate_limit_sleep_seconds=float(args.gemini_rate_limit_sleep_seconds),
        rate_limit_max_retries=int(args.gemini_rate_limit_max_retries),
        missing_snippet_retries=0,
        show_progress=bool(args.show_progress),
        progress_desc="monitor relabel oracle",
        progress_leave=False,
    )


def _print_scan_summary(
    summary: ScanSummary,
    *,
    oracle_chunk_count: int,
    oracle_request_count: int,
    store_path: Path,
) -> None:
    per_split = ", ".join(
        f"{split}:{count}"
        for split, count in sorted(summary.affected_files_by_split.items())
    ) or "none"
    print(
        "[monitor_relabel] summary: "
        f"scanned_files={summary.total_files_scanned}, "
        f"affected_files={summary.affected_files_total}, "
        f"queued_files={len(summary.queued_files)}, "
        f"skipped_already_processed={summary.skipped_existing_files}, "
        f"skipped_encoding={summary.skipped_encoding_files}, "
        f"oracle_chunks={int(oracle_chunk_count)}, "
        f"oracle_requests_est={int(oracle_request_count)}, "
        f"affected_by_split={per_split}, "
        f"store={store_path}",
        flush=True,
    )


def _confirm_continue(
    *,
    chunks: Sequence[MonitorRelabelChunk],
    include_monitor_b: bool,
    allow_monitor_b_training: bool,
) -> bool:
    if not chunks:
        return False
    if include_monitor_b and not allow_monitor_b_training:
        print(
            "[monitor_relabel] WARNING: monitor_b is usually evaluation-only. "
            "Writing those rows into a training store can contaminate monitor metrics.",
            flush=True,
        )
        try:
            answer = input("Type 'train monitor_b' to continue, anything else aborts: ").strip()
        except EOFError:
            return False
        return answer == "train monitor_b"
    try:
        answer = input("Continue with oracle relabel + DB write? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def run_monitor_relabel(args: argparse.Namespace) -> Dict[str, object]:
    store = LabelStore(args.store)
    progress_db = MonitorRelabelProgress(args.progress_db)
    if args.allow_repeat_hashes:
        existing_hashes: set[str] = set()
    else:
        existing_hashes = store.existing_sample_hashes() | progress_db.processed_hashes()
    predictor = _build_predictor(args)
    oracle = _build_oracle(args)
    sources = _load_monitor_sources(args.monitor_splits)

    summary = scan_monitor_sources(
        sources=sources,
        predictor=predictor,
        min_diff_chars=int(args.min_diff_chars),
        other_threshold=float(args.other_threshold),
        allow_repeat_hashes=bool(args.allow_repeat_hashes),
        existing_hashes=existing_hashes,
        limit_files_per_split=int(args.limit_files_per_split) if args.limit_files_per_split > 0 else None,
        progress_every=int(args.progress_every),
        batch_size=int(args.scan_batch_size),
    )
    prepared_files = prepare_relabel_files(
        summary.queued_files,
        oracle_max_chars=int(args.oracle_max_chars),
    )
    request_groups = group_relabel_requests(
        prepared_files,
        max_snippets_per_request=int(getattr(oracle, "batch_size", 1)),
    )
    chunks = [chunk for prepared in prepared_files for chunk in prepared.chunks]
    _print_scan_summary(
        summary,
        oracle_chunk_count=len(chunks),
        oracle_request_count=len(request_groups),
        store_path=Path(args.store).resolve(),
    )

    if not chunks:
        print("[monitor_relabel] nothing to write.", flush=True)
        return {
            "status": "no_op",
            "scanned_files": int(summary.total_files_scanned),
            "affected_files": int(summary.affected_files_total),
            "queued_files": int(len(summary.queued_files)),
            "written_rows": 0,
        }

    if args.dry_run:
        print("[monitor_relabel] dry-run enabled; skipping oracle + DB write.", flush=True)
        return {
            "status": "dry_run",
            "scanned_files": int(summary.total_files_scanned),
            "affected_files": int(summary.affected_files_total),
            "queued_files": int(len(summary.queued_files)),
            "oracle_chunks": int(len(chunks)),
            "written_rows": 0,
        }

    include_monitor_b = any(file.source_split == "monitor_b" for file in summary.queued_files)
    if args.yes and include_monitor_b and not bool(args.allow_monitor_b_training):
        raise RuntimeError(
            "--yes with monitor_b selected requires --allow-monitor-b-training, "
            "because those rows can contaminate monitor evaluation if reused for training."
        )

    if not args.yes:
        if not _confirm_continue(
            chunks=chunks,
            include_monitor_b=include_monitor_b,
            allow_monitor_b_training=bool(args.allow_monitor_b_training),
        ):
            print("[monitor_relabel] aborted by user.", flush=True)
            return {
                "status": "aborted",
                "scanned_files": int(summary.total_files_scanned),
                "affected_files": int(summary.affected_files_total),
                "queued_files": int(len(summary.queued_files)),
                "oracle_chunks": int(len(chunks)),
                "written_rows": 0,
            }

    round_id = str(args.round_id or f"monitor-relabel-{_utc_stamp()}")
    written_files = 0
    written_rows = 0
    ignored_files = 0
    ignored_by_reason: Counter[str] = Counter()
    ignored_by_split: Counter[str] = Counter()
    ignored_by_split_reason: Counter[Tuple[str, str]] = Counter()
    file_num = 0

    for request_num, request_group in enumerate(request_groups, start=1):
        group_chunks = [chunk for prepared in request_group for chunk in prepared.chunks]
        snippets = [chunk.snippet for chunk in group_chunks]
        refined = oracle.annotate(snippets)  # type: ignore[attr-defined]
        snippet_sources = getattr(oracle, "last_snippet_sources", {})
        if not isinstance(snippet_sources, dict):
            snippet_sources = {}
        for prepared in request_group:
            file_num += 1
            affected_file = prepared.affected_file
            file_chunks = prepared.chunks
            file_snippets = [chunk.snippet for chunk in file_chunks]
            ignore_info = _file_ignore_reason(
                snippets=file_snippets,
                refined=refined,
                snippet_sources=snippet_sources,
            )
            if ignore_info is not None:
                reason, details = ignore_info
                ignored_files += 1
                ignored_by_reason[reason] += 1
                ignored_by_split[affected_file.source_split] += 1
                ignored_by_split_reason[(affected_file.source_split, reason)] += 1
                progress_db.record(
                    affected_file=affected_file,
                    status="ignored",
                    reason=str(reason),
                    chunk_count=len(file_chunks),
                    row_count=0,
                    metadata={
                        "round_id": round_id,
                        "request_batch_index": int(request_num),
                        "details": details,
                        "snippet_sources": {
                            str(key): str(value) for key, value in sorted(snippet_sources.items())
                        },
                    },
                )
                print(
                    f"[monitor_relabel] ignored file {file_num}/{len(summary.queued_files)} "
                    f"{affected_file.source_split}:{affected_file.file_idx} "
                    f"(reason={reason}, chunks={len(file_chunks)}, request_batch={request_num}/{len(request_groups)})",
                    flush=True,
                )
                continue

            rows = build_store_rows(
                chunks=file_chunks,
                refined=refined,
                round_id=round_id,
                oracle_name=str(getattr(oracle, "name", "oracle")),
                oracle_model=str(getattr(oracle, "model", "")),
            )
            inserted = store.add_many(rows)
            written_files += 1
            written_rows += int(inserted)
            progress_db.record(
                affected_file=affected_file,
                status="written",
                reason="written",
                chunk_count=len(file_chunks),
                row_count=int(inserted),
                metadata={
                    "round_id": round_id,
                    "request_batch_index": int(request_num),
                },
            )
            print(
                f"[monitor_relabel] wrote file {file_num}/{len(summary.queued_files)} "
                f"{affected_file.source_split}:{affected_file.file_idx} "
                f"(rows={inserted}, chunks={len(file_chunks)}, request_batch={request_num}/{len(request_groups)})",
                flush=True,
            )

    if ignored_files > 0:
        reason_summary = ", ".join(
            f"{reason}:{count}" for reason, count in sorted(ignored_by_reason.items())
        )
        split_summary = ", ".join(
            f"{split}:{count}" for split, count in sorted(ignored_by_split.items())
        )
        split_reason_summary = ", ".join(
            f"{split}/{reason}:{count}"
            for (split, reason), count in sorted(ignored_by_split_reason.items())
        )
        print(
            "[monitor_relabel] ignored summary: "
            f"files={ignored_files}, "
            f"by_reason={reason_summary or 'none'}, "
            f"by_split={split_summary or 'none'}, "
            f"by_split_reason={split_reason_summary or 'none'}",
            flush=True,
        )

    print(
        f"[monitor_relabel] wrote {written_rows} refinements across {written_files} files to {store.path} "
        f"(round_id={round_id}).",
        flush=True,
    )
    return {
        "status": "ok",
        "round_id": round_id,
        "scanned_files": int(summary.total_files_scanned),
        "affected_files": int(summary.affected_files_total),
        "queued_files": int(len(summary.queued_files)),
        "oracle_chunks": int(len(chunks)),
        "written_files": int(written_files),
        "written_rows": int(written_rows),
        "ignored_files": int(ignored_files),
        "ignored_by_reason": dict(ignored_by_reason),
        "ignored_by_split": dict(ignored_by_split),
        "ignored_by_split_reason": {
            f"{split}/{reason}": int(count)
            for (split, reason), count in sorted(ignored_by_split_reason.items())
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan monitor_a/monitor_b with a checkpoint, then relabel mismatching files with the AL oracle."
    )
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint (.msgpack or Orbax dir).")
    parser.add_argument(
        "--monitor-splits",
        type=str,
        default="monitor_a,monitor_b",
        help="Comma-separated monitor splits to scan (monitor_a,monitor_b).",
    )
    parser.add_argument("--store", type=str, default=str(DEFAULT_STORE_PATH))
    parser.add_argument("--progress-db", type=str, default=str(DEFAULT_PROGRESS_DB_PATH))
    parser.add_argument("--round-id", type=str, default=None)
    parser.add_argument("--min-diff-chars", type=int, default=4, help="Queue files with strictly more than this many differing chars.")
    parser.add_argument("--oracle-max-chars", type=int, default=4000, help="Max chars per oracle snippet chunk.")
    parser.add_argument("--limit-files-per-split", type=int, default=0, help="Optional safety limit for smoke tests.")
    parser.add_argument("--scan-batch-size", type=int, default=8, help="Files per scan batch passed to GPU inference.")
    parser.add_argument("--progress-every", type=int, default=256, help="Print scan progress every N files per split.")
    parser.add_argument("--allow-repeat-hashes", action="store_true", help="Do not skip files already present in the label store.")
    parser.add_argument("--allow-monitor-b-training", action="store_true", help="Acknowledge that monitor_b rows may contaminate eval if reused for training.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and plan oracle chunks without writing.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")

    parser.add_argument("--oracle", choices=("stub", "gemini"), default="gemini")
    parser.add_argument("--gemini-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument(
        "--gemini-thinking-level",
        type=str,
        choices=("minimal", "low", "medium", "high"),
        default="medium",
    )
    parser.add_argument(
        "--gemini-batch-size",
        type=int,
        default=8,
        help="Max oracle snippets per Gemini request. The relabel script batches files together up to this snippet budget.",
    )
    parser.add_argument("--gemini-rate-limit-sleep-seconds", type=float, default=65.0)
    parser.add_argument("--gemini-rate-limit-max-retries", type=int, default=8)
    parser.add_argument(
        "--gemini-missing-snippet-retries",
        type=int,
        default=0,
        help="Ignored for this script; retries are forced to 0 so each file is attempted only once.",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--proxy", type=str, default=None)
    parser.add_argument("--show-progress", action="store_true")

    parser.add_argument("--arch", type=str, default=None, choices=("unet1d", "mamba"))
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--mamba-layers", type=int, default=None)
    parser.add_argument("--mamba-d-state", type=int, default=None)
    parser.add_argument("--mamba-expand", type=int, default=None)
    parser.add_argument("--mamba-dt-rank", type=int, default=None)
    parser.add_argument("--mamba-conv", type=int, default=None)
    parser.add_argument("--mamba-bidirectional", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--other-threshold", type=float, default=0.5)
    parser.add_argument("--predict-batch-size", type=int, default=12)
    return parser


def _parse_monitor_splits(raw: str) -> List[str]:
    parts = [piece.strip().lower() for piece in str(raw or "").split(",") if piece.strip()]
    if not parts:
        return ["monitor_a", "monitor_b"]
    for part in parts:
        if part not in DEFAULT_MONITOR_ROOTS:
            raise ValueError(f"Unknown monitor split '{part}'.")
    return parts


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.monitor_splits = _parse_monitor_splits(args.monitor_splits)
    summary = run_monitor_relabel(args)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
