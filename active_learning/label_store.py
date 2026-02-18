from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class StoredRefinement:
    round_id: str
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
    status: str
    acquisition_score: float
    predicted_segments: List[Dict[str, object]]
    refined_segments: List[Dict[str, object]]
    metadata: Dict[str, object]


@dataclass(frozen=True)
class StoredInferenceSample:
    round_id: str
    source_split: str
    source_lang: str
    sample_index: int
    sample_hash: str
    sample_text: str
    char_count: int
    queried_for_oracle: bool
    candidate_count: int
    trigger_ranges: List[Dict[str, object]]
    predicted_segments: List[Dict[str, object]]
    metadata: Dict[str, object]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LabelStore:
    """SQLite-backed persistence for active-learning oracle outcomes."""

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
                CREATE TABLE IF NOT EXISTS refinements (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TEXT NOT NULL,
                  round_id TEXT NOT NULL,
                  source_split TEXT NOT NULL,
                  source_lang TEXT NOT NULL,
                  sample_index INTEGER NOT NULL,
                  sample_hash TEXT NOT NULL,
                  boundary_index INTEGER NOT NULL,
                  snippet_start INTEGER NOT NULL,
                  snippet_end INTEGER NOT NULL,
                  snippet_text TEXT NOT NULL,
                  oracle_name TEXT NOT NULL,
                  oracle_model TEXT NOT NULL,
                  oracle_run_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  acquisition_score REAL NOT NULL,
                  predicted_segments_json TEXT NOT NULL,
                  refined_segments_json TEXT NOT NULL,
                  metadata_json TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_refinements_round
                ON refinements(round_id)
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_refinements_sample
                ON refinements(sample_hash, snippet_start, snippet_end)
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_samples (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  created_at TEXT NOT NULL,
                  round_id TEXT NOT NULL,
                  source_split TEXT NOT NULL,
                  source_lang TEXT NOT NULL,
                  sample_index INTEGER NOT NULL,
                  sample_hash TEXT NOT NULL,
                  sample_text TEXT NOT NULL,
                  char_count INTEGER NOT NULL,
                  queried_for_oracle INTEGER NOT NULL,
                  candidate_count INTEGER NOT NULL,
                  trigger_ranges_json TEXT NOT NULL,
                  predicted_segments_json TEXT NOT NULL,
                  metadata_json TEXT NOT NULL
                )
                """
            )
            cols = {
                str(row["name"])
                for row in con.execute("PRAGMA table_info(inference_samples)").fetchall()
            }
            if "predicted_segments_json" not in cols:
                con.execute(
                    """
                    ALTER TABLE inference_samples
                    ADD COLUMN predicted_segments_json TEXT NOT NULL DEFAULT '[]'
                    """
                )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_inference_samples_round
                ON inference_samples(round_id)
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_inference_samples_query
                ON inference_samples(queried_for_oracle)
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_inference_samples_sample
                ON inference_samples(sample_hash, sample_index)
                """
            )

    def add(self, row: StoredRefinement) -> int:
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO refinements(
                  created_at, round_id, source_split, source_lang, sample_index, sample_hash,
                  boundary_index, snippet_start, snippet_end, snippet_text,
                  oracle_name, oracle_model, oracle_run_id, status, acquisition_score,
                  predicted_segments_json, refined_segments_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now(),
                    row.round_id,
                    row.source_split,
                    row.source_lang,
                    int(row.sample_index),
                    row.sample_hash,
                    int(row.boundary_index),
                    int(row.snippet_start),
                    int(row.snippet_end),
                    row.snippet_text,
                    row.oracle_name,
                    row.oracle_model,
                    row.oracle_run_id,
                    row.status,
                    float(row.acquisition_score),
                    json.dumps(row.predicted_segments, ensure_ascii=False),
                    json.dumps(row.refined_segments, ensure_ascii=False),
                    json.dumps(row.metadata, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def add_many(self, rows: Iterable[StoredRefinement]) -> int:
        count = 0
        with self._connect() as con:
            for row in rows:
                con.execute(
                    """
                    INSERT INTO refinements(
                      created_at, round_id, source_split, source_lang, sample_index, sample_hash,
                      boundary_index, snippet_start, snippet_end, snippet_text,
                      oracle_name, oracle_model, oracle_run_id, status, acquisition_score,
                      predicted_segments_json, refined_segments_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _utc_now(),
                        row.round_id,
                        row.source_split,
                        row.source_lang,
                        int(row.sample_index),
                        row.sample_hash,
                        int(row.boundary_index),
                        int(row.snippet_start),
                        int(row.snippet_end),
                        row.snippet_text,
                        row.oracle_name,
                        row.oracle_model,
                        row.oracle_run_id,
                        row.status,
                        float(row.acquisition_score),
                        json.dumps(row.predicted_segments, ensure_ascii=False),
                        json.dumps(row.refined_segments, ensure_ascii=False),
                        json.dumps(row.metadata, ensure_ascii=False),
                    ),
                )
                count += 1
        return count

    def add_inference_samples_many(self, rows: Iterable[StoredInferenceSample]) -> int:
        count = 0
        with self._connect() as con:
            for row in rows:
                con.execute(
                    """
                    INSERT INTO inference_samples(
                      created_at, round_id, source_split, source_lang, sample_index, sample_hash,
                      sample_text, char_count, queried_for_oracle, candidate_count,
                      trigger_ranges_json, predicted_segments_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _utc_now(),
                        row.round_id,
                        row.source_split,
                        row.source_lang,
                        int(row.sample_index),
                        row.sample_hash,
                        row.sample_text,
                        int(row.char_count),
                        1 if bool(row.queried_for_oracle) else 0,
                        int(row.candidate_count),
                        json.dumps(row.trigger_ranges, ensure_ascii=False),
                        json.dumps(row.predicted_segments, ensure_ascii=False),
                        json.dumps(row.metadata, ensure_ascii=False),
                    ),
                )
                count += 1
        return count

    def mark_inference_samples_queried(
        self,
        *,
        round_id: str,
        sample_keys: Iterable[Tuple[str, int]],
    ) -> int:
        """Mark inference rows as oracle-queried when refinements were persisted.

        The `queried_for_oracle` flag is treated as "has stored refinement slices"
        so viewer/tracking state remains consistent when a round aborts before
        writing refinements.
        """
        unique_keys = {
            (str(sample_hash), int(sample_index))
            for sample_hash, sample_index in sample_keys
            if str(sample_hash)
        }
        with self._connect() as con:
            con.execute(
                "UPDATE inference_samples SET queried_for_oracle = 0 WHERE round_id = ?",
                (str(round_id),),
            )
            if unique_keys:
                con.executemany(
                    """
                    UPDATE inference_samples
                    SET queried_for_oracle = 1
                    WHERE round_id = ? AND sample_hash = ? AND sample_index = ?
                    """,
                    [
                        (str(round_id), str(sample_hash), int(sample_index))
                        for sample_hash, sample_index in sorted(unique_keys)
                    ],
                )
            row = con.execute(
                """
                SELECT COUNT(1) AS c
                FROM inference_samples
                WHERE round_id = ? AND queried_for_oracle = 1
                """,
                (str(round_id),),
            ).fetchone()
        return int(row["c"] if row else 0)

    def count(self) -> int:
        with self._connect() as con:
            row = con.execute("SELECT COUNT(1) AS c FROM refinements").fetchone()
        return int(row["c"] if row else 0)

    def existing_sample_hashes(self) -> set[str]:
        hashes: set[str] = set()
        with self._connect() as con:
            rows = con.execute(
                "SELECT DISTINCT sample_hash FROM inference_samples WHERE sample_hash != ''"
            ).fetchall()
            hashes.update(str(row["sample_hash"]) for row in rows if row["sample_hash"])
            rows = con.execute(
                "SELECT DISTINCT sample_hash FROM refinements WHERE sample_hash != ''"
            ).fetchall()
            hashes.update(str(row["sample_hash"]) for row in rows if row["sample_hash"])
        return hashes

    def iter_rows(
        self,
        *,
        limit: Optional[int] = None,
        statuses: Optional[Sequence[str]] = ("ok",),
    ) -> Iterator[sqlite3.Row]:
        query = "SELECT * FROM refinements"
        params: List[object] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            params.extend(list(statuses))
        query += " ORDER BY id ASC"
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(int(limit))
        with self._connect() as con:
            cur = con.execute(query, params)
            rows = cur.fetchall()
        for row in rows:
            yield row

    @staticmethod
    def _char_to_byte_offsets(text: str) -> List[int]:
        offsets = [0]
        running = 0
        for ch in text:
            running += len(ch.encode("utf-8", "ignore"))
            offsets.append(running)
        return offsets

    @staticmethod
    def _segment_dicts(row: sqlite3.Row) -> List[Dict[str, object]]:
        raw = row["refined_segments_json"] or "[]"
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    def build_training_windows(
        self,
        *,
        window_bytes: int,
        pad_byte_id: int,
        pad_label_id: int,
        label_to_id: Dict[str, int],
        max_windows: Optional[int] = None,
        fallback_label: str = "other",
        statuses: Sequence[str] = ("ok",),
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert stored refinements into fixed-size byte-level windows.
        """
        windows: List[Tuple[np.ndarray, np.ndarray]] = []
        fallback_id = int(label_to_id.get(fallback_label, pad_label_id))
        for row in self.iter_rows(statuses=statuses):
            text = str(row["snippet_text"] or "")
            if not text:
                continue
            offsets = self._char_to_byte_offsets(text)
            byte_buf = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
            if byte_buf.size == 0:
                continue

            labels = np.full((byte_buf.size,), fallback_id, dtype=np.uint8)
            segments = self._segment_dicts(row)
            for seg in segments:
                try:
                    start_char = int(seg.get("start", 0))
                    end_char = int(seg.get("end", 0))
                    label_name = str(seg.get("label", fallback_label)).strip().lower()
                except Exception:
                    continue
                if end_char <= start_char:
                    continue
                if start_char < 0 or end_char > len(offsets) - 1:
                    continue
                start_byte = offsets[start_char]
                end_byte = offsets[end_char]
                label_id = int(label_to_id.get(label_name, fallback_id))
                labels[start_byte:end_byte] = label_id

            start = 0
            stride = max(1, window_bytes)
            while start < byte_buf.size:
                end = min(start + window_bytes, byte_buf.size)
                x = np.full((window_bytes,), int(pad_byte_id), dtype=np.int32)
                y = np.full((window_bytes,), int(pad_label_id), dtype=np.uint8)
                piece_len = end - start
                x[:piece_len] = byte_buf[start:end].astype(np.int32)
                y[:piece_len] = labels[start:end]
                windows.append((x, y))
                if max_windows is not None and max_windows > 0 and len(windows) >= max_windows:
                    return windows
                start += stride
        return windows
