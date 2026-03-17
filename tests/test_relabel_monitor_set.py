from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg
from active_learning.label_store import LabelStore
from active_learning.oracle import OracleSegment, StubOracle
from active_learning.relabel_monitor_set import (
    AffectedMonitorFile,
    MonitorRelabelProgress,
    MonitorSource,
    PreparedRelabelFile,
    _file_ignore_reason,
    build_relabel_chunks,
    build_store_rows,
    group_relabel_requests,
    prepare_relabel_files,
    run_monitor_relabel,
    scan_monitor_sources,
)
from train.utils.monitor_eval import FILE_DTYPE, SEG_DTYPE


class _FakePredictor:
    def __init__(self, mapping: dict[bytes, list[int]]) -> None:
        self.mapping = {bytes(key): list(value) for key, value in mapping.items()}

    def _segment_bytes_batch(self, byte_arrays, chunk=None):
        labels_out = []
        probs_out = []
        spans_out = []
        for arr in byte_arrays:
            key = bytes(np.asarray(arr, dtype=np.uint8).tolist())
            label_ids = np.asarray(self.mapping[key], dtype=np.int32)
            probs = np.zeros((len(label_ids), cfg.NUM_CLASSES), dtype=np.float32)
            probs[np.arange(len(label_ids)), label_ids] = 1.0
            labels_out.append(label_ids)
            probs_out.append(probs)
            spans_out.append([])
        return labels_out, probs_out, spans_out


def _monitor_source() -> MonitorSource:
    py_id = int(cfg.LANG2ID["python"])
    sql_id = int(cfg.LANG2ID["sql"])
    files = np.array(
        [
            (0, 6, 0, 1, 0, py_id),
            (6, 6, 1, 1, 0, sql_id),
        ],
        dtype=FILE_DTYPE,
    )
    segments = np.array(
        [
            (0, 0, 6, py_id),
            (1, 0, 6, sql_id),
        ],
        dtype=SEG_DTYPE,
    )
    contents = np.frombuffer(b"aaaaaabbbbbb", dtype=np.uint8)
    return MonitorSource(
        split_name="monitor_a",
        root=Path("/tmp/monitor_a"),
        data={
            "meta": {},
            "files": files,
            "segments": segments,
            "contents": contents,
        },
    )


def _fully_affected_monitor_source() -> MonitorSource:
    sql_id = int(cfg.LANG2ID["sql"])
    files = np.array(
        [
            (0, 6, 0, 1, 0, sql_id),
            (6, 6, 1, 1, 0, sql_id),
        ],
        dtype=FILE_DTYPE,
    )
    segments = np.array(
        [
            (0, 0, 6, sql_id),
            (1, 0, 6, sql_id),
        ],
        dtype=SEG_DTYPE,
    )
    contents = np.frombuffer(b"aaaaaabbbbbb", dtype=np.uint8)
    return MonitorSource(
        split_name="monitor_a",
        root=Path("/tmp/monitor_a"),
        data={
            "meta": {},
            "files": files,
            "segments": segments,
            "contents": contents,
        },
    )


def _encoding_monitor_source() -> MonitorSource:
    encoding_id = int(cfg.LANG2ID["encoding_base64"])
    sql_id = int(cfg.LANG2ID["sql"])
    files = np.array(
        [
            (0, 6, 0, 1, 0, encoding_id),
            (6, 6, 1, 1, 0, sql_id),
        ],
        dtype=FILE_DTYPE,
    )
    segments = np.array(
        [
            (0, 0, 6, encoding_id),
            (1, 0, 6, sql_id),
        ],
        dtype=SEG_DTYPE,
    )
    contents = np.frombuffer(b"aaaaaabbbbbb", dtype=np.uint8)
    return MonitorSource(
        split_name="monitor_a",
        root=Path("/tmp/monitor_a"),
        data={
            "meta": {},
            "files": files,
            "segments": segments,
            "contents": contents,
        },
    )


class _SelectiveOracle:
    def __init__(self, fail_sample_hash: str, success_label: str) -> None:
        self.fail_sample_hash = str(fail_sample_hash)
        self.success_label = str(success_label)
        self.name = "selective"
        self.model = "selective-test"
        self.batch_size = 8
        self.last_snippet_sources: dict[str, str] = {}
        self.call_count = 0
        self.call_sizes: list[int] = []

    def annotate(self, snippets):
        self.call_count += 1
        self.call_sizes.append(len(snippets))
        self.last_snippet_sources = {}
        if not snippets:
            return {}

        out = {}
        for snippet in snippets:
            sid = str(snippet.snippet_id)
            sample_hash = str(snippet.metadata.get("sample_hash", ""))
            if sample_hash == self.fail_sample_hash:
                self.last_snippet_sources[sid] = "skipped_parse_failed"
                continue
            self.last_snippet_sources[sid] = "model"
            out[sid] = [OracleSegment(0, len(snippet.text), self.success_label)]
        return out


class TestRelabelMonitorSet(unittest.TestCase):
    def test_scan_monitor_sources_queues_only_files_above_diff_threshold(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        predictor = _FakePredictor(
            {
                b"aaaaaa": [py_id] * 6,
                b"bbbbbb": [py_id] * 6,
            }
        )

        summary = scan_monitor_sources(
            sources=[_monitor_source()],
            predictor=predictor,
            min_diff_chars=4,
            other_threshold=0.0,
            allow_repeat_hashes=False,
            existing_hashes=set(),
            limit_files_per_split=None,
            progress_every=0,
            batch_size=2,
        )

        self.assertEqual(summary.total_files_scanned, 2)
        self.assertEqual(summary.affected_files_total, 1)
        self.assertEqual(summary.affected_files_by_split, {"monitor_a": 1})
        self.assertEqual(summary.skipped_existing_files, 0)
        self.assertEqual(summary.skipped_encoding_files, 0)
        self.assertEqual(len(summary.queued_files), 1)
        queued = summary.queued_files[0]
        self.assertEqual(queued.file_idx, 1)
        self.assertEqual(queued.file_type, "sql")
        self.assertEqual(queued.diff_count, 6)
        self.assertAlmostEqual(queued.diff_ratio, 1.0)

    def test_scan_monitor_sources_skips_encoding_file_types(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        predictor = _FakePredictor(
            {
                b"bbbbbb": [py_id] * 6,
            }
        )

        summary = scan_monitor_sources(
            sources=[_encoding_monitor_source()],
            predictor=predictor,
            min_diff_chars=4,
            other_threshold=0.0,
            allow_repeat_hashes=False,
            existing_hashes=set(),
            limit_files_per_split=None,
            progress_every=0,
            batch_size=2,
        )

        self.assertEqual(summary.total_files_scanned, 1)
        self.assertEqual(summary.affected_files_total, 1)
        self.assertEqual(summary.affected_files_by_split, {"monitor_a": 1})
        self.assertEqual(summary.skipped_existing_files, 0)
        self.assertEqual(summary.skipped_encoding_files, 1)
        self.assertEqual(len(summary.queued_files), 1)
        self.assertEqual(summary.queued_files[0].file_type, "sql")

    def test_build_relabel_chunks_splits_on_newlines_when_possible(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        sql_id = int(cfg.LANG2ID["sql"])
        text = "aaaa\nbbbb\ncccc\n"
        truth = np.array([py_id] * len(text), dtype=np.int32)
        pred = truth.copy()
        pred[5:9] = sql_id

        affected = AffectedMonitorFile(
            source_split="monitor_a",
            source_root="/tmp/monitor_a",
            file_idx=7,
            file_type_id=py_id,
            file_type="python",
            byte_len=len(text),
            sample_hash="sample-hash",
            text=text,
            truth_ids=truth,
            pred_ids=pred,
            diff_count=4,
            valid_count=len(text),
            diff_ratio=4.0 / float(len(text)),
            first_diff_index=5,
        )

        chunks = build_relabel_chunks([affected], oracle_max_chars=6)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].snippet.text, "aaaa\n")
        self.assertEqual(chunks[1].snippet.text, "bbbb\n")
        self.assertEqual(chunks[2].snippet.text, "cccc\n")
        self.assertEqual(chunks[1].diff_count, 4)
        self.assertEqual(chunks[1].first_diff_global, 5)

    def test_build_store_rows_round_trips_with_stub_oracle(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        sql_id = int(cfg.LANG2ID["sql"])
        text = "bbbbbb"
        affected = AffectedMonitorFile(
            source_split="monitor_a",
            source_root="/tmp/monitor_a",
            file_idx=1,
            file_type_id=sql_id,
            file_type="sql",
            byte_len=len(text),
            sample_hash="sample-hash",
            text=text,
            truth_ids=np.array([sql_id] * len(text), dtype=np.int32),
            pred_ids=np.array([py_id] * len(text), dtype=np.int32),
            diff_count=len(text),
            valid_count=len(text),
            diff_ratio=1.0,
            first_diff_index=0,
        )
        chunks = build_relabel_chunks([affected], oracle_max_chars=128)
        oracle = StubOracle()
        refined = oracle.annotate([chunks[0].snippet])
        rows = build_store_rows(
            chunks=chunks,
            refined=refined,
            round_id="monitor-relabel-test",
            oracle_name=oracle.name,
            oracle_model=oracle.model,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.source_split, "monitor_a")
        self.assertEqual(row.source_lang, "sql")
        self.assertEqual(row.sample_index, 1)
        self.assertEqual(row.snippet_text, text)
        self.assertEqual(row.predicted_segments, [{"start": 0, "end": 6, "label": "python"}])
        self.assertEqual(row.refined_segments, [{"start": 0, "end": 6, "label": "python"}])
        self.assertEqual(row.metadata["relabel_kind"], "monitor_file_relabel")

        with tempfile.TemporaryDirectory() as tmpdir:
            store = LabelStore(Path(tmpdir) / "label_store.sqlite")
            inserted = store.add_many(rows)
            self.assertEqual(inserted, 1)
            self.assertEqual(store.count(), 1)

    def test_file_ignore_reason_marks_parse_failed_file(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        affected = AffectedMonitorFile(
            source_split="monitor_a",
            source_root="/tmp/monitor_a",
            file_idx=3,
            file_type_id=py_id,
            file_type="python",
            byte_len=6,
            sample_hash="sample-hash",
            text="aaaaaa",
            truth_ids=np.array([py_id] * 6, dtype=np.int32),
            pred_ids=np.array([py_id] * 6, dtype=np.int32),
            diff_count=6,
            valid_count=6,
            diff_ratio=1.0,
            first_diff_index=0,
        )
        chunks = build_relabel_chunks([affected], oracle_max_chars=128)

        ignore = _file_ignore_reason(
            snippets=[chunks[0].snippet],
            refined={},
            snippet_sources={chunks[0].snippet.snippet_id: "skipped_parse_failed"},
        )

        self.assertIsNotNone(ignore)
        reason, details = ignore
        self.assertEqual(reason, "parse_failed")
        self.assertEqual(details["parse_failed_snippet_ids"], [chunks[0].snippet.snippet_id])

    def test_progress_db_persists_processed_hashes(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        affected = AffectedMonitorFile(
            source_split="monitor_b",
            source_root="/tmp/monitor_b",
            file_idx=11,
            file_type_id=py_id,
            file_type="python",
            byte_len=4,
            sample_hash="persisted-hash",
            text="aaaa",
            truth_ids=np.array([py_id] * 4, dtype=np.int32),
            pred_ids=np.array([py_id] * 4, dtype=np.int32),
            diff_count=4,
            valid_count=4,
            diff_ratio=1.0,
            first_diff_index=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            progress = MonitorRelabelProgress(Path(tmpdir) / "progress.sqlite")
            progress.record(
                affected_file=affected,
                status="ignored",
                reason="parse_failed",
                chunk_count=1,
                row_count=0,
                metadata={"note": "test"},
            )

            self.assertEqual(progress.processed_hashes(), {"persisted-hash"})

    def test_group_relabel_requests_batches_files_by_snippet_budget(self) -> None:
        py_id = int(cfg.LANG2ID["python"])

        def _prepared(sample_hash: str, chunk_count: int) -> PreparedRelabelFile:
            affected = AffectedMonitorFile(
                source_split="monitor_a",
                source_root="/tmp/monitor_a",
                file_idx=int(sample_hash[-1], 16),
                file_type_id=py_id,
                file_type="python",
                byte_len=chunk_count,
                sample_hash=sample_hash,
                text="a" * chunk_count,
                truth_ids=np.array([py_id] * chunk_count, dtype=np.int32),
                pred_ids=np.array([py_id] * chunk_count, dtype=np.int32),
                diff_count=chunk_count,
                valid_count=chunk_count,
                diff_ratio=1.0,
                first_diff_index=0,
            )
            chunks = [
                build_relabel_chunks([affected], oracle_max_chars=1)[i]
                for i in range(chunk_count)
            ]
            return PreparedRelabelFile(affected_file=affected, chunks=chunks)

        groups = group_relabel_requests(
            [
                _prepared("hash-1", 3),
                _prepared("hash-2", 2),
                _prepared("hash-3", 4),
            ],
            max_snippets_per_request=5,
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual([p.affected_file.sample_hash for p in groups[0]], ["hash-1", "hash-2"])
        self.assertEqual([p.affected_file.sample_hash for p in groups[1]], ["hash-3"])

    def test_run_monitor_relabel_skips_written_and_ignored_files_on_rerun(self) -> None:
        sql_id = int(cfg.LANG2ID["sql"])
        py_id = int(cfg.LANG2ID["python"])
        predictor = _FakePredictor(
            {
                b"aaaaaa": [py_id] * 6,
                b"bbbbbb": [py_id] * 6,
            }
        )
        source = _fully_affected_monitor_source()
        preview = scan_monitor_sources(
            sources=[source],
            predictor=predictor,
            min_diff_chars=4,
            other_threshold=0.0,
            allow_repeat_hashes=False,
            existing_hashes=set(),
            limit_files_per_split=None,
            progress_every=0,
            batch_size=2,
        )
        self.assertEqual(len(preview.queued_files), 2)
        fail_hash = preview.queued_files[0].sample_hash

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                checkpoint="unused.msgpack",
                monitor_splits=["monitor_a"],
                store=str(Path(tmpdir) / "label_store.sqlite"),
                progress_db=str(Path(tmpdir) / "progress.sqlite"),
                round_id="monitor-relabel-test",
                min_diff_chars=4,
                oracle_max_chars=128,
                limit_files_per_split=0,
                scan_batch_size=2,
                progress_every=0,
                allow_repeat_hashes=False,
                allow_monitor_b_training=False,
                dry_run=False,
                yes=True,
                other_threshold=0.0,
            )
            oracle = _SelectiveOracle(fail_sample_hash=fail_hash, success_label="sql")

            with mock.patch("active_learning.relabel_monitor_set._build_predictor", return_value=predictor), mock.patch(
                "active_learning.relabel_monitor_set._build_oracle",
                return_value=oracle,
            ), mock.patch(
                "active_learning.relabel_monitor_set._load_monitor_sources",
                return_value=[source],
            ):
                first = run_monitor_relabel(args)
                second = run_monitor_relabel(args)

            self.assertEqual(first["status"], "ok")
            self.assertEqual(first["written_files"], 1)
            self.assertEqual(first["ignored_files"], 1)
            self.assertEqual(first["ignored_by_reason"], {"parse_failed": 1})
            self.assertEqual(first["ignored_by_split"], {"monitor_a": 1})
            self.assertEqual(oracle.call_count, 1)
            self.assertEqual(oracle.call_sizes, [2])

            store = LabelStore(args.store)
            self.assertEqual(store.count(), 1)

            progress = MonitorRelabelProgress(args.progress_db)
            self.assertEqual(len(progress.processed_hashes()), 2)

            self.assertEqual(second["status"], "no_op")
            self.assertEqual(second["queued_files"], 0)
            self.assertEqual(second["affected_files"], 2)


if __name__ == "__main__":
    unittest.main()
