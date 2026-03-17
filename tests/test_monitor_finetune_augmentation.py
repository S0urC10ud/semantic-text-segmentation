from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import datasets as hfds
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg
from active_learning.label_store import LabelStore, StoredRefinement
from train.utils.epoch_batcher import (
    MonitorFineTuneBatcher,
    _apply_monitor_removal_span,
    _build_augmented_fragment_datasets,
)
from train.utils.monitor_eval import FILE_DTYPE, SEG_DTYPE


def _monitor_data() -> dict:
    py_id = int(cfg.LANG2ID["python"])
    sql_id = int(cfg.LANG2ID["sql"])
    files = np.array(
        [(0, 6, 0, 2, 0, py_id)],
        dtype=FILE_DTYPE,
    )
    segments = np.array(
        [
            (0, 0, 2, py_id),
            (0, 2, 6, sql_id),
        ],
        dtype=SEG_DTYPE,
    )
    contents = np.frombuffer(b"aaBBBB", dtype=np.uint8)
    return {
        "meta": {},
        "files": files,
        "segments": segments,
        "contents": contents,
    }


def _monitor_data_two_files() -> dict:
    py_id = int(cfg.LANG2ID["python"])
    cs_id = int(cfg.LANG2ID["csharp"])
    files = np.array(
        [
            (0, 4, 0, 1, 0, py_id),
            (4, 4, 1, 1, 0, cs_id),
        ],
        dtype=FILE_DTYPE,
    )
    segments = np.array(
        [
            (0, 0, 4, py_id),
            (1, 0, 4, cs_id),
        ],
        dtype=SEG_DTYPE,
    )
    contents = np.frombuffer(b"aaaaBBBB", dtype=np.uint8)
    return {
        "meta": {},
        "files": files,
        "segments": segments,
        "contents": contents,
    }


class TestMonitorFineTuneAugmentation(unittest.TestCase):
    def test_apply_monitor_removal_span_line_keeps_following_labels_aligned(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        sql_id = int(cfg.LANG2ID["sql"])
        file_bytes = np.frombuffer(b"aa\nbb\nCCCC", dtype=np.uint8)
        file_segments = [
            (0, 6, py_id),
            (6, 10, sql_id),
        ]

        trimmed, trimmed_segments = _apply_monitor_removal_span(
            file_bytes,
            file_segments,
            3,
            6,
        )

        np.testing.assert_array_equal(
            trimmed,
            np.frombuffer(b"aa\nCCCC", dtype=np.uint8),
        )
        self.assertEqual(trimmed_segments, [(0, 3, py_id), (3, 7, sql_id)])

    def test_apply_monitor_removal_span_space_slice_merges_same_label(self) -> None:
        text_id = int(cfg.LANG2ID["text"])
        sql_id = int(cfg.LANG2ID["sql"])
        original = b"Hello world this is a text message\nSELECT"
        file_bytes = np.frombuffer(original, dtype=np.uint8)
        split = len(b"Hello world this is a text message\n")
        file_segments = [
            (0, split, text_id),
            (split, len(original), sql_id),
        ]

        trimmed, trimmed_segments = _apply_monitor_removal_span(
            file_bytes,
            file_segments,
            len(b"Hello "),
            len(b"Hello world this "),
        )

        expected = b"Hello is a text message\nSELECT"
        np.testing.assert_array_equal(
            trimmed,
            np.frombuffer(expected, dtype=np.uint8),
        )
        text_len = len(b"Hello is a text message\n")
        self.assertEqual(trimmed_segments, [(0, text_len, text_id), (text_len, len(expected), sql_id)])

    def test_build_augmented_fragment_datasets_includes_monitor_and_sqlite_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LabelStore(Path(tmpdir) / "al.sqlite")
            store.add_many(
                [
                    StoredRefinement(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=0,
                        sample_hash="hash-1",
                        boundary_index=2,
                        snippet_start=0,
                        snippet_end=4,
                        snippet_text="xyZZ",
                        oracle_name="stub",
                        oracle_model="stub",
                        oracle_run_id="",
                        status="ok",
                        acquisition_score=1.0,
                        predicted_segments=[],
                        refined_segments=[
                            {"start": 0, "end": 2, "label": "python"},
                            {"start": 2, "end": 4, "label": "sql"},
                        ],
                        metadata={},
                    ),
                    StoredRefinement(
                        round_id="r1",
                        source_split="monitor_b",
                        source_lang="python",
                        sample_index=1,
                        sample_hash="hash-monitor-b",
                        boundary_index=2,
                        snippet_start=0,
                        snippet_end=4,
                        snippet_text="qqRR",
                        oracle_name="stub",
                        oracle_model="stub",
                        oracle_run_id="",
                        status="ok",
                        acquisition_score=1.0,
                        predicted_segments=[],
                        refined_segments=[
                            {"start": 0, "end": 2, "label": "python"},
                            {"start": 2, "end": 4, "label": "sql"},
                        ],
                        metadata={},
                    ),
                ]
            )

            dsets = _build_augmented_fragment_datasets(
                _monitor_data(),
                active_learning_store=str(store.path),
                active_learning_limit=100,
            )

        py_rows = list(dsets[int(cfg.LANG2ID["python"])])
        sql_rows = list(dsets[int(cfg.LANG2ID["sql"])])
        self.assertEqual(sorted(str(row["content"]) for row in py_rows), ["aa", "xy"])
        self.assertEqual(sorted(str(row["content"]) for row in sql_rows), ["BBBB", "ZZ"])

    def test_build_augmented_window_pure_keeps_raw_monitor_offsets(self) -> None:
        data_cfg = cfg.DataConfig(window_min_bytes=6, window_max_bytes=6)
        monitor_data = _monitor_data()
        x, y = MonitorFineTuneBatcher._build_augmented_window(
            np.random.default_rng(0),
            6,
            monitor_data["files"],
            monitor_data["contents"],
            monitor_data["segments"],
            len(monitor_data["files"]),
            {},
            data_cfg,
            forced_mode="pure",
        )

        py_id = int(cfg.LANG2ID["python"])
        sql_id = int(cfg.LANG2ID["sql"])
        np.testing.assert_array_equal(x[:6], np.frombuffer(b"aaBBBB", dtype=np.uint8).astype(np.int32))
        np.testing.assert_array_equal(
            y[:6],
            np.array([py_id, py_id, sql_id, sql_id, sql_id, sql_id], dtype=np.uint8),
        )

    def test_build_window_dense_bias_prefers_requested_label(self) -> None:
        monitor_data = _monitor_data_two_files()
        cs_id = int(cfg.LANG2ID["csharp"])
        x, y = MonitorFineTuneBatcher._build_window(
            np.random.default_rng(0),
            4,
            monitor_data["files"],
            monitor_data["contents"],
            monitor_data["segments"],
            len(monitor_data["files"]),
            preferred_file_indices=np.array([1], dtype=np.int64),
            preferred_prob=1.0,
        )

        np.testing.assert_array_equal(
            x[:4],
            np.frombuffer(b"BBBB", dtype=np.uint8).astype(np.int32),
        )
        np.testing.assert_array_equal(
            y[:4],
            np.array([cs_id, cs_id, cs_id, cs_id], dtype=np.uint8),
        )

    def test_build_augmented_window_mixed_keeps_labels_aligned(self) -> None:
        data_cfg = cfg.DataConfig(window_min_bytes=12, window_max_bytes=12, min_seg_len=2)
        fragment_dsets = {
            int(cfg.LANG2ID["python"]): hfds.Dataset.from_list(
                [{"content": "alpha\nbeta\n", "source": "monitor:py"}]
            ),
            int(cfg.LANG2ID["sql"]): hfds.Dataset.from_list(
                [{"content": "SELECT\nFROM\n", "source": "al:sql"}]
            ),
        }
        monitor_data = _monitor_data()

        with patch("train.utils.window_generator._sample_mixed_segment_count", return_value=2):
            x, y = MonitorFineTuneBatcher._build_augmented_window(
                np.random.default_rng(4),
                12,
                monitor_data["files"],
                monitor_data["contents"],
                monitor_data["segments"],
                len(monitor_data["files"]),
                fragment_dsets,
                data_cfg,
                forced_mode="mixed",
            )

        valid = y != cfg.PAD_ID
        self.assertTrue(np.any(valid))
        self.assertTrue(np.all(x[valid] != cfg.PAD_BYTE_ID))
        labels = {int(val) for val in y[valid].tolist()}
        self.assertTrue(
            labels.issubset({int(cfg.LANG2ID["python"]), int(cfg.LANG2ID["sql"])}),
        )


if __name__ == "__main__":
    unittest.main()
