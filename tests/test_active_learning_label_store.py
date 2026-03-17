from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from active_learning.label_store import LabelStore, StoredInferenceSample, StoredRefinement


class TestLabelStore(unittest.TestCase):
    def test_existing_sample_hashes_unions_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "al.sqlite"
            store = LabelStore(db_path)

            store.add_inference_samples_many(
                [
                    StoredInferenceSample(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=0,
                        sample_hash="hash_a",
                        sample_text="print('a')",
                        char_count=10,
                        queried_for_oracle=False,
                        candidate_count=0,
                        trigger_ranges=[],
                        predicted_segments=[],
                        metadata={},
                    ),
                    StoredInferenceSample(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=1,
                        sample_hash="hash_b",
                        sample_text="print('b')",
                        char_count=10,
                        queried_for_oracle=True,
                        candidate_count=1,
                        trigger_ranges=[],
                        predicted_segments=[],
                        metadata={},
                    ),
                ]
            )

            store.add_many(
                [
                    StoredRefinement(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=2,
                        sample_hash="hash_c",
                        boundary_index=5,
                        snippet_start=0,
                        snippet_end=10,
                        snippet_text="print('c')",
                        oracle_name="stub",
                        oracle_model="stub",
                        oracle_run_id="",
                        status="ok",
                        acquisition_score=0.0,
                        predicted_segments=[],
                        refined_segments=[],
                        metadata={},
                    ),
                    StoredRefinement(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=3,
                        sample_hash="hash_b",
                        boundary_index=6,
                        snippet_start=0,
                        snippet_end=11,
                        snippet_text="print('bb')",
                        oracle_name="stub",
                        oracle_model="stub",
                        oracle_run_id="",
                        status="ok",
                        acquisition_score=0.0,
                        predicted_segments=[],
                        refined_segments=[],
                        metadata={},
                    ),
                ]
            )

            self.assertEqual(store.existing_sample_hashes(), {"hash_a", "hash_b", "hash_c"})

    def test_mark_inference_samples_queried_sets_flags_from_stored_refinements(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "al.sqlite"
            store = LabelStore(db_path)

            store.add_inference_samples_many(
                [
                    StoredInferenceSample(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=0,
                        sample_hash="hash_a",
                        sample_text="print('a')",
                        char_count=10,
                        queried_for_oracle=True,
                        candidate_count=1,
                        trigger_ranges=[],
                        predicted_segments=[],
                        metadata={},
                    ),
                    StoredInferenceSample(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=1,
                        sample_hash="hash_b",
                        sample_text="print('b')",
                        char_count=10,
                        queried_for_oracle=False,
                        candidate_count=1,
                        trigger_ranges=[],
                        predicted_segments=[],
                        metadata={},
                    ),
                ]
            )

            marked = store.mark_inference_samples_queried(
                round_id="r1",
                sample_keys={("hash_b", 1), ("does_not_exist", 9)},
            )
            self.assertEqual(marked, 1)

            with sqlite3.connect(str(db_path)) as con:
                con.row_factory = sqlite3.Row
                rows = con.execute(
                    """
                    SELECT sample_hash, sample_index, queried_for_oracle
                    FROM inference_samples
                    WHERE round_id = 'r1'
                    ORDER BY sample_index ASC
                    """
                ).fetchall()
            as_tuples = [
                (str(row["sample_hash"]), int(row["sample_index"]), int(row["queried_for_oracle"]))
                for row in rows
            ]
            self.assertEqual(as_tuples, [("hash_a", 0, 0), ("hash_b", 1, 1)])

    def test_open_set_labels_persist_in_sqlite_but_train_as_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "al.sqlite"
            store = LabelStore(db_path)

            store.add(
                StoredRefinement(
                    round_id="r1",
                    source_split="train",
                    source_lang="javascript_typescript",
                    sample_index=0,
                    sample_hash="hash_jsx",
                    boundary_index=8,
                    snippet_start=0,
                    snippet_end=17,
                    snippet_text="<div>Save</div>\n",
                    oracle_name="gemini",
                    oracle_model="gemini-test",
                    oracle_run_id="run-1",
                    status="ok",
                    acquisition_score=0.5,
                    predicted_segments=[],
                    refined_segments=[
                        {"start": 0, "end": 15, "label": "other", "open_set_label": "other_jsx"},
                        {"start": 15, "end": 16, "label": "javascript_typescript"},
                    ],
                    metadata={"oracle_open_set_labels": ["other_jsx"]},
                )
            )

            with sqlite3.connect(str(db_path)) as con:
                row = con.execute(
                    "SELECT refined_segments_json, metadata_json FROM refinements WHERE sample_hash = ?",
                    ("hash_jsx",),
                ).fetchone()
            self.assertIsNotNone(row)
            self.assertIn('"open_set_label": "other_jsx"', str(row[0]))
            self.assertIn('"other_jsx"', str(row[1]))

            windows = store.build_training_windows(
                window_bytes=32,
                pad_byte_id=0,
                pad_label_id=255,
                label_to_id={"javascript_typescript": 0, "other": 1},
                max_windows=1,
                fallback_label="other",
            )
            self.assertEqual(len(windows), 1)
            _, labels = windows[0]
            self.assertTrue(all(int(v) == 1 for v in labels[:15]))
            self.assertEqual(int(labels[15]), 0)

    def test_build_training_windows_excludes_monitor_b_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "al.sqlite"
            store = LabelStore(db_path)

            store.add_many(
                [
                    StoredRefinement(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=0,
                        sample_hash="hash_train",
                        boundary_index=2,
                        snippet_start=0,
                        snippet_end=4,
                        snippet_text="abcd",
                        oracle_name="stub",
                        oracle_model="stub",
                        oracle_run_id="",
                        status="ok",
                        acquisition_score=1.0,
                        predicted_segments=[],
                        refined_segments=[{"start": 0, "end": 4, "label": "python"}],
                        metadata={},
                    ),
                    StoredRefinement(
                        round_id="r1",
                        source_split="monitor_b",
                        source_lang="python",
                        sample_index=1,
                        sample_hash="hash_monitor_b",
                        boundary_index=2,
                        snippet_start=0,
                        snippet_end=4,
                        snippet_text="WXYZ",
                        oracle_name="stub",
                        oracle_model="stub",
                        oracle_run_id="",
                        status="ok",
                        acquisition_score=1.0,
                        predicted_segments=[],
                        refined_segments=[{"start": 0, "end": 4, "label": "python"}],
                        metadata={},
                    ),
                ]
            )

            default_windows = store.build_training_windows(
                window_bytes=8,
                pad_byte_id=0,
                pad_label_id=255,
                label_to_id={"python": 0, "other": 1},
                fallback_label="other",
            )
            self.assertEqual(len(default_windows), 1)
            default_x, _ = default_windows[0]
            np.testing.assert_array_equal(
                default_x[:4],
                np.frombuffer(b"abcd", dtype=np.uint8).astype(np.int32),
            )

            all_windows = store.build_training_windows(
                window_bytes=8,
                pad_byte_id=0,
                pad_label_id=255,
                label_to_id={"python": 0, "other": 1},
                fallback_label="other",
                exclude_source_splits=(),
            )
            self.assertEqual(len(all_windows), 2)


if __name__ == "__main__":
    unittest.main()
