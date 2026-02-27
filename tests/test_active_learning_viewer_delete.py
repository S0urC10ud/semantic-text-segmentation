from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from active_learning.label_store import LabelStore, StoredInferenceSample, StoredRefinement
from viewers.active_learning_viewer import ActiveLearningStore


class TestActiveLearningViewerDelete(unittest.TestCase):
    def test_delete_sample_inference_mode_removes_linked_refinements(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "al.sqlite"
            ls = LabelStore(db_path)

            ls.add_inference_samples_many(
                [
                    StoredInferenceSample(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=7,
                        sample_hash="hash_a",
                        sample_text="print('x')",
                        char_count=10,
                        queried_for_oracle=False,
                        candidate_count=2,
                        trigger_ranges=[],
                        predicted_segments=[],
                        metadata={},
                    )
                ]
            )
            ls.add_many(
                [
                    StoredRefinement(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=7,
                        sample_hash="hash_a",
                        boundary_index=1,
                        snippet_start=0,
                        snippet_end=5,
                        snippet_text="print",
                        oracle_name="stub",
                        oracle_model="stub",
                        oracle_run_id="",
                        status="ok",
                        acquisition_score=0.1,
                        predicted_segments=[],
                        refined_segments=[],
                        metadata={},
                    ),
                    StoredRefinement(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=7,
                        sample_hash="hash_a",
                        boundary_index=2,
                        snippet_start=5,
                        snippet_end=10,
                        snippet_text="('x')",
                        oracle_name="stub",
                        oracle_model="stub",
                        oracle_run_id="",
                        status="ok",
                        acquisition_score=0.2,
                        predicted_segments=[],
                        refined_segments=[],
                        metadata={},
                    ),
                ]
            )

            store = ActiveLearningStore(db_path)
            result = store.delete_sample(1)
            self.assertEqual(result["mode"], "inference_sample")
            self.assertEqual(int(result["deleted_inference_samples"]), 1)
            self.assertEqual(int(result["deleted_refinements"]), 2)

            with sqlite3.connect(str(db_path)) as con:
                c_inf = int(con.execute("SELECT COUNT(*) FROM inference_samples").fetchone()[0])
                c_ref = int(con.execute("SELECT COUNT(*) FROM refinements").fetchone()[0])
            self.assertEqual(c_inf, 0)
            self.assertEqual(c_ref, 0)

    def test_delete_sample_refinement_mode_removes_single_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "al.sqlite"
            ls = LabelStore(db_path)
            ls.add_many(
                [
                    StoredRefinement(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=7,
                        sample_hash="hash_a",
                        boundary_index=1,
                        snippet_start=0,
                        snippet_end=5,
                        snippet_text="print",
                        oracle_name="stub",
                        oracle_model="stub",
                        oracle_run_id="",
                        status="ok",
                        acquisition_score=0.1,
                        predicted_segments=[],
                        refined_segments=[],
                        metadata={},
                    ),
                    StoredRefinement(
                        round_id="r1",
                        source_split="train",
                        source_lang="python",
                        sample_index=7,
                        sample_hash="hash_a",
                        boundary_index=2,
                        snippet_start=5,
                        snippet_end=10,
                        snippet_text="('x')",
                        oracle_name="stub",
                        oracle_model="stub",
                        oracle_run_id="",
                        status="ok",
                        acquisition_score=0.2,
                        predicted_segments=[],
                        refined_segments=[],
                        metadata={},
                    ),
                ]
            )

            store = ActiveLearningStore(db_path)
            result = store.delete_sample(1)
            self.assertEqual(result["mode"], "refinement_row")
            self.assertEqual(int(result["deleted_inference_samples"]), 0)
            self.assertEqual(int(result["deleted_refinements"]), 1)

            with sqlite3.connect(str(db_path)) as con:
                c_inf = int(con.execute("SELECT COUNT(*) FROM inference_samples").fetchone()[0])
                c_ref = int(con.execute("SELECT COUNT(*) FROM refinements").fetchone()[0])
            self.assertEqual(c_inf, 0)
            self.assertEqual(c_ref, 1)


if __name__ == "__main__":
    unittest.main()
