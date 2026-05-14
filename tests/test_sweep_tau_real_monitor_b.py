from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import evaluation.sweep_tau_real_monitor_b as mod


class TestSweepTauRealMonitorB(unittest.TestCase):
    def test_resolve_runner_args_supports_magika_without_checkpoint(self) -> None:
        args = SimpleNamespace(
            checkpoint="",
            arch="magika",
            model_dim=None,
            channels=None,
            dtype=None,
            chunk=None,
            device="cpu",
            batch_size=128,
            inference_backend="auto",
            mamba_layers=None,
            mamba_d_state=None,
            mamba_expand=None,
            mamba_dt_rank=None,
            mamba_conv=None,
            mamba_bidirectional=None,
        )

        resolved, runner_kwargs = mod._resolve_runner_args(args)

        self.assertEqual(resolved["checkpoint"], "magika://default")
        self.assertEqual(resolved["arch"], "magika")
        self.assertEqual(int(resolved["chunk"]), int(mod.DEFAULT_MAGIKA_CHUNK))
        self.assertEqual(runner_kwargs["arch"], "magika")
        self.assertNotIn("checkpoint", runner_kwargs)

    def test_select_monitor_file_indices_is_deterministic(self) -> None:
        picked_a = mod.select_monitor_file_indices(10, 4, 17)
        picked_b = mod.select_monitor_file_indices(10, 4, 17)
        picked_c = mod.select_monitor_file_indices(10, 4, 23)

        self.assertEqual(picked_a.tolist(), picked_b.tolist())
        self.assertEqual(sorted(picked_a.tolist()), picked_a.tolist())
        self.assertEqual(len(picked_a), 4)
        self.assertNotEqual(picked_a.tolist(), picked_c.tolist())

    def test_accumulate_open_set_binary_counts_matches_expected(self) -> None:
        tau_grid = [0.5]
        counts = {
            "tp_other": np.zeros((1,), dtype=np.int64),
            "fn_other": np.zeros((1,), dtype=np.int64),
            "fp_other": np.zeros((1,), dtype=np.int64),
            "tn_other": np.zeros((1,), dtype=np.int64),
        }

        mod.accumulate_open_set_binary_counts(
            counts,
            truth_is_other=np.array([True, True, False, False], dtype=np.bool_),
            max_prob=np.array([0.10, 0.90, 0.20, 0.80], dtype=np.float32),
            tau_grid=tau_grid,
        )

        rows = mod._build_rows_from_counts(
            tau_grid=tau_grid,
            counts=counts,
            select_by="other_f1",
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["tp_other"], 1)
        self.assertEqual(row["fn_other"], 1)
        self.assertEqual(row["fp_other"], 1)
        self.assertEqual(row["tn_other"], 1)
        self.assertEqual(row["truth_other"], 2)
        self.assertEqual(row["truth_non_other"], 2)
        self.assertEqual(row["predicted_other"], 2)
        self.assertEqual(row["predicted_non_other"], 2)
        self.assertAlmostEqual(float(row["other_precision"]), 0.5)
        self.assertAlmostEqual(float(row["other_recall"]), 0.5)
        self.assertAlmostEqual(float(row["other_f1"]), 0.5)
        self.assertAlmostEqual(float(row["specificity"]), 0.5)
        self.assertAlmostEqual(float(row["balanced_acc"]), 0.5)

    def test_base_predicted_other_counts_even_without_tau_trigger(self) -> None:
        tau_grid = [0.5]
        counts = {
            "tp_other": np.zeros((1,), dtype=np.int64),
            "fn_other": np.zeros((1,), dtype=np.int64),
            "fp_other": np.zeros((1,), dtype=np.int64),
            "tn_other": np.zeros((1,), dtype=np.int64),
        }

        mod.accumulate_open_set_binary_counts(
            counts,
            truth_is_other=np.array([False, True], dtype=np.bool_),
            max_prob=np.array([0.99, 0.99], dtype=np.float32),
            tau_grid=tau_grid,
            base_pred_is_other=np.array([True, False], dtype=np.bool_),
        )

        rows = mod._build_rows_from_counts(
            tau_grid=tau_grid,
            counts=counts,
            select_by="other_f1",
        )
        row = rows[0]
        self.assertEqual(row["tp_other"], 0)
        self.assertEqual(row["fn_other"], 1)
        self.assertEqual(row["fp_other"], 1)
        self.assertEqual(row["tn_other"], 0)


if __name__ == "__main__":
    unittest.main()
