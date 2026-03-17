from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg
from viewers.monitor_error_viewer import _compress_label_runs, _focus_metrics


class TestMonitorErrorViewer(unittest.TestCase):
    def test_focus_metrics_computes_label_precision_recall_and_f1(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        sql_id = int(cfg.LANG2ID["sql"])
        tokens = np.array([65, 66, 67, 68], dtype=np.int32)
        truth = np.array([py_id, py_id, sql_id, sql_id], dtype=np.int32)
        pred = np.array([py_id, sql_id, sql_id, sql_id], dtype=np.int32)

        metrics = _focus_metrics(tokens, truth, pred, focus_label_id=sql_id)

        self.assertAlmostEqual(float(metrics["overall_accuracy"]), 0.75)
        self.assertAlmostEqual(float(metrics["diff_ratio"]), 0.25)
        self.assertEqual(int(metrics["focus_support"]), 2)
        self.assertAlmostEqual(float(metrics["focus_precision"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(metrics["focus_recall"]), 1.0)
        self.assertAlmostEqual(float(metrics["focus_f1"]), 0.8)
        self.assertEqual(int(metrics["tp"]), 2)
        self.assertEqual(int(metrics["fp"]), 1)
        self.assertEqual(int(metrics["fn"]), 0)

    def test_compress_label_runs_merges_adjacent_labels(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        sql_id = int(cfg.LANG2ID["sql"])
        runs = _compress_label_runs([py_id, py_id, sql_id, sql_id, sql_id, py_id])

        self.assertEqual(
            runs,
            [
                {"start": 0, "end": 2, "label": "python"},
                {"start": 2, "end": 5, "label": "sql"},
                {"start": 5, "end": 6, "label": "python"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
