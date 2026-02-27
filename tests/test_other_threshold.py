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

from viewers.core import Predictor


class TestOtherThreshold(unittest.TestCase):
    def test_threshold_predictions_routes_low_conf_to_other(self) -> None:
        probs = np.array(
            [
                [0.70, 0.20, 0.10],  # keep class 0
                [0.34, 0.33, 0.33],  # low confidence -> OTHER
                [0.20, 0.75, 0.05],  # keep class 1
                [0.40, 0.39, 0.21],  # low confidence -> OTHER
            ],
            dtype=np.float32,
        )
        pred = Predictor.threshold_predictions(
            probs,
            other_threshold=0.60,
            other_id=3,
        )
        self.assertEqual(pred.tolist(), [0, 3, 1, 3])


if __name__ == "__main__":
    unittest.main()
