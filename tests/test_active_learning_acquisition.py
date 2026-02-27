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

from active_learning.acquisition import select_candidate_spans


class TestAcquisition(unittest.TestCase):
    def test_select_candidate_spans_focuses_boundary_uncertainty(self) -> None:
        t = 48
        c = 3
        probs = np.zeros((t, c), dtype=np.float32)
        probs[:24, 0] = 0.97
        probs[:24, 1] = 0.02
        probs[:24, 2] = 0.01
        probs[24:, 1] = 0.97
        probs[24:, 0] = 0.02
        probs[24:, 2] = 0.01

        # High uncertainty and quick local flips around the true boundary.
        probs[21] = np.array([0.45, 0.45, 0.10], dtype=np.float32)
        probs[22] = np.array([0.52, 0.40, 0.08], dtype=np.float32)
        probs[23] = np.array([0.48, 0.44, 0.08], dtype=np.float32)
        probs[24] = np.array([0.44, 0.48, 0.08], dtype=np.float32)
        probs[25] = np.array([0.40, 0.52, 0.08], dtype=np.float32)

        labels = np.argmax(probs, axis=-1).astype(np.int32)
        spans = select_candidate_spans(
            probs,
            labels=labels,
            context_chars=6,
            top_k=3,
            min_score=0.0,
        )
        self.assertTrue(spans, "Expected at least one candidate span.")
        self.assertTrue(any(abs(span.boundary - 24) <= 2 for span in spans))
        for span in spans:
            self.assertGreater(span.end, span.start)
            self.assertLessEqual(span.end - span.start, 12)


if __name__ == "__main__":
    unittest.main()
