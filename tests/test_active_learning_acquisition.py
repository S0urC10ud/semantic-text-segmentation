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


    def test_other_boundary_uses_inverted_entropy_scoring(self) -> None:
        """Other-adjacent boundaries should score via 1 - entropy (not entropy + flip)."""
        t = 48
        c = 3
        other_id = c  # label 3 = "other"
        probs = np.zeros((t, c), dtype=np.float32)
        # Left half: confidently class 0
        probs[:24, 0] = 0.97
        probs[:24, 1] = 0.02
        probs[:24, 2] = 0.01
        # Right half: near-uniform (model thinks it's "other")
        probs[24:] = 1.0 / c

        # Labels: class 0 on left, "other" on right
        labels = np.zeros(t, dtype=np.int32)
        labels[24:] = other_id

        spans = select_candidate_spans(
            probs,
            labels=labels,
            context_chars=6,
            top_k=3,
            min_score=0.0,
            other_id=other_id,
        )
        self.assertTrue(spans, "Expected at least one candidate span.")
        for span in spans:
            self.assertTrue(span.is_other_boundary)
            # Model already outputs uniform on right side → ent_mean ≈ 1 → score ≈ 0
            self.assertLess(span.score, 0.5, "Uniform-outputting 'other' region should have LOW score")

    def test_other_boundary_peaked_output_gets_high_score(self) -> None:
        """Other-adjacent boundary where model outputs peaked (wrong) gets high score."""
        t = 48
        c = 3
        other_id = c
        probs = np.zeros((t, c), dtype=np.float32)
        # Left half: confidently class 0
        probs[:24, 0] = 0.97
        probs[:24, 1] = 0.02
        probs[:24, 2] = 0.01
        # Right half: PEAKED at class 1 (but label says "other" — model is wrong)
        probs[24:, 1] = 0.95
        probs[24:, 0] = 0.03
        probs[24:, 2] = 0.02

        labels = np.zeros(t, dtype=np.int32)
        labels[24:] = other_id

        spans = select_candidate_spans(
            probs,
            labels=labels,
            context_chars=6,
            top_k=3,
            min_score=0.0,
            other_id=other_id,
        )
        self.assertTrue(spans)
        for span in spans:
            self.assertTrue(span.is_other_boundary)
            # Model wrongly outputs peaked → ent_mean ≈ 0 → score ≈ 1
            self.assertGreater(span.score, 0.5, "Peaked output for 'other' region should have HIGH score")

    def test_no_other_id_backward_compatible(self) -> None:
        """Without other_id, all boundaries use standard entropy+flip scoring."""
        t = 48
        c = 3
        probs = np.zeros((t, c), dtype=np.float32)
        probs[:24, 0] = 0.97
        probs[:24, 1] = 0.02
        probs[:24, 2] = 0.01
        probs[24:, 1] = 0.97
        probs[24:, 0] = 0.02
        probs[24:, 2] = 0.01
        probs[22:26] = 1.0 / c  # uncertain zone

        labels = np.argmax(probs, axis=-1).astype(np.int32)
        spans = select_candidate_spans(
            probs,
            labels=labels,
            context_chars=6,
            top_k=3,
            min_score=0.0,
            other_id=None,  # no "other" class
        )
        for span in spans:
            self.assertFalse(span.is_other_boundary)


if __name__ == "__main__":
    unittest.main()
