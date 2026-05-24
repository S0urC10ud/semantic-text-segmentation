from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import evaluation.evaluation as evalmod


def _prob_row(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


class TestEvaluationThesisPostprocess(unittest.TestCase):
    def test_min_run_collapses_short_interior_island(self) -> None:
        text = "AAAABBBAAAA"
        labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        probs = [_prob_row(0.95, 0.05) for _ in range(4)]
        probs.extend(_prob_row(0.10, 0.90) for _ in range(3))
        probs.extend(_prob_row(0.95, 0.05) for _ in range(4))

        result, stats = evalmod._apply_thesis_postprocess(
            text,
            labels,
            probs,
            other_threshold=0.0,
            other_id=2,
            min_run_chars=5,
            boundary_snap_max_shift=0,
        )

        self.assertEqual(result, [0] * len(text))
        self.assertEqual(stats["min_run_changed_chars"], 3)

    def test_min_run_preserves_short_edge_run(self) -> None:
        text = "BBBBAAAAAAA"
        labels = [1, 1, 1, 1] + [0] * 7
        probs = [_prob_row(0.05, 0.95) for _ in range(4)]
        probs.extend(_prob_row(0.95, 0.05) for _ in range(7))

        result, _stats = evalmod._apply_thesis_postprocess(
            text,
            labels,
            probs,
            other_threshold=0.0,
            other_id=2,
            min_run_chars=5,
            boundary_snap_max_shift=0,
        )

        self.assertEqual(result, labels)

    def test_min_run_treats_other_like_any_other_label(self) -> None:
        text = "AAAAOOOAAAA"
        other_id = 2
        labels = [0, 0, 0, 0, other_id, other_id, other_id, 0, 0, 0, 0]
        probs = [_prob_row(0.95, 0.03, 0.02) for _ in range(4)]
        probs.extend(_prob_row(0.10, 0.10, 0.80) for _ in range(3))
        probs.extend(_prob_row(0.95, 0.03, 0.02) for _ in range(4))

        result, _stats = evalmod._apply_thesis_postprocess(
            text,
            labels,
            probs,
            other_threshold=0.0,
            other_id=other_id,
            min_run_chars=5,
            boundary_snap_max_shift=0,
        )

        self.assertEqual(result, [0] * len(text))

    def test_boundary_snap_moves_by_at_most_two_chars_toward_delimiter(self) -> None:
        text = "<abX"
        labels = [0, 0, 0, 1]
        probs = [
            _prob_row(0.90, 0.10),
            _prob_row(0.52, 0.50),
            _prob_row(0.51, 0.50),
            _prob_row(0.05, 0.95),
        ]

        result, _stats = evalmod._apply_thesis_postprocess(
            text,
            labels,
            probs,
            other_threshold=0.0,
            other_id=2,
            min_run_chars=1,
            boundary_snap_max_shift=2,
        )
        limited, _limited_stats = evalmod._apply_thesis_postprocess(
            text,
            labels,
            probs,
            other_threshold=0.0,
            other_id=2,
            min_run_chars=1,
            boundary_snap_max_shift=1,
        )

        self.assertEqual(result, [0, 1, 1, 1])
        self.assertEqual(limited, labels)

    def test_boundary_snap_does_not_cross_newline(self) -> None:
        text = "a\nb"
        labels = [0, 0, 1]
        probs = [
            _prob_row(0.95, 0.05),
            _prob_row(0.60, 0.35),
            _prob_row(0.05, 0.95),
        ]

        result, _stats = evalmod._apply_thesis_postprocess(
            text,
            labels,
            probs,
            other_threshold=0.0,
            other_id=2,
            min_run_chars=1,
            boundary_snap_max_shift=2,
        )

        self.assertEqual(result, labels)

    def test_cli_default_postprocess_profile_is_off(self) -> None:
        args = evalmod.parse_args([])

        self.assertEqual(args.postprocess_profile, evalmod.POSTPROCESS_PROFILE_OFF)
        self.assertEqual(args.postprocess_min_run, 5)
        self.assertEqual(args.postprocess_boundary_snap_max_shift, 2)


if __name__ == "__main__":
    unittest.main()
