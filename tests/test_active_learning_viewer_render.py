from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from active_learning.oracle import OracleSegment
from viewers.active_learning_viewer import _segments_to_html


class TestActiveLearningViewerRender(unittest.TestCase):
    def test_segments_to_html_splits_diff_runs_within_single_segment(self) -> None:
        text = "abcdef"
        segs = [OracleSegment(0, 6, "python")]
        diff_mask = [False, False, True, True, False, False]
        html = _segments_to_html(text, segs, {"python": "#112233"}, diff_mask=diff_mask)
        self.assertIn('class="tok same"', html)
        self.assertIn('class="tok diff"', html)
        self.assertIn(">ab</span>", html)
        self.assertIn(">cd</span>", html)
        self.assertIn(">ef</span>", html)

    def test_segments_to_html_without_diff_mask_keeps_single_tok_class(self) -> None:
        text = "abcdef"
        segs = [OracleSegment(0, 6, "python")]
        html = _segments_to_html(text, segs, {"python": "#112233"}, diff_mask=None)
        self.assertIn('class="tok"', html)
        self.assertNotIn('class="tok diff"', html)
        self.assertNotIn('class="tok same"', html)
        self.assertIn(">abcdef</span>", html)

    def test_segments_to_html_tolerates_short_diff_mask(self) -> None:
        text = "abcdef"
        segs = [OracleSegment(0, 6, "python")]
        diff_mask = [True, True]
        html = _segments_to_html(text, segs, {"python": "#112233"}, diff_mask=diff_mask)
        self.assertIn(">ab</span>", html)
        self.assertIn(">cdef</span>", html)


if __name__ == "__main__":
    unittest.main()
