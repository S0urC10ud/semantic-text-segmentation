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

from active_learning.oracle import BoundarySnippet
from active_learning.round import _limit_snippets_for_oracle_requests, _make_snippet_id


def _snippet(sid: str) -> BoundarySnippet:
    text = "abc"
    return BoundarySnippet(
        snippet_id=sid,
        text=text,
        global_start=0,
        global_end=len(text),
        boundary=1,
        predicted_labels=["python"] * len(text),
        metadata={"source_lang": "python"},
    )


class TestRoundOracleRequestCap(unittest.TestCase):
    def test_make_snippet_id_is_short_and_deterministic(self) -> None:
        sid_a = _make_snippet_id("abc123", 10, 20, 15)
        sid_b = _make_snippet_id("abc123", 10, 20, 15)
        sid_c = _make_snippet_id("abc123", 10, 21, 15)
        self.assertEqual(len(sid_a), 10)
        self.assertEqual(sid_a, sid_b)
        self.assertNotEqual(sid_a, sid_c)

    def test_unlimited_keeps_all_snippets(self) -> None:
        snippets = [_snippet(f"s{i}") for i in range(5)]
        score_by_id = {s.snippet_id: float(i) for i, s in enumerate(snippets)}
        kept = _limit_snippets_for_oracle_requests(
            snippets,
            score_by_id=score_by_id,
            oracle_batch_size=2,
            max_oracle_requests=None,
        )
        self.assertEqual([s.snippet_id for s in kept], [s.snippet_id for s in snippets])

    def test_limit_selects_top_scores(self) -> None:
        snippets = [_snippet(f"s{i}") for i in range(8)]
        score_by_id = {
            "s0": 0.10,
            "s1": 0.95,
            "s2": 0.22,
            "s3": 0.83,
            "s4": 0.57,
            "s5": 0.44,
            "s6": 0.99,
            "s7": 0.66,
        }
        kept = _limit_snippets_for_oracle_requests(
            snippets,
            score_by_id=score_by_id,
            oracle_batch_size=2,
            max_oracle_requests=3,
        )
        # 3 requests * batch size 2 => keep top 6 by acquisition score.
        self.assertEqual(len(kept), 6)
        self.assertEqual(
            [s.snippet_id for s in kept],
            ["s6", "s1", "s3", "s7", "s4", "s5"],
        )

    def test_zero_limit_returns_empty(self) -> None:
        snippets = [_snippet("s0"), _snippet("s1")]
        score_by_id = {"s0": 0.2, "s1": 0.7}
        kept = _limit_snippets_for_oracle_requests(
            snippets,
            score_by_id=score_by_id,
            oracle_batch_size=2,
            max_oracle_requests=0,
        )
        self.assertEqual(kept, [])


if __name__ == "__main__":
    unittest.main()
