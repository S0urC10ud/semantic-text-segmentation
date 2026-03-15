from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg
from active_learning.oracle import BoundarySnippet
from active_learning.round import (
    _build_parser,
    _build_snippets_for_samples_batch,
    _iter_split_examples,
    _limit_snippets_for_oracle_requests,
    _make_snippet_id,
)


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

    def test_build_snippets_for_samples_batch_uses_batched_predictor(self) -> None:
        html_id = int(cfg.LANG2ID["html"])
        css_id = int(cfg.LANG2ID["css"])

        class FakePredictor:
            def __init__(self) -> None:
                self.calls = []

            def segment_texts(self, texts, min_run_chars=6, chunk=None):
                self.calls.append(
                    {
                        "texts": list(texts),
                        "min_run_chars": int(min_run_chars),
                        "chunk": chunk,
                    }
                )
                out = []
                for text in texts:
                    split = max(1, len(text) // 2)
                    labels = [html_id] * split + [css_id] * (len(text) - split)
                    probs = []
                    for idx in range(len(text)):
                        if idx < split:
                            probs.append({str(html_id): 0.99, str(css_id): 0.01})
                        else:
                            probs.append({str(html_id): 0.01, str(css_id): 0.99})
                    out.append(([], labels, probs, []))
                return out

        predictor = FakePredictor()
        samples = [
            ("html", 11, "abcd", "hash-a"),
            ("html", 12, "wxyzuv", "hash-b"),
        ]

        results = _build_snippets_for_samples_batch(
            samples=samples,
            predictor=predictor,
            max_candidates_per_sample=1,
            context_chars=16,
            min_score=0.0,
        )

        self.assertEqual(
            predictor.calls,
            [{"texts": ["abcd", "wxyzuv"], "min_run_chars": 1, "chunk": None}],
        )
        self.assertEqual(len(results), 2)

        sample_snippets_a, pred_segments_a = results[0]
        self.assertEqual(
            pred_segments_a,
            [
                {"start": 0, "end": 2, "label": "html"},
                {"start": 2, "end": 4, "label": "css"},
            ],
        )
        self.assertEqual(len(sample_snippets_a), 1)
        snippet_a, cand_a = sample_snippets_a[0]
        self.assertEqual(snippet_a.text, "abcd")
        self.assertEqual(snippet_a.metadata["sample_hash"], "hash-a")
        self.assertEqual(snippet_a.predicted_labels, ["html", "html", "css", "css"])
        self.assertEqual((cand_a.start, cand_a.end, cand_a.boundary), (0, 4, 2))

        sample_snippets_b, pred_segments_b = results[1]
        self.assertEqual(
            pred_segments_b,
            [
                {"start": 0, "end": 3, "label": "html"},
                {"start": 3, "end": 6, "label": "css"},
            ],
        )
        self.assertEqual(len(sample_snippets_b), 1)
        snippet_b, cand_b = sample_snippets_b[0]
        self.assertEqual(snippet_b.text, "wxyzuv")
        self.assertEqual(snippet_b.metadata["sample_hash"], "hash-b")
        self.assertEqual(snippet_b.predicted_labels, ["html", "html", "html", "css", "css", "css"])
        self.assertEqual((cand_b.start, cand_b.end, cand_b.boundary), (0, 6, 3))

    def test_round_parser_defaults_predict_batch_size_to_twelve(self) -> None:
        args = _build_parser().parse_args(
            [
                "--ckpt",
                "checkpoint.msgpack",
                "--store",
                "active_learning/label_store.sqlite",
            ]
        )
        self.assertEqual(args.predict_batch_size, 12)
        self.assertEqual(args.gemini_thinking_level, "medium")

    def test_iter_split_examples_uses_training_window_augmentation_for_train_split(self) -> None:
        data_root = ROOT / "tmp-data-root"
        train_dsets = {
            int(cfg.LANG2ID["python"]): object(),
            int(cfg.LANG2ID["javascript_typescript"]): object(),
        }
        padded = int(cfg.PAD_BYTE_ID)
        first_tokens = np.array([ord("a"), ord("b"), ord("c"), padded], dtype=np.int32)
        second_text = "# demo\n```js\nx\n```"
        second_tokens = np.array([*(ord(ch) for ch in second_text), padded], dtype=np.int32)

        first_meta = {
            "mode": "mixed",
            "requested_mode": "mixed",
            "samples": [
                {"language": "python", "final_bytes": 3},
                {"language": "javascript_typescript", "final_bytes": 1},
            ],
            "host": {"language": "python", "source": "demo.py", "final_bytes": 3},
            "line_injections": [],
            "markdown_blocks": [],
        }
        second_meta = {
            "mode": "markdown",
            "requested_mode": "markdown",
            "samples": [
                {"language": "text", "final_bytes": 7},
                {"language": "javascript_typescript", "final_bytes": 4},
            ],
            "line_injections": [],
            "markdown_blocks": [{"language": "javascript_typescript"}],
        }

        with patch(
            "active_learning.round.prepare_dsets_by_lang_with_splits",
            return_value={"train": train_dsets},
        ) as prep_mock, patch(
            "active_learning.round.make_training_window_with_metadata",
            side_effect=[
                (first_tokens, np.zeros_like(first_tokens, dtype=np.uint8), first_meta),
                (second_tokens, np.zeros_like(second_tokens, dtype=np.uint8), second_meta),
            ],
        ) as make_mock:
            samples = list(
                _iter_split_examples(
                    data_root,
                    "train",
                    ["python", "javascript_typescript"],
                    1,
                    rng=np.random.default_rng(0),
                    seen_hashes=set(),
                )
            )

        self.assertEqual(prep_mock.call_count, 1)
        self.assertEqual(make_mock.call_count, 2)
        self.assertEqual(len(samples), 2)

        lang_a, idx_a, text_a, hash_a, meta_a = samples[0]
        self.assertEqual(lang_a, "python")
        self.assertEqual(idx_a, 0)
        self.assertEqual(text_a, "abc")
        self.assertTrue(hash_a)
        self.assertEqual(meta_a["sampling_mode"], "training_window")
        self.assertEqual(meta_a["augmentation_mode"], "mixed")
        self.assertEqual(meta_a["source_langs"], ["python", "javascript_typescript"])

        lang_b, idx_b, text_b, hash_b, meta_b = samples[1]
        self.assertEqual(lang_b, "markdown")
        self.assertEqual(idx_b, 1)
        self.assertEqual(text_b, second_text)
        self.assertTrue(hash_b)
        self.assertEqual(meta_b["sampling_mode"], "training_window")
        self.assertEqual(meta_b["augmentation_mode"], "markdown")
        self.assertEqual(meta_b["source_langs"][0], "markdown")


if __name__ == "__main__":
    unittest.main()
