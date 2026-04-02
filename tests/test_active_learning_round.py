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
from active_learning.acquisition import CandidateSpan
from active_learning.oracle import BoundarySnippet
from active_learning.round import (
    _build_parser,
    _build_snippets_for_samples_batch,
    _configure_oracle_parallel_requests,
    _iter_split_examples,
    _limit_snippets_for_oracle_requests,
    _make_snippet_id,
    run_one_round,
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
    def test_configure_oracle_parallel_requests_prefers_explicit_setting(self) -> None:
        class Oracle:
            def __init__(self) -> None:
                self.max_parallel_requests = 1

        oracle = Oracle()
        parallel = _configure_oracle_parallel_requests(
            oracle,
            max_oracle_requests=80,
            gemini_parallel_requests=16,
        )
        self.assertEqual(parallel, 16)
        self.assertEqual(oracle.max_parallel_requests, 16)

    def test_configure_oracle_parallel_requests_uses_request_cap(self) -> None:
        class Oracle:
            def __init__(self) -> None:
                self.max_parallel_requests = 1

        oracle = Oracle()
        parallel = _configure_oracle_parallel_requests(
            oracle,
            max_oracle_requests=5,
        )
        self.assertEqual(parallel, 5)
        self.assertEqual(oracle.max_parallel_requests, 5)

    def test_configure_oracle_parallel_requests_defaults_to_one_when_unlimited(self) -> None:
        class Oracle:
            pass

        oracle = Oracle()
        parallel = _configure_oracle_parallel_requests(
            oracle,
            max_oracle_requests=None,
        )
        self.assertEqual(parallel, 1)
        self.assertFalse(hasattr(oracle, "max_parallel_requests"))

    def test_parser_accepts_gemini_parallel_requests(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--ckpt",
                "dummy.msgpack",
                "--gemini-parallel-requests",
                "16",
            ]
        )
        self.assertEqual(args.gemini_parallel_requests, 16)

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

    def test_build_snippets_for_samples_batch_full_files_queries_whole_sample(self) -> None:
        html_id = int(cfg.LANG2ID["html"])
        css_id = int(cfg.LANG2ID["css"])

        class FakePredictor:
            def segment_texts(self, texts, min_run_chars=6, chunk=None):
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
        samples = [("html", 11, "abcd", "hash-a", {"sampling_mode": "training_full_sequence"})]
        results = _build_snippets_for_samples_batch(
            samples=samples,
            predictor=predictor,
            max_candidates_per_sample=3,
            context_chars=16,
            min_score=0.0,
            full_files=True,
        )

        sample_snippets, pred_segments = results[0]
        self.assertEqual(
            pred_segments,
            [
                {"start": 0, "end": 2, "label": "html"},
                {"start": 2, "end": 4, "label": "css"},
            ],
        )
        self.assertEqual(len(sample_snippets), 1)
        snippet, cand = sample_snippets[0]
        self.assertEqual(snippet.text, "abcd")
        self.assertTrue(bool(snippet.metadata["full_file_mode"]))
        self.assertEqual(snippet.metadata["sample_hash"], "hash-a")
        self.assertGreaterEqual(len(snippet.metadata["trigger_ranges"]), 1)
        self.assertEqual(snippet.global_start, 0)
        self.assertEqual(snippet.global_end, 4)
        self.assertEqual(cand.boundary, 2)

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

    def test_run_one_round_skips_round_when_oracle_is_unavailable(self) -> None:
        class FakePredictor:
            inference_batch_size = 1

        class FakeStore:
            def __init__(self, _path: str) -> None:
                self.path = _path

            def existing_sample_hashes(self):
                return set()

            def add_inference_samples_many(self, rows):
                return len(rows)

        class UnavailableOracle:
            name = "gemini"
            model = "gemini-3-flash-preview"
            batch_size = 4

            def annotate(self, _snippets):
                raise RuntimeError(
                    "Oracle failed completely. Error: 503 UNAVAILABLE. "
                    "{'error': {'code': 503, 'message': 'This model is currently experiencing high demand.', "
                    "'status': 'UNAVAILABLE'}}"
                )

        snippet = _snippet("s-unavailable")
        candidate = CandidateSpan(
            start=0,
            end=3,
            boundary=1,
            score=0.9,
            entropy_mean=0.4,
            flip_rate=0.2,
            left_label=int(cfg.LANG2ID["python"]),
            right_label=int(cfg.LANG2ID["python"]),
        )

        with patch("active_learning.round._build_predictor", return_value=FakePredictor()), patch(
            "active_learning.round.LabelStore",
            FakeStore,
        ), patch(
            "active_learning.round._iter_split_examples",
            return_value=iter([("python", 0, "abc", "hash-a", {})]),
        ), patch(
            "active_learning.round._build_snippets_for_samples_batch",
            return_value=[([(snippet, candidate)], [{"start": 0, "end": 3, "label": "python"}])],
        ):
            summary = run_one_round(
                ckpt_path="checkpoint.msgpack",
                data_root="downloader/arrow_out",
                split="train",
                langs=["python"],
                store_path="active_learning/label_store.sqlite",
                oracle=UnavailableOracle(),
                max_samples_per_lang=1,
                max_candidates_per_sample=1,
                predictor_kwargs={"predict_batch_size": 1},
            )

        self.assertEqual(summary["status"], "oracle_unavailable")
        self.assertEqual(summary["stored"], 0)
        self.assertEqual(summary["inference_samples"], 1)
        self.assertEqual(summary["oracle_model_outputs"], 0)
        self.assertIn("503", str(summary["oracle_error"]))

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
