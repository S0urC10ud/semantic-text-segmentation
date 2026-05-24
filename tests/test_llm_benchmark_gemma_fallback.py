from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.llm_benchmark import label_test_set as lts


class _FakeOpenRouterClient:
    pass


class TestGemmaFallbackRouting(unittest.TestCase):
    def test_google_blocking_error_detection_is_specific(self) -> None:
        self.assertTrue(lts._is_google_blocking_error(RuntimeError("429 RESOURCE_EXHAUSTED quota")))
        self.assertTrue(lts._is_google_blocking_error(RuntimeError("503 service unavailable")))
        self.assertTrue(lts._is_google_blocking_error(RuntimeError("500 INTERNAL internal error encountered")))
        self.assertTrue(lts._is_google_blocking_error(RuntimeError("504 DEADLINE_EXCEEDED")))
        self.assertTrue(lts._is_google_blocking_error(RuntimeError("The read operation timed out")))
        self.assertFalse(lts._is_google_blocking_error(RuntimeError("API key not valid")))

    def test_google_text_collector_skips_thought_parts(self) -> None:
        text = lts._non_thought_text(
            [
                SimpleNamespace(text="analysis", thought=True),
                SimpleNamespace(text="<SINGLE-", thought=None),
                SimpleNamespace(text="SEGMENT:python>", thought=False),
            ]
        )
        self.assertEqual(text, "<SINGLE-SEGMENT:python>")

    def test_internal_errors_use_short_cooldown(self) -> None:
        self.assertEqual(lts._cooldown_seconds_for_error(RuntimeError("500 INTERNAL internal error"), 65), 5.0)
        self.assertEqual(lts._cooldown_seconds_for_error(RuntimeError("The read operation timed out"), 65), 5.0)

    def test_max_parallel_is_capped_at_twenty_five(self) -> None:
        self.assertEqual(lts.validate_max_parallel(25), 25)
        with self.assertRaises(ValueError):
            lts.validate_max_parallel(26)
        with self.assertRaises(ValueError):
            lts.validate_max_parallel(0)

    def test_retryable_cache_statuses_are_not_terminal(self) -> None:
        self.assertTrue(lts.is_cache_terminal({"status": "ok"}))
        self.assertTrue(lts.is_cache_terminal({"status": "failed_parse"}))
        self.assertFalse(lts.is_cache_terminal({"status": "failed_api"}))
        self.assertFalse(lts.is_cache_terminal({"status": "failed_rate_limit"}))
        self.assertFalse(lts.is_cache_terminal({"status": "failed_timeout"}))
        self.assertFalse(lts.is_cache_terminal(None))

    def test_google_thought_leak_parse_failures_are_retryable(self) -> None:
        rec = {
            "status": "failed_parse",
            "error": "no_content_blocks",
            "response_text": "*   Input: A Python file.\n*   Content analysis: pure Python\n<SINGLE-SEGMENT:python>",
        }
        self.assertTrue(lts.is_thought_leak_parse_failure(rec))
        self.assertFalse(lts.is_cache_terminal(rec))

    def test_google_pool_waits_for_per_key_rpm_before_declaring_blocked(self) -> None:
        clock = [100.0]
        pool = lts.GoogleKeyPool(["k0"], cooldown_seconds=60, requests_per_minute=15)

        with patch.object(lts.time, "monotonic", side_effect=lambda: clock[0]):
            idx, client, state = pool.acquire()
            self.assertEqual((idx, client, state), (0, "k0", "acquired"))
            pool.release(idx)

            idx, client, state = pool.acquire()
            self.assertEqual((idx, client, state), (None, None, "busy"))
            self.assertGreater(pool.snapshot()["rate_wait"][0]["remaining_s"], 3.9)

            clock[0] += 4.0
            idx, client, state = pool.acquire()
            self.assertEqual((idx, client, state), (0, "k0", "acquired"))

    def test_rotates_to_second_google_key_before_openrouter(self) -> None:
        sample = {
            "benchmark_id": "realistic::x",
            "task": "realistic",
            "host": "python",
            "content": "abc",
            "content_sha256": "sha",
            "input_characters": 3,
        }
        pool = lts.GoogleKeyPool(["k0", "k1"], cooldown_seconds=60)
        calls: list[str] = []

        def fake_google(*, client, **kwargs):
            calls.append(client)
            if client == "k0":
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
            return "<SINGLE-SEGMENT:python>", lts._UsageMetadata(
                prompt_token_count=10,
                candidates_token_count=1,
                total_token_count=11,
            )

        def fake_parse(*, source_text, generated_code):
            return ([{"type": "python", "content": source_text}], {}, None)

        with patch.object(lts, "call_google_model", side_effect=fake_google), \
             patch.object(lts, "request_openrouter") as openrouter, \
             patch.object(lts, "parse_and_heal", side_effect=fake_parse):
            rec = lts.process_sample_google_openrouter_fallback(
                sample,
                google_pool=pool,
                openrouter_client=_FakeOpenRouterClient(),
                openrouter_api_key="or-key",
                types_mod=object(),
                google_model="gemma-4-31b-it",
                openrouter_model="google/gemma-4-31b-it",
                system_prompt="sys",
                user_prompt="user",
                thinking_budget=0,
                per_sample_timeout_s=30,
                max_google_attempts=10,
                secrets=["or-key"],
            )

        self.assertEqual(calls, ["k0", "k1"])
        openrouter.assert_not_called()
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["provider"], "google")
        self.assertEqual(rec["google_key_index"], 1)
        self.assertEqual(len(rec["google_attempts"]), 1)

    def test_falls_back_to_openrouter_only_after_all_google_keys_block(self) -> None:
        sample = {
            "benchmark_id": "realistic::x",
            "task": "realistic",
            "host": "python",
            "content": "abc",
            "content_sha256": "sha",
            "input_characters": 3,
        }
        pool = lts.GoogleKeyPool(["k0", "k1"], cooldown_seconds=60)

        def fake_google(**kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")

        def fake_openrouter(**kwargs):
            return "<SINGLE-SEGMENT:python>", lts._UsageMetadata(
                prompt_token_count=1000,
                candidates_token_count=1000,
                total_token_count=2000,
            )

        def fake_parse(*, source_text, generated_code):
            return ([{"type": "python", "content": source_text}], {}, None)

        with patch.object(lts, "call_google_model", side_effect=fake_google), \
             patch.object(lts, "request_openrouter", side_effect=fake_openrouter) as openrouter, \
             patch.object(lts, "parse_and_heal", side_effect=fake_parse):
            rec = lts.process_sample_google_openrouter_fallback(
                sample,
                google_pool=pool,
                openrouter_client=_FakeOpenRouterClient(),
                openrouter_api_key="or-key",
                types_mod=object(),
                google_model="gemma-4-31b-it",
                openrouter_model="google/gemma-4-31b-it",
                system_prompt="sys",
                user_prompt="user",
                thinking_budget=0,
                per_sample_timeout_s=30,
                max_google_attempts=10,
                secrets=["or-key"],
            )

        self.assertEqual(openrouter.call_count, 1)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["provider"], "openrouter")
        self.assertEqual(rec["fallback_reason"], "all_google_keys_blocked")
        self.assertEqual(len(rec["google_attempts"]), 2)
        self.assertGreater(rec["cost_usd"], 0.0)


class _TinyDataset(list):
    def select(self, indices):
        return _TinyDataset([self[i] for i in indices])


class TestLLMScoringHelpers(unittest.TestCase):
    def test_partial_filter_matches_label_test_set_percentiles(self) -> None:
        from evaluation import score_llm_predictions as scorer

        rows = _TinyDataset(
            [
                {"example_id": f"id-{i}", "metadata_json": json.dumps({"host_lang": "python"})}
                for i in range(25)
            ]
        )
        selected = scorer._filter_dataset_by_partial("realistic", rows, 10)
        expected = [
            row for row in rows
            if scorer._percentile_key("realistic", "python", row["example_id"]) < 0.10
        ]
        self.assertEqual(list(selected), expected)


if __name__ == "__main__":
    unittest.main()
