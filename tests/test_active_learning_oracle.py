from __future__ import annotations

import sys
import tempfile
import types as pytypes
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from active_learning.oracle import (
    BoundarySnippet,
    GeminiBoundaryOracle,
    OracleSegment,
    _compute_rate_limit_sleep_seconds,
    _extract_quota_id,
    _openrouter_model_name,
    _extract_retry_after_seconds,
    _is_rate_limited_error,
    _is_non_retriable_quota_error,
    _normalize_segments,
    _resolve_oracle_backend,
    merge_adjacent_segments,
)


class TestOracleUtils(unittest.TestCase):
    def test_merge_adjacent_same_label_segments(self) -> None:
        segments = [
            OracleSegment(3, 5, "html"),
            OracleSegment(0, 3, "html"),
            OracleSegment(5, 7, "javascript"),
            OracleSegment(7, 10, "javascript_typescript"),
            OracleSegment(10, 12, "css"),
        ]
        merged = merge_adjacent_segments(segments)
        self.assertEqual(
            merged,
            [
                OracleSegment(0, 5, "html"),
                OracleSegment(5, 10, "javascript_typescript"),
                OracleSegment(10, 12, "css"),
            ],
        )

    def test_normalize_segments_fallback_on_generic_collapse(self) -> None:
        text = "FROM python:3.12\nRUN pip install -r requirements.txt\n"
        snippet = BoundarySnippet(
            snippet_id="s1",
            text=text,
            global_start=0,
            global_end=len(text),
            boundary=10,
            predicted_labels=["dockerfile"] * len(text),
            metadata={"source_lang": "dockerfile"},
        )
        normalized = _normalize_segments(snippet, [OracleSegment(0, len(text), "text", "text")])
        self.assertEqual(normalized, [OracleSegment(0, len(text), "dockerfile", None)])

    def test_normalize_segments_trims_markdown_frontmatter_yaml_prefix(self) -> None:
        text = "---\ntitle: demo\n---\n\n# Heading\nBody\n"
        frontmatter_end = text.find("\n---\n", len("---\n")) + len("\n---\n")
        self.assertGreater(frontmatter_end, 0)
        snippet = BoundarySnippet(
            snippet_id="s2",
            text=text,
            global_start=0,
            global_end=len(text),
            boundary=frontmatter_end + 3,
            predicted_labels=["yaml"] * frontmatter_end + ["markdown"] * (len(text) - frontmatter_end),
            metadata={"source_lang": "markdown"},
        )
        normalized = _normalize_segments(snippet, [OracleSegment(0, len(text), "yaml", "yaml")])
        self.assertEqual(
            normalized,
            [
                OracleSegment(0, frontmatter_end, "yaml", None),
                OracleSegment(frontmatter_end, len(text), "markdown", None),
            ],
        )

    def test_rate_limit_error_detection(self) -> None:
        self.assertTrue(_is_rate_limited_error(RuntimeError("429 RESOURCE_EXHAUSTED: rate limit exceeded")))
        self.assertTrue(_is_rate_limited_error(RuntimeError("Too many requests; retry later")))
        self.assertFalse(_is_rate_limited_error(RuntimeError("socket timeout")))

    def test_retry_after_parsing_seconds_and_minutes(self) -> None:
        sec_err = RuntimeError("Rate limited; try again in 12.5 seconds")
        min_err = RuntimeError("Retry-After: 2m due to quota")
        self.assertEqual(_extract_retry_after_seconds(sec_err), 12.5)
        self.assertEqual(_extract_retry_after_seconds(min_err), 120.0)

    def test_non_retriable_quota_detection_from_per_day_quota_id(self) -> None:
        err = RuntimeError(
            "429 RESOURCE_EXHAUSTED. "
            "Quota exceeded; quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'."
        )
        self.assertEqual(
            _extract_quota_id(err),
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        )
        self.assertTrue(_is_non_retriable_quota_error(err))
        self.assertFalse(_is_non_retriable_quota_error(RuntimeError("429 rate limit; retry in 10s")))

    def test_rate_limit_sleep_prefers_retry_after(self) -> None:
        self.assertEqual(
            _compute_rate_limit_sleep_seconds(
                configured_default_seconds=65.0,
                retry_after_seconds=5.0,
            ),
            6.0,
        )
        self.assertEqual(
            _compute_rate_limit_sleep_seconds(
                configured_default_seconds=65.0,
                retry_after_seconds=None,
            ),
            65.0,
        )

    def test_resolve_oracle_backend_uses_openrouter_when_google_missing(self) -> None:
        with patch.dict("os.environ", {"OPEN_ROUTER_API_KEY": "or-key"}, clear=True):
            provider, key = _resolve_oracle_backend(None)
        self.assertEqual(provider, "openrouter")
        self.assertEqual(key, "or-key")

    def test_resolve_oracle_backend_prefers_google_when_both_are_set(self) -> None:
        with patch.dict(
            "os.environ",
            {"GOOGLE_API_KEY": "google-key", "OPEN_ROUTER_API_KEY": "or-key"},
            clear=True,
        ):
            provider, key = _resolve_oracle_backend(None)
        self.assertEqual(provider, "google")
        self.assertEqual(key, "google-key")

    def test_openrouter_model_name_maps_gemini_short_name(self) -> None:
        self.assertEqual(
            _openrouter_model_name("gemini-3-flash-preview"),
            "google/gemini-3-flash-preview",
        )
        self.assertEqual(
            _openrouter_model_name("google/gemini-3-flash-preview"),
            "google/gemini-3-flash-preview",
        )

    def test_openrouter_request_pins_google_ai_studio_provider(self) -> None:
        class _FakeResponse:
            status_code = 200
            reason_phrase = "OK"
            text = ""

            @staticmethod
            def json():
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"snippets":[{"snippet_id":"s1","segments":[{"label":"python","text":"abc"}]}]}'
                            }
                        }
                    ]
                }

        class _FakeClient:
            def __init__(self) -> None:
                self.last_payload = None

            def post(self, _url, headers=None, json=None):
                self.last_payload = json
                return _FakeResponse()

        oracle = GeminiBoundaryOracle.__new__(GeminiBoundaryOracle)
        oracle._openrouter_client = _FakeClient()
        oracle._provider_api_key = "or-key"
        oracle._openrouter_model = "google/gemini-3-flash-preview"

        with patch.dict("os.environ", {}, clear=True):
            _parts, _usage = oracle._request_segments_openrouter("hello")

        payload = oracle._openrouter_client.last_payload
        self.assertIsInstance(payload, dict)
        self.assertIn("provider", payload)
        provider = payload["provider"]
        self.assertIsInstance(provider, dict)
        self.assertEqual(provider.get("order"), ["google-ai-studio"])
        self.assertEqual(provider.get("allow_fallbacks"), False)

    def test_oracle_init_picks_openrouter_if_only_openrouter_key_exists(self) -> None:
        with patch.dict("os.environ", {"OPEN_ROUTER_API_KEY": "or-key"}, clear=True):
            oracle = GeminiBoundaryOracle(model="gemini-3-flash-preview", batch_size=2)
        self.assertEqual(getattr(oracle, "_provider", ""), "openrouter")
        self.assertEqual(oracle.model, "google/gemini-3-flash-preview")
        if getattr(oracle, "_openrouter_client", None) is not None:
            oracle._openrouter_client.close()  # type: ignore[attr-defined]

    def test_parse_segments_payload_accepts_text_chunks(self) -> None:
        raw = (
            '{"snippets":[{"snippet_id":"s1","segments":['
            '{"label":"python","text":"abc"},'
            '{"label":"sql","text":"DEF"}'
            ']}]}'
        )
        parsed = GeminiBoundaryOracle._parse_segments_payload(
            raw,
            snippet_text_by_id={"s1": "abcDEF"},
        )
        self.assertIn("s1", parsed)
        self.assertEqual(
            parsed["s1"],
            [
                OracleSegment(0, 3, "python", "python"),
                OracleSegment(3, 6, "sql", "sql"),
            ],
        )

    def test_parse_segments_payload_falls_back_to_offsets_on_text_mismatch(self) -> None:
        raw = (
            '{"snippets":[{"snippet_id":"s1","segments":['
            '{"start":0,"end":3,"label":"python","text":"zzz"},'
            '{"start":3,"end":6,"label":"sql","text":"yyy"}'
            ']}]}'
        )
        parsed = GeminiBoundaryOracle._parse_segments_payload(
            raw,
            snippet_text_by_id={"s1": "abcDEF"},
        )
        self.assertIn("s1", parsed)
        self.assertEqual(
            parsed["s1"],
            [
                OracleSegment(0, 3, "python", "python"),
                OracleSegment(3, 6, "sql", "sql"),
            ],
        )

    def test_build_prompt_contains_svg_inline_style_example(self) -> None:
        text = '<text id="text823" style="font-size:3.88584304px;fill:#00cedb"></text>'
        n = len(text)
        snippet = BoundarySnippet(
            snippet_id="svg-1",
            text=text,
            global_start=0,
            global_end=n,
            boundary=max(1, n // 2),
            predicted_labels=["svg"] * n,
            metadata={"source_lang": "svg"},
        )
        prompt = GeminiBoundaryOracle._build_batch_prompt([snippet])
        self.assertIn("Example (correct SVG inline-style split)", prompt)
        self.assertIn("font-size:3.88584304px;fill:#00cedb", prompt)
        self.assertIn("{'label':'svg','start':0,'end':7,'text':'style=\"'}", prompt)
        self.assertIn("{'label':'css','start':7,'end':110,'text':'font-size:3.88584304px", prompt)
        self.assertIn("'start':7,'end':110", prompt)

    def test_annotate_batch_recovers_missing_snippet_ids(self) -> None:
        snippet_a = BoundarySnippet(
            snippet_id="a",
            text="abcdef",
            global_start=0,
            global_end=6,
            boundary=3,
            predicted_labels=["python"] * 6,
            metadata={"source_lang": "python"},
        )
        snippet_b = BoundarySnippet(
            snippet_id="b",
            text="uvwxyz",
            global_start=0,
            global_end=6,
            boundary=3,
            predicted_labels=["python"] * 6,
            metadata={"source_lang": "python"},
        )

        oracle = GeminiBoundaryOracle.__new__(GeminiBoundaryOracle)
        oracle.model = "gemini-test"
        oracle.allowed_labels = []
        oracle.missing_snippet_retries = 2

        calls = {"n": 0}

        def fake_request(self, snippets):
            calls["n"] += 1
            if len(snippets) == 2:
                return (
                    {"a": [OracleSegment(0, 6, "python", "python")]},
                    ['{"snippets":[{"snippet_id":"a","segments":[{"start":0,"end":6,"label":"python"}]}]}'],
                    None,
                    "al_oracle_ok",
                    None,
                    0,
                )
            return (
                {"b": [OracleSegment(0, 6, "python", "python")]},
                ['{"snippets":[{"snippet_id":"b","segments":[{"start":0,"end":6,"label":"python"}]}]}'],
                None,
                "al_oracle_ok",
                None,
                0,
            )

        oracle._request_segments = pytypes.MethodType(fake_request, oracle)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("active_learning.oracle.llm_requestor.LOG_OUTPUT_DIR", Path(tmpdir)):
                finalized, sources, batch_info = oracle._annotate_batch([snippet_a, snippet_b])

        self.assertIn("a", finalized)
        self.assertIn("b", finalized)
        self.assertEqual(sources.get("a"), "model")
        self.assertEqual(sources.get("b"), "model")
        self.assertEqual(int(batch_info.get("initial_missing_snippets", -1)), 1)
        self.assertEqual(int(batch_info.get("recovered_missing_snippets", -1)), 1)
        self.assertGreaterEqual(int(batch_info.get("missing_retry_requests", 0)), 1)
        self.assertGreaterEqual(calls["n"], 2)

    def test_request_segments_fails_fast_on_non_retriable_quota(self) -> None:
        snippet = BoundarySnippet(
            snippet_id="q1",
            text="abc",
            global_start=0,
            global_end=3,
            boundary=1,
            predicted_labels=["python", "python", "python"],
            metadata={"source_lang": "python"},
        )

        class _FakeModels:
            def __init__(self) -> None:
                self.calls = 0

            def generate_content_stream(self, **kwargs):
                self.calls += 1
                raise RuntimeError(
                    "429 RESOURCE_EXHAUSTED. quotaId': "
                    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'."
                )

        class _FakeClient:
            def __init__(self) -> None:
                self.models = _FakeModels()

        oracle = GeminiBoundaryOracle.__new__(GeminiBoundaryOracle)
        oracle.model = "gemini-test"
        oracle.rate_limit_sleep_seconds = 65.0
        oracle.rate_limit_max_retries = 8
        oracle._client = _FakeClient()

        with patch("active_learning.oracle.time.sleep") as sleep_mock:
            parsed, _parts, _usage, status, error, throttled = oracle._request_segments([snippet])

        self.assertEqual(parsed, {})
        self.assertEqual(status, "al_oracle_failed")
        self.assertIn("non-retriable", str(error))
        self.assertEqual(int(throttled), 0)
        self.assertEqual(int(oracle._client.models.calls), 1)
        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
