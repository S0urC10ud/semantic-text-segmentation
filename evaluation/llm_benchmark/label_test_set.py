#!/usr/bin/env python3
"""Label every row in ``evaluation/test/`` with an LLM (default Flash) and
score against the Pro-labelled ground truth that lives in the same arrow
datasets.

Supports incremental partial runs:

    --partial 10   # label first 10% of the benchmark (evenly across task x host)
    --partial 20   # label first 20% cumulative; second invocation skips cached

Rows are ordered deterministically by ``(task, host_lang, sha)`` so partial
fractions are evenly spread across both axes. Cache + live cost telemetry +
hard $ cap + per-sample timeout follow the same conventions as
``label_unseen_set.py``.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import getpass
import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downloader.utils.llm_requestor import (  # noqa: E402
    MODEL_PRICING_USD_PER_MTOKENS,
    _effective_rate,
    _heal_newline_discrepancies,
    _heal_small_replacements,
    _maybe_apply_literal_escape_fix,
    _maybe_encode_literal_sequences,
    _maybe_restore_backslashes,
    _remove_empty_segments,
    _require_exact_source_match,
    _serialize_usage_metadata,
    _strip_embedded_markers_from_segments,
    _usd_cost,
    extract_segments_from_content_blocks,
    load_dense_prompt,
    verify_segments_locally,
    LocalVerificationError,
)
from evaluation.llm_benchmark.label_unseen_set import (  # noqa: E402
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_THINKING_BUDGET,
    DEFAULT_PER_SAMPLE_TIMEOUT_S,
    DEFAULT_TELEMETRY_EVERY,
    RATE_LIMIT_SLEEP_S,
    GeminiSampleTimeout,
    SYNTHETIC_HOSTS,
    call_gemini,
    compute_cost_usd,
    detect_single_segment_shortcut,
    load_prompt_template,
    parse_and_heal,
    synthesize_encoding_record,
    write_cache_line,
    slugify_model,
    new_run_id,
    _is_rate_limit_error,
)

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it"
DEFAULT_PROVIDER = "google"
GOOGLE_POOL_PROVIDER = "google-pool"
FALLBACK_PROVIDER = "google-openrouter-fallback"
OPENROUTER_PROVIDER = "openrouter"
LLAMACPP_PROVIDER = "llamacpp"
DEFAULT_TEST_ROOT = REPO_ROOT / "evaluation" / "test"
DEFAULT_RUN_DIR = REPO_ROOT / "evaluation" / "llm_benchmark" / "test_runs"
DEFAULT_MAX_SPEND_USD = 50.0
DEFAULT_MAX_PARALLEL_TEST = 5
MAX_PARALLEL_TEST = 25
MAX_GOOGLE_KEYS = 5
DEFAULT_GOOGLE_RPM_PER_KEY = 15.0
DEFAULT_GOOGLE_TIMEOUT_S = 45.0
DEFAULT_GOOGLE_INPUT_TPM = 0
RETRYABLE_CACHE_STATUSES = frozenset({"failed_api", "failed_rate_limit", "failed_timeout"})
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_APP_NAME = "text-segmentation-thesis-eval"
DEFAULT_LOCAL_OPENAI_URL = "http://127.0.0.1:8080/v1/chat/completions"

OPENROUTER_PRICING_USD_PER_MTOKENS = {
    "google/gemma-4-31b-it": {
        "prompt": 0.12,
        "response": 0.37,
    },
}

GOOGLE_BLOCKING_HINTS = (
    "429",
    "resource_exhausted",
    "rate_limit",
    "rate limit",
    "quota",
    "too many requests",
    "capacity",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "internal error",
    "internal.",
    "500",
    "503",
    "504",
    "deadline expired",
    "deadline_exceeded",
    "timeout",
    "timed out",
    "read operation timed out",
)


def _percentile_key(task: str, host: str, example_id: str) -> float:
    """Deterministic [0, 1) percentile for a row from (task, host, example_id)."""
    h = hashlib.sha256(f"{task}::{host}::{example_id}".encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big")
    return n / (1 << 64)


def _row_host(row: dict) -> str:
    try:
        meta = json.loads(row["metadata_json"])
    except Exception:
        return "?"
    return meta.get("host_lang") or meta.get("first_lang") or "?"


def load_test_rows(test_root: Path) -> list[dict]:
    """Flatten all arrow datasets under test_root into a deterministic list of
    sample dicts ``{benchmark_id, task, host, content, percentile}``.
    """
    from datasets import load_from_disk

    out: list[dict] = []
    for task_dir in sorted(test_root.iterdir()):
        if not task_dir.is_dir():
            continue
        try:
            ds = load_from_disk(str(task_dir))
        except Exception:
            continue
        task = task_dir.name
        n = len(ds)
        for i in range(n):
            row = ds[i]
            host = _row_host(row)
            example_id = str(row["example_id"])
            pct = _percentile_key(task, host, example_id)
            out.append({
                "benchmark_id": f"{task}::{example_id}",
                "task": task,
                "host": host,
                "content": row["content"],
                "input_characters": len(row["content"]),
                "content_sha256": hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
                "percentile": pct,
            })
    return out


def filter_by_partial(rows: list[dict], partial_pct: float) -> list[dict]:
    """Keep rows whose percentile < ``partial_pct``/100, preserving order
    grouped by (task, host) so dispatch stays balanced across hosts and tasks.
    """
    p = float(partial_pct) / 100.0
    selected = [r for r in rows if r["percentile"] < p]
    # Order: round-robin across (task, host) buckets so dispatch is even.
    by_bucket: dict[tuple[str, str], list[dict]] = {}
    for r in selected:
        by_bucket.setdefault((r["task"], r["host"]), []).append(r)
    bucket_order = sorted(by_bucket.keys())
    interleaved: list[dict] = []
    pointers = {b: 0 for b in bucket_order}
    while True:
        progressed = False
        for b in bucket_order:
            idx = pointers[b]
            lst = by_bucket[b]
            if idx < len(lst):
                interleaved.append(lst[idx])
                pointers[b] = idx + 1
                progressed = True
        if not progressed:
            break
    return interleaved


def load_cache(cache_path: Path) -> tuple[dict[str, dict], float]:
    cache: dict[str, dict] = {}
    spent = 0.0
    if not cache_path.is_file():
        return cache, spent
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            bid = rec.get("benchmark_id")
            if not bid:
                continue
            cache[bid] = rec
            cost = rec.get("cost_usd")
            if isinstance(cost, (int, float)):
                spent += float(cost)
    return cache, spent


def is_cache_terminal(record: Optional[dict]) -> bool:
    if not record:
        return False
    status = str(record.get("status") or "")
    if status in RETRYABLE_CACHE_STATUSES:
        return False
    if status == "failed_parse" and is_thought_leak_parse_failure(record):
        return False
    return True


def is_thought_leak_parse_failure(record: dict) -> bool:
    """Detect parse failures caused by accidentally caching Google thought text.

    Older cache rows from the first Gemma run included text parts whose
    ``part.thought`` flag was true. Those rows usually start with markdown-ish
    analysis and may contain the actual shortcut or marker later, so retry them
    after the collector fix rather than treating them as model-output failures.
    """
    if str(record.get("status") or "") != "failed_parse":
        return False
    if str(record.get("error") or "") != "no_content_blocks":
        return False
    text = str(record.get("response_text") or "").lstrip()
    if not text:
        return False
    if text.startswith("<SINGLE-SEGMENT:") or text.startswith("<CONTENT-TYPE:"):
        return False
    thoughtish_prefixes = ("*   Input:", "- Input:", "Input:", "Analysis:", "Plan:")
    return text.startswith(thoughtish_prefixes) and (
        "<SINGLE-SEGMENT:" in text or "<CONTENT-TYPE:" in text or "Content analysis" in text
    )


def _redact_secrets(text: object, secrets: Iterable[str] = ()) -> str:
    out = str(text)
    for secret in secrets:
        s = str(secret or "")
        if s:
            out = out.replace(s, "<redacted>")
    return out


def load_key_file(path: Optional[Path]) -> list[str]:
    if path is None:
        return []
    keys: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            keys.append(stripped)
    return keys


def collect_google_api_keys(args: argparse.Namespace) -> list[str]:
    keys: list[str] = []
    keys.extend(str(k).strip() for k in (args.google_api_key or []) if str(k).strip())
    keys.extend(load_key_file(args.google_api_keys_file))
    if not keys:
        env_key = str(os.environ.get("GOOGLE_API_KEY", "")).strip()
        if env_key:
            keys.append(env_key)

    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def resolve_openrouter_api_key(args: argparse.Namespace) -> str:
    explicit = str(args.openrouter_api_key or "").strip()
    if explicit:
        return explicit
    file_keys = load_key_file(args.openrouter_api_key_file)
    if file_keys:
        return file_keys[0]
    for env_name in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
        value = str(os.environ.get(env_name, "")).strip()
        if value:
            return value
    return ""


def _is_google_blocking_error(err: BaseException) -> bool:
    text = (str(err) or "").lower()
    return any(hint in text for hint in GOOGLE_BLOCKING_HINTS)


def validate_max_parallel(value: int) -> int:
    parallel = int(value)
    if parallel <= 0 or parallel > MAX_PARALLEL_TEST:
        raise ValueError(f"--max-parallel must be in [1, {MAX_PARALLEL_TEST}]")
    return parallel


def _extract_retry_after_seconds(err: BaseException) -> Optional[float]:
    text = str(err) or ""
    patterns = (
        r"retry[- ]after[:=]\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>s|sec|seconds?|m|min|minutes?)?",
        r"try again in\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>s|sec|seconds?|m|min|minutes?)?",
        r"retry in\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>s|sec|seconds?|m|min|minutes?)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = float(match.group("num"))
        unit = (match.groupdict().get("unit") or "s").lower()
        if unit.startswith("m"):
            value *= 60.0
        return value
    return None


def _cooldown_seconds_for_error(err: BaseException, default_seconds: float) -> float:
    retry_after = _extract_retry_after_seconds(err)
    if retry_after is not None and retry_after > 0:
        return max(1.0, retry_after + 1.0)
    text = (str(err) or "").lower()
    if (
        "504" in text
        or "deadline expired" in text
        or "deadline_exceeded" in text
        or "timeout" in text
        or "timed out" in text
    ):
        return 5.0
    if (
        "500" in text
        or "internal error" in text
        or "internal." in text
    ):
        return 5.0
    return max(1.0, float(default_seconds))


def _non_thought_text(parts: Iterable[object]) -> str:
    out: list[str] = []
    for part in parts:
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            out.append(text)
    return "".join(out)


class _UsageMetadata:
    def __init__(
        self,
        *,
        prompt_token_count: Optional[int] = None,
        candidates_token_count: Optional[int] = None,
        total_token_count: Optional[int] = None,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count


class GoogleAttemptLimitError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: list[dict[str, object]],
        google_pool: dict[str, Any],
        google_token_budget: Optional[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.google_pool = google_pool
        self.google_token_budget = google_token_budget


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_openrouter_cost_usd(model: str, usage_metadata: object) -> float:
    pricing = OPENROUTER_PRICING_USD_PER_MTOKENS.get(str(model))
    if not pricing or usage_metadata is None:
        return 0.0
    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    response_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
    return float(
        (prompt_tokens / 1_000_000.0) * float(pricing["prompt"])
        + (response_tokens / 1_000_000.0) * float(pricing["response"])
    )


def _message_content_as_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return str(text) if isinstance(text, str) else ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif isinstance(item, str) and item:
                parts.append(item)
        return "".join(parts)
    return ""


class GoogleKeyPool:
    """Thread-safe Google client pool with per-key request-start pacing."""

    def __init__(
        self,
        clients: list[object],
        *,
        cooldown_seconds: float = RATE_LIMIT_SLEEP_S,
        requests_per_minute: Optional[float] = None,
        max_inflight_per_key: Optional[int] = None,
    ) -> None:
        self.clients = list(clients)
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self.requests_per_minute = (
            float(requests_per_minute)
            if requests_per_minute is not None and float(requests_per_minute) > 0
            else None
        )
        self._min_interval_s = (
            60.0 / self.requests_per_minute
            if self.requests_per_minute is not None
            else 0.0
        )
        self.max_inflight_per_key = (
            max(1, int(max_inflight_per_key))
            if max_inflight_per_key is not None
            else None
        )
        self._blocked_until = [0.0 for _ in self.clients]
        self._block_reasons = ["" for _ in self.clients]
        self._next_available_at = [0.0 for _ in self.clients]
        self._in_use_counts = [0 for _ in self.clients]
        self._lock = Lock()

    def acquire(
        self,
        *,
        exclude_indices: Optional[set[int]] = None,
    ) -> tuple[Optional[int], Optional[object], str]:
        excluded = exclude_indices or set()
        with self._lock:
            now = time.monotonic()
            for idx, client in enumerate(self.clients):
                if idx in excluded:
                    continue
                if (
                    self.max_inflight_per_key is not None
                    and self._in_use_counts[idx] >= self.max_inflight_per_key
                ):
                    continue
                if self._blocked_until[idx] > now:
                    continue
                if self._next_available_at[idx] > now:
                    continue
                self._in_use_counts[idx] += 1
                if self._min_interval_s > 0:
                    self._next_available_at[idx] = now + self._min_interval_s
                return idx, client, "acquired"
            candidate_indices = [
                idx for idx in range(len(self.clients)) if idx not in excluded
            ]
            if not candidate_indices:
                return None, None, "excluded"
            if any(self._in_use_counts[idx] > 0 for idx in candidate_indices):
                return None, None, "busy"
            if any(
                self._blocked_until[idx] <= now < self._next_available_at[idx]
                for idx in candidate_indices
            ):
                return None, None, "busy"
            return None, None, "blocked"

    def release(self, idx: Optional[int]) -> None:
        if idx is None:
            return
        with self._lock:
            idx_i = int(idx)
            self._in_use_counts[idx_i] = max(0, self._in_use_counts[idx_i] - 1)

    def block(self, idx: int, *, reason: str, seconds: Optional[float] = None) -> None:
        cooldown = self.cooldown_seconds if seconds is None else max(1.0, float(seconds))
        with self._lock:
            self._blocked_until[int(idx)] = time.monotonic() + cooldown
            self._block_reasons[int(idx)] = str(reason)[:500]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            return {
                "key_count": len(self.clients),
                "in_use": [
                    idx for idx, count in enumerate(self._in_use_counts) if count > 0
                ],
                "in_use_counts": list(self._in_use_counts),
                "blocked": [
                    {
                        "google_key_index": idx,
                        "remaining_s": max(0.0, self._blocked_until[idx] - now),
                        "reason": self._block_reasons[idx],
                    }
                    for idx in range(len(self.clients))
                    if self._blocked_until[idx] > now
                ],
                "rate_wait": [
                    {
                        "google_key_index": idx,
                        "remaining_s": max(0.0, self._next_available_at[idx] - now),
                    }
                    for idx in range(len(self.clients))
                    if self._blocked_until[idx] <= now < self._next_available_at[idx]
                ],
                "requests_per_minute_per_key": self.requests_per_minute,
                "max_inflight_per_key": self.max_inflight_per_key,
            }


class TokenBudgetLimiter:
    """Simple rolling-window limiter for approximate prompt/input tokens."""

    def __init__(self, tokens_per_minute: int) -> None:
        self.tokens_per_minute = max(0, int(tokens_per_minute))
        self._events: deque[tuple[float, int]] = deque()
        self._lock = Lock()

    def acquire(self, tokens: int) -> None:
        if self.tokens_per_minute <= 0:
            return
        charge = max(1, min(int(tokens), self.tokens_per_minute))
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= 60.0:
                    self._events.popleft()
                used = sum(t for _, t in self._events)
                if used + charge <= self.tokens_per_minute:
                    self._events.append((now, charge))
                    return
                wait_s = max(0.05, 60.0 - (now - self._events[0][0]))
            time.sleep(min(wait_s, 1.0))

    def snapshot(self) -> dict[str, Any]:
        if self.tokens_per_minute <= 0:
            return {"tokens_per_minute": 0, "used_last_minute": 0, "queued_events": 0}
        with self._lock:
            now = time.monotonic()
            while self._events and now - self._events[0][0] >= 60.0:
                self._events.popleft()
            return {
                "tokens_per_minute": self.tokens_per_minute,
                "used_last_minute": sum(t for _, t in self._events),
                "queued_events": len(self._events),
            }


def estimate_google_input_tokens(*parts: str) -> int:
    # Conservative enough for quota pacing without needing provider tokenizers.
    chars = sum(len(p or "") for p in parts)
    return max(1, int(chars / 3.5) + 128)


def call_google_model(
    *,
    client,
    types_mod,
    model: str,
    content: str,
    system_prompt: str,
    user_prompt: str,
    thinking_budget: int,
    per_sample_timeout_s: float,
) -> tuple[str, object]:
    contents = [
        types_mod.Content(
            role="user",
            parts=[types_mod.Part.from_text(text=user_prompt + content + "\n</INPUT>\n")],
        )
    ]
    config_kwargs: dict[str, object] = {
        "system_instruction": system_prompt,
        "temperature": 0.0,
    }
    if per_sample_timeout_s > 0 and hasattr(types_mod, "HttpOptions"):
        # Google documents Gemma examples with the non-streaming endpoint. Keep
        # an SDK-level timeout so a wedged request still returns control.
        config_kwargs["http_options"] = types_mod.HttpOptions(
            timeout=int(float(per_sample_timeout_s) * 1000)
        )
    # Gemma 4's Gemini API surface treats thinking as an on/off feature. The
    # default benchmark path keeps it off by omitting thinking_config entirely.
    if int(thinking_budget) != 0:
        config_kwargs["thinking_config"] = types_mod.ThinkingConfig(
            thinking_budget=thinking_budget
        )
    config = types_mod.GenerateContentConfig(**config_kwargs)

    start = time.monotonic()
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    elapsed = time.monotonic() - start
    if per_sample_timeout_s > 0 and elapsed > per_sample_timeout_s:
        raise GeminiSampleTimeout(f"sample exceeded {per_sample_timeout_s:.1f}s wall-clock")

    usage_metadata = getattr(response, "usage_metadata", None)
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content_obj = getattr(candidates[0], "content", None)
        parts = getattr(content_obj, "parts", None) or []
        text = _non_thought_text(parts)
        if text:
            return text, usage_metadata

    text = getattr(response, "text", None)
    return text if isinstance(text, str) else "", usage_metadata


def request_openrouter(
    *,
    client,
    api_key: str,
    model: str,
    content: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, object]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-OpenRouter-Title": OPENROUTER_APP_NAME,
    }
    referer = str(os.environ.get("OPENROUTER_HTTP_REFERER", "")).strip()
    if referer:
        headers["HTTP-Referer"] = referer

    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + content + "\n</INPUT>\n"},
        ],
        "temperature": 0.0,
        "reasoning": {"effort": "none", "exclude": True},
    }
    resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"OpenRouter {resp.status_code} {getattr(resp, 'reason_phrase', '')}: {resp.text}"
        )
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("OpenRouter returned a non-object JSON payload.")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenRouter response missing choices: {json.dumps(data)[:800]}")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"OpenRouter first choice malformed: {json.dumps(first)[:800]}")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"OpenRouter message missing in first choice: {json.dumps(first)[:800]}")
    text = _message_content_as_text(message.get("content"))
    if not text.strip():
        raise RuntimeError(f"OpenRouter returned empty message content: {json.dumps(first)[:800]}")
    usage_metadata = None
    usage = data.get("usage")
    if isinstance(usage, dict):
        usage_metadata = _UsageMetadata(
            prompt_token_count=_safe_int(usage.get("prompt_tokens")),
            candidates_token_count=_safe_int(usage.get("completion_tokens")),
            total_token_count=_safe_int(usage.get("total_tokens")),
        )
    return text, usage_metadata


def request_local_openai(
    *,
    client,
    url: str,
    model: str,
    content: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> tuple[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + content + "\n</INPUT>\n"},
        ],
        "temperature": 0.0,
        "stream": False,
    }
    if int(max_tokens) > 0:
        payload["max_tokens"] = int(max_tokens)

    resp = client.post(url, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"local OpenAI endpoint {resp.status_code} "
            f"{getattr(resp, 'reason_phrase', '')}: {resp.text[:1200]}"
        )
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("local OpenAI endpoint returned a non-object JSON payload.")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"local OpenAI response missing choices: {json.dumps(data)[:800]}")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"local OpenAI first choice malformed: {json.dumps(first)[:800]}")
    message = first.get("message")
    if isinstance(message, dict):
        text = _message_content_as_text(message.get("content"))
    else:
        text = _message_content_as_text(first.get("text"))
    if not text.strip():
        raise RuntimeError(f"local OpenAI returned empty content: {json.dumps(first)[:800]}")
    usage_metadata = None
    usage = data.get("usage")
    if isinstance(usage, dict):
        usage_metadata = _UsageMetadata(
            prompt_token_count=_safe_int(usage.get("prompt_tokens")),
            candidates_token_count=_safe_int(usage.get("completion_tokens")),
            total_token_count=_safe_int(usage.get("total_tokens")),
        )
    return text, usage_metadata


def generate_with_google_primary_fallback(
    *,
    sample: dict,
    google_pool: GoogleKeyPool,
    openrouter_client,
    openrouter_api_key: str,
    types_mod,
    google_model: str,
    openrouter_model: str,
    system_prompt: str,
    user_prompt: str,
    thinking_budget: int,
    per_sample_timeout_s: float,
    max_google_attempts: int,
    google_token_budget: Optional[TokenBudgetLimiter],
    secrets: Iterable[str],
) -> tuple[str, object, dict[str, Any]]:
    google_attempts: list[dict[str, object]] = []
    content = sample["content"]
    estimated_google_input_tokens = estimate_google_input_tokens(
        system_prompt,
        user_prompt,
        content,
        "\n</INPUT>\n",
    )
    attempted_google_keys: set[int] = set()
    while True:
        exclude_indices = (
            attempted_google_keys
            if len(attempted_google_keys) < len(google_pool.clients)
            else None
        )
        idx, client, state = google_pool.acquire(exclude_indices=exclude_indices)
        if client is None:
            if state in {"busy", "excluded"}:
                time.sleep(0.05)
                continue
            if not openrouter_api_key:
                if len(attempted_google_keys) >= len(google_pool.clients):
                    attempted_google_keys.clear()
                time.sleep(1.0)
                continue
            generated, usage = request_openrouter(
                client=openrouter_client,
                api_key=openrouter_api_key,
                model=openrouter_model,
                content=content,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            return generated, usage, {
                "provider": "openrouter",
                "primary_provider": "google",
                "fallback_provider": "openrouter",
                "google_key_index": None,
                "fallback_reason": "all_google_keys_blocked",
                "google_attempts": google_attempts,
                "google_pool": google_pool.snapshot(),
                "google_token_budget": google_token_budget.snapshot() if google_token_budget is not None else None,
                "estimated_google_input_tokens": estimated_google_input_tokens,
                "openrouter_model": openrouter_model,
            }
        try:
            if google_token_budget is not None:
                google_token_budget.acquire(estimated_google_input_tokens)
            generated, usage = call_google_model(
                client=client,
                types_mod=types_mod,
                model=google_model,
                content=content,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                thinking_budget=thinking_budget,
                per_sample_timeout_s=per_sample_timeout_s,
            )
            return generated, usage, {
                "provider": "google",
                "primary_provider": "google",
                "fallback_provider": None,
                "google_key_index": idx,
                "fallback_reason": None,
                "google_attempts": google_attempts,
                "google_pool": google_pool.snapshot(),
                "google_token_budget": google_token_budget.snapshot() if google_token_budget is not None else None,
                "estimated_google_input_tokens": estimated_google_input_tokens,
                "openrouter_model": None,
            }
        except GeminiSampleTimeout:
            raise
        except BaseException as exc:
            if not _is_google_blocking_error(exc):
                raise
            attempted_google_keys.add(int(idx))
            reason = _redact_secrets(str(exc), secrets)
            cooldown = _cooldown_seconds_for_error(exc, RATE_LIMIT_SLEEP_S)
            google_pool.block(int(idx), reason=reason, seconds=cooldown)
            google_attempts.append(
                {
                    "google_key_index": int(idx),
                    "status": "blocked",
                    "cooldown_s": cooldown,
                    "reason": reason[:500],
                }
            )
            if len(google_attempts) >= max(1, int(max_google_attempts)):
                if not openrouter_api_key:
                    raise GoogleAttemptLimitError(
                        f"Google attempt limit reached without OpenRouter fallback; "
                        f"attempts={len(google_attempts)}",
                        attempts=google_attempts,
                        google_pool=google_pool.snapshot(),
                        google_token_budget=(
                            google_token_budget.snapshot()
                            if google_token_budget is not None
                            else None
                        ),
                    )
                generated, usage = request_openrouter(
                    client=openrouter_client,
                    api_key=openrouter_api_key,
                    model=openrouter_model,
                    content=content,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                return generated, usage, {
                    "provider": "openrouter",
                    "primary_provider": "google",
                    "fallback_provider": "openrouter",
                    "google_key_index": None,
                    "fallback_reason": "google_attempt_limit_reached",
                    "google_attempts": google_attempts,
                    "google_pool": google_pool.snapshot(),
                    "google_token_budget": google_token_budget.snapshot() if google_token_budget is not None else None,
                    "estimated_google_input_tokens": estimated_google_input_tokens,
                    "openrouter_model": openrouter_model,
                }
        finally:
            google_pool.release(idx)


def process_sample(sample, *, client, types_mod, model, system_prompt, user_prompt, thinking_budget, per_sample_timeout_s, secrets: Iterable[str] = ()):
    """Single API call + parse. Mirrors label_unseen_set.process_sample but
    works with the test-row schema (task, host, content)."""
    started = time.monotonic()
    record: dict[str, Any] = {
        "benchmark_id": sample["benchmark_id"],
        "task": sample["task"],
        "host": sample["host"],
        "model": model,
        "provider": "google",
        "primary_provider": "google",
        "fallback_provider": None,
        "google_key_index": None,
        "fallback_reason": None,
        "content_sha256": sample.get("content_sha256"),
        "input_characters": sample.get("input_characters"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    content = sample["content"]
    rate_limit_attempts = 0
    while True:
        try:
            generated, usage = call_gemini(
                client=client,
                types_mod=types_mod,
                model=model,
                content=content,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                thinking_budget=thinking_budget,
                per_sample_timeout_s=per_sample_timeout_s,
            )
            break
        except GeminiSampleTimeout as exc:
            record.update({"status": "failed_timeout", "error": _redact_secrets(str(exc), secrets), "cost_usd": 0.0, "latency_s": time.monotonic() - started})
            return record
        except BaseException as exc:
            if _is_rate_limit_error(exc) and rate_limit_attempts == 0:
                rate_limit_attempts += 1
                time.sleep(RATE_LIMIT_SLEEP_S)
                continue
            if _is_rate_limit_error(exc):
                record.update({"status": "failed_rate_limit", "error": _redact_secrets(str(exc), secrets)[:1000], "cost_usd": 0.0, "latency_s": time.monotonic() - started})
                return record
            record.update({
                "status": "failed_api",
                "error": _redact_secrets(str(exc), secrets)[:1000],
                "error_traceback": traceback.format_exc()[-1500:],
                "cost_usd": 0.0,
                "latency_s": time.monotonic() - started,
            })
            return record

    usage_serialised = _serialize_usage_metadata(usage)
    cost = compute_cost_usd(model, usage)
    record["usage_metadata"] = usage_serialised
    record["cost_usd"] = cost
    record["latency_s"] = time.monotonic() - started
    record["response_character_count"] = len(generated)

    try:
        segments, healing, error = parse_and_heal(source_text=content, generated_code=generated)
    except BaseException as exc:
        record["status"] = "failed_parse"
        record["error"] = _redact_secrets(f"parse_chain_exception: {exc}", secrets)
        record["error_traceback"] = traceback.format_exc()[-2000:]
        record["healing"] = {}
        record["response_text"] = generated
        return record

    record["healing"] = healing
    if segments is None:
        record["status"] = "failed_parse"
        record["error"] = error
        record["response_text"] = generated
        return record
    record["status"] = "ok"
    record["segments"] = segments
    return record


def process_sample_google_openrouter_fallback(
    sample,
    *,
    google_pool: GoogleKeyPool,
    openrouter_client,
    openrouter_api_key: str,
    types_mod,
    google_model: str,
    openrouter_model: str,
    system_prompt: str,
    user_prompt: str,
    thinking_budget: int,
    per_sample_timeout_s: float,
    max_google_attempts: int = 1,
    google_token_budget: Optional[TokenBudgetLimiter] = None,
    secrets: Iterable[str] = (),
):
    started = time.monotonic()
    record: dict[str, Any] = {
        "benchmark_id": sample["benchmark_id"],
        "task": sample["task"],
        "host": sample["host"],
        "model": google_model,
        "openrouter_model": openrouter_model,
        "primary_provider": "google",
        "fallback_provider": None,
        "provider": None,
        "google_key_index": None,
        "fallback_reason": None,
        "content_sha256": sample.get("content_sha256"),
        "input_characters": sample.get("input_characters"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    content = sample["content"]

    try:
        generated, usage, provider_meta = generate_with_google_primary_fallback(
            sample=sample,
            google_pool=google_pool,
            openrouter_client=openrouter_client,
            openrouter_api_key=openrouter_api_key,
            types_mod=types_mod,
            google_model=google_model,
            openrouter_model=openrouter_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            thinking_budget=thinking_budget,
            per_sample_timeout_s=per_sample_timeout_s,
            max_google_attempts=max_google_attempts,
            google_token_budget=google_token_budget,
            secrets=secrets,
        )
    except GeminiSampleTimeout as exc:
        record.update(
            {
                "status": "failed_timeout",
                "provider": "google",
                "error": _redact_secrets(str(exc), secrets),
                "cost_usd": 0.0,
                "latency_s": time.monotonic() - started,
            }
        )
        return record
    except BaseException as exc:
        status = "failed_rate_limit" if _is_rate_limit_error(exc) else "failed_api"
        google_attempts = getattr(exc, "attempts", None)
        google_pool_snapshot = getattr(exc, "google_pool", google_pool.snapshot())
        google_token_budget_snapshot = getattr(exc, "google_token_budget", None)
        failed_provider = (
            "openrouter"
            if openrouter_api_key and "OpenRouter" in str(exc)
            else "google"
        )
        record.update(
            {
                "status": status,
                "provider": failed_provider,
                "error": _redact_secrets(str(exc), secrets)[:1000],
                "error_traceback": traceback.format_exc()[-1500:],
                "cost_usd": 0.0,
                "latency_s": time.monotonic() - started,
                "google_pool": google_pool_snapshot,
            }
        )
        if google_attempts is not None:
            record["google_attempts"] = google_attempts
        if google_token_budget_snapshot is not None:
            record["google_token_budget"] = google_token_budget_snapshot
        return record

    record.update(provider_meta)
    usage_serialised = _serialize_usage_metadata(usage)
    record["usage_metadata"] = usage_serialised
    if record.get("provider") == "openrouter":
        record["cost_usd"] = compute_openrouter_cost_usd(openrouter_model, usage)
    else:
        # Google AI Studio free-tier availability is account/key dependent.
        # Keep this numeric for the existing budget code and record the note.
        record["cost_usd"] = 0.0
        record["cost_note"] = "google_ai_studio_primary_cost_not_counted"
    record["latency_s"] = time.monotonic() - started
    record["response_character_count"] = len(generated)

    try:
        segments, healing, error = parse_and_heal(source_text=content, generated_code=generated)
    except BaseException as exc:
        record["status"] = "failed_parse"
        record["error"] = _redact_secrets(f"parse_chain_exception: {exc}", secrets)
        record["error_traceback"] = traceback.format_exc()[-2000:]
        record["healing"] = {}
        record["response_text"] = generated
        return record

    record["healing"] = healing
    if segments is None:
        record["status"] = "failed_parse"
        record["error"] = _redact_secrets(error, secrets)
        record["response_text"] = generated
        return record
    record["status"] = "ok"
    record["segments"] = segments
    return record


def process_sample_openrouter(
    sample,
    *,
    openrouter_client,
    openrouter_api_key: str,
    openrouter_model: str,
    system_prompt: str,
    user_prompt: str,
    secrets: Iterable[str] = (),
):
    started = time.monotonic()
    record: dict[str, Any] = {
        "benchmark_id": sample["benchmark_id"],
        "task": sample["task"],
        "host": sample["host"],
        "model": openrouter_model,
        "openrouter_model": openrouter_model,
        "primary_provider": "openrouter",
        "fallback_provider": None,
        "provider": "openrouter",
        "google_key_index": None,
        "fallback_reason": None,
        "content_sha256": sample.get("content_sha256"),
        "input_characters": sample.get("input_characters"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    content = sample["content"]
    try:
        generated, usage = request_openrouter(
            client=openrouter_client,
            api_key=openrouter_api_key,
            model=openrouter_model,
            content=content,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except BaseException as exc:
        record.update(
            {
                "status": "failed_api",
                "error": _redact_secrets(str(exc), secrets)[:1000],
                "error_traceback": traceback.format_exc()[-1500:],
                "cost_usd": 0.0,
                "latency_s": time.monotonic() - started,
            }
        )
        return record

    usage_serialised = _serialize_usage_metadata(usage)
    record["usage_metadata"] = usage_serialised
    record["cost_usd"] = compute_openrouter_cost_usd(openrouter_model, usage)
    record["latency_s"] = time.monotonic() - started
    record["response_character_count"] = len(generated)

    try:
        segments, healing, error = parse_and_heal(source_text=content, generated_code=generated)
    except BaseException as exc:
        record["status"] = "failed_parse"
        record["error"] = _redact_secrets(f"parse_chain_exception: {exc}", secrets)
        record["error_traceback"] = traceback.format_exc()[-2000:]
        record["healing"] = {}
        record["response_text"] = generated
        return record

    record["healing"] = healing
    if segments is None:
        record["status"] = "failed_parse"
        record["error"] = error
        record["response_text"] = generated
        return record
    record["status"] = "ok"
    record["segments"] = segments
    return record


def process_sample_local_openai(
    sample,
    *,
    local_client,
    local_openai_url: str,
    local_model: str,
    provider_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    secrets: Iterable[str] = (),
):
    started = time.monotonic()
    record: dict[str, Any] = {
        "benchmark_id": sample["benchmark_id"],
        "task": sample["task"],
        "host": sample["host"],
        "model": local_model,
        "local_openai_url": local_openai_url,
        "primary_provider": provider_name,
        "fallback_provider": None,
        "provider": provider_name,
        "google_key_index": None,
        "fallback_reason": None,
        "content_sha256": sample.get("content_sha256"),
        "input_characters": sample.get("input_characters"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    content = sample["content"]
    try:
        generated, usage = request_local_openai(
            client=local_client,
            url=local_openai_url,
            model=local_model,
            content=content,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
    except BaseException as exc:
        record.update(
            {
                "status": "failed_api",
                "error": _redact_secrets(str(exc), secrets)[:1000],
                "error_traceback": traceback.format_exc()[-1500:],
                "cost_usd": 0.0,
                "latency_s": time.monotonic() - started,
            }
        )
        return record

    record["usage_metadata"] = _serialize_usage_metadata(usage)
    record["cost_usd"] = 0.0
    record["cost_note"] = "local_llamacpp_no_api_cost"
    record["latency_s"] = time.monotonic() - started
    record["response_character_count"] = len(generated)

    try:
        segments, healing, error = parse_and_heal(source_text=content, generated_code=generated)
    except BaseException as exc:
        record["status"] = "failed_parse"
        record["error"] = _redact_secrets(f"parse_chain_exception: {exc}", secrets)
        record["error_traceback"] = traceback.format_exc()[-2000:]
        record["healing"] = {}
        record["response_text"] = generated
        return record

    record["healing"] = healing
    if segments is None:
        record["status"] = "failed_parse"
        record["error"] = _redact_secrets(error, secrets)
        record["response_text"] = generated
        return record
    record["status"] = "ok"
    record["segments"] = segments
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--provider", choices=(DEFAULT_PROVIDER, GOOGLE_POOL_PROVIDER, FALLBACK_PROVIDER, OPENROUTER_PROVIDER, LLAMACPP_PROVIDER), default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--openrouter-model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--google-api-key", action="append", default=None,
                        help="Google AI Studio / Gemini API key. May be passed up to five times.")
    parser.add_argument("--google-api-keys-file", type=Path, default=None,
                        help="File containing one Google API key per line. Blank lines and # comments are ignored.")
    parser.add_argument("--openrouter-api-key", default=None,
                        help="OpenRouter API key for fallback. May also be supplied via OPENROUTER_API_KEY.")
    parser.add_argument("--openrouter-api-key-file", type=Path, default=None,
                        help="File containing the OpenRouter API key. First non-comment line is used.")
    parser.add_argument("--local-openai-url", default=DEFAULT_LOCAL_OPENAI_URL,
                        help="OpenAI-compatible local endpoint for --provider llamacpp.")
    parser.add_argument("--local-max-tokens", type=int, default=12000,
                        help="Maximum completion tokens for the local OpenAI-compatible endpoint. 0 omits the limit.")
    parser.add_argument("--partial", type=float, required=True,
                        help="Cumulative percent of rows to process (e.g. 10, 20, 50, 100). "
                             "Rows are deterministically ordered by sha((task,host,example_id)) so "
                             "successive runs at higher percentiles include all rows from lower ones.")
    parser.add_argument("--max-spend", type=float, default=DEFAULT_MAX_SPEND_USD,
                        help="Hard cumulative USD cap across this run (excluding already-cached rows).")
    parser.add_argument("--max-spend-openrouter", type=float, default=None,
                        help="Hard cumulative USD cap for OpenRouter fallback spend. Defaults to --max-spend.")
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL_TEST)
    parser.add_argument("--google-rpm-per-key", type=float, default=DEFAULT_GOOGLE_RPM_PER_KEY,
                        help="Proactive Google request-start limit per key. Default 15 requests/minute/key.")
    parser.add_argument("--google-timeout", type=float, default=DEFAULT_GOOGLE_TIMEOUT_S,
                        help="Google call timeout in fallback mode. Default 45s; OpenRouter still uses --per-sample-timeout.")
    parser.add_argument("--google-input-tpm", type=int, default=DEFAULT_GOOGLE_INPUT_TPM,
                        help="Optional global Google input-token pacing budget per minute. Use 16000 if your keys share the observed free-tier Gemma quota; 0 disables.")
    parser.add_argument("--google-max-inflight-per-key", type=int, default=1,
                        help="Maximum simultaneous Google requests per API key. Default 1 keeps the pool gentle even when --max-parallel exceeds key count.")
    parser.add_argument("--max-google-attempts-per-sample", type=int, default=1,
                        help="Fallback to OpenRouter after this many Google blocking/timeout failures for a sample.")
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="ThinkingConfig.thinking_budget. Default 0 (off) for cheapest path. "
                             "Pass -1 for auto / N>0 for explicit cap.")
    parser.add_argument("--per-sample-timeout", type=float, default=DEFAULT_PER_SAMPLE_TIMEOUT_S)
    parser.add_argument("--run-id", default=None,
                        help="Resume / extend a specific run id. Default derives stable id from "
                             "the model name so all --partial runs share one cache file.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--prompt-template", type=Path, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--telemetry-every", type=int, default=DEFAULT_TELEMETRY_EVERY)
    synth_group = parser.add_mutually_exclusive_group()
    synth_group.add_argument("--encoding-synth", dest="encoding_synth", action="store_true",
                             default=True,
                             help="Short-circuit realistic encoding hosts with deterministic single-segment records (default).")
    synth_group.add_argument("--no-encoding-synth", dest="encoding_synth", action="store_false",
                             help="Send realistic encoding hosts to the provider instead of synthesizing them locally.")
    synth_group.add_argument("--skip-encoding-synth", dest="encoding_synth", action="store_false",
                             help="Deprecated alias for --no-encoding-synth.")
    args = parser.parse_args()

    if args.partial <= 0 or args.partial > 100:
        print("[run] --partial must be in (0, 100].", file=sys.stderr)
        return 2
    try:
        args.max_parallel = validate_max_parallel(args.max_parallel)
    except ValueError as exc:
        print(f"[run] {exc}; Gemma evaluation is capped at {MAX_PARALLEL_TEST}-fold parallelism.", file=sys.stderr)
        return 2
    if args.google_rpm_per_key <= 0:
        print("[run] --google-rpm-per-key must be positive.", file=sys.stderr)
        return 2
    if args.google_timeout <= 0:
        print("[run] --google-timeout must be positive.", file=sys.stderr)
        return 2
    if args.google_input_tpm < 0:
        print("[run] --google-input-tpm must be non-negative.", file=sys.stderr)
        return 2
    if args.google_max_inflight_per_key <= 0:
        print("[run] --google-max-inflight-per-key must be positive.", file=sys.stderr)
        return 2
    if args.max_google_attempts_per_sample <= 0:
        print("[run] --max-google-attempts-per-sample must be positive.", file=sys.stderr)
        return 2

    try:
        from google import genai
        from google.genai import types as types_mod
    except ImportError as exc:
        print(f"[run] missing google.genai SDK: {exc}", file=sys.stderr)
        return 2

    print(f"[run] loading test set from {args.test_root}...", flush=True)
    rows_all = load_test_rows(args.test_root)
    if not rows_all:
        print("[run] test set is empty.", file=sys.stderr)
        return 2
    print(f"[run] total rows in test set: {len(rows_all):,}", flush=True)

    selected = filter_by_partial(rows_all, args.partial)
    print(f"[run] partial={args.partial}% -> {len(selected):,} rows in scope", flush=True)

    # Run id: stable across partial invocations (so cache accumulates).
    if args.run_id:
        run_id = args.run_id
    elif args.provider == GOOGLE_POOL_PROVIDER:
        run_id = f"{slugify_model(args.model)}__google_pool__test_set"
    elif args.provider == FALLBACK_PROVIDER:
        run_id = f"{slugify_model(args.model)}__google_primary_openrouter_fallback__test_set"
    elif args.provider == OPENROUTER_PROVIDER:
        run_id = f"{slugify_model(args.openrouter_model)}__openrouter__test_set"
    elif args.provider == LLAMACPP_PROVIDER:
        run_id = f"{slugify_model(args.model)}__llamacpp__test_set"
    else:
        run_id = f"{slugify_model(args.model)}__test_set"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.run_dir / f"{run_id}.jsonl"
    state_path = args.run_dir / f"{run_id}.state.json"

    cache, spent_already = load_cache(cache_path)
    cached_terminal_at_start = sum(1 for r in selected if is_cache_terminal(cache.get(r["benchmark_id"])))
    retryable_cached_at_start = sum(
        1
        for r in selected
        if cache.get(r["benchmark_id"]) is not None and not is_cache_terminal(cache.get(r["benchmark_id"]))
    )
    pending = [r for r in selected if not is_cache_terminal(cache.get(r["benchmark_id"]))]
    spend_cap = float(args.max_spend_openrouter if args.max_spend_openrouter is not None else args.max_spend)
    print(
        f"[run] run_id={run_id}\n"
        f"[run]   model={args.model}\n"
        f"[run]   provider={args.provider}  openrouter_model={args.openrouter_model}\n"
        f"[run]   max_spend=${spend_cap:.2f}  already_spent=${spent_already:.4f}\n"
        f"[run]   cache_path={cache_path}\n"
        f"[run]   in_scope={len(selected)}  cached_terminal={cached_terminal_at_start}  "
        f"retryable_cached={retryable_cached_at_start}  pending={len(pending)}\n"
        f"[run]   max_parallel={args.max_parallel} thinking_budget={args.thinking_budget} "
        f"per_sample_timeout={args.per_sample_timeout}s google_rpm_per_key={args.google_rpm_per_key:g} "
        f"google_timeout={args.google_timeout:g}s google_input_tpm={args.google_input_tpm}",
        flush=True,
    )

    if not pending:
        print("[run] nothing to do at this --partial level.")
        return 0
    if spent_already >= spend_cap:
        print(f"[run] already over cap (${spent_already:.4f} >= ${spend_cap:.2f}); refusing.")
        return 1

    google_keys = collect_google_api_keys(args)
    openrouter_key = resolve_openrouter_api_key(args)
    secrets_to_redact = [*google_keys]
    if openrouter_key:
        secrets_to_redact.append(openrouter_key)

    google_pool = None
    google_token_budget = None
    openrouter_client = None
    local_client = None
    client = None
    if args.provider in (GOOGLE_POOL_PROVIDER, FALLBACK_PROVIDER):
        if not google_keys:
            print("[run] no Google API keys supplied; use --google-api-key, --google-api-keys-file, or GOOGLE_API_KEY.", file=sys.stderr)
            return 2
        if len(google_keys) > MAX_GOOGLE_KEYS:
            print(f"[run] at most {MAX_GOOGLE_KEYS} Google API keys are accepted for this capped evaluation.", file=sys.stderr)
            return 2
        if args.provider == FALLBACK_PROVIDER and not openrouter_key:
            print("[run] no OpenRouter fallback key supplied; use --openrouter-api-key or OPENROUTER_API_KEY.", file=sys.stderr)
            return 2
        if args.provider == FALLBACK_PROVIDER:
            import httpx

        google_clients = [genai.Client(api_key=key) for key in google_keys]
        google_pool = GoogleKeyPool(
            google_clients,
            cooldown_seconds=RATE_LIMIT_SLEEP_S,
            requests_per_minute=args.google_rpm_per_key,
            max_inflight_per_key=args.google_max_inflight_per_key,
        )
        google_token_budget = TokenBudgetLimiter(args.google_input_tpm) if args.google_input_tpm > 0 else None
        if args.provider == FALLBACK_PROVIDER:
            openrouter_client = httpx.Client(timeout=max(10.0, float(args.per_sample_timeout)))
            print(
                f"[run] Google-primary fallback mode: google_keys={len(google_keys)} "
                f"openrouter_fallback=enabled max_parallel={args.max_parallel} "
                f"google_rpm_per_key={args.google_rpm_per_key:g} "
                f"google_timeout={args.google_timeout:g}s "
                f"google_input_tpm={args.google_input_tpm} "
                f"google_max_inflight_per_key={args.google_max_inflight_per_key} "
                f"max_google_attempts_per_sample={args.max_google_attempts_per_sample}",
                flush=True,
            )
            pricing = OPENROUTER_PRICING_USD_PER_MTOKENS.get(args.openrouter_model)
            if pricing:
                print(
                    f"[run] OpenRouter fallback pricing estimate: "
                    f"${pricing['prompt']}/M input, ${pricing['response']}/M output",
                    flush=True,
                )
            else:
                print(f"[run] WARNING: no OpenRouter pricing entry for {args.openrouter_model!r}; fallback cost telemetry will be $0.")
        else:
            print(
                f"[run] Google-pool-only mode: google_keys={len(google_keys)} "
                f"openrouter_fallback=disabled max_parallel={args.max_parallel} "
                f"google_rpm_per_key={args.google_rpm_per_key:g} "
                f"google_timeout={args.google_timeout:g}s "
                f"google_input_tpm={args.google_input_tpm} "
                f"google_max_inflight_per_key={args.google_max_inflight_per_key} "
                f"max_google_attempts_per_sample={args.max_google_attempts_per_sample}",
                flush=True,
            )
    elif args.provider == OPENROUTER_PROVIDER:
        if not openrouter_key:
            print("[run] no OpenRouter key supplied; use --openrouter-api-key or OPENROUTER_API_KEY.", file=sys.stderr)
            return 2
        import httpx

        openrouter_client = httpx.Client(timeout=max(10.0, float(args.per_sample_timeout)))
        print(
            f"[run] OpenRouter-only mode: openrouter_model={args.openrouter_model} "
            f"max_parallel={args.max_parallel}",
            flush=True,
        )
        pricing = OPENROUTER_PRICING_USD_PER_MTOKENS.get(args.openrouter_model)
        if pricing:
            print(
                f"[run] OpenRouter pricing estimate: "
                f"${pricing['prompt']}/M input, ${pricing['response']}/M output",
                flush=True,
            )
        else:
            print(f"[run] WARNING: no OpenRouter pricing entry for {args.openrouter_model!r}; cost telemetry will be $0.")
    elif args.provider == LLAMACPP_PROVIDER:
        import httpx

        local_client = httpx.Client(timeout=max(10.0, float(args.per_sample_timeout)))
        print(
            f"[run] llama.cpp local mode: local_openai_url={args.local_openai_url} "
            f"model={args.model} max_parallel={args.max_parallel} "
            f"local_max_tokens={args.local_max_tokens}",
            flush=True,
        )
    else:
        if not google_keys:
            print("[run] enter the Google API key (hidden input)", flush=True)
            api_key = getpass.getpass("GOOGLE_API_KEY: ").strip()
            if not api_key:
                print("[run] no API key; abort.", file=sys.stderr)
                return 2
            google_keys = [api_key]
            secrets_to_redact = [api_key]
        client = genai.Client(api_key=google_keys[0])
        pricing = MODEL_PRICING_USD_PER_MTOKENS.get(args.model)
        if not pricing:
            print(f"[run] WARNING: no pricing entry for {args.model!r}; cost telemetry will be $0.")
        elif pricing.get("notes"):
            print(f"[run] PRICING NOTE: {pricing['notes']}")

    if args.prompt_template.is_file():
        system_prompt, user_prompt = load_prompt_template(args.prompt_template)
        print(f"[run] prompt template: {args.prompt_template}")
    else:
        system_prompt, user_prompt = load_dense_prompt()
        print(f"[run] WARN: falling back to upstream dense prompt (no shortcut)")

    write_lock = Lock()
    counters = {"ok": 0, "failed_parse": 0, "failed_api": 0, "failed_rate_limit": 0, "failed_timeout": 0}
    provider_counters: dict[str, int] = {"google": 0, "openrouter": 0, "synthetic": 0}
    spent = spent_already
    completed = 0
    start_time = time.monotonic()
    stop_reason = None

    state = {
        "run_id": run_id,
        "model": args.model,
        "provider": args.provider,
        "openrouter_model": args.openrouter_model if args.provider in (FALLBACK_PROVIDER, OPENROUTER_PROVIDER) else None,
        "local_openai_url": args.local_openai_url if args.provider == LLAMACPP_PROVIDER else None,
        "test_root": str(args.test_root),
        "partial_pct": args.partial,
        "max_spend": spend_cap,
        "thinking_budget": args.thinking_budget,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "samples_total": len(selected),
        "samples_cached_at_start": cached_terminal_at_start,
        "samples_retryable_cached_at_start": retryable_cached_at_start,
        "spent_at_start": spent_already,
        "google_key_count": len(google_keys),
        "google_input_tpm": args.google_input_tpm if args.provider in (GOOGLE_POOL_PROVIDER, FALLBACK_PROVIDER) else None,
        "google_max_inflight_per_key": args.google_max_inflight_per_key if args.provider in (GOOGLE_POOL_PROVIDER, FALLBACK_PROVIDER) else None,
        "encoding_synth": bool(args.encoding_synth),
    }

    def write_state(extra: dict | None = None) -> None:
        snap = dict(state)
        if extra:
            snap.update(extra)
        snap["last_updated"] = datetime.now(timezone.utc).isoformat()
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2)

    write_state({"status": "running"})

    # Short-circuit encoding hosts in realistic (synthetic pure files; deterministic label).
    if args.encoding_synth:
        synth_pending = [r for r in pending if r["task"] == "realistic" and r["host"] in SYNTHETIC_HOSTS]
        if synth_pending:
            print(f"[run] synthesizing {len(synth_pending)} encoding-host realistic rows (free).", flush=True)
            for s in synth_pending:
                rec = {
                    "benchmark_id": s["benchmark_id"],
                    "task": s["task"],
                    "host": s["host"],
                    "model": args.model,
                    "provider": "synthetic",
                    "primary_provider": args.provider,
                    "fallback_provider": None,
                    "google_key_index": None,
                    "fallback_reason": None,
                    "content_sha256": s["content_sha256"],
                    "input_characters": s["input_characters"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "ok",
                    "segments": [{"type": s["host"], "content": s["content"]}],
                    "synthetic": True,
                    "synthetic_reason": "encoding_short_circuit",
                    "cost_usd": 0.0,
                    "latency_s": 0.0,
                    "response_character_count": s["input_characters"],
                    "healing": {},
                }
                write_cache_line(cache_path, rec, write_lock)
                counters["ok"] += 1
                provider_counters["synthetic"] = provider_counters.get("synthetic", 0) + 1
            pending = [r for r in pending if not (r["task"] == "realistic" and r["host"] in SYNTHETIC_HOSTS)]
            write_state({
                "status": "synthetic_done",
                "synth_count": len(synth_pending),
                "counters": dict(counters),
                "provider_counters": dict(provider_counters),
            })

    if not pending:
        print("[run] no API-bound work after synthesis; done.")
        if openrouter_client is not None:
            with contextlib.suppress(Exception):
                openrouter_client.close()
        if local_client is not None:
            with contextlib.suppress(Exception):
                local_client.close()
        return 0

    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        in_flight: dict[Future, dict] = {}
        idx = 0

        def submit_next():
            nonlocal idx
            sample = pending[idx]
            idx += 1
            if args.provider in (GOOGLE_POOL_PROVIDER, FALLBACK_PROVIDER):
                fut = pool.submit(
                    process_sample_google_openrouter_fallback,
                    sample,
                    google_pool=google_pool,
                    openrouter_client=openrouter_client,
                    openrouter_api_key=openrouter_key if args.provider == FALLBACK_PROVIDER else "",
                    types_mod=types_mod,
                    google_model=args.model,
                    openrouter_model=args.openrouter_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    thinking_budget=args.thinking_budget,
                    per_sample_timeout_s=args.google_timeout,
                    max_google_attempts=args.max_google_attempts_per_sample,
                    google_token_budget=google_token_budget,
                    secrets=secrets_to_redact,
                )
            elif args.provider == OPENROUTER_PROVIDER:
                fut = pool.submit(
                    process_sample_openrouter,
                    sample,
                    openrouter_client=openrouter_client,
                    openrouter_api_key=openrouter_key,
                    openrouter_model=args.openrouter_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    secrets=secrets_to_redact,
                )
            elif args.provider == LLAMACPP_PROVIDER:
                fut = pool.submit(
                    process_sample_local_openai,
                    sample,
                    local_client=local_client,
                    local_openai_url=args.local_openai_url,
                    local_model=args.model,
                    provider_name=LLAMACPP_PROVIDER,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=args.local_max_tokens,
                    secrets=secrets_to_redact,
                )
            else:
                fut = pool.submit(
                    process_sample,
                    sample,
                    client=client,
                    types_mod=types_mod,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    thinking_budget=args.thinking_budget,
                    per_sample_timeout_s=args.per_sample_timeout,
                    secrets=secrets_to_redact,
                )
            in_flight[fut] = sample

        try:
            for _ in range(min(args.max_parallel, len(pending))):
                submit_next()

            while in_flight:
                done = [f for f in list(in_flight.keys()) if f.done()]
                if not done:
                    time.sleep(0.05)
                    continue
                for fut in done:
                    sample = in_flight.pop(fut)
                    try:
                        rec = fut.result()
                    except BaseException as exc:
                        rec = {
                            "benchmark_id": sample["benchmark_id"],
                            "task": sample["task"],
                            "host": sample["host"],
                            "model": args.model,
                            "status": "failed_api",
                            "error": _redact_secrets(f"unhandled_exception: {exc}", secrets_to_redact),
                            "error_traceback": traceback.format_exc()[-1500:],
                            "cost_usd": 0.0,
                            "latency_s": 0.0,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    write_cache_line(cache_path, rec, write_lock)
                    status = rec.get("status", "unknown")
                    counters[status] = counters.get(status, 0) + 1
                    provider_name = str(rec.get("provider") or "unknown")
                    provider_counters[provider_name] = provider_counters.get(provider_name, 0) + 1
                    spent += float(rec.get("cost_usd", 0.0) or 0.0)
                    completed += 1

                    if completed % args.telemetry_every == 0 or completed == len(pending):
                        elapsed = max(time.monotonic() - start_time, 1e-6)
                        rate = completed / elapsed
                        remaining = len(pending) - completed
                        eta_s = remaining / rate if rate > 0 else float("inf")
                        per_sample = (spent - spent_already) / max(completed, 1)
                        print(
                            f"[run] {completed}/{len(pending)} "
                            f"ok={counters['ok']} "
                            f"parse_fail={counters['failed_parse']} "
                            f"api_fail={counters['failed_api']} "
                            f"timeout={counters['failed_timeout']} "
                            f"google={provider_counters.get('google', 0)} "
                            f"openrouter={provider_counters.get('openrouter', 0)} "
                            f"spent=${spent:.4f}/${spend_cap:.2f} "
                            f"$/sample=${per_sample:.4f} "
                            f"eta={eta_s/60:.1f}min",
                            flush=True,
                        )
                        write_state({
                            "status": "running",
                            "samples_completed_this_run": completed,
                            "cumulative_spent": spent,
                            "counters": dict(counters),
                            "provider_counters": dict(provider_counters),
                            "google_pool": google_pool.snapshot() if google_pool is not None else None,
                            "google_token_budget": google_token_budget.snapshot() if google_token_budget is not None else None,
                        })

                    if spent >= spend_cap and stop_reason is None:
                        stop_reason = f"budget cap hit: ${spent:.4f} >= ${spend_cap:.2f}"
                        print(f"[run] {stop_reason}; draining workers, then exit.")
                    if stop_reason is None and idx < len(pending):
                        submit_next()
        except KeyboardInterrupt:
            stop_reason = "KeyboardInterrupt"
            print("[run] Ctrl-C caught; draining workers.")

    final_state = {
        "status": "stopped" if stop_reason else "completed",
        "stop_reason": stop_reason or "all_pending_processed",
        "samples_completed_this_run": completed,
        "cumulative_spent": spent,
        "counters": dict(counters),
        "provider_counters": dict(provider_counters),
        "google_pool": google_pool.snapshot() if google_pool is not None else None,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_state(final_state)
    if openrouter_client is not None:
        with contextlib.suppress(Exception):
            openrouter_client.close()
    if local_client is not None:
        with contextlib.suppress(Exception):
            local_client.close()

    print()
    print("=" * 70)
    print(f"[run] FINISHED  run_id={run_id}  partial={args.partial}%")
    print(f"[run]   completed_this_run={completed}/{len(pending)}  cached_total={len(cache) + completed}")
    print(f"[run]   ok={counters['ok']}  fail_parse={counters['failed_parse']}  "
          f"fail_api={counters['failed_api']}  fail_rate={counters['failed_rate_limit']}  "
          f"fail_timeout={counters['failed_timeout']}")
    print(f"[run]   provider_counts={provider_counters}")
    print(f"[run]   spent_this_run=${spent - spent_already:.4f}  cumulative=${spent:.4f}  cap=${spend_cap:.2f}")
    print(f"[run]   cache: {cache_path}")
    return 0 if stop_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
