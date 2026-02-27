from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from dotenv import load_dotenv
load_dotenv()


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    genai = None
    types = None
try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    _tqdm = None

from downloader.utils import llm_requestor
import utils.config as cfg


ORACLE_RESPONSE_SCHEMA: Dict[str, object] = {
    "type": "object",
    "required": ["snippets"],
    "properties": {
        "snippets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["snippet_id", "segments"],
                "properties": {
                    "snippet_id": {"type": "string"},
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["label", "text"],
                            "properties": {
                                "label": {"type": "string"},
                                "text": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
    },
}


_LABEL_ALIASES = {
    "javascript": "javascript_typescript",
    "typescript": "javascript_typescript",
    "js": "javascript_typescript",
    "ts": "javascript_typescript",
    "c": "c_family",
    "cpp": "c_family",
    "c++": "c_family",
    "gettext-catalog": "gettext_catalog",
    "vb": "visual_basic",
    "plain_text": "text",
}

ALLOWED_LABELS_BY_ID: List[str] = [
    name for name, _ in sorted(cfg.LANG2ID.items(), key=lambda kv: kv[1])
]
if "other" not in ALLOWED_LABELS_BY_ID:
    ALLOWED_LABELS_BY_ID.append("other")
ALLOWED_LABEL_SET = set(ALLOWED_LABELS_BY_ID)
try:
    ORACLE_RESPONSE_SCHEMA["properties"]["snippets"]["items"]["properties"]["segments"]["items"]["properties"][
        "label"
    ]["enum"] = list(ALLOWED_LABELS_BY_ID)
except Exception:
    pass

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_GENERIC_LABELS = {"text", "other"}
_RATE_LIMIT_DEFAULT_SLEEP_SECONDS = 65.0
_RATE_LIMIT_MAX_RETRIES = 8
_MISSING_SNIPPET_RECOVERY_RETRIES = 2
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_APP_NAME_DEFAULT = "text-segmentation"


@dataclass(frozen=True)
class OracleSegment:
    start: int
    end: int
    label: str
    raw_label: Optional[str] = None


@dataclass(frozen=True)
class BoundarySnippet:
    snippet_id: str
    text: str
    global_start: int
    global_end: int
    boundary: int
    predicted_labels: List[str]
    metadata: Dict[str, object]


class Oracle(Protocol):
    def annotate(self, snippets: Sequence[BoundarySnippet]) -> Dict[str, List[OracleSegment]]:
        ...


@dataclass(frozen=True)
class _UsageMetadata:
    prompt_token_count: Optional[int] = None
    candidates_token_count: Optional[int] = None
    total_token_count: Optional[int] = None


def canonicalize_label(label: str) -> str:
    raw = str(label or "").strip().lower()
    if not raw:
        return "other"
    raw = raw.replace("-", "_")
    return _LABEL_ALIASES.get(raw, raw)


def normalize_label_to_allowed(label: str) -> tuple[str, Optional[str]]:
    normalized = canonicalize_label(label)
    if normalized in ALLOWED_LABEL_SET:
        return normalized, None
    if not normalized:
        return "other", None
    return "other", normalized


def merge_adjacent_segments(segments: Iterable[OracleSegment]) -> List[OracleSegment]:
    ordered = sorted(segments, key=lambda seg: (int(seg.start), int(seg.end)))
    merged: List[OracleSegment] = []
    for seg in ordered:
        s = int(seg.start)
        e = int(seg.end)
        if e <= s:
            continue
        label_source = seg.raw_label if seg.raw_label is not None else seg.label
        label, open_set_label = normalize_label_to_allowed(label_source)
        if not merged:
            merged.append(OracleSegment(s, e, label, open_set_label))
            continue
        last = merged[-1]
        if label == last.label and open_set_label == last.raw_label and s <= last.end:
            merged[-1] = OracleSegment(last.start, max(last.end, e), label, open_set_label)
            continue
        if label == last.label and open_set_label == last.raw_label and s == last.end:
            merged[-1] = OracleSegment(last.start, e, label, open_set_label)
            continue
        merged.append(OracleSegment(s, e, label, open_set_label))
    return merged


def _segments_from_labels(labels: Sequence[str]) -> List[OracleSegment]:
    if not labels:
        return []
    out: List[OracleSegment] = []
    start = 0
    cur, cur_open_set = normalize_label_to_allowed(labels[0])
    for i in range(1, len(labels)):
        nxt, nxt_open_set = normalize_label_to_allowed(labels[i])
        if nxt != cur or nxt_open_set != cur_open_set:
            out.append(OracleSegment(start, i, cur, cur_open_set))
            start = i
            cur = nxt
            cur_open_set = nxt_open_set
    out.append(OracleSegment(start, len(labels), cur, cur_open_set))
    return out


def _labels_from_segments(text_len: int, segments: Sequence[OracleSegment]) -> List[str]:
    labels = ["unlabeled"] * max(0, int(text_len))
    for seg in segments:
        start = max(0, min(text_len, int(seg.start)))
        end = max(start, min(text_len, int(seg.end)))
        if end <= start:
            continue
        for idx in range(start, end):
            labels[idx] = str(seg.label)
    return labels


def _detect_markdown_frontmatter_end(text: str) -> Optional[int]:
    if not text:
        return None
    if text.startswith("---\n"):
        search_from = len("---\n")
        marker = "\n---\n"
        idx = text.find(marker, search_from)
        if idx >= 0:
            return idx + len(marker)
        marker = "\n...\n"
        idx = text.find(marker, search_from)
        if idx >= 0:
            return idx + len(marker)
        return None
    if text.startswith("---\r\n"):
        search_from = len("---\r\n")
        marker = "\r\n---\r\n"
        idx = text.find(marker, search_from)
        if idx >= 0:
            return idx + len(marker)
        marker = "\r\n...\r\n"
        idx = text.find(marker, search_from)
        if idx >= 0:
            return idx + len(marker)
        return None
    return None


def _apply_frontmatter_guard(snippet: BoundarySnippet, segments: Sequence[OracleSegment]) -> List[OracleSegment]:
    source_lang = canonicalize_label(str(snippet.metadata.get("source_lang", "")))
    if source_lang != "markdown":
        return list(segments)
    if int(getattr(snippet, "global_start", 0)) != 0:
        return list(segments)
    n = len(snippet.text)
    if n <= 0:
        return list(segments)
    frontmatter_end = _detect_markdown_frontmatter_end(snippet.text)
    if frontmatter_end is None or frontmatter_end <= 0 or frontmatter_end >= n:
        return list(segments)

    labels = _labels_from_segments(n, segments)
    yaml_prefix_end = 0
    while yaml_prefix_end < n and labels[yaml_prefix_end] == "yaml":
        yaml_prefix_end += 1
    if yaml_prefix_end <= frontmatter_end:
        return list(segments)

    for idx in range(frontmatter_end, yaml_prefix_end):
        labels[idx] = "markdown"
    return _segments_from_labels(labels)


def _looks_like_generic_collapse(snippet: BoundarySnippet, segments: Sequence[OracleSegment]) -> bool:
    source_lang = canonicalize_label(str(snippet.metadata.get("source_lang", "")))
    if source_lang in _GENERIC_LABELS:
        return False
    n = len(snippet.text)
    if n <= 0 or not segments:
        return False
    seg_labels = {canonicalize_label(seg.label) for seg in segments}
    if not seg_labels.issubset(_GENERIC_LABELS):
        return False

    fallback_segments = _segments_from_labels(snippet.predicted_labels)
    fallback_labels = {canonicalize_label(seg.label) for seg in fallback_segments}
    if fallback_labels.issubset(_GENERIC_LABELS):
        return False

    if len(segments) == 1:
        seg = segments[0]
        if int(seg.start) <= 0 and int(seg.end) >= n:
            return True

    pred_labels = list(snippet.predicted_labels[:n])
    if len(pred_labels) < n:
        pred_labels.extend(["other"] * (n - len(pred_labels)))
    ref_labels = _labels_from_segments(n, segments)
    diff = sum(1 for idx in range(n) if pred_labels[idx] != ref_labels[idx])
    return (float(diff) / float(n)) >= 0.80


def _postprocess_oracle_segments(
    snippet: BoundarySnippet,
    segments: Sequence[OracleSegment],
) -> List[OracleSegment]:
    adjusted = merge_adjacent_segments(_apply_frontmatter_guard(snippet, segments))
    if _looks_like_generic_collapse(snippet, adjusted):
        return []
    return adjusted


def _normalize_segments(
    snippet: BoundarySnippet,
    segments: Sequence[OracleSegment],
) -> List[OracleSegment]:
    n = len(snippet.text)
    cleaned: List[OracleSegment] = []
    for seg in segments:
        start = max(0, min(n, int(seg.start)))
        end = max(start, min(n, int(seg.end)))
        if end <= start:
            continue
        label_source = seg.raw_label if seg.raw_label is not None else seg.label
        label, open_set_label = normalize_label_to_allowed(label_source)
        cleaned.append(OracleSegment(start, end, label, open_set_label))
    cleaned = merge_adjacent_segments(cleaned)

    if not cleaned:
        return []

    cursor = 0
    out: List[OracleSegment] = []
    for seg in cleaned:
        if int(seg.start) != int(cursor):
            return []
        if seg.end <= cursor:
            return []
        out.append(OracleSegment(int(seg.start), int(seg.end), seg.label, seg.raw_label))
        cursor = seg.end
    if int(cursor) != int(n):
        return []
    return _postprocess_oracle_segments(snippet, merge_adjacent_segments(out))


def _segments_cover_full_snippet(snippet: BoundarySnippet, segments: Sequence[OracleSegment]) -> bool:
    n = len(snippet.text)
    if n <= 0:
        return len(segments) == 0
    if not segments:
        return False
    cursor = 0
    for seg in segments:
        start = int(seg.start)
        end = int(seg.end)
        if start != cursor:
            return False
        if end <= start:
            return False
        cursor = end
    return cursor == n


def _extract_json_payload(raw: str) -> Dict[str, object]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _is_rate_limited_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    if not msg:
        return False
    tokens = (
        "rate limit",
        "rate_limit",
        "resource_exhausted",
        "resource exhausted",
        "too many requests",
        "429",
        "quota",
        "retrydelay",
    )
    return any(token in msg for token in tokens)


def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
            if retry_after is not None:
                try:
                    parsed = float(str(retry_after).strip())
                    if parsed > 0:
                        return parsed
                except ValueError:
                    pass

    msg = str(exc or "")
    if not msg:
        return None

    patterns = [
        r"retrydelay[\"'=:\s]+([0-9]+(?:\.[0-9]+)?)s",
        r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)\b",
        r"retry(?:\s|-)?after[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        unit = (match.group(2) if match.lastindex and match.lastindex >= 2 else "s").lower()
        if unit.startswith("m"):
            value *= 60.0
        if value > 0:
            return value
    return None


def _extract_quota_id(exc: Exception) -> Optional[str]:
    msg = str(exc or "")
    if not msg:
        return None
    patterns = [
        r"quotaid[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']",
        r"quotaid[^a-z0-9]+([a-z0-9_\-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, msg, flags=re.IGNORECASE)
        if not match:
            continue
        quota_id = str(match.group(1)).strip()
        if quota_id:
            return quota_id
    return None


def _is_non_retriable_quota_error(exc: Exception) -> bool:
    """Detect quota failures that should fail fast instead of sleeping/retrying.

    These are typically daily free-tier limits (e.g. GenerateRequestsPerDay...),
    which will not recover within the current run.
    """
    msg = str(exc or "").lower()
    if not msg:
        return False
    if "quota" not in msg and "resource_exhausted" not in msg:
        return False

    quota_id = (_extract_quota_id(exc) or "").lower()
    if "perday" in quota_id or "daily" in quota_id:
        return True

    non_retriable_tokens = (
        "generaterequestsperday",
        "perdayperproject",
        "per day",
        "daily quota",
    )
    return any(token in msg for token in non_retriable_tokens)


def _compute_rate_limit_sleep_seconds(
    *,
    configured_default_seconds: float,
    retry_after_seconds: Optional[float],
) -> float:
    # Prefer provider-reported retry timing when available.
    if retry_after_seconds is not None and float(retry_after_seconds) > 0.0:
        return max(1.0, float(retry_after_seconds) + 1.0)
    return max(1.0, float(configured_default_seconds))


def _resolve_oracle_backend(cli_api_key: Optional[str]) -> tuple[str, str]:
    explicit_key = str(cli_api_key or "").strip()
    if explicit_key:
        return "google", explicit_key

    google_key = str(os.environ.get("GOOGLE_API_KEY", "")).strip()
    openrouter_key = str(os.environ.get("OPEN_ROUTER_API_KEY", "")).strip()

    if google_key:
        return "google", google_key
    if openrouter_key:
        return "openrouter", openrouter_key

    raise RuntimeError(
        "Missing API key. Provide --api-key, export GOOGLE_API_KEY, or export OPEN_ROUTER_API_KEY."
    )


def _openrouter_model_name(model: str) -> str:
    model_name = str(model or "").strip()
    if not model_name:
        return "google/gemini-3-flash-preview"
    if "/" in model_name:
        return model_name
    if model_name.startswith("gemini-"):
        return f"google/{model_name}"
    return model_name


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_bool_env(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    raw = str(value).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


class StubOracle:
    """Deterministic no-network oracle (returns current model labels)."""

    name = "stub"
    model = "stub"

    def annotate(self, snippets: Sequence[BoundarySnippet]) -> Dict[str, List[OracleSegment]]:
        return {
            snippet.snippet_id: _segments_from_labels(snippet.predicted_labels)
            for snippet in snippets
        }


class GeminiBoundaryOracle:
    """
    Batch Gemini oracle for uncertain boundary snippets.

    One request covers multiple snippets, so system/meta prompt cost is amortized.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        batch_size: int = 8,
        proxy: Optional[str] = None,
        rate_limit_sleep_seconds: float = _RATE_LIMIT_DEFAULT_SLEEP_SECONDS,
        rate_limit_max_retries: int = _RATE_LIMIT_MAX_RETRIES,
        missing_snippet_retries: int = _MISSING_SNIPPET_RECOVERY_RETRIES,
        show_progress: bool = False,
        progress_desc: str = "oracle batches",
        progress_leave: bool = False,
    ) -> None:
        self.model = str(model)
        self.batch_size = max(1, int(batch_size))
        self.rate_limit_sleep_seconds = max(1.0, float(rate_limit_sleep_seconds))
        self.rate_limit_max_retries = max(0, int(rate_limit_max_retries))
        self.missing_snippet_retries = max(0, int(missing_snippet_retries))
        self.show_progress = bool(show_progress)
        self.progress_desc = str(progress_desc or "oracle batches")
        self.progress_leave = bool(progress_leave)
        self.allowed_labels = list(ALLOWED_LABELS_BY_ID)
        self.last_batches: List[Dict[str, object]] = []
        self.last_snippet_sources: Dict[str, str] = {}
        self._provider, resolved_key = _resolve_oracle_backend(api_key)
        self._provider_api_key = str(resolved_key)
        self._client = None
        self._openrouter_client = None
        self._openrouter_model = ""

        if self._provider == "openrouter":
            import httpx  # lazy import

            self._openrouter_model = _openrouter_model_name(self.model)
            self.model = self._openrouter_model
            client_kwargs: Dict[str, Any] = {"timeout": 120.0}
            if proxy:
                client_kwargs["proxy"] = proxy
                client_kwargs["verify"] = False
            self._openrouter_client = httpx.Client(**client_kwargs)
            print(
                f"Gemini oracle backend: openrouter (model={self._openrouter_model}).",
                flush=True,
            )
        else:
            if genai is None or types is None:
                raise RuntimeError("google-genai is not available in this environment.")
            http_options = None
            if proxy:
                import httpx  # lazy import

                sync_httpx_client = httpx.Client(proxy=proxy, verify=False)
                async_httpx_client = httpx.AsyncClient(proxy=proxy, verify=False)
                http_options = types.HttpOptions(
                    httpx_client=sync_httpx_client,
                    httpx_async_client=async_httpx_client,
                )
            self._client = genai.Client(api_key=resolved_key, http_options=http_options)

    @property
    def name(self) -> str:
        return "gemini_boundary_refine"

    def annotate(self, snippets: Sequence[BoundarySnippet]) -> Dict[str, List[OracleSegment]]:
        out: Dict[str, List[OracleSegment]] = {}
        self.last_batches = []
        self.last_snippet_sources = {}
        starts = range(0, len(snippets), self.batch_size)
        iterator = starts
        progress_bar = None
        processed_snippets = 0
        cumulative_failed = 0
        cumulative_skipped = 0
        failed_states = {
            "fallback_failed_segmentation",
            "fallback_oracle_error",
            "fallback_runtime_error",
        }
        if self.show_progress and _tqdm is not None and len(snippets) > 0:
            total_batches = int((len(snippets) + self.batch_size - 1) // self.batch_size)
            progress_bar = _tqdm(
                starts,
                total=total_batches,
                desc=self.progress_desc,
                unit="batch",
                dynamic_ncols=True,
                leave=self.progress_leave,
            )
            iterator = progress_bar
        for start in iterator:
            batch = list(snippets[start : start + self.batch_size])
            batch_result, snippet_sources, batch_info = self._annotate_batch(batch)
            out.update(batch_result)
            self.last_snippet_sources.update(snippet_sources)
            self.last_batches.append(batch_info)
            processed_snippets += int(len(batch))
            cumulative_failed += int(
                sum(1 for state in snippet_sources.values() if str(state) in failed_states)
            )
            cumulative_skipped += int(
                sum(1 for state in snippet_sources.values() if str(state) != "model")
            )
            if progress_bar is not None:
                progress_bar.set_postfix(
                    failed=int(cumulative_failed),
                    skipped=int(cumulative_skipped),
                    ok=int(max(0, processed_snippets - cumulative_skipped)),
                    refresh=False,
                )
        if progress_bar is not None:
            progress_bar.close()
        return out

    @staticmethod
    def _system_instruction() -> str:
        return (
            "You are a meticulous text-segmentation and boundary-refinement assistant. "
            "Follow rules literally, preserve snippet bytes, and output strict JSON only. "
            "Use only the provided allowed labels; if uncertain or out-of-taxonomy, use 'other'. "
            "Never add commentary."
        )

    @staticmethod
    def _message_content_as_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            text = content.get("text")
            return str(text) if isinstance(text, str) else ""
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif isinstance(item, str) and item:
                    parts.append(item)
            return "".join(parts)
        return ""

    def _request_segments_openrouter(self, prompt: str) -> tuple[List[str], object]:
        if self._openrouter_client is None:
            raise RuntimeError("OpenRouter client is not initialized.")

        headers = {
            "Authorization": f"Bearer {self._provider_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        referer = str(os.environ.get("OPENROUTER_HTTP_REFERER", "")).strip()
        if referer:
            headers["HTTP-Referer"] = referer
        app_name = str(os.environ.get("OPENROUTER_APP_NAME", _OPENROUTER_APP_NAME_DEFAULT)).strip()
        if app_name:
            headers["X-Title"] = app_name

        payload: Dict[str, object] = {
            "model": self._openrouter_model,
            "messages": [
                {"role": "system", "content": self._system_instruction()},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

        resp = self._openrouter_client.post(_OPENROUTER_URL, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{resp.status_code} {resp.reason_phrase or ''}".strip() + f". {resp.text}"
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
        raw_content = message.get("content")
        text = self._message_content_as_text(raw_content).strip()
        if not text:
            raise RuntimeError(f"OpenRouter returned empty message content: {json.dumps(first)[:800]}")

        usage_metadata = None
        usage = data.get("usage")
        if isinstance(usage, dict):
            usage_metadata = _UsageMetadata(
                prompt_token_count=_safe_int(usage.get("prompt_tokens")),
                candidates_token_count=_safe_int(usage.get("completion_tokens")),
                total_token_count=_safe_int(usage.get("total_tokens")),
            )
        return [text], usage_metadata

    @staticmethod
    def _segments_from_text_chunks(
        snippet_text: str,
        segs_raw: Sequence[object],
    ) -> Optional[List[OracleSegment]]:
        exact = GeminiBoundaryOracle._segments_from_text_chunks_exact(
            snippet_text,
            segs_raw,
        )
        if exact is not None:
            return exact
        return GeminiBoundaryOracle._segments_from_text_chunks_relaxed_whitespace(
            snippet_text,
            segs_raw,
        )

    @staticmethod
    def _segments_from_text_chunks_exact(
        snippet_text: str,
        segs_raw: Sequence[object],
    ) -> Optional[List[OracleSegment]]:
        if not isinstance(snippet_text, str):
            return None
        out: List[OracleSegment] = []
        cursor = 0
        total = len(snippet_text)
        for seg in segs_raw:
            if not isinstance(seg, dict):
                return None
            seg_text = seg.get("text")
            if not isinstance(seg_text, str):
                return None
            seg_len = len(seg_text)
            if seg_len <= 0:
                continue
            end = cursor + seg_len
            if end > total:
                return None
            if snippet_text[cursor:end] != seg_text:
                return None
            label = str(seg.get("label", "other"))
            out.append(
                OracleSegment(
                    start=int(cursor),
                    end=int(end),
                    label=label,
                    raw_label=label,
                )
            )
            cursor = end
        if cursor != total:
            return None
        return out

    @staticmethod
    def _is_boundary_ignorable_char(ch: str) -> bool:
        if not ch:
            return False
        if ch.isspace():
            return True
        return not ch.isprintable()

    @staticmethod
    def _segments_from_text_chunks_relaxed_whitespace(
        snippet_text: str,
        segs_raw: Sequence[object],
    ) -> Optional[List[OracleSegment]]:
        if not isinstance(snippet_text, str):
            return None
        out: List[OracleSegment] = []
        cursor = 0
        total = len(snippet_text)
        whitespace_only_labels: List[str] = []

        for seg in segs_raw:
            if not isinstance(seg, dict):
                return None
            seg_text = seg.get("text")
            if not isinstance(seg_text, str):
                return None
            if not seg_text:
                continue
            label = str(seg.get("label", "other"))
            seg_core_chars = [
                ch
                for ch in seg_text
                if not GeminiBoundaryOracle._is_boundary_ignorable_char(ch)
            ]
            if not seg_core_chars:
                whitespace_only_labels.append(label)
                continue

            start = cursor
            i = cursor
            for ch in seg_core_chars:
                while i < total and GeminiBoundaryOracle._is_boundary_ignorable_char(snippet_text[i]):
                    i += 1
                if i >= total or snippet_text[i] != ch:
                    return None
                i += 1
            end = i
            if end <= start:
                continue
            out.append(
                OracleSegment(
                    start=int(start),
                    end=int(end),
                    label=label,
                    raw_label=label,
                )
            )
            cursor = end

        i = cursor
        while i < total and GeminiBoundaryOracle._is_boundary_ignorable_char(snippet_text[i]):
            i += 1
        if i != total:
            return None

        if out:
            if out[-1].end < total:
                last = out[-1]
                out[-1] = OracleSegment(
                    start=int(last.start),
                    end=int(total),
                    label=str(last.label),
                    raw_label=last.raw_label,
                )
            return out

        if total <= 0:
            return []
        label = whitespace_only_labels[0] if whitespace_only_labels else "other"
        return [
            OracleSegment(
                start=0,
                end=int(total),
                label=label,
                raw_label=label,
            )
        ]

    @staticmethod
    def _parse_segments_payload(
        raw_text: str,
        *,
        snippet_text_by_id: Optional[Dict[str, str]] = None,
    ) -> tuple[Dict[str, List[OracleSegment]], Dict[str, str], set[str]]:
        parsed_map: Dict[str, List[OracleSegment]] = {}
        parse_failures: Dict[str, str] = {}
        returned_snippet_ids: set[str] = set()
        parsed = _extract_json_payload(raw_text)
        raw_items = parsed.get("snippets") if isinstance(parsed, dict) else None
        if not isinstance(raw_items, list):
            return parsed_map, parse_failures, returned_snippet_ids
        for entry in raw_items:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("snippet_id", "")).strip()
            if not sid:
                continue
            returned_snippet_ids.add(sid)
            if sid in parsed_map or sid in parse_failures:
                parse_failures[sid] = "duplicate_snippet_id"
                continue
            segs_raw = entry.get("segments")
            if not isinstance(segs_raw, list):
                parse_failures[sid] = "segments_not_list"
                continue
            if not isinstance(snippet_text_by_id, dict):
                parse_failures[sid] = "snippet_text_unavailable"
                continue
            if sid not in snippet_text_by_id:
                parse_failures[sid] = "unknown_snippet_id"
                continue
            missing_text_field = False
            for seg in segs_raw:
                if not isinstance(seg, dict):
                    missing_text_field = True
                    break
                if not isinstance(seg.get("text"), str):
                    missing_text_field = True
                    break
            if missing_text_field:
                parse_failures[sid] = "segments_missing_text_field"
                continue
            snippet_text = str(snippet_text_by_id.get(sid, ""))
            text_chunk_segs = GeminiBoundaryOracle._segments_from_text_chunks(
                snippet_text,
                segs_raw,
            )
            if text_chunk_segs is None:
                parse_failures[sid] = "invalid_segment_text_chunks"
                continue
            parsed_map[sid] = text_chunk_segs
        return parsed_map, parse_failures, returned_snippet_ids

    @staticmethod
    def _format_failed_segmentation_summary(
        *,
        parse_failed_ids: Sequence[str],
        missing_ids: Sequence[str],
        parse_failure_reasons: Dict[str, str],
        max_items: int = 16,
    ) -> str:
        parts: List[str] = []
        for sid in parse_failed_ids:
            reason = str(parse_failure_reasons.get(sid, "parse_failed")).strip() or "parse_failed"
            parts.append(f"{sid}(parse_failed:{reason})")
        for sid in missing_ids:
            parts.append(f"{sid}(missing)")
        if len(parts) <= max_items:
            return ", ".join(parts)
        head = ", ".join(parts[:max_items])
        return f"{head}, ... +{len(parts) - max_items} more"

    def _request_segments(
        self,
        snippets: Sequence[BoundarySnippet],
    ) -> tuple[
        Dict[str, List[OracleSegment]],
        Dict[str, str],
        set[str],
        List[str],
        object,
        str,
        Optional[str],
        int,
    ]:
        prompt = self._build_batch_prompt(snippets)
        snippet_text_by_id = {str(s.snippet_id): str(s.text) for s in snippets}
        usage_metadata = None
        response_text_parts: List[str] = []
        parsed_map: Dict[str, List[OracleSegment]] = {}
        parse_failures: Dict[str, str] = {}
        returned_snippet_ids: set[str] = set()
        status = "al_oracle_ok"
        error: Optional[str] = None
        throttled_attempts = 0
        provider = str(getattr(self, "_provider", "google")).strip().lower()
        contents = None
        config = None
        if provider != "openrouter":
            if types is None:
                raise RuntimeError("google-genai types are unavailable for Gemini direct backend.")
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ]
            config = types.GenerateContentConfig(
                system_instruction=self._system_instruction(),
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
                response_schema=ORACLE_RESPONSE_SCHEMA,
            )

        for attempt in range(self.rate_limit_max_retries + 1):
            usage_metadata = None
            response_text_parts = []
            parsed_map = {}
            parse_failures = {}
            returned_snippet_ids = set()
            try:
                if provider == "openrouter":
                    response_text_parts, usage_metadata = self._request_segments_openrouter(prompt)
                else:
                    if self._client is None:
                        raise RuntimeError("Gemini client is not initialized.")
                    for chunk in self._client.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=config,
                    ):
                        if chunk.usage_metadata is not None:
                            usage_metadata = chunk.usage_metadata
                        if not chunk.candidates:
                            continue
                        content_obj = chunk.candidates[0].content
                        if not content_obj or not getattr(content_obj, "parts", None):
                            continue
                        for part in content_obj.parts:
                            txt = getattr(part, "text", None)
                            if isinstance(txt, str) and txt:
                                response_text_parts.append(txt)

                response_text = "".join(response_text_parts).strip()
                parsed_map, parse_failures, returned_snippet_ids = self._parse_segments_payload(
                    response_text,
                    snippet_text_by_id=snippet_text_by_id,
                )
                status = "al_oracle_ok"
                error = None
                break
            except Exception as exc:  # pragma: no cover - runtime/network path
                if _is_non_retriable_quota_error(exc):
                    quota_id = _extract_quota_id(exc)
                    quota_suffix = f", quota_id={quota_id}" if quota_id else ""
                    status = "al_oracle_failed"
                    error = (
                        "Gemini quota exhausted (non-retriable within this run"
                        f"{quota_suffix}). {str(exc).strip()}"
                    )
                    parsed_map = {}
                    parse_failures = {}
                    returned_snippet_ids = set()
                    print(
                        "Gemini hard quota exhaustion detected for oracle batch; "
                        "aborting retries immediately. "
                        f"Error: {str(exc).strip()}",
                        flush=True,
                    )
                    break
                if _is_rate_limited_error(exc) and attempt < self.rate_limit_max_retries:
                    throttled_attempts += 1
                    retry_after = _extract_retry_after_seconds(exc)
                    sleep_seconds = _compute_rate_limit_sleep_seconds(
                        configured_default_seconds=self.rate_limit_sleep_seconds,
                        retry_after_seconds=retry_after,
                    )
                    print(
                        "Gemini rate limit hit for oracle batch "
                        f"(attempt={attempt + 1}, sleeping={sleep_seconds:.1f}s, "
                        f"retry_after={retry_after if retry_after is not None else 'n/a'}). "
                        f"Error: {str(exc).strip()}",
                        flush=True,
                    )
                    time.sleep(float(sleep_seconds))
                    continue
                status = "al_oracle_failed"
                error = str(exc)
                parsed_map = {}
                parse_failures = {}
                returned_snippet_ids = set()
                break
        return (
            parsed_map,
            parse_failures,
            returned_snippet_ids,
            response_text_parts,
            usage_metadata,
            status,
            error,
            throttled_attempts,
        )

    def _annotate_batch(
        self,
        snippets: Sequence[BoundarySnippet],
    ) -> tuple[Dict[str, List[OracleSegment]], Dict[str, str], Dict[str, object]]:
        timestamp, run_id = llm_requestor._new_run_identifiers()
        (
            parsed_map,
            parse_failures,
            returned_snippet_ids,
            response_text_parts,
            usage_metadata,
            status,
            error,
            throttled_attempts,
        ) = self._request_segments(snippets)
        requested_ids = [str(snippet.snippet_id) for snippet in snippets]
        requested_id_set = set(requested_ids)
        initial_missing_ids = [sid for sid in requested_ids if sid not in returned_snippet_ids]
        initial_parse_failed_ids = [
            sid for sid in requested_ids if sid in parse_failures and sid not in parsed_map
        ]
        initial_missing_set = set(initial_missing_ids)
        initial_parse_failed_set = set(initial_parse_failed_ids)
        unresolved_initial_ids = [
            sid for sid in requested_ids if sid in initial_missing_set or sid in initial_parse_failed_set
        ]
        initial_missing = int(len(initial_missing_ids))
        initial_parse_failed = int(len(initial_parse_failed_ids))
        recovered_missing = 0
        recovered_parse_failed = 0
        missing_retry_requests = 0
        parse_failed_retry_requests = 0
        final_parse_failures: Dict[str, str] = {
            sid: str(reason)
            for sid, reason in parse_failures.items()
            if sid in requested_id_set
        }

        if (
            status == "al_oracle_ok"
            and unresolved_initial_ids
            and self.missing_snippet_retries > 0
        ):
            print(
                "Oracle failed to produce valid segmentations for "
                f"{len(unresolved_initial_ids)}/{len(snippets)} snippets "
                f"(missing={initial_missing}, parse_failed={initial_parse_failed}); "
                "retrying unresolved snippets.",
                flush=True,
            )
            recovered_ids: set[str] = set()
            retry_ids = set(unresolved_initial_ids)
            for snippet in snippets:
                sid = str(snippet.snippet_id)
                if sid in parsed_map:
                    continue
                if sid not in retry_ids:
                    continue
                recovered = False
                for _ in range(self.missing_snippet_retries):
                    if sid in initial_parse_failed_set:
                        parse_failed_retry_requests += 1
                    else:
                        missing_retry_requests += 1
                    (
                        retry_parsed,
                        retry_parse_failures,
                        retry_returned_snippet_ids,
                        retry_text_parts,
                        _retry_usage,
                        retry_status,
                        retry_error,
                        retry_throttled,
                    ) = self._request_segments([snippet])
                    throttled_attempts += int(retry_throttled)
                    if retry_text_parts:
                        response_text_parts.append("\n")
                        response_text_parts.extend(retry_text_parts)
                    if retry_status != "al_oracle_ok":
                        if retry_error:
                            error = retry_error
                        continue
                    if sid in retry_parsed:
                        parsed_map[sid] = retry_parsed[sid]
                        final_parse_failures.pop(sid, None)
                        recovered_ids.add(sid)
                        recovered = True
                        break
                    retry_reason = str(retry_parse_failures.get(sid, "")).strip()
                    if retry_reason:
                        final_parse_failures[sid] = retry_reason
                    elif sid in retry_returned_snippet_ids:
                        final_parse_failures[sid] = "invalid_segment_text_chunks"
                    else:
                        final_parse_failures.pop(sid, None)
                if not recovered:
                    continue
            recovered_missing = int(sum(1 for sid in initial_missing_set if sid in recovered_ids))
            recovered_parse_failed = int(
                sum(1 for sid in initial_parse_failed_set if sid in recovered_ids)
            )
            if recovered_missing > 0 or recovered_parse_failed > 0:
                print(
                    "Recovered unresolved snippets via retry: "
                    f"missing={recovered_missing}/{initial_missing}, "
                    f"parse_failed={recovered_parse_failed}/{initial_parse_failed}.",
                    flush=True,
                )

        snippet_by_id = {str(snippet.snippet_id): snippet for snippet in snippets}
        normalized_map: Dict[str, List[OracleSegment]] = {}
        normalization_failures: Dict[str, str] = {}
        for sid, raw_segments in parsed_map.items():
            snippet = snippet_by_id.get(str(sid))
            if snippet is None:
                continue
            normalized = _normalize_segments(snippet, raw_segments)
            if not normalized:
                normalization_failures[str(sid)] = "normalized_empty_or_rejected"
                continue
            if not _segments_cover_full_snippet(snippet, normalized):
                normalization_failures[str(sid)] = "normalized_invalid_coverage"
                continue
            normalized_map[str(sid)] = list(normalized)
        for sid, reason in normalization_failures.items():
            final_parse_failures[sid] = str(reason)

        model_ids = {sid for sid in requested_ids if sid in normalized_map}
        final_parse_failed_ids = [
            sid for sid in requested_ids if sid not in model_ids and sid in final_parse_failures
        ]
        final_missing_ids = [
            sid for sid in requested_ids if sid not in model_ids and sid not in final_parse_failures
        ]
        if status == "al_oracle_ok" and (final_parse_failed_ids or final_missing_ids):
            summary = self._format_failed_segmentation_summary(
                parse_failed_ids=final_parse_failed_ids,
                missing_ids=final_missing_ids,
                parse_failure_reasons=final_parse_failures,
            )
            print(
                "Failed segmentation for "
                f"{len(final_parse_failed_ids) + len(final_missing_ids)}/{len(snippets)} snippets; "
                "marking unresolved snippets as failed/skipped. "
                f"{summary}",
                flush=True,
            )

        if status == "al_oracle_failed":
            raise RuntimeError(f"Oracle failed completely. Error: {error}")
            
        if final_parse_failed_ids or final_missing_ids:
            summary = self._format_failed_segmentation_summary(
                parse_failed_ids=final_parse_failed_ids,
                missing_ids=final_missing_ids,
                parse_failure_reasons=final_parse_failures,
            )
            raise RuntimeError(f"Oracle returned incomplete or unparseable segments. {summary}")

        failed_states = set()
        snippet_sources: Dict[str, str] = {}
        finalized: Dict[str, List[OracleSegment]] = {}
        for snippet in snippets:
            sid = str(snippet.snippet_id)
            if sid in normalized_map:
                snippet_sources[sid] = "model"
                finalized[sid] = list(normalized_map.get(sid, []))
        model_output_count = int(sum(1 for state in snippet_sources.values() if state == "model"))
        failed_segmentation_count = int(
            sum(1 for state in snippet_sources.values() if str(state) in failed_states)
        )
        skipped_segmentation_count = int(
            sum(1 for state in snippet_sources.values() if str(state) != "model")
        )

        metadata = {
            "timestamp": timestamp,
            "model": self.model,
            "oracle": self.name,
            "snippet_ids": [s.snippet_id for s in snippets],
            "batch_size": len(snippets),
            "rate_limit_retries": int(throttled_attempts),
            "initial_missing_snippets": int(initial_missing),
            "initial_parse_failed_snippets": int(initial_parse_failed),
            "recovered_missing_snippets": int(recovered_missing),
            "recovered_parse_failed_snippets": int(recovered_parse_failed),
            "missing_retry_requests": int(missing_retry_requests),
            "parse_failed_retry_requests": int(parse_failed_retry_requests),
            "final_parse_failed_snippets": int(len(final_parse_failed_ids)),
            "final_missing_snippets": int(len(final_missing_ids)),
            "parse_failed_snippet_ids": list(final_parse_failed_ids),
            "missing_snippet_ids": list(final_missing_ids),
            "failed_segmentation_snippets": int(failed_segmentation_count),
            "skipped_segmentation_snippets": int(skipped_segmentation_count),
            "parse_failed_reasons": {
                sid: str(final_parse_failures.get(sid, "parse_failed"))
                for sid in final_parse_failed_ids
            },
            "response_schema": ORACLE_RESPONSE_SCHEMA,
            "allowed_labels": self.allowed_labels,
            "usage_metadata": llm_requestor._serialize_usage_metadata(usage_metadata),
        }
        log_path = llm_requestor.log_model_output(
            run_id=run_id,
            metadata=metadata,
            generated_code="".join(response_text_parts),
            status=status,
            verification_mode="active_learning",
            segments=[
                {
                    "snippet_id": sid,
                    "segments": [seg.__dict__ for seg in segs],
                }
                for sid, segs in normalized_map.items()
            ],
            error=error,
        )

        batch_info: Dict[str, object] = {
            "run_id": run_id,
            "status": status,
            "requested": int(len(snippets)),
            "parsed": int(len(normalized_map)),
            "model_output_count": int(model_output_count),
            "rate_limit_retries": int(throttled_attempts),
            "initial_missing_snippets": int(initial_missing),
            "initial_parse_failed_snippets": int(initial_parse_failed),
            "recovered_missing_snippets": int(recovered_missing),
            "recovered_parse_failed_snippets": int(recovered_parse_failed),
            "missing_retry_requests": int(missing_retry_requests),
            "parse_failed_retry_requests": int(parse_failed_retry_requests),
            "final_parse_failed_snippets": int(len(final_parse_failed_ids)),
            "final_missing_snippets": int(len(final_missing_ids)),
            "failed_segmentation_snippets": int(failed_segmentation_count),
            "skipped_segmentation_snippets": int(skipped_segmentation_count),
            "parse_failed_snippet_ids": list(final_parse_failed_ids),
            "missing_snippet_ids": list(final_missing_ids),
            "parse_failed_reasons": {
                sid: str(final_parse_failures.get(sid, "parse_failed"))
                for sid in final_parse_failed_ids
            },
            "error": error or "",
            "log_path": str(log_path),
        }
        return finalized, snippet_sources, batch_info

    @staticmethod
    def _build_batch_prompt(snippets: Sequence[BoundarySnippet]) -> str:
        allowed_labels = ", ".join(ALLOWED_LABELS_BY_ID)
        rules = (
            "Task: refine uncertain class boundaries for each snippet.\n"
            "Return JSON with this schema:\n"
            "{'snippets':[{'snippet_id':str,'segments':[{'label':str,'text':str}]}]}.\n"
            "Rules:\n"
            "- Prefer segmenting by exact text chunks: provide `text` and `label` for each segment in order.\n"
            "- `text` must be exact substring bytes from snippet text; concatenating all segment texts must equal snippet text exactly.\n"
            "- Segments must cover full snippet text exactly once (after merge), in order, without overlap.\n"
            "- Do not output `start`/`end`; output only `label` + exact `text` chunks.\n"
            "- Preserve bytes exactly: never rewrite snippet text content.\n"
            "- Prefer changing only uncertain boundary regions; keep stable regions intact.\n"
            f"- Allowed labels (strict): {allowed_labels}.\n"
            "- If content does not match an allowed label clearly, use 'other'.\n"
            "- Avoid collapsing a whole snippet to a single generic label ('text'/'other') unless truly homogeneous.\n"
            "- For markdown with frontmatter, the frontmatter block should be labeled as yaml. Other yaml-formatted blocks within the document may also be labeled as yaml if they clearly match that format.\n"
            "- Be precise for embedded code: keep wrappers in host language, but label bodies by true language.\n"
            "- For embedded strings or encodings (e.g., 'key=value'), the 'key=' part and quotes are the host language; only the raw 'value' is the embedded language.\n"
            "- For HTML-like regions, inline event-handler values and javascript: URLs are javascript_typescript.\n"
            "- For HTML-like regions, style attribute values are css; style wrappers remain host/wrapper text.\n"
            "- Example (correct SVG inline-style split):\n"
            "  snippet: style=\"font-size:3.88584304px;fill:#00cedb;fill-opacity:1;stroke:none;"
            "stroke-width:0.29143822;stroke-opacity:1\"></tspan>\n"
            "  expected segments: [\n"
            "    {'label':'svg','text':'style=\"'},\n"
            "    {'label':'css','text':'font-size:3.88584304px;fill:#00cedb;fill-opacity:1;"
            "stroke:none;stroke-width:0.29143822;stroke-opacity:1'},\n"
            "    {'label':'svg','text':'\"></tspan>'}\n"
            "  ]\n"
            "- In SVG/XML, tag syntax + attribute names/quotes/ids/coords stay svg/xml; "
            "only the CSS declaration body inside style=\"...\" is css.\n"
            "- Keep tiny foreign-code injections typed correctly when boundaries clearly indicate a switch.\n"
            "- Use predicted_segments as a strong prior unless clearly contradicted by snippet text.\n"
            "- Return exactly one entry for every input snippet_id; never omit any snippet_id.\n"
            "- Output JSON only.\n"
        )
        payload: List[Dict[str, object]] = []
        for snippet in snippets:
            payload.append(
                {
                    "snippet_id": snippet.snippet_id,
                    "source_lang": str(snippet.metadata.get("source_lang", "")),
                    "boundary_index": int(snippet.boundary),
                    "predicted_segments": [
                        {"start": int(seg.start), "end": int(seg.end), "label": str(seg.label)}
                        for seg in _segments_from_labels(snippet.predicted_labels)
                    ],
                    "text": snippet.text,
                }
            )
        return rules + "\nINPUT:\n" + json.dumps({"snippets": payload}, ensure_ascii=False)
