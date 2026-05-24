#!/usr/bin/env python3
"""Use Gemini-3.1-Pro to dense-label the static unseen evaluation set.

This script does NOT benchmark Gemini Pro itself. It uses Pro as a label
generator to create ground-truth annotations for a new, never-before-seen
evaluation set. The resulting labels become the gold standard against which
other models (U-Net, Mamba, smaller Geminis, ...) will later be scored.

Reads ``static_unseen_v1.jsonl`` (produced by ``build_unseen_set.py``),
sends each row through the dense (non-AL) prompt template, parses + heals
the response with the existing ``downloader/utils/llm_requestor.py`` machinery,
and stores a per-sample JSONL cache under ``runs/`` — these JSONL records are
the labelled benchmark that future evaluation scripts consume.

Dispatch is round-robin across the 35 content types so that the cache stays
evenly distributed if the run is interrupted (Ctrl-C or budget cap).

Key invariants:
    * single API attempt per sample (no retries on parse/API errors)
    * one 65-second sleep on the first 429 per sample; second 429 fails fast
    * hard cumulative USD budget cap; safe to resume the same ``--run-id``
    * append-only JSONL cache, guarded by ``fcntl.flock`` for thread safety
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import getpass
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse all the dense-prompt machinery so we don't drift from the production
# labeller's parsing + healing chain.
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

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_STATIC = REPO_ROOT / "evaluation" / "llm_benchmark" / "static_unseen_v1.jsonl"
DEFAULT_RUN_DIR = REPO_ROOT / "evaluation" / "llm_benchmark" / "runs"
DEFAULT_PROMPT_TEMPLATE = REPO_ROOT / "evaluation" / "llm_benchmark" / "dense_prompt_v2.template"
# Recognises the single-segment shortcut response. We tolerate optional
# leading/trailing whitespace and an optional <OUTPUT>...</OUTPUT> wrapper.
_SINGLE_SEGMENT_RE = re.compile(
    r"\A\s*(?:<OUTPUT>\s*)?<SINGLE-SEGMENT:(?P<type>[A-Za-z0-9_\-+.]+)>\s*(?:</OUTPUT>\s*)?\Z"
)
# Flash sometimes outputs the tag as a prefix followed by echoed content:
#   <SINGLE-SEGMENT:python>def foo(): ...
# We still honour it: type is valid, content after the tag is the source echo.
_SINGLE_SEGMENT_PREFIX_RE = re.compile(
    r"\A\s*(?:<OUTPUT>\s*)?<SINGLE-SEGMENT:(?P<type>[A-Za-z0-9_\-+.]+)>"
)
DEFAULT_MAX_PARALLEL = 35
DEFAULT_MAX_SPEND_USD = 50.0
DEFAULT_TELEMETRY_EVERY = 10
RATE_LIMIT_SLEEP_S = 65

# Encoding content types are synthetic: the whole file is, by construction,
# that one encoding. Sending them to Pro wastes money AND triggers worst-case
# reasoning + O(n^2) healing diffs (the homogeneous-charset 10K-char inputs
# made the parse chain hang earlier). Short-circuit them locally with a
# deterministic single-segment label so the cache stays complete.
SYNTHETIC_HOSTS = frozenset(
    {
        "encoding_hex",
        "encoding_base64",
        "encoding_base32",
        "encoding_base58",
        "encoding_base85",
    }
)
# Finite reasoning cap: ~8K thinking tokens => worst case ~$0.08 per sample at
# the response-token rate. Pass --thinking-budget -1 explicitly to lift the
# cap (NOT recommended in batch runs).
DEFAULT_THINKING_BUDGET = 8192
# Per-sample wall-clock cap. Streaming chunks are checked between iterations;
# a slow / runaway sample is aborted client-side and logged as failed_timeout.
DEFAULT_PER_SAMPLE_TIMEOUT_S = 180


class GeminiSampleTimeout(TimeoutError):
    """Raised when a single sample exceeds --per-sample-timeout."""

_RATE_LIMIT_HINTS = (
    "429",
    "resource_exhausted",
    "rate_limit",
    "quota",
    "rate limit",
)


def slugify_model(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\.-]+", "_", model)


def load_prompt_template(path: Path) -> tuple[str, str]:
    """Loader for our dense_prompt_v2.template that adds the SINGLE-SEGMENT
    shortcut rule. Same split convention as ``load_dense_prompt`` in
    ``downloader/utils/llm_requestor.py``.
    """
    text = path.read_text(encoding="utf-8")
    parts = text.split("=== USER ===\n")
    if len(parts) != 2:
        raise ValueError(f"prompt template at {path} missing '=== USER ===' delimiter")
    system_prompt = parts[0].replace("=== SYSTEM ===\n", "").strip()
    user_prompt = parts[1].strip() + "\n"
    return system_prompt, user_prompt


def detect_single_segment_shortcut(model_text: str) -> str | None:
    """Return the type if the response is a SINGLE-SEGMENT shortcut, else None.

    Accepts both the exact form (<SINGLE-SEGMENT:type> alone) and the prefix
    form Flash sometimes uses (<SINGLE-SEGMENT:type> followed by echoed source).
    """
    if not model_text:
        return None
    m = _SINGLE_SEGMENT_RE.match(model_text) or _SINGLE_SEGMENT_PREFIX_RE.match(model_text)
    if not m:
        return None
    return m.group("type").strip().lower()


def new_run_id(model: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slugify_model(model)}__{stamp}"


def load_static(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_debug_one_per_type(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        if r["host"] in seen:
            continue
        seen.add(r["host"])
        out.append(r)
    return out


def round_robin_order(rows: list[dict]) -> list[dict]:
    """Re-order so that round k contains exactly one sample per host (if available)."""
    by_host: dict[str, list[dict]] = {}
    host_order: list[str] = []
    for r in rows:
        if r["host"] not in by_host:
            by_host[r["host"]] = []
            host_order.append(r["host"])
        by_host[r["host"]].append(r)
    out = []
    max_per_host = max(len(v) for v in by_host.values()) if by_host else 0
    for i in range(max_per_host):
        for host in host_order:
            if i < len(by_host[host]):
                out.append(by_host[host][i])
    return out


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


def write_cache_line(cache_path: Path, record: dict, lock: Lock) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with cache_path.open("a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                with contextlib.suppress(Exception):
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def compute_cost_usd(model: str, usage_metadata: Any) -> float:
    if usage_metadata is None:
        return 0.0
    pricing = MODEL_PRICING_USD_PER_MTOKENS.get(model)
    if not pricing:
        return 0.0
    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    response_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
    thoughts_tokens = getattr(usage_metadata, "thoughts_token_count", 0) or 0
    # Bill thoughts at the response rate (matches Google billing rules: hidden
    # thinking tokens are charged at the output rate).
    prompt_rate = _effective_rate(pricing.get("prompt"), prompt_tokens)
    response_rate = _effective_rate(pricing.get("response"), prompt_tokens)
    if prompt_rate is None or response_rate is None:
        return 0.0
    cost = (
        _usd_cost(prompt_tokens, prompt_rate)
        + _usd_cost(response_tokens + thoughts_tokens, response_rate)
    )
    return float(cost)


def _is_rate_limit_error(err: BaseException) -> bool:
    text = (str(err) or "").lower()
    return any(h in text for h in _RATE_LIMIT_HINTS)


def call_gemini(
    *,
    client,
    types_mod,
    model: str,
    content: str,
    system_prompt: str,
    user_prompt: str,
    thinking_budget: int,
    per_sample_timeout_s: float,
) -> tuple[str, Any]:
    """Stream a dense-prompt segmentation request. Returns (text, usage_metadata).

    Raises ``GeminiSampleTimeout`` if the stream takes longer than
    ``per_sample_timeout_s`` seconds. The check runs between streamed chunks
    so a wedged HTTP read can still hang up to the underlying network timeout;
    that's acceptable as a backstop given chunk frequency under streaming.
    """
    contents = [
        types_mod.Content(
            role="user",
            parts=[types_mod.Part.from_text(text=user_prompt + content + "\n</INPUT>\n")],
        )
    ]
    config = types_mod.GenerateContentConfig(
        thinking_config=types_mod.ThinkingConfig(thinking_budget=thinking_budget),
        system_instruction=system_prompt,
    )

    str_chunks: list[str] = []
    usage_metadata = None
    start = time.monotonic()
    stream = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    )
    for chunk in stream:
        if per_sample_timeout_s > 0:
            elapsed = time.monotonic() - start
            if elapsed > per_sample_timeout_s:
                # Best-effort close of the underlying stream so we stop the
                # server from sending more tokens we'd be billed for.
                with contextlib.suppress(Exception):
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                raise GeminiSampleTimeout(
                    f"sample exceeded {per_sample_timeout_s:.1f}s wall-clock "
                    f"(elapsed={elapsed:.1f}s)"
                )
        if chunk.usage_metadata is not None:
            usage_metadata = chunk.usage_metadata
        if not chunk.candidates:
            continue
        content_obj = chunk.candidates[0].content
        if not content_obj or not getattr(content_obj, "parts", None):
            continue
        for part in content_obj.parts:
            if getattr(part, "text", None):
                str_chunks.append(part.text)

    return "".join(str_chunks), usage_metadata


def parse_and_heal(
    *,
    source_text: str,
    generated_code: str,
) -> tuple[list[dict] | None, dict, str | None]:
    """Apply the production parser + healing chain.

    Returns ``(segments_or_None, healing_info, error_str_or_None)``.
    """
    healing: dict[str, Any] = {}

    if not generated_code or not generated_code.strip():
        return None, healing, "empty_response"

    # Fast path: SINGLE-SEGMENT shortcut. Model is asserting the whole input
    # is one content type; reconstruct without diff/healing machinery (the
    # source text is already coverage-correct by definition).
    shortcut_type = detect_single_segment_shortcut(generated_code)
    if shortcut_type is not None:
        healing["shortcut"] = "single_segment"
        return (
            [{"type": shortcut_type, "content": source_text}],
            healing,
            None,
        )

    segments = extract_segments_from_content_blocks(generated_code)
    if segments is None:
        return None, healing, "no_content_blocks"

    segments, embedded_info = _strip_embedded_markers_from_segments(segments)
    if embedded_info:
        healing["embedded_marker_cleanup"] = embedded_info

    nl_info = _heal_newline_discrepancies(source_text, segments)
    if nl_info.get("applied_patches"):
        healing["newline_healing"] = nl_info

    literal_fix = _maybe_apply_literal_escape_fix(source_text, segments)
    if literal_fix.get("applied"):
        segments = literal_fix["segments"]
        healing["literal_escape_fix"] = {
            "diff_before": literal_fix["diff_before"],
            "diff_after": literal_fix["diff_after"],
        }

    bs_fix = _maybe_restore_backslashes(source_text, segments)
    if bs_fix.get("applied"):
        segments = bs_fix["segments"]
        healing["backslash_restoration"] = {
            "diff_before": bs_fix["diff_before"],
            "diff_after": bs_fix["diff_after"],
        }

    encode_fix = _maybe_encode_literal_sequences(source_text, segments)
    if encode_fix.get("applied"):
        segments = encode_fix["segments"]
        healing["literal_sequence_encoding"] = {
            "diff_before": encode_fix["diff_before"],
            "diff_after": encode_fix["diff_after"],
        }

    small_fix = _heal_small_replacements(source_text, segments)
    if small_fix.get("applied"):
        healing["small_replacement_healing"] = small_fix

    segments, removed = _remove_empty_segments(segments)
    if removed:
        healing["empty_segments_removed"] = removed

    try:
        verification = verify_segments_locally(source_text, segments, fuzzy=True)
        healing["local_verification"] = verification
    except LocalVerificationError as exc:
        return None, healing, f"local_verification_failed: {exc}"

    try:
        _require_exact_source_match(source_text, segments)
    except LocalVerificationError as exc:
        return None, healing, f"exact_source_match_failed: {exc}"

    return segments, healing, None


def synthesize_encoding_record(sample: dict, *, model: str) -> dict:
    """Build a deterministic 'ok' record for a synthetic encoding sample.

    The benchmark builder guarantees that every row under a SYNTHETIC_HOSTS
    label is, by construction, that one encoding from start to finish, so the
    correct dense-prompt segmentation is a single segment of the matching type
    spanning the entire content. No API call, no cost, no healing chain.
    """
    content = sample["content"]
    host = sample["host"]
    return {
        "benchmark_id": sample["benchmark_id"],
        "host": host,
        "model": model,
        "content_sha256": sample.get("content_sha256"),
        "input_characters": sample.get("input_characters"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "segments": [{"type": host, "content": content}],
        "synthetic": True,
        "synthetic_reason": "encoding_short_circuit",
        "cost_usd": 0.0,
        "latency_s": 0.0,
        "response_character_count": len(content),
        "healing": {},
    }


def process_sample(
    sample: dict,
    *,
    client,
    types_mod,
    model: str,
    system_prompt: str,
    user_prompt: str,
    thinking_budget: int,
    per_sample_timeout_s: float,
) -> dict:
    """Single API call + parse. Returns the full cache record (success or failure)."""
    started = time.monotonic()
    record: dict[str, Any] = {
        "benchmark_id": sample["benchmark_id"],
        "host": sample["host"],
        "model": model,
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
            # Treat as a real failure (no retries). Cost is unknown — usage
            # metadata wasn't delivered before the abort — but Google still
            # bills tokens up to the cut-off, so we surface an estimated cost
            # if the streamed content was non-empty.
            record.update(
                {
                    "status": "failed_timeout",
                    "error": str(exc),
                    "cost_usd": 0.0,
                    "latency_s": time.monotonic() - started,
                }
            )
            return record
        except BaseException as exc:
            if _is_rate_limit_error(exc) and rate_limit_attempts == 0:
                rate_limit_attempts += 1
                time.sleep(RATE_LIMIT_SLEEP_S)
                continue
            if _is_rate_limit_error(exc):
                record.update(
                    {
                        "status": "failed_rate_limit",
                        "error": str(exc)[:1000],
                        "cost_usd": 0.0,
                        "latency_s": time.monotonic() - started,
                    }
                )
                return record
            record.update(
                {
                    "status": "failed_api",
                    "error": str(exc)[:1000],
                    "error_traceback": traceback.format_exc()[-1500:],
                    "cost_usd": 0.0,
                    "latency_s": time.monotonic() - started,
                }
            )
            return record

    usage_serialised = _serialize_usage_metadata(usage)
    cost = compute_cost_usd(model, usage)

    record["usage_metadata"] = usage_serialised
    record["cost_usd"] = cost
    record["latency_s"] = time.monotonic() - started
    record["response_character_count"] = len(generated)

    try:
        segments, healing, error = parse_and_heal(
            source_text=content, generated_code=generated
        )
    except BaseException as exc:
        # Never lose cost info: API call already succeeded and we already paid.
        # Preserve the full response text so an offline re-parse is possible
        # after a parser bug fix without re-calling the API.
        record["status"] = "failed_parse"
        record["error"] = f"parse_chain_exception: {exc}"
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-spend",
        type=float,
        default=DEFAULT_MAX_SPEND_USD,
        help="Hard cumulative USD cap; stops cleanly when exceeded.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help="Concurrent in-flight requests. Default 35 (one per content type).",
    )
    parser.add_argument(
        "--debug-1-per-type",
        action="store_true",
        help="Run a single sample per host (35 total) as a smoke test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on total samples to dispatch (after round-robin ordering).",
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=None,
        help=(
            "Keep at most N samples per host (applied before round-robin). "
            "Use to scale up gradually, e.g. 1 (smoke) -> 10 (medium) -> 50 (full). "
            "Synthetic encoding hosts still respect the same cap."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Resume an existing run by id (otherwise a new id is generated).",
    )
    parser.add_argument(
        "--seed-cache-from",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Path(s) to existing run JSONL caches. Their successful records are "
            "copied into this run's cache at startup so already-labelled samples "
            "are not re-billed. Accepts one or more files."
        ),
    )
    parser.add_argument(
        "--auto-seed",
        action="store_true",
        help=(
            "Auto-seed this run's cache with every successful record found in any "
            "*.jsonl under --run-dir (skip the one belonging to this run). Use this "
            "to forward debug runs into the main run without listing them explicitly."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Where per-run JSONL caches live.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=DEFAULT_THINKING_BUDGET,
        help=(
            "ThinkingConfig.thinking_budget. Default is %(default)s (~8K thinking "
            "tokens; ~$0.08 worst case per sample). Pass -1 to remove the cap (auto), "
            "0 to disable thinking entirely, or any N>0 for an explicit cap."
        ),
    )
    parser.add_argument(
        "--per-sample-timeout",
        type=float,
        default=DEFAULT_PER_SAMPLE_TIMEOUT_S,
        help=(
            "Wall-clock seconds before a single in-flight sample is aborted "
            "client-side and logged as failed_timeout. Default %(default)s. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=DEFAULT_PROMPT_TEMPLATE,
        help=(
            "Path to the prompt template. Default is the v2 template under "
            "evaluation/llm_benchmark/ which adds the SINGLE-SEGMENT shortcut "
            "(token-saving for pure files). Falls back to the upstream "
            "load_dense_prompt() if the file is missing."
        ),
    )
    parser.add_argument(
        "--telemetry-every",
        type=int,
        default=DEFAULT_TELEMETRY_EVERY,
        help="Print a status line every N completed samples.",
    )
    args = parser.parse_args()

    if not args.static.is_file():
        print(f"[run] static benchmark not found: {args.static}", file=sys.stderr)
        return 2

    # Lazy import google.genai so a missing dep produces a clear message.
    try:
        from google import genai
        from google.genai import types as types_mod
    except ImportError as exc:
        print(f"[run] missing google.genai SDK: {exc}", file=sys.stderr)
        return 2

    rows = load_static(args.static)
    if args.debug_1_per_type:
        rows = filter_debug_one_per_type(rows)
    elif args.samples_per_type is not None:
        kept: dict[str, list[dict]] = {}
        for r in rows:
            kept.setdefault(r["host"], []).append(r)
        rows = []
        for host_rows in kept.values():
            rows.extend(host_rows[: args.samples_per_type])
    rows = round_robin_order(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        print("[run] static benchmark is empty after filtering.", file=sys.stderr)
        return 2

    run_id = args.run_id or new_run_id(args.model)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.run_dir / f"{run_id}.jsonl"
    state_path = args.run_dir / f"{run_id}.state.json"

    seed_paths: list[Path] = []
    if args.seed_cache_from:
        seed_paths.extend(p for p in args.seed_cache_from if p.is_file())
    if args.auto_seed:
        for p in sorted(args.run_dir.glob("*.jsonl")):
            if p.resolve() == cache_path.resolve():
                continue
            if p.name.endswith(".pre_rescue.bak"):
                continue
            seed_paths.append(p)

    seeded = 0
    if seed_paths:
        existing_bids, _ = load_cache(cache_path)
        target_bids = {r["benchmark_id"] for r in rows}
        with cache_path.open("a", encoding="utf-8") as out:
            for sp in seed_paths:
                with sp.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        bid = rec.get("benchmark_id")
                        if not bid or bid in existing_bids:
                            continue
                        if bid not in target_bids:
                            continue
                        if rec.get("status") != "ok":
                            continue
                        rec = dict(rec)
                        rec["seeded_from"] = str(sp.relative_to(args.run_dir.parent) if args.run_dir.parent in sp.parents else sp)
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        existing_bids[bid] = rec
                        seeded += 1
        if seeded:
            print(f"[run] seeded {seeded} cached records from {len(seed_paths)} prior run file(s).", flush=True)

    cache, spent_already = load_cache(cache_path)
    pending = [r for r in rows if r["benchmark_id"] not in cache]
    print(
        f"[run] run_id={run_id}\n"
        f"[run]   model={args.model}\n"
        f"[run]   max_spend=${args.max_spend:.2f}; already_spent=${spent_already:.4f}\n"
        f"[run]   cache_path={cache_path}\n"
        f"[run]   total_in_static={len(rows)} cached={len(cache)} pending={len(pending)}\n"
        f"[run]   max_parallel={args.max_parallel} thinking_budget={args.thinking_budget} "
        f"per_sample_timeout={args.per_sample_timeout}s",
        flush=True,
    )

    if not pending:
        print("[run] nothing to do — all rows cached.")
        return 0

    if spent_already >= args.max_spend:
        print(
            f"[run] cumulative cost ${spent_already:.4f} already exceeds cap "
            f"${args.max_spend:.2f}; refusing to dispatch."
        )
        return 1

    pricing = MODEL_PRICING_USD_PER_MTOKENS.get(args.model)
    if not pricing:
        print(
            f"[run] WARNING: no pricing table entry for model {args.model!r} — "
            "cost telemetry will report $0.00 per sample."
        )
    elif pricing.get("notes"):
        print(f"[run] PRICING NOTE: {pricing['notes']}")

    print(
        "[run] you will now be prompted for the Google API key.\n"
        "[run] (input is hidden; paste with Ctrl-V/Cmd-V and press Enter)",
        flush=True,
    )
    api_key = getpass.getpass("GOOGLE_API_KEY: ").strip()
    if not api_key:
        print("[run] no API key provided; aborting.", file=sys.stderr)
        return 2

    client = genai.Client(api_key=api_key)
    if args.prompt_template.is_file():
        system_prompt, user_prompt = load_prompt_template(args.prompt_template)
        print(f"[run] using prompt template: {args.prompt_template}", flush=True)
    else:
        print(
            f"[run] WARNING: --prompt-template {args.prompt_template} not found; "
            f"falling back to upstream load_dense_prompt() (no SINGLE-SEGMENT shortcut).",
            flush=True,
        )
        system_prompt, user_prompt = load_dense_prompt()

    write_lock = Lock()
    state = {
        "run_id": run_id,
        "model": args.model,
        "static": str(args.static),
        "max_spend": args.max_spend,
        "thinking_budget": args.thinking_budget,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "samples_total": len(rows),
        "samples_done_at_start": len(cache),
        "spent_at_start": spent_already,
    }

    def write_state(extra: dict | None = None) -> None:
        snapshot = dict(state)
        if extra:
            snapshot.update(extra)
        snapshot["last_updated"] = datetime.now(timezone.utc).isoformat()
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)

    write_state({"status": "running"})

    counters = {
        "ok": 0,
        "failed_parse": 0,
        "failed_api": 0,
        "failed_rate_limit": 0,
        "failed_timeout": 0,
    }
    spent = spent_already
    completed = 0
    start_time = time.monotonic()

    stop_reason = None

    synthetic_pending = [s for s in pending if s["host"] in SYNTHETIC_HOSTS]
    api_pending = [s for s in pending if s["host"] not in SYNTHETIC_HOSTS]
    synthetic_total = len(synthetic_pending)
    api_total = len(api_pending)

    if synthetic_pending:
        print(
            f"[run] short-circuiting {synthetic_total} samples for "
            f"synthetic encoding hosts (no API calls).",
            flush=True,
        )
        for sample in synthetic_pending:
            record = synthesize_encoding_record(sample, model=args.model)
            write_cache_line(cache_path, record, write_lock)
            counters["ok"] += 1
        write_state(
            {
                "status": "synthetic_done",
                "synthetic_completed": synthetic_total,
                "cumulative_spent": spent,
                "counters": dict(counters),
            }
        )
        print(
            f"[run] synthetic done: {synthetic_total} ok, "
            f"{api_total} samples to dispatch via API.",
            flush=True,
        )

    # `completed` now counts ONLY API-loop progress, so telemetry math is
    # consistent: completed in [0, api_total]. The synthetic phase is reported
    # separately above.
    pending = api_pending  # rest of loop only deals with API-bound samples

    if not pending:
        # All remaining work was synthetic; jump straight to the final report.
        final_state = {
            "status": "completed",
            "stop_reason": "synthetic_only_no_api_work",
            "samples_completed_this_run": completed,
            "cumulative_spent": spent,
            "counters": dict(counters),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        write_state(final_state)
        print()
        print("=" * 70)
        print(f"[run] FINISHED  run_id={run_id}")
        print(f"[run]   completed={completed} (all synthetic, no API)")
        print(f"[run]   ok={counters['ok']}")
        print(f"[run]   spent=${spent:.4f}  cap=${args.max_spend:.2f}")
        print(f"[run]   cache: {cache_path}")
        print(f"[run]   state: {state_path}")
        return 0

    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        in_flight: dict[Future, dict] = {}
        # Prime the pool, but never beyond the budget remaining.
        dispatch_idx = 0

        def can_dispatch_more() -> bool:
            return dispatch_idx < len(pending) and stop_reason is None

        def submit_next() -> None:
            nonlocal dispatch_idx
            sample = pending[dispatch_idx]
            dispatch_idx += 1
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
            )
            in_flight[fut] = sample

        try:
            for _ in range(min(args.max_parallel, len(pending))):
                submit_next()

            while in_flight:
                done_futs = []
                for fut in list(in_flight.keys()):
                    if fut.done():
                        done_futs.append(fut)
                if not done_futs:
                    time.sleep(0.05)
                    continue

                for fut in done_futs:
                    sample = in_flight.pop(fut)
                    try:
                        record = fut.result()
                    except BaseException as exc:
                        record = {
                            "benchmark_id": sample["benchmark_id"],
                            "host": sample["host"],
                            "model": args.model,
                            "status": "failed_api",
                            "error": f"unhandled_exception: {exc}",
                            "error_traceback": traceback.format_exc()[-1500:],
                            "cost_usd": 0.0,
                            "latency_s": 0.0,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    write_cache_line(cache_path, record, write_lock)

                    status = record.get("status", "unknown")
                    counters[status] = counters.get(status, 0) + 1
                    spent += float(record.get("cost_usd", 0.0) or 0.0)
                    completed += 1

                    if (
                        completed % args.telemetry_every == 0
                        or completed == len(pending)
                    ):
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
                            f"rate_limit_fail={counters['failed_rate_limit']} "
                            f"timeout={counters['failed_timeout']} "
                            f"spent=${spent:.4f}/${args.max_spend:.2f} "
                            f"$/sample=${per_sample:.4f} "
                            f"eta={eta_s/60:.1f}min",
                            flush=True,
                        )
                        write_state(
                            {
                                "status": "running",
                                "samples_completed_this_run": completed,
                                "cumulative_spent": spent,
                                "counters": dict(counters),
                            }
                        )

                    if spent >= args.max_spend and stop_reason is None:
                        stop_reason = (
                            f"budget cap hit: ${spent:.4f} >= ${args.max_spend:.2f}"
                        )
                        print(f"[run] {stop_reason}; draining workers, then exit.")

                    if can_dispatch_more():
                        submit_next()

        except KeyboardInterrupt:
            stop_reason = "KeyboardInterrupt"
            print("[run] caught Ctrl-C; cancelling pending dispatches, draining.", flush=True)

    final_state = {
        "status": "stopped" if stop_reason else "completed",
        "stop_reason": stop_reason or "all_pending_processed",
        "samples_completed_this_run": completed,
        "cumulative_spent": spent,
        "counters": dict(counters),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_state(final_state)

    print()
    print("=" * 70)
    print(f"[run] FINISHED  run_id={run_id}")
    print(f"[run]   completed={completed}/{len(pending)} pending  (cached_before={len(cache)})")
    print(f"[run]   ok={counters['ok']}  failed_parse={counters['failed_parse']}  "
          f"failed_api={counters['failed_api']}  failed_rate_limit={counters['failed_rate_limit']}  "
          f"failed_timeout={counters['failed_timeout']}")
    print(f"[run]   spent=${spent:.4f}  cap=${args.max_spend:.2f}")
    print(f"[run]   stop_reason={final_state['stop_reason']}")
    print(f"[run]   cache: {cache_path}")
    print(f"[run]   state: {state_path}")
    return 0 if stop_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
