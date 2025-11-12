from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import inspect
import io
import json
import os
import re
import sys
import tokenize
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx
import requests
from google import genai
from google.genai import types
import urllib3
from rich.console import Console

console = Console()

DEFAULT_MODEL = "gemini-2.5-flash"


class LocalVerificationError(RuntimeError):
    """Raised when local coverage verification fails."""

    pass


LOCAL_VERIFICATION_ERROR_STYLE = "bold red"

FUZZY_MAX_DIFF_CHARS = 4
FUZZY_MAX_BLOCKS = 2
FUZZY_MAX_BLOCK_SIZE = 2
FUZZY_MIN_RATIO = 0.999

MODEL_PRICING_USD_PER_MTOKENS = {
    # Source: https://ai.google.dev/gemini-api/docs/pricing.
    "gemini-2.5-pro": {
        "prompt": [
            {"max_prompt_tokens": 200_000, "rate": 1.25},
            {"max_prompt_tokens": None, "rate": 2.50},
        ],
        "response": [
            {"max_prompt_tokens": 200_000, "rate": 10.00},
            {"max_prompt_tokens": None, "rate": 15.00},
        ],
        "notes": "Uses ≤200K-token tiered pricing for both prompt and response tokens.",
    },
    "gemini-2.5-flash": {
        "prompt": [
            {"max_prompt_tokens": None, "rate": 0.30},
        ],
        "response": [
            {"max_prompt_tokens": None, "rate": 2.50},
        ],
        "notes": None,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_OUTPUT_DIR = PROJECT_ROOT / "gemini_output_logs"
SEGMENTATIONS_DIR = PROJECT_ROOT / "gemini_segmentations"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _proxy_mapping(proxy_url: str | None) -> dict[str, str] | None:
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


def _new_run_identifiers() -> tuple[str, str]:
    now = datetime.datetime.utcnow()
    timestamp_iso = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    run_id = f"{now.strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid.uuid4().hex[:8]}"
    return timestamp_iso, run_id


def _serialize_usage_metadata(usage_metadata):
    if usage_metadata is None:
        return None
    return {
        "prompt_token_count": getattr(usage_metadata, "prompt_token_count", None),
        "candidates_token_count": getattr(usage_metadata, "candidates_token_count", None),
        "total_token_count": getattr(usage_metadata, "total_token_count", None),
    }


def log_model_output(
    *,
    run_id: str,
    metadata: dict,
    generated_code: str,
    status: str,
    verification_mode: str,
    segments,
    error: str | None,
) -> Path:
    _ensure_dir(LOG_OUTPUT_DIR)
    payload = {
        "run_id": run_id,
        "status": status,
        "verification_mode": verification_mode,
        "metadata": metadata,
        "generated_code": generated_code,
        "segments_present": segments is not None,
        "segments": segments,
    }
    if error:
        payload["error"] = error
    log_path = LOG_OUTPUT_DIR / f"{run_id}_{status}.json"
    log_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return log_path


def save_segments_snapshot(run_id: str, metadata: dict, segments) -> Path:
    _ensure_dir(SEGMENTATIONS_DIR)
    snapshot = {
        "run_id": run_id,
        "metadata": metadata,
        "segments": segments,
    }
    snapshot_path = SEGMENTATIONS_DIR / f"{run_id}.json"
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return snapshot_path

ALLOWED_TYPE_NAMES = [
    "php",
    "csharp",
    "javascript",
    "typescript",
    "go",
    "sql",
    "rust",
    "yaml",
    "ruby",
    "python",
    "java",
    "c",
    "cpp",
    "json",
    "css",
    "html",
    "csv",
    "shell",
    "powershell",
    "visual_basic",
    "dockerfile",
    "xml",
    "markdown",
    "svg",
    "gettext-catalog",
    "scala",
    "swift",
    "restructuredtext",
    "kotlin",
    "dart",
    "encoding_hex",
    "encoding_base64",
    "encoding_base32",
    "encoding_base58",
    "encoding_base85",
    "other",
]

_ALLOWED_TYPES = set(ALLOWED_TYPE_NAMES)


def _detect_adjacent_duplicate_types(segments) -> list[str]:
    """
    Best-effort detection of adjacent segments that repeat the same type.
    Used for logging only; validation still handles malformed structures.
    """
    duplicates: list[str] = []
    if not isinstance(segments, list):
        return duplicates
    last_type: str | None = None
    for entry in segments:
        if not isinstance(entry, dict):
            break
        seg_type = entry.get("type")
        if not isinstance(seg_type, str):
            break
        if seg_type == last_type:
            duplicates.append(seg_type)
        else:
            last_type = seg_type
    return duplicates


def _stitch_segments_with_validation(segments):
    if not isinstance(segments, list):
        raise TypeError("segments must be a list.")
    if not segments:
        raise ValueError("segments cannot be empty.")

    stitched = []
    last_type = None
    for idx, entry in enumerate(segments):
        if not isinstance(entry, dict):
            raise TypeError(f"Segment {idx} is not a dict.")
        seg_type = entry.get("type")
        seg_content = entry.get("content")
        if not isinstance(seg_type, str) or not isinstance(seg_content, str):
            raise TypeError(f"Segment {idx} must define 'type' and 'content' strings.")
        if seg_type not in _ALLOWED_TYPES and not seg_type.startswith("discovered_"):
            raise ValueError(f"Segment {idx} has unsupported type {seg_type!r}.")
        if not seg_content:
            raise ValueError(f"Segment {idx} is empty; delete it or merge with neighbors.")
        if seg_type == last_type and stitched:
            # Merge adjacent blocks of the same type instead of forcing callers to re-segment.
            stitched[-1] = stitched[-1] + seg_content
            continue
        stitched.append(seg_content)
        last_type = seg_type

    return "".join(stitched)


def _is_newline_only(text: str) -> bool:
    return not text or set(text) <= {"\n", "\r"}


def _locate_segment_position(segments, absolute_index: int) -> tuple[int, int]:
    if absolute_index < 0:
        raise ValueError("absolute_index must be non-negative")
    running = 0
    for idx, entry in enumerate(segments):
        content = entry.get("content", "")
        seg_len = len(content)
        if absolute_index < running + seg_len:
            return idx, absolute_index - running
        running += seg_len
    if absolute_index == running and segments:
        return len(segments) - 1, len(segments[-1].get("content", ""))
    raise ValueError(
        f"absolute_index {absolute_index} out of range for segments with length {running}"
    )


def _remove_text_range_in_segments(segments, start: int, end: int) -> None:
    if end <= start:
        return
    remaining = end - start
    while remaining > 0:
        seg_idx, offset = _locate_segment_position(segments, start)
        content = segments[seg_idx]["content"]
        take = min(remaining, len(content) - offset)
        segments[seg_idx]["content"] = content[:offset] + content[offset + take :]
        remaining -= take


def _insert_text_at_pos(segments, pos: int, text: str) -> None:
    if not text:
        return
    seg_idx, offset = _locate_segment_position(segments, pos)
    content = segments[seg_idx]["content"]
    segments[seg_idx]["content"] = content[:offset] + text + content[offset:]


def _replace_text_range_in_segments(segments, start: int, end: int, replacement: str) -> None:
    _remove_text_range_in_segments(segments, start, end)
    _insert_text_at_pos(segments, start, replacement)


def _heal_newline_discrepancies(source_text: str, segments) -> dict[str, Any]:
    stitched_text = "".join(entry.get("content", "") for entry in segments)
    if stitched_text == source_text:
        return {"applied_patches": 0}
    matcher = SequenceMatcher(a=source_text, b=stitched_text, autojunk=False)
    newline_blocks: list[tuple[str, int, int, int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        src_fragment = source_text[i1:i2]
        seg_fragment = stitched_text[j1:j2]
        if _is_newline_only(src_fragment) and _is_newline_only(seg_fragment):
            newline_blocks.append((tag, i1, i2, j1, j2))
    if not newline_blocks:
        return {"applied_patches": 0}

    newline_blocks.sort(key=lambda block: (block[3], block[4]), reverse=True)
    for tag, i1, i2, j1, j2 in newline_blocks:
        replacement = source_text[i1:i2]
        _replace_text_range_in_segments(segments, j1, j2, replacement)

    total_delta = 0
    for tag, i1, i2, j1, j2 in newline_blocks:
        total_delta += (i2 - i1) - (j2 - j1)
    return {
        "applied_patches": len(newline_blocks),
        "char_delta": total_delta,
        "blocks": [
            {
                "tag": tag,
                "source_range": [i1, i2],
                "segments_range": [j1, j2],
            }
            for tag, i1, i2, j1, j2 in reversed(newline_blocks)
        ],
    }


def _assert_segments_cover_source(source_text, segments):
    if not isinstance(source_text, str):
        raise TypeError("SOURCE_TEXT must be a string copy of the <INPUT> contents.")

    stitched_text = _stitch_segments_with_validation(segments)
    if stitched_text != source_text:
        mismatch = 0
        for mismatch, (expected, actual) in enumerate(zip(source_text, stitched_text)):
            if expected != actual:
                break
        else:
            mismatch = min(len(source_text), len(stitched_text))
        head = max(0, mismatch - 40)
        tail = mismatch + 40
        expected_snippet = source_text[head:tail]
        actual_snippet = stitched_text[head:tail]
        raise AssertionError(
            "segments does not reproduce the <INPUT> text.\n"
            f"First difference at offset {mismatch}.\n"
            f"expected snippet: {expected_snippet!r}\n"
            f"actual snippet:   {actual_snippet!r}"
        )

    print(
        f"Coverage verified: {len(source_text)} characters reconstructed by "
        f"{len(segments)} segments."
    )


def _fuzzy_compare_source_and_segments(source_text: str, stitched_text: str) -> tuple[bool, dict[str, Any]]:
    matcher = SequenceMatcher(a=source_text, b=stitched_text, autojunk=False)
    diff_blocks = []
    total_diff = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        block = {
            "tag": tag,
            "source_range": [i1, i2],
            "segments_range": [j1, j2],
            "source_snippet": source_text[i1:i2],
            "segments_snippet": stitched_text[j1:j2],
        }
        diff_blocks.append(block)
        total_diff += max(i2 - i1, j2 - j1)

    similarity = matcher.ratio()
    if not diff_blocks:
        return True, {
            "mode": "strict",
            "diff_chars": 0,
            "blocks": [],
            "similarity": similarity,
            "reason": "exact_match",
        }

    if total_diff > FUZZY_MAX_DIFF_CHARS:
        return False, {
            "reason": "diff_chars_exceeded",
            "diff_chars": total_diff,
            "blocks": diff_blocks,
            "similarity": similarity,
        }
    if len(diff_blocks) > FUZZY_MAX_BLOCKS:
        return False, {
            "reason": "too_many_blocks",
            "diff_chars": total_diff,
            "blocks": diff_blocks,
            "similarity": similarity,
        }
    diff_tags = {block["tag"] for block in diff_blocks}
    if len(diff_tags) > 1:
        return False, {
            "reason": "mixed_diff_types",
            "diff_chars": total_diff,
            "blocks": diff_blocks,
            "similarity": similarity,
        }

    for block in diff_blocks:
        src_len = block["source_range"][1] - block["source_range"][0]
        seg_len = block["segments_range"][1] - block["segments_range"][0]
        tag = block["tag"]
        if tag == "replace":
            if max(src_len, seg_len) > FUZZY_MAX_BLOCK_SIZE:
                return False, {
                    "reason": "replace_block_too_large",
                    "diff_chars": total_diff,
                    "blocks": diff_blocks,
                    "similarity": similarity,
                }
        elif tag == "delete" and src_len > FUZZY_MAX_BLOCK_SIZE:
            return False, {
                "reason": "delete_block_too_large",
                "diff_chars": total_diff,
                "blocks": diff_blocks,
                "similarity": similarity,
            }
        elif tag == "insert" and seg_len > FUZZY_MAX_BLOCK_SIZE:
            return False, {
                "reason": "insert_block_too_large",
                "diff_chars": total_diff,
                "blocks": diff_blocks,
                "similarity": similarity,
            }

    if similarity < FUZZY_MIN_RATIO:
        return False, {
            "reason": "similarity_below_threshold",
            "diff_chars": total_diff,
            "blocks": diff_blocks,
            "similarity": similarity,
        }

    return True, {
        "mode": "fuzzy",
        "diff_chars": total_diff,
        "blocks": diff_blocks,
        "similarity": similarity,
        "reason": "minor_diff_allowed",
    }


_ALLOWED_TYPES_LITERAL = (
    "_ALLOWED_TYPES = {\n"
    + "\n".join(f'    "{name}",' for name in ALLOWED_TYPE_NAMES)
    + "\n}"
)
try:
    _STITCH_SEGMENTS_SOURCE = inspect.getsource(_stitch_segments_with_validation)
except (OSError, TypeError):
    _STITCH_SEGMENTS_SOURCE = inspect.cleandoc(
        """
        def _stitch_segments_with_validation(segments):
            if not isinstance(segments, list):
                raise TypeError("segments must be a list.")
            if not segments:
                raise ValueError("segments cannot be empty.")

            stitched = []
            last_type = None
            for idx, entry in enumerate(segments):
                if not isinstance(entry, dict):
                    raise TypeError(f"Segment {idx} is not a dict.")
                seg_type = entry.get("type")
                seg_content = entry.get("content")
                if not isinstance(seg_type, str) or not isinstance(seg_content, str):
                    raise TypeError(
                        f"Segment {idx} must define 'type' and 'content' strings."
                    )
                if seg_type not in _ALLOWED_TYPES and not seg_type.startswith(
                    "discovered_"
                ):
                    raise ValueError(f"Segment {idx} has unsupported type {seg_type!r}.")
                if not seg_content:
                    raise ValueError(
                        f"Segment {idx} is empty; delete it or merge with neighbors."
                    )
                if seg_type == last_type:
                    raise ValueError(
                        f"Segment {idx} repeats type '{seg_type}'. Merge adjacent matching types."
                    )
                stitched.append(seg_content)
                last_type = seg_type

            return "".join(stitched)
        """
    )

_ASSERT_SEGMENTS_SOURCE = inspect.getsource(_assert_segments_cover_source)


COVERAGE_VERIFIER_FUNCTION_CODE = (
    f"{_ALLOWED_TYPES_LITERAL}\n\n{_STITCH_SEGMENTS_SOURCE}\n\n{_ASSERT_SEGMENTS_SOURCE}"
)
PROMPT_EXAMPLE_BLOCK = '''

Example:
<INPUT>
# This is how you use magika
Run
```

pip install magika

```
then open an editor and type:
```

import magika
def adfadf():
return True

````
</INPUT>
<OUTPUT>
segments = [
 {"type": "markdown",
 "content": """# This is how you use magika
Run
```""",
},
{
"type": "bash",
"content": "pip install magika",
},
{
"type": "markdown",
"content": """```
then open an editor and type:
```""",

},
{
"type": "python",
"content": """import magika
def adfadf():
  return True""",
},
{
"type": "markdown",
"content": """```
""""
}
</OUTPUT>

Your turn:
<INPUT>
'''

PROMPT_WITH_CODE_EXECUTION = f"""
Segment the file embedded between <INPUT> and </INPUT> into its constituent content types (e.g., markdown, shell, python) and reproduce the content accordingly.
Only treat <INPUT></INPUT> and <OUTPUT></OUTPUT> as control tokens.

Rules:
- map to one of the following content types: php,csharp,javascript,typescript,go,sql,rust,yaml,ruby,python,java,c,cpp,json,css,html,csv,shell,powershell,visual_basic,dockerfile,xml,markdown,go,svg,gettext-catalog,scala,swift,restructuredtext,kotlin,dart,encoding_hex,encoding_base64,encoding_base32,encoding_base58,encoding_base85; if nothing fits, use discovered_<lang> or "other" as a last resort.
- be precise: classify JS/CSS inside HTML as JS/CSS, HTML blocks inside Markdown as HTML, etc.
- text or comments belong to the wrapping/adjacent content type.
- IMPORTANT INSTRUCTION: preserve every character exactly as provided (including trailing spaces or backslashes).
  - Emit exactly one literal assignment 'segments = [ {{...}}, ... ]'. Build every entry directly inside that literal. Do not call segments.append, loops, helper functions, or reassignment. My tooling parses the AST and only sees literal lists, so any procedural construction fails.
  - Also markdown fences should remain the same type (for instance, if you think there is a powershell block but the author wrote it in a shell-fence, still label it powershell but keep the shell fence in the output bytes) - i.e., keep wrong type hints from the files but label them correctly
- also small segments should be classified appropriately - like an alert(...) statement within onclick="alert(...)" should definitely already be JS, or html inside JS strings would be HTML, etc. - generalize this to all content types
- preserve every character exactly as provided (including trailing spaces or backslashes) so SOURCE_TEXT matches the original input.
- output only executable Python: define SOURCE_TEXT with the literal data between <INPUT> tags and create a single `segments` list. Do not emit <OUTPUT> tags.
- after defining `segments`, run `_assert_segments_cover_source(SOURCE_TEXT, segments)` using the coverage verifier below via the code-execution tool. Do not modify that verifier; just execute it.

Coverage verifier to copy/paste verbatim (expects SOURCE_TEXT and segments):

{COVERAGE_VERIFIER_FUNCTION_CODE}
{PROMPT_EXAMPLE_BLOCK}
"""

PROMPT_WITH_LOCAL_VERIFICATION = f"""
Segment the file embedded between <INPUT> and </INPUT> into its constituent content types (e.g., markdown, shell, python) and reproduce the content accordingly.
Only treat <INPUT></INPUT> and <OUTPUT></OUTPUT> as control tokens.

Rules:
- map to one of the following content types: php,csharp,javascript,typescript,go,sql,rust,yaml,ruby,python,java,c,cpp,json,css,html,csv,shell,powershell,visual_basic,dockerfile,xml,markdown,go,svg,gettext-catalog,scala,swift,restructuredtext,kotlin,dart,encoding_hex,encoding_base64,encoding_base32,encoding_base58,encoding_base85; if nothing fits, use discovered_<lang> or "other" as a last resort.
- be precise: classify JS/CSS inside HTML as JS/CSS, HTML blocks inside Markdown as HTML, etc.
- text or comments belong to the wrapping/adjacent content type.
- IMPORTANT INSTRUCTION: preserve every character exactly as provided (including trailing spaces or backslashes).
  - Also markdown fences should remain the same type (for instance, if you think there is a powershell block but the author wrote it in a shell-fence, still label it powershell but keep the shell fence in the output bytes) - i.e., keep wrong type hints from the files but label them correctly
- also small segments should be classified appropriately - like an alert(...) statement within onclick="alert(...)" should definitely already be JS, or html inside JS strings would be HTML, etc. - generalize this to all content types
- output only executable Python (but do not execute it): create a single `segments` list. Do not emit <OUTPUT> tags.
- after defining `segments`, do NOT run anything; my client runs the verifier locally, so ensure your code passes without modification and do not call any tools.
Coverage verifier shown for reference (do not execute it in this mode):
{COVERAGE_VERIFIER_FUNCTION_CODE}
{PROMPT_EXAMPLE_BLOCK}
"""


SYSTEM_PROMPT = (
    "You are a meticulous text-segmentation and coverage assistant. "
    "Follow every rule literally, preserve SOURCE_TEXT bytes, emit only the "
    "required Python artifacts, and never add commentary or extra output."
)


def build_prompt(use_code_execution: bool) -> str:
    return (
        PROMPT_WITH_CODE_EXECUTION
        if use_code_execution
        else PROMPT_WITH_LOCAL_VERIFICATION
    )

DEFAULT_SAMPLE_INPUT = """# syntax=docker/dockerfile:1

FROM golang:1.24

WORKDIR /src

COPY <<EOF ./main.go

package main



import "fmt"



func main() {

  fmt.Println("hello, world")

}

EOF

RUN go build -o /bin/hello ./main.go



FROM scratch

COPY --from=0 /bin/hello /bin/hello

CMD ["/bin/hello"]"""


INPUT_WAS_TRUNCATED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Gemini responses that segment content into typed chunks."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--file", type=Path, help="Path to the text file to segment.")
    source_group.add_argument("--url", help="URL pointing to content to segment.")
    source_group.add_argument("--text", help="Raw text content to segment.")
    source_group.add_argument(
        "--stdin", action="store_true", help="Read content to segment from STDIN."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to call (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="Explicit Google API key. Falls back to GOOGLE_API_KEY env",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=10_000,
        help="Maximum number of characters from any input source to send to Gemini (default: 10000).",
    )
    parser.add_argument(
        "--code-exectution",
        action="store_true",
        help=(
            "Allow Gemini to run the coverage verifier via the code-execution tool; "
            "when omitted, verification runs locally."
        ),
    )
    parser.add_argument(
        "--proxy",
        nargs="?",
        const="http://localhost:8080",
        help=(
            "Route Gemini API traffic through an HTTP proxy (e.g. Burp on port 8080). "
            "When specified without a value, defaults to http://localhost:8080."
        ),
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help=(
            "Allow a tiny amount of tolerance in local coverage verification to recover "
            "from single-character mistakes (still fails on overlaps or larger diffs)."
        ),
    )
    args = parser.parse_args()
    if args.max_input_length <= 0:
        parser.error("--max-input-length must be a positive integer.")
    return args


def resolve_api_key(cli_api_key: str | None) -> str:
    if cli_api_key:
        return cli_api_key
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return env_key
    raise SystemExit(
        "[red]Missing API key.[/red] Provide --api-key, export GOOGLE_API_KEY, "
        "or configure google.colab.userdata."
    )


def _limit_input_length(content: str, max_length: int, *, source: str) -> str:
    global INPUT_WAS_TRUNCATED
    if len(content) <= max_length:
        return content
    console.print(
        f"Truncating {source} from {len(content):,} to {max_length:,} characters to respect --max-input-length.",
        style="yellow",
    )
    INPUT_WAS_TRUNCATED = True
    return content[:max_length]


def read_input_text(
    args: argparse.Namespace,
    *,
    proxies: dict[str, str] | None = None,
    verify: bool = True,
) -> tuple[str, str]:
    if args.text:
        return (
            _limit_input_length(args.text, args.max_input_length, source="--text input"),
            "--text input",
        )
    if args.file:
        try:
            file_text = args.file.read_text()
            return (
                _limit_input_length(
                    file_text, args.max_input_length, source=f"contents of {args.file}"
                ),
                f"file:{args.file}",
            )
        except OSError as exc:
            raise SystemExit(f"[red]Failed to read file {args.file}: {exc}[/red]")
    if args.url:
        try:
            response = requests.get(args.url, proxies=proxies, verify=verify)
            response.raise_for_status()
            return (
                _limit_input_length(
                    response.text, args.max_input_length, source=f"response from {args.url}"
                ),
                f"url:{args.url}",
            )
        except requests.RequestException as exc:
            raise SystemExit(f"[red]Failed to fetch URL {args.url}: {exc}[/red]")
    if args.stdin:
        data = sys.stdin.read()
        if not data:
            raise SystemExit("[red]STDIN was selected but no data was provided.[/red]")
        return (
            _limit_input_length(data, args.max_input_length, source="STDIN input"),
            "stdin",
        )
    console.print(
        "No input source provided; using built-in Dockerfile sample.",
        style="yellow",
    )
    return (
        _limit_input_length(
            DEFAULT_SAMPLE_INPUT,
            args.max_input_length,
            source="built-in Dockerfile sample",
        ),
        "default_sample",
    )


def _usd_cost(token_count: int, usd_per_mtokens: float) -> float:
    return (token_count / 1_000_000) * usd_per_mtokens


def _effective_rate(rate_tiers, prompt_tokens: int | None) -> float | None:
    if not rate_tiers:
        return None
    if prompt_tokens is None:
        prompt_tokens = 0
    for tier in rate_tiers:
        max_tokens = tier.get("max_prompt_tokens")
        if max_tokens is None or prompt_tokens <= max_tokens:
            return tier["rate"]
    return rate_tiers[-1]["rate"]


def _print_usage_and_pricing(model_name: str, usage_metadata) -> None:
    if usage_metadata is None:
        console.print(
            "No usage metadata returned; cannot compute usage or pricing.",
            style="yellow",
        )
        return

    prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    response_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
    api_total_tokens = getattr(usage_metadata, "total_token_count", None)
    calculated_total = prompt_tokens + response_tokens
    total_tokens = api_total_tokens if api_total_tokens is not None else calculated_total

    console.print(
        "Gemini token usage "
        f"prompt={prompt_tokens:,} (input you sent) • "
        f"response={response_tokens:,} (model outputs incl. reasoning/tool instructions) • "
        f"total={total_tokens:,} "
        "(API-reported total; might exceed prompt+response when hidden thinking tokens are counted)",
        style="bold cyan",
    )
    if api_total_tokens is not None and api_total_tokens != calculated_total:
        console.print(
            f"Prompt+response summed = {calculated_total:,}; API also counted "
            f"{api_total_tokens - calculated_total:,} internal tokens (e.g., hidden "
            "thinking/tool orchestration).",
            style="dim",
        )

    pricing = MODEL_PRICING_USD_PER_MTOKENS.get(model_name)
    if not pricing:
        console.print(
            f"No pricing table for {model_name}; update MODEL_PRICING_USD_PER_MTOKENS to see costs.",
            style="yellow",
        )
        return

    prompt_rate = _effective_rate(pricing.get("prompt"), prompt_tokens)
    response_rate = _effective_rate(pricing.get("response"), prompt_tokens)
    if prompt_rate is None or response_rate is None:
        console.print(
            "Pricing table is missing prompt/response tiers; update MODEL_PRICING_USD_PER_MTOKENS.",
            style="yellow",
        )
        return

    prompt_cost = _usd_cost(prompt_tokens, prompt_rate)
    response_cost = _usd_cost(response_tokens, response_rate)
    total_cost = prompt_cost + response_cost

    console.print(
        f"Estimated cost prompt=${prompt_cost:.6f} • response=${response_cost:.6f} • total=${total_cost:.6f}",
        style="bold green",
    )
    if note := pricing.get("notes"):
        console.print(note, style="dim")


def segment(
    content: str,
    client: genai.Client,
    *,
    model: str,
    use_code_execution: bool,
) -> tuple[str, Any | None]:
    """
    Ask Gemini to produce a Python program that defines `segments = [...]` and
    optionally runs the verifier via the code-execution tool.

    We collect all streamed text and code parts into a single Python source string.
    """
    prompt_text = build_prompt(use_code_execution)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text + content + "\n</INPUT>\n")],
        ),
    ]

    config_kwargs = {
        "thinking_config": types.ThinkingConfig(thinking_budget=-1),
        "system_instruction": SYSTEM_PROMPT,
    }
    if use_code_execution:
        config_kwargs["tools"] = [types.Tool(code_execution=types.ToolCodeExecution)]

    generate_content_config = types.GenerateContentConfig(**config_kwargs)

    str_chunks: list[str] = []
    usage_metadata = None

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.usage_metadata is not None:
            usage_metadata = chunk.usage_metadata

        if not chunk.candidates:
            continue
        content_obj = chunk.candidates[0].content
        if not content_obj or not getattr(content_obj, "parts", None):
            continue

        for part in content_obj.parts:
            # Collect plain text emitted by the model (sometimes code comes via text).
            if getattr(part, "text", None):
                print(part.text, end="")
                str_chunks.append(part.text)
            # Collect executable code blocks (use .code string).
            if getattr(part, "executable_code", None):
                code_str = getattr(part.executable_code, "code", None)
                if isinstance(code_str, str):
                    print(code_str)
                    str_chunks.append(code_str)
            # We ignore code_execution_result here; it's for display/logging only.

    _print_usage_and_pricing(model, usage_metadata)
    generated_code = "".join(str_chunks).strip()
    return generated_code, usage_metadata


_OUTER_CODE_FENCE_RE = re.compile(
    r"^\s*```[^\r\n]*\r?\n(?P<body>.*)\r?\n```\s*$",
    re.DOTALL,
)


def _strip_markdown_code_fences(text: str) -> str:
    """
    Gemini occasionally wraps the *entire* answer inside ```...```; peel only that shell.
    Embedded fences inside SOURCE_TEXT segments remain untouched.
    """
    match = _OUTER_CODE_FENCE_RE.match(text)
    if match:
        return match.group("body")
    return text


def _remove_standalone_code_fence_lines(code: str) -> str:
    """
    Drop stray ```lang fences that wrap model output but sit outside string literals.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except tokenize.TokenError:
        return code

    cleaned_tokens: list[tokenize.TokenInfo] = []
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok.type == tokenize.ERRORTOKEN and tok.string == "`" and tok.line.lstrip().startswith("```"):
            fence_line = tok.start[0]
            while idx < len(tokens) and tokens[idx].start[0] == fence_line:
                idx += 1
            cleaned_tokens.append(
                tokenize.TokenInfo(tokenize.NL, "\n", (fence_line, 0), (fence_line, 0), "\n")
            )
            continue
        cleaned_tokens.append(tok)
        idx += 1

    return tokenize.untokenize(cleaned_tokens)


def extract_segments_from_code_string(python_code_string: str):
    """
    Extracts the 'segments' variable (expected to be a list of dictionaries)
    from a Python code string using the ast module.
    """
    segments_value = None
    try:
        cleaned_code = _strip_markdown_code_fences(python_code_string)
        cleaned_code = _remove_standalone_code_fence_lines(cleaned_code)
        tree = ast.parse(cleaned_code)
        for node in ast.walk(tree):
            value_node = None
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "segments":
                        value_node = node.value
                        break
            elif isinstance(node, ast.AnnAssign):
                target = node.target
                if (
                    isinstance(target, ast.Name)
                    and target.id == "segments"
                    and node.value is not None
                ):
                    value_node = node.value

            if value_node is not None:
                try:
                    candidate = ast.literal_eval(value_node)
                except (ValueError, SyntaxError):
                    continue
                segments_value = candidate

        if segments_value is not None:
            return segments_value
        else:
            print("The 'segments' array was not found or could not be extracted from the code.")
            return None

    except SyntaxError as e:
        print(f"SyntaxError encountered during AST parsing: {e}")
        print("Please review the generated code for syntax errors.")
        return None
    except ValueError as e:
        print(f"ValueError encountered during literal evaluation of 'segments': {e}")
        print("The structure of 'segments' might not be a simple literal list of dictionaries.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def pretty_print_segments(segments) -> None:
    for segment in segments:
        seg_type = segment.get("type", "unknown")
        console.print(
            f"########## I think the text below is: {seg_type} #####",
            style="bold green",
        )
        print(segment.get("content", ""))


def verify_segments_locally(source_text: str, segments, *, fuzzy: bool) -> dict[str, Any]:
    duplicate_types = _detect_adjacent_duplicate_types(segments)
    if duplicate_types:
        unique_types = ", ".join(sorted(set(duplicate_types)))
        console.print(
            (
                f"Detected {len(duplicate_types)} adjacent segment(s) repeating type(s): "
                f"{unique_types}. Merging before coverage verification."
            ),
            style="yellow",
        )
    try:
        _assert_segments_cover_source(source_text, segments)
        return {"mode": "strict"}
    except AssertionError as exc:
        if not fuzzy:
            raise LocalVerificationError(str(exc)) from exc
        stitched_text = _stitch_segments_with_validation(segments)
        ok, info = _fuzzy_compare_source_and_segments(source_text, stitched_text)
        if not ok:
            reason = info.get("reason", "fuzzy_comparison_failed")
            raise LocalVerificationError(reason) from exc
        diff_desc = info.get("blocks", [])
        summary = "Fuzzy verification accepted" if diff_desc else "Fuzzy verification not needed"
        if diff_desc:
            block = diff_desc[0]
            summary = (
                "Fuzzy verification accepted minor diff "
                f"(tag={block['tag']}, source_range={block['source_range']}, diff_chars={info.get('diff_chars')})."
            )
        console.print(summary, style="yellow")
        return info
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        raise LocalVerificationError(str(exc)) from exc


def main() -> None:
    args = parse_args()
    api_key = resolve_api_key(args.api_key)
    proxy_url = args.proxy
    proxy_dict = _proxy_mapping(proxy_url)
    verify_tls = True
    if proxy_dict:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        verify_tls = False
        console.print(
            "Proxy mode enabled: skipping TLS certificate verification (insecure).",
            style="bold yellow",
        )
    content, input_source = read_input_text(args, proxies=proxy_dict, verify=verify_tls)
    timestamp_iso, run_id = _new_run_identifiers()
    metadata = {
        "timestamp": timestamp_iso,
        "model": args.model,
        "code_execution_enabled": args.code_exectution,
        "fuzzy_mode": args.fuzzy,
        "proxy": args.proxy,
        "proxy_skip_tls": not verify_tls,
        "input_source": input_source,
        "input_characters": len(content),
        "input_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "max_input_length": args.max_input_length,
        "input_was_truncated": INPUT_WAS_TRUNCATED,
    }
    verification_mode = "code_execution" if args.code_exectution else "local"
    http_options = None
    if proxy_dict:
        console.print(
            f"Routing Gemini API calls through proxy {proxy_url}",
            style="dim",
        )
        sync_httpx_client = httpx.Client(proxy=proxy_url, verify=verify_tls)
        async_httpx_client = httpx.AsyncClient(proxy=proxy_url, verify=verify_tls)
        http_options = types.HttpOptions(
            httpx_client=sync_httpx_client,
            httpx_async_client=async_httpx_client,
        )
    client = genai.Client(api_key=api_key, http_options=http_options)

    generated_code, usage_metadata = segment(
        content,
        client=client,
        model=args.model,
        use_code_execution=args.code_exectution,
    )
    metadata["usage_metadata"] = _serialize_usage_metadata(usage_metadata)
    if not generated_code:
        error_message = "Model returned no code to parse."
        log_model_output(
            run_id=run_id,
            metadata=metadata,
            generated_code=generated_code,
            status="empty_response",
            verification_mode=verification_mode,
            segments=None,
            error=error_message,
        )
        console.print(
            "The model did not return any code/text to parse.",
            style="yellow",
        )
        return

    segments_result = extract_segments_from_code_string(generated_code)
    log_status = "segments_not_extracted"
    error_message: str | None = None
    verification_details: dict[str, Any] | None = None

    try:
        if segments_result is not None:
            log_status = "segments_extracted"
            print("Successfully extracted 'segments' array:")
            print(segments_result)
            pretty_print_segments(segments_result)
            newline_healing_info = None
            if not args.code_exectution:
                newline_healing_info = _heal_newline_discrepancies(content, segments_result)
                if newline_healing_info.get("applied_patches"):
                    metadata["newline_healing"] = newline_healing_info
                    console.print(
                        (
                            "Normalized {count} newline-only diff(s) before local verification."
                        ).format(count=newline_healing_info["applied_patches"]),
                        style="dim",
                    )
            if not args.code_exectution:
                try:
                    verification_details = verify_segments_locally(
                        content,
                        segments_result,
                        fuzzy=args.fuzzy,
                    )
                    if verification_details.get("mode") == "fuzzy":
                        log_status = "local_verification_fuzzy_passed"
                    else:
                        log_status = "local_verification_passed"
                except LocalVerificationError as exc:
                    error_message = f"Local coverage verification failed: {exc}"
                    log_status = "local_verification_failed"
                    console.print(error_message, style=LOCAL_VERIFICATION_ERROR_STYLE)
                    raise SystemExit(1) from exc
            else:
                console.print(
                    "Skipped local verification because --code-exectution was provided.",
                    style="dim",
                )
                log_status = "delegated_verification"
        else:
            error_message = "Failed to extract 'segments' array from code."
            print("Failed to extract 'segments' array from code.")
    finally:
        if verification_details is not None:
            metadata["local_verification"] = verification_details
        log_model_output(
            run_id=run_id,
            metadata=metadata,
            generated_code=generated_code,
            status=log_status,
            verification_mode=verification_mode,
            segments=segments_result,
            error=error_message,
        )
        if segments_result is not None and error_message is None:
            save_segments_snapshot(run_id, metadata, segments_result)

    if INPUT_WAS_TRUNCATED:
        console.print(
            f"Warning: Input was truncated to {args.max_input_length:,} characters due to --max-input-length.",
            style="bold yellow",
        )


if __name__ == "__main__":
    main()
