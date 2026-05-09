from __future__ import annotations

import argparse
import ast
import copy
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

from dotenv import load_dotenv
load_dotenv()

import httpx
import requests
from google import genai
from google.genai import types
import urllib3
from rich.console import Console

console = Console()

DEFAULT_MODEL = "gemini-3-flash-preview"


class LocalVerificationError(RuntimeError):
    """Raised when local coverage verification fails."""

    pass


LOCAL_VERIFICATION_ERROR_STYLE = "bold red"

FUZZY_MAX_BLOCKS: int | None = None  # None -> unlimited adjustments
FUZZY_MAX_BLOCK_SIZE: int | None = None  # None -> no per-block cap
FUZZY_MIN_RATIO: float | None = None  # None -> accept any similarity
FUZZY_ALLOW_MIXED_DIFF_TYPES = True

REPLACEMENT_CHARACTER = "\uFFFD"
REPLACEMENT_SUBSTITUTE = "\u00A4"  # ¤

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
    "gemini-3-flash-preview": {
        "prompt": [
            {"max_prompt_tokens": None, "rate": 0.50},
        ],
        "response": [
            {"max_prompt_tokens": None, "rate": 3.00},
        ],
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
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
    "makefile",
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
_LITERAL_ESCAPE_PATTERN = re.compile(
    r"\\(?:[\\\"'abfnrtv]|x[0-9A-Fa-f]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8})"
)
_CONTENT_BLOCK_RE = re.compile(
    r"<CONTENT-TYPE:(?P<type>[A-Za-z0-9_\-+.]+)>"
)
_EMBEDDED_MARKER_PATTERN = re.compile(
    r"<CONTENT-TYPE:[A-Za-z0-9_\-+.]+>(?:\r?\n)?"
)


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
        if (
            seg_type not in _ALLOWED_TYPES
            and not seg_type.startswith("discovered_")
            and not seg_type.startswith("other_")
        ):
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


def _diff_stats(source_text: str, stitched_text: str) -> tuple[int, float, list]:
    matcher = SequenceMatcher(a=source_text, b=stitched_text, autojunk=False)
    diff_blocks = []
    total_diff = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        diff_blocks.append(
            {
                "tag": tag,
                "source_range": [i1, i2],
                "segments_range": [j1, j2],
                "source_snippet": source_text[i1:i2],
                "segments_snippet": stitched_text[j1:j2],
            }
        )
        total_diff += max(i2 - i1, j2 - j1)
    return total_diff, matcher.ratio(), diff_blocks


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


def _strip_inserted_text(segments, diff_blocks):
    """
    Remove bytes that the model inserted which are not present in SOURCE_TEXT.
    Returns a summary dictionary describing what was removed.
    """
    insert_blocks = [block for block in diff_blocks or [] if block.get("tag") == "insert"]
    if not insert_blocks:
        return {"removed_bytes": 0, "blocks": 0, "empty_segments_removed": 0}
    removed_bytes = 0
    # Remove from the end backward so offsets remain valid.
    for block in sorted(
        insert_blocks,
        key=lambda b: (b.get("segments_range", [0, 0])[0], b.get("segments_range", [0, 0])[1]),
        reverse=True,
    ):
        seg_range = block.get("segments_range") or [0, 0]
        if not isinstance(seg_range, list) or len(seg_range) != 2:
            continue
        start, end = seg_range
        if not isinstance(start, int) or not isinstance(end, int) or end <= start:
            continue
        _remove_text_range_in_segments(segments, start, end)
        removed_bytes += end - start
    cleaned_segments, empty_removed = _remove_empty_segments(segments)
    if empty_removed:
        segments[:] = cleaned_segments
    return {
        "removed_bytes": removed_bytes,
        "blocks": len(insert_blocks),
        "empty_segments_removed": empty_removed,
    }


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


def _decode_literal_escape_sequences(text: str) -> str | None:
    """
    Best-effort decoding of literal escape sequences (e.g., \" -> ", \\n -> newline).
    Returns None when decoding fails.
    """
    try:
        raw = text.encode("latin-1", "backslashreplace")
        return raw.decode("unicode_escape")
    except UnicodeDecodeError:
        return None


def _maybe_apply_literal_escape_fix(source_text: str, segments):
    """
    Detects when the model emitted literal backslash escapes (e.g., \" or \\n) instead of
    real characters and attempts to decode them. Only adopts the decoded segments when
    similarity to the source text improves.
    """
    stitched_original = "".join(entry.get("content", "") for entry in segments)
    if "\\" not in stitched_original:
        return {"applied": False}

    decoded_segments = []
    changed = False
    for entry in segments:
        content = entry.get("content", "")
        if not isinstance(content, str):
            return {"applied": False}
        if not _LITERAL_ESCAPE_PATTERN.search(content):
            decoded_segments.append(entry)
            continue
        decoded = _decode_literal_escape_sequences(content)
        if decoded is None or decoded == content:
            decoded_segments.append(entry)
            continue
        new_entry = dict(entry)
        new_entry["content"] = decoded
        decoded_segments.append(new_entry)
        changed = True

    if not changed:
        return {"applied": False}

    stitched_decoded = "".join(entry.get("content", "") for entry in decoded_segments)
    original_diff, original_similarity, _ = _diff_stats(source_text, stitched_original)
    decoded_diff, decoded_similarity, _ = _diff_stats(source_text, stitched_decoded)

    if decoded_diff < original_diff or decoded_similarity > original_similarity:
        return {
            "applied": True,
            "segments": decoded_segments,
            "diff_before": original_diff,
            "diff_after": decoded_diff,
            "similarity_before": original_similarity,
            "similarity_after": decoded_similarity,
        }

    return {"applied": False}


def _maybe_restore_backslashes(source_text: str, segments):
    """
    Restores missing literal backslashes (\\) when the model stripped them from content.
    Only applies when every diff block between SOURCE_TEXT and stitched segments involves
    backslashes exclusively.
    """
    if "\\" not in source_text:
        return {"applied": False}
    stitched_text = "".join(entry.get("content", "") for entry in segments)
    if stitched_text == source_text:
        return {"applied": False}

    working_segments = copy.deepcopy(segments)
    matcher = SequenceMatcher(a=source_text, b=stitched_text, autojunk=False)
    delta = 0
    inserted = 0
    removed = 0
    applied = False

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        src_slice = source_text[i1:i2]
        seg_slice = stitched_text[j1:j2]
        target_start = j1 + delta
        target_end = j2 + delta

        def only_backslashes(text: str) -> bool:
            return text and set(text) <= {"\\"}

        if tag == "delete":
            if not only_backslashes(src_slice):
                return {"applied": False}
            _insert_text_at_pos(working_segments, target_start, src_slice)
            delta += len(src_slice)
            inserted += len(src_slice)
            applied = True
        elif tag == "insert":
            if not only_backslashes(seg_slice):
                return {"applied": False}
            _remove_text_range_in_segments(working_segments, target_start, target_end)
            delta -= len(seg_slice)
            removed += len(seg_slice)
            applied = True
        elif tag == "replace":
            if (src_slice and not only_backslashes(src_slice)) or (
                seg_slice and not only_backslashes(seg_slice)
            ):
                return {"applied": False}
            if seg_slice:
                _remove_text_range_in_segments(working_segments, target_start, target_end)
                delta -= len(seg_slice)
                removed += len(seg_slice)
            if src_slice:
                _insert_text_at_pos(working_segments, target_start, src_slice)
                delta += len(src_slice)
                inserted += len(src_slice)
            applied = True
        else:
            return {"applied": False}

    if not applied:
        return {"applied": False}

    new_stitched = "".join(entry.get("content", "") for entry in working_segments)
    diff_before, similarity_before, _ = _diff_stats(source_text, stitched_text)
    diff_after, similarity_after, _ = _diff_stats(source_text, new_stitched)

    if diff_after > diff_before:
        return {"applied": False}

    return {
        "applied": True,
        "segments": working_segments,
        "diff_before": diff_before,
        "diff_after": diff_after,
        "similarity_before": similarity_before,
        "similarity_after": similarity_after,
        "inserted": inserted,
        "removed": removed,
    }


def _maybe_encode_literal_sequences(source_text: str, segments):
    """
    When SOURCE_TEXT stores escape sequences literally (e.g., '\\n') but the model
    emitted the decoded characters (e.g., newline), convert the segments back to
    the literal form so coverage matches exactly.
    Applies only when every diff block decodes cleanly via unicode_escape.
    """
    stitched_text = "".join(entry.get("content", "") for entry in segments)
    if stitched_text == source_text:
        return {"applied": False}

    matcher = SequenceMatcher(a=source_text, b=stitched_text, autojunk=False)
    replacements: list[tuple[int, int, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        src_slice = source_text[i1:i2]
        seg_slice = stitched_text[j1:j2]
        try:
            decoded = bytes(src_slice, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return {"applied": False}
        if decoded != seg_slice:
            return {"applied": False}
        replacements.append((j1, j2, src_slice))

    if not replacements:
        return {"applied": False}

    working_segments = copy.deepcopy(segments)
    for start, end, replacement in sorted(replacements, key=lambda item: (item[0], item[1]), reverse=True):
        _replace_text_range_in_segments(working_segments, start, end, replacement)

    stitched_after = "".join(entry.get("content", "") for entry in working_segments)
    diff_before, similarity_before, _ = _diff_stats(source_text, stitched_text)
    diff_after, similarity_after, _ = _diff_stats(source_text, stitched_after)
    if diff_after >= diff_before:
        return {"applied": False}

    return {
        "applied": True,
        "segments": working_segments,
        "diff_before": diff_before,
        "diff_after": diff_after,
        "similarity_before": similarity_before,
        "similarity_after": similarity_after,
    }


def _heal_small_replacements(source_text: str, segments, *, max_chars: int = 64) -> dict[str, Any]:
    """
    Repairs tiny replace/delete diffs by overwriting segments with the expected SOURCE_TEXT
    bytes when the differing spans are short (≤ max_chars).
    """
    stitched_text = "".join(entry.get("content", "") for entry in segments)
    matcher = SequenceMatcher(a=source_text, b=stitched_text, autojunk=False)
    patches = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("replace", "delete"):
            continue
        src_len = i2 - i1
        seg_len = j2 - j1
        if max(src_len, seg_len) > max_chars:
            continue
        expected = source_text[i1:i2]
        if tag == "replace":
            _replace_text_range_in_segments(segments, j1, j2, expected)
        elif tag == "delete":
            _insert_text_at_pos(segments, j1, expected)
        patches.append(
            {
                "tag": tag,
                "source_range": [i1, i2],
                "segments_range": [j1, j2],
                "expected": expected,
            }
        )
    if not patches:
        return {"applied": False}
    return {"applied": True, "patches": patches, "count": len(patches)}


def _remove_empty_segments(segments):
    cleaned = [entry for entry in segments if isinstance(entry.get("content"), str) and entry.get("content")]
    removed = len(segments) - len(cleaned)
    return cleaned, removed


def _require_exact_source_match(source_text: str, segments, *, diff_hint: dict[str, Any] | None = None) -> None:
    """
    Enforce that stitched segments reproduce SOURCE_TEXT exactly.
    Raises LocalVerificationError with a brief diff summary on mismatch.
    """
    stitched_text = _stitch_segments_with_validation(segments)
    if stitched_text == source_text:
        return
    _, info = _fuzzy_compare_source_and_segments(source_text, stitched_text)
    diff_info = diff_hint or info or {}
    blocks = diff_info.get("blocks") or []
    first_block = blocks[0] if blocks else {}
    tag = first_block.get("tag", "unknown")
    src_range = first_block.get("source_range") or [0, 0]
    seg_range = first_block.get("segments_range") or [0, 0]
    src_snip = first_block.get("source_snippet", "")
    seg_snip = first_block.get("segments_snippet", "")
    detail: str
    if tag == "delete":
        detail = (
            f"expected {src_snip!r} at source[{src_range[0]}:{src_range[1]}] "
            f"but it is missing in segments (segments[{seg_range[0]}:{seg_range[1]}])"
        )
    elif tag == "insert":
        detail = (
            f"segments inserted {seg_snip!r} at segments[{seg_range[0]}:{seg_range[1]}] "
            f"(no bytes exist at source[{src_range[0]}:{src_range[1]}])"
        )
    elif tag == "replace":
        detail = (
            f"expected {src_snip!r} at source[{src_range[0]}:{src_range[1]}] "
            f"but segments have {seg_snip!r} at segments[{seg_range[0]}:{seg_range[1]}]"
        )
    else:
        detail = (
            f"mismatch tag={tag} source[{src_range[0]}:{src_range[1]}] "
            f"segments[{seg_range[0]}:{seg_range[1]}] expected {src_snip!r} got {seg_snip!r}"
        )
    raise LocalVerificationError(
        f"Exact coverage mismatch: {detail}; total_diff_chars={diff_info.get('diff_chars')}."
    )


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
    content_drop_chars = 0
    max_block_span = 0
    diff_tags: set[str] = set()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        src_len = i2 - i1
        seg_len = j2 - j1
        block = {
            "tag": tag,
            "source_range": [i1, i2],
            "segments_range": [j1, j2],
            "source_snippet": source_text[i1:i2],
            "segments_snippet": stitched_text[j1:j2],
        }
        diff_blocks.append(block)
        diff_tags.add(tag)
        max_block_span = max(max_block_span, src_len, seg_len)
        if src_len > seg_len:
            content_drop_chars += src_len - seg_len
        total_diff += max(src_len, seg_len)

    similarity = matcher.ratio()
    if not diff_blocks:
        return True, {
            "mode": "strict",
            "diff_chars": 0,
            "blocks": [],
            "similarity": similarity,
            "reason": "exact_match",
            "has_content_drop": False,
            "content_drop_chars": 0,
        }

    exceeded_limits: dict[str, Any] = {}
    if FUZZY_MAX_BLOCKS is not None and len(diff_blocks) > FUZZY_MAX_BLOCKS:
        exceeded_limits["max_blocks"] = len(diff_blocks)
    if FUZZY_MAX_BLOCK_SIZE is not None and max_block_span > FUZZY_MAX_BLOCK_SIZE:
        exceeded_limits["max_block_size"] = max_block_span
    if FUZZY_MIN_RATIO is not None and similarity < FUZZY_MIN_RATIO:
        exceeded_limits["similarity"] = similarity
    if not FUZZY_ALLOW_MIXED_DIFF_TYPES and len(diff_tags) > 1:
        exceeded_limits["mixed_diff_types"] = sorted(diff_tags)

    info: dict[str, Any] = {
        "mode": "fuzzy",
        "diff_chars": total_diff,
        "blocks": diff_blocks,
        "similarity": similarity,
        "reason": "diff_detected",
        "has_content_drop": content_drop_chars > 0,
        "content_drop_chars": content_drop_chars,
        "unique_diff_tags": sorted(diff_tags),
        "max_block_span": max_block_span,
    }
    if exceeded_limits:
        info["exceeded_limits"] = exceeded_limits
    if info["has_content_drop"] and diff_blocks:
        info["content_drop_warning"] = {
            "block": diff_blocks[0],
            "total_diff_chars": total_diff,
        }
    return True, info


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
                ) and not seg_type.startswith("other_"):
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

Keep this literal text (do not escape the characters):

Path: C:\Tools\bin
Quote: "Segment everything exactly once"
</INPUT>
<OUTPUT>
segments = [
  {
    "type": "markdown",
    "content": """# This is how you use magika
Run
```"""
  },
  {
    "type": "shell",
    "content": "pip install magika"
  },
  {
    "type": "markdown",
    "content": """```
then open an editor and type:
```"""
  },
  {
    "type": "python",
    "content": """import magika
def adfadf():
return True"""
  },
  {
    "type": "markdown",
    "content": """```
"""
  },
  {
    "type": "markdown",
    "content": """Keep this literal text (do not escape the characters):

Path: C:\\Tools\\bin
Quote: "Segment everything exactly once"
"""
  }
]
</OUTPUT>

Notice how the markdown snippet above keeps `Path: C:\\\\Tools\\\\bin` verbatim and the HTML snippet below keeps `<div class=\\\"stats muted\\\">` rather than decoding the escapes; always retain those literal backslash sequences exactly as SOURCE_TEXT provides them.

Example (React JSX with custom component and inline JS/CSS):
<INPUT>
return <SomeView onClick="alert('hi')" style="color: red;">Save</SomeView>;
</INPUT>
<OUTPUT>
segments = [
  {
    "type": "javascript",
    "content": "return "
  },
  {
    "type": "other_jsx",
    "content": "<SomeView onClick=\\""
  },
  {
    "type": "javascript",
    "content": "alert('hi')"
  },
  {
    "type": "other_jsx",
    "content": "\\" style=\\""
  },
  {
    "type": "css",
    "content": "color: red;"
  },
  {
    "type": "other_jsx",
    "content": "\\">Save</SomeView>"
  },
  {
    "type": "javascript",
    "content": ";"
  }
]
</OUTPUT>

Example (Angular template string with Angular-only bits tagged as other_angular_template):
<INPUT>
const tpl = `
<div class="card" *ngIf="hasCard">
  <h1>{{ title }}</h1>
  <button (click)="save()">Save</button>
</div>`;
</INPUT>
<OUTPUT>
segments = [
  {
    "type": "javascript",
    "content": "const tpl = `\\n"
  },
  {
    "type": "other_angular_template",
    "content": "<div class=\\"card\\" *ngIf=\\""
  },
  {
    "type": "javascript",
    "content": "hasCard"
  },
  {
    "type": "other_angular_template",
    "content": "\\">\\n  <h1>{{ "
  },
  {
    "type": "javascript",
    "content": "title"
  },
  {
    "type": "other_angular_template",
    "content": " }}</h1>\\n  <button (click)=\\""
  },
  {
    "type": "javascript",
    "content": "save()"
  },
  {
    "type": "other_angular_template",
    "content": "\\">Save</button>\\n</div>`;"
  }
]
</OUTPUT>

Example (Django template with inline CSS; keep template syntax as other_django_template):
<INPUT>
{% extends "base.html" %}
{% block content %}
<div class="card" style="color: red;">
  Hello {{ user.name|default:"Anonymous" }}
</div>
{% endblock %}
</INPUT>
<OUTPUT>
segments = [
  {
    "type": "other_django_template",
    "content": """{% extends "base.html" %}
{% block content %}
<div class="card" style=\\""""
  },
  {
    "type": "css",
    "content": "color: red;"
  },
  {
    "type": "other_django_template",
    "content": """\\">
  Hello {{ user.name|default:"Anonymous" }}
</div>
{% endblock %}
"""
  }
]
</OUTPUT>

Example (HTML page script + inline handlers):
<INPUT>
<script type="text/javascript">
$(function() {
  initMenu('',true,false,'search.php','Search');
  $(document).ready(function() { init_search(); });
});
</script>
<div id="MSearchSelectWindow"
     onmouseover="return searchBox.OnSearchSelectShow()"
     onmouseout="return searchBox.OnSearchSelectHide()"
     onkeydown="return searchBox.OnSearchSelectKey(event)">
</div>
</INPUT>
<OUTPUT>
segments = [
  {
    "type": "html",
    "content": "<script type=\\"text/javascript\\">\\n"
  },
  {
    "type": "javascript",
    "content": "$(function() {\\n  initMenu('',true,false,'search.php','Search');\\n  $(document).ready(function() { init_search(); });\\n});\\n"
  },
  {
    "type": "html",
    "content": "</script>\\n<div id=\\"MSearchSelectWindow\\"\\n     onmouseover=\\""
  },
  {
    "type": "javascript",
    "content": "return searchBox.OnSearchSelectShow()"
  },
  {
    "type": "html",
    "content": "\\"\\n     onmouseout=\\""
  },
  {
    "type": "javascript",
    "content": "return searchBox.OnSearchSelectHide()"
  },
  {
    "type": "html",
    "content": "\\"\\n     onkeydown=\\""
  },
  {
    "type": "javascript",
    "content": "return searchBox.OnSearchSelectKey(event)"
  },
  {
    "type": "html",
    "content": "\\">\\n</div>\\n"
  }
]
</OUTPUT>

Your turn:
<INPUT>
'''

LOCAL_RESPONSE_EXAMPLE = '''
Example (HTML with embedded CSS):
<INPUT>
<div class="stats muted">
  <style>
    body {
      color: crimson;
    }
  </style>
</div>
</INPUT>
<OUTPUT>
<CONTENT-TYPE:html><div class="stats muted">
  <style>
<CONTENT-TYPE:css>    body {
      color: crimson;
    }
<CONTENT-TYPE:html>  </style>
</div>
</OUTPUT>


Example (HTML/CSS inside a JavaScript string literal):
<INPUT>
const snippet = "<div><style>p{color:red;}</style></div>";
</INPUT>
<OUTPUT>
<CONTENT-TYPE:javascript>const snippet = "<CONTENT-TYPE:html><div><style><CONTENT-TYPE:css>p{color:red;}<CONTENT-TYPE:html></style></div><CONTENT-TYPE:javascript>";
</OUTPUT>

Example (inline HTML style attribute):
<INPUT>
<div style="color:red; background: #fff;">Hello</div>
</INPUT>
<OUTPUT>
<CONTENT-TYPE:html><div style="<CONTENT-TYPE:css>color:red; background: #fff;<CONTENT-TYPE:html>">Hello</div>
</OUTPUT>

Example (Dockerfile RUN with inline shell):
<INPUT>
FROM python:3.11-slim
RUN set -eux; \
    apt-get update; \
    pip install --no-cache-dir flask==3.0.2
CMD ["python","app.py"]
</INPUT>
<OUTPUT>
<CONTENT-TYPE:dockerfile>FROM python:3.11-slim
RUN <CONTENT-TYPE:shell>set -eux; \
    apt-get update; \
    pip install --no-cache-dir flask==3.0.2
<CONTENT-TYPE:dockerfile>
CMD ["python","app.py"]
</OUTPUT>

Example (HTML page script + inline handlers):
<INPUT>
<script type="text/javascript">
$(function() {
  initMenu('',true,false,'search.php','Search');
  $(document).ready(function() { init_search(); });
});
</script>
<div id="MSearchSelectWindow"
     onmouseover="return searchBox.OnSearchSelectShow()"
     onmouseout="return searchBox.OnSearchSelectHide()"
     onkeydown="return searchBox.OnSearchSelectKey(event)">
</div>
</INPUT>
<OUTPUT>
<CONTENT-TYPE:html><script type="text/javascript">
<CONTENT-TYPE:javascript>$(function() {
  initMenu('',true,false,'search.php','Search');
  $(document).ready(function() { init_search(); });
});
<CONTENT-TYPE:html>
</script>
<div id="MSearchSelectWindow"
     onmouseover="<CONTENT-TYPE:javascript>return searchBox.OnSearchSelectShow()<CONTENT-TYPE:html>"
     onmouseout="<CONTENT-TYPE:javascript>return searchBox.OnSearchSelectHide()<CONTENT-TYPE:html>"
     onkeydown="<CONTENT-TYPE:javascript>return searchBox.OnSearchSelectKey(event)<CONTENT-TYPE:html>">
</div>
</OUTPUT>

Example (Makefile rule with inline shell):
<INPUT>
cmd_/tools/include/xen/.install := /bin/sh scripts/headers_install.sh; echo done
</INPUT>
<OUTPUT>
<CONTENT-TYPE:makefile>cmd_/tools/include/xen/.install := <CONTENT-TYPE:shell>/bin/sh scripts/headers_install.sh; echo done
</OUTPUT>

Example (HTML consecutive style blocks on one line):
<INPUT>
<style type="text/css">.a{color:red;}</style><style>.b{color:blue;}</style>
</INPUT>
<OUTPUT>
<CONTENT-TYPE:html><style type="text/css"><CONTENT-TYPE:css>.a{color:red;}<CONTENT-TYPE:html></style><style><CONTENT-TYPE:css>.b{color:blue;}<CONTENT-TYPE:html></style>
</OUTPUT>

Example (HTML table with inline style attributes):
<INPUT>
<td style="width:200px;"><strong>Parameter</strong></td><td style="width:500px;">Description</td>
</INPUT>
<OUTPUT>
<CONTENT-TYPE:html><td style="<CONTENT-TYPE:css>width:200px;<CONTENT-TYPE:html>"><strong>Parameter</strong></td><td style="<CONTENT-TYPE:css>width:500px;<CONTENT-TYPE:html>">Description</td>
</OUTPUT>

Example (Windows batch script → shell):
<INPUT>
@echo off
set FOO=bar
if defined FOO echo %FOO%
goto :END
:END
</INPUT>
<OUTPUT>
<CONTENT-TYPE:shell>@echo off
set FOO=bar
if defined FOO echo %FOO%
goto :END
:END
</OUTPUT>

Example (mixed Markdown + shell + python):
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

Keep this literal text (do not escape the characters):

Path: C:\Tools\bin
Quote: "Segment everything exactly once"
</INPUT>
<OUTPUT>
<CONTENT-TYPE:markdown># This is how you use magika
Run
```
<CONTENT-TYPE:shell>pip install magika
<CONTENT-TYPE:markdown>
```
then open an editor and type:
```
<CONTENT-TYPE:python>import magika
def adfadf():
return True
<CONTENT-TYPE:markdown>
````
Keep this literal text (do not escape the characters):

Path: C:\Tools\bin
Quote: "Segment everything exactly once"
</OUTPUT>

Example (React JSX with custom component and inline JS/CSS):
<INPUT>
return <SomeView onClick="alert('hi')" style="color: red;">Save</SomeView>;
</INPUT>
<OUTPUT>
<CONTENT-TYPE:javascript>return <CONTENT-TYPE:other_jsx><SomeView onClick="<CONTENT-TYPE:javascript>alert('hi')<CONTENT-TYPE:other_jsx>" style="<CONTENT-TYPE:css>color: red;<CONTENT-TYPE:other_jsx>">Save</SomeView><CONTENT-TYPE:javascript>;
</OUTPUT>

Example (Angular template string with Angular-only bits tagged as other_angular_template):
<INPUT>
const tpl = `
<div class="card" *ngIf="hasCard">
  <h1>{{ title }}</h1>
  <button (click)="save()">Save</button>
</div>`;
</INPUT>
<OUTPUT>
<CONTENT-TYPE:javascript>const tpl = `
<CONTENT-TYPE:other_angular_template><div class="card" *ngIf="<CONTENT-TYPE:javascript>hasCard<CONTENT-TYPE:other_angular_template>">
  <h1>{{ <CONTENT-TYPE:javascript>title<CONTENT-TYPE:other_angular_template> }}</h1>
  <button (click)="<CONTENT-TYPE:javascript>save()<CONTENT-TYPE:other_angular_template>">Save</button>
</div>`;<CONTENT-TYPE:javascript>
</OUTPUT>

Example (Django template with inline CSS):
<INPUT>
{% block body %}
<div class="card" style="color: red;">
  Hello {{ user.name|default:"Anonymous" }}
</div>
{% endblock %}
</INPUT>
<OUTPUT>
<CONTENT-TYPE:other_django_template>{% block body %}
<CONTENT-TYPE:html>
<div class="card" style="<CONTENT-TYPE:css>color: red;<CONTENT-TYPE:html>">
  Hello <CONTENT-TYPE:other_django_template>{{ user.name|default:"Anonymous" }}<CONTENT-TYPE:html>
</div>
<CONTENT-TYPE:other_django_template>{% endblock %}
</OUTPUT>


Markers are never part of SOURCE_TEXT—they simply bracket each block. Place them inline or on their own lines as needed, but never introduce or delete whitespace/bytes around them. Every byte between markers must be copied verbatim from SOURCE_TEXT (including blank lines, spaces, and literal escape sequences).

Your turn:
<INPUT>
'''

def load_dense_prompt() -> tuple[str, str]:
    import os
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dense_prompt.template")
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("=== USER ===\n")
    system_prompt = parts[0].replace("=== SYSTEM ===\n", "").strip()
    user_prompt = parts[1].strip() + "\n"
    return system_prompt, user_prompt

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
        description="Stream Gemini responses that segment content into typed blocks."
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



def _normalize_question_mark_characters(text: str) -> tuple[str, dict[str, Any] | None]:
    """Replace Unicode replacement characters with ¤ immediately after reading."""
    count = text.count(REPLACEMENT_CHARACTER)
    if not count:
        return text, None
    sanitized = text.replace(REPLACEMENT_CHARACTER, REPLACEMENT_SUBSTITUTE)
    return sanitized, {
        "code_point": "U+FFFD",
        "replacement": "U+00A4",
        "count": count,
    }


def read_input_text(
    args: argparse.Namespace,
    *,
    proxies: dict[str, str] | None = None,
    verify: bool = True,
) -> tuple[str, str, dict[str, Any] | None]:
    def _prepare(text: str, source_label: str) -> tuple[str, str, dict[str, Any] | None]:
        sanitized, normalization_info = _normalize_question_mark_characters(text)
        if normalization_info:
            console.print(
                (
                    "Replaced {count} occurrence(s) of the replacement character (�) with ¤ "
                    "right after reading {source}."
                ).format(count=normalization_info["count"], source=source_label),
                style="dim",
            )
        limited = _limit_input_length(sanitized, args.max_input_length, source=source_label)
        return limited, source_label, normalization_info

    if args.text:
        return _prepare(args.text, "--text input")
    if args.file:
        try:
            file_text = args.file.read_text()
            return _prepare(file_text, f"contents of {args.file}")
        except OSError as exc:
            raise SystemExit(f"[red]Failed to read file {args.file}: {exc}[/red]")
    if args.url:
        try:
            response = requests.get(args.url, proxies=proxies, verify=verify)
            response.raise_for_status()
            return _prepare(response.text, f"response from {args.url}")
        except requests.RequestException as exc:
            raise SystemExit(f"[red]Failed to fetch URL {args.url}: {exc}[/red]")
    if args.stdin:
        data = sys.stdin.read()
        if not data:
            raise SystemExit("[red]STDIN was selected but no data was provided.[/red]")
        return _prepare(data, "STDIN input")
    console.print(
        "No input source provided; using built-in Dockerfile sample.",
        style="yellow",
    )
    return _prepare(DEFAULT_SAMPLE_INPUT, "built-in Dockerfile sample")


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
    system_prompt, user_prompt = load_dense_prompt()
    prompt_text = user_prompt
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt_text + content + "\n</INPUT>\n")],
        ),
    ]

    config_kwargs = {
        "thinking_config": types.ThinkingConfig(thinking_budget=-1),
        "system_instruction": system_prompt,
    }

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
            if getattr(part, "text", None):
                print(part.text, end="")
                str_chunks.append(part.text)

    _print_usage_and_pricing(model, usage_metadata)
    generated_code = "".join(str_chunks)
    return generated_code, usage_metadata


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


def _strip_optional_output_wrapper(text: str) -> str:
    """
    Removes a surrounding <OUTPUT>...</OUTPUT> envelope when it encloses the whole response.
    """
    start_tag = "<OUTPUT>"
    end_tag = "</OUTPUT>"
    start_idx = text.find(start_tag)
    end_idx = text.rfind(end_tag)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return text
    leading = text[:start_idx]
    trailing = text[end_idx + len(end_tag) :]
    if leading.strip() or trailing.strip():
        # Non-whitespace outside the wrapper means these tokens belong to SOURCE_TEXT.
        return text
    return text[start_idx + len(start_tag) : end_idx]


def _skip_single_linebreak(text: str, pos: int) -> int:
    """
    Advances past at most one newline sequence (\\n or \\r\\n) starting at pos.
    """
    if pos >= len(text):
        return pos
    if text[pos] == "\r":
        pos += 1
        if pos < len(text) and text[pos] == "\n":
            pos += 1
        return pos
    if text[pos] == "\n":
        return pos + 1
    return pos


def extract_segments_from_content_blocks(model_text: str) -> list[dict[str, str]] | None:
    """
    Parses responses that follow the CONTENT-TYPE block protocol:

    <CONTENT-TYPE:markdown>
    ...bytes...
    <CONTENT-TYPE:python>
    ...bytes...

    Returns a list of {type, content} dictionaries or None when parsing fails.
    """
    cleaned_text = _strip_markdown_code_fences(model_text)
    matches = list(_CONTENT_BLOCK_RE.finditer(cleaned_text))
    if not matches:
        wrapped_text = _strip_optional_output_wrapper(cleaned_text)
        if wrapped_text != cleaned_text:
            cleaned_text = wrapped_text
            matches = list(_CONTENT_BLOCK_RE.finditer(cleaned_text))
    if not matches:
        return None
    prefix = cleaned_text[: matches[0].start()]
    if prefix.strip():
        return None
    segments: list[dict[str, str]] = []
    for idx, match in enumerate(matches):
        seg_type = match.group("type").strip()
        block_start = _skip_single_linebreak(cleaned_text, match.end())
        block_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned_text)
        block_content = cleaned_text[block_start:block_end]
        segments.append(
            {
                "type": seg_type.lower(),
                "content": block_content,
            }
        )
    return segments


def _strip_embedded_markers_from_segments(segments):
    """
    Removes stray CONTENT-TYPE markers that Gemini accidentally injected inside
    segment bodies. These markers are control tokens and should never appear in
    SOURCE_TEXT, so dropping them brings us back in sync with the input bytes.
    """
    removals: list[dict[str, int]] = []
    total_removed = 0
    for idx, entry in enumerate(segments):
        content = entry.get("content")
        if not isinstance(content, str):
            continue
        cleaned, count = _EMBEDDED_MARKER_PATTERN.subn("", content)
        if count:
            entry["content"] = cleaned
            removals.append({"index": idx, "removed_markers": count})
            total_removed += count
    if not removals:
        return segments, None
    return segments, {"total_removed": total_removed, "segments": removals}


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


def _print_diff_report(details: dict[str, Any]) -> None:
    blocks = details.get("blocks") or []
    if not blocks:
        return
    has_drop = details.get("has_content_drop", False)
    header_style = "bold red" if has_drop else "bold yellow"
    intro = (
        "!!! WARNING: Local verification detected missing SOURCE_TEXT bytes. Proceeding with fuzzy coverage."
        if has_drop
        else "Fuzzy verification differences (bytes left unmatched):"
    )
    console.print(f"\n{intro}", style=header_style)
    diff_chars = details.get("diff_chars")
    similarity = details.get("similarity")
    unique_tags = ", ".join(details.get("unique_diff_tags", []) or ["n/a"])
    if diff_chars is not None or similarity is not None:
        diff_value = diff_chars if diff_chars is not None else "unknown"
        stats = f"Total differing characters: {diff_value}"
        if similarity is not None:
            stats += f" • sequence matcher ratio={similarity:.6f}"
        stats += f" • diff blocks={len(blocks)}"
        console.print(stats, style=header_style)
    console.print(f"Diff operation types: {unique_tags}", style=header_style)
    for idx, block in enumerate(blocks, start=1):
        src_range = block.get("source_range")
        seg_range = block.get("segments_range")
        src_len = (src_range[1] - src_range[0]) if src_range else 0
        seg_len = (seg_range[1] - seg_range[0]) if seg_range else 0
        console.print(
            (
                "[{idx}] tag={tag} src_range={src} (len={src_len}) "
                "segments_range={seg} (len={seg_len})"
            ).format(
                idx=idx,
                tag=block.get("tag"),
                src=src_range,
                src_len=src_len,
                seg=seg_range,
                seg_len=seg_len,
            ),
            style="red" if has_drop else "yellow",
        )
        console.print(
            f"     expected snippet: {block.get('source_snippet', '')!r}",
            style="red" if has_drop else "yellow",
        )
        console.print(
            f"     actual snippet:   {block.get('segments_snippet', '')!r}",
            style="red" if has_drop else "yellow",
        )


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
        _, info = _fuzzy_compare_source_and_segments(source_text, stitched_text)
        diff_desc = info.get("blocks", [])
        insert_cleanup = None
        if diff_desc:
            insert_cleanup = _strip_inserted_text(segments, diff_desc)
            if insert_cleanup.get("removed_bytes"):
                console.print(
                    (
                        "Removed {removed} inserted byte(s) from model output before verification."
                    ).format(removed=insert_cleanup["removed_bytes"]),
                    style="yellow",
                )
                try:
                    _assert_segments_cover_source(source_text, segments)
                    info["mode"] = "strict_after_insert_trim"
                    info.setdefault("postprocess", {})["insertions_removed"] = insert_cleanup
                    return info
                except Exception:
                    stitched_text = _stitch_segments_with_validation(segments)
                    _, info = _fuzzy_compare_source_and_segments(source_text, stitched_text)
        if insert_cleanup is not None:
            info.setdefault("postprocess", {})["insertions_removed"] = insert_cleanup
        diff_desc = info.get("blocks", [])
        summary = "Fuzzy verification accepted" if diff_desc else "Fuzzy verification not needed"
        if diff_desc:
            block = diff_desc[0]
            summary = (
                "Fuzzy verification accepted minor diff "
                f"(tag={block['tag']}, source_range={block['source_range']}, diff_chars={info.get('diff_chars')})."
            )
        console.print(summary, style="yellow")
        _require_exact_source_match(source_text, segments, diff_hint=info)
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
    content, input_source, normalization_info = read_input_text(
        args, proxies=proxy_dict, verify=verify_tls
    )
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
    if normalization_info:
        metadata["input_normalization"] = normalization_info
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

    console.print(
        f"Requesting segmentation for {len(content):,} characters.",
        style="bold cyan",
    )
    generated_code, usage_metadata = segment(
        content,
        client=client,
        model=args.model,
        use_code_execution=args.code_exectution,
    )
    metadata["usage_metadata"] = _serialize_usage_metadata(usage_metadata)
    metadata["response_character_count"] = len(generated_code or "")

    segments_result = None
    log_status = "segments_not_extracted"
    error_message: str | None = None
    verification_details: dict[str, Any] | None = None
    diff_report_details: dict[str, Any] | None = None

    if not generated_code or not generated_code.strip():
        error_message = "Model returned no response to parse."
        log_status = "empty_response"
        console.print(error_message, style="yellow")
    else:
        if args.code_exectution:
            segments_result = extract_segments_from_code_string(generated_code)
        else:
            segments_result = extract_segments_from_content_blocks(generated_code)
            if segments_result is not None:
                metadata["response_format"] = {
                    "mode": "content_blocks",
                    "segments": len(segments_result),
                }
        if segments_result is None and error_message is None:
            error_message = "Failed to extract 'segments' data from model response."
            console.print(error_message, style="yellow")

    try:
        if segments_result is not None:
            log_status = "segments_extracted"
            print("Successfully extracted 'segments' array:")
            print(segments_result)
            pretty_print_segments(segments_result)
            newline_healing_info = None
            if not args.code_exectution:
                segments_result, embedded_marker_info = _strip_embedded_markers_from_segments(
                    segments_result
                )
                if embedded_marker_info:
                    metadata["embedded_marker_cleanup"] = embedded_marker_info
                    console.print(
                        (
                            "Removed {total} stray CONTENT-TYPE marker(s) that were "
                            "accidentally inserted inside segments."
                        ).format(total=embedded_marker_info["total_removed"]),
                        style="dim",
                    )
                newline_healing_info = _heal_newline_discrepancies(content, segments_result)
                if newline_healing_info.get("applied_patches"):
                    metadata["newline_healing"] = newline_healing_info
                    console.print(
                        (
                            "Normalized {count} newline-only diff(s) before local verification."
                        ).format(count=newline_healing_info["applied_patches"]),
                        style="dim",
                    )
                literal_fix_info = _maybe_apply_literal_escape_fix(content, segments_result)
                if literal_fix_info.get("applied"):
                    segments_result = literal_fix_info["segments"]
                    metadata["literal_escape_fix"] = {
                        "diff_before": literal_fix_info["diff_before"],
                        "diff_after": literal_fix_info["diff_after"],
                        "similarity_before": literal_fix_info["similarity_before"],
                        "similarity_after": literal_fix_info["similarity_after"],
                    }
                    console.print(
                        "Decoded literal escape sequences emitted by the model before verification.",
                        style="dim",
                    )
                backslash_fix_info = _maybe_restore_backslashes(content, segments_result)
                if backslash_fix_info.get("applied"):
                    segments_result = backslash_fix_info["segments"]
                    metadata["backslash_restoration"] = {
                        "diff_before": backslash_fix_info["diff_before"],
                        "diff_after": backslash_fix_info["diff_after"],
                        "similarity_before": backslash_fix_info["similarity_before"],
                        "similarity_after": backslash_fix_info["similarity_after"],
                        "inserted": backslash_fix_info.get("inserted"),
                        "removed": backslash_fix_info.get("removed"),
                    }
                    console.print(
                        "Reinserted literal backslashes that were stripped from quoted strings.",
                        style="dim",
                    )
                literal_encode_info = _maybe_encode_literal_sequences(content, segments_result)
                if literal_encode_info.get("applied"):
                    segments_result = literal_encode_info["segments"]
                    metadata["literal_sequence_encoding"] = {
                        "diff_before": literal_encode_info["diff_before"],
                        "diff_after": literal_encode_info["diff_after"],
                        "similarity_before": literal_encode_info["similarity_before"],
                        "similarity_after": literal_encode_info["similarity_after"],
                    }
                    console.print(
                        "Converted decoded control characters back into literal escape sequences as in SOURCE_TEXT.",
                        style="dim",
                    )
                small_replace_info = _heal_small_replacements(content, segments_result)
                if small_replace_info.get("applied"):
                    metadata["small_replacement_healing"] = small_replace_info
                    console.print(
                        (
                            "Repaired {count} tiny diff(s) by restoring SOURCE_TEXT bytes."
                        ).format(count=small_replace_info.get("count")),
                        style="dim",
                    )
                segments_result, removed_segments = _remove_empty_segments(segments_result)
                if removed_segments:
                    metadata["empty_segments_removed"] = removed_segments
                    console.print(
                        f"Removed {removed_segments} empty segment(s) introduced during normalization.",
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
            try:
                _require_exact_source_match(content, segments_result)
            except LocalVerificationError as exc:
                error_message = f"Final coverage check failed: {exc}"
                log_status = "local_verification_failed"
                console.print(error_message, style=LOCAL_VERIFICATION_ERROR_STYLE)
                raise SystemExit(1) from exc
        else:
            if error_message is None:
                error_message = "Failed to extract segments from the model response."
                console.print(error_message, style="yellow")
    finally:
        if verification_details is not None:
            metadata["local_verification"] = verification_details
            if verification_details.get("blocks"):
                diff_report_details = verification_details
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

    if diff_report_details:
        _print_diff_report(diff_report_details)


if __name__ == "__main__":
    main()
