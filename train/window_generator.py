"""
Functions for generating training windows from datasets, including pure, mixed,
and line-injected augmentation. Also includes the multi-threaded prefetcher.
"""
import math
import random
import re
import threading
import queue
import time
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING, Any, Set

import numpy as np
import datasets as hfds

import config as cfg

if TYPE_CHECKING:
    from config import DataConfig
from data_utils import bytes_from_text
from token_utils import sanitize_tokens

_PROHIBITED_INJECTION_LANGS = {"csv", "json", "yaml", "text", "html"}
_PROHIBITED_INJECTION_LIDS = {
    lid
    for name, lid in getattr(cfg, "LANG2ID", {}).items()
    if isinstance(name, str) and name.lower() in _PROHIBITED_INJECTION_LANGS and lid is not None
}
if hasattr(cfg, "ID2LANG"):
    _PROHIBITED_INJECTION_LIDS.update(
        {
            int(lid)
            for lid, name in cfg.ID2LANG.items()
            if isinstance(name, str) and name.lower() in _PROHIBITED_INJECTION_LANGS
        }
    )

_MAX_LINE_INJECT_COMMENT_RATIO = 0.4
_LANGUAGE_PAIR_WEIGHTS = (
    (("html", "css"), 10),
    (("html", "javascript_typescript"), 10),
    (("javascript_typescript", "json"), 9),
    (("php", "html"), 9),
    (("php", "sql"), 9),
    (("javascript_typescript", "css"), 8),
    (("python", "json"), 8),
    (("python", "sql"), 8),
    (("sql", "java"), 8),
    (("sql", "csharp"), 8),
    (("sql", "javascript_typescript"), 8),
    (("json", "java"), 8),
    (("json", "csharp"), 8),
    (("json", "go"), 8),
    (("dockerfile", "shell"), 8),
    (("powershell", "json"), 8),
    (("rust", "c_family"), 8),
    (("php", "javascript_typescript"), 7),
    (("python", "csv"), 7),
    (("sql", "go"), 7),
    (("sql", "ruby"), 7),
    (("json", "ruby"), 7),
    (("yaml", "go"), 7),
    (("yaml", "python"), 7),
    (("powershell", "csharp"), 7),
    (("encoding_base64", "json"), 7),
    (("encoding_hex", "c_family"), 7),
    (("visual_basic", "sql"), 7),
    (("python", "shell"), 7),
    (("dockerfile", "yaml"), 7),
    (("yaml", "json"), 6),
    (("yaml", "ruby"), 6),
    (("shell", "c_family"), 6),
    (("visual_basic", "csharp"), 6),
    (("encoding_base64", "html"), 6),
    (("encoding_base85", "c_family"), 6),
    (("encoding_base58", "rust"), 6),
    (("encoding_base32", "yaml"), 4),
)
_DEFAULT_LANGUAGE_PAIR_MODE_PROB = 0.5


_LANGUAGE_PAIR_ADJACENCY: Dict[int, Set[int]] = {}
for (lang_a, lang_b), _ in _LANGUAGE_PAIR_WEIGHTS:
    lid_a = cfg.LANG2ID.get(lang_a)
    lid_b = cfg.LANG2ID.get(lang_b)
    if lid_a is None or lid_b is None:
        continue
    ia, ib = int(lid_a), int(lid_b)
    _LANGUAGE_PAIR_ADJACENCY.setdefault(ia, set()).add(ib)
    _LANGUAGE_PAIR_ADJACENCY.setdefault(ib, set()).add(ia)


_LINE_INJECT_CALL_COUNTER = 0
_LINE_INJECT_COUNTER_LOCK = threading.Lock()


_MARKDOWN_FENCE_TOKEN_MAP: Dict[str, List[str]] = {
    "javascript_typescript": ["javascript", "js", "typescript", "ts"],
    "typescript": ["typescript", "ts"],  # safeguard if future configs split typescript
    "php": ["php"],
    "csharp": ["csharp", "cs"],
    "go": ["go"],
    "sql": ["sql"],
    "rust": ["rust"],
    "yaml": ["yaml", "yml"],
    "ruby": ["ruby", "rb"],
    "python": ["python", "py"],
    "java": ["java"],
    "c_family": ["c", "cpp"],
    "json": ["json"],
    "css": ["css"],
    "html": ["html", "xml"],
    "csv": ["csv"],
    "shell": ["bash", "sh", "shell"],
    "powershell": ["powershell", "ps1"],
    "visual_basic": ["vb", "vbnet"],
    "dockerfile": ["dockerfile", "docker"],
    "markdown": ["markdown", "md"],
}
_MARKDOWN_GENERIC_TOKENS: List[str] = ["text", "plaintext", "plain", "none"]
_MARKDOWN_RANDOM_TOKEN_POOL: List[str] = sorted(
    {
        token
        for tokens in _MARKDOWN_FENCE_TOKEN_MAP.values()
        for token in tokens
    }.union(_MARKDOWN_GENERIC_TOKENS)
)


def _markdown_fence_tokens_for_lid(lid: int) -> List[str]:
    name = _lang_name(lid)
    if not name:
        return []
    lname = name.lower()
    if lname.startswith("encoding") or lname in {"text", "markdown_text"}:
        return []
    tokens = _MARKDOWN_FENCE_TOKEN_MAP.get(lname)
    if tokens:
        return list(tokens)
    cleaned = lname.replace("_", "")
    if cleaned and cleaned.isalpha() and not cleaned.startswith("encoding"):
        return [cleaned]
    return []


def _pick_markdown_fence_label(
    lid: Optional[int],
    *,
    allow_generic: bool = True,
    random_pool: Optional[List[str]] = None,
) -> str:
    if lid is not None:
        tokens = _markdown_fence_tokens_for_lid(int(lid))
        if tokens:
            return random.choice(tokens)
    if allow_generic:
        pool = random_pool if random_pool is not None else _MARKDOWN_RANDOM_TOKEN_POOL
        if pool:
            return random.choice(pool)
    return ""


def _available_language_pairs(lids: List[int]) -> List[Tuple[Tuple[int, int], int]]:
    """Return list of ((lid_a, lid_b), weight) for lids present in the dataset view."""
    if not lids:
        return []
    lid_set = set(lids)
    pairs: List[Tuple[Tuple[int, int], int]] = []
    for (lang_a, lang_b), weight in _LANGUAGE_PAIR_WEIGHTS:
        lid_a = cfg.LANG2ID.get(lang_a)
        lid_b = cfg.LANG2ID.get(lang_b)
        if lid_a is None or lid_b is None:
            continue
        if lid_a in lid_set and lid_b in lid_set:
            pairs.append(((int(lid_a), int(lid_b)), weight))
    return pairs


def _maybe_choose_language_pair(
    lids: List[int],
    data_cfg: Optional["DataConfig"],
) -> Optional[Tuple[int, int]]:
    """
    Optionally choose a language pair based on configured probability and weights.
    The returned pair is unordered, expressed as a sorted tuple of language ids.
    """
    prob = _DEFAULT_LANGUAGE_PAIR_MODE_PROB
    if data_cfg is not None:
        prob = getattr(data_cfg, "language_pair_mode_prob", prob)
    if prob <= 0.0 or random.random() >= prob:
        return None
    weighted_pairs = _available_language_pairs(lids)
    if not weighted_pairs:
        return None
    pairs, weights = zip(*weighted_pairs)
    normalized_pairs = [tuple(sorted(pair)) for pair in pairs]
    choice = random.choices(normalized_pairs, weights=weights, k=1)[0]
    return choice


def _init_language_pair_meta(
    pair_lids: Optional[Tuple[int, int]],
    *,
    mode: str,
    probability: float,
    transitive_ids: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    """Prepare metadata payload describing a selected language pair."""
    if not pair_lids:
        return None
    ordered = tuple(int(x) for x in pair_lids)
    meta = {
        "mode": mode,
        "ids": list(ordered),
        "languages": [_lang_name(lid) for lid in ordered],
        "probability": float(probability),
        "selected": True,
        "active": True,
        "applied": False,
    }
    if transitive_ids:
        dedup = list(dict.fromkeys(int(x) for x in transitive_ids))
        meta["transitive_seed_ids"] = dedup
        meta["transitive_seed_languages"] = [_lang_name(lid) for lid in dedup]
    return meta


def _unlock_language_neighbors(
    lid: int,
    available: Set[int],
    unlocked: Set[int],
) -> List[int]:
    """Add lid and its adjacent popular-pair neighbors to the unlocked set.

    Returns a list of newly unlocked language ids.
    """
    newly_added: List[int] = []
    lid = int(lid)
    if lid in available and lid not in unlocked:
        unlocked.add(lid)
        newly_added.append(lid)
    for neighbor in _LANGUAGE_PAIR_ADJACENCY.get(lid, ()):  # immediate neighbors only
        if neighbor in available and neighbor not in unlocked:
            unlocked.add(neighbor)
            newly_added.append(int(neighbor))
    return newly_added


def _sample_fallback_text_snippet(
    dsets_by_lang: Dict[int, hfds.Dataset],
    text_lid: Optional[int],
    *,
    max_chars: int,
    min_letters: int = 0,
    min_length: int = 24,
    attempts: int = 24,
) -> Tuple[str, Optional[dict]]:
    """
    Draw a text snippet directly from the text dataset when regular sampling fails.
    Attempts to keep a reasonable character and letter count while respecting the byte budget.
    """
    if text_lid is None:
        return "", None
    ds = dsets_by_lang.get(text_lid)
    if ds is None:
        return "", None
    tries = 0
    while tries < attempts:
        ex = _random_example(ds)
        tries += 1
        if not ex:
            continue
        content = ex.get("content", "")
        if not isinstance(content, str):
            continue
        cleaned = _clean_snippet_text(content)
        if not cleaned:
            continue
        if len(cleaned) > max_chars:
            start = random.randint(0, max(0, len(cleaned) - max_chars))
            cleaned = cleaned[start:start + max_chars]
        cleaned = cleaned.strip()
        if len(cleaned) < min_length:
            continue
        if min_letters > 0 and _count_letters(cleaned) < min_letters:
            continue
        return cleaned, ex
    return "", None


def _count_ascii_letters(text: str) -> int:
    """Count ASCII alphabetic characters."""
    return sum(1 for ch in text if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))


def _line_looks_terminated(line: str) -> bool:
    stripped = line.rstrip()
    if not stripped:
        return True
    tail = stripped[-1]
    if tail in {";", "}", "]", ")", ">", ","}:
        return True
    if stripped.endswith(("```", '"""', "'''")):
        return True
    return False

# ---------------------------
# Helpers
# ---------------------------

def _random_example(ds) -> Optional[dict]:
    """Return a random example from a (non-streaming) HF Dataset.
    Falls back to a best-effort sample for streaming/unsized datasets.
    """
    n = None
    try:
        n = len(ds)  # Works for non-streaming Arrow datasets
    except Exception:
        pass

    if n is not None and n > 0:
        idx = random.randrange(0, n)
        return ds[idx]


def _sample_nonempty_example(
    dsets_by_lang: Dict[int, hfds.Dataset],
    lids: List[int],
    *,
    prefer_lid: Optional[int] = None,
    per_lid_attempts: int = 4,
    total_attempts: int = 32,
) -> Tuple[Optional[int], Optional[dict], Optional[np.ndarray]]:
    """Best-effort draw of an example that yields at least one UTF-8 byte."""
    if not lids:
        return None, None, None
    attempts = 0
    while attempts < total_attempts:
        if prefer_lid is not None and attempts == 0:
            lid = prefer_lid
        else:
            lid = random.choice(lids)
        ds = dsets_by_lang.get(lid)
        if ds is None:
            attempts += 1
            continue
        for _ in range(per_lid_attempts):
            ex = _random_example(ds)
            if not ex:
                continue
            content = ex.get("content", "")
            if not content:
                continue
            byte_content = bytes_from_text(content)
            if byte_content.size > 0:
                return lid, ex, byte_content
        attempts += 1
    return None, None, None


def _sample_mixed_segment_count(max_segments: int = 3, continue_prob: float = 0.2) -> int:
    """Geometric-like sampler favouring fewer segments when mixing windows."""
    count = 1
    while count < max_segments and random.random() < continue_prob:
        count += 1
    return count


def _choose_segment_slice(
    byte_content: np.ndarray,
    seg_len: int,
    min_letters: int = 4,
    guard_len: int = 64,
    max_attempts: int = 6,
    return_start: bool = False,
) -> np.ndarray:
    """
    Pick a slice of ``byte_content`` up to ``seg_len`` bytes. For shorter slices we
    bias toward samples that contain at least ``min_letters`` ASCII alphabetic characters.
    """
    total = int(len(byte_content))
    if seg_len <= 0 or total == 0:
        return np.empty((0,), dtype=np.uint8)

    seg_len = min(seg_len, total)
    max_overrun = max(32, seg_len // 4)

    # Prefer aligned line snippets when the fragment has line structure.
    buffer_bytes = byte_content.tobytes()
    try:
        text = buffer_bytes.decode("utf-8", "ignore")
    except Exception:
        text = ""
    if text and "\n" in text:
        lines = text.splitlines(keepends=True)
        if lines:
            line_byte_lengths = [len(line.encode("utf-8", "ignore")) for line in lines]
            line_offsets: List[int] = []
            offset = 0
            for length in line_byte_lengths:
                line_offsets.append(offset)
                offset += length
            total_lines = len(lines)

            interior = list(range(1, total_lines - 1)) if total_lines > 2 else []
            edges = [idx for idx in (0, total_lines - 1) if 0 <= idx < total_lines]
            random.shuffle(interior)
            random.shuffle(edges)
            start_candidates = interior + edges if interior else edges
            if not start_candidates:
                start_candidates = [0]

            max_candidates = min(len(start_candidates), 64)
            desired_min = max(int(seg_len * 0.75), min(seg_len, 96))
            contiguous_candidates: List[Tuple[Tuple[int, int, int, int, int], np.ndarray, int]] = []

            for start_index in start_candidates[:max_candidates]:
                block_start = start_index
                back_steps = 0
                while block_start > 0 and back_steps < 3:
                    prev_line = lines[block_start - 1]
                    if not prev_line.strip():
                        break
                    if line_byte_lengths[block_start - 1] > seg_len:
                        break
                    block_start -= 1
                    back_steps += 1

                end_line = block_start
                total_bytes = 0
                letters_accum = 0
                while end_line < total_lines:
                    next_len = line_byte_lengths[end_line]
                    if total_bytes + next_len > seg_len + max_overrun:
                        break
                    total_bytes += next_len
                    letters_accum += _count_letters(lines[end_line])
                    end_line += 1
                    if total_bytes >= seg_len:
                        break

                if total_bytes <= 0:
                    continue

                # Extend forward to avoid chopping mid-block when budget allows.
                while end_line < total_lines and total_bytes <= seg_len + max_overrun:
                    prev_line = lines[end_line - 1]
                    if _line_looks_terminated(prev_line):
                        break
                    next_len = line_byte_lengths[end_line]
                    if total_bytes + next_len > seg_len + max_overrun:
                        break
                    letters_accum += _count_letters(lines[end_line])
                    total_bytes += next_len
                    end_line += 1

                if end_line < total_lines and not lines[end_line].strip():
                    blank_len = line_byte_lengths[end_line]
                    if total_bytes + blank_len <= seg_len + max_overrun:
                        total_bytes += blank_len
                        end_line += 1

                if block_start >= len(line_offsets):
                    continue
                start_byte = line_offsets[block_start]
                end_byte = start_byte + total_bytes
                if end_byte > total:
                    end_byte = total
                    total_bytes = end_byte - start_byte
                if total_bytes <= 0:
                    continue

                snippet = byte_content[start_byte:end_byte]
                snippet_text = snippet.tobytes().decode("utf-8", "ignore")
                snippet_letters = _count_ascii_letters(snippet_text)
                if snippet_letters < max(min_letters, 1):
                    continue

                overrun = max(0, total_bytes - seg_len)
                if overrun > max_overrun:
                    continue

                priority = 0
                if total_bytes < seg_len:
                    priority = 1 if total_bytes >= desired_min else 2
                edge_penalty = 0 if 0 < block_start < total_lines - 1 else 1
                quality = (
                    priority,
                    abs(seg_len - total_bytes),
                    edge_penalty,
                    -snippet_letters,
                    block_start,
                )
                contiguous_candidates.append((quality, snippet, start_byte))

            if contiguous_candidates:
                contiguous_candidates.sort(key=lambda item: item[0])
                _, best_segment, best_start = contiguous_candidates[0]
                if return_start:
                    return best_segment, best_start  # type: ignore[return-value]
                return best_segment

    chosen_start = 0
    if seg_len > guard_len:
        if total == seg_len:
            segment = byte_content[:seg_len]
            return (segment, chosen_start) if return_start else segment
        start = random.randint(0, total - seg_len)
        segment = byte_content[start:start + seg_len]
        return (segment, start) if return_start else segment

    best_slice = byte_content[:seg_len]
    best_start = 0
    best_letters = -1
    for _ in range(max_attempts):
        if total == seg_len:
            start = 0
        else:
            start = random.randint(0, total - seg_len)
        segment = byte_content[start:start + seg_len]
        text = segment.tobytes().decode("utf-8", "ignore")
        letters = _count_ascii_letters(text)
        if letters >= min_letters:
            return (segment, start) if return_start else segment
        if letters > best_letters:
            best_slice = segment
            best_start = start
            best_letters = letters
    return (best_slice, best_start) if return_start else best_slice


def _lang_name(lid: int) -> str:
    return cfg.ID2LANG.get(int(lid), f"id_{int(lid)}")


def _extract_example_source(example: Optional[dict]) -> str:
    if not example:
        return "unknown"

    def _stringify(val) -> Optional[str]:
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            for item in val:
                s = _stringify(item)
                if s:
                    return s
            return None
        text = str(val)
        return text.strip() or None

    primary_keys = [
        "source_path",
        "file_path",
        "filepath",
        "path",
        "filename",
        "source",
        "source_file",
        "doc_path",
        "doc_id",
        "uid",
        "id",
        "url",
    ]
    for key in primary_keys:
        if key in example:
            candidate = _stringify(example.get(key))
            if candidate:
                return candidate

    for meta_key in ("meta", "metadata", "_metadata"):
        meta_val = example.get(meta_key)
        if isinstance(meta_val, dict):
            for key in primary_keys:
                if key in meta_val:
                    candidate = _stringify(meta_val.get(key))
                    if candidate:
                        return candidate
            for key, value in meta_val.items():
                candidate = _stringify(value)
                if candidate:
                    return f"{key}:{candidate}"

    content_preview = _stringify(example.get("content"))
    if content_preview:
        return f"content:{content_preview[:40]}..."

    return "unknown"


def _compute_final_segments(labels_u8: np.ndarray) -> List[Dict[str, int]]:
    segments = []
    pad_id = cfg.PAD_ID
    L = int(labels_u8.shape[0])
    i = 0
    while i < L:
        lid = int(labels_u8[i])
        j = i
        while j < L and int(labels_u8[j]) == lid:
            j += 1
        if lid != pad_id and i != j:
            segments.append(
                {
                    "language_id": lid,
                    "start": i,
                    "end": j,
                    "length": j - i,
                }
            )
        i = j
    return segments


def _apply_final_byte_contributions(metadata: Optional[Dict], source_idx_bytes: Optional[np.ndarray]):
    if not metadata or source_idx_bytes is None or source_idx_bytes.size == 0:
        return
    samples = metadata.get("samples") or []
    if not samples:
        return

    counts: Dict[int, Dict[str, Any]] = {}
    current_idx: Optional[int] = None
    run_start = 0

    for pos, raw_idx in enumerate(source_idx_bytes.tolist()):
        src_idx = int(raw_idx)
        if src_idx == current_idx:
            continue
        if current_idx is not None and current_idx >= 0:
            span_len = pos - run_start
            if span_len > 0:
                entry = counts.setdefault(current_idx, {"bytes": 0, "spans": []})
                entry["bytes"] += span_len
                entry["spans"].append((run_start, pos))
        run_start = pos
        current_idx = src_idx

    if current_idx is not None and current_idx >= 0:
        span_len = len(source_idx_bytes) - run_start
        if span_len > 0:
            entry = counts.setdefault(current_idx, {"bytes": 0, "spans": []})
            entry["bytes"] += span_len
            entry["spans"].append((run_start, len(source_idx_bytes)))

    for idx, data in counts.items():
        if idx < 0 or idx >= len(samples):
            continue
        sample = samples[idx]
        sample["final_bytes"] = int(data["bytes"])
        sample["final_spans"] = [
            {"start": int(start), "end": int(end)}
            for start, end in data["spans"]
            if end > start
        ]

    for sample in samples:
        sample.setdefault("final_bytes", 0)
        sample.setdefault("final_spans", [])
# ---------------------------
# Window builders (supports partial fill + masking)
# ---------------------------

def _make_pure_window_impl(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    collect_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    meta = {"samples": []} if collect_meta else None
    lids = list(dsets_by_lang.keys())
    x = np.full((target_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    y = np.full((target_len,), cfg.PAD_ID, dtype=np.uint8)
    if not lids:
        return sanitize_tokens(x), y, meta
    lid, ex, byte_content = _sample_nonempty_example(dsets_by_lang, lids)
    if ex is None or byte_content is None or byte_content.size == 0:
        if collect_meta:
            meta["samples"].append(
                {
                    "origin": "base",
                    "language_id": -1,
                    "language": "unknown",
                    "source": "unknown",
                    "status": "failed_to_sample_nonempty",
                }
            )
        return sanitize_tokens(x), y, meta

    total_len = int(byte_content.shape[0])
    start = 0
    slice_bytes = byte_content
    if total_len >= target_len:
        start = random.randint(0, total_len - target_len)
        slice_bytes = byte_content[start:start + target_len]

    L = int(slice_bytes.shape[0])
    if L <= 0:
        if collect_meta:
            meta["samples"].append(
                {
                    "origin": "base",
                    "language_id": int(lid),
                    "language": _lang_name(lid),
                    "source": _extract_example_source(ex),
                    "status": "zero_bytes",
                }
            )
        return sanitize_tokens(x), y, meta

    x[:L] = slice_bytes.astype(np.int32)
    y[:L] = lid
    if collect_meta:
        meta["samples"].append(
            {
                "origin": "base",
                "language_id": int(lid),
                "language": _lang_name(lid),
                "source": _extract_example_source(ex),
                "bytes": int(L),
                "start": 0,
                "end": int(L),
                "source_offset": int(start),
            }
        )
    return sanitize_tokens(x), y, meta


def make_pure_window(dsets_by_lang: Dict[int, hfds.Dataset],
                     target_len: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y, _ = _make_pure_window_impl(dsets_by_lang, target_len, collect_meta=False)
    return sanitize_tokens(x), y


def _make_mixed_window_impl(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    min_seg: int,
    collect_meta: bool = False,
    *,
    data_cfg: Optional["DataConfig"] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    # inputs are int32 to allow PAD_BYTE_ID=256
    x_buf = np.full((target_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    y_buf = np.full((target_len,), cfg.PAD_ID, dtype=np.uint8)
    meta = {"samples": []} if collect_meta else None

    lids = list(dsets_by_lang.keys())
    if not lids:
        return sanitize_tokens(x_buf), y_buf, meta

    pair_prob = _DEFAULT_LANGUAGE_PAIR_MODE_PROB
    if data_cfg is not None:
        pair_prob = getattr(data_cfg, "language_pair_mode_prob", pair_prob)
    pair_lids = _maybe_choose_language_pair(lids, data_cfg)
    available_set: Set[int] = {int(l) for l in lids}
    unlocked_lids: Set[int] = set()
    unlock_trace: List[Dict[str, Any]] = []
    if pair_lids:
        unlocked_lids = {int(lid) for lid in pair_lids if int(lid) in available_set}
        if unlocked_lids:
            unlock_trace.append(
                {
                    "event": "seed",
                    "language_ids": sorted(unlocked_lids),
                    "languages": [_lang_name(lid) for lid in sorted(unlocked_lids)],
                }
            )
    if collect_meta and meta is not None:
        transitive_seeds = sorted(unlocked_lids) if unlocked_lids else None
        pair_meta = _init_language_pair_meta(
            pair_lids,
            mode="mixed",
            probability=pair_prob,
            transitive_ids=transitive_seeds,
        )
        if pair_meta:
            meta["language_pair_mode"] = pair_meta
    available_lids = list(unlocked_lids) if unlocked_lids else (list(pair_lids) if pair_lids else list(lids))
    if not available_lids:
        available_lids = list(lids)
        if collect_meta and meta is not None and meta.get("language_pair_mode"):
            meta["language_pair_mode"]["active"] = False
            meta["language_pair_mode"]["reason"] = "pair_languages_unavailable"

    pair_unique: List[int] = []
    required_pair_lids: Set[int] = set()
    if pair_lids:
        pair_unique = list(dict.fromkeys(pair_lids))
        required_pair_lids = {int(lid) for lid in pair_unique}

    max_langs = int(getattr(data_cfg, "max_mixed_languages", 3)) if data_cfg is not None else 3
    if unlocked_lids:
        max_langs = max(max_langs, len(unlocked_lids))
    if max_langs <= 0:
        max_langs = 1
    used_lids: List[int] = []

    nsegs = _sample_mixed_segment_count()
    if required_pair_lids:
        nsegs = max(nsegs, len(required_pair_lids))

    left, pos = target_len, 0
    for i in range(nsegs):
        seg_len = random.randint(min_seg, left) if left > min_seg else left
        if i == nsegs - 1:
            seg_len = left
        if seg_len <= 0:
            break

        if unlocked_lids:
            available_lids = list(unlocked_lids)
        prioritized = [lid for lid in available_lids if lid in required_pair_lids]
        if prioritized:
            candidate_lids = prioritized
        elif len(used_lids) >= max_langs:
            candidate_lids = [lid for lid in available_lids if lid in used_lids] or used_lids or available_lids
        else:
            candidate_lids = available_lids

        preferred = random.choice(candidate_lids)
        lid, ex, byte_content = _sample_nonempty_example(
            dsets_by_lang, candidate_lids, prefer_lid=preferred
        )
        if ex is None or byte_content is None or byte_content.size == 0:
            if collect_meta:
                meta["samples"].append(
                    {
                        "origin": "base",
                        "language_id": int(preferred),
                        "language": _lang_name(preferred),
                        "source": _extract_example_source(ex),
                        "start": int(pos),
                        "end": int(pos),
                        "bytes": 0,
                        "requested_bytes": int(seg_len),
                        "status": "failed_to_sample_nonempty",
                    }
                )
            pos += seg_len
            left -= seg_len
            continue

        if collect_meta:
            segment, src_start = _choose_segment_slice(
                byte_content, seg_len, return_start=True
            )
        else:
            segment = _choose_segment_slice(byte_content, seg_len)
            src_start = None
        L = int(segment.shape[0])
        if L > left and L > 0:
            segment = segment[:left]
            L = int(segment.shape[0])
        write_len = min(L, left)
        start_pos = pos
        if write_len > 0:
            x_buf[pos:pos + write_len] = segment[:write_len].astype(np.int32)
            y_buf[pos:pos + write_len] = lid
            if lid not in used_lids:
                used_lids.append(lid)
            if unlocked_lids:
                newly_added = _unlock_language_neighbors(lid, available_set, unlocked_lids)
                if newly_added:
                    max_langs = max(max_langs, len(unlocked_lids))
                    unlock_trace.append(
                        {
                            "event": "expand",
                            "trigger": int(lid),
                            "trigger_language": _lang_name(lid),
                            "new_language_ids": sorted(newly_added),
                            "new_languages": [_lang_name(x) for x in sorted(newly_added)],
                            "available_language_ids": sorted(unlocked_lids),
                            "available_languages": [_lang_name(x) for x in sorted(unlocked_lids)],
                        }
                    )
            if lid in required_pair_lids:
                required_pair_lids.discard(lid)
        if collect_meta:
            entry = {
                "origin": "base",
                "language_id": int(lid),
                "language": _lang_name(lid),
                "source": _extract_example_source(ex),
                "start": int(start_pos),
                "end": int(start_pos + write_len),
                "bytes": int(write_len),
                "requested_bytes": int(seg_len),
            }
            if src_start is not None:
                entry["source_offset"] = int(src_start)
            if write_len == 0:
                entry["status"] = "zero_bytes"
            meta["samples"].append(entry)

        used_bytes = write_len if write_len > 0 else seg_len
        pos += used_bytes
        left = max(0, left - used_bytes)

    if collect_meta and meta is not None:
        pair_meta = meta.get("language_pair_mode")
        if pair_meta:
            if unlock_trace:
                pair_meta["unlock_trace"] = unlock_trace
        if pair_meta and unlocked_lids:
            sorted_unlocked = sorted(unlocked_lids)
            pair_meta["unlocked_language_ids"] = sorted_unlocked
            pair_meta["unlocked_languages"] = [_lang_name(lid) for lid in sorted_unlocked]
            pair_meta["unlocked_count"] = len(sorted_unlocked)
    return sanitize_tokens(x_buf), y_buf, meta


def make_mixed_window(dsets_by_lang: Dict[int, hfds.Dataset],
                      target_len: int,
                      min_seg: int) -> Tuple[np.ndarray, np.ndarray]:
    x, y, _ = _make_mixed_window_impl(
        dsets_by_lang, target_len, min_seg, collect_meta=False, data_cfg=None
    )
    return sanitize_tokens(x), y


# ---------------------------
# Shared sampler for train/eval/preview
# ---------------------------

def _line_inject_mode_prob(data_cfg: "DataConfig") -> float:
    return max(0.0, getattr(data_cfg, "line_inject_prob", 0.0))


def _resolve_mix_probability(data_cfg: "DataConfig") -> float:
    mix_prob = getattr(data_cfg, "mix_prob", None)
    pure_prob = max(0.0, getattr(data_cfg, "pure_prob", 0.0))
    line_mode_prob = _line_inject_mode_prob(data_cfg)
    markdown_prob = max(0.0, getattr(data_cfg, "markdown_prob", 0.0))
    if mix_prob is None:
        mix_prob = 1.0 - pure_prob - line_mode_prob - markdown_prob
    return max(0.0, mix_prob)


def _choose_window_mode(data_cfg: "DataConfig") -> str:
    pure_prob = max(0.0, getattr(data_cfg, "pure_prob", 0.0))
    line_mode_prob = _line_inject_mode_prob(data_cfg)
    markdown_prob = max(0.0, getattr(data_cfg, "markdown_prob", 0.0))
    mix_prob = _resolve_mix_probability(data_cfg)
    total = pure_prob + line_mode_prob + markdown_prob + mix_prob
    if total <= 0.0:
        return "mixed"

    r = random.random() * total
    if r < pure_prob:
        return "pure"
    if r < pure_prob + line_mode_prob:
        return "line_inject"
    if r < pure_prob + line_mode_prob + markdown_prob:
        return "markdown"
    return "mixed"


def _make_training_window_internal(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    data_cfg: "DataConfig",
    collect_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Draw a window using the same probability distribution as the training
    prefetchers, including mixed windows, line injection, optional overlays,
    and tail padding variation.
    """
    metadata = None
    if collect_meta:
        metadata = {
            "mode": "",
            "requested_mode": "",
            "samples": [],
            "line_injections": [],
            "overlays": [],
            "host": None,
            "final_segments": [],
            "length": int(target_len),
        }

    mode = _choose_window_mode(data_cfg)
    base_mode = mode
    resolved_mode = mode
    if mode == "pure":
        x, y, partial = _make_pure_window_impl(dsets_by_lang, target_len, collect_meta)
        if collect_meta and metadata is not None:
            metadata["mode"] = "pure"
            metadata["requested_mode"] = "pure"
            if partial:
                metadata["samples"].extend(partial.get("samples", []))
    elif mode == "line_inject":
        x, y, partial = _make_line_injected_window_impl(dsets_by_lang, target_len, data_cfg, collect_meta)
        if collect_meta and metadata is not None:
            metadata["mode"] = "line_inject"
            metadata["requested_mode"] = "line_inject"
            if partial:
                metadata["samples"].extend(partial.get("samples", []))
                metadata["line_injections"].extend(partial.get("line_injections", []))
                metadata["host"] = partial.get("host", metadata.get("host"))
                if metadata.get("language_pair_mode") is None and partial.get("language_pair_mode"):
                    metadata["language_pair_mode"] = partial["language_pair_mode"]
    elif mode == "markdown":
        x, y, partial = _make_markdown_window_impl(dsets_by_lang, target_len, data_cfg, collect_meta)
        if collect_meta and metadata is not None:
            metadata["mode"] = "markdown"
            metadata["requested_mode"] = "markdown"
            if partial:
                metadata["samples"].extend(partial.get("samples", []))
                markdown_meta = partial.get("markdown_blocks")
                if markdown_meta:
                    metadata["markdown_blocks"] = markdown_meta
                metadata["line_injections"].extend(partial.get("line_injections", []))
                if metadata.get("host") is None:
                    metadata["host"] = partial.get("host")
                if metadata.get("language_pair_mode") is None and partial.get("language_pair_mode"):
                    metadata["language_pair_mode"] = partial["language_pair_mode"]
    else:
        x, y, partial = _make_mixed_window_impl(
            dsets_by_lang,
            target_len,
            data_cfg.min_seg_len,
            collect_meta,
            data_cfg=data_cfg,
        )
        if collect_meta and metadata is not None:
            metadata["mode"] = "mixed"
            metadata["requested_mode"] = "mixed"
            if partial:
                metadata["samples"].extend(partial.get("samples", []))
                if metadata.get("language_pair_mode") is None and partial.get("language_pair_mode"):
                    metadata["language_pair_mode"] = partial["language_pair_mode"]

    # Optionally overlay mixed slices on top of base window
    both_prob = getattr(data_cfg, "both_prob", 0.2)
    allow_overlay = (base_mode == "pure")
    if allow_overlay and both_prob > 0.0 and random.random() < both_prob:
        xm, ym, overlay_meta = _make_mixed_window_impl(
            dsets_by_lang,
            target_len,
            data_cfg.min_seg_len,
            collect_meta,
            data_cfg=data_cfg,
        )
        nslices = random.randint(1, 3)
        for _ in range(nslices):
            max_len = max(data_cfg.min_seg_len, target_len // 4)
            seg_len = random.randint(data_cfg.min_seg_len, max_len)
            if seg_len >= target_len:
                seg_len = target_len - 1
            start = random.randint(0, target_len - seg_len)
            end = start + seg_len
            x[start:end] = xm[start:end]
            y[start:end] = ym[start:end]
            if collect_meta and metadata is not None:
                overlay_entry = {
                    "start": int(start),
                    "end": int(end),
                    "mode": "mixed",
                    "segments": [],
                }
                if overlay_meta:
                    for seg in overlay_meta.get("samples", []):
                        seg_start = int(seg.get("start", 0))
                        seg_end = int(seg.get("end", 0))
                        overlap_start = max(start, seg_start)
                        overlap_end = min(end, seg_end)
                        if overlap_start >= overlap_end:
                            continue
                        new_seg = dict(seg)
                        new_seg["origin"] = "overlay"
                        new_seg["start"] = int(overlap_start)
                        new_seg["end"] = int(overlap_end)
                        new_seg["bytes"] = int(overlap_end - overlap_start)
                        overlay_entry["segments"].append(new_seg)
                        metadata["samples"].append(new_seg)
                metadata["overlays"].append(overlay_entry)
                resolved_mode = "mixed" if resolved_mode != "line_inject" else resolved_mode

    # Tail padding variation mirrors training augmentation
    pad_tail_prob = getattr(data_cfg, "pad_tail_prob", 0.1)
    if pad_tail_prob > 0.0 and random.random() < pad_tail_prob:
        max_frac = getattr(data_cfg, "pad_tail_max_frac", 0.9)
        max_pad = max(1, int(target_len * max_frac))
        pad_len = random.randint(0, max_pad)
        if pad_len > 0:
            content_end = max(1, target_len - pad_len)
            # Keep the longest contiguous non-pad block by sliding the window if possible
            last_content = np.where(y != cfg.PAD_ID)[0]
            if last_content.size and last_content[-1] + 1 < content_end:
                shift = min(content_end - (int(last_content[-1]) + 1), pad_len)
                x = np.roll(x, -shift)
                y = np.roll(y, -shift)
                x[-shift:] = cfg.PAD_BYTE_ID
                y[-shift:] = cfg.PAD_ID
            else:
                x[content_end:] = cfg.PAD_BYTE_ID
                y[content_end:] = cfg.PAD_ID

    if not np.any(y != cfg.PAD_ID):
        fallback_x, fallback_y, fallback_meta = _make_pure_window_impl(
            dsets_by_lang, target_len, collect_meta
        )
        if not np.any(fallback_y != cfg.PAD_ID):
            raise RuntimeError("Failed to sample non-empty training window")
        x, y = fallback_x, fallback_y
        resolved_mode = "pure"
        if collect_meta and metadata is not None:
            prev_requested = metadata.get("requested_mode") or metadata.get("mode") or mode
            metadata["mode"] = "pure"
            metadata["requested_mode"] = prev_requested
            metadata["line_injections"] = []
            metadata["overlays"] = []
            metadata["host"] = None
            metadata["samples"] = fallback_meta.get("samples", []) if fallback_meta else []
            metadata["fallback"] = "pure_due_to_empty"
            metadata.pop("language_pair_mode", None)

    if collect_meta and metadata is not None:
        final_segments = []
        for seg in _compute_final_segments(y):
            final_segments.append(
                {
                    **seg,
                    "language": _lang_name(seg["language_id"]),
                }
            )
        metadata["final_segments"] = final_segments
        pair_meta = metadata.get("language_pair_mode")
        if pair_meta and pair_meta.get("selected"):
            requested_ids = tuple(int(i) for i in pair_meta.get("ids", []))
            observed_ids = {int(seg["language_id"]) for seg in final_segments}
            used_ids = [rid for rid in requested_ids if rid in observed_ids]
            pair_meta["used_language_ids"] = used_ids
            pair_meta["used_languages"] = [_lang_name(rid) for rid in used_ids]
            missing_ids = [rid for rid in requested_ids if rid not in observed_ids]
            if missing_ids:
                pair_meta["missing_language_ids"] = missing_ids
                pair_meta["missing_languages"] = [_lang_name(rid) for rid in missing_ids]
            else:
                pair_meta.pop("missing_language_ids", None)
                pair_meta.pop("missing_languages", None)
            active_flag = bool(pair_meta.get("active", True))
            if not active_flag:
                pair_meta["applied"] = False
            else:
                pair_meta["applied"] = not missing_ids and bool(requested_ids)
            pair_meta["present_language_ids"] = sorted(observed_ids)
            pair_meta["present_languages"] = [_lang_name(rid) for rid in sorted(observed_ids)]
        unique_langs = {seg["language_id"] for seg in final_segments}
        if metadata["line_injections"]:
            actual_mode = "line_inject"
        elif len(unique_langs) <= 1:
            actual_mode = resolved_mode if resolved_mode == "line_inject" else "pure"
        else:
            actual_mode = "mixed"
        metadata["actual_mode"] = actual_mode
        metadata["mode"] = actual_mode
    return sanitize_tokens(x), y, metadata


def make_training_window(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    data_cfg: "DataConfig",
) -> Tuple[np.ndarray, np.ndarray]:
    x, y, _ = _make_training_window_internal(dsets_by_lang, target_len, data_cfg, collect_meta=False)
    return sanitize_tokens(x), y


def make_training_window_with_metadata(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    data_cfg: "DataConfig",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    x, y, meta = _make_training_window_internal(dsets_by_lang, target_len, data_cfg, collect_meta=True)
    return sanitize_tokens(x), y, meta or {}


# ---------------------------
# LINE-LEVEL INJECTION
# ---------------------------

def _split_keepends_lines(text: str) -> List[str]:
    return text.splitlines(keepends=True) if text else []


def _classify_comment_lines(lines: List[str]) -> List[bool]:
    """Heuristic detection of comment-only lines."""
    flags: List[bool] = []
    inside_c_block = False
    inside_doc_block: Optional[str] = None
    inside_html_comment = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        is_comment = False

        if inside_doc_block:
            is_comment = True
            if inside_doc_block in stripped:
                occurrences = stripped.count(inside_doc_block)
                if occurrences % 2 == 1:
                    inside_doc_block = None
        elif inside_html_comment:
            is_comment = True
            if "-->" in line:
                inside_html_comment = False
        elif inside_c_block:
            is_comment = True
            if "*/" in line:
                inside_c_block = False
        else:
            if stripped.startswith("//") or stripped.startswith("#") or stripped.startswith("--") or stripped.startswith("%"):
                is_comment = True
            elif lower.startswith("rem "):
                is_comment = True
            elif stripped.startswith("<!--"):
                is_comment = True
                if "-->" not in stripped:
                    inside_html_comment = True
            elif stripped.startswith("*/"):
                is_comment = True
            elif stripped.startswith("/*"):
                is_comment = True
                if "*/" not in stripped or stripped.find("*/") < stripped.find("/*"):
                    inside_c_block = True
            else:
                comment_pos = stripped.find("/*")
                if comment_pos != -1:
                    if "*/" not in stripped[comment_pos + 2:]:
                        inside_c_block = True
                    if stripped[:comment_pos].strip() == "":
                        is_comment = True
            if stripped.startswith('"""') or stripped.startswith("'''"):
                is_comment = True
                delim = stripped[:3]
                quote_count = stripped.count(delim)
                if quote_count % 2 == 1:
                    inside_doc_block = delim

        flags.append(is_comment)
    return flags


def _count_letters(text: str) -> int:
    return _count_ascii_letters(text)


def _clean_snippet_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _trim_text_to_budget(text: str, byte_budget: int) -> Tuple[str, int]:
    if byte_budget <= 0 or not text:
        return "", 0
    encoded = text.encode("utf-8", "ignore")
    if not encoded:
        return "", 0
    if len(encoded) <= byte_budget:
        return text, len(encoded)
    trimmed = encoded[:byte_budget].decode("utf-8", "ignore")
    trimmed_encoded = trimmed.encode("utf-8", "ignore")
    return trimmed, len(trimmed_encoded)


def _sample_lang_snippet(
    dsets_by_lang: Dict[int, hfds.Dataset],
    lid: int,
    *,
    max_chars: int = 320,
    min_letters: int = 0,
    strip: bool = False,
) -> Tuple[str, Optional[dict]]:
    ds = dsets_by_lang.get(lid)
    if ds is None:
        return "", None
    attempts = 0
    while attempts < 8:
        ex = _random_example(ds)
        attempts += 1
        if not ex:
            continue
        content = ex.get("content", "")
        if not isinstance(content, str):
            continue
        snippet = _clean_snippet_text(content)
        if strip:
            snippet = snippet.strip()
        if not snippet:
            continue
        if len(snippet) > max_chars:
            if len(snippet) > max_chars:
                start = random.randint(0, max(0, len(snippet) - max_chars))
                snippet = snippet[start:start + max_chars]
        if min_letters > 0 and _count_letters(snippet) < min_letters:
            continue
        return snippet, ex
    return "", None


def _sample_truncated_exp_lines(lam: float, max_lines: int) -> int:
    k = int(math.ceil(random.expovariate(lam))) if lam > 0 else 1
    return max(1, min(k, max_lines))


def _choose_injection_boundaries(
    host_text: str,
    skip_top_min: int,
    skip_top_max: int,
    max_inj: int,
    *,
    required_count: Optional[int] = None,
) -> List[int]:
    lines = _split_keepends_lines(host_text)
    if not lines:
        return []
    boundaries = np.cumsum([len(ln) for ln in lines]).tolist()
    if not boundaries:
        return []

    skip_min = max(0, skip_top_min)
    skip_max = max(skip_min, skip_top_max)

    if skip_max > 0:
        skip = random.randint(skip_min, skip_max)
        valid = [b for idx, b in enumerate(boundaries) if idx >= skip]
    else:
        valid = boundaries[:]

    if not valid:
        return []

    if required_count is not None and required_count > 0:
        n = min(required_count, len(valid), max_inj)
    else:
        n = 1
        # More injections per file
        while n < max_inj and random.random() < 0.25:
            n += 1
        n = min(n, len(valid))
    picks = random.sample(valid, k=n)
    return sorted(picks, reverse=True)


def _leading_indent_of_line(chars: List[str], at_char_idx: int) -> str:
    i = at_char_idx - 1
    while i >= 0 and chars[i] != "\n":
        i -= 1
    line_start = i + 1
    j = line_start
    indent = []
    while j < len(chars) and chars[j] in (" ", "\t"):
        indent.append(chars[j]); j += 1
    return "".join(indent)


def _prepare_donor_block(donor_text: str, data_cfg: "DataConfig", insertion_indent: str) -> str:
    lines = _split_keepends_lines(donor_text)
    if not lines:
        return ""

    has_interior = len(lines) > 2
    if has_interior:
        interior_start = 1
        interior_end = len(lines) - 2
        if interior_start > interior_end:
            has_interior = False
    if not has_interior:
        interior_start = 0
        interior_end = len(lines) - 1

    if has_interior:
        skip_min = max(interior_start, min(data_cfg.donor_skip_top_min, interior_end))
        skip_max = min(interior_end, max(data_cfg.donor_skip_top_min, data_cfg.donor_skip_top_max))
        if skip_min > skip_max:
            skip_min = skip_max = interior_end
        start_idx = random.randint(skip_min, skip_max)
        end_limit = len(lines) - 1  # keep the final line untouched when interior exists
    else:
        start_idx = 0
        end_limit = len(lines)

    L = _sample_truncated_exp_lines(data_cfg.line_inject_exp_rate, data_cfg.line_inject_max_lines)
    end_idx = min(end_limit, start_idx + L)
    if end_idx <= start_idx:
        end_idx = min(len(lines), start_idx + 1)
    pick = lines[start_idx:end_idx]
    if not pick:
        if has_interior:
            return ""
        pick = lines[:]  # fall back to whole donor when no interior exists
        start_idx = 0
        end_idx = len(lines)

    comment_flags = _classify_comment_lines(lines)
    letter_lengths = [_count_letters(ln) for ln in lines]
    letter_count = sum(letter_lengths[start_idx:end_idx])
    comment_count = sum(1 for flag in comment_flags[start_idx:end_idx] if flag)

    if len(pick) == 1 and letter_count < data_cfg.line_inject_min_single_len:
        if end_idx < end_limit:
            pick.append(lines[end_idx])
            letter_count += letter_lengths[end_idx]
            comment_count += int(comment_flags[end_idx])
            end_idx += 1
        elif start_idx > interior_start:
            start_idx -= 1
            pick.insert(0, lines[start_idx])
            letter_count += letter_lengths[start_idx]
            comment_count += int(comment_flags[start_idx])

    if has_interior:
        max_total_lines = min(len(lines) - 2, data_cfg.line_inject_max_lines)
    else:
        max_total_lines = min(len(lines), data_cfg.line_inject_max_lines)
    min_letters = getattr(data_cfg, "line_inject_min_letters", 6)
    threshold = _MAX_LINE_INJECT_COMMENT_RATIO

    while letter_count < min_letters and len(pick) < max_total_lines:
        expanded = False
        if end_idx < end_limit and len(pick) < max_total_lines:
            pick.append(lines[end_idx])
            letter_count += letter_lengths[end_idx]
            comment_count += int(comment_flags[end_idx])
            end_idx += 1
            expanded = True
        if letter_count < min_letters and len(pick) < max_total_lines and start_idx > interior_start:
            start_idx -= 1
            pick.insert(0, lines[start_idx])
            letter_count += letter_lengths[start_idx]
            comment_count += int(comment_flags[start_idx])
            expanded = True
        if not expanded:
            break

    comment_ratio = (comment_count / len(pick)) if pick else 1.0
    while len(pick) < max_total_lines and comment_ratio > threshold:
        expanded = False
        if end_idx < end_limit and len(pick) < max_total_lines:
            pick.append(lines[end_idx])
            letter_count += letter_lengths[end_idx]
            comment_count += int(comment_flags[end_idx])
            end_idx += 1
            expanded = True
        if comment_ratio > threshold and len(pick) < max_total_lines and start_idx > interior_start:
            start_idx -= 1
            pick.insert(0, lines[start_idx])
            letter_count += letter_lengths[start_idx]
            comment_count += int(comment_flags[start_idx])
            expanded = True
        if not expanded:
            break
        comment_ratio = comment_count / len(pick)

    if not pick or comment_ratio > threshold:
        return ""

    processed = []
    for ln in pick:
        newline = '\n' if ln.endswith('\n') else ''
        core = ln[:-1] if newline else ln
        if random.random() < data_cfg.reindent_prob:
            core = insertion_indent + core.lstrip()
        processed.append(core + newline)

    block = "".join(processed)
    if _count_letters(block) < min_letters:
        return ""
    return block


def _insert_block_at(
    chars: List[str],
    labels: List[int],
    sources: Optional[List[int]],
    idx: int,
    block: str,
    lid: int,
    source_idx: Optional[int],
):
    if not block:
        return
    ins_chars = list(block)
    ins_labs = [lid] * len(ins_chars)
    chars[idx:idx] = ins_chars
    labels[idx:idx] = ins_labs
    if sources is not None:
        assign_idx = -1 if source_idx is None else int(source_idx)
        sources[idx:idx] = [assign_idx] * len(ins_chars)


def _ensure_merge_without_newline(
    chars: List[str],
    labels: List[int],
    sources: Optional[List[int]],
    idx: int,
) -> int:
    if idx > 0 and chars[idx - 1] == "\n":
        del chars[idx - 1]
        del labels[idx - 1]
        if sources is not None:
            del sources[idx - 1]
        return idx - 1
    return idx


def _to_bytes_with_byte_labels(
    chars: List[str],
    char_labels: List[int],
    char_sources: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    xb, yb, sb = [], [], []
    for idx, (ch, lid) in enumerate(zip(chars, char_labels)):
        bs = ch.encode("utf-8", "ignore")
        xb.extend(bs)
        yb.extend([lid] * len(bs))
        if char_sources is not None:
            src_idx = char_sources[idx] if idx < len(char_sources) else -1
            sb.extend([src_idx] * len(bs))
    xb_arr = np.array(xb, dtype=np.uint8)
    yb_arr = np.array(yb, dtype=np.uint8)
    if char_sources is not None:
        sb_arr = np.array(sb, dtype=np.int32) if sb else np.empty((0,), dtype=np.int32)
        return xb_arr, yb_arr, sb_arr
    return xb_arr, yb_arr, None


def _make_line_injected_window_impl(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    data_cfg: "DataConfig",
    collect_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    meta = {"samples": [], "line_injections": [], "host": None} if collect_meta else None
    lids = list(dsets_by_lang.keys())
    text_lid = None
    if hasattr(cfg, "LANG2ID"):
        text_lid = cfg.LANG2ID.get("text")
    if text_lid is None and hasattr(cfg, "ID2LANG"):
        text_lid = next((lid for lid, name in cfg.ID2LANG.items() if name == "text"), None)

    x = np.full((target_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    y = np.full((target_len,), cfg.PAD_ID, dtype=np.uint8)
    if not lids:
        return sanitize_tokens(x), y, meta

    global _LINE_INJECT_CALL_COUNTER
    with _LINE_INJECT_COUNTER_LOCK:
        current_counter = _LINE_INJECT_CALL_COUNTER
        _LINE_INJECT_CALL_COUNTER = (_LINE_INJECT_CALL_COUNTER + 1) % 3
    enforce_three_injections = current_counter == 2

    pair_prob = getattr(data_cfg, "language_pair_mode_prob", _DEFAULT_LANGUAGE_PAIR_MODE_PROB)
    pair_lids = _maybe_choose_language_pair(lids, data_cfg)
    available_set: Set[int] = {int(l) for l in lids}
    transitive_unlocked: Set[int] = set()
    if pair_lids:
        transitive_unlocked = {int(lid) for lid in pair_lids if int(lid) in available_set}
    unlock_trace: List[Dict[str, Any]] = []
    if transitive_unlocked:
        unlock_trace.append(
            {
                "event": "seed",
                "language_ids": sorted(transitive_unlocked),
                "languages": [_lang_name(lid) for lid in sorted(transitive_unlocked)],
            }
        )
    pair_meta = _init_language_pair_meta(
        pair_lids,
        mode="line_inject",
        probability=pair_prob,
        transitive_ids=sorted(transitive_unlocked) if transitive_unlocked else None,
    )
    if collect_meta and meta is not None and pair_meta:
        meta["language_pair_mode"] = pair_meta
    host_candidates = list(transitive_unlocked) if transitive_unlocked else (list(pair_lids) if pair_lids else list(lids))
    if not host_candidates:
        host_candidates = list(lids)

    preferred_host = random.choice(host_candidates)
    host_lid, host_ex, _ = _sample_nonempty_example(
        dsets_by_lang, host_candidates, prefer_lid=preferred_host
    )
    if pair_lids and host_lid not in pair_lids:
        if pair_meta is not None:
            pair_meta["active"] = False
            pair_meta["reason"] = "host_not_in_pair"
        pair_lids = None
        transitive_unlocked.clear()
    if host_ex is None:
        if collect_meta and meta is not None:
            meta["samples"].append(
                {
                    "origin": "host",
                    "language_id": int(preferred_host),
                    "language": _lang_name(preferred_host),
                    "source": _extract_example_source(host_ex),
                    "status": "failed_to_sample_nonempty",
                }
            )
        return _make_pure_window_impl(dsets_by_lang, target_len, collect_meta)

    host_text = host_ex.get("content", "") or ""
    if not host_text:
        return _make_pure_window_impl(dsets_by_lang, target_len, collect_meta)

    if pair_meta is not None and host_lid is not None:
        pair_meta["host_language_id"] = int(host_lid)
        pair_meta["host_language"] = _lang_name(host_lid)

    if transitive_unlocked:
        host_added = _unlock_language_neighbors(host_lid, available_set, transitive_unlocked)
        if host_added:
            unlock_trace.append(
                {
                    "event": "host_unlocked",
                    "trigger": int(host_lid),
                    "trigger_language": _lang_name(host_lid),
                    "new_language_ids": sorted(host_added),
                    "new_languages": [_lang_name(x) for x in sorted(host_added)],
                    "available_language_ids": sorted(transitive_unlocked),
                    "available_languages": [_lang_name(x) for x in sorted(transitive_unlocked)],
                }
            )

    donor_lids: List[int]
    transitive_active = bool(transitive_unlocked)

    def _current_transitive_donors() -> List[int]:
        candidates = {lid for lid in transitive_unlocked if lid != host_lid}
        if text_lid is not None:
            candidates.discard(int(text_lid))
        candidates.difference_update(_PROHIBITED_INJECTION_LIDS)
        return sorted(candidates)

    if transitive_active:
        donor_lids = _current_transitive_donors()
        if pair_meta is not None:
            pair_meta["donor_candidates"] = [int(lid) for lid in donor_lids]
            pair_meta["donor_candidate_languages"] = [_lang_name(lid) for lid in donor_lids]
        if not donor_lids:
            if pair_meta is not None:
                pair_meta["active"] = False
                pair_meta["reason"] = "no_pair_donor_available"
            transitive_unlocked.clear()
            transitive_active = False

    if not transitive_active:
        if text_lid is not None:
            donor_lids = [lid for lid in lids if lid != text_lid]
        else:
            donor_lids = list(lids)
        donor_lids = [lid for lid in donor_lids if lid not in _PROHIBITED_INJECTION_LIDS]
    else:
        donor_lids = list(donor_lids)
    chars, labs = list(host_text), [host_lid] * len(host_text)
    char_sources: Optional[List[int]] = None
    host_idx: Optional[int] = None
    host_source = _extract_example_source(host_ex)
    if collect_meta and meta is not None:
        host_entry = {
            "origin": "host",
            "language_id": int(host_lid),
            "language": _lang_name(host_lid),
            "source": host_source,
            "chars": len(chars),
            "requested_chars": len(chars),
        }
        meta["host"] = {
            "language_id": int(host_lid),
            "language": _lang_name(host_lid),
            "source": host_source,
            "characters": len(chars),
            "sample_index": len(meta["samples"]),
        }
        meta["samples"].append(host_entry)
        host_idx = len(meta["samples"]) - 1
        char_sources = [host_idx] * len(chars)

    boundaries = _choose_injection_boundaries(
        host_text,
        data_cfg.host_skip_top_min,
        data_cfg.host_skip_top_max,
        data_cfg.line_inject_max_injections,
        required_count=3 if enforce_three_injections else None,
    )

    if collect_meta and meta is not None:
        meta.setdefault("policies", {})["line_inject_force_three_every_third"] = {
            "requested": enforce_three_injections,
            "actual_segments": len(boundaries),
        }
        if enforce_three_injections and len(boundaries) < 3:
            meta["policies"]["line_inject_force_three_every_third"]["note"] = "insufficient_host_boundaries"

    for bidx in boundaries:
        if transitive_active:
            donor_lids = _current_transitive_donors()
            if pair_meta is not None:
                pair_meta["donor_candidates"] = [int(lid) for lid in donor_lids]
                pair_meta["donor_candidate_languages"] = [_lang_name(lid) for lid in donor_lids]
        candidates = donor_lids
        if not candidates:
            continue
        if not data_cfg.allow_same_lang_injection:
            alts = [lid for lid in candidates if lid != host_lid]
            if alts:
                candidates = alts
            elif host_lid in candidates:
                continue
        donor_choice = random.choice(candidates)
        donor_lid, donor_ex, _ = _sample_nonempty_example(
            dsets_by_lang, candidates, prefer_lid=donor_choice
        )
        if donor_ex is None:
            continue
        donor_text = donor_ex.get("content", "") or ""

        insertion_indent = _leading_indent_of_line(chars, bidx)
        donor_block = _prepare_donor_block(donor_text, data_cfg, insertion_indent)
        if not donor_block:
            continue

        if transitive_active:
            newly = _unlock_language_neighbors(donor_lid, available_set, transitive_unlocked)
            if newly:
                if pair_meta is not None:
                    expanded = sorted(transitive_unlocked)
                    pair_meta["expanded_language_ids"] = expanded
                    pair_meta["expanded_languages"] = [_lang_name(lid) for lid in expanded]
                unlock_trace.append(
                    {
                        "event": "donor_unlocked",
                        "trigger": int(donor_lid),
                        "trigger_language": _lang_name(donor_lid),
                        "new_language_ids": sorted(newly),
                        "new_languages": [_lang_name(x) for x in sorted(newly)],
                        "available_language_ids": sorted(transitive_unlocked),
                        "available_languages": [_lang_name(x) for x in sorted(transitive_unlocked)],
                    }
                )

        newline_max = max(
            0,
            getattr(
                data_cfg,
                "inject_extra_newlines_max",
                getattr(data_cfg, "line_inject_extra_newlines_max", 4),
            ),
        )
        pre_pad_count = random.randint(0, newline_max) if newline_max > 0 else 0
        post_pad_count = random.randint(0, newline_max) if newline_max > 0 else 0
        start_with_newline_prob = getattr(data_cfg, "start_with_newline_prob", 0.5)
        if random.random() > start_with_newline_prob:
            bidx = _ensure_merge_without_newline(chars, labs, char_sources, bidx)
            pre_pad_count = 0

        if pre_pad_count > 0:
            pre_block = "\n" * pre_pad_count
            _insert_block_at(chars, labs, char_sources, bidx, pre_block, host_lid, host_idx)
            bidx += len(pre_block)
        donor_insert_idx = bidx

        sample_idx = None
        donor_source = _extract_example_source(donor_ex)

        if collect_meta and meta is not None:
            sample_entry = {
                "origin": "injection",
                "language_id": int(donor_lid),
                "language": _lang_name(donor_lid),
                "source": donor_source,
                "chars": len(donor_block),
                "requested_chars": len(donor_block),
            }
            meta["samples"].append(sample_entry)
            sample_idx = len(meta["samples"]) - 1
        _insert_block_at(chars, labs, char_sources, donor_insert_idx, donor_block, donor_lid, sample_idx)
        bidx = donor_insert_idx + len(donor_block)

        if post_pad_count > 0:
            post_block = "\n" * post_pad_count
            _insert_block_at(chars, labs, char_sources, bidx, post_block, host_lid, host_idx)
            bidx += len(post_block)

        if collect_meta and meta is not None:
            meta["line_injections"].append(
                {
                    "language_id": int(donor_lid),
                    "language": _lang_name(donor_lid),
                    "source": donor_source,
                    "insert_char_index": int(donor_insert_idx),
                    "chars_inserted": len(donor_block),
                    "newlines_before": int(pre_pad_count),
                    "newlines_after": int(post_pad_count),
                    "preview": donor_block[:120],
                    "sample_index": sample_idx,
                }
            )

    xb_u8, yb_u8, source_idx_bytes = _to_bytes_with_byte_labels(chars, labs, char_sources)

    # Windowing/padding
    if len(xb_u8) >= target_len:
        start = random.randint(0, len(xb_u8) - target_len)
        x = xb_u8[start:start + target_len].astype(np.int32)
        y = yb_u8[start:start + target_len]
        if source_idx_bytes is not None and source_idx_bytes.size:
            source_idx_bytes = source_idx_bytes[start:start + target_len]
    else:
        x[:len(xb_u8)] = xb_u8.astype(np.int32)
        y[:len(yb_u8)] = yb_u8
        if source_idx_bytes is not None and source_idx_bytes.size:
            pad_len = target_len - len(xb_u8)
            if pad_len > 0:
                source_idx_bytes = np.concatenate(
                    [
                        source_idx_bytes,
                        np.full((pad_len,), -1, dtype=np.int32),
                    ]
                )
    if collect_meta and meta is not None:
        if source_idx_bytes is not None and source_idx_bytes.size:
            _apply_final_byte_contributions(meta, source_idx_bytes)
            samples_list = meta.get("samples", [])
            for inj in meta.get("line_injections", []):
                idx = inj.get("sample_index")
                if isinstance(idx, int) and 0 <= idx < len(samples_list):
                    inj["final_bytes"] = samples_list[idx].get("final_bytes", 0)
                    inj["final_spans"] = samples_list[idx].get("final_spans", [])
            host_info = meta.get("host")
            if isinstance(host_info, dict):
                idx = host_info.get("sample_index")
                if isinstance(idx, int) and 0 <= idx < len(samples_list):
                    host_info["final_bytes"] = samples_list[idx].get("final_bytes", 0)
                    host_info["final_spans"] = samples_list[idx].get("final_spans", [])
        for inj in meta.get("line_injections", []):
            inj.pop("sample_index", None)
        host_info = meta.get("host")
        if isinstance(host_info, dict):
            host_info.pop("sample_index", None)
        pair_meta_out = meta.get("language_pair_mode")
        if pair_meta_out:
            if unlock_trace:
                pair_meta_out["unlock_trace"] = unlock_trace
        if pair_meta_out and transitive_unlocked:
            unlocked_sorted = sorted(transitive_unlocked)
            pair_meta_out["unlocked_language_ids"] = unlocked_sorted
            pair_meta_out["unlocked_languages"] = [_lang_name(lid) for lid in unlocked_sorted]
            pair_meta_out["unlocked_count"] = len(unlocked_sorted)
    return sanitize_tokens(x), y, meta


def _make_markdown_window_impl(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    data_cfg: "DataConfig",
    collect_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    text_label_id = cfg.LANG2ID.get("text", cfg.PAD_ID)
    lids = list(dsets_by_lang.keys())
    code_lids = [lid for lid in lids if lid != text_label_id]
    if not code_lids:
        return _make_mixed_window_impl(
            dsets_by_lang,
            target_len,
            data_cfg.min_seg_len,
            collect_meta,
            data_cfg=data_cfg,
        )

    host_lid = random.choice(code_lids)
    host_snippet, host_ex = _sample_lang_snippet(
        dsets_by_lang,
        host_lid,
        max_chars=360,
        min_letters=0,
        strip=False,
    )
    if not host_snippet:
        return _make_mixed_window_impl(
            dsets_by_lang,
            target_len,
            data_cfg.min_seg_len,
            collect_meta,
            data_cfg=data_cfg,
        )

    block_plan: List[Dict[str, Any]] = [
        {"lid": host_lid, "snippet": host_snippet, "ex": host_ex, "role": "host"}
    ]
    if len(code_lids) > 1 and random.random() < 0.65:
        alt_candidates = [lid for lid in code_lids if lid != host_lid]
        random.shuffle(alt_candidates)
        for lid in alt_candidates:
            other_snippet, other_ex = _sample_lang_snippet(
                dsets_by_lang,
                lid,
                max_chars=300,
                min_letters=0,
                strip=False,
            )
            if other_snippet:
                block_plan.append({"lid": lid, "snippet": other_snippet, "ex": other_ex, "role": "other"})
                break

    if len(block_plan) > 1 and random.random() < 0.5:
        random.shuffle(block_plan)

    text_label = text_label_id if text_label_id is not None else host_lid
    byte_budget = target_len
    chars: List[str] = []
    labels: List[int] = []
    char_sources: Optional[List[int]] = [] if collect_meta else None
    added_code = False
    markdown_blocks_meta: List[Dict[str, Any]] = []
    host_sample_idx: Optional[int] = None
    host_block_lid = host_lid
    host_block_source: Optional[str] = None

    meta = {"samples": [], "line_injections": [], "markdown_blocks": []} if collect_meta else None
    inline_code_prob = float(getattr(data_cfg, "markdown_inline_code_prob", 0.25))

    def append_markup(text: str, label: Optional[int] = None) -> None:
        nonlocal byte_budget
        if byte_budget <= 0:
            return
        cleaned = _clean_snippet_text(text)
        trimmed, used = _trim_text_to_budget(cleaned, byte_budget)
        if used <= 0:
            byte_budget = 0
            return
        target_label = label if label is not None else text_label
        chars.extend(list(trimmed))
        labels.extend([target_label] * len(trimmed))
        if char_sources is not None:
            char_sources.extend([-1] * len(trimmed))
        byte_budget -= used

    def append_dataset_segment(
        lid: int,
        text: str,
        ex: Optional[dict],
        origin: str,
        *,
        mark_code: bool = False,
        source_hint: Optional[str] = None,
    ) -> Optional[int]:
        nonlocal byte_budget, added_code, host_block_source
        if byte_budget <= 0:
            return None
        cleaned = _clean_snippet_text(text)
        trimmed, used = _trim_text_to_budget(cleaned, byte_budget)
        if used <= 0:
            byte_budget = 0
            return None
        sample_idx: Optional[int] = None
        if collect_meta and meta is not None:
            requested_bytes = len(cleaned.encode("utf-8", "ignore"))
            entry = {
                "origin": origin,
                "language_id": int(lid),
                "language": _lang_name(lid),
                "source": source_hint or (_extract_example_source(ex) if ex else "synthetic_markdown"),
                "requested_bytes": requested_bytes,
                "chars": len(trimmed),
                "preview": trimmed[:160],
            }
            truncation = used < requested_bytes
            if truncation:
                entry["truncated"] = True
            meta["samples"].append(entry)
            sample_idx = len(meta["samples"]) - 1
            if origin == "markdown_code" and mark_code and host_block_source is None and lid == host_block_lid:
                host_block_source = entry["source"]
        chars.extend(list(trimmed))
        labels.extend([lid] * len(trimmed))
        if char_sources is not None:
            idx_val = sample_idx if sample_idx is not None else -1
            char_sources.extend([idx_val] * len(trimmed))
        byte_budget -= used
        if mark_code:
            added_code = True
        return sample_idx

    def sample_text_paragraph(max_chars: int = 220) -> Tuple[str, Optional[dict], bool]:
        snippet = ""
        ex: Optional[dict] = None
        if text_label_id is not None and text_label_id in dsets_by_lang:
            snippet, ex = _sample_lang_snippet(
                dsets_by_lang,
                text_label_id,
                max_chars=max_chars,
                min_letters=12,
                strip=True,
            )
        if not snippet:
            snippet, ex = _sample_fallback_text_snippet(
                dsets_by_lang,
                text_label_id,
                max_chars=max_chars,
                min_letters=8,
                min_length=32,
            )
        if not snippet:
            snippet, ex = _sample_fallback_text_snippet(
                dsets_by_lang,
                text_label_id,
                max_chars=max_chars,
                min_letters=0,
                min_length=8,
            )
        synthetic = False
        snippet = snippet.strip()
        return snippet, ex, synthetic

    def append_markdown_text(
        paragraph: str,
        paragraph_ex: Optional[dict],
        synthetic: bool,
        role: str,
        *,
        suffix: str = "",
    ) -> None:
        if not (paragraph or suffix):
            return

        text_source_hint = "synthetic_markdown_text" if synthetic else None

        def _add_text_piece(piece: str) -> Optional[int]:
            if not piece:
                return None
            idx = append_dataset_segment(
                text_label,
                piece,
                paragraph_ex,
                "markdown_text",
                source_hint=text_source_hint,
            )
            if collect_meta and meta is not None and idx is not None:
                meta["samples"][idx]["role"] = role
            return idx

        def _fallback():
            _add_text_piece((paragraph or "") + suffix)

        embed = bool(code_lids) and random.random() < inline_code_prob
        alt_pool = [lid for lid in code_lids if lid != host_lid] or code_lids
        if not embed or not alt_pool:
            _fallback()
            return

        alt_lid = random.choice(alt_pool)
        alt_snippet, alt_ex = _sample_lang_snippet(
            dsets_by_lang,
            alt_lid,
            max_chars=160,
            min_letters=0,
            strip=True,
        )
        alt_clean = alt_snippet.strip() if alt_snippet else ""
        if not alt_clean:
            _fallback()
            return

        lang_token = _pick_markdown_fence_label(alt_lid, allow_generic=False)

        code_mode = random.random()
        pre_markup: List[str]
        post_markup: List[str]
        code_text: Optional[str] = None
        spacer_after = True

        if code_mode < 0.33:
            inline_body = " ".join(alt_clean.split())
            inline_body = inline_body.replace("`", "'")[:160]
            if not inline_body:
                _fallback()
                return
            pre_markup = ["`"]
            post_markup = ["`"]
            code_text = inline_body
        elif code_mode < 0.66:
            fenced_body = alt_snippet.strip("\n")
            if not fenced_body:
                _fallback()
                return
            if not fenced_body.endswith("\n"):
                fenced_body += "\n"
            pre_markup = [f"```{lang_token}\n"] if lang_token else ["```\n"]
            post_markup = ["```\n"]
            code_text = fenced_body
            spacer_after = False
        else:
            safe_snippet = alt_snippet.replace("</code>", "&lt;/code&gt;")
            if not safe_snippet:
                _fallback()
                return
            class_token = lang_token or "plain"
            pre_markup = [f"<code class=\"language-{class_token}\">"]
            post_markup = ["</code>"]
            code_text = safe_snippet

        content = paragraph or ""
        insertion = len(content) // 2
        if insertion < len(content):
            while insertion < len(content) and not content[insertion].isspace():
                insertion += 1
        if insertion >= len(content):
            insertion = len(content) // 2
            while insertion > 0 and not content[insertion - 1].isspace():
                insertion -= 1
        prefix_text = content[:insertion]
        suffix_text = content[insertion:]

        _add_text_piece(prefix_text)
        if prefix_text and not prefix_text.endswith((" ", "\t", "\n")):
            append_markup(" ")

        for chunk in pre_markup:
            append_markup(chunk)

        code_idx = None
        if code_text:
            code_idx = append_dataset_segment(
                alt_lid,
                code_text,
                alt_ex,
                "markdown_inline_code",
                mark_code=True,
            )
        if collect_meta and meta is not None and code_idx is not None:
            meta["samples"][code_idx]["role"] = "inline_code"

        for chunk in post_markup:
            append_markup(chunk)

        trailing = suffix_text + suffix
        if spacer_after and trailing and not trailing[0].isspace():
            append_markup(" ")
        _add_text_piece(trailing)

    if random.random() < 0.25:
        if random.random() < 0.6 and code_lids:
            stray_lid = random.choice(code_lids)
            lang_token = _pick_markdown_fence_label(stray_lid, allow_generic=True)
            append_markup(f"```{lang_token}\n" if lang_token else "```\n")
        else:
            append_markup("```\n")

    if random.random() < 0.8 and byte_budget > 0:
        paragraph, paragraph_ex, synthetic = sample_text_paragraph()
        suffix = "\n\n" if random.random() < 0.6 else "\n"
        append_markdown_text(paragraph, paragraph_ex, synthetic, "intro_text", suffix=suffix)

    for idx, block in enumerate(block_plan):
        fenced = random.random() < 0.85
        include_lang = random.random() < 0.85
        mismatch = include_lang and random.random() < 0.2 and len(code_lids) > 1
        fence_lang_token = ""
        fence_label_source = "none"
        label_lid_for_meta: Optional[int] = None
        if include_lang:
            lang_for_token = block["lid"]
            fence_label_source = "actual"
            if mismatch:
                alt = [lid for lid in code_lids if lid != block["lid"] and _markdown_fence_tokens_for_lid(lid)]
                if alt:
                    lang_for_token = random.choice(alt)
                    fence_label_source = "mismatch"
            candidate_tokens = _markdown_fence_tokens_for_lid(lang_for_token)
            fence_lang_token = _pick_markdown_fence_label(
                lang_for_token if candidate_tokens else None,
                allow_generic=fence_label_source != "actual",
            )
            if not fence_lang_token and fence_label_source == "mismatch":
                fence_lang_token = _pick_markdown_fence_label(None, allow_generic=True)
                if fence_lang_token:
                    fence_label_source = "generic"
            if not fence_lang_token:
                fence_label_source = "none"
            else:
                label_lid_for_meta = int(lang_for_token)
        if fenced:
            fence_text = "```" + (fence_lang_token if fence_lang_token else "")
            if random.random() < 0.3:
                fence_text += " "
            fence_text += "\n"
            if random.random() < 0.1:
                fence_text = fence_text.rstrip("\n")
            append_markup(fence_text)
        elif random.random() < 0.2:
            append_markup("```\n")

        snippet = block["snippet"]
        if random.random() < 0.4:
            snippet = snippet.strip()
        if not snippet.endswith("\n"):
            snippet += "\n"
        if random.random() < 0.25:
            snippet = "\n".join(line.rstrip() for line in snippet.splitlines()) + "\n"

        sample_idx = append_dataset_segment(
            block["lid"],
            snippet,
            block.get("ex"),
            "markdown_code",
            mark_code=True,
        )
        if block["role"] == "host":
            host_sample_idx = sample_idx
            host_block_lid = block["lid"]
            if block.get("ex") is not None:
                host_block_source = _extract_example_source(block["ex"])

        close_added = False
        if fenced:
            must_close = (idx != len(block_plan) - 1) or random.random() < 0.8
            if must_close:
                closing = "```\n"
                if random.random() < 0.35:
                    closing = closing.rstrip("\n")
                append_markup(closing)
                close_added = True

        if collect_meta and meta is not None:
            markdown_blocks_meta.append(
                {
                    "order": len(markdown_blocks_meta),
                    "language_id": int(block["lid"]),
                    "language": _lang_name(block["lid"]),
                    "role": block["role"],
                    "fenced": bool(fenced),
                    "fence_language": fence_lang_token,
                    "fence_language_source": fence_label_source,
                    "fence_label_target_language_id": label_lid_for_meta,
                    "fence_label_mismatch": bool(fence_label_source in {"mismatch", "generic"}),
                    "closed": bool(close_added),
                }
            )

        if idx < len(block_plan) - 1 and byte_budget > 0:
            if random.random() < 0.75:
                paragraph, paragraph_ex, synthetic = sample_text_paragraph(max_chars=180)
                joiner = "\n\n" if random.random() < 0.5 else "\n"
                append_markdown_text(paragraph, paragraph_ex, synthetic, "between_text", suffix=joiner)
            elif random.random() < 0.3:
                append_markup("```\n")

    if byte_budget > 0 and random.random() < 0.65:
        paragraph, paragraph_ex, synthetic = sample_text_paragraph(max_chars=200)
        tail = "\n" if random.random() < 0.7 else ""
        append_markdown_text(paragraph, paragraph_ex, synthetic, "outro_text", suffix=tail)

    if byte_budget > 0 and random.random() < 0.3:
        append_markup("```")

    if not added_code or not chars:
        return _make_mixed_window_impl(
            dsets_by_lang,
            target_len,
            data_cfg.min_seg_len,
            collect_meta,
            data_cfg=data_cfg,
        )

    xb_u8, yb_u8, source_idx_bytes = _to_bytes_with_byte_labels(chars, labels, char_sources)
    if xb_u8.size == 0:
        return _make_mixed_window_impl(
            dsets_by_lang,
            target_len,
            data_cfg.min_seg_len,
            collect_meta,
            data_cfg=data_cfg,
        )

    x = np.full((target_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    y = np.full((target_len,), cfg.PAD_ID, dtype=np.uint8)
    used = min(int(xb_u8.shape[0]), target_len)
    x[:used] = xb_u8[:used].astype(np.int32)
    y[:used] = yb_u8[:used]

    if xb_u8.shape[0] > target_len:
        if source_idx_bytes is not None and source_idx_bytes.size:
            source_idx_bytes = source_idx_bytes[:target_len]
    elif source_idx_bytes is not None and source_idx_bytes.size < target_len:
        pad_len = target_len - source_idx_bytes.size
        if pad_len > 0:
            source_idx_bytes = np.concatenate(
                [source_idx_bytes, np.full((pad_len,), -1, dtype=np.int32)]
            )

    if collect_meta and meta is not None:
        if source_idx_bytes is not None and source_idx_bytes.size:
            _apply_final_byte_contributions(meta, source_idx_bytes)
        meta["markdown_blocks"] = markdown_blocks_meta
        if host_sample_idx is not None:
            samples_list = meta.get("samples", [])
            final_bytes = 0
            final_spans: List[Dict[str, int]] = []
            if 0 <= host_sample_idx < len(samples_list):
                final_bytes = samples_list[host_sample_idx].get("final_bytes", 0)
                final_spans = samples_list[host_sample_idx].get("final_spans", [])
            meta["host"] = {
                "language_id": int(host_block_lid),
                "language": _lang_name(host_block_lid),
                "source": host_block_source or (_extract_example_source(host_ex) if host_ex else "synthetic_markdown"),
                "final_bytes": final_bytes,
                "final_spans": final_spans,
            }
        else:
            meta["host"] = None

    return sanitize_tokens(x), y, meta


def make_line_injected_window(dsets_by_lang: Dict[int, hfds.Dataset],
                              target_len: int,
                              data_cfg: "DataConfig") -> Tuple[np.ndarray, np.ndarray]:
    x, y, _ = _make_line_injected_window_impl(dsets_by_lang, target_len, data_cfg, collect_meta=False)
    return sanitize_tokens(x), y


# ---------------------------
# Prefetcher (multi-thread, variable-length buckets)
# ---------------------------

class PrefetchBatcher:
    def __init__(self, dsets_by_lang, data_cfg: "DataConfig"):
        self.dsets_by_lang = dsets_by_lang
        self.cfg = data_cfg
        # Smaller queue size to prevent memory buildup
        self.q = queue.Queue(maxsize=max(2, data_cfg.prefetch_batches))
        self.stop_flag = threading.Event()
        self.buckets = data_cfg.buckets()
        self.threads: List[threading.Thread] = []
        self._gc_counter = 0
        # Use fewer worker threads
        for wid in range(max(1, data_cfg.num_workers // 2)):
            t = threading.Thread(target=self._worker, args=(wid,), daemon=True)
            t.start()
            self.threads.append(t)

    def _worker(self, wid: int):
        random.seed(self.cfg.seed ^ wid ^ int(time.time()))
        hold = max(1, self.cfg.bucket_hold_steps)
        L = random.choice(self.buckets)
        k = 0
        while not self.stop_flag.is_set():
            if k % hold == 0:
                L = random.choice(self.buckets)
            k += 1

            xb = np.full((self.cfg.batch_size, L), cfg.PAD_BYTE_ID, dtype=np.int32)
            yb = np.full((self.cfg.batch_size, L), cfg.PAD_ID, dtype=np.uint8)

            for i in range(self.cfg.batch_size):
                x, y = make_training_window(self.dsets_by_lang, L, self.cfg)
                xb[i], yb[i] = x, y

            try:
                self.q.put((xb, yb), timeout=1.0)
            except queue.Full:
                continue

    def get(self):
        return self.q.get()

    def close(self):
        self.stop_flag.set()
        # Drain queue to unblock workers
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        for t in self.threads:
            t.join(timeout=2.0)
