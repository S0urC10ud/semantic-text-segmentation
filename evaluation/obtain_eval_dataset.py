#!/usr/bin/env python3
"""Build curated evaluation datasets from the validation split.

This script samples fragments from the downloader-produced validation split and
derives a set of benchmark tasks covering:

  • Pure single-label fragments (baseline accuracy by content type).
  • Needle-in-the-haystack injections with multiple size buckets.
  • Back-to-back multi-label sequences (pairs and triplets).
  • Markdown-flavoured mixtures with optional prose/context blocks.
  • Throughput stress datasets targeting fixed byte budgets.

Each task is written as its own Hugging Face Dataset (Arrow format) under the
requested output directory together with an inventory manifest.

The generated records share a common schema:

  - ``task``: identifier for the evaluation scenario.
  - ``example_id``: deterministic unique id per example.
  - ``content``: UTF-8 text presented to the model.
  - ``segments``: ordered ground-truth segments with ``label``, ``char_start``,
    and ``char_end`` (exclusive) indices.
  - ``source_langs``: distinct labels contributing to the example.
  - ``metadata_json``: JSON-encoded auxiliary metadata (host/donor UIDs, etc.).

The default configuration samples 1,000 host fragments per content type for
each task as requested, but smaller counts can be supplied to keep dataset size
manageable during iteration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import string
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

import datasets as hfds
from datasets import Dataset, Features, Sequence as SeqFeature, Value


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from train import config as cfg  # noqa: E402

try:
    from evaluation.payloads import malicious_injections_to_discover  # type: ignore  # noqa: E402
except ModuleNotFoundError:  # running from within evaluation directory
    PAYLOADS_PATH = (REPO_ROOT / "evaluation" / "payloads.py").resolve()
    spec = importlib.util.spec_from_file_location("evaluation_payloads", PAYLOADS_PATH)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluation_payloads"] = module
    spec.loader.exec_module(module)
    malicious_injections_to_discover = module.malicious_injections_to_discover


@dataclass
class Fragment:
    """Represents a single validation fragment available for sampling."""

    lang: str
    content: str
    uid: Optional[str]
    extra_meta: Optional[dict]

NON_ASCII_PLACEHOLDER = "\u00A4"
_VISIBLE_ASCII_MIN = 0x20
_VISIBLE_ASCII_MAX = 0x7E
_ALLOWED_TEXT_CONTROLS = {"\n", "\t"}

_MAX_NEEDLE_COMMENT_RATIO = 0.4
_PAYLOAD_MAX_VISIBLE = 1000
_PAYLOAD_SHELL_CHOICES = ["/bin/bash", "/bin/sh", "/usr/bin/env python3", "/bin/zsh", "powershell", "cmd.exe"]
_PROHIBITED_INJECTION_LANGS = {"csv", "json", "yaml", "text", "html"}
_MARKDOWN_INLINE_CODE_PROB = 0.25


def _visible_char_count(text: str) -> int:
    """Count printable, non-whitespace characters."""
    return sum(1 for ch in text if ch.isprintable() and not ch.isspace())

def _ascii_letter_count(text: str) -> int:
    """Count only ASCII alphabetic characters."""
    return sum(1 for ch in text if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))

def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    out_chars: List[str] = []
    for ch in text:
        if ch == "\r":
            ch = "\n"
        code = ord(ch)
        if ch in _ALLOWED_TEXT_CONTROLS or _VISIBLE_ASCII_MIN <= code <= _VISIBLE_ASCII_MAX or ch == NON_ASCII_PLACEHOLDER:
            out_chars.append(ch)
        else:
            out_chars.append(NON_ASCII_PLACEHOLDER)
    return "".join(out_chars)


def _log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _load_val_fragments(
    data_root: Path,
    per_lang_target: int,
    *,
    extra_multiplier: int,
    seed: int,
) -> Dict[str, List[Fragment]]:
    """Load and sample validation fragments per language."""

    out: Dict[str, List[Fragment]] = {}
    missing: List[str] = []
    
    langs = sorted(cfg.LANG2ID.keys())
    _log("📂 Loading validation fragments...")
    for lang in tqdm(langs, desc="Loading languages", unit="lang"):
        path = data_root / "val" / lang / "dataset"
        if not path.exists():
            missing.append(lang)
            continue
        try:
            ds = hfds.load_from_disk(str(path))
        except Exception as exc:  # pragma: no cover - informative logging
            _log(f"⚠️  Failed to load validation dataset for '{lang}': {exc}")
            continue

        if len(ds) == 0:
            _log(f"⚠️  Validation dataset for '{lang}' is empty; skipping.")
            continue

        target = min(len(ds), per_lang_target * max(1, extra_multiplier))
        if target <= 0:
            continue

        seed_offset = seed + cfg.LANG2ID.get(lang, 0)
        sampled = ds.shuffle(seed=seed_offset).select(range(target))
        fragments: List[Fragment] = []
        for row in sampled:
            raw_text = row.get("content")
            text = _normalize_text(raw_text if isinstance(raw_text, str) else "")
            if not text:
                continue
            uid = row.get("uid") if isinstance(row, dict) else None
            extra_meta = {
                k: row[k]
                for k in ("source_repo", "source_repo_path", "source_ext", "license")
                if k in row and isinstance(row[k], str)
            }
            fragments.append(Fragment(lang=lang, content=text, uid=uid, extra_meta=extra_meta or None))

        if not fragments:
            _log(f"⚠️  No usable fragments collected for '{lang}'.")
            continue
        out[lang] = fragments

    if missing:
        _log(
            "⚠️  Missing validation splits for: "
            + ", ".join(sorted(missing))
        )
    return out


def _features_schema() -> Features:
    return Features(
        {
            "task": Value("string"),
            "example_id": Value("string"),
            "content": Value("string"),
            "segments": SeqFeature(
                feature={
                    "label": Value("string"),
                    "char_start": Value("int32"),
                    "char_end": Value("int32"),
                }
            ),
            "source_langs": SeqFeature(feature=Value("string")),
            "metadata_json": Value("string"),
        }
    )


def _hash_example_id(task: str, lang: str, index: int, salt: Optional[str] = None) -> str:
    base = f"{task}|{lang}|{index}"
    if salt:
        base += f"|{salt}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return f"{task}-{lang}-{index:05d}-{digest}"


def _segment(label: str, start: int, end: int) -> dict:
    return {"label": label, "char_start": int(start), "char_end": int(end)}


def _join_parts(parts: Iterable[Tuple[str, str]]) -> Tuple[str, List[dict], List[str]]:
    """Concatenate labelled text parts and return content, segments, labels."""

    segments: List[dict] = []
    langs: List[str] = []
    pieces: List[str] = []
    cursor = 0
    for label, text in parts:
        if not text:
            continue
        pieces.append(text)
        start = cursor
        cursor += len(text)
        segments.append(_segment(label, start, cursor))
        langs.append(label)
    combined = _normalize_text("".join(pieces))
    return combined, segments, langs


def _clean_snippet_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _trim_text_to_budget(text: str, max_chars: int) -> str:
    if max_chars <= 0 or not text:
        return ""
    if len(text) <= max_chars:
        return text
    start = 0
    if len(text) > max_chars:
        start = random.randint(0, max(0, len(text) - max_chars))
    return text[start:start + max_chars]


def _sample_fragment_snippet(
    fragments_by_lang: Dict[str, List[Fragment]],
    lang: str,
    rng: random.Random,
    *,
    max_chars: int,
    min_letters: int = 0,
    strip: bool = False,
) -> Tuple[str, Optional[Fragment]]:
    pool = fragments_by_lang.get(lang)
    if not pool:
        return "", None
    for _ in range(12):
        fragment = rng.choice(pool)
        snippet = _clean_snippet_text(fragment.content)
        if strip:
            snippet = snippet.strip()
        if not snippet:
            continue
        snippet = _trim_text_to_budget(snippet, max_chars)
        if min_letters and _ascii_letter_count(snippet) < min_letters:
            continue
        return snippet, fragment
    return "", None


def _sample_fallback_text_fragment(
    fragments_by_lang: Dict[str, List[Fragment]],
    rng: random.Random,
    *,
    max_chars: int,
    min_letters: int = 0,
    min_length: int = 24,
    attempts: int = 24,
) -> Tuple[str, Optional[Fragment]]:
    """
    Fallback sampler that best-effort draws prose from the validation text pool.
    Mirrors the behaviour used during training so eval data stays consistent.
    """
    pool = fragments_by_lang.get("text")
    if not pool:
        return "", None
    for _ in range(attempts):
        fragment = rng.choice(pool)
        snippet = _clean_snippet_text(fragment.content)
        if not snippet:
            continue
        if len(snippet) > max_chars:
            start = rng.randint(0, max(0, len(snippet) - max_chars))
            snippet = snippet[start:start + max_chars]
        snippet = snippet.strip()
        if len(snippet) < min_length:
            continue
        if min_letters > 0 and _ascii_letter_count(snippet) < min_letters:
            continue
        return snippet, fragment
    return "", None


def _make_record(
    task: str,
    lang: str,
    index: int,
    content: str,
    segments: List[dict],
    source_langs: Sequence[str],
    metadata: dict,
) -> dict:
    content = _normalize_text(content)
    example_id = _hash_example_id(task, lang, index, salt=str(metadata.get("seed", "")))
    uniq_langs = sorted({lbl for lbl in source_langs if isinstance(lbl, str) and lbl})
    return {
        "task": task,
        "example_id": example_id,
        "content": content,
        "segments": segments,
        "source_langs": uniq_langs,
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


def _classify_comment_lines(lines: Sequence[str]) -> List[bool]:
    """Return heuristic comment flags for each line."""
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


def _select_line_block(
    text: str,
    min_bytes: int,
    max_bytes: Optional[int],
    rng: random.Random,
) -> Optional[dict]:
    """Select a contiguous set of whole lines that fit the byte window."""
    if not text:
        return None
    lines = text.splitlines(keepends=True)
    if not lines:
        return None
    flags = _classify_comment_lines(lines)
    letter_lengths = [_ascii_letter_count(line) for line in lines]

    candidates: List[dict] = []
    total_lines = len(lines)
    for start in range(total_lines):
        total_letters = 0
        comment_lines = 0
        snippet_lines: List[str] = []
        for end in range(start, total_lines):
            snippet_lines.append(lines[end])
            total_letters += letter_lengths[end]
            if flags[end]:
                comment_lines += 1

            if total_letters < min_bytes:
                continue
            if max_bytes is not None and total_letters > max_bytes:
                break

            non_comment_count = len(snippet_lines) - comment_lines
            if non_comment_count <= 0:
                continue
            comment_ratio = comment_lines / len(snippet_lines)
            if comment_ratio > _MAX_NEEDLE_COMMENT_RATIO:
                continue

            snippet_text = "".join(snippet_lines)
            if not snippet_text.strip():
                continue

            candidates.append(
                {
                    "start_line": start,
                    "end_line": end + 1,
                    "text": snippet_text,
                    "visible_chars": total_letters,
                    "letter_chars": total_letters,
                    "comment_ratio": comment_ratio,
                    "comment_lines": comment_lines,
                    "total_lines": len(snippet_lines),
                }
            )

    if not candidates:
        return None

    min_ratio = min(candidate["comment_ratio"] for candidate in candidates)
    relaxed = [c for c in candidates if c["comment_ratio"] <= min_ratio + 0.1]
    chosen = rng.choice(relaxed if relaxed else candidates)
    return chosen


def _random_shell(rng: random.Random) -> str:
    return rng.choice(_PAYLOAD_SHELL_CHOICES)


def _random_ip(rng: random.Random) -> str:
    return ".".join(str(rng.randint(11, 223)) for _ in range(4))


def _random_port(rng: random.Random) -> int:
    return rng.randint(1024, 65535)


def _truncate_payload(snippet: str) -> str:
    if _visible_char_count(snippet) <= _PAYLOAD_MAX_VISIBLE:
        return _normalize_text(snippet)

    import re

    tokens = list(re.finditer(r"\S+", snippet, re.MULTILINE))
    if not tokens:
        data = snippet.encode("utf-8", "ignore")
        cut = data[: _PAYLOAD_MAX_VISIBLE].decode("utf-8", "ignore")
        return _normalize_text(cut)

    best = None
    best_visible = 0

    for start_idx, start_match in enumerate(tokens):
        start = start_match.start()
        for end_idx in range(start_idx, len(tokens)):
            end = tokens[end_idx].end()
            candidate = snippet[start:end]
            visible = _visible_char_count(candidate)
            if visible <= _PAYLOAD_MAX_VISIBLE and visible > best_visible:
                best = (start, end_idx)
                best_visible = visible
            if visible > _PAYLOAD_MAX_VISIBLE:
                break

    if best is None:
        data = snippet.encode("utf-8", "ignore")
        cut = data[: _PAYLOAD_MAX_VISIBLE].decode("utf-8", "ignore")
        return _normalize_text(cut)

    start, end_token_idx = best
    end = tokens[end_token_idx].end()
    trailing = snippet[end:]
    for offset, ch in enumerate(trailing):
        if not ch.isspace():
            break
        candidate = snippet[start : end + offset + 1]
        if _visible_char_count(candidate) <= _PAYLOAD_MAX_VISIBLE:
            end = end + offset + 1
        else:
            break

    truncated = snippet[start:end]
    return _normalize_text(truncated)


def _generate_payload_snippet(payload_lang: str, rng: random.Random) -> Optional[str]:
    options = malicious_injections_to_discover.get(payload_lang)
    if not options:
        return None
    template = rng.choice(options)
    shell = _random_shell(rng)
    ip = _random_ip(rng)
    port = _random_port(rng)
    snippet = template.replace("{shell}", shell).replace("{ip}", ip).replace("{port}", str(port))
    snippet = _truncate_payload(snippet.strip("\n"))
    return snippet if snippet else None


def _build_pure_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
) -> Tuple[str, List[dict], str]:
    task = "pure_fragments"
    examples: List[dict] = []
    desc = "Single-label validation fragments (baseline accuracy)."
    for lang, frags in fragments_by_lang.items():
        limit = min(per_label, len(frags))
        for idx in range(limit):
            frag = frags[idx]
            content = frag.content
            segments = [_segment(lang, 0, len(content))]
            meta = {
                "host_lang": lang,
                "host_uid": frag.uid,
                "source_meta": frag.extra_meta,
            }
            record = _make_record(task, lang, idx, content, segments, [lang], meta)
            examples.append(record)
    return task, examples, desc


def _choose_injection(
    fragments_by_lang: Dict[str, List[Fragment]],
    host_lang: str,
    rng: random.Random,
    min_bytes: int,
    max_bytes: Optional[int],
) -> Optional[Tuple[str, Fragment, dict]]:
    donor_langs = [
        lang
        for lang in fragments_by_lang.keys()
        if lang != host_lang
        and fragments_by_lang[lang]
        and lang.lower() not in _PROHIBITED_INJECTION_LANGS
    ]
    if not donor_langs:
        return None

    for _ in range(100):
        d_lang = rng.choice(donor_langs)
        donor = rng.choice(fragments_by_lang[d_lang])
        snippet_info = _select_line_block(donor.content, min_bytes, max_bytes, rng)
        if snippet_info:
            return d_lang, donor, snippet_info
    return None


def _build_injection_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    bucket_name: str,
    min_bytes: int,
    max_bytes: Optional[int],
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = f"needle_{bucket_name}"
    desc = (
        "Host fragments with a foreign-language needle injection sized "
        f"{min_bytes}-{'∞' if max_bytes is None else max_bytes} printable chars (whitespace ignored)."
    )
    examples: List[dict] = []
    _log(f"💉 Building injection dataset for size {min_bytes}-{'∞' if max_bytes is None else max_bytes} bytes...")

    for lang, frags in fragments_by_lang.items():
        if not frags:
            continue
        limit = min(per_label, len(frags))
        if limit == 0:
            continue
        for idx in range(limit):
            host = frags[idx]
            inj = _choose_injection(fragments_by_lang, lang, rng, min_bytes, max_bytes)
            if not inj:
                continue
            donor_lang, donor_frag, snippet_info = inj
            if donor_lang == lang:
                continue
            snippet = snippet_info.get("text")
            if not snippet:
                continue
            host_text = host.content
            if not host_text:
                continue
            insert_at = rng.randrange(0, len(host_text) + 1)
            left = host_text[:insert_at]
            right = host_text[insert_at:]
            parts = []
            if left:
                parts.append((lang, left))
            parts.append((donor_lang, snippet))
            if right:
                parts.append((lang, right))
            content, segments, langs = _join_parts(parts)
            inj_letters = _ascii_letter_count(snippet)
            inj_raw_bytes = len(snippet.encode("utf-8", "ignore"))
            meta = {
                "host_lang": lang,
                "host_uid": host.uid,
                "donor_lang": donor_lang,
                "donor_uid": donor_frag.uid,
                "insertion_char": insert_at,
                "injection_visible_chars": inj_letters,
                "injection_letter_chars": inj_letters,
                "injection_bytes": inj_raw_bytes,
                "injection_lines": snippet_info.get("total_lines"),
                "injection_comment_lines": snippet_info.get("comment_lines"),
                "injection_comment_ratio": snippet_info.get("comment_ratio"),
                "donor_start_line": snippet_info.get("start_line"),
                "donor_end_line": snippet_info.get("end_line"),
                "size_bucket": bucket_name,
            }
            record = _make_record(task, lang, idx, content, segments, langs, meta)
            examples.append(record)

    return task, examples, desc


def _build_malicious_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "mal_injection"
    desc = "Host fragments with malicious payload injections."
    examples: List[dict] = []
    payload_langs = [lang for lang, payloads in malicious_injections_to_discover.items() if payloads]
    if not payload_langs:
        return task, examples, desc

    _log("☠️  Building malicious injection dataset...")

    for host_lang, frags in fragments_by_lang.items():
        if not frags:
            continue
        available_payload_langs = [lang for lang in payload_langs if lang != host_lang]
        if not available_payload_langs:
            _log(f"⚠️  Skipping malicious payloads for '{host_lang}' — no distinct payload languages available.")
            continue
        limit = min(per_label, len(frags))
        if limit == 0:
            continue
        for idx in range(limit):
            host_frag = frags[idx]
            host_text = host_frag.content
            if not host_text:
                continue

            payload_lang = rng.choice(available_payload_langs)
            snippet = _generate_payload_snippet(payload_lang, rng)
            if not snippet:
                continue

            insert_at = rng.randrange(0, len(host_text) + 1)
            left = host_text[:insert_at]
            right = host_text[insert_at:]
            parts: List[Tuple[str, str]] = []
            if left:
                parts.append((host_lang, left))
            parts.append((payload_lang, snippet))
            if right:
                parts.append((host_lang, right))
            content, segments, langs = _join_parts(parts)
            payload_visible = _visible_char_count(snippet)
            payload_bytes = len(snippet.encode("utf-8", "ignore"))
            meta = {
                "host_lang": host_lang,
                "host_uid": host_frag.uid,
                "payload_lang": payload_lang,
                "payload_visible_chars": payload_visible,
                "payload_bytes": payload_bytes,
            }
            examples.append(_make_record(task, host_lang, idx, content, segments, langs, meta))

    return task, examples, desc


def _build_pair_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "sequence_pair"
    desc = "Two-language back-to-back sequences A->B."
    examples: List[dict] = []
    _log("🔄 Building sequence pair dataset...")
    labels = [lang for lang, frags in fragments_by_lang.items() if frags]
    for lang in labels:
        partners = [lbl for lbl in labels if lbl != lang]
        if not partners:
            continue
        host_list = fragments_by_lang[lang]
        limit = min(per_label, len(host_list))
        for idx in range(limit):
            host = host_list[idx]
            partner_lang = rng.choice(partners)
            partner_frag = rng.choice(fragments_by_lang[partner_lang])
            parts = [
                (lang, host.content.rstrip() + "\n\n"),
                (partner_lang, partner_frag.content.lstrip()),
            ]
            content, segments, langs = _join_parts(parts)
            meta = {
                "first_lang": lang,
                "second_lang": partner_lang,
                "host_uid": host.uid,
                "partner_uid": partner_frag.uid,
            }
            examples.append(_make_record(task, lang, idx, content, segments, langs, meta))
    return task, examples, desc


def _build_triplet_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "sequence_triplet"
    desc = "Three-language back-to-back sequences A->B->C."
    examples: List[dict] = []
    _log("🔄 Building sequence triplet dataset...")
    labels = [lang for lang, frags in fragments_by_lang.items() if frags]
    for lang in labels:
        others = [lbl for lbl in labels if lbl != lang]
        if len(others) < 2:
            continue
        host_list = fragments_by_lang[lang]
        limit = min(per_label, len(host_list))
        for idx in range(limit):
            host = host_list[idx]
            partner_langs = rng.sample(others, 2)
            mid_frag = rng.choice(fragments_by_lang[partner_langs[0]])
            tail_frag = rng.choice(fragments_by_lang[partner_langs[1]])
            parts = [
                (lang, host.content.rstrip() + "\n\n"),
                (partner_langs[0], mid_frag.content.strip() + "\n\n"),
                (partner_langs[1], tail_frag.content.lstrip()),
            ]
            content, segments, langs = _join_parts(parts)
            meta = {
                "first_lang": lang,
                "second_lang": partner_langs[0],
                "third_lang": partner_langs[1],
                "host_uid": host.uid,
                "mid_uid": mid_frag.uid,
                "tail_uid": tail_frag.uid,
            }
            examples.append(_make_record(task, lang, idx, content, segments, langs, meta))
    return task, examples, desc


def _maybe_trim_text(text: str, max_chars: int = 512) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n…\n" + text[-tail:]


def _build_markdown_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "markdown_mix"
    desc = "Markdown-like text/code interleavings with optional fences."
    examples: List[dict] = []
    _log("📝 Building markdown mix dataset...")
    labels = [lbl for lbl, frags in fragments_by_lang.items() if lbl != "text" and frags]
    text_available = bool(fragments_by_lang.get("text"))
    if not text_available:
        _log("⚠️  No 'text' fragments available; markdown dataset will omit prose blocks.")

    def _sample_text_paragraph(max_chars: int = 256) -> Tuple[str, Optional[Fragment], bool]:
        snippet, frag = _sample_fragment_snippet(
            fragments_by_lang, "text", rng, max_chars=max_chars, min_letters=12, strip=True
        )
        if not snippet:
            snippet, frag = _sample_fallback_text_fragment(
                fragments_by_lang,
                rng,
                max_chars=max_chars,
                min_letters=8,
                min_length=32,
            )
        if not snippet:
            snippet, frag = _sample_fallback_text_fragment(
                fragments_by_lang,
                rng,
                max_chars=max_chars,
                min_letters=0,
                min_length=8,
            )
        synthetic = False
        snippet = snippet.strip()
        return snippet, frag, synthetic

    for lang in labels:
        host_list = fragments_by_lang[lang]
        if not host_list:
            continue
        limit = min(per_label, len(host_list))
        alt_langs = [lbl for lbl in labels if lbl != lang]
        if not alt_langs:
            continue
        for idx in range(limit):
            host = host_list[idx]
            other_lang = rng.choice(alt_langs)
            other_pool = fragments_by_lang.get(other_lang)
            if not other_pool:
                continue
            other = rng.choice(other_pool)

            host_snippet = _clean_snippet_text(host.content).strip()
            host_snippet = _trim_text_to_budget(host_snippet, 540).strip()
            if not host_snippet:
                continue
            other_snippet = _clean_snippet_text(other.content).strip()
            other_snippet = _trim_text_to_budget(other_snippet, 360).strip()
            if not other_snippet:
                continue

            wrap_host = rng.random() < 0.5
            wrap_other = rng.random() < 0.5

            code_lids = [lbl for lbl in labels if fragments_by_lang.get(lbl)]

            content_chunks: List[str] = []
            segments: List[dict] = []
            langs_seen: List[str] = []
            cursor = 0

            inline_blocks: List[dict] = []
            markdown_blocks: List[dict] = []

            def append_part(label: str, text: str) -> Optional[dict]:
                nonlocal cursor
                if not text:
                    return None
                normalized = _normalize_text(text)
                if not normalized:
                    return None
                start = cursor
                end = start + len(normalized)
                content_chunks.append(normalized)
                seg = _segment(label, start, end)
                segments.append(seg)
                langs_seen.append(label)
                cursor = end
                return seg

            def append_markdown_text(
                paragraph: str,
                paragraph_frag: Optional[Fragment],
                synthetic: bool,
                role: str,
                *,
                suffix: str = "",
            ) -> None:
                if not (paragraph or suffix):
                    return

                embed = bool(code_lids) and rng.random() < _MARKDOWN_INLINE_CODE_PROB
                alt_pool = [lbl for lbl in code_lids if lbl != lang]
                if not embed or not alt_pool:
                    if paragraph:
                        append_part("text", paragraph)
                    if suffix:
                        append_part("text", suffix)
                    return

                alt_lang = rng.choice(alt_pool)
                alt_snippet_raw, alt_fragment = _sample_fragment_snippet(
                    fragments_by_lang, alt_lang, rng, max_chars=160, strip=True
                )
                alt_clean = alt_snippet_raw.strip()
                if not alt_clean:
                    if paragraph:
                        append_part("text", paragraph)
                    if suffix:
                        append_part("text", suffix)
                    return

                lang_token = alt_lang.replace("_", "")

                code_mode = rng.random()
                pre_markup: List[str]
                post_markup: List[str]
                code_text = ""
                wrapper_type = "inline_backtick"
                spacer_after = True
                display_label: Optional[str] = None

                if code_mode < 0.33:
                    inline_body = " ".join(alt_clean.split()).replace("`", "'")
                    inline_body = inline_body[:160]
                    if not inline_body:
                        if paragraph:
                            append_part("text", paragraph)
                        if suffix:
                            append_part("text", suffix)
                        return
                    pre_markup = ["`"]
                    post_markup = ["`"]
                    code_text = inline_body
                    wrapper_type = "inline_backtick"
                elif code_mode < 0.66:
                    fenced_body = alt_snippet_raw.strip("\n")
                    if not fenced_body:
                        if paragraph:
                            append_part("text", paragraph)
                        if suffix:
                            append_part("text", suffix)
                        return
                    if not fenced_body.endswith("\n"):
                        fenced_body += "\n"
                    pre_markup = [f"```{lang_token}\n"]
                    post_markup = ["```\n"]
                    code_text = fenced_body
                    wrapper_type = "inline_fence"
                    display_label = lang_token
                    spacer_after = False
                else:
                    html_body = alt_snippet_raw.replace("</code>", "&lt;/code&gt;")
                    if not html_body:
                        if paragraph:
                            append_part("text", paragraph)
                        if suffix:
                            append_part("text", suffix)
                        return
                    pre_markup = [f"<code class=\"language-{lang_token}\">"]
                    post_markup = ["</code>"]
                    code_text = html_body
                    wrapper_type = "html_code"

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

                if prefix_text:
                    append_part("text", prefix_text)
                if prefix_text and not prefix_text.endswith((" ", "\t", "\n")):
                    append_part("text", " ")

                for chunk in pre_markup:
                    append_part("text", chunk)

                code_seg = append_part(alt_lang, code_text)
                if code_seg:
                    inline_blocks.append(
                        {
                            "language": alt_lang,
                            "wrapper": wrapper_type,
                            "display_label": display_label,
                            "char_start": code_seg["char_start"],
                            "char_end": code_seg["char_end"],
                            "context_role": role,
                            "source_uid": alt_fragment.uid if alt_fragment else None,
                        }
                    )

                for chunk in post_markup:
                    append_part("text", chunk)

                trailing = suffix_text + suffix
                if spacer_after and trailing and not trailing[0].isspace():
                    append_part("text", " ")
                if trailing:
                    append_part("text", trailing)

            intro_text, intro_frag, intro_synth = _sample_text_paragraph(240)
            if rng.random() < 0.65:
                append_markdown_text(intro_text, intro_frag, intro_synth, "intro_text", suffix="\n\n")

            block_plan: List[Dict[str, Any]] = [
                {
                    "role": "host",
                    "language": lang,
                    "snippet": host_snippet,
                    "fragment": host,
                    "wrapped": wrap_host,
                }
            ]
            alt_block_snippet = other_snippet
            block_plan.append(
                {
                    "role": "other",
                    "language": other_lang,
                    "snippet": alt_block_snippet,
                    "fragment": other,
                    "wrapped": wrap_other,
                }
            )

            if len(block_plan) > 1 and rng.random() < 0.5:
                rng.shuffle(block_plan)

            host_block_meta: Optional[dict] = None
            other_block_meta: Optional[dict] = None

            for block_idx, block in enumerate(block_plan):
                block_role = block["role"]
                block_lang = block["language"]
                block_fragment = block["fragment"]
                block_snippet = block["snippet"]
                wrapped = bool(block.get("wrapped", False))

                include_lang = rng.random() < 0.85
                mismatch = include_lang and rng.random() < 0.2 and len(labels) > 1
                display_lang_name = ""
                display_token = ""
                if wrapped and include_lang:
                    candidate = block_lang
                    if mismatch:
                        alternatives = [lbl for lbl in labels if lbl != block_lang]
                        if alternatives:
                            candidate = rng.choice(alternatives)
                        else:
                            mismatch = False
                    display_lang_name = candidate
                    display_token = candidate.replace("_", "")

                snippet_body = block_snippet
                if rng.random() < 0.4:
                    snippet_body = snippet_body.strip()
                if not snippet_body.endswith("\n"):
                    snippet_body += "\n"
                if rng.random() < 0.25:
                    snippet_body = "\n".join(line.rstrip() for line in snippet_body.splitlines()) + "\n"

                if wrapped:
                    fence_prefix = "```"
                    if display_token:
                        fence_prefix += display_token
                        if rng.random() < 0.3:
                            fence_prefix += " "
                    fence_prefix += "\n"
                    if rng.random() < 0.1:
                        fence_prefix = fence_prefix.rstrip("\n")
                    append_part("text", fence_prefix)
                elif rng.random() < 0.2:
                    append_part("text", "```\n")

                code_segment = append_part(block_lang, snippet_body)

                closed = False
                if wrapped:
                    must_close = (block_idx != len(block_plan) - 1) or rng.random() < 0.8
                    if must_close:
                        closing = "```\n"
                        if rng.random() < 0.35:
                            closing = closing.rstrip("\n")
                        append_part("text", closing)
                        closed = True

                if code_segment:
                    block_entry = {
                        "role": block_role,
                        "wrapped": bool(wrapped),
                        "language": block_lang,
                        "display_label": display_token,
                        "display_language": display_lang_name,
                        "mismatched": bool(
                            wrapped
                            and include_lang
                            and mismatch
                            and display_lang_name
                            and display_lang_name != block_lang
                        ),
                        "char_start": code_segment["char_start"],
                        "char_end": code_segment["char_end"],
                        "source_uid": block_fragment.uid if block_fragment else None,
                        "wrapper": "wrapped" if wrapped else "plain",
                        "closed": closed,
                    }
                    markdown_blocks.append(block_entry)
                    if block_role == "host":
                        host_block_meta = block_entry
                    elif block_role == "other":
                        other_block_meta = block_entry

                if block_idx < len(block_plan) - 1 and rng.random() < 0.75:
                    between_text, between_frag, between_synth = _sample_text_paragraph(180)
                    joiner = "\n\n" if rng.random() < 0.5 else "\n"
                    append_markdown_text(between_text, between_frag, between_synth, "between_text", suffix=joiner)
                elif block_idx < len(block_plan) - 1 and rng.random() < 0.3:
                    append_part("text", "```\n")

            if rng.random() < 0.65:
                outro_text, outro_frag, outro_synth = _sample_text_paragraph(200)
                tail = "\n" if rng.random() < 0.7 else ""
                append_markdown_text(outro_text, outro_frag, outro_synth, "outro_text", suffix=tail)

            if rng.random() < 0.3:
                append_part("text", "```")

            if not any(seg["label"] != "text" for seg in segments):
                continue

            content = "".join(content_chunks)
            host_display_token = host_block_meta["display_label"] if host_block_meta else ""
            host_display_language = host_block_meta["display_language"] if host_block_meta else ""
            other_display_token = other_block_meta["display_label"] if other_block_meta else ""
            other_display_language = other_block_meta["display_language"] if other_block_meta else ""

            meta = {
                "host_lang": lang,
                "host_uid": host.uid,
                "other_lang": other_lang,
                "other_uid": other.uid,
                "wrapped_host": bool(host_block_meta.get("wrapped") if host_block_meta else wrap_host),
                "wrapped_other": bool(other_block_meta.get("wrapped") if other_block_meta else wrap_other),
                "host_fence_label": host_display_token,
                "host_fence_language": host_display_language,
                "host_fence_mismatch": bool(host_block_meta.get("mismatched") if host_block_meta else False),
                "other_fence_label": other_display_token,
                "other_fence_language": other_display_language,
                "other_fence_mismatch": bool(other_block_meta.get("mismatched") if other_block_meta else False),
                "markdown_blocks": markdown_blocks,
                "inline_blocks": inline_blocks,
            }
            examples.append(
                _make_record(task, lang, idx, content, segments, langs_seen, meta)
            )

    return task, examples, desc


def _assemble_label_block(
    fragments: List[Fragment],
    target_bytes: int,
    rng: random.Random,
) -> Tuple[str, int]:
    parts: List[str] = []
    total = 0
    attempt = 0
    while total < target_bytes and attempt < target_bytes * 4:
        frag = rng.choice(fragments)
        snippet = frag.content.strip()
        if not snippet:
            attempt += 1
            continue
        if parts:
            snippet = "\n" + snippet
        parts.append(snippet)
        total = len("".join(parts).encode("utf-8", "ignore"))
        attempt += 1
    combined = "".join(parts)
    data = combined.encode("utf-8", "ignore")
    if not data:
        return "", 0
    if len(data) > target_bytes:
        data = data[:target_bytes]
        combined = data.decode("utf-8", "ignore")
    return _normalize_text(combined), len(data)


def _build_throughput_dataset(
    size_bytes: int,
    num_examples: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = f"throughput_{size_bytes}"
    pretty = {
        1024: "1 KiB",
        10_240: "10 KiB",
        102_400: "100 KiB",
        1_048_576: "1 MiB",
    }.get(size_bytes, f"{size_bytes} bytes")
    desc = f"Synthetic random text blobs for throughput ({pretty})."
    alphabet = string.ascii_letters + string.digits + string.punctuation + "\n "
    examples: List[dict] = []
    sample_count = max(1, num_examples)
    _log(f"🚀 Building throughput dataset for {pretty} with {sample_count} random samples...")
    for idx in range(sample_count):
        chars = [rng.choice(alphabet) for _ in range(size_bytes)]
        raw_content = "".join(chars)
        content = _normalize_text(raw_content)
        actual_bytes = len(content.encode("utf-8", "ignore"))
        meta = {
            "target_bytes": size_bytes,
            "actual_bytes": actual_bytes,
            "mode": "synthetic_random",
        }
        segments = [_segment("text", 0, len(content))]
        examples.append(
            _make_record(
                task,
                "synthetic",
                idx,
                content,
                segments,
                ["text"],
                meta,
            )
        )
    return task, examples, desc


def _write_dataset(
    root: Path,
    task: str,
    examples: List[dict],
    features: Features,
    overwrite: bool,
) -> int:
    if not examples:
        _log(f"⚠️  Task '{task}' produced no examples; skipping write.")
        return 0
    out_dir = root / task
    if out_dir.exists():
        if overwrite:
            _log(f"🔄 Removing existing dataset at {out_dir}")
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Dataset already exists at {out_dir}; use --overwrite to replace it.")
    ds = Dataset.from_list(examples, features=features)
    ds_size_mb = sum(sys.getsizeof(ex) for ex in examples) / (1024 * 1024)
    _log(f"💾 Writing {len(ds):,} examples ({ds_size_mb:.1f} MB) to {out_dir}")
    ds.save_to_disk(str(out_dir))
    return len(ds)


def build_all_datasets(args) -> dict:
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    fragments_by_lang = _load_val_fragments(
        data_root,
        args.per_label,
        extra_multiplier=args.extra_pool_multiplier,
        seed=args.seed,
    )
    if not fragments_by_lang:
        raise RuntimeError("No validation fragments available – cannot build evaluation datasets.")

    rng = random.Random(args.seed)
    features = _features_schema()
    manifest_tasks = []

    _log("🔧 Building evaluation datasets...")
    builders = []
    _log("📊 Building pure fragments dataset...")
    builders.append(_build_pure_dataset(fragments_by_lang, args.per_label))

    buckets = [
        ("4_15", 4, 15),
        ("16_31", 16, 31),
        ("32_63", 32, 63),
        ("64_plus", 64, None),
    ]
    for name, lo, hi in buckets:
        builders.append(
            _build_injection_dataset(
                fragments_by_lang,
                args.per_label,
                name,
                lo,
                hi,
                rng,
            )
        )

    builders.append(_build_malicious_dataset(fragments_by_lang, args.per_label, rng))
    builders.append(_build_pair_dataset(fragments_by_lang, args.per_label, rng))
    builders.append(_build_triplet_dataset(fragments_by_lang, args.per_label, rng))
    builders.append(_build_markdown_dataset(fragments_by_lang, args.per_label, rng))

    for size in (1024, 10_240, 102_400, 1_048_576):
        builders.append(
            _build_throughput_dataset(size, args.throughput_examples, rng)
        )

    total_examples = 0
    _log("\n� Writing datasets to disk...")
    for task, examples, desc in tqdm(builders, desc="Writing datasets", unit="dataset"):
        count = _write_dataset(output_root, task, examples, features, overwrite=args.overwrite)
        total_examples += count
        manifest_tasks.append({
            "task": task,
            "description": desc,
            "count": count,
            "path": str((output_root / task).relative_to(output_root)),
        })
    _log(f"\n📊 Summary: Generated {len(manifest_tasks)} datasets with {total_examples:,} total examples")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "per_label": args.per_label,
        "throughput_examples": args.throughput_examples,
        "tasks": manifest_tasks,
    }

    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Generate evaluation datasets from validation fragments.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / "downloader" / "arrow_out"),
        help="Path to downloader outputs (expects <split>/<lang>/dataset directories).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(REPO_ROOT / "evaluation" / "data"),
        help="Directory where evaluation datasets will be written.",
    )
    parser.add_argument(
        "--per-label",
        type=int,
        default=100,
        help="Examples per label for most benchmarks (pure, injection, sequences, markdown).",
    )
    parser.add_argument(
        "--throughput-examples",
        type=int,
        default=8,
        help="Number of synthetic samples to create for each throughput size.",
    )
    parser.add_argument(
        "--extra-pool-multiplier",
        type=int,
        default=3,
        help="Multiplier to expand the initial validation sample pool per label.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing datasets in the output directory.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = build_all_datasets(args)
    _log(f"✅ Generated {len(manifest['tasks'])} datasets under {manifest['output_root']}")


if __name__ == "__main__":
    main()
