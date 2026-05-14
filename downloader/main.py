"""
End-to-end The Stack → filtered windows → Arrow datasets (per label) with 70/10/10/10 splits

This version ALWAYS writes four disjoint datasets per content type (label) to:
  <out-root>/<split>/<label>/dataset
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Set
from bisect import bisect_left
import html
from html.parser import HTMLParser

from dotenv import load_dotenv
load_dotenv()

# Keep native threadpools from over-subscribing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 3rd-party deps expected:
#   datasets>=2.14, magika==1.0.*, numpy, tqdm
from datasets import Dataset, Features, Value, load_dataset, load_from_disk, concatenate_datasets
from datasets.exceptions import DatasetGenerationError
from tqdm import tqdm
import numpy as np
from magika_label_map import canonical_label, label_matches_target


# ============================================================
#                   Utility / Canonicalization
# ============================================================

def safe_filename(name: str) -> str:
    s = (name or "")
    lower = s.lower()
    if lower in ("c", "c++", "cpp", "c-family", "cfamily"):
        return "c_family"
    if lower in ("c#", "c-sharp", "csharp", "cs"):  # accept cs
        return "csharp"
    if lower in ("yml",):
        return "yaml"
    s = s.replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)

MADLAD_REPO_ID = "allenai/MADLAD-400"
MADLAD_DATA_GLOB = "hf://datasets/allenai/MADLAD-400/data/{lang}/*_clean_*.jsonl.gz"
MADLAD_MIN_CHARS = 50
_MADLAD_TAG_RE = re.compile(r"<[^>]+>")
_MADLAD_WS_RE = re.compile(r"\s+")
_MADLAD_LANG_CACHE: Optional[List[str]] = None

W3TECHS_PRIOR = {
    "en": 0.493,
    "es": 0.060,
    "de": 0.059,
    "ja": 0.051,
    "fr": 0.045,
    "pt": 0.040,
    "ru": 0.037,
    "it": 0.028,
    "nl": 0.022,
    "pl": 0.018,
    "tr": 0.016,
    "fa": 0.011,
    "zh": 0.011,
    "vi": 0.010,
    "cs": 0.010,
    "id": 0.009,
    "ko": 0.008,
    "uk": 0.007,
    "hu": 0.006,
    "sv": 0.005,
    "ar": 0.005,
    "ro": 0.005,
    "el": 0.005,
    "da": 0.004,
    "fi": 0.004,
    "he": 0.004,
    "sk": 0.004,
    "th": 0.003,
    "bg": 0.003,
    "hr": 0.002,
    "no": 0.002,
    "lt": 0.002,
    "sr": 0.002,
    "sl": 0.001,
    "ca": 0.001,
    "et": 0.001,
    "lv": 0.001,
}
def resolve_hf_token_pair(use_auth_token: bool) -> Tuple[Optional[str], Optional[Any]]:
    token_str: Optional[str] = None
    token_arg: Optional[Any] = None
    if not use_auth_token:
        return token_str, token_arg
    try:
        from huggingface_hub import get_token  # type: ignore
        tok = get_token()
        if tok:
            token_str = tok
            token_arg = tok
        else:
            token_arg = True
    except Exception:
        token_arg = True
    if token_str is None:
        for env_key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_API_TOKEN"):
            val = os.getenv(env_key)
            if val:
                token_str = val
                if token_arg in (None, True):
                    token_arg = val
                break
    return token_str, token_arg

def normalize_madlad_text(text: Optional[str]) -> str:
    if not text:
        return ""
    t = html.unescape(text)
    t = t.replace("\\n", "\n")
    t = _MADLAD_TAG_RE.sub("", t)
    t = _MADLAD_WS_RE.sub(" ", t).strip()
    return t

def madlad_window_chunks(
    text: str,
    window_chars: int,
    stride_chars: int,
    min_chars: int,
) -> Iterator[Tuple[int, str]]:
    if not text:
        return
    n = len(text)
    if n < min_chars:
        return
    stride = stride_chars if stride_chars > 0 else window_chars
    idx = 0
    start = 0
    while start < n:
        chunk = text[start:start + window_chars]
        if len(chunk) < min_chars:
            break
        if len(chunk) == window_chars:
            last_ws = chunk.rfind(" ")
            if last_ws >= window_chars // 2:
                chunk = chunk[:last_ws]
        if len(chunk) < min_chars:
            break
        yield idx, chunk
        idx += 1
        start += stride

def madlad_list_languages(token: Optional[str]) -> List[str]:
    global _MADLAD_LANG_CACHE
    if _MADLAD_LANG_CACHE is not None:
        return list(_MADLAD_LANG_CACHE)
    try:
        from huggingface_hub import HfFileSystem  # type: ignore
    except Exception as e:
        raise RuntimeError("huggingface_hub is required to enumerate MADLAD languages") from e
    fs = HfFileSystem(token=token)
    try:
        entries = fs.ls(f"datasets/{MADLAD_REPO_ID}/data", detail=True)
    except Exception as e:
        raise RuntimeError(f"Failed to list MADLAD languages: {e}") from e
    langs = sorted(
        entry["name"].split("/")[-1]
        for entry in entries
        if entry.get("type") == "directory"
    )
    if not langs:
        raise RuntimeError("No language directories found in MADLAD-400 /data.")
    _MADLAD_LANG_CACHE = list(langs)
    return list(langs)

def madlad_lang_window_iter(
    lang: str,
    *,
    window_chars: int,
    stride_chars: int,
    min_chars: int,
    token: Optional[Any],
) -> Iterator[Tuple[int, str]]:
    data_glob = MADLAD_DATA_GLOB.format(lang=lang)
    load_kwargs = dict(
        path="json",
        data_files=data_glob,
        split="train",
        streaming=True,
    )
    if token is not None:
        load_kwargs["token"] = token
    try:
        ds = load_dataset(**load_kwargs)
    except TypeError:
        load_kwargs.pop("token", None)
        ds = load_dataset(**load_kwargs)
    except FileNotFoundError as e:
        raise RuntimeError(f"No MADLAD clean files for language '{lang}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to open MADLAD language '{lang}': {e}") from e

    for ex in ds:
        raw = (
            ex.get("text")
            or ex.get("document")
            or ex.get("content")
            or ex.get("body")
            or ""
        )
        norm = normalize_madlad_text(raw)
        if not norm:
            continue
        for idx, chunk in madlad_window_chunks(
            norm,
            window_chars=window_chars,
            stride_chars=stride_chars,
            min_chars=min_chars,
        ):
            yield idx, chunk

def gen_text_windows_from_madlad(
    *,
    window_bytes: int,
    add_meta: bool,
    progress_mode: str,
    budget_per_split: Dict[str, int],
    seen_uids: Optional[Set[str]],
    use_auth_token: bool,
    base_seed: int,
) -> Iterator[dict]:
    window_chars = max(1, int(window_bytes))
    stride_chars = window_chars
    min_chars = window_chars if window_chars < MADLAD_MIN_CHARS else MADLAD_MIN_CHARS
    total_budget = sum(max(0, b) for b in budget_per_split.values())
    is_tty = sys.stderr.isatty()
    show_progress = ((progress_mode == "always") or (progress_mode == "auto" and is_tty))
    pbar = tqdm(total=total_budget if total_budget > 0 else None,
                unit="win",
                desc="text (madlad)",
                disable=not show_progress)

    token_str, token_arg = resolve_hf_token_pair(use_auth_token)
    try:
        langs = madlad_list_languages(token=token_str)
    except Exception as e:
        pbar.close()
        raise
    rng = random.Random(base_seed)

    available_langs = set(langs)
    prioritized = [(lang, W3TECHS_PRIOR[lang]) for lang in W3TECHS_PRIOR if lang in available_langs]
    lang_targets: Dict[str, int] = {}
    if total_budget > 0 and prioritized:
        total_weight = sum(weight for _, weight in prioritized)
        if total_weight > 0:
            assigned = 0
            fractions: Dict[str, Tuple[float, int]] = {}
            for idx, (lang, weight) in enumerate(prioritized):
                share = (weight / total_weight) * total_budget
                base = int(math.floor(share))
                lang_targets[lang] = base
                fractions[lang] = (share - base, idx)
                assigned += base
            remainder = total_budget - assigned
            if remainder > 0:
                order = sorted(
                    ((lang, frac[0], frac[1]) for lang, frac in fractions.items()),
                    key=lambda item: (-item[1], item[2]),
                )
                if order:
                    pos = 0
                    while remainder > 0:
                        lang = order[pos % len(order)][0]
                        lang_targets[lang] += 1
                        remainder -= 1
                        pos += 1
            lang_targets = {lang: count for lang, count in lang_targets.items() if count > 0}

    plan_items = list(lang_targets.items())
    rng.shuffle(plan_items)
    prioritized_set = set(lang_targets.keys())
    fallback_langs = [lang for lang in langs if lang not in prioritized_set]
    rng.shuffle(fallback_langs)

    kept_total = 0
    kept_per_split = {s: 0 for s in WINDOW_SPLITS}
    rejected = 0
    lang_id = np.int16(LANG2ID.get("text", -1)).item()
    last_postfix = time.time()

    def iter_language(lang: str, quota: Optional[int]) -> Iterator[dict]:
        nonlocal kept_total, rejected, last_postfix
        if quota is not None and quota <= 0:
            return
        try:
            iterator = madlad_lang_window_iter(
                lang,
                window_chars=window_chars,
                stride_chars=stride_chars,
                min_chars=min_chars,
                token=token_arg,
            )
        except Exception as e:
            sys.stderr.write(f"[warn] MADLAD '{lang}' iteration failed: {e}\n")
            return

        got_for_lang = 0
        for win_idx, chunk in iterator:
            if total_budget > 0 and kept_total >= total_budget:
                break
            if quota is not None and got_for_lang >= quota:
                break

            ascii_text = map_text_to_ascii(chunk)
            if not ascii_text:
                rejected += 1
                continue
            ascii_text, ascii_bytes = clamp_utf8_bytes(ascii_text, window_bytes)
            if not ascii_text or not ascii_bytes:
                rejected += 1
                continue

            uid = stable_uid_for_window(ascii_bytes)
            if seen_uids is not None and uid in seen_uids:
                rejected += 1
                continue

            split = split_for_uid(uid)
            if budget_per_split.get(split, 0) <= 0:
                rejected += 1
                continue

            payload = {
                "content": ascii_text,
                "lang_id": lang_id,
                "uid": uid,
                "split": split,
            }
            if add_meta:
                payload.update({
                    "win_idx": np.int64(win_idx).item(),
                    "source_ext": "",
                    "source_hexsha": "",
                    "source_repo": MADLAD_REPO_ID,
                    "source_repo_path": lang,
                    "license": "unknown",
                })

            budget_per_split[split] = max(0, budget_per_split[split] - 1)
            kept_per_split[split] += 1
            kept_total += 1
            got_for_lang += 1
            if seen_uids is not None:
                seen_uids.add(uid)

            yield payload
            pbar.update(1)

            now = time.time()
            if (now - last_postfix) >= 0.3:
                pbar.set_postfix(
                    kept_total=kept_total,
                    rej=rejected,
                    k_train=kept_per_split["train"],
                    k_val=kept_per_split["val"],
                )
                last_postfix = now

    def process_languages(sequence: Sequence[Tuple[str, Optional[int]]]) -> Iterator[dict]:
        nonlocal kept_total
        for lang, quota in sequence:
            if total_budget > 0 and kept_total >= total_budget:
                break
            remaining = sum(budget_per_split.values())
            if remaining <= 0:
                break
            for payload in iter_language(lang, quota):
                yield payload

    # First pass: enforce W3Techs quotas for prioritized languages.
    for item in process_languages(plan_items):
        yield item

    # Second pass: fill any remaining budget with fallback languages (or retry prioritized ones).
    pending = sum(budget_per_split.values())
    if pending > 0:
        if not fallback_langs:
            fallback_langs = [lang for lang, _ in plan_items]
            rng.shuffle(fallback_langs)
        remaining_langs = len(fallback_langs)
        for lang in fallback_langs:
            if pending <= 0 or (total_budget > 0 and kept_total >= total_budget):
                break
            quota = (pending + remaining_langs - 1) // remaining_langs if remaining_langs > 0 else None
            for payload in iter_language(lang, quota):
                yield payload
            remaining_langs -= 1
            pending = sum(budget_per_split.values())

    pbar.set_postfix(
        kept_total=kept_total,
        rej=rejected,
        k_train=kept_per_split["train"],
        k_val=kept_per_split["val"],
    )
    pbar.close()

_SCRIPT_STYLE_TAGS = {"script", "style"}
_SCRIPT_STYLE_BLOCK_RE = re.compile(r"(?is)<\s*(script|style)\b[^>]*>.*?</\s*\1\s*>")
_SCRIPT_STYLE_SELF_CLOSE_RE = re.compile(r"(?is)<\s*(script|style)\b[^>]*/>")
_SCRIPT_STYLE_OPEN_RE = re.compile(r"(?is)<\s*(script|style)\b[^>]*>")
_SCRIPT_STYLE_CLOSE_RE = re.compile(r"(?is)</\s*(script|style)\s*>")
_FRAMEWORK_KEYWORDS_RE = re.compile(r"(?i)\b(angular|react|svelte|vue)\b")

_SVG_STYLE_BLOCK_RE = re.compile(r"(?is)<\s*style\b[^>]*>.*?</\s*style\s*>")
_SVG_STYLE_SELF_CLOSE_RE = re.compile(r"(?is)<\s*style\b[^>]*/>")

_DANGEROUS_URI_ATTRS = {"href", "src", "xlink:href", "formaction", "action", "data", "poster"}


def _is_event_attribute(name: str) -> bool:
    if not name:
        return False
    lower = name.lower()
    if lower.startswith("on"):
        return True
    if ":on" in lower:
        return True
    if lower.startswith("@"):
        return True
    if lower.startswith("x-on") or lower.startswith("hx-on"):
        return True
    return False


def _has_javascript_scheme(value: Optional[str]) -> bool:
    if value is None:
        return False
    val = html.unescape(value).strip().lower()
    return val.startswith("javascript:")


def _sanitize_attrs(attrs: List[Tuple[str, Optional[str]]]) -> List[Tuple[str, Optional[str]]]:
    sanitized: List[Tuple[str, Optional[str]]] = []
    for name, val in attrs:
        if not name:
            continue
        if _is_event_attribute(name):
            continue
        if name.lower() in _DANGEROUS_URI_ATTRS and _has_javascript_scheme(val):
            continue
        sanitized.append((name, val))
    return sanitized


def _serialize_attrs(attrs: List[Tuple[str, Optional[str]]]) -> str:
    if not attrs:
        return ""
    parts: List[str] = []
    for name, val in attrs:
        if val is None:
            parts.append(name)
        else:
            escaped = html.escape(val, quote=True)
            parts.append(f'{name}="{escaped}"')
    return " " + " ".join(parts)


class _ScriptStyleStripper(HTMLParser):
    __slots__ = ("_parts", "_skip_depth")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._parts: List[str] = []
        self._skip_depth = 0

    def _handle_start_like(self, tag: str) -> bool:
        lower = tag.lower()
        if lower in _SCRIPT_STYLE_TAGS:
            self._skip_depth += 1
            return True
        return False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if self._handle_start_like(tag):
            return
        if self._skip_depth:
            return
        cleaned_attrs = _sanitize_attrs(attrs)
        self._parts.append(f"<{tag}{_serialize_attrs(cleaned_attrs)}>")

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() in _SCRIPT_STYLE_TAGS:
            return
        if self._skip_depth:
            return
        cleaned_attrs = _sanitize_attrs(attrs)
        self._parts.append(f"<{tag}{_serialize_attrs(cleaned_attrs)}/>")

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in _SCRIPT_STYLE_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        self._parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def handle_comment(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(f"<!--{data}-->")

    def handle_entityref(self, name: str) -> None:
        if not self._skip_depth:
            self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if not self._skip_depth:
            self._parts.append(f"&#{name};")

    def handle_decl(self, decl: str) -> None:
        if not self._skip_depth:
            self._parts.append(f"<!{decl}>")

    def handle_pi(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(f"<?{data}>")

    def unknown_decl(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(f"<![{data}]>")

    def get_output(self) -> str:
        return "".join(self._parts)


def strip_html_script_and_style(text: str) -> str:
    """
    Remove script/style blocks from HTML content while preserving other markup.
    """
    if not text:
        return ""

    current = text
    prev = None
    while prev != current:
        prev = current
        current = _SCRIPT_STYLE_BLOCK_RE.sub("", current)
    current = _SCRIPT_STYLE_SELF_CLOSE_RE.sub("", current)

    parser = _ScriptStyleStripper()
    try:
        parser.feed(current)
        parser.close()
        cleaned = parser.get_output()
    except Exception:
        cleaned = current

    cleaned = _SCRIPT_STYLE_OPEN_RE.sub("", cleaned)
    cleaned = _SCRIPT_STYLE_CLOSE_RE.sub("", cleaned)
    return cleaned


def strip_svg_style_tags(text: str) -> str:
    """
    Remove CSS <style> blocks from SVG content.
    """
    if not text:
        return ""
    current = text
    prev = None
    while prev != current:
        prev = current
        current = _SVG_STYLE_BLOCK_RE.sub("", current)
    current = _SVG_STYLE_SELF_CLOSE_RE.sub("", current)
    return current


_MARKDOWN_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+?)`(?!`)")
_HTML_TAG_RE = re.compile(r"</?[^>\s]+(?:\s+[^<>]*)?>", re.IGNORECASE)
_RST_CODE_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+code(?:-block)?::", re.IGNORECASE)


def _leading_whitespace_width(text: str) -> int:
    stripped = text.lstrip(" \t")
    return len(text) - len(stripped)


def _markdown_fence_details(stripped_line: str) -> Optional[Tuple[str, int]]:
    if not stripped_line:
        return None
    leader = stripped_line[0]
    if leader not in ("`", "~"):
        return None
    count = 0
    for ch in stripped_line:
        if ch == leader:
            count += 1
        else:
            break
    if count >= 3:
        return leader, count
    return None


def _strip_markdown_fenced_blocks(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: List[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in lines:
        stripped = line.lstrip()
        fence_info = _markdown_fence_details(stripped)
        if not in_fence and fence_info:
            in_fence = True
            fence_char, fence_len = fence_info
            out.append(line)
            continue
        if in_fence:
            if fence_info and fence_info[0] == fence_char and fence_info[1] >= fence_len:
                in_fence = False
                fence_char = ""
                fence_len = 0
                out.append(line)
            # drop everything between fences
            continue
        out.append(line)

    return "".join(out)


def clean_markdown_text(text: str) -> str:
    if not text:
        return ""
    stripped = _strip_markdown_fenced_blocks(text)
    without_tags = _HTML_TAG_RE.sub("", stripped)
    cleaned = _MARKDOWN_INLINE_CODE_RE.sub("``", without_tags)
    return cleaned


def clean_restructuredtext(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines(keepends=True)
    out: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()
        indent = _leading_whitespace_width(line)
        stripped_r = line.rstrip("\r\n")

        is_code_directive = bool(_RST_CODE_DIRECTIVE_RE.match(stripped_r))
        is_literal_block = (
            bool(stripped)
            and stripped.endswith("::")
            and not stripped.endswith(":::")
            and not stripped.lstrip().startswith("..")
        )

        if is_code_directive or is_literal_block:
            out.append(line)
            i += 1

            if is_code_directive:
                while i < n:
                    opt_line = lines[i]
                    opt_stripped = opt_line.strip()
                    opt_indent = _leading_whitespace_width(opt_line)
                    if opt_stripped.startswith(":") and opt_indent > indent:
                        out.append(opt_line)
                        i += 1
                    else:
                        break

            if i < n and lines[i].strip() == "":
                out.append(lines[i])
                i += 1

            block_indent: Optional[int] = None
            while i < n:
                block_line = lines[i]
                block_stripped = block_line.strip()
                current_indent = _leading_whitespace_width(block_line)

                if block_stripped == "":
                    i += 1
                    continue

                if block_indent is None:
                    if current_indent > indent:
                        block_indent = current_indent
                        i += 1
                        continue
                    break

                if current_indent >= block_indent:
                    i += 1
                    continue
                break
            continue

        out.append(line)
        i += 1

    return "".join(out)

# Placeholder for any character outside ASCII range (0–127).
NON_ASCII_PLACEHOLDER = "\u00A4"  # displayed sentinel (¤) for non-ASCII content
_VISIBLE_ASCII_MIN = 0x20
_VISIBLE_ASCII_MAX = 0x7E
_ALLOWED_CTRL = {"\n", "\t"}

def map_text_to_ascii(text: str, placeholder: str = NON_ASCII_PLACEHOLDER) -> str:
    """
    Sanitize ``text`` so that it contains only visible ASCII characters plus
    newline/tab/space. All other bytes collapse to the placeholder (¤), matching
    the historical behaviour across language pipelines.
    """
    if not text:
        return ""
    out_chars: List[str] = []
    for ch in text:
        code = ord(ch)
        if ch == "\r":
            ch = "\n"
            code = 0x0A
        if ch in _ALLOWED_CTRL:
            out_chars.append(ch)
            continue
        if _VISIBLE_ASCII_MIN <= code <= _VISIBLE_ASCII_MAX:
            out_chars.append(ch)
            continue
        if ch == " ":
            out_chars.append(ch)
            continue
        out_chars.append(placeholder)
    return "".join(out_chars)

def clamp_utf8_bytes(text: str, max_bytes: int) -> Tuple[str, bytes]:
    """
    Clamp ``text`` to at most ``max_bytes`` when encoded as UTF-8. Returns the
    possibly shortened text along with the corresponding UTF-8 bytes.
    """
    if not text or max_bytes <= 0:
        return "", b""

    encoded = text.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return text, encoded

    # Trim to the max byte budget and drop any trailing partial code point.
    encoded = encoded[:max_bytes]
    trimmed = encoded.decode("utf-8", errors="ignore")
    if not trimmed:
        return "", b""

    trimmed_bytes = trimmed.encode("utf-8", errors="ignore")
    # Extremely defensive: if re-encoding still exceeds the budget, shave chars.
    while trimmed and len(trimmed_bytes) > max_bytes:
        trimmed = trimmed[:-1]
        trimmed_bytes = trimmed.encode("utf-8", errors="ignore")
    return trimmed, trimmed_bytes

def extract_primary_text(ex: Dict[str, Any]) -> Optional[str]:
    """
    Fetch the best-effort textual payload from a dataset example.
    Falls back through common field names used by different sources.
    """
    if not ex:
        return None
    candidates = ("content", "text", "raw_content", "body", "document", "content_text")
    for key in candidates:
        val = ex.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def describe_exc(exc: BaseException) -> str:
    parts: List[str] = []
    seen: Set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        name = cur.__class__.__name__
        msg = str(cur)
        parts.append(f"{name}: {msg}" if msg else name)
        cur = cur.__cause__ or cur.__context__
    return " -> ".join(parts)


# ============================================================
#                     License filtering
# ============================================================

_ALLOWED_FAMILIES = {"mit", "apache", "bsd", "unlicense", "0bsd", "mit-0"}
_DISALLOWED_KEYWORDS = {
    "gpl", "agpl", "lgpl", "mpl", "epl", "cdla", "cddl", "artistic",
    "cern", "cecill", "affero", "proprietary", "arr", "cc"
}

def _normalize_license_string(s: str) -> str:
    return re.sub(r"[^a-z0-9.+-]+", " ", (s or "").lower())

def _iter_license_fields(example: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    preferred = [
        "max_stars_repo_license",
        "max_forks_repo_license",
        "max_issues_repo_license",
        "max_stars_repo_licenses",
        "max_forks_repo_licenses",
        "max_issues_repo_licenses",
        "licenses",
        "license",
    ]
    seen = set()
    for k in preferred:
        if k in example and k not in seen and example.get(k) is not None:
            seen.add(k)
            yield k, example[k]
    for k, v in example.items():
        if k in seen:
            continue
        if "license" in k.lower() and v is not None:
            yield k, v

def _extract_license_strings(example: Dict[str, Any]) -> List[str]:
    vals: List[str] = []
    for _, v in _iter_license_fields(example):
        if isinstance(v, list):
            vals.extend([str(x) for x in v if x is not None])
        else:
            vals.append(str(v))
    return vals

def license_is_allowed(example: Dict[str, Any]) -> Tuple[bool, str]:
    lic_strings = _extract_license_strings(example)
    if not lic_strings:
        return False, "no_license_info"
    any_allowed = False
    for raw in lic_strings:
        s = _normalize_license_string(raw)
        if any(k in s for k in _DISALLOWED_KEYWORDS):
            return False, "disallowed_license"
        if any(k in s for k in _ALLOWED_FAMILIES):
            any_allowed = True
    if not any_allowed:
        return False, "no_allowed_license"
    return True, ""


# ============================================================
#               PHP foreign content removal (regex)
# ============================================================

_BLADE_DIRECTIVE_RE = re.compile(
    r"(?im)^\s*@(?:extends|section|yield|endsection|include|component|slot|push|stop|parent|stack|csrf|method|error|lang|forelse|empty|endforelse|php|endphp|verbatim|endverbatim)\b"
)
_BLADE_COMMENT_RE = re.compile(r"(?is)\{\{--.*?--\}\}")
_TWIG_TAG_RE = re.compile(r"{%\s*[a-z]", re.IGNORECASE)
_SMARTY_TAG_RE = re.compile(r"{/?\s*(?:if|foreach|section|block|capture|literal|extends|include)\b", re.IGNORECASE)
_DOUBLE_CURLY_RE = re.compile(r"\{\{[^}]+\}\}")

def is_probable_php_template(text: str, *, path: Optional[str] = None) -> bool:
    """
    Heuristic detection for PHP template languages (Blade/Twig/Smarty/etc.).
    Returns True when the file is likely a template instead of executable PHP.
    """
    if not text:
        return False

    path_lower = (path or "").lower()
    if (
        path_lower.endswith(".blade.php")
        or ".blade.php" in path_lower
        or path_lower.endswith(".twig")
        or path_lower.endswith(".tpl")
        or path_lower.endswith(".tpl.php")
        or "/resources/views/" in path_lower
    ):
        return True

    php_tag_present = "<?" in text
    markers = 0

    if _BLADE_DIRECTIVE_RE.search(text):
        markers += 1
    if _BLADE_COMMENT_RE.search(text):
        markers += 1

    moustache_hits = len(_DOUBLE_CURLY_RE.findall(text))
    if moustache_hits >= 2 and not php_tag_present:
        markers += 1
    if moustache_hits >= 1 and _BLADE_DIRECTIVE_RE.search(text):
        markers += 1

    if _TWIG_TAG_RE.search(text) and not php_tag_present:
        markers += 1
    if _SMARTY_TAG_RE.search(text) and not php_tag_present:
        markers += 1

    return markers >= 2

_PHP_BLOCK_RE = re.compile(
    r"(?is)<\?(?!xml)(?:php|=)?(.*?)\?>"
)
_ASP_PHP_BLOCK_RE = re.compile(
    r"(?is)<%(.*?)%>"
)

def extract_php_code_only(text: str, *, path: Optional[str] = None) -> str:
    """
    Keep ONLY PHP code regions, drop all HTML/other template text.
    """
    if not text:
        return ""
    if is_probable_php_template(text, path=path):
        return ""
    parts: List[str] = []
    for m in _PHP_BLOCK_RE.finditer(text):
        inner = m.group(1)
        if inner is not None:
            parts.append(inner)
    if not parts:
        for m in _ASP_PHP_BLOCK_RE.finditer(text):
            inner = m.group(1)
            if inner is not None:
                parts.append(inner)
    sample = text[:4096].lower()
    if not parts:
        if ("$" in sample and ("function" in sample or "class" in sample or "->" in sample or "::" in sample)):
            return text
        return ""
    return "\n".join(parts)


# ============================================================
#               Streaming dataset for The Stack
# ============================================================

LANG_CANDIDATE_DIRS: Dict[str, List[str]] = {
    "php": ["php"],
    "csharp": ["c-sharp", "c#", "csharp"],
    "typescript": ["typescript"],
    "csv": ["csv"],
    "go": ["go"],
    "sql": ["sql"],
    "rust": ["rust"],
    "yaml": ["yaml", "yml"],
    "ruby": ["ruby"],
    "python": ["python"],
    "javascript": ["javascript", "js"],
    "java": ["java"],
    "c_family": ["c", "c++", "cpp"],
    "json": ["json"],
    "css": ["css"],
    "html": ["html", "xhtml"],
    "text": ["text"],
    "shell": ["shell", "bash", "sh", "zsh", "fish"],
    "powershell": ["powershell", "ps1"],
    "batchfile": ["batchfile", "batch", "bat", "cmd"],
    "visual_basic": ["visual-basic", "visualbasic", "vb", "vb.net", "vba"],
    "dockerfile": ["dockerfile", "docker"],
    "dart": ["dart"],
    "gettext_catalog": ["gettext-catalog"],
    "kotlin": ["kotlin"],
    "markdown": ["markdown", "md"],
    "restructuredtext": ["restructuredtext", "rst"],
    "scala": ["scala"],
    "swift": ["swift"],
    "svg": ["svg"],
    "tex": ["latex", "tex"],
    "xml": ["xml"],
    # derived enc/enc are generated locally and do not map to The Stack
}

def stream_madlad_iterable(*, shuffle_buffer: int, use_auth_token: bool) -> Optional[Any]:
    """
    Streaming loader for allenai/MADLAD-400 (clean split) backing the 'text' label.
    """
    token_str, token_arg = resolve_hf_token_pair(use_auth_token)
    try:
        languages = madlad_list_languages(token=token_str)
    except Exception as e:
        sys.stderr.write(f"[warn] failed to enumerate MADLAD-400 languages: {e}\n")
        return None
    if not languages:
        sys.stderr.write("[warn] failed to enumerate MADLAD-400 languages: none found\n")
        return None

    load_kwargs: Dict[str, Any] = {
        "path": "allenai/madlad-400",
        "split": "clean",
        "streaming": True,
        "languages": languages,
    }
    if token_arg is not None:
        load_kwargs["token"] = token_arg

    try:
        ds = load_dataset(**load_kwargs)
    except TypeError:
        load_kwargs.pop("token", None)
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        sys.stderr.write(f"[warn] failed to open MADLAD-400 (clean split): {e}\n")
        return None
    if shuffle_buffer > 0:
        try:
            ds = ds.shuffle(seed=42, buffer_size=shuffle_buffer)
        except Exception as e:
            sys.stderr.write(f"[warn] MADLAD shuffle failed (falling back to sequential order): {e}\n")
    return ds

def try_load_streaming_dir(lang_dir: str, *, shuffle_buffer: int, token: Optional[bool]) -> Optional[Any]:
    load_kwargs = dict(
        path="bigcode/the-stack",
        data_dir=f"data/{lang_dir}",
        split="train",
        streaming=True,
    )
    if token:
        load_kwargs["token"] = token
    try:
        ds = load_dataset(**load_kwargs)
    except TypeError:
        load_kwargs.pop("token", None)
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        sys.stderr.write(f"[warn] skipping dataset dir '{lang_dir}': {e}\n")
        return None
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=42, buffer_size=shuffle_buffer)
    return ds

COMBINED_LABEL_SOURCES: Dict[str, List[str]] = {
    "javascript_typescript": ["javascript", "typescript"],
    "shell": ["shell", "batchfile"],
}

def _stream_single_language_iterable(
    logical_label: str,
    *,
    shuffle_buffer: int,
    use_auth_token: bool,
    shard_count: int,
    shard_index: int,
    skip_first_n: int,
    prefer_stack: bool,
) -> Optional[Any]:
    label = canonical_label(logical_label)
    source_tag = label
    if label == "text":
        ds = stream_madlad_iterable(
            shuffle_buffer=shuffle_buffer,
            use_auth_token=use_auth_token,
        )
        source_tag = "madlad"
    else:
        cands = LANG_CANDIDATE_DIRS.get(label, [label])
        token_val: Optional[bool] = None
        if use_auth_token:
            try:
                from huggingface_hub import get_token  # type: ignore
                token_val = get_token() or True
            except Exception:
                token_val = True

        ds = None
        for d in cands:
            ds = try_load_streaming_dir(d, shuffle_buffer=shuffle_buffer, token=token_val)
            if ds is not None:
                source_tag = d
                break
    if ds is None:
        return None

    if shard_count > 1:
        try:
            ds = ds.shard(num_shards=shard_count, index=shard_index)
        except Exception as e:
            sys.stderr.write(f"[warn] shard() failed for {label}: {e}\n")

    if skip_first_n > 0:
        try:
            ds = ds.skip(skip_first_n)
        except Exception:
            def _drop(it):
                i = 0
                for ex in it:
                    i += 1
                    if i <= skip_first_n:
                        continue
                    yield ex
            ds = _drop(ds)

    def _attach_source() -> Iterator[Dict[str, Any]]:
        for ex in ds:
            if isinstance(ex, dict):
                ex["_stack_src"] = source_tag
            yield ex

    return _attach_source()

def stream_language_iterable(
    logical_label: str,
    *,
    shuffle_buffer: int,
    use_auth_token: bool,
    shard_count: int,
    shard_index: int,
    skip_first_n: int,
    prefer_stack: bool = False,
) -> Optional[Any]:
    label = canonical_label(logical_label)

    if label in COMBINED_LABEL_SOURCES:
        sources = COMBINED_LABEL_SOURCES[label]
        count = len(sources)
        if count == 0:
            return None

        def _distribute_skip(total: int, parts: int) -> List[int]:
            if parts <= 0:
                return []
            base = total // parts
            remainder = total % parts
            return [base + (1 if i < remainder else 0) for i in range(parts)]

        skip_alloc = _distribute_skip(max(0, skip_first_n), count)

        datasets: List[Iterator[Any]] = []
        missing: List[str] = []
        for idx, src in enumerate(sources):
            ds = _stream_single_language_iterable(
                src,
                shuffle_buffer=shuffle_buffer,
                use_auth_token=use_auth_token,
                shard_count=shard_count,
                shard_index=shard_index,
                skip_first_n=skip_alloc[idx] if idx < len(skip_alloc) else 0,
                prefer_stack=prefer_stack,
            )
            if ds is None:
                missing.append(src)
            else:
                datasets.append(iter(ds))

        if missing:
            raise RuntimeError(
                f"Combined label '{label}' is missing source datasets: {', '.join(missing)}"
            )
        if not datasets:
            return None

        def _round_robin():
            active = list(datasets)
            idx = 0
            while active:
                pos = idx % len(active)
                it = active[pos]
                try:
                    yield next(it)
                    idx += 1
                except StopIteration:
                    active.pop(pos)
            return

        return _round_robin()

    return _stream_single_language_iterable(
        label,
        shuffle_buffer=shuffle_buffer,
        use_auth_token=use_auth_token,
        shard_count=shard_count,
        shard_index=shard_index,
        skip_first_n=skip_first_n,
        prefer_stack=prefer_stack,
    )


# ============================================================
#                  Windowing & Magika batching
# ============================================================

def byte_windows(b: bytes, window_bytes: int) -> Iterator[Tuple[int, bytes]]:
    if not b:
        return
    n = len(b)
    step = window_bytes
    idx = 0
    for off in range(0, n, step):
        yield idx, b[off: off + step]
        idx += 1

def stable_uid_for_window(raw: bytes) -> str:
    return hashlib.blake2s(raw, digest_size=16).hexdigest()

# Split configuration: train/val use windowed samples, monitor/test keep raw files.
WINDOW_SPLITS = ("train", "val")
RAW_SPLITS = ("monitor", "test")
ALL_SPLITS = WINDOW_SPLITS + RAW_SPLITS
DEFAULT_RATIOS = {
    "train": 0.70,
    "val": 0.10,
    "monitor": 0.10,
    "test": 0.10,
}
RAW_FILE_MAX_BYTES = 10_000

def split_for_uid(uid: str) -> str:
    # 32-bit digest for quick modulo
    h = hashlib.blake2s(uid.encode("utf-8"), digest_size=4).digest()
    v = int.from_bytes(h, byteorder="big") % 80
    if v < 70:
        return "train"
    else:
        return "val"


def compute_ideal_split_counts(total_cap: int) -> Dict[str, int]:
    total_cap = max(0, int(total_cap))
    counts: Dict[str, int] = {}
    assigned = 0
    for split in ALL_SPLITS:
        ratio = DEFAULT_RATIOS.get(split, 0.0)
        target = int(total_cap * ratio)
        counts[split] = target
        assigned += target
    remainder = total_cap - assigned
    while remainder > 0:
        # distribute leftover counts deterministically in split order
        for split in ALL_SPLITS:
            if remainder <= 0:
                break
            counts[split] += 1
            remainder -= 1
    return counts


def compute_split_shortfall(
    target_counts: Dict[str, int], existing_counts: Dict[str, int], splits: Sequence[str]
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for split in splits:
        target = max(0, target_counts.get(split, 0))
        current = max(0, existing_counts.get(split, 0))
        out[split] = max(0, target - current)
    return out


def _source_identity_for_split(ex: Dict[str, Any], label: str) -> str:
    parts: List[str] = [label or ""]
    have_unique = False
    candidate_keys = (
        "hexsha",
        "id",
        "sha",
        "path",
        "max_stars_repo_path",
        "max_forks_repo_path",
        "max_issues_repo_path",
    )
    for key in candidate_keys:
        val = ex.get(key)
        if val:
            parts.append(str(val))
            have_unique = True
            break
    stack_src = ex.get("_stack_src")
    if stack_src:
        parts.append(str(stack_src))
    if not have_unique:
        raw_text = ex.get("content") or ex.get("text")
        if isinstance(raw_text, str) and raw_text:
            parts.append(raw_text[:256])
    return "|".join(parts)


def route_for_example(ex: Dict[str, Any], label: str) -> str:
    ident = _source_identity_for_split(ex, label)
    h = hashlib.blake2s(ident.encode("utf-8"), digest_size=4).digest()
    v = int.from_bytes(h, byteorder="big") % 100
    if v < int((DEFAULT_RATIOS["train"] + DEFAULT_RATIOS["val"]) * 100):
        return "window"
    elif v < int((DEFAULT_RATIOS["train"] + DEFAULT_RATIOS["val"] + DEFAULT_RATIOS["monitor"]) * 100):
        return "monitor"
    else:
        return "test"


def stack_source_label(ex: Dict[str, Any], fallback: str) -> str:
    src = ex.get("_stack_src")
    if isinstance(src, str) and src:
        return src
    return fallback

@dataclass
class MagikaResult:
    ok: bool
    label: Optional[str]
    mime: Optional[str]
    score: float

class MagikaBatcher:
    def __init__(self) -> None:
        from magika import Magika  # lazy import
        self._m = Magika()

    def identify_many(self, chunks: List[bytes]) -> List[MagikaResult]:
        out: List[MagikaResult] = []
        for ch in chunks:
            try:
                res = self._m.identify_bytes(ch)
                ok = bool(getattr(res, "ok", False))
                if ok:
                    o = getattr(res, "output", None)
                    label = getattr(o, "label", None)
                    mime = getattr(o, "mime_type", None)
                    score = float(getattr(res, "score", 0.0))
                    out.append(MagikaResult(True, label, mime, score))
                else:
                    out.append(MagikaResult(False, None, None, 0.0))
            except Exception:
                out.append(MagikaResult(False, None, None, 0.0))
        return out


# ============================================================
#               Derived data (encodings only)
# ============================================================

ENCODING_METHODS = {"hex", "base64", "base32", "base58", "base85"}

def is_derived_label(lbl: str) -> bool:
    c = canonical_label(lbl)
    if c.startswith("encoding_"):
        return c.split("encoding_", 1)[1] in ENCODING_METHODS
    return False

# Deterministic pseudo-random bytes (PRF) based on base_seed, split, method tag, and index.
def _prf_bytes(base_seed: int, split: str, tag: str, idx: int, n: int) -> bytes:
    out = bytearray()
    ctr = 0
    # domain-separated personalization ensures no collisions across tags/methods/splits
    while len(out) < n:
        h = hashlib.blake2b(
            f"{base_seed}|{split}|{tag}|{idx}|{ctr}".encode("utf-8"),
            digest_size=32,
        ).digest()
        out.extend(h)
        ctr += 1
    return bytes(out[:n])

def _prf_int(base_seed: int, split: str, tag: str, idx: int, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("Modulo must be positive")
    raw = _prf_bytes(base_seed, split, tag, idx, 8)
    return int.from_bytes(raw, "big") % modulo

@dataclass
class PlaintextPool:
    split: str
    datasets: List[Dataset]
    labels: List[str]
    cumulative_counts: List[int]

    @property
    def total(self) -> int:
        return self.cumulative_counts[-1] if self.cumulative_counts else 0

    def locate(self, global_index: int) -> Tuple[int, int]:
        if not (0 <= global_index < self.total):
            raise IndexError(f"Plaintext index {global_index} out of range for split '{self.split}' (total={self.total})")
        pos = bisect_left(self.cumulative_counts, global_index + 1)
        prev_total = self.cumulative_counts[pos - 1] if pos > 0 else 0
        return pos, global_index - prev_total

def build_plaintext_pool(out_root: Path, split: str, exclude_label: str) -> PlaintextPool:
    datasets: List[Dataset] = []
    labels: List[str] = []
    cumulative: List[int] = []
    total = 0
    split_root = out_root / split
    if not split_root.exists():
        return PlaintextPool(split, datasets, labels, cumulative)
    for child in sorted(split_root.iterdir()):
        if not child.is_dir():
            continue
        lbl = canonical_label(child.name)
        if lbl == exclude_label:
            continue
        if lbl.startswith("encoding_"):
            continue
        ds_dir = child / "dataset"
        if not ds_dir.exists():
            continue
        try:
            ds = load_from_disk(str(ds_dir))
        except Exception:
            continue
        length = len(ds)
        if length == 0:
            continue
        datasets.append(ds)
        labels.append(lbl)
        total += length
        cumulative.append(total)
    return PlaintextPool(split, datasets, labels, cumulative)

class PlaintextSampler:
    __slots__ = ("total", "mode", "start", "step", "base_seed", "split", "method", "_tag")

    def __init__(self, *, total: int, need: int, base_seed: int, split: str, method: str) -> None:
        if total <= 0:
            raise ValueError("Plaintext pool is empty")
        self.total = total
        self.base_seed = base_seed
        self.split = split
        self.method = method
        self._tag = f"encode_src::{method}"
        if total > 1 and total >= need:
            start = _prf_int(base_seed, split, f"{method}_start", 0, total)
            step = _prf_int(base_seed, split, f"{method}_step", 0, total)
            step = step or 1
            while math.gcd(step, total) != 1:
                step = (step + 1) % total
                if step == 0:
                    step = 1
            self.mode = "cycle"
            self.start = start
            self.step = step
        else:
            self.mode = "prf"
            self.start = 0
            self.step = 0

    def index_for(self, position: int) -> int:
        if self.mode == "cycle":
            return (self.start + self.step * position) % self.total
        return _prf_int(self.base_seed, self.split, self._tag, position, self.total)

# Base58 (Bitcoin alphabet) encoder (no checksum)
_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
# Cache for maximum raw byte calculations per window size
_BASE58_RAW_LIMIT_CACHE: Dict[int, int] = {}
def b58encode(b: bytes) -> str:
    # Convert big-endian bytes to integer
    n = int.from_bytes(b, "big")
    if n == 0:
        # preserve leading zeros faithfully
        zeros = len(b) - len(b.lstrip(b"\0"))
        return ("1" * zeros) or "1"
    chars = []
    while n > 0:
        n, rem = divmod(n, 58)
        chars.append(chr(_B58_ALPHABET[rem]))
    chars.reverse()
    # Handle leading zeros: each 0x00 → leading '1'
    pad = 0
    for byte in b:
        if byte == 0:
            pad += 1
        else:
            break
    return ("1" * pad) + "".join(chars)


def _max_base58_raw_bytes(window_bytes: int) -> int:
    if window_bytes <= 0:
        return 0
    cached = _BASE58_RAW_LIMIT_CACHE.get(window_bytes)
    if cached is not None:
        return cached
    factor = math.log(256, 58)
    estimate = max(0, int(window_bytes / factor))
    test_len = estimate or 1
    while len(b58encode(b"\xff" * test_len)) > window_bytes and test_len > 0:
        test_len -= 1
    while len(b58encode(b"\xff" * (test_len + 1))) <= window_bytes:
        test_len += 1
    _BASE58_RAW_LIMIT_CACHE[window_bytes] = test_len
    return test_len

def encode_bytes(method: str, raw: bytes) -> str:
    m = method.lower()
    if m == "hex":
        return raw.hex()
    import base64
    if m == "base64":
        return base64.b64encode(raw).decode("ascii")
    if m == "base32":
        return base64.b32encode(raw).decode("ascii")
    if m == "base85":
        return base64.a85encode(raw).decode("ascii")
    if m == "base58":
        return b58encode(raw)
    raise ValueError(f"Unknown encoding method: {method}")


def max_input_bytes_for_encoding(method: str, window_bytes: int) -> int:
    if window_bytes <= 0:
        return 0
    m = method.lower()
    if m == "hex":
        return max(0, window_bytes // 2)
    if m == "base64":
        return max(0, (window_bytes // 4) * 3)
    if m == "base32":
        return max(0, (window_bytes // 8) * 5)
    if m == "base85":
        return max(0, (window_bytes // 5) * 4)
    if m == "base58":
        return _max_base58_raw_bytes(window_bytes)
    raise ValueError(f"Unknown encoding method: {method}")

def gen_transformed_for_label(
    *,
    label_c: str,
    window_bytes: int,
    add_meta: bool,
    progress_mode: str,
    demo: bool,
    seen_uids: Optional[Set[str]],
    budget_per_split: Dict[str, int],
    base_seed: int,
    out_root: Path,
) -> Iterator[dict]:
    is_tty = sys.stderr.isatty()
    show_progress = ((progress_mode == "always") or (progress_mode == "auto" and is_tty))
    total_budget = sum(max(0, b) for b in budget_per_split.values())
    progress_desc = f"{label_c} (derived)"

    pbar = tqdm(total=total_budget if total_budget > 0 else None,
                unit="win",
                desc=progress_desc,
                disable=not show_progress)

    encoding_input_limit: Optional[int] = None
    if label_c.startswith("encoding_"):
        method = label_c.split("encoding_", 1)[1]
        encoding_input_limit = max_input_bytes_for_encoding(method, window_bytes)
    else:
        raise ValueError(f"Unsupported derived label '{label_c}'")

    kept_total = 0
    kept_per_split = {s: 0 for s in WINDOW_SPLITS}
    lang_id = LANG2ID.get(label_c, -1)
    out_root = Path(out_root)
    pool_cache: Dict[str, PlaintextPool] = {}

    for split in WINDOW_SPLITS:
        target = budget_per_split.get(split, 0)
        if target <= 0:
            continue

        pool = pool_cache.get(split)
        if pool is None:
            pool = build_plaintext_pool(out_root, split, label_c)
            pool_cache[split] = pool
        if pool.total == 0:
            raise RuntimeError(
                f"No base windows available in split '{split}' under {out_root} to build '{label_c}'. "
                "Ensure core language labels are generated before derived transforms."
            )

        sampler = PlaintextSampler(total=pool.total, need=target, base_seed=base_seed, split=split, method=method)
        produced = 0
        while produced < target:
            batch_count = min(4096, target - produced)
            positions = list(range(produced, produced + batch_count))
            per_dataset_rows: Dict[int, List[int]] = {}
            per_dataset_positions: Dict[int, List[int]] = {}
            for pos in positions:
                global_idx = sampler.index_for(pos)
                ds_idx, local_idx = pool.locate(global_idx)
                per_dataset_rows.setdefault(ds_idx, []).append(local_idx)
                per_dataset_positions.setdefault(ds_idx, []).append(pos)

            plaintext_by_position: Dict[int, Tuple[bytes, str]] = {}
            for ds_idx, rows in per_dataset_rows.items():
                ds = pool.datasets[ds_idx]
                try:
                    subset = ds[rows]
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to retrieve plaintext rows for split '{split}' (dataset index {ds_idx})"
                    ) from e
                contents = subset["content"]
                base_label = pool.labels[ds_idx] if ds_idx < len(pool.labels) else ""
                for content, pos in zip(contents, per_dataset_positions[ds_idx]):
                    if isinstance(content, str):
                        raw = content.encode("utf-8", errors="ignore")
                    elif isinstance(content, bytes):
                        raw = content
                    elif isinstance(content, bytearray):
                        raw = bytes(content)
                    elif isinstance(content, memoryview):
                        raw = content.tobytes()
                    elif content is None:
                        raw = b""
                    else:
                        raw = str(content).encode("utf-8", errors="ignore")
                    plaintext_by_position[pos] = (raw, base_label)

            for pos in positions:
                raw_plain, base_label = plaintext_by_position.get(pos, (b"", ""))
                raw_for_encoding = raw_plain
                if encoding_input_limit is not None and len(raw_for_encoding) > encoding_input_limit:
                    raw_for_encoding = raw_for_encoding[:encoding_input_limit]
                try:
                    content_str = encode_bytes(method, raw_for_encoding)
                except Exception as e:
                    raise RuntimeError(f"Encoding failed for method '{method}'") from e
                if len(content_str) > window_bytes:
                    trim_len = len(raw_for_encoding)
                    while trim_len > 0 and len(content_str) > window_bytes:
                        trim_len -= 1
                        raw_for_encoding = raw_for_encoding[:trim_len]
                        content_str = encode_bytes(method, raw_for_encoding)

                uid = stable_uid_for_window(content_str.encode("utf-8", errors="ignore"))
                if seen_uids is not None and uid in seen_uids:
                    continue

                ex = {
                    "content": content_str,
                    "lang_id": np.int16(lang_id).item(),
                    "uid": uid,
                    "split": split,
                }
                if add_meta:
                    ex.update({
                        "win_idx": np.int64(0).item(),
                        "source_ext": f"encoding:{method}",
                        "source_hexsha": "",
                        "source_repo": "derived",
                        "source_repo_path": f"{base_label}->encoding:{method}",
                        "license": "derived",
                    })
                if seen_uids is not None:
                    seen_uids.add(uid)
                budget_per_split[split] = max(0, budget_per_split[split] - 1)
                kept_total += 1
                kept_per_split[split] += 1
                pbar.update(1)
                yield ex
            produced += batch_count

    pbar.set_postfix(k_train=kept_per_split["train"],
                     k_val=kept_per_split["val"])
    pbar.close()


def gen_transformed_raw_for_label(
    *,
    label_c: str,
    add_meta: bool,
    progress_mode: str,
    budget_per_split: Dict[str, int],
    seen_uids: Optional[Set[str]],
    base_seed: int,
    out_root: str,
    max_bytes: int,
) -> Iterator[dict]:
    lbl = canonical_label(label_c)
    if not is_derived_label(lbl):
        raise ValueError(f"Derived raw generator expected encoding_* label, got '{label_c}'")

    method = lbl.split("encoding_", 1)[1]
    encoding_input_limit = max_input_bytes_for_encoding(method, max_bytes)
    out_root_path = Path(out_root)

    is_tty = sys.stderr.isatty()
    show_progress = ((progress_mode == "always") or (progress_mode == "auto" and is_tty))
    total_budget = sum(max(0, b) for b in budget_per_split.values())
    pbar = tqdm(
        total=total_budget if total_budget > 0 else None,
        unit="file",
        desc=f"{lbl} (derived raw)",
        disable=not show_progress,
    )

    lang_id = LANG2ID.get(lbl, -1)
    kept_total = 0
    kept_per_split = {s: 0 for s in RAW_SPLITS}
    pool_cache: Dict[str, PlaintextPool] = {}

    def _value_from(seq: Optional[Sequence[Any]], idx: int, default: str = "") -> str:
        if not seq:
            return default
        if idx < 0 or idx >= len(seq):
            return default
        val = seq[idx]
        if val is None:
            return default
        return str(val)

    for split in RAW_SPLITS:
        target = budget_per_split.get(split, 0)
        if target <= 0:
            continue

        pool = pool_cache.get(split)
        if pool is None:
            pool = build_plaintext_pool(out_root_path, split, lbl)
            pool_cache[split] = pool
        if pool.total == 0:
            raise RuntimeError(
                f"No base samples available in split '{split}' under {out_root_path} to build '{lbl}'. "
                "Ensure base language monitor/test datasets exist before derived transforms."
            )

        sampler = PlaintextSampler(total=pool.total, need=target, base_seed=base_seed, split=split, method=method)
        produced = 0
        while produced < target:
            batch_count = min(4096, target - produced)
            positions = list(range(produced, produced + batch_count))
            per_dataset_rows: Dict[int, List[int]] = {}
            per_dataset_positions: Dict[int, List[int]] = {}
            for pos in positions:
                global_idx = sampler.index_for(pos)
                ds_idx, local_idx = pool.locate(global_idx)
                per_dataset_rows.setdefault(ds_idx, []).append(local_idx)
                per_dataset_positions.setdefault(ds_idx, []).append(pos)

            plaintext_by_position: Dict[int, Tuple[bytes, str, Optional[Tuple[str, str, str, str, str]]]] = {}
            for ds_idx, rows in per_dataset_rows.items():
                ds = pool.datasets[ds_idx]
                try:
                    subset = ds[rows]
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to retrieve raw rows for split '{split}' (dataset index {ds_idx})"
                    ) from e

                contents = subset["content"]
                stack_labels = subset.get("stack_label")
                meta_fields: Dict[str, Optional[Sequence[Any]]] = {}
                if add_meta:
                    meta_fields = {
                        "source_ext": subset.get("source_ext"),
                        "source_hexsha": subset.get("source_hexsha"),
                        "source_repo": subset.get("source_repo"),
                        "source_repo_path": subset.get("source_repo_path"),
                        "license": subset.get("license"),
                    }

                for idx, pos in enumerate(per_dataset_positions[ds_idx]):
                    content = contents[idx]
                    if isinstance(content, str):
                        raw = content.encode("utf-8", errors="ignore")
                    elif isinstance(content, bytes):
                        raw = content
                    elif isinstance(content, bytearray):
                        raw = bytes(content)
                    elif isinstance(content, memoryview):
                        raw = content.tobytes()
                    elif content is None:
                        raw = b""
                    else:
                        raw = str(content).encode("utf-8", errors="ignore")

                    base_label = pool.labels[ds_idx] if ds_idx < len(pool.labels) else ""
                    stack_lbl = _value_from(stack_labels, idx, base_label)
                    meta_tuple: Optional[Tuple[str, str, str, str, str]] = None
                    if add_meta:
                        meta_tuple = tuple(
                            _value_from(meta_fields.get(key), idx, "")
                            for key in ("source_ext", "source_hexsha", "source_repo", "source_repo_path", "license")
                        )
                    plaintext_by_position[pos] = (raw, stack_lbl, meta_tuple)

            for pos in positions:
                raw_plain, stack_lbl, meta_tuple = plaintext_by_position.get(pos, (b"", "", None))
                raw_for_encoding = raw_plain
                if encoding_input_limit is not None and len(raw_for_encoding) > encoding_input_limit:
                    raw_for_encoding = raw_for_encoding[:encoding_input_limit]
                try:
                    content_str = encode_bytes(method, raw_for_encoding)
                except Exception as e:
                    raise RuntimeError(f"Encoding failed for method '{method}' (raw split)") from e

                if len(content_str) > max_bytes:
                    trim_len = len(raw_for_encoding)
                    while trim_len > 0 and len(content_str) > max_bytes:
                        trim_len -= 1
                        raw_for_encoding = raw_for_encoding[:trim_len]
                        content_str = encode_bytes(method, raw_for_encoding)
                if not content_str:
                    continue

                uid = stable_uid_for_window(content_str.encode("utf-8", errors="ignore"))
                if seen_uids is not None and uid in seen_uids:
                    continue

                payload = {
                    "content": content_str,
                    "lang_id": np.int16(lang_id).item(),
                    "uid": uid,
                    "split": split,
                    "stack_label": stack_lbl or lbl,
                }
                if add_meta:
                    source_ext, source_hexsha, source_repo, source_repo_path, license_val = meta_tuple or ("", "", "", "", "")
                    payload.update({
                        "win_idx": np.int64(0).item(),
                        "source_ext": source_ext or f"encoding:{method}",
                        "source_hexsha": source_hexsha or "",
                        "source_repo": source_repo or "derived",
                        "source_repo_path": source_repo_path or f"derived::{method}",
                        "license": license_val or "derived",
                    })

                if seen_uids is not None:
                    seen_uids.add(uid)
                budget_per_split[split] = max(0, budget_per_split[split] - 1)
                kept_total += 1
                kept_per_split[split] += 1
                pbar.update(1)
                yield payload
            produced += batch_count

    pbar.set_postfix(
        monitor=kept_per_split.get("monitor", 0),
        test=kept_per_split.get("test", 0),
    )
    pbar.close()


# ============================================================
#                 Generator: one pass per label
# ============================================================

def gen_windows_for_label(
    *,
    logical_label: str,
    window_bytes: int,
    magika_batch: int,
    threshold: float,
    add_meta: bool,
    shuffle_buffer: int,
    use_auth_token: bool,
    shard_count: int,
    shard_index: int,
    skip_first_n: int,
    progress_mode: str,
    demo: bool,
    seen_uids: Optional[Set[str]] = None,
    budget_per_split: Dict[str, int],
    base_seed: int,
    out_root: str,
) -> Iterator[dict]:
    label_c = canonical_label(logical_label)

    # Derived families: generate locally using existing plaintext (no Magika or The Stack streaming)
    if is_derived_label(label_c):
        yield from gen_transformed_for_label(
            label_c=label_c,
            window_bytes=window_bytes,
            add_meta=add_meta,
            progress_mode=progress_mode,
            demo=demo,
            seen_uids=seen_uids,
            budget_per_split=budget_per_split,
            base_seed=base_seed,
            out_root=out_root,
        )
        return

    if label_c == "text":
        yield from gen_text_windows_from_madlad(
            window_bytes=window_bytes,
            add_meta=add_meta,
            progress_mode=progress_mode,
            budget_per_split=budget_per_split,
            seen_uids=seen_uids,
            use_auth_token=use_auth_token,
            base_seed=base_seed,
        )
        return

    # Real languages: stream from The Stack and filter via Magika
    is_tty = sys.stderr.isatty()
    show_progress = ((progress_mode == "always") or (progress_mode == "auto" and is_tty))
    total_budget = sum(max(0, b) for b in budget_per_split.values())
    pbar = tqdm(total=total_budget if total_budget > 0 else None,
                unit="win",
                desc=f"{logical_label}",
                disable=not show_progress)

    ds = stream_language_iterable(
        logical_label,
        shuffle_buffer=shuffle_buffer,
        use_auth_token=use_auth_token,
        shard_count=shard_count,
        shard_index=shard_index,
        skip_first_n=skip_first_n,
    )
    if ds is None:
        pbar.close()
        raise RuntimeError(f"Could not open streaming dataset for '{logical_label}'")

    batcher = MagikaBatcher()

    kept_total = 0
    rejected = 0
    kept_per_split = {"train": 0, "val": 0, "test": 0}

    lang_id = LANG2ID.get(canonical_label(logical_label), -1)

    buf_bytes: List[bytes] = []
    buf_meta: List[Tuple[int, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], str]] = []
    buf_uids: List[str] = []
    buf_ascii_texts: List[str] = []

    def _flush_batch():
        nonlocal kept_total, rejected
        if not buf_bytes:
            return
        results = batcher.identify_many(buf_bytes)
        for (res, meta, raw, uid, ascii_text) in zip(results, buf_meta, buf_bytes, buf_uids, buf_ascii_texts):
            if kept_total >= total_budget and total_budget > 0:
                break
            win_idx, ext, hexsha, repo_name, repo_path, license_str, split = meta
            if budget_per_split.get(split, 0) <= 0:
                rejected += 1
                continue
            if res.ok and res.score >= threshold and label_matches_target(logical_label, res.label, res.mime):
                payload = {
                    "content": ascii_text,
                    "lang_id": np.int16(lang_id).item(),
                    "uid": uid,
                    "split": split,
                }
                if add_meta:
                    payload.update({
                        "win_idx": np.int64(win_idx).item(),
                        "source_ext": str(ext or ""),
                        "source_hexsha": str(hexsha or ""),
                        "source_repo": str(repo_name or ""),
                        "source_repo_path": str(repo_path or ""),
                        "license": str(license_str or ""),
                    })
                if seen_uids is not None:
                    seen_uids.add(uid)
                budget_per_split[split] = max(0, budget_per_split[split] - 1)
                kept_per_split[split] += 1
                kept_total += 1
                yield payload
                pbar.update(1)
            else:
                rejected += 1
        buf_bytes.clear()
        buf_meta.clear()
        buf_uids.clear()
        buf_ascii_texts.clear()

    last_postfix = time.time()
    for ex in ds:
        if kept_total >= total_budget and total_budget > 0:
            break

        route = route_for_example(ex, label_c)
        if route in RAW_SPLITS:
            continue

        raw_text = extract_primary_text(ex)
        if not raw_text:
            continue

        ext = ex.get("ext")
        hexsha = ex.get("hexsha")
        repo_name = ex.get("max_stars_repo_name") or ex.get("repo_name")
        repo_path_raw = ex.get("max_stars_repo_path") or ex.get("path")
        if repo_path_raw is None:
            repo_path = None
        elif isinstance(repo_path_raw, str):
            repo_path = repo_path_raw
        else:
            repo_path = str(repo_path_raw)
        license_str = ex.get("max_stars_repo_license") or ex.get("license")

        lbl_canon = canonical_label(logical_label)
        if lbl_canon == "php":
            content_filtered = extract_php_code_only(raw_text, path=repo_path)
            if not content_filtered:
                continue
        elif lbl_canon == "html":
            if _FRAMEWORK_KEYWORDS_RE.search(raw_text):
                continue
            content_filtered = strip_html_script_and_style(raw_text)
            if _FRAMEWORK_KEYWORDS_RE.search(content_filtered):
                continue
            if not content_filtered.strip():
                continue
        elif lbl_canon == "markdown":
            content_filtered = clean_markdown_text(raw_text)
            if not content_filtered.strip():
                continue
        elif lbl_canon == "restructuredtext":
            content_filtered = clean_restructuredtext(raw_text)
            if not content_filtered.strip():
                continue
        elif lbl_canon == "svg":
            # For training/validation windows, strip embedded CSS style blocks
            # from SVG content before windowing. Raw monitor/test files are
            # produced via gen_raw_files_for_label and remain untouched.
            content_filtered = strip_svg_style_tags(raw_text)
            if not content_filtered.strip():
                continue
        else:
            content_filtered = raw_text

        if lbl_canon == "text":
            ok_lic = True
        else:
            ok_lic, _ = license_is_allowed(ex)
        if not ok_lic:
            continue

        try:
            b = content_filtered.encode("utf-8", errors="ignore")
        except Exception:
            continue
        if not b:
            continue

        for widx, wbytes in byte_windows(b, window_bytes):
            if kept_total >= total_budget and total_budget > 0:
                break
            window_text = wbytes.decode("utf-8", errors="ignore")
            if not window_text:
                rejected += 1
                continue

            ascii_text = map_text_to_ascii(window_text)
            if not ascii_text:
                rejected += 1
                continue

            ascii_text, ascii_bytes = clamp_utf8_bytes(ascii_text, window_bytes)
            if not ascii_text or not ascii_bytes:
                rejected += 1
                continue

            uid = stable_uid_for_window(ascii_bytes)

            if seen_uids is not None and uid in seen_uids:
                rejected += 1
                continue

            split = split_for_uid(uid)
            if budget_per_split.get(split, 0) <= 0:
                rejected += 1
                continue

            buf_bytes.append(wbytes)
            buf_meta.append((widx, ext, hexsha, repo_name, repo_path, license_str, split))
            buf_uids.append(uid)
            buf_ascii_texts.append(ascii_text)

            if len(buf_bytes) >= magika_batch:
                for out in _flush_batch():
                    yield out
                if kept_total >= total_budget and total_budget > 0:
                    break

        now = time.time()
        if (now - last_postfix) >= 0.3:
            pbar.set_postfix(kept_total=kept_total, rej=rejected,
                             k_train=kept_per_split["train"],
                             k_val=kept_per_split["val"])
            last_postfix = now

        raw_text = ""
        del b
        gc.collect()

    for out in _flush_batch():
        yield out
    pbar.set_postfix(kept_total=kept_total, rej=rejected,
                     k_train=kept_per_split["train"],
                     k_val=kept_per_split["val"])
    pbar.close()


def gen_raw_files_for_label(
    *,
    logical_label: str,
    add_meta: bool,
    shuffle_buffer: int,
    use_auth_token: bool,
    shard_count: int,
    shard_index: int,
    skip_first_n: int,
    progress_mode: str,
    budget_per_split: Dict[str, int],
    seen_uids: Optional[Set[str]],
    max_bytes: int,
) -> Iterator[dict]:
    label_c = canonical_label(logical_label)
    if is_derived_label(label_c):
        return

    is_tty = sys.stderr.isatty()
    show_progress = ((progress_mode == "always") or (progress_mode == "auto" and is_tty))
    total_budget = sum(max(0, b) for b in budget_per_split.values())
    pbar = tqdm(
        total=total_budget if total_budget > 0 else None,
        unit="file",
        desc=f"{logical_label} (raw)",
        disable=not show_progress,
    )

    ds = stream_language_iterable(
        logical_label,
        shuffle_buffer=shuffle_buffer,
        use_auth_token=use_auth_token,
        shard_count=shard_count,
        shard_index=shard_index,
        skip_first_n=skip_first_n,
        prefer_stack=True,
    )
    if ds is None:
        pbar.close()
        raise RuntimeError(f"Could not open streaming dataset for '{logical_label}' (raw mode)")

    kept_total = 0
    kept_per_split = {s: 0 for s in RAW_SPLITS}
    rejected = 0
    lang_id = LANG2ID.get(label_c, -1)
    last_postfix = time.time()

    for ex in ds:
        if total_budget > 0 and kept_total >= total_budget:
            break

        split = route_for_example(ex, label_c)
        if split not in RAW_SPLITS:
            continue
        if budget_per_split.get(split, 0) <= 0:
            continue

        if label_c == "text":
            ok = True
        else:
            ok, _reason = license_is_allowed(ex)
        if not ok:
            rejected += 1
            continue

        raw_text = extract_primary_text(ex)
        if not raw_text:
            rejected += 1
            continue

        trimmed_text, trimmed_bytes = clamp_utf8_bytes(raw_text, max_bytes)
        if not trimmed_bytes:
            rejected += 1
            continue

        uid_material = trimmed_bytes + split.encode("utf-8")
        uid = stable_uid_for_window(uid_material)
        if seen_uids is not None and uid in seen_uids:
            rejected += 1
            continue

        ext = ex.get("ext")
        hexsha = ex.get("hexsha")
        repo_name = ex.get("max_stars_repo_name") or ex.get("repo_name")
        repo_path_raw = ex.get("max_stars_repo_path") or ex.get("path")
        if repo_path_raw is None:
            repo_path = None
        elif isinstance(repo_path_raw, str):
            repo_path = repo_path_raw
        else:
            repo_path = str(repo_path_raw)
        license_str = ex.get("max_stars_repo_license") or ex.get("license")

        payload = {
            "content": trimmed_text,
            "lang_id": np.int16(lang_id).item(),
            "uid": uid,
            "split": split,
            "stack_label": stack_source_label(ex, label_c),
        }
        if add_meta:
            payload.update({
                "win_idx": np.int64(0).item(),
                "source_ext": str(ext or ""),
                "source_hexsha": str(hexsha or ""),
                "source_repo": str(repo_name or ""),
                "source_repo_path": str(repo_path or ""),
                "license": str(license_str or ""),
            })

        if seen_uids is not None:
            seen_uids.add(uid)
        budget_per_split[split] = max(0, budget_per_split[split] - 1)
        kept_per_split[split] += 1
        kept_total += 1
        pbar.update(1)
        yield payload

        now = time.time()
        if (now - last_postfix) >= 0.3:
            pbar.set_postfix(
                kept_total=kept_total,
                rej=rejected,
                monitor=kept_per_split["monitor"],
                test=kept_per_split["test"],
            )
            last_postfix = now

    pbar.set_postfix(
        kept_total=kept_total,
        rej=rejected,
        monitor=kept_per_split["monitor"],
        test=kept_per_split["test"],
    )
    pbar.close()


# ============================================================
#    Building (train/val windows + monitor/test raw per label)
# ============================================================

LANG2ID: Dict[str, int] = {
    # real languages
    "php": 0,
    "csharp": 1,
    "javascript_typescript": 2,
    "go": 3,
    "sql": 4,
    "rust": 5,
    "yaml": 6,
    "ruby": 7,
    "python": 8,
    "java": 9,
    "c_family": 10,
    "json": 11,
    "css": 12,
    "html": 13,
    "text": 14,
    "csv": 15,
    "shell": 16,
    "powershell": 17,
    "visual_basic": 18,
    "dockerfile": 19,
    "dart": 20,
    "gettext_catalog": 21,
    "kotlin": 22,
    "markdown": 23,
    "restructuredtext": 24,
    "scala": 25,
    "svg": 26,
    "swift": 27,
    "tex": 28,
    "xml": 29,
    # derived encodings
    "encoding_hex": 100,
    "encoding_base64": 101,
    "encoding_base32": 102,
    "encoding_base58": 103,
    "encoding_base85": 104,
}

def ensure_recovery_dirs(out_dir: Path) -> None:
    backup_dir = out_dir.with_name(out_dir.name + ".bak")
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp_write")
    if backup_dir.exists() and not out_dir.exists():
        os.replace(backup_dir, out_dir)
    if backup_dir.exists() and out_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

def collect_existing_uids(out_dir: Path) -> Tuple[Optional[Dataset], Set[str], bool]:
    seen: Set[str] = set()
    if not out_dir.exists():
        return None, seen, False
    ds = load_from_disk(str(out_dir))
    has_uid = "uid" in ds.column_names
    if has_uid:
        for uid in ds["uid"]:
            seen.add(uid)
    else:
        for chunk in ds.iter(10000):
            for s in chunk["content"]:
                uid = stable_uid_for_window(s.encode("utf-8", errors="ignore"))
                seen.add(uid)
    return ds, seen, has_uid

def maybe_add_uid_and_resave(ds: Dataset, out_dir: Path) -> Dataset:
    if "uid" in ds.column_names:
        return ds
    def _mk_uid(batch):
        return {"uid": [stable_uid_for_window(x.encode("utf-8", errors="ignore")) for x in batch["content"]]}
    ds2 = ds.map(_mk_uid, batched=True, batch_size=8192)
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp_uid")
    backup_dir = out_dir.with_name(out_dir.name + ".bak_uid")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    ds2.save_to_disk(str(tmp_dir))
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    if out_dir.exists():
        os.replace(str(out_dir), str(backup_dir))
    os.replace(str(tmp_dir), str(out_dir))
    shutil.rmtree(str(backup_dir), ignore_errors=True)
    return load_from_disk(str(out_dir))

def atomic_replace_dir(src_tmp: Path, dst: Path) -> None:
    backup_dir = dst.with_name(dst.name + ".bak")
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    if dst.exists():
        os.replace(str(dst), str(backup_dir))
    os.replace(str(src_tmp), str(dst))
    shutil.rmtree(str(backup_dir), ignore_errors=True)

def dir_for_label_split(out_root: Path, label: str, split: str) -> Path:
    """
    New layout: <out-root>/<split>/<label>/dataset
    """
    return out_root / split / canonical_label(label) / "dataset"

def load_existing_splits(
    out_root: Path,
    label: str,
    splits: Sequence[str],
    rebuild: bool,
) -> Tuple[Dict[str, Optional[Dataset]], Dict[str, int], Set[str]]:
    existing_ds: Dict[str, Optional[Dataset]] = {}
    existing_counts: Dict[str, int] = {}
    seen_union: Set[str] = set()

    for split in splits:
        out_dir = dir_for_label_split(out_root, label, split)
        ensure_recovery_dirs(out_dir)
        if rebuild and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        ds, seen, has_uid = collect_existing_uids(out_dir)
        if ds is not None and not has_uid:
            ds = maybe_add_uid_and_resave(ds, out_dir)
            ds, seen, _ = collect_existing_uids(out_dir)

        existing_ds[split] = ds
        existing_counts[split] = len(ds) if ds is not None else 0
        seen_union.update(seen)

    return existing_ds, existing_counts, seen_union

def build_arrow_for_label_with_splits(
    *,
    label: str,
    out_root: Path,
    window_bytes: int,
    magika_batch: int,
    threshold: float,
    total_cap: int,
    target_counts: Dict[str, int],
    add_meta: bool,
    shuffle_buffer: int,
    use_auth_token: bool,
    shard_count: int,
    shard_index: int,
    skip_first_n: int,
    progress_mode: str,
    demo: bool,
    writer_batch_size: int,
    rebuild: bool,
    base_seed: int,
) -> Tuple[Dict[str, int], Dict[str, Path]]:
    label_c = canonical_label(label)
    if label_c not in LANG2ID:
        raise ValueError(f"Unknown/unsupported label '{label}'")

    existing_ds, existing_counts, seen_uids = load_existing_splits(
        out_root, label_c, WINDOW_SPLITS, rebuild
    )

    cap_value = 100 if demo else total_cap
    total_target = sum(target_counts.get(s, 0) for s in WINDOW_SPLITS)
    total_existing = sum(existing_counts.values())

    if cap_value > 0 and total_existing >= total_target and total_target > 0:
        out_dirs = {s: dir_for_label_split(out_root, label_c, s) for s in WINDOW_SPLITS}
        return existing_counts, out_dirs

    if cap_value == 0:
        per_split_budget = {s: 2**63 - 1 for s in WINDOW_SPLITS}
    else:
        per_split_budget = compute_split_shortfall(target_counts, existing_counts, WINDOW_SPLITS)

    features = {
        "content": Value("string"),
        "lang_id": Value("int16"),
        "uid": Value("string"),
        "split": Value("string"),
    }
    if add_meta:
        features.update({
            "win_idx": Value("int64"),
            "source_ext": Value("string"),
            "source_hexsha": Value("string"),
            "source_repo": Value("string"),
            "source_repo_path": Value("string"),
            "license": Value("string"),
        })
    feats = Features(features)

    gen_kwargs = dict(
        logical_label=label_c,
        window_bytes=window_bytes,
        magika_batch=magika_batch,
        threshold=threshold,
        add_meta=add_meta,
        shuffle_buffer=shuffle_buffer,
        use_auth_token=use_auth_token,
        shard_count=shard_count,
        shard_index=shard_index,
        skip_first_n=skip_first_n,
        progress_mode=progress_mode,
        demo=demo,
        seen_uids=seen_uids,
        budget_per_split=per_split_budget,
        base_seed=base_seed,
        out_root=str(out_root),
    )

    # Ensure Hugging Face datasets cache stays within the writable workspace
    local_cache_dir = out_root / ".hf_cache"
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds_new_total = Dataset.from_generator(
            gen_windows_for_label,
            gen_kwargs=gen_kwargs,
            features=feats,
            cache_dir=str(local_cache_dir),
            keep_in_memory=False,
            writer_batch_size=writer_batch_size,
        )
    except DatasetGenerationError as e:
        root = e.__cause__ or e.__context__
        detail = describe_exc(root) if root else describe_exc(e)
        raise RuntimeError(
            f"Dataset generation failed for label '{label_c}': {detail}"
        ) from (root if root else e)
    except Exception as e:
        raise RuntimeError(
            f"Failed to build dataset for label '{label_c}': {describe_exc(e)}"
        ) from e

    new_by_split: Dict[str, Optional[Dataset]] = {}
    for s in WINDOW_SPLITS:
        if len(ds_new_total) > 0:
            part = ds_new_total.filter(lambda ex, _s=None: ex["split"] == _s, fn_kwargs={"_s": s})
            if "split" in part.column_names:
                part = part.remove_columns(["split"])
        else:
            part = None
        new_by_split[s] = part

    kept_after_run: Dict[str, int] = dict(existing_counts)
    out_dirs: Dict[str, Path] = {}

    for s in WINDOW_SPLITS:
        out_dir = dir_for_label_split(out_root, label_c, s)
        ensure_recovery_dirs(out_dir)
        out_dirs[s] = out_dir

        ds_new = new_by_split[s]
        ds_exist = existing_ds[s]

        if ds_new is None or len(ds_new) == 0:
            continue

        if ds_exist is not None:
            full = concatenate_datasets([ds_exist, ds_new])
        else:
            full = ds_new

        tmp_dir = out_dir.with_name(out_dir.name + ".tmp_write")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.parent.mkdir(parents=True, exist_ok=True)
        full.save_to_disk(str(tmp_dir))
        atomic_replace_dir(tmp_dir, out_dir)

        kept_after_run[s] = (existing_counts.get(s, 0) + len(ds_new))

    return kept_after_run, out_dirs


def build_raw_monitor_test_for_label(
    *,
    label: str,
    out_root: Path,
    total_cap: int,
    target_counts: Dict[str, int],
    add_meta: bool,
    shuffle_buffer: int,
    use_auth_token: bool,
    shard_count: int,
    shard_index: int,
    skip_first_n: int,
    progress_mode: str,
    demo: bool,
    writer_batch_size: int,
    rebuild: bool,
    base_seed: int,
) -> Tuple[Dict[str, int], Dict[str, Path]]:
    label_c = canonical_label(label)

    existing_ds, existing_counts, seen_uids = load_existing_splits(
        out_root, label_c, RAW_SPLITS, rebuild
    )

    cap_value = 100 if demo else total_cap
    total_target = sum(target_counts.get(s, 0) for s in RAW_SPLITS)
    total_existing = sum(existing_counts.values())

    if total_target == 0 and cap_value > 0:
        out_dirs = {s: dir_for_label_split(out_root, label_c, s) for s in RAW_SPLITS}
        return existing_counts, out_dirs

    if cap_value > 0 and total_existing >= total_target and total_target > 0:
        out_dirs = {s: dir_for_label_split(out_root, label_c, s) for s in RAW_SPLITS}
        return existing_counts, out_dirs

    if cap_value == 0:
        per_split_budget = {s: 2**63 - 1 for s in RAW_SPLITS}
    else:
        per_split_budget = compute_split_shortfall(target_counts, existing_counts, RAW_SPLITS)

    features = {
        "content": Value("string"),
        "lang_id": Value("int16"),
        "uid": Value("string"),
        "split": Value("string"),
        "stack_label": Value("string"),
    }
    if add_meta:
        features.update({
            "win_idx": Value("int64"),
            "source_ext": Value("string"),
            "source_hexsha": Value("string"),
            "source_repo": Value("string"),
            "source_repo_path": Value("string"),
            "license": Value("string"),
        })
    feats = Features(features)

    local_cache_dir = out_root / ".hf_cache"
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        if is_derived_label(label_c):
            ds_new_total = Dataset.from_generator(
                gen_transformed_raw_for_label,
                gen_kwargs=dict(
                    label_c=label_c,
                    add_meta=add_meta,
                    progress_mode=progress_mode,
                    budget_per_split=dict(per_split_budget),
                    seen_uids=seen_uids,
                    base_seed=base_seed,
                    out_root=str(out_root),
                    max_bytes=RAW_FILE_MAX_BYTES,
                ),
                features=feats,
                cache_dir=str(local_cache_dir),
                keep_in_memory=False,
                writer_batch_size=writer_batch_size,
            )
        else:
            ds_new_total = Dataset.from_generator(
                gen_raw_files_for_label,
                gen_kwargs=dict(
                    logical_label=label_c,
                    add_meta=add_meta,
                    shuffle_buffer=shuffle_buffer,
                    use_auth_token=use_auth_token,
                    shard_count=shard_count,
                    shard_index=shard_index,
                    skip_first_n=skip_first_n,
                    progress_mode=progress_mode,
                    budget_per_split=dict(per_split_budget),
                    seen_uids=seen_uids,
                    max_bytes=RAW_FILE_MAX_BYTES,
                ),
                features=feats,
                cache_dir=str(local_cache_dir),
                keep_in_memory=False,
                writer_batch_size=writer_batch_size,
            )
    except DatasetGenerationError as e:
        root = e.__cause__ or e.__context__
        detail = describe_exc(root) if root else describe_exc(e)
        raise RuntimeError(
            f"Raw dataset generation failed for label '{label_c}': {detail}"
        ) from (root if root else e)
    except Exception as e:
        raise RuntimeError(
            f"Failed to build raw dataset for label '{label_c}': {describe_exc(e)}"
        ) from e

    new_by_split: Dict[str, Optional[Dataset]] = {}
    for s in RAW_SPLITS:
        if len(ds_new_total) > 0:
            part = ds_new_total.filter(lambda ex, _s=None: ex["split"] == _s, fn_kwargs={"_s": s})
            if "split" in part.column_names:
                part = part.remove_columns(["split"])
        else:
            part = None
        new_by_split[s] = part

    kept_after_run: Dict[str, int] = dict(existing_counts)
    out_dirs: Dict[str, Path] = {}

    for s in RAW_SPLITS:
        out_dir = dir_for_label_split(out_root, label_c, s)
        ensure_recovery_dirs(out_dir)
        out_dirs[s] = out_dir

        ds_new = new_by_split[s]
        ds_exist = existing_ds[s]

        if ds_new is None or len(ds_new) == 0:
            continue

        if ds_exist is not None:
            full = concatenate_datasets([ds_exist, ds_new])
        else:
            full = ds_new

        tmp_dir = out_dir.with_name(out_dir.name + ".tmp_write")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.parent.mkdir(parents=True, exist_ok=True)
        full.save_to_disk(str(tmp_dir))
        atomic_replace_dir(tmp_dir, out_dir)

        kept_after_run[s] = (existing_counts.get(s, 0) + len(ds_new))

    return kept_after_run, out_dirs


# ============================================================
#                           CLI
# ============================================================

def parse_skip_map(s: Optional[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not s:
        return out
    for item in s.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = canonical_label(k.strip())
        if k in {"javascript", "typescript"}:
            k = "javascript_typescript"
        try:
            out[k] = int(v.strip())
        except Exception:
            pass
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Stream The Stack and build Arrow datasets of windowed code per label with train/val + raw monitor/test splits.\n"
            "• Writes to <out-root>/<split>/<label>/dataset\n"
            "• Filters licenses (permissive only)\n"
            "• PHP: strips foreign (HTML/etc.) content via regex\n"
            "• HTML: strips <script>/<style> blocks, event handlers, and drops framework-heavy samples\n"
            "• In-process Magika prefilter (no temp files)\n"
            "• Resume-safe, duplicate-proof with per-window uid and deterministic splits (70/10 train/val + 10/10 raw monitor/test)\n"
            "• Derived families: encoding_* reuse existing splits and apply deterministic transforms"
        )
    )
    # Core IO
    ap.add_argument("--out-root", type=Path, default=Path("arrow_out"),
                    help="Output root for Arrow datasets (<out-root>/<split>/<label>/dataset)")
    # NEW default labels per request
    ap.add_argument("--langs", type=str,
                    default=(
                        "html,css,javascript_typescript,c_family,csharp,go,rust,csv,java,json,python,ruby,text,"
                        "shell,powershell,visual_basic,php,sql,yaml,dockerfile,dart,gettext_catalog,kotlin,"
                        "markdown,restructuredtext,scala,swift,svg,tex,xml,"
                        "encoding_hex,encoding_base64,encoding_base32,encoding_base58,encoding_base85"
                    ),
                    help="Comma-separated labels to process (logical names).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete existing output directories for selected labels (per split) BEFORE writing")
    ap.add_argument("--add-meta", action="store_true",
                    help="Include metadata columns (win_idx, ext, repo, license, etc.)")

    # Windowing / filtering
    ap.add_argument("--window-bytes", type=int, default=1536,
                    help="Window size in BYTES (default: 1536)")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="Magika confidence threshold (keep if score >= threshold)")
    ap.add_argument("--magika-batch", type=int, default=1024,
                    help="How many windows to run through Magika per batch")
    ap.add_argument("--max-windows-per-label", type=int, default=100_000,
                    help="TOTAL cap of kept samples per label (train/val windows + monitor/test raw). Default: 100,000")
    ap.add_argument("--writer-batch-size", type=int, default=8192,
                    help="HF writer batch size to Arrow (bigger = fewer flushes)")

    # Streaming controls
    ap.add_argument("--shuffle-buffer", type=int, default=0,
                    help="Streaming shuffle buffer size (0 disables; large may use a lot of RAM)")
    ap.add_argument("--use-auth-token", action="store_true",
                    help="Pass cached HF auth token if required")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="Shard the stream into N parts to jump ahead without scanning from the start")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="Which shard index to read (0..shard-count-1)")
    ap.add_argument("--skip-per-lang", type=str, default=None,
                    help='Optional extra skip per label, e.g. "php=500000,css=200000"')

    # UX
    ap.add_argument("--progress", choices=["auto", "always", "never"], default="auto",
                    help="Show live progress bars (default: auto)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (also used for derived transforms)")
    ap.add_argument("--demo", action="store_true",
                    help="Demo mode: keep at most 100 windows per label TOTAL (overrides --max-windows-per-label)")
    return ap.parse_args()


# ============================================================
#                           Main
# ============================================================

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    raw_labels = [canonical_label(s) for s in args.langs.split(",") if s.strip()]
    labels: List[str] = []
    seen_labels: Set[str] = set()
    for label in raw_labels:
        if label not in seen_labels:
            labels.append(label)
            seen_labels.add(label)
    if "javascript" in seen_labels or "typescript" in seen_labels:
        labels = [lbl for lbl in labels if lbl not in {"javascript", "typescript"}]
        seen_labels.discard("javascript")
        seen_labels.discard("typescript")
        if "javascript_typescript" not in seen_labels:
            labels.append("javascript_typescript")
            seen_labels.add("javascript_typescript")
    if not labels:
        raise SystemExit("[fatal] No labels provided via --langs")

    out_root: Path = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    max_per_label_total = 100 if args.demo else args.max_windows_per_label  # keep name consistent
    # fix potential split line artifacts
    max_per_label_total = 100 if args.demo else args.max_windows_per_label

    skip_map = parse_skip_map(args.skip_per_lang)

    print("⚙️  The Stack → Arrow (windowed + raw monitor/test, deterministic 70/10/10/10 splits)")
    print(f"   Out root:        {out_root}  (layout: <split>/<label>/dataset)")
    print(f"   Labels:          {', '.join(labels)}")
    print(f"   Window bytes:    {args.window_bytes}")
    print(f"   Magika batch:    {args.magika_batch}")
    print(f"   Threshold:       {args.threshold:.2f}")
    print(f"   Max/label:       {max_per_label_total} {'(DEMO)' if args.demo else ''}  [70% train, 10% val, 10% monitor, 10% test]")
    print(f"   Shuffle buffer:  {args.shuffle_buffer}")
    print(f"   Shard:           count={args.shard_count}, index={args.shard_index}")
    if skip_map:
        print(f"   Extra skips:     " + ", ".join(f"{k}={v}" for k, v in skip_map.items()))
    print(f"   Add meta:        {'yes' if args.add_meta else 'no'}")
    print(f"   Rebuild:         {'yes' if args.rebuild else 'no'}")
    print("")

    grand_kept_per_split = {s: 0 for s in ALL_SPLITS}
    failures: List[str] = []

    t0 = time.time()
    for lbl in labels:
        print(f"— Building label: {lbl}")
        try:
            target_counts = compute_ideal_split_counts(max_per_label_total)
            skip_extra = int(skip_map.get(canonical_label(lbl), 0))
            kept_map, out_dirs = build_arrow_for_label_with_splits(
                label=lbl,
                out_root=out_root,
                window_bytes=args.window_bytes,
                magika_batch=args.magika_batch,
                threshold=args.threshold,
                total_cap=max_per_label_total,
                target_counts=target_counts,
                add_meta=args.add_meta,
                shuffle_buffer=args.shuffle_buffer,
                use_auth_token=args.use_auth_token,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                skip_first_n=skip_extra,
                progress_mode=args.progress,
                demo=args.demo,
                writer_batch_size=args.writer_batch_size,
                rebuild=args.rebuild,
                base_seed=args.seed,
            )
            raw_kept, _ = build_raw_monitor_test_for_label(
                label=lbl,
                out_root=out_root,
                total_cap=max_per_label_total,
                target_counts=target_counts,
                add_meta=args.add_meta,
                shuffle_buffer=args.shuffle_buffer,
                use_auth_token=args.use_auth_token,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                skip_first_n=skip_extra,
                progress_mode=args.progress,
                demo=args.demo,
                writer_batch_size=args.writer_batch_size,
                rebuild=args.rebuild,
                base_seed=args.seed,
            )

            for split in WINDOW_SPLITS:
                grand_kept_per_split[split] += kept_map.get(split, 0)
            for split in RAW_SPLITS:
                grand_kept_per_split[split] += raw_kept.get(split, 0)

            base = out_root / "train" / canonical_label(lbl)
            print(
                "  ✓ Totals now: train={train:,}  val={val:,}  monitor={monitor:,}  test={test:,}  → {root} / <train|val|monitor|test> / {lbl_c}/dataset".format(
                    train=kept_map.get("train", 0),
                    val=kept_map.get("val", 0),
                    monitor=raw_kept.get("monitor", 0),
                    test=raw_kept.get("test", 0),
                    root=base.parent.parent,
                    lbl_c=canonical_label(lbl),
                )
                + "\n"
            )
        except KeyboardInterrupt:
            print("\n[!] Interrupted by user.", file=sys.stderr)
            raise
        except Exception as e:
            detail = describe_exc(e)
            failures.append(f"{lbl}: {detail}")
            print(f"  ✗ Failed: {detail}\n", file=sys.stderr)

        gc.collect()

    elapsed = time.time() - t0
    total = sum(grand_kept_per_split.values())
    rate = total / elapsed if elapsed > 0 else 0.0
    print("🎯 Summary")
    print(f"  Labels processed: {len(labels)}")
    print(f"  Total kept (train):   {grand_kept_per_split['train']:,}")
    print(f"  Total kept (val):     {grand_kept_per_split['val']:,}")
    print(f"  Total kept (monitor): {grand_kept_per_split['monitor']:,}")
    print(f"  Total kept (test):    {grand_kept_per_split['test']:,}")
    print(f"  Elapsed:            {elapsed/60.0:.1f} min  ({rate:.1f} samples/s)")
    if failures:
        print("  Failures:")
        for f in failures:
            print(f"    - {f}")
    print(f"\nOutput ready under: {out_root}  (layout: <split>/<label>/dataset)")

if __name__ == "__main__":
    main()
