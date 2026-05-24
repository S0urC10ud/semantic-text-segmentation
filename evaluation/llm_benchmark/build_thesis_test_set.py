#!/usr/bin/env python3
"""Build the thesis-aligned ``evaluation/test/`` benchmark suite from Pro labels.

Reads a Pro-labelled run cache JSONL (produced by ``label_unseen_set.py``),
canonicalises Pro's raw labels to the LANG_ORDER label set, and emits one
HuggingFace Arrow dataset per benchmark task with the same schema as the
existing ``evaluation/data_b/<task>/dataset/`` folders, so the existing
scoring code in ``evaluation/evaluation.py`` and
``evaluation/compare_report_metrics.py`` consumes the new outputs unchanged.

Tasks produced (one subdir per task under ``--out-root``):
  realistic, near_pure, sequence_pair, sequence_triplet,
  needle_32_63, needle_64_plus, markdown_mix

No additional API calls are made --- every task is synthesised locally from the
cached Pro labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.utils.config import LANG_ORDER  # noqa: E402

LANG_SET = set(LANG_ORDER)

DEFAULT_CACHE = REPO_ROOT / "evaluation" / "llm_benchmark" / "runs" / "gemini-3.1-pro-preview__20260515T131145Z.jsonl"
DEFAULT_OUT_ROOT = REPO_ROOT / "evaluation" / "test"
DEFAULT_SEED = 0xC0FFEE

NEAR_PURE_CAP_CHARS = 1536
PAIR_ANCHOR_CHARS = 768
TRIPLET_ANCHOR_CHARS = 512
# Hard ceiling on synthetic-task output length (needle, markdown_mix). Realistic
# is already capped upstream at 10K by the Pro labeller. Without this cap, an
# injection on top of a 10K host can push the total above 10K, which is
# inconsistent with the rest of the suite. We truncate from the end and re-clip
# segments so the injection itself is preserved (the builder chooses cut
# positions that leave room for the injection within the cap).
MAX_OUTPUT_CHARS = 10_000
PAIR_PER_FIRST_LANG = 50
TRIPLET_PER_FIRST_LANG = 50
NEEDLE_PER_HOST = 30
NEEDLE_32_63_RANGE = (32, 63)
NEEDLE_64_PLUS_RANGE = (64, 200)
MARKDOWN_HOST_LIMIT = 50  # all markdown records (we have ~50 ok ones)
MARKDOWN_MODES = ("fenced", "plain", "inline")
MARKDOWN_INLINE_MAX_CHARS = 64
MARKDOWN_PLAIN_MAX_CHARS = 600
MARKDOWN_FENCED_MAX_CHARS = 800
MARKDOWN_MISMATCH_FRACTION = 0.33  # fraction of fenced blocks with mismatched fence label


def _canon(label: str | None) -> str:
    """Normalise raw Pro label to the LANG_ORDER label set."""
    if not label:
        return "other"
    L = label.lower().strip()
    if L in ("javascript", "typescript", "js", "ts"):
        return "javascript_typescript"
    if L in ("c", "cpp", "c++", "objective_c", "objectivec"):
        return "c_family"
    if L == "gettext-catalog":
        return "gettext_catalog"
    if L.startswith("other_") or L == "other":
        return "other"
    if L not in LANG_SET:
        return "other"
    return L


def _segments_from_record(rec: dict) -> list[dict]:
    """Convert a cache record's ``segments`` list-of-{type,content} into the
    Arrow-friendly list-of-{label, char_start, char_end} representation, while
    canonicalising labels.
    """
    out: list[dict] = []
    pos = 0
    for s in rec.get("segments", []):
        content = s.get("content", "")
        L = len(content)
        out.append({
            "label": _canon(s.get("type")),
            "char_start": pos,
            "char_end": pos + L,
        })
        pos += L
    # Merge adjacent segments with the same canonicalised label (Pro sometimes
    # emits raw `javascript` + `typescript` adjacently which we collapse).
    merged: list[dict] = []
    for s in out:
        if merged and merged[-1]["label"] == s["label"]:
            merged[-1] = {**merged[-1], "char_end": s["char_end"]}
        else:
            merged.append(s)
    return merged


def _hash_id(task: str, key: str) -> str:
    digest = hashlib.sha256(f"{task}::{key}".encode("utf-8")).hexdigest()
    return digest[:24]


def _visible_count(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def _host_label_ratio(segments: list[dict], host: str, content: str) -> float:
    if not segments or not content:
        return 0.0
    host_chars = sum(s["char_end"] - s["char_start"] for s in segments if s["label"] == host)
    total = len(content)
    return host_chars / max(total, 1)


def _clip_segments(segments: list[dict], end: int) -> list[dict]:
    """Restrict a segment list to ``content[:end]``."""
    out = []
    for s in segments:
        if s["char_start"] >= end:
            break
        out.append({
            "label": s["label"],
            "char_start": s["char_start"],
            "char_end": min(s["char_end"], end),
        })
    return [s for s in out if s["char_end"] > s["char_start"]]


def _shift_segments(segments: list[dict], offset: int) -> list[dict]:
    return [
        {"label": s["label"], "char_start": s["char_start"] + offset, "char_end": s["char_end"] + offset}
        for s in segments
    ]


def _take_window(content: str, segments: list[dict], target_chars: int) -> tuple[str, list[dict]]:
    """Take ``content[:target_chars]`` aligned with the segment grid."""
    end = min(len(content), target_chars)
    return content[:end], _clip_segments(segments, end)


def _make_row(task: str, key: str, content: str, segments: list[dict], metadata: dict, source_langs: Iterable[str]) -> dict:
    return {
        "task": task,
        "example_id": _hash_id(task, key),
        "content": content,
        "segments": segments,
        "source_langs": sorted(set(source_langs)),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


# ---------------------------------------------------------------------------
# Pool construction


@dataclass
class LabelledRecord:
    benchmark_id: str
    host: str
    content: str
    segments: list[dict]
    host_label_ratio: float = field(default=0.0)

    def primary_label(self) -> str:
        if not self.segments:
            return "other"
        return max(self.segments, key=lambda s: s["char_end"] - s["char_start"])["label"]


def load_pool(cache_path: Path) -> list[LabelledRecord]:
    pool: list[LabelledRecord] = []
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            host = r["host"]
            content = ""
            # Reconstruct content from segments (the cache stores Pro's reproduction).
            for s in r.get("segments", []):
                content += s.get("content", "")
            if not content:
                continue
            segments = _segments_from_record(r)
            host_lr = _host_label_ratio(segments, host, content)
            pool.append(LabelledRecord(
                benchmark_id=r["benchmark_id"],
                host=host,
                content=content,
                segments=segments,
                host_label_ratio=host_lr,
            ))
    return pool


def index_by_host(pool: list[LabelledRecord]) -> dict[str, list[LabelledRecord]]:
    out: dict[str, list[LabelledRecord]] = {}
    for r in pool:
        out.setdefault(r.host, []).append(r)
    return out


# ---------------------------------------------------------------------------
# Per-task builders


def build_realistic(pool: list[LabelledRecord]) -> list[dict]:
    rows = []
    for r in pool:
        meta = {
            "host_lang": r.host,
            "host_label_ratio": r.host_label_ratio,
            "cache_benchmark_id": r.benchmark_id,
        }
        rows.append(_make_row(
            "realistic", r.benchmark_id, r.content, r.segments, meta,
            [s["label"] for s in r.segments],
        ))
    return rows


def build_near_pure(pool: list[LabelledRecord]) -> list[dict]:
    rows = []
    for r in pool:
        if r.host_label_ratio < 0.95:
            continue
        content, segments = _take_window(r.content, r.segments, NEAR_PURE_CAP_CHARS)
        if not content:
            continue
        meta = {
            "host_lang": r.host,
            "host_label_ratio": _host_label_ratio(segments, r.host, content),
            "cache_benchmark_id": r.benchmark_id,
            "char_cap_applied": NEAR_PURE_CAP_CHARS,
        }
        rows.append(_make_row(
            "near_pure", r.benchmark_id, content, segments, meta,
            [s["label"] for s in segments],
        ))
    return rows


def build_sequence_pair(by_host: dict[str, list[LabelledRecord]], rng: random.Random) -> list[dict]:
    hosts = [h for h in LANG_ORDER if by_host.get(h)]
    rows = []
    for first in hosts:
        first_pool = by_host[first]
        for k in range(PAIR_PER_FIRST_LANG):
            choices = [h for h in hosts if h != first]
            second = rng.choice(choices)
            second_pool = by_host[second]
            r_a = rng.choice(first_pool)
            r_b = rng.choice(second_pool)
            c_a, s_a = _take_window(r_a.content, r_a.segments, PAIR_ANCHOR_CHARS)
            c_b, s_b = _take_window(r_b.content, r_b.segments, PAIR_ANCHOR_CHARS)
            if not c_a or not c_b:
                continue
            content = c_a + c_b
            segments = s_a + _shift_segments(s_b, len(c_a))
            key = f"{first}-{second}-{k}-{r_a.benchmark_id[:12]}-{r_b.benchmark_id[:12]}"
            meta = {
                "first_lang": first,
                "second_lang": second,
                "host_lang": first,
                "first_uid": r_a.benchmark_id,
                "second_uid": r_b.benchmark_id,
                "anchor_chars": PAIR_ANCHOR_CHARS,
                "first_anchor_len": len(c_a),
                "second_anchor_len": len(c_b),
            }
            rows.append(_make_row(
                "sequence_pair", key, content, segments, meta,
                [s["label"] for s in segments],
            ))
    return rows


def build_sequence_triplet(by_host: dict[str, list[LabelledRecord]], rng: random.Random) -> list[dict]:
    hosts = [h for h in LANG_ORDER if by_host.get(h)]
    rows = []
    for first in hosts:
        first_pool = by_host[first]
        for k in range(TRIPLET_PER_FIRST_LANG):
            others = [h for h in hosts if h != first]
            mid = rng.choice(others)
            third = rng.choice([h for h in others if h != mid] or others)
            r_a = rng.choice(first_pool)
            r_b = rng.choice(by_host[mid])
            r_c = rng.choice(by_host[third])
            c_a, s_a = _take_window(r_a.content, r_a.segments, TRIPLET_ANCHOR_CHARS)
            c_b, s_b = _take_window(r_b.content, r_b.segments, TRIPLET_ANCHOR_CHARS)
            c_c, s_c = _take_window(r_c.content, r_c.segments, TRIPLET_ANCHOR_CHARS)
            if not c_a or not c_b or not c_c:
                continue
            content = c_a + c_b + c_c
            segments = (
                s_a
                + _shift_segments(s_b, len(c_a))
                + _shift_segments(s_c, len(c_a) + len(c_b))
            )
            key = f"{first}-{mid}-{third}-{k}-{r_a.benchmark_id[:8]}-{r_b.benchmark_id[:8]}-{r_c.benchmark_id[:8]}"
            meta = {
                "first_lang": first,
                "second_lang": mid,
                "third_lang": third,
                "host_lang": first,
                "first_uid": r_a.benchmark_id,
                "mid_uid": r_b.benchmark_id,
                "tail_uid": r_c.benchmark_id,
                "anchor_chars": TRIPLET_ANCHOR_CHARS,
            }
            rows.append(_make_row(
                "sequence_triplet", key, content, segments, meta,
                [s["label"] for s in segments],
            ))
    return rows


def _sample_donor_span(donor: LabelledRecord, n_visible_min: int, n_visible_max: int, rng: random.Random) -> tuple[str, list[dict]] | None:
    """Find a substring of ``donor.content`` containing between min..max visible
    tokens (non-whitespace chars), and return its per-segment labels.

    Returns ``(span, span_segments)`` where ``span_segments`` is the list of
    ``{label, char_start, char_end}`` covering ``[0, len(span))`` using the
    donor's actual Pro labels (so a span that lands in a sub-region of a
    different content type gets the correct label, not the donor record's
    overall primary label). Returns None if no valid span is found.
    """
    content = donor.content
    n_vis = _visible_count(content)
    if n_vis < n_visible_min:
        return None
    target_vis = rng.randint(n_visible_min, min(n_visible_max, n_vis))
    vis_positions = [i for i, c in enumerate(content) if not c.isspace()]
    if len(vis_positions) < target_vis:
        return None
    start_pick_max = len(vis_positions) - target_vis
    start_idx = rng.randint(0, start_pick_max)
    start = vis_positions[start_idx]
    end = vis_positions[start_idx + target_vis - 1] + 1
    line_start = content.rfind("\n", 0, start)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line_end = content.find("\n", end)
    if line_end == -1:
        line_end = len(content)
    span = content[line_start:line_end]
    if not span.strip():
        return None
    if _visible_count(span) < n_visible_min:
        # Shrink back to the inner range.
        line_start = start
        line_end = end
        span = content[line_start:line_end]
    if _visible_count(span) > n_visible_max * 2:
        return None
    # Build per-segment labels for the span using donor's actual Pro labels.
    span_segments: list[dict] = []
    for s in donor.segments:
        if s["char_end"] <= line_start:
            continue
        if s["char_start"] >= line_end:
            break
        seg_start = max(s["char_start"], line_start) - line_start
        seg_end = min(s["char_end"], line_end) - line_start
        if seg_end > seg_start:
            span_segments.append({
                "label": s["label"],
                "char_start": seg_start,
                "char_end": seg_end,
            })
    if not span_segments:
        return None
    # Ensure the segment list covers [0, len(span)).
    span_segments[0]["char_start"] = 0
    span_segments[-1]["char_end"] = len(span)
    return span, span_segments


def _build_needle_bucket(
    by_host: dict[str, list[LabelledRecord]],
    bucket_name: str,
    n_visible_range: tuple[int, int],
    rng: random.Random,
) -> list[dict]:
    hosts = [h for h in LANG_ORDER if by_host.get(h)]
    rows = []
    for host in hosts:
        host_pool = by_host[host]
        attempts = 0
        produced = 0
        while produced < NEEDLE_PER_HOST and attempts < NEEDLE_PER_HOST * 6:
            attempts += 1
            r_host = rng.choice(host_pool)
            host_content = r_host.content
            if "\n" not in host_content:
                continue
            donor_choice = rng.choice([h for h in hosts if h != host])
            r_donor = rng.choice(by_host[donor_choice])
            span_pair = _sample_donor_span(r_donor, *n_visible_range, rng=rng)
            if span_pair is None:
                continue
            span, span_segments = span_pair
            injected = span if span.endswith("\n") else span + "\n"
            # Choose a host newline injection point that leaves room for the
            # full injection + a small suffix tail within MAX_OUTPUT_CHARS.
            min_tail = 1
            max_cut = max(0, MAX_OUTPUT_CHARS - len(injected) - min_tail)
            candidate_cuts = [i + 1 for i, c in enumerate(host_content) if c == "\n" and (i + 1) <= max_cut]
            if not candidate_cuts:
                continue
            cut = rng.choice(candidate_cuts)
            prefix = host_content[:cut]
            suffix = host_content[cut:]
            content = prefix + injected + suffix
            # Build segments: host prefix segments + donor span segments
            # (with their real per-region labels) + shifted host suffix segments.
            seg_prefix = _clip_segments(r_host.segments, cut)
            seg_suffix_local = [
                {"label": s["label"], "char_start": max(0, s["char_start"] - cut), "char_end": s["char_end"] - cut}
                for s in r_host.segments if s["char_end"] > cut
            ]
            seg_suffix = _shift_segments(seg_suffix_local, len(prefix) + len(injected))
            # Shift donor span segments to their absolute position. The donor
            # span_segments cover [0, len(span)); pad the trailing newline (if
            # we appended one) onto the last donor segment so coverage is exact.
            donor_segments_abs = []
            for s in span_segments:
                donor_segments_abs.append({
                    "label": s["label"],
                    "char_start": s["char_start"] + len(prefix),
                    "char_end": s["char_end"] + len(prefix),
                })
            expected_end = len(prefix) + len(injected)
            if donor_segments_abs and donor_segments_abs[-1]["char_end"] < expected_end:
                donor_segments_abs[-1]["char_end"] = expected_end
            segments = seg_prefix + donor_segments_abs + seg_suffix
            # Apply hard cap (defensive; cut selection above already keeps us under).
            if len(content) > MAX_OUTPUT_CHARS:
                content = content[:MAX_OUTPUT_CHARS]
                segments = _clip_segments(segments, MAX_OUTPUT_CHARS)
            # Coalesce same-label adjacent segments.
            merged = []
            for s in segments:
                if merged and merged[-1]["label"] == s["label"] and merged[-1]["char_end"] == s["char_start"]:
                    merged[-1] = {**merged[-1], "char_end": s["char_end"]}
                else:
                    merged.append(s)
            segments = merged
            # Donor "primary" lang for metadata is the largest-by-chars label
            # inside the span, not the donor record's overall primary label.
            donor_primary = max(span_segments, key=lambda s: s["char_end"] - s["char_start"])["label"]
            key = f"{host}-{donor_primary}-{produced}-{r_host.benchmark_id[:8]}-{r_donor.benchmark_id[:8]}"
            meta = {
                "host_lang": host,
                "donor_lang": donor_primary,
                "donor_record_host_lang": donor_choice,
                "donor_span_label_breakdown": {
                    s["label"]: s["char_end"] - s["char_start"]
                    for s in span_segments
                },
                "size_bucket": bucket_name.replace("needle_", ""),
                "host_uid": r_host.benchmark_id,
                "donor_uid": r_donor.benchmark_id,
                "donor_visible_tokens": _visible_count(span),
                "injection_char_start": len(prefix),
                "injection_char_end": min(len(prefix) + len(injected), MAX_OUTPUT_CHARS),
            }
            rows.append(_make_row(
                bucket_name, key, content, segments, meta,
                [s["label"] for s in segments],
            ))
            produced += 1
    return rows


def build_needle_32_63(by_host: dict[str, list[LabelledRecord]], rng: random.Random) -> list[dict]:
    return _build_needle_bucket(by_host, "needle_32_63", NEEDLE_32_63_RANGE, rng)


def build_needle_64_plus(by_host: dict[str, list[LabelledRecord]], rng: random.Random) -> list[dict]:
    return _build_needle_bucket(by_host, "needle_64_plus", NEEDLE_64_PLUS_RANGE, rng)


def build_markdown_mix(by_host: dict[str, list[LabelledRecord]], rng: random.Random) -> list[dict]:
    md_records = by_host.get("markdown", [])
    if not md_records:
        return []
    md_records = md_records[:MARKDOWN_HOST_LIMIT]
    candidate_donor_langs = [h for h in LANG_ORDER if h != "markdown" and by_host.get(h)]
    rows = []
    for md_rec in md_records:
        host_content = md_rec.content
        for mode in MARKDOWN_MODES:
            donor_lang = rng.choice(candidate_donor_langs)
            r_donor = rng.choice(by_host[donor_lang])
            # Donor chunk: bounded by mode
            if mode == "inline":
                cap = MARKDOWN_INLINE_MAX_CHARS
            elif mode == "plain":
                cap = MARKDOWN_PLAIN_MAX_CHARS
            else:
                cap = MARKDOWN_FENCED_MAX_CHARS
            donor_chunk = r_donor.content[:cap].rstrip("\n")
            if not donor_chunk:
                continue
            # Build a worst-case wrapper-length estimate so we pick a cut that
            # leaves room for prefix + injection + suffix within MAX_OUTPUT_CHARS.
            wrapper_overhead = 64 if mode == "fenced" else (4 if mode == "plain" else 6)
            min_tail = 1
            max_cut = max(0, MAX_OUTPUT_CHARS - len(donor_chunk) - wrapper_overhead - min_tail)
            newlines = [i + 1 for i, c in enumerate(host_content) if c == "\n" and (i + 1) <= max_cut]
            if not newlines:
                continue
            cut = rng.choice(newlines)
            mismatched_fence = False
            if mode == "fenced":
                display_lang = donor_lang
                if rng.random() < MARKDOWN_MISMATCH_FRACTION:
                    candidates_wrong = [h for h in candidate_donor_langs if h != donor_lang]
                    if candidates_wrong:
                        display_lang = rng.choice(candidates_wrong)
                        mismatched_fence = True
                pre_wrap = f"```{display_lang}\n"
                post_wrap = "\n```\n"
                injected = pre_wrap + donor_chunk + post_wrap
                donor_start = cut + len(pre_wrap)
                donor_end = cut + len(pre_wrap) + len(donor_chunk)
            elif mode == "plain":
                pre_wrap = "\n"
                post_wrap = "\n\n"
                injected = pre_wrap + donor_chunk + post_wrap
                donor_start = cut + len(pre_wrap)
                donor_end = cut + len(pre_wrap) + len(donor_chunk)
            else:  # inline
                # use first line of donor_chunk, kept short
                short = donor_chunk.splitlines()[0][:MARKDOWN_INLINE_MAX_CHARS]
                if not short:
                    continue
                pre_wrap = " `"
                post_wrap = "` "
                injected = pre_wrap + short + post_wrap
                donor_start = cut + len(pre_wrap)
                donor_end = cut + len(pre_wrap) + len(short)
                donor_chunk = short  # actual chunk that was injected
            new_content = host_content[:cut] + injected + host_content[cut:]
            # Build segments: re-clip host markdown segments, insert donor segment, then suffix shifted
            seg_prefix = _clip_segments(md_rec.segments, cut)
            seg_suffix = _shift_segments(
                [
                    {"label": s["label"], "char_start": max(0, s["char_start"] - cut), "char_end": s["char_end"] - cut}
                    for s in md_rec.segments if s["char_end"] > cut
                ],
                cut + len(injected),
            )
            # Wrapper bytes around donor stay markdown
            wrapper_before = {"label": "markdown", "char_start": cut, "char_end": donor_start}
            donor_segment = {"label": donor_lang, "char_start": donor_start, "char_end": donor_end}
            wrapper_after = {"label": "markdown", "char_start": donor_end, "char_end": cut + len(injected)}
            wrapper_segments = [seg for seg in (wrapper_before, donor_segment, wrapper_after) if seg["char_end"] > seg["char_start"]]
            segments = seg_prefix + wrapper_segments + seg_suffix
            # Hard cap (defensive; cut selection already keeps content under).
            if len(new_content) > MAX_OUTPUT_CHARS:
                new_content = new_content[:MAX_OUTPUT_CHARS]
                segments = _clip_segments(segments, MAX_OUTPUT_CHARS)
                # Trim markdown_blocks too if injection got chopped.
                donor_end = min(donor_end, MAX_OUTPUT_CHARS)
            # Coalesce
            merged = []
            for s in segments:
                if merged and merged[-1]["label"] == s["label"] and merged[-1]["char_end"] == s["char_start"]:
                    merged[-1] = {**merged[-1], "char_end": s["char_end"]}
                else:
                    merged.append(s)
            segments = merged
            markdown_blocks = [{
                "role": "other",
                "char_start": donor_start,
                "char_end": donor_end,
                "language": donor_lang,
                "wrapper": mode,
                "mismatched": mismatched_fence,
            }]
            key = f"{md_rec.benchmark_id[:12]}-{mode}-{donor_lang}-{r_donor.benchmark_id[:8]}"
            meta = {
                "host_lang": "markdown",
                "wrapper_mode": mode,
                "donor_lang": donor_lang,
                "mismatched_fence": mismatched_fence,
                "md_uid": md_rec.benchmark_id,
                "donor_uid": r_donor.benchmark_id,
                "markdown_blocks": markdown_blocks,
            }
            rows.append(_make_row(
                "markdown_mix", key, new_content, segments, meta,
                [s["label"] for s in segments],
            ))
    return rows


# ---------------------------------------------------------------------------
# Arrow writer


def features_schema():
    from datasets import Features, Sequence, Value
    return Features({
        "task": Value("string"),
        "example_id": Value("string"),
        "content": Value("string"),
        "segments": Sequence({
            "label": Value("string"),
            "char_start": Value("int32"),
            "char_end": Value("int32"),
        }),
        "source_langs": Sequence(Value("string")),
        "metadata_json": Value("string"),
    })


def write_task(task: str, rows: list[dict], out_root: Path) -> dict:
    from datasets import Dataset
    if not rows:
        print(f"[build] WARNING: task {task!r} produced 0 rows; skipping write.")
        return {"task": task, "count": 0}
    target = out_root / task / "dataset"
    target.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows, features=features_schema())
    ds.save_to_disk(str(target))
    print(f"[build] wrote {task:24s} count={len(rows):>5,d}  ->  {target.relative_to(REPO_ROOT)}")
    return {
        "task": task,
        "count": len(rows),
        "path": str(target.relative_to(REPO_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["realistic", "near_pure", "sequence_pair", "sequence_triplet",
                 "needle_32_63", "needle_64_plus", "markdown_mix"],
    )
    args = parser.parse_args()

    print(f"[build] loading {args.cache}", flush=True)
    pool = load_pool(args.cache)
    print(f"[build] loaded {len(pool)} ok records from cache")
    by_host = index_by_host(pool)
    for h in LANG_ORDER:
        n = len(by_host.get(h, []))
        if n < 10:
            print(f"[build] WARNING: host '{h}' has only {n} record(s)")

    rng = random.Random(args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifests: list[dict] = []

    builders = {
        "realistic": lambda: build_realistic(pool),
        "near_pure": lambda: build_near_pure(pool),
        "sequence_pair": lambda: build_sequence_pair(by_host, random.Random(args.seed ^ 0x1)),
        "sequence_triplet": lambda: build_sequence_triplet(by_host, random.Random(args.seed ^ 0x2)),
        "needle_32_63": lambda: build_needle_32_63(by_host, random.Random(args.seed ^ 0x3)),
        "needle_64_plus": lambda: build_needle_64_plus(by_host, random.Random(args.seed ^ 0x4)),
        "markdown_mix": lambda: build_markdown_mix(by_host, random.Random(args.seed ^ 0x5)),
    }

    for task in args.tasks:
        if task not in builders:
            print(f"[build] unknown task: {task}; skipping")
            continue
        print(f"[build] building task: {task}", flush=True)
        rows = builders[task]()
        manifests.append(write_task(task, rows, args.out_root))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache": str(args.cache.relative_to(REPO_ROOT) if REPO_ROOT in args.cache.parents else args.cache),
        "seed": args.seed,
        "anchor_chars": {
            "sequence_pair": PAIR_ANCHOR_CHARS,
            "sequence_triplet": TRIPLET_ANCHOR_CHARS,
        },
        "near_pure_cap_chars": NEAR_PURE_CAP_CHARS,
        "pair_per_first_lang": PAIR_PER_FIRST_LANG,
        "triplet_per_first_lang": TRIPLET_PER_FIRST_LANG,
        "needle_per_host": NEEDLE_PER_HOST,
        "needle_visible_token_ranges": {
            "32_63": NEEDLE_32_63_RANGE,
            "64_plus": NEEDLE_64_PLUS_RANGE,
        },
        "markdown": {
            "modes": list(MARKDOWN_MODES),
            "mismatch_fraction": MARKDOWN_MISMATCH_FRACTION,
        },
        "pool_size": len(pool),
        "tasks": manifests,
    }
    manifest_path = args.out_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build] wrote manifest: {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
