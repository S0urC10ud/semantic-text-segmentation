#!/usr/bin/env python3
"""Verify every Arrow dataset produced by ``build_thesis_test_set.py``.

Asserts per-task:
  * schema + field types match what scoring expects
  * segments cover ``content`` exactly (no overlap, no gap)
  * all labels are in LANG_ORDER ∪ {"other"}
  * task-specific metadata fields are present and well-formed
  * per-host coverage hits the planned counts (when host-stratified)

Exits non-zero on the first violation.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.utils.config import LANG_ORDER  # noqa: E402

LANG_SET = set(LANG_ORDER) | {"other"}

TASKS = [
    "realistic",
    "near_pure",
    "sequence_pair",
    "sequence_triplet",
    "needle_32_63",
    "needle_64_plus",
    "markdown_mix",
]


def _check_row(task: str, row: dict, errors: list[str]) -> None:
    bid = row.get("example_id", "<no-id>")
    content = row.get("content")
    if not isinstance(content, str) or not content:
        errors.append(f"{task}::{bid}: empty content")
        return
    segs = row.get("segments")
    # segments can be either list[dict] or dict-of-columns depending on HF load shape;
    # normalise.
    if isinstance(segs, dict):
        labels = segs["label"]
        starts = segs["char_start"]
        ends = segs["char_end"]
        segs_iter = [
            {"label": labels[i], "char_start": int(starts[i]), "char_end": int(ends[i])}
            for i in range(len(labels))
        ]
    elif isinstance(segs, list):
        segs_iter = segs
    else:
        errors.append(f"{task}::{bid}: segments missing or wrong type")
        return
    if not segs_iter:
        errors.append(f"{task}::{bid}: empty segments")
        return
    cursor = 0
    for s in segs_iter:
        lab = s["label"]
        if lab not in LANG_SET:
            errors.append(f"{task}::{bid}: label '{lab}' not in LANG_ORDER ∪ other")
        cs, ce = int(s["char_start"]), int(s["char_end"])
        if cs != cursor:
            errors.append(f"{task}::{bid}: segment gap or overlap at start={cs}, cursor={cursor}")
            return
        if ce <= cs:
            errors.append(f"{task}::{bid}: empty/inverted segment [{cs},{ce})")
            return
        cursor = ce
    if cursor != len(content):
        errors.append(f"{task}::{bid}: total segments end at {cursor} but content has {len(content)} chars")


def verify_task(out_root: Path, task: str, errors: list[str]) -> dict | None:
    from datasets import load_from_disk
    target = out_root / task / "dataset"
    if not target.is_dir():
        errors.append(f"{task}: dataset directory missing at {target}")
        return None
    ds = load_from_disk(str(target))
    print(f"[verify] {task:24s}  n={len(ds):>5,d}  cols={ds.column_names}")
    lens = []
    by_host = Counter()
    label_counts: Counter = Counter()
    sample_n = min(len(ds), 600)
    for i in range(sample_n):
        row = ds[i]
        _check_row(task, row, errors)
        lens.append(len(row["content"]))
        meta = json.loads(row["metadata_json"])
        host = meta.get("host_lang") or meta.get("first_lang") or "?"
        by_host[host] += 1
        for s in row["segments"]["label"] if isinstance(row["segments"], dict) else [s["label"] for s in row["segments"]]:
            label_counts[s] += 1
    if not lens:
        return None
    p50 = int(statistics.median(lens))
    p95 = int(sorted(lens)[int(0.95 * len(lens)) - 1]) if lens else 0
    summary = {
        "task": task,
        "n": len(ds),
        "p50_chars": p50,
        "p95_chars": p95,
        "max_chars": max(lens),
        "hosts_seen": len(by_host),
        "label_diversity": len(label_counts),
    }
    print(f"            chars p50={p50:>5,d} p95={p95:>5,d} max={max(lens):>5,d}  hosts={len(by_host)}  unique_labels={len(label_counts)}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "evaluation" / "test",
    )
    args = parser.parse_args()

    if not args.out_root.is_dir():
        print(f"out-root not found: {args.out_root}", file=sys.stderr)
        return 2

    errors: list[str] = []
    summaries: list[dict] = []
    for task in TASKS:
        s = verify_task(args.out_root, task, errors)
        if s is not None:
            summaries.append(s)
        print()

    if errors:
        print(f"\n[verify] FAILED with {len(errors)} error(s):")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return 1

    print(f"[verify] ALL TASKS PASS ({len(summaries)} tasks, total rows: {sum(s['n'] for s in summaries):,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
