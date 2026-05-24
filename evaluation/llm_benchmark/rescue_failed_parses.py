#!/usr/bin/env python3
"""Re-parse cached ``failed_parse`` records using the current parser.

When ``label_unseen_set.py`` hits a parse failure it stores the raw response
text alongside the API cost. After upstream parser bug fixes, those records
can be promoted to ``ok`` without paying for the API call again.

The original cache file is rewritten in place; a ``.pre_rescue.bak`` copy is
created next to it. Records that still fail to parse keep their old state.

Usage:
    python evaluation/llm_benchmark/rescue_failed_parses.py runs/<run_id>.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.llm_benchmark.label_unseen_set import parse_and_heal  # noqa: E402

DEFAULT_STATIC = REPO_ROOT / "evaluation" / "llm_benchmark" / "static_unseen_v1.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="runs/<run_id>.jsonl to rescue")
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    args = parser.parse_args()

    if not args.cache.is_file():
        print(f"cache file not found: {args.cache}", file=sys.stderr)
        return 2
    if not args.static.is_file():
        print(f"static file not found: {args.static}", file=sys.stderr)
        return 2

    static_content: dict[str, str] = {}
    with args.static.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            static_content[row["benchmark_id"]] = row["content"]

    backup = args.cache.with_suffix(args.cache.suffix + ".pre_rescue.bak")
    shutil.copy2(args.cache, backup)
    print(f"backup: {backup}")

    with args.cache.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rescued = 0
    still_failing = 0
    untouched = 0
    for rec in records:
        if rec.get("status") != "failed_parse":
            untouched += 1
            continue
        resp = rec.get("response_text")
        src = static_content.get(rec.get("benchmark_id", ""))
        if not resp or src is None:
            still_failing += 1
            continue
        try:
            segments, healing, error = parse_and_heal(
                source_text=src, generated_code=resp
            )
        except BaseException as exc:
            rec["rescue_attempted"] = True
            rec["rescue_error"] = f"parse_chain_exception: {exc}"
            still_failing += 1
            continue
        if segments is None:
            rec["rescue_attempted"] = True
            rec["rescue_error"] = error
            still_failing += 1
            continue
        rec["status"] = "ok"
        rec["segments"] = segments
        rec["healing"] = healing
        rec["rescue_attempted"] = True
        rec["rescue_promoted_at"] = "now"
        # response_text was kept on failure; drop it from rescued records to
        # keep the cache compact (segments cover identical bytes anyway).
        rec.pop("response_text", None)
        rec.pop("error", None)
        rec.pop("response_text_head", None)
        rec.pop("response_text_tail", None)
        rescued += 1

    with args.cache.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"rescue summary: rescued={rescued} "
        f"still_failing={still_failing} unchanged_ok_or_other={untouched}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
