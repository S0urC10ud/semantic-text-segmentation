#!/usr/bin/env python3
"""Build a static, never-seen-before LLM evaluation set.

Samples 50 rows per content type from ``downloader/arrow_out/test/<lang>/dataset``,
truncates each to 10K characters (matching the dense prompt's default cap),
and excludes any SHA256 already present in ``gemini_segmentations/{test,monitor}/``
or among the top-level ``gemini_segmentations/*.json`` files.

The resulting JSONL is the frozen benchmark used by ``run_pro_benchmark.py``.

Usage:
    python evaluation/llm_benchmark/build_unseen_set.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.utils.config import LANG_ORDER  # noqa: E402

MAX_INPUT_CHARS = 10_000
DEFAULT_SAMPLES_PER_TYPE = 50
DEFAULT_SEED = 0xD15EA5E
DEFAULT_ARROW_ROOT = REPO_ROOT / "downloader" / "arrow_out" / "test"
DEFAULT_GEMINI_ROOT = REPO_ROOT / "gemini_segmentations"
DEFAULT_OUT = REPO_ROOT / "evaluation" / "llm_benchmark" / "static_unseen_v1.jsonl"


def collect_excluded_shas(gemini_root: Path) -> set[str]:
    """Union of every SHA256 ever submitted to Gemini in this repo.

    The 64-char file names under ``gemini_segmentations/{test,monitor}/<lang>/``
    are the SHA256 of the content sent to Gemini. The top-level JSONs carry
    ``metadata.input_sha256``.
    """
    shas: set[str] = set()

    for subdir in ("test", "monitor"):
        root = gemini_root / subdir
        if not root.is_dir():
            continue
        for lang_dir in root.iterdir():
            if not lang_dir.is_dir():
                continue
            for entry in lang_dir.iterdir():
                if entry.suffix != ".json":
                    continue
                stem = entry.stem
                if len(stem) == 64 and all(c in "0123456789abcdef" for c in stem):
                    shas.add(stem)

    for entry in gemini_root.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        try:
            with entry.open() as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        meta = payload.get("metadata") if isinstance(payload, dict) else None
        sha = meta.get("input_sha256") if isinstance(meta, dict) else None
        if isinstance(sha, str) and len(sha) == 64:
            shas.add(sha)

    return shas


def sample_for_lang(
    lang: str,
    arrow_root: Path,
    excluded_shas: set[str],
    rng: random.Random,
    n_target: int,
) -> tuple[list[dict], dict]:
    """Sample ``n_target`` rows for one content type. Returns (rows, stats)."""
    from datasets import load_from_disk  # type: ignore

    ds_path = arrow_root / lang / "dataset"
    stats = {
        "lang": lang,
        "total_rows": 0,
        "after_sha_exclusion": 0,
        "sampled": 0,
        "missing_dataset": not ds_path.is_dir(),
    }
    if stats["missing_dataset"]:
        return [], stats

    ds = load_from_disk(str(ds_path))
    stats["total_rows"] = len(ds)

    eligible: list[tuple[int, str, str]] = []
    for i in range(len(ds)):
        content = ds[i]["content"]
        if not isinstance(content, str) or not content:
            continue
        truncated = content[:MAX_INPUT_CHARS]
        sha = hashlib.sha256(truncated.encode("utf-8")).hexdigest()
        if sha in excluded_shas:
            continue
        eligible.append((i, truncated, sha))

    stats["after_sha_exclusion"] = len(eligible)

    rng.shuffle(eligible)
    picked = eligible[:n_target]
    stats["sampled"] = len(picked)

    rows = []
    for idx_in_ds, content, sha in picked:
        ds_row = ds[idx_in_ds]
        rows.append(
            {
                "benchmark_id": f"{lang}::{sha}",
                "host": lang,
                "source_dataset": str(ds_path.relative_to(REPO_ROOT)),
                "source_row_index": idx_in_ds,
                "uid": ds_row.get("uid"),
                "stack_label": ds_row.get("stack_label"),
                "lang_id": int(ds_row.get("lang_id", -1)),
                "content": content,
                "content_sha256": sha,
                "input_characters": len(content),
                "original_len": len(ds_row["content"]),
                "truncated": len(ds_row["content"]) > MAX_INPUT_CHARS,
            }
        )
    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-root", type=Path, default=DEFAULT_ARROW_ROOT)
    parser.add_argument("--gemini-root", type=Path, default=DEFAULT_GEMINI_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--samples-per-type", type=int, default=DEFAULT_SAMPLES_PER_TYPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(f"[build] collecting SHA exclusion set from {args.gemini_root}...", flush=True)
    excluded = collect_excluded_shas(args.gemini_root)
    print(f"[build] excluding {len(excluded):,} previously-seen SHAs", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    per_lang_stats = []
    total_rows = 0

    with args.out.open("w", encoding="utf-8") as out_f:
        for lang in LANG_ORDER:
            print(f"[build] sampling {lang}...", flush=True)
            lang_rng = random.Random(args.seed ^ hash(lang) & 0xFFFFFFFF)
            rows, stats = sample_for_lang(
                lang,
                args.arrow_root,
                excluded,
                lang_rng,
                args.samples_per_type,
            )
            per_lang_stats.append(stats)
            for row in rows:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_rows += len(rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "samples_per_type_target": args.samples_per_type,
        "max_input_chars": MAX_INPUT_CHARS,
        "arrow_root": str(args.arrow_root),
        "gemini_root": str(args.gemini_root),
        "excluded_sha_count": len(excluded),
        "lang_order": list(LANG_ORDER),
        "per_lang": per_lang_stats,
        "total_samples": total_rows,
        "out_path": str(args.out),
    }
    manifest_path = args.out.with_name(args.out.stem + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print()
    print("=" * 70)
    print(f"Composition (target {args.samples_per_type}/type):")
    print("=" * 70)
    print(f"  {'lang':24s} {'pool_total':>10s} {'after_excl':>11s} {'sampled':>8s}")
    for s in per_lang_stats:
        marker = "" if s["sampled"] >= args.samples_per_type else "  <-- GAP"
        print(
            f"  {s['lang']:24s} {s['total_rows']:>10,d} "
            f"{s['after_sha_exclusion']:>11,d} {s['sampled']:>8,d}{marker}"
        )
    print(f"  {'TOTAL':24s} {'':>10s} {'':>11s} {total_rows:>8,d}")
    print()
    print(f"[build] wrote {total_rows:,} rows to {args.out}")
    print(f"[build] manifest at {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
