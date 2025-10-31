"""Count window records stored in Arrow datasets produced by the downloader.

The default directory is ``arrow_out`` which follows the structure:

    arrow_out/
      train/<label>/dataset/
      val/<label>/dataset/
      test/<label>/dataset/

For each split the script loads every Arrow dataset via Hugging Face Datasets
and prints a compact table (and bar chart) of window counts per label plus
split totals.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from datasets import load_from_disk


DEFAULT_ROOT = Path("arrow_out")
BAR_WIDTH = 50


def load_dataset_count(dataset_dir: Path) -> int:
    try:
        ds = load_from_disk(str(dataset_dir))
    except Exception as exc:
        print(f"[!] Failed to load {dataset_dir}: {exc}")
        return 0
    try:
        return len(ds)
    finally:
        del ds


def collect_counts(root: Path) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(dict)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        split = split_dir.name
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            dataset_dir = label_dir / "dataset"
            if not dataset_dir.exists():
                print(f"[!] Skipping {label_dir}: no dataset/ directory")
                continue
            count = load_dataset_count(dataset_dir)
            counts[split][label_dir.name] = count
    return counts


def format_bar(count: int, scale: float, bar_max: int) -> str:
    length = int(count * scale)
    length = min(length, bar_max)
    return "▰" * length


def print_counts(root: Path, counts: Dict[str, Dict[str, int]]):
    term_width = shutil.get_terminal_size((100, 20)).columns
    for split in sorted(counts.keys()):
        label_counts = counts[split]
        if not label_counts:
            continue
        max_label_len = max(len(label) for label in label_counts)
        max_count = max(label_counts.values())
        bar_max = min(BAR_WIDTH, term_width - max_label_len - 20)
        bar_max = max(bar_max, 10)
        scale = (bar_max / max_count) if max_count > 0 else 0.0

        print(f"\n📊 Split: {split} (root: {root})")
        total = 0
        for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])):
            bar = format_bar(count, scale, bar_max) if scale > 0 else ""
            print(f"  {label.ljust(max_label_len)} | {bar.ljust(bar_max)} | {count:,}")
            total += count
        print(f"  {'TOTAL'.ljust(max_label_len)} | {'-' * bar_max} | {total:,}\n")


def parse_args() -> Tuple[Path]:
    parser = argparse.ArgumentParser(description="Count windows stored in Arrow datasets")
    parser.add_argument("root", nargs="?", default=str(DEFAULT_ROOT), help="Root directory (default: arrow_out)")
    args = parser.parse_args()
    return Path(args.root),


def main() -> None:
    (root,) = parse_args()
    try:
        counts = collect_counts(root)
    except FileNotFoundError as exc:
        print(f"[!] {exc}")
        return

    if not counts:
        print(f"No datasets found under {root}")
        return

    print_counts(root, counts)


if __name__ == "__main__":
    main()
