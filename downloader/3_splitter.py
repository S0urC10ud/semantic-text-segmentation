"""
splitter.py — Create train/val/test splits for file samples.

It expects a root directory (e.g., stack_web_sample) containing
source folders (e.g., css/, html/, javascript/, etc.).

It will automatically discover all source folders and create:
  root/
    train/{source_folder_1, source_folder_2, ...}/
    val/{source_folder_1, source_folder_2, ...}/
    test/{source_folder_1, source_folder_2, ...}/

...and MOVE files from the source folders into those splits,
keeping the per-folder distribution and using a deterministic shuffle.
"""

from __future__ import annotations
import argparse
import math
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Set

def gather_source_files(source_dir: Path) -> List[Path]:
    """
    Return a list of all files under a given source directory, recursively.
    Returns an empty list if the directory does not exist.
    """
    if not source_dir.exists() or not source_dir.is_dir():
        return []
    # Collect all files recursively
    return [p for p in source_dir.rglob("*") if p.is_file()]

def compute_split_counts(n: int, train_p: float, val_p: float, test_p: float) -> Tuple[int, int, int]:
    """
    Compute per-split counts that sum exactly to n.
    Train/Val are floored; Test is the remainder.
    """
    n_train = math.floor(n * train_p)
    n_val = math.floor(n * val_p)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test

def ensure_split_dirs(root: Path, splits: List[str], source_folders: List[str]) -> Dict[str, Dict[str, Path]]:
    """
    Ensure split directories exist; return mapping split->source_folder->Path.
    """
    mapping: Dict[str, Dict[str, Path]] = {}
    for split in splits:
        mapping[split] = {}
        for source in source_folders:
            d = root / split / source
            d.mkdir(parents=True, exist_ok=True)
            mapping[split][source] = d
    return mapping

def move_or_copy(src: Path, dst_dir: Path, copy: bool = False) -> Path:
    """
    Move (default) or copy src into dst_dir, preserving the filename.
    Creates parent dirs as needed. Returns destination path.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    # If destination exists, try to find a non-colliding name
    if dst.exists():
        stem = dst.stem
        suffix = "".join(src.suffixes) if src.suffixes else ""
        counter = 1
        while True:
            candidate = dst_dir / f"{stem}__{counter}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            counter += 1
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))
    return dst

def main():
    ap = argparse.ArgumentParser(description="Split a dataset into train/val/test sets by source folder.")
    ap.add_argument("--root", type=Path, default=Path("stack_web_sample"),
                    help="Root dataset directory (default: stack_web_sample)")
    ap.add_argument("--train", type=float, default=0.7, help="Train proportion (default: 0.7)")
    ap.add_argument("--val", type=float, default=0.2, help="Val proportion (default: 0.2)")
    ap.add_argument("--test", type=float, default=0.1, help="Test proportion (default: 0.1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for deterministic shuffling (default: 42)")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without changing anything")
    args = ap.parse_args()

    # Basic checks
    total_p = args.train + args.val + args.test
    if not (0.9999 <= total_p <= 1.0001):
        ap.error(f"Proportions must sum to 1.0 (got {total_p:.6f})")

    root: Path = args.root.resolve()
    if not root.exists() or not root.is_dir():
        ap.error(f"Root directory not found: {root}")

    splits = ["train", "val", "test"]
    excluded_dirs = set(splits)

    # Automatically discover source directories to process
    source_folders = [d.name for d in root.iterdir() if d.is_dir() and d.name not in excluded_dirs]
    if not source_folders:
        ap.error(f"No source directories found in {root} to process. (It ignores 'train', 'val', 'test'.)")
    print(f"✅ Found source directories: {', '.join(source_folders)}")

    # Prepare all necessary destination split directories
    split_dirs = ensure_split_dirs(root, splits, source_folders)

    # Deterministic shuffle
    rng = random.Random(args.seed)

    grand_total = 0
    moved_total = 0

    for source_name in source_folders:
        source_dir = root / source_name
        
        # --- Robustness: Find files already in a split to avoid re-processing ---
        existing_files_in_splits: Set[str] = set()
        for split in splits:
            split_source_dir = root / split / source_name
            if split_source_dir.exists():
                for p in split_source_dir.rglob("*"):
                    if p.is_file():
                        existing_files_in_splits.add(p.name)

        # Gather source files and filter out those already processed
        all_source_files = gather_source_files(source_dir)
        files_to_process = [p for p in all_source_files if p.name not in existing_files_in_splits]
        
        n = len(files_to_process)
        grand_total += n

        if n == 0:
            total_in_source = len(all_source_files)
            if total_in_source > 0:
                print(f"[{source_name}] All {total_in_source} files already split; skipping.")
            else:
                print(f"[{source_name}] No source files found in {source_dir}; skipping.")
            continue

        rng.shuffle(files_to_process)

        n_train, n_val, n_test = compute_split_counts(n, args.train, args.val, args.test)
        idx_train_end = n_train
        idx_val_end = n_train + n_val

        train_files = files_to_process[:idx_train_end]
        val_files = files_to_process[idx_train_end:idx_val_end]
        test_files = files_to_process[idx_val_end:]

        print(f"[{source_name}] Total to process: {n} -> train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")

        # Execute moves/copies
        for split_name, batch in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dest_dir = split_dirs[split_name][source_name]
            for src in batch:
                if args.dry_run:
                    print(f"[DRY-RUN] {'COPY' if args.copy else 'MOVE'} {src} -> {dest_dir / src.name}")
                    continue
                dst = move_or_copy(src, dest_dir, copy=args.copy)
                moved_total += 1

    if args.dry_run:
        print(f"\n[DRY-RUN] Would process {grand_total} file(s).")
    else:
        action = "copied" if args.copy else "moved"
        print(f"\n🎉 Done. {action.capitalize()} {moved_total} file(s).")

if __name__ == "__main__":
    main()