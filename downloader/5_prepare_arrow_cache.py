import os
import argparse
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from datasets import Dataset, Features, Value, load_from_disk
import numpy as np

# Matplotlib for saving charts to files (no GUI backend required)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANG2ID = {"html": 0, "css": 1, "javascript": 2, "c": 3, "cpp": 4, "csv": 5, "java": 6, "json": 7, "python": 8, "text": 9}


def _collect_files(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            out.append(os.path.join(dirpath, fn))
    return out


def _read_text(fp: str, cap_chars: int) -> str:
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(cap_chars)
    except Exception:
        return ""


def _sample_files(files: List[str], k: int, rng: np.random.Generator) -> List[str]:
    """Randomly sample k unique files without replacement (no duplicates)."""
    if k >= len(files):
        # Nothing to sample down to; return a shallow copy to avoid accidental external mutation.
        return list(files)
    # Use numpy for fast, reproducible sampling without replacement
    indices = rng.choice(len(files), size=k, replace=False)
    return [files[i] for i in indices]


def build_split_lang(
    data_root: str,
    split: str,
    lang: str,
    out_root: str,
    cap_chars: int,
    files: Optional[List[str]] = None
):
    """Build a single language dataset for a split from the provided file list (or all files if None)."""
    lang_dir = os.path.join(data_root, split, lang)
    if not os.path.isdir(lang_dir):
        raise FileNotFoundError(f"Missing: {lang_dir}")

    if files is None:
        files = _collect_files(lang_dir)
    if not files:
        raise FileNotFoundError(f"No files in {lang_dir}")

    out_dir = os.path.join(out_root, split, lang)
    os.makedirs(out_dir, exist_ok=True)

    # If already exists, skip (rebuild by deleting the folder)
    if os.path.exists(os.path.join(out_dir, "dataset_info.json")):
        print(f"[skip] {split}/{lang} already built at {out_dir}")
        return

    lid = LANG2ID[lang]
    feats = Features({
        "content": Value("string"),
        "lang_id": Value("int8"),
    })

    # Use a generator (low RAM) + progress bar
    def gen():
        pbar = tqdm(files, desc=f"Building {split}/{lang} ({len(files)} files)", unit="file")
        for fp in pbar:
            yield {
                "content": _read_text(fp, cap_chars),
                "lang_id": np.int8(lid).item(),  # compact on-disk
            }

    ds = Dataset.from_generator(gen, features=feats, keep_in_memory=False)
    ds.save_to_disk(out_dir)
    # quick sanity
    reloaded = load_from_disk(out_dir)
    print(f"[done] {split}/{lang}: {len(reloaded)} examples -> {out_dir}")


def _gather_split_files(data_root: str, split: str) -> Dict[str, List[str]]:
    """Collect all files per language for a given split with validation."""
    per_lang: Dict[str, List[str]] = {}
    for lang in LANG2ID.keys():
        lang_dir = os.path.join(data_root, split, lang)
        if not os.path.isdir(lang_dir):
            raise FileNotFoundError(f"Missing: {lang_dir}")
        files = _collect_files(lang_dir)
        if not files:
            raise FileNotFoundError(f"No files in {lang_dir}")
        per_lang[lang] = files
    return per_lang


def _save_distribution_bar_chart(
    counts: Dict[str, int],
    title: str,
    out_path: str
) -> None:
    """Save a simple bar chart of counts per language."""
    langs = sorted(counts.keys(), key=lambda k: k.lower())
    values = [counts[k] for k in langs]

    plt.figure(figsize=(10, 5))
    plt.bar(langs, values)
    plt.title(title)
    plt.xlabel("Language")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[chart] Saved distribution chart to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root that contains {train_windows,val,test}/<lang>")
    ap.add_argument("--out_root", type=str, default=None,
                    help="Where to save Arrow datasets; default: <data_root>/arrow_cache")
    ap.add_argument("--cap_chars", type=int, default=262_144,
                    help="Max chars read per file (keeps disk size reasonable)")

    # Mutually exclusive sampling strategies
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--balance", action="store_true", default=False,
                       help="If set, cap each language per split to the min class size (random sampling, no duplicates).")
    group.add_argument("--max_samples_per_class", type=int, default=None,
                       help="For the TRAIN split ONLY (train_windows), randomly sample up to this many files per language. "
                            "Other splits remain unchanged. This is NOT fully balanced; it's a per-class cap.")

    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed used for reproducible sampling.")
    args = ap.parse_args()

    base_out_root = args.out_root or os.path.join(args.data_root, "arrow_cache")
    # If balancing, keep outputs separate so we never clobber unbalanced builds
    if args.balance:
        out_root = os.path.join(base_out_root, "balanced")
    else:
        out_root = base_out_root
    os.makedirs(out_root, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Track distributions actually built (after sampling) to plot at the end
    built_counts: Dict[str, Dict[str, int]] = {"train_windows": {}, "val": {}, "test": {}}

    for split in ("train_windows", "val", "test"):
        # Gather all files for this split (validates presence)
        per_lang_files = _gather_split_files(args.data_root, split)

        min_count: Optional[int] = None
        if args.balance:
            counts = {lang: len(files) for lang, files in per_lang_files.items()}
            min_lang = min(counts, key=counts.get)
            min_count = counts[min_lang]
            print(f"[balance] {split}: min class size = {min_count} (lang='{min_lang}'). "
                  f"Capping all languages to {min_count} samples.")

        # Build each language (sampling strategy depends on flags)
        for lang, files in per_lang_files.items():
            if args.balance:
                # Balanced sampling per split
                selected_files = _sample_files(files, min_count, rng)  # type: ignore[arg-type]
            elif args.max_samples_per_class is not None and split == "train_windows":
                # Per-class cap for TRAIN ONLY
                k = min(args.max_samples_per_class, len(files))
                selected_files = _sample_files(files, k, rng)
            else:
                # Use all files
                selected_files = files

            # Ensure no duplicates in the selected list (defensive check)
            if len(selected_files) != len(set(selected_files)):
                # This should never happen due to choice without replacement; raise for safety.
                raise RuntimeError(f"Duplicate file detected in selection for {split}/{lang}.")

            # Record distribution for charting
            built_counts[split][lang] = len(selected_files)

            # Build dataset shard
            build_split_lang(args.data_root, split, lang, out_root, args.cap_chars, files=selected_files)

    # ---- Save distribution charts ----
    for split, counts in built_counts.items():
        if not counts:
            continue
        chart_path = os.path.join(out_root, f"{split}_class_distribution.png")
        title = f"{split} class distribution"
        _save_distribution_bar_chart(counts, title, chart_path)

    # Print a concise summary table to stdout
    print("\n[summary] Samples built per split/language:")
    for split in ("train_windows", "val", "test"):
        if not built_counts[split]:
            continue
        print(f"  {split}:")
        for lang in sorted(built_counts[split].keys(), key=lambda k: k.lower()):
            print(f"    - {lang:>10}: {built_counts[split][lang]:7d}")


if __name__ == "__main__":
    main()
