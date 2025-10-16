#!/usr/bin/env python3
"""
build_arrow_windows.py (fixed)

Direct-to-Arrow windowing with Magika filtering for train/val/test (treated equally).
- No tiny window files on disk
- Progress bars per (split/label)
- Clear summaries (processed/kept/rejected)
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

# Keep native threadpools from over-subscribing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from magika import Magika  # magika==0.6.2
from tqdm import tqdm
import numpy as np
from datasets import Dataset, Features, Value, load_from_disk

# -------------------- Label set & ids --------------------

LANG2ID: Dict[str, int] = {
    "html": 0, "css": 1, "javascript": 2, "c": 3, "cpp": 4,
    "csv": 5, "java": 6, "json": 7, "python": 8, "text": 9,
}

# -------------------- Magika label matching --------------------

_DIR_ACCEPTS: Dict[str, set] = {
    "text": {"txt"},  # Magika’s generic text vs your "text"
    # Others: strict equality
}

def dir_matches_label(dir_label: Optional[str], label: Optional[str], mime: Optional[str]) -> bool:
    if dir_label is None or not label:
        return False
    dl = dir_label.lower()
    ll = str(label).lower()
    allowed = _DIR_ACCEPTS.get(dl)
    if allowed is not None:
        return ll in allowed
    return ll == dl

# -------------------- File discovery --------------------

def iter_split_files(root: Path, split: str, allowed_labels: Optional[Set[str]] = None) -> Iterable[Tuple[str, Path, Path]]:
    """
    Yield (dir_label, abs_path, rel_under_label) for files under <root>/<split>/<label>/**
    """
    split_root = root / split
    if not split_root.exists() or not split_root.is_dir():
        print(f"[!] Split not found: {split_root}", file=sys.stderr)
        return
    for label_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        dir_label = label_dir.name
        if allowed_labels and dir_label not in allowed_labels:
            continue
        for p in label_dir.rglob("*"):
            if p.is_file():
                try:
                    rel = p.relative_to(label_dir)
                except Exception:
                    rel = Path(p.name)
                yield dir_label, p, rel

def discover_labels(root: Path, splits: List[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for split in splits:
        sroot = root / split
        if not sroot.exists():
            out[split] = set()
            continue
        labels = {d.name for d in sroot.iterdir() if d.is_dir()}
        known = labels & set(LANG2ID.keys())
        unknown = labels - known
        for u in sorted(unknown):
            print(f"[warn] Ignoring unknown label '{u}' in {split}/ (not in LANG2ID).", file=sys.stderr)
        out[split] = known
    return out

# -------------------- Helpers --------------------

def _fmt_int(n: int) -> str:
    return f"{n:,}"

def estimate_windows_for_files(files: List[Tuple[str, Path, Path]], window_size: int) -> int:
    total = 0
    for _label, fpath, _rel in files:
        try:
            size = fpath.stat().st_size
        except Exception:
            size = 0
        if size > 0:
            total += math.ceil(size / window_size)
    return total

# -------------------- Observability (no temp files) --------------------

# Updated by the generator to report exact counts back to the caller.
_OBS_COUNTS: Dict[Tuple[str, str], Dict[str, int]] = {}

# -------------------- Top-level generator function (PICKLABLE) --------------------

def gen_label_examples(
    *,
    split: str,
    label: str,
    files_abs_rel: List[Tuple[str, str]],  # (abs_path_str, rel_under_label_str)
    window_size: int,
    threshold: float,
    add_meta: bool,
    lang_id: int,
    max_windows_for_this_label: Optional[int],
    progress_mode: str,
    est_total: int,
) -> Iterator[dict]:
    """
    Top-level generator FUNCTION (not a generator object). Everything in gen_kwargs must be picklable.
    Creates its own Magika+tqdm inside to avoid pickling issues.
    """
    # Local state
    kept = 0
    rejected = 0
    processed = 0

    # UI
    is_tty = sys.stderr.isatty()
    show_progress = ((progress_mode == "always") or (progress_mode == "auto" and is_tty))
    total = max(est_total, 1)
    pbar = tqdm(
        total=total,
        unit="win",
        desc=f"{split}/{label}",
        disable=not show_progress,
        leave=False,
    )

    # Classifier local to this function (no pickling needed)
    m = Magika()

    try:
        last_postfix = time.time()
        for abs_str, rel_str in files_abs_rel:
            abs_path = Path(abs_str)
            rel_under_label = Path(rel_str)

            try:
                with abs_path.open("rb") as f:
                    win_idx = 0
                    while True:
                        chunk = f.read(window_size)
                        if not chunk:
                            break
                        processed += 1
                        pbar.update(1)

                        # Magika prefilter
                        match = False
                        try:
                            res = m.identify_bytes(chunk)
                            ok = getattr(res, "ok", False)
                            out = getattr(res, "output", None) if ok else None
                            label_pred = getattr(out, "label", None) if out else None
                            mime = getattr(out, "mime_type", None) if out else None
                            conf = float(getattr(res, "score", 0.0)) if ok else 0.0
                            match = ok and dir_matches_label(label, label_pred, mime) and conf >= threshold
                        except Exception:
                            match = False

                        if match:
                            content = chunk.decode("utf-8", errors="ignore")
                            if add_meta:
                                yield {
                                    "content": content,
                                    "lang_id": np.int8(lang_id).item(),
                                    "source_relpath": str(rel_under_label),
                                    "window_idx": np.int32(win_idx).item(),
                                }
                            else:
                                yield {
                                    "content": content,
                                    "lang_id": np.int8(lang_id).item(),
                                }
                            kept += 1
                            if max_windows_for_this_label is not None and kept >= max_windows_for_this_label:
                                # Cap reached: record and return cleanly.
                                pbar.set_postfix(kept=kept, rej=rejected, lbl=label)
                                return
                        else:
                            rejected += 1

                        if (processed % 256 == 0) or (time.time() - last_postfix) > 0.3:
                            pbar.set_postfix(kept=kept, rej=rejected, lbl=label)
                            last_postfix = time.time()

                        win_idx += 1
            except Exception as e:
                pbar.write(f"[ERROR] Reading '{abs_path}': {e}")
                # continue to next file
    finally:
        pbar.set_postfix(kept=kept, rej=rejected, lbl=label)
        pbar.close()
        # Store exact counts for summaries
        _OBS_COUNTS[(split, label)] = {"processed": processed, "kept": kept, "rejected": rejected}

# -------------------- Dataset building --------------------

def build_label_dataset(
    *,
    root: Path,
    split: str,
    label: str,
    out_root: Path,
    window_size: int,
    threshold: float,
    add_meta: bool,
    files_for_label: List[Tuple[str, Path, Path]],
    est_windows_for_label: int,
    max_windows_per_class: Optional[int],
    progress_mode: str,
    rebuild: bool,
) -> Tuple[int, int, int, Path]:
    """
    Build one HF dataset at <out_root>/<split>/<label> directly from windowed, filtered bytes.
    Returns (processed, kept, rejected, out_dir) with exact counts from the generator.
    """
    out_dir = out_root / split / label
    if rebuild and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    # Features (optionally with metadata)
    features = {
        "content": Value("string"),
        "lang_id": Value("int8"),
    }
    if add_meta:
        features.update({
            "source_relpath": Value("string"),
            "window_idx": Value("int32"),
        })
    feats = Features(features)

    # Prepare picklable kwargs for the generator function
    files_abs_rel: List[Tuple[str, str]] = [(str(p), str(rel)) for (_d, p, rel) in files_for_label]

    gen_kwargs = dict(
        split=split,
        label=label,
        files_abs_rel=files_abs_rel,
        window_size=window_size,
        threshold=threshold,
        add_meta=add_meta,
        lang_id=LANG2ID[label],
        max_windows_for_this_label=max_windows_per_class,
        progress_mode=progress_mode,
        est_total=est_windows_for_label,
    )

    # Build dataset from generator FUNCTION (no pickling error)
    ds = Dataset.from_generator(gen_label_examples, gen_kwargs=gen_kwargs, features=feats, keep_in_memory=False)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))

    # Read exact counts from the generator’s side-channel
    obs = _OBS_COUNTS.get((split, label), {"processed": est_windows_for_label, "kept": len(ds), "rejected": 0})
    processed = int(obs.get("processed", est_windows_for_label))
    kept = int(obs.get("kept", len(ds)))
    rejected = int(obs.get("rejected", max(processed - kept, 0)))

    return processed, kept, rejected, out_dir

# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Direct-to-Arrow windowing with Magika filtering for train/val/test (treated equally)."
    )
    ap.add_argument("--root", type=Path, default=Path("stack_web_sample"),
                    help="Root dataset directory containing {train,val,test}/<label>/**")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Where to save Arrow datasets; default: <root>/arrow_windows")
    ap.add_argument("--splits", type=str, default="train,val,test",
                    help="Comma-separated splits to process (default: train,val,test)")
    ap.add_argument("--labels", type=str, default=None,
                    help="Comma-separated label dirs to process (e.g., 'text,html'); default: all known under each split")
    ap.add_argument("--window-size", type=int, default=1536,
                    help="Window size in bytes (default: 1536)")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="Keep only if Magika score ≥ threshold (default: 0.80)")
    ap.add_argument("--max-windows-per-class", type=int, default=None,
                    help="Cap the number of KEPT windows per split/label (applied equally to all splits).")
    ap.add_argument("--progress", choices=["auto", "always", "never"], default="auto",
                    help="Show live progress bars (default: auto)")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete existing output for the selected splits/labels before writing")
    ap.add_argument("--add-meta", action="store_true",
                    help="Include 'source_relpath' and 'window_idx' columns for observability")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed (reserved for any future randomized behavior)")
    return ap.parse_args()

# -------------------- Main --------------------

def main() -> None:
    args = parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[fatal] Root directory not found: {root}")

    out_root = args.out_root or (root / "arrow_windows")
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("[fatal] No splits specified.")

    # Determine labels per split and optionally filter by --labels
    labels_per_split = discover_labels(root, splits)
    if args.labels:
        requested = {x.strip() for x in args.labels.split(",") if x.strip()}
        requested = requested & set(LANG2ID.keys())
        for split in splits:
            found = labels_per_split.get(split, set())
            effective = requested & found
            missing = requested - found
            for m in sorted(missing):
                print(f"[!] '{m}' not found under {split}/; it will be skipped.", file=sys.stderr)
            labels_per_split[split] = effective
    for split in splits:
        if not labels_per_split.get(split):
            print(f"[!] No valid labels to process under {split}/. Skipping.", file=sys.stderr)

    np.random.seed(args.seed)

    print("⚙️  Direct-to-Arrow windowing with Magika filtering")
    print(f"   Root:         {root}")
    print(f"   Out root:     {out_root}")
    print(f"   Splits:       {', '.join(splits)}")
    print(f"   Window size:  {args.window_size} bytes")
    print(f"   Threshold:    {args.threshold:.2f}")
    if args.max_windows_per_class is not None:
        print(f"   Max/class:    {args.max_windows_per_class} windows (per split/label)")
    print(f"   Rebuild:      {'yes' if args.rebuild else 'no'}")
    print(f"   Add meta:     {'yes' if args.add_meta else 'no'}")
    print("")

    run_processed = 0
    run_kept = 0
    run_rejected = 0

    for split in splits:
        labels = sorted(labels_per_split.get(split, set()))
        if not labels:
            continue

        files_by_label: Dict[str, List[Tuple[str, Path, Path]]] = {}
        est_by_label: Dict[str, int] = {}

        for lbl in labels:
            files = [(d, p, rel) for (d, p, rel) in iter_split_files(root, split, {lbl})]
            if not files:
                print(f"[warn] No files under {split}/{lbl}; skipping.", file=sys.stderr)
                continue
            files_by_label[lbl] = files
            est_by_label[lbl] = estimate_windows_for_files(files, args.window_size)

        if not files_by_label:
            print(f"[warn] Split '{split}' has no eligible labels with files; skipping.", file=sys.stderr)
            continue

        total_est = sum(est_by_label.values()) or 1
        print(f"— Processing split: {split}  (labels: {', '.join(sorted(files_by_label.keys()))})")
        print(f"  Estimated windows: {_fmt_int(total_est)}")

        split_processed = 0
        split_kept = 0
        split_rejected = 0

        for lbl in sorted(files_by_label.keys()):
            processed, kept, rejected, out_dir = build_label_dataset(
                root=root,
                split=split,
                label=lbl,
                out_root=out_root,
                window_size=args.window_size,
                threshold=args.threshold,
                add_meta=args.add_meta,
                files_for_label=files_by_label[lbl],
                est_windows_for_label=est_by_label[lbl],
                max_windows_per_class=args.max_windows_per_class,
                progress_mode=args.progress,
                rebuild=args.rebuild,
            )
            split_processed += processed
            split_kept += kept
            split_rejected += rejected

            print(f"  ✓ {split}/{lbl}: kept {_fmt_int(kept)} "
                  f"(processed {_fmt_int(processed)}, rejected {_fmt_int(rejected)})  -> {out_dir}")

        print(f"  ▶︎ Split summary [{split}]")
        print(f"     Processed:    {_fmt_int(split_processed)}")
        print(f"     Kept:         {_fmt_int(split_kept)}")
        print(f"     Rejected:     {_fmt_int(split_rejected)}\n")

        run_processed += split_processed
        run_kept += split_kept
        run_rejected += split_rejected

    print("🎯 Run Summary")
    print(f"  Processed windows: {_fmt_int(run_processed)}")
    print(f"  Kept windows:      {_fmt_int(run_kept)}")
    print(f"  Rejected windows:  {_fmt_int(run_rejected)}")
    print("  Mode: Magika prefilter (direct-to-Arrow)")
    print("  Output is ready under:", out_root)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.", file=sys.stderr)
        raise
