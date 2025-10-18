#!/usr/bin/env python3
"""
magika_eval_windows.py

Evaluate Magika on random 1536-byte windows from files in language-named folders.

Constraints (matches your usage style):
- Create ONE Magika() instance and run entirely in the main thread.
- "Batching" is done at the application level (we collect windows into chunks,
  then iterate and call identify_bytes() on each entry in that chunk).
  This is compatible with Magika versions that only expose identify_bytes.
- Reports per-label accuracy + an ASCII bar chart and overall accuracy.
- NEW: For each class, also lists the top-K misclassification classes with percentages
       (relative to total windows for that class).

Example:
    python magika_eval_windows.py --root /path/to/root
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Keep native threadpools from over-subscribing (run in main thread only)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from magika import Magika                 # e.g., magika==0.6.2
from tqdm import tqdm

# -------------------- Defaults --------------------

DEFAULT_DIRS = [
    "c", "cpp", "csharp", "css", "csv", "go", "html", "java", "javascript",
    "json", "php", "python", "ruby", "rust", "sql", "text", "typescript", "yaml"
]
WINDOW_SIZE = 1536
SAMPLES_PER_DIR = 1000
BATCH_SIZE = 512
BAR_WIDTH = 40

# Robust synonyms so different Magika label strings still count as correct.
SYNONYMS: Dict[str, Sequence[str]] = {
    "c":            ("c", "text/x-c", "c-source"),
    "cpp":          ("cpp", "c++", "cxx", "text/x-c++", "cplusplus"),
    "csharp":       ("csharp", "c#", "text/x-csharp", "cs"),
    "css":          ("css", "text/css"),
    "csv":          ("csv", "text/csv"),
    "go":           ("go", "golang", "text/x-go"),
    "html":         ("html", "text/html"),
    "java":         ("java", "text/x-java"),
    "javascript":   ("javascript", "js", "application/javascript", "text/javascript"),
    "json":         ("json", "application/json"),
    "php":          ("php", "text/x-php", "application/x-php"),
    "python":       ("python", "py", "text/x-python"),
    "ruby":         ("ruby", "rb", "text/x-ruby"),
    "rust":         ("rust", "rs", "text/x-rust"),
    "sql":          ("sql", "text/x-sql"),
    "text":         ("text", "plain", "plain-text", "text/plain", "txt"),  # Magika often emits "txt"
    "typescript":   ("typescript", "ts", "text/typescript"),
    "yaml":         ("yaml", "yml", "text/yaml", "application/x-yaml"),
}

# -------------------- Helpers --------------------

def ascii_bar(pct: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(pct * width))
    return "#" * filled + "-" * (width - filled)

def pick_random_window(fp: Path, window_size: int) -> Optional[bytes]:
    try:
        size = fp.stat().st_size
        if size < window_size:
            return None
        start = random.randint(0, size - window_size)
        with fp.open("rb") as f:
            f.seek(start)
            return f.read(window_size)
    except Exception:
        return None

def is_correct(dir_label: str, magika_label: Optional[str]) -> bool:
    if not magika_label:
        return False
    gt = dir_label.lower()
    pred = str(magika_label).lower()
    for syn in SYNONYMS.get(gt, (gt,)):
        if syn in pred:
            return True
    return False

def extract_label_from_result(res) -> Optional[str]:
    """
    Compatible with common Magika result objects.
    Prefers res.ok + res.output.label; falls back to res.label if present.
    """
    try:
        ok = getattr(res, "ok", None)
        if ok is True:
            out = getattr(res, "output", None)
            if out is not None and hasattr(out, "label"):
                return str(getattr(out, "label"))
        # Fallbacks
        if hasattr(res, "label"):
            return str(getattr(res, "label"))
        if hasattr(res, "output") and hasattr(res.output, "label"):
            return str(getattr(res.output, "label"))
    except Exception:
        pass
    return None

def find_label_dir(root: Path, label: str) -> Optional[Path]:
    p = root / label
    if p.exists() and p.is_dir():
        return p
    for q in root.rglob(label):
        if q.is_dir() and q.name == label:
            return q
    return None

def choose_files(dir_path: Path, k: int, window_size: int) -> List[Path]:
    candidates: List[Path] = []
    for p in dir_path.rglob("*"):
        try:
            if p.is_file() and p.stat().st_size >= window_size:
                candidates.append(p)
        except Exception:
            continue
    if not candidates:
        return []
    if len(candidates) <= k:
        random.shuffle(candidates)
        return candidates
    return random.sample(candidates, k)

def canonicalize_label(pred_label: Optional[str], targets: Sequence[str]) -> str:
    """
    Map a raw Magika label to one of the 'targets' (our canonical label set) using SYNONYMS.
    If nothing matches, return the original lowered label (or '(none)' if absent).
    """
    if not pred_label:
        return "(none)"
    p = str(pred_label).lower()
    for canon in targets:
        for syn in SYNONYMS.get(canon, (canon,)):
            if syn in p:
                return canon
    return p  # keep the raw label bucket so we can see off-manifold predictions

# -------------------- Core evaluation (main-thread, app-level batching) --------------------

from typing import Mapping

def eval_label(
    *,
    m: Magika,
    label: str,
    dir_path: Path,
    samples: int,
    window_size: int,
    batch_size: int,
    show_progress: bool,
    all_labels: Sequence[str],
) -> Tuple[int, int, int, Dict[str, int]]:
    """
    Returns (total_evaluated, correct, skipped_small, misclass_counts)
    where misclass_counts maps predicted label -> count (only incorrect predictions).
    """
    files = choose_files(dir_path, samples, window_size)
    skipped_small = 0

    # Collect one random window per selected file
    windows: List[bytes] = []
    for fp in files:
        buf = pick_random_window(fp, window_size)
        if buf is None:
            skipped_small += 1
            continue
        windows.append(buf)

    total = len(windows)
    if total == 0:
        return 0, 0, skipped_small, {}

    correct = 0
    miscls: Dict[str, int] = {}

    indices = range(0, total, batch_size)
    iterator = indices if not show_progress else tqdm(
        indices, total=math.ceil(total / batch_size), desc=label, unit="batch", leave=False
    )

    # Application-level batching: process windows in chunks while calling identify_bytes() per window.
    for i in iterator:
        chunk = windows[i:i + batch_size]
        for w in chunk:
            res = m.identify_bytes(w)
            pred_label = extract_label_from_result(res)
            if is_correct(label, pred_label):
                correct += 1
            else:
                canon = canonicalize_label(pred_label, all_labels)
                miscls[canon] = miscls.get(canon, 0) + 1

    return total, correct, skipped_small, miscls

# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate Magika accuracy on random windows per label directory.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Root folder containing label directories.")
    ap.add_argument("--dirs", type=str, default=",".join(DEFAULT_DIRS), help="Comma-separated label dirs to use.")
    ap.add_argument("--samples-per-dir", type=int, default=SAMPLES_PER_DIR, help="Max files per directory.")
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Window size in bytes.")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="App-level batch size (grouping).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for file sampling and window offsets.")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    ap.add_argument("--bar-width", type=int, default=BAR_WIDTH, help="ASCII bar width.")
    ap.add_argument("--topk-miscls", type=int, default=3, help="How many top misclassification classes to list per label.")
    return ap.parse_args()

# -------------------- Main --------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    global BAR_WIDTH
    BAR_WIDTH = max(10, int(args.bar_width))

    root = args.root.resolve()
    labels = [x.strip() for x in args.dirs.split(",") if x.strip()]
    if not labels:
        print("[fatal] No labels to evaluate.", file=sys.stderr)
        raise SystemExit(2)

    print(f"Evaluating Magika on random {args.window_size}-byte windows (up to {args.samples_per_dir} files/dir)")
    print(f"Root: {root}")
    print("Labels:", ", ".join(labels))
    print("")

    # ONE Magika instance; main thread only.
    m = Magika()

    per_label_stats: Dict[str, Dict[str, object]] = {}
    overall_total = 0
    overall_correct = 0

    for label in labels:
        d = find_label_dir(root, label)
        if not d:
            per_label_stats[label] = {"total": 0, "correct": 0, "skipped_small": 0, "missing": 1, "miscls": {}}
            print(f"[warn] Missing directory: {root / label}")
            continue

        total, correct, skipped_small, miscls = eval_label(
            m=m,
            label=label,
            dir_path=d,
            samples=args.samples_per_dir,
            window_size=args.window_size,
            batch_size=args.batch_size,
            show_progress=not args.no_progress and sys.stderr.isatty(),
            all_labels=labels,
        )
        per_label_stats[label] = {
            "total": total,
            "correct": correct,
            "skipped_small": skipped_small,
            "missing": 0,
            "miscls": miscls,
        }
        overall_total += total
        overall_correct += correct

    # ---- ASCII bar chart + top misclassifications ----
    print("\nAccuracy (Magika correct label on random window).")
    print("Top misclassification percentages are relative to the total windows for that label.\n")

    for label in labels:
        s = per_label_stats[label]
        total = int(s["total"])
        correct = int(s["correct"])
        skipped_small = int(s["skipped_small"])
        miscls: Dict[str, int] = s.get("miscls", {}) if isinstance(s, dict) else {}

        if total > 0:
            pct = correct / total
            print(f"{label:12} | {ascii_bar(pct)} | {pct*100:5.1f}%  ({correct}/{total}, skipped_small={skipped_small})")

            if miscls:
                # sort by count desc, then label asc; take top-K
                topk = sorted(miscls.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, args.topk_miscls)]
                parts = [f"{k} {v/total*100:4.1f}% ({v}/{total})" for k, v in topk]
                print(" " * 15 + "↳ top miscls: " + ", ".join(parts))
        else:
            miss = " (missing dir)" if s.get("missing") else ""
            print(f"{label:12} | {'-'*BAR_WIDTH} |   n/a   (0/0, skipped_small={skipped_small}){miss}")

    if overall_total > 0:
        overall_pct = overall_correct / overall_total
        print("")
        print(f"OVERALL       | {ascii_bar(overall_pct)} | {overall_pct*100:5.1f}%  ({overall_correct}/{overall_total})")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted.", file=sys.stderr)
        raise
