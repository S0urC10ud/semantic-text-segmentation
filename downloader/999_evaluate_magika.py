

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

from magika import Magika
from tqdm import tqdm

# -------------------- Defaults --------------------

DEFAULT_DIRS = [
    "c", "cpp", "csharp", "css", "csv", "go", "html", "java", "javascript", "typescript",
    "json", "php", "python", "ruby", "rust", "sql", "text", "shell", "yaml"
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
    "javascript": ("javascript", "js", "application/javascript", "text/javascript"),
    "typescript":   ("typescript", "ts", "text/typescript"),
    "json":         ("json", "application/json"),
    "php":          ("php", "text/x-php", "application/x-php"),
    "python":       ("python", "py", "text/x-python"),
    "ruby":         ("ruby", "rb", "text/x-ruby"),
    "rust":         ("rust", "rs", "text/x-rust"),
    "sql":          ("sql", "text/x-sql"),
    "text":         ("text", "plain", "plain-text", "text/plain", "txt"),  # Magika often emits "txt"
    "shell":        ("shell", "bash", "sh", "zsh", "fish", "batchfile", "bat", "cmd", "shell_batchfile"),
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

def _count_candidate_files(dir_path: Path, window_size: int) -> int:
    """Count files large enough to sample ``window_size`` bytes from."""
    count = 0
    for p in dir_path.rglob("*"):
        try:
            if p.is_file() and p.stat().st_size >= window_size:
                count += 1
        except OSError:
            continue
    return count


def find_label_dir(root: Path, label: str, *, window_size: int) -> Tuple[Optional[Path], int]:
    """Locate the best directory under ``root`` that matches ``label``."""
    best_path: Optional[Path] = None
    best_count = -1

    direct = root / label
    if direct.exists() and direct.is_dir():
        count = _count_candidate_files(direct, window_size)
        best_path = direct if count >= 0 else None
        best_count = count

    for q in root.rglob(label):
        if not (q.is_dir() and q.name == label):
            continue
        if best_path is not None and q.resolve() == best_path.resolve():
            continue
        count = _count_candidate_files(q, window_size)
        if count > best_count:
            best_path = q
            best_count = count

    return best_path, max(best_count, 0)

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
    """Map raw Magika label to a canonical label from ``targets`` using SYNONYMS."""
    if not pred_label:
        return "(none)"

    p = str(pred_label).strip().lower()
    if not targets:
        return p

    # First pass: exact matches against synonyms (preferred).
    for canon in targets:
        canon_lower = canon.lower()
        synonyms = SYNONYMS.get(canon_lower, (canon_lower,))
        for syn in synonyms:
            syn_lower = syn.lower()
            if p == syn_lower:
                return canon

    # Second pass: pick the synonym with the longest substring match.
    best_canon: Optional[str] = None
    best_len = -1
    for canon in targets:
        canon_lower = canon.lower()
        synonyms = SYNONYMS.get(canon_lower, (canon_lower,))
        for syn in synonyms:
            syn_lower = syn.lower()
            if syn_lower and syn_lower in p:
                match_len = len(syn_lower)
                if match_len > best_len:
                    best_len = match_len
                    best_canon = canon

    if best_canon is not None:
        return best_canon

    return p  # off-manifold prediction bucket


def iter_windows_cover_all_bytes(fp: Path, window_size: int):
    """
    Yield 1536-byte windows that *cover all bytes* of the file.
    Uses non-overlapping windows from 0, window_size, ... and, if there is
    a tail shorter than window_size, adds ONE final (overlapping) window
    starting at size - window_size so the tail bytes are covered.
    """
    size = fp.stat().st_size
    if size < window_size:
        return  # nothing to yield
    try:
        with fp.open("rb") as f:
            # Non-overlapping full windows
            pos = 0
            while pos + window_size <= size:
                f.seek(pos)
                buf = f.read(window_size)
                if not buf or len(buf) < window_size:
                    break
                yield buf
                pos += window_size

            # If there is a remainder, add one more window covering the tail.
            if size % window_size != 0:
                start = max(0, size - window_size)
                # Avoid duplicating if the last non-overlapping step already hit 'start'
                if start != pos - window_size:
                    f.seek(start)
                    buf = f.read(window_size)
                    if buf and len(buf) == window_size:
                        yield buf
    except Exception:
        return  # silently ignore unreadable files


# -------------------- Core evaluation (main-thread, app-level batching) --------------------

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
    threshold: float,
) -> Tuple[int, int, int, Dict[str, int]]:
    """
    Per-window evaluation (existing behavior).
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
        indices, total=math.ceil(total / batch_size), desc=f"{label} (windows)", unit="batch", leave=False
    )

    # Application-level batching: process windows in chunks while calling identify_bytes() per window.
    for i in iterator:
        chunk = windows[i:i + batch_size]
        for w in chunk:
            res = m.identify_bytes(w)
            pred_label = extract_label_from_result(res)
            pred_str = str(pred_label).strip().lower() if pred_label else "(none)"

            ok = bool(getattr(res, "ok", False))
            score = float(getattr(res, "score", 0.0)) if ok else 0.0

            canon = canonicalize_label(pred_label, all_labels)
            matches_label = canon == label
            meets_threshold = score > threshold

            if matches_label and meets_threshold:
                correct += 1
                continue

            if matches_label:
                key = f"{pred_str} (below-threshold)"
            else:
                key = canon if canon != "(none)" else pred_str
            miscls[key] = miscls.get(key, 0) + 1

    return total, correct, skipped_small, miscls


def eval_label_filewise(
    *,
    m: Magika,
    label: str,
    dir_path: Path,
    samples: int,
    window_size: int,
    batch_size: int,
    show_progress: bool,
    all_labels: Sequence[str],
    threshold: float,
) -> Tuple[int, int, int, Dict[str, int]]:
    """
    Per-file evaluation.

    For each sampled file, we create windows that *cover all bytes* of the file
    (1536-byte windows as above), and we count the file as 'correct' only if
    **every** window:
        - canonicalizes to the target label, and
        - satisfies score > threshold (if a score is available).

    Returns (files_total, files_correct, skipped_small, per_file_failure_modes)
    where per_file_failure_modes aggregates *file-level* failures by the most
    frequent failing prediction within each failed file.
    """
    files = choose_files(dir_path, samples, window_size)
    skipped_small = 0
    total_files = 0
    correct_files = 0
    file_fail_modes: Dict[str, int] = {}

    iterator = files if not show_progress else tqdm(
        files, total=len(files), desc=f"{label} (files)", unit="file", leave=False
    )

    for fp in iterator:
        try:
            size = fp.stat().st_size
        except Exception:
            # If we can't stat, skip as "small/unreadable"
            skipped_small += 1
            continue

        if size < window_size:
            skipped_small += 1
            continue

        total_files += 1
        fail_counts: Dict[str, int] = {}
        batch_bufs: List[bytes] = []

        # Stream windows, batching at the application level.
        for w in iter_windows_cover_all_bytes(fp, window_size):
            if not w:
                continue
            batch_bufs.append(w)
            if len(batch_bufs) >= batch_size:
                # Process batch
                for wb in batch_bufs:
                    res = m.identify_bytes(wb)
                    pred_label = extract_label_from_result(res)
                    pred_str = str(pred_label).strip().lower() if pred_label else "(none)"

                    ok = bool(getattr(res, "ok", False))
                    score = float(getattr(res, "score", 0.0)) if ok else 0.0

                    canon = canonicalize_label(pred_label, all_labels)
                    matches_label = canon == label
                    meets_threshold = score > threshold

                    if not (matches_label and meets_threshold):
                        if matches_label:
                            key = f"{pred_str} (below-threshold)"
                        else:
                            key = canon if canon != "(none)" else pred_str
                        fail_counts[key] = fail_counts.get(key, 0) + 1
                batch_bufs = []

        # Flush any remaining windows in the final (possibly small) batch
        if batch_bufs:
            for wb in batch_bufs:
                res = m.identify_bytes(wb)
                pred_label = extract_label_from_result(res)
                pred_str = str(pred_label).strip().lower() if pred_label else "(none)"

                ok = bool(getattr(res, "ok", False))
                score = float(getattr(res, "score", 0.0)) if ok else 0.0

                canon = canonicalize_label(pred_label, all_labels)
                matches_label = canon == label
                meets_threshold = score > threshold

                if not (matches_label and meets_threshold):
                    if matches_label:
                        key = f"{pred_str} (below-threshold)"
                    else:
                        key = canon if canon != "(none)" else pred_str
                    fail_counts[key] = fail_counts.get(key, 0) + 1

        # If there were no failures across the file's windows, it's a success.
        if not fail_counts:
            correct_files += 1
        else:
            # Attribute this file's failure to the predominant failing key.
            top_key = max(fail_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            file_fail_modes[top_key] = file_fail_modes.get(top_key, 0) + 1

    return total_files, correct_files, skipped_small, file_fail_modes


# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate Magika accuracy on random windows per label directory, plus per-file (all-windows) accuracy.")
    ap.add_argument("--root", type=Path, default=Path("stack_super_small"), help="Root folder containing label directories.")
    ap.add_argument("--dirs", type=str, default=",".join(DEFAULT_DIRS), help="Comma-separated label dirs to use.")
    ap.add_argument("--samples-per-dir", type=int, default=SAMPLES_PER_DIR, help="Max files per directory.")
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Window size in bytes.")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="App-level batch size (grouping).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for file sampling and window offsets.")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    ap.add_argument("--bar-width", type=int, default=BAR_WIDTH, help="ASCII bar width.")
    ap.add_argument("--topk-miscls", type=int, default=3, help="How many top misclassification classes to list per label.")
    ap.add_argument("--threshold", type=float, default=0.9, help="Min confidence score (0-1) to count a correct prediction.")
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
    if args.threshold > 0:
        print(f"Confidence threshold: > {args.threshold:.2f}")
    print("")

    # ONE Magika instance; main thread only.
    m = Magika()

    per_label_stats: Dict[str, Dict[str, object]] = {}
    overall_total_windows = 0
    overall_correct_windows = 0

    # Per-file overall accumulators
    overall_total_files = 0
    overall_correct_files = 0

    for label in labels:
        d, candidate_count = find_label_dir(root, label, window_size=args.window_size)
        if d and d != (root / label):
            print(f"[info] Using {d} ({candidate_count} files ≥ {args.window_size} bytes) for label '{label}'")
        if not d:
            per_label_stats[label] = {
                "total": 0, "correct": 0, "skipped_small": 0, "missing": 1, "miscls": {},
                "files_total": 0, "files_correct": 0, "files_skipped_small": 0, "files_miscls": {}
            }
            print(f"[warn] Missing directory: {root / label}")
            continue

        # ----- Per-window evaluation -----
        total_w, correct_w, skipped_small_w, miscls_w = eval_label(
            m=m,
            label=label,
            dir_path=d,
            samples=args.samples_per_dir,
            window_size=args.window_size,
            batch_size=args.batch_size,
            show_progress=not args.no_progress and sys.stderr.isatty(),
            all_labels=labels,
            threshold=float(args.threshold),
        )

        # ----- Per-file evaluation (all bytes must pass) -----
        total_f, correct_f, skipped_small_f, miscls_f = eval_label_filewise(
            m=m,
            label=label,
            dir_path=d,
            samples=args.samples_per_dir,
            window_size=args.window_size,
            batch_size=args.batch_size,
            show_progress=not args.no_progress and sys.stderr.isatty(),
            all_labels=labels,
            threshold=float(args.threshold),
        )

        per_label_stats[label] = {
            # windows
            "total": total_w,
            "correct": correct_w,
            "skipped_small": skipped_small_w,
            "missing": 0,
            "miscls": miscls_w,
            # files
            "files_total": total_f,
            "files_correct": correct_f,
            "files_skipped_small": skipped_small_f,
            "files_miscls": miscls_f,
        }

        overall_total_windows += total_w
        overall_correct_windows += correct_w

        overall_total_files += total_f
        overall_correct_files += correct_f

    # ---- ASCII bar chart + top misclassifications (per-window) ----
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

    if overall_total_windows > 0:
        overall_pct = overall_correct_windows / overall_total_windows
        print("")
        print(f"OVERALL (windows) | {ascii_bar(overall_pct)} | {overall_pct*100:5.1f}%  ({overall_correct_windows}/{overall_total_windows})")

    # ---- NEW: Per-file bar chart (all windows must pass) ----
    print("\nPer-file accuracy (all 1536-byte windows across each file must match the label and pass threshold).")
    print("A file counts as correct only if every covering window passes.\n")

    for label in labels:
        s = per_label_stats[label]
        f_total = int(s["files_total"])
        f_correct = int(s["files_correct"])
        f_skipped = int(s["files_skipped_small"])
        f_miscls: Dict[str, int] = s.get("files_miscls", {}) if isinstance(s, dict) else {}

        if f_total > 0:
            fpct = f_correct / f_total
            print(f"{label:12} | {ascii_bar(fpct)} | {fpct*100:5.1f}%  ({f_correct}/{f_total}, skipped_small={f_skipped})")
            if f_miscls:
                # Provide a brief look at predominant failure modes at the *file* level.
                topk_f = sorted(f_miscls.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, args.topk_miscls)]
                parts_f = [f"{k} {v/f_total*100:4.1f}% ({v}/{f_total})" for k, v in topk_f]
                print(" " * 15 + "↳ top file failures: " + ", ".join(parts_f))
        else:
            miss = " (missing dir)" if s.get("missing") else ""
            print(f"{label:12} | {'-'*BAR_WIDTH} |   n/a   (0/0, skipped_small={f_skipped}){miss}")

    if overall_total_files > 0:
        overall_file_pct = overall_correct_files / overall_total_files
        print("")
        print(f"OVERALL (files)   | {ascii_bar(overall_file_pct)} | {overall_file_pct*100:5.1f}%  ({overall_correct_files}/{overall_total_files})")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted.", file=sys.stderr)
        raise
