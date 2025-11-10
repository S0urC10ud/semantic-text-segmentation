

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import Counter
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

def evaluate_label(
    *,
    m: Magika,
    label: str,
    files: Sequence[Path],
    window_size: int,
    batch_size: int,
    show_progress: bool,
    all_labels: Sequence[str],
    threshold: float,
) -> Tuple[Dict[str, object], Counter]:
    """
    Evaluate random-window accuracy (one randomly selected window per file) and
    per-file accuracy (every window must pass) on the *same* sampled files.

    Returns ``(stats_dict, pred_counter)`` where ``stats_dict`` retains the
    legacy schema and ``pred_counter`` tallies canonical predictions for the
    random-window sample (values include ``"(other)"`` for off-manifold outputs).
    """
    _ = batch_size  # retained for CLI compatibility; batching handled per file.

    label_set = set(all_labels)
    skipped_small = 0
    files_skipped_small = 0
    total_windows = 0
    correct_windows = 0
    per_window_miscls: Dict[str, int] = {}
    total_files = 0
    correct_files = 0
    file_fail_modes: Dict[str, int] = {}
    pred_counter: Counter[str] = Counter()

    iterator = files if not show_progress else tqdm(
        files, total=len(files), desc=f"{label} (files)", unit="file", leave=False
    )

    for fp in iterator:
        try:
            size = fp.stat().st_size
        except Exception:
            skipped_small += 1
            files_skipped_small += 1
            continue

        if size < window_size:
            skipped_small += 1
            files_skipped_small += 1
            continue

        fail_counts: Dict[str, int] = {}
        windows_seen = 0
        random_choice: Optional[Tuple[bool, Optional[str], str]] = None

        for wb in iter_windows_cover_all_bytes(fp, window_size):
            if not wb:
                continue

            res = m.identify_bytes(wb)
            pred_label = extract_label_from_result(res)
            pred_str = str(pred_label).strip().lower() if pred_label else "(none)"

            ok = bool(getattr(res, "ok", False))
            score = float(getattr(res, "score", 0.0)) if ok else 0.0

            canon = canonicalize_label(pred_label, all_labels)
            matches_label = canon == label
            meets_threshold = score > threshold
            success = matches_label and meets_threshold

            fail_key: Optional[str] = None
            if not success:
                if matches_label:
                    fail_key = f"{pred_str} (below-threshold)"
                else:
                    fail_key = canon if canon != "(none)" else pred_str
                fail_counts[fail_key] = fail_counts.get(fail_key, 0) + 1

            windows_seen += 1
            if random_choice is None or random.randrange(windows_seen) == 0:
                random_choice = (success, fail_key, canon)

        if windows_seen == 0 or random_choice is None:
            skipped_small += 1
            files_skipped_small += 1
            continue

        total_files += 1
        total_windows += 1
        success, fail_key, canon = random_choice
        if success:
            correct_windows += 1
        else:
            key = fail_key or "(unknown)"
            per_window_miscls[key] = per_window_miscls.get(key, 0) + 1

        pred_counter[canon if canon in label_set else "(other)"] += 1

        if not fail_counts:
            correct_files += 1
        else:
            top_key = max(fail_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            file_fail_modes[top_key] = file_fail_modes.get(top_key, 0) + 1

    stats = {
        "total": total_windows,
        "correct": correct_windows,
        "skipped_small": skipped_small,
        "missing": 0,
        "miscls": per_window_miscls,
        "files_total": total_files,
        "files_correct": correct_files,
        "files_skipped_small": files_skipped_small,
        "files_miscls": file_fail_modes,
    }
    return stats, pred_counter


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
    pred_buckets = labels + ["(other)"]
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    pred_to_idx = {lbl: idx for idx, lbl in enumerate(pred_buckets)}
    confusion = [[0 for _ in pred_buckets] for _ in labels]

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

        files = choose_files(d, args.samples_per_dir, args.window_size)
        stats, pred_counts = evaluate_label(
            m=m,
            label=label,
            files=files,
            window_size=args.window_size,
            batch_size=args.batch_size,
            show_progress=not args.no_progress and sys.stderr.isatty(),
            all_labels=labels,
            threshold=float(args.threshold),
        )
        per_label_stats[label] = stats

        row_idx = label_to_idx[label]
        for pred_label, count in pred_counts.items():
            col_idx = pred_to_idx.get(pred_label, pred_to_idx["(other)"])
            confusion[row_idx][col_idx] += int(count)

        total_w = int(stats["total"])
        correct_w = int(stats["correct"])
        total_f = int(stats["files_total"])
        correct_f = int(stats["files_correct"])

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

    total_samples = sum(sum(row) for row in confusion)
    if total_samples > 0:
        print("\nPer-label confusion counts (random window sample):")
        for label in labels:
            row_idx = label_to_idx[label]
            row = confusion[row_idx]
            tp = row[pred_to_idx[label]]
            fn = sum(row) - tp
            col_idx = pred_to_idx[label]
            fp = sum(confusion[r][col_idx] for r in range(len(labels))) - tp
            tn = total_samples - (tp + fp + fn)
            print(f"{label:12} TP={tp:5d} TN={tn:5d} FP={fp:5d} FN={fn:5d}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted.", file=sys.stderr)
        raise
