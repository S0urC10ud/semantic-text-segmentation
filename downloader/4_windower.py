"""
windower.py — Slice TRAIN files into fixed byte windows, keep only those
that Magika says match the directory label with confidence >= threshold.

Key features:
  • Only processes <root>/train/<label>/** (as requested)
  • NEW: --labels "text,html,css" to limit processing to specific labels
  • NEW: --rebuild to delete existing windows for selected labels before writing
  • Default window size: 1536 bytes, non-overlapping
  • DEFAULT: prefilter with Magika (identify_bytes) before writing
    -> massively reduces IO/memory and avoids OOM ("killed")
  • Optional post-write batch mode with --no-prefilter (kept for parity)
  • Single-line progress bar with ETA (by windows, not bytes)

Output layout:
  <root>/train_windows/<label>/**/<file>__winXXXXXX.bin
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Keep native threadpools from over-subscribing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from magika import Magika  # magika==0.6.2

# ---------------- Label matching (aligned with your cleaner) ----------------

_DIR_ACCEPTS: Dict[str, set] = {
    "text": {"txt"},         # Magika’s generic text label vs your "text"
    # Others: strict equality
}

def dir_matches_label(dir_label: Optional[str], label: Optional[str], mime: Optional[str]) -> bool:
    """
    True if Magika's label is considered a match for the directory.
    Alias table first; else strict equality.
    """
    if dir_label is None or not label:
        return False
    dl = dir_label.lower()
    ll = str(label).lower()
    allowed = _DIR_ACCEPTS.get(dl)
    if allowed is not None:
        return ll in allowed
    return ll == dl

# ---------------- Progress helpers ----------------

def _fmt_int(n: int) -> str:
    return f"{n:,}"

def _fmt_rate(items: float) -> str:
    if items < 1:
        return f"{items:.2f}/s"
    if items < 10:
        return f"{items:.1f}/s"
    return f"{int(items):d}/s"

def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds):
        return "ETA --:--"
    m, s = divmod(int(seconds + 0.5), 60)
    h, m = divmod(m, 60)
    if h:
        return f"ETA {h:02d}:{m:02d}:{s:02d}"
    return f"ETA {m:02d}:{s:02d}"

def _render_bar(progress: float, width: int = 24) -> str:
    progress = 0 if math.isnan(progress) else max(0.0, min(1.0, progress))
    filled = int(progress * width + 0.5)
    filled = min(filled, width)
    return "▰" * filled + "▱" * (width - filled)

def _update_progress_line(*, processed: int, total: int, written: int, rejected: int,
                          start_time: float, label: str = "") -> None:
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(total - processed, 0)
    eta = remaining / rate if rate > 0 else float("inf")
    cols = os.get_terminal_size().columns if sys.stderr.isatty() else 100
    bar = _render_bar(processed / total if total else 0.0, width=24)
    parts = [
        bar,
        f"{_fmt_int(processed)}/{_fmt_int(total)} windows",
        f"| written {_fmt_int(written)}",
        f"| rejected {_fmt_int(rejected)}",
        f"| {_fmt_rate(rate)}",
        f"| {_fmt_eta(eta)}",
    ]
    if label:
        parts.insert(0, f"[{label}]")
    line = "  ".join(parts)
    if len(line) > cols:
        line = line[: max(10, cols - 1)]
    sys.stderr.write("\r" + line + " " * max(0, cols - len(line) - 1))
    sys.stderr.flush()

# ---------------- Train file discovery ----------------

def iter_train_files(root: Path, allowed_labels: Optional[Set[str]] = None) -> Iterable[Tuple[str, Path, Path]]:
    """
    Yield (dir_label, abs_path, rel_under_label) for files under <root>/train/<label>/**,
    optionally restricted to a set of allowed_labels.
    """
    train_root = root / "train"
    if not train_root.exists() or not train_root.is_dir():
        print(f"[!] Train split not found: {train_root}", file=sys.stderr)
        return
    for label_dir in train_root.iterdir():
        if not label_dir.is_dir():
            continue
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

# ---------------- Paths & writing ----------------

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def window_dest_path(out_root: Path, dir_label: str, rel_under_label: Path, win_idx: int) -> Path:
    base = out_root / dir_label / rel_under_label
    name = f"{base.name}__win{win_idx:06d}.bin"
    return base.with_name(name)

# ---------------- Classification & pruning (post-write mode) ----------------

def classify_and_filter_paths(
    magika: Magika,
    batch_paths_with_labels: List[Tuple[str, str]],
    *,
    threshold: float
) -> Tuple[int, int]:
    """
    Post-write mode: run Magika on written window files and delete non-matching/low-confidence ones.
    Returns (kept, deleted).
    """
    if not batch_paths_with_labels:
        return (0, 0)

    paths = [p for (p, _lbl) in batch_paths_with_labels]
    kept = 0
    deleted = 0

    try:
        results = magika.identify_paths(paths)
    except Exception as e:
        print(f"\n[ERROR] Magika identify_paths failed on {len(paths)} paths: {e}", file=sys.stderr)
        # Conservative: delete all
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
                deleted += 1
            except Exception as de:
                print(f"[ERROR] Deleting '{p}' after Magika failure: {de}", file=sys.stderr)
        return (kept, deleted)

    for res, (p, dir_label) in zip(results, batch_paths_with_labels):
        try:
            if not getattr(res, "ok", False):
                Path(p).unlink(missing_ok=True)
                deleted += 1
                continue
            out = getattr(res, "output", None)
            label = getattr(out, "label", None) if out else None
            mime = getattr(out, "mime_type", None) if out else None
            conf = float(getattr(res, "score", 0.0))  # magika==0.6.2
            if dir_matches_label(dir_label, label, mime) and conf >= threshold:
                kept += 1
            else:
                Path(p).unlink(missing_ok=True)
                deleted += 1
        except Exception as e:
            try:
                Path(p).unlink(missing_ok=True)
                deleted += 1
            except Exception as de:
                print(f"[ERROR] Deleting '{p}' after exception: {de}", file=sys.stderr)
            print(f"[ERROR] Processing result for '{p}': {e}", file=sys.stderr)

    return (kept, deleted)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create 1536-byte windows from TRAIN files and keep only those whose "
            "Magika output.label matches the directory label with confidence ≥ threshold."
        )
    )
    ap.add_argument("--root", type=Path, default=Path("stack_web_sample"),
                    help="Root dataset directory (default: stack_web_sample)")
    ap.add_argument("--window-size", type=int, default=1536,
                    help="Window size in bytes (default: 1536)")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="Keep only if Magika score ≥ threshold (default: 0.80)")
    ap.add_argument("--out-split-name", type=str, default="train_windows",
                    help="Output split name under root (default: train_windows)")
    ap.add_argument("--batch-size", type=int, default=4096,
                    help="POST-WRITE mode only: window files per Magika batch (default: 4096)")
    ap.add_argument("--no-prefilter", action="store_true",
                    help="Disable in-memory classification before writing (not recommended)")
    ap.add_argument("--progress", choices=["auto", "always", "never"], default="auto",
                    help="Show a live progress line (default: auto)")
    ap.add_argument("--labels", type=str, default=None,
                    help="Comma-separated list of TRAIN label dirs to process (e.g., 'text,html'); default: all")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete existing windows under the selected labels before processing")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        ap.error(f"Root directory not found: {root}")

    train_root = root / "train"
    if not train_root.exists() or not train_root.is_dir():
        ap.error(f"Train split not found at: {train_root}")

    out_root = root / args.out_split_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Discover all available labels in train/
    all_labels = sorted([d.name for d in train_root.iterdir() if d.is_dir()])

    # Parse and validate --labels
    allowed_labels: Optional[Set[str]] = None
    if args.labels:
        requested = {x.strip() for x in args.labels.split(",") if x.strip()}
        unknown = requested - set(all_labels)
        if unknown:
            print(f"[!] Ignoring unknown label(s) (not found under train/): {', '.join(sorted(unknown))}", file=sys.stderr)
        allowed_labels = requested & set(all_labels)
        if not allowed_labels:
            ap.error("No valid labels to process after filtering. Check --labels.")
    else:
        allowed_labels = set(all_labels)

    # Optionally rebuild (delete output for selected labels)
    if args.rebuild:
        for lbl in sorted(allowed_labels):
            target = out_root / lbl
            if target.exists():
                print(f"[rebuild] Removing existing windows: {target}")
                shutil.rmtree(target, ignore_errors=True)

    # Gather files for the selected labels and estimate total windows for progress
    files: List[Tuple[str, Path, Path]] = list(iter_train_files(root, allowed_labels))
    if not files:
        print("No train files found for the selected label(s).")
        return

    total_est_windows = 0
    for _label, fpath, _rel in files:
        try:
            size = fpath.stat().st_size
        except Exception:
            size = 0
        total_est_windows += math.ceil(size / args.window_size) if size > 0 else 0
    if total_est_windows == 0:
        total_est_windows = 1  # avoid division by zero

    # Progress setup
    is_tty = sys.stderr.isatty()
    show_progress = ((args.progress == "always") or (args.progress == "auto" and is_tty))
    start_time = time.time()
    processed_windows = 0
    written_windows = 0
    rejected_windows = 0

    def tick(label: str = ""):
        if show_progress:
            _update_progress_line(processed=processed_windows,
                                  total=total_est_windows,
                                  written=written_windows,
                                  rejected=rejected_windows,
                                  start_time=start_time,
                                  label=label)

    m = Magika()
    prefilter = (not args.no_prefilter)

    # For post-write mode
    post_batch: List[Tuple[str, str]] = []
    def flush_post_batch():
        nonlocal post_batch, written_windows, rejected_windows
        if not post_batch:
            return
        kept, deleted = classify_and_filter_paths(m, post_batch, threshold=args.threshold)
        # we already counted all windows as written; deleted ones are removed now
        rejected_windows += deleted
        written_windows -= deleted  # only kept remain on disk
        post_batch = []

    try:
        for dir_label, abs_path, rel_under_label in files:
            try:
                with abs_path.open("rb") as f:
                    idx = 0
                    while True:
                        chunk = f.read(args.window_size)
                        if not chunk:
                            break
                        processed_windows += 1

                        if prefilter:
                            # classify bytes before writing
                            try:
                                res = m.identify_bytes(chunk)
                                if not getattr(res, "ok", False):
                                    rejected_windows += 1
                                    tick(dir_label)
                                    continue
                                out = getattr(res, "output", None)
                                label = getattr(out, "label", None) if out else None
                                mime = getattr(out, "mime_type", None) if out else None
                                conf = float(getattr(res, "score", 0.0))
                                if not (dir_matches_label(dir_label, label, mime) and conf >= args.threshold):
                                    rejected_windows += 1
                                    tick(dir_label)
                                    continue
                            except Exception:
                                # On classifier error, reject conservatively
                                rejected_windows += 1
                                tick(dir_label)
                                continue

                            # write only passing window
                            dst = window_dest_path(out_root, dir_label, rel_under_label, idx)
                            ensure_parent(dst)
                            if dst.exists():
                                # avoid collision (unlikely)
                                c = 1
                                stem = dst.stem
                                suff = "".join(dst.suffixes)
                                alt = dst.with_name(f"{stem}__dup{c}{suff}")
                                while alt.exists():
                                    c += 1
                                    alt = dst.with_name(f"{stem}__dup{c}{suff}")
                                dst = alt
                            with dst.open("wb") as out_f:
                                out_f.write(chunk)
                            written_windows += 1
                            tick(dir_label)

                        else:
                            # post-write mode: write first, then batch-classify & delete
                            dst = window_dest_path(out_root, dir_label, rel_under_label, idx)
                            ensure_parent(dst)
                            if dst.exists():
                                c = 1
                                stem = dst.stem
                                suff = "".join(dst.suffixes)
                                alt = dst.with_name(f"{stem}__dup{c}{suff}")
                                while alt.exists():
                                    c += 1
                                    alt = dst.with_name(f"{stem}__dup{c}{suff}")
                                dst = alt
                            with dst.open("wb") as out_f:
                                out_f.write(chunk)
                            written_windows += 1
                            post_batch.append((str(dst), dir_label))
                            if len(post_batch) >= args.batch_size:
                                flush_post_batch()
                            tick(dir_label)

                        idx += 1
            except Exception as e:
                print(f"\n[ERROR] Reading '{abs_path}': {e}", file=sys.stderr)
                tick(dir_label)

        # Final cleanup for post-write mode
        if not prefilter:
            flush_post_batch()

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.", file=sys.stderr)

    # Final progress line + summary
    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()

    print("🎯 Summary")
    print(f"  Labels processed:       {', '.join(sorted(allowed_labels))}")
    print(f"  Estimated windows:      {_fmt_int(total_est_windows)}")
    print(f"  Processed windows:      {_fmt_int(processed_windows)}")
    print(f"  Windows written/kept:   {_fmt_int(written_windows)}")
    print(f"  Windows rejected:       {_fmt_int(rejected_windows)}")
    print(f"  Mode: {'prefilter' if prefilter else 'post-write'}")
    if args.rebuild:
        print("  Rebuild: output for selected labels was cleared before processing")

if __name__ == "__main__":
    main()
