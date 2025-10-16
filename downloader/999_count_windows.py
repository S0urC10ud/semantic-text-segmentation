"""
count_windows.py — Count how many window files exist per label under:
    stack_web_sample/train_windows/<label>/**

Outputs a text-based bar chart showing the number of files per label.
"""

from pathlib import Path
import shutil

# --- Configuration ---
ROOT = Path("stack_web_sample/train_windows")
BAR_WIDTH = 50  # max width of bar in characters

def count_files_in_dir(d: Path) -> int:
    """Count all files recursively under directory d."""
    if not d.exists():
        return 0
    return sum(1 for p in d.rglob("*") if p.is_file())

def main():
    if not ROOT.exists():
        print(f"[!] Directory not found: {ROOT}")
        return

    # Count files per label (subdirectory of train_windows)
    counts = {}
    for label_dir in sorted([d for d in ROOT.iterdir() if d.is_dir()]):
        count = count_files_in_dir(label_dir)
        counts[label_dir.name] = count

    if not counts:
        print("No label directories found under train_windows/")
        return

    # Determine scaling
    max_label_len = max(len(k) for k in counts)
    max_count = max(counts.values())
    term_width = shutil.get_terminal_size((100, 20)).columns
    bar_max = min(BAR_WIDTH, term_width - max_label_len - 20)
    scale = bar_max / max_count if max_count > 0 else 1.0

    print(f"\n📊 Window File Counts per Label in: {ROOT}\n")
    total = 0
    for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        bar_len = int(count * scale)
        bar = "▰" * bar_len
        print(f"{label.ljust(max_label_len)} | {bar.ljust(bar_max)} | {count:,}")
        total += count

    print(f"\nTotal window files: {total:,}\n")

if __name__ == "__main__":
    main()
