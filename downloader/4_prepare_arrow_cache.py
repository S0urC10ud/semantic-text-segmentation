import os
import argparse
from typing import List
from tqdm import tqdm
from datasets import Dataset, Features, Value, load_from_disk
import numpy as np

LANG2ID = {"html": 0, "css": 1, "javascript": 2, "c":3, "cpp": 4, "csv":5, "java":6, "json":7, "python":8, "text":9}

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

def build_split_lang(data_root: str, split: str, lang: str, out_root: str, cap_chars: int):
    lang_dir = os.path.join(data_root, split, lang)
    if not os.path.isdir(lang_dir):
        raise FileNotFoundError(f"Missing: {lang_dir}")

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
        pbar = tqdm(files, desc=f"Building {split}/{lang}", unit="file")
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root that contains {train,val,test}/lang")
    ap.add_argument("--out_root", type=str, default=None,
                    help="Where to save Arrow datasets; default: <data_root>/arrow_cache")
    ap.add_argument("--cap_chars", type=int, default=262_144,
                    help="Max chars read per file (keeps disk size reasonable)")
    args = ap.parse_args()

    out_root = args.out_root or os.path.join(args.data_root, "arrow_cache")
    os.makedirs(out_root, exist_ok=True)

    for split in ("train", "val", "test"):
        for lang in LANG2ID.keys():
            build_split_lang(args.data_root, split, lang, out_root, args.cap_chars)

if __name__ == "__main__":
    main()
