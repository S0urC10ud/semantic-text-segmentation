"""
Utilities for loading, preparing, and augmenting the dataset.
"""
import os

from typing import List, Tuple, Dict
import numpy as np
import datasets as hfds
from datasets import load_dataset
from datasets import load_from_disk

from config import LANG2ID

# ---------------------------
# Dataset utilities
# ---------------------------

def bytes_from_text(s: str) -> np.ndarray:
    # Returns uint8 array of 0..255; callers upcast to int32 as needed
    return np.frombuffer(s.encode("utf-8", "ignore"), dtype=np.uint8)

# -------- Local OR HF fallback per-language loading --------

def _read_text(fp: str, cap_chars: int = 262_144) -> str:
    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(cap_chars)
    except Exception:
        return ""

def _collect_files(root: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dirpath, fn))
    return out

def _build_local_dataset_for_lang(lang: str, data_root: str, split: str):
    # 1) Prefer prebuilt Arrow cache: <data_root>/arrow_cache/<split>/<lang>
    arrow_dir = os.path.join(data_root, "arrow_cache", split, lang)
    info_fp = os.path.join(arrow_dir, "dataset_info.json")
    if os.path.exists(info_fp):
        print(f"Loaded prebuilt Arrow for '{lang}' {split}: {arrow_dir}")
        return load_from_disk(arrow_dir)  # -> indexable HF Dataset


def prepare_dsets_by_lang_with_splits(data_root: str) -> Dict[str, Dict[int, hfds.Dataset]]:
    """
    Returns a nested dict: splits['train'|'val'|'test'][lang_id] -> dataset
    Prefers local folders under data_root; if missing and allow_hf_fallback==True, uses HF dataset.
    """
    splits = {"train": {}, "val": {}, "test": {}}
    for split in ("train", "val", "test"):
        for lang in LANG2ID.keys():
            lang_id = LANG2ID[lang]
            ds = _build_local_dataset_for_lang(lang, data_root, split)
            print(f"Loaded local '{lang}' {split} samples (streaming)...")
            splits[split][lang_id] = ds


    # Sanity check - just verify all languages are present
    for split_name, mp in splits.items():
        missing_langs = [lang for lang_id in LANG2ID.values() if lang_id not in mp]
        if missing_langs:
            raise ValueError(f"Split '{split_name}' is missing languages: {missing_langs}")
    return splits
