"""
Utilities for loading, preparing, and augmenting the dataset.
"""
import os
from typing import List, Tuple, Dict, Optional
import numpy as np
import datasets as hfds
from datasets import load_dataset
from datasets import load_from_disk

from config import LANG2ID, update_lang_mappings

# ---------------------------
# Dataset utilities
# ---------------------------

def bytes_from_text(s: str) -> np.ndarray:
    # Returns uint8 array of 0..255; callers upcast to int32 as needed
    return np.frombuffer(s.encode("utf-8", "ignore"), dtype=np.uint8)

# -------- Data loading and validation --------

def _build_local_dataset_for_lang(lang: str, data_root: str, split: str, use_train_windows = True):
    """Load and analyze dataset for a specific language and split."""
    
    # Data is in <data_root>/<split>/<lang>/dataset
    arrow_dir = os.path.join(data_root, split, lang, "dataset")
    if not os.path.exists(arrow_dir):
        print(f"❌ ERROR: No dataset found at {arrow_dir}")
        return None
    
    try:
        print(f"📂 Loading Arrow dataset for '{lang}' {split} from: {arrow_dir}")
        ds = load_from_disk(arrow_dir)
        
        if len(ds) == 0:
            print(f"⚠️  WARNING: Empty dataset for '{lang}' {split}")
            return None
            
        # Analyze content
        if "content" not in ds.features:
            print(f"❌ ERROR: Dataset '{lang}' {split} missing 'content' column")
            return None
        
        # Sample size statistics (from first 100 samples)
        sample_size = min(100, len(ds))
        sizes = [len(bytes_from_text(ex["content"])) for ex in ds.select(range(sample_size))]
        min_size = min(sizes)
        max_size = max(sizes)
        avg_size = sum(sizes) / len(sizes)
        
        print(f"✅ Loaded {len(ds)} samples for '{lang}' {split}")
        print(f"   Content sizes (bytes): min={min_size}, max={max_size}, avg={avg_size:.1f}")
        
        if max_size > 1536:
            print(f"⚠️  WARNING: Some content exceeds target window size (1536 bytes)")
            print(f"   These will be truncated during training")
        
        return ds
        
    except Exception as e:
        print(f"❌ ERROR loading dataset for '{lang}' {split}: {e}")
        return None


def prepare_dsets_by_lang_with_splits(data_root: str, use_train_windows: bool = True) -> Dict[str, Dict[int, hfds.Dataset]]:
    """
    Returns a nested dict: splits['train'|'val'|'test'][lang_id] -> dataset
    Datasets can contain windows of varying sizes - padding/truncation happens during training.
    """
    print("\n" + "="*60)
    print("🔍 SCANNING DATASETS IN:", data_root)
    print("="*60)
    
    if not os.path.exists(data_root):
        raise ValueError(f"❌ DATA ROOT NOT FOUND: {data_root}")
        
    # First scan: check existence and sample counts
    splits = {"train": {}, "val": {}, "test": {}}
    lang_stats = {lang: {"splits": {}, "total_samples": 0} for lang in LANG2ID}
    
    for split in ("train", "val", "test"):
        print(f"\n📂 Checking {split} split...")
        for lang in LANG2ID:
            ds = _build_local_dataset_for_lang(lang, data_root, split, use_train_windows=use_train_windows)
            if ds is not None and len(ds) > 0:
                lang_id = LANG2ID[lang]
                splits[split][lang_id] = ds
                lang_stats[lang]["splits"][split] = len(ds)
                lang_stats[lang]["total_samples"] += len(ds)
    
    # Print summary table
    print("\n" + "="*80)
    print("📊 DATASET AVAILABILITY REPORT")
    print("="*80)
    print(f"{'Language':12} | {'Train':>10} | {'Val':>10} | {'Test':>10} | {'Total':>10} | Status")
    print("-" * 80)
    
    available_langs = set()
    all_missing = True
    for lang in sorted(LANG2ID.keys()):
        stats = lang_stats[lang]
        train_count = stats["splits"].get("train", 0)
        val_count = stats["splits"].get("val", 0)
        test_count = stats["splits"].get("test", 0)
        
        # Determine status
        status_parts = []
        if train_count == 0:
            status_parts.append("❌ NO TRAIN")
        if val_count == 0:
            status_parts.append("❌ NO VAL")
        if test_count == 0:
            status_parts.append("⚠️ NO TEST")
            
        if not status_parts:
            status = "✅ Complete"
            available_langs.add(lang)
            all_missing = False
        else:
            status = " ".join(status_parts)
            
        # Format counts with color indicators
        def fmt_count(n):
            return f"{n:>10}" if n > 0 else "     ----"
            
        print(f"{lang:12} | {fmt_count(train_count)} | {fmt_count(val_count)} | {fmt_count(test_count)} | {stats['total_samples']:>10} | {status}")
    
    if all_missing:
        raise ValueError("\n" + "!"*80 + "\n" +
                       "❌ CRITICAL ERROR: NO VALID DATASETS FOUND!\n" +
                       f"- Checked in: {data_root}\n" +
                       "- Required structure: <data_root>/<split>/<language>/dataset\n" +
                       "- Each language needs at least train and val splits\n" +
                       "!"*80)
    
    # Rebuild LANG2ID with only available languages
    LANG2ID.clear()
    for i, lang in enumerate(sorted(available_langs)):
        LANG2ID[lang] = i
        
    print("\n" + "="*80)
    print(f"✅ USING {len(available_langs)} LANGUAGES: {sorted(available_langs)}")
    print(f"🔢 Language ID mapping: {dict(sorted(LANG2ID.items(), key=lambda x: x[1]))}")
    print("="*80 + "\n")
    
    # Build the final splits dict
    splits = {"train": {}, "val": {}, "test": {}}
    for split in ("train", "val", "test"):
        for lang, lang_id in LANG2ID.items():
            ds = _build_local_dataset_for_lang(lang, data_root, split, use_train_windows=use_train_windows)
            if ds is not None and len(ds) > 0:
                splits[split][lang_id] = ds
                
    # Final validation
    if len(splits["train"]) == 0:
        raise ValueError("\n" + "!"*80 + "\n" +
                       "❌ CRITICAL ERROR: No training data loaded!\n" +
                       "This should not happen - implementation error.\n" +
                       "!"*80)
    if len(splits["val"]) == 0:
        raise ValueError("\n" + "!"*80 + "\n" +
                       "❌ CRITICAL ERROR: No validation data loaded!\n" +
                       "This should not happen - implementation error.\n" +
                       "!"*80)
    
    # Rebuild the splits dictionary with new language IDs
    new_splits = {"train": {}, "val": {}, "test": {}}
    for split_name, mp in splits.items():
        for lang, lang_id in LANG2ID.items():
            ds = _build_local_dataset_for_lang(lang, data_root, split_name, use_train_windows=use_train_windows)
            if ds is not None and len(ds) > 0:
                new_splits[split_name][lang_id] = ds
    
    print(f"Found {len(LANG2ID)} available languages: {sorted(LANG2ID.keys())}")
    print(f"Language ID mapping: {dict(sorted(LANG2ID.items(), key=lambda x: x[1]))}")
    
    # Update the global mappings in config
    update_lang_mappings()
    return new_splits
