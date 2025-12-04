"""
Utilities for loading, preparing, and augmenting the dataset.
"""
import os
from typing import List, Tuple, Dict, Optional
import contextlib
import io
import numpy as np
import datasets as hfds
from datasets import load_dataset
from datasets import load_from_disk
from datasets import concatenate_datasets

import utils.config as cfg

LANG_ALIASES: Dict[str, List[str]] = {
    "c_family": ["c", "cpp"],
}

# ---------------------------
# Dataset utilities
# ---------------------------

def bytes_from_text(s: str) -> np.ndarray:
    # Returns uint8 array of 0..255; callers upcast to int32 as needed
    return np.frombuffer(s.encode("utf-8", "ignore"), dtype=np.uint8)

# -------- Data loading and validation --------

def _build_local_dataset_for_lang(
    lang: str,
    data_root: str,
    split: str,
    use_train_windows: bool = True,
    *,
    verbose: bool = False,
):
    """Load and analyze dataset for a specific language and split."""
    
    # Data is in <data_root>/<split>/<lang>/dataset
    arrow_dir = os.path.join(data_root, split, lang, "dataset")
    if not os.path.exists(arrow_dir):
        if verbose:
            print(f"❌ ERROR: No dataset found at {arrow_dir}")
        return None
    
    try:
        if verbose:
            print(f"📂 Loading Arrow dataset for '{lang}' {split} from: {arrow_dir}")
        ds = load_from_disk(arrow_dir)
        
        if len(ds) == 0:
            if verbose:
                print(f"⚠️  WARNING: Empty dataset for '{lang}' {split}")
            return None
            
        # Analyze content
        if "content" not in ds.features:
            if verbose:
                print(f"❌ ERROR: Dataset '{lang}' {split} missing 'content' column")
            return None
        
        # Sample size statistics (from first 100 samples)
        if verbose:
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
        if verbose:
            print(f"❌ ERROR loading dataset for '{lang}' {split}: {e}")
        return None


def _load_dataset_with_aliases(
    lang: str,
    data_root: str,
    split: str,
    use_train_windows: bool = True,
    *,
    verbose: bool = False,
) -> Tuple[Optional[hfds.Dataset], List[str]]:
    """
    Attempt to load the dataset for ``lang``; if not present, fall back to any alias
    directories defined in LANG_ALIASES (e.g., old 'c'/'cpp' buckets now merged into
    'c_family'). Returns the combined dataset (if any) plus the list of source labels
    that contributed data.
    """
    primary_path = os.path.join(data_root, split, lang, "dataset")
    if os.path.exists(primary_path):
        loader = _build_local_dataset_for_lang
        if verbose:
            primary = loader(lang, data_root, split, use_train_windows=use_train_windows, verbose=verbose)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                primary = loader(lang, data_root, split, use_train_windows=use_train_windows, verbose=verbose)
        if primary is not None and len(primary) > 0:
            return primary, [lang]

    alias_sources = []
    alias_datasets = []
    for alias in LANG_ALIASES.get(lang, []):
        alias_path = os.path.join(data_root, split, alias, "dataset")
        if not os.path.exists(alias_path):
            continue
        loader = _build_local_dataset_for_lang
        if verbose:
            alias_ds = loader(alias, data_root, split, use_train_windows=use_train_windows, verbose=verbose)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                alias_ds = loader(alias, data_root, split, use_train_windows=use_train_windows, verbose=verbose)
        if alias_ds is not None and len(alias_ds) > 0:
            alias_sources.append(alias)
            alias_datasets.append(alias_ds)

    if not alias_datasets:
        return None, []

    if len(alias_datasets) == 1:
        return alias_datasets[0], alias_sources

    combined = concatenate_datasets(alias_datasets)
    return combined, alias_sources


def prepare_dsets_by_lang_with_splits(
    data_root: str,
    use_train_windows: bool = True,
    include_languages: Optional[List[str]] = None,
    *,
    preserve_lang_order: bool = True,
    preferred_order: Optional[List[str]] = None,
    always_include_labels: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Dict[int, hfds.Dataset]]:
    """
    Returns a nested dict: splits['train'|'val'|'test'][lang_id] -> dataset
    Datasets can contain windows of varying sizes - padding/truncation happens during training.
    """
    if verbose:
        print("\n" + "="*60)
        print("🔍 SCANNING DATASETS IN:", data_root)
        print("="*60)
    
    if not os.path.exists(data_root):
        raise ValueError(f"❌ DATA ROOT NOT FOUND: {data_root}")

    optional_labels = set(always_include_labels or getattr(cfg, "OPTIONAL_LABELS", []))
    configured_langs = list(cfg.LANG2ID.keys())
    if preferred_order:
        configured_order = list(preferred_order)
    else:
        configured_order = [name for name, _ in sorted(cfg.LANG2ID.items(), key=lambda kv: kv[1])]
    canonical_map = {lang.lower(): lang for lang in configured_langs}
    for canonical, aliases in LANG_ALIASES.items():
        for alias in aliases:
            canonical_map.setdefault(alias.lower(), canonical)
    # Common textual variants for merged families
    canonical_map.setdefault("c++", "c_family")
    canonical_map.setdefault("c-family", "c_family")
    canonical_map.setdefault("cfamily", "c_family")
    requested_langs: Optional[List[str]] = None
    if include_languages:
        requested_langs = []
        invalid = []
        for lang in include_languages:
            key = lang.lower()
            if key in canonical_map:
                requested_langs.append(canonical_map[key])
            else:
                invalid.append(lang)
        if invalid:
            raise ValueError(
                f"❌ Unknown languages requested via --lang: {invalid}. "
                f"Available languages: {sorted(configured_langs)}"
            )
        # Deduplicate while preserving the configured order
        deduped = list(dict.fromkeys(requested_langs))
        requested_langs = [lang for lang in configured_order if lang in deduped]
        if not requested_langs:
            requested_langs = None
    
    target_langs = requested_langs if requested_langs is not None else configured_order

    # First scan: check existence and sample counts
    splits = {"train": {}, "val": {}, "test": {}}
    path_registry: Dict[str, str] = {}
    lang_stats = {lang: {"splits": {}, "total_samples": 0} for lang in target_langs}
    
    for split in ("train", "val", "test"):
        if verbose:
            print(f"\n📂 Checking {split} split...")
        for lang in target_langs:
            ds, source_labels = _load_dataset_with_aliases(
                lang,
                data_root,
                split,
                use_train_windows=use_train_windows,
                verbose=verbose,
            )
            if ds is not None and len(ds) > 0:
                lang_id = cfg.LANG2ID[lang]
                splits[split][lang_id] = ds
                lang_stats[lang]["splits"][split] = len(ds)
                lang_stats[lang]["total_samples"] += len(ds)
                if source_labels and source_labels != [lang]:
                    if verbose:
                        print(
                            f"   ℹ️  Loaded '{lang}' {split} data from legacy buckets: {', '.join(source_labels)}"
                        )
                for source_label in (source_labels or [lang]):
                    data_path = os.path.join(data_root, split, source_label, "dataset")
                    canonical_path = os.path.realpath(data_path)
                    prev_split = path_registry.get(canonical_path)
                    if prev_split is not None and prev_split != split:
                        raise ValueError(
                            f"❌ DATA LEAKAGE: dataset at {canonical_path} reused for both '{prev_split}' and '{split}' splits."
                        )
                    path_registry[canonical_path] = split
    
    # Print summary table
    if verbose:
        print("\n" + "="*80)
        print("📊 DATASET AVAILABILITY REPORT")
        print("="*80)
        print(f"{'Language':12} | {'Train':>10} | {'Val':>10} | {'Test':>10} | {'Total':>10} | Status")
        print("-" * 80)
    
    available_langs = set()
    all_missing = True
    for lang in sorted(target_langs):
        stats = lang_stats[lang]
        train_count = stats["splits"].get("train", 0)
        val_count = stats["splits"].get("val", 0)
        test_count = stats["splits"].get("test", 0)
        is_optional = lang in optional_labels
        
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
            status_prefix = "⚠️ OPTIONAL " if is_optional else ""
            status = status_prefix + " ".join(status_parts)
            if is_optional:
                # Keep optional labels in the mapping even if data is missing
                available_langs.add(lang)
                all_missing = False
        
        # Format counts with color indicators
        def fmt_count(n):
            return f"{n:>10}" if n > 0 else "     ----"
            
        if verbose:
            print(f"{lang:12} | {fmt_count(train_count)} | {fmt_count(val_count)} | {fmt_count(test_count)} | {stats['total_samples']:>10} | {status}")
    
    if all_missing:
        raise ValueError("\n" + "!"*80 + "\n" +
                       "❌ CRITICAL ERROR: NO VALID DATASETS FOUND!\n" +
                       f"- Checked in: {data_root}\n" +
                       "- Required structure: <data_root>/<split>/<language>/dataset\n" +
                       "- Each language needs at least train and val splits\n" +
                       "!"*80)

    if requested_langs is not None:
        missing_requested = set(requested_langs) - available_langs
        if missing_requested:
            raise ValueError(
                f"❌ Requested languages missing required splits: {sorted(missing_requested)}"
            )
    else:
        missing_required = [lang for lang in target_langs if lang not in available_langs and lang not in optional_labels]
        if missing_required:
            raise ValueError(
                f"❌ Required languages missing train/val data: {sorted(missing_required)} "
                f"(expected under {data_root})"
            )
    
    # Rebuild LANG2ID with only available languages
    cfg.LANG2ID.clear()
    if preserve_lang_order:
        ordered_langs = [lang for lang in configured_order if lang in available_langs]
    else:
        ordered_langs = sorted(available_langs)
    for i, lang in enumerate(ordered_langs):
        cfg.LANG2ID[lang] = i
    
    if verbose:
        print("\n" + "="*80)
        print(f"✅ USING {len(available_langs)} LANGUAGES: {sorted(available_langs)}")
        print(f"🔢 Language ID mapping: {dict(sorted(cfg.LANG2ID.items(), key=lambda x: x[1]))}")
        print("="*80 + "\n")
    
    # Build the final splits dict
    splits = {"train": {}, "val": {}, "test": {}}
    for split in ("train", "val", "test"):
        for lang, lang_id in cfg.LANG2ID.items():
            ds, _ = _load_dataset_with_aliases(
                lang,
                data_root,
                split,
                use_train_windows=use_train_windows,
                verbose=verbose,
            )
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
        for lang, lang_id in cfg.LANG2ID.items():
            ds, _ = _load_dataset_with_aliases(
                lang,
                data_root,
                split_name,
                use_train_windows=use_train_windows,
                verbose=verbose,
            )
            if ds is not None and len(ds) > 0:
                new_splits[split_name][lang_id] = ds
    
    if verbose:
        print(f"Found {len(cfg.LANG2ID)} available languages: {sorted(cfg.LANG2ID.keys())}")
        print(f"Language ID mapping: {dict(sorted(cfg.LANG2ID.items(), key=lambda x: x[1]))}")
    
    # Update the global mappings in config
    cfg.update_lang_mappings()
    return new_splits
