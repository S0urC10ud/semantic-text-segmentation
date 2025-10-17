"""
End-to-end The Stack → filtered windows → Arrow datasets (per label) with 70/20/10 splits

This version ALWAYS writes three disjoint datasets per content type (label) to:
  <out-root>/<split>/<label>/dataset
    - e.g., arrow_out/train/php/dataset
            arrow_out/val/php/dataset
            arrow_out/test/php/dataset

What this does (single pass per label, no temp raw files):
  • Streams from BigCode "bigcode/the-stack" per language folder (no full download).
  • Strict license filter: keeps permissive families only (MIT/Apache/BSD/Unlicense).
  • Windows content to fixed-size byte chunks (default: 1536 bytes).
  • Magika pre-filtering in batches (in-process; no multiprocessing issues).
  • PHP: strips all non-PHP blocks via regex.
  • Writes **only** final Arrow datasets to disk (train/val/test per label).
  • Default cap: **1,000,000 kept windows per label** (post-filter). `--demo` → 100 per label (total).
  • Robust streaming controls: shard/offset/skip to jump ahead in huge sorted datasets.
  • Detailed progress bars & summaries (tqdm) and periodic logging.

Resume, dedupe & splits:
  • Each window gets a stable content hash `uid`.
  • Split assignment is a pure function of `uid` → deterministic 70/20/10 buckets.
  • Resume loads existing train/val/test datasets, unions their UIDs, and skips dups on the fly.
  • New data are written to a temporary directory, then atomically swapped in per split.

Example:
  python build_stack_windows_arrow_splits.py \
      --out-root arrow_out \
      --langs php,csharp,typescript,go,sql,rust,yaml,ruby \
      --max-windows-per-label 200000 \
      --window-bytes 1536 --magika-batch 1024 --threshold 0.82 \
      --shard-count 64 --shard-index 17 \
      --skip-per-lang php=500000,csharp=200000 --use-auth-token
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Set

# Keep native threadpools from over-subscribing
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 3rd-party deps expected:
#   datasets>=2.14, magika==0.6.*, numpy, tqdm
from datasets import Dataset, Features, Value, load_dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm
import numpy as np


# ============================================================
#                   Utility / Canonicalization
# ============================================================

def safe_filename(name: str) -> str:
    s = (name or "")
    if s.lower() == "c++":
        return "cpp"
    if s.lower() in ("c#", "c-sharp", "csharp", "cs"):  # accept cs
        return "csharp"
    if s.lower() in ("yml",):
        return "yaml"
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)


def canonical_label(name: str) -> str:
    s = (name or "").strip().lower()
    if s in {"c++", "cpp"}:
        return "cpp"
    if s in {"c#", "c-sharp", "csharp", "cs"}:
        return "csharp"
    if s in {"js", "javascript"}:
        return "javascript"
    if s in {"ts", "typescript"}:
        return "typescript"
    if s in {"yml", "yaml"}:
        return "yaml"
    return s


# Magika may return different but equivalent labels; accept these as matches.
LABEL_ACCEPTS: Dict[str, set] = {
    "text": {"txt", "text"},
    "cpp": {"cpp", "c++"},
    "csv": {"csv"},
    "csharp": {"c#", "csharp", "c-sharp", "cs"},
    "javascript": {"javascript", "js"},
    "typescript": {"typescript", "ts"},
    "yaml": {"yaml", "yml"},
    "php": {"php"},
    "go": {"go"},
    "sql": {"sql"},
    "rust": {"rust"},
    "ruby": {"ruby"},
    "python": {"python"},
    "java": {"java"},
    "c": {"c"},
    "json": {"json"},
    "css": {"css"},
    "html": {"html", "xhtml"},
}

def label_matches_target(target: str, magika_label: Optional[str], mime: Optional[str]) -> bool:
    if not magika_label:
        return False
    tgt = canonical_label(target)
    ml = canonical_label(magika_label)
    accepts = LABEL_ACCEPTS.get(tgt, {tgt})
    return ml in accepts


# ============================================================
#                     License filtering
# ============================================================

_ALLOWED_FAMILIES = {"mit", "apache", "bsd", "unlicense", "0bsd", "mit-0"}
_DISALLOWED_KEYWORDS = {
    "gpl", "agpl", "lgpl", "mpl", "epl", "cdla", "cddl", "artistic",
    "cern", "cecill", "affero", "proprietary", "arr", "cc"
}

def _normalize_license_string(s: str) -> str:
    return re.sub(r"[^a-z0-9.+-]+", " ", (s or "").lower())

def _iter_license_fields(example: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    preferred = [
        "max_stars_repo_license",
        "max_forks_repo_license",
        "max_issues_repo_license",
        "max_stars_repo_licenses",
        "max_forks_repo_licenses",
        "max_issues_repo_licenses",
        "licenses",
        "license",
    ]
    seen = set()
    for k in preferred:
        if k in example and k not in seen and example.get(k) is not None:
            seen.add(k)
            yield k, example[k]
    for k, v in example.items():
        if k in seen:
            continue
        if "license" in k.lower() and v is not None:
            yield k, v

def _extract_license_strings(example: Dict[str, Any]) -> List[str]:
    vals: List[str] = []
    for _, v in _iter_license_fields(example):
        if isinstance(v, list):
            vals.extend([str(x) for x in v if x is not None])
        else:
            vals.append(str(v))
    return vals

def license_is_allowed(example: Dict[str, Any]) -> Tuple[bool, str]:
    lic_strings = _extract_license_strings(example)
    if not lic_strings:
        return False, "no_license_info"
    any_allowed = False
    for raw in lic_strings:
        s = _normalize_license_string(raw)
        if any(k in s for k in _DISALLOWED_KEYWORDS):
            return False, "disallowed_license"
        if any(k in s for k in _ALLOWED_FAMILIES):
            any_allowed = True
    if not any_allowed:
        return False, "no_allowed_license"
    return True, ""


# ============================================================
#               PHP foreign content removal (regex)
# ============================================================

_PHP_BLOCK_RE = re.compile(
    r"(?is)<\?(?!xml)(?:php|=)?(.*?)\?>"
)
_ASP_PHP_BLOCK_RE = re.compile(
    r"(?is)<%(.*?)%>"
)

def extract_php_code_only(text: str) -> str:
    """
    Keep ONLY PHP code regions, drop all HTML/other template text.
    """
    if not text:
        return ""
    parts: List[str] = []
    for m in _PHP_BLOCK_RE.finditer(text):
        inner = m.group(1)
        if inner is not None:
            parts.append(inner)
    if not parts:
        for m in _ASP_PHP_BLOCK_RE.finditer(text):
            inner = m.group(1)
            if inner is not None:
                parts.append(inner)
    sample = text[:4096].lower()
    if not parts:
        if ("$" in sample and ("function" in sample or "class" in sample or "->" in sample or "::" in sample)):
            return text
        return ""
    return "\n".join(parts)


# ============================================================
#               Streaming dataset for The Stack
# ============================================================

LANG_CANDIDATE_DIRS: Dict[str, List[str]] = {
    "php": ["php"],
    "csharp": ["c-sharp", "c#", "csharp"],
    "typescript": ["typescript"],
    "csv": ["csv"],
    "go": ["go"],
    "sql": ["sql"],
    "rust": ["rust"],
    "yaml": ["yaml", "yml"],
    "ruby": ["ruby"],
    "python": ["python"],
    "javascript": ["javascript", "js"],
    "java": ["java"],
    "c": ["c"],
    "cpp": ["c++", "cpp"],
    "json": ["json"],
    "css": ["css"],
    "html": ["html", "xhtml", "xml", "svg"],
    "text": ["text"],
}

def try_load_streaming_dir(lang_dir: str, *, shuffle_buffer: int, token: Optional[bool]) -> Optional[Any]:
    load_kwargs = dict(
        path="bigcode/the-stack",
        data_dir=f"data/{lang_dir}",
        split="train",
        streaming=True,
    )
    if token:
        load_kwargs["token"] = token
    try:
        ds = load_dataset(**load_kwargs)
    except TypeError:
        load_kwargs.pop("token", None)
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        sys.stderr.write(f"[warn] skipping dataset dir '{lang_dir}': {e}\n")
        return None
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=42, buffer_size=shuffle_buffer)
    return ds

def stream_language_iterable(
    logical_label: str,
    *,
    shuffle_buffer: int,
    use_auth_token: bool,
    shard_count: int,
    shard_index: int,
    skip_first_n: int,
) -> Optional[Any]:
    logical_label = canonical_label(logical_label)
    cands = LANG_CANDIDATE_DIRS.get(logical_label, [logical_label])
    token_val: Optional[bool] = None
    if use_auth_token:
        try:
            from huggingface_hub import get_token  # type: ignore
            token_val = get_token() or True
        except Exception:
            token_val = True

    ds = None
    for d in cands:
        ds = try_load_streaming_dir(d, shuffle_buffer=shuffle_buffer, token=token_val)
        if ds is not None:
            break
    if ds is None:
        return None

    if shard_count > 1:
        try:
            ds = ds.shard(num_shards=shard_count, index=shard_index)
        except Exception as e:
            sys.stderr.write(f"[warn] shard() failed for {logical_label}: {e}\n")

    if skip_first_n > 0:
        try:
            ds = ds.skip(skip_first_n)
        except Exception:
            def _drop(it):
                i = 0
                for ex in it:
                    i += 1
                    if i <= skip_first_n:
                        continue
                    yield ex
            ds = _drop(ds)

    return ds


# ============================================================
#                  Windowing & Magika batching
# ============================================================

def byte_windows(b: bytes, window_bytes: int) -> Iterator[Tuple[int, bytes]]:
    if not b:
        return
    n = len(b)
    step = window_bytes
    idx = 0
    for off in range(0, n, step):
        yield idx, b[off: off + step]
        idx += 1

def stable_uid_for_window(raw: bytes) -> str:
    return hashlib.blake2s(raw, digest_size=16).hexdigest()

def split_for_uid(uid: str) -> str:
    # 32-bit digest for quick modulo
    h = hashlib.blake2s(uid.encode("utf-8"), digest_size=4).digest()
    v = int.from_bytes(h, byteorder="big") % 100
    if v < 70:
        return "train"
    elif v < 90:
        return "val"
    else:
        return "test"

@dataclass
class MagikaResult:
    ok: bool
    label: Optional[str]
    mime: Optional[str]
    score: float

class MagikaBatcher:
    def __init__(self) -> None:
        from magika import Magika  # lazy import
        self._m = Magika()

    def identify_many(self, chunks: List[bytes]) -> List[MagikaResult]:
        out: List[MagikaResult] = []
        for ch in chunks:
            try:
                res = self._m.identify_bytes(ch)
                ok = bool(getattr(res, "ok", False))
                if ok:
                    o = getattr(res, "output", None)
                    label = getattr(o, "label", None)
                    mime = getattr(o, "mime_type", None)
                    score = float(getattr(res, "score", 0.0))
                    out.append(MagikaResult(True, label, mime, score))
                else:
                    out.append(MagikaResult(False, None, None, 0.0))
            except Exception:
                out.append(MagikaResult(False, None, None, 0.0))
        return out


# ============================================================
#                 Generator: one pass per label
# ============================================================

def gen_windows_for_label(
    *,
    logical_label: str,
    window_bytes: int,
    magika_batch: int,
    threshold: float,
    add_meta: bool,
    shuffle_buffer: int,
    use_auth_token: bool,
    shard_count: int,
    shard_index: int,
    skip_first_n: int,
    progress_mode: str,
    demo: bool,
    seen_uids: Optional[Set[str]] = None,
    budget_per_split: Dict[str, int],
) -> Iterator[dict]:
    is_tty = sys.stderr.isatty()
    show_progress = ((progress_mode == "always") or (progress_mode == "auto" and is_tty))
    total_budget = sum(max(0, b) for b in budget_per_split.values())
    pbar = tqdm(total=total_budget if total_budget > 0 else None,
                unit="win",
                desc=f"{logical_label}",
                disable=not show_progress)

    ds = stream_language_iterable(
        logical_label,
        shuffle_buffer=shuffle_buffer,
        use_auth_token=use_auth_token,
        shard_count=shard_count,
        shard_index=shard_index,
        skip_first_n=skip_first_n,
    )
    if ds is None:
        pbar.close()
        raise RuntimeError(f"Could not open streaming dataset for '{logical_label}'")

    batcher = MagikaBatcher()

    kept_total = 0
    rejected = 0
    kept_per_split = {"train": 0, "val": 0, "test": 0}

    lang_id = LANG2ID.get(canonical_label(logical_label), -1)

    buf_bytes: List[bytes] = []
    buf_meta: List[Tuple[int, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], str, str]] = []
    buf_uids: List[str] = []

    def _flush_batch():
        nonlocal kept_total, rejected
        if not buf_bytes:
            return
        results = batcher.identify_many(buf_bytes)
        for (res, meta, raw, uid) in zip(results, buf_meta, buf_bytes, buf_uids):
            if kept_total >= total_budget and total_budget > 0:
                break
            win_idx, ext, hexsha, repo_name, repo_path, license_str, split, _uid = meta
            if budget_per_split.get(split, 0) <= 0:
                rejected += 1
                continue
            if res.ok and res.score >= threshold and label_matches_target(logical_label, res.label, res.mime):
                payload = {
                    "content": raw.decode("utf-8", errors="ignore"),
                    "lang_id": np.int16(lang_id).item(),
                    "uid": uid,
                    "split": split,
                }
                if add_meta:
                    payload.update({
                        "win_idx": np.int64(win_idx).item(),
                        "source_ext": str(ext or ""),
                        "source_hexsha": str(hexsha or ""),
                        "source_repo": str(repo_name or ""),
                        "source_repo_path": str(repo_path or ""),
                        "license": str(license_str or ""),
                    })
                if seen_uids is not None:
                    seen_uids.add(uid)
                budget_per_split[split] = max(0, budget_per_split[split] - 1)
                kept_per_split[split] += 1
                kept_total += 1
                yield payload
                pbar.update(1)
            else:
                rejected += 1
        buf_bytes.clear()
        buf_meta.clear()
        buf_uids.clear()

    last_postfix = time.time()
    for ex in ds:
        if kept_total >= total_budget and total_budget > 0:
            break

        ok_lic, _ = license_is_allowed(ex)
        if not ok_lic:
            continue

        content = ex.get("content", "")
        if not isinstance(content, str) or not content:
            continue

        if canonical_label(logical_label) == "php":
            content = extract_php_code_only(content)
            if not content:
                continue

        try:
            b = content.encode("utf-8", errors="ignore")
        except Exception:
            continue
        if not b:
            continue

        ext = ex.get("ext")
        hexsha = ex.get("hexsha")
        repo_name = ex.get("max_stars_repo_name") or ex.get("repo_name")
        repo_path = ex.get("max_stars_repo_path") or ex.get("path")
        license_str = ex.get("max_stars_repo_license") or ex.get("license")

        for widx, wbytes in byte_windows(b, window_bytes):
            if kept_total >= total_budget and total_budget > 0:
                break
            uid = stable_uid_for_window(wbytes)

            if seen_uids is not None and uid in seen_uids:
                rejected += 1
                continue

            split = split_for_uid(uid)
            if budget_per_split.get(split, 0) <= 0:
                rejected += 1
                continue

            buf_bytes.append(wbytes)
            buf_meta.append((widx, ext, hexsha, repo_name, repo_path, license_str, split, uid))
            buf_uids.append(uid)

            if len(buf_bytes) >= magika_batch:
                for out in _flush_batch():
                    yield out
                if kept_total >= total_budget and total_budget > 0:
                    break

        now = time.time()
        if (now - last_postfix) >= 0.3:
            pbar.set_postfix(kept_total=kept_total, rej=rejected,
                             k_train=kept_per_split["train"],
                             k_val=kept_per_split["val"],
                             k_test=kept_per_split["test"])
            last_postfix = now

        content = ""
        del b
        gc.collect()

    for out in _flush_batch():
        yield out
    pbar.set_postfix(kept_total=kept_total, rej=rejected,
                     k_train=kept_per_split["train"],
                     k_val=kept_per_split["val"],
                     k_test=kept_per_split["test"])
    pbar.close()


# ============================================================
#               Building (train/val/test per label)
# ============================================================

LANG2ID: Dict[str, int] = {
    "php": 0,
    "csharp": 1,
    "typescript": 2,
    "go": 3,
    "sql": 4,
    "rust": 5,
    "yaml": 6,
    "ruby": 7,
    "python": 8,
    "javascript": 9,
    "java": 10,
    "c": 11,
    "cpp": 12,
    "json": 13,
    "css": 14,
    "html": 15,
    "text": 16,
    "csv": 17,
}

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}

def ensure_recovery_dirs(out_dir: Path) -> None:
    backup_dir = out_dir.with_name(out_dir.name + ".bak")
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp_write")
    if backup_dir.exists() and not out_dir.exists():
        os.replace(backup_dir, out_dir)
    if backup_dir.exists() and out_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

def collect_existing_uids(out_dir: Path) -> Tuple[Optional[Dataset], Set[str], bool]:
    seen: Set[str] = set()
    if not out_dir.exists():
        return None, seen, False
    ds = load_from_disk(str(out_dir))
    has_uid = "uid" in ds.column_names
    if has_uid:
        for uid in ds["uid"]:
            seen.add(uid)
    else:
        for chunk in ds.iter(10000):
            for s in chunk["content"]:
                uid = stable_uid_for_window(s.encode("utf-8", errors="ignore"))
                seen.add(uid)
    return ds, seen, has_uid

def maybe_add_uid_and_resave(ds: Dataset, out_dir: Path) -> Dataset:
    if "uid" in ds.column_names:
        return ds
    def _mk_uid(batch):
        return {"uid": [stable_uid_for_window(x.encode("utf-8", errors="ignore")) for x in batch["content"]]}
    ds2 = ds.map(_mk_uid, batched=True, batch_size=8192)
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp_uid")
    backup_dir = out_dir.with_name(out_dir.name + ".bak_uid")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    ds2.save_to_disk(str(tmp_dir))
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    if out_dir.exists():
        os.replace(str(out_dir), str(backup_dir))
    os.replace(str(tmp_dir), str(out_dir))
    shutil.rmtree(str(backup_dir), ignore_errors=True)
    return load_from_disk(str(out_dir))

def atomic_replace_dir(src_tmp: Path, dst: Path) -> None:
    backup_dir = dst.with_name(dst.name + ".bak")
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    if dst.exists():
        os.replace(str(dst), str(backup_dir))
    os.replace(str(src_tmp), str(dst))
    shutil.rmtree(str(backup_dir), ignore_errors=True)

def dir_for_label_split(out_root: Path, label: str, split: str) -> Path:
    """
    New layout: <out-root>/<split>/<label>/dataset
    """
    return out_root / split / canonical_label(label) / "dataset"

def compute_split_targets(total_cap: int, existing_counts: Dict[str, int]) -> Dict[str, int]:
    total_cap = max(0, int(total_cap))
    ideal = {
        "train": int(total_cap * DEFAULT_RATIOS["train"]),
        "val": int(total_cap * DEFAULT_RATIOS["val"]),
    }
    ideal["test"] = total_cap - ideal["train"] - ideal["val"]

    remaining = {}
    for s in SPLITS:
        need = ideal[s] - existing_counts.get(s, 0)
        remaining[s] = max(0, need)
    return remaining

def load_existing_splits(out_root: Path, label: str, rebuild: bool) -> Tuple[Dict[str, Optional[Dataset]], Dict[str, int], Set[str]]:
    existing_ds: Dict[str, Optional[Dataset]] = {}
    existing_counts: Dict[str, int] = {}
    seen_union: Set[str] = set()

    for split in SPLITS:
        out_dir = dir_for_label_split(out_root, label, split)
        ensure_recovery_dirs(out_dir)
        if rebuild and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        ds, seen, has_uid = collect_existing_uids(out_dir)
        if ds is not None and not has_uid:
            ds = maybe_add_uid_and_resave(ds, out_dir)
            ds, seen, _ = collect_existing_uids(out_dir)

        existing_ds[split] = ds
        existing_counts[split] = len(ds) if ds is not None else 0
        seen_union.update(seen)

    return existing_ds, existing_counts, seen_union

def build_arrow_for_label_with_splits(
    *,
    label: str,
    out_root: Path,
    window_bytes: int,
    magika_batch: int,
    threshold: float,
    max_windows_total: int,
    add_meta: bool,
    shuffle_buffer: int,
    use_auth_token: bool,
    shard_count: int,
    shard_index: int,
    skip_first_n: int,
    progress_mode: str,
    demo: bool,
    writer_batch_size: int,
    rebuild: bool,
) -> Tuple[Dict[str, int], Dict[str, Path]]:
    label_c = canonical_label(label)
    if label_c not in LANG2ID:
        raise ValueError(f"Unknown/unsupported label '{label}'")

    existing_ds, existing_counts, seen_uids = load_existing_splits(out_root, label_c, rebuild)

    total_cap = 100 if demo else max_windows_total
    total_existing = sum(existing_counts.values())

    if total_cap > 0 and total_existing >= total_cap:
        out_dirs = {s: dir_for_label_split(out_root, label_c, s) for s in SPLITS}
        return existing_counts, out_dirs

    per_split_budget = compute_split_targets(total_cap, existing_counts)

    if total_cap == 0:
        per_split_budget = {s: 2**63 - 1 for s in SPLITS}
    else:
        shortfall = total_cap - total_existing
        sum_budgets = sum(per_split_budget.values())
        if sum_budgets > shortfall:
            if sum_budgets > 0:
                scale = shortfall / sum_budgets
                per_split_budget = {s: int(per_split_budget[s] * scale) for s in SPLITS}
            while sum(per_split_budget.values()) < shortfall:
                for s in SPLITS:
                    if sum(per_split_budget.values()) >= shortfall:
                        break
                    per_split_budget[s] += 1

    features = {
        "content": Value("string"),
        "lang_id": Value("int16"),
        "uid": Value("string"),
        "split": Value("string"),
    }
    if add_meta:
        features.update({
            "win_idx": Value("int64"),
            "source_ext": Value("string"),
            "source_hexsha": Value("string"),
            "source_repo": Value("string"),
            "source_repo_path": Value("string"),
            "license": Value("string"),
        })
    feats = Features(features)

    gen_kwargs = dict(
        logical_label=label_c,
        window_bytes=window_bytes,
        magika_batch=magika_batch,
        threshold=threshold,
        add_meta=add_meta,
        shuffle_buffer=shuffle_buffer,
        use_auth_token=use_auth_token,
        shard_count=shard_count,
        shard_index=shard_index,
        skip_first_n=skip_first_n,
        progress_mode=progress_mode,
        demo=demo,
        seen_uids=seen_uids,
        budget_per_split=per_split_budget,
    )

    ds_new_total = Dataset.from_generator(
        gen_windows_for_label,
        gen_kwargs=gen_kwargs,
        features=feats,
        keep_in_memory=False,
        writer_batch_size=writer_batch_size,
    )

    new_by_split: Dict[str, Optional[Dataset]] = {}
    for s in SPLITS:
        if len(ds_new_total) > 0:
            part = ds_new_total.filter(lambda ex, _s=None: ex["split"] == _s, fn_kwargs={"_s": s})
            if "split" in part.column_names:
                part = part.remove_columns(["split"])
        else:
            part = None
        new_by_split[s] = part

    kept_after_run: Dict[str, int] = dict(existing_counts)
    out_dirs: Dict[str, Path] = {}

    for s in SPLITS:
        out_dir = dir_for_label_split(out_root, label_c, s)
        ensure_recovery_dirs(out_dir)
        out_dirs[s] = out_dir

        ds_new = new_by_split[s]
        ds_exist = existing_ds[s]

        if ds_new is None or len(ds_new) == 0:
            continue

        if ds_exist is not None:
            full = concatenate_datasets([ds_exist, ds_new])
        else:
            full = ds_new

        tmp_dir = out_dir.with_name(out_dir.name + ".tmp_write")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.parent.mkdir(parents=True, exist_ok=True)
        full.save_to_disk(str(tmp_dir))
        atomic_replace_dir(tmp_dir, out_dir)

        kept_after_run[s] = (existing_counts.get(s, 0) + len(ds_new))

    return kept_after_run, out_dirs


# ============================================================
#                           CLI
# ============================================================

def parse_skip_map(s: Optional[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not s:
        return out
    for item in s.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = canonical_label(k.strip())
        try:
            out[k] = int(v.strip())
        except Exception:
            pass
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Stream The Stack and build Arrow datasets of windowed code per label with train/val/test splits.\n"
            "• Writes to <out-root>/<split>/<label>/dataset\n"
            "• Filters licenses (permissive only)\n"
            "• PHP: strips foreign (HTML/etc.) content via regex\n"
            "• In-process Magika prefilter (no temp files)\n"
            "• Resume-safe, duplicate-proof with per-window uid and deterministic splits (70/20/10)"
        )
    )
    # Core IO
    ap.add_argument("--out-root", type=Path, default=Path("arrow_out"),
                    help="Output root for Arrow datasets (<out-root>/<split>/<label>/dataset)")
    ap.add_argument("--langs", type=str,
                    default="json,html,javascript,css,csv,text,java,c,c++,python,typescript,php,csharp,go,sql,rust,yaml,ruby",
                    help="Comma-separated labels to process (logical names).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete existing output directories for selected labels (per split) BEFORE writing")
    ap.add_argument("--add-meta", action="store_true",
                    help="Include metadata columns (win_idx, ext, repo, license, etc.)")

    # Windowing / filtering
    ap.add_argument("--window-bytes", type=int, default=1536,
                    help="Window size in BYTES (default: 1536)")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="Magika confidence threshold (keep if score >= threshold)")
    ap.add_argument("--magika-batch", type=int, default=1024,
                    help="How many windows to run through Magika per batch")
    ap.add_argument("--max-windows-per-label", type=int, default=40_000,
                    help="TOTAL cap of kept windows per label (post-filter) across splits. Default: 40,000")
    ap.add_argument("--writer-batch-size", type=int, default=8192,
                    help="HF writer batch size to Arrow (bigger = fewer flushes)")

    # Streaming controls
    ap.add_argument("--shuffle-buffer", type=int, default=0,
                    help="Streaming shuffle buffer size (0 disables; large may use a lot of RAM)")
    ap.add_argument("--use-auth-token", action="store_true",
                    help="Pass cached HF auth token if required")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="Shard the stream into N parts to jump ahead without scanning from the start")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="Which shard index to read (0..shard-count-1)")
    ap.add_argument("--skip-per-lang", type=str, default=None,
                    help='Optional extra skip per label, e.g. "php=500000,csharp=200000"')

    # UX
    ap.add_argument("--progress", choices=["auto", "always", "never"], default="auto",
                    help="Show live progress bars (default: auto)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--demo", action="store_true",
                    help="Demo mode: keep at most 100 windows per label TOTAL (overrides --max-windows-per-label)")
    return ap.parse_args()


# ============================================================
#                           Main
# ============================================================

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    labels = [canonical_label(s) for s in args.langs.split(",") if s.strip()]
    if not labels:
        raise SystemExit("[fatal] No labels provided via --langs")

    out_root: Path = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    max_per_label_total = 100 if args.demo else args.max_windows_per_label
    skip_map = parse_skip_map(args.skip_per_lang)

    print("⚙️  The Stack → Arrow (windowed, Magika-filtered, resume-safe) with 70/20/10 splits")
    print(f"   Out root:        {out_root}  (layout: <split>/<label>/dataset)")
    print(f"   Labels:          {', '.join(labels)}")
    print(f"   Window bytes:    {args.window_bytes}")
    print(f"   Magika batch:    {args.magika_batch}")
    print(f"   Threshold:       {args.threshold:.2f}")
    print(f"   Max/label:       {max_per_label_total} {'(DEMO)' if args.demo else ''}  [split 70/20/10]")
    print(f"   Shuffle buffer:  {args.shuffle_buffer}")
    print(f"   Shard:           count={args.shard_count}, index={args.shard_index}")
    if skip_map:
        print(f"   Extra skips:     " + ", ".join(f"{k}={v}" for k, v in skip_map.items()))
    print(f"   Add meta:        {'yes' if args.add_meta else 'no'}")
    print(f"   Rebuild:         {'yes' if args.rebuild else 'no'}")
    print("")

    grand_kept_per_split = {"train": 0, "val": 0, "test": 0}
    failures: List[str] = []

    t0 = time.time()
    for lbl in labels:
        print(f"— Building label: {lbl}")
        try:
            kept_map, out_dirs = build_arrow_for_label_with_splits(
                label=lbl,
                out_root=out_root,
                window_bytes=args.window_bytes,
                magika_batch=args.magika_batch,
                threshold=args.threshold,
                max_windows_total=max_per_label_total,
                add_meta=args.add_meta,
                shuffle_buffer=args.shuffle_buffer,
                use_auth_token=args.use_auth_token,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                skip_first_n=int(skip_map.get(canonical_label(lbl), 0)),
                progress_mode=args.progress,
                demo=args.demo,
                writer_batch_size=args.writer_batch_size,
                rebuild=args.rebuild,
            )
            grand_kept_per_split["train"] += kept_map.get("train", 0)
            grand_kept_per_split["val"] += kept_map.get("val", 0)
            grand_kept_per_split["test"] += kept_map.get("test", 0)
            base = out_root / "train" / canonical_label(lbl)
            print(f"  ✓ Totals now: train={kept_map.get('train',0):,}  "
                  f"val={kept_map.get('val',0):,}  test={kept_map.get('test',0):,}  "
                  f"→ {base.parent.parent} / <train|val|test> / {canonical_label(lbl)}/dataset\n")
        except KeyboardInterrupt:
            print("\n[!] Interrupted by user.", file=sys.stderr)
            raise
        except Exception as e:
            failures.append(f"{lbl}: {e}")
            print(f"  ✗ Failed: {e}\n", file=sys.stderr)

        gc.collect()

    elapsed = time.time() - t0
    total = grand_kept_per_split["train"] + grand_kept_per_split["val"] + grand_kept_per_split["test"]
    rate = total / elapsed if elapsed > 0 else 0.0
    print("🎯 Summary")
    print(f"  Labels processed: {len(labels)}")
    print(f"  Total kept (train): {grand_kept_per_split['train']:,}")
    print(f"  Total kept (val):   {grand_kept_per_split['val']:,}")
    print(f"  Total kept (test):  {grand_kept_per_split['test']:,}")
    print(f"  Elapsed:            {elapsed/60.0:.1f} min  ({rate:.1f} win/s)")
    if failures:
        print("  Failures:")
        for f in failures:
            print(f"    - {f}")
    print(f"\nOutput ready under: {out_root}  (layout: <split>/<label>/dataset)")

if __name__ == "__main__":
    main()
