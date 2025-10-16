"""
End-to-end The Stack → filtered windows → Arrow datasets (single script)

What this does (single pass per language, no temp files):
  • Streams from BigCode "bigcode/the-stack" per language folder (no full download).
  • Strict license filter: keeps permissive families only (MIT/Apache/BSD/Unlicense).
  • Windows content to fixed-size byte chunks (default: 1536 bytes).
  • Magika pre-filtering in batches (kept in-process; no multiprocessing bugs).
  • PHP: strips all non-PHP blocks via regex (keeps only code between <?php ... ?> / <?= ... ?>).
  • Writes **only** final Arrow datasets to disk (one dataset per language label).
  • Default cap: **1,000,000 kept windows per label** (post-filter). `--demo` → 100 per label.
  • Robust streaming controls: shard/offset/skip to jump ahead in huge sorted datasets.
  • Detailed progress bars & summaries (tqdm) and periodic logging.

Resume & dedupe:
  • Each window gets a stable content hash field `uid`.
  • If an output dataset already exists, we (a) recover from any prior partial write,
    (b) collect existing `uid`s (cheap), and (c) skip duplicates on the fly.
  • We write new data to a temporary directory, then atomically swap it in.
  • Aborting mid-save never corrupts the prior dataset; reruns are safe and cheap.

Notes:
  - No stage-wise “write many raw files” — only Arrow datasets are written.
  - Magika is invoked in batches inside the same process (stable, fast enough, avoids fork issues).
  - You can extend languages with `--langs` or tweak Magika threshold, window bytes, etc.
  - Splits (train/val/test) are intentionally omitted to keep one-pass/no-temp constraint;
    you can create splits later with HF datasets if needed.

Example:
  python build_stack_windows_arrow.py \
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
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
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
    # include "cs" so Magika extension-based labels map to csharp
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
    # add both "c-sharp" and "cs" explicitly
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
# Rare legacy ASP-style tags; keep but low precedence.
_ASP_PHP_BLOCK_RE = re.compile(
    r"(?is)<%(.*?)%>"
)

def extract_php_code_only(text: str) -> str:
    """
    Keep ONLY PHP code regions, drop all HTML/other template text.
    - Matches <?php ... ?> and <?= ... ?> (captures inner).
    - Ignores <?xml ...?>.
    - Also recognizes legacy <% ... %> blocks (very rare).
    - Does NOT normalize/pretty-print; preserves inner code as-is.
    """
    if not text:
        return ""
    parts: List[str] = []
    for m in _PHP_BLOCK_RE.finditer(text):
        inner = m.group(1)
        if inner is not None:
            parts.append(inner)
    # If nothing found, try ASP-style tags
    if not parts:
        for m in _ASP_PHP_BLOCK_RE.finditer(text):
            inner = m.group(1)
            if inner is not None:
                parts.append(inner)
    # If still nothing and the file looks like a pure PHP file without tags (rare), keep as-is
    sample = text[:4096].lower()
    if not parts:
        if ("$" in sample and ("function" in sample or "class" in sample or "->" in sample or "::" in sample)):
            return text
        return ""
    return "\n".join(parts)


# ============================================================
#               Streaming dataset for The Stack
# ============================================================

# Candidate dataset folder names per logical label
LANG_CANDIDATE_DIRS: Dict[str, List[str]] = {
    "php": ["php"],
    # prefer "c-sharp" first to avoid warnings; fallbacks kept
    "csharp": ["c-sharp", "c#", "csharp"],
    "typescript": ["typescript"],
    "go": ["go"],
    "sql": ["sql"],
    "rust": ["rust"],
    "yaml": ["yaml", "yml"],
    "ruby": ["ruby"],
    # Some common extras (optional to include via --langs)
    "python": ["python"],
    "javascript": ["javascript", "js"],
    "java": ["java"],
    "c": ["c"],
    "cpp": ["c++", "cpp"],
    "json": ["json"],
    "css": ["css"],
    "html": ["html", "xhtml", "xml", "svg"],
    "text": ["text"],  # very rare as a dedicated dir
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
    """
    Returns an IterableDataset for a language, applying shard/skip as requested.
    Tries multiple folder names for robustness.
    """
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

    # Shard first to "jump ahead" without touching earlier items.
    if shard_count > 1:
        try:
            ds = ds.shard(num_shards=shard_count, index=shard_index)
        except Exception as e:
            sys.stderr.write(f"[warn] shard() failed for {logical_label}: {e}\n")

    # Skip within shard if requested
    if skip_first_n > 0:
        try:
            ds = ds.skip(skip_first_n)
        except Exception:
            # Fallback: manual dropper
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
    """
    Yield (window_idx, window_bytes) for a bytes object without copying more than needed.
    """
    if not b:
        return
    n = len(b)
    step = window_bytes
    idx = 0
    for off in range(0, n, step):
        yield idx, b[off: off + step]
        idx += 1

def stable_uid_for_window(raw: bytes) -> str:
    """
    Stable, compact hash for a window's raw bytes. Language-agnostic; duplicates (even
    from different sources) collapse, which is desirable for dedupe-on-resume.
    """
    return hashlib.blake2s(raw, digest_size=16).hexdigest()

@dataclass
class MagikaResult:
    ok: bool
    label: Optional[str]
    mime: Optional[str]
    score: float

class MagikaBatcher:
    """
    Simple in-process batcher for Magika bytes identification.
    Keeps a single Magika instance and identifies sequentially per batch.
    (Magika doesn't expose a public bytes-batch API; we still group calls
     to keep the rest of the pipeline vectorized.)
    """
    def __init__(self) -> None:
        from magika import Magika  # lazy import for stability
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
    max_windows: int,
    add_meta: bool,
    shuffle_buffer: int,
    use_auth_token: bool,
    shard_count: int,
    shard_index: int,
    skip_first_n: int,
    progress_mode: str,
    demo: bool,
    seen_uids: Optional[Set[str]] = None,
) -> Iterator[dict]:
    """
    Stream The Stack for a given language (logical_label), window it, Magika-filter,
    de-duplicate against seen_uids, and yield dicts. max_windows limits *kept* count.
    """
    # UI setup
    is_tty = sys.stderr.isatty()
    show_progress = ((progress_mode == "always") or (progress_mode == "auto" and is_tty))
    pbar = tqdm(total=max_windows if max_windows > 0 else None,
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

    kept = 0
    rejected = 0  # includes non-matching, low score, or duplicates

    # Precompute lang_id mapping here (consistent enumeration below)
    lang_id = LANG2ID.get(canonical_label(logical_label), -1)

    # Rolling buffers for batched Magika calls
    buf_bytes: List[bytes] = []
    buf_meta: List[Tuple[int, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]] = []
    buf_uids: List[str] = []
    # meta per window: (window_idx, ext, hexsha, repo_name, repo_path, license_str)

    def _flush_batch():
        nonlocal kept, rejected
        if not buf_bytes:
            return
        results = batcher.identify_many(buf_bytes)
        for (res, meta, raw, uid) in zip(results, buf_meta, buf_bytes, buf_uids):
            if kept >= max_windows:
                break
            if res.ok and res.score >= threshold and label_matches_target(logical_label, res.label, res.mime):
                payload = {
                    "content": raw.decode("utf-8", errors="ignore"),
                    "lang_id": np.int16(lang_id).item(),  # int16 to be safe for many classes
                    "uid": uid,
                }
                if add_meta:
                    win_idx, ext, hexsha, repo_name, repo_path, license_str = meta
                    payload.update({
                        "win_idx": np.int64(win_idx).item(),
                        "source_ext": str(ext or ""),
                        "source_hexsha": str(hexsha or ""),
                        "source_repo": str(repo_name or ""),
                        "source_repo_path": str(repo_path or ""),
                        "license": str(license_str or ""),
                    })
                # update seen set as we keep
                if seen_uids is not None:
                    seen_uids.add(uid)
                yield payload
                kept += 1
                pbar.update(1)
            else:
                rejected += 1
        buf_bytes.clear()
        buf_meta.clear()
        buf_uids.clear()

    last_postfix = time.time()
    for ex in ds:
        if kept >= max_windows:
            break

        # License gate
        ok_lic, _ = license_is_allowed(ex)
        if not ok_lic:
            continue

        # Grab content
        content = ex.get("content", "")
        if not isinstance(content, str) or not content:
            continue

        # PHP special handling: drop all HTML/foreign sections
        label_lc = canonical_label(logical_label)
        if label_lc == "php":
            content = extract_php_code_only(content)
            if not content:
                continue

        # Turn to bytes and window
        try:
            b = content.encode("utf-8", errors="ignore")
        except Exception:
            continue
        if not b:
            continue

        # Basic metadata (if present)
        ext = ex.get("ext")
        hexsha = ex.get("hexsha")
        repo_name = ex.get("max_stars_repo_name") or ex.get("repo_name")
        repo_path = ex.get("max_stars_repo_path") or ex.get("path")
        license_str = ex.get("max_stars_repo_license") or ex.get("license")

        for widx, wbytes in byte_windows(b, window_bytes):
            # pre-dedupe via uid before Magika (saves compute)
            uid = stable_uid_for_window(wbytes)
            if seen_uids is not None and uid in seen_uids:
                rejected += 1
                continue
            buf_bytes.append(wbytes)
            buf_meta.append((widx, ext, hexsha, repo_name, repo_path, license_str))
            buf_uids.append(uid)
            # Flush when batch is full
            if len(buf_bytes) >= magika_batch:
                for out in _flush_batch():
                    yield out
                if kept >= max_windows:
                    break

        # Periodic UI refresh
        now = time.time()
        if (now - last_postfix) >= 0.3:
            pbar.set_postfix(kept=kept, rej=rejected)
            last_postfix = now

        # Aggressive early GC to keep RAM stable during huge streams
        content = ""
        del b
        gc.collect()

    # Flush remainder
    for out in _flush_batch():
        yield out
    pbar.set_postfix(kept=kept, rej=rejected)
    pbar.close()


# ============================================================
#               Building (one dataset per label)
# ============================================================

# Full label↔id table (extensible). Keep stable ordering across runs.
LANG2ID: Dict[str, int] = {
    # Focus classes requested by the user first:
    "php": 0,
    "csharp": 1,
    "typescript": 2,
    "go": 3,
    "sql": 4,
    "rust": 5,
    "yaml": 6,
    "ruby": 7,
    # Optionals (if included via --langs)
    "python": 8,
    "javascript": 9,
    "java": 10,
    "c": 11,
    "cpp": 12,
    "json": 13,
    "css": 14,
    "html": 15,
    "text": 16,
}

def ensure_recovery_dirs(out_dir: Path) -> None:
    """
    Make resume operations safe:
      - If a previous swap left a .bak, restore or remove appropriately.
      - If a previous tmp exists, remove it.
    """
    backup_dir = out_dir.with_name(out_dir.name + ".bak")
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp_write")
    # If backup exists but out_dir missing, restore
    if backup_dir.exists() and not out_dir.exists():
        os.replace(backup_dir, out_dir)
    # If both exist, discard stale backup
    if backup_dir.exists() and out_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    # Remove any stale tmp from interrupted save
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

def collect_existing_uids(out_dir: Path) -> Tuple[Optional[Dataset], Set[str], bool]:
    """
    Load any existing dataset and return (dataset, set_of_uids, has_uid_column).
    If uid column is missing, compute it on the fly to dedupe; we may add it later.
    """
    seen: Set[str] = set()
    if not out_dir.exists():
        return None, seen, False
    ds = load_from_disk(str(out_dir))
    has_uid = "uid" in ds.column_names
    if has_uid:
        # Build set cheaply
        for uid in ds["uid"]:
            seen.add(uid)
    else:
        # Compute on the fly from content; cheap for tens/hundreds of thousands
        for chunk in ds.iter(10000):
            for s in chunk["content"]:
                uid = stable_uid_for_window(s.encode("utf-8", errors="ignore"))
                seen.add(uid)
    return ds, seen, has_uid

def maybe_add_uid_and_resave(ds: Dataset, out_dir: Path) -> Dataset:
    """
    If the existing dataset lacks 'uid', add it (via content hash) and write
    back atomically. This accelerates future resumptions.
    """
    if "uid" in ds.column_names:
        return ds
    def _mk_uid(batch):
        return {"uid": [stable_uid_for_window(x.encode("utf-8", errors="ignore")) for x in batch["content"]]}
    ds2 = ds.map(_mk_uid, batched=True, batch_size=8192)
    # Save atomically
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
    """
    Swap in src_tmp to dst atomically (best-effort):
      - Move dst -> dst.bak
      - Move src_tmp -> dst
      - Remove dst.bak
    On crash between steps, ensure_recovery_dirs(...) at next start fixes things.
    """
    backup_dir = dst.with_name(dst.name + ".bak")
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    if dst.exists():
        os.replace(str(dst), str(backup_dir))
    os.replace(str(src_tmp), str(dst))
    shutil.rmtree(str(backup_dir), ignore_errors=True)

def build_arrow_for_label(
    *,
    label: str,
    out_root: Path,
    window_bytes: int,
    magika_batch: int,
    threshold: float,
    max_windows: int,
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
) -> Tuple[int, Path]:
    """
    Create/extend a Hugging Face Dataset directly from a generator for one label and save to disk.
    Resume-safe and duplicate-proof via 'uid'.
    Returns (total_kept_after_run, output_dir)
    """
    label_c = canonical_label(label)
    if label_c not in LANG2ID:
        raise ValueError(f"Unknown/unsupported label '{label}'")

    out_dir = out_root / label_c

    # Recover from any prior interrupted writes
    ensure_recovery_dirs(out_dir)

    # Optional full rebuild
    if rebuild and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    # Load existing (if any) and collect uids
    existing_ds, seen_uids, has_uid = collect_existing_uids(out_dir)
    kept_before = len(seen_uids)

    # If existing dataset lacks uid, add it for faster future resumes
    if existing_ds is not None and not has_uid:
        existing_ds = maybe_add_uid_and_resave(existing_ds, out_dir)
        # Recollect (now cheap)
        existing_ds, seen_uids, _ = collect_existing_uids(out_dir)
        kept_before = len(seen_uids)

    # Features schema (always include 'uid' for resume/dedupe)
    features = {
        "content": Value("string"),
        "lang_id": Value("int16"),
        "uid": Value("string"),
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

    # Compute remaining budget
    remaining = max(0, (max_windows if max_windows > 0 else max_windows) - kept_before)
    if remaining == 0 and max_windows > 0:
        # Nothing to do; already at/above cap
        return kept_before, out_dir

    # Build new (non-duplicate) windows
    gen_kwargs = dict(
        logical_label=label_c,
        window_bytes=window_bytes,
        magika_batch=magika_batch,
        threshold=threshold,
        max_windows=remaining if max_windows > 0 else 0,  # 0 = unlimited; but we keep budgeted
        add_meta=add_meta,
        shuffle_buffer=shuffle_buffer,
        use_auth_token=use_auth_token,
        shard_count=shard_count,
        shard_index=shard_index,
        skip_first_n=skip_first_n,
        progress_mode=progress_mode,
        demo=demo,
        seen_uids=seen_uids,  # live dedupe
    )

    ds_new = Dataset.from_generator(
        gen_windows_for_label,
        gen_kwargs=gen_kwargs,
        features=feats,
        keep_in_memory=False,
        writer_batch_size=writer_batch_size,
    )

    # If nothing new, we're done
    if len(ds_new) == 0:
        return kept_before, out_dir

    # Merge with existing (if any) and save atomically
    if existing_ds is not None:
        # Ensure both have the same columns ordering
        full = concatenate_datasets([existing_ds, ds_new])
    else:
        full = ds_new

    tmp_dir = out_dir.with_name(out_dir.name + ".tmp_write")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.parent.mkdir(parents=True, exist_ok=True)
    full.save_to_disk(str(tmp_dir))
    atomic_replace_dir(tmp_dir, out_dir)

    kept_after = kept_before + len(ds_new)
    return kept_after, out_dir


# ============================================================
#                          CLI
# ============================================================

def parse_skip_map(s: Optional[str]) -> Dict[str, int]:
    """
    Parse "php=500000,csharp=200000" → {"php": 500000, "csharp": 200000}
    """
    out: Dict[str, int] = {}
    if not s:
        return out
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
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
            "Stream The Stack and build Arrow datasets of windowed code per label.\n"
            "• Filters licenses (permissive only)\n"
            "• PHP: strips foreign (HTML/etc.) content via regex\n"
            "• Magika prefilter in-process batches (no temp files)\n"
            "• Writes ONLY Arrow datasets at the end\n"
            "• Resume-safe, duplicate-proof with per-window uid"
        )
    )
    # Core IO
    ap.add_argument("--out-root", type=Path, default=Path("arrow_windows"),
                    help="Output directory for Arrow datasets (one subdir per label)")
    ap.add_argument("--langs", type=str,
                    default="json,html,javascript,css,csv,text,java,c,c++,python,typescript,php,csharp,go,sql,rust,yaml,ruby",
                    help="Comma-separated labels to process (logical names).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Delete existing output directories for selected labels before writing")
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
                    help="Cap of kept windows per label (post-filter). Default: 40_000")
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
                    help="Demo mode: keep at most 100 windows per label (overrides --max-windows-per-label)")
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

    max_per_label = 100 if args.demo else args.max_windows_per_label
    skip_map = parse_skip_map(args.skip_per_lang)

    print("⚙️  The Stack → Arrow (windowed, Magika-filtered, resume-safe)")
    print(f"   Out root:        {out_root}")
    print(f"   Labels:          {', '.join(labels)}")
    print(f"   Window bytes:    {args.window_bytes}")
    print(f"   Magika batch:    {args.magika_batch}")
    print(f"   Threshold:       {args.threshold:.2f}")
    print(f"   Max/label:       {max_per_label} {'(DEMO)' if args.demo else ''}")
    print(f"   Shuffle buffer:  {args.shuffle_buffer}")
    print(f"   Shard:           count={args.shard_count}, index={args.shard_index}")
    if skip_map:
        print(f"   Extra skips:     " + ", ".join(f"{k}={v}" for k, v in skip_map.items()))
    print(f"   Add meta:        {'yes' if args.add_meta else 'no'}")
    print(f"   Rebuild:         {'yes' if args.rebuild else 'no'}")
    print("")

    grand_kept = 0
    failures: List[str] = []

    t0 = time.time()
    for lbl in labels:
        print(f"— Building label: {lbl}")
        try:
            kept, out_dir = build_arrow_for_label(
                label=lbl,
                out_root=out_root,
                window_bytes=args.window_bytes,
                magika_batch=args.magika_batch,
                threshold=args.threshold,
                max_windows=max_per_label,
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
            grand_kept += kept
            print(f"  ✓ Total kept now: {kept:,} windows  →  {out_dir}\n")
        except KeyboardInterrupt:
            print("\n[!] Interrupted by user.", file=sys.stderr)
            raise
        except Exception as e:
            failures.append(f"{lbl}: {e}")
            print(f"  ✗ Failed: {e}\n", file=sys.stderr)

        # Encourage GC between labels to cap memory
        gc.collect()

    elapsed = time.time() - t0
    rate = grand_kept / elapsed if elapsed > 0 else 0.0
    print("🎯 Summary")
    print(f"  Labels processed: {len(labels)}")
    print(f"  Total kept (sum of totals per label): {grand_kept:,} windows")
    print(f"  Elapsed:          {elapsed/60.0:.1f} min  ({rate:.1f} win/s)")
    if failures:
        print("  Failures:")
        for f in failures:
            print(f"    - {f}")
    print(f"\nOutput ready under: {out_root}")

if __name__ == "__main__":
    main()
