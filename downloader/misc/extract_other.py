#!/usr/bin/env python3
"""
Build an Arrow dataset for open-set "other" content from The Stack languages
that are not part of the supervised label mapping.

Output layout matches the project convention:
  <out-root>/<split>/<label>/dataset

Default usage:
  python downloader/misc/extract_other.py \
    --out-root downloader/arrow_out_other \
    --label other \
    --split train \
    --max-samples 200000 \
    --max-bytes 1536 \
    --use-auth-token
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
from datasets import Dataset, Features, Value, concatenate_datasets, load_from_disk
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downloader.main import (  # noqa: E402
    LANG_CANDIDATE_DIRS,
    atomic_replace_dir,
    clamp_utf8_bytes,
    collect_existing_uids,
    ensure_recovery_dirs,
    extract_primary_text,
    license_is_allowed,
    maybe_add_uid_and_resave,
    resolve_hf_token_pair,
    stable_uid_for_window,
    try_load_streaming_dir,
)


def _parse_csv_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip().lower() for part in str(value).split(",") if part.strip()]


def _list_stack_data_dirs(use_auth_token: bool) -> List[str]:
    token_str, _ = resolve_hf_token_pair(use_auth_token)
    try:
        from huggingface_hub import HfFileSystem  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required to list The Stack language dirs."
        ) from e
    fs = HfFileSystem(token=token_str)
    try:
        entries = fs.ls("datasets/bigcode/the-stack/data", detail=True)
    except Exception as e:
        raise RuntimeError(f"Failed to list The Stack data dirs: {e}") from e
    out = sorted(
        entry["name"].split("/")[-1].lower()
        for entry in entries
        if entry.get("type") == "directory"
    )
    if not out:
        raise RuntimeError("No The Stack data dirs discovered under bigcode/the-stack/data.")
    return out


def _default_mapped_stack_dirs() -> Set[str]:
    out: Set[str] = set()
    for dirs in LANG_CANDIDATE_DIRS.values():
        for d in dirs:
            out.add(str(d).strip().lower())
    return out


def _load_existing_for_output(out_dir: Path, rebuild: bool) -> Tuple[Optional[Dataset], Set[str]]:
    ensure_recovery_dirs(out_dir)
    if rebuild and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)

    ds_exist, seen_uids, has_uid = collect_existing_uids(out_dir)
    if ds_exist is not None and not has_uid:
        ds_exist = maybe_add_uid_and_resave(ds_exist, out_dir)
        ds_exist, seen_uids, _ = collect_existing_uids(out_dir)
    return ds_exist, seen_uids


def _flush_chunk_to_output(
    *,
    out_dir: Path,
    ds_exist: Optional[Dataset],
    rows: List[Dict[str, Any]],
    features: Features,
) -> Tuple[Optional[Dataset], int]:
    if not rows:
        return ds_exist, 0

    ds_new = Dataset.from_list(rows, features=features)
    if len(ds_new) <= 0:
        return ds_exist, 0

    full = concatenate_datasets([ds_exist, ds_new]) if ds_exist is not None else ds_new
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp_write")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.parent.mkdir(parents=True, exist_ok=True)
    full.save_to_disk(str(tmp_dir))
    atomic_replace_dir(tmp_dir, out_dir)
    return load_from_disk(str(out_dir)), int(len(ds_new))


def _iter_other_rows(
    *,
    source_dirs: List[str],
    shuffle_buffer: int,
    max_open_streams: int,
    rotate_stream_every: int,
    token_arg: Optional[Any],
    max_samples: int,
    max_bytes: int,
    seen_uids: Set[str],
    add_meta: bool,
    progress: str,
) -> Iterator[Dict[str, Any]]:
    source_dirs = [str(s).strip().lower() for s in source_dirs if str(s).strip()]
    if not source_dirs:
        raise RuntimeError("No source dirs were provided for OTHER extraction.")

    # Keep total shuffle memory bounded by splitting a global shuffle budget
    # across concurrently open streams, instead of applying it per stream.
    open_cap = max(1, int(max_open_streams))
    open_target = min(open_cap, len(source_dirs))
    global_shuffle = max(0, int(shuffle_buffer))
    per_stream_shuffle = global_shuffle // max(1, open_target)
    if global_shuffle > 0 and per_stream_shuffle <= 0:
        per_stream_shuffle = 1

    active: List[Tuple[str, Iterator[Dict[str, Any]]]] = []
    active_names: Set[str] = set()
    cursor = 0
    opened = 0

    def _try_open_one() -> bool:
        nonlocal cursor, opened
        attempts = 0
        while attempts < len(source_dirs):
            src = source_dirs[cursor % len(source_dirs)]
            cursor += 1
            attempts += 1
            if src in active_names:
                continue
            ds = try_load_streaming_dir(
                src,
                shuffle_buffer=per_stream_shuffle,
                token=token_arg,
            )
            if ds is None:
                continue
            active.append((src, iter(ds)))
            active_names.add(src)
            opened += 1
            return True
        return False

    while len(active) < open_target and _try_open_one():
        pass
    if not active:
        raise RuntimeError("Failed to open any streaming source dir for OTHER extraction.")

    keep_pbar = tqdm(
        total=max_samples if max_samples > 0 else None,
        unit="sample",
        desc="extract_other",
        disable=(progress == "never"),
    )
    kept = 0
    rejected = 0
    idx = 0
    rotate_every = max(0, int(rotate_stream_every))
    rotations = 0
    if progress != "never":
        keep_pbar.set_postfix(
            kept=kept,
            rejected=rejected,
            active_streams=len(active),
            per_stream_shuffle=per_stream_shuffle,
        )
    try:
        while active and kept < max_samples:
            if idx >= len(active):
                idx = 0
            src, it = active[idx]
            try:
                ex = next(it)
                idx += 1
            except StopIteration:
                active.pop(idx)
                active_names.discard(src)
                _try_open_one()
                continue

            ok_lic, _ = license_is_allowed(ex)
            if not ok_lic:
                rejected += 1
                continue

            raw_text = extract_primary_text(ex)
            if not raw_text:
                rejected += 1
                continue

            trimmed_text, trimmed_bytes = clamp_utf8_bytes(raw_text, max_bytes)
            if not trimmed_bytes:
                rejected += 1
                continue

            uid = stable_uid_for_window(trimmed_bytes)
            if uid in seen_uids:
                rejected += 1
                continue
            seen_uids.add(uid)

            payload: Dict[str, Any] = {
                "content": trimmed_text,
                "lang_id": np.int16(-1).item(),
                "uid": uid,
                "stack_label": src,
            }
            if add_meta:
                payload.update(
                    {
                        "source_ext": str(ex.get("ext") or ""),
                        "source_hexsha": str(ex.get("hexsha") or ""),
                        "source_repo": str(ex.get("max_stars_repo_name") or ex.get("repo_name") or ""),
                        "source_repo_path": str(ex.get("max_stars_repo_path") or ex.get("path") or ""),
                        "license": str(ex.get("max_stars_repo_license") or ex.get("license") or ""),
                    }
                )
            kept += 1
            keep_pbar.update(1)
            # Rotate one active stream periodically to spread sampling over many
            # non-mapped dirs without keeping all iterators open at once.
            if rotate_every > 0 and (kept % rotate_every) == 0 and active:
                drop_idx = (idx - 1) % len(active)
                drop_src, _ = active.pop(drop_idx)
                active_names.discard(drop_src)
                if idx > drop_idx:
                    idx -= 1
                _try_open_one()
                rotations += 1
            if (kept % 200) == 0:
                keep_pbar.set_postfix(
                    kept=kept,
                    rejected=rejected,
                    active_streams=len(active),
                    opened=opened,
                    rotated=rotations,
                    per_stream_shuffle=per_stream_shuffle,
                )
            yield payload
    finally:
        keep_pbar.close()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract open-set 'other' Arrow samples from non-mapped The Stack languages."
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("downloader/arrow_out_other"),
        help="Output root (<out-root>/<split>/<label>/dataset).",
    )
    ap.add_argument("--split", type=str, default="train", help="Output split name (default: train).")
    ap.add_argument("--label", type=str, default="other", help="Output label directory name.")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="Max number of OTHER samples to keep in the output split.",
    )
    ap.add_argument(
        "--max-bytes",
        type=int,
        default=1536,
        help="UTF-8 byte clamp per sample to keep windows bounded.",
    )
    ap.add_argument("--shuffle-buffer", type=int, default=20000)
    ap.add_argument(
        "--max-open-streams",
        type=int,
        default=16,
        help="Maximum concurrently open The Stack streaming dirs.",
    )
    ap.add_argument(
        "--rotate-stream-every",
        type=int,
        default=500,
        help="Rotate one active stream every N kept samples (0 disables rotation).",
    )
    ap.add_argument("--use-auth-token", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-source-dirs",
        type=int,
        default=256,
        help="Cap how many non-mapped The Stack dirs to open.",
    )
    ap.add_argument(
        "--include-dirs",
        type=str,
        default="",
        help="Optional explicit comma-separated The Stack data dirs to use instead of auto-discovery.",
    )
    ap.add_argument(
        "--exclude-dirs",
        type=str,
        default="",
        help="Optional comma-separated The Stack data dirs to exclude.",
    )
    ap.add_argument(
        "--commit-first",
        type=int,
        default=32,
        help="Force an early durable commit after this many new samples (when starting from empty).",
    )
    ap.add_argument(
        "--commit-every",
        type=int,
        default=5000,
        help="Durable commit cadence (samples per save_to_disk checkpoint).",
    )
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--add-meta", action="store_true")
    ap.add_argument(
        "--progress",
        choices=("auto", "always", "never"),
        default="auto",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    out_root = args.out_root.resolve()
    split = str(args.split or "train").strip().lower()
    label = str(args.label or "other").strip().lower()
    if not split:
        raise SystemExit("--split must be non-empty.")
    if not label:
        raise SystemExit("--label must be non-empty.")
    if int(args.max_samples) <= 0:
        raise SystemExit("--max-samples must be > 0.")
    if int(args.max_bytes) <= 0:
        raise SystemExit("--max-bytes must be > 0.")
    if int(args.max_open_streams) <= 0:
        raise SystemExit("--max-open-streams must be > 0.")
    if int(args.commit_first) <= 0:
        raise SystemExit("--commit-first must be > 0.")
    if int(args.commit_every) <= 0:
        raise SystemExit("--commit-every must be > 0.")

    out_dir = out_root / split / label / "dataset"
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    ds_exist, seen_uids = _load_existing_for_output(out_dir, rebuild=bool(args.rebuild))
    existing_count = int(len(ds_exist)) if ds_exist is not None else 0
    target_total = int(args.max_samples)
    remaining = max(0, target_total - existing_count)
    print(
        f"extract_other: out_dir={out_dir} existing={existing_count} target={target_total} remaining={remaining}",
        flush=True,
    )
    if remaining <= 0:
        print("extract_other: nothing to add; target already reached.", flush=True)
        return

    include_dirs = _parse_csv_arg(args.include_dirs)
    exclude_dirs = set(_parse_csv_arg(args.exclude_dirs))
    if include_dirs:
        source_dirs = [d for d in include_dirs if d not in exclude_dirs]
    else:
        all_dirs = _list_stack_data_dirs(use_auth_token=bool(args.use_auth_token))
        mapped_dirs = _default_mapped_stack_dirs()
        source_dirs = [d for d in all_dirs if d not in mapped_dirs and d not in exclude_dirs]
    if not source_dirs:
        raise RuntimeError("No source dirs selected for OTHER extraction.")

    rng.shuffle(source_dirs)
    max_source_dirs = max(1, int(args.max_source_dirs))
    if len(source_dirs) > max_source_dirs:
        source_dirs = source_dirs[:max_source_dirs]
    max_open_streams = min(max(1, int(args.max_open_streams)), len(source_dirs))
    per_stream_shuffle = max(0, int(args.shuffle_buffer)) // max(1, max_open_streams)
    if int(args.shuffle_buffer) > 0 and per_stream_shuffle <= 0:
        per_stream_shuffle = 1
    print(
        "extract_other: "
        f"source_dirs={len(source_dirs)} "
        f"max_open_streams={max_open_streams} "
        f"global_shuffle_buffer={int(args.shuffle_buffer)} "
        f"per_stream_shuffle≈{per_stream_shuffle} "
        f"rotate_every={int(args.rotate_stream_every)} "
        f"(sample: {source_dirs[:10]})",
        flush=True,
    )

    token_arg: Optional[Any] = None
    if bool(args.use_auth_token):
        _, token_arg = resolve_hf_token_pair(True)

    features = {
        "content": Value("string"),
        "lang_id": Value("int16"),
        "uid": Value("string"),
        "stack_label": Value("string"),
    }
    if bool(args.add_meta):
        features.update(
            {
                "source_ext": Value("string"),
                "source_hexsha": Value("string"),
                "source_repo": Value("string"),
                "source_repo_path": Value("string"),
                "license": Value("string"),
            }
        )
    feats = Features(features)

    progress_mode = str(args.progress)
    if progress_mode == "auto":
        progress_mode = "always" if sys.stderr.isatty() else "never"

    commit_first = max(1, int(args.commit_first))
    commit_every = max(commit_first, int(args.commit_every))
    buffer_rows: List[Dict[str, Any]] = []
    new_written = 0

    row_iter = _iter_other_rows(
        source_dirs=source_dirs,
        shuffle_buffer=int(args.shuffle_buffer),
        max_open_streams=max_open_streams,
        rotate_stream_every=int(args.rotate_stream_every),
        token_arg=token_arg,
        max_samples=int(remaining),
        max_bytes=int(args.max_bytes),
        seen_uids=seen_uids,
        add_meta=bool(args.add_meta),
        progress=progress_mode,
    )

    for row in row_iter:
        buffer_rows.append(row)
        need_early_commit = (
            existing_count == 0
            and new_written == 0
            and len(buffer_rows) >= commit_first
        )
        need_regular_commit = len(buffer_rows) >= commit_every
        if need_early_commit or need_regular_commit:
            ds_exist, wrote = _flush_chunk_to_output(
                out_dir=out_dir,
                ds_exist=ds_exist,
                rows=buffer_rows,
                features=feats,
            )
            new_written += int(wrote)
            buffer_rows = []
            print(
                f"extract_other: checkpoint commit (+{wrote}, total_new={new_written}, total={existing_count + new_written})",
                flush=True,
            )

    if buffer_rows:
        ds_exist, wrote = _flush_chunk_to_output(
            out_dir=out_dir,
            ds_exist=ds_exist,
            rows=buffer_rows,
            features=feats,
        )
        new_written += int(wrote)
        print(
            f"extract_other: final commit (+{wrote}, total_new={new_written}, total={existing_count + new_written})",
            flush=True,
        )

    if new_written <= 0:
        print("extract_other: generator produced 0 new samples.", flush=True)
        return

    total_now = existing_count + new_written
    print(
        f"extract_other: wrote {new_written} new samples; total now {total_now} at {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
