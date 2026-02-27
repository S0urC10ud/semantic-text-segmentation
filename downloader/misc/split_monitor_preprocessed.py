#!/usr/bin/env python3
"""
Split an existing memmapped monitor set into two stratified subsets.

Given downloader/monitor_preprocessed built by prepare_monitor_set.py,
this script creates two new directories with the same memmap layout:
  - downloader/monitor_preprocessed_a : N_train files per content type
  - downloader/monitor_preprocessed_b : N_eval files per content type
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Repository root (text-segmentation/)
REPO_ROOT = Path(__file__).resolve().parents[2]


# Must mirror downloader/misc/prepare_monitor_set.py and train/utils/monitor_eval.py
FILE_DTYPE = np.dtype(
    [
        ("byte_start", "<i8"),
        ("byte_len", "<i4"),
        ("seg_start", "<i8"),
        ("seg_count", "<i4"),
        ("source", "u1"),
        ("type_id", "<i2"),
    ]
)
SEG_DTYPE = np.dtype(
    [
        ("file_id", "<i4"),
        ("start", "<i4"),
        ("end", "<i4"),
        ("label", "<i2"),
    ]
)


def _load_monitor_root(root: Path):
    meta_path = root / "meta.json"
    files_path = root / "files.npy"
    segments_path = root / "segments.npy"
    contents_path = root / "contents.bin"

    if not (
        meta_path.exists()
        and files_path.exists()
        and segments_path.exists()
        and contents_path.exists()
    ):
        raise FileNotFoundError(f"Monitor set is incomplete under {root}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    files = np.load(files_path, mmap_mode="r")
    segments = np.load(segments_path, mmap_mode="r")
    contents = np.memmap(contents_path, mode="r", dtype=np.uint8)

    lang2id = {k: int(v) for k, v in (meta.get("lang2id") or {}).items()}
    id2lang = {int(k): v for k, v in (meta.get("id2lang") or {}).items()}

    # Ensure id2lang is populated even if absent on disk.
    if not id2lang and lang2id:
        id2lang = {v: k for k, v in lang2id.items()}

    return {
        "meta": meta,
        "files": files,
        "segments": segments,
        "contents": contents,
        "lang2id": lang2id,
        "id2lang": id2lang,
    }


def _stratified_indices_by_type(files: np.ndarray) -> Dict[int, List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, row in enumerate(files):
        type_id = int(row["type_id"])
        groups[type_id].append(int(idx))
    return groups


def _build_split_indices(
    groups: Dict[int, List[int]],
    id2lang: Dict[int, str],
    train_per_type: int,
    eval_per_type: int,
) -> Tuple[List[int], List[int]]:
    train_indices: List[int] = []
    eval_indices: List[int] = []
    required = train_per_type + eval_per_type

    for type_id, idxs in sorted(groups.items(), key=lambda kv: kv[0]):
        label = id2lang.get(type_id, str(type_id))
        if len(idxs) < required:
            raise ValueError(
                f"Not enough files for type '{label}' (id={type_id}): "
                f"have {len(idxs)}, need at least {required}"
            )
        train_indices.extend(idxs[:train_per_type])
        eval_indices.extend(idxs[train_per_type : train_per_type + eval_per_type])

    return train_indices, eval_indices


def _write_split(
    out_dir: Path,
    split_name: str,
    indices: Sequence[int],
    base_meta: Dict,
    files: np.ndarray,
    segments: np.ndarray,
    contents: np.memmap,
    id2lang: Dict[int, str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    contents_path = out_dir / "contents.bin"
    files_path = out_dir / "files.npy"
    segments_path = out_dir / "segments.npy"
    meta_path = out_dir / "meta.json"

    buffer = bytearray()
    new_files = np.zeros(len(indices), dtype=FILE_DTYPE)
    seg_rows: List[Tuple[int, int, int, int]] = []

    for new_id, old_idx in enumerate(indices):
        row = files[int(old_idx)]
        byte_len = int(row["byte_len"])
        if byte_len <= 0:
            continue

        old_byte_start = int(row["byte_start"])
        byte_start = len(buffer)
        file_bytes = contents[old_byte_start : old_byte_start + byte_len]
        buffer.extend(file_bytes)

        old_seg_start = int(row["seg_start"])
        seg_count = int(row["seg_count"])
        seg_start = len(seg_rows)

        seg_slice = segments[old_seg_start : old_seg_start + seg_count]
        for seg in seg_slice:
            seg_rows.append(
                (
                    int(new_id),
                    int(seg["start"]),
                    int(seg["end"]),
                    int(seg["label"]),
                )
            )

        new_files[new_id] = (
            int(byte_start),
            int(byte_len),
            int(seg_start),
            int(seg_count),
            int(row["source"]),
            int(row["type_id"]),
        )

    with open(contents_path, "wb") as f:
        f.write(buffer)

    np.save(files_path, new_files)

    new_segments = np.zeros(len(seg_rows), dtype=SEG_DTYPE)
    for i, (fid, start, end, label_id) in enumerate(seg_rows):
        new_segments[i] = (int(fid), int(start), int(end), int(label_id))
    np.save(segments_path, new_segments)

    # Recompute simple stats for sanity.
    files_per_label: Counter[str] = Counter()
    bytes_per_label: Counter[str] = Counter()
    segment_bytes_per_label: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for row in new_files:
        type_id = int(row["type_id"])
        label = id2lang.get(type_id, str(type_id))
        files_per_label[label] += 1
        bytes_per_label[label] += int(row["byte_len"])
        src = int(row["source"])
        if src == 0:
            source_counts["segmented"] += 1
        elif src == 1:
            source_counts["pure"] += 1
        else:
            source_counts[str(src)] += 1

    for seg in new_segments:
        label_id = int(seg["label"])
        label = id2lang.get(label_id, str(label_id))
        segment_bytes_per_label[label] += int(seg["end"]) - int(seg["start"])

    meta = dict(base_meta)
    meta["num_files"] = int(len(new_files))
    meta["num_segments"] = int(len(new_segments))
    meta["total_bytes"] = int(len(buffer))
    meta["sources"] = dict(source_counts)
    meta["files_per_label"] = dict(files_per_label)
    meta["bytes_per_label"] = dict(bytes_per_label)
    meta["segment_bytes_per_label"] = dict(segment_bytes_per_label)
    meta["max_files_per_type"] = int(
        max(files_per_label.values()) if files_per_label else 0
    )
    meta["split_from"] = str(
        base_meta.get("paths", {}).get("segments_root", "")
    ) or str(base_meta.get("split_from", ""))
    meta["split_name"] = split_name

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split downloader/monitor_preprocessed into stratified monitor_preprocessed_a / _b."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "downloader" / "monitor_preprocessed",
        help="Root directory of the original monitor_preprocessed memmap.",
    )
    parser.add_argument(
        "--out-a",
        type=Path,
        default=REPO_ROOT / "downloader" / "monitor_preprocessed_a",
        help="Output directory for the A split (e.g., 800 per type).",
    )
    parser.add_argument(
        "--out-b",
        type=Path,
        default=REPO_ROOT / "downloader" / "monitor_preprocessed_b",
        help="Output directory for the B split (e.g., 200 per type).",
    )
    parser.add_argument(
        "--train-per-type",
        type=int,
        default=800,
        help="Number of files per content type for split A.",
    )
    parser.add_argument(
        "--eval-per-type",
        type=int,
        default=200,
        help="Number of files per content type for split B.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="(Deprecated) Unused; split is deterministic based on file order.",
    )
    args = parser.parse_args()

    data = _load_monitor_root(args.input_root)
    meta = data["meta"]
    files = data["files"]
    segments = data["segments"]
    contents = data["contents"]
    id2lang = data["id2lang"]

    groups = _stratified_indices_by_type(files)
    train_indices, eval_indices = _build_split_indices(
        groups,
        id2lang=id2lang,
        train_per_type=args.train_per_type,
        eval_per_type=args.eval_per_type,
    )

    print(
        f"Splitting monitor_preprocessed at {args.input_root} into:\n"
        f"  A: {args.out_a} ({len(train_indices)} files total)\n"
        f"  B: {args.out_b} ({len(eval_indices)} files total)"
    )

    _write_split(
        args.out_a,
        split_name="monitor_preprocessed_a",
        indices=train_indices,
        base_meta=meta,
        files=files,
        segments=segments,
        contents=contents,
        id2lang=id2lang,
    )
    _write_split(
        args.out_b,
        split_name="monitor_preprocessed_b",
        indices=eval_indices,
        base_meta=meta,
        files=files,
        segments=segments,
        contents=contents,
        id2lang=id2lang,
    )

    print("✅ Finished writing stratified monitor splits.")


if __name__ == "__main__":
    main()
