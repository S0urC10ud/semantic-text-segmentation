#!/usr/bin/env python3
"""
Apply stored monitor relabel refinements back into monitor_preprocessed_a/_b.

This script is intentionally strict:
- it only targets downloader/monitor_preprocessed_a and _b
- it verifies sample hashes against the current on-disk bytes
- it verifies snippet text against the current ASCII projection used by relabeling
- it requires contiguous full-file coverage for every applied refinement
- it writes new files.npy / segments.npy / meta.json via temp files, then replaces
  the targets only after validation succeeds

The raw contents.bin bytes are never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_PATH = (REPO_ROOT / "active_learning" / "label_store.sqlite").resolve()
DEFAULT_SPLIT_ROOTS = {
    "monitor_a": (REPO_ROOT / "downloader" / "monitor_preprocessed_a").resolve(),
    "monitor_b": (REPO_ROOT / "downloader" / "monitor_preprocessed_b").resolve(),
}

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

VISIBLE_BYTE_SET = set(range(0x20, 0x7F))


@dataclass(frozen=True)
class RefinementRow:
    row_id: int
    source_split: str
    source_lang: str
    sample_index: int
    sample_hash: str
    snippet_start: int
    snippet_end: int
    snippet_text: str
    refined_segments: Tuple[Dict[str, object], ...]


@dataclass(frozen=True)
class FileRefinement:
    source_split: str
    sample_index: int
    sample_hash: str
    rows: Tuple[RefinementRow, ...]


@dataclass
class SplitPlan:
    split_name: str
    root: Path
    new_files: np.ndarray
    new_segments: np.ndarray
    new_meta: Dict[str, object]
    file_updates: int
    files_total: int
    segments_before: int
    segments_after: int


def _parse_monitor_splits(raw: str) -> List[str]:
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected at least one monitor split.")
    unknown = sorted(set(parts) - set(DEFAULT_SPLIT_ROOTS))
    if unknown:
        raise ValueError(f"Unknown monitor splits: {unknown}; expected one of {sorted(DEFAULT_SPLIT_ROOTS)}")
    return parts


def _bytes_to_ascii_text(byte_values: Sequence[int]) -> str:
    chars: List[str] = []
    for raw in byte_values:
        value = int(raw)
        if value == 0x0D:
            chars.append("\n")
            continue
        if value in VISIBLE_BYTE_SET or value in (0x09, 0x0A):
            chars.append(chr(value))
        else:
            chars.append("?")
    return "".join(chars)


def _hash_monitor_file(split_name: str, file_idx: int, raw_bytes: np.ndarray) -> str:
    import hashlib

    h = hashlib.blake2s(digest_size=16)
    h.update(str(split_name).encode("utf-8", "ignore"))
    h.update(b":")
    h.update(str(int(file_idx)).encode("utf-8", "ignore"))
    h.update(b":")
    h.update(np.asarray(raw_bytes, dtype=np.uint8).tobytes())
    return h.hexdigest()


def _load_monitor_root(root: Path) -> Dict[str, object]:
    meta_path = root / "meta.json"
    files_path = root / "files.npy"
    segments_path = root / "segments.npy"
    contents_path = root / "contents.bin"
    if not (meta_path.exists() and files_path.exists() and segments_path.exists() and contents_path.exists()):
        raise FileNotFoundError(f"Monitor set is incomplete under {root}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    files = np.load(files_path, mmap_mode="r")
    segments = np.load(segments_path, mmap_mode="r")
    contents = np.memmap(contents_path, mode="r", dtype=np.uint8)
    lang2id = {str(k): int(v) for k, v in (meta.get("lang2id") or {}).items()}
    id2lang = {int(k): str(v) for k, v in (meta.get("id2lang") or {}).items()}
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


def _resolve_round_id(store_path: Path, explicit_round_id: Optional[str], latest_monitor_round: bool) -> str:
    if explicit_round_id:
        return str(explicit_round_id)
    if not latest_monitor_round:
        raise ValueError("Provide --round-id explicitly, or pass --latest-monitor-round.")
    with sqlite3.connect(store_path) as con:
        row = con.execute(
            """
            SELECT round_id
            FROM refinements
            WHERE round_id LIKE 'monitor-relabel-%'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None or not row[0]:
        raise ValueError("No monitor-relabel round found in label_store.sqlite.")
    return str(row[0])


def _iter_refinement_rows(
    store_path: Path,
    *,
    round_id: str,
    splits: Sequence[str],
) -> Iterator[RefinementRow]:
    placeholders = ",".join("?" for _ in splits)
    query = f"""
        SELECT id, source_split, source_lang, sample_index, sample_hash,
               snippet_start, snippet_end, snippet_text, refined_segments_json
        FROM refinements
        WHERE round_id = ? AND status = 'ok' AND source_split IN ({placeholders})
        ORDER BY source_split ASC, sample_index ASC, snippet_start ASC, id ASC
    """
    params: List[object] = [str(round_id), *list(splits)]
    con = sqlite3.connect(store_path)
    con.row_factory = sqlite3.Row
    try:
        for row in con.execute(query, params):
            try:
                refined = json.loads(row["refined_segments_json"] or "[]")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Row {row['id']} has invalid refined_segments_json") from exc
            if not isinstance(refined, list):
                raise ValueError(f"Row {row['id']} refined_segments_json is not a list")
            yield RefinementRow(
                row_id=int(row["id"]),
                source_split=str(row["source_split"]),
                source_lang=str(row["source_lang"]),
                sample_index=int(row["sample_index"]),
                sample_hash=str(row["sample_hash"]),
                snippet_start=int(row["snippet_start"]),
                snippet_end=int(row["snippet_end"]),
                snippet_text=str(row["snippet_text"] or ""),
                refined_segments=tuple(refined),
            )
    finally:
        con.close()


def _group_refinements(rows: Iterable[RefinementRow]) -> Dict[str, Dict[int, FileRefinement]]:
    grouped: Dict[str, Dict[int, List[RefinementRow]]] = defaultdict(lambda: defaultdict(list))
    hashes: Dict[Tuple[str, int], str] = {}
    for row in rows:
        key = (row.source_split, int(row.sample_index))
        prev_hash = hashes.get(key)
        if prev_hash is None:
            hashes[key] = row.sample_hash
        elif prev_hash != row.sample_hash:
            raise ValueError(
                f"Sample {row.source_split}:{row.sample_index} has multiple sample hashes: "
                f"{prev_hash} vs {row.sample_hash}"
            )
        grouped[row.source_split][int(row.sample_index)].append(row)

    out: Dict[str, Dict[int, FileRefinement]] = {}
    for split_name, by_index in grouped.items():
        split_map: Dict[int, FileRefinement] = {}
        for sample_index, sample_rows in by_index.items():
            ordered = tuple(sorted(sample_rows, key=lambda row: (row.snippet_start, row.row_id)))
            split_map[int(sample_index)] = FileRefinement(
                source_split=str(split_name),
                sample_index=int(sample_index),
                sample_hash=str(ordered[0].sample_hash),
                rows=ordered,
            )
        out[str(split_name)] = split_map
    return out


def _validate_and_build_segments(
    *,
    split_name: str,
    file_idx: int,
    raw_bytes: np.ndarray,
    refinement: FileRefinement,
    label_to_id: Mapping[str, int],
) -> List[Tuple[int, int, int]]:
    current_hash = _hash_monitor_file(split_name, file_idx, raw_bytes)
    if current_hash != refinement.sample_hash:
        raise ValueError(
            f"Hash mismatch for {split_name}:{file_idx}: current={current_hash}, stored={refinement.sample_hash}"
        )

    file_text = _bytes_to_ascii_text(np.asarray(raw_bytes, dtype=np.uint8))
    file_len = len(file_text)
    pieces: List[Tuple[int, int, str]] = []

    expected_start = 0
    for row in refinement.rows:
        snippet_len = int(row.snippet_end) - int(row.snippet_start)
        if snippet_len < 0:
            raise ValueError(f"Negative snippet length for row {row.row_id}")
        if int(row.snippet_start) != expected_start:
            raise ValueError(
                f"Non-contiguous snippet coverage for {split_name}:{file_idx}: "
                f"expected start {expected_start}, got {row.snippet_start}"
            )
        if int(row.snippet_end) > file_len:
            raise ValueError(
                f"Snippet end out of range for {split_name}:{file_idx}: "
                f"{row.snippet_end} > {file_len}"
            )
        current_snippet = file_text[int(row.snippet_start) : int(row.snippet_end)]
        if current_snippet != row.snippet_text:
            raise ValueError(
                f"Snippet text mismatch for {split_name}:{file_idx} at "
                f"[{row.snippet_start}:{row.snippet_end}]"
            )
        if len(row.snippet_text) != snippet_len:
            raise ValueError(
                f"Snippet text length mismatch for {split_name}:{file_idx}: "
                f"{len(row.snippet_text)} vs expected {snippet_len}"
            )

        prev_local_end = 0
        for seg in row.refined_segments:
            try:
                start = int(seg["start"])
                end = int(seg["end"])
                label = str(seg["label"])
            except Exception as exc:
                raise ValueError(f"Invalid segment payload in row {row.row_id}: {seg!r}") from exc
            if label not in label_to_id:
                raise ValueError(
                    f"Unknown label '{label}' in row {row.row_id} for {split_name}:{file_idx}"
                )
            if start != prev_local_end:
                raise ValueError(
                    f"Non-contiguous local segments in row {row.row_id}: "
                    f"expected {prev_local_end}, got {start}"
                )
            if end <= start or end > snippet_len:
                raise ValueError(
                    f"Invalid local segment bounds in row {row.row_id}: [{start}, {end}) / {snippet_len}"
                )
            pieces.append((int(row.snippet_start) + start, int(row.snippet_start) + end, label))
            prev_local_end = end
        if prev_local_end != snippet_len:
            raise ValueError(
                f"Row {row.row_id} does not fully cover its snippet: ended at {prev_local_end}, "
                f"expected {snippet_len}"
            )
        expected_start = int(row.snippet_end)

    if expected_start != file_len:
        raise ValueError(
            f"Refinement does not cover full file for {split_name}:{file_idx}: "
            f"ended at {expected_start}, expected {file_len}"
        )

    merged: List[Tuple[int, int, int]] = []
    for start, end, label in pieces:
        label_id = int(label_to_id[label])
        if not merged:
            merged.append((start, end, label_id))
            continue
        last_start, last_end, last_label_id = merged[-1]
        if start != last_end:
            raise ValueError(
                f"Global segment coverage gap/overlap for {split_name}:{file_idx}: "
                f"expected {last_end}, got {start}"
            )
        if last_label_id == label_id:
            merged[-1] = (last_start, end, last_label_id)
        else:
            merged.append((start, end, label_id))

    if not merged:
        raise ValueError(f"No refined segments for {split_name}:{file_idx}")
    if merged[0][0] != 0 or merged[-1][1] != file_len:
        raise ValueError(
            f"Merged refined segments do not span full file for {split_name}:{file_idx}: "
            f"{merged[0][0]}..{merged[-1][1]} of {file_len}"
        )
    return merged


def _existing_segments_for_file(row: np.void, segments: np.ndarray) -> List[Tuple[int, int, int]]:
    seg_start = int(row["seg_start"])
    seg_count = int(row["seg_count"])
    seg_slice = segments[seg_start : seg_start + seg_count]
    return [
        (int(seg["start"]), int(seg["end"]), int(seg["label"]))
        for seg in seg_slice
    ]


def _recompute_meta(
    *,
    base_meta: Mapping[str, object],
    files: np.ndarray,
    segments: np.ndarray,
    contents_len: int,
    id2lang: Mapping[int, str],
) -> Dict[str, object]:
    files_per_label: Counter[str] = Counter()
    bytes_per_label: Counter[str] = Counter()
    segment_bytes_per_label: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for row in files:
        type_id = int(row["type_id"])
        label = str(id2lang.get(type_id, str(type_id)))
        files_per_label[label] += 1
        bytes_per_label[label] += int(row["byte_len"])
        src = int(row["source"])
        if src == 0:
            source_counts["segmented"] += 1
        elif src == 1:
            source_counts["pure"] += 1
        else:
            source_counts[str(src)] += 1

    for seg in segments:
        label_id = int(seg["label"])
        label = str(id2lang.get(label_id, str(label_id)))
        segment_bytes_per_label[label] += int(seg["end"]) - int(seg["start"])

    meta = dict(base_meta)
    meta["num_files"] = int(len(files))
    meta["num_segments"] = int(len(segments))
    meta["total_bytes"] = int(contents_len)
    meta["sources"] = dict(source_counts)
    meta["files_per_label"] = dict(files_per_label)
    meta["bytes_per_label"] = dict(bytes_per_label)
    meta["segment_bytes_per_label"] = dict(segment_bytes_per_label)
    meta["max_files_per_type"] = int(max(files_per_label.values()) if files_per_label else 0)
    return meta


def build_split_plan(
    *,
    split_name: str,
    root: Path,
    refinements: Mapping[int, FileRefinement],
) -> SplitPlan:
    data = _load_monitor_root(root)
    meta = data["meta"]
    files = data["files"]
    segments = data["segments"]
    contents = data["contents"]
    lang2id = data["lang2id"]
    id2lang = data["id2lang"]

    new_files = np.zeros(len(files), dtype=FILE_DTYPE)
    seg_rows: List[Tuple[int, int, int, int]] = []
    file_updates = 0

    missing_indices = sorted(set(refinements) - set(range(len(files))))
    if missing_indices:
        raise ValueError(f"{split_name} contains refinement rows for out-of-range files: {missing_indices[:10]}")

    for file_idx, row in enumerate(files):
        byte_start = int(row["byte_start"])
        byte_len = int(row["byte_len"])
        seg_start = len(seg_rows)

        if int(file_idx) in refinements:
            raw = np.asarray(contents[byte_start : byte_start + byte_len], dtype=np.uint8)
            merged = _validate_and_build_segments(
                split_name=split_name,
                file_idx=int(file_idx),
                raw_bytes=raw,
                refinement=refinements[int(file_idx)],
                label_to_id=lang2id,
            )
            current = _existing_segments_for_file(row, segments)
            if current != merged:
                file_updates += 1
            for start, end, label_id in merged:
                seg_rows.append((int(file_idx), int(start), int(end), int(label_id)))
            seg_count = len(merged)
        else:
            current = _existing_segments_for_file(row, segments)
            for start, end, label_id in current:
                seg_rows.append((int(file_idx), int(start), int(end), int(label_id)))
            seg_count = len(current)

        new_files[file_idx] = (
            int(byte_start),
            int(byte_len),
            int(seg_start),
            int(seg_count),
            int(row["source"]),
            int(row["type_id"]),
        )

    new_segments = np.zeros(len(seg_rows), dtype=SEG_DTYPE)
    for i, (fid, start, end, label_id) in enumerate(seg_rows):
        new_segments[i] = (int(fid), int(start), int(end), int(label_id))

    new_meta = _recompute_meta(
        base_meta=meta,
        files=new_files,
        segments=new_segments,
        contents_len=int(len(contents)),
        id2lang=id2lang,
    )
    return SplitPlan(
        split_name=str(split_name),
        root=root,
        new_files=new_files,
        new_segments=new_segments,
        new_meta=new_meta,
        file_updates=int(file_updates),
        files_total=int(len(files)),
        segments_before=int(len(segments)),
        segments_after=int(len(new_segments)),
    )


def _write_temp_npy(root: Path, stem: str, array: np.ndarray) -> Path:
    fd, tmp_path = tempfile.mkstemp(prefix=f".{stem}.", suffix=".npy", dir=root)
    with open(fd, "wb", closefd=True) as f:
        np.save(f, array)
    return Path(tmp_path)


def _write_temp_json(root: Path, stem: str, payload: Mapping[str, object]) -> Path:
    fd, tmp_path = tempfile.mkstemp(prefix=f".{stem}.", suffix=".json", dir=root)
    with open(fd, "w", closefd=True) as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return Path(tmp_path)


def _backup_targets(root: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ("files.npy", "segments.npy", "meta.json"):
        shutil.copy2(root / name, backup_dir / name)


def apply_split_plan(plan: SplitPlan) -> Path:
    root = plan.root
    files_tmp = _write_temp_npy(root, "files", plan.new_files)
    segments_tmp = _write_temp_npy(root, "segments", plan.new_segments)
    meta_tmp = _write_temp_json(root, "meta", plan.new_meta)

    # Re-open the temp outputs before replacing anything.
    loaded_files = np.load(files_tmp, mmap_mode="r")
    loaded_segments = np.load(segments_tmp, mmap_mode="r")
    if loaded_files.shape != plan.new_files.shape:
        raise ValueError(f"Temporary files.npy shape mismatch for {root}")
    if loaded_segments.shape != plan.new_segments.shape:
        raise ValueError(f"Temporary segments.npy shape mismatch for {root}")
    with open(meta_tmp, "r") as f:
        meta_loaded = json.load(f)
    if int(meta_loaded.get("num_segments", -1)) != int(len(plan.new_segments)):
        raise ValueError(f"Temporary meta.json num_segments mismatch for {root}")

    backup_dir = root / ".relabel_writeback_backups" / plan.new_meta["split_name"]  # type: ignore[index]
    suffix = str(plan.new_meta.get("writeback_round_id", "manual"))
    backup_dir = backup_dir / suffix
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")
    _backup_targets(root, backup_dir)

    try:
        files_tmp.replace(root / "files.npy")
        segments_tmp.replace(root / "segments.npy")
        meta_tmp.replace(root / "meta.json")
    except Exception:
        shutil.copy2(backup_dir / "files.npy", root / "files.npy")
        shutil.copy2(backup_dir / "segments.npy", root / "segments.npy")
        shutil.copy2(backup_dir / "meta.json", root / "meta.json")
        raise
    finally:
        for tmp in (files_tmp, segments_tmp, meta_tmp):
            tmp.unlink(missing_ok=True)

    return backup_dir


def run_writeback(args: argparse.Namespace) -> Dict[str, object]:
    splits = _parse_monitor_splits(args.monitor_splits)
    round_id = _resolve_round_id(Path(args.store), args.round_id, bool(args.latest_monitor_round))
    grouped = _group_refinements(
        _iter_refinement_rows(
            Path(args.store),
            round_id=round_id,
            splits=splits,
        )
    )
    if not grouped:
        raise ValueError(f"No refinements found for round_id={round_id!r} and splits={splits}")

    split_plans: List[SplitPlan] = []
    for split_name in splits:
        root = DEFAULT_SPLIT_ROOTS[split_name]
        refinements = grouped.get(split_name, {})
        plan = build_split_plan(split_name=split_name, root=root, refinements=refinements)
        plan.new_meta["writeback_round_id"] = str(round_id)
        split_plans.append(plan)

    total_updates = 0
    total_segments_before = 0
    total_segments_after = 0
    for plan in split_plans:
        total_updates += int(plan.file_updates)
        total_segments_before += int(plan.segments_before)
        total_segments_after += int(plan.segments_after)
        print(
            f"[monitor_writeback] {plan.split_name}: files={plan.files_total}, "
            f"updated={plan.file_updates}, segments={plan.segments_before}->{plan.segments_after}, "
            f"root={plan.root}",
            flush=True,
        )

    if not args.apply:
        print(
            f"[monitor_writeback] dry-run ok for round_id={round_id}; "
            f"would update {total_updates} files across {len(split_plans)} splits.",
            flush=True,
        )
        return {
            "status": "dry_run",
            "round_id": round_id,
            "splits": {plan.split_name: plan.file_updates for plan in split_plans},
            "updated_files": int(total_updates),
            "segments_before": int(total_segments_before),
            "segments_after": int(total_segments_after),
        }

    backup_dirs: Dict[str, str] = {}
    for plan in split_plans:
        backup_dir = apply_split_plan(plan)
        backup_dirs[plan.split_name] = str(backup_dir)
        print(
            f"[monitor_writeback] applied {plan.split_name}: updated={plan.file_updates}, "
            f"backup={backup_dir}",
            flush=True,
        )
    return {
        "status": "ok",
        "round_id": round_id,
        "splits": {plan.split_name: plan.file_updates for plan in split_plans},
        "updated_files": int(total_updates),
        "segments_before": int(total_segments_before),
        "segments_after": int(total_segments_after),
        "backup_dirs": backup_dirs,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply stored monitor relabel refinements back into monitor_preprocessed_a/_b.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE_PATH,
        help="Path to label_store.sqlite containing monitor relabel refinements.",
    )
    parser.add_argument(
        "--round-id",
        type=str,
        default=None,
        help="Explicit monitor-relabel round id to apply.",
    )
    parser.add_argument(
        "--latest-monitor-round",
        action="store_true",
        help="Resolve the latest round whose id starts with 'monitor-relabel-'.",
    )
    parser.add_argument(
        "--monitor-splits",
        type=str,
        default="monitor_a,monitor_b",
        help="Comma-separated subset of {monitor_a,monitor_b}.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite files.npy / segments.npy / meta.json in place. Default is dry-run.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        result = run_writeback(args)
    except Exception as exc:
        print(f"[monitor_writeback] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
