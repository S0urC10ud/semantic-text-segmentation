#!/usr/bin/env python3
"""
Build a memmap-friendly monitor set that blends:
  1) Gemini segmentations under gemini_segmentations/monitor/<type>
  2) Pure (unsegmented) monitor samples from downloader/arrow_out/monitor/<type>

Output is written to downloader/monitor_preprocessed with:
  - contents.bin   : raw bytes for all files (np.memmap'able)
  - files.npy      : structured index into contents.bin and segments.npy
  - segments.npy   : structured segment rows (file_id, start, end, label_id)
  - meta.json      : mapping + stats for sanity checks
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg  # noqa: E402

# Structured dtypes (kept small for mmap speed)
FILE_DTYPE = np.dtype(
    [
        ("byte_start", "<i8"),  # offset into contents.bin
        ("byte_len", "<i4"),    # length of this file in bytes
        ("seg_start", "<i8"),   # offset into segments.npy
        ("seg_count", "<i4"),   # how many segments belong to this file
        ("source", "u1"),       # 0 = segmented, 1 = pure/arrow
        ("type_id", "<i2"),     # canonical label id for the file's declared type
    ]
)
SEG_DTYPE = np.dtype(
    [
        ("file_id", "<i4"),
        ("start", "<i4"),  # byte offsets relative to the file start
        ("end", "<i4"),
        ("label", "<i2"),
    ]
)

SOURCE_SEGMENTED = 0
SOURCE_PURE = 1


ENCODING_METHODS = {"hex", "base64", "base32", "base58", "base85"}


def _is_derived_label(label: str) -> bool:
    if not label:
        return False
    if not label.startswith("encoding_"):
        return False
    suffix = label.split("encoding_", 1)[1]
    return suffix in ENCODING_METHODS


def _canonical_type(raw: str, known: Set[str], top_level_hint: str | None = None) -> str:
    key = (raw or "").strip().lower().replace(" ", "_")
    key = key.replace("-", "_").replace(".", "_")
    if not key and top_level_hint:
        key = top_level_hint

    alias_map = {
        "js": "javascript",
        "jsx": "javascript",
        "tsx": "typescript",
        "ts": "typescript",
        "shell_batchfile": "shell",
        "batchfile": "shell",
        "bat": "shell",
        "cmd": "shell",
        "bash": "shell",
        "sh": "shell",
        "zsh": "shell",
        "ps": "powershell",
        "ps1": "powershell",
        "vb": "visual_basic",
        "vbnet": "visual_basic",
        "vb_net": "visual_basic",
        "visualbasic": "visual_basic",
        "c++": "c_family",
        "cpp": "c_family",
        "c-family": "c_family",
        "cfamily": "c_family",
        "c#": "csharp",
        "c-sharp": "csharp",
        "yml": "yaml",
        "htm": "html",
        "md": "markdown",
        "rst": "restructuredtext",
        "plaintext": "text",
        "plain_text": "text",
    }
    key = alias_map.get(key, key)

    if key.startswith("other_") or key.startswith("discovered_"):
        return "other"

    if "template" in key or key.endswith("_config"):
        return "other"

    # Merge close variants into canonical training buckets
    if key in {"javascript", "typescript"}:
        key = "javascript_typescript"
    if key in {"c", "h"}:
        key = "c_family"
    if key == "makefile":
        return "other"
    if key == "batch":
        key = "shell"

    if key not in known:
        if top_level_hint and top_level_hint in known:
            return top_level_hint
        return "other"
    return key


_SVG_STYLE_BLOCK_RE = re.compile(r"(?is)<\s*style\b[^>]*>.*?</\s*style\s*>")
_SVG_STYLE_SELF_CLOSE_RE = re.compile(r"(?is)<\s*style\b[^>]*/>")


def _strip_svg_style_tags(text: str) -> str:
    """
    Remove CSS <style> blocks from SVG content.

    This is applied only to Arrow-only SVG monitor files before they
    are added to the memmapped monitor set.
    """
    if not text:
        return ""
    current = text
    prev = None
    # Iteratively strip nested/duplicated style blocks, if any.
    while prev != current:
        prev = current
        current = _SVG_STYLE_BLOCK_RE.sub("", current)
    current = _SVG_STYLE_SELF_CLOSE_RE.sub("", current)
    return current


class MagikaFileFilter:
    """
    Lightweight Magika wrapper for file-level type checks.

    We intentionally do NOT use window-wise Magika like downloader/0_main.py;
    each monitor example is treated as a whole file here.
    """

    def __init__(self, score_threshold: float = 0.9) -> None:
        try:
            from magika import Magika  # type: ignore
        except Exception as e:  # pragma: no cover - import-time failure is fatal
            raise RuntimeError(
                "Magika is required for monitor filtering. "
                "Install magika==1.0.* to run 999_prepare_monitor_set."
            ) from e
        self._m = Magika()
        self.score_threshold = float(score_threshold)
        self.total: int = 0
        self.accepted: int = 0
        self.rejected_low_score: int = 0
        self.rejected_mismatch: int = 0
        self.rejected_error: int = 0

    def accept(self, content: bytes, canon_declared: str, known_labels: Set[str]) -> bool:
        """
        Return True if Magika confidently agrees with the declared canonical type.
        """
        self.total += 1

        # Derived encodings are generated locally and do not have a meaningful
        # Magika label; accept them without Magika filtering.
        if _is_derived_label(canon_declared):
            self.accepted += 1
            return True

        if not content:
            self.rejected_error += 1
            return False

        try:
            res = self._m.identify_bytes(content)
        except Exception:
            self.rejected_error += 1
            return False

        ok = bool(getattr(res, "ok", False))
        if not ok:
            self.rejected_error += 1
            return False

        score = float(getattr(res, "score", 0.0))
        if score < self.score_threshold:
            self.rejected_low_score += 1
            return False

        output = getattr(res, "output", None)
        magika_label: Optional[str] = getattr(output, "label", None) if output is not None else None
        if not magika_label:
            self.rejected_error += 1
            return False

        canon_magika = _canonical_type(magika_label, known_labels, top_level_hint=canon_declared)
        if canon_magika != canon_declared:
            self.rejected_mismatch += 1
            return False

        self.accepted += 1
        return True


class MonitorAccumulator:
    def __init__(self, known_labels: Set[str]):
        self.known_labels = known_labels
        self.buffer = bytearray()
        self.file_rows: List[Dict[str, int]] = []
        self.segment_rows: List[Tuple[int, int, int, int]] = []
        self.files_per_label: Counter[str] = Counter()
        self.bytes_per_label: Counter[str] = Counter()
        self.segment_bytes_per_label: Counter[str] = Counter()
        self.source_counts: Counter[str] = Counter()
        self.skipped_empty = 0

    def add_file(
        self,
        byte_data: bytes,
        segments: Sequence[Tuple[int, int, str]],
        type_label: str,
        source: str,
    ) -> None:
        if not byte_data:
            self.skipped_empty += 1
            return
        canon_type = _canonical_type(type_label, self.known_labels)
        byte_start = len(self.buffer)
        self.buffer.extend(byte_data)

        file_id = len(self.file_rows)
        seg_start = len(self.segment_rows)
        seg_count = 0
        other_id = getattr(cfg, "OTHER_CLASS_INDEX", cfg.NUM_CLASSES)
        for start, end, label in segments:
            if end <= start:
                continue
            canon_label = _canonical_type(label, self.known_labels, top_level_hint=canon_type)
            # Map unknown/long-tail labels into a derived "other" bucket that
            # does not correspond to an explicit model logit.
            label_id = cfg.LANG2ID.get(canon_label, other_id)
            self.segment_rows.append((file_id, int(start), int(end), int(label_id)))
            self.segment_bytes_per_label[canon_label] += int(end - start)
            seg_count += 1

        if seg_count == 0:
            self.skipped_empty += 1
            return

        self.file_rows.append(
            {
                "byte_start": byte_start,
                "byte_len": len(byte_data),
                "seg_start": seg_start,
                "seg_count": seg_count,
                "source": SOURCE_SEGMENTED if source == "segmented" else SOURCE_PURE,
                # File-level type id mirrors the per-segment mapping above.
                "type_id": cfg.LANG2ID.get(canon_type, other_id),
            }
        )
        self.source_counts[source] += 1
        self.files_per_label[canon_type] += 1
        self.bytes_per_label[canon_type] += len(byte_data)


def _iter_segmented_files(seg_root: Path) -> Iterable[Tuple[bytes, List[Tuple[int, int, str]], str]]:
    for type_dir in sorted(seg_root.iterdir()):
        if not type_dir.is_dir():
            continue
        declared_type = type_dir.name
        for path in sorted(type_dir.glob("*.json")):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"⚠️  Skipping unreadable segmentation {path}: {e}")
                continue
            segments = data.get("segments") or []
            if not segments:
                continue
            content = bytearray()
            seg_list: List[Tuple[int, int, str]] = []
            for seg in segments:
                text = seg.get("content") or ""
                raw_type = seg.get("type") or declared_type
                encoded = text.encode("utf-8", "ignore")
                if not encoded:
                    continue
                start = len(content)
                content.extend(encoded)
                seg_list.append((start, len(content), raw_type))
            if content:
                yield bytes(content), seg_list, declared_type


def build_monitor_set(
    seg_root: Path,
    arrow_root: Path,
    out_dir: Path,
    max_files_per_type: int = 0,
    magika_threshold: float = 0.9,
) -> Dict[str, int]:
    cfg.update_lang_mappings()
    known_labels = set(cfg.LANG2ID.keys())
    acc = MonitorAccumulator(known_labels)
    # Use Magika to filter monitor examples for types that only have Arrow data
    magika_filter = MagikaFileFilter(score_threshold=float(magika_threshold))

    segmented_labels = (
        {_canonical_type(p.name, known_labels) for p in seg_root.iterdir() if p.is_dir()}
        if seg_root.exists()
        else set()
    )

    type_cap = int(max_files_per_type) if max_files_per_type and max_files_per_type > 0 else 0
    type_counts: Counter[str] = Counter()

    print(f"📥 Loading segmented monitor files from: {seg_root}")
    for content, segs, declared in _iter_segmented_files(seg_root):
        canon_declared = _canonical_type(declared, known_labels)
        if type_cap and type_counts[canon_declared] >= type_cap:
            continue
        acc.add_file(content, segs, declared, "segmented")
        type_counts[canon_declared] += 1
    print(f"   ↳ Segmented files: {acc.source_counts.get('segmented', 0)}")

    print(f"📥 Loading pure monitor files from: {arrow_root}")
    for type_dir in sorted(arrow_root.iterdir()):
        if not type_dir.is_dir():
            continue
        declared_type = type_dir.name
        canon_declared = _canonical_type(declared_type, known_labels)
        if canon_declared in segmented_labels:
            # This type already has Gemini segmentations; we only need Arrow-only types here.
            continue
        if type_cap and type_counts[canon_declared] >= type_cap:
            # Already satisfied the cap for this type (from segmented files).
            continue

        ds_path = type_dir / "dataset"
        if not ds_path.exists():
            continue

        ds = load_from_disk(str(ds_path))
        for ex in ds:
            if type_cap and type_counts[canon_declared] >= type_cap:
                break
            text = ex.get("content") or ""
            if declared_type == "svg":
                # Strip embedded CSS from SVG monitor files before they enter the set.
                text = _strip_svg_style_tags(text)
            encoded = text.encode("utf-8", "ignore")
            if not encoded:
                continue
            if not magika_filter.accept(encoded, canon_declared, known_labels):
                # Below Magika threshold or mismatched type; try the next file.
                continue
            seg = (0, len(encoded), declared_type)
            acc.add_file(encoded, [seg], declared_type, "pure")
            type_counts[canon_declared] += 1
    print(
        f"   ↳ Pure files: {acc.source_counts.get('pure', 0)} "
        f"(Magika kept {magika_filter.accepted}/{magika_filter.total}, "
        f"low_score={magika_filter.rejected_low_score}, "
        f"mismatch={magika_filter.rejected_mismatch}, "
        f"errors={magika_filter.rejected_error})"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    contents_path = out_dir / "contents.bin"
    files_path = out_dir / "files.npy"
    segments_path = out_dir / "segments.npy"
    meta_path = out_dir / "meta.json"

    # Write contents.bin
    with open(contents_path, "wb") as f:
        f.write(acc.buffer)

    files_arr = np.zeros(len(acc.file_rows), dtype=FILE_DTYPE)
    for i, row in enumerate(acc.file_rows):
        files_arr[i] = (
            int(row["byte_start"]),
            int(row["byte_len"]),
            int(row["seg_start"]),
            int(row["seg_count"]),
            int(row["source"]),
            int(row["type_id"]),
        )
    np.save(files_path, files_arr)

    segments_arr = np.zeros(len(acc.segment_rows), dtype=SEG_DTYPE)
    for i, (fid, start, end, label_id) in enumerate(acc.segment_rows):
        segments_arr[i] = (int(fid), int(start), int(end), int(label_id))
    np.save(segments_path, segments_arr)

    meta = {
        "lang2id": cfg.LANG2ID,
        "id2lang": cfg.ID2LANG,
        "num_files": len(acc.file_rows),
        "num_segments": len(acc.segment_rows),
        "total_bytes": len(acc.buffer),
        "window_bytes": cfg.MODEL_WINDOW_BYTES,
        "sources": dict(acc.source_counts),
        "files_per_label": dict(acc.files_per_label),
        "bytes_per_label": dict(acc.bytes_per_label),
        "segment_bytes_per_label": dict(acc.segment_bytes_per_label),
        "skipped_empty": acc.skipped_empty,
        "max_files_per_type": type_cap,
        "magika": {
            "score_threshold": magika_filter.score_threshold,
            "total_evaluated": magika_filter.total,
            "accepted": magika_filter.accepted,
            "rejected_low_score": magika_filter.rejected_low_score,
            "rejected_mismatch": magika_filter.rejected_mismatch,
            "rejected_error": magika_filter.rejected_error,
        },
        "paths": {
            "segments_root": str(seg_root),
            "arrow_root": str(arrow_root),
        },
        "dtypes": {
            "files": str(FILE_DTYPE),
            "segments": str(SEG_DTYPE),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "files": len(acc.file_rows),
        "segments": len(acc.segment_rows),
        "bytes": len(acc.buffer),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare memmap-friendly monitor set.")
    parser.add_argument(
        "--segments-root",
        type=Path,
        default=REPO_ROOT / "gemini_segmentations" / "monitor",
        help="Root of segmented monitor JSON files.",
    )
    parser.add_argument(
        "--arrow-root",
        type=Path,
        default=REPO_ROOT / "downloader" / "arrow_out" / "monitor",
        help="Root of pure monitor Arrow datasets.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "downloader" / "monitor_preprocessed",
        help="Output directory for memmap files.",
    )
    parser.add_argument(
        "--max-files-per-type",
        type=int,
        default=1000,
        help="Cap files ingested per canonical type (0 = no cap).",
    )
    parser.add_argument(
        "--magika-threshold",
        type=float,
        default=0.9,
        help="Minimum Magika confidence score for Arrow-only monitor files.",
    )
    args = parser.parse_args()

    stats = build_monitor_set(
        args.segments_root,
        args.arrow_root,
        args.out_dir,
        max_files_per_type=args.max_files_per_type,
        magika_threshold=args.magika_threshold,
    )
    print(
        f"✅ Wrote monitor set to {args.out_dir} "
        f"({stats['files']} files, {stats['segments']} segments, {stats['bytes']:,} bytes)"
    )


if __name__ == "__main__":
    main()
