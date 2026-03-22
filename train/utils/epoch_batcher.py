"""
Functions for generating training windows from datasets, with lightweight
epoch-style progress tracking that matches the sampling distribution used
for validation.
"""
import queue
import random
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import datasets as hfds
import numpy as np
import utils.config as cfg
from utils.full_sequence import build_monitor_file_sequence, pack_sequence_pieces
from utils.window_generator import (
    _choose_window_mode,
    _make_markdown_window_impl,
    make_line_injected_window,
    make_mixed_window,
    make_training_window,
)

if TYPE_CHECKING:
    from utils.config import DataConfig


_PLACEHOLDER_CHAR = "\u00A4"
_VISIBLE_ASCII_BYTES = tuple(range(0x20, 0x7F))
_ALLOWED_TEXT_CHARS = {chr(b) for b in _VISIBLE_ASCII_BYTES}
_ALLOWED_TEXT_CHARS.update({"\n", "\t", _PLACEHOLDER_CHAR})
_MONITOR_SUBSTRING_REMOVAL_PROB = 0.5
_MONITOR_SUBSTRING_REMOVAL_MAX_REMOVALS = 3
_MONITOR_SPACE_BYTE = 0x20
_MONITOR_NEWLINE_BYTE = 0x0A

MonitorSegment = Tuple[int, int, int]


def _normalize_monitor_text(text: str) -> str:
    out_chars: List[str] = []
    for ch in text:
        if ch == "\r":
            ch = "\n"
        if ch in _ALLOWED_TEXT_CHARS:
            out_chars.append(ch)
        else:
            out_chars.append(_PLACEHOLDER_CHAR)
    return "".join(out_chars)


def _label_name_to_id(label_name: str) -> Optional[int]:
    low = str(label_name or "").strip().lower()
    if not low:
        return None
    if low in cfg.LANG2ID:
        return int(cfg.LANG2ID[low])
    other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
    if low == "other" and other_idx is not None:
        return int(other_idx)
    return None


def _sample_monitor_file_index(
    rng: np.random.Generator,
    total_files: int,
    *,
    preferred_file_indices: Optional[np.ndarray] = None,
    preferred_prob: float = 0.0,
) -> int:
    if total_files <= 0:
        raise ValueError("total_files must be positive when sampling monitor files")

    prob = float(max(0.0, min(1.0, preferred_prob)))
    if (
        preferred_file_indices is not None
        and int(len(preferred_file_indices)) > 0
        and prob > 0.0
        and float(rng.random()) < prob
    ):
        pick = int(rng.integers(0, int(len(preferred_file_indices))))
        return int(preferred_file_indices[pick])

    return int(rng.integers(0, int(total_files)))


def _monitor_id_to_label() -> Dict[int, str]:
    mapping = {int(idx): str(name) for idx, name in cfg.ID2LANG.items()}
    other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
    if other_idx is not None:
        mapping[int(other_idx)] = "other"
    return mapping


def _append_fragment(
    pools: Dict[int, List[Dict[str, object]]],
    *,
    label_id: int,
    text: str,
    source: str,
) -> None:
    if not text:
        return
    pools.setdefault(int(label_id), []).append(
        {
            "content": str(text),
            "source": str(source),
        }
    )


def _build_augmented_fragment_datasets(
    monitor_data: Dict[str, np.ndarray],
    *,
    active_learning_store: str = "",
    active_learning_limit: Optional[int] = None,
) -> Dict[int, hfds.Dataset]:
    pools: Dict[int, List[Dict[str, object]]] = {}
    files = monitor_data["files"]
    segments = monitor_data["segments"]
    contents = monitor_data["contents"]
    id2label = _monitor_id_to_label()

    for file_idx, row in enumerate(files):
        byte_len = int(row["byte_len"])
        if byte_len <= 0:
            continue
        byte_start = int(row["byte_start"])
        file_bytes = np.asarray(contents[byte_start : byte_start + byte_len], dtype=np.uint8)
        raw_text = file_bytes.tobytes().decode("latin-1")
        normalized_text = _normalize_monitor_text(raw_text)
        if not normalized_text:
            continue
        seg_start = int(row["seg_start"])
        seg_count = int(row["seg_count"])
        seg_slice = segments[seg_start : seg_start + seg_count]
        for seg_idx, seg in enumerate(seg_slice):
            start = int(seg["start"])
            end = int(seg["end"])
            label_id = int(seg["label"])
            if end <= start or start < 0 or end > len(normalized_text):
                continue
            if label_id == int(cfg.PAD_ID):
                continue
            if label_id not in id2label:
                continue
            fragment_text = normalized_text[start:end]
            _append_fragment(
                pools,
                label_id=label_id,
                text=fragment_text,
                source=f"monitor:{file_idx}:{seg_idx}",
            )

    store_path = str(active_learning_store or "").strip()
    if store_path:
        from active_learning.label_store import (
            DEFAULT_EXCLUDED_TRAINING_SOURCE_SPLITS,
            LabelStore,
        )

        store = LabelStore(store_path)
        for row in store.iter_rows(
            limit=active_learning_limit,
            statuses=("ok",),
            exclude_source_splits=DEFAULT_EXCLUDED_TRAINING_SOURCE_SPLITS,
        ):
            text = str(row["snippet_text"] or "")
            if not text:
                continue
            segments_json = LabelStore._segment_dicts(row)
            for seg_idx, seg in enumerate(segments_json):
                try:
                    start = int(seg.get("start", 0))
                    end = int(seg.get("end", 0))
                    label_id = _label_name_to_id(str(seg.get("label", "")))
                except Exception:
                    continue
                if label_id is None or end <= start or start < 0 or end > len(text):
                    continue
                fragment_text = text[start:end]
                _append_fragment(
                    pools,
                    label_id=int(label_id),
                    text=fragment_text,
                    source=f"al:{int(row['id'])}:{seg_idx}",
                )

    out: Dict[int, hfds.Dataset] = {}
    for label_id, rows in pools.items():
        if not rows:
            continue
        out[int(label_id)] = hfds.Dataset.from_list(rows)
    return out


def _monitor_file_segments(
    row: np.void,
    segments: np.ndarray,
    byte_len: int,
) -> List[MonitorSegment]:
    file_segments: List[MonitorSegment] = []
    seg_start = int(row["seg_start"])
    seg_count = int(row["seg_count"])
    for seg in segments[seg_start : seg_start + seg_count]:
        start = max(0, min(int(seg["start"]), int(byte_len)))
        end = max(0, min(int(seg["end"]), int(byte_len)))
        if end <= start:
            continue
        file_segments.append((start, end, int(seg["label"])))
    return file_segments


def _merge_monitor_segments(segments: List[MonitorSegment]) -> List[MonitorSegment]:
    if not segments:
        return []
    merged: List[MonitorSegment] = []
    for start, end, label in sorted(
        ((int(start), int(end), int(label)) for start, end, label in segments),
        key=lambda item: (item[0], item[1], item[2]),
    ):
        if end <= start:
            continue
        if merged:
            prev_start, prev_end, prev_label = merged[-1]
            if label == prev_label and start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end), prev_label)
                continue
        merged.append((start, end, label))
    return merged


def _crop_monitor_window(
    rng: np.random.Generator,
    window_len: int,
    file_bytes: np.ndarray,
    file_segments: List[MonitorSegment],
):
    byte_len = int(file_bytes.size)
    if byte_len <= 0:
        return None

    start = 0
    if byte_len > window_len:
        start = int(rng.integers(0, byte_len - window_len + 1))
    end = start + min(window_len, byte_len)

    x = np.full(window_len, cfg.PAD_BYTE_ID, dtype=np.int32)
    x[: end - start] = np.asarray(file_bytes[start:end], dtype=np.uint8)

    y = np.full(window_len, cfg.PAD_ID, dtype=np.uint8)
    for seg_start, seg_end, label in file_segments:
        overlap_s = max(int(seg_start), start)
        overlap_e = min(int(seg_end), start + window_len)
        if overlap_e <= overlap_s:
            continue
        y_start = overlap_s - start
        y_end = overlap_e - start
        y[y_start:y_end] = int(label)

    return x, y


def _monitor_line_bounds(file_bytes: np.ndarray) -> List[Tuple[int, int, int]]:
    bounds: List[Tuple[int, int, int]] = []
    line_start = 0
    total = int(file_bytes.size)
    while line_start < total:
        idx = line_start
        while idx < total and int(file_bytes[idx]) != _MONITOR_NEWLINE_BYTE:
            idx += 1
        content_end = idx
        full_end = idx + 1 if idx < total and int(file_bytes[idx]) == _MONITOR_NEWLINE_BYTE else idx
        bounds.append((line_start, content_end, full_end))
        line_start = full_end
    return bounds


def _choose_monitor_line_removal_span(
    rng: np.random.Generator,
    file_bytes: np.ndarray,
) -> Optional[Tuple[int, int]]:
    total = int(file_bytes.size)
    if total <= 1:
        return None
    candidates = [
        (line_start, full_end)
        for line_start, _content_end, full_end in _monitor_line_bounds(file_bytes)
        if full_end > line_start and (full_end - line_start) < total
    ]
    if not candidates:
        return None
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def _choose_monitor_space_removal_span(
    rng: np.random.Generator,
    file_bytes: np.ndarray,
) -> Optional[Tuple[int, int]]:
    total = int(file_bytes.size)
    if total <= 1:
        return None
    candidate_lines: List[List[int]] = []
    for line_start, content_end, _full_end in _monitor_line_bounds(file_bytes):
        boundaries = [int(line_start)]
        for idx in range(line_start, content_end):
            if int(file_bytes[idx]) == _MONITOR_SPACE_BYTE and idx + 1 <= content_end:
                boundaries.append(int(idx + 1))
        if boundaries[-1] != int(content_end):
            boundaries.append(int(content_end))
        deduped: List[int] = []
        for boundary in boundaries:
            if not deduped or boundary != deduped[-1]:
                deduped.append(int(boundary))
        if len(deduped) >= 3:
            candidate_lines.append(deduped)
    if not candidate_lines:
        return None

    for _ in range(12):
        boundaries = candidate_lines[int(rng.integers(0, len(candidate_lines)))]
        start_idx = int(rng.integers(0, len(boundaries) - 1))
        end_idx = int(rng.integers(start_idx + 1, len(boundaries)))
        start = int(boundaries[start_idx])
        end = int(boundaries[end_idx])
        if end <= start:
            continue
        if start == boundaries[0] and end == boundaries[-1]:
            continue
        return start, end
    return None


def _choose_monitor_removal_span(
    rng: np.random.Generator,
    file_bytes: np.ndarray,
) -> Optional[Tuple[int, int]]:
    pick_line_first = bool(rng.random() < 0.5)
    first = _choose_monitor_line_removal_span if pick_line_first else _choose_monitor_space_removal_span
    second = _choose_monitor_space_removal_span if pick_line_first else _choose_monitor_line_removal_span
    span = first(rng, file_bytes)
    if span is not None:
        return span
    return second(rng, file_bytes)


def _apply_monitor_removal_span(
    file_bytes: np.ndarray,
    file_segments: List[MonitorSegment],
    start: int,
    end: int,
) -> Tuple[np.ndarray, List[MonitorSegment]]:
    start = max(0, min(int(start), int(file_bytes.size)))
    end = max(0, min(int(end), int(file_bytes.size)))
    if end <= start:
        return np.asarray(file_bytes, dtype=np.uint8), list(file_segments)

    removed = end - start
    trimmed = np.concatenate(
        [
            np.asarray(file_bytes[:start], dtype=np.uint8),
            np.asarray(file_bytes[end:], dtype=np.uint8),
        ]
    )

    out_segments: List[MonitorSegment] = []
    for seg_start, seg_end, label in file_segments:
        seg_start = int(seg_start)
        seg_end = int(seg_end)
        label = int(label)
        if seg_end <= seg_start:
            continue
        if seg_end <= start:
            out_segments.append((seg_start, seg_end, label))
            continue
        if seg_start >= end:
            out_segments.append((seg_start - removed, seg_end - removed, label))
            continue
        if seg_start < start and seg_end > end:
            out_segments.append((seg_start, seg_end - removed, label))
            continue
        if seg_start < start < seg_end <= end:
            out_segments.append((seg_start, start, label))
            continue
        if start <= seg_start < end < seg_end:
            out_segments.append((start, seg_end - removed, label))
            continue

    return trimmed, _merge_monitor_segments(out_segments)


def _apply_monitor_substring_removal(
    rng: np.random.Generator,
    file_bytes: np.ndarray,
    file_segments: List[MonitorSegment],
    *,
    apply_prob: float = _MONITOR_SUBSTRING_REMOVAL_PROB,
    max_removals: int = _MONITOR_SUBSTRING_REMOVAL_MAX_REMOVALS,
) -> Tuple[np.ndarray, List[MonitorSegment]]:
    base_bytes = np.asarray(file_bytes, dtype=np.uint8)
    base_segments = list(file_segments)
    if base_bytes.size <= 1 or not base_segments:
        return base_bytes, base_segments
    if float(apply_prob) <= 0.0 or bool(rng.random() >= float(apply_prob)):
        return base_bytes, base_segments

    current_bytes = base_bytes
    current_segments = list(base_segments)
    target = max(1, int(rng.integers(1, max(1, int(max_removals)) + 1)))
    attempts = 0
    applied = 0
    max_attempts = max(3, target * 4)
    while applied < target and attempts < max_attempts:
        attempts += 1
        span = _choose_monitor_removal_span(rng, current_bytes)
        if span is None:
            break
        next_bytes, next_segments = _apply_monitor_removal_span(
            current_bytes,
            current_segments,
            span[0],
            span[1],
        )
        if next_bytes.size <= 0 or not next_segments:
            continue
        current_bytes = next_bytes
        current_segments = next_segments
        applied += 1
    return current_bytes, current_segments


class EpochPrefetchBatcher:
    def __init__(self, dsets_by_lang: Dict[int, hfds.Dataset], data_cfg: "DataConfig"):
        import multiprocessing as mp
        import sys
        from pathlib import Path
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from train.utils.data import prepare_dsets_by_lang_with_splits
        
        self.cfg = data_cfg
        
        # We cannot pass dsets_by_lang to the worker because it contains hfds.Dataset objects which cannot
        # be pickled. Instead, we determine token targets in the main thread and pass arguments to the
        # worker to rebuild the dataset.
        
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=max(2, data_cfg.prefetch_batches))
        self.stop_flag = ctx.Event()
        self.buckets = data_cfg.buckets()
        self.threads: List[mp.Process] = []

        # Track per-language token usage to estimate fractional epochs
        self._lang_token_counts = {lang_id: 0 for lang_id in dsets_by_lang.keys()}
        self._token_targets = {}
        max_len = max(1, data_cfg.window_max_bytes)
        for lang_id, dataset in dsets_by_lang.items():
            try:
                length = len(dataset)
            except Exception:
                length = None
            if length is None or length <= 0:
                length = 1
            self._token_targets[lang_id] = max_len * length

        for wid in range(max(1, data_cfg.num_workers)):
            t = ctx.Process(target=self._worker_entry, args=(self.cfg, wid, self.q, self.stop_flag, self.buckets), daemon=True)
            t.start()
            self.threads.append(t)

        # Wait briefly then verify at least some workers survived startup
        time.sleep(5)
        self._check_workers_alive("during startup")

    def _check_workers_alive(self, context: str = ""):
        """Raise RuntimeError if all worker processes have died."""
        alive = [t for t in self.threads if t.is_alive()]
        if not alive:
            dead_codes = [
                f"worker-{i} exit={t.exitcode}" for i, t in enumerate(self.threads)
            ]
            raise RuntimeError(
                f"All EpochPrefetchBatcher workers are dead {context}! "
                f"Statuses: {', '.join(dead_codes)}. "
                f"Check stderr for worker tracebacks."
            )

    def _update_token_counts(self, labels: np.ndarray):
        valid = labels[labels != cfg.PAD_ID]
        if valid.size == 0:
            return
        unique, counts = np.unique(valid, return_counts=True)
        for lid, cnt in zip(unique.astype(int), counts.astype(int)):
            if lid in self._lang_token_counts:
                self._lang_token_counts[lid] += cnt

    @staticmethod
    def _worker_entry(cfg_obj, wid, q, stop_flag, buckets):
        import sys
        from pathlib import Path
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        
        from train.utils.data import prepare_dsets_by_lang_with_splits
        import utils.config as cfg
        
        # Reconstruct the dataset entirely within the spawned process
        dsets = prepare_dsets_by_lang_with_splits(
            cfg_obj.data_root,
            use_train_windows=True, # We assume True here for standard training workflow
            include_languages=None, # It's harder to propagate specific langs, but the main logic relies on standard dirs
            verbose=False,
        )
        dsets_by_lang = dsets["train"]

        random.seed(cfg_obj.seed ^ wid ^ int(time.time()))
        hold = max(1, cfg_obj.bucket_hold_steps)
        L = random.choice(buckets)
        k = 0

        while not stop_flag.is_set():
            if k % hold == 0:
                L = random.choice(buckets)
            k += 1

            xb = np.full((cfg_obj.batch_size, L), cfg.PAD_BYTE_ID, dtype=np.int32)
            yb = np.full((cfg_obj.batch_size, L), cfg.PAD_ID, dtype=np.uint8)

            for i in range(cfg_obj.batch_size):
                x, y = make_training_window(dsets_by_lang, L, cfg_obj)
                xb[i], yb[i] = x, y

            try:
                q.put((xb, yb), timeout=1.0)
            except queue.Full:
                pass

    def get_epochs(self) -> Dict[int, float]:
        """Approximate fractional epochs per language based on token usage."""
        epochs = {}
        for lang_id, tokens in self._lang_token_counts.items():
            target = max(1, self._token_targets.get(lang_id, 1))
            epochs[lang_id] = tokens / target
        return epochs

    def get(self, timeout: float = 30.0):
        """Get a batch, with timeout and dead-worker detection."""
        while True:
            try:
                xb, yb = self.q.get(timeout=timeout)
                self._update_token_counts(yb)
                return xb, yb
            except queue.Empty:
                self._check_workers_alive("while waiting for batch")

    def close(self):
        self.stop_flag.set()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        for t in self.threads:
            t.join(timeout=2.0)
            if t.is_alive():
                t.terminate()
                t.join(timeout=2.0)
            if t.is_alive() and hasattr(t, "kill"):
                t.kill()
                t.join(timeout=2.0)
        try:
            self.q.close()
        except Exception:
            pass
        try:
            self.q.cancel_join_thread()
        except Exception:
            pass


class MonitorFineTuneBatcher:
    """
    Simple prefetching batcher for fine-tuning on the preprocessed monitor
    memmap (monitor_preprocessed_a / _b). By default it samples random files
    and random windows without augmentation. When enabled, it can apply the
    training window augmentations on top of monitor/al fragment pools.
    """

    def __init__(
        self,
        monitor_data: Dict[str, np.ndarray],
        data_cfg: "DataConfig",
        monitor_root: str,
        *,
        augment: bool = False,
        active_learning_store: str = "",
        active_learning_limit: Optional[int] = None,
        dense_bias_label: str = "",
        dense_bias_prob: float = 0.0,
        full_files: bool = False,
        full_file_max_bytes: int = 10000,
    ):
        import multiprocessing as mp
        import sys
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        if not monitor_root:
            raise ValueError(
                "MonitorFineTuneBatcher requires a non-empty monitor_root path."
            )

        self.cfg = data_cfg
        self.augment = bool(augment)
        self.full_files = bool(full_files)
        self.full_file_max_bytes = max(1, int(full_file_max_bytes))
        self.dense_bias_label = str(dense_bias_label or "").strip().lower()
        self.dense_bias_prob = float(max(0.0, min(1.0, dense_bias_prob)))
        self.dense_bias_type_id: Optional[int] = None

        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=max(2, data_cfg.prefetch_batches))
        self.stop_flag = ctx.Event()
        self.threads: List[mp.Process] = []

        self.total_files = int(len(monitor_data["files"]))
        self._processed_windows = 0

        if self.dense_bias_label and self.dense_bias_prob > 0.0:
            meta = monitor_data.get("meta") or {}
            meta_lang2id = {
                str(k).strip().lower(): int(v)
                for k, v in (meta.get("lang2id") or {}).items()
            }
            type_id = meta_lang2id.get(self.dense_bias_label)
            if type_id is None:
                type_id = _label_name_to_id(self.dense_bias_label)
            if type_id is None:
                print(
                    "Monitor fine-tune dense bias disabled: "
                    f"unknown label '{self.dense_bias_label}'.",
                    flush=True,
                )
            else:
                match_count = int(
                    np.count_nonzero(monitor_data["files"]["type_id"] == int(type_id))
                )
                if match_count <= 0:
                    print(
                        "Monitor fine-tune dense bias disabled: "
                        f"label '{self.dense_bias_label}' has no files in {monitor_root}.",
                        flush=True,
                    )
                else:
                    self.dense_bias_type_id = int(type_id)
                    print(
                        "Monitor fine-tune dense bias enabled: "
                        f"label={self.dense_bias_label} "
                        f"type_id={self.dense_bias_type_id} "
                        f"prob={self.dense_bias_prob:.3f} "
                        f"files={match_count}/{self.total_files}",
                        flush=True,
                    )

        for wid in range(max(1, data_cfg.num_workers)):
            t = ctx.Process(
                target=self._worker_entry,
                args=(
                    self.cfg,
                    wid,
                    monitor_root,
                    self.q,
                    self.stop_flag,
                    self.total_files,
                    bool(self.augment),
                    str(active_learning_store or ""),
                    None if active_learning_limit is None else int(active_learning_limit),
                    self.dense_bias_type_id,
                    self.dense_bias_prob,
                    self.full_files,
                    self.full_file_max_bytes,
                ),
                daemon=True,
            )
            t.start()
            self.threads.append(t)

        # Wait briefly then verify at least some workers survived startup
        time.sleep(5)
        self._check_workers_alive("during startup")

    def _check_workers_alive(self, context: str = ""):
        """Raise RuntimeError if all worker processes have died."""
        alive = [t for t in self.threads if t.is_alive()]
        if not alive:
            dead_codes = [
                f"worker-{i} exit={t.exitcode}" for i, t in enumerate(self.threads)
            ]
            raise RuntimeError(
                f"All MonitorFineTuneBatcher workers are dead {context}! "
                f"Statuses: {', '.join(dead_codes)}. "
                f"Check stderr for worker tracebacks."
            )

    @staticmethod
    def _build_window(
        rng: np.random.Generator,
        window_len: int,
        files,
        contents,
        segments,
        total_files,
        *,
        allow_substring_removal: bool = False,
        preferred_file_indices: Optional[np.ndarray] = None,
        preferred_prob: float = 0.0,
    ):
        max_attempts = 32
        for _ in range(max_attempts):
            if total_files <= 0:
                return None
            file_idx = _sample_monitor_file_index(
                rng,
                total_files,
                preferred_file_indices=preferred_file_indices,
                preferred_prob=preferred_prob,
            )
            row = files[file_idx]
            byte_len = int(row["byte_len"])
            if byte_len <= 0:
                continue

            file_bytes = np.asarray(
                contents[
                    int(row["byte_start"]) : int(row["byte_start"]) + byte_len
                ],
                dtype=np.uint8,
            )
            file_segments = _monitor_file_segments(row, segments, byte_len)
            if allow_substring_removal:
                file_bytes, file_segments = _apply_monitor_substring_removal(
                    rng,
                    file_bytes,
                    file_segments,
                )
            window = _crop_monitor_window(
                rng,
                window_len,
                file_bytes,
                file_segments,
            )
            if window is None:
                continue
            return window
        return None

    @staticmethod
    def _build_augmented_window(
        rng: np.random.Generator,
        window_len: int,
        files,
        contents,
        segments,
        total_files: int,
        fragment_dsets: Dict[int, hfds.Dataset],
        data_cfg: "DataConfig",
        *,
        forced_mode: Optional[str] = None,
        preferred_file_indices: Optional[np.ndarray] = None,
        preferred_prob: float = 0.0,
    ):
        mode = str(forced_mode or _choose_window_mode(data_cfg)).strip().lower()
        if mode == "pure":
            return MonitorFineTuneBatcher._build_window(
                rng,
                window_len,
                files,
                contents,
                segments,
                total_files,
                allow_substring_removal=True,
                preferred_file_indices=preferred_file_indices,
                preferred_prob=preferred_prob,
            )
        if not fragment_dsets:
            return MonitorFineTuneBatcher._build_window(
                rng,
                window_len,
                files,
                contents,
                segments,
                total_files,
                allow_substring_removal=True,
                preferred_file_indices=preferred_file_indices,
                preferred_prob=preferred_prob,
            )

        seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            if mode == "line_inject":
                return make_line_injected_window(fragment_dsets, window_len, data_cfg)
            if mode == "markdown":
                x, y, _ = _make_markdown_window_impl(
                    fragment_dsets,
                    window_len,
                    data_cfg,
                    collect_meta=False,
                )
                return x, y
            if mode == "mixed":
                return make_mixed_window(
                    fragment_dsets,
                    window_len,
                    int(data_cfg.min_seg_len),
                )
            return make_training_window(fragment_dsets, window_len, data_cfg)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    @staticmethod
    def _build_full_sequence(
        rng: np.random.Generator,
        target_len: int,
        files,
        contents,
        segments,
        total_files,
        *,
        preferred_file_indices: Optional[np.ndarray] = None,
        preferred_prob: float = 0.0,
    ):
        if total_files <= 0:
            return None
        file_idx = _sample_monitor_file_index(
            rng,
            total_files,
            preferred_file_indices=preferred_file_indices,
            preferred_prob=preferred_prob,
        )
        x, y, _ = build_monitor_file_sequence(
            files,
            contents,
            segments,
            int(file_idx),
            target_len=int(target_len),
            pad_byte_id=int(cfg.PAD_BYTE_ID),
            pad_label_id=int(cfg.PAD_ID),
            rng=rng,
            random_crop=True,
        )
        return x, y

    @staticmethod
    def _build_augmented_full_sequence(
        rng: np.random.Generator,
        target_len: int,
        files,
        contents,
        segments,
        total_files: int,
        fragment_dsets: Dict[int, hfds.Dataset],
        data_cfg: "DataConfig",
        *,
        preferred_file_indices: Optional[np.ndarray] = None,
        preferred_prob: float = 0.0,
    ):
        pieces: List[tuple[np.ndarray, np.ndarray]] = []
        built = 0
        component_len = max(1, min(int(cfg.MODEL_WINDOW_BYTES), int(target_len)))
        attempts = 0
        max_attempts = max(4, (int(target_len) // max(1, component_len)) * 8)
        while built < int(target_len) and attempts < max_attempts:
            attempts += 1
            forced_mode = str(_choose_window_mode(data_cfg)).strip().lower()
            if forced_mode == "pure":
                piece = MonitorFineTuneBatcher._build_full_sequence(
                    rng,
                    component_len,
                    files,
                    contents,
                    segments,
                    total_files,
                    preferred_file_indices=preferred_file_indices,
                    preferred_prob=preferred_prob,
                )
            else:
                piece = MonitorFineTuneBatcher._build_augmented_window(
                    rng,
                    component_len,
                    files,
                    contents,
                    segments,
                    total_files,
                    fragment_dsets,
                    data_cfg,
                    forced_mode=forced_mode,
                    preferred_file_indices=preferred_file_indices,
                    preferred_prob=preferred_prob,
                )
            if piece is None:
                continue
            pieces.append(piece)
            built += int(np.count_nonzero((np.asarray(piece[0]) >= 0) & (np.asarray(piece[0]) < cfg.BYTE_VOCAB_SIZE)))
            if len(pieces) > 1:
                built += 2
        if not pieces:
            return None
        x, y, _ = pack_sequence_pieces(
            pieces,
            target_len=int(target_len),
            pad_byte_id=int(cfg.PAD_BYTE_ID),
            pad_label_id=int(cfg.PAD_ID),
            separator_label_id=int(cfg.PAD_ID),
        )
        return x, y

    @staticmethod
    def _worker_entry(
        cfg_obj,
        wid,
        monitor_root,
        q,
        stop_flag,
        total_files,
        augment,
        active_learning_store,
        active_learning_limit,
        dense_bias_type_id,
        dense_bias_prob,
        full_files,
        full_file_max_bytes,
    ):
        import sys
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
            
        from train.utils.monitor_eval import load_monitor_memmaps
        import os
        
        # Load the memmap exclusively for this child process
        if not monitor_root or not os.path.exists(monitor_root):
            raise RuntimeError(
                f"MonitorFineTuneBatcher worker {wid}: monitor_root is missing or "
                f"does not exist: '{monitor_root}'"
            )
        monitor_data = load_monitor_memmaps(Path(monitor_root))

        files = monitor_data["files"]
        segments = monitor_data["segments"]
        contents = monitor_data["contents"]
        fragment_dsets: Dict[int, hfds.Dataset] = {}
        if augment:
            fragment_dsets = _build_augmented_fragment_datasets(
                monitor_data,
                active_learning_store=active_learning_store,
                active_learning_limit=active_learning_limit,
            )

        preferred_file_indices: Optional[np.ndarray] = None
        bias_prob = float(max(0.0, min(1.0, dense_bias_prob)))
        if dense_bias_type_id is not None and bias_prob > 0.0:
            matches = np.flatnonzero(files["type_id"] == int(dense_bias_type_id))
            if int(matches.size) > 0:
                preferred_file_indices = matches.astype(np.int64, copy=False)

        rng = np.random.default_rng(cfg_obj.seed ^ wid ^ int(time.time()))
        full_files = bool(full_files)
        target_sequence_len = max(1, int(full_file_max_bytes)) if full_files else int(cfg_obj.window_max_bytes)
        buckets = cfg_obj.buckets() if augment and not full_files else [target_sequence_len]
        buckets = [int(b) for b in buckets if int(b) > 0]
        if not buckets:
            buckets = [target_sequence_len]
        hold = max(1, int(getattr(cfg_obj, "bucket_hold_steps", 1))) if augment else 1
        window_len = int(buckets[0])
        step_idx = 0

        while not stop_flag.is_set():
            if step_idx % hold == 0:
                bucket_idx = int(rng.integers(0, len(buckets))) if len(buckets) > 1 else 0
                window_len = int(buckets[bucket_idx])
            step_idx += 1
            xb = np.full(
                (cfg_obj.batch_size, window_len),
                cfg.PAD_BYTE_ID,
                dtype=np.int32,
            )
            yb = np.full(
                (cfg_obj.batch_size, window_len),
                cfg.PAD_ID,
                dtype=np.uint8,
            )

            filled = 0
            while filled < cfg_obj.batch_size and not stop_flag.is_set():
                if full_files and augment:
                    window = MonitorFineTuneBatcher._build_augmented_full_sequence(
                        rng,
                        window_len,
                        files,
                        contents,
                        segments,
                        total_files,
                        fragment_dsets,
                        cfg_obj,
                        preferred_file_indices=preferred_file_indices,
                        preferred_prob=bias_prob,
                    )
                elif full_files:
                    window = MonitorFineTuneBatcher._build_full_sequence(
                        rng,
                        window_len,
                        files,
                        contents,
                        segments,
                        total_files,
                        preferred_file_indices=preferred_file_indices,
                        preferred_prob=bias_prob,
                    )
                elif augment:
                    window = MonitorFineTuneBatcher._build_augmented_window(
                        rng,
                        window_len,
                        files,
                        contents,
                        segments,
                        total_files,
                        fragment_dsets,
                        cfg_obj,
                        preferred_file_indices=preferred_file_indices,
                        preferred_prob=bias_prob,
                    )
                else:
                    window = MonitorFineTuneBatcher._build_window(
                        rng,
                        window_len,
                        files,
                        contents,
                        segments,
                        total_files,
                        preferred_file_indices=preferred_file_indices,
                        preferred_prob=bias_prob,
                    )
                if window is None:
                    # If we repeatedly fail to build a window, just break and
                    # reuse whatever portion we have so far.
                    break
                x, y = window
                xb[filled] = x
                yb[filled] = y
                filled += 1

            if filled == 0:
                continue

            try:
                q.put((xb, yb, filled), timeout=1.0)
            except queue.Full:
                pass

    def get_epochs(self) -> Dict[int, float]:
        """Approximate global epochs over the monitor memmap."""
        if self.total_files <= 0:
            return {}
        epochs = float(self._processed_windows) / float(self.total_files)
        # Use a single synthetic key for logging.
        return {0: epochs}

    def get(self, timeout: float = 30.0):
        """Get a batch, with timeout and dead-worker detection."""
        while True:
            try:
                xb, yb, filled = self.q.get(timeout=timeout)
                self._processed_windows += filled
                return xb, yb
            except queue.Empty:
                # Queue was empty for `timeout` seconds — check if workers died
                self._check_workers_alive("while waiting for batch")

    def close(self):
        self.stop_flag.set()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        for t in self.threads:
            t.join(timeout=2.0)
            if t.is_alive():
                t.terminate()
                t.join(timeout=2.0)
            if t.is_alive() and hasattr(t, "kill"):
                t.kill()
                t.join(timeout=2.0)
        try:
            self.q.close()
        except Exception:
            pass
        try:
            self.q.cancel_join_thread()
        except Exception:
            pass
