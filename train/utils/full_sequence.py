from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import utils.config as cfg
from utils.window_generator import make_training_window_with_metadata


FULL_SEQUENCE_SEPARATOR_BYTES = b"\n\n"


def _piece_effective_len(tokens: np.ndarray) -> int:
    arr = np.asarray(tokens, dtype=np.int32).reshape(-1)
    if arr.size == 0:
        return 0
    valid = np.flatnonzero((arr >= 0) & (arr < cfg.BYTE_VOCAB_SIZE))
    if valid.size <= 0:
        return 0
    return int(valid[-1]) + 1


def _trim_piece(tokens: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr_x = np.asarray(tokens, dtype=np.int32).reshape(-1)
    arr_y = np.asarray(labels, dtype=np.uint8).reshape(-1)
    keep = min(_piece_effective_len(arr_x), int(arr_y.shape[0]))
    if keep <= 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.uint8),
        )
    return arr_x[:keep].astype(np.int32, copy=False), arr_y[:keep].astype(np.uint8, copy=False)


def pack_sequence_pieces(
    pieces: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    target_len: int,
    pad_byte_id: int,
    pad_label_id: int,
    separator_bytes: bytes = FULL_SEQUENCE_SEPARATOR_BYTES,
    separator_label_id: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
    target = max(1, int(target_len))
    sep_label = int(pad_label_id if separator_label_id is None else separator_label_id)
    sep_arr = np.frombuffer(separator_bytes, dtype=np.uint8).astype(np.int32, copy=False)

    out_x = np.full((target,), int(pad_byte_id), dtype=np.int32)
    out_y = np.full((target,), int(pad_label_id), dtype=np.uint8)
    cursor = 0
    component_ranges: List[Dict[str, int]] = []
    written_pieces = 0

    for piece_index, (piece_tokens, piece_labels) in enumerate(pieces):
        trimmed_x, trimmed_y = _trim_piece(piece_tokens, piece_labels)
        if trimmed_x.size <= 0:
            continue

        if written_pieces > 0 and cursor < target and sep_arr.size > 0:
            sep_take = min(int(sep_arr.size), target - cursor)
            if sep_take > 0:
                out_x[cursor : cursor + sep_take] = sep_arr[:sep_take]
                out_y[cursor : cursor + sep_take] = np.full((sep_take,), sep_label, dtype=np.uint8)
                cursor += sep_take

        if cursor >= target:
            break

        take = min(int(trimmed_x.size), target - cursor)
        if take <= 0:
            break

        start = cursor
        out_x[cursor : cursor + take] = trimmed_x[:take]
        out_y[cursor : cursor + take] = trimmed_y[:take]
        cursor += take
        component_ranges.append(
            {
                "piece_index": int(piece_index),
                "start": int(start),
                "end": int(cursor),
                "source_bytes": int(trimmed_x.size),
            }
        )
        written_pieces += 1

    return out_x, out_y, component_ranges


def make_training_full_sequence_with_metadata(
    dsets_by_lang: Dict[int, dict],
    data_cfg: "cfg.DataConfig",
    *,
    target_len: int,
    component_window_bytes: int = cfg.MODEL_WINDOW_BYTES,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    pieces: List[tuple[np.ndarray, np.ndarray]] = []
    component_meta: List[Dict[str, object]] = []
    built_bytes = 0
    separator_bytes = FULL_SEQUENCE_SEPARATOR_BYTES
    sep_cost = len(separator_bytes)
    target = max(1, int(target_len))
    component_len = max(1, int(component_window_bytes))

    attempts = 0
    max_attempts = max(4, (target // max(1, component_len)) * 8)
    while built_bytes < target and attempts < max_attempts:
        attempts += 1
        x, y, meta = make_training_window_with_metadata(dsets_by_lang, component_len, data_cfg)
        trimmed_x, trimmed_y = _trim_piece(x, y)
        if trimmed_x.size <= 0:
            continue
        pieces.append((trimmed_x, trimmed_y))
        component_meta.append(dict(meta) if isinstance(meta, dict) else {})
        built_bytes += int(trimmed_x.size)
        if len(pieces) > 1:
            built_bytes += sep_cost

    x_full, y_full, ranges = pack_sequence_pieces(
        pieces,
        target_len=target,
        pad_byte_id=int(cfg.PAD_BYTE_ID),
        pad_label_id=int(cfg.PAD_ID),
        separator_bytes=separator_bytes,
        separator_label_id=int(cfg.PAD_ID),
    )
    return (
        x_full,
        y_full,
        {
            "sampling_mode": "training_full_sequence",
            "target_bytes": int(target),
            "component_window_bytes": int(component_len),
            "component_count": int(len(pieces)),
            "component_ranges": ranges,
            "components": component_meta,
        },
    )


def build_monitor_file_sequence(
    files: np.ndarray,
    contents: np.ndarray,
    segments: np.ndarray,
    file_idx: int,
    *,
    target_len: int,
    pad_byte_id: int,
    pad_label_id: int,
    rng: Optional[np.random.Generator] = None,
    random_crop: bool = False,
) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    row = files[int(file_idx)]
    byte_len = int(row["byte_len"])
    target = max(1, int(target_len))
    out_x = np.full((target,), int(pad_byte_id), dtype=np.int32)
    out_y = np.full((target,), int(pad_label_id), dtype=np.uint8)
    if byte_len <= 0:
        return out_x, out_y, {"file_idx": int(file_idx), "byte_len": 0, "crop_start": 0}

    crop_start = 0
    if byte_len > target and random_crop and rng is not None:
        crop_start = int(rng.integers(0, byte_len - target + 1))
    crop_end = min(byte_len, crop_start + target)
    take = max(0, crop_end - crop_start)

    file_bytes = np.asarray(
        contents[int(row["byte_start"]) : int(row["byte_start"]) + byte_len],
        dtype=np.uint8,
    )
    if take > 0:
        out_x[:take] = file_bytes[crop_start:crop_end].astype(np.int32, copy=False)

    seg_slice = segments[int(row["seg_start"]) : int(row["seg_start"]) + int(row["seg_count"])]
    for seg in seg_slice:
        seg_start = int(seg["start"])
        seg_end = int(seg["end"])
        if seg_end <= seg_start:
            continue
        overlap_start = max(seg_start, crop_start)
        overlap_end = min(seg_end, crop_start + target)
        if overlap_end <= overlap_start:
            continue
        y_start = overlap_start - crop_start
        y_end = overlap_end - crop_start
        out_y[y_start:y_end] = int(seg["label"])

    return out_x, out_y, {
        "file_idx": int(file_idx),
        "byte_len": int(byte_len),
        "crop_start": int(crop_start),
    }
