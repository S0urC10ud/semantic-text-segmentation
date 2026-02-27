"""
Utilities for loading the preprocessed monitor set (memmap) and running fast evaluations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import utils.config as cfg
from utils.metrics_helper import (
    _forward_logits,
    _valid_metric_mask,
    accumulate_confusion,
)
from utils.token_utils import sanitize_tokens

# Must mirror downloader/999_prepare_monitor_set.py
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


def load_monitor_memmaps(root: Path) -> Dict[str, Any]:
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

    # Allow a backward-compatible case where the on-disk mapping includes
    # an extra 'other' label, which is now treated as a derived bucket
    # driven by a confidence threshold rather than an explicit model logit.
    disk_map = {k: int(v) for k, v in (meta.get("lang2id") or {}).items()}
    code_map = dict(cfg.LANG2ID)

    if disk_map:
        optional = set(getattr(cfg, "OPTIONAL_LABELS", ()))

        def _strip_optional(m: Dict[str, int]) -> Dict[str, int]:
            return {k: int(v) for k, v in m.items() if k not in optional}

        disk_core = _strip_optional(disk_map)
        code_core = _strip_optional(code_map)

        if disk_core != code_core:
            raise ValueError(
                "Monitor lang2id mismatch.\n"
                f"On disk: {disk_map}\nIn code: {code_map}\n"
                "Rebuild the monitor set or update config to stay in sync."
            )

        # If an 'other' label exists on disk, ensure its id matches the
        # configured OTHER_CLASS_INDEX so that thresholded metrics can
        # safely treat it as a derived bucket.
        other_label = None
        for lbl in optional:
            if lbl in disk_map:
                other_label = lbl
                break
        if other_label is not None:
            other_idx_disk = int(disk_map[other_label])
            other_idx_cfg = getattr(cfg, "OTHER_CLASS_INDEX", None)
            if other_idx_cfg is not None and other_idx_disk != int(other_idx_cfg):
                raise ValueError(
                    "Monitor 'other' index mismatch.\n"
                    f"On disk: {other_label} -> {other_idx_disk}\n"
                    f"In code: OTHER_CLASS_INDEX -> {other_idx_cfg}\n"
                    "Rebuild the monitor set or update config to stay in sync."
                )

    return {
        "meta": meta,
        "files": files,
        "segments": segments,
        "contents": contents,
    }


def _build_window(
    files: np.ndarray,
    segments: np.ndarray,
    contents: np.memmap,
    file_idx: int,
    window_len: int,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    row = files[file_idx]
    byte_len = int(row["byte_len"])
    if byte_len <= 0:
        return None

    start = 0
    if byte_len > window_len:
        start = int(rng.integers(0, byte_len - window_len + 1))
    end = start + min(window_len, byte_len)

    full_slice = contents[int(row["byte_start"]) : int(row["byte_start"] + byte_len)]
    x = np.full(window_len, cfg.PAD_BYTE_ID, dtype=np.int32)
    x[: end - start] = np.asarray(full_slice[start:end], dtype=np.uint8)

    y = np.full(window_len, cfg.PAD_ID, dtype=np.uint8)
    seg_slice = segments[int(row["seg_start"]) : int(row["seg_start"] + row["seg_count"])]
    for seg in seg_slice:
        seg_start = int(seg["start"])
        seg_end = int(seg["end"])
        label = int(seg["label"])
        overlap_s = max(seg_start, start)
        overlap_e = min(seg_end, start + window_len)
        if overlap_e <= overlap_s:
            continue
        y_start = overlap_s - start
        y_end = overlap_e - start
        y[y_start:y_end] = label

    return sanitize_tokens(x), y


def evaluate_monitor_set(
    state,
    monitor_data: Dict[str, Any],
    L: int,
    batch_size: int,
    rng,
    limit: Optional[int] = None,
    eval_step_fn=None,
    other_threshold: float = 0.0,
) -> Dict[str, Any]:
    if eval_step_fn is None:
        raise ValueError("evaluate_monitor_set requires eval_step_fn=eval_step")

    files = monitor_data["files"]
    segments = monitor_data["segments"]
    contents = monitor_data["contents"]
    num_files = len(files)

    rng, choice_rng = jax.random.split(rng)
    seed = int(jax.random.randint(choice_rng, (), 0, 2**31 - 1).item())
    np_rng = np.random.default_rng(seed)

    indices = np.arange(num_files, dtype=np.int64)
    if limit is not None and limit > 0 and limit < len(indices):
        indices = np_rng.choice(indices, size=int(limit), replace=False)

    losses: list[float] = []
    accs: list[float] = []
    # Base confusion tracks only the trained classes.
    conf_mat = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
    conf_thresh = None
    accs_thresh: list[float] = []
    # Thresholded confusion can route low-confidence predictions into a
    # derived "other" bucket that has no explicit logit.
    use_threshold = other_threshold > 0.0
    other_id = getattr(cfg, "OTHER_CLASS_INDEX", None) if use_threshold else None
    if use_threshold and other_id is not None:
        conf_thresh = np.zeros(
            (cfg.NUM_CLASSES + 1, cfg.NUM_CLASSES + 1), dtype=np.int64
        )
    skipped = 0

    xb_batch: list[np.ndarray] = []
    yb_batch: list[np.ndarray] = []

    def _flush_batch(batch_rng):
        nonlocal xb_batch, yb_batch
        if not xb_batch:
            return
        xb = np.stack(xb_batch)
        yb = np.stack(yb_batch)

        # For loss/accuracy, treat any labels that map to a derived
        # "other" bucket as PAD so they are not forced into a missing
        # explicit logit.
        labels_for_loss = yb.copy()
        other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
        if other_idx is not None:
            labels_for_loss[labels_for_loss == int(other_idx)] = cfg.PAD_ID

        batch_rng, logits_rng = jax.random.split(batch_rng)
        loss, acc = eval_step_fn(
            state,
            jnp.array(xb, dtype=jnp.int32),
            jnp.array(labels_for_loss, dtype=jnp.uint8),
            batch_rng,
        )
        losses.append(float(loss))
        accs.append(float(acc))
        logits = _forward_logits(state, jnp.array(xb, dtype=jnp.int32), logits_rng)
        logits_np = np.asarray(logits)
        preds = np.asarray(np.argmax(logits_np, axis=-1), dtype=np.int32)
        y_true = yb.astype(np.int32)

        # Base confusion: only for the explicit trained classes;
        # ignore any derived "other" targets.
        mask_all = _valid_metric_mask(y_true, xb)
        core_mask = mask_all & (y_true < cfg.NUM_CLASSES)
        if core_mask.any():
            accumulate_confusion(conf_mat, y_true[core_mask], preds[core_mask])

        # Thresholded confusion: allow the derived 'other' bucket as a
        # true label at OTHER_CLASS_INDEX and as a routed prediction
        # when max softmax is below the given threshold.
        if mask_all.any() and conf_thresh is not None and other_id is not None:
            logits_shift = logits_np - logits_np.max(axis=-1, keepdims=True)
            probs = np.exp(logits_shift)
            probs /= np.maximum(probs.sum(axis=-1, keepdims=True), 1e-9)
            max_prob = probs.max(axis=-1)
            preds_thresh = preds.copy()
            preds_thresh[max_prob < other_threshold] = other_id

            thresh_mask = mask_all & (y_true <= other_id)
            if thresh_mask.any():
                accumulate_confusion(
                    conf_thresh, y_true[thresh_mask], preds_thresh[thresh_mask]
                )
                accs_thresh.append(
                    float(
                        np.mean(
                            (preds_thresh[thresh_mask] == y_true[thresh_mask]).astype(
                                np.float32
                            )
                        )
                    )
                )
        xb_batch = []
        yb_batch = []

    for idx in indices:
        window = _build_window(files, segments, contents, int(idx), L, np_rng)
        if window is None:
            skipped += 1
            continue
        xb, yb = window
        xb_batch.append(xb)
        yb_batch.append(yb)
        if len(xb_batch) == batch_size:
            rng, batch_rng = jax.random.split(rng)
            _flush_batch(batch_rng)

    if xb_batch:
        rng, batch_rng = jax.random.split(rng)
        _flush_batch(batch_rng)

    import numpy as _np

    return {
        "loss_mean": float(_np.mean(losses)) if losses else 0.0,
        "acc_mean": float(_np.mean(accs)) if accs else 0.0,
        "conf_mat": conf_mat,
        "files_used": int(len(indices)),
        "skipped": int(skipped),
        "windows": int(len(indices) - skipped),
        "conf_thresh": conf_thresh,
        "acc_thresh_mean": float(_np.mean(accs_thresh)) if accs_thresh else None,
    }
