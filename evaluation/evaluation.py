#!/usr/bin/env python3
"""Comprehensive evaluation harness for the segmentation model.

The script expects evaluation datasets produced by ``obtain_eval_dataset.py``
and runs a battery of benchmarks:

  • Task accuracy on curated validation-derived datasets (pure fragments,
    injections, mixed sequences, markdown blends).
  • Throughput measurements on fixed-length windows for both CPU and GPU
    (when available), including coarse memory usage observations.

Results are aggregated into a Markdown report containing:

  - Summary metrics per task (overall and macro accuracy).
  - Per-label accuracy tables for each task (sorted by recall).
  - Throughput/latency tables for the stress benchmarks.

Example:

```
python evaluation.py \
  --checkpoint checkpoints/seg-unet1d.msgpack \
  --data-root evaluation/data \
  --report-path evaluation/report.md
```
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re

import numpy as np

# Optional dependencies -----------------------------------------------------
try:  # pragma: no cover - psutil is optional
    import psutil
except Exception:  # pragma: no cover
    psutil = None

DEFAULT_CHANNELS: Tuple[int, ...] = (96, 128, 192, 256)
PREDICTION_LABEL_ALIASES: Dict[str, str] = {
    "c": "c_family",
    "cpp": "c_family",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))


import datasets as hfds  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import flax.serialization as serialization  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402

from train import config as cfg  # noqa: E402
from train.model import UNet1D  # noqa: E402
from train.token_utils import sanitize_bytes, sanitize_tokens  # noqa: E402
from train.metrics_helper import compute_metrics_from_confusion  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint loading helpers (adapted from segment_viewer.app without CLI)
# ---------------------------------------------------------------------------

def _looks_like_orbax_step_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    names = {x.name for x in p.iterdir()}
    return (
        "manifest.ocdbt" in names
        or "_CHECKPOINT_METADATA" in names
        or "ocdbt.process_0" in names
        or "_METADATA" in names
    )


def _find_latest_orbax_step_dir(root: Path) -> Optional[Path]:
    root = Path(root)
    if _looks_like_orbax_step_dir(root):
        return root
    if not root.exists() or not root.is_dir():
        return None
    step_dirs = []
    for d in root.iterdir():
        if d.is_dir() and _looks_like_orbax_step_dir(d):
            name = d.name
            step = -1
            if "-" in name:
                try:
                    step = int(name.rsplit("-", 1)[-1])
                except ValueError:
                    step = -1
            step_dirs.append((step, name, d))
    if not step_dirs:
        return None
    step_dirs.sort(key=lambda t: (t[0], '.msgpack' in t[1]))
    return step_dirs[-1][2]


def _extract_params_tree(obj):
    if hasattr(obj, "params"):
        try:
            return getattr(obj, "params")
        except Exception:
            pass
    try:
        from flax.core.frozen_dict import FrozenDict

        if isinstance(obj, FrozenDict):
            return _extract_params_tree(obj.unfreeze())
    except Exception:
        pass
    from collections.abc import Mapping

    if isinstance(obj, Mapping):
        if "params" in obj:
            return obj["params"]
        for key in ("target", "state", "train_state", "flax_state"):
            if key in obj:
                try:
                    return _extract_params_tree(obj[key])
                except Exception:
                    continue
        for v in obj.values():
            try:
                return _extract_params_tree(v)
            except Exception:
                continue
    if isinstance(obj, (list, tuple)):
        for v in obj:
            try:
                return _extract_params_tree(v)
            except Exception:
                continue
    if hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            try:
                return _extract_params_tree(v)
            except Exception:
                continue
    raise KeyError("Could not find 'params' subtree in restored checkpoint object.")


def _load_params_from_any(ckpt_path: str, params_template):
    p = Path(ckpt_path).resolve()

    if p.is_file():
        with open(p, "rb") as f:
            raw = f.read()
        try:
            return serialization.from_bytes(params_template, raw)
        except Exception:
            data = serialization.from_bytes({"params": params_template}, raw)
            return data["params"]

    step_dir = _find_latest_orbax_step_dir(p)
    if step_dir is None:
        raise FileNotFoundError(f"Checkpoint path not found/unsupported: {ckpt_path}")

    step_dir_str = step_dir.resolve().as_posix()
    checkpointer = ocp.StandardCheckpointer()
    try:
        restored = checkpointer.restore(step_dir_str)
        return _extract_params_tree(restored)
    except Exception:
        pass

    try:
        template = {"params": params_template}
        restored = checkpointer.restore(step_dir_str, target=template, strict=False)
        return _extract_params_tree(restored)
    except Exception:
        pass

    try:
        from flax.training import train_state as ts
        import optax

        tx = optax.identity()
        dummy = ts.TrainState.create(apply_fn=lambda *a, **k: None, params=params_template, tx=tx)
        restored = checkpointer.restore(step_dir_str, target=dummy, strict=False)
        return restored.params
    except Exception as exc:
        raise RuntimeError(
            f"Failed to restore checkpoint at '{step_dir_str}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------


def _resolve_backend(device_name: Optional[str]) -> Optional[str]:
    if not device_name or device_name.lower() in {"auto", "default"}:
        return None
    lower = device_name.lower()
    if lower in {"cpu", "gpu", "tpu"}:
        return lower
    if lower == "cuda":
        return "gpu"
    raise ValueError(f"Unknown device name '{device_name}'. Use cpu/gpu/cuda/auto.")


def _window_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((length,), dtype=np.float32)
    positions = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
    sigma = 0.5
    weights = np.exp(-0.5 * (positions / sigma) ** 2)
    return weights.astype(np.float32)


def _byte_labels_to_char_labels(
    text: str,
    byte_labels: np.ndarray,
    byte_probs: Optional[np.ndarray],
    num_classes: int,
) -> Tuple[List[int], List[np.ndarray]]:
    labels: List[int] = []
    probs: List[np.ndarray] = []
    bpos = 0
    for ch in text:
        chunk = ch.encode("utf-8", "ignore")
        length = len(chunk)
        if length == 0:
            labels.append(0)
            probs.append(np.zeros((num_classes,), dtype=np.float32))
            continue
        seg = byte_labels[bpos:bpos + length]
        if len(seg) == 0:
            labels.append(0)
            probs.append(np.zeros((num_classes,), dtype=np.float32))
        else:
            values, counts = np.unique(seg, return_counts=True)
            lbl = int(values[np.argmax(counts)])
            labels.append(lbl)
            if byte_probs is not None and len(byte_probs) >= bpos + length:
                avg = np.mean(byte_probs[bpos:bpos + length], axis=0)
                probs.append(avg.astype(np.float32))
            else:
                p = np.zeros((num_classes,), dtype=np.float32)
                p[lbl] = 1.0
                probs.append(p)
        bpos += length
    return labels, probs


def _smooth_min_run(labels: List[int], min_run: int) -> List[int]:
    if min_run <= 1 or len(labels) == 0:
        return labels
    runs = []
    current = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            runs.append((start, i, current))
            start = i
            current = labels[i]
    runs.append((start, len(labels), current))
    if len(runs) <= 2:
        return labels
    arr = labels[:]
    for idx, (s, e, lbl) in enumerate(runs):
        length = e - s
        if length >= min_run:
            continue
        left_lbl = runs[idx - 1][2] if idx - 1 >= 0 else lbl
        right_lbl = runs[idx + 1][2] if idx + 1 < len(runs) else lbl
        left_len = runs[idx - 1][1] - runs[idx - 1][0] if idx - 1 >= 0 else 0
        right_len = runs[idx + 1][1] - runs[idx + 1][0] if idx + 1 < len(runs) else 0
        new_lbl = left_lbl if left_len >= right_len else right_lbl
        for i in range(s, e):
            arr[i] = new_lbl
    return arr


class SegmenterRunner:
    def __init__(
        self,
        checkpoint_path: str,
        *,
        model_dim: int,
        channels: Sequence[int],
        dtype: str = "bfloat16",
        chunk: int = 1024,
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        backend = _resolve_backend(device)
        available = {dev.platform for dev in jax.devices()}
        if backend is not None and backend not in available:
            raise RuntimeError(
                f"Requested backend '{backend}' not available. Available: {sorted(available)}"
            )
        self.backend = backend
        self.chunk = int(chunk)
        self.batch_size = int(batch_size)
        self.num_classes = cfg.NUM_CLASSES
        dt = getattr(jnp, dtype)
        self.model = UNet1D(num_classes=self.num_classes, emb_dim=model_dim, channels=tuple(channels), dtype=dt)
        self._weight_cache: Dict[int, np.ndarray] = {}

        dummy_tokens = jnp.full((1, 256), cfg.PAD_BYTE_ID, dtype=jnp.int32)
        variables = self.model.init({"params": jax.random.PRNGKey(0)}, dummy_tokens, train=False)
        template = variables["params"]
        self.params = _load_params_from_any(checkpoint_path, template)

        def apply_fn(tokens: jnp.ndarray):
            return self.model.apply({"params": self.params}, tokens, train=False)

        if backend:
            self._apply = jax.jit(apply_fn, backend=backend)
        else:
            self._apply = jax.jit(apply_fn)

    # ------------------------------------------------------------------
    # Low-level segmentation
    # ------------------------------------------------------------------

    def _window_weights_cached(self, length: int) -> np.ndarray:
        cached = self._weight_cache.get(length)
        if cached is None:
            cached = _window_weights(length)
            self._weight_cache[length] = cached
        return cached

    def _segment_bytes(self, byte_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        byte_arr = sanitize_bytes(byte_arr)
        N = int(len(byte_arr))
        if N == 0:
            return np.zeros((0,), dtype=np.uint8), np.zeros((0, self.num_classes), dtype=np.float32)

        win = max(64, int(self.chunk))
        stride = max(1, win // 2)
        windows: List[np.ndarray] = []
        spans: List[Tuple[int, int]] = []
        for start in range(0, max(1, N - win + 1), stride):
            end = min(start + win, N)
            windows.append(byte_arr[start:end])
            spans.append((start, end))
        if not windows:
            windows = [byte_arr]
            spans = [(0, N)]

        out_bytes = np.zeros((N,), dtype=np.uint8)
        probs_accum = np.zeros((N, self.num_classes), dtype=np.float32)
        weight_accum = np.zeros((N,), dtype=np.float32)

        batch_size = self.batch_size
        for i in range(0, len(windows), batch_size):
            span_slice = spans[i:i + batch_size]
            actual = len(span_slice)
            tokens = np.full((batch_size, self.chunk), cfg.PAD_BYTE_ID, dtype=np.int32)
            for j in range(actual):
                win_bytes = windows[i + j]
                length = min(len(win_bytes), self.chunk)
                tokens[j, :length] = win_bytes[:length].astype(np.int32)
            tokens = sanitize_tokens(tokens)
            logits = self._apply(jnp.array(tokens, dtype=jnp.int32))
            logits = np.array(logits)[:actual, :self.chunk]
            probs = np.array(jax.nn.softmax(logits, axis=-1))

            for j, (start, end) in enumerate(span_slice):
                plen = end - start
                if plen <= 0:
                    continue
                plen = min(plen, self.chunk)
                weights = self._window_weights_cached(plen)
                probs_slice = probs[j, :plen]
                probs_accum[start:end] += probs_slice * weights[:, None]
                weight_accum[start:end] += weights

        nonzero = weight_accum > 0
        if np.any(nonzero):
            probs_accum[nonzero] /= weight_accum[nonzero][:, None]
        zero_mask = ~nonzero
        if np.any(zero_mask):
            probs_accum[zero_mask] = 1.0 / self.num_classes

        out_bytes[:] = np.argmax(probs_accum, axis=-1).astype(np.uint8)
        return out_bytes, probs_accum

    def segment_text(self, text: str, *, min_run_chars: int = 1) -> Tuple[List[Tuple[int, int, int]], List[int], List[np.ndarray]]:
        byte_arr = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
        byte_labels, byte_probs = self._segment_bytes(byte_arr)
        char_labels, char_probs = _byte_labels_to_char_labels(text, byte_labels, byte_probs, self.num_classes)
        char_labels = _smooth_min_run(char_labels, min_run_chars)
        segments: List[Tuple[int, int, int]] = []
        if char_labels:
            cur = char_labels[0]
            start = 0
            for idx in range(1, len(char_labels)):
                if char_labels[idx] != cur:
                    segments.append((start, idx, cur))
                    start = idx
                    cur = char_labels[idx]
            segments.append((start, len(char_labels), cur))
        return segments, char_labels, char_probs


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_datasets(data_root: Path, tasks: Optional[Sequence[str]]) -> Dict[str, hfds.Dataset]:
    datasets = {}
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if tasks and name not in tasks:
            continue
        try:
            ds = hfds.load_from_disk(str(entry))
        except Exception as exc:
            print(f"⚠️  Failed to load dataset '{name}': {exc}", flush=True)
            continue
        if len(ds) == 0:
            continue
        datasets[name] = ds
    return datasets


def _derive_task_seed(base_seed: Optional[int], task_name: str) -> int:
    base = int(base_seed or 0) & 0xFFFFFFFF
    name_bytes = task_name.encode("utf-8", "ignore")
    mix = int(np.frombuffer(name_bytes, dtype=np.uint8).sum()) & 0xFFFFFFFF
    seq = np.random.SeedSequence([base, mix])
    return int(seq.generate_state(1)[0])


def _prepare_dataset(
    task_name: str,
    dataset: hfds.Dataset,
    *,
    max_samples: int,
    base_seed: Optional[int],
    preserve_all: bool,
) -> Tuple[hfds.Dataset, int, int, bool]:
    original_len = len(dataset)
    if preserve_all or max_samples <= 0 or original_len <= max_samples:
        return dataset, original_len, original_len, False
    derived_seed = _derive_task_seed(base_seed, task_name)
    rng = np.random.default_rng(derived_seed)
    indices = rng.choice(original_len, size=max_samples, replace=False)
    indices.sort()
    subset = dataset.select(indices.tolist())
    return subset, original_len, max_samples, True


def _normalize_segments(segments) -> List[Dict[str, int]]:
    """Handle either list-of-dicts or dict-of-lists segment structures."""
    if segments is None:
        return []
    if isinstance(segments, dict):
        def _ensure_list(value, size_hint: int, default=0):
            if value is None:
                return [default] * size_hint
            if isinstance(value, (str, bytes)):
                return [value] * (size_hint or 1)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return list(value)
            try:
                return list(value)
            except TypeError:
                return [value] * size_hint if size_hint else [value]

        raw_labels = segments.get("label") or segments.get("labels") or []
        labels = _ensure_list(raw_labels, 0)
        starts = _ensure_list(segments.get("char_start") or segments.get("start"), len(labels))
        ends = _ensure_list(segments.get("char_end") or segments.get("end"), len(labels))
        normalized: List[Dict[str, int]] = []
        for label, start, end in zip(labels, starts, ends):
            if label is None:
                continue
            try:
                start_i = int(start)
                end_i = int(end)
            except (TypeError, ValueError):
                continue
            normalized.append({"label": str(label), "char_start": start_i, "char_end": end_i})
        return normalized
    if isinstance(segments, (list, tuple)):
        normalized = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            label = seg.get("label")
            if label is None:
                continue
            try:
                start_i = int(seg.get("char_start", 0))
                end_i = int(seg.get("char_end", start_i))
            except (TypeError, ValueError):
                continue
            normalized.append({"label": str(label), "char_start": start_i, "char_end": end_i})
        return normalized
    raise TypeError(f"Unsupported segments type: {type(segments)}")


def _segments_to_labels(content: str, segments: Sequence[dict], label_map: Dict[str, int]) -> np.ndarray:
    length = len(content)
    labels = np.full((length,), -1, dtype=np.int32)
    for seg in segments:
        label = seg.get("label")
        start = int(seg.get("char_start", 0))
        end = int(seg.get("char_end", 0))
        start = max(0, min(length, start))
        end = max(start, min(length, end))
        idx = label_map.get(label, label_map.get("__unknown__", -1))
        if idx < 0:
            continue
        labels[start:end] = idx
    return labels


def _confusion_size(labels: Iterable[str]) -> Tuple[List[str], Dict[str, int]]:
    uniq = sorted({label for label in labels})
    mapping = {label: idx for idx, label in enumerate(uniq)}
    return uniq, mapping


@dataclass
class TaskMetrics:
    name: str
    description: str
    samples: int
    total_chars: int
    confusion: np.ndarray
    label_names: List[str]
    per_label_counts: Dict[str, int] = field(default_factory=dict)
    per_label_correct: Dict[str, int] = field(default_factory=dict)
    extras: Dict[str, float] = field(default_factory=dict)

    def overall_accuracy(self) -> float:
        total = self.confusion.sum()
        return float(np.trace(self.confusion) / total) if total else 0.0

    def per_label_accuracy(self) -> Dict[str, float]:
        rows = {}
        for idx, label in enumerate(self.label_names):
            support = float(self.confusion[idx].sum())
            if support <= 0:
                rows[label] = float("nan")
                continue
            rows[label] = float(self.confusion[idx, idx] / support)
        return rows


def _analyze_pure_fragments(metrics: TaskMetrics) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Analyze pure fragment classification results.
    
    Returns:
        Tuple containing:
        - Dict mapping label -> purity percentage
        - Dict mapping label -> (incorrect_label -> percentage) for misclassifications
    """
    purity_scores = {}
    misclassification_details = {}
    
    for idx, true_label in enumerate(metrics.label_names):
        if true_label == "__unknown__":
            continue
            
        true_total = metrics.confusion[idx].sum()
        if true_total == 0:
            continue
            
        correct = metrics.confusion[idx, idx]
        purity = (correct / true_total) * 100 if true_total > 0 else 0
        purity_scores[true_label] = purity
        
        if purity < 100:
            # Get misclassification distribution
            mistakes = {}
            for pred_idx, pred_label in enumerate(metrics.label_names):
                if pred_idx == idx or pred_label == "__unknown__":
                    continue
                count = metrics.confusion[idx, pred_idx]
                if count > 0:
                    mistakes[pred_label] = (count / true_total) * 100
            mistakes = dict(sorted(mistakes.items(), key=lambda x: x[1], reverse=True))
            misclassification_details[true_label] = mistakes
            
    return purity_scores, misclassification_details

def evaluate_task(
    name: str,
    description: str,
    dataset: hfds.Dataset,
    runner: SegmenterRunner,
    *,
    min_run_chars: int,
) -> TaskMetrics:
    total_samples = len(dataset)
    label_candidates = {"__unknown__"}
    for example in dataset:
        for seg in _normalize_segments(example["segments"]):
            label_candidates.add(seg["label"])
    for alias_target in PREDICTION_LABEL_ALIASES.values():
        label_candidates.add(alias_target)
    label_names, label_to_idx = _confusion_size(label_candidates)
    confusion = np.zeros((len(label_names), len(label_names)), dtype=np.int64)

    total_chars = 0
    per_label_counts: Dict[str, int] = {label: 0 for label in label_names}
    per_label_correct: Dict[str, int] = {label: 0 for label in label_names}

    start_time = time.perf_counter()
    log_interval = getattr(evaluate_task, "_log_interval", 0) or 0

    for idx, example in enumerate(dataset):
        content = example["content"]
        normalized_segments = _normalize_segments(example["segments"])
        truth = _segments_to_labels(content, normalized_segments, label_to_idx)
        segments, pred_labels, _ = runner.segment_text(content, min_run_chars=min_run_chars)
        pred_idx_array = np.full((len(pred_labels),), -1, dtype=np.int32)
        for i, lbl_id in enumerate(pred_labels):
            label_name = cfg.ID2LANG.get(int(lbl_id), None)
            if label_name is not None:
                alias = PREDICTION_LABEL_ALIASES.get(label_name)
                if alias and alias in label_to_idx:
                    label_name = alias
            if label_name is None or label_name not in label_to_idx:
                pred_idx_array[i] = label_to_idx["__unknown__"]
            else:
                pred_idx_array[i] = label_to_idx[label_name]

        valid_mask = (truth >= 0)
        truth_valid = truth[valid_mask]
        pred_valid = pred_idx_array[valid_mask]
        same_mask = (pred_valid == truth_valid)

        valid_pred_mask = pred_valid >= 0
        np.add.at(confusion, (truth_valid[valid_pred_mask], pred_valid[valid_pred_mask]), 1)

        total_chars += int(valid_mask.sum())
        for lbl_idx, lbl_name in enumerate(label_names):
            mask = truth_valid == lbl_idx
            count = int(mask.sum())
            if count == 0:
                continue
            per_label_counts[lbl_name] += count
            correct = int((same_mask & mask).sum())
            per_label_correct[lbl_name] += correct

        if log_interval > 0 and ((idx + 1) % log_interval == 0 or (idx + 1) == total_samples):
            elapsed = time.perf_counter() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            percent = ((idx + 1) / total_samples) * 100 if total_samples else 100.0
            print(
                f"    [{name}] {idx + 1}/{total_samples} samples ({percent:.1f}%) • {elapsed:.1f}s elapsed • {rate:.1f} samp/s",
                flush=True,
            )

    elapsed_total = time.perf_counter() - start_time

    return TaskMetrics(
        name=name,
        description=description,
        samples=total_samples,
        total_chars=total_chars,
        confusion=confusion,
        label_names=label_names,
        per_label_counts=per_label_counts,
        per_label_correct=per_label_correct,
        extras={"elapsed_seconds": elapsed_total},
    )


# ---------------------------------------------------------------------------
# Throughput benchmarking
# ---------------------------------------------------------------------------


@dataclass
class ThroughputResult:
    task: str
    device: str
    samples: int
    total_bytes: int
    elapsed: float
    throughput: float
    rss_delta: Optional[float]
    device_mem_delta: Optional[float]


def _process_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def _device_mem_mb(device: jax.Device) -> Optional[float]:
    try:
        stats = device.memory_stats()
        used = stats.get("bytes_in_use") or stats.get("allocation_bytes")
        if used is None:
            return None
        return float(used) / (1024 ** 2)
    except Exception:
        return None


def measure_throughput(
    task: str,
    dataset: hfds.Dataset,
    runner: SegmenterRunner,
    *,
    min_run_chars: int,
    device_name: str,
) -> ThroughputResult:
    samples = len(dataset)
    if samples == 0:
        return ThroughputResult(task, device_name, 0, 0, 0.0, 0.0, None, None)

    total_bytes = 0
    for example in dataset:
        meta_raw = example.get("metadata_json")
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
                total_bytes += int(meta.get("actual_bytes", 0))
            except Exception:
                pass
    if total_bytes <= 0:
        total_bytes = int(sum(len(example["content"].encode("utf-8", "ignore")) for example in dataset))

    devices = jax.devices()
    device = devices[0]
    if runner.backend:
        matched = [d for d in devices if d.platform == runner.backend]
        if matched:
            device = matched[0]

    rss_before = _process_rss_mb()
    dev_before = _device_mem_mb(device)

    # Warm-up with first example to trigger compilation
    warm_example = dataset[0]
    runner.segment_text(warm_example["content"], min_run_chars=min_run_chars)

    start = time.perf_counter()
    for example in dataset:
        runner.segment_text(example["content"], min_run_chars=min_run_chars)
    elapsed = time.perf_counter() - start

    rss_after = _process_rss_mb()
    dev_after = _device_mem_mb(device)

    rss_delta = (rss_after - rss_before) if (rss_after is not None and rss_before is not None) else None
    dev_delta = (dev_after - dev_before) if (dev_after is not None and dev_before is not None) else None

    throughput = (total_bytes / elapsed) if elapsed > 0 else 0.0
    return ThroughputResult(
        task=task,
        device=device_name,
        samples=samples,
        total_bytes=total_bytes,
        elapsed=elapsed,
        throughput=throughput,
        rss_delta=rss_delta,
        device_mem_delta=dev_delta,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _extract_run_id_from_checkpoint(path: Path) -> Optional[str]:
    name = path.name.lower()
    match = re.search(r"([a-z0-9]{8})", name)
    if not match:
        return None
    return match.group(1)


def _load_checkpoint_hparams(ckpt_path: Path) -> Dict[str, Any]:
    run_id = _extract_run_id_from_checkpoint(ckpt_path)
    if not run_id:
        return {}
    wandb_root = REPO_ROOT / "train" / "wandb"
    if not wandb_root.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    pattern = f"run-*-{run_id}"
    for run_dir in wandb_root.glob(pattern):
        config_path = run_dir / "files" / "config.yaml"
        if not config_path.exists():
            continue
        try:
            config_data = yaml.safe_load(config_path.read_text())
        except Exception:
            continue
        result: Dict[str, Any] = {}
        if isinstance(config_data, dict):
            channels_val = config_data.get("channels", {}).get("value")
            if isinstance(channels_val, (list, tuple)):
                try:
                    result["channels"] = [int(x) for x in channels_val]
                except (TypeError, ValueError):
                    pass
            model_dim_val = config_data.get("model_dim", {}).get("value")
            if isinstance(model_dim_val, (int, float)):
                result["model_dim"] = int(model_dim_val)
            dtype_val = config_data.get("dtype", {}).get("value")
            if isinstance(dtype_val, str):
                result["dtype"] = dtype_val.rsplit(".", 1)[-1]

        summary_path = run_dir / "files" / "wandb-summary.json"
        if summary_path.exists():
            try:
                summary_data = json.loads(summary_path.read_text())
            except Exception:
                summary_data = None
            if isinstance(summary_data, dict):
                label_names: Dict[int, str] = {}
                prefix = "val/per_class/"
                for key in summary_data.keys():
                    if not key.startswith(prefix):
                        continue
                    remainder = key[len(prefix):]
                    head = remainder.split("/", 1)[0]
                    if "_" not in head:
                        continue
                    idx_str, label = head.split("_", 1)
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
                    label_names[idx] = label
                if label_names:
                    ordered = [label_names[i] for i in sorted(label_names)]
                    result["label_names"] = ordered
        if result:
            return result
    return {}


def _apply_label_mapping(label_names: Sequence[str]) -> None:
    cfg.LANG2ID.clear()
    cfg.LANG2ID.update({name: idx for idx, name in enumerate(label_names)})
    cfg.update_lang_mappings()


def _format_bytes_per_sec(value: float) -> str:
    if value <= 0:
        return "0"
    units = ["B/s", "KB/s", "MB/s", "GB/s"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _render_metrics_table(metrics: TaskMetrics) -> str:
    rows = []
    for label in metrics.label_names:
        support = metrics.per_label_counts.get(label, 0)
        correct = metrics.per_label_correct.get(label, 0)
        acc = (correct / support) if support else float("nan")
        rows.append((label, support, acc))
    rows.sort(key=lambda r: (-r[1], r[0]))
    lines = ["| Label | Support | Accuracy |", "| --- | ---: | ---: |"]
    for label, support, acc in rows:
        acc_str = "nan" if math.isnan(acc) else f"{acc:.4f}"
        lines.append(f"| {label} | {support} | {acc_str} |")
    return "\n".join(lines)


def _render_summary_table(task_metrics: List[TaskMetrics]) -> str:
    lines = ["| Task | Samples | Characters | Char Acc | Macro Recall |", "| --- | ---: | ---: | ---: | ---: |"]
    for metrics in task_metrics:
        per_class, aggregates = compute_metrics_from_confusion(metrics.confusion.astype(np.int64), len(metrics.label_names), ignore_class=None)
        char_acc = metrics.overall_accuracy()
        macro_rec = aggregates["macro"]["recall"]
        lines.append(
            f"| {metrics.name} | {metrics.samples} | {metrics.total_chars} | {char_acc:.4f} | {macro_rec:.4f} |"
        )
    return "\n".join(lines)


def _render_throughput_table(results: List[ThroughputResult]) -> str:
    lines = ["| Task | Device | Samples | Total Bytes | Throughput | Latency (s) | RSS Δ (MB) | Device Δ (MB) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for res in results:
        lines.append(
            "| {task} | {device} | {samples} | {bytes} | {through} | {lat:.2f} | {rss} | {dev} |".format(
                task=res.task,
                device=res.device,
                samples=res.samples,
                bytes=res.total_bytes,
                through=_format_bytes_per_sec(res.throughput),
                lat=res.elapsed,
                rss="n/a" if res.rss_delta is None else f"{res.rss_delta:.2f}",
                dev="n/a" if res.device_mem_delta is None else f"{res.device_mem_delta:.2f}",
            )
        )
    return "\n".join(lines)


def write_report(
    report_path: Path,
    *,
    manifest: dict,
    args,
    task_metrics: List[TaskMetrics],
    throughput_results: List[ThroughputResult],
) -> None:
    def _format_channels(value) -> str:
        if isinstance(value, str):
            return value
        try:
            return ", ".join(str(ch) for ch in value)
        except TypeError:
            return str(value)

    report_lines: List[str] = []
    report_lines.append("# Segmenter Evaluation Report")
    report_lines.append("")
    report_lines.append(f"- Checkpoint: `{args.checkpoint}`")
    report_lines.append(f"- Model dim: {args.model_dim}")
    report_lines.append(f"- Channels: {_format_channels(args.channels)}")
    report_lines.append(f"- Chunk: {args.chunk}")
    report_lines.append(f"- Batch size: {args.batch_size}")
    report_lines.append(f"- Max samples per task: {'all' if args.max_samples <= 0 else args.max_samples}")
    report_lines.append(f"- Sample seed: {args.sample_seed}")
    report_lines.append(f"- Evaluation data root: `{manifest.get('output_root')}`")
    report_lines.append(f"- Generated at: {manifest.get('generated_at')}")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append(_render_summary_table(task_metrics))
    report_lines.append("")

    report_lines.append("## Task Details")
    report_lines.append("")
    for metrics in task_metrics:
        report_lines.append(f"### {metrics.name}")
        report_lines.append("")
        report_lines.append(metrics.description)
        report_lines.append("")
        report_lines.append(f"- Samples: {metrics.samples}")
        report_lines.append(f"- Characters evaluated: {metrics.total_chars}")
        report_lines.append(f"- Overall accuracy: {metrics.overall_accuracy():.4f}")
        
        # Special analysis for pure fragments dataset
        if metrics.name == "pure_fragments":
            purity_scores, misclassifications = _analyze_pure_fragments(metrics)
            report_lines.append("\n### Purity Analysis\n")
            report_lines.append("| Language | Purity % | Top Misclassifications |")
            report_lines.append("| --- | ---: | --- |")
            
            for lang, purity in sorted(purity_scores.items(), key=lambda x: x[1], reverse=True):
                misclass_str = ""
                if lang in misclassifications:
                    top_mistakes = [f"{label} ({pct:.1f}%)" 
                                  for label, pct in list(misclassifications[lang].items())[:3]]
                    misclass_str = ", ".join(top_mistakes)
                report_lines.append(f"| {lang} | {purity:.1f}% | {misclass_str} |")
            report_lines.append("")
            
        per_label_table = _render_metrics_table(metrics)
        report_lines.append("")
        report_lines.append(per_label_table)
        report_lines.append("")

    if throughput_results:
        report_lines.append("## Throughput Benchmarks")
        report_lines.append("")
        report_lines.append(_render_throughput_table(throughput_results))

    report_lines.append("")
    report_lines.append("Report generated at " + time.strftime("%Y-%m-%d %H:%M:%S"))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on curated benchmarks.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.msgpack or Orbax directory).")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "evaluation" / "data"), help="Path to evaluation datasets.")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON (defaults to <data-root>/manifest.json).")
    parser.add_argument("--tasks", nargs="*", help="Subset of task names to evaluate.")
    parser.add_argument("--device", default="auto", help="Device for accuracy workloads (cpu/gpu/cuda/auto).")
    parser.add_argument("--cpu-device", default="cpu", help="Device name for throughput CPU benchmark.")
    parser.add_argument("--gpu-device", default="cuda", help="Device name for throughput GPU benchmark (ignored if unavailable).")
    parser.add_argument("--model-dim", type=int, default=None, help="Model embedding dimension. Defaults to checkpoint config or 256.")
    parser.add_argument("--channels", type=str, default=None, help="Comma-separated channel widths. Defaults to checkpoint config or 96,128,192,256.")
    parser.add_argument("--dtype", type=str, default=None, help="JAX dtype name for inference (e.g. bfloat16). Defaults to checkpoint config or bfloat16.")
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--min-run", type=int, default=1, help="Minimum run-length smoothing for character labels.")
    parser.add_argument("--report-path", default=str(REPO_ROOT / "evaluation" / "report.md"))
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference windows.")
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum samples per accuracy task (0 = all).")
    parser.add_argument("--sample-seed", type=int, default=13, help="Seed for subsampling large datasets.")
    parser.add_argument("--log-interval", type=int, default=250, help="Progress logging interval (in samples).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root).resolve()
    manifest_path = Path(args.manifest) if args.manifest else data_root / "manifest.json"
    overall_start = time.perf_counter()

    try:
        if manifest_path.exists():
            manifest = _load_manifest(manifest_path)
        else:
            manifest = {"generated_at": "n/a", "output_root": str(data_root)}

        if not data_root.exists() or not data_root.is_dir():
            raise RuntimeError(f"Evaluation data directory not found or not a directory: {data_root}")

        ckpt_path = Path(args.checkpoint).resolve()
        auto_hparams = _load_checkpoint_hparams(ckpt_path)

        label_names = auto_hparams.get("label_names")
        if label_names:
            _apply_label_mapping(label_names)

        model_dim = args.model_dim if args.model_dim is not None else auto_hparams.get("model_dim", 256)
        dtype = args.dtype if args.dtype is not None else auto_hparams.get("dtype", "bfloat16")
        dtype = dtype.rsplit(".", 1)[-1]
        if args.channels:
            channels = [int(ch) for ch in args.channels.split(",") if ch.strip()]
        else:
            channels_source = auto_hparams.get("channels", DEFAULT_CHANNELS)
            channels = [int(ch) for ch in channels_source]

        args.model_dim = model_dim
        args.dtype = dtype
        args.channels = channels

        if auto_hparams:
            extra = f", classes={len(label_names)}" if label_names else ""
            print(f"ℹ️  Using checkpoint hyperparameters: model_dim={model_dim}, channels={channels}, dtype={dtype}{extra}", flush=True)

        accuracy_runner = SegmenterRunner(
            args.checkpoint,
            model_dim=model_dim,
            channels=channels,
            dtype=dtype,
            chunk=args.chunk,
            device=args.device,
            batch_size=args.batch_size,
        )

        datasets = _collect_datasets(data_root, args.tasks)
        if not datasets:
            raise RuntimeError(f"No datasets found under {data_root}")

        evaluate_task._log_interval = args.log_interval  # type: ignore[attr-defined]

        task_descriptions = {
            entry.get("task"): entry.get("description", "")
            for entry in manifest.get("tasks", [])
            if isinstance(entry, dict) and entry.get("task")
        }

        prepared_datasets: Dict[str, hfds.Dataset] = {}
        dataset_meta: Dict[str, Dict[str, Any]] = {}
        for name, ds in datasets.items():
            preserve_all = name.startswith("throughput_")
            subset, original_len, subset_len, sampled = _prepare_dataset(
                name,
                ds,
                max_samples=args.max_samples,
                base_seed=args.sample_seed,
                preserve_all=preserve_all,
            )
            prepared_datasets[name] = subset
            dataset_meta[name] = {
                "original": original_len,
                "selected": subset_len,
                "sampled": sampled,
                "description": task_descriptions.get(name, ""),
            }

        accuracy_names = [name for name in prepared_datasets if not name.startswith("throughput_")]
        throughput_names = [name for name in prepared_datasets if name.startswith("throughput_")]
        print(
            f"ℹ️  Prepared {len(prepared_datasets)} datasets "
            f"(accuracy={len(accuracy_names)}, throughput={len(throughput_names)})",
            flush=True,
        )

        task_metrics: List[TaskMetrics] = []
        throughput_results: List[ThroughputResult] = []

        for name in sorted(accuracy_names):
            ds = prepared_datasets[name]
            meta = dataset_meta[name]
            sample_note = " • subsampled" if meta["sampled"] else ""
            print(
                f"▶️  Evaluating task '{name}' [{meta['selected']}/{meta['original']} samples{sample_note}]",
                flush=True,
            )
            metrics = evaluate_task(
                name,
                meta["description"],
                ds,
                accuracy_runner,
                min_run_chars=args.min_run,
            )
            elapsed = metrics.extras.get("elapsed_seconds", 0.0)
            print(
                f"    [{name}] completed in {elapsed:.1f}s • char_acc={metrics.overall_accuracy():.4f}",
                flush=True,
            )
            task_metrics.append(metrics)

        # Throughput datasets evaluated separately on CPU and GPU (if available)
        if throughput_names:
            try:
                cpu_runner = SegmenterRunner(
                    args.checkpoint,
                    model_dim=args.model_dim,
                    channels=channels,
                    dtype=args.dtype,
                    chunk=args.chunk,
                    device=args.cpu_device,
                    batch_size=args.batch_size,
                )
            except RuntimeError as exc:
                print(f"⚠️  Skipping CPU throughput ({exc})", flush=True)
                cpu_runner = None
            if cpu_runner is not None:
                for name in sorted(throughput_names):
                    ds = prepared_datasets[name]
                    meta = dataset_meta[name]
                    print(
                        f"▶️  Throughput '{name}' on {args.cpu_device} [{meta['selected']} samples]",
                        flush=True,
                    )
                    result = measure_throughput(
                        name,
                        ds,
                        cpu_runner,
                        min_run_chars=args.min_run,
                        device_name=args.cpu_device,
                    )
                    print(
                        f"    [{name}/{args.cpu_device}] {_format_bytes_per_sec(result.throughput)} • {result.elapsed:.2f}s",
                        flush=True,
                    )
                    throughput_results.append(result)

            # GPU benchmark only if GPU available
            try:
                gpu_devices = jax.devices("gpu")
            except Exception:
                gpu_devices = []
            if gpu_devices:
                try:
                    gpu_runner = SegmenterRunner(
                        args.checkpoint,
                        model_dim=args.model_dim,
                        channels=channels,
                        dtype=args.dtype,
                        chunk=args.chunk,
                        device=args.gpu_device,
                        batch_size=args.batch_size,
                    )
                except RuntimeError as exc:
                    print(f"⚠️  Skipping {args.gpu_device} throughput ({exc})", flush=True)
                    gpu_runner = None
                if gpu_runner is not None:
                    for name in sorted(throughput_names):
                        ds = prepared_datasets[name]
                        print(
                            f"▶️  Throughput '{name}' on {args.gpu_device} [{len(ds)} samples]",
                            flush=True,
                        )
                        result = measure_throughput(
                            name,
                            ds,
                            gpu_runner,
                            min_run_chars=args.min_run,
                            device_name=args.gpu_device,
                        )
                        print(
                            f"    [{name}/{args.gpu_device}] {_format_bytes_per_sec(result.throughput)} • {result.elapsed:.2f}s",
                            flush=True,
                        )
                        throughput_results.append(result)
            else:
                print("⚠️  GPU device not available; skipping CUDA throughput benchmarks.", flush=True)

        report_path = Path(args.report_path)

        if not task_metrics and not throughput_results:
            raise RuntimeError("No tasks were evaluated. Check if the evaluation data directory contains valid datasets.")

        write_report(report_path, manifest=manifest, args=args, task_metrics=task_metrics, throughput_results=throughput_results)
        total_elapsed = time.perf_counter() - overall_start
        print(f"✅ Evaluation complete in {total_elapsed:.1f}s. Report written to {report_path}", flush=True)
        return 0

    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
