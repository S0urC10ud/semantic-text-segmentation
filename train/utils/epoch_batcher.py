"""
Functions for generating training windows from datasets, with lightweight
epoch-style progress tracking that matches the sampling distribution used
for validation.
"""
import queue
import random
import threading
import time
from typing import TYPE_CHECKING, Dict, List

import datasets as hfds
import numpy as np
import utils.config as cfg
from utils.window_generator import make_training_window

if TYPE_CHECKING:
    from utils.config import DataConfig


class EpochPrefetchBatcher:
    def __init__(self, dsets_by_lang: Dict[int, hfds.Dataset], data_cfg: "DataConfig"):
        self.dsets_by_lang = dsets_by_lang
        self.cfg = data_cfg
        self.q = queue.Queue(maxsize=max(2, data_cfg.prefetch_batches))
        self.stop_flag = threading.Event()
        self.buckets = data_cfg.buckets()
        self.threads: List[threading.Thread] = []

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
        self._lock = threading.Lock()

        # Use fewer worker threads for better determinism
        for wid in range(max(1, data_cfg.num_workers // 2)):
            t = threading.Thread(target=self._worker, args=(wid,), daemon=True)
            t.start()
            self.threads.append(t)

    def _update_token_counts(self, labels: np.ndarray):
        valid = labels[labels != cfg.PAD_ID]
        if valid.size == 0:
            return
        unique, counts = np.unique(valid, return_counts=True)
        with self._lock:
            for lid, cnt in zip(unique.astype(int), counts.astype(int)):
                if lid in self._lang_token_counts:
                    self._lang_token_counts[lid] += cnt

    def _worker(self, wid: int):
        random.seed(self.cfg.seed ^ wid ^ int(time.time()))
        hold = max(1, self.cfg.bucket_hold_steps)
        L = random.choice(self.buckets)
        k = 0

        while not self.stop_flag.is_set():
            if k % hold == 0:
                L = random.choice(self.buckets)
            k += 1

            xb = np.full((self.cfg.batch_size, L), cfg.PAD_BYTE_ID, dtype=np.int32)
            yb = np.full((self.cfg.batch_size, L), cfg.PAD_ID, dtype=np.uint8)

            for i in range(self.cfg.batch_size):
                x, y = make_training_window(self.dsets_by_lang, L, self.cfg)
                xb[i], yb[i] = x, y
                self._update_token_counts(y)

            try:
                self.q.put((xb, yb), timeout=1.0)
            except queue.Full:
                continue

    def get_epochs(self) -> Dict[int, float]:
        """Approximate fractional epochs per language based on token usage."""
        with self._lock:
            epochs = {}
            for lang_id, tokens in self._lang_token_counts.items():
                target = max(1, self._token_targets.get(lang_id, 1))
                epochs[lang_id] = tokens / target
            return epochs

    def get(self):
        return self.q.get()

    def close(self):
        self.stop_flag.set()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        for t in self.threads:
            t.join(timeout=2.0)


class MonitorFineTuneBatcher:
    """
    Simple prefetching batcher for fine-tuning on the preprocessed monitor
    memmap (monitor_preprocessed_a / _b). It samples random files and random
    windows without any mixing/augmentation.
    """

    def __init__(self, monitor_data: Dict[str, np.ndarray], data_cfg: "DataConfig"):
        self.files = monitor_data["files"]
        self.segments = monitor_data["segments"]
        self.contents = monitor_data["contents"]
        self.cfg = data_cfg

        self.q = queue.Queue(maxsize=max(2, data_cfg.prefetch_batches))
        self.stop_flag = threading.Event()
        self.threads: List[threading.Thread] = []

        self.total_files = int(len(self.files))
        self._lock = threading.Lock()
        self._processed_windows = 0

        # Use fewer worker threads for better determinism
        for wid in range(max(1, data_cfg.num_workers // 2)):
            t = threading.Thread(target=self._worker, args=(wid,), daemon=True)
            t.start()
            self.threads.append(t)

    def _build_window(self, rng: np.random.Generator, window_len: int):
        max_attempts = 32
        for _ in range(max_attempts):
            if self.total_files <= 0:
                return None
            file_idx = int(rng.integers(0, self.total_files))
            row = self.files[file_idx]
            byte_len = int(row["byte_len"])
            if byte_len <= 0:
                continue

            start = 0
            if byte_len > window_len:
                start = int(rng.integers(0, byte_len - window_len + 1))
            end = start + min(window_len, byte_len)

            full_slice = self.contents[
                int(row["byte_start"]) : int(row["byte_start"]) + byte_len
            ]
            x = np.full(window_len, cfg.PAD_BYTE_ID, dtype=np.int32)
            x[: end - start] = np.asarray(full_slice[start:end], dtype=np.uint8)

            y = np.full(window_len, cfg.PAD_ID, dtype=np.uint8)
            seg_start = int(row["seg_start"])
            seg_count = int(row["seg_count"])
            seg_slice = self.segments[seg_start : seg_start + seg_count]
            for seg in seg_slice:
                seg_s = int(seg["start"])
                seg_e = int(seg["end"])
                label = int(seg["label"])
                overlap_s = max(seg_s, start)
                overlap_e = min(seg_e, start + window_len)
                if overlap_e <= overlap_s:
                    continue
                y_start = overlap_s - start
                y_end = overlap_e - start
                y[y_start:y_end] = label

            return x, y
        return None

    def _worker(self, wid: int):
        rng = np.random.default_rng(self.cfg.seed ^ wid ^ int(time.time()))
        window_len = int(self.cfg.window_max_bytes)
        if window_len <= 0:
            window_len = cfg.MODEL_WINDOW_BYTES

        while not self.stop_flag.is_set():
            xb = np.full(
                (self.cfg.batch_size, window_len),
                cfg.PAD_BYTE_ID,
                dtype=np.int32,
            )
            yb = np.full(
                (self.cfg.batch_size, window_len),
                cfg.PAD_ID,
                dtype=np.uint8,
            )

            filled = 0
            while filled < self.cfg.batch_size and not self.stop_flag.is_set():
                window = self._build_window(rng, window_len)
                if window is None:
                    # If we repeatedly fail to build a window, just break and
                    # reuse whatever portion we have so far.
                    break
                x, y = window
                xb[filled] = x
                yb[filled] = y
                filled += 1
                with self._lock:
                    self._processed_windows += 1

            if filled == 0:
                continue

            try:
                self.q.put((xb, yb), timeout=1.0)
            except queue.Full:
                continue

    def get_epochs(self) -> Dict[int, float]:
        """Approximate global epochs over the monitor memmap."""
        with self._lock:
            if self.total_files <= 0:
                return {}
            epochs = float(self._processed_windows) / float(self.total_files)
        # Use a single synthetic key for logging.
        return {0: epochs}

    def get(self):
        return self.q.get()

    def close(self):
        self.stop_flag.set()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        for t in self.threads:
            t.join(timeout=2.0)
