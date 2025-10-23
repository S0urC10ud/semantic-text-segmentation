"""
Functions for generating training windows from datasets, with lightweight
epoch-style progress tracking that matches the sampling distribution used
for validation.
"""
import random
import queue
import threading
import time
from typing import Dict, List

import numpy as np
import datasets as hfds

from config import DataConfig, PAD_ID, PAD_BYTE_ID
from window_generator import make_training_window


class EpochPrefetchBatcher:
    def __init__(self, dsets_by_lang: Dict[int, hfds.Dataset], cfg: DataConfig):
        self.dsets_by_lang = dsets_by_lang
        self.cfg = cfg
        self.q = queue.Queue(maxsize=max(2, cfg.prefetch_batches))
        self.stop_flag = threading.Event()
        self.buckets = cfg.buckets()
        self.threads: List[threading.Thread] = []

        # Track per-language token usage to estimate fractional epochs
        self._lang_token_counts = {lang_id: 0 for lang_id in dsets_by_lang.keys()}
        self._token_targets = {}
        max_len = max(1, cfg.window_max_bytes)
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
        for wid in range(max(1, cfg.num_workers // 2)):
            t = threading.Thread(target=self._worker, args=(wid,), daemon=True)
            t.start()
            self.threads.append(t)

    def _update_token_counts(self, labels: np.ndarray):
        valid = labels[labels != PAD_ID]
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

            xb = np.full((self.cfg.batch_size, L), PAD_BYTE_ID, dtype=np.int32)
            yb = np.full((self.cfg.batch_size, L), PAD_ID, dtype=np.uint8)

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
