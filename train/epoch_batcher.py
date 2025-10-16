"""
Functions for generating training windows from datasets, with epoch-based iteration.
"""
import math
import random
import queue
import threading
import time
from typing import List, Tuple, Dict, Optional, Iterator

import numpy as np
import datasets as hfds

from config import DataConfig, PAD_ID, PAD_BYTE_ID
from data_utils import bytes_from_text

class EpochIterator:
    def __init__(self, dataset: hfds.Dataset):
        self.dataset = dataset
        self.current_idx = 0
        self._length = None
        self.epoch = 0
        # Try to get length, fallback for streaming datasets
        try:
            self._length = len(dataset)
        except Exception:
            pass
    
    def __iter__(self):
        return self
    
    def __next__(self) -> dict:
        if self._length is not None:
            if self.current_idx >= self._length:
                self.current_idx = 0
                self.epoch += 1
            example = self.dataset[self.current_idx]
            self.current_idx += 1
            return example
        else:
            # For streaming datasets, do our best to iterate
            try:
                return next(iter(self.dataset))
            except StopIteration:
                self.epoch += 1
                return next(iter(self.dataset))

class EpochPrefetchBatcher:
    def __init__(self, dsets_by_lang: Dict[int, hfds.Dataset], cfg: DataConfig):
        self.dsets_by_lang = dsets_by_lang
        self.cfg = cfg
        self.q = queue.Queue(maxsize=max(2, cfg.prefetch_batches))
        self.stop_flag = threading.Event()
        self.buckets = cfg.buckets()
        self.threads: List[threading.Thread] = []
        
        # Create iterators for each dataset
        self.iterators = {
            lang_id: EpochIterator(dataset) 
            for lang_id, dataset in dsets_by_lang.items()
        }

        # Use fewer worker threads for better determinism
        for wid in range(max(1, cfg.num_workers // 2)):
            t = threading.Thread(target=self._worker, args=(wid,), daemon=True)
            t.start()
            self.threads.append(t)

    def _make_window(self, example: dict, lang_id: int, target_len: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.full((target_len,), PAD_BYTE_ID, dtype=np.int32)
        y = np.full((target_len,), PAD_ID, dtype=np.uint8)
        
        if not example or not example.get("content", ""):
            return x, y

        b = bytes_from_text(example["content"])
        if len(b) == 0:
            return x, y

        if len(b) >= target_len:
            start = random.randint(0, len(b) - target_len)
            b = b[start:start + target_len]

        L = len(b)
        x[:L] = b.astype(np.int32)
        y[:L] = lang_id
        return x, y

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

            # For each item in the batch
            for i in range(self.cfg.batch_size):
                # Randomly choose a language but iterate through its files sequentially
                lang_id = random.choice(list(self.iterators.keys()))
                example = next(self.iterators[lang_id])
                
                # Create window from example
                x, y = self._make_window(example, lang_id, L)
                xb[i], yb[i] = x, y

            try:
                self.q.put((xb, yb), timeout=1.0)
            except queue.Full:
                continue

    def get_epochs(self) -> Dict[int, int]:
        """Returns a dictionary of language ID to current epoch number"""
        return {lang_id: it.epoch for lang_id, it in self.iterators.items()}

    def get(self):
        return self.q.get()

    def close(self):
        self.stop_flag.set()
        # Drain queue to unblock workers
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        for t in self.threads:
            t.join(timeout=2.0)