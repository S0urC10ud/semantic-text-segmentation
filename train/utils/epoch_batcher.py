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

    def get(self):
        xb, yb = self.q.get()
        self._update_token_counts(yb)
        return xb, yb

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


class MonitorFineTuneBatcher:
    """
    Simple prefetching batcher for fine-tuning on the preprocessed monitor
    memmap (monitor_preprocessed_a / _b). It samples random files and random
    windows without any mixing/augmentation.
    """

    def __init__(
        self,
        monitor_data: Dict[str, np.ndarray],
        data_cfg: "DataConfig",
        monitor_root: str,
    ):
        import multiprocessing as mp
        import sys
        from pathlib import Path
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        if not monitor_root:
            raise ValueError(
                "MonitorFineTuneBatcher requires a non-empty monitor_root path."
            )

        self.cfg = data_cfg

        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=max(2, data_cfg.prefetch_batches))
        self.stop_flag = ctx.Event()
        self.threads: List[mp.Process] = []

        self.total_files = int(len(monitor_data["files"]))
        self._processed_windows = 0

        for wid in range(max(1, data_cfg.num_workers)):
            t = ctx.Process(target=self._worker_entry, args=(self.cfg, wid, monitor_root, self.q, self.stop_flag, self.total_files), daemon=True)
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
    def _build_window(rng: np.random.Generator, window_len: int, files, contents, segments, total_files):
        max_attempts = 32
        for _ in range(max_attempts):
            if total_files <= 0:
                return None
            file_idx = int(rng.integers(0, total_files))
            row = files[file_idx]
            byte_len = int(row["byte_len"])
            if byte_len <= 0:
                continue

            start = 0
            if byte_len > window_len:
                start = int(rng.integers(0, byte_len - window_len + 1))
            end = start + min(window_len, byte_len)

            full_slice = contents[
                int(row["byte_start"]) : int(row["byte_start"]) + byte_len
            ]
            x = np.full(window_len, cfg.PAD_BYTE_ID, dtype=np.int32)
            x[: end - start] = np.asarray(full_slice[start:end], dtype=np.uint8)

            y = np.full(window_len, cfg.PAD_ID, dtype=np.uint8)
            seg_start = int(row["seg_start"])
            seg_count = int(row["seg_count"])
            seg_slice = segments[seg_start : seg_start + seg_count]
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

    @staticmethod
    def _worker_entry(cfg_obj, wid, monitor_root, q, stop_flag, total_files):
        import sys
        from pathlib import Path
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

        rng = np.random.default_rng(cfg_obj.seed ^ wid ^ int(time.time()))
        window_len = int(cfg_obj.window_max_bytes)
        if window_len <= 0:
            window_len = cfg.MODEL_WINDOW_BYTES

        while not stop_flag.is_set():
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
                window = MonitorFineTuneBatcher._build_window(rng, window_len, files, contents, segments, total_files)
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

