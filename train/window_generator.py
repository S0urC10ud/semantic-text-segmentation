"""
Functions for generating training windows from datasets, including pure, mixed,
and line-injected augmentation. Also includes the multi-threaded prefetcher.
"""
import math
import random
import re
import threading
import queue
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import datasets as hfds

from config import DataConfig, PAD_ID, PAD_BYTE_ID
from data_utils import bytes_from_text

# ---------------------------
# Helpers
# ---------------------------

def _random_example(ds) -> Optional[dict]:
    """Return a random example from a (non-streaming) HF Dataset.
    Falls back to a best-effort sample for streaming/unsized datasets.
    """
    n = None
    try:
        n = len(ds)  # Works for non-streaming Arrow datasets
    except Exception:
        pass

    if n is not None and n > 0:
        idx = random.randrange(0, n)
        return ds[idx]
# ---------------------------
# Window builders (supports partial fill + masking)
# ---------------------------

def make_pure_window(dsets_by_lang: Dict[int, hfds.Dataset],
                     target_len: int) -> Tuple[np.ndarray, np.ndarray]:
    lids = list(dsets_by_lang.keys())
    x = np.full((target_len,), PAD_BYTE_ID, dtype=np.int32)
    y = np.full((target_len,), PAD_ID, dtype=np.uint8)
    if not lids:
        return x, y

    lid = random.choice(lids)
    sub = dsets_by_lang[lid]

    ex = _random_example(sub)
    if not ex or not ex.get("content", ""):
        return x, y

    b = bytes_from_text(ex["content"])
    if len(b) == 0:
        return x, y

    if len(b) >= target_len:
        start = random.randint(0, len(b) - target_len)
        b = b[start:start + target_len]

    L = len(b)
    x[:L] = b.astype(np.int32)
    y[:L] = lid
    return x, y


def make_mixed_window(dsets_by_lang: Dict[int, hfds.Dataset],
                      target_len: int,
                      min_seg: int) -> Tuple[np.ndarray, np.ndarray]:
    # inputs are int32 to allow PAD_BYTE_ID=256
    x_buf = np.full((target_len,), PAD_BYTE_ID, dtype=np.int32)
    y_buf = np.full((target_len,), PAD_ID, dtype=np.uint8)

    lids = list(dsets_by_lang.keys())
    if not lids:
        return x_buf, y_buf

    nsegs = random.randint(1, 6)
    left, pos = target_len, 0
    for i in range(nsegs):
        seg_len = random.randint(min_seg, left) if left > min_seg else left
        if i == nsegs - 1:
            seg_len = left
        if seg_len <= 0:
            break

        lid = random.choice(lids)
        sub = dsets_by_lang[lid]

        ex = _random_example(sub)
        if not ex or not ex.get("content", ""):
            # leave this segment as PAD
            pos += seg_len
            left -= seg_len
            continue

        b = bytes_from_text(ex["content"])
        if len(b) > seg_len:
            start = random.randint(0, len(b) - seg_len)
            b = b[start:start + seg_len]

        L = min(seg_len, len(b))
        if L > 0:
            x_buf[pos:pos + L] = b[:L].astype(np.int32)
            y_buf[pos:pos + L] = lid

        pos += seg_len
        left -= seg_len

    return x_buf, y_buf


# ---------------------------
# LINE-LEVEL INJECTION
# ---------------------------

def _split_keepends_lines(text: str) -> List[str]:
    return text.splitlines(keepends=True) if text else []


def _sample_truncated_exp_lines(lam: float, max_lines: int) -> int:
    k = int(math.ceil(random.expovariate(lam))) if lam > 0 else 1
    return max(1, min(k, max_lines))


def _choose_injection_boundaries(host_text: str, skip_top_min: int, skip_top_max: int, max_inj: int) -> List[int]:
    lines = _split_keepends_lines(host_text)
    if not lines:
        return []
    boundaries = np.cumsum([len(ln) for ln in lines]).tolist()
    skip = random.randint(skip_top_min, max(skip_top_min, skip_top_max))
    valid = [b for b in boundaries if b > boundaries[skip - 1]] if skip < len(boundaries) else []
    if not valid:
        return []
    n = 1
    # More injections per file
    while n < max_inj and random.random() < 0.75:
        n += 1
    n = min(n, len(valid))
    picks = random.sample(valid, k=n)
    return sorted(picks, reverse=True)


def _leading_indent_of_line(chars: List[str], at_char_idx: int) -> str:
    i = at_char_idx - 1
    while i >= 0 and chars[i] != "\n":
        i -= 1
    line_start = i + 1
    j = line_start
    indent = []
    while j < len(chars) and chars[j] in (" ", "\t"):
        indent.append(chars[j]); j += 1
    return "".join(indent)


def _prepare_donor_block(donor_text: str, cfg: DataConfig, insertion_indent: str) -> str:
    lines = _split_keepends_lines(donor_text)
    if not lines:
        return ""

    skip = random.randint(cfg.donor_skip_top_min, max(cfg.donor_skip_top_min, cfg.donor_skip_top_max))
    start_idx = min(skip, len(lines) - 1)

    L = _sample_truncated_exp_lines(cfg.line_inject_exp_rate, cfg.line_inject_max_lines)
    end_idx = min(len(lines), start_idx + L)
    pick = lines[start_idx:end_idx]

    if len(pick) == 1 and len(pick[0]) < cfg.line_inject_min_single_len:
        if end_idx < len(lines):
            pick.append(lines[end_idx])
        elif start_idx > 0:
            pick.insert(0, lines[start_idx - 1])

    w_none, w_l, w_r, w_b = cfg.strip_weights
    mode = random.choices(["none", "lstrip", "rstrip", "strip"], weights=[w_none, w_l, w_r, w_b], k=1)[0]

    processed = []
    for ln in pick:
        core = ln.rstrip('\n')
        if mode == "lstrip":
            core = core.lstrip()
        elif mode == "rstrip":
            core = core.rstrip()
        elif mode == "strip":
            core = core.strip()

        if random.random() < cfg.reindent_prob:
            core = insertion_indent + core.lstrip()

        processed.append(core + ('\n' if ln.endswith('\n') else ''))
    return "".join(processed)


def _insert_block_at(chars: List[str], labels: List[int], idx: int, block: str, lid: int):
    if not block:
        return
    ins_chars = list(block)
    ins_labs = [lid] * len(ins_chars)
    chars[idx:idx] = ins_chars
    labels[idx:idx] = ins_labs


def _ensure_merge_without_newline(chars: List[str], labels: List[int], idx: int) -> int:
    if idx > 0 and chars[idx - 1] == "\n":
        del chars[idx - 1]
        del labels[idx - 1]
        return idx - 1
    return idx


def _to_bytes_with_byte_labels(chars: List[str], char_labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    xb, yb = [], []
    for ch, lid in zip(chars, char_labels):
        bs = ch.encode("utf-8", "ignore")
        xb.extend(bs)
        yb.extend([lid] * len(bs))
    return np.array(xb, dtype=np.uint8), np.array(yb, dtype=np.uint8)


def make_line_injected_window(dsets_by_lang: Dict[int, hfds.Dataset],
                              target_len: int,
                              cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
    lids = list(dsets_by_lang.keys())
    x = np.full((target_len,), PAD_BYTE_ID, dtype=np.int32)
    y = np.full((target_len,), PAD_ID, dtype=np.uint8)
    if not lids:
        return x, y

    host_lid = random.choice(lids)
    host_ds = dsets_by_lang[host_lid]

    host_ex = _random_example(host_ds)
    if not host_ex or not host_ex.get("content", ""):
        return make_pure_window(dsets_by_lang, target_len)

    host_text = host_ex["content"] or ""
    if not host_text:
        return make_pure_window(dsets_by_lang, target_len)

    chars, labs = list(host_text), [host_lid] * len(host_text)

    boundaries = _choose_injection_boundaries(
        host_text, cfg.host_skip_top_min, cfg.host_skip_top_max, cfg.line_inject_max_injections
    )

    for bidx in boundaries:
        donor_lid = random.choice(lids)
        if not cfg.allow_same_lang_injection and donor_lid == host_lid:
            alts = [l for l in lids if l != host_lid]
            if alts:
                donor_lid = random.choice(alts)

        donor_ds = dsets_by_lang[donor_lid]
        donor_ex = _random_example(donor_ds)
        donor_text = donor_ex.get("content", "") if donor_ex else ""

        insertion_indent = _leading_indent_of_line(chars, bidx)
        donor_block = _prepare_donor_block(donor_text, cfg, insertion_indent)

        if random.random() > cfg.start_with_newline_prob:
            bidx = _ensure_merge_without_newline(chars, labs, bidx)

        _insert_block_at(chars, labs, bidx, donor_block, donor_lid)

    xb_u8, yb_u8 = _to_bytes_with_byte_labels(chars, labs)

    # Windowing/padding
    if len(xb_u8) >= target_len:
        start = random.randint(0, len(xb_u8) - target_len)
        x = xb_u8[start:start + target_len].astype(np.int32)
        y = yb_u8[start:start + target_len]
    else:
        x[:len(xb_u8)] = xb_u8.astype(np.int32)
        y[:len(yb_u8)] = yb_u8
    return x, y


# ---------------------------
# Prefetcher (multi-thread, variable-length buckets)
# ---------------------------

class PrefetchBatcher:
    def __init__(self, dsets_by_lang, cfg: DataConfig):
        self.dsets_by_lang = dsets_by_lang
        self.cfg = cfg
        # Smaller queue size to prevent memory buildup
        self.q = queue.Queue(maxsize=max(2, cfg.prefetch_batches))
        self.stop_flag = threading.Event()
        self.buckets = cfg.buckets()
        self.threads: List[threading.Thread] = []
        self._gc_counter = 0
        # Use fewer worker threads
        for wid in range(max(1, cfg.num_workers // 2)):
            t = threading.Thread(target=self._worker, args=(wid,), daemon=True)
            t.start()
            self.threads.append(t)

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
                r = random.random()
                # Base mode selection
                if r < self.cfg.pure_prob:
                    x, y = make_pure_window(self.dsets_by_lang, L)
                elif r < self.cfg.pure_prob + self.cfg.line_inject_prob:
                    x, y = make_line_injected_window(self.dsets_by_lang, L, self.cfg)
                else:
                    x, y = make_mixed_window(self.dsets_by_lang, L, self.cfg.min_seg_len)

                # Occasionally combine both (line-injected + mixed) in one sample
                both_prob = getattr(self.cfg, "both_prob", 0.2)
                if random.random() < both_prob:
                    xm, ym = make_mixed_window(self.dsets_by_lang, L, self.cfg.min_seg_len)
                    # Replace 1-3 random slices to blend modes
                    nslices = random.randint(1, 3)
                    for _ in range(nslices):
                        max_len = max(self.cfg.min_seg_len, L // 4)
                        seg_len = random.randint(self.cfg.min_seg_len, max_len)
                        if seg_len >= L:
                            seg_len = L - 1
                        start = random.randint(0, L - seg_len)
                        end = start + seg_len
                        x[start:end] = xm[start:end]
                        y[start:end] = ym[start:end]

                # Intentionally introduce varied padding within the fixed-length window
                pad_tail_prob = getattr(self.cfg, "pad_tail_prob", 0.6)
                if random.random() < pad_tail_prob:
                    max_frac = getattr(self.cfg, "pad_tail_max_frac", 0.9)
                    max_pad = max(1, int(L * max_frac))
                    pad_len = random.randint(0, max_pad)
                    if pad_len > 0:
                        content_end = max(1, L - pad_len)
                        x[content_end:] = PAD_BYTE_ID
                        y[content_end:] = PAD_ID

                xb[i], yb[i] = x, y

            try:
                self.q.put((xb, yb), timeout=1.0)
            except queue.Full:
                continue

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
