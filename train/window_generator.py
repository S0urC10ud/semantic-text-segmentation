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
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

import numpy as np
import datasets as hfds

import config as cfg

if TYPE_CHECKING:
    from config import DataConfig
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


def _sample_mixed_segment_count(max_segments: int = 6, stop_prob: float = 0.6) -> int:
    """Geometric-like sampler favouring fewer segments when mixing windows."""
    count = 1
    while count < max_segments and random.random() > stop_prob:
        count += 1
    return count


def _choose_segment_slice(
    byte_content: np.ndarray,
    seg_len: int,
    min_letters: int = 4,
    guard_len: int = 64,
    max_attempts: int = 6,
) -> np.ndarray:
    """
    Pick a slice of ``byte_content`` up to ``seg_len`` bytes. For shorter slices we
    bias toward samples that contain at least ``min_letters`` ASCII alphabetic chars.
    """
    total = int(len(byte_content))
    if seg_len <= 0 or total == 0:
        return np.empty((0,), dtype=np.uint8)

    seg_len = min(seg_len, total)
    if seg_len > guard_len:
        if total == seg_len:
            return byte_content[:seg_len]
        start = random.randint(0, total - seg_len)
        return byte_content[start:start + seg_len]

    best_slice = byte_content[:seg_len]
    best_letters = -1
    for _ in range(max_attempts):
        if total == seg_len:
            start = 0
        else:
            start = random.randint(0, total - seg_len)
        segment = byte_content[start:start + seg_len]
        text = segment.tobytes().decode("utf-8", "ignore")
        letters = sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text)
        if letters >= min_letters:
            return segment
        if letters > best_letters:
            best_slice = segment
            best_letters = letters
    return best_slice
# ---------------------------
# Window builders (supports partial fill + masking)
# ---------------------------

def make_pure_window(dsets_by_lang: Dict[int, hfds.Dataset],
                     target_len: int) -> Tuple[np.ndarray, np.ndarray]:
    lids = list(dsets_by_lang.keys())
    x = np.full((target_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    y = np.full((target_len,), cfg.PAD_ID, dtype=np.uint8)
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
    x_buf = np.full((target_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    y_buf = np.full((target_len,), cfg.PAD_ID, dtype=np.uint8)

    lids = list(dsets_by_lang.keys())
    if not lids:
        return x_buf, y_buf

    nsegs = _sample_mixed_segment_count()
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
        segment = _choose_segment_slice(b, seg_len)
        L = min(seg_len, len(segment))
        if L > 0:
            x_buf[pos:pos + L] = segment[:L].astype(np.int32)
            y_buf[pos:pos + L] = lid

        pos += seg_len
        left -= seg_len

    return x_buf, y_buf


# ---------------------------
# Shared sampler for train/eval/preview
# ---------------------------

def _resolve_mix_probability(data_cfg: "DataConfig") -> float:
    mix_prob = getattr(data_cfg, "mix_prob", None)
    pure_prob = max(0.0, getattr(data_cfg, "pure_prob", 0.0))
    line_prob = max(0.0, getattr(data_cfg, "line_inject_prob", 0.0))
    if mix_prob is None:
        mix_prob = 1.0 - pure_prob - line_prob
    return max(0.0, mix_prob)


def _choose_window_mode(data_cfg: "DataConfig") -> str:
    pure_prob = max(0.0, getattr(data_cfg, "pure_prob", 0.0))
    line_prob = max(0.0, getattr(data_cfg, "line_inject_prob", 0.0))
    mix_prob = _resolve_mix_probability(data_cfg)
    total = pure_prob + line_prob + mix_prob
    if total <= 0.0:
        return "mixed"

    r = random.random() * total
    if r < pure_prob:
        return "pure"
    if r < pure_prob + line_prob:
        return "line_inject"
    return "mixed"


def make_training_window(
    dsets_by_lang: Dict[int, hfds.Dataset],
    target_len: int,
    data_cfg: "DataConfig",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw a window using the same probability distribution as the training
    prefetchers, including mixed windows, line injection, optional overlays,
    and tail padding variation.
    """
    mode = _choose_window_mode(data_cfg)
    if mode == "pure":
        x, y = make_pure_window(dsets_by_lang, target_len)
    elif mode == "line_inject":
        x, y = make_line_injected_window(dsets_by_lang, target_len, data_cfg)
    else:
        x, y = make_mixed_window(dsets_by_lang, target_len, data_cfg.min_seg_len)

    # Optionally overlay mixed slices on top of base window
    both_prob = getattr(data_cfg, "both_prob", 0.2)
    if both_prob > 0.0 and random.random() < both_prob:
        xm, ym = make_mixed_window(dsets_by_lang, target_len, data_cfg.min_seg_len)
        nslices = random.randint(1, 3)
        for _ in range(nslices):
            max_len = max(data_cfg.min_seg_len, target_len // 4)
            seg_len = random.randint(data_cfg.min_seg_len, max_len)
            if seg_len >= target_len:
                seg_len = target_len - 1
            start = random.randint(0, target_len - seg_len)
            end = start + seg_len
            x[start:end] = xm[start:end]
            y[start:end] = ym[start:end]

    # Tail padding variation mirrors training augmentation
    pad_tail_prob = getattr(data_cfg, "pad_tail_prob", 0.6)
    if pad_tail_prob > 0.0 and random.random() < pad_tail_prob:
        max_frac = getattr(data_cfg, "pad_tail_max_frac", 0.9)
        max_pad = max(1, int(target_len * max_frac))
        pad_len = random.randint(0, max_pad)
        if pad_len > 0:
            content_end = max(1, target_len - pad_len)
            x[content_end:] = cfg.PAD_BYTE_ID
            y[content_end:] = cfg.PAD_ID

    return x, y


# ---------------------------
# LINE-LEVEL INJECTION
# ---------------------------

def _split_keepends_lines(text: str) -> List[str]:
    return text.splitlines(keepends=True) if text else []


def _count_letters(text: str) -> int:
    return sum(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text)


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


def _prepare_donor_block(donor_text: str, data_cfg: "DataConfig", insertion_indent: str) -> str:
    lines = _split_keepends_lines(donor_text)
    if not lines:
        return ""

    skip = random.randint(data_cfg.donor_skip_top_min, max(data_cfg.donor_skip_top_min, data_cfg.donor_skip_top_max))
    start_idx = min(skip, len(lines) - 1)

    L = _sample_truncated_exp_lines(data_cfg.line_inject_exp_rate, data_cfg.line_inject_max_lines)
    end_idx = min(len(lines), start_idx + L)
    pick = lines[start_idx:end_idx]

    if len(pick) == 1 and len(pick[0]) < data_cfg.line_inject_min_single_len:
        if end_idx < len(lines):
            pick.append(lines[end_idx])
        elif start_idx > 0:
            pick.insert(0, lines[start_idx - 1])

    max_total_lines = min(len(lines), data_cfg.line_inject_max_lines)
    min_letters = getattr(data_cfg, "line_inject_min_letters", 4)

    def pick_letter_count() -> int:
        return sum(_count_letters(ln) for ln in pick)

    letter_count = pick_letter_count()
    while letter_count < min_letters and len(pick) < max_total_lines:
        expanded = False
        if end_idx < len(lines) and len(pick) < max_total_lines:
            pick.append(lines[end_idx])
            end_idx += 1
            expanded = True
            letter_count = pick_letter_count()
        if letter_count < min_letters and len(pick) < max_total_lines and start_idx > 0:
            start_idx -= 1
            pick.insert(0, lines[start_idx])
            expanded = True
            letter_count = pick_letter_count()
        if not expanded:
            break

    w_none, w_l, w_r, w_b = data_cfg.strip_weights
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

        if random.random() < data_cfg.reindent_prob:
            core = insertion_indent + core.lstrip()

        processed.append(core + ('\n' if ln.endswith('\n') else ''))

    block = "".join(processed)
    if _count_letters(block) < min_letters:
        return ""
    return block


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
                              data_cfg: "DataConfig") -> Tuple[np.ndarray, np.ndarray]:
    lids = list(dsets_by_lang.keys())
    x = np.full((target_len,), cfg.PAD_BYTE_ID, dtype=np.int32)
    y = np.full((target_len,), cfg.PAD_ID, dtype=np.uint8)
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
        host_text, data_cfg.host_skip_top_min, data_cfg.host_skip_top_max, data_cfg.line_inject_max_injections
    )

    for bidx in boundaries:
        donor_lid = random.choice(lids)
        if not data_cfg.allow_same_lang_injection and donor_lid == host_lid:
            alts = [l for l in lids if l != host_lid]
            if alts:
                donor_lid = random.choice(alts)

        donor_ds = dsets_by_lang[donor_lid]
        donor_ex = _random_example(donor_ds)
        donor_text = donor_ex.get("content", "") if donor_ex else ""

        insertion_indent = _leading_indent_of_line(chars, bidx)
        donor_block = _prepare_donor_block(donor_text, data_cfg, insertion_indent)

        if random.random() > data_cfg.start_with_newline_prob:
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
    def __init__(self, dsets_by_lang, data_cfg: "DataConfig"):
        self.dsets_by_lang = dsets_by_lang
        self.cfg = data_cfg
        # Smaller queue size to prevent memory buildup
        self.q = queue.Queue(maxsize=max(2, data_cfg.prefetch_batches))
        self.stop_flag = threading.Event()
        self.buckets = data_cfg.buckets()
        self.threads: List[threading.Thread] = []
        self._gc_counter = 0
        # Use fewer worker threads
        for wid in range(max(1, data_cfg.num_workers // 2)):
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

            xb = np.full((self.cfg.batch_size, L), cfg.PAD_BYTE_ID, dtype=np.int32)
            yb = np.full((self.cfg.batch_size, L), cfg.PAD_ID, dtype=np.uint8)

            for i in range(self.cfg.batch_size):
                x, y = make_training_window(self.dsets_by_lang, L, self.cfg)
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
