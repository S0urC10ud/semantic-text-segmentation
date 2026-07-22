"""Character-level byte tokenization (matches the training/inference pipeline)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

PAD_BYTE_ID = 256
CURRENCY_BYTE_ID = 0xA4  # 164, the non-ASCII / disallowed placeholder

# Allowed raw bytes: visible ASCII (0x20..0x7E), tab/newline/CR, and the placeholder.
_ALLOWED = sorted(set(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D} | {CURRENCY_BYTE_ID})
_ALLOWED_MASK = np.zeros(256, dtype=bool)
_ALLOWED_MASK[_ALLOWED] = True


def text_to_bytes(text: str) -> np.ndarray:
    """UTF-8 encode then sanitize: disallowed bytes map to the placeholder (0xA4)."""
    raw = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
    if raw.size == 0:
        return raw.astype(np.int32)
    out = raw.copy()
    bad = ~_ALLOWED_MASK[raw]
    if bad.any():
        out[bad] = CURRENCY_BYTE_ID
    return out.astype(np.int32)


def char_byte_lengths(text: str) -> List[int]:
    """Number of UTF-8 bytes contributed by each character."""
    return [len(ch.encode("utf-8", "ignore")) for ch in text]


def byte_probs_to_char(text: str, byte_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Average per-byte class probabilities over each character's bytes.

    Returns (char_probs (N, C), char_labels (N,)).
    """
    n = len(text)
    c = byte_probs.shape[1] if byte_probs.size else 0
    # Fast path: ASCII text is 1 byte per character, so per-byte probs already are
    # the per-character probs (no grouping). Covers the ASCII-restricted setting.
    if c and n and text.isascii() and byte_probs.shape[0] >= n:
        char_probs = np.ascontiguousarray(byte_probs[:n], dtype=np.float32)
        return char_probs, char_probs.argmax(axis=1)
    lengths = char_byte_lengths(text)
    char_probs = np.zeros((n, c), dtype=np.float32)
    pos = 0
    for i, L in enumerate(lengths):
        if L > 0 and pos < byte_probs.shape[0]:
            seg = byte_probs[pos:pos + L]
            if seg.shape[0] > 0:
                char_probs[i] = seg.mean(axis=0)
        pos += L
    char_labels = char_probs.argmax(axis=1) if c else np.zeros(n, dtype=np.int64)
    return char_probs, char_labels
