import numpy as np

import config as cfg

VISIBLE_ASCII_BYTES = tuple(range(0x20, 0x7F))
WHITESPACE_BYTES = (0x09, 0x0A, 0x0D)
CURRENCY_BYTE_ID = np.int32(0xA4)

_ALLOWED_BYTE_VALUES = np.array(
    sorted(set(VISIBLE_ASCII_BYTES) | set(WHITESPACE_BYTES) | {int(CURRENCY_BYTE_ID)}),
    dtype=np.int32,
)
_ALLOWED_TOKEN_VALUES = np.array(
    sorted(set(_ALLOWED_BYTE_VALUES.tolist()) | {int(cfg.PAD_BYTE_ID)}),
    dtype=np.int32,
)


def sanitize_tokens(tokens: np.ndarray) -> np.ndarray:
    """Map disallowed token IDs to the currency sign (¤) and return the sanitized array."""
    arr = np.asarray(tokens, dtype=np.int32)
    if arr.size == 0:
        return arr
    invalid = ~np.isin(arr, _ALLOWED_TOKEN_VALUES)
    if np.any(invalid):
        arr[invalid] = CURRENCY_BYTE_ID
    return arr


def sanitize_bytes(byte_arr: np.ndarray) -> np.ndarray:
    """Return a copy of byte_arr with unsupported bytes mapped to the currency sign (¤)."""
    arr = np.asarray(byte_arr, dtype=np.uint8)
    if arr.size == 0:
        return arr
    invalid = ~np.isin(arr.astype(np.int32), _ALLOWED_BYTE_VALUES)
    if np.any(invalid):
        arr = arr.copy()
        arr[invalid] = np.uint8(CURRENCY_BYTE_ID)
    return arr


def allowed_token_values() -> np.ndarray:
    return _ALLOWED_TOKEN_VALUES.copy()


def allowed_byte_values() -> np.ndarray:
    return _ALLOWED_BYTE_VALUES.copy()
