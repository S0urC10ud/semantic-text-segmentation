import numpy as np
import utils.config as cfg

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


# ---------------------------------------------------------------------------
# Compact vocabulary (slimmed checkpoints)
# ---------------------------------------------------------------------------
# Released checkpoints may carry a slimmed 130-row embedding instead of the full
# cfg.NUM_TOKEN_EMBEDDINGS (=257) table. Because every token is sanitized into
# {9, 10, 13, 32..126, 164 (=0xA4), 256 (=PAD)} before the lookup, only those
# rows are ever read. The compact layout keeps rows 0..127 as-is, folds the
# currency placeholder (0xA4 = 164) into row 128 and the pad (256) into row 129.
# COMPACT_TOKEN_TABLE maps the full 257-id space into the compact space and is
# applied inside the model right before the embedding lookup, so sanitization,
# padding and masking everywhere else keep operating in the original id space.
COMPACT_NUM_TOKEN_EMBEDDINGS = 130


def _build_compact_token_table() -> np.ndarray:
    table = np.empty(cfg.NUM_TOKEN_EMBEDDINGS, dtype=np.int32)  # indices 0..256
    table[:128] = np.arange(128, dtype=np.int32)
    table[128:256] = 128  # non-ASCII / disallowed bytes sanitize to 0xA4 -> row 128
    table[256] = 129  # PAD_BYTE_ID -> row 129
    return table


COMPACT_TOKEN_TABLE = _build_compact_token_table()


def peek_checkpoint_vocab_size(ckpt_path) -> int:
    """Embedding row count of a checkpoint (257 legacy, 130 slimmed)."""
    from pathlib import Path

    try:
        from flax import serialization

        p = Path(ckpt_path)
        if not p.is_file():
            return cfg.NUM_TOKEN_EMBEDDINGS
        tree = serialization.msgpack_restore(p.read_bytes())
        if isinstance(tree, dict) and "params" in tree and "Embed_0" not in tree:
            tree = tree["params"]
        rows = int(np.asarray(tree["Embed_0"]["embedding"]).shape[0])
        return rows if rows > 0 else cfg.NUM_TOKEN_EMBEDDINGS
    except Exception:
        return cfg.NUM_TOKEN_EMBEDDINGS
