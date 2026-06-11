"""Weight loading, windowed inference, and result assembly."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import _cupy_backend as cub
from . import _numpy_backend as nb
from . import _onnx_backend as ob
from ._numpy_backend import Weights, mamba_forward, unet_forward
from ._options import Options
from ._postprocess import (
    build_segments,
    confidence_gate,
    normalize_short_runs,
    paired_delimiter_fill,
    relabel_whitespace,
    snap_boundaries,
)
from ._segmentation import Segmentation
from ._tokenize import PAD_BYTE_ID, byte_probs_to_char, text_to_bytes

try:  # Python 3.9+
    from importlib.resources import files as _files
except ImportError:  # pragma: no cover
    from importlib_resources import files as _files  # type: ignore

DEFAULT_CHUNK = 1536


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads((_files("typeseg") / "data" / "manifest.json").read_text())


@lru_cache(maxsize=4)
def _weights(npz_name: str) -> Weights:
    with (_files("typeseg") / "data" / npz_name).open("rb") as fh:
        data = np.load(fh)
        flat = {k.replace("__", "/"): np.asarray(data[k], dtype=np.float32) for k in data.files}
    return Weights(flat)


def _window_spans(length: int, chunk: int) -> List[Tuple[int, int]]:
    if length <= 0:
        return []
    win = max(64, chunk)
    if length <= win:
        return [(0, length)]
    stride = max(1, win // 2)
    spans, start = [], 0
    while True:
        end = min(start + win, length)
        spans.append((start, end))
        if end >= length:
            break
        start += stride
    return spans


def _window_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(length, 0),), dtype=np.float32)
    pos = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
    return np.exp(-0.5 * (pos / 0.5) ** 2).astype(np.float32)


def _unet_windows(byte_tokens: np.ndarray, spans) -> Tuple[np.ndarray, List[int]]:
    """Stack each span into a fixed-width (1536) PAD-filled row.

    The U-Net pools 7x; every window runs at the fixed model width (padding short
    windows with PAD) so they don't collapse, then output is sliced back. Mirrors
    the viewer."""
    rows = np.full((len(spans), DEFAULT_CHUNK), PAD_BYTE_ID, dtype=np.int64)
    lens: List[int] = []
    for i, (s, e) in enumerate(spans):
        win_len = e - s
        rows[i, :win_len] = byte_tokens[s:e]
        lens.append(win_len)
    return rows, lens


def _unet_byte_probs(byte_tokens: np.ndarray, channels, num_classes: int) -> np.ndarray:
    length = int(byte_tokens.shape[0])
    spans = _window_spans(length, DEFAULT_CHUNK)
    rows, _lens = _unet_windows(byte_tokens, spans)
    if ob.available():
        logits_all = ob.unet_window_logits(rows)                       # (N, 1536, C)
    else:
        w = _weights(_manifest()["unet"]["file"])
        logits_all = np.stack(
            [unet_forward(w, rows[i], channels=tuple(channels)) for i in range(rows.shape[0])],
            axis=0,
        )
    accum = np.zeros((length, num_classes), dtype=np.float32)
    wsum = np.zeros((length,), dtype=np.float32)
    for i, (s, e) in enumerate(spans):
        win_len = e - s
        probs = nb._softmax(logits_all[i, :win_len], axis=-1)
        wt = _window_weights(win_len)
        accum[s:e] += probs * wt[:, None]
        wsum[s:e] += wt
    wsum = np.where(wsum <= 0, 1.0, wsum)
    return accum / wsum[:, None]


def _mamba_byte_probs(byte_tokens: np.ndarray, cfg: dict) -> np.ndarray:
    # CuPy GPU (parallel scan) > ONNX CPU > pure numpy. The ONNX `Scan` op is
    # slower on GPU than CPU, so GPU acceleration for Mamba comes via CuPy.
    if cub.available():
        logits = cub.mamba_logits(byte_tokens)
    elif ob.available():
        logits = ob.mamba_logits(byte_tokens)
    else:
        w = _weights(_manifest()["mamba"]["file"])
        logits = mamba_forward(w, byte_tokens, n_layers=cfg["n_layers"], d_state=cfg["d_state"],
                               dt_rank=cfg["dt_rank"], d_conv=cfg["d_conv"])
    return nb._softmax(logits, axis=-1)


def _assemble(text: str, byte_probs: np.ndarray, options: Options) -> Segmentation:
    man = _manifest()
    labels_names: List[str] = man["labels"]
    other_index = man["other_index"]
    other_label = man["other_label"]

    char_probs, char_labels = byte_probs_to_char(text, byte_probs)
    labels = [int(x) for x in char_labels]

    # Order follows the thesis four steps (whitespace -> confidence -> boundary
    # snap -> min-run); the extra paired-delimiter fill runs last, after min-run
    # has reabsorbed short low-confidence runs that would otherwise split a
    # delimiter-wrapped region.
    if options.whitespace_relabel:
        labels = relabel_whitespace(text, labels)
    if options.confidence_gating:
        labels = confidence_gate(char_probs, labels, options.other_threshold, other_index)
    if options.boundary_snap:
        labels = snap_boundaries(text, labels, char_probs, options.boundary_snap_max_shift,
                                 options.min_run_chars)
    if options.min_run_normalize:
        labels = normalize_short_runs(labels, char_probs, options.min_run_chars, other_index)
    if options.paired_delimiter_fill:
        labels = paired_delimiter_fill(text, labels, char_probs, options.paired_delimiter_max_shift)
    if options.whitespace_relabel:
        labels = relabel_whitespace(text, labels)

    segments, char_label_names, char_conf = build_segments(
        text, labels, char_probs, labels_names, other_index, other_label
    )
    return Segmentation(text=text, segments=segments, char_labels=char_label_names,
                        char_confidence=char_conf, char_probs=char_probs, labels=labels_names)


def _empty(text: str) -> Segmentation:
    man = _manifest()
    char_probs = np.zeros((0, int(man["num_classes"])), dtype=np.float32)
    return Segmentation(text=text, segments=[], char_labels=[], char_confidence=[],
                        char_probs=char_probs, labels=list(man["labels"]))


def run(model: str, text: str, options: Optional[Options]) -> Segmentation:
    if options is None:
        options = Options()
    if not text:
        return _empty(text)
    byte_tokens = text_to_bytes(text)
    if byte_tokens.size == 0:
        return _empty(text)
    man = _manifest()
    num_classes = man["num_classes"]
    if model == "fast":
        byte_probs = _unet_byte_probs(byte_tokens, man["unet"]["channels"], num_classes)
    else:
        byte_probs = _mamba_byte_probs(byte_tokens, man["mamba"])
    return _assemble(text, byte_probs, options)


def backend_info() -> dict:
    """Report the active inference backend (for diagnostics).

    Providers reflect what each model's session *actually loaded* — a CUDA provider
    that fails to initialise (and falls back to CPU) is reported as CPU. ``fast``
    (U-Net) may run on CUDA while ``precise`` (Mamba) stays on CPU, because the
    Mamba selective-scan is slower on GPU than CPU.
    """
    # fast() = U-Net (ONNX, CUDA-capable). precise() = Mamba: CuPy GPU > ONNX CPU > numpy.
    fast_p = ob.active_providers("unet_al.onnx") if ob.available() else []
    backend = "onnx" if fast_p or ob.available() else "numpy"
    try:
        cupy_on = cub.available()
    except Exception:
        cupy_on = False
    if cupy_on:
        precise_p = cub.active_providers()
    elif ob.available():
        precise_p = ob.active_providers("mamba_al.onnx")
    else:
        precise_p = []
    return {
        "backend": backend,
        "gpu": ("CUDAExecutionProvider" in fast_p) or cupy_on,
        "fast_providers": fast_p,
        "precise_providers": precise_p,
        "precise_gpu": cupy_on,
    }
