"""Guard against post-processing performance regressions.

The boundary-snap, minimum-run, and byte->char steps must stay (near-)linear in
the input length. They previously contained O(n^2) restart-after-each-change
loops that made large inputs pathologically slow; this test fails if that creeps
back in. Pure numpy, no model load.
"""
import time

import numpy as np

from typeseg import _postprocess as pp
from typeseg._tokenize import byte_probs_to_char


def _synthetic(n_blocks):
    """Heavily segmented labels + matching probs (worst case for the run loops)."""
    rng = np.random.default_rng(0)
    labels, probs = [], []
    for b in range(n_blocks):
        lab = b % 5
        run_len = 1 + (b % 4)  # lots of short interior runs
        labels += [lab] * run_len
        row = np.full(6, 0.05, dtype=np.float32)
        row[lab] = 0.75
        probs += [row] * run_len
    text = "x" * len(labels)
    return text, labels, np.array(probs, dtype=np.float32)


def _time_postproc(n_blocks):
    text, labels, probs = _synthetic(n_blocks)
    t = time.perf_counter()
    out = pp.snap_boundaries(text, list(labels), probs, 2, 3)
    out = pp.normalize_short_runs(out, probs, 3)
    out = pp.paired_delimiter_fill(text, out, probs, 2)
    return time.perf_counter() - t, len(labels)


def test_postprocess_scales_linearly():
    # Warm up (import/JIT-ish) then compare n and ~4n.
    _time_postproc(2000)
    t1, n1 = _time_postproc(4000)
    t4, n4 = _time_postproc(16000)
    # 4x the input should be well under 16x the time if it is ~linear; allow
    # generous slack for noise but still catch quadratic blow-up (>=16x).
    ratio = (t4 + 1e-6) / (t1 + 1e-6)
    assert ratio < 10.0, f"post-processing scaling looks super-linear: {n1}->{t1:.3f}s, {n4}->{t4:.3f}s (x{ratio:.1f})"


def test_large_input_postproc_is_quick():
    t, n = _time_postproc(40000)  # ~100k characters
    assert t < 5.0, f"post-processing on {n} chars took {t:.2f}s (expected < 5s)"


def test_byte_probs_to_char_ascii_fast_path():
    n = 50000
    rng = np.random.default_rng(1)
    bp = rng.random((n, 6), dtype=np.float32)
    text = "a" * n
    t = time.perf_counter()
    char_probs, char_labels = byte_probs_to_char(text, bp)
    dt = time.perf_counter() - t
    assert char_probs.shape == (n, 6)
    assert np.array_equal(char_labels, bp.argmax(axis=1))  # 1:1 for ASCII
    assert dt < 0.5, f"ASCII byte->char took {dt:.3f}s (expected fast vectorised path)"
