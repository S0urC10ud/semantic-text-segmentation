"""The exposed per-character distribution (`Segmentation.char_probs`) must be a
well-formed probability matrix over `Segmentation.labels`, and consistent with the
per-character labels/confidence and segment spans. Runs both models on CPU.
"""
import numpy as np
import pytest

import typeseg


TEXTS = [
    "def f(): return 1",
    "<html><body>cG93ZXJzaGVsbA==</body></html>",
    "SELECT * FROM t WHERE x = 1; -- café ☕ unicode",
    "",
]


@pytest.mark.parametrize("fn", [typeseg.fast, typeseg.precise])
@pytest.mark.parametrize("text", TEXTS)
def test_char_probs_is_a_valid_distribution(fn, text):
    r = fn(text)
    n, c = len(text), len(r.labels)

    # Shape, dtype, column order.
    assert c == 35
    assert r.char_probs is not None
    assert r.char_probs.shape == (n, c)
    assert r.char_probs.dtype == np.float32

    if n == 0:
        return

    # Valid probabilities: in [0, 1] and each row sums to ~1.
    assert np.all(r.char_probs >= -1e-6)
    assert np.all(r.char_probs <= 1 + 1e-4)
    np.testing.assert_allclose(r.char_probs.sum(axis=1), 1.0, atol=1e-3)

    # Lengths line up with the other per-character views.
    assert len(r.char_labels) == n
    assert len(r.char_confidence) == n

    # Every per-character label is a known class or the virtual open-set label.
    known = set(r.labels) | {"other"}
    assert set(r.char_labels) <= known

    # char_distribution() is a convenience view over the same numbers.
    d = r.char_distribution(0)
    assert set(d) == set(r.labels)
    np.testing.assert_allclose(sum(d.values()), 1.0, atol=1e-3)


def test_confidence_matches_distribution_top_prob():
    """For non-`other` characters, char_confidence is the probability the model
    assigned to the (post-processed) label in char_probs."""
    r = typeseg.fast("def square(x): return x * x")
    probs, labels = r.char_probs, r.labels
    idx = {name: i for i, name in enumerate(labels)}
    checked = 0
    for i, lab in enumerate(r.char_labels):
        if lab == "other":
            continue
        assert abs(r.char_confidence[i] - float(probs[i, idx[lab]])) < 1e-5
        checked += 1
    assert checked > 0


def test_segment_text_matches_span():
    r = typeseg.precise("print('hi'); SELECT 1;")
    for s in r.segments:
        assert s.text == r.text[s.start:s.end]
