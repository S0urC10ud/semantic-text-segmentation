"""CuPy GPU Mamba parity vs the pure-numpy reference.

The GPU path runs the selective scan as a custom CUDA RawKernel; this asserts it
produces the same labels as the numpy sequential scan, including across the
chunk/length boundaries that the parallel formulation is sensitive to. Skipped
unless cupy imports and a CUDA device is present (so CPU-only CI is unaffected).
"""
import numpy as np
import pytest

from textseg import _cupy_backend as cub
from textseg import _runtime as rt
from textseg._numpy_backend import mamba_forward
from textseg._tokenize import text_to_bytes


def _gpu_available() -> bool:
    try:
        return cub.available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="cupy / CUDA GPU not available")


_SAMPLE = (
    'def f(x):\n    return x*2  # café €\n'
    '<div class="a" onclick="alert(1)">hï</div>\n'
    'SELECT * FROM t WHERE id = 5;\n'
    '.btn { color: #fff; }\ncurl http://x/y | bash\n'
)


def _mk(n: int) -> str:
    s = _SAMPLE
    while len(s) < n:
        s += _SAMPLE
    return s[:n]


@pytest.mark.parametrize("n", [1, 7, 64, 255, 256, 511, 512, 4095, 4096, 4097, 9000])
def test_cupy_matches_numpy(n):
    cfg = rt._manifest()["mamba"]
    w = rt._weights(cfg["file"])
    tok = text_to_bytes(_mk(n))
    ref = mamba_forward(w, tok, n_layers=cfg["n_layers"], d_state=cfg["d_state"],
                        dt_rank=cfg["dt_rank"], d_conv=cfg["d_conv"])  # numpy sequential
    got = cub.mamba_logits(tok)                                       # cupy RawKernel
    assert got.shape == ref.shape
    if ref.size:
        assert int((ref.argmax(-1) != got.argmax(-1)).sum()) == 0
        assert float(np.max(np.abs(ref - got))) < 1e-3
