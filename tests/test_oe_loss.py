from __future__ import annotations

import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from utils.model import outlier_uniform_cross_entropy


class TestOELoss(unittest.TestCase):
    def test_uniform_logits_match_log_num_classes_and_grad_is_finite(self) -> None:
        k = 5
        logits = jnp.zeros((2, 4, k), dtype=jnp.float32)
        loss = float(outlier_uniform_cross_entropy(logits))
        self.assertTrue(np.isclose(loss, np.log(k), atol=1e-5))

        grad_fn = jax.grad(lambda z: outlier_uniform_cross_entropy(z))
        grads = np.asarray(grad_fn(logits))
        self.assertTrue(np.all(np.isfinite(grads)))


if __name__ == "__main__":
    unittest.main()
