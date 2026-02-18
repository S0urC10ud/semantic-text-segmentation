from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg
from utils.model import _supervision_mask, cross_entropy_masked


class TestMaskedLoss(unittest.TestCase):
    def test_masked_loss_ignores_configured_positions(self) -> None:
        # Shape: [batch=1, length=6, classes=3]
        logits_a = jnp.array(
            [
                [
                    [4.0, 0.1, 0.1],  # counted
                    [0.1, 4.0, 0.1],  # ignored (space token)
                    [0.1, 4.0, 0.1],  # ignored (newline token)
                    [0.1, 0.1, 4.0],  # counted
                    [4.0, 0.1, 0.1],  # ignored (tab token)
                    [4.0, 0.1, 0.1],  # counted
                ]
            ],
            dtype=jnp.float32,
        )
        # Same as logits_a except ignored positions are heavily perturbed.
        logits_b = logits_a.at[0, 1, :].set(jnp.array([90.0, -90.0, 0.0], dtype=jnp.float32))
        logits_b = logits_b.at[0, 2, :].set(jnp.array([-60.0, 60.0, 0.0], dtype=jnp.float32))
        logits_b = logits_b.at[0, 4, :].set(jnp.array([0.0, -75.0, 75.0], dtype=jnp.float32))

        labels = jnp.array([[0, 1, 1, 2, 0, 0]], dtype=jnp.uint8)
        tokens = jnp.array([[65, ord(" "), ord("\n"), 66, ord("\t"), 67]], dtype=jnp.int32)

        mask = _supervision_mask(labels, tokens, pad_id=cfg.PAD_ID)
        self.assertEqual(mask.tolist(), [[True, False, False, True, False, True]])

        loss_a = float(cross_entropy_masked(logits_a, labels, tokens=tokens, pad_id=cfg.PAD_ID))
        loss_b = float(cross_entropy_masked(logits_b, labels, tokens=tokens, pad_id=cfg.PAD_ID))
        self.assertTrue(np.isclose(loss_a, loss_b, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
