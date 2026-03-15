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
from utils.model import (
    _shift_sequence,
    _supervision_mask,
    auxiliary_neighbor_cross_entropy,
    merge_compatible_state,
    seed_missing_auxiliary_heads_from_main,
)


class TestAuxiliaryNeighborLoss(unittest.TestCase):
    def test_auxiliary_loss_ignores_out_of_bounds_and_ignored_targets(self) -> None:
        labels = jnp.array([[0, 1, 2, 1, 0, 2]], dtype=jnp.uint8)
        tokens = jnp.array([[65, ord(" "), 66, ord("\n"), 67, 68]], dtype=jnp.int32)
        logits_a = jnp.zeros((1, 6, len(cfg.AUX_NEIGHBOR_OFFSETS), 3), dtype=jnp.float32)

        target_mask = _supervision_mask(labels, tokens, pad_id=cfg.PAD_ID)
        source_mask = labels != cfg.PAD_ID

        logits_b = np.zeros_like(np.asarray(logits_a))
        for head_idx, offset in enumerate(cfg.AUX_NEIGHBOR_OFFSETS):
            shifted_mask = _shift_sequence(target_mask, offset, False)
            shifted_mask = jnp.logical_and(shifted_mask, source_mask)
            shifted_mask_np = np.asarray(shifted_mask)
            for pos in range(logits_b.shape[1]):
                if not shifted_mask_np[0, pos]:
                    logits_b[0, pos, head_idx, :] = np.array(
                        [25.0, -25.0, 5.0], dtype=np.float32
                    )

        loss_a = float(
            auxiliary_neighbor_cross_entropy(
                logits_a,
                labels,
                tokens=tokens,
                pad_id=cfg.PAD_ID,
            )
        )
        loss_b = float(
            auxiliary_neighbor_cross_entropy(
                jnp.array(logits_b),
                labels,
                tokens=tokens,
                pad_id=cfg.PAD_ID,
            )
        )
        self.assertTrue(np.isclose(loss_a, loss_b, atol=1e-8))


class TestCompatibleRestore(unittest.TestCase):
    def test_merge_compatible_state_keeps_new_leaves_initialized(self) -> None:
        target = {
            "dense": {
                "kernel": jnp.zeros((2, 3), dtype=jnp.float32),
                "bias": jnp.zeros((3,), dtype=jnp.float32),
            },
            "aux_logits_head": {
                "kernel": jnp.ones((2, 12), dtype=jnp.float32),
            },
        }
        source = {
            "dense": {
                "kernel": np.full((2, 3), 7.0, dtype=np.float32),
                "bias": np.full((3,), 5.0, dtype=np.float32),
            },
            "obsolete_head": {
                "kernel": np.full((2, 4), 9.0, dtype=np.float32),
            },
        }

        merged, stats = merge_compatible_state(target, source)

        self.assertTrue(np.allclose(np.asarray(merged["dense"]["kernel"]), 7.0))
        self.assertTrue(np.allclose(np.asarray(merged["dense"]["bias"]), 5.0))
        self.assertTrue(
            np.allclose(np.asarray(merged["aux_logits_head"]["kernel"]), 1.0)
        )
        self.assertIn(("aux_logits_head", "kernel"), stats["missing"])
        self.assertIn(("obsolete_head", "kernel"), stats["extra"])

    def test_missing_aux_head_is_seeded_from_center_head(self) -> None:
        target = {
            "Conv_0": {
                "kernel": np.arange(6, dtype=np.float32).reshape(1, 2, 3),
                "bias": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            },
            "aux_logits_head": {
                "kernel": np.zeros((1, 2, 12), dtype=np.float32),
                "bias": np.zeros((12,), dtype=np.float32),
            },
        }
        source = {
            "Conv_0": {
                "kernel": np.full((1, 2, 3), 7.0, dtype=np.float32),
                "bias": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            },
        }

        merged, _ = merge_compatible_state(target, source)
        seeded, note = seed_missing_auxiliary_heads_from_main(merged, source)

        expected_kernel = np.concatenate([source["Conv_0"]["kernel"]] * 4, axis=-1)
        expected_bias = np.concatenate([source["Conv_0"]["bias"]] * 4, axis=-1)
        self.assertTrue(np.allclose(np.asarray(seeded["aux_logits_head"]["kernel"]), expected_kernel))
        self.assertTrue(np.allclose(np.asarray(seeded["aux_logits_head"]["bias"]), expected_bias))
        self.assertIsNotNone(note)
        self.assertIn("initialized from Conv_0", note)

    def test_merge_compatible_state_preserves_empty_optimizer_slots(self) -> None:
        target = {
            "opt_state": (
                {},
                {"count": jnp.array(0, dtype=jnp.int32)},
            ),
        }
        source = {
            "opt_state": (
                {},
                {"count": np.array(7, dtype=np.int32)},
            ),
        }

        merged, stats = merge_compatible_state(target, source)

        self.assertEqual(len(merged["opt_state"]), 2)
        self.assertEqual(dict(merged["opt_state"][0]), {})
        self.assertEqual(int(np.asarray(merged["opt_state"][1]["count"])), 7)
        self.assertFalse(stats["missing"])
        self.assertFalse(stats["mismatched"])


if __name__ == "__main__":
    unittest.main()
