from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from active_learning.label_store import LabelStore, StoredRefinement


class TestActiveLearningCorrectness(unittest.TestCase):
    def test_label_mapping_and_padding(self) -> None:
        """
        Verify that stored refinements are correctly mapped to windows,
        with proper padding applied to unrefined areas.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "al.sqlite"
            store = LabelStore(db_path)

            text = "abcdefghijklmnopqrstuvwxyz" # 26 chars
            
            store.add(
                StoredRefinement(
                    round_id="r1",
                    source_split="train",
                    source_lang="python",
                    sample_index=0,
                    sample_hash="hash_a",
                    boundary_index=10,
                    snippet_start=0,
                    snippet_end=26,
                    snippet_text=text,
                    oracle_name="stub",
                    oracle_model="stub",
                    oracle_run_id="",
                    status="ok",
                    acquisition_score=1.0,
                    predicted_segments=[],
                    refined_segments=[
                        # 0-5 is class A
                        {"start": 0, "end": 5, "label": "class_a"},
                        # 5-10 is class B
                        {"start": 5, "end": 10, "label": "class_b"},
                        # 10-15 is unknown (open set)
                        {"start": 10, "end": 15, "label": "other"},
                        # 15-26 is unmapped/omitted implicitly to test padding behavior,
                        # but in reality the whole 26 chars are returned as refined segments.
                        # Let's say only 0-15 were refined, the rest should be padding.
                    ],
                    metadata={},
                )
            )

            label_to_id = {
                "class_a": 0,
                "class_b": 1,
                "other": 2,
            }
            pad_byte_id = 0
            pad_label_id = 255
            window_bytes = 26

            windows = store.build_training_windows(
                window_bytes=window_bytes,
                pad_byte_id=pad_byte_id,
                pad_label_id=pad_label_id,
                label_to_id=label_to_id,
                max_windows=10,
                fallback_label="other",
            )

            self.assertEqual(len(windows), 1)
            x, y = windows[0]

            self.assertEqual(x.shape, (window_bytes,))
            self.assertEqual(y.shape, (window_bytes,))
            
            # Check characters (ASCII)
            expected_x = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
            np.testing.assert_array_equal(x, expected_x)
            
            # Check labels
            expected_y = np.full((window_bytes,), pad_label_id, dtype=np.uint8)
            expected_y[0:5] = 0   # class_a
            expected_y[5:10] = 1  # class_b
            expected_y[10:15] = 2 # other (open set / fallback)
            # 15:26 remains 255 (pad_label_id)

            np.testing.assert_array_equal(y, expected_y)

    def test_gradient_flow_with_ignore_index(self) -> None:
        """
        Verify that cross entropy loss with ignore_index correctly masks out
        gradients for unrefined (padded) tokens, ensuring active learning
        only trains on refined areas.
        """
        batch_size = 2
        seq_len = 10
        num_classes = 3
        pad_label_id = 255

        # Create dummy logits (B, L, C) and require gradients.
        key = jax.random.PRNGKey(0)
        logits = jax.random.normal(key, (batch_size, seq_len, num_classes))

        # Create target labels (B, L)
        targets = np.full((batch_size, seq_len), pad_label_id, dtype=np.int32)
        
        # Add some refined labels
        # Batch 0: first 3 tokens are class 1
        targets[0, 0:3] = 1
        # Batch 1: middle 2 tokens are class 0
        targets[1, 4:6] = 0
        
        targets_jnp = jnp.array(targets)

        def loss_fn(logits, targets):
            # Compute cross entropy loss per token
            # optax.softmax_cross_entropy_with_integer_labels expects logits of shape (..., C)
            # and labels of shape (...)
            # Since targets has ignore_index, we mask them out.
            # Using the same logic as in test_masked_loss.py
            valid_mask = targets != pad_label_id
            
            # For optax, we must ensure targets are valid range even if masked out, 
            # otherwise it might error or return NaN.
            safe_targets = jnp.where(valid_mask, targets, 0)
            
            per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=safe_targets
            )
            
            masked_loss = per_token_loss * valid_mask
            
            # Return sum for gradient calculation (or mean over valid_count)
            # Sum is simpler to verify zero gradients exactly
            return jnp.sum(masked_loss)

        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(logits, targets_jnp)

        self.assertIsNotNone(grads)
        
        # Gradients for padded regions should be exactly 0
        self.assertTrue(jnp.all(grads[0, 3:, :] == 0.0), "Gradients should be zero for ignored index")
        
        # Gradients for refined regions should be non-zero
        self.assertTrue(jnp.any(grads[0, 0:3, :] != 0.0), "Gradients should be non-zero for target labels")
        
        # Batch 1
        self.assertTrue(jnp.all(grads[1, 0:4, :] == 0.0), "Gradients should be zero for ignored index")
        self.assertTrue(jnp.any(grads[1, 4:6, :] != 0.0), "Gradients should be non-zero for target labels")
        self.assertTrue(jnp.all(grads[1, 6:, :] == 0.0), "Gradients should be zero for ignored index")

if __name__ == "__main__":
    unittest.main()
