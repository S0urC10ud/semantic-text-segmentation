from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg
from train.utils.monitor_eval import FILE_DTYPE, SEG_DTYPE, evaluate_monitor_set


def _monitor_data() -> dict:
    label_id = 0
    file_len = 32
    n_files = 8
    files = np.zeros(n_files, dtype=FILE_DTYPE)
    segments = np.zeros(n_files, dtype=SEG_DTYPE)
    chunks = []
    offset = 0
    for i in range(n_files):
        arr = np.arange(i * 31, i * 31 + file_len, dtype=np.uint8)
        chunks.append(arr)
        files[i] = (offset, file_len, i, 1, 0, label_id)
        segments[i] = (i, 0, file_len, label_id)
        offset += file_len
    contents = np.concatenate(chunks, axis=0)
    return {
        "meta": {"num_files": n_files},
        "files": files,
        "segments": segments,
        "contents": contents,
    }


def _eval_step_fn(state, batch_tokens, batch_labels, rng):
    del state, batch_labels, rng
    return jnp.mean(batch_tokens.astype(jnp.float32)), jnp.array(0.0, dtype=jnp.float32)


def _forward_logits(state, x, rng):
    del state, rng
    return jnp.zeros((x.shape[0], x.shape[1], cfg.NUM_CLASSES), dtype=jnp.float32)


def _eval_step_with_logits_fn(state, batch_tokens, batch_labels, rng):
    del state, batch_labels, rng
    logits = jnp.zeros((batch_tokens.shape[0], batch_tokens.shape[1], cfg.NUM_CLASSES), dtype=jnp.float32)
    return jnp.mean(batch_tokens.astype(jnp.float32)), jnp.array(0.0, dtype=jnp.float32), logits


class TestMonitorEvalDeterminism(unittest.TestCase):
    def test_deterministic_monitor_eval_is_independent_of_input_rng(self) -> None:
        monitor_data = _monitor_data()
        with patch("train.utils.monitor_eval._forward_logits", side_effect=_forward_logits):
            stats_a = evaluate_monitor_set(
                state=None,
                monitor_data=monitor_data,
                L=8,
                batch_size=2,
                rng=jax.random.PRNGKey(0),
                limit=4,
                eval_step_fn=_eval_step_fn,
                deterministic=True,
                deterministic_seed=17,
            )
            stats_b = evaluate_monitor_set(
                state=None,
                monitor_data=monitor_data,
                L=8,
                batch_size=2,
                rng=jax.random.PRNGKey(999),
                limit=4,
                eval_step_fn=_eval_step_fn,
                deterministic=True,
                deterministic_seed=17,
            )

        self.assertEqual(stats_a["files_used"], 4)
        self.assertEqual(stats_a["windows"], 4)
        self.assertEqual(stats_a["deterministic"], True)
        self.assertEqual(stats_a["deterministic_seed"], 17)
        self.assertEqual(stats_a["loss_mean"], stats_b["loss_mean"])
        np.testing.assert_array_equal(stats_a["conf_mat"], stats_b["conf_mat"])

    def test_deterministic_monitor_eval_seed_controls_fixed_subset(self) -> None:
        monitor_data = _monitor_data()
        with patch("train.utils.monitor_eval._forward_logits", side_effect=_forward_logits):
            stats_a = evaluate_monitor_set(
                state=None,
                monitor_data=monitor_data,
                L=8,
                batch_size=2,
                rng=jax.random.PRNGKey(0),
                limit=4,
                eval_step_fn=_eval_step_fn,
                deterministic=True,
                deterministic_seed=17,
            )
            stats_b = evaluate_monitor_set(
                state=None,
                monitor_data=monitor_data,
                L=8,
                batch_size=2,
                rng=jax.random.PRNGKey(0),
                limit=4,
                eval_step_fn=_eval_step_fn,
                deterministic=True,
                deterministic_seed=23,
            )

        self.assertNotEqual(stats_a["loss_mean"], stats_b["loss_mean"])

    def test_monitor_eval_uses_single_pass_logits_fn_when_provided(self) -> None:
        monitor_data = _monitor_data()
        with patch(
            "train.utils.monitor_eval._forward_logits",
            side_effect=AssertionError("_forward_logits should not be called"),
        ):
            stats = evaluate_monitor_set(
                state=None,
                monitor_data=monitor_data,
                L=8,
                batch_size=4,
                rng=jax.random.PRNGKey(0),
                limit=4,
                eval_step_fn=_eval_step_fn,
                eval_step_with_logits_fn=_eval_step_with_logits_fn,
                deterministic=True,
                deterministic_seed=17,
            )

        self.assertEqual(stats["files_used"], 4)
        self.assertEqual(stats["windows"], 4)
        self.assertIsNotNone(stats["conf_mat"])


if __name__ == "__main__":
    unittest.main()
