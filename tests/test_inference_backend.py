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

import utils.config as cfg
from inference.backend import FastInferenceEngine, build_window_spans, window_weights
from inference.mamba_cuda import _selective_scan_associative, has_cuda_mamba_kernel, selective_scan_cuda
from train.utils.model import Mamba1D, UNet1D
from utils.token_utils import sanitize_bytes, sanitize_tokens


def _ascii_bytes(length: int, offset: int = 0) -> np.ndarray:
    alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789_-\n"
    data = bytearray()
    for idx in range(max(int(length), 0)):
        data.append(alphabet[(idx + offset) % len(alphabet)])
    return np.frombuffer(bytes(data), dtype=np.uint8)


def _legacy_segment_bytes_batch(
    *,
    apply_fn,
    num_classes: int,
    chunk_size: int,
    batch_size: int,
    byte_arrays,
):
    arrays = [sanitize_bytes(np.asarray(arr, dtype=np.uint8)) for arr in byte_arrays]
    probs_accum_list = [
        np.zeros((int(arr.shape[0]), int(num_classes)), dtype=np.float32)
        for arr in arrays
    ]
    weight_accum_list = [
        np.zeros((int(arr.shape[0]),), dtype=np.float32)
        for arr in arrays
    ]
    spans_by_text = []
    window_refs = []

    for text_idx, arr in enumerate(arrays):
        spans = build_window_spans(int(arr.shape[0]), int(chunk_size))
        spans_by_text.append(spans)
        for start, end in spans:
            window_refs.append((text_idx, int(start), int(end), arr[start:end]))

    if not window_refs:
        byte_labels = [np.zeros((int(arr.shape[0]),), dtype=np.uint8) for arr in arrays]
        return byte_labels, probs_accum_list, spans_by_text

    for i in range(0, len(window_refs), int(batch_size)):
        batch = window_refs[i:i + int(batch_size)]
        actual = len(batch)
        tokens = np.full((int(batch_size), int(chunk_size)), int(cfg.PAD_BYTE_ID), dtype=np.int32)
        for j, (_, _, _, window_bytes) in enumerate(batch):
            length = min(int(window_bytes.shape[0]), int(chunk_size))
            if length > 0:
                tokens[j, :length] = window_bytes[:length].astype(np.int32)
        tokens = sanitize_tokens(tokens)
        logits = apply_fn(jnp.asarray(tokens, dtype=jnp.int32))
        probs_batch = np.asarray(jax.device_get(jax.nn.softmax(logits, axis=-1)), dtype=np.float32)
        probs_batch = probs_batch[:actual, : int(chunk_size)]

        for j, (text_idx, start, end, _) in enumerate(batch):
            plen = int(end) - int(start)
            if plen <= 0:
                continue
            weights = window_weights(plen)
            window_probs = probs_batch[j, :plen]
            probs_accum_list[text_idx][start:end] += window_probs * weights[:, None]
            weight_accum_list[text_idx][start:end] += weights

    byte_labels = []
    normalized_probs = []
    for probs_accum, weight_accum in zip(probs_accum_list, weight_accum_list):
        if probs_accum.size <= 0:
            byte_labels.append(np.zeros((0,), dtype=np.uint8))
            normalized_probs.append(probs_accum)
            continue
        nonzero = weight_accum > 0
        if np.any(nonzero):
            normalized = probs_accum.copy()
            normalized[nonzero] /= weight_accum[nonzero, None]
            zero_mask = ~nonzero
            if np.any(zero_mask):
                normalized[zero_mask] = 1.0 / int(num_classes)
        else:
            normalized = np.full_like(probs_accum, 1.0 / int(num_classes), dtype=np.float32)
        normalized_probs.append(normalized)
        byte_labels.append(np.argmax(normalized, axis=-1).astype(np.uint8))
    return byte_labels, normalized_probs, spans_by_text


def _char_labels_for_ascii(byte_labels: np.ndarray) -> list[int]:
    return [int(x) for x in np.asarray(byte_labels, dtype=np.int32).tolist()]


def _build_model_apply(arch: str, num_classes: int):
    arch_name = str(arch).lower().strip()
    if arch_name == "mamba":
        model = Mamba1D(
            num_classes=int(num_classes),
            d_model=16,
            n_layers=1,
            d_state=4,
            expand=1,
            dt_rank=4,
            d_conv=2,
            bidirectional=True,
            dtype=jnp.float32,
        )
        meta = {
            "arch": "mamba",
            "model_dim": 16,
            "channels": (),
            "mamba_layers": 1,
            "mamba_d_state": 4,
            "mamba_expand": 1,
            "mamba_bidirectional": True,
        }
    else:
        model = UNet1D(
            num_classes=int(num_classes),
            emb_dim=16,
            channels=(16, 24),
            dropout_rate=0.0,
            dtype=jnp.float32,
        )
        meta = {
            "arch": "unet1d",
            "model_dim": 16,
            "channels": (16, 24),
            "mamba_layers": 1,
            "mamba_d_state": 4,
            "mamba_expand": 1,
            "mamba_bidirectional": True,
        }
    dummy_tokens = jnp.full((1, 8), int(cfg.PAD_BYTE_ID), dtype=jnp.int32)
    params = model.init({"params": jax.random.PRNGKey(0)}, dummy_tokens, train=False)["params"]
    apply_fn = jax.jit(lambda tok: model.apply({"params": params}, tok, train=False), backend="cpu")
    return apply_fn, meta


class TestInferenceBackend(unittest.TestCase):
    def _make_engine(
        self,
        *,
        apply_fn,
        meta,
        num_classes: int,
        chunk_size: int,
        batch_size: int,
        inference_backend: str = "fast",
        log_sink=None,
        full_memory_budget_bytes=None,
        execution_platform_override=None,
        cuda_kernel_available: bool = False,
    ) -> FastInferenceEngine:
        messages = log_sink if log_sink is not None else []
        return FastInferenceEngine(
            apply_tokens=apply_fn,
            num_classes=int(num_classes),
            pad_token_id=int(cfg.PAD_BYTE_ID),
            chunk_size=int(chunk_size),
            batch_size=int(batch_size),
            sanitize_bytes=sanitize_bytes,
            sanitize_tokens=sanitize_tokens,
            arch=str(meta["arch"]),
            inference_backend=str(inference_backend),
            actual_backend="cpu",
            log_fn=messages.append,
            model_dim=int(meta["model_dim"]),
            channels=tuple(meta["channels"]),
            mamba_layers=int(meta["mamba_layers"]),
            mamba_d_state=int(meta["mamba_d_state"]),
            mamba_expand=int(meta["mamba_expand"]),
            mamba_bidirectional=bool(meta["mamba_bidirectional"]),
            cuda_kernel_available=bool(cuda_kernel_available),
            full_memory_budget_bytes=full_memory_budget_bytes,
            execution_platform_override=execution_platform_override,
        )

    def test_fast_full_matches_legacy_for_chunk_sized_inputs(self) -> None:
        for arch in ("unet1d", "mamba"):
            apply_fn, meta = _build_model_apply(arch, num_classes=4)
            for chunk_size, offsets in ((64, (0, 7)), (96, (3, 19)), (128, (5, 11))):
                arrays = [_ascii_bytes(chunk_size, offset=offset) for offset in offsets]
                engine = self._make_engine(
                    apply_fn=apply_fn,
                    meta=meta,
                    num_classes=4,
                    chunk_size=chunk_size,
                    batch_size=2,
                )
                fast_labels, fast_probs, _ = engine.segment_bytes_batch(arrays)
                legacy_labels, legacy_probs, _ = _legacy_segment_bytes_batch(
                    apply_fn=apply_fn,
                    num_classes=4,
                    chunk_size=chunk_size,
                    batch_size=2,
                    byte_arrays=arrays,
                )
                for got, want in zip(fast_labels, legacy_labels):
                    np.testing.assert_array_equal(got, want)
                    self.assertEqual(_char_labels_for_ascii(got), _char_labels_for_ascii(want))
                for got, want in zip(fast_probs, legacy_probs):
                    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)

    def test_fast_stream_matches_legacy_for_over_chunk_inputs(self) -> None:
        for arch in ("unet1d", "mamba"):
            apply_fn, meta = _build_model_apply(arch, num_classes=4)
            chunk_size = 64
            arrays = [_ascii_bytes(97, offset=2), _ascii_bytes(129, offset=9)]
            engine = self._make_engine(
                apply_fn=apply_fn,
                meta=meta,
                num_classes=4,
                chunk_size=chunk_size,
                batch_size=3,
                inference_backend="fast",
                full_memory_budget_bytes=1,
            )
            fast_labels, fast_probs, _ = engine.segment_bytes_batch(arrays)
            legacy_labels, legacy_probs, _ = _legacy_segment_bytes_batch(
                apply_fn=apply_fn,
                num_classes=4,
                chunk_size=chunk_size,
                batch_size=3,
                byte_arrays=arrays,
            )
            for got, want in zip(fast_labels, legacy_labels):
                np.testing.assert_array_equal(got, want)
                self.assertEqual(_char_labels_for_ascii(got), _char_labels_for_ascii(want))
            for got, want in zip(fast_probs, legacy_probs):
                np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)

    def test_fast_max_probs_matches_legacy_for_over_chunk_inputs(self) -> None:
        for arch in ("unet1d", "mamba"):
            apply_fn, meta = _build_model_apply(arch, num_classes=4)
            chunk_size = 64
            arrays = [_ascii_bytes(97, offset=2), _ascii_bytes(129, offset=9)]
            engine = self._make_engine(
                apply_fn=apply_fn,
                meta=meta,
                num_classes=4,
                chunk_size=chunk_size,
                batch_size=3,
                inference_backend="fast",
                full_memory_budget_bytes=1,
            )
            fast_labels, fast_max_probs, _ = engine.segment_bytes_batch_labels_and_max_probs(arrays)
            legacy_labels, legacy_probs, _ = _legacy_segment_bytes_batch(
                apply_fn=apply_fn,
                num_classes=4,
                chunk_size=chunk_size,
                batch_size=3,
                byte_arrays=arrays,
            )
            for got, want in zip(fast_labels, legacy_labels):
                np.testing.assert_array_equal(got, want)
            for got, want in zip(fast_max_probs, legacy_probs):
                np.testing.assert_allclose(got, np.max(want, axis=-1), atol=1e-5, rtol=1e-5)

    def test_full_path_pads_to_batch_local_max(self) -> None:
        def apply_fn(tok: jnp.ndarray) -> jnp.ndarray:
            tok_f = tok.astype(jnp.float32)
            return jnp.stack([tok_f, tok_f * 0.0, -tok_f], axis=-1)

        engine = self._make_engine(
            apply_fn=jax.jit(apply_fn, backend="cpu"),
            meta={"arch": "unet1d", "model_dim": 8, "channels": (8,), "mamba_layers": 1, "mamba_d_state": 4, "mamba_expand": 1, "mamba_bidirectional": True},
            num_classes=3,
            chunk_size=64,
            batch_size=3,
        )
        arrays = [_ascii_bytes(9), _ascii_bytes(13, offset=5), _ascii_bytes(5, offset=11)]
        engine.segment_bytes_batch(arrays)
        records = [record for record in engine.execution_history if record.mode == "segment_full"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].actual_batch_size, 3)
        self.assertEqual(records[0].padded_length, 13)

    def test_partial_microbatches_use_actual_batch_size(self) -> None:
        def apply_fn(tok: jnp.ndarray) -> jnp.ndarray:
            tok_f = tok.astype(jnp.float32)
            return jnp.stack([tok_f, -tok_f, tok_f * 0.5], axis=-1)

        engine = self._make_engine(
            apply_fn=jax.jit(apply_fn, backend="cpu"),
            meta={"arch": "unet1d", "model_dim": 8, "channels": (8,), "mamba_layers": 1, "mamba_d_state": 4, "mamba_expand": 1, "mamba_bidirectional": True},
            num_classes=3,
            chunk_size=64,
            batch_size=2,
        )
        arrays = [_ascii_bytes(10), _ascii_bytes(12, offset=4), _ascii_bytes(8, offset=7)]
        engine.segment_bytes_batch(arrays)
        records = [record for record in engine.execution_history if record.mode == "segment_full"]
        self.assertEqual([record.actual_batch_size for record in records], [2, 1])
        self.assertEqual([record.padded_length for record in records], [12, 8])

    def test_low_memory_cpu_buckets_short_stream_shapes_to_chunk_size(self) -> None:
        def apply_fn(tok: jnp.ndarray) -> jnp.ndarray:
            tok_f = tok.astype(jnp.float32)
            return jnp.stack([tok_f, -tok_f, tok_f * 0.5], axis=-1)

        engine = self._make_engine(
            apply_fn=jax.jit(apply_fn, backend="cpu"),
            meta={"arch": "unet1d", "model_dim": 8, "channels": (8,), "mamba_layers": 1, "mamba_d_state": 4, "mamba_expand": 1, "mamba_bidirectional": True},
            num_classes=3,
            chunk_size=64,
            batch_size=1,
            inference_backend="auto",
            full_memory_budget_bytes=1,
        )
        engine.segment_bytes_batch([_ascii_bytes(13)])
        records = [record for record in engine.execution_history if record.mode == "segment_stream"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].padded_length, 64)

    def test_build_window_spans_covers_tail_without_gaps(self) -> None:
        length = 97
        chunk_size = 64
        spans = build_window_spans(length, chunk_size)
        self.assertTrue(spans)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], length)
        coverage = np.zeros((length,), dtype=np.int32)
        for start, end in spans:
            coverage[start:end] += 1
        self.assertTrue(np.all(coverage > 0))

    def test_auto_logs_size_fallback_to_stream(self) -> None:
        messages = []

        def apply_fn(tok: jnp.ndarray) -> jnp.ndarray:
            tok_f = tok.astype(jnp.float32)
            return jnp.stack([tok_f, tok_f * 0.0, -tok_f], axis=-1)

        engine = self._make_engine(
            apply_fn=jax.jit(apply_fn, backend="cpu"),
            meta={"arch": "unet1d", "model_dim": 8, "channels": (8,), "mamba_layers": 1, "mamba_d_state": 4, "mamba_expand": 1, "mamba_bidirectional": True},
            num_classes=3,
            chunk_size=32,
            batch_size=2,
            inference_backend="auto",
            log_sink=messages,
            full_memory_budget_bytes=1,
        )
        engine.segment_bytes_batch([_ascii_bytes(71)])
        joined = "\n".join(messages)
        self.assertIn("fast_full -> fast_stream", joined)
        self.assertIn("trigger=oom_or_size", joined)

    def test_auto_logs_pallas_probe_fallback_for_mamba_gpu_candidate(self) -> None:
        messages = []
        apply_fn, meta = _build_model_apply("mamba", num_classes=4)
        engine = self._make_engine(
            apply_fn=apply_fn,
            meta=meta,
            num_classes=4,
            chunk_size=16,
            batch_size=1,
            inference_backend="auto",
            log_sink=messages,
            execution_platform_override="gpu",
        )
        engine.segment_bytes_batch([_ascii_bytes(16)])
        joined = "\n".join(messages)
        self.assertIn("fast_pallas -> fast_full", joined)
        self.assertIn("trigger=missing_kernel_support", joined)

    def test_auto_keeps_fast_full_when_cuda_kernel_is_available(self) -> None:
        messages = []

        def apply_fn(tok: jnp.ndarray) -> jnp.ndarray:
            tok_f = tok.astype(jnp.float32)
            return jnp.stack([tok_f, -tok_f, tok_f * 0.5], axis=-1)

        engine = self._make_engine(
            apply_fn=jax.jit(apply_fn, backend="cpu"),
            meta={"arch": "mamba1d", "model_dim": 8, "channels": (), "mamba_layers": 1, "mamba_d_state": 4, "mamba_expand": 1, "mamba_bidirectional": True},
            num_classes=3,
            chunk_size=16,
            batch_size=1,
            inference_backend="auto",
            log_sink=messages,
            execution_platform_override="gpu",
            cuda_kernel_available=True,
        )
        engine.segment_bytes_batch([_ascii_bytes(16)])
        joined = "\n".join(messages)
        self.assertNotIn("fast_pallas -> fast_full", joined)
        records = [record for record in engine.execution_history if record.mode == "segment_full"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].path, "fast_full")

    @unittest.skipUnless(has_cuda_mamba_kernel(), "CUDA Pallas/Triton kernel is unavailable")
    def test_cuda_selective_scan_matches_associative_reference(self) -> None:
        gpu_device = jax.devices("gpu")[0]
        key = jax.random.PRNGKey(0)
        batch = 2
        length = 33
        d_inner = 16
        d_state = 4
        x = jax.device_put(jax.random.normal(key, (batch, length, d_inner), dtype=jnp.float32), gpu_device)
        dt = jax.device_put(
            jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(1), (batch, length, d_inner), dtype=jnp.float32)) + 1e-4,
            gpu_device,
        )
        B = jax.device_put(jax.random.normal(jax.random.PRNGKey(2), (batch, length, d_state), dtype=jnp.float32), gpu_device)
        C = jax.device_put(jax.random.normal(jax.random.PRNGKey(3), (batch, length, d_state), dtype=jnp.float32), gpu_device)
        A = jax.device_put(
            -jnp.exp(jax.random.normal(jax.random.PRNGKey(4), (d_inner, d_state), dtype=jnp.float32)),
            gpu_device,
        )
        D = jax.device_put(jax.random.normal(jax.random.PRNGKey(5), (d_inner,), dtype=jnp.float32), gpu_device)

        want = np.asarray(jax.device_get(_selective_scan_associative(x, dt, B, C, A, D)), dtype=np.float32)
        got = np.asarray(jax.device_get(selective_scan_cuda(x, dt, B, C, A, D)), dtype=np.float32)
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
