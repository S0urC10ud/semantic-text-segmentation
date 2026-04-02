from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import viewers.content_type_segmentor as appmod
import viewers.core as core


STATIC_ROOT = ROOT / "viewers" / "content_type_segmentor_static"
MANIFEST_PATH = STATIC_ROOT / "assets" / "model_manifest.json"
WEIGHTS_PATH = STATIC_ROOT / "assets" / "sfullfiles4_weights.npz"
DEMO_INPUT_PATH = STATIC_ROOT / "assets" / "demo_input.txt"
CHECKPOINT_PATH = ROOT / "checkpoints" / "sweeps" / "sfullfiles4.msgpack"
RUNTIME_PATH = STATIC_ROOT / "py" / "runtime.py"


def _load_runtime_module():
    spec = importlib.util.spec_from_file_location("content_type_segmentor_runtime_test", RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load runtime module from {RUNTIME_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expand_segments(text_len: int, segments: list[dict]) -> list[int]:
    labels = [0] * text_len
    for segment in segments:
        label_id = int(segment["label_id"])
        for idx in range(int(segment["start"]), int(segment["end"])):
            labels[idx] = label_id
    return labels


def _native_full_sequence_probs(runtime_mod, predictor, text: str) -> np.ndarray:
    normalized = runtime_mod._normalize_input_text(text)
    raw_bytes = np.frombuffer(normalized.encode("utf-8", "ignore"), dtype=np.uint8)
    sanitized = runtime_mod._sanitize_model_bytes(raw_bytes).astype(np.int32, copy=False)
    logits = predictor._apply_legacy(core.jnp.array(sanitized[None, :], dtype=core.jnp.int32))
    return np.asarray(core.jax.nn.softmax(logits, axis=-1), dtype=np.float32)[0]


class TestContentTypeSegmentor(unittest.TestCase):
    def test_runtime_matches_predictor_for_demo_input(self) -> None:
        runtime_mod = _load_runtime_module()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        sample_text = DEMO_INPUT_PATH.read_text(encoding="utf-8")

        model = runtime_mod.NumpyMambaSegmentor.from_files(MANIFEST_PATH, WEIGHTS_PATH)
        payload = model.segment_text_payload(sample_text, top_k=5)

        predictor = core.Predictor(
            ckpt_path=str(CHECKPOINT_PATH),
            num_classes=len(manifest["label_order"]),
            model_dim=int(manifest["model"]["d_model"]),
            channels=tuple(core.DEFAULT_CHANNELS),
            arch="mamba",
            mamba_layers=int(manifest["model"]["n_layers"]),
            mamba_d_state=int(manifest["model"]["d_state"]),
            mamba_expand=int(manifest["model"]["expand"]),
            mamba_dt_rank=int(manifest["model"]["dt_rank"]),
            mamba_conv=int(manifest["model"]["d_conv"]),
            mamba_bidirectional=bool(manifest["model"]["bidirectional"]),
            dtype_str=str(manifest["model"]["dtype"]),
            chunk=int(manifest["window_bytes"]),
            other_threshold=float(manifest["other_threshold"]),
            device="cpu",
            inference_backend="legacy",
        )

        expected_byte_probs = _native_full_sequence_probs(runtime_mod, predictor, sample_text)
        actual_byte_probs, spans = model.segment_byte_probs(
            np.frombuffer(payload["text"].encode("utf-8", "ignore"), dtype=np.uint8)
        )

        self.assertEqual(payload["text"], core._normalize_input_text(sample_text))
        self.assertEqual(spans, [(0, int(payload["input_bytes"]))])
        self.assertTrue(np.allclose(actual_byte_probs, expected_byte_probs, atol=1e-5, rtol=1e-4))
        self.assertGreater(len(payload["stats"]), 0)
        self.assertGreater(int(payload["input_bytes"]), 0)

        short_text = (
            "<div>hello</div>\n"
            "<script>const x = {a: 1, b: 2}; function f(){ return x.a + x.b; }</script>\n"
            "<style>body{color:red}</style>"
        )
        short_payload = model.segment_text_payload(short_text, top_k=5)
        short_expected = _native_full_sequence_probs(runtime_mod, predictor, short_text)
        short_actual, short_spans = model.segment_byte_probs(
            np.frombuffer(short_payload["text"].encode("utf-8", "ignore"), dtype=np.uint8)
        )
        self.assertEqual(short_payload["text"], core._normalize_input_text(short_text))
        self.assertEqual(short_spans, [(0, int(short_payload["input_bytes"]))])
        self.assertTrue(np.allclose(short_actual, short_expected, atol=1e-5, rtol=1e-4))

    def test_runtime_allows_threshold_override(self) -> None:
        runtime_mod = _load_runtime_module()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        sample_text = DEMO_INPUT_PATH.read_text(encoding="utf-8")
        other_id = int(manifest["num_classes"])

        model = runtime_mod.NumpyMambaSegmentor.from_files(MANIFEST_PATH, WEIGHTS_PATH)
        default_payload = model.segment_text_payload(sample_text, top_k=5)
        stricter_payload = model.segment_text_payload(sample_text, top_k=5, threshold=0.95)

        default_other = _expand_segments(len(default_payload["text"]), default_payload["segments"]).count(other_id)
        stricter_other = _expand_segments(len(stricter_payload["text"]), stricter_payload["segments"]).count(other_id)

        self.assertAlmostEqual(float(manifest["other_threshold"]), 0.3, places=6)
        self.assertAlmostEqual(float(default_payload["other_threshold"]), 0.3, places=6)
        self.assertAlmostEqual(float(stricter_payload["other_threshold"]), 0.95, places=6)
        self.assertGreaterEqual(stricter_other, default_other)

    def test_public_demo_serves_static_ui_without_backend_segment_route(self) -> None:
        client = TestClient(appmod.create_app())

        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Content Type Segmentor", response.text)

        manifest_response = client.get("/static/assets/model_manifest.json")
        self.assertEqual(manifest_response.status_code, 200)
        self.assertEqual(manifest_response.json()["model_id"], "sfullfiles4")
        self.assertAlmostEqual(float(manifest_response.json()["other_threshold"]), 0.3, places=6)

        healthz = client.get("/healthz")
        self.assertEqual(healthz.status_code, 200)
        self.assertTrue(bool(healthz.json()["ok"]))

        missing_backend = client.post("/api/segment", json={"text": "x"})
        self.assertEqual(missing_backend.status_code, 404)


if __name__ == "__main__":
    unittest.main()
