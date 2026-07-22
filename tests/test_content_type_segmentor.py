from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

os.environ.setdefault("JAX_PLATFORMS", "cpu")

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


def _fake_runtime_model(runtime_mod, *, num_classes: int = 2, label_order: list[str] | None = None):
    model = runtime_mod.NumpyMambaSegmentor.__new__(runtime_mod.NumpyMambaSegmentor)
    model.num_classes = int(num_classes)
    model.other_label_id = int(num_classes)
    model.other_threshold = 0.0
    model.max_input_bytes = 0
    model.postprocess_min_run_chars = 5
    model.postprocess_boundary_snap_max_shift = 2
    if label_order is not None:
        model.label_order = [str(label) for label in label_order]
    elif int(num_classes) == 2:
        model.label_order = ["javascript_typescript", "json"]
    else:
        model.label_order = [f"label_{idx}" for idx in range(int(num_classes))]
    model.display_labels = list(model.label_order)
    model.local_host_postprocess_rules = runtime_mod._resolve_local_host_postprocess_rules(model.label_order)
    model.markdown_label_id = runtime_mod._resolve_label_id(model.label_order, "markdown")
    model.html_label_id = runtime_mod._resolve_label_id(model.label_order, "html")
    return model


class TestContentTypeMapmentor(unittest.TestCase):
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

    def test_runtime_payload_applies_postprocess_defaults_on_crafted_cases(self) -> None:
        runtime_mod = _load_runtime_module()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual(int(manifest["postprocess_min_run_chars"]), 5)
        self.assertEqual(int(manifest["postprocess_boundary_snap_max_shift"]), 2)

        island_model = _fake_runtime_model(runtime_mod, num_classes=2)
        island_text = "AAAABBBAAAA"
        island_probs = np.array(
            [[0.95, 0.05]] * 4 + [[0.10, 0.90]] * 3 + [[0.95, 0.05]] * 4,
            dtype=np.float32,
        )
        island_model.segment_byte_probs = lambda byte_arr: (island_probs, [(0, int(len(byte_arr)))])
        island_payload = island_model.segment_text_payload(island_text, top_k=2, threshold=0.0)
        self.assertEqual(
            island_payload["segments"],
            [{"start": 0, "end": len(island_text), "label_id": 0}],
        )

        snap_model = _fake_runtime_model(runtime_mod, num_classes=2)
        snap_text = "<abX"
        snap_probs = np.array(
            [
                [0.90, 0.10],
                [0.52, 0.50],
                [0.51, 0.50],
                [0.05, 0.95],
            ],
            dtype=np.float32,
        )
        snap_model.segment_byte_probs = lambda byte_arr: (snap_probs, [(0, int(len(byte_arr)))])
        snap_payload = snap_model.segment_text_payload(snap_text, top_k=2, threshold=0.0)
        self.assertEqual(
            snap_payload["segments"],
            [
                {"start": 0, "end": 1, "label_id": 0},
                {"start": 1, "end": len(snap_text), "label_id": 1},
            ],
        )

        indent_model = _fake_runtime_model(runtime_mod, num_classes=2)
        indent_text = ">\n  b"
        indent_probs = np.array(
            [
                [0.98, 0.02],
                [0.98, 0.02],
                [0.05, 0.95],
                [0.05, 0.95],
                [0.05, 0.95],
            ],
            dtype=np.float32,
        )
        indent_model.segment_byte_probs = lambda byte_arr: (indent_probs, [(0, int(len(byte_arr)))])
        indent_payload = indent_model.segment_text_payload(indent_text, top_k=2, threshold=0.0)
        self.assertEqual(
            indent_payload["segments"],
            [
                {"start": 0, "end": 2, "label_id": 0},
                {"start": 2, "end": len(indent_text), "label_id": 1},
            ],
        )

        tag_opener_model = _fake_runtime_model(runtime_mod, num_classes=2)
        tag_opener_text = "x\n</"
        tag_opener_probs = np.array(
            [
                [0.08, 0.90],
                [0.08, 0.90],
                [0.93, 0.04],
                [0.94, 0.03],
            ],
            dtype=np.float32,
        )
        tag_opener_model.segment_byte_probs = lambda byte_arr: (tag_opener_probs, [(0, int(len(byte_arr)))])
        tag_opener_payload = tag_opener_model.segment_text_payload(tag_opener_text, top_k=2, threshold=0.0)
        self.assertEqual(
            tag_opener_payload["segments"],
            [
                {"start": 0, "end": 2, "label_id": 1},
                {"start": 2, "end": len(tag_opener_text), "label_id": 0},
            ],
        )

        wrapped_model = _fake_runtime_model(runtime_mod, num_classes=2)
        wrapped_text = 'x="alert(\'clicked\')">'
        wrapped_probs = np.array(
            [
                [0.98, 0.02],
                [0.97, 0.03],
                [0.96, 0.04],
                [0.54, 0.46],
            ]
            + [[0.08, 0.92]] * 14
            + [
                [0.45, 0.55],
                [0.97, 0.03],
                [0.98, 0.02],
            ],
            dtype=np.float32,
        )
        wrapped_model.segment_byte_probs = lambda byte_arr: (wrapped_probs, [(0, int(len(byte_arr)))])
        wrapped_payload = wrapped_model.segment_text_payload(wrapped_text, top_k=2, threshold=0.0)
        self.assertEqual(
            wrapped_payload["segments"],
            [
                {"start": 0, "end": 3, "label_id": 0},
                {"start": 3, "end": 19, "label_id": 1},
                {"start": 19, "end": len(wrapped_text), "label_id": 0},
            ],
        )

        noisy_wrapped_text = '<button class="btn" on<!--hi yanick-->click="alert(\'clicked\')">Click</button>'
        noisy_wrapped_model = _fake_runtime_model(runtime_mod, num_classes=2)
        noisy_wrapped_probs = np.array([[0.82, 0.08]] * len(noisy_wrapped_text), dtype=np.float32)
        wrapped_text = '"alert(\'clicked\')"'
        open_quote = noisy_wrapped_text.index(wrapped_text)
        inner_start = open_quote + 1
        close_quote = open_quote + len(wrapped_text) - 1
        noisy_wrapped_probs[open_quote] = np.array([0.45, 0.55], dtype=np.float32)
        noisy_wrapped_probs[inner_start:close_quote] = np.array([0.10, 0.82], dtype=np.float32)
        noisy_wrapped_probs[close_quote] = np.array([0.82, 0.10], dtype=np.float32)
        noisy_wrapped_model.segment_byte_probs = (
            lambda byte_arr: (noisy_wrapped_probs, [(0, int(len(byte_arr)))])
        )
        noisy_wrapped_payload = noisy_wrapped_model.segment_text_payload(
            noisy_wrapped_text,
            top_k=2,
            threshold=0.0,
        )
        noisy_wrapped_labels = _expand_segments(len(noisy_wrapped_text), noisy_wrapped_payload["segments"])
        self.assertEqual(noisy_wrapped_labels[open_quote], 0)
        self.assertTrue(all(label == 1 for label in noisy_wrapped_labels[inner_start:close_quote]))
        self.assertEqual(noisy_wrapped_labels[close_quote], 0)

        json_in_js_model = _fake_runtime_model(runtime_mod, num_classes=2)
        json_in_js_text = 'const x = {"a":1};'
        json_in_js_probs = np.array([[0.70, 0.20]] * len(json_in_js_text), dtype=np.float32)
        json_start = json_in_js_text.index("{")
        json_end = json_in_js_text.index("}") + 1
        for idx in range(json_start, json_end):
            json_in_js_probs[idx] = np.array([0.25, 0.65], dtype=np.float32)
        json_in_js_model.segment_byte_probs = lambda byte_arr: (json_in_js_probs, [(0, int(len(byte_arr)))])
        json_in_js_payload = json_in_js_model.segment_text_payload(json_in_js_text, top_k=2, threshold=0.0)
        self.assertEqual(
            json_in_js_payload["segments"],
            [{"start": 0, "end": len(json_in_js_text), "label_id": 0}],
        )

        fence_model = _fake_runtime_model(
            runtime_mod,
            num_classes=3,
            label_order=["markdown", "shell", "text"],
        )
        fence_text = "- ```bash\necho hi\n```"
        fence_probs = np.array([[0.05, 0.05, 0.90]] * len(fence_text), dtype=np.float32)
        fence_probs[:8] = np.array([0.90, 0.05, 0.05], dtype=np.float32)
        fence_probs[8:19] = np.array([0.05, 0.90, 0.05], dtype=np.float32)
        fence_probs[19:] = np.array([0.90, 0.05, 0.05], dtype=np.float32)
        fence_model.segment_byte_probs = lambda byte_arr: (fence_probs, [(0, int(len(byte_arr)))])
        fence_payload = fence_model.segment_text_payload(fence_text, top_k=3, threshold=0.0)
        self.assertEqual(
            fence_payload["segments"],
            [
                {"start": 0, "end": 9, "label_id": 0},
                {"start": 9, "end": 18, "label_id": 1},
                {"start": 18, "end": len(fence_text), "label_id": 0},
            ],
        )

        inline_ticks_model = _fake_runtime_model(
            runtime_mod,
            num_classes=3,
            label_order=["markdown", "config", "other"],
        )
        inline_ticks_text = "file ```conf/org/YOUR_ORG.yaml``` end"
        inline_ticks_probs = np.array([[0.90, 0.05, 0.05]] * len(inline_ticks_text), dtype=np.float32)
        open_start = inline_ticks_text.index("```")
        body_start = open_start + 3
        close_start = inline_ticks_text.index("```", body_start)
        inline_ticks_probs[body_start:close_start] = np.array([0.15, 0.75, 0.10], dtype=np.float32)
        inline_ticks_probs[close_start:] = np.array([0.10, 0.10, 0.80], dtype=np.float32)
        inline_ticks_model.segment_byte_probs = lambda byte_arr: (inline_ticks_probs, [(0, int(len(byte_arr)))])
        inline_ticks_payload = inline_ticks_model.segment_text_payload(inline_ticks_text, top_k=3, threshold=0.0)
        self.assertEqual(
            inline_ticks_payload["segments"],
            [
                {"start": 0, "end": body_start, "label_id": 0},
                {"start": body_start, "end": close_start, "label_id": 1},
                {"start": close_start, "end": close_start + 3, "label_id": 0},
                {"start": close_start + 3, "end": len(inline_ticks_text), "label_id": 2},
            ],
        )

    def test_public_demo_serves_static_ui_without_backend_segment_route(self) -> None:
        client = TestClient(appmod.create_app())

        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Content Type Segmentor", response.text)

        manifest_response = client.get("/static/assets/model_manifest.json")
        self.assertEqual(manifest_response.status_code, 200)
        self.assertEqual(manifest_response.json()["model_id"], "sfullfiles4")
        self.assertAlmostEqual(float(manifest_response.json()["other_threshold"]), 0.3, places=6)
        self.assertEqual(int(manifest_response.json()["postprocess_min_run_chars"]), 5)
        self.assertEqual(int(manifest_response.json()["postprocess_boundary_snap_max_shift"]), 2)

        healthz = client.get("/healthz")
        self.assertEqual(healthz.status_code, 200)
        self.assertTrue(bool(healthz.json()["ok"]))

        missing_backend = client.post("/api/segment", json={"text": "x"})
        self.assertEqual(missing_backend.status_code, 404)


if __name__ == "__main__":
    unittest.main()
