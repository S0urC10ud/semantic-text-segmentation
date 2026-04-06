from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
VIEWER_PATH = ROOT / "viewers" / "interactive_viewer.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_interactive_viewer_module():
    fake_core = types.ModuleType("viewers.core")

    class FakePredictor:
        instances: list["FakePredictor"] = []

        def __init__(self, **_kwargs) -> None:
            self.chunk = 1536
            self.calls: list[dict[str, object]] = []
            self.last_postprocess_trace = None
            type(self).instances.append(self)

        def segment_text(self, text: str, min_run_chars: int = 5, chunk=None):
            self.calls.append(
                {
                    "text": str(text),
                    "min_run_chars": int(min_run_chars),
                    "chunk": chunk,
                }
            )
            probs = [{str(0): 1.0} for _ in text]
            if len(text) >= 2:
                self.last_postprocess_trace = {
                    "changed_mask": [False, True] + [False] * max(0, len(text) - 2),
                    "descriptions": ["", "markdown structure fill, paired delimiter fill"] + [""] * max(0, len(text) - 2),
                    "original_labels": [0, 1] + [0] * max(0, len(text) - 2),
                }
            else:
                self.last_postprocess_trace = None
            return [(0, len(text), 0)], [0] * len(text), probs, []

    fake_core.DEFAULT_CHANNELS = (96, 128, 192, 256)
    fake_core.DEFAULT_CHUNK_SIZE = 1536
    fake_core.Predictor = FakePredictor
    fake_core.auto_color = lambda index, total: "#112233"
    fake_core._apply_label_mapping = lambda label_names: None
    fake_core._hex_to_rgba = lambda color, alpha: f"rgba(17,34,51,{float(alpha):.2f})"
    fake_core._hex_to_rgba_confidence = (
        lambda color, alpha, confidence: f"rgba(17,34,51,{float(alpha):.2f})"
    )
    fake_core._infer_checkpoint_architecture = lambda path: {
        "arch": "unet1d",
        "channels": [96, 128, 192, 256],
        "dtype": "float32",
        "model_dim": 256,
        "num_classes": 1,
    }
    fake_core._load_checkpoint_hparams = lambda path: {}
    fake_core._normalize_input_text = lambda text: str(text or "").replace("\r", "\n")
    fake_core._resolve_colors = lambda langs, colors: ["#112233" for _ in langs]
    fake_core._resolve_hparam = (
        lambda cli, auto, inferred, default: (
            cli if cli is not None else auto if auto is not None else inferred if inferred is not None else default,
            "test",
        )
    )
    fake_core._resolve_langs_and_display = lambda lang_arg: (["html"], ["html"])
    fake_core.make_slug = lambda name: str(name).lower()

    saved_core = sys.modules.get("viewers.core")
    sys.modules["viewers.core"] = fake_core
    old_argv = sys.argv[:]

    with tempfile.NamedTemporaryFile(suffix=".msgpack") as ckpt:
        try:
            sys.argv = [str(VIEWER_PATH), "--ckpt", ckpt.name]
            spec = importlib.util.spec_from_file_location("viewers.interactive_viewer_test", VIEWER_PATH)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load interactive viewer from {VIEWER_PATH}")
            module = importlib.util.module_from_spec(spec)
            module.__package__ = "viewers"
            spec.loader.exec_module(module)
            return module, FakePredictor
        finally:
            sys.argv = old_argv
            if saved_core is None:
                sys.modules.pop("viewers.core", None)
            else:
                sys.modules["viewers.core"] = saved_core


class TestInteractiveViewer(unittest.TestCase):
    def test_default_min_run_is_five_and_api_uses_it(self) -> None:
        module, fake_predictor_cls = _load_interactive_viewer_module()
        self.assertEqual(module.SegmentRequest(text="abc").min_run, 5)
        self.assertAlmostEqual(float(module.args.other_threshold), 0.3, places=6)
        self.assertEqual(len(fake_predictor_cls.instances), 1)

        client = TestClient(module.app)
        response = client.post("/api/segment", json={"text": "abc"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["segments"], [{"start": 0, "end": 3, "label": 0}])
        self.assertEqual(fake_predictor_cls.instances[0].calls[0]["min_run_chars"], 5)
        self.assertEqual(payload["postprocess"]["changed_chars"], 1)
        self.assertIn("class=\"char postprocessed\"", payload["html"])
        self.assertIn("data-postprocess=", payload["html"])
        self.assertIn("markdown structure fill", payload["html"])
        self.assertIn("paired delimiter fill", payload["html"])


if __name__ == "__main__":
    unittest.main()
