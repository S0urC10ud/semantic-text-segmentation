from __future__ import annotations

import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

VIEWER_PATH = ROOT / "viewers" / "evaluation_viewer.py"


def _load_viewer_module(data_root: Path) -> dict:
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(VIEWER_PATH),
            "--dataset-only",
            "--data-root",
            str(data_root),
        ]
        return runpy.run_path(str(VIEWER_PATH), run_name="evaluation_viewer_test")
    finally:
        sys.argv = old_argv


class _FakeStore:
    max_preview_chars = 6000

    def get_example(self, task: str, index: int):
        return (
            {
                "content": "AB",
                "segments": {
                    "label": ["python"],
                    "char_start": [0],
                    "char_end": [2],
                },
                "example_id": "viewer-test-0001",
                "source_langs": ["python"],
                "metadata_json": "{}",
            },
            1,
        )


class _FakeRunner:
    def __init__(self, num_classes: int, python_id: int) -> None:
        self.num_classes = int(num_classes)
        self.python_id = int(python_id)

    def segment_text(self, text: str, *, min_run_chars: int = 1):
        probs = [
            np.eye(self.num_classes, dtype=np.float32)[self.python_id] * 0.40,
            np.eye(self.num_classes, dtype=np.float32)[self.python_id] * 0.95,
        ]
        pred = [self.num_classes, self.python_id]
        return pred, pred, probs


class TestEvaluationViewerOtherThreshold(unittest.TestCase):
    def test_default_threshold_and_other_palette(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = _load_viewer_module(Path(tmpdir))

        self.assertAlmostEqual(float(mod["args"].other_threshold), 0.85)
        palette = mod["_build_sample_palette"](["python", "other"])
        self.assertEqual(palette["other"], "#7f8c8d")
        self.assertEqual(
            mod["_prediction_label_from_id"](
                int(mod["cfg"].NUM_CLASSES),
                num_classes=int(mod["cfg"].NUM_CLASSES),
                unknown_label="__unknown__",
            ),
            "other",
        )

    def test_api_sample_surfaces_virtual_other_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mod = _load_viewer_module(Path(tmpdir))

        mod["api_sample"].__globals__["runner"] = _FakeRunner(
            int(mod["cfg"].NUM_CLASSES),
            int(mod["cfg"].LANG2ID["python"]),
        )
        mod["api_sample"].__globals__["store"] = _FakeStore()

        response = mod["api_sample"]("unit_task", 0)

        self.assertFalse(response["dataset_only"])
        self.assertEqual(response["prediction_config"]["other_threshold"], 0.85)
        self.assertEqual(response["prediction_config"]["other_label"], "other")
        self.assertEqual(response["char_data"][0]["pred"], "other")
        self.assertIn("other", response["label_order"])
        self.assertEqual(response["label_colors"]["other"], "#7f8c8d")


if __name__ == "__main__":
    unittest.main()
