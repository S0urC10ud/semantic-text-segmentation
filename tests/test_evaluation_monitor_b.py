from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import evaluation.evaluation as evalmod
from train.utils.monitor_eval import FILE_DTYPE, SEG_DTYPE


def _monitor_data() -> dict:
    chunks = [
        np.frombuffer(b"abc\n", dtype=np.uint8),
        np.frombuffer(b"TEXT", dtype=np.uint8),
    ]
    files = np.zeros(2, dtype=FILE_DTYPE)
    segments = np.zeros(2, dtype=SEG_DTYPE)
    offset = 0
    for idx, arr in enumerate(chunks):
        files[idx] = (
            offset,
            int(arr.shape[0]),
            idx,
            1,
            0,
            int(evalmod.cfg.LANG2ID["python" if idx == 0 else "text"]),
        )
        segments[idx] = (
            idx,
            0,
            int(arr.shape[0]),
            int(evalmod.cfg.LANG2ID["python" if idx == 0 else "text"]),
        )
        offset += int(arr.shape[0])
    return {
        "meta": {"num_files": 2},
        "files": files,
        "segments": segments,
        "contents": np.concatenate(chunks, axis=0),
    }


class _FakeUnetMonitorRunner:
    arch = "unet1d"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _segment_bytes_labels_only(self, byte_arr: np.ndarray) -> np.ndarray:
        self.calls.append("fast")
        return np.zeros((int(byte_arr.shape[0]),), dtype=np.uint8)

    def _segment_bytes_labels_only_legacy(self, byte_arr: np.ndarray) -> np.ndarray:
        self.calls.append("legacy")
        content = np.asarray(byte_arr, dtype=np.uint8).tobytes()
        label = "python" if content.startswith(b"abc") else "text"
        return np.full(
            (int(byte_arr.shape[0]),),
            int(evalmod.cfg.LANG2ID[label]),
            dtype=np.uint8,
        )


class _FakeMambaMonitorRunner:
    arch = "mamba"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _segment_bytes_labels_only(self, byte_arr: np.ndarray) -> np.ndarray:
        self.calls.append("normal")
        content = np.asarray(byte_arr, dtype=np.uint8).tobytes()
        label = "python" if content.startswith(b"abc") else "text"
        return np.full(
            (int(byte_arr.shape[0]),),
            int(evalmod.cfg.LANG2ID[label]),
            dtype=np.uint8,
        )

    def _segment_bytes_labels_only_legacy(self, byte_arr: np.ndarray) -> np.ndarray:
        self.calls.append("legacy")
        return np.zeros((int(byte_arr.shape[0]),), dtype=np.uint8)


class TestEvaluationMonitorB(unittest.TestCase):
    def test_configure_evaluation_mode_defaults_to_fine_tuned(self) -> None:
        args = SimpleNamespace(
            data_root=str(evalmod.REPO_ROOT / "evaluation" / "data"),
            fine_tuned=False,
            non_fine_tuned=False,
        )
        info = evalmod._configure_evaluation_mode(args)
        self.assertTrue(info["fine_tuned_mode"])
        self.assertEqual(
            Path(args.data_root).resolve(),
            (evalmod.REPO_ROOT / "evaluation" / "data_b").resolve(),
        )
        self.assertTrue(any("Fine-tuned mode (default)" in msg for msg in info["messages"]))

    def test_configure_evaluation_mode_warns_for_non_fine_tuned(self) -> None:
        args = SimpleNamespace(
            data_root=str(evalmod.REPO_ROOT / "evaluation" / "data"),
            fine_tuned=False,
            non_fine_tuned=True,
        )
        info = evalmod._configure_evaluation_mode(args)
        self.assertFalse(info["fine_tuned_mode"])
        self.assertEqual(
            Path(args.data_root).resolve(),
            (evalmod.REPO_ROOT / "evaluation" / "data").resolve(),
        )
        self.assertTrue(any("legacy/non-default path" in msg for msg in info["messages"]))

    def test_full_monitor_b_uses_legacy_sliding_window_for_unet(self) -> None:
        runner = _FakeUnetMonitorRunner()
        with mock.patch.object(evalmod, "load_monitor_memmaps", return_value=_monitor_data()):
            payload = evalmod._evaluate_full_monitor_b(Path("/tmp/monitor_b"), runner)

        self.assertEqual(runner.calls, ["legacy", "legacy"])
        self.assertEqual(payload["inference_mode"], "sliding_window_legacy")
        self.assertEqual(payload["files_used"], 2)
        self.assertEqual(payload["skipped"], 0)
        self.assertEqual(payload["evaluated_bytes"], 7)
        self.assertEqual(payload["rows"][0]["label"], "ALL (agg)")
        self.assertAlmostEqual(float(payload["aggregates"]["micro_acc"]), 1.0)
        self.assertEqual(int(payload["by_label"]["python"]["support"]), 3)
        self.assertEqual(int(payload["by_label"]["text"]["support"]), 4)

        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            sample_seed=13,
            other_threshold=0.0,
        )
        comparison = evalmod._collect_comparison_metrics(
            args,
            [],
            [],
            None,
            monitor_b_report=payload,
        )
        self.assertIn("monitor_b", comparison)
        self.assertEqual(comparison["monitor_b"]["rows"][0]["label"], "ALL (agg)")

    def test_full_monitor_b_keeps_normal_path_for_mamba(self) -> None:
        runner = _FakeMambaMonitorRunner()
        with mock.patch.object(evalmod, "load_monitor_memmaps", return_value=_monitor_data()):
            payload = evalmod._evaluate_full_monitor_b(Path("/tmp/monitor_b"), runner)

        self.assertEqual(runner.calls, ["normal", "normal"])
        self.assertEqual(payload["inference_mode"], "full_file_auto_with_stream_fallback")
        self.assertAlmostEqual(float(payload["aggregates"]["micro_acc"]), 1.0)


if __name__ == "__main__":
    unittest.main()
