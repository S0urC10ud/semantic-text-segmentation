from __future__ import annotations

import json
import sys
import tempfile
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


def _monitor_data_with_other() -> dict:
    chunks = [
        np.frombuffer(b"abc\n", dtype=np.uint8),
        np.frombuffer(b"TEXT", dtype=np.uint8),
    ]
    files = np.zeros(2, dtype=FILE_DTYPE)
    segments = np.zeros(2, dtype=SEG_DTYPE)
    offset = 0
    labels = [
        int(evalmod.cfg.LANG2ID["python"]),
        int(evalmod.cfg.OTHER_CLASS_INDEX),
    ]
    for idx, (arr, label) in enumerate(zip(chunks, labels)):
        files[idx] = (
            offset,
            int(arr.shape[0]),
            idx,
            1,
            0,
            label,
        )
        segments[idx] = (
            idx,
            0,
            int(arr.shape[0]),
            label,
        )
        offset += int(arr.shape[0])
    return {
        "meta": {"num_files": 2},
        "files": files,
        "segments": segments,
        "contents": np.concatenate(chunks, axis=0),
    }


def _monitor_data_svg_host_with_xml_tail() -> dict:
    chunk = np.frombuffer(b"SSSSSSSSSSXXXX", dtype=np.uint8)
    files = np.zeros(1, dtype=FILE_DTYPE)
    segments = np.zeros(2, dtype=SEG_DTYPE)
    files[0] = (
        0,
        int(chunk.shape[0]),
        0,
        2,
        0,
        int(evalmod.cfg.LANG2ID["svg"]),
    )
    segments[0] = (
        0,
        0,
        10,
        int(evalmod.cfg.LANG2ID["svg"]),
    )
    segments[1] = (
        0,
        10,
        int(chunk.shape[0]),
        int(evalmod.cfg.LANG2ID["xml"]),
    )
    return {
        "meta": {"num_files": 1},
        "files": files,
        "segments": segments,
        "contents": chunk,
    }


def _monitor_data_with_transitions() -> dict:
    chunk = np.frombuffer(b"ABCDEFGHIJKLMN", dtype=np.uint8)
    files = np.zeros(1, dtype=FILE_DTYPE)
    segments = np.zeros(3, dtype=SEG_DTYPE)
    python_id = int(evalmod.cfg.LANG2ID["python"])
    shell_id = int(evalmod.cfg.LANG2ID["shell"])
    js_id = int(evalmod.cfg.LANG2ID["javascript_typescript"])
    files[0] = (
        0,
        int(chunk.shape[0]),
        0,
        3,
        0,
        python_id,
    )
    segments[0] = (0, 0, 5, python_id)
    segments[1] = (0, 5, 10, shell_id)
    segments[2] = (0, 10, int(chunk.shape[0]), js_id)
    return {
        "meta": {"num_files": 1},
        "files": files,
        "segments": segments,
        "contents": chunk,
    }


class _FakeUnetMonitorRunner:
    arch = "unet1d"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _segment_bytes_legacy(self, byte_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append("legacy")
        content = np.asarray(byte_arr, dtype=np.uint8).tobytes()
        label = "python" if content.startswith(b"abc") else "text"
        label_id = int(evalmod.cfg.LANG2ID[label])
        length = int(byte_arr.shape[0])
        labels = np.full((length,), label_id, dtype=np.uint8)
        probs = np.zeros((length, int(evalmod.cfg.NUM_CLASSES)), dtype=np.float32)
        probs[:, label_id] = 1.0
        return labels, probs

    def _segment_bytes(self, byte_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._segment_bytes_legacy(byte_arr)

    def _segment_bytes_labels_only(self, byte_arr: np.ndarray) -> np.ndarray:
        self.calls.append("fast")
        return np.zeros((int(byte_arr.shape[0]),), dtype=np.uint8)

    def _segment_bytes_labels_only_legacy(self, byte_arr: np.ndarray) -> np.ndarray:
        labels, _ = self._segment_bytes_legacy(byte_arr)
        return labels


class _FakeMambaMonitorRunner:
    arch = "mamba"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _segment_bytes(self, byte_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append("normal")
        content = np.asarray(byte_arr, dtype=np.uint8).tobytes()
        label = "python" if content.startswith(b"abc") else "text"
        label_id = int(evalmod.cfg.LANG2ID[label])
        length = int(byte_arr.shape[0])
        labels = np.full((length,), label_id, dtype=np.uint8)
        probs = np.zeros((length, int(evalmod.cfg.NUM_CLASSES)), dtype=np.float32)
        probs[:, label_id] = 1.0
        return labels, probs

    def _segment_bytes_labels_only(self, byte_arr: np.ndarray) -> np.ndarray:
        labels, _ = self._segment_bytes(byte_arr)
        return labels

    def _segment_bytes_labels_only_legacy(self, byte_arr: np.ndarray) -> np.ndarray:
        self.calls.append("legacy")
        return np.zeros((int(byte_arr.shape[0]),), dtype=np.uint8)


class _FakeUnetOpenSetRunner:
    arch = "unet1d"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _segment_bytes_legacy(self, byte_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append("legacy")
        content = np.asarray(byte_arr, dtype=np.uint8).tobytes()
        length = int(byte_arr.shape[0])
        probs = np.zeros((length, int(evalmod.cfg.NUM_CLASSES)), dtype=np.float32)
        python_id = int(evalmod.cfg.LANG2ID["python"])
        text_id = int(evalmod.cfg.LANG2ID["text"])
        if content.startswith(b"abc"):
            probs[:, python_id] = 0.95
            probs[:, text_id] = 0.05
            labels = np.full((length,), python_id, dtype=np.uint8)
        else:
            probs[:, text_id] = 0.55
            probs[:, python_id] = 0.45
            labels = np.full((length,), text_id, dtype=np.uint8)
        return labels, probs


class _FakeSvgMonitorRunner:
    arch = "unet1d"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _segment_bytes_legacy(self, byte_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append("legacy")
        length = int(byte_arr.shape[0])
        label_id = int(evalmod.cfg.LANG2ID["svg"])
        labels = np.full((length,), label_id, dtype=np.uint8)
        probs = np.zeros((length, int(evalmod.cfg.NUM_CLASSES)), dtype=np.float32)
        probs[:, label_id] = 1.0
        return labels, probs


class _FakeBoundaryMonitorRunner:
    arch = "unet1d"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._labels = np.asarray(
            [
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["javascript_typescript"]),
                int(evalmod.cfg.LANG2ID["javascript_typescript"]),
                int(evalmod.cfg.LANG2ID["javascript_typescript"]),
            ],
            dtype=np.uint8,
        )

    def _segment_bytes_legacy(self, byte_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append("legacy")
        length = int(byte_arr.shape[0])
        labels = self._labels.copy()
        if int(labels.shape[0]) != length:
            raise AssertionError(f"expected {int(labels.shape[0])} bytes, got {length}")
        probs = np.zeros((length, int(evalmod.cfg.NUM_CLASSES)), dtype=np.float32)
        probs[np.arange(length), labels.astype(np.int32)] = 1.0
        return labels, probs


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

    def test_full_monitor_b_applies_open_set_threshold_and_scores_true_other(self) -> None:
        runner = _FakeUnetOpenSetRunner()
        with mock.patch.object(evalmod, "load_monitor_memmaps", return_value=_monitor_data_with_other()):
            payload = evalmod._evaluate_full_monitor_b(
                Path("/tmp/monitor_b"),
                runner,
                other_threshold=0.60,
            )

        self.assertEqual(runner.calls, ["legacy", "legacy"])
        self.assertTrue(bool(payload["open_set"]))
        self.assertAlmostEqual(float(payload["other_threshold"]), 0.60)
        self.assertEqual(int(payload["truth_other_bytes"]), 4)
        self.assertEqual(int(payload["low_confidence_bytes_routed_to_other"]), 4)
        self.assertIn("other", payload["by_label"])
        self.assertEqual(int(payload["by_label"]["other"]["support"]), 4)
        self.assertAlmostEqual(float(payload["aggregates"]["micro_acc"]), 1.0)

        confusion_payload = evalmod._monitor_b_confusion_payload(payload)
        self.assertIsNotNone(confusion_payload)
        _, label_names = confusion_payload  # type: ignore[misc]
        self.assertIn("other", label_names)

    def test_full_monitor_b_applies_relevant_content_postprocessing(self) -> None:
        runner = _FakeSvgMonitorRunner()
        with mock.patch.object(evalmod, "load_monitor_memmaps", return_value=_monitor_data_svg_host_with_xml_tail()):
            payload = evalmod._evaluate_full_monitor_b(Path("/tmp/monitor_b"), runner)

        self.assertEqual(runner.calls, ["legacy"])
        self.assertAlmostEqual(float(payload["aggregates"]["micro_acc"]), 1.0)
        self.assertNotIn("xml", payload["by_label"])
        self.assertEqual(int(payload["by_label"]["svg"]["support"]), 14)

    def test_full_monitor_b_boundary_region_metrics_flow_to_json_and_report(self) -> None:
        runner = _FakeBoundaryMonitorRunner()
        with mock.patch.object(evalmod, "load_monitor_memmaps", return_value=_monitor_data_with_transitions()):
            payload = evalmod._evaluate_full_monitor_b(Path("/tmp/monitor_b"), runner)

        boundary = payload["boundary_region"]
        self.assertEqual(runner.calls, ["legacy"])
        self.assertIsInstance(boundary, dict)
        self.assertEqual(int(boundary["window_radius_tokens"]), 4)
        self.assertEqual(int(boundary["support_chars"]), 13)
        self.assertEqual(int(boundary["samples"]), 1)
        self.assertEqual(int(boundary["boundaries"]), 2)
        self.assertAlmostEqual(float(boundary["aggregates"]["micro_acc"]), 11.0 / 13.0)
        self.assertAlmostEqual(float(boundary["aggregates"]["macro_precision"]), 13.0 / 15.0)
        self.assertAlmostEqual(float(boundary["aggregates"]["macro_recall"]), 17.0 / 20.0)
        self.assertAlmostEqual(
            float(boundary["aggregates"]["macro_f1"]),
            (8.0 / 9.0 + 4.0 / 5.0 + 6.0 / 7.0) / 3.0,
        )

        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            chunk=1536,
            batch_size=128,
            max_samples=0,
            sample_seed=13,
            other_threshold=0.0,
            fine_tuned_mode=True,
        )
        manifest = {"output_root": "evaluation/data_b", "generated_at": "2026-03-24T00:00:00"}

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            evalmod.write_report(
                report_path,
                manifest=manifest,
                args=args,
                task_metrics=[],
                throughput_results=[],
                monitor_b_report=payload,
            )

            report = report_path.read_text(encoding="utf-8")
            comparison = json.loads((report_path.parent / "comparison_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("Boundary-region label metrics (`±4` tokens around effective truth transitions):", report)
            self.assertIn("| ALL (agg) |  | 0.8462 | 0.8667 | 0.8500 | 0.8487 |", report)
            self.assertIn("monitor_b", comparison)
            self.assertIn("boundary_region", comparison["monitor_b"])
            self.assertEqual(int(comparison["monitor_b"]["boundary_region"]["boundaries"]), 2)

    def test_write_report_emits_monitor_b_confusion_artifact(self) -> None:
        runner = _FakeUnetMonitorRunner()
        with mock.patch.object(evalmod, "load_monitor_memmaps", return_value=_monitor_data()):
            payload = evalmod._evaluate_full_monitor_b(Path("/tmp/monitor_b"), runner)

        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            chunk=1536,
            batch_size=128,
            max_samples=0,
            sample_seed=13,
            other_threshold=0.0,
            fine_tuned_mode=True,
        )
        manifest = {"output_root": "evaluation/data_b", "generated_at": "2026-03-24T00:00:00"}

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            evalmod.write_report(
                report_path,
                manifest=manifest,
                args=args,
                task_metrics=[],
                throughput_results=[],
                monitor_b_report=payload,
            )

            report = report_path.read_text(encoding="utf-8")
            comparison = json.loads((report_path.parent / "comparison_metrics.json").read_text(encoding="utf-8"))
            monitor_confusion_path = report_path.parent / "confusion_matrices" / "monitor_b.png"

            self.assertIn(
                "- Full monitor_b confusion matrix: [confusion_matrices/monitor_b.png](confusion_matrices/monitor_b.png)",
                report,
            )
            self.assertIn(
                "- Confusion matrix: [confusion_matrices/monitor_b.png](confusion_matrices/monitor_b.png)",
                report,
            )
            self.assertIn("- Open-set classification: True", report)
            self.assertIn("- Other threshold: 0.0000", report)
            self.assertTrue(monitor_confusion_path.exists())
            self.assertGreater(monitor_confusion_path.stat().st_size, 0)
            self.assertIn("monitor_b", comparison)
            self.assertNotIn("_confusion_matrix", comparison["monitor_b"])


if __name__ == "__main__":
    unittest.main()
