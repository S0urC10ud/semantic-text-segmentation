from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import datasets as hfds
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import evaluation.evaluation as evalmod
from train.utils.monitor_eval import FILE_DTYPE, SEG_DTYPE


def _record(*, content: str, segments: list[dict], metadata_json: str = "{}") -> dict:
    return {
        "task": "magika_test",
        "example_id": "magika-test-0001",
        "content": content,
        "segments": {
            "label": [str(seg["label"]) for seg in segments],
            "char_start": [int(seg["char_start"]) for seg in segments],
            "char_end": [int(seg["char_end"]) for seg in segments],
        },
        "source_langs": sorted({str(seg["label"]) for seg in segments}),
        "metadata_json": metadata_json,
    }


class _FakeWindowedMagikaSegmenter:
    last_init: dict | None = None

    def __init__(self, *, label_to_id, other_class_index, batch_size, window_size) -> None:
        self.label_to_id = dict(label_to_id)
        self.other_class_index = int(other_class_index)
        self.batch_size = int(batch_size)
        self.window_size = int(window_size)
        self.module_version = "1.0.2"
        self.model_name = "standard_v3_3"
        self.validation = SimpleNamespace(
            package_version=self.module_version,
            model_name=self.model_name,
        )
        self._history: list[object] = []
        type(self).last_init = {
            "label_to_id": dict(label_to_id),
            "other_class_index": int(other_class_index),
            "batch_size": int(batch_size),
            "window_size": int(window_size),
        }

    def segment_bytes(self, byte_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        length = int(np.asarray(byte_arr, dtype=np.uint8).shape[0])
        label_id = int(self.label_to_id["python"])
        labels = np.full((length,), label_id, dtype=np.int32)
        probs = np.zeros((length, self.other_class_index + 1), dtype=np.float32)
        probs[:, label_id] = 0.95
        return labels, probs

    def segment_byte_arrays_batch_labels_only(
        self,
        byte_arrays: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[list[tuple[int, int]]]]:
        labels = []
        spans = []
        for arr in byte_arrays:
            arr_np = np.asarray(arr, dtype=np.uint8)
            label_arr, _ = self.segment_bytes(arr_np)
            labels.append(label_arr)
            spans.append(evalmod.build_window_spans(int(arr_np.shape[0]), self.window_size))
        return labels, spans

    def clear_execution_history(self) -> None:
        self._history.clear()

    def get_execution_history(self) -> list[object]:
        return list(self._history)


class _HighConfidenceRunner:
    def segment_text(self, text: str, *, min_run_chars: int = 1):
        del min_run_chars
        label_id = int(evalmod.cfg.LANG2ID["python"])
        probs = np.zeros((len(text), int(evalmod.cfg.OTHER_CLASS_INDEX) + 1), dtype=np.float32)
        probs[:, label_id] = 0.95
        return [], [label_id] * len(text), [row.copy() for row in probs]


class _LowConfidenceRunner:
    def segment_text(self, text: str, *, min_run_chars: int = 1):
        del min_run_chars
        label_id = int(evalmod.cfg.LANG2ID["python"])
        probs = np.zeros((len(text), int(evalmod.cfg.OTHER_CLASS_INDEX) + 1), dtype=np.float32)
        probs[:, label_id] = 0.20
        return [], [label_id] * len(text), [row.copy() for row in probs]


def _monitor_data_with_encoding_file() -> dict:
    chunks = [
        np.frombuffer(b"abcd", dtype=np.uint8),
        np.frombuffer(b"WXYZ", dtype=np.uint8),
    ]
    files = np.zeros(2, dtype=FILE_DTYPE)
    segments = np.zeros(2, dtype=SEG_DTYPE)
    offset = 0
    python_id = int(evalmod.cfg.LANG2ID["python"])
    encoding_id = int(evalmod.cfg.LANG2ID["encoding_base64"])
    labels = [python_id, encoding_id]
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


class _FakeMonitorMagikaRunner:
    arch = "magika"

    def _segment_bytes(self, byte_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        length = int(np.asarray(byte_arr, dtype=np.uint8).shape[0])
        label_id = int(evalmod.cfg.LANG2ID["python"])
        labels = np.full((length,), label_id, dtype=np.int32)
        probs = np.zeros((length, int(evalmod.cfg.OTHER_CLASS_INDEX) + 1), dtype=np.float32)
        probs[:, label_id] = 0.9
        return labels, probs

    def _segment_bytes_legacy(self, byte_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._segment_bytes(byte_arr)


class TestEvaluationMagika(unittest.TestCase):
    def test_segmenter_runner_initializes_magika_without_checkpoint(self) -> None:
        with mock.patch.object(evalmod, "SlidingWindowMagikaSegmenter", _FakeWindowedMagikaSegmenter):
            runner = evalmod.SegmenterRunner(
                None,
                arch="magika",
                model_dim=0,
                channels=(),
                dtype="float32",
                chunk=1536,
                device="cpu",
                batch_size=32,
            )

        self.assertEqual(runner.arch, "magika")
        self.assertEqual(runner.backend, "cpu")
        self.assertEqual(runner.magika_module_version, "1.0.2")
        self.assertEqual(runner.magika_model_name, "standard_v3_3")
        self.assertEqual(_FakeWindowedMagikaSegmenter.last_init["window_size"], 1536)
        self.assertEqual(_FakeWindowedMagikaSegmenter.last_init["batch_size"], 32)

    def test_evaluate_task_skips_encoding_truth_samples(self) -> None:
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    content="ABCD",
                    segments=[{"label": "python", "char_start": 0, "char_end": 4}],
                ),
                _record(
                    content="WXYZ",
                    segments=[{"label": "encoding_base64", "char_start": 0, "char_end": 4}],
                ),
            ]
        )

        metrics = evalmod.evaluate_task(
            "pure_fragments",
            "test",
            dataset,
            _HighConfidenceRunner(),
            min_run_chars=1,
            other_threshold=0.0,
            excluded_truth_labels=["encoding_base64"],
        )

        exclusions = metrics.extras["truth_label_exclusions"]
        self.assertEqual(metrics.samples, 1)
        self.assertEqual(exclusions["requested"], 2)
        self.assertEqual(exclusions["counted"], 1)
        self.assertEqual(exclusions["skipped_for_truth_label"], 1)
        self.assertEqual(exclusions["skipped_label_counts"]["encoding_base64"], 1)
        self.assertNotIn("encoding_base64", metrics.label_names)

    def test_evaluate_task_routes_low_confidence_magika_predictions_to_other(self) -> None:
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    content="ABCD",
                    segments=[{"label": "python", "char_start": 0, "char_end": 4}],
                )
            ]
        )
        metrics = evalmod.evaluate_task(
            "sequence_pair",
            "test",
            dataset,
            _LowConfidenceRunner(),
            min_run_chars=1,
            other_threshold=0.30,
        )

        python_idx = metrics.label_names.index("python")
        other_idx = metrics.label_names.index("other")
        self.assertEqual(int(metrics.confusion[python_idx, other_idx]), 4)

    def test_full_monitor_b_skips_encoding_truth_files_for_magika(self) -> None:
        runner = _FakeMonitorMagikaRunner()
        with mock.patch.object(evalmod, "load_monitor_memmaps", return_value=_monitor_data_with_encoding_file()):
            payload = evalmod._evaluate_full_monitor_b(
                Path("/tmp/monitor_b"),
                runner,
                excluded_truth_labels=["encoding_base64"],
            )

        self.assertEqual(payload["arch"], "magika")
        self.assertEqual(payload["inference_mode"], "sliding_window_magika_rawdl")
        self.assertEqual(payload["files_used"], 1)
        self.assertEqual(payload["skipped_for_truth_label"], 1)
        exclusions = payload["truth_label_exclusions"]
        self.assertEqual(exclusions["requested"], 2)
        self.assertEqual(exclusions["counted"], 1)
        self.assertEqual(exclusions["skipped_label_counts"]["encoding_base64"], 1)

    def test_report_and_comparison_json_include_reduced_support_note(self) -> None:
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    content="ABCD",
                    segments=[{"label": "python", "char_start": 0, "char_end": 4}],
                ),
                _record(
                    content="WXYZ",
                    segments=[{"label": "encoding_base64", "char_start": 0, "char_end": 4}],
                ),
            ]
        )
        metrics = evalmod.evaluate_task(
            "sequence_pair",
            "test",
            dataset,
            _HighConfidenceRunner(),
            min_run_chars=1,
            other_threshold=0.30,
            excluded_truth_labels=list(evalmod.THESIS_ENCODING_LABELS),
        )
        args = SimpleNamespace(
            checkpoint="magika://default",
            arch="magika",
            model_dim=0,
            channels=[],
            dtype="float32",
            chunk=1536,
            batch_size=128,
            max_samples=0,
            sample_seed=13,
            other_threshold=0.30,
            exclude_truth_labels=list(evalmod.THESIS_ENCODING_LABELS),
            fine_tuned_mode=True,
            magika_module_version="1.0.2",
            magika_model_name="standard_v3_3",
        )
        manifest = {"output_root": "evaluation/data_b", "generated_at": "2026-04-09T00:00:00"}

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            evalmod.write_report(
                report_path,
                manifest=manifest,
                args=args,
                task_metrics=[metrics],
                throughput_results=[],
            )
            report_text = report_path.read_text(encoding="utf-8")
            payload = json.loads((report_path.parent / "comparison_metrics.json").read_text(encoding="utf-8"))

        self.assertIn("Reduced-support note", report_text)
        self.assertIn("Magika version: 1.0.2", report_text)
        self.assertEqual(payload["meta"]["magika_module_version"], "1.0.2")
        self.assertEqual(payload["meta"]["magika_model_name"], "standard_v3_3")
        self.assertEqual(payload["meta"]["truth_label_exclusion_note"], evalmod.MAGIKA_REDUCED_SUPPORT_NOTE)
        self.assertIn("reduced_support", payload)

    def test_magika_wrapper_script_only_dispatches_magika_run(self) -> None:
        script_path = ROOT / "scripts" / "run_eval_magika_sw1536_tau03_open_set_trimmed.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env.update(
                {
                    "REPO_DIR": str(ROOT),
                    "PYTHON_BIN": "/bin/echo",
                    "DRY_RUN": "1",
                    "TIMESTAMP": "20260409_120000",
                    "REPORTS_ROOT": str(Path(tmpdir) / "reports"),
                    "MATRIX_CSV_PATH": str(Path(tmpdir) / "matrix.csv"),
                }
            )
            result = subprocess.run(
                ["bash", str(script_path)],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        combined = result.stdout + "\n" + result.stderr
        self.assertEqual(result.returncode, 0, msg=combined)
        self.assertIn("magika_sw1536_rawdl_noenc", combined)
        self.assertIn("--arch magika", combined)
        self.assertIn("--exclude-truth-labels", combined)
        self.assertNotIn("--checkpoint", combined)


if __name__ == "__main__":
    unittest.main()
