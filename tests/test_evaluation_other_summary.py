from __future__ import annotations

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


class _FakeRunner:
    def __init__(self, labels: list[int], probs: list[np.ndarray]) -> None:
        self._labels = list(labels)
        self._probs = [np.asarray(row, dtype=np.float32) for row in probs]

    def segment_text(self, text: str, *, min_run_chars: int = 1):
        return [], list(self._labels), [row.copy() for row in self._probs]


def _record(*, content: str, segments: list[dict]) -> dict:
    return {
        "task": "other_summary_test",
        "example_id": "other-test-0001",
        "content": content,
        "segments": {
            "label": [str(seg["label"]) for seg in segments],
            "char_start": [int(seg["char_start"]) for seg in segments],
            "char_end": [int(seg["char_end"]) for seg in segments],
        },
        "source_langs": sorted({str(seg["label"]) for seg in segments}),
        "metadata_json": "{}",
    }


class TestEvaluationOtherSummary(unittest.TestCase):
    def _prob_row(self, label: str, prob: float) -> np.ndarray:
        row = np.zeros((int(evalmod.cfg.NUM_CLASSES),), dtype=np.float32)
        row[int(evalmod.cfg.LANG2ID[label])] = float(prob)
        return row

    def test_other_summary_counts_rates_and_json(self) -> None:
        content = "ABCD"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    content=content,
                    segments=[
                        {"label": "other", "char_start": 0, "char_end": 2},
                        {"label": "python", "char_start": 2, "char_end": 4},
                    ],
                )
            ]
        )
        python_id = int(evalmod.cfg.LANG2ID["python"])
        runner = _FakeRunner(
            labels=[python_id] * len(content),
            probs=[
                self._prob_row("python", 0.40),
                self._prob_row("python", 0.90),
                self._prob_row("python", 0.40),
                self._prob_row("python", 0.90),
            ],
        )

        metrics = evalmod.evaluate_task(
            "other_summary_test",
            "test",
            dataset,
            runner,
            min_run_chars=1,
            other_threshold=0.50,
        )

        summary = evalmod._aggregate_other_confusion([metrics])
        counts = summary["counts"]
        rates = summary["rates"]
        merged_confusion, merged_labels = evalmod._merge_task_confusions([metrics])
        self.assertEqual(counts["tp_other"], 1)
        self.assertEqual(counts["fn_other"], 1)
        self.assertEqual(counts["fp_other"], 1)
        self.assertEqual(counts["tn_other"], 1)
        self.assertEqual(counts["truth_other"], 2)
        self.assertEqual(counts["truth_non_other"], 2)
        self.assertEqual(counts["predicted_other"], 2)
        self.assertEqual(counts["predicted_non_other"], 2)
        self.assertAlmostEqual(float(rates["too_often_other"]), 0.5)
        self.assertAlmostEqual(float(rates["too_little_other"]), 0.5)
        self.assertAlmostEqual(float(rates["other_precision"]), 0.5)
        self.assertAlmostEqual(float(rates["other_recall"]), 0.5)
        self.assertAlmostEqual(float(rates["truth_other_rate"]), 0.5)
        self.assertAlmostEqual(float(rates["predicted_other_rate"]), 0.5)
        self.assertIn("other", merged_labels)
        self.assertEqual(merged_confusion.shape, (len(merged_labels), len(merged_labels)))

        highlights = "\n".join(evalmod._collect_task_highlights([metrics]))
        self.assertIn("##### other_open_set", highlights)
        self.assertIn("| other | 1 (50.0%) | 1 (50.0%) |", highlights)
        self.assertIn("| not other | 1 (50.0%) | 1 (50.0%) |", highlights)
        self.assertIn("Too often `other`: 50.0%. Too little `other`: 50.0%.", highlights)

        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            sample_seed=13,
            other_threshold=0.50,
        )
        payload = evalmod._collect_comparison_metrics(args, [metrics], [])
        self.assertEqual(payload["meta"]["other_threshold"], 0.5)
        self.assertEqual(payload["summary"]["other_confusion"]["counts"]["tp_other"], 1)
        self.assertAlmostEqual(
            float(payload["summary"]["other_confusion"]["rates"]["too_often_other"]),
            0.5,
        )

    def test_write_report_includes_other_threshold_header(self) -> None:
        content = "ABCD"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    content=content,
                    segments=[
                        {"label": "other", "char_start": 0, "char_end": 2},
                        {"label": "python", "char_start": 2, "char_end": 4},
                    ],
                )
            ]
        )
        python_id = int(evalmod.cfg.LANG2ID["python"])
        runner = _FakeRunner(
            labels=[python_id] * len(content),
            probs=[
                self._prob_row("python", 0.40),
                self._prob_row("python", 0.90),
                self._prob_row("python", 0.40),
                self._prob_row("python", 0.90),
            ],
        )
        metrics = evalmod.evaluate_task(
            "other_summary_test",
            "test",
            dataset,
            runner,
            min_run_chars=1,
            other_threshold=0.50,
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
            other_threshold=0.50,
        )
        manifest = {"output_root": "evaluation/data", "generated_at": "2026-03-22T00:00:00"}

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            evalmod.write_report(
                report_path,
                manifest=manifest,
                args=args,
                task_metrics=[metrics],
                throughput_results=[],
            )
            report = report_path.read_text(encoding="utf-8")
            comparison_path = report_path.parent / "comparison_metrics.json"
            task_confusion_path = report_path.parent / "confusion_matrices" / "other_summary_test.png"
            all_confusion_path = report_path.parent / "confusion_matrices" / "all_tasks.png"
            self.assertIn("- Other threshold: 0.5000", report)
            self.assertIn("- Comparison JSON: [comparison_metrics.json](comparison_metrics.json)", report)
            self.assertIn("- Confusion matrix: [confusion_matrices/other_summary_test.png](confusion_matrices/other_summary_test.png)", report)
            self.assertIn("- Aggregated confusion matrix: [confusion_matrices/all_tasks.png](confusion_matrices/all_tasks.png)", report)
            self.assertIn("##### other_open_set", report)
            self.assertTrue(comparison_path.exists())
            self.assertGreater(comparison_path.stat().st_size, 0)
            self.assertTrue(task_confusion_path.exists())
            self.assertGreater(task_confusion_path.stat().st_size, 0)
            self.assertTrue(all_confusion_path.exists())
            self.assertGreater(all_confusion_path.stat().st_size, 0)

    def test_resolve_report_path_defaults_to_reports_run_directory(self) -> None:
        args = SimpleNamespace(
            checkpoint="checkpoints/sweeps/sfullfiles3.msgpack",
            data_root=str(ROOT / "evaluation" / "data"),
            tasks=None,
            report_path=None,
        )
        with mock.patch.object(evalmod.time, "strftime", return_value="20260322_090000"):
            report_path = evalmod._resolve_report_path(args)

        expected_dir = (
            evalmod.REPO_ROOT
            / "evaluation"
            / "reports"
            / "sfullfiles3__data__20260322_090000"
        )
        self.assertEqual(report_path, expected_dir / "report.md")


if __name__ == "__main__":
    unittest.main()
