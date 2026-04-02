from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


def _prob_row(label: str) -> np.ndarray:
    row = np.zeros((int(evalmod.cfg.NUM_CLASSES),), dtype=np.float32)
    row[int(evalmod.cfg.LANG2ID[label])] = 1.0
    return row


def _record(
    *,
    task: str,
    content: str,
    segments: list[dict],
    metadata: dict,
) -> dict:
    return {
        "task": task,
        "example_id": f"{task}-test-0001",
        "content": content,
        "segments": {
            "label": [str(seg["label"]) for seg in segments],
            "char_start": [int(seg["char_start"]) for seg in segments],
            "char_end": [int(seg["char_end"]) for seg in segments],
        },
        "source_langs": sorted({str(seg["label"]) for seg in segments}),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


class TestEvaluationRegionTasks(unittest.TestCase):
    def test_text_relabels_to_markup_with_sufficient_evidence(self) -> None:
        content = "MMMMMMMMMMTTTT"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="postprocess_text_markup",
                    content=content,
                    segments=[
                        {"label": "markdown", "char_start": 0, "char_end": 10},
                        {"label": "text", "char_start": 10, "char_end": 14},
                    ],
                    metadata={},
                )
            ]
        )
        runner = _FakeRunner(
            labels=[int(evalmod.cfg.LANG2ID["markdown"])] * len(content),
            probs=[_prob_row("markdown") for _ in content],
        )

        metrics = evalmod.evaluate_task(
            "postprocess_text_markup",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        self.assertAlmostEqual(metrics.overall_accuracy(), 1.0)
        self.assertEqual(int(metrics.per_label_counts.get("text", 0)), 0)
        self.assertEqual(int(metrics.per_label_counts.get("markdown", 0)), len(content))

    def test_text_relabels_to_markup_with_more_evidence(self) -> None:
        content = "MMMMMMMMMMRRRRRRRRRRRRTTTT"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="postprocess_text_dominant_markup",
                    content=content,
                    segments=[
                        {"label": "markdown", "char_start": 0, "char_end": 10},
                        {"label": "restructuredtext", "char_start": 10, "char_end": 22},
                        {"label": "text", "char_start": 22, "char_end": 26},
                    ],
                    metadata={},
                )
            ]
        )
        runner = _FakeRunner(
            labels=(
                [int(evalmod.cfg.LANG2ID["markdown"])] * 10
                + [int(evalmod.cfg.LANG2ID["restructuredtext"])] * 16
            ),
            probs=(
                [_prob_row("markdown") for _ in range(10)]
                + [_prob_row("restructuredtext") for _ in range(16)]
            ),
        )

        metrics = evalmod.evaluate_task(
            "postprocess_text_dominant_markup",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        self.assertAlmostEqual(metrics.overall_accuracy(), 1.0)
        self.assertEqual(int(metrics.per_label_counts.get("text", 0)), 0)
        self.assertEqual(int(metrics.per_label_counts.get("restructuredtext", 0)), 16)

    def test_text_relabel_requires_ten_tokens_of_host_evidence(self) -> None:
        content = "MMMMMMMMMTTTT"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="postprocess_text_threshold",
                    content=content,
                    segments=[
                        {"label": "markdown", "char_start": 0, "char_end": 9},
                        {"label": "text", "char_start": 9, "char_end": 13},
                    ],
                    metadata={},
                )
            ]
        )
        runner = _FakeRunner(
            labels=[int(evalmod.cfg.LANG2ID["markdown"])] * len(content),
            probs=[_prob_row("markdown") for _ in content],
        )

        metrics = evalmod.evaluate_task(
            "postprocess_text_threshold",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        self.assertAlmostEqual(metrics.overall_accuracy(), 9.0 / 13.0)
        self.assertEqual(int(metrics.per_label_counts.get("text", 0)), 4)

    def test_json_relabels_to_javascript_typescript_with_wrapping_evidence(self) -> None:
        content = "JJJJJJJJJJQQQQJJJJJJJJJJ"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="postprocess_json_js",
                    content=content,
                    segments=[
                        {"label": "javascript_typescript", "char_start": 0, "char_end": 10},
                        {"label": "json", "char_start": 10, "char_end": 14},
                        {"label": "javascript_typescript", "char_start": 14, "char_end": 24},
                    ],
                    metadata={},
                )
            ]
        )
        runner = _FakeRunner(
            labels=[int(evalmod.cfg.LANG2ID["javascript_typescript"])] * len(content),
            probs=[_prob_row("javascript_typescript") for _ in content],
        )

        metrics = evalmod.evaluate_task(
            "postprocess_json_js",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        self.assertAlmostEqual(metrics.overall_accuracy(), 1.0)
        self.assertEqual(int(metrics.per_label_counts.get("json", 0)), 0)
        self.assertEqual(int(metrics.per_label_counts.get("javascript_typescript", 0)), len(content))

    def test_xml_relabels_to_svg_with_wrapping_evidence(self) -> None:
        content = "SSSSSSSSSSXXXXSSSSSSSSSS"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="postprocess_xml_svg",
                    content=content,
                    segments=[
                        {"label": "svg", "char_start": 0, "char_end": 10},
                        {"label": "xml", "char_start": 10, "char_end": 14},
                        {"label": "svg", "char_start": 14, "char_end": 24},
                    ],
                    metadata={},
                )
            ]
        )
        runner = _FakeRunner(
            labels=[int(evalmod.cfg.LANG2ID["svg"])] * len(content),
            probs=[_prob_row("svg") for _ in content],
        )

        metrics = evalmod.evaluate_task(
            "postprocess_xml_svg",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        self.assertAlmostEqual(metrics.overall_accuracy(), 1.0)
        self.assertEqual(int(metrics.per_label_counts.get("xml", 0)), 0)
        self.assertEqual(int(metrics.per_label_counts.get("svg", 0)), len(content))

    def test_needle_exact_region_uses_inserted_span_not_donor_label_mask(self) -> None:
        content = "HHABCD"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="needle_4_15",
                    content=content,
                    segments=[
                        {"label": "sql", "char_start": 0, "char_end": 2},
                        {"label": "python", "char_start": 2, "char_end": 4},
                        {"label": "shell", "char_start": 4, "char_end": 6},
                    ],
                    metadata={
                        "host_lang": "sql",
                        "donor_lang": "python",
                        "inserted_char_start": 2,
                        "inserted_char_end": 6,
                    },
                )
            ]
        )
        runner = _FakeRunner(
            labels=[
                int(evalmod.cfg.LANG2ID["sql"]),
                int(evalmod.cfg.LANG2ID["sql"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
            ],
            probs=[_prob_row("sql"), _prob_row("sql"), _prob_row("python"), _prob_row("python"), _prob_row("shell"), _prob_row("shell")],
        )

        metrics = evalmod.evaluate_task(
            "needle_4_15",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        needle_stats = metrics.extras["needle_detection"]
        self.assertEqual(int(needle_stats["needle_chars"]), 4)
        self.assertEqual(int(needle_stats["correct_chars"]), 4)
        self.assertEqual(int(needle_stats["detected"]), 1)
        self.assertEqual(int(needle_stats["by_lang"]["python"]["truth_chars"]), 4)
        self.assertEqual(int(needle_stats["any_detection"]["correct_chars"]), 4)

    def test_needle_report_and_comparison_payload_expose_donor_support_weighting(self) -> None:
        content = "HHABCD"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="needle_4_15",
                    content=content,
                    segments=[
                        {"label": "sql", "char_start": 0, "char_end": 2},
                        {"label": "python", "char_start": 2, "char_end": 4},
                        {"label": "shell", "char_start": 4, "char_end": 6},
                    ],
                    metadata={
                        "host_lang": "sql",
                        "donor_lang": "python",
                        "inserted_char_start": 2,
                        "inserted_char_end": 6,
                    },
                )
            ]
        )
        runner = _FakeRunner(
            labels=[
                int(evalmod.cfg.LANG2ID["sql"]),
                int(evalmod.cfg.LANG2ID["sql"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
            ],
            probs=[_prob_row("sql"), _prob_row("sql"), _prob_row("python"), _prob_row("python"), _prob_row("shell"), _prob_row("shell")],
        )

        metrics = evalmod.evaluate_task(
            "needle_4_15",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )
        manifest = {
            "output_root": "evaluation/data",
            "generated_at": "2026-03-22T00:00:00",
            "tasks": [
                {
                    "task": "needle_4_15",
                    "requested_count": 1,
                    "requested_per_anchor_label": 1,
                    "actual_count": 1,
                    "actual_by_anchor_label": {"sql": 1},
                    "candidate_regions_by_anchor_label": {"sql": 1},
                    "shortfall_by_anchor_label": {"sql": 0},
                    "donor_floor_visible_chars": 100,
                    "qualified_donor_labels": ["python"],
                    "donor_candidate_visible_support_by_donor_label": {"python": 128},
                    "donor_selected_visible_support_by_donor_label": {"python": 4},
                    "donor_support_shortfall_by_donor_label": {"python": 0},
                }
            ],
        }
        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            sample_seed=13,
            other_threshold=0.0,
            chunk=1536,
            batch_size=8,
            max_samples=0,
        )

        payload = evalmod._collect_comparison_metrics(args, [metrics], [], manifest)
        needle_payload = payload["tasks"]["needle"]["needle_4_15"]
        self.assertEqual(
            needle_payload["coverage_weighting"],
            {"mode": "micro", "unit": "inserted_chars"},
        )
        self.assertIn("per_donor_language", needle_payload)
        self.assertIn("python", needle_payload["per_donor_language"])

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
            self.assertIn("support-weighted over inserted-region chars", report)
            self.assertIn("Label support below counts all evaluated non-whitespace characters", report)
            self.assertIn("| Donor label | Samples | Selected visible chars | Qualified |", report)

    def test_needle_boundary_region_metrics_flow_to_json_and_report(self) -> None:
        content = "HHHHHHHHPPPPHHHHHHHH"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="needle_4_15",
                    content=content,
                    segments=[
                        {"label": "sql", "char_start": 0, "char_end": 8},
                        {"label": "python", "char_start": 8, "char_end": 12},
                        {"label": "sql", "char_start": 12, "char_end": 20},
                    ],
                    metadata={
                        "host_lang": "sql",
                        "donor_lang": "python",
                        "inserted_char_start": 8,
                        "inserted_char_end": 12,
                    },
                )
            ]
        )
        runner = _FakeRunner(
            labels=(
                [int(evalmod.cfg.LANG2ID["sql"])] * 10
                + [int(evalmod.cfg.LANG2ID["python"])] * 2
                + [int(evalmod.cfg.LANG2ID["sql"])] * 2
                + [int(evalmod.cfg.LANG2ID["python"])]
                + [int(evalmod.cfg.LANG2ID["sql"])] * 5
            ),
            probs=(
                [_prob_row("sql") for _ in range(10)]
                + [_prob_row("python") for _ in range(2)]
                + [_prob_row("sql") for _ in range(2)]
                + [_prob_row("python")]
                + [_prob_row("sql") for _ in range(5)]
            ),
        )

        metrics = evalmod.evaluate_task(
            "needle_4_15",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )
        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            sample_seed=13,
            other_threshold=0.0,
            chunk=1536,
            batch_size=8,
            max_samples=0,
        )
        manifest = {
            "output_root": "evaluation/data",
            "generated_at": "2026-03-22T00:00:00",
            "tasks": [],
        }

        payload = evalmod._collect_comparison_metrics(args, [metrics], [], manifest)
        boundary_payload = payload["tasks"]["needle"]["needle_4_15"]["boundary_region"]
        self.assertEqual(int(boundary_payload["window_radius_tokens"]), 4)
        self.assertEqual(int(boundary_payload["support_chars"]), 12)
        self.assertEqual(int(boundary_payload["samples"]), 1)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["micro_acc"]), 0.75)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["macro_precision"]), 13.0 / 18.0)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["macro_recall"]), 11.0 / 16.0)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["macro_f1"]), 83.0 / 119.0)

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
            self.assertIn("Boundary-region label metrics (`±4` tokens around inserted start/end):", report)
            self.assertIn("| ALL (agg) |  | 0.7500 | 0.7222 | 0.6875 | 0.6975 |", report)

    def test_sequence_regions_ignore_same_label_outside_region(self) -> None:
        content = "AABBP"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="sequence_pair",
                    content=content,
                    segments=[
                        {"label": "python", "char_start": 0, "char_end": 2},
                        {"label": "shell", "char_start": 2, "char_end": 4},
                        {"label": "python", "char_start": 4, "char_end": 5},
                    ],
                    metadata={
                        "first_lang": "python",
                        "second_lang": "shell",
                        "sequence_regions": {
                            "first": {"char_start": 0, "char_end": 2},
                            "second": {"char_start": 2, "char_end": 4},
                        },
                    },
                )
            ]
        )
        runner = _FakeRunner(
            labels=[
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
            ],
            probs=[
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("shell"),
                _prob_row("shell"),
                _prob_row("shell"),
            ],
        )

        metrics = evalmod.evaluate_task(
            "sequence_pair",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        seq_stats = metrics.extras["sequence_purity"]["segments"]
        self.assertEqual(int(seq_stats["first"]["total"]), 2)
        self.assertEqual(int(seq_stats["first"]["correct"]), 2)
        self.assertEqual(int(seq_stats["second"]["total"]), 2)
        self.assertEqual(int(seq_stats["second"]["correct"]), 2)

    def test_sequence_pair_boundary_region_metrics_flow_to_json_and_report(self) -> None:
        content = "PPPPPPSSSSSS"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="sequence_pair",
                    content=content,
                    segments=[
                        {"label": "python", "char_start": 0, "char_end": 6},
                        {"label": "shell", "char_start": 6, "char_end": 12},
                    ],
                    metadata={
                        "first_lang": "python",
                        "second_lang": "shell",
                        "sequence_regions": {
                            "first": {"char_start": 0, "char_end": 6},
                            "second": {"char_start": 6, "char_end": 12},
                        },
                    },
                )
            ]
        )
        runner = _FakeRunner(
            labels=[
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
            ],
            probs=[
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("shell"),
                _prob_row("python"),
                _prob_row("shell"),
                _prob_row("shell"),
                _prob_row("shell"),
                _prob_row("shell"),
                _prob_row("shell"),
            ],
        )

        metrics = evalmod.evaluate_task(
            "sequence_pair",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )
        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            sample_seed=13,
            other_threshold=0.0,
            chunk=1536,
            batch_size=8,
            max_samples=0,
        )
        manifest = {
            "output_root": "evaluation/data",
            "generated_at": "2026-03-22T00:00:00",
            "tasks": [],
        }

        payload = evalmod._collect_comparison_metrics(args, [metrics], [], manifest)
        boundary_payload = payload["tasks"]["sequence_pair"]["boundary_region"]
        self.assertEqual(int(boundary_payload["window_radius_tokens"]), 4)
        self.assertEqual(int(boundary_payload["support_chars"]), 8)
        self.assertEqual(int(boundary_payload["samples"]), 1)
        self.assertEqual(int(boundary_payload["boundaries"]), 1)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["micro_acc"]), 0.75)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["macro_precision"]), 0.75)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["macro_recall"]), 0.75)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["macro_f1"]), 0.75)

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
            self.assertIn("Boundary-region label metrics (`±4` tokens around sequence transitions):", report)
            self.assertIn("| ALL (agg) |  | 0.7500 | 0.7500 | 0.7500 | 0.7500 |", report)

    def test_markdown_exact_region_counts_full_mixed_block_and_support_surfaces(self) -> None:
        content = "MMABCDTT"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="markdown_mix",
                    content=content,
                    segments=[
                        {"label": "markdown", "char_start": 0, "char_end": 2},
                        {"label": "python", "char_start": 2, "char_end": 4},
                        {"label": "shell", "char_start": 4, "char_end": 6},
                        {"label": "markdown", "char_start": 6, "char_end": 8},
                    ],
                    metadata={
                        "markdown_blocks": [
                            {
                                "role": "other",
                                "wrapped": False,
                                "language": "python",
                                "char_start": 2,
                                "char_end": 6,
                                "truth_mode": "exact_region",
                            }
                        ],
                        "inline_blocks": [],
                    },
                )
            ]
        )
        runner = _FakeRunner(
            labels=[
                int(evalmod.cfg.LANG2ID["markdown"]),
                int(evalmod.cfg.LANG2ID["markdown"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["text"]),
                int(evalmod.cfg.LANG2ID["text"]),
            ],
            probs=[
                _prob_row("markdown"),
                _prob_row("markdown"),
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("shell"),
                _prob_row("shell"),
                _prob_row("text"),
                _prob_row("text"),
            ],
        )

        metrics = evalmod.evaluate_task(
            "markdown_mix",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        md_stats = metrics.extras["markdown_segments"]["overall"]["plain"]
        self.assertEqual(int(md_stats["truth_chars"]), 4)
        self.assertEqual(int(md_stats["correct_chars"]), 4)
        self.assertEqual(int(md_stats["detected_correct"]), 1)
        text_like_binary = metrics.extras["markdown_segments"]["text_like_binary"]
        self.assertIn("tex", text_like_binary["positive_labels"])
        self.assertAlmostEqual(float(text_like_binary["by_label"]["text_like"]["f1"]), 1.0)

        manifest = {
            "output_root": "evaluation/data",
            "generated_at": "2026-03-22T00:00:00",
            "tasks": [
                {
                    "task": "markdown_mix",
                    "requested_count": 10,
                    "actual_count": 7,
                    "actual_by_anchor_label": {"python": 4, "shell": 3},
                    "candidate_regions_by_anchor_label": {"python": 4, "shell": 3},
                    "shortfall_by_anchor_label": {"python": 1, "shell": 2},
                }
            ],
        }
        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            sample_seed=13,
            other_threshold=0.0,
            chunk=1536,
            batch_size=8,
            max_samples=0,
        )
        payload = evalmod._collect_comparison_metrics(args, [metrics], [], manifest)
        self.assertEqual(payload["tasks"]["markdown_mix"]["support_summary"]["requested_count"], 10)
        self.assertIn("text_like_binary", payload["tasks"]["markdown_mix"])
        self.assertAlmostEqual(
            float(payload["tasks"]["markdown_mix"]["text_like_binary"]["by_label"]["text_like"]["precision"]),
            1.0,
        )

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
            report_section = report.split("### markdown_mix", 1)[1]
            self.assertIn("Support: 7/10 requested examples.", report)
            self.assertIn("Shortfall by anchor: shell -2, python -1.", report)
            self.assertIn("Binary text-like metrics", report)
            self.assertIn("| text_like | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |", report)
            self.assertIn("- Characters evaluated: 4", report_section)
            self.assertIn("- Overall accuracy: 1.0000", report_section)
            self.assertNotIn("| markdown |", report_section)
            self.assertNotIn("| text |", report_section)

    def test_markdown_boundary_region_metrics_flow_to_json_and_report(self) -> None:
        content = "MMPPSSMM"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="markdown_mix",
                    content=content,
                    segments=[
                        {"label": "markdown", "char_start": 0, "char_end": 2},
                        {"label": "python", "char_start": 2, "char_end": 4},
                        {"label": "shell", "char_start": 4, "char_end": 6},
                        {"label": "markdown", "char_start": 6, "char_end": 8},
                    ],
                    metadata={
                        "markdown_blocks": [
                            {
                                "role": "other",
                                "wrapped": False,
                                "language": "python",
                                "char_start": 2,
                                "char_end": 4,
                                "truth_mode": "exact_region",
                            }
                        ],
                        "inline_blocks": [
                            {
                                "language": "shell",
                                "char_start": 4,
                                "char_end": 6,
                                "truth_mode": "exact_region",
                            }
                        ],
                    },
                )
            ]
        )
        runner = _FakeRunner(
            labels=[
                int(evalmod.cfg.LANG2ID["markdown"]),
                int(evalmod.cfg.LANG2ID["markdown"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["markdown"]),
                int(evalmod.cfg.LANG2ID["markdown"]),
            ],
            probs=[
                _prob_row("markdown"),
                _prob_row("markdown"),
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("shell"),
                _prob_row("markdown"),
                _prob_row("markdown"),
            ],
        )

        metrics = evalmod.evaluate_task(
            "markdown_mix",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )
        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            sample_seed=13,
            other_threshold=0.0,
            chunk=1536,
            batch_size=8,
            max_samples=0,
        )
        manifest = {
            "output_root": "evaluation/data",
            "generated_at": "2026-03-22T00:00:00",
            "tasks": [],
        }

        payload = evalmod._collect_comparison_metrics(args, [metrics], [], manifest)
        boundary_payload = payload["tasks"]["markdown_mix"]["boundary_region"]
        self.assertEqual(int(boundary_payload["window_radius_tokens"]), 4)
        self.assertEqual(int(boundary_payload["support_chars"]), 8)
        self.assertEqual(int(boundary_payload["samples"]), 1)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["micro_acc"]), 7.0 / 8.0)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["macro_precision"]), 8.0 / 9.0)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["macro_recall"]), 5.0 / 6.0)
        self.assertAlmostEqual(float(boundary_payload["aggregates"]["macro_f1"]), 37.0 / 45.0)

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
            self.assertIn("Boundary-region label metrics (`±4` tokens around markdown/code transitions):", report)
            self.assertIn("| ALL (agg) |  | 0.8750 | 0.8889 | 0.8333 | 0.8222 |", report)

    def test_markdown_report_filters_truth_labels_but_keeps_prediction_targets_in_confusions(self) -> None:
        content = "MMPPSSTT"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="markdown_mix",
                    content=content,
                    segments=[
                        {"label": "markdown", "char_start": 0, "char_end": 2},
                        {"label": "python", "char_start": 2, "char_end": 4},
                        {"label": "shell", "char_start": 4, "char_end": 6},
                        {"label": "text", "char_start": 6, "char_end": 8},
                    ],
                    metadata={
                        "markdown_blocks": [
                            {
                                "role": "other",
                                "wrapped": False,
                                "language": "python",
                                "char_start": 2,
                                "char_end": 6,
                                "truth_mode": "exact_region",
                            }
                        ],
                        "inline_blocks": [],
                    },
                )
            ]
        )
        runner = _FakeRunner(
            labels=[
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["markdown"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["text"]),
                int(evalmod.cfg.LANG2ID["shell"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
            ],
            probs=[
                _prob_row("shell"),
                _prob_row("shell"),
                _prob_row("markdown"),
                _prob_row("python"),
                _prob_row("text"),
                _prob_row("shell"),
                _prob_row("python"),
                _prob_row("python"),
            ],
        )

        metrics = evalmod.evaluate_task(
            "markdown_mix",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            sample_seed=13,
            other_threshold=0.0,
            chunk=1536,
            batch_size=8,
            max_samples=0,
        )
        manifest = {
            "output_root": "evaluation/data",
            "generated_at": "2026-03-22T00:00:00",
            "tasks": [],
        }

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
            report_section = report.split("### markdown_mix", 1)[1]
            self.assertIn("- Characters evaluated: 4", report_section)
            self.assertIn("- Overall accuracy: 0.5000", report_section)
            self.assertIn(
                "- High confusions: python->markdown 1 (50.0%), shell->text 1 (50.0%)",
                report_section,
            )
            self.assertIn("| python | 2 | 0.5000 |", report_section)
            self.assertIn("| shell | 2 | 0.5000 |", report_section)
            self.assertNotIn("| markdown |", report_section)
            self.assertNotIn("| text |", report_section)

    def test_restructuredtext_comparison_metrics_include_text_like_binary(self) -> None:
        content = "RRABCDTT"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    task="restructuredtext_mix",
                    content=content,
                    segments=[
                        {"label": "restructuredtext", "char_start": 0, "char_end": 2},
                        {"label": "python", "char_start": 2, "char_end": 6},
                        {"label": "text", "char_start": 6, "char_end": 8},
                    ],
                    metadata={
                        "markdown_blocks": [
                            {
                                "role": "other",
                                "wrapped": False,
                                "language": "python",
                                "char_start": 2,
                                "char_end": 6,
                                "truth_mode": "exact_region",
                            }
                        ],
                        "inline_blocks": [],
                    },
                )
            ]
        )
        runner = _FakeRunner(
            labels=[
                int(evalmod.cfg.LANG2ID["restructuredtext"]),
                int(evalmod.cfg.LANG2ID["restructuredtext"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["python"]),
                int(evalmod.cfg.LANG2ID["markdown"]),
                int(evalmod.cfg.LANG2ID["markdown"]),
            ],
            probs=[
                _prob_row("restructuredtext"),
                _prob_row("restructuredtext"),
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("python"),
                _prob_row("markdown"),
                _prob_row("markdown"),
            ],
        )

        metrics = evalmod.evaluate_task(
            "restructuredtext_mix",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        args = SimpleNamespace(
            checkpoint="ckpt.msgpack",
            model_dim=256,
            channels=[96, 128, 192, 256],
            dtype="bfloat16",
            sample_seed=13,
            other_threshold=0.0,
            chunk=1536,
            batch_size=8,
            max_samples=0,
        )
        payload = evalmod._collect_comparison_metrics(args, [metrics], [], None)
        self.assertIn("text_like_binary", payload["tasks"]["restructuredtext_mix"])
        self.assertAlmostEqual(
            float(payload["tasks"]["restructuredtext_mix"]["text_like_binary"]["by_label"]["text_like"]["recall"]),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
