from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.export_report_matrix as exportmod


def _boundary_region_payload(
    *,
    acc: float,
    precision: float,
    recall: float,
    f1: float,
    support_chars: int = 12,
    samples: int = 3,
) -> dict:
    return {
        "aggregates": {
            "micro_acc": acc,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
        },
        "support_chars": support_chars,
        "samples": samples,
    }


def _payload(
    *,
    fenced_support: int = 10,
    fenced_text_rate: float = 0.2,
    fenced_coverage: float = 0.9,
    plain_support: int = 10,
    plain_text_rate: float = 0.1,
    plain_coverage: float = 0.8,
    inline_support: int = 4,
    inline_text_rate: float = 0.25,
    inline_coverage: float = 0.5,
    binary_f1: float = 0.75,
    pure_per_language: dict | None = None,
    seq_pair: tuple[float, float] = (0.7, 0.8),
    seq_triplet: tuple[float, float, float] = (0.1, 0.2, 0.3),
    markdown_boundary: dict | None = None,
    sequence_boundary: dict | None = None,
    needle_64_any: tuple[float, int] = (0.6, 5),
    needle_64_donor: tuple[float, int] = (0.4, 5),
    needle_32_any: tuple[float, int] = (0.5, 5),
    needle_32_donor: tuple[float, int] = (0.2, 5),
    needle_boundary: dict | None = None,
    monitor_metrics: dict | None = None,
    monitor_boundary: dict | None = None,
) -> dict:
    if pure_per_language is None:
        pure_per_language = {
            "python": {
                "fully_pure_rate": 2.0 / 3.0,
                "within_threshold_rate": 1.0,
                "support": 3,
            },
            "shell": {
                "fully_pure_rate": 0.5,
                "within_threshold_rate": 0.5,
                "support": 2,
            },
        }
    if monitor_metrics is None:
        monitor_metrics = {
            "micro_acc": 0.91,
            "macro_precision": 0.82,
            "macro_recall": 0.73,
            "macro_f1": 0.64,
        }
    if needle_boundary is None:
        needle_boundary = {
            "needle_64_plus": _boundary_region_payload(acc=0.71, precision=0.61, recall=0.51, f1=0.41),
            "needle_32_63": _boundary_region_payload(acc=0.72, precision=0.62, recall=0.52, f1=0.42),
            "needle_16_31": _boundary_region_payload(acc=0.73, precision=0.63, recall=0.53, f1=0.43),
            "needle_4_15": _boundary_region_payload(acc=0.74, precision=0.64, recall=0.54, f1=0.44),
        }
    if sequence_boundary is None:
        sequence_boundary = {
            "sequence_pair": _boundary_region_payload(acc=0.65, precision=0.55, recall=0.45, f1=0.35),
            "sequence_triplet": _boundary_region_payload(acc=0.66, precision=0.56, recall=0.46, f1=0.36),
        }
    if markdown_boundary is None:
        markdown_boundary = _boundary_region_payload(acc=0.67, precision=0.57, recall=0.47, f1=0.37)
    if monitor_boundary is None:
        monitor_boundary = _boundary_region_payload(acc=0.68, precision=0.58, recall=0.48, f1=0.38)
    return {
        "tasks": {
            "markdown_mix": {
                "block": {
                    "overall": {
                        "fenced": {
                            "support": fenced_support,
                            "text_rate@0.5": fenced_text_rate,
                            "coverage": fenced_coverage,
                        },
                        "plain": {
                            "support": plain_support,
                            "text_rate@0.5": plain_text_rate,
                            "coverage": plain_coverage,
                        },
                    }
                },
                "inline": {
                    "overall": {
                        "support": inline_support,
                        "text_rate@0.5": inline_text_rate,
                        "coverage": inline_coverage,
                    }
                },
                "text_like_binary": {
                    "aggregates": {
                        "macro_f1": binary_f1,
                    }
                },
                "boundary_region": markdown_boundary,
            },
            "pure_fragments": {
                "per_language": pure_per_language,
            },
            "sequence_pair": {
                "segments": {
                    "first": {"coverage": seq_pair[0]},
                    "second": {"coverage": seq_pair[1]},
                },
                "boundary_region": sequence_boundary["sequence_pair"],
            },
            "sequence_triplet": {
                "segments": {
                    "first": {"coverage": seq_triplet[0]},
                    "second": {"coverage": seq_triplet[1]},
                    "third": {"coverage": seq_triplet[2]},
                },
                "boundary_region": sequence_boundary["sequence_triplet"],
            },
            "needle": {
                "needle_64_plus": {
                    "any": {"det_rate@0.5": needle_64_any[0], "support": needle_64_any[1]},
                    "donor": {"det_rate@0.5": needle_64_donor[0], "support": needle_64_donor[1]},
                    "boundary_region": needle_boundary["needle_64_plus"],
                },
                "needle_32_63": {
                    "any": {"det_rate@0.5": needle_32_any[0], "support": needle_32_any[1]},
                    "donor": {"det_rate@0.5": needle_32_donor[0], "support": needle_32_donor[1]},
                    "boundary_region": needle_boundary["needle_32_63"],
                },
                "needle_16_31": {
                    "boundary_region": needle_boundary["needle_16_31"],
                },
                "needle_4_15": {
                    "boundary_region": needle_boundary["needle_4_15"],
                },
            },
        },
        "monitor_b": {
            "aggregates": monitor_metrics,
            "boundary_region": monitor_boundary,
        },
    }


def _write_report_dir(root: Path, name: str, payload: dict) -> None:
    report_dir = root / name
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "comparison_metrics.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _read_rows(csv_path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {row["metric"]: row for row in reader}
        assert reader.fieldnames is not None
        return list(reader.fieldnames), rows


class TestExportReportMatrix(unittest.TestCase):
    def test_cli_writes_numeric_csv_and_reconstructs_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports_dir = root / "reports"
            reports_dir.mkdir()

            _write_report_dir(
                reports_dir,
                "alpha__data_b__tau0p5__20260324_150315",
                _payload(),
            )
            _write_report_dir(
                reports_dir,
                "beta__data_b__tau0p5__20260324_150315",
                _payload(
                    fenced_text_rate=0.0,
                    fenced_coverage=0.95,
                    plain_text_rate=0.4,
                    plain_coverage=0.65,
                    inline_text_rate=0.5,
                    inline_coverage=0.25,
                    binary_f1=0.88,
                    pure_per_language={
                        "python": {
                            "fully_pure_rate": 1.0,
                            "within_threshold_rate": 1.0,
                            "support": 2,
                        },
                        "shell": {
                            "fully_pure_rate": 0.5,
                            "within_threshold_rate": 1.0,
                            "support": 2,
                        },
                    },
                    seq_pair=(0.6, 0.5),
                    seq_triplet=(0.9, 0.8, 0.7),
                    markdown_boundary=_boundary_region_payload(
                        acc=0.91,
                        precision=0.92,
                        recall=0.93,
                        f1=0.94,
                        support_chars=20,
                        samples=4,
                    ),
                    sequence_boundary={
                        "sequence_pair": _boundary_region_payload(
                            acc=0.77,
                            precision=0.78,
                            recall=0.79,
                            f1=0.8,
                            support_chars=18,
                            samples=4,
                        ),
                        "sequence_triplet": _boundary_region_payload(
                            acc=0.87,
                            precision=0.88,
                            recall=0.89,
                            f1=0.9,
                            support_chars=22,
                            samples=4,
                        ),
                    },
                    needle_64_any=(0.75, 8),
                    needle_64_donor=(0.5, 8),
                    needle_32_any=(0.25, 8),
                    needle_32_donor=(0.125, 8),
                    needle_boundary={
                        "needle_64_plus": _boundary_region_payload(
                            acc=0.81,
                            precision=0.82,
                            recall=0.83,
                            f1=0.84,
                            support_chars=16,
                            samples=4,
                        ),
                        "needle_32_63": _boundary_region_payload(
                            acc=0.85,
                            precision=0.86,
                            recall=0.87,
                            f1=0.88,
                            support_chars=14,
                            samples=4,
                        ),
                        "needle_16_31": _boundary_region_payload(
                            acc=0.89,
                            precision=0.9,
                            recall=0.91,
                            f1=0.92,
                            support_chars=10,
                            samples=4,
                        ),
                        "needle_4_15": _boundary_region_payload(
                            acc=0.93,
                            precision=0.94,
                            recall=0.95,
                            f1=0.96,
                            support_chars=8,
                            samples=4,
                        ),
                    },
                    monitor_metrics={
                        "micro_acc": 0.98,
                        "macro_precision": 0.97,
                        "macro_recall": 0.96,
                        "macro_f1": 0.95,
                    },
                    monitor_boundary=_boundary_region_payload(
                        acc=0.85,
                        precision=0.86,
                        recall=0.87,
                        f1=0.88,
                        support_chars=24,
                        samples=4,
                    ),
                ),
            )

            output_path = root / "matrix.csv"
            exit_code = exportmod.main(
                [
                    "--reports-dir",
                    str(reports_dir),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            header, rows = _read_rows(output_path)
            self.assertEqual(header, ["metric", "optimal", "alpha", "beta"])

            for row in rows.values():
                for key, value in row.items():
                    if key == "metric":
                        continue
                    float(value)

            self.assertEqual(rows["markdown_mix.fenced.no_text_hit_samples"]["alpha"], "8.0")
            self.assertEqual(rows["markdown_mix.fenced.no_text_hit_samples"]["beta"], "10.0")
            self.assertEqual(rows["markdown_mix.fenced.no_text_hit_samples"]["optimal"], "10.0")
            self.assertEqual(rows["markdown_mix.plain.no_text_hit_samples"]["beta"], "6.0")
            self.assertEqual(rows["markdown_mix.inline.no_text_hit_samples"]["alpha"], "3.0")
            self.assertEqual(rows["markdown_mix.fenced.exact_region_avg_coverage"]["alpha"], "0.9")
            self.assertEqual(
                rows["markdown_mix.binary_text_like.aggregated_f1"]["beta"],
                "0.88",
            )
            self.assertEqual(rows["markdown_mix.boundary_pm4.acc"]["alpha"], "0.67")
            self.assertEqual(rows["markdown_mix.boundary_pm4.f1"]["beta"], "0.94")
            self.assertEqual(rows["pure_fragments.no_misclassified_host_label_bytes"]["alpha"], "3.0")
            self.assertEqual(rows["pure_fragments.within_50pct_host_byte_error"]["alpha"], "4.0")
            self.assertEqual(
                rows["needle_64_plus.any_non_wrapper.coverage_ge_50_samples"]["beta"],
                "6.0",
            )
            self.assertEqual(
                rows["needle_32_63.exact_inserted_region.coverage_ge_50_samples"]["beta"],
                "1.0",
            )
            self.assertEqual(rows["sequence_pair.boundary_pm4.acc"]["alpha"], "0.65")
            self.assertEqual(rows["sequence_pair.boundary_pm4.f1"]["beta"], "0.8")
            self.assertEqual(rows["sequence_triplet.boundary_pm4.recall"]["beta"], "0.89")
            self.assertEqual(rows["needle_64_plus.boundary_pm4.acc"]["alpha"], "0.71")
            self.assertEqual(rows["needle_64_plus.boundary_pm4.f1"]["beta"], "0.84")
            self.assertEqual(rows["needle_4_15.boundary_pm4.recall"]["beta"], "0.95")
            self.assertEqual(rows["sequence_triplet.third.exact_region_coverage"]["beta"], "0.7")
            self.assertEqual(rows["monitor_b.f1"]["beta"], "0.95")
            self.assertEqual(rows["monitor_b.boundary_pm4.precision"]["alpha"], "0.58")
            self.assertEqual(rows["monitor_b.boundary_pm4.f1"]["beta"], "0.88")

    def test_duplicate_model_names_and_missing_metrics_emit_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports_dir = root / "reports"
            reports_dir.mkdir()
            (reports_dir / "logs").mkdir()
            (reports_dir / "notes.txt").write_text("ignored", encoding="utf-8")
            (reports_dir / "broken__data_b__tau0p5__20260324_150315").mkdir()

            _write_report_dir(
                reports_dir,
                "model__data_b__tau0p5__20260324_150315",
                _payload(),
            )
            missing_monitor = _payload()
            missing_monitor.pop("monitor_b")
            _write_report_dir(
                reports_dir,
                "model__data_b__tau0p5__20260325_150315",
                missing_monitor,
            )

            output_path = root / "matrix.csv"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = exportmod.main(
                    [
                        "--reports-dir",
                        str(reports_dir),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            header, rows = _read_rows(output_path)
            self.assertEqual(
                header,
                [
                    "metric",
                    "optimal",
                    "model",
                    "model__data_b__tau0p5__20260325_150315",
                ],
            )
            self.assertEqual(rows["monitor_b.acc"]["model"], "0.91")
            self.assertEqual(
                rows["monitor_b.acc"]["model__data_b__tau0p5__20260325_150315"],
                "nan",
            )
            self.assertIn("Disambiguated duplicate model label", stderr.getvalue())
            self.assertIn("failed to extract value for 'monitor_b.acc'", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
