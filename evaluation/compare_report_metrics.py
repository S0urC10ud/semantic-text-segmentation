#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import csv
from pathlib import Path
from typing import Any

from export_report_matrix import COMPARISON_FILENAME, METRIC_SPECS


DISPLAY_NAME_OVERRIDES: dict[str, str] = {
    "markdown_mix.fenced.no_text_hit_samples": "Markdown fenced: no text-hit samples",
    "markdown_mix.fenced.exact_region_avg_coverage": "Markdown fenced: exact code coverage",
    "markdown_mix.plain.no_text_hit_samples": "Markdown plain: no text-hit samples",
    "markdown_mix.plain.exact_region_avg_coverage": "Markdown plain: exact code coverage",
    "markdown_mix.inline.no_text_hit_samples": "Markdown inline: no text-hit samples",
    "markdown_mix.inline.exact_region_avg_coverage": "Markdown inline: exact code coverage",
    "markdown_mix.binary_text_like.aggregated_f1": "Markdown: text-vs-nontext F1",
    "markdown_mix.boundary_pm4.acc": "Markdown: boundary Acc (±4)",
    "markdown_mix.boundary_pm4.precision": "Markdown: boundary Precision (±4)",
    "markdown_mix.boundary_pm4.recall": "Markdown: boundary Recall (±4)",
    "markdown_mix.boundary_pm4.f1": "Markdown: boundary F1 (±4)",
    "pure_fragments.no_misclassified_host_label_bytes": "Pure hosts: fully pure",
    "pure_fragments.within_50pct_host_byte_error": "Pure hosts: within 50% host-byte error",
    "sequence_pair.first.exact_region_coverage": "Sequence pair: first-region coverage",
    "sequence_pair.second.exact_region_coverage": "Sequence pair: second-region coverage",
    "sequence_triplet.first.exact_region_coverage": "Sequence triplet: first-region coverage",
    "sequence_triplet.second.exact_region_coverage": "Sequence triplet: middle-region coverage",
    "sequence_triplet.third.exact_region_coverage": "Sequence triplet: third-region coverage",
    "sequence_pair.boundary_pm4.acc": "Sequence pair: boundary Acc (±4)",
    "sequence_pair.boundary_pm4.precision": "Sequence pair: boundary Precision (±4)",
    "sequence_pair.boundary_pm4.recall": "Sequence pair: boundary Recall (±4)",
    "sequence_pair.boundary_pm4.f1": "Sequence pair: boundary F1 (±4)",
    "sequence_triplet.boundary_pm4.acc": "Sequence triplet: boundary Acc (±4)",
    "sequence_triplet.boundary_pm4.precision": "Sequence triplet: boundary Precision (±4)",
    "sequence_triplet.boundary_pm4.recall": "Sequence triplet: boundary Recall (±4)",
    "sequence_triplet.boundary_pm4.f1": "Sequence triplet: boundary F1 (±4)",
    "needle_64_plus.any_non_wrapper.coverage_ge_50_samples": "Needle 64+: any foreign region detected",
    "needle_64_plus.exact_inserted_region.coverage_ge_50_samples": "Needle 64+: exact donor region detected",
    "needle_32_63.any_non_wrapper.coverage_ge_50_samples": "Needle 32-63: any foreign region detected",
    "needle_32_63.exact_inserted_region.coverage_ge_50_samples": "Needle 32-63: exact donor region detected",
    "needle_64_plus.boundary_pm4.acc": "Needle 64+: boundary Acc (±4)",
    "needle_64_plus.boundary_pm4.precision": "Needle 64+: boundary Precision (±4)",
    "needle_64_plus.boundary_pm4.recall": "Needle 64+: boundary Recall (±4)",
    "needle_64_plus.boundary_pm4.f1": "Needle 64+: boundary F1 (±4)",
    "needle_32_63.boundary_pm4.acc": "Needle 32-63: boundary Acc (±4)",
    "needle_32_63.boundary_pm4.precision": "Needle 32-63: boundary Precision (±4)",
    "needle_32_63.boundary_pm4.recall": "Needle 32-63: boundary Recall (±4)",
    "needle_32_63.boundary_pm4.f1": "Needle 32-63: boundary F1 (±4)",
    "needle_16_31.boundary_pm4.acc": "Needle 16-31: boundary Acc (±4)",
    "needle_16_31.boundary_pm4.precision": "Needle 16-31: boundary Precision (±4)",
    "needle_16_31.boundary_pm4.recall": "Needle 16-31: boundary Recall (±4)",
    "needle_16_31.boundary_pm4.f1": "Needle 16-31: boundary F1 (±4)",
    "needle_4_15.boundary_pm4.acc": "Needle 4-15: boundary Acc (±4)",
    "needle_4_15.boundary_pm4.precision": "Needle 4-15: boundary Precision (±4)",
    "needle_4_15.boundary_pm4.recall": "Needle 4-15: boundary Recall (±4)",
    "needle_4_15.boundary_pm4.f1": "Needle 4-15: boundary F1 (±4)",
    "monitor_b.acc": "Realistic full-file: overall Acc",
    "monitor_b.precision": "Realistic full-file: overall Precision",
    "monitor_b.recall": "Realistic full-file: overall Recall",
    "monitor_b.f1": "Realistic full-file: overall F1",
    "monitor_b.boundary_pm4.acc": "Realistic full-file: boundary Acc (±4)",
    "monitor_b.boundary_pm4.precision": "Realistic full-file: boundary Precision (±4)",
    "monitor_b.boundary_pm4.recall": "Realistic full-file: boundary Recall (±4)",
    "monitor_b.boundary_pm4.f1": "Realistic full-file: boundary F1 (±4)",
}


REALISTIC_COMPOSITE_METRICS: tuple[str, ...] = (
    "monitor_b.f1",
    "monitor_b.boundary_pm4.f1",
)

SYNTHETIC_BOUNDARY_COMPOSITE_METRICS: tuple[str, ...] = (
    "pure_fragments.no_misclassified_host_label_bytes",
    "pure_fragments.within_50pct_host_byte_error",
    "sequence_pair.first.exact_region_coverage",
    "sequence_pair.second.exact_region_coverage",
    "sequence_pair.boundary_pm4.f1",
    "sequence_triplet.first.exact_region_coverage",
    "sequence_triplet.second.exact_region_coverage",
    "sequence_triplet.third.exact_region_coverage",
    "sequence_triplet.boundary_pm4.f1",
    "needle_64_plus.any_non_wrapper.coverage_ge_50_samples",
    "needle_64_plus.exact_inserted_region.coverage_ge_50_samples",
    "needle_64_plus.boundary_pm4.f1",
    "needle_32_63.any_non_wrapper.coverage_ge_50_samples",
    "needle_32_63.exact_inserted_region.coverage_ge_50_samples",
    "needle_32_63.boundary_pm4.f1",
    "markdown_mix.fenced.exact_region_avg_coverage",
    "markdown_mix.plain.exact_region_avg_coverage",
    "markdown_mix.inline.exact_region_avg_coverage",
    "markdown_mix.binary_text_like.aggregated_f1",
    "markdown_mix.boundary_pm4.f1",
)

PURITY_COMPOSITE_METRICS: tuple[str, ...] = (
    "pure_fragments.no_misclassified_host_label_bytes",
    "pure_fragments.within_50pct_host_byte_error",
)

TRANSITIONS_COMPOSITE_METRICS: tuple[str, ...] = (
    "sequence_pair.first.exact_region_coverage",
    "sequence_pair.second.exact_region_coverage",
    "sequence_pair.boundary_pm4.f1",
    "sequence_triplet.first.exact_region_coverage",
    "sequence_triplet.second.exact_region_coverage",
    "sequence_triplet.third.exact_region_coverage",
    "sequence_triplet.boundary_pm4.f1",
)

NEEDLES_COMPOSITE_METRICS: tuple[str, ...] = (
    "needle_64_plus.any_non_wrapper.coverage_ge_50_samples",
    "needle_64_plus.exact_inserted_region.coverage_ge_50_samples",
    "needle_64_plus.boundary_pm4.f1",
    "needle_32_63.any_non_wrapper.coverage_ge_50_samples",
    "needle_32_63.exact_inserted_region.coverage_ge_50_samples",
    "needle_32_63.boundary_pm4.f1",
)

MARKDOWN_COMPOSITE_METRICS: tuple[str, ...] = (
    "markdown_mix.fenced.exact_region_avg_coverage",
    "markdown_mix.plain.exact_region_avg_coverage",
    "markdown_mix.inline.exact_region_avg_coverage",
    "markdown_mix.binary_text_like.aggregated_f1",
    "markdown_mix.boundary_pm4.f1",
)

COMPOSITES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Realistic composite", REALISTIC_COMPOSITE_METRICS),
    ("Synthetic boundary composite", SYNTHETIC_BOUNDARY_COMPOSITE_METRICS),
    ("Purity composite", PURITY_COMPOSITE_METRICS),
    ("Transitions composite", TRANSITIONS_COMPOSITE_METRICS),
    ("Needles composite", NEEDLES_COMPOSITE_METRICS),
    ("Markdown composite", MARKDOWN_COMPOSITE_METRICS),
)


SPEC_BY_NAME = {spec.name: spec for spec in METRIC_SPECS}


def _load_payload(report_dir: Path) -> dict[str, Any]:
    comparison_path = report_dir / COMPARISON_FILENAME
    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing {COMPARISON_FILENAME}: {comparison_path}")
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{comparison_path} did not contain a JSON object")
    return payload


def _normalized_metric_value(payload: dict[str, Any], metric_name: str) -> float:
    spec = SPEC_BY_NAME.get(metric_name)
    if spec is None:
        raise KeyError(f"Unknown metric '{metric_name}'")
    value = float(spec.value_fn(payload))
    optimal = float(spec.optimal_fn(payload))
    if not math.isfinite(value) or not math.isfinite(optimal) or optimal == 0.0:
        raise ValueError(f"Non-finite metric '{metric_name}' (value={value}, optimal={optimal})")
    return value / optimal


def _maybe_normalized_metric_value(payload: dict[str, Any], metric_name: str) -> float | None:
    try:
        return _normalized_metric_value(payload, metric_name)
    except Exception:
        return None


def _detect_delimiter(csv_path: Path) -> str:
    first_line = csv_path.read_text(encoding="utf-8").splitlines()[0]
    if first_line.count(";") > first_line.count(","):
        return ";"
    return ","


def _parse_csv_number(raw: str) -> float:
    value = raw.strip()
    if not value:
        raise ValueError("Empty numeric field")
    return float(value.replace(",", "."))


def _format_float(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _metric_display_name(metric_name: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(metric_name, metric_name)


def _build_detailed_rows(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
) -> tuple[list[tuple[str, float, float]], list[str]]:
    rows: list[tuple[str, float, float]] = []
    omitted_metric_names: list[str] = []
    for spec in METRIC_SPECS:
        value_a = metrics_a.get(spec.name)
        value_b = metrics_b.get(spec.name)
        if value_a is None and value_b is None:
            omitted_metric_names.append(spec.name)
            continue
        if value_a is None or value_b is None:
            raise ValueError(f"Metric '{spec.name}' was present in only one report, which would make the comparison invalid.")
        rows.append((_metric_display_name(spec.name), value_a, value_b))
    return rows, omitted_metric_names


def _build_composite_rows(metrics_a: dict[str, float], metrics_b: dict[str, float]) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for label, metric_names in COMPOSITES:
        values_a = [metrics_a[metric_name] for metric_name in metric_names]
        values_b = [metrics_b[metric_name] for metric_name in metric_names]
        rows.append((label, sum(values_a) / len(values_a), sum(values_b) / len(values_b)))
    return rows


def _normalized_metrics_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for spec in METRIC_SPECS:
        value = _maybe_normalized_metric_value(payload, spec.name)
        if value is not None:
            metrics[spec.name] = value
    return metrics


def _normalized_metrics_from_matrix(csv_path: Path, column_name: str) -> dict[str, float]:
    delimiter = _detect_delimiter(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        metric_key = "metric/path" if "metric/path" in reader.fieldnames else "metric"
        if column_name not in reader.fieldnames:
            raise KeyError(f"CSV column '{column_name}' not found in {csv_path}")
        metrics: dict[str, float] = {}
        for row in reader:
            raw_metric_name = (row.get(metric_key) or "").strip()
            raw_value = (row.get(column_name) or "").strip()
            raw_optimal = (row.get("optimal") or "").strip()
            if not raw_metric_name or not raw_value or not raw_optimal:
                continue
            metric_name = raw_metric_name.replace("/", ".")
            value = _parse_csv_number(raw_value)
            optimal = _parse_csv_number(raw_optimal)
            if optimal == 0.0 or not math.isfinite(value) or not math.isfinite(optimal):
                continue
            metrics[metric_name] = value / optimal
    return metrics


def _load_metrics(source: Path, matrix_column: str | None) -> dict[str, float]:
    if source.is_dir():
        if matrix_column is not None:
            raise ValueError(f"Matrix column was provided for directory source {source}")
        return _normalized_metrics_from_payload(_load_payload(source))
    if source.is_file():
        if matrix_column is None:
            raise ValueError(f"CSV source requires --matrix-column: {source}")
        return _normalized_metrics_from_matrix(source, matrix_column)
    raise FileNotFoundError(f"Source does not exist: {source}")


def _render_markdown_table(
    title: str,
    label_a: str,
    label_b: str,
    rows: list[tuple[str, float, float]],
    *,
    decimals: int,
) -> str:
    lines = [
        f"## {title}",
        "",
        f"| Metric | {label_a} | {label_b} |",
        "| --- | ---: | ---: |",
    ]
    for metric_label, value_a, value_b in rows:
        lines.append(
            f"| {metric_label} | {_format_float(value_a, decimals)} | {_format_float(value_b, decimals)} |"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two evaluation metric sources using thesis-style normalized metrics.",
    )
    parser.add_argument("--source-a", "--report-a", required=True, dest="source_a", help="First source: report directory or exported matrix CSV.")
    parser.add_argument("--source-b", "--report-b", required=True, dest="source_b", help="Second source: report directory or exported matrix CSV.")
    parser.add_argument("--matrix-column-a", default=None, help="Column name to use when source A is a matrix CSV.")
    parser.add_argument("--matrix-column-b", default=None, help="Column name to use when source B is a matrix CSV.")
    parser.add_argument("--label-a", required=True, help="Column label for the first report.")
    parser.add_argument("--label-b", required=True, help="Column label for the second report.")
    parser.add_argument("--decimals", type=int, default=3, help="Number of decimal places to print.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_a = Path(args.source_a).expanduser().resolve()
    source_b = Path(args.source_b).expanduser().resolve()
    metrics_a = _load_metrics(source_a, args.matrix_column_a)
    metrics_b = _load_metrics(source_b, args.matrix_column_b)

    composite_rows = _build_composite_rows(metrics_a, metrics_b)
    detailed_rows, omitted_metric_names = _build_detailed_rows(metrics_a, metrics_b)

    print(
        _render_markdown_table(
            "Composite Summary",
            args.label_a,
            args.label_b,
            composite_rows,
            decimals=args.decimals,
        )
    )
    print()
    if omitted_metric_names:
        print(
            "Omitted metrics missing in both reports: "
            + ", ".join(omitted_metric_names)
        )
        print()
    print(
        _render_markdown_table(
            "Detailed Metrics",
            args.label_a,
            args.label_b,
            detailed_rows,
            decimals=args.decimals,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
