#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"
DEFAULT_OUTPUT_PATH = DEFAULT_REPORTS_DIR / "model_report_matrix.csv"
COMPARISON_FILENAME = "comparison_metrics.json"
NAN = float("nan")


@dataclass(frozen=True)
class LoadedReport:
    path: Path
    model_label: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    value_fn: Callable[[Mapping[str, Any]], float]
    optimal_fn: Callable[[Mapping[str, Any]], float]


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise KeyError("Expected mapping")


def _required(data: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        cur = _as_mapping(cur)[key]
    return cur


def _required_mapping(data: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    return _as_mapping(_required(data, *keys))


def _required_float(data: Mapping[str, Any], *keys: str) -> float:
    value = _required(data, *keys)
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid numeric metric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Metric value is not finite")
    return result


def _rounded_hits(rate: float, support: float) -> float:
    return float(int(round(rate * support)))


def _markdown_group(payload: Mapping[str, Any], group_name: str) -> Mapping[str, Any]:
    markdown = _required_mapping(payload, "tasks", "markdown_mix")
    if group_name == "inline":
        return _required_mapping(markdown, "inline", "overall")
    return _required_mapping(markdown, "block", "overall", group_name)


def _markdown_no_text_hit_samples(payload: Mapping[str, Any], group_name: str) -> float:
    group = _markdown_group(payload, group_name)
    support = _required_float(group, "support")
    hits = _rounded_hits(_required_float(group, "text_rate@0.5"), support)
    return float(support - hits)


def _markdown_group_support(payload: Mapping[str, Any], group_name: str) -> float:
    return _required_float(_markdown_group(payload, group_name), "support")


def _markdown_coverage(payload: Mapping[str, Any], group_name: str) -> float:
    return _required_float(_markdown_group(payload, group_name), "coverage")


def _markdown_binary_f1(payload: Mapping[str, Any]) -> float:
    return _required_float(
        payload,
        "tasks",
        "markdown_mix",
        "text_like_binary",
        "aggregates",
        "macro_f1",
    )


def _markdown_boundary_metric(payload: Mapping[str, Any], metric_name: str) -> float:
    return _required_float(payload, "tasks", "markdown_mix", "boundary_region", "aggregates", metric_name)


def _pure_fragments_total(payload: Mapping[str, Any], rate_key: str) -> tuple[float, float]:
    per_language = _required_mapping(payload, "tasks", "pure_fragments", "per_language")
    support_total = 0.0
    hits_total = 0.0
    for entry in per_language.values():
        group = _as_mapping(entry)
        support = _required_float(group, "support")
        hits_total += _rounded_hits(_required_float(group, rate_key), support)
        support_total += support
    return hits_total, support_total


def _pure_fragments_hits(payload: Mapping[str, Any], rate_key: str) -> float:
    hits, _ = _pure_fragments_total(payload, rate_key)
    return hits


def _pure_fragments_support(payload: Mapping[str, Any], rate_key: str) -> float:
    _, support = _pure_fragments_total(payload, rate_key)
    return support


def _sequence_coverage(payload: Mapping[str, Any], task_name: str, segment_name: str) -> float:
    return _required_float(payload, "tasks", task_name, "segments", segment_name, "coverage")


def _sequence_boundary_metric(payload: Mapping[str, Any], task_name: str, metric_name: str) -> float:
    return _required_float(payload, "tasks", task_name, "boundary_region", "aggregates", metric_name)


def _needle_entry(payload: Mapping[str, Any], bucket: str, variant: str) -> Mapping[str, Any]:
    return _required_mapping(payload, "tasks", "needle", bucket, variant)


def _needle_hits(payload: Mapping[str, Any], bucket: str, variant: str) -> float:
    entry = _needle_entry(payload, bucket, variant)
    support = _required_float(entry, "support")
    return _rounded_hits(_required_float(entry, "det_rate@0.5"), support)


def _needle_support(payload: Mapping[str, Any], bucket: str, variant: str) -> float:
    return _required_float(_needle_entry(payload, bucket, variant), "support")


def _needle_boundary_metric(payload: Mapping[str, Any], bucket: str, metric_name: str) -> float:
    return _required_float(payload, "tasks", "needle", bucket, "boundary_region", "aggregates", metric_name)


def _monitor_metric(payload: Mapping[str, Any], metric_name: str) -> float:
    return _required_float(payload, "monitor_b", "aggregates", metric_name)


def _monitor_boundary_metric(payload: Mapping[str, Any], metric_name: str) -> float:
    return _required_float(payload, "monitor_b", "boundary_region", "aggregates", metric_name)


_NEEDLE_BOUNDARY_BUCKETS: tuple[str, ...] = (
    "needle_64_plus",
    "needle_32_63",
    "needle_16_31",
    "needle_4_15",
)
_NEEDLE_BOUNDARY_METRICS: tuple[tuple[str, str], ...] = (
    ("acc", "micro_acc"),
    ("precision", "macro_precision"),
    ("recall", "macro_recall"),
    ("f1", "macro_f1"),
)
_SEQUENCE_BOUNDARY_TASKS: tuple[str, ...] = (
    "sequence_pair",
    "sequence_triplet",
)


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "markdown_mix.fenced.no_text_hit_samples",
        lambda payload: _markdown_no_text_hit_samples(payload, "fenced"),
        lambda payload: _markdown_group_support(payload, "fenced"),
    ),
    MetricSpec(
        "markdown_mix.fenced.exact_region_avg_coverage",
        lambda payload: _markdown_coverage(payload, "fenced"),
        lambda payload: 1.0,
    ),
    MetricSpec(
        "markdown_mix.plain.no_text_hit_samples",
        lambda payload: _markdown_no_text_hit_samples(payload, "plain"),
        lambda payload: _markdown_group_support(payload, "plain"),
    ),
    MetricSpec(
        "markdown_mix.plain.exact_region_avg_coverage",
        lambda payload: _markdown_coverage(payload, "plain"),
        lambda payload: 1.0,
    ),
    MetricSpec(
        "markdown_mix.inline.no_text_hit_samples",
        lambda payload: _markdown_no_text_hit_samples(payload, "inline"),
        lambda payload: _markdown_group_support(payload, "inline"),
    ),
    MetricSpec(
        "markdown_mix.inline.exact_region_avg_coverage",
        lambda payload: _markdown_coverage(payload, "inline"),
        lambda payload: 1.0,
    ),
    MetricSpec(
        "markdown_mix.binary_text_like.aggregated_f1",
        _markdown_binary_f1,
        lambda payload: 1.0,
    ),
    *tuple(
        MetricSpec(
            f"markdown_mix.boundary_pm4.{public_name}",
            lambda payload, aggregate_name=aggregate_name: _markdown_boundary_metric(
                payload,
                aggregate_name,
            ),
            lambda payload: 1.0,
        )
        for public_name, aggregate_name in _NEEDLE_BOUNDARY_METRICS
    ),
    MetricSpec(
        "pure_fragments.no_misclassified_host_label_bytes",
        lambda payload: _pure_fragments_hits(payload, "fully_pure_rate"),
        lambda payload: _pure_fragments_support(payload, "fully_pure_rate"),
    ),
    MetricSpec(
        "pure_fragments.within_50pct_host_byte_error",
        lambda payload: _pure_fragments_hits(payload, "within_threshold_rate"),
        lambda payload: _pure_fragments_support(payload, "within_threshold_rate"),
    ),
    MetricSpec(
        "sequence_pair.first.exact_region_coverage",
        lambda payload: _sequence_coverage(payload, "sequence_pair", "first"),
        lambda payload: 1.0,
    ),
    MetricSpec(
        "sequence_pair.second.exact_region_coverage",
        lambda payload: _sequence_coverage(payload, "sequence_pair", "second"),
        lambda payload: 1.0,
    ),
    MetricSpec(
        "sequence_triplet.first.exact_region_coverage",
        lambda payload: _sequence_coverage(payload, "sequence_triplet", "first"),
        lambda payload: 1.0,
    ),
    MetricSpec(
        "sequence_triplet.second.exact_region_coverage",
        lambda payload: _sequence_coverage(payload, "sequence_triplet", "second"),
        lambda payload: 1.0,
    ),
    MetricSpec(
        "sequence_triplet.third.exact_region_coverage",
        lambda payload: _sequence_coverage(payload, "sequence_triplet", "third"),
        lambda payload: 1.0,
    ),
    *tuple(
        MetricSpec(
            f"{task_name}.boundary_pm4.{public_name}",
            lambda payload, task_name=task_name, aggregate_name=aggregate_name: _sequence_boundary_metric(
                payload,
                task_name,
                aggregate_name,
            ),
            lambda payload: 1.0,
        )
        for task_name in _SEQUENCE_BOUNDARY_TASKS
        for public_name, aggregate_name in _NEEDLE_BOUNDARY_METRICS
    ),
    MetricSpec(
        "needle_64_plus.any_non_wrapper.coverage_ge_50_samples",
        lambda payload: _needle_hits(payload, "needle_64_plus", "any"),
        lambda payload: _needle_support(payload, "needle_64_plus", "any"),
    ),
    MetricSpec(
        "needle_64_plus.exact_inserted_region.coverage_ge_50_samples",
        lambda payload: _needle_hits(payload, "needle_64_plus", "donor"),
        lambda payload: _needle_support(payload, "needle_64_plus", "donor"),
    ),
    MetricSpec(
        "needle_32_63.any_non_wrapper.coverage_ge_50_samples",
        lambda payload: _needle_hits(payload, "needle_32_63", "any"),
        lambda payload: _needle_support(payload, "needle_32_63", "any"),
    ),
    MetricSpec(
        "needle_32_63.exact_inserted_region.coverage_ge_50_samples",
        lambda payload: _needle_hits(payload, "needle_32_63", "donor"),
        lambda payload: _needle_support(payload, "needle_32_63", "donor"),
    ),
    *tuple(
        MetricSpec(
            f"{bucket}.boundary_pm4.{public_name}",
            lambda payload, bucket=bucket, aggregate_name=aggregate_name: _needle_boundary_metric(
                payload,
                bucket,
                aggregate_name,
            ),
            lambda payload: 1.0,
        )
        for bucket in _NEEDLE_BOUNDARY_BUCKETS
        for public_name, aggregate_name in _NEEDLE_BOUNDARY_METRICS
    ),
    MetricSpec(
        "monitor_b.acc",
        lambda payload: _monitor_metric(payload, "micro_acc"),
        lambda payload: 1.0,
    ),
    MetricSpec(
        "monitor_b.precision",
        lambda payload: _monitor_metric(payload, "macro_precision"),
        lambda payload: 1.0,
    ),
    MetricSpec(
        "monitor_b.recall",
        lambda payload: _monitor_metric(payload, "macro_recall"),
        lambda payload: 1.0,
    ),
    MetricSpec(
        "monitor_b.f1",
        lambda payload: _monitor_metric(payload, "macro_f1"),
        lambda payload: 1.0,
    ),
    *tuple(
        MetricSpec(
            f"monitor_b.boundary_pm4.{public_name}",
            lambda payload, aggregate_name=aggregate_name: _monitor_boundary_metric(
                payload,
                aggregate_name,
            ),
            lambda payload: 1.0,
        )
        for public_name, aggregate_name in _NEEDLE_BOUNDARY_METRICS
    ),
)


def _derive_base_label(report_dir: Path) -> str:
    return report_dir.name.split("__", 1)[0]


def _derive_collision_suffix(report_dir: Path) -> str:
    parts = report_dir.name.split("__", 1)
    if len(parts) == 2 and parts[1]:
        return parts[1]
    return report_dir.name


def _load_report_payload(report_dir: Path) -> Mapping[str, Any] | None:
    json_path = report_dir / COMPARISON_FILENAME
    if not json_path.exists():
        _warn(f"Skipping {report_dir}: missing {COMPARISON_FILENAME}")
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _warn(f"Skipping {report_dir}: failed to read {COMPARISON_FILENAME} ({exc})")
        return None
    if not isinstance(payload, Mapping):
        _warn(f"Skipping {report_dir}: {COMPARISON_FILENAME} did not contain a JSON object")
        return None
    return payload


def load_reports(reports_dir: Path) -> list[LoadedReport]:
    if not reports_dir.exists() or not reports_dir.is_dir():
        return []

    raw_entries: list[tuple[str, Path, Mapping[str, Any]]] = []
    for child in reports_dir.iterdir():
        if child.name == "logs" or not child.is_dir():
            continue
        payload = _load_report_payload(child)
        if payload is None:
            continue
        raw_entries.append((_derive_base_label(child), child, payload))

    raw_entries.sort(key=lambda item: (item[0], item[1].name))
    used_labels: set[str] = set()
    loaded_reports: list[LoadedReport] = []
    for base_label, path, payload in raw_entries:
        label = base_label
        if label in used_labels:
            suffix = _derive_collision_suffix(path)
            label = f"{base_label}__{suffix}"
            disambiguator = 2
            while label in used_labels:
                label = f"{base_label}__{suffix}__{disambiguator}"
                disambiguator += 1
            _warn(f"Disambiguated duplicate model label '{base_label}' as '{label}'")
        used_labels.add(label)
        loaded_reports.append(LoadedReport(path=path, model_label=label, payload=payload))
    return loaded_reports


def _extract_metric_value(
    spec: MetricSpec,
    report: LoadedReport,
    *,
    kind: str,
) -> float:
    fn = spec.value_fn if kind == "value" else spec.optimal_fn
    try:
        value = float(fn(report.payload))
    except Exception as exc:
        _warn(f"{report.path.name}: failed to extract {kind} for '{spec.name}' ({exc})")
        return NAN
    if math.isnan(value):
        return NAN
    if not math.isfinite(value):
        _warn(f"{report.path.name}: non-finite {kind} for '{spec.name}'")
        return NAN
    return value


def _resolve_optimal(metric_name: str, values: Sequence[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return NAN
    best = max(finite_values)
    if any(not math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-9) for value in finite_values):
        _warn(f"Metric '{metric_name}' has differing optimal values across reports; using {best}")
    return best


def build_matrix(loaded_reports: Sequence[LoadedReport]) -> tuple[list[str], list[list[Any]]]:
    header = ["metric", "optimal", *[report.model_label for report in loaded_reports]]
    rows: list[list[Any]] = []
    for spec in METRIC_SPECS:
        row_values: list[float] = []
        row_optimals: list[float] = []
        for report in loaded_reports:
            row_values.append(_extract_metric_value(spec, report, kind="value"))
            row_optimals.append(_extract_metric_value(spec, report, kind="optimal"))
        rows.append([spec.name, _resolve_optimal(spec.name, row_optimals), *row_values])
    return header, rows


def write_matrix(output_path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def export_report_matrix(reports_dir: Path, output_path: Path) -> int:
    loaded_reports = load_reports(reports_dir)
    if not loaded_reports:
        raise FileNotFoundError(f"No valid report directories found under {reports_dir}")
    header, rows = build_matrix(loaded_reports)
    write_matrix(output_path, header, rows)
    return len(loaded_reports)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a numeric metric-by-model CSV from evaluation reports.",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(DEFAULT_REPORTS_DIR),
        help="Directory containing per-run evaluation report directories.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the output CSV file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    reports_dir = Path(args.reports_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    try:
        report_count = export_report_matrix(reports_dir, output_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote {report_count} report columns to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
