#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import numpy. Run this script with `.venv/bin/python` from the repo root."
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import matplotlib. Run this script with `.venv/bin/python` from the repo root."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "active_learning" / "labels_full.sqlite"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "confusion_images"
DEFAULT_TOP_K = 12
DEFAULT_DPI = 350
DEFAULT_TOP_PAIR_COUNT = 10
DEFAULT_LABEL = "unlabeled"
DEFAULT_LAYOUT = "double"


@dataclass(frozen=True)
class TopPair:
    true_label: str
    predicted_label: str
    count: int
    rate_vs_true: float


@dataclass(frozen=True)
class ConfusionAnalysis:
    row_count: int
    distinct_file_count: int
    snippet_start_zero_count: int
    full_file_mode_count: int
    total_bytes: int
    total_changed_bytes: int
    total_confused_bytes: int
    changed_bytes_per_file: list[int]
    changed_fraction_per_file: list[float]
    true_support: dict[str, int]
    offdiag_as_true: dict[str, int]
    offdiag_as_pred: dict[str, int]
    pair_counts: dict[tuple[str, str], int]
    train_file_support: dict[str, int]
    train_gemini_file_pair_counts: dict[tuple[str, str], int]
    plot_labels: list[str]
    plot_raw_matrix: np.ndarray
    plot_rate_matrix: np.ndarray
    applicable_file_count: int
    selected_applicable_file_count: int
    selected_labels: list[str]
    selected_raw_matrix: np.ndarray
    selected_rate_matrix: np.ndarray
    selected_confused_bytes: int
    top_pairs: list[TopPair]

    @property
    def selected_capture(self) -> float:
        if self.total_confused_bytes <= 0:
            return 0.0
        return float(self.selected_confused_bytes) / float(self.total_confused_bytes)

    @property
    def plot_capture(self) -> float:
        if self.applicable_file_count <= 0:
            return 0.0
        return float(self.selected_applicable_file_count) / float(self.applicable_file_count)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze active-learning refinement confusions from a SQLite label store "
            "and save a high-resolution confusion matrix PNG."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to the SQLite label store (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to confusion_images/active_learning_refinement_confusion_top{K}.png",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of labels to keep in the filtered confusion view (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG resolution in dots per inch (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--layout",
        choices=("double", "left", "right"),
        default=DEFAULT_LAYOUT,
        help=(
            "Which confusion panel layout to save: "
            "'double' keeps both panels, 'left' keeps the applicable-files panel, "
            "and 'right' keeps the train-label-rate panel."
        ),
    )
    parser.add_argument(
        "--hide-suptitle",
        action="store_true",
        help="Omit the figure-level title/subtitle lines.",
    )
    parser.add_argument(
        "--x-tick-rotation",
        type=float,
        default=None,
        help="Override x-axis tick rotation in degrees.",
    )
    return parser.parse_args(argv)


def _resolve_output_path(raw_out: Path | None, *, top_k: int) -> Path:
    if raw_out is not None:
        return raw_out.resolve()
    return (
        DEFAULT_OUTPUT_DIR
        / f"active_learning_refinement_confusion_top{int(top_k)}.png"
    ).resolve()


def _resolve_ecdf_output_path(confusion_out: str | Path) -> Path:
    base = Path(confusion_out).resolve()
    return base.with_name(f"{base.stem}_change_ecdfs.png")


def _connect(db_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(Path(db_path).resolve()))
    con.row_factory = sqlite3.Row
    return con


def _normalize_label(value: object, *, default: str = DEFAULT_LABEL) -> str:
    text = str(value or "").strip()
    return text if text else default


def _parse_segments(raw_json: object) -> list[dict[str, object]]:
    raw_text = str(raw_json or "[]")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [segment for segment in parsed if isinstance(segment, dict)]


def _parse_metadata(raw_json: object) -> dict[str, object]:
    raw_text = str(raw_json or "{}")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0]
    running = 0
    for ch in text:
        running += len(ch.encode("utf-8", "ignore"))
        offsets.append(running)
    return offsets


def _dense_byte_labels(
    text: str,
    segments: Iterable[dict[str, object]],
    *,
    default_label: str = DEFAULT_LABEL,
) -> list[str]:
    offsets = _char_to_byte_offsets(text)
    total_bytes = offsets[-1]
    labels = [default_label] * total_bytes
    text_chars = len(text)
    for seg in segments:
        try:
            start_char = max(0, min(text_chars, int(seg.get("start", 0))))
            end_char = max(0, min(text_chars, int(seg.get("end", 0))))
        except Exception:
            continue
        if end_char <= start_char:
            continue
        label = _normalize_label(
            seg.get("label") or seg.get("language"),
            default=default_label,
        )
        start_byte = offsets[start_char]
        end_byte = offsets[end_char]
        if end_byte <= start_byte:
            continue
        labels[start_byte:end_byte] = [label] * (end_byte - start_byte)
    return labels


def _rank_labels(
    offdiag_as_true: Counter[str],
    offdiag_as_pred: Counter[str],
    true_support: Counter[str],
    *,
    top_k: int,
) -> list[str]:
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    combined: Counter[str] = Counter()
    combined.update(offdiag_as_true)
    combined.update(offdiag_as_pred)
    if combined:
        ranked = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
        return [label for label, _ in ranked[:top_k]]
    ranked_support = sorted(true_support.items(), key=lambda kv: (-kv[1], kv[0]))
    return [label for label, _ in ranked_support[:top_k]]


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(round((len(ordered) - 1) * float(q)))
    index = max(0, min(index, len(ordered) - 1))
    return float(ordered[index])


def analyze_store(db_path: str | Path, *, top_k: int = DEFAULT_TOP_K) -> ConfusionAnalysis:
    row_count = 0
    snippet_start_zero_count = 0
    full_file_mode_count = 0
    total_bytes = 0
    total_confused_bytes = 0
    changed_bytes_per_file: list[int] = []
    changed_fraction_per_file: list[float] = []
    file_keys: set[tuple[str, str, int]] = set()
    file_source_labels: dict[tuple[str, str, int], str] = {}
    file_gemini_labels: dict[tuple[str, str, int], set[str]] = {}

    true_support: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    offdiag_pair_counts: Counter[tuple[str, str]] = Counter()
    offdiag_as_true: Counter[str] = Counter()
    offdiag_as_pred: Counter[str] = Counter()

    with _connect(db_path) as con:
        rows = con.execute(
            """
            SELECT
              round_id,
              source_lang,
              sample_hash,
              sample_index,
              snippet_start,
              snippet_text,
              predicted_segments_json,
              refined_segments_json,
              metadata_json
            FROM refinements
            ORDER BY id ASC
            """
        ).fetchall()

    for row in rows:
        row_count += 1
        file_key = (
            _normalize_label(row["round_id"], default=""),
            _normalize_label(row["sample_hash"], default=""),
            int(row["sample_index"]),
        )
        file_keys.add(file_key)
        if int(row["snippet_start"]) == 0:
            snippet_start_zero_count += 1

        metadata = _parse_metadata(row["metadata_json"])
        if bool(metadata.get("full_file_mode")):
            full_file_mode_count += 1

        text = str(row["snippet_text"] or "")
        pred_segments = _parse_segments(row["predicted_segments_json"])
        ref_segments = _parse_segments(row["refined_segments_json"])
        source_label = _normalize_label(row["source_lang"])
        prev_source_label = file_source_labels.setdefault(file_key, source_label)
        if prev_source_label != source_label:
            raise ValueError(
                "Mismatched source_lang values found for one file:"
                f" file_key={file_key!r} values={prev_source_label!r}/{source_label!r}"
            )
        gemini_labels = file_gemini_labels.setdefault(file_key, set())
        for seg in ref_segments:
            gemini_labels.add(
                _normalize_label(seg.get("label") or seg.get("language"))
            )

        pred_labels = _dense_byte_labels(text, pred_segments)
        ref_labels = _dense_byte_labels(text, ref_segments)
        if len(pred_labels) != len(ref_labels):
            raise ValueError(
                f"Pred/ref byte label lengths disagree for sample {row['sample_hash']}"
            )

        total_bytes += len(ref_labels)
        changed_bytes = 0
        for true_label, predicted_label in zip(ref_labels, pred_labels):
            pair_counts[(true_label, predicted_label)] += 1
            true_support[true_label] += 1
            if true_label == predicted_label:
                continue
            changed_bytes += 1
            total_confused_bytes += 1
            offdiag_pair_counts[(true_label, predicted_label)] += 1
            offdiag_as_true[true_label] += 1
            offdiag_as_pred[predicted_label] += 1

        changed_bytes_per_file.append(changed_bytes)
        if ref_labels:
            changed_fraction_per_file.append(float(changed_bytes) / float(len(ref_labels)))
        else:
            changed_fraction_per_file.append(0.0)

    train_file_support: Counter[str] = Counter()
    train_gemini_file_pair_counts: Counter[tuple[str, str]] = Counter()
    plot_offdiag_as_train: Counter[str] = Counter()
    plot_offdiag_as_gemini: Counter[str] = Counter()
    applicable_files: set[tuple[str, str, int]] = set()

    for file_key in sorted(file_keys):
        source_label = file_source_labels.get(file_key, DEFAULT_LABEL)
        train_file_support[source_label] += 1
        gemini_labels = file_gemini_labels.get(file_key) or {DEFAULT_LABEL}
        for gemini_label in sorted(gemini_labels):
            train_gemini_file_pair_counts[(source_label, gemini_label)] += 1
            if gemini_label == source_label:
                continue
            plot_offdiag_as_train[source_label] += 1
            plot_offdiag_as_gemini[gemini_label] += 1
            applicable_files.add(file_key)

    plot_labels = _rank_labels(
        plot_offdiag_as_train,
        plot_offdiag_as_gemini,
        train_file_support,
        top_k=top_k,
    )
    plot_label_to_index = {label: idx for idx, label in enumerate(plot_labels)}
    plot_raw_matrix = np.zeros((len(plot_labels), len(plot_labels)), dtype=np.int64)
    plot_rate_matrix = np.zeros((len(plot_labels), len(plot_labels)), dtype=np.float64)

    for (source_label, gemini_label), count in train_gemini_file_pair_counts.items():
        source_idx = plot_label_to_index.get(source_label)
        gemini_idx = plot_label_to_index.get(gemini_label)
        if source_idx is None or gemini_idx is None:
            continue
        plot_raw_matrix[source_idx, gemini_idx] = int(count)

    for source_label, source_idx in plot_label_to_index.items():
        denom = int(train_file_support.get(source_label, 0))
        if denom <= 0:
            continue
        plot_rate_matrix[source_idx, :] = plot_raw_matrix[source_idx, :].astype(
            np.float64
        ) / float(denom)

    selected_applicable_files: set[tuple[str, str, int]] = set()
    for file_key in applicable_files:
        source_label = file_source_labels.get(file_key, DEFAULT_LABEL)
        if source_label not in plot_label_to_index:
            continue
        gemini_labels = file_gemini_labels.get(file_key) or {DEFAULT_LABEL}
        if any(
            gemini_label != source_label and gemini_label in plot_label_to_index
            for gemini_label in gemini_labels
        ):
            selected_applicable_files.add(file_key)

    selected_labels = _rank_labels(
        offdiag_as_true,
        offdiag_as_pred,
        true_support,
        top_k=top_k,
    )
    label_to_index = {label: idx for idx, label in enumerate(selected_labels)}
    raw_matrix = np.zeros((len(selected_labels), len(selected_labels)), dtype=np.int64)
    rate_matrix = np.zeros((len(selected_labels), len(selected_labels)), dtype=np.float64)
    selected_confused_bytes = 0

    for (true_label, predicted_label), count in pair_counts.items():
        true_idx = label_to_index.get(true_label)
        pred_idx = label_to_index.get(predicted_label)
        if true_idx is None or pred_idx is None:
            continue
        raw_matrix[true_idx, pred_idx] = int(count)
        if true_label != predicted_label:
            selected_confused_bytes += int(count)

    for true_label, true_idx in label_to_index.items():
        denom = int(true_support.get(true_label, 0))
        if denom <= 0:
            continue
        rate_matrix[true_idx, :] = raw_matrix[true_idx, :].astype(np.float64) / float(denom)

    sorted_pairs = sorted(
        offdiag_pair_counts.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )
    top_pairs = [
        TopPair(
            true_label=true_label,
            predicted_label=predicted_label,
            count=int(count),
            rate_vs_true=float(count) / float(true_support[true_label])
            if true_support[true_label] > 0
            else 0.0,
        )
        for (true_label, predicted_label), count in sorted_pairs[:DEFAULT_TOP_PAIR_COUNT]
    ]

    return ConfusionAnalysis(
        row_count=row_count,
        distinct_file_count=len(file_keys),
        snippet_start_zero_count=snippet_start_zero_count,
        full_file_mode_count=full_file_mode_count,
        total_bytes=total_bytes,
        total_changed_bytes=sum(changed_bytes_per_file),
        total_confused_bytes=total_confused_bytes,
        changed_bytes_per_file=changed_bytes_per_file,
        changed_fraction_per_file=changed_fraction_per_file,
        true_support=dict(true_support),
        offdiag_as_true=dict(offdiag_as_true),
        offdiag_as_pred=dict(offdiag_as_pred),
        pair_counts=dict(pair_counts),
        train_file_support=dict(train_file_support),
        train_gemini_file_pair_counts=dict(train_gemini_file_pair_counts),
        plot_labels=plot_labels,
        plot_raw_matrix=plot_raw_matrix,
        plot_rate_matrix=plot_rate_matrix,
        applicable_file_count=len(applicable_files),
        selected_applicable_file_count=len(selected_applicable_files),
        selected_labels=selected_labels,
        selected_raw_matrix=raw_matrix,
        selected_rate_matrix=rate_matrix,
        selected_confused_bytes=selected_confused_bytes,
        top_pairs=top_pairs,
    )


def _annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    value_formatter,
    max_fontsize: int = 7,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    vmax = float(finite.max()) if finite.size else 0.0
    threshold = vmax * 0.55 if vmax > 0.0 else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if not np.isfinite(value) or value <= 0:
                continue
            color = "white" if value >= threshold else "#1b1b1b"
            ax.text(
                col_idx,
                row_idx,
                value_formatter(float(value)),
                ha="center",
                va="center",
                fontsize=max_fontsize,
                color=color,
            )


def _draw_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    matrix: np.ndarray,
    labels: Sequence[str],
    *,
    title: str,
    colorbar_label: str,
    value_formatter,
    x_label: str = "Predicted label",
    y_label: str = "True label",
    x_tick_rotation: float = 45.0,
) -> None:
    display = matrix.astype(np.float64).copy()
    np.fill_diagonal(display, np.nan)
    masked = np.ma.masked_invalid(display)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#f4f4f4")
    image = ax.imshow(masked, interpolation="nearest", aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=x_tick_rotation, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(labels), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1.0), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label, rotation=270, labelpad=16)
    _annotate_heatmap(ax, display, value_formatter=value_formatter)


def save_confusion_png(
    analysis: ConfusionAnalysis,
    out_path: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    layout: str = DEFAULT_LAYOUT,
    hide_suptitle: bool = False,
    x_tick_rotation: float | None = None,
) -> Path:
    if dpi < 300:
        raise ValueError("dpi must be at least 300 for a high-resolution PNG")
    labels = analysis.plot_labels
    if not labels:
        raise ValueError("No labels available to plot")
    if layout not in {"double", "left", "right"}:
        raise ValueError(f"Unsupported layout: {layout}")

    out_file = Path(out_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    side = max(7.5, 0.5 * len(labels) + 2.0)
    effective_rotation = float(
        x_tick_rotation if x_tick_rotation is not None else (45.0 if layout == "double" else 32.0)
    )

    if layout == "double":
        fig, axes = plt.subplots(1, 2, figsize=(side * 2.2, side), constrained_layout=True)
        _draw_panel(
            fig,
            axes[0],
            analysis.plot_raw_matrix,
            labels,
            title="Applicable files per label pair\n(diagonal hidden)",
            colorbar_label="Applicable files",
            value_formatter=lambda value: f"{int(round(value)):,}",
            x_label="Gemini label",
            y_label="Train set label",
            x_tick_rotation=effective_rotation,
        )
        _draw_panel(
            fig,
            axes[1],
            analysis.plot_rate_matrix,
            labels,
            title="Train-label file rate with Gemini label\n(diagonal hidden)",
            colorbar_label="Applicable files / train-label files",
            value_formatter=lambda value: f"{value:.1%}",
            x_label="Gemini label",
            y_label="Train set label",
            x_tick_rotation=effective_rotation,
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(side * 1.35, side), constrained_layout=True)
        if layout == "left":
            _draw_panel(
                fig,
                ax,
                analysis.plot_raw_matrix,
                labels,
                title="Applicable files per label pair\n(diagonal hidden)",
                colorbar_label="Applicable files",
                value_formatter=lambda value: f"{int(round(value)):,}",
                x_label="Gemini label",
                y_label="Train set label",
                x_tick_rotation=effective_rotation,
            )
        else:
            _draw_panel(
                fig,
                ax,
                analysis.plot_rate_matrix,
                labels,
                title="Train-label file rate with Gemini label\n(diagonal hidden)",
                colorbar_label="Applicable files / train-label files",
                value_formatter=lambda value: f"{value:.1%}",
                x_label="Gemini label",
                y_label="Train set label",
                x_tick_rotation=effective_rotation,
            )
    if not hide_suptitle:
        fig.suptitle(
            (
                f"Active Learning Refinement Label Applicability (top {len(labels)} labels)\n"
                f"Applicable files: {analysis.applicable_file_count:,} | "
                f"Subview capture: {analysis.plot_capture:.1%}"
            ),
            fontsize=13,
        )
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_file


def _ecdf_points(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    cumulative = np.arange(1, ordered.size + 1, dtype=np.float64) / float(ordered.size)
    return ordered, cumulative


def _draw_ecdf_panel(
    ax: plt.Axes,
    values: Sequence[float],
    *,
    title: str,
    x_label: str,
    percent_x_axis: bool = False,
) -> None:
    x, y = _ecdf_points(values)
    ax.step(x, y, where="post", color="#145da0", linewidth=2.2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative fraction of files")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if percent_x_axis:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))

    for q, label in ((0.5, "median"), (0.9, "p90"), (0.95, "p95")):
        value = _quantile(values, q)
        ax.axvline(value, color="#cc4c02", linestyle="--", linewidth=1.1, alpha=0.9)
        ax.text(
            value,
            min(0.98, q + 0.03),
            f"{label}: {value:.1%}" if percent_x_axis else f"{label}: {int(round(value)):,}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#7f2704",
            backgroundcolor="white",
        )


def save_change_ecdf_png(
    analysis: ConfusionAnalysis,
    out_path: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> Path:
    if dpi < 300:
        raise ValueError("dpi must be at least 300 for a high-resolution PNG")
    out_file = Path(out_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.8), constrained_layout=True)
    _draw_ecdf_panel(
        ax,
        analysis.changed_fraction_per_file,
        title="ECDF of changed fraction per file",
        x_label="Changed fraction",
        percent_x_axis=True,
    )
    fig.suptitle(
        (
            "Active Learning Change Distribution ECDF\n"
            f"Files: {analysis.distinct_file_count:,} | Total relabeled tokens: {analysis.total_changed_bytes:,}"
        ),
        fontsize=13,
    )
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_file


def _format_label_list(labels: Sequence[str]) -> str:
    if not labels:
        return "(none)"
    return ", ".join(labels)


def _print_change_plot_ideas() -> None:
    print("Follow-up plot ideas:")
    print("  - ECDF of changed tokens per file with median, p90, and p95 markers")
    print("  - ECDF of changed fraction per file")
    print("  - Log-scale histogram of changed tokens per file")
    print("  - Pareto/Lorenz-style cumulative curve of relabeled tokens by ranked files")


def print_summary(
    analysis: ConfusionAnalysis,
    confusion_out_path: str | Path,
    *,
    ecdf_out_path: str | Path | None = None,
) -> None:
    median_changed = _quantile(analysis.changed_bytes_per_file, 0.5)
    p90_changed = _quantile(analysis.changed_bytes_per_file, 0.9)
    p95_changed = _quantile(analysis.changed_bytes_per_file, 0.95)
    max_changed = max(analysis.changed_bytes_per_file) if analysis.changed_bytes_per_file else 0

    median_ratio = _quantile(analysis.changed_fraction_per_file, 0.5)
    p90_ratio = _quantile(analysis.changed_fraction_per_file, 0.9)
    p95_ratio = _quantile(analysis.changed_fraction_per_file, 0.95)
    max_ratio = max(analysis.changed_fraction_per_file) if analysis.changed_fraction_per_file else 0.0

    print(f"Analyzed refinement rows: {analysis.row_count:,}")
    print(f"Distinct files: {analysis.distinct_file_count:,}")
    print(
        "Full-file indicators:"
        f" snippet_start==0 for {analysis.snippet_start_zero_count:,}/{analysis.row_count:,} rows |"
        f" metadata.full_file_mode=true for {analysis.full_file_mode_count:,}/{analysis.row_count:,} rows"
    )
    print(f"Selected labels: {_format_label_list(analysis.selected_labels)}")
    print(
        "Confused bytes captured by selected submatrix:"
        f" {analysis.selected_confused_bytes:,}/{analysis.total_confused_bytes:,}"
        f" ({analysis.selected_capture:.1%})"
    )
    print(f"Total relabeled tokens: {analysis.total_changed_bytes:,}")
    print(
        "Changed tokens per file:"
        f" median={int(round(median_changed)):,}"
        f" p90={int(round(p90_changed)):,}"
        f" p95={int(round(p95_changed)):,}"
        f" max={int(max_changed):,}"
    )
    print(
        "Changed fraction per file:"
        f" median={median_ratio:.2%}"
        f" p90={p90_ratio:.2%}"
        f" p95={p95_ratio:.2%}"
        f" max={max_ratio:.2%}"
    )
    print("Top confusion pairs:")
    for pair in analysis.top_pairs:
        print(
            "  -"
            f" {pair.true_label} -> {pair.predicted_label}: {pair.count:,}"
            f" ({pair.rate_vs_true:.2%} of true {pair.true_label})"
        )
    print(f"Saved confusion PNG: {Path(confusion_out_path).resolve()}")
    if ecdf_out_path is not None:
        print(f"Saved ECDF PNG: {Path(ecdf_out_path).resolve()}")
    else:
        _print_change_plot_ideas()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    analysis = analyze_store(args.db, top_k=int(args.top_k))
    out_path = _resolve_output_path(args.out, top_k=int(args.top_k))
    ecdf_out_path = _resolve_ecdf_output_path(out_path)
    save_confusion_png(
        analysis,
        out_path,
        dpi=int(args.dpi),
        layout=str(args.layout),
        hide_suptitle=bool(args.hide_suptitle),
        x_tick_rotation=args.x_tick_rotation,
    )
    save_change_ecdf_png(analysis, ecdf_out_path, dpi=int(args.dpi))
    print_summary(analysis, out_path, ecdf_out_path=ecdf_out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
