from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
SCRIPT_PATH = ROOT / "scripts" / "analyze_active_learning_confusion.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from active_learning.label_store import LabelStore, StoredRefinement


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "active_learning_confusion_analysis_script",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()


def _add_refinement(
    store: LabelStore,
    *,
    sample_index: int,
    source_lang: str,
    snippet_text: str,
    predicted_segments: list[dict[str, object]],
    refined_segments: list[dict[str, object]],
) -> None:
    store.add(
        StoredRefinement(
            round_id="analysis-test-round",
            source_split="train",
            source_lang=source_lang,
            sample_index=sample_index,
            sample_hash=f"hash-{sample_index}",
            boundary_index=sample_index,
            snippet_start=0,
            snippet_end=len(snippet_text),
            snippet_text=snippet_text,
            oracle_name="stub",
            oracle_model="stub",
            oracle_run_id="",
            status="ok",
            acquisition_score=0.0,
            predicted_segments=predicted_segments,
            refined_segments=refined_segments,
            metadata={"full_file_mode": True},
        )
    )


def _build_store(store_path: Path) -> None:
    store = LabelStore(store_path)
    _add_refinement(
        store,
        sample_index=0,
        source_lang="css",
        snippet_text="aaaaaa",
        predicted_segments=[{"start": 0, "end": 6, "label": "html"}],
        refined_segments=[{"start": 0, "end": 6, "label": "css"}],
    )
    _add_refinement(
        store,
        sample_index=1,
        source_lang="html",
        snippet_text="bbbb",
        predicted_segments=[{"start": 0, "end": 4, "label": "css"}],
        refined_segments=[{"start": 0, "end": 4, "label": "html"}],
    )
    _add_refinement(
        store,
        sample_index=2,
        source_lang="html",
        snippet_text="cccccc",
        predicted_segments=[{"start": 0, "end": 6, "label": "html"}],
        refined_segments=[{"start": 0, "end": 6, "label": "html"}],
    )
    _add_refinement(
        store,
        sample_index=3,
        source_lang="css",
        snippet_text="AéB",
        predicted_segments=[{"start": 0, "end": 3, "label": "html"}],
        refined_segments=[
            {"start": 0, "end": 1, "label": "html"},
            {"start": 1, "end": 2, "label": "css"},
            {"start": 2, "end": 3, "label": "html"},
        ],
    )


def test_analyze_store_counts_utf8_bytes_and_row_normalizes_by_true_support(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "labels.sqlite"
    _build_store(store_path)

    analysis = mod.analyze_store(store_path, top_k=2)

    assert analysis.row_count == 4
    assert analysis.distinct_file_count == 4
    assert analysis.snippet_start_zero_count == 4
    assert analysis.full_file_mode_count == 4
    assert analysis.total_changed_bytes == 12
    assert analysis.total_confused_bytes == 12
    assert analysis.changed_bytes_per_file == [6, 4, 0, 2]
    assert analysis.train_file_support == {"css": 2, "html": 2}
    assert analysis.applicable_file_count == 1
    assert analysis.selected_applicable_file_count == 1
    assert analysis.plot_capture == 1.0

    assert set(analysis.plot_labels) == {"html", "css"}
    plot_label_to_idx = {label: idx for idx, label in enumerate(analysis.plot_labels)}
    plot_html_idx = plot_label_to_idx["html"]
    plot_css_idx = plot_label_to_idx["css"]
    assert int(analysis.plot_raw_matrix[plot_css_idx, plot_html_idx]) == 1
    assert int(analysis.plot_raw_matrix[plot_html_idx, plot_css_idx]) == 0
    assert float(analysis.plot_rate_matrix[plot_css_idx, plot_html_idx]) == 0.5
    assert float(analysis.plot_rate_matrix[plot_html_idx, plot_css_idx]) == 0.0

    assert set(analysis.selected_labels) == {"html", "css"}
    label_to_idx = {label: idx for idx, label in enumerate(analysis.selected_labels)}
    html_idx = label_to_idx["html"]
    css_idx = label_to_idx["css"]

    assert analysis.true_support["html"] == 12
    assert analysis.true_support["css"] == 8
    assert int(analysis.selected_raw_matrix[html_idx, css_idx]) == 4
    assert int(analysis.selected_raw_matrix[css_idx, html_idx]) == 8
    assert float(analysis.selected_rate_matrix[html_idx, css_idx]) == 4.0 / 12.0
    assert float(analysis.selected_rate_matrix[css_idx, html_idx]) == 1.0
    assert analysis.selected_confused_bytes == 12
    assert analysis.selected_capture == 1.0

    assert analysis.top_pairs[0].true_label == "css"
    assert analysis.top_pairs[0].predicted_label == "html"
    assert analysis.top_pairs[0].count == 8
    assert analysis.top_pairs[0].rate_vs_true == 1.0


def test_main_writes_high_resolution_png_and_prints_summary(
    tmp_path: Path,
    capsys,
) -> None:
    store_path = tmp_path / "labels.sqlite"
    out_path = tmp_path / "confusion.png"
    ecdf_out_path = tmp_path / "confusion_change_ecdfs.png"
    _build_store(store_path)

    exit_code = mod.main(
        [
            "--db",
            str(store_path),
            "--out",
            str(out_path),
            "--top-k",
            "2",
            "--dpi",
            "320",
        ]
    )

    assert exit_code == 0
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert ecdf_out_path.exists()
    assert ecdf_out_path.stat().st_size > 0

    with Image.open(out_path) as image:
        assert image.width >= 2500
        assert image.height >= 1000
        dpi = image.info.get("dpi")
        if dpi is not None:
            assert dpi[0] >= 299
            assert dpi[1] >= 299

    with Image.open(ecdf_out_path) as image:
        assert image.width >= 2500
        assert image.height >= 1000
        dpi = image.info.get("dpi")
        if dpi is not None:
            assert dpi[0] >= 299
            assert dpi[1] >= 299

    output = capsys.readouterr().out
    assert "Selected labels:" in output
    assert "Top confusion pairs:" in output
    assert "Saved confusion PNG:" in output
    assert "Saved ECDF PNG:" in output
