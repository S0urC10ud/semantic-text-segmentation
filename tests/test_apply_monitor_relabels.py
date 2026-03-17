from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from active_learning.apply_monitor_relabels import (
    FILE_DTYPE,
    SEG_DTYPE,
    _bytes_to_ascii_text,
    _group_refinements,
    _hash_monitor_file,
    _iter_refinement_rows,
    build_split_plan,
    run_writeback,
)
from active_learning.label_store import LabelStore, StoredRefinement


def _write_monitor_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    file0 = b"abcDEFgh"
    file1 = b"puts hi\n"
    contents = file0 + file1
    (root / "contents.bin").write_bytes(contents)

    files = np.zeros(2, dtype=FILE_DTYPE)
    files[0] = (0, len(file0), 0, 1, 0, 0)
    files[1] = (len(file0), len(file1), 1, 1, 1, 2)
    np.save(root / "files.npy", files)

    segments = np.zeros(2, dtype=SEG_DTYPE)
    segments[0] = (0, 0, len(file0), 0)
    segments[1] = (1, 0, len(file1), 2)
    np.save(root / "segments.npy", segments)

    meta = {
        "lang2id": {"html": 0, "css": 1, "ruby": 2, "yaml": 3, "python": 4, "other": 5},
        "id2lang": {"0": "html", "1": "css", "2": "ruby", "3": "yaml", "4": "python", "5": "other"},
        "num_files": 2,
        "num_segments": 2,
        "total_bytes": len(contents),
        "sources": {"segmented": 1, "pure": 1},
        "files_per_label": {"html": 1, "ruby": 1},
        "bytes_per_label": {"html": len(file0), "ruby": len(file1)},
        "segment_bytes_per_label": {"html": len(file0), "ruby": len(file1)},
        "max_files_per_type": 1,
        "split_name": root.name,
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _add_refinement(store_path: Path, *, root: Path, split_name: str, sample_index: int, raw: bytes, sample_hash: str | None = None, snippet_text: str | None = None) -> None:
    store = LabelStore(store_path)
    store.add_many(
        [
            StoredRefinement(
                round_id="monitor-relabel-test",
                source_split=split_name,
                source_lang="html",
                sample_index=sample_index,
                sample_hash=sample_hash or _hash_monitor_file(split_name, sample_index, np.frombuffer(raw, dtype=np.uint8)),
                boundary_index=0,
                snippet_start=0,
                snippet_end=len(raw),
                snippet_text=snippet_text if snippet_text is not None else _bytes_to_ascii_text(np.frombuffer(raw, dtype=np.uint8)),
                oracle_name="stub",
                oracle_model="stub",
                oracle_run_id="",
                status="ok",
                acquisition_score=1.0,
                predicted_segments=[{"start": 0, "end": len(raw), "label": "html"}],
                refined_segments=[
                    {"start": 0, "end": 3, "label": "html"},
                    {"start": 3, "end": len(raw), "label": "css"},
                ],
                metadata={},
            )
        ]
    )


def test_build_split_plan_updates_only_segments_and_files(tmp_path: Path) -> None:
    root = tmp_path / "monitor_preprocessed_a"
    _write_monitor_root(root)
    store_path = tmp_path / "label_store.sqlite"

    raw0 = (root / "contents.bin").read_bytes()[:8]
    _add_refinement(store_path, root=root, split_name="monitor_a", sample_index=0, raw=raw0)

    grouped = _group_refinements(
        _iter_refinement_rows(store_path, round_id="monitor-relabel-test", splits=["monitor_a"])
    )
    plan = build_split_plan(split_name="monitor_a", root=root, refinements=grouped["monitor_a"])

    assert plan.file_updates == 1
    assert plan.segments_before == 2
    assert plan.segments_after == 3
    assert plan.new_files[0]["seg_count"] == 2
    assert plan.new_files[1]["seg_count"] == 1
    assert tuple(plan.new_segments[0]) == (0, 0, 3, 0)
    assert tuple(plan.new_segments[1]) == (0, 3, 8, 1)
    assert tuple(plan.new_segments[2]) == (1, 0, 8, 2)
    assert int(plan.new_meta["num_segments"]) == 3
    assert int(plan.new_meta["total_bytes"]) == 16


def test_run_writeback_apply_rewrites_root_and_keeps_contents(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "monitor_preprocessed_a"
    _write_monitor_root(root)
    store_path = tmp_path / "label_store.sqlite"
    original_contents = (root / "contents.bin").read_bytes()

    raw0 = original_contents[:8]
    _add_refinement(store_path, root=root, split_name="monitor_a", sample_index=0, raw=raw0)

    from active_learning import apply_monitor_relabels as mod

    monkeypatch.setattr(mod, "DEFAULT_STORE_PATH", store_path)
    monkeypatch.setattr(mod, "DEFAULT_SPLIT_ROOTS", {"monitor_a": root, "monitor_b": tmp_path / "unused_b"})

    class Args:
        store = store_path
        round_id = "monitor-relabel-test"
        latest_monitor_round = False
        monitor_splits = "monitor_a"
        apply = True

    result = run_writeback(Args())
    assert result["status"] == "ok"
    assert result["updated_files"] == 1
    assert (root / "contents.bin").read_bytes() == original_contents

    files = np.load(root / "files.npy", mmap_mode="r")
    segments = np.load(root / "segments.npy", mmap_mode="r")
    assert int(files[0]["seg_count"]) == 2
    assert len(segments) == 3

    with open(root / "meta.json", "r") as f:
        meta = json.load(f)
    assert int(meta["num_segments"]) == 3
    assert (root / ".relabel_writeback_backups" / "monitor_preprocessed_a" / "monitor-relabel-test").exists()


def test_build_split_plan_rejects_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "monitor_preprocessed_a"
    _write_monitor_root(root)
    store_path = tmp_path / "label_store.sqlite"

    raw0 = (root / "contents.bin").read_bytes()[:8]
    _add_refinement(
        store_path,
        root=root,
        split_name="monitor_a",
        sample_index=0,
        raw=raw0,
        sample_hash="deadbeefdeadbeefdeadbeefdeadbeef",
    )

    grouped = _group_refinements(
        _iter_refinement_rows(store_path, round_id="monitor-relabel-test", splits=["monitor_a"])
    )
    try:
        build_split_plan(split_name="monitor_a", root=root, refinements=grouped["monitor_a"])
    except ValueError as exc:
        assert "Hash mismatch" in str(exc)
    else:
        raise AssertionError("expected hash mismatch to raise")


def test_build_split_plan_rejects_snippet_text_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "monitor_preprocessed_a"
    _write_monitor_root(root)
    store_path = tmp_path / "label_store.sqlite"

    raw0 = (root / "contents.bin").read_bytes()[:8]
    _add_refinement(
        store_path,
        root=root,
        split_name="monitor_a",
        sample_index=0,
        raw=raw0,
        snippet_text="WRONGTXT",
    )

    grouped = _group_refinements(
        _iter_refinement_rows(store_path, round_id="monitor-relabel-test", splits=["monitor_a"])
    )
    try:
        build_split_plan(split_name="monitor_a", root=root, refinements=grouped["monitor_a"])
    except ValueError as exc:
        assert "Snippet text mismatch" in str(exc)
    else:
        raise AssertionError("expected snippet mismatch to raise")
