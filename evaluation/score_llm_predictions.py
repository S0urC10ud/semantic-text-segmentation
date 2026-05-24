"""Score cached LLM predictions against the dense/test ground truth.

This uses exactly the same metric pipeline as evaluation.py by exposing a
runner-shaped object with ``segment_text(...)``. It works for Gemini Flash,
Gemma via Google/OpenRouter, and any future dense-prompt cache with the same
JSONL record shape produced by ``evaluation/llm_benchmark/label_test_set.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import importlib.util

import numpy as np

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent
TRAIN_ROOT = REPO_ROOT / "train"
for p in (REPO_ROOT, TRAIN_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_EV_PATH = HERE.parent / "evaluation.py"
_spec = importlib.util.spec_from_file_location("evaluation_main", str(_EV_PATH))
ev = importlib.util.module_from_spec(_spec)
sys.modules["evaluation_main"] = ev
_spec.loader.exec_module(ev)  # type: ignore[union-attr]
import utils.config as cfg  # noqa: E402


def _canonicalize_llm_label(name: str) -> str:
    s = str(name).strip().lower().replace("-", "_")
    direct = {
        "rst": "restructuredtext",
        "restructured_text": "restructuredtext",
        "gettext_catalog": "gettext_catalog",
        "objective_c": "c_family",
        "objectivec": "c_family",
        "c++": "c_family",
        "cpp": "c_family",
        "c": "c_family",
        "javascript": "javascript_typescript",
        "typescript": "javascript_typescript",
        "js": "javascript_typescript",
        "ts": "javascript_typescript",
        "html_template": "html",
        "xhtml": "html",
        "yml": "yaml",
        "vbnet": "visual_basic",
        "vb": "visual_basic",
        "powershell_script": "powershell",
        "bash": "shell",
        "sh": "shell",
        "shellscript": "shell",
        "shell_batchfile": "shell",
        "batchfile": "shell",
    }
    if s in direct:
        return direct[s]
    if s.startswith("encoding_"):
        return s
    return s


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _percentile_key(task: str, host: str, example_id: str) -> float:
    h = hashlib.sha256(f"{task}::{host}::{example_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / (1 << 64)


def _row_host(row: dict) -> str:
    try:
        meta = json.loads(row["metadata_json"])
    except Exception:
        return "?"
    return meta.get("host_lang") or meta.get("first_lang") or "?"


def _filter_dataset_by_partial(task: str, ds, partial_pct: Optional[float]):
    if partial_pct is None:
        return ds
    p = float(partial_pct) / 100.0
    indices: list[int] = []
    for i, row in enumerate(ds):
        example_id = str(row["example_id"])
        host = _row_host(row)
        if _percentile_key(task, host, example_id) < p:
            indices.append(i)
    return ds.select(indices)


def _load_predictions(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sha = row.get("content_sha256")
            if not sha:
                continue
            existing = out.get(sha)
            if existing is not None and existing.get("status") == "ok" and row.get("status") != "ok":
                continue
            out[sha] = row
    return out


def _segments_to_char_labels(
    text: str,
    llm_segments: Sequence[Dict[str, Any]],
    *,
    other_id: int,
) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    n = len(text)
    char_labels: List[int] = [other_id] * n
    segments: List[Tuple[int, int, int]] = []
    if not llm_segments:
        return segments, char_labels

    offset = 0
    for seg in llm_segments:
        if offset >= n:
            break
        raw_type = seg.get("type") if isinstance(seg, dict) else None
        content = seg.get("content") if isinstance(seg, dict) else None
        if not isinstance(content, str) or not content:
            continue
        canon = _canonicalize_llm_label(raw_type or "other")
        label_id = cfg.LANG2ID.get(canon)
        if label_id is None:
            label_id = other_id
        end = min(offset + len(content), n)
        if end > offset:
            for i in range(offset, end):
                char_labels[i] = int(label_id)
            segments.append((offset, end, int(label_id)))
        offset = end
    return segments, char_labels


class LLMPredictionRunner:
    backend = "remote"

    def __init__(self, predictions: Dict[str, Dict[str, Any]], *, arch: str):
        self.predictions = predictions
        self.arch = arch
        self.num_classes = int(cfg.NUM_CLASSES)
        self._other_id = int(getattr(cfg, "OTHER_CLASS_INDEX", self.num_classes - 1))
        self.hits = 0
        self.misses = 0
        self.parse_fails = 0

    def segment_text(
        self, text: str, *, min_run_chars: int = 1
    ) -> Tuple[List[Tuple[int, int, int]], List[int], Optional[List[Optional[np.ndarray]]]]:
        normalized = ev.normalize_eval_text(text if isinstance(text, str) else "")
        n = len(normalized)
        row = self.predictions.get(_digest_text(normalized))
        if row is None:
            row = self.predictions.get(_digest_text(text if isinstance(text, str) else ""))
        if row is None or row.get("status") != "ok":
            self.misses += 1
            if row is not None and row.get("status") != "ok":
                self.parse_fails += 1
            char_labels = [self._other_id] * n
            segments = [(0, n, self._other_id)] if n else []
            return segments, char_labels, None

        self.hits += 1
        segments, char_labels = _segments_to_char_labels(
            normalized, row.get("segments") or [], other_id=self._other_id
        )
        if min_run_chars and min_run_chars > 1:
            char_labels = ev._smooth_min_run(char_labels, min_run_chars)
            segments = []
            if char_labels:
                cur = char_labels[0]
                start = 0
                for idx in range(1, len(char_labels)):
                    if char_labels[idx] != cur:
                        segments.append((start, idx, int(cur)))
                        start = idx
                        cur = char_labels[idx]
                segments.append((start, len(char_labels), int(cur)))
        return segments, char_labels, None


FlashRunner = LLMPredictionRunner


@dataclass
class _Args:
    arch: str = "llm-predictions"
    checkpoint: str = "llm"
    model_dim: Optional[int] = None
    channels: Sequence[int] = ()
    dtype: str = "n/a"
    sample_seed: int = 13
    other_threshold: float = 0.0
    exclude_truth_labels: Sequence[str] = ()
    magika_module_version: Optional[str] = None
    magika_model_name: Optional[str] = None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score cached LLM predictions with evaluation.py metrics")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(REPO_ROOT, "evaluation/llm_benchmark/test_runs/gemini-3-flash-preview__test_set.jsonl"),
    )
    parser.add_argument("--data-root", type=Path, default=Path(REPO_ROOT, "evaluation/test"))
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--checkpoint", default="gemini-3-flash-preview")
    parser.add_argument("--arch", default="gemini-flash")
    parser.add_argument("--partial", type=float, default=None,
                        help="Cumulative percent to score, matching label_test_set.py --partial.")
    parser.add_argument("--max-samples", type=int, default=0, help="Cap per task (0 = all).")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--other-threshold", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--min-run", type=int, default=1)
    parser.add_argument("--tasks", nargs="*", default=None)
    args_cli = parser.parse_args(argv)

    if args_cli.partial is not None and (args_cli.partial <= 0 or args_cli.partial > 100):
        raise SystemExit("--partial must be in (0, 100].")

    predictions = _load_predictions(args_cli.predictions)
    print(f"[score] loaded {len(predictions)} predictions from {args_cli.predictions}", flush=True)
    ok_rows = sum(1 for r in predictions.values() if r.get("status") == "ok")
    print(f"[score] status=ok rows: {ok_rows}", flush=True)

    manifest_path = args_cli.data_root / "manifest.json"
    manifest: Dict[str, Any] = ev._load_manifest(manifest_path) if manifest_path.exists() else {}
    datasets = ev._collect_datasets(args_cli.data_root, args_cli.tasks)
    if not datasets:
        raise RuntimeError(f"No datasets found under {args_cli.data_root}")

    task_descriptions = {
        entry.get("task"): entry.get("description", "")
        for entry in manifest.get("tasks", [])
        if isinstance(entry, dict) and entry.get("task")
    }

    prepared: Dict[str, Any] = {}
    dataset_meta: Dict[str, Dict[str, Any]] = {}
    for name, ds in datasets.items():
        ds_for_score = _filter_dataset_by_partial(name, ds, args_cli.partial)
        preserve_all = name.startswith("throughput_")
        subset, original_len, subset_len, sampled = ev._prepare_dataset(
            name,
            ds_for_score,
            max_samples=args_cli.max_samples,
            base_seed=args_cli.sample_seed,
            preserve_all=preserve_all,
        )
        prepared[name] = subset
        dataset_meta[name] = {
            "original": original_len,
            "selected": subset_len,
            "sampled": sampled,
            "description": task_descriptions.get(name, ""),
        }

    accuracy_names = [n for n in prepared if not n.startswith("throughput_")]
    print(f"[score] prepared {len(accuracy_names)} accuracy datasets", flush=True)

    normalized_predictions: Dict[str, Dict[str, Any]] = {}
    matched_raw = 0
    unmatched_raw = 0
    for ds in prepared.values():
        for row in ds:
            content = row.get("content")
            if not isinstance(content, str):
                continue
            pred = predictions.get(_digest_text(content))
            if pred is None:
                unmatched_raw += 1
                continue
            matched_raw += 1
            normalized_predictions[_digest_text(ev.normalize_eval_text(content))] = pred
    print(
        f"[score] cross-linked predictions: matched={matched_raw}, unmatched={unmatched_raw}",
        flush=True,
    )

    runner = LLMPredictionRunner(normalized_predictions, arch=args_cli.arch)
    ev.evaluate_task._log_interval = args_cli.log_interval  # type: ignore[attr-defined]

    task_metrics = []
    for name in sorted(accuracy_names, key=ev._task_name_sort_key):
        ds = prepared[name]
        meta = dataset_meta[name]
        note = " sampled" if meta["sampled"] else ""
        print(f"[score] evaluating {name}: {meta['selected']}/{meta['original']}{note}", flush=True)
        m = ev.evaluate_task(
            name,
            meta["description"],
            ds,
            runner,
            min_run_chars=args_cli.min_run,
            other_threshold=args_cli.other_threshold,
            excluded_truth_labels=None,
        )
        print(f"[score] {name} char_acc={m.overall_accuracy():.4f}", flush=True)
        task_metrics.append(m)

    print(
        f"[score] runner hits={runner.hits} misses={runner.misses} parse_fail_rows={runner.parse_fails}",
        flush=True,
    )

    fake_args = _Args(
        arch=args_cli.arch,
        checkpoint=args_cli.checkpoint,
        model_dim=None,
        channels=(),
        dtype="n/a",
        sample_seed=args_cli.sample_seed,
        other_threshold=args_cli.other_threshold,
        exclude_truth_labels=(),
    )
    comparison_payload = ev._collect_comparison_metrics(
        fake_args,
        task_metrics,
        throughput_results=[],
        manifest=manifest,
        monitor_b_report=None,
    )

    args_cli.report_path.mkdir(parents=True, exist_ok=True)
    out_json = args_cli.report_path / "comparison_metrics.json"
    ev._write_comparison_metrics(out_json, comparison_payload)
    print(f"[score] comparison metrics written to {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
