#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional

from datasets import load_from_disk  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = PROJECT_ROOT / "gemini_segmentations"
DEFAULT_SEGMENT_ROOT = SNAPSHOT_ROOT
DATASET_CHOICES = ("monitor", "test")
DEFAULT_DATASET = "monitor"
ARROW_OUT_ROOT = Path(__file__).resolve().parent / "arrow_out"
DEFAULT_DATASET_ROOTS = {split: ARROW_OUT_ROOT / split for split in DATASET_CHOICES}
DEFAULT_CHECK_SCRIPT = Path(__file__).resolve().parent / "utils" / "llm_requestor.py"
DEFAULT_PURITY_SCRIPT = Path(__file__).resolve().parent / "utils" / "llm_purity_checker.py"
DEFAULT_TEMP_DIR = Path(tempfile.gettempdir()) / "gemini_monitor_inputs"
BLACKLIST_DIR = PROJECT_ROOT / "gemini_segmentations" / "blacklist"

ALLOWED_LANGS = {
    "php",
    "javascript",
    "typescript",
    "yaml",
    "html",
    "shell",
    "powershell",
    "dockerfile",
    "markdown",
    "restructuredtext",
    "javascript_typescript",
    "python"
}
ALLOWED_LANGS_LOWER = {lang.lower() for lang in ALLOWED_LANGS}

MODEL_PRICING_USD_PER_MTOKENS = {
    "gemini-2.5-pro": {
        "prompt": [
            {"max_prompt_tokens": 200_000, "rate": 1.25},
            {"max_prompt_tokens": None, "rate": 2.50},
        ],
        "response": [
            {"max_prompt_tokens": 200_000, "rate": 10.00},
            {"max_prompt_tokens": None, "rate": 15.00},
        ],
        "notes": "Uses ≤200K-token tiered pricing for both prompt and response tokens.",
    },
    "gemini-2.5-flash": {
        "prompt": [
            {"max_prompt_tokens": None, "rate": 0.30},
        ],
        "response": [
            {"max_prompt_tokens": None, "rate": 2.50},
        ],
        "notes": None,
    },
}


class RunStats:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.response_tokens = 0
        self.total_tokens = 0
        self.estimated_cost = 0.0
        self.samples = 0
        self.purity_prompt_tokens = 0
        self.purity_response_tokens = 0
        self.purity_total_tokens = 0
        self.purity_estimated_cost = 0.0
        self.purity_checks = 0
        self.model_usage: dict[str, dict[str, float]] = {}

    def update(
        self,
        prompt_tokens: Optional[int],
        response_tokens: Optional[int],
        total_tokens: Optional[int],
        estimated_cost: Optional[float],
        model: Optional[str] = None,
    ) -> None:
        if prompt_tokens is not None:
            self.prompt_tokens += prompt_tokens
        if response_tokens is not None:
            self.response_tokens += response_tokens
        if total_tokens is not None:
            self.total_tokens += total_tokens
        if estimated_cost is not None:
            self.estimated_cost += estimated_cost
        self.samples += 1
        if model:
            self._update_model_usage(model, prompt_tokens, response_tokens, total_tokens, estimated_cost)

    def update_purity(
        self,
        prompt_tokens: Optional[int],
        response_tokens: Optional[int],
        total_tokens: Optional[int],
        estimated_cost: Optional[float],
    ) -> None:
        if prompt_tokens is not None:
            self.purity_prompt_tokens += prompt_tokens
        if response_tokens is not None:
            self.purity_response_tokens += response_tokens
        if total_tokens is not None:
            self.purity_total_tokens += total_tokens
        if estimated_cost is not None:
            self.purity_estimated_cost += estimated_cost
        self.purity_checks += 1

    def _update_model_usage(
        self,
        model: str,
        prompt_tokens: Optional[int],
        response_tokens: Optional[int],
        total_tokens: Optional[int],
        estimated_cost: Optional[float],
    ) -> None:
        entry = self.model_usage.setdefault(
            model,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "response_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
            },
        )
        entry["requests"] += 1
        if prompt_tokens is not None:
            entry["prompt_tokens"] += prompt_tokens
        if response_tokens is not None:
            entry["response_tokens"] += response_tokens
        if total_tokens is not None:
            entry["total_tokens"] += total_tokens
        if estimated_cost is not None:
            entry["estimated_cost"] += estimated_cost


class GlobalProgress:
    def __init__(self, total_target: Optional[int]) -> None:
        self.total_target = total_target
        self.existing_total = 0
        self.new_created = 0

    def add_existing(self, count: int) -> None:
        self.existing_total += count

    def add_new(self, count: int) -> None:
        self.new_created += count

    def fmt_overall(self, extra_new: int = 0) -> str:
        current = self.existing_total + self.new_created + extra_new
        if self.total_target:
            return f"{current}/{self.total_target} target outputs"
        return f"{current} outputs stored"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment monitor/test samples with Gemini and store outputs under "
            "gemini_segmentations/<split>/<lang>/<sha256>.json"
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=DEFAULT_DATASET,
        help="Dataset split to process (controls both the arrow_out input and gemini_segmentations output subfolders).",
    )
    parser.add_argument(
        "--dataset-root",
        "--monitor-root",
        dest="dataset_root",
        type=Path,
        default=None,
        help=(
            "Root directory containing arrow_out/<split> data (default: downloader/arrow_out/<dataset>). "
            "The legacy flag name --monitor-root is preserved for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--segment-root",
        type=Path,
        default=None,
        help=(
            "Destination directory root for language-specific outputs "
            "(default: gemini_segmentations/<dataset>)."
        ),
    )
    parser.add_argument(
        "--check-script",
        type=Path,
        default=DEFAULT_CHECK_SCRIPT,
        help="Path to downloader/999_check_gemini.py.",
    )
    parser.add_argument(
        "--purity-script",
        type=Path,
        default=DEFAULT_PURITY_SCRIPT,
        help="Path to downloader/999_check_purity.py for the initial purity pass.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=DEFAULT_TEMP_DIR,
        help=(
            "Directory for temporary <hash>.tmp files that are fed into 999_check_gemini "
            "(default: system temp under 'gemini_monitor_inputs')."
        ),
    )
    parser.add_argument(
        "--skip-purity-check",
        action="store_true",
        help="Disable the initial purity prompt and always call 999_check_gemini.",
    )
    parser.add_argument(
        "--show-segmentation-output",
        action="store_true",
        help="Print raw stdout/stderr from 999_check_gemini (hidden by default to reduce noise).",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        help=(
            "Optional subset of languages to process. Values outside the allow-list are ignored. "
            "Defaults to every allow-listed language present under arrow_out/<split>."
        ),
    )
    parser.add_argument(
        "--max-per-lang",
        type=int,
        default=100,
        help=(
            "Optional cap on stored samples per language (enforced by counting existing outputs "
            "under the destination directory)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run segmentation even if gemini_segmentations/<split>/<lang>/<hash>.json already exists.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Process only the next unprocessed sample (useful to preview behavior).",
    )
    parser.add_argument(
        "--pro-only",
        action="store_true",
        help="Use gemini-2.5-pro for segmentation (999_check_gemini); purity checks remain unchanged.",
    )
    args = parser.parse_args()
    if args.dataset_root is None:
        default_root = DEFAULT_DATASET_ROOTS.get(args.dataset)
        if default_root is None:
            raise RuntimeError(f"No default dataset root registered for split '{args.dataset}'.")
        args.dataset_root = default_root
    if args.segment_root is None:
        args.segment_root = DEFAULT_SEGMENT_ROOT / args.dataset
    return args


def _list_snapshot_files(root: Path) -> Dict[str, Path]:
    return {p.name: p for p in root.glob("*.json")}


def _build_purity_metadata(
    purity_payload: dict,
    *,
    used_purity_snapshot: bool,
    skip_purity_check: bool,
) -> Optional[dict]:
    if purity_payload:
        return {
            "status": "purity_snapshot_used" if used_purity_snapshot else "segmented_after_purity",
            "is_pure": bool(purity_payload.get("is_pure")),
            "language": purity_payload.get("language"),
            "mixed_types": purity_payload.get("mixed_types"),
            "reason": purity_payload.get("reason"),
            "run_id": purity_payload.get("run_id"),
            "model": purity_payload.get("model"),
            "usage_metadata": purity_payload.get("usage_metadata"),
            "input_sha256": purity_payload.get("input_sha256"),
        }
    if skip_purity_check:
        return {"status": "skipped"}
    return None


def _ensure_lang_dirs(dataset_root: Path, allowed_langs: set[str], requested: set[str] | None) -> Dict[str, Path]:
    results: Dict[str, Path] = {}
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root '{dataset_root}' does not exist.")
    for lang_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        lang_key = lang_dir.name.lower()
        if lang_key not in allowed_langs:
            continue
        if requested and lang_key not in requested:
            continue
        dataset_dir = lang_dir / "dataset"
        if not dataset_dir.exists():
            continue
        results[lang_dir.name] = dataset_dir
    if not results:
        raise RuntimeError(
            "No matching datasets were found. "
            f"Check that '{dataset_root}' contains the requested languages."
        )
    return results


def _write_temp_input(content: str, temp_dir: Path, file_hash: str) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = temp_dir / f"{file_hash}.tmp"
    tmp_path.write_text(content, encoding="utf-8")
    return tmp_path


def _blacklist_path(file_hash: str) -> Path:
    return BLACKLIST_DIR / f"{file_hash}.fail"


def _append_meta_stats(stats: RunStats, args: argparse.Namespace, *, status: str) -> None:
    combined_cost = stats.estimated_cost + stats.purity_estimated_cost
    current_snapshot = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": status,
        "dataset": args.dataset,
        "max_per_lang": args.max_per_lang,
        "overwrite": args.overwrite,
        "samples_processed": stats.samples,
        "prompt_tokens": stats.prompt_tokens,
        "response_tokens": stats.response_tokens,
        "total_tokens": stats.total_tokens,
        "estimated_cost": stats.estimated_cost,
        "purity_estimated_cost": stats.purity_estimated_cost,
        "combined_estimated_cost": combined_cost,
        "cost_breakdown": {
            "segmentation_total": stats.estimated_cost,
            "purity_total": stats.purity_estimated_cost,
            "combined_total": combined_cost,
            "by_model": {
                model: {
                    "estimated_cost": usage.get("estimated_cost", 0.0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "response_tokens": usage.get("response_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "requests": usage.get("requests", 0),
                }
                for model, usage in stats.model_usage.items()
            },
        },
        "purity_checks": stats.purity_checks,
        "purity_prompt_tokens": stats.purity_prompt_tokens,
        "purity_response_tokens": stats.purity_response_tokens,
        "purity_total_tokens": stats.purity_total_tokens,
        "purity_estimated_cost": stats.purity_estimated_cost,
        "model_usage": stats.model_usage,
        "priced_with": "prompt + effective_response (using total_token_count when larger)",
    }
    meta_path = SNAPSHOT_ROOT / "meta_stats.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if not isinstance(existing, dict):
            existing = {}
    except Exception:
        existing = {}

    prev_cumulative = existing.get("cumulative") or {}
    prev_snapshot = existing.get("last_snapshot") or {}

    def _get_prev(field: str) -> float:
        val = prev_snapshot.get(field, 0)
        return val if isinstance(val, (int, float)) else 0

    cumulative = dict(prev_cumulative)
    numeric_fields = [
        "samples_processed",
        "prompt_tokens",
        "response_tokens",
        "total_tokens",
        "estimated_cost",
        "purity_prompt_tokens",
        "purity_response_tokens",
        "purity_total_tokens",
        "purity_estimated_cost",
        "combined_estimated_cost",
    ]
    for field in numeric_fields:
        prev_val = cumulative.get(field, 0) if isinstance(cumulative.get(field, 0), (int, float)) else 0
        delta = current_snapshot.get(field, 0) - _get_prev(field)
        cumulative[field] = prev_val + delta

    # Merge model usage cumulatively using deltas from last_snapshot.
    cumulative_model_usage = cumulative.get("model_usage", {}) if isinstance(cumulative.get("model_usage", {}), dict) else {}
    prev_model_usage = prev_snapshot.get("model_usage", {}) if isinstance(prev_snapshot.get("model_usage", {}), dict) else {}
    for model, usage in (current_snapshot.get("model_usage") or {}).items():
        prev_usage = prev_model_usage.get(model, {})
        cum_usage = cumulative_model_usage.get(model, {"requests": 0, "prompt_tokens": 0, "response_tokens": 0, "total_tokens": 0, "estimated_cost": 0.0})
        def _num(val): return val if isinstance(val, (int, float)) else 0
        cum_usage["requests"] = _num(cum_usage.get("requests")) + _num(usage.get("requests")) - _num(prev_usage.get("requests"))
        cum_usage["prompt_tokens"] = _num(cum_usage.get("prompt_tokens")) + _num(usage.get("prompt_tokens")) - _num(prev_usage.get("prompt_tokens"))
        cum_usage["response_tokens"] = _num(cum_usage.get("response_tokens")) + _num(usage.get("response_tokens")) - _num(prev_usage.get("response_tokens"))
        cum_usage["total_tokens"] = _num(cum_usage.get("total_tokens")) + _num(usage.get("total_tokens")) - _num(prev_usage.get("total_tokens"))
        cum_usage["estimated_cost"] = _num(cum_usage.get("estimated_cost")) + _num(usage.get("estimated_cost")) - _num(prev_usage.get("estimated_cost"))
        cumulative_model_usage[model] = cum_usage
    cumulative["model_usage"] = cumulative_model_usage

    # Recompute combined cost breakdown for cumulative.
    cumulative["cost_breakdown"] = {
        "segmentation_total": cumulative.get("estimated_cost", 0.0),
        "purity_total": cumulative.get("purity_estimated_cost", 0.0),
        "combined_total": cumulative.get("estimated_cost", 0.0) + cumulative.get("purity_estimated_cost", 0.0),
        "by_model": {
            model: {
                "estimated_cost": usage.get("estimated_cost", 0.0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "response_tokens": usage.get("response_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "requests": usage.get("requests", 0),
            }
            for model, usage in cumulative_model_usage.items()
        },
    }

    state = {
        "timestamp": current_snapshot["timestamp"],
        "status": status,
        "cumulative": cumulative,
        "last_snapshot": current_snapshot,
    }
    meta_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_json_from_stdout(output: str) -> dict:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def _run_purity_script(purity_script: Path, input_file: Path) -> tuple[bool, Optional[Path], dict]:
    if not purity_script.exists():
        raise FileNotFoundError(f"Purity script '{purity_script}' does not exist.")
    cmd = [
        sys.executable,
        str(purity_script),
        "--file",
        str(input_file),
        "--json-output",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"{purity_script.name} exited with status {result.returncode} for '{input_file}'. {stderr}"
        )
    payload = _parse_json_from_stdout(result.stdout)
    if not payload:
        raise RuntimeError(f"{purity_script.name} produced no parseable JSON for '{input_file}'.")
    is_pure = bool(payload.get("is_pure"))
    snapshot_value = payload.get("snapshot_path")
    snapshot_path = Path(snapshot_value) if snapshot_value else None
    if is_pure and (snapshot_path is None or not snapshot_path.exists()):
        # Fall back to segmentation if the purity check did not emit a snapshot.
        return False, None, payload
    return is_pure, snapshot_path, payload


def _extract_usage(payload: dict) -> tuple[Optional[int], Optional[int], Optional[int], Optional[str]]:
    usage = payload.get("usage_metadata") or {}
    prompt_tokens = usage.get("prompt_token_count")
    response_tokens = usage.get("candidates_token_count")
    total_tokens = usage.get("total_token_count")
    model = payload.get("model")
    return prompt_tokens, response_tokens, total_tokens, model


def _run_check_script(
    check_script: Path,
    input_file: Path,
    *,
    verbose: bool,
    model: str | None = None,
) -> Path:
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    before = _list_snapshot_files(SNAPSHOT_ROOT)
    cmd = [sys.executable, str(check_script), "--file", str(input_file), "--fuzzy"]
    if model:
        cmd.extend(["--model", model])
    capture = not verbose
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=capture,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if capture else ""
        stdout = result.stdout.strip() if capture else ""
        msg = f"999_check_gemini exited with status {result.returncode} for '{input_file}'."
        if stderr or stdout:
            msg += f" stdout={stdout!r} stderr={stderr!r}"
        raise RuntimeError(msg)
    after = _list_snapshot_files(SNAPSHOT_ROOT)
    new_entries = [after[name] for name in after.keys() - before.keys()]
    if not new_entries:
        raise RuntimeError("No new segmentation snapshot produced by 999_check_gemini.")
    if len(new_entries) == 1:
        return new_entries[0]
    # Multiple files were created; pick the newest by modification time.
    return max(new_entries, key=lambda p: p.stat().st_mtime)


def _load_snapshot_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _calculate_tiered_cost(token_count: Optional[int], tiers: list[dict]) -> Optional[float]:
    if token_count is None:
        return None
    remaining = token_count
    cost = 0.0
    for tier in tiers:
        if remaining <= 0:
            break
        limit = tier.get("max_prompt_tokens")
        chunk = remaining if limit is None else min(remaining, limit)
        cost += (chunk / 1_000_000) * tier["rate"]
        remaining -= chunk
    if remaining > 0 and tiers:
        cost += (remaining / 1_000_000) * tiers[-1]["rate"]
    return cost


def _estimate_cost(
    model: Optional[str],
    prompt_tokens: Optional[int],
    response_tokens: Optional[int],
    total_tokens: Optional[int] = None,
) -> Optional[float]:
    if model is None:
        return None
    pricing = MODEL_PRICING_USD_PER_MTOKENS.get(model)
    if not pricing:
        return None
    prompt_cost = _calculate_tiered_cost(prompt_tokens, pricing.get("prompt", []))
    effective_response_tokens = response_tokens
    if (
        effective_response_tokens is None
        and total_tokens is not None
        and isinstance(prompt_tokens, int)
    ):
        effective_response_tokens = max(total_tokens - prompt_tokens, 0)
    if (
        effective_response_tokens is not None
        and total_tokens is not None
        and isinstance(prompt_tokens, int)
    ):
        effective_response_tokens = max(effective_response_tokens, total_tokens - prompt_tokens)
    response_cost = _calculate_tiered_cost(effective_response_tokens, pricing.get("response", []))
    if prompt_cost is None and response_cost is None:
        return None
    total_cost = (prompt_cost or 0.0) + (response_cost or 0.0)
    return total_cost


def _print_sample_summary(
    *,
    lang: str,
    sample_idx: int,
    dest_path: Path,
    snapshot_data: dict,
    stats: RunStats,
) -> Optional[float]:
    metadata = snapshot_data.get("metadata", {})
    usage = metadata.get("usage_metadata") or {}
    model = metadata.get("model")
    prompt_tokens = usage.get("prompt_token_count")
    response_tokens = usage.get("candidates_token_count")
    total_tokens = usage.get("total_token_count")
    estimated_cost = _estimate_cost(model, prompt_tokens, response_tokens, total_tokens)
    stats.update(prompt_tokens, response_tokens, total_tokens, estimated_cost, model=model)
    run_id = snapshot_data.get("run_id")
    fmt_prompt = f"{prompt_tokens:,}" if isinstance(prompt_tokens, int) else "n/a"
    fmt_response = f"{response_tokens:,}" if isinstance(response_tokens, int) else "n/a"
    fmt_total = f"{total_tokens:,}" if isinstance(total_tokens, int) else "n/a"
    fmt_cost = f"${estimated_cost:.4f}" if isinstance(estimated_cost, float) else "n/a"
    fmt_running_cost = f"${stats.estimated_cost:.4f}"
    print(
        (
            f"{lang}: stored sample {sample_idx} as {dest_path.name} "
            f"(model={model}, prompt={fmt_prompt}, response={fmt_response}, total={fmt_total}, est_cost={fmt_cost}); "
            f"run_totals prompt={stats.prompt_tokens:,}, response={stats.response_tokens:,}, "
            f"total={stats.total_tokens:,}, est_cost={fmt_running_cost}."
        )
    )
    return estimated_cost


def _print_remaining_forecast(
    *,
    lang: str,
    estimated_remaining: int,
    dataset_remaining: int,
    avg_cost_per_sample: Optional[float],
    quota_limited: bool,
    overwrite_mode: bool,
) -> None:
    estimated_remaining = max(estimated_remaining, 0)
    dataset_remaining = max(dataset_remaining, 0)
    quota_note = ""
    if quota_limited and dataset_remaining > estimated_remaining:
        if overwrite_mode:
            quota_note = (
                f" (overwriting run; new-sample quota reached but dataset entries left: {dataset_remaining})"
            )
        else:
            quota_note = f" (limited by max-per-lang; dataset entries left: {dataset_remaining})"
    elif dataset_remaining:
        quota_note = f" (dataset entries left: {dataset_remaining})"
    remaining_cost = None
    if estimated_remaining > 0 and avg_cost_per_sample is not None:
        remaining_cost = avg_cost_per_sample * estimated_remaining
    elif estimated_remaining == 0 and avg_cost_per_sample is not None:
        remaining_cost = 0.0
    if remaining_cost is None:
        fmt_cost = "n/a"
    else:
        fmt_cost = f"${remaining_cost:.4f}"
    avg_note = ""
    if estimated_remaining > 0 and avg_cost_per_sample is not None:
        avg_note = f" (avg sample cost ≈ ${avg_cost_per_sample:.4f})"
    print(
        f"{lang}: remaining samples requiring Gemini calls ≈ {estimated_remaining}{quota_note}. "
        f"Rough remaining cost: {fmt_cost}{avg_note}."
    )


def process_language(
    lang: str,
    dataset_dir: Path,
    args: argparse.Namespace,
    stats: RunStats,
    global_progress: GlobalProgress,
) -> bool:
    dataset = load_from_disk(str(dataset_dir))
    total = len(dataset)
    print(f"{lang}: Loaded {total} samples from {dataset_dir}.")
    if args.pro_only:
        print(
            f"{lang}: PRO-ONLY MODE ENABLED → segmentation will use gemini-2.5-pro (flash skipped).",
        )
    lang_output_dir = args.segment_root / lang
    lang_output_dir.mkdir(parents=True, exist_ok=True)
    existing_output_count = sum(1 for _ in lang_output_dir.glob("*.json"))
    global_progress.add_existing(existing_output_count)
    remaining_new_slots: Optional[int] = None
    quota_stop_logged = False
    forecast_emitted = False
    lang_cost_total = 0.0
    lang_cost_samples = 0
    target_note = (
        f"{existing_output_count}/{args.max_per_lang} target for {lang}"
        if args.max_per_lang
        else f"{existing_output_count} existing for {lang}"
    )
    print(
        f"{lang}: {target_note}. Overall: {global_progress.fmt_overall()}. "
        f"Dataset entries: {total}."
    )
    if args.max_per_lang:
        remaining_new_slots = args.max_per_lang - existing_output_count
        if remaining_new_slots < 0:
            remaining_new_slots = 0
        if remaining_new_slots == 0 and not args.overwrite:
            print(
                f"{lang}: Output directory {lang_output_dir} already has {existing_output_count} files "
                f"which meets/exceeds --max-per-lang={args.max_per_lang}; skipping."
            )
            _print_remaining_forecast(
                lang=lang,
                estimated_remaining=0,
                dataset_remaining=total,
                avg_cost_per_sample=None,
                quota_limited=True,
                overwrite_mode=args.overwrite,
            )
            return False
        if remaining_new_slots == 0 and args.overwrite:
            print(
                f"{lang}: Output quota satisfied ({existing_output_count} >= --max-per-lang={args.max_per_lang}); "
                "new samples will be skipped while overwriting existing files."
            )
            quota_stop_logged = True
        elif remaining_new_slots > 0 and existing_output_count:
            print(
                f"{lang}: existing outputs={existing_output_count}; "
                f"remaining new slots before --max-per-lang={args.max_per_lang}: {remaining_new_slots}."
            )
    processed = 0
    skipped_existing = 0
    processed_any = False
    new_outputs_created = 0
    for idx, record in enumerate(dataset):
        content = record.get("content")
        if not isinstance(content, str):
            print(f"[{lang}] Skipping sample {idx} with missing 'content'.")
            continue
        if len(content) <= 20:
            print(f"[{lang}] Skipping sample {idx} (<=20 characters; too short to learn from).")
            continue
        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        blacklist_file = _blacklist_path(file_hash)
        if blacklist_file.exists():
            print(f"{lang}: Sample {idx} is blacklisted ({blacklist_file}); skipping.")
            continue
        dest_path = lang_output_dir / f"{file_hash}.json"
        dest_exists = dest_path.exists()
        if dest_exists and not args.overwrite:
            skipped_existing += 1
            if skipped_existing <= 5:
                print(f"{lang}: Sample {idx} already processed ({dest_path}); skipping.")
            continue
        is_new_output = not dest_exists
        if (
            is_new_output
            and remaining_new_slots is not None
            and remaining_new_slots <= 0
        ):
            if not quota_stop_logged:
                print(
                    f"{lang}: Reached --max-per-lang={args.max_per_lang} when accounting for existing outputs; "
                    "no additional new samples will be processed."
                )
                quota_stop_logged = True
            if args.overwrite:
                continue
            break
        progress_note = (
            f"{lang}: sample {idx + 1}/{total} "
            f"(new_this_run={new_outputs_created}, skipped_existing={skipped_existing}); "
            f"overall {global_progress.fmt_overall(extra_new=new_outputs_created)}"
        )
        print(progress_note)
        temp_input = _write_temp_input(content, args.temp_dir, file_hash)
        try:
            snapshot_path: Path
            snapshot_payload: dict
            purity_payload: dict = {}
            used_purity = False
            purity_snapshot_for_cleanup: Path | None = None
            if not args.skip_purity_check:
                try:
                    is_pure, purity_snapshot_path, purity_payload = _run_purity_script(
                        args.purity_script, temp_input
                    )
                except Exception as exc:
                    print(
                        f"[{lang}] Purity check failed for sample {idx}; falling back to segmentation ({exc})."
                    )
                else:
                    if is_pure and purity_snapshot_path:
                        purity_snapshot_for_cleanup = purity_snapshot_path
                        lang_guess_raw = purity_payload.get("language")
                        lang_guess = lang_guess_raw.lower() if isinstance(lang_guess_raw, str) else ""
                        lang_match = lang_guess == lang.lower()
                        if not lang_match and lang.lower() == "javascript_typescript":
                            if lang_guess in {"javascript", "typescript", "javascript_typescript", "js", "ts"}:
                                lang_match = True
                        if lang_match:
                            snapshot_path = purity_snapshot_path
                            snapshot_payload = _load_snapshot_payload(snapshot_path)
                            used_purity = True
                    reason = purity_payload.get("reason") or "pure content"
                    lang_guess = purity_payload.get("language") or "unknown"
                    p_prompt, p_response, p_total, p_model = _extract_usage(purity_payload)
                    purity_cost = _estimate_cost(p_model, p_prompt, p_response)
                    stats.update_purity(p_prompt, p_response, p_total, purity_cost)
                    cost_note = f", est_cost={purity_cost:.4f}" if purity_cost is not None else ""
                    print(
                        f"{lang}: Purity check → is_pure={is_pure}, language={lang_guess}, reason={reason}"
                        f"{cost_note}"
                    )
                    if is_pure and not used_purity and purity_snapshot_for_cleanup:
                        print(
                            f"{lang}: Purity language mismatch; discarding purity snapshot and running full segmentation."
                        )
                    if used_purity:
                        print(
                            f"{lang}: Purity accepted; using single-segment snapshot {snapshot_path.name}."
                        )
            if not used_purity:
                if args.pro_only:
                    print(
                        f"[{lang}] PRO-ONLY MODE: calling 999_check_gemini with gemini-2.5-pro (flash skipped)."
                    )
                    try:
                        snapshot_path = _run_check_script(
                            args.check_script,
                            temp_input,
                            verbose=args.show_segmentation_output,
                            model="gemini-2.5-pro",
                        )
                    except RuntimeError as exc:
                        BLACKLIST_DIR.mkdir(parents=True, exist_ok=True)
                        blacklist_file.write_text(
                            f"lang={lang} idx={idx} hash={file_hash} reason={exc}",
                            encoding="utf-8",
                        )
                        print(
                            f"[{lang}] Segmentation failed in pro-only mode; blacklisting sample ({blacklist_file})."
                        )
                        continue
                else:
                    fallback_model = "gemini-2.5-pro"
                    segmentation_failed = False
                    snapshot_path: Path
                    current_model = None  # default = flash
                    attempted_model_switch = False
                    while True:
                        try:
                            snapshot_path = _run_check_script(
                                args.check_script,
                                temp_input,
                                verbose=args.show_segmentation_output,
                                model=current_model,
                            )
                            break
                        except RuntimeError as exc:
                            if attempted_model_switch or fallback_model is None:
                                BLACKLIST_DIR.mkdir(parents=True, exist_ok=True)
                                blacklist_file.write_text(
                                    f"lang={lang} idx={idx} hash={file_hash} reason={exc}",
                                    encoding="utf-8",
                                )
                                print(
                                    f"[{lang}] Segmentation failed after fallback; blacklisting sample ({blacklist_file})."
                                )
                                segmentation_failed = True
                                break
                            print(
                                f"[{lang}] Segmentation failed with model=default flash ({exc}); "
                                f"retrying with gemini-2.5-pro in --fuzzy mode."
                            )
                            current_model = fallback_model
                            attempted_model_switch = True
                    if segmentation_failed:
                        continue
                snapshot_payload = _load_snapshot_payload(snapshot_path)
            purity_meta = _build_purity_metadata(
                purity_payload,
                used_purity_snapshot=used_purity,
                skip_purity_check=args.skip_purity_check,
            )
            # Persist per-sample cost into metadata for downstream aggregation.
            if "metadata" not in snapshot_payload:
                snapshot_payload["metadata"] = {}
            usage_meta = snapshot_payload["metadata"].get("usage_metadata") or {}
            snapshot_payload["metadata"]["estimated_cost"] = _estimate_cost(
                snapshot_payload["metadata"].get("model"),
                usage_meta.get("prompt_token_count"),
                usage_meta.get("candidates_token_count"),
                usage_meta.get("total_token_count"),
            )
            if purity_meta:
                snapshot_payload.setdefault("metadata", {})["purity_check"] = purity_meta
            dest_path.write_text(
                json.dumps(snapshot_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if snapshot_path.parent == SNAPSHOT_ROOT and snapshot_path.exists():
                snapshot_path.unlink(missing_ok=True)
            sample_cost = _print_sample_summary(
                lang=lang,
                sample_idx=idx,
                dest_path=dest_path,
                snapshot_data=snapshot_payload,
                stats=stats,
            )
            _append_meta_stats(stats, args, status=f"in_progress:{lang}:{idx}")
            processed += 1
            processed_any = True
            if sample_cost is not None:
                lang_cost_total += sample_cost
                lang_cost_samples += 1
            avg_cost_per_sample: Optional[float] = (
                lang_cost_total / lang_cost_samples if lang_cost_samples else None
            )
            if is_new_output:
                new_outputs_created += 1
                if remaining_new_slots is not None:
                    remaining_new_slots = max(remaining_new_slots - 1, 0)
            dataset_entries_remaining = max(total - (idx + 1), 0)
            if args.overwrite:
                estimated_remaining = dataset_entries_remaining
                quota_limited = (
                    remaining_new_slots is not None and remaining_new_slots < dataset_entries_remaining
                )
            elif remaining_new_slots is None:
                estimated_remaining = dataset_entries_remaining
                quota_limited = False
            else:
                estimated_remaining = min(dataset_entries_remaining, remaining_new_slots)
                quota_limited = remaining_new_slots < dataset_entries_remaining
            _print_remaining_forecast(
                lang=lang,
                estimated_remaining=estimated_remaining,
                dataset_remaining=dataset_entries_remaining,
                avg_cost_per_sample=avg_cost_per_sample,
                quota_limited=quota_limited,
                overwrite_mode=args.overwrite,
            )
            forecast_emitted = True
            if purity_snapshot_for_cleanup and purity_snapshot_for_cleanup.exists():
                purity_snapshot_for_cleanup.unlink(missing_ok=True)
        finally:
            temp_input.unlink(missing_ok=True)
        if args.demo:
            print(f"[{lang}] Demo mode: processed one sample and exiting.")
            return True
        if (
            not args.overwrite
            and remaining_new_slots is not None
            and remaining_new_slots <= 0
        ):
            if not quota_stop_logged:
                print(
                    f"{lang}: Reached --max-per-lang={args.max_per_lang} when accounting for existing outputs; "
                    "no additional new samples will be processed."
                )
                quota_stop_logged = True
            # No capacity left for new samples and overwriting is disabled, so bail out early.
            break
    print(
        f"{lang}: Completed. {processed} processed, {skipped_existing} skipped. "
        f"New outputs this run: {new_outputs_created}. "
        f"Overall: {global_progress.fmt_overall(extra_new=new_outputs_created)}."
    )
    global_progress.add_new(new_outputs_created)
    if not forecast_emitted:
        avg_cost_per_sample = lang_cost_total / lang_cost_samples if lang_cost_samples else None
        _print_remaining_forecast(
            lang=lang,
            estimated_remaining=0,
            dataset_remaining=0,
            avg_cost_per_sample=avg_cost_per_sample,
            quota_limited=False,
            overwrite_mode=args.overwrite,
        )
    return processed_any


def main() -> None:
    args = parse_args()
    requested = {lang.lower() for lang in args.langs} if args.langs else None
    unknown = sorted(requested - ALLOWED_LANGS_LOWER) if requested else []
    if unknown:
        print(f"Ignoring unsupported languages: {', '.join(unknown)}")
        requested -= set(unknown)
    lang_dirs = _ensure_lang_dirs(args.dataset_root, ALLOWED_LANGS_LOWER, requested)
    split_label = args.dataset
    print(
        f"Found {len(lang_dirs)} {split_label} dataset(s) to process under {args.dataset_root}: "
        f"{', '.join(sorted(lang_dirs))}."
    )
    total_target = args.max_per_lang * len(lang_dirs) if args.max_per_lang else None
    global_progress = GlobalProgress(total_target)
    stats = RunStats()
    for lang, dataset_dir in sorted(lang_dirs.items()):
        processed = process_language(lang, dataset_dir, args, stats, global_progress)
        if args.demo and processed:
            print("Demo mode satisfied; stopping after first processed sample.")
            break
    if stats.samples:
        print(
            (
                "Run totals: samples={samples}, prompt_tokens={prompt:,}, response_tokens={response:,}, "
                "total_tokens={total:,}, estimated_cost=${cost:.4f}"
            ).format(
                samples=stats.samples,
                prompt=stats.prompt_tokens,
                response=stats.response_tokens,
                total=stats.total_tokens,
                cost=stats.estimated_cost,
            )
        )
    else:
        print("No new samples were processed. Costs and token totals remain unchanged.")
    if stats.purity_checks:
        print(
            (
                "Purity checks: total={checks}, prompt_tokens={prompt:,}, response_tokens={response:,}, "
                "total_tokens={total:,}, estimated_cost=${cost:.4f}"
            ).format(
                checks=stats.purity_checks,
                prompt=stats.purity_prompt_tokens,
                response=stats.purity_response_tokens,
                total=stats.purity_total_tokens,
                cost=stats.purity_estimated_cost,
            )
        )
    _append_meta_stats(stats, args, status="completed")
    print(f"Recorded run statistics to {SNAPSHOT_ROOT / 'meta_stats.json'}.")


if __name__ == "__main__":
    main()
