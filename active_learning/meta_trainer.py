from __future__ import annotations

import argparse
import json
import os
import queue
import re
import secrets
import shlex
import shutil
import sqlite3
import string
import subprocess
import sys
import threading
import time
from pathlib import Path

from active_learning.persistent_control import (
    encode_trainer_command,
    maybe_parse_trainer_event_line,
)


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _run_capture(cmd: list[str]) -> str:
    """Run a command, stream output to the terminal, and return all captured stdout."""
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return "".join(lines)


class PersistentTrainerSession:
    def __init__(self, cmd: list[str]) -> None:
        self.cmd = list(cmd)
        self.proc: subprocess.Popen[str] | None = None
        self._events: "queue.Queue[dict]" = queue.Queue()
        self._reader_thread: threading.Thread | None = None

    def _reader_loop(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            print(line, end="", flush=True)
            try:
                event = maybe_parse_trainer_event_line(line)
            except Exception as exc:
                self._events.put({"event": "reader_parse_error", "error": str(exc)})
                continue
            if event is not None:
                self._events.put(event)
        returncode = self.proc.wait()
        self._events.put({"event": "process_exited", "returncode": int(returncode)})

    def start(self, *, ready_timeout_seconds: float = 600.0) -> dict:
        if self.proc is not None:
            raise RuntimeError("Persistent trainer session already started.")
        print("$ " + " ".join(shlex.quote(part) for part in self.cmd), flush=True)
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="persistent-trainer-stdout",
            daemon=True,
        )
        self._reader_thread.start()
        return self.wait_for_event(
            {"ready"},
            timeout_seconds=float(ready_timeout_seconds),
        )

    def wait_for_event(
        self,
        expected_events: set[str],
        *,
        timeout_seconds: float,
    ) -> dict:
        deadline = time.time() + max(1.0, float(timeout_seconds))
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for trainer events {sorted(expected_events)}."
                )
            try:
                event = self._events.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if self.proc is not None and self.proc.poll() is not None:
                    raise RuntimeError(
                        f"Persistent trainer exited unexpectedly with code {self.proc.returncode} "
                        f"while waiting for {sorted(expected_events)}."
                    )
                continue
            event_name = str(event.get("event", "")).strip()
            if event_name == "process_exited":
                raise RuntimeError(
                    f"Persistent trainer exited unexpectedly with code "
                    f"{int(event.get('returncode', -1))} while waiting for {sorted(expected_events)}."
                )
            if event_name == "reader_parse_error":
                raise RuntimeError(
                    f"Failed to parse trainer control event from stdout: {event.get('error', '')}"
                )
            if event_name in expected_events:
                return event

    def send_command(self, command: str, **payload) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Persistent trainer is not running.")
        self.proc.stdin.write(encode_trainer_command(command, **payload) + "\n")
        self.proc.stdin.flush()

    def train_until(
        self,
        *,
        target_step: int,
        max_minutes: int,
        phase_label: str,
        active_learning_mix_prob: float,
    ) -> dict:
        self.send_command(
            "train",
            steps=int(target_step),
            max_minutes=int(max_minutes),
            phase_label=str(phase_label),
            active_learning_mix_prob=float(active_learning_mix_prob),
        )
        return self.wait_for_event(
            {"chunk_done", "chunk_failed"},
            timeout_seconds=24.0 * 60.0 * 60.0,
        )

    def shutdown(self, *, reason: str = "meta_trainer_done") -> None:
        if self.proc is None:
            return
        try:
            self.send_command("shutdown", reason=str(reason))
        except Exception:
            pass
        try:
            self.wait_for_event({"shutdown_ack"}, timeout_seconds=15.0)
        except Exception:
            pass
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # type: ignore[union-attr]
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10.0)
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=5.0)
                except Exception:
                    pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)


def _parse_round_summary(stdout: str) -> dict[str, object]:
    """Extract the last JSON object from round.py stdout output."""
    # round.py prints the summary as the last JSON block via json.dumps().
    # Walk backwards through lines to find the last complete JSON block.
    lines = stdout.splitlines()
    json_candidates: list[str] = []
    brace_depth = 0
    collecting = False
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not collecting:
            if "}" in stripped:
                collecting = True
                brace_depth = 0
        if collecting:
            json_candidates.insert(0, lines[i])
            # Count closing braces as +1 (deeper), opening as -1 (shallower)
            # since we're walking backwards.
            brace_depth += stripped.count("}") - stripped.count("{")
            if brace_depth <= 0:
                break
    if not json_candidates:
        return {}
    candidate = "\n".join(json_candidates)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _generate_wandb_run_id(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(max(4, int(length))))


def _count_oracle_refinement_rows(store_path: str) -> int:
    db_path = Path(store_path).expanduser().resolve()
    if not db_path.exists():
        return 0
    try:
        with sqlite3.connect(str(db_path)) as con:
            row = con.execute("SELECT COUNT(*) FROM refinements WHERE status = 'ok'").fetchone()
        if not row:
            return 0
        return int(row[0] or 0)
    except sqlite3.Error:
        return 0


def _sqlite_size_mb(store_path: str) -> float:
    db_path = Path(store_path).expanduser().resolve()
    if not db_path.exists():
        return 0.0
    try:
        return float(db_path.stat().st_size) / (1024.0 * 1024.0)
    except OSError:
        return 0.0


def _linear_oracle_mix_prob(
    refinement_rows: int,
    *,
    max_prob: float,
    full_at_rows: int,
) -> float:
    max_p = float(max(0.0, min(1.0, max_prob)))
    full_at = max(1, int(full_at_rows))
    rows = max(0, int(refinement_rows))
    ratio = min(1.0, float(rows) / float(full_at))
    return max_p * ratio


def _wandb_safe_log(data: dict, step: int, commit: bool = True) -> None:
    try:
        import wandb

        wandb.log(data, step=step, commit=commit)
    except Exception as e:
        print(f"Wandb logging failed: {e}", flush=True)


def _resolve_ckpt_paths(path: str) -> tuple[Path, str, Path]:
    blob_abs = Path(path).expanduser().resolve()
    ckpt_dir_abs = blob_abs.parent
    prefix = blob_abs.name + "-"
    return ckpt_dir_abs, prefix, blob_abs


def _latest_train_checkpoint_step(path: str) -> int:
    ckpt_dir, prefix, _ = _resolve_ckpt_paths(path)
    if not ckpt_dir.exists():
        return 0
    latest = 0
    for entry in ckpt_dir.iterdir():
        name = entry.name
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if suffix.isdigit():
            latest = max(latest, int(suffix))
    return latest


def _ensure_bootstrap_checkpoint(target_path: str, init_path: str | None) -> None:
    target = Path(target_path).expanduser().resolve()
    if target.exists():
        return
    if not init_path:
        raise FileNotFoundError(
            f"Checkpoint not found at {target}. Pass --init-ckpt-path to bootstrap a fresh run."
        )
    source = Path(init_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Bootstrap checkpoint not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"Bootstrapped checkpoint: {source} -> {target}", flush=True)


def _compute_train_target_step(current_step: int, additional_updates: int) -> int:
    updates = max(0, int(additional_updates))
    if updates == 0:
        return max(0, int(current_step))
    # train/main.py loops inclusively over `range(current_step, steps + 1)`,
    # so subtract one here to request exactly `additional_updates` optimizer steps.
    return max(0, int(current_step) + updates - 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train -> active-learning -> train loop using the default training script."
    )
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--ckpt-path", type=str, default="train/checkpoints/al_loop.msgpack")
    parser.add_argument(
        "--init-ckpt-path",
        type=str,
        default="",
        help=(
            "Optional source checkpoint used to bootstrap --ckpt-path when starting a fresh run "
            "with a new checkpoint path."
        ),
    )
    parser.add_argument("--arch", type=str, default=None, choices=("unet1d", "mamba"),
                        help="Model architecture (auto-detected from checkpoint if omitted).")
    parser.add_argument("--data-root", type=str, default="downloader/arrow_out")
    parser.add_argument("--train-max-minutes", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--train-extra-args", type=str, default="")
    parser.add_argument(
        "--wandb-mode",
        choices=("shared", "per-round"),
        default="shared",
        help="Use one shared W&B run across all rounds (default) or create one run per round.",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default="",
        help="Optional shared W&B run id for child training processes; if omitted, one is generated.",
    )
    parser.add_argument(
        "--wandb-parent-run-id",
        type=str,
        default="",
        help="Optional W&B run id for the parent meta-trainer process; if omitted, one is generated.",
    )
    parser.add_argument(
        "--persistent-trainer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse one long-lived train/main.py subprocess across rounds in shared W&B mode "
            "to avoid repeated JAX compile startup costs."
        ),
    )

    parser.add_argument("--al-store", type=str, default="active_learning/label_store.sqlite")
    parser.add_argument("--full-files", action="store_true", help="Use full-file/long-sample AL training and querying paths.")
    parser.add_argument("--full-file-max-bytes", type=int, default=10000)
    parser.add_argument("--al-split", type=str, default="train")
    parser.add_argument("--al-langs", type=str, default=None)
    parser.add_argument("--al-oracle", choices=("stub", "gemini"), default="gemini")
    parser.add_argument("--al-max-samples-per-lang", type=int, default=16)
    parser.add_argument(
        "--al-sample-workers",
        type=int,
        default=1,
        help="CPU worker processes for assembling AL long samples before scoring.",
    )
    parser.add_argument(
        "--al-sample-prefetch",
        type=int,
        default=0,
        help="Prefetch queue size for assembled AL long samples.",
    )
    parser.add_argument("--al-max-candidates-per-sample", type=int, default=3)
    parser.add_argument("--al-min-score", type=float, default=0.5)
    parser.add_argument("--al-context-chars", type=int, default=250)
    parser.add_argument("--al-predict-batch-size", type=int, default=12)
    parser.add_argument("--al-gemini-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument(
        "--al-gemini-thinking-level",
        type=str,
        choices=("minimal", "low", "medium", "high"),
        default="medium",
    )
    parser.add_argument("--al-gemini-batch-size", type=int, default=32)
    parser.add_argument("--al-gemini-rate-limit-sleep-seconds", type=float, default=65.0)
    parser.add_argument("--al-gemini-rate-limit-max-retries", type=int, default=8)
    parser.add_argument("--al-gemini-missing-snippet-retries", type=int, default=2)
    parser.add_argument(
        "--al-max-oracle-requests",
        type=int,
        default=3,
        help="Maximum oracle requests per AL round (default: 3).",
    )
    parser.add_argument(
        "--al-unlimited-oracle",
        action="store_true",
        help="Disable oracle request cap and query all candidate snippets (previous behavior).",
    )

    parser.add_argument(
        "--al-mix-prob",
        type=float,
        default=0.5,
        help="Maximum AL replay mix probability at saturation (default: 0.5).",
    )
    parser.add_argument(
        "--al-mix-full-at-rows",
        type=int,
        default=1000,
        help=(
            "Number of stored oracle refinements at which --al-mix-prob is reached. "
            "Mix grows linearly from 0 to max."
        ),
    )
    parser.add_argument("--al-max-windows", type=int, default=1000000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ckpt_path = str(Path(args.ckpt_path))
    init_ckpt_path = (args.init_ckpt_path or "").strip() or None
    _ensure_bootstrap_checkpoint(ckpt_path, init_ckpt_path)
    shared_wandb_run_id = ""
    requested_parent_wandb_run_id = (args.wandb_parent_run_id or "").strip()
    parent_wandb_run_id = requested_parent_wandb_run_id or _generate_wandb_run_id()
    initial_train_step = _latest_train_checkpoint_step(ckpt_path)
    shared_schedule_final_step = _compute_train_target_step(
        initial_train_step,
        int(args.rounds) * int(args.train_steps),
    )
    if args.wandb_mode == "shared":
        shared_wandb_run_id = (args.wandb_run_id or "").strip() or _generate_wandb_run_id()
        print(
            f"Using shared child W&B run id across all rounds: {shared_wandb_run_id}",
            flush=True,
        )
    elif (args.wandb_run_id or "").strip():
        print("Ignoring --wandb-run-id because --wandb-mode=per-round.", flush=True)
    print(
        f"Using parent W&B run id for meta-trainer logging: {parent_wandb_run_id}",
        flush=True,
    )

    # ── Initialize wandb for meta-trainer logging ──
    wandb_available = False
    try:
        import wandb
        from wandb import Settings

        wandb_project = os.getenv("WANDB_PROJECT", "code-segmentation-v2")
        wandb_kwargs: dict = {
            "project": wandb_project,
            "settings": Settings(init_timeout=300, start_method="thread"),
            "tags": ["active-learning", "meta-trainer"],
            "id": parent_wandb_run_id,
            "resume": "never",
        }
        wandb.init(**wandb_kwargs)
        wandb.config.update(
            {
                "parent_wandb_run_id": str(parent_wandb_run_id),
                "child_wandb_run_id": str(shared_wandb_run_id) if shared_wandb_run_id else None,
                "al_rounds": int(args.rounds),
                "al_store": str(args.al_store),
                "full_files": bool(args.full_files),
                "full_file_max_bytes": int(args.full_file_max_bytes),
                "al_split": str(args.al_split),
                "al_oracle": str(args.al_oracle),
                "al_gemini_model": str(args.al_gemini_model),
                "al_gemini_thinking_level": str(args.al_gemini_thinking_level),
                "al_gemini_batch_size": int(args.al_gemini_batch_size),
                "al_max_samples_per_lang": int(args.al_max_samples_per_lang),
                "al_sample_workers": int(args.al_sample_workers),
                "al_sample_prefetch": int(args.al_sample_prefetch),
                "al_max_candidates_per_sample": int(args.al_max_candidates_per_sample),
                "al_predict_batch_size": int(args.al_predict_batch_size),
                "al_mix_prob_max": float(args.al_mix_prob),
                "al_mix_full_at_rows": int(args.al_mix_full_at_rows),
                "al_max_oracle_requests": None if args.al_unlimited_oracle else int(args.al_max_oracle_requests),
                "train_steps_per_round": int(args.train_steps),
                "train_schedule_steps": int(shared_schedule_final_step) if shared_wandb_run_id else None,
                "persistent_trainer": bool(args.persistent_trainer),
                "ckpt_path": str(ckpt_path),
            },
            allow_val_change=True,
        )
        wandb_available = True
        print(f"Wandb initialized for meta-trainer logging (project={wandb_project}).", flush=True)
    except Exception as e:
        if requested_parent_wandb_run_id:
            raise RuntimeError(
                f"Failed to initialize parent W&B run {requested_parent_wandb_run_id}: {e}"
            ) from e
        print(f"⚠️  Wandb init failed; metrics will only be printed: {e}", flush=True)

    # ── Cumulative counters across all rounds ──
    cumulative = {
        "llm_total_requests": 0,
        "llm_retry_requests": 0,
        "llm_rate_limit_retries": 0,
        "llm_prompt_tokens": 0,
        "llm_candidates_tokens": 0,
        "llm_total_tokens": 0,
        "llm_failed_batches": 0,
        "llm_final_parse_failed": 0,
        "llm_final_missing": 0,
        "oracle_skipped_snippets": 0,
        "refinements_stored": 0,
        "oracle_queries_sent": 0,
    }
    persistent_trainer_enabled = bool(args.persistent_trainer) and args.wandb_mode == "shared"
    if bool(args.persistent_trainer) and args.wandb_mode != "shared":
        print(
            "Persistent trainer requested, but --wandb-mode=per-round is active; "
            "falling back to one trainer subprocess per round.",
            flush=True,
        )
    trainer_session: PersistentTrainerSession | None = None

    def _build_train_cmd(*, current_train_step: int) -> list[str]:
        should_continue_shared_run = bool(shared_wandb_run_id) and current_train_step > 0
        train_cmd = [
            args.python,
            "train/main.py",
            "--data_root",
            args.data_root,
            "--ckpt_path",
            ckpt_path,
            "--steps",
            str(max(0, int(current_train_step) - 1)),
            "--max_minutes",
            str(args.train_max_minutes),
            "--fine-tune",
            "--fine_tune_ckpt_path",
            ckpt_path,
            "--active_learning_store",
            args.al_store,
            "--active_learning_mix_prob",
            "0.0",
            "--active_learning_max_windows",
            str(args.al_max_windows),
            "--full-files" if bool(args.full_files) else "",
            "--full-file-max-bytes",
            str(int(args.full_file_max_bytes)),
            "--persistent-trainer" if persistent_trainer_enabled else "",
        ]
        train_cmd = [part for part in train_cmd if part != ""]
        if shared_wandb_run_id:
            train_cmd.extend(["--schedule_steps", str(shared_schedule_final_step)])
        if should_continue_shared_run:
            train_cmd.extend(["--continue", shared_wandb_run_id])
        elif shared_wandb_run_id:
            train_cmd.extend(["--wandb-run-id", shared_wandb_run_id])
            print(
                f"Starting fresh shared W&B run {shared_wandb_run_id} from params at {ckpt_path}.",
                flush=True,
            )
        if args.train_extra_args.strip():
            train_cmd.extend(shlex.split(args.train_extra_args))
        return train_cmd

    try:
        for round_idx in range(1, max(1, int(args.rounds)) + 1):
            current_train_step = _latest_train_checkpoint_step(ckpt_path)
            print(f"\n=== META ROUND {round_idx}/{args.rounds}: acquire + oracle ===", flush=True)
            al_cmd = [
                args.python,
                "-m",
                "active_learning.round",
                "--ckpt",
                ckpt_path,
                "--data-root",
                args.data_root,
                "--split",
                args.al_split,
                "--store",
                args.al_store,
                "--full-files" if bool(args.full_files) else "",
                "--full-file-max-bytes",
                str(int(args.full_file_max_bytes)),
                "--max-samples-per-lang",
                str(args.al_max_samples_per_lang),
                "--sample-workers",
                str(max(1, int(args.al_sample_workers))),
                "--sample-prefetch",
                str(max(0, int(args.al_sample_prefetch))),
                "--max-candidates-per-sample",
                str(args.al_max_candidates_per_sample),
                "--min-score",
                str(args.al_min_score),
                "--context-chars",
                str(min(250, max(8, int(args.al_context_chars)))),
                "--predict-batch-size",
                str(max(1, int(args.al_predict_batch_size))),
                "--oracle",
                args.al_oracle,
                "--gemini-model",
                args.al_gemini_model,
                "--gemini-thinking-level",
                args.al_gemini_thinking_level,
                "--gemini-batch-size",
                str(args.al_gemini_batch_size),
                "--gemini-rate-limit-sleep-seconds",
                str(args.al_gemini_rate_limit_sleep_seconds),
                "--gemini-rate-limit-max-retries",
                str(args.al_gemini_rate_limit_max_retries),
                "--gemini-missing-snippet-retries",
                str(args.al_gemini_missing_snippet_retries),
            ]
            al_cmd = [part for part in al_cmd if part != ""]
            if args.arch:
                al_cmd.extend(["--arch", args.arch])
            if args.al_unlimited_oracle:
                al_cmd.append("--unlimited-oracle")
            else:
                al_cmd.extend(["--max-oracle-requests", str(max(0, int(args.al_max_oracle_requests)))])
            if args.al_langs:
                al_cmd.extend(["--langs", args.al_langs])

            al_stdout = _run_capture(al_cmd)
            round_summary = _parse_round_summary(al_stdout)

            refinement_rows = _count_oracle_refinement_rows(args.al_store)
            scheduled_mix_prob = _linear_oracle_mix_prob(
                refinement_rows,
                max_prob=float(args.al_mix_prob),
                full_at_rows=int(args.al_mix_full_at_rows),
            )
            print(
                "AL replay schedule: "
                f"refinement_rows={refinement_rows}, "
                f"mix_prob={scheduled_mix_prob:.4f} "
                f"(max={float(args.al_mix_prob):.3f} @ rows={int(args.al_mix_full_at_rows)}).",
                flush=True,
            )

            cumulative["llm_total_requests"] += int(round_summary.get("llm_total_requests", 0))
            cumulative["llm_retry_requests"] += int(round_summary.get("llm_retry_requests", 0))
            cumulative["llm_rate_limit_retries"] += int(round_summary.get("llm_rate_limit_retries", 0))
            cumulative["llm_prompt_tokens"] += int(round_summary.get("llm_prompt_tokens", 0))
            cumulative["llm_candidates_tokens"] += int(round_summary.get("llm_candidates_tokens", 0))
            cumulative["llm_total_tokens"] += int(round_summary.get("llm_total_tokens", 0))
            cumulative["llm_failed_batches"] += int(round_summary.get("llm_failed_batches", 0))
            cumulative["llm_final_parse_failed"] += int(round_summary.get("llm_final_parse_failed", 0))
            cumulative["llm_final_missing"] += int(round_summary.get("llm_final_missing", 0))
            cumulative["oracle_skipped_snippets"] += int(round_summary.get("oracle_skipped_snippets", 0))
            cumulative["refinements_stored"] += int(round_summary.get("stored", 0))
            cumulative["oracle_queries_sent"] += int(round_summary.get("samples", 0))

            sqlite_mb = _sqlite_size_mb(args.al_store)
            al_metrics = {
                "al/round": int(round_idx),
                "al/round_status": str(round_summary.get("status", "unknown")),
                "al/candidate_snippets": int(round_summary.get("candidate_snippets", 0)),
                "al/oracle_queries_sent": int(round_summary.get("samples", 0)),
                "al/oracle_model_outputs": int(round_summary.get("oracle_model_outputs", 0)),
                "al/oracle_fallback_snippets": int(round_summary.get("oracle_fallback_snippets", 0)),
                "al/oracle_skipped_snippets": int(round_summary.get("oracle_skipped_snippets", 0)),
                "al/refinements_stored": int(round_summary.get("stored", 0)),
                "al/inference_samples_scanned": int(round_summary.get("inference_samples", 0)),
                "al/llm_requests": int(round_summary.get("llm_requests", 0)),
                "al/llm_retry_requests": int(round_summary.get("llm_retry_requests", 0)),
                "al/llm_total_requests": int(round_summary.get("llm_total_requests", 0)),
                "al/llm_rate_limit_retries": int(round_summary.get("llm_rate_limit_retries", 0)),
                "al/llm_failed_batches": int(round_summary.get("llm_failed_batches", 0)),
                "al/llm_final_parse_failed": int(round_summary.get("llm_final_parse_failed", 0)),
                "al/llm_final_missing": int(round_summary.get("llm_final_missing", 0)),
                "al/llm_prompt_tokens": int(round_summary.get("llm_prompt_tokens", 0)),
                "al/llm_candidates_tokens": int(round_summary.get("llm_candidates_tokens", 0)),
                "al/llm_total_tokens": int(round_summary.get("llm_total_tokens", 0)),
                "al/cumulative_llm_total_requests": int(cumulative["llm_total_requests"]),
                "al/cumulative_llm_retry_requests": int(cumulative["llm_retry_requests"]),
                "al/cumulative_llm_rate_limit_retries": int(cumulative["llm_rate_limit_retries"]),
                "al/cumulative_llm_prompt_tokens": int(cumulative["llm_prompt_tokens"]),
                "al/cumulative_llm_candidates_tokens": int(cumulative["llm_candidates_tokens"]),
                "al/cumulative_llm_total_tokens": int(cumulative["llm_total_tokens"]),
                "al/cumulative_llm_failed_batches": int(cumulative["llm_failed_batches"]),
                "al/cumulative_llm_final_parse_failed": int(cumulative["llm_final_parse_failed"]),
                "al/cumulative_llm_final_missing": int(cumulative["llm_final_missing"]),
                "al/cumulative_oracle_skipped_snippets": int(cumulative["oracle_skipped_snippets"]),
                "al/cumulative_refinements_stored": int(cumulative["refinements_stored"]),
                "al/cumulative_oracle_queries_sent": int(cumulative["oracle_queries_sent"]),
                "al/total_refinement_rows": int(refinement_rows),
                "al/sqlite_size_mb": float(round(sqlite_mb, 3)),
                "al/replay_mix_prob": float(scheduled_mix_prob),
            }
            if wandb_available:
                _wandb_safe_log(al_metrics, step=int(round_idx), commit=True)
            print(
                f"AL round {round_idx} metrics: "
                f"queries={al_metrics['al/oracle_queries_sent']}, "
                f"stored={al_metrics['al/refinements_stored']}, "
                f"llm_requests={al_metrics['al/llm_total_requests']}, "
                f"llm_tokens={al_metrics['al/llm_total_tokens']}, "
                f"retries={al_metrics['al/llm_retry_requests']}, "
                f"rate_limit_retries={al_metrics['al/llm_rate_limit_retries']}, "
                f"failed_batches={al_metrics['al/llm_failed_batches']}, "
                f"parse_failed_skipped={al_metrics['al/llm_final_parse_failed']}, "
                f"missing_skipped={al_metrics['al/llm_final_missing']}, "
                f"oracle_skipped={al_metrics['al/oracle_skipped_snippets']}, "
                f"sqlite={sqlite_mb:.2f}MB, "
                f"cum_requests={cumulative['llm_total_requests']}, "
                f"cum_tokens={cumulative['llm_total_tokens']}",
                flush=True,
            )

            print(f"\n=== META ROUND {round_idx}/{args.rounds}: train (monitor + AL replay) ===", flush=True)
            train_target_step = _compute_train_target_step(current_train_step, int(args.train_steps))
            if persistent_trainer_enabled:
                if trainer_session is None:
                    trainer_session = PersistentTrainerSession(
                        _build_train_cmd(current_train_step=current_train_step)
                    )
                    ready_event = trainer_session.start()
                    print(
                        "Persistent trainer ready: "
                        f"current_step={int(ready_event.get('current_step', current_train_step))}, "
                        f"wandb_run_id={ready_event.get('wandb_run_id', '')}",
                        flush=True,
                    )
                train_event = trainer_session.train_until(
                    target_step=int(train_target_step),
                    max_minutes=int(args.train_max_minutes),
                    phase_label=f"meta_round_{round_idx}",
                    active_learning_mix_prob=float(scheduled_mix_prob),
                )
                print(
                    "Persistent trainer chunk result: "
                    f"event={train_event.get('event')} "
                    f"current_step={int(train_event.get('current_step', train_target_step))} "
                    f"target_step={int(train_event.get('target_step', train_target_step))} "
                    f"reason={train_event.get('final_reason', '')}",
                    flush=True,
                )
                if str(train_event.get("event", "")) == "chunk_failed":
                    raise RuntimeError(
                        "Persistent trainer chunk failed: "
                        f"{train_event.get('error_message', '')}"
                    )
            else:
                train_cmd = _build_train_cmd(current_train_step=current_train_step)
                train_cmd[train_cmd.index("--active_learning_mix_prob") + 1] = str(scheduled_mix_prob)
                train_cmd[train_cmd.index("--steps") + 1] = str(train_target_step)
                _run(train_cmd)
    finally:
        if trainer_session is not None:
            trainer_session.shutdown(reason="meta_trainer_done")

    finished = f"\nFinished {args.rounds} rounds. Checkpoint: {ckpt_path} | label store: {args.al_store}"
    if shared_wandb_run_id:
        finished += f" | child_shared_wandb_run_id: {shared_wandb_run_id}"
    finished += f" | parent_wandb_run_id: {parent_wandb_run_id}"
    print(finished, flush=True)
    print(
        "Cumulative stats: "
        f"llm_requests={cumulative['llm_total_requests']}, "
        f"llm_tokens={cumulative['llm_total_tokens']}, "
        f"retries={cumulative['llm_retry_requests']}, "
        f"rate_limit_retries={cumulative['llm_rate_limit_retries']}, "
        f"failed_batches={cumulative['llm_failed_batches']}, "
        f"parse_failed_skipped={cumulative['llm_final_parse_failed']}, "
        f"missing_skipped={cumulative['llm_final_missing']}, "
        f"oracle_skipped={cumulative['oracle_skipped_snippets']}, "
        f"refinements={cumulative['refinements_stored']}, "
        f"queries={cumulative['oracle_queries_sent']}",
        flush=True,
    )

    if wandb_available:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
