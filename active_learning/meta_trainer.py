from __future__ import annotations

import argparse
import secrets
import shlex
import sqlite3
import string
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train -> active-learning -> train loop using the default training script."
    )
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--ckpt-path", type=str, default="train/checkpoints/al_loop.msgpack")
    parser.add_argument("--data-root", type=str, default="downloader/arrow_out")
    parser.add_argument("--train-max-minutes", type=int, default=10)
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
        help="Optional W&B run id for shared mode; if omitted, one is generated.",
    )

    parser.add_argument("--al-store", type=str, default="active_learning/label_store.sqlite")
    parser.add_argument("--al-split", type=str, default="train")
    parser.add_argument("--al-langs", type=str, default=None)
    parser.add_argument("--al-oracle", choices=("stub", "gemini"), default="gemini")
    parser.add_argument("--al-max-samples-per-lang", type=int, default=16)
    parser.add_argument("--al-max-candidates-per-sample", type=int, default=3)
    parser.add_argument("--al-min-score", type=float, default=0.5)
    parser.add_argument("--al-context-chars", type=int, default=250)
    parser.add_argument("--al-gemini-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--al-gemini-batch-size", type=int, default=2)
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
    shared_wandb_run_id = ""
    if args.wandb_mode == "shared":
        shared_wandb_run_id = (args.wandb_run_id or "").strip() or _generate_wandb_run_id()
        print(
            f"Using shared W&B run id across all rounds: {shared_wandb_run_id}",
            flush=True,
        )
    elif (args.wandb_run_id or "").strip():
        print("Ignoring --wandb-run-id because --wandb-mode=per-round.", flush=True)

    for round_idx in range(1, max(1, int(args.rounds)) + 1):
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
            "--max-samples-per-lang",
            str(args.al_max_samples_per_lang),
            "--max-candidates-per-sample",
            str(args.al_max_candidates_per_sample),
            "--min-score",
            str(args.al_min_score),
            "--context-chars",
            str(min(250, max(8, int(args.al_context_chars)))),
            "--oracle",
            args.al_oracle,
            "--gemini-model",
            args.al_gemini_model,
            "--gemini-batch-size",
            str(args.al_gemini_batch_size),
            "--gemini-rate-limit-sleep-seconds",
            str(args.al_gemini_rate_limit_sleep_seconds),
            "--gemini-rate-limit-max-retries",
            str(args.al_gemini_rate_limit_max_retries),
            "--gemini-missing-snippet-retries",
            str(args.al_gemini_missing_snippet_retries),
        ]
        if args.al_unlimited_oracle:
            al_cmd.append("--unlimited-oracle")
        else:
            al_cmd.extend(["--max-oracle-requests", str(max(0, int(args.al_max_oracle_requests)))])
        if args.al_langs:
            al_cmd.extend(["--langs", args.al_langs])
        _run(al_cmd)

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

        print(f"\n=== META ROUND {round_idx}/{args.rounds}: train (monitor + AL replay) ===", flush=True)
        train_cmd = [
            args.python,
            "train/main.py",
            "--data_root",
            args.data_root,
            "--ckpt_path",
            ckpt_path,
            "--steps",
            str(args.train_steps),
            "--max_minutes",
            str(args.train_max_minutes),
            "--fine-tune",
            "--fine_tune_ckpt_path",
            ckpt_path,
            "--active_learning_store",
            args.al_store,
            "--active_learning_mix_prob",
            str(scheduled_mix_prob),
            "--active_learning_max_windows",
            str(args.al_max_windows),
        ]
        if shared_wandb_run_id:
            train_cmd.extend(["--continue", shared_wandb_run_id])
        if args.train_extra_args.strip():
            train_cmd.extend(shlex.split(args.train_extra_args))
        _run(train_cmd)

    finished = f"\nFinished {args.rounds} rounds. Checkpoint: {ckpt_path} | label store: {args.al_store}"
    if shared_wandb_run_id:
        finished += f" | shared_wandb_run_id: {shared_wandb_run_id}"
    print(finished, flush=True)


if __name__ == "__main__":
    main()
