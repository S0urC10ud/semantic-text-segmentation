from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CAP_SELECTED_RE = re.compile(r"Oracle request cap active:.*selected=(\d+)/(\d+)\.")
DIAG_RE = re.compile(
    r"Oracle batch diagnostics: batches=(\d+), requested=(\d+), model_outputs=(\d+), failed_batches=(\d+),"
)


@dataclass(frozen=True)
class ProbeResult:
    batch_size: int
    return_code: int
    selected_snippets: Optional[int]
    candidate_snippets: Optional[int]
    requested_snippets: Optional[int]
    failed_batches: Optional[int]
    summary_status: str
    output: str


def _extract_selected(text: str) -> tuple[Optional[int], Optional[int]]:
    m = CAP_SELECTED_RE.search(text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _extract_diag(text: str) -> tuple[Optional[int], Optional[int]]:
    m = DIAG_RE.search(text)
    if not m:
        return None, None
    requested = int(m.group(2))
    failed_batches = int(m.group(4))
    return requested, failed_batches


def _extract_summary_status(text: str) -> str:
    m = re.search(r'"status"\s*:\s*"([^"]+)"', text)
    if not m:
        return "unknown"
    return str(m.group(1))


def _run_once(args: argparse.Namespace, batch_size: int) -> ProbeResult:
    store_path = Path(args.store).expanduser().resolve()
    if args.reset_store_each_run and store_path.exists():
        store_path.unlink()

    cmd = [
        args.python,
        "-m",
        "active_learning.round",
        "--ckpt",
        args.ckpt,
        "--data-root",
        args.data_root,
        "--split",
        args.split,
        "--store",
        str(store_path),
        "--oracle",
        "gemini",
        "--gemini-model",
        args.model,
        "--gemini-batch-size",
        str(int(batch_size)),
        "--gemini-rate-limit-sleep-seconds",
        str(float(args.rate_limit_sleep_seconds)),
        "--gemini-rate-limit-max-retries",
        str(int(args.rate_limit_max_retries)),
        "--gemini-missing-snippet-retries",
        str(int(args.missing_snippet_retries)),
        "--max-samples-per-lang",
        str(int(args.max_samples_per_lang)),
        "--max-candidates-per-sample",
        str(int(args.max_candidates_per_sample)),
        "--context-chars",
        str(int(args.context_chars)),
        "--min-score",
        str(float(args.min_score)),
        "--max-oracle-requests",
        str(int(args.max_oracle_requests)),
        "--allow-repeat-hashes",
    ]
    if args.langs:
        cmd.extend(["--langs", args.langs])
    if args.extra_round_args.strip():
        cmd.extend(shlex.split(args.extra_round_args))

    print("\n$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=None if args.timeout_seconds <= 0 else int(args.timeout_seconds),
    )
    combined = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    selected, candidates = _extract_selected(combined)
    requested, failed_batches = _extract_diag(combined)
    status = _extract_summary_status(combined)
    return ProbeResult(
        batch_size=int(batch_size),
        return_code=int(proc.returncode),
        selected_snippets=selected,
        candidate_snippets=candidates,
        requested_snippets=requested,
        failed_batches=failed_batches,
        summary_status=status,
        output=combined,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe maximum usable --gemini-batch-size by repeatedly doubling and running "
            "active_learning.round until failure."
        )
    )
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--data-root", type=str, default="downloader/arrow_out")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--store", type=str, default="/tmp/al_batch_probe.sqlite")
    parser.add_argument(
        "--langs",
        type=str,
        default="html,css,javascript_typescript,svg,xml,markdown",
        help="Comma-separated subset for cheap probing.",
    )
    parser.add_argument("--start-batch-size", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--max-oracle-requests", type=int, default=1)
    parser.add_argument("--max-samples-per-lang", type=int, default=2)
    parser.add_argument("--max-candidates-per-sample", type=int, default=3)
    parser.add_argument("--context-chars", type=int, default=250)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--rate-limit-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--rate-limit-max-retries", type=int, default=0)
    parser.add_argument("--missing-snippet-retries", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument(
        "--reset-store-each-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete probe store before each run to avoid cross-run carry-over.",
    )
    parser.add_argument(
        "--extra-round-args",
        type=str,
        default="",
        help="Extra raw args appended to each active_learning.round invocation.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    batch = max(1, int(args.start_batch_size))
    max_batch = max(batch, int(args.max_batch_size))
    last_ok: Optional[int] = None
    results: list[ProbeResult] = []

    while batch <= max_batch:
        print(
            f"\n=== Probe batch_size={batch} ===",
            flush=True,
        )
        result = _run_once(args, batch)
        results.append(result)
        selected = result.selected_snippets
        requested = result.requested_snippets
        print(
            "Result: "
            f"rc={result.return_code}, "
            f"status={result.summary_status}, "
            f"selected={selected if selected is not None else 'n/a'}, "
            f"requested={requested if requested is not None else 'n/a'}, "
            f"failed_batches={result.failed_batches if result.failed_batches is not None else 'n/a'}",
            flush=True,
        )

        if result.return_code != 0:
            print("Stopping: round.py exited with failure.", flush=True)
            break
        if selected is not None and selected < batch:
            print(
                "Stopping: selected snippet count is smaller than batch size, "
                "so this run could not fully stress the configured batch.",
                flush=True,
            )
            break
        if result.failed_batches is not None and result.failed_batches > 0:
            print("Stopping: oracle reported failed batches.", flush=True)
            break

        last_ok = batch
        batch *= 2

    print("\n=== Probe summary ===", flush=True)
    for row in results:
        print(
            f"batch={row.batch_size:<4} rc={row.return_code:<2} "
            f"status={row.summary_status:<12} selected={row.selected_snippets if row.selected_snippets is not None else 'n/a':<4} "
            f"failed_batches={row.failed_batches if row.failed_batches is not None else 'n/a'}",
            flush=True,
        )

    if last_ok is None:
        print("No successful batch size found.", flush=True)
    else:
        print(f"Last successful batch size: {last_ok}", flush=True)

    if results and results[-1].return_code != 0:
        tail = "\n".join(results[-1].output.strip().splitlines()[-40:])
        print("\nLast failure tail:\n" + tail, flush=True)


if __name__ == "__main__":
    main()

