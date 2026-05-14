#!/usr/bin/env python3
"""Benchmark batched segmentation throughput for byte-level checkpoints.

This script focuses on inference throughput rather than accuracy. It exercises
the public batched byte-array API on SegmenterRunner so long samples can use
the fast sliding-window path where independent windows are grouped into model
microbatches and merged post-hoc with overlap weighting.

For the byte-level models in this repo, "tokens" are bytes, so tokens/s and
bytes/s are identical here. The generated synthetic samples are ASCII-only, so
chars/s also matches bytes/s.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import string
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import available_backends, resolve_backend  # noqa: E402

_EVAL_MODULE_PATH = REPO_ROOT / "evaluation" / "evaluation.py"
_EVAL_SPEC = importlib.util.spec_from_file_location("eval_harness", _EVAL_MODULE_PATH)
if _EVAL_SPEC is None or _EVAL_SPEC.loader is None:
    raise ImportError(f"Could not load evaluation module from {_EVAL_MODULE_PATH}")
_EVAL_MODULE = importlib.util.module_from_spec(_EVAL_SPEC)
sys.modules[_EVAL_SPEC.name] = _EVAL_MODULE
_EVAL_SPEC.loader.exec_module(_EVAL_MODULE)

DEFAULT_CHUNK_SIZE = _EVAL_MODULE.DEFAULT_CHUNK_SIZE
SegmenterRunner = _EVAL_MODULE.SegmenterRunner


ASCII_ALPHABET = np.frombuffer(
    (string.ascii_letters + string.digits + string.punctuation + "\n ").encode("ascii"),
    dtype=np.uint8,
)
DEFAULT_AUTO_BATCH_LIMIT = 64


@dataclass
class BenchmarkResult:
    device: str
    arch: str
    text_bytes: int
    batch_size: int
    samples: int
    repeats: int
    total_bytes: int
    total_windows: int
    elapsed_seconds: float
    bytes_per_second: float
    chars_per_second: float
    tokens_per_second: float
    windows_per_second: float
    model_calls: int
    mean_model_call_batch: float
    max_model_call_batch: int
    mean_padded_length: float
    path_counts: Dict[str, int]


def _parse_csv_ints(raw: str) -> List[int]:
    values: List[int] = []
    for part in str(raw).split(","):
        value = part.strip()
        if not value:
            continue
        values.append(int(value))
    if not values:
        raise ValueError(f"Expected at least one integer in '{raw}'.")
    return values


def _parse_csv_strings(raw: str) -> List[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one value in '{raw}'.")
    return values


def _auto_batch_sizes(limit: int) -> List[int]:
    cap = max(1, int(limit))
    values: List[int] = []
    current = 1
    while current <= cap:
        values.append(current)
        current *= 2
    if values[-1] != cap:
        values.append(cap)
    return values


def _resolve_batch_sizes(raw: str, *, auto_limit: int) -> tuple[List[int], bool]:
    raw_clean = str(raw).strip().lower()
    if raw_clean == "auto":
        return _auto_batch_sizes(auto_limit), True
    return _parse_csv_ints(raw), False


def _iter_batches(items: Sequence[np.ndarray], batch_size: int) -> Iterable[Sequence[np.ndarray]]:
    step = max(1, int(batch_size))
    for start in range(0, len(items), step):
        yield items[start:start + step]


def _generate_ascii_samples(*, text_bytes: int, count: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_matrix = rng.choice(ASCII_ALPHABET, size=(count, text_bytes), replace=True)
    return [np.asarray(row, dtype=np.uint8) for row in sample_matrix]


def _device_is_available(device_name: str) -> bool:
    backend = resolve_backend(device_name)
    if backend is None:
        return True
    return backend in available_backends()


def _build_runner(args: argparse.Namespace, *, device_name: str, batch_size: int) -> SegmenterRunner:
    channels = tuple(_parse_csv_ints(args.channels)) if args.channels else ()
    budget_bytes = None
    if int(args.full_memory_budget_mb) > 0:
        budget_bytes = int(args.full_memory_budget_mb) * 1024 * 1024
    return SegmenterRunner(
        args.checkpoint,
        arch=args.arch,
        model_dim=int(args.model_dim),
        channels=channels,
        mamba_layers=int(args.mamba_layers),
        mamba_d_state=int(args.mamba_d_state),
        mamba_expand=int(args.mamba_expand),
        mamba_dt_rank=int(args.mamba_dt_rank),
        mamba_conv=int(args.mamba_conv),
        mamba_bidirectional=bool(args.mamba_bidirectional),
        dtype=args.dtype,
        chunk=int(args.chunk),
        device=device_name,
        batch_size=batch_size,
        inference_backend=args.inference_backend,
        full_memory_budget_bytes=budget_bytes,
    )


def _benchmark_one(
    *,
    runner: SegmenterRunner,
    byte_arrays: Sequence[np.ndarray],
    batch_size: int,
    repeats: int,
    warmup_batches: int,
    device_name: str,
) -> BenchmarkResult:
    if not byte_arrays:
        raise ValueError("Benchmark requires at least one sample.")

    warmup_iter = list(_iter_batches(byte_arrays, batch_size))
    for batch in warmup_iter[: max(1, int(warmup_batches))]:
        runner.segment_byte_arrays_batch_labels_only(batch)
    runner.clear_fast_execution_history()

    total_bytes = 0
    total_windows = 0

    start = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        for batch in _iter_batches(byte_arrays, batch_size):
            _, spans_by_text = runner.segment_byte_arrays_batch_labels_only(batch)
            total_bytes += int(sum(int(arr.shape[0]) for arr in batch))
            total_windows += int(sum(len(spans) for spans in spans_by_text))
    elapsed = time.perf_counter() - start

    history = runner.get_fast_execution_history()
    path_counts: Dict[str, int] = {}
    actual_batches: List[int] = []
    padded_lengths: List[int] = []
    for record in history:
        path_counts[record.path] = path_counts.get(record.path, 0) + 1
        actual_batches.append(int(record.actual_batch_size))
        padded_lengths.append(int(record.padded_length))

    bytes_per_second = (total_bytes / elapsed) if elapsed > 0 else 0.0
    windows_per_second = (total_windows / elapsed) if elapsed > 0 else 0.0
    mean_model_call_batch = (
        float(np.mean(actual_batches)) if actual_batches else 0.0
    )
    max_model_call_batch = max(actual_batches) if actual_batches else 0
    mean_padded_length = (
        float(np.mean(padded_lengths)) if padded_lengths else 0.0
    )
    mean_text_bytes = float(np.mean([int(arr.shape[0]) for arr in byte_arrays]))

    return BenchmarkResult(
        device=str(device_name),
        arch=str(runner.arch),
        text_bytes=int(round(mean_text_bytes)),
        batch_size=int(batch_size),
        samples=int(len(byte_arrays)),
        repeats=int(repeats),
        total_bytes=int(total_bytes),
        total_windows=int(total_windows),
        elapsed_seconds=float(elapsed),
        bytes_per_second=float(bytes_per_second),
        chars_per_second=float(bytes_per_second),
        tokens_per_second=float(bytes_per_second),
        windows_per_second=float(windows_per_second),
        model_calls=int(len(history)),
        mean_model_call_batch=float(mean_model_call_batch),
        max_model_call_batch=int(max_model_call_batch),
        mean_padded_length=float(mean_padded_length),
        path_counts=path_counts,
    )


def _format_path_counts(path_counts: Dict[str, int]) -> str:
    if not path_counts:
        return "-"
    return ", ".join(f"{key}:{path_counts[key]}" for key in sorted(path_counts))


def _render_markdown(results: Sequence[BenchmarkResult]) -> str:
    lines = [
        "| Device | Text bytes | Batch | Samples | Repeats | Tokens/s | Windows/s | Model calls | Mean call batch | Mean pad | Paths |",
        "|--------|-----------:|------:|--------:|--------:|---------:|----------:|------------:|----------------:|---------:|-------|",
    ]
    for result in results:
        lines.append(
            "| {device} | {text_bytes:,} | {batch_size} | {samples} | {repeats} | {tokens_per_second:,.0f} | "
            "{windows_per_second:,.1f} | {model_calls} | {mean_model_call_batch:.2f} | {mean_padded_length:.1f} | {paths} |".format(
                device=result.device,
                text_bytes=result.text_bytes,
                batch_size=result.batch_size,
                samples=result.samples,
                repeats=result.repeats,
                tokens_per_second=result.tokens_per_second,
                windows_per_second=result.windows_per_second,
                model_calls=result.model_calls,
                mean_model_call_batch=result.mean_model_call_batch,
                mean_padded_length=result.mean_padded_length,
                paths=_format_path_counts(result.path_counts),
            )
        )
    return "\n".join(lines)


def _select_best_results(results: Sequence[BenchmarkResult]) -> List[BenchmarkResult]:
    best_by_key: Dict[tuple[str, int], BenchmarkResult] = {}
    for result in results:
        key = (str(result.device), int(result.text_bytes))
        current = best_by_key.get(key)
        if current is None or result.tokens_per_second > current.tokens_per_second:
            best_by_key[key] = result
    return [
        best_by_key[key]
        for key in sorted(best_by_key, key=lambda item: (item[0], item[1]))
    ]


def _render_best_markdown(results: Sequence[BenchmarkResult]) -> str:
    lines = [
        "| Device | Text bytes | Best batch | Tokens/s | Windows/s | Paths |",
        "|--------|-----------:|-----------:|---------:|----------:|-------|",
    ]
    for result in results:
        lines.append(
            "| {device} | {text_bytes:,} | {batch_size} | {tokens_per_second:,.0f} | {windows_per_second:,.1f} | {paths} |".format(
                device=result.device,
                text_bytes=result.text_bytes,
                batch_size=result.batch_size,
                tokens_per_second=result.tokens_per_second,
                windows_per_second=result.windows_per_second,
                paths=_format_path_counts(result.path_counts),
            )
        )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batched byte-level inference throughput on CPU/GPU."
    )
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (.msgpack or Orbax dir).")
    parser.add_argument("--arch", default="unet1d", choices=("unet1d", "mamba", "magika"))
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--channels", default="32,64,64,128,128,128,128,256")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--inference-backend", default="fast", choices=("auto", "fast", "legacy"))
    parser.add_argument("--devices", default="cpu,cuda", help="Comma-separated device names, e.g. cpu,cuda.")
    parser.add_argument(
        "--batch-sizes",
        default="auto",
        help="Comma-separated batch sizes, or 'auto' to sweep powers of two and keep the fastest.",
    )
    parser.add_argument(
        "--auto-batch-limit",
        type=int,
        default=DEFAULT_AUTO_BATCH_LIMIT,
        help="Largest batch size considered when --batch-sizes=auto.",
    )
    parser.add_argument("--text-bytes", default="1024,10240,102400")
    parser.add_argument("--samples-per-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--json-output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--full-memory-budget-mb",
        type=int,
        default=0,
        help="Override fast_full memory budget in MiB. Set small values to force fast_stream.",
    )
    parser.add_argument("--mamba-layers", type=int, default=6)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-expand", type=int, default=1)
    parser.add_argument("--mamba-dt-rank", type=int, default=16)
    parser.add_argument("--mamba-conv", type=int, default=4)
    parser.add_argument("--mamba-bidirectional", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    checkpoint_path: Optional[Path]
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = None
        if str(args.arch).lower().strip() != "magika":
            raise FileNotFoundError("--checkpoint is required unless --arch=magika.")
        args.checkpoint = "magika://default"

    device_names = _parse_csv_strings(args.devices)
    batch_sizes, batch_sizes_auto = _resolve_batch_sizes(
        args.batch_sizes,
        auto_limit=int(args.auto_batch_limit),
    )
    text_sizes = _parse_csv_ints(args.text_bytes)

    prepared: Dict[int, List[np.ndarray]] = {}
    for idx, text_bytes in enumerate(text_sizes):
        prepared[int(text_bytes)] = _generate_ascii_samples(
            text_bytes=int(text_bytes),
            count=max(1, int(args.samples_per_size)),
            seed=int(args.seed) + idx,
        )

    results: List[BenchmarkResult] = []
    print(
        f"Benchmarking {(checkpoint_path.name if checkpoint_path is not None else 'magika://default')} | arch={args.arch} | backend={args.inference_backend}",
        flush=True,
    )
    for device_name in device_names:
        if str(args.arch).lower().strip() == "magika" and str(device_name).lower().strip() != "cpu":
            print(f"SKIP: Magika benchmarking is CPU-only, skipping device '{device_name}'.", flush=True)
            continue
        if not _device_is_available(device_name):
            print(f"SKIP: device '{device_name}' is not available on this machine.", flush=True)
            continue
        for text_bytes in text_sizes:
            byte_arrays = prepared[int(text_bytes)]
            for batch_size in batch_sizes:
                print(
                    f"▶ {device_name} | text_bytes={text_bytes:,} | batch_size={batch_size} | "
                    f"samples={len(byte_arrays)} | repeats={args.repeats}",
                    flush=True,
                )
                runner = _build_runner(args, device_name=device_name, batch_size=int(batch_size))
                result = _benchmark_one(
                    runner=runner,
                    byte_arrays=byte_arrays,
                    batch_size=int(batch_size),
                    repeats=int(args.repeats),
                    warmup_batches=int(args.warmup_batches),
                    device_name=device_name,
                )
                results.append(result)
                print(
                    f"    {result.tokens_per_second:,.0f} tokens/s | {result.windows_per_second:,.1f} windows/s | "
                    f"paths={_format_path_counts(result.path_counts)}",
                    flush=True,
                )

    if not results:
        raise RuntimeError("No benchmarks ran. Check device availability and arguments.")

    markdown = _render_markdown(results)
    print()
    print(markdown)

    if batch_sizes_auto:
        best_results = _select_best_results(results)
        print()
        print("Fastest per device/text size:")
        print(_render_best_markdown(best_results))

    if args.json_output:
        output_path = Path(args.json_output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else str(args.checkpoint),
            "arch": str(args.arch),
            "inference_backend": str(args.inference_backend),
            "devices": device_names,
            "batch_sizes": batch_sizes,
            "batch_sizes_auto": bool(batch_sizes_auto),
            "auto_batch_limit": int(args.auto_batch_limit),
            "text_bytes": text_sizes,
            "samples_per_size": int(args.samples_per_size),
            "repeats": int(args.repeats),
            "full_memory_budget_mb": int(args.full_memory_budget_mb),
            "results": [asdict(result) for result in results],
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON results to {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
