import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, TYPE_CHECKING

from dotenv import load_dotenv
load_dotenv()

# Unbuffered/stdout-friendly logs
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Multiprocessing plays nicer with 'spawn' in JAX/CUDA contexts
import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Environment
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")  # Limit JAX memory usage

import gc
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax.training import checkpoints
from flax import serialization
import wandb
from wandb import Settings

# Enable memory-efficient JAX flags
jax.config.update("jax_disable_jit", False)
jax.config.update("jax_enable_x64", False)

import utils.config as cfg
from utils.data import prepare_dsets_by_lang_with_splits, LANG_ALIASES
from utils.window_generator import (
    make_training_window,
    make_training_window_with_metadata,
)
from utils.model import (
    create_train_state,
    train_step,
    train_step_with_oe,
    eval_step,
    TrainState,
    count_params,
    train_step_no_jit,
    train_step_with_oe_no_jit,
    microbatch_grad_step,
    microbatch_grad_step_with_oe,
    microbatch_grad_step_no_jit,
    microbatch_grad_step_with_oe_no_jit,
    grad_global_norm,
    # Multi-GPU (pmap) variants
    replicate_state,
    unreplicate_state,
    p_train_step,
    p_train_step_with_oe,
    p_microbatch_grad_step,
    p_microbatch_grad_step_with_oe,
    p_eval_step,
)
from utils.preview import build_preview_html

from utils.metrics_helper import (
    evaluate_split_with_metrics,
    compute_metrics_from_confusion,
    print_metrics_table,
    wandb_log_metrics,
)
from utils.monitor_eval import load_monitor_memmaps, evaluate_monitor_set
from utils.token_utils import sanitize_tokens
from utils.outlier_data import OutlierBatcher

if TYPE_CHECKING:
    from utils.config import DataConfig, TrainConfig

import signal

# global-ish flag that both the handler and loop can see
_STOP = {"flag": False}


def _signal_handler(sig, frame):
    _STOP["flag"] = True
    print(f"Signal {sig} received; stopping...", flush=True)


# register early, before long inits
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _compute_running_stats(history, new_value, window=100):
    history.append(float(new_value))
    if len(history) > window:
        history.pop(0)
    import numpy as _np

    return {
        "mean": float(_np.mean(history)),
        "std": float(_np.std(history)) if len(history) > 1 else 0.0,
        "trend": (
            float(history[-1] - history[0]) / len(history) if len(history) > 1 else 0.0
        ),
    }


def _wandb_safe_log(data: dict, step: int, commit: bool = True):
    try:
        wandb.log(data, step=step, commit=commit)
    except Exception as e:
        print(f"Wandb logging failed: {e}", flush=True)


def resolve_ckpt_paths(path: str):
    blob_abs = os.path.abspath(path)
    ckpt_dir_abs = os.path.dirname(blob_abs) or os.getcwd()
    base = os.path.basename(blob_abs)
    prefix = base + "-"
    return ckpt_dir_abs, prefix, blob_abs


def _find_local_wandb_config(repo_root: Path, run_id: str) -> Path | None:
    wandb_root = repo_root / "wandb"
    if not wandb_root.exists():
        return None
    candidates = sorted(wandb_root.glob(f"run-*-{run_id}/files/config.yaml"))
    return candidates[-1] if candidates else None


def _read_wandb_config_value(config_path: Path, key: str) -> str | None:
    """Read a simple top-level `key: { value: ... }` entry from wandb config.yaml."""
    try:
        lines = config_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    needle = f"{key}:"
    for i, line in enumerate(lines):
        if not line.startswith(" ") and line.strip() == needle:
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line and not next_line.startswith(" "):
                    break
                stripped = next_line.strip()
                if stripped.startswith("value:"):
                    value = stripped.split("value:", 1)[1].strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                        value = value[1:-1]
                    return value
    return None


def _make_eval_batch(
    dsets_by_lang: Dict[int, dict],
    L: int,
    batch_size: int,
    data_cfg: "DataConfig",
) -> Tuple[np.ndarray, np.ndarray]:
    xb = np.full((batch_size, L), cfg.PAD_BYTE_ID, dtype=np.int32)
    yb = np.full((batch_size, L), cfg.PAD_ID, dtype=np.uint8)
    for i in range(batch_size):
        x, y = make_training_window(dsets_by_lang, L, data_cfg)
        xb[i] = x
        yb[i] = y
    xb = sanitize_tokens(xb)
    return xb, yb


def evaluate_split(
    state: TrainState,
    dsets_by_lang: Dict[int, dict],
    L: int,
    batch_size: int,
    batches: int,
    data_cfg: "DataConfig",
    rng,
) -> Tuple[float, float]:
    losses, accs = [], []
    for i in range(batches):
        data_rng, eval_rng = jax.random.split(jax.random.fold_in(rng, i))
        seed_val = int(jax.random.randint(data_rng, (), 0, 2**31 - 1).item())
        np.random.seed(seed_val)
        random.seed(seed_val)
        xb, yb = _make_eval_batch(dsets_by_lang, L, batch_size, data_cfg)
        loss, acc = eval_step(
            state,
            jnp.array(xb, dtype=jnp.int32),
            jnp.array(yb, dtype=jnp.uint8),
            eval_rng,
        )
        losses.append(float(loss))
        accs.append(float(acc))
    import numpy as _np

    return float(_np.mean(losses)), float(_np.mean(accs))


def main():
    # Force garbage collection at start
    gc.collect()

    ckpt_async_manager = None

    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    default_monitor_root = repo_root / "downloader" / "monitor_preprocessed_b"
    default_fine_tune_train_root = repo_root / "downloader" / "monitor_preprocessed_a"
    default_fine_tune_val_root = repo_root / "downloader" / "monitor_preprocessed_b"

    # Data args
    parser.add_argument(
        "--data_root", type=str, default=str(repo_root / "downloader" / "arrow_out")
    )
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # 4KB fixed windows by default
    parser.add_argument("--bucket_step", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--max_minutes", type=int, default=0)
    parser.add_argument("--stop_file", type=str, default="STOP_SWEEP")
    parser.add_argument("--dont_use_train_windows", action="store_true", default=False,
                      help="Use train directory instead of train_windows for training data")
    parser.add_argument(
        "--language_pair_prob",
        type=float,
        default=None,
        help="Probability of enabling curated language pair mixing/transitivity (0 disables, default).",
    )
    parser.add_argument(
        "--lang",
        dest="langs",
        action="append",
        default=None,
        help="Restrict training to specific languages (repeatable or comma-separated).",
    )

    # Pruning
    parser.add_argument("--prune_min_minutes", type=int, default=15)
    parser.add_argument("--prune_patience_evals", type=int, default=9999999999)
    parser.add_argument("--prune_delta", type=float, default=0.000001)

    # Train args
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument(
        "--arch",
        type=str,
        default="unet1d",
        choices=("unet1d", "mamba"),
        help="Model architecture (unet1d or mamba).",
    )
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--channels", type=str, default="32,64,64,128,128,128,128,256")
    parser.add_argument("--dropout_rate", type=float, default=0.15)
    # Mamba-only knobs (ignored for unet1d). Defaults match TrainConfig.
    parser.add_argument("--mamba_layers", type=int, default=6)
    parser.add_argument("--mamba_d_state", type=int, default=8)
    parser.add_argument("--mamba_expand", type=int, default=1)
    parser.add_argument("--mamba_dt_rank", type=int, default=16)
    parser.add_argument("--mamba_conv", type=int, default=4)
    parser.add_argument(
        "--mamba_bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bidirectional scan for per-position segmentation.",
    )
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument(
        "--eval_batches",
        type=int,
        default=50,
        help="Number of random validation batches to sample at each eval.",
    )
    parser.add_argument(
        "--monitor_eval_every",
        type=int,
        default=-1,
        help="Run monitor validation after each val eval (0 disables).",
    )
    parser.add_argument(
        "--monitor_eval_limit",
        type=int,
        default=4096,
        help="Max monitor files to sample per monitor eval (-1 = use all files; be careful, this is slow).",
    )
    parser.add_argument(
        "--monitor_eval_root",
        type=str,
        default=str(default_monitor_root),
        help="Path to downloader/monitor_preprocessed_b (monitor validation memmap).",
    )
    parser.add_argument(
        "--monitor_other_threshold",
        type=float,
        default=0.3,
        help="If >0, route monitor predictions with max softmax below this threshold to 'other'.",
    )
    parser.add_argument("--ckpt_path", type=str, default="auto")
    parser.add_argument(
        "--continue",
        dest="continue_run_id",
        type=str,
        default="",
        help="Resume from an existing W&B run id (reuses its checkpoint and W&B run).",
    )
    parser.add_argument(
        "--fine-tune",
        dest="fine_tune",
        nargs="?",
        const="",
        default=None,
        metavar="RUNID-STEP|CKPT",
        help=(
            "Enable fine-tuning from a pre-existing W&B run. "
            "Optionally specify 'RUNID-STEP' (e.g. fkbz80vt-468000) to "
            "load a specific checkpoint step from checkpoints/sweeps, "
            "or pass a local checkpoint path (.msgpack) to fine-tune from."
        ),
    )
    parser.add_argument(
        "--fine_tune_run_id",
        type=str,
        default="",
        help="Source W&B run id to load the checkpoint from when --fine-tune is set and not resuming a fine-tune run.",
    )
    parser.add_argument(
        "--fine_tune_ckpt_path",
        type=str,
        default="",
        help=(
            "Optional local checkpoint path to load from when fine-tuning "
            "(raw params .msgpack or a Flax checkpoint dir). Overrides --fine_tune_run_id."
        ),
    )
    parser.add_argument(
        "--fine_tune_train_root",
        type=str,
        default=str(default_fine_tune_train_root),
        help="Root directory for the fine-tune training memmap (monitor_preprocessed_a).",
    )
    parser.add_argument(
        "--fine_tune_val_root",
        type=str,
        default=str(default_fine_tune_val_root),
        help="Root directory for the fine-tune validation/monitor memmap (monitor_preprocessed_b).",
    )
    parser.add_argument("--sweep_id", type=str, default="")
    parser.add_argument("--no_jit", action="store_true")
    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--preview_start", type=int, default=0)
    parser.add_argument("--preview_count", type=int, default=10)
    parser.add_argument(
        "--active_learning_store",
        type=str,
        default="",
        help="Optional SQLite label store produced by active_learning/round.py.",
    )
    parser.add_argument(
        "--active_learning_mix_prob",
        type=float,
        default=0.1,
        help="Probability of replacing each training row with active-learning replay data.",
    )
    parser.add_argument(
        "--active_learning_max_windows",
        type=int,
        default=1000000,
        help="Cap the number of replay windows loaded from the active-learning store.",
    )
    parser.add_argument(
        "--oe-lambda",
        type=float,
        default=0.1,
        help="Weight for OE uniform-target loss on outlier batches (0 disables OE).",
    )
    parser.add_argument(
        "--oe-ratio",
        type=float,
        default=0.05,
        help="Probability of adding one outlier batch to a training step.",
    )
    parser.add_argument(
        "--oe-source",
        type=str,
        default="mixed",
        help=(
            "Outlier sampling mode. Default/expected is 'mixed': sample random and heldout "
            "OTHER data at 50:50 when heldout is available."
        ),
    )
    parser.add_argument(
        "--oe-heldout-root",
        type=str,
        default=str(repo_root / "downloader" / "arrow_out_other"),
        help="Root containing heldout Arrow data in <root>/train/<label>/dataset (typically train/other/dataset).",
    )
    parser.add_argument(
        "--fine_tune_use_oe",
        action="store_true",
        default=False,
        help=(
            "Enable OE outlier augmentation during fine-tuning. "
            "By default, fine-tuning disables OE and relies on monitor labels (including OTHER) only."
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs for data-parallel training (default: 2). Fails if fewer devices are available.",
    )

    args = parser.parse_args()

    # Interpret --fine-tune argument (optional RUNID-STEP shorthand or a local ckpt path).
    fine_tune_raw = getattr(args, "fine_tune", None)
    fine_tune_run_id = getattr(args, "fine_tune_run_id", "")
    fine_tune_ckpt_path = getattr(args, "fine_tune_ckpt_path", "")
    fine_tune_step = None
    if isinstance(fine_tune_raw, str):
        if fine_tune_raw:
            raw = str(fine_tune_raw).strip()
            raw_path = Path(raw)
            looks_like_path = (
                raw.endswith(".msgpack")
                or "/" in raw
                or "\\" in raw
                or raw_path.exists()
            )
            if looks_like_path:
                fine_tune_ckpt_path = raw
            else:
                parts = raw.rsplit("-", 1)
                if (
                    len(parts) == 2
                    and parts[1].isdigit()
                    and len(parts[0]) == 8
                    and parts[0].isalnum()
                ):
                    fine_tune_run_id = parts[0]
                    fine_tune_step = int(parts[1])
                else:
                    fine_tune_run_id = raw
        else:
            # '--fine-tune' present without RUNID-STEP; rely on --fine_tune_run_id
            pass
    fine_tune_enabled = (fine_tune_raw is not None) or bool(fine_tune_run_id) or bool(fine_tune_ckpt_path)
    args.fine_tune = fine_tune_enabled
    args.fine_tune_run_id = fine_tune_run_id
    args.fine_tune_ckpt_path = fine_tune_ckpt_path
    args.fine_tune_step = fine_tune_step

    # Resolve learning rate depending on mode if not explicitly provided.
    if args.lr is None:
        # Use a smaller default LR for fine-tuning.
        args.lr = 3e-5 if args.fine_tune else 1e-3

    if args.fine_tune and not args.continue_run_id:
        # For a fresh fine-tune run we require a source run id or an explicit checkpoint path.
        if not args.fine_tune_run_id and not args.fine_tune_ckpt_path:
            raise ValueError(
                "--fine-tune requires either RUNID-STEP, --fine_tune_run_id, "
                "or --fine_tune_ckpt_path when not resuming with --continue."
            )

    if args.accum_steps < 1:
        raise ValueError("--accum_steps must be >= 1")
    if args.oe_lambda < 0.0:
        raise ValueError("--oe-lambda must be >= 0")
    if args.oe_ratio < 0.0:
        raise ValueError("--oe-ratio must be >= 0")

    # --- Multi-GPU validation (fail fast) ---
    num_devices = args.num_gpus
    available_devices = jax.device_count()
    if num_devices > available_devices:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"  FATAL: Requested --num-gpus={num_devices} but only "
            f"{available_devices} JAX device(s) available!\n"
            f"  Detected devices: {jax.devices()}\n"
            f"{'='*60}"
        )
    if num_devices < 1:
        raise ValueError("--num-gpus must be >= 1")
    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by "
            f"num-gpus ({num_devices}). "
            f"Try --batch_size={args.batch_size - args.batch_size % num_devices + num_devices}"
        )
    per_device_batch = args.batch_size // num_devices
    use_pmap = num_devices > 1

    if args.no_jit:
        print(
            f"\n{'!'*60}\n"
            f"  WARNING: --no_jit is active! Training will use a SINGLE\n"
            f"  device without JIT compilation (ignoring --num-gpus={num_devices}).\n"
            f"  This is for DEBUGGING ONLY and will be extremely slow.\n"
            f"{'!'*60}\n",
            flush=True,
        )
        use_pmap = False
        num_devices = 1

    if args.fine_tune and not bool(getattr(args, "fine_tune_use_oe", False)):
        if float(args.oe_lambda) > 0.0 or float(args.oe_ratio) > 0.0:
            print(
                "Fine-tune mode: disabling OE outlier augmentation by default "
                "(set --fine_tune_use_oe to keep OE enabled).",
                flush=True,
            )
        args.oe_lambda = 0.0
        args.oe_ratio = 0.0

    def _resolve_cli_path(raw_path: str) -> Path:
        p = Path(str(raw_path or "").strip()).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p)
        return p.resolve()

    def _has_heldout_arrow_dataset(root: Path) -> bool:
        primary = root / "train" / "other" / "dataset"
        if primary.exists():
            return True
        train_root = root / "train"
        if not train_root.exists() or not train_root.is_dir():
            return False
        for child in train_root.iterdir():
            if child.is_dir() and (child / "dataset").exists():
                return True
        return False

    oe_lambda = float(args.oe_lambda)
    oe_ratio = float(args.oe_ratio)
    oe_source_raw = str(args.oe_source or "mixed").strip()
    oe_source = oe_source_raw.lower()
    if oe_lambda > 0.0 and oe_ratio > 0.0 and oe_source != "random":
        heldout_hint = ""
        if oe_source.startswith("heldout:"):
            heldout_hint = str(oe_source_raw.split(":", 1)[1]).strip()
        heldout_root_raw = heldout_hint if heldout_hint else str(args.oe_heldout_root)
        heldout_root = _resolve_cli_path(heldout_root_raw)
        if not _has_heldout_arrow_dataset(heldout_root):
            raise FileNotFoundError(
                "OE is enabled with non-random source "
                f"('--oe-source {oe_source_raw}'), but no heldout Arrow dataset was found under "
                f"'{heldout_root}'. Expected at least '<root>/train/other/dataset'."
            )

    selected_langs = None
    if args.langs:
        normalized = []
        for entry in args.langs:
            parts = [part.strip() for part in entry.split(",") if part.strip()]
            if parts:
                normalized.extend(parts)
        if normalized:
            seen = set()
            selected_langs = []
            alias_map = {
                alias.lower(): canonical
                for canonical, aliases in LANG_ALIASES.items()
                for alias in aliases
            }
            alias_map.update({
                "c++": "c_family",
                "c-family": "c_family",
                "cfamily": "c_family",
            })
            for lang in normalized:
                lang_key = lang.lower()
                canonical = alias_map.get(lang_key, lang_key)
                if canonical not in seen:
                    seen.add(canonical)
                    selected_langs.append(canonical)
            if selected_langs:
                print(f"Restricting training to languages: {selected_langs}", flush=True)
            else:
                selected_langs = None
        else:
            print("No valid languages provided via --lang; falling back to all languages.", flush=True)

    # Build configs
    d_cfg = cfg.DataConfig(
        data_root=args.data_root,
        num_proc=args.num_proc,
        seed=args.seed,
        bucket_step=args.bucket_step,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.language_pair_prob is not None:
        prob = max(0.0, min(1.0, float(args.language_pair_prob)))
        d_cfg.language_pair_mode_prob = prob
    monitor_every = 0 if args.monitor_eval_every == 0 else args.eval_every

    # For fine-tuning, always use the dedicated monitor_preprocessed_b memmap
    # as the monitor/validation set, regardless of any custom monitor_eval_root.
    if args.fine_tune:
        args.monitor_eval_root = args.fine_tune_val_root

    t_cfg = cfg.TrainConfig(
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        accum_steps=args.accum_steps,
        arch=args.arch,
        model_dim=args.model_dim,
        channels=tuple(map(int, args.channels.split(","))),
        dropout_rate=args.dropout_rate,
        mamba_layers=int(args.mamba_layers),
        mamba_d_state=int(args.mamba_d_state),
        mamba_expand=int(args.mamba_expand),
        mamba_dt_rank=int(args.mamba_dt_rank),
        mamba_conv=int(args.mamba_conv),
        mamba_bidirectional=bool(args.mamba_bidirectional),
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=int(args.eval_batches),
        monitor_eval_every=int(monitor_every),
        monitor_eval_limit=int(args.monitor_eval_limit),
        monitor_eval_root=args.monitor_eval_root,
        monitor_other_threshold=float(args.monitor_other_threshold),
        oe_lambda=float(args.oe_lambda),
        oe_ratio=float(args.oe_ratio),
        oe_source=str(args.oe_source),
        oe_heldout_root=str(args.oe_heldout_root),
        ckpt_path=args.ckpt_path,
        sweep_id=args.sweep_id,
        no_jit=args.no_jit,
        preview_only=args.preview_only,
        preview_start=args.preview_start,
        preview_count=args.preview_count,
        fine_tune=args.fine_tune,
        fine_tune_run_id=args.fine_tune_run_id,
        fine_tune_train_root=args.fine_tune_train_root,
        fine_tune_val_root=args.fine_tune_val_root,
        fine_tune_step=args.fine_tune_step,
    )

    # Prepare datasets
    print("Preparing datasets...", flush=True)
    dsets = prepare_dsets_by_lang_with_splits(
        d_cfg.data_root,
        use_train_windows=not args.dont_use_train_windows,
        include_languages=selected_langs,
        verbose=False,
    )
    train_dsets = dsets["train"]

    # Preview mode (mirrors training distribution, including mixed overlays)
    if t_cfg.preview_only:
        print(
            f"--- PREVIEW MODE: generating {t_cfg.preview_count} examples ---",
            flush=True,
        )
        examples = []

        for i in range(t_cfg.preview_start, t_cfg.preview_start + t_cfg.preview_count):
            np.random.seed(i)
            random.seed(i)
            L = random.choice(d_cfg.buckets())

            x, y, meta = make_training_window_with_metadata(train_dsets, L, d_cfg)
            x = sanitize_tokens(x)
            examples.append({"tokens": x, "labels": y, "metadata": meta})

        out_path = "preview.html"
        build_preview_html(examples, out_path)
        print(f"Preview HTML written to: {os.path.abspath(out_path)}")
        return

    fine_tune_data = None
    fine_tune_train_files_count = 0
    if args.fine_tune:
        fine_tune_root = Path(args.fine_tune_train_root)
        if not fine_tune_root.exists():
            raise ValueError(
                f"Fine-tune training root not found at {fine_tune_root}. "
                "Ensure downloader/monitor_preprocessed_a has been created."
            )
        try:
            fine_tune_data = load_monitor_memmaps(fine_tune_root)
            meta_ft = fine_tune_data.get("meta", {})
            total_files_ft = meta_ft.get("num_files", len(fine_tune_data["files"]))
            fine_tune_train_files_count = int(total_files_ft)
            print(
                f"Fine-tune training enabled: {total_files_ft} files from {fine_tune_root}",
                flush=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load fine-tune monitor memmap from {fine_tune_root}: {e}"
            ) from e

    monitor_data = None
    monitor_eval_files_count = 0
    if t_cfg.monitor_eval_every > 0:
        monitor_root = Path(t_cfg.monitor_eval_root)
        if monitor_root.exists():
            try:
                monitor_data = load_monitor_memmaps(monitor_root)
                meta = monitor_data.get("meta", {})
                total_files = meta.get("num_files", len(monitor_data["files"]))
                monitor_eval_files_count = int(total_files)
                print(
                    f"Monitor eval enabled: {total_files} files from {monitor_root}",
                    flush=True,
                )
            except Exception as e:
                print(f"⚠️  Monitor eval disabled (load failure): {e}", flush=True)
                t_cfg.monitor_eval_every = 0
        else:
            print(f"⚠️  Monitor eval root not found at {monitor_root}, disabling.", flush=True)
            t_cfg.monitor_eval_every = 0

    # --- WandB: single init here (longer timeout) ---
    wandb_init_kwargs = {
        "project": os.getenv("WANDB_PROJECT", "code-segmentation-v2"),
        "settings": Settings(init_timeout=300, start_method="thread"),
    }
    if getattr(args, "continue_run_id", ""):
        # Resume an existing run (and reuse its checkpoint path).
        wandb_init_kwargs.update(
            {
                "id": args.continue_run_id,
                "resume": "allow",
            }
        )
        print(f"Resuming W&B run: {args.continue_run_id}", flush=True)
    wandb.init(**wandb_init_kwargs)
    wandb.config.update({**d_cfg.__dict__, **t_cfg.__dict__}, allow_val_change=True)
    wandb.config.update(
        {
            "active_learning_store": args.active_learning_store,
            "active_learning_mix_prob": float(args.active_learning_mix_prob),
            "active_learning_max_windows": int(args.active_learning_max_windows),
            "monitor_train_files": int(fine_tune_train_files_count),
            "monitor_eval_files": int(monitor_eval_files_count),
            "num_gpus": int(num_devices),
            "data_parallel": use_pmap,
        },
        allow_val_change=True,
    )
    _wandb_safe_log(
        {
            "meta/monitor_train_files": int(fine_tune_train_files_count),
            "meta/monitor_eval_files": int(monitor_eval_files_count),
        },
        step=0,
        commit=False,
    )

    # Derive unique checkpoint path from run id if requested/placeholder-ish
    if t_cfg.ckpt_path == "auto" or "${" in t_cfg.ckpt_path:
        auto_ckpt = f"checkpoints/sweeps/{wandb.run.id}.msgpack"
        os.makedirs(os.path.dirname(auto_ckpt), exist_ok=True)
        t_cfg.ckpt_path = auto_ckpt
        print(f"Using auto checkpoint path: {t_cfg.ckpt_path}", flush=True)

    # Warm up device early so any XLA/CUDA issues show now
    print("JAX devices:", jax.devices(), flush=True)
    if use_pmap:
        print(
            f"Data-parallel training: {num_devices} GPUs, "
            f"global batch={args.batch_size}, per-device batch={per_device_batch}",
            flush=True,
        )
    _ = jnp.ones((1,)).block_until_ready()

    rng = jax.random.PRNGKey(t_cfg.rng_seed)
    rng, init_rng = jax.random.split(rng)

    print("Creating train state (may trigger JIT/compile)...", flush=True)
    state = create_train_state(init_rng, t_cfg, cfg.NUM_CLASSES)
    num_params = count_params(state.params)
    print(f"Model created with {num_params/1e6:.2f}M parameters.", flush=True)

    def _strict_load_params(target_params, raw_bytes, path_desc=""):
        from flax.traverse_util import flatten_dict
        msgpack_dict = serialization.msgpack_restore(raw_bytes)
        flat_msgpack = flatten_dict(msgpack_dict)
        flat_target = flatten_dict(serialization.to_state_dict(target_params))
        missing_keys = set(flat_target.keys()) - set(flat_msgpack.keys())
        extra_keys = set(flat_msgpack.keys()) - set(flat_target.keys())
        if missing_keys or extra_keys:
            raise ValueError(
                f"Checkpoint structure mismatch for {path_desc}!\n"
                f"Missing from checkpoint: {missing_keys}\n"
                f"Extra in checkpoint: {extra_keys}"
            )
        return serialization.from_state_dict(target_params, msgpack_dict)

    # If we're starting a new fine-tune run (not resuming), load weights
    # from the source W&B run's checkpoint before configuring this run's
    # own checkpoint path.
    if t_cfg.fine_tune and not getattr(args, "continue_run_id", ""):
        source_run_id = t_cfg.fine_tune_run_id
        source_ckpt_path = (getattr(args, "fine_tune_ckpt_path", "") or "").strip()
        source_ckpt_source = "cli"

        if not source_ckpt_path:
            if not source_run_id:
                raise ValueError(
                    "Fine-tune requested but no source provided. "
                    "Pass --fine_tune_run_id or --fine_tune_ckpt_path."
                )

            sweeps_path = (repo_root / "checkpoints" / "sweeps" / f"{source_run_id}.msgpack").resolve()
            if sweeps_path.exists():
                source_ckpt_path = str(sweeps_path)
                source_ckpt_source = "checkpoints/sweeps"
            else:
                config_path = _find_local_wandb_config(repo_root, source_run_id)
                ckpt_from_wandb = (
                    _read_wandb_config_value(config_path, "ckpt_path")
                    if config_path is not None
                    else None
                )
                if ckpt_from_wandb:
                    candidate = Path(ckpt_from_wandb)
                    if not candidate.is_absolute():
                        candidate = (repo_root / candidate).resolve()
                    if candidate.exists():
                        source_ckpt_path = str(candidate)
                        source_ckpt_source = f"wandb:{config_path}"

                if not source_ckpt_path:
                    raise ValueError(
                        "Could not resolve a fine-tune checkpoint for run id "
                        f"{source_run_id}. Tried:\n"
                        f"  - {sweeps_path}\n"
                        f"  - wandb local config (wandb/run-*-{source_run_id}/files/config.yaml)\n"
                        "Provide a path explicitly via --fine_tune_ckpt_path."
                    )

        src_ckpt_dir, src_ckpt_prefix, src_ckpt_blob = resolve_ckpt_paths(source_ckpt_path)
        step_arg = t_cfg.fine_tune_step
        print(
            f"Fine-tune source checkpoint: {source_ckpt_path} ({source_ckpt_source})",
            flush=True,
        )

        # When a specific step is requested (RUNID-STEP), prefer the
        # raw-params snapshot named <run_id>-<step>.msgpack.
        if step_arg is not None:
            base = os.path.basename(src_ckpt_blob)
            stem, ext = os.path.splitext(base)
            hist_path = os.path.join(src_ckpt_dir, f"{stem}-{step_arg}{ext}")
            if os.path.exists(hist_path):
                print(
                    f"Restoring fine-tune params from {hist_path}",
                    flush=True,
                )
                with open(hist_path, "rb") as f:
                    raw = f.read()
                params = _strict_load_params(state.params, raw, path_desc=hist_path)
                state = state.replace(params=params)
            else:
                # Fallback: try the Flax checkpoint layout
                flax_ckpt_path = os.path.join(src_ckpt_dir, f"{src_ckpt_prefix}{step_arg}")
                if os.path.exists(flax_ckpt_path):
                    print(
                        f"Restoring fine-tune checkpoint at step {step_arg}",
                        flush=True,
                    )
                    restored = checkpoints.restore_checkpoint(
                        src_ckpt_dir, state, step=step_arg, prefix=src_ckpt_prefix
                    )
                    state = state.replace(params=restored.params)
                else:
                    raise ValueError(
                        f"Requested fine-tune checkpoint step {step_arg} for run "
                        f"{source_run_id} not found. Expected either:\n"
                        f"  - {hist_path} (raw params snapshot), or\n"
                        f"  - {flax_ckpt_path} (Flax TrainState checkpoint)."
                    )
        else:
            # No explicit step: load the raw-params blob if present; otherwise fall back to
            # the latest Flax checkpoint, but only keep its params (do not reuse optimizer state).
            if os.path.exists(src_ckpt_blob):
                print(
                    f"Restoring fine-tune params from {src_ckpt_blob}",
                    flush=True,
                )
                with open(src_ckpt_blob, "rb") as f:
                    raw = f.read()
                params = _strict_load_params(state.params, raw, path_desc=src_ckpt_blob)
                state = state.replace(params=params)
            else:
                restored = checkpoints.restore_checkpoint(
                    src_ckpt_dir, state, prefix=src_ckpt_prefix
                )
                if restored is state:
                    raise ValueError(
                        f"Fine-tune source checkpoint not found at {source_ckpt_path}. "
                        "Provide --fine_tune_ckpt_path or a RUNID-STEP that exists."
                    )
                state = state.replace(params=restored.params)

    ckpt_dir, ckpt_prefix, ckpt_blob = resolve_ckpt_paths(t_cfg.ckpt_path)
    ckpt_async_manager = checkpoints.AsyncManager() if hasattr(checkpoints, "AsyncManager") else None
    if os.path.exists(ckpt_blob) or os.path.exists(
        os.path.join(ckpt_dir, f"{ckpt_prefix}0")
    ):
        print("Restoring checkpoint...", flush=True)
        state = checkpoints.restore_checkpoint(ckpt_dir, state, prefix=ckpt_prefix)

    # Replicate state across devices for data-parallel training
    if use_pmap:
        state = replicate_state(state, num_devices)
        print(f"State replicated across {num_devices} devices.", flush=True)

    # Switch to epoch-based batching
    from utils.epoch_batcher import EpochPrefetchBatcher, MonitorFineTuneBatcher

    if t_cfg.fine_tune:
        if fine_tune_data is None:
            raise RuntimeError(
                "Fine-tune mode enabled but fine-tune data failed to load."
            )
        data_fetcher = MonitorFineTuneBatcher(fine_tune_data, d_cfg)
    else:
        data_fetcher = EpochPrefetchBatcher(train_dsets, d_cfg)

    al_store_path = (args.active_learning_store or "").strip()
    al_mix_prob = float(max(0.0, min(1.0, args.active_learning_mix_prob)))
    if al_store_path and al_mix_prob > 0.0:
        try:
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from active_learning.training import ActiveLearningReplay, MixedBatcher

            al_label_to_id = dict(cfg.LANG2ID)
            if getattr(cfg, "OTHER_CLASS_INDEX", None) is not None:
                al_label_to_id["other"] = int(cfg.OTHER_CLASS_INDEX)
            replay = ActiveLearningReplay.from_store(
                store_path=al_store_path,
                window_bytes=int(d_cfg.window_max_bytes),
                pad_byte_id=int(cfg.PAD_BYTE_ID),
                pad_label_id=int(cfg.PAD_ID),
                label_to_id=al_label_to_id,
                max_windows=int(args.active_learning_max_windows),
                seed=int(args.seed),
                fallback_label="other",
            )
            if replay.size > 0:
                data_fetcher = MixedBatcher(
                    data_fetcher,
                    replay,
                    mix_prob=al_mix_prob,
                    seed=int(args.seed),
                )
                print(
                    f"Active-learning replay enabled: {replay.size} windows from {al_store_path} "
                    f"(mix_prob={al_mix_prob:.2f}).",
                    flush=True,
                )
            else:
                print(
                    f"Active-learning store '{al_store_path}' contained no usable windows; replay disabled.",
                    flush=True,
                )
        except Exception as e:
            print(
                f"⚠️  Active-learning replay disabled (load failure): {e}",
                flush=True,
            )

    oe_lambda = float(max(0.0, getattr(t_cfg, "oe_lambda", 0.0)))
    oe_ratio = float(max(0.0, min(1.0, getattr(t_cfg, "oe_ratio", 0.0))))
    oe_batcher = None
    if oe_lambda > 0.0 and oe_ratio > 0.0:
        oe_batcher = OutlierBatcher(
            source=str(getattr(t_cfg, "oe_source", "random")),
            data_root=str(d_cfg.data_root),
            heldout_root=str(getattr(t_cfg, "oe_heldout_root", "")),
            window_bytes=int(d_cfg.window_max_bytes),
            batch_size=int(d_cfg.batch_size),
            seed=int(args.seed),
        )
        oe_desc = oe_batcher.describe()
        source_mode = str(oe_desc.get("mode", "random"))
        if source_mode == "random" and str(oe_desc.get("source", "")).strip().lower() not in {
            "random",
            "",
        }:
            raise RuntimeError(
                "OE source requested heldout/mixed data, but heldout data was not loaded. "
                "Refusing to fall back to random-only outliers."
            )
        print(
            "OE enabled: "
            f"lambda={oe_lambda:.4f}, ratio={oe_ratio:.3f}, "
            f"source={oe_desc.get('source')}, mode={source_mode}, "
            f"heldout_langs={oe_desc.get('heldout_langs')}",
            flush=True,
        )
    elif oe_lambda > 0.0:
        print(
            f"⚠️  OE lambda set to {oe_lambda:.4f} but oe-ratio is {oe_ratio:.3f}; OE is effectively disabled.",
            flush=True,
        )

    # Select step functions: pmap (multi-GPU) vs jit (single-GPU) vs no_jit (debug)
    if use_pmap:
        train_step_fn = p_train_step
        train_step_oe_fn = p_train_step_with_oe
        micro_step_fn = p_microbatch_grad_step
        micro_step_oe_fn = p_microbatch_grad_step_with_oe
    elif t_cfg.no_jit:
        train_step_fn = train_step_no_jit
        train_step_oe_fn = train_step_with_oe_no_jit
        micro_step_fn = microbatch_grad_step_no_jit
        micro_step_oe_fn = microbatch_grad_step_with_oe_no_jit
    else:
        train_step_fn = train_step
        train_step_oe_fn = train_step_with_oe
        micro_step_fn = microbatch_grad_step
        micro_step_oe_fn = microbatch_grad_step_with_oe
    accum_steps = max(1, int(t_cfg.accum_steps))
    if accum_steps > 1:
        effective_batch = accum_steps * d_cfg.batch_size
        print(
            f"Gradient accumulation: {accum_steps} microbatches "
            f"(effective batch size ≈ {effective_batch})",
            flush=True,
        )

    def shard_batch(*arrays):
        """Reshape arrays for pmap: (B, ...) -> (num_devices, B//num_devices, ...)."""
        return tuple(a.reshape(num_devices, -1, *a.shape[1:]) for a in arrays)

    def make_pmap_rngs(rng_key):
        """Split an RNG key into one per device for pmap."""
        return jax.random.split(rng_key, num_devices)

    def run_val_and_monitor_eval(
        step: int,
        *,
        title: str = "Evaluating",
        train_loss: float | None = None,
        train_acc: float | None = None,
    ) -> tuple[float, float]:
        """Run val eval and, if enabled, monitor eval; log everything to W&B."""
        nonlocal rng

        print(f"{title}...", flush=True)
        rng, eval_rng = jax.random.split(rng)

        # Eval runs single-device; unreplicate if using pmap
        eval_state = unreplicate_state(state) if use_pmap else state

        val_loss, val_acc, conf_mat = evaluate_split_with_metrics(
            state=eval_state,
            dsets_by_lang=dsets["val"],
            L=d_cfg.window_max_bytes,
            batch_size=d_cfg.batch_size,
            batches=t_cfg.eval_batches,
            data_cfg=d_cfg,
            rng=eval_rng,
            eval_step_fn=eval_step,
        )

        per_class, aggregates = compute_metrics_from_confusion(
            conf_mat, cfg.NUM_CLASSES, cfg.PAD_ID
        )

        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}", flush=True)
        val_metrics = {
            "val/loss": val_loss,
            "val/acc": val_acc,
        }
        if train_loss is not None:
            val_metrics["val/train_gap"] = float(train_loss) - val_loss
        if train_acc is not None:
            val_metrics["val/acc_gap"] = float(train_acc) - val_acc
        _wandb_safe_log(val_metrics, step=step, commit=False)

        print_metrics_table(per_class, aggregates, cfg.ID2LANG, cfg.NUM_CLASSES, cfg.PAD_ID)

        if monitor_data is not None and t_cfg.monitor_eval_every > 0:
            print("Running monitor evaluation...", flush=True)
            eval_rng, monitor_rng = jax.random.split(eval_rng)
            limit = None if t_cfg.monitor_eval_limit < 0 else int(t_cfg.monitor_eval_limit)
            monitor_stats = evaluate_monitor_set(
                state=eval_state,
                monitor_data=monitor_data,
                L=d_cfg.window_max_bytes,
                batch_size=d_cfg.batch_size,
                rng=monitor_rng,
                limit=limit,
                eval_step_fn=eval_step,
                other_threshold=t_cfg.monitor_other_threshold,
            )

            per_class_m, aggregates_m = compute_metrics_from_confusion(
                monitor_stats["conf_mat"], cfg.NUM_CLASSES, cfg.PAD_ID
            )
            print_metrics_table(
                per_class_m,
                aggregates_m,
                cfg.ID2LANG,
                cfg.NUM_CLASSES,
                cfg.PAD_ID,
                title="Monitor",
            )
            monitor_scalar_logs = {
                "monitor/loss": monitor_stats["loss_mean"],
                "monitor/acc": monitor_stats["acc_mean"],
                "monitor/windows": monitor_stats["windows"],
                "monitor/skipped": monitor_stats["skipped"],
                "monitor/files_used": monitor_stats["files_used"],
                "monitor/threshold": t_cfg.monitor_other_threshold,
            }
            monitor_gap_logs = {
                "monitor_gap/micro_accuracy": float(aggregates_m["micro"]["acc"] - aggregates["micro"]["acc"]),
                "monitor_gap/macro_f1": float(aggregates_m["macro"]["f1"] - aggregates["macro"]["f1"]),
                "monitor_gap/macro_precision": float(aggregates_m["macro"]["precision"] - aggregates["macro"]["precision"]),
                "monitor_gap/macro_recall": float(aggregates_m["macro"]["recall"] - aggregates["macro"]["recall"]),
                "monitor_gap/weighted_f1": float(aggregates_m["weighted"]["f1"] - aggregates["weighted"]["f1"]),
            }
            monitor_combined_logs = {**monitor_scalar_logs, **monitor_gap_logs}
            wandb_log_metrics(
                step,
                per_class_m,
                aggregates_m,
                cfg.ID2LANG,
                cfg.NUM_CLASSES,
                cfg.PAD_ID,
                monitor_stats["conf_mat"],
                prefix="monitor",
                extra_logs=monitor_combined_logs,
                commit=False,
            )

            if monitor_stats.get("conf_thresh") is not None and monitor_stats.get("acc_thresh_mean") is not None:
                num_classes_with_other = cfg.NUM_CLASSES + 1
                per_class_mt, aggregates_mt = compute_metrics_from_confusion(
                    monitor_stats["conf_thresh"],
                    num_classes_with_other,
                    cfg.PAD_ID,
                )
                print_metrics_table(
                    per_class_mt,
                    aggregates_mt,
                    cfg.ID2LANG,
                    cfg.NUM_CLASSES,
                    cfg.PAD_ID,
                    title="Monitor (thresholded)",
                )
                thresh_logs = {
                    "monitor_thresh/acc": monitor_stats.get("acc_thresh_mean", 0.0) or 0.0,
                    "monitor_thresh/threshold": t_cfg.monitor_other_threshold,
                    "monitor_thresh/windows": monitor_stats["windows"],
                    "monitor_thresh/skipped": monitor_stats["skipped"],
                    "monitor_thresh/files_used": monitor_stats["files_used"],
                    "monitor_gap_thresh/micro_accuracy": float(aggregates_mt["micro"]["acc"] - aggregates["micro"]["acc"]),
                    "monitor_gap_thresh/macro_f1": float(aggregates_mt["macro"]["f1"] - aggregates["macro"]["f1"]),
                    "monitor_gap_thresh/macro_precision": float(aggregates_mt["macro"]["precision"] - aggregates["macro"]["precision"]),
                    "monitor_gap_thresh/macro_recall": float(aggregates_mt["macro"]["recall"] - aggregates["macro"]["recall"]),
                    "monitor_gap_thresh/weighted_f1": float(aggregates_mt["weighted"]["f1"] - aggregates["weighted"]["f1"]),
                }
                wandb_log_metrics(
                    step,
                    per_class_mt,
                    aggregates_mt,
                    cfg.ID2LANG,
                    cfg.NUM_CLASSES,
                    cfg.PAD_ID,
                    monitor_stats["conf_thresh"],
                    prefix="monitor_thresh",
                    extra_logs=thresh_logs,
                    commit=False,
                )

        wandb_log_metrics(
            step,
            per_class,
            aggregates,
            cfg.ID2LANG,
            cfg.NUM_CLASSES,
            cfg.PAD_ID,
            conf_mat,
            commit=True,
        )
        return val_loss, val_acc

    current_step = int(unreplicate_state(state).step) if use_pmap else int(state.step)
    if current_step == 0:
        run_val_and_monitor_eval(step=0, title="Baseline validation (step 0)")

    print("Starting training...", flush=True)
    start_time = time.time()
    last_log_time = time.time()
    metrics_history = {"loss": [], "acc": [], "step_time": [], "grad_norm": []}

    best_val = float("inf")
    non_improve_evals = 0
    pruned = False
    stopped_reason = ""
    error_reason = ""  # Track any unexpected error from the loop
    last_heartbeat = time.time()

    # Track min/max epochs across languages
    min_epochs = 0
    max_epochs = 0

    try:
        try:
            # +1 to make sure the final eval and checkpoint triggers
            for step in range(current_step, t_cfg.steps + 1):
                # Periodic garbage collection
                if step % 100 == 0:
                    gc.collect()  # Regular Python garbage collection

                elapsed_min = (time.time() - start_time) / 60.0
                if args.max_minutes > 0 and elapsed_min >= args.max_minutes:
                    stopped_reason = f"time_cap_{args.max_minutes}min"
                    print(
                        f"Time cap reached ({args.max_minutes} min). Stopping gracefully.",
                        flush=True,
                    )
                    break
                if os.path.exists(args.stop_file):
                    stopped_reason = "stop_file_detected"
                    print(
                        f"Stop file detected at '{args.stop_file}'. Stopping gracefully.",
                        flush=True,
                    )
                    break

                if _STOP["flag"]:
                    stopped_reason = f"signal_{int(_STOP['flag'])}"
                    print("Stop requested by signal. Exiting loop.", flush=True)
                    break

                step_start = time.time()
                data_time = 0.0
                compute_time = 0.0
                grad_norm_value = None
                oe_losses: list[float] = []
                oe_batches_used = 0

                rng, step_base_rng = jax.random.split(rng)

                if accum_steps == 1:
                    data_start = time.time()
                    batch_tokens, batch_labels = data_fetcher.get()
                    batch_tokens = sanitize_tokens(batch_tokens)
                    data_time = time.time() - data_start

                    step_base_rng, oe_decision_rng = jax.random.split(step_base_rng)
                    use_oe = (
                        oe_batcher is not None
                        and float(jax.random.uniform(oe_decision_rng, ()).item()) < oe_ratio
                    )
                    compute_start = time.time()
                    if use_pmap:
                        # Shard batch across devices
                        batch_tokens, batch_labels = shard_batch(
                            batch_tokens, batch_labels
                        )
                        step_rngs = make_pmap_rngs(step_base_rng)
                    if use_oe:
                        data_start = time.time()
                        outlier_tokens = oe_batcher.get()
                        outlier_tokens = sanitize_tokens(outlier_tokens)
                        data_time += time.time() - data_start
                        if use_pmap:
                            (outlier_tokens,) = shard_batch(outlier_tokens)
                            state, loss, acc, oe_loss = train_step_oe_fn(
                                state,
                                batch_tokens,
                                batch_labels,
                                outlier_tokens,
                                oe_lambda,
                                step_rngs,
                            )
                            loss = float(loss[0])
                            acc = float(acc[0])
                            oe_loss = float(oe_loss[0])
                        else:
                            state, loss, acc, oe_loss = train_step_oe_fn(
                                state,
                                batch_tokens,
                                batch_labels,
                                outlier_tokens,
                                oe_lambda,
                                step_base_rng,
                            )
                        oe_losses.append(float(oe_loss))
                        oe_batches_used += 1
                        del outlier_tokens
                    else:
                        if use_pmap:
                            state, loss, acc = train_step_fn(
                                state, batch_tokens, batch_labels, step_rngs
                            )
                            loss = float(loss[0])
                            acc = float(acc[0])
                        else:
                            state, loss, acc = train_step_fn(
                                state, batch_tokens, batch_labels, step_base_rng
                            )
                    loss_value = float(loss)
                    acc_value = float(acc)
                    compute_time = time.time() - compute_start
                    loss = loss_value
                    acc = acc_value

                    del batch_tokens
                    del batch_labels
                else:
                    losses = []
                    accs = []
                    grad_accum = None
                    micro_rng = step_base_rng

                    for _ in range(accum_steps):
                        data_start = time.time()
                        mb_tokens, mb_labels = data_fetcher.get()
                        mb_tokens = sanitize_tokens(mb_tokens)
                        data_time += time.time() - data_start

                        micro_rng, oe_decision_rng = jax.random.split(micro_rng)
                        use_oe = (
                            oe_batcher is not None
                            and float(jax.random.uniform(oe_decision_rng, ()).item()) < oe_ratio
                        )
                        micro_rng, use_rng = jax.random.split(micro_rng)
                        compute_start = time.time()
                        if use_pmap:
                            mb_tokens_s, mb_labels_s = shard_batch(mb_tokens, mb_labels)
                            use_rngs = make_pmap_rngs(use_rng)
                        if use_oe:
                            data_start = time.time()
                            outlier_tokens = oe_batcher.get()
                            outlier_tokens = sanitize_tokens(outlier_tokens)
                            data_time += time.time() - data_start
                            if use_pmap:
                                (outlier_tokens_s,) = shard_batch(outlier_tokens)
                                grads, micro_loss, micro_acc, micro_oe_loss = micro_step_oe_fn(
                                    state,
                                    mb_tokens_s,
                                    mb_labels_s,
                                    outlier_tokens_s,
                                    oe_lambda,
                                    use_rngs,
                                )
                                micro_loss = float(micro_loss[0])
                                micro_acc = float(micro_acc[0])
                                micro_oe_loss = float(micro_oe_loss[0])
                            else:
                                grads, micro_loss, micro_acc, micro_oe_loss = micro_step_oe_fn(
                                    state,
                                    mb_tokens,
                                    mb_labels,
                                    outlier_tokens,
                                    oe_lambda,
                                    use_rng,
                                )
                            oe_losses.append(float(micro_oe_loss))
                            oe_batches_used += 1
                            del outlier_tokens
                        else:
                            if use_pmap:
                                grads, micro_loss, micro_acc = micro_step_fn(
                                    state, mb_tokens_s, mb_labels_s, use_rngs
                                )
                                micro_loss = float(micro_loss[0])
                                micro_acc = float(micro_acc[0])
                            else:
                                grads, micro_loss, micro_acc = micro_step_fn(
                                    state, mb_tokens, mb_labels, use_rng
                                )
                        micro_loss_value = float(micro_loss)
                        micro_acc_value = float(micro_acc)
                        compute_time += time.time() - compute_start

                        if grad_accum is None:
                            grad_accum = grads
                        else:
                            grad_accum = jtu.tree_map(
                                lambda a, b: a + b, grad_accum, grads
                            )

                        losses.append(micro_loss_value)
                        accs.append(micro_acc_value)

                        del mb_tokens
                        del mb_labels

                    scale = jnp.asarray(accum_steps, dtype=jnp.float32)
                    grad_accum = jtu.tree_map(lambda g: g / scale, grad_accum)
                    if use_pmap:
                        # Grads are already pmean-ed per device; take device 0 for norm
                        grad_norm_value = float(grad_global_norm(
                            jax.tree.map(lambda x: x[0], grad_accum)
                        ))
                        # Apply gradients on the replicated state
                        state = state.apply_gradients(grads=grad_accum)
                    else:
                        grad_norm_value = float(grad_global_norm(grad_accum))
                        state = state.apply_gradients(grads=grad_accum)
                    apply_start = time.time()
                    compute_time += time.time() - apply_start
                    loss = float(sum(losses) / len(losses))
                    acc = float(sum(accs) / len(accs))

                oe_loss_value = float(sum(oe_losses) / len(oe_losses)) if oe_losses else 0.0
                step_time = time.time() - step_start

                # LR best-effort
                try:
                    if hasattr(state.opt_state[1], "hyperparams"):
                        step_lr = float(state.opt_state[1].hyperparams["learning_rate"])
                    elif len(state.opt_state) > 2 and hasattr(
                        state.opt_state[2], "count"
                    ):
                        step_lr = float(
                            t_cfg.lr * min(1.0, state.opt_state[2].count / t_cfg.warmup)
                        )
                    else:
                        step_lr = t_cfg.lr
                except Exception:
                    step_lr = t_cfg.lr

                # Get current epochs from the batcher
                epochs_by_lang = data_fetcher.get_epochs()
                if epochs_by_lang:
                    min_epochs = float(min(epochs_by_lang.values()))
                    max_epochs = float(max(epochs_by_lang.values()))
                else:
                    min_epochs = max_epochs = 0.0

                metrics = {
                    "train/loss": float(loss),
                    "train/acc": float(acc),
                    "train/learning_rate": step_lr,
                    "train/min_epochs": min_epochs,
                    "train/max_epochs": max_epochs,
                    "perf/data_time": data_time,
                    "perf/compute_time": compute_time,
                    "perf/total_step_time": step_time,
                }
                if oe_lambda > 0.0:
                    metrics.update(
                        {
                            "oe/loss_uniform": float(oe_loss_value),
                            "oe/active_batches": int(oe_batches_used),
                            "oe/lambda": float(oe_lambda),
                            "oe/ratio": float(oe_ratio),
                        }
                    )

                if grad_norm_value is not None:
                    rs = _compute_running_stats(
                        metrics_history["grad_norm"], grad_norm_value
                    )
                    metrics.update(
                        {
                            "train/grad_norm": grad_norm_value,
                            "train/grad_norm_mean": rs["mean"],
                            "train/grad_norm_std": rs["std"],
                            "train/grad_norm_trend": rs["trend"],
                        }
                    )

                rs = _compute_running_stats(metrics_history["loss"], float(loss))
                metrics.update(
                    {
                        "train/loss_mean": rs["mean"],
                        "train/loss_std": rs["std"],
                        "train/loss_trend": rs["trend"],
                    }
                )
                rs = _compute_running_stats(metrics_history["acc"], float(acc))
                metrics.update(
                    {
                        "train/acc_mean": rs["mean"],
                        "train/acc_std": rs["std"],
                        "train/acc_trend": rs["trend"],
                    }
                )
                rs = _compute_running_stats(metrics_history["step_time"], step_time)
                metrics.update(
                    {"perf/step_time_mean": rs["mean"], "perf/step_time_std": rs["std"]}
                )

                # Determine if we should commit the logs now or wait
                should_commit = (
                    step % t_cfg.log_every == 0 and step % t_cfg.eval_every != 0
                )

                _wandb_safe_log(metrics, step=step, commit=should_commit)
                last_heartbeat = time.time()

                if step % t_cfg.log_every == 0:
                    elapsed = time.time() - last_log_time
                    sps = t_cfg.log_every / elapsed if elapsed > 0 else 0
                    print(
                        f"Step {step}/{t_cfg.steps} [Epochs {min_epochs:.3f}-{max_epochs:.3f}] | "
                        f"Loss: {loss:.4f} (±{metrics['train/loss_std']:.4f}), "
                        f"Acc: {acc:.4f} (±{metrics['train/acc_std']:.4f}), SPS: {sps:.2f}",
                        flush=True,
                    )
                    last_log_time = time.time()

                # Watchdog
                if time.time() - last_heartbeat > 600:
                    stopped_reason = "watchdog_no_progress_10min"
                    print(
                        "Watchdog: no progress for 10 minutes. Stopping.",
                        flush=True,
                    )
                    break

                if step > 0 and step % t_cfg.eval_every == 0:
                    print("Evaluating...", flush=True)
                    rng, eval_rng = jax.random.split(rng)

                    # >>> NEW: eval with confusion matrix + metrics <<<
                    # Eval runs single-device; unreplicate if using pmap
                    eval_state = unreplicate_state(state) if use_pmap else state
                    val_loss, val_acc, conf_mat = evaluate_split_with_metrics(
                        state=eval_state,
                        dsets_by_lang=dsets["val"],
                        L=d_cfg.window_max_bytes,
                        batch_size=d_cfg.batch_size,
                        batches=t_cfg.eval_batches,
                        data_cfg=d_cfg,
                        rng=eval_rng,
                        eval_step_fn=eval_step,
                    )

                    # Compute per-class and aggregates
                    per_class, aggregates = compute_metrics_from_confusion(
                        conf_mat, cfg.NUM_CLASSES, cfg.PAD_ID
                    )

                    print(
                        f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}",
                        flush=True,
                    )
                    # Existing scalar gap logs
                    val_metrics = {
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "val/train_gap": float(loss) - val_loss,
                        "val/acc_gap": float(acc) - val_acc,
                    }
                    # Do not commit yet; eval may also log monitor metrics.
                    _wandb_safe_log(val_metrics, step=step, commit=False)

                    # Print table (console) and defer W&B commit until after optional monitor eval
                    print_metrics_table(
                        per_class, aggregates, cfg.ID2LANG, cfg.NUM_CLASSES, cfg.PAD_ID
                    )

                    if monitor_data is not None and t_cfg.monitor_eval_every > 0:
                        print("Running monitor evaluation...", flush=True)
                        eval_rng, monitor_rng = jax.random.split(eval_rng)
                        limit = None if t_cfg.monitor_eval_limit < 0 else int(t_cfg.monitor_eval_limit)
                        monitor_stats = evaluate_monitor_set(
                            state=eval_state,
                            monitor_data=monitor_data,
                            L=d_cfg.window_max_bytes,
                            batch_size=d_cfg.batch_size,
                            rng=monitor_rng,
                            limit=limit,
                            eval_step_fn=eval_step,
                            other_threshold=t_cfg.monitor_other_threshold,
                        )
                        # Base monitor confusion: only trained classes.
                        per_class_m, aggregates_m = compute_metrics_from_confusion(
                            monitor_stats["conf_mat"], cfg.NUM_CLASSES, cfg.PAD_ID
                        )
                        print_metrics_table(
                            per_class_m,
                            aggregates_m,
                            cfg.ID2LANG,
                            cfg.NUM_CLASSES,
                            cfg.PAD_ID,
                            title="Monitor",
                        )
                        monitor_scalar_logs = {
                            "monitor/loss": monitor_stats["loss_mean"],
                            "monitor/acc": monitor_stats["acc_mean"],
                            "monitor/windows": monitor_stats["windows"],
                            "monitor/skipped": monitor_stats["skipped"],
                            "monitor/files_used": monitor_stats["files_used"],
                            "monitor/threshold": t_cfg.monitor_other_threshold,
                        }
                        # Gap metrics (monitor - val) for quick drift detection
                        monitor_gap_logs = {
                            "monitor_gap/micro_accuracy": float(aggregates_m["micro"]["acc"] - aggregates["micro"]["acc"]),
                            "monitor_gap/macro_f1": float(aggregates_m["macro"]["f1"] - aggregates["macro"]["f1"]),
                            "monitor_gap/macro_precision": float(aggregates_m["macro"]["precision"] - aggregates["macro"]["precision"]),
                            "monitor_gap/macro_recall": float(aggregates_m["macro"]["recall"] - aggregates["macro"]["recall"]),
                            "monitor_gap/weighted_f1": float(aggregates_m["weighted"]["f1"] - aggregates["weighted"]["f1"]),
                        }
                        monitor_combined_logs = {**monitor_scalar_logs, **monitor_gap_logs}
                        wandb_log_metrics(
                            step,
                            per_class_m,
                            aggregates_m,
                            cfg.ID2LANG,
                            cfg.NUM_CLASSES,
                            cfg.PAD_ID,
                            monitor_stats["conf_mat"],
                            prefix="monitor",
                            extra_logs=monitor_combined_logs,
                            commit=False,
                        )
                        if monitor_stats.get("conf_thresh") is not None and monitor_stats.get("acc_thresh_mean") is not None:
                            # Thresholded confusion includes a derived "other" bucket
                            # at index cfg.OTHER_CLASS_INDEX.
                            num_classes_with_other = cfg.NUM_CLASSES + 1
                            per_class_mt, aggregates_mt = compute_metrics_from_confusion(
                                monitor_stats["conf_thresh"],
                                num_classes_with_other,
                                cfg.PAD_ID,
                            )
                            print_metrics_table(
                                per_class_mt,
                                aggregates_mt,
                                cfg.ID2LANG,
                                cfg.NUM_CLASSES,
                                cfg.PAD_ID,
                                title="Monitor (thresholded)",
                            )
                            thresh_logs = {
                                "monitor_thresh/acc": monitor_stats.get("acc_thresh_mean", 0.0) or 0.0,
                                "monitor_thresh/threshold": t_cfg.monitor_other_threshold,
                                "monitor_thresh/windows": monitor_stats["windows"],
                                "monitor_thresh/skipped": monitor_stats["skipped"],
                                "monitor_thresh/files_used": monitor_stats["files_used"],
                                "monitor_gap_thresh/micro_accuracy": float(aggregates_mt["micro"]["acc"] - aggregates["micro"]["acc"]),
                                "monitor_gap_thresh/macro_f1": float(aggregates_mt["macro"]["f1"] - aggregates["macro"]["f1"]),
                                "monitor_gap_thresh/macro_precision": float(aggregates_mt["macro"]["precision"] - aggregates["macro"]["precision"]),
                                "monitor_gap_thresh/macro_recall": float(aggregates_mt["macro"]["recall"] - aggregates["macro"]["recall"]),
                                "monitor_gap_thresh/weighted_f1": float(aggregates_mt["weighted"]["f1"] - aggregates["weighted"]["f1"]),
                            }
                            wandb_log_metrics(
                                step,
                                per_class_mt,
                                aggregates_mt,
                                cfg.ID2LANG,
                                cfg.NUM_CLASSES,
                                cfg.PAD_ID,
                                monitor_stats["conf_thresh"],
                                prefix="monitor_thresh",
                                extra_logs=thresh_logs,
                                commit=False,
                            )

                    # Finally, log val confusion + metrics and commit the step atomically
                    wandb_log_metrics(
                        step,
                        per_class,
                        aggregates,
                        cfg.ID2LANG,
                        cfg.NUM_CLASSES,
                        cfg.PAD_ID,
                        conf_mat,
                        commit=True,
                    )

                    # Save checkpoints — unreplicate if using pmap
                    ckpt_dir, ckpt_prefix, ckpt_blob = resolve_ckpt_paths(
                        t_cfg.ckpt_path
                    )
                    save_state = unreplicate_state(state) if use_pmap else state
                    checkpoints.save_checkpoint(
                        ckpt_dir,
                        save_state,
                        step=step,
                        prefix=ckpt_prefix,
                        keep=2,
                        overwrite=True,
                        async_manager=ckpt_async_manager,
                    )

                    try:
                        os.makedirs(ckpt_dir, exist_ok=True)
                        with open(ckpt_blob, "wb") as f:
                            f.write(serialization.to_bytes(save_state.params))
                        base = os.path.basename(ckpt_blob)
                        stem, ext = os.path.splitext(base)
                        hist_path = os.path.join(ckpt_dir, f"{stem}-{step}{ext}")
                        with open(hist_path, "wb") as f:
                            f.write(serialization.to_bytes(save_state.params))
                    except Exception as e:
                        print(
                            f"WARNING: writing raw params msgpack failed: {e}",
                            flush=True,
                        )

                    # Simple pruning (unchanged)
                    elapsed_min = (time.time() - start_time) / 60.0
                    if elapsed_min >= args.prune_min_minutes:
                        if (
                            val_loss
                            + args.prune_delta
                            * (best_val if best_val < float("inf") else val_loss)
                            < best_val
                        ):
                            best_val = val_loss
                            non_improve_evals = 0
                        else:
                            non_improve_evals += 1
                            if non_improve_evals >= args.prune_patience_evals:
                                pruned = True
                                stopped_reason = (
                                    f"pruned_no_improve_{args.prune_patience_evals}"
                                    f"evals_delta{args.prune_delta}"
                                )
                                print(
                                    f"[PRUNE] {stopped_reason}. Stopping run.",
                                    flush=True,
                                )
                                break

        except KeyboardInterrupt:
            stopped_reason = "keyboard_interrupt"
            print("Interrupted. Saving checkpoint and finishing.", flush=True)
        except Exception as e:
            # Catch any other exception (like CUDA OOM)
            error_reason = str(e)
            print(
                f"\nFATAL ERROR in training loop: {type(e).__name__}: {e}",
                flush=True,
            )
            # This error will be logged in the finally block
    finally:
        try:
            data_fetcher.close()
        except Exception:
            pass

        final_reason = "completed"
        if error_reason:
            final_reason = f"error_{error_reason.__class__.__name__}"
            _wandb_safe_log(
                {"meta/error_message": error_reason},
                step=int(getattr(state, "step", 0)),
                commit=False,
            )
        elif stopped_reason:
            final_reason = stopped_reason

        if ckpt_async_manager is not None:
            try:
                ckpt_async_manager.wait_previous_save()
            except Exception:
                pass

        _wandb_safe_log(
            {
                "meta/stopped_reason": final_reason,
                "meta/pruned": int(pruned),
                "meta/runtime_minutes": (time.time() - start_time) / 60.0,
            },
            step=int(getattr(state, "step", 0)),
        )
        wandb.finish()
        print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
