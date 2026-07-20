import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    eval_step_with_logits,
    TrainState,
    count_params,
    train_step_no_jit,
    train_step_with_oe_no_jit,
    microbatch_grad_step,
    microbatch_grad_step_with_oe,
    microbatch_grad_step_no_jit,
    microbatch_grad_step_with_oe_no_jit,
    grad_global_norm,
    checkpoint_params_subtree,
    merge_compatible_state,
    seed_missing_auxiliary_heads_from_main,
    # Multi-GPU (pmap) variants
    replicate_state,
    unreplicate_state,
    p_train_step,
    p_train_step_with_oe,
    p_microbatch_grad_step,
    p_microbatch_grad_step_with_oe,
    p_eval_step,
    p_apply_gradients,
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
from active_learning.persistent_control import (
    encode_trainer_event,
    parse_trainer_command_line,
)

if TYPE_CHECKING:
    from utils.config import DataConfig, TrainConfig

import signal

# global-ish flag that both the handler and loop can see
_STOP = {"flag": False}


def _signal_handler(sig, frame):
    _STOP["flag"] = True
    print(f"Signal {sig} received; stopping...", flush=True)


def _checkpoint_embedding_rows(path_value: str) -> int:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file() or path.suffix != ".msgpack":
        return int(cfg.NUM_TOKEN_EMBEDDINGS)
    try:
        restored = serialization.msgpack_restore(path.read_bytes())
        params = checkpoint_params_subtree(restored)
        return int(np.asarray(params["Embed_0"]["embedding"]).shape[0])
    except Exception as exc:
        print(f"Checkpoint vocabulary inference failed for {path}: {exc}", flush=True)
        return int(cfg.NUM_TOKEN_EMBEDDINGS)


# register early, before long inits
if mp.current_process().name == "MainProcess":
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


def _wandb_current_step(default: int = 0) -> int:
    try:
        run = getattr(wandb, "run", None)
        if run is None:
            return int(default)
        return max(int(default), int(getattr(run, "step", default) or 0))
    except Exception:
        return int(default)


def _state_step(state, *, use_pmap: bool) -> int:
    if use_pmap:
        return int(unreplicate_state(state).step)
    return int(getattr(state, "step", 0))


def _emit_persistent_trainer_event(event: str, **payload):
    print(encode_trainer_event(event, **payload), flush=True)


def _read_persistent_trainer_command() -> dict:
    while True:
        line = sys.stdin.readline()
        if line == "":
            return {"command": "shutdown", "reason": "stdin_eof"}
        stripped = str(line).strip()
        if not stripped:
            continue
        return parse_trainer_command_line(stripped)


def resolve_ckpt_paths(path: str):
    blob_abs = os.path.abspath(path)
    ckpt_dir_abs = os.path.dirname(blob_abs) or os.getcwd()
    base = os.path.basename(blob_abs)
    prefix = base + "-"
    return ckpt_dir_abs, prefix, blob_abs


def _save_training_checkpoint(
    save_state,
    ckpt_path: str,
    step: int,
    *,
    async_manager=None,
):
    ckpt_dir, ckpt_prefix, ckpt_blob = resolve_ckpt_paths(ckpt_path)
    checkpoints.save_checkpoint(
        ckpt_dir,
        save_state,
        step=step,
        prefix=ckpt_prefix,
        keep=2,
        overwrite=True,
        async_manager=async_manager,
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    with open(ckpt_blob, "wb") as f:
        f.write(serialization.to_bytes(save_state.params))
    base = os.path.basename(ckpt_blob)
    stem, ext = os.path.splitext(base)
    hist_path = os.path.join(ckpt_dir, f"{stem}-{step}{ext}")
    with open(hist_path, "wb") as f:
        f.write(serialization.to_bytes(save_state.params))


def _find_local_wandb_config(repo_root: Path, run_id: str) -> Path | None:
    wandb_root = repo_root / "wandb"
    if not wandb_root.exists():
        return None
    candidates = sorted(wandb_root.glob(f"run-*-{run_id}/files/config.yaml"))
    return candidates[-1] if candidates else None


def _find_local_wandb_summary(repo_root: Path, run_id: str) -> Path | None:
    wandb_root = repo_root / "wandb"
    if not wandb_root.exists():
        return None
    candidates = sorted(wandb_root.glob(f"run-*-{run_id}/files/wandb-summary.json"))
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


def _read_local_wandb_summary(summary_path: Path) -> Dict[str, object] | None:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _collect_monitor_per_class_f1(
    summary: Dict[str, object],
    *,
    prefix: str,
) -> list[tuple[float, str, str]]:
    rows: list[tuple[float, str, str]] = []
    for key, raw_value in summary.items():
        if not key.startswith(prefix) or not key.endswith("/f1"):
            continue
        try:
            score = float(raw_value)
        except Exception:
            continue
        if not np.isfinite(score):
            continue
        class_token = key[len(prefix):-len("/f1")]
        label = class_token.split("_", 1)[1] if "_" in class_token else class_token
        label = str(label).strip().lower()
        if not label or label == "other":
            continue
        rows.append((score, label, key))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    return rows


def _select_weakest_monitor_label(
    summary: Dict[str, object],
) -> tuple[str, str, float] | None:
    for prefix in ("monitor_thresh/per_class/", "monitor/per_class/"):
        rows = _collect_monitor_per_class_f1(summary, prefix=prefix)
        if rows:
            score, label, key = rows[0]
            return label, key, score
    return None


def _resolve_fine_tune_dense_bias(
    repo_root: Path,
    args,
) -> tuple[str, str, float, str]:
    prob = float(max(0.0, min(1.0, getattr(args, "fine_tune_dense_bias_prob", 0.0))))
    raw_label = str(getattr(args, "fine_tune_dense_bias_label", "") or "").strip().lower()
    if prob <= 0.0:
        return "", "", float("nan"), ""

    if raw_label and raw_label != "auto":
        print(
            f"Dense bias label fixed by CLI: label={raw_label} prob={prob:.3f}",
            flush=True,
        )
        return raw_label, "", float("nan"), ""

    candidate_run_ids: list[str] = []
    for candidate in (
        getattr(args, "continue_run_id", ""),
        getattr(args, "wandb_run_id", ""),
        getattr(args, "fine_tune_dense_bias_source_run_id", ""),
        getattr(args, "fine_tune_run_id", ""),
    ):
        run_id = str(candidate or "").strip()
        if run_id and run_id not in candidate_run_ids:
            candidate_run_ids.append(run_id)

    for run_id in candidate_run_ids:
        summary_path = _find_local_wandb_summary(repo_root, run_id)
        if summary_path is None:
            continue
        summary = _read_local_wandb_summary(summary_path)
        if not summary:
            continue
        selected = _select_weakest_monitor_label(summary)
        if selected is None:
            continue
        label, key, score = selected
        metric_prefix = key.rsplit("/", 2)[0] + "/"
        weakest = _collect_monitor_per_class_f1(summary, prefix=metric_prefix)[:3]
        weakest_preview = ", ".join(
            f"{lbl}={val:.4f}" for val, lbl, _ in weakest
        )
        print(
            "Auto dense bias selection: "
            f"run={run_id} summary={summary_path} "
            f"metric={key} value={score:.4f} "
            f"chosen_label={label} prob={prob:.3f}"
            + (f" weakest=[{weakest_preview}]" if weakest_preview else ""),
            flush=True,
        )
        return label, key, score, run_id

    print(
        "Auto dense bias disabled: no local W&B summary with per-class monitor F1 "
        f"was found for candidate runs {candidate_run_ids}.",
        flush=True,
    )
    return "", "", float("nan"), ""


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
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument(
        "--full-files",
        action="store_true",
        help="Use fixed-length full-file/long-sample mode instead of 1536-byte windows.",
    )
    parser.add_argument(
        "--full-file-max-bytes",
        type=int,
        default=10000,
        help="Target padded sequence length used when --full-files is enabled.",
    )
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
        "--pure_prob",
        type=float,
        default=None,
        help="Override probability of pure (single-language) windows (default from DataConfig: 0.65).",
    )
    parser.add_argument(
        "--mix_prob",
        type=float,
        default=None,
        help="Override probability of mixed (concatenation) windows (default from DataConfig: 0.15).",
    )
    parser.add_argument(
        "--line_inject_prob",
        type=float,
        default=None,
        help="Override probability of line-injection windows (default from DataConfig: 0.15).",
    )
    parser.add_argument(
        "--markdown_prob",
        type=float,
        default=None,
        help="Override probability of synthetic markdown wrapping windows (default from DataConfig: 0.1).",
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
    parser.add_argument(
        "--schedule_steps",
        type=int,
        default=None,
        help="Optional learning-rate decay horizon. Defaults to --steps when omitted.",
    )
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
    parser.add_argument(
        "--compute_dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Model compute dtype; float32 is a compatibility fallback for long Mamba training on some GPUs.",
    )
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
        "--eval_on_start",
        action="store_true",
        help="Run validation and monitor evaluation immediately after restore, before training steps.",
    )
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
        "--monitor_eval_batch_size",
        type=int,
        default=0,
        help="Batch size for monitor eval only (0 = reuse train batch size).",
    )
    parser.add_argument(
        "--monitor_eval_deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a fixed-seed monitor eval subset and fixed window sampling on every monitor run.",
    )
    parser.add_argument(
        "--monitor_eval_seed",
        type=int,
        default=123,
        help="Seed used when --monitor_eval_deterministic is enabled.",
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
        "--wandb-run-id",
        type=str,
        default="",
        help="Use a specific W&B run id without restoring optimizer/checkpoint state.",
    )
    parser.add_argument(
        "--persistent-trainer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the trainer process alive after startup and accept chunk commands over stdin. "
            "Used by the active-learning meta-trainer to avoid recompiling every round."
        ),
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
        "--fine_tune_augment_monitor",
        action="store_true",
        default=False,
        help=(
            "Apply training-style augmentations during monitor fine-tuning. "
            "When enabled, augmentation fragments are drawn from monitor segments and, if available, "
            "the active-learning SQLite store."
        ),
    )
    parser.add_argument(
        "--fine_tune_carrier_balanced",
        action="store_true",
        default=False,
        help=(
            "Sample carrier families uniformly when the fine-tune memmap provides "
            "a carrierByFile entry in meta.json."
        ),
    )
    parser.add_argument(
        "--fine_tune_boundary_sample_prob",
        type=float,
        default=0.0,
        help="Probability that a U-Net fine-tune window is centered around a labeled transition.",
    )
    parser.add_argument(
        "--fine_tune_boundary_margin",
        type=int,
        default=64,
        help="Minimum preferred context on each side of a sampled transition.",
    )
    parser.add_argument(
        "--fine_tune_boundary_require_transition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When boundary sampling is selected, retry file selection until a "
            "file with a genuine labeled transition is found."
        ),
    )
    parser.add_argument(
        "--fine_tune_boundary_pair_balanced",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Choose a distinct adjacent label pair before choosing one of its "
            "boundary occurrences, preventing repeated pairs from dominating."
        ),
    )
    parser.add_argument(
        "--boundary_loss_weight",
        type=float,
        default=0.0,
        help="Additional cross-entropy weight within --boundary_loss_radius of a true transition.",
    )
    parser.add_argument(
        "--boundary_loss_radius",
        type=int,
        default=4,
        help="Radius in model byte tokens for optional boundary-weighted loss.",
    )
    parser.add_argument(
        "--aux_neighbor_loss_weight",
        type=float,
        default=1.0,
        help="Weight for auxiliary neighboring-label heads; set to zero for legacy checkpoints.",
    )
    parser.add_argument(
        "--fine_tune_dense_bias_label",
        type=str,
        default="",
        help=(
            "Optional monitor label to sample slightly more often from the dense fine-tune set. "
            "Use 'auto' or leave empty to pick the weakest class from a local W&B summary."
        ),
    )
    parser.add_argument(
        "--fine_tune_dense_bias_prob",
        type=float,
        default=0.0,
        help=(
            "Probability of drawing a dense fine-tune monitor file from "
            "--fine_tune_dense_bias_label instead of uniform sampling."
        ),
    )
    parser.add_argument(
        "--fine_tune_dense_bias_source_run_id",
        type=str,
        default="",
        help=(
            "Optional fallback W&B run id whose latest local summary should be used when "
            "auto-selecting the dense-bias label."
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

    dense_bias_label_resolved, dense_bias_metric_key, dense_bias_metric_value, dense_bias_metric_run_id = (
        _resolve_fine_tune_dense_bias(repo_root, args)
    )
    args.fine_tune_dense_bias_label = dense_bias_label_resolved

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
    d_cfg.fine_tune_carrier_balanced = bool(args.fine_tune_carrier_balanced)
    d_cfg.fine_tune_boundary_sample_prob = float(
        max(0.0, min(1.0, args.fine_tune_boundary_sample_prob))
    )
    d_cfg.fine_tune_boundary_margin = max(0, int(args.fine_tune_boundary_margin))
    d_cfg.fine_tune_boundary_require_transition = bool(
        args.fine_tune_boundary_require_transition
    )
    d_cfg.fine_tune_boundary_pair_balanced = bool(
        args.fine_tune_boundary_pair_balanced
    )
    cfg.BOUNDARY_LOSS_WEIGHT = float(max(0.0, args.boundary_loss_weight))
    cfg.BOUNDARY_LOSS_RADIUS = max(0, int(args.boundary_loss_radius))
    cfg.AUX_NEIGHBOR_LOSS_WEIGHT = float(max(0.0, args.aux_neighbor_loss_weight))
    # Override mode probabilities when provided via CLI (for ablation studies).
    if args.pure_prob is not None:
        d_cfg.pure_prob = max(0.0, float(args.pure_prob))
    if args.mix_prob is not None:
        d_cfg.mix_prob = max(0.0, float(args.mix_prob))
    if args.line_inject_prob is not None:
        d_cfg.line_inject_prob = max(0.0, float(args.line_inject_prob))
    if args.markdown_prob is not None:
        d_cfg.markdown_prob = max(0.0, float(args.markdown_prob))
    if bool(args.full_files):
        full_len = max(1, int(args.full_file_max_bytes))
        d_cfg.window_min_bytes = full_len
        d_cfg.window_max_bytes = full_len
        d_cfg.bucket_step = full_len
    arch_name = str(args.arch).lower().strip()
    if arch_name == "unet1d":
        required_window = int(cfg.MODEL_WINDOW_BYTES)
        if int(d_cfg.window_min_bytes) != required_window or int(d_cfg.window_max_bytes) != required_window:
            raise ValueError(
                "U-Net runs must use the fixed 1536-byte window size. "
                f"Received window_min_bytes={int(d_cfg.window_min_bytes)} and "
                f"window_max_bytes={int(d_cfg.window_max_bytes)}. "
                "Disable full-file mode and keep the canonical U-Net window."
            )
    if args.language_pair_prob is not None:
        prob = max(0.0, min(1.0, float(args.language_pair_prob)))
        d_cfg.language_pair_mode_prob = prob
    monitor_every = 0 if args.monitor_eval_every == 0 else args.eval_every

    # For fine-tuning, always use the dedicated monitor_preprocessed_b memmap
    # as the monitor/validation set, regardless of any custom monitor_eval_root.
    if args.fine_tune:
        args.monitor_eval_root = args.fine_tune_val_root

    checkpoint_vocab_size = (
        _checkpoint_embedding_rows(args.fine_tune_ckpt_path)
        if args.fine_tune and args.fine_tune_ckpt_path
        else int(cfg.NUM_TOKEN_EMBEDDINGS)
    )
    if checkpoint_vocab_size != int(cfg.NUM_TOKEN_EMBEDDINGS):
        print(
            f"Fine-tune checkpoint uses compact token vocabulary: {checkpoint_vocab_size} rows.",
            flush=True,
        )
    t_cfg = cfg.TrainConfig(
        steps=args.steps,
        schedule_steps=args.schedule_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        accum_steps=args.accum_steps,
        arch=args.arch,
        model_dim=args.model_dim,
        channels=tuple(map(int, args.channels.split(","))),
        dropout_rate=args.dropout_rate,
        dtype=jnp.float32 if args.compute_dtype == "float32" else jnp.bfloat16,
        num_token_embeddings=checkpoint_vocab_size,
        boundary_loss_weight=float(cfg.BOUNDARY_LOSS_WEIGHT),
        boundary_loss_radius=int(cfg.BOUNDARY_LOSS_RADIUS),
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
        monitor_eval_batch_size=int(args.monitor_eval_batch_size),
        monitor_eval_root=args.monitor_eval_root,
        monitor_eval_deterministic=bool(args.monitor_eval_deterministic),
        monitor_eval_seed=int(args.monitor_eval_seed),
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
    t_cfg.MODEL_WINDOW_BYTES = int(d_cfg.window_max_bytes)
    t_cfg.full_files = bool(args.full_files)
    t_cfg.full_file_max_bytes = int(args.full_file_max_bytes)

    # Dedicated monitor fine-tuning is self-contained and does not require the
    # original Arrow corpus to be available on the adaptation machine.
    if t_cfg.fine_tune:
        print("Fine-tune mode: skipping unrelated base Arrow datasets.", flush=True)
        dsets = {"train": {}, "val": {}}
        train_dsets = {}
    else:
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
                monitor_bs = int(t_cfg.monitor_eval_batch_size) if int(t_cfg.monitor_eval_batch_size) > 0 else int(d_cfg.batch_size)
                print(
                    f"Monitor eval batch size: {monitor_bs}",
                    flush=True,
                )
                if bool(t_cfg.monitor_eval_deterministic):
                    limit_desc = "all files" if int(t_cfg.monitor_eval_limit) < 0 else f"{int(t_cfg.monitor_eval_limit)} files"
                    print(
                        f"Monitor eval deterministic mode: fixed {limit_desc} with seed {int(t_cfg.monitor_eval_seed)}.",
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
                "resume": "must",
            }
        )
        print(f"Resuming W&B run: {args.continue_run_id}", flush=True)
    elif getattr(args, "wandb_run_id", ""):
        wandb_init_kwargs.update(
            {
                "id": args.wandb_run_id,
                "resume": "never",
            }
        )
        print(f"Using explicit W&B run id: {args.wandb_run_id}", flush=True)
    wandb.init(**wandb_init_kwargs)
    wandb.config.update({**d_cfg.__dict__, **t_cfg.__dict__}, allow_val_change=True)
    wandb.config.update(
        {
            "active_learning_store": args.active_learning_store,
            "active_learning_mix_prob": float(args.active_learning_mix_prob),
            "active_learning_max_windows": int(args.active_learning_max_windows),
            "full_files": bool(args.full_files),
            "full_file_max_bytes": int(args.full_file_max_bytes),
            "fine_tune_augment_monitor": bool(args.fine_tune_augment_monitor),
            "fine_tune_carrier_balanced": bool(args.fine_tune_carrier_balanced),
            "fine_tune_boundary_sample_prob": float(d_cfg.fine_tune_boundary_sample_prob),
            "fine_tune_boundary_margin": int(d_cfg.fine_tune_boundary_margin),
            "fine_tune_boundary_require_transition": bool(
                d_cfg.fine_tune_boundary_require_transition
            ),
            "fine_tune_boundary_pair_balanced": bool(
                d_cfg.fine_tune_boundary_pair_balanced
            ),
            "boundary_loss_weight": float(cfg.BOUNDARY_LOSS_WEIGHT),
            "boundary_loss_radius": int(cfg.BOUNDARY_LOSS_RADIUS),
            "aux_neighbor_loss_weight": float(cfg.AUX_NEIGHBOR_LOSS_WEIGHT),
            "fine_tune_dense_bias_label": str(args.fine_tune_dense_bias_label or ""),
            "fine_tune_dense_bias_prob": float(args.fine_tune_dense_bias_prob),
            "fine_tune_dense_bias_source_run_id": str(args.fine_tune_dense_bias_source_run_id or ""),
            "fine_tune_dense_bias_metric_key": str(dense_bias_metric_key or ""),
            "fine_tune_dense_bias_metric_run_id": str(dense_bias_metric_run_id or ""),
            "fine_tune_dense_bias_metric_value": (
                float(dense_bias_metric_value)
                if np.isfinite(dense_bias_metric_value)
                else None
            ),
            "monitor_train_files": int(fine_tune_train_files_count),
            "monitor_eval_files": int(monitor_eval_files_count),
            "num_gpus": int(num_devices),
            "data_parallel": use_pmap,
        },
        allow_val_change=True,
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

    def _format_restore_keys(keys, limit=4):
        shown = ["/".join(map(str, key)) for key in keys[:limit]]
        if len(keys) > limit:
            shown.append("...")
        return ", ".join(shown)

    def _log_restore_stats(label: str, path_desc: str, stats: dict):
        loaded = len(stats.get("loaded", ()))
        missing = tuple(stats.get("missing", ()))
        mismatched = tuple(stats.get("mismatched", ()))
        extra = tuple(stats.get("extra", ()))
        print(
            f"{label} restore from {path_desc}: loaded={loaded}, "
            f"missing={len(missing)}, mismatched={len(mismatched)}, extra={len(extra)}",
            flush=True,
        )
        if missing:
            print(
                f"  Missing leaves kept from current init: {_format_restore_keys(missing)}",
                flush=True,
            )
        if mismatched:
            print(
                f"  Shape-mismatched leaves kept from current init: {_format_restore_keys(mismatched)}",
                flush=True,
            )
        if extra:
            print(
                f"  Extra checkpoint leaves ignored: {_format_restore_keys(extra)}",
                flush=True,
            )

    def _restore_params_from_raw(target_params, raw_bytes, path_desc=""):
        restored_obj = serialization.msgpack_restore(raw_bytes)
        source_params = checkpoint_params_subtree(restored_obj)
        params, stats = merge_compatible_state(target_params, source_params)
        params, aux_note = seed_missing_auxiliary_heads_from_main(params, source_params)
        _log_restore_stats("Param", path_desc, stats)
        if aux_note:
            print(f"  NOTE: {aux_note}", flush=True)
        return params

    def _restore_params_from_flax(target_params, ckpt_dir, ckpt_prefix, *, step=None, path_desc=""):
        try:
            restored_obj = checkpoints.restore_checkpoint(
                ckpt_dir,
                target=None,
                step=step,
                prefix=ckpt_prefix,
            )
        except Exception:
            return None
        if restored_obj is None:
            return None
        source_params = checkpoint_params_subtree(restored_obj)
        params, stats = merge_compatible_state(target_params, source_params)
        params, aux_note = seed_missing_auxiliary_heads_from_main(params, source_params)
        _log_restore_stats("Param", path_desc or ckpt_dir, stats)
        if aux_note:
            print(f"  NOTE: {aux_note}", flush=True)
        return params

    def _restore_train_state_from_flax(target_state, ckpt_dir, ckpt_prefix, *, step=None, path_desc=""):
        try:
            restored_obj = checkpoints.restore_checkpoint(
                ckpt_dir,
                target=None,
                step=step,
                prefix=ckpt_prefix,
            )
        except Exception:
            return None
        if restored_obj is None:
            return None
        restored_state, stats = merge_compatible_state(target_state, restored_obj)
        source_params = checkpoint_params_subtree(restored_obj)
        restored_params, aux_note = seed_missing_auxiliary_heads_from_main(
            restored_state.params,
            source_params,
        )
        if aux_note:
            restored_state = restored_state.replace(params=restored_params)
        _log_restore_stats("TrainState", path_desc or ckpt_dir, stats)
        if aux_note:
            print(f"  NOTE: {aux_note}", flush=True)
        return restored_state

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
                params = _restore_params_from_raw(state.params, raw, path_desc=hist_path)
                state = state.replace(params=params)
            else:
                # Fallback: try the Flax checkpoint layout
                flax_ckpt_path = os.path.join(src_ckpt_dir, f"{src_ckpt_prefix}{step_arg}")
                if os.path.exists(flax_ckpt_path):
                    print(
                        f"Restoring fine-tune checkpoint at step {step_arg}",
                        flush=True,
                    )
                    params = _restore_params_from_flax(
                        state.params,
                        src_ckpt_dir,
                        src_ckpt_prefix,
                        step=step_arg,
                        path_desc=flax_ckpt_path,
                    )
                    if params is None:
                        raise ValueError(
                            f"Fine-tune checkpoint step {step_arg} could not be restored from {flax_ckpt_path}."
                        )
                    state = state.replace(params=params)
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
                params = _restore_params_from_raw(state.params, raw, path_desc=src_ckpt_blob)
                state = state.replace(params=params)
            else:
                params = _restore_params_from_flax(
                    state.params,
                    src_ckpt_dir,
                    src_ckpt_prefix,
                    path_desc=f"{src_ckpt_dir} (prefix={src_ckpt_prefix})",
                )
                if params is None:
                    raise ValueError(
                        f"Fine-tune source checkpoint not found at {source_ckpt_path}. "
                        "Provide --fine_tune_ckpt_path or a RUNID-STEP that exists."
                    )
                state = state.replace(params=params)

    ckpt_dir, ckpt_prefix, ckpt_blob = resolve_ckpt_paths(t_cfg.ckpt_path)
    ckpt_async_manager = checkpoints.AsyncManager() if hasattr(checkpoints, "AsyncManager") else None
    
    if getattr(args, "continue_run_id", ""):
        print(f"Restoring checkpoint for continued run {args.continue_run_id}...", flush=True)
        restored = _restore_train_state_from_flax(
            state,
            ckpt_dir,
            ckpt_prefix,
            path_desc=f"{ckpt_dir} (prefix={ckpt_prefix})",
        )
        if restored is None:
            raise RuntimeError(
                f"FATAL: --continue was requested for run {args.continue_run_id}, "
                f"but no Flax checkpoint could be loaded from {ckpt_dir} (prefix: {ckpt_prefix}). "
                "Failing fast to prevent overwriting the run's history."
            )
        state = restored
    elif os.path.exists(ckpt_blob) or os.path.exists(
        os.path.join(ckpt_dir, f"{ckpt_prefix}0")
    ):
        print("Restoring checkpoint...", flush=True)
        restored = _restore_train_state_from_flax(
            state,
            ckpt_dir,
            ckpt_prefix,
            path_desc=f"{ckpt_dir} (prefix={ckpt_prefix})",
        )
        if restored is not None:
            state = restored
        elif os.path.exists(ckpt_blob):
            with open(ckpt_blob, "rb") as f:
                raw = f.read()
            params = _restore_params_from_raw(state.params, raw, path_desc=ckpt_blob)
            state = state.replace(params=params)

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
        data_fetcher = MonitorFineTuneBatcher(
            fine_tune_data,
            d_cfg,
            monitor_root=args.fine_tune_train_root,
            augment=bool(args.fine_tune_augment_monitor),
            active_learning_store=args.active_learning_store,
            active_learning_limit=int(args.active_learning_max_windows),
            dense_bias_label=str(args.fine_tune_dense_bias_label or ""),
            dense_bias_prob=float(args.fine_tune_dense_bias_prob),
            full_files=bool(args.full_files),
            full_file_max_bytes=int(args.full_file_max_bytes),
        )
    else:
        data_fetcher = EpochPrefetchBatcher(train_dsets, d_cfg)

    al_store_path = (args.active_learning_store or "").strip()
    al_mix_prob = float(max(0.0, min(1.0, args.active_learning_mix_prob)))
    al_label_to_id = dict(cfg.LANG2ID)
    if getattr(cfg, "OTHER_CLASS_INDEX", None) is not None:
        al_label_to_id["other"] = int(cfg.OTHER_CLASS_INDEX)

    def _maybe_enable_active_learning_replay(
        fetcher,
        *,
        mix_prob: float,
        phase_label: str = "",
        phase_nonce: int = 0,
    ):
        requested_mix_prob = float(max(0.0, min(1.0, mix_prob)))
        if not al_store_path:
            return fetcher

        try:
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from active_learning.training import ActiveLearningReplay, MixedBatcher
        except Exception as e:
            print(
                f"⚠️  Active-learning replay disabled (load failure): {e}",
                flush=True,
            )
            return fetcher

        replay_unit = "sequences" if bool(args.full_files) else "windows"
        prefix = f"[{phase_label}] " if phase_label else ""

        if hasattr(fetcher, "refresh_replay_from_store") and hasattr(fetcher, "replay"):
            old_mix_prob = float(getattr(fetcher, "mix_prob", requested_mix_prob))
            if requested_mix_prob <= 0.0:
                setattr(fetcher, "mix_prob", 0.0)
                if phase_label and abs(old_mix_prob) > 1e-12:
                    print(
                        f"{prefix}Active-learning replay mix_prob updated from {old_mix_prob:.4f} to 0.0000.",
                        flush=True,
                    )
                return fetcher
            old_size = int(getattr(getattr(fetcher, "replay", None), "size", 0))
            try:
                replay_size = int(
                    fetcher.refresh_replay_from_store(
                        store_path=al_store_path,
                        window_bytes=int(d_cfg.window_max_bytes),
                        pad_byte_id=int(cfg.PAD_BYTE_ID),
                        pad_label_id=int(cfg.PAD_ID),
                        label_to_id=al_label_to_id,
                        max_windows=int(args.active_learning_max_windows),
                        seed=int(args.seed),
                        fallback_label="other",
                        full_files=bool(args.full_files),
                        phase_nonce=int(phase_nonce),
                        mix_prob=requested_mix_prob,
                    )
                )
            except Exception as e:
                print(
                    f"⚠️  Active-learning replay refresh skipped; continuing with previous snapshot: {e}",
                    flush=True,
                )
                return fetcher
            print(
                f"{prefix}Active-learning replay refreshed: {replay_size} {replay_unit} from {al_store_path} "
                f"(mix_prob={requested_mix_prob:.2f}, previous_size={old_size}, previous_mix_prob={old_mix_prob:.4f}).",
                flush=True,
            )
            return fetcher

        if requested_mix_prob <= 0.0:
            return fetcher

        try:
            replay = ActiveLearningReplay.from_store(
                store_path=al_store_path,
                window_bytes=int(d_cfg.window_max_bytes),
                pad_byte_id=int(cfg.PAD_BYTE_ID),
                pad_label_id=int(cfg.PAD_ID),
                label_to_id=al_label_to_id,
                max_windows=int(args.active_learning_max_windows),
                seed=int(args.seed),
                fallback_label="other",
                full_files=bool(args.full_files),
            )
        except Exception as e:
            print(
                f"⚠️  Active-learning replay disabled (load failure): {e}",
                flush=True,
            )
            return fetcher

        if replay.size <= 0:
            if requested_mix_prob > 0.0:
                print(
                    f"Active-learning store '{al_store_path}' contained no usable replay samples; replay disabled.",
                    flush=True,
                )
            return fetcher

        fetcher = MixedBatcher(
            fetcher,
            replay,
            mix_prob=requested_mix_prob,
            seed=int(args.seed),
        )
        print(
            f"{prefix}Active-learning replay enabled: {replay.size} {replay_unit} from {al_store_path} "
            f"(mix_prob={requested_mix_prob:.2f}).",
            flush=True,
        )
        return fetcher

    data_fetcher = _maybe_enable_active_learning_replay(
        data_fetcher,
        mix_prob=al_mix_prob,
        phase_nonce=int(_state_step(state, use_pmap=use_pmap)),
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
        commit: bool = True,
    ) -> tuple[float, float]:
        """Run val eval and, if enabled, monitor eval; log everything to W&B."""
        nonlocal rng

        print(f"{title}...", flush=True)
        rng, eval_rng = jax.random.split(rng)

        # Eval runs single-device; unreplicate if using pmap
        eval_state = unreplicate_state(state) if use_pmap else state
        full_files_mode = bool(getattr(t_cfg, "full_files", False))
        primary_loss = float("nan")
        primary_acc = float("nan")
        aggregates = None
        per_class = None
        conf_mat = None

        if not full_files_mode and not t_cfg.fine_tune:
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
            primary_loss = float(val_loss)
            primary_acc = float(val_acc)
        else:
            print(
                "Fine-tune/full-file mode: skipping ordinary base-corpus validation.",
                flush=True,
            )
            _wandb_safe_log({"meta/val_skipped_full_files": 1}, step=step, commit=False)

        if monitor_data is not None and t_cfg.monitor_eval_every > 0:
            if bool(t_cfg.monitor_eval_deterministic):
                print(
                    f"Running monitor evaluation (deterministic seed={int(t_cfg.monitor_eval_seed)})...",
                    flush=True,
                )
            else:
                print("Running monitor evaluation...", flush=True)
            eval_rng, monitor_rng = jax.random.split(eval_rng)
            limit = None if t_cfg.monitor_eval_limit < 0 else int(t_cfg.monitor_eval_limit)
            monitor_batch_size = (
                int(t_cfg.monitor_eval_batch_size)
                if int(t_cfg.monitor_eval_batch_size) > 0
                else int(d_cfg.batch_size)
            )
            monitor_stats = evaluate_monitor_set(
                state=eval_state,
                monitor_data=monitor_data,
                L=d_cfg.window_max_bytes,
                batch_size=monitor_batch_size,
                rng=monitor_rng,
                limit=limit,
                eval_step_fn=eval_step,
                eval_step_with_logits_fn=eval_step_with_logits,
                other_threshold=t_cfg.monitor_other_threshold,
                deterministic=bool(t_cfg.monitor_eval_deterministic),
                deterministic_seed=int(t_cfg.monitor_eval_seed),
                full_files=bool(full_files_mode),
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
            monitor_gap_logs = {}
            if aggregates is not None:
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
                }
                if aggregates is not None:
                    thresh_logs.update(
                        {
                            "monitor_gap_thresh/micro_accuracy": float(aggregates_mt["micro"]["acc"] - aggregates["micro"]["acc"]),
                            "monitor_gap_thresh/macro_f1": float(aggregates_mt["macro"]["f1"] - aggregates["macro"]["f1"]),
                            "monitor_gap_thresh/macro_precision": float(aggregates_mt["macro"]["precision"] - aggregates["macro"]["precision"]),
                            "monitor_gap_thresh/macro_recall": float(aggregates_mt["macro"]["recall"] - aggregates["macro"]["recall"]),
                            "monitor_gap_thresh/weighted_f1": float(aggregates_mt["weighted"]["f1"] - aggregates["weighted"]["f1"]),
                        }
                    )
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
            if full_files_mode:
                primary_loss = float(monitor_stats["loss_mean"])
                primary_acc = float(monitor_stats["acc_mean"])

        if per_class is not None and aggregates is not None and conf_mat is not None:
            wandb_log_metrics(
                step,
                per_class,
                aggregates,
                cfg.ID2LANG,
                cfg.NUM_CLASSES,
                cfg.PAD_ID,
                conf_mat,
                commit=commit,
            )
        else:
            _wandb_safe_log({"meta/full_file_eval": int(full_files_mode)}, step=step, commit=commit)
        return float(primary_loss), float(primary_acc)

    current_step = _state_step(state, use_pmap=use_pmap)
    startup_log_step = _wandb_current_step(default=current_step)
    run_start_eval = bool(args.eval_on_start or current_step == 0)
    _wandb_safe_log(
        {
            "meta/monitor_train_files": int(fine_tune_train_files_count),
            "meta/monitor_eval_files": int(monitor_eval_files_count),
        },
        step=startup_log_step,
        commit=bool(current_step > 0 and not run_start_eval),
    )
    if run_start_eval:
        startup_title = (
            "Baseline validation (step 0)"
            if current_step == 0
            else f"Startup validation (pre-train, checkpoint step {current_step})"
        )
        run_val_and_monitor_eval(
            step=startup_log_step,
            title=startup_title,
            commit=bool(current_step == 0),
        )
    persistent_mode = bool(args.persistent_trainer)

    def _phase_restart_rng(current_step: int):
        phase_rng = jax.random.PRNGKey(t_cfg.rng_seed)
        phase_rng, _ = jax.random.split(phase_rng)
        if bool(args.eval_on_start) or int(current_step) == 0:
            phase_rng, _ = jax.random.split(phase_rng)
        return phase_rng

    def _reset_phase_runtime_state(*, phase_label: str) -> None:
        nonlocal rng

        current_phase_step = int(_state_step(state, use_pmap=use_pmap))
        rng = _phase_restart_rng(current_phase_step)

        drained = 0
        if hasattr(data_fetcher, "reset_phase"):
            drained = int(
                data_fetcher.reset_phase(
                    seed=int(args.seed),
                    phase_nonce=current_phase_step,
                    refresh_active_learning=bool(
                        al_store_path and bool(args.fine_tune_augment_monitor)
                    ),
                )
                or 0
            )
        if drained > 0:
            print(
                f"[{phase_label}] Cleared {drained} prefetched batch(es) before starting the next persistent phase.",
                flush=True,
            )
        if oe_batcher is not None and hasattr(oe_batcher, "reset_phase"):
            oe_batcher.reset_phase(
                seed=int(args.seed),
                phase_nonce=current_phase_step,
            )

    def _run_training_phase(*, target_step: int, phase_label: str, phase_max_minutes: int) -> dict:
        nonlocal state, rng

        current_phase_step = _state_step(state, use_pmap=use_pmap)
        target_step = int(target_step)
        if target_step < current_phase_step:
            print(
                f"[{phase_label}] No training needed: current_step={current_phase_step} target_step={target_step}.",
                flush=True,
            )
            return {
                "status": "noop",
                "phase_label": str(phase_label),
                "current_step": int(current_phase_step),
                "target_step": int(target_step),
                "final_reason": "noop_already_at_or_beyond_target",
                "runtime_minutes": 0.0,
                "pruned": 0,
                "error_message": "",
            }

        requested_updates = max(0, int(target_step) - int(current_phase_step) + 1)
        print(
            f"Starting training... phase={phase_label} current_step={current_phase_step} "
            f"target_step={target_step} updates={requested_updates} persistent={int(persistent_mode)}",
            flush=True,
        )
        start_time = time.time()
        last_log_time = time.time()
        metrics_history = {"loss": [], "acc": [], "step_time": [], "grad_norm": []}

        best_val = float("inf")
        non_improve_evals = 0
        pruned = False
        stopped_reason = ""
        error_reason = ""
        error_type_name = ""
        last_heartbeat = time.time()

        min_epochs = 0.0
        max_epochs = 0.0
        target_logged_step = int(target_step) + 1
        first_batch_announced = False
        first_batch_ready_announced = False
        first_compile_started = False
        first_compile_started_at = 0.0

        try:
            try:
                for step in range(current_phase_step, target_step + 1):
                    if step % 100 == 0:
                        gc.collect()

                    elapsed_min = (time.time() - start_time) / 60.0
                    if phase_max_minutes > 0 and elapsed_min >= phase_max_minutes:
                        stopped_reason = f"time_cap_{phase_max_minutes}min"
                        print(
                            f"[{phase_label}] Time cap reached ({phase_max_minutes} min). Stopping gracefully.",
                            flush=True,
                        )
                        break
                    if os.path.exists(args.stop_file):
                        stopped_reason = "stop_file_detected"
                        print(
                            f"[{phase_label}] Stop file detected at '{args.stop_file}'. Stopping gracefully.",
                            flush=True,
                        )
                        break

                    if _STOP["flag"]:
                        stopped_reason = f"signal_{int(_STOP['flag'])}"
                        print(f"[{phase_label}] Stop requested by signal. Exiting loop.", flush=True)
                        break

                    step_start = time.time()
                    data_time = 0.0
                    compute_time = 0.0
                    grad_norm_value = None
                    oe_losses: list[float] = []
                    oe_batches_used = 0

                    rng, step_base_rng = jax.random.split(rng)

                    if accum_steps == 1:
                        if not first_batch_announced:
                            print(
                                f"[{phase_label}] Preparing first training batch from the fine-tune/replay pipeline...",
                                flush=True,
                            )
                            first_batch_announced = True
                        data_start = time.time()
                        batch_tokens, batch_labels = data_fetcher.get()
                        batch_tokens = sanitize_tokens(batch_tokens)
                        data_time = time.time() - data_start
                        if not first_batch_ready_announced:
                            print(
                                f"[{phase_label}] First batch ready in {data_time:.2f}s "
                                f"(tokens_shape={tuple(batch_tokens.shape)}, labels_shape={tuple(batch_labels.shape)}).",
                                flush=True,
                            )
                            first_batch_ready_announced = True

                        step_base_rng, oe_decision_rng = jax.random.split(step_base_rng)
                        use_oe = (
                            oe_batcher is not None
                            and float(jax.random.uniform(oe_decision_rng, ()).item()) < oe_ratio
                        )
                        if not first_compile_started:
                            first_compile_started = True
                            first_compile_started_at = time.time()
                            print(
                                f"[{phase_label}] Launching first compiled train step "
                                f"(this can take a while for new shapes, especially 10k full-file mode)...",
                                flush=True,
                            )
                        compute_start = time.time()
                        if use_pmap:
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
                        if first_compile_started_at > 0.0:
                            print(
                                f"[{phase_label}] First compiled train step finished in "
                                f"{time.time() - first_compile_started_at:.2f}s. Subsequent steps should be much faster.",
                                flush=True,
                            )
                            first_compile_started_at = 0.0
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

                        for micro_idx in range(accum_steps):
                            if not first_batch_announced:
                                print(
                                    f"[{phase_label}] Preparing first microbatch from the fine-tune/replay pipeline...",
                                    flush=True,
                                )
                                first_batch_announced = True
                            data_start = time.time()
                            mb_tokens, mb_labels = data_fetcher.get()
                            mb_tokens = sanitize_tokens(mb_tokens)
                            data_time += time.time() - data_start
                            if not first_batch_ready_announced:
                                print(
                                    f"[{phase_label}] First microbatch ready in {data_time:.2f}s "
                                    f"(tokens_shape={tuple(mb_tokens.shape)}, labels_shape={tuple(mb_labels.shape)}, "
                                    f"accum_steps={accum_steps}, micro_idx={micro_idx + 1}/{accum_steps}).",
                                    flush=True,
                                )
                                first_batch_ready_announced = True

                            micro_rng, oe_decision_rng = jax.random.split(micro_rng)
                            use_oe = (
                                oe_batcher is not None
                                and float(jax.random.uniform(oe_decision_rng, ()).item()) < oe_ratio
                            )
                            micro_rng, use_rng = jax.random.split(micro_rng)
                            if not first_compile_started:
                                first_compile_started = True
                                first_compile_started_at = time.time()
                                print(
                                    f"[{phase_label}] Launching first compiled train step "
                                    f"(this can take a while for new shapes, especially 10k full-file mode)...",
                                    flush=True,
                                )
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
                            if first_compile_started_at > 0.0:
                                print(
                                    f"[{phase_label}] First compiled train step finished in "
                                    f"{time.time() - first_compile_started_at:.2f}s. Subsequent steps should be much faster.",
                                    flush=True,
                                )
                                first_compile_started_at = 0.0
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
                            grad_norm_value = float(grad_global_norm(
                                jax.tree.map(lambda x: x[0], grad_accum)
                            ))
                            state = p_apply_gradients(state, grad_accum)
                        else:
                            grad_norm_value = float(grad_global_norm(grad_accum))
                            state = state.apply_gradients(grads=grad_accum)
                        apply_start = time.time()
                        compute_time += time.time() - apply_start
                        loss = float(sum(losses) / len(losses))
                        acc = float(sum(accs) / len(accs))

                    oe_loss_value = float(sum(oe_losses) / len(oe_losses)) if oe_losses else 0.0
                    step_time = time.time() - step_start
                    logged_step = int(step) + 1

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

                    should_commit = (
                        logged_step % t_cfg.log_every == 0
                        and logged_step % t_cfg.eval_every != 0
                    )

                    _wandb_safe_log(metrics, step=logged_step, commit=should_commit)
                    last_heartbeat = time.time()

                    if logged_step % t_cfg.log_every == 0:
                        elapsed = time.time() - last_log_time
                        sps = t_cfg.log_every / elapsed if elapsed > 0 else 0
                        print(
                            f"[{phase_label}] Step {logged_step}/{target_logged_step} [Epochs {min_epochs:.3f}-{max_epochs:.3f}] | "
                            f"Loss: {loss:.4f} (±{metrics['train/loss_std']:.4f}), "
                            f"Acc: {acc:.4f} (±{metrics['train/acc_std']:.4f}), SPS: {sps:.2f}",
                            flush=True,
                        )
                        last_log_time = time.time()

                    if time.time() - last_heartbeat > 600:
                        stopped_reason = "watchdog_no_progress_10min"
                        print(
                            f"[{phase_label}] Watchdog: no progress for 10 minutes. Stopping.",
                            flush=True,
                        )
                        break

                    if logged_step > 0 and logged_step % t_cfg.eval_every == 0:
                        val_loss, val_acc = run_val_and_monitor_eval(
                            step=logged_step,
                            title=f"Evaluating ({phase_label})",
                            train_loss=float(loss),
                            train_acc=float(acc),
                            commit=True,
                        )

                        save_state = unreplicate_state(state) if use_pmap else state
                        try:
                            _save_training_checkpoint(
                                save_state,
                                t_cfg.ckpt_path,
                                logged_step,
                                async_manager=ckpt_async_manager,
                            )
                        except Exception as e:
                            print(
                                f"WARNING: writing checkpoint failed: {e}",
                                flush=True,
                            )

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
                                        f"[PRUNE:{phase_label}] {stopped_reason}. Stopping run.",
                                        flush=True,
                                    )
                                    break

            except KeyboardInterrupt:
                stopped_reason = "keyboard_interrupt"
                print(f"[{phase_label}] Interrupted. Saving checkpoint and finishing chunk.", flush=True)
            except Exception as e:
                error_reason = str(e)
                error_type_name = type(e).__name__
                print(
                    f"\nFATAL ERROR in training loop ({phase_label}): {type(e).__name__}: {e}",
                    flush=True,
                )
        finally:
            if ckpt_async_manager is not None:
                try:
                    ckpt_async_manager.wait_previous_save()
                except Exception:
                    pass

        final_reason = "completed"
        if error_reason:
            final_reason = f"error_{error_type_name or 'runtime'}"
            current_log_step = _state_step(state, use_pmap=use_pmap)
            _wandb_safe_log(
                {"meta/error_message": error_reason},
                step=current_log_step,
                commit=False,
            )
        elif stopped_reason:
            final_reason = stopped_reason

        final_step = _state_step(state, use_pmap=use_pmap)
        if final_step > 0:
            save_state = unreplicate_state(state) if use_pmap else state
            try:
                _save_training_checkpoint(save_state, t_cfg.ckpt_path, final_step)
            except Exception as e:
                print(f"WARNING: final checkpoint save failed: {e}", flush=True)

        runtime_minutes = (time.time() - start_time) / 60.0
        _wandb_safe_log(
            {
                "meta/chunk_stopped_reason": final_reason,
                "meta/chunk_phase": str(phase_label),
                "meta/chunk_pruned": int(pruned),
                "meta/chunk_runtime_minutes": float(runtime_minutes),
                "meta/chunk_target_step": int(target_step),
                "meta/persistent_trainer": int(persistent_mode),
            },
            step=final_step,
        )
        return {
            "status": "error" if error_reason else "ok",
            "phase_label": str(phase_label),
            "current_step": int(final_step),
            "target_step": int(target_step),
            "requested_updates": int(requested_updates),
            "completed_updates": int(max(0, final_step - current_phase_step)),
            "final_reason": str(final_reason),
            "runtime_minutes": float(runtime_minutes),
            "pruned": int(pruned),
            "error_message": str(error_reason),
        }

    process_final_reason = "completed"
    process_error_message = ""
    try:
        if persistent_mode:
            ready_step = _state_step(state, use_pmap=use_pmap)
            print(
                f"Persistent trainer ready at checkpoint step {ready_step}; waiting for commands on stdin.",
                flush=True,
            )
            _emit_persistent_trainer_event(
                "ready",
                current_step=int(ready_step),
                wandb_run_id=str(getattr(getattr(wandb, "run", None), "id", "")),
                ckpt_path=str(t_cfg.ckpt_path),
            )
            while True:
                try:
                    command = _read_persistent_trainer_command()
                except Exception as exc:
                    print(f"Persistent trainer command parse error: {exc}", flush=True)
                    _emit_persistent_trainer_event("command_error", error=str(exc))
                    continue

                command_name = str(command.get("command", "")).strip().lower()
                if command_name == "shutdown":
                    reason = str(command.get("reason", "shutdown")).strip() or "shutdown"
                    process_final_reason = f"shutdown_{reason}"
                    _emit_persistent_trainer_event(
                        "shutdown_ack",
                        current_step=int(_state_step(state, use_pmap=use_pmap)),
                        reason=reason,
                    )
                    break
                if command_name != "train":
                    message = f"Unsupported trainer command: {command_name}"
                    print(message, flush=True)
                    _emit_persistent_trainer_event("command_error", error=message)
                    continue

                try:
                    command_target_step = int(command.get("steps"))
                except Exception as exc:
                    message = f"Missing/invalid 'steps' in trainer command: {exc}"
                    print(message, flush=True)
                    _emit_persistent_trainer_event("command_error", error=message)
                    continue

                command_phase_label = str(
                    command.get("phase_label", f"persistent_step_{command_target_step}")
                ).strip() or f"persistent_step_{command_target_step}"
                command_max_minutes = int(command.get("max_minutes", args.max_minutes))
                current_phase_step = int(_state_step(state, use_pmap=use_pmap))
                if "active_learning_mix_prob" in command:
                    data_fetcher = _maybe_enable_active_learning_replay(
                        data_fetcher,
                        mix_prob=float(command.get("active_learning_mix_prob", 0.0)),
                        phase_label=command_phase_label,
                        phase_nonce=current_phase_step,
                    )
                _reset_phase_runtime_state(phase_label=command_phase_label)
                chunk_summary = _run_training_phase(
                    target_step=command_target_step,
                    phase_label=command_phase_label,
                    phase_max_minutes=command_max_minutes,
                )
                event_name = "chunk_failed" if str(chunk_summary.get("status")) == "error" else "chunk_done"
                _emit_persistent_trainer_event(event_name, **chunk_summary)
                if str(chunk_summary.get("status")) == "error":
                    process_final_reason = str(chunk_summary.get("final_reason", "error"))
                    process_error_message = str(chunk_summary.get("error_message", ""))
                    break
                if _STOP["flag"]:
                    process_final_reason = str(chunk_summary.get("final_reason", "signal"))
                    break
        else:
            _reset_phase_runtime_state(phase_label="train_main")
            chunk_summary = _run_training_phase(
                target_step=int(t_cfg.steps),
                phase_label="train_main",
                phase_max_minutes=int(args.max_minutes),
            )
            process_final_reason = str(chunk_summary.get("final_reason", "completed"))
            process_error_message = str(chunk_summary.get("error_message", ""))
    finally:
        try:
            data_fetcher.close()
        except Exception:
            pass

        if ckpt_async_manager is not None:
            try:
                ckpt_async_manager.wait_previous_save()
            except Exception:
                pass

        current_step = _state_step(state, use_pmap=use_pmap)
        if current_step > 0:
            save_state = unreplicate_state(state) if use_pmap else state
            try:
                _save_training_checkpoint(save_state, t_cfg.ckpt_path, current_step)
            except Exception as e:
                print(f"WARNING: final checkpoint save failed: {e}", flush=True)

        if process_error_message:
            _wandb_safe_log(
                {"meta/error_message": process_error_message},
                step=current_step,
                commit=False,
            )
        _wandb_safe_log(
            {
                "meta/stopped_reason": str(process_final_reason),
                "meta/persistent_trainer": int(persistent_mode),
            },
            step=current_step,
        )
        wandb.finish()
        print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
