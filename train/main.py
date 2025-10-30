import argparse
import os
import random
import sys
import time
from typing import Dict, Tuple, TYPE_CHECKING

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
import numpy as np
from flax.training import checkpoints
from flax import serialization
import wandb
from wandb import Settings

# Enable memory-efficient JAX flags
jax.config.update("jax_disable_jit", False)
jax.config.update("jax_enable_x64", False)

import config as cfg
from data_utils import prepare_dsets_by_lang_with_splits, LANG_ALIASES
from window_generator import make_training_window, make_training_window_with_metadata
from model import (
    create_train_state,
    train_step,
    eval_step,
    TrainState,
    count_params,
    train_step_no_jit,
)
from preview import build_preview_html

from metrics_helper import (
    evaluate_split_with_metrics,
    compute_metrics_from_confusion,
    print_metrics_table,
    wandb_log_metrics,
)
from token_utils import sanitize_tokens

if TYPE_CHECKING:
    from config import DataConfig, TrainConfig

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

    # Data args
    parser.add_argument(
        "--data_root", type=str, default="../downloader/arrow_out"
    )
    parser.add_argument("--allow_hf_fallback", action="store_true", default=False)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # 4KB fixed windows by default
    parser.add_argument("--bucket_step", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_minutes", type=int, default=0)
    parser.add_argument("--stop_file", type=str, default="STOP_SWEEP")
    parser.add_argument("--dont_use_train_windows", action="store_true", default=False,
                      help="Use train directory instead of train_windows for training data")
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
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--channels", type=str, default="96,128,160,192,224,256,288,320")
    parser.add_argument("--dropout_rate", type=float, default=0.15)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--ckpt_path", type=str, default="auto")
    parser.add_argument("--sweep_id", type=str, default="")
    parser.add_argument("--no_jit", action="store_true")
    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--preview_start", type=int, default=0)
    parser.add_argument("--preview_count", type=int, default=10)

    args = parser.parse_args()

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
        allow_hf_fallback=args.allow_hf_fallback,
        num_proc=args.num_proc,
        seed=args.seed,
        bucket_step=args.bucket_step,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    t_cfg = cfg.TrainConfig(
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup=args.warmup,
        model_dim=args.model_dim,
        channels=tuple(map(int, args.channels.split(","))),
        dropout_rate=args.dropout_rate,
        log_every=args.log_every,
        eval_every=args.eval_every,
        ckpt_path=args.ckpt_path,
        sweep_id=args.sweep_id,
        no_jit=args.no_jit,
        preview_only=args.preview_only,
        preview_start=args.preview_start,
        preview_count=args.preview_count,
    )

    # Prepare datasets
    print("Preparing datasets...", flush=True)
    dsets = prepare_dsets_by_lang_with_splits(
        d_cfg.data_root,
        use_train_windows=not args.dont_use_train_windows,
        include_languages=selected_langs,
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

    # --- WandB: single init here (longer timeout) ---
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "code-segmentation"),
        settings=Settings(init_timeout=300),
    )
    wandb.config.update({**d_cfg.__dict__, **t_cfg.__dict__}, allow_val_change=True)

    # Derive unique checkpoint path from run id if requested/placeholder-ish
    if t_cfg.ckpt_path == "auto" or "${" in t_cfg.ckpt_path:
        auto_ckpt = f"checkpoints/sweeps/{wandb.run.id}.msgpack"
        os.makedirs(os.path.dirname(auto_ckpt), exist_ok=True)
        t_cfg.ckpt_path = auto_ckpt
        print(f"Using auto checkpoint path: {t_cfg.ckpt_path}", flush=True)

    # Warm up device early so any XLA/CUDA issues show now
    print("JAX devices:", jax.devices(), flush=True)
    _ = jnp.ones((1,)).block_until_ready()

    rng = jax.random.PRNGKey(t_cfg.rng_seed)
    rng, init_rng = jax.random.split(rng)

    print("Creating train state (may trigger JIT/compile)...", flush=True)
    state = create_train_state(init_rng, t_cfg, cfg.NUM_CLASSES)
    num_params = count_params(state.params)
    print(f"Model created with {num_params/1e6:.2f}M parameters.", flush=True)

    ckpt_dir, ckpt_prefix, ckpt_blob = resolve_ckpt_paths(t_cfg.ckpt_path)
    ckpt_async_manager = checkpoints.AsyncManager() if hasattr(checkpoints, "AsyncManager") else None
    if os.path.exists(ckpt_blob) or os.path.exists(
        os.path.join(ckpt_dir, f"{ckpt_prefix}0")
    ):
        print("Restoring checkpoint...", flush=True)
        state = checkpoints.restore_checkpoint(ckpt_dir, state, prefix=ckpt_prefix)

    # Switch to epoch-based batching
    from epoch_batcher import EpochPrefetchBatcher
    data_fetcher = EpochPrefetchBatcher(train_dsets, d_cfg)
    train_step_fn = train_step_no_jit if t_cfg.no_jit else train_step

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
            for step in range(state.step, t_cfg.steps + 1):
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

                data_start = time.time()
                batch_tokens, batch_labels = data_fetcher.get()
                batch_tokens = sanitize_tokens(batch_tokens)
                data_time = time.time() - data_start

                rng, step_rng = jax.random.split(rng)
                state, loss, acc = train_step_fn(
                    state, batch_tokens, batch_labels, step_rng
                )
                # Explicitly delete batch data to free memory
                del batch_tokens
                del batch_labels
                compute_time = time.time() - data_start - data_time

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
                    val_loss, val_acc, conf_mat = evaluate_split_with_metrics(
                        state=state,
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
                    _wandb_safe_log(val_metrics, step=step, commit=False)

                    # Print table and log to W&B
                    print_metrics_table(
                        per_class, aggregates, cfg.ID2LANG, cfg.NUM_CLASSES, cfg.PAD_ID
                    )
                    wandb_log_metrics(
                        step,
                        per_class,
                        aggregates,
                        cfg.ID2LANG,
                        cfg.NUM_CLASSES,
                        cfg.PAD_ID,
                        conf_mat,
                    )

                    # Save checkpoints (unchanged)
                    ckpt_dir, ckpt_prefix, ckpt_blob = resolve_ckpt_paths(
                        t_cfg.ckpt_path
                    )
                    checkpoints.save_checkpoint(
                        ckpt_dir,
                        state,
                        step=step,
                        prefix=ckpt_prefix,
                        keep=2,
                        overwrite=True,
                        async_manager=ckpt_async_manager,
                    )

                    try:
                        os.makedirs(ckpt_dir, exist_ok=True)
                        with open(ckpt_blob, "wb") as f:
                            f.write(serialization.to_bytes(state.params))
                        base = os.path.basename(ckpt_blob)
                        stem, ext = os.path.splitext(base)
                        hist_path = os.path.join(ckpt_dir, f"{stem}-{step}{ext}")
                        with open(hist_path, "wb") as f:
                            f.write(serialization.to_bytes(state.params))
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
