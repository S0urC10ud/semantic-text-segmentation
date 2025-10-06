"""
Metrics helpers for per-class and aggregated accuracy / precision / recall / F1,
plus nice console tables and Weights & Biases logging.
"""
from typing import Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import wandb

from config import NUM_CLASSES, PAD_ID, PAD_BYTE_ID
from window_generator import make_pure_window, make_mixed_window


# ----------------------------- batch building ---------------------------------

def _make_eval_batch(
    dsets_by_lang: Dict[int, dict],
    L: int,
    batch_size: int,
    mix_prob: float,
    min_seg: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an eval batch mirroring the data pipeline logic."""
    xb = np.full((batch_size, L), PAD_BYTE_ID, dtype=np.int32)
    yb = np.full((batch_size, L), PAD_ID, dtype=np.uint8)
    for i in range(batch_size):
        if np.random.rand() < mix_prob:
            x, y = make_mixed_window(dsets_by_lang, target_len=L, min_seg=min_seg)
        else:
            x, y = make_pure_window(dsets_by_lang, target_len=L)
        xb[i] = x
        yb[i] = y
    return xb, yb


# ----------------------------- forward helper ---------------------------------

def _forward_logits(state, x: jnp.ndarray, rng):
    """
    Run a pure forward pass to get logits for metrics; robust across
    different TrainState.apply_fn call signatures found in Flax codebases.

    We try, in order:
      1) variables dict with batch_stats + train=False + rngs
      2) variables dict with only params + train=False + rngs
      3) raw params + train=False + rngs
      4) variables dict with batch_stats (no rngs/flags)
      5) variables dict with only params (no rngs/flags)
      6) raw params only
    """
    variables_params = {"params": state.params}
    if hasattr(state, "batch_stats"):
        variables_w_stats = {"params": state.params, "batch_stats": state.batch_stats}
    else:
        variables_w_stats = None

    try_order = []
    if variables_w_stats is not None:
        try_order.append(("vars+stats+flags", lambda: state.apply_fn(variables_w_stats, x, train=False, rngs={"dropout": rng})))
    try_order.append(("vars+flags",         lambda: state.apply_fn(variables_params, x, train=False, rngs={"dropout": rng})))
    try_order.append(("params+flags",       lambda: state.apply_fn(state.params, x, train=False, rngs={"dropout": rng})))
    if variables_w_stats is not None:
        try_order.append(("vars+stats",     lambda: state.apply_fn(variables_w_stats, x)))
    try_order.append(("vars",               lambda: state.apply_fn(variables_params, x)))
    try_order.append(("params",             lambda: state.apply_fn(state.params, x)))

    last_err = None
    for _, fn in try_order:
        try:
            return fn()
        except Exception as e:
            last_err = e
            continue
    raise last_err


# --------------------------- confusion + metrics -------------------------------

def accumulate_confusion(conf_mat: np.ndarray, y_true_np: np.ndarray, y_pred_np: np.ndarray):
    """In-place accumulation of counts into conf_mat[C,C]."""
    y_true_np = y_true_np.reshape(-1)
    y_pred_np = y_pred_np.reshape(-1)
    np.add.at(conf_mat, (y_true_np, y_pred_np), 1)


def compute_metrics_from_confusion(
    conf_mat: np.ndarray, num_classes: int, ignore_class: int
):
    """
    Compute per-class + aggregated metrics from a confusion matrix.
    Per-class 'acc' is recall over true instances (TP / (TP + FN)), which is
    usually the most useful per-class 'accuracy'. Switch if you prefer literal
    (TP+TN)/(TP+TN+FP+FN).
    """
    cm = conf_mat.copy()
    if ignore_class is not None and 0 <= ignore_class < num_classes:
        cm[ignore_class, :] = 0
        cm[:, ignore_class] = 0

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    support = cm.sum(axis=1)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1        = np.divide(2*precision*recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    # Per-class "accuracy" as recall over true instances:
    class_acc = recall.copy()

    present = support > 0
    macro_prec = float(precision[present].mean()) if present.any() else 0.0
    macro_rec  = float(recall[present].mean())    if present.any() else 0.0
    macro_f1   = float(f1[present].mean())        if present.any() else 0.0

    total = float(cm.sum())
    micro_acc = float(tp.sum() / total) if total > 0 else 0.0
    # For single-label multiclass, micro P/R/F1 all equal micro accuracy.
    micro_prec = micro_acc
    micro_rec  = micro_acc
    micro_f1   = micro_acc

    weighted_f1 = float((f1 * support).sum() / support.sum()) if support.sum() > 0 else 0.0

    per_class = {
        "support": support,
        "acc": class_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    aggregates = {
        "micro": {"acc": micro_acc, "precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
        "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
        "weighted": {"f1": weighted_f1},
    }
    return per_class, aggregates


# --------------------------- pretty printing ----------------------------------

def print_metrics_table(
    per_class,
    aggregates,
    id2label: Dict[int, str],
    num_classes: int,
    ignore_class: int,
):
    """Pretty console table: aggregated row first, then classes sorted by index."""
    header = f"{'label':<20} {'support':>8} {'acc':>8} {'prec':>8} {'recall':>8} {'f1':>8}"
    print("\n[Validation] Per-class metrics (per-class acc = recall over true instances):", flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    agg_row = [
        "ALL (agg)",
        "",  # only affects console print; W&B table uses NaN below
        f"{aggregates['micro']['acc']:.4f}",
        f"{aggregates['macro']['precision']:.4f}",
        f"{aggregates['macro']['recall']:.4f}",
        f"{aggregates['macro']['f1']:.4f}",
    ]
    print(f"{agg_row[0]:<20} {agg_row[1]:>8} {agg_row[2]:>8} {agg_row[3]:>8} {agg_row[4]:>8} {agg_row[5]:>8}", flush=True)

    support = per_class["support"]; acc = per_class["acc"]; prec = per_class["precision"]; rec = per_class["recall"]; f1 = per_class["f1"]
    for cid in range(num_classes):
        if cid == ignore_class:
            continue
        if support[cid] == 0:
            continue
        name = id2label.get(cid, str(cid))
        print(f"{name:<20} {int(support[cid]):>8} {acc[cid]:>8.4f} {prec[cid]:>8.4f} {rec[cid]:>8.4f} {f1[cid]:>8.4f}", flush=True)
    print("", flush=True)


# ----------------------------- W&B logging ------------------------------------

def _wandb_safe_log(data: dict, step: int, commit: bool = True):
    try:
        wandb.log(data, step=step, commit=commit)
    except Exception as e:
        print(f"Wandb logging failed: {e}", flush=True)


def wandb_log_metrics(
    step: int,
    per_class,
    aggregates,
    id2label: Dict[int, str],
    num_classes: int,
    ignore_class: int,
    conf_mat: np.ndarray,
):
    """Log a wandb.Table, aggregated scalars, per-class scalars, and a confusion-matrix image."""
    # 1) Table — ensure consistent numeric types across ALL rows.
    columns = ["label", "support", "acc", "precision", "recall", "f1"]
    rows = []

    # Aggregated row: make 'support' a NUMBER (NaN) instead of "" to keep column numeric
    rows.append([
        str("ALL (agg)"),
        float("nan"),
        float(aggregates["micro"]["acc"]),
        float(aggregates["macro"]["precision"]),
        float(aggregates["macro"]["recall"]),
        float(aggregates["macro"]["f1"]),
    ])

    support = per_class["support"]; acc = per_class["acc"]; prec = per_class["precision"]; rec = per_class["recall"]; f1 = per_class["f1"]
    for cid in range(num_classes):
        if cid == ignore_class or support[cid] == 0:
            continue
        rows.append([
            str(id2label.get(cid, str(cid))),
            float(support[cid]),
            float(acc[cid]),
            float(prec[cid]),
            float(rec[cid]),
            float(f1[cid]),
        ])

    metrics_table = wandb.Table(columns=columns, data=rows)

    # 2) Aggregated scalars
    scalar_logs = {
        "val/agg/micro/accuracy": float(aggregates["micro"]["acc"]),
        "val/agg/macro/precision": float(aggregates["macro"]["precision"]),
        "val/agg/macro/recall": float(aggregates["macro"]["recall"]),
        "val/agg/macro/f1": float(aggregates["macro"]["f1"]),
        "val/agg/weighted/f1": float(aggregates["weighted"]["f1"]),
    }

    # 3) Per-class scalars
    per_class_logs = {}
    for cid in range(num_classes):
        if cid == ignore_class or support[cid] == 0:
            continue
        label = id2label.get(cid, str(cid))
        base = f"val/per_class/{cid:02d}_{label}"
        per_class_logs[f"{base}/support"]   = float(support[cid])
        per_class_logs[f"{base}/acc"]       = float(acc[cid])
        per_class_logs[f"{base}/precision"] = float(prec[cid])
        per_class_logs[f"{base}/recall"]    = float(rec[cid])
        per_class_logs[f"{base}/f1"]        = float(f1[cid])

    # 4) Confusion matrix image
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 5))
        cm_disp = conf_mat.copy()
        if ignore_class is not None and 0 <= ignore_class < num_classes:
            cm_disp[ignore_class, :] = 0
            cm_disp[:, ignore_class] = 0
        plt.imshow(cm_disp, interpolation='nearest', aspect='auto')
        plt.title("Validation Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        _wandb_safe_log({"val/confusion_matrix": wandb.Image(fig)}, step=step, commit=False)
        plt.close(fig)
    except Exception:
        pass

    _wandb_safe_log({"val/metrics_table": metrics_table, **scalar_logs, **per_class_logs}, step=step, commit=True)


# ------------------------------- evaluation -----------------------------------

def evaluate_split_with_metrics(
    state,
    dsets_by_lang: Dict[int, dict],
    L: int,
    batch_size: int,
    batches: int,
    mix_prob: float,
    min_seg: int,
    rng,
    eval_step_fn=None,
) -> Tuple[float, float, np.ndarray]:
    """
    Runs eval over 'batches' mini-batches, returns mean loss, mean acc,
    and a confusion matrix over all non-PAD tokens.
    """
    if eval_step_fn is None:
        raise ValueError("evaluate_split_with_metrics requires eval_step_fn=eval_step")

    losses, accs = [], []
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for i in range(batches):
        data_rng, eval_rng = jax.random.split(jax.random.fold_in(rng, i))
        seed_val = int(jax.random.randint(data_rng, (), 0, 2**31 - 1).item())
        np.random.seed(seed_val)
        xb, yb = _make_eval_batch(dsets_by_lang, L, batch_size, mix_prob, min_seg)

        # Standard eval (loss/acc)
        loss, acc = eval_step_fn(
            state,
            jnp.array(xb, dtype=jnp.int32),
            jnp.array(yb, dtype=jnp.uint8),
            eval_rng,
        )
        losses.append(float(loss))
        accs.append(float(acc))

        # Extra forward for logits -> preds
        logits = _forward_logits(state, jnp.array(xb, dtype=jnp.int32), eval_rng)
        preds = np.asarray(jnp.argmax(logits, axis=-1), dtype=np.int32)
        y_true = yb.astype(np.int32)

        # Mask PAD
        mask = (y_true != PAD_ID)
        if mask.any():
            accumulate_confusion(conf_mat, y_true[mask], preds[mask])

    import numpy as _np
    return float(_np.mean(losses)), float(_np.mean(accs)), conf_mat
