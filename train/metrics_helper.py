"""
Metrics helpers for per-class and aggregated accuracy / precision / recall / F1,
plus nice console tables, a high-res Sankey diagram,
and Weights & Biases logging.
"""
from typing import Dict, Tuple
import os
import random
import numpy as np
import jax
import jax.numpy as jnp
import wandb

from config import DataConfig, NUM_CLASSES, PAD_ID, PAD_BYTE_ID
from window_generator import make_training_window

# Ensure any later matplotlib usage in this process prefers a headless backend.
# (This is belt-and-suspenders; we don't import pyplot anywhere below.)
os.environ.setdefault("MPLBACKEND", "Agg")

# --- Matplotlib compatibility shim -------------------------------------------
# Some downstream libs (or environments) may access `matplotlib.pyplot`
# attribute directly. Ensure it's importable and set to a headless backend.
try:
    import matplotlib as _matplotlib
    _matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as _plt  # noqa: F401
except Exception as _e:
    print(f"Matplotlib setup warning (safe to ignore if you don't need plots): {_e}", flush=True)


# ----------------------------- batch building ---------------------------------

def _make_eval_batch(
    dsets_by_lang: Dict[int, dict],
    L: int,
    batch_size: int,
    cfg: DataConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an eval batch mirroring the data pipeline logic."""
    xb = np.full((batch_size, L), PAD_BYTE_ID, dtype=np.int32)
    yb = np.full((batch_size, L), PAD_ID, dtype=np.uint8)
    for i in range(batch_size):
        x, y = make_training_window(dsets_by_lang, L, cfg)
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


# ----------------------------- W&B logging + Sankey ---------------------------

def _wandb_safe_log(data: dict, step: int, commit: bool = True):
    try:
        wandb.log(data, step=step, commit=commit)
    except Exception as e:
        print(f"Wandb logging failed: {e}", flush=True)


def _save_sankey_from_confusion(
    conf_mat: np.ndarray,
    id2label: Dict[int, str],
    num_classes: int,
    ignore_class: int,
    step: int,
) -> str:
    """
    Create a high-res horizontal Sankey (true → predicted) as a PNG and return its path.
    - Always overwrites the same file for the current run: sankey_{wandb.run.id}.png
    - Ignores PAD / 'ignore_class'.
    - Shows all flows; very tiny flows are pruned for readability.
    """
    # Headless, explicit Agg rendering (no pyplot, no GUI backend)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.patches import Rectangle
    from matplotlib.cm import get_cmap

    cm = conf_mat.copy()
    if ignore_class is not None and 0 <= ignore_class < num_classes:
        cm[ignore_class, :] = 0
        cm[:, ignore_class] = 0

    # If nothing to show, bail
    total = float(cm.sum())
    if total <= 0:
        return ""

    # Normalize to fractions for geometry, but keep counts for labels.
    C = num_classes
    support_true = cm.sum(axis=1).astype(float)
    support_pred = cm.sum(axis=0).astype(float)

    # Build class indices to display (skip zero-support classes)
    idx_true = [i for i in range(C) if i != ignore_class and support_true[i] > 0]
    idx_pred = [j for j in range(C) if j != ignore_class and support_pred[j] > 0]

    # Edge list (i -> j) with values
    flows = []
    for i in idx_true:
        for j in idx_pred:
            v = float(cm[i, j])
            if v > 0:
                flows.append((i, j, v))

    # Prune minuscule flows for visual clarity (keep at least top-K per true class)
    min_frac = 0.002  # 0.2% of all tokens
    keep = []
    by_src = {}
    for i, j, v in flows:
        by_src.setdefault(i, []).append((i, j, v))
    for i, lst in by_src.items():
        lst_sorted = sorted(lst, key=lambda x: x[2], reverse=True)
        for k, (ii, jj, vv) in enumerate(lst_sorted):
            if vv / total >= min_frac or k < 3:
                keep.append((ii, jj, vv))
    flows = keep

    # Layout parameters
    left_x = 0.05
    right_x = 0.95
    mid_x0 = 0.42
    mid_x1 = 0.58
    node_width = 0.02

    # Vertical stacking for left (true) and right (pred)
    y_gap = 0.008
    y_margin = 0.02  # use symmetric top/bottom margins

    # Heights (fractions). We'll scale them to fit inside [y_margin, 1 - y_margin].
    left_heights_raw = [support_true[i] / total for i in idx_true]
    right_heights_raw = [support_pred[j] / total for j in idx_pred]

    def stack_positions(heights, n_nodes):
        # scale to available vertical space taking gaps into account
        available = 1.0 - 2.0 * y_margin - y_gap * max(n_nodes - 1, 0)
        total_h = sum(heights) if len(heights) > 0 else 1.0
        scale = available / total_h if total_h > 0 else 0.0
        y = y_margin
        pos = []
        for h in heights:
            hh = h * scale
            pos.append((y, y + hh))
            y = y + hh + y_gap
        return pos, scale

    left_pos, left_scale = stack_positions(left_heights_raw, len(idx_true))
    right_pos, right_scale = stack_positions(right_heights_raw, len(idx_pred))

    # Running offsets inside each node for ribbons (scaled to side)
    left_offsets = {i: 0.0 for i in idx_true}
    right_offsets = {j: 0.0 for j in idx_pred}

    # Map class -> color (consistent, bright)
    tab = get_cmap("tab20")
    def class_color(k):
        return tab((hash(k) % 20) / 20.0)

    # Build index maps for quick lookups
    idx_true_to_order = {i: n for n, i in enumerate(idx_true)}
    idx_pred_to_order = {j: n for n, j in enumerate(idx_pred)}

    # Prepare figure (explicit Agg canvas, no pyplot)
    fig = Figure(figsize=(18, 10), dpi=220)
    _ = FigureCanvas(fig)  # attaches an Agg canvas to the figure
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Draw nodes (left true, right pred)
    for i, (y0, y1) in zip(idx_true, left_pos):
        h = y1 - y0
        rect = Rectangle((left_x - node_width/2, y0), node_width, h,
                         facecolor=(0, 0, 0, 0.05), edgecolor=(0, 0, 0, 0.25), linewidth=1.0)
        ax.add_patch(rect)
        label = id2label.get(i, str(i))
        ax.text(left_x - node_width/2 - 0.01, (y0 + y1)/2, f"{label}\n{int(support_true[i])}",
                va="center", ha="right", fontsize=8.5, fontweight="600")

    for j, (y0, y1) in zip(idx_pred, right_pos):
        h = y1 - y0
        rect = Rectangle((right_x - node_width/2, y0), node_width, h,
                         facecolor=(0, 0, 0, 0.05), edgecolor=(0, 0, 0, 0.25), linewidth=1.0)
        ax.add_patch(rect)
        label = id2label.get(j, str(j))
        ax.text(right_x + node_width/2 + 0.01, (y0 + y1)/2, f"{label}\n{int(support_pred[j])}",
                va="center", ha="left", fontsize=8.5, fontweight="600")

    # Helper: build a ribbon as a closed Path between (left segment) and (right segment)
    def add_ribbon(x0, ya0, yb0, x1, ya1, yb1, color, alpha):
        # Sample top and bottom curves with a smooth cubic Bezier
        def bezier(t, p0, p1, p2, p3):
            return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3

        T = 20  # smoothness
        ts = np.linspace(0, 1, T)
        # Top curve
        top_x = bezier(ts, x0, mid_x0, mid_x1, x1)
        top_y = bezier(ts, ya0, ya0, ya1, ya1)
        # Bottom curve
        bot_x = bezier(ts, x1, mid_x1, mid_x0, x0)
        bot_y = bezier(ts, yb1, yb1, yb0, yb0)

        xs = np.concatenate([top_x, bot_x])
        ys = np.concatenate([top_y, bot_y])

        from matplotlib.path import Path  # local import remains
        codes = [Path.MOVETO] + [Path.CURVE4] * (T - 1) + [Path.CURVE4] * T + [Path.CLOSEPOLY]
        verts = np.column_stack([np.append(xs, xs[0]), np.append(ys, ys[0])])

        path = Path(verts, codes)
        from matplotlib.patches import PathPatch  # local import remains
        patch = PathPatch(path, facecolor=color, edgecolor=(0, 0, 0, 0.12), linewidth=0.4, alpha=alpha, antialiased=True)
        ax.add_patch(patch)

    # Draw ribbons (source color, thicker for mispred)
    for i, j, v in flows:
        li = idx_true_to_order[i]
        rj = idx_pred_to_order[j]
        y0a, y0b = left_pos[li]
        y1a, y1b = right_pos[rj]

        # Determine segment within each node (scaled separately per side)
        src_h = (v / total) * left_scale
        dst_h = (v / total) * right_scale

        ya0 = y0a + left_offsets[i]
        yb0 = ya0 + src_h
        left_offsets[i] += src_h

        ya1 = y1a + right_offsets[j]
        yb1 = ya1 + dst_h
        right_offsets[j] += dst_h

        col = class_color(i)
        is_correct = (i == j)
        alpha = 0.35 if is_correct else 0.85

        add_ribbon(left_x + node_width/2, ya0, yb0, right_x - node_width/2, ya1, yb1, col, alpha)

    # Title / subtitle
    run_id = getattr(wandb.run, "id", "run")
    acc_line = ""
    acc_val = None
    if getattr(wandb, "run", None) is not None and wandb.run is not None:
        try:
            acc_val = wandb.run.summary.get("val/agg/micro/accuracy", None)
        except Exception:
            acc_val = None
    if isinstance(acc_val, (int, float)):
        acc_line = f" • Acc {acc_val:.3f}"

    ax.text(0.5, 1.06, f"Flow: True → Predicted  (Run {run_id}{acc_line})",
            ha="center", va="bottom", fontsize=12.5, fontweight="700", transform=ax.transAxes)
    ax.text(0.5, 1.03, f"Step {step} • Width ∝ byte/token count • Node labels show byte counts",
            ha="center", va="top", fontsize=9.0, color=(0, 0, 0, 0.65), transform=ax.transAxes)

    # Save — overwrite per run id
    out_path = f"sankey_images/sankey_{run_id}_{step}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.clear()
    return out_path


def wandb_log_metrics(
    step: int,
    per_class,
    aggregates,
    id2label: Dict[int, str],
    num_classes: int,
    ignore_class: int,
    conf_mat: np.ndarray,
):
    """Log a wandb.Table, aggregated scalars, per-class scalars, a confusion-matrix image, and a Sankey diagram."""
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

    # 4) Confusion matrix image (Agg, no pyplot, fail hard on errors)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = Figure(figsize=(6, 5))
    _ = FigureCanvas(fig)  # attach Agg canvas
    ax = fig.add_subplot(111)
    cm_disp = conf_mat.copy()
    if ignore_class is not None and 0 <= ignore_class < num_classes:
        cm_disp[ignore_class, :] = 0
        cm_disp[:, ignore_class] = 0
    im = ax.imshow(cm_disp, interpolation='nearest', aspect='auto')
    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    
    #_wandb_safe_log({"val/confusion_matrix": wandb.Image(fig)}, step=step, commit=False)
    fig.clear()

    # 5) Sankey diagram (single file per run, overwritten) — fail on errors if any
    sankey_path = _save_sankey_from_confusion(conf_mat, id2label, num_classes, ignore_class, step)
    #if sankey_path:
        #wandb.log({"val/sankey_diagram": wandb.Image(sankey_path)}, step=step, commit=False)

    
    _wandb_safe_log({"val/metrics_table": metrics_table, **scalar_logs, **per_class_logs}, step=step, commit=True)


# ------------------------------- evaluation -----------------------------------

def evaluate_split_with_metrics(
    state,
    dsets_by_lang: Dict[int, dict],
    L: int,
    batch_size: int,
    batches: int,
    data_cfg: DataConfig,
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
        random.seed(seed_val)
        xb, yb = _make_eval_batch(dsets_by_lang, L, batch_size, data_cfg)

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
