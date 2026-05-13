#!/usr/bin/env bash
# run_ablation_dataset_modes.sh — Pre-training ablation study on dataset augmentation modes.
#
# Runs 4 sequential ablation variants on the gCloud VM, each capped at 100k steps.
# Every run disables exactly ONE augmentation mode and redistributes its probability
# mass to the remaining modes (keeping relative ratios). All runs log directly to wandb.
#
# Base run: qwy0hvt2  (pure_prob=0.65, mix_prob=0.15, line_inject_prob=0.15, markdown_prob=0.10)
#
# Ablation variants:
#   1. no_pure_windows      — pure_prob=0.0  → remaining mass split over mix/inject/md
#   2. no_mix_concat        — mix_prob=0.0   → remaining mass split over pure/inject/md
#   3. no_line_inject       — line_inject_prob=0.0 → remaining mass split over pure/mix/md
#   4. no_markdown_wrap     — markdown_prob=0.0    → remaining mass split over pure/mix/inject
#
# The VM auto-stops after all 4 runs finish or any run crashes.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
PROJECT="magika-segment-feb-2026"
ZONE="europe-west1-c"
INSTANCE="thesis-l4-robust"
REMOTE_USER="martindallinger2002_gmail_com"
REPO_DIR="/home/${REMOTE_USER}/semantic-text-segmentation"

# ── The training script that runs ON the VM ────────────────────
read -r -d '' TRAINING_SCRIPT << 'TRAINING_EOF' || true
#!/usr/bin/env bash
set -uo pipefail

REMOTE_USER="martindallinger2002_gmail_com"
REPO_DIR="/home/${REMOTE_USER}/semantic-text-segmentation"
LOG="/tmp/ablation_training.log"

# Redirect all output to the log file
exec > >(tee -a "$LOG") 2>&1

echo "=============================="
echo "🚀 VM ablation study startup script"
echo "  Started at: \$(date)"
echo "=============================="

sleep 10

su - "${REMOTE_USER}" << 'USEREOF'
set -uo pipefail

REPO_DIR="/home/martindallinger2002_gmail_com/semantic-text-segmentation"
cd "${REPO_DIR}"
source .venv/bin/activate

echo "📥 Pulling latest code ..."
git pull

# ── Common training arguments ─────────────────────────────────
# These match the base run qwy0hvt2 configuration:
#   arch=unet1d, channels=32,64,64,128,128,128,128,256, model_dim=256
#   batch_size=64, dropout=0.15, weight_decay=0.01
#   eval_every=2000, monitor_eval_every=2000, monitor_eval_limit=4096
#   monitor_other_threshold=0.3, oe_lambda=0.1, oe_ratio=0.05
#   Capped at 100k steps for ablation.

COMMON_ARGS=(
    --arch unet1d
    --channels 32,64,64,128,128,128,128,256
    --model_dim 256
    --batch_size 64
    --accum_steps 1
    --steps 100000
    --lr 1e-3
    --weight_decay 0.01
    --dropout_rate 0.15
    --eval_every 2000
    --eval_batches 50
    --monitor_eval_every 2000
    --monitor_eval_limit 4096
    --monitor_other_threshold 0.3
    --num_workers 4
    --num-gpus 1
)

OVERALL_EXIT=0

# ──────────────────────────────────────────────────────────────
# Ablation 1: No pure windows (pure_prob=0)
#   Redistribute 0.65 over mix(0.15), inject(0.15), md(0.10) → ratios 3:3:2
#   mix=0.15+0.65*(3/8)=0.394, inject=0.15+0.65*(3/8)=0.394, md=0.10+0.65*(2/8)=0.2625
#   → normalized: mix=0.375, inject=0.375, md=0.25
# ──────────────────────────────────────────────────────────────
echo "=============================="
echo "🔬 Ablation 1/4: No pure windows"
echo "  \$(date)"
echo "=============================="

CUDA_VISIBLE_DEVICES=0 python "${REPO_DIR}/train/main.py" \
    "${COMMON_ARGS[@]}" \
    --pure_prob 0.0 \
    --mix_prob 0.375 \
    --line_inject_prob 0.375 \
    --markdown_prob 0.25

ABL1_EXIT=\$?
if [ \$ABL1_EXIT -ne 0 ]; then
    echo "❌ Ablation 1 (no_pure_windows) CRASHED (exit code \$ABL1_EXIT)"
    OVERALL_EXIT=1
else
    echo "✅ Ablation 1 (no_pure_windows) finished successfully"
fi

# ──────────────────────────────────────────────────────────────
# Ablation 2: No mixture concatenation (mix_prob=0)
#   Redistribute 0.15 over pure(0.65), inject(0.15), md(0.10) → ratios 13:3:2
#   pure=0.65+0.15*(13/18)=0.758, inject=0.15+0.15*(3/18)=0.175, md=0.10+0.15*(2/18)=0.117
#   → normalized: pure=0.7222, inject=0.1667, md=0.1111
# ──────────────────────────────────────────────────────────────
echo "=============================="
echo "🔬 Ablation 2/4: No mixture concatenation"
echo "  \$(date)"
echo "=============================="

CUDA_VISIBLE_DEVICES=0 python "${REPO_DIR}/train/main.py" \
    "${COMMON_ARGS[@]}" \
    --pure_prob 0.7222 \
    --mix_prob 0.0 \
    --line_inject_prob 0.1667 \
    --markdown_prob 0.1111

ABL2_EXIT=\$?
if [ \$ABL2_EXIT -ne 0 ]; then
    echo "❌ Ablation 2 (no_mix_concat) CRASHED (exit code \$ABL2_EXIT)"
    OVERALL_EXIT=1
else
    echo "✅ Ablation 2 (no_mix_concat) finished successfully"
fi

# ──────────────────────────────────────────────────────────────
# Ablation 3: No line injection (line_inject_prob=0)
#   Redistribute 0.15 over pure(0.65), mix(0.15), md(0.10) → ratios 13:3:2
#   pure=0.7222, mix=0.1667, md=0.1111
# ──────────────────────────────────────────────────────────────
echo "=============================="
echo "🔬 Ablation 3/4: No line injection"
echo "  \$(date)"
echo "=============================="

CUDA_VISIBLE_DEVICES=0 python "${REPO_DIR}/train/main.py" \
    "${COMMON_ARGS[@]}" \
    --pure_prob 0.7222 \
    --mix_prob 0.1667 \
    --line_inject_prob 0.0 \
    --markdown_prob 0.1111

ABL3_EXIT=\$?
if [ \$ABL3_EXIT -ne 0 ]; then
    echo "❌ Ablation 3 (no_line_inject) CRASHED (exit code \$ABL3_EXIT)"
    OVERALL_EXIT=1
else
    echo "✅ Ablation 3 (no_line_inject) finished successfully"
fi

# ──────────────────────────────────────────────────────────────
# Ablation 4: No synthetic markdown wrapping (markdown_prob=0)
#   Redistribute 0.10 over pure(0.65), mix(0.15), inject(0.15) → ratios 13:3:3
#   pure=0.65+0.10*(13/19)=0.7184, mix=0.15+0.10*(3/19)=0.1658, inject=0.15+0.10*(3/19)=0.1658
#   → normalized: pure=0.6842, mix=0.1579, inject=0.1579
# ──────────────────────────────────────────────────────────────
echo "=============================="
echo "🔬 Ablation 4/4: No synthetic markdown wrapping"
echo "  \$(date)"
echo "=============================="

CUDA_VISIBLE_DEVICES=0 python "${REPO_DIR}/train/main.py" \
    "${COMMON_ARGS[@]}" \
    --pure_prob 0.6842 \
    --mix_prob 0.1579 \
    --line_inject_prob 0.1579 \
    --markdown_prob 0.0

ABL4_EXIT=\$?
if [ \$ABL4_EXIT -ne 0 ]; then
    echo "❌ Ablation 4 (no_markdown_wrap) CRASHED (exit code \$ABL4_EXIT)"
    OVERALL_EXIT=1
else
    echo "✅ Ablation 4 (no_markdown_wrap) finished successfully"
fi

echo "=============================="
echo "📊 Ablation study complete"
echo "  Exit codes: abl1=\$ABL1_EXIT abl2=\$ABL2_EXIT abl3=\$ABL3_EXIT abl4=\$ABL4_EXIT"
if [ \$OVERALL_EXIT -eq 0 ]; then
    echo "  ✅ All 4 ablation runs succeeded!"
else
    echo "  ⚠️  Some runs failed — check wandb for partial results."
fi
echo "  \$(date)"
echo "=============================="
USEREOF

echo "🛑 Shutting down VM in 60 seconds ..."
echo "   To cancel: SSH in and run 'sudo shutdown -c'"
sleep 60
shutdown -h now
TRAINING_EOF

# ── 1. Set the startup script as instance metadata ────────────
echo "📝 Setting ablation startup script on VM ..."
TMP_SCRIPT=$(mktemp)
echo "$TRAINING_SCRIPT" > "$TMP_SCRIPT"
gcloud compute instances add-metadata "${INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --metadata-from-file=startup-script="${TMP_SCRIPT}"
rm -f "$TMP_SCRIPT"

# ── 2. Start (or restart) the VM ─────────────────────────────
STATUS=$(gcloud compute instances describe "${INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --format="value(status)" 2>/dev/null || echo "UNKNOWN")

if [ "${STATUS}" = "RUNNING" ]; then
    echo "🔄 VM is already running — resetting to trigger startup script ..."
    gcloud compute instances reset "${INSTANCE}" \
        --project="${PROJECT}" \
        --zone="${ZONE}"
else
    echo "🚀 Starting VM '${INSTANCE}' ..."
    gcloud compute instances start "${INSTANCE}" \
        --project="${PROJECT}" \
        --zone="${ZONE}"
fi

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  🎉 VM is booting and will run ablation study automatically!"
echo ""
echo "  📋 Ablation variants (100k steps each, sequential):"
echo "     1. no_pure_windows      (pure_prob=0)"
echo "     2. no_mix_concat        (mix_prob=0)"
echo "     3. no_line_inject       (line_inject_prob=0)"
echo "     4. no_markdown_wrap     (markdown_prob=0)"
echo ""
echo "  ✅ All runs log directly to wandb project 'code-segmentation-v2'"
echo "  ✅ VM auto-stops after all runs finish OR any run crashes."
echo ""
echo "  To monitor logs:"
echo "    gcloud compute ssh ${REMOTE_USER}@${INSTANCE} \\"
echo "        --project=${PROJECT} --zone=${ZONE} \\"
echo "        --tunnel-through-iap -- tail -f /tmp/ablation_training.log"
echo ""
echo "  To check VM status:"
echo "    gcloud compute instances describe ${INSTANCE} \\"
echo "        --project=${PROJECT} --zone=${ZONE} \\"
echo "        --format='value(status)'"
echo "══════════════════════════════════════════════════════════"
