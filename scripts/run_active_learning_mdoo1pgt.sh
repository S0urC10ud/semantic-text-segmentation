#!/usr/bin/env bash
# run_active_learning_mdoo1pgt.sh — Run the active learning meta-loop on the
# GCP VM using the mdoo1pgt (small Mamba fine-tune) checkpoint.
#
# This script:
#   1. Runs active_learning/round.py  — inference + acquisition + Gemini oracle
#   2. Runs train/main.py             — re-trains with oracle-refined segments mixed in
#   3. Repeats for N rounds (via active_learning/meta_trainer.py)
#
# Usage:  ./scripts/run_active_learning_mdoo1pgt.sh
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
PROJECT="magika-segment-feb-2026"
ZONE="europe-west1-c"
INSTANCE="thesis-t4-robust"
REMOTE_USER="martindallinger2002_gmail_com"
REPO_DIR="/home/${REMOTE_USER}/semantic-text-segmentation"

# ── The training script that runs ON the VM ────────────────────
read -r -d '' TRAINING_SCRIPT << 'TRAINING_EOF' || true
#!/usr/bin/env bash
set -uo pipefail

REMOTE_USER="martindallinger2002_gmail_com"
REPO_DIR="/home/${REMOTE_USER}/semantic-text-segmentation"
LOG="/tmp/active_learning.log"

exec > >(tee -a "$LOG") 2>&1

echo "=============================="
echo "🧠 Active Learning Meta-Loop"
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

# Load API keys (GOOGLE_API_KEY etc.) from .env if present
if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    source "${REPO_DIR}/.env"
    set +a
    echo "✅ Loaded environment from .env"
else
    echo "⚠️  No .env file found — GOOGLE_API_KEY must be set elsewhere"
fi

echo "=============================="
echo "🧠 Starting Active Learning Meta-Trainer ..."
echo "  \$(date)"
echo "=============================="

# ── Active Learning Meta-Trainer ──
# Uses the mdoo1pgt checkpoint as the starting point.
# Each round:
#   1. Runs inference → selects uncertain boundaries → Gemini oracle refines them
#   2. Re-trains the model with oracle-refined segments mixed into the training data
#
# Adjust --rounds, --train-steps, --train-max-minutes, --al-mix-prob etc. as needed.

python -m active_learning.meta_trainer \
    --ckpt-path "${REPO_DIR}/checkpoints/sweeps/mdoo1pgt.msgpack" \
    --arch mamba \
    --data-root "${REPO_DIR}/downloader/arrow_out" \
    --rounds 10 \
    --train-steps 600 \
    --al-store "${REPO_DIR}/active_learning/label_store.sqlite" \
    --al-split train \
    --al-oracle gemini \
    --al-gemini-model "gemini-3-flash-preview" \
    --al-gemini-batch-size 32 \
    --al-max-samples-per-lang 16 \
    --al-max-candidates-per-sample 2 \
    --al-min-score 0.5 \
    --al-context-chars 250 \
    --al-max-oracle-requests 3 \
    --al-mix-prob 0.3 \
    --al-mix-full-at-rows 500 \
    --al-max-windows 1000000 \
    --wandb-mode shared \
    --wandb-run-id mdoo1pgt \
    --train-extra-args "--arch mamba --model_dim 256 --mamba_layers 6 --mamba_d_state 16 --mamba_expand 1 --batch_size 32 --accum_steps 1 --lr 2e-5 --num_workers 12 --monitor_eval_limit 512 --monitor_other_threshold 0.5 --fine_tune_use_oe --num-gpus 1"

EXIT_CODE=\$?

echo "=============================="
if [ \$EXIT_CODE -eq 0 ]; then
    echo "✅ Active learning loop finished successfully at \$(date)"
else
    echo "❌ Active learning loop CRASHED (exit code \$EXIT_CODE) at \$(date)"
fi
echo "=============================="
USEREOF

echo "🛑 Shutting down VM in 60 seconds ..."
sleep 60
shutdown -h now
TRAINING_EOF

# ── 1. Set the startup script as instance metadata ────────────
echo "📝 Setting active learning startup script on VM ..."
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
echo "  🎉 VM is booting and will run the active learning loop!"
echo "══════════════════════════════════════════════════════════"
