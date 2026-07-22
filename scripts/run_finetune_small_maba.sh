#!/usr/bin/env bash
# run_finetune_small_maba.sh — Start the GCP VM for FINE-TUNING the small Mamba model.
#
# Usage:  ./scripts/run_finetune_small_maba.sh
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
PROJECT="magika-segment-feb-2026"
ZONE="europe-west1-c"
INSTANCE="thesis-l4-2"
REMOTE_USER="REDACTED"
REPO_DIR="/home/${REMOTE_USER}/semantic-text-segmentation"

# ── The training script that runs ON the VM ────────────────────
read -r -d '' TRAINING_SCRIPT << 'TRAINING_EOF' || true
#!/usr/bin/env bash
set -uo pipefail

REMOTE_USER="REDACTED"
REPO_DIR="/home/${REMOTE_USER}/semantic-text-segmentation"
LOG="/tmp/training.log"

exec > >(tee -a "$LOG") 2>&1

echo "=============================="
echo "🚀 VM fine-tuning startup script"
echo "  Started at: \$(date)"
echo "=============================="

sleep 10

su - "${REMOTE_USER}" << 'USEREOF'
set -uo pipefail

REPO_DIR="/home/REDACTED/semantic-text-segmentation"
cd "${REPO_DIR}"
source .venv/bin/activate

echo "📥 Pulling latest code ..."
git pull

echo "=============================="
echo "🏋️ Starting FINE-TUNING small Mamba ..."
echo "  \$(date)"
echo "=============================="

# IMPORTANT: REPLACE iddg3pz9 with the actual msgpack file BEFORE running!
# Example: "${REPO_DIR}/checkpoints/sweeps/run_id-150000.msgpack"
python "${REPO_DIR}/train/main.py" \
    --arch mamba --model_dim 256 --mamba_layers 6 --mamba_d_state 16 --mamba_expand 1 \
    --batch_size 32 --accum_steps 1 --lr 2e-5 --steps 150000 --eval_every 250 \
    --monitor_other_threshold 0.5 --num_workers 12 --monitor_eval_limit 512 \
    --fine-tune "${REPO_DIR}/checkpoints/sweeps/iddg3pz9.msgpack" --fine_tune_use_oe \
    --continue mdoo1pgt

EXIT_CODE=\$?

echo "=============================="
if [ \$EXIT_CODE -eq 0 ]; then
    echo "✅ Training finished successfully at \$(date)"
else
    echo "❌ Training CRASHED (exit code \$EXIT_CODE) at \$(date)"
fi
echo "=============================="
USEREOF

echo "🛑 Shutting down VM in 60 seconds ..."
sleep 60
shutdown -h now
TRAINING_EOF

# ── 1. Set the startup script as instance metadata ────────────
echo "📝 Setting training startup script on VM ..."
gcloud compute instances add-metadata "${INSTANCE}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --metadata=startup-script="${TRAINING_SCRIPT}"

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
echo "  🎉 VM is booting and will run fine-tuning automatically!"
echo "══════════════════════════════════════════════════════════"
