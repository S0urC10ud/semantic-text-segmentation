#!/usr/bin/env bash
# run_finetune.sh — Start the GCP VM with a self-contained training job.
#
# The training script is injected as instance metadata and runs on boot.
# No persistent SSH connection is needed. The VM ALWAYS stops itself
# after training completes or crashes — even if you never SSH in at all.
#
# Usage:  ./scripts/run_finetune.sh
#
# To monitor:
#   gcloud compute ssh REDACTED@thesis-l4-2 \
#       --project=magika-segment-feb-2026 --zone=europe-west1-c \
#       --tunnel-through-iap -- tail -f /tmp/training.log
#
# Or attach to the tmux session:
#   gcloud compute ssh REDACTED@thesis-l4-2 \
#       --project=magika-segment-feb-2026 --zone=europe-west1-c \
#       --tunnel-through-iap -- tmux attach -t train
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
PROJECT="magika-segment-feb-2026"
ZONE="europe-west1-c"
INSTANCE="thesis-l4-2"
REMOTE_USER="REDACTED"
REPO_DIR="/home/${REMOTE_USER}/semantic-text-segmentation"

# ── The training script that runs ON the VM ────────────────────
# This is set as instance metadata and executed by the startup script.
# It runs completely independently of any SSH session.
read -r -d '' TRAINING_SCRIPT << 'TRAINING_EOF' || true
#!/usr/bin/env bash
set -uo pipefail

REMOTE_USER="REDACTED"
REPO_DIR="/home/${REMOTE_USER}/semantic-text-segmentation"
LOG="/tmp/training.log"

# Redirect all output to the log file
exec > >(tee -a "$LOG") 2>&1

echo "=============================="
echo "🚀 VM training startup script"
echo "  Started at: $(date)"
echo "=============================="

# Wait for the user environment to be ready
sleep 10

# Run everything as the actual user (startup scripts run as root)
su - "${REMOTE_USER}" << 'USEREOF'
set -uo pipefail

REPO_DIR="/home/REDACTED/semantic-text-segmentation"
cd "${REPO_DIR}"
source .venv/bin/activate

echo "📥 Pulling latest code ..."
git pull

echo "=============================="
echo "🏋️ Starting fine-tuning ..."
echo "  $(date)"
echo "=============================="

python "${REPO_DIR}/train/main.py" \
    --arch mamba --model_dim 256 --mamba_layers 6 --mamba_d_state 16 --mamba_expand 2 \
    --batch_size 4 --accum_steps 4 --lr 2e-5 --steps 150000 --eval_every 250 \
    --monitor_other_threshold 0.5 --num_workers 12 --monitor_eval_limit 512 \
    --fine-tune "${REPO_DIR}/checkpoints/sweeps/el33oo1u-20000.msgpack" --fine_tune_use_oe \
    --continue bobz5tt6

EXIT_CODE=$?

echo "=============================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training finished successfully at $(date)"
else
    echo "❌ Training CRASHED (exit code $EXIT_CODE) at $(date)"
fi
echo "=============================="
USEREOF

echo "🛑 Shutting down VM in 60 seconds ..."
echo "   To cancel: SSH in and run 'sudo shutdown -c'"
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
echo "  🎉 VM is booting and will run training automatically!"
echo ""
echo "  ✅ No SSH connection needed — the VM runs independently."
echo "  ✅ VM auto-stops after training finishes OR crashes."
echo ""
echo "  To monitor logs:"
echo "    gcloud compute ssh ${REMOTE_USER}@${INSTANCE} \\"
echo "        --project=${PROJECT} --zone=${ZONE} \\"
echo "        --tunnel-through-iap -- tail -f /tmp/training.log"
echo ""
echo "  To check VM status:"
echo "    gcloud compute instances describe ${INSTANCE} \\"
echo "        --project=${PROJECT} --zone=${ZONE} \\"
echo "        --format='value(status)'"
echo ""
echo "  To cancel the auto-shutdown (if training succeeded and you"
echo "  want to keep the VM running):"
echo "    gcloud compute ssh ${REMOTE_USER}@${INSTANCE} \\"
echo "        --project=${PROJECT} --zone=${ZONE} \\"
echo "        --tunnel-through-iap -- sudo shutdown -c"
echo "══════════════════════════════════════════════════════════"
