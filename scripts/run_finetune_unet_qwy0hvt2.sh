#!/usr/bin/env bash
# run_finetune_unet_qwy0hvt2.sh — Fine-tune U-Net from checkpoint qwy0hvt2.
#
# The training script is injected as instance metadata and runs on boot.
# No persistent SSH connection is needed. The VM ALWAYS stops itself
# after training finishes or crashes — even if you never SSH in at all.
#
# Source model: U-Net run qwy0hvt2 (pre-trained, 200k steps)
# Fine-tunes on: monitor_preprocessed_a  →  validated on monitor_preprocessed_b

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
PROJECT="magika-segment-feb-2026"
ZONE="europe-west1-c"
INSTANCE="thesis-l4-robust"
REMOTE_USER="REDACTED"
REPO_DIR="/home/${REMOTE_USER}/semantic-text-segmentation"

# ── The training script that runs ON the VM ────────────────────
read -r -d '' TRAINING_SCRIPT << 'TRAINING_EOF' || true
#!/usr/bin/env bash
set -uo pipefail

REMOTE_USER="REDACTED"
REPO_DIR="/home/${REMOTE_USER}/semantic-text-segmentation"
LOG="/tmp/training.log"

# Redirect all output to the log file
exec > >(tee -a "$LOG") 2>&1

echo "=============================="
echo "🚀 VM fine-tune startup script"
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
echo "🎯 Fine-tuning U-Net from qwy0hvt2 ..."
echo "  \$(date)"
echo "=============================="

CUDA_VISIBLE_DEVICES=0 python "${REPO_DIR}/train/main.py" \
    --arch unet1d \
    --channels 32,64,64,128,128,128,128,256 \
    --model_dim 256 \
    --batch_size 64 \
    --accum_steps 1 \
    --steps 50000 \
    --eval_every 2500 \
    --monitor_eval_every 2500 \
    --monitor_eval_limit 4096 \
    --monitor_other_threshold 0.3 \
    --dropout_rate 0.15 \
    --weight_decay 0.01 \
    --num_workers 4 \
    --num-gpus 1 \
    --fine-tune qwy0hvt2

EXIT_CODE=\$?

echo "=============================="
if [ \$EXIT_CODE -eq 0 ]; then
    echo "✅ Fine-tuning finished successfully at \$(date)"
else
    echo "❌ Fine-tuning CRASHED (exit code \$EXIT_CODE) at \$(date)"
fi
echo "=============================="
USEREOF

echo "🛑 Shutting down VM in 60 seconds ..."
echo "   To cancel: SSH in and run 'sudo shutdown -c'"
sleep 60
shutdown -h now
TRAINING_EOF

# ── 1. Set the startup script as instance metadata ────────────
echo "📝 Setting fine-tune startup script on VM ..."
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
echo "  🎉 VM is booting and will run fine-tuning automatically!"
echo ""
echo "  ✅ No SSH connection needed — the VM runs independently."
echo "  ✅ VM auto-stops after fine-tuning finishes OR crashes."
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
echo "══════════════════════════════════════════════════════════"
