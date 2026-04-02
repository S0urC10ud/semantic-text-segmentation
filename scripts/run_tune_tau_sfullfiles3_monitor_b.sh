#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-$REPO_DIR/evaluation/sweep_tau_real_monitor_b.py}"

CHECKPOINT="${CHECKPOINT:-$REPO_DIR/checkpoints/sweeps/sfullfiles3.msgpack}"
MONITOR_ROOT="${MONITOR_ROOT:-$REPO_DIR/downloader/monitor_preprocessed_b}"
LIMIT_FILES="${LIMIT_FILES:-1024}"
SUBSET_SEED="${SUBSET_SEED:-123}"
SELECT_BY="${SELECT_BY:-other_f1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
DEVICE="${DEVICE:-auto}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/evaluation/reports/tau_sweeps/sfullfiles3__monitor_b_real__${TIMESTAMP}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$SWEEP_SCRIPT" ]]; then
  echo "Sweep script not found: $SWEEP_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "Output dir: $OUT_DIR" >&2
echo "Checkpoint: $CHECKPOINT" >&2
echo "Monitor root: $MONITOR_ROOT" >&2
echo "Subset: limit_files=$LIMIT_FILES subset_seed=$SUBSET_SEED" >&2

exec "$PYTHON_BIN" "$SWEEP_SCRIPT" \
  --checkpoint "$CHECKPOINT" \
  --arch mamba \
  --monitor-root "$MONITOR_ROOT" \
  --limit-files "$LIMIT_FILES" \
  --subset-seed "$SUBSET_SEED" \
  --select-by "$SELECT_BY" \
  --batch-size "$BATCH_SIZE" \
  --chunk "$CHUNK_SIZE" \
  --device "$DEVICE" \
  --inference-backend "$INFERENCE_BACKEND" \
  --out-csv "$OUT_DIR/results.csv" \
  --out-json "$OUT_DIR/results.json" \
  "$@"
