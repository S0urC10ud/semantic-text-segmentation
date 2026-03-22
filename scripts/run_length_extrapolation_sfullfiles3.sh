#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
MONITOR_ROOT="${MONITOR_ROOT:-$REPO_DIR/downloader/monitor_preprocessed_b}"
CHECKPOINT="${CHECKPOINT:-$REPO_DIR/checkpoints/sweeps/sfullfiles3.msgpack}"
JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
DEVICE="${DEVICE:-cpu}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"
SEED_FILE_IDX="${SEED_FILE_IDX:-2767}"
OUTPUT_JSON="${OUTPUT_JSON:-$REPO_DIR/evaluation/length_extrapolation_sfullfiles3_html_js.json}"
OUTPUT_MD="${OUTPUT_MD:-$REPO_DIR/evaluation/length_extrapolation_sfullfiles3_html_js.md}"

JAX_PLATFORMS="$JAX_PLATFORMS" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$REPO_DIR/.venv/bin/python" "$REPO_DIR/evaluation/length_extrapolation_experiment.py" \
  --monitor-root "$MONITOR_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --arch mamba \
  --model-dim 256 \
  --mamba-layers 6 \
  --mamba-d-state 16 \
  --mamba-expand 1 \
  --mamba-dt-rank 16 \
  --mamba-conv 4 \
  --mamba-bidirectional \
  --chunk 10000 \
  --batch-size 8 \
  --device "$DEVICE" \
  --inference-backend "$INFERENCE_BACKEND" \
  --target-bytes 10000 \
  --size-tolerance 512 \
  --min-minority-share 0.02 \
  --max-candidate-segments 4 \
  --required-labels html javascript_typescript \
  --seed-file-idx "$SEED_FILE_IDX" \
  --repeat-factors 1,2,4,8,16,32 \
  --other-threshold 0.85 \
  --output-json "$OUTPUT_JSON" \
  --output-md "$OUTPUT_MD" \
  "$@"
