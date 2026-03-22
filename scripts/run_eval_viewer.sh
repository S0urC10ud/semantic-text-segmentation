#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8007}"
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/evaluation/data}"
DEVICE="${DEVICE:-auto}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"

"$REPO_DIR/.venv/bin/python" "$REPO_DIR/viewers/evaluation_viewer.py" \
  --data-root "$DATA_ROOT" \
  --checkpoint=checkpoints/sweeps/sfullfiles3.msgpack \
  --device "$DEVICE" \
  --inference-backend "$INFERENCE_BACKEND" \
  --host "$HOST" \
  --port "$PORT"
