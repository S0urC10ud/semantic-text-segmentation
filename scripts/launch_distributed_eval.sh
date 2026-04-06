#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-magika-segment-feb-2026}"
ZONE="${ZONE:-europe-west1-c}"
REMOTE_USER="${REMOTE_USER:-martindallinger2002_gmail_com}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/${REMOTE_USER}/semantic-text-segmentation}"
AGGREGATOR_INSTANCE="${AGGREGATOR_INSTANCE:-thesis-l4-robust}"
USE_IAP="${USE_IAP:-1}"
DRY_RUN="${DRY_RUN:-0}"
STOP_TRAINING_AFTER_SYNC="${STOP_TRAINING_AFTER_SYNC:-1}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-30}"

OTHER_THRESHOLD="${OTHER_THRESHOLD:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
DEVICE="${DEVICE:-auto}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-auto}"
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
UNET_CHUNK_SIZE="${UNET_CHUNK_SIZE:-1536}"
EVAL_MODE="${EVAL_MODE:-fine_tuned}"
DATA_ROOT="${DATA_ROOT:-}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"

TRAINING_L4_2_IDS="${TRAINING_L4_2_IDS:-mamba_full_files_scratch_r0warm_v1,04eyv59a,qwy0hvt2}"
TRAINING_L4_3_IDS="${TRAINING_L4_3_IDS:-sfullfiles4,unet_al_scratch_r0warm_v1,iddg3pz9}"
TRAINING_L4_4_IDS="${TRAINING_L4_4_IDS:-mamba_full_files_pretrain_10k_v1,unet_al_without_ft,mdoo1pgt}"
THESIS_L4_ROBUST_IDS="${THESIS_L4_ROBUST_IDS:-sfullfiles3,mamba_full_files_from_pt,unet_al_ft_redo1}"

COMMON_ARGS=("$@")

DEFAULT_DATA_ROOT_REL="evaluation/data"
FINE_TUNED_DATA_ROOT_REL="evaluation/data_b"

case "$EVAL_MODE" in
  fine_tuned)
    if [[ -n "$DATA_ROOT" ]]; then
      DATA_KEY="$(basename "$DATA_ROOT")"
    else
      DATA_KEY="$(basename "$FINE_TUNED_DATA_ROOT_REL")"
    fi
    ;;
  non_fine_tuned|legacy)
    if [[ -n "$DATA_ROOT" ]]; then
      DATA_KEY="$(basename "$DATA_ROOT")"
    else
      DATA_KEY="$(basename "$DEFAULT_DATA_ROOT_REL")"
    fi
    ;;
  auto)
    if [[ -n "$DATA_ROOT" ]]; then
      DATA_KEY="$(basename "$DATA_ROOT")"
    else
      DATA_KEY="$(basename "$FINE_TUNED_DATA_ROOT_REL")"
    fi
    ;;
  *)
    echo "Unsupported EVAL_MODE='$EVAL_MODE' (expected: auto, fine_tuned, non_fine_tuned, legacy)" >&2
    exit 1
    ;;
esac

TAU_KEY="tau$(printf '%s' "$OTHER_THRESHOLD" | tr '.' 'p')"
REPORTS_REL="evaluation/reports_distributed/$TIMESTAMP"
LOGS_REL="$REPORTS_REL/logs"
STATUS_REL=".distributed_eval/$TIMESTAMP"
FINAL_CSV_REL="$REPORTS_REL/model_report_matrix__${DATA_KEY}__${TAU_KEY}__${TIMESTAMP}.csv"

INSTANCES=(
  "training-l4-2"
  "training-l4-3"
  "training-l4-4"
  "thesis-l4-robust"
)

GCLOUD_BASE_ARGS=(
  --project="$PROJECT"
  --zone="$ZONE"
  --quiet
  --ssh-flag=-T
)

if [[ "$USE_IAP" == "1" ]]; then
  GCLOUD_BASE_ARGS+=(--tunnel-through-iap)
fi

TEMP_FILES=()

cleanup_temp_files() {
  local path=""
  for path in "${TEMP_FILES[@]}"; do
    [[ -n "$path" && -e "$path" ]] || continue
    rm -f "$path"
  done
}
trap cleanup_temp_files EXIT

log() {
  printf '%s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    die "Required command not found: $cmd"
  fi
}

remote_status_root_abs() {
  printf '%s/%s\n' "$REMOTE_REPO_DIR" "$STATUS_REL"
}

remote_reports_root_abs() {
  printf '%s/%s\n' "$REMOTE_REPO_DIR" "$REPORTS_REL"
}

remote_logs_root_abs() {
  printf '%s/%s\n' "$REMOTE_REPO_DIR" "$LOGS_REL"
}

worker_script_name() {
  local instance="$1"
  printf 'worker_%s.sh\n' "$instance"
}

remote_worker_script_abs() {
  local instance="$1"
  printf '%s/%s\n' "$(remote_status_root_abs)" "$(worker_script_name "$instance")"
}

worker_stdout_log_rel() {
  local instance="$1"
  printf '%s/%s__worker__%s.log\n' "$LOGS_REL" "$instance" "$TIMESTAMP"
}

worker_stdout_log_abs() {
  local instance="$1"
  printf '%s/%s\n' "$REMOTE_REPO_DIR" "$(worker_stdout_log_rel "$instance")"
}

instance_requested_ids() {
  local instance="$1"
  case "$instance" in
    training-l4-2)
      printf '%s\n' "$TRAINING_L4_2_IDS"
      ;;
    training-l4-3)
      printf '%s\n' "$TRAINING_L4_3_IDS"
      ;;
    training-l4-4)
      printf '%s\n' "$TRAINING_L4_4_IDS"
      ;;
    thesis-l4-robust)
      printf '%s\n' "$THESIS_L4_ROBUST_IDS"
      ;;
    *)
      die "Unknown instance: $instance"
      ;;
  esac
}

instance_is_training() {
  local instance="$1"
  [[ "$instance" != "$AGGREGATOR_INSTANCE" ]]
}

model_checkpoint_candidates() {
  local requested_id="$1"
  case "$requested_id" in
    unet_al_ft_redo1)
      printf 'unet_al_ft_redo1\n'
      ;;
    unet_al_without_ft)
      printf 'unet_al_without_ft-13800\n'
      ;;
    unet_al_scratch_r0warm_v1)
      printf 'unet_al_scratch_r0warm_v1\n'
      ;;
    sfullfiles3)
      printf 'sfullfiles3\n'
      ;;
    sfullfiles4)
      printf 'sfullfiles4\n'
      ;;
    mamba_full_files_from_pt)
      printf 'mamba_full_files_from_pt\n'
      ;;
    mamba_full_files_scratch_r0warm_v1)
      printf 'mamba_full_files_scratch_r0warm_v1\n'
      ;;
    mamba_full_files_pretrain_10k_v1)
      printf 'mamba_full_files_pretrain_10k_v1\n'
      ;;
    04eyv59a)
      printf '04eyv59a\n'
      ;;
    mdoo1pgt)
      printf 'mdoo1pgt\n'
      ;;
    qwy0hvt2)
      printf 'qwy0hvt2\n'
      ;;
    iddg3pz9)
      printf 'iddg3pz9\n'
      ;;
    *)
      die "Unknown requested id: $requested_id"
      ;;
  esac
}

ordered_requested_ids_for_instance() {
  local instance="$1"
  local requested_ids_csv=""
  local requested_id=""
  local -a requested_ids=()

  requested_ids_csv="$(instance_requested_ids "$instance")"
  IFS=',' read -r -a requested_ids <<< "$requested_ids_csv"
  for requested_id in "${requested_ids[@]}"; do
    requested_id="${requested_id//[[:space:]]/}"
    [[ -n "$requested_id" ]] || continue
    printf '%s\n' "$requested_id"
  done
}

instance_report_rel_paths() {
  local instance="$1"
  local requested_id=""
  while IFS= read -r requested_id; do
    [[ -n "$requested_id" ]] || continue
    printf '%s/%s__%s__%s__%s\n' "$REPORTS_REL" "$requested_id" "$DATA_KEY" "$TAU_KEY" "$TIMESTAMP"
  done < <(ordered_requested_ids_for_instance "$instance")
}

instance_log_rel_paths() {
  local instance="$1"
  local requested_id=""
  while IFS= read -r requested_id; do
    [[ -n "$requested_id" ]] || continue
    printf '%s/%s__%s__%s.log\n' "$LOGS_REL" "$requested_id" "$TAU_KEY" "$TIMESTAMP"
  done < <(ordered_requested_ids_for_instance "$instance")
  worker_stdout_log_rel "$instance"
}

instance_status() {
  local instance="$1"
  gcloud compute instances describe "$instance" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(status)'
}

gcloud_ssh() {
  local instance="$1"
  local remote_script="$2"
  gcloud compute ssh "${REMOTE_USER}@${instance}" "${GCLOUD_BASE_ARGS[@]}" \
    --command "bash -lc $(printf '%q' "$remote_script")"
}

print_summary() {
  local instance=""
  log "Shared timestamp: $TIMESTAMP"
  log "Aggregator: ${AGGREGATOR_INSTANCE}:${REMOTE_REPO_DIR}/${REPORTS_REL}"
  log "Final CSV: ${AGGREGATOR_INSTANCE}:${REMOTE_REPO_DIR}/${FINAL_CSV_REL}"
  for instance in "${INSTANCES[@]}"; do
    log "Node ${instance}: $(instance_requested_ids "$instance")"
    log "  worker script: $(remote_worker_script_abs "$instance")"
  done
}

build_preflight_remote_script() {
  local instance="$1"
  local requested_id=""
  local candidate=""
  local -a candidates=()

  cat <<EOF
set -euo pipefail
cd $(printf '%q' "$REMOTE_REPO_DIR")
[[ -x .venv/bin/python ]] || { echo "Missing executable python: .venv/bin/python" >&2; exit 1; }
[[ -f evaluation/evaluation.py ]] || { echo "Missing evaluation script: evaluation/evaluation.py" >&2; exit 1; }
[[ -d evaluation/data_b ]] || { echo "Missing fine-tuned dataset root: evaluation/data_b" >&2; exit 1; }
[[ ! -e $(printf '%q' "$REPORTS_REL") ]] || { echo "Run reports path already exists: $REPORTS_REL" >&2; exit 1; }
[[ ! -e $(printf '%q' "$STATUS_REL") ]] || { echo "Run status path already exists: $STATUS_REL" >&2; exit 1; }
check_candidates() {
  local requested_id="\$1"
  shift
  local candidate=""
  for candidate in "\$@"; do
    if [[ -e "checkpoints/sweeps/\${candidate}.msgpack" ]]; then
      return 0
    fi
  done
  echo "Missing checkpoint for \${requested_id}. Tried: \$*" >&2
  exit 1
}
EOF

  while IFS= read -r requested_id; do
    [[ -n "$requested_id" ]] || continue
    candidates=()
    while IFS= read -r candidate; do
      [[ -n "$candidate" ]] || continue
      candidates+=("$candidate")
    done < <(model_checkpoint_candidates "$requested_id")
    printf 'check_candidates %q' "$requested_id"
    for candidate in "${candidates[@]}"; do
      printf ' %q' "$candidate"
    done
    printf '\n'
  done < <(ordered_requested_ids_for_instance "$instance")
}

preflight_instance() {
  local instance="$1"
  local status=""
  log "Preflight: ${instance}"
  status="$(instance_status "$instance")"
  [[ "$status" == "RUNNING" ]] || die "Instance ${instance} is not RUNNING (status=${status:-unknown})"
  gcloud_ssh "$instance" "$(build_preflight_remote_script "$instance")" >/dev/null
}

build_remote_worker_script() {
  local instance="$1"
  local requested_ids_csv="$2"
  local common_arg=""

  cat <<EOF
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(printf '%q' "$REMOTE_REPO_DIR")
PYTHON_BIN="\${PYTHON_BIN:-\$REPO_DIR/.venv/bin/python}"
EVAL_SCRIPT="\${EVAL_SCRIPT:-\$REPO_DIR/evaluation/evaluation.py}"
DEFAULT_DATA_ROOT="\$REPO_DIR/evaluation/data"
FINE_TUNED_DATA_ROOT="\$REPO_DIR/evaluation/data_b"
DATA_ROOT=$(printf '%q' "$DATA_ROOT")
REPORTS_ROOT="\$REPO_DIR/evaluation/reports_distributed/$(printf '%s' "$TIMESTAMP")"
LOGS_ROOT="\$REPORTS_ROOT/logs"
EVAL_MODE=$(printf '%q' "$EVAL_MODE")

OTHER_THRESHOLD=$(printf '%q' "$OTHER_THRESHOLD")
BATCH_SIZE=$(printf '%q' "$BATCH_SIZE")
MAX_SAMPLES=$(printf '%q' "$MAX_SAMPLES")
LOG_INTERVAL=$(printf '%q' "$LOG_INTERVAL")
DEVICE=$(printf '%q' "$DEVICE")
INFERENCE_BACKEND=$(printf '%q' "$INFERENCE_BACKEND")
CHUNK_SIZE=$(printf '%q' "$CHUNK_SIZE")
UNET_CHUNK_SIZE=$(printf '%q' "$UNET_CHUNK_SIZE")
DRY_RUN="\${REMOTE_DRY_RUN:-0}"

TIMESTAMP=$(printf '%q' "$TIMESTAMP")
TAU_KEY="tau\$(printf '%s' "\$OTHER_THRESHOLD" | tr '.' 'p')"
SELECT_REQUESTED_IDS=$(printf '%q' "$requested_ids_csv")
INSTANCE_NAME=$(printf '%q' "$instance")
STATUS_ROOT="\$REPO_DIR/.distributed_eval/\$TIMESTAMP"

TASKS=(
  pure_fragments
  needle_32_63
  needle_64_plus
  sequence_pair
  sequence_triplet
  markdown_mix
)

DEFAULT_ORDERED_REQUESTED_IDS=(
  unet_al_ft_redo1
  unet_al_without_ft
  unet_al_scratch_r0warm_v1
  sfullfiles3
  sfullfiles4
  mamba_full_files_from_pt
  mamba_full_files_scratch_r0warm_v1
  mamba_full_files_pretrain_10k_v1
  04eyv59a
  mdoo1pgt
  qwy0hvt2
  iddg3pz9
)

COMMON_ARGS=(
EOF

  for common_arg in "${COMMON_ARGS[@]}"; do
    printf '  %q\n' "$common_arg"
  done

  cat <<'EOF'
)

mkdir -p "$REPORTS_ROOT" "$LOGS_ROOT" "$STATUS_ROOT"
rm -f "$STATUS_ROOT/worker.success" "$STATUS_ROOT/worker.failure" "$STATUS_ROOT/worker.exit_code"
printf '%s\n' "$$" > "$STATUS_ROOT/worker.pid"

on_exit() {
  local rc=$?
  printf '%s\n' "$rc" > "$STATUS_ROOT/worker.exit_code"
  if [[ "$rc" -eq 0 ]]; then
    : > "$STATUS_ROOT/worker.success"
    rm -f "$STATUS_ROOT/worker.failure"
  else
    printf '%s\n' "$rc" > "$STATUS_ROOT/worker.failure"
    rm -f "$STATUS_ROOT/worker.success"
  fi
}
trap on_exit EXIT

MODE_LABEL=""
MODE_ARGS=()
DATA_KEY=""
ORDERED_REQUESTED_IDS=()

register_requested_ids() {
  local raw_ids="$1"
  local requested_id=""
  local -a parsed_ids=()

  if [[ -z "$raw_ids" ]]; then
    ORDERED_REQUESTED_IDS=("${DEFAULT_ORDERED_REQUESTED_IDS[@]}")
    return 0
  fi

  IFS=',' read -r -a parsed_ids <<< "$raw_ids"
  for requested_id in "${parsed_ids[@]}"; do
    requested_id="${requested_id//[[:space:]]/}"
    [[ -n "$requested_id" ]] || continue
    ORDERED_REQUESTED_IDS+=("$requested_id")
  done
}

select_eval_mode() {
  case "$EVAL_MODE" in
    auto)
      if [[ -n "$DATA_ROOT" ]]; then
        if [[ "$DATA_ROOT" == "$DEFAULT_DATA_ROOT" ]]; then
          MODE_LABEL="legacy/non-fine-tuned (explicit evaluation/data)"
          MODE_ARGS=(--non-fine-tuned)
        else
          MODE_LABEL="custom"
          MODE_ARGS=()
        fi
      elif [[ -d "$FINE_TUNED_DATA_ROOT" ]]; then
        DATA_ROOT="$FINE_TUNED_DATA_ROOT"
        MODE_LABEL="fine-tuned (auto-selected evaluation/data_b)"
        MODE_ARGS=()
      elif [[ -d "$DEFAULT_DATA_ROOT" ]]; then
        DATA_ROOT="$DEFAULT_DATA_ROOT"
        MODE_LABEL="legacy/non-fine-tuned (auto-fallback: evaluation/data_b missing)"
        MODE_ARGS=(--non-fine-tuned)
      else
        echo "No evaluation dataset root found." >&2
        echo "Expected either fine-tuned datasets at $FINE_TUNED_DATA_ROOT" >&2
        echo "or legacy datasets at $DEFAULT_DATA_ROOT" >&2
        exit 1
      fi
      ;;
    fine_tuned)
      DATA_ROOT="${DATA_ROOT:-$FINE_TUNED_DATA_ROOT}"
      MODE_LABEL="fine-tuned"
      MODE_ARGS=()
      ;;
    non_fine_tuned|legacy)
      DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
      MODE_LABEL="legacy/non-fine-tuned"
      MODE_ARGS=(--non-fine-tuned)
      ;;
    *)
      echo "Unsupported EVAL_MODE='$EVAL_MODE' (expected: auto, fine_tuned, non_fine_tuned, legacy)" >&2
      exit 1
      ;;
  esac

  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Evaluation data root not found: $DATA_ROOT" >&2
    if [[ "$DATA_ROOT" == "$FINE_TUNED_DATA_ROOT" ]]; then
      echo "Build the fine-tuned evaluation datasets with obtain_eval_dataset.py or rerun with EVAL_MODE=non_fine_tuned." >&2
    fi
    exit 1
  fi
}

resolve_checkpoint_id() {
  local candidate=""
  for candidate in "$@"; do
    if [[ -e "$REPO_DIR/checkpoints/sweeps/${candidate}.msgpack" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

run_eval() {
  local requested_id="$1"
  local display_name="$2"
  local arch="$3"
  shift 3

  local checkpoint_id=""
  local checkpoint_path=""
  local report_dir=""
  local log_file=""
  local eval_chunk="$CHUNK_SIZE"
  local -a cmd=()

  checkpoint_id="$(resolve_checkpoint_id "$@")" || {
    echo "Missing checkpoint for ${display_name}. Tried: $*" >&2
    exit 1
  }
  checkpoint_path="$REPO_DIR/checkpoints/sweeps/${checkpoint_id}.msgpack"
  report_dir="$REPORTS_ROOT/${requested_id}__${DATA_KEY}__${TAU_KEY}__${TIMESTAMP}"
  log_file="$LOGS_ROOT/${requested_id}__${TAU_KEY}__${TIMESTAMP}.log"

  if [[ "$arch" == "unet1d" ]]; then
    eval_chunk="$UNET_CHUNK_SIZE"
  fi

  cmd=(
    "$PYTHON_BIN" "$EVAL_SCRIPT"
    --checkpoint "$checkpoint_path"
    --data-root "$DATA_ROOT"
    --arch "$arch"
    --model-dim 256
    --chunk "$eval_chunk"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --inference-backend "$INFERENCE_BACKEND"
    --other-threshold "$OTHER_THRESHOLD"
    --max-samples "$MAX_SAMPLES"
    --log-interval "$LOG_INTERVAL"
    --report-path "$report_dir"
    --tasks "${TASKS[@]}"
  )

  if [[ "$arch" == "unet1d" ]]; then
    cmd+=(--channels 32,64,64,128,128,128,128,256)
  else
    cmd+=(
      --mamba-layers 6
      --mamba-d-state 16
      --mamba-expand 1
      --mamba-dt-rank 16
      --mamba-conv 4
      --mamba-bidirectional
    )
  fi

  if [[ "$requested_id" != "$checkpoint_id" ]]; then
    echo "[$display_name] requested_id=${requested_id} resolved_checkpoint=${checkpoint_id}" >&2
  fi
  echo "[$display_name] report_dir=${report_dir}" >&2
  echo "[$display_name] log_file=${log_file}" >&2

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[%s] ' "$display_name"
    printf '%q ' "${cmd[@]}" "${MODE_ARGS[@]}" "${COMMON_ARGS[@]}"
    printf '\n'
    return 0
  fi

  if (
    set -euo pipefail
    "${cmd[@]}" "${MODE_ARGS[@]}" "${COMMON_ARGS[@]}"
  ) >"$log_file" 2>&1; then
    echo "OK: ${display_name} (${requested_id} -> ${checkpoint_id})"
    echo "  report: ${report_dir}"
    echo "  log:    ${log_file}"
    return 0
  fi

  echo "FAILED: ${display_name} (${requested_id} -> ${checkpoint_id})" >&2
  echo "  report: ${report_dir}" >&2
  echo "  log:    ${log_file}" >&2
  return 1
}

run_requested_eval() {
  local requested_id="$1"
  case "$requested_id" in
    unet_al_ft_redo1)
      run_eval "$requested_id" "U-Net AL from FT (1536 windows)" "unet1d" "unet_al_ft_redo1"
      ;;
    unet_al_without_ft)
      run_eval "$requested_id" "U-Net AL from PT (joint FT+AL, 1536 windows)" "unet1d" "unet_al_without_ft-13800"
      ;;
    unet_al_scratch_r0warm_v1)
      run_eval "$requested_id" "U-Net AL from scratch with round-0 warmup" "unet1d" "unet_al_scratch_r0warm_v1"
      ;;
    sfullfiles3)
      run_eval "$requested_id" "Mamba FT full-files 10k (no AL replay)" "mamba" "sfullfiles3"
      ;;
    sfullfiles4)
      run_eval "$requested_id" "Mamba AL full-files 10k" "mamba" "sfullfiles4"
      ;;
    mamba_full_files_from_pt)
      run_eval "$requested_id" "Mamba AL full-files 10k from PT" "mamba" "mamba_full_files_from_pt"
      ;;
    mamba_full_files_scratch_r0warm_v1)
      run_eval "$requested_id" "Mamba AL full-files 10k from scratch with round-0 warmup" "mamba" "mamba_full_files_scratch_r0warm_v1"
      ;;
    mamba_full_files_pretrain_10k_v1)
      run_eval "$requested_id" "Mamba pretrain-only full-files 10k" "mamba" "mamba_full_files_pretrain_10k_v1"
      ;;
    04eyv59a)
      run_eval "$requested_id" "U-Net FT (1536 windows)" "unet1d" "04eyv59a"
      ;;
    mdoo1pgt)
      run_eval "$requested_id" "Mamba FT (1536 windows)" "mamba" "mdoo1pgt"
      ;;
    qwy0hvt2)
      run_eval "$requested_id" "U-Net PT (1536 windows)" "unet1d" "qwy0hvt2"
      ;;
    iddg3pz9)
      run_eval "$requested_id" "Mamba PT (1536 windows)" "mamba" "iddg3pz9"
      ;;
    *)
      echo "Unknown requested id: $requested_id" >&2
      return 1
      ;;
  esac
}

register_requested_ids "$SELECT_REQUESTED_IDS"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Evaluation script not found: $EVAL_SCRIPT" >&2
  exit 1
fi

select_eval_mode
DATA_KEY="$(basename "$DATA_ROOT")"

echo "Instance: $INSTANCE_NAME" >&2
echo "Evaluation mode: $MODE_LABEL" >&2
echo "Evaluation data root: $DATA_ROOT" >&2
echo "Report root: $REPORTS_ROOT" >&2
echo "Status root: $STATUS_ROOT" >&2
echo "Task subset: ${TASKS[*]}" >&2
echo "Requested ids: ${ORDERED_REQUESTED_IDS[*]}" >&2

overall_status=0

for requested_id in "${ORDERED_REQUESTED_IDS[@]}"; do
  if ! run_requested_eval "$requested_id"; then
    overall_status=1
  fi
done

exit "$overall_status"
EOF
}

upload_worker_script() {
  local instance="$1"
  local local_script="$2"
  local remote_script_path=""
  local upload_command=""

  remote_script_path="$(remote_worker_script_abs "$instance")"
  upload_command="$(cat <<EOF
set -euo pipefail
mkdir -p $(printf '%q' "$(remote_status_root_abs)") $(printf '%q' "$(remote_logs_root_abs)")
cat > $(printf '%q' "$remote_script_path")
chmod 700 $(printf '%q' "$remote_script_path")
EOF
)"

  gcloud compute ssh "${REMOTE_USER}@${instance}" "${GCLOUD_BASE_ARGS[@]}" \
    --command "bash -lc $(printf '%q' "$upload_command")" < "$local_script"
}

launch_worker() {
  local instance="$1"
  local remote_script_path=""
  local launch_command=""

  remote_script_path="$(remote_worker_script_abs "$instance")"
  launch_command="$(cat <<EOF
set -euo pipefail
mkdir -p $(printf '%q' "$(remote_status_root_abs)") $(printf '%q' "$(remote_logs_root_abs)")
rm -f $(printf '%q' "$(remote_status_root_abs)/worker.pid") \
      $(printf '%q' "$(remote_status_root_abs)/worker.success") \
      $(printf '%q' "$(remote_status_root_abs)/worker.failure") \
      $(printf '%q' "$(remote_status_root_abs)/worker.exit_code")
nohup $(printf '%q' "$remote_script_path") > $(printf '%q' "$(worker_stdout_log_abs "$instance")") 2>&1 < /dev/null &
printf '%s\n' \$! > $(printf '%q' "$(remote_status_root_abs)/worker.pid")
EOF
)"

  gcloud_ssh "$instance" "$launch_command" >/dev/null
}

fetch_worker_state() {
  local instance="$1"
  local remote_script=""
  local output=""

  remote_script="$(cat <<EOF
set -euo pipefail
status_root=$(printf '%q' "$(remote_status_root_abs)")
state="running"
if [[ -f "\$status_root/worker.success" ]]; then
  state="success"
elif [[ -f "\$status_root/worker.failure" ]]; then
  state="failure"
fi
exit_code=""
if [[ -f "\$status_root/worker.exit_code" ]]; then
  exit_code="\$(cat "\$status_root/worker.exit_code")"
fi
printf '%s:%s\n' "\$state" "\$exit_code"
EOF
)"

  if ! output="$(gcloud_ssh "$instance" "$remote_script" 2>/dev/null)"; then
    printf 'unknown:\n'
    return 0
  fi

  printf '%s\n' "$output"
}

sync_instance_results_to_aggregator() {
  local instance="$1"
  local requested_rel=""
  local source_collect_script=""
  local target_extract_script=""

  [[ "$instance" != "$AGGREGATOR_INSTANCE" ]] || return 0

  source_collect_script="$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "$REMOTE_REPO_DIR")
paths=()
EOF
)"

  while IFS= read -r requested_rel; do
    [[ -n "$requested_rel" ]] || continue
    source_collect_script+=$'\n'
    source_collect_script+="if [[ -e $(printf '%q' "$requested_rel") ]]; then paths+=($(printf '%q' "$requested_rel")); fi"
  done < <(instance_report_rel_paths "$instance")

  while IFS= read -r requested_rel; do
    [[ -n "$requested_rel" ]] || continue
    source_collect_script+=$'\n'
    source_collect_script+="if [[ -e $(printf '%q' "$requested_rel") ]]; then paths+=($(printf '%q' "$requested_rel")); fi"
  done < <(instance_log_rel_paths "$instance")

  source_collect_script+=$'\n''if (( ${#paths[@]} == 0 )); then'
  source_collect_script+=$'\n''  echo "No report or log artifacts available to sync." >&2'
  source_collect_script+=$'\n''  exit 1'
  source_collect_script+=$'\n''fi'
  source_collect_script+=$'\n''printf '\''Including %s\n'\'' "${paths[@]}" >&2'
  source_collect_script+=$'\n''tar -czf - -- "${paths[@]}"'

  target_extract_script="$(cat <<EOF
set -euo pipefail
mkdir -p $(printf '%q' "$(remote_reports_root_abs)") $(printf '%q' "$(remote_logs_root_abs)")
tar -xzf - -C $(printf '%q' "$REMOTE_REPO_DIR")
EOF
)"

  log "Syncing ${instance} results to ${AGGREGATOR_INSTANCE}:${REMOTE_REPO_DIR}/${REPORTS_REL}"
  gcloud compute ssh "${REMOTE_USER}@${instance}" "${GCLOUD_BASE_ARGS[@]}" \
    --command "bash -lc $(printf '%q' "$source_collect_script")" | \
    gcloud compute ssh "${REMOTE_USER}@${AGGREGATOR_INSTANCE}" "${GCLOUD_BASE_ARGS[@]}" \
      --command "bash -lc $(printf '%q' "$target_extract_script")"
}

stop_training_instance() {
  local instance="$1"
  [[ "$STOP_TRAINING_AFTER_SYNC" == "1" ]] || return 0
  [[ "$instance" != "$AGGREGATOR_INSTANCE" ]] || return 0
  log "Stopping instance ${instance}"
  gcloud compute instances stop "$instance" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --quiet >/dev/null
}

verify_and_export_matrix() {
  local requested_rel=""
  local export_script=""
  local output=""
  local missing_rel_paths=()
  local instance=""

  export_script="$(cat <<EOF
set -euo pipefail
cd $(printf '%q' "$REMOTE_REPO_DIR")
missing=()
EOF
)"

  for instance in "${INSTANCES[@]}"; do
    while IFS= read -r requested_rel; do
      [[ -n "$requested_rel" ]] || continue
      export_script+=$'\n'
      export_script+="[[ -d $(printf '%q' "$requested_rel") ]] || missing+=($(printf '%q' "$requested_rel"))"
    done < <(instance_report_rel_paths "$instance")
  done

  export_script+=$'\n''if (( ${#missing[@]} > 0 )); then'
  export_script+=$'\n''  printf '\''MISSING:%s\n'\'' "${missing[@]}"'
  export_script+=$'\n''  exit 3'
  export_script+=$'\n''fi'
  export_script+=$'\n'
  export_script+="$(printf '%q' "$REMOTE_REPO_DIR/.venv/bin/python") $(printf '%q' "$REMOTE_REPO_DIR/evaluation/export_report_matrix.py") --reports-dir $(printf '%q' "$REMOTE_REPO_DIR/$REPORTS_REL") --output $(printf '%q' "$REMOTE_REPO_DIR/$FINAL_CSV_REL")"

  if output="$(gcloud_ssh "$AGGREGATOR_INSTANCE" "$export_script" 2>&1)"; then
    printf '%s\n' "$output" >&2
    return 0
  fi

  while IFS= read -r requested_rel; do
    [[ "$requested_rel" == MISSING:* ]] || continue
    missing_rel_paths+=("${requested_rel#MISSING:}")
  done <<< "$output"

  if [[ "${#missing_rel_paths[@]}" -gt 0 ]]; then
    log "Skipping final matrix export because some expected report directories are missing on ${AGGREGATOR_INSTANCE}:"
    for requested_rel in "${missing_rel_paths[@]}"; do
      log "  ${requested_rel}"
    done
    return 1
  fi

  printf '%s\n' "$output" >&2
  return 1
}

main() {
  local instance=""
  local worker_state=""
  local worker_rc=""
  local status_payload=""
  local tmp_script=""
  local pending_instances=()
  local next_pending_instances=()
  local success_instances=()
  local failed_instances=()
  local synced_instances=()
  local already_synced=""
  local overall_status=0

  require_cmd gcloud
  require_cmd tar
  require_cmd mktemp

  print_summary

  for instance in "${INSTANCES[@]}"; do
    preflight_instance "$instance"
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1, preflight completed. Skipping upload, launch, sync, stop, and export."
    return 0
  fi

  for instance in "${INSTANCES[@]}"; do
    tmp_script="$(mktemp "${TMPDIR:-/tmp}/launch_distributed_eval.${instance}.XXXXXX.sh")"
    TEMP_FILES+=("$tmp_script")
    build_remote_worker_script "$instance" "$(instance_requested_ids "$instance")" > "$tmp_script"
    upload_worker_script "$instance" "$tmp_script"
    launch_worker "$instance"
    pending_instances+=("$instance")
    log "Launched worker on ${instance}"
  done

  while [[ "${#pending_instances[@]}" -gt 0 ]]; do
    next_pending_instances=()
    for instance in "${pending_instances[@]}"; do
      status_payload="$(fetch_worker_state "$instance")"
      worker_state="${status_payload%%:*}"
      worker_rc="${status_payload#*:}"

      case "$worker_state" in
        success|failure)
          log "Worker ${instance} completed with state=${worker_state} exit_code=${worker_rc:-unknown}"
          already_synced="0"
          if instance_is_training "$instance"; then
            for already_synced in "${synced_instances[@]}"; do
              if [[ "$already_synced" == "$instance" ]]; then
                already_synced="1"
                break
              fi
            done
            if [[ "$already_synced" != "1" ]]; then
              if sync_instance_results_to_aggregator "$instance"; then
                synced_instances+=("$instance")
                if [[ "$worker_state" == "success" ]]; then
                  stop_training_instance "$instance"
                else
                  overall_status=1
                fi
              else
                overall_status=1
                log "Sync failed for ${instance}; leaving the VM running for inspection."
              fi
            fi
          fi

          if [[ "$worker_state" == "success" ]]; then
            success_instances+=("$instance")
          else
            failed_instances+=("$instance")
            overall_status=1
          fi
          ;;
        running|unknown)
          if [[ "$worker_state" == "unknown" ]]; then
            log "Status probe for ${instance} failed; will retry."
          fi
          next_pending_instances+=("$instance")
          ;;
        *)
          overall_status=1
          log "Unexpected worker state from ${instance}: ${status_payload}"
          next_pending_instances+=("$instance")
          ;;
      esac
    done

    pending_instances=("${next_pending_instances[@]}")
    if [[ "${#pending_instances[@]}" -gt 0 ]]; then
      sleep "$POLL_INTERVAL_SECONDS"
    fi
  done

  if ! verify_and_export_matrix; then
    overall_status=1
  fi

  if [[ "${#success_instances[@]}" -gt 0 ]]; then
    log "Successful workers: ${success_instances[*]}"
  fi
  if [[ "${#failed_instances[@]}" -gt 0 ]]; then
    log "Failed workers: ${failed_instances[*]}"
  fi

  log
  log "Distributed evaluation complete."
  log "Aggregator reports root: ${AGGREGATOR_INSTANCE}:${REMOTE_REPO_DIR}/${REPORTS_REL}"
  log "Final CSV: ${AGGREGATOR_INSTANCE}:${REMOTE_REPO_DIR}/${FINAL_CSV_REL}"

  return "$overall_status"
}

main "$@"
