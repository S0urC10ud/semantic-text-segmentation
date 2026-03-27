REPO_DIR="$PWD"
SOURCE_RUN_ID="sfullfiles4"
CHILD_RUN_ID="${CHILD_RUN_ID:-sfullfiles4}"
PARENT_RUN_ID="${PARENT_RUN_ID:-qfullfiles4}" # make sure they are not equal!

"$REPO_DIR/.venv/bin/python" -m active_learning.meta_trainer \
  --rounds 300 \
  --ckpt-path "$REPO_DIR/checkpoints/sweeps/${CHILD_RUN_ID}.msgpack" \
  --init-ckpt-path "$REPO_DIR/checkpoints/sweeps/${SOURCE_RUN_ID}.msgpack" \
  --arch mamba \
  --persistent-trainer \
  --train-steps 150 \
  --al-store "$REPO_DIR/active_learning/labels_full.sqlite" \
  --full-files \
  --full-file-max-bytes 10000 \
  --al-split train \
  --al-oracle gemini \
  --al-gemini-batch-size 1 \
  --al-predict-batch-size 30 \
  --al-sample-workers 6 \
  --al-sample-prefetch 64 \
  --al-max-samples-per-lang 8 \
  --al-max-candidates-per-sample 4 \
  --al-min-score 0.3 \
  --al-context-chars 5000 \
  --al-max-oracle-requests 30 \
  --al-gemini-missing-snippet-retries 0 \
  --al-mix-prob 0.5 \
  --al-mix-full-at-rows 500 \
  --al-gemini-model "gemini-3-flash-preview" \
  --al-gemini-thinking-level medium \
  --al-max-windows 10000000 \
  --wandb-mode shared \
  --wandb-run-id "$CHILD_RUN_ID" \
  --wandb-parent-run-id "$PARENT_RUN_ID" \
  --train-extra-args "--eval_every 150 --arch mamba
--model_dim 256 --mamba_layers 6 --mamba_d_state 16 \
--mamba_expand 1 --batch_size 4 --accum_steps 6 --lr 2e-5 \
--num_workers 6 \
--full-files --full-file-max-bytes 10000 \
--monitor_eval_limit 2048 --monitor_eval_batch_size 16 \
--monitor_eval_deterministic --monitor_eval_seed 123 \
--monitor_other_threshold 0.5 \
--fine_tune_dense_bias_prob 0.0 \
--fine_tune_use_oe --num-gpus 1"
