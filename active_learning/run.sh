REPO_DIR="$PWD"
CHILD_RUN_ID="ke2m7q9vd"
PARENT_RUN_ID="k9c4r1px"

"$REPO_DIR/.venv/bin/python" -m active_learning.meta_trainer \
  --ckpt-path "$REPO_DIR/checkpoints/sweeps/${CHILD_RUN_ID}.msgpack" \
  --init-ckpt-path "$REPO_DIR/checkpoints/sweeps/ie2m7q9vd.msgpack" \
  --arch mamba \
  --train-steps 300 \
  --al-store "$REPO_DIR/active_learning/label_store.sqlite" \
  --al-split train \
  --al-oracle gemini \
  --al-gemini-batch-size 16 \
  --al-predict-batch-size 12 \
  --al-max-samples-per-lang 64 \
  --al-max-candidates-per-sample 4 \
  --al-min-score 0.3 \
  --al-context-chars 250 \
  --al-max-oracle-requests 10 \
  --al-gemini-missing-snippet-retries 0 \
  --al-mix-prob 0.3 \
  --al-mix-full-at-rows 500 \
  --al-gemini-model "gemini-3-flash-preview" \
  --al-gemini-thinking-level medium \
  --al-max-windows 10000000 \
  --wandb-mode shared \
  --wandb-run-id "$CHILD_RUN_ID" \
  --wandb-parent-run-id "$PARENT_RUN_ID" \
  --train-extra-args "--eval_every 300 --arch mamba
--model_dim 256 --mamba_layers 6 --mamba_d_state 16 \
--mamba_expand 1 --batch_size 12 --accum_steps 3 --lr 5e-5
--num_workers 6 \
--monitor_eval_limit 512 --monitor_other_threshold 0.5
--fine_tune_use_oe --fine_tune_augment_monitor --num-gpus 1"
