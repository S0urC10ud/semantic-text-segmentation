# Text Segmenter

Byte-wise segmentation of 35 content types (including shell, HTML, CSS, JavaScript/TypeScript, Python, Markdown, the five base-N encoding families, and so on) in JAX/Flax.

The model works directly on bytes and predicts, for every character position, which language or content type the position belongs to (`html`, `css`, `javascript_typescript`, `python`, `encoding_base64`, ...). Two architecture families ship in this repository: a 1D U-Net over byte tokens for fast windowed inference and a Mamba state-space model for long-context predictions. The repo contains everything needed to go from raw code corpora to training data, checkpoints, evaluation reports, and an interactive viewer.

---

## TL;DR: How can I segment my text?

The repository ships with two ready-to-use checkpoints under `checkpoints/`. The U-Net `AL` model is the fast local segmenter and is recommended for a first try. The Mamba `AL` model is slower per call but stronger on segment-heavy mixed-content inputs.

1. [install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
2. get the packages by running `uv sync` in the root directory of the project
3. run the segment viewer with one of the bundled checkpoints (from the repository root):

   U-Net AL:
   ```bash
   uv run python viewers/interactive_viewer.py \
     --ckpt checkpoints/unet_al.msgpack \
     --arch unet1d \
     --model-dim 256 \
     --channels 32,64,64,128,128,128,128,256 \
     --dtype bfloat16
   ```

   Mamba AL:
   ```bash
   uv run python viewers/interactive_viewer.py \
     --ckpt checkpoints/mamba_al.msgpack \
     --arch mamba \
     --model-dim 256 \
     --dtype bfloat16 \
     --mamba-layers 6 --mamba-d-state 16 --mamba-expand 1 \
     --mamba-dt-rank 16 --mamba-conv 4 --mamba-bidirectional
   ```

4. open <http://127.0.0.1:8000> or use the API. The docs are available at <http://127.0.0.1:8000/openapi.json> when `interactive_viewer.py` is started with `--openapi`.

The bundled checkpoints are flat `.msgpack` dumps without a co-located W&B run record, so the architecture and shape flags above are needed and the auto-detection used inside the repository's own training output does not apply to them.

Note that the first request takes far longer than the rest because JAX has to trace, optimise, generate device-specific code, possibly autotune, and run the model. Subsequent requests reuse the compiled program.


## What lives where

- `checkpoints/` ships the two finalist checkpoints. `unet_al.msgpack` is the U-Net `AL` finalist and `mamba_al.msgpack` is the Mamba `AL` finalist. Both are flat `flax.serialization` msgpack dumps and load through the same `SegmenterRunner` entry point as any other checkpoint produced by `train/main.py`.
- `downloader/` streams [The Stack](https://huggingface.co/datasets/bigcode/the-stack) (`bigcode/the-stack`) and [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400) (`allenai/MADLAD-400`), filters and cleans files, and builds Arrow datasets.
  - `main.py` turns raw repos into windowed datasets under `<out-root>/<split>/<label>/dataset` with deterministic 70/10/10/10 train/val/monitor/test splits.
  - Heuristics handle things like stripping `<script>` and `<style>` from HTML, dropping framework-heavy samples, and extracting only the PHP code from mixed templates for the train and validation splits. The monitor and test splits are constructed using Gemini through `downloader/llm_segmentor.py`.
- `train/` contains the model training and utility code.
  - `main.py` drives training, W&B logging, checkpointing, and optional dense fine-tuning on a subset of the LLM-segmented data. Disjoint subsets `_a` and `_b` of the monitor set keep evaluation honest.
  - `utils/model.py` defines both finalists. The U-Net uses a default window of 1536 bytes. The Mamba model uses a default chunk size of 10000 bytes and a bidirectional selective scan.
  - `utils/window_generator.py` builds pure, mixed, and line-injection training windows with the augmentation modes used by the weak-label pre-training stage.
  - `utils/config.py` centralises the label set (`LANG_ORDER`, `LANG2ID`, `ID2LANG`) along with `DataConfig` and `TrainConfig`.
- `evaluation/` contains the benchmark datasets and metrics.
  - `obtain_eval_dataset.py` samples from the monitor set and the Gemini segmentations to build curated Hugging Face datasets (`near_pure`, `needle_*`, `sequence_pair`, `sequence_triplet`, `markdown_mix`, `realistic`, plus the throughput stress tasks).
  - `evaluation.py` loads a checkpoint, runs the full benchmark battery, and by default writes each run into its own directory under `evaluation/reports/<run_name>/` with `report.md`, `comparison_metrics.json`, and the task-wise plus aggregated confusion-matrix plots. The deployment-time post-processing pipeline (whitespace relabel, low-confidence routing into `other`, local boundary snap, minimum-run normalisation) is reachable through `--postprocess-profile thesis`. Per-sample raw outputs can be cached with `--cache-predictions-dir` and replayed offline with `--rescore-from-cache` to sweep post-processing parameters without re-running inference.
  - `reports/` is the destination of every evaluation run.
  - `prediction_cache/` is the default destination of the cached raw outputs.
- `viewers/` contains small FastAPI frontends for playing with the model.
  - `interactive_viewer.py` exposes a `/api/segment` endpoint where one can provide custom inputs and see colour-coded spans, per-character probabilities, and the sliding windows the model actually processed.
  - `confusion_viewer.py`, `evaluation_viewer.py`, `dataset_viewer.py`, and `active_learning_viewer.py` are focused viewers for confusion matrices, eval runs, datasets, and active-learning refinements respectively.
- `active_learning/` runs the boundary-focused active-learning loop.
  - `round.py` runs one refinement round (`infer → acquire uncertain boundaries → oracle → store`).
  - `acquisition.py` scores candidate boundary spans with entropy and local flip-rate.
  - `oracle.py` provides a stub oracle and a batched Gemini oracle (requests logged into `gemini_output_logs/` for usage and cost tracking).
  - `label_store.py` persists outcomes in SQLite and can build replay windows for subsequent training.
  - `meta_trainer.py` orchestrates the full `train → AL round → train` loop.
- `gemini_segmentations/` and `gemini_output_logs/` are the expected locations for LLM-based segmentations and their logs (used by `obtain_eval_dataset.py` when present).

If you are trying to understand how things fit together, a good reading path is `downloader/main.py` → `train/utils/window_generator.py` → `train/main.py` → `evaluation/obtain_eval_dataset.py` → `evaluation/evaluation.py` → `viewers/interactive_viewer.py`.


## Setup

The project targets Python 3.11 and JAX/Flax.

On Windows note that WSL is likely necessary if you want to use your GPU, as JAX has no native Windows CUDA support.

Basic setup (from the repo root):

```bash
uv sync
```

You will also need access to the Hugging Face datasets mentioned above (`bigcode/the-stack` and `allenai/MADLAD-400`) if you want to re-build the dataset rather than just run the bundled checkpoints. Some configurations assume you have a valid HF token cached.

In case you want to re-build the Gemini-augmented dataset, you also need to set the environment variable `$GOOGLE_API_KEY` to a suitable value.


## Typical workflows

### 1. Build Arrow training data from The Stack

From the repo root:

```bash
cd downloader
uv run python main.py \
  --out-root downloader/arrow_out \
  --use-auth-token \
  --window-bytes 1536
```

This will create Arrow datasets under `downloader/arrow_out/<split>/<label>/dataset` for the labels in `--langs`. License filtering and content cleaning happen inside `downloader/main.py`.

### 2. Train the segmenter

Once `downloader/arrow_out` exists, train either family of model.

U-Net training run:

```bash
uv run python train/main.py \
  --data_root downloader/arrow_out \
  --steps 2000000 \
  --model_dim 256 \
  --channels 32,64,64,128,128,128,128,256
```

Mamba training run:

```bash
uv run python train/main.py \
  --data_root downloader/arrow_out \
  --arch mamba \
  --steps 2000000 \
  --model_dim 256 \
  --mamba-layers 6 --mamba-d-state 16 --mamba-expand 1 \
  --mamba-dt-rank 16 --mamba-conv 4 --mamba-bidirectional
```

The training script:

- samples windows using `utils.window_generator.make_training_window`,
- logs metrics to W&B (if configured),
- periodically evaluates on validation splits and, optionally, the monitor memmap,
- writes checkpoints under `train/checkpoints/` (path controlled by `--ckpt_path` or `TrainConfig.ckpt_path`).

For fine-tuning on a frozen monitor split, look at the `--fine-tune` family of flags in `train/main.py`.

### 3. Build evaluation datasets

With monitor data and (optionally) Gemini JSON segmentations in place:

```bash
uv run python evaluation/obtain_eval_dataset.py \
  --data-root downloader/arrow_out \
  --monitor-seg-root gemini_segmentations/monitor \
  --output-root evaluation/data
```

This produces several Hugging Face datasets under `evaluation/data/<task>/dataset` plus a `manifest.json` describing what was generated.

If you are evaluating a strictly fine-tuned model on `monitor_preprocessed_b`, pass `--fine-tuned` so that the builder only uses the dedicated fine-tune split.

### 4. Run the evaluation harness

Given a trained checkpoint:

```bash
uv run python evaluation/evaluation.py \
  --checkpoint checkpoints/unet_al.msgpack \
  --arch unet1d \
  --model-dim 256 \
  --channels 32,64,64,128,128,128,128,256 \
  --dtype bfloat16 \
  --data-root evaluation/test
```

The script runs the model over all configured tasks, aggregates metrics (including IoU coverage for needles, markdown mixes, and malicious payloads), and writes a dedicated artifact directory under `evaluation/reports/` containing the Markdown report, JSON summary, and confusion-matrix images. You can still override the destination with `--report-path`, either as a markdown file path or a directory.

To enable the deployment-time post-processing pipeline (the four cheap, local steps defined in `viewers/core.py`), pass `--postprocess-profile thesis`:

```bash
uv run python evaluation/evaluation.py \
  --checkpoint checkpoints/mamba_al.msgpack \
  --arch mamba \
  --model-dim 256 \
  --dtype bfloat16 \
  --mamba-layers 6 --mamba-d-state 16 --mamba-expand 1 \
  --mamba-dt-rank 16 --mamba-conv 4 --mamba-bidirectional \
  --data-root evaluation/test \
  --postprocess-profile thesis
```

If you want to sweep post-processing parameters (for example the minimum-run threshold) without paying for inference each time, run a single pass with `--cache-predictions-dir evaluation/prediction_cache/<model>` and then replay the cached raw outputs against any post-processing setting through `--rescore-from-cache evaluation/prediction_cache/<model>`. A rescore pass on the full test set takes about a minute on CPU.

### 5. Launch the interactive viewer

The TL;DR section shows the bundled-checkpoint command. The viewer also accepts arbitrary training checkpoints and a few additional flags:

```bash
uv run python viewers/interactive_viewer.py \
  --ckpt train/checkpoints/seg-unet1d.msgpack
```

Open <http://127.0.0.1:8000> in a browser. The UI shows colour-coded spans, per-character probabilities, and the sliding windows the model actually processed.

Open-set inference (route low-confidence positions into `other`):

```bash
uv run python viewers/interactive_viewer.py \
  --ckpt train/checkpoints/seg-unet1d.msgpack \
  --tau 0.30
```

Sweep `tau` on the monitor validation memmap (`monitor_preprocessed_b`) and pick the best threshold:

```bash
uv run python evaluation/sweep_tau.py \
  --checkpoint train/checkpoints/seg-unet1d.msgpack \
  --monitor-root downloader/monitor_preprocessed_b \
  --tau-start 0.05 \
  --tau-end 0.95 \
  --tau-step 0.05 \
  --select-by macro_f1
```

By default, `sweep_tau.py` augments monitor evaluation with true-`other` rows from `downloader/arrow_out_other/train/other/dataset` (`--other-limit 1024`). Set `--other-limit 0` to disable this augmentation.

OE training (uniform-target loss on outlier batches):

```bash
uv run python train/main.py \
  --data_root downloader/arrow_out \
  --oe-lambda 0.1 \
  --oe-ratio 0.05
```

Build a heldout `other` dataset from non-mapped The Stack languages:

```bash
uv run python downloader/misc/extract_other.py \
  --out-root downloader/arrow_out_other \
  --label other \
  --split train \
  --max-samples 200000 \
  --max-bytes 1536 \
  --use-auth-token
```

Use that heldout Arrow dataset during training/fine-tuning:

```bash
uv run python train/main.py \
  --data_root downloader/arrow_out \
  --oe-lambda 0.1 \
  --oe-ratio 0.05 \
  --oe-source mixed \
  --oe-heldout-root downloader/arrow_out_other
```

### 6. Run one active-learning round (boundary refinement)

Example (no network, deterministic stub oracle):

```bash
uv run python -m active_learning.round \
  --ckpt checkpoints/unet_al.msgpack \
  --arch unet1d --model-dim 256 \
  --channels 32,64,64,128,128,128,128,256 --dtype bfloat16 \
  --data-root downloader/arrow_out \
  --split monitor \
  --langs html,css,javascript_typescript,python \
  --store active_learning/label_store.sqlite \
  --oracle stub \
  --max-samples-per-lang 16 \
  --max-candidates-per-sample 3 \
  --context-chars 250
```

Gemini-backed refinement (batched requests, logs written to `gemini_output_logs/`):

```bash
uv run python -m active_learning.round \
  --ckpt checkpoints/unet_al.msgpack \
  --arch unet1d --model-dim 256 \
  --channels 32,64,64,128,128,128,128,256 --dtype bfloat16 \
  --data-root downloader/arrow_out \
  --split monitor \
  --store active_learning/label_store.sqlite \
  --oracle gemini \
  --gemini-model gemini-3-flash-preview \
  --gemini-batch-size 2 \
  --context-chars 250
```

By default, `active_learning.round` caps oracle traffic to 3 requests per round (`--max-oracle-requests 3`). Use `--unlimited-oracle` to restore unbounded querying.

Build or rebuild the fixed curated benchmark set (reviewable JSON, JSONL, and Markdown):

```bash
uv run python -m active_learning.build_curated_benchmark_set \
  --target-length 256 \
  --target-samples 1000 \
  --seed 11
```

Outputs:

- `active_learning/benchmark_data/curated_oracle_segments_v1.json`
- `active_learning/benchmark_data/curated_oracle_segments_v1.jsonl`
- `active_learning/benchmark_data/curated_oracle_segments_v1.md`

Review the curated JSONL for annotation mistakes (schema, offsets, overlaps, label inconsistencies):

```bash
uv run python viewers/curated_benchmark_viewer.py \
  --jsonl active_learning/benchmark_data/curated_oracle_segments_v1.jsonl \
  --expected-length 256 \
  --port 8093
```

Then open <http://127.0.0.1:8093> and inspect:

- summary and issue-code counts,
- filterable sample table (`all`, `issues`, `errors`, `ok`),
- per-sample truth, predicted, and diff renderings,
- raw JSON and metadata,
- live Gemini benchmark overlays (choose `Gemini Live Run` plus `Batch` in the top controls, loaded from `active_learning/benchmark_results/*.json` and linked `gemini_output_logs`).

Benchmark Gemini oracle quality vs batch size (scored, 256-char snippets, mixed and non-mixed):

```bash
uv run python -m active_learning.benchmark_gemini_batch_size \
  --samples 128 \
  --pure-samples 48 \
  --sample-length 256 \
  --batch-sizes 1,2,4,8,16,32
```

By default, `benchmark_gemini_batch_size.py` loads `active_learning/benchmark_data/curated_oracle_segments_v1.json`. Defaults now use the full curated benchmark (`--samples 0`, `--pure-samples -1`). Fallback to dynamic sampling from `evaluation/data` is disabled, so the script fails fast if the curated benchmark dataset file is missing.

Important guard: this script does **not** run oracle requests unless you explicitly pass `--run-live`. Otherwise it only computes the no-cost prior baseline.

Live run (requires explicit consent):

```bash
uv run python -m active_learning.benchmark_gemini_batch_size \
  --samples 128 \
  --pure-samples 48 \
  --sample-length 256 \
  --batch-sizes 1,2,4,8,16,32 \
  --run-live
```

Train with replayed AL labels mixed into normal batches:

```bash
uv run python train/main.py \
  --data_root downloader/arrow_out \
  --ckpt_path train/checkpoints/seg-unet1d.msgpack \
  --active_learning_store active_learning/label_store.sqlite \
  --active_learning_mix_prob 0.25 \
  --active_learning_max_windows 10000
```

Full loop (`train → AL round → train → ...`):

```bash
uv run python -m active_learning.meta_trainer \
  --rounds 3 \
  --ckpt-path train/checkpoints/al_loop.msgpack \
  --data-root downloader/arrow_out \
  --train-max-minutes 20 \
  --al-store active_learning/label_store.sqlite \
  --al-oracle stub
```

`active_learning.meta_trainer` uses one shared W&B run across all rounds by default. Use `--wandb-run-id <RUN_ID>` to pin or reuse a specific run id across invocations, or `--wandb-mode per-round` for one run per round. Each round runs in this order:

1. acquire and oracle on the standard train split,
2. monitor fine-tune training (`--fine-tune`) with AL replay enabled.

Its AL step defaults to 3 oracle requests per round (`--al-max-oracle-requests 3`). Use `--al-unlimited-oracle` to disable this cap. Replay mix is scheduled linearly from 0 to `--al-mix-prob` (default `0.5`) as stored oracle refinements grow from 0 to `--al-mix-full-at-rows` (default `1000`).

### 7. Explore stored active-learning refinements

```bash
uv run python viewers/active_learning_viewer.py \
  --store active_learning/label_store.sqlite \
  --port 8061
```

Then open <http://127.0.0.1:8061> to inspect summary stats, label distributions, refined-versus-predicted confusion, and per-sample predicted and refined renderings.


## Notes

The canonical label ordering and IDs live in `train/utils/config.py` (`LANG_ORDER`, `LANG2ID`, `ID2LANG`). Training, evaluation, and the viewers all rely on this mapping being consistent.

This is a research-style codebase rather than a polished library. Some parts certainly are a bit too over-engineered (for example the window generators and the eval tasks). 🙂
