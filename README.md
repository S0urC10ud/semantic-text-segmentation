# Text Segmenter

Byte-wise segmentation of 38 content types (including shell, HTML, CSS, JS, ...) in JAX/Flax.

The model works directly on bytes and predicts, for every position, which language or content type it belongs to (e.g. `html`, `css`, `javascript_typescript`, `python`, `encoding_base64`, …). The repo contains everything needed to go from raw code corpora to training data, checkpoints, evaluation reports, and an interactive viewer.

---

TL;DR: How can I segment my text?

0. [install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
1. get the packages by running `uv sync` in the root directory of the project and activate the venv (Linux: `source ./venv/bin/activate`)
2. download the model: https://drive.google.com/file/d/1HvrDp0NOuSr_xVjzzBSxmLg8LsrKk98k/view?usp=sharing
3. run the segment viewer: `cd viewers` and  `uv run python interactive_viewer.py --ckpt path_to_model.msgpack`
4. Open http://127.0.0.1:8000 or use the API - the docs are available at http://127.0.0.1:8000/openapi.json (if `interactive_viewer.py` is started with `--openapi`)

Note that the first request takes far longer than the rest because JAX has to trace + optimize + generate device-specific code + possibly autotune + run the model.


## What lives where

- `downloader/` – Streams The Stack (`bigcode/the-stack`) and MADLAD‑400 (`allenai/MADLAD-400`), filters/cleans files, and builds Arrow datasets:
  - `main.py` turns raw repos into windowed datasets under `<out-root>/<split>/<label>/dataset` with deterministic 70/10/10/10 train/val/monitor/test splits.
  - Heuristics handle things like stripping `<script>/<style>` from HTML, dropping framework-heavy samples, and extracting only the PHP code from mixed templates for the train and validation splits. The monitor and test splits are constructed using Gemini (`downloader/llm_segmentor.py`).
- `train/` – Model training and utilities:
  - `main.py` drives training, W&B logging, checkpointing, and optional fine-tuning on a subset of the LLM-segmented data (of course, this has to be regarded in the evaluation by using disjoint subsets `_a` and `_b` of the monitor set).
  - `utils/model.py` defines the currently used model: a 1D U‑Net over byte tokens (default window size 1536 bytes).
  - `utils/window_generator.py` builds pure, mixed, and line‑injection training windows with various augmentation tricks.
  - `utils/config.py` centralizes the label set (`LANG_ORDER`), window size, and `DataConfig`/`TrainConfig`.
- `evaluation/` – Benchmark datasets and metrics:
  - `obtain_eval_dataset.py` samples from the monitor set + Gemini segmentations to build curated HF datasets (`pure_fragments`, `needle_*`, `mal_injection`, `sequence_pair/triplet`, `markdown_mix`, throughput stress tests, …).
  - `evaluation.py` loads a checkpoint, runs the full benchmark battery, and by default writes each run into its own directory under `evaluation/reports/<run_name>/` with `report.md`, `comparison_metrics.json`, and task-wise plus aggregated confusion-matrix plots.
- `evaluation/reports/` – Per-run evaluation artifact directories.
- `viewers/` – Small FastAPI frontends for playing with the model:
  - `interactive_viewer.py` exposes a `/api/segment` endpoint where one can provide custom inputs to the model and see its output probabilities.
  - `confusion_viewer.py`, `evaluation_viewer.py`, `dataset_viewer.py` are focused viewers for confusion matrices, eval runs, and datasets.
- `active_learning/` – Boundary-focused active learning:
  - `round.py` runs one refinement round (`infer -> acquire uncertain boundaries -> oracle -> store`).
  - `acquisition.py` scores candidate boundary spans with entropy + local flip-rate.
  - `oracle.py` provides a stub oracle and a batched Gemini oracle (requests logged into `gemini_output_logs/` for usage/cost tracking).
  - `label_store.py` persists outcomes in SQLite and can build replay windows for training.
  - `meta_trainer.py` runs train/AL loops.
- `gemini_segmentations/`, `gemini_output_logs/` – Expected locations for LLM‑based segmentations and their logs (used by `obtain_eval_dataset.py` when present).

If you’re trying to understand how things fit together, a good path is:
`downloader/main.py` → `train/utils/window_generator.py` → `train/main.py` → `evaluation/obtain_eval_dataset.py` → `evaluation/evaluation.py` → `viewers/interactive_viewer.py`.


## Setup

The project targets Python 3.11 and JAX/Flax.

On Windows note that WSL is likely necessary if you want to use your GPU as JAX has no native Windows CUDA support.

Basic setup (from the repo root):
```bash
uv sync
```

You will also need access to the HuggingFace datasets mentioned above (`bigcode/the-stack` and `allenai/MADLAD-400`); some configurations assume you have a valid HF token cached.

In case you want to re-build the Gemini-augmented dataset, you also need to set the environment variable `$GOOGLE_API_KEY` to a suitable value.

## Typical workflows

### 1. Build Arrow training data from The Stack

From the repo root:

```bash
cd downloader
python main.py \
  --out-root downloader/arrow_out \
  --use-auth-token \
  --window-bytes 1536
```

This will create Arrow datasets under `downloader/arrow_out/<split>/<label>/dataset` for the labels in `--langs`. License filtering and content cleaning happen inside `downloader/main.py`.

### 2. Train the segmenter

Once `downloader/arrow_out` exists:

```bash
python train/main.py \
  --data_root downloader/arrow_out \
  --steps 2000000 \
  --model_dim 256 \
  --channels 32,64,64,128,128,128,128,256
```

The training script:

- samples windows using `utils.window_generator.make_training_window`,
- logs metrics to W&B (if configured),
- periodically evaluates on validation splits and, optionally, the monitor memmap,
- writes checkpoints under `train/checkpoints/…` (path controlled by `--ckpt_path` / `TrainConfig.ckpt_path`).

For fine‑tuning on a frozen monitor split, look at the `--fine-tune` family of flags in `train/main.py`.

### 3. Build evaluation datasets

With monitor data and (optionally) Gemini JSON segmentations in place:

```bash
python evaluation/obtain_eval_dataset.py \
  --data-root downloader/arrow_out \
  --monitor-seg-root gemini_segmentations/monitor \
  --output-root evaluation/data
```

This produces several HF datasets under `evaluation/data/<task>/dataset` plus a `manifest.json` describing what was generated.

If you’re evaluating a strictly fine‑tuned model on `monitor_preprocessed_b`, pass `--fine-tuned` so that the builder only uses the dedicated fine‑tune split.

### 4. Run the evaluation harness

Given a trained checkpoint:

```bash
python evaluation/evaluation.py \
  --checkpoint train/checkpoints/seg-unet1d.msgpack \
  --data-root evaluation/data
```

The script runs the model over all configured tasks, aggregates metrics (including IoU coverage for needles, markdown mixes, and malicious payloads), and writes a dedicated artifact directory under `evaluation/reports/` containing the Markdown report, JSON summary, and confusion-matrix images. You can still override the destination with `--report-path`, either as a markdown file path or a directory.

### 5. Launch the interactive viewer

To quickly inspect how a checkpoint segments real snippets:

```bash
python viewers/interactive_viewer.py \
  --ckpt train/checkpoints/seg-unet1d.msgpack \
```

Then open `http://127.0.0.1:8000` in a browser. The UI will show color‑coded spans, per‑character probabilities, and the sliding windows the model actually processed.

Open-set inference (`OTHER` from low confidence):

```bash
python viewers/interactive_viewer.py \
  --ckpt train/checkpoints/seg-unet1d.msgpack \
  --other-threshold 0.2
```

Sweep `tau` on monitor validation memmap (`monitor_preprocessed_b`) and pick the best threshold:

```bash
python evaluation/sweep_tau.py \
  --checkpoint train/checkpoints/seg-unet1d.msgpack \
  --monitor-root downloader/monitor_preprocessed_b \
  --tau-start 0.05 \
  --tau-end 0.95 \
  --tau-step 0.05 \
  --select-by macro_f1
```

By default, `sweep_tau.py` augments monitor evaluation with true-`OTHER` rows
from `downloader/arrow_out_other/train/other/dataset` (`--other-limit 1024`).
Set `--other-limit 0` to disable this augmentation.

OE training (uniform-target loss on outlier batches):

```bash
python train/main.py \
  --data_root downloader/arrow_out \
  --oe-lambda 0.1 \
  --oe-ratio 0.05
```

Build a heldout OTHER dataset from non-mapped The Stack languages:

```bash
python downloader/misc/extract_other.py \
  --out-root downloader/arrow_out_other \
  --label other \
  --split train \
  --max-samples 200000 \
  --max-bytes 1536 \
  --use-auth-token
```

Use that heldout Arrow dataset during training/fine-tuning:

```bash
python train/main.py \
  --data_root downloader/arrow_out \
  --oe-lambda 0.1 \
  --oe-ratio 0.05 \
  --oe-source mixed \
  --oe-heldout-root downloader/arrow_out_other
```

### 6. Run one active-learning round (boundary refinement)

Example (no network, deterministic stub oracle):

```bash
python -m active_learning.round \
  --ckpt train/checkpoints/seg-unet1d.msgpack \
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
python -m active_learning.round \
  --ckpt train/checkpoints/seg-unet1d.msgpack \
  --data-root downloader/arrow_out \
  --split monitor \
  --store active_learning/label_store.sqlite \
  --oracle gemini \
  --gemini-model gemini-3-flash-preview \
  --gemini-batch-size 2 \
  --context-chars 250
```

By default, `active_learning.round` caps oracle traffic to `3` requests per round
(`--max-oracle-requests 3`). Use `--unlimited-oracle` to restore unbounded querying.

Build/rebuild the fixed curated benchmark set (reviewable JSON + JSONL + Markdown):

```bash
python -m active_learning.build_curated_benchmark_set \
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
python viewers/curated_benchmark_viewer.py \
  --jsonl active_learning/benchmark_data/curated_oracle_segments_v1.jsonl \
  --expected-length 256 \
  --port 8093
```

Then open `http://127.0.0.1:8093` and inspect:
- summary + issue code counts
- filterable sample table (`all`, `issues`, `errors`, `ok`)
- per-sample truth/predicted/diff renderings
- raw JSON and metadata
- live Gemini benchmark overlays: choose `Gemini Live Run` + `Batch` in the top controls
  (loaded from `active_learning/benchmark_results/*.json` and linked `gemini_output_logs`)

Benchmark Gemini oracle quality vs batch size (scored, 256-char snippets, mixed + non-mixed):

```bash
python -m active_learning.benchmark_gemini_batch_size \
  --samples 128 \
  --pure-samples 48 \
  --sample-length 256 \
  --batch-sizes 1,2,4,8,16,32
```

By default, `benchmark_gemini_batch_size.py` loads
`active_learning/benchmark_data/curated_oracle_segments_v1.json`.
Defaults now use the full curated benchmark (`--samples 0`, `--pure-samples -1`).
Fallback to dynamic sampling from `evaluation/data` is disabled; the script now fails fast
if the curated benchmark dataset file is missing.

Important guard: this script does **not** run oracle requests unless you explicitly pass
`--run-live`; otherwise it only computes the no-cost prior baseline.

Live run (requires explicit consent):

```bash
python -m active_learning.benchmark_gemini_batch_size \
  --samples 128 \
  --pure-samples 48 \
  --sample-length 256 \
  --batch-sizes 1,2,4,8,16,32 \
  --run-live
```

Train with replayed AL labels mixed into normal batches:

```bash
python train/main.py \
  --data_root downloader/arrow_out \
  --ckpt_path train/checkpoints/seg-unet1d.msgpack \
  --active_learning_store active_learning/label_store.sqlite \
  --active_learning_mix_prob 0.25 \
  --active_learning_max_windows 10000
```

Full loop (`train -> AL round -> train -> ...`):

```bash
python -m active_learning.meta_trainer \
  --rounds 3 \
  --ckpt-path train/checkpoints/al_loop.msgpack \
  --data-root downloader/arrow_out \
  --train-max-minutes 20 \
  --al-store active_learning/label_store.sqlite \
  --al-oracle stub
```

`active_learning.meta_trainer` uses one shared W&B run across all rounds by default.
Use `--wandb-run-id <RUN_ID>` to pin/reuse a specific run id across invocations, or
`--wandb-mode per-round` for one run per round.
Each round runs in this order:
1) acquire/oracle on the standard train split
2) monitor fine-tune training (`--fine-tune`) with AL replay enabled

Its AL step defaults to `3` oracle requests per round (`--al-max-oracle-requests 3`);
use `--al-unlimited-oracle` to disable this cap. Replay mix is scheduled linearly
from `0` to `--al-mix-prob` (default `0.5`) as stored oracle refinements grow from
`0` to `--al-mix-full-at-rows` (default `1000`).

### 7. Explore stored active-learning refinements

```bash
python viewers/active_learning_viewer.py \
  --store active_learning/label_store.sqlite \
  --port 8061
```

Then open `http://127.0.0.1:8061` to inspect summary stats, label distributions,
refined-vs-predicted confusion, and per-sample predicted/refined renderings.


## Notes

- The canonical label ordering and IDs live in `train/utils/config.py` (`LANG_ORDER`, `LANG2ID`, `ID2LANG`). Training, evaluation, and the viewers all rely on this mapping being consistent.
- This is a research‑style codebase rather than a polished library. Some parts certainly are a bit too over-engineered (e.g. window generators, eval tasks). 🙂
