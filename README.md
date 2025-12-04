# Text Segmenter

Byte-wise segmentation of 38 content types (including shell, HTML, CSS, JS, ...) in JAX/Flax.

The model works directly on bytes and predicts, for every position, which language or content type it belongs to (e.g. `html`, `css`, `javascript_typescript`, `python`, `encoding_base64`, …). The repo contains everything needed to go from raw code corpora to training data, checkpoints, evaluation reports, and an interactive viewer.

---

TL;DR: How can I segment my text?

0. install `uv`
1. get the packages by `uv sync` and activate the venv 
2. download the model: https://drive.google.com/file/d/1HvrDp0NOuSr_xVjzzBSxmLg8LsrKk98k/view?usp=sharing
3. run the segment viewer: `python interactive_viewer.py --ckpt best_model.msgpack`
4. Open http://127.0.0.1:8000 or use the API: http://127.0.0.1:8000/openapi.json (if `interactive_viewer.py` is started with `--openapi`)

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
  - `evaluation.py` loads a checkpoint, runs the full benchmark battery, and writes a Markdown report to `evaluation/report.md`.
  - `report.md` contains the metrics of the last run evaluation setting. All run metrics are also exported as json and pushed to `comparisons`:
- `comparisons/` – JSON snapshots for comparing different evaluation runs.
- `viewers/` – Small FastAPI frontends for playing with the model:
  - `interactive_viewer.py` exposes a `/api/segment` endpoint where one can provide custom inputs to the model and see its output probabilities.
  - `confusion_viewer.py`, `evaluation_viewer.py`, `dataset_viewer.py` are focused viewers for confusion matrices, eval runs, and datasets.
- `gemini_segmentations/`, `gemini_output_logs/` – Expected locations for LLM‑based segmentations and their logs (used by `obtain_eval_dataset.py` when present).

If you’re trying to understand how things fit together, a good path is:
`downloader/main.py` → `train/utils/window_generator.py` → `train/main.py` → `evaluation/obtain_eval_dataset.py` → `evaluation/evaluation.py` → `viewers/interactive_viewer.py`.

---

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
  --data-root evaluation/data \
  --report-path evaluation/report.md
```

The script runs the model over all configured tasks, aggregates metrics (including IoU coverage for needles, markdown mixes, and malicious payloads), and writes a Markdown report similar to the existing `evaluation/report.md`.

### 5. Launch the interactive viewer

To quickly inspect how a checkpoint segments real snippets:

```bash
python viewers/interactive_viewer.py \
  --ckpt train/checkpoints/seg-unet1d.msgpack \
```

Then open `http://127.0.0.1:8000` in a browser. The UI will show color‑coded spans, per‑character probabilities, and the sliding windows the model actually processed.

---

## Notes

- The canonical label ordering and IDs live in `train/utils/config.py` (`LANG_ORDER`, `LANG2ID`, `ID2LANG`). Training, evaluation, and the viewers all rely on this mapping being consistent.
- This is a research‑style codebase rather than a polished library. Some parts certainly are a bit too over-engineered (e.g. window generators, eval tasks). 🙂
