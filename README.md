# TypeSeg

**Fine-grained, character-level content-type segmentation for textual (arbitrarily mangled) inputs.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![PyPI](https://img.shields.io/pypi/v/typeseg.svg)](https://pypi.org/project/typeseg/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/demo-typeseg.martin--dallinger.me-f2994a.svg)](https://typeseg.martin-dallinger.me)

`typeseg` localizes *where* each content type begins and ends inside a text stream, labeling
**every character position** with one of 35 textual content types (`html`, `css`,
`javascript_typescript`, `python`, `powershell`, `encoding_base64`, …). Unlike file-level detectors
that reduce a heterogeneous input to a single label, it recovers the internal structure of mixed,
malformed, or convention-breaking inputs — for example a reverse shell hidden inside an HTML comment,
or a Base64 payload pasted into an LLM prompt.

<p align="center">
  <img src="./images/llm_segmentation.png" alt="Correct model output on a heavily mangled HTML input with a hidden shell payload" width="460">
</p>
<p align="center">
  <em>Model's output on a heavily mangled input (stress-test): the script/style tags are missing and a reverse shell
  targeting an LLM is hidden inside an HTML comment, yet the embedded <code>shell</code> region is
  recovered correctly. Color saturation reflects per-character confidence.</em>
</p>

---

## Highlights

- **Character-level boundaries.** Predicts a content type for every character, so a segment boundary
  can land between any two adjacent characters. Subword tokenizers cannot represent such boundaries —
  on `IgnoreAbovecG93ZXJzaGVsbA==`, a BPE vocabulary merges `IgnoreAbove` with the start of the
  Base64 payload, contaminating the transition; `typeseg` does not.
- **Two models, one API.** A long-context bidirectional **Mamba** state-space model is the
  recommended default for the best segmentation quality; a fast, heavily parallelizable **U-Net**
  trades some quality for much higher throughput on near-pure inputs.
- **Best-effort on broken inputs.** Designed for mangled, truncated, or weakly-delimited text mixed
  with natural language — no delimiters or grammars required.
- **Open-set aware.** Out-of-distribution regions are routed to an auxiliary `other` label instead of
  being misclassified into a known type.
- **Fast and portable.** GPU when available, but easily installable and quick on CPU. The inference
  package depends only on `numpy` and `onnxruntime`.
- **Engineered for runtime.** GPU Mamba runs on a custom CUDA scan kernel (one thread per
  channel, single launch) that **beats the ONNX `Scan` op by ~100×** and edges out the research JAX
  `associative_scan` at ~58,000 tok/s. The browser demo reimplements the same model in JavaScript +
  **WebGPU**, fully client-side at arbitrary input length.
- **Per-character confidence.** Every position carries a confidence score, exposed alongside the
  merged segments.


## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Use cases](#use-cases)
- [Options](#options)
- [Live demo](#live-demo)
- [Supported content types](#supported-content-types)
- [Models](#models)
- [Benchmarks](#benchmarks)
- [How it works](#how-it-works)
- [Development (research code)](#development-research-code)
- [Limitations & non-goals](#limitations--non-goals)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

```bash
pip install typeseg              # CPU: numpy + ONNX Runtime (fast; the default)
```

The wheel bundles both models and runs them at **arbitrary input length** on the fast ONNX Runtime
CPU backend out of the box. For CUDA, install the `gpu` extra (identical API and output); `typeseg`
auto-selects the GPU when one is present:

```bash
pip install "typeseg[gpu]"       # GPU/CUDA: onnxruntime-gpu (U-Net) + cupy (Mamba scan)
```

A pure-numpy fallback still ships in the wheel for environments where onnxruntime is unavailable
(or with `TYPESEG_BACKEND=numpy`).

## Quick start

`typeseg` exposes two functions that mirror the two models:

```python
import typeseg

text = "<div>hi</div>\n.btn { color: red; }\nalert('x');"

# Recommended: highest-quality, long-context segmentation (Mamba).
result = typeseg.precise(text)

# Faster alternative, when throughput matters more than quality (U-Net):
#   result = typeseg.fast(text)

for seg in result.segments:
    print(f"{seg.start:>4}-{seg.end:<4} {seg.label:<22} conf={seg.confidence:.2f}")
    print(f"     {text[seg.start:seg.end]!r}")

#    0-14   html                   conf=0.86   '<div>hi</div>\n'
#   14-35   css                    conf=0.91   '.btn { color: red; }\n'
#   35-46   javascript_typescript  conf=0.87   "alert('x');"
```

The bundled `examples/segcat.py` renders the result in the terminal:

<p align="center">
  <img src="./images/typeseg-example.png" alt="typeseg segmenting a mixed CSS/JS/HTML/SQL/shell input in the terminal" width="760">
</p>

Both functions return a `Segmentation` with:

- `segments` — merged runs as `Segment(start, end, label, confidence, text)`,
- `char_labels` — the per-character label,
- `char_confidence` — the per-character confidence,
- `char_probs` — the full per-character probability distribution (`float32`,
  shape `(len(text), len(labels))`, rows sum to ~1) — the raw model output before
  post-processing; columns follow `labels`. `other` is open-set and not a column.
- `labels` — the class names, in `char_probs` column order.

Printed in a terminal, a `Segment` shows a colour-tinted chip of its text (auto-detected:
TTY only, honours `NO_COLOR`; force with `TYPESEG_COLOR=always`).

**Use `precise()` (Mamba) by default** — it gives the best segmentation quality and handles heavily
segmented or long inputs where far-reaching context matters. Reach for `fast()` (U-Net) only when you
need maximum throughput and the input is near-pure, where the quality gap is small.

## Options

Post-processing mirrors the interactive viewer and is controlled through an `Options` object:

```python
from typeseg import precise, Options

opts = Options(
    other_threshold=0.30,          # route positions below this confidence to `other`
    min_run_chars=3,               # absorb runs shorter than this into their neighbors
    boundary_snap_max_shift=2,     # max chars a boundary may snap to nearby whitespace
    paired_delimiter_max_shift=2,  # max chars a wrapped run may snap into a delimiter pair
    whitespace_relabel=True,       # relabel boundary whitespace to its host segment
    confidence_gating=True,        # enable open-set routing to `other`
    boundary_snap=True,            # enable local boundary snapping
    min_run_normalize=True,        # enable minimum-run normalization
    paired_delimiter_fill=True,    # snap delimiter-wrapped runs inside their pair
)

result = precise(text, opts)
```

Each step and its provenance (which of the thesis's four steps, which are viewer extras)
is documented in [`packages/typeseg/README.md`](packages/typeseg/README.md#post-processing-methods).

## Use cases

<p align="center">
  <img src="./images/use_cases.png" alt="Representative use cases for granular content-type segmentation" width="820">
</p>

- **Content-aware routing & LLM-agent guardrails.** Move the routing decision from the whole file
  (as in Magika) down to individual spans: send `shell`, `powershell`, `json`, or `yaml` regions to
  type-specific checks while surrounding prose is skipped. A useful first layer in a defense-in-depth
  pipeline for flagging command-like or encoded regions before they reach a tool or interpreter.
- **Upload hardening & polyglot-style bypass resistance.** Make script-like regions hidden inside
  benign-looking carriers inspectable, even when they are short, malformed, or weakly delimited.
- **Digital forensics on damaged or metadata-free data.** Infer type from content alone and return
  precise span boundaries instead of coarse, block-level ones — useful for recovered text where the
  exact span of a foreign-typed region matters.
- **Repository, document & stream inspection.** Provide content-type hints (e.g. for syntax
  highlighting) when a file extension or declared media type is missing, conflicting, or incomplete.


## Live demo

A fully client-side viewer runs the Mamba model **directly in your browser** — no server, no upload,
no ONNX runtime: **[typeseg.martin-dallinger.me](https://typeseg.martin-dallinger.me)**. The model is
reimplemented from scratch in JavaScript, with the heavy dense projections **offloaded to WebGPU** (a
custom WGSL matmul) and the selective-scan recurrence streamed over the full input. It runs at
**arbitrary input length entirely on the client GPU** — the same variable-length segmentation as the
Python package, with zero backend.

## Supported content types

The 35 supervised content types, plus the virtual open-set `other` label:

- **Code & markup:** `c_family`, `csharp`, `css`, `dart`, `dockerfile`, `go`, `html`, `java`,
  `javascript_typescript`, `json`, `kotlin`, `php`, `powershell`, `python`, `ruby`, `rust`, `scala`,
  `shell`, `sql`, `swift`, `visual_basic`, `xml`, `yaml`
- **Text & docs:** `csv`, `gettext_catalog`, `markdown`, `restructuredtext`, `svg`, `tex`, `text`
- **Encodings:** `encoding_base32`, `encoding_base58`, `encoding_base64`, `encoding_base85`,
  `encoding_hex`

The label `text` marks only *standalone* prose: any local syntax cue for another content type
overrides it. The `other` label is distinct — it flags structured-looking regions that match no known
content type (e.g. an unseen language or a random character sequence), not natural language.

## Models

Both models are small (roughly 1.5M parameters) and operate on ASCII; non-ASCII characters are
redirected to a dedicated placeholder so the model can still react to them.

| | `precise()` — **Mamba** *(recommended)* | `fast()` — **U-Net** |
|---|---|---|
| Architecture | Bidirectional selective state-space (Mamba) | 1D U-Net over character tokens |
| Context | Long-context, arbitrary length | Fixed window (1536), sliding |
| Output | Piecewise-constant | Piecewise-constant |
| Strongest on | Highest quality; heavily segmented / long-context inputs | Throughput; near-pure host windows |
| Trade-off | Best quality; lower throughput (GPU recommended) | Fastest; weaker on far-reaching dependencies |

Both are trained in stages: weak-label pretraining on coarsely labeled files, dense fine-tuning on
LLM-assisted segmentations that are validated and repaired, and uncertainty-driven active-learning
refinement.

## Benchmarks

The Mamba model reaches a **macro F₁ of 0.951** over the 35 supervised content types on realistic
GitHub files, and tops transition, needle-in-a-haystack, and Markdown-mixture benchmarks. The U-Net
is strongest on near-pure host windows and is by far the faster model.

**Reference model throughput** (the research JAX/Flax models — raw forward pass on 10 KB inputs,
tokens/s; higher is better):

| Model | GPU (notebook-class) | CPU |
|---|---:|---:|
| U-Net (`fast`) | 4,825,475 | 291,560 |
| Mamba (`precise`) | 54,892 | 2,248 |
| Magika (file-level, reference) | — | 196,897 |

The research U-Net comfortably clears the design criterion of 100,000 characters per second.

**`typeseg` package throughput.** The pip package is a dependency-light deployment wrapper
(numpy/ONNX/CuPy inference + sliding-window tiling + character-level post-processing), so it runs
slower than the raw research model above. Measured end-to-end, warm (a laptop RTX 5070 for GPU):

| Entry point | CPU (onnxruntime) | GPU (`typeseg[gpu]`) | Notes |
|---|---:|---:|---|
| `fast()` — U-Net | ~75,000 chars/s | ~140,000 chars/s | piecewise-constant; clears the 100k criterion on GPU |
| `precise()` — Mamba | ~2,000 chars/s | ~25,000–35,000 chars/s | raw model ~58,000 tok/s (see below) |

The two models take different GPU routes (see [Backends](packages/typeseg/README.md#backends)):
the U-Net runs on **onnxruntime-gpu**, while the Mamba selective-scan runs through a **custom
CUDA scan kernel** (a CuPy `RawKernel`). The selective-scan is a first-order linear — and therefore
*associative* — recurrence; the stock ONNX `Scan` op evaluates it one timestep per kernel launch and
ends up *slower* on GPU than on CPU. Our kernel instead assigns **one GPU thread per inner channel**:
each thread keeps its own state vector in registers and sweeps the entire sequence in a single launch,
so the whole forward is one launch of fully parallel work — touching each state element exactly once.
It reaches **~58,000 tokens/s** for the raw Mamba forward, **edging out the research model's JAX
`associative_scan`** (54,892 tok/s) and beating the ONNX `Scan` path by **~100×**. End-to-end
`precise()` is lower because it shares the
character-level post-processing, which is linear in input length (a few hundred milliseconds on a
200 KB input). The **first** CUDA call pays a one-time kernel-compilation warmup (seconds, longer on
brand-new GPU architectures) — keep a process warm for repeated calls.

## How it works

The task is a 1D analog of semantic segmentation: the input is a character stream and the output
assigns a content-type label to each position; contiguous runs of identical labels form a
piecewise-constant segmentation. The `other` label is *virtual* — the models emit exactly 35 logits,
and out-of-distribution positions are surfaced through open-set regularization at training time plus a
maximum-softmax-probability threshold at inference. A four-step, local post-processing pass
(whitespace relabeling, low-confidence routing to `other`, boundary snapping, minimum-run
normalization) cleans up the raw character labels.

The full methodology — data pipeline, architectures, open-set routing, and active learning — is
described in the accompanying MSc thesis (see [Citation](#citation)).

## Development (research code)

This repository also contains the full research stack used to build the dataset, train both model
families, and evaluate them. It is a research-style codebase run via scripts, not an installable
library.

The project targets Python 3.11 and JAX/Flax. On Windows, WSL is recommended for GPU support, since
JAX has no native Windows CUDA support.

The LLM baselines use two released prompt protocols: [`dense_prompt_v2.template`](evaluation/llm_benchmark/dense_prompt_v2.template)
for the base benchmark and [`dense_prompt_security.template`](evaluation/llm_benchmark/dense_prompt_security.template)
for the security benchmark. The latter is the exact security-adapted prompt used for the reported
Gemini 3 Flash results.

```bash
# install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

Run the interactive Python viewer on a bundled checkpoint:

```bash
# Fast U-Net
uv run python viewers/interactive_viewer.py --ckpt checkpoints/unet_al.msgpack --port 8000

# Mamba
uv run python viewers/interactive_viewer.py \
  --ckpt checkpoints/mamba_al.msgpack --arch mamba --model-dim 256 --dtype bfloat16 \
  --mamba-layers 6 --mamba-d-state 16 --mamba-expand 1 \
  --mamba-dt-rank 16 --mamba-conv 4 --mamba-bidirectional --port 8000
```

Then open <http://127.0.0.1:8000>. The viewer shows color-coded spans, per-character probabilities,
and the sliding windows the model actually processed.

The repository is organized as follows:

- `checkpoints/` — the two finalist checkpoints (`unet_al.msgpack`, `mamba_al.msgpack`).
- `downloader/` — streams [The Stack](https://huggingface.co/datasets/bigcode/the-stack) and
  [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400), filters and cleans files, and
  builds Arrow datasets; the monitor/test splits are densely segmented with Gemini.
- `train/` — model definitions and the training driver (`train/main.py`); `utils/config.py` holds the
  canonical label set (`LANG_ORDER`, `LANG2ID`, `ID2LANG`).
- `evaluation/` — benchmark datasets, the evaluation harness, and reports.
- `active_learning/` — the boundary-focused active-learning loop (`infer → acquire → oracle → store`).
- `viewers/` — small FastAPI / browser frontends, including the fully client-side WebGPU viewer.
- `packages/typeseg/` — the minimal, inference-only package published to PyPI (numpy + ONNX Runtime
  core; CuPy/onnxruntime-gpu via the optional `gpu` extra).

A good reading path is `downloader/main.py` → `train/utils/window_generator.py` → `train/main.py` →
`evaluation/obtain_eval_dataset.py` → `evaluation/evaluation.py` → `viewers/interactive_viewer.py`.

### Building the `typeseg` wheel

The package's bundled weights are regenerated from the released checkpoints (they are not committed):

```bash
python scripts/export_typeseg_weights.py   # checkpoints/*.msgpack -> packages/typeseg/typeseg/data/*.npz
python scripts/export_typeseg_onnx.py      # *.npz                 -> dynamic-length *.onnx graphs
uv build packages/typeseg          # -> packages/typeseg/dist/*.whl
```

Re-building the Gemini-augmented dataset or running the Gemini oracle requires
`GOOGLE_API_KEY` to be set; access to the Hugging Face datasets above is needed to rebuild the
training data rather than just run the bundled checkpoints.

## Limitations & non-goals

- **Short-fragment ambiguity.** At short context lengths a fragment can be valid under several
  grammars (e.g. Java vs C#, or `print("hello")` across Python/Ruby/R/Lua). The goal is best-effort
  localization at scale, not perfect language identification for every plausible fragment.
- **ASCII focus.** Experiments are restricted to ASCII, since the supported languages' syntax relies
  predominantly on it; non-ASCII is handled via a placeholder rather than modeled directly.
- **Not a general prompt-injection detector.** Segmentation only helps the narrower case where
  untrusted input *embeds* a region of a different content type; an instruction written in plain prose
  carries no distinguishing content type.
- **No formal guarantees.** This is a probabilistic best-effort heuristic, not a parser; it provides
  no correctness guarantee on well-formed grammars.

## Citation

If you use `typeseg` in your research, please cite the thesis:

```bibtex
@mastersthesis{dallinger2026typeseg,
  author  = {Dallinger, Martin},
  title   = {Fine-Grained Content-Type Segmentation of Mixed Text Using Deep Learning},
  school  = {Johannes Kepler University Linz},
  address = {Linz, Austria},
  type    = {Master's thesis},
  year    = {2026}
}
```

## License

`typeseg` is licensed under the [Apache License 2.0](./LICENSE).

## Acknowledgments

This work was carried out with extensive technical feedback from and many informative discussions with
Dr. Yanick Fratantonio and Dr. Luca Invernizzi from Google Security Research (authors of
[Magika](https://github.com/google/magika)), who also provided generous access to Google Cloud compute
resources. Gemini models were used to create and refine the dense segment annotations and to act as the
active-learning oracle. Thanks also go to Univ.-Prof. Stefan Rass (JKU Secure Systems Group) for his
guidance, especially regarding parsing and classical computer-science approaches. I am especially
grateful for his detailed feedback on the accompanying master's thesis and for him supporting this
academic collaboration.
