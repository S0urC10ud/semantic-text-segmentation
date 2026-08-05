# TypeSeg

**Fine-grained, character-level content-type segmentation for textual inputs.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/S0urC10ud/semantic-text-segmentation/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/typeseg.svg)](https://pypi.org/project/typeseg/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/demo-typeseg.martin--dallinger.me-f2994a.svg)](https://typeseg.martin-dallinger.me)

`typeseg` labels every character position of a text with one of 35 content types
(`html`, `css`, `javascript_typescript`, `python`, `powershell`, `encoding_base64`, …),
recovering the internal structure of mixed, malformed, or convention-breaking inputs.
Runtime dependencies are just `numpy` and `onnxruntime` — both pulled by `pip install
typeseg`, so the fast ONNX CPU backend works out of the box at arbitrary input length,
no GPU required. (A pure-numpy fallback also ships, used when onnxruntime is absent.)

<p align="center">
  <img src="https://typeseg.martin-dallinger.me/images/llm_segmentation.png" alt="Correct model output on a heavily mangled HTML input with a hidden shell payload" width="460">
</p>
<p align="center">
  <em>Model output on a heavily mangled input (stress-test): the script/style tags are missing and a
  reverse shell targeting an LLM is hidden inside an HTML comment, yet the embedded <code>shell</code>
  region is recovered correctly. Color saturation reflects per-character confidence.</em>
</p>

```bash
pip install typeseg              # CPU: numpy + ONNX Runtime (fast; the default)
pip install "typeseg[gpu]"       # + CUDA: onnxruntime-gpu (U-Net) and cupy (Mamba scan)
```

```python
import typeseg

text = "<div>hi</div>\n.btn { color: red; }\nalert('x');"

result = typeseg.precise(text)   # Mamba: highest quality, long-context  (recommended)
# result = typeseg.fast(text)    # U-Net: faster, when throughput matters more than quality

for seg in result.segments:
    print(f"{seg.start:>4}-{seg.end:<4} {seg.label:<22} {seg.confidence:.2f}")

#    0-14   html                   0.86
#   14-35   css                    0.91
#   35-46   javascript_typescript  0.87
```

Installing the package also gives you a `typeseg` command (alias `segcat`) that
renders a file — tinted by content type, with a legend and a per-segment
confidence table — straight in the terminal, no Python REPL needed:

```bash
typeseg file.html              # segment a file with Mamba (precise)
typeseg --model fast file.html # use the faster U-Net instead
cat foo.txt | typeseg          # read from stdin
typeseg --demo                 # built-in mixed / prompt-injection sample
python -m typeseg --demo       # equivalent module form
```

<p align="center">
  <img src="https://typeseg.martin-dallinger.me/images/typeseg-example.png" alt="typeseg segmenting a mixed CSS/JS/HTML/SQL/shell input in the terminal" width="760">
</p>

A `Segmentation` exposes:

- `segments` — merged runs as `Segment(start, end, label, confidence, text)`,
- `char_labels` — per-character label (length `len(text)`),
- `char_confidence` — per-character confidence in `[0, 1]`,
- `char_probs` — the full per-character probability distribution, a float32 array of
  shape `(len(text), len(labels))` whose rows sum to ~1. This is the **raw model
  output** (before post-processing relabelling); columns follow `labels`. The open-set
  `other` class is *not* a column — it is derived by confidence gating, so a character
  routed to `other` still keeps its distribution over the known classes here.
- `labels` — the class names, in `char_probs` column order.

```python
r = typeseg.precise("SELECT 1")
r.char_probs.shape            # (8, 35)
r.char_distribution(0)        # {'sql': 0.99, 'text': 0.001, ...} for char 0
```

When printed to a terminal, a `Segment` renders as a colour-tinted chip of its text
(matching the interactive viewer). Colour is auto-detected: it is emitted only to a TTY
and honours `NO_COLOR`; force it with `TYPESEG_COLOR=always` or disable with
`TYPESEG_COLOR=never`.

## Content types

The models classify each character into one of **35 content types**, plus the open-set
`other`. The 35 names below are exactly the `char_probs` columns (see `typeseg.precise("").labels`
for the live column order); `other` (index 35) is the virtual class with no column, assigned by
confidence gating to below-threshold / unknown characters.

| category | classes |
|---|---|
| **Programming languages** (17) | `python`, `javascript_typescript`, `java`, `c_family` (C / C++ / Obj-C), `csharp` (C#), `go`, `rust`, `ruby`, `php`, `swift`, `kotlin`, `scala`, `dart`, `visual_basic`, `shell` (sh / bash), `powershell`, `sql` |
| **Markup & docs** (7) | `html`, `xml`, `svg`, `css`, `markdown`, `restructuredtext`, `tex` (TeX / LaTeX) |
| **Data & config** (6) | `json`, `yaml`, `csv`, `text` (plain natural language), `dockerfile`, `gettext_catalog` (gettext `.po`) |
| **Binary-to-text encodings** (5) | `encoding_hex`, `encoding_base64`, `encoding_base32`, `encoding_base58`, `encoding_base85` (Ascii85) |
| **Open-set** | `other` — below-threshold / unrecognised; *not* a `char_probs` column |

## Use cases

<p align="center">
  <img src="https://typeseg.martin-dallinger.me/images/use_cases.png" alt="Representative use cases for granular content-type segmentation" width="820">
</p>

Content-aware routing and LLM-agent guardrails, span-level scanning of mixed/encoded payloads,
structure recovery in malformed or convention-breaking inputs, and dataset triage — moving the
decision from the whole file down to individual character spans. See the
[project README](https://github.com/S0urC10ud/semantic-text-segmentation#use-cases) for details.

### Backends

`pip install typeseg` pulls `numpy` and `onnxruntime`, so both models run on the fast
ONNX CPU backend at **arbitrary input length** out of the box (≈8× faster for the
U-Net, ≈1.5× for Mamba vs. the pure-numpy loop). The `gpu` extra transparently swaps
in CUDA backends — the API and output are identical (verified bit-close). A pure-numpy
fallback still ships in the box and runs whenever onnxruntime is unavailable or
`TYPESEG_BACKEND=numpy` is set. Inspect the active backend:

```python
import typeseg
typeseg.backend_info()
# {'backend': 'onnx', 'gpu': True, 'precise_gpu': True,
#  'fast_providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
#  'precise_providers': ['CuPyCUDA:NVIDIA GeForce RTX 5070 Laptop GPU']}
```

Providers report what each model **actually loaded** — if the CUDA provider fails to
initialise (missing CUDA/cuDNN libraries) it falls back to CPU and is reported as CPU
honestly.

**Per-model device (auto).** The two models reach the GPU by different routes:

- `fast()` (U-Net) runs on **onnxruntime-gpu** — ~2× faster than CPU end-to-end; ~140k chars/s.
- `precise()` (Mamba) runs its selective-scan through a **custom CUDA scan kernel**
  (a CuPy `RawKernel`, `cupy-cuda12x`), reaching ~58k tokens/s for the raw forward —
  edging out the research model's JAX `associative_scan` (54.9k tok/s). The selective-scan
  is a first-order linear (associative) recurrence; the stock ONNX `Scan` op evaluates it
  one timestep per kernel launch and is actually ~4× *slower* on CUDA than CPU. Our kernel
  instead gives **one GPU thread per inner channel** — each thread holds its own state
  vector in registers and sweeps the whole sequence in a single launch — so the ONNX Mamba
  path always stays on CPU and CuPy carries the GPU acceleration instead (~100× over the
  ONNX `Scan` path). When CuPy/GPU is absent, `precise()` falls back to ONNX (or numpy) on CPU.

Select the backend with the `TYPESEG_BACKEND` environment variable:

| `TYPESEG_BACKEND` | behaviour |
|---|---|
| *(unset)* / `onnx` / `cpu` | auto: U-Net on CUDA (onnxruntime) when it loads, Mamba on CuPy CUDA when present; otherwise CPU/numpy |
| `numpy` | force the pure-numpy backend |
| `gpu` / `cuda` | **require** CUDA for *both* models — no CPU fallback; raise immediately if it cannot initialise |

`gpu`/`cuda` is the "fail fast" mode: rather than silently running on CPU it errors if
the GPU backends are missing, the CUDA provider is absent, or CUDA fails to load. GPU
needs `pip install "typeseg[gpu]"` (onnxruntime-gpu for the U-Net, CuPy for the Mamba
scan) plus CUDA 12.x + cuDNN 9.x on the library path.

#### Running on GPU

Requires an NVIDIA GPU with the CUDA 12.x runtime and cuDNN 9.x. The simplest setup
pulls the CUDA libraries as pip wheels so nothing has to be installed system-wide:

```bash
# 1. The GPU extra: onnxruntime-gpu (U-Net) + cupy-cuda12x (Mamba scan)
pip install "typeseg[gpu]"

# 2. CUDA 12 + cuDNN 9 libraries that onnxruntime-gpu needs (CuPy bundles its own).
#    Skip any you already have system-wide.
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
            nvidia-cufft-cu12 nvidia-curand-cu12

# 3. Put those wheels' libs on the loader path (one-off per shell; not needed if
#    CUDA/cuDNN are already installed system-wide):
export LD_LIBRARY_PATH="$(python - <<'PY'
import os, nvidia, glob
root = os.path.dirname(nvidia.__file__)
print(":".join(glob.glob(os.path.join(root, "*", "lib"))))
PY
):$LD_LIBRARY_PATH"
```

Verify both models are on the GPU:

```python
import typeseg
typeseg.backend_info()
# {'backend': 'onnx', 'gpu': True, 'precise_gpu': True,
#  'fast_providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
#  'precise_providers': ['CuPyCUDA:NVIDIA GeForce RTX 5070 Laptop GPU']}

text = "<div>hi</div>\n.btn { color: red; }\nalert('x');"
typeseg.fast(text)      # U-Net on onnxruntime-gpu  (~140k chars/s)
typeseg.precise(text)   # Mamba on the CuPy scan    (~58k tokens/s raw forward)
```

To make GPU mandatory (raise instead of silently using CPU), set
`TYPESEG_BACKEND=gpu`. Notes:

- The **first** CUDA call compiles kernels — a one-time warmup of seconds (longer on
  brand-new GPU architectures, e.g. Blackwell `sm_120`). Keep the process warm.
- `backend_info()` reflects what each model **actually** loaded; if it shows
  `CPUExecutionProvider` for `fast_providers`, a CUDA/cuDNN library failed to load —
  re-check step 2/3 (a common miss is `libcurand.so.10` → `nvidia-curand-cu12`).
- On Windows use WSL2 for CUDA.

Post-processing mirrors the interactive viewer via an `Options` object; every step
can be tuned or disabled:

```python
from typeseg import precise, Options

result = precise(text, Options(other_threshold=0.30, min_run_chars=3))
result = precise(text, Options(paired_delimiter_fill=False))   # disable one step
```

### Building from source

The model weights (`typeseg/data/*.npz`, `*.onnx`) are generated from the released
checkpoints and are not checked in. From the repository root:

```bash
python scripts/export_typeseg_weights.py   # checkpoints -> data/*.npz + manifest.json
python scripts/export_typeseg_onnx.py      # data/*.npz   -> data/*.onnx
python -m build packages/typeseg   # or: uv build packages/typeseg
```

See the project repository and the accompanying MSc thesis for methodology, models,
and benchmarks. Licensed under Apache-2.0.

## Post-processing methods

The raw model gives a probability vector per character. Post-processing applies a few
cheap local passes (each `O(n)`, no parsing) before segments are exposed. They run in
the order below; toggle each via `Options`. Implementation: `typeseg/_postprocess.py`
(the interactive viewer in `viewers/core.py` has a few additional heuristics).

| step | `Options` flag | what it does |
|---|---|---|
| 1. Whitespace relabeling | `whitespace_relabel` | Whitespace carries no real prediction, so each space/tab/newline copies the label of its nearest non-whitespace neighbour (ties go left). Keeps segments from fragmenting on indentation. *(Thesis step 1.)* |
| 2. Confidence gating | `confidence_gating`, `other_threshold` | A character whose top class probability is below `other_threshold` (default `0.30`) is routed to a virtual open-set `other` label. *(Thesis step 2.)* |
| 3. Boundary snapping | `boundary_snap`, `boundary_snap_max_shift` | A boundary between two runs may shift by ≤ `boundary_snap_max_shift` (default `2`) onto a nearby delimiter symbol (`< > " ' \` ( ) [ ] { } / \ , ; : =`) or whitespace, so cuts land on natural seams. A shift is rejected if a moved non-delimiter position's destination-label probability is materially lower than the model's. *(Thesis step 3.)* |
| 4. Minimum-run normalization | `min_run_normalize`, `min_run_chars` | Interior runs shorter than `min_run_chars` (default `3`) are absorbed into the better-supported neighbour (`AABBAA → AAAAAA`); short *edge* runs are kept. Removes single-character noise. *(Thesis step 4.)* |
| 5. Paired-delimiter fill | `paired_delimiter_fill`, `paired_delimiter_max_shift` | For a run wrapped by the same host on both sides (`A B A`), nudge both edges (≤ shift, default `2`) so the inner run sits **inside a matching delimiter pair** — `" "`, `' '`, `` ` ` ``, `( )`, `[ ]`, `{ }` — pushing the delimiters out to the host. Example: in `onclick="alert(…)"` the `="` is ejected to `html` and only `alert(…)` stays `javascript`. **Not part of the thesis four steps; matches the deployment viewer.** |

Steps 1–4 are the four steps documented in the thesis (off by default for the thesis
benchmark numbers; on by default here for deployment-style output). Step 5 is an extra
the interactive viewer also applies. The viewer has further heuristics
(`markdown_structure_fill`, `local_host_fill`, `newline_snap`) not ported here — see
`viewers/core.py` for those.

These passes are cheap, but on long inputs they run on the CPU and can outweigh the
model's own forward pass — so the highest throughput is with post-processing off,
exposing the raw per-character argmax:

```python
raw = Options(whitespace_relabel=False, confidence_gating=False, boundary_snap=False,
              min_run_normalize=False, paired_delimiter_fill=False)
result = precise(text, raw)
```

The trade-off is noisier segments (single-character runs, frayed boundaries); the
default keeps post-processing on for cleaner, deployment-style output.

## Acknowledgments

This work was carried out with extensive technical feedback from and many informative discussions with
Dr. Yanick Fratantonio and Dr. Luca Invernizzi from Google Security Research (authors of
[Magika](https://github.com/google/magika)), who also provided generous access to Google Cloud compute
resources. Gemini models were used to create and refine the dense segment annotations and to act as the
active-learning oracle. Thanks also go to Univ.-Prof. Stefan Rass (JKU Secure Systems Group) for his
guidance, especially regarding parsing and classical computer-science approaches. I am especially
grateful for his detailed feedback on the accompanying master's thesis and for him supporting this
academic collaboration.
