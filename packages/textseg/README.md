# textseg

**Fine-grained, character-level content-type segmentation for textual inputs.**

`textseg` labels every character position of a text with one of 35 content types
(`html`, `css`, `javascript_typescript`, `python`, `powershell`, `encoding_base64`, …),
recovering the internal structure of mixed, malformed, or convention-breaking inputs.
The only runtime dependency is `numpy`.

```bash
pip install textseg              # CPU, numpy only
pip install "textseg[onnx]"      # + ONNX Runtime (faster CPU)
pip install "textseg[gpu]"       # + CUDA: onnxruntime-gpu (U-Net) and cupy (Mamba scan)
```

```python
import textseg

text = "<html><body>IgnoreAbovecG93ZXJzaGVsbA==</body></html>"

result = textseg.precise(text)   # Mamba: highest quality, long-context  (recommended)
result = textseg.fast(text)      # U-Net: faster, when throughput matters more than quality

for seg in result.segments:
    print(f"{seg.start:>4}-{seg.end:<4} {seg.label:<22} {seg.confidence:.2f}")
```

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
r = textseg.precise("SELECT 1")
r.char_probs.shape            # (8, 35)
r.char_distribution(0)        # {'sql': 0.99, 'text': 0.001, ...} for char 0
```

When printed to a terminal, a `Segment` renders as a colour-tinted chip of its text
(matching the interactive viewer). Colour is auto-detected: it is emitted only to a TTY
and honours `NO_COLOR`; force it with `TEXTSEG_COLOR=always` or disable with
`TEXTSEG_COLOR=never`.

### Backends

The only required dependency is `numpy`; both models run at **arbitrary input
length** on it. The `onnx`/`gpu` extras transparently swap in faster backends — the
API and output are identical (verified bit-close). Inspect the active backend:

```python
import textseg
textseg.backend_info()
# {'backend': 'onnx', 'gpu': True, 'precise_gpu': True,
#  'fast_providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
#  'precise_providers': ['CuPyCUDA:NVIDIA GeForce RTX 5070 Laptop GPU']}
```

Providers report what each model **actually loaded** — if the CUDA provider fails to
initialise (missing CUDA/cuDNN libraries) it falls back to CPU and is reported as CPU
honestly.

**Per-model device (auto).** The two models reach the GPU by different routes:

- `fast()` (U-Net) runs on **onnxruntime-gpu** — ~3–4× faster on GPU; ~140k chars/s.
- `precise()` (Mamba) runs its selective-scan through a **custom CUDA scan kernel**
  (a CuPy `RawKernel`, `cupy-cuda12x`), reaching ~58k tokens/s for the raw forward —
  edging out the research model's JAX `associative_scan` (54.9k tok/s). The selective-scan
  is a first-order linear (associative) recurrence; the stock ONNX `Scan` op evaluates it
  one timestep per kernel launch and is actually ~4× *slower* on CUDA than CPU. Our kernel
  instead gives **one GPU thread per inner channel** — each thread holds its own state
  vector in registers and sweeps the whole sequence in a single launch — so the ONNX Mamba
  path always stays on CPU and CuPy carries the GPU acceleration instead (~100× over the
  ONNX `Scan` path). When CuPy/GPU is absent, `precise()` falls back to ONNX (or numpy) on CPU.

Select the backend with the `TEXTSEG_BACKEND` environment variable:

| `TEXTSEG_BACKEND` | behaviour |
|---|---|
| *(unset)* / `onnx` / `cpu` | auto: U-Net on CUDA (onnxruntime) when it loads, Mamba on CuPy CUDA when present; otherwise CPU/numpy |
| `numpy` | force the pure-numpy backend |
| `gpu` / `cuda` | **require** CUDA for *both* models — no CPU fallback; raise immediately if it cannot initialise |

`gpu`/`cuda` is the "fail fast" mode: rather than silently running on CPU it errors if
the GPU backends are missing, the CUDA provider is absent, or CUDA fails to load. GPU
needs `pip install "textseg[gpu]"` (onnxruntime-gpu for the U-Net, CuPy for the Mamba
scan) plus CUDA 12.x + cuDNN 9.x on the library path.

#### Running on GPU

Requires an NVIDIA GPU with the CUDA 12.x runtime and cuDNN 9.x. The simplest setup
pulls the CUDA libraries as pip wheels so nothing has to be installed system-wide:

```bash
# 1. The GPU extra: onnxruntime-gpu (U-Net) + cupy-cuda12x (Mamba scan)
pip install "textseg[gpu]"

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
import textseg
textseg.backend_info()
# {'backend': 'onnx', 'gpu': True, 'precise_gpu': True,
#  'fast_providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
#  'precise_providers': ['CuPyCUDA:NVIDIA GeForce RTX 5070 Laptop GPU']}

textseg.fast(text)      # U-Net on onnxruntime-gpu  (~140k chars/s)
textseg.precise(text)   # Mamba on the CuPy scan    (~58k tokens/s raw forward)
```

To make GPU mandatory (raise instead of silently using CPU), set
`TEXTSEG_BACKEND=gpu`. Notes:

- The **first** CUDA call compiles kernels — a one-time warmup of seconds (longer on
  brand-new GPU architectures, e.g. Blackwell `sm_120`). Keep the process warm.
- `backend_info()` reflects what each model **actually** loaded; if it shows
  `CPUExecutionProvider` for `fast_providers`, a CUDA/cuDNN library failed to load —
  re-check step 2/3 (a common miss is `libcurand.so.10` → `nvidia-curand-cu12`).
- On Windows use WSL2 for CUDA.

Post-processing mirrors the interactive viewer via an `Options` object; every step
can be tuned or disabled:

```python
from textseg import precise, Options

result = precise(text, Options(other_threshold=0.30, min_run_chars=3))
result = precise(text, Options(paired_delimiter_fill=False))   # disable one step
```

### Building from source

The model weights (`textseg/data/*.npz`, `*.onnx`) are generated from the released
checkpoints and are not checked in. From the repository root:

```bash
python scripts/export_textseg_weights.py   # checkpoints -> data/*.npz + manifest.json
python scripts/export_textseg_onnx.py      # data/*.npz   -> data/*.onnx
python -m build packages/textseg   # or: uv build packages/textseg
```

See the project repository and the accompanying MSc thesis for methodology, models,
and benchmarks. Licensed under Apache-2.0.

## Post-processing methods

The raw model gives a probability vector per character. Post-processing applies a few
cheap local passes (each `O(n)`, no parsing) before segments are exposed. They run in
the order below; toggle each via `Options`. Implementation: `textseg/_postprocess.py`
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
