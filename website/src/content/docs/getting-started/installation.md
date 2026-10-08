---
title: Installation
description: Install the TypeSeg Python package for local CPU inference, with optional CUDA support.
---
Install [TypeSeg from PyPI](https://pypi.org/project/typeseg/) with Python 3.9 or later:

```bash
pip install typeseg
```

The package includes the general U-Net and Mamba models. NumPy and ONNX Runtime are installed automatically. No API key, model download step, or GPU is needed for CPU inference.

For throughput-oriented text analysis, start with `typeseg.fast(text)` or `typeseg --model fast file.txt`.

## Optional GPU support

On a compatible NVIDIA CUDA system:

```bash
pip install "typeseg[gpu]"
```

The same Python API selects an available GPU backend. CUDA support uses ONNX Runtime for U-Net and CuPy for Mamba. macOS uses CPU inference; the GPU extra does not provide an Apple GPU backend.

## Check the installation

```bash
python -c "import typeseg; print(typeseg.fast('SELECT name FROM users;'))"
```

Continue with [Quick Start](../quick-start/) for CLI and Python examples, or try the [browser demo](../../introduction/web-demo/) without installing anything.
