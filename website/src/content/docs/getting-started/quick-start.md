---
title: Quick Start
description: Segment a text file with the TypeSeg CLI or Python API.
---
## Command line

Use the fast U-Net model to display a text file with content-type highlighting:

```bash
typeseg --model fast file.txt
```

For the Mamba model, use `typeseg --model precise file.txt`. The CLI defaults to Mamba if you omit `--model`.

## Python

```python
import typeseg

text = '<html><style>body { color: blue; }</style></html>'
result = typeseg.fast(text)  # U-Net: start here for throughput

for segment in result.segments:
    print(segment.label, segment.start, segment.end, segment.confidence)
    print(text[segment.start:segment.end])
```

`result.segments` contains contiguous runs with a label, start, end, confidence, and matched text. Boundaries are **Python character offsets**, with the end excluded. The exact predictions depend on the model and post-processing.

Use `typeseg.precise(text)` for Mamba’s full-file context. Both functions return the same result type, so you can compare them without changing downstream code.

## Build a file-composition view

Preserve segment order. Draw each region with width proportional to `segment.end - segment.start`, and use a consistent color for each label. Link each region to the corresponding source range.

For byte-based viewers, map decoded character offsets to the original encoding before displaying byte ranges. [Understanding the Output](../../core-concepts/understanding-the-output/) includes a UTF-8 example.
