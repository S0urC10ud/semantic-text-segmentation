---
title: How TypeSeg Works
description: From decoded text to contiguous, character-level content-type regions.
---
TypeSeg takes decoded text and predicts a content type at every character position. Consecutive positions with the same label become a segment.

## Three steps

1. **Map the input.** A compact character representation preserves the syntax cues used by the supported types. The models focus on printable ASCII; other characters use placeholders.
2. **Score every position.** U-Net processes overlapping 1,536-character windows. Mamba propagates context across the file. Both produce class probabilities.
3. **Build readable regions.** Confidence gating can assign `other` to uncertain positions. Local post-processing relabels whitespace, adjusts boundaries, and absorbs very short runs.

The Python API exposes segments and the per-character distributions. Post-processing can be configured through `typeseg.Options`; defaults and implementation details are documented in the [package reference](https://github.com/S0urC10ud/semantic-text-segmentation/tree/main/packages/typeseg#post-processing-methods).

## Where it fits

A file-level detector answers “What kind of file is this?” TypeSeg adds “Which types appear inside this text, and where?” Run it after selecting textual inputs, then use its regions for visualization or downstream analysis.

TypeSeg does not parse, execute, or recursively decode the input. A detected script region can be a fragment rather than a complete program. An analyzer may need expansion to surrounding syntax and validation before it can consume a region.
