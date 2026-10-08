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

Run TypeSeg on decoded text to locate its content types. The regions can be displayed in a source viewer or passed to further analysis tools. A file detector such as Magika can help select textual inputs.

TypeSeg predicts labels from the input text. Parsing, execution, and recursive decoding are separate tasks. A detected script region can be incomplete; an analyzer may need surrounding syntax and validation before it can process the region.
