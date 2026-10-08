---
title: Use Cases
description: Inspect mixed text, locate embedded languages and encodings, and review text datasets with TypeSeg.
---
TypeSeg labels the content types within a text file. The resulting regions can support source viewers, analysis scripts, and dataset review.

## Inspect mixed text

Display regions in file order, with colors and labels for each type. Selecting a region can highlight its source. This makes it easier to inspect PowerShell inside a Python script, JavaScript inside markup, or Base64 embedded in another language. The [browser demo](../../introduction/web-demo/) shows a composition bar and highlighted text.

The model also handles incomplete or malformed text, including web content with missing script or style tags. Short fragments and boundaries can still be ambiguous; inspect predictions alongside the source.

## Analyze regions by content type

Content-type labels can help locate input for language-specific tools or decoders:

1. Decode the text and retain a mapping to the original bytes if needed.
2. Run `typeseg.fast(text)` or `typeseg.precise(text)`.
3. Select regions by label, such as `powershell` or `encoding_base64`.
4. Expand code fragments to a suitable boundary and validate them before analysis.
5. Pass suitable regions to an analyzer or decoder.

The [Python quick start](../../getting-started/quick-start/) shows how to read segment labels and ranges. The paper evaluates region expansion and validation in [Performance & Evaluation](../performance/).

## Review text datasets

Summarize the types present in a corpus, find mixed-language samples, and select examples for annotation or closer review. Keep the regions and confidence values so predictions can be checked against the original text.

For security use, combine these labels with other analysis signals. Ordinary documents and scripts often contain several languages. [Known limitations](../../resources/limitations/) covers scope, offsets, and model uncertainty.
