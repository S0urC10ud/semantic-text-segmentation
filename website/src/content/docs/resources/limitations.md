---
title: Known Limitations
description: The scope and practical limits of content-type segmentation.
---
TypeSeg is a structural signal for text analysis. Use it alongside file identification, validation, and security analyzers.

- **Content type is not maliciousness.** Legitimate text often mixes languages. A predicted encoding or script region is a candidate for inspection, not a detection verdict.
- **Fragments can be ambiguous.** Very short code can fit multiple languages. Exact boundaries can differ from a parser’s boundaries; route complete, validated inputs to analyzers.
- **A fixed textual vocabulary.** The model learns 35 types and focuses on ASCII syntax. Non-ASCII characters use placeholders. `other` is a confidence-based fallback with imperfect unknown-type detection.
- **Text, rather than arbitrary binary formats.** Executables, archives, and images need other tools. TypeSeg assigns one apparent type per position, so overlapping interpretations of the same bytes are outside the task.
- **No automatic payload decoding.** Encoding labels identify visible regions. Decoding and recursive inspection belong in downstream components.
- **Model and domain matter.** General checkpoints perform differently from the security-adapted variants. Short guests remain challenging, and the 165-file security audit does not represent all malware.

The browser demo additionally normalizes line endings and unsupported characters. Its offsets describe the normalized input. Preserve the original decoding map when you need byte-accurate file annotations.

For an integration, evaluate errors and downstream usefulness on your own text corpus. The paper measures segmentation and validator-accepted routing; an end-to-end reduction in missed threats or analyst effort still needs evaluation.
