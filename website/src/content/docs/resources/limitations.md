---
title: Known Limitations
description: The scope and practical limits of content-type segmentation.
---
TypeSeg labels the content types in decoded text. These limits apply when inspecting results or using them in other tools.

- **Security interpretation.** Legitimate text often mixes languages. Content-type labels locate regions for inspection; security verdicts require separate analysis.
- **Fragments can be ambiguous.** Very short code can fit multiple languages. Exact boundaries can differ from a parser’s boundaries; route complete, validated inputs to analyzers.
- **A fixed textual vocabulary.** The model learns 35 types and focuses on ASCII syntax. Non-ASCII characters use placeholders. `other` is a confidence-based fallback with imperfect unknown-type detection.
- **Decoded text inputs.** Executables, archives, and images need format-specific tools. TypeSeg assigns one apparent type per position, which limits its treatment of overlapping interpretations.
- **Encoded regions.** Encoding labels identify visible text. Separate decoders handle decoding and recursive inspection.
- **Model and domain.** General checkpoints perform differently from the security-adapted variants. Short embedded regions remain challenging. The security audit covers a stratified sample of 165 text files; results can vary on other corpora.

The browser demo additionally normalizes line endings and unsupported characters. Its offsets describe the normalized input. Preserve the original decoding map when you need byte-accurate file annotations.

Evaluate errors and runtime on representative text from your application. The paper measures segmentation and validator-accepted routing. Effects on threat detection and analyst effort need further evaluation.
