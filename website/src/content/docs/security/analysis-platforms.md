---
title: For Analysis Platforms
description: A practical TypeSeg workflow for VirusTotal-style file details, triage, and analyzer routing.
---
TypeSeg can add an internal content map to a file-details page. An analyst sees embedded scripts and encodings at a glance, then jumps directly to the relevant source range.

## Three useful additions

- **Composition view.** Show one bar in file order, colored by content type. Hover reveals type, location, length, and confidence; selection highlights the source. Keep text labels and a segment list so color is not the only cue.
- **Embedded-content navigation.** Expose PowerShell inside another script, JavaScript inside markup, or an encoded region as a separate item to inspect. Use the carrier type and expected structure to distinguish ordinary mixtures from combinations worth reviewing.
- **Analyzer routing.** Use candidate locations to guide language-specific analyzers or decoders. Expand fragmented regions to an appropriate boundary, then validate them before analysis.

These are proposed integration uses. TypeSeg is not currently integrated with VirusTotal, and the paper does not measure improvements to VirusTotal detections or analyst workload.

## Start with the text branch

1. Identify the whole file with your existing detector, such as Magika.
2. Select textual inputs, decode them, and retain the mapping to original bytes.
3. Run `typeseg.fast(text)` with the model loaded once per worker.
4. Store ordered ranges, labels, and confidence alongside the file’s existing metadata.
5. Render the composition view; send selected candidates to downstream validators and analyzers.

Ordinary HTML with CSS and JavaScript is mixed too. Mixedness alone is not evidence of malware. Use TypeSeg’s structural information together with your existing analysis signals.

## Evidence for a pilot

In a stratified audit of **165 MalwareBazaar text files**, 131 contained mixed content and 96 had an unexpected embedded type. The paper’s security-adapted U-Net recovered **81.8% of guest segments** and **91.9% of encoded regions**. This is evidence from that audited set, not a malware-wide prevalence estimate or a result for the general demo model.

The routing experiment also shows why validation matters: raw crops recovered 15.3% of eligible regions; expansion and validation raised recovery to 86.5%, with a 37.3% route match rate. [Performance & Evaluation](../performance/) gives the definitions and trade-offs.

For a pilot, measure local latency by file size, false embedded-type suggestions, validator acceptance, and extra analyzer findings on your own representative text corpus.
