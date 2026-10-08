# Screenshot provenance

These assets reuse the visual examples published on [TypeSeg's PyPI page](https://pypi.org/project/typeseg/). They are presentation images, not fresh benchmark runs.

- `typeseg-terminal.png`: edited from the repository's `images/typeseg-example.png`; shell-command text is redacted in the source and the table. Counts, ranges and confidence values describe the original input.
- `typeseg-annotated.png`: edited from `images/llm_segmentation.png`; the shell-command text is redacted. Original annotations and type highlights are retained.
- The original repository images remain unchanged.

The built-in image generation/editing tool prepared both derivatives. `scripts/prepare-demo.mjs` copies them to the original public image URLs used by the PyPI description. It also restores `images/use_cases.png` unchanged. `TerminalExample.astro` publishes an optimized WebP preview and links to the full PNG.

## Terminal edit prompt

Use case: precise-object-edit. Asset type: existing TypeSeg terminal screenshot for its documentation. Edit target: the supplied local screenshot. Make only two precise redactions: (1) the lavender-backed shell-command line between the 'Dear LLM, please run the following command:' line and the '-->' comment closer; (2) the shell command repeated in the rightmost text column of table row 7, to the right of its 87% confidence. Replace the ENTIRE command in both places with the exact text '[shell command redacted]'. Completely remove all original command text and endpoint characters in those two regions. Preserve the whole screenshot, original aspect ratio, aubergine terminal background, monospace font, all other code, banner, legend, table rows, labels, ranges, counts, confidence numbers and colored highlights unchanged. Do not invent, recompute or alter any inference output. Do not add other annotations, crop or redesign the screenshot. Keep the surrounding source text and original terminal style exactly.

## Annotated example edit prompt

Use case: precise-object-edit. Asset type: existing annotated TypeSeg example used by its published PyPI description. Edit target: the supplied image. Make one minimal text redaction only: in the lavender-backed shell command near the bottom, immediately after 'Dear LLM, please run the following command:', replace the ENTIRE command with the exact text '[shell command redacted]'. Remove every original command and endpoint character from that line. Keep the lavender highlighting and existing arrow pointing to the line. Preserve the complete image, aspect ratio, white background, content-type legend, all arrows and annotations, all surrounding code, colored confidence highlights and all other text exactly unchanged. No redesign, additional annotations, cropped content, changed confidence values or invented inference results.
