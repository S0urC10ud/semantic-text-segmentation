# TypeSeg documentation site

Built with Astro + Starlight, the framework used by Magika’s documentation.
Deploys as static HTML on the existing GitHub Pages custom domain.

```sh
npm ci
npm run dev
npm run check
npm test
npm run build
```

`prepare-demo.mjs` copies the existing browser runtime into `public/demo/`
without training/export artifacts. The original viewer remains the source of truth.
The documentation overview is the entry point; `/demo/` is the standalone demo.
The original `/images/` URLs remain available for the published PyPI description.
Edited screenshots live in `src/assets/`; `TerminalExample.astro` shares an optimized
terminal preview between the overview and CLI guide. Clicking opens the full image.
The paper is withheld until after the conference. Do not put PDFs in the public
directory; builds reject public PDF files. Publishing the paper requires an explicit
change to that check. Quantitative claims in the docs cite the manuscript's table numbers.

For a repository-subpath deployment, build with
`TYPESEG_BASE=/semantic-text-segmentation`. The custom-domain deployment uses `/`.
The site stores no user input and introduces no backend or analytics.
