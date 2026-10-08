---
title: Research & Citation
description: The TypeSeg AISec 2026 paper, released artifacts, and citation.
---
**TypeSeg: Content-Type Segmentation for Mixed Text**  
Martin Dallinger, Yanick Fratantonio, and Luca Invernizzi  
AISec ’26, The Hague, Netherlands, November 15–19, 2026

The paper will be available after AISec ’26. [Source repository](https://github.com/S0urC10ud/semantic-text-segmentation).

The paper defines character-level textual segmentation and evaluates general text, CPU/GPU throughput, and security-domain adaptation on a human-labeled MalwareBazaar audit. These docs use the paper’s results and distinguish general checkpoints from security-adapted variants.

## Released artifacts

The repository contains the general U-Net and Mamba checkpoints, inference code, training and dataset-building code, annotation prompts, and evaluation scripts. The [Open Science artifact map](https://github.com/S0urC10ud/semantic-text-segmentation/blob/main/OPEN_SCIENCE.md) describes availability and access requirements.

Code and general model artifacts are under [Apache 2.0](https://github.com/S0urC10ud/semantic-text-segmentation/blob/main/LICENSE).

## Acknowledgments

This work benefited from feedback and discussions with Yanick Fratantonio and Luca Invernizzi at Google Security Research, and from Google Cloud compute resources. Thanks to Stefan Rass at JKU for guidance on parsing and support for the academic collaboration. The repository records the full acknowledgments and use of Gemini for training annotations.

## Citation

```bibtex
@inproceedings{dallinger2026typeseg,
  author = {Dallinger, Martin and Fratantonio, Yanick and Invernizzi, Luca},
  title = {{TypeSeg}: Content-Type Segmentation for Mixed Text},
  booktitle = {19th Workshop on Artificial Intelligence and Security},
  series = {AISec '26},
  year = {2026},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3847352.3848105},
  url = {https://doi.org/10.1145/3847352.3848105}
}
```
