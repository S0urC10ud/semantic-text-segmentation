---
title: Performance & Evaluation
description: CPU throughput, model quality, and security results from the TypeSeg paper, with measurement context.
---
## Throughput

The paper reports warmed inference on an **AMD Ryzen AI 7 350** notebook with an **NVIDIA RTX 5070 mobile GPU (85 W)**. Rates count distinct input characters, including the cost of processing overlapping windows. Model loading and the first compilation pass are excluded.

| Model | CPU characters/s | GPU characters/s |
| --- | ---: | ---: |
| U-Net | **291,560** | **4,825,475** |
| Mamba | 2,248 | 54,892 |

These are the paper’s research-runtime measurements (Table 3). The PyPI ONNX package and JavaScript demo use different runtimes. Measure your runtime, thread settings, file sizes, and concurrency when estimating throughput.

TypeSeg reads the text it segments, so work grows with input length. Select textual inputs before running it. Magika’s whole-file classification uses a small sample of a file; the timings cover different tasks.

## General text quality

Table 1 evaluates **875 human-labeled files across 35 content types**, 25 per learned class, using the same prefix of at most 10,000 characters for each system. Dense F1 scores the per-character content-type labels. Scores are computed per file and averaged.

| System | Dense F1 | Boundary-neighborhood F1 |
| --- | ---: | ---: |
| Magika, whole-file | 72.0% | 32.0% |
| Magika, sliding windows | 66.4% | 29.6% |
| U-Net | **94.9%** | 45.3% |
| Mamba | 95.1% | 57.2% |
| Gemini 3 Flash | 96.6% | 70.8% |

The Magika rows score file or window predictions **on this segmentation task**. Magika’s whole-file classification quality requires its own task-specific evaluation.

## Security-domain evaluation

The MalwareBazaar audit covers 165 files across 11 textual carrier families. Adaptation uses a separate 1,400-file training split. In Table 4, general U-Net reaches **69.85% dense F1** on the audit; U-Net-Sec reaches **88.02%**.

| Security-adapted model | Guest-segment recovery | Encoding recovery | Guest-class precision |
| --- | ---: | ---: | ---: |
| U-Net-Sec | **81.8%** | **91.9%** | 52.1% |
| Mamba-Sec | 80.0% | 89.6% | **76.6%** |

Recovery requires at least 50% of a reference segment to receive the correct type; boundaries can differ. Guest-class precision counts distinct embedded classes per file.

**The public package and demo ship general checkpoints.** U-Net-Sec and Mamba-Sec are the paper’s separate security-adapted models. Their results apply to those models.

## From regions to analyzer inputs

Table 6 evaluates 1,735 validator-eligible guest references. Expansion and validation give U-Net-Sec **86.5% recovery** and **37.3% route match rate**, compared with **15.3% recovery from raw crops**. Mamba-Sec trades lower recovery (60.6%) for a higher route match rate (56.6%).

Recovery counts reference guests covered by a same-class, validator-accepted route. Match rate counts produced routes that contain a same-class reference. One route may cover multiple references. Validator acceptance does not establish malicious behavior or successful downstream detection.

See [limitations](../../resources/limitations/) for interpretation and [research and citation](../../resources/research/) for publication details. The paper will be available after the conference.
