---
title: Performance & Evaluation
description: CPU throughput, model quality, and security results from the TypeSeg paper, with measurement context.
---
## CPU-first throughput

The paper reports warmed inference on an **AMD Ryzen AI 7 350** notebook with an **NVIDIA RTX 5070 mobile GPU (85 W)**. Rates count distinct input characters, including the cost of processing overlapping windows. Model loading and the first compilation pass are excluded.

| Model | CPU characters/s | GPU characters/s |
| --- | ---: | ---: |
| U-Net | **291,560** | **4,825,475** |
| Mamba | 2,248 | 54,892 |

**Choose U-Net for a CPU throughput pilot.** These are the paper’s research-runtime measurements (Table 3), not fresh benchmarks of the PyPI ONNX package or JavaScript demo. Benchmark your runtime, thread settings, file sizes, and concurrency before estimating production capacity.

TypeSeg reads the textual content it segments, so work grows with input length. Magika’s whole-file classification samples a small subset and serves a different purpose. Use TypeSeg on the text branch of the pipeline; it does not need to process every binary upload.

## General text quality

Table 1 evaluates **875 human-labeled files**, 25 per learned class, using the same prefix of at most 10,000 characters for each system. Dense F1 is computed per file and averaged; it is not whole-file classification accuracy.

| System | Dense F1 | Boundary-neighborhood F1 |
| --- | ---: | ---: |
| Magika, whole-file | 72.0% | 32.0% |
| Magika, sliding windows | 66.4% | 29.6% |
| U-Net | **94.9%** | 45.3% |
| Mamba | 95.1% | 57.2% |
| Gemini 3 Flash | 96.6% | 70.8% |

The Magika rows measure how file or window predictions perform **on this segmentation task**. They do not imply that Magika has those accuracies for its own whole-file task.

## Security-domain evaluation

The MalwareBazaar audit covers 165 files across 11 textual carrier families. Adaptation uses a separate 1,400-file training split. In Table 4, general U-Net reaches **69.85% dense F1** on the audit; U-Net-Sec reaches **88.02%**.

| Security-adapted model | Guest-segment recovery | Encoding recovery | Guest-class precision |
| --- | ---: | ---: | ---: |
| U-Net-Sec | **81.8%** | **91.9%** | 52.1% |
| Mamba-Sec | 80.0% | 89.6% | **76.6%** |

Recovery requires at least 50% of a reference segment to receive the correct type. It does not require exact boundaries. Guest-class precision counts distinct embedded classes per file; it is a separate measure from segment recovery.

**The public package and demo ship general checkpoints.** The security-adapted results above describe the paper’s separate models and must not be attributed to `pip install typeseg`.

## From regions to analyzer inputs

Table 6 evaluates 1,735 validator-eligible guest references. Expansion and validation give U-Net-Sec **86.5% recovery** and **37.3% route match rate**, compared with **15.3% recovery from raw crops**. Mamba-Sec trades lower recovery (60.6%) for a higher route match rate (56.6%).

Recovery counts reference guests covered by a same-class, validator-accepted route. Match rate counts produced routes that contain a same-class reference. One route may cover multiple references. Validator acceptance does not establish malicious behavior or successful downstream detection.

See the [paper](../../resources/research/) for the complete protocol and [limitations](../../resources/limitations/) for interpretation.
