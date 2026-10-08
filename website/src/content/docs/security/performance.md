---
title: Performance & Evaluation
description: CPU throughput, model quality, and security results from the TypeSeg paper, with measurement context.
---
## Throughput

The paper benchmarks the JAX/Flax research implementation on **100,000-character inputs**, using an **AMD Ryzen AI 7 350** notebook and an **NVIDIA RTX 5070 laptop GPU (85 W)**. The model is already loaded and its first compilation is complete when timing starts. Rates count each input character once, including the work of processing overlapping windows.

| Model | CPU characters/s | GPU characters/s |
| --- | ---: | ---: |
| U-Net | **291,560** | **4,825,475** |
| Mamba | 2,248 | 54,892 |

**CPU core and thread count: not reported.** The paper does not specify how many CPU cores or threads were used for this run. Treat the CPU rate as a result for this laptop and research implementation; a per-core rate would need a benchmark with an explicit core and thread limit.

These are the paper’s research implementation rates (Table 3). The installed Python package and browser demo use different implementations. The [research benchmark timer](https://github.com/S0urC10ud/semantic-text-segmentation/blob/main/evaluation/evaluation.py#L4444) measures model labeling; file decoding, character-label post-processing, and building the final segments are outside that timer. Measure a complete package call on your own files when estimating application throughput.

TypeSeg reads the text it segments, so work grows with input length. Select textual inputs before running it. A tool such as Magika can identify a file’s overall type from a small sample.

## General text quality

The general benchmark uses **875 human-labeled files across 35 content types**, 25 per type. Every system sees the same first 10,000 characters of each file, or the full file if shorter.

**F1** combines precision (how often a predicted type is correct) and recall (how much of that type the system finds). Higher is better, with 100% representing perfect agreement with the human labels. Two scores are reported:

- **Overall label F1**, called *dense F1* in the paper, scores the non-whitespace characters throughout each file. Each type is weighted by its character count within the file; the final score averages the file scores equally.
- **F1 near type changes**, called *boundary-neighborhood F1*, applies the same score only within **four characters on each side of a human-labeled type change**. For a Python-to-PowerShell transition, it checks the nearby Python and PowerShell labels. This focuses on the difficult starts and ends of regions, so the score can be much lower than overall F1.

| System | Overall label F1 | F1 near type changes |
| --- | ---: | ---: |
| Magika, whole-file | 72.0% | 32.0% |
| Magika, sliding windows | 66.4% | 29.6% |
| U-Net | **94.9%** | 45.3% |
| Mamba | 95.1% | 57.2% |
| Gemini 3 Flash | 96.6% | 70.8% |

Magika normally returns one type for a whole file. For the first row, that type is assigned to every character before scoring. The sliding-window version labels overlapping 1,536-character sections and combines their predictions. Both rows therefore measure character labels inside mixed text. A Python file containing PowerShell can have the correct overall Python label while its PowerShell characters receive the wrong label.

## Security-domain evaluation

The security benchmark uses **165 text files from MalwareBazaar**, a public malware repository. There are 15 files from each of 11 file formats, such as Python, PowerShell, and HTML. The containing format is called the *carrier*. A *reference segment* is a region labeled by a human.

The [security evaluation code](https://github.com/S0urC10ud/semantic-text-segmentation/blob/main/evaluation/malwarebazaar_finetune_eval.py#L29) groups labels into **15 categories**, including XML/SVG and JSON/YAML. These scores use different files and label groups from the general benchmark. General U-Net reaches **69.85% dense F1** here; U-Net-Sec, trained on a separate 1,400-file security dataset, reaches **88.02%** (Table 4).

**The public package and demo include the general models.** U-Net-Sec and Mamba-Sec are the paper’s separate models with additional security training.

### Finding embedded scripts and encodings

A *guest* in this experiment is a script or encoding unexpected for the containing format. PowerShell inside Python counts; ordinary JavaScript inside HTML is expected and excluded. The tracked guest types are JavaScript, PowerShell, VBScript, shell, Python, SQL, PHP, Base64, and Hex.

**Segment recovery** counts a human-labeled guest as found when the correct type covers at least half the region. **Encoding recovery** applies that rule to encoded regions. **Guest-class precision** measures how often predicted guest types match the human labels, counting each type once per file. For example, three shell regions in one file count as one shell prediction for this precision measure.

| Model with security training | Guest-segment recovery | Encoding recovery | Guest-class precision |
| --- | ---: | ---: | ---: |
| U-Net-Sec | **81.8%** | **91.9%** | 52.1% |
| Mamba-Sec | 80.0% | 89.6% | **76.6%** |

## From regions to analyzer inputs

A separate experiment tests whether predicted regions can become complete inputs for an analyzer. A *route* is a candidate input prepared from a prediction. A language parser or encoding validator checks whether the candidate has valid syntax or encoding.

Of 1,899 human-labeled guests, **1,735 pass their validator at the annotated boundaries**. Those 1,735 are eligible for this experiment; unsupported or invalid fragments receive no credit.

- **Validated-input recovery** is the share of eligible reference segments fully covered by a valid candidate of the correct type. This requires a complete input, beyond the half-segment coverage used above.
- **Route match rate** is the share of emitted candidates that fully cover an eligible reference of the correct type. One candidate can contain several references, so these two measures count different things.

The experiment compares taking the predicted substring directly (*raw crop*) with expanding it to surrounding syntax and validating the result:

| Model with security training | Recovery from raw crops | Recovery after expansion | Route match rate after expansion |
| --- | ---: | ---: | ---: |
| U-Net-Sec | 15.3% | **86.5%** | 37.3% |
| Mamba-Sec | 11.4% | 60.6% | **56.6%** |

These are Table 6 results on the 1,735 eligible references. U-Net-Sec recovers more references; Mamba-Sec produces a larger share of matching candidates. A valid candidate may still be harmless. Effects on threat detection require a separate evaluation.

See [limitations](../../resources/limitations/) for interpretation and [research and citation](../../resources/research/) for publication details. The paper will be available after the conference.
