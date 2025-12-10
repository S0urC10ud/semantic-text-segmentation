# Segmenter Evaluation Report

- Checkpoint: `../train/checkpoints/sweeps/y325i63s-20000.msgpack`
- Model dim: 256
- Channels: 32, 64, 64, 128, 128, 128, 128, 256
- Chunk: 1536
- Batch size: 128
- Max samples per task: 2000
- Sample seed: 13
- Evaluation data root: `/home/s0urc10ud/text-segmentation/evaluation/data_b`
- Generated at: 2025-12-09T21:28:23

### Task Highlights

##### mal_injection
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Avg coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 1091/2000 | 568/2000 | 0.32 | 66.0% |
| Correct payload | 614/2000 | 574/2000 | 0.27 | 41.6% |

##### markdown_mix
_Text hits column: lower is better._
| Wrapper | Non-text cov ≥50% | Non-text IoU ≥50% | Non-text avg coverage | Non-text avg IoU | Text hits | Correct cov ≥50% | Correct IoU ≥50% | Correct avg coverage | Correct avg IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 217/217 | 217/217 | 100.0% | 1.00 | 0/217 | 148/217 | 148/217 | 80.7% | 0.81 |
| bare code | 170/170 | 170/170 | 100.0% | 1.00 | 0/170 | 103/170 | 103/170 | 79.4% | 0.79 |

##### restructuredtext_mix
_Text hits column: lower is better._
| Foreign language | Blocks | Correct cov ≥50% | Correct avg coverage |
| --- | ---: | ---: | ---: |
| c_family | 7 | 6/7 | 95.8% |
| css | 13 | 1/13 | 4.8% |
| html | 17 | 17/17 | 98.8% |
| javascript_typescript | 7 | 4/7 | 38.0% |
| json | 2 | 2/2 | 73.5% |
| markdown | 1 | 1/1 | 100.0% |
| other | 5 | 0/5 | 0.0% |
| php | 3 | 3/3 | 97.5% |
| python | 76 | 70/76 | 86.0% |
| scala | 4 | 4/4 | 98.9% |
| shell | 91 | 53/91 | 33.2% |
| yaml | 6 | 6/6 | 95.5% |

##### pure_fragments
753/2000 samples stayed fully pure (no foreign chars). 1911/2000 stayed within ≤50% foreign coverage.
Expected foreign bytes for a 1536-byte fragment: 144.3/1536

##### sequence_pair
First segment coverage 93.7%
Second segment coverage 85.8%

##### sequence_triplet
First segment coverage 93.8%
Second segment coverage 92.6%
Third segment coverage 84.5%

##### needle_64_plus
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 124/142 | 61/142 | 0.48 | 86.3% |
| Correct payload | 108/142 | 97/142 | 0.67 | 74.5% |

##### needle_32_63
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 51/56 | 19/56 | 0.37 | 85.7% |
| Correct payload | 43/56 | 41/56 | 0.66 | 70.1% |

##### needle_16_31
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 36/56 | 15/56 | 0.28 | 56.7% |
| Correct payload | 29/56 | 25/56 | 0.42 | 47.9% |

##### needle_4_15
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 19/48 | 5/48 | 0.13 | 44.8% |
| Correct payload | 12/48 | 12/48 | 0.23 | 31.3% |


## Task Details

### mal_injection

Monitor-based hosts with synthetic malicious payload injections.

- Samples: 2000
- Characters evaluated: 6727462
- Overall accuracy: 0.8707
- High confusions: java->javascript_typescript 31851 (12.4%), python->javascript_typescript 28954 (13.7%), text->other 21399 (9.0%), encoding_base64->other 17655 (6.8%), powershell->other 15644 (6.4%)

| Language | Non-wrapper cov ≥50% | Non-wrapper coverage (avg) | Non-wrapper IoU ≥50% | Non-wrapper avg IoU | Correct cov ≥50% | Correct coverage (avg) | Correct IoU ≥50% | Correct avg IoU | Top misclassifications |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- |
| csharp | 179/188 | 94.5% | 103/188 | 0.59 | 175/188 | 92.3% (overall 93.2%) | 158/188 | 0.81 | other (3.5%), xml (0.9%), java (0.9%) |
| go | 110/244 | 47.0% | 34/244 | 0.24 | 1/244 | 1.6% (overall 71.1%) | 1/244 | 0.02 | other (6.1%), javascript_typescript (6.0%), shell (2.8%) |
| java | 177/212 | 82.6% | 104/212 | 0.51 | 105/212 | 32.7% (overall 77.0%) | 95/212 | 0.43 | javascript_typescript (12.4%), other (4.6%), csharp (1.1%) |
| javascript_typescript | 112/213 | 63.4% | 44/213 | 0.25 | 73/213 | 52.3% (overall 81.4%) | 70/213 | 0.30 | other (6.4%), dart (1.0%), rust (1.0%) |
| php | 104/238 | 42.1% | 51/238 | 0.26 | 77/238 | 29.0% (overall 61.1%) | 76/238 | 0.29 | other (6.0%), shell (3.2%), javascript_typescript (2.7%) |
| powershell | 144/229 | 80.4% | 118/229 | 0.45 | 129/229 | 66.9% (overall 79.0%) | 122/229 | 0.47 | other (6.4%), csharp (4.2%), go (1.5%) |
| python | 142/246 | 55.6% | 66/246 | 0.32 | 0/246 | 0.0% (overall 66.6%) | 0/246 | 0.00 | javascript_typescript (13.7%), other (5.7%), yaml (1.2%) |
| ruby | 48/223 | 23.9% | 19/223 | 0.11 | 0/223 | 0.0% (overall 58.3%) | 0/223 | 0.00 | other (7.0%), javascript_typescript (4.9%), shell (2.2%) |
| shell | 75/207 | 37.5% | 29/207 | 0.18 | 54/207 | 24.1% (overall 73.0%) | 52/207 | 0.24 | other (6.8%), c_family (3.1%), restructuredtext (2.7%) |

### markdown_mix

Monitor markdown documents with natural code/text interleavings.

- Samples: 69
- Characters evaluated: 193787
- Overall accuracy: 0.8540
- High confusions: markdown->other 12894 (10.0%), shell->markdown 2124 (28.8%), other->markdown 1707 (47.1%), html->markdown 1606 (17.8%), other->csharp 767 (21.2%)

_Text hits column: lower is better._
| Wrapper | Non-text cov ≥50% | Non-text IoU ≥50% | Non-text avg coverage | Non-text avg IoU | Text hits | Correct cov ≥50% | Correct IoU ≥50% | Correct avg coverage | Correct avg IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 217/217 | 217/217 | 100.0% | 1.00 | 0/217 | 148/217 | 148/217 | 80.7% | 0.81 |
| bare code | 170/170 | 170/170 | 100.0% | 1.00 | 0/170 | 103/170 | 103/170 | 79.4% | 0.79 |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| java | 4910 | 0.9971 |
| scala | 698 | 0.9971 |
| swift | 654 | 0.9908 |
| rust | 4084 | 0.9905 |
| powershell | 193 | 0.9896 |
| go | 318 | 0.9623 |
| python | 1940 | 0.9515 |
| php | 1727 | 0.9514 |
| yaml | 11881 | 0.8933 |
| json | 4195 | 0.8856 |
| markdown | 128887 | 0.8823 |
| csharp | 1385 | 0.8671 |
| javascript_typescript | 8450 | 0.8363 |
| html | 9045 | 0.7978 |
| c_family | 2979 | 0.7043 |
| dockerfile | 571 | 0.6935 |
| shell | 7371 | 0.6367 |
| css | 849 | 0.3651 |
| other | 3621 | 0.0561 |
| ruby | 29 | 0.0000 |

### pure_fragments

Full monitor documents with original multi-label segmentations.

- Samples: 2000
- Characters evaluated: 5978722
- Overall accuracy: 0.9316
- High confusions: text->other 14197 (7.0%), json->other 11462 (8.3%), svg->other 11004 (7.6%), encoding_base32->other 10392 (3.5%), encoding_base64->other 10321 (4.3%)

Expected foreign bytes for a 1536-byte fragment: 144.3/1536

#### Purity Analysis

| Language | Support | Accuracy % | File purity | Top Misclassifications |
| --- | ---: | ---: | --- | --- |
| csv | 308416 | 97.4% | 17/58 (29.3%) | text (0.5%), markdown (0.0%) |
| rust | 236168 | 97.1% | 13/59 (22.0%) | html (0.0%), javascript_typescript (0.0%), sql (0.0%) |
| encoding_base58 | 207795 | 96.9% | 28/58 (48.3%) | — |
| encoding_base85 | 276609 | 96.7% | 16/57 (28.1%) | encoding_base64 (0.0%), csv (0.0%) |
| encoding_hex | 279518 | 96.5% | 18/57 (31.6%) | — |
| encoding_base32 | 296984 | 96.5% | 24/67 (35.8%) | — |
| gettext_catalog | 228146 | 96.1% | 7/46 (15.2%) | markdown (0.0%) |
| csharp | 109307 | 96.0% | 28/55 (50.9%) | powershell (0.0%) |
| go | 213614 | 95.6% | 13/54 (24.1%) | markdown (0.1%) |
| tex | 167029 | 95.4% | 18/50 (36.0%) | text (0.6%), python (0.2%), markdown (0.2%) |
| swift | 138586 | 95.2% | 34/65 (52.3%) | — |
| java | 203757 | 94.6% | 21/66 (31.8%) | c_family (0.9%) |
| html | 240883 | 94.2% | 18/61 (29.5%) | markdown (1.2%), gettext_catalog (0.7%), php (0.7%) |
| sql | 155309 | 94.1% | 30/55 (54.5%) | html (3.2%), powershell (0.2%), php (0.1%) |
| css | 229456 | 93.5% | 19/57 (33.3%) | encoding_base64 (1.3%), c_family (1.2%), html (0.7%) |
| dart | 141579 | 93.1% | 24/58 (41.4%) | javascript_typescript (0.6%), rust (0.5%) |
| text | 203463 | 93.0% | 6/53 (11.3%) | php (0.0%) |
| powershell | 151394 | 92.8% | 22/54 (40.7%) | markdown (0.3%), csharp (0.3%), json (0.2%) |
| yaml | 80856 | 92.8% | 26/47 (55.3%) | markdown (1.0%), encoding_base64 (0.8%), html (0.2%) |
| encoding_base64 | 241115 | 92.7% | 16/53 (30.2%) | css (2.7%), powershell (0.2%), svg (0.1%) |
| scala | 124357 | 92.5% | 24/56 (42.9%) | kotlin (0.5%), html (0.0%) |
| c_family | 145957 | 92.0% | 21/55 (38.2%) | java (0.9%), dart (0.4%), go (0.3%) |
| python | 170691 | 92.0% | 19/55 (34.5%) | json (2.2%), html (0.8%), markdown (0.5%) |
| javascript_typescript | 146946 | 91.9% | 30/59 (50.8%) | html (0.9%), c_family (0.6%), json (0.2%) |
| kotlin | 81428 | 91.8% | 30/54 (55.6%) | — |
| svg | 145314 | 91.7% | 33/60 (55.0%) | javascript_typescript (0.6%), html (0.1%) |
| visual_basic | 183882 | 91.5% | 14/59 (23.7%) | sql (3.2%), xml (0.6%), csv (0.3%) |
| ruby | 91258 | 90.9% | 30/53 (56.6%) | python (1.4%), rust (0.9%), shell (0.1%) |
| xml | 132754 | 90.6% | 38/59 (64.4%) | html (2.9%), encoding_base64 (1.6%), php (0.3%) |
| restructuredtext | 128051 | 89.9% | 32/64 (50.0%) | python (2.0%), markdown (2.0%), shell (0.4%) |
| json | 138795 | 86.8% | 19/60 (31.7%) | html (2.4%), csv (1.3%), yaml (0.6%) |
| shell | 99990 | 86.2% | 40/54 (74.1%) | c_family (2.5%), restructuredtext (2.0%), yaml (0.9%) |
| markdown | 137549 | 86.0% | 20/61 (32.8%) | restructuredtext (2.7%), yaml (2.0%), csv (0.9%) |
| dockerfile | 24276 | 85.0% | 40/48 (83.3%) | c_family (7.9%), shell (4.5%) |
| php | 86782 | 82.5% | 29/54 (53.7%) | csv (11.5%), html (1.1%), javascript_typescript (0.2%) |
| other | 30708 | 3.7% | — | — |

### restructuredtext_mix

Monitor reStructuredText documents with natural code/text interleavings.

- Samples: 44
- Characters evaluated: 164503
- Overall accuracy: 0.8516
- High confusions: restructuredtext->other 4610 (3.9%), shell->restructuredtext 4090 (28.0%), shell->c_family 2451 (16.8%), restructuredtext->python 2438 (2.1%), shell->yaml 1786 (12.2%)

_Text hits column: lower is better._
| Wrapper | Non-text cov ≥50% | Non-text IoU ≥50% | Non-text avg coverage | Non-text avg IoU | Text hits | Correct cov ≥50% | Correct IoU ≥50% | Correct avg coverage | Correct avg IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 14/14 | 14/14 | 100.0% | 1.00 | 0/14 | 10/14 | 10/14 | 65.1% | 0.65 |
| bare code | 218/218 | 218/218 | 100.0% | 1.00 | 0/218 | 157/218 | 157/218 | 70.9% | 0.71 |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| markdown | 17 | 1.0000 |
| scala | 931 | 0.9989 |
| html | 3418 | 0.9968 |
| php | 2761 | 0.9721 |
| c_family | 2317 | 0.9599 |
| yaml | 1107 | 0.9530 |
| restructuredtext | 118022 | 0.9151 |
| python | 18236 | 0.8711 |
| json | 88 | 0.7159 |
| javascript_typescript | 2349 | 0.4134 |
| shell | 14615 | 0.3312 |
| css | 173 | 0.0462 |
| other | 469 | 0.0000 |

### sequence_pair

Two-language back-to-back sequences A->B.

- Samples: 2000
- Characters evaluated: 12221980
- Overall accuracy: 0.8973
- High confusions: dockerfile->shell 57387 (42.9%), php->html 50541 (20.9%), powershell->encoding_base64 30506 (11.0%), text->markdown 26402 (6.1%), restructuredtext->python 25526 (8.0%)

| Segment | Coverage |
| --- | ---: |
| First | 93.7% |
| Second | 85.8% |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| gettext_catalog | 754848 | 0.9714 |
| encoding_hex | 635268 | 0.9678 |
| encoding_base32 | 537760 | 0.9667 |
| csv | 620411 | 0.9634 |
| visual_basic | 402972 | 0.9587 |
| encoding_base85 | 522333 | 0.9562 |
| rust | 383831 | 0.9506 |
| encoding_base58 | 496390 | 0.9493 |
| encoding_base64 | 382076 | 0.9456 |
| css | 446957 | 0.9455 |
| go | 462701 | 0.9377 |
| tex | 441715 | 0.9373 |
| dart | 256576 | 0.9339 |
| java | 320231 | 0.9331 |
| scala | 289599 | 0.9317 |
| csharp | 212654 | 0.9276 |
| ruby | 184050 | 0.9222 |
| kotlin | 194895 | 0.9205 |
| python | 328029 | 0.9130 |
| swift | 251673 | 0.9116 |
| sql | 302072 | 0.8983 |
| xml | 301253 | 0.8962 |
| json | 205666 | 0.8862 |
| c_family | 368428 | 0.8861 |
| svg | 208976 | 0.8641 |
| text | 431854 | 0.8487 |
| shell | 110200 | 0.8432 |
| powershell | 277052 | 0.8151 |
| html | 452234 | 0.8120 |
| javascript_typescript | 266393 | 0.8050 |
| yaml | 191142 | 0.8026 |
| restructuredtext | 318576 | 0.7069 |
| php | 242186 | 0.6731 |
| markdown | 287183 | 0.6513 |
| dockerfile | 133796 | 0.2850 |
| other | 0 | nan |

### sequence_triplet

Three-language back-to-back sequences A->B->C.

- Samples: 2000
- Characters evaluated: 18434131
- Overall accuracy: 0.9032
- High confusions: dockerfile->shell 106283 (45.5%), powershell->encoding_base64 61128 (12.9%), php->html 59927 (20.8%), html->javascript_typescript 53745 (7.5%), text->markdown 42326 (5.8%)

| Segment | Coverage |
| --- | ---: |
| First | 93.8% |
| Second | 92.6% |
| Third | 84.5% |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base85 | 801721 | 0.9783 |
| gettext_catalog | 1076943 | 0.9756 |
| encoding_base32 | 746240 | 0.9730 |
| encoding_base58 | 679308 | 0.9698 |
| encoding_hex | 833930 | 0.9689 |
| csv | 904670 | 0.9680 |
| rust | 577531 | 0.9659 |
| encoding_base64 | 740908 | 0.9639 |
| visual_basic | 593260 | 0.9520 |
| tex | 553170 | 0.9496 |
| go | 635827 | 0.9488 |
| css | 710709 | 0.9442 |
| python | 510050 | 0.9438 |
| swift | 356029 | 0.9434 |
| csharp | 327866 | 0.9431 |
| java | 514322 | 0.9322 |
| sql | 448125 | 0.9319 |
| xml | 373089 | 0.9298 |
| dart | 442037 | 0.9280 |
| scala | 413082 | 0.9278 |
| ruby | 313731 | 0.9273 |
| svg | 508314 | 0.9130 |
| c_family | 574970 | 0.9119 |
| kotlin | 303075 | 0.9061 |
| json | 247531 | 0.8724 |
| shell | 193965 | 0.8601 |
| javascript_typescript | 418293 | 0.8412 |
| text | 723839 | 0.8221 |
| powershell | 473717 | 0.8087 |
| yaml | 337797 | 0.7861 |
| html | 712768 | 0.7620 |
| restructuredtext | 430284 | 0.7477 |
| markdown | 435594 | 0.6978 |
| php | 287985 | 0.6308 |
| dockerfile | 233451 | 0.2715 |
| other | 0 | nan |

### needle_64_plus

Natural monitor files with a short foreign-language needle sized 64-∞ printable chars (whitespace ignored).

- Samples: 142
- Characters evaluated: 508694
- Overall accuracy: 0.7950
- High confusions: other->shell 31303 (73.2%), markdown->other 9743 (8.3%), encoding_base64->css 6527 (25.7%), html->other 5479 (4.7%), dockerfile->shell 3605 (35.9%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| xml | 808 | 1.0000 |
| java | 2144 | 0.9925 |
| ruby | 438 | 0.9909 |
| swift | 654 | 0.9908 |
| rust | 4084 | 0.9905 |
| csharp | 1019 | 0.9725 |
| restructuredtext | 33769 | 0.9669 |
| powershell | 15780 | 0.9531 |
| svg | 7421 | 0.9435 |
| yaml | 34549 | 0.9130 |
| html | 117089 | 0.9038 |
| shell | 39586 | 0.8893 |
| javascript_typescript | 18949 | 0.8785 |
| markdown | 117757 | 0.8574 |
| python | 3513 | 0.8480 |
| php | 13851 | 0.7906 |
| json | 6904 | 0.7701 |
| encoding_base64 | 25365 | 0.6874 |
| dockerfile | 10037 | 0.6221 |
| css | 10082 | 0.5793 |
| other | 42780 | 0.0396 |
| sql | 421 | 0.0024 |
| go | 1694 | 0.0000 |
| c_family | 0 | nan |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 124/142 | 61/142 | 0.48 | 86.3% | html (23.4%), other (21.1%), markdown (12.2%) |
| Correct label | 108/142 | 97/142 | 0.67 | 74.5% | — |

### needle_32_63

Natural monitor files with a short foreign-language needle sized 32-63 printable chars (whitespace ignored).

- Samples: 56
- Characters evaluated: 151810
- Overall accuracy: 0.8077
- High confusions: other->php 9741 (67.5%), markdown->other 3095 (8.7%), html->other 2459 (5.5%), other->html 2198 (15.2%), restructuredtext->other 1620 (10.2%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| dockerfile | 3076 | 1.0000 |
| svg | 4378 | 0.9589 |
| php | 2544 | 0.9481 |
| html | 44854 | 0.9405 |
| powershell | 10845 | 0.9306 |
| markdown | 35638 | 0.9032 |
| restructuredtext | 15864 | 0.8957 |
| css | 560 | 0.8696 |
| xml | 94 | 0.8298 |
| yaml | 4119 | 0.8235 |
| shell | 10442 | 0.7368 |
| javascript_typescript | 1157 | 0.6975 |
| json | 2172 | 0.6220 |
| python | 168 | 0.4464 |
| other | 14423 | 0.0261 |
| c_family | 1195 | 0.0000 |
| rust | 145 | 0.0000 |
| ruby | 80 | 0.0000 |
| encoding_base64 | 56 | 0.0000 |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 51/56 | 19/56 | 0.37 | 85.7% | shell (30.4%), other (20.5%), restructuredtext (10.0%) |
| Correct label | 43/56 | 41/56 | 0.66 | 70.1% | — |

### needle_16_31

Natural monitor files with a short foreign-language needle sized 16-31 printable chars (whitespace ignored).

- Samples: 56
- Characters evaluated: 168351
- Overall accuracy: 0.9067
- High confusions: html->other 1891 (3.3%), json->other 1842 (19.5%), markdown->other 1669 (3.8%), restructuredtext->other 1667 (10.3%), other->shell 1525 (44.6%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| dockerfile | 1614 | 1.0000 |
| powershell | 1146 | 1.0000 |
| yaml | 10966 | 0.9892 |
| markdown | 44059 | 0.9602 |
| html | 57497 | 0.9424 |
| php | 2949 | 0.9373 |
| shell | 7095 | 0.9294 |
| javascript_typescript | 11506 | 0.9102 |
| csharp | 33 | 0.9091 |
| python | 1104 | 0.8569 |
| restructuredtext | 16209 | 0.8089 |
| json | 9470 | 0.8024 |
| css | 383 | 0.4726 |
| other | 3419 | 0.2150 |
| encoding_hex | 733 | 0.1514 |
| xml | 72 | 0.0000 |
| sql | 42 | 0.0000 |
| dart | 27 | 0.0000 |
| ruby | 27 | 0.0000 |
| c_family | 0 | nan |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 36/56 | 15/56 | 0.28 | 56.7% | yaml (46.6%), html (16.4%), other (9.3%) |
| Correct label | 29/56 | 25/56 | 0.42 | 47.9% | — |

### needle_4_15

Natural monitor files with a short foreign-language needle sized 4-15 printable chars (whitespace ignored).

- Samples: 48
- Characters evaluated: 118277
- Overall accuracy: 0.8130
- High confusions: html->other 7051 (10.5%), css->other 2122 (37.2%), php->html 1876 (20.2%), markdown->html 1324 (8.9%), other->html 1090 (38.8%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| dockerfile | 771 | 1.0000 |
| restructuredtext | 386 | 1.0000 |
| rust | 8168 | 0.9905 |
| yaml | 3286 | 0.9382 |
| html | 66977 | 0.8770 |
| shell | 1652 | 0.8717 |
| markdown | 14910 | 0.8389 |
| php | 9281 | 0.7421 |
| javascript_typescript | 1924 | 0.6071 |
| css | 5697 | 0.4887 |
| other | 2809 | 0.1096 |
| ruby | 1002 | 0.0000 |
| go | 842 | 0.0000 |
| sql | 358 | 0.0000 |
| scala | 142 | 0.0000 |
| c_family | 36 | 0.0000 |
| xml | 36 | 0.0000 |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 19/48 | 5/48 | 0.13 | 44.8% | html (47.4%), markdown (22.9%), yaml (11.8%) |
| Correct label | 12/48 | 12/48 | 0.23 | 31.3% | — |

## Throughput Benchmarks

| Task | Device | Samples | Total Bytes | Throughput | Latency (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| throughput_1024 | cpu | 4 | 4096 | 4.57 KB/s | 0.88 |
| throughput_10240 | cpu | 4 | 40960 | 32.04 KB/s | 1.25 |
| throughput_102400 | cpu | 4 | 409600 | 65.66 KB/s | 6.09 |
| throughput_1048576 | cpu | 4 | 4194304 | 73.45 KB/s | 55.77 |
| throughput_1024 | cuda | 4 | 4096 | 18.35 KB/s | 0.22 |
| throughput_10240 | cuda | 4 | 40960 | 75.21 KB/s | 0.53 |
| throughput_102400 | cuda | 4 | 409600 | 84.89 KB/s | 4.71 |
| throughput_1048576 | cuda | 4 | 4194304 | 88.31 KB/s | 46.38 |

Latency is the total wall-clock time to run the benchmark loop over all samples per benchmark type, excluding the one warmup inference call that triggers JAX’s JIT compilation beforehand.

Report generated at 2025-12-09 21:48:11