# Segmenter Evaluation Report

- Checkpoint: `../train/checkpoints/sweeps/y325i63s-20000.msgpack`
- Model dim: 256
- Channels: 32, 64, 64, 128, 128, 128, 128, 256
- Chunk: 1536
- Batch size: 128
- Max samples per task: 2000
- Sample seed: 13
- Evaluation data root: `/home/s0urc10ud/text-segmentation/evaluation/data`
- Generated at: 2025-12-04T12:07:15

### Task Highlights

##### mal_injection
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Avg coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 1062/2000 | 547/2000 | 0.31 | 67.3% |
| Correct payload | 642/2000 | 558/2000 | 0.27 | 47.3% |

##### markdown_mix
_Text hits column: lower is better._
| Wrapper | Non-text cov ≥50% | Non-text IoU ≥50% | Non-text avg coverage | Non-text avg IoU | Text hits | Correct cov ≥50% | Correct IoU ≥50% | Correct avg coverage | Correct avg IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 175/175 | 175/175 | 100.0% | 1.00 | 0/175 | 123/175 | 123/175 | 85.1% | 0.85 |
| bare code | 199/199 | 199/199 | 100.0% | 1.00 | 0/199 | 132/199 | 132/199 | 83.8% | 0.84 |

Text coverage (IoU ≥50%): 0.8%

| Metric | Value |
| --- | --- |
| Host fenced IoU ≥50% | 93/129 hits, mean IoU 0.86, coverage 86.2% |
| Host bare IoU ≥50% | 103/145 hits, mean IoU 0.84, coverage 84.3% |
| Other fenced IoU ≥50% | 30/46 hits, mean IoU 0.73, coverage 73.1% |
| Other bare IoU ≥50% | 29/54 hits, mean IoU 0.80, coverage 80.4% |

##### pure_fragments
749/2000 samples stayed fully pure (no foreign chars). 1910/2000 stayed within ≤50% foreign coverage.
Expected foreign bytes for a 1536-byte fragment: 141.8/1536

##### sequence_pair
First segment coverage 93.6%
Second segment coverage 86.3%

##### sequence_triplet
First segment coverage 93.8%
Second segment coverage 91.3%
Third segment coverage 85.3%

##### needle_64_plus
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 111/125 | 57/125 | 0.50 | 90.7% |
| Correct payload | 100/125 | 95/125 | 0.73 | 82.6% |

##### needle_32_63
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 49/51 | 20/51 | 0.42 | 90.2% |
| Correct payload | 41/51 | 41/51 | 0.72 | 76.7% |

##### needle_16_31
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 35/51 | 15/51 | 0.29 | 50.0% |
| Correct payload | 28/51 | 24/51 | 0.42 | 38.8% |

##### needle_4_15
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 14/43 | 5/43 | 0.13 | 38.2% |
| Correct payload | 12/43 | 12/43 | 0.25 | 34.0% |


## Task Details

### mal_injection

Host fragments with malicious payload injections.

- Samples: 2000
- Characters evaluated: 8014522
- Overall accuracy: 0.8599
- High confusions: dockerfile->shell 38794 (47.7%), php->html 34003 (16.5%), java->javascript_typescript 32747 (10.3%), python->javascript_typescript 28351 (10.2%), text->php 22286 (6.9%)

| Language | Non-wrapper cov ≥50% | Non-wrapper coverage (avg) | Non-wrapper IoU ≥50% | Non-wrapper avg IoU | Correct cov ≥50% | Correct coverage (avg) | Correct IoU ≥50% | Correct avg IoU | Top misclassifications |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- |
| csharp | 209/220 | 94.2% | 130/220 | 0.62 | 202/220 | 90.5% (overall 91.8%) | 182/220 | 0.78 | php (5.2%), xml (0.9%), go (0.8%) |
| go | 85/228 | 40.6% | 30/228 | 0.23 | 3/228 | 2.6% (overall 76.6%) | 3/228 | 0.03 | php (5.7%), javascript_typescript (3.7%), shell (1.9%) |
| java | 177/229 | 76.4% | 94/229 | 0.47 | 107/229 | 38.4% (overall 75.0%) | 93/229 | 0.42 | javascript_typescript (10.3%), php (5.7%), csharp (2.8%) |
| javascript_typescript | 105/227 | 60.2% | 49/227 | 0.24 | 71/227 | 52.3% (overall 77.7%) | 68/227 | 0.27 | php (6.3%), html (2.8%), c_family (2.3%) |
| php | 84/219 | 37.8% | 33/219 | 0.21 | 68/219 | 28.7% (overall 58.5%) | 32/219 | 0.19 | html (16.5%), shell (3.0%), javascript_typescript (1.8%) |
| powershell | 149/221 | 84.3% | 109/221 | 0.46 | 135/221 | 72.7% (overall 79.0%) | 127/221 | 0.52 | encoding_base64 (5.8%), php (5.2%), csharp (3.5%) |
| python | 123/222 | 55.1% | 65/222 | 0.33 | 0/222 | 0.1% (overall 74.5%) | 0/222 | 0.00 | javascript_typescript (10.2%), php (5.0%), json (1.6%) |
| ruby | 51/218 | 26.1% | 9/218 | 0.09 | 0/218 | 0.0% (overall 71.8%) | 0/218 | 0.00 | php (6.3%), javascript_typescript (4.1%), python (2.1%) |
| shell | 79/216 | 39.7% | 28/216 | 0.17 | 56/216 | 28.2% (overall 67.9%) | 53/216 | 0.24 | php (6.4%), java (3.5%), xml (2.3%) |

### markdown_mix

Monitor markdown documents with natural code/text interleavings.

- Samples: 67
- Characters evaluated: 244176
- Overall accuracy: 0.2580
- High confusions: text->__unknown__ 154004 (89.9%), text->php 9140 (5.3%), text->restructuredtext 3500 (2.0%), html->__unknown__ 1988 (10.5%), shell->__unknown__ 1336 (24.3%)

_Text hits column: lower is better._
| Wrapper | Non-text cov ≥50% | Non-text IoU ≥50% | Non-text avg coverage | Non-text avg IoU | Text hits | Correct cov ≥50% | Correct IoU ≥50% | Correct avg coverage | Correct avg IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 175/175 | 175/175 | 100.0% | 1.00 | 0/175 | 123/175 | 123/175 | 85.1% | 0.85 |
| bare code | 199/199 | 199/199 | 100.0% | 1.00 | 0/199 | 132/199 | 132/199 | 83.8% | 0.84 |

Text coverage (IoU ≥50%): 0.8%

| Label | Support | Accuracy |
| --- | ---: | ---: |
| restructuredtext | 143 | 1.0000 |
| java | 10619 | 0.9932 |
| swift | 859 | 0.9907 |
| php | 1927 | 0.9766 |
| yaml | 10056 | 0.9151 |
| csharp | 1712 | 0.8773 |
| html | 18927 | 0.8711 |
| javascript_typescript | 12302 | 0.8276 |
| json | 4193 | 0.8235 |
| xml | 103 | 0.7961 |
| c_family | 3949 | 0.6900 |
| python | 1071 | 0.6555 |
| shell | 5498 | 0.6490 |
| dockerfile | 335 | 0.6239 |
| css | 153 | 0.2157 |
| ruby | 1058 | 0.0189 |
| text | 171271 | 0.0083 |
| __unknown__ | 0 | nan |

### pure_fragments

Single-label monitor fragments (baseline accuracy).

- Samples: 2000
- Characters evaluated: 7330306
- Overall accuracy: 0.9120
- High confusions: php->html 47411 (31.9%), dockerfile->shell 36671 (55.5%), powershell->encoding_base64 26651 (11.5%), html->javascript_typescript 25739 (7.8%), text->php 17099 (6.6%)

Expected foreign bytes for a 1536-byte fragment: 141.8/1536

#### Purity Analysis

| Language | Purity % | Top Misclassifications |
| --- | ---: | --- |
| csv | 98.1% | php (1.9%), yaml (0.0%), markdown (0.0%) |
| encoding_base64 | 97.2% | php (2.8%) |
| gettext_catalog | 96.6% | php (3.4%) |
| tex | 96.6% | php (3.1%), restructuredtext (0.3%) |
| rust | 96.4% | php (3.2%), restructuredtext (0.2%), ruby (0.1%) |
| encoding_base85 | 96.2% | php (3.8%) |
| encoding_hex | 95.8% | php (4.2%) |
| go | 95.7% | php (3.8%), c_family (0.3%), html (0.2%) |
| encoding_base32 | 95.4% | php (4.6%) |
| encoding_base58 | 95.1% | php (4.9%) |
| visual_basic | 95.0% | php (3.7%), kotlin (0.6%), xml (0.5%) |
| sql | 94.6% | php (3.5%), visual_basic (1.7%), javascript_typescript (0.1%) |
| csharp | 94.5% | php (4.6%), visual_basic (0.6%), powershell (0.2%) |
| swift | 94.3% | php (5.1%), c_family (0.6%) |
| c_family | 93.9% | php (4.6%), java (0.8%), restructuredtext (0.4%) |
| xml | 93.6% | php (4.5%), encoding_base64 (1.2%), tex (0.7%) |
| scala | 93.6% | php (6.4%) |
| text | 93.3% | php (6.6%), markdown (0.0%) |
| ruby | 92.8% | php (4.4%), python (1.6%), rust (0.9%) |
| css | 92.3% | php (2.9%), c_family (1.9%), encoding_base64 (1.7%) |
| python | 92.3% | php (4.2%), json (1.8%), c_family (0.4%) |
| java | 92.0% | php (5.1%), c_family (2.8%), scala (0.1%) |
| dart | 91.6% | php (7.9%), rust (0.5%) |
| svg | 90.1% | php (5.8%), css (2.5%), encoding_base64 (1.6%) |
| shell | 90.0% | php (4.4%), java (2.2%), xml (1.8%) |
| yaml | 89.4% | php (4.4%), javascript_typescript (2.8%), shell (0.9%) |
| kotlin | 89.2% | php (6.9%), dart (2.4%), python (0.7%) |
| javascript_typescript | 88.9% | html (4.4%), php (4.2%), c_family (1.8%) |
| json | 88.2% | php (8.6%), html (2.7%), tex (0.6%) |
| powershell | 84.4% | encoding_base64 (11.5%), php (3.2%), csharp (0.6%) |
| html | 83.8% | javascript_typescript (7.8%), php (3.0%), encoding_base64 (2.9%) |
| restructuredtext | 74.8% | python (6.7%), php (5.4%), shell (3.9%) |
| markdown | 68.6% | java (7.3%), php (6.2%), javascript_typescript (4.7%) |
| php | 66.3% | html (31.9%), javascript_typescript (1.0%), svg (0.3%) |
| dockerfile | 32.3% | shell (55.5%), php (8.9%), python (1.3%) |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| csv | 321188 | 0.9811 |
| encoding_base64 | 243336 | 0.9716 |
| gettext_catalog | 270307 | 0.9663 |
| tex | 190336 | 0.9661 |
| rust | 319899 | 0.9645 |
| encoding_base85 | 262975 | 0.9623 |
| encoding_hex | 291748 | 0.9579 |
| go | 226177 | 0.9571 |
| encoding_base32 | 284896 | 0.9538 |
| encoding_base58 | 240140 | 0.9510 |
| visual_basic | 299498 | 0.9497 |
| sql | 185856 | 0.9457 |
| csharp | 150077 | 0.9449 |
| swift | 195327 | 0.9434 |
| c_family | 201432 | 0.9392 |
| xml | 183065 | 0.9360 |
| scala | 215334 | 0.9358 |
| text | 257283 | 0.9335 |
| ruby | 126931 | 0.9280 |
| css | 286423 | 0.9230 |
| python | 231295 | 0.9226 |
| java | 275347 | 0.9201 |
| dart | 165671 | 0.9162 |
| svg | 196457 | 0.9008 |
| shell | 72186 | 0.8999 |
| yaml | 139037 | 0.8945 |
| kotlin | 104008 | 0.8921 |
| javascript_typescript | 143513 | 0.8885 |
| json | 139006 | 0.8820 |
| powershell | 231263 | 0.8444 |
| html | 330610 | 0.8377 |
| restructuredtext | 175338 | 0.7485 |
| markdown | 159465 | 0.6857 |
| php | 148856 | 0.6635 |
| dockerfile | 66026 | 0.3228 |
| __unknown__ | 0 | nan |

### sequence_pair

Two-language back-to-back sequences A->B.

- Samples: 2000
- Characters evaluated: 14357042
- Overall accuracy: 0.8995
- High confusions: dockerfile->shell 77449 (49.8%), php->html 73281 (26.2%), html->javascript_typescript 33032 (5.5%), powershell->encoding_base64 31294 (8.9%), restructuredtext->python 30869 (8.3%)

| Segment | Coverage |
| --- | ---: |
| First | 93.6% |
| Second | 86.3% |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| gettext_catalog | 781157 | 0.9722 |
| encoding_base32 | 515540 | 0.9690 |
| encoding_base64 | 421300 | 0.9679 |
| encoding_base58 | 416059 | 0.9622 |
| encoding_hex | 578954 | 0.9620 |
| csv | 540946 | 0.9594 |
| encoding_base85 | 462015 | 0.9588 |
| rust | 600445 | 0.9539 |
| visual_basic | 522444 | 0.9506 |
| java | 443006 | 0.9481 |
| tex | 439102 | 0.9444 |
| go | 513877 | 0.9424 |
| dart | 379444 | 0.9382 |
| csharp | 368522 | 0.9362 |
| ruby | 228586 | 0.9287 |
| css | 470567 | 0.9234 |
| c_family | 531110 | 0.9206 |
| scala | 406456 | 0.9170 |
| xml | 323403 | 0.9159 |
| sql | 344971 | 0.9156 |
| swift | 349545 | 0.9141 |
| python | 500205 | 0.9106 |
| svg | 295022 | 0.9063 |
| json | 352845 | 0.9041 |
| kotlin | 268533 | 0.8903 |
| text | 526828 | 0.8552 |
| javascript_typescript | 275079 | 0.8463 |
| html | 596240 | 0.8357 |
| powershell | 353481 | 0.8314 |
| shell | 148418 | 0.8156 |
| yaml | 272650 | 0.7599 |
| restructuredtext | 371211 | 0.6884 |
| php | 280038 | 0.6868 |
| markdown | 323456 | 0.6655 |
| dockerfile | 155587 | 0.3232 |
| __unknown__ | 0 | nan |

### sequence_triplet

Three-language back-to-back sequences A->B->C.

- Samples: 2000
- Characters evaluated: 21819852
- Overall accuracy: 0.9013
- High confusions: dockerfile->shell 138161 (43.7%), php->html 121665 (26.6%), html->javascript_typescript 73990 (8.8%), text->markdown 56265 (6.6%), dockerfile->markdown 48176 (15.2%)

| Segment | Coverage |
| --- | ---: |
| First | 93.8% |
| Second | 91.3% |
| Third | 85.3% |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| gettext_catalog | 1251310 | 0.9790 |
| encoding_hex | 1006676 | 0.9780 |
| encoding_base85 | 858258 | 0.9731 |
| encoding_base32 | 823412 | 0.9707 |
| encoding_base58 | 789693 | 0.9685 |
| rust | 859577 | 0.9680 |
| csv | 1009782 | 0.9672 |
| encoding_base64 | 657326 | 0.9641 |
| visual_basic | 882961 | 0.9572 |
| csharp | 475250 | 0.9546 |
| go | 698252 | 0.9479 |
| swift | 541562 | 0.9446 |
| tex | 596637 | 0.9437 |
| css | 736122 | 0.9361 |
| dart | 472731 | 0.9332 |
| kotlin | 309809 | 0.9323 |
| svg | 461032 | 0.9307 |
| xml | 560718 | 0.9169 |
| scala | 581213 | 0.9135 |
| ruby | 397065 | 0.9091 |
| c_family | 663131 | 0.9079 |
| java | 702658 | 0.9024 |
| sql | 487781 | 0.8889 |
| json | 389427 | 0.8880 |
| powershell | 501420 | 0.8804 |
| python | 674700 | 0.8791 |
| shell | 225160 | 0.8577 |
| text | 848134 | 0.8312 |
| javascript_typescript | 445552 | 0.8196 |
| html | 837059 | 0.8136 |
| yaml | 424111 | 0.8075 |
| restructuredtext | 407516 | 0.7024 |
| markdown | 470549 | 0.6788 |
| php | 457291 | 0.6578 |
| dockerfile | 315977 | 0.2577 |
| __unknown__ | 0 | nan |

### needle_64_plus

Natural monitor files with a short foreign-language needle sized 64-∞ printable chars (whitespace ignored).

- Samples: 125
- Characters evaluated: 518480
- Overall accuracy: 0.8668
- High confusions: markdown->php 9155 (7.0%), html->php 7898 (6.2%), encoding_base64->css 6547 (25.6%), dockerfile->shell 3845 (34.7%), markdown->html 3717 (2.9%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| xml | 525 | 1.0000 |
| rust | 5811 | 0.9919 |
| java | 1319 | 0.9909 |
| swift | 859 | 0.9907 |
| ruby | 582 | 0.9863 |
| restructuredtext | 39541 | 0.9727 |
| csharp | 1350 | 0.9622 |
| powershell | 21649 | 0.9441 |
| svg | 9008 | 0.9291 |
| yaml | 44159 | 0.9111 |
| html | 126555 | 0.9071 |
| shell | 31713 | 0.8950 |
| javascript_typescript | 23546 | 0.8784 |
| python | 4524 | 0.8601 |
| markdown | 130308 | 0.8538 |
| php | 18695 | 0.8246 |
| json | 7883 | 0.7373 |
| encoding_base64 | 25532 | 0.6896 |
| dockerfile | 11076 | 0.6359 |
| css | 11304 | 0.5923 |
| sql | 677 | 0.0355 |
| go | 1864 | 0.0000 |
| __unknown__ | 0 | nan |
| c_family | 0 | nan |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 111/125 | 57/125 | 0.50 | 90.7% | html (20.6%), __unknown__ (19.3%), php (19.0%) |
| Correct label | 100/125 | 95/125 | 0.73 | 82.6% | — |

### needle_32_63

Natural monitor files with a short foreign-language needle sized 32-63 printable chars (whitespace ignored).

- Samples: 51
- Characters evaluated: 160827
- Overall accuracy: 0.8899
- High confusions: html->php 3190 (5.4%), markdown->php 2338 (6.0%), shell->html 1752 (15.7%), c_family->markdown 1413 (99.9%), restructuredtext->php 1387 (8.8%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| dockerfile | 3544 | 0.9992 |
| php | 2961 | 0.9774 |
| svg | 4553 | 0.9464 |
| html | 59065 | 0.9407 |
| powershell | 12461 | 0.9255 |
| restructuredtext | 15748 | 0.9086 |
| markdown | 38803 | 0.9081 |
| css | 870 | 0.8391 |
| yaml | 5674 | 0.8162 |
| xml | 103 | 0.7961 |
| javascript_typescript | 1374 | 0.7082 |
| shell | 11135 | 0.6919 |
| json | 2625 | 0.5840 |
| python | 195 | 0.4615 |
| c_family | 1414 | 0.0000 |
| rust | 159 | 0.0000 |
| ruby | 87 | 0.0000 |
| encoding_base64 | 56 | 0.0000 |
| __unknown__ | 0 | nan |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 49/51 | 20/51 | 0.42 | 90.2% | php (34.8%), html (13.3%), yaml (13.2%) |
| Correct label | 41/51 | 41/51 | 0.72 | 76.7% | — |

### needle_16_31

Natural monitor files with a short foreign-language needle sized 16-31 printable chars (whitespace ignored).

- Samples: 51
- Characters evaluated: 178110
- Overall accuracy: 0.9230
- High confusions: html->php 2668 (3.4%), restructuredtext->shell 1963 (13.2%), json->__unknown__ 1827 (19.5%), restructuredtext->php 1507 (10.1%), html->javascript_typescript 923 (1.2%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| powershell | 1383 | 1.0000 |
| dockerfile | 1787 | 0.9983 |
| yaml | 12952 | 0.9879 |
| markdown | 29774 | 0.9659 |
| php | 3758 | 0.9476 |
| html | 77537 | 0.9467 |
| python | 1327 | 0.9375 |
| shell | 8376 | 0.9368 |
| javascript_typescript | 15567 | 0.9239 |
| csharp | 38 | 0.8947 |
| json | 9386 | 0.7988 |
| restructuredtext | 14853 | 0.7664 |
| css | 421 | 0.4869 |
| encoding_hex | 764 | 0.1479 |
| xml | 78 | 0.0000 |
| sql | 49 | 0.0000 |
| dart | 31 | 0.0000 |
| ruby | 29 | 0.0000 |
| __unknown__ | 0 | nan |
| c_family | 0 | nan |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 35/51 | 15/51 | 0.29 | 50.0% | yaml (49.9%), html (16.7%), php (13.9%) |
| Correct label | 28/51 | 24/51 | 0.42 | 38.8% | — |

### needle_4_15

Natural monitor files with a short foreign-language needle sized 4-15 printable chars (whitespace ignored).

- Samples: 43
- Characters evaluated: 126512
- Overall accuracy: 0.8249
- High confusions: html->php 5722 (7.1%), html->__unknown__ 2957 (3.7%), css->__unknown__ 2160 (49.3%), php->html 2148 (18.7%), markdown->html 1851 (13.6%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| restructuredtext | 443 | 1.0000 |
| dockerfile | 881 | 0.9955 |
| rust | 5811 | 0.9919 |
| yaml | 4214 | 0.9440 |
| html | 80097 | 0.8817 |
| markdown | 13638 | 0.7907 |
| php | 11500 | 0.7745 |
| javascript_typescript | 2422 | 0.5917 |
| shell | 488 | 0.5266 |
| css | 4383 | 0.2900 |
| ruby | 1085 | 0.0184 |
| go | 908 | 0.0000 |
| sql | 423 | 0.0000 |
| scala | 143 | 0.0000 |
| xml | 39 | 0.0000 |
| c_family | 37 | 0.0000 |
| __unknown__ | 0 | nan |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 14/43 | 5/43 | 0.13 | 38.2% | html (49.6%), markdown (22.5%), yaml (14.0%) |
| Correct label | 12/43 | 12/43 | 0.25 | 34.0% | — |

## Throughput Benchmarks

| Task | Device | Samples | Total Bytes | Throughput | Latency (s) | RSS Δ (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| throughput_1024 | cpu | 4 | 4096 | 4.59 KB/s | 0.87 | 530.94 |
| throughput_10240 | cpu | 4 | 40960 | 30.94 KB/s | 1.29 | 72.47 |
| throughput_102400 | cpu | 4 | 409600 | 63.78 KB/s | 6.27 | 259.42 |
| throughput_1048576 | cpu | 4 | 4194304 | 76.91 KB/s | 53.26 | 507.34 |
| throughput_1024 | cuda | 4 | 4096 | 18.84 KB/s | 0.21 | -318.89 |
| throughput_10240 | cuda | 4 | 40960 | 74.77 KB/s | 0.53 | 52.31 |
| throughput_102400 | cuda | 4 | 409600 | 72.96 KB/s | 5.48 | 78.66 |
| throughput_1048576 | cuda | 4 | 4194304 | 88.23 KB/s | 46.42 | 357.85 |

Latency is the total wall-clock time to run the benchmark loop over all samples per benchmark type, excluding the one warmup inference call that triggers JAX’s JIT compilation beforehand.
RSS Δ (MB) is the difference in the Python process’s resident set size (RSS) measured via `psutil` immediately before and after each throughput benchmark, approximating the net change in host memory usage attributable to the model and runtime.

Report generated at 2025-12-04 12:25:49