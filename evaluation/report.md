# Segmenter Evaluation Report

- Checkpoint: `../train/checkpoints/sweeps/7d62dvio.msgpack`
- Model dim: 256
- Channels: 32, 64, 64, 128, 128, 128, 128, 256
- Chunk: 1024
- Batch size: 128
- Max samples per task: 2000
- Sample seed: 13
- Evaluation data root: `/home/s0urc10ud/text-segmentation/evaluation/data`
- Generated at: 2025-11-02T11:52:32

### Task Highlights

##### mal_injection
| Scenario | IoU ≥50% | Coverage |
| --- | --- | --- |
| Any non-wrapper | 1284/2000 (mean 0.61) | 76.0% |
| Correct payload | 649/2000 (mean 0.29) | 49.3% |

##### markdown_mix
| Wrapper | Non-text hits | Text hits | Correct hits | Non-text coverage | Correct coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 2024/2027 | 3/2027 | 1757/2027 | 99.0% | 85.0% |
| bare code | 1968/1973 | 6/1973 | 1627/1973 | 98.9% | 82.0% |
| inline code (\`...\`) | 1018/1028 | 10/1028 | 716/1028 | 97.5% | 68.1% |

Text coverage (IoU ≥50%): 75.6%

| Metric | Value |
| --- | --- |
| Host fenced IoU ≥50% | 927/998 hits, mean IoU 0.91, coverage 91.0% |
| Host bare IoU ≥50% | 911/1002 hits, mean IoU 0.89, coverage 89.2% |
| Other fenced IoU ≥50% | 830/1029 hits, mean IoU 0.76, coverage 76.5% |
| Other bare IoU ≥50% | 716/971 hits, mean IoU 0.71, coverage 71.1% |
| Wrong fence label fooled | 2/363 cases |

##### pure_fragments
1527/2000 samples stayed fully pure (no foreign chars). 1973/2000 stayed within ≤50% foreign coverage.
Expected foreign bytes for a 1536-byte fragment: 97.7/1536

##### sequence_pair
First segment coverage 98.0%
Second segment coverage 86.9%

##### sequence_triplet
First segment coverage 98.0%
Second segment coverage 97.4%
Third segment coverage 83.9%

##### needle buckets
| Bucket | Donor IoU ≥50% | Donor avg IoU | Donor coverage | Any IoU ≥50% | Any avg IoU | Any coverage |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| needle_64_plus | 1688/2000 | 0.80 | 92.1% | 1877/2000 | 0.92 | 97.1% |
| needle_32_63 | 987/2000 | 0.45 | 52.5% | 1441/2000 | 0.68 | 70.9% |
| needle_16_31 | 632/2000 | 0.29 | 34.1% | 1088/2000 | 0.50 | 53.4% |
| needle_4_15 | 261/2000 | 0.12 | 18.0% | 650/2000 | 0.30 | 38.1% |


## Task Details

### mal_injection

Host fragments with malicious payload injections.

- Samples: 2000
- Characters evaluated: 3292830
- Overall accuracy: 0.7892
- High confusions: go->javascript_typescript 25085 (15.3%), python->c_family 21032 (12.9%), javascript_typescript->c_family 20561 (11.6%), php->c_family 19703 (12.1%), powershell->c_family 18673 (8.0%)

| Language | Non-wrapper IoU ≥50% | Non-wrapper avg IoU | Non-wrapper coverage | Correct IoU ≥50% | Correct avg IoU | Correct coverage | Top misclassifications |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| csharp | 216/221 | 0.97 | 97.5% | 204/221 | 0.83 | 87.5% (overall 85.0%) | c_family (6.7%), java (4.2%), dockerfile (3.5%) |
| go | 142/213 | 0.60 | 59.6% | 0/213 | 0.00 | 0.0% (overall 59.9%) | javascript_typescript (15.3%), c_family (10.2%), csharp (2.6%) |
| java | 196/212 | 0.89 | 89.3% | 135/212 | 0.57 | 59.6% (overall 72.6%) | c_family (9.3%), javascript_typescript (9.2%), dockerfile (2.6%) |
| javascript_typescript | 157/263 | 0.55 | 79.1% | 90/263 | 0.32 | 57.5% (overall 69.7%) | c_family (11.6%), java (6.9%), rust (1.9%) |
| php | 114/232 | 0.49 | 48.8% | 62/232 | 0.24 | 24.4% (overall 61.7%) | c_family (12.1%), javascript_typescript (3.2%), shell (2.8%) |
| powershell | 155/208 | 0.72 | 90.5% | 133/208 | 0.56 | 75.6% (overall 81.2%) | c_family (8.0%), shell (3.1%), csharp (3.0%) |
| python | 134/219 | 0.56 | 55.6% | 1/219 | 0.01 | 1.0% (overall 54.8%) | c_family (12.9%), javascript_typescript (9.9%), java (5.9%) |
| ruby | 84/228 | 0.34 | 36.0% | 8/228 | 0.04 | 2.6% (overall 64.3%) | c_family (12.5%), javascript_typescript (4.5%), yaml (2.0%) |
| shell | 86/204 | 0.39 | 47.9% | 16/204 | 0.08 | 7.9% (overall 65.4%) | c_family (11.9%), dockerfile (6.8%), powershell (2.1%) |

### markdown_mix

Markdown-like text/code interleavings with optional fences.

- Samples: 2000
- Characters evaluated: 2762356
- Overall accuracy: 0.8012
- High confusions: text->c_family 192015 (21.9%), json->c_family 12920 (15.8%), rust->c_family 12804 (16.0%), html->c_family 12597 (15.1%), yaml->c_family 12269 (16.3%)

| Wrapper | Non-text hits | Text hits | Correct hits | Non-text coverage | Correct coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 2024/2027 | 3/2027 | 1757/2027 | 99.0% | 85.0% |
| bare code | 1968/1973 | 6/1973 | 1627/1973 | 98.9% | 82.0% |
| inline code (\`...\`) | 1018/1028 | 10/1028 | 716/1028 | 97.5% | 68.1% |

Text coverage (IoU ≥50%): 75.6%

| Label | Support | Accuracy |
| --- | ---: | ---: |
| c_family | 77014 | 0.8920 |
| encoding_base32 | 76520 | 0.8828 |
| encoding_base64 | 80277 | 0.8813 |
| encoding_hex | 81654 | 0.8717 |
| encoding_base85 | 82887 | 0.8699 |
| encoding_base58 | 78590 | 0.8668 |
| visual_basic | 80642 | 0.8578 |
| csv | 75878 | 0.8567 |
| css | 87749 | 0.8511 |
| php | 77280 | 0.8307 |
| dockerfile | 73171 | 0.8306 |
| go | 76598 | 0.8261 |
| java | 80144 | 0.8244 |
| sql | 75401 | 0.8194 |
| html | 83351 | 0.8151 |
| powershell | 81212 | 0.8100 |
| json | 81864 | 0.8039 |
| yaml | 75485 | 0.7910 |
| rust | 79862 | 0.7902 |
| csharp | 75019 | 0.7830 |
| shell | 68731 | 0.7772 |
| ruby | 78938 | 0.7673 |
| text | 875624 | 0.7559 |
| python | 76913 | 0.7188 |
| javascript_typescript | 81552 | 0.7053 |
| __unknown__ | 0 | nan |

### pure_fragments

Single-label validation fragments (baseline accuracy).

- Samples: 2000
- Characters evaluated: 2675343
- Overall accuracy: 0.9339
- High confusions: encoding_base85->c_family 37889 (31.4%), encoding_base58->c_family 34650 (30.1%), text->c_family 14064 (13.3%), csharp->c_family 10999 (10.6%), visual_basic->c_family 7851 (6.9%)

Expected foreign bytes for a 1536-byte fragment: 97.7/1536

#### Purity Analysis

| Language | Purity % | Top Misclassifications |
| --- | ---: | --- |
| csv | 99.4% | c_family (0.6%), dockerfile (0.0%), ruby (0.0%) |
| encoding_hex | 99.4% | c_family (0.6%) |
| css | 99.0% | javascript_typescript (0.5%), c_family (0.4%), html (0.1%) |
| encoding_base64 | 98.7% | c_family (1.3%) |
| encoding_base32 | 98.5% | c_family (1.5%) |
| rust | 97.6% | go (1.1%), c_family (1.0%), javascript_typescript (0.2%) |
| c_family | 97.5% | java (0.8%), sql (0.5%), shell (0.5%) |
| html | 97.5% | c_family (2.5%), text (0.1%) |
| powershell | 97.4% | c_family (1.3%), encoding_base58 (0.9%), shell (0.3%) |
| python | 97.3% | c_family (1.1%), ruby (0.8%), shell (0.6%) |
| go | 97.2% | c_family (2.0%), powershell (0.4%), json (0.4%) |
| dockerfile | 96.5% | c_family (3.2%), yaml (0.2%), csharp (0.0%) |
| yaml | 96.5% | c_family (1.8%), encoding_base64 (1.2%), python (0.4%) |
| java | 96.4% | c_family (2.9%), php (0.6%), javascript_typescript (0.0%) |
| ruby | 95.7% | c_family (2.7%), php (1.0%), text (0.2%) |
| shell | 95.2% | c_family (3.3%), json (1.2%), dockerfile (0.2%) |
| sql | 95.1% | c_family (4.6%), shell (0.2%), rust (0.1%) |
| json | 94.9% | c_family (3.0%), csv (1.0%), text (0.9%) |
| php | 94.1% | c_family (5.4%), javascript_typescript (0.4%), csharp (0.1%) |
| javascript_typescript | 91.0% | c_family (2.6%), java (2.1%), css (1.9%) |
| visual_basic | 89.6% | c_family (6.9%), html (1.0%), sql (0.9%) |
| csharp | 89.0% | c_family (10.6%), encoding_base64 (0.4%), dockerfile (0.0%) |
| text | 86.5% | c_family (13.3%), csv (0.2%) |
| encoding_base58 | 69.9% | c_family (30.1%) |
| encoding_base85 | 68.6% | c_family (31.4%) |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| csv | 116802 | 0.9939 |
| encoding_hex | 107526 | 0.9936 |
| css | 127914 | 0.9899 |
| encoding_base64 | 114116 | 0.9868 |
| encoding_base32 | 122440 | 0.9848 |
| rust | 106354 | 0.9764 |
| c_family | 104597 | 0.9752 |
| html | 120578 | 0.9745 |
| powershell | 110100 | 0.9744 |
| python | 112079 | 0.9732 |
| go | 105381 | 0.9717 |
| dockerfile | 75658 | 0.9652 |
| yaml | 107221 | 0.9648 |
| java | 114493 | 0.9641 |
| ruby | 99733 | 0.9571 |
| shell | 72060 | 0.9517 |
| sql | 99041 | 0.9510 |
| json | 91562 | 0.9492 |
| php | 89490 | 0.9414 |
| javascript_typescript | 118472 | 0.9100 |
| visual_basic | 114546 | 0.8957 |
| csharp | 103769 | 0.8901 |
| text | 105594 | 0.8649 |
| encoding_base58 | 115238 | 0.6993 |
| encoding_base85 | 120579 | 0.6858 |
| __unknown__ | 0 | nan |

### sequence_pair

Two-language back-to-back sequences A->B.

- Samples: 2000
- Characters evaluated: 5315911
- Overall accuracy: 0.9251
- High confusions: java->c_family 17947 (7.3%), csharp->c_family 16811 (8.5%), dockerfile->c_family 16629 (10.6%), json->c_family 16226 (8.2%), text->c_family 16182 (7.5%)

| Segment | Coverage |
| --- | ---: |
| First | 98.0% |
| Second | 86.9% |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| csv | 250033 | 0.9628 |
| encoding_base85 | 210869 | 0.9597 |
| encoding_hex | 232844 | 0.9543 |
| c_family | 211272 | 0.9528 |
| encoding_base58 | 221677 | 0.9470 |
| css | 228147 | 0.9454 |
| go | 230328 | 0.9443 |
| encoding_base64 | 225624 | 0.9437 |
| encoding_base32 | 234894 | 0.9422 |
| html | 222484 | 0.9407 |
| rust | 218697 | 0.9405 |
| visual_basic | 216538 | 0.9248 |
| sql | 189966 | 0.9219 |
| text | 214931 | 0.9175 |
| python | 218013 | 0.9173 |
| php | 187238 | 0.9166 |
| java | 244568 | 0.9141 |
| yaml | 216610 | 0.9071 |
| ruby | 196713 | 0.9052 |
| powershell | 203302 | 0.9045 |
| csharp | 198659 | 0.8972 |
| json | 197182 | 0.8913 |
| shell | 143220 | 0.8846 |
| dockerfile | 156179 | 0.8836 |
| javascript_typescript | 245923 | 0.8701 |
| __unknown__ | 0 | nan |

### sequence_triplet

Three-language back-to-back sequences A->B->C.

- Samples: 2000
- Characters evaluated: 8039933
- Overall accuracy: 0.9305
- High confusions: visual_basic->c_family 22483 (5.7%), powershell->c_family 20694 (6.1%), yaml->c_family 20080 (6.0%), javascript_typescript->c_family 19786 (5.8%), text->c_family 19694 (6.6%)

| Segment | Coverage |
| --- | ---: |
| First | 98.0% |
| Second | 97.4% |
| Third | 83.9% |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base64 | 336746 | 0.9579 |
| csv | 352950 | 0.9578 |
| c_family | 335600 | 0.9574 |
| encoding_base32 | 339546 | 0.9572 |
| encoding_hex | 364264 | 0.9562 |
| encoding_base58 | 331445 | 0.9558 |
| encoding_base85 | 328521 | 0.9495 |
| html | 366242 | 0.9490 |
| go | 335244 | 0.9425 |
| rust | 357641 | 0.9369 |
| css | 368186 | 0.9362 |
| sql | 306969 | 0.9311 |
| json | 274767 | 0.9280 |
| text | 300368 | 0.9236 |
| csharp | 282810 | 0.9236 |
| java | 325049 | 0.9222 |
| yaml | 336900 | 0.9204 |
| visual_basic | 395279 | 0.9186 |
| ruby | 298089 | 0.9176 |
| php | 286337 | 0.9172 |
| dockerfile | 206925 | 0.9126 |
| powershell | 340966 | 0.9089 |
| shell | 211571 | 0.8983 |
| python | 315451 | 0.8858 |
| javascript_typescript | 342067 | 0.8677 |
| __unknown__ | 0 | nan |

### needle_64_plus

Host fragments with a foreign-language needle injection sized 64-∞ printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 4094149
- Overall accuracy: 0.8653
- High confusions: encoding_base58->c_family 27617 (10.6%), encoding_base85->c_family 22756 (9.0%), css->c_family 22338 (10.0%), java->c_family 20293 (12.5%), visual_basic->c_family 19812 (12.1%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_hex | 245872 | 0.9332 |
| encoding_base64 | 258432 | 0.9328 |
| encoding_base32 | 237176 | 0.9187 |
| encoding_base85 | 251814 | 0.9077 |
| c_family | 140643 | 0.8979 |
| encoding_base58 | 261058 | 0.8896 |
| css | 223022 | 0.8830 |
| sql | 143609 | 0.8768 |
| csv | 126288 | 0.8606 |
| rust | 151783 | 0.8599 |
| html | 119111 | 0.8596 |
| go | 168570 | 0.8557 |
| dockerfile | 96494 | 0.8552 |
| powershell | 175301 | 0.8511 |
| yaml | 106793 | 0.8398 |
| java | 161768 | 0.8353 |
| visual_basic | 163514 | 0.8321 |
| php | 141915 | 0.8319 |
| json | 95594 | 0.8228 |
| ruby | 148877 | 0.8202 |
| text | 98870 | 0.8199 |
| shell | 98540 | 0.8138 |
| javascript_typescript | 190865 | 0.8098 |
| csharp | 136651 | 0.8035 |
| python | 151589 | 0.8013 |
| __unknown__ | 0 | nan |

| Scenario | IoU ≥50% | Mean IoU | Coverage | Top misclassifications |
| --- | --- | ---: | ---: | --- |
| Any non-wrapper | 1877/2000 | 0.92 | 97.1% | c_family (47.8%), java (5.8%), rust (5.0%) |
| Correct label | 1688/2000 | 0.80 | 92.1% | — |

### needle_32_63

Host fragments with a foreign-language needle injection sized 32-63 printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 2839542
- Overall accuracy: 0.9008
- High confusions: python->c_family 8319 (6.6%), powershell->c_family 8228 (6.4%), java->c_family 7780 (6.1%), visual_basic->c_family 7761 (6.7%), csharp->c_family 7521 (6.5%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base58 | 104514 | 0.9444 |
| encoding_base64 | 111376 | 0.9421 |
| encoding_hex | 113342 | 0.9402 |
| csv | 120959 | 0.9377 |
| encoding_base32 | 118464 | 0.9369 |
| html | 122002 | 0.9356 |
| encoding_base85 | 101377 | 0.9355 |
| text | 107096 | 0.9292 |
| yaml | 108431 | 0.9257 |
| css | 127971 | 0.9234 |
| dockerfile | 87849 | 0.9117 |
| c_family | 122734 | 0.9096 |
| json | 93934 | 0.8973 |
| go | 131920 | 0.8968 |
| powershell | 128119 | 0.8930 |
| sql | 106851 | 0.8926 |
| java | 127285 | 0.8895 |
| php | 109147 | 0.8848 |
| rust | 132375 | 0.8809 |
| csharp | 116333 | 0.8777 |
| shell | 76547 | 0.8711 |
| visual_basic | 115987 | 0.8584 |
| ruby | 109704 | 0.8541 |
| python | 126246 | 0.8367 |
| javascript_typescript | 118979 | 0.8239 |
| __unknown__ | 0 | nan |

| Scenario | IoU ≥50% | Mean IoU | Coverage | Top misclassifications |
| --- | --- | ---: | ---: | --- |
| Any non-wrapper | 1441/2000 | 0.68 | 70.9% | c_family (13.5%), csharp (8.2%), java (7.7%) |
| Correct label | 987/2000 | 0.45 | 52.5% | — |

### needle_16_31

Host fragments with a foreign-language needle injection sized 16-31 printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 2766550
- Overall accuracy: 0.9270
- High confusions: sql->c_family 6612 (6.3%), csharp->c_family 6573 (6.2%), ruby->c_family 6336 (5.5%), java->c_family 5730 (5.2%), visual_basic->c_family 5527 (4.1%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base85 | 117006 | 0.9677 |
| html | 112889 | 0.9647 |
| csv | 120636 | 0.9631 |
| encoding_base58 | 113637 | 0.9623 |
| encoding_hex | 113210 | 0.9583 |
| encoding_base64 | 108104 | 0.9569 |
| encoding_base32 | 108440 | 0.9541 |
| text | 105895 | 0.9506 |
| yaml | 104951 | 0.9480 |
| css | 117265 | 0.9420 |
| powershell | 128463 | 0.9321 |
| json | 93592 | 0.9270 |
| c_family | 121938 | 0.9267 |
| dockerfile | 78123 | 0.9224 |
| go | 122950 | 0.9219 |
| php | 109117 | 0.9146 |
| rust | 116455 | 0.9128 |
| java | 109309 | 0.9018 |
| visual_basic | 135540 | 0.8985 |
| sql | 104394 | 0.8979 |
| ruby | 115466 | 0.8939 |
| csharp | 106227 | 0.8933 |
| python | 113601 | 0.8894 |
| shell | 72352 | 0.8834 |
| javascript_typescript | 116990 | 0.8772 |
| __unknown__ | 0 | nan |

| Scenario | IoU ≥50% | Mean IoU | Coverage | Top misclassifications |
| --- | --- | ---: | ---: | --- |
| Any non-wrapper | 1088/2000 | 0.50 | 53.4% | c_family (10.7%), rust (7.3%), csharp (7.0%) |
| Correct label | 632/2000 | 0.29 | 34.1% | — |

### needle_4_15

Host fragments with a foreign-language needle injection sized 4-15 printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 2729927
- Overall accuracy: 0.9446
- High confusions: csharp->c_family 5672 (5.1%), sql->c_family 5501 (5.4%), php->c_family 5147 (5.2%), ruby->c_family 3829 (3.4%), visual_basic->c_family 3822 (3.1%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base58 | 113187 | 0.9824 |
| encoding_base85 | 109072 | 0.9780 |
| csv | 115899 | 0.9770 |
| encoding_base32 | 112696 | 0.9722 |
| html | 126641 | 0.9697 |
| encoding_hex | 100450 | 0.9696 |
| encoding_base64 | 102144 | 0.9696 |
| text | 107494 | 0.9674 |
| yaml | 103316 | 0.9593 |
| css | 126927 | 0.9524 |
| c_family | 116285 | 0.9519 |
| go | 122831 | 0.9513 |
| java | 112289 | 0.9474 |
| json | 98221 | 0.9446 |
| ruby | 112796 | 0.9374 |
| powershell | 112293 | 0.9365 |
| dockerfile | 68651 | 0.9358 |
| rust | 119779 | 0.9310 |
| shell | 78354 | 0.9230 |
| sql | 101411 | 0.9192 |
| visual_basic | 121482 | 0.9139 |
| python | 112614 | 0.9135 |
| csharp | 110944 | 0.9123 |
| php | 99816 | 0.9119 |
| javascript_typescript | 124335 | 0.8819 |
| __unknown__ | 0 | nan |

| Scenario | IoU ≥50% | Mean IoU | Coverage | Top misclassifications |
| --- | --- | ---: | ---: | --- |
| Any non-wrapper | 650/2000 | 0.30 | 38.1% | c_family (9.3%), sql (6.5%), csharp (6.0%) |
| Correct label | 261/2000 | 0.12 | 18.0% | — |

## Throughput Benchmarks

| Task | Device | Samples | Total Bytes | Throughput | Latency (s) | RSS Δ (MB) | Device Δ (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| throughput_1024 | cuda | 8 | 8192 | 23.35 KB/s | 0.34 | n/a | n/a |
| throughput_10240 | cuda | 8 | 81920 | 86.21 KB/s | 0.93 | n/a | n/a |
| throughput_102400 | cuda | 8 | 819200 | 85.29 KB/s | 9.38 | n/a | n/a |
| throughput_1048576 | cuda | 8 | 8388608 | 99.35 KB/s | 82.45 | n/a | n/a |

Report generated at 2025-11-03 20:46:02