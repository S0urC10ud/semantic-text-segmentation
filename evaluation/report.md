# Segmenter Evaluation Report

- Checkpoint: `../train/checkpoints/sweeps/4q2ai4x7.msgpack`
- Model dim: 256
- Channels: 32, 64, 64, 128, 128, 128, 128, 256
- Chunk: 1024
- Batch size: 16
- Max samples per task: 2000
- Sample seed: 13
- Evaluation data root: `/home/s0urc10ud/text-segmentation/evaluation/data`
- Generated at: 2025-11-02T11:52:32

### Task Highlights

##### mal_injection
| Scenario | IoU ≥50% | Coverage |
| --- | --- | --- |
| Any non-wrapper | 1162/2000 (mean 0.56) | 73.2% |
| Correct payload | 597/2000 (mean 0.28) | 48.3% |

##### markdown_mix
| Wrapper | Non-text hits | Text hits | Correct hits | Non-text coverage | Correct coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 2026/2027 | 1/2027 | 1742/2027 | 99.3% | 84.7% |
| bare code | 1967/1973 | 6/1973 | 1603/1973 | 99.2% | 81.4% |
| inline code (\`...\`) | 1024/1028 | 4/1028 | 716/1028 | 98.6% | 68.5% |

Text coverage (IoU ≥50%): 76.5%

| Metric | Value |
| --- | --- |
| Host fenced IoU ≥50% | 926/998 hits, mean IoU 0.91, coverage 91.5% |
| Host bare IoU ≥50% | 908/1002 hits, mean IoU 0.89, coverage 89.1% |
| Other fenced IoU ≥50% | 816/1029 hits, mean IoU 0.75, coverage 75.1% |
| Other bare IoU ≥50% | 695/971 hits, mean IoU 0.69, coverage 69.8% |
| Wrong fence label fooled | 1/363 cases |

##### pure_fragments
1561/2000 samples stayed fully pure (no foreign chars). 1980/2000 stayed within ≤50% foreign coverage.
Expected foreign bytes for a 1536-byte fragment: 95.6/1536

##### sequence_pair
First segment coverage 97.9%
Second segment coverage 86.8%

##### sequence_triplet
First segment coverage 97.9%
Second segment coverage 97.1%
Third segment coverage 83.9%

##### needle buckets
| Bucket | Donor IoU ≥50% | Donor avg IoU | Donor coverage | Any IoU ≥50% | Any avg IoU | Any coverage |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| needle_64_plus | 1672/2000 | 0.79 | 92.0% | 1868/2000 | 0.92 | 97.0% |
| needle_32_63 | 1065/2000 | 0.49 | 55.7% | 1479/2000 | 0.69 | 72.0% |
| needle_16_31 | 730/2000 | 0.33 | 38.3% | 1142/2000 | 0.53 | 55.3% |
| needle_4_15 | 305/2000 | 0.14 | 19.1% | 680/2000 | 0.31 | 37.5% |


## Task Details

### mal_injection

Host fragments with malicious payload injections.

- Samples: 2000
- Characters evaluated: 3292830
- Overall accuracy: 0.7861
- High confusions: java->javascript_typescript 37988 (19.4%), python->javascript_typescript 27873 (17.1%), javascript_typescript->c_family 20176 (11.4%), python->c_family 19411 (11.9%), java->c_family 18840 (9.6%)

| Language | Non-wrapper IoU ≥50% | Non-wrapper avg IoU | Non-wrapper coverage | Correct IoU ≥50% | Correct avg IoU | Correct coverage | Top misclassifications |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| csharp | 221/221 | 0.99 | 99.1% | 215/221 | 0.89 | 97.3% (overall 91.3%) | c_family (6.7%), java (0.6%), powershell (0.4%) |
| go | 95/213 | 0.44 | 44.0% | 0/213 | 0.06 | 6.3% (overall 62.9%) | c_family (10.0%), json (3.8%), html (3.8%) |
| java | 182/212 | 0.83 | 83.8% | 79/212 | 0.35 | 32.0% (overall 58.9%) | javascript_typescript (19.4%), c_family (9.6%), csharp (6.0%) |
| javascript_typescript | 132/263 | 0.49 | 76.2% | 98/263 | 0.34 | 62.8% (overall 72.2%) | c_family (11.4%), java (3.8%), csharp (2.5%) |
| php | 90/232 | 0.38 | 37.9% | 43/232 | 0.18 | 17.8% (overall 59.4%) | c_family (11.3%), javascript_typescript (3.3%), shell (2.8%) |
| powershell | 156/208 | 0.73 | 91.7% | 140/208 | 0.60 | 78.5% (overall 82.5%) | c_family (7.9%), csharp (2.9%), shell (2.5%) |
| python | 155/219 | 0.64 | 64.2% | 0/219 | 0.00 | 0.0% (overall 56.0%) | javascript_typescript (17.1%), c_family (11.9%), java (3.8%) |
| ruby | 68/228 | 0.29 | 32.8% | 0/228 | 0.02 | 2.9% (overall 63.1%) | c_family (12.2%), javascript_typescript (3.8%), css (2.6%) |
| shell | 63/204 | 0.29 | 35.9% | 22/204 | 0.10 | 12.6% (overall 65.4%) | c_family (12.3%), powershell (3.5%), dockerfile (3.4%) |

### markdown_mix

Markdown-like text/code interleavings with optional fences.

- Samples: 2000
- Characters evaluated: 2762356
- Overall accuracy: 0.8017
- High confusions: text->c_family 191946 (21.9%), rust->c_family 12971 (16.2%), json->c_family 12920 (15.8%), html->c_family 12601 (15.1%), yaml->c_family 12270 (16.3%)

| Wrapper | Non-text hits | Text hits | Correct hits | Non-text coverage | Correct coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 2026/2027 | 1/2027 | 1742/2027 | 99.3% | 84.7% |
| bare code | 1967/1973 | 6/1973 | 1603/1973 | 99.2% | 81.4% |
| inline code (\`...\`) | 1024/1028 | 4/1028 | 716/1028 | 98.6% | 68.5% |

Text coverage (IoU ≥50%): 76.5%

| Label | Support | Accuracy |
| --- | ---: | ---: |
| c_family | 77014 | 0.8869 |
| encoding_base32 | 76520 | 0.8830 |
| encoding_base64 | 80277 | 0.8816 |
| encoding_hex | 81654 | 0.8716 |
| encoding_base85 | 82887 | 0.8702 |
| visual_basic | 80642 | 0.8695 |
| encoding_base58 | 78590 | 0.8692 |
| csv | 75878 | 0.8538 |
| css | 87749 | 0.8500 |
| dockerfile | 73171 | 0.8446 |
| go | 76598 | 0.8378 |
| php | 77280 | 0.8201 |
| html | 83351 | 0.8129 |
| powershell | 81212 | 0.8050 |
| json | 81864 | 0.8035 |
| sql | 75401 | 0.7933 |
| csharp | 75019 | 0.7908 |
| yaml | 75485 | 0.7893 |
| rust | 79862 | 0.7875 |
| java | 80144 | 0.7820 |
| text | 875624 | 0.7654 |
| ruby | 78938 | 0.7502 |
| python | 76913 | 0.7460 |
| shell | 68731 | 0.7405 |
| javascript_typescript | 81552 | 0.6921 |
| __unknown__ | 0 | nan |

### pure_fragments

Single-label validation fragments (baseline accuracy).

- Samples: 2000
- Characters evaluated: 2675343
- Overall accuracy: 0.9345
- High confusions: encoding_base85->c_family 37877 (31.4%), encoding_base58->c_family 34650 (30.1%), text->c_family 14064 (13.3%), csharp->c_family 10999 (10.6%), visual_basic->c_family 7851 (6.9%)

Expected foreign bytes for a 1536-byte fragment: 95.6/1536

#### Purity Analysis

| Language | Purity % | Top Misclassifications |
| --- | ---: | --- |
| css | 99.6% | c_family (0.4%), text (0.0%) |
| encoding_hex | 99.4% | c_family (0.6%) |
| csv | 99.3% | c_family (0.6%), dockerfile (0.0%), html (0.0%) |
| encoding_base64 | 98.7% | c_family (1.3%) |
| encoding_base32 | 98.5% | c_family (1.5%) |
| c_family | 98.3% | shell (0.7%), java (0.4%), sql (0.3%) |
| rust | 97.8% | c_family (0.7%), go (0.7%), python (0.6%) |
| go | 97.8% | c_family (2.0%), shell (0.1%), powershell (0.1%) |
| python | 97.7% | c_family (1.1%), shell (0.6%), ruby (0.4%) |
| html | 97.5% | c_family (2.5%) |
| powershell | 97.5% | c_family (1.3%), encoding_base58 (0.9%), shell (0.3%) |
| java | 96.8% | c_family (2.5%), javascript_typescript (0.7%), csharp (0.0%) |
| dockerfile | 96.8% | c_family (3.2%), visual_basic (0.0%) |
| yaml | 96.3% | c_family (1.8%), encoding_base64 (1.2%), shell (0.3%) |
| ruby | 95.9% | c_family (2.7%), python (0.9%), rust (0.3%) |
| shell | 95.5% | c_family (3.1%), json (1.1%), dockerfile (0.2%) |
| json | 95.2% | c_family (3.0%), csv (1.0%), text (0.9%) |
| sql | 93.7% | c_family (4.6%), csv (0.9%), encoding_hex (0.4%) |
| php | 93.5% | c_family (5.4%), java (0.8%), csharp (0.3%) |
| visual_basic | 90.7% | c_family (6.9%), html (1.0%), python (1.0%) |
| javascript_typescript | 89.4% | json (3.2%), java (2.6%), c_family (2.6%) |
| csharp | 89.0% | c_family (10.6%), encoding_base64 (0.4%), sql (0.0%) |
| text | 86.7% | c_family (13.3%) |
| encoding_base58 | 69.9% | c_family (30.1%) |
| encoding_base85 | 68.6% | c_family (31.4%) |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| css | 127914 | 0.9957 |
| encoding_hex | 107526 | 0.9936 |
| csv | 116802 | 0.9935 |
| encoding_base64 | 114116 | 0.9868 |
| encoding_base32 | 122440 | 0.9848 |
| c_family | 104597 | 0.9826 |
| rust | 106354 | 0.9783 |
| go | 105381 | 0.9777 |
| python | 112079 | 0.9770 |
| html | 120578 | 0.9752 |
| powershell | 110100 | 0.9745 |
| java | 114493 | 0.9681 |
| dockerfile | 75658 | 0.9675 |
| yaml | 107221 | 0.9634 |
| ruby | 99733 | 0.9593 |
| shell | 72060 | 0.9552 |
| json | 91562 | 0.9518 |
| sql | 99041 | 0.9367 |
| php | 89490 | 0.9352 |
| visual_basic | 114546 | 0.9073 |
| javascript_typescript | 118472 | 0.8944 |
| csharp | 103769 | 0.8900 |
| text | 105594 | 0.8668 |
| encoding_base58 | 115238 | 0.6993 |
| encoding_base85 | 120579 | 0.6859 |
| __unknown__ | 0 | nan |

### sequence_pair

Two-language back-to-back sequences A->B.

- Samples: 2000
- Characters evaluated: 5315911
- Overall accuracy: 0.9238
- High confusions: java->c_family 18529 (7.6%), csharp->c_family 16722 (8.4%), dockerfile->c_family 16627 (10.6%), json->c_family 16225 (8.2%), text->c_family 16185 (7.5%)

| Segment | Coverage |
| --- | ---: |
| First | 97.9% |
| Second | 86.8% |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| csv | 250033 | 0.9620 |
| encoding_base85 | 210869 | 0.9599 |
| encoding_hex | 232844 | 0.9542 |
| c_family | 211272 | 0.9494 |
| go | 230328 | 0.9491 |
| encoding_base58 | 221677 | 0.9476 |
| css | 228147 | 0.9468 |
| encoding_base64 | 225624 | 0.9439 |
| encoding_base32 | 234894 | 0.9421 |
| rust | 218697 | 0.9413 |
| html | 222484 | 0.9404 |
| visual_basic | 216538 | 0.9258 |
| python | 218013 | 0.9215 |
| text | 214931 | 0.9190 |
| sql | 189966 | 0.9130 |
| java | 244568 | 0.9112 |
| powershell | 203302 | 0.9107 |
| php | 187238 | 0.9101 |
| ruby | 196713 | 0.9060 |
| yaml | 216610 | 0.9060 |
| csharp | 198659 | 0.9050 |
| dockerfile | 156179 | 0.8827 |
| json | 197182 | 0.8805 |
| shell | 143220 | 0.8768 |
| javascript_typescript | 245923 | 0.8502 |
| __unknown__ | 0 | nan |

### sequence_triplet

Three-language back-to-back sequences A->B->C.

- Samples: 2000
- Characters evaluated: 8039933
- Overall accuracy: 0.9290
- High confusions: visual_basic->c_family 22489 (5.7%), powershell->c_family 20860 (6.1%), yaml->c_family 20073 (6.0%), java->c_family 19776 (6.1%), text->c_family 19676 (6.6%)

| Segment | Coverage |
| --- | ---: |
| First | 97.9% |
| Second | 97.1% |
| Third | 83.9% |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base64 | 336746 | 0.9583 |
| encoding_base32 | 339546 | 0.9572 |
| encoding_base58 | 331445 | 0.9567 |
| encoding_hex | 364264 | 0.9562 |
| csv | 352950 | 0.9552 |
| c_family | 335600 | 0.9520 |
| encoding_base85 | 328521 | 0.9489 |
| html | 366242 | 0.9484 |
| go | 335244 | 0.9460 |
| css | 368186 | 0.9384 |
| rust | 357641 | 0.9375 |
| csharp | 282810 | 0.9333 |
| text | 300368 | 0.9278 |
| visual_basic | 395279 | 0.9248 |
| yaml | 336900 | 0.9191 |
| json | 274767 | 0.9190 |
| php | 286337 | 0.9186 |
| sql | 306969 | 0.9173 |
| ruby | 298089 | 0.9148 |
| dockerfile | 206925 | 0.9147 |
| powershell | 340966 | 0.9076 |
| java | 325049 | 0.9022 |
| shell | 211571 | 0.8942 |
| python | 315451 | 0.8885 |
| javascript_typescript | 342067 | 0.8571 |
| __unknown__ | 0 | nan |

### needle_64_plus

Host fragments with a foreign-language needle injection sized 64-∞ printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 4094149
- Overall accuracy: 0.8651
- High confusions: encoding_base58->c_family 27609 (10.6%), encoding_base85->c_family 22741 (9.0%), css->c_family 22386 (10.0%), java->c_family 20006 (12.4%), visual_basic->c_family 19876 (12.2%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_hex | 245872 | 0.9333 |
| encoding_base64 | 258432 | 0.9330 |
| encoding_base32 | 237176 | 0.9188 |
| c_family | 140643 | 0.9106 |
| encoding_base85 | 251814 | 0.9075 |
| encoding_base58 | 261058 | 0.8909 |
| css | 223022 | 0.8821 |
| go | 168570 | 0.8715 |
| rust | 151783 | 0.8666 |
| dockerfile | 96494 | 0.8587 |
| csv | 126288 | 0.8577 |
| html | 119111 | 0.8550 |
| sql | 143609 | 0.8534 |
| powershell | 175301 | 0.8495 |
| visual_basic | 163514 | 0.8419 |
| text | 98870 | 0.8367 |
| php | 141915 | 0.8322 |
| yaml | 106793 | 0.8259 |
| ruby | 148877 | 0.8171 |
| python | 151589 | 0.8170 |
| csharp | 136651 | 0.8157 |
| java | 161768 | 0.8127 |
| javascript_typescript | 190865 | 0.8087 |
| json | 95594 | 0.8018 |
| shell | 98540 | 0.7941 |
| __unknown__ | 0 | nan |

| Scenario | IoU ≥50% | Mean IoU | Coverage | Top misclassifications |
| --- | --- | ---: | ---: | --- |
| Any non-wrapper | 1868/2000 | 0.92 | 97.0% | c_family (46.9%), csharp (5.8%), rust (5.2%) |
| Correct label | 1672/2000 | 0.79 | 92.0% | — |

### needle_32_63

Host fragments with a foreign-language needle injection sized 32-63 printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 2839542
- Overall accuracy: 0.9045
- High confusions: powershell->c_family 8131 (6.3%), python->c_family 8051 (6.4%), visual_basic->c_family 7699 (6.6%), csharp->c_family 7517 (6.5%), encoding_base32->c_family 7407 (6.3%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base58 | 104514 | 0.9443 |
| encoding_base64 | 111376 | 0.9417 |
| encoding_hex | 113342 | 0.9400 |
| encoding_base32 | 118464 | 0.9367 |
| html | 122002 | 0.9366 |
| encoding_base85 | 101377 | 0.9355 |
| csv | 120959 | 0.9350 |
| text | 107096 | 0.9292 |
| css | 127971 | 0.9280 |
| yaml | 108431 | 0.9224 |
| dockerfile | 87849 | 0.9157 |
| c_family | 122734 | 0.9129 |
| json | 93934 | 0.9034 |
| go | 131920 | 0.9018 |
| powershell | 128119 | 0.8949 |
| csharp | 116333 | 0.8932 |
| php | 109147 | 0.8882 |
| rust | 132375 | 0.8871 |
| sql | 106851 | 0.8834 |
| java | 127285 | 0.8826 |
| visual_basic | 115987 | 0.8804 |
| shell | 76547 | 0.8696 |
| python | 126246 | 0.8627 |
| ruby | 109704 | 0.8606 |
| javascript_typescript | 118979 | 0.8303 |
| __unknown__ | 0 | nan |

| Scenario | IoU ≥50% | Mean IoU | Coverage | Top misclassifications |
| --- | --- | ---: | ---: | --- |
| Any non-wrapper | 1479/2000 | 0.69 | 72.0% | c_family (12.9%), rust (9.3%), csharp (8.5%) |
| Correct label | 1065/2000 | 0.49 | 55.7% | — |

### needle_16_31

Host fragments with a foreign-language needle injection sized 16-31 printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 2766550
- Overall accuracy: 0.9277
- High confusions: sql->c_family 6757 (6.5%), csharp->c_family 6485 (6.1%), ruby->c_family 6202 (5.4%), visual_basic->c_family 5494 (4.1%), java->c_family 5276 (4.8%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base85 | 117006 | 0.9678 |
| csv | 120636 | 0.9629 |
| encoding_base58 | 113637 | 0.9627 |
| html | 112889 | 0.9605 |
| encoding_hex | 113210 | 0.9582 |
| encoding_base64 | 108104 | 0.9569 |
| encoding_base32 | 108440 | 0.9538 |
| text | 105895 | 0.9502 |
| css | 117265 | 0.9489 |
| yaml | 104951 | 0.9478 |
| powershell | 128463 | 0.9340 |
| go | 122950 | 0.9320 |
| c_family | 121938 | 0.9244 |
| dockerfile | 78123 | 0.9232 |
| visual_basic | 135540 | 0.9216 |
| rust | 116455 | 0.9165 |
| json | 93592 | 0.9157 |
| php | 109117 | 0.9075 |
| ruby | 115466 | 0.9050 |
| java | 109309 | 0.9025 |
| csharp | 106227 | 0.8984 |
| python | 113601 | 0.8891 |
| shell | 72352 | 0.8869 |
| sql | 104394 | 0.8834 |
| javascript_typescript | 116990 | 0.8629 |
| __unknown__ | 0 | nan |

| Scenario | IoU ≥50% | Mean IoU | Coverage | Top misclassifications |
| --- | --- | ---: | ---: | --- |
| Any non-wrapper | 1142/2000 | 0.53 | 55.3% | c_family (9.6%), rust (8.2%), csharp (6.9%) |
| Correct label | 730/2000 | 0.33 | 38.3% | — |

### needle_4_15

Host fragments with a foreign-language needle injection sized 4-15 printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 2729927
- Overall accuracy: 0.9451
- High confusions: csharp->c_family 5627 (5.1%), sql->c_family 5580 (5.5%), php->c_family 5145 (5.2%), visual_basic->c_family 3799 (3.1%), ruby->c_family 3781 (3.4%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base58 | 113187 | 0.9825 |
| encoding_base85 | 109072 | 0.9779 |
| csv | 115899 | 0.9765 |
| encoding_base32 | 112696 | 0.9721 |
| text | 107494 | 0.9701 |
| encoding_hex | 100450 | 0.9690 |
| encoding_base64 | 102144 | 0.9688 |
| html | 126641 | 0.9682 |
| yaml | 103316 | 0.9617 |
| css | 126927 | 0.9578 |
| c_family | 116285 | 0.9561 |
| go | 122831 | 0.9551 |
| java | 112289 | 0.9450 |
| powershell | 112293 | 0.9396 |
| json | 98221 | 0.9372 |
| dockerfile | 68651 | 0.9372 |
| ruby | 112796 | 0.9338 |
| shell | 78354 | 0.9272 |
| rust | 119779 | 0.9259 |
| visual_basic | 121482 | 0.9251 |
| csharp | 110944 | 0.9205 |
| python | 112614 | 0.9194 |
| php | 99816 | 0.9141 |
| sql | 101411 | 0.9078 |
| javascript_typescript | 124335 | 0.8736 |
| __unknown__ | 0 | nan |

| Scenario | IoU ≥50% | Mean IoU | Coverage | Top misclassifications |
| --- | --- | ---: | ---: | --- |
| Any non-wrapper | 680/2000 | 0.31 | 37.5% | c_family (9.1%), csharp (6.3%), sql (6.1%) |
| Correct label | 305/2000 | 0.14 | 19.1% | — |

## Throughput Benchmarks

| Task | Device | Samples | Total Bytes | Throughput | Latency (s) | RSS Δ (MB) | Device Δ (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| throughput_1024 | cuda | 8 | 8192 | 33.50 KB/s | 0.24 | n/a | n/a |
| throughput_10240 | cuda | 8 | 81920 | 78.72 KB/s | 1.02 | n/a | n/a |
| throughput_102400 | cuda | 8 | 819200 | 88.88 KB/s | 9.00 | n/a | n/a |
| throughput_1048576 | cuda | 8 | 8388608 | 92.35 KB/s | 88.70 | n/a | n/a |

Report generated at 2025-11-02 20:36:57