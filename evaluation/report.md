# Segmenter Evaluation Report

- Checkpoint: `../train/checkpoints/sweeps/l68s24dx.msgpack`
- Model dim: 256
- Channels: 32, 64, 64, 128, 128, 128, 128, 256
- Chunk: 1536
- Batch size: 128
- Max samples per task: 2000
- Sample seed: 13
- Evaluation data root: `/home/s0urc10ud/text-segmentation/evaluation/data`
- Generated at: 2025-11-02T11:52:32

### Task Highlights

##### mal_injection
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Avg coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 1204/2000 | 556/2000 | 0.38 | 72.6% |
| Correct payload | 611/2000 | 604/2000 | 0.30 | 47.5% |

##### markdown_mix
_Text hits column: lower is better._
| Wrapper | Non-text cov ≥50% | Non-text IoU ≥50% | Non-text avg coverage | Non-text avg IoU | Text hits | Correct cov ≥50% | Correct IoU ≥50% | Correct avg coverage | Correct avg IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 2027/2027 | 2027/2027 | 99.7% | 1.00 | 0/2027 | 1988/2027 | 1988/2027 | 97.4% | 0.97 |
| bare code | 1971/1973 | 1971/1973 | 99.6% | 1.00 | 2/1973 | 1925/1973 | 1925/1973 | 96.9% | 0.97 |
| inline code (\`...\`) | 1027/1028 | 1027/1028 | 99.6% | 1.00 | 1/1028 | 876/1028 | 876/1028 | 84.9% | 0.85 |

Text coverage (IoU ≥50%): 92.1%

| Metric | Value |
| --- | --- |
| Host fenced IoU ≥50% | 988/998 hits, mean IoU 0.98, coverage 97.9% |
| Host bare IoU ≥50% | 983/1002 hits, mean IoU 0.97, coverage 97.0% |
| Other fenced IoU ≥50% | 1000/1029 hits, mean IoU 0.96, coverage 96.6% |
| Other bare IoU ≥50% | 942/971 hits, mean IoU 0.97, coverage 96.8% |
| Wrong fence label fooled | 0/363 cases |

##### pure_fragments
1957/2000 samples stayed fully pure (no foreign chars). 1996/2000 stayed within ≤50% foreign coverage.
Expected foreign bytes for a 1536-byte fragment: 3.7/1536

##### sequence_pair
First segment coverage 99.3%
Second segment coverage 83.1%

##### sequence_triplet
First segment coverage 99.3%
Second segment coverage 98.7%
Third segment coverage 78.7%

##### needle_64_plus
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 1918/2000 | 1331/2000 | 0.68 | 97.9% |
| Correct payload | 1791/2000 | 1737/2000 | 0.83 | 90.8% |

##### needle_32_63
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 1707/2000 | 636/2000 | 0.51 | 83.3% |
| Correct payload | 1474/2000 | 1408/2000 | 0.66 | 72.7% |

##### needle_16_31
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 1419/2000 | 501/2000 | 0.41 | 68.7% |
| Correct payload | 1135/2000 | 1068/2000 | 0.49 | 54.7% |

##### needle_4_15
| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage |
| --- | --- | --- | --- | --- |
| Any non-wrapper | 954/2000 | 298/2000 | 0.25 | 52.2% |
| Correct payload | 620/2000 | 569/2000 | 0.26 | 33.4% |


## Task Details

### mal_injection

Host fragments with malicious payload injections.

- Samples: 2000
- Characters evaluated: 3292830
- Overall accuracy: 0.7812
- High confusions: csharp->c_family 28051 (12.8%), javascript_typescript->c_family 27157 (15.3%), powershell->c_family 24835 (10.6%), python->c_family 23084 (14.2%), python->javascript_typescript 22960 (14.1%)

| Language | Non-wrapper cov ≥50% | Non-wrapper coverage (avg) | Non-wrapper IoU ≥50% | Non-wrapper avg IoU | Correct cov ≥50% | Correct coverage (avg) | Correct IoU ≥50% | Correct avg IoU | Top misclassifications |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | ---: | --- |
| csharp | 219/221 | 98.1% | 114/221 | 0.63 | 188/221 | 84.8% (overall 85.8%) | 186/221 | 0.83 | c_family (12.8%), java (0.6%), javascript_typescript (0.2%) |
| go | 91/213 | 40.6% | 35/213 | 0.26 | 6/213 | 4.9% (overall 60.0%) | 6/213 | 0.05 | c_family (13.0%), javascript_typescript (8.7%), dockerfile (2.3%) |
| java | 181/212 | 80.3% | 90/212 | 0.54 | 118/212 | 47.9% (overall 69.2%) | 117/212 | 0.54 | c_family (11.6%), javascript_typescript (9.2%), html (3.2%) |
| javascript_typescript | 163/263 | 79.6% | 70/263 | 0.39 | 98/263 | 62.5% (overall 73.0%) | 97/263 | 0.35 | c_family (15.3%), java (2.0%), csharp (1.9%) |
| php | 101/232 | 42.4% | 38/232 | 0.27 | 19/232 | 9.1% (overall 56.1%) | 18/232 | 0.09 | javascript_typescript (10.9%), c_family (9.9%), shell (2.4%) |
| powershell | 149/208 | 90.5% | 101/208 | 0.52 | 129/208 | 77.7% (overall 81.6%) | 129/208 | 0.57 | c_family (10.6%), csharp (2.9%), shell (1.1%) |
| python | 123/219 | 52.3% | 43/219 | 0.33 | 2/219 | 1.3% (overall 56.7%) | 1/219 | 0.01 | c_family (14.2%), javascript_typescript (14.1%), java (2.3%) |
| ruby | 94/228 | 43.2% | 32/228 | 0.25 | 15/228 | 10.0% (overall 63.4%) | 14/228 | 0.08 | c_family (15.0%), javascript_typescript (5.9%), python (2.0%) |
| shell | 83/204 | 42.8% | 33/204 | 0.25 | 36/204 | 17.1% (overall 66.9%) | 36/204 | 0.17 | c_family (12.6%), dockerfile (4.4%), powershell (3.1%) |

### markdown_mix

Markdown-like text/code interleavings with optional fences.

- Samples: 2000
- Characters evaluated: 2762356
- Overall accuracy: 0.9485
- High confusions: text->c_family 62214 (7.1%), javascript_typescript->java 2940 (3.6%), ruby->python 2389 (3.0%), sql->c_family 2300 (3.1%), dockerfile->c_family 1936 (2.6%)

_Text hits column: lower is better._
| Wrapper | Non-text cov ≥50% | Non-text IoU ≥50% | Non-text avg coverage | Non-text avg IoU | Text hits | Correct cov ≥50% | Correct IoU ≥50% | Correct avg coverage | Correct avg IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| \`\`\` fenced \`\`\` | 2027/2027 | 2027/2027 | 99.7% | 1.00 | 0/2027 | 1988/2027 | 1988/2027 | 97.4% | 0.97 |
| bare code | 1971/1973 | 1971/1973 | 99.6% | 1.00 | 2/1973 | 1925/1973 | 1925/1973 | 96.9% | 0.97 |
| inline code (\`...\`) | 1027/1028 | 1027/1028 | 99.6% | 1.00 | 1/1028 | 876/1028 | 876/1028 | 84.9% | 0.85 |

Text coverage (IoU ≥50%): 92.1%

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_hex | 81654 | 0.9899 |
| encoding_base58 | 78590 | 0.9858 |
| go | 76598 | 0.9847 |
| csv | 75878 | 0.9843 |
| encoding_base85 | 82887 | 0.9806 |
| php | 77280 | 0.9790 |
| encoding_base32 | 76520 | 0.9789 |
| encoding_base64 | 80277 | 0.9754 |
| html | 83351 | 0.9719 |
| c_family | 77014 | 0.9659 |
| json | 81864 | 0.9631 |
| shell | 68731 | 0.9629 |
| rust | 79862 | 0.9556 |
| dockerfile | 73171 | 0.9544 |
| sql | 75401 | 0.9541 |
| css | 87749 | 0.9534 |
| visual_basic | 80642 | 0.9522 |
| powershell | 81212 | 0.9521 |
| csharp | 75019 | 0.9515 |
| python | 76913 | 0.9502 |
| yaml | 75485 | 0.9468 |
| java | 80144 | 0.9398 |
| ruby | 78938 | 0.9287 |
| text | 875624 | 0.9215 |
| javascript_typescript | 81552 | 0.9067 |
| __unknown__ | 0 | nan |

### pure_fragments

Single-label validation fragments (baseline accuracy).

- Samples: 2000
- Characters evaluated: 2675343
- Overall accuracy: 0.9980
- High confusions: visual_basic->sql 1262 (1.1%), yaml->encoding_base64 1260 (1.2%), javascript_typescript->java 909 (0.8%), csharp->encoding_base64 375 (0.4%), go->json 361 (0.3%)

Expected foreign bytes for a 1536-byte fragment: 3.7/1536

#### Purity Analysis

| Language | Purity % | Top Misclassifications |
| --- | ---: | --- |
| csv | 100.0% | — |
| dockerfile | 100.0% | — |
| encoding_base32 | 100.0% | — |
| encoding_base58 | 100.0% | — |
| encoding_base64 | 100.0% | — |
| encoding_base85 | 100.0% | — |
| encoding_hex | 100.0% | — |
| html | 100.0% | — |
| java | 100.0% | — |
| ruby | 100.0% | — |
| text | 100.0% | — |
| json | 100.0% | csv (0.0%) |
| php | 100.0% | dockerfile (0.0%) |
| c_family | 100.0% | css (0.0%), csv (0.0%), text (0.0%) |
| css | 100.0% | javascript_typescript (0.0%) |
| rust | 99.9% | go (0.0%), c_family (0.0%) |
| powershell | 99.9% | shell (0.1%), python (0.0%), rust (0.0%) |
| sql | 99.9% | javascript_typescript (0.1%), go (0.0%), csharp (0.0%) |
| python | 99.8% | shell (0.1%), text (0.0%), csharp (0.0%) |
| csharp | 99.6% | encoding_base64 (0.4%), sql (0.0%) |
| go | 99.5% | json (0.3%), powershell (0.1%), csharp (0.0%) |
| shell | 99.5% | c_family (0.3%), json (0.2%), python (0.0%) |
| javascript_typescript | 99.1% | java (0.8%), rust (0.0%), csharp (0.0%) |
| visual_basic | 98.9% | sql (1.1%), csharp (0.0%) |
| yaml | 98.7% | encoding_base64 (1.2%), python (0.1%), dockerfile (0.0%) |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base32 | 122440 | 1.0000 |
| encoding_base85 | 120579 | 1.0000 |
| html | 120578 | 1.0000 |
| csv | 116802 | 1.0000 |
| encoding_base58 | 115238 | 1.0000 |
| java | 114493 | 1.0000 |
| encoding_base64 | 114116 | 1.0000 |
| encoding_hex | 107526 | 1.0000 |
| text | 105594 | 1.0000 |
| ruby | 99733 | 1.0000 |
| dockerfile | 75658 | 1.0000 |
| json | 91562 | 0.9999 |
| php | 89490 | 0.9999 |
| c_family | 104597 | 0.9998 |
| css | 127914 | 0.9997 |
| rust | 106354 | 0.9994 |
| powershell | 110100 | 0.9991 |
| sql | 99041 | 0.9991 |
| python | 112079 | 0.9983 |
| csharp | 103769 | 0.9963 |
| go | 105381 | 0.9952 |
| shell | 72060 | 0.9948 |
| javascript_typescript | 118472 | 0.9914 |
| visual_basic | 114546 | 0.9890 |
| yaml | 107221 | 0.9873 |
| __unknown__ | 0 | nan |

### sequence_pair

Two-language back-to-back sequences A->B.

- Samples: 2000
- Characters evaluated: 5315911
- Overall accuracy: 0.9127
- High confusions: dockerfile->c_family 25894 (16.6%), java->c_family 24943 (10.2%), text->c_family 24334 (11.3%), csharp->c_family 21579 (10.9%), json->c_family 20938 (10.6%)

| Segment | Coverage |
| --- | ---: |
| First | 99.3% |
| Second | 83.1% |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| c_family | 211272 | 0.9877 |
| css | 228147 | 0.9391 |
| csv | 250033 | 0.9383 |
| encoding_hex | 232844 | 0.9375 |
| encoding_base64 | 225624 | 0.9344 |
| encoding_base85 | 210869 | 0.9330 |
| rust | 218697 | 0.9293 |
| html | 222484 | 0.9215 |
| python | 218013 | 0.9200 |
| encoding_base58 | 221677 | 0.9199 |
| go | 230328 | 0.9152 |
| sql | 189966 | 0.9140 |
| visual_basic | 216538 | 0.9139 |
| encoding_base32 | 234894 | 0.9109 |
| javascript_typescript | 245923 | 0.9092 |
| php | 187238 | 0.9071 |
| powershell | 203302 | 0.9038 |
| ruby | 196713 | 0.8979 |
| yaml | 216610 | 0.8954 |
| java | 244568 | 0.8904 |
| json | 197182 | 0.8871 |
| csharp | 198659 | 0.8868 |
| text | 214931 | 0.8842 |
| shell | 143220 | 0.8562 |
| dockerfile | 156179 | 0.8336 |
| __unknown__ | 0 | nan |

### sequence_triplet

Three-language back-to-back sequences A->B->C.

- Samples: 2000
- Characters evaluated: 8039933
- Overall accuracy: 0.9214
- High confusions: visual_basic->c_family 34525 (8.7%), powershell->c_family 29886 (8.8%), javascript_typescript->c_family 29546 (8.6%), css->c_family 29181 (7.9%), yaml->c_family 28877 (8.6%)

| Segment | Coverage |
| --- | ---: |
| First | 99.3% |
| Second | 98.7% |
| Third | 78.7% |

| Label | Support | Accuracy |
| --- | ---: | ---: |
| c_family | 335600 | 0.9895 |
| csv | 352950 | 0.9474 |
| encoding_base32 | 339546 | 0.9419 |
| encoding_base64 | 336746 | 0.9408 |
| encoding_base58 | 331445 | 0.9400 |
| encoding_hex | 364264 | 0.9374 |
| csharp | 282810 | 0.9299 |
| rust | 357641 | 0.9289 |
| json | 274767 | 0.9281 |
| html | 366242 | 0.9263 |
| encoding_base85 | 328521 | 0.9235 |
| go | 335244 | 0.9214 |
| sql | 306969 | 0.9193 |
| java | 325049 | 0.9192 |
| css | 368186 | 0.9186 |
| ruby | 298089 | 0.9140 |
| php | 286337 | 0.9135 |
| python | 315451 | 0.9047 |
| yaml | 336900 | 0.9027 |
| visual_basic | 395279 | 0.9026 |
| powershell | 340966 | 0.9021 |
| text | 300368 | 0.8987 |
| dockerfile | 206925 | 0.8968 |
| javascript_typescript | 342067 | 0.8834 |
| shell | 211571 | 0.8773 |
| __unknown__ | 0 | nan |

### needle_64_plus

Host fragments with a foreign-language needle injection sized 64-∞ printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 4094149
- Overall accuracy: 0.8529
- High confusions: encoding_base58->c_family 40863 (15.7%), encoding_base85->c_family 34814 (13.8%), css->c_family 32621 (14.6%), javascript_typescript->c_family 26710 (14.0%), java->c_family 26088 (16.1%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| c_family | 140643 | 0.9611 |
| encoding_hex | 245872 | 0.9111 |
| encoding_base64 | 258432 | 0.9050 |
| encoding_base32 | 237176 | 0.9020 |
| encoding_base85 | 251814 | 0.8604 |
| powershell | 175301 | 0.8505 |
| go | 168570 | 0.8499 |
| json | 95594 | 0.8492 |
| ruby | 148877 | 0.8480 |
| css | 223022 | 0.8463 |
| rust | 151783 | 0.8452 |
| yaml | 106793 | 0.8435 |
| encoding_base58 | 261058 | 0.8414 |
| sql | 143609 | 0.8413 |
| python | 151589 | 0.8401 |
| php | 141915 | 0.8400 |
| dockerfile | 96494 | 0.8362 |
| html | 119111 | 0.8344 |
| shell | 98540 | 0.8312 |
| csv | 126288 | 0.8304 |
| csharp | 136651 | 0.8270 |
| visual_basic | 163514 | 0.8177 |
| javascript_typescript | 190865 | 0.8128 |
| java | 161768 | 0.8019 |
| text | 98870 | 0.7683 |
| __unknown__ | 0 | nan |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 1918/2000 | 1331/2000 | 0.68 | 97.9% | c_family (73.8%), csharp (4.7%), powershell (3.4%) |
| Correct label | 1791/2000 | 1737/2000 | 0.83 | 90.8% | — |

### needle_32_63

Host fragments with a foreign-language needle injection sized 32-63 printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 2839542
- Overall accuracy: 0.9360
- High confusions: css->c_family 6962 (5.4%), javascript_typescript->c_family 6841 (5.7%), powershell->c_family 6698 (5.2%), csv->c_family 6354 (5.3%), visual_basic->c_family 6331 (5.5%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| c_family | 122734 | 0.9664 |
| encoding_base64 | 111376 | 0.9574 |
| dockerfile | 87849 | 0.9556 |
| encoding_base58 | 104514 | 0.9525 |
| encoding_hex | 113342 | 0.9512 |
| encoding_base32 | 118464 | 0.9502 |
| encoding_base85 | 101377 | 0.9497 |
| html | 122002 | 0.9481 |
| csv | 120959 | 0.9464 |
| text | 107096 | 0.9446 |
| json | 93934 | 0.9430 |
| css | 127971 | 0.9409 |
| csharp | 116333 | 0.9389 |
| yaml | 108431 | 0.9380 |
| go | 131920 | 0.9336 |
| rust | 132375 | 0.9313 |
| shell | 76547 | 0.9291 |
| sql | 106851 | 0.9269 |
| powershell | 128119 | 0.9260 |
| ruby | 109704 | 0.9241 |
| java | 127285 | 0.9231 |
| php | 109147 | 0.9224 |
| python | 126246 | 0.9109 |
| visual_basic | 115987 | 0.9080 |
| javascript_typescript | 118979 | 0.8917 |
| __unknown__ | 0 | nan |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 1707/2000 | 636/2000 | 0.51 | 83.3% | c_family (12.5%), csharp (11.9%), python (9.9%) |
| Correct label | 1474/2000 | 1408/2000 | 0.66 | 72.7% | — |

### needle_16_31

Host fragments with a foreign-language needle injection sized 16-31 printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 2766550
- Overall accuracy: 0.9564
- High confusions: javascript_typescript->c_family 3682 (3.1%), go->c_family 3658 (3.0%), java->c_family 3632 (3.3%), csv->c_family 3549 (2.9%), encoding_base58->c_family 3479 (3.1%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base64 | 108104 | 0.9767 |
| encoding_hex | 113210 | 0.9757 |
| text | 105895 | 0.9740 |
| encoding_base85 | 117006 | 0.9738 |
| c_family | 121938 | 0.9737 |
| encoding_base32 | 108440 | 0.9726 |
| html | 112889 | 0.9709 |
| csv | 120636 | 0.9699 |
| json | 93592 | 0.9689 |
| encoding_base58 | 113637 | 0.9684 |
| css | 117265 | 0.9632 |
| dockerfile | 78123 | 0.9571 |
| powershell | 128463 | 0.9549 |
| yaml | 104951 | 0.9546 |
| csharp | 106227 | 0.9515 |
| ruby | 115466 | 0.9511 |
| rust | 116455 | 0.9480 |
| sql | 104394 | 0.9457 |
| go | 122950 | 0.9450 |
| php | 109117 | 0.9410 |
| visual_basic | 135540 | 0.9407 |
| java | 109309 | 0.9348 |
| javascript_typescript | 116990 | 0.9333 |
| shell | 72352 | 0.9306 |
| python | 113601 | 0.9299 |
| __unknown__ | 0 | nan |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 1419/2000 | 501/2000 | 0.41 | 68.7% | c_family (11.6%), csharp (10.1%), python (8.4%) |
| Correct label | 1135/2000 | 1068/2000 | 0.49 | 54.7% | — |

### needle_4_15

Host fragments with a foreign-language needle injection sized 4-15 printable chars (whitespace ignored).

- Samples: 2000
- Characters evaluated: 2729927
- Overall accuracy: 0.9701
- High confusions: rust->c_family 2123 (1.8%), csharp->c_family 2078 (1.9%), encoding_base85->c_family 2032 (1.9%), go->c_family 1953 (1.6%), java->c_family 1928 (1.7%)

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_base58 | 113187 | 0.9855 |
| encoding_hex | 100450 | 0.9853 |
| encoding_base64 | 102144 | 0.9851 |
| encoding_base32 | 112696 | 0.9847 |
| csv | 115899 | 0.9846 |
| html | 126641 | 0.9834 |
| encoding_base85 | 109072 | 0.9809 |
| css | 126927 | 0.9780 |
| c_family | 116285 | 0.9763 |
| json | 98221 | 0.9754 |
| text | 107494 | 0.9753 |
| ruby | 112796 | 0.9702 |
| sql | 101411 | 0.9694 |
| yaml | 103316 | 0.9684 |
| dockerfile | 68651 | 0.9657 |
| go | 122831 | 0.9643 |
| powershell | 112293 | 0.9632 |
| csharp | 110944 | 0.9621 |
| visual_basic | 121482 | 0.9609 |
| shell | 78354 | 0.9594 |
| java | 112289 | 0.9579 |
| php | 99816 | 0.9546 |
| rust | 119779 | 0.9541 |
| python | 112614 | 0.9525 |
| javascript_typescript | 124335 | 0.9522 |
| __unknown__ | 0 | nan |

| Scenario | Coverage ≥50% | IoU ≥50% | Avg IoU | Coverage | Top misclassifications |
| --- | --- | --- | ---: | ---: | --- |
| Any non-wrapper | 954/2000 | 298/2000 | 0.25 | 52.2% | c_family (10.0%), csharp (9.8%), powershell (6.7%) |
| Correct label | 620/2000 | 569/2000 | 0.26 | 33.4% | — |

## Throughput Benchmarks

| Task | Device | Samples | Total Bytes | Throughput | Latency (s) | RSS Δ (MB) | Device Δ (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| throughput_1024 | cpu | 8 | 8192 | 3.55 KB/s | 2.25 | 610.45 | n/a |
| throughput_10240 | cpu | 8 | 81920 | 22.19 KB/s | 3.61 | 93.09 | n/a |
| throughput_102400 | cpu | 8 | 819200 | 55.56 KB/s | 14.40 | 203.51 | n/a |
| throughput_1048576 | cpu | 8 | 8388608 | 72.45 KB/s | 113.06 | 383.29 | n/a |
| throughput_1024 | cuda | 8 | 8192 | 10.30 KB/s | 0.78 | 1.88 | n/a |
| throughput_10240 | cuda | 8 | 81920 | 53.42 KB/s | 1.50 | 38.18 | n/a |
| throughput_102400 | cuda | 8 | 819200 | 82.77 KB/s | 9.67 | 19.38 | n/a |
| throughput_1048576 | cuda | 8 | 8388608 | 90.11 KB/s | 90.91 | 256.11 | n/a |

Report generated at 2025-11-26 20:18:23