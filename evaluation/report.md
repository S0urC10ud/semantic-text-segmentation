# Segmenter Evaluation Report

- Checkpoint: `../train/checkpoints/sweeps/g0lrlnry.msgpack`
- Model dim: 256
- Channels: 96, 128, 160, 192, 224, 256, 288, 320
- Chunk: 1024
- Batch size: 16
- Max samples per task: 2000
- Sample seed: 13
- Evaluation data root: `/home/s0urc10ud/text-segmentation/evaluation/data`
- Generated at: 2025-10-30T17:59:58

## Summary

| Task | Samples | Characters | Char Acc | Macro Recall |
| --- | ---: | ---: | ---: | ---: |
| markdown_mix | 2000 | 6755353 | 0.8979 | 0.9215 |
| needle_16_31 | 2000 | 2905216 | 0.9305 | 0.9315 |
| needle_32_63 | 2000 | 2939360 | 0.9165 | 0.9174 |
| needle_4_15 | 2000 | 2866923 | 0.9437 | 0.9423 |
| needle_64_plus | 2000 | 3003659 | 0.9001 | 0.8975 |
| pure_fragments | 2000 | 2878826 | 0.9443 | 0.9440 |
| sequence_pair | 2000 | 5773099 | 0.9119 | 0.9073 |
| sequence_triplet | 2000 | 8606853 | 0.9161 | 0.9125 |

## Task Details

### markdown_mix

Markdown-like text/code interleavings with optional fences.

- Samples: 2000
- Characters evaluated: 6755353
- Overall accuracy: 0.8979

| Label | Support | Accuracy |
| --- | ---: | ---: |
| text | 991389 | 0.6872 |
| encoding_hex | 410536 | 0.9708 |
| encoding_base32 | 345362 | 0.9662 |
| encoding_base64 | 280534 | 0.9598 |
| encoding_base58 | 266611 | 0.9674 |
| css | 256422 | 0.9584 |
| html | 249229 | 0.9204 |
| rust | 248532 | 0.9341 |
| go | 246803 | 0.9491 |
| visual_basic | 241358 | 0.9536 |
| javascript | 240030 | 0.8492 |
| encoding_base85 | 237830 | 0.9631 |
| csv | 222579 | 0.9436 |
| c_family | 209499 | 0.9011 |
| typescript | 205114 | 0.8474 |
| ruby | 203769 | 0.9372 |
| powershell | 198249 | 0.9491 |
| python | 194967 | 0.9229 |
| yaml | 194460 | 0.9256 |
| json | 192523 | 0.9412 |
| csharp | 191496 | 0.9121 |
| java | 179702 | 0.9195 |
| sql | 178549 | 0.9185 |
| shell | 172641 | 0.9232 |
| php | 154439 | 0.9078 |
| dockerfile | 138653 | 0.9059 |
| batchfile | 104077 | 0.9463 |
| __unknown__ | 0 | nan |

### needle_16_31

Host fragments with a foreign-language needle injection sized 16-31 bytes.

- Samples: 2000
- Characters evaluated: 2905216
- Overall accuracy: 0.9305

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_hex | 185922 | 0.9582 |
| encoding_base32 | 152672 | 0.8298 |
| encoding_base64 | 138352 | 0.9690 |
| yaml | 136721 | 0.9639 |
| encoding_base85 | 127042 | 0.8023 |
| css | 126444 | 0.9720 |
| csv | 125895 | 0.9785 |
| encoding_base58 | 119756 | 0.9449 |
| html | 113715 | 0.9502 |
| rust | 112432 | 0.9636 |
| javascript | 111762 | 0.8457 |
| c_family | 110542 | 0.9512 |
| go | 108283 | 0.9782 |
| visual_basic | 107508 | 0.9360 |
| typescript | 102959 | 0.8657 |
| python | 102921 | 0.9541 |
| sql | 100073 | 0.8929 |
| shell | 95423 | 0.9178 |
| text | 92652 | 0.9374 |
| powershell | 92179 | 0.9498 |
| java | 91477 | 0.9372 |
| csharp | 88634 | 0.9415 |
| php | 82551 | 0.9458 |
| ruby | 79438 | 0.9565 |
| json | 70251 | 0.9358 |
| dockerfile | 66172 | 0.9037 |
| batchfile | 63440 | 0.9692 |
| __unknown__ | 0 | nan |

### needle_32_63

Host fragments with a foreign-language needle injection sized 32-63 bytes.

- Samples: 2000
- Characters evaluated: 2939360
- Overall accuracy: 0.9165

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_hex | 206762 | 0.9674 |
| encoding_base32 | 165536 | 0.8192 |
| python | 127528 | 0.9109 |
| text | 125199 | 0.9145 |
| encoding_base58 | 122139 | 0.9419 |
| encoding_base85 | 121971 | 0.7945 |
| css | 118847 | 0.9559 |
| encoding_base64 | 118752 | 0.9533 |
| rust | 118585 | 0.9542 |
| html | 118286 | 0.9045 |
| csharp | 115994 | 0.9125 |
| visual_basic | 107480 | 0.9369 |
| c_family | 106838 | 0.9118 |
| csv | 105648 | 0.9521 |
| ruby | 104275 | 0.9348 |
| yaml | 102560 | 0.9206 |
| powershell | 102043 | 0.9276 |
| sql | 101878 | 0.8918 |
| java | 99252 | 0.9167 |
| go | 97125 | 0.9389 |
| javascript | 96252 | 0.8650 |
| typescript | 91267 | 0.8893 |
| php | 84601 | 0.9235 |
| dockerfile | 78032 | 0.9363 |
| json | 77728 | 0.9500 |
| shell | 63323 | 0.9132 |
| batchfile | 61459 | 0.9324 |
| __unknown__ | 0 | nan |

### needle_4_15

Host fragments with a foreign-language needle injection sized 4-15 bytes.

- Samples: 2000
- Characters evaluated: 2866923
- Overall accuracy: 0.9437

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_hex | 191282 | 0.9832 |
| encoding_base58 | 151369 | 0.9684 |
| encoding_base32 | 142008 | 0.8431 |
| rust | 126757 | 0.9752 |
| encoding_base64 | 122800 | 0.9745 |
| python | 117561 | 0.9644 |
| csv | 115090 | 0.9832 |
| css | 113886 | 0.9550 |
| java | 111584 | 0.9458 |
| csharp | 108121 | 0.9320 |
| go | 104485 | 0.9673 |
| javascript | 102695 | 0.9333 |
| ruby | 102259 | 0.9651 |
| sql | 99909 | 0.9067 |
| text | 99595 | 0.9249 |
| visual_basic | 97621 | 0.9430 |
| html | 97620 | 0.9647 |
| shell | 96279 | 0.9292 |
| powershell | 95415 | 0.9541 |
| yaml | 92265 | 0.9761 |
| encoding_base85 | 91663 | 0.8071 |
| typescript | 90801 | 0.9263 |
| c_family | 90289 | 0.9712 |
| json | 89405 | 0.9178 |
| php | 84995 | 0.9286 |
| dockerfile | 77479 | 0.9327 |
| batchfile | 53690 | 0.9688 |
| __unknown__ | 0 | nan |

### needle_64_plus

Host fragments with a foreign-language needle injection sized 64-∞ bytes.

- Samples: 2000
- Characters evaluated: 3003659
- Overall accuracy: 0.9001

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_hex | 212510 | 0.9620 |
| encoding_base32 | 151496 | 0.8204 |
| css | 140829 | 0.9414 |
| encoding_base64 | 137384 | 0.9429 |
| html | 129821 | 0.9206 |
| encoding_base58 | 128859 | 0.9147 |
| csv | 127077 | 0.9371 |
| sql | 126041 | 0.8819 |
| java | 121645 | 0.9019 |
| python | 113302 | 0.8905 |
| php | 110038 | 0.9156 |
| powershell | 107432 | 0.9058 |
| typescript | 107361 | 0.8404 |
| rust | 106560 | 0.9113 |
| json | 106315 | 0.8983 |
| text | 103821 | 0.8829 |
| encoding_base85 | 101887 | 0.7901 |
| yaml | 99076 | 0.9243 |
| c_family | 98909 | 0.8599 |
| ruby | 95914 | 0.9181 |
| visual_basic | 95329 | 0.9256 |
| go | 94589 | 0.9220 |
| javascript | 92894 | 0.8230 |
| shell | 87405 | 0.9001 |
| csharp | 77442 | 0.8962 |
| dockerfile | 73944 | 0.8872 |
| batchfile | 55779 | 0.9193 |
| __unknown__ | 0 | nan |

### pure_fragments

Single-label validation fragments (baseline accuracy).

- Samples: 2000
- Characters evaluated: 2878826
- Overall accuracy: 0.9443

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_hex | 194994 | 0.9807 |
| encoding_base58 | 157072 | 0.9640 |
| encoding_base64 | 148540 | 0.9827 |
| encoding_base32 | 145424 | 0.8400 |
| rust | 131092 | 0.9896 |
| encoding_base85 | 126756 | 0.8121 |
| go | 121580 | 0.9803 |
| html | 116699 | 0.9333 |
| php | 115726 | 0.9433 |
| css | 113282 | 0.9886 |
| python | 109503 | 0.9713 |
| sql | 105383 | 0.9161 |
| c_family | 100165 | 0.9512 |
| csv | 98938 | 0.9880 |
| javascript | 98794 | 0.9181 |
| text | 98172 | 0.8939 |
| powershell | 95777 | 0.9812 |
| typescript | 94948 | 0.9454 |
| ruby | 89603 | 0.9731 |
| csharp | 83833 | 0.9005 |
| yaml | 83035 | 0.9550 |
| shell | 83033 | 0.9264 |
| json | 82287 | 0.9347 |
| java | 81060 | 0.9448 |
| visual_basic | 80920 | 0.9720 |
| dockerfile | 73276 | 0.9662 |
| batchfile | 48934 | 0.9345 |
| __unknown__ | 0 | nan |

### sequence_pair

Two-language back-to-back sequences <A><B>.

- Samples: 2000
- Characters evaluated: 5773099
- Overall accuracy: 0.9119

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_hex | 379296 | 0.9634 |
| encoding_base32 | 341964 | 0.9259 |
| encoding_base58 | 302142 | 0.9529 |
| encoding_base64 | 259898 | 0.9563 |
| encoding_base85 | 254649 | 0.9099 |
| csv | 244669 | 0.9307 |
| html | 226259 | 0.9183 |
| rust | 219172 | 0.9216 |
| java | 214096 | 0.8952 |
| python | 213014 | 0.9183 |
| text | 211187 | 0.9220 |
| javascript | 209486 | 0.8710 |
| c_family | 207841 | 0.9059 |
| go | 205831 | 0.9300 |
| css | 200259 | 0.9305 |
| csharp | 198973 | 0.8787 |
| php | 196654 | 0.9033 |
| sql | 194918 | 0.8896 |
| ruby | 184420 | 0.8918 |
| json | 184143 | 0.8596 |
| yaml | 183272 | 0.9009 |
| visual_basic | 179947 | 0.8980 |
| powershell | 179467 | 0.9092 |
| typescript | 178374 | 0.8151 |
| shell | 135934 | 0.8730 |
| batchfile | 135558 | 0.9351 |
| dockerfile | 131676 | 0.8897 |
| __unknown__ | 0 | nan |

### sequence_triplet

Three-language back-to-back sequences <A><B><C>.

- Samples: 2000
- Characters evaluated: 8606853
- Overall accuracy: 0.9161

| Label | Support | Accuracy |
| --- | ---: | ---: |
| encoding_hex | 588800 | 0.9716 |
| encoding_base32 | 468942 | 0.9436 |
| encoding_base58 | 414771 | 0.9617 |
| encoding_base64 | 395292 | 0.9528 |
| javascript | 341856 | 0.8435 |
| csv | 330438 | 0.9551 |
| encoding_base85 | 327612 | 0.9344 |
| css | 323543 | 0.9425 |
| go | 317397 | 0.9255 |
| rust | 316933 | 0.9340 |
| powershell | 312140 | 0.9116 |
| html | 310872 | 0.9182 |
| c_family | 309513 | 0.8749 |
| sql | 309030 | 0.8958 |
| php | 306059 | 0.8923 |
| typescript | 304250 | 0.8027 |
| python | 303841 | 0.9051 |
| text | 297228 | 0.9422 |
| csharp | 290576 | 0.9086 |
| java | 288369 | 0.8953 |
| visual_basic | 282844 | 0.9216 |
| yaml | 282139 | 0.8966 |
| json | 277842 | 0.9169 |
| ruby | 267534 | 0.8800 |
| shell | 254857 | 0.8662 |
| dockerfile | 212576 | 0.8994 |
| batchfile | 171599 | 0.9448 |
| __unknown__ | 0 | nan |

## Throughput Benchmarks

| Task | Device | Samples | Total Bytes | Throughput | Latency (s) | RSS Δ (MB) | Device Δ (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| throughput_1024 | cuda | 8 | 8192 | 30.51 KB/s | 0.26 | n/a | n/a |
| throughput_10240 | cuda | 8 | 81920 | 78.16 KB/s | 1.02 | n/a | n/a |
| throughput_102400 | cuda | 8 | 819200 | 100.18 KB/s | 7.99 | n/a | n/a |
| throughput_1048576 | cuda | 8 | 8388608 | 101.19 KB/s | 80.95 | n/a | n/a |

Report generated at 2025-10-31 08:31:37