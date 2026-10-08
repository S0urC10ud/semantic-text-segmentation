---
title: Understanding the Output
description: Segment boundaries, confidence values, and byte offsets for downstream inspection.
---
## Segments

Each `Segment` exposes:

| Field | Meaning |
| --- | --- |
| `start`, `end` | Character range in the input string: `text[start:end]` |
| `label` | Predicted textual content type |
| `confidence` | Segment confidence score |
| `text` | Matched substring |

The end offset is exclusive. Keep segments in input order when drawing a composition bar: repeated regions of the same type may be separated by other content.

## Character details

A `Segmentation` also includes `char_labels`, `char_confidence`, and `char_probs`. The probability array has shape `(len(text), 35)`; columns follow `result.labels`. These are raw model distributions, before post-processing. The auxiliary `other` label has no probability column.

Confidence reflects the model’s content-type prediction. It is not a maliciousness score, and should not be presented as one.

## Convert UTF-8 character ranges to bytes

Python offsets count characters. A platform displaying byte offsets needs an encoding-aware mapping:

```python
byte_offsets = [0]
for character in text:
    byte_offsets.append(byte_offsets[-1] + len(character.encode('utf-8')))

for segment in result.segments:
    start_byte = byte_offsets[segment.start]
    end_byte = byte_offsets[segment.end]
    print(segment.label, start_byte, end_byte)
```

This maps a string decoded from UTF-8 **without normalization** back to its bytes. For UTF-16, a BOM, replacement decoding, or normalized line endings, retain the decoding map from the original byte stream instead.

The browser demo normalizes line endings and maps unsupported characters to placeholders. Its composition bar reports offsets in that normalized text, not the original uploaded file.
