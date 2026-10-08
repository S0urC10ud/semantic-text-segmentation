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
| `confidence` | Mean of the assigned label’s character confidence scores within the segment |
| `text` | Matched substring |

The end offset is exclusive. Keep segments in input order when drawing a composition bar: repeated regions of the same type may be separated by other content.

## Character details

A `Segmentation` includes three arrays:

- `char_probs`: the model’s probabilities before label post-processing, with shape `(len(text), 35)`. Columns follow `result.labels`.
- `char_labels`: the final labels after the enabled post-processing steps. These labels also define the segments.
- `char_confidence`: the model probability of each character’s final assigned label. If post-processing changes the label, this score can differ from the highest probability in that row.

The auxiliary `other` label has no probability column. Its confidence is a derived score: `1 - max(char_probs[i])`. Segment confidence averages `char_confidence` across the segment. For example, 0.92 means an average label score of 92%; it is not a measured 92% chance that the whole segment is correct.

Confidence reflects the model’s content-type prediction. Security verdicts require separate analysis.

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

This maps a string decoded from UTF-8 **without normalization** back to its bytes. For UTF-16, a BOM, replacement decoding, or normalized line endings, retain the decoding map from the original byte stream.

The browser demo normalizes line endings and maps unsupported characters to placeholders. Its composition bar reports offsets in that normalized text. Mapping these offsets to an uploaded file requires tracking the normalization steps.
