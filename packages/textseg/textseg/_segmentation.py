from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Segment:
    """A contiguous run of one content type over ``text[start:end]``."""

    start: int
    end: int
    label: str
    confidence: float

    def __repr__(self) -> str:
        return f"Segment({self.start}:{self.end} {self.label!r} conf={self.confidence:.2f})"


@dataclass
class Segmentation:
    """Result of segmenting a text.

    Attributes:
        text: the input text.
        segments: merged runs as :class:`Segment` objects.
        char_labels: per-character content-type label (length == len(text)).
        char_confidence: per-character confidence in [0, 1] (length == len(text)).
    """

    text: str
    segments: List[Segment] = field(default_factory=list)
    char_labels: List[str] = field(default_factory=list)
    char_confidence: List[float] = field(default_factory=list)

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __repr__(self) -> str:
        head = ", ".join(repr(s) for s in self.segments[:4])
        more = "" if len(self.segments) <= 4 else f", … (+{len(self.segments) - 4})"
        return f"Segmentation(<{len(self.text)} chars>, {len(self.segments)} segments: [{head}{more}])"
