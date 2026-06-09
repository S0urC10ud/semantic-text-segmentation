from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ._color import color_enabled, colorize


def _visible(s: str, limit: int = 60) -> str:
    """Make whitespace visible and clamp length for a one-line repr."""
    s = s.replace("\t", "  ").replace("\r", "").replace("\n", "⏎")
    return s if len(s) <= limit else s[: limit - 1] + "…"


@dataclass
class Segment:
    """A contiguous run of one content type over ``text[start:end]``.

    ``text`` is the matched substring; it powers the colourised ``repr`` shown in
    a terminal (a tinted background chip per content type).
    """

    start: int
    end: int
    label: str
    confidence: float
    text: str = ""

    def __repr__(self) -> str:
        if self.text and color_enabled():
            chip = colorize(f" {_visible(self.text)} ", self.label)
            return f"{chip} {self.label} {self.confidence:.2f}"
        return f"Segment({self.start}:{self.end} {self.label!r} conf={self.confidence:.2f})"


@dataclass
class Segmentation:
    """Result of segmenting a text.

    Attributes:
        text: the input text.
        segments: merged runs as :class:`Segment` objects.
        char_labels: per-character content-type label (length == len(text)).
        char_confidence: per-character confidence in [0, 1] (length == len(text)).
        char_probs: per-character probability distribution from the model, shape
            ``(len(text), len(labels))``, float32, each row summing to ~1. This is
            the *raw* model output (before post-processing relabelling); columns
            follow :attr:`labels`. The open-set ``other`` class is not a column —
            it is derived by confidence gating, so a character routed to ``other``
            still has its full distribution over the known classes here.
        labels: the class names, in ``char_probs`` column order (length == num classes).
    """

    text: str
    segments: List[Segment] = field(default_factory=list)
    char_labels: List[str] = field(default_factory=list)
    char_confidence: List[float] = field(default_factory=list)
    char_probs: Optional[np.ndarray] = None
    labels: List[str] = field(default_factory=list)

    def char_distribution(self, index: int) -> dict:
        """``{label: probability}`` for character ``index`` (convenience view)."""
        if self.char_probs is None or not self.labels:
            return {}
        return {lab: float(p) for lab, p in zip(self.labels, self.char_probs[index])}

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    def __repr__(self) -> str:
        head = ", ".join(repr(s) for s in self.segments[:4])
        more = "" if len(self.segments) <= 4 else f", … (+{len(self.segments) - 4})"
        return f"Segmentation(<{len(self.text)} chars>, {len(self.segments)} segments: [{head}{more}])"
