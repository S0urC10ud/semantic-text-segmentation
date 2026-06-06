from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Options:
    """Post-processing options, mirroring the interactive viewer's controls.

    Attributes:
        other_threshold: characters whose maximum class probability is below this
            value are routed to the open-set ``other`` label. Set to 0 to disable.
        min_run_chars: runs shorter than this (in characters) are absorbed into a
            neighbouring segment.
        boundary_snap_max_shift: max characters a segment boundary may be nudged
            to land on a nearby delimiter symbol or whitespace.
        paired_delimiter_max_shift: max characters each edge of a wrapped run may be
            nudged so it sits inside a matching delimiter pair.
        whitespace_relabel: relabel boundary whitespace to its host segment.
        confidence_gating: enable routing of low-confidence characters to ``other``.
        boundary_snap: enable boundary snapping to nearby delimiters/whitespace.
        paired_delimiter_fill: enable matching-delimiter-pair run refinement
            (e.g. snap a value inside ``"..."`` so the quotes stay with the host).
        min_run_normalize: enable minimum-run normalisation.
    """

    other_threshold: float = 0.30
    min_run_chars: int = 3
    boundary_snap_max_shift: int = 2
    paired_delimiter_max_shift: int = 2
    whitespace_relabel: bool = True
    confidence_gating: bool = True
    boundary_snap: bool = True
    paired_delimiter_fill: bool = True
    min_run_normalize: bool = True
