"""Active-learning utilities for boundary refinement."""

from .acquisition import CandidateSpan, select_candidate_spans
from .label_store import LabelStore, StoredRefinement
from .oracle import BoundarySnippet, OracleSegment, StubOracle


def run_one_round(*args, **kwargs):
    from .round import run_one_round as _run_one_round

    return _run_one_round(*args, **kwargs)

__all__ = [
    "BoundarySnippet",
    "CandidateSpan",
    "LabelStore",
    "OracleSegment",
    "StoredRefinement",
    "StubOracle",
    "run_one_round",
    "select_candidate_spans",
]
