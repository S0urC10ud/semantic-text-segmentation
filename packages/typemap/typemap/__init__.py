"""typemap — fine-grained, character-level content-type segmentation.

Two entry points mirror the two models:

    >>> import typemap
    >>> result = typemap.fast("<html>...</html>")     # U-Net, piecewise-constant
    >>> result = typemap.precise("...")               # Mamba, long-context
    >>> for seg in result.segments:
    ...     print(seg.start, seg.end, seg.label, seg.confidence)
"""
from __future__ import annotations

from typing import Optional

from ._options import Options
from ._runtime import backend_info, run as _run
from ._segmentation import Segment, Segmentation

__all__ = ["fast", "precise", "Options", "Segment", "Segmentation", "backend_info", "__version__"]

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("typemap")
except (ImportError, PackageNotFoundError):  # not installed / running from source tree
    __version__ = "0.0.0"


def fast(text: str, options: Optional[Options] = None) -> Segmentation:
    """Fast, piecewise-constant segmentation using the U-Net model."""
    return _run("fast", text, options)


def precise(text: str, options: Optional[Options] = None) -> Segmentation:
    """Higher-quality, long-context segmentation using the Mamba model."""
    return _run("precise", text, options)
