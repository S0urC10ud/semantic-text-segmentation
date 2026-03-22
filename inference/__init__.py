"""Shared inference utilities for non-training entrypoints."""

from .backend import (
    FastExecutionRecord,
    FastInferenceEngine,
    FastInferenceFailure,
    available_backends,
    build_window_spans,
    format_auto_fallback_message,
    resolve_backend,
    window_weights,
)
from .mamba_cuda import has_cuda_mamba_kernel, selective_scan_cuda, selective_scan_inference

__all__ = [
    "FastExecutionRecord",
    "FastInferenceEngine",
    "FastInferenceFailure",
    "available_backends",
    "build_window_spans",
    "format_auto_fallback_message",
    "has_cuda_mamba_kernel",
    "resolve_backend",
    "selective_scan_cuda",
    "selective_scan_inference",
    "window_weights",
]
