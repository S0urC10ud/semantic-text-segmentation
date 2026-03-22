from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency in some environments
    psutil = None  # type: ignore[assignment]

try:  # pragma: no cover - availability depends on the local JAX build
    from jax.experimental import pallas as _pallas  # noqa: F401
    from jax.experimental.pallas import triton as _pallas_triton  # noqa: F401
    _PALLAS_AVAILABLE = True
except Exception:  # pragma: no cover - availability depends on the local JAX build
    _PALLAS_AVAILABLE = False


LogFn = Callable[[str], None]
SanitizeBytesFn = Callable[[np.ndarray], np.ndarray]
SanitizeTokensFn = Callable[[np.ndarray], np.ndarray]
ApplyTokensFn = Callable[[jnp.ndarray], jnp.ndarray]


def resolve_backend(device_name: Optional[str]) -> Optional[str]:
    if not device_name or str(device_name).lower() in {"auto", "default"}:
        return None
    lower = str(device_name).lower()
    if lower in {"cpu", "gpu", "tpu"}:
        return lower
    if lower == "cuda":
        return "gpu"
    raise ValueError(f"Unknown device name '{device_name}'. Use cpu/gpu/cuda/auto.")


def available_backends() -> set[str]:
    platforms: set[str] = set()
    try:
        for dev in jax.devices():
            platforms.add(str(dev.platform))
    except Exception:
        pass
    for candidate in ("cpu", "gpu", "tpu"):
        try:
            devices = jax.devices(candidate)
        except Exception:
            continue
        if devices:
            platforms.add(str(devices[0].platform))
    return platforms


def window_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(length, 0),), dtype=np.float32)
    positions = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
    sigma = 0.5
    weights = np.exp(-0.5 * (positions / sigma) ** 2)
    return weights.astype(np.float32)


def build_window_spans(length: int, chunk_size: int) -> List[Tuple[int, int]]:
    n = int(length)
    if n <= 0:
        return []
    win = max(64, int(chunk_size))
    stride = max(1, win // 2)
    start_positions = list(range(0, max(1, n - win + 1), stride))
    if not start_positions:
        start_positions = [0]
    tail_start = max(0, n - win)
    if start_positions[-1] + win < n and tail_start not in start_positions:
        start_positions.append(tail_start)
    spans: List[Tuple[int, int]] = []
    seen = set()
    for start in start_positions:
        if start in seen:
            continue
        seen.add(start)
        end = min(start + win, n)
        spans.append((int(start), int(end)))
    if not spans:
        spans = [(0, n)]
    return spans


def format_auto_fallback_message(
    *,
    from_path: str,
    to_path: str,
    trigger: str,
    reason: str,
) -> str:
    return (
        "⚠️  FAST INFERENCE AUTO FALLBACK: "
        f"{from_path} -> {to_path} | trigger={trigger} | reason={reason}"
    )


@dataclass(frozen=True)
class FastExecutionRecord:
    mode: str
    path: str
    actual_batch_size: int
    padded_length: int
    max_sequence_length: int


class FastInferenceFailure(RuntimeError):
    def __init__(
        self,
        *,
        source_path: str,
        trigger: str,
        reason: str,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(reason)
        self.source_path = str(source_path)
        self.trigger = str(trigger)
        self.reason = str(reason)
        self.cause = cause


class FastInferenceEngine:
    """Shared inference engine for evaluation and viewer-style inference.

    The engine is inference-only. It keeps the old, file-local implementations
    available as the surrounding wrapper's `legacy` path while providing:

    - `fast_full`: dynamic padding to the batch-local maximum length
    - `fast_stream`: tail-safe windowed inference with weighted overlap merge
    - `auto`: optional CUDA Pallas probe for Mamba, then `fast_full`,
      then `fast_stream`, with loud fallback logging
    """

    def __init__(
        self,
        *,
        apply_tokens: ApplyTokensFn,
        num_classes: int,
        pad_token_id: int,
        chunk_size: int,
        batch_size: int,
        sanitize_bytes: SanitizeBytesFn,
        sanitize_tokens: SanitizeTokensFn,
        arch: str = "unet1d",
        inference_backend: str = "auto",
        actual_backend: Optional[str] = None,
        log_fn: Optional[LogFn] = None,
        model_dim: Optional[int] = None,
        channels: Sequence[int] = (),
        mamba_layers: int = 6,
        mamba_d_state: int = 8,
        mamba_expand: int = 1,
        mamba_bidirectional: bool = True,
        cuda_kernel_available: bool = False,
        full_memory_budget_bytes: Optional[int] = None,
        execution_platform_override: Optional[str] = None,
    ) -> None:
        mode = str(inference_backend).lower().strip()
        if mode not in {"auto", "fast"}:
            raise ValueError(
                f"FastInferenceEngine only supports 'auto' or 'fast', got '{inference_backend}'."
            )
        self.apply_tokens = apply_tokens
        self.num_classes = int(num_classes)
        self.pad_token_id = int(pad_token_id)
        self.chunk_size = max(64, int(chunk_size))
        self.batch_size = max(1, int(batch_size))
        self.sanitize_bytes = sanitize_bytes
        self.sanitize_tokens = sanitize_tokens
        self.arch = str(arch).lower().strip()
        self.inference_backend = mode
        self.actual_backend = str(actual_backend).lower().strip() if actual_backend else None
        self.execution_platform = (
            str(execution_platform_override).lower().strip()
            if execution_platform_override
            else (self.actual_backend or str(jax.default_backend()).lower().strip())
        )
        self.log_fn = log_fn or (lambda message: print(message, flush=True))
        self.model_dim = int(model_dim) if model_dim is not None else None
        self.channels = tuple(int(ch) for ch in channels)
        self.mamba_layers = int(mamba_layers)
        self.mamba_d_state = int(mamba_d_state)
        self.mamba_expand = int(mamba_expand)
        self.mamba_bidirectional = bool(mamba_bidirectional)
        self.cuda_kernel_available = bool(cuda_kernel_available)
        self.full_memory_budget_bytes = (
            int(full_memory_budget_bytes)
            if full_memory_budget_bytes is not None
            else int(self._default_full_memory_budget_bytes())
        )
        self._weight_cache: Dict[int, np.ndarray] = {}
        self._emitted_fallbacks: set[Tuple[str, str, str, str]] = set()
        self.execution_history: List[FastExecutionRecord] = []

    def _bucket_padded_length(self, length: int) -> int:
        """Round sequence lengths into a small set of GPU-friendly buckets.

        JAX/XLA specializes compiled programs by shape. Accuracy evaluation often
        feeds many one-off sample lengths into the fast path, which can trigger
        large compile churn on GPU. Bucketing lengths keeps the number of unique
        shapes small while preserving the existing pad-and-slice semantics.
        """
        n = int(length)
        if n <= 0:
            return 0
        if self.execution_platform != "gpu":
            return n
        bucket = 256
        return int(((n + bucket - 1) // bucket) * bucket)

    def _is_mamba_arch(self) -> bool:
        return self.arch in {"mamba", "mamba1d", "bimamba", "ssm"}

    def _default_full_memory_budget_bytes(self) -> int:
        platform = self.execution_platform or "cpu"
        if platform == "cpu":
            available = None
            if psutil is not None:
                try:
                    available = int(psutil.virtual_memory().available)
                except Exception:
                    available = None
            if available is None:
                budget = 512 * 1024 * 1024
            else:
                budget = max(256 * 1024 * 1024, min(int(available * 0.15), 4 * 1024 * 1024 * 1024))
            if self._is_mamba_arch():
                budget = max(192 * 1024 * 1024, int(budget * 0.5))
            return int(budget)

        if platform == "gpu":
            budget = self._device_memory_budget_from_jax()
            if budget is None:
                budget = 2 * 1024 * 1024 * 1024
            if self._is_mamba_arch():
                budget = max(768 * 1024 * 1024, int(budget * 0.75))
            return int(budget)

        return 1024 * 1024 * 1024

    def _device_memory_budget_from_jax(self) -> Optional[int]:
        try:
            if self.actual_backend:
                devices = jax.devices(self.actual_backend)
            else:
                devices = jax.devices()
        except Exception:
            return None
        if not devices:
            return None
        device = devices[0]
        try:
            stats = device.memory_stats()
        except Exception:
            stats = None
        if not isinstance(stats, dict):
            return None
        for key in ("bytes_limit", "bytes_reserved_limit", "device_memory_size", "total_bytes"):
            value = stats.get(key)
            if value:
                try:
                    return max(512 * 1024 * 1024, int(float(value) * 0.2))
                except Exception:
                    continue
        return None

    def _record_execution(
        self,
        *,
        mode: str,
        path: str,
        actual_batch_size: int,
        padded_length: int,
        max_sequence_length: int,
    ) -> None:
        self.execution_history.append(
            FastExecutionRecord(
                mode=str(mode),
                path=str(path),
                actual_batch_size=int(actual_batch_size),
                padded_length=int(padded_length),
                max_sequence_length=int(max_sequence_length),
            )
        )

    def _emit_auto_fallback(
        self,
        *,
        from_path: str,
        to_path: str,
        trigger: str,
        reason: str,
    ) -> None:
        key = (str(from_path), str(to_path), str(trigger), str(reason))
        if key in self._emitted_fallbacks:
            return
        self._emitted_fallbacks.add(key)
        self.log_fn(
            format_auto_fallback_message(
                from_path=from_path,
                to_path=to_path,
                trigger=trigger,
                reason=reason,
            )
        )

    def _window_weights_cached(self, length: int) -> np.ndarray:
        cached = self._weight_cache.get(length)
        if cached is None:
            cached = window_weights(length)
            self._weight_cache[length] = cached
        return cached

    def _supports_pallas_candidate(self) -> bool:
        return self._is_mamba_arch() and self.execution_platform == "gpu"

    def _pallas_kernel_ready(self) -> Tuple[bool, str]:
        if not self._supports_pallas_candidate():
            return False, f"unsupported_device_backend:{self.execution_platform or 'unknown'}"
        if self.cuda_kernel_available:
            return True, "available"
        if not _PALLAS_AVAILABLE:
            return False, "missing_kernel_support:pallas_unavailable"
        return False, "missing_kernel_support:no_registered_mamba_kernel"

    def _estimate_batch_bytes(self, lengths: Sequence[int]) -> int:
        if not lengths:
            return 0
        batch = len(lengths)
        padded = self._bucket_padded_length(max(int(length) for length in lengths))
        if self._is_mamba_arch():
            d_model = max(int(self.model_dim or 256), 1)
            d_inner = max(d_model * max(self.mamba_expand, 1), 1)
            d_state = max(self.mamba_d_state, 1)
            directional = 2 if self.mamba_bidirectional else 1
            layers = max(self.mamba_layers, 1)
            scan_buffers = batch * padded * d_inner * d_state * 4 * directional
            activations = batch * padded * (d_model + d_inner + self.num_classes) * 4
            return int((scan_buffers * 3) + (activations * (layers + 4)))

        channel_total = sum(self.channels) if self.channels else max(int(self.model_dim or 256), 1)
        activations = batch * padded * (channel_total + max(int(self.model_dim or channel_total), 1) + self.num_classes) * 4
        return int(activations * 8)

    @staticmethod
    def _looks_like_oom(exc: BaseException) -> bool:
        message = str(exc).lower()
        return (
            "out of memory" in message
            or "resource exhausted" in message
            or "oom" in message
            or "memory exhausted" in message
        )

    def _prepare_tokens(
        self,
        arrays: Sequence[np.ndarray],
        *,
        path: str,
        mode: str,
    ) -> np.ndarray:
        actual = len(arrays)
        max_length = max((int(arr.shape[0]) for arr in arrays), default=0)
        padded_length = self._bucket_padded_length(max_length)
        tokens = np.full((actual, padded_length), self.pad_token_id, dtype=np.int32)
        for row, arr in enumerate(arrays):
            length = int(arr.shape[0])
            if length > 0:
                tokens[row, :length] = np.asarray(arr[:length], dtype=np.int32)
        tokens = self.sanitize_tokens(tokens)
        self._record_execution(
            mode=mode,
            path=path,
            actual_batch_size=actual,
            padded_length=padded_length,
            max_sequence_length=max_length,
        )
        return tokens

    def _apply_and_softmax(
        self,
        arrays: Sequence[np.ndarray],
        *,
        path: str,
        mode: str,
    ) -> np.ndarray:
        tokens = self._prepare_tokens(arrays, path=path, mode=mode)
        if tokens.size == 0:
            return np.zeros((tokens.shape[0], tokens.shape[1], self.num_classes), dtype=np.float32)
        logits = self.apply_tokens(jnp.asarray(tokens, dtype=jnp.int32))
        probs = jax.nn.softmax(logits, axis=-1)
        return np.asarray(jax.device_get(probs), dtype=np.float32)

    def _apply_argmax(
        self,
        arrays: Sequence[np.ndarray],
        *,
        path: str,
        mode: str,
    ) -> np.ndarray:
        tokens = self._prepare_tokens(arrays, path=path, mode=mode)
        if tokens.size == 0:
            return np.zeros((tokens.shape[0], tokens.shape[1]), dtype=np.uint8)
        logits = self.apply_tokens(jnp.asarray(tokens, dtype=jnp.int32))
        labels = jnp.argmax(logits, axis=-1).astype(jnp.uint8)
        return np.asarray(jax.device_get(labels), dtype=np.uint8)

    def predict_logits(self, token_batch: np.ndarray) -> np.ndarray:
        arr = np.asarray(token_batch, dtype=np.int32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"predict_logits expects rank-1 or rank-2 input, got {arr.shape}")
        arr = self.sanitize_tokens(arr.copy())
        self._record_execution(
            mode="predict_logits",
            path="fast_full",
            actual_batch_size=int(arr.shape[0]),
            padded_length=int(arr.shape[1]),
            max_sequence_length=int(arr.shape[1]),
        )
        logits = self.apply_tokens(jnp.asarray(arr, dtype=jnp.int32))
        return np.asarray(jax.device_get(logits), dtype=np.float32)

    def segment_bytes(
        self,
        byte_arr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        labels, probs, spans = self.segment_bytes_batch([byte_arr])
        if not labels:
            empty_labels = np.zeros((0,), dtype=np.uint8)
            empty_probs = np.zeros((0, self.num_classes), dtype=np.float32)
            return empty_labels, empty_probs, []
        return labels[0], probs[0], spans[0]

    def segment_bytes_labels_only(
        self,
        byte_arr: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        labels, spans = self.segment_bytes_batch_labels_only([byte_arr])
        if not labels:
            return np.zeros((0,), dtype=np.uint8), []
        return labels[0], spans[0]

    def segment_bytes_batch(
        self,
        byte_arrays: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[Tuple[int, int]]]]:
        arrays = [self.sanitize_bytes(np.asarray(arr, dtype=np.uint8)) for arr in byte_arrays]
        out_labels: List[np.ndarray] = []
        out_probs: List[np.ndarray] = []
        out_spans: List[List[Tuple[int, int]]] = []
        if not arrays:
            return out_labels, out_probs, out_spans

        idx = 0
        while idx < len(arrays):
            take, path = self._choose_text_batch(arrays, idx)
            batch = arrays[idx:idx + take]
            try:
                if path == "fast_full":
                    batch_labels, batch_probs, batch_spans = self._segment_full_batch(batch)
                else:
                    batch_labels, batch_probs, batch_spans = self._segment_stream_batch(batch)
            except Exception as exc:
                if path == "fast_full":
                    trigger = "oom_or_size" if self._looks_like_oom(exc) else "runtime_error"
                    reason = str(exc)
                    if self.inference_backend == "auto":
                        self._emit_auto_fallback(
                            from_path="fast_full",
                            to_path="fast_stream",
                            trigger=trigger,
                            reason=reason,
                        )
                        try:
                            batch_labels, batch_probs, batch_spans = self._segment_stream_batch(batch)
                        except Exception as stream_exc:
                            final_trigger = "oom_or_size" if self._looks_like_oom(stream_exc) else "runtime_error"
                            raise FastInferenceFailure(
                                source_path="fast_stream",
                                trigger=final_trigger,
                                reason=str(stream_exc),
                                cause=stream_exc,
                            ) from stream_exc
                    else:
                        raise FastInferenceFailure(
                            source_path="fast_full",
                            trigger=trigger,
                            reason=reason,
                            cause=exc,
                        ) from exc
                else:
                    trigger = "oom_or_size" if self._looks_like_oom(exc) else "runtime_error"
                    raise FastInferenceFailure(
                        source_path="fast_stream",
                        trigger=trigger,
                        reason=str(exc),
                        cause=exc,
                    ) from exc
            out_labels.extend(batch_labels)
            out_probs.extend(batch_probs)
            out_spans.extend(batch_spans)
            idx += take
        return out_labels, out_probs, out_spans

    def segment_bytes_batch_labels_only(
        self,
        byte_arrays: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
        arrays = [self.sanitize_bytes(np.asarray(arr, dtype=np.uint8)) for arr in byte_arrays]
        out_labels: List[np.ndarray] = []
        out_spans: List[List[Tuple[int, int]]] = []
        if not arrays:
            return out_labels, out_spans

        idx = 0
        while idx < len(arrays):
            take, path = self._choose_text_batch(arrays, idx)
            batch = arrays[idx:idx + take]
            try:
                if path == "fast_full":
                    batch_labels, batch_spans = self._segment_full_batch_labels_only(batch)
                else:
                    batch_labels, batch_spans = self._segment_stream_batch_labels_only(batch)
            except Exception as exc:
                if path == "fast_full":
                    trigger = "oom_or_size" if self._looks_like_oom(exc) else "runtime_error"
                    reason = str(exc)
                    if self.inference_backend == "auto":
                        self._emit_auto_fallback(
                            from_path="fast_full",
                            to_path="fast_stream",
                            trigger=trigger,
                            reason=reason,
                        )
                        try:
                            batch_labels, batch_spans = self._segment_stream_batch_labels_only(batch)
                        except Exception as stream_exc:
                            final_trigger = "oom_or_size" if self._looks_like_oom(stream_exc) else "runtime_error"
                            raise FastInferenceFailure(
                                source_path="fast_stream",
                                trigger=final_trigger,
                                reason=str(stream_exc),
                                cause=stream_exc,
                            ) from stream_exc
                    else:
                        raise FastInferenceFailure(
                            source_path="fast_full",
                            trigger=trigger,
                            reason=reason,
                            cause=exc,
                        ) from exc
                else:
                    trigger = "oom_or_size" if self._looks_like_oom(exc) else "runtime_error"
                    raise FastInferenceFailure(
                        source_path="fast_stream",
                        trigger=trigger,
                        reason=str(exc),
                        cause=exc,
                    ) from exc
            out_labels.extend(batch_labels)
            out_spans.extend(batch_spans)
            idx += take
        return out_labels, out_spans

    def _choose_text_batch(
        self,
        arrays: Sequence[np.ndarray],
        start_idx: int,
    ) -> Tuple[int, str]:
        max_take = min(self.batch_size, len(arrays) - start_idx)
        if self.inference_backend == "auto" and self._supports_pallas_candidate():
            pallas_ready, reason = self._pallas_kernel_ready()
            if not pallas_ready:
                trigger, detail = reason.split(":", 1) if ":" in reason else ("missing_kernel_support", reason)
                self._emit_auto_fallback(
                    from_path="fast_pallas",
                    to_path="fast_full",
                    trigger=trigger,
                    reason=detail,
                )

        candidate = max_take
        while candidate > 0:
            batch_lengths = [int(arr.shape[0]) for arr in arrays[start_idx:start_idx + candidate]]
            estimate = self._estimate_batch_bytes(batch_lengths)
            if estimate <= self.full_memory_budget_bytes:
                return candidate, "fast_full"
            if candidate == 1:
                if self.inference_backend == "auto":
                    reason = (
                        f"estimated_full_batch_bytes={estimate} exceeds budget={self.full_memory_budget_bytes}"
                    )
                    self._emit_auto_fallback(
                        from_path="fast_full",
                        to_path="fast_stream",
                        trigger="oom_or_size",
                        reason=reason,
                    )
                return 1, "fast_stream"
            candidate = candidate - 1 if candidate <= 4 else max(1, candidate // 2)
        return 1, "fast_stream"

    def _segment_full_batch(
        self,
        arrays: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[Tuple[int, int]]]]:
        probs = self._apply_and_softmax(arrays, path="fast_full", mode="segment_full")
        out_labels: List[np.ndarray] = []
        out_probs: List[np.ndarray] = []
        out_spans: List[List[Tuple[int, int]]] = []
        for row, arr in enumerate(arrays):
            length = int(arr.shape[0])
            row_probs = np.asarray(probs[row, :length], dtype=np.float32)
            out_probs.append(row_probs)
            out_labels.append(np.argmax(row_probs, axis=-1).astype(np.uint8) if length > 0 else np.zeros((0,), dtype=np.uint8))
            out_spans.append([(0, length)] if length > 0 else [])
        return out_labels, out_probs, out_spans

    def _segment_full_batch_labels_only(
        self,
        arrays: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
        labels_batch = self._apply_argmax(arrays, path="fast_full", mode="segment_full_labels")
        out_labels: List[np.ndarray] = []
        out_spans: List[List[Tuple[int, int]]] = []
        for row, arr in enumerate(arrays):
            length = int(arr.shape[0])
            out_labels.append(np.asarray(labels_batch[row, :length], dtype=np.uint8) if length > 0 else np.zeros((0,), dtype=np.uint8))
            out_spans.append([(0, length)] if length > 0 else [])
        return out_labels, out_spans

    def _segment_stream_batch(
        self,
        arrays: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[Tuple[int, int]]]]:
        probs_accum_list: List[np.ndarray] = [
            np.zeros((int(arr.shape[0]), self.num_classes), dtype=np.float32)
            for arr in arrays
        ]
        weight_accum_list: List[np.ndarray] = [
            np.zeros((int(arr.shape[0]),), dtype=np.float32)
            for arr in arrays
        ]
        spans_by_text: List[List[Tuple[int, int]]] = []
        window_refs: List[Tuple[int, int, int, np.ndarray]] = []

        for text_idx, arr in enumerate(arrays):
            spans = build_window_spans(int(arr.shape[0]), self.chunk_size)
            spans_by_text.append(spans)
            for start, end in spans:
                window_refs.append((text_idx, int(start), int(end), arr[start:end]))

        if not window_refs:
            empty_labels = [np.zeros((int(arr.shape[0]),), dtype=np.uint8) for arr in arrays]
            return empty_labels, probs_accum_list, spans_by_text

        win_idx = 0
        while win_idx < len(window_refs):
            take = min(self.batch_size, len(window_refs) - win_idx)
            while take > 1:
                batch_lengths = [int(window_refs[win_idx + offset][3].shape[0]) for offset in range(take)]
                if self._estimate_batch_bytes(batch_lengths) <= self.full_memory_budget_bytes:
                    break
                take = take - 1 if take <= 4 else max(1, take // 2)
            batch_refs = window_refs[win_idx:win_idx + take]
            batch_arrays = [ref[3] for ref in batch_refs]
            probs_batch = self._apply_and_softmax(batch_arrays, path="fast_stream", mode="segment_stream")
            for row, (text_idx, start, end, _) in enumerate(batch_refs):
                plen = int(end) - int(start)
                if plen <= 0:
                    continue
                weights = self._window_weights_cached(plen)
                window_probs = np.asarray(probs_batch[row, :plen], dtype=np.float32)
                probs_accum_list[text_idx][start:end] += window_probs * weights[:, None]
                weight_accum_list[text_idx][start:end] += weights
            win_idx += take

        out_labels: List[np.ndarray] = []
        out_probs: List[np.ndarray] = []
        for probs_accum, weight_accum in zip(probs_accum_list, weight_accum_list):
            if probs_accum.size == 0:
                out_probs.append(probs_accum)
                out_labels.append(np.zeros((0,), dtype=np.uint8))
                continue
            nonzero = weight_accum > 0
            if np.any(nonzero):
                probs_accum = probs_accum.copy()
                probs_accum[nonzero] /= weight_accum[nonzero, None]
                zero_mask = ~nonzero
                if np.any(zero_mask):
                    probs_accum[zero_mask] = 1.0 / self.num_classes
            else:
                probs_accum = np.full_like(probs_accum, 1.0 / self.num_classes, dtype=np.float32)
            out_probs.append(probs_accum)
            out_labels.append(np.argmax(probs_accum, axis=-1).astype(np.uint8))
        return out_labels, out_probs, spans_by_text

    def _segment_stream_batch_labels_only(
        self,
        arrays: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
        vote_accum_list: List[np.ndarray] = [
            np.zeros((int(arr.shape[0]), self.num_classes), dtype=np.float32)
            for arr in arrays
        ]
        spans_by_text: List[List[Tuple[int, int]]] = []
        window_refs: List[Tuple[int, int, int, np.ndarray]] = []

        for text_idx, arr in enumerate(arrays):
            spans = build_window_spans(int(arr.shape[0]), self.chunk_size)
            spans_by_text.append(spans)
            for start, end in spans:
                window_refs.append((text_idx, int(start), int(end), arr[start:end]))

        if not window_refs:
            empty_labels = [np.zeros((int(arr.shape[0]),), dtype=np.uint8) for arr in arrays]
            return empty_labels, spans_by_text

        win_idx = 0
        while win_idx < len(window_refs):
            take = min(self.batch_size, len(window_refs) - win_idx)
            while take > 1:
                batch_lengths = [int(window_refs[win_idx + offset][3].shape[0]) for offset in range(take)]
                if self._estimate_batch_bytes(batch_lengths) <= self.full_memory_budget_bytes:
                    break
                take = take - 1 if take <= 4 else max(1, take // 2)
            batch_refs = window_refs[win_idx:win_idx + take]
            batch_arrays = [ref[3] for ref in batch_refs]
            labels_batch = self._apply_argmax(batch_arrays, path="fast_stream", mode="segment_stream_labels")
            for row, (text_idx, start, end, _) in enumerate(batch_refs):
                plen = int(end) - int(start)
                if plen <= 0:
                    continue
                weights = self._window_weights_cached(plen)
                window_labels = np.asarray(labels_batch[row, :plen], dtype=np.int64)
                positions = np.arange(int(start), int(end), dtype=np.int64)
                np.add.at(vote_accum_list[text_idx], (positions, window_labels), weights)
            win_idx += take

        out_labels: List[np.ndarray] = []
        for vote_accum in vote_accum_list:
            if vote_accum.size == 0:
                out_labels.append(np.zeros((0,), dtype=np.uint8))
                continue
            out_labels.append(np.argmax(vote_accum, axis=-1).astype(np.uint8))
        return out_labels, spans_by_text
