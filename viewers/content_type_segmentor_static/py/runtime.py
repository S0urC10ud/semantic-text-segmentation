from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


BYTE_VOCAB_SIZE = 256
PAD_BYTE_ID = 256
NUM_TOKEN_EMBEDDINGS = BYTE_VOCAB_SIZE + 1

_VISIBLE_ASCII_BYTES = tuple(range(0x20, 0x7F))
_WHITESPACE_BYTES = (0x09, 0x0A, 0x0D)
_CURRENCY_BYTE_ID = np.int32(0xA4)
_ALLOWED_MODEL_BYTE_VALUES = np.array(
    sorted(set(_VISIBLE_ASCII_BYTES) | set(_WHITESPACE_BYTES) | {int(_CURRENCY_BYTE_ID)}),
    dtype=np.int32,
)
_ALLOWED_MODEL_TOKEN_VALUES = np.array(
    sorted(set(_ALLOWED_MODEL_BYTE_VALUES.tolist()) | {int(PAD_BYTE_ID)}),
    dtype=np.int32,
)

_PLACEHOLDER_CHAR = "\u00A4"
_ALLOWED_TEXT_CHARS = {chr(b) for b in _VISIBLE_ASCII_BYTES}
_ALLOWED_TEXT_CHARS.update({" ", "\n", "\t", _PLACEHOLDER_CHAR})
_VISUAL_WHITESPACE_SET = frozenset((" ", "\t", "\n"))
_LAYER_NORM_EPS = 1e-6


_MODEL: Optional["NumpyMambaSegmentor"] = None


def _normalize_input_text(text: Optional[str]) -> str:
    if not text:
        return ""
    out_chars: List[str] = []
    for ch in text:
        if ch == "\r":
            ch = "\n"
        if ch in _ALLOWED_TEXT_CHARS:
            out_chars.append(ch)
        else:
            out_chars.append(_PLACEHOLDER_CHAR)
    return "".join(out_chars)


def _sanitize_model_bytes(arr: np.ndarray) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.uint8)
    if arr_np.size == 0:
        return arr_np
    invalid = ~np.isin(arr_np.astype(np.int32), _ALLOWED_MODEL_BYTE_VALUES)
    if np.any(invalid):
        arr_np = arr_np.copy()
        arr_np[invalid] = np.uint8(_CURRENCY_BYTE_ID)
    return arr_np


def _sanitize_model_tokens(arr: np.ndarray) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.int32)
    if arr_np.size == 0:
        return arr_np
    invalid = ~np.isin(arr_np, _ALLOWED_MODEL_TOKEN_VALUES)
    if np.any(invalid):
        arr_np = arr_np.copy()
        arr_np[invalid] = int(_CURRENCY_BYTE_ID)
    return arr_np


def _window_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((max(length, 0),), dtype=np.float32)
    positions = np.linspace(-1.0, 1.0, num=length, dtype=np.float32)
    sigma = 0.5
    weights = np.exp(-0.5 * (positions / sigma) ** 2)
    return weights.astype(np.float32)


def _build_window_spans(length: int, chunk_size: int, stride: Optional[int] = None) -> List[Tuple[int, int]]:
    n = int(length)
    if n <= 0:
        return []
    win = max(64, int(chunk_size))
    if n <= win:
        return [(0, n)]
    step = int(stride) if stride is not None and int(stride) > 0 else max(1, win // 2)
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + win, n)
        spans.append((int(start), int(end)))
        if end >= n:
            break
        start += step
    return spans


def _layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    normalized = centered / np.sqrt(var + np.float32(_LAYER_NORM_EPS))
    return normalized * scale[None, :] + bias[None, :]


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-abs_x))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    denom = np.maximum(np.sum(exp_x, axis=-1, keepdims=True), 1e-9)
    return exp_x / denom


def _depthwise_conv_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    k = int(kernel.shape[0])
    pad_left = max(0, (k - 1) // 2)
    pad_right = max(0, (k - 1) - pad_left)
    padded = np.pad(x, ((pad_left, pad_right), (0, 0)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=k, axis=0)
    if windows.ndim != 3:
        raise ValueError(f"Unexpected depthwise conv window shape: {windows.shape}")
    if windows.shape[1] != k:
        windows = np.moveaxis(windows, -1, 1)
    return np.einsum("tkc,kc->tc", windows, kernel, optimize=True) + bias[None, :]


def _selective_scan(
    x_in: np.ndarray,
    dt_in: np.ndarray,
    b_in: np.ndarray,
    c_in: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    x_f32 = np.asarray(x_in, dtype=np.float32)
    dt_f32 = np.asarray(dt_in, dtype=np.float32)
    b_f32 = np.asarray(b_in, dtype=np.float32)
    c_f32 = np.asarray(c_in, dtype=np.float32)
    a_f32 = np.asarray(a, dtype=np.float32)
    d_f32 = np.asarray(d, dtype=np.float32)

    length = int(x_f32.shape[0])
    d_inner = int(x_f32.shape[1])
    d_state = int(b_f32.shape[1])
    state = np.zeros((d_inner, d_state), dtype=np.float32)
    out = np.zeros((length, d_inner), dtype=np.float32)
    for idx in range(length):
        dt_t = dt_f32[idx][:, None]
        a_t = np.exp(dt_t * a_f32)
        state = a_t * state + x_f32[idx][:, None] * (dt_t * b_f32[idx][None, :])
        out[idx] = np.sum(state * c_f32[idx][None, :], axis=-1) + x_f32[idx] * d_f32
    return out


def _threshold_predictions(probs: np.ndarray, other_threshold: Optional[float], other_id: int) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float32)
    pred = np.argmax(arr, axis=-1).astype(np.int32)
    thr = float(other_threshold) if other_threshold is not None else 0.0
    if thr > 0.0 and pred.size > 0:
        conf = np.max(arr, axis=-1)
        pred[conf < thr] = int(other_id)
    return pred


def _resolve_threshold(value: Optional[float], default: float) -> float:
    threshold = default if value is None else float(value)
    if not np.isfinite(threshold):
        threshold = default
    return float(np.clip(threshold, 0.0, 1.0))


def _relabel_whitespace_from_neighbors(text: str, labels: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(text)
    if n == 0 or labels.size == 0 or labels.shape[0] != n:
        return labels, probs
    if not any(ch in _VISUAL_WHITESPACE_SET for ch in text):
        return labels, probs

    new_labels = np.asarray(labels, dtype=np.int32).copy()
    new_probs = np.asarray(probs, dtype=np.float32).copy()

    line_starts: List[int] = [0]
    for idx, ch in enumerate(text):
        if ch == "\n" and idx + 1 < n:
            line_starts.append(idx + 1)
    line_starts = sorted(set(line_starts))
    line_segments: List[Tuple[int, int]] = []
    for idx, start in enumerate(line_starts):
        end = line_starts[idx + 1] if idx + 1 < len(line_starts) else n
        if start < end:
            line_segments.append((start, end))

    left_same_line = [-1] * n
    right_same_line = [-1] * n
    for start, end in line_segments:
        last_non_ws = -1
        for pos in range(start, end):
            if text[pos] not in _VISUAL_WHITESPACE_SET:
                last_non_ws = pos
            left_same_line[pos] = last_non_ws
        last_non_ws = -1
        for pos in range(end - 1, start - 1, -1):
            if text[pos] not in _VISUAL_WHITESPACE_SET:
                last_non_ws = pos
            right_same_line[pos] = last_non_ws

    left_any = [-1] * n
    right_any = [-1] * n
    last_non_ws = -1
    for pos in range(n):
        if text[pos] not in _VISUAL_WHITESPACE_SET:
            last_non_ws = pos
        left_any[pos] = last_non_ws
    last_non_ws = -1
    for pos in range(n - 1, -1, -1):
        if text[pos] not in _VISUAL_WHITESPACE_SET:
            last_non_ws = pos
        right_any[pos] = last_non_ws

    for idx, ch in enumerate(text):
        if ch not in _VISUAL_WHITESPACE_SET:
            continue
        src = -1
        ls = left_same_line[idx]
        rs = right_same_line[idx]
        if ls != -1 or rs != -1:
            if ls == -1:
                src = rs
            elif rs == -1:
                src = ls
            else:
                dist_l = idx - ls
                dist_r = rs - idx
                src = ls if dist_l <= dist_r else rs
        else:
            la = left_any[idx]
            ra = right_any[idx]
            if la != -1 or ra != -1:
                if la == -1:
                    src = ra
                elif ra == -1:
                    src = la
                else:
                    dist_l = idx - la
                    dist_r = ra - idx
                    src = la if dist_l <= dist_r else ra
        if src == -1:
            continue
        new_labels[idx] = int(labels[src])
        new_probs[idx] = probs[src]
    return new_labels, new_probs


class NumpyMambaSegmentor:
    def __init__(self, manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
        self.manifest = dict(manifest)
        self.model_id = str(self.manifest.get("model_id", "sfullfiles4"))
        self.label_order = [str(label) for label in self.manifest.get("label_order", [])]
        self.display_labels = [str(label) for label in self.manifest.get("display_labels", self.label_order)]
        self.num_classes = int(self.manifest.get("num_classes", len(self.label_order)))
        self.window_bytes = int(self.manifest.get("window_bytes", 1536))
        self.window_stride_bytes = int(self.manifest.get("window_stride_bytes", self.window_bytes // 2))
        self.other_threshold = _resolve_threshold(self.manifest.get("other_threshold", 0.3), 0.3)
        self.max_input_bytes = max(0, int(self.manifest.get("max_input_bytes", 6144)))
        self.other_label_id = int(self.num_classes)

        self.embed = np.asarray(arrays["embed/embedding"], dtype=np.float32)
        self.final_ln_scale = np.asarray(arrays["final/ln_scale"], dtype=np.float32)
        self.final_ln_bias = np.asarray(arrays["final/ln_bias"], dtype=np.float32)
        self.final_dense_kernel = np.asarray(arrays["final/dense_kernel"], dtype=np.float32)
        self.final_dense_bias = np.asarray(arrays["final/dense_bias"], dtype=np.float32)

        self.blocks: List[Dict[str, np.ndarray]] = []
        n_layers = int(self.manifest.get("model", {}).get("n_layers", 0))
        for idx in range(n_layers):
            prefix = f"blocks/{idx}/"
            block = {
                "ln_scale": np.asarray(arrays[prefix + "ln_scale"], dtype=np.float32),
                "ln_bias": np.asarray(arrays[prefix + "ln_bias"], dtype=np.float32),
                "in_proj_kernel": np.asarray(arrays[prefix + "in_proj_kernel"], dtype=np.float32),
                "in_proj_bias": np.asarray(arrays[prefix + "in_proj_bias"], dtype=np.float32),
                "conv_kernel": np.asarray(arrays[prefix + "conv_kernel"], dtype=np.float32),
                "conv_bias": np.asarray(arrays[prefix + "conv_bias"], dtype=np.float32),
                "x_proj_kernel": np.asarray(arrays[prefix + "x_proj_kernel"], dtype=np.float32),
                "x_proj_bias": np.asarray(arrays[prefix + "x_proj_bias"], dtype=np.float32),
                "dt_proj_kernel": np.asarray(arrays[prefix + "dt_proj_kernel"], dtype=np.float32),
                "dt_proj_bias": np.asarray(arrays[prefix + "dt_proj_bias"], dtype=np.float32),
                "out_proj_kernel": np.asarray(arrays[prefix + "out_proj_kernel"], dtype=np.float32),
                "out_proj_bias": np.asarray(arrays[prefix + "out_proj_bias"], dtype=np.float32),
                "a": -np.exp(np.asarray(arrays[prefix + "a_log"], dtype=np.float32)),
                "d": np.asarray(arrays[prefix + "d"], dtype=np.float32),
            }
            self.blocks.append(block)

    @classmethod
    def from_files(cls, manifest_path: str | Path, weights_path: str | Path) -> "NumpyMambaSegmentor":
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        with np.load(Path(weights_path), allow_pickle=False) as data:
            arrays = {str(key): np.asarray(data[key], dtype=np.float32) for key in data.files}
        return cls(manifest, arrays)

    @classmethod
    def from_bytes(cls, manifest_bytes: bytes, weights_bytes: bytes) -> "NumpyMambaSegmentor":
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        with np.load(io.BytesIO(weights_bytes), allow_pickle=False) as data:
            arrays = {str(key): np.asarray(data[key], dtype=np.float32) for key in data.files}
        return cls(manifest, arrays)

    def _block_forward(self, x: np.ndarray, block: Mapping[str, np.ndarray]) -> np.ndarray:
        h = _layer_norm(x, block["ln_scale"], block["ln_bias"])
        xz = h @ block["in_proj_kernel"] + block["in_proj_bias"][None, :]
        split = xz.shape[1] // 2
        u = xz[:, :split]
        gate = xz[:, split:]

        u = _depthwise_conv_same(u, block["conv_kernel"], block["conv_bias"])
        u = _silu(u)

        x_dbl = u @ block["x_proj_kernel"] + block["x_proj_bias"][None, :]
        dt_rank = int(block["dt_proj_kernel"].shape[0])
        d_state = int(block["a"].shape[1])
        dt_raw = x_dbl[:, :dt_rank]
        b_in = x_dbl[:, dt_rank : dt_rank + d_state]
        c_in = x_dbl[:, dt_rank + d_state : dt_rank + (2 * d_state)]
        dt = _softplus(dt_raw @ block["dt_proj_kernel"] + block["dt_proj_bias"][None, :]) + 1e-4

        y = _selective_scan(u, dt, b_in, c_in, block["a"], block["d"])
        if bool(self.manifest.get("model", {}).get("bidirectional", True)):
            y_rev = _selective_scan(
                u[::-1],
                dt[::-1],
                b_in[::-1],
                c_in[::-1],
                block["a"],
                block["d"],
            )[::-1]
            y = y + y_rev

        y = y * _silu(gate)
        y = y @ block["out_proj_kernel"] + block["out_proj_bias"][None, :]
        return x + y

    def predict_window_probs(self, tokens: np.ndarray) -> np.ndarray:
        tok = _sanitize_model_tokens(np.asarray(tokens, dtype=np.int32).reshape(-1))
        length = int(tok.shape[0])
        if length <= 0:
            return np.zeros((0, self.num_classes), dtype=np.float32)
        h = np.asarray(self.embed[tok], dtype=np.float32)
        for block in self.blocks:
            h = self._block_forward(h, block)
        h = _layer_norm(h, self.final_ln_scale, self.final_ln_bias)
        logits = h @ self.final_dense_kernel + self.final_dense_bias[None, :]
        return _softmax(logits).astype(np.float32)

    def segment_byte_probs(self, byte_arr: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        arr = _sanitize_model_bytes(np.asarray(byte_arr, dtype=np.uint8).reshape(-1))
        length = int(arr.shape[0])
        if length == 0:
            return np.zeros((0, self.num_classes), dtype=np.float32), []
        probs = self.predict_window_probs(arr.astype(np.int32, copy=False))
        return probs, [(0, length)]

    def _byte_probs_to_char_payload(
        self,
        text: str,
        byte_probs: np.ndarray,
        *,
        top_k: int,
        other_threshold: float,
    ) -> Tuple[List[Dict[str, int]], List[List[Dict[str, float]]], List[float]]:
        n_chars = len(text)
        if n_chars == 0:
            return [], [], []

        labels = np.zeros((n_chars,), dtype=np.int32)
        char_probs = np.zeros((n_chars, self.num_classes), dtype=np.float32)
        byte_pos = 0
        for idx, ch in enumerate(text):
            byte_len = len(ch.encode("utf-8", "ignore"))
            if byte_len <= 0:
                continue
            avg_probs = np.mean(byte_probs[byte_pos : byte_pos + byte_len], axis=0, dtype=np.float32)
            labels[idx] = int(np.argmax(avg_probs))
            char_probs[idx] = avg_probs
            byte_pos += byte_len

        labels, char_probs = _relabel_whitespace_from_neighbors(text, labels, char_probs)
        labels = _threshold_predictions(char_probs, other_threshold, self.other_label_id)

        segments: List[Dict[str, int]] = []
        start = 0
        current = int(labels[0])
        for idx in range(1, n_chars):
            nxt = int(labels[idx])
            if nxt != current:
                segments.append({"start": int(start), "end": int(idx), "label_id": int(current)})
                start = idx
                current = nxt
        segments.append({"start": int(start), "end": int(n_chars), "label_id": int(current)})

        top_payload: List[List[Dict[str, float]]] = []
        confidences = np.max(char_probs, axis=-1).astype(np.float32)
        top_k_eff = max(1, min(int(top_k), self.num_classes))
        for idx in range(n_chars):
            order = np.argsort(char_probs[idx])[::-1][:top_k_eff]
            top_payload.append(
                [
                    {"id": int(label_id), "prob": float(char_probs[idx, label_id])}
                    for label_id in order
                ]
            )
        return segments, top_payload, [float(value) for value in confidences]

    def segment_text_payload(
        self,
        text: str,
        *,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        effective_threshold = _resolve_threshold(threshold, self.other_threshold)
        normalized = _normalize_input_text(text)
        raw_bytes = np.frombuffer(normalized.encode("utf-8", "ignore"), dtype=np.uint8)
        sanitized_bytes = _sanitize_model_bytes(raw_bytes)
        input_bytes = int(sanitized_bytes.shape[0])

        if self.max_input_bytes > 0 and input_bytes > self.max_input_bytes:
            raise ValueError(
                f"Input exceeds the public demo limit of {self.max_input_bytes} bytes after sanitization."
            )

        if input_bytes == 0:
            return {
                "text": normalized,
                "segments": [],
                "char_top_probs": [],
                "char_confidences": [],
                "stats": [],
                "input_bytes": 0,
                "window_count": 0,
                "other_threshold": effective_threshold,
                "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
            }

        byte_probs, spans = self.segment_byte_probs(sanitized_bytes)
        segments, char_top_probs, char_confidences = self._byte_probs_to_char_payload(
            normalized,
            byte_probs,
            top_k=top_k,
            other_threshold=effective_threshold,
        )

        counts: Dict[int, int] = {}
        for segment in segments:
            label_id = int(segment["label_id"])
            counts[label_id] = counts.get(label_id, 0) + int(segment["end"] - segment["start"])
        total_chars = max(1, len(normalized))
        stats = [
            {
                "id": int(label_id),
                "count": int(count),
                "pct": float(count / total_chars * 100.0),
            }
            for label_id, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

        return {
            "text": normalized,
            "segments": segments,
            "char_top_probs": char_top_probs,
            "char_confidences": char_confidences,
            "stats": stats,
            "input_bytes": input_bytes,
            "window_count": len(spans),
            "other_threshold": effective_threshold,
            "elapsed_ms": float((time.perf_counter() - started) * 1000.0),
        }


def load_model_json(manifest_path: str, weights_path: str) -> str:
    global _MODEL
    _MODEL = NumpyMambaSegmentor.from_files(manifest_path, weights_path)
    payload = {
        "loaded": True,
        "model_id": _MODEL.model_id,
        "num_classes": _MODEL.num_classes,
        "window_bytes": _MODEL.window_bytes,
        "window_stride_bytes": _MODEL.window_stride_bytes,
        "other_threshold": _MODEL.other_threshold,
        "max_input_bytes": _MODEL.max_input_bytes,
        "label_order": list(_MODEL.label_order),
        "display_labels": list(_MODEL.display_labels),
        "runtime": "pyodide-wasm",
    }
    return json.dumps(payload, separators=(",", ":"))


def segment_text_json(text: str, top_k: int = 5, threshold: Optional[float] = None) -> str:
    if _MODEL is None:
        raise RuntimeError("Model has not been loaded yet.")
    payload = _MODEL.segment_text_payload(
        text,
        top_k=max(1, int(top_k)),
        threshold=threshold,
    )
    return json.dumps(payload, separators=(",", ":"))
