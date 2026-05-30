from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from magika import Magika
try:
    from magika.types.seekable import Seekable
except ImportError:
    Seekable = None

from inference.backend import FastExecutionRecord, build_window_spans, window_weights
from magika_label_map import (
    OTHER_THESIS_LABEL,
    MagikaInventoryValidation,
    load_and_validate_installed_magika_mapping,
    magika_label_to_thesis_label,
)

DEFAULT_MAGIKA_WINDOW_SIZE = 1536


@dataclass(frozen=True)
class MagikaWindowPrediction:
    raw_label: str
    mapped_label: str
    score: float
    label_id: int
    probs: np.ndarray


class SlidingWindowMagikaSegmenter:
    def __init__(
        self,
        *,
        label_to_id: Mapping[str, int],
        other_class_index: int,
        batch_size: int = 128,
        window_size: int = DEFAULT_MAGIKA_WINDOW_SIZE,
    ) -> None:
        self.detector = Magika()
        self.validation: MagikaInventoryValidation = load_and_validate_installed_magika_mapping()
        self.label_to_id = {str(label): int(idx) for label, idx in label_to_id.items()}
        self.other_class_index = int(other_class_index)
        self.batch_size = max(1, int(batch_size))
        self.window_size = int(window_size)
        self.output_dim = max(self.other_class_index + 1, max(self.label_to_id.values(), default=-1) + 1)
        self.execution_history: List[FastExecutionRecord] = []
        self._weight_cache: Dict[int, np.ndarray] = {}

        model_config = getattr(self.detector, "_model_config", None)
        if model_config is None:
            raise RuntimeError("Installed Magika package does not expose _model_config.")
        self._model_config = model_config

    @property
    def module_version(self) -> str:
        return str(self.validation.package_version)

    @property
    def model_name(self) -> str:
        return str(self.validation.model_name)

    def clear_execution_history(self) -> None:
        self.execution_history.clear()

    def get_execution_history(self) -> List[FastExecutionRecord]:
        return list(self.execution_history)

    def _window_weights_cached(self, length: int) -> np.ndarray:
        cached = self._weight_cache.get(int(length))
        if cached is None:
            cached = window_weights(int(length))
            self._weight_cache[int(length)] = cached
        return cached

    def _extract_features(self, window_bytes: np.ndarray):
        content = np.asarray(window_bytes, dtype=np.uint8).tobytes()
        
        cfg = self._model_config
        is_dict = isinstance(cfg, dict)
        if is_dict:
            inner_cfg = cfg.get("cfg", {})
            sizes = inner_cfg.get("input_sizes", {})
            beg_size = int(sizes.get("beg", 512))
            mid_size = int(sizes.get("mid", 512))
            end_size = int(sizes.get("end", 512))
            padding_token = int(inner_cfg.get("dense_v4.padding_byte", 256))
            block_size = 4096
            use_inputs = False
        else:
            beg_size = int(cfg.beg_size)
            mid_size = int(cfg.mid_size)
            end_size = int(cfg.end_size)
            padding_token = int(cfg.padding_token)
            block_size = int(cfg.block_size)
            use_inputs = bool(cfg.use_inputs_at_offsets)

        if Seekable is None:
            return self.detector._extract_features_from_bytes(
                content,
                beg_size,
                mid_size,
                end_size,
                padding_token,
                block_size,
            )
        
        seekable = Seekable(io.BytesIO(content))
        return self.detector._extract_features_from_seekable(
            seekable,
            beg_size,
            mid_size,
            end_size,
            padding_token,
            block_size,
            use_inputs,
        )

    def _prediction_from_output(self, raw_label: str, score: float) -> MagikaWindowPrediction:
        mapped_label = magika_label_to_thesis_label(raw_label)
        probs = np.zeros((self.output_dim,), dtype=np.float32)
        if mapped_label == OTHER_THESIS_LABEL:
            label_id = self.other_class_index
        else:
            label_id = int(self.label_to_id.get(mapped_label, self.other_class_index))
        probs[label_id] = float(score)
        return MagikaWindowPrediction(
            raw_label=str(raw_label),
            mapped_label=str(mapped_label),
            score=float(score),
            label_id=int(label_id),
            probs=probs,
        )

    def _predict_window_batch(
        self,
        windows: Sequence[np.ndarray],
        *,
        mode: str,
        path: str,
    ) -> List[MagikaWindowPrediction]:
        if not windows:
            return []
        features = [
            (Path(f"{path}_{idx:06d}"), self._extract_features(window))
            for idx, window in enumerate(windows)
        ]
        outputs = self.detector._get_model_outputs_from_features(features)
        self.execution_history.append(
            FastExecutionRecord(
                mode=str(mode),
                path=str(path),
                actual_batch_size=int(len(windows)),
                padded_length=int(self.window_size),
                max_sequence_length=int(max(int(window.shape[0]) for window in windows)),
            )
        )
        return [
            self._prediction_from_output(
                str(getattr(output, "ct_label", getattr(output, "label", "txt"))), 
                float(output.score)
            )
            for _, output in outputs
        ]

    def predict_windows(
        self,
        windows: Sequence[np.ndarray],
        *,
        mode: str = "segment_full",
        path: str = "magika_sliding_window",
    ) -> List[MagikaWindowPrediction]:
        predictions: List[MagikaWindowPrediction] = []
        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
            predictions.extend(self._predict_window_batch(batch, mode=mode, path=path))
        return predictions

    def _stitch_prob_predictions(
        self,
        length: int,
        spans: Sequence[Tuple[int, int]],
        predictions: Sequence[MagikaWindowPrediction],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if length <= 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, self.output_dim), dtype=np.float32),
            )
        probs_accum = np.zeros((length, self.output_dim), dtype=np.float32)
        vote_accum = np.zeros((length, self.output_dim), dtype=np.float32)
        weight_accum = np.zeros((length,), dtype=np.float32)

        for (start, end), prediction in zip(spans, predictions):
            span_len = max(0, int(end) - int(start))
            if span_len <= 0:
                continue
            weights = self._window_weights_cached(span_len)
            probs_accum[start:end] += prediction.probs[None, :] * weights[:, None]
            vote_accum[start:end, int(prediction.label_id)] += weights
            weight_accum[start:end] += weights

        nonzero = weight_accum > 0
        if np.any(nonzero):
            probs_accum[nonzero] /= weight_accum[nonzero][:, None]
        if np.any(~nonzero):
            probs_accum[~nonzero, self.other_class_index] = 1.0
            vote_accum[~nonzero, self.other_class_index] = 1.0

        labels = np.argmax(vote_accum, axis=-1).astype(np.int32)
        return labels, probs_accum

    def _stitch_label_predictions(
        self,
        length: int,
        spans: Sequence[Tuple[int, int]],
        predictions: Sequence[MagikaWindowPrediction],
    ) -> np.ndarray:
        if length <= 0:
            return np.zeros((0,), dtype=np.int32)
        vote_accum = np.zeros((length, self.output_dim), dtype=np.float32)
        for (start, end), prediction in zip(spans, predictions):
            span_len = max(0, int(end) - int(start))
            if span_len <= 0:
                continue
            vote_accum[start:end, int(prediction.label_id)] += self._window_weights_cached(span_len)
        uncovered = np.sum(vote_accum, axis=-1) <= 0
        if np.any(uncovered):
            vote_accum[uncovered, self.other_class_index] = 1.0
        return np.argmax(vote_accum, axis=-1).astype(np.int32)

    def segment_bytes(self, byte_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        byte_arr = np.asarray(byte_arr, dtype=np.uint8)
        length = int(byte_arr.shape[0])
        if length <= 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, self.output_dim), dtype=np.float32),
            )
        spans = build_window_spans(length, self.window_size)
        windows = [byte_arr[start:end] for start, end in spans]
        predictions = self.predict_windows(windows, mode="segment_full", path="magika_sliding_window")
        return self._stitch_prob_predictions(length, spans, predictions)

    def segment_bytes_labels_only(self, byte_arr: np.ndarray) -> np.ndarray:
        labels, _ = self.segment_bytes(byte_arr)
        return labels

    def segment_byte_arrays_batch_labels_only(
        self,
        byte_arrays: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
        arrays = [np.asarray(arr, dtype=np.uint8) for arr in byte_arrays]
        spans_by_text = [build_window_spans(int(arr.shape[0]), self.window_size) for arr in arrays]

        flat_windows: List[np.ndarray] = []
        flat_refs: List[Tuple[int, int]] = []
        for file_idx, (arr, spans) in enumerate(zip(arrays, spans_by_text)):
            for span_idx, (start, end) in enumerate(spans):
                flat_windows.append(arr[start:end])
                flat_refs.append((file_idx, span_idx))

        flat_predictions = self.predict_windows(flat_windows, mode="segment_full", path="magika_sliding_window")

        predictions_by_file: List[List[MagikaWindowPrediction]] = [[] for _ in arrays]
        for prediction, (file_idx, _) in zip(flat_predictions, flat_refs):
            predictions_by_file[file_idx].append(prediction)

        labels_by_text: List[np.ndarray] = []
        for arr, spans, predictions in zip(arrays, spans_by_text, predictions_by_file):
            labels_by_text.append(
                self._stitch_label_predictions(int(arr.shape[0]), spans, predictions)
            )
        return labels_by_text, spans_by_text
