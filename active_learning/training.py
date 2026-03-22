from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .label_store import LabelStore, DEFAULT_EXCLUDED_TRAINING_SOURCE_SPLITS


@dataclass
class ActiveLearningReplay:
    windows_x: np.ndarray
    windows_y: np.ndarray
    rng: np.random.Generator

    @classmethod
    def from_store(
        cls,
        *,
        store_path: str | Path,
        window_bytes: int,
        pad_byte_id: int,
        pad_label_id: int,
        label_to_id: Dict[str, int],
        max_windows: Optional[int],
        seed: int,
        fallback_label: str = "other",
        exclude_source_splits: Tuple[str, ...] = DEFAULT_EXCLUDED_TRAINING_SOURCE_SPLITS,
        full_files: bool = False,
    ) -> "ActiveLearningReplay":
        store = LabelStore(store_path)
        if full_files:
            windows = store.build_training_sequences(
                sequence_bytes=int(window_bytes),
                pad_byte_id=int(pad_byte_id),
                pad_label_id=int(pad_label_id),
                label_to_id=label_to_id,
                max_sequences=max_windows,
                fallback_label=str(fallback_label),
                exclude_source_splits=exclude_source_splits,
            )
        else:
            windows = store.build_training_windows(
                window_bytes=int(window_bytes),
                pad_byte_id=int(pad_byte_id),
                pad_label_id=int(pad_label_id),
                label_to_id=label_to_id,
                max_windows=max_windows,
                fallback_label=str(fallback_label),
                exclude_source_splits=exclude_source_splits,
            )
        if not windows:
            return cls(
                windows_x=np.empty((0, int(window_bytes)), dtype=np.int32),
                windows_y=np.empty((0, int(window_bytes)), dtype=np.uint8),
                rng=np.random.default_rng(int(seed)),
            )
        xs = np.stack([x for x, _ in windows], axis=0).astype(np.int32, copy=False)
        ys = np.stack([y for _, y in windows], axis=0).astype(np.uint8, copy=False)
        return cls(
            windows_x=xs,
            windows_y=ys,
            rng=np.random.default_rng(int(seed)),
        )

    @property
    def size(self) -> int:
        return int(self.windows_x.shape[0])

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.size <= 0 or n <= 0:
            return (
                np.empty((0, self.windows_x.shape[1]), dtype=np.int32),
                np.empty((0, self.windows_y.shape[1]), dtype=np.uint8),
            )
        idx = self.rng.integers(0, self.size, size=int(n))
        return self.windows_x[idx], self.windows_y[idx]


class MixedBatcher:
    """Wrap a base batcher and inject replay windows with a configurable ratio."""

    def __init__(
        self,
        base_batcher,
        replay: ActiveLearningReplay,
        *,
        mix_prob: float,
        seed: int,
    ) -> None:
        self.base = base_batcher
        self.replay = replay
        self.mix_prob = float(max(0.0, min(1.0, mix_prob)))
        self.rng = np.random.default_rng(int(seed) ^ 0xA11CE)
        self.total_batches = 0
        self.total_replay_rows = 0

    def get(self):
        xb, yb = self.base.get()
        if self.replay.size <= 0 or self.mix_prob <= 0.0:
            return xb, yb
        batch = int(xb.shape[0])
        picks = self.rng.random(batch) < self.mix_prob
        count = int(np.count_nonzero(picks))
        if count <= 0:
            return xb, yb
        rx, ry = self.replay.sample(count)
        xb = np.array(xb, copy=True)
        yb = np.array(yb, copy=True)
        xb[picks] = rx
        yb[picks] = ry
        self.total_batches += 1
        self.total_replay_rows += count
        return xb, yb

    def get_epochs(self):
        return self.base.get_epochs()

    def close(self):
        return self.base.close()
