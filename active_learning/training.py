from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .label_store import LabelStore, DEFAULT_EXCLUDED_TRAINING_SOURCE_SPLITS


def _phase_seed(seed: int, phase_nonce: int = 0, *, salt: int = 0) -> int:
    value = int(seed) ^ int(salt)
    value ^= (int(phase_nonce) + 1) * 0x9E3779B1
    return value & 0xFFFFFFFFFFFFFFFF


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

    def reset_rng(self, seed: int, *, phase_nonce: int = 0) -> None:
        self.rng = np.random.default_rng(_phase_seed(seed, phase_nonce))

    def reload_from_store(
        self,
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
        phase_nonce: int = 0,
    ) -> int:
        refreshed = type(self).from_store(
            store_path=store_path,
            window_bytes=window_bytes,
            pad_byte_id=pad_byte_id,
            pad_label_id=pad_label_id,
            label_to_id=label_to_id,
            max_windows=max_windows,
            seed=_phase_seed(seed, phase_nonce),
            fallback_label=fallback_label,
            exclude_source_splits=exclude_source_splits,
            full_files=full_files,
        )
        self.windows_x = refreshed.windows_x
        self.windows_y = refreshed.windows_y
        self.rng = refreshed.rng
        return self.size

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

    def reset_rng(self, seed: int, *, phase_nonce: int = 0) -> None:
        phase_seed = _phase_seed(seed, phase_nonce, salt=0xA11CE)
        self.rng = np.random.default_rng(phase_seed)
        self.replay.reset_rng(seed, phase_nonce=phase_nonce)

    def refresh_replay_from_store(
        self,
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
        phase_nonce: int = 0,
        mix_prob: Optional[float] = None,
    ) -> int:
        if mix_prob is not None:
            self.mix_prob = float(max(0.0, min(1.0, mix_prob)))
        size = self.replay.reload_from_store(
            store_path=store_path,
            window_bytes=window_bytes,
            pad_byte_id=pad_byte_id,
            pad_label_id=pad_label_id,
            label_to_id=label_to_id,
            max_windows=max_windows,
            seed=seed,
            fallback_label=fallback_label,
            exclude_source_splits=exclude_source_splits,
            full_files=full_files,
            phase_nonce=phase_nonce,
        )
        self.reset_rng(seed, phase_nonce=phase_nonce)
        return size

    def reset_phase(
        self,
        *,
        seed: int,
        phase_nonce: int = 0,
        refresh_active_learning: bool = False,
    ) -> int:
        drained = 0
        if hasattr(self.base, "reset_phase"):
            drained = int(
                self.base.reset_phase(
                    seed=seed,
                    phase_nonce=phase_nonce,
                    refresh_active_learning=refresh_active_learning,
                )
                or 0
            )
        self.reset_rng(seed, phase_nonce=phase_nonce)
        self.total_batches = 0
        self.total_replay_rows = 0
        return drained

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
