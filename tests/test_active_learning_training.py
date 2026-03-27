from __future__ import annotations

import queue
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from active_learning.label_store import LabelStore, StoredRefinement
from active_learning.training import ActiveLearningReplay, MixedBatcher, _phase_seed
from train.utils.epoch_batcher import EpochPrefetchBatcher, MonitorFineTuneBatcher


class _DummyBaseBatcher:
    def __init__(self) -> None:
        self.reset_phase_calls: list[tuple[int, int, bool]] = []

    def get(self):
        xb = np.zeros((2, 8), dtype=np.int32)
        yb = np.zeros((2, 8), dtype=np.uint8)
        return xb, yb

    def get_epochs(self):
        return {}

    def close(self):
        return None

    def reset_phase(
        self,
        *,
        seed: int,
        phase_nonce: int = 0,
        refresh_active_learning: bool = False,
    ) -> int:
        self.reset_phase_calls.append(
            (int(seed), int(phase_nonce), bool(refresh_active_learning))
        )
        return 3


class _FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeValue:
    def __init__(self, value: int):
        self.value = int(value)

    def get_lock(self):
        return _FakeLock()


class _FakeQueue:
    def __init__(self, items):
        self._items = list(items)

    def get_nowait(self):
        if not self._items:
            raise queue.Empty
        return self._items.pop(0)


def _make_refinement(*, sample_hash: str, snippet_text: str, label: str) -> StoredRefinement:
    return StoredRefinement(
        round_id="r1",
        source_split="train",
        source_lang=label,
        sample_index=0,
        sample_hash=sample_hash,
        boundary_index=len(snippet_text) // 2,
        snippet_start=0,
        snippet_end=len(snippet_text),
        snippet_text=snippet_text,
        oracle_name="stub",
        oracle_model="stub",
        oracle_run_id="",
        status="ok",
        acquisition_score=1.0,
        predicted_segments=[],
        refined_segments=[{"start": 0, "end": len(snippet_text), "label": label}],
        metadata={},
    )


class TestActiveLearningTraining(unittest.TestCase):
    def test_mixed_batcher_refresh_replay_from_store_picks_up_new_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "al.sqlite"
            store = LabelStore(db_path)
            store.add(_make_refinement(sample_hash="hash_a", snippet_text="abcd", label="python"))

            replay = ActiveLearningReplay.from_store(
                store_path=db_path,
                window_bytes=8,
                pad_byte_id=0,
                pad_label_id=255,
                label_to_id={"python": 0, "sql": 1, "other": 2},
                max_windows=16,
                seed=7,
                fallback_label="other",
            )
            batcher = MixedBatcher(_DummyBaseBatcher(), replay, mix_prob=0.5, seed=7)
            self.assertEqual(batcher.replay.size, 1)

            store.add(_make_refinement(sample_hash="hash_b", snippet_text="WXYZ", label="sql"))

            new_size = batcher.refresh_replay_from_store(
                store_path=db_path,
                window_bytes=8,
                pad_byte_id=0,
                pad_label_id=255,
                label_to_id={"python": 0, "sql": 1, "other": 2},
                max_windows=16,
                seed=7,
                fallback_label="other",
                phase_nonce=5,
                mix_prob=0.25,
            )

            self.assertEqual(new_size, 2)
            self.assertEqual(batcher.replay.size, 2)
            self.assertAlmostEqual(batcher.mix_prob, 0.25)
            np.testing.assert_array_equal(
                batcher.replay.windows_x[1, :4],
                np.frombuffer(b"WXYZ", dtype=np.uint8).astype(np.int32),
            )

    def test_mixed_batcher_reset_phase_delegates_and_reseeds(self) -> None:
        base = _DummyBaseBatcher()
        replay = ActiveLearningReplay(
            windows_x=np.arange(16, dtype=np.int32).reshape(2, 8),
            windows_y=np.zeros((2, 8), dtype=np.uint8),
            rng=np.random.default_rng(1),
        )
        batcher = MixedBatcher(base, replay, mix_prob=0.5, seed=11)
        batcher.total_batches = 9
        batcher.total_replay_rows = 17

        drained = batcher.reset_phase(seed=11, phase_nonce=4, refresh_active_learning=True)

        self.assertEqual(drained, 3)
        self.assertEqual(base.reset_phase_calls, [(11, 4, True)])
        self.assertEqual(batcher.total_batches, 0)
        self.assertEqual(batcher.total_replay_rows, 0)
        expected = np.random.default_rng(_phase_seed(11, 4, salt=0xA11CE)).random()
        self.assertAlmostEqual(float(batcher.rng.random()), float(expected))

    def test_monitor_finetune_batcher_reset_phase_drains_queue_and_bumps_refresh(self) -> None:
        batcher = MonitorFineTuneBatcher.__new__(MonitorFineTuneBatcher)
        batcher.q = _FakeQueue([1, 2])
        batcher.augment = True
        batcher._phase_nonce = _FakeValue(0)
        batcher._al_refresh_nonce = _FakeValue(0)

        drained = MonitorFineTuneBatcher.reset_phase(
            batcher,
            seed=123,
            phase_nonce=5,
            refresh_active_learning=True,
        )

        self.assertEqual(drained, 2)
        self.assertEqual(int(batcher._phase_nonce.value), 6)
        self.assertEqual(int(batcher._al_refresh_nonce.value), 1)

    def test_epoch_prefetch_batcher_reset_phase_drains_queue(self) -> None:
        batcher = EpochPrefetchBatcher.__new__(EpochPrefetchBatcher)
        batcher.q = _FakeQueue([1, 2, 3])
        batcher._phase_nonce = _FakeValue(1)

        drained = EpochPrefetchBatcher.reset_phase(
            batcher,
            seed=99,
            phase_nonce=4,
            refresh_active_learning=False,
        )

        self.assertEqual(drained, 3)
        self.assertEqual(int(batcher._phase_nonce.value), 5)


if __name__ == "__main__":
    unittest.main()
