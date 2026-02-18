from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg
from utils.outlier_data import OutlierBatcher


class TestOutlierBatcher(unittest.TestCase):
    def test_heldout_mode_loads_arrow_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ds_path = root / "train" / "other" / "dataset"
            ds_path.parent.mkdir(parents=True, exist_ok=True)
            Dataset.from_dict({"content": ["alpha", "beta", "gamma"]}).save_to_disk(
                str(ds_path)
            )

            batcher = OutlierBatcher(
                source="heldout",
                data_root=str(root / "unused"),
                heldout_root=str(root),
                window_bytes=16,
                batch_size=4,
                seed=123,
            )
            self.assertEqual(batcher.mode, "mixed")
            self.assertEqual(batcher.heldout_langs, ["other"])
            xb = batcher.get()
            self.assertEqual(xb.shape, (4, 16))
            self.assertEqual(xb.dtype, np.int32)
            self.assertTrue(np.any(xb != int(cfg.PAD_BYTE_ID)))

    def test_heldout_mode_falls_back_to_random_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            batcher = OutlierBatcher(
                source="heldout",
                data_root=str(root / "unused"),
                heldout_root=str(root / "missing"),
                window_bytes=8,
                batch_size=2,
                seed=7,
            )
            self.assertEqual(batcher.mode, "random")
            xb = batcher.get()
            self.assertEqual(xb.shape, (2, 8))
            self.assertTrue(np.any(xb != int(cfg.PAD_BYTE_ID)))

    def test_mixed_mode_is_default_when_heldout_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ds_path = root / "train" / "other" / "dataset"
            ds_path.parent.mkdir(parents=True, exist_ok=True)
            Dataset.from_dict({"content": ["delta", "epsilon", "zeta"]}).save_to_disk(
                str(ds_path)
            )

            batcher = OutlierBatcher(
                source="mixed",
                data_root=str(root / "unused"),
                heldout_root=str(root),
                window_bytes=16,
                batch_size=4,
                seed=42,
            )
            desc = batcher.describe()
            self.assertEqual(desc.get("mode"), "mixed")
            self.assertTrue(np.isclose(float(desc.get("holdout_mix_prob", 0.0)), 0.5))


if __name__ == "__main__":
    unittest.main()
