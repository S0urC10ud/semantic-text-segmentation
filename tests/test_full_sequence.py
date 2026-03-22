from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg
from train.utils.full_sequence import build_monitor_file_sequence, pack_sequence_pieces
from train.utils.monitor_eval import FILE_DTYPE, SEG_DTYPE


class TestFullSequenceHelpers(unittest.TestCase):
    def test_pack_sequence_pieces_masks_separator_with_pad(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        sql_id = int(cfg.LANG2ID["sql"])
        x, y, _ = pack_sequence_pieces(
            [
                (
                    np.array([ord("a"), ord("b"), cfg.PAD_BYTE_ID], dtype=np.int32),
                    np.array([py_id, py_id, cfg.PAD_ID], dtype=np.uint8),
                ),
                (
                    np.array([ord("C"), ord("D"), cfg.PAD_BYTE_ID], dtype=np.int32),
                    np.array([sql_id, sql_id, cfg.PAD_ID], dtype=np.uint8),
                ),
            ],
            target_len=8,
            pad_byte_id=int(cfg.PAD_BYTE_ID),
            pad_label_id=int(cfg.PAD_ID),
            separator_label_id=int(cfg.PAD_ID),
        )
        self.assertEqual(bytes(x[:6].astype(np.uint8)), b"ab\n\nCD")
        self.assertEqual(int(y[0]), py_id)
        self.assertEqual(int(y[1]), py_id)
        self.assertEqual(int(y[2]), int(cfg.PAD_ID))
        self.assertEqual(int(y[3]), int(cfg.PAD_ID))
        self.assertEqual(int(y[4]), sql_id)
        self.assertEqual(int(y[5]), sql_id)

    def test_build_monitor_file_sequence_preserves_labels_and_pads(self) -> None:
        py_id = int(cfg.LANG2ID["python"])
        sql_id = int(cfg.LANG2ID["sql"])
        files = np.array([(0, 6, 0, 2, 0, py_id)], dtype=FILE_DTYPE)
        segments = np.array(
            [
                (0, 0, 2, py_id),
                (0, 2, 6, sql_id),
            ],
            dtype=SEG_DTYPE,
        )
        contents = np.frombuffer(b"aaBBBB", dtype=np.uint8)
        x, y, meta = build_monitor_file_sequence(
            files,
            contents,
            segments,
            0,
            target_len=10,
            pad_byte_id=int(cfg.PAD_BYTE_ID),
            pad_label_id=int(cfg.PAD_ID),
            random_crop=False,
        )
        self.assertEqual(meta["byte_len"], 6)
        self.assertEqual(bytes(x[:6].astype(np.uint8)), b"aaBBBB")
        np.testing.assert_array_equal(
            y[:6],
            np.array([py_id, py_id, sql_id, sql_id, sql_id, sql_id], dtype=np.uint8),
        )
        self.assertTrue(np.all(y[6:] == int(cfg.PAD_ID)))


if __name__ == "__main__":
    unittest.main()
