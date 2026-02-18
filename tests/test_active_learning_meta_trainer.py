from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.meta_trainer import _count_oracle_refinement_rows, _linear_oracle_mix_prob


class TestMetaTrainerMixSchedule(unittest.TestCase):
    def test_linear_oracle_mix_prob(self) -> None:
        self.assertEqual(_linear_oracle_mix_prob(0, max_prob=0.5, full_at_rows=1000), 0.0)
        self.assertAlmostEqual(_linear_oracle_mix_prob(500, max_prob=0.5, full_at_rows=1000), 0.25)
        self.assertAlmostEqual(_linear_oracle_mix_prob(1000, max_prob=0.5, full_at_rows=1000), 0.5)
        self.assertAlmostEqual(_linear_oracle_mix_prob(2000, max_prob=0.5, full_at_rows=1000), 0.5)

    def test_count_oracle_refinement_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "label_store.sqlite"
            with sqlite3.connect(str(db_path)) as con:
                con.execute(
                    "CREATE TABLE refinements (id INTEGER PRIMARY KEY AUTOINCREMENT, status TEXT NOT NULL)"
                )
                con.execute("INSERT INTO refinements (status) VALUES ('ok')")
                con.execute("INSERT INTO refinements (status) VALUES ('ok')")
                con.execute("INSERT INTO refinements (status) VALUES ('failed')")
            self.assertEqual(_count_oracle_refinement_rows(str(db_path)), 2)

    def test_count_rows_handles_missing_db(self) -> None:
        self.assertEqual(_count_oracle_refinement_rows("/tmp/definitely_missing_al_store.sqlite"), 0)


if __name__ == "__main__":
    unittest.main()

