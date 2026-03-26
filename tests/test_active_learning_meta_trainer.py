from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_learning.meta_trainer import (
    _build_train_cmd,
    _count_oracle_refinement_rows,
    _linear_oracle_mix_prob,
    _parse_round_summary,
    _sqlite_size_mb,
)


class TestMetaTrainerMixSchedule(unittest.TestCase):
    def test_linear_oracle_mix_prob(self) -> None:
        self.assertEqual(_linear_oracle_mix_prob(0, max_prob=0.5, full_at_rows=1000), 0.0)
        self.assertAlmostEqual(_linear_oracle_mix_prob(500, max_prob=0.5, full_at_rows=1000), 0.25)
        self.assertAlmostEqual(_linear_oracle_mix_prob(1000, max_prob=0.5, full_at_rows=1000), 0.5)
        self.assertAlmostEqual(_linear_oracle_mix_prob(2000, max_prob=0.5, full_at_rows=1000), 0.5)

    def test_build_train_cmd_uses_requested_replay_mix_prob(self) -> None:
        cmd = _build_train_cmd(
            python_executable="/tmp/python",
            data_root="downloader/arrow_out",
            ckpt_path="/tmp/demo.msgpack",
            current_train_step=49,
            train_max_minutes=1000,
            al_store="/tmp/labels.sqlite",
            active_learning_mix_prob=0.394,
            al_max_windows=10_000_000,
            full_files=True,
            full_file_max_bytes=10_000,
            persistent_trainer_enabled=True,
            shared_wandb_run_id="sfullfiles4",
            shared_schedule_final_step=45_048,
            train_extra_args="--eval_every 150 --lr 2e-5",
        )
        self.assertEqual(cmd[cmd.index("--active_learning_mix_prob") + 1], "0.394")
        self.assertEqual(cmd[cmd.index("--steps") + 1], "48")
        self.assertIn("--continue", cmd)
        self.assertEqual(cmd[cmd.index("--continue") + 1], "sfullfiles4")

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


class TestParseRoundSummary(unittest.TestCase):
    def test_parses_json_from_mixed_output(self) -> None:
        stdout = (
            "Preparing datasets...\n"
            "Model created with 1.23M parameters.\n"
            "Sampling python: selected=16/16\n"
            "Oracle returned refinements for 6/6 snippets.\n"
            "{\n"
            '  "round_id": "al-20260312-073000-abc12345",\n'
            '  "status": "ok",\n'
            '  "samples": 6,\n'
            '  "stored": 4,\n'
            '  "llm_total_requests": 2,\n'
            '  "llm_total_tokens": 15000\n'
            "}\n"
        )
        result = _parse_round_summary(stdout)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["samples"], 6)
        self.assertEqual(result["stored"], 4)
        self.assertEqual(result["llm_total_requests"], 2)
        self.assertEqual(result["llm_total_tokens"], 15000)

    def test_parses_empty_stdout(self) -> None:
        self.assertEqual(_parse_round_summary(""), {})

    def test_parses_no_json(self) -> None:
        self.assertEqual(_parse_round_summary("just some log lines\nno json here\n"), {})

    def test_parses_last_json_block(self) -> None:
        stdout = (
            '{"early": "json"}\n'
            "Some log output\n"
            '{"round_id": "latest", "status": "ok"}\n'
        )
        result = _parse_round_summary(stdout)
        self.assertEqual(result["round_id"], "latest")
        self.assertEqual(result["status"], "ok")


class TestSqliteSizeMb(unittest.TestCase):
    def test_returns_size_for_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            with sqlite3.connect(str(db_path)) as con:
                con.execute("CREATE TABLE t (id INTEGER)")
                for i in range(100):
                    con.execute("INSERT INTO t (id) VALUES (?)", (i,))
            size = _sqlite_size_mb(str(db_path))
            self.assertGreater(size, 0.0)

    def test_returns_zero_for_missing_file(self) -> None:
        self.assertEqual(_sqlite_size_mb("/tmp/definitely_missing_db.sqlite"), 0.0)


if __name__ == "__main__":
    unittest.main()
