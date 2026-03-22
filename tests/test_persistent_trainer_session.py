from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from active_learning.meta_trainer import PersistentTrainerSession
from active_learning.persistent_control import (
    TRAINER_EVENT_PREFIX,
    encode_trainer_command,
    encode_trainer_event,
    maybe_parse_trainer_event_line,
    parse_trainer_command_line,
)


class TestPersistentTrainerControl(unittest.TestCase):
    def test_event_roundtrip(self) -> None:
        line = encode_trainer_event("ready", current_step=123, ckpt_path="demo.msgpack")
        self.assertTrue(line.startswith(TRAINER_EVENT_PREFIX))
        parsed = maybe_parse_trainer_event_line(line)
        self.assertEqual(parsed["event"], "ready")
        self.assertEqual(int(parsed["current_step"]), 123)
        self.assertEqual(parsed["ckpt_path"], "demo.msgpack")

    def test_command_roundtrip(self) -> None:
        line = encode_trainer_command("train", steps=900, max_minutes=100, phase_label="meta_round_1")
        parsed = parse_trainer_command_line(line)
        self.assertEqual(parsed["command"], "train")
        self.assertEqual(int(parsed["steps"]), 900)
        self.assertEqual(int(parsed["max_minutes"]), 100)
        self.assertEqual(parsed["phase_label"], "meta_round_1")

    def test_persistent_trainer_session_happy_path(self) -> None:
        child_code = (
            "import sys\n"
            "from active_learning.persistent_control import encode_trainer_event, parse_trainer_command_line\n"
            "print(encode_trainer_event('ready', current_step=7, wandb_run_id='demo-run'), flush=True)\n"
            "for line in sys.stdin:\n"
            "    cmd = parse_trainer_command_line(line)\n"
            "    if cmd['command'] == 'train':\n"
            "        print(encode_trainer_event('chunk_done', current_step=cmd['steps'], target_step=cmd['steps'], final_reason='completed'), flush=True)\n"
            "    elif cmd['command'] == 'shutdown':\n"
            "        print(encode_trainer_event('shutdown_ack', current_step=11, reason=cmd.get('reason', 'shutdown')), flush=True)\n"
            "        break\n"
        )
        session = PersistentTrainerSession([sys.executable, "-c", child_code])
        ready = session.start(ready_timeout_seconds=10.0)
        self.assertEqual(ready["event"], "ready")
        self.assertEqual(int(ready["current_step"]), 7)
        chunk = session.train_until(
            target_step=11,
            max_minutes=1000,
            phase_label="meta_round_1",
            active_learning_mix_prob=0.25,
        )
        self.assertEqual(chunk["event"], "chunk_done")
        self.assertEqual(int(chunk["current_step"]), 11)
        session.shutdown(reason="test_done")


if __name__ == "__main__":
    unittest.main()
