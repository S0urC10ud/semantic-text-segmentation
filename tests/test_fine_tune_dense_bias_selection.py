from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import train.main as train_main


class TestFineTuneDenseBiasSelection(unittest.TestCase):
    def test_select_weakest_monitor_label_prefers_thresholded_f1(self) -> None:
        summary = {
            "monitor/per_class/01_csharp/f1": 0.91,
            "monitor/per_class/10_c_family/f1": 0.92,
            "monitor_thresh/per_class/01_csharp/f1": 0.908,
            "monitor_thresh/per_class/10_c_family/f1": 0.901,
            "monitor_thresh/per_class/23_markdown/f1": 0.94,
        }

        selected = train_main._select_weakest_monitor_label(summary)

        self.assertEqual(
            selected,
            ("c_family", "monitor_thresh/per_class/10_c_family/f1", 0.901),
        )

    def test_resolve_fine_tune_dense_bias_uses_source_run_summary_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            summary_dir = repo_root / "wandb" / "run-20260315_000000-src12345" / "files"
            summary_dir.mkdir(parents=True)
            (summary_dir / "wandb-summary.json").write_text(
                json.dumps(
                    {
                        "monitor_thresh/per_class/01_csharp/f1": 0.91,
                        "monitor_thresh/per_class/10_c_family/f1": 0.89,
                        "monitor/per_class/10_c_family/f1": 0.92,
                    }
                ),
                encoding="utf-8",
            )

            args = SimpleNamespace(
                fine_tune_dense_bias_prob=0.1,
                fine_tune_dense_bias_label="auto",
                continue_run_id="",
                wandb_run_id="newchild1",
                fine_tune_dense_bias_source_run_id="src12345",
                fine_tune_run_id="",
            )

            label, metric_key, metric_value, metric_run_id = (
                train_main._resolve_fine_tune_dense_bias(repo_root, args)
            )

        self.assertEqual(label, "c_family")
        self.assertEqual(metric_key, "monitor_thresh/per_class/10_c_family/f1")
        self.assertEqual(metric_value, 0.89)
        self.assertEqual(metric_run_id, "src12345")


if __name__ == "__main__":
    unittest.main()
