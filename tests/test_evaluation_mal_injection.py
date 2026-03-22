from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import datasets as hfds
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import evaluation.evaluation as evalmod


class _FakeRunner:
    def __init__(self, labels: list[int], probs: list[np.ndarray]) -> None:
        self._labels = list(labels)
        self._probs = [np.asarray(row, dtype=np.float32) for row in probs]

    def segment_text(self, text: str, *, min_run_chars: int = 1):
        return [], list(self._labels), [row.copy() for row in self._probs]


def _record(
    *,
    content: str,
    segments: list[dict],
    host_lang: str,
    payload_lang: str,
) -> dict:
    return {
        "task": "mal_injection",
        "example_id": "mal-test-0001",
        "content": content,
        "segments": {
            "label": [str(seg["label"]) for seg in segments],
            "char_start": [int(seg["char_start"]) for seg in segments],
            "char_end": [int(seg["char_end"]) for seg in segments],
        },
        "source_langs": sorted({str(seg["label"]) for seg in segments}),
        "metadata_json": json.dumps(
            {
                "host_lang": host_lang,
                "payload_lang": payload_lang,
            },
            sort_keys=True,
        ),
    }


class TestMalInjectionSoftDetection(unittest.TestCase):
    def _prob_row(self, base_label: str, base_prob: float, **extra_probs: float) -> np.ndarray:
        row = np.zeros((int(evalmod.cfg.NUM_CLASSES),), dtype=np.float32)
        row[int(evalmod.cfg.LANG2ID[base_label])] = float(base_prob)
        for label, prob in extra_probs.items():
            row[int(evalmod.cfg.LANG2ID[label])] = float(prob)
        return row

    def test_soft_hit_can_succeed_when_hard_payload_detection_fails(self) -> None:
        host_lang = "sql"
        payload_lang = "python"
        host_id = int(evalmod.cfg.LANG2ID[host_lang])
        content = "HHPPPP"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    content=content,
                    segments=[
                        {"label": host_lang, "char_start": 0, "char_end": 2},
                        {"label": payload_lang, "char_start": 2, "char_end": 6},
                    ],
                    host_lang=host_lang,
                    payload_lang=payload_lang,
                )
            ]
        )
        runner = _FakeRunner(
            labels=[host_id] * len(content),
            probs=[
                self._prob_row(host_lang, 0.99),
                self._prob_row(host_lang, 0.99),
                self._prob_row(host_lang, 0.90, python=0.10),
                self._prob_row(host_lang, 0.93, python=0.07),
                self._prob_row(host_lang, 0.96, python=0.04),
                self._prob_row(host_lang, 0.97, python=0.03),
            ],
        )

        metrics = evalmod.evaluate_task(
            "mal_injection",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        payload_stats = metrics.extras["mal_payload_detection"]
        hard_correct = payload_stats["by_lang"][payload_lang]
        soft_stats = payload_stats["soft_detection"]
        soft_correct = soft_stats["correct_detection"]
        soft_any = soft_stats["any_detection"]

        self.assertEqual(hard_correct["detected"], 0)
        self.assertEqual(soft_correct["detected"], 1)
        self.assertEqual(soft_correct["truth_chars"], 4)
        self.assertEqual(soft_correct["matched_chars"], 2)
        self.assertEqual(soft_any["detected"], 1)
        self.assertEqual(soft_any["matched_chars"], 2)

        highlights = "\n".join(evalmod._collect_task_highlights([metrics]))
        self.assertIn("| Any non-wrapper |", highlights)
        self.assertIn("| Correct payload |", highlights)
        self.assertIn("| Any non-wrapper >=5% | 1/1 | 50.0% |", highlights)
        self.assertIn("| Correct payload >=5% | 1/1 | 50.0% |", highlights)

    def test_any_non_wrapper_soft_hit_uses_summed_non_host_probability(self) -> None:
        host_lang = "sql"
        payload_lang = "python"
        host_id = int(evalmod.cfg.LANG2ID[host_lang])
        content = "HHPPPP"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    content=content,
                    segments=[
                        {"label": host_lang, "char_start": 0, "char_end": 2},
                        {"label": payload_lang, "char_start": 2, "char_end": 6},
                    ],
                    host_lang=host_lang,
                    payload_lang=payload_lang,
                )
            ]
        )
        runner = _FakeRunner(
            labels=[host_id] * len(content),
            probs=[
                self._prob_row(host_lang, 0.99),
                self._prob_row(host_lang, 0.99),
                self._prob_row(host_lang, 0.93, python=0.04, shell=0.03),
                self._prob_row(host_lang, 0.93, python=0.04, shell=0.03),
                self._prob_row(host_lang, 0.97, python=0.02, shell=0.01),
                self._prob_row(host_lang, 0.97, python=0.02, shell=0.01),
            ],
        )

        metrics = evalmod.evaluate_task(
            "mal_injection",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        soft_stats = metrics.extras["mal_payload_detection"]["soft_detection"]
        self.assertEqual(soft_stats["correct_detection"]["detected"], 0)
        self.assertEqual(soft_stats["correct_detection"]["matched_chars"], 0)
        self.assertEqual(soft_stats["any_detection"]["detected"], 1)
        self.assertEqual(soft_stats["any_detection"]["matched_chars"], 2)

    def test_whitespace_inside_payload_truth_is_filtered_from_soft_counts(self) -> None:
        payload_lang = "python"
        host_lang = "sql"
        host_id = int(evalmod.cfg.LANG2ID[host_lang])
        content = "PP\nPP"
        dataset = hfds.Dataset.from_list(
            [
                _record(
                    content=content,
                    segments=[
                        {"label": payload_lang, "char_start": 0, "char_end": len(content)},
                    ],
                    host_lang=host_lang,
                    payload_lang=payload_lang,
                )
            ]
        )
        runner = _FakeRunner(
            labels=[host_id] * len(content),
            probs=[
                self._prob_row(host_lang, 0.90, python=0.10),
                self._prob_row(host_lang, 0.93, python=0.07),
                self._prob_row(host_lang, 0.01, python=0.99),
                self._prob_row(host_lang, 0.96, python=0.04),
                self._prob_row(host_lang, 0.97, python=0.03),
            ],
        )

        metrics = evalmod.evaluate_task(
            "mal_injection",
            "test",
            dataset,
            runner,
            min_run_chars=1,
        )

        soft_correct = metrics.extras["mal_payload_detection"]["soft_detection"]["correct_detection"]
        self.assertEqual(soft_correct["truth_chars"], 4)
        self.assertEqual(soft_correct["matched_chars"], 2)
        self.assertEqual(soft_correct["detected"], 1)


if __name__ == "__main__":
    unittest.main()
