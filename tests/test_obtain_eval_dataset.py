from __future__ import annotations

import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import evaluation.obtain_eval_dataset as obtain_eval_dataset


def _fragment_content(lang: str, idx: int) -> str:
    if lang == "python":
        return f"alpha = beta\nprint(alpha + {idx})\n"
    if lang == "sql":
        return f"SELECT a{idx};\n"
    if lang == "shell":
        return f"echo alpha{idx}\nprintf '%s\\n' done\n"
    if lang == "ruby":
        return f"value = alpha{idx}\nputs value\n"
    if lang == "yaml":
        return f"name: alpha_{idx}\nenabled: true\n"
    if lang == "text":
        return f"This is paragraph number {idx} with enough words for prose mixing.\n"
    return f"value {idx}\n"


def _fragments(*langs: str, count: int = 4) -> dict[str, list[obtain_eval_dataset.Fragment]]:
    out: dict[str, list[obtain_eval_dataset.Fragment]] = {}
    for lang in langs:
        out[lang] = [
            obtain_eval_dataset.Fragment(
                lang=lang,
                content=_fragment_content(lang, idx),
                uid=f"{lang}-{idx}",
                extra_meta=None,
            )
            for idx in range(count)
        ]
    return out


def _monitor_docs(*langs: str, count: int = 3) -> list[obtain_eval_dataset.MonitorDoc]:
    docs: list[obtain_eval_dataset.MonitorDoc] = []
    for lang in langs:
        for idx in range(count):
            prefix = f"{lang} host section {idx}\n"
            middle = "shared bridge text\n"
            suffix = f"{lang} tail section {idx}\n"
            content = prefix + middle + suffix
            docs.append(
                obtain_eval_dataset.MonitorDoc(
                    declared_lang=lang,
                    content=content,
                    segments=[
                        {"label": lang, "char_start": 0, "char_end": len(prefix)},
                        {
                            "label": "text",
                            "char_start": len(prefix),
                            "char_end": len(prefix) + len(middle),
                        },
                        {
                            "label": lang,
                            "char_start": len(prefix) + len(middle),
                            "char_end": len(content),
                        },
                    ],
                    uid=f"{lang}-doc-{idx}",
                    source="unit_test",
                    extra_meta=None,
                )
            )
    return docs


def _mixed_monitor_doc(
    declared_lang: str,
    pieces: list[tuple[str, str]],
    *,
    uid: str,
) -> obtain_eval_dataset.MonitorDoc:
    content_parts: list[str] = []
    segments: list[dict] = []
    cursor = 0
    for label, text in pieces:
        content_parts.append(text)
        start = cursor
        cursor += len(text)
        segments.append({"label": label, "char_start": start, "char_end": cursor})
    return obtain_eval_dataset.MonitorDoc(
        declared_lang=declared_lang,
        content="".join(content_parts),
        segments=segments,
        uid=uid,
        source="unit_test",
        extra_meta=None,
    )


def _segments_list(example: dict) -> list[dict]:
    segments = example["segments"]
    if isinstance(segments, dict):
        labels = list(segments.get("label", []))
        starts = list(segments.get("char_start", []))
        ends = list(segments.get("char_end", []))
        out: list[dict] = []
        for idx, label in enumerate(labels):
            out.append(
                {
                    "label": str(label),
                    "char_start": int(starts[idx]),
                    "char_end": int(ends[idx]),
                }
            )
        return out
    return [dict(seg) for seg in segments]


def _labels_by_char(example: dict) -> list[str]:
    content = str(example["content"])
    labels = [""] * len(content)
    for seg in _segments_list(example):
        start = int(seg["char_start"])
        end = int(seg["char_end"])
        for pos in range(start, min(end, len(labels))):
            labels[pos] = str(seg["label"])
    return labels


class TestObtainEvalDatasetBalancing(unittest.TestCase):
    def test_sequence_pair_balances_second_role_counts(self) -> None:
        fragments = _fragments("python", "sql", "yaml", count=5)

        task, examples, _ = obtain_eval_dataset._build_pair_dataset(
            fragments,
            per_label=4,
            rng=random.Random(7),
        )

        self.assertEqual(task, "sequence_pair")
        self.assertEqual(len(examples), 12)

        second_counts = Counter()
        for example in examples:
            meta = json.loads(example["metadata_json"])
            second_counts[str(meta["second_lang"])] += 1

        self.assertEqual(set(second_counts), {"python", "sql", "yaml"})
        self.assertLessEqual(max(second_counts.values()) - min(second_counts.values()), 1)

    def test_injection_dataset_balances_donor_languages(self) -> None:
        fragments = _fragments("python", "sql", "shell", "ruby", count=4)

        task, examples, _ = obtain_eval_dataset._build_injection_dataset(
            fragments,
            per_label=3,
            bucket_name="4_15",
            min_bytes=4,
            max_bytes=15,
            rng=random.Random(11),
        )

        self.assertEqual(task, "needle_4_15")
        self.assertEqual(len(examples), 12)

        donor_counts = Counter()
        for example in examples:
            meta = json.loads(example["metadata_json"])
            donor_counts[str(meta["donor_lang"])] += 1

        self.assertEqual(set(donor_counts), {"python", "sql", "shell", "ruby"})
        self.assertLessEqual(max(donor_counts.values()) - min(donor_counts.values()), 1)

    def test_injection_dataset_records_styles_and_needle_only_span(self) -> None:
        fragments = _fragments("python", "sql", "shell", "ruby", count=8)

        task, examples, _ = obtain_eval_dataset._build_injection_dataset(
            fragments,
            per_label=8,
            bucket_name="4_15",
            min_bytes=4,
            max_bytes=15,
            rng=random.Random(23),
        )

        self.assertEqual(task, "needle_4_15")
        self.assertGreaterEqual(len(examples), 16)

        styles = Counter()
        expected_styles = {
            "raw_random",
            "line_or_tab_prefixed",
            "control_wrapped",
        }
        for example in examples:
            meta = json.loads(example["metadata_json"])
            style = str(meta["injection_style"])
            host_lang = str(meta["host_lang"])
            donor_lang = str(meta["donor_lang"])
            needle_start = int(meta["needle_char_start"])
            needle_end = int(meta["needle_char_end"])
            insertion_char = int(meta["insertion_char"])
            context_prefix = str(meta.get("context_prefix", ""))
            context_suffix = str(meta.get("context_suffix", ""))
            content = str(example["content"])
            labels = _labels_by_char(example)

            styles[style] += 1
            self.assertIn(style, expected_styles)
            self.assertNotEqual(host_lang, donor_lang)
            self.assertTrue(0 <= needle_start < needle_end <= len(content))
            self.assertEqual(needle_start, insertion_char + len(context_prefix))

            if context_prefix:
                self.assertEqual(content[needle_start - len(context_prefix):needle_start], context_prefix)
            if context_suffix:
                self.assertEqual(content[needle_end:needle_end + len(context_suffix)], context_suffix)

            if style == "raw_random":
                self.assertEqual(context_prefix, "")
                self.assertEqual(context_suffix, "")
            elif style == "line_or_tab_prefixed":
                self.assertIn(context_prefix, ("\n", "\t"))
                self.assertEqual(context_suffix, "")
            elif style == "control_wrapped":
                self.assertTrue(context_prefix or context_suffix)

            for pos in range(needle_start, needle_end):
                self.assertEqual(labels[pos], donor_lang)

            for pos in range(needle_start - len(context_prefix), needle_start):
                if 0 <= pos < len(labels):
                    self.assertEqual(labels[pos], host_lang)
            for pos in range(needle_end, needle_end + len(context_suffix)):
                if 0 <= pos < len(labels):
                    self.assertEqual(labels[pos], host_lang)

        self.assertTrue(expected_styles.issubset(set(styles)))

    def test_injection_dataset_sources_needles_from_line_start(self) -> None:
        fragments = {
            "python": [
                obtain_eval_dataset.Fragment(
                    lang="python",
                    content="host_value = 1\nprint(host_value)\n",
                    uid="python-host-0",
                    extra_meta=None,
                )
            ],
            "sql": [
                obtain_eval_dataset.Fragment(
                    lang="sql",
                    content="    SELECT id\n        FROM t\n",
                    uid="sql-donor-0",
                    extra_meta=None,
                )
            ],
        }

        task, examples, _ = obtain_eval_dataset._build_injection_dataset(
            fragments,
            per_label=1,
            bucket_name="12_20",
            min_bytes=12,
            max_bytes=20,
            rng=random.Random(5),
        )

        self.assertEqual(task, "needle_12_20")
        example = next(
            ex
            for ex in examples
            if json.loads(ex["metadata_json"])["host_lang"] == "python"
        )
        meta = json.loads(example["metadata_json"])
        needle_start = int(meta["needle_char_start"])
        needle_end = int(meta["needle_char_end"])
        needle_text = str(example["content"])[needle_start:needle_end]

        self.assertEqual(str(meta["donor_lang"]), "sql")
        self.assertEqual(int(meta["donor_start_line"]), 0)
        self.assertEqual(int(meta["donor_end_line"]), 2)
        self.assertEqual(needle_text, "SELECT id\n        FROM t\n")
        self.assertFalse(needle_text.startswith((" ", "\t")))
        self.assertTrue(needle_text.splitlines()[1].startswith("        "))

    def test_markup_mix_uses_wrapper_labels_and_balances_other_langs(self) -> None:
        fragments = _fragments("text", "python", "sql", "yaml", count=4)

        task, examples, _ = obtain_eval_dataset._build_markdown_dataset(
            fragments,
            per_label=2,
            rng=random.Random(13),
            task_name="markdown_mix",
            wrapper_label="markdown",
            markup_style="markdown",
        )

        self.assertEqual(task, "markdown_mix")
        self.assertEqual(len(examples), 6)

        host_counts = Counter()
        other_counts = Counter()
        for example in examples:
            meta = json.loads(example["metadata_json"])
            host_counts[str(meta["host_lang"])] += 1
            other_counts[str(meta["other_lang"])] += 1
            self.assertIn("markdown", set(example["segments"]["label"]))
            self.assertNotIn("text", set(example["segments"]["label"]))

        self.assertEqual(host_counts, Counter({"python": 2, "sql": 2, "yaml": 2}))
        self.assertLessEqual(max(other_counts.values()) - min(other_counts.values()), 1)

        rst_task, rst_examples, _ = obtain_eval_dataset._build_markdown_dataset(
            fragments,
            per_label=1,
            rng=random.Random(13),
            task_name="restructuredtext_mix",
            wrapper_label="restructuredtext",
            markup_style="restructuredtext",
        )

        self.assertEqual(rst_task, "restructuredtext_mix")
        self.assertEqual(len(rst_examples), 3)
        for example in rst_examples:
            self.assertIn("restructuredtext", set(example["segments"]["label"]))


class TestObtainEvalDatasetMonitorWindows(unittest.TestCase):
    def test_pure_fragments_require_more_than_95_percent_host_chars(self) -> None:
        docs = [
            obtain_eval_dataset.MonitorDoc(
                declared_lang="python",
                content=("P" * 96) + ("T" * 4),
                segments=[
                    {"label": "python", "char_start": 0, "char_end": 96},
                    {"label": "text", "char_start": 96, "char_end": 100},
                ],
                uid="python-good",
                source="unit_test",
                extra_meta=None,
            ),
            obtain_eval_dataset.MonitorDoc(
                declared_lang="python",
                content=("P" * 95) + ("T" * 5),
                segments=[
                    {"label": "python", "char_start": 0, "char_end": 95},
                    {"label": "text", "char_start": 95, "char_end": 100},
                ],
                uid="python-borderline",
                source="unit_test",
                extra_meta=None,
            ),
        ]

        task, examples, _ = obtain_eval_dataset._build_pure_dataset_from_monitor(
            docs,
            per_label=2,
            rng=random.Random(19),
        )

        self.assertEqual(task, "pure_fragments")
        self.assertEqual(len(examples), 1)

        example = examples[0]
        meta = json.loads(example["metadata_json"])
        self.assertEqual(str(meta["host_uid"]), "python-good")
        self.assertGreater(float(meta["host_label_ratio"]), 0.95)

        labels = _labels_by_char(example)
        host_chars = sum(1 for label in labels if label == "python")
        self.assertGreater(host_chars / len(labels), 0.95)

    def test_malicious_payloads_are_balanced_across_languages(self) -> None:
        docs = _monitor_docs("sql", "rust", "yaml", count=3)

        task, examples, _ = obtain_eval_dataset._build_malicious_dataset_from_monitor(
            docs,
            per_label=3,
            rng=random.Random(17),
        )

        self.assertEqual(task, "mal_injection")
        self.assertEqual(len(examples), 9)

        payload_counts = Counter()
        for example in examples:
            meta = json.loads(example["metadata_json"])
            payload_counts[str(meta["payload_lang"])] += 1
            self.assertLessEqual(len(example["content"]), obtain_eval_dataset.cfg.MODEL_WINDOW_BYTES + 1200)

        self.assertEqual(sum(payload_counts.values()), 9)
        self.assertLessEqual(max(payload_counts.values()) - min(payload_counts.values()), 1)

    def test_malicious_payloads_record_style_and_payload_only_span(self) -> None:
        docs = _monitor_docs("sql", "rust", "yaml", count=8)

        task, examples, _ = obtain_eval_dataset._build_malicious_dataset_from_monitor(
            docs,
            per_label=8,
            rng=random.Random(23),
        )

        self.assertEqual(task, "mal_injection")
        self.assertGreaterEqual(len(examples), 12)

        styles = Counter()
        expected_styles = {
            "raw_random",
            "line_or_tab_prefixed",
            "control_wrapped",
        }
        for example in examples:
            meta = json.loads(example["metadata_json"])
            style = str(meta["injection_style"])
            payload_lang = str(meta["payload_lang"])
            payload_start = int(meta["payload_char_start"])
            payload_end = int(meta["payload_char_end"])
            insertion_char = int(meta["insertion_char"])
            context_prefix = str(meta.get("context_prefix", ""))
            context_suffix = str(meta.get("context_suffix", ""))
            content = str(example["content"])
            labels = _labels_by_char(example)

            styles[style] += 1
            self.assertIn(style, expected_styles)
            self.assertTrue(0 <= payload_start < payload_end <= len(content))
            self.assertEqual(payload_start, insertion_char + len(context_prefix))

            if context_prefix:
                self.assertEqual(content[payload_start - len(context_prefix):payload_start], context_prefix)
            if context_suffix:
                self.assertEqual(content[payload_end:payload_end + len(context_suffix)], context_suffix)

            if style == "raw_random":
                self.assertEqual(context_prefix, "")
                self.assertEqual(context_suffix, "")
            elif style == "line_or_tab_prefixed":
                self.assertIn(context_prefix, ("\n", "\t"))
                self.assertEqual(context_suffix, "")
            elif style == "control_wrapped":
                self.assertTrue(context_prefix or context_suffix)

            for pos in range(payload_start, payload_end):
                self.assertEqual(labels[pos], payload_lang)

            for pos in range(payload_start - len(context_prefix), payload_start):
                if 0 <= pos < len(labels):
                    self.assertNotEqual(labels[pos], payload_lang)
            for pos in range(payload_end, payload_end + len(context_suffix)):
                if 0 <= pos < len(labels):
                    self.assertNotEqual(labels[pos], payload_lang)

        self.assertTrue(expected_styles.issubset(set(styles)))


class TestSegmentBackedSyntheticTasks(unittest.TestCase):
    def test_monitor_injection_preserves_mixed_inserted_region_labels(self) -> None:
        docs = [
            _mixed_monitor_doc(
                "yaml",
                [("yaml", "name: value\n")],
                uid="yaml-host",
            ),
        ]
        for idx in range(12):
            docs.append(
                _mixed_monitor_doc(
                    "python",
                    [("python", "alpha12"), ("shell", "XY"), ("python", "\n")],
                    uid=f"python-donor-{idx}",
                )
            )

        task, examples, _, support = obtain_eval_dataset._build_injection_dataset_from_monitor(
            docs,
            per_label=1,
            bucket_name="4_15",
            min_visible=4,
            max_visible=15,
            rng=random.Random(5),
        )

        self.assertEqual(task, "needle_4_15")
        example = next(ex for ex in examples if json.loads(ex["metadata_json"])["host_lang"] == "yaml")
        meta = json.loads(example["metadata_json"])
        labels = _labels_by_char(example)
        start = int(meta["inserted_char_start"])
        end = int(meta["inserted_char_end"])

        self.assertEqual(str(meta["donor_lang"]), "python")
        self.assertEqual(set(labels[start:end]), {"python", "shell"})
        self.assertIn("python", list(meta["inserted_source_labels"]))
        self.assertIn("shell", list(meta["inserted_source_labels"]))
        self.assertEqual(int(support["actual_count"]), len(examples))
        self.assertGreaterEqual(int(support["requested_count"]), int(support["actual_count"]))

    def test_monitor_injection_qualifies_donors_by_visible_support_and_recovers_short_svg(self) -> None:
        docs = _monitor_docs("python", "sql", count=4)
        for idx in range(4):
            docs.append(
                _mixed_monitor_doc(
                    "yaml",
                    [("yaml", f"name: alpha_beta_gamma_delta_value_{idx}\n")],
                    uid=f"yaml-donor-{idx}",
                )
            )
            docs.append(
                _mixed_monitor_doc(
                    "svg",
                    [("svg", f"<svg><path d=\"M{'A' * 72}{idx}\"/></svg>\n")],
                    uid=f"svg-donor-{idx}",
                )
            )
        docs.append(
            _mixed_monitor_doc(
                "csharp",
                [("csharp", "using System; class X { string Name; }\n")],
                uid="csharp-short",
            )
        )
        docs.append(
            _mixed_monitor_doc(
                "text",
                [("text", "This plain text block should remain excluded as a donor needle.\n")],
                uid="text-excluded",
            )
        )

        task, examples, _, support = obtain_eval_dataset._build_injection_dataset_from_monitor(
            docs,
            per_label=3,
            bucket_name="32_63",
            min_visible=32,
            max_visible=63,
            rng=random.Random(9),
        )

        self.assertEqual(task, "needle_32_63")

        donor_counts = Counter()
        for example in examples:
            meta = json.loads(example["metadata_json"])
            donor_counts[str(meta["donor_lang"])] += 1

        self.assertIn("yaml", donor_counts)
        self.assertIn("svg", donor_counts)
        self.assertNotIn("text", donor_counts)

        self.assertIn("yaml", support["qualified_donor_labels"])
        self.assertIn("svg", support["qualified_donor_labels"])
        self.assertNotIn("csharp", support["qualified_donor_labels"])
        self.assertGreaterEqual(
            int(support["donor_candidate_visible_support_by_donor_label"]["yaml"]),
            100,
        )
        self.assertGreaterEqual(
            int(support["donor_candidate_visible_support_by_donor_label"]["svg"]),
            100,
        )
        self.assertNotIn("csharp", support["donor_candidate_visible_support_by_donor_label"])
        self.assertGreater(
            int(support["donor_selected_visible_support_by_donor_label"].get("svg", 0)),
            0,
        )

    def test_monitor_sequence_pair_records_exact_region_metadata(self) -> None:
        docs = [
            _mixed_monitor_doc(
                "python",
                [("python", "alphaxy"), ("shell", "ZZ"), ("python", "\n")],
                uid="python-seq",
            ),
            _mixed_monitor_doc(
                "shell",
                [("shell", "SHELLxx"), ("sql", "QQ"), ("shell", "\n")],
                uid="shell-seq",
            ),
        ]

        task, examples, _, support = obtain_eval_dataset._build_pair_dataset_from_monitor(
            docs,
            per_label=1,
            rng=random.Random(7),
        )

        self.assertEqual(task, "sequence_pair")
        example = next(ex for ex in examples if json.loads(ex["metadata_json"])["first_lang"] == "python")
        meta = json.loads(example["metadata_json"])
        regions = meta["sequence_regions"]
        labels = _labels_by_char(example)
        first = regions["first"]
        second = regions["second"]

        self.assertLess(int(first["char_start"]), int(first["char_end"]))
        self.assertLess(int(second["char_start"]), int(second["char_end"]))
        self.assertGreater(len(set(labels[int(first["char_start"]):int(first["char_end"])])), 1)
        self.assertGreater(len(set(labels[int(second["char_start"]):int(second["char_end"])])), 1)
        self.assertIn("shortfall_by_anchor_label", support)

    def test_monitor_markdown_blocks_keep_exact_region_spans_and_support(self) -> None:
        docs = [
            _mixed_monitor_doc(
                "python",
                [("python", "alphaxy"), ("shell", "ZZ"), ("python", "\n")],
                uid="python-md",
            ),
            _mixed_monitor_doc(
                "shell",
                [("shell", "SHELLxx"), ("sql", "QQ"), ("shell", "\n")],
                uid="shell-md",
            ),
            _mixed_monitor_doc(
                "text",
                [("text", "Some prose paragraph for markdown mixing.\n")],
                uid="text-md",
            ),
        ]

        task, examples, _, support = obtain_eval_dataset._build_markdown_dataset_from_regions(
            docs,
            per_label=1,
            rng=random.Random(13),
            task_name="markdown_mix",
            wrapper_label="markdown",
            markup_style="markdown",
        )

        self.assertEqual(task, "markdown_mix")
        example = examples[0]
        meta = json.loads(example["metadata_json"])
        blocks = meta["markdown_blocks"]
        self.assertTrue(blocks)
        labels = _labels_by_char(example)
        first_block = blocks[0]
        block_labels = set(labels[int(first_block["char_start"]):int(first_block["char_end"])])
        self.assertGreater(len(block_labels), 1)
        self.assertEqual(str(first_block["truth_mode"]), "exact_region")
        self.assertIn("requested_count", support)
        self.assertIn("actual_by_anchor_label", support)

    def test_monitor_markdown_inline_regions_are_common_and_longer(self) -> None:
        def _inline_docs(lang: str, count: int) -> list[obtain_eval_dataset.MonitorDoc]:
            docs: list[obtain_eval_dataset.MonitorDoc] = []
            for idx in range(count):
                docs.append(
                    _mixed_monitor_doc(
                        lang,
                        [
                            (
                                lang,
                                f"{lang}_command_{idx}_alpha beta gamma delta epsilon theta lambda omega value_{idx}\n",
                            ),
                            ("text", "tip\n"),
                            (
                                lang,
                                f"{lang}_followup_{idx}_important token stream repeated content sample_{idx} more data here\n",
                            ),
                        ],
                        uid=f"{lang}-inline-{idx}",
                    )
                )
            return docs

        docs = _inline_docs("python", 6) + _inline_docs("shell", 6) + _inline_docs("sql", 6)

        task, examples, _, support = obtain_eval_dataset._build_markdown_dataset_from_regions(
            docs,
            per_label=2,
            rng=random.Random(29),
            task_name="markdown_mix",
            wrapper_label="markdown",
            markup_style="markdown",
        )

        self.assertEqual(task, "markdown_mix")
        self.assertEqual(len(examples), 6)
        self.assertEqual(int(support["inline_target_count"]), 6)
        self.assertGreaterEqual(int(support["inline_actual_count"]), 6)

        total_inline_blocks = 0
        for example in examples:
            meta = json.loads(example["metadata_json"])
            inline_blocks = meta.get("inline_blocks", [])
            self.assertTrue(inline_blocks)
            total_inline_blocks += len(inline_blocks)
            content = str(example["content"])
            for block in inline_blocks:
                start = int(block["char_start"])
                end = int(block["char_end"])
                snippet = content[start:end]
                visible = sum(ch not in {" ", "\t", "\n"} for ch in snippet)
                self.assertGreaterEqual(visible, obtain_eval_dataset._MARKDOWN_INLINE_MIN_VISIBLE)
                self.assertNotIn("\n", snippet)
                self.assertIn(str(block["wrapper"]), {"inline_backtick", "html_code"})

        self.assertGreaterEqual(total_inline_blocks, 6)

        rst_task, rst_examples, _, rst_support = obtain_eval_dataset._build_markdown_dataset_from_regions(
            docs,
            per_label=2,
            rng=random.Random(29),
            task_name="restructuredtext_mix",
            wrapper_label="restructuredtext",
            markup_style="restructuredtext",
        )

        self.assertEqual(rst_task, "restructuredtext_mix")
        self.assertEqual(len(rst_examples), 6)
        self.assertEqual(int(rst_support["inline_target_count"]), 6)
        self.assertGreaterEqual(int(rst_support["inline_actual_count"]), 6)
        for example in rst_examples:
            meta = json.loads(example["metadata_json"])
            inline_blocks = meta.get("inline_blocks", [])
            self.assertTrue(inline_blocks)
            for block in inline_blocks:
                self.assertIn(str(block["wrapper"]), {"inline_literal", "inline_role"})


if __name__ == "__main__":
    unittest.main()
