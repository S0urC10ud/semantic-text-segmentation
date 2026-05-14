from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magika_label_map import (
    MAGIKA_EXPECTED_ALIAS_GROUPS,
    OTHER_THESIS_LABEL,
    canonical_label,
    load_and_validate_installed_magika_mapping,
    magika_label_to_thesis_label,
    validate_magika_label_inventory,
)


class TestMagikaLabelMap(unittest.TestCase):
    def test_canonical_label_matches_downloader_aliases(self) -> None:
        self.assertEqual(canonical_label("C"), "c_family")
        self.assertEqual(canonical_label("cpp"), "c_family")
        self.assertEqual(canonical_label("c#"), "csharp")
        self.assertEqual(canonical_label("js"), "javascript")
        self.assertEqual(canonical_label("ts"), "typescript")
        self.assertEqual(canonical_label("batch"), "batchfile")
        self.assertEqual(canonical_label("vb.net"), "visual_basic")
        self.assertEqual(canonical_label("po"), "gettext_catalog")
        self.assertEqual(canonical_label("latex"), "tex")
        self.assertEqual(canonical_label("rst"), "restructuredtext")

    def test_magika_label_to_thesis_label_handles_tricky_aliases(self) -> None:
        cases = {
            "javascript": "javascript_typescript",
            "typescript": "javascript_typescript",
            "js": "javascript_typescript",
            "ts": "javascript_typescript",
            "c": "c_family",
            "cpp": "c_family",
            "shell": "shell",
            "batch": "shell",
            "batchfile": "shell",
            "vba": "visual_basic",
            "vb.net": "visual_basic",
            "latex": "tex",
            "po": "gettext_catalog",
            "txt": "text",
            "randomtxt": "text",
            "rst": "restructuredtext",
            "cs": "csharp",
            "randombytes": OTHER_THESIS_LABEL,
            "unknown": OTHER_THESIS_LABEL,
            "handlebars": OTHER_THESIS_LABEL,
        }
        for raw_label, expected in cases.items():
            with self.subTest(raw_label=raw_label):
                self.assertEqual(magika_label_to_thesis_label(raw_label), expected)

    def test_validate_magika_label_inventory_flags_missing_required_group(self) -> None:
        raw_labels = {
            alias
            for thesis_label, aliases in MAGIKA_EXPECTED_ALIAS_GROUPS.items()
            if thesis_label != "javascript_typescript"
            for alias in aliases
        }
        with self.assertRaisesRegex(RuntimeError, "javascript_typescript"):
            validate_magika_label_inventory(raw_labels)

    def test_validate_magika_label_inventory_routes_unsupported_labels_to_other(self) -> None:
        raw_labels = {
            alias
            for aliases in MAGIKA_EXPECTED_ALIAS_GROUPS.values()
            for alias in aliases
        }
        raw_labels.update({"handlebars", "zip", "png"})
        validation = validate_magika_label_inventory(
            raw_labels,
            package_version="test",
            model_name="dummy",
        )
        self.assertIn("handlebars", validation.unsupported_labels)
        self.assertIn("zip", validation.unsupported_labels)
        self.assertIn("png", validation.unsupported_labels)
        self.assertIn("javascript_typescript", validation.supported_labels)
        self.assertIn("text", validation.supported_labels)

    def test_installed_magika_inventory_matches_locked_contract(self) -> None:
        validation = load_and_validate_installed_magika_mapping()
        self.assertTrue(validation.package_version)
        self.assertTrue(validation.model_name)
        self.assertIn("javascript", validation.installed_labels)
        self.assertIn("typescript", validation.installed_labels)
        self.assertIn("randomtxt", validation.installed_labels)
        self.assertIn("javascript_typescript", validation.supported_labels)
        self.assertIn("text", validation.supported_labels)
        for raw_label in validation.unsupported_labels:
            with self.subTest(raw_label=raw_label):
                self.assertEqual(magika_label_to_thesis_label(raw_label), OTHER_THESIS_LABEL)


if __name__ == "__main__":
    unittest.main()
