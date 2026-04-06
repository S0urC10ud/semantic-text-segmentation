from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "train"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import viewers.core as core


MANIFEST_PATH = ROOT / "viewers" / "content_type_segmentor_static" / "assets" / "model_manifest.json"
CHECKPOINT_PATH = ROOT / "checkpoints" / "sweeps" / "sfullfiles4.msgpack"

TIREX_SAMPLE = """---


# TiRex v2

## Setup Environment
Environment management is done with **conda** and **uv**, where conda takes care of the system level dependencies (for example, CUDA
and gcc) and uv manages both the Python interpreter and all Python packages.

### Automated Setup
The easiest way to set up the environment is to use the provided setup script for your shell:

**Bash:**
```bash
source ./env_setup.sh [ENV_NAME] [cuda12x|cuda=12x|--cuda=12x]
```

where you can either use `cuda128` for installing environment with cuda 12.8 or `cuda126` for cuda 12.6. Default is `12.8` **Always check cuda version of the node for compatibility!** NVIDIA Blackwell GPUs require at least `12.8` but older nodes with Hopper/Ampere cards may not support it fully, in those cases `12.6` provides best compatibility.

We also provide a `fish` script but please note it may not be maintained as often as the Bash script. Bash remains the recommended version.

**Fish:**
```fish
source ./env_setup.fish [ENV_NAME]
```

These scripts will:
- Create/update the conda environment from `environment_28.yaml` named `<ENV_NAME>` or `tirex-cu128` per default (or `environment_26.yaml` and `tirex-cu126` respectively)
- Automatically download and install 3.11 <= Python < 3.14 by way of uv
- Install all Python dependencies from `pyproject.toml`
- Configure automatic activation of the uv virtual environment when activating the conda environment

After running the setup script once, activate the environment with:
```bash
conda activate tirex-cu128
```
Both the conda environment and the uv virtual environment will be activated automatically.

Note that you may need to sync your environment with the proper versions using

`uv sync --group cuda128` or `cuda126` depending on your installed version.

### Manual Setup
If you prefer to set up the environment manually:

1. Create the conda environment (system dependencies only):
```bash
conda env create -f ./environment_28.yaml
conda activate tirex-cu128
```

2. Sync Python packages (uv will automatically download Python 3.11 if needed):
```bash
uv sync --group cuda128
```

**Note:** By default, the `.venv` directory is created in the repository root.
To control the virtual environment's location, set the environment variable
`UV_PYTHON_INSTALL_DIR` before running the setup script or `uv sync`:
```bash
UV_PYTHON_INSTALL_DIR=/path/to/venv source ./env_setup.sh
```

## Training
To begin training, some user / organization specific variables need to be set.
To create a config for your specific use-case, create the file ```conf/org/YOUR_ORG.yaml```.
You can base it on ```conf/org/jku.yaml```.
All configuration is based on **hydra**.

### Shared HF-Cache

<span style="color: red;">IMPORTANT for NXAI servers!</span>
"""


def _prob_row(*values: float) -> dict[str, float]:
    return {str(idx): float(value) for idx, value in enumerate(values)}


def _count_short_interior_runs(labels: list[int], *, min_run_chars: int) -> int:
    runs = core._build_label_runs(labels)
    return sum(
        1
        for idx, (start, end, _label) in enumerate(runs)
        if 0 < idx < (len(runs) - 1) and (end - start) < int(min_run_chars)
    )


class TestViewerPostprocess(unittest.TestCase):
    def test_min_run_collapses_short_interior_island(self) -> None:
        text = "AAAABBBAAAA"
        labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        probs = [_prob_row(0.95, 0.05) for _ in range(4)]
        probs.extend(_prob_row(0.10, 0.90) for _ in range(3))
        probs.extend(_prob_row(0.95, 0.05) for _ in range(4))

        result = core._postprocess_char_labels(
            text,
            labels,
            probs,
            min_run_chars=5,
            boundary_snap_max_shift=0,
        )

        self.assertEqual(result, [0] * len(text))

    def test_min_run_preserves_short_edge_runs(self) -> None:
        text = "BBBBAAAAAAA"
        labels = [1, 1, 1, 1] + [0] * 7
        probs = [_prob_row(0.05, 0.95) for _ in range(4)]
        probs.extend(_prob_row(0.95, 0.05) for _ in range(7))

        result = core._postprocess_char_labels(
            text,
            labels,
            probs,
            min_run_chars=5,
            boundary_snap_max_shift=0,
        )

        self.assertEqual(result, labels)

    def test_min_run_treats_other_like_any_other_label(self) -> None:
        text = "AAAAOOOAAAA"
        other_id = 2
        labels = [0, 0, 0, 0, other_id, other_id, other_id, 0, 0, 0, 0]
        probs = [_prob_row(0.95, 0.03, 0.02) for _ in range(4)]
        probs.extend(_prob_row(0.10, 0.10, 0.80) for _ in range(3))
        probs.extend(_prob_row(0.95, 0.03, 0.02) for _ in range(4))

        result = core._postprocess_char_labels(
            text,
            labels,
            probs,
            min_run_chars=5,
            boundary_snap_max_shift=0,
        )

        self.assertEqual(result, [0] * len(text))

    def test_boundary_snap_moves_by_two_chars_toward_delimiter(self) -> None:
        text = "<abX"
        labels = [0, 0, 0, 1]
        probs = [
            _prob_row(0.90, 0.10),
            _prob_row(0.52, 0.50),
            _prob_row(0.51, 0.50),
            _prob_row(0.05, 0.95),
        ]

        result = core._postprocess_char_labels(
            text,
            labels,
            probs,
            min_run_chars=1,
            boundary_snap_max_shift=2,
        )

        self.assertEqual(result, [0, 1, 1, 1])

    def test_boundary_snap_respects_strong_identifier_evidence(self) -> None:
        text = "<abX"
        labels = [0, 0, 0, 1]
        probs = [
            _prob_row(0.90, 0.10),
            _prob_row(0.95, 0.01),
            _prob_row(0.95, 0.01),
            _prob_row(0.05, 0.95),
        ]

        result = core._postprocess_char_labels(
            text,
            labels,
            probs,
            min_run_chars=1,
            boundary_snap_max_shift=2,
        )

        self.assertEqual(result, labels)

    def test_boundary_snap_does_not_cross_newline(self) -> None:
        text = "a\nb"
        labels = [0, 0, 1]
        probs = [
            _prob_row(0.95, 0.05),
            _prob_row(0.60, 0.35),
            _prob_row(0.05, 0.95),
        ]

        result = core._postprocess_char_labels(
            text,
            labels,
            probs,
            min_run_chars=1,
            boundary_snap_max_shift=2,
        )

        self.assertEqual(result, labels)

    def test_boundary_snap_does_not_steal_indentation_with_strong_same_line_support(self) -> None:
        text = ">\n  b"
        labels = [0, 0, 1, 1, 1]
        probs = [
            _prob_row(0.98, 0.02),
            _prob_row(0.98, 0.02),
            _prob_row(0.05, 0.95),
            _prob_row(0.05, 0.95),
            _prob_row(0.05, 0.95),
        ]

        result, trace = core._postprocess_char_labels_with_trace(
            text,
            labels,
            probs,
            min_run_chars=1,
            boundary_snap_max_shift=2,
        )

        self.assertEqual(result, labels)
        self.assertEqual(trace["descriptions"][2], "")

    def test_boundary_snap_does_not_shrink_short_middle_run_before_min_run(self) -> None:
        text = "tokenizing */<tag>"
        labels = [0] * 10 + [1] * 3 + [0] * 5
        probs = [_prob_row(0.39, 0.31) for _ in range(10)]
        probs.extend(
            [
                _prob_row(0.33, 0.37),  # g
                _prob_row(0.33, 0.37),  # space
                _prob_row(0.30, 0.49),  # *
            ]
        )
        probs.extend(_prob_row(0.55, 0.19) for _ in range(5))

        result, trace = core._postprocess_char_labels_with_trace(
            text,
            labels,
            probs,
            min_run_chars=5,
            boundary_snap_max_shift=2,
        )

        self.assertEqual(result, [0] * len(text))
        self.assertNotIn("boundary snap", trace["descriptions"][10])

    def test_boundary_snap_does_not_steal_strongly_supported_tag_opener(self) -> None:
        text = "x\n</"
        labels = [1, 1, 0, 0]
        probs = [
            _prob_row(0.08, 0.90),
            _prob_row(0.08, 0.90),
            _prob_row(0.93, 0.04),
            _prob_row(0.94, 0.03),
        ]

        result, trace = core._postprocess_char_labels_with_trace(
            text,
            labels,
            probs,
            min_run_chars=1,
            boundary_snap_max_shift=2,
        )

        self.assertEqual(result, labels)
        self.assertFalse(trace["changed_mask"][2])
        self.assertEqual(trace["descriptions"][2], "")

    def test_paired_delimiter_fill_expands_embedded_run_inside_quotes(self) -> None:
        text = 'x="alert(\'clicked\')">'
        labels = [0, 0, 0, 0] + [1] * 14 + [0, 0, 0]
        probs = [
            _prob_row(0.98, 0.02),  # x
            _prob_row(0.97, 0.03),  # =
            _prob_row(0.96, 0.04),  # opening quote
            _prob_row(0.54, 0.46),  # a (slightly host-leaning; local snap should not absorb this alone)
        ]
        probs.extend(_prob_row(0.08, 0.92) for _ in range(14))  # lert('clicked'
        probs.extend(
            [
                _prob_row(0.45, 0.55),  # )
                _prob_row(0.97, 0.03),  # closing quote
                _prob_row(0.98, 0.02),  # >
            ]
        )

        result = core._postprocess_char_labels(
            text,
            labels,
            probs,
            min_run_chars=1,
            boundary_snap_max_shift=2,
        )

        expected = [0, 0, 0] + [1] * 16 + [0, 0]
        self.assertEqual(result, expected)

    def test_postprocess_trace_marks_chars_changed_by_paired_fill(self) -> None:
        text = 'x="alert(\'clicked\')">'
        labels = [0, 0, 0, 0] + [1] * 14 + [0, 0, 0]
        probs = [
            _prob_row(0.98, 0.02),
            _prob_row(0.97, 0.03),
            _prob_row(0.96, 0.04),
            _prob_row(0.54, 0.46),
        ]
        probs.extend(_prob_row(0.08, 0.92) for _ in range(14))
        probs.extend(
            [
                _prob_row(0.45, 0.55),
                _prob_row(0.97, 0.03),
                _prob_row(0.98, 0.02),
            ]
        )

        result, trace = core._postprocess_char_labels_with_trace(
            text,
            labels,
            probs,
            min_run_chars=1,
            boundary_snap_max_shift=2,
        )

        self.assertEqual(result[3], 1)
        self.assertEqual(result[18], 1)
        self.assertTrue(bool(trace["changed_mask"][3]))
        self.assertTrue(bool(trace["changed_mask"][18]))
        self.assertTrue(bool(trace["descriptions"][3]))
        self.assertTrue(bool(trace["descriptions"][18]))

    def test_paired_delimiter_fill_relocks_quotes_through_host_noise(self) -> None:
        text = '<button class="btn" on<!--hi yanick-->click="alert(\'clicked\')">Click</button>'
        html_label = 0
        js_label = 1
        wrapped_text = '"alert(\'clicked\')"'
        open_quote = text.index(wrapped_text)
        inner_start = open_quote + 1
        close_quote = open_quote + len(wrapped_text) - 1
        labels = [html_label] * open_quote + [js_label] * (close_quote - open_quote) + [html_label] * (len(text) - close_quote)
        text_start = text.index(">Click")
        probs: list[dict[str, float]] = []
        for idx in range(len(text)):
            if inner_start <= idx < close_quote:
                probs.append(_prob_row(0.10, 0.82))
            elif idx == open_quote:
                probs.append(_prob_row(0.45, 0.55))
            elif idx == close_quote:
                probs.append(_prob_row(0.82, 0.10))
            elif idx >= text_start:
                probs.append(_prob_row(0.82, 0.08))
            else:
                probs.append(_prob_row(0.82, 0.08))

        result = core._postprocess_char_labels(
            text,
            labels,
            probs,
            min_run_chars=5,
            boundary_snap_max_shift=2,
        )

        self.assertEqual(result[open_quote], html_label)
        self.assertTrue(all(label == js_label for label in result[inner_start:close_quote]))
        self.assertEqual(result[close_quote], html_label)

    def test_json_run_surrounded_by_javascript_is_swallowed_into_javascript(self) -> None:
        text = 'const x = {"a":1};'
        js_label = 0
        json_label = 1
        json_start = text.index("{")
        json_end = text.index("}") + 1
        labels = [js_label] * json_start + [json_label] * (json_end - json_start) + [js_label] * (len(text) - json_end)
        probs = [_prob_row(0.70, 0.20) for _ in range(json_start)]
        probs.extend(_prob_row(0.25, 0.65) for _ in range(json_end - json_start))
        probs.extend(_prob_row(0.70, 0.20) for _ in range(len(text) - json_end))

        result, trace = core._postprocess_char_labels_with_trace(
            text,
            labels,
            probs,
            min_run_chars=5,
            boundary_snap_max_shift=2,
            local_host_rules=((json_label, js_label, "json"),),
        )

        self.assertEqual(result, [js_label] * len(text))
        self.assertTrue(bool(trace["descriptions"][json_start]))

    def test_markdown_structure_fill_normalizes_fence_tokens(self) -> None:
        text = "- ```bash\necho hi\n```"
        opening_prefix = "- ```bas"
        shell_part = "h\necho hi\n`"
        closing_suffix = "``"
        labels = [0] * len(opening_prefix) + [1] * len(shell_part) + [0] * len(closing_suffix)
        probs = [_prob_row(0.90, 0.05, 0.05) for _ in range(len(opening_prefix))]
        probs.extend(_prob_row(0.05, 0.90, 0.05) for _ in range(len(shell_part)))
        probs.extend(_prob_row(0.90, 0.05, 0.05) for _ in range(len(closing_suffix)))

        result, trace = core._postprocess_char_labels_with_trace(
            text,
            labels,
            probs,
            min_run_chars=5,
            boundary_snap_max_shift=2,
            markdown_label=0,
        )

        self.assertTrue(all(label == 0 for label in result[2:9]))
        self.assertTrue(all(label == 1 for label in result[9:18]))
        self.assertTrue(all(label == 0 for label in result[18:21]))
        self.assertIn("markdown structure fill", trace["descriptions"][8])
        self.assertIn("markdown structure fill", trace["descriptions"][18])

    def test_inline_triple_backticks_keep_wrappers_markdown_and_unify_body(self) -> None:
        text = "file ```conf/org/YOUR_ORG.yaml``` end"
        open_start = text.index("```")
        body_start = open_start + 3
        close_start = text.index("```", body_start)
        labels = [0] * len(text)
        for idx in range(body_start, close_start):
            labels[idx] = 1 if (idx - body_start) % 5 else 2
        for idx in range(close_start, len(text)):
            labels[idx] = 2
        probs = [_prob_row(0.90, 0.05, 0.05) for _ in range(len(text))]
        for idx in range(body_start, close_start):
            probs[idx] = _prob_row(0.10, 0.82, 0.08)
        for idx in range(close_start, len(text)):
            probs[idx] = _prob_row(0.10, 0.10, 0.80)

        result = core._postprocess_char_labels(
            text,
            labels,
            probs,
            min_run_chars=5,
            boundary_snap_max_shift=2,
            markdown_label=0,
        )

        self.assertTrue(all(label == 0 for label in result[open_start:body_start]))
        self.assertTrue(all(label == 1 for label in result[body_start:close_start]))
        self.assertTrue(all(label == 0 for label in result[close_start:close_start + 3]))

    def test_min_run_does_not_increase_short_interior_runs(self) -> None:
        text = "AAAABBBCDDDDEEFFFGGGG"
        labels = [0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6]
        probs = [_prob_row(0.90, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01) for _ in range(4)]
        probs.extend(_prob_row(0.02, 0.90, 0.02, 0.02, 0.02, 0.01, 0.01) for _ in range(3))
        probs.append(_prob_row(0.02, 0.02, 0.90, 0.01, 0.01, 0.02, 0.02))
        probs.extend(_prob_row(0.02, 0.02, 0.01, 0.90, 0.02, 0.02, 0.01) for _ in range(4))
        probs.extend(_prob_row(0.02, 0.02, 0.01, 0.02, 0.90, 0.02, 0.01) for _ in range(2))
        probs.extend(_prob_row(0.02, 0.02, 0.01, 0.02, 0.02, 0.90, 0.01) for _ in range(3))
        probs.extend(_prob_row(0.02, 0.02, 0.01, 0.02, 0.01, 0.02, 0.90) for _ in range(4))

        before = _count_short_interior_runs(labels, min_run_chars=5)
        result = core._postprocess_char_labels(
            text,
            labels,
            probs,
            min_run_chars=5,
            boundary_snap_max_shift=0,
        )
        after = _count_short_interior_runs(result, min_run_chars=5)

        self.assertLessEqual(after, before)

    def test_tirex_sample_integration_stabilizes_fences_and_important(self) -> None:
        if not MANIFEST_PATH.exists() or not CHECKPOINT_PATH.exists():
            self.skipTest("Viewer integration assets are unavailable")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        predictor = core.Predictor(
            ckpt_path=str(CHECKPOINT_PATH),
            num_classes=len(manifest["label_order"]),
            model_dim=int(manifest["model"]["d_model"]),
            channels=tuple(core.DEFAULT_CHANNELS),
            arch="mamba",
            mamba_layers=int(manifest["model"]["n_layers"]),
            mamba_d_state=int(manifest["model"]["d_state"]),
            mamba_expand=int(manifest["model"]["expand"]),
            mamba_dt_rank=int(manifest["model"]["dt_rank"]),
            mamba_conv=int(manifest["model"]["d_conv"]),
            mamba_bidirectional=bool(manifest["model"]["bidirectional"]),
            dtype_str=str(manifest["model"]["dtype"]),
            chunk=int(manifest["window_bytes"]),
            other_threshold=0.3,
            device="cpu",
            inference_backend="legacy",
        )
        segs, labels, _probs, _windows = predictor.segment_text(TIREX_SAMPLE, min_run_chars=5)
        label_order = [str(name) for name in manifest["label_order"]]
        label_by_id = {idx: name for idx, name in enumerate(label_order)}

        fence_start = TIREX_SAMPLE.index("```bash")
        fence_body_start = fence_start + len("```bash\n")
        fence_body_end = TIREX_SAMPLE.index("```", fence_body_start)
        closing_end = fence_body_end + 3
        important_start = TIREX_SAMPLE.index("IMPORTANT")
        important_end = TIREX_SAMPLE.index("</span>")

        self.assertTrue(all(label_by_id[labels[idx]] == "markdown" for idx in range(fence_start, fence_body_start)))
        self.assertTrue(all(label_by_id[labels[idx]] == "shell" for idx in range(fence_body_start, fence_body_end)))
        self.assertTrue(all(label_by_id[labels[idx]] == "markdown" for idx in range(fence_body_end, closing_end)))
        important_labels = {label_by_id[labels[idx]] for idx in range(important_start, important_end)}
        self.assertEqual(len(important_labels), 1)


if __name__ == "__main__":
    unittest.main()
