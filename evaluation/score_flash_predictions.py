"""Compatibility wrapper for the generic cached-LLM scorer."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SCORER_PATH = HERE / "score_llm_predictions.py"
_spec = importlib.util.spec_from_file_location("score_llm_predictions_main", str(_SCORER_PATH))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["score_llm_predictions_main"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
main = _mod.main


if __name__ == "__main__":
    sys.exit(main())
