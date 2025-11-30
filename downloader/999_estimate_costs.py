#!/usr/bin/env python3
"""
Estimate daily Gemini costs from log files under gemini_output_logs/.

The script reads per-call usage metadata and applies the current pricing table,
including hidden/“thinking” tokens via total_token_count when present.
It emits a per-day breakdown for segmentation vs. purity runs and per-model totals.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Keep pricing aligned with downloader/999_check_gemini.py and 999_llm_segmentor.py
MODEL_PRICING_USD_PER_MTOKENS = {
    "gemini-2.5-pro": {
        "prompt": [
            {"max_prompt_tokens": 200_000, "rate": 1.25},
            {"max_prompt_tokens": None, "rate": 2.50},
        ],
        "response": [
            {"max_prompt_tokens": 200_000, "rate": 10.00},
            {"max_prompt_tokens": None, "rate": 15.00},
        ],
    },
    "gemini-2.5-flash": {
        "prompt": [
            {"max_prompt_tokens": None, "rate": 0.30},
        ],
        "response": [
            {"max_prompt_tokens": None, "rate": 2.50},
        ],
    },
}


def _calculate_tiered_cost(token_count: Optional[int], tiers: list[dict]) -> float:
    if token_count is None:
        return 0.0
    remaining = token_count
    cost = 0.0
    for tier in tiers:
        limit = tier.get("max_prompt_tokens")
        chunk = remaining if limit is None else min(remaining, limit)
        cost += (chunk / 1_000_000) * tier["rate"]
        remaining -= chunk
        if remaining <= 0:
            break
    if remaining > 0 and tiers:
        cost += (remaining / 1_000_000) * tiers[-1]["rate"]
    return cost


def estimate_cost(model: str, prompt_tokens: Optional[int], response_tokens: Optional[int], total_tokens: Optional[int]) -> float:
    pricing = MODEL_PRICING_USD_PER_MTOKENS.get(model)
    if not pricing:
        return 0.0
    # Use total_token_count to cover hidden/internal tokens.
    effective_response = response_tokens
    if effective_response is None and total_tokens is not None and prompt_tokens is not None:
        effective_response = max(total_tokens - prompt_tokens, 0)
    if (
        effective_response is not None
        and total_tokens is not None
        and prompt_tokens is not None
    ):
        effective_response = max(effective_response, total_tokens - prompt_tokens)
    prompt_cost = _calculate_tiered_cost(prompt_tokens, pricing.get("prompt", []))
    response_cost = _calculate_tiered_cost(effective_response, pricing.get("response", []))
    return prompt_cost + response_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate per-day Gemini costs from output logs.")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "gemini_output_logs",
        help="Directory containing *_*.json logs (default: gemini_output_logs/ at repo root).",
    )
    parser.add_argument(
        "--date-prefix",
        help="Optional YYYYMMDD prefix filter; when omitted, all dates are included.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_dir = args.logs_dir
    if not logs_dir.exists():
        raise SystemExit(f"Logs directory {logs_dir} does not exist.")

    daily = defaultdict(lambda: {"seg_cost": 0.0, "purity_cost": 0.0, "models": defaultdict(float), "calls": 0})

    for log_path in logs_dir.glob("*.json"):
        name = log_path.name
        if args.date_prefix and not name.startswith(args.date_prefix):
            continue
        try:
            payload = json.loads(log_path.read_text())
        except Exception:
            continue
        metadata = payload.get("metadata") or {}
        usage = metadata.get("usage_metadata") or {}
        model = metadata.get("model")
        prompt_tokens = usage.get("prompt_token_count")
        response_tokens = usage.get("candidates_token_count")
        total_tokens = usage.get("total_token_count")
        if not model:
            continue
        cost = estimate_cost(model, prompt_tokens, response_tokens, total_tokens)
        date_key = name.split("T", 1)[0]  # YYYYMMDD from filename prefix
        is_purity = (metadata.get("status") == "purity_check") or ("purity" in (payload.get("status") or "")) or metadata.get("purity_check")
        if is_purity:
            daily[date_key]["purity_cost"] += cost
        else:
            daily[date_key]["seg_cost"] += cost
        daily[date_key]["models"][model] += cost
        daily[date_key]["calls"] += 1

    if not daily:
        print("No matching log files found.")
        return

    for day in sorted(daily):
        entry = daily[day]
        seg = entry["seg_cost"]
        purity = entry["purity_cost"]
        total = seg + purity
        models = ", ".join(f"{m}=${entry['models'][m]:.4f}" for m in sorted(entry["models"]))
        print(
            f"{day}: total=${total:.4f} (seg=${seg:.4f}, purity=${purity:.4f}); calls={entry['calls']}; by_model: {models}"
        )


if __name__ == "__main__":
    main()
