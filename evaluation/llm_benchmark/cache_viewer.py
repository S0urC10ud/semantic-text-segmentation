#!/usr/bin/env python3
"""Tiny web viewer for an LLM-benchmark run cache JSONL.

Renders every cached sample as the source text with coloured segment overlays
so you can eyeball how fine-grained Pro's labelling actually was. No external
build step — single-file FastAPI app + inline HTML/CSS.

Usage:
    python evaluation/llm_benchmark/cache_viewer.py \
        --cache evaluation/llm_benchmark/runs/<run_id>.jsonl \
        --port 8094
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "evaluation" / "llm_benchmark" / "runs"


def load_records(cache: Path) -> list[dict]:
    out = []
    with cache.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    # Sort by request timestamp ascending; records without a timestamp sort first.
    out.sort(key=lambda r: r.get("timestamp") or "")
    return out


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return ""
    # Render only HH:MM:SS for compactness; date is typically the same across a run.
    try:
        from datetime import datetime
        # Python <3.11 doesn't accept trailing 'Z' in fromisoformat
        t = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        dt = datetime.fromisoformat(t)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ts[:19]


def load_static(static_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not static_path.is_file():
        return out
    with static_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["benchmark_id"]] = row
    return out


def color_for_label(label: str) -> str:
    """Stable HSL colour from label hash."""
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % 360 / 360
    r, g, b = colorsys.hls_to_rgb(h, 0.78, 0.55)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def render_segments_html(segments: list[dict]) -> str:
    parts: list[str] = []
    for seg in segments:
        label = seg.get("type", "?")
        content = seg.get("content", "")
        color = color_for_label(label)
        parts.append(
            f'<span class="seg" style="background:{color}" '
            f'data-label="{html.escape(label)}" '
            f'title="{html.escape(label)} ({len(content)} chars)">'
            f"<span class=\"seg-label\">{html.escape(label)}</span>"
            f"<span class=\"seg-content\">{html.escape(content)}</span>"
            "</span>"
        )
    return "".join(parts)


def build_app(cache_path: Path, static_path: Path | None) -> FastAPI:
    app = FastAPI(title="LLM Benchmark Cache Viewer")

    state: dict[str, Any] = {
        "cache_path": cache_path,
        "static_path": static_path,
        "records": load_records(cache_path),
        "static": load_static(static_path) if static_path else {},
    }

    @app.get("/api/reload")
    def reload_records() -> dict:
        state["records"] = load_records(cache_path)
        if static_path:
            state["static"] = load_static(static_path)
        return {"loaded": len(state["records"])}

    @app.get("/api/summary")
    def summary() -> dict:
        recs = state["records"]
        status_counts = Counter(r.get("status", "unknown") for r in recs)
        host_counts = Counter(r.get("host", "?") for r in recs)
        total_cost = sum(float(r.get("cost_usd") or 0) for r in recs)
        seg_counts = [len(r.get("segments", [])) for r in recs if r.get("status") == "ok"]
        per_host = []
        for host, n in sorted(host_counts.items()):
            host_recs = [r for r in recs if r.get("host") == host]
            ok = sum(1 for r in host_recs if r.get("status") == "ok")
            mean_segs = (
                sum(len(r.get("segments", [])) for r in host_recs if r.get("status") == "ok") / max(ok, 1)
            )
            cost = sum(float(r.get("cost_usd") or 0) for r in host_recs)
            per_host.append(
                {
                    "host": host,
                    "n": n,
                    "ok": ok,
                    "mean_segs": round(mean_segs, 1),
                    "total_cost": round(cost, 4),
                }
            )
        return {
            "n": len(recs),
            "status_counts": dict(status_counts),
            "total_cost": round(total_cost, 4),
            "per_host": per_host,
            "max_segs": max(seg_counts) if seg_counts else 0,
        }

    @app.get("/api/record/{idx}")
    def record(idx: int) -> JSONResponse:
        recs = state["records"]
        if idx < 0 or idx >= len(recs):
            raise HTTPException(404, "out of range")
        rec = recs[idx]
        return JSONResponse(rec)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        recs = state["records"]
        rows_html: list[str] = []
        for i, r in enumerate(recs):
            status = r.get("status", "?")
            host = r.get("host", "?")
            n_segs = len(r.get("segments", []))
            chars = r.get("input_characters", 0)
            cost = r.get("cost_usd", 0.0) or 0.0
            synthetic = " synthetic" if r.get("synthetic") else ""
            status_class = "ok" if status == "ok" else "fail"
            ts_short = _fmt_ts(r.get("timestamp"))
            rows_html.append(
                f'<tr class="row {status_class}{synthetic}" data-idx="{i}" onclick="show({i})">'
                f"<td>{i}</td>"
                f'<td class="ts">{html.escape(ts_short)}</td>'
                f"<td>{html.escape(host)}</td>"
                f'<td class="status">{html.escape(status)}{synthetic}</td>'
                f"<td>{chars:,}</td>"
                f"<td>{n_segs}</td>"
                f"<td>${cost:.4f}</td>"
                "</tr>"
            )
        rows = "\n".join(rows_html)
        return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>LLM Benchmark Cache Viewer — {html.escape(cache_path.name)}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }}
  #side {{ width: 560px; overflow-y: auto; border-right: 1px solid #ddd; padding: 8px; background: #fafafa; }}
  td.ts {{ font-family: ui-monospace, monospace; font-size: 11px; white-space: nowrap; color: #555; }}
  #main {{ flex: 1; overflow-y: auto; padding: 14px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ padding: 4px 6px; text-align: left; }}
  th {{ border-bottom: 1px solid #ccc; background: #eee; position: sticky; top: 0; }}
  tr.row {{ cursor: pointer; }}
  tr.row:hover {{ background: #eef; }}
  tr.fail {{ color: #b00; }}
  tr.synthetic {{ color: #888; font-style: italic; }}
  .seg {{
    display: inline; white-space: pre-wrap; border-radius: 3px;
    padding: 0 1px; box-shadow: inset 0 -1px 0 rgba(0,0,0,0.12);
  }}
  .seg-label {{
    font-family: -apple-system, monospace; font-size: 9px; font-weight: 700;
    background: rgba(0,0,0,0.55); color: white; padding: 1px 3px;
    border-radius: 2px; margin-right: 2px; vertical-align: 1px;
  }}
  .seg-content {{
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px;
  }}
  #segment-pane {{ white-space: pre-wrap; word-break: break-all; }}
  .meta {{ font-size: 11px; color: #555; margin-bottom: 8px; font-family: monospace; }}
  .summary {{ padding: 6px 0; font-size: 11px; }}
  .summary span {{ display: inline-block; padding: 1px 6px; margin-right: 4px; border-radius: 3px; background: #ddd; }}
  .summary span.ok {{ background: #cfe8cf; }}
  .summary span.fail {{ background: #f3c7c7; }}
</style>
</head><body>
<div id="side">
  <div class="summary" id="summary">loading…</div>
  <table>
    <thead><tr><th>#</th><th>ts</th><th>host</th><th>status</th><th>chars</th><th>segs</th><th>$</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
<div id="main">
  <div class="meta" id="meta">click a row to view its segmentation</div>
  <div id="segment-pane"></div>
</div>
<script>
async function loadSummary() {{
  const r = await fetch('/api/summary');
  const j = await r.json();
  const sc = j.status_counts || {{}};
  const parts = [`<b>${{j.n}} records</b>`, `total $${{j.total_cost.toFixed(4)}}`];
  for (const [k, v] of Object.entries(sc)) {{
    const cls = k === 'ok' ? 'ok' : 'fail';
    parts.push(`<span class="${{cls}}">${{k}}: ${{v}}</span>`);
  }}
  document.getElementById('summary').innerHTML = parts.join(' ');
}}
async function show(idx) {{
  const r = await fetch('/api/record/' + idx);
  const rec = await r.json();
  const meta = [
    `#${{idx}}`, rec.host, rec.status,
    `chars=${{rec.input_characters || 0}}`,
    `segs=${{(rec.segments||[]).length}}`,
    `$${{(rec.cost_usd||0).toFixed(4)}}`,
    `latency=${{(rec.latency_s||0).toFixed(1)}}s`,
  ];
  if (rec.synthetic) meta.push('SYNTHETIC');
  if (rec.error) meta.push('error=' + rec.error);
  document.getElementById('meta').textContent = meta.join(' • ');
  const pane = document.getElementById('segment-pane');
  if (rec.status !== 'ok') {{
    pane.innerHTML = '<pre>' + (rec.response_text || rec.error || '(no body)').replace(/[<>&]/g, c => ({{'<':'&lt;','>':'&gt;','&':'&amp;'}})[c]) + '</pre>';
    return;
  }}
  // Render segments
  const segs = rec.segments || [];
  const html_parts = [];
  for (const s of segs) {{
    const label = s.type || '?';
    const c = labelColor(label);
    const content = (s.content || '').replace(/[<>&]/g, c => ({{'<':'&lt;','>':'&gt;','&':'&amp;'}})[c]);
    html_parts.push(`<span class="seg" style="background:${{c}}" title="${{label}}"><span class="seg-label">${{label}}</span><span class="seg-content">${{content}}</span></span>`);
  }}
  pane.innerHTML = html_parts.join('');
}}
function labelColor(label) {{
  let h = 0; for (const c of label) h = (h*31 + c.charCodeAt(0)) & 0xffffffff;
  const hue = Math.abs(h) % 360 / 360;
  return `hsl(${{hue*360}}, 70%, 80%)`;
}}
loadSummary();
</script>
</body></html>"""

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Path to a run cache JSONL (evaluation/llm_benchmark/runs/<run_id>.jsonl).",
    )
    parser.add_argument(
        "--static",
        type=Path,
        default=REPO_ROOT / "evaluation" / "llm_benchmark" / "static_unseen_v1.jsonl",
        help="Path to static_unseen_v1.jsonl (for source text fallback if a record is missing 'segments').",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()

    if not args.cache.is_file():
        # Allow passing a glob-friendly run_id stem; pick the matching file.
        candidates = list(DEFAULT_CACHE.glob(args.cache.name + "*.jsonl"))
        if len(candidates) == 1:
            args.cache = candidates[0]
        else:
            print(f"cache not found: {args.cache}", file=sys.stderr)
            return 2

    print(f"[viewer] serving {args.cache}")
    print(f"[viewer] open http://{args.host}:{args.port}/")
    app = build_app(args.cache, args.static if args.static.is_file() else None)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
