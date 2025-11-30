#!/usr/bin/env python3
"""FastAPI viewer for Gemini segmentation snapshots.

The viewer exposes a lightweight UI that lists all JSON files in
``gemini_segmentations`` (or a user-provided directory) and renders the
ground-truth segments that Gemini produced.  It intentionally avoids any
model inference logic from segment_viewer/app.py while reusing its FastAPI
style.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


parser = argparse.ArgumentParser(description="Inspect Gemini segmentation output snapshots.")
parser.add_argument(
    "--segments-dir",
    type=str,
    default="../gemini_segmentations",
    help="Directory that stores *.json segmentation snapshots.",
)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8800)
parser.add_argument("--openapi", action="store_true", help="Expose FastAPI /docs and /openapi.json.")
args, _ = parser.parse_known_args()

SEGMENTS_DIR = Path(args.segments_dir).expanduser().resolve()
if not SEGMENTS_DIR.exists() or not SEGMENTS_DIR.is_dir():
    parser.error(f"Segments directory not found: {SEGMENTS_DIR}")

STATIC_DIR = Path(__file__).with_name("static")


@dataclass
class SegmentRecord:
    index: int
    type: str
    content: str

    @property
    def length(self) -> int:
        return len(self.content)


def _iter_snapshot_files() -> Iterable[Path]:
    if not SEGMENTS_DIR.exists():
        return []
    return sorted(
        SEGMENTS_DIR.rglob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _load_snapshot(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_segment_records(raw_segments: Optional[Iterable[Any]]) -> List[SegmentRecord]:
    records: List[SegmentRecord] = []
    if not raw_segments:
        return records
    for idx, raw in enumerate(raw_segments):
        if isinstance(raw, dict):
            seg_type = str(raw.get("type") or "unknown")
            content_value = raw.get("content")
        else:
            seg_type = "unknown"
            content_value = raw
        if content_value is None:
            text = ""
        else:
            text = str(content_value)
        records.append(SegmentRecord(index=idx, type=seg_type, content=text))
    return records


def _summaries() -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path in _iter_snapshot_files():
        relative_path = path.relative_to(SEGMENTS_DIR)
        relative_dir = relative_path.parent if relative_path.parent != Path(".") else Path(".")
        try:
            payload = _load_snapshot(path)
            segments = _ensure_segment_records(payload.get("segments"))
            metadata = payload.get("metadata") or {}
            run_id = payload.get("run_id") or path.stem
            purity = metadata.get("purity_check") or {}
            summaries.append(
                {
                    "run_id": run_id,
                    "file_name": path.name,
                    "path": str(relative_path),
                    "directory": str(relative_dir),
                    "timestamp": metadata.get("timestamp"),
                    "model": metadata.get("model"),
                    "input_source": metadata.get("input_source"),
                    "segments": len(segments),
                    "characters": sum(seg.length for seg in segments),
                    "size_bytes": path.stat().st_size,
                    "purity_check": purity,
                    "purity_status": purity.get("status"),
                    "purity_is_pure": purity.get("is_pure"),
                    "purity_language": purity.get("language"),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            summaries.append(
                {
                    "run_id": path.stem,
                    "file_name": path.name,
                    "path": str(relative_path),
                    "directory": str(relative_dir),
                    "error": f"Failed to read snapshot: {exc}",
                    "segments": 0,
                    "characters": 0,
                    "size_bytes": path.stat().st_size,
                }
            )
    return summaries


def _snapshot_path(run_id: str) -> Path:
    normalized = run_id.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Empty run_id.")
    candidate = (SEGMENTS_DIR / normalized).resolve()
    try:
        candidate.relative_to(SEGMENTS_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Run path escapes configured directory.")
    if candidate.is_file():
        return candidate
    candidate_with_suffix = candidate.with_suffix(".json")
    if candidate_with_suffix.is_file():
        return candidate_with_suffix
    for path in _iter_snapshot_files():
        if path.stem == normalized or path.name == normalized:
            return path
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found.")


def _load_run(run_id: str) -> Dict[str, Any]:
    path = _snapshot_path(run_id)
    payload = _load_snapshot(path)
    records = _ensure_segment_records(payload.get("segments"))
    metadata = payload.get("metadata") or {}
    relative_path = path.relative_to(SEGMENTS_DIR)
    relative_dir = relative_path.parent if relative_path.parent != Path(".") else Path(".")
    return {
        "run_id": payload.get("run_id") or path.stem,
        "file_name": path.name,
        "directory": str(relative_dir),
        "relative_path": str(relative_path),
        "path": str(path),
        "metadata": metadata,
        "segments": [
            {
                "index": rec.index,
                "type": rec.type,
                "length": rec.length,
                "content": rec.content,
            }
            for rec in records
        ],
        "unique_types": sorted({rec.type for rec in records}),
        "total_characters": sum(rec.length for rec in records),
    }


app = FastAPI(
    title="Gemini Ground Truth Viewer",
    docs_url="/docs" if args.openapi else None,
    redoc_url=None,
)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="UI assets missing. Expected index.html.")
    return FileResponse(html_path)


@app.get("/api/runs")
def api_runs():
    return {
        "directory": str(SEGMENTS_DIR),
        "runs": _summaries(),
    }


@app.get("/api/runs/{run_id}")
def api_run_detail(run_id: str, path: Optional[str] = None):
    identifier = path or run_id
    return _load_run(identifier)


@app.get("/api/runs/{run_id}/download")
def api_download(run_id: str, path: Optional[str] = None):
    identifier = path or run_id
    path = _snapshot_path(identifier)
    return FileResponse(
        path,
        media_type="application/json",
        filename=path.name,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
