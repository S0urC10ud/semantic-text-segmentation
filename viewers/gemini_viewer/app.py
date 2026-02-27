#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

parser = argparse.ArgumentParser(description="Inspect Gemini segmentation output snapshots.")
parser.add_argument(
    "--segments-dir",
    type=str,
    default="../../gemini_segmentations",
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
BASE_PATH = "/ground-truth"


@dataclass
class SegmentRecord:
    index: int
    type: str
    content: str

    @property
    def length(self) -> int:
        return len(self.content)


def _segment_type(raw: Any) -> str:
    if isinstance(raw, dict):
        value = raw.get("type")
        return str(value) if value else "unknown"
    return "unknown"


def _segment_text(raw: Any) -> str:
    if isinstance(raw, dict):
        content_value = raw.get("content")
    else:
        content_value = raw
    if content_value is None:
        return ""
    return str(content_value)


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


def _ensure_segment_records(
    raw_segments: Optional[Iterable[Any]], *, filter_type: Optional[str] = None
) -> List[SegmentRecord]:
    records: List[SegmentRecord] = []
    if not raw_segments:
        return records
    for idx, raw in enumerate(raw_segments):
        seg_type = _segment_type(raw)
        if filter_type and seg_type != filter_type:
            continue
        text = _segment_text(raw)
        records.append(SegmentRecord(index=idx, type=seg_type, content=text))
    return records


def _summaries() -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path in _iter_snapshot_files():
        relative_path = path.relative_to(SEGMENTS_DIR)
        relative_dir = relative_path.parent if relative_path.parent != Path(".") else Path(".")
        try:
            payload = _load_snapshot(path)
            raw_segments = payload.get("segments") or []
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
                    "segments": len(raw_segments),
                    "characters": sum(len(_segment_text(seg)) for seg in raw_segments),
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


def _build_run_detail(path: Path, payload: Dict[str, Any], *, filter_type: Optional[str] = None) -> Dict[str, Any]:
    raw_segments = payload.get("segments") or []
    metadata = payload.get("metadata") or {}
    records = _ensure_segment_records(raw_segments, filter_type=filter_type)
    unique_types_all = {_segment_type(raw) for raw in raw_segments}
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
        "unique_types_all": sorted(unique_types_all),
        "segments_total": len(raw_segments),
        "segments_returned": len(records),
        "total_characters": sum(rec.length for rec in records),
        "total_characters_all": sum(len(_segment_text(seg)) for seg in raw_segments),
        "selected_type": filter_type,
    }


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


def _load_run(run_id: str, *, filter_type: Optional[str] = None) -> Dict[str, Any]:
    path = _snapshot_path(run_id)
    payload = _load_snapshot(path)
    return _build_run_detail(path, payload, filter_type=filter_type)


def _content_type_overview() -> Tuple[List[Dict[str, Any]], int]:
    counts: Dict[str, int] = {}
    total_files = 0
    for path in _iter_snapshot_files():
        total_files += 1
        relative_dir = path.relative_to(SEGMENTS_DIR).parent
        key = relative_dir if relative_dir != Path(".") else Path(".")
        counts[str(key)] = counts.get(str(key), 0) + 1
    overview = []
    for key, count in counts.items():
        parts = Path(key).parts
        label = parts[-1] if parts else "(root)"
        if key == ".":
            label = "(root)"
        overview.append({"id": key, "label": label, "file_count": count})
    sorted_types = sorted(overview, key=lambda item: (-item["file_count"], item["label"]))
    return sorted_types, total_files


def _random_sample_for_type(content_type: str) -> Dict[str, Any]:
    normalized = content_type.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Content type must be provided.")
    target_dir = Path(normalized)
    selected_path: Optional[Path] = None
    selected_payload: Optional[Dict[str, Any]] = None
    seen = 0
    for path in _iter_snapshot_files():
        relative_dir = path.relative_to(SEGMENTS_DIR).parent
        relative_str = str(relative_dir if relative_dir != Path(".") else Path("."))
        if relative_str != str(target_dir):
            continue
        seen += 1
        try:
            payload = _load_snapshot(path)
        except Exception:
            continue
        if selected_path is None or random.randint(1, seen) == 1:
            selected_path = path
            selected_payload = payload
    if not selected_path or not selected_payload:
        raise HTTPException(
            status_code=404,
            detail=f"No snapshots found for content type directory {normalized!r}.",
        )
    detail = _build_run_detail(selected_path, selected_payload, filter_type=None)
    detail["selected_type"] = normalized
    return detail


app = FastAPI(
    title="Gemini Ground Truth Viewer",
    docs_url=f"{BASE_PATH}/docs" if args.openapi else None,
    redoc_url=None,
    openapi_url=f"{BASE_PATH}/openapi.json" if args.openapi else None,
)

if STATIC_DIR.is_dir():
    app.mount(f"{BASE_PATH}/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url=f"{BASE_PATH}/", status_code=307)


@app.get(f"{BASE_PATH}/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="UI assets missing. Expected index.html.")
    return FileResponse(html_path)


@app.get(f"{BASE_PATH}/api/runs")
def api_runs():
    return {
        "directory": str(SEGMENTS_DIR),
        "runs": _summaries(),
    }


@app.get(f"{BASE_PATH}/api/runs/{{run_id}}")
def api_run_detail(
    run_id: str,
    path: Optional[str] = None,
    type: Optional[str] = None,
    segment_type: Optional[str] = None,
):
    identifier = path or run_id
    filter_type = segment_type or type
    return _load_run(identifier, filter_type=filter_type)


@app.get(f"{BASE_PATH}/api/runs/{{run_id}}/download")
def api_download(run_id: str, path: Optional[str] = None):
    identifier = path or run_id
    path = _snapshot_path(identifier)
    return FileResponse(
        path,
        media_type="application/json",
        filename=path.name,
    )


@app.get(f"{BASE_PATH}/api/content-types")
def api_content_types():
    types, total_files = _content_type_overview()
    return {
        "directory": str(SEGMENTS_DIR),
        "types": types,
        "total_files": total_files,
    }


@app.get(f"{BASE_PATH}/api/content-types/sample")
def api_content_type_sample(content_type: Optional[str] = None, type: Optional[str] = None):
    chosen_type = content_type or type
    if not chosen_type:
        raise HTTPException(status_code=400, detail="Content type query parameter is required.")
    return _random_sample_for_type(chosen_type)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
