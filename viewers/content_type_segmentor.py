#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = (REPO_ROOT / "viewers" / "content_type_segmentor_static").resolve()


def create_app(static_root: Path = STATIC_ROOT) -> FastAPI:
    app = FastAPI(title="Content Type Segmentor", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
    manifest_path = (Path(static_root) / "assets" / "model_manifest.json").resolve()

    @app.get("/")
    def index():
        return FileResponse(str(static_root / "index.html"))

    @app.get("/healthz")
    def healthz():
        return JSONResponse(
            {
                "ok": bool(manifest_path.exists()),
                "static_root": str(static_root),
                "model_manifest": str(manifest_path),
            }
        )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the public Content Type Segmentor demo.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=int(args.port), reload=False)
    return 0


app = create_app()


if __name__ == "__main__":
    raise SystemExit(main())
