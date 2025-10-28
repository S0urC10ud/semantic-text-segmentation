#!/usr/bin/env python3
"""
Interactive viewer for Arrow datasets produced by downloader/0_main.py.

Run:
    python downloader/dataset_viewer.py --data-root downloader/arrow_out

Then open http://127.0.0.1:8501 (or the host/port you choose).
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, load_from_disk  # type: ignore
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn  # type: ignore

try:
    from train import config as train_config  # type: ignore

    train_config.update_lang_mappings()
    _ID2LANG = dict(train_config.ID2LANG)
except Exception:  # pragma: no cover - optional dependency
    _ID2LANG = {}


@dataclass
class DatasetEntry:
    split: str
    label: str
    path: Path
    num_rows: int
    columns: List[str]


class DatasetManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._entries: Dict[Tuple[str, str], DatasetEntry] = {}
        self._datasets: Dict[Tuple[str, str], Dataset] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist.")

        splits: List[Path] = [
            p for p in sorted(self.root.iterdir()) if p.is_dir() and not p.name.startswith(".")
        ]
        if not splits:
            raise RuntimeError(f"No split directories found under '{self.root}'.")

        for split_dir in splits:
            split_name = split_dir.name
            for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                ds_dir = label_dir / "dataset"
                if not ds_dir.exists():
                    continue
                ds_key = (split_name, label_dir.name)
                dataset = load_from_disk(str(ds_dir))
                entry = DatasetEntry(
                    split=split_name,
                    label=label_dir.name,
                    path=ds_dir,
                    num_rows=len(dataset),
                    columns=list(dataset.column_names),
                )
                self._entries[ds_key] = entry
                self._datasets[ds_key] = dataset

        if not self._entries:
            raise RuntimeError(f"No datasets found under '{self.root}'.")

    def list_summaries(self) -> Dict[str, Any]:
        by_split: Dict[str, List[Dict[str, Any]]] = {}
        for (split, label), entry in sorted(self._entries.items()):
            by_split.setdefault(split, []).append(
                {
                    "label": label,
                    "count": entry.num_rows,
                    "columns": entry.columns,
                }
            )
        return {
            "root": str(self.root),
            "splits": [
                {"name": split, "labels": labels}
                for split, labels in sorted(by_split.items())
            ],
            "has_lang_lookup": bool(_ID2LANG),
        }

    def get_entry(self, split: str, label: str) -> DatasetEntry:
        key = (split, label)
        if key not in self._entries:
            raise KeyError(f"Unknown dataset for split='{split}', label='{label}'.")
        return self._entries[key]

    def get_dataset(self, split: str, label: str) -> Dataset:
        key = (split, label)
        if key not in self._datasets:
            raise KeyError(f"Dataset not loaded for split='{split}', label='{label}'.")
        return self._datasets[key]

    def get_sample(
        self, split: str, label: str, *, index: Optional[int] = None, mode: Optional[str] = None
    ) -> Dict[str, Any]:
        entry = self.get_entry(split, label)
        dataset = self.get_dataset(split, label)
        total = entry.num_rows
        if total == 0:
            raise RuntimeError(f"Dataset '{split}/{label}' is empty.")

        if mode == "random":
            idx = random.randint(0, total - 1)
        else:
            idx = 0 if index is None else int(index)
            idx = max(0, min(idx, total - 1))

        record = dataset[idx]
        content = record.get("content", "")
        meta = {
            k: self._coerce_value(v)
            for k, v in record.items()
            if k != "content"
        }
        resolved_lang = None
        if "lang_id" in record and _ID2LANG:
            try:
                resolved_lang = _ID2LANG.get(int(record["lang_id"]))
            except Exception:
                resolved_lang = None

        sample = {
            "split": split,
            "label": label,
            "index": idx,
            "total": total,
            "content": content,
            "fields": meta,
            "has_prev": idx > 0,
            "has_next": idx < (total - 1),
            "resolved_lang": resolved_lang,
            "columns": entry.columns,
            "char_count": len(content) if isinstance(content, str) else None,
            "line_count": (content.count("\n") + 1) if isinstance(content, str) else None,
        }
        return sample

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (list, tuple)):
            return [DatasetManager._coerce_value(v) for v in value]
        if isinstance(value, dict):
            return json.loads(json.dumps(value, default=str))
        return str(value)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Content Window Viewer</title>
  <style>
    :root {
      --bg: #f5f6fa;
      --pane: #ffffff;
      --accent: #2563eb;
      --accent-soft: rgba(37, 99, 235, 0.12);
      --border: rgba(15, 23, 42, 0.08);
      --text: #1a1c23;
      --text-muted: #5a6072;
      --code-bg: #ffffff;
      --scrollbar: rgba(60, 80, 120, 0.35);
    }
    body {
      margin: 0;
      font-family: "Inter", "SF Pro Display", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      min-height: 100vh;
    }
    * {
      box-sizing: border-box;
    }
    .layout {
      display: flex;
      flex-direction: column;
      flex: 1;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      gap: 16px;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(14,165,233,0.10));
      box-shadow: 0 12px 35px rgba(15, 23, 42, 0.15);
    }
    .title-block h1 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0.01em;
    }
    .title-block p {
      margin: 4px 0 0;
      font-size: 13px;
      color: var(--text-muted);
    }
    .controls {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    select, input, button {
      font: inherit;
      padding: 8px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--pane);
      color: inherit;
      outline: none;
      transition: border 0.15s ease, box-shadow 0.15s ease;
    }
    input[type="search"] {
      min-width: 240px;
    }
    button {
      cursor: pointer;
    }
    button:hover {
      border-color: rgba(37,99,235,0.35);
      box-shadow: 0 0 0 3px var(--accent-soft);
    }
    .main {
      display: grid;
      grid-template-columns: minmax(240px, 260px) 1fr;
      gap: 20px;
      flex: 1;
    }
    @media (max-width: 960px) {
      .main {
        grid-template-columns: 1fr;
      }
    }
    aside {
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.65);
      backdrop-filter: blur(12px);
      overflow: hidden;
      max-height: calc(100vh - 160px);
    }
    aside h2 {
      margin: 0;
      font-size: 15px;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .labels {
      overflow-y: auto;
      flex: 1;
      padding-right: 6px;
    }
    .labels::-webkit-scrollbar {
      width: 8px;
    }
    .labels::-webkit-scrollbar-thumb {
      background: var(--scrollbar);
      border-radius: 999px;
    }
    .labels button {
      display: block;
      width: 100%;
      text-align: left;
      margin-bottom: 6px;
      background: rgba(255,255,255,0.65);
      border: 1px solid var(--border);
      padding: 10px 12px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.35;
      transition: transform 0.12s ease, box-shadow 0.18s ease;
    }
    .labels button:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 20px rgba(15, 23, 42, 0.12);
    }
    .labels button.active {
      background: linear-gradient(135deg, rgba(37,99,235,0.14), rgba(37,99,235,0.22));
      border-color: rgba(37,99,235,0.45);
      color: #1a2e6f;
      font-weight: 600;
    }
    main {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .sample-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
    }
    .sample-header h2 {
      margin: 0;
      font-size: 18px;
    }
    .sample-header span {
      font-size: 13px;
      color: var(--text-muted);
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .toolbar span {
      font-size: 13px;
      color: var(--text-muted);
    }
    .content-panel {
      display: flex;
      flex-direction: column;
      gap: 14px;
      border-radius: 18px;
      padding: 18px;
      background: rgba(255,255,255,0.75);
      border: 1px solid var(--border);
      box-shadow: 0 16px 40px rgba(15,23,42,0.12);
      min-height: 320px;
    }
    .code-wrapper {
      border-radius: 14px;
      border: 1px solid rgba(15, 23, 42, 0.1);
      background: var(--code-bg);
      overflow: hidden;
      width: 100%;
      max-height: 520px;
      overflow-y: auto;
      font-family: "JetBrains Mono", "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.45;
    }
    .code-wrapper::-webkit-scrollbar {
      width: 10px;
      height: 10px;
    }
    .code-wrapper::-webkit-scrollbar-thumb {
      background: rgba(80, 112, 160, 0.4);
      border-radius: 999px;
    }
    .code-block {
      display: block;
    }
    .code-line {
      display: grid;
      grid-template-columns: 60px 1fr;
      gap: 14px;
      padding: 6px 18px;
      border-bottom: 1px solid rgba(15,23,42,0.05);
      width: 100%;
      align-items: start;
    }
    .code-line:nth-child(odd) {
      background: rgba(37,99,235,0.04);
    }
    .code-line-number {
      color: rgba(30,64,175,0.7);
      user-select: none;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .code-text {
      display: block;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      max-width: 100%;
      tab-size: 4;
    }
    .placeholder {
      text-align: center;
      padding: 120px 40px;
      color: var(--text-muted);
      border: 1px dashed var(--border);
      border-radius: 16px;
    }
    .meta {
      border: 1px solid rgba(15,23,42,0.08);
      border-radius: 12px;
      padding: 12px 16px;
      font-size: 13px;
      display: grid;
      gap: 8px;
      background: rgba(37,99,235,0.05);
    }
    .meta-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }
    .meta-key {
      font-weight: 600;
      color: var(--text-muted);
      flex: 0 0 140px;
    }
    .meta-value {
      flex: 1;
      text-align: right;
      word-break: break-word;
    }
    .empty-meta {
      font-style: italic;
      color: var(--text-muted);
      text-align: center;
    }
    .error {
      color: #dc2626;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <div class="layout">
    <header>
      <div class="title-block">
        <h1>Stack Content Explorer</h1>
        <p>Browse the Arrow windows emitted by downloader/0_main.py. Click a content type and fly through samples.</p>
      </div>
      <div class="controls">
        <label>
          <span style="font-size:12px; display:block; color:var(--text-muted);">Split</span>
          <select id="splitSelect"></select>
        </label>
        <label>
          <span style="font-size:12px; display:block; color:var(--text-muted);">Filter types</span>
          <input id="labelFilter" type="search" placeholder="Search content types..." />
        </label>
      </div>
    </header>
    <div class="main">
      <aside>
        <h2>
          <span>Content Types</span>
          <span id="labelCounter"></span>
        </h2>
        <div id="labels" class="labels"></div>
      </aside>
      <main>
        <div class="sample-header">
          <h2 id="sampleTitle">Select a content type to begin</h2>
          <span id="sampleStats"></span>
        </div>
        <div id="toolbar" class="toolbar" style="display:none;">
          <div>
            <button id="prevBtn" title="← Previous sample">← Prev</button>
            <button id="nextBtn" title="→ Next sample">Next →</button>
            <button id="randomBtn" title="Random sample">Feeling lucky</button>
          </div>
          <span id="positionLabel"></span>
          <label>
            <span style="font-size:12px; display:block; color:var(--text-muted);">Jump to index</span>
            <input id="indexInput" type="number" min="1" value="1" style="width:120px;" />
          </label>
        </div>
        <div id="contentPanel" class="content-panel">
          <div id="placeholder" class="placeholder">
            Choose a content type from the list to preview individual windows.
          </div>
          <div id="codeWrapper" class="code-wrapper" style="display:none;">
            <div id="codeBlock" class="code-block"></div>
          </div>
          <div id="meta" class="meta" style="display:none;"></div>
          <div id="error" class="error" style="display:none;"></div>
        </div>
      </main>
    </div>
  </div>
  <script>
    (() => {
      const state = {
        summary: null,
        currentSplit: null,
        currentLabel: null,
        currentIndex: 0,
        labelsForSplit: [],
      };

      const splitSelect = document.getElementById('splitSelect');
      const labelFilter = document.getElementById('labelFilter');
      const labelsContainer = document.getElementById('labels');
      const labelCounter = document.getElementById('labelCounter');
      const sampleTitle = document.getElementById('sampleTitle');
      const sampleStats = document.getElementById('sampleStats');
      const toolbar = document.getElementById('toolbar');
      const prevBtn = document.getElementById('prevBtn');
      const nextBtn = document.getElementById('nextBtn');
      const randomBtn = document.getElementById('randomBtn');
      const indexInput = document.getElementById('indexInput');
      const codeWrapper = document.getElementById('codeWrapper');
      const codeBlock = document.getElementById('codeBlock');
      const metaBlock = document.getElementById('meta');
      const placeholder = document.getElementById('placeholder');
      const positionLabel = document.getElementById('positionLabel');
      const errorBox = document.getElementById('error');

      function escapeHtml(str) {
        return str.replace(/[&<>"']/g, (c) => ({
          '&': '&amp;',
          '<': '&lt;',
          '>': '&gt;',
          '"': '&quot;',
          "'": '&#39;'
        }[c]));
      }

      function formatInteger(n) {
        return Intl.NumberFormat().format(n);
      }

      function clearActiveLabel() {
        document.querySelectorAll('.labels button.active').forEach((btn) => {
          btn.classList.remove('active');
        });
      }

      function updateToolbar(sample) {
        toolbar.style.display = 'flex';
        prevBtn.disabled = !sample.has_prev;
        nextBtn.disabled = !sample.has_next;
        indexInput.max = sample.total;
        indexInput.value = sample.index + 1;
        positionLabel.textContent = `Showing ${formatInteger(sample.index + 1)} of ${formatInteger(sample.total)}`;
        randomBtn.disabled = sample.total <= 1;
        sampleStats.textContent = `${formatInteger(sample.total)} windows`;
        sampleTitle.textContent = `${sample.label} · ${sample.split}`;
        if (sample.resolved_lang) {
          sampleTitle.textContent += ` (lang_id → ${sample.resolved_lang})`;
        }
      }

      function renderContent(sample) {
        if (!sample.content) {
          codeWrapper.style.display = 'none';
          placeholder.style.display = 'block';
          placeholder.textContent = 'This sample does not have textual content.';
          return;
        }
        const lines = sample.content.split('\\n');
        const html = lines.map((line, idx) => {
          const safe = line.length ? escapeHtml(line) : '&nbsp;';
          return [
            '<div class="code-line">',
            `<span class="code-line-number">${idx + 1}</span>`,
            `<span class="code-text">${safe}</span>`,
            '</div>',
          ].join('');
        }).join('');
        codeBlock.innerHTML = html;
        placeholder.style.display = 'none';
        codeWrapper.style.display = 'block';
      }

      function renderMeta(sample) {
        const entries = Object.entries(sample.fields || {});
        if (!entries.length) {
          metaBlock.style.display = 'block';
          metaBlock.innerHTML = '<div class="empty-meta">No additional metadata for this sample.</div>';
          return;
        }
        const html = entries.map(([key, value]) => {
          const formatted = typeof value === 'object'
            ? escapeHtml(JSON.stringify(value))
            : escapeHtml(String(value));
          return [
            '<div class="meta-row">',
            `<div class="meta-key">${escapeHtml(key)}</div>`,
            `<div class="meta-value">${formatted}</div>`,
            '</div>',
          ].join('');
        }).join('');
        metaBlock.innerHTML = html;
        metaBlock.style.display = 'block';
      }

      function setError(message) {
        errorBox.style.display = 'block';
        errorBox.textContent = message;
      }

      function clearError() {
        errorBox.style.display = 'none';
        errorBox.textContent = '';
      }

      async function fetchSummary() {
        const resp = await fetch('/api/summary');
        if (!resp.ok) {
          throw new Error('Failed to load dataset summary.');
        }
        const data = await resp.json();
        state.summary = data;
        const splits = data.splits || [];
        splitSelect.innerHTML = splits.map((item) => `<option value="${item.name}">${item.name}</option>`).join('');
        if (splits.length) {
          setSplit(splits[0].name);
        }
      }

      function applyFilter() {
        const q = labelFilter.value.trim().toLowerCase();
        const filtered = q
          ? state.labelsForSplit.filter((item) => item.label.toLowerCase().includes(q))
          : state.labelsForSplit;
        renderLabelButtons(filtered);
      }

      function renderLabelButtons(list) {
        labelsContainer.innerHTML = list.map((item) => {
          const count = formatInteger(item.count);
          return [
            `<button data-label="${item.label}">`,
            `<strong>${escapeHtml(item.label)}</strong>`,
            '<br/>',
            `<small>${count} windows</small>`,
            '</button>',
          ].join('');
        }).join('');
        labelCounter.textContent = `${list.length} types`;
      }

      function setSplit(name) {
        state.currentSplit = name;
        const splitData = (state.summary.splits || []).find((s) => s.name === name);
        state.labelsForSplit = splitData ? splitData.labels : [];
        labelFilter.value = '';
        renderLabelButtons(state.labelsForSplit);
        clearActiveLabel();
        sampleTitle.textContent = 'Select a content type to begin';
        sampleStats.textContent = '';
        placeholder.style.display = 'block';
        placeholder.textContent = 'Choose a content type from the list to preview individual windows.';
        codeWrapper.style.display = 'none';
        metaBlock.style.display = 'none';
        toolbar.style.display = 'none';
        errorBox.style.display = 'none';
      }

      async function loadSample(mode, forcedIndex) {
        if (!state.currentSplit || !state.currentLabel) return;
        clearError();
        const params = new URLSearchParams({
          split: state.currentSplit,
          label: state.currentLabel,
        });
        if (mode === 'random') {
          params.set('mode', 'random');
        } else if (typeof forcedIndex === 'number') {
          params.set('index', String(forcedIndex));
        } else {
          params.set('index', String(state.currentIndex));
        }
        const resp = await fetch(`/api/sample?${params.toString()}`);
        if (!resp.ok) {
          setError('Unable to fetch sample (HTTP ' + resp.status + ').');
          return;
        }
        const sample = await resp.json();
        state.currentIndex = sample.index;
        updateToolbar(sample);
        renderContent(sample);
        renderMeta(sample);
      }

      function handleLabelClick(event) {
        const btn = event.target.closest('button[data-label]');
        if (!btn) return;
        const label = btn.getAttribute('data-label');
        state.currentLabel = label;
        state.currentIndex = 0;
        clearActiveLabel();
        btn.classList.add('active');
        loadSample();
      }

      splitSelect.addEventListener('change', (event) => {
        setSplit(event.target.value);
      });

      labelFilter.addEventListener('input', () => {
        applyFilter();
      });

      labelsContainer.addEventListener('click', handleLabelClick);

      prevBtn.addEventListener('click', () => {
        if (state.currentIndex > 0) {
          state.currentIndex -= 1;
          loadSample();
        }
      });
      nextBtn.addEventListener('click', () => {
        state.currentIndex += 1;
        loadSample();
      });
      randomBtn.addEventListener('click', () => {
        loadSample('random');
      });
      indexInput.addEventListener('change', () => {
        const value = parseInt(indexInput.value, 10);
        if (Number.isNaN(value)) return;
        const target = Math.max(1, value) - 1;
        state.currentIndex = target;
        loadSample();
      });
      document.addEventListener('keydown', (event) => {
        if (event.target === labelFilter || event.target === indexInput) return;
        if (event.key === 'ArrowLeft' && !prevBtn.disabled) {
          event.preventDefault();
          prevBtn.click();
        } else if (event.key === 'ArrowRight' && !nextBtn.disabled) {
          event.preventDefault();
          nextBtn.click();
        } else if (event.key.toLowerCase() === 'r' && !randomBtn.disabled) {
          event.preventDefault();
          randomBtn.click();
        }
      });

      fetchSummary().catch((err) => {
        console.error(err);
        setError(err.message || 'Failed to load dataset metadata.');
      });
    })();
  </script>
</body>
</html>
"""


def create_app(manager: DatasetManager) -> FastAPI:
    app = FastAPI(title="Stack Dataset Viewer", version="1.0")
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/api/summary")
    def get_summary(request: Request) -> JSONResponse:
        manager_ref: DatasetManager = request.app.state.manager
        data = manager_ref.list_summaries()
        return JSONResponse(data)

    @app.get("/api/sample")
    def get_sample(
        request: Request,
        split: str,
        label: str,
        index: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> JSONResponse:
        manager_ref: DatasetManager = request.app.state.manager
        try:
            sample = manager_ref.get_sample(split, label, index=index, mode=mode)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - runtime guard
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(sample)

    return app


def _launch_browser(host: str, port: int) -> None:
    url = f"http://{host}:{port}"
    threading.Timer(0.8, lambda: webbrowser.open_new_tab(url)).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a browser-based viewer for Arrow datasets.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("downloader/arrow_out"),
        help="Root directory containing split/label/dataset outputs from downloader/0_main.py.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the web server.")
    parser.add_argument("--port", type=int, default=8501, help="Port for the web server.")
    parser.add_argument("--reload", action="store_true", help="Enable FastAPI autoreload (for development).")
    parser.add_argument("--open", action="store_true", help="Automatically open the viewer in your browser.")
    args = parser.parse_args()

    manager = DatasetManager(args.data_root.resolve())
    app = create_app(manager)

    if args.open:
        _launch_browser(args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
