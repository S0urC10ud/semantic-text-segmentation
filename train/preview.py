"""
Utilities for generating an HTML preview of augmented data samples.
"""
import os
import json
from typing import List, Tuple
import numpy as np

from config import ID2LANG, PAD_ID

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Data Preview</title>
<style>
  body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; margin: 24px; }
  .toolbar { display:flex; gap:12px; align-items:center; margin-bottom:12px; }
  button { padding:6px 10px; }
  .legend span { display:inline-block; padding:2px 8px; border-radius:4px; margin-right:8px; color:#fff; }
  .html { background:#1f77b4; }
  .css { background:#2ca02c; }
  .javascript { background:#d62728; }
  .pad { background:#999; color:#000; }
  .muted { color:#666; font-size:0.9em; }
  .example { display:none; white-space:pre-wrap; word-break:break-word; border:1px solid #eee; padding:12px; border-radius:8px; background:#fafafa; }
  .example.active { display:block; }
  .tok.html { background:rgba(31,119,180,.15); }
  .tok.css { background:rgba(44,160,44,.15); }
  .tok.javascript { background:rgba(214,39,40,.15); }
  .tok.pad { background:rgba(153,153,153,.20); color:#555; }
  .stats { margin-bottom:8px; }
  .code { font-size: 12.5px; line-height: 1.35; }
</style>
</head>
<body>
<div class="toolbar">
  <button id="prevBtn">← Prev</button>
  <button id="nextBtn">Next →</button>
  <div id="counter"></div>
  <div class="legend">
    <span class="html">HTML</span>
    <span class="css">CSS</span>
    <span class="javascript">JavaScript</span>
    <span class="pad">PAD</span>
  </div>
</div>
<div id="examples"></div>
<script>
  const total = %TOTAL%;
  const counter = document.getElementById('counter');
  const container = document.getElementById('examples');
  const examples = %EXAMPLES_HTML%;
  // render
  examples.forEach((html, i) => {
    const d = document.createElement('div');
    d.className = 'example code' + (i===0 ? ' active' : '');
    d.innerHTML = html;
    container.appendChild(d);
  });
  function updateCounter(idx) {
    counter.textContent = `Example ${idx+1} / ${total}`;
  }
  let idx = 0;
  updateCounter(idx);
  function show(n) {
    const nodes = container.children;
    if (n < 0 || n >= nodes.length) return;
    nodes[idx].classList.remove('active');
    idx = n;
    nodes[idx].classList.add('active');
    updateCounter(idx);
  }
  document.getElementById('prevBtn').onclick = () => show(Math.max(0, idx-1));
  document.getElementById('nextBtn').onclick = () => show(Math.min(total-1, idx+1));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') show(Math.max(0, idx-1));
    if (e.key === 'ArrowRight') show(Math.min(total-1, idx+1));
  });
</script>
</body>
</html>
"""

def _escape_html(s: str) -> str:
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
              .replace('"',"&quot;").replace("'", "&#39;"))

def _labels_to_name(lid: int) -> str:
    return ID2LANG.get(int(lid), "pad") if lid != PAD_ID else "pad"

def _render_example_html(tokens_i32: np.ndarray, labels_u8: np.ndarray) -> str:
    """Render a single example (window) into an HTML snippet with spans colored by label."""
    L = int(tokens_i32.shape[0])
    valid = np.where(labels_u8 != PAD_ID)[0]
    last_valid = int(valid[-1]) + 1 if valid.size else 0

    html_parts: List[str] = []
    if last_valid > 0:
        vals, counts = np.unique(labels_u8[:last_valid], return_counts=True)
        dist = ", ".join(f"{_labels_to_name(int(v))}:{int(c)}" for v,c in zip(vals, counts))
    else:
        dist = "empty"
    html_parts.append(f'<div class="stats muted">valid_len={last_valid}, class_dist=[{_escape_html(dist)}]</div>')

    i = 0
    while i < L:
        label_id = labels_u8[i]
        j = i
        while j < L and labels_u8[j] == label_id:
            j += 1
        
        chunk_tokens = tokens_i32[i:j]
        # Decode bytes, ignoring PAD_BYTE_ID and errors
        valid_bytes = chunk_tokens[chunk_tokens < 256].astype(np.uint8).tobytes()
        text = valid_bytes.decode('utf-8', 'replace')

        class_name = _labels_to_name(label_id)
        html_parts.append(f'<span class="tok {class_name}">{_escape_html(text)}</span>')
        i = j
        
    return "".join(html_parts)

def build_preview_html(examples: List[Tuple[np.ndarray, np.ndarray]], out_path: str):
    html_snippets = [_render_example_html(x, y) for (x, y) in examples]
    
    def _js_quote(s: str) -> str:
        return json.dumps(s)

    js_array = "[" + ",".join(_js_quote(sn) for sn in html_snippets) + "]"
    html = _HTML_TEMPLATE.replace("%TOTAL%", str(len(html_snippets))).replace("%EXAMPLES_HTML%", js_array)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
