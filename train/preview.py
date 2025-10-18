"""
Utilities for generating an HTML preview of augmented data samples.
"""
import os
import json
import colorsys
from typing import List, Tuple, Dict
import numpy as np

import config

def generate_color(index, total):
    """Generate a distinct color from HSL color space"""
    import colorsys
    hue = index / total
    saturation = 0.7
    lightness = 0.5
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

# Generate CSS for all language classes
def generate_lang_css():
    num_classes = len(config.ID2LANG)
    css_rules = []
    for lang_id, lang_name in config.ID2LANG.items():
        color = generate_color(lang_id, num_classes)
        # Legend style (solid color)
        css_rules.append(f".{lang_name} {{ background:{color}; }}")
        # Token style (transparent background)
        css_rules.append(f".tok.{lang_name} {{ background:rgba({int(int(color[1:3], 16))}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15); }}")
    return "\n  ".join(css_rules)

def generate_legend_spans():
    """Generate legend spans for all languages"""
    spans = []
    for lang_id, lang_name in sorted(config.ID2LANG.items()):
        spans.append(f'<span class="{lang_name}">{lang_name.upper()}</span>')
    spans.append('<span class="pad">PAD</span>')
    return "\n    ".join(spans)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Data Preview</title>
<style>
  body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; margin: 24px; }
  .toolbar { display:flex; gap:12px; align-items:center; margin-bottom:12px; }
  button { padding:6px 10px; }
  .example-legend { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
  .chip { display:inline-block; padding:2px 8px; border-radius:4px; margin-right:0; color:#fff; font-size:12px; }
  .muted { color:#666; font-size:0.9em; }
  .example { display:none; white-space:pre-wrap; word-break:break-word; border:1px solid #eee; padding:12px; border-radius:8px; background:#fafafa; }
  .example.active { display:block; }
  .tok { transition: all 0.15s ease; }
  .tok:hover { filter: brightness(0.9); }
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
    mapping = config.ID2LANG
    pad_id = config.PAD_ID
    lid_int = int(lid)
    return mapping.get(lid_int, "pad") if lid_int != pad_id else "pad"

def generate_example_colors(labels_u8: np.ndarray) -> dict:
    """Generate colors for unique labels in this example."""
    pad_id = config.PAD_ID
    unique_labels = sorted(int(x) for x in np.unique(labels_u8) if x != pad_id)
    
    # Create a mapping of actual label indices to dense indices (0, 1, 2, ...)
    # This ensures colors are consistently spaced regardless of which IDs are present
    dense_indices = {label: idx for idx, label in enumerate(unique_labels)}
    total_labels = len(unique_labels)
    
    colors = {}
    for label in unique_labels:
        # Use the dense index for color generation to ensure even spacing
        hue = dense_indices[label] / max(1, total_labels)
        # Tweak saturation and lightness for better visibility
        saturation = 0.65
        lightness = 0.6
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors[label] = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    return colors

def _render_example_html(tokens_i32: np.ndarray, labels_u8: np.ndarray) -> str:
    """Render a single example (window) into an HTML snippet with spans colored by label."""
    L = int(tokens_i32.shape[0])
    pad_id = config.PAD_ID
    valid = np.where(labels_u8 != pad_id)[0]
    last_valid = int(valid[-1]) + 1 if valid.size else 0

    # Generate colors for this specific example
    colors = generate_example_colors(labels_u8)
    
    # Create legend for this example - sort by language name for consistent ordering
    legend_parts = []
    sorted_labels = sorted(colors.items(), key=lambda x: _labels_to_name(x[0]))
    for label_id, color in sorted_labels:
        lang_name = _labels_to_name(label_id)
        legend_parts.append(
            f'<span class="chip" style="background: {color};" title="ID: {label_id}">'
            f'{_escape_html(lang_name.upper())}</span>'
        )
    if legend_parts:
        legend_html = '<div class="example-legend">' + "".join(legend_parts) + '</div>'
    else:
        legend_html = ''

    html_parts: List[str] = []
    if last_valid > 0:
        vals, counts = np.unique(labels_u8[:last_valid], return_counts=True)
        dist = ", ".join(f"{_labels_to_name(int(v))}:{int(c)}" for v,c in zip(vals, counts))
    else:
        dist = "empty"
    html_parts.append(f'<div class="stats muted">valid_len={last_valid}, class_dist=[{_escape_html(dist)}]</div>')
    html_parts.append(legend_html)

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

        if label_id == pad_id:
            html_parts.append(f'<span class="tok pad" title="PAD">{_escape_html(text)}</span>')
        else:
            color = colors[int(label_id)]
            lang_name = _labels_to_name(label_id)
            html_parts.append(
                f'<span class="tok" style="background:rgba({int(int(color[1:3], 16))}, '
                f'{int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15);" '
                f'title="{_escape_html(lang_name.upper())}">{_escape_html(text)}</span>'
            )
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
