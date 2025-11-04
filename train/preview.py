"""
Utilities for generating an HTML preview of augmented data samples.
"""
import os
import json
import colorsys
from typing import List, Dict, Any, Optional
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
  .meta { margin-bottom:12px; border:1px solid #e3e3e3; background:#fff; padding:10px 12px; border-radius:8px; font-size:12px; line-height:1.4; box-shadow:0 1px 2px rgba(0,0,0,0.03); }
  .meta-mode { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:6px; }
  .badge { display:inline-flex; align-items:center; justify-content:center; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; background:#444; color:#fff; }
  .badge.mode-pure { background:#4caf50; }
  .badge.mode-mixed { background:#3f51b5; }
  .badge.mode-line_inject { background:#ff9800; }
  .meta-note { font-size:11.5px; color:#555; }
  .meta-length { font-weight:600; color:#333; }
  .meta-segments { font-size:11.5px; color:#555; display:block; }
  .meta-table { width:100%; border-collapse:collapse; margin-top:6px; }
  .meta-table th, .meta-table td { padding:4px 6px; border-top:1px solid #f0f0f0; text-align:left; vertical-align:top; }
  .meta-table th { background:#f6f6f6; font-size:11.5px; font-weight:600; color:#444; }
  .meta-table td { font-size:11.5px; color:#333; }
  .meta-table td.meta-source { font-family:inherit; word-break:break-all; }
  .meta-section { margin-top:6px; }
  .meta-section summary { cursor:pointer; font-weight:600; color:#333; }
  .meta-list { margin:6px 0 0 18px; padding:0; list-style:disc; }
  .meta-list li { margin-bottom:6px; }
  .meta-preview { margin-top:4px; padding:6px 8px; background:#f0f4ff; border-radius:4px; white-space:pre-wrap; border:1px solid #d9e2ff; }
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


def _shorten(text: Optional[str], limit: int = 80) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _format_sample_measure(sample: Dict[str, Any]) -> str:
    if sample.get("final_bytes") is not None:
        return f"{int(sample['final_bytes'])} B"
    if sample.get("bytes") is not None:
        return f"{int(sample['bytes'])} B"
    if sample.get("chars") is not None:
        return f"{int(sample['chars'])} chars"
    return "—"


def _render_metadata_summary(metadata: Optional[Dict[str, Any]]) -> str:
    if not metadata:
        return ""

    parts: List[str] = ['<div class="meta">']
    mode_key = metadata.get("mode", "unknown")
    mode_display = mode_key.replace("_", " ").upper() if isinstance(mode_key, str) else str(mode_key)
    badge_class = f"badge mode-{mode_key}" if isinstance(mode_key, str) else "badge"
    parts.append('<div class="meta-mode">')
    parts.append(f'<span class="{badge_class}">{_escape_html(mode_display)}</span>')
    length = metadata.get("length")
    if length:
        parts.append(f'<span class="meta-note">window {int(length)} bytes</span>')
    requested_key = metadata.get("requested_mode")
    if requested_key and requested_key != mode_key:
        requested_display = requested_key.replace("_", " ").upper() if isinstance(requested_key, str) else str(requested_key)
        parts.append(f'<span class="meta-note">requested {_escape_html(requested_display)}</span>')

    pair_meta = metadata.get("language_pair_mode")
    if isinstance(pair_meta, dict) and pair_meta.get("selected"):
        langs = pair_meta.get("languages") or []
        if langs:
            pair_display = " + ".join(str(lang).upper() for lang in langs)
        else:
            ids = pair_meta.get("ids") or []
            pair_display = " + ".join(f"ID {int(lid)}" for lid in ids)
        status_bits = []
        prob = pair_meta.get("probability")
        if isinstance(prob, (int, float)) and prob > 0:
            status_bits.append(f"p={prob:.0%}")
        if not pair_meta.get("active", True):
            reason = str(pair_meta.get("reason") or "").replace("_", " ").strip()
            status_bits.append(f"inactive: {reason}" if reason else "inactive")
        else:
            host_name = pair_meta.get("host_language")
            if host_name:
                status_bits.append(f"host {str(host_name).upper()}")
            donors = pair_meta.get("donor_candidate_languages") or []
            if donors:
                donor_disp = " & ".join(str(name).upper() for name in donors)
                status_bits.append(f"donors {donor_disp}")
            unlocked = pair_meta.get("unlocked_languages") or []
            if unlocked:
                unlocked_disp = " & ".join(str(name).upper() for name in unlocked)
                status_bits.append(f"unlocked {unlocked_disp}")
            expanded = pair_meta.get("expanded_languages") or []
            if expanded:
                expanded_disp = " & ".join(str(name).upper() for name in expanded)
                status_bits.append(f"expanded {expanded_disp}")
            if pair_meta.get("applied"):
                status_bits.append("applied")
            else:
                used = pair_meta.get("used_languages") or []
                missing = pair_meta.get("missing_languages") or []
                if used:
                    used_disp = " & ".join(str(name).upper() for name in used)
                    status_bits.append(f"used {used_disp}")
                if missing:
                    missing_disp = " & ".join(str(name).upper() for name in missing)
                    status_bits.append(f"missing {missing_disp}")
                if not used and not missing:
                    status_bits.append("not used")
        pair_note = f"pair {pair_display}"
        if status_bits:
            pair_note += " (" + "; ".join(status_bits) + ")"
        parts.append(f'<span class="meta-note">{_escape_html(pair_note)}</span>')

        unlock_trace = pair_meta.get("unlock_trace") or []
        if unlock_trace:
            parts.append('<details class="meta-section"><summary>Language Unlocks</summary><ul class="meta-list">')
            for idx, event in enumerate(unlock_trace, start=1):
                event_type = _escape_html(str(event.get("event", "event")))
                trigger_lang = event.get("trigger_language")
                trigger_text = f" by {_escape_html(str(trigger_lang).upper())}" if trigger_lang else ""
                new_langs = event.get("new_languages") or []
                avail_langs = event.get("available_languages") or []
                new_text = f" → new: {', '.join(_escape_html(str(lang).upper()) for lang in new_langs)}" if new_langs else ""
                avail_text = f" | available: {', '.join(_escape_html(str(lang).upper()) for lang in avail_langs)}" if avail_langs else ""
                parts.append(f"<li>{idx}. {event_type}{trigger_text}{new_text}{avail_text}</li>")
            parts.append('</ul></details>')

    final_segments = metadata.get("final_segments") or []
    if final_segments:
        seg_bits = []
        for seg in final_segments:
            lang = seg.get("language") or config.ID2LANG.get(int(seg.get("language_id", -1)), "?")
            length_b = seg.get("length")
            seg_bits.append(f'{_escape_html(str(lang).upper())} <span class="meta-length">({int(length_b)} B)</span>')
        parts.append(f'<span class="meta-segments">segments: {" → ".join(seg_bits)}</span>')
    parts.append('</div>')

    samples = metadata.get("samples") or []
    if samples:
        parts.append('<details class="meta-section" open>')
        parts.append('<summary>Source Contributions</summary>')
        parts.append('<table class="meta-table"><thead><tr><th>Origin</th><th>Language</th><th>Span</th><th>Source</th><th>Notes</th></tr></thead><tbody>')
        for sample in samples:
            origin = _escape_html(str(sample.get("origin", "base")).replace("_", " ").title())
            lang = sample.get("language")
            if not lang and sample.get("language_id") is not None:
                lang = config.ID2LANG.get(int(sample["language_id"]), str(sample["language_id"]))
            lang_display = _escape_html(str(lang).upper()) if lang else "?"
            span_parts = []
            final_spans = sample.get("final_spans") or []
            if final_spans:
                first_span = final_spans[0]
                if first_span.get("end", 0) > first_span.get("start", 0):
                    span_parts.append(f"{int(first_span['start'])}-{int(first_span['end'])}")
                if len(final_spans) > 1:
                    span_parts.append(f"+{len(final_spans) - 1} more")
            elif sample.get("start") is not None and sample.get("end") is not None and sample.get("end") > sample.get("start"):
                span_parts.append(f"{int(sample['start'])}-{int(sample['end'])}")
            measure = _format_sample_measure(sample)
            if measure != "—":
                span_parts.append(measure)
            span = "<br>".join(span_parts) if span_parts else measure
            source = _escape_html(_shorten(sample.get("source"), 100))
            notes_parts = []
            if sample.get("requested_bytes") is not None:
                notes_parts.append(f"requested {int(sample['requested_bytes'])} B")
            if sample.get("requested_chars") is not None:
                notes_parts.append(f"requested {int(sample['requested_chars'])} chars")
            if sample.get("status"):
                notes_parts.append(str(sample["status"]))
            parts.append(
                "<tr>"
                f"<td>{origin}</td>"
                f"<td>{lang_display}</td>"
                f"<td>{span}</td>"
                f"<td class=\"meta-source\">{source}</td>"
                f"<td>{_escape_html(', '.join(notes_parts)) if notes_parts else ''}</td>"
                "</tr>"
            )
        parts.append("</tbody></table></details>")

    injections = metadata.get("line_injections") or []
    if injections:
        parts.append(f'<details class="meta-section" open><summary>Line Injections ({len(injections)})</summary>')
        parts.append('<ul class="meta-list">')
        for inj in injections:
            lang = inj.get("language")
            if not lang and inj.get("language_id") is not None:
                lang = config.ID2LANG.get(int(inj["language_id"]), str(inj["language_id"]))
            lang_display = _escape_html(str(lang).upper())
            source = _escape_html(_shorten(inj.get("source"), 100))
            char_idx = inj.get("insert_char_index")
            chars_inserted = inj.get("chars_inserted")
            pre_nl = inj.get("newlines_before", 0)
            post_nl = inj.get("newlines_after", 0)
            preview = inj.get("preview")
            final_bytes = inj.get("final_bytes")
            final_spans = inj.get("final_spans") or []
            span_desc = ""
            if final_spans:
                first = final_spans[0]
                if first.get("end", 0) > first.get("start", 0):
                    span_desc = f" span {int(first['start'])}-{int(first['end'])}"
                if len(final_spans) > 1:
                    span_desc += f" (+{len(final_spans)-1} more)"
            metrics = []
            if final_bytes is not None:
                metrics.append(f"{int(final_bytes)} B in window")
            if chars_inserted is not None:
                metrics.append(f"{int(chars_inserted)} chars inserted")
            metrics.append(f"pre {pre_nl} / post {post_nl} newlines")
            parts.append(
                "<li>"
                f"<strong>{lang_display}</strong> @ char {int(char_idx) if char_idx is not None else '?'}"
                f"{' ' + span_desc if span_desc else ''}"
                f" ({', '.join(metrics)}) from {source}"
            )
            if preview:
                parts.append(f'<div class="meta-preview">{_escape_html(_shorten(preview, 240))}</div>')
            parts.append("</li>")
        parts.append("</ul></details>")

    overlays = metadata.get("overlays") or []
    if overlays:
        parts.append(f'<details class="meta-section"><summary>Overlays ({len(overlays)})</summary>')
        parts.append('<ul class="meta-list">')
        for idx, overlay in enumerate(overlays, start=1):
            start = overlay.get("start")
            end = overlay.get("end")
            seg_desc = []
            for seg in overlay.get("segments", []):
                lang = seg.get("language") or config.ID2LANG.get(int(seg.get("language_id", -1)), "?")
                bytes_len = seg.get("bytes") or seg.get("end", 0) - seg.get("start", 0)
                src = _escape_html(_shorten(seg.get("source"), 80))
                seg_desc.append(f"{_escape_html(str(lang).upper())} {int(bytes_len)}B ({src})")
            segment_text = "; ".join(seg_desc) if seg_desc else "mixed slice"
            parts.append(
                "<li>"
                f"<strong>Overlay {idx}</strong> [{int(start) if start is not None else '?'} - {int(end) if end is not None else '?'}): "
                f"{segment_text}"
                "</li>"
            )
        parts.append("</ul></details>")

    parts.append("</div>")
    return "".join(parts)

def _render_example_html(tokens_i32: np.ndarray, labels_u8: np.ndarray, metadata: Optional[Dict[str, Any]]) -> str:
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

    meta_block = _render_metadata_summary(metadata)
    if meta_block:
        html_parts.append(meta_block)
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

def build_preview_html(examples: List[Dict[str, Any]], out_path: str):
    html_snippets = []
    for ex in examples:
        tokens = ex.get("tokens")
        labels = ex.get("labels")
        meta = ex.get("metadata")
        html_snippets.append(_render_example_html(tokens, labels, meta))
    
    def _js_quote(s: str) -> str:
        return json.dumps(s)

    js_array = "[" + ",".join(_js_quote(sn) for sn in html_snippets) + "]"
    html = _HTML_TEMPLATE.replace("%TOTAL%", str(len(html_snippets))).replace("%EXAMPLES_HTML%", js_array)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
