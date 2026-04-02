const STATIC_BASE = '/static/';
const WORKER_URL = `${STATIC_BASE}segmentor-worker.js`;
const SANITIZE_REGEX = /[^\x20-\x7E¤\n\t]/g;
const DEFAULT_THRESHOLD = 0.3;

const STATE = {
  manifest: null,
  worker: null,
  ready: false,
  busy: false,
  pending: new Map(),
  nextRequestId: 1,
  lastPayload: null,
  palette: new Map(),
  labelById: new Map(),
};

let tooltip = null;

function el(selector){
  return document.querySelector(selector);
}

function esc(text){
  return String(text || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function escAttr(text){
  return esc(text).replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}

function sanitizeToViewerText(text){
  return String(text || '').replace(/\r/g, '\n').replace(SANITIZE_REGEX, '¤');
}

function enforceSanitizedTextarea(textarea){
  if (!textarea){
    return;
  }
  const sanitized = sanitizeToViewerText(textarea.value);
  if (sanitized !== textarea.value){
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    textarea.value = sanitized;
    try{
      textarea.setSelectionRange(start, end);
    }catch(_err){
      // Selection updates can fail on blurred textareas. Ignore.
    }
  }
}

function countSanitizedBytes(text){
  return new TextEncoder().encode(sanitizeToViewerText(text)).length;
}

function clampThreshold(value){
  const numeric = Number(value);
  const fallback = Number(STATE.manifest?.other_threshold ?? DEFAULT_THRESHOLD);
  if (!Number.isFinite(numeric)){
    return Math.min(1, Math.max(0, fallback));
  }
  return Math.min(1, Math.max(0, numeric));
}

function formatThreshold(value){
  return clampThreshold(value).toFixed(2);
}

function getThresholdValue(){
  return clampThreshold(el('#thresholdInput')?.value);
}

function syncThresholdUi(value){
  const next = clampThreshold(value);
  const formatted = formatThreshold(next);
  const range = el('#thresholdRange');
  const input = el('#thresholdInput');
  if (range){
    range.value = formatted;
  }
  if (input){
    input.value = formatted;
  }
  el('#metaThreshold').textContent = `${formatted} other`;
  return next;
}

function slugify(value){
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function autoColor(index, total){
  const hue = ((index / Math.max(total, 1)) % 1) * 360;
  const saturation = 65;
  const lightness = 55;
  const s = saturation / 100;
  const l = lightness / 100;
  const c = (1 - Math.abs((2 * l) - 1)) * s;
  const hp = hue / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0;
  let g = 0;
  let b = 0;
  if (hp >= 0 && hp < 1){
    r = c; g = x;
  } else if (hp >= 1 && hp < 2){
    r = x; g = c;
  } else if (hp >= 2 && hp < 3){
    g = c; b = x;
  } else if (hp >= 3 && hp < 4){
    g = x; b = c;
  } else if (hp >= 4 && hp < 5){
    r = x; b = c;
  } else {
    r = c; b = x;
  }
  const m = l - (c / 2);
  const toHex = value => Math.round((value + m) * 255).toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function hexToRgb(color){
  const value = String(color || '').trim();
  if (value.startsWith('hsl(') || value.startsWith('hsla(')){
    return null;
  }
  let hex = value.replace('#', '');
  if (hex.length === 3){
    hex = hex.split('').map(ch => ch + ch).join('');
  }
  if (hex.length !== 6){
    return null;
  }
  const r = Number.parseInt(hex.slice(0, 2), 16);
  const g = Number.parseInt(hex.slice(2, 4), 16);
  const b = Number.parseInt(hex.slice(4, 6), 16);
  if ([r, g, b].some(Number.isNaN)){
    return null;
  }
  return { r, g, b };
}

function hexToRgba(color, alpha){
  const rgb = hexToRgb(color);
  const clamped = Math.max(0, Math.min(Number(alpha || 0), 1));
  if (!rgb){
    return `rgba(136, 136, 136, ${clamped.toFixed(2)})`;
  }
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${clamped.toFixed(2)})`;
}

function hexToRgbaConfidence(color, alpha, confidence){
  const rgb = hexToRgb(color);
  const clampedAlpha = Math.max(0, Math.min(Number(alpha || 0), 1));
  const clampedConfidence = Math.max(0, Math.min(Number(confidence || 0), 1));
  if (!rgb){
    return `rgba(136, 136, 136, ${clampedAlpha.toFixed(2)})`;
  }
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${(clampedAlpha * (0.35 + 0.65 * clampedConfidence)).toFixed(2)})`;
}

function labelName(labelId){
  const entry = STATE.labelById.get(Number(labelId));
  if (!entry){
    return String(labelId);
  }
  return entry.display;
}

function buildPalette(stats){
  const palette = new Map();
  const otherId = Number(STATE.manifest.num_classes);
  const ranked = (stats || [])
    .map(item => Number(item.id))
    .filter(id => id !== otherId);
  ranked.forEach((labelId, idx) => {
    palette.set(labelId, autoColor(idx, ranked.length));
  });
  palette.set(otherId, '#7f8c8d');
  return palette;
}

function buildLegend(stats){
  const holder = el('#legend');
  holder.innerHTML = '';
  (stats || []).forEach(item => {
    const labelId = Number(item.id);
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.innerHTML = `<span class="dot" style="background:${STATE.palette.get(labelId) || '#888888'}"></span>${esc(labelName(labelId))}`;
    holder.appendChild(chip);
  });
}

function renderStats(stats){
  const holder = el('#stats');
  holder.innerHTML = '';
  (stats || []).forEach(item => {
    const labelId = Number(item.id);
    const color = STATE.palette.get(labelId) || '#888888';
    const stat = document.createElement('div');
    stat.className = 'stat';
    stat.innerHTML = `
      <span class="dot" style="background:${color}"></span>
      <b>${esc(labelName(labelId))}</b>
      <span>${Number(item.count || 0)} chars</span>
      <div class="bar"><i style="width:${Number(item.pct || 0).toFixed(1)}%; background:${color}"></i></div>
      <span>${Number(item.pct || 0).toFixed(1)}%</span>
    `;
    holder.appendChild(stat);
  });
}

function renderSegments(payload){
  const render = el('#renderedOutput');
  if (!payload || !Array.isArray(payload.segments) || payload.segments.length === 0){
    render.classList.add('empty');
    render.textContent = 'No segmentation result yet.';
    return;
  }

  const text = String(payload.text || '');
  const probs = Array.isArray(payload.char_top_probs) ? payload.char_top_probs : [];
  const confidences = Array.isArray(payload.char_confidences) ? payload.char_confidences : [];
  const pieces = [];

  for (const segment of payload.segments){
    const labelId = Number(segment.label_id);
    const color = STATE.palette.get(labelId) || '#888888';
    const borderColor = hexToRgba(color, 0.35);
    for (let idx = Number(segment.start); idx < Number(segment.end); idx += 1){
      const ch = text[idx] || '';
      const confidence = Number(confidences[idx] || 0);
      const bg = hexToRgbaConfidence(color, 0.22, confidence);
      const topProbs = Array.isArray(probs[idx]) ? probs[idx] : [];
      const probsAttr = escAttr(JSON.stringify(topProbs));
      const style = `--seg-color:${color};background-color:${bg};box-shadow:inset 0 -1px 0 ${borderColor};`;
      if (ch === '\n'){
        pieces.push(
          `<span class="char newline" style="${style}" data-probs="${probsAttr}" data-label-id="${labelId}"><br></span>`
        );
      } else {
        pieces.push(
          `<span class="char" style="${style}" data-probs="${probsAttr}" data-label-id="${labelId}">${esc(ch)}</span>`
        );
      }
    }
  }

  render.classList.remove('empty');
  render.innerHTML = pieces.join('');
}

function updateResultMeta(payload){
  const meta = el('#resultMeta');
  if (!payload){
    meta.textContent = 'No segmentation result yet.';
    return;
  }
  meta.textContent = `${Number(payload.input_bytes || 0)} bytes, ${Number((payload.text || '').length)} characters, threshold ${formatThreshold(payload.other_threshold)}`;
}

function setStatus(message, mode='neutral'){
  const status = el('#statusLine');
  status.textContent = message;
}

function setRuntimeBadge(text, mode='neutral'){
  const badge = el('#runtimeBadge');
  badge.textContent = text;
  badge.classList.remove('neutral', 'success', 'error');
  badge.classList.add(mode);
}

function updateByteCounter(){
  const counter = el('#byteCounter');
  const limit = Number(STATE.manifest?.max_input_bytes || 0);
  const value = countSanitizedBytes(el('#inputText').value);
  counter.textContent = limit > 0 ? `${value} / ${limit} demo cap` : `${value} bytes`;
  counter.style.color = limit > 0 && value > limit ? '#b42318' : '';
}

function updateInputHint(){
  const hint = el('#inputHint');
  if (!hint){
    return;
  }
  const limit = Number(STATE.manifest?.max_input_bytes || 0);
  const limitText = limit > 0
    ? `This public demo caps input at ${limit} bytes so the current Pyodide/WASM runtime does not bog down or freeze the browser on large pastes.`
    : 'This build does not enforce a demo input-size cap.';
  hint.textContent = `Browser-side inference means the first load may take a few seconds while the runtime and model assets warm up. Long inputs run through the native Mamba sequence path in one pass. ${limitText}`;
}

function clearOutput(){
  STATE.lastPayload = null;
  STATE.palette = new Map();
  el('#renderedOutput').classList.add('empty');
  el('#renderedOutput').textContent = 'Segment results will appear here.';
  el('#stats').innerHTML = '';
  el('#legend').innerHTML = '';
  el('#elapsedBadge').textContent = '-';
  updateResultMeta(null);
}

function copyHtml(){
  const render = el('#renderedOutput');
  navigator.clipboard.writeText(render.innerHTML || '').then(() => {
    setStatus('Rendered HTML copied to the clipboard.', 'success');
  }).catch(err => {
    setStatus(`Copy failed: ${err.message}`, 'error');
  });
}

function downloadJson(){
  if (!STATE.lastPayload){
    setStatus('Run a segmentation first to download JSON.', 'error');
    return;
  }
  const blob = new Blob([JSON.stringify(STATE.lastPayload, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'content-type-segmentor.json';
  link.click();
  URL.revokeObjectURL(link.href);
}

function showTooltip(event){
  const target = event.target;
  if (!target || !target.classList || !target.classList.contains('char')){
    hideTooltip();
    return;
  }
  if (!tooltip){
    tooltip = el('#tooltip');
  }

  const probs = JSON.parse(target.dataset.probs || '[]');
  const labelId = Number(target.dataset.labelId || -1);
  let html = `<div class="prob-bar"><div class="label"><b>${esc(labelName(labelId))}</b></div></div>`;
  for (const item of probs){
    const itemId = Number(item.id);
    const prob = Number(item.prob || 0);
    const percentage = (prob * 100).toFixed(1);
    const color = STATE.palette.get(itemId) || '#888888';
    html += `
      <div class="prob-bar">
        <div class="label">${esc(labelName(itemId))}</div>
        <div class="bar">
          <div class="fill" style="width:${percentage}%; --color:${color}"></div>
        </div>
        <div class="value">${percentage}%</div>
      </div>
    `;
  }

  tooltip.innerHTML = html;
  tooltip.style.display = 'block';

  const rect = target.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();
  let left = rect.left;
  let top = rect.bottom + 8;
  if (left + tooltipRect.width > window.innerWidth){
    left = window.innerWidth - tooltipRect.width - 8;
  }
  if (top + tooltipRect.height > window.innerHeight){
    top = rect.top - tooltipRect.height - 8;
  }
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function hideTooltip(){
  if (tooltip){
    tooltip.style.display = 'none';
  }
}

function setBusy(busy){
  STATE.busy = Boolean(busy);
  for (const id of ['segmentBtn', 'loadDemoBtn', 'clearBtn', 'copyHtmlBtn', 'downloadJsonBtn']){
    el(`#${id}`).disabled = STATE.busy || (!STATE.ready && id !== 'clearBtn');
  }
  for (const id of ['thresholdRange', 'thresholdInput']){
    el(`#${id}`).disabled = STATE.busy;
  }
}

function handleWorkerMessage(event){
  const { type, requestId, payload } = event.data || {};
  const entry = STATE.pending.get(requestId);
  if (!entry){
    return;
  }
  STATE.pending.delete(requestId);
  if (type === 'error'){
    entry.reject(new Error(payload?.message || 'Unknown worker error.'));
    return;
  }
  entry.resolve(payload);
}

function callWorker(type, payload){
  return new Promise((resolve, reject) => {
    const requestId = STATE.nextRequestId++;
    STATE.pending.set(requestId, { resolve, reject });
    STATE.worker.postMessage({ type, requestId, payload });
  });
}

async function loadManifest(){
  const response = await fetch(`${STATIC_BASE}assets/model_manifest.json`);
  if (!response.ok){
    throw new Error(`Failed to load manifest (${response.status}).`);
  }
  STATE.manifest = await response.json();
  STATE.labelById = new Map();
  (STATE.manifest.label_order || []).forEach((label, idx) => {
    STATE.labelById.set(idx, {
      canonical: label,
      display: (STATE.manifest.display_labels || [])[idx] || label,
    });
  });
  STATE.labelById.set(Number(STATE.manifest.num_classes), {
    canonical: 'other',
    display: 'other',
  });

  el('#modelBadge').textContent = STATE.manifest.model_id || 'sfullfiles4';
  el('#metaModel').textContent = STATE.manifest.model_id || 'sfullfiles4';
  el('#metaInference').textContent = 'Native Mamba';
  syncThresholdUi(STATE.manifest.other_threshold);
  updateInputHint();
  updateByteCounter();
}

async function loadDemoText(){
  const path = String(STATE.manifest?.default_text_path || 'assets/demo_input.txt').replace(/^\//, '');
  const response = await fetch(`${STATIC_BASE}${path}`);
  if (!response.ok){
    throw new Error(`Failed to load demo input (${response.status}).`);
  }
  const text = await response.text();
  const textarea = el('#inputText');
  textarea.value = sanitizeToViewerText(text);
  updateByteCounter();
}

async function ensureWorkerLoaded(){
  setStatus('Loading the client-side runtime and model assets...', 'neutral');
  setBusy(true);
  try{
    const meta = await callWorker('load', {});
    STATE.ready = true;
    setRuntimeBadge(`Client-side ${meta.runtime}`, 'success');
    setStatus(`Client-side runtime ready: ${meta.runtime}.`, 'success');
    return meta;
  }catch(err){
    STATE.ready = false;
    setRuntimeBadge('Browser runtime failed', 'error');
    setStatus(`Failed to initialize browser inference: ${err.message}`, 'error');
    throw err;
  }finally{
    setBusy(false);
  }
}

async function segmentCurrentText(){
  const textarea = el('#inputText');
  enforceSanitizedTextarea(textarea);
  updateByteCounter();

  const inputBytes = countSanitizedBytes(textarea.value);
  const maxBytes = Number(STATE.manifest?.max_input_bytes || 0);
  if (maxBytes > 0 && inputBytes > maxBytes){
    setStatus(
      `Input exceeds the public demo limit of ${maxBytes} bytes after sanitization. The cap is only there to keep this Pyodide/WASM demo responsive.`,
      'error'
    );
    return;
  }
  if (!textarea.value.trim()){
    setStatus('Paste or load a sample before running segmentation.', 'error');
    return;
  }

  setBusy(true);
  setStatus('Running sfullfiles4 in your browser...', 'neutral');
  try{
    const payload = await callWorker('segment', {
      text: textarea.value,
      threshold: getThresholdValue(),
    });
    STATE.lastPayload = payload;
    STATE.palette = buildPalette(payload.stats || []);
    renderSegments(payload);
    buildLegend(payload.stats || []);
    renderStats(payload.stats || []);
    updateResultMeta(payload);
    el('#elapsedBadge').textContent = `${Number(payload.elapsed_ms || 0).toFixed(1)} ms`;
    setStatus(`Segmented successfully with ${payload.runtime}.`, 'success');
  }catch(err){
    setStatus(`Segmentation failed: ${err.message}`, 'error');
  }finally{
    setBusy(false);
  }
}

function bindUi(){
  document.addEventListener('mousemove', showTooltip);
  document.addEventListener('mouseleave', hideTooltip);

  const textarea = el('#inputText');
  const thresholdRange = el('#thresholdRange');
  const thresholdInput = el('#thresholdInput');
  textarea.addEventListener('input', () => {
    enforceSanitizedTextarea(textarea);
    updateByteCounter();
  });
  textarea.addEventListener('blur', () => enforceSanitizedTextarea(textarea));

  thresholdRange.addEventListener('input', () => {
    syncThresholdUi(thresholdRange.value);
  });
  thresholdRange.addEventListener('change', () => {
    syncThresholdUi(thresholdRange.value);
    if (STATE.lastPayload && !STATE.busy){
      setStatus(`Threshold updated to ${formatThreshold(thresholdRange.value)}. Click Segment to re-run.`, 'neutral');
    }
  });
  thresholdInput.addEventListener('change', () => {
    syncThresholdUi(thresholdInput.value);
    if (STATE.lastPayload && !STATE.busy){
      setStatus(`Threshold updated to ${formatThreshold(thresholdInput.value)}. Click Segment to re-run.`, 'neutral');
    }
  });
  thresholdInput.addEventListener('blur', () => {
    syncThresholdUi(thresholdInput.value);
  });

  el('#segmentBtn').addEventListener('click', segmentCurrentText);
  el('#loadDemoBtn').addEventListener('click', async () => {
    try{
      await loadDemoText();
      setStatus('Loaded the default demo sample.', 'success');
    }catch(err){
      setStatus(`Failed to load the demo sample: ${err.message}`, 'error');
    }
  });
  el('#clearBtn').addEventListener('click', () => {
    el('#inputText').value = '';
    updateByteCounter();
    clearOutput();
    setStatus('Cleared the current input and result.', 'neutral');
  });
  el('#copyHtmlBtn').addEventListener('click', copyHtml);
  el('#downloadJsonBtn').addEventListener('click', downloadJson);
}

async function bootstrap(){
  bindUi();
  clearOutput();
  updateByteCounter();

  STATE.worker = new Worker(WORKER_URL);
  STATE.worker.addEventListener('message', handleWorkerMessage);

  try{
    await loadManifest();
    await loadDemoText();
    await ensureWorkerLoaded();
    await segmentCurrentText();
  }catch(err){
    console.error(err);
  }
}

bootstrap();
