let CONFUSION = null;
let LABELS = [];
let ACTIVE_DATASET = null;
let DATASET_OPTIONS = [];
let tooltip = null;
let exampleRequestController = null;
const DEFAULT_CELL_WIDTH = 34;
const MIN_CELL_WIDTH = 22;
const TRUE_AXIS_WIDTH = 70;
const LONG_RUN_THRESHOLD = 60;

function el(sel){ return document.querySelector(sel); }
function esc(str){ return (str || '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'); }

function formatPct(value){
  if (!isFinite(value)) return '0%';
  return (value * 100).toFixed(value >= 0.1 ? 1 : 2) + '%';
}

function buildMetrics(aggregates){
  const holder = el('#aggMetrics');
  holder.innerHTML = '';
  const items = [
    ['Micro Acc', aggregates?.micro?.acc],
    ['Macro F1', aggregates?.macro?.f1],
    ['Weighted F1', aggregates?.weighted?.f1]
  ];
  items.forEach(([label, value]) => {
    if (typeof value !== 'number') return;
    const span = document.createElement('span');
    span.className = 'metric';
    span.textContent = `${label}: ${(value * 100).toFixed(2)}%`;
    holder.appendChild(span);
  });
}

function setMatrixLoading(message = 'Loading confusion matrix…'){
  const matrixEl = el('#matrix');
  if (!matrixEl) return;
  matrixEl.classList.add('loading');
  matrixEl.textContent = message;
  if (matrixEl.previousElementSibling && matrixEl.previousElementSibling.classList.contains('matrix-note')){
    matrixEl.previousElementSibling.remove();
  }
}

function resetSampleView(){
  if (exampleRequestController){
    exampleRequestController.abort();
    exampleRequestController = null;
  }
  const meta = el('#cellMeta');
  if (meta){
    meta.textContent = 'Select any cell to view examples.';
  }
  const stats = el('#sampleStats');
  if (stats){
    stats.innerHTML = '';
  }
  const palette = el('#samplePalette');
  if (palette){
    palette.hidden = true;
    palette.innerHTML = '';
  }
  const pred = el('#samplePred');
  if (pred){
    pred.textContent = 'Nothing selected.';
  }
  const truth = el('#sampleTruth');
  if (truth){
    truth.textContent = 'No ground truth.';
  }
  const metaBox = el('#sampleMeta');
  if (metaBox){
    metaBox.textContent = '';
  }
}

function updateDatasetSwitch(){
  const holder = el('#datasetSwitch');
  if (!holder){
    return;
  }
  if (!DATASET_OPTIONS.length){
    holder.innerHTML = '';
    holder.style.display = 'none';
    return;
  }
  holder.style.display = 'flex';
  holder.innerHTML = '';
  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = 'Dataset';
  holder.appendChild(label);
  DATASET_OPTIONS.forEach(option => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = option.name || option.id;
    if (option.summary){
      btn.title = option.summary;
    }
    if (option.id === ACTIVE_DATASET){
      btn.classList.add('active');
    }
    if (!option.available){
      btn.disabled = true;
    }
    btn.addEventListener('click', () => {
      if (option.id === ACTIVE_DATASET || !option.available){
        return;
      }
      loadDataset(option.id);
    });
    holder.appendChild(btn);
  });
}

function getMatrixWrapperWidth(){
  const wrapper = document.querySelector('.matrix-wrapper');
  if (!wrapper){
    return window.innerWidth || 1200;
  }
  const style = getComputedStyle(wrapper);
  const padding = parseFloat(style.paddingLeft || '0') + parseFloat(style.paddingRight || '0');
  return Math.max(0, wrapper.clientWidth - padding);
}

function computeCellWidth(columnCount){
  if (!columnCount){
    return DEFAULT_CELL_WIDTH;
  }
  const available = getMatrixWrapperWidth() - TRUE_AXIS_WIDTH - 8;
  if (available <= 0){
    return MIN_CELL_WIDTH;
  }
  const width = Math.floor(available / columnCount);
  return Math.max(MIN_CELL_WIDTH, Math.min(DEFAULT_CELL_WIDTH, width));
}

function renderMatrix(data){
  const matrixEl = el('#matrix');
  matrixEl.classList.remove('loading');
  if (matrixEl.previousElementSibling && matrixEl.previousElementSibling.classList.contains('matrix-note')){
    matrixEl.previousElementSibling.remove();
  }
  const n = data.labels.length;
  const cellWidth = computeCellWidth(n);
  matrixEl.style.setProperty('--cell-size', `${cellWidth}px`);
  matrixEl.style.gridTemplateColumns = `${TRUE_AXIS_WIDTH}px repeat(${n}, ${cellWidth}px)`;
  matrixEl.innerHTML = '';
  matrixEl.appendChild(makeAxisCell('True ↓ / Pred →', 'corner'));
  data.labels.forEach(lbl => {
    const cell = makeAxisCell(shortLabel(lbl.name), 'col');
    cell.title = `Predicted ${lbl.name}`;
    matrixEl.appendChild(cell);
  });
  const note = document.createElement('div');
  note.className = 'matrix-note';
  note.textContent = 'Shading uses log-scale row-normalized percentages.';
  matrixEl.before(note);
  for (let row = 0; row < n; row++){
    const axis = makeAxisCell(shortLabel(data.labels[row].name), 'row');
    axis.title = `True ${data.labels[row].name}`;
    matrixEl.appendChild(axis);
    for (let col = 0; col < n; col++){
      const pct = data.row_normalized[row][col] || 0;
      const count = data.matrix[row][col] || 0;
      const pool = Number(data.cell_examples?.[`${row}_${col}`] || 0);
      const hasExample = pool > 0;
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.row = row;
      cell.dataset.col = col;
      cell.dataset.pool = String(pool);
      cell.innerHTML = `<div class="count">${count}×</div>
        <div class="pct">${(pct * 100).toFixed(1)}%</div>`;
      const logComponent = Math.min(1, Math.log10(pct * 9 + 1));
      const intensity = count > 0 ? logComponent : 0;
      cell.style.background = `rgba(11,94,215,${intensity * 0.7})`;
      if (hasExample){
        cell.title = `${data.labels[row].name} → ${data.labels[col].name}: ${(pct * 100).toFixed(2)}% (${count}, pool ${pool})`;
        cell.addEventListener('click', () => loadExample(row, col));
      }else{
        cell.classList.add('empty');
        cell.title = `${data.labels[row].name} → ${data.labels[col].name}: no cached example`;
      }
      matrixEl.appendChild(cell);
    }
  }
}

function makeAxisCell(text, variant){
  const div = document.createElement('div');
  div.className = 'axis';
  if (variant){
    div.classList.add(`axis-${variant}`);
  }
  div.textContent = text;
  return div;
}

function shortLabel(name){
  if (!name) return '';
  if (name.toLowerCase() === 'javascript_typescript') return 'js_ts';
  if (name.startsWith('encoding_')) return name.replace('encoding_', '');
  return name;
}

async function loadExample(trueId, predId){
  const meta = el('#cellMeta');
  meta.textContent = 'Loading example…';
  if (exampleRequestController){
    exampleRequestController.abort();
  }
  const controller = new AbortController();
  exampleRequestController = controller;
  setSampleLoading(true);
  try{
    const params = new URLSearchParams({
      true_id: String(trueId),
      pred_id: String(predId),
      ts: String(Date.now())
    });
    if (ACTIVE_DATASET){
      params.set('dataset', ACTIVE_DATASET);
    }
    const res = await fetch(`/api/example?${params.toString()}`, {
      cache: 'no-store',
      signal: controller.signal
    });
    if (!res.ok){
      const err = await res.json().catch(()=>({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    updateSample(data);
  }catch(err){
    if (err.name === 'AbortError'){
      return;
    }
    if (err.message === 'No cached examples for this cell.'){
      meta.textContent = 'No cached example for this cell. Pick a highlighted cell to inspect.';
      return;
    }
    meta.textContent = `Error: ${err.message}`;
  }finally{
    if (exampleRequestController === controller){
      exampleRequestController = null;
      setSampleLoading(false);
    }
  }
}

function updateSample(data){
  if (!data || !data.cell){
    console.warn('updateSample received invalid payload', data);
    return;
  }
  const meta = el('#cellMeta');
  if (!meta){
    return;
  }
  const cell = data.cell;
  meta.innerHTML = `
    <span class="cell-heading">
      <strong>${esc(cell.true.name)}</strong>
      <span aria-hidden="true">→</span>
      <strong>${esc(cell.pred.name)}</strong>
    </span>
    &nbsp;• ${cell.count} hits (${(cell.row_pct * 100).toFixed(2)}% of row, pool ${cell.pool})
  `;
  const stats = el('#sampleStats');
  if (stats){
    stats.innerHTML = '';
  }
  [
    `chars ${data.char_count || 0}`,
    `highlight ${data.highlighted_chars || 0}`,
    `pool ${cell.pool}`
  ].forEach(text => {
    const span = document.createElement('span');
    span.className = 'badge';
    span.textContent = text;
    if (stats){
      stats.appendChild(span);
    }
  });
  const paletteEl = el('#samplePalette');
  if (paletteEl){
    paletteEl.innerHTML = '';
    const palette = Array.isArray(data.palette) ? data.palette : [];
    if (palette.length){
      paletteEl.hidden = false;
      palette.forEach(entry => {
        const chip = document.createElement('div');
        chip.className = 'chip';
        const color = entry.color || '#999';
        const label = entry.name || entry.id || '?';
        chip.innerHTML = `<span class="dot" style="background:${color}"></span>${esc(label)}`;
        paletteEl.appendChild(chip);
      });
    }else{
      paletteEl.hidden = true;
    }
  }
  const predEl = el('#samplePred');
  if (predEl){
    predEl.innerHTML = data.html || '<em>No renderable text.</em>';
    attachTooltip(predEl);
    applyWrapIndicator(predEl);
  }

  const truthEl = el('#sampleTruth');
  if (truthEl){
    truthEl.innerHTML = data.truth_html || '<em>No ground truth sample.</em>';
    attachTooltip(truthEl, {mode: 'label'});
    applyWrapIndicator(truthEl);
  }

  const metaBox = el('#sampleMeta');
  if (!metaBox){
    return;
  }
  const metaItems = [];
  const summary = data.meta || {};
  if (summary.mode){
    metaItems.push(`mode: ${summary.mode}`);
  }
  if (summary.host_language){
    metaItems.push(`host: ${summary.host_language}`);
  }
  const lines = [];
  if (metaItems.length){
    lines.push(metaItems.join(' • '));
  }
  if (summary.sources){
    lines.push('sources:');
    lines.push('<ul>' + summary.sources.map(src => {
      const byteVal = typeof src.bytes === 'number' ? src.bytes : src.bytes && !Number.isNaN(Number(src.bytes)) ? Number(src.bytes) : '?';
      const label = esc(src.language || '?');
      return `<li>${label}: ${byteVal} bytes</li>`;
    }).join('') + '</ul>');
  }
  metaBox.innerHTML = lines.join('<br/>') || 'No metadata.';
}

function setSampleLoading(active){
  const loader = el('#sampleLoading');
  if (!loader) return;
  loader.hidden = !active;
  const pred = el('#samplePred');
  const truth = el('#sampleTruth');
  [pred, truth].forEach(elm => {
    if (elm){
      elm.classList.toggle('dim', active);
    }
  });
}

function attachTooltip(container, options = {}){
  const mode = options.mode || 'probs';
  tooltip = el('#tooltip');
  container.onmousemove = event => {
    const target = event.target;
    if (!target || !target.classList.contains('char')){
      tooltip.style.display = 'none';
      return;
    }
    if (mode === 'label'){
      const label = esc(target.dataset.true || target.dataset.label || target.textContent || 'Content');
      tooltip.innerHTML = `
        <div class="prob-bar label-only">
          <div class="label">Content type</div>
          <div class="value">${label}</div>
        </div>`;
    }else{
      let probs = {};
      try{
        probs = JSON.parse(target.dataset.probs || '{}') || {};
      }catch(err){
        probs = {};
      }
      const entries = Object.entries(probs).map(([label, prob]) => {
        return {label, prob: Number(prob)};
      }).filter(item => !Number.isNaN(item.prob)).sort((a,b)=>b.prob - a.prob);
      if (entries.length){
        tooltip.innerHTML = entries.map(e => `
          <div class="prob-bar">
            <div class="label">${esc(e.label)}</div>
            <div class="bar"><div class="fill" style="width:${(e.prob*100).toFixed(1)}%"></div></div>
            <div class="value">${(e.prob*100).toFixed(1)}%</div>
          </div>`).join('');
      }else{
        const fallbackLabel = esc(target.dataset.label || target.dataset.true || target.textContent || 'No data');
        tooltip.innerHTML = `
          <div class="prob-bar label-only">
            <div class="label">Prediction</div>
            <div class="value">${fallbackLabel}</div>
          </div>`;
      }
    }
    tooltip.style.display = 'block';
    const rect = target.getBoundingClientRect();
    tooltip.style.left = rect.left + 'px';
    tooltip.style.top = rect.bottom + 8 + 'px';
  };
  container.onmouseleave = () => {
    tooltip.style.display = 'none';
  };
}

function applyWrapIndicator(container){
  if (!container){
    return;
  }
  container.classList.remove('wrapped');
  container.querySelectorAll('.wrap-indicator').forEach(node => node.remove());
  const text = (container.textContent || '').trim();
  if (!text){
    return;
  }
  const longest = text.split(/\s+/).reduce((max, part) => Math.max(max, part.length), 0);
  if (longest < LONG_RUN_THRESHOLD){
    return;
  }
  const badge = document.createElement('div');
  badge.className = 'wrap-indicator';
  badge.textContent = '↪ wrap';
  badge.title = 'Long line wrapped automatically';
  container.appendChild(badge);
  container.classList.add('wrapped');
}

async function loadDataset(requestedId){
  if (requestedId){
    ACTIVE_DATASET = requestedId;
  }
  setMatrixLoading('Loading confusion matrix…');
  resetSampleView();
  const meta = el('#cellMeta');
  if (meta){
    meta.textContent = 'Loading dataset…';
  }
  setSampleLoading(true);
  try{
    const params = new URLSearchParams();
    if (ACTIVE_DATASET){
      params.set('dataset', ACTIVE_DATASET);
    }
    const query = params.toString();
    const res = await fetch(`/api/confusion${query ? `?${query}` : ''}`, {
      cache: 'no-store'
    });
    if (!res.ok){
      const err = await res.json().catch(()=>({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    CONFUSION = data;
    LABELS = data.labels || [];
    ACTIVE_DATASET = data.dataset || ACTIVE_DATASET;
    DATASET_OPTIONS = Array.isArray(data.datasets) ? data.datasets : [];
    updateDatasetSwitch();
    el('#deviceLabel').textContent = data.device || 'Device n/a';
    const summary = data.summary || `${data.total_windows || 0} windows`;
    el('#sampleCount').textContent = summary;
    buildMetrics(data.aggregates);
    renderMatrix(data);
  }catch(err){
    CONFUSION = null;
    const matrixEl = el('#matrix');
    if (matrixEl){
      matrixEl.classList.remove('loading');
      matrixEl.textContent = `Failed to load confusion data: ${err.message}`;
    }
    const metaBox = el('#cellMeta');
    if (metaBox){
      metaBox.textContent = 'Failed to load dataset.';
    }
  }finally{
    setSampleLoading(false);
  }
}

async function init(){
  setMatrixLoading();
  resetSampleView();
  await loadDataset();
}

window.addEventListener('DOMContentLoaded', () => {
  init();
});
let resizeTimer = null;
window.addEventListener('resize', () => {
  if (!CONFUSION){
    return;
  }
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => renderMatrix(CONFUSION), 120);
});
