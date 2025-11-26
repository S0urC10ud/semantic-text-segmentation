let CONFUSION = null;
let LABELS = [];
let tooltip = null;
const CELL_WIDTH = 34;
const TRUE_AXIS_WIDTH = 78;

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

function renderMatrix(data){
  const matrixEl = el('#matrix');
  matrixEl.classList.remove('loading');
  if (matrixEl.previousElementSibling && matrixEl.previousElementSibling.classList.contains('matrix-note')){
    matrixEl.previousElementSibling.remove();
  }
  const n = data.labels.length;
  matrixEl.style.gridTemplateColumns = `${TRUE_AXIS_WIDTH}px repeat(${n}, ${CELL_WIDTH}px)`;
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
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.row = row;
      cell.dataset.col = col;
      cell.innerHTML = `<div class="count">${count}×</div>
        <div class="pct">${(pct * 100).toFixed(1)}%</div>`;
      const logComponent = Math.min(1, Math.log10(pct * 9 + 1));
      const intensity = count > 0 ? logComponent : 0;
      cell.style.background = `rgba(11,94,215,${intensity * 0.7})`;
      cell.title = `${data.labels[row].name} → ${data.labels[col].name}: ${(pct * 100).toFixed(2)}% (${count})`;
      cell.addEventListener('click', () => loadExample(row, col));
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
  setSampleLoading(true);
  try{
    const res = await fetch(`/api/example?true_id=${trueId}&pred_id=${predId}&ts=${Date.now()}`, {
      cache: 'no-store'
    });
    if (!res.ok){
      const err = await res.json().catch(()=>({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    updateSample(data);
  }catch(err){
    meta.textContent = `Error: ${err.message}`;
  }finally{
    setSampleLoading(false);
  }
}

function updateSample(data){
  const meta = el('#cellMeta');
  const cell = data.cell;
  meta.innerHTML = `
    <strong>${esc(cell.true.name)}</strong>
    <span aria-hidden="true">→</span>
    <strong>${esc(cell.pred.name)}</strong>
    &nbsp;• ${cell.count} hits (${(cell.row_pct * 100).toFixed(2)}% of row, pool ${cell.pool})
  `;
  const stats = el('#sampleStats');
  stats.innerHTML = '';
  [
    `chars ${data.char_count || 0}`,
    `highlight ${data.highlighted_chars || 0}`,
    `pool ${cell.pool}`
  ].forEach(text => {
    const span = document.createElement('span');
    span.className = 'badge';
    span.textContent = text;
    stats.appendChild(span);
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
  predEl.innerHTML = data.html || '<em>No renderable text.</em>';
  attachTooltip(predEl);

  const truthEl = el('#sampleTruth');
  truthEl.innerHTML = data.truth_html || '<em>No ground truth sample.</em>';

  const metaBox = el('#sampleMeta');
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

function attachTooltip(container){
  tooltip = el('#tooltip');
  container.onmousemove = event => {
    const target = event.target;
    if (!target || !target.classList.contains('char')){
      tooltip.style.display = 'none';
      return;
    }
    const probs = JSON.parse(target.dataset.probs || '{}');
    const entries = Object.entries(probs).map(([label, prob]) => {
      return {label, prob};
    }).sort((a,b)=>b.prob - a.prob);
    tooltip.innerHTML = entries.map(e => `
      <div class="prob-bar">
        <div class="label">${esc(e.label)}</div>
        <div class="bar"><div class="fill" style="width:${(e.prob*100).toFixed(1)}%"></div></div>
        <div class="value">${(e.prob*100).toFixed(1)}%</div>
      </div>`).join('') || '<div class="prob-bar"><div class="label">No data</div></div>';
    tooltip.style.display = 'block';
    const rect = target.getBoundingClientRect();
    tooltip.style.left = rect.left + 'px';
    tooltip.style.top = rect.bottom + 8 + 'px';
  };
  container.onmouseleave = () => {
    tooltip.style.display = 'none';
  };
}

async function init(){
  try{
    const res = await fetch('/api/confusion');
    if (!res.ok){
      const err = await res.json().catch(()=>({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    CONFUSION = data;
    LABELS = data.labels;
    el('#deviceLabel').textContent = data.device || 'Device n/a';
    el('#sampleCount').textContent = `${data.total_windows || 0} windows`;
    buildMetrics(data.aggregates);
    renderMatrix(data);
    setSampleLoading(false);
  }catch(err){
    const matrixEl = el('#matrix');
    matrixEl.classList.remove('loading');
    matrixEl.textContent = `Failed to load confusion data: ${err.message}`;
  }
}

window.addEventListener('DOMContentLoaded', init);
