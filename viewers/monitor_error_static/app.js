let STATE = {
  config: null,
  samples: null,
  currentPage: 1,
  pageSize: 20,
  selectedId: null,
  detail: null,
  hoverIndex: null,
};

let TOOLTIP = null;
let SCROLL_LOCK = false;

function el(sel){ return document.querySelector(sel); }
function esc(str){
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
function pct(v){ return `${(100 * Number(v || 0)).toFixed(1)}%`; }
function fmtInt(v){ return Number(v || 0).toLocaleString(); }
function fmtFloat(v, digits = 3){ return Number(v || 0).toFixed(digits); }
function safeColor(color){
  const c = String(color || "").trim();
  return /^#[0-9a-fA-F]{6}$/.test(c) ? c : "#999999";
}
function hexToRgba(hexColor, alpha){
  const h = safeColor(hexColor).slice(1);
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
function focusLabel(){
  return String(el("#focusSelect")?.value || "").trim();
}
function currentFilters(){
  return {
    focus_label: focusLabel(),
    sort: String(el("#sortSelect")?.value || "f1_asc"),
    search: String(el("#searchInput")?.value || "").trim(),
    mistakes_only: !!el("#mistakesOnlyCheckbox")?.checked,
  };
}
function selectedLabelStat(){
  const config = STATE.config;
  if (!config) return null;
  const label = focusLabel();
  if (!label){
    return null;
  }
  return (config.label_stats || []).find(item => item.label === label) || null;
}
function labelColor(label){
  const palette = (STATE.detail && STATE.detail.label_colors) || (STATE.config && STATE.config.palette) || {};
  return safeColor(palette[label] || "#999999");
}
function displayCharName(ch){
  if (ch === "\n") return "\\n";
  if (ch === "\t") return "\\t";
  if (ch === " ") return "space";
  return ch;
}

async function fetchJson(url){
  const res = await fetch(url, {cache: "no-store"});
  if (!res.ok){
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${res.status})`);
  }
  return await res.json();
}

function applySampleActiveState(){
  document.querySelectorAll("#sampleRows tr").forEach(row => {
    row.classList.toggle("active", Number(row.dataset.id || 0) === Number(STATE.selectedId || 0));
  });
}

function renderOverviewCards(){
  const cards = el("#cards");
  cards.innerHTML = "";
  const config = STATE.config;
  const samples = STATE.samples;
  if (!config){
    cards.innerHTML = `<div class="card"><div class="k">Status</div><div class="v">Loading…</div></div>`;
    return;
  }

  const allStats = config.label_stats || [];
  const weightedAvgF1 = (() => {
    const total = allStats.reduce((sum, row) => sum + Number(row.count || 0), 0);
    if (!total) return 0;
    const weighted = allStats.reduce((sum, row) => sum + Number(row.avg_f1 || 0) * Number(row.count || 0), 0);
    return weighted / total;
  })();
  const selected = selectedLabelStat();
  const filtered = (samples && samples.filtered_stats) || {};
  const chosenLabel = focusLabel() || "all content types";
  const rows = [
    ["Indexed windows", fmtInt(config.total_samples || 0), `per label limit ${fmtInt(config.per_label_limit || 0)}`],
    ["Checkpoint chunk", fmtInt(config.checkpoint_chunk || 0), `other threshold ${fmtFloat(config.other_threshold || 0, 2)}`],
    ["Global avg F1", pct(weightedAvgF1), `${fmtInt(allStats.length)} focus labels indexed`],
    ["Current filter", chosenLabel, selected ? `avg F1 ${pct(selected.avg_f1 || 0)}` : "click a label below to focus"],
    ["Filtered avg F1", pct(filtered.avg_focus_f1 || 0), `diff ${pct(filtered.avg_diff_ratio || 0)}`],
    ["Monitor root", String(config.monitor_root || "").split("/").slice(-2).join("/"), `${fmtInt(samples?.total || config.total_samples || 0)} rows in view`],
  ];

  rows.forEach(([k, v, s]) => {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div><div class="s">${esc(String(s))}</div>`;
    cards.appendChild(card);
  });
}

function renderSelectionPanel(){
  const node = el("#selectionMeta");
  const grid = el("#selectionStats");
  const samples = STATE.samples;
  const config = STATE.config;
  if (!config || !samples){
    node.textContent = "Loading selection details…";
    grid.innerHTML = "";
    return;
  }

  const filters = currentFilters();
  const label = focusLabel() || "all content types";
  node.textContent = `Showing ${fmtInt(samples.total || 0)} windows for ${label}${filters.mistakes_only ? " • mistakes only" : ""}.`;

  const stats = [
    ["Avg F1", pct(samples.filtered_stats?.avg_focus_f1 || 0)],
    ["Avg Diff", pct(samples.filtered_stats?.avg_diff_ratio || 0)],
    ["Avg Accuracy", pct(samples.filtered_stats?.avg_overall_accuracy || 0)],
    ["Sort", (config.sort_options || []).find(item => item.value === filters.sort)?.label || filters.sort],
  ];
  grid.innerHTML = "";
  stats.forEach(([k, v]) => {
    const div = document.createElement("div");
    div.className = "mini-stat";
    div.innerHTML = `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div>`;
    grid.appendChild(div);
  });
}

function renderLabelRows(){
  const holder = el("#labelRows");
  holder.innerHTML = "";
  const config = STATE.config;
  if (!config){
    holder.innerHTML = `<div class="empty-note">Loading labels…</div>`;
    return;
  }

  const active = focusLabel();
  (config.label_stats || []).forEach(item => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "label-row";
    if (item.label === active){
      row.classList.add("active");
    }
    row.innerHTML = `
      <div class="label-main">
        <span class="label-chip" style="background:${safeColor(item.color)}"></span>
        <span class="label-name">${esc(item.label)}</span>
      </div>
      <div class="label-value" title="Average F1">F1 ${pct(item.avg_f1 || 0)}</div>
      <div class="label-value" title="Average diff ratio">Diff ${pct(item.avg_diff_ratio || 0)}</div>
      <div class="label-value" title="Indexed windows">${fmtInt(item.count || 0)}</div>
    `;
    row.addEventListener("click", async () => {
      const focus = el("#focusSelect");
      focus.value = item.label;
      STATE.currentPage = 1;
      renderLabelRows();
      await loadSamples({autoSelect: true});
    });
    holder.appendChild(row);
  });
}

function populateControls(){
  const config = STATE.config;
  if (!config){
    return;
  }

  const focus = el("#focusSelect");
  const sort = el("#sortSelect");
  const currentFocus = focus.value;
  const currentSort = sort.value || "f1_asc";

  focus.innerHTML = `<option value="">all</option>`;
  (config.focus_labels || []).forEach(label => {
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label;
    focus.appendChild(opt);
  });
  if (currentFocus && (config.focus_labels || []).includes(currentFocus)){
    focus.value = currentFocus;
  }

  sort.innerHTML = "";
  (config.sort_options || []).forEach(item => {
    const opt = document.createElement("option");
    opt.value = item.value;
    opt.textContent = item.label;
    sort.appendChild(opt);
  });
  sort.value = (config.sort_options || []).some(item => item.value === currentSort) ? currentSort : "f1_asc";
}

function renderSamples(payload){
  const tbody = el("#sampleRows");
  tbody.innerHTML = "";
  const items = payload.items || [];
  if (!items.length){
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="10" class="empty-note">No indexed windows matched the current filters.</td>`;
    tbody.appendChild(tr);
  }

  items.forEach(item => {
    const tr = document.createElement("tr");
    tr.className = "sample-row";
    tr.dataset.id = String(item.id);
    if (Number(item.id) === Number(STATE.selectedId || 0)){
      tr.classList.add("active");
    }
    tr.innerHTML = `
      <td class="mono">${fmtInt(item.id)}</td>
      <td>${esc(item.focus_label || "")}</td>
      <td>${esc(item.file_type || "")}</td>
      <td class="mono">${fmtInt(item.focus_support || 0)}</td>
      <td class="mono">${pct(item.focus_f1 || 0)}</td>
      <td class="mono">${pct(item.focus_precision || 0)}</td>
      <td class="mono">${pct(item.focus_recall || 0)}</td>
      <td class="mono">${pct(item.diff_ratio || 0)}</td>
      <td class="mono">${pct(item.overall_accuracy || 0)}</td>
      <td title="${esc(item.preview || "")}">${esc(item.preview || "")}</td>
    `;
    tr.addEventListener("click", async () => {
      STATE.selectedId = item.id;
      applySampleActiveState();
      await loadDetail(item.id);
    });
    tbody.appendChild(tr);
  });

  el("#sampleMeta").textContent = `${fmtInt(payload.total || 0)} rows • page ${fmtInt(payload.page || 1)} / ${fmtInt(payload.total_pages || 1)}`;
  el("#pageInfo").textContent = `${fmtInt(payload.page || 1)} / ${fmtInt(payload.total_pages || 1)}`;
  el("#prevPage").disabled = Number(payload.page || 1) <= 1;
  el("#nextPage").disabled = Number(payload.page || 1) >= Number(payload.total_pages || 1);
}

function clearDetail(message){
  STATE.detail = null;
  el("#detailMeta").textContent = message;
  el("#detailCards").innerHTML = "";
  el("#legend").innerHTML = "";
  el("#confusionRows").innerHTML = `<div class="confusion-row empty">No sample selected.</div>`;
  el("#truthRuns").innerHTML = `<div class="run-row empty">No sample selected.</div>`;
  el("#predRuns").innerHTML = `<div class="run-row empty">No sample selected.</div>`;
  el("#truthRender").innerHTML = "";
  el("#predRender").innerHTML = "";
  hideTooltip();
}

function renderDetailCards(detail){
  const cards = el("#detailCards");
  cards.innerHTML = "";
  const metrics = detail.metrics || {};
  const rows = [
    ["Focus label", detail.focus_label || "", `${esc(detail.file_type || "")} • file ${fmtInt(detail.file_idx || 0)}`],
    ["Focus F1", pct(metrics.focus_f1 || 0), `precision ${pct(metrics.focus_precision || 0)} • recall ${pct(metrics.focus_recall || 0)}`],
    ["Diff ratio", pct(metrics.diff_ratio || 0), `accuracy ${pct(metrics.overall_accuracy || 0)}`],
    ["Confusion counts", `tp ${fmtInt(metrics.tp || 0)} • fp ${fmtInt(metrics.fp || 0)}`, `fn ${fmtInt(metrics.fn || 0)} • valid ${fmtInt(metrics.valid_bytes || 0)}`],
    ["Window bytes", `${fmtInt(detail.window_start || 0)}:${fmtInt(detail.window_end || 0)}`, `support ${fmtInt(detail.focus_support || 0)} of ${fmtInt(detail.focus_total_bytes || 0)}`],
  ];
  rows.forEach(([k, v, s]) => {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div><div class="s">${String(s)}</div>`;
    cards.appendChild(card);
  });
}

function renderLegend(detail){
  const holder = el("#legend");
  holder.innerHTML = "";
  const order = detail.label_order || [];
  order.forEach(label => {
    const div = document.createElement("div");
    div.className = "legend-chip-row";
    div.innerHTML = `<span class="legend-swatch" style="background:${safeColor(detail.label_colors?.[label])}"></span><span>${esc(label)}</span>`;
    holder.appendChild(div);
  });

  const diffChip = document.createElement("div");
  diffChip.className = "legend-chip-row diff";
  diffChip.innerHTML = `<span class="legend-swatch"></span><span>mismatch highlight</span>`;
  holder.appendChild(diffChip);

  const matchChip = document.createElement("div");
  matchChip.className = "legend-chip-row match";
  matchChip.innerHTML = `<span class="legend-swatch"></span><span>matching label</span>`;
  holder.appendChild(matchChip);
}

function renderConfusions(detail){
  const holder = el("#confusionRows");
  holder.innerHTML = "";
  const rows = detail.top_confusions || [];
  if (!rows.length){
    holder.innerHTML = `<div class="confusion-row empty">No disagreements in this window.</div>`;
    return;
  }
  rows.forEach(row => {
    const div = document.createElement("div");
    div.className = "confusion-row";
    div.innerHTML = `
      <div><b>${esc(row.truth || "")}</b></div>
      <div>${esc(row.pred || "")}</div>
      <div class="mono">${fmtInt(row.count || 0)}</div>
    `;
    holder.appendChild(div);
  });
}

function renderRuns(holderId, runs){
  const holder = el(holderId);
  holder.innerHTML = "";
  if (!(runs || []).length){
    holder.innerHTML = `<div class="run-row empty">No runs to display.</div>`;
    return;
  }
  runs.forEach(run => {
    const div = document.createElement("div");
    div.className = "run-row";
    div.innerHTML = `
      <div>${fmtInt(run.start || 0)}</div>
      <div>${fmtInt(run.end || 0)}</div>
      <div><span class="label-chip" style="background:${labelColor(run.label)}"></span> ${esc(run.label || "")}</div>
    `;
    holder.appendChild(div);
  });
}

function buildCharSpan(idx, entry, paneLabel){
  const label = paneLabel === "truth" ? entry.truth : entry.pred;
  const span = document.createElement("span");
  span.className = `char-cell ${entry.valid ? "valid" : "invalid"} ${entry.match ? "match" : "diff"}`;
  span.dataset.index = String(idx);
  const alpha = entry.match ? 0.18 : 0.30;
  span.style.background = hexToRgba(labelColor(label), alpha);
  span.textContent = entry.char;
  return span;
}

function renderDetailText(detail){
  const truth = el("#truthRender");
  const pred = el("#predRender");
  truth.innerHTML = "";
  pred.innerHTML = "";

  const truthFrag = document.createDocumentFragment();
  const predFrag = document.createDocumentFragment();
  (detail.char_data || []).forEach((entry, idx) => {
    truthFrag.appendChild(buildCharSpan(idx, entry, "truth"));
    predFrag.appendChild(buildCharSpan(idx, entry, "pred"));
  });
  truth.appendChild(truthFrag);
  pred.appendChild(predFrag);
  bindScrollSync(truth, pred);
}

function renderDetail(detail){
  STATE.detail = detail;
  el("#detailMeta").textContent = `Sample ${fmtInt(detail.id || 0)} • focus ${detail.focus_label || ""} • file ${fmtInt(detail.file_idx || 0)} (${detail.file_type || ""})`;
  renderDetailCards(detail);
  renderLegend(detail);
  renderConfusions(detail);
  renderRuns("#truthRuns", detail.truth_runs || []);
  renderRuns("#predRuns", detail.pred_runs || []);
  renderDetailText(detail);
}

function bindScrollSync(leftNode, rightNode){
  if (!leftNode || !rightNode){
    return;
  }
  const sync = (source, target) => {
    if (SCROLL_LOCK){
      return;
    }
    SCROLL_LOCK = true;
    target.scrollTop = source.scrollTop;
    target.scrollLeft = source.scrollLeft;
    window.requestAnimationFrame(() => {
      SCROLL_LOCK = false;
    });
  };
  leftNode.onscroll = () => sync(leftNode, rightNode);
  rightNode.onscroll = () => sync(rightNode, leftNode);
}

function hoverTargets(index){
  return document.querySelectorAll(`.char-cell[data-index="${index}"]`);
}

function clearHoverSync(){
  if (STATE.hoverIndex === null){
    return;
  }
  hoverTargets(STATE.hoverIndex).forEach(node => node.classList.remove("hover-sync"));
  STATE.hoverIndex = null;
}

function showTooltipForIndex(index, event){
  if (!STATE.detail){
    return;
  }
  const entry = (STATE.detail.char_data || [])[index];
  if (!entry){
    return;
  }
  if (!TOOLTIP){
    TOOLTIP = el("#tooltip");
  }
  clearHoverSync();
  STATE.hoverIndex = index;
  hoverTargets(index).forEach(node => node.classList.add("hover-sync"));

  const rows = (entry.probs || []).map(item => `
    <div class="prob-bar">
      <div class="prob-label">${esc(item.label || "")}</div>
      <div class="prob-track"><div class="prob-fill" style="width:${(100 * Number(item.prob || 0)).toFixed(1)}%; background:${labelColor(item.label)}"></div></div>
      <div class="prob-value">${pct(item.prob || 0)}</div>
    </div>
  `).join("");
  TOOLTIP.innerHTML = `
    <div class="tooltip-head">Byte ${fmtInt(index)} • ${esc(displayCharName(entry.char))}</div>
    <div class="tooltip-sub">truth ${esc(entry.truth || "")} • pred ${esc(entry.pred || "")}${entry.match ? "" : " • mismatch"}</div>
    ${rows || '<div class="prob-bar"><div class="prob-label">No probability data</div></div>'}
  `;
  TOOLTIP.classList.remove("hidden");

  const margin = 12;
  const rect = TOOLTIP.getBoundingClientRect();
  let left = Number(event.clientX || 0) + margin;
  let top = Number(event.clientY || 0) + margin;
  if (left + rect.width > window.innerWidth - 8){
    left = window.innerWidth - rect.width - 8;
  }
  if (top + rect.height > window.innerHeight - 8){
    top = Number(event.clientY || 0) - rect.height - margin;
  }
  TOOLTIP.style.left = `${Math.max(8, left)}px`;
  TOOLTIP.style.top = `${Math.max(8, top)}px`;
}

function hideTooltip(){
  clearHoverSync();
  if (!TOOLTIP){
    TOOLTIP = el("#tooltip");
  }
  TOOLTIP.classList.add("hidden");
  TOOLTIP.innerHTML = "";
}

async function loadConfig(){
  STATE.config = await fetchJson("/api/config");
  populateControls();
  renderOverviewCards();
  renderLabelRows();
}

async function loadSamples({autoSelect = false} = {}){
  const params = new URLSearchParams();
  const filters = currentFilters();
  params.set("page", String(STATE.currentPage || 1));
  params.set("page_size", String(STATE.pageSize || 20));
  if (filters.focus_label){
    params.set("focus_label", filters.focus_label);
  }
  if (filters.sort){
    params.set("sort", filters.sort);
  }
  if (filters.search){
    params.set("search", filters.search);
  }
  if (filters.mistakes_only){
    params.set("mistakes_only", "true");
  }

  const payload = await fetchJson(`/api/samples?${params.toString()}`);
  STATE.samples = payload;
  renderOverviewCards();
  renderSelectionPanel();
  renderLabelRows();
  renderSamples(payload);

  const ids = new Set((payload.items || []).map(item => Number(item.id)));
  if (!ids.size){
    STATE.selectedId = null;
    clearDetail("No sample matched the current filters.");
    return;
  }
  if (!ids.has(Number(STATE.selectedId || 0)) || autoSelect){
    STATE.selectedId = Number((payload.items || [])[0]?.id || 0);
    applySampleActiveState();
    await loadDetail(STATE.selectedId);
    return;
  }
  applySampleActiveState();
}

async function loadDetail(id){
  const detail = await fetchJson(`/api/sample/${id}`);
  renderDetail(detail);
}

function attachEvents(){
  el("#focusSelect").addEventListener("change", async () => {
    STATE.currentPage = 1;
    await loadSamples({autoSelect: true});
  });
  el("#sortSelect").addEventListener("change", async () => {
    STATE.currentPage = 1;
    await loadSamples({autoSelect: true});
  });
  el("#pageSizeSelect").addEventListener("change", async event => {
    STATE.pageSize = Number(event.target.value || 20);
    STATE.currentPage = 1;
    await loadSamples({autoSelect: true});
  });
  el("#mistakesOnlyCheckbox").addEventListener("change", async () => {
    STATE.currentPage = 1;
    await loadSamples({autoSelect: true});
  });
  el("#refreshBtn").addEventListener("click", async () => {
    await loadSamples({autoSelect: false});
  });
  el("#searchInput").addEventListener("keydown", async event => {
    if (event.key !== "Enter"){
      return;
    }
    STATE.currentPage = 1;
    await loadSamples({autoSelect: true});
  });
  el("#prevPage").addEventListener("click", async () => {
    if ((STATE.samples?.page || 1) <= 1){
      return;
    }
    STATE.currentPage = Math.max(1, Number(STATE.samples?.page || 1) - 1);
    await loadSamples({autoSelect: false});
  });
  el("#nextPage").addEventListener("click", async () => {
    const totalPages = Number(STATE.samples?.total_pages || 1);
    if ((STATE.samples?.page || 1) >= totalPages){
      return;
    }
    STATE.currentPage = Math.min(totalPages, Number(STATE.samples?.page || 1) + 1);
    await loadSamples({autoSelect: false});
  });

  ["#truthRender", "#predRender"].forEach(sel => {
    const node = el(sel);
    node.addEventListener("mousemove", event => {
      const cell = event.target.closest(".char-cell");
      if (!cell){
        hideTooltip();
        return;
      }
      const index = Number(cell.dataset.index || -1);
      if (!Number.isFinite(index) || index < 0){
        hideTooltip();
        return;
      }
      showTooltipForIndex(index, event);
    });
    node.addEventListener("mouseleave", () => {
      hideTooltip();
    });
  });
}

async function boot(){
  TOOLTIP = el("#tooltip");
  STATE.pageSize = Number(el("#pageSizeSelect").value || 20);
  attachEvents();
  await loadConfig();
  await loadSamples({autoSelect: true});
}

boot().catch(err => {
  console.error(err);
  clearDetail(`Failed to load viewer: ${err.message}`);
  el("#sampleMeta").textContent = err.message;
  el("#selectionMeta").textContent = err.message;
});
