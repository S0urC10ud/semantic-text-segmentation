let STATE = {
  summary: null,
  samples: null,
  currentPage: 1,
  pageSize: 20,
  selectedId: null,
  detail: null,
  detailView: "trigger",
  selectedPairIndex: 0,
};

function el(sel){ return document.querySelector(sel); }
function esc(str){
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
function pct(v){ return `${(100 * v).toFixed(1)}%`; }
function fmtInt(v){ return Number(v || 0).toLocaleString(); }
function hasInferenceDetail(detail){
  return detail && Object.prototype.hasOwnProperty.call(detail, "source_html");
}

function setDeleteStatus(message, kind = ""){
  const node = el("#deleteStatus");
  if (!node) return;
  node.className = "delete-status";
  if (kind === "ok") node.classList.add("ok");
  if (kind === "err") node.classList.add("err");
  node.textContent = String(message || "");
}

function safeColor(color){
  const c = String(color || "").trim();
  return /^#[0-9a-fA-F]{6}$/.test(c) ? c : "#999999";
}

function sanitizeRenderHtml(raw){
  const html = String(raw || "");
  if (!html) return "";
  const tpl = document.createElement("template");
  tpl.innerHTML = html;
  const allowedTags = new Set(["SPAN", "EM", "BR"]);
  const allowedAttrs = new Set(["class", "title", "data-label", "style"]);
  const styleRe = /^background\s*:\s*rgba\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*(?:0|0?\.\d+|1(?:\.0+)?)\s*\)\s*;?\s*$/i;

  const walk = node => {
    if (!node || !node.childNodes) return;
    [...node.childNodes].forEach(child => {
      if (child.nodeType === Node.ELEMENT_NODE){
        const tag = child.tagName || "";
        if (!allowedTags.has(tag)){
          const txt = document.createTextNode(child.textContent || "");
          child.replaceWith(txt);
          return;
        }
        [...child.attributes].forEach(attr => {
          const name = String(attr.name || "").toLowerCase();
          if (!allowedAttrs.has(name)){
            child.removeAttribute(attr.name);
            return;
          }
          if (name === "style"){
            const styleVal = String(attr.value || "").trim();
            if (!styleRe.test(styleVal)){
              child.removeAttribute("style");
            }
          }
        });
        walk(child);
      }
    });
  };
  walk(tpl.content);
  return tpl.innerHTML;
}

async function fetchJson(url){
  const res = await fetch(url, {cache: "no-store"});
  if (!res.ok){
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${res.status})`);
  }
  return await res.json();
}

async function deleteJson(url){
  const res = await fetch(url, {method: "DELETE", cache: "no-store"});
  if (!res.ok){
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${res.status})`);
  }
  return await res.json();
}

function paramsBase(){
  const status = el("#statusSelect").value;
  const roundId = el("#roundSelect").value;
  const queried = el("#queriedSelect").value;
  const search = el("#searchInput").value.trim();
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (roundId) params.set("round_id", roundId);
  if (queried) params.set("queried", queried);
  if (search) params.set("search", search);
  return params;
}

function renderCards(summary){
  const cards = el("#cards");
  cards.innerHTML = "";
  const statusCount = Object.values(summary.status_counts || {}).reduce((a,b)=>a+Number(b||0),0);
  const rows = [
    ["Inference Rows", fmtInt(summary.inference_filtered_total || summary.inference_total || 0)],
    ["Queried Samples", fmtInt(summary.inference_queried || 0)],
    ["Query Rate", pct(summary.inference_query_rate || 0)],
    ["Avg Candidates", Number(summary.inference_avg_candidates || 0).toFixed(2)],
    ["Refinement Rows", fmtInt(summary.total_rows || 0)],
    ["Mean Diff Ratio", pct(summary.diff_ratio_mean || 0)],
    ["Selected Status Rows", fmtInt(statusCount)],
  ];
  rows.forEach(([k,v]) => {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div>`;
    cards.appendChild(card);
  });
}

function renderBars(holderId, items){
  const holder = el(holderId);
  holder.innerHTML = "";
  const maxCount = items.length ? Math.max(...items.map(x => Number(x.count || 0))) : 1;
  items.slice(0, 24).forEach(item => {
    const count = Number(item.count || 0);
    const width = maxCount > 0 ? (100 * count / maxCount) : 0;
    const color = safeColor(item.color);
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `
      <div class="label">
        <span class="dot" style="background:${color}"></span>
        <span title="${esc(item.label)}">${esc(item.label)}</span>
      </div>
      <div class="track"><div class="fill" style="width:${width.toFixed(1)}%; background:${color}"></div></div>
      <div class="value">${fmtInt(count)}</div>
    `;
    holder.appendChild(row);
  });
}

function renderConfusion(conf){
  const holder = el("#confusion");
  holder.innerHTML = "";
  const labels = conf.labels || [];
  const matrix = conf.matrix || [];
  if (!labels.length || !matrix.length){
    holder.textContent = "No confusion data available.";
    return;
  }
  let maxVal = 0;
  matrix.forEach(row => row.forEach(v => { if (v > maxVal) maxVal = v; }));
  const grid = document.createElement("div");
  grid.className = "matrix";
  grid.style.gridTemplateColumns = `repeat(${labels.length + 1}, minmax(74px, auto))`;

  const corner = document.createElement("div");
  corner.className = "axis";
  corner.textContent = "Ref ↓ / Pred →";
  grid.appendChild(corner);

  labels.forEach(label => {
    const d = document.createElement("div");
    d.className = "axis";
    d.title = label;
    d.textContent = label;
    grid.appendChild(d);
  });

  labels.forEach((rowLabel, r) => {
    const axis = document.createElement("div");
    axis.className = "axis";
    axis.title = rowLabel;
    axis.textContent = rowLabel;
    grid.appendChild(axis);
    labels.forEach((colLabel, c) => {
      const val = Number((matrix[r] || [])[c] || 0);
      const frac = maxVal > 0 ? val / maxVal : 0;
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.title = `${rowLabel} → ${colLabel}: ${fmtInt(val)} chars`;
      cell.style.background = `rgba(22,83,211,${(0.08 + 0.72 * frac).toFixed(3)})`;
      cell.textContent = fmtInt(val);
      grid.appendChild(cell);
    });
  });
  holder.appendChild(grid);
}

function renderRounds(summary){
  const select = el("#roundSelect");
  const current = select.value;
  const rounds = summary.rounds || [];
  select.innerHTML = `<option value="">all</option>`;
  rounds.forEach(entry => {
    const opt = document.createElement("option");
    opt.value = entry.round_id;
    opt.textContent = `${entry.round_id} (${fmtInt(entry.count)})`;
    select.appendChild(opt);
  });
  if (current){
    select.value = current;
  }
}

async function loadSummary(){
  const params = paramsBase();
  params.delete("search"); // summary unaffected by free text
  const summary = await fetchJson(`/api/summary?${params.toString()}`);
  STATE.summary = summary;
  renderCards(summary);
  renderRounds(summary);
  renderBars("#refinedBars", summary.label_counts_refined || []);
  renderBars("#predBars", summary.label_counts_predicted || []);
  renderConfusion(summary.confusion || {labels:[], matrix:[]});
}

function renderSamples(payload){
  const tbody = el("#sampleRows");
  tbody.innerHTML = "";
  const items = payload.items || [];
  items.forEach(item => {
    const tr = document.createElement("tr");
    tr.className = "sample-row";
    tr.dataset.id = String(item.id);
    if (STATE.selectedId === item.id){
      tr.classList.add("active");
    }
    tr.innerHTML = `
      <td>${esc(String(item.id ?? ""))}</td>
      <td title="${esc(item.round_id)}">${esc(item.round_id || "")}</td>
      <td>${esc(item.source_lang || "")}</td>
      <td>${fmtInt(item.char_count || 0)}</td>
      <td>${item.queried_for_oracle ? "yes" : "no"}</td>
      <td>${fmtInt(item.candidate_count || 0)} / ${fmtInt(item.trigger_count || 0)}</td>
      <td title="${esc(item.preview || "")}">${esc(item.preview || "")}</td>
    `;
    tr.addEventListener("click", async () => {
      STATE.selectedId = item.id;
      [...tbody.querySelectorAll("tr")].forEach(node => node.classList.remove("active"));
      tr.classList.add("active");
      await loadDetail(item.id);
    });
    tbody.appendChild(tr);
  });

  el("#sampleMeta").textContent = `${fmtInt(payload.total)} rows • page ${payload.page}/${payload.total_pages}`;
  el("#pageInfo").textContent = `Page ${payload.page} / ${payload.total_pages}`;
  el("#prevPage").disabled = payload.page <= 1;
  el("#nextPage").disabled = payload.page >= payload.total_pages;
}

async function loadSamples(){
  const params = paramsBase();
  params.set("page", String(STATE.currentPage));
  params.set("page_size", String(STATE.pageSize));
  let payload = await fetchJson(`/api/samples?${params.toString()}`);
  if (payload && Number(payload.total_pages || 1) < Number(STATE.currentPage || 1)){
    STATE.currentPage = Math.max(1, Number(payload.total_pages || 1));
    params.set("page", String(STATE.currentPage));
    payload = await fetchJson(`/api/samples?${params.toString()}`);
  }
  STATE.samples = payload;
  renderSamples(payload);
}

function renderDetail(){
  const data = STATE.detail;
  const deleteBtn = el("#deleteSampleBtn");
  if (!data){
    el("#detailMeta").textContent = "Select a sample row.";
    el("#sourceRender").innerHTML = "";
    el("#triggerRows").innerHTML = "";
    el("#predictedRender").innerHTML = "";
    el("#refinedRender").innerHTML = "";
    el("#refinementRows").innerHTML = "";
    el("#metadataJson").textContent = "";
    el("#pairSelectWrap").classList.add("hidden");
    el("#triggerView").classList.remove("hidden");
    el("#predictionView").classList.add("hidden");
    if (deleteBtn) deleteBtn.disabled = true;
    return;
  }
  if (deleteBtn) deleteBtn.disabled = false;
  const hasInferenceView = hasInferenceDetail(data);
  const meta = el("#detailMeta");
  const metaParts = [`id ${data.id}`, data.source_lang, `chars ${fmtInt(data.char_count)}`];
  if (hasInferenceView){
    metaParts.push(`queried ${data.queried_for_oracle ? "yes" : "no"}`);
    metaParts.push(`candidates ${fmtInt(data.candidate_count || 0)}`);
    metaParts.push(`triggers ${fmtInt(data.trigger_count || 0)}`);
  }else{
    metaParts.push(`diff ${fmtInt(data.diff_chars || 0)} (${pct(data.diff_ratio || 0)})`);
  }
  meta.textContent = metaParts.join(" • ");

  const triggerRows = el("#triggerRows");
  const triggerItems = hasInferenceView ? (data.trigger_ranges || []) : [];
  if (!triggerItems.length){
    triggerRows.innerHTML = hasInferenceView
      ? "<em>No trigger spans for this sample.</em>"
      : "<em>Trigger spans are only available for inference-sample rows.</em>";
  }else{
    triggerRows.innerHTML = triggerItems.map((row, idx) => `
      <div class="trigger-row">
        <span>#${idx + 1}</span>
        <span>[${row.start}, ${row.end})</span>
        <span>b=${row.boundary}</span>
        <span>s=${Number(row.score || 0).toFixed(3)}</span>
        <span>e=${Number(row.entropy_mean || 0).toFixed(3)}</span>
        <span>f=${Number(row.flip_rate || 0).toFixed(3)}</span>
        <span>${esc(row.left_label || "?")}→${esc(row.right_label || "?")}</span>
      </div>
    `).join("");
  }
  el("#sourceRender").innerHTML = sanitizeRenderHtml(
    data.source_html || data.refined_html || data.predicted_html || "<em>No source render.</em>"
  );

  const refinementRows = el("#refinementRows");
  const linked = hasInferenceView ? (data.linked_refinements || []) : [];
  if (!linked.length){
    refinementRows.innerHTML = hasInferenceView
      ? "<em>No linked refinement snippets stored for this sample.</em>"
      : "<em>Linked snippets are shown only for inference-sample rows.</em>";
  }else{
    refinementRows.innerHTML = linked.map(row => `
      <div class="linked-row">
        <div class="linked-head">
          <span>ref ${row.id}</span>
          <span>boundary ${row.boundary_index}</span>
          <span>score ${Number(row.acquisition_score || 0).toFixed(3)}</span>
          <span>diff ${fmtInt(row.diff_chars || 0)} (${pct(row.diff_ratio || 0)})</span>
        </div>
        <div class="linked-preview" title="${esc(row.preview || "")}">${esc(row.preview || "")}</div>
      </div>
    `).join("");
  }

  const pairWrap = el("#pairSelectWrap");
  const pairSelect = el("#pairSelect");
  const pairs = hasInferenceView ? (data.refinement_pairs || []) : [];
  const showPrediction = STATE.detailView === "prediction";
  el("#detailViewSelect").value = showPrediction ? "prediction" : "trigger";
  if (pairs.length){
    pairWrap.classList.toggle("hidden", !showPrediction);
    pairSelect.innerHTML = pairs.map((row, idx) => (
      `<option value="${idx}">ref ${esc(String(row.id ?? ""))} • b=${fmtInt(row.boundary_index || 0)} • diff=${pct(row.diff_ratio || 0)}</option>`
    )).join("");
    if (STATE.selectedPairIndex >= pairs.length){
      STATE.selectedPairIndex = 0;
    }
    pairSelect.value = String(STATE.selectedPairIndex);
  }else{
    pairWrap.classList.add("hidden");
    pairSelect.innerHTML = "";
  }
  el("#triggerView").classList.toggle("hidden", showPrediction);
  el("#predictionView").classList.toggle("hidden", !showPrediction);

  if (showPrediction){
    if (hasInferenceView){
      if (pairs.length){
        const pair = pairs[Math.max(0, Math.min(STATE.selectedPairIndex, pairs.length - 1))];
        el("#predictedRender").innerHTML = sanitizeRenderHtml(
          pair.predicted_html || "<em>No model prediction for this refinement slice.</em>"
        );
        el("#refinedRender").innerHTML = sanitizeRenderHtml(
          pair.refined_html || "<em>No Gemini refinement payload for this slice.</em>"
        );
      }else{
        if (data.prediction_has_model_segments){
          el("#predictedRender").innerHTML = sanitizeRenderHtml(
            data.predicted_html || "<em>No model prediction stored for this sample.</em>"
          );
        }else{
          el("#predictedRender").innerHTML = "<em>No per-character prediction was stored for this sample row.</em>";
        }
        if (data.queried_for_oracle){
          el("#refinedRender").innerHTML = "<em>Sample was queried but no refinement slices were stored.</em>";
        }else{
          if ((data.candidate_count || 0) > 0){
            el("#refinedRender").innerHTML = "<em>Sample had oracle candidates, but no refinement slices were stored (likely an aborted/failed oracle round).</em>";
          }else{
            el("#refinedRender").innerHTML = "<em>Sample was not queried, so no Gemini correction exists.</em>";
          }
        }
      }
    }else{
      el("#predictedRender").innerHTML = sanitizeRenderHtml(
        data.predicted_html || "<em>No model prediction available.</em>"
      );
      el("#refinedRender").innerHTML = sanitizeRenderHtml(
        data.refined_html || "<em>No refinement available.</em>"
      );
    }
  }

  el("#metadataJson").textContent = JSON.stringify(data.metadata || {}, null, 2);
}

async function loadDetail(id){
  const data = await fetchJson(`/api/sample/${id}`);
  STATE.detail = data;
  STATE.selectedPairIndex = 0;
  renderDetail();
}

async function deleteSelectedSample(){
  if (!STATE.detail || !STATE.selectedId){
    setDeleteStatus("Select a sample row first.", "err");
    return;
  }
  const data = STATE.detail;
  const hasInference = hasInferenceDetail(data);
  const prompt = hasInference
    ? `Delete sample id=${data.id} and all linked refinement slices?`
    : `Delete refinement row id=${data.id}?`;
  if (!window.confirm(prompt)){
    return;
  }
  const deleteBtn = el("#deleteSampleBtn");
  if (deleteBtn) deleteBtn.disabled = true;
  try{
    const result = await deleteJson(`/api/sample/${STATE.selectedId}`);
    const deletedRef = fmtInt(result.deleted_refinements || 0);
    const deletedInf = fmtInt(result.deleted_inference_samples || 0);
    setDeleteStatus(
      `Deleted row ${result.deleted_sample_id}. inference=${deletedInf}, refinements=${deletedRef}.`,
      "ok"
    );
    STATE.detail = null;
    STATE.selectedId = null;
    STATE.selectedPairIndex = 0;
    renderDetail();
    await refreshAll({resetPage:false});
  }catch(err){
    setDeleteStatus(`Delete failed: ${err.message}`, "err");
    if (deleteBtn) deleteBtn.disabled = false;
  }
}

async function refreshAll({resetPage = true} = {}){
  if (resetPage){
    STATE.currentPage = 1;
  }
  await loadSummary();
  await loadSamples();
}

function bind(){
  el("#refreshBtn").addEventListener("click", () => refreshAll({resetPage:false}));
  el("#statusSelect").addEventListener("change", () => refreshAll({resetPage:true}));
  el("#roundSelect").addEventListener("change", () => refreshAll({resetPage:true}));
  el("#queriedSelect").addEventListener("change", () => refreshAll({resetPage:true}));
  el("#searchInput").addEventListener("keydown", event => {
    if (event.key === "Enter"){
      refreshAll({resetPage:true});
    }
  });
  el("#detailViewSelect").addEventListener("change", () => {
    STATE.detailView = el("#detailViewSelect").value === "prediction" ? "prediction" : "trigger";
    renderDetail();
  });
  el("#pairSelect").addEventListener("change", () => {
    STATE.selectedPairIndex = Math.max(0, Number(el("#pairSelect").value || 0));
    renderDetail();
  });
  el("#deleteSampleBtn").addEventListener("click", async () => {
    await deleteSelectedSample();
  });
  el("#prevPage").addEventListener("click", async () => {
    if (STATE.currentPage <= 1) return;
    STATE.currentPage -= 1;
    await loadSamples();
  });
  el("#nextPage").addEventListener("click", async () => {
    if (!STATE.samples || STATE.currentPage >= STATE.samples.total_pages) return;
    STATE.currentPage += 1;
    await loadSamples();
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  bind();
  renderDetail();
  setDeleteStatus("");
  try{
    await refreshAll({resetPage:true});
  }catch(err){
    el("#sampleMeta").textContent = `Failed to load data: ${err.message}`;
  }
});
