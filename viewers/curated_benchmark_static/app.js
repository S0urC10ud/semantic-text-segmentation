"use strict";

const STATE = {
  summary: null,
  samples: null,
  currentPage: 1,
  pageSize: 25,
  selectedIndex: null,
  selectedDetail: null,
  liveRuns: [],
  selectedLiveRunId: "",
  selectedLiveBatchSize: "",
  runOnlySelectedRun: false,
};

function el(sel){
  return document.querySelector(sel);
}

function esc(raw){
  return String(raw ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtInt(v){
  return Number(v || 0).toLocaleString();
}

function safeColor(color){
  const c = String(color || "").trim();
  return /^#[0-9a-fA-F]{6}$/.test(c) ? c : "#7d7d7d";
}

function isFailedSegmentationSource(source){
  const src = String(source || "").trim();
  if (!src) return false;
  if (src.startsWith("fallback_")) return true;
  return src === "gemini_live_unavailable" || src === "dataset_prior_live_missing";
}

function sanitizeRenderHtml(raw){
  const html = String(raw || "");
  if (!html) return "";
  const tpl = document.createElement("template");
  tpl.innerHTML = html;
  const allowedTags = new Set(["SPAN", "EM", "BR"]);
  const allowedAttrs = new Set(["class", "title", "style"]);
  const styleRe = /^background\s*:\s*rgba\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*(?:0|0?\.\d+|1(?:\.0+)?)\s*\)\s*;?\s*$/i;

  const walk = node => {
    if (!node || !node.childNodes) return;
    [...node.childNodes].forEach(child => {
      if (child.nodeType !== Node.ELEMENT_NODE){
        return;
      }
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
        if (name === "style" && !styleRe.test(String(attr.value || "").trim())){
          child.removeAttribute("style");
        }
      });
      walk(child);
    });
  };
  walk(tpl.content);
  return tpl.innerHTML;
}

async function fetchJson(url, options = {}){
  const res = await fetch(url, {cache: "no-store", ...options});
  if (!res.ok){
    const payload = await res.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${res.status})`);
  }
  return await res.json();
}

function currentParams(){
  const params = new URLSearchParams();
  params.set("page", String(STATE.currentPage));
  params.set("page_size", String(STATE.pageSize));
  params.set("status", el("#statusSelect").value || "all");
  const q = el("#searchInput").value.trim();
  if (q) params.set("q", q);
  if (STATE.selectedLiveRunId){
    params.set("run_id", STATE.selectedLiveRunId);
    params.set("run_only", STATE.runOnlySelectedRun ? "true" : "false");
    if (STATE.selectedLiveBatchSize){
      params.set("batch_size", STATE.selectedLiveBatchSize);
    }
  }
  return params;
}

function detailParams(){
  const params = new URLSearchParams();
  if (STATE.selectedLiveRunId){
    params.set("run_id", STATE.selectedLiveRunId);
  }
  if (STATE.selectedLiveBatchSize){
    params.set("batch_size", STATE.selectedLiveBatchSize);
  }
  return params;
}

function issueBadgeClass(severity){
  const sev = String(severity || "").toLowerCase();
  if (sev === "error") return "sev-error";
  if (sev === "warn") return "sev-warn";
  return "sev-info";
}

function liveRunById(runId){
  const key = String(runId || "");
  return (STATE.liveRuns || []).find(row => String(row.run_id || "") === key) || null;
}

function liveRunLabel(row){
  const runId = String(row.run_id || "");
  const created = String(row.created_at || "");
  const shortCreated = created ? created.replace("T", " ").replace("Z", " UTC") : "unknown time";
  const batches = Array.isArray(row.batch_sizes) && row.batch_sizes.length
    ? row.batch_sizes.join(",")
    : "no batches";
  const liveFlag = row.ran_live_calls ? "live" : "dry";
  return `${runId} • ${liveFlag} • b=[${batches}] • n=${fmtInt(row.samples_total || 0)} • ${shortCreated}`;
}

function renderLiveRunMeta(){
  const node = el("#liveRunMeta");
  const run = liveRunById(STATE.selectedLiveRunId);
  if (!run){
    node.innerHTML = `
      <span><strong>Prediction Source:</strong> dataset prior only</span>
      <span><strong>Live Runs Detected:</strong> ${fmtInt((STATE.liveRuns || []).length)}</span>
    `;
    return;
  }
  const batchLabel = STATE.selectedLiveBatchSize || "auto";
  node.innerHTML = `
    <span><strong>Prediction Source:</strong> Gemini live run</span>
    <span><strong>Run:</strong> <code>${esc(run.run_id || "")}</code></span>
    <span><strong>Batch:</strong> ${esc(String(batchLabel))}</span>
    <span><strong>Samples In Run:</strong> ${fmtInt(run.samples_total || 0)}</span>
    <span><strong>Mode:</strong> ${run.ran_live_calls ? "live" : "dry/no live calls"}</span>
  `;
}

function populateBatchSelect(){
  const select = el("#liveBatchSelect");
  const run = liveRunById(STATE.selectedLiveRunId);
  select.innerHTML = `<option value="">auto</option>`;
  if (!run){
    select.disabled = true;
    STATE.selectedLiveBatchSize = "";
    return;
  }
  const batches = Array.isArray(run.batch_sizes) ? [...run.batch_sizes] : [];
  batches.sort((a, b) => Number(a) - Number(b));
  batches.forEach(batch => {
    const opt = document.createElement("option");
    opt.value = String(batch);
    opt.textContent = String(batch);
    select.appendChild(opt);
  });
  select.disabled = batches.length === 0;
  if (STATE.selectedLiveBatchSize && batches.includes(Number(STATE.selectedLiveBatchSize))){
    select.value = String(STATE.selectedLiveBatchSize);
  } else if (batches.length){
    const maxBatch = Math.max(...batches);
    STATE.selectedLiveBatchSize = String(maxBatch);
    select.value = String(maxBatch);
  } else {
    STATE.selectedLiveBatchSize = "";
    select.value = "";
  }
}

function renderLiveRunSelect(){
  const select = el("#liveRunSelect");
  select.innerHTML = `<option value="">prior-only (dataset)</option>`;
  (STATE.liveRuns || []).forEach(row => {
    const opt = document.createElement("option");
    opt.value = String(row.run_id || "");
    opt.textContent = liveRunLabel(row);
    select.appendChild(opt);
  });
  if (STATE.selectedLiveRunId && liveRunById(STATE.selectedLiveRunId)){
    select.value = STATE.selectedLiveRunId;
  } else {
    STATE.selectedLiveRunId = "";
    select.value = "";
  }
  populateBatchSelect();
  renderLiveRunMeta();
}

async function loadLiveRuns(){
  const payload = await fetchJson("/api/live-runs");
  STATE.liveRuns = Array.isArray(payload.runs) ? payload.runs : [];
  renderLiveRunSelect();
}

function getCurrentItems(){
  if (!STATE.samples || !Array.isArray(STATE.samples.items)) return [];
  return STATE.samples.items;
}

function selectedPositionInPage(){
  const items = getCurrentItems();
  return items.findIndex(item => Number(item.index) === Number(STATE.selectedIndex));
}

function updateDetailNavControls(){
  const prevBtn = el("#detailPrevBtn");
  const nextBtn = el("#detailNextBtn");
  const info = el("#detailNavInfo");
  if (!prevBtn || !nextBtn || !info) return;

  const totalRows = Number((STATE.summary && STATE.summary.total_rows) || 0);
  if (STATE.selectedIndex == null){
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    info.textContent = "No sample selected";
    return;
  }

  const items = getCurrentItems();
  const pos = selectedPositionInPage();
  const page = Number((STATE.samples && STATE.samples.page) || 1);
  const totalPages = Number((STATE.samples && STATE.samples.total_pages) || 1);
  const canPrev = pos > 0 || page > 1;
  const canNext = (pos >= 0 && pos < (items.length - 1)) || page < totalPages;
  prevBtn.disabled = !canPrev;
  nextBtn.disabled = !canNext;
  info.textContent = `Sample ${fmtInt(Number(STATE.selectedIndex) + 1)} / ${fmtInt(totalRows)}`;
}

function renderCards(summary){
  const cards = el("#cards");
  cards.innerHTML = "";
  const severityCounts = summary.severity_counts || {};
  const infoOnly = Number(summary.samples_with_issues || 0) - Number(summary.samples_with_problems || 0);
  const rows = [
    ["Total Rows", fmtInt(summary.total_rows)],
    ["Parse Errors", fmtInt(summary.parse_errors)],
    ["Rows With Mistakes", fmtInt(summary.samples_with_problems)],
    ["Rows Clean (No Warn/Error)", fmtInt(summary.samples_without_problems)],
    ["Rows With Info Only", fmtInt(Math.max(0, infoOnly))],
    ["Live Runs Available", fmtInt(summary.live_runs_count || 0)],
    ["Mixed Rows", fmtInt(summary.mixed_rows)],
    ["Pure Rows", fmtInt(summary.non_mixed_rows)],
    ["Avg Text Length", Number(summary.avg_text_length || 0).toFixed(2)],
    ["Severity Errors", fmtInt(severityCounts.error || 0)],
    ["Severity Warn", fmtInt(severityCounts.warn || 0)],
    ["Severity Info", fmtInt(severityCounts.info || 0)],
  ];
  rows.forEach(([k, v]) => {
    const node = document.createElement("div");
    node.className = "card";
    node.innerHTML = `<div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div>`;
    cards.appendChild(node);
  });
}

function renderCountRows(holderId, rows, palette = null){
  const holder = el(holderId);
  holder.innerHTML = "";
  const entries = Object.entries(rows || {});
  entries.sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0));
  const maxV = entries.length ? Math.max(...entries.map(([, v]) => Number(v || 0))) : 1;
  entries.slice(0, 30).forEach(([label, value]) => {
    const count = Number(value || 0);
    const width = maxV > 0 ? (100 * count / maxV) : 0;
    const color = palette ? safeColor(palette[label]) : "#3768ff";
    const row = document.createElement("div");
    row.className = "count-row";
    row.innerHTML = `
      <div class="label">
        <span class="dot" style="background:${color}"></span>
        <span title="${esc(label)}">${esc(label)}</span>
      </div>
      <div class="track"><div class="fill" style="width:${width.toFixed(1)}%; background:${color}"></div></div>
      <div class="value">${fmtInt(count)}</div>
    `;
    holder.appendChild(row);
  });
  if (!entries.length){
    holder.textContent = "No data.";
  }
}

function renderParseErrors(summary){
  const holder = el("#parseErrors");
  const rows = summary.parse_error_rows || [];
  if (!rows.length){
    holder.textContent = "No parse errors.";
    return;
  }
  const html = rows
    .map(row => `<div class="parse-row"><strong>line ${fmtInt(row.line_number)}:</strong> ${esc(row.message)}</div>`)
    .join("");
  holder.innerHTML = html;
}

async function loadSummary(){
  const summary = await fetchJson("/api/summary");
  STATE.summary = summary;
  el("#datasetMeta").innerHTML = `
    <span><strong>File:</strong> <code>${esc(summary.path || "")}</code></span>
    <span><strong>Expected Length:</strong> ${fmtInt(summary.expected_length || 0)}</span>
    <span><strong>Length Range:</strong> ${fmtInt(summary.min_text_length)}..${fmtInt(summary.max_text_length)}</span>
  `;
  renderCards(summary);
  renderCountRows("#issueCodeRows", summary.issue_code_counts || {});
  renderCountRows("#labelCharRows", summary.label_char_counts || {}, summary.palette || {});
  renderParseErrors(summary);
}

function renderSamples(payload){
  const tbody = el("#sampleRows");
  tbody.innerHTML = "";
  const items = payload.items || [];
  items.forEach(item => {
    const tr = document.createElement("tr");
    tr.className = "sample-row";
    tr.dataset.index = String(item.index);
    if (item.error_count > 0) tr.classList.add("row-error");
    if (item.warn_count > 0 && item.error_count === 0) tr.classList.add("row-warn");
    if (Number(item.mismatch_chars || 0) > 0) tr.classList.add("row-mismatch");
    if (isFailedSegmentationSource(item.prediction_source)) tr.classList.add("row-failed-seg");
    if (STATE.selectedIndex === item.index) tr.classList.add("active");
    const labels = (item.source_langs || []).join(", ");
    const issues = item.problem_count > 0
      ? `${fmtInt(item.error_count)}E / ${fmtInt(item.warn_count)}W`
      : (item.info_count > 0 ? `${fmtInt(item.info_count)}I` : "0");
    const mismatchChars = Number(item.mismatch_chars || 0);
    const mismatchRate = Number(item.mismatch_rate || 0);
    const predictionSource = String(item.prediction_source || "dataset_prior");
    const failedSource = isFailedSegmentationSource(predictionSource);
    const mismatchText = mismatchChars > 0
      ? `${fmtInt(mismatchChars)} (${(100 * mismatchRate).toFixed(1)}%)`
      : "0";
    const shownMismatch = failedSource ? "failed" : mismatchText;
    const predictionReason = String(item.prediction_failure_reason || "");
    tr.innerHTML = `
      <td>${fmtInt(item.index)}</td>
      <td>${fmtInt(item.line_number)}</td>
      <td title="${esc(item.snippet_id)}">${esc(item.snippet_id)}</td>
      <td title="${esc(item.task)}">${esc(item.task || "")}</td>
      <td>${item.mixed_truth ? "yes" : "no"}</td>
      <td>${fmtInt(item.text_len)}</td>
      <td>${esc(shownMismatch)}</td>
      <td>${esc(issues)}</td>
      <td title="${esc(labels)}">${esc(labels)}</td>
      <td title="${esc(predictionReason)}">${esc(predictionSource)}</td>
      <td title="${esc(item.preview)}">${esc(item.preview)}</td>
    `;
    tr.addEventListener("click", async () => {
      STATE.selectedIndex = item.index;
      [...tbody.querySelectorAll("tr")].forEach(node => node.classList.remove("active"));
      tr.classList.add("active");
      await loadDetail(item.index);
    });
    tbody.appendChild(tr);
  });

  el("#sampleMeta").textContent =
    `${fmtInt(payload.total)} rows • page ${fmtInt(payload.page)} / ${fmtInt(payload.total_pages)}`
    + (payload.run_id ? ` • run=${payload.run_id}` : "")
    + (payload.run_only ? ` • filtered to run samples (${fmtInt(payload.run_samples_total || 0)})` : "");
  el("#pageInfo").textContent = `Page ${fmtInt(payload.page)} of ${fmtInt(payload.total_pages)}`;
  el("#prevPage").disabled = payload.page <= 1;
  el("#nextPage").disabled = payload.page >= payload.total_pages;
  updateDetailNavControls();
}

async function loadSamples(){
  const params = currentParams();
  const payload = await fetchJson(`/api/samples?${params.toString()}`);
  STATE.samples = payload;
  renderSamples(payload);

  if (STATE.selectedIndex == null && payload.items && payload.items.length){
    STATE.selectedIndex = payload.items[0].index;
    await loadDetail(STATE.selectedIndex);
  }
  updateDetailNavControls();
}

function renderSegmentTable(holderId, segments){
  const holder = el(holderId);
  if (!segments || !segments.length){
    holder.innerHTML = "<em>No segments.</em>";
    return;
  }
  const rows = segments.map(seg => {
    const width = Number(seg.end || 0) - Number(seg.start || 0);
    return `<tr><td>${fmtInt(seg.start)}</td><td>${fmtInt(seg.end)}</td><td>${fmtInt(width)}</td><td>${esc(seg.label)}</td></tr>`;
  }).join("");
  holder.innerHTML = `
    <table>
      <thead><tr><th>start</th><th>end</th><th>len</th><th>label</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function loadDetail(index){
  const params = detailParams();
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const detail = await fetchJson(`/api/sample/${index}${suffix}`);
  STATE.selectedDetail = detail;

  const issueText = detail.issues.length
    ? `${fmtInt(detail.issues.length)} notes`
    : "No notes";
  const failedSource = isFailedSegmentationSource(detail.prediction_source);
  const failureReason = String(detail.prediction_failure_reason || "").trim();
  const failureReasonText = failureReason || "segmentation unavailable from live run";
  const mismatchText = failedSource
    ? "n/a (failed segmentation)"
    : `${fmtInt(detail.mismatch_chars || 0)} (${(100 * Number(detail.mismatch_rate || 0)).toFixed(1)}%)`;
  el("#detailMeta").innerHTML = `
    <span><strong>#${fmtInt(detail.index)}</strong> line ${fmtInt(detail.line_number)}</span>
    <span><strong>snippet:</strong> ${esc(detail.snippet_id)}</span>
    <span><strong>task:</strong> ${esc(detail.task || "")}</span>
    <span><strong>mixed:</strong> ${detail.mixed_truth ? "yes" : "no"}</span>
    <span><strong>len:</strong> ${fmtInt(detail.text_len)}</span>
    <span><strong>boundary:</strong> ${fmtInt(detail.boundary)}</span>
    <span><strong>labels:</strong> ${esc((detail.source_langs || []).join(", "))}</span>
    <span><strong>status:</strong> ${issueText}</span>
    <span><strong>prediction source:</strong> ${esc(String(detail.prediction_source || "dataset_prior"))}</span>
    ${failedSource ? `<span><strong>failed segmentation:</strong> ${esc(failureReasonText)}</span>` : ""}
    <span><strong>mismatch:</strong> ${esc(mismatchText)}</span>
  `;

  const issueRows = el("#issueRows");
  const uiIssues = Array.isArray(detail.issues) ? [...detail.issues] : [];
  if (failedSource){
    uiIssues.push({
      severity: "warn",
      code: "failed_segmentation",
      message: `Live run did not yield a trusted segmentation (${failureReasonText}).`,
    });
  }
  if (!uiIssues.length){
    issueRows.innerHTML = `<div class="issue sev-info">No validation issues detected.</div>`;
  } else {
    issueRows.innerHTML = uiIssues.map(issue => `
      <div class="issue ${issueBadgeClass(issue.severity)}">
        <span class="code">${esc(issue.code)}</span>
        <span class="msg">${esc(issue.message)}</span>
      </div>
    `).join("");
  }

  el("#truthRender").innerHTML = sanitizeRenderHtml(detail.truth_html || "");
  el("#predictedRender").innerHTML = sanitizeRenderHtml(detail.predicted_html || "");
  el("#diffRender").innerHTML = sanitizeRenderHtml(detail.diff_html || "");

  renderSegmentTable("#truthTable", detail.truth_segments || []);
  renderSegmentTable("#predTable", detail.predicted_segments || []);

  el("#metadataJson").textContent = JSON.stringify(detail.metadata || {}, null, 2);
  el("#rawJson").textContent = JSON.stringify(detail.raw || {}, null, 2);
  updateDetailNavControls();
}

async function gotoNextDetail(){
  if (!STATE.samples) return;
  const items = getCurrentItems();
  if (!items.length) return;

  let pos = selectedPositionInPage();
  if (pos < 0){
    STATE.selectedIndex = items[0].index;
    await loadDetail(STATE.selectedIndex);
    return;
  }
  if (pos < items.length - 1){
    STATE.selectedIndex = items[pos + 1].index;
    await loadDetail(STATE.selectedIndex);
    renderSamples(STATE.samples);
    return;
  }
  if (STATE.samples.page < STATE.samples.total_pages){
    STATE.currentPage += 1;
    await loadSamples();
    const newItems = getCurrentItems();
    if (newItems.length){
      STATE.selectedIndex = newItems[0].index;
      await loadDetail(STATE.selectedIndex);
      renderSamples(STATE.samples);
    }
  }
}

async function gotoPrevDetail(){
  if (!STATE.samples) return;
  const items = getCurrentItems();
  if (!items.length) return;

  let pos = selectedPositionInPage();
  if (pos < 0){
    STATE.selectedIndex = items[0].index;
    await loadDetail(STATE.selectedIndex);
    return;
  }
  if (pos > 0){
    STATE.selectedIndex = items[pos - 1].index;
    await loadDetail(STATE.selectedIndex);
    renderSamples(STATE.samples);
    return;
  }
  if (STATE.samples.page > 1){
    STATE.currentPage -= 1;
    await loadSamples();
    const newItems = getCurrentItems();
    if (newItems.length){
      STATE.selectedIndex = newItems[newItems.length - 1].index;
      await loadDetail(STATE.selectedIndex);
      renderSamples(STATE.samples);
    }
  }
}

function setupEvents(){
  el("#statusSelect").addEventListener("change", async () => {
    STATE.currentPage = 1;
    await loadSamples();
  });
  el("#pageSizeSelect").addEventListener("change", async () => {
    STATE.pageSize = Number(el("#pageSizeSelect").value || 25);
    STATE.currentPage = 1;
    await loadSamples();
  });
  el("#searchInput").addEventListener("keydown", async ev => {
    if (ev.key !== "Enter") return;
    STATE.currentPage = 1;
    await loadSamples();
  });
  el("#refreshBtn").addEventListener("click", async () => {
    await loadLiveRuns();
    await loadSummary();
    await loadSamples();
  });
  el("#reloadBtn").addEventListener("click", async () => {
    await fetchJson("/api/reload", {method: "POST"});
    STATE.currentPage = 1;
    STATE.selectedIndex = null;
    await loadLiveRuns();
    await loadSummary();
    await loadSamples();
  });
  el("#liveRunSelect").addEventListener("change", async () => {
    STATE.selectedLiveRunId = el("#liveRunSelect").value || "";
    STATE.currentPage = 1;
    STATE.selectedIndex = null;
    populateBatchSelect();
    renderLiveRunMeta();
    await loadSamples();
  });
  el("#liveBatchSelect").addEventListener("change", async () => {
    STATE.selectedLiveBatchSize = el("#liveBatchSelect").value || "";
    renderLiveRunMeta();
    await loadSamples();
    if (STATE.selectedIndex != null){
      await loadDetail(STATE.selectedIndex);
    }
  });
  el("#runOnlyCheck").addEventListener("change", async () => {
    STATE.runOnlySelectedRun = Boolean(el("#runOnlyCheck").checked);
    STATE.currentPage = 1;
    STATE.selectedIndex = null;
    await loadSamples();
  });
  el("#prevPage").addEventListener("click", async () => {
    if (!STATE.samples || STATE.samples.page <= 1) return;
    STATE.currentPage -= 1;
    await loadSamples();
  });
  el("#nextPage").addEventListener("click", async () => {
    if (!STATE.samples || STATE.samples.page >= STATE.samples.total_pages) return;
    STATE.currentPage += 1;
    await loadSamples();
  });
  el("#detailPrevBtn").addEventListener("click", async () => {
    await gotoPrevDetail();
  });
  el("#detailNextBtn").addEventListener("click", async () => {
    await gotoNextDetail();
  });
  document.addEventListener("keydown", async ev => {
    if (ev.defaultPrevented) return;
    const target = ev.target;
    const tag = (target && target.tagName ? String(target.tagName).toLowerCase() : "");
    if (tag === "input" || tag === "textarea" || tag === "select"){
      return;
    }
    if (ev.key === "ArrowRight"){
      ev.preventDefault();
      await gotoNextDetail();
      return;
    }
    if (ev.key === "ArrowLeft"){
      ev.preventDefault();
      await gotoPrevDetail();
    }
  });
}

async function boot(){
  try {
    setupEvents();
    el("#runOnlyCheck").checked = Boolean(STATE.runOnlySelectedRun);
    await loadLiveRuns();
    await loadSummary();
    await loadSamples();
    renderLiveRunMeta();
  } catch (err){
    console.error(err);
    alert(String(err && err.message ? err.message : err));
  }
}

window.addEventListener("DOMContentLoaded", boot);
