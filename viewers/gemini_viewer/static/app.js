const BASE_PATH = "/ground-truth";
const API_BASE = `${BASE_PATH}/api`;

const state = {
  types: [],
  directory: "",
  totalFiles: 0,
  selectedType: null,
};

const elements = {};

const palette = [
  "#0ea5e9",
  "#ec4899",
  "#f97316",
  "#22c55e",
  "#a855f7",
  "#14b8a6",
  "#8b5cf6",
  "#facc15",
  "#ef4444",
  "#3b82f6",
];
const typeColors = new Map();

function directoryLabel(value) {
  if (!value || value === ".") {
    return "(root)";
  }
  return value;
}

function colorForType(type) {
  const key = type || "unknown";
  if (typeColors.has(key)) {
    return typeColors.get(key);
  }
  let hash = 0;
  for (let i = 0; i < key.length; i += 1) {
    hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  }
  const color = palette[hash % palette.length];
  typeColors.set(key, color);
  return color;
}

function formatNumber(value) {
  return Intl.NumberFormat().format(value ?? 0);
}

function formatTimestamp(ts) {
  if (!ts) {
    return "Unknown";
  }
  const parsed = new Date(ts);
  if (Number.isNaN(parsed.getTime())) {
    return ts;
  }
  return parsed.toLocaleString();
}

function setDetailPlaceholder(message) {
  elements.detailPanel.innerHTML = `<div class="detail-placeholder">${message}</div>`;
}

function buildDownloadUrl(detail) {
  const url = new URL(`${API_BASE}/runs/${encodeURIComponent(detail.run_id)}/download`, window.location.origin);
  if (detail.relative_path) {
    url.searchParams.set("path", detail.relative_path);
  }
  return url.toString();
}

function createMetaCard(label, value, options = {}) {
  const card = document.createElement("div");
  card.className = "meta-card";
  if (options.className) {
    card.classList.add(options.className);
  }
  const labelEl = document.createElement("div");
  labelEl.className = "label";
  labelEl.textContent = label;
  const valueEl = document.createElement("div");
  valueEl.className = "value";
  valueEl.textContent = value ?? "—";
  if (options.title) {
    valueEl.title = options.title;
  }
  card.append(labelEl, valueEl);
  return card;
}

function formatPurity(purity) {
  if (!purity) {
    return { label: "Not recorded", reason: "" };
  }
  if (purity.status === "skipped") {
    return { label: "Skipped", reason: "" };
  }
  const primary = purity.is_pure ? "Pure" : "Mixed";
  const lang = purity.language || "unknown";
  const mixed = Array.isArray(purity.mixed_types) ? purity.mixed_types.filter(Boolean).join(", ") : "";
  let label = `${primary} (${lang})`;
  if (!purity.is_pure && mixed) {
    label += ` • ${mixed}`;
  }
  return { label, reason: purity.reason || "" };
}

function renderLegend(types) {
  const legend = document.createElement("div");
  legend.className = "type-legend";
  if (!types.length) {
    const empty = document.createElement("div");
    empty.className = "segments-empty";
    empty.textContent = "No segments were emitted in this run.";
    legend.appendChild(empty);
    return legend;
  }
  types.forEach((type) => {
    const pill = document.createElement("div");
    pill.className = "type-pill";
    const color = colorForType(type);
    const swatch = document.createElement("span");
    swatch.className = "type-pill-color";
    swatch.style.backgroundColor = color;
    pill.appendChild(swatch);
    pill.append(type || "unknown");
    legend.appendChild(pill);
  });
  return legend;
}

function renderSegments(segments) {
  const wrapper = document.createElement("div");
  wrapper.className = "segments-list";
  if (!segments.length) {
    const empty = document.createElement("div");
    empty.className = "segments-empty";
    empty.textContent = "Gemini did not emit any ground-truth blocks for this selection.";
    wrapper.appendChild(empty);
    return wrapper;
  }
  segments.forEach((segment) => {
    const block = document.createElement("article");
    block.className = "segment-block";
    const color = colorForType(segment.type);
    block.style.borderLeftColor = color;
    block.style.boxShadow = `0 1px 3px ${color}22`;

    const header = document.createElement("div");
    header.className = "segment-header";
    const typeEl = document.createElement("div");
    typeEl.className = "segment-type";
    typeEl.textContent = segment.type || "unknown";
    const metaEl = document.createElement("div");
    metaEl.className = "segment-meta";
    metaEl.textContent = `#${segment.index} · ${formatNumber(segment.length)} chars`;
    header.append(typeEl, metaEl);

    const content = document.createElement("pre");
    content.className = "segment-content";
    content.textContent = segment.content || "";

    block.append(header, content);
    wrapper.appendChild(block);
  });
  return wrapper;
}

function renderDetail(detail) {
  const panel = elements.detailPanel;
  panel.innerHTML = "";

  const header = document.createElement("div");
  header.className = "detail-header";
  const title = document.createElement("div");
  title.className = "detail-run-id";
  title.textContent = detail.run_id;

  if (detail.selected_type) {
    const selectedPill = document.createElement("div");
    selectedPill.className = "type-pill";
    selectedPill.title = "Showing a random file from this source type.";
    const swatch = document.createElement("span");
    swatch.className = "type-pill-color";
    swatch.style.backgroundColor = colorForType(detail.selected_type);
    selectedPill.append(swatch, detail.selected_type);
    header.appendChild(selectedPill);
  }

  const download = document.createElement("a");
  download.className = "download-link";
  download.href = buildDownloadUrl(detail);
  download.textContent = "Download JSON";
  header.append(title, download);
  panel.appendChild(header);

  const metaGrid = document.createElement("div");
  metaGrid.className = "meta-grid";
  metaGrid.appendChild(createMetaCard("Directory", directoryLabel(detail.directory)));
  metaGrid.appendChild(
    createMetaCard("File", detail.relative_path || detail.file_name, { className: "file-meta", title: detail.path })
  );
  metaGrid.appendChild(createMetaCard("Timestamp", formatTimestamp(detail.metadata?.timestamp)));
  metaGrid.appendChild(createMetaCard("Model", detail.metadata?.model || "Unknown"));
  metaGrid.appendChild(createMetaCard("Content Type", detail.selected_type || "Unknown"));
  metaGrid.appendChild(createMetaCard("Segments (shown)", formatNumber(detail.segments?.length ?? 0)));
  if (typeof detail.segments_total === "number") {
    metaGrid.appendChild(createMetaCard("Segments (file)", formatNumber(detail.segments_total)));
  }
  metaGrid.appendChild(createMetaCard("Characters (shown)", formatNumber(detail.total_characters)));
  if (typeof detail.total_characters_all === "number") {
    metaGrid.appendChild(createMetaCard("Characters (file)", formatNumber(detail.total_characters_all)));
  }
  const purityInfo = formatPurity(detail.metadata?.purity_check);
  const purityValue = purityInfo.reason ? `${purityInfo.label} — ${purityInfo.reason}` : purityInfo.label;
  metaGrid.appendChild(createMetaCard("Purity Check", purityValue, { title: purityValue }));
  panel.appendChild(metaGrid);

  const legend = renderLegend(detail.unique_types || []);
  panel.appendChild(legend);
  panel.appendChild(renderSegments(detail.segments || []));
}

async function loadRandomSample(typeId) {
  if (!typeId) {
    return;
  }
  state.selectedType = typeId;
  renderTypeList();
  setDetailPlaceholder(`Loading a random ${typeId} file…`);
  try {
    const url = new URL(`${API_BASE}/content-types/sample`, window.location.origin);
    url.searchParams.set("content_type", typeId);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }
    const detail = await response.json();
    renderDetail(detail);
  } catch (err) {
    setDetailPlaceholder(`Failed to load a sample for ${typeId}: ${err}`);
  }
}

function renderTypeList() {
  const container = elements.typesList;
  container.innerHTML = "";
  if (!state.types.length) {
    container.innerHTML = '<div class="placeholder">No content types found in this directory.</div>';
    return;
  }
  state.types.forEach((typeInfo) => {
    const card = elements.typeCardTemplate.content.firstElementChild.cloneNode(true);
    const typeName = typeInfo.label || typeInfo.id || "unknown";
    const typeId = typeInfo.id || typeName;
    card.dataset.type = typeId;
    card.querySelector(".type-name").textContent = typeName.toUpperCase();
    card.querySelector(".type-card-files").textContent = formatNumber(typeInfo.file_count);
    const pathEl = card.querySelector(".type-card-segments");
    pathEl.textContent = typeId === "." ? "(root)" : typeId;
    pathEl.classList.add("muted");
    const swatch = card.querySelector(".type-color");
    swatch.style.backgroundColor = colorForType(typeId);
    if (state.selectedType === typeId) {
      card.classList.add("selected");
    }
    card.addEventListener("click", () => loadRandomSample(typeId));
    container.appendChild(card);
  });
}

async function fetchContentTypes() {
  elements.typesList.innerHTML = '<div class="placeholder">Loading content types…</div>';
  try {
    const response = await fetch(`${API_BASE}/content-types`);
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }
    const payload = await response.json();
    state.types = payload.types || [];
    state.directory = payload.directory || "";
    state.totalFiles = payload.total_files ?? 0;
    state.selectedType = null;
    elements.directory.textContent = state.directory ? `Directory: ${state.directory}` : "Directory unknown";
    elements.snapshotStats.textContent = `Files: ${formatNumber(state.totalFiles)}`;
    elements.typeCount.textContent = formatNumber(state.types.length);
    elements.fileCount.textContent = formatNumber(state.totalFiles);
    renderTypeList();
    setDetailPlaceholder("Select a content type to view a random file's segmentation.");
  } catch (err) {
    elements.typesList.innerHTML = `<div class="placeholder">Failed to load content types: ${err}</div>`;
    setDetailPlaceholder("Unable to load snapshots. Ensure the backend is running.");
  }
}

function init() {
  elements.directory = document.getElementById("snapshotDirectory");
  elements.snapshotStats = document.getElementById("snapshotStats");
  elements.typeCount = document.getElementById("typeCount");
  elements.fileCount = document.getElementById("fileCount");
  elements.typesList = document.getElementById("typesList");
  elements.detailPanel = document.getElementById("detailPanel");
  elements.typeCardTemplate = document.getElementById("typeCardTemplate");
  fetchContentTypes();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
