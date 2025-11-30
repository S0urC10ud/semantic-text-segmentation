const state = {
  runs: [],
  filteredRuns: [],
  selectedRunPath: null,
  directory: "",
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

function groupRunsByDirectory(runs) {
  const grouped = new Map();
  runs.forEach((run) => {
    const dir = run.directory || ".";
    if (!grouped.has(dir)) {
      grouped.set(dir, []);
    }
    grouped.get(dir).push(run);
  });
  const sortDirectories = (a, b) => {
    if (a === "." && b !== ".") return -1;
    if (b === "." && a !== ".") return 1;
    return a.localeCompare(b);
  };
  return Array.from(grouped.entries())
    .sort(([a], [b]) => sortDirectories(a, b))
    .map(([directory, groupRuns]) => ({ directory, runs: groupRuns }));
}

function buildRunRequestUrl(run) {
  const url = new URL(`/api/runs/${encodeURIComponent(run.run_id)}`, window.location.origin);
  if (run.path) {
    url.searchParams.set("path", run.path);
  }
  return url.toString();
}

function buildDownloadUrl(detail) {
  const url = new URL(`/api/runs/${encodeURIComponent(detail.run_id)}/download`, window.location.origin);
  if (detail.relative_path) {
    url.searchParams.set("path", detail.relative_path);
  }
  return url.toString();
}

function setDetailPlaceholder(message) {
  elements.detailPanel.innerHTML = `<div class="detail-placeholder">${message}</div>`;
}

async function fetchRuns() {
  elements.runsList.innerHTML = '<div class="placeholder">Loading runs…</div>';
  try {
    const response = await fetch("/api/runs");
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }
    const payload = await response.json();
    state.runs = payload.runs || [];
    state.filteredRuns = [...state.runs];
    state.directory = payload.directory || "";
    state.selectedRunPath = null;
    elements.directory.textContent = state.directory || "Unknown directory";
    elements.runCount.textContent = formatNumber(state.runs.length);
    const searchPlaceholder = "Filter by run id, directory, or source…";
    elements.runSearch.placeholder = searchPlaceholder;
    renderRunList();
    if (state.filteredRuns.length) {
      const firstRun = state.filteredRuns[0];
      selectRun(firstRun.path || firstRun.run_id);
    } else {
      setDetailPlaceholder("No snapshots found in the configured directory.");
    }
  } catch (err) {
    elements.runsList.innerHTML = `<div class="placeholder">Failed to load runs: ${err}</div>`;
    setDetailPlaceholder("Unable to load snapshots. Ensure the backend is running.");
  }
}

function handleSearch(event) {
  const query = event.target.value.trim().toLowerCase();
  if (!query) {
    state.filteredRuns = [...state.runs];
  } else {
    state.filteredRuns = state.runs.filter((run) => {
      const haystack = [
        run.run_id,
        run.model,
        run.input_source,
        run.directory,
        run.file_name,
        run.path,
        run.purity_status,
        run.purity_language,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    });
  }
  renderRunList();
}

function renderRunList() {
  const runs = state.filteredRuns;
  const container = elements.runsList;
  container.innerHTML = "";
  if (!runs.length) {
    container.innerHTML = '<div class="placeholder">No runs match the current filter.</div>';
    return;
  }
  const grouped = groupRunsByDirectory(runs);
  grouped.forEach((group) => {
    const groupEl = document.createElement("div");
    groupEl.className = "run-group";
    const header = document.createElement("div");
    header.className = "run-group-header";
    const title = document.createElement("div");
    title.className = "run-group-title";
    title.textContent = directoryLabel(group.directory);
    const count = document.createElement("div");
    count.className = "run-group-count";
    const countValue = group.runs.length;
    count.textContent = `${countValue} file${countValue === 1 ? "" : "s"}`;
    header.append(title, count);
    groupEl.appendChild(header);

    group.runs.forEach((run) => {
      const card = elements.runCardTemplate.content.firstElementChild.cloneNode(true);
      const runKey = run.path || run.run_id;
      card.dataset.runPath = runKey || "";
      card.querySelector(".run-card-id").textContent = run.run_id || run.file_name;
      card.querySelector(".run-card-timestamp").textContent = formatTimestamp(run.timestamp);
      const dirEl = card.querySelector(".run-card-dir");
      if (dirEl) {
        const pill = document.createElement("span");
        pill.className = "dir-pill";
        pill.textContent = directoryLabel(run.directory);
        dirEl.textContent = "";
        dirEl.appendChild(pill);
      }
      card.querySelector(".run-card-segments").textContent = formatNumber(run.segments);
      card.querySelector(".run-card-characters").textContent = formatNumber(run.characters);
      card.querySelector(".run-card-model").textContent = run.model || "—";
      card.querySelector(".run-card-source").textContent = run.input_source || "Input source unknown";
      const purityInfo = formatPurity(run.purity_check);
      const purityEl = document.createElement("div");
      purityEl.className = "run-card-purity";
      purityEl.textContent = purityInfo.label;
      purityEl.title = purityInfo.reason || purityInfo.label;
      card.querySelector(".run-card-meta").appendChild(purityEl);
      if (run.error) {
        card.classList.add("errored");
        card.querySelector(".run-card-source").textContent = run.error;
      }
      if (runKey && runKey === state.selectedRunPath) {
        card.classList.add("selected");
      }
      card.addEventListener("click", () => selectRun(runKey));
      groupEl.appendChild(card);
    });

    container.appendChild(groupEl);
  });
}

function createMetaCard(label, value, options = {}) {
  const card = document.createElement("div");
  card.className = "meta-card";
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
    empty.textContent = "Gemini did not emit any ground-truth blocks for this run.";
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
  const download = document.createElement("a");
  download.className = "download-link";
  download.href = buildDownloadUrl(detail);
  download.textContent = "Download JSON";
  header.append(title, download);
  panel.appendChild(header);

  const metaGrid = document.createElement("div");
  metaGrid.className = "meta-grid";
  metaGrid.appendChild(createMetaCard("Directory", directoryLabel(detail.directory)));
  metaGrid.appendChild(createMetaCard("File", detail.relative_path || detail.file_name));
  metaGrid.appendChild(createMetaCard("Timestamp", formatTimestamp(detail.metadata?.timestamp)));
  metaGrid.appendChild(createMetaCard("Model", detail.metadata?.model || "Unknown"));
  metaGrid.appendChild(createMetaCard("Input Source", detail.metadata?.input_source || "Unspecified"));
  metaGrid.appendChild(createMetaCard("Segments", formatNumber(detail.segments.length)));
  metaGrid.appendChild(createMetaCard("Characters", formatNumber(detail.total_characters)));
  metaGrid.appendChild(createMetaCard("Unique Types", formatNumber(detail.unique_types.length)));
  const purityInfo = formatPurity(detail.metadata?.purity_check);
  const purityValue = purityInfo.reason ? `${purityInfo.label} — ${purityInfo.reason}` : purityInfo.label;
  metaGrid.appendChild(createMetaCard("Purity Check", purityValue, { title: purityValue }));
  panel.appendChild(metaGrid);

  const legend = renderLegend(detail.unique_types);
  panel.appendChild(legend);
  panel.appendChild(renderSegments(detail.segments));
}

async function selectRun(runPath) {
  if (!runPath) {
    return;
  }
  const run = state.runs.find((item) => (item.path || item.run_id) === runPath);
  if (!run) {
    setDetailPlaceholder("Selected run is no longer available.");
    state.selectedRunPath = null;
    renderRunList();
    return;
  }
  const runKey = run.path || run.run_id;
  if (runKey === state.selectedRunPath) {
    return;
  }
  state.selectedRunPath = runKey;
  renderRunList();
  setDetailPlaceholder("Loading run details…");
  try {
    const response = await fetch(buildRunRequestUrl(run));
    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }
    const detail = await response.json();
    renderDetail(detail);
  } catch (err) {
    const label = run.run_id || run.file_name || "run";
    setDetailPlaceholder(`Failed to load run ${label}: ${err}`);
  }
}

function init() {
  elements.directory = document.getElementById("snapshotDirectory");
  elements.runCount = document.getElementById("runCount");
  elements.runsList = document.getElementById("runsList");
  elements.runSearch = document.getElementById("runSearch");
  elements.detailPanel = document.getElementById("detailPanel");
  elements.runCardTemplate = document.getElementById("runCardTemplate");
  elements.runSearch.addEventListener("input", handleSearch);
  fetchRuns();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
