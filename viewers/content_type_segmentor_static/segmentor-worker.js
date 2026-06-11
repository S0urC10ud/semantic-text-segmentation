const STATIC_BASE = './';

// Reuse the cache-busting query the page passed on the worker URL.
const ASSET_VERSION = self.location.search || '';

importScripts(STATIC_BASE + 'js/mamba.js' + ASSET_VERSION);
importScripts(STATIC_BASE + 'js/webgpu_matmul.js' + ASSET_VERSION);
importScripts(STATIC_BASE + 'js/postprocess.js' + ASSET_VERSION);
importScripts(STATIC_BASE + 'js/unet.js' + ASSET_VERSION);

const STATE = {
  manifest: null,
  // Mamba state
  mambaModel: null,
  mambaGpu: null,
  mambaLoaded: false,
  mambaLoading: null,
  // U-Net state
  unetModel: null,
  unetLoaded: false,
  unetLoading: null,
  // Current selection
  activeModel: 'mamba',
};

async function fetchOrThrow(url, type='text') {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url} (${response.status}).`);
  return type === 'arrayBuffer' ? response.arrayBuffer() : response.text();
}

async function loadManifest() {
  if (STATE.manifest) return STATE.manifest;
  const manifestStr = await fetchOrThrow(`${STATIC_BASE}assets/model_manifest.json`, 'text');
  STATE.manifest = JSON.parse(manifestStr);
  return STATE.manifest;
}

async function loadMamba() {
  if (STATE.mambaLoaded) return;
  if (STATE.mambaLoading) return STATE.mambaLoading;
  
  STATE.mambaLoading = (async () => {
    const manifest = await loadManifest();
    
    const weightsBuffer = await fetchOrThrow(`${STATIC_BASE}assets/sfullfiles4_weights.bin`, 'arrayBuffer');
    const metaStr = await fetchOrThrow(`${STATIC_BASE}assets/sfullfiles4_weights_meta.json`, 'text');
    const meta = JSON.parse(metaStr);
    
    const weightsDict = {};
    for (const [key, info] of Object.entries(meta)) {
      weightsDict[key] = new Float32Array(weightsBuffer, info.offset * 4, info.length);
    }
    STATE.mambaModel = new MambaSegmentor(manifest, weightsDict);
    
    const gpu = new WebGPUMatMul();
    await gpu.init();
    STATE.mambaGpu = gpu;
    
    STATE.mambaLoaded = true;
  })();
  
  await STATE.mambaLoading;
}

async function loadUNet() {
  if (STATE.unetLoaded) return;
  if (STATE.unetLoading) return STATE.unetLoading;
  
  STATE.unetLoading = (async () => {
    const manifest = await loadManifest();
    
    const weightsBuffer = await fetchOrThrow(`${STATIC_BASE}assets/unet_al_weights.bin`, 'arrayBuffer');
    const metaStr = await fetchOrThrow(`${STATIC_BASE}assets/unet_al_weights_meta.json`, 'text');
    const meta = JSON.parse(metaStr);
    
    const weightsDict = {};
    for (const [key, info] of Object.entries(meta)) {
      weightsDict[key] = new Float32Array(weightsBuffer, info.offset * 4, info.length);
    }
    STATE.unetModel = new UNetSegmentor(manifest, weightsDict);
    
    STATE.unetLoaded = true;
  })();
  
  await STATE.unetLoading;
}

async function ensureModelLoaded(modelName) {
  STATE.activeModel = modelName || 'mamba';
  if (STATE.activeModel === 'unet') {
    await loadUNet();
  } else {
    await loadMamba();
  }
  const manifest = await loadManifest();
  return {
    loaded: true,
    model_id: STATE.activeModel === 'unet' ? 'U-Net (CNN)' : 'Mamba (SSM)',
    num_classes: manifest.num_classes,
    window_bytes: manifest.window_bytes,
    window_stride_bytes: manifest.window_stride_bytes,
    other_threshold: manifest.other_threshold,
    max_input_bytes: manifest.max_input_bytes,
    postprocess_min_run_chars: manifest.postprocess_min_run_chars,
    postprocess_boundary_snap_max_shift: manifest.postprocess_boundary_snap_max_shift,
    label_order: manifest.label_order,
    display_labels: manifest.display_labels,
    runtime: STATE.activeModel === 'unet' ? 'unet-js' : 'hybrid-webgpu',
    activeModel: STATE.activeModel,
  };
}

async function runSegment(payloadObj) {
  const text = payloadObj?.text || '';
  const threshold = payloadObj?.threshold;
  const ppOptions = payloadObj?.ppOptions || {};
  
  const manifest = await loadManifest();
  
  const nextThreshold = Number.isFinite(Number(threshold))
    ? Number(threshold)
    : Number(manifest?.other_threshold ?? 0.3);
    
  const started = performance.now();
  
  const encoder = new TextEncoder();
  const bytes = encoder.encode(text);
  const seq = bytes.length;
  
  const inputArr = new Int32Array(seq);
  for (let i = 0; i < seq; i++) inputArr[i] = bytes[i];
  
  if (seq === 0) {
    return {
      text, segments: [], char_top_probs: [], char_confidences: [], stats: [],
      input_bytes: 0, window_count: 0, other_threshold: nextThreshold,
      elapsed_ms: 0, runtime: STATE.activeModel
    };
  }

  // Run inference based on active model
  let probs;
  if (STATE.activeModel === 'unet') {
    await loadUNet();
    probs = STATE.unetModel.predict_window_probs(inputArr);
  } else {
    await loadMamba();
    probs = await STATE.mambaModel.predict_window_probs(inputArr, STATE.mambaGpu);
  }
  
  const num_classes = manifest.num_classes;
  const raw_labels = new Int32Array(seq);
  for(let i = 0; i < seq; i++) {
    let best_j = 0;
    let best_p = -1;
    for(let j=0; j < num_classes; j++) {
      if (probs[i*num_classes+j] > best_p) {
        best_p = probs[i*num_classes+j];
        best_j = j;
      }
    }
    raw_labels[i] = best_j;
  }

  const final_labels = postprocessCharLabels(text, raw_labels, probs, {
    threshold: nextThreshold,
    otherId: num_classes,
    ppOptions: ppOptions
  });
  
  const segments = [];
  let currentLabel = -1;
  let currentStart = 0;
  for (let i = 0; i < seq; i++) {
    const L = final_labels[i];
    if (L !== currentLabel) {
      if (currentLabel !== -1) {
        segments.push({ start: currentStart, end: i, label_id: currentLabel });
      }
      currentLabel = L;
      currentStart = i;
    }
  }
  if (currentLabel !== -1) {
    segments.push({ start: currentStart, end: seq, label_id: currentLabel });
  }

  const statsMap = new Map();
  for(let i=0; i<seq; i++) {
    const L = final_labels[i];
    statsMap.set(L, (statsMap.get(L) || 0) + 1);
  }
  const stats = [];
  for(const [id, count] of statsMap.entries()) {
    stats.push({ id, count, pct: (count / seq) * 100 });
  }

  const char_top_probs = [];
  const char_confidences = [];
  for(let i=0; i<seq; i++) {
    const p = [];
    for(let j=0; j<num_classes; j++) {
      p.push({ id: j, prob: probs[i*num_classes+j] });
    }
    p.sort((a,b) => b.prob - a.prob);
    char_top_probs.push(p.slice(0, 5));
    char_confidences.push(p[0].prob);
  }

  return {
    text,
    segments,
    char_top_probs,
    char_confidences,
    stats,
    input_bytes: seq,
    window_count: 1,
    other_threshold: nextThreshold,
    elapsed_ms: Number((performance.now() - started).toFixed(1)),
    runtime: STATE.activeModel === 'unet' ? 'unet-js' : 'hybrid-webgpu',
    model_id: STATE.activeModel === 'unet' ? 'U-Net (CNN)' : 'Mamba (SSM)',
    activeModel: STATE.activeModel,
  };
}

self.addEventListener('message', async event => {
  const { type, requestId, payload } = event.data || {};
  try {
    if (type === 'load') {
      const result = await ensureModelLoaded(payload?.model);
      self.postMessage({ type: 'result', requestId, payload: result });
      return;
    }
    if (type === 'switch_model') {
      const result = await ensureModelLoaded(payload?.model);
      self.postMessage({ type: 'result', requestId, payload: result });
      return;
    }
    if (type === 'segment') {
      const result = await runSegment(payload || {});
      self.postMessage({ type: 'result', requestId, payload: result });
      return;
    }
    throw new Error(`Unknown worker message type: ${String(type)}`);
  } catch (error) {
    self.postMessage({
      type: 'error',
      requestId,
      payload: { message: error instanceof Error ? error.message : String(error) },
    });
  }
});
