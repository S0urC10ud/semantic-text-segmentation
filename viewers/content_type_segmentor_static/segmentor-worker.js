const STATIC_BASE = '/static/';
const DEFAULT_PYODIDE_JS_URL = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js';
const DEFAULT_PYODIDE_INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';

const STATE = {
  bootPromise: null,
  pyodide: null,
  manifest: null,
  runtime: 'pyodide-wasm',
};

async function fetchOrThrow(url, as){
  const response = await fetch(url);
  if (!response.ok){
    throw new Error(`Failed to fetch ${url} (${response.status}).`);
  }
  if (as === 'arrayBuffer'){
    return response.arrayBuffer();
  }
  return response.text();
}

async function ensureLoaded(){
  if (STATE.bootPromise){
    return STATE.bootPromise;
  }

  STATE.bootPromise = (async () => {
    const manifestText = await fetchOrThrow(`${STATIC_BASE}assets/model_manifest.json`, 'text');
    const manifest = JSON.parse(manifestText);
    const pyodideJsUrl = manifest.pyodide_js_url || DEFAULT_PYODIDE_JS_URL;
    const pyodideIndexUrl = manifest.pyodide_index_url || DEFAULT_PYODIDE_INDEX_URL;

    importScripts(pyodideJsUrl);

    const [runtimeSource, weightsBuffer] = await Promise.all([
      fetchOrThrow(`${STATIC_BASE}py/runtime.py`, 'text'),
      fetchOrThrow(`${STATIC_BASE}assets/sfullfiles4_weights.npz`, 'arrayBuffer'),
    ]);

    const pyodide = await loadPyodide({ indexURL: pyodideIndexUrl });
    await pyodide.loadPackage(['numpy']);
    pyodide.FS.mkdirTree('/app');
    pyodide.FS.writeFile('/app/model_manifest.json', manifestText);
    pyodide.FS.writeFile('/app/model_weights.npz', new Uint8Array(weightsBuffer));
    await pyodide.runPythonAsync(runtimeSource);

    const loadResult = await pyodide.runPythonAsync(
      'load_model_json("/app/model_manifest.json", "/app/model_weights.npz")'
    );

    STATE.pyodide = pyodide;
    STATE.manifest = manifest;
    STATE.runtime = 'pyodide-wasm';

    return {
      ...JSON.parse(loadResult),
      runtime: STATE.runtime,
    };
  })().catch(error => {
    STATE.bootPromise = null;
    throw error;
  });

  return STATE.bootPromise;
}

async function runSegment(text, threshold){
  await ensureLoaded();
  STATE.pyodide.globals.set('segmentor_input_text', String(text || ''));
  const nextThreshold = Number.isFinite(Number(threshold))
    ? Number(threshold)
    : Number(STATE.manifest?.other_threshold ?? 0.3);
  STATE.pyodide.globals.set('segmentor_threshold', nextThreshold);
  const started = performance.now();
  const raw = await STATE.pyodide.runPythonAsync(
    'segment_text_json(segmentor_input_text, top_k=5, threshold=segmentor_threshold)'
  );
  const payload = JSON.parse(raw);
  payload.runtime = STATE.runtime;
  payload.elapsed_ms = Number((performance.now() - started).toFixed(1));
  payload.model_id = STATE.manifest?.model_id || 'sfullfiles4';
  return payload;
}

self.addEventListener('message', async event => {
  const { type, requestId, payload } = event.data || {};
  try{
    if (type === 'load'){
      const result = await ensureLoaded();
      self.postMessage({ type: 'result', requestId, payload: result });
      return;
    }
    if (type === 'segment'){
      const result = await runSegment(payload?.text || '', payload?.threshold);
      self.postMessage({ type: 'result', requestId, payload: result });
      return;
    }
    throw new Error(`Unknown worker message type: ${String(type)}`);
  }catch(error){
    self.postMessage({
      type: 'error',
      requestId,
      payload: {
        message: error instanceof Error ? error.message : String(error),
      },
    });
  }
});
