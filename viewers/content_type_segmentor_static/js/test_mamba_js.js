import fs from 'fs';
import { MambaSegmentor } from './mamba.js';

const manifest = JSON.parse(fs.readFileSync('../assets/model_manifest.json', 'utf-8'));
const meta = JSON.parse(fs.readFileSync('../assets/sfullfiles4_weights_meta.json', 'utf-8'));
const binData = fs.readFileSync('../assets/sfullfiles4_weights.bin');

// Map binary data to Float32Arrays using the meta offsets
const buffer = new Float32Array(binData.buffer, binData.byteOffset, binData.byteLength / 4);

const weights = {};
for (const [key, info] of Object.entries(meta)) {
  weights[key] = buffer.subarray(info.offset, info.offset + info.length);
}

const segmentor = new MambaSegmentor(manifest, weights);

// Test inference speed
const seq = 1536;
const dummy_tokens = new Int32Array(seq);
for(let i = 0; i < seq; i++) dummy_tokens[i] = i % 256;

console.log("Warming up...");
for(let i=0; i<3; i++) segmentor.predict_window_probs(dummy_tokens);

console.log("Measuring...");
const iters = 10;
const start = Date.now();
for(let i=0; i<iters; i++) {
  segmentor.predict_window_probs(dummy_tokens);
}
const end = Date.now();
console.log(`Average pure JS latency for 1536 tokens: ${(end-start)/iters} ms`);
