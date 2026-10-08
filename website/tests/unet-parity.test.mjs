import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = new URL('../../viewers/content_type_segmentor_static/', import.meta.url);
const context = { module: { exports: {} }, Float32Array, Int32Array, Math };
vm.runInNewContext(readFileSync(new URL('js/unet.js', source), 'utf8'), context);
const { UNetSegmentor } = context.module.exports;
const manifest = JSON.parse(readFileSync(new URL('assets/model_manifest.json', source), 'utf8'));
const meta = JSON.parse(readFileSync(new URL('assets/unet_al_weights_meta.json', source), 'utf8'));
const blob = readFileSync(new URL('assets/unet_al_weights.bin', source));
const weights = Object.fromEntries(Object.entries(meta).map(([key, info]) => [key,
  new Float32Array(blob.buffer, blob.byteOffset + info.offset * 4, info.length),
]));

test('browser U-Net matches the independent NumPy reference on a short padded text input', () => {
  const fixture = JSON.parse(readFileSync(new URL('./unet-reference.json', import.meta.url), 'utf8'));
  const model = new UNetSegmentor(manifest, weights);
  const actual = model.predict_window_probs(Int32Array.from(new TextEncoder().encode(fixture.text)));
  assert.equal(actual.length, fixture.text.length * manifest.num_classes);
  for (let i = 0; i < fixture.positions.length; i++) {
    const offset = fixture.positions[i] * manifest.num_classes;
    for (let c = 0; c < manifest.num_classes; c++) {
      assert.ok(Math.abs(actual[offset + c] - fixture.probabilities[i][c]) < 0.0002,
        `Probability mismatch at position ${fixture.positions[i]}, class ${c}`);
    }
  }
});

test('window merging covers long inputs and produces normalized probabilities at window boundaries', () => {
  const model = new UNetSegmentor(manifest, weights);
  const text = ('SELECT name FROM samples;\n').repeat(67);
  const actual = model.predict_window_probs(Int32Array.from(new TextEncoder().encode(text)));
  assert.equal(actual.length, text.length * manifest.num_classes);
  for (const position of [0, 767, 768, 1535, 1536, text.length - 1]) {
    const row = actual.subarray(position * manifest.num_classes, (position + 1) * manifest.num_classes);
    assert.ok(row.every(Number.isFinite));
    assert.ok(Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 0.00001);
  }
});
