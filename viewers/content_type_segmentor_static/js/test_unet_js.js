// Smoke test for UNet1D JS implementation
// Verifies architecture shape consistency with random weights

const { UNetSegmentor } = require('./unet.js');

// Build a mock weightsDict with correct shapes
function makeWeight(shape) {
  let size = 1;
  for (const s of shape) size *= s;
  const arr = new Float32Array(size);
  for (let i = 0; i < size; i++) arr[i] = (Math.random() - 0.5) * 0.01;
  return arr;
}

const channels = [32, 64, 64, 128, 128, 128, 128, 256];
const emb_dim = 256;
const num_classes = 35;

// Build the expected block configs (in_ch, out_ch) for each ConvBlock1D
const blockConfigs = [];

// Down path
let curCh = emb_dim;
for (const ch of channels) {
  blockConfigs.push([curCh, ch]); // first block
  blockConfigs.push([ch, ch]);    // second block
  curCh = ch;
}

// Up path
const revCh = channels.slice(0, -1).reverse();
curCh = channels[channels.length - 1]; // 256 after bottleneck
for (let i = 0; i < revCh.length; i++) {
  const ch = revCh[i];
  const skipIdx = channels.length - 2 - i;
  const skipCh = channels[skipIdx];
  const concatCh = curCh + skipCh;
  blockConfigs.push([concatCh, ch]); // first up block
  blockConfigs.push([ch, ch]);       // second up block
  curCh = ch;
}

console.log(`Total blocks: ${blockConfigs.length}`);
console.log('Block configs (in_ch → out_ch):');
for (let i = 0; i < blockConfigs.length; i++) {
  const [ic, oc] = blockConfigs[i];
  console.log(`  ConvBlock1D_${i}: kernel=(3, ${ic}, ${oc})`);
}

// Build weightsDict
const weightsDict = {};
weightsDict['Embed_0/embedding'] = makeWeight([257, emb_dim]);

for (let i = 0; i < blockConfigs.length; i++) {
  const [ic, oc] = blockConfigs[i];
  const prefix = `ConvBlock1D_${i}`;
  weightsDict[`${prefix}/Conv_0/kernel`] = makeWeight([3, ic, oc]);
  weightsDict[`${prefix}/Conv_0/bias`] = makeWeight([oc]);
  weightsDict[`${prefix}/GroupNorm_0/scale`] = makeWeight([oc]);
  weightsDict[`${prefix}/GroupNorm_0/bias`] = makeWeight([oc]);
  // Set scale to 1.0 for better test stability
  const scale = weightsDict[`${prefix}/GroupNorm_0/scale`];
  for (let j = 0; j < scale.length; j++) scale[j] = 1.0;
}

weightsDict['Conv_0/kernel'] = makeWeight([1, 32, num_classes]);
weightsDict['Conv_0/bias'] = makeWeight([num_classes]);

const manifest = { num_classes };
const model = new UNetSegmentor(manifest, weightsDict);

// Test with seq=128 (must be divisible by 2^7=128 for 7 maxpools)
const seq = 128;
const inputArr = new Int32Array(seq);
for (let i = 0; i < seq; i++) inputArr[i] = Math.floor(Math.random() * 256);

console.log(`\nRunning forward pass with seq=${seq}...`);
const t0 = performance.now();
const probs = model.predict_window_probs(inputArr);
const elapsed = performance.now() - t0;

console.log(`Output length: ${probs.length} (expected ${seq * num_classes})`);
console.log(`Elapsed: ${elapsed.toFixed(1)} ms`);

// Verify probabilities sum to ~1 for each position
let maxErr = 0;
for (let i = 0; i < seq; i++) {
  let sum = 0;
  for (let j = 0; j < num_classes; j++) sum += probs[i * num_classes + j];
  maxErr = Math.max(maxErr, Math.abs(sum - 1.0));
}
console.log(`Max softmax sum error: ${maxErr.toExponential(3)} (should be ~0)`);

// Verify no NaNs
let nanCount = 0;
for (let i = 0; i < probs.length; i++) {
  if (isNaN(probs[i])) nanCount++;
}
console.log(`NaN count: ${nanCount} (should be 0)`);

// Larger test
const seq2 = 512;
const inputArr2 = new Int32Array(seq2);
for (let i = 0; i < seq2; i++) inputArr2[i] = Math.floor(Math.random() * 256);
console.log(`\nRunning forward pass with seq=${seq2}...`);
const t1 = performance.now();
const probs2 = model.predict_window_probs(inputArr2);
const elapsed2 = performance.now() - t1;
console.log(`Output length: ${probs2.length} (expected ${seq2 * num_classes})`);
console.log(`Elapsed: ${elapsed2.toFixed(1)} ms`);

let nanCount2 = 0;
for (let i = 0; i < probs2.length; i++) {
  if (isNaN(probs2[i])) nanCount2++;
}
console.log(`NaN count: ${nanCount2} (should be 0)`);

let maxErr2 = 0;
for (let i = 0; i < seq2; i++) {
  let sum = 0;
  for (let j = 0; j < num_classes; j++) sum += probs2[i * num_classes + j];
  maxErr2 = Math.max(maxErr2, Math.abs(sum - 1.0));
}
console.log(`Max softmax sum error: ${maxErr2.toExponential(3)}`);

if (nanCount === 0 && nanCount2 === 0 && maxErr < 1e-5 && maxErr2 < 1e-5 &&
    probs.length === seq * num_classes && probs2.length === seq2 * num_classes) {
  console.log('\n✅ All checks passed!');
} else {
  console.log('\n❌ Some checks failed!');
  process.exit(1);
}
