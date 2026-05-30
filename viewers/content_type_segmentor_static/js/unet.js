// ──────────────────────────────────────────────────────────────────
// UNet1D Segmentor — pure JavaScript, no dependencies
// Architecture: channels = (32, 64, 64, 128, 128, 128, 128, 256)
//   Down: 8 levels × 2 ConvBlock1D each, maxpool(2) between 0..6
//   Up:   7 levels × 2 ConvBlock1D each, upsample(2) + concat skip
//   Final: 1×1 Conv → num_classes logits → softmax
// ──────────────────────────────────────────────────────────────────

class UNetSegmentor {
  /**
   * @param {Object} manifest - model manifest (must have .num_classes)
   * @param {Object<string, Float32Array>} weightsDict - flat weight dict
   */
  constructor(manifest, weightsDict) {
    this.manifest = manifest;
    this.weights = weightsDict;
    this.num_classes = manifest.num_classes;
    this.channels = [32, 64, 64, 128, 128, 128, 128, 256];
    this.num_groups = 8;
    this.gn_eps = 1e-5;

    // Embedding: (257, 256)
    this.embed = weightsDict['Embed_0/embedding'];
    this.emb_dim = 256; // inferred from (257, 256)

    // Parse conv-block weights  (30 blocks total: 16 down + 14 up)
    const numDownBlocks = this.channels.length * 2; // 16
    const revCh = this.channels.slice(0, -1).reverse(); // [128,128,128,128,64,64,32]
    const numUpBlocks = revCh.length * 2; // 14
    this.numBlocks = numDownBlocks + numUpBlocks; // 30

    this.blocks = new Array(this.numBlocks);
    for (let i = 0; i < this.numBlocks; i++) {
      const prefix = `ConvBlock1D_${i}`;
      this.blocks[i] = {
        conv_kernel: weightsDict[`${prefix}/Conv_0/kernel`],   // (3, in_ch, out_ch)
        conv_bias:   weightsDict[`${prefix}/Conv_0/bias`],     // (out_ch,)
        gn_scale:    weightsDict[`${prefix}/GroupNorm_0/scale`],// (out_ch,)
        gn_bias:     weightsDict[`${prefix}/GroupNorm_0/bias`], // (out_ch,)
      };
    }

    // Final 1×1 conv: (1, 32, num_classes)
    this.final_kernel = weightsDict['Conv_0/kernel'];
    this.final_bias   = weightsDict['Conv_0/bias'];

    // Pre-compute channel layout for down/up paths
    this._downLevels = this.channels.length;        // 8
    this._upLevels   = this.channels.length - 1;    // 7
    this._revChannels = revCh;

    // Scratch buffers — lazily allocated on first call / when seq grows
    this._allocSeq = 0;
  }

  // ───────────── buffer management ─────────────

  /**
   * Ensure scratch buffers are large enough for `seq`.
   * We allocate once per max-seq seen and reuse thereafter.
   */
  _ensureBuffers(seq) {
    if (seq <= this._allocSeq) return;
    this._allocSeq = seq;

    // We need buffers for the down path at every resolution.
    // Max channels across all levels = 256.  Max seq = seq.
    const maxCh = 256;
    // Two general-purpose activation buffers sized for (seq, maxCh)
    this._buf1 = new Float32Array(seq * maxCh);
    this._buf2 = new Float32Array(seq * maxCh);
    this._buf3 = new Float32Array(seq * maxCh);  // conv scratch
    // For skip connections we will store Float32Array snapshots
    // (allocated per-call because sizes depend on seq at each level)

    // Logits buffer
    this._logits = new Float32Array(seq * this.num_classes);
  }

  // ───────────── core ops ─────────────

  /**
   * Conv1D, kernel_size=k, padding=SAME.
   * kernel layout: (k, in_ch, out_ch)  — matches Flax Conv default.
   * @param {Float32Array} inp  - (seq, in_ch)
   * @param {number} seq
   * @param {number} in_ch
   * @param {Float32Array} kernel - (k, in_ch, out_ch)
   * @param {Float32Array} bias   - (out_ch,)
   * @param {number} out_ch
   * @param {Float32Array} out    - (seq, out_ch)
   * @param {number} k            - kernel size
   */
  _conv1d(inp, seq, in_ch, kernel, bias, out_ch, out, k) {
    const pad_left = (k - 1) >> 1;
    // Zero-fill output and add bias
    for (let i = 0; i < seq; i++) {
      const oBase = i * out_ch;
      for (let o = 0; o < out_ch; o++) {
        out[oBase + o] = bias[o];
      }
    }
    // Accumulate convolution
    for (let kk = 0; kk < k; kk++) {
      const kernelSlice = kk * in_ch * out_ch; // offset into kernel for this tap
      for (let i = 0; i < seq; i++) {
        const in_pos = i - pad_left + kk;
        if (in_pos < 0 || in_pos >= seq) continue;
        const iBase = in_pos * in_ch;
        const oBase = i * out_ch;
        for (let ic = 0; ic < in_ch; ic++) {
          const x_val = inp[iBase + ic];
          const kBase = kernelSlice + ic * out_ch;
          for (let o = 0; o < out_ch; o++) {
            out[oBase + o] += x_val * kernel[kBase + o];
          }
        }
      }
    }
  }

  /**
   * GroupNorm (in-place friendly: out may alias inp).
   * @param {Float32Array} inp - (seq, ch)
   * @param {number} seq
   * @param {number} ch
   * @param {number} num_groups
   * @param {Float32Array} scale - (ch,)
   * @param {Float32Array} bias  - (ch,)
   * @param {Float32Array} out   - (seq, ch)
   */
  _group_norm(inp, seq, ch, num_groups, scale, bias, out) {
    const group_size = ch / num_groups;
    for (let i = 0; i < seq; i++) {
      const base = i * ch;
      for (let g = 0; g < num_groups; g++) {
        const gStart = g * group_size;
        // Compute mean
        let sum = 0;
        for (let j = 0; j < group_size; j++) {
          sum += inp[base + gStart + j];
        }
        const mean = sum / group_size;
        // Compute variance
        let varSum = 0;
        for (let j = 0; j < group_size; j++) {
          const d = inp[base + gStart + j] - mean;
          varSum += d * d;
        }
        const invStd = 1.0 / Math.sqrt(varSum / group_size + this.gn_eps);
        // Normalize, scale, bias
        for (let j = 0; j < group_size; j++) {
          const idx = base + gStart + j;
          const chIdx = gStart + j;
          out[idx] = (inp[idx] - mean) * invStd * scale[chIdx] + bias[chIdx];
        }
      }
    }
  }

  /**
   * GELU activation (in-place OK).
   * Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
   */
  _gelu(arr, len) {
    const c = 0.7978845608028654; // sqrt(2/pi)
    for (let i = 0; i < len; i++) {
      const x = arr[i];
      const x3 = x * x * x;
      arr[i] = 0.5 * x * (1.0 + Math.tanh(c * (x + 0.044715 * x3)));
    }
  }

  /**
   * ConvBlock1D forward: Conv1D → GroupNorm → GELU
   * @param {Object} block - {conv_kernel, conv_bias, gn_scale, gn_bias}
   * @param {Float32Array} inp - (seq, in_ch)
   * @param {number} seq
   * @param {number} in_ch
   * @param {number} out_ch
   * @param {Float32Array} out - (seq, out_ch)
   * @param {Float32Array} scratch - temp buffer >= seq * out_ch
   */
  _convBlock(block, inp, seq, in_ch, out_ch, out, scratch) {
    // Conv1D k=3 padding=SAME
    this._conv1d(inp, seq, in_ch, block.conv_kernel, block.conv_bias, out_ch, scratch, 3);
    // GroupNorm
    this._group_norm(scratch, seq, out_ch, this.num_groups, block.gn_scale, block.gn_bias, out);
    // GELU in-place
    this._gelu(out, seq * out_ch);
  }

  /**
   * MaxPool1D: window=2, stride=2.
   * @param {Float32Array} inp - (seq, ch)
   * @param {number} seq
   * @param {number} ch
   * @param {Float32Array} out - (floor(seq/2), ch)
   * @returns {number} output seq length
   */
  _maxpool(inp, seq, ch, out) {
    const outSeq = seq >> 1;
    for (let i = 0; i < outSeq; i++) {
      const i0 = (i * 2) * ch;
      const i1 = (i * 2 + 1) * ch;
      const oBase = i * ch;
      for (let c = 0; c < ch; c++) {
        out[oBase + c] = inp[i0 + c] > inp[i1 + c] ? inp[i0 + c] : inp[i1 + c];
      }
    }
    return outSeq;
  }

  /**
   * Nearest-neighbour upsample ×2.
   * @param {Float32Array} inp - (seq, ch)
   * @param {number} seq
   * @param {number} ch
   * @param {Float32Array} out - (seq*2, ch)
   * @returns {number} output seq length
   */
  _upsample(inp, seq, ch, out) {
    const outSeq = seq * 2;
    for (let i = seq - 1; i >= 0; i--) {
      const iBase = i * ch;
      const o1 = (i * 2) * ch;
      const o2 = (i * 2 + 1) * ch;
      for (let c = 0; c < ch; c++) {
        const v = inp[iBase + c];
        out[o1 + c] = v;
        out[o2 + c] = v;
      }
    }
    return outSeq;
  }

  /**
   * Concatenate along channel axis: [a, b] → out.
   * a: (seq, ch_a),  b: (seq, ch_b),  out: (seq, ch_a + ch_b)
   */
  _concat(a, b, seq, ch_a, ch_b, out) {
    const ch_out = ch_a + ch_b;
    for (let i = 0; i < seq; i++) {
      const aBase = i * ch_a;
      const bBase = i * ch_b;
      const oBase = i * ch_out;
      for (let c = 0; c < ch_a; c++) out[oBase + c] = a[aBase + c];
      for (let c = 0; c < ch_b; c++) out[oBase + ch_a + c] = b[bBase + c];
    }
  }

  /**
   * Softmax over last axis (in-place into `out`).
   * logits: (seq, C), out: (seq, C)
   */
  _softmax(logits, seq, C, out) {
    for (let i = 0; i < seq; i++) {
      const base = i * C;
      let maxVal = -Infinity;
      for (let j = 0; j < C; j++) {
        if (logits[base + j] > maxVal) maxVal = logits[base + j];
      }
      let sum = 0;
      for (let j = 0; j < C; j++) {
        const e = Math.exp(logits[base + j] - maxVal);
        out[base + j] = e;
        sum += e;
      }
      const inv = 1.0 / Math.max(sum, 1e-9);
      for (let j = 0; j < C; j++) {
        out[base + j] *= inv;
      }
    }
  }

  // ───────────── main forward pass ─────────────

  /**
   * Run the U-Net on a window of byte tokens.
   * @param {Int32Array} inputArr - token ids (byte values 0-255, 256=PAD)
   * @returns {Float32Array} probabilities, flat (seq × num_classes)
   */
  predict_window_probs(inputArr) {
    const seq = inputArr.length;
    if (seq === 0) return new Float32Array(0);

    this._ensureBuffers(seq);

    const channels = this.channels;
    const numDown  = channels.length;      // 8
    const emb_dim  = this.emb_dim;         // 256

    // ── Embedding lookup ──
    // h: (seq, emb_dim)
    let h = new Float32Array(seq * emb_dim);
    const emb = this.embed;
    for (let i = 0; i < seq; i++) {
      const tok = inputArr[i];
      const src = tok * emb_dim;
      const dst = i * emb_dim;
      for (let j = 0; j < emb_dim; j++) {
        h[dst + j] = emb[src + j];
      }
    }

    // ── Down path ──
    // Track (seq_at_level, ch_at_level, data) for skips
    const skips = new Array(numDown);
    let curSeq = seq;
    let curCh  = emb_dim;
    let blockIdx = 0;

    for (let level = 0; level < numDown; level++) {
      const ch = channels[level];

      // First ConvBlock
      const out1 = new Float32Array(curSeq * ch);
      const scratch1 = new Float32Array(curSeq * ch);
      this._convBlock(this.blocks[blockIdx++], h, curSeq, curCh, ch, out1, scratch1);

      // Second ConvBlock
      const out2 = new Float32Array(curSeq * ch);
      const scratch2 = new Float32Array(curSeq * ch);
      this._convBlock(this.blocks[blockIdx++], out1, curSeq, ch, ch, out2, scratch2);

      // Save skip connection (copy since we'll mutate h)
      skips[level] = { data: out2, seq: curSeq, ch: ch };

      if (level < numDown - 1) {
        // MaxPool
        const pooledSeq = curSeq >> 1;
        const pooled = new Float32Array(pooledSeq * ch);
        this._maxpool(out2, curSeq, ch, pooled);
        h = pooled;
        curSeq = pooledSeq;
      } else {
        h = out2;
      }
      curCh = ch;
    }

    // ── Up path ──
    const revCh = this._revChannels; // [128, 128, 128, 128, 64, 64, 32]

    for (let i = 0; i < revCh.length; i++) {
      const ch = revCh[i];

      // Upsample
      const upSeq = curSeq * 2;
      const upBuf = new Float32Array(upSeq * curCh);
      this._upsample(h, curSeq, curCh, upBuf);

      // Get skip from down path: skip index = numDown - 2 - i
      const skipIdx = numDown - 2 - i;
      const skip = skips[skipIdx];

      // Handle potential length mismatch (pad upsampled if shorter)
      let upData = upBuf;
      let upLen  = upSeq;
      if (upSeq < skip.seq) {
        // Pad with zeros at the end
        const padded = new Float32Array(skip.seq * curCh);
        padded.set(upBuf.subarray(0, upSeq * curCh));
        upData = padded;
        upLen  = skip.seq;
      } else if (upSeq > skip.seq) {
        // Truncate (shouldn't normally happen with power-of-2, but be safe)
        upLen = skip.seq;
      }

      // Concat [upsampled, skip] along channel axis
      const concatCh = curCh + skip.ch;
      const concatBuf = new Float32Array(upLen * concatCh);
      this._concat(upData, skip.data, upLen, curCh, skip.ch, concatBuf);

      // Two ConvBlocks
      const out1 = new Float32Array(upLen * ch);
      const scratch1 = new Float32Array(upLen * ch);
      this._convBlock(this.blocks[blockIdx++], concatBuf, upLen, concatCh, ch, out1, scratch1);

      const out2 = new Float32Array(upLen * ch);
      const scratch2 = new Float32Array(upLen * ch);
      this._convBlock(this.blocks[blockIdx++], out1, upLen, ch, ch, out2, scratch2);

      h = out2;
      curSeq = upLen;
      curCh  = ch;
    }

    // ── Final 1×1 Conv: (1, curCh, num_classes) ──
    const C = this.num_classes;
    const logits = new Float32Array(curSeq * C);
    this._conv1d(h, curSeq, curCh, this.final_kernel, this.final_bias, C, logits, 1);

    // ── Softmax ──
    const probs = new Float32Array(curSeq * C);
    this._softmax(logits, curSeq, C, probs);

    return probs;
  }
}

// Export for both Web Workers (importScripts) and Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { UNetSegmentor };
}
