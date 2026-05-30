// Mamba Engine in Pure JS
class MambaSegmentor {
  constructor(manifest, weights) {
    this.manifest = manifest;
    this.weights = weights; 
    this.num_classes = manifest.num_classes;
    const m = manifest.model;
    this.d_model = m.d_model;
    this.n_layers = m.n_layers;
    this.d_state = m.d_state;
    this.expand = m.expand;
    this.d_inner = this.d_model * this.expand;
    this.dt_rank = m.dt_rank;
    this.d_conv = m.d_conv;
    this.bidirectional = m.bidirectional;
    this.layer_norm_eps = 1e-5;
    
    // Unpack weights
    this.embed = weights["embed/embedding"]; // (vocab, d_model)
    this.final_ln_scale = weights["final/ln_scale"]; // (d_model)
    this.final_ln_bias = weights["final/ln_bias"]; // (d_model)
    this.final_dense_kernel = this._transpose(weights["final/dense_kernel"], this.d_model, this.num_classes); // (num_classes, d_model)
    this.final_dense_bias = weights["final/dense_bias"]; // (num_classes)
    
    this.blocks = [];
    for (let i = 0; i < this.n_layers; i++) {
      const p = `blocks/${i}/`;
      const block = {
        ln_scale: weights[p + "ln_scale"],
        ln_bias: weights[p + "ln_bias"],
        in_proj_kernel: this._transpose(weights[p + "in_proj_kernel"], this.d_model, this.d_inner * 2),
        in_proj_bias: weights[p + "in_proj_bias"],
        conv_kernel: weights[p + "conv_kernel"],
        conv_bias: weights[p + "conv_bias"],
        x_proj_kernel: this._transpose(weights[p + "x_proj_kernel"], this.d_inner, this.dt_rank + 2 * this.d_state),
        x_proj_bias: weights[p + "x_proj_bias"],
        dt_proj_kernel: this._transpose(weights[p + "dt_proj_kernel"], this.dt_rank, this.d_inner),
        dt_proj_bias: weights[p + "dt_proj_bias"],
        out_proj_kernel: this._transpose(weights[p + "out_proj_kernel"], this.d_inner, this.d_model),
        out_proj_bias: weights[p + "out_proj_bias"],
        a_log: weights[p + "a_log"],
        d: weights[p + "d"]
      };
      
      const a = new Float32Array(block.a_log.length);
      for(let j=0; j<a.length; j++) a[j] = -Math.exp(block.a_log[j]);
      block.a = a;
      this.blocks.push(block);
    }
  }

  _layer_norm(x, seq, dim, scale, bias, out) {
    for (let i = 0; i < seq; i++) {
      let sum = 0;
      for (let j = 0; j < dim; j++) sum += x[i * dim + j];
      const mean = sum / dim;
      let varSum = 0;
      for (let j = 0; j < dim; j++) {
        const diff = x[i * dim + j] - mean;
        varSum += diff * diff;
      }
      const variance = varSum / dim;
      const inv_std = 1.0 / Math.sqrt(variance + this.layer_norm_eps);
      for (let j = 0; j < dim; j++) {
        out[i * dim + j] = (x[i * dim + j] - mean) * inv_std * scale[j] + bias[j];
      }
    }
  }

  _transpose(matrix, rows, cols) {
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        out[c * rows + r] = matrix[r * cols + c];
      }
    }
    return out;
  }

  _dense(x, seq, in_dim, out_dim, kernel_T, bias, out) {
    for (let i = 0; i < seq; i++) {
      const out_offset = i * out_dim;
      for (let o = 0; o < out_dim; o++) {
        out[out_offset + o] = bias[o];
      }
    }
    const BLOCK = 64;
    for (let i0 = 0; i0 < seq; i0 += BLOCK) {
      const i_max = Math.min(i0 + BLOCK, seq);
      for (let o0 = 0; o0 < out_dim; o0 += BLOCK) {
        const o_max = Math.min(o0 + BLOCK, out_dim);
        for (let j0 = 0; j0 < in_dim; j0 += BLOCK) {
          const j_max = Math.min(j0 + BLOCK, in_dim);
          
          for (let i = i0; i < i_max; i++) {
            const x_offset = i * in_dim;
            const out_offset = i * out_dim;
            for (let j = j0; j < j_max; j++) {
              const x_val = x[x_offset + j];
              for (let o = o0; o < o_max; o++) {
                out[out_offset + o] += x_val * kernel_T[o * in_dim + j];
              }
            }
          }
        }
      }
    }
  }

  _silu(x, out) {
    for (let i = 0; i < x.length; i++) {
      out[i] = x[i] / (1.0 + Math.exp(-x[i]));
    }
  }

  _softplus(x, out) {
    for (let i = 0; i < x.length; i++) {
      const abs_x = Math.abs(x[i]);
      out[i] = Math.max(x[i], 0.0) + Math.log1p(Math.exp(-abs_x)) + 1e-4; 
    }
  }

  _depthwise_conv_same(x, seq, dim, kernel, bias, out) {
    const k = kernel.length / dim;
    const pad_left = Math.floor((k - 1) / 2);
    for (let i = 0; i < seq; i++) {
      for (let d = 0; d < dim; d++) {
        let sum = bias[d];
        for (let kj = 0; kj < k; kj++) {
          const in_idx = i - pad_left + kj;
          if (in_idx >= 0 && in_idx < seq) {
            sum += x[in_idx * dim + d] * kernel[kj * dim + d];
          }
        }
        out[i * dim + d] = sum;
      }
    }
  }

  _selective_scan(u, dt, b_in, c_in, a, d, seq, out, state) {
    const d_inner = this.d_inner;
    const d_state = this.d_state;
    
    for(let i=0; i<state.length; i++) state[i] = 0;
    
    for (let i = 0; i < seq; i++) {
      const u_offset = i * d_inner;
      const dt_offset = i * d_inner;
      const b_offset = i * d_state;
      const c_offset = i * d_state;
      
      for (let j = 0; j < d_inner; j++) {
        const u_val = u[u_offset + j];
        const dt_val = dt[dt_offset + j];
        const d_val = d[j];
        
        let sum = 0;
        for (let k = 0; k < d_state; k++) {
          const a_val = a[j * d_state + k];
          const b_val = b_in[b_offset + k];
          const c_val = c_in[c_offset + k];
          const a_t = Math.exp(dt_val * a_val);
          
          const s_idx = j * d_state + k;
          state[s_idx] = a_t * state[s_idx] + u_val * dt_val * b_val;
          sum += state[s_idx] * c_val;
        }
        out[u_offset + j] = sum + u_val * d_val;
      }
    }
  }

  _selective_scan_rev(u, dt, b_in, c_in, a, d, seq, out, state) {
    const d_inner = this.d_inner;
    const d_state = this.d_state;
    
    for(let i=0; i<state.length; i++) state[i] = 0;
    
    for (let i = seq - 1; i >= 0; i--) {
      const u_offset = i * d_inner;
      const dt_offset = i * d_inner;
      const b_offset = i * d_state;
      const c_offset = i * d_state;
      
      for (let j = 0; j < d_inner; j++) {
        const u_val = u[u_offset + j];
        const dt_val = dt[dt_offset + j];
        const d_val = d[j];
        
        let sum = 0;
        for (let k = 0; k < d_state; k++) {
          const a_val = a[j * d_state + k];
          const b_val = b_in[b_offset + k];
          const c_val = c_in[c_offset + k];
          const a_t = Math.exp(dt_val * a_val);
          
          const s_idx = j * d_state + k;
          state[s_idx] = a_t * state[s_idx] + u_val * dt_val * b_val;
          sum += state[s_idx] * c_val;
        }
        out[u_offset + j] = sum + u_val * d_val;
      }
    }
  }

  async predict_window_probs(tokens, gpu) {
    const seq = tokens.length;
    if (seq === 0) return new Float32Array(0);
    
    if (!this._h || this._h.length < seq * this.d_model) {
      this._h = new Float32Array(seq * this.d_model);
      this._h_norm = new Float32Array(seq * this.d_model);
      this._xz = new Float32Array(seq * this.d_inner * 2);
      this._u = new Float32Array(seq * this.d_inner);
      this._gate = new Float32Array(seq * this.d_inner);
      this._u_conv = new Float32Array(seq * this.d_inner);
      this._u_silu = new Float32Array(seq * this.d_inner);
      this._x_dbl = new Float32Array(seq * (this.dt_rank + 2 * this.d_state));
      this._dt_raw = new Float32Array(seq * this.dt_rank);
      this._b_in = new Float32Array(seq * this.d_state);
      this._c_in = new Float32Array(seq * this.d_state);
      this._dt_proj = new Float32Array(seq * this.d_inner);
      this._dt = new Float32Array(seq * this.d_inner);
      this._y = new Float32Array(seq * this.d_inner);
      this._y_rev = new Float32Array(seq * this.d_inner);
      this._y_out = new Float32Array(seq * this.d_model);
      this._state = new Float32Array(this.d_inner * this.d_state);
      this._gate_silu = new Float32Array(seq * this.d_inner);
      this._h_final = new Float32Array(seq * this.d_model);
      this._logits = new Float32Array(seq * this.num_classes);
    }
    
    let h = this._h;
    for (let i = 0; i < seq; i++) {
      const tok = tokens[i];
      for (let j = 0; j < this.d_model; j++) {
        h[i * this.d_model + j] = this.embed[tok * this.d_model + j];
      }
    }
    
    for (const block of this.blocks) {
      this._layer_norm(h, seq, this.d_model, block.ln_scale, block.ln_bias, this._h_norm);
      await gpu.dense(this._h_norm, block.in_proj_kernel, block.in_proj_bias, seq, this.d_model, this.d_inner * 2, this._xz);
      
      const xz = this._xz;
      const u = this._u;
      const gate = this._gate;
      for (let i = 0; i < seq; i++) {
        for (let j = 0; j < this.d_inner; j++) {
          u[i * this.d_inner + j] = xz[i * (this.d_inner * 2) + j];
          gate[i * this.d_inner + j] = xz[i * (this.d_inner * 2) + this.d_inner + j];
        }
      }
      
      this._depthwise_conv_same(u, seq, this.d_inner, block.conv_kernel, block.conv_bias, this._u_conv);
      this._silu(this._u_conv, this._u_silu);
      
      await gpu.dense(this._u_silu, block.x_proj_kernel, block.x_proj_bias, seq, this.d_inner, this.dt_rank + 2 * this.d_state, this._x_dbl);
      
      const x_dbl = this._x_dbl;
      const dt_raw = this._dt_raw;
      const b_in = this._b_in;
      const c_in = this._c_in;
      for (let i = 0; i < seq; i++) {
        for (let j = 0; j < this.dt_rank; j++) dt_raw[i * this.dt_rank + j] = x_dbl[i * (this.dt_rank + 2 * this.d_state) + j];
        for (let j = 0; j < this.d_state; j++) {
          b_in[i * this.d_state + j] = x_dbl[i * (this.dt_rank + 2 * this.d_state) + this.dt_rank + j];
          c_in[i * this.d_state + j] = x_dbl[i * (this.dt_rank + 2 * this.d_state) + this.dt_rank + this.d_state + j];
        }
      }
      
      await gpu.dense(dt_raw, block.dt_proj_kernel, block.dt_proj_bias, seq, this.dt_rank, this.d_inner, this._dt_proj);
      this._softplus(this._dt_proj, this._dt);
      
      if (gpu && gpu.ssm) {
        await gpu.ssm(this._u_silu, this._dt, b_in, c_in, block.a, block.d, seq, this.d_inner, this.d_state, this._y, false);
      } else {
        this._selective_scan(this._u_silu, this._dt, b_in, c_in, block.a, block.d, seq, this._y, this._state);
      }
      
      if (this.bidirectional) {
        if (gpu && gpu.ssm) {
          await gpu.ssm(this._u_silu, this._dt, b_in, c_in, block.a, block.d, seq, this.d_inner, this.d_state, this._y_rev, true);
        } else {
          this._selective_scan_rev(this._u_silu, this._dt, b_in, c_in, block.a, block.d, seq, this._y_rev, this._state);
        }
        for (let i = 0; i < this._y.length; i++) this._y[i] += this._y_rev[i];
      }
      
      this._silu(gate, this._gate_silu);
      for (let i = 0; i < this._y.length; i++) this._y[i] *= this._gate_silu[i];
      
      await gpu.dense(this._y, block.out_proj_kernel, block.out_proj_bias, seq, this.d_inner, this.d_model, this._y_out);
      for (let i = 0; i < h.length; i++) h[i] += this._y_out[i];
    }
    
    this._layer_norm(h, seq, this.d_model, this.final_ln_scale, this.final_ln_bias, this._h_final);
    await gpu.dense(this._h_final, this.final_dense_kernel, this.final_dense_bias, seq, this.d_model, this.num_classes, this._logits);
    
    const probs = new Float32Array(seq * this.num_classes);
    for (let i = 0; i < seq; i++) {
      let max_val = -Infinity;
      for (let j = 0; j < this.num_classes; j++) {
        if (this._logits[i * this.num_classes + j] > max_val) max_val = this._logits[i * this.num_classes + j];
      }
      let sum = 0;
      for (let j = 0; j < this.num_classes; j++) {
        const e = Math.exp(this._logits[i * this.num_classes + j] - max_val);
        probs[i * this.num_classes + j] = e;
        sum += e;
      }
      const denom = Math.max(sum, 1e-9);
      for (let j = 0; j < this.num_classes; j++) {
        probs[i * this.num_classes + j] /= denom;
      }
    }
    return probs;
  }
}
