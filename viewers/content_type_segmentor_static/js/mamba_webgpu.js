export class MambaWebGPU {
  constructor(manifest, weightsData) {
    this.manifest = manifest;
    this.weightsData = weightsData; // Float32Array of all weights
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    this.adapter = await navigator.gpu.requestAdapter();
    this.device = await this.adapter.requestDevice();

    // Create shader module for Dense
    const shader = `
      struct Matrix {
        size: vec2<u32>,
        numbers: array<f32>,
      }
      
      @group(0) @binding(0) var<storage, read> x: array<f32>;
      @group(0) @binding(1) var<storage, read> w: array<f32>;
      @group(0) @binding(2) var<storage, read> bias: array<f32>;
      @group(0) @binding(3) var<storage, read_write> out: array<f32>;
      
      struct Uniforms {
        seq: u32,
        in_dim: u32,
        out_dim: u32,
      }
      @group(0) @binding(4) var<uniform> uniforms: Uniforms;

      @compute @workgroup_size(16, 16)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row = global_id.x;
        let col = global_id.y;
        
        if (row >= uniforms.seq || col >= uniforms.out_dim) {
          return;
        }
        
        var sum = bias[col];
        for (var k = 0u; k < uniforms.in_dim; k = k + 1u) {
          sum = sum + x[row * uniforms.in_dim + k] * w[k * uniforms.out_dim + col];
        }
        
        out[row * uniforms.out_dim + col] = sum;
      }
    `;
    this.module = this.device.createShaderModule({ code: shader });
    
    // ... I will need to finish this if we decide to go this route
  }
}
