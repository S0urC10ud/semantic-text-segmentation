class WebGPUMatMul {
  constructor() {
    this.device = null;
    this.pipeline = null;
    this.weightCache = new WeakMap();
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter");
    this.device = await adapter.requestDevice();

    const shaderCode = `
      @group(0) @binding(0) var<storage, read> x: array<f32>;
      @group(0) @binding(1) var<storage, read> w: array<f32>;
      @group(0) @binding(2) var<storage, read> b: array<f32>;
      @group(0) @binding(3) var<storage, read_write> y: array<f32>;
      
      struct Uniforms {
        M: u32,
        K: u32,
        N: u32,
      };
      @group(0) @binding(4) var<uniform> uniforms: Uniforms;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let idx = global_id.x;
        let total = uniforms.M * uniforms.N;
        if (idx >= total) { return; }
        
        let row = idx / uniforms.N;
        let col = idx % uniforms.N;
        
        var sum = b[col];
        for (var k = 0u; k < uniforms.K; k = k + 1u) {
          sum = sum + x[row * uniforms.K + k] * w[col * uniforms.K + k]; // Transposed!
        }
        y[idx] = sum;
      }
    `;

    const module = this.device.createShaderModule({ code: shaderCode });
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });

    const ssmShaderCode = `
      @group(0) @binding(0) var<storage, read> u: array<f32>;
      @group(0) @binding(1) var<storage, read> dt: array<f32>;
      @group(0) @binding(2) var<storage, read> b_in: array<f32>;
      @group(0) @binding(3) var<storage, read> c_in: array<f32>;
      @group(0) @binding(4) var<storage, read> a: array<f32>;
      @group(0) @binding(5) var<storage, read> d: array<f32>;
      @group(0) @binding(6) var<storage, read_write> out_buf: array<f32>;

      struct Uniforms {
        seq: u32,
        d_inner: u32,
        d_state: u32,
        reverse: u32,
      };
      @group(0) @binding(7) var<uniform> uniforms: Uniforms;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let j = global_id.x;
        if (j >= uniforms.d_inner) { return; }
        
        var state_arr: array<f32, 16>; // max d_state = 16
        for (var k = 0u; k < uniforms.d_state; k = k + 1u) {
          state_arr[k] = 0.0;
        }
        
        for (var step = 0u; step < uniforms.seq; step = step + 1u) {
          var i = step;
          if (uniforms.reverse == 1u) {
            i = uniforms.seq - 1u - step;
          }
          
          let u_val = u[i * uniforms.d_inner + j];
          let dt_val = dt[i * uniforms.d_inner + j];
          let d_val = d[j];
          
          var sum = 0.0;
          for (var k = 0u; k < uniforms.d_state; k = k + 1u) {
            let a_val = a[j * uniforms.d_state + k];
            let b_val = b_in[i * uniforms.d_state + k];
            let c_val = c_in[i * uniforms.d_state + k];
            
            let a_t = exp(dt_val * a_val);
            state_arr[k] = a_t * state_arr[k] + u_val * dt_val * b_val;
            sum = sum + state_arr[k] * c_val;
          }
          out_buf[i * uniforms.d_inner + j] = sum + u_val * d_val;
        }
      }
    `;
    const ssmModule = this.device.createShaderModule({ code: ssmShaderCode });
    this.ssmPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: ssmModule, entryPoint: 'main' }
    });
  }

  async dense(x_cpu, w_T_cpu, b_cpu, M, K, N, y_cpu) {
    const device = this.device;
    
    const xBuf = device.createBuffer({ size: x_cpu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(xBuf, 0, x_cpu);

    let wBuf = this.weightCache.get(w_T_cpu);
    if (!wBuf) {
      wBuf = device.createBuffer({
        size: w_T_cpu.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(wBuf, 0, w_T_cpu);
      this.weightCache.set(w_T_cpu, wBuf);
    }

    let bBuf = this.weightCache.get(b_cpu);
    if (!bBuf) {
      bBuf = device.createBuffer({
        size: b_cpu.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(bBuf, 0, b_cpu);
      this.weightCache.set(b_cpu, bBuf);
    }

    const yBuf = device.createBuffer({ size: y_cpu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const uData = new Uint32Array([M, K, N]);
    const uBuf = device.createBuffer({ size: uData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uBuf, 0, uData);

    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xBuf } },
        { binding: 1, resource: { buffer: wBuf } },
        { binding: 2, resource: { buffer: bBuf } },
        { binding: 3, resource: { buffer: yBuf } },
        { binding: 4, resource: { buffer: uBuf } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil((M * N) / 64));
    passEncoder.end();

    const readBuf = device.createBuffer({ size: y_cpu.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    commandEncoder.copyBufferToBuffer(yBuf, 0, readBuf, 0, y_cpu.byteLength);

    device.queue.submit([commandEncoder.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const arr = new Float32Array(readBuf.getMappedRange());
    y_cpu.set(arr);
    readBuf.unmap();
    
    xBuf.destroy();
    yBuf.destroy();
    uBuf.destroy();
    readBuf.destroy();
  }

  async ssm(u_cpu, dt_cpu, b_in_cpu, c_in_cpu, a_cpu, d_cpu, seq, d_inner, d_state, out_cpu, reverse) {
    const device = this.device;
    
    const uBuf = device.createBuffer({ size: u_cpu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uBuf, 0, u_cpu);

    const dtBuf = device.createBuffer({ size: dt_cpu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dtBuf, 0, dt_cpu);

    const bBuf = device.createBuffer({ size: b_in_cpu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bBuf, 0, b_in_cpu);

    const cBuf = device.createBuffer({ size: c_in_cpu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(cBuf, 0, c_in_cpu);

    let aBuf = this.weightCache.get(a_cpu);
    if (!aBuf) {
      aBuf = device.createBuffer({ size: a_cpu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(aBuf, 0, a_cpu);
      this.weightCache.set(a_cpu, aBuf);
    }

    let dBuf = this.weightCache.get(d_cpu);
    if (!dBuf) {
      dBuf = device.createBuffer({ size: d_cpu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(dBuf, 0, d_cpu);
      this.weightCache.set(d_cpu, dBuf);
    }

    const outBuf = device.createBuffer({ size: out_cpu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const unifData = new Uint32Array([seq, d_inner, d_state, reverse ? 1 : 0]);
    const unifBuf = device.createBuffer({ size: unifData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(unifBuf, 0, unifData);

    const bindGroup = device.createBindGroup({
      layout: this.ssmPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uBuf } },
        { binding: 1, resource: { buffer: dtBuf } },
        { binding: 2, resource: { buffer: bBuf } },
        { binding: 3, resource: { buffer: cBuf } },
        { binding: 4, resource: { buffer: aBuf } },
        { binding: 5, resource: { buffer: dBuf } },
        { binding: 6, resource: { buffer: outBuf } },
        { binding: 7, resource: { buffer: unifBuf } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(this.ssmPipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(d_inner / 64));
    passEncoder.end();

    const readBuf = device.createBuffer({ size: out_cpu.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    commandEncoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, out_cpu.byteLength);

    device.queue.submit([commandEncoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    out_cpu.set(new Float32Array(readBuf.getMappedRange()));
    readBuf.unmap();
    
    uBuf.destroy();
    dtBuf.destroy();
    bBuf.destroy();
    cBuf.destroy();
    outBuf.destroy();
    unifBuf.destroy();
    readBuf.destroy();
  }
}
