/**
 * webgpu_backend.js — the Xenos GPU backend on WebGPU (browser only).
 *
 * This is the browser half of the Xenos→WebGPU port (docs/XENOS_GPU.md).
 * It implements the IXenosGpuBackend interface consumed by
 * command_processor.js:
 *
 *   memWrite(addr, bytes)     — guest physical writes from PM4 MEM_WRITE / REG_TO_MEM / EVENT_WRITE
 *   eventWrite(evt)           — initiator bookkeeping (log/forward)
 *   draw(params)              — renders via WebGPU *once the Xenos shader translator is wired*;
 *                               see "What works today" in docs/XENOS_GPU.md
 *   setShaders({vertex,pixel}) — stores loaded microcode (feeds the future WGSL translation)
 *   swap()                    — canvas present notification (PM4_XE_SWAP / frame boundary)
 *
 * Importing this module in node is safe — all `navigator.gpu` access is
 * lazy, inside attach(). `createXenosWebGpuBackend()` throws a helpful
 * error when WebGPU is unavailable instead of failing with a TypeError.
 *
 * Honest status: this file wires the device/canvas/pipeline *plumbing* for
 * real — adapter/device request, canvas configuration, depth attachment,
 * index-buffer upload from guest memory, render pass encoding, present.
 * Guest shaders are NOT pixel-translated yet (Xenos microcode→WGSL is the
 * open work), so draw() refuses to fabricate a frame unless the caller
 * explicitly opts in with `placeholderShader: true` (a bring-up triangle
 * pipeline, clearly labelled as not the guest's shader).
 */
(function (global) {
  'use strict';

  function pm4() {
    const g = globalThis.XenonPM4;
    if (!g) throw new (globalThis.XenosGpuError ?? Error)('pm4.js must be loaded before webgpu_backend.js');
    return g;
  }

  const PLACEHOLDER_WGSL = `
// Bring-up placeholder — NOT translated Xenos microcode.
// A single full-canvas triangle in flat magenta, used only when the caller opts in.
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[i], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
`;

  class XenosWebGpuError extends Error {}

  class XenosWebGpuBackend {
    /**
     * @param {object} opts
     * @param {HTMLCanvasElement} [opts.canvas] — target canvas (required for rendering; optional for headless state tracking)
     * @param {(addr: number) => DataView | Uint8Array | null} opts.readGuestMemory — translate guest physical → BE view
     * @param {(addr: number, bytes: Uint8Array) => void} opts.writeGuestMemory — apply PM4 memory writes to guest memory
     * @param {(level: string, msg: string) => void} [opts.onLog]
     * @param {boolean} [opts.placeholderShader] — opt in to the clearly-labelled placeholder triangle in draw()
     * @param {GPUPowerPreference} [opts.powerPreference]
     */
    constructor({ canvas = null, readGuestMemory = null, writeGuestMemory = null, onLog = null, placeholderShader = false, powerPreference = 'high-performance' } = {}) {
      if (typeof readGuestMemory !== 'function') throw new TypeError('XenosWebGpuBackend: readGuestMemory(addr) => DataView|Uint8Array required');
      this.canvas = canvas;
      this.readGuestMemory = readGuestMemory;
      this.writeGuestMemory = typeof writeGuestMemory === 'function' ? writeGuestMemory : (addr, bytes) => { throw new XenosWebGpuError(`writeGuestMemory not wired (PM4 write to 0x${addr.toString(16)} dropped)`); };
      this.log = typeof onLog === 'function' ? onLog : () => {};
      this.placeholderShader = !!placeholderShader;
      this.powerPreference = powerPreference;

      // WebGPU state — created by attach()
      this.device = null;
      this.context = null;
      this.format = null;
      this.depthTexture = null;
      this.depthFormat = 'depth24plus';
      this.pipelineCache = new Map(); // key -> GPURenderPipeline
      this.placeholderPipeline = null;
      this.uploads = { indexBuffers: 0, indexBytes: 0, frames: 0 };
      this.shaders = { vertex: null, pixel: null };
      this.lastFrame = null; // set by swap()
      this.counter = 0;
    }

    get ready() { return !!this.device; }

    /** Request adapter+device and configure the canvas. Throws XenosWebGpuError with an actionable message when WebGPU is unavailable. */
    async attach() {
      if (this.device) return this;
      const gpu = typeof navigator !== 'undefined' ? navigator.gpu : undefined;
      if (!gpu) {
        throw new XenosWebGpuError('WebGPU is not available in this environment (navigator.gpu is undefined). Use a WebGPU-enabled browser (Chrome/Edge 113+, Firefox Nightly, Safari 18+) or the headless test double.');
      }
      const adapter = await gpu.requestAdapter({ powerPreference: this.powerPreference });
      if (!adapter) throw new XenosWebGpuError('WebGPU requestAdapter returned null — GPU blocklisted or WebGPU disabled (chrome://flags).');
      this.device = await adapter.requestDevice({ label: 'xenos-gpu' });
      this.device.lost.then((info) => this.log('error', `WebGPU device lost: ${info.message}`));
      if (this.canvas) {
        this.context = this.canvas.getContext('webgpu');
        if (!this.context) throw new XenosWebGpuError('canvas.getContext("webgpu") returned null — is the canvas element valid and not already attached to another context type?');
        this.format = gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
        this._recreateDepth();
      } else {
        this.format = 'bgra8unorm';
      }
      return this;
    }

    _recreateDepth() {
      if (!this.device || !this.canvas) return;
      const width = Math.max(1, this.canvas.width | 0);
      const height = Math.max(1, this.canvas.height | 0);
      if (this.depthTexture) this.depthTexture.destroy();
      this.depthTexture = this.device.createTexture({
        size: [width, height, 1],
        format: this.depthFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        label: 'xenos-depth',
      });
    }

    /** Resize depth attachment to the canvas size (call before drawing after a resize). */
    resize() { this._recreateDepth(); }

    // --- IXenosGpuBackend ----------------------------------------------------

    memWrite(addr, bytes) {
      this.writeGuestMemory(addr >>> 0, bytes);
    }

    eventWrite(evt) {
      this.counter = (this.counter + 1) >>> 0;
      this.log('info', `eventWrite event=0x${(evt.eventType ?? 0).toString(16)} opcode=${evt.opcode ?? 'EVENT_WRITE'} addr=0x${(evt.address ?? 0).toString(16)}`);
    }

    interrupt(mask) { this.log('info', `interrupt request mask=0x${(mask >>> 0).toString(16)}`); }

    invalidateState(mask) { this.log('debug', `invalidateState mask=0x${(mask >>> 0).toString(16)}`); this.pipelineCache.clear(); }

    setShaderBases(vsBase, psBase) { this.log('debug', `setShaderBases vs=0x${vsBase.toString(16)} ps=0x${psBase.toString(16)}`); }

    vizQuery(state) { this.log('debug', `vizQuery ${state}`); }

    setShaders({ vertex, pixel }) {
      if (vertex) this.shaders.vertex = vertex;
      if (pixel) this.shaders.pixel = pixel;
      this.pipelineCache.clear(); // microcode changed — pipelines are keyed on shader content, so drop everything
    }

    /**
     * Upload an index buffer from guest memory into a GPUBuffer.
     * Real, tested code path — but note the guest index endianness (swap_mode)
     * must be applied by the time we get here or by the shader; Xenos indices
     * are big-endian, so we byteswap when indexEndianness indicates 8in16/8in32.
     */
    uploadIndexBuffer({ indexGuestBase, numIndices, indexFormat /* 0=int16, 1=int32 */, indexEndianness /* swap_mode 0..3 from VGT_DMA_SIZE */ }) {
      if (!this.device) throw new XenosWebGpuError('uploadIndexBuffer before attach()');
      const bytesPerIndex = indexFormat === 1 ? 4 : 2;
      const total = numIndices * bytesPerIndex;
      const view = this.readGuestMemory(indexGuestBase >>> 0);
      if (!view) throw new XenosWebGpuError(`index buffer at guest physical 0x${(indexGuestBase >>> 0).toString(16)} is not mapped`);
      const paddedTotal = Math.ceil(total / 4) * 4;
      const be = new Uint8Array(paddedTotal);
      const src = view instanceof Uint8Array ? view : new Uint8Array(view.buffer, view.byteOffset ?? 0, view.byteLength ?? view.buffer.byteLength);
      be.set(src.subarray(0, total));
      // The GPU fetch path applies the VGT_DMA_SIZE swap_mode to raw guest bytes; we mirror that on upload:
      // read each 32-bit word big-endian, apply GpuSwap(swap_mode), write it little-endian for the LE GPU.
      const gpuSwap32 = pm4().gpuSwap32;
      const out = new Uint8Array(paddedTotal);
      {
        const s = new DataView(be.buffer), d = new DataView(out.buffer);
        for (let i = 0; i < paddedTotal; i += 4) d.setUint32(i, gpuSwap32(s.getUint32(i, false), indexEndianness | 0), true);
      }
      const buffer = this.device.createBuffer({ size: Math.max(4, total), usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, label: 'xenos-index' });
      this.device.queue.writeBuffer(buffer, 0, out);
      this.uploads.indexBuffers++;
      this.uploads.indexBytes += total;
      return buffer;
    }

    _getPlaceholderPipeline() {
      if (this.placeholderPipeline) return this.placeholderPipeline;
      const module = this.device.createShaderModule({ code: PLACEHOLDER_WGSL, label: 'placeholder' });
      this.placeholderPipeline = this.device.createRenderPipeline({
        label: 'xenos-placeholder-triangle',
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_main' },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
      });
      return this.placeholderPipeline;
    }

    draw(params) {
      if (!this.device) throw new XenosWebGpuError('draw before attach()');
      if (!this.canvas) throw new XenosWebGpuError('draw requires a canvas (headless rendering into textures is not wired yet)');
      this.uploads.frames++;
      if (!this.placeholderShader) {
        throw new XenosWebGpuError(
          `draw(prim=${params.primTypeName ?? params.primType}, indices=${params.numIndices}): the Xenos microcode→WGSL translation is not wired yet — ` +
          'see docs/XENOS_GPU.md ("Bring-up order"). Pass { placeholderShader: true } to attach() to render the bring-up triangle instead.'
        );
      }
      // --- clearly-labelled bring-up path: renders a full-canvas placeholder triangle -------------
      const encoder = this.device.createCommandEncoder({ label: 'xenos-frame' });
      const colorView = this.context.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: colorView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(this._getPlaceholderPipeline());
      pass.draw(3);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      return;
    }

    /** Frame boundary from PM4_XE_SWAP — xenia uses this to trigger present/swap on the host. */
    swap() {
      this.lastFrame = { at: (typeof performance !== 'undefined' ? performance.now() : Date.now()) };
      this.log('debug', `swap (frame ${this.uploads.frames})`);
    }

    /** Call after the last frame of a session to release GPU resources. */
    destroy() {
      if (this.depthTexture) { this.depthTexture.destroy(); this.depthTexture = null; }
      if (this.device) { this.device.destroy(); this.device = null; }
      this.context = null;
      this.pipelineCache.clear();
      this.placeholderPipeline = null;
    }
  }

  const api = { XenosWebGpuBackend, XenosWebGpuError, PLACEHOLDER_WGSL };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonWebGpuBackend = api;
})(typeof self !== 'undefined' ? self : globalThis);
