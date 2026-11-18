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

  function vertexFetch() {
    const g = globalThis.XenonGpuVertexFetch;
    if (!g) throw new XenosWebGpuError('vertex_fetch.js must be loaded before webgpu_backend.js');
    return g;
  }

  // ---- vertex-fetch-aware placeholder (NOT translated Xenos microcode) -------------------
  // Generated per-attribute WGSL for the bring-up pipeline so the *real* WebGPU vertex
  // layout/binding path is exercised: the declared @location(N) parameters match the
  // descriptors produced by buildGpuVertexBufferDescriptors(). Attribute 0 is passed
  // through as position with no Xenos-specific transform — this draw proves the fetch
  // path works (descriptors → upload → setVertexBuffer → GPU), nothing more.
  function wgslTypeForFormat(format) {
    switch (format) {
      case 'float32': return 'f32';
      case 'float32x2': return 'vec2<f32>';
      case 'float32x3': return 'vec3<f32>';
      case 'float32x4': return 'vec4<f32>';
      case 'float16x2': return 'vec2<f16>';
      case 'float16x4': return 'vec4<f16>';
      case 'unorm16x2': case 'snorm16x2': return 'vec2<f32>';
      case 'unorm8x4': case 'unorm16x4': case 'snorm8x4': case 'snorm16x4': case 'unorm10-10-10-2': return 'vec4<f32>';
      case 'uint8x4': return 'vec4<u32>';
      case 'sint8x4': return 'vec4<i32>';
      case 'uint16x2': return 'vec2<u32>';
      case 'sint16x2': return 'vec2<i32>';
      case 'uint16x4': return 'vec4<u32>';
      case 'sint16x4': return 'vec4<i32>';
      case 'uint32': return 'u32';
      case 'sint32': return 'i32';
      case 'uint32x2': return 'vec2<u32>';
      case 'sint32x2': return 'vec2<i32>';
      case 'uint32x4': return 'vec4<u32>';
      case 'sint32x4': return 'vec4<i32>';
      default:
        throw new XenosWebGpuError(`no WGSL type for vertex format '${format}' — refusing to emit a pipeline that cannot validate`);
    }
  }

  function componentCountOf(wgslType) {
    if (wgslType === 'f32' || wgslType === 'u32' || wgslType === 'i32') return 1;
    if (wgslType.startsWith('vec2')) return 2;
    if (wgslType.startsWith('vec3')) return 3;
    if (wgslType.startsWith('vec4')) return 4;
    throw new XenosWebGpuError(`unhandled WGSL type '${wgslType}'`);
  }

  /** FNV-1a 32-bit content hash — identifies uploaded buffer contents (guest memory rewrites same address). */
  function fnv1a32(bytes) {
    let h = 0x811c9dc5 >>> 0;
    for (let i = 0; i < bytes.length; i++) { h ^= bytes[i]; h = Math.imul(h, 0x01000193) >>> 0; }
    return h >>> 0;
  }

  /** Build the WGSL source for the fetch bring-up vertex shader. Attribute 0 → position. */
  function buildFetchPlaceholderWGSL(descriptors) {
    const attrs = [];
    for (const d of descriptors) for (const a of d.attributes) attrs.push(a);
    if (!attrs.length) throw new XenosWebGpuError('vertex fetch plan has no attributes — nothing to declare');
    const needsF16 = attrs.some((a) => a.format.startsWith('float16'));
    const params = attrs.map((a, i) => `@location(${a.shaderLocation}) a${i}: ${wgslTypeForFormat(a.format)}`).join(', ');
    const t0 = wgslTypeForFormat(attrs[0].format);
    const n0 = componentCountOf(t0);
    const f32 = (expr) => (t0.includes('u32') || t0.includes('i32') || t0.includes('f16')) ? `f32(${expr})` : expr;
    // vec4<f32> position from attribute 0 — component conversion via scalar constructors (valid WGSL for f32/u32/i32/f16 lanes).
    let pos;
    if (n0 === 1) pos = `vec4<f32>(${f32('a0')}, 0.0, 0.0, 1.0)`;
    else if (n0 === 2) pos = `vec4<f32>(${f32('a0.x')}, ${f32('a0.y')}, 0.0, 1.0)`;
    else if (n0 === 3) pos = `vec4<f32>(${f32('a0.x')}, ${f32('a0.y')}, ${f32('a0.z')}, 1.0)`;
    else pos = t0 === 'vec4<f32>' ? 'a0' : `vec4<f32>(${f32('a0.x')}, ${f32('a0.y')}, ${f32('a0.z')}, ${f32('a0.w')})`;
    return `${needsF16 ? 'enable f16;\n' : ''}// Bring-up placeholder — NOT translated Xenos microcode.
// Attribute 0 passes through as position with no transform; the rest exist to make the
// declared interface match the vertex layout. Only reachable with placeholderShader: true.
@vertex fn vs_main(${params}) -> @builtin(position) vec4<f32> {
  return ${pos};
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.5, 0.0, 1.0);
}
`;
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
      this.uploads = { indexBuffers: 0, indexBytes: 0, vertexBuffers: 0, vertexBytes: 0, frames: 0, vertexBufferCacheHits: 0, indexBufferCacheHits: 0 };
      this.lastVertexFetch = null;    // set by prepareVertexFetch() (draw path) — descriptors + constants
      this.lastFetchConstants = null;
      // Content-addressed GPU buffer cache (key: kind:base:bytes:endian → hash-verified buffer).
      // Guest memory is rewritten at the same addresses between frames; the hash is what keeps
      // reuse safe. Bounded by count (gpuBufferCacheLimit) and total bytes (gpuBufferCacheBytesLimit).
      this.gpuBufferCache = new Map();
      this.gpuBufferCacheClock = 0;
      this.gpuBufferCacheLimit = 32;
      this.gpuBufferCacheBytesLimit = 128 * 1024 * 1024;
      this.gpuBufferCacheBytes = 0;
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
     * Fetch (or create) a cached GPUBuffer for `raw` (the padded BE bytes about to be uploaded).
     * Returns { buffer, hit }. On a miss, `makeBuffer(desc)` is called with the upload desc.
     * Never caches buffers bigger than the byte limit.
     */
    _cachedBuffer(kind, base, total, endian, raw, makeBuffer) {
      if (raw.byteLength > this.gpuBufferCacheBytesLimit) {
        return { buffer: makeBuffer(), hit: false }; // too big to cache — upload through
      }
      const hash = fnv1a32(raw);
      const key = `${kind}:${base >>> 0}:${total >>> 0}:${endian | 0}`;
      const hit = this.gpuBufferCache.get(key);
      if (hit && hit.hash === hash) {
        hit.lastUsed = ++this.gpuBufferCacheClock;
        if (kind === 'vb') this.uploads.vertexBufferCacheHits++; else this.uploads.indexBufferCacheHits++;
        return { buffer: hit.buffer, hit: true };
      }
      const buffer = makeBuffer();
      if (hit) { // content changed at the same key — drop the stale buffer
        this._dropCachedBuffer(key, hit);
      }
      this.gpuBufferCache.set(key, { buffer, hash, bytes: raw.byteLength, lastUsed: ++this.gpuBufferCacheClock });
      this.gpuBufferCacheBytes += raw.byteLength;
      this._evictBufferCache();
      return { buffer, hit: false };
    }

    _dropCachedBuffer(key, entry) {
      this.gpuBufferCache.delete(key);
      this.gpuBufferCacheBytes -= entry.bytes;
      if (entry.buffer && typeof entry.buffer.destroy === 'function') entry.buffer.destroy();
    }

    _evictBufferCache() {
      while (this.gpuBufferCache.size > this.gpuBufferCacheLimit || this.gpuBufferCacheBytes > this.gpuBufferCacheBytesLimit) {
        let oldestKey = null, oldest = Infinity;
        for (const [k, v] of this.gpuBufferCache) { if (v.lastUsed < oldest) { oldest = v.lastUsed; oldestKey = k; } }
        if (oldestKey === null) break;
        this._dropCachedBuffer(oldestKey, this.gpuBufferCache.get(oldestKey));
      }
    }

    /**
     * Analyze the loaded vertex shader + a snapshot of the fetch constants.
     * Pure with respect to the GPU device — returns the plan the upload/draw
     * steps consume. Loud (XenosFetchError) on anything WebGPU cannot represent.
     * @param {{vertexShaderCode?: Uint32Array|number[], fetchConstants?: Uint32Array|number[]|((i:number)=>number)}} opts
     */
    prepareVertexFetch({ vertexShaderCode = null, fetchConstants = null } = {}) {
      const code = vertexShaderCode ?? this.shaders.vertex?.code;
      if (!code) throw new XenosWebGpuError('prepareVertexFetch: no vertex shader loaded (IM_LOAD/IM_LOAD_IMMEDIATE with shader type 0)');
      const consts = fetchConstants ?? this.lastFetchConstants;
      if (!consts) throw new XenosWebGpuError('prepareVertexFetch: no fetch constants — the draw packet must carry vertexFetchConstants (the command processor attaches them automatically)');
      const VF = vertexFetch();
      const readDword = typeof consts === 'function' ? consts : (i) => (consts[i] >>> 0);
      const constants = VF.decodeVertexFetchConstants(readDword);
      const analysis = VF.analyzeVertexShader(code);
      const descriptors = VF.buildGpuVertexBufferDescriptors(analysis);
      const bindings = descriptors.map((d) => {
        const fc = constants[d.fetchConstant];
        if (!fc) throw new XenosWebGpuError(`fetch constant ${d.fetchConstant} out of range`);
        if (fc.type !== 3 /* FETCH_CONSTANT_TYPE.vertex */) {
          throw new XenosWebGpuError(`fetch constant ${d.fetchConstant} is type ${fc.typeName} — expected vertex; refusing to guess how to bind it`);
        }
        if (fc.sizeWords === 0) throw new XenosWebGpuError(`fetch constant ${d.fetchConstant} has size 0 — nothing to upload`);
        return { ...d, constant: fc, byteSize: fc.sizeWords * 4 };
      });
      const plan = { descriptors: bindings, analysis, constants, buffers: null };
      this.lastVertexFetch = plan;
      this.lastFetchConstants = consts;
      return plan;
    }

    /**
     * Upload the vertex buffers of a fetch plan (default: the last one) into
     * GPUBuffers, applying each fetch constant's GpuSwap endianness to the
     * big-endian guest dwords (same scheme as uploadIndexBuffer; endian==kNone
     * passes the BE words through, which is what a dword-aligned BE arena needs
     * for integer-read data).
     */
    uploadVertexBuffers(plan = null) {
      if (!this.device) throw new XenosWebGpuError('uploadVertexBuffers before attach()');
      const p = plan ?? this.lastVertexFetch;
      if (!p) throw new XenosWebGpuError('uploadVertexBuffers: no fetch plan — call prepareVertexFetch() first');
      if (typeof GPUBufferUsage === 'undefined') throw new XenosWebGpuError('uploadVertexBuffers: GPUBufferUsage is not defined in this environment');
      const gpuSwap32 = pm4().gpuSwap32;
      const buffers = [];
      for (const b of p.descriptors) {
        const base = b.constant.baseAddress >>> 0;
        const view = this.readGuestMemory(base);
        if (!view) throw new XenosWebGpuError(`vertex buffer (fetch constant ${b.fetchConstant}) at guest physical 0x${base.toString(16)} is not mapped`);
        const total = b.byteSize;
        const src = view instanceof Uint8Array ? view : new Uint8Array(view.buffer, view.byteOffset ?? 0, view.byteLength ?? view.buffer.byteLength);
        const paddedTotal = Math.ceil(total / 4) * 4;
        const raw = new Uint8Array(paddedTotal);
        raw.set(src.subarray(0, Math.min(total, src.length)));
        const { buffer, hit } = this._cachedBuffer('vb', base, total, b.constant.endian | 0, raw, () => {
          const buf = this.device.createBuffer({
            size: paddedTotal,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            label: `xenos-vertex-${b.fetchConstant}`,
          });
          const out = new Uint8Array(paddedTotal);
          const dvw = new DataView(raw.buffer), dvo = new DataView(out.buffer);
          for (let i = 0; i < paddedTotal; i += 4) dvo.setUint32(i, gpuSwap32(dvw.getUint32(i, false), b.constant.endian | 0), true);
          this.device.queue.writeBuffer(buf, 0, out);
          return buf;
        });
        if (!hit) {
          this.uploads.vertexBuffers++;
          this.uploads.vertexBytes += total;
        }
        buffers.push({ bindingIndex: b.bindingIndex, fetchConstant: b.fetchConstant, buffer, arrayStride: b.arrayStride, attributes: b.attributes });
      }
      p.buffers = buffers;
      return buffers;
    }

    /** Placeholder pipeline whose vertex stage consumes the plan's attributes (fetch bring-up). */
    _getFetchPlaceholderPipeline(descriptors) {
      const key = 'fetch:' + JSON.stringify(descriptors.map((d) => [d.arrayStride, d.attributes])) + ':' + this.format;
      let p = this.pipelineCache.get(key);
      if (p) return p;
      const wgsl = buildFetchPlaceholderWGSL(descriptors);
      const module = this.device.createShaderModule({ code: wgsl, label: 'fetch-placeholder' });
      const vertexBuffers = descriptors.map((d) => ({
        arrayStride: d.arrayStride,
        stepMode: d.stepMode ?? 'vertex',
        attributes: d.attributes.map((a) => ({ shaderLocation: a.shaderLocation, offset: a.offset, format: a.format })),
      }));
      p = this.device.createRenderPipeline({
        label: 'xenos-placeholder-fetch',
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_main', buffers: vertexBuffers },
        fragment: { module, entryPoint: 'fs_main', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
      });
      this.pipelineCache.set(key, p);
      return p;
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
      const base = indexGuestBase >>> 0;
      const { buffer, hit } = this._cachedBuffer('ib', base, total, indexEndianness | 0, be, () => {
        const out = new Uint8Array(paddedTotal);
        const s = new DataView(be.buffer), d = new DataView(out.buffer);
        for (let i = 0; i < paddedTotal; i += 4) d.setUint32(i, gpuSwap32(s.getUint32(i, false), indexEndianness | 0), true);
        const buf = this.device.createBuffer({ size: paddedTotal, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, label: 'xenos-index' });
        this.device.queue.writeBuffer(buf, 0, out);
        return buf;
      });
      if (!hit) {
        this.uploads.indexBuffers++;
        this.uploads.indexBytes += total;
      }
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
      // --- clearly-labelled bring-up path ---------------------------------------------------------
      // When a vertex shader is loaded AND the draw carries fetch constants (the command
      // processor attaches them), the bring-up goes through the REAL vertex fetch path:
      // analyze → descriptors → upload → setVertexBuffer → generated placeholder pipeline.
      // Any failure there is loud (never falls back to a fabricated frame). Without a
      // vertex shader we cannot fetch anything — the full-canvas triangle remains.
      const fetchReady = !!(this.shaders.vertex && params.vertexFetchConstants);
      if (fetchReady) {
        const plan = this.prepareVertexFetch({ vertexShaderCode: this.shaders.vertex.code, fetchConstants: params.vertexFetchConstants });
        const encoder = this.device.createCommandEncoder({ label: 'xenos-frame' });
        const colorView = this.context.getCurrentTexture().createView();
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: colorView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
        });
        if (plan.descriptors.length > 0) {
          const buffers = this.uploadVertexBuffers(plan);
          pass.setPipeline(this._getFetchPlaceholderPipeline(plan.descriptors));
          for (const b of buffers) pass.setVertexBuffer(b.bindingIndex, b.buffer);
          this._setIndexBufferAndDraw(pass, params);
        } else {
          // VS loaded but it fetches nothing — draw the full-canvas placeholder (documented).
          this.log('info', 'draw: vertex shader fetches nothing — issuing the plain bring-up triangle');
          pass.setPipeline(this._getPlaceholderPipeline());
          pass.draw(3);
        }
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        return;
      }
      if (this.shaders.vertex) {
        this.log('warn', 'draw: vertex shader loaded but the packet carried no fetch constants — bypassing the fetch path (full-canvas placeholder)');
      }
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

    /** Bind the index buffer (if any) and issue the draw, from a fetch-planned pass. */
    _setIndexBufferAndDraw(pass, params) {
      if (params.isIndexed) {
        const ib = this.uploadIndexBuffer({
          indexGuestBase: params.indexGuestBase,
          numIndices: params.numIndices,
          indexFormat: params.indexFormat,
          indexEndianness: params.indexEndianness,
        });
        pass.setIndexBuffer(ib, params.indexFormat === 1 ? 'uint32' : 'uint16');
        pass.drawIndexed(params.numIndices, 1, 0, 0, 0);
        return;
      }
      pass.draw(params.numIndices);
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
