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

  /** FNV-1a 32-bit over decoded u32 words — cheap identity check for fetch-constant snapshots. */
  function fnv1a32Words(words) {
    let h = 0x811c9dc5 >>> 0;
    for (let i = 0; i < words.length; i++) {
      const w = words[i] >>> 0;
      h ^= w & 0xff; h = Math.imul(h, 0x01000193) >>> 0;
      h ^= (w >>> 8) & 0xff; h = Math.imul(h, 0x01000193) >>> 0;
      h ^= (w >>> 16) & 0xff; h = Math.imul(h, 0x01000193) >>> 0;
      h ^= (w >>> 24) & 0xff; h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h >>> 0;
  }

  /** FNV-1a 32-bit content hash — identifies uploaded buffer contents (guest memory rewrites same address). */
  function fnv1a32(bytes) {
    let h = 0x811c9dc5 >>> 0;
    for (let i = 0; i < bytes.length; i++) { h ^= bytes[i]; h = Math.imul(h, 0x01000193) >>> 0; }
    return h >>> 0;
  }
  /** Cache-key fragment: alpha test compare function (translation-time decision). */
  function alphaTestKey(program) { return program && program.alphaTest ? String(program.alphaTest.compare) : '0'; }
  /** Cache-key fragment: gamma encode presence. */
  function convertKey(program) { return program && program.convertColor0ToGamma ? '1' : '0'; }
  /** Cache-key fragment: the PS statically exports oC0 (pipeline write mask depends on it). */
  function colorExportKey(program) { return program && program.fragment && program.fragment.exportsColor0 === false ? '0' : '1'; }

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

  // Fixed fragment stage for the translated-vertex-shader path. The guest PIXEL
  // shader has no translator yet (step 6), so a trivially constant FS is used:
  // it shades the real, guest-microcode-transformed geometry and nothing else.
  // It never feeds back into guest state. Not reachable when a pixel shader is
  // loaded — draw() traps loudly instead of fabricating a fragment stage.
  const FLAT_FRAGMENT_WGSL = `// Bring-up fragment stage — NOT translated Xenos microcode (pixel-shader translation is step 6).
@fragment fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(0.9, 0.55, 0.15, 1.0);
}
`;

  /** Integer-component GPUVertexFormat strings (need vec4<u32>/vec4<i32> shader locations, not vec4<f32>). */
  function isIntegerGpuVertexFormat(format) {
    return format.startsWith('uint') || format.startsWith('sint');
  }

  // Xenos VGT_DRAW_INITIATOR prim_type (pm4.js PRIMITIVE_TYPE) → GPUPrimitiveTopology.
  // Anything WebGPU cannot represent (triangle fan, quads, polygon, rect lists)
  // traps loudly instead of silently drawing with the wrong connectivity.
  const PRIM_TYPE_TOPOLOGY = new Map([
    [0x01 /* kPointList */, 'point-list'],
    [0x02 /* kLineList */, 'line-list'],
    [0x03 /* kLineStrip */, 'line-strip'],
    [0x04 /* kTriangleList */, 'triangle-list'],
    [0x06 /* kTriangleStrip */, 'triangle-strip'],
  ]);
  function topologyForPrimType(primType, primTypeName) {
    const topology = PRIM_TYPE_TOPOLOGY.get(primType & 0x3f);
    if (!topology) {
      // Called at draw time, long after XenosWebGpuError (declared below) is initialized.
      throw new XenosWebGpuError(`prim ${primTypeName ?? `prim_0x${(primType & 0x3f).toString(16)}`}: primitive topology is not encodable in WebGPU (triangle-fan/quad/polygon connectivity would need index conversion) — refusing to draw with the wrong connectivity`);
    }
    return topology;
  }

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
      // Per-frame clear state for the color/depth attachments: the guest is
      // responsible for clearing RT contents — the host clears only the FIRST
      // pass of a frame (defined image), never a draw mid-frame (which would
      // erase earlier draws' output and break depth sequencing). swap() marks
      // the frame boundary; attach()/resize() reset both targets.
      this._frameClearColor = false;
      this._frameClearDepth = false;
      this.depthTexture = null;
      this.depthFormat = 'depth24plus-stencil8'; // must match pipeline depth-stencil format
      this.pipelineCache = new Map(); // key -> GPURenderPipeline
      // Translated-program path: WeakMap<vs ucode object, Map<pairKey, program pair>>
      // and a per-VS-shader Map<layoutKey, { pipeline, bindGroup }>. Pipelines/bind groups
      // are created once per (shader pair, layout, topology) — never per draw.
      this.programMemo = new WeakMap();
      this.translatedPipelines = new WeakMap();
      this.psCodeIds = new WeakMap();
      this.psCodeIdCounter = 0;
      this.aluConstantBuffer = null;     // GPUBuffer, persistent — contents re-uploaded only on change
      this.boolConstantBuffer = null;    // GPUBuffer, persistent likewise
      this.aluHash = -1;                 // content hash of the last uploaded ALU constants (-1 = never)
      this.boolHash = -1;
      this.uploads = { indexBuffers: 0, indexBytes: 0, vertexBuffers: 0, vertexBytes: 0, frames: 0, vertexBufferCacheHits: 0, indexBufferCacheHits: 0, constantUploads: 0 };
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
      // Per-shader single-entry plan memo: WeakMap<code, { hash, plan }> — the analysis
      // (ucode walk + descriptor build) and constant decode are deterministic in
      // (code, fetch-constant snapshot); re-running them per draw is waste.
      this.planMemo = new WeakMap();
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
      this._frameClearColor = false;
      this._frameClearDepth = false;
      this.depthTexture = this.device.createTexture({
        size: [width, height, 1],
        format: this.depthFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        label: 'xenos-depth',
      });
      this.depthView = typeof this.depthTexture.createView === 'function' ? this.depthTexture.createView() : null;
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

    invalidateState(mask) { this.log('debug', `invalidateState mask=0x${(mask >>> 0).toString(16)}`); this.pipelineCache.clear(); this.programMemo = new WeakMap(); this.translatedPipelines = new WeakMap(); }

    setShaderBases(vsBase, psBase) { this.log('debug', `setShaderBases vs=0x${vsBase.toString(16)} ps=0x${psBase.toString(16)}`); }

    vizQuery(state) { this.log('debug', `vizQuery ${state}`); }

    setShaders({ vertex, pixel }) {
      if (vertex) this.shaders.vertex = vertex;
      if (pixel) this.shaders.pixel = pixel;
      this.pipelineCache.clear(); // microcode changed — pipelines are keyed on shader content, so drop everything
      this.planMemo = new WeakMap();
      this.programMemo = new WeakMap();
      this.translatedPipelines = new WeakMap();
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
      const memoizable = typeof consts !== 'function' && consts.length === 192;
      const constHash = memoizable ? fnv1a32Words(consts) : -1;
      if (memoizable) {
        const memo = this.planMemo.get(code);
        if (memo && memo.hash === constHash) {
          this.lastFetchConstants = consts;
          return memo.plan; // descriptors/analysis unchanged — refresh the constants reference only
        }
      }
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
      const plan = { descriptors: bindings, analysis, constants, buffers: null, _pipeline: null };
      if (memoizable) this.planMemo.set(code, { hash: constHash, plan });
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

    /** Stable identity key for a pixel shader ucode object (pipelines differ per PS). */
    _psId(code) {
      let id = this.psCodeIds.get(code);
      if (!id) {
        id = ++this.psCodeIdCounter;
        this.psCodeIds.set(code, id);
      }
      return id;
    }

    /**
     * Translate the loaded vertex shader and (optionally) its pixel shader as one
     * consistent program — memoized per (vs, ps) ucode object pair. The fragment
     * stage is null when no pixel shader is loaded (the caller binds the
     * bring-up FS then). Loud on any unsupported construct.
     */
    _translateProgram(vsCode, psCode, alphaTest, convertColor0ToGamma = false) {
      const atKey = alphaTest ? `at${alphaTest.compare}` : 'at-';
      const gKey = convertColor0ToGamma ? 'g1' : 'g0';
      const pairKey = (psCode ? `ps${this._psId(psCode)}` : 'flat') + '|' + atKey + '|' + gKey;
      let perVs = this.programMemo.get(vsCode);
      if (!perVs) {
        perVs = new Map();
        this.programMemo.set(vsCode, perVs);
      }
      let pair = perVs.get(pairKey);
      if (pair) return pair;
      const TX = globalThis.XenonGpuTranslate;
      if (!TX) throw new XenosWebGpuError('XenonGpuTranslate is not loaded — public/js/gpu/wgsl_translate.js must be loaded before the backend builds a draw pipeline');
      try {
        pair = TX.translateShaderProgram(vsCode, psCode, { alphaTest: alphaTest ?? undefined, convertColor0ToGamma });
      } catch (e) {
        const name = e && e.constructor && e.constructor.name ? e.constructor.name : 'Error';
        const detail = e && e.message ? e.message : String(e);
        throw new XenosWebGpuError(`draw: the guest shaders failed to translate to WGSL (${name}): ${detail}`);
      }
      perVs.set(pairKey, pair);
      return pair;
    }

    /**
     * Upload the ALU (fc[]) and bool constant snapshots into persistent GPU
     * buffers. Content-hashed: the ~1032-word hash costs microseconds and the
     * 4 KB writeBuffer only happens on draws whose constants actually changed.
     */
    _uploadShaderConstants(params, program) {
      if (typeof GPUBufferUsage === 'undefined') throw new XenosWebGpuError('_uploadShaderConstants: GPUBufferUsage is not defined in this environment');
      const alu = params.aluConstants, bools = params.boolConstants;
      if (program.usesFloatConstants) {
        if (!alu) throw new XenosWebGpuError('draw: the translated vertex shader reads fc[] (ALU constants) but the packet carried no ALU constants — refusing to bind fabricated constants');
        const hash = fnv1a32Words(alu);
        if (hash !== this.aluHash || !this.aluConstantBuffer) {
          if (!this.aluConstantBuffer) {
            this.aluConstantBuffer = this.device.createBuffer({ size: 4096, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'xenos-alu-constants' });
          }
          const bytes = new Uint8Array(4096);
          const dv = new DataView(bytes.buffer);
          for (let i = 0; i < 1024; i++) dv.setUint32(i * 4, alu[i] >>> 0, true);
          this.device.queue.writeBuffer(this.aluConstantBuffer, 0, bytes);
          this.aluHash = hash;
          this.uploads.constantUploads++;
        }
      }
      if (program.usesBoolConstants) {
        if (!bools) throw new XenosWebGpuError('draw: the translated vertex shader reads bool constants (bc(n)) but the packet carried none — refusing to bind fabricated bools');
        const hash = fnv1a32Words(bools);
        if (hash !== this.boolHash || !this.boolConstantBuffer) {
          if (!this.boolConstantBuffer) {
            this.boolConstantBuffer = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'xenos-bool-constants' });
          }
          const bytes = new Uint8Array(32);
          const dv = new DataView(bytes.buffer);
          for (let i = 0; i < 8; i++) dv.setUint32(i * 4, bools[i] >>> 0, true);
          this.device.queue.writeBuffer(this.boolConstantBuffer, 0, bytes);
          this.boolHash = hash;
          this.uploads.constantUploads++;
        }
      }
    }

    /**
     * Upload RB_ALPHA_REF (IEEE-754 bitcast) into a persistent 16-byte uniform
     * for the translated guest pixel shader's alpha test. Content-hashed: the
     * write happens only when the reference dword changes. Loud when the
     * translated program declares the alpha test but the snapshot carried no
     * reference — never fabricate one.
     */
    _uploadAlphaTest(params, program) {
      if (!program.alphaTest) return;
      const dword = params.renderTargets?.alphaRef;
      if (dword === undefined) {
        throw new XenosWebGpuError('draw: the translated guest pixel shader has an alpha test enabled but the render-target snapshot carries no alpha reference (params.renderTargets.alphaRef) — refusing to draw with a fabricated reference');
      }
      if (dword === this.alphaRefHash && this.alphaTestBuffer) return;
      if (!this.alphaTestBuffer) {
        this.alphaTestBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'xenos-alpha-test' });
      }
      const bytes = new Uint8Array(16);
      new DataView(bytes.buffer).setUint32(0, dword >>> 0, true);
      this.device.queue.writeBuffer(this.alphaTestBuffer, 0, bytes);
      this.alphaRefHash = dword >>> 0;
      this.uploads.constantUploads++;
    }

    /** The color target descriptor for the canvas attachment (format + write mask + blend). */
    _colorTargetDesc(rtState, writesColor = true) {
      const target = { format: this.format, writeMask: writesColor ? (rtState ? rtState.writeMask : 0xF) : 0x0 };
      if (writesColor && rtState && rtState.blend) target.blend = rtState.blend;
      return target;
    }

    /** The depth-stencil state for every pipeline this backend builds. */
    _depthStencilDesc(rtState) {
      if (rtState && rtState.depthStencil) return { ...rtState.depthStencil };
      return { format: this.depthFormat, depthWriteEnabled: false, depthCompare: 'always' };
    }

    /**
     * Decode the render-target register snapshot for this draw. Refuses loudly
     * when the snapshot is missing (the command processor attaches it; a host
     * not doing so is not executing the guest's ROP state) or when the guest
     * programmed something WebGPU cannot represent.
     */
    _resolveRenderTargets(params, program) {
      const rs = globalThis.XenonGpuRenderState;
      if (!rs) {
        throw new XenosWebGpuError('XenonGpuRenderState is not loaded — public/js/gpu/render_state.js must be loaded before the backend builds a draw');
      }
      let rtState;
      try {
        rtState = rs.buildRenderTargetState(params.renderTargets, {
          // No pixel shader → the bring-up FS writes the attachment unconditionally.
          psExportsColor0: program?.fragment ? program.exportsColor0 : true,
          psExportsDepth: program?.exportsDepth ?? false,
        });
      } catch (e) {
        if (e && e.constructor && e.constructor.name === 'XenosRenderStateError') {
          throw new XenosWebGpuError(`draw: ${e.message}`);
        }
        throw e;
      }
      return rtState;
    }

    /**
     * Decode the PA_SC_* scissor snapshot (xenia GetScissor semantics, see
     * render_state.js) into a canvas-clamped rect. Missing snapshot traps — a
     * host that does not attach it must not draw with the guest's scissor
     * silently ignored.
     */
    _resolveScissor(params) {
      const rs = globalThis.XenonGpuRenderState;
      if (!rs) {
        throw new XenosWebGpuError('XenonGpuRenderState is not loaded — public/js/gpu/render_state.js must be loaded before the backend builds a draw');
      }
      try {
        return rs.buildScissorState(params.scissor, {
          width: this.canvas?.width ?? 0,
          height: this.canvas?.height ?? 0,
        });
      } catch (e) {
        if (e && e.constructor && e.constructor.name === 'XenosRenderStateError') {
          throw new XenosWebGpuError(`draw: ${e.message}`);
        }
        throw e;
      }
    }

    /** Begin the render pass for a frame segment: canvas color attachment + depth/stencil attachment. */
    _beginPass(encoder, colorView, rtState, scissor) {
      const colorAttachment = this._frameClearColor
        ? { view: colorView, loadOp: 'load', storeOp: 'store' }
        : { view: colorView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' };
      this._frameClearColor = true;
      const desc = { colorAttachments: [colorAttachment] };
      if (this.depthView) {
        desc.depthStencilAttachment = this._frameClearDepth
          ? {
            view: this.depthView,
            depthLoadOp: 'load',
            depthStoreOp: 'store',
            stencilLoadOp: 'load',
            stencilStoreOp: 'store',
          }
          : {
            view: this.depthView,
            depthLoadOp: 'clear',
            depthClearValue: 1.0,
            depthStoreOp: 'store',
            stencilLoadOp: 'clear',
            stencilClearValue: 0,
            stencilStoreOp: 'store',
          };
        this._frameClearDepth = true;
      }
      const pass = encoder.beginRenderPass(desc);
      if (rtState) {
        pass.setBlendConstant(rtState.blendConstant);
        pass.setStencilReference(rtState.stencilRef);
      }
      if (scissor) {
        pass.setScissorRect(scissor.x, scissor.y, scissor.width, scissor.height);
      }
      return pass;
    }

    /**
     * Fetch or build the { pipeline, bindGroup } for the translated shader program
     * at a given vertex-buffer layout + topology. One pipeline per
     * (vs ucode, ps ucode, layoutKey) — created once, reused for every draw. The
     * bind group is likewise created once per pipeline; the constant buffers it
     * binds are persistent and are re-uploaded only when the guest changes the
     * constants (see _uploadShaderConstants).
     *
     * When no pixel shader is loaded the fragment stage is the fixed bring-up
     * FS (amber; clearly labelled, never the guest's shader). When one is
     * loaded, the translated guest pixel shader runs.
     *
     * Pipelines also key on the render-target register state (write mask,
     * blend, depth/stencil) — the same shader pair drawn with different ROP
     * state gets a different pipeline.
     */
    _getTranslatedProgramResources(vsCode, psCode, plan, topology, program, rtState) {
      let perShader = this.translatedPipelines.get(vsCode);
      if (!perShader) {
        perShader = new Map();
        this.translatedPipelines.set(vsCode, perShader);
      }
      const pairKey = (psCode ? `ps${this._psId(psCode)}` : 'flat') + `|a${alphaTestKey(program)}g${convertKey(program)}c${colorExportKey(program)}`;
      const rtKey = rtState ? rtState.key : 'rt:none';
      const layoutKey = `${pairKey}|${topology}:${this.format}:${rtKey}:` + JSON.stringify((plan?.descriptors ?? []).map((d) => [d.arrayStride, d.attributes]));
      let entry = perShader.get(layoutKey);
      if (entry) return entry;

      if (!program) program = this._translateProgram(vsCode, psCode);
      const vsModule = this.device.createShaderModule({ code: program.vertex.wgsl, label: 'xenos-translated-vs' });
      const fsModule = this.device.createShaderModule({
        code: program.fragment ? program.fragment.wgsl : FLAT_FRAGMENT_WGSL,
        label: program.fragment ? 'xenos-translated-ps' : 'xenos-bringup-fs',
      });
      const vertexBuffers = (plan?.descriptors ?? []).map((d) => ({
        arrayStride: d.arrayStride,
        stepMode: d.stepMode ?? 'vertex',
        attributes: d.attributes.map((a) => ({ shaderLocation: a.shaderLocation, offset: a.offset, format: a.format })),
      }));
      const pipeline = this.device.createRenderPipeline({
        label: program.fragment ? 'xenos-translated-vs-ps' : 'xenos-translated-vs',
        layout: 'auto',
        vertex: { module: vsModule, entryPoint: 'vs_main', buffers: vertexBuffers },
        fragment: { module: fsModule, entryPoint: 'fs_main', targets: [this._colorTargetDesc(rtState, colorExportKey(program) === '1')] },
        depthStencil: this._depthStencilDesc(rtState),
        primitive: { topology },
      });
      let bindGroup = null;
      if (program.usesFloatConstants || program.usesBoolConstants || program.alphaTest) {
        const entries = [];
        if (program.usesFloatConstants) entries.push({ binding: 0, resource: { buffer: this.aluConstantBuffer } });
        if (program.usesBoolConstants) entries.push({ binding: program.usesFloatConstants ? 1 : 0, resource: { buffer: this.boolConstantBuffer } });
        if (program.alphaTest) entries.push({ binding: program.alphaTest.binding, resource: { buffer: this.alphaTestBuffer } });
        bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      }
      entry = { pipeline, bindGroup };
      perShader.set(layoutKey, entry);
      return entry;
    }

    /** Placeholder pipeline whose vertex stage consumes the plan's attributes (fetch bring-up). */
    _getFetchPlaceholderPipeline(descriptors, rtState) {
      const key = 'fetch:' + JSON.stringify(descriptors.map((d) => [d.arrayStride, d.attributes])) + ':' + this.format + ':' + (rtState ? rtState.key : 'rt:none');
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
        fragment: { module, entryPoint: 'fs_main', targets: [this._colorTargetDesc(rtState)] },
        depthStencil: this._depthStencilDesc(rtState),
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

    _getPlaceholderPipeline(rtState) {
      const key = 'full-canvas:' + this.format + ':' + (rtState ? rtState.key : 'rt:none');
      let p = this.pipelineCache.get(key);
      if (p) return p;
      const module = this.device.createShaderModule({ code: PLACEHOLDER_WGSL, label: 'placeholder' });
      p = this.device.createRenderPipeline({
        label: 'xenos-placeholder-triangle',
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_main' },
        fragment: { module, entryPoint: 'fs_main', targets: [this._colorTargetDesc(rtState)] },
        depthStencil: this._depthStencilDesc(rtState),
        primitive: { topology: 'triangle-list' },
      });
      this.pipelineCache.set(key, p);
      return p;
    }

    draw(params) {
      if (!this.device) throw new XenosWebGpuError('draw before attach()');
      if (!this.canvas) throw new XenosWebGpuError('draw requires a canvas (headless rendering into textures is not wired yet)');
      this.uploads.frames++;
      if (!this.placeholderShader) {
        // --- the real path: translated guest shaders ------------------------------------------------
        // Both stages execute the guest's microcode (translated to WGSL). When no pixel shader is
        // loaded the fragment stage is a fixed bring-up FS (amber, labelled, never guest state).
        const code = this.shaders.vertex?.code ?? null;
        if (!code) {
          throw new XenosWebGpuError(
            `draw(prim=${params.primTypeName ?? params.primType}, indices=${params.numIndices}): no vertex shader is loaded (IM_LOAD/IM_LOAD_IMMEDIATE with shader type 0 first) — ` +
            'refusing to fabricate geometry. Pass { placeholderShader: true } to attach() to render the bring-up triangle instead.');
        }
        // A pixel shader, when loaded, IS translated and run — no guessed fragment stage anywhere.
        const psCode = this.shaders.pixel?.code ?? null;
        const topology = topologyForPrimType(params.primType ?? 0x04, params.primTypeName);
        const alphaTest = globalThis.XenonGpuRenderState?.decodeAlphaTest(params.renderTargets?.colorControl ?? 0) ?? null;
        // format nibble of RB_COLOR_INFO at bits 16-19 (1 = k_8_8_8_8_GAMMA);
        // read the field directly — the full decode runs (memoized) right after.
        const convertColor0ToGamma = (((params.renderTargets?.colorInfo0 ?? 0) >>> 16) & 0xF) === 1;
        const program = this._translateProgram(code, psCode, alphaTest, convertColor0ToGamma);

        // The fetch plan is memoized per (code, fetch-constants snapshot); the upload path is content-addressed.
        if (!params.vertexFetchConstants && program.vertex.inputLocations.length > 0) {
          throw new XenosWebGpuError(
            `draw: the translated vertex shader reads attributes at locations [${program.vertex.inputLocations}] but the packet carried no fetch constants — ` +
            'the command processor normally attaches them; refusing to bind fabricated attributes');
        }
        const plan = params.vertexFetchConstants
          ? this.prepareVertexFetch({ vertexShaderCode: code, fetchConstants: params.vertexFetchConstants })
          : null;
        // The translated VS declares vec4f inputs; pure uint/sint attribute formats cannot feed a
        // vec4<f32> location (WebGPU interface matching) — trap loudly instead of erroring inside wgpu.
        if (plan) {
          const wanted = new Set(program.vertex.inputLocations);
          for (const d of plan.descriptors) {
            for (const a of d.attributes) {
              if (wanted.has(a.shaderLocation) && isIntegerGpuVertexFormat(a.format)) {
                throw new XenosWebGpuError(
                  `draw: attribute at location ${a.shaderLocation} (fetch constant ${d.fetchConstant}) has integer GPU format '${a.format}' — ` +
                  'the translated vertex shader declares vec4f inputs (Xenos ucode carries no format information) and WebGPU forbids uint/sint ' +
                  'attributes at a vec4<f32> location. Fetch this attribute with the sign/normalization bit set (unorm/snorm) or as a float format.');
              }
            }
          }
        }
        const buffers = plan ? this.uploadVertexBuffers(plan) : [];
        this._uploadShaderConstants(params, program);
        this._uploadAlphaTest(params, program);
        const rtState = this._resolveRenderTargets(params, program);
        const scissor = this._resolveScissor(params);
        const { pipeline, bindGroup } = this._getTranslatedProgramResources(code, psCode, plan, topology, program, rtState);
        const encoder = this.device.createCommandEncoder({ label: 'xenos-frame' });
        const colorView = this.context.getCurrentTexture().createView();
        const pass = this._beginPass(encoder, colorView, rtState, scissor);
        pass.setPipeline(pipeline);
        if (bindGroup) pass.setBindGroup(0, bindGroup);
        for (const b of buffers) pass.setVertexBuffer(b.bindingIndex, b.buffer);
        this._setIndexBufferAndDraw(pass, params);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        return;
      }
      // --- clearly-labelled bring-up path ---------------------------------------------------------
      // When a vertex shader is loaded AND the draw carries fetch constants (the command
      // processor attaches them), the bring-up goes through the REAL vertex fetch path:
      // analyze → descriptors → upload → setVertexBuffer → generated placeholder pipeline.
      // Any failure there is loud (never falls back to a fabricated frame). Without a
      // vertex shader we cannot fetch anything — the full-canvas triangle remains.
      const fetchReady = !!(this.shaders.vertex && params.vertexFetchConstants);
      if (fetchReady) {
        const rtState = this._resolveRenderTargets(params, null);
        const scissor = this._resolveScissor(params);
        const plan = this.prepareVertexFetch({ vertexShaderCode: this.shaders.vertex.code, fetchConstants: params.vertexFetchConstants });
        const encoder = this.device.createCommandEncoder({ label: 'xenos-frame' });
        const colorView = this.context.getCurrentTexture().createView();
        const pass = this._beginPass(encoder, colorView, rtState, scissor);
        if (plan.descriptors.length > 0) {
          const buffers = this.uploadVertexBuffers(plan);
          pass.setPipeline((plan._pipeline ??= this._getFetchPlaceholderPipeline(plan.descriptors, rtState)));
          for (const b of buffers) pass.setVertexBuffer(b.bindingIndex, b.buffer);
          this._setIndexBufferAndDraw(pass, params);
        } else {
          // VS loaded but it fetches nothing — draw the full-canvas placeholder (documented).
          this.log('info', 'draw: vertex shader fetches nothing — issuing the plain bring-up triangle');
          pass.setPipeline(this._getPlaceholderPipeline(rtState));
          pass.draw(3);
        }
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        return;
      }
      if (this.shaders.vertex) {
        this.log('warn', 'draw: vertex shader loaded but the packet carried no fetch constants — bypassing the fetch path (full-canvas placeholder)');
      }
      const rtState = this._resolveRenderTargets(params, null);
      const scissor = this._resolveScissor(params);
      const encoder = this.device.createCommandEncoder({ label: 'xenos-frame' });
      const colorView = this.context.getCurrentTexture().createView();
      const pass = this._beginPass(encoder, colorView, rtState, scissor);
      pass.setPipeline(this._getPlaceholderPipeline(rtState));
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
      this._frameClearColor = false;
      this._frameClearDepth = false;
      this.lastFrame = { at: (typeof performance !== 'undefined' ? performance.now() : Date.now()) };
      this.log('debug', `swap (frame ${this.uploads.frames})`);
    }

    /** Call after the last frame of a session to release GPU resources. */
    destroy() {
      if (this.depthTexture) { this.depthTexture.destroy(); this.depthTexture = null; this.depthView = null; }
      if (this.aluConstantBuffer) { this.aluConstantBuffer.destroy(); this.aluConstantBuffer = null; }
      if (this.boolConstantBuffer) { this.boolConstantBuffer.destroy(); this.boolConstantBuffer = null; }
      if (this.device) { this.device.destroy(); this.device = null; }
      this.context = null;
      this.pipelineCache.clear();
      this.programMemo = new WeakMap();
      this.translatedPipelines = new WeakMap();
    }
  }

  // Internals exposed for unit tests only — unstable shapes, not public API.
  const _internal = { topologyForPrimType, isIntegerGpuVertexFormat, FLAT_FRAGMENT_WGSL, PRIM_TYPE_TOPOLOGY };

  const api = { XenosWebGpuBackend, XenosWebGpuError, PLACEHOLDER_WGSL, _internal };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonWebGpuBackend = api;
})(typeof self !== 'undefined' ? self : globalThis);
