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
  /** Cache-key fragment: color_exp_bias exponent (baked as a literal into the WGSL). */
  function expBiasKey(program) { return program && Number.isFinite(program.colorExpBias) ? String(program.colorExpBias | 0) : '0'; }
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

  // The depth-only fragment stage (xenia's RB_MODECONTROL.edram_mode = kDepth):
  // no guest pixel shader runs, nothing is written to the color attachment
  // (the pipeline's color target carries a zero write mask) and the depth
  // comes from the rasterizer — the stage exists only because WebGPU requires
  // a fragment stage for a pipeline that is used in a pass with a color
  // attachment (all the draws of a frame share one pass).
  const DEPTH_ONLY_FRAGMENT_WGSL = `// Depth-only draw (xenia kDepth mode) — no guest pixel shader, no color output.
@fragment fn fs_main() {
}
`;

  /**
   * The present blit: a fullscreen triangle that copies the offscreen
   * render target into the canvas texture. The guest's render-target format
   * is not the canvas format (8-bit unorm RGBA), so the frame is drawn into a
   * texture with the guest's format and copied here — textureLoad, so the
   * copy is exact (no filtering, no colour-space conversion beyond what the
   * canvas format does on write).
   */
  const PRESENT_BLIT_WGSL = `
@group(0) @binding(0) var src: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  // One oversized triangle covering the whole target.
  var pos = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(pos[index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return textureLoad(src, vec2i(pos.xy), 0);
}
`;

  /** Integer-component GPUVertexFormat strings (need vec4<u32>/vec4<i32> shader locations, not vec4<f32>). */
  function isIntegerGpuVertexFormat(format) {
    return format.startsWith('uint') || format.startsWith('sint');
  }

  // Xenos VGT_DRAW_INITIATOR prim_type (pm4.js PRIMITIVE_TYPE) → GPUPrimitiveTopology.
  // WebGPU cannot represent fan/loop/quad connectivity; those three are
  // converted host-side with index expansion (see PRIM_TYPE_CONVERT) — the
  // guest's geometry is always drawn, never silently skipped or misdrawn.
  // The remaining unencodable values (kTriangleWithWFlags 0x07, kQuadStrip
  // 0x0E, kPolygon 0x0F, kRectangleList 0x08 and the 2D copies) trap loudly —
  // xenia's primitive processor refuses them too (they need a geometry
  // shader / vertex-shader expansion, which is not implemented yet).
  const PRIM_TYPE_TOPOLOGY = new Map([
    [0x01 /* kPointList */, 'point-list'],
    [0x02 /* kLineList */, 'line-list'],
    [0x03 /* kLineStrip */, 'line-strip'],
    [0x04 /* kTriangleList */, 'triangle-list'],
    [0x06 /* kTriangleStrip */, 'triangle-strip'],
  ]);
  // Host-side index conversion for the primitives WebGPU has no topology for.
  // Mirrors xenia's primitive_processor.cc conversion paths (used there when
  // the host lacks the primitive natively — WebGPU always lacks these):
  //   kTriangleFan → triangle-list: triangles (i-1, i, i0) — xenia's
  //                  TriangleFanToList order: the previous index, the current
  //                  one and the fan's first index close every triangle.
  //   kLineLoop    → line-strip with the loop closed (first index re-appended)
  //   kQuadList    → triangle-list: (v0,v1,v2), (v0,v2,v3) per quad (xenia order)
  // Leftover indices that do not complete a primitive are dropped, matching
  // real hardware's primitive assembly. Fan/loop/quad draws are polygonal
  // (fan/quads) or lines (loop) exactly as before the conversion.
  const PRIM_TYPE_CONVERT = new Map([
    [0x05 /* kTriangleFan */, { topology: 'triangle-list', kind: 'fan' }],
    [0x0C /* kLineLoop */, { topology: 'line-strip', kind: 'loop' }],
    [0x0D /* kQuadList */, { topology: 'triangle-list', kind: 'quads' }],
  ]);
  // The primitive types for which xenia's primitive processor honors the
  // guest's primitive reset index (VGT_MULTI_PRIM_IB_RESET_INDX). The list
  // topologies are deliberately excluded: "Vulkan explicitly disallows
  // primitive restart index for 'list' topologies. In Direct3D 12, it's valid
  // for non-strips, but has implementation-defined behavior." (xenia
  // primitive_processor.cc).
  const PRIM_TYPE_RESETTABLE = new Set([
    0x03 /* kLineStrip */, 0x05 /* kTriangleFan */, 0x06 /* kTriangleStrip */,
    0x0C /* kLineLoop */, 0x0E /* kQuadStrip */, 0x0F /* kPolygon */,
    0x15 /* k2DLineStrip */, 0x16 /* k2DTriStrip */,
  ]);
  // The resettable topologies WebGPU has natively. The guest's reset index
  // terminates the strip, so every range between two reset markers is a strip
  // of its own — xenia hands the whole buffer to the host with the host's
  // primitive restart enabled; WebGPU's restart index is fixed at
  // 0xFFFF/0xFFFFFFFF, so the same result is produced by drawing each range
  // separately. The remaining resettable topologies (quad strip, polygon, the
  // 2D copies) are not encodable in WebGPU at all and trap before this point.
  const PRIM_TYPE_STRIP_RESETTABLE = new Set([
    0x03 /* kLineStrip */, 0x06 /* kTriangleStrip */,
  ]);
  // Split an index buffer at the guest's primitive reset index; the marker
  // itself is dropped and every range is drawn on its own. Mirrors xenia's
  // GetMultiPrimitiveHostIndexCountAndRanges range gathering.
  function splitResetRanges(indices, isResetIndex) {
    const ranges = [];
    let start = 0;
    for (let i = 0; i < indices.length; i++) {
      if (!isResetIndex(indices[i])) continue;
      if (i > start) ranges.push([start, i - start]);
      start = i + 1;
    }
    if (start < indices.length) ranges.push([start, indices.length - start]);
    return ranges;
  }
  // Expand a decoded (little-endian, in-register typed array) index buffer for
  // a converted primitive type. Returns null when there is nothing to draw.
  // When isResetIndex is given, the index buffer is split at the guest's
  // primitive reset index and every range is converted on its own — xenia's
  // GetMultiPrimitiveHostIndexCountAndRanges + ConvertSinglePrimitiveRanges.
  function expandConvertedIndices(indices, kind, isResetIndex = null) {
    const n = indices.length;
    // xenia's index-count getters: GetTriangleFanListIndexCount,
    // GetLineLoopStripIndexCount, GetQuadListTriangleListIndexCount.
    const convertedCount = (count) => {
      if (kind === 'fan') return count > 2 ? (count - 2) * 3 : 0;
      if (kind === 'loop') return count > 1 ? count + 1 : 0;
      return Math.floor(count / 4) * 6;
    };
    const convertRange = (src, count) => {
      const out = new src.constructor(convertedCount(count));
      // xenia's converters bail out when the range cannot complete a primitive
      // ("if (source_index_count <= 2) return;" / "<= 1" / a whole number of
      // quads) — the destination stays empty and the range contributes nothing.
      if (out.length === 0) return out;
      if (kind === 'fan') {
        // xenia's TriangleFanToList (primitive_processor.h): every triangle is
        // (previous, current, first) — the fan's first index closes each
        // triangle. The winding matches the hardware's fan assembly.
        let o = 0;
        for (let i = 2; i < count; i++) { out[o++] = src[i - 1]; out[o++] = src[i]; out[o++] = src[0]; }
        return out;
      }
      if (kind === 'loop') {
        // xenia's LineLoopToStrip: the source, with the first index re-appended.
        out.set(src.subarray(0, count), 0);
        out[count] = src[0];
        return out;
      }
      // quads: two triangles per quad, xenia's order (v0,v1,v2), (v0,v2,v3)
      let o = 0;
      for (let q = 0; q < Math.floor(count / 4); q++) {
        const b = q * 4, i0 = src[b], i1 = src[b + 1], i2 = src[b + 2], i3 = src[b + 3];
        out[o++] = i0; out[o++] = i1; out[o++] = i2;
        out[o++] = i0; out[o++] = i2; out[o++] = i3;
      }
      return out;
    };
    if (!isResetIndex) {
      if (convertedCount(n) === 0) return null;
      return convertRange(indices, n);
    }
    // Split at the reset index; the reset value itself is dropped.
    const ranges = [];
    let start = 0;
    for (let i = 0; i < n; i++) {
      if (isResetIndex(indices[i])) { ranges.push([start, i - start]); start = i + 1; }
    }
    ranges.push([start, n - start]);
    let total = 0;
    for (const [, count] of ranges) total += convertedCount(count);
    if (total === 0) return null;
    const out = new indices.constructor(total);
    let o = 0;
    for (const [off, count] of ranges) {
      const part = convertRange(indices.subarray(off, off + count), count);
      out.set(part, o);
      o += part.length;
    }
    return out;
  }
  // xenia draw_util::IsPrimitivePolygonal: triangle-list/fan/strip/W-flags, quad-list/strip,
  // polygon — these have faces that culling/wireframe/polygon-offset apply to.
  function isPrimitivePolygonal(primType) {
    const t = (primType ?? 0) & 0x3f;
    return (t >= 0x04 && t <= 0x07) || t === 0x0d || t === 0x0e || t === 0x0f;
  }

  function topologyForPrimType(primType, primTypeName) {
    const convert = PRIM_TYPE_CONVERT.get((primType ?? 0) & 0x3f);
    if (convert) return convert.topology; // host-side index expansion — see _setIndexBufferAndDraw
    const topology = PRIM_TYPE_TOPOLOGY.get(primType & 0x3f);
    if (!topology) {
      // Called at draw time, long after XenosWebGpuError (declared below) is initialized.
      throw new XenosWebGpuError(`prim ${primTypeName ?? `prim_0x${(primType & 0x3f).toString(16)}`}: primitive topology is not encodable in WebGPU (only kTriangleFan/kLineLoop/kQuadList are converted host-side; the rest — e.g. kTriangleWithWFlags, kQuadStrip, kPolygon, kRectangleList — need shader expansion and are refused by xenia's primitive processor as well) — refusing to draw with the wrong connectivity`);
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
      this._frame = null; // { encoder, pass, colorView, draws } while a frame is being built
      this.viewportBuffer = null;   // persistent 32-byte uniform (scale xyz / offset xyz), content-compared
      this._viewportLast = null;
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
      this.uploads = { indexBuffers: 0, indexBytes: 0, vertexBuffers: 0, vertexBytes: 0, frames: 0, vertexBufferCacheHits: 0, indexBufferCacheHits: 0, constantUploads: 0, presents: 0 };
      this._colorTargets = new Map(); // `${format}:${samples}:${w}x${h}` -> { texture, format, sampleCount }
      this._presentPending = false; // an offscreen frame is waiting to be blitted to the canvas
      this._presentPipeline = new Map(); // source format -> { pipeline, bindGroup } for the present blit
      this.implicitIndexCache = new Map();
      this.implicitIndexBytes = 0;
      this.implicitIndexClock = 0;
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
      this.depthSampleCount = 1;
      this.depthAttachmentFormat = this.depthFormat;
      this.depthTexture = this.device.createTexture({
        size: [width, height, 1],
        format: this.depthFormat,
        sampleCount: 1,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        label: 'xenos-depth',
      });
      this.depthView = typeof this.depthTexture.createView === 'function' ? this.depthTexture.createView() : null;
    }

    /**
     * The depth attachment must have the frame's sample count (a WebGPU
     * render pass requires the depth attachment's sample count to match the
     * color attachment's), so the depth texture is recreated when the guest
     * switches between 1x and multisampled rendering.
     */
    _ensureDepthTexture(sampleCount, format = this.depthFormat) {
      if (this.depthTexture && this.depthSampleCount === sampleCount && this.depthAttachmentFormat === format) return;
      if (!this.device || !this.canvas) return;
      const width = Math.max(1, this.canvas.width | 0);
      const height = Math.max(1, this.canvas.height | 0);
      if (this.depthTexture) this.depthTexture.destroy();
      this.depthSampleCount = sampleCount;
      this.depthAttachmentFormat = format;
      this._frameClearDepth = false; // the new attachment starts undefined
      this.depthTexture = this.device.createTexture({
        size: [width, height, 1],
        format,
        sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        label: 'xenos-depth',
      });
      this.depthView = typeof this.depthTexture.createView === 'function' ? this.depthTexture.createView() : null;
    }

    /** Release every color target texture (resize / destroy). */
    _destroyColorTargets() {
      for (const entry of this._colorTargets.values()) entry.texture.destroy();
      this._colorTargets.clear();
      this._presentPipeline.clear();
      this._presentPending = false;
      this._presentKey = null;
    }

    /** Resize depth attachment to the canvas size (call before drawing after a resize). */
    resize() {
      this._flushFrame();
      // The color targets are sized like the canvas (the guest's RT extent is
      // assumed to be the canvas extent) — drop them so they are recreated.
      this._destroyColorTargets();
      this._recreateDepth();
    }

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

    vizQuery(state) {
      // Occlusion queries need a querySet + resolve + guest-memory writeback — none of which exists yet (step 9).
      // xenia issues real D3D queries here; silently no-op'ing would leave the guest polling a status that never lands.
      throw new XenosWebGpuError(`PM4_VIZ_QUERY '${state}' is not supported yet — the occlusion-query path (querySet, resolve, guest-memory writeback) is step 9; refusing to silently no-op`);
    }

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
    _translateProgram(vsCode, psCode, alphaTest, convertColor0ToGamma = false, colorExpBias = 0) {
      const atKey = alphaTest ? `at${alphaTest.compare}` : 'at-';
      const gKey = convertColor0ToGamma ? 'g1' : 'g0';
      const bKey = `eb${Math.max(-32, Math.min(31, colorExpBias | 0))}`;
      const pairKey = (psCode ? `ps${this._psId(psCode)}` : 'flat') + '|' + atKey + '|' + gKey + '|' + bKey;
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
        pair = TX.translateShaderProgram(vsCode, psCode, { alphaTest: alphaTest ?? undefined, convertColor0ToGamma, colorExpBias });
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
      // The target format is the guest's render-target format (render_state.js
      // maps it exactly like xenia maps it to a host format); null means the
      // canvas format (8-bit unorm RGBA — k_8_8_8_8 / k_8_8_8_8_GAMMA).
      const format = (rtState && rtState.colorFormat) ? rtState.colorFormat : this.format;
      const target = { format, writeMask: writesColor ? (rtState ? rtState.writeMask : 0xF) : 0x0 };
      if (writesColor && rtState && rtState.blend) target.blend = rtState.blend;
      return target;
    }

    /** The depth-stencil state for every pipeline this backend builds. */
    _depthStencilDesc(rtState) {
      if (rtState && rtState.depthStencil) {
        const desc = { ...rtState.depthStencil };
        if (rtState.raster && (rtState.raster.depthBias !== 0 || rtState.raster.depthBiasSlopeScale !== 0)) {
          desc.depthBias = rtState.raster.depthBias;
          desc.depthBiasSlopeScale = rtState.raster.depthBiasSlopeScale;
          desc.depthBiasClamp = rtState.raster.depthBiasClamp;
        }
        return desc;
      }
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
          psExportsColor0: params.depthOnly ? false : (program?.fragment ? program.exportsColor0 : true),
          psExportsDepth: program?.exportsDepth ?? false,
          // xenia IsPrimitivePolygonal — culling/wireframe/polygon offset only for polygonal prims.
          primitivePolygonal: isPrimitivePolygonal(params.primType),
        });
      } catch (e) {
        if (e && e.constructor && e.constructor.name === 'XenosRenderStateError') {
          throw new XenosWebGpuError(`draw: ${e.message}`);
        }
        throw e;
      }
      if (rtState.raster && rtState.raster.bothCull) {
        // xenia: both faces culled → rasterization disabled; the draw produces no fragments.
        // (A memexporting VS would still run — memory exports trap in this port — so dropping is exact.)
        this.log('info', `draw(prim=${params.primTypeName ?? params.primType}): both faces culled (PA_SU_SC_MODE_CNTL cull_front+cull_back) — no fragments are produced; draw dropped`);
        return null;
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
    // Per-frame batched encoder + pass: draws accumulate; swap() (or 256 pending draws,
    // or resize()/destroy()) ends the pass and submits once. Scissor/blend/stencil state
    // is applied per draw INSIDE the open pass (all dynamic pass commands). A trap in any
    // draw drops the pending frame unsubmitted — no partial frame reaches the canvas.
    static MAX_DRAWS_PER_SUBMIT = 256;

    _dropFrame() {
      this._frame = null; // encoder + pass are garbage; never submitted
      this._frameClearColor = false;
      this._frameClearDepth = false;
    }

    _flushFrame() {
      if (!this._frame) return;
      const { encoder, pass, offscreen, presentKey } = this._frame;
      this._frame = null; // release before submit so a re-entrant draw starts a fresh frame
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      // A frame that rendered into the offscreen target has not reached the
      // canvas yet — swap() presents it.
      if (offscreen) {
        this._presentPending = true;
        this._presentKey = presentKey;
      }
    }

    /**
     * The color attachment for the frame's pass. The guest's render-target
     * format decides where the frame is drawn:
     *   - k_8_8_8_8 / k_8_8_8_8_GAMMA (8-bit unorm RGBA) go straight into the
     *     canvas texture — the same format the canvas is.
     *   - every other representable format goes into an offscreen texture with
     *     exactly that format, so the guest's writes are quantized and ranged
     *     like the hardware's; the offscreen target is presented to the canvas
     *     by a blit at swap() (_presentOffscreen).
     */
    _colorAttachment(rtState) {
      const format = rtState?.colorFormat ?? null;   // null = the canvas format
      const samples = rtState?.sampleCount ?? 1;
      // The single-sample target: the canvas texture, or an offscreen texture
      // with the guest's format (the canvas cannot carry it).
      let single = null;
      if (format) {
        single = this._colorTargetTexture(format, 1);
      } else {
        single = { texture: this.context.getCurrentTexture(), format: this.format, sampleCount: 1, key: null };
      }
      if (samples === 1) {
        return { view: single.texture.createView(), format: single.format, sampleCount: 1, offscreen: format !== null, resolveTarget: null, sourceKey: single.key ?? null };
      }
      // Multisampled: the frame renders into a multisampled attachment of the
      // same format, and WebGPU's resolve (a box filter over the samples)
      // writes the single-sample target — the same box filter xenia's EDRAM
      // resolve applies for "all samples".
      const msaa = this._colorTargetTexture(single.format, samples);
      return {
        view: msaa.texture.createView(),
        format: single.format,
        sampleCount: samples,
        offscreen: format !== null,
        resolveTarget: single.texture.createView(),
        sourceKey: single.key ?? null,
      };
    }

    /**
     * A color target texture: the guest's render-target format (offscreen) or
     * the canvas format, single- or multisampled. Keyed by format, sample
     * count and size (the guest's RT extent is assumed to be the canvas
     * extent, like the viewport decode assumes).
     */
    _colorTargetTexture(format, sampleCount) {
      const width = Math.max(1, (this.canvas?.width ?? 1) | 0);
      const height = Math.max(1, (this.canvas?.height ?? 1) | 0);
      const key = `${format}:${sampleCount}:${width}x${height}`;
      const existing = this._colorTargets.get(key);
      if (existing) return existing;
      if (typeof GPUTextureUsage === 'undefined') throw new XenosWebGpuError('GPUTextureUsage is not defined in this environment');
      const texture = this.device.createTexture({
        label: sampleCount > 1 ? `xenos-msaa-${format}` : `xenos-rt-${format}`,
        size: [width, height, 1],
        format,
        sampleCount,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | (sampleCount > 1 ? 0 : GPUTextureUsage.TEXTURE_BINDING),
      });
      const entry = { texture, format, sampleCount, key };
      this._colorTargets.set(key, entry);
      return entry;
    }

    /** Open (or reuse) the batched frame's pass. Returns the pass — never ends it per draw. */
    _openFrame(rtState = null) {
      if (this._frame && this._frame.draws >= XenosWebGpuBackend.MAX_DRAWS_PER_SUBMIT) this._flushFrame();
      // A render pass has one color attachment format: if the guest changed the
      // render-target format mid-frame, end the pass and start a new one. The
      // old frame's canvas update is dropped (its offscreen content is
      // superseded by the new frame — a guest that changes the render-target
      // format mid-frame is not a shape this port reproduces faithfully).
      const wantFormat = (rtState?.colorFormat) ?? this.format;
      const wantSamples = rtState?.sampleCount ?? 1;
      if (this._frame && (this._frame.colorFormat !== wantFormat || this._frame.sampleCount !== wantSamples)) {
        this._flushFrame();
        this._presentPending = false;
        this._frameClearColor = false;
        this._frameClearDepth = false;
      }
      if (!this._frame) {
        const encoder = this.device.createCommandEncoder({ label: 'xenos-frame' });
        const attachment = this._colorAttachment(rtState);
        this._ensureDepthTexture(attachment.sampleCount, rtState?.depthStencil?.format ?? this.depthFormat);
        const pass = this._beginFramePass(encoder, attachment);
        this._frame = {
          encoder, pass, colorView: attachment.view, draws: 0,
          colorFormat: attachment.format, sampleCount: attachment.sampleCount, offscreen: attachment.offscreen,
          presentKey: attachment.sourceKey,
        };
      }
      return this._frame.pass;
    }

    /** The frame pass descriptor: defined image on the first pass of a frame, load after. */
    _beginFramePass(encoder, attachment) {
      const colorView = attachment.view;
      // With MSAA the frame renders into the multisampled attachment and the
      // resolve writes the single-sample target — the multisample content
      // itself is not needed afterwards.
      const colorAttachment = this._frameClearColor
        ? { view: colorView, loadOp: 'load', storeOp: attachment.resolveTarget ? 'discard' : 'store', resolveTarget: attachment.resolveTarget ?? undefined }
        : { view: colorView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: attachment.resolveTarget ? 'discard' : 'store', resolveTarget: attachment.resolveTarget ?? undefined };
      this._frameClearColor = true;
      const desc = { colorAttachments: [colorAttachment] };
      if (this.depthView) {
        // A float depth attachment (depth32float) has no stencil plane — the
        // stencil load/store ops must be omitted for it.
        const hasStencil = this.depthAttachmentFormat !== 'depth32float';
        desc.depthStencilAttachment = this._frameClearDepth
          ? {
            view: this.depthView,
            depthLoadOp: 'load',
            depthStoreOp: 'store',
            ...(hasStencil ? { stencilLoadOp: 'load', stencilStoreOp: 'store' } : {}),
          }
          : {
            view: this.depthView,
            depthLoadOp: 'clear',
            depthClearValue: 1.0,
            depthStoreOp: 'store',
            ...(hasStencil ? { stencilLoadOp: 'clear', stencilClearValue: 0, stencilStoreOp: 'store' } : {}),
          };
        this._frameClearDepth = true;
      }
      return encoder.beginRenderPass(desc);
    }

    /** Per-draw dynamic state inside the open frame pass (blend constant, stencil ref, scissor). */
    _frameDrawState(pass, rtState, scissor) {
      if (rtState) {
        pass.setBlendConstant(rtState.blendConstant);
        pass.setStencilReference(rtState.stencilRef);
      }
      if (scissor) {
        pass.setScissorRect(scissor.x, scissor.y, scissor.width, scissor.height);
      }
    }

    /** Count the draw against the batch; flushes the frame if the batch is full. */
    _finishDraw() {
      if (!this._frame) return;
      this._frame.draws++;
      if (this._frame.draws >= XenosWebGpuBackend.MAX_DRAWS_PER_SUBMIT) this._flushFrame();
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
    _getTranslatedProgramResources(vsCode, psCode, plan, topology, program, rtState, depthOnly = false) {
      let perShader = this.translatedPipelines.get(vsCode);
      if (!perShader) {
        perShader = new Map();
        this.translatedPipelines.set(vsCode, perShader);
      }
      const pairKey = (psCode ? `ps${this._psId(psCode)}` : 'flat') + `|a${alphaTestKey(program)}g${convertKey(program)}c${colorExportKey(program)}eb${expBiasKey(program)}` + (depthOnly ? '|depth' : '');
      const rtKey = rtState ? rtState.key : 'rt:none';
      const layoutKey = `${pairKey}|${topology}:${this.format}:${rtKey}:` + JSON.stringify((plan?.descriptors ?? []).map((d) => [d.arrayStride, d.attributes]));
      let entry = perShader.get(layoutKey);
      if (entry) return entry;

      if (!program) program = this._translateProgram(vsCode, psCode);
      const vsModule = this.device.createShaderModule({ code: program.vertex.wgsl, label: 'xenos-translated-vs' });
      // xenia's kDepth mode runs no guest pixel shader at all (the depth comes
      // from the rasterizer): the pipeline gets a stage that writes nothing,
      // and the color target carries a zero write mask.
      const fsModule = this.device.createShaderModule({
        code: depthOnly ? DEPTH_ONLY_FRAGMENT_WGSL : (program.fragment ? program.fragment.wgsl : FLAT_FRAGMENT_WGSL),
        label: depthOnly ? 'xenos-depth-only-fs' : (program.fragment ? 'xenos-translated-ps' : 'xenos-bringup-fs'),
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
        fragment: {
          module: fsModule,
          entryPoint: 'fs_main',
          targets: [this._colorTargetDesc(rtState, depthOnly ? false : colorExportKey(program) === '1')],
        },
        depthStencil: this._depthStencilDesc(rtState),
        multisample: { count: rtState?.sampleCount ?? 1 },
        primitive: { topology, cullMode: rtState.raster.cullMode, frontFace: rtState.raster.frontFace },
      });
      let bindGroup = null;
      {
        const entries = [];
        if (program.usesFloatConstants) entries.push({ binding: 0, resource: { buffer: this.aluConstantBuffer } });
        if (program.usesBoolConstants) entries.push({ binding: program.usesFloatConstants ? 1 : 0, resource: { buffer: this.boolConstantBuffer } });
        if (program.alphaTest) entries.push({ binding: program.alphaTest.binding, resource: { buffer: this.alphaTestBuffer } });
        if (program.viewportBinding !== undefined) entries.push({ binding: program.viewportBinding, resource: { buffer: this.viewportBuffer } });
        bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      }
      entry = { pipeline, bindGroup };
      perShader.set(layoutKey, entry);
      return entry;
    }

    /**
     * Resolve the guest viewport (PA_CL_VTE_CNTL / PA_CL_CLIP_CNTL / PA_CL_VPORT_*) into
     * the coefficients the translated vertex shader applies. WebGPU has no programmable
     * viewport — see decodeViewportState() for the exact math (xenia's formula).
     * Throws XenosWebGpuError on missing prerequisites or unsupported clip state (UCP).
     */
    _resolveViewport(params) {
      const rs = globalThis.XenonGpuRenderState;
      if (!rs) {
        throw new XenosWebGpuError('XenonGpuRenderState is not loaded — public/js/gpu/render_state.js must be loaded before the backend builds a draw');
      }
      try {
        return rs.decodeViewportState(params.renderTargets, params.scissor, this.canvas ?? { width: 1, height: 1 });
      } catch (e) {
        if (e && e.constructor && e.constructor.name === 'XenosRenderStateError') {
          throw new XenosWebGpuError(`draw: ${e.message}`);
        }
        throw e;
      }
    }

    /**
     * Upload the composed viewport coefficients into a persistent 32-byte uniform.
     * Skipped entirely while the six floats are unchanged (content compare — the hash
     * costs compares, the write costs a queue submission).
     */
    _uploadViewport(viewport) {
      if (typeof GPUBufferUsage === 'undefined') throw new XenosWebGpuError('_uploadViewport: GPUBufferUsage is not defined in this environment');
      const sx = viewport.scale[0], sy = viewport.scale[1], sz = viewport.scale[2];
      const ox = viewport.offset[0], oy = viewport.offset[1], oz = viewport.offset[2];
      const last = this._viewportLast;
      if (last && last[0] === sx && last[1] === sy && last[2] === sz && last[3] === ox && last[4] === oy && last[5] === oz) {
        return;
      }
      const bytes = new Uint8Array(32);
      const f = new Float32Array(bytes.buffer);
      f[0] = sx; f[1] = sy; f[2] = sz; f[3] = 0;
      f[4] = ox; f[5] = oy; f[6] = oz; f[7] = 0;
      if (!this.viewportBuffer) {
        this.viewportBuffer = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'xenos-viewport' });
      }
      this.device.queue.writeBuffer(this.viewportBuffer, 0, bytes);
      this._viewportLast = [sx, sy, sz, ox, oy, oz];
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
        multisample: { count: rtState?.sampleCount ?? 1 },
        primitive: { topology: 'triangle-list', cullMode: rtState?.raster?.cullMode ?? 'none', frontFace: rtState?.raster?.frontFace ?? 'ccw' },
      });
      this.pipelineCache.set(key, p);
      return p;
    }

    /**
     * Upload an index buffer from guest memory into a GPUBuffer.
     * Xenos indices are big-endian, so we byteswap when indexEndianness indicates 8in16/8in32 —
     * and, for the primitive types WebGPU has no topology for (kTriangleFan, kLineLoop,
     * kQuadList), EXPAND the indices host-side (xenia's primitive-processor conversion):
     * the draw then uses a supported topology with the expanded buffer. The returned
     * object carries the effective index format and the (possibly expanded) count —
     * 0 means fewer guest indices than one primitive; the caller skips the draw.
     */
    uploadIndexBuffer({ indexGuestBase, numIndices, indexFormat /* 0=int16, 1=int32 */, indexEndianness /* swap_mode 0..3 from VGT_DMA_SIZE */, primType = 0, multiPrimIbEnable = false, multiPrimResetIndex = 0 }) {
      if (!this.device) throw new XenosWebGpuError('uploadIndexBuffer before attach()');
      const bytesPerIndex = indexFormat === 1 ? 4 : 2;
      const total = numIndices * bytesPerIndex;
      // xenia's primitive processor normalizes the swap mode for 16-bit
      // indices: "32-bit endian swap mode {} is used for 16-bit indices ...
      // Currently disabling the swap for 16-and-32 and replacing 8-in-32 with
      // 8-in-16."
      let swapMode = indexEndianness | 0;
      if (indexFormat === 0 && swapMode !== 0 && swapMode !== 1) {
        this.log('warn', `draw: 32-bit endian swap mode ${swapMode} is used for 16-bit indices — xenia replaces 8-in-32 with 8-in-16 and disables the other 32-bit modes`);
        swapMode = swapMode === 2 ? 1 : 0;
      }
      const view = this.readGuestMemory(indexGuestBase >>> 0);
      if (!view) throw new XenosWebGpuError(`index buffer at guest physical 0x${(indexGuestBase >>> 0).toString(16)} is not mapped`);
      const paddedTotal = Math.ceil(total / 4) * 4;
      const be = new Uint8Array(paddedTotal);
      const src = view instanceof Uint8Array ? view : new Uint8Array(view.buffer, view.byteOffset ?? 0, view.byteLength ?? view.buffer.byteLength);
      be.set(src.subarray(0, total));
      // The GPU fetch path applies the VGT_DMA_SIZE swap_mode to raw guest bytes; we mirror that on upload:
      // read the guest words big-endian, apply GpuSwap(swap_mode), write them little-endian for the LE GPU.
      // The swap width follows the index width: 16-bit indices are swapped with
      // GpuSwap16 (xenia's xenos.h uint16 overload — the 32-bit-only modes are
      // normalized away above), 32-bit indices with GpuSwap32.
      const gpuSwap32 = pm4().gpuSwap32;
      const gpuSwap16 = pm4().gpuSwap16;
      const base = indexGuestBase >>> 0;
      const decode = () => {
        // The swap is only forced for primitive conversion — the plain path swaps lazily
        // on cache miss, so identical repeated index buffers cost only the hash below.
        const decoded = new Uint8Array(paddedTotal);
        const s = new DataView(be.buffer), d = new DataView(decoded.buffer);
        if (indexFormat === 1) {
          for (let i = 0; i + 3 < paddedTotal; i += 4) d.setUint32(i, gpuSwap32(s.getUint32(i, false), swapMode), true);
        } else {
          for (let i = 0; i + 1 < paddedTotal; i += 2) d.setUint16(i, gpuSwap16(s.getUint16(i, false), swapMode), true);
        }
        return decoded;
      };
      // xenia's primitive reset index: it applies only when the guest enables
      // multi-primitive index buffers (PA_SU_SC_MODE_CNTL.multi_prim_ib_ena)
      // and only to the strip-like primitive types. The reset value is
      // compared in the guest's index endianness (xenia:
      // "guest_primitive_reset_index_guest_endian = GpuSwap(reset_indx,
      // guest_index_endian)"), so the same swap the indices went through is
      // applied to the reset index before the comparison; 16-bit indices
      // compare all 16 bits, 32-bit indices only the low 24 (xenia's vertex
      // index mask).
      let resetMatcher = null;
      if (multiPrimIbEnable && PRIM_TYPE_RESETTABLE.has((primType ?? 0) & 0x3f)) {
        const resetIndex = multiPrimResetIndex >>> 0;
        if (indexFormat === 0) {
          // xenia only enables the reset for 16-bit indices when the reset
          // index fits into 16 bits ("guest_primitive_reset_index_guest_endian
          // <= UINT16_MAX"); a wider guest value cannot be represented and
          // would silently alias a real index.
          if ((resetIndex >>> 16) === 0) {
            const value = gpuSwap16(resetIndex, swapMode);
            resetMatcher = (i) => i === value;
          }
        } else {
          const value = gpuSwap32(resetIndex, swapMode) & 0xffffff;
          resetMatcher = (i) => (i & 0xffffff) === value;
        }
      }
      // Host-side primitive conversion (fan/loop/quad-list → list/strip).
      const convert = PRIM_TYPE_CONVERT.get((primType ?? 0) & 0x3f);
      if (convert) {
        const decoded = decode();
        if (((primType ?? 0) & 0x3f) === 0x0d /* kQuadList */ && (numIndices & 3) !== 0) {
          this.log('warn', `draw: ${numIndices} indices do not complete ${Math.floor(numIndices / 4)} quad(s) — the ${numIndices & 3} leftover index value(s) are dropped (hardware primitive assembly behaves the same)`);
        }
        const typed = new (indexFormat === 1 ? Uint32Array : Uint16Array)(decoded.buffer, decoded.byteOffset, numIndices);
        const expanded = expandConvertedIndices(typed, convert.kind, resetMatcher);
        if (!expanded) {
          return { buffer: null, numIndices: 0, indexFormat }; // fewer indices than one primitive — nothing to draw
        }
        const finalBytes = new Uint8Array(expanded.buffer, expanded.byteOffset, expanded.byteLength);
        // salt the cache key so the converted buffer at the same guest base never aliases the raw one
        const salt = ((swapMode | 0) | ((primType & 0x3f) << 4) | (resetMatcher ? 0x40 : 0) | ((multiPrimResetIndex & 0xffffff) << 8)) >>> 0;
        const { buffer, hit } = this._cachedBuffer('ib', base, finalBytes.byteLength, salt, finalBytes, () => {
          const buf = this.device.createBuffer({ size: Math.ceil(finalBytes.byteLength / 4) * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, label: 'xenos-index' });
          const padded = new Uint8Array(Math.ceil(finalBytes.byteLength / 4) * 4);
          padded.set(finalBytes);
          this.device.queue.writeBuffer(buf, 0, padded);
          return buf;
        });
        if (!hit) {
          this.uploads.indexBuffers++;
          this.uploads.indexBytes += finalBytes.byteLength;
        }
        return { buffer, numIndices: expanded.length, indexFormat };
      }
      const { buffer, hit } = this._cachedBuffer('ib', base, total, swapMode | 0, be, () => {
        const buf = this.device.createBuffer({ size: paddedTotal, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, label: 'xenos-index' });
        this.device.queue.writeBuffer(buf, 0, decode());
        return buf;
      });
      if (!hit) {
        this.uploads.indexBuffers++;
        this.uploads.indexBytes += total;
      }
      // The guest's primitive reset index ends a strip: the markers are not
      // vertices, and every range between two markers is a separate strip.
      let ranges = null;
      if (resetMatcher && PRIM_TYPE_STRIP_RESETTABLE.has((primType ?? 0) & 0x3f)) {
        const decoded = decode();
        const typed = new (indexFormat === 1 ? Uint32Array : Uint16Array)(decoded.buffer, decoded.byteOffset, numIndices);
        ranges = splitResetRanges(typed, resetMatcher);
      }
      return { buffer, numIndices, indexFormat, ranges };
    }

    /**
     * Synthesize (and memoize) the little-endian index buffer for an implicit
     * (non-indexed) draw of a primitive type that was converted host-side:
     * fan/loop/quad-list over vertices 0..numIndices-1 expanded the same way
     * as guest indices. Keyed on (primType, numIndices) — bounded, oldest first.
     */
    _implicitConvertedIndexBuffer(primType, numIndices) {
      const convert = PRIM_TYPE_CONVERT.get(primType & 0x3f);
      const key = `${primType & 0x3f}:${numIndices}`;
      let entry = this.implicitIndexCache.get(key);
      if (entry) { entry.lastUsed = ++this.implicitIndexClock; return entry; }
      const use16 = numIndices <= 0x10000; // implicit vertex counts are bounded by VGT (u16)
      const seq = new (use16 ? Uint16Array : Uint32Array)(numIndices);
      for (let i = 0; i < numIndices; i++) seq[i] = i;
      const expanded = expandConvertedIndices(seq, convert.kind);
      const count = expanded ? expanded.length : 0;
      const bytes = count * seq.BYTES_PER_ELEMENT;
      // WebGPU buffer sizes must be a multiple of 4 — an odd 16-bit index count
      // (a 2-index line loop expands to 3) is 6 bytes, so round the allocation up.
      const allocSize = Math.max(4, Math.ceil(bytes / 4) * 4);
      const buffer = this.device.createBuffer({ size: allocSize, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, label: `xenos-implicit-${primType & 0x3f}` });
      if (count) this.device.queue.writeBuffer(buffer, 0, new Uint8Array(expanded.buffer, expanded.byteOffset, bytes));
      entry = { buffer, format: use16 ? 'uint16' : 'uint32', numIndices: count, bytes, lastUsed: ++this.implicitIndexClock };
      this.implicitIndexCache.set(key, entry);
      while (this.implicitIndexCache.size > 16) {
        let oldestKey = null, oldest = Infinity;
        for (const [k, v] of this.implicitIndexCache) if (v.lastUsed < oldest) { oldest = v.lastUsed; oldestKey = k; }
        if (oldestKey === null) break;
        const dead = this.implicitIndexCache.get(oldestKey);
        this.implicitIndexCache.delete(oldestKey);
        this.implicitIndexBytes -= dead.bytes;
        if (dead.buffer.destroy) dead.buffer.destroy();
      }
      this.implicitIndexBytes += bytes;
      return entry;
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
        multisample: { count: rtState?.sampleCount ?? 1 },
        primitive: { topology: 'triangle-list', cullMode: rtState?.raster?.cullMode ?? 'none', frontFace: rtState?.raster?.frontFace ?? 'ccw' },
      });
      this.pipelineCache.set(key, p);
      return p;
    }

    draw(params) {
      try {
        return this._drawImpl(params);
      } catch (e) {
        // Drop (never submit) the frame being built — partial frames stay off the canvas; the next frame starts with defined targets.
        this._dropFrame();
        throw e;
      }
    }

    _drawImpl(params) {
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
        // xenia's kDepth mode (RB_MODECONTROL.edram_mode) runs no pixel shader.
        const psCode = params.depthOnly ? null : (this.shaders.pixel?.code ?? null);
        const topology = topologyForPrimType(params.primType ?? 0x04, params.primTypeName);
        const alphaTest = globalThis.XenonGpuRenderState?.decodeAlphaTest(params.renderTargets?.colorControl ?? 0) ?? null;
        // format nibble of RB_COLOR_INFO at bits 16-19 (1 = k_8_8_8_8_GAMMA);
        // read the field directly — the full decode runs (memoized) right after.
        const convertColor0ToGamma = (((params.renderTargets?.colorInfo0 ?? 0) >>> 16) & 0xF) === 1;
        // 6-bit signed exponent at RB_COLOR_INFO bits 20-25 (xenia's int32_t :6, sign-extended)
        let colorExpBias = ((params.renderTargets?.colorInfo0 ?? 0) >>> 20) & 0x3F;
        if (colorExpBias & 0x20) colorExpBias -= 0x40;
        const program = this._translateProgram(code, psCode, alphaTest, convertColor0ToGamma, colorExpBias);

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
        const viewport = this._resolveViewport(params); // traps loudly on UCP before any GPU work happens
        this._uploadViewport(viewport);
        const rtState = this._resolveRenderTargets(params, program);
        if (!rtState) return;
        const scissor = this._resolveScissor(params);
        const { pipeline, bindGroup } = this._getTranslatedProgramResources(code, psCode, plan, topology, program, rtState, !!params.depthOnly);
        const pass = this._openFrame(rtState);
        this._frameDrawState(pass, rtState, scissor);
        pass.setPipeline(pipeline);
        if (bindGroup) pass.setBindGroup(0, bindGroup);
        for (const b of buffers) pass.setVertexBuffer(b.bindingIndex, b.buffer);
        this._setIndexBufferAndDraw(pass, params);
        this._finishDraw();
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
        if (!rtState) return;
        const scissor = this._resolveScissor(params);
        const plan = this.prepareVertexFetch({ vertexShaderCode: this.shaders.vertex.code, fetchConstants: params.vertexFetchConstants });
        const pass = this._openFrame(rtState);
        this._frameDrawState(pass, rtState, scissor);
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
        this._finishDraw();
        return;
      }
      if (this.shaders.vertex) {
        this.log('warn', 'draw: vertex shader loaded but the packet carried no fetch constants — bypassing the fetch path (full-canvas placeholder)');
      }
      const rtState = this._resolveRenderTargets(params, null);
      if (!rtState) return;
      const scissor = this._resolveScissor(params);
      const pass = this._openFrame(rtState);
      this._frameDrawState(pass, rtState, scissor);
      pass.setPipeline(this._getPlaceholderPipeline(rtState));
      pass.draw(3);
      this._finishDraw();
      return;
    }

    /** Bind the index buffer (if any) and issue the draw, from a fetch-planned pass. */
    _setIndexBufferAndDraw(pass, params) {
      const primType = (params.primType ?? 0) & 0x3f;
      if (params.isIndexed) {
        const ib = this.uploadIndexBuffer({
          indexGuestBase: params.indexGuestBase,
          numIndices: params.numIndices,
          indexFormat: params.indexFormat,
          indexEndianness: params.indexEndianness,
          primType,
          multiPrimIbEnable: !!params.multiPrimIbEnable,
          multiPrimResetIndex: params.multiPrimResetIndex >>> 0,
        });
        if (ib.numIndices === 0) { this.log('info', `draw: prim 0x${primType.toString(16)} with fewer indices than one primitive — nothing to draw`); return; }
        pass.setIndexBuffer(ib.buffer, ib.indexFormat === 1 ? 'uint32' : 'uint16');
        // A guest primitive reset splits the strip into several strips — one
        // draw per range (the reset markers themselves are not vertices).
        if (ib.ranges && ib.ranges.length > 1) {
          for (const [first, count] of ib.ranges) {
            if (count > 0) pass.drawIndexed(count, 1, first, 0, 0);
          }
          return;
        }
        pass.drawIndexed(ib.numIndices, 1, 0, 0, 0);
        return;
      }
      if (PRIM_TYPE_CONVERT.has(primType)) {
        const ib = this._implicitConvertedIndexBuffer(primType, params.numIndices);
        if (ib.numIndices === 0) { this.log('info', `draw: prim 0x${primType.toString(16)} with fewer vertices than one primitive — nothing to draw`); return; }
        pass.setIndexBuffer(ib.buffer, ib.format);
        pass.drawIndexed(ib.numIndices, 1, 0, 0, 0);
        return;
      }
      pass.draw(params.numIndices);
    }

    /**
     * Present the offscreen color target to the canvas (the guest's
     * render-target format is not the canvas format). One blit per frame; the
     * pipeline and bind group are cached per source format.
     */
    _presentOffscreen() {
      if (!this._presentPending) return;
      this._presentPending = false;
      const source = this._presentKey ? this._colorTargets.get(this._presentKey) : null;
      if (!source) return;
      const canvasTexture = this.context.getCurrentTexture();
      let entry = this._presentPipeline.get(source.format);
      if (!entry) {
        const module = this.device.createShaderModule({ code: PRESENT_BLIT_WGSL, label: 'xenos-present-blit' });
        const pipeline = this.device.createRenderPipeline({
          label: 'xenos-present-blit',
          layout: 'auto',
          vertex: { module, entryPoint: 'vs_main' },
          fragment: { module, entryPoint: 'fs_main', targets: [{ format: this.format }] },
          primitive: { topology: 'triangle-list' },
        });
        const bindGroup = this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: source.texture.createView() }],
        });
        entry = { pipeline, bindGroup };
        this._presentPipeline.set(source.format, entry);
      }
      const encoder = this.device.createCommandEncoder({ label: 'xenos-present' });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: canvasTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      });
      pass.setPipeline(entry.pipeline);
      pass.setBindGroup(0, entry.bindGroup);
      pass.draw(3);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      this.uploads.presents++;
    }

    /** Frame boundary from PM4_XE_SWAP — xenia uses this to trigger present/swap on the host. */
    swap() {
      this._flushFrame();
      this._presentOffscreen();
      this._frameClearColor = false;
      this._frameClearDepth = false;
      this.lastFrame = { at: (typeof performance !== 'undefined' ? performance.now() : Date.now()) };
      this.log('debug', `swap (frame ${this.uploads.frames})`);
    }

    /** Call after the last frame of a session to release GPU resources. */
    destroy() {
      this._flushFrame();
      if (this.depthTexture) { this.depthTexture.destroy(); this.depthTexture = null; this.depthView = null; }
      if (this.aluConstantBuffer) { this.aluConstantBuffer.destroy(); this.aluConstantBuffer = null; }
      if (this.viewportBuffer) { this.viewportBuffer.destroy(); this.viewportBuffer = null; this._viewportLast = null; }
      this._destroyColorTargets();
      for (const entry of this.implicitIndexCache.values()) if (entry.buffer.destroy) entry.buffer.destroy();
      this.implicitIndexCache.clear();
      this.implicitIndexBytes = 0;
      if (this.boolConstantBuffer) { this.boolConstantBuffer.destroy(); this.boolConstantBuffer = null; }
      if (this.device) { this.device.destroy(); this.device = null; }
      this.context = null;
      this.pipelineCache.clear();
      this.programMemo = new WeakMap();
      this.translatedPipelines = new WeakMap();
    }
  }

  // Internals exposed for unit tests only — unstable shapes, not public API.
  const _internal = { topologyForPrimType, isPrimitivePolygonal, isIntegerGpuVertexFormat, FLAT_FRAGMENT_WGSL, DEPTH_ONLY_FRAGMENT_WGSL, PRESENT_BLIT_WGSL, PRIM_TYPE_TOPOLOGY, PRIM_TYPE_CONVERT, expandConvertedIndices };

  const api = { XenosWebGpuBackend, XenosWebGpuError, PLACEHOLDER_WGSL, _internal };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonWebGpuBackend = api;
})(typeof self !== 'undefined' ? self : globalThis);
