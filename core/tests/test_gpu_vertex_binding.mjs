/**
 * Vertex-binding tests — the draw path consumes what vertex_fetch.js decodes:
 * fetch constants → descriptors → GPU buffers → setVertexBuffer + generated
 * placeholder pipeline whose declared attributes match the descriptors.
 *
 * No real GPU in node: the backend's `device`/`canvas`/`context` are replaced
 * with recording fakes AFTER construction, exercising everything except the
 * driver itself. The command-processor half is covered end-to-end with real
 * packets (the backend double records what the CP hands it).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../../public/js/gpu/xenos_registers.js';
import '../../public/js/gpu/register_file.js';
import '../../public/js/gpu/pm4.js';
import '../../public/js/gpu/command_processor.js';
import '../../public/js/gpu/vertex_fetch.js';
import '../../public/js/gpu/wgsl_translate.js';
import '../../public/js/gpu/webgpu_backend.js';
import '../../public/js/gpu/render_state.js';

const PM4 = globalThis.XenonPM4;
const { XenosCommandProcessor } = globalThis.XenonGpuCommandProcessor;
const { XenosWebGpuBackend } = globalThis.XenonWebGpuBackend;

globalThis.GPUBufferUsage = { VERTEX: 0x20, INDEX: 0x10, COPY_DST: 0x08 };

// ------------------------------------------------------------ microcode fixture
function execCf(address, count, sequence, opcode = 2) {
  return { dword0: (address & 0xFFF) | ((count & 0x7) << 12) | ((sequence & 0xFFF) << 16), dword1: (opcode & 0xF) << 12 };
}
const NOP_CF = { dword0: 0, dword1: 0 };
function cfPair(a, b) {
  return [
    a.dword0 >>> 0,
    (a.dword1 & 0xFFFF) | ((b.dword0 & 0xFFFF) << 16),
    ((b.dword0 >>> 16) & 0xFFFF) | ((b.dword1 & 0xFFFF) << 16),
  ];
}
function vfetchWords({ dstReg = 1, srcReg = 0, constIndex = 0, constIndexSel = 0, dstSwiz = 0x688, format = 57, stride = 3, offset = 0, expAdjust = 0, isSigned = 0 } = {}) {
  const w0 = ((srcReg & 0x3F) << 5) | ((dstReg & 0x3F) << 12) | (1 << 19) | ((constIndex & 0x1F) << 20) | ((constIndexSel & 0x3) << 25);
  const w1 = (dstSwiz & 0xFFF) | ((isSigned & 1) << 12) | (format << 16) | ((expAdjust & 0x3F) << 24);
  const w2 = (stride & 0xFF) | ((offset & 0x7FFFFF) << 8);
  return [w0 >>> 0, w1 >>> 0, w2 >>> 0];
}
const ALU_ZERO = [0, 0, 0];
const SEQ_FETCH = 0b01;
const FETCH_VS = new Uint32Array([
  ...cfPair(execCf(1, 1, SEQ_FETCH, 2), NOP_CF),
  ...vfetchWords({}),
  ...ALU_ZERO,
]);
const UNREPRESENTABLE_VS = new Uint32Array([
  ...cfPair(execCf(1, 1, SEQ_FETCH, 2), NOP_CF),
  ...vfetchWords({ format: 17 /* k_11_11_10 */ }),
  ...ALU_ZERO,
]);
const STRIDE_ZERO_VS = new Uint32Array([
  ...cfPair(execCf(1, 1, SEQ_FETCH, 2), NOP_CF),
  ...vfetchWords({ stride: 0 }),
  ...ALU_ZERO,
]);

const VB_BASE = 0x00100000;
const IB_BASE = 0x00200000;

function makeGuest() {
  const arena = new Uint8Array(0x02000000);
  const view = new DataView(arena.buffer);
  // 9 BE float32 words at VB_BASE: 1..9
  for (let i = 0; i < 9; i++) view.setFloat32(VB_BASE + i * 4, i + 1, false);
  // int16 BE indices 0,1,2 at IB_BASE
  for (let i = 0; i < 3; i++) view.setUint16(IB_BASE + i * 2, i, false);
  return {
    arena,
    readGuestMemory(addr) { return new DataView(arena.buffer, addr, arena.length - addr); },
    writeGuestMemory(addr, bytes) { arena.set(bytes, addr); },
  };
}

const FETCH_CONSTS = (() => {
  const c = new Uint32Array(192);
  c[0] = (VB_BASE & 0x1FFFFFFC) | 3;   // type vertex, base VB_BASE
  c[1] = (9 << 2) | 0;                  // sizeWords 9 (36 bytes), endian kNone
  return c;
})();

// ------------------------------------------------------------ fake WebGPU device
function makeFakeDevice() {
  const calls = { createBuffer: [], writeBuffer: [], createShaderModule: [], createRenderPipeline: [], passes: [] };
  const queue = {
    writeBuffer(buffer, offset, data) { calls.writeBuffer.push({ buffer, offset, data }); },
    submit() {},
  };
  const pass = () => {
    const p = {
      vbs: [], ib: null, pipeline: null, drawN: null, drawIN: null,
      setPipeline(pl) { p.pipeline = pl; },
      setVertexBuffer(i, b) { p.vbs[i] = b; },
      setIndexBuffer(b, f) { p.ib = { buffer: b, format: f }; },
      setBlendConstant(c) { p.blendConstant = c; },
      setScissorRect(x, y, w, h) { p.scissorRect = [x, y, w, h]; },
      setStencilReference(ref) { p.stencilRef = ref; },
      draw(n) { p.drawN = n; },
      drawIndexed(n, inst, first, baseVertex) { p.drawIN = { n, inst, first, baseVertex }; },
      end() { calls.passes.push(p); },
    };
    return p;
  };
  const device = {
    calls,
    queue,
    lost: new Promise(() => {}),
    createBuffer(desc) { const b = { desc, kind: 'buffer' }; calls.createBuffer.push(b); return b; },
    createTexture() { return { destroy() {} }; },
    createShaderModule(desc) { const m = { desc }; calls.createShaderModule.push(m); return m; },
    createRenderPipeline(desc) { const p = { desc }; calls.createRenderPipeline.push(p); return p; },
    createCommandEncoder() {
      const p = pass();
      return {
        beginRenderPass() { return p; },
        finish() { return { pass: p }; },
      };
    },
    destroy() {},
  };
  return device;
}

function makeBackend(guest, { withShader = true, placeholderShader = true } = {}) {
  const backend = new XenosWebGpuBackend({
    readGuestMemory: guest.readGuestMemory,
    writeGuestMemory: guest.writeGuestMemory,
    placeholderShader,
  });
  backend.device = makeFakeDevice();
  backend.format = 'bgra8unorm';
  const canvas = {
    width: 64, height: 64,
    getContext: () => ({ getCurrentTexture: () => ({ createView: () => ({}) }) }),
  };
  backend.canvas = canvas;
  backend.context = canvas.getContext('webgpu');
  if (withShader) backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  return backend;
}

/** A sane zeroed render-target snapshot: 8_8_8_8 RT0, all writes, no blend/depth/stencil (xenia: RB_BLENDCONTROL=1 <<16|1 disables blending). */
const DEFAULT_RT = {
  colorInfo0: 0, colorInfo1: 0, colorInfo2: 0, colorInfo3: 0,
  depthInfo: 0,
  colorMask: 0x0000FFFF,
  blendRed: 0, blendGreen: 0, blendBlue: 0, blendAlpha: 0,
  stencilRefMask: 0, stencilRefMaskBf: 0,
  depthControl: 0,
  blendControl0: 0x00010001, blendControl1: 0x00010001, blendControl2: 0x00010001, blendControl3: 0x00010001,
  colorControl: 0,
};
const baseParams = {
  primType: 4, primTypeName: 'triangle-list', numIndices: 3,
  isIndexed: false, indexGuestBase: 0, indexFormat: 0, indexEndianness: 0, indexLengthWords: 0,
  vertexFetchConstants: FETCH_CONSTS,
  renderTargets: DEFAULT_RT,
  scissor: { windowScissorTl: 0, windowScissorBr: 0x00400040, windowOffset: 0, screenScissorTl: 0, screenScissorBr: 0x20002000 },
};

test('fetch draw uploads the guest vertex buffer with BE→LE dword swap and binds it', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  backend.draw({ ...baseParams });
  assert.equal(backend.uploads.vertexBuffers, 1);
  assert.equal(backend.uploads.vertexBytes, 36);
  const fb = backend.device.calls.createBuffer[0];
  assert.equal(fb.desc.usage, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
  assert.equal(fb.desc.size, 36);
  // bytes handed to the GPU decode as the original floats
  const wb = backend.device.calls.writeBuffer[0];
  const floats = new Float32Array(wb.data.buffer, wb.data.byteOffset, 9);
  assert.deepEqual([...floats], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  // pass state: vertex buffer bound at slot 0, non-indexed draw of 3 vertices
  const pass = backend.device.calls.passes[0];
  assert.equal(pass.vbs[0], fb);
  assert.equal(pass.drawN, 3);
  assert.equal(pass.ib, null);
});

test('fetch draw declares the descriptor layout on the placeholder pipeline', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  backend.draw({ ...baseParams });
  const pipeline = backend.device.calls.createRenderPipeline[0];
  assert.equal(pipeline.desc.label, 'xenos-placeholder-fetch');
  assert.equal(pipeline.desc.vertex.buffers.length, 1);
  const vb = pipeline.desc.vertex.buffers[0];
  assert.equal(vb.arrayStride, 12);
  assert.deepEqual(vb.attributes, [{ shaderLocation: 1, offset: 0, format: 'float32x3' }]);
  const wgsl = backend.device.calls.createShaderModule[0].desc.code;
  assert.match(wgsl, /@location\(1\) a0: vec3<f32>/);
  assert.match(wgsl, /return vec4<f32>\(a0\.x, a0\.y, a0\.z, 1\.0\);/);
});

test('fetch draw honors isIndexed: index buffer uploaded and drawIndexed issued', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  backend.draw({ ...baseParams, isIndexed: true, indexGuestBase: IB_BASE, indexFormat: 0, indexEndianness: 1, numIndices: 3 });
  assert.equal(backend.uploads.indexBuffers, 1);
  const pass = backend.device.calls.passes[0];
  assert.equal(pass.ib.format, 'uint16');
  assert.deepEqual(pass.drawIN, { n: 3, inst: 1, first: 0, baseVertex: 0 });
  // index buffer created with INDEX usage
  const ib = backend.device.calls.createBuffer.find((b) => b.desc.usage & GPUBufferUsage.INDEX);
  assert.ok(ib, 'an INDEX-usage buffer was created');
});

test('fetch draw refuses loudly when the fetch constant is not a vertex buffer', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const consts = new Uint32Array(192);
  consts[0] = (VB_BASE & 0x1FFFFFFC) | 2; // type texture!
  consts[1] = (9 << 2);
  assert.throws(() => backend.draw({ ...baseParams, vertexFetchConstants: consts }), /expected vertex/);
});

test('fetch draw refuses loudly on a zero-size vertex buffer', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const consts = new Uint32Array(192);
  consts[0] = (VB_BASE & 0x1FFFFFFC) | 3;
  consts[1] = 0; // sizeWords 0
  assert.throws(() => backend.draw({ ...baseParams, vertexFetchConstants: consts }), /size 0/);
});

test('fetch draw refuses loudly on a format WebGPU cannot represent', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest, { withShader: false });
  backend.shaders.vertex = { kind: 'vertex', code: UNREPRESENTABLE_VS, dwords: UNREPRESENTABLE_VS.length };
  assert.throws(() => backend.draw({ ...baseParams }), /R11G11B10|GPUVertexFormat/);
});

test('fetch draw refuses loudly on stride 0 (legal Xenos, not expressible in WebGPU)', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest, { withShader: false });
  backend.shaders.vertex = { kind: 'vertex', code: STRIDE_ZERO_VS, dwords: STRIDE_ZERO_VS.length };
  assert.throws(() => backend.draw({ ...baseParams }), /stride 0/);
});

test('draw without a vertex shader falls back to the plain triangle (documented), uploads nothing', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest, { withShader: false });
  backend.draw({ ...baseParams });
  assert.equal(backend.uploads.vertexBuffers, 0);
  const pass = backend.device.calls.passes[0];
  assert.equal(pass.drawN, 3); // plain bring-up triangle
});

test('draw without placeholderShader runs the translated VS — the placeholder is opt-in only', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest, { placeholderShader: false });
  backend.draw({ ...baseParams });
  const moduleCodes = backend.device.calls.createShaderModule.map((m) => m.desc.code);
  assert.ok(moduleCodes.some((c) => /Generated by wgsl_translate\.js/.test(c)), 'translated shader module was created');
  assert.ok(!moduleCodes.some((c) => /Bring-up placeholder/.test(c)), 'no placeholder shader is used implicitly');
  assert.equal(backend.device.calls.createRenderPipeline[0].desc.label, 'xenos-translated-vs');
  assert.equal(backend.uploads.vertexBuffers, 1, 'vertex buffers upload on the translated path too');
});

test('VETEX constants flow from the register file through the CP into the draw packet', () => {
  const guest = makeGuest();
  const calls = [];
  const backend = {
    memWrite() {}, eventWrite() {}, swap() {}, invalidateState() {}, setShaders() {},
    draw(p) { calls.push(p); },
  };
  const cp = new XenosCommandProcessor({ backend, readGuestMemory: guest.readGuestMemory });
  const beWords = (words) => {
    const buf = new Uint8Array(words.length * 4);
    const view = new DataView(buf.buffer);
    words.forEach((w, i) => view.setUint32(i * 4, w >>> 0, false));
    return buf;
  };
  // FETCH constant window: index 0 ← 0x00100003, 0x00000024 (type vertex, 36 bytes)
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_CONSTANT, 3), (1 << 16) | 0, 0x00100003, 0x00000024]));
  // vertex shader via IM_LOAD_IMMEDIATE (type 0 = vertex, 9 dwords inline)
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_IM_LOAD_IMMEDIATE, 11), 0, 9, ...FETCH_VS]));
  // draw, source = kAutoIndex (no index DMA), 3 vertices
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator]));
  assert.equal(calls.length, 1);
  assert.equal(calls[0].isIndexed, false);
  assert.equal(calls[0].numIndices, 3);
  assert.ok(calls[0].vertexFetchConstants instanceof Uint32Array);
  assert.equal(calls[0].vertexFetchConstants.length, 192);
  assert.equal(calls[0].vertexFetchConstants[0], 0x00100003);
  assert.equal(calls[0].vertexFetchConstants[1], 0x00000024);
  assert.equal(cp.stats.draws, 1);
});
