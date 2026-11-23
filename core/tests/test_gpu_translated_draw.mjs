/**
 * Translated-draw path tests — the default draw() path runs the real
 * microcode→WGSL translation behind a recording fake device (same pattern
 * the hot-path bench uses). No GPU answers are faked: the backend's own
 * logic (translation memoization, pipeline/bind-group caching, constant
 * content-hash uploads, pre-draw traps) is what is asserted. WGSL itself
 * is validated by Dawn in test_wgsl_translate.mjs; rendering is manual
 * (docs/XENOS_GPU.md).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { assertValidWgsl, checkProgramLinks } from './helpers/dawn_validate.mjs';
import '../../public/js/gpu/xenos_registers.js';
import '../../public/js/gpu/register_file.js';
import '../../public/js/gpu/pm4.js';
import '../../public/js/gpu/command_processor.js';
import '../../public/js/gpu/vertex_fetch.js';
import '../../public/js/gpu/wgsl_translate.js';
import '../../public/js/gpu/render_state.js';
import '../../public/js/gpu/webgpu_backend.js';

const { XenosWebGpuBackend, XenosWebGpuError } = globalThis.XenonWebGpuBackend;
const INTERNAL = globalThis.XenonWebGpuBackend._internal;

globalThis.GPUBufferUsage = { VERTEX: 0x20, INDEX: 0x10, COPY_DST: 0x08, UNIFORM: 0x40 };

// ---------------------------------------------------------------- ucode fixtures (same conventions as the translator tests)
const SEQ_ALU = 0b00;
const SEQ_FETCH = 0b01;
const seq = (...bits) => bits.reduce((s, b, i) => s | (b << (2 * i)), 0);
const execCf = (address, count, sequence, opcode = 2, extraW1 = 0) => ({
  dword0: (address & 0xFFF) | ((count & 0x7) << 12) | ((sequence & 0xFFF) << 16),
  dword1: ((opcode & 0xF) << 12) | (extraW1 & 0x0FFF),
});
const condExecCf = (address, count, sequence, boolAddress, condition, end = false) =>
  execCf(address, count, sequence, end ? 4 : 3, ((boolAddress & 0xFF) << 2) | ((condition & 1) << 10));
const NOP_CF = { dword0: 0, dword1: 0 };
const cfPair = (a, b) => [
  a.dword0 >>> 0,
  (a.dword1 & 0xFFFF) | ((b.dword0 & 0xFFFF) << 16),
  ((b.dword0 >>> 16) & 0xFFFF) | ((b.dword1 & 0xFFFF) << 16),
];
const FORMAT_DWORD_SIZE = { 6: 1, 25: 2, 26: 4, 31: 2, 32: 4, 33: 1, 34: 2, 35: 4, 36: 1, 37: 2, 38: 4, 57: 3 };
const vfetchWords = ({ dstReg = 1, format = 57, dstSwiz = 0x688, isSigned = 0, isNormalized = 1, isPredicated = 0, stride = null } = {}) => {
  const w0 = ((dstReg & 0x3F) << 12) | (1 << 19);
  // num_format_all bit 13: 0 = normalized (isNormalized true); bit 12 = sign
  const w1 = (dstSwiz & 0xFFF) | (format << 16) | ((isSigned & 1) << 12) | (((isNormalized ? 0 : 1) & 1) << 13) | ((isPredicated & 1) << 31);
  const strideWords = stride ?? FORMAT_DWORD_SIZE[format] ?? 1;
  return [w0 >>> 0, w1 >>> 0, strideWords >>> 0];
};
const temp = (reg) => ({ sel: 1, reg: reg & 0x3F });
const konst = (index) => ({ sel: 0, reg: index & 0xFF });
const aluWords = ({
  vectorDest = 0, scalarDest = 0, isExport = 0, vectorWriteMask = 0xF, scalarWriteMask = 0,
  scalarOpcode = 50, vectorOpcode = 2, predicated = 0, predicateCondition = 0,
  vectorDestRelative = 0, scalarDestRelative = 0, absConstants = 0, constRelAbs = 0,
  src = [temp(0), temp(1), temp(1)],
} = {}) => {
  const swz = src.map((s) => s.swizzle ?? 0);
  const neg = src.map((s) => s.negate ?? 0);
  const w0 = ((vectorDest & 0x3F)
    | ((vectorDestRelative & 1) << 6)
    | ((absConstants & 1) << 7)
    | ((scalarDest & 0x3F) << 8)
    | ((scalarDestRelative & 1) << 14)
    | ((isExport & 1) << 15)
    | ((vectorWriteMask & 0xF) << 16)
    | ((scalarWriteMask & 0xF) << 20)
    | ((scalarOpcode & 0x3F) << 26)) >>> 0;
  const w1 = ((swz[2] & 0xFF) | ((swz[1] & 0xFF) << 8) | ((swz[0] & 0xFF) << 16)
    | ((neg[2] & 1) << 24) | ((neg[1] & 1) << 25) | ((neg[0] & 1) << 26)
    | ((predicateCondition & 1) << 27) | ((predicated & 1) << 28) | ((constRelAbs & 1) << 30)) >>> 0;
  const w2 = ((src[2].reg & 0xFF) | ((src[1].reg & 0xFF) << 8) | ((src[0].reg & 0xFF) << 16)
    | ((vectorOpcode & 0x1F) << 24) | ((src[2].sel & 1) << 29) | ((src[1].sel & 1) << 30) | ((src[0].sel & 1) << 31)) >>> 0;
  return [w0, w1, w2];
};
const shader = (...words) => new Uint32Array(words.flat());

// A minimal VS: fetch r1 (float32x3), export oPos = r1 + r1.
const FETCH_VS = shader(
  cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
  vfetchWords({ dstReg: 1, format: 57 }),
  aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 0 /* add */, src: [temp(1), temp(1), temp(1)] }),
);
// A VS that also reads ALU constants: fetch r1, mad r3 = r1*fc0 + fc1, export oPos = r3 + r1.
const CONST_VS = shader(
  cfPair(execCf(1, 3, seq(SEQ_FETCH, SEQ_ALU, SEQ_ALU)), NOP_CF),
  vfetchWords({ dstReg: 1, format: 57 }),
  aluWords({ vectorDest: 3, vectorOpcode: 11 /* mad */, src: [temp(1), konst(0), konst(1)] }),
  aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 0, src: [temp(3), temp(1), temp(1)] }),
);
// A VS that reads a bool constant (cexec wrapper): fetch r1, add r3 = r1 + r1.
const BOOL_VS = shader(
  cfPair(condExecCf(1, 2, seq(SEQ_FETCH, SEQ_ALU), 4, 1), NOP_CF),
  vfetchWords({ dstReg: 1, format: 57 }),
  aluWords({ vectorDest: 3, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
);
// A VS that draws from constants only (no attribute reads).
const CONST_ONLY_VS = shader(
  cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
  aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 0, src: [konst(0), konst(0), konst(0)] }),
);
// A VS that trips the translator (unassigned vector opcode 30).
// A VS that also exports an interpolator (oT7) alongside oPos.
const VARYING_VS = shader(
  cfPair(execCf(1, 3, seq(SEQ_FETCH, SEQ_ALU, SEQ_ALU)), NOP_CF),
  vfetchWords({ dstReg: 1, format: 57 }),
  aluWords({ isExport: 1, vectorDest: 7, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
);
const BROKEN_VS = shader(
  cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
  vfetchWords({ dstReg: 1, format: 57 }),
  aluWords({ vectorDest: 3, vectorOpcode: 30 }),
);
// A VS fetching signed 8-bit attributes (sint8x4) — cannot feed a vec4f location.
const SINT_VS = shader(
  cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
  vfetchWords({ dstReg: 1, format: 6 /* k_8_8_8_8 */, isSigned: 1, isNormalized: 0 }),
  aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
);

// ---------------------------------------------------------------- recording fake device
function makeFakeDevice() {
  const log = { modules: [], pipelines: [], bindGroupsCreated: [], bindGroupsSet: [], buffers: [], writes: [], drawCalls: [], vertexBindings: [], submits: 0 };
  const device = {
    queue: {
      writeBuffer(buffer, offset, data) { log.writes.push({ buffer, offset, bytes: Uint8Array.from(data) }); },
      submit() { log.submits++; },
    },
    createBuffer(desc) { const b = { desc }; log.buffers.push(b); return b; },
    createTexture(desc) {
      const tex = { desc, destroy() { tex.destroyed = true; }, createView() { return { kind: 'texture-view' }; } };
      (log.textures ??= []).push(tex);
      return tex;
    },
    createShaderModule(desc) { log.modules.push(desc); return { desc }; },
    createRenderPipeline(desc) {
      const pipeline = { desc, getBindGroupLayout() { return { kind: 'fake-bgl' }; } };
      log.pipelines.push(pipeline);
      return pipeline;
    },
    createBindGroup(desc) { log.bindGroupsCreated.push(desc); return { desc }; },
    createCommandEncoder() {
      const pass = {
        setPipeline(p) { log.boundPipeline = p; },
        setVertexBuffer(i, b) { log.vertexBindings.push([i, b]); },
        setBindGroup(i, bg) { log.bindGroupsSet.push([i, bg]); },
        setIndexBuffer(b, fmt) { log.indexBuffer = { buffer: b, format: fmt }; },
        setBlendConstant(c) { log.blendConstant = c; },
        setScissorRect(x, y, w, h) { log.scissorRect = [x, y, w, h]; },
        setStencilReference(ref) { log.stencilRef = ref; },
        draw(n) { log.drawCalls.push(['draw', n]); },
        drawIndexed(n) { log.drawCalls.push(['drawIndexed', n]); },
        end() {},
      };
      return { beginRenderPass(desc) { (log.passes ??= []).push(desc); log.passDesc = desc; return pass; }, finish() { return { fake: true }; } };
    },
    destroy() {},
  };
  return { device, log };
}

function makeGuest(size = 0x02000000) {
  const arena = new Uint8Array(size);
  const view = new DataView(arena.buffer);
  return {
    arena,
    readGuestMemory(addr) { const a = addr >>> 0; return new DataView(arena.buffer, a, arena.length - a); },
    writeGuestMemory(addr, arr) { arena.set(arr, addr); },
    writeU32BE(addr, v) { view.setUint32(addr, v >>> 0, false); },
  };
}

function makeBackend(guest, { placeholderShader = false } = {}) {
  const { device, log } = makeFakeDevice();
  const backend = new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory, placeholderShader });
  backend.device = device;
  backend.format = 'bgra8unorm';
  const canvas = { width: 64, height: 64, getContext: () => ({ getCurrentTexture: () => ({ createView: () => ({}) }) }) };
  backend.canvas = canvas;
  backend.context = canvas.getContext('webgpu');
  backend._recreateDepth(); // attach() would do this — the headless harness mirrors it
  return { backend, log };
}

const VB_BASE = 0x00100000, IB_BASE = 0x00200000;
function seedVertexBuffer(guest) {
  const view = new DataView(guest.arena.buffer);
  for (let i = 0; i < 9; i++) view.setFloat32(VB_BASE + i * 4, i + 1, false);
}
function fetchConstants() {
  const consts = new Uint32Array(192);
  consts[0] = (VB_BASE & 0x1FFFFFFC) | 3;      // type=vertex, dword-aligned byte address
  consts[1] = 9 << 2;                          // sizeWords = 9
  return consts;
}
function aluConstantArray() {
  const alu = new Uint32Array(1024);
  alu[0] = 0x3f800000; // 1.0f
  alu[1] = 0x40000000; // 2.0f
  alu[4] = 0x40400000; // 3.0f (fc[1].x)
  return alu;
}
/** A sane zeroed-register render-target snapshot: 8_8_8_8 RT0, all writes enabled, no blend/depth/stencil. */
const defaultRenderTargets = (overrides = {}) => ({
  surfaceInfo: 0,
  colorInfo0: 0, colorInfo1: 0, colorInfo2: 0, colorInfo3: 0,
  depthInfo: 0,
  colorMask: 0x0000FFFF, // all components of all four RTs
  blendRed: 0, blendGreen: 0, blendBlue: 0, blendAlpha: 0,
  stencilRefMask: 0, stencilRefMaskBf: 0,
  depthControl: 0,
  // RB_BLENDCONTROL=0 maps to src=ZERO/dest=ZERO/add (blend ENABLED in xenia's rule);
  // one/zero/add for both color and alpha is what actually disables blending.
  blendControl0: 0x00010001, blendControl1: 0x00010001, blendControl2: 0x00010001, blendControl3: 0x00010001,
  colorControl: 0,
  alphaRef: 0x3F000000, // 0.5f
  ...overrides,
});
/** A sane zeroed scissor snapshot: full 64x64 canvas (the test canvas size). */
const defaultScissor = (overrides = {}) => ({
  windowScissorTl: 0,
  windowScissorBr: 0x00400040, // br_x=64, br_y=64 (u14 fields)
  windowOffset: 0,
  screenScissorTl: 0,
  screenScissorBr: 0x20002000, // register-file reset default 8192x8192 — no effect
  ...overrides,
});
const drawParams = (overrides = {}) => ({
  primType: 4 /* triangle list */, primTypeName: 'kTriangleList', numIndices: 3,
  isIndexed: false, indexGuestBase: 0, indexFormat: 0, indexEndianness: 0, indexLengthWords: 0,
  vertexFetchConstants: fetchConstants(),
  renderTargets: defaultRenderTargets(),
  scissor: defaultScissor(),
  ...overrides,
});

// ---------------------------------------------------------------- the real path
test('draw() runs the translated vertex shader: module has both entry points, topology maps, VS fetches', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams());
  assert.equal(log.modules.length, 2, 'separate vertex and fragment modules');
  assert.match(log.modules[0].code, /fn vs_main/);
  assert.match(log.modules[1].code, /fn fs_main/);
  assert.equal(log.pipelines.length, 1);
  assert.equal(log.pipelines[0].desc.primitive.topology, 'triangle-list');
  assert.deepEqual(log.pipelines[0].desc.vertex.buffers[0].attributes.map((a) => a.shaderLocation), [1]);
  assert.deepEqual(log.drawCalls, [['draw', 3]]);
  assert.equal(log.bindGroupsCreated.length, 1, 'the viewport uniform is always bound (the translated VS always applies it)');
  assert.equal(log.bindGroupsCreated[0].entries.length, 1);
  assert.equal(log.bindGroupsCreated[0].entries[0].binding, 0, 'no fc/bools — the viewport takes binding 0');
  assert.equal(log.vertexBindings.length, 1);
  assert.equal(log.vertexBindings[0][0], 0, 'fetch binding index 0');
});

test('repeated draws reuse the pipeline and skip constant re-uploads when content is unchanged', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: CONST_VS, dwords: CONST_VS.length };
  backend.draw(drawParams({ aluConstants: aluConstantArray() }));
  const pipelinesAfterFirst = log.pipelines.length;
  const writesAfterFirst = log.writes.length;
  backend.draw(drawParams({ aluConstants: aluConstantArray() }));
  assert.equal(log.pipelines.length, pipelinesAfterFirst, 'pipeline built once per shader+layout');
  assert.equal(log.writes.length, writesAfterFirst, 'unchanged constants are not re-uploaded');
  // one ALU constant dword changed → exactly one new 4 KB write
  const modified = aluConstantArray();
  modified[0] = 0x7f800000;
  backend.draw(drawParams({ aluConstants: modified }));
  assert.equal(log.writes.length, writesAfterFirst + 1);
  const write = log.writes[log.writes.length - 1];
  assert.equal(write.bytes.length, 4096);
  assert.equal(new DataView(write.bytes.buffer).getUint32(0, true), 0x7f800000);
});

test('translated VS with fc[] binds the ALU uniform at binding 0', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: CONST_VS, dwords: CONST_VS.length };
  backend.draw(drawParams({ aluConstants: aluConstantArray() }));
  assert.equal(log.bindGroupsCreated.length, 1);
  assert.equal(log.bindGroupsCreated[0].entries.length, 2, 'fc + the viewport uniform');
  assert.deepEqual(log.bindGroupsCreated[0].entries.map((e) => e.binding), [0, 1]);
  assert.equal(log.bindGroupsCreated[0].entries[0].binding, 0);
  assert.equal(log.bindGroupsSet.length, 1);
  assert.equal(log.bindGroupsSet[0][0], 0);
});

test('translated VS with bc() binds bools at binding 0 when no fc[] is used', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: BOOL_VS, dwords: BOOL_VS.length };
  const boolConstants = new Uint32Array(8);
  boolConstants[0] = 1 << 4; // bool 4
  backend.draw(drawParams({ boolConstants }));
  const boolWrites = log.writes.filter((w) => w.buffer === backend.boolConstantBuffer);
  assert.equal(boolWrites.length, 1, 'one 32-byte bool-constants write');
  assert.equal(new DataView(boolWrites[0].bytes.buffer).getUint32(0, true), 1 << 4);
  assert.equal(log.bindGroupsCreated.length, 1);
  assert.equal(log.bindGroupsCreated[0].entries.length, 2, 'bools + the viewport uniform');
  assert.deepEqual(log.bindGroupsCreated[0].entries.map((e) => e.binding), [0, 1]);
  assert.equal(log.bindGroupsCreated[0].entries[0].binding, 0);
});

test('VS without attribute reads and without fetch constants draws (constants-fed geometry)', () => {
  const guest = makeGuest();
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: CONST_ONLY_VS, dwords: CONST_ONLY_VS.length };
  backend.draw({ primType: 4, primTypeName: 'kTriangleList', numIndices: 3, isIndexed: false, vertexFetchConstants: undefined, aluConstants: aluConstantArray(), renderTargets: defaultRenderTargets(), scissor: defaultScissor() });
  assert.equal(log.pipelines.length, 1);
  assert.deepEqual(log.pipelines[0].desc.vertex.buffers, []);
  assert.deepEqual(log.drawCalls, [['draw', 3]]);
});

// ---------------------------------------------------------------- loud traps
test('VS translation failure traps loudly with the instruction context', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: BROKEN_VS, dwords: BROKEN_VS.length };
  assert.throws(() => backend.draw(drawParams()), (e) => e instanceof XenosWebGpuError && /failed to translate/.test(e.message));
  assert.equal(log.pipelines.length, 0, 'no pipeline is built when translation fails');
});

// A minimal PS: oC0 = r0 + r0 (reads interpolator lane 0 — needs the VS to export @location(0)…
// FETCH_VS exports no interpolators, so the linked FS declares no inputs and r[0] reads as 0).
const PS_ADD = shader(
  cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
  aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(0), temp(0), temp(0)] }),
);
// PS reading fc[0]+fc[1] — pure ALU, no fetches.
const PS_READ_R7 = shader(
  cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
  aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(7), temp(7), temp(7)] }),
);
const PS_CONST_VS = shader(
  cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
  aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [konst(0), konst(1), konst(1)] }),
);

test('a loaded pixel shader is translated and run — no flat FS, no guessed fragment stage', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.shaders.pixel = { kind: 'pixel', code: PS_CONST_VS, dwords: PS_CONST_VS.length };
  backend.draw(drawParams({ aluConstants: aluConstantArray() }));
  assert.equal(log.modules.length, 2);
  assert.match(log.modules[0].code, /fn vs_main/);
  assert.match(log.modules[1].code, /fn fs_main/);
  assert.match(log.modules[1].code, /Generated by wgsl_translate\.js — Xenos pixel shader/);
  assert.doesNotMatch(log.modules[1].code, /Bring-up fragment stage/);
  assert.equal(log.pipelines[0].desc.label, 'xenos-translated-vs-ps');
  assert.equal(log.pipelines[0].desc.fragment.module.desc.code, log.modules[1].code, 'the pipeline binds the translated PS module');
  assert.equal(log.pipelines[0].desc.vertex.module.desc.code, log.modules[0].code);
  assert.deepEqual(log.drawCalls, [['draw', 3]]);
});

test('the FS declares only interpolator locations the VS exports (link-safe by construction)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: VARYING_VS, dwords: VARYING_VS.length };
  backend.shaders.pixel = { kind: 'pixel', code: PS_READ_R7, dwords: PS_READ_R7.length };
  backend.draw(drawParams());
  assert.match(log.modules[0].code, /@location\(7\) v7: vec4f/, 'VS exports location 7');
  assert.match(log.modules[1].code, /@location\(7\) i7: vec4f/, 'VS exports location 7 — the FS declares it');
  assert.doesNotMatch(log.modules[1].code, /@location\(0\) i0:/, 'no other input locations may be declared (color0 output lives at location 0 by design)');
  assert.match(log.modules[1].code, /r\[7\] = in\.i7/, 'r7 is the lane the VS wired at location 7');
  // Lanes the VS does not export read as zeros (deterministic, like xenia's zero-initialized registers)
  assert.doesNotMatch(log.modules[1].code, /r\[0\] = in\./);
});

test('every stage of the translated pair uses the shared constant bindings (fc=0, bools=1 when both exist)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: CONST_VS, dwords: CONST_VS.length }; // uses fc + no bools
  backend.shaders.pixel = { kind: 'pixel', code: PS_CONST_VS, dwords: PS_CONST_VS.length }; // uses fc
  backend.draw(drawParams({ aluConstants: aluConstantArray() }));
  assert.equal(log.bindGroupsCreated.length, 1);
  assert.deepEqual(log.bindGroupsCreated[0].entries.map((e) => e.binding), [0, 1], 'fc at binding 0, no bools anywhere, viewport after them');
  assert.match(log.modules[0].code, /@group\(0\) @binding\(0\) var<uniform> fc/);
  assert.match(log.modules[1].code, /@group\(0\) @binding\(0\) var<uniform> fc/);
});

test('the upload happens once for the pair even when only the PS reads constants', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length }; // no constants at all
  backend.shaders.pixel = { kind: 'pixel', code: PS_CONST_VS, dwords: PS_CONST_VS.length }; // fc
  backend.draw(drawParams({ aluConstants: aluConstantArray() }));
  const aluWrites = log.writes.filter((w) => w.bytes.length === 4096);
  assert.equal(aluWrites.length, 1, 'one ALU constant upload serves both stages');
});

test('no vertex shader loaded traps (never fabricates geometry)', () => {
  const guest = makeGuest();
  const { backend } = makeBackend(guest);
  assert.throws(() => backend.draw(drawParams()), (e) => e instanceof XenosWebGpuError && /no vertex shader is loaded/.test(e.message));
});

test('kTriangleWithWFlags / kQuadStrip / kPolygon trap loudly — xenia refuses them too', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  for (const [prim, name] of [[0x07, 'kTriangleWithWFlags'], [0x0e, 'kQuadStrip'], [0x0f, 'kPolygon']]) {
    assert.throws(() => backend.draw(drawParams({ primType: prim, primTypeName: name })), (e) => e instanceof XenosWebGpuError && /topology is not encodable/.test(e.message), name);
  }
});

test('point lists map to point-list topology', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ primType: 1, primTypeName: 'kPointList', numIndices: 8 }));
  assert.equal(log.pipelines[0].desc.primitive.topology, 'point-list');
});

test('pure uint/sint attribute formats cannot feed the translated vec4f inputs — traps loudly', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: SINT_VS, dwords: SINT_VS.length };
  assert.throws(() => backend.draw(drawParams()), (e) => e instanceof XenosWebGpuError && /integer GPU format/.test(e.message));
  assert.equal(log.pipelines.length, 0);
});

test('translated VS reading fc[] with no ALU constants in the packet traps', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: CONST_VS, dwords: CONST_VS.length };
  assert.throws(() => backend.draw(drawParams()), (e) => e instanceof XenosWebGpuError && /reads fc\[\]/.test(e.message));
});

test('translated VS reading attributes with no fetch constants in the packet traps', () => {
  const guest = makeGuest();
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  assert.throws(() => backend.draw(drawParams({ vertexFetchConstants: undefined })), (e) => e instanceof XenosWebGpuError && /carried no fetch constants/.test(e.message));
});

// ---------------------------------------------------------------- bring-up opt-in stays bring-up
test('placeholderShader: true still routes to the clearly-labelled fetch bring-up pipeline', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest, { placeholderShader: true });
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams());
  assert.equal(log.pipelines.length, 1);
  assert.equal(log.pipelines[0].desc.label, 'xenos-placeholder-fetch');
  assert.match(log.modules[0].code, /Bring-up placeholder — NOT translated Xenos microcode/);
});

// The exact text the draw path hands to createShaderModule must pass the real
// WebGPU compiler — Tint validates the entry points and the interface together.
test('the translated modules (VS + bring-up FS) pass Dawn validation and link', async () => {
  const TX = globalThis.XenonGpuTranslate;
  assert.ok(TX, 'translator must be loaded for backend tests');
  const FLAT = INTERNAL.FLAT_FRAGMENT_WGSL;
  for (const [label, fixture] of [['fetch', FETCH_VS], ['constants', CONST_VS], ['bools', BOOL_VS], ['constants-only', CONST_ONLY_VS], ['varying export', VARYING_VS]]) {
    const translated = TX.translateVertexShader(fixture);
    await assertValidWgsl(translated.wgsl);
    await assertValidWgsl(FLAT);
    const linkError = await checkProgramLinks(translated.wgsl, FLAT, [{
      arrayStride: 12, stepMode: 'vertex',
      attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }],
    }]);
    assert.equal(linkError, null, `${label}: Dawn pipeline rejected the bring-up pair: ${linkError}`);
  }
});

// =================================================== render-target state (RB_*)
test('translated draw binds an explicit write mask and blend state from RB_BLENDCONTROL0/RB_COLOR_MASK', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  // src-alpha / one-minus-src-alpha for color and alpha, write RG only (0b0101? use 0b1010)
  const blendAlpha = (6 | (0 << 5) | (7 << 8) | (6 << 16) | (0 << 21) | (7 << 24)) >>> 0;
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorMask: 0x0000000A, blendControl0: blendAlpha }) }));
  const target = log.pipelines[0].desc.fragment.targets[0];
  assert.equal(target.writeMask, 0xA);
  assert.deepEqual(target.blend, {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  });
  assert.deepEqual(log.blendConstant, [0, 0, 0, 0], 'blend constant is set on the pass');
});

test('blend disabled (one/zero/add) yields a target without a blend state, matching xenia', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams());
  const target = log.pipelines[0].desc.fragment.targets[0];
  assert.equal(target.blend, undefined);
  assert.equal(target.writeMask, 0xF);
});

test('blend constants from RB_BLEND_RED..ALPHA reach the pass as floats', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({
    renderTargets: defaultRenderTargets({ blendRed: 0x3F000000, blendGreen: 0x3F800000, blendBlue: 0x40000000, blendAlpha: 0x40400000 }), // 0.5, 1, 2, 3
  }));
  assert.deepEqual(log.blendConstant, [0.5, 1, 2, 3]);
});

test('z_enable + zfunc + z_write map to the pipeline depth state; depth attachment present with clear 1.0', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ depthControl: 0x2 | (0x3 << 4) | 0x4 }) })); // less-equal, write on
  const depth = log.pipelines[0].desc.depthStencil;
  assert.equal(depth.format, 'depth24plus-stencil8');
  assert.equal(depth.depthCompare, 'less-equal');
  assert.equal(depth.depthWriteEnabled, true);
  const att = log.passDesc.depthStencilAttachment;
  assert.ok(att, 'the pass carries the depth/stencil attachment');
  assert.equal(att.depthLoadOp, 'clear');
  assert.equal(att.depthClearValue, 1.0);
  assert.equal(att.stencilLoadOp, 'clear');
  assert.deepEqual(att.view, { kind: 'texture-view' }, 'fake depth view recorded');
});

test('stencil enable maps func/ops into front/back state and sets the pass reference', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  // stencil on; z on; front: func=less(1), fail=replace(2), zpass=increment-clamp(3), zfail=invert(5)
  const dc = 0x1 | 0x2 | (0x1 << 4) | (0x1 << 8) | (0x2 << 11) | (0x3 << 14) | (0x5 << 17);
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ depthControl: dc, stencilRefMask: 0x0000427F }) }));
  const depth = log.pipelines[0].desc.depthStencil;
  assert.deepEqual(depth.stencilFront, { compare: 'less', failOp: 'replace', depthFailOp: 'invert', passOp: 'increment-clamp' });
  assert.deepEqual(depth.stencilBack, depth.stencilFront);
  assert.equal(depth.stencilReadMask, 0x42);
  assert.equal(depth.stencilWriteMask, 0x00);
  assert.equal(log.stencilRef, 0x7F);
});

test('different render-target states produce distinct cached pipelines; identical states reuse', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const rtA = defaultRenderTargets({ colorMask: 0xF });
  const rtB = defaultRenderTargets({ colorMask: 0x5 });
  backend.draw(drawParams({ renderTargets: rtA }));
  backend.draw(drawParams({ renderTargets: rtB }));
  backend.draw(drawParams({ renderTargets: { ...rtA } })); // same raw values → same key → same pipeline
  assert.equal(log.pipelines.length, 2, 'one pipeline per (shader, layout, topology, RT state)');
  assert.equal(log.drawCalls.length, 3);
});

// A pixel shader statically exporting oC0 (add r0+r0 → oC0).
const psWithExportsColor0 = () => shader(
  cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
  aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(0), temp(0), temp(0)] }),
);

test('alpha test: the translated PS declares the discard + uniform bound at bindings.alphaTest', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const PS_BLACK = psWithExportsColor0();
  backend.shaders.pixel = { kind: 'pixel', code: PS_BLACK, dwords: PS_BLACK.length };
  assert.ok(globalThis.XenonGpuTranslate.translateShaderProgram(FETCH_VS, PS_BLACK, { alphaTest: { compare: 1 } }).fragment.wgsl.includes('discard'), 'translator emits the discard when exportsColor0');
  const state = backend.draw(drawParams({
    renderTargets: defaultRenderTargets({ colorControl: 0x8 | 0x1, alphaRef: 0x3F000000 }), // enable | kLess
  }));
  const fsModule = log.modules.find((m) => m.label === 'xenos-translated-ps');
  assert.ok(fsModule, 'translated PS module');
  assert.match(fsModule.code, /fn alphaTestPass/);
  assert.match(fsModule.code, /if \(!alphaTestPass\(out\.color0\.w\)\) \{ discard; \}/);
  assert.match(fsModule.code, /var<uniform> alphaTest: vec4f;/);
  assert.equal(log.pipelines.length, 1, 'pipeline created with the alpha-test PS');
  const bg = log.bindGroupsCreated.find((g) => g.entries.some((e) => e.resource.buffer === backend.alphaTestBuffer));
  assert.ok(bg, 'bind group binds the alpha-test uniform');
  assert.equal(bg.entries.find((e) => e.resource.buffer === backend.alphaTestBuffer).binding, 0, 'binding 0 when no fc/bools are declared by either stage');
  assert.equal(backend.alphaRefHash, 0x3F000000 >>> 0);
  assert.equal(backend.uploads.constantUploads, 1, 'one 16-byte upload for the ref');
});

test('alpha test: the reference upload is content-hashed (rewritten only when RB_ALPHA_REF changes)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const PS_BLACK = psWithExportsColor0();
  backend.shaders.pixel = { kind: 'pixel', code: PS_BLACK, dwords: PS_BLACK.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorControl: 0x8 | 0x4, alphaRef: 0x3F000000 }) }));
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorControl: 0x8 | 0x4, alphaRef: 0x3F000000 }) }));
  assert.equal(backend.uploads.constantUploads, 1, 'identical ref → one upload');
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorControl: 0x8 | 0x4, alphaRef: 0x40000000 }) }));
  assert.equal(backend.uploads.constantUploads, 2, 'changed ref → re-upload');
  assert.equal(backend.alphaRefHash, 0x40000000 >>> 0);
});

test('alpha test: different compare functions produce distinct pipelines (translation keyed)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const PS_BLACK = psWithExportsColor0();
  backend.shaders.pixel = { kind: 'pixel', code: PS_BLACK, dwords: PS_BLACK.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorControl: 0x8 | 0x1 }) })); // kLess
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorControl: 0x8 | 0x3 }) })); // kLessEqual
  assert.equal(log.pipelines.length, 2, 'alpha-test enable/func is part of the pipeline key');
});

test('alpha test: kAlways draws normally (decode treats the test as disabled, xenia skips it)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const PS_BLACK = psWithExportsColor0();
  backend.shaders.pixel = { kind: 'pixel', code: PS_BLACK, dwords: PS_BLACK.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorControl: 0x8 | 0x7 }) })); // enable | kAlways
  const fsModule = log.modules.find((m) => m.label === 'xenos-translated-ps');
  assert.ok(!/discard/.test(fsModule.code), 'no discard emitted for kAlways');
  assert.equal(backend.alphaTestBuffer, undefined, 'no alpha-test uniform allocated');
});

test('GAMMA RT0: the translated PS is gamma-encoded (xenia PWL); the draw runs, no trap', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const PS_BLACK = psWithExportsColor0();
  backend.shaders.pixel = { kind: 'pixel', code: PS_BLACK, dwords: PS_BLACK.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 1 << 16 }) })); // k_8_8_8_8_GAMMA
  const fsModule = log.modules.find((m) => m.label === 'xenos-translated-ps');
  assert.match(fsModule.code, /fn xenosGamma/);
  assert.match(fsModule.code, /out\.color0 = vec4f\(xenosGamma/);
  assert.equal(log.pipelines.length, 1, 'pipeline created for the GAMMA PS');

  // same ucode with an unorm RT0: no gamma code — distinct pipelines
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 0 }) }));
  assert.equal(log.pipelines.length, 2, 'gamma conversion is part of the translation/pipeline key');
  const fsModule2 = log.modules[log.modules.length - 1];
  assert.doesNotMatch(fsModule2.code, /fn xenosGamma/);
});

test('traps: a color format WebGPU only has behind an optional feature is refused naming the format', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  assert.throws(
    () => backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 5 << 16 }) })),
    /k_16_16_16_16/);
});

test('a guest color format the canvas cannot carry renders into an offscreen target of that format', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  // k_2_10_10_10 → rgb10a2unorm (xenia's GetColorDXGIFormat)
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 2 << 16 }) }));
  // the pipeline's color target carries the guest's format
  assert.equal(log.pipelines[0].desc.fragment.targets[0].format, 'rgb10a2unorm');
  // the frame renders into an offscreen texture of that format, not the canvas
  const rt = log.textures.find((t) => t.desc.label === 'xenos-rt-rgb10a2unorm');
  assert.ok(rt, 'the offscreen render target exists');
  assert.equal(rt.desc.format, 'rgb10a2unorm');
  assert.equal(rt.desc.size[0], 64);
  assert.equal(rt.desc.size[1], 64);
  // ... and the canvas is NOT the pass's color attachment (the frame renders offscreen)
  assert.equal(log.passes.length, 1);
  assert.ok(log.textures.some((t) => t.desc.label === 'xenos-rt-rgb10a2unorm'));
});

test('the present blit shader compiles with Dawn', async () => {
  await assertValidWgsl(INTERNAL.PRESENT_BLIT_WGSL);
});

test('a frame that rendered into the canvas is not overwritten by a stale offscreen target', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  // frame 1: an offscreen format → the canvas is only updated at swap()
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 2 << 16 }) }));
  backend.swap();
  const drawsAfterFirstSwap = log.drawCalls.length;
  // frame 2: the canvas format → the frame goes straight to the canvas
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 0 }) }));
  backend.swap();
  assert.equal(log.drawCalls.length, drawsAfterFirstSwap + 1, 'only the guest draw — no stale blit over the canvas');
});

test('the offscreen target is presented to the canvas with a blit at swap()', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 2 << 16 }) }));
  const submitsBefore = log.submits;
  const passesBefore = log.passes.length;
  backend.swap();
  assert.equal(log.submits, submitsBefore + 2, 'the frame flush and the present pass are submitted');
  assert.equal(log.passes.length, passesBefore + 1, 'the present pass is a new pass');
  // the blit is a fullscreen triangle into the canvas texture
  const blit = log.drawCalls.at(-1);
  assert.deepEqual(blit, ['draw', 3]);
  // the blit pipeline targets the canvas format and samples the offscreen texture
  const blitPipeline = log.pipelines.at(-1);
  assert.equal(blitPipeline.desc.fragment.targets[0].format, 'bgra8unorm');
  assert.match(blitPipeline.desc.label, /present/);
});

test('k_8_8_8_8 still renders straight into the canvas (no offscreen target, no blit)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 0 }) }));
  assert.equal(log.pipelines[0].desc.fragment.targets[0].format, 'bgra8unorm');
  assert.equal(log.textures.filter((t) => t.desc.label?.startsWith('xenos-rt-')).length, 0, 'no offscreen target');
  const drawsBefore = log.drawCalls.length;
  backend.swap();
  assert.equal(log.drawCalls.length, drawsBefore, 'no present blit — the canvas already holds the frame');
});

test('the guest changing the render-target format mid-frame ends the pass and starts a new one', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 0 }) }));
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 2 << 16 }) }));
  assert.equal(log.passes.length, 2, 'a render pass has one color attachment format');
});

test('the offscreen target is released on resize() and destroy()', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 2 << 16 }) }));
  const rt = log.textures.find((t) => t.desc.label === 'xenos-rt-rgb10a2unorm');
  assert.ok(rt);
  let destroyed = false;
  rt.destroy = () => { destroyed = true; };
  backend.resize();
  assert.equal(destroyed, true, 'the stale offscreen target is destroyed on resize');
});

test('traps: kD24FS8 float depth refused when depth testing is enabled', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  assert.throws(
    () => backend.draw(drawParams({ renderTargets: defaultRenderTargets({ depthInfo: 1 << 16, depthControl: 0x2 }) })),
    /kD24FS8/);
});

test('traps: back-face stencil reference differing from the front face is refused (WebGPU has one reference)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const dc = 0x1 | 0x80; // stencil on + backface_enable
  assert.throws(
    () => backend.draw(drawParams({ renderTargets: defaultRenderTargets({ depthControl: dc, stencilRefMask: 0x01, stencilRefMaskBf: 0x02 }) })),
    /STENCILREFMASK_BF/);
});

test('traps: a draw without the render-target snapshot refuses (the CP always attaches it)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const params = drawParams();
  delete params.renderTargets;
  assert.throws(() => backend.draw(params), /render-target register snapshot missing/);
});

test('traps are skipped when the PS does not write oC0 (RT0 format never observed)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  // A PS that exports nothing yet references no interpolators: constant0 export only.
  const PS_BLACK = shader(
    cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 3, vectorOpcode: 2, src: [temp(1), temp(1), temp(1)] }), // dead write to r3, no exports
  );
  backend.shaders.pixel = { kind: 'pixel', code: PS_BLACK, dwords: PS_BLACK.length };
  const state = backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 1 << 16 }) })); // GAMMA RT0
  assert.equal(log.pipelines.length, 1, 'no oC0 export → no RT0 format validation');
});

test('the pass receives setScissorRect decoded from the PA_SC_* snapshot (offset applied, clamped to canvas)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({
    scissor: defaultScissor({ windowScissorTl: (4 << 16) | 4, windowScissorBr: (36 << 16) | 36, windowOffset: (2 << 16) | 2 }),
  }));
  assert.deepEqual(log.scissorRect, [6, 6, 32, 32]);

  // screen scissor clips the window rect from outside
  backend.draw(drawParams({
    scissor: defaultScissor({ screenScissorTl: (8 << 16) | 12, screenScissorBr: (24 << 16) | 60 }),
  }));
  assert.deepEqual(log.scissorRect, [12, 8, 48, 16]);

  // a rect fully outside the canvas clamps to a zero-sized scissor
  backend.draw(drawParams({
    scissor: defaultScissor({ windowScissorTl: (100 << 16) | 100, windowScissorBr: 0x00400040 }),
  }));
  assert.deepEqual(log.scissorRect, [64, 64, 0, 0]);
});

test('a draw without the scissor snapshot refuses (the CP always attaches it)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const params = drawParams();
  delete params.scissor;
  assert.throws(() => backend.draw(params), /scissor register snapshot missing/);
});

test('a PS that does not export oC0 gets writeMask 0 on the color target (no fabricated black writes)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  // dead-write PS: no exports at all
  const PS_BLACK = shader(
    cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 3, vectorOpcode: 2, vectorWriteMask: 0xF, src: [temp(1), temp(1), temp(1)] }),
  );
  backend.shaders.pixel = { kind: 'pixel', code: PS_BLACK, dwords: PS_BLACK.length };
  backend.draw(drawParams());
  assert.equal(log.pipelines.length, 1);
  assert.equal(log.pipelines[0].desc.fragment.targets[0].writeMask, 0, 'no color writes when the PS does not export oC0');
  assert.equal(log.pipelines[0].desc.fragment.targets[0].blend, undefined, 'blend disabled with mask 0');
  // same ucode with an oC0 export flips the mask — distinct pipeline
  const PS_O = PS_ADD; // exports oC0
  backend.shaders.pixel = { kind: 'pixel', code: PS_O, dwords: PS_O.length };
  backend.draw(drawParams());
  assert.equal(log.pipelines.length, 2, 'color-export bit is part of the pipeline key');
  assert.equal(log.pipelines[1].desc.fragment.targets[0].writeMask, 0xF);
});

test('an oDepth-only PS writes depth but never color (writeMask 0, depth stays enabled)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  // PS exports only oDepth (dest 61): r1.x + r1.x → oDepth
  const PS_ODEPTH = shader(
    cfPair(execCf(1, 2, seq(SEQ_ALU, SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 1, vectorOpcode: 0, src: [temp(0), temp(0), temp(0)] }),
    aluWords({ isExport: 1, vectorDest: 61, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  );
  backend.shaders.pixel = { kind: 'pixel', code: PS_ODEPTH, dwords: PS_ODEPTH.length };
  backend.draw(drawParams());
  assert.equal(log.pipelines[0].desc.fragment.targets[0].writeMask, 0, 'color writes disabled');
  assert.ok(log.pipelines[0].desc.fragment.module.desc.code.includes('frag_depth'), 'depth export preserved');
  assert.ok(log.pipelines[0].desc.depthStencil, 'depth-stencil state present');
});

test('frame-clear semantics: first pass of a frame clears; draws batch into that one pass (guest clears, not the host)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest); // makeBackend mirrors attach()/_recreateDepth — frame starts fresh
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const passDescs = () => log.passes;
  backend.draw(drawParams());
  backend.draw(drawParams());
  backend.draw(drawParams());
  // one batched frame = one pass, descriptor opened on the first draw of the frame
  assert.equal(passDescs().length, 1);
  assert.equal(passDescs()[0].colorAttachments[0].loadOp, 'clear');
  assert.equal(passDescs()[0].depthStencilAttachment.depthLoadOp, 'clear');
  assert.equal(passDescs()[0].depthStencilAttachment.stencilLoadOp, 'clear');
  // PM4_XE_SWAP ends the frame — next frame starts with defined targets again
  backend.swap();
  backend.draw(drawParams());
  assert.equal(passDescs().length, 2);
  assert.equal(passDescs()[1].colorAttachments[0].loadOp, 'clear');
  assert.equal(passDescs()[1].depthStencilAttachment.depthLoadOp, 'clear');
  // depth clear value and stencil clear are the frame-start defaults
  assert.equal(passDescs()[1].depthStencilAttachment.depthClearValue, 1.0);
  assert.equal(passDescs()[1].depthStencilAttachment.stencilClearValue, 0);
});

test('batched frame: draws accumulate; swap() ends the pass and submits once per frame', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams());
  backend.draw(drawParams());
  assert.equal(log.passes.length, 1, 'both draws share one pass');
  assert.equal(log.submits, 0, 'nothing reaches the canvas until swap()');
  backend.swap();
  assert.equal(log.passes.length, 1);
  assert.equal(log.submits, 1, 'one submit per swap');
  backend.draw(drawParams());
  assert.equal(log.submits, 1, 'next frame not yet submitted');
  backend.swap();
  assert.equal(log.submits, 2);
  assert.equal(log.passes.length, 2, 'one pass per frame');
});

test('batched frame: a trap mid-frame drops the pending frame unsubmitted — no partial frame reaches the canvas', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams());
  assert.equal(log.submits, 0, 'pending');
  // a 4x-MSAA render target refuses to draw — the trap drops the WHOLE pending frame
  assert.throws(
    () => backend.draw(drawParams({ renderTargets: defaultRenderTargets({ surfaceInfo: 2 << 16 }) })),
    /4x MSAA/);
  assert.equal(log.submits, 0, 'no partial frame is submitted after a trap');
  backend.swap(); // nothing pending — swap has nothing to flush
  assert.equal(log.submits, 0);
  // the next frame starts with defined targets again (clear, not load)
  backend.draw(drawParams());
  backend.swap();
  const last = log.passes[log.passes.length - 1];
  assert.equal(last.colorAttachments[0].loadOp, 'clear');
  assert.equal(last.depthStencilAttachment.depthLoadOp, 'clear');
});

test('resize() forces a defined-target frame (resize of the canvas = new frame)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams());
  backend.resize();
  backend.draw(drawParams());
  assert.equal(log.passes[0].colorAttachments[0].loadOp, 'clear');
  assert.equal(log.passes[1].colorAttachments[0].loadOp, 'clear');
});

test('traps: a 4x-MSAA guest render target refuses to draw (single-sampled canvas)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  assert.throws(
    () => backend.draw(drawParams({ renderTargets: defaultRenderTargets({ surfaceInfo: 2 << 16 }) })),
    /4x MSAA/);
});

test('color_exp_bias: the translated PS scales oC0 by 2^bias (all lanes) and the draw runs', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.shaders.pixel = { kind: 'pixel', code: PS_ADD, dwords: PS_ADD.length };
  // RB_COLOR_INFO: format 0 (k_8_8_8_8), exp_bias = 3 at bits 20+
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 3 << 20 }) }));
  const fsModule = log.modules.find((m) => m.label === 'xenos-translated-ps');
  assert.match(fsModule.code, /out\.color0 = out\.color0 \* vec4f\(8\.0000000000e\+0\);/);

  // distinct pipeline for bias 0 (eb is part of the cache keys)
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 0 }) }));
  assert.equal(log.pipelines.length, 2, 'bias is part of the pipeline translation key');
  const unbiased = log.modules[log.modules.length - 1];
  assert.doesNotMatch(unbiased.code, /out\.color0 = out\.color0 \* vec4f\(/);
});

test('color_exp_bias: negative exponent decodes signed through the whole stack (CP-format snapshot)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.shaders.pixel = { kind: 'pixel', code: PS_ADD, dwords: PS_ADD.length };
  // 0b111110 << 20 = 0x3E00000 → expBias = -2 → scale 0.25
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ colorInfo0: 0x3E00000 }) }));
  const fsModule = log.modules.find((m) => m.label === 'xenos-translated-ps');
  assert.match(fsModule.code, /out\.color0 = out\.color0 \* vec4f\(2\.5000000000e\-1\);/);
});

// ------------------------------------------------------------ rasterizer state (PA_SU_SC_MODE_CNTL / PA_SU_POLY_OFFSET_*)
const bits32 = (x) => new Uint32Array(new Float32Array([x]).buffer)[0];

test('rasterizer state: cull mode and front face reach the pipeline descriptor (polygonal primitive)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ paSuScModeCntl: 0x2 }) })); // cull back
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ paSuScModeCntl: 0x1 }) })); // cull front
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ paSuScModeCntl: 0x4 }) })); // CW front faces
  assert.equal(log.pipelines[0].desc.primitive.cullMode, 'back');
  assert.equal(log.pipelines[1].desc.primitive.cullMode, 'front');
  assert.equal(log.pipelines[2].desc.primitive.frontFace, 'cw');
  assert.equal(log.pipelines[2].desc.primitive.cullMode, 'none');
  assert.equal(log.pipelines.length, 3, 'raster state is part of the pipeline key');
});

test('rasterizer state: points are never culled even with cull bits set (xenia: filled front faces only)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ primType: 0x01, primTypeName: 'kPointList', renderTargets: defaultRenderTargets({ paSuScModeCntl: 0x3 }) }));
  assert.equal(log.pipelines[0].desc.primitive.cullMode, 'none');
  assert.equal(log.drawCalls.length, 1, 'both cull bits have no effect on points');
});

test('rasterizer state: both faces culled — the draw is dropped, no pass, no submit (xenia semantics)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ paSuScModeCntl: 0x3 }) }));
  assert.equal(log.drawCalls.length, 0);
  assert.equal((log.passes ?? []).length, 0, 'no render pass is opened for a rasterization-disabled draw');
  backend.swap();
  assert.equal(log.submits, 0, 'nothing to submit — no frame was opened');
  // the next good draw starts a defined frame
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ paSuScModeCntl: 0x0 }) }));
  backend.swap();
  assert.equal(log.submits, 1);
  assert.equal(log.passes[log.passes.length - 1].colorAttachments[0].loadOp, 'clear');
});

test('rasterizer state: polygon offset reaches the depth-stencil descriptor (UNORM24 integer bias)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({
    renderTargets: defaultRenderTargets({
      paSuScModeCntl: 0x800, // front polygon-offset enable
      paSuPolyOffsetFrontScale: bits32(2.0),
      paSuPolyOffsetFrontOffset: bits32(0.5),
    }),
  }));
  const ds = log.pipelines[0].desc.depthStencil;
  assert.equal(ds.depthBias, 8388608); // ceil(0.5 * 16777215)
  assert.equal(ds.depthBiasSlopeScale, 0.125); // 2 / 16
  assert.equal(ds.depthBiasClamp, 0);
});

test('rasterizer state: kDualMode wireframe traps loudly instead of silently drawing filled', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  assert.throws(
    () => backend.draw(drawParams({ renderTargets: defaultRenderTargets({ paSuScModeCntl: (1 << 3) | (1 << 5) }) })),
    /kDualMode/);
});

// ------------------------------------------------------------ guest viewport (PA_CL_VTE_CNTL / PA_CL_CLIP_CNTL / PA_CL_VPORT_*)
test('viewport uniform is uploaded once, re-uploaded only when the composed coefficients change', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams());
  backend.draw(drawParams());
  let vpWrites = log.writes.filter((w) => w.buffer === backend.viewportBuffer);
  assert.equal(vpWrites.length, 1, 'identical viewport coefficients upload once');
  // 10.0f at PA_CL_VPORT_XOFFSET (with x offset enable)
  backend.draw(drawParams({ renderTargets: defaultRenderTargets({ paClVteCntl: 0x2, paClVportXoffset: 0x41200000 }) }));
  vpWrites = log.writes.filter((w) => w.buffer === backend.viewportBuffer);
  assert.equal(vpWrites.length, 2, 'a changed coefficient re-uploads');
  const f = new Float32Array(vpWrites[1].bytes.buffer, vpWrites[1].bytes.byteOffset, 8);
  // composed: scale 2/64 (unchanged), offset (10+0.5 half-pixel)*(2/64)-1 — pix_center defaults to kD3DZero
  assert.equal(f[0], 2 / 64);
  assert.equal(f[4], Math.fround(10.5 * (2 / 64) - 1));
  // and back to the zeroed config — content compare catches the change regardless of direction
  backend.draw(drawParams());
  vpWrites = log.writes.filter((w) => w.buffer === backend.viewportBuffer);
  assert.equal(vpWrites.length, 3);
});

test('viewport: the translated vertex module declares and applies the guest-viewport uniform (clip space, pre-w-divide)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams());
  const vsCode = log.modules[0].code;
  assert.match(vsCode, /@group\(0\) @binding\(\d+\) var<uniform> viewport: array<vec4f, 2>;/);
  assert.match(vsCode, /out\.pos = vec4f\(out\.pos\.x \* viewport\[0\]\.x/);
  // applied AFTER the guest ucode ran, right before `return out`
  assert.ok(vsCode.lastIndexOf('viewport[0]') < vsCode.indexOf('return out'), 'viewport applied before the return');
  // the fragment stage never declares it
  assert.doesNotMatch(log.modules[1].code, /viewport/);
});

test('viewport: user clip planes (PA_CL_CLIP_CNTL.ucp_ena) trap loudly — the frame is dropped, no submit', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  assert.throws(
    () => backend.draw(drawParams({ renderTargets: defaultRenderTargets({ paClClipCntl: 0x00000001 }) })),
    /ucp_ena/);
  assert.equal(log.submits, 0, 'nothing is submitted');
  assert.equal(log.drawCalls.length, 0);
  // the next clean frame starts with defined targets
  backend.draw(drawParams());
  backend.swap();
  assert.equal(log.submits, 1);
  assert.equal(log.passes[log.passes.length - 1].colorAttachments[0].loadOp, 'clear');
});

// ------------------------------------------------------------ primitive conversion (fan/loop/quad-list → list/strip)
const writeGuestIndices16BE = (guest, base, indices) => {
  const view = new DataView(guest.arena.buffer);
  indices.forEach((v, i) => view.setUint16(base + i * 2, v, false));
};
const INDEX_DWORDS = { idxBase: 0, };

test('triangle-fan (indexed): guest indices expand fan→list, pipeline uses triangle-list', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const IB = 0x00200000;
  writeGuestIndices16BE(guest, IB, [7, 8, 9, 10, 11]); // BE uint16, as the guest stores them
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ primType: 5, primTypeName: 'kTriangleFan', isIndexed: true, indexGuestBase: IB, indexFormat: 0, indexEndianness: 3 /* k16in32: swap 16-bit lanes in each 32-bit word — BE uint16 -> LE */, numIndices: 5 }));
  assert.equal(log.pipelines[0].desc.primitive.topology, 'triangle-list');
  assert.equal(log.vertexBindings.length, 1);
  assert.deepEqual(log.drawCalls, [['drawIndexed', 9]]); // (7,8,9),(7,9,10),(7,10,11)
  assert.equal(log.indexBuffer.format, 'uint16');
  const wp = log.writes.find((w) => w.buffer === log.indexBuffer.buffer);
  const out = new Uint16Array(wp.bytes.buffer, wp.bytes.byteOffset, 9);
  assert.deepEqual([...out], [7, 8, 9, 7, 9, 10, 7, 10, 11]);
});

test('triangle-fan: the converted index buffer is content-cached across draws (one upload)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const IB = 0x00200000;
  writeGuestIndices16BE(guest, IB, [0, 1, 2, 3]);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const params = () => drawParams({ primType: 5, primTypeName: 'kTriangleFan', isIndexed: true, indexGuestBase: IB, indexFormat: 0, indexEndianness: 3, numIndices: 4 });
  backend.draw(params());
  backend.draw(params());
  const ibWrites = log.writes.filter((w) => w.buffer === log.indexBuffer.buffer);
  assert.equal(ibWrites.length, 1, 'identical content expands and uploads once');
  assert.deepEqual(log.drawCalls, [['drawIndexed', 6], ['drawIndexed', 6]]);
});

test('line loop (indexed): closes the loop by re-appending the first index, line-strip topology', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const IB = 0x00200000;
  writeGuestIndices16BE(guest, IB, [3, 4, 5, 6]);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ primType: 0x0c, primTypeName: 'kLineLoop', isIndexed: true, indexGuestBase: IB, indexFormat: 0, indexEndianness: 3, numIndices: 4 }));
  assert.equal(log.pipelines[0].desc.primitive.topology, 'line-strip');
  assert.deepEqual(log.drawCalls, [['drawIndexed', 5]]);
  const wp = log.writes.find((w) => w.buffer === log.indexBuffer.buffer);
  const out = new Uint16Array(wp.bytes.buffer, wp.bytes.byteOffset, 5);
  assert.deepEqual([...out], [3, 4, 5, 6, 3]);
});

test('quad list (indexed): each quad becomes (v0,v1,v2),(v0,v2,v3); leftover indices are dropped with a warning', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const IB = 0x00200000;
  writeGuestIndices16BE(guest, IB, [0, 1, 2, 3, 4, 5, 6, 7, 99]); // 9 indices: 2 quads + 1 leftover
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ primType: 0x0d, primTypeName: 'kQuadList', isIndexed: true, indexGuestBase: IB, indexFormat: 0, indexEndianness: 3, numIndices: 9 }));
  assert.equal(log.pipelines[0].desc.primitive.topology, 'triangle-list');
  assert.deepEqual(log.drawCalls, [['drawIndexed', 12]]); // leftover dropped
  const wp = log.writes.find((w) => w.buffer === log.indexBuffer.buffer);
  const out = new Uint16Array(wp.bytes.buffer, wp.bytes.byteOffset, 12);
  assert.deepEqual([...out], [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
});

test('triangle-fan (implicit vertices, non-indexed): a synthetic index buffer is drawn instead of a wrong topology', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ primType: 5, primTypeName: 'kTriangleFan', numIndices: 4, isIndexed: false }));
  assert.equal(log.pipelines[0].desc.primitive.topology, 'triangle-list');
  assert.equal(log.vertexBindings.length, 1);
  assert.deepEqual(log.drawCalls, [['drawIndexed', 6]]);
  assert.equal(log.indexBuffer.format, 'uint16');
  const wp = log.writes.find((w) => w.buffer === log.indexBuffer.buffer);
  const out = new Uint16Array(wp.bytes.buffer, wp.bytes.byteOffset, 6);
  assert.deepEqual([...out], [0, 1, 2, 0, 2, 3]);
  // and the synthetic buffer is memoized
  backend.draw(drawParams({ primType: 5, primTypeName: 'kTriangleFan', numIndices: 4, isIndexed: false }));
  const ibWrites = log.writes.filter((w) => w.buffer === log.indexBuffer.buffer);
  assert.equal(ibWrites.length, 1, 'same implicit expansion is reused, not re-uploaded');
});

test('line loop (implicit vertices): synthesized closing strip is drawn', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ primType: 0x0c, primTypeName: 'kLineLoop', numIndices: 3, isIndexed: false }));
  assert.equal(log.pipelines[0].desc.primitive.topology, 'line-strip');
  assert.deepEqual(log.drawCalls, [['drawIndexed', 4]]);
  const wp = log.writes.find((w) => w.buffer === log.indexBuffer.buffer);
  const out = new Uint16Array(wp.bytes.buffer, wp.bytes.byteOffset, 4);
  assert.deepEqual([...out], [0, 1, 2, 0]);
});

test('implicit converted index buffers are allocated with a 4-byte-aligned size (WebGPU requirement)', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  // a 2-index line loop expands to 3 uint16 indices = 6 bytes -> the allocation must round up to 8
  backend.draw(drawParams({ primType: 0x0c, primTypeName: 'kLineLoop', numIndices: 2, isIndexed: false }));
  const implicit = log.buffers.filter((b) => b.desc.label && b.desc.label.startsWith('xenos-implicit-'));
  assert.equal(implicit.length, 1);
  assert.equal(implicit[0].desc.size % 4, 0, 'buffer size must be a multiple of 4');
  assert.equal(implicit[0].desc.size, 8);
  assert.deepEqual(log.drawCalls, [['drawIndexed', 3]]);
});

test('degenerate fan/loop draws (fewer indices than one primitive) skip the draw instead of corrupting state', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const IB = 0x00200000;
  writeGuestIndices16BE(guest, IB, [1, 2]); // only 2 indices for a fan
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  backend.draw(drawParams({ primType: 5, primTypeName: 'kTriangleFan', isIndexed: true, indexGuestBase: IB, indexFormat: 0, indexEndianness: 3, numIndices: 2 }));
  assert.deepEqual(log.drawCalls, [], 'no draw issued');
  backend.draw(drawParams({ primType: 5, primTypeName: 'kTriangleFan', numIndices: 2, isIndexed: false }));
  assert.deepEqual(log.drawCalls, [], 'no implicit draw either');
});

test('expandConvertedIndices unit semantics (fan/loop/quads, widths preserved, degenerates)', () => {
  const { expandConvertedIndices } = INTERNAL;
  assert.deepEqual([...expandConvertedIndices(new Uint16Array([0, 1, 2, 3, 4]), 'fan')], [0, 1, 2, 0, 2, 3, 0, 3, 4]);
  assert.deepEqual([...expandConvertedIndices(new Uint32Array([9, 8, 7]), 'fan')], [9, 8, 7]);
  assert.equal(expandConvertedIndices(new Uint16Array([0, 1]), 'fan'), null);
  assert.deepEqual([...expandConvertedIndices(new Uint16Array([5, 6]), 'loop')], [5, 6, 5]);
  assert.equal(expandConvertedIndices(new Uint16Array([5]), 'loop'), null);
  assert.deepEqual([...expandConvertedIndices(new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7]), 'quads')], [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
  assert.equal(expandConvertedIndices(new Uint16Array([0, 1, 2]), 'quads'), null);
});
