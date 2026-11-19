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
    createTexture() { return { destroy() {} }; },
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
        draw(n) { log.drawCalls.push(['draw', n]); },
        drawIndexed(n) { log.drawCalls.push(['drawIndexed', n]); },
        end() {},
      };
      return { beginRenderPass() { return pass; }, finish() { return { fake: true }; } };
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
const drawParams = (overrides = {}) => ({
  primType: 4 /* triangle list */, primTypeName: 'kTriangleList', numIndices: 3,
  isIndexed: false, indexGuestBase: 0, indexFormat: 0, indexEndianness: 0, indexLengthWords: 0,
  vertexFetchConstants: fetchConstants(),
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
  assert.equal(log.bindGroupsCreated.length, 0, 'no constants used — no bind group');
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
  assert.equal(log.bindGroupsCreated[0].entries.length, 1);
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
  const boolWrites = log.writes.filter((w) => w.bytes.length === 32);
  assert.equal(boolWrites.length, 1, 'one 32-byte bool-constants write');
  assert.equal(new DataView(boolWrites[0].bytes.buffer).getUint32(0, true), 1 << 4);
  assert.equal(log.bindGroupsCreated.length, 1);
  assert.equal(log.bindGroupsCreated[0].entries.length, 1);
  assert.equal(log.bindGroupsCreated[0].entries[0].binding, 0);
});

test('VS without attribute reads and without fetch constants draws (constants-fed geometry)', () => {
  const guest = makeGuest();
  const { backend, log } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: CONST_ONLY_VS, dwords: CONST_ONLY_VS.length };
  backend.draw({ primType: 4, primTypeName: 'kTriangleList', numIndices: 3, isIndexed: false, vertexFetchConstants: undefined, aluConstants: aluConstantArray() });
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
  assert.deepEqual(log.bindGroupsCreated[0].entries.map((e) => e.binding), [0], 'fc at binding 0, no bools anywhere');
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

test('triangle-fan is not encodable in WebGPU and traps instead of drawing wrong connectivity', () => {
  const guest = makeGuest();
  seedVertexBuffer(guest);
  const { backend } = makeBackend(guest);
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  assert.throws(() => backend.draw(drawParams({ primType: 5, primTypeName: 'kTriangleFan' })), (e) => e instanceof XenosWebGpuError && /topology is not encodable/.test(e.message));
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
