/**
 * GPU hot-path tests — content-addressed buffer caching, opt-in register
 * write log, ring short-view fallback, index-buffer padding.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../../public/js/gpu/xenos_registers.js';
import '../../public/js/gpu/register_file.js';
import '../../public/js/gpu/pm4.js';
import '../../public/js/gpu/command_processor.js';
import '../../public/js/gpu/vertex_fetch.js';
import '../../public/js/gpu/webgpu_backend.js';
import '../../public/js/gpu/ring_buffer.js';

const { XenosRegisterFile } = globalThis.XenonGpuRegisterFile;
const { XenosCommandProcessor } = globalThis.XenonGpuCommandProcessor;
const { XenosWebGpuBackend } = globalThis.XenonWebGpuBackend;
const { XenosRingBuffer } = globalThis.XenonGpuRingBuffer;
const PM4 = globalThis.XenonPM4;

globalThis.GPUBufferUsage = { VERTEX: 0x20, INDEX: 0x10, COPY_DST: 0x08 };

// microcode fixture (same as test_gpu_vertex_binding)
function execCf(address, count, sequence, opcode = 2) {
  return { dword0: (address & 0xFFF) | ((count & 0x7) << 12) | ((sequence & 0xFFF) << 16), dword1: opcode & 0xF };
}
function cfPair(a, b) {
  return [a.dword0 >>> 0, (a.dword1 & 0xFFFF) | ((b.dword0 & 0xFFFF) << 16), ((b.dword0 >>> 16) & 0xFFFF) | ((b.dword1 & 0xFFFF) << 16)];
}
const NOP_CF = { dword0: 0, dword1: 0 };
function vfetchWords({ dstReg = 1, srcReg = 0, constIndex = 0, constIndexSel = 0, dstSwiz = 0x688, format = 57, stride = 3, offset = 0, expAdjust = 0, isSigned = 0 } = {}) {
  const w0 = ((srcReg & 0x3F) << 5) | ((dstReg & 0x3F) << 12) | (1 << 19) | ((constIndex & 0x1F) << 20) | ((constIndexSel & 0x3) << 25);
  const w1 = (dstSwiz & 0xFFF) | ((isSigned & 1) << 12) | (format << 16) | ((expAdjust & 0x3F) << 24);
  const w2 = (stride & 0xFF) | ((offset & 0x7FFFFF) << 8);
  return [w0 >>> 0, w1 >>> 0, w2 >>> 0];
}
const ALU_ZERO = [0, 0, 0];
const FETCH_VS = new Uint32Array([
  ...cfPair(execCf(1, 1, 0b01, 2), NOP_CF),
  ...vfetchWords({}),
  ...ALU_ZERO,
]);
const VB_BASE = 0x00100000, IB_BASE = 0x00200000;

function makeGuest() {
  const arena = new Uint8Array(0x02000000);
  const view = new DataView(arena.buffer);
  for (let i = 0; i < 9; i++) view.setFloat32(VB_BASE + i * 4, i + 1, false);
  for (let i = 0; i < 3; i++) view.setUint16(IB_BASE + i * 2, i, false);
  return {
    arena,
    readGuestMemory(addr) { return new DataView(arena.buffer, addr, arena.length - addr); },
    writeGuestMemory(addr, bytes) { arena.set(bytes, addr); },
    writeU32BE(addr, v) { new DataView(arena.buffer).setUint32(addr, v >>> 0, false); },
    readU32BE(addr) { return new DataView(arena.buffer).getUint32(addr, false); },
  };
}
const FETCH_CONSTS = (() => { const c = new Uint32Array(192); c[0] = (VB_BASE & 0x1FFFFFFC) | 3; c[1] = 9 << 2; return c; })();

function makeBackend(guest) {
  const calls = { buf: 0, write: 0 };
  const device = {
    queue: { writeBuffer() { calls.write++; }, submit() {} },
    createBuffer(desc) { calls.buf++; return { desc, destroy() { this.destroyed = true; } }; },
    createTexture() { return {}; },
    createShaderModule() { return {}; },
    createRenderPipeline() { return {}; },
    createCommandEncoder() {
      const pass = { setPipeline() {}, setVertexBuffer() {}, setIndexBuffer() {}, draw() {}, drawIndexed() {}, end() {} };
      return { beginRenderPass() { return pass; }, finish() { return {}; } };
    },
    destroy() {},
  };
  const backend = new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory, placeholderShader: true });
  backend.device = device;
  backend.format = 'bgra8unorm';
  const canvas = { width: 64, height: 64, getContext: () => ({ getCurrentTexture: () => ({ createView: () => ({}) }) }) };
  backend.canvas = canvas;
  backend.context = canvas.getContext('webgpu');
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  return { backend, calls };
}
const drawParams = {
  primType: 4, primTypeName: 'triangle-list', numIndices: 3,
  isIndexed: true, indexGuestBase: IB_BASE, indexFormat: 0, indexEndianness: 1, indexLengthWords: 2,
  vertexFetchConstants: FETCH_CONSTS,
};

test('identical draws upload each GPU buffer exactly once (cache hit thereafter)', () => {
  const guest = makeGuest();
  const { backend, calls } = makeBackend(guest);
  backend.draw({ ...drawParams });
  backend.draw({ ...drawParams });
  backend.draw({ ...drawParams });
  assert.equal(calls.buf, 2, 'one vertex buffer + one index buffer created across 3 draws');
  assert.equal(calls.write, 2, 'one writeBuffer per unique buffer');
  assert.equal(backend.uploads.vertexBufferCacheHits, 2);
  assert.equal(backend.uploads.indexBufferCacheHits, 2);
  assert.equal(backend.uploads.vertexBuffers, 1);
  assert.equal(backend.uploads.indexBuffers, 1);
});

test('guest mutating the same buffer address is noticed (content hash) and re-uploaded', () => {
  const guest = makeGuest();
  const { backend, calls } = makeBackend(guest);
  backend.draw({ ...drawParams });
  const view = new DataView(guest.arena.buffer);
  view.setFloat32(VB_BASE, 42.5, false);
  backend.draw({ ...drawParams });
  assert.equal(calls.buf, 3, 'vertex buffer re-uploaded after content change');
  assert.equal(backend.uploads.vertexBuffers, 2);
  assert.equal(backend.uploads.vertexBufferCacheHits, 0);
});

test('buffer cache is bounded by count: oldest entry evicted (and destroyed)', () => {
  const guest = makeGuest();
  const { backend, calls } = makeBackend(guest);
  backend.gpuBufferCacheLimit = 1; // only one cached entry at a time
  backend.draw({ ...drawParams });
  // change vertex content → forces a new upload → eviction of the oldest entry (the index one or the vb — whichever is oldest)
  const view = new DataView(guest.arena.buffer);
  view.setFloat32(VB_BASE, 7.25, false);
  backend.draw({ ...drawParams });
  assert.ok(backend.gpuBufferCache.size <= 1, 'cache holds at most one entry with limit 1');
});

test('index buffer for an odd int16 count is padded to 4-byte buffer size', () => {
  const guest = makeGuest();
  const { backend, calls } = makeBackend(guest);
  const sizes = [];
  const origCreate = backend.device.createBuffer;
  backend.device.createBuffer = (desc) => { const b = origCreate(desc); sizes.push(desc.size); return b; };
  backend.draw({ ...drawParams, numIndices: 3, indexFormat: 0 }); // 3 × int16 = 6 bytes → 8
  assert.ok(sizes.includes(8), `padded index buffer of 8 bytes created (got ${sizes})`);
});

test('register write log is off by default and opt-in via enableWriteLog()', () => {
  const rf = new XenosRegisterFile();
  rf.write(0x2104, 0xdeadbeef);
  assert.deepEqual(rf.takeWriteLog(), [], 'nothing recorded without opt-in');
  rf.enableWriteLog();
  rf.write(0x2104, 0xdeadbeef);
  rf.write(0x2105, 0x0badf00d);
  const log = rf.takeWriteLog();
  assert.deepEqual(log, [{ index: 0x2104, value: 0xdeadbeef }, { index: 0x2105, value: 0x0badf00d }]);
  rf.enableWriteLog(false);
  rf.write(0x2106, 1);
  assert.deepEqual(rf.takeWriteLog(), [], 'recording stopped after enableWriteLog(false)');
  assert.equal(rf.writeCount, 4, 'writeCount always counts');
});

test('ring drain works with a short-view readGuestMemory (per-dword fallback)', () => {
  const guest = makeGuest();
  const RING = 0x01000000, OUT = 0x00110000;
  const header = PM4.makePacketType3(PM4.PM4_OPCODE.PM4_MEM_WRITE, 2);
  const shortViewGuest = {
    ...guest,
    readGuestMemory(addr) { const a = addr >>> 0; return new DataView(guest.arena.buffer, a, Math.min(4, guest.arena.length - a)); },
  };
  const backend = new XenosWebGpuBackend({ readGuestMemory: shortViewGuest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  const cp = new XenosCommandProcessor({ backend, readGuestMemory: shortViewGuest.readGuestMemory });
  const ring = new XenosRingBuffer({ commandProcessor: cp, readGuestMemory: shortViewGuest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  ring.initializeRingBuffer(RING, 4);
  // the kernel zeroes the ring at init (xenia memset) — packet goes in afterwards
  guest.writeU32BE(RING + 0, header);
  guest.writeU32BE(RING + 4, OUT | 0 /* endianness kNone */);
  guest.writeU32BE(RING + 8, 0xcafebabe);
  const result = ring.updateWritePointer(3);
  assert.equal(result.executed, true);
  assert.equal(shortViewGuest.readU32BE(OUT), 0xcafebabe);
});

test('ring drain executes wraparound splits across two readGuestMemory runs', () => {
  const guest = makeGuest();
  const RING = 0x01000000;
  const sizeBytes = 1 << (4 + 3); // size_log2 = 4 → 128 bytes
  // fill the ring: 2 NOP words at the tail, 2 NOP words at the head; read index starts at sizeDwords-2 after a first drain
  const cp = new XenosCommandProcessor({ backend: new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory }), readGuestMemory: guest.readGuestMemory });
  const ring = new XenosRingBuffer({ commandProcessor: cp, readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  ring.initializeRingBuffer(RING, 4);
  const words = Array.from({ length: 32 }, () => PM4.makePacketType2());
  words.forEach((w, i) => guest.writeU32BE(RING + i * 4, w));
  ring.updateWritePointer(31); // consume 31 dwords, read pointer lands at dword 31 (ring offset 124)
  guest.writeU32BE(RING + 124, PM4.makePacketType2()); // last dword of the ring
  guest.writeU32BE(RING + 0, PM4.makePacketType2());   // first dword (wraps)
  const result = ring.updateWritePointer(33); // one at 124, one at 0 — must wrap
  assert.equal(result.executed, true);
  assert.equal(result.ringWraps, 1);
  assert.equal(ring.getReadIndex(), 33);
});

test('prepareVertexFetch memoizes on (code, fetch-constant snapshot); constant change recomputes', () => {
  const guest = makeGuest();
  const { backend } = makeBackend(guest);
  const p1 = backend.prepareVertexFetch({ fetchConstants: FETCH_CONSTS });
  const p2 = backend.prepareVertexFetch({ fetchConstants: FETCH_CONSTS });
  assert.equal(p1, p2, 'same inputs → same plan object (memoized)');
  const consts = new Uint32Array(192);
  consts[0] = (VB_BASE & 0x1FFFFFFC) | 3;
  consts[1] = (8 << 2); // one dword smaller
  const p3 = backend.prepareVertexFetch({ fetchConstants: consts });
  assert.notEqual(p3, p1, 'fetch constants changed → fresh plan');
});

test('plan memo is per shader code object; setShaders drops the memo', () => {
  const guest = makeGuest();
  const { backend } = makeBackend(guest);
  const p1 = backend.prepareVertexFetch({ fetchConstants: FETCH_CONSTS });
  const code2 = Uint32Array.from(FETCH_VS); // identical content, different object
  backend.shaders.vertex = { kind: 'vertex', code: code2, dwords: code2.length };
  const p2 = backend.prepareVertexFetch({ vertexShaderCode: code2 });
  assert.notEqual(p1, p2, 'new code object → fresh analysis');
  backend.setShaders({ vertex: { kind: 'vertex', code: code2, dwords: code2.length } });
  const p3 = backend.prepareVertexFetch({ vertexShaderCode: code2 });
  assert.notEqual(p2, p3, 'setShaders dropped the memo');
});
