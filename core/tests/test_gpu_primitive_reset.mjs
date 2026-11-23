/**
 * Xenos GPU primitive-reset (multi-primitive index buffer) tests.
 *
 * xenia's primitive processor honors the guest's primitive reset index
 * (VGT_MULTI_PRIM_IB_RESET_INDX) only when PA_SU_SC_MODE_CNTL.multi_prim_ib_ena
 * is set, and only for the strip-like primitive types — the list topologies are
 * filtered out on purpose. The converted topologies (triangle fan, line loop)
 * are split at the reset index and every range is converted on its own
 * (xenia's GetMultiPrimitiveHostIndexCountAndRanges + ConvertSinglePrimitiveRanges);
 * the topologies WebGPU has natively (line strip, triangle strip) are drawn as
 * one draw per range, which is what the hardware's primitive restart does.
 *
 * The command processor must hand the guest's reset configuration to the
 * backend, and the backend must split the index buffer exactly like xenia.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../../public/js/gpu/xenos_registers.js';
import '../../public/js/gpu/register_file.js';
import '../../public/js/gpu/pm4.js';
import '../../public/js/gpu/command_processor.js';
import '../../public/js/gpu/vertex_fetch.js';
import '../../public/js/gpu/webgpu_backend.js';
import '../../public/js/gpu/render_state.js';

const PM4 = globalThis.XenonPM4;
const { XenosCommandProcessor } = globalThis.XenonGpuCommandProcessor;
const { XenosWebGpuBackend } = globalThis.XenonWebGpuBackend;

globalThis.GPUBufferUsage = { VERTEX: 0x20, INDEX: 0x10, COPY_DST: 0x08 };
globalThis.GPUTextureUsage = { COPY_SRC: 0x01, COPY_DST: 0x02, TEXTURE_BINDING: 0x04, STORAGE_BINDING: 0x08, RENDER_ATTACHMENT: 0x10 };

const IB_BASE = 0x00200000;

/** A guest arena with a 16-bit index buffer at IB_BASE (big-endian, as the guest stores it). */
function makeGuest(indices) {
  const arena = new Uint8Array(0x02000000);
  const view = new DataView(arena.buffer);
  indices.forEach((v, i) => view.setUint16(IB_BASE + i * 2, v, false));
  return {
    arena,
    readGuestMemory(addr) { return new DataView(arena.buffer, addr, arena.length - addr); },
    writeGuestMemory(addr, bytes) { arena.set(bytes, addr); },
    writeU32BE(addr, value) { new DataView(arena.buffer).setUint32(addr, value >>> 0, false); },
  };
}

/** A fake WebGPU device that records every writeBuffer payload. */
function makeDevice(writes) {
  return {
    queue: { writeBuffer(buffer, offset, data) { writes.push(Uint8Array.from(data)); }, submit() {} },
    createBuffer(desc) { return { desc, destroy() { this.destroyed = true; } }; },
    createTexture() { return {}; },
    createShaderModule() { return {}; },
    createRenderPipeline() { return {}; },
    createCommandEncoder() {
      const pass = { setPipeline() {}, setVertexBuffer() {}, setIndexBuffer() {}, setBlendConstant() {}, setStencilReference() {}, setScissorRect() {}, draw() {}, drawIndexed() {}, end() {} };
      return { beginRenderPass() { return pass; }, finish() { return {}; } };
    },
    destroy() {},
  };
}

function makeBackend(guest) {
  const writes = [];
  const backend = new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  backend.device = makeDevice(writes);
  backend.format = 'bgra8unorm';
  return { backend, writes };
}

/** The uploaded index buffer as an array of 16-bit values (little-endian on the GPU). */
function uploadedIndices16(writes, count = null) {
  const out = [];
  const view = new DataView(writes[0].buffer);
  const n = count === null ? writes[0].byteLength >> 1 : count;
  for (let i = 0; i < n; i++) out.push(view.getUint16(i * 2, true));
  return out;
}

/** The uploaded index buffer as an array of 32-bit values (little-endian on the GPU). */
function uploadedIndices32(writes, count = null) {
  const out = [];
  const view = new DataView(writes[0].buffer);
  const n = count === null ? writes[0].byteLength >> 2 : count;
  for (let i = 0; i < n; i++) out.push(view.getUint32(i * 4, true));
  return out;
}

function beWords(words) {
  const buf = new Uint8Array(words.length * 4);
  const view = new DataView(buf.buffer);
  words.forEach((w, i) => view.setUint32(i * 4, w >>> 0, false));
  return buf;
}

test('the command processor hands the guest\'s primitive-reset configuration to the backend', () => {
  const guest = makeGuest([0, 1, 2]);
  const calls = [];
  const backend = {
    calls,
    memWrite() {},
    draw(params) { calls.push(params); },
    eventWrite() {},
    interrupt() {},
    swap() {},
  };
  const cp = new XenosCommandProcessor({ backend, readGuestMemory: guest.readGuestMemory });
  // PA_SU_SC_MODE_CNTL.multi_prim_ib_ena (bit 21) + the reset index itself.
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2205, 1), 0x00200000]));
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2103, 1), 0x0000ffff]));
  // xenia's draw gate: RB_MODECONTROL = kColorDepth, RB_SURFACE_INFO.pitch != 0.
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2208, 1), 0x00000004]));
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2000, 1), 0x00001000]));
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kDMA << 6) | PM4.PRIMITIVE_TYPE.kTriangleFan;
  cp.executeCommandBuffer(beWords([
    PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX, 4),
    0x00000000, initiator, IB_BASE, (3 << 2) | (1 << 30),
  ]));
  assert.equal(calls.length, 1, 'one draw issued');
  assert.equal(calls[0].multiPrimIbEnable, true, 'multi_prim_ib_ena reached the backend');
  assert.equal(calls[0].multiPrimResetIndex, 0x0000ffff, 'VGT_MULTI_PRIM_IB_RESET_INDX reached the backend');
});

test('without multi_prim_ib_ena the reset index is an ordinary index (xenia)', () => {
  const guest = makeGuest([0, 1, 2, 3, 4]);
  const { backend, writes } = makeBackend(guest);
  // A triangle fan over [0,1,2,3,4] — no reset enabled, so nothing is split.
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 5, indexFormat: 0, indexEndianness: 0,
    primType: 0x05 /* kTriangleFan */, multiPrimIbEnable: false, multiPrimResetIndex: 0xffff,
  });
  // xenia's TriangleFanToList emits (previous, current, first) per triangle.
  assert.deepEqual(uploadedIndices16(writes, ib.numIndices), [1, 2, 0, 2, 3, 0, 3, 4, 0]);
});

test('with multi_prim_ib_ena the fan is split at the reset index (xenia)', () => {
  const guest = makeGuest([0, 1, 2, 3, 0xffff, 4, 5, 6]);
  const { backend, writes } = makeBackend(guest);
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 8, indexFormat: 0, indexEndianness: 0,
    primType: 0x05 /* kTriangleFan */, multiPrimIbEnable: true, multiPrimResetIndex: 0xffff,
  });
  // Two ranges: [0,1,2,3] → (1,2,0),(2,3,0) and [4,5,6] → (5,6,4).
  assert.deepEqual(uploadedIndices16(writes, ib.numIndices), [1, 2, 0, 2, 3, 0, 5, 6, 4]);
  assert.equal(ib.numIndices, 9, 'host index count is the sum of the converted ranges');
});

test('the line loop is split at the reset index too (xenia)', () => {
  const guest = makeGuest([0, 1, 0xffff, 2, 3]);
  const { backend, writes } = makeBackend(guest);
  backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 5, indexFormat: 0, indexEndianness: 0,
    primType: 0x0c /* kLineLoop */, multiPrimIbEnable: true, multiPrimResetIndex: 0xffff,
  });
  // Range [0,1] → strip (0,1,0); range [2,3] → strip (2,3,2).
  assert.deepEqual(uploadedIndices16(writes), [0, 1, 0, 2, 3, 2]);
});

test('a reset range that cannot complete a primitive is dropped, not drawn (xenia)', () => {
  // The range after the reset marker holds a single index — xenia's
  // GetLineLoopStripIndexCount returns 0 for it, so nothing is drawn for it.
  const guest = makeGuest([0, 1, 0xffff, 2]);
  const { backend, writes } = makeBackend(guest);
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 4, indexFormat: 0, indexEndianness: 0,
    primType: 0x0c /* kLineLoop */, multiPrimIbEnable: true, multiPrimResetIndex: 0xffff,
  });
  assert.deepEqual(uploadedIndices16(writes, ib.numIndices), [0, 1, 0]);
  assert.equal(ib.numIndices, 3, 'the one-index range contributes nothing');
});

test('the quad list is a list topology — the reset index does not split it (xenia)', () => {
  const guest = makeGuest([0, 1, 2, 3, 0xffff, 4, 5, 6]);
  const { backend, writes } = makeBackend(guest);
  backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 8, indexFormat: 0, indexEndianness: 0,
    primType: 0x0d /* kQuadList */, multiPrimIbEnable: true, multiPrimResetIndex: 0xffff,
  });
  // Two quads: (0,1,2),(0,2,3) and (0xFFFF,4,5),(0xFFFF,5,6).
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 8, indexFormat: 0, indexEndianness: 0,
    primType: 0x0d /* kQuadList */, multiPrimIbEnable: true, multiPrimResetIndex: 0xffff,
  });
  assert.deepEqual(uploadedIndices16(writes, ib.numIndices), [0, 1, 2, 0, 2, 3, 0xffff, 4, 5, 0xffff, 5, 6]);
});

test('a 32-bit reset index compares only the low 24 bits (xenia\'s vertex index mask)', () => {
  const guest = makeGuest([]);
  const view = new DataView(guest.arena.buffer);
  [0, 1, 2, 3, 0x00ffffff, 4, 5, 6].forEach((v, i) => view.setUint32(IB_BASE + i * 4, v, false));
  const { backend, writes } = makeBackend(guest);
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 8, indexFormat: 1, indexEndianness: 0,
    primType: 0x05 /* kTriangleFan */, multiPrimIbEnable: true, multiPrimResetIndex: 0x00ffffff,
  });
  // The 0x00FFFFFF index ended the first range: (1,2,0),(2,3,0) and (5,6,4).
  assert.deepEqual(uploadedIndices32(writes, ib.numIndices), [1, 2, 0, 2, 3, 0, 5, 6, 4]);
});

test('a 32-bit index above the 24-bit mask does not end the range (xenia)', () => {
  const guest = makeGuest([]);
  const view = new DataView(guest.arena.buffer);
  [0, 1, 2, 3, 0x01000000, 4, 5, 6].forEach((v, i) => view.setUint32(IB_BASE + i * 4, v, false));
  const { backend, writes } = makeBackend(guest);
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 8, indexFormat: 1, indexEndianness: 0,
    primType: 0x05 /* kTriangleFan */, multiPrimIbEnable: true, multiPrimResetIndex: 0x00ffffff,
  });
  // 0x01000000 & 0xFFFFFF == 0, so it does not match the reset index and the
  // whole buffer stays one range — the fan is expanded as a single fan.
  assert.deepEqual(uploadedIndices32(writes, ib.numIndices), [
    1, 2, 0, 2, 3, 0, 3, 0x01000000, 0, 0x01000000, 4, 0, 4, 5, 0, 5, 6, 0,
  ]);
});

test('a triangle strip is drawn as one draw per reset range (xenia\'s primitive restart)', () => {
  const guest = makeGuest([0, 1, 2, 3, 0xffff, 4, 5, 6]);
  const { backend, writes } = makeBackend(guest);
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 8, indexFormat: 0, indexEndianness: 0,
    primType: 0x06 /* kTriangleStrip */, multiPrimIbEnable: true, multiPrimResetIndex: 0xffff,
  });
  // The index buffer keeps the guest's indices (the marker included) — the
  // ranges tell the draw path which indices form a strip.
  assert.deepEqual(uploadedIndices16(writes, ib.numIndices), [0, 1, 2, 3, 0xffff, 4, 5, 6]);
  assert.deepEqual(ib.ranges, [[0, 4], [5, 3]], 'two strips: [0..3] and [5..7]');
});

test('a line strip is split at the reset index too (xenia)', () => {
  const guest = makeGuest([0, 1, 0xffff, 2, 3]);
  const { backend } = makeBackend(guest);
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 5, indexFormat: 0, indexEndianness: 0,
    primType: 0x03 /* kLineStrip */, multiPrimIbEnable: true, multiPrimResetIndex: 0xffff,
  });
  assert.deepEqual(ib.ranges, [[0, 2], [3, 2]], 'two line strips: [0,1] and [3,4]');
});

test('a triangle list is never split — the reset index is an ordinary index (xenia)', () => {
  const guest = makeGuest([0, 1, 2, 3, 0xffff]);
  const { backend } = makeBackend(guest);
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 5, indexFormat: 0, indexEndianness: 0,
    primType: 0x04 /* kTriangleList */, multiPrimIbEnable: true, multiPrimResetIndex: 0xffff,
  });
  assert.equal(ib.ranges, null, 'list topologies are filtered out of the reset handling');
});

test('a 16-bit index buffer with a 32-bit swap mode is normalized like xenia (8-in-32 → 8-in-16)', () => {
  const guest = makeGuest([0x0102, 0x0304]);
  const { backend, writes } = makeBackend(guest);
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 2, indexFormat: 0, indexEndianness: 3 /* k16in32 */,
    primType: 0x04 /* kTriangleList */, multiPrimIbEnable: false, multiPrimResetIndex: 0xffff,
  });
  // k16in32 on 16-bit indices is disabled by xenia → the guest bytes pass through.
  assert.deepEqual(uploadedIndices16(writes, ib.numIndices), [0x0102, 0x0304]);
});

test('a 16-bit index buffer with 8-in-32 is normalized to 8-in-16 (xenia)', () => {
  const guest = makeGuest([0x0102, 0x0304]);
  const { backend, writes } = makeBackend(guest);
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 2, indexFormat: 0, indexEndianness: 2 /* k8in32 */,
    primType: 0x04 /* kTriangleList */, multiPrimIbEnable: false, multiPrimResetIndex: 0xffff,
  });
  // xenia replaces 8-in-32 with 8-in-16 → the bytes inside every half word swap.
  assert.deepEqual(uploadedIndices16(writes, ib.numIndices), [0x0201, 0x0403]);
});

test('the reset index is compared in the guest\'s index endianness (xenia\'s GpuSwap)', () => {
  // The guest's 0x00FF reset index is swapped to 0xFF00 by k8in16, and the
  // guest's 0x00FF index is swapped to 0xFF00 as well — xenia compares the
  // swapped reset index against the swapped indices, so they still match.
  const guest = makeGuest([0x0000, 0x00ff, 0x0001]);
  const { backend } = makeBackend(guest);
  const ib = backend.uploadIndexBuffer({
    indexGuestBase: IB_BASE, numIndices: 3, indexFormat: 0, indexEndianness: 1 /* k8in16 */,
    primType: 0x06 /* kTriangleStrip */, multiPrimIbEnable: true, multiPrimResetIndex: 0x00ff,
  });
  assert.deepEqual(ib.ranges, [[0, 1], [2, 1]], 'the swapped 0xFF00 index ended the first strip');
});
