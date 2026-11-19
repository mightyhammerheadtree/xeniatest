/**
 * XenosRingBuffer tests — the doorbell path, end to end, against a flat
 * 64 MiB guest arena (the same shape the recompiled module sees).
 *
 * No mocks of the pipeline: packets are hand-assembled big-endian, executed
 * through the real XenosCommandProcessor, and observed through real register
 * file state + a recording backend (like test_gpu_pm4.mjs).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../../public/js/gpu/xenos_registers.js';
import '../../public/js/gpu/register_file.js';
import '../../public/js/gpu/pm4.js';
import '../../public/js/gpu/command_processor.js';
import '../../public/js/gpu/ring_buffer.js';
import '../../public/js/recomp/hle.js';       // side effect: globalThis.XenonHLE
import '../../public/js/recomp/hle_tables.js'; // side effect: globalThis.XenonHLETables
import '../../public/js/gpu/webgpu_backend.js';
import '../../public/js/gpu/gpu_host.js';     // side effect: globalThis.XenonGpuHostFactory
import '../../public/js/gpu/render_state.js';

const PM4 = globalThis.XenonPM4;
const { XenosCommandProcessor } = globalThis.XenonGpuCommandProcessor;
const { XenosRingBuffer, XenosRingError } = globalThis.XenonGpuRingBuffer;

const GUEST_SIZE = 64 * 1024 * 1024;

function makeGuest(size = GUEST_SIZE) {
  const arena = new Uint8Array(size);
  const be = new DataView(arena.buffer);
  return {
    arena,
    be,
    readGuestMemory(addr) { return new DataView(arena.buffer, addr, Math.min(4, size - addr)); },
    writeGuestMemory(addr, bytes) { arena.set(bytes, addr); },
    writeU32BE(addr, v) { be.setUint32(addr, v >>> 0, false); },
    readU32BE(addr) { return be.getUint32(addr, false) >>> 0; },
  };
}

function makeRecordingBackend(guest) {
  const calls = [];
  return {
    calls,
    memWrite(addr, bytes) { calls.push({ kind: 'memWrite', addr, bytes: Uint8Array.from(bytes) }); guest.arena.set(bytes, addr); },
    swap() { calls.push({ kind: 'swap' }); },
    draw(params) { calls.push({ kind: 'draw', params }); },
  };
}

function makeRing(guest) {
  const backend = makeRecordingBackend(guest);
  const cp = new XenosCommandProcessor({ backend, readGuestMemory: guest.readGuestMemory });
  const ring = new XenosRingBuffer({ commandProcessor: cp, readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  return { backend, cp, ring };
}

const RING_BASE = 0x00100000;
const WRITEBACK = 0x00200000;

test('initialize zeroes the ring and seeds read index 0', () => {
  const guest = makeGuest();
  const { ring } = makeRing(guest);
  guest.arena.fill(0xAA, RING_BASE, RING_BASE + 0x8000);
  const { sizeBytes } = ring.initializeRingBuffer(RING_BASE, 10); // 1 << 13 = 8192 bytes
  assert.equal(sizeBytes, 8192);
  assert.ok(guest.arena.subarray(RING_BASE, RING_BASE + 8192).every((b) => b === 0), 'ring was zeroed');
  assert.equal(guest.arena[RING_BASE + 8192], 0xAA, 'bytes after the ring are untouched');
  assert.equal(ring.getReadIndex(), 0);
  assert.equal(ring.getSizeBytes(), 8192);
});

test('initialize rejects an absurd size_log2', () => {
  const guest = makeGuest();
  const { ring } = makeRing(guest);
  assert.throws(() => ring.initializeRingBuffer(RING_BASE, 29), /too large/);
});

test('doorbell before initialization traps loudly', () => {
  const guest = makeGuest();
  const { ring } = makeRing(guest);
  assert.throws(() => ring.updateWritePointer(1), XenosRingError);
});

test('doorbell executes the pending packets and advances the read pointer', () => {
  const guest = makeGuest();
  const { cp, ring } = makeRing(guest);
  ring.initializeRingBuffer(RING_BASE, 4); // 1 << 7 = 128 bytes
  // ring contents: type-0 write reg 0x2104 <- 0xdeadbeef ; then type-2 no-op
  guest.writeU32BE(RING_BASE + 0, PM4.makePacketType0(0x2104, 1));
  guest.writeU32BE(RING_BASE + 4, 0xdeadbeef);
  guest.writeU32BE(RING_BASE + 8, PM4.makePacketType2());
  const result = ring.updateWritePointer(3);
  assert.equal(result.executed, true);
  assert.equal(result.words, 3);
  assert.equal(cp.registers.read(0x2104), 0xdeadbeef >>> 0);
  assert.equal(ring.getReadIndex(), 3);
  assert.equal(cp.stats.packets, 2);
  // a second doorbell with the same pointer is a no-op
  const again = ring.updateWritePointer(3);
  assert.equal(again.executed, false);
  assert.equal(again.reason, 'ring empty');
});

test('doorbell executes across the ring wraparound boundary', () => {
  const guest = makeGuest();
  const { cp, ring } = makeRing(guest);
  // Tiny ring: 2 dwords.
  ring.initializeRingBuffer(RING_BASE, 0); // 1 << 3 = 8 bytes
  // First drain: fill both slots with no-ops and consume them (read pointer now at 2)
  guest.writeU32BE(RING_BASE + 0, PM4.makePacketType2());
  guest.writeU32BE(RING_BASE + 4, PM4.makePacketType2());
  ring.updateWritePointer(2);
  assert.equal(ring.getReadIndex(), 2);
  // Now read index = 2 (ring offset 0). Submit: type-0 reg write at index 2..3,
  // which wraps: index 2 -> offset 0, index 3 -> offset 4. Wait — index 2,3 map
  // to offsets 0,4 again. To cross the boundary we need read=1. Re-init for a clean scenario:
  ring.initializeRingBuffer(RING_BASE, 0);
  guest.writeU32BE(RING_BASE + 0, PM4.makePacketType2()); // consumed now
  ring.updateWritePointer(1);
  assert.equal(ring.getReadIndex(), 1);
  // write pointer wraps: submit dwords at absolute indices 1 and 2
  guest.writeU32BE(RING_BASE + 4, PM4.makePacketType0(0x2104, 1)); // index 1 -> offset 4
  guest.writeU32BE(RING_BASE + 0, 0xfeedface);                      // index 2 -> offset 0 (wrapped)
  const result = ring.updateWritePointer(3); // delta = 2, crosses the wrap
  assert.equal(result.executed, true);
  assert.equal(result.ringWraps, 1);
  assert.equal(cp.registers.read(0x2104), 0xfeedface >>> 0);
  assert.equal(ring.getReadIndex(), 3);
});

test('doorbell overflow (pointer more than one ring ahead) traps loudly', () => {
  const guest = makeGuest();
  const { ring } = makeRing(guest);
  ring.initializeRingBuffer(RING_BASE, 4); // 128 bytes = 32 dwords
  assert.throws(() => ring.updateWritePointer(0xFFFFFFFF), (e) => e instanceof XenosRingError && /more than one ring/.test(e.message));
});

test('read-pointer writeback stores the new pointer big-endian at the configured address', () => {
  const guest = makeGuest();
  const { ring } = makeRing(guest);
  ring.initializeRingBuffer(RING_BASE, 4);
  ring.enableReadPointerWriteBack(WRITEBACK, 6);
  guest.writeU32BE(RING_BASE + 0, PM4.makePacketType2());
  ring.updateWritePointer(1);
  assert.equal(guest.readU32BE(WRITEBACK), 1);
  const state = ring.getState();
  assert.equal(state.readPtrWritebackAddr, WRITEBACK >>> 0);
  assert.equal(state.readPtrUpdateFrequency, 16); // 1 << 6 >> 2
});

test('uninitialized-pointer sentinel (0xBAADF00D) is reported, not executed', () => {
  const guest = makeGuest();
  const { ring } = makeRing(guest);
  ring.initializeRingBuffer(RING_BASE, 4);
  const result = ring.updateWritePointer(0xBAADF00D);
  assert.equal(result.executed, false);
  assert.match(result.reason, /BAADF00D/);
  assert.equal(ring.getReadIndex(), 0);
});

test('end-to-end: ring -> command processor -> draw packet reaches the backend', () => {
  const guest = makeGuest();
  const { backend, ring } = makeRing(guest);
  ring.initializeRingBuffer(RING_BASE, 4);
  // VGT_DRAW_INITIATOR: auto-indexed triangle list, 3 indices
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
  const words = [PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator];
  words.forEach((w, i) => guest.writeU32BE(RING_BASE + i * 4, w));
  const result = ring.updateWritePointer(words.length);
  assert.equal(result.executed, true);
  const draws = backend.calls.filter((c) => c.kind === 'draw');
  assert.equal(draws.length, 1);
  assert.equal(draws[0].params.numIndices, 3);
});

test('vd exports forward to the attached GPU host and trap loudly without one', () => {
  const { createHleHost, HleError } = globalThis.XenonHLE;
  // host absent → loud trap naming the export
  const memory = new WebAssembly.Memory({ initial: 1024, maximum: 1024 });
  const h = createHleHost({});
  h.attachMemory(memory);
  const HLE_CTX_ADDR = 0x00F11000;
  const view = new DataView(memory.buffer);
  const setReg = (i, v) => view.setBigUint64(HLE_CTX_ADDR + i * 8, v, false);
  setReg(3, 0x00100000n); // ptr
  setReg(4, 10n);         // size_log2
  const saved = globalThis.XenonGpuHost;
  delete globalThis.XenonGpuHost;
  assert.throws(() => h.call('xboxkrnl', 0x01c3), (e) => e instanceof HleError && /VdInitializeRingBuffer.*GPU host not attached/.test(e.message));
  assert.throws(() => h.call('xboxkrnl', 0x01b6), (e) => e instanceof HleError && /VdEnableRingBufferRPtrWriteBack.*GPU host not attached/.test(e.message));
  // host present → receives the arguments, low 32 bits, and returns STATUS_SUCCESS-ish 0n
  const calls = [];
  globalThis.XenonGpuHost = {
    initializeRingBuffer: (ptr, sizeLog2) => { calls.push(['init', ptr, sizeLog2]); },
    enableReadPointerWriteBack: (ptr, blockLog2) => { calls.push(['writeback', ptr, blockLog2]); },
  };
  assert.equal(h.call('xboxkrnl', 0x01c3), 0n);
  assert.equal(h.call('xboxkrnl', 0x01b6), 0n);
  assert.deepEqual(calls, [
    ['init', 0x00100000n, 10n],
    ['writeback', 0x00100000n, 10n],
  ]);
  if (saved === undefined) delete globalThis.XenonGpuHost; else globalThis.XenonGpuHost = saved;
});

test('XenonGpuHost end-to-end: HLE Vd exports drive the same ring the page uses', () => {
  const memory = new WebAssembly.Memory({ initial: 1024, maximum: 1024 });
  const savedHost = globalThis.XenonGpuHost;
  const host = new globalThis.XenonGpuHostFactory({ memory, onLog: () => {} });
  globalThis.XenonGpuHost = host;
  try {
    const { createHleHost } = globalThis.XenonHLE;
    const h = createHleHost({});
    h.attachMemory(memory);
    const HLE_CTX_ADDR = 0x00F11000;
    const view = new DataView(memory.buffer);
    const setReg = (i, v) => view.setBigUint64(HLE_CTX_ADDR + i * 8, v, false);
    const RING = 0x00100000, WB = 0x00200000;
    // guest: VdInitializeRingBuffer(0x00100000, size_log2=4 → 1 << 7 = 128-byte ring)
    setReg(3, BigInt(RING)); setReg(4, 4n);
    assert.equal(h.call('xboxkrnl', 0x01c3), 0n);
    // guest stages a command buffer in the ring: type-0 write reg 0x2104 <- 0xC0FFEE00
    const words = [PM4.makePacketType0(0x2104, 1), 0xc0ffee00];
    words.forEach((w, i) => view.setUint32(RING + i * 4, w >>> 0, false));
    // guest: VdEnableRingBufferRPtrWriteBack(0x00200000, block_log2=6)
    setReg(3, BigInt(WB)); setReg(4, 6n);
    assert.equal(h.call('xboxkrnl', 0x01b6), 0n);
    // guest kicks CP_RB_WPTR (the MMIO doorbell itself is not routed yet — see docs/XENOS_GPU.md)
    const result = host.notifyWritePointer(words.length);
    assert.equal(result.executed, true);
    assert.equal(host.commandProcessor.registers.read(0x2104), 0xc0ffee00 >>> 0);
    assert.equal(view.getUint32(WB >>> 0, false), words.length, 'read pointer written back big-endian');
    assert.equal(host.getState().ring.readPtrWritebackAddr, WB);
  } finally {
    if (savedHost === undefined) delete globalThis.XenonGpuHost; else globalThis.XenonGpuHost = savedHost;
  }
});
