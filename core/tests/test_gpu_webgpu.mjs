/**
 * WebGPU Xenos backend tests — everything that can run without a GPU.
 * The backend is browser-only at runtime (navigator.gpu); in node we assert
 * the honest failure modes and the guest-memory plumbing that does not need
 * a device (memWrite routing, constructor validation, destroy idempotency).
 * Actual attach()/draw() against a real canvas is covered by the manual
 * checklist in docs/XENOS_GPU.md until a headless WebGPU implementation is
 * available in CI.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../../public/js/gpu/xenos_registers.js';
import '../../public/js/gpu/register_file.js';
import '../../public/js/gpu/pm4.js';
import '../../public/js/gpu/command_processor.js';
import '../../public/js/gpu/webgpu_backend.js';
import '../../public/js/gpu/render_state.js';

const { XenosWebGpuBackend, XenosWebGpuError } = globalThis.XenonWebGpuBackend;
const PM4 = globalThis.XenonPM4;
const { XenosCommandProcessor } = globalThis.XenonGpuCommandProcessor;

function makeGuest(size = 0x02000000) {
  const arena = new Uint8Array(size);
  return {
    arena,
    readGuestMemory(addr) { return new DataView(arena.buffer, addr, arena.length - addr); },
    writeGuestMemory(addr, bytes) { arena.set(bytes, addr); },
  };
}

test('constructor requires readGuestMemory', () => {
  assert.throws(() => new XenosWebGpuBackend({}), TypeError);
});

test('attach() fails with an actionable error when WebGPU is unavailable (node)', async () => {
  const guest = makeGuest();
  const backend = new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  await assert.rejects(backend.attach(), (e) => e instanceof XenosWebGpuError && /navigator\.gpu|WebGPU/.test(e.message));
});

test('memWrite routes PM4 memory writes into guest memory', () => {
  const guest = makeGuest();
  const backend = new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  backend.memWrite(0x01000000, new Uint8Array([0xde, 0xad, 0xbe, 0xef]));
  assert.equal(new DataView(guest.arena.buffer).getUint32(0x01000000, false), 0xdeadbeef);
});

test('memWrite before wiring writeGuestMemory fails loudly, never drops silently', () => {
  const guest = makeGuest();
  const backend = new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory });
  assert.throws(() => backend.memWrite(0x01000000, new Uint8Array([1])), (e) => /writeGuestMemory not wired/.test(e.message));
});

test('draw() before attach() fails loudly', () => {
  const guest = makeGuest();
  const backend = new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  assert.throws(() => backend.draw({ primType: 4, numIndices: 3 }), (e) => /attach\(\)/.test(e.message));
});

test('end-to-end: command processor + backend without a GPU — registers update, draw traps honestly', () => {
  const guest = makeGuest();
  const backend = new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  const cp = new XenosCommandProcessor({ backend, readGuestMemory: guest.readGuestMemory });
  // program register 0x2104 via a properly encoded type-0 packet
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
  {
    const words = [PM4.makePacketType0(0x2104, 1), 0xffffffff];
    const buf = new Uint8Array(words.length * 4);
    const view = new DataView(buf.buffer);
    words.forEach((w, i) => view.setUint32(i * 4, w >>> 0, false));
    cp.executeCommandBuffer(buf);
  }
  assert.equal(cp.registers.read(0x2104), 0xffffffff);
  // draw via a properly encoded packet buffer
  const words = [PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator];
  const buf = new Uint8Array(words.length * 4);
  const view = new DataView(buf.buffer);
  words.forEach((w, i) => view.setUint32(i * 4, w >>> 0, false));
  // draw() cannot be reached without attach() (no GPU in node). The honest
  // trap is 'draw before attach()'; the 'not wired yet' trap sits behind a
  // real device and is exercised in the browser bring-up (docs/XENOS_GPU.md).
  assert.throws(() => cp.executeCommandBuffer(buf), (e) => /draw before attach/.test(e.message));
});

test('topologyForPrimType maps the WebGPU-encodable primitive types and traps the rest', () => {
  const { topologyForPrimType } = globalThis.XenonWebGpuBackend._internal;
  assert.equal(topologyForPrimType(1, 'kPointList'), 'point-list');
  assert.equal(topologyForPrimType(2, 'kLineList'), 'line-list');
  assert.equal(topologyForPrimType(3, 'kLineStrip'), 'line-strip');
  assert.equal(topologyForPrimType(4, 'kTriangleList'), 'triangle-list');
  assert.equal(topologyForPrimType(6, 'kTriangleStrip'), 'triangle-strip');
  for (const [prim, name] of [[5, 'kTriangleFan'], [7, 'kTriangleWithWFlags'], [8, 'kRectangleList'], [0x0c, 'kLineLoop'], [0x0d, 'kQuadList'], [0x0f, 'kPolygon']]) {
    assert.throws(() => topologyForPrimType(prim, name), (e) => e instanceof XenosWebGpuError && /topology is not encodable/.test(e.message), name);
  }
});

test('isIntegerGpuVertexFormat flags only pure uint/sint formats', () => {
  const { isIntegerGpuVertexFormat } = globalThis.XenonWebGpuBackend._internal;
  for (const f of ['uint32', 'uint32x2', 'uint32x4', 'uint8x4', 'uint16x2', 'sint32', 'sint8x4', 'sint16x4']) {
    assert.equal(isIntegerGpuVertexFormat(f), true, f);
  }
  for (const f of ['float32', 'float32x3', 'float16x2', 'unorm8x4', 'snorm16x4', 'unorm10-10-10-2']) {
    assert.equal(isIntegerGpuVertexFormat(f), false, f);
  }
});
