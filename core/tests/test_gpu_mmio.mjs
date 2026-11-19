/**
 * GPU MMIO aperture tests — bring-up step 4.5.
 *
 * Guest loads/stores to [0x00F02000, 0x00F03000) (byte offset = register × 4)
 * are routed by the recompiler to xenorecomp.gpu_mmio_store / gpu_mmio_load,
 * which forward to globalThis.XenonGpuHost. CP_RB_WPTR (0x01C5, the doorbell)
 * drains the ring; CP_RB_RPTR (0x01C4) answers reads. Everything else traps.
 *
 * No browser: tests 1–2 use a recording host; test 3 drives the real
 * XenonGpuHost + ring + command processor with a MEM_WRITE packet.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { assemble } from '../../public/js/ppc_asm.js';
import { buildImage, recompileImage, makeImports, runImage, GPU_MMIO_BASE } from '../../public/js/recomp/recompiler.js';
import '../../public/js/gpu/xenos_registers.js';
import '../../public/js/gpu/register_file.js';
import '../../public/js/gpu/pm4.js';
import '../../public/js/gpu/command_processor.js';
import '../../public/js/gpu/ring_buffer.js';
import '../../public/js/gpu/webgpu_backend.js';
import '../../public/js/gpu/vertex_fetch.js';
import '../../public/js/gpu/render_state.js';
await import('../../public/js/gpu/gpu_host.js');
const XenonGpuHost = globalThis.XenonGpuHostFactory;

const PM4 = globalThis.XenonPM4;
const DOORBELL = GPU_MMIO_BASE + 0x01C5 * 4; // 0x00F02714
const RPTR = GPU_MMIO_BASE + 0x01C4 * 4;     // 0x00F02710

function withGpuHost(host, fn) {
  const saved = globalThis.XenonGpuHost;
  globalThis.XenonGpuHost = host;
  return Promise.resolve()
    .then(fn)
    .finally(() => {
      if (saved === undefined) delete globalThis.XenonGpuHost;
      else globalThis.XenonGpuHost = saved;
    });
}

const EXIT_DOOR = `
exit_door:
    li    r9, 0
    lis   r10, 0x00f0
    ori   r10, r10, 0x0ffc
    stb   r9, 0(r10)
    b     exit_door
`;

test('aperture store reaches the GPU host with register address and dword size', async () => {
  const calls = [];
  const host = {
    mmioStore(addr, value, size) { calls.push({ addr, value, size }); },
    mmioLoad(addr, size) { calls.push({ addr, size }); return 0n; },
  };
  const source = `
    li   r3, 2
    lis  r4, 0x00f0
    ori  r4, r4, 0x2714
    stw  r3, 0(r4)
` + EXIT_DOOR;
  const img = assemble(source);
  const built = buildImage('bin', null, img.bytes);
  await withGpuHost(host, async () => {
    const res = await runImage(built, {});
    assert.equal(res.exitCode, 0);
    // the aperture swallows the store: nothing lands at the aperture address in flat memory
    const arena = new Uint8Array(res.memory.buffer);
    assert.equal(arena[DOORBELL], 0);
  });
  assert.deepEqual(calls, [{ addr: DOORBELL, value: 2n, size: 4 }]);
});

test('aperture load routes to the GPU host and returns the value to the guest', async () => {
  const calls = [];
  const host = {
    mmioLoad(addr, size) { calls.push({ addr, size }); return 0x11223344n; },
    mmioStore() { throw new Error('not used'); },
  };
  // guest reads RPTR (0x00F02710 = -4 from the doorbell) and stores the dword at 0x00100000
  const source = `
    lis  r4, 0x00f0
    ori  r4, r4, 0x2714
    lwz  r5, -4(r4)
    lis  r6, 0x0010
    stw  r5, 0(r6)
` + EXIT_DOOR;
  const img = assemble(source);
  const built = buildImage('bin', null, img.bytes);
  let memory;
  await withGpuHost(host, async () => {
    const res = await runImage(built, {});
    assert.equal(res.exitCode, 0);
    memory = res.memory;
  });
  assert.deepEqual(calls, [{ addr: RPTR, size: 4 }]);
  assert.deepEqual([...new Uint8Array(memory.buffer, 0x00100000, 4)], [0x11, 0x22, 0x33, 0x44]);
});

test('GPU MMIO store without a GPU host traps loudly', async () => {
  const source = `
    li   r3, 2
    lis  r4, 0x00f0
    ori  r4, r4, 0x2714
    stw  r3, 0(r4)
` + EXIT_DOOR;
  const img = assemble(source);
  const built = buildImage('bin', null, img.bytes);
  const saved = globalThis.XenonGpuHost;
  delete globalThis.XenonGpuHost;
  try {
    await assert.rejects(() => runImage(built, {}), /GPU MMIO store.*no GPU host/);
  } finally {
    if (saved !== undefined) globalThis.XenonGpuHost = saved;
  }
});

test('end-to-end: guest rings the doorbell via MMIO, ring drains a MEM_WRITE packet into memory', async () => {
  const RING_BASE = 0x00800000;
  const OUT = 0x00100000;
  // packet: PM4_MEM_WRITE, count = 3 dwords follow, addr word endianness kNone (0)
  const header = PM4.makePacketType3(PM4.PM4_OPCODE.PM4_MEM_WRITE, 3);
  const hi = (v) => '0x' + ((v >>> 16) & 0xffff).toString(16).padStart(4, '0');
  const lo = (v) => '0x' + (v & 0xffff).toString(16).padStart(4, '0');
  const source = `
    lis  r5, 0x0080          # ring base 0x00800000
    lis  r6, ${hi(header)}
    ori  r6, r6, ${lo(header)}
    stw  r6, 0(r5)
    lis  r6, ${hi(OUT)}
    ori  r6, r6, ${lo(OUT)}  # addr word, kNone endianness
    stw  r6, 4(r5)
    lis  r6, 0xcafe
    ori  r6, r6, 0xbabe
    stw  r6, 8(r5)
    lis  r6, 0x1234
    ori  r6, r6, 0x5678
    stw  r6, 12(r5)
    li   r3, 4               # WPTR = 4 dwords submitted
    lis  r4, 0x00f0
    ori  r4, r4, 0x2714
    stw  r3, 0(r4)
` + EXIT_DOOR;
  const img = assemble(source);
  const built = buildImage('bin', null, img.bytes);
  const { wasm } = recompileImage(built.arena, built.entry);
  const imports = makeImports({});
  const { instance } = await WebAssembly.instantiate(wasm, imports);
  imports.__exports = instance.exports;
  imports.__attachMemory(instance.exports.memory);
  const memory = instance.exports.memory;
  new Uint8Array(memory.buffer).set(built.arena);
  // attach the real GPU host AFTER memory exists, BEFORE running
  const host = new XenonGpuHost({ memory, onLog: () => {} });
  globalThis.XenonGpuHost = host;
  try {
    host.initializeRingBuffer(RING_BASE, 4); // 1 << (4+3) = 128 bytes
    try {
      instance.exports['fn_' + built.entry.toString(16)]();
    } catch (e) {
      if (!(e && e.constructor && e.constructor.name === 'RecompExit')) throw e;
    }
    const view = new DataView(memory.buffer);
    assert.equal(view.getUint32(OUT, false), 0xcafebabe, 'MEM_WRITE dword 0 landed in guest memory');
    assert.equal(view.getUint32(OUT + 4, false), 0x12345678, 'MEM_WRITE dword 1 landed in guest memory');
    assert.equal(host.ringBuffer.getReadIndex(), 4, 'doorbell advanced the read pointer past 4 dwords');
    assert.equal(host.mmioLoad(RPTR, 4), 4n, 'CP_RB_RPTR reads the drained read pointer');
  } finally {
    delete globalThis.XenonGpuHost;
  }
});

test('aperture FP store routes to the GPU host too', async () => {
  const calls = [];
  const host = {
    mmioStore(addr, value, size) { calls.push({ addr, value, size }); },
    mmioLoad() { throw new Error('not used'); },
  };
  // build the double 1.0 (0x3FF0000000000000) in guest memory, load it, store it to the aperture
  const source = `
    lis  r1, 0x00e0          # scratch "stack" at 0x00E00000
    lis  r6, 0x3ff0
    stw  r6, -16(r1)
    li   r7, 0
    stw  r7, -12(r1)
    lfd  f1, -16(r1)
    lis  r4, 0x00f0
    ori  r4, r4, 0x2714
    stfd f1, 0(r4)
` + EXIT_DOOR;
  const img = assemble(source);
  const built = buildImage('bin', null, img.bytes);
  await withGpuHost(host, async () => {
    const res = await runImage(built, {});
    assert.equal(res.exitCode, 0);
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].addr, DOORBELL);
  assert.equal(calls[0].size, 8);
  assert.equal(calls[0].value, 0x3FF0000000000000n);
});
