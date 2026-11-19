// Micro-benchmarks for the GPU hot paths (ring drain, draw-path GPU ops).
// Not a unit test — prints timings; run after changes to the ring/backend:
//   node tools/bench_gpu_hot_paths.mjs
// Note: pass full-length guest views (the gpu_host default) for realistic numbers.
import '/home/user/wifi-tool/public/js/gpu/xenos_registers.js';
import '/home/user/wifi-tool/public/js/gpu/register_file.js';
import '/home/user/wifi-tool/public/js/gpu/pm4.js';
import '/home/user/wifi-tool/public/js/gpu/command_processor.js';
import '/home/user/wifi-tool/public/js/gpu/vertex_fetch.js';
import '/home/user/wifi-tool/public/js/gpu/wgsl_translate.js';
import '/home/user/wifi-tool/public/js/gpu/webgpu_backend.js';
import '/home/user/wifi-tool/public/js/gpu/ring_buffer.js';

const PM4 = globalThis.XenonPM4;
const { XenosCommandProcessor } = globalThis.XenonGpuCommandProcessor;
const { XenosWebGpuBackend } = globalThis.XenonWebGpuBackend;
const { XenosRingBuffer } = globalThis.XenonGpuRingBuffer;

function makeGuest(bytes = 96 * 1024 * 1024) {
  const arena = new Uint8Array(bytes);
  const view = new DataView(arena.buffer);
  return {
    arena,
    readGuestMemory(addr) { const a = addr >>> 0; return new DataView(arena.buffer, a, arena.length - a); },
    readGuestMemoryFull(addr) { const a = addr >>> 0; return new DataView(arena.buffer, a, arena.length - a); },
    writeGuestMemory(addr, arr) { arena.set(arr, addr); },
    writeU32BE(addr, v) { view.setUint32(addr, v >>> 0, false); },
    readU32BE(addr) { return view.getUint32(addr, false); },
  };
}

// ---- bench 1: ring drain of N dwords (type-2 NOP packets) ----
{
  const guest = makeGuest();
  const backend = new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  const cp = new XenosCommandProcessor({ backend, readGuestMemory: guest.readGuestMemory });
  const RING = 0x01000000, N = 16384; // 16K dwords of NOPs
  const words = new Array(N).fill(0x80000000); // PM4_NOP type 2 = (2<<30)
  const ring = new XenosRingBuffer({ commandProcessor: cp, readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  ring.initializeRingBuffer(RING, 15); // 1 << 18 = 256 KiB ring
  const t0 = performance.now();
  for (let i = 0; i < N; i++) guest.writeU32BE(RING + i * 4, words[i]);
  const tFill = performance.now() - t0;
  const t1 = performance.now();
  ring.updateWritePointer(N);
  const tDrain = performance.now() - t1;
  console.log(`ring drain: ${N} dwords in ${tDrain.toFixed(2)} ms (${(N / (tDrain / 1000) / 1e6).toFixed(1)} M dwords/s; CP packets=${cp.stats.packets})`);
}

// ---- bench 2: repeated fetch-draw of static geometry — GPU op counts + ms ----
globalThis.GPUBufferUsage = { VERTEX: 0x20, INDEX: 0x10, COPY_DST: 0x08 };
{
  const guest = makeGuest();
  const VB_BASE = 0x00100000, IB_BASE = 0x00200000;
  const view = new DataView(guest.arena.buffer);
  for (let i = 0; i < 9; i++) view.setFloat32(VB_BASE + i * 4, i + 1, false);
  for (let i = 0; i < 3; i++) view.setUint16(IB_BASE + i * 2, i, false);
  // microcode: exec CF (fetch sequence) -> vfetch -> alu nop  (same fixture as the unit tests)
  const execCf = (address, count, sequence, opcode = 2) => ({ dword0: (address & 0xFFF) | ((count & 0x7) << 12) | ((sequence & 0xFFF) << 16), dword1: opcode & 0xF });
  const cfPair = (a, b) => [a.dword0 >>> 0, (a.dword1 & 0xFFFF) | ((b.dword0 & 0xFFFF) << 16), ((b.dword0 >>> 16) & 0xFFFF) | ((b.dword1 & 0xFFFF) << 16)];
  const vfetch = () => {
    const w0 = ((0 & 0x3F) << 5) | (1 << 12) | (1 << 19) | (0 << 20) | (0 << 25);
    const w1 = 0x688 | (57 << 16);
    const w2 = 3;
    return [w0 >>> 0, w1 >>> 0, w2 >>> 0].concat([0, 0, 0]);
  };
  const FETCH_VS = new Uint32Array([...cfPair(execCf(1, 1, 0b01, 2), { dword0: 0, dword1: 0 }), ...vfetch()]);
  const consts = new Uint32Array(192);
  consts[0] = (VB_BASE & 0x1FFFFFFC) | 3;
  consts[1] = 9 << 2;
  const ops = { buf: 0, write: 0 };
  const fakeDevice = {
    queue: { writeBuffer() { ops.write++; }, submit() {} },
    createBuffer(desc) { ops.buf++; return { desc }; },
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
  backend.device = fakeDevice;
  backend.format = 'bgra8unorm';
  const canvas = { width: 64, height: 64, getContext: () => ({ getCurrentTexture: () => ({ createView: () => ({}) }) }) };
  backend.canvas = canvas;
  backend.context = canvas.getContext('webgpu');
  backend.shaders.vertex = { kind: 'vertex', code: FETCH_VS, dwords: FETCH_VS.length };
  const params = { primType: 4, primTypeName: 'triangle-list', numIndices: 3, isIndexed: true, indexGuestBase: IB_BASE, indexFormat: 0, indexEndianness: 1, indexLengthWords: 2, vertexFetchConstants: consts };
  const REPS = 2000;
  const t0 = performance.now();
  for (let i = 0; i < REPS; i++) backend.draw({ ...params });
  const ms = performance.now() - t0;
  console.log(`fetch draws: ${REPS} in ${ms.toFixed(1)} ms (${(ms / REPS * 1000).toFixed(1)} µs/draw); GPU buffers created=${ops.buf}, writeBuffer=${ops.write}`);
}

// ---- bench 3: repeated draws on the translated path — translation/pipeline memoized, constants hashed
globalThis.GPUBufferUsage = { VERTEX: 0x20, INDEX: 0x10, COPY_DST: 0x08, UNIFORM: 0x40 };
{
  const guest = makeGuest();
  const VB_BASE = 0x00100000;
  const view = new DataView(guest.arena.buffer);
  for (let i = 0; i < 9; i++) view.setFloat32(VB_BASE + i * 4, i + 1, false);
  // ucode: exec CF (fetch, alu, alu) -> vfetch -> mad r3 = r1*fc0 + fc1 -> export oPos = r3 + r1
  const execCf = (address, count, sequence, opcode = 2) => ({ dword0: (address & 0xFFF) | ((count & 0x7) << 12) | ((sequence & 0xFFF) << 16), dword1: (opcode & 0xF) << 12 });
  const NOP_CF = { dword0: 0, dword1: 0 };
  const cfPair = (a, b) => [a.dword0 >>> 0, (a.dword1 & 0xFFFF) | ((b.dword0 & 0xFFFF) << 16), ((b.dword0 >>> 16) & 0xFFFF) | ((b.dword1 & 0xFFFF) << 16)];
  const temp = (reg) => ({ sel: 1, reg: reg & 0x3F });
  const konst = (index) => ({ sel: 0, reg: index & 0xFF });
  const aluWords = ({ vectorDest = 0, isExport = 0, vectorWriteMask = 0xF, vectorOpcode = 2, src = [temp(0), temp(1), temp(1)] } = {}) => {
    const w0 = ((vectorDest & 0x3F) | ((isExport & 1) << 15) | ((vectorWriteMask & 0xF) << 16) | (50 << 26)) >>> 0;
    const w2 = (((src[2].reg & 0xFF) | ((src[1].reg & 0xFF) << 8) | ((src[0].reg & 0xFF) << 16)) | ((vectorOpcode & 0x1F) << 24) | ((src[2].sel & 1) << 29) | ((src[1].sel & 1) << 30) | ((src[0].sel & 1) << 31)) >>> 0;
    return [w0, 0, w2];
  };
  const vfetchW = () => [((1 & 0x3F) << 12) | (1 << 19), 0x688 | (57 << 16), 3];
  const CONST_VS = new Uint32Array([
    ...cfPair(execCf(1, 3, 1 /* seq(FETCH, ALU, ALU) */, 2), NOP_CF),
    ...vfetchW(),
    ...aluWords({ vectorDest: 3, vectorOpcode: 11 /* mad */, src: [temp(1), konst(0), konst(1)] }),
    ...aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 0 /* add */, src: [temp(3), temp(1), temp(1)] }),
  ]);
  const consts = new Uint32Array(192);
  consts[0] = (VB_BASE & 0x1FFFFFFC) | 3;
  consts[1] = 9 << 2;
  const alu = new Uint32Array(1024);
  alu[0] = 0x3f800000;
  alu[4] = 0x40000000;
  const ops = { buf: 0, write: 0, pipeline: 0, module: 0, bindGroup: 0 };
  const fakeDevice = {
    queue: { writeBuffer() { ops.write++; }, submit() {} },
    createBuffer(desc) { ops.buf++; return { desc }; },
    createTexture() { return { destroy() {} }; },
    createShaderModule(desc) { ops.module++; return { desc }; },
    createRenderPipeline(desc) { ops.pipeline++; return { desc, getBindGroupLayout() { return {}; } }; },
    createBindGroup(desc) { ops.bindGroup++; return { desc }; },
    createCommandEncoder() {
      const pass = { setPipeline() {}, setVertexBuffer() {}, setBindGroup() {}, setIndexBuffer() {}, draw() {}, drawIndexed() {}, end() {} };
      return { beginRenderPass() { return pass; }, finish() { return {}; } };
    },
    destroy() {},
  };
  const backend = new XenosWebGpuBackend({ readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
  backend.device = fakeDevice;
  backend.format = 'bgra8unorm';
  const canvas = { width: 64, height: 64, getContext: () => ({ getCurrentTexture: () => ({ createView: () => ({}) }) }) };
  backend.canvas = canvas;
  backend.context = canvas.getContext('webgpu');
  backend.shaders.vertex = { kind: 'vertex', code: CONST_VS, dwords: CONST_VS.length };
  const params = { primType: 4, primTypeName: 'triangle-list', numIndices: 3, isIndexed: false, vertexFetchConstants: consts, aluConstants: alu };
  const REPS = 2000;
  const t0 = performance.now();
  for (let i = 0; i < REPS; i++) backend.draw({ ...params });
  const ms = performance.now() - t0;
  console.log(`translated draws: ${REPS} in ${ms.toFixed(1)} ms (${(ms / REPS * 1000).toFixed(1)} µs/draw); pipelines=${ops.pipeline}, modules=${ops.module}, shader modules created once, buffers created=${ops.buf}, writeBuffer=${ops.write}, bindGroups=${ops.bindGroup}`);
}
