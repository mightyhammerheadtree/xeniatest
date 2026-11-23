/**
 * Xenos GPU PM4 command processor tests — drive the JS command processor with
 * hand-assembled big-endian command buffers and a real (in-memory) guest
 * physical arena, and assert both register-file state and backend-visible
 * side effects, exactly as the browser would observe them.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../../public/js/gpu/xenos_registers.js';
import '../../public/js/gpu/register_file.js';
import '../../public/js/gpu/pm4.js';
import '../../public/js/gpu/command_processor.js';
import '../../public/js/gpu/render_state.js';

const PM4 = globalThis.XenonPM4;
const { XenosCommandProcessor, XenosGpuError } = globalThis.XenonGpuCommandProcessor;
const { XenosRegisterFile } = globalThis.XenonGpuRegisterFile;
const REGISTERS = globalThis.XenonGpuRegisters;

const GUEST_ADDR = 0x01000000;

function makeGuest(size = 0x02000000) {
  const arena = new Uint8Array(size);
  return {
    arena,
    readGuestMemory(addr) { return new DataView(arena.buffer, addr, arena.length - addr); },
    writeGuestMemory(addr, bytes) { arena.set(bytes, addr); },
    writeU32BE(addr, value) { new DataView(arena.buffer).setUint32(addr, value >>> 0, false); },
    readU32BE(addr) { return new DataView(arena.buffer).getUint32(addr, false) >>> 0; },
  };
}

function makeBackend(guest) {
  const calls = [];
  return {
    calls,
    memWrite(addr, bytes) { calls.push({ kind: 'memWrite', addr, bytes: Uint8Array.from(bytes) }); guest.writeGuestMemory(addr, bytes); },
    draw(params) { calls.push({ kind: 'draw', ...params }); },
    eventWrite(evt) { calls.push({ kind: 'eventWrite', ...evt }); },
    interrupt(mask) { calls.push({ kind: 'interrupt', mask }); },
    swap() { calls.push({ kind: 'swap' }); },
  };
}

function makeCP(guest, backend) {
  return new XenosCommandProcessor({ backend, readGuestMemory: guest.readGuestMemory, writeGuestMemory: guest.writeGuestMemory });
}

/**
 * Put the command processor into xenia's ordinary color+depth draw mode.
 * xenia's IssueDraw ignores a draw while RB_MODECONTROL.edram_mode is
 * kIgnore (0 — the reset value) and while RB_SURFACE_INFO.surface_pitch is 0
 * ("doesn't actually draw"), so a guest that wants its draws rasterized
 * programs both registers.
 */
function enableDraws(cp) {
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2208, 1), 0x00000004])); // RB_MODECONTROL = kColorDepth
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2000, 1), 0x00001000])); // RB_SURFACE_INFO.surface_pitch = 0x1000
  return cp;
}

/** Encode padded list of dwords as BE bytes. */
function beWords(words) {
  const buf = new Uint8Array(words.length * 4);
  const view = new DataView(buf.buffer);
  words.forEach((w, i) => view.setUint32(i * 4, w >>> 0, false));
  return buf;
}

test('register table: 3435 registers, known names at known indices, reset defaults applied', () => {
  assert.equal(REGISTERS.XENOS_GPU_REGISTERS.length, 3435);
  assert.equal(REGISTERS.XENOS_GPU_REGISTER_BY_INDEX.get(0x21fc).name, 'VGT_DRAW_INITIATOR');
  assert.equal(REGISTERS.XENOS_GPU_REGISTER_BY_INDEX.get(0x4000).name, 'SHADER_CONSTANT_000_X');
  assert.equal(REGISTERS.XENOS_GPU_REGISTER_BY_NAME.get('RB_COLOR_MASK').index, 0x2104);
  const rf = new XenosRegisterFile();
  assert.equal(rf.read(0x2100), 0x0000ffff, 'VGT_MAX_VTX_INDX reset default');
  assert.equal(rf.read(0x200f), 0x20002000, 'PA_SC_SCREEN_SCISSOR_BR reset default');
  assert.equal(rf.read(0x4000), 0, 'non-default register starts at 0');
});

test('type-0 packet writes a register range big-endian', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2104, 3), 0xaaaa0001, 0xaaaa0002, 0xaaaa0003]));
  assert.equal(cp.registers.read(0x2104), 0xaaaa0001);
  assert.equal(cp.registers.read(0x2105), 0xaaaa0002);
  assert.equal(cp.registers.read(0x2106), 0xaaaa0003);
  assert.equal(cp.stats.registerWrites, 3);
});

test('type-0 one_reg mode writes repeated values from count words', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2104, 2, /* oneReg */ true), 0x11111111, 0x22222222]));
  assert.equal(cp.registers.read(0x2104), 0x22222222, 'last write wins, both words consumed');
  assert.equal(cp.stats.registerWrites, 2);
});

test('type-2 packet is a no-op', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  cp.executeCommandBuffer(beWords([PM4.makePacketType2()]));
  assert.equal(cp.stats.packets, 1);
  assert.equal(cp.stats.registerWrites, 0);
});

test('type-1 packet writes two low registers', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  cp.executeCommandBuffer(beWords([PM4.makePacketType1(0x0040, 0x0041), 0x12345678, 0x9abcdef0]));
  assert.equal(cp.registers.read(0x0040), 0x12345678);
  assert.equal(cp.registers.read(0x0041), 0x9abcdef0);
});

test('SET_CONSTANT (type ALU) lands in the ALU constant window', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  const offsetType = (0 << 16) | 5; // type 0 (ALU), index 5
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_CONSTANT, 3), offsetType, 0x3f800000, 0x40000000]));
  assert.equal(cp.registers.read(0x4000 + 5), 0x3f800000);
  assert.equal(cp.registers.read(0x4000 + 6), 0x40000000);
  assert.equal(cp.stats.constantBytes, 8);
});

test('SET_CONSTANT (type FETCH/BOOL/LOOP) lands in their windows; unknown type traps', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_CONSTANT, 2), (1 << 16) | 2, 0x11223344]));
  assert.equal(cp.registers.read(0x4800 + 2), 0x11223344);
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_CONSTANT, 2), (2 << 16) | 3, 0x00000001]));
  assert.equal(cp.registers.read(0x4900 + 3), 1);
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_CONSTANT, 2), (3 << 16) | 1, 0x00000002]));
  assert.equal(cp.registers.read(0x4908 + 1), 2);
  assert.throws(() => cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_CONSTANT, 2), (7 << 16) | 0, 0xff])), XenosGpuError);
});

test('SET_CONSTANT2 writes a direct register range', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_CONSTANT2, 3), 0x2104, 0x0badf00d & 0, 0xfeedface]));
  assert.equal(cp.registers.read(0x2104), 0);
  assert.equal(cp.registers.read(0x2105), 0xfeedface);
});

test('LOAD_ALU_CONSTANT pulls constants from guest physical memory', () => {
  const guest = makeGuest();
  guest.writeU32BE(GUEST_ADDR, 0xdeadbeef);
  guest.writeU32BE(GUEST_ADDR + 4, 0x0badf00d);
  const cp = makeCP(guest, makeBackend(guest));
  const offsetType = (0 << 16) | 9; // ALU, index 9
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_LOAD_ALU_CONSTANT, 3), GUEST_ADDR, offsetType, 2 /* size_dwords */]));
  assert.equal(cp.registers.read(0x4000 + 9), 0xdeadbeef);
  assert.equal(cp.registers.read(0x4000 + 10), 0x0badf00d);
});

test('MEM_WRITE stores big-endian words through the backend into guest memory', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  const addr = GUEST_ADDR; // endianness kNone (lowest 2 bits = 0)
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_MEM_WRITE, 3), addr, 0xcafebabe, 0x12345678]));
  assert.equal(guest.readU32BE(addr), 0xcafebabe);
  assert.equal(guest.readU32BE(addr + 4), 0x12345678);
  assert.equal(backend.calls.filter((c) => c.kind === 'memWrite').length, 1);
});

test('MEM_WRITE with k8in32 endianness swaps bytes', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  const addr = GUEST_ADDR | 2; // k8in32
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_MEM_WRITE, 2), addr, 0x11223344]));
  assert.equal(guest.readU32BE(GUEST_ADDR), 0x44332211, 'bytes swapped per k8in32');
});

test('REG_TO_MEM copies a register to guest memory with endianness', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2104, 1), 0xaabbccdd]));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_REG_TO_MEM, 2), 0x2104, GUEST_ADDR | 2 /* k8in32 */]));
  assert.equal(guest.readU32BE(GUEST_ADDR), 0xddccbbaa);
});

test('REG_RMW supports immediate and register operands', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  // REG_RMW masks its register target at 13 bits (xenia); use the low config window
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x0040, 2), 0xffff00ff, 0x0000ffff]));
  // value = (0xffff00ff & 0x0f0f0f0f) | 0x00000f0f = 0x0f0f0f0f
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_REG_RMW, 3), 0x0040, 0x0f0f0f0f, 0x00000f0f]));
  assert.equal(cp.registers.read(0x0040), 0x0f0f0f0f);
  // register OR operand: or_mask from register 0x0041 (0x0000ffff)
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_REG_RMW, 3), 0x40000000 | 0x0040, 0xffffffff, 0x0041]));
  assert.equal(cp.registers.read(0x0040), 0x0f0fffff);
});

test('EVENT_WRITE_SHD writes the payload; the counter bit writes the GPU frame counter', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  const addr = GUEST_ADDR | 0; // kNone
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_EVENT_WRITE_SHD, 3), 0x00000005 /* initiator: CONTEXT_DONE */, addr, 0xcafe1234]));
  assert.equal(guest.readU32BE(GUEST_ADDR), 0xcafe1234);
  assert.equal(cp.registers.read(0x21f9), 0x00000005, 'VGT_EVENT_INITIATOR written');
  // counter bit: initiator with 0x80000000 writes the GPU counter instead of the payload
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_EVENT_WRITE_SHD, 2), 0x80000000 | 0x05, GUEST_ADDR]));
  assert.equal(guest.readU32BE(GUEST_ADDR), 0, 'the frame counter starts at 0');
});

test('the GPU frame counter is bumped by the swap (xenia: ++counter_ in PM4_XE_SWAP)', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  // xenia's swap packet starts with the 'SWAP' magic.
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_XE_SWAP, 4), 0x53574150, 0, 0, 0]));
  assert.equal(backend.calls.filter((c) => c.kind === 'swap').length, 1);
  // The counter bit of EVENT_WRITE_SHD now writes 1.
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_EVENT_WRITE_SHD, 2), 0x80000000 | 0x05, GUEST_ADDR]));
  assert.equal(guest.readU32BE(GUEST_ADDR), 1, 'the counter was bumped by the swap');
  // A second swap bumps it again.
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_XE_SWAP, 4), 0x53574150, 0, 0, 0]));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_EVENT_WRITE_SHD, 2), 0x80000000 | 0x05, GUEST_ADDR]));
  assert.equal(guest.readU32BE(GUEST_ADDR), 2);
});

test('WAIT_REG_MEM evaluates xenia\'s comparison and refuses loudly when it would have to block', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  // Register case: wait until register 0x2104 (RB_COLOR_MASK) == 0x1234.
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2104, 1), 0x00001234]));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_WAIT_REG_MEM, 5),
    0x00000003 /* compare: equal */, 0x00002104, 0x00001234, 0xffffffff, 0x00000000]));
  // The comparison is not met → xenia would block; this port refuses loudly.
  assert.throws(
    () => cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_WAIT_REG_MEM, 5),
      0x00000003, 0x00002104, 0x00005678, 0xffffffff, 0x00000000])),
    /wait condition is not met/);
  // ... and a satisfied condition passes.
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_WAIT_REG_MEM, 5),
    0x00000003, 0x00002104, 0x00001234, 0xffffffff, 0x00000000]));
  // The memory case reads guest memory with the address's endianness.
  guest.writeU32BE(GUEST_ADDR, 0xdeadbeef);
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_WAIT_REG_MEM, 5),
    0x00000010 | 0x00000003, GUEST_ADDR, 0xdeadbeef, 0xffffffff, 0x00000000]));
});

test('COND_WRITE performs xenia\'s conditional write to a register', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2104, 1), 0x00001234]));
  // wait_info = 0x03 (equal), poll register 0x2104, reference 0x1234, mask all.
  const packet = (ref, dest, data) => beWords([
    PM4.makePacketType3(PM4.PM4_OPCODE.PM4_COND_WRITE, 6),
    0x00000003, 0x00002104, ref >>> 0, 0xffffffff, dest >>> 0, data >>> 0,
  ]);
  // The condition holds → the register is written.
  cp.executeCommandBuffer(packet(0x00001234, 0x00002110, 0xdeadbeef));
  assert.equal(cp.registers.read(0x2110), 0xdeadbeef);
  // The condition does not hold → nothing is written (xenia's "if (matched)").
  cp.executeCommandBuffer(packet(0x00005678, 0x00002110, 0xfeedface));
  assert.equal(cp.registers.read(0x2110), 0xdeadbeef, 'the register kept its value');
});

test('COND_WRITE performs xenia\'s conditional write to memory', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  guest.writeU32BE(GUEST_ADDR, 0x00001234);
  // Poll memory (wait_info bit 4), write memory (wait_info bit 8).
  // The *poll* address carries the endianness of the polled value (kNone
  // here: the plain dword), the *write* address the endianness of the
  // stored value (k8in32 = 2: the bytes are swapped).
  cp.executeCommandBuffer(beWords([
    PM4.makePacketType3(PM4.PM4_OPCODE.PM4_COND_WRITE, 6),
    0x00000010 | 0x00000100 | 0x00000003, GUEST_ADDR, 0x1234, 0xffffffff, GUEST_ADDR | 2, 0xdeadbeef,
  ]));
  const write = backend.calls.find((c) => c.kind === 'memWrite');
  assert.ok(write, 'the memory was written');
  assert.equal(write.addr, GUEST_ADDR);
  // xenia: GpuSwap(write_data, k8in32) then a plain store → byte-swapped.
  assert.equal(new DataView(write.bytes.buffer).getUint32(0, false), 0xefbeadde);
});

test('COND_EXEC still traps loudly (xenia has no case for it either — its default asserts)', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  assert.throws(
    () => cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_COND_EXEC, 6), 0, 0, 0, 0, 0, 0])),
    /conditional execution is not implemented/);
});

test('EVENT_WRITE_EXT writes xenia\'s screen extent (six 16-bit values) to guest memory', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  // The address carries the k8in16 endianness (xenia asserts it).
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_EVENT_WRITE_EXT, 2), 0x00000005, GUEST_ADDR | 1]));
  assert.equal(cp.registers.read(0x21f9), 0x00000005, 'VGT_EVENT_INITIATOR written');
  const write = backend.calls.find((c) => c.kind === 'memWrite');
  assert.ok(write, 'the screen extent was written to guest memory');
  assert.equal(write.addr, GUEST_ADDR);
  // xenia: {0>>3, 2560>>3, 0>>3, 2560>>3, 0, 1} with the bytes of each half
  // word swapped (k8in16).
  const expected = new Uint8Array(12);
  [0, 320, 0, 320, 0, 1].forEach((v, i) => {
    expected[i * 2] = (v >>> 8) & 0xff;
    expected[i * 2 + 1] = v & 0xff;
  });
  assert.deepEqual(Array.from(write.bytes), Array.from(expected));
});

test('EVENT_WRITE maps the writeback window through WRITEBACK_START/SIZE', () => {
  const guest = makeGuest();
  // Recording backend: the mapped window address (0x7F000000+) is not inside our flat arena; capture it instead of writing
  const writes = [];
  const backend = { memWrite(addr, bytes) { writes.push({ addr, bytes: Uint8Array.from(bytes) }); } };
  const cp = new XenosCommandProcessor({ backend, readGuestMemory: guest.readGuestMemory });
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x0a04, 2), 0x20000000, 0x00001000])); // WRITEBACK_START=0x20000000, SIZE=0x1000
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_EVENT_WRITE_SHD, 3), 0x00000005, 0x20000040, 0xdeadbeef]));
  assert.equal(writes.length, 1);
  assert.equal(writes[0].addr, 0x7f000040, 'writeback window maps to 0x7F000000 + offset');
  assert.equal(new DataView(writes[0].bytes.buffer).getUint32(0, false), 0xdeadbeef);
});

test('DRAW_INDX (indexed, DMA) decodes the initiator and notifies the backend', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = enableDraws(makeCP(guest, backend));
  const numIndices = 0x00f0;
  const primType = PM4.PRIMITIVE_TYPE.kTriangleList;
  const sourceSelect = PM4.SOURCE_SELECT.kDMA;
  const initiator = ((numIndices & 0xffff) << 16) | (0 << 12) | (0 << 11) | (sourceSelect << 6) | primType;
  const dmaBase = 0x00c00000; // GPU bus (29 bits)
  const dmaSize = (6 & 0x00ffffff) | (2 << 30); // 6 words, k8in32
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX, 4), 0x00000000 /* viz */, initiator, dmaBase, dmaSize]));
  const draws = backend.calls.filter((c) => c.kind === 'draw');
  assert.equal(draws.length, 1);
  assert.deepEqual(
    { primType: draws[0].primType, numIndices: draws[0].numIndices, isIndexed: draws[0].isIndexed, indexGuestBase: draws[0].indexGuestBase, indexFormat: draws[0].indexFormat, indexEndianness: draws[0].indexEndianness, indexLengthWords: draws[0].indexLengthWords, majorModeExplicit: draws[0].majorModeExplicit },
    { primType: 4, numIndices: 0x00f0, isIndexed: true, indexGuestBase: 0x00c00000, indexFormat: 0, indexEndianness: 2, indexLengthWords: 6, majorModeExplicit: false },
  );
  assert.equal(cp.registers.read(0x21fc), initiator >>> 0, 'VGT_DRAW_INITIATOR written');
  assert.equal(cp.registers.read(0x21fa), 0x00c00000, 'VGT_DMA_BASE written');
  assert.equal(cp.stats.draws, 1);
});

test('DRAW_INDX_2 (auto-index) draws without an index DMA', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = enableDraws(makeCP(guest, backend));
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleStrip;
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator]));
  const draws = backend.calls.filter((c) => c.kind === 'draw');
  assert.equal(draws.length, 1);
  assert.equal(draws[0].isIndexed, false);
  assert.equal(draws[0].numIndices, 3);
});

test('predicated packets are skipped while the bin predicate fails and execute when it passes', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = enableDraws(makeCP(guest, backend));
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
  // mask = 0 → predicate fails
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_BIN_MASK, 2), 0x00000000, 0x00000000]));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_BIN_SELECT, 2), 0x00000000, 0x00000000]));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1, /* predicate */ true), initiator]));
  assert.equal(backend.calls.filter((c) => c.kind === 'draw').length, 0, 'predicated draw skipped');
  // mask = 1, select = 1 → predicate passes
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_BIN_MASK, 2), 0x00000000, 0x00000001]));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_BIN_SELECT, 2), 0x00000000, 0x00000001]));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1, /* predicate */ true), initiator]));
  assert.equal(backend.calls.filter((c) => c.kind === 'draw').length, 1, 'predicated draw executed');
});

test('a predicated swap is skipped even when the bin predicate passes (xenia: predicated swaps are never valid)', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  // mask = 1, select = 1 → the predicate passes
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_BIN_MASK, 2), 0x00000000, 0x00000001]));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_BIN_SELECT, 2), 0x00000000, 0x00000001]));
  const swapsBefore = backend.calls.filter((c) => c.kind === 'swap').length;
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_XE_SWAP, 1, /* predicate */ true), 0x00000000]));
  assert.equal(backend.calls.filter((c) => c.kind === 'swap').length, swapsBefore, 'the predicated swap was skipped');
});

test('PM4_CONTEXT_UPDATE with a 0 payload is a no-op (xenia); a non-zero payload traps', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_CONTEXT_UPDATE, 1), 0x00000000]));
  assert.throws(
    () => cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_CONTEXT_UPDATE, 1), 0x00000001])),
    /CONTEXT_UPDATE/);
});

test('IM_LOAD_IMMEDIATE loads a shader and reports it to the backend', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const shaders = [];
  const cp = new XenosCommandProcessor({ backend: { ...backend, setShaders: ({ vertex, pixel }) => shaders.push({ vertex, pixel }) }, readGuestMemory: guest.readGuestMemory });
  // ShaderType kVertex = 0; size_dwords = 2
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_IM_LOAD_IMMEDIATE, 4), 0x00000000 /* shader type vertex */, 0x00000002 /* start=0 size=2 */, 0xdeadbeef, 0x0badf00d]));
  assert.equal(cp.vertexShader.dwords, 2);
  assert.deepEqual([...cp.vertexShader.code], [0xdeadbeef, 0x0badf00d]);
  assert.equal(shaders.length, 1);
});

test('INDIRECT_BUFFER recurses into a guest-memory command buffer', () => {
  const guest = makeGuest();
  guest.writeU32BE(GUEST_ADDR, PM4.makePacketType0(0x2104, 1));
  guest.writeU32BE(GUEST_ADDR + 4, 0x5eed5eed);
  const cp = makeCP(guest, makeBackend(guest));
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_INDIRECT_BUFFER, 2), GUEST_ADDR, 2 /* dwords */]));
  assert.equal(cp.registers.read(0x2104), 0x5eed5eed);
  assert.equal(cp.stats.indirectBuffers, 1);
});

test('unimplemented opcodes trap loudly with the opcode number', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  assert.throws(() => cp.executeCommandBuffer(beWords([PM4.makePacketType3(0x7f, 1), 0])), (e) => e instanceof XenosGpuError && /unimplemented GPU opcode 0x7F/.test(e.message));
  assert.equal(cp.stats.unimplementedTraps, 1);
});

test('truncated packets throw instead of reading out of bounds', () => {
  const guest = makeGuest();
  const cp = makeCP(guest, makeBackend(guest));
  assert.throws(() => cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2104, 2), 0x11111111])), /truncated/);
  assert.throws(() => cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_CONSTANT, 3), 0, 0])), /truncated/);
});

test('PM4_XE_SWAP calls backend.swap (frame boundary)', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_XE_SWAP, 1), 0x00010000]));
  assert.equal(backend.calls.filter((c) => c.kind === 'swap').length, 1);
});

test('draw without a backend trap: clear error naming the missing method', () => {
  const guest = makeGuest();
  const cp = enableDraws(new XenosCommandProcessor({ readGuestMemory: guest.readGuestMemory })); // no backend
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
  assert.throws(() => cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator])), (e) => /backend does not implement draw\(\)/.test(e.message));
});

test('draw packet snapshots ALU and bool constants from the register file', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = enableDraws(makeCP(guest, backend));
  // SET_CONSTANT ALU index 0: two dwords (1.0f, 2.0f bits) — packet has 3 payload dwords (index + 2)
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_CONSTANT, 3), 0, 0x3f800000, 0x40000000]));
  // SET_CONSTANT BOOL index 4: value with bit for bool 4
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_SET_CONSTANT, 2), (2 << 16) | 4, 1 << 4]));
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator]));
  const draw = backend.calls.find((c) => c.kind === 'draw');
  assert.ok(draw, 'draw packet reached the backend');
  assert.equal(draw.aluConstants.length, 1024);
  assert.equal(draw.aluConstants[0], 0x3f800000);
  assert.equal(draw.aluConstants[1], 0x40000000);
  assert.equal(draw.boolConstants.length, 8);
  assert.equal(draw.boolConstants[4], 1 << 4);
  // Fetch constants remain attached too (regression: same snapshot path)
  assert.equal(draw.vertexFetchConstants.length, 192);
});

test('draw packet snapshots the RB_* render-target registers from the register file', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = enableDraws(makeCP(guest, backend));
  // Write the render-target registers via direct context-register writes (index base 0x2000).
  // RB_COLOR_INFO (0x2001): format 5 (k_16_16_16_16) at +16, base 0x42.
  // RB_COLOR_MASK (0x2104): 0xA5A5A5A5.
  // RB_BLEND_RED/GREEN/BLUE/ALPHA (0x2105..0x2108): 0.25f / 0.5f / 0.75f / 1.0f bits.
  // RB_DEPTHCONTROL (0x2200): z_enable|z_write|zfunc GreaterEqual(6)<<4|stencil off.
  // RB_COLORCONTROL (0x2202): 0 (no alpha test).
  // RB_BLENDCONTROL0 (0x2201): src=src-alpha(6), one-minus-src-alpha(7) — 0x07000506.
  // PA_SC_WINDOW_SCISSOR_TL/BR (0x2081/0x2082): (8,8)-(40,40); WINDOW_OFFSET (0x2080): (+2,+3);
  // PA_SC_SCREEN_SCISSOR_TL/BR (0x200e/0x200f): (4,4)-(52,52).
  const setReg = (index, value) => cp.executeCommandBuffer(beWords([PM4.makePacketType0(index, 1), value]));
  setReg(0x2080, (3 << 16) | 2);
  setReg(0x2081, (8 << 16) | 8);
  setReg(0x2082, (40 << 16) | 40);
  setReg(0x200e, (4 << 16) | 4);
  setReg(0x200f, (52 << 16) | 52);
  setReg(0x2001, (5 << 16) | 0x42);
  setReg(0x2002, 0); // kD24S8
  setReg(0x2104, 0xA5A5A5A5);
  setReg(0x2105, 0x3E800000);
  setReg(0x2106, 0x3F000000);
  setReg(0x2107, 0x3F400000);
  setReg(0x2108, 0x3F800000);
  setReg(0x210d, 0x0000427F); // RB_STENCILREFMASK: ref 0x7F, mask 0x42
  setReg(0x2200, 0x2 | 0x4 | (0x6 << 4));
  setReg(0x2201, (7 << 8) | (0 << 5) | 6);
  setReg(0x210e, 0x3F000000); // RB_ALPHA_REF: 0.5f
  setReg(0x2000, (1 << 16) | 0x0123); // RB_SURFACE_INFO: 2x MSAA, surface_pitch 0x0123 (snapshot is raw; the backend traps on MSAA)
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator]));
  const draw = backend.calls.find((c) => c.kind === 'draw');
  assert.ok(draw, 'draw packet reached the backend');
  assert.ok(draw.renderTargets, 'render-target snapshot is attached');
  assert.equal(draw.renderTargets.colorInfo0, (5 << 16) | 0x42);
  assert.equal(draw.renderTargets.depthInfo, 0);
  assert.equal(draw.renderTargets.colorMask, 0xA5A5A5A5);
  assert.equal(draw.renderTargets.blendRed, 0x3E800000);
  assert.equal(draw.renderTargets.blendAlpha, 0x3F800000);
  assert.equal(draw.renderTargets.stencilRefMask, 0x0000427F);
  assert.equal(draw.renderTargets.stencilRefMaskBf, 0x00FFFF00, 'hardware reset default from the generated register table');
  assert.equal(draw.renderTargets.depthControl, 0x2 | 0x4 | (0x6 << 4));
  assert.equal(draw.renderTargets.blendControl0, (7 << 8) | 6);
  assert.equal(draw.renderTargets.colorControl, 0);
  assert.equal(draw.renderTargets.alphaRef, 0x3F000000);
  assert.equal(draw.renderTargets.surfaceInfo, (1 << 16) | 0x0123);
  assert.ok(draw.scissor, 'scissor snapshot is attached');
  assert.equal(draw.scissor.windowScissorTl, (8 << 16) | 8);
  assert.equal(draw.scissor.windowScissorBr, (40 << 16) | 40);
  assert.equal(draw.scissor.windowOffset, (3 << 16) | 2);
  assert.equal(draw.scissor.screenScissorTl, (4 << 16) | 4);
  assert.equal(draw.scissor.screenScissorBr, (52 << 16) | 52);
});

test('draw with PA_SC_VIZ_QUERY ena+kill (0x81) is dropped — occlusion hint, not an RT write (xenia rule)', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  const setReg = (index, value) => cp.executeCommandBuffer(beWords([PM4.makePacketType0(index, 1), value]));
  setReg(0x2293, 0x81); // viz_query_ena | kill_pix_post_hi_z
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator]));
  assert.equal(backend.calls.filter((c) => c.kind === 'draw').length, 0, 'no draw issued — xenia skips it');
});

test('PA_SC_VIZ_QUERY ena without kill (or kill without ena) still draws (xenia rule)', () => {
  for (const viz of [0x01, 0x80, 0x00]) {
    const guest = makeGuest();
    const backend = makeBackend(guest);
    const cp = enableDraws(makeCP(guest, backend));
    const setReg = (index, value) => cp.executeCommandBuffer(beWords([PM4.makePacketType0(index, 1), value]));
    setReg(0x2293, viz);
    const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
    cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator]));
    assert.equal(backend.calls.filter((c) => c.kind === 'draw').length, 1, `PA_SC_VIZ_QUERY=0x${viz.toString(16)} must still draw`);
  }
});

test('RB_MODECONTROL gates the draw exactly like xenia (kIgnore / kCopy / kDepth)', () => {
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
  // kIgnore (0, the reset value) → the draw is ignored (no fragments).
  {
    const guest = makeGuest();
    const backend = makeBackend(guest);
    const cp = enableDraws(makeCP(guest, backend));
    cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2208, 1), 0x00000000]));
    cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator]));
    assert.equal(backend.calls.filter((c) => c.kind === 'draw').length, 0, 'a kIgnore draw never reaches the backend');
  }
  // kCopy (6) → the guest wants a render-target copy: this port cannot do it.
  {
    const guest = makeGuest();
    const backend = makeBackend(guest);
    const cp = enableDraws(makeCP(guest, backend));
    cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2208, 1), 0x00000006]));
    assert.throws(
      () => cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator])),
      /render-target copy/);
  }
  // kDepth (5) → depth only: the draw reaches the backend flagged as such.
  {
    const guest = makeGuest();
    const backend = makeBackend(guest);
    const cp = enableDraws(makeCP(guest, backend));
    cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2208, 1), 0x00000005]));
    cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator]));
    const draw = backend.calls.find((c) => c.kind === 'draw');
    assert.ok(draw, 'the depth-only draw reached the backend');
    assert.equal(draw.depthOnly, true);
  }
  // kColorDepth (4) → an ordinary draw.
  {
    const guest = makeGuest();
    const backend = makeBackend(guest);
    const cp = enableDraws(makeCP(guest, backend));
    cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator]));
    const draw = backend.calls.find((c) => c.kind === 'draw');
    assert.ok(draw);
    assert.equal(draw.depthOnly, false);
  }
});

test('a draw with RB_SURFACE_INFO.surface_pitch == 0 does not draw (xenia: "doesn\'t actually draw")', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = enableDraws(makeCP(guest, backend));
  cp.executeCommandBuffer(beWords([PM4.makePacketType0(0x2000, 1), 0x00000000])); // surface_pitch = 0
  const initiator = ((3 & 0xffff) << 16) | (PM4.SOURCE_SELECT.kAutoIndex << 6) | PM4.PRIMITIVE_TYPE.kTriangleList;
  cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_DRAW_INDX_2, 1), initiator]));
  assert.equal(backend.calls.filter((c) => c.kind === 'draw').length, 0, 'the draw never reaches the backend');
});

test('PM4_VIZ_QUERY traps loudly (occlusion queries unimplemented — step 9)', () => {
  const guest = makeGuest();
  const backend = makeBackend(guest);
  const cp = makeCP(guest, backend);
  assert.throws(
    () => cp.executeCommandBuffer(beWords([PM4.makePacketType3(PM4.PM4_OPCODE.PM4_VIZ_QUERY, 1), 0])),
    (e) => e instanceof XenosGpuError && /VIZ_QUERY/.test(e.message));
});
