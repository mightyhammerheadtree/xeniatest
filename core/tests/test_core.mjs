/**
 * xeniatest core (WASM) test suite — runs with: npm run test:core
 *
 * Tests decode coverage against the xenia-canary opcode table, the
 * disassembler, and the interpreter by executing real (hand-assembled)
 * big-endian Xenon PPC binaries in the sandbox and checking architectural
 * state afterwards.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const WASM_PATH = join(ROOT, 'public', 'core', 'xeniatest-core.wasm');

export async function loadCore() {
  const bytes = readFileSync(WASM_PATH);
  const imports = {
    wasi_snapshot_preview1: {
      args_get: () => 0,
      args_sizes_get: () => 0,
      proc_exit: (code) => {
        throw new Error(`guest called proc_exit(${code})`);
      },
    },
  };
  const { instance } = await WebAssembly.instantiate(bytes, imports);
  return instance.exports;
}

class Core {
  constructor(exports) {
    this.x = exports;
    this.mem = new DataView(exports.memory.buffer);
    this.ramBase = exports.xeniatest_guest_ram_pointer();
    this.ramSize = exports.xeniatest_guest_ram_size();
  }
  // Must be called AFTER reset() (reset clears guest RAM).
  writeBE32(offset, value) { this.mem.setUint32(this.ramBase + offset, value >>> 0, false); }
  writeBE64(offset, value) { this.mem.setBigUint64(this.ramBase + offset, BigInt(value), false); }
  writeBytes(offset, bytes) { bytes.forEach((b, i) => this.mem.setUint8(this.ramBase + offset + i, b)); }
  readBE64(offset) { return this.mem.getBigUint64(this.ramBase + offset, false); }
  loadProgram(entry, words) { words.forEach((w, i) => this.writeBE32(entry + i * 4, w)); }
  consoleText() {
    const len = this.x.xeniatest_console_length();
    const start = this.x.xeniatest_console_data();
    const bytes = new Uint8Array(this.x.memory.buffer, start, len);
    const text = new TextDecoder().decode(bytes);
    this.x.xeniatest_console_consume(len);
    return text;
  }
  gpr(i) { return this.x.xeniatest_gpr(i); }
  xer() { return this.x.xeniatest_xer(); }
}

// --- Instruction encoders (big-endian, PowerISA layout) ----------------------
const enc = {
  addi: (rt, ra, simm) => (14 << 26) | (rt << 21) | (ra << 16) | (simm & 0xffff),
  addis: (rt, ra, simm) => (15 << 26) | (rt << 21) | (ra << 16) | (simm & 0xffff),
  li: (rt, imm) => enc.addi(rt, 0, imm),
  lis: (rt, imm) => enc.addis(rt, 0, imm),
  addx: (rt, ra, rb) => (31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (266 << 1),
  addex: (rt, ra, rb) => (31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (138 << 1),
  addic: (rt, ra, simm) => (12 << 26) | (rt << 21) | (ra << 16) | (simm & 0xffff),
  subf: (rt, ra, rb) => (31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (40 << 1),
  xorx: (ra, rs, rb, rc = 0) => (31 << 26) | (rs << 21) | (ra << 16) | (rb << 11) | (316 << 1) | rc,
  xori: (ra, rs, uimm) => (26 << 26) | (rs << 21) | (ra << 16) | (uimm & 0xffff),
  lwz: (rt, ra, simm) => (32 << 26) | (rt << 21) | (ra << 16) | (simm & 0xffff),
  lwa: (rt, ra, ds) => (58 << 26) | (rt << 21) | (ra << 16) | ((ds >> 2) & 0x3fff) << 2 | 2,
  ld: (rt, ra, ds) => (58 << 26) | (rt << 21) | (ra << 16) | ((ds >> 2) & 0x3fff) << 2,
  stb: (rs, ra, simm) => (38 << 26) | (rs << 21) | (ra << 16) | (simm & 0xffff),
  stw: (rs, ra, simm) => (36 << 26) | (rs << 21) | (ra << 16) | (simm & 0xffff),
  std: (rs, ra, ds) => (62 << 26) | (rs << 21) | (ra << 16) | ((ds >> 2) & 0x3fff) << 2,
  cmpi: (crf, l, ra, simm) => (11 << 26) | (((crf << 2) | l) << 21) | (ra << 16) | (simm & 0xffff),
  cmpw: (crf, ra, rb, l = 0) => (31 << 26) | (((crf << 2) | l) << 21) | (ra << 16) | (rb << 11),
  mtspr: (rt, spr) => {
    const swapped = ((spr & 0x1f) << 5) | ((spr >> 5) & 0x1f);
    return (31 << 26) | (rt << 21) | (swapped << 11) | (467 << 1);
  },
  mfspr: (rt, spr) => {
    const swapped = ((spr & 0x1f) << 5) | ((spr >> 5) & 0x1f);
    return (31 << 26) | (rt << 21) | (swapped << 11) | (339 << 1);
  },
  b: (li, aa = 0, lk = 0) => (18 << 26) | ((li & 0xffffff) << 2) | (aa << 1) | lk,
  bc: (bo, bi, bd, aa = 0, lk = 0) =>
    (16 << 26) | (bo << 21) | (bi << 16) | (((bd >> 2) & 0x3fff) << 2) | (aa << 1) | lk,
  bclr: (bo, bi, lk = 0) => (19 << 26) | (bo << 21) | (bi << 16) | (16 << 1) | lk,
  sc: (lev = 0) => (17 << 26) | (lev << 5) | 2,
};
const CTR = 9;
const ENTRY = 0x10000;

// --- Decoder / disassembler ---------------------------------------------------
test('decoder matches the xenia-canary opcode table', async () => {
  const x = await loadCore();
  const cases = [
    [0x7c000214, 'addx', 'integer'],
    [0x7c0001d4, 'addmex', 'integer'],
    [0x38000000, 'addi', 'integer'],
    [0x3c000000, 'addis', 'integer'],
    [0x7c000114, 'addex', 'integer'],
    [0x4c000020, 'bclrx', 'branch'],
    [0x40000000, 'bcx', 'branch'],
    [0x48000000, 'bx', 'branch'],
    [0x7c000000, 'cmp', 'integer'],
    [0x10000000, 'vaddubm', 'vmx'],
    [0x1000000a, 'vaddfp', 'vmx'],
    [0x88000000, 'lbz', 'memory'],
    [0xe8000000, 'ld', 'memory'],
    [0xfc00002a, 'faddx', 'floating-point'],
    [0x7c0002a6, 'mfspr', 'control'],
    [0x00000000, '', 'unknown'],
  ];
  for (const [code, name, group] of cases) {
    assert.equal(readCString(x, x.xeniatest_opcode_name(code)), name, `name of 0x${code.toString(16)}`);
    assert.equal(readCString(x, x.xeniatest_opcode_group(code)), group, `group of 0x${code.toString(16)}`);
  }
});

function readCString(exports, ptr) {
  const memory = exports.memory;
  const bytes = new Uint8Array(memory.buffer, ptr);
  let end = 0;
  while (bytes[end] !== 0 && end < 64) end++;
  return new TextDecoder().decode(bytes.subarray(0, end));
}

test('disassembler renders instructions from the xenia-canary tables', async () => {
  const x = await loadCore();
  const len = x.xeniatest_disasm(0x10000, 0x38600005); // addi r3, r0, 0x5
  assert.ok(len > 0);
  const line = readCString(x, x.xeniatest_disasm_line());
  assert.match(line, /addi/);
  assert.match(line, /r3/);
  assert.match(line, /0x5/);

  const len2 = x.xeniatest_disasm(0x10000, 0x7c635a14 >>> 0); // add r3, r4, r5
  assert.ok(len2 > 0);
  const line2 = readCString(x, x.xeniatest_disasm_line());
  assert.match(line2, /add/);
});

// --- Interpreter ---------------------------------------------------------------
test('sum 1..10 program computes 55', async () => {
  const x = await loadCore();
  const core = new Core(x);
  x.xeniatest_reset(BigInt(ENTRY));
  core.loadProgram(ENTRY, [
    enc.li(3, 0),                    // sum = 0
    enc.li(4, 1),                    // i = 1
    enc.li(5, 11),                   // n = 11 (stop after adding 10)
    enc.addx(3, 3, 4),               // loop: sum += i
    enc.addi(4, 4, 1),               // i++
    enc.cmpw(0, 4, 5),               // compare i, n
    enc.bc(4, 2, 0x0c - 0x18),       // bne loop (cr0.EQ == 0)
    enc.sc(),
  ]);
  const status = x.xeniatest_run(1000);
  assert.equal(status, 1); // sc halted
  assert.equal(core.gpr(3), 55n);
  assert.equal(x.xeniatest_instructions_executed(), 43n); // sc does not count as a retired instruction
});

test('addic/addex propagate carry through XER', async () => {
  const x = await loadCore();
  const core = new Core(x);
  x.xeniatest_reset(BigInt(ENTRY));
  core.loadProgram(ENTRY, [
    enc.li(3, -1),
    enc.addic(4, 3, 1),   // r4 = 0x00000000, XER.CA = 1
    enc.addex(5, 3, 4),   // r5 = -1 + 0 + 1 = 0, XER.CA = 1
    enc.addex(6, 0, 0),   // r6 = 0 + 0 + CA = 1, XER.CA = 0
    enc.sc(),
  ]);
  const status = x.xeniatest_run(1000);
  assert.equal(status, 1);
  assert.equal(core.gpr(4), 0n);
  assert.equal(core.gpr(5), 0n);
  assert.equal(core.gpr(6), 1n);
  assert.equal(core.xer() & 0x20000000n, 0n);
});

test('64-bit ld/std roundtrip through guest RAM', async () => {
  const x = await loadCore();
  const core = new Core(x);
  x.xeniatest_reset(BigInt(ENTRY));
  // Code lives at ENTRY; data lives at 0x20000.
  core.writeBE64(0x20000, 0x1122334455667788n);
  core.loadProgram(ENTRY, [
    enc.lis(3, 0x0002),  // r3 = 0x00020000
    enc.ld(4, 3, 0),      // r4 = mem[0x20000]
    enc.std(4, 3, 8),     // mem[0x20008] = r4
    enc.sc(),
  ]);
  const status = x.xeniatest_run(1000);
  assert.equal(status, 1);
  assert.equal(core.gpr(4), 0x1122334455667788n);
  assert.equal(core.readBE64(0x20008), 0x1122334455667788n);
});

test('lwz zero-extends and lwa sign-extends', async () => {
  const x = await loadCore();
  const core = new Core(x);
  x.xeniatest_reset(BigInt(ENTRY));
  core.writeBE32(0x20000, 0xffff0001);
  core.loadProgram(ENTRY, [
    enc.lis(3, 0x0002),  // r3 = 0x00020000
    enc.lwz(4, 3, 0),
    enc.lwa(5, 3, 0),
    enc.sc(),
  ]);
  const status = x.xeniatest_run(1000);
  assert.equal(status, 1);
  assert.equal(core.gpr(4), 0xffff0001n);
  assert.equal(core.gpr(5) & 0xffffffffffffffffn, 0xffffffffffff0001n); // wasm i64 returns are signed JS BigInts
});

test('console MMIO window prints bytes and sc halts', async () => {
  const x = await loadCore();
  const core = new Core(x);
  x.xeniatest_reset(BigInt(ENTRY));
  core.loadProgram(ENTRY, [
    enc.addis(3, 0, 0x00f0), // r3 = 0x00F00000 (console MMIO base)
    enc.li(4, 72),           // 'H'
    enc.stb(4, 3, 0),
    enc.li(4, 105),          // 'i'
    enc.stb(4, 3, 0),
    enc.sc(),
  ]);
  const status = x.xeniatest_run(1000);
  assert.equal(status, 1);
  assert.equal(core.consoleText(), 'Hi');
});

test('CTR-counted loop with bdnz', async () => {
  const x = await loadCore();
  const core = new Core(x);
  x.xeniatest_reset(BigInt(ENTRY));
  core.loadProgram(ENTRY, [
    enc.li(3, 0),
    enc.li(4, 5),
    enc.mtspr(4, CTR),           // CTR = 5
    enc.addi(3, 3, 1),           // loop:
    enc.bc(0x12, 0, 0x0c - 0x10),// bdnz loop
    enc.sc(),
  ]);
  const status = x.xeniatest_run(1000);
  assert.equal(status, 1);
  assert.equal(core.gpr(3), 5n);
});

test('bl/blr roundtrip through the link register', async () => {
  const x = await loadCore();
  const core = new Core(x);
  x.xeniatest_reset(BigInt(ENTRY));
  core.loadProgram(ENTRY, [
    enc.b(2, 0, 1),   // bl ENTRY+8 (target = CIA + LI*4, CIA = ENTRY)
    enc.sc(),
    enc.addi(3, 3, 1),
    enc.bclr(20, 0),  // blr
  ]);
  const status = x.xeniatest_run(1000);
  assert.equal(status, 1);
  assert.equal(core.gpr(3), 1n);
});

test('vmx instruction traps with trap-on-unimplemented and is skippable', async () => {
  const x = await loadCore();
  const core = new Core(x);
  x.xeniatest_reset(BigInt(ENTRY));
  core.loadProgram(ENTRY, [
    0x10000000 | (1 << 21) | (2 << 16) | (3 << 11) | (1 << 6), // vaddubm v1, v2, v3
    enc.li(3, 42),
    enc.sc(),
  ]);
  let status = x.xeniatest_run(100);
  assert.equal(status, 2); // unimplemented, trapped
  assert.equal(x.xeniatest_unimplemented_count(), 1n);
  // Disable trapping and run to completion; VMX is skipped.
  x.xeniatest_reset(BigInt(ENTRY));
  x.xeniatest_set_trap_on_unimplemented(0);
  core.loadProgram(ENTRY, [
    0x10000000 | (1 << 21) | (2 << 16) | (3 << 11) | (1 << 6),
    enc.li(3, 42),
    enc.sc(),
  ]);
  status = x.xeniatest_run(100);
  assert.equal(status, 1);
  assert.equal(core.gpr(3), 42n);
  assert.equal(x.xeniatest_unimplemented_count(), 1n);
});

test('out-of-bounds guest access faults cleanly', async () => {
  const x = await loadCore();
  const core = new Core(x);
  x.xeniatest_reset(BigInt(ENTRY));
  core.loadProgram(ENTRY, [
    enc.addis(3, 0, 0x1000), // r3 = 0x10000000 (256 MiB > 64 MiB RAM)
    enc.lwz(4, 3, 0),
    enc.sc(),
  ]);
  const status = x.xeniatest_run(100);
  assert.equal(status, 3);
  assert.equal(x.xeniatest_last_fault_address(), 0x10000000n);
});

test('guest RAM size is the documented fixed budget', async () => {
  const x = await loadCore();
  assert.equal(x.xeniatest_guest_ram_size(), 64 * 1024 * 1024);
  assert.ok(x.xeniatest_guest_ram_pointer() > 0);
});
