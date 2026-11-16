/**
 * Built-in program test — assembles every program shipped in the web app's
 * library and runs it on the real WASM core, asserting on console output
 * and halt status. Keeps js/programs.js honest: if the assembler or the
 * interpreter breaks, these fail.
 *
 * Run with: npm run test:builtins
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { assemble } from '../../public/js/ppc_asm.js';
import { BUILTIN_PROGRAMS } from '../../public/js/programs.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const WASM_PATH = join(ROOT, 'public', 'core', 'xeniatest-core.wasm');

async function instantiate() {
  const bytes = readFileSync(WASM_PATH);
  const imports = {
    wasi_snapshot_preview1: {
      args_get: () => 0,
      args_sizes_get: () => 0,
      proc_exit: (code) => { throw new Error(`proc_exit(${code})`); },
    },
  };
  const { instance } = await WebAssembly.instantiate(bytes, imports);
  return instance.exports;
}

function runProgram(x, bytes, entry, maxInstructions = 10_000_000, base = 0x10000) {
  x.xeniatest_reset(BigInt(entry));
  x.xeniatest_set_trap_on_unimplemented(0);
  const ramBase = x.xeniatest_guest_ram_pointer();
  new Uint8Array(x.memory.buffer, ramBase + base).set(bytes);
  const status = x.xeniatest_run(maxInstructions);
  const len = x.xeniatest_console_length();
  const start = x.xeniatest_console_data();
  const out = new TextDecoder().decode(new Uint8Array(x.memory.buffer, start, len).slice());
  x.xeniatest_console_consume(len);
  return { status, out };
}

test('every built-in program assembles', () => {
  for (const program of BUILTIN_PROGRAMS) {
    const { bytes } = assemble(program.source);
    assert.ok(bytes.length > 100, `${program.id}: suspiciously small payload`);
    assert.ok(bytes.length < 0x10000, `${program.id}: too large`);
  }
});

test('hello prints its greeting and halts via the exit door', async () => {
  const x = await instantiate();
  const program = BUILTIN_PROGRAMS.find((p) => p.id === 'builtin:hello');
  const image = assemble(program.source);
  const { status, out } = runProgram(x, image.bytes, program.entry, 10_000_000, image.base);
  assert.equal(status, 1, `halted status (got ${status})`);
  assert.equal(out, 'Hello from Xenon PPC, assembled and executed inside your browser!\n');
});

test('fib computes fib(25) = 75025', async () => {
  const x = await instantiate();
  const program = BUILTIN_PROGRAMS.find((p) => p.id === 'builtin:fib');
  const image = assemble(program.source);
  const { status, out } = runProgram(x, image.bytes, program.entry, 10_000_000, image.base);
  assert.equal(status, 1, `halted status (got ${status})`);
  assert.equal(out, 'fib(25) = 75025\n');
});

test('countdown prints 10..1 then liftoff', async () => {
  const x = await instantiate();
  const program = BUILTIN_PROGRAMS.find((p) => p.id === 'builtin:countdown');
  const image = assemble(program.source);
  const { status, out } = runProgram(x, image.bytes, program.entry, 10_000_000, image.base);
  assert.equal(status, 1, `halted status (got ${status})`);
  const lines = out.trimEnd().split('\n');
  assert.deepEqual(lines.slice(1, 11), ['10', '9', '8', '7', '6', '5', '4', '3', '2', '1']);
  assert.equal(lines.at(-1), 'Liftoff!');
});

test('branch displacement: backward branches assemble with signed offsets', () => {
  const { bytes, base } = assemble('.org 0x10000\nstart:\n addi r3, r3, 1\n b start\n');
  assert.equal(base, 0x10000);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const instr = view.getUint32(4, false);
  assert.equal(instr >>> 26, 18, 'opcode bx');
  const li = (instr >>> 2) & 0xffffff;
  assert.equal(li, (-1 & 0xffffff), 'branch back to itself is -1 words');
});

test('assembler rejects unknown instructions with line numbers', () => {
  assert.throws(() => assemble('.org 0x10000\nfrob r3, r4\n'), /line 2: unsupported instruction 'frob'/);
});
