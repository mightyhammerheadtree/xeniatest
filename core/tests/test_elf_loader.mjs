/**
 * ELF loader test — builds a minimal big-endian ELF64 PowerPC executable
 * around the built-in hello program and boots it through Runner.loadFile,
 * the exact path the Run view uses. Proves ELF homebrew works end to end.
 *
 * Run with: npm run test:elf
 */
import { test, before } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { Runner } from '../../public/js/runner.js';
import { assemble } from '../../public/js/ppc_asm.js';
import { BUILTIN_PROGRAMS } from '../../public/js/programs.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const PUBLIC = join(ROOT, 'public');

before(() => {
  // fetch shim for the WASM core (relative URL → local file).
  globalThis.fetch = async (url, options = {}) => {
    const resolved = new URL(String(url), 'http://localhost:3000/');
    const rel = decodeURIComponent(resolved.pathname);
    const filePath = join(PUBLIC, rel);
    if ((options.method ?? 'GET').toUpperCase() === 'HEAD') {
      return new Response(null, { status: 200, headers: { 'content-length': String(statSync(filePath).size) } });
    }
    return new Response(readFileSync(filePath), { status: 200, headers: { 'content-type': 'application/wasm' } });
  };
  // The runner schedules frames via rAF; emulate with timers.
  globalThis.requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 1);
  globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
});

function buildElf64Be(code, entry) {
  const PHOFF = 64;
  const PHENT = 56;
  const CODE_OFF = PHOFF + PHENT;
  const total = CODE_OFF + code.length;
  const buf = new Uint8Array(total);
  const dv = new DataView(buf.buffer);
  buf.set([0x7f, 0x45, 0x4c, 0x46, 2, 2, 1, 0]); // ELF64, MSB, current, SVR4
  dv.setUint16(16, 2, false);                    // ET_EXEC
  dv.setUint16(18, 21, false);                   // EM_PPC64
  dv.setUint32(20, 1, false);                    // e_version
  dv.setBigUint64(24, BigInt(entry), false);     // e_entry
  dv.setBigUint64(32, BigInt(PHOFF), false);     // e_phoff
  dv.setBigUint64(40, 0n, false);                // e_shoff
  dv.setUint32(48, 0, false);                    // e_flags
  dv.setUint16(52, 64, false);                   // e_ehsize
  dv.setUint16(54, PHENT, false);                // e_phentsize
  dv.setUint16(56, 1, false);                    // e_phnum
  dv.setUint16(58, 64, false);                   // e_shentsize
  dv.setUint16(60, 0, false);                    // e_shnum
  dv.setUint16(62, 0, false);                    // e_shstrndx
  // Program header: one R+X LOAD at `entry`.
  dv.setUint32(PHOFF + 0, 1, false);             // PT_LOAD
  dv.setUint32(PHOFF + 4, 5, false);             // PF_R | PF_X
  dv.setBigUint64(PHOFF + 8, BigInt(CODE_OFF), false);
  dv.setBigUint64(PHOFF + 16, BigInt(entry), false);
  dv.setBigUint64(PHOFF + 24, BigInt(entry), false);
  dv.setBigUint64(PHOFF + 32, BigInt(code.length), false);
  dv.setBigUint64(PHOFF + 40, BigInt(code.length), false);
  dv.setBigUint64(PHOFF + 48, 0x1000n, false);
  buf.set(code, CODE_OFF);
  return buf;
}

async function waitFor(predicate, { timeout = 15000, step = 10 } = {}) {
  const start = Date.now();
  for (;;) {
    const value = predicate();
    if (value) return value;
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, step));
  }
}

test('ELF64 BE PPC homebrew boots to the entry point and runs', async () => {
  const program = BUILTIN_PROGRAMS.find((p) => p.id === 'builtin:hello');
  const image = assemble(program.source);
  const elfBytes = buildElf64Be(image.bytes, program.entry);
  const file = new File([elfBytes], 'hello-homebrew.elf');

  const runner = new Runner();
  await runner.init();

  let consoleOut = '';
  let halted = null;
  runner.onChange((kind, payload) => {
    if (kind === 'console') consoleOut += payload;
    if (kind === 'halted') halted = payload;
  });

  const result = await runner.loadFile(file);
  assert.equal(result.ok, true, result.reason);
  assert.equal(result.kind, 'elf');

  runner.run();
  await waitFor(() => halted);

  assert.equal(halted.status, 1, `expected clean exit-door halt, got ${halted.status}`);
  assert.match(consoleOut, /inside your browser/);
});

test('Xbox 360-space ELF is rejected with a clear reason, not a crash', async () => {
  const program = BUILTIN_PROGRAMS.find((p) => p.id === 'builtin:hello');
  const image = assemble(program.source);
  // Same ELF but linked at 0x82000000 (Xbox 360 user space).
  const elfBytes = buildElf64Be(image.bytes, 0x82000000);
  const file = new File([elfBytes], 'xbox360-homebrew.elf');

  const runner = new Runner();
  await runner.init();
  const result = await runner.loadFile(file);
  assert.equal(result.ok, false);
  assert.equal(result.kind, 'elf');
  assert.match(result.reason, /Xbox 360 address space/);
});
