/**
 * ELF loader test — builds a minimal big-endian ELF64 PowerPC executable
 * around the bundled Hello program, parses it with the same inspect.js code
 * the page uses, lifts it to WASM and runs it. Proves ELF homebrew works
 * end to end without the old interpreter.
 *
 * Run with: npm run test:elf
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { assemble } from '../../public/js/ppc_asm.js';
import { BUILTIN_PROGRAMS } from '../../public/js/programs.js';
import { parseElf } from '../../public/js/inspect.js';
import { buildImage, runImage } from '../../public/js/recomp/recompiler.js';

const HELLO = BUILTIN_PROGRAMS.find((p) => p.id === 'builtin:hello');

function makeElf(loads) {
  // loads: [{ vaddr, align, bytes }]
  const EHDR = 64, PHDR = 56;
  const n = loads.length;
  let off = EHDR + PHDR * n;
  const segs = loads.map((l) => {
    const s = { ...l, fileOffset: off };
    off += l.bytes.length;
    return s;
  });
  const total = off;
  const buf = new Uint8Array(total);
  const dv = new DataView(buf.buffer);
  buf.set([0x7f, 0x45, 0x4c, 0x46, 2, 2, 1, 0]); // ELF64, big-endian, current, SYSV
  dv.setUint16(16, 2, false);      // ET_EXEC
  dv.setUint16(18, 21, false);     // EM_PPC64
  dv.setUint32(20, 1, false);      // e_version
  dv.setBigUint64(24, BigInt(loads[0].vaddr), false); // e_entry
  dv.setBigUint64(32, BigInt(EHDR), false);           // e_phoff
  dv.setBigUint64(40, 0n, false);                     // e_shoff
  dv.setUint32(48, 0, false);                         // e_flags
  dv.setUint16(52, EHDR, false);   // e_ehsize
  dv.setUint16(54, PHDR, false);   // e_phentsize
  dv.setUint16(56, n, false);      // e_phnum
  dv.setUint16(58, 0, false);
  dv.setUint16(60, 0, false);
  dv.setUint16(62, 0, false);
  segs.forEach((s, i) => {
    const p = EHDR + i * PHDR;
    dv.setUint32(p, 1, false);      // PT_LOAD
    dv.setUint32(p + 4, 5, false);  // PF_R | PF_X
    dv.setBigUint64(p + 8, BigInt(s.fileOffset), false);
    dv.setBigUint64(p + 16, BigInt(s.vaddr), false);
    dv.setBigUint64(p + 24, BigInt(s.vaddr), false);
    dv.setBigUint64(p + 32, BigInt(s.bytes.length), false); // p_filesz
    dv.setBigUint64(p + 40, BigInt(s.bytes.length), false); // p_memsz
    dv.setBigUint64(p + 48, 0x1000n, false);                // p_align
    buf.set(s.bytes, s.fileOffset);
  });
  return buf;
}

test('synthetic ELF64 around the hello sample recompiles and runs', async () => {
  const { bytes } = assemble(HELLO.source);
  const elf = makeElf([{ vaddr: 0x10000, align: 0x1000, bytes }]);
  const parsed = parseElf(elf);
  assert.equal(parsed.format, 'ELF64');
  assert.equal(parsed.machine, 21);
  const built = buildImage('elf', parsed, elf);
  assert.equal(built.entry, 0x10000);
  assert.equal(built.arena.length >= 0x10000 + bytes.length, true);
  // bytes landed in the arena
  assert.equal(new Uint8Array(built.arena.slice(0x10000, 0x10000 + bytes.length)).every((b, i) => b === bytes[i]), true);
  let out = '';
  const res = await runImage(built, { onConsole: (s) => { out += s; } });
  assert.match(out, /Hello from Xenon PPC, assembled and executed inside your browser!\n/);
  assert.equal(res.exitCode, 0);
});
