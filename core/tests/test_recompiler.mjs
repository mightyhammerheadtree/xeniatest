/**
 * Recompiler pipeline tests — assemble built-in Xenon PPC64 programs, lift
 * them to WebAssembly client-side, and execute the wasm to completion.
 * No browser, no network.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { assemble } from '../../public/js/ppc_asm.js';
import { BUILTIN_PROGRAMS } from '../../public/js/programs.js';
import { buildImage, runImage } from '../../public/js/recomp/recompiler.js';
import { makeZip, crc32 } from '../../public/js/recomp/zip.js';
import { buildProjectFiles } from '../../public/js/recomp/templates.js';

async function runBuiltIn(nameFragment) {
  const p = BUILTIN_PROGRAMS.find((p) => p.name.toLowerCase().includes(nameFragment));
  assert.ok(p, 'sample exists: ' + nameFragment);
  const img = assemble(p.source);
  const built = buildImage('bin', null, img.bytes);
  let out = '';
  const res = await runImage(built, { onConsole: (s) => { out += s; } });
  return { out, res };
}

test('hello sample prints and halts at the exit door', async () => {
  const { out, res } = await runBuiltIn('hello');
  assert.match(out, /Hello from Xenon PPC, assembled and executed inside your browser!\n/);
  assert.equal(res.exitCode, 0);
  assert.ok(res.stats.insns > 0 && res.stats.fragments > 0);
});

test('fibonacci sample computes fib(25) = 75025', async () => {
  const { out, res } = await runBuiltIn('fibonacci');
  assert.match(out, /fib\(25\) = 75025\n/);
  assert.equal(res.exitCode, 0);
});

test('countdown sample prints the full sequence and liftoff', async () => {
  const { out, res } = await runBuiltIn('countdown');
  assert.match(out, /Counting down:\n10\n9\n8\n7\n6\n5\n4\n3\n2\n1\nLiftoff!\n/);
  assert.equal(res.exitCode, 0);
});

test('zip writer produces a structurally valid store-only archive', () => {
  const entries = [
    { name: 'README.md', data: new TextEncoder().encode('# ok\n') },
    { name: 'a/b/c.bin', data: new Uint8Array([0, 1, 2, 3, 250, 128]) },
    { name: 'empty', data: new Uint8Array(0) },
  ];
  const zip = makeZip(entries, { now: new Date(2026, 10, 15) });
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  assert.equal(view.getUint32(0, true), 0x04034b50, 'local file header magic');
  // walk central directory
  let eocd = -1;
  for (let i = zip.length - 22; i >= 0; i--) {
    if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  assert.ok(eocd >= 0, 'EOCD present');
  const count = view.getUint16(eocd + 10, true);
  assert.equal(count, entries.length);
  let off = view.getUint32(eocd + 16, true);
  const names = [];
  for (let i = 0; i < count; i++) {
    assert.equal(view.getUint32(off, true), 0x02014b50);
    const method = view.getUint16(off + 10, true);
    const crc = view.getUint32(off + 16, true);
    const size = view.getUint32(off + 20, true);
    const nameLen = view.getUint16(off + 28, true);
    const extraLen = view.getUint16(off + 30, true);
    const commentLen = view.getUint16(off + 32, true);
    const localOff = view.getUint32(off + 42, true);
    const name = new TextDecoder().decode(zip.subarray(off + 46, off + 46 + nameLen));
    assert.equal(method, 0, 'store method');
    const lName = view.getUint16(localOff + 26, true);
    const lExtra = view.getUint16(localOff + 28, true);
    const data = zip.subarray(localOff + 30 + lName + lExtra, localOff + 30 + lName + lExtra + size);
    assert.equal(crc32(data), crc, 'crc matches for ' + name);
    names.push(name);
    off += 46 + nameLen + extraLen + commentLen;
  }
  assert.deepEqual(names.sort(), ['README.md', 'a/b/c.bin', 'empty'].sort());
});

test('project zip file set is complete and self-consistent', async () => {
  const p = BUILTIN_PROGRAMS[1]; // fib
  const built = buildImage('bin', null, assemble(p.source).bytes);
  const { recompileImage } = await import('../../public/js/recomp/recompiler.js');
  const { wasm, stats } = recompileImage(built.arena, built.entry);
  const files = buildProjectFiles({
    name: 'fib', kind: 'bin', entry: built.entry, wasm, segments: built.segments, stats, now: new Date(2026, 10, 15),
  });
  const byName = new Map(files.map((f) => [f.name, f.data]));
  for (const expected of ['README.md', 'manifest.json', 'index.html', 'runtime.js', 'guest-wasm.js', 'guest.wasm',
    'run.js', 'docs/KERNEL.md', 'kernel/.gitkeep', 'segment-0.bin', 'guest.bin']) {
    assert.ok(byName.has(expected), 'zip contains ' + expected);
  }
  // guest.wasm is a real module (magic + version)
  assert.deepEqual([...byName.get('guest.wasm').slice(0, 4)], [0, 0x61, 0x73, 0x6d]);
  // manifest parses and carries the stats
  const manifest = JSON.parse(new TextDecoder().decode(byName.get('manifest.json')));
  assert.equal(manifest.stats.fragments, stats.fragments);
  assert.equal(manifest.entryHex, '0x10000');
  // guest-wasm.js embeds the module and the image segment(s)
  const gjs = new TextDecoder().decode(byName.get('guest-wasm.js'));
  assert.ok(gjs.includes('GUEST_WASM_B64') && gjs.includes('GUEST_SEGMENTS'));
});
