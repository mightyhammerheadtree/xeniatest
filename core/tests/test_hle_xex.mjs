/**
 * HLE XEX2 pipeline tests — build synthetic (but fully valid) retail/devkit
 * XEX2 images, decrypt + decompress them client-side, lift them to wasm, and
 * run them with the browser HLE host handling xboxkrnl!DbgPrint.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildSyntheticXex,
  buildSyntheticXexFromImage,
  buildEncryptedUncompressedXex,
  EXPECTED_OUTPUT,
  HELLO,
  ANSWER,
  IMAGE_BASE,
  ENTRY,
  THUNK_ADDR,
} from './hle_xex.mjs';
import { assemble } from '../../public/js/ppc_asm.js';
import { buildImage, runImage, RecompError } from '../../public/js/recomp/recompiler.js';
import { loadXex2, XexLoadError } from '../../public/js/recomp/xex_loader.js';

function maskedEntry() { return ENTRY & 0x03ffffff; }
function maskedBase() { return IMAGE_BASE & 0x03ffffff; }
function maskedThunk(i = 0) { return (THUNK_ADDR + i * 4) & 0x03ffffff; }

test('retail-key BASIC image decrypts and exposes its import thunks', () => {
  const { xex } = buildSyntheticXex({ compression: 'basic', encryption: 'normal' });
  const loaded = loadXex2(xex);
  assert.equal(loaded.base, maskedBase());
  assert.equal(loaded.entry, maskedEntry());
  assert.equal(loaded.usedDevkitKey, false);
  assert.ok(loaded.image.length >= 4100, 'decrypted image is present');
  assert.deepEqual(
    loaded.imports.map((i) => [i.addr, i.type, i.ordinal, i.moduleId, i.name]),
    [
      [maskedThunk(0), 1, 0x0003, 'xboxkrnl', 'DbgPrint'],
      [maskedThunk(1), 1, 0x0003, 'xboxkrnl', 'DbgPrint'],
    ],
  );
});

test('unencrypted BASIC image loads and runs identically', async () => {
  const { xex } = buildSyntheticXex({ compression: 'basic', encryption: 'none' });
  const built = buildImage('xex2', null, xex);
  assert.equal(built.entry, maskedEntry());
  assert.equal(built.hleCalls.size, 2);
  let out = '';
  const res = await runImage(built, { onConsole: (s) => { out += s; } });
  assert.equal(out, EXPECTED_OUTPUT);
  assert.equal(res.exitCode, 0);
});

test('uncompressed but AES-encrypted image loads and runs', async () => {
  const { xex } = buildEncryptedUncompressedXex();
  const built = buildImage('xex2', null, xex);
  let out = '';
  const res = await runImage(built, { onConsole: (s) => { out += s; } });
  assert.equal(out, EXPECTED_OUTPUT);
  assert.equal(res.exitCode, 0);
});

test('BASIC compression with implicit zero span runs to the exit door', async () => {
  const { xex, image } = buildSyntheticXex({ compression: 'basic', encryption: 'normal', trailingZero: 0x1000 });
  const loaded = loadXex2(xex);
  assert.equal(loaded.image.length, Math.ceil(image.length / 16) * 16 + 0x1000);
  assert.ok(loaded.image.subarray(Math.ceil(image.length / 16) * 16).every((b) => b === 0), 'implicit zeros are present');
  let out = '';
  const res = await runImage(buildImage('xex2', null, xex), { onConsole: (s) => { out += s; } });
  assert.equal(out, EXPECTED_OUTPUT);
  assert.equal(res.exitCode, 0);
});

test('devkit-key image falls back to the devkit key automatically', async () => {
  const { xex } = buildSyntheticXex({ devkit: true });
  const loaded = loadXex2(xex);
  assert.equal(loaded.usedDevkitKey, true);
  let out = '';
  const res = await runImage(buildImage('xex2', null, xex), { onConsole: (s) => { out += s; } });
  assert.equal(out, EXPECTED_OUTPUT);
  assert.equal(res.exitCode, 0);
});

test('retail image runs end-to-end with DbgPrint routed through the HLE host', async () => {
  const { xex } = buildSyntheticXex({ compression: 'basic', encryption: 'normal' });
  const built = buildImage('xex2', null, xex);
  assert.equal(built.hleCalls.get(maskedThunk(0)).name, 'DbgPrint');
  let out = '';
  const res = await runImage(built, { onConsole: (s) => { out += s; } });
  assert.ok(out.includes(HELLO), 'DbgPrint fmt1 reached the console');
  assert.ok(out.includes(ANSWER), 'DbgPrint fmt2 with %d argument reached the console');
  assert.equal(out, EXPECTED_OUTPUT);
  assert.equal(res.exitCode, 0);
  assert.ok(res.stats.insns > 0 && res.stats.fragments > 0);
});

test('import of an unimplemented kernel ordinal traps with the export name', async () => {
  // ordinal 0x0001 = DbgBreakPoint — a real xboxkrnl export the HLE host does not implement
  const { xex } = buildSyntheticXex({ compression: 'basic', encryption: 'normal', importOrdinals: [0x0001] });
  const built = buildImage('xex2', null, xex);
  assert.equal(built.hleCalls.get(maskedThunk(0)).name, 'DbgBreakPoint');
  await assert.rejects(
    () => runImage(built, { onConsole: () => {} }),
    (e) => /unimplemented/i.test(String(e && e.message)) && /DbgBreakPoint/.test(String(e && e.message)),
  );
});

test('image wrapped in the retail key with the wrong session key is rejected loudly', () => {
  const { xex } = buildSyntheticXex({ wrongKey: true });
  assert.throws(() => buildImage('xex2', null, xex), (e) => e instanceof RecompError && /decrypt|PE header/i.test(e.message));
  assert.throws(() => loadXex2(xex), (e) => e instanceof XexLoadError && /decrypt|PE header/i.test(e.message));
});

test('LZX ("normal") compression is rejected with an actionable message', () => {
  const { xex } = buildSyntheticXex({ compression: 'basic', encryption: 'normal' });
  const patched = Uint8Array.from(xex);
  // FILE_FORMAT_INFO @ 0x100: u16 info_size, u16 encryption_type @ 0x104, u16 compression_type @ 0x106
  patched[0x106] = 0x00;
  patched[0x107] = 0x02; // compression_type = 2 (NORMAL / LZX)
  assert.throws(() => loadXex2(patched), (e) => e instanceof XexLoadError && /LZX/i.test(e.message));
  assert.throws(() => buildImage('xex2', null, patched), (e) => e instanceof RecompError && /LZX/i.test(e.message));
});

test('delta compression is rejected with an actionable message', () => {
  const { xex } = buildSyntheticXex({ compression: 'basic', encryption: 'normal' });
  const patched = Uint8Array.from(xex);
  patched[0x106] = 0x00;
  patched[0x107] = 0x03; // compression_type = 3 (DELTA)
  assert.throws(() => loadXex2(patched), (e) => e instanceof XexLoadError && /delta/i.test(e.message));
});

test('trailing-zero blocks are accounted for in the arena image', () => {
  const { xex, image } = buildSyntheticXex({ compression: 'basic', encryption: 'normal', trailingZero: 0x300 });
  const aligned = Math.ceil(image.length / 16) * 16;
  const loaded = loadXex2(xex);
  assert.equal(loaded.image.length, aligned + 0x300);
  assert.ok([...loaded.image.subarray(aligned)].every((b) => b === 0));
});

test('exported project zip runs standalone via `node run.js` with no firmware', async () => {
  const { mkdtempSync, writeFileSync, mkdirSync, rmSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join, dirname } = await import('node:path');
  const { execFileSync } = await import('node:child_process');
  const { buildProjectFiles } = await import('../../public/js/recomp/templates.js');
  const { recompileImage } = await import('../../public/js/recomp/recompiler.js');

  // synthetic retail-key XEX2 → recompiled wasm
  const { xex } = buildSyntheticXex({ compression: 'basic', encryption: 'normal' });
  const built = buildImage('xex2', null, xex);
  const { wasm, stats } = recompileImage(built.arena, built.entry, { hleCalls: built.hleCalls });
  const files = buildProjectFiles({
    name: 'hello-xex', kind: 'xex2', entry: built.entry, wasm,
    segments: built.segments, stats, now: new Date(2026, 10, 15), hleCalls: built.hleCalls,
  });
  const byName = new Map(files.map((f) => [f.name, f.data]));

  // manifest carries the HLE import map and announces no firmware requirement
  const manifest = JSON.parse(new TextDecoder().decode(byName.get('manifest.json')));
  assert.equal(manifest.hle.model, 'hle');
  assert.equal(manifest.hle.firmwareRequired, false);
  assert.deepEqual(manifest.hle.modules, ['xboxkrnl', 'xam', 'xbdm']);
  assert.deepEqual(
    manifest.hle.calls.map((c) => [c.moduleId, c.ordinal, c.name]).sort(),
    [
      ['xboxkrnl', 3, 'DbgPrint'],
      ['xboxkrnl', 3, 'DbgPrint'],
    ],
  );
  // guest-wasm.js embeds the same map for the browser runner
  const gjs = new TextDecoder().decode(byName.get('guest-wasm.js'));
  assert.match(gjs, /HLE_CALLS/);

  // unzip into a temp dir and run it for real
  const dir = mkdtempSync(join(tmpdir(), 'xenon-xex-e2e-'));
  try {
    for (const [name, data] of byName) {
      const dest = join(dir, name);
      mkdirSync(dirname(dest), { recursive: true });
      writeFileSync(dest, data);
    }
    const stdout = execFileSync(process.execPath, ['run.js'], { cwd: dir, encoding: 'utf8' });
    assert.equal(stdout, EXPECTED_OUTPUT);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('guest allocates memory through NtAllocateVirtualMemory and round-trips a byte through it', async () => {
  // Hand-written guest: asks the (HLE) kernel for a page, stores 0x2a in it,
  // reads it back, and DbgPrints it with %d. Output must be exactly "42\n".
  const src = [
    '.org 0x82000000',
    "    .byte 0x4D, 0x5A            # 'MZ' PE signature",
    '    .space 0x1FE',
    '    lis   r31, slot@ha',
    '    addi  r31, r31, slot@l',
    '    li    r3, 0',
    '    stw   r3, 0(r31)            # base_addr = 0 — kernel chooses',
    '    li    r3, 0x1234',
    '    stw   r3, 4(r31)            # region_size = 0x1234',
    '    mr    r3, r31              # &base_addr',
    '    addi  r4, r31, 4           # &region_size',
    '    li    r5, 0x1000           # X_MEM_COMMIT',
    '    li    r6, 0x04             # X_PAGE_READWRITE',
    '    li    r7, 0                # debug_memory = FALSE',
    '    bl    thunk_0              # NtAllocateVirtualMemory(...)',
    '    lwz   r5, 0(r31)           # allocated base (written back by the kernel)',
    '    li    r6, 0x2a             # 42',
    '    stb   r6, 0(r5)',
    '    lbz   r4, 0(r5)',
    '    lis   r3, fmt@ha',
    '    addi  r3, r3, fmt@l',
    '    bl    thunk_1              # DbgPrint(fmt, 42)',
    '    li    r9, 0',
    '    lis   r10, 0x00f0',
    '    ori   r10, r10, 0x0ffc',
    '    stw   r9, 0(r10)            # exit door',
    'spin:',
    '    b     spin',
    'fmt:',
    '    .asciiz "%d\\n"',
    'slot:',
    '    .space 8',
    '    .org 0x82001000',
    'thunk_0:',
    '    .word 0x010000cc            # 0x01000000 | 0x00cc NtAllocateVirtualMemory',
    'thunk_1:',
    '    .word 0x01000003            # 0x01000000 | 0x0003 DbgPrint',
  ].join('\n');
  const { bytes } = assemble(src);
  const { xex } = buildSyntheticXexFromImage(bytes, { importOrdinals: [0x00cc, 0x0003] });
  let out = '';
  const res = await runImage(buildImage('xex2', null, xex), { onConsole: (s) => { out += s; } });
  assert.equal(out, '42\n');
  assert.equal(res.exitCode, 0);
});
