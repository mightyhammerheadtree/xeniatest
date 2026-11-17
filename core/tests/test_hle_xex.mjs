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
  buildPdataXex,
  buildSwitchXex,
  SWITCH_CASE_ADDRS,
  SWITCH_CASE_OUTPUTS,
  SWITCH_DEFAULT_OUTPUT,
  EXPECTED_OUTPUT,
  HELLO,
  ANSWER,
  IMAGE_BASE,
  ENTRY,
  THUNK_ADDR,
  PDATA_HELPER_ADDR,
  PDATA_HELPER_R3,
  PDATA_OUTPUT,
  PDATA_THUNK_ADDR,
  buildVmxStubXex,
  VMX_SAVE_66,
  VMX_REST_66,
  VMX_PASS_OUTPUT,
  VMX_BAD_OUTPUT,
  buildFprStubXex,
  FPR_PASS_OUTPUT,
  FPR_SAVE_15,
} from './hle_xex.mjs';
import { detectSaveRestoreStubs } from '../../public/js/recomp/save_restore.js';
import { assemble } from '../../public/js/ppc_asm.js';
import { buildImage, recompileImage, runImage, RecompError, RecompUnimpl } from '../../public/js/recomp/recompiler.js';
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
  const discovery = { peSections: built.peSections?.length ?? null, peError: built.peError ?? null, seededFunctionStarts: built.functionStarts?.length ?? 0, switchTables: built.switchTables?.size ?? 0, vmxStubs: built.vmxStubs?.size ?? 0 };
  const files = buildProjectFiles({
    name: 'hello-xex', kind: 'xex2', entry: built.entry, wasm,
    segments: built.segments, stats, now: new Date(2026, 10, 15), hleCalls: built.hleCalls, discovery,
  });
  const byName = new Map(files.map((f) => [f.name, f.data]));

  // manifest carries the HLE import map and announces no firmware requirement
  const manifest = JSON.parse(new TextDecoder().decode(byName.get('manifest.json')));
  // manifest carries discovery provenance (seeds / switch tables / VMX stubs / PE breadcrumb)
  assert.deepEqual(manifest.discovery, discovery);
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

test('.pdata is parsed: sections, function seeds, HLE thunk excluded', () => {
  const { xex } = buildPdataXex();
  const loaded = loadXex2(xex);
  assert.equal(loaded.peError, null);
  assert.ok(Array.isArray(loaded.peSections));
  assert.deepEqual(loaded.peSections.map((s) => s.name), ['.text', '.pdata']);
  assert.equal(loaded.peSections[0].isCode, true);
  // Seeds are masked guest addresses, sorted; the HLE thunk record is filtered
  // out at the buildImage layer (never at parse time).
  assert.deepEqual(
    loaded.pdataFunctions.map((f) => f.addr),
    [0x82000200 & 0x03ffffff, 0x82000300 & 0x03ffffff, PDATA_THUNK_ADDR & 0x03ffffff],
  );
  assert.deepEqual(loaded.pdataFunctions.map((f) => f.size), [0x100, 0x10, 0x04]);
  const built = buildImage('xex2', null, xex);
  assert.equal(built.peError, null);
  // buildImage masks + drops HLE thunk seeds — the two real functions remain.
  assert.deepEqual(built.functionStarts, [0x82000200 & 0x03ffffff, 0x82000300 & 0x03ffffff]);
});

test('.pdata-seeded discovery exports a function no branch reaches', async () => {
  const { xex } = buildPdataXex();
  const built = buildImage('xex2', null, xex);
  let out = '';
  const res = await runImage(built, { onConsole: (s) => { out += s; } });
  assert.equal(out, PDATA_OUTPUT);
  assert.equal(res.exitCode, 0);
  // helper_fn: only visible through .pdata. It must exist, be callable, and
  // leave PDATA_HELPER_R3 in the exported r3 global.
  const helperName = 'fn_' + (PDATA_HELPER_ADDR & 0x03ffffff).toString(16);
  assert.equal(typeof res.exports[helperName], 'function', 'pdata-only function is exported');
  res.exports[helperName]();
  assert.equal(res.exports.r3.value, BigInt(PDATA_HELPER_R3));
  // The HLE thunk must NOT be recompiled into an export.
  assert.equal(res.exports['fn_' + (PDATA_THUNK_ADDR & 0x03ffffff).toString(16)], undefined);
});

test('control-flow-only discovery (seeds disabled) does not export the pdata-only function', () => {
  // Negative control for the test above: without .pdata seeds, helper_fn is
  // unreachable and the wasm module must not contain fn_<helper>.
  const { xex } = buildPdataXex();
  const built = buildImage('xex2', null, xex);
  const { wasm } = recompileImage(built.arena, built.entry, { hleCalls: built.hleCalls, functionStarts: [] });
  const names = WebAssembly.Module.exports(new WebAssembly.Module(wasm)).map((e) => e.name);
  assert.ok(!names.includes('fn_' + (PDATA_HELPER_ADDR & 0x03ffffff).toString(16)));
  assert.ok(names.includes('fn_' + (built.entry & 0x03ffffff).toString(16)));
});

test('jump table (absolute) is detected and dispatched statically', async () => {
  const { xex } = buildSwitchXex({ index: 1, tableType: 'absolute' });
  const built = buildImage('xex2', null, xex);
  assert.equal(built.peError, null);
  assert.equal(built.switchTables.size, 1);
  const [table] = built.switchTables.values();
  assert.equal(table.type, 0); // SWITCH_ABSOLUTE
  assert.equal(table.indexReg, 4); // r4 is the bounds-check register
  assert.deepEqual(table.labels, SWITCH_CASE_ADDRS.map((a) => a & 0x03ffffff));
  assert.equal(table.bctr, table.bctr & 0x03ffffff);
  // case fragments seeded: both case labels and (not) the table itself
  for (const a of SWITCH_CASE_ADDRS) {
    assert.ok(built.functionStarts.includes(a & 0x03ffffff), `case label 0x${a.toString(16)} seeded`);
  }
  assert.equal(built.stats === undefined, true); // sanity: stats live on recompileImage
  let out = '';
  const res = await runImage(built, { onConsole: (s) => { out += s; } });
  assert.equal(out, 'case1\n');
  assert.equal(res.exitCode, 0);
  assert.equal(res.stats.switchTables, 1);
  for (const a of SWITCH_CASE_ADDRS) {
    assert.equal(typeof res.exports['fn_' + (a & 0x03ffffff).toString(16)], 'function');
  }
});

for (const [tableType, typeId] of [['absolute', 0], ['computed', 1], ['byteoffset', 2], ['shortoffset', 3]]) {
  test(`jump table (${tableType}) dispatches to the right case`, async () => {
    for (const [index, expected] of [[0, SWITCH_CASE_OUTPUTS[0]], [2, SWITCH_CASE_OUTPUTS[2]], [7, SWITCH_DEFAULT_OUTPUT]]) {
      const { xex } = buildSwitchXex({ index, tableType });
      const built = buildImage('xex2', null, xex);
      assert.equal(built.switchTables.size, 1, `${tableType}: table detected`);
      assert.equal([...built.switchTables.values()][0].type, typeId);
      let out = '';
      const res = await runImage(built, { onConsole: (s) => { out += s; } });
      assert.equal(out, expected, `${tableType} index=${index}`);
      assert.equal(res.exitCode, 0);
    }
  });
}

test('without any table support the switch dispatch traps loudly', async () => {
  // Pre-jump-table world: no detection and no seeds either — the icall at the
  // bctr has no fn_<case> to call and must trap loudly, never misrun.
  const { xex } = buildSwitchXex({ index: 1, tableType: 'absolute' });
  const built = buildImage('xex2', null, xex);
  await assert.rejects(
    () => runImage({ arena: built.arena, entry: built.entry, hleCalls: built.hleCalls, functionStarts: [], switchTables: new Map() }, { onConsole: () => {} }),
    (e) => e instanceof RecompUnimpl || /unimplemented/i.test(String(e && e.message)),
  );
});

test('pdata-style seeding alone lets the icall fall through to the case fragment', async () => {
  // Historical accident but worth pinning: with the case fragments seeded but
  // no switchTables, the bctr's runtime icall resolves fn_<case> by name and
  // runs it — control still reaches the case (no trap). Detection is what
  // makes dispatch deterministic instead of dependent on that naming fallback.
  const { xex } = buildSwitchXex({ index: 1, tableType: 'absolute' });
  const built = buildImage('xex2', null, xex);
  let out = '';
  const res = await runImage({ arena: built.arena, entry: built.entry, hleCalls: built.hleCalls, functionStarts: built.functionStarts, switchTables: new Map() }, { onConsole: (s) => { out += s; } });
  assert.equal(out, 'case1\n');
  assert.equal(res.exitCode, 0);
});

// ---------- save/restore stub detection ----------

function writeWordBE(arena, off, word) {
  arena[off] = (word >>> 24) & 0xff;
  arena[off + 1] = (word >>> 16) & 0xff;
  arena[off + 2] = (word >>> 8) & 0xff;
  arena[off + 3] = word & 0xff;
}
const GUEST_MASK = 0x03ffffff;
const BLR_WORD = 0x4E800020;
const stfdWord = (f, imm) => (54 << 26) | (f << 21) | (12 << 16) | ((imm & 0xffff));
const liR11 = (imm) => (14 << 26) | (11 << 21) | (imm & 0xffff);
const lvxWord = (v) => (31 << 26) | ((v & 31) << 21) | (11 << 16) | (12 << 11) | (103 << 1);
const stvxWord = (v) => (31 << 26) | ((v & 31) << 21) | (11 << 16) | (12 << 11) | (231 << 1);
const vd128Word = (v) => ((v & 0x60) >> 3) | ((v & 0x1f) << 21);
const lvx128Word = (v) => (4 << 26) | vd128Word(v) | (11 << 16) | (12 << 11) | (195 & 0x7f3);
const stvx128Word = (v) => (4 << 26) | vd128Word(v) | (11 << 16) | (12 << 11) | (451 & 0x7f3);

function detectorArena(builds) {
  const arena = new Uint8Array(0x1000);
  for (const { offset, words } of builds) {
    words.forEach((w, i) => writeWordBE(arena, offset + i * 4, w >>> 0));
  }
  return arena;
}
const detect = (arena) => detectSaveRestoreStubs(arena, { ranges: [[0, arena.length]], mask: GUEST_MASK });

test('save/restore stub detection — FPR/GPR stubs seed every mid-entry', () => {
  const savefpr = [];
  for (let f = 14; f <= 31; f++) savefpr.push(stfdWord(f, -0x90 + (f - 14) * 8));
  savefpr.push(BLR_WORD);
  const restgpr = [];
  for (let r = 14; r <= 31; r++) {
    // `ld rN, imm(r12)` with imm ascending by 8 — detector only cares about
    // the uniform sequence, not the absolute origin.
    restgpr.push((58 << 26) | (r << 21) | (12 << 16) | ((((-0x98 + (r - 14) * 8) >> 2) & 0x3fff) << 2));
  }
  restgpr.push(BLR_WORD);
  const arena = detectorArena([{ offset: 0x000, words: savefpr }, { offset: 0x080, words: restgpr }]);
  const res = detect(arena);
  assert.equal(res.stubs.length, 2);
  assert.equal(res.vmx.length, 0);
  assert.equal(res.seedAddresses.length, 36); // 18 FPR saves + 18 GPR restores
  assert.deepEqual(res.seedAddresses.slice(0, 3), [0, 4, 8]);
  // every mid-entry is seeded: caller can target any __restgprlr_N directly
  assert.deepEqual(res.seedAddresses.slice(18), Array.from({ length: 18 }, (_, i) => 0x80 + i * 4));
  assert.equal(res.dropped, 0);
});

test('save/restore stub detection — VMX64 stubs seed pairs and describe synthesis', () => {
  const savevmx = [];
  for (let k = 0; k < 64; k++) { savevmx.push(liR11(-0x400 + k * 0x10), stvx128Word(64 + k)); }
  savevmx.push(BLR_WORD);
  const restvmx = [];
  for (let k = 0; k < 64; k++) { restvmx.push(liR11(-0x400 + k * 0x10), lvx128Word(64 + k)); }
  restvmx.push(BLR_WORD);
  const arena = detectorArena([{ offset: 0x000, words: savevmx }, { offset: 0x220, words: restvmx }]);
  const res = detect(arena);
  assert.equal(res.vmx.length, 2);
  assert.equal(res.seedAddresses.length, 128);
  assert.equal(res.vmxStubs.size, 128);
  assert.equal(res.dropped, 0);
  const entry = res.vmxStubs.get(0x10); // __savevmx_66 == pair k=2
  assert.deepEqual({ ...entry, simmBase: entry.simmBase }, { save: true, firstReg: 66, lastReg: 127, simmBase: -0x400, famFirst: 64 });
  const lastSave = res.vmxStubs.get(0x1f8); // pair k=63
  assert.equal(lastSave.firstReg, 127);
  assert.equal(lastSave.save, true);
  const restEntry = res.vmxStubs.get(0x220 + 0x10);
  assert.equal(restEntry.save, false);
});

test('save/restore stub detection — VMX14 stubs (opcode 31 lvx/stvx) detected', () => {
  const savevmx = [];
  for (let k = 0; k < 18; k++) { savevmx.push(liR11(-0x120 + k * 0x10), stvxWord(14 + k)); }
  savevmx.push(BLR_WORD);
  const arena = detectorArena([{ offset: 0x000, words: savevmx }]);
  const res = detect(arena);
  assert.equal(res.vmx.length, 1);
  assert.equal(res.seedAddresses.length, 18);
  const e = res.vmxStubs.get(0x08); // __savevmx_15 == pair k=1
  assert.deepEqual({ ...e }, { save: true, firstReg: 15, lastReg: 31, simmBase: -0x120, famFirst: 14 });
});

test('save/restore stub detection — broken stubs are dropped, never misrun', () => {
  // FPR sequence missing its blr terminator
  const noBlr = [];
  for (let f = 14; f <= 31; f++) noBlr.push(stfdWord(f, -0x90 + (f - 14) * 8));
  noBlr.push(0x00000000);
  // VMX pair sequence with one corrupted vector op (wrong opcode)
  const brokenVmx = [];
  for (let k = 0; k < 64; k++) { brokenVmx.push(liR11(-0x400 + k * 0x10), k === 7 ? 0x60000000 : stvx128Word(64 + k)); }
  brokenVmx.push(BLR_WORD);
  const arena = detectorArena([{ offset: 0x000, words: noBlr }, { offset: 0x100, words: brokenVmx }]);
  const res = detect(arena);
  assert.equal(res.stubs.length, 0);
  assert.equal(res.vmx.length, 0);
  assert.equal(res.seedAddresses.length, 0);
  assert.ok(res.dropped >= 1);
});

// ---------- VMX stub synthesis, end to end ----------

test('VMX save/restore stubs are synthesized — guest runs past them', async () => {
  const { xex } = buildVmxStubXex();
  const built = buildImage('xex2', null, xex);
  assert.equal(built.vmxStubs.size, 128);
  // the mid-entry descriptors are what the bctr dispatch resolves to
  assert.ok(built.vmxStubs.has(VMX_SAVE_66 & GUEST_MASK));
  assert.ok(built.vmxStubs.has(VMX_REST_66 & GUEST_MASK));
  assert.ok(built.functionStarts.includes(VMX_SAVE_66 & GUEST_MASK));
  let out = '';
  const res = await runImage(built, { onConsole: (s) => { out += s; } });
  assert.equal(out, VMX_PASS_OUTPUT); // sentinels intact, slots zeroed
  assert.equal(res.exitCode, 0);
  assert.equal(res.stats.vmxStubs, 128);
  assert.equal(typeof res.exports['fn_' + (VMX_SAVE_66 & GUEST_MASK).toString(16)], 'function');
});

test('without VMX stub synthesis the stub traps loudly', async () => {
  const { xex } = buildVmxStubXex();
  const built = buildImage('xex2', null, xex);
  // Pre-synthesis world: no vmxStubs and no seeded stub fragments — the bctr
  // finds no fn_<savevmx> export and must trap rather than misrun.
  const strippedSeeds = built.functionStarts.filter((a) => !built.vmxStubs.has(a));
  await assert.rejects(
    () => runImage({ arena: built.arena, entry: built.entry, hleCalls: built.hleCalls, functionStarts: strippedSeeds, switchTables: new Map(), vmxStubs: new Map() }, { onConsole: () => {} }),
    (e) => e instanceof RecompUnimpl || /unimplemented/i.test(String(e && e.message)),
  );
});

test('FPR save/restore stubs lift and round-trip (mid-entry, lfd/stfd)', async () => {
  const { xex } = buildFprStubXex();
  const built = buildImage('xex2', null, xex);
  assert.ok(built.functionStarts.includes(FPR_SAVE_15 & GUEST_MASK));
  let out = '';
  const res = await runImage(built, { onConsole: (s) => { out += s; } });
  assert.equal(out, FPR_PASS_OUTPUT);
  assert.equal(res.exitCode, 0);
});
