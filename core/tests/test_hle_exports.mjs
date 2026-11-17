/**
 * Direct HLE host tests — exercise the implemented XenonHLE exports against a
 * fresh 64 MiB WebAssembly.Memory with registers staged in the HLE scratch
 * area, exactly as the recompiler stages them before an import call.
 *
 * These complement the end-to-end synthetic-XEX2 tests in test_hle_xex.mjs:
 * here we assert the precise guest-visible contract of each export (out
 * parameter layout, status codes, heap behaviour) without executing PPC.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../../public/js/recomp/hle.js';       // side effect: globalThis.XenonHLE
import '../../public/js/recomp/hle_tables.js'; // side effect: globalThis.XenonHLETables

const HLE_CTX_ADDR = 0x00F11000;
const GUEST_MEM = 1024; // 64 MiB, same as the recompiled module
const X_STATUS = {
  SUCCESS: 0n,
  NOT_IMPLEMENTED: 0xc0000002n,
  INVALID_PARAMETER: 0xc000000dn,
  NO_MEMORY: 0xc0000017n,
  ACCESS_DENIED: 0xc0000022n,
  OBJECT_NAME_NOT_FOUND: 0xc0000034n,
  INVALID_PAGE_PROTECTION: 0xc0000045n,
  MEMORY_NOT_ALLOCATED: 0xc00000a0n,
};
const X_MEM = { COMMIT: 0x1000, RESERVE: 0x2000, DECOMMIT: 0x4000, RELEASE: 0x8000, TOP_DOWN: 0x00100000, LARGE_PAGES: 0x20000000 };
const X_PAGE = { READONLY: 0x02, READWRITE: 0x04, EXECUTE_READWRITE: 0x40 };

function makeHost() {
  const memory = new WebAssembly.Memory({ initial: GUEST_MEM, maximum: GUEST_MEM });
  const logs = [];
  const host = globalThis.XenonHLE.createHleHost({ onLog: (s) => logs.push(String(s)) });
  host.attachMemory(memory);
  return { memory, host, logs };
}

function setArgs(memory, ...args) {
  const view = new DataView(memory.buffer);
  for (let i = 0; i < 32; i++) {
    const v = i >= 3 && i < 3 + args.length ? BigInt(args[i - 3]) & 0xffffffffffffffffn : 0n;
    view.setBigUint64(HLE_CTX_ADDR + i * 8, v, false);
  }
}

function bytes(memory) { return new Uint8Array(memory.buffer); }
function viewOf(memory) { return new DataView(memory.buffer); }
function readU32(memory, addr) { return viewOf(memory).getUint32(addr, false) >>> 0; }
function writeU32(memory, addr, v) { viewOf(memory).setUint32(addr, v >>> 0, false); }
function writeBytes(memory, addr, src) { bytes(memory).set(src, addr); }
function writeCString(memory, addr, s) { const b = new Uint8Array([...s].map((c) => c.charCodeAt(0)).concat([0])); writeBytes(memory, addr, b); }
function writeCStringW(memory, addr, s) {
  const buf = [];
  for (const ch of s) buf.push(0, ch.charCodeAt(0) & 0xff);
  buf.push(0, 0);
  writeBytes(memory, addr, buf);
}

test('RtlCompareMemory counts every equal byte (not stop-at-first-mismatch)', () => {
  const { memory, host } = makeHost();
  writeBytes(memory, 0x01000000, [1, 2, 3, 9, 5, 6, 7]);
  writeBytes(memory, 0x01001000, [1, 2, 3, 8, 5, 0, 7]);
  setArgs(memory, 0x01000000n, 0x01001000n, 7n);
  assert.equal(host.call('xboxkrnl', 0x011a), 5n);
});

test('RtlCompareMemoryUlong compares BE u32s against a pattern until first mismatch', () => {
  const { memory, host } = makeHost();
  // source: DE AD BE EF, 00 00 00 00, CA FE BA BE, FF ... — pattern 0xDEADBEEF
  writeBytes(memory, 0x01000000, [0xde, 0xad, 0xbe, 0xef, 0x00, 0x00, 0x00, 0x00, 0xde, 0xad, 0xbe, 0xef]);
  setArgs(memory, 0x01000000n, 12n, 0xdeadbeefn);
  assert.equal(host.call('xboxkrnl', 0x011b), 4n, 'stops after the first mismatching ULONG');
  writeBytes(memory, 0x01000000, [0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef]);
  setArgs(memory, 0x01000000n, 8n, 0xdeadbeefn);
  assert.equal(host.call('xboxkrnl', 0x011b), 8n);
  // length smaller than one ULONG compares nothing
  setArgs(memory, 0x01000000n, 3n, 0xdeadbeefn);
  assert.equal(host.call('xboxkrnl', 0x011b), 0n);
});

test('RtlInitializeCriticalSection packs xenia-compatible fields (28-byte struct)', () => {
  const { memory, host } = makeHost();
  const cs = 0x01000000;
  bytes(memory).fill(0xaa, cs, cs + 28);
  setArgs(memory, BigInt(cs));
  host.call('xboxkrnl', 0x012e);
  assert.equal(bytes(memory)[cs + 0x00], 0x01, 'type = EventSynchronizationObject');
  assert.equal(bytes(memory)[cs + 0x01], 0x00, 'absolute (spin) = 0');
  assert.equal(viewOf(memory).getInt32(cs + 0x10, false), -1, 'lock_count = -1');
  assert.equal(viewOf(memory).getInt32(cs + 0x14, false), 0, 'recursion_count = 0');
  assert.equal(readU32(memory, cs + 0x18), 0, 'owning_thread = 0');
});

test('RtlInitializeCriticalSectionAndSpinCount packs spin/256 and returns SUCCESS', () => {
  const { memory, host } = makeHost();
  const cs = 0x01000000;
  setArgs(memory, BigInt(cs), 1000n); // 1000 → (1000+255)>>8 = 4
  assert.equal(host.call('xboxkrnl', 0x012f), X_STATUS.SUCCESS);
  assert.equal(bytes(memory)[cs + 0x01], 4);
  const cs2 = 0x01001000;
  setArgs(memory, BigInt(cs2), 100000n); // caps at 255
  host.call('xboxkrnl', 0x012f);
  assert.equal(bytes(memory)[cs2 + 0x01], 255);
});

test('critical section: enter, recurse, leave; unbalanced leave traps loudly', () => {
  const { memory, host } = makeHost();
  const cs = 0x01000000;
  setArgs(memory, BigInt(cs));
  host.call('xboxkrnl', 0x012e);
  setArgs(memory, BigInt(cs));
  host.call('xboxkrnl', 0x0125); // enter
  assert.equal(readU32(memory, cs + 0x18) !== 0, true, 'owned after enter');
  assert.equal(viewOf(memory).getInt32(cs + 0x14, false), 1, 'recursion 1');
  setArgs(memory, BigInt(cs));
  host.call('xboxkrnl', 0x0125); // recurse
  assert.equal(viewOf(memory).getInt32(cs + 0x14, false), 2, 'recursion 2');
  setArgs(memory, BigInt(cs));
  host.call('xboxkrnl', 0x0130); // leave (still owned)
  assert.equal(viewOf(memory).getInt32(cs + 0x14, false), 1);
  setArgs(memory, BigInt(cs));
  host.call('xboxkrnl', 0x0130); // leave (released)
  assert.equal(readU32(memory, cs + 0x18), 0, 'released after final leave');
  assert.equal(viewOf(memory).getInt32(cs + 0x10, false), -1, 'lock_count back to -1');
  // unbalanced leave must trap, not corrupt
  setArgs(memory, BigInt(cs));
  assert.throws(() => host.call('xboxkrnl', 0x0130), /unbalanced/i);
});

test('RtlTryEnterCriticalSection succeeds on free and owned-by-self, never blocks', () => {
  const { memory, host } = makeHost();
  const cs = 0x01000000;
  setArgs(memory, BigInt(cs));
  host.call('xboxkrnl', 0x012e);
  setArgs(memory, BigInt(cs));
  assert.equal(host.call('xboxkrnl', 0x0141), 1n, 'free lock acquired');
  setArgs(memory, BigInt(cs));
  assert.equal(host.call('xboxkrnl', 0x0141), 1n, 'owned by self → recursion');
  assert.equal(viewOf(memory).getInt32(cs + 0x14, false), 2);
});

test('RtlInitAnsiString writes { u16 length, u16 max, u32 ptr }; null source resets', () => {
  const { memory, host } = makeHost();
  const dst = 0x01000000, src = 0x01001000;
  writeCString(memory, src, 'hello xenos');
  setArgs(memory, BigInt(dst), BigInt(src));
  host.call('xboxkrnl', 0x012c);
  assert.equal(viewOf(memory).getUint16(dst, false), 11);
  assert.equal(viewOf(memory).getUint16(dst + 2, false), 12);
  assert.equal(readU32(memory, dst + 4), src);
  setArgs(memory, BigInt(dst), 0n);
  host.call('xboxkrnl', 0x012c);
  assert.equal(viewOf(memory).getUint16(dst, false), 0);
  assert.equal(viewOf(memory).getUint16(dst + 2, false), 0);
  assert.equal(readU32(memory, dst + 4), 0);
});

test('RtlInitUnicodeString writes byte counts and returns the destination address', () => {
  const { memory, host } = makeHost();
  const dst = 0x01000000, src = 0x01001000;
  writeCStringW(memory, src, 'xenon');
  setArgs(memory, BigInt(dst), BigInt(src));
  const ret = host.call('xboxkrnl', 0x012d);
  assert.equal(ret, BigInt(dst));
  assert.equal(viewOf(memory).getUint16(dst, false), 10, 'length in bytes');
  assert.equal(viewOf(memory).getUint16(dst + 2, false), 12);
  assert.equal(readU32(memory, dst + 4), src);
});

test('NtAllocateVirtualMemory allocates 64K-aligned zeroed pages; second alloc does not overlap', () => {
  const { memory, host } = makeHost();
  const baseSlot = 0x01000000, sizeSlot = 0x01000004;
  writeU32(memory, baseSlot, 0);
  writeU32(memory, sizeSlot, 0x2345); // unrounded, exercises alignment
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.COMMIT | X_MEM.RESERVE), BigInt(X_PAGE.READWRITE), 0n);
  assert.equal(host.call('xboxkrnl', 0x00cc), X_STATUS.SUCCESS);
  const base1 = readU32(memory, baseSlot);
  const size1 = readU32(memory, sizeSlot);
  assert.equal(base1 % 0x10000, 0, 'aligned to 64 KiB');
  assert.equal(size1, 0x3000, 'rounded up to 4K pages');
  assert.ok(base1 >= 0x00100000 && base1 + size1 <= 0x00f00000, 'inside the managed guest heap');
  assert.ok(bytes(memory).subarray(base1, base1 + size1).every((b) => b === 0), 'committed pages are zeroed');
  // allocate another, ensure no overlap
  writeU32(memory, baseSlot, 0);
  writeU32(memory, sizeSlot, 0x1000);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.COMMIT), BigInt(X_PAGE.READWRITE), 0n);
  host.call('xboxkrnl', 0x00cc);
  const base2 = readU32(memory, baseSlot);
  assert.ok(base2 >= base1 + size1 || base1 >= base2 + 0x1000, 'no overlap between allocations');
  // write to it, free it, allocate again — reuse works
  bytes(memory)[base1] = 0x5a;
  const rbase = 0x01000000;
  writeU32(memory, rbase, base1);
  writeU32(memory, sizeSlot, 0);
  setArgs(memory, BigInt(rbase), BigInt(sizeSlot), BigInt(X_MEM.RELEASE), 0x04n, 0n);
  assert.equal(host.call('xboxkrnl', 0x00dc), X_STATUS.SUCCESS);
  assert.equal(readU32(memory, sizeSlot), size1, 'RELEASE returns the region size');
});

test('NtAllocateVirtualMemory honours a requested base address and rejects nonsense', () => {
  const { memory, host } = makeHost();
  const baseSlot = 0x01000000, sizeSlot = 0x01000004;
  writeU32(memory, baseSlot, 0x00300000); // requested base inside the managed heap
  writeU32(memory, sizeSlot, 0x1000);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.COMMIT), BigInt(X_PAGE.READWRITE), 0n);
  assert.equal(host.call('xboxkrnl', 0x00cc), X_STATUS.SUCCESS);
  assert.equal(readU32(memory, baseSlot), 0x00300000);
  // fixed base outside the managed heap → INVALID_PARAMETER (xenia rejects non-guest-virtual heaps)
  writeU32(memory, baseSlot, 0x08000000);
  writeU32(memory, sizeSlot, 0x1000);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.COMMIT), BigInt(X_PAGE.READWRITE), 0n);
  assert.equal(host.call('xboxkrnl', 0x00cc), X_STATUS.INVALID_PARAMETER);
  // zero size → INVALID_PARAMETER
  writeU32(memory, baseSlot, 0);
  writeU32(memory, sizeSlot, 0);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.COMMIT), BigInt(X_PAGE.READWRITE), 0n);
  assert.equal(host.call('xboxkrnl', 0x00cc), X_STATUS.INVALID_PARAMETER);
  // no commit/reserve/reset bits → INVALID_PARAMETER
  writeU32(memory, baseSlot, 0);
  writeU32(memory, sizeSlot, 0x1000);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), 0n, BigInt(X_PAGE.READWRITE), 0n);
  assert.equal(host.call('xboxkrnl', 0x00cc), X_STATUS.INVALID_PARAMETER);
});

test('NtFreeVirtualMemory on an unknown base is rejected; double free is not silent', () => {
  const { memory, host } = makeHost();
  const baseSlot = 0x01000000, sizeSlot = 0x01000004;
  writeU32(memory, baseSlot, 0x00500000); // never allocated
  writeU32(memory, sizeSlot, 0x1000);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.RELEASE), 0x04n, 0n);
  assert.equal(host.call('xboxkrnl', 0x00dc), X_STATUS.INVALID_PARAMETER);
  // decommit zeroes in place but keeps the region allocated
  writeU32(memory, baseSlot, 0);
  writeU32(memory, sizeSlot, 0x1000);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.COMMIT), BigInt(X_PAGE.READWRITE), 0n);
  host.call('xboxkrnl', 0x00cc);
  const base1 = readU32(memory, baseSlot);
  bytes(memory)[base1] = 0xff;
  writeU32(memory, baseSlot, base1);
  writeU32(memory, sizeSlot, 0x1000);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.DECOMMIT), 0x04n, 0n);
  assert.equal(host.call('xboxkrnl', 0x00dc), X_STATUS.SUCCESS);
  assert.equal(bytes(memory)[base1], 0, 'decommitted pages are zeroed');
  // release then double-release
  writeU32(memory, baseSlot, base1);
  writeU32(memory, sizeSlot, 0);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.RELEASE), 0x04n, 0n);
  assert.equal(host.call('xboxkrnl', 0x00dc), X_STATUS.SUCCESS);
  writeU32(memory, baseSlot, base1);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.RELEASE), 0x04n, 0n);
  assert.equal(host.call('xboxkrnl', 0x00dc), X_STATUS.INVALID_PARAMETER);
});

test('NtProtectVirtualMemory changes protection of managed regions, rejects EXECUTE', () => {
  const { memory, host } = makeHost();
  const baseSlot = 0x01000000, sizeSlot = 0x01000004, oldSlot = 0x01000008;
  writeU32(memory, baseSlot, 0);
  writeU32(memory, sizeSlot, 0x1000);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_MEM.COMMIT), BigInt(X_PAGE.READWRITE), 0n);
  host.call('xboxkrnl', 0x00cc);
  const base1 = readU32(memory, baseSlot);
  writeU32(memory, baseSlot, base1);
  writeU32(memory, sizeSlot, 0x1000);
  writeU32(memory, oldSlot, 0);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_PAGE.READONLY), BigInt(oldSlot), 0n);
  assert.equal(host.call('xboxkrnl', 0x00e1), X_STATUS.SUCCESS);
  assert.equal(readU32(memory, oldSlot), X_PAGE.READWRITE, 'old protection reported back');
  // EXECUTE protection is refused, like xenia
  writeU32(memory, baseSlot, base1);
  writeU32(memory, sizeSlot, 0x1000);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_PAGE.EXECUTE_READWRITE), BigInt(oldSlot), 0n);
  assert.equal(host.call('xboxkrnl', 0x00e1), X_STATUS.INVALID_PAGE_PROTECTION);
  // region we do not own → ACCESS_DENIED
  writeU32(memory, baseSlot, 0x02000000); // outside the managed heap
  writeU32(memory, sizeSlot, 0x1000);
  setArgs(memory, BigInt(baseSlot), BigInt(sizeSlot), BigInt(X_PAGE.READONLY), BigInt(oldSlot), 0n);
  assert.equal(host.call('xboxkrnl', 0x00e1), X_STATUS.ACCESS_DENIED);
});

test('XexGetModuleHandle returns the synthetic main-module handle; name lookup refuses honestly', () => {
  const { memory, host, logs } = makeHost();
  const outSlot = 0x01000000, nameSlot = 0x01001000;
  writeCString(memory, nameSlot, 'xam.xex');
  setArgs(memory, 0n, BigInt(outSlot)); // NULL name → main module
  assert.equal(host.call('xboxkrnl', 0x0195), X_STATUS.SUCCESS);
  assert.notEqual(readU32(memory, outSlot), 0, 'handle written back');
  setArgs(memory, BigInt(nameSlot), BigInt(outSlot));
  assert.equal(host.call('xboxkrnl', 0x0195), X_STATUS.OBJECT_NAME_NOT_FOUND);
  assert.ok(logs.some((l) => /not implemented/i.test(l)), 'logs that name lookup is unsupported');
});

test('unimplemented exports still trap loudly with module!name and ordinal', () => {
  const { memory, host } = makeHost();
  setArgs(memory, 0n);
  assert.throws(() => host.call('xboxkrnl', 0x00bf), (e) => /unimplemented export/.test(e.message));
});
