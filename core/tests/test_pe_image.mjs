/**
 * pe_image.js unit tests — synthetic PE images built by hand (no assembler):
 * section table parsing, .pdata (IMAGE_CE_RUNTIME_FUNCTION) decoding, and
 * malformed-input handling. The end-to-end XEX2 path is covered in
 * test_hle_xex.mjs.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  parsePeSections,
  pdataFunctionStarts,
  PeImageError,
} from '../../public/js/recomp/pe_image.js';

const MASK = 0x03ffffff;

function buildPeImage(sectionDefs) {
  const eLfanew = 0x40;
  const optSize = 0xE0;
  const fileHeader = eLfanew + 4;
  const sectionTable = fileHeader + 20 + optSize;
  const numSections = sectionDefs.length;
  const headerEnd = sectionTable + numSections * 40;
  const size = Math.max(
    headerEnd,
    ...sectionDefs.map((s) => s.rva + s.data.length),
  );
  const img = new Uint8Array(size);
  const dv = new DataView(img.buffer);
  img[0] = 0x4d; img[1] = 0x5a;                    // 'MZ'
  dv.setUint32(0x3c, eLfanew, true);
  dv.setUint32(eLfanew, 0x00004550, true);          // 'PE\0\0'
  dv.setUint16(fileHeader, 0x01f2, true);           // Machine (POWERPCFP)
  dv.setUint16(fileHeader + 2, numSections, true);
  dv.setUint32(fileHeader + 4, 0, true);            // TimeDateStamp
  dv.setUint16(fileHeader + 16, optSize, true);     // SizeOfOptionalHeader
  dv.setUint16(fileHeader + 18, 0x0102, true);      // Characteristics
  dv.setUint16(sectionTable - optSize, 0x010b, true); // PE32 magic
  sectionDefs.forEach((s, i) => {
    const off = sectionTable + i * 40;
    for (let j = 0; j < s.name.length && j < 8; j++) img[off + j] = s.name.charCodeAt(j);
    dv.setUint32(off + 8, s.data.length, true);     // VirtualSize
    dv.setUint32(off + 12, s.rva, true);            // VirtualAddress
    dv.setUint32(off + 16, s.data.length, true);    // SizeOfRawData
    dv.setUint32(off + 20, s.rva, true);            // PointerToRawData (contiguous)
    dv.setUint32(off + 36, s.characteristics >>> 0, true); // Characteristics
    img.set(s.data, s.rva);
  });
  return img;
}

function pdataRecords(records) {
  const out = new Uint8Array(records.length * 8);
  const dv = new DataView(out.buffer);
  records.forEach(([begin, data], i) => {
    dv.setUint32(i * 8, begin >>> 0, false);        // big-endian
    dv.setUint32(i * 8 + 4, data >>> 0, false);
  });
  return out;
}

// Data word layout: PrologLength:8 | FunctionLength:22 | ThirtyTwoBit:1 | ExceptionFlag:1
const dataWord = (functionLength, { prolog = 0, thirtyTwoBit = 1, exceptionFlag = 0 } = {}) =>
  ((prolog & 0xff) << 24) | ((functionLength & 0x3fffff) << 2) | ((thirtyTwoBit & 1) << 1) | (exceptionFlag & 1);

test('parsePeSections reads names, RVAs and CODE characteristic', () => {
  const img = buildPeImage([
    { name: '.text', rva: 0x200, data: new Uint8Array(0x10), characteristics: 0x60000020 },
    { name: '.pdata', rva: 0x400, data: new Uint8Array(8), characteristics: 0x40000040 },
  ]);
  const { machine, sections } = parsePeSections(img);
  assert.equal(machine, 0x01f2);
  assert.deepEqual(sections.map((s) => s.name), ['.text', '.pdata']);
  assert.deepEqual(sections.map((s) => s.virtualAddress), [0x200, 0x400]);
  assert.deepEqual(sections.map((s) => s.isCode), [true, false]);
});

test('pdataFunctionStarts decodes big-endian records, sizes and the HLE-thunk exclusion input', () => {
  const img = buildPeImage([
    { name: '.text', rva: 0x200, data: new Uint8Array(0x200), characteristics: 0x60000020 },
    {
      name: '.pdata', rva: 0x400, characteristics: 0x40000040,
      data: pdataRecords([
        [0x82000200, dataWord(0x40)],   // entry, 0x100 bytes
        [0x82000300, dataWord(4)],      // helper, 0x10 bytes
        [0x00000000, dataWord(1)],      // zero BeginAddress — skipped
        [0x82000300, dataWord(4)],      // duplicate — deduplicated
      ]),
    },
  ]);
  const fns = pdataFunctionStarts(img, MASK);
  assert.deepEqual(fns, [
    { addr: 0x82000200 & MASK, size: 0x100 },
    { addr: 0x82000300 & MASK, size: 0x10 },
  ]);
});

test('missing .pdata yields an empty seed list (not an error)', () => {
  const img = buildPeImage([
    { name: '.text', rva: 0x200, data: new Uint8Array(4), characteristics: 0x60000020 },
  ]);
  assert.deepEqual(pdataFunctionStarts(img, MASK), []);
});

test('.pdata with a size that is not a multiple of 8 throws PeImageError', () => {
  const img = buildPeImage([
    { name: '.pdata', rva: 0x400, data: new Uint8Array(12), characteristics: 0x40000040 },
  ]);
  assert.throws(() => pdataFunctionStarts(img, MASK), (e) => e instanceof PeImageError && /multiple of 8/.test(e.message));
});

test('.pdata section pointing outside the image throws PeImageError', () => {
  // Point VirtualAddress at the end of the image so start + size overruns.
  const img = buildPeImage([
    { name: '.pdata', rva: 0x400, data: new Uint8Array(8), characteristics: 0x40000040 },
  ]);
  // Patch the section header's VirtualSize to something huge (offset of .pdata header = table + 40).
  const fileHeader = 0x44;
  const sectionTable = fileHeader + 20 + 0xE0;
  const dv = new DataView(img.buffer);
  dv.setUint32(sectionTable + 8, 0x1000, true); // VirtualSize
  assert.throws(() => pdataFunctionStarts(img, MASK), (e) => e instanceof PeImageError && /outside the image/.test(e.message));
});

test('corrupt PE containers throw PeImageError, not garbage', () => {
  assert.throws(() => parsePeSections(new Uint8Array(8)), PeImageError);          // too small / no MZ
  const bad = buildPeImage([]);
  bad[0] = 0x00;
  assert.throws(() => parsePeSections(bad), PeImageError);                        // no MZ
  const badSig = buildPeImage([]);
  new DataView(badSig.buffer).setUint32(0x40, 0xdeadbeef, true);
  assert.throws(() => parsePeSections(badSig), PeImageError);                     // no PE signature
  const badOff = buildPeImage([]);
  new DataView(badOff.buffer).setUint32(0x3c, 0x00ffff00, true);
  assert.throws(() => parsePeSections(badOff), PeImageError);                     // e_lfanew outside image
});
