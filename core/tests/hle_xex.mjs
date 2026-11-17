/**
 * hle_xex.mjs — builds a synthetic but fully valid XEX2 image for tests.
 *
 * The guest program (assembled at 0x82000000):
 *   - prints "hello from XEX HLE - no firmware, no BIOS\n" through DbgPrint
 *   - prints "answer=42\n" through DbgPrint with a %d argument
 *   - stores to the MMIO exit door (0x00F00FFC) and halts
 *
 * The image is PE-shaped ('MZ' at 0) because the loader (like xenia) sniffs
 * the retail/devkit key by checking for the PE signature after decryption.
 *
 * buildSyntheticXex() variants exercise:
 *   - BASIC compression + AES-128-CBC (zero IV), public retail XEX2 key
 *   - unencrypted BASIC
 *   - uncompressed ("none") + AES-128-CBC
 *   - uncompressed, unencrypted
 *   - trailing implicit-zero span after the used image bytes
 *   - LZX rejection, wrong-key rejection
 */
import { assemble } from '../../public/js/ppc_asm.js';
import { aesEcbEncrypt, aesCbcEncrypt } from '../../public/js/recomp/aes.js';

export const RETAIL_KEY = new Uint8Array([
  0x20, 0xb1, 0x85, 0xa5, 0x9d, 0x28, 0xfd, 0xc3,
  0x40, 0x58, 0x3f, 0xbb, 0x08, 0x96, 0xbf, 0x91,
]);

// The per-image session key the fixture encrypts with. The XEX wraps it with
// the public retail key in the security info — exactly like a real retail XEX.
export const TEST_SESSION_KEY = new Uint8Array([
  0xde, 0xad, 0xbe, 0xef, 0x01, 0x23, 0x45, 0x67,
  0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98,
]);

// xenia's devkit XEX2 wrapping key is 16 zero bytes (mirrored here for devkit fixtures).
export const DEVKIT_KEY = new Uint8Array(16);

export const IMAGE_BASE = 0x82000000;
export const ENTRY = 0x82000200;
export const THUNK_ADDR = 0x82001000;      // guest address of the DbgPrint thunk record
export const HELLO = 'hello from XEX HLE - no firmware, no BIOS\n';
export const ANSWER = 'answer=42\n';
export const EXPECTED_OUTPUT = HELLO + ANSWER;

// Build the guest source. `importOrdinals` lists the ordinals of the thunks the
// guest bl's to, in call order; each thunk record is 4 bytes at 0x82001000+i*4.
// The default (no args) is the canonical DbgPrint-only Hello image.
export function guestSource({ importOrdinals = [0x0003, 0x0003], calls = ['fmt1', 'fmt2'] } = {}) {
  const parts = [
    '.org 0x82000000',
    "    .byte 0x4D, 0x5A            # 'MZ' — the PE signature the loader sniffs for",
    '    .space 0x1FE',
  ];
  calls.forEach((c, i) => {
    if (c === 'fmt1') {
      parts.push('    lis   r3, fmt1@ha', '    addi  r3, r3, fmt1@l');
    } else {
      parts.push('    lis   r3, fmt2@ha', '    addi  r3, r3, fmt2@l', '    li    r4, 42');
    }
    parts.push(`    bl    thunk_${i}`);
  });
  parts.push(
    '    li    r9, 0',
    '    lis   r10, 0x00f0',
    '    ori   r10, r10, 0x0ffc',
    '    stw   r9, 0(r10)            # exit door',
    'spin:',
    '    b     spin',
    'fmt1:',
    `    .asciiz "${HELLO.replace('\n', '\\n')}"`,
    'fmt2:',
    `    .asciiz "${ANSWER.replace('\n', '\\n')}"`,
    '    .org 0x82001000',
  );
  for (let i = 0; i < importOrdinals.length; i++) {
    parts.push(`thunk_${i}:`);
    // type=1 (thunk), reserved 0x00, ordinal 0x000X
    parts.push(`    .word 0x${((0x01000000 | (importOrdinals[i] & 0xffff)) >>> 0).toString(16).padStart(8, '0')}`);
  }
  return parts.join('\n');
}

export function assembleGuest(opts) {
  const { bytes } = assemble(guestSource(opts));
  return bytes;
}

function u32be(v) {
  return new Uint8Array([(v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff]);
}
function u16be(v) {
  return new Uint8Array([(v >>> 8) & 0xff, v & 0xff]);
}

function buildHeader({ imageSize, aesKeyWrapped, blocks, compressionType, encryptionType, importLibrariesOffset, fileFormatInfoOffset, entry, base, importAddrs = [THUNK_ADDR] }) {
  // Fixed header layout: opt dir at 0x18 (5 entries), opt data at 0x100/0x120,
  // security info at 0x180, header ends at 0x320.
  const buf = new Uint8Array(0x320);
  const view = new DataView(buf.buffer);
  const put = (off, bytes) => buf.set(bytes, off);
  put(0, [0x58, 0x45, 0x58, 0x32]);           // 'XEX2'
  view.setUint32(4, 0x00000080, false);       // module_flags
  view.setUint32(8, buf.length, false);       // header_size
  view.setUint32(12, 0, false);               // reserved
  view.setUint32(16, 0x180, false);           // security_offset
  view.setUint32(20, 5, false);               // header_count
  const opts = [
    [0x00010100, entry, false],               // ENTRY_POINT (inline)
    [0x00010201, base, false],                // IMAGE_BASE_ADDRESS (inline)
    [0x000003ff, fileFormatInfoOffset, true], // FILE_FORMAT_INFO (offset)
    [0x000103ff, importLibrariesOffset, true],// IMPORT_LIBRARIES (offset)
    [0x00020200, 0x00100000, false],          // DEFAULT_STACK_SIZE (inline)
  ];
  opts.forEach(([key, value], i) => {
    view.setUint32(0x18 + i * 8, key >>> 0, false);
    view.setUint32(0x18 + i * 8 + 4, value >>> 0, false);
  });

  // FILE_FORMAT_INFO
  const infoSize = 8 + blocks.length * 8;
  let p = fileFormatInfoOffset;
  view.setUint32(p, infoSize, false);
  view.setUint16(p + 4, encryptionType, false);
  view.setUint16(p + 6, compressionType, false);
  p += 8;
  for (const b of blocks) {
    view.setUint32(p, b.dataSize >>> 0, false);
    view.setUint32(p + 4, b.zeroSize >>> 0, false);
    p += 8;
  }

  // IMPORT_LIBRARIES: one library ('xboxkrnl.xex'), one thunk import.
  const libName = 'xboxkrnl.xex\0';
  const tableBytes = Math.ceil(libName.length / 4) * 4;
  const stringTable = new Uint8Array(tableBytes);
  stringTable.set(new TextEncoder().encode(libName));
  const lib = new Uint8Array(0x28 + importAddrs.length * 4);
  const lv = new DataView(lib.buffer);
  lv.setUint32(0x0, lib.length, false);       // size
  // next_import_digest[0x14] at 0x4 — zeros
  lv.setUint32(0x18, 0x10000, false);         // id
  lv.setUint32(0x1c, 0x00000001, false);      // version
  lv.setUint32(0x20, 0x00000001, false);      // version_min
  lv.setUint16(0x24, 0, false);               // name_index
  lv.setUint16(0x26, importAddrs.length, false);
  importAddrs.forEach((a, i) => lv.setUint32(0x28 + i * 4, a >>> 0, false));
  const totalSize = 0x0c + 0x10 + lib.length;
  p = importLibrariesOffset;
  view.setUint32(p, totalSize, false);
  view.setUint32(p + 4, tableBytes, false);   // string_table.size
  view.setUint32(p + 8, 1, false);            // string_table.count
  buf.set(stringTable, p + 0x0c);
  buf.set(lib, p + 0x0c + tableBytes);

  // Security info @ 0x180
  const sec = 0x180;
  view.setUint32(sec + 0x0, 0x19c, false);    // security header_size
  view.setUint32(sec + 0x4, imageSize, false);// image_size
  // rsa_signature[0x100] at +0x8 — zeros (never verified)
  view.setUint32(sec + 0x108, 0x198, false);  // info_size
  view.setUint32(sec + 0x10c, 0x10000000, false); // image_flags: XEX_IMAGE_PAGE_SIZE_4KB
  view.setUint32(sec + 0x110, base >>> 0, false); // load_address
  buf.set(aesKeyWrapped, sec + 0x150);        // aes_key @ 0x150
  view.setUint32(sec + 0x180, 1, false);      // page_descriptor_count
  const pageCount = Math.ceil(imageSize / 0x1000);
  view.setUint32(sec + 0x184, ((pageCount << 4) | 1) >>> 0, false); // CODE pages
  // remaining descriptor digest bytes are zero
  return buf;
}

function wrapSessionKey(key, devkit) {
  // Devkit images are wrapped with the devkit key (all-zero) instead of the public retail key.
  return aesEcbEncrypt(devkit ? DEVKIT_KEY : RETAIL_KEY, key);
}

/**
 * @param {object} [opts]
 * @param {'basic'|'none'} [opts.compression]
 * @param {'normal'|'none'} [opts.encryption]
 * @param {number} [opts.trailingZero] implicit zero bytes appended via BASIC zero_size
 * @param {boolean} [opts.wrongKey] XOR the session key so retail-key unwrapping mismatches
 */
export function buildSyntheticXex({ compression = 'basic', encryption = 'normal', trailingZero = 0, wrongKey = false, devkit = false, importOrdinals } = {}) {
  const image = assembleGuest(importOrdinals ? { importOrdinals, calls: importOrdinals.map((_, i) => (i % 2 === 0 ? 'fmt1' : 'fmt2')) } : undefined);
  const importAddrs = (importOrdinals ?? [0x0003, 0x0003]).map((_, i) => THUNK_ADDR + i * 4);
  // `wrongKey` wraps a *different* key than the one the body is encrypted
  // with — a tampered/foreign image that neither public key can rescue.
  const sessionKey = TEST_SESSION_KEY;
  const wrappedKey = wrongKey ? new Uint8Array(16).fill(0x55) : TEST_SESSION_KEY;
  const aesKeyWrapped = wrapSessionKey(wrappedKey, devkit);
  const imageSize = image.length + trailingZero;

  if (compression === 'none') {
    if (encryption === 'normal') throw new Error('test bug: compression=none + encryption=normal must go through the uncompressed path — supported, flip it deliberately');
    const header = buildHeader({
      imageSize: image.length, aesKeyWrapped, blocks: [], compressionType: 0, encryptionType: 0,
      importLibrariesOffset: 0x120, fileFormatInfoOffset: 0x100, entry: ENTRY, base: IMAGE_BASE, importAddrs,
    });
    return { xex: concat(header, image), image, sessionKey };
  }

  // BASIC
  const blocks = [{ dataSize: image.length, zeroSize: trailingZero }];
  if (encryption === 'normal') {
    // xenia: BASIC block sizes stay 16-aligned when encrypted; pad the image into a 16-aligned buffer
    const aligned = Math.ceil(image.length / 16) * 16;
    if (aligned !== image.length) {
      const padded = new Uint8Array(aligned);
      padded.set(image);
      blocks[0] = { dataSize: aligned, zeroSize: trailingZero };
      const encrypted = aesCbcEncrypt(sessionKey, padded);
      const header = buildHeader({
        imageSize: aligned + trailingZero, aesKeyWrapped, blocks, compressionType: 1, encryptionType: 1,
        importLibrariesOffset: 0x120, fileFormatInfoOffset: 0x100, entry: ENTRY, base: IMAGE_BASE, importAddrs,
      });
      return { xex: concat(header, encrypted), image, sessionKey };
    }
    const encrypted = aesCbcEncrypt(sessionKey, image);
    const header = buildHeader({
      imageSize, aesKeyWrapped, blocks, compressionType: 1, encryptionType: 1,
      importLibrariesOffset: 0x120, fileFormatInfoOffset: 0x100, entry: ENTRY, base: IMAGE_BASE, importAddrs,
    });
    return { xex: concat(header, encrypted), image, sessionKey };
  }
  const header = buildHeader({
    imageSize, aesKeyWrapped, blocks, compressionType: 1, encryptionType: 0,
    importLibrariesOffset: 0x120, fileFormatInfoOffset: 0x100, entry: ENTRY, base: IMAGE_BASE, importAddrs,
  });
  return { xex: concat(header, image), image, sessionKey };
}

/** compression=none + encryption=normal helper (uncompressed but encrypted image) */
export function buildEncryptedUncompressedXex({ importOrdinals } = {}) {
  const image = assembleGuest(importOrdinals ? { importOrdinals, calls: importOrdinals.map((_, i) => (i % 2 === 0 ? 'fmt1' : 'fmt2')) } : undefined);
  const importAddrs = (importOrdinals ?? [0x0003, 0x0003]).map((_, i) => THUNK_ADDR + i * 4);
  const aligned = Math.ceil(image.length / 16) * 16;
  const padded = new Uint8Array(aligned);
  padded.set(image);
  const encrypted = aesCbcEncrypt(TEST_SESSION_KEY, padded);
  const header = buildHeader({
    imageSize: aligned, aesKeyWrapped: wrapSessionKey(TEST_SESSION_KEY), blocks: [], // consistent wrap: valid image
    compressionType: 0, encryptionType: 1,
    importLibrariesOffset: 0x120, fileFormatInfoOffset: 0x100, entry: ENTRY, base: IMAGE_BASE, importAddrs,
  });
  return { xex: concat(header, encrypted), image, sessionKey: TEST_SESSION_KEY };
}

function concat(a, b) {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}
