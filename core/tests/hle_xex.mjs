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
  return buildSyntheticXexFromImage(image, { compression, encryption, trailingZero, wrongKey, devkit, importOrdinals });
}

/**
 * Wrap a pre-assembled guest image into a synthetic XEX2 — used by tests
 * whose guest program is hand-written rather than generated by guestSource().
 */
export function buildSyntheticXexFromImage(image, { compression = 'basic', encryption = 'normal', trailingZero = 0, wrongKey = false, devkit = false, importOrdinals = [0x0003, 0x0003] } = {}) {
  const importAddrs = importOrdinals.map((_, i) => THUNK_ADDR + i * 4);
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

/**
 * PDATA_* — a PE-shaped guest image with a real `.pdata`
 * (IMAGE_CE_RUNTIME_FUNCTION) section, mirroring what XenonRecomp parses:
 *
 *   - record 0: the entry function (baseline — control-flow discovery finds it too)
 *   - record 1: `helper_fn`, a function NOTHING in the image branches to —
 *     only visible through .pdata. This is the regression test for
 *     pdata-seeded fragment discovery.
 *   - record 2: the HLE import thunk — must NOT be exported as a wasm
 *     function (HLE imports resolve to host calls, not recompiled code).
 *
 * PE headers are little-endian (Windows PE convention); the code and the
 * .pdata records are big-endian PPC/guest data. The image is wrapped with
 * the public retail key like every other fixture in this file.
 */
export const PDATA_ENTRY_ADDR = 0x82000200;
export const PDATA_HELPER_ADDR = 0x82000300; // pdata-only, unreachable from entry
export const PDATA_THUNK_ADDR = THUNK_ADDR;  // 0x82001000
export const PDATA_HELPER_R3 = 0x2a;         // helper_fn leaves 42 in r3
export const PDATA_OUTPUT = 'pdata seeded run\n';

const PDATA_SOURCE = `
.org 0x82000000
.byte 0x4D, 0x5A            # 'MZ'
.space 0x3A                 # 0x02..0x3B
.byte 0x80, 0x00, 0x00, 0x00  # e_lfanew = 0x80 (LE)
.space 0x40                 # 0x40..0x7F
.byte 0x50, 0x45, 0x00, 0x00  # 'PE\\0\\0'
.byte 0xF2, 0x01            # Machine = POWERPCFP (0x01F2, LE)
.byte 0x02, 0x00            # NumberOfSections = 2
.byte 0x00, 0x00, 0x00, 0x00  # TimeDateStamp
.byte 0x00, 0x00, 0x00, 0x00  # PointerToSymbolTable
.byte 0x00, 0x00, 0x00, 0x00  # NumberOfSymbols
.byte 0xE0, 0x00            # SizeOfOptionalHeader = 0xE0
.byte 0x02, 0x01            # Characteristics = EXECUTABLE_IMAGE | 32BIT_MACHINE
# ---- optional header @ 0x98 (0xE0 bytes, little-endian) ----
.byte 0x0B, 0x01            # Magic = PE32
.byte 0x09, 0x00            # Linker 9.0
.byte 0x00, 0x02, 0x00, 0x00  # SizeOfCode = 0x200
.byte 0x00, 0x00, 0x00, 0x00  # SizeOfInitializedData
.byte 0x00, 0x00, 0x00, 0x00  # SizeOfUninitializedData
.byte 0x00, 0x02, 0x00, 0x00  # AddressOfEntryPoint = 0x200
.byte 0x00, 0x02, 0x00, 0x00  # BaseOfCode = 0x200
.byte 0x00, 0x04, 0x00, 0x00  # BaseOfData = 0x400
.byte 0x00, 0x00, 0x00, 0x82  # ImageBase = 0x82000000
.byte 0x00, 0x02, 0x00, 0x00  # SectionAlignment = 0x200
.byte 0x00, 0x02, 0x00, 0x00  # FileAlignment = 0x200
.byte 0x06, 0x00, 0x02, 0x00  # OS 6.2
.byte 0x00, 0x00, 0x00, 0x00  # Image 0.0
.byte 0x06, 0x00, 0x02, 0x00  # Subsystem 6.2
.byte 0x00, 0x00, 0x00, 0x00  # Win32VersionValue
.byte 0x00, 0x20, 0x00, 0x00  # SizeOfImage = 0x2000
.byte 0x00, 0x02, 0x00, 0x00  # SizeOfHeaders = 0x200
.byte 0x00, 0x00, 0x00, 0x00  # CheckSum
.byte 0x02, 0x00            # Subsystem = WINDOWS_GUI
.byte 0x00, 0x00            # DllCharacteristics
.byte 0x00, 0x00, 0x10, 0x00  # SizeOfStackReserve = 0x00100000
.byte 0x00, 0x10, 0x00, 0x00  # SizeOfStackCommit = 0x00001000
.byte 0x00, 0x00, 0x10, 0x00  # SizeOfHeapReserve = 0x00100000
.byte 0x00, 0x10, 0x00, 0x00  # SizeOfHeapCommit = 0x00001000
.byte 0x00, 0x00, 0x00, 0x00  # LoaderFlags
.byte 0x10, 0x00, 0x00, 0x00  # NumberOfRvaAndSizes = 16
.space 128                  # DataDirectory
# ---- section table @ 0x178 (2 × 40 bytes) ----
.byte 0x2E, 0x74, 0x65, 0x78, 0x74, 0x00, 0x00, 0x00  # '.text'
.byte 0x00, 0x02, 0x00, 0x00  # VirtualSize = 0x200
.byte 0x00, 0x02, 0x00, 0x00  # VirtualAddress = 0x200
.byte 0x00, 0x02, 0x00, 0x00  # SizeOfRawData = 0x200
.byte 0x00, 0x02, 0x00, 0x00  # PointerToRawData = 0x200
.byte 0x00, 0x00, 0x00, 0x00  # PointerToRelocations
.byte 0x00, 0x00, 0x00, 0x00  # PointerToLinenumbers
.byte 0x00, 0x00            # NumberOfRelocations
.byte 0x00, 0x00            # NumberOfLinenumbers
.byte 0x20, 0x00, 0x00, 0x60  # Characteristics = CODE|EXECUTE|READ (0x60000020)
.byte 0x2E, 0x70, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00  # '.pdata'
.byte 0x18, 0x00, 0x00, 0x00  # VirtualSize = 0x18 (3 records)
.byte 0x00, 0x04, 0x00, 0x00  # VirtualAddress = 0x400
.byte 0x18, 0x00, 0x00, 0x00  # SizeOfRawData = 0x18
.byte 0x00, 0x04, 0x00, 0x00  # PointerToRawData = 0x400
.byte 0x00, 0x00, 0x00, 0x00  # PointerToRelocations
.byte 0x00, 0x00, 0x00, 0x00  # PointerToLinenumbers
.byte 0x00, 0x00            # NumberOfRelocations
.byte 0x00, 0x00            # NumberOfLinenumbers
.byte 0x40, 0x00, 0x00, 0x40  # Characteristics = INITIALIZED_DATA|READ (0x40000040)
.space 0x38                 # 0x1C8..0x1FF pad to 0x200
# ---- .text @ 0x200 ----
.org 0x82000200
entry:
    lis   r3, fmt@ha
    addi  r3, r3, fmt@l
    bl    thunk_0
    li    r3, 0
    lis   r10, 0x00f0
    ori   r10, r10, 0x0ffc
    stw   r3, 0(r10)            # exit door
spin:
    b     spin
fmt:
    .asciiz "pdata seeded run\\n"
    .org 0x82000300
helper_fn:                      # never referenced — only visible via .pdata
    li    r3, 0x2a
    blr
    .org 0x82000400
# ---- .pdata @ 0x400 (big-endian IMAGE_CE_RUNTIME_FUNCTION records) ----
    .byte 0x82, 0x00, 0x02, 0x00  # record 0: entry (0x82000200)
    .byte 0x00, 0x00, 0x01, 0x02  # FunctionLength=0x40 (0x100 B), ThirtyTwoBit
    .byte 0x82, 0x00, 0x03, 0x00  # record 1: helper_fn (0x82000300)
    .byte 0x00, 0x00, 0x00, 0x12  # FunctionLength=4 (0x10 B), ThirtyTwoBit
    .byte 0x82, 0x00, 0x10, 0x00  # record 2: HLE thunk (0x82001000)
    .byte 0x00, 0x00, 0x00, 0x06  # FunctionLength=1 (0x04 B), ThirtyTwoBit
    .org 0x82001000
thunk_0:
    .word 0x01000003            # type=1 (thunk), ordinal 3 (DbgPrint)
`;

/** Build the PE-shaped synthetic XEX2 carrying a `.pdata` section. */
export function buildPdataXex() {
  const { bytes } = assemble(PDATA_SOURCE);
  return buildSyntheticXexFromImage(bytes, { compression: 'basic', encryption: 'normal', importOrdinals: [0x0003] });
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
