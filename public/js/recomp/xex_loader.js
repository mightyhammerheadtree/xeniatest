/**
 * xex_loader.js — load (and, for retail images, decrypt + decompress) an
 * XEX2 executable into the flat 64 MiB guest arena.
 *
 * This is a faithful port of the image-loading half of xenia-canary's
 * XexModule::ReadImage() (src/xenia/cpu/xex_module.cc), mirrored here:
 *
 *   - The per-image session key is AES-128-ECB-wrapped in the security info
 *     with a *public, well-known* retail key
 *     (20 B1 85 A5 9D 28 FD C3 40 58 3F BB 08 96 BF 91) — the same constant
 *     xenia ships publicly. Devkit images use the all-zero devkit key
 *     (xenia's xe_xex2_devkit_key). No console, CPU key or NAND is involved.
 *   - The image is AES-128-CBC decrypted with a zero IV; the chain carries
 *     across the whole image (xenia's aes_decrypt_buffer). BASIC-compressed
 *     images decrypt block-by-block with the chain persisting across blocks.
 *   - BASIC decompression: FILE_FORMAT_INFO opt header (0x000003FF) describes
 *     {data_size, zero_size} blocks; after each compressed block, zero_size
 *     implicit zero bytes follow. LZX ("normal", compression type 2) and
 *     delta (type 3) are NOT ported — they fail loudly with a clear error.
 *   - Image base address comes from the security info load_address,
 *     overridden by the IMAGE_BASE_ADDRESS opt header (0x00010201) when
 *     present. Entry point comes from the ENTRY_POINT opt header
 *     (0x00010100).
 *   - Total image size is derived from the security info page descriptors
 *     (low nibble = section type, high 28 bits = page count; page size is
 *     4 KiB when XEX_IMAGE_PAGE_SIZE_4KB (0x10000000) is set, else 64 KiB).
 *   - Import libraries (opt header 0x000103FF) are resolved against the
 *     bundled xenia export tables (hle_tables.js) at load time and returned
 *     as `imports`: the recompiler turns calls to those addresses into HLE
 *     host calls. Variable imports get the 0xF00DF00D sentinel written into
 *     their slot, like xenia's unresolved-variable path.
 *
 * Like xenia, we try the retail key first and fall back to the devkit key
 * (xenia's actual fallback: it retries ReadImage with the devkit key when
 * the retail attempt fails; both are public constants).
 */

import { aesCbcDecryptBlock, aesEcbDecrypt } from './aes.js';
import './hle_tables.js'; // UMD side effect: sets globalThis.XenonHLETables

const XENIA_EXPORTS = globalThis.XenonHLETables.XENIA_EXPORTS;

export const XEX2_RETAIL_KEY = new Uint8Array([
  0x20, 0xb1, 0x85, 0xa5, 0x9d, 0x28, 0xfd, 0xc3,
  0x40, 0x58, 0x3f, 0xbb, 0x08, 0x96, 0xbf, 0x91,
]);
export const XEX2_DEVKIT_KEY = new Uint8Array(16); // all zeros (xenia's xe_xex2_devkit_key)

export const XEX_HEADER_FILE_FORMAT_INFO = 0x000003ff;
export const XEX_HEADER_ORIGINAL_BASE_ADDRESS = 0x00010001;
export const XEX_HEADER_ENTRY_POINT = 0x00010100;
export const XEX_HEADER_IMAGE_BASE_ADDRESS = 0x00010201;
export const XEX_HEADER_IMPORT_LIBRARIES = 0x000103ff;

export const XEX_COMPRESSION_NONE = 0;
export const XEX_COMPRESSION_BASIC = 1;
export const XEX_COMPRESSION_NORMAL = 2; // LZX
export const XEX_COMPRESSION_DELTA = 3;

export const XEX_ENCRYPTION_NONE = 0;
export const XEX_ENCRYPTION_NORMAL = 1;

export const XEX_IMAGE_PAGE_SIZE_4KB = 0x10000000;
export const XEX_SECTION_CODE = 1;
export const XEX_SECTION_DATA = 2;
export const XEX_SECTION_READONLY_DATA = 3;

export class XexLoadError extends Error {
  constructor(message) { super(message); this.name = 'XexLoadError'; }
}

const MASK = 0x03ffffff; // 64 MiB flat guest arena

function u32be(bytes, off) {
  return ((bytes[off] << 24) | (bytes[off + 1] << 16) | (bytes[off + 2] << 8) | bytes[off + 3]) >>> 0;
}
function u16be(bytes, off) {
  return (bytes[off] << 8) | bytes[off + 1];
}

/** Parse the XEX2 header + opt-header directory (no decryption). */
export function parseXex2Header(bytes) {
  if (!(bytes instanceof Uint8Array)) throw new XexLoadError('XEX2 image must be a Uint8Array');
  if (bytes.length < 0x18) throw new XexLoadError('not an XEX2 file: too small');
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== 'XEX2') throw new XexLoadError(`not an XEX2 file: magic is '${magic}'`);
  const headerSize = u32be(bytes, 0x8);
  if (headerSize < 0x18 || headerSize > bytes.length) {
    throw new XexLoadError(`XEX2 header size ${headerSize} is outside the file (${bytes.length} bytes)`);
  }
  const moduleFlags = u32be(bytes, 0x4);
  const securityOffset = u32be(bytes, 0x10);
  const headerCount = u32be(bytes, 0x14);
  const options = new Map();
  let cursor = 0x18;
  for (let i = 0; i < headerCount; i++, cursor += 8) {
    if (cursor + 8 > headerSize) throw new XexLoadError('XEX2 opt-header directory overruns the header');
    const key = u32be(bytes, cursor);
    const value = u32be(bytes, cursor + 4);
    options.set(key, { key, value, kind: key & 0xff });
  }
  return { moduleFlags, headerSize, securityOffset, headerCount, options };
}

/** Resolve an opt header the way xenia's GetOptHeader does. Returns null when absent. */
function optValue(header, key) {
  const row = header.options.get(key);
  if (!row) return null;
  if (row.kind === 0x00 || row.kind === 0x01) return { inline: true, value: row.value };
  return { inline: false, value: row.value }; // value is an offset from the XEX2 header base
}

/** Read security info fields (image_size, flags, load address, page descriptors). */
export function parseSecurityInfo(bytes, header) {
  const sec = header.securityOffset;
  if (!sec || sec + 0x184 > bytes.length) {
    throw new XexLoadError('XEX2 security info is missing or truncated');
  }
  const imageSize = u32be(bytes, sec + 0x4);
  const imageFlags = u32be(bytes, sec + 0x10c);
  const loadAddress = u32be(bytes, sec + 0x110);
  const pageSize = (imageFlags & XEX_IMAGE_PAGE_SIZE_4KB) ? 0x1000 : 0x10000;
  const aesKey = bytes.slice(sec + 0x150, sec + 0x160);
  const pageDescriptorCount = u32be(bytes, sec + 0x180);
  const pageDescriptors = [];
  let totalSize = 0;
  let pd = sec + 0x184;
  for (let i = 0; i < pageDescriptorCount; i++, pd += 0x18) {
    if (pd + 4 > bytes.length) throw new XexLoadError('XEX2 page descriptors overrun the file');
    const word = u32be(bytes, pd);
    const type = word & 0x0f;
    const pageCount = (word >>> 4) & 0x0fffffff;
    pageDescriptors.push({ type, pageCount });
    totalSize += pageCount * pageSize;
  }
  return { imageSize, imageFlags, loadAddress, pageDescriptors, totalSize, pageSize, aesKey };
}

function imageFitsCheck(security, format) {
  if (format.compressionType === XEX_COMPRESSION_BASIC) {
    let n = 0;
    for (const b of format.blocks) n += b.dataSize + b.zeroSize;
    return Math.max(n, security.totalSize || 0);
  }
  return security.totalSize || 0;
}

/** Parse the FILE_FORMAT_INFO opt header. */
export function parseFileFormatInfo(bytes, header) {
  const opt = optValue(header, XEX_HEADER_FILE_FORMAT_INFO);
  if (!opt) throw new XexLoadError('XEX2 is missing the FILE_FORMAT_INFO opt header');
  if (opt.inline) throw new XexLoadError('XEX2 FILE_FORMAT_INFO unexpectedly stored inline');
  const off = opt.value;
  if (off + 8 > bytes.length) throw new XexLoadError('XEX2 FILE_FORMAT_INFO offset is outside the file');
  const infoSize = u32be(bytes, off);
  const encryptionType = u16be(bytes, off + 4);
  const compressionType = u16be(bytes, off + 6);
  const info = { infoSize, encryptionType, compressionType, blocks: [] };
  if (compressionType === XEX_COMPRESSION_BASIC) {
    const blockCount = Math.max(0, Math.floor((infoSize - 8) / 8));
    for (let i = 0; i < blockCount; i++) {
      const boff = off + 8 + i * 8;
      if (boff + 8 > bytes.length) throw new XexLoadError('XEX2 BASIC block table overruns the file');
      info.blocks.push({ dataSize: u32be(bytes, boff), zeroSize: u32be(bytes, boff + 4) });
    }
  }
  return info;
}

/** Parse the IMPORT_LIBRARIES opt header (header-side metadata only). */
export function parseImportLibraries(bytes, header) {
  const opt = optValue(header, XEX_HEADER_IMPORT_LIBRARIES);
  if (!opt) return [];
  if (opt.inline) throw new XexLoadError('XEX2 IMPORT_LIBRARIES unexpectedly stored inline');
  const off = opt.value;
  if (off + 12 > bytes.length) throw new XexLoadError('XEX2 IMPORT_LIBRARIES offset is outside the file');
  const size = u32be(bytes, off);
  if (!size || off + size > bytes.length) throw new XexLoadError('XEX2 IMPORT_LIBRARIES size is outside the file');
  const tableSize = u32be(bytes, off + 4);
  const tableCount = u32be(bytes, off + 8);
  // String table: NUL-terminated entries, each padded to 4 bytes.
  const strings = [];
  let i = 0, o = 0;
  const tableEnd = Math.min(off + 12 + tableSize, bytes.length);
  while (o < tableCount && off + 12 + i < tableEnd) {
    let str = '';
    let j = off + 12 + i;
    while (j < tableEnd && bytes[j] !== 0) { str += String.fromCharCode(bytes[j]); j++; }
    strings.push(str);
    i = j - (off + 12) + 1;
    if (i % 4 !== 0) i += 4 - (i % 4);
    o++;
  }
  // xex2_import_library records follow the string table.
  const libraries = [];
  let cursor = off + 12 + tableSize;
  const end = off + size;
  while (cursor + 0x28 <= Math.min(end, bytes.length)) {
    const libSize = u32be(bytes, cursor);
    if (!libSize || libSize < 0x28) break;
    const nameIndex = u16be(bytes, cursor + 0x24) & 0xff;
    const count = u16be(bytes, cursor + 0x26);
    const importTable = [];
    for (let k = 0; k < count; k++) {
      const recordAddrOff = cursor + 0x28 + k * 4;
      if (recordAddrOff + 4 > bytes.length) throw new XexLoadError('XEX2 import table overruns the file');
      importTable.push(u32be(bytes, recordAddrOff));
    }
    libraries.push({ name: strings[nameIndex] ?? `module-${nameIndex}`, count, importTable });
    cursor += libSize;
  }
  return libraries;
}

function moduleIdFor(name) {
  const base = String(name).toLowerCase().replace(/\\/g, '/').split('/').pop().replace(/\.(xex|dll)$/, '');
  return base;
}

function lookupExport(moduleId, ordinal) {
  const table = XENIA_EXPORTS[moduleId];
  return table ? (table.get(ordinal) ?? null) : null;
}

/**
 * Decrypt + decompress the image. Returns { image, base, entry, security, format }.
 * `keyChoice`: 'retail' | 'devkit'
 */
function decryptImage(bytes, header, security, format, keyChoice) {
  const key = keyChoice === 'devkit' ? XEX2_DEVKIT_KEY : XEX2_RETAIL_KEY;
  const sessionKey = aesEcbDecrypt(key, security.aesKey);
  const imageBase = security.loadAddress & MASK;
  const source = bytes.subarray(header.headerSize);

  if (format.compressionType === XEX_COMPRESSION_NORMAL) {
    throw new XexLoadError('XEX2 uses LZX ("normal") compression, which is not supported in the browser — recompress the image to BASIC/none (e.g. with xextool) and try again');
  }
  if (format.compressionType === XEX_COMPRESSION_DELTA) {
    throw new XexLoadError('XEX2 uses delta compression, which is not supported — recompress the image (e.g. with xextool) and try again');
  }

  if (format.compressionType === XEX_COMPRESSION_NONE) {
    if (format.encryptionType === XEX_ENCRYPTION_NONE) {
      return { image: source.slice(), derivedBase: imageBase, derivedSize: source.length };
    }
    const chain = new Uint8Array(16); // zero IV, image-wide chain
    const out = new Uint8Array(source.length - (source.length % 16));
    for (let off = 0; off < out.length; off += 16) {
      out.set(aesCbcDecryptBlock(sessionKey, source.subarray(off, off + 16), chain), off);
    }
    return { image: out, derivedBase: imageBase, derivedSize: out.length };
  }

  // BASIC
  let uncompressed = 0;
  for (const b of format.blocks) uncompressed += b.dataSize + b.zeroSize;
  const image = new Uint8Array(uncompressed);
  const chain = new Uint8Array(16); // zero IV, persists across blocks
  let src = 0, dst = 0;
  for (let n = 0; n < format.blocks.length; n++) {
    const { dataSize, zeroSize } = format.blocks[n];
    if (format.encryptionType === XEX_ENCRYPTION_NORMAL && (dataSize % 16) !== 0) {
      throw new XexLoadError(`XEX2 BASIC block ${n} has a non-16-byte-aligned data size (${dataSize}) with encryption on — refusing to guess`);
    }
    const copySize = format.encryptionType === XEX_ENCRYPTION_NORMAL ? Math.ceil(dataSize / 16) * 16 : dataSize;
    if (src + copySize > source.length || dst + dataSize > image.length) {
      throw new XexLoadError(`XEX2 BASIC block ${n} is out of bounds (file is corrupt or this is a delta image)`);
    }
    if (format.encryptionType === XEX_ENCRYPTION_NONE) {
      image.set(source.subarray(src, src + dataSize), dst);
    } else {
      for (let m = 0; m < copySize; m += 16) {
        const pt = aesCbcDecryptBlock(sessionKey, source.subarray(src + m, src + m + 16), chain);
        image.set(pt.subarray(0, Math.min(16, dataSize - m)), dst + m);
      }
    }
    // zero_size bytes are already zero in `image`
    src += dataSize;
    dst += dataSize + zeroSize;
  }
  return { image, derivedBase: imageBase, derivedSize: uncompressed };
}

/**
 * Load an XEX2 image.
 * @param {Uint8Array} bytes  the whole .xex2 file
 * @param {{devKey?: boolean}} [opts]  force the devkit key (skip the retail attempt)
 * @returns {{
 *   image: Uint8Array, base: number, entry: number,
 *   moduleFlags: number, imageSize: number, totalSize: number, pageSize: number,
 *   pageDescriptors: Array<{type:number, pageCount:number}>,
 *   encryptionType: number, compressionType: number, usedDevkitKey: boolean,
 *   imports: Array<{addr:number, type:number, ordinal:number, moduleId:string, moduleName:string, name:string}>,
 * }}
 */
export function loadXex2(bytes, opts = {}) {
  const header = parseXex2Header(bytes);
  const security = parseSecurityInfo(bytes, header);
  const format = parseFileFormatInfo(bytes, header);
  const importLibraries = parseImportLibraries(bytes, header);

  // Image base: security info load address, overridden by IMAGE_BASE_ADDRESS.
  // Guest addresses are arbitrary 32-bit values; the tool masks them into the
  // flat 64 MiB arena at the wasm memory boundary (same convention as the rest
  // of the recompiler), so any base is legal.
  let base = security.loadAddress >>> 0;
  const baseOpt = optValue(header, XEX_HEADER_IMAGE_BASE_ADDRESS);
  if (baseOpt) base = baseOpt.value >>> 0;

  // Entry point opt header (inline value).
  let entry = base;
  const entryOpt = optValue(header, XEX_HEADER_ENTRY_POINT);
  if (entryOpt) entry = entryOpt.value >>> 0;

  const maskedBase = base & MASK;
  if (imageFitsCheck(security, format) > (MASK + 1) - maskedBase) {
    throw new XexLoadError('XEX2 image is larger than the 64 MiB guest arena');
  }

  // Try retail first, then devkit (mirrors xenia's fallback; both keys are public).
  const attempts = opts.devKey ? ['devkit'] : ['retail', 'devkit'];
  let loaded = null, usedDevkitKey = false, lastError = null;
  for (const choice of attempts) {
    try {
      const result = decryptImage(bytes, header, security, format, choice);
      // Sanity check: a valid image starts with a PE header. Mirrors xenia's
      // "try the other key if the PE check fails" behavior.
      if (result.image.length >= 2 && result.image[0] === 0x4d && result.image[1] === 0x5a) {
        loaded = result;
        usedDevkitKey = choice === 'devkit';
        break;
      }
      lastError = new XexLoadError(`XEX2 ${choice}-key decryption produced no PE header (wrong key?)`);
    } catch (e) {
      if (e instanceof XexLoadError) throw e; // format errors are not key problems — report immediately
      lastError = e;
    }
  }
  if (!loaded) {
    throw new XexLoadError(`XEX2 image did not decrypt with the ${attempts.join(' or ')} key — ${lastError ? lastError.message : 'no PE header found'}. Decrypted images are required.`);
  }

  const image = loaded.image;
  const totalSize = security.totalSize || image.length;
  if (security.totalSize && image.length > totalSize) {
    throw new XexLoadError(`XEX2 image (${image.length} bytes) exceeds the size declared by its page descriptors (${totalSize} bytes) — corrupt or unsupported image`);
  }

  // Resolve import records (they live inside the now-decrypted image).
  const baseMasked = base & MASK;
  const imports = [];
  for (const library of importLibraries) {
    const moduleId = moduleIdFor(library.name);
    for (const recordAddr of library.importTable) {
      const ix = (recordAddr & MASK) - baseMasked;
      if (ix < 0 || ix + 4 > image.length) continue; // record outside the image — skip, the call will trap loudly
      const value = u32be(image, ix);
      const type = (value >>> 24) & 0xff;
      const ordinal = value & 0xffff;
      const name = lookupExport(moduleId, ordinal) ?? `ordinal_${ordinal.toString(16).toUpperCase().padStart(3, '0')}`;
      if (type === 0) {
        // Variable import: poke the unresolved-import sentinel (xenia parity).
        image[ix] = 0xf0; image[ix + 1] = 0x0d; image[ix + 2] = 0xf0; image[ix + 3] = 0x0d;
        continue;
      }
      imports.push({
        addr: recordAddr & MASK,
        type,
        ordinal,
        moduleId,
        moduleName: library.name,
        name,
      });
    }
  }

  return {
    image,
    base: baseMasked,
    entry: entry & MASK,
    moduleFlags: header.moduleFlags >>> 0,
    imageSize: security.imageSize >>> 0,
    totalSize,
    pageSize: security.pageSize,
    pageDescriptors: security.pageDescriptors,
    encryptionType: format.encryptionType,
    compressionType: format.compressionType,
    usedDevkitKey,
    imports,
  };
}
