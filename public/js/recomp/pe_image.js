/**
 * pe_image.js — minimal PE structures reader for a *decrypted* Xbox 360
 * image (XEX2 payload). The Xbox 360 PE container uses little-endian
 * Windows-PE header fields; the code/data sections themselves are
 * big-endian PPC.
 *
 * Only what the recompiler needs, ported from hedge-dev/XenonRecomp
 * (MIT, © hedge-dev 2025):
 *   - XenonUtils/xex.cpp — locates the PE via e_lfanew and maps sections
 *     as `image.data + VirtualAddress` (XEX2 flattens sections
 *     contiguously, so an RVA is an offset into the decrypted image).
 *   - XenonUtils/xbox.h — `_IMAGE_CE_RUNTIME_FUNCTION` (8 bytes each:
 *     big-endian BeginAddress, then PrologLength:8 | FunctionLength:22 |
 *     ThirtyTwoBit:1 | ExceptionFlag:1 as a big-endian 32-bit word).
 *   - XenonRecomp/recompiler.cpp — seeds function discovery from the
 *     `.pdata` section (`FunctionLength * 4` = function size in bytes).
 *
 * `BeginAddress` in .pdata is an absolute guest VA (e.g. 0x82001000),
 * like every other address in the guest; callers mask it into the flat
 * 64 MiB arena. A missing or unparsable .pdata yields an empty seed
 * list — the caller falls back to pure control-flow discovery (the
 * behavior before .pdata seeding existed) and gets a `peError`
 * breadcrumb instead of an exception.
 */

export class PeImageError extends Error {
  constructor(message) { super(message); this.name = 'PeImageError'; }
}

export const IMAGE_SCN_CNT_CODE = 0x00000020;

const PE_SIGNATURE = 0x00004550; // 'PE\0\0' as little-endian u32

function u16le(b, o) { return b[o] | (b[o + 1] << 8); }
function u32le(b, o) { return (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0; }
function u32be(b, o) { return ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0; }

/**
 * Parse the PE section table.
 * @param {Uint8Array} image  the decrypted image bytes (offset 0 == image base)
 * @returns {{ machine: number, sections: Array<{name: string, virtualAddress: number,
 *   virtualSize: number, sizeOfRawData: number, pointerToRawData: number,
 *   characteristics: number, isCode: boolean}> }}
 * @throws {PeImageError} when the PE headers are missing or truncated
 */
export function parsePeSections(image) {
  if (!(image instanceof Uint8Array)) throw new PeImageError('PE image must be a Uint8Array');
  if (image.length < 0x40 || image[0] !== 0x4d || image[1] !== 0x5a) {
    throw new PeImageError("not a PE image: no 'MZ' signature");
  }
  const eLfanew = u32le(image, 0x3c);
  const fileHeader = eLfanew + 4;
  if (fileHeader + 20 > image.length) {
    throw new PeImageError('PE header offset (e_lfanew) points outside the image');
  }
  if (u32le(image, eLfanew) !== PE_SIGNATURE) {
    throw new PeImageError("not a PE image: no 'PE\\0\\0' signature at e_lfanew");
  }
  const machine = u16le(image, fileHeader);
  const numberOfSections = u16le(image, fileHeader + 2);
  const sizeOfOptionalHeader = u16le(image, fileHeader + 16);
  const sectionTable = fileHeader + 20 + sizeOfOptionalHeader;
  const sections = [];
  for (let i = 0; i < numberOfSections; i++) {
    const off = sectionTable + i * 40;
    if (off + 40 > image.length) {
      throw new PeImageError(`section table entry ${i} overruns the image`);
    }
    let name = '';
    for (let j = 0; j < 8 && image[off + j]; j++) name += String.fromCharCode(image[off + j]);
    sections.push({
      name,
      virtualSize: u32le(image, off + 8),
      virtualAddress: u32le(image, off + 12),
      sizeOfRawData: u32le(image, off + 16),
      pointerToRawData: u32le(image, off + 20),
      characteristics: u32le(image, off + 36) >>> 0,
      isCode: !!(u32le(image, off + 36) & IMAGE_SCN_CNT_CODE),
    });
  }
  return { machine, sections };
}

/**
 * Parse the `.pdata` section (IMAGE_CE_RUNTIME_FUNCTION[], big-endian) from
 * an already-parsed section list.
 * @returns {Array<{addr: number, size: number}>} function starts as masked
 *   guest addresses plus their byte sizes (FunctionLength * 4); sorted,
 *   deduplicated, zero-BeginAddress records skipped.
 */
function sectionBaseName(name) {
  // Section names are NUL-padded 8-byte fields; tolerate trailing junk defensively.
  const i = name.indexOf('\0');
  return i === -1 ? name : name.slice(0, i);
}

export function pdataFunctionStartsFromSections(image, sections, mask) {
  const pdata = sections.find((s) => sectionBaseName(s.name).toLowerCase() === '.pdata');
  if (!pdata) return [];
  if (pdata.virtualSize % 8 !== 0) {
    throw new PeImageError(`.pdata section size ${pdata.virtualSize} is not a multiple of 8 (IMAGE_CE_RUNTIME_FUNCTION)`);
  }
  const start = pdata.virtualAddress;
  if (start + pdata.virtualSize > image.length) {
    throw new PeImageError('.pdata section header points outside the image');
  }
  const out = [];
  const seen = new Set();
  const count = pdata.virtualSize / 8;
  for (let i = 0; i < count; i++) {
    const off = start + i * 8;
    const begin = u32be(image, off);
    const data = u32be(image, off + 4);
    const functionLength = (data >>> 2) & 0x3fffff; // bits [23:2]
    if (begin === 0) continue;
    const addr = begin & mask;
    if (seen.has(addr)) continue;
    seen.add(addr);
    out.push({ addr, size: functionLength * 4 });
  }
  out.sort((a, b) => a.addr - b.addr);
  return out;
}

/**
 * Convenience: parse the image and return its .pdata function starts.
 * Throws PeImageError on malformed input; returns [] when there is no .pdata.
 */
export function pdataFunctionStarts(image, mask) {
  const { sections } = parsePeSections(image);
  return pdataFunctionStartsFromSections(image, sections, mask);
}
