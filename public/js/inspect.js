/**
 * inspect.js — readers for XEX2 and ELF containers.
 *
 * XEX2 layout (from xenia-canary's xex2_info.h / xex_module.cc):
 *   magic 'XEX2' (4 bytes)
 *   module_flags   uint32 BE
 *   header_size    uint32 BE   (size of the XEX header block)
 *   security_offset uint32 BE  (file offset of the security info, or 0)
 *   header_count   uint32 BE
 *   header_count × { key: uint32 BE, value: uint32 BE }
 *       key & 0xFF == 0x00 → value is a direct uint32
 *       key & 0xFF == 0x01 → value is inline data (pointer field)
 *       otherwise          → value is an OFFSET into the header block
 *   ... then the optional-header payloads, then the image pages.
 *
 * ELF support covers ELF32/ELF64 in either byte order, but targets the
 * PowerPC (EM_PPC 20 / EM_PPC64 21) family used by Xenon.
 */

export const XEX_MODULE_FLAGS = {
  0x00000002: 'TITLE',
  0x00000008: 'DLL',
  0x00000010: 'MODULE_PATCH',
  0x00000080: 'USER_MODE',
  0x00010000: 'TITLE_MANAGEMENT',
  0x00020000: 'FILESYSTEM_MANAGEMENT',
  0x00040000: 'NETWORK_MANAGEMENT',
  0x00080000: 'ONLINE_INSTRUMENTATION',
  0x00100000: 'RUNNING_CONTENT',
  0x00200000: 'OVER_THE_NETWORK',
};

const XEX_OPT_KEYS = {
  0x00000002: 'RESOURCE_INFO',
  0x000001FF: 'XEX_SECTIONS',
  0x000003FF: 'FILE_FORMAT_INFO',
  0x00008105: 'DEVICE_ID',
  0x00010001: 'ORIGINAL_BASE_ADDRESS',
  0x00010100: 'ENTRY_POINT',
  0x00010201: 'IMAGE_BASE_ADDRESS',
  0x000103FF: 'IMPORT_LIBRARIES',
  0x000183FF: 'ORIGINAL_PE_NAME',
  0x000200FF: 'STATIC_LIBRARIES',
  0x00020104: 'TLS_INFO',
  0x00020200: 'DEFAULT_STACK_SIZE',
  0x00020301: 'DEFAULT_FILESYSTEM_CACHE_SIZE',
  0x00020401: 'DEFAULT_HEAP_SIZE',
  0x00028002: 'PAGE_HEAP_SIZE_AND_FLAGS',
  0x00030000: 'SYSTEM_FLAGS',
  0x00040006: 'EXECUTION_INFO',
  0x000401FF: 'SERVICE_ID_LIST',
  0x00040201: 'TITLE_WORKSPACE_SIZE',
  0x00040404: 'LAN_KEY',
  0x000406FF: 'MULTIDISC_MEDIA_IDS',
  0x000407FF: 'ALTERNATE_TITLE_IDS',
  0x00040801: 'ADDITIONAL_TITLE_MEMORY',
  0x00E10302: 'IMPORTS_BY_NAME',
  0x00E10402: 'EXPORTS_BY_NAME',
};

const COMPRESSION_NAMES = { 0: 'raw', 1: 'lzx', 2: 'basic-compression', 3: 'others' };

/** Parse an XEX2 header. Pass the first `min(size, 4 MiB)` bytes of the file. */
export function parseXex2(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 24) throw new Error('not an XEX2 file: too small');
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== 'XEX2') throw new Error(`not an XEX2 file: magic is '${magic}'`);

  const moduleFlags = view.getUint32(4, false);
  const headerSize = view.getUint32(8, false);
  const securityOffset = view.getUint32(12, false);
  const headerCount = view.getUint32(16, false);

  const info = {
    format: 'XEX2',
    moduleFlags,
    moduleFlagNames: Object.entries(XEX_MODULE_FLAGS).filter(([bit]) => moduleFlags & Number(bit)).map(([, name]) => name),
    headerSize,
    securityOffset,
    headerCount,
    headers: [],
    entryPoint: null,
    imageBaseAddress: null,
    originalBaseAddress: null,
    defaultStackSize: null,
    defaultHeapSize: null,
    imageSize: null,
    loadAddress: null,
    pageDescriptors: [],
    compression: null,
    importLibraries: [],
  };

  const optOffset = 20;
  for (let i = 0; i < headerCount && optOffset + i * 8 + 8 <= bytes.length; i++) {
    const key = view.getUint32(optOffset + i * 8, false);
    const value = view.getUint32(optOffset + i * 8 + 4, false);
    const kind = key & 0xff;
    const row = { index: i, key, keyName: XEX_OPT_KEYS[key] ?? `UNKNOWN_${key.toString(16).toUpperCase()}`, kind };
    if (kind === 0x00) {
      row.value = value >>> 0;
    } else if (kind === 0x01) {
      row.value = value >>> 0; // inline raw data; semantics depend on key
    } else {
      row.offset = value >>> 0;
      row.value = safeReadUint32(view, value >>> 0);
    }
    info.headers.push(row);

    if (key === 0x00010100) info.entryPoint = row.value;
    else if (key === 0x00010201) info.imageBaseAddress = row.value;
    else if (key === 0x00010001) info.originalBaseAddress = row.value;
    else if (key === 0x00020200) info.defaultStackSize = row.value;
    else if (key === 0x00020401) info.defaultHeapSize = row.value;
    else if (key === 0x000003FF) info.compression = parseFormatInfo(view, value >>> 0);
    else if (key === 0x000103FF) info.importLibraries = parseImportLibraries(bytes, view, value >>> 0);
  }

  // Security info (only scalar fields; signature skipped).
  if (securityOffset && securityOffset + 24 <= bytes.length) {
    const sec = securityOffset;
    info.securityHeaderSize = view.getUint32(sec, false);
    info.imageSize = view.getUint32(sec + 4, false);
    // rsa_signature[0x100] skipped at sec + 8
    const infoSizeOff = sec + 8 + 0x100;
    if (infoSizeOff + 20 <= bytes.length) {
      info.securityInfoSize = view.getUint32(infoSizeOff, false);
      info.imageFlags = view.getUint32(infoSizeOff + 4, false);
      info.loadAddress = view.getUint32(infoSizeOff + 8, false);
      info.pageDescriptorCount = view.getUint32(infoSizeOff + 12, false);
      let pd = infoSizeOff + 16;
      for (let i = 0; i < info.pageDescriptorCount && pd + 24 <= bytes.length; i++) {
        const type = (view.getUint32(pd, false) >>> 28) & 0xf;
        const pageCount = view.getUint32(pd, false) & 0x0fffffff;
        info.pageDescriptors.push({ type, pageCount, bytes: pageCount * 4096 });
        pd += 24;
      }
    }
  }
  return info;
}

const PAGE_TYPE_NAMES = { 1: 'code', 2: 'data', 3: 'rodata' };

function parseFormatInfo(view, offset) {
  if (!offset || offset + 8 > view.byteLength) return null;
  const infoOffset = view.getUint32(offset, false);
  const size = view.byteLength;
  if (infoOffset >= size || infoOffset + 16 > size) return { raw: null };
  const info = view.getUint32(infoOffset, false);
  // xex2_opt_file_format_info: flags (u16), type (u16), ... compression_type at +? best effort:
  return { compressionType: info & 0xffff, compressionName: COMPRESSION_NAMES[info & 0xffff] ?? `type ${info & 0xffff}` };
}

function parseImportLibraries(bytes, view, offset) {
  if (!offset || offset + 8 > bytes.length) return [];
  const count = view.getUint32(offset, false);
  const names = [];
  let cursor = offset + 4;
  // Each import library descriptor: name (null-terminated ASCII) then a table
  // of (ordinal, hint) pairs. Read names best-effort.
  for (let i = 0; i < Math.min(count, 64) && cursor + 4 <= bytes.length; i++) {
    const libNameOff = view.getUint32(cursor, false);
    cursor += 4;
    const name = readAsciiZ(bytes, libNameOff, 64);
    if (name) names.push(name);
  }
  return names;
}

function readAsciiZ(bytes, offset, maxLen) {
  if (!offset || offset >= bytes.length) return '';
  let out = '';
  for (let i = 0; i < maxLen && offset + i < bytes.length; i++) {
    const c = bytes[offset + i];
    if (c === 0) break;
    if (c < 0x20 || c > 0x7e) return '';
    out += String.fromCharCode(c);
  }
  return out;
}

function safeReadUint32(view, offset) {
  if (offset + 4 <= view.byteLength) return view.getUint32(offset, false) >>> 0;
  return null;
}

export function xexPageTypeName(type) {
  return PAGE_TYPE_NAMES[type] ?? `type-${type}`;
}

/** Parse ELF headers. Accepts the full file or its first bytes (needs e_shoff). */
export function parseElf(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 16 || bytes[0] !== 0x7f || bytes[1] !== 0x45 || bytes[2] !== 0x4c || bytes[3] !== 0x46) {
    throw new Error('not an ELF file');
  }
  const elfClass = bytes[4];   // 1 = ELF32, 2 = ELF64
  const littleEndian = bytes[5] === 1;
  const is64 = elfClass === 2;
  const rd32 = (off) => view.getUint32(off, littleEndian);
  const rd16 = (off) => view.getUint16(off, littleEndian);
  const rd64 = (off) => view.getBigUint64(off, littleEndian);

  const info = {
    format: is64 ? 'ELF64' : 'ELF32',
    endianness: littleEndian ? 'little' : 'big',
    klass: is64 ? 64 : 32,
    osAbi: bytes[7],
    eType: rd16(16),
    machine: rd16(18),
    machineName: ELF_MACHINES[rd16(18)] ?? `machine ${rd16(18)} (0x${rd16(18).toString(16)})`,
    entry: null,
    programHeaders: [],
    sections: [],
  };

  if (is64) {
    info.entry = rd64(24);
    info.phoff = rd64(32);
    info.shoff = rd64(40);
    info.phnum = rd16(56);
    info.shnum = rd16(60);
    info.shstrndx = rd16(62);
    const phoff = Number(info.phoff);
    for (let i = 0; i < info.phnum && phoff + i * 56 + 56 <= bytes.length; i++) {
      const off = phoff + i * 56;
      info.programHeaders.push({
        type: rd32(off),
        flags: rd32(off + 4),
        offset: rd64(off + 8),
        vaddr: rd64(off + 16),
        filesz: rd64(off + 32),
        memsz: rd64(off + 40),
      });
    }
  } else {
    info.entry = BigInt(rd32(24));
    info.phoff = BigInt(rd32(28));
    info.shoff = BigInt(rd32(32));
    info.phnum = rd16(44);
    info.shnum = rd16(48);
    info.shstrndx = rd16(50);
    const phoff = Number(info.phoff);
    for (let i = 0; i < info.phnum && phoff + i * 32 + 32 <= bytes.length; i++) {
      const off = phoff + i * 32;
      info.programHeaders.push({
        type: rd32(off),
        offset: BigInt(rd32(off + 4)),
        vaddr: BigInt(rd32(off + 8)),
        filesz: BigInt(rd32(off + 16)),
        memsz: BigInt(rd32(off + 20)),
        flags: rd32(off + 24),
      });
    }
  }

  // Section names (bounded walk).
  if (info.shnum > 0 && info.shoff) {
    const shoff = Number(info.shoff);
    const shEntSize = rd16(is64 ? 58 : 46);
    const names = [];
    for (let i = 0; i < info.shnum && shoff + i * shEntSize + shEntSize <= bytes.length; i++) {
      const off = shoff + i * shEntSize;
      const nameOff = rd32(off);
      const shType = rd32(off + 4);
      const shAddr = is64 ? rd64(off + 16) : BigInt(rd32(off + 12));
      const shOffset = is64 ? rd64(off + 24) : BigInt(rd32(off + 16));
      const shSize = is64 ? rd64(off + 32) : BigInt(rd32(off + 20));
      names.push({ nameOff, shType, addr: shAddr, offset: shOffset, size: shSize });
    }
    // Resolve names against .shstrtab when present.
    const shstr = info.shstrndx < names.length ? names[info.shstrndx] : null;
    if (shstr) {
      const base = Number(shstr.offset);
      for (const s of names) {
        s.name = readAsciiZ(bytes, base + s.nameOff, 128);
      }
    }
    info.sections = names;
  }
  return info;
}

export const ELF_MACHINES = {
  20: 'EM_PPC (PowerPC 32)',
  21: 'EM_PPC64 (PowerPC 64)',
  0x14: 'EM_PPC',
  40: 'EM_ARM',
  62: 'EM_X86_64',
  183: 'EM_AARCH64',
  0x28: 'EM_AARCH64',
};

export const ELF_PT = {
  0: 'NULL', 1: 'LOAD', 2: 'DYNAMIC', 3: 'INTERP', 4: 'NOTE',
  6: 'PHDR', 7: 'TLS', 0x6474e550: 'GNU_EH_FRAME', 0x6474e551: 'GNU_STACK', 0x6474e552: 'GNU_RELRO',
};

/** Quick classification of what a file is. */
export function classifyMagic(bytes) {
  if (bytes.length >= 4 && String.fromCharCode(...bytes.slice(0, 4)) === 'XEX2') return 'xex';
  if (bytes.length >= 4 && bytes[0] === 0x7f && bytes[1] === 0x45 && bytes[2] === 0x4c && bytes[3] === 0x46) return 'elf';
  return 'unknown';
}
