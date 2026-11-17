/**
 * pm4.js — Xenos PM4 packet types, Type-3 opcodes, and packet header
 * construction. Encodings and names follow xenia-canary
 * (gpu/xenos.h, gpu/pm4_command_processor_implement.h), BSD-3-Clause.
 *
 * The guest command stream is big-endian — the helpers here read/write
 * u32s big-endian through DataView. Addresses of register writes are the
 * Xenos dword indices (see xenos_registers.js).
 */
(function (global) {
  'use strict';

  // Type-3 opcodes (IT_OPCODE). Values from xenia's Type3Opcode enum.
  const PM4_OPCODE = Object.freeze({
    PM4_ME_INIT: 0x48,
    PM4_NOP: 0x10,
    PM4_INDIRECT_BUFFER: 0x3f,
    PM4_INDIRECT_BUFFER_PFD: 0x37,
    PM4_WAIT_FOR_IDLE: 0x26,
    PM4_WAIT_REG_MEM: 0x3c,
    PM4_WAIT_REG_EQ: 0x52,
    PM4_WAIT_REG_GTE: 0x53,
    PM4_WAIT_UNTIL_READ: 0x5c,
    PM4_WAIT_IB_PFD_COMPLETE: 0x5d,
    PM4_REG_RMW: 0x21,
    PM4_REG_TO_MEM: 0x3e,
    PM4_MEM_WRITE: 0x3d,
    PM4_MEM_WRITE_CNTR: 0x4f,
    PM4_COND_EXEC: 0x44,
    PM4_COND_WRITE: 0x45,
    PM4_EVENT_WRITE: 0x46,
    PM4_EVENT_WRITE_SHD: 0x58,
    PM4_EVENT_WRITE_CFL: 0x59,
    PM4_EVENT_WRITE_EXT: 0x5a,
    PM4_EVENT_WRITE_ZPD: 0x5b,
    PM4_DRAW_INDX: 0x22,
    PM4_DRAW_INDX_2: 0x36,
    PM4_DRAW_INDX_BIN: 0x34,
    PM4_DRAW_INDX_2_BIN: 0x35,
    PM4_VIZ_QUERY: 0x23,
    PM4_SET_STATE: 0x25,
    PM4_SET_CONSTANT: 0x2d,
    PM4_SET_CONSTANT2: 0x55,
    PM4_SET_SHADER_CONSTANTS: 0x56,
    PM4_LOAD_ALU_CONSTANT: 0x2f,
    PM4_IM_LOAD: 0x27,
    PM4_IM_LOAD_IMMEDIATE: 0x2b,
    PM4_LOAD_CONSTANT_CONTEXT: 0x2e,
    PM4_INVALIDATE_STATE: 0x3b,
    PM4_SET_SHADER_BASES: 0x4a,
    PM4_SET_BIN_BASE_OFFSET: 0x4b,
    PM4_SET_BIN_MASK: 0x50,
    PM4_SET_BIN_SELECT: 0x51,
    PM4_CONTEXT_UPDATE: 0x5e,
    PM4_INTERRUPT: 0x54,
    PM4_XE_SWAP: 0x64,
    PM4_IM_STORE: 0x2c,
    PM4_SET_BIN_MASK_LO: 0x60,
    PM4_SET_BIN_MASK_HI: 0x61,
    PM4_SET_BIN_SELECT_LO: 0x62,
    PM4_SET_BIN_SELECT_HI: 0x63,
  });

  const PM4_OPCODE_BY_VALUE = new Map(Object.entries(PM4_OPCODE).map(([k, v]) => [v, k]));

  // CP event types (for EVENT_WRITE payloads).
  const CP_EVENT = Object.freeze({
    PS_DEALLOC: 1,
    VS_DONE_TS: 2,
    PS_DONE_TS: 3,
    CACHE_FLUSH_TS: 4,
    CONTEXT_DONE: 5,
    CACHE_FLUSH: 6,
    VIZQUERY_START: 7,
    VIZQUERY_END: 8,
    SC_WAIT_WC: 9,
    MPASS_PS_CP_REFETCH: 10,
    MPASS_PS_RST_START: 11,
    MPASS_PS_INCR_START: 12,
    RST_PIX_CNT: 13,
    RST_VTX_CNT: 14,
    TILE_FLUSH: 15,
    ZPASS_DONE: 21,
    CACHE_FLUSH_AND_INV_EVENT: 22,
    SCREEN_EXT_INIT: 25,
    SCREEN_EXT_RPT: 26,
    VS_FETCH_DONE_TS: 27,
  });

  // SourceSelect (VGT_DRAW_INITIATOR::source_select)
  const SOURCE_SELECT = Object.freeze({ kDMA: 0, kImmediate: 1, kAutoIndex: 2 });
  // IndexFormat (VGT_DRAW_INITIATOR::index_size)
  const INDEX_FORMAT = Object.freeze({ kInt16: 0, kInt32: 1 });
  // MajorMode (VGT_DRAW_INITIATOR::major_mode)
  const MAJOR_MODE = Object.freeze({ kImplicit: 0, kExplicit: 1 });
  // PrimitiveType (VGT_DRAW_INITIATOR::prim_type), only the values D3D9 emits
  const PRIMITIVE_TYPE = Object.freeze({
    kNone: 0x00, kPointList: 0x01, kLineList: 0x02, kLineStrip: 0x03,
    kTriangleList: 0x04, kTriangleFan: 0x05, kTriangleStrip: 0x06,
    kTriangleWithWFlags: 0x07, kRectangleList: 0x08, kLineLoop: 0x0c,
    kQuadList: 0x0d, kQuadStrip: 0x0e, kPolygon: 0x0f,
    k2DCopyRectListV0: 0x10, k2DCopyRectListV1: 0x11, k2DCopyRectListV2: 0x12,
    k2DCopyRectListV3: 0x13, k2DFillRectList: 0x14, k2DLineStrip: 0x15, k2DTriStrip: 0x16,
  });
  const PRIMITIVE_TYPE_NAMES = new Map(Object.entries(PRIMITIVE_TYPE).map(([k, v]) => [v, k]));

  /** ttcccccc cccccccc oiiiiiii iiiiiiii — type-0 header. */
  function makePacketType0(index, count, oneReg = false) {
    if (index > 0x7fff) throw new RangeError('type-0 register index out of range');
    if (count < 1 || count > 0x4000) throw new RangeError('type-0 count out of range');
    return ((0 << 30) | (((count - 1) & 0x3fff) << 16) | ((oneReg ? 1 : 0) << 15) | (index & 0x7fff)) >>> 0;
  }
  /** tt?????? ??222222 22222111 11111111 — type-1 header (two registers). */
  function makePacketType1(index1, index2) {
    if (index1 > 0x7ff || index2 > 0x7ff) throw new RangeError('type-1 register index out of range');
    return ((1 << 30) | ((index2 & 0x7ff) << 11) | (index1 & 0x7ff)) >>> 0;
  }
  /** type-2 no-op. */
  function makePacketType2() { return (2 << 30) >>> 0; }
  /** ttcccccc cccccccc ?ooooooo ???????p — type-3 header. count = dwords that follow. */
  function makePacketType3(opcode, count, predicate = false) {
    if (opcode > 0x7f) throw new RangeError('type-3 opcode out of range');
    if (count < 1 || count > 0x4000) throw new RangeError('type-3 count out of range');
    return ((3 << 30) | (((count - 1) & 0x3fff) << 16) | ((opcode & 0x7f) << 8) | (predicate ? 1 : 0)) >>> 0;
  }

  /** Decode a packet header word. */
  function parsePacketHeader(packet) {
    const p = packet >>> 0;
    const type = p >>> 30;
    if (type === 0) {
      return { type, count: ((p >>> 16) & 0x3fff) + 1, baseIndex: p & 0x7fff, writeOneReg: !!(p & 0x8000) };
    }
    if (type === 1) {
      return { type, reg1: p & 0x7ff, reg2: (p >>> 11) & 0x7ff };
    }
    if (type === 2) {
      return { type };
    }
    return {
      type: 3,
      count: ((p >>> 16) & 0x3fff) + 1,
      opcode: (p >>> 8) & 0x7f,
      predicate: !!(p & 1),
      opcodeName: PM4_OPCODE_BY_VALUE.get((p >>> 8) & 0x7f) ?? `UNKNOWN_0x${((p >>> 8) & 0x7f).toString(16).toUpperCase().padStart(2, '0')}`,
    };
  }

  /** VGT_DRAW_INITIATOR bitfield decode. */
  function parseVgtDrawInitiator(value) {
    const v = value >>> 0;
    return {
      value: v,
      primType: v & 0x3f,
      primTypeName: PRIMITIVE_TYPE_NAMES.get(v & 0x3f) ?? `prim_0x${(v & 0x3f).toString(16)}`,
      sourceSelect: (v >>> 6) & 0x3,
      majorMode: (v >>> 8) & 0x3,
      indexSize: (v >>> 11) & 0x1,
      notEop: !!((v >>> 12) & 0x1),
      numIndices: (v >>> 16) & 0xffff,
    };
  }

  /** GpuSwap for 32-bit values; endianness: 0=kNone, 1=k8in16, 2=k8in32, 3=k16in32. */
  function gpuSwap32(value, endianness) {
    const v = value >>> 0;
    switch (endianness & 0x3) {
      case 0: return v; // kNone
      case 1: return ((v << 8) & 0xff00ff00) | ((v >>> 8) & 0x00ff00ff); // k8in16
      case 2: return (((v & 0x000000ff) << 24) | ((v & 0x0000ff00) << 8) | ((v & 0x00ff0000) >>> 8) | ((v >>> 24) & 0x000000ff)) >>> 0; // k8in32
      case 3: return (((v >>> 16) & 0xffff) | (v << 16)) >>> 0; // k16in32
    }
    return v;
  }

  /** VGT_DMA_SIZE bitfield decode. */
  function parseVgtDmaSize(value) {
    const v = value >>> 0;
    return { value: v, numWords: v & 0x00ffffff, swapMode: (v >>> 30) & 0x3 };
  }

  const XenosPM4 = {
    PM4_OPCODE,
    PM4_OPCODE_BY_VALUE,
    CP_EVENT,
    SOURCE_SELECT,
    INDEX_FORMAT,
    MAJOR_MODE,
    PRIMITIVE_TYPE,
    PRIMITIVE_TYPE_NAMES,
    makePacketType0,
    makePacketType1,
    makePacketType2,
    makePacketType3,
    parsePacketHeader,
    parseVgtDrawInitiator,
    parseVgtDmaSize,
    gpuSwap32,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = XenosPM4;
  global.XenonPM4 = XenosPM4;
})(typeof self !== 'undefined' ? self : globalThis);
