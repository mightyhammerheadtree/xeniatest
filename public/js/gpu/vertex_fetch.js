/**
 * vertex_fetch.js — Xenos fetch-shader decoding into WebGPU-ready vertex
 * buffer layouts. Mirrors xenia (BSD-3-Clause):
 *
 *   - fetch constant decode  ← xenos.h  (xe_gpu_vertex_fetch_t)
 *   - ucode stream walking   ← shader_translator.cc (AnalyzeUcode /
 *                              GatherExecInformation / GatherVertexFetchInformation)
 *                              ucode.h   (CF pair unpacking, VertexFetchInstruction,
 *                              FetchOpcode, FetchDestinationSwizzle, VertexFormat)
 *
 * The Xenos microcode is: one or more control-flow pairs (2 instructions per
 * 3-dword pair) at the top of the blob, followed by ALU/fetch instructions
 * (3 dwords each). Control-flow `address` fields are *instruction indices*
 * (each instruction is 3 dwords). Each exec-family CF instruction carries a
 * 12-bit `sequence`, 2 bits per instruction: bit0 = fetch(1)/alu(0), bit1 =
 * serialize. We only walk exec-family CFs and we only decode fetch
 * instructions; ALU instructions are skipped.
 *
 * What this produces:
 *   - decodeVertexFetchConstants(): the 96 vertex fetch constants
 *     (2 dwords each: type/address, endian/size) read straight out of the
 *     register file (SHADER_CONSTANT_FETCH_*, 0x4800–0x48BF).
 *   - analyzeVertexShader(): bindings (grouped by fetch constant index, with
 *     dword stride) and attributes (dest register, offset, Xenos format,
 *     written mask, used components) from a vertex shader blob.
 *   - mapVertexFormat(): Xenos VertexFormat → GPUVertexFormat string; formats
 *     WebGPU cannot represent (11-11-10 floats, 32-bit unorm/snorm, 10-bit
 *     signed…) trap loudly instead of silently substituting a different
 *     numeric type.
 *   - buildGpuVertexBufferDescriptors(): the GPUVertexBufferLayout-shaped
 *     descriptors a future draw() call will consume (arrayStride, attributes
 *     with shaderLocation/offset/format). shaderLocation is **provisional**:
 *     the microcode has no location semantics, so the destination GPR number
 *     is used; it must match the @location(N) of the translated vertex
 *     shader when that lands.
 *
 * Honest limitations (by design, not oversight):
 *   - gpu formats that need per-vertex software conversion (R11G11B10,
 *     32-bit unorm/snorm, signed 2-10-10-10) are not encodable in
 *     GPUVertexFormat and trap. A conversion path can be added later.
 *   - exp_adjust != 0 on integer formats traps (hardware scales ints by
 *     2^exp_adjust; WebGPU has no equivalent conversion).
 *   - Bindings with stride == 0 ("fetch same location every vertex") are legal
 *     on Xenos but not expressible as a GPU vertex buffer layout; trap.
 *   - Predicated vfetch instructions are still included in layout inference —
 *     a conditionally-written attribute still needs a bound buffer.
 *   - used-component analysis follows xenia (lanes written by the dest
 *     swizzle); instructions writing zero lanes are skipped as attributes.
 */
(function (global) {
  'use strict';

  class XenosFetchError extends Error {}

  // ==========================================================================
  // Fetch constants (96 × 2 dwords, xe_gpu_vertex_fetch_t)
  // ==========================================================================

  const FETCH_CONSTANT_TYPE = Object.freeze([
    'invalidTexture', // 0
    'invalidVertex',  // 1
    'texture',        // 2
    'vertex',         // 3
  ]);

  const ENDIAN_NAMES = Object.freeze(['none', '8in16', '8in32', '16in32']);

  /**
   * @param {(dwordIndex: number) => number} readDword — read register-file
   *   dword at offset 0x4800 + index (0..191), host order.
   * @returns {Array<object>} 96 fetch constants.
   */
  function decodeVertexFetchConstants(readDword) {
    if (typeof readDword !== 'function') throw new TypeError('decodeVertexFetchConstants requires readDword(index) => u32');
    const out = new Array(96);
    for (let i = 0; i < 96; i++) {
      const dword0 = readDword(i * 2) >>> 0;
      const dword1 = readDword(i * 2 + 1) >>> 0;
      const type = dword0 & 0x3;
      // xenia's xe_gpu_vertex_fetch_t:
      //   FetchConstantType type : 2;   // +0
      //   uint32_t address : 30;        // +2 address in dwords
      // The guest address is "vfetch_constant.address << 2" (xenia
      // d3d12_command_processor.cc: shared_memory_->RequestRange(
      // vfetch_constant.address << 2, vfetch_constant.size << 2)), so the
      // address field counts *dwords* and has to be shifted left by two to
      // become a byte address. It is kept inside the 29-bit Xenos physical
      // address space (512 MiB).
      const addressDwords = (dword0 >>> 2) & 0x07ffffff;
      out[i] = {
        index: i,
        type,
        typeName: FETCH_CONSTANT_TYPE[type],
        /** Byte address of the buffer in the guest's physical memory (29 bits, dword aligned). */
        baseAddress: (addressDwords << 2) >>> 0,
        /** GpuSwap endianness of the buffer data (Endian: 0 none, 1 k8in16, 2 k8in32, 3 k16in32). */
        endian: dword1 & 0x3,
        endianName: ENDIAN_NAMES[dword1 & 0x3],
        /** Buffer size in 32-bit words (24 bits). */
        sizeWords: (dword1 >>> 2) & 0x00FFFFFF,
      };
    }
    return out;
  }

  // ==========================================================================
  // Control flow + instruction stream walking
  // ==========================================================================

  const CF_K_NOP = 0;
  const CF_K_EXEC = 1;
  const CF_K_EXEC_END = 2;
  const CF_K_COND_EXEC = 3;
  const CF_K_COND_EXEC_END = 4;
  const CF_K_COND_EXEC_PRED = 5;
  const CF_K_COND_EXEC_PRED_END = 6;
  const CF_K_LOOP_START = 7;
  const CF_K_LOOP_END = 8;
  const CF_K_COND_CALL = 9;
  const CF_K_RETURN = 10;
  const CF_K_COND_JMP = 11;
  const CF_K_ALLOC = 12;
  const CF_K_COND_EXEC_PRED_CLEAN = 13;
  const CF_K_COND_EXEC_PRED_CLEAN_END = 14;
  const CF_K_MARK_VS_FETCH_DONE = 15;

  const EXEC_FAMILY = new Set([
    CF_K_EXEC, CF_K_EXEC_END,
    CF_K_COND_EXEC, CF_K_COND_EXEC_END,
    CF_K_COND_EXEC_PRED, CF_K_COND_EXEC_PRED_END,
    CF_K_COND_EXEC_PRED_CLEAN, CF_K_COND_EXEC_PRED_CLEAN_END,
  ]);

  // Xenos VertexFormat : uint32_t (xenos.h)
  const VERTEX_FORMAT = Object.freeze({
    kUndefined: 0,
    k_8_8_8_8: 6,
    k_2_10_10_10: 7,
    k_10_11_11: 16,
    k_11_11_10: 17,
    k_16_16: 25,
    k_16_16_16_16: 26,
    k_16_16_FLOAT: 31,
    k_16_16_16_16_FLOAT: 32,
    k_32: 33,
    k_32_32: 34,
    k_32_32_32_32: 35,
    k_32_FLOAT: 36,
    k_32_32_FLOAT: 37,
    k_32_32_32_32_FLOAT: 38,
    k_32_32_32_FLOAT: 57,
  });
  const VERTEX_FORMAT_NAME = new Map(Object.entries(VERTEX_FORMAT).map(([n, v]) => [v, n]));

  /** Component count per Xenos vertex format (xenos.h GetVertexFormatComponentCount). */
  const VERTEX_FORMAT_COMPONENTS = new Map([
    [VERTEX_FORMAT.k_32, 1], [VERTEX_FORMAT.k_32_FLOAT, 1],
    [VERTEX_FORMAT.k_16_16, 2], [VERTEX_FORMAT.k_16_16_FLOAT, 2],
    [VERTEX_FORMAT.k_32_32, 2], [VERTEX_FORMAT.k_32_32_FLOAT, 2],
    [VERTEX_FORMAT.k_10_11_11, 3], [VERTEX_FORMAT.k_11_11_10, 3],
    [VERTEX_FORMAT.k_32_32_32_FLOAT, 3],
    [VERTEX_FORMAT.k_8_8_8_8, 4], [VERTEX_FORMAT.k_2_10_10_10, 4],
    [VERTEX_FORMAT.k_16_16_16_16, 4], [VERTEX_FORMAT.k_16_16_16_16_FLOAT, 4],
    [VERTEX_FORMAT.k_32_32_32_32, 4], [VERTEX_FORMAT.k_32_32_32_32_FLOAT, 4],
  ]);

  /** Bytes per component for the integer/float vertex formats (float16 = 2). */
  function vertexFormatBytes(format) {
    switch (format) {
      case VERTEX_FORMAT.k_8_8_8_8: return 4;
      case VERTEX_FORMAT.k_2_10_10_10: return 4;
      case VERTEX_FORMAT.k_10_11_11: return 4;
      case VERTEX_FORMAT.k_11_11_10: return 4;
      case VERTEX_FORMAT.k_16_16: return 4;
      case VERTEX_FORMAT.k_16_16_FLOAT: return 4;
      case VERTEX_FORMAT.k_16_16_16_16: return 8;
      case VERTEX_FORMAT.k_16_16_16_16_FLOAT: return 8;
      case VERTEX_FORMAT.k_32: return 4;
      case VERTEX_FORMAT.k_32_32: return 8;
      case VERTEX_FORMAT.k_32_32_32_32: return 16;
      case VERTEX_FORMAT.k_32_FLOAT: return 4;
      case VERTEX_FORMAT.k_32_32_FLOAT: return 8;
      case VERTEX_FORMAT.k_32_32_32_32_FLOAT: return 16;
      case VERTEX_FORMAT.k_32_32_32_FLOAT: return 12;
      default: return 0;
    }
  }

  function signExtend(value, bits) {
    const shift = 32 - bits;
    return (value << shift) >> shift;
  }

  /** Unpack the 3-dword CF pair (dwords[i*3..i*3+2]) into the two CF instructions. */
  function unpackControlFlowPair(dwords, pairIndex) {
    const d0 = dwords[pairIndex * 3] >>> 0;
    const d1 = dwords[pairIndex * 3 + 1] >>> 0;
    const d2 = dwords[pairIndex * 3 + 2] >>> 0;
    return [
      { dword0: d0, dword1: d1 & 0xFFFF },
      { dword0: ((d1 >>> 16) | (d2 << 16)) >>> 0, dword1: (d2 >>> 16) & 0xFFFF },
    ];
  }

  /** Exec-family CF decode: opcode/address/count/sequence (layouts shared by all exec opcodes). */
  function decodeExecCf(cf) {
    // Xenia layout (ucode.h): word1 (16 bits) = vc_lo:2, [bool_address:8 for
    // kCondExec], is_predicate_clean/condition:1 @9/10, address_mode:1 @11,
    // opcode:4 @12.
    return {
      opcode: (cf.dword1 >>> 12) & 0xF,
      address: cf.dword0 & 0xFFF,
      count: (cf.dword0 >>> 12) & 0x7,
      isYield: (cf.dword0 >>> 15) & 1,
      sequence: (cf.dword0 >>> 16) & 0xFFF,
      isPredicateClean: (cf.dword1 >>> 9) & 1,
      addressingMode: (cf.dword1 >>> 11) & 1,
    };
  }

  /** VertexFetchInstruction decode (3 dwords). */
  function decodeVertexFetchInstruction(dwords, instructionIndex) {
    const base = instructionIndex * 3;
    const w0 = dwords[base] >>> 0;
    const w1 = dwords[base + 1] >>> 0;
    const w2 = dwords[base + 2] >>> 0;
    const constIndex = (w0 >>> 20) & 0x1F;
    const constIndexSel = (w0 >>> 25) & 0x3;
    const dstSwizzle = w1 & 0xFFF;
    // Lanes written by the dest swizzle (kKeep = 7 = don't write).
    let writtenMask = 0;
    for (let lane = 0; lane < 4; lane++) {
      const swizzle = (dstSwizzle >>> (3 * lane)) & 0x7;
      if (swizzle !== 0b111) writtenMask |= 1 << lane;
    }
    // Distinct source lanes among written lanes (for used-component analysis).
    let usedSourceComponents = 0;
    for (let lane = 0; lane < 4; lane++) {
      if (!(writtenMask & (1 << lane))) continue;
      const swizzle = (dstSwizzle >>> (3 * lane)) & 0x7;
      if (swizzle <= 0b011) usedSourceComponents |= 1 << swizzle;
    }
    return {
      instructionIndex,
      opcode: w0 & 0x1F, // 0 = kVertexFetch (other values = texture fetch; caller filters)
      srcReg: (w0 >>> 5) & 0x3F,
      srcRegRelative: (w0 >>> 11) & 1,
      destReg: (w0 >>> 12) & 0x3F,
      destRegRelative: (w0 >>> 18) & 1,
      constIndex,
      constIndexSel,
      fetchConstantIndex: constIndex * 3 + constIndexSel, // 0..95
      prefetchCount: (w0 >>> 27) & 0x7,
      srcSwizzle: (w0 >>> 30) & 0x3,
      destSwizzle: dstSwizzle,
      isSigned: (w1 >>> 12) & 1,
      isNormalized: ((w1 >>> 13) & 1) === 0, // num_format_all == 0 → normalized
      signedRFMode: (w1 >>> 14) & 1,
      isIndexRounded: (w1 >>> 15) & 1,
      format: (w1 >>> 16) & 0x3F,
      expAdjust: signExtend((w1 >>> 24) & 0x3F, 6),
      isMiniFetch: (w1 >>> 30) & 1,
      isPredicated: (w1 >>> 31) & 1,
      writtenMask,
      usedSourceComponents,
      strideWords: w2 & 0xFF,
      offsetWords: signExtend((w2 >>> 8) & 0x7FFFFF, 23),
      predicateCondition: (w2 >>> 31) & 1,
    };
  }

  const FETCH_OPCODE_VERTEX = 0;

  /**
   * Walk a vertex shader blob and gather vertex fetch bindings.
   * @param {Uint32Array | number[]} ucodeWords — host-order dwords of the
   *   vertex shader microcode (as loaded by IM_LOAD).
   */
  function analyzeVertexShader(ucodeWords) {
    const dwords = ucodeWords instanceof Uint32Array ? ucodeWords : Uint32Array.from(ucodeWords, (w) => w >>> 0);
    if (dwords.length % 3 !== 0) {
      throw new XenosFetchError(`vertex shader ucode length ${dwords.length} dwords is not a multiple of 3`);
    }
    const instructionBound = Math.floor(dwords.length / 3);
    // Pass 1: find the CF-section upper bound = min exec CF address. The bound
    // SHRINKS live (xenia AnalyzeUcode does the same): instruction words beyond
    // the CF section can alias an exec-family opcode and must never be read as
    // control flow — with a fixed loop bound they would corrupt the bound and
    // silently drop real fetches at the top of the blob.
    let cfPairBound = instructionBound;
    for (let i = 0; i < cfPairBound; i++) {
      const [a, b] = unpackControlFlowPair(dwords, i);
      for (const cf of [a, b]) {
        const opcode = decodeExecCf(cf).opcode;
        if (EXEC_FAMILY.has(opcode)) {
          cfPairBound = Math.min(cfPairBound, decodeExecCf(cf).address);
        }
      }
    }
    // Pass 2: execute exec CFs in CF order, collecting vfetch instructions.
    /** @type {Array<object>} */
    const vfetches = [];
    /** previous vfetch_full, for vfetch_mini inheritance (xenia semantics) */
    let previousVfetchFull = null;
    for (let i = 0; i < cfPairBound; i++) {
      const pair = unpackControlFlowPair(dwords, i);
      for (const cf of pair) {
        const opcode = decodeExecCf(cf).opcode;
        if (!EXEC_FAMILY.has(opcode)) continue;
        const exec = decodeExecCf(cf);
        for (let k = 0; k < exec.count; k++) {
          const instructionIndex = exec.address + k;
          if (instructionIndex >= instructionBound) {
            throw new XenosFetchError(`exec CF at pair ${i}: instruction index ${instructionIndex} out of shader (${instructionBound} instructions)`);
          }
          const seq = (exec.sequence >>> (2 * k)) & 0b11;
          if (!(seq & 0b01)) continue; // ALU instruction
          const fetchOp = decodeVertexFetchInstruction(dwords, instructionIndex);
          if (fetchOp.opcode === FETCH_OPCODE_VERTEX) {
            if (fetchOp.isMiniFetch && !previousVfetchFull) {
              throw new XenosFetchError(`vfetch_mini at instruction ${instructionIndex} has no preceding vfetch_full (xenia inherits stride/index rounding from it)`);
            }
            fetchOp.full = fetchOp.isMiniFetch ? previousVfetchFull : fetchOp;
            vfetches.push(fetchOp);
            if (!fetchOp.isMiniFetch) previousVfetchFull = fetchOp;
          }
          // Texture fetches: skipped (handled by the texture path).
        }
      }
    }

    // Group into bindings by the instruction's own fetch constant index
    // (xenia GatherVertexFetchInformation semantics).
    /** @type {Map<number, object>} */
    const bindingMap = new Map();
    for (const op of vfetches) {
      if (op.writtenMask === 0) continue; // fetches nothing (xenia GetUsedResultComponents()==0)
      const full = op.full ?? op;
      let binding = bindingMap.get(op.fetchConstantIndex);
      if (!binding) {
        binding = {
          bindingIndex: bindingMap.size,
          fetchConstant: op.fetchConstantIndex,
          strideWords: full.strideWords,
          attributes: [],
        };
        bindingMap.set(op.fetchConstantIndex, binding);
      }
      binding.attributes.push({
        instructionIndex: op.instructionIndex,
        destRegister: op.destReg,
        destRegisterRelative: op.destRegRelative,
        writtenMask: op.writtenMask,
        usedSourceComponents: op.usedSourceComponents,
        offsetWords: op.offsetWords,
        byteOffset: op.offsetWords * 4,
        format: op.format,
        formatName: VERTEX_FORMAT_NAME.get(op.format) ?? `unknown_${op.format}`,
        componentCount: VERTEX_FORMAT_COMPONENTS.get(op.format) ?? 0,
        isSigned: !!op.isSigned,
        isNormalized: !!op.isNormalized,
        signedRFMode: op.signedRFMode,
        expAdjust: op.expAdjust,
        isIndexRounded: !!op.isIndexRounded,
        isPredicated: !!op.isPredicated,
        isMiniFetch: !!op.isMiniFetch,
      });
    }

    return {
      ucodeInstructionBound: instructionBound,
      controlFlowPairBound: cfPairBound,
      vertexFetchInstructionCount: vfetches.length,
      bindings: [...bindingMap.values()],
    };
  }

  // ==========================================================================
  // Xenos VertexFormat → GPUVertexFormat
  // ==========================================================================

  /**
   * Map a Xenos vertex format to a GPUVertexFormat string.
   * Throws XenosFetchError when WebGPU cannot represent the format — never
   * silently reinterprets the bits.
   */
  function mapVertexFormat(attr) {
    const { format, isSigned, isNormalized, expAdjust } = attr;
    const name = VERTEX_FORMAT_NAME.get(format);
    if (!name) throw new XenosFetchError(`mapVertexFormat: unknown Xenos vertex format ${format}`);
    const intLike = format !== VERTEX_FORMAT.k_32_FLOAT && format !== VERTEX_FORMAT.k_32_32_FLOAT &&
      format !== VERTEX_FORMAT.k_32_32_32_FLOAT && format !== VERTEX_FORMAT.k_32_32_32_32_FLOAT &&
      format !== VERTEX_FORMAT.k_16_16_FLOAT && format !== VERTEX_FORMAT.k_16_16_16_16_FLOAT;
    if (intLike && expAdjust !== 0) {
      throw new XenosFetchError(`mapVertexFormat: ${name} with exp_adjust=${expAdjust} — Xenos scales integer fetches by 2^exp_adjust; WebGPU has no equivalent and silently changing it would corrupt data`);
    }
    switch (format) {
      case VERTEX_FORMAT.k_32_FLOAT: return 'float32';
      case VERTEX_FORMAT.k_32_32_FLOAT: return 'float32x2';
      case VERTEX_FORMAT.k_32_32_32_FLOAT: return 'float32x3';
      case VERTEX_FORMAT.k_32_32_32_32_FLOAT: return 'float32x4';
      case VERTEX_FORMAT.k_16_16_FLOAT: return 'float16x2';
      case VERTEX_FORMAT.k_16_16_16_16_FLOAT: return 'float16x4';
      case VERTEX_FORMAT.k_32: return isNormalized ? bitTrap(name, '32-bit unorm/snorm') : isSigned ? 'sint32' : 'uint32';
      case VERTEX_FORMAT.k_32_32: return isNormalized ? bitTrap(name, '32-bit unorm/snorm') : isSigned ? 'sint32x2' : 'uint32x2';
      case VERTEX_FORMAT.k_32_32_32_32: return isNormalized ? bitTrap(name, '32-bit unorm/snorm') : isSigned ? 'sint32x4' : 'uint32x4';
      case VERTEX_FORMAT.k_16_16:
        if (isNormalized) return isSigned ? 'snorm16x2' : 'unorm16x2';
        return isSigned ? 'sint16x2' : 'uint16x2';
      case VERTEX_FORMAT.k_16_16_16_16:
        if (isNormalized) return isSigned ? 'snorm16x4' : 'unorm16x4';
        return isSigned ? 'sint16x4' : 'uint16x4';
      case VERTEX_FORMAT.k_8_8_8_8:
        if (isNormalized) return isSigned ? 'snorm8x4' : 'unorm8x4';
        return isSigned ? 'sint8x4' : 'uint8x4';
      case VERTEX_FORMAT.k_2_10_10_10:
        if (isNormalized && !isSigned) return 'unorm10-10-10-2';
        return bitTrap(name, 'only unsigned normalized 2-10-10-10 maps to a WebGPU vertex format (unorm10-10-10-2); signed or integer 2-10-10-10 has no GPUVertexFormat');
      case VERTEX_FORMAT.k_10_11_11:
      case VERTEX_FORMAT.k_11_11_10:
        return bitTrap(name, 'R11G11B10-style float formats have no GPUVertexFormat; a software expansion pass is required');
      default:
        return bitTrap(name, 'no WebGPU vertex format mapping');
    }
  }
  function bitTrap(name, why) {
    throw new XenosFetchError(`mapVertexFormat: ${name}: ${why}`);
  }

  /**
   * Build GPUVertexBufferLayout-shaped descriptors from an analysis result.
   * @returns {Array<object>} one descriptor per binding (in first-use order).
   */
  function buildGpuVertexBufferDescriptors(analysis) {
    const descriptors = [];
    const usedLocations = new Set();
    for (const binding of analysis.bindings) {
      if (binding.strideWords === 0) {
        throw new XenosFetchError(`vertex binding (fetch constant ${binding.fetchConstant}) has stride 0 — "fetch the same location" is legal Xenos but not expressible as a GPU vertex buffer; this draw needs a dedicated path`);
      }
      const arrayStride = binding.strideWords * 4;
      const attributes = [];
      for (const attr of binding.attributes) {
        if (attr.byteOffset < 0) {
          throw new XenosFetchError(`attribute at ucode instruction ${attr.instructionIndex}: negative offset ${attr.offsetWords} dwords is not expressible in WebGPU`);
        }
        const format = mapVertexFormat(attr); // traps when unrepresentable
        const size = vertexFormatBytes(attr.format);
        if (attr.byteOffset + size > arrayStride) {
          throw new XenosFetchError(`attribute at ucode instruction ${attr.instructionIndex}: offset ${attr.byteOffset} + ${size} bytes exceeds stride ${arrayStride}`);
        }
        if (usedLocations.has(attr.destRegister)) {
          throw new XenosFetchError(`attributes writing GPR${attr.destRegister} appear in more than one binding attribute — shaderLocation collision (shaderLocation is provisional: dest GPR)`);
        }
        usedLocations.add(attr.destRegister);
        attributes.push({
          shaderLocation: attr.destRegister, // provisional — must match the translated VS @location(N)
          offset: attr.byteOffset,
          format,
        });
      }
      attributes.sort((a, b) => a.offset - b.offset);
      descriptors.push({
        bindingIndex: binding.bindingIndex,
        fetchConstant: binding.fetchConstant,
        arrayStride,
        stepMode: 'vertex',
        attributes,
      });
    }
    return descriptors;
  }

  // Internals shared with the WGSL translator (wgsl_translate.js). Not public
  // API — the shapes are unstable. Kept in one place so both walkers agree on
  // CF-pair unpacking and exec decoding.
  const _internal = {
    CF_K_EXEC,
    CF_K_EXEC_END,
    CF_K_COND_EXEC,
    CF_K_COND_EXEC_END,
    CF_K_COND_EXEC_PRED,
    CF_K_COND_EXEC_PRED_END,
    CF_K_LOOP_START,
    CF_K_LOOP_END,
    CF_K_COND_CALL,
    CF_K_RETURN,
    CF_K_COND_JMP,
    CF_K_ALLOC,
    CF_K_COND_EXEC_PRED_CLEAN,
    CF_K_COND_EXEC_PRED_CLEAN_END,
    CF_K_MARK_VS_FETCH_DONE,
    EXEC_FAMILY,
    unpackControlFlowPair,
    decodeExecCf,
  };

  const api = {
    XenosFetchError,
    VERTEX_FORMAT,
    FETCH_CONSTANT_TYPE,
    decodeVertexFetchConstants,
    analyzeVertexShader,
    mapVertexFormat,
    buildGpuVertexBufferDescriptors,
    vertexFormatBytes,
    _internal,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonGpuVertexFetch = api;
})(typeof self !== 'undefined' ? self : globalThis);
