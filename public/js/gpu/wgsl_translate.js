/**
 * Xenos ucode → WGSL translator (v1: vertex shaders).
 *
 * Translates vertex-shader microcode (fetch instructions + the scalar/vector
 * ALU subset Xenos vertex shaders are built from) into a `@vertex` WGSL module:
 *
 *   - `@location(N)` inputs named after the fetch dest GPR (the
 *     `shaderLocation = dest GPR` contract the fetch path already uses);
 *   - `@builtin(position)` from kVSPosition exports; other exports →
 *     `@location(M)` interpolator outputs (0–15);
 *   - `r[N]` = the guest vector GPR array (zero-initialized, as xenia does, so
 *     conditionally-written registers read deterministically);
 *   - `ps` = the guest previous-scalar register, `p0` = the hardware predicate;
 *   - float/bool constant windows bound only when used (`fc` / `bools`).
 *
 * Supported: adds/subs/muls (+_prev), maxs/mins, seqs/sgts/sges/snes, frcs/
 * truncs/floors, maxs/mins-select semantics with Shader Model 3 NaN behavior;
 * setp_eq/ne/gt/ge/inv/pop/clr/rstr (write `ps` and `p0`); retain_prev;
 * vector add/mul/max/min/seq/sgt/sge/sne/frc/trunc/floor/mad/cndeq/cndge/
 * cndgt/dp4/dp3/dp2add/max4/dst/setp_eq_push/setp_ne_push/setp_gt_push/
 * setp_ge_push; exports with dual-issue vector+scalar lane masks (constant
 * 0/1 lanes), dest-mask, clamp, abs/neg, per-lane source swizzles;
 * cexec/cexece → `if (bc(Nu))` / `if (!bc(Nu))`, predicated instructions →
 * `if (p0)` / `if (!p0)`, predicated exec → same.
 *
 * Traps loudly (XenosTranslateError) with the CF pair / instruction index on
 * anything outside the subset: loops, calls, jumps, texture fetches, memory
 * exports, point-size/edge/kill-vertex exports, pixel ops (kills), a0/+aL
 * relative addressing, register-relative dests, vector dest GPR 30–31, scalar
 * dest GPR ≥ 32, unknown opcodes, and Shader Model 3 transcendental math
 * (exp/log/rcp/rsq/sin/cos/sqrt) whose clamped-IEEE semantics cannot be
 * reproduced without GLSL.std.450/precise control flow — nothing is silently
 * miscompiled.
 *
 * Shader Model 3 zero/denormal × 0 → ±0 folding for mul is not reproduced
 * (WGSL gives no control over denormals); plain mul is emitted instead.
 * max/min use `select(a >= b, a, b)` exactly like xenia (not NMax/NMin), so
 * NaN behavior matches the guest.
 *
 * Every shader asserted in core/tests/test_wgsl_translate.mjs is compiled by
 * Dawn (the real WebGPU compiler) via core/tests/helpers/dawn_validate.mjs —
 * parse AND semantic errors, not just structure.
 */
(function (global) {
'use strict';

const XenosTranslateError = class extends Error {
  constructor(message) {
    super(message);
    this.name = 'XenosTranslateError';
  }
};

// Fetch the CF-pair decoders shared with vertex_fetch.js so both walkers agree
// on the bit layout. Loaded lazily at translate time (load order:
// vertex_fetch.js first).
function fetchInternal() {
  const api = global.XenonGpuVertexFetch;
  if (!api || !api._internal) {
    throw new XenosTranslateError('wgsl_translate.js requires vertex_fetch.js to be loaded first');
  }
  return api._internal;
}

// ---------------------------------------------------------------- op tables
const ALU_VECTOR_OPS = Object.freeze({
  0: 'add', 1: 'mul', 2: 'max', 3: 'min', 4: 'seq', 5: 'sgt', 6: 'sge', 7: 'sne',
  8: 'frc', 9: 'trunc', 10: 'floor', 11: 'mad', 12: 'cndeq', 13: 'cndge', 14: 'cndgt',
  15: 'dp4', 16: 'dp3', 17: 'dp2add', 19: 'max4', 20: 'setp_eq_push', 21: 'setp_ne_push',
  22: 'setp_gt_push', 23: 'setp_ge_push', 28: 'dst',
});
const ALU_VECTOR_TRAPS = Object.freeze({
  18: 'cube', 24: 'kill_eq', 25: 'kill_gt', 26: 'kill_ge', 27: 'kill_ne',
  29: 'maxa', 30: 'opcode_30', 31: 'opcode_31',
});
// Pixel shaders additionally support the kill ops (opcode 24–27), which
// demote the invocation (discard) with an any-lane compare and write 0.0.
const ALU_VECTOR_OPS_PIXEL = Object.freeze({
  ...ALU_VECTOR_OPS,
  24: 'kill_eq', 25: 'kill_gt', 26: 'kill_ge', 27: 'kill_ne',
});
const ALU_VECTOR_TRAPS_PIXEL = Object.freeze({
  18: 'cube',
  29: 'maxa', 30: 'opcode_30', 31: 'opcode_31',
});
const ALU_SCALAR_OPS = Object.freeze({
  0: 'adds', 1: 'adds_prev', 2: 'muls', 3: 'muls_prev', 5: 'maxs', 6: 'mins',
  7: 'seqs', 8: 'sgts', 9: 'sges', 10: 'snes', 11: 'frcs', 12: 'truncs', 13: 'floors',
  25: 'subs', 26: 'subs_prev', 27: 'setp_eq', 28: 'setp_ne', 29: 'setp_gt', 30: 'setp_ge',
  31: 'setp_inv', 32: 'setp_pop', 33: 'setp_clr', 34: 'setp_rstr', 50: 'retain_prev',
});
const ALU_SCALAR_TRAPS = Object.freeze({
  4: 'muls_prev2', 14: 'exp', 15: 'logc', 16: 'log', 17: 'rcpc', 18: 'rcpf', 19: 'rcp',
  20: 'rsqc', 21: 'rsqf', 22: 'rsq', 23: 'maxas', 24: 'maxasf', 35: 'kills_eq',
  36: 'kills_gt', 37: 'kills_ge', 38: 'kills_ne', 39: 'kills_one', 40: 'sqrt',
  41: 'opcode_41', 42: 'mulsc', 43: 'mulsc', 44: 'addsc', 45: 'addsc', 46: 'subsc',
  47: 'subsc', 48: 'sin', 49: 'cos',
  51: 'opcode_51', 52: 'opcode_52', 53: 'opcode_53', 54: 'opcode_54', 55: 'opcode_55',
  56: 'opcode_56', 57: 'opcode_57', 58: 'opcode_58', 59: 'opcode_59', 60: 'opcode_60',
  61: 'opcode_61', 62: 'opcode_62', 63: 'opcode_63',
});
// Pixel shaders additionally support the scalar kill ops (opcode 35–39):
// demote the invocation on a comparison against 0.0 (1.0 for kills_one),
// lane W of the third source, writing 0.0.
const ALU_SCALAR_OPS_PIXEL = Object.freeze({
  ...ALU_SCALAR_OPS,
  35: 'kills_eq', 36: 'kills_gt', 37: 'kills_ge', 38: 'kills_ne', 39: 'kills_one',
});
const ALU_SCALAR_TRAPS_PIXEL = Object.freeze({
  4: 'muls_prev2', 14: 'exp', 15: 'logc', 16: 'log', 17: 'rcpc', 18: 'rcpf', 19: 'rcp',
  20: 'rsqc', 21: 'rsqf', 22: 'rsq', 23: 'maxas', 24: 'maxasf', 40: 'sqrt',
  41: 'opcode_41', 42: 'mulsc', 43: 'mulsc', 44: 'addsc', 45: 'addsc', 46: 'subsc',
  47: 'subsc', 48: 'sin', 49: 'cos',
  51: 'opcode_51', 52: 'opcode_52', 53: 'opcode_53', 54: 'opcode_54', 55: 'opcode_55',
  56: 'opcode_56', 57: 'opcode_57', 58: 'opcode_58', 59: 'opcode_59', 60: 'opcode_60',
  61: 'opcode_61', 62: 'opcode_62', 63: 'opcode_63',
});
/** Op tables for a shader type. */
function opcodeTablesFor(shaderType) {
  return shaderType === 'pixel'
    ? { vectorOps: ALU_VECTOR_OPS_PIXEL, vectorTraps: ALU_VECTOR_TRAPS_PIXEL, scalarOps: ALU_SCALAR_OPS_PIXEL, scalarTraps: ALU_SCALAR_TRAPS_PIXEL }
    : { vectorOps: ALU_VECTOR_OPS, vectorTraps: ALU_VECTOR_TRAPS, scalarOps: ALU_SCALAR_OPS, scalarTraps: ALU_SCALAR_TRAPS };
}

// ---------------------------------------------------------------- decode
const SEQ_FETCH = 0b01;

/** Swizzled source component index (component-relative swizzles, xenia semantics). */
function swizzledComponent(swizzle, componentIndex) {
  return ((swizzle >>> (2 * componentIndex)) + componentIndex) & 3;
}

/** Decode one ALU instruction (3 dwords). */
function decodeAluInstruction(dwords, instructionIndex, shaderType) {
  const tables = opcodeTablesFor(shaderType);
  const base = instructionIndex * 3;
  const w0 = dwords[base] >>> 0, w1 = dwords[base + 1] >>> 0, w2 = dwords[base + 2] >>> 0;
  const swz = [(w1 >>> 16) & 0xFF, (w1 >>> 8) & 0xFF, w1 & 0xFF];
  const neg = [(w1 >>> 26) & 1, (w1 >>> 25) & 1, (w1 >>> 24) & 1];
  const sel = [(w2 >>> 31) & 1, (w2 >>> 30) & 1, (w2 >>> 29) & 1];
  const reg = [(w2 >>> 16) & 0xFF, (w2 >>> 8) & 0xFF, w2 & 0xFF];
  // Relative-addressing bits (unsupported subset).
  const constAddressRegisterRelative = (w1 >>> 29) & 1;
  const const0RelAbs = (w1 >>> 30) & 1;
  const const1RelAbs = (w1 >>> 31) & 1;
  const absConstants = (w0 >>> 7) & 1;

  const operands = [0, 1, 2].map((i) => {
    if (sel[i]) {
      const t = reg[i];
      return {
        storageSource: 'register',
        index: t & 0x3F,
        aLRelative: (t & 0x40) !== 0,
        isAbsoluteValue: (t & 0x80) !== 0,
        swizzle: swz[i],
        isNegated: neg[i] === 1,
        isConstant0RelativeAddressed: false,
      };
    }
    return {
      storageSource: 'constant',
      index: reg[i],
      aLRelative: false,
      isAbsoluteValue: absConstants === 1,
      swizzle: swz[i],
      isNegated: neg[i] === 1,
      isConstant0RelativeAddressed: i === 0 ? const0RelAbs === 1 : i === 1 ? (sel[0] ? const0RelAbs : const1RelAbs) === 1 : (sel[0] && sel[1] ? const0RelAbs : const1RelAbs) === 1,
    };
  });
  if (constAddressRegisterRelative && operands.some((o) => o.storageSource === 'constant')) {
    throw new XenosTranslateError(
      `alu[${instructionIndex}]: constants addressed by a0 are not supported (register-relative constant addressing)`);
  }
  for (const op of operands) {
    if (op.isConstant0RelativeAddressed) {
      throw new XenosTranslateError(
        `alu[${instructionIndex}]: aL-relative constant addressing is not supported`);
    }
    if (op.aLRelative) {
      throw new XenosTranslateError(
        `alu[${instructionIndex}]: +aL register-relative temporary addressing is not supported`);
    }
  }

  const vectorOpcode = (w2 >>> 24) & 0x1F;
  const scalarOpcode = (w0 >>> 26) & 0x3F;
  const isExport = ((w0 >>> 15) & 1) === 1;
  const vectorDestRelative = ((w0 >>> 6) & 1) === 1;
  const scalarDestRelative = ((w0 >>> 14) & 1) === 1;
  const vectorWriteMask = (w0 >>> 16) & 0xF;
  const scalarWriteMask = (w0 >>> 20) & 0xF;

  // Resolve combined export masks exactly like ucode.h:
  // vector writes = export ? vector & ~scalar : vector; scalar writes = export ?
  // scalar & ~vector : scalar; const 0 lanes (export + scalar_dest_rel + neither
  // mask) write 0; lanes with both masks write 1.
  const vectorOpResultWriteMask = isExport ? vectorWriteMask & ~scalarWriteMask : vectorWriteMask;
  const scalarOpResultWriteMask = isExport ? scalarWriteMask & ~vectorWriteMask : scalarWriteMask;
  const constant0WriteMask = isExport && scalarDestRelative ? ~(vectorWriteMask | scalarWriteMask) & 0xF : 0;
  const constant1WriteMask = isExport ? vectorWriteMask & scalarWriteMask : 0;

  return {
    instructionIndex,
    isExport,
    vectorOpcode,
    scalarOpcode,
    vectorOpName: tables.vectorOps[vectorOpcode] ?? null,
    scalarOpName: tables.scalarOps[scalarOpcode] ?? null,
    vectorDest: w0 & 0x3F,
    vectorDestRelative,
    scalarDest: (w0 >>> 8) & 0x3F,
    scalarDestRelative,
    vectorWriteMask,
    scalarWriteMask,
    vectorOpResultWriteMask,
    scalarOpResultWriteMask,
    constant0WriteMask,
    constant1WriteMask,
    vectorClamp: ((w0 >>> 24) & 1) === 1,
    scalarClamp: ((w0 >>> 25) & 1) === 1,
    absConstants: absConstants === 1,
    isPredicated: ((w1 >>> 28) & 1) === 1,
    predicateCondition: ((w1 >>> 27) & 1) === 1,
    operands,
    vectorDestTarget: null, // filled by resolveTargets (needs isExport)
    scalarDestTarget: null,
  };
}

// ---------------------------------------------------------------- decode walk
function decodeCondExec(cf) {
  // Xenia CondExecInstruction word1 layout: bool_address @2–9, condition @10.
  return { kind: 'bool', index: (cf.dword1 >>> 2) & 0xFF, condition: ((cf.dword1 >>> 10) & 1) !== 0 };
}
function decodeCondExecPred(cf) {
  // Xenia CondExecPredInstruction word1 layout: condition @10.
  return { kind: 'pred', condition: ((cf.dword1 >>> 10) & 1) !== 0 };
}

/**
 * @returns {{blocks: Array<object>, instructionBound: number, cfPairBound: number,
 *            inputLocations: number[], interpolatorLocations: number[]}}
 * blocks: [{ condition: {kind:'bool',index,condition}|{kind:'pred',condition}|null,
 *            instructions: [{kind:'vfetch',instructionIndex} | {kind:'alu', decoded}] }]
 */
function decodeShader(dwords, shaderType) {
  const FW = fetchInternal();
  if (dwords.length % 3 !== 0) {
    throw new XenosTranslateError(`shader ucode length ${dwords.length} dwords is not a multiple of 3`);
  }
  const instructionBound = Math.floor(dwords.length / 3);
  let cfPairBound = instructionBound;
  // Live-shrinking bound, exactly like xenia's cf_pair_index_bound_: instruction
  // words beyond the CF section must NOT be re-read as CF pairs (they can
  // alias an exec opcode and corrupt the bound).
  for (let i = 0; i < cfPairBound; i++) {
    const [a, b] = FW.unpackControlFlowPair(dwords, i);
    for (const cf of [a, b]) {
      if (FW.EXEC_FAMILY.has(FW.decodeExecCf(cf).opcode)) {
        cfPairBound = Math.min(cfPairBound, FW.decodeExecCf(cf).address);
      }
    }
  }

  const blocks = [];
  const inputSet = new Set();
  const interpolatorSet = new Set();
  for (let i = 0; i < cfPairBound; i++) {
    const pair = FW.unpackControlFlowPair(dwords, i);
    for (const cf of pair) {
      const opcode = FW.decodeExecCf(cf).opcode;
      let condition = null;
      switch (opcode) {
        case 0: // CF nop
        case FW.CF_K_MARK_VS_FETCH_DONE:
        case FW.CF_K_ALLOC:
          continue;
        case FW.CF_K_EXEC:
        case FW.CF_K_EXEC_END:
          break;
        case FW.CF_K_COND_EXEC:
        case FW.CF_K_COND_EXEC_END:
          condition = decodeCondExec(cf);
          break;
        case FW.CF_K_COND_EXEC_PRED:
        case FW.CF_K_COND_EXEC_PRED_END:
        case FW.CF_K_COND_EXEC_PRED_CLEAN:
        case FW.CF_K_COND_EXEC_PRED_CLEAN_END:
          condition = decodeCondExecPred(cf);
          break;
        case FW.CF_K_LOOP_START:
          throw new XenosTranslateError(`cf[${i}]: loop_start is not supported yet (the WGSL translator cannot emit data-dependent loops without a loop-constant upload path)`);
        case FW.CF_K_LOOP_END:
          throw new XenosTranslateError(`cf[${i}]: loop_end is not supported yet`);
        case FW.CF_K_COND_CALL:
          throw new XenosTranslateError(`cf[${i}]: cond_call (subroutine calls) is not supported yet`);
        case FW.CF_K_RETURN:
          throw new XenosTranslateError(`cf[${i}]: return is not supported yet`);
        case FW.CF_K_COND_JMP:
          throw new XenosTranslateError(`cf[${i}]: cond_jmp (unstructured jumps) is not supported yet`);
        default:
          throw new XenosTranslateError(`cf[${i}]: unsupported control-flow opcode ${opcode}`);
      }
      const exec = FW.decodeExecCf(cf);
      const instructions = [];
      for (let k = 0; k < exec.count; k++) {
        const instructionIndex = exec.address + k;
        if (instructionIndex >= instructionBound) {
          throw new XenosTranslateError(`exec CF at pair ${i}: instruction index ${instructionIndex} out of shader (${instructionBound} instructions)`);
        }
        const sequence = (exec.sequence >>> (2 * k)) & 0b11;
        if (sequence & SEQ_FETCH) {
          const fetch = decodeFetchForAttribute(dwords, instructionIndex, shaderType);
          if (fetch.kind === 'tfetch') {
            throw new XenosTranslateError(shaderType === 'pixel'
              ? `fetch[${instructionIndex}]: texture fetch in a pixel shader is not supported yet (texture path is bring-up step 8)`
              : `fetch[${instructionIndex}]: texture fetch in a vertex shader is not supported (vertex shaders fetch attributes only)`);
          }
          if (fetch.kind === 'vfetch' && shaderType === 'pixel') {
            throw new XenosTranslateError(`fetch[${instructionIndex}]: a vertex fetch inside a pixel shader is not supported (pixel shaders fetch textures only)`);
          }
          if (fetch.writtenLanes !== 0) {
            if (fetch.destReg >= 16) {
              throw new XenosTranslateError(`fetch[${instructionIndex}]: attribute dest GPR ${fetch.destReg} cannot be bound to a vertex-attribute location (WebGPU maxVertexAttributes is 16)`);
            }
            inputSet.add(fetch.destReg);
          }
          instructions.push({ kind: 'vfetch', instructionIndex });
        } else {
          const decoded = decodeAluInstruction(dwords, instructionIndex, shaderType);
          resolveTargets(decoded, shaderType, interpolatorSet);
          instructions.push({ kind: 'alu', instructionIndex, decoded });
        }
      }
      blocks.push({ condition, instructions });
    }
  }
  if (blocks.length === 0 && instructionBound > 0) {
    // Either the CF bound collapsed to 0 (an exec CF pointed at slot 0 — impossible
    // in a well-formed blob: the CF pairs occupy slots before any instruction) or
    // there was no executable control flow at all. Emitting an empty module
    // would shade nothing, silently.
    throw new XenosTranslateError(`${shaderType} shader ucode: no executable control flow (malformed blob or missing exec CF)`);
  }
  return {
    blocks,
    instructionBound,
    cfPairBound,
    inputLocations: [...inputSet].sort((a, b) => a - b),
    interpolatorLocations: [...interpolatorSet].sort((a, b) => a - b),
  };
}

/** Decode a fetch instruction's dest lane mask (for attribute binding input). */
function decodeFetchForAttribute(dwords, instructionIndex, _shaderType) {
  const FW = fetchInternal();
  const base = instructionIndex * 3;
  const w0 = dwords[base] >>> 0, w1 = dwords[base + 1] >>> 0;
  const opcode = w0 & 0x1F;
  if (opcode !== 0) {
    return { kind: 'tfetch' };
  }
  const destSwizzle = w1 & 0xFFF;
  let writtenLanes = 0;
  for (let lane = 0; lane < 4; lane++) {
    if (((destSwizzle >>> (3 * lane)) & 0b111) !== 0b111) writtenLanes |= 1 << lane;
  }
  return { kind: 'vfetch', destReg: (w0 >>> 12) & 0x3F, writtenLanes };
}

/**
 * Resolve vector/scalar storage targets; records interpolator locations.
 * Traps on targets outside the v1 subset.
 */
function resolveTargets(decoded, shaderType, interpolatorSet) {
  const { instructionIndex } = decoded;
  // Vector destination (or export register).
  if (decoded.isExport) {
    decoded.vectorDestTarget = resolveExportTarget(decoded.vectorDest, shaderType, interpolatorSet);
    decoded.scalarDestTarget = decoded.vectorDestTarget; // exports go to vector_dest for both halves
  } else {
    if (decoded.vectorDest >= 30) {
      throw new XenosTranslateError(`alu[${instructionIndex}]: vector dest GPR ${decoded.vectorDest} is unassigned (only 0–29 exist)`);
    }
    if (decoded.scalarDest >= 32) {
      throw new XenosTranslateError(`alu[${instructionIndex}]: scalar dest GPR ${decoded.scalarDest} is unassigned (only 0–31 exist)`);
    }
    if (decoded.vectorDestRelative) {
      throw new XenosTranslateError(`alu[${instructionIndex}]: +aL register-relative vector destination addressing is not supported`);
    }
    if (decoded.scalarDestRelative) {
      throw new XenosTranslateError(`alu[${instructionIndex}]: +aL register-relative scalar destination addressing is not supported`);
    }
    decoded.vectorDestTarget = { kind: 'register', index: decoded.vectorDest };
    decoded.scalarDestTarget = { kind: 'register', index: decoded.scalarDest };
  }
}

function resolveExportTarget(vectorDest, shaderType, interpolatorSet) {
  if (shaderType === 'pixel') {
    // kPSColor0..3 / kPSDepth — xenia ExportRegister; games virtually always export oC0.
    if (vectorDest === 0) return { kind: 'color0' };
    if (vectorDest >= 1 && vectorDest <= 3) {
      throw new XenosTranslateError(`export: pixel color target ${vectorDest} (MRT) is not supported yet — write to oC0 only`);
    }
    if (vectorDest === 61) return { kind: 'oDepth' };
    if (vectorDest >= 32 && vectorDest <= 37) {
      throw new XenosTranslateError(`export: memory exports (eA/eM${vectorDest - 33}) are not supported yet`);
    }
    throw new XenosTranslateError(`export: unsupported pixel export register ${vectorDest}`);
  }
  if (vectorDest === 62) return { kind: 'position' };
  if (vectorDest >= 0 && vectorDest <= 15) {
    interpolatorSet.add(vectorDest);
    return { kind: 'interpolator', index: vectorDest };
  }
  if (vectorDest === 63) {
    throw new XenosTranslateError('export: VS point-size/edge-flag/kill-vertex exports are not supported yet');
  }
  if (vectorDest >= 32 && vectorDest <= 37) {
    throw new XenosTranslateError(`export: memory exports (eA/eM${vectorDest - 33}) are not supported yet`);
  }
  throw new XenosTranslateError(`export: unsupported export register ${vectorDest}`);
}

/** Trap on an opcode the subset does not support for this shader type. */
function assertOpcodeSupported(decoded, shaderType) {
  const { instructionIndex, vectorOpcode, scalarOpcode } = decoded;
  const tables = opcodeTablesFor(shaderType);
  const vectorTrap = tables.vectorTraps[vectorOpcode];
  if (vectorTrap) {
    if (vectorTrap === 'cube') throw new XenosTranslateError(`alu[${instructionIndex}]: cube is not supported yet`);
    if (vectorTrap === 'maxa') throw new XenosTranslateError(`alu[${instructionIndex}]: maxa (a0 write) is not supported yet`);
    if (vectorTrap.startsWith('kill_')) throw new XenosTranslateError(`alu[${instructionIndex}]: ${vectorTrap} is pixel-shader-only (kills do not exist in a vertex stage)`);
    throw new XenosTranslateError(`alu[${instructionIndex}]: vector opcode ${vectorOpcode} (${vectorTrap}) is invalid`);
  }
  const scalarTrap = tables.scalarTraps[scalarOpcode];
  if (scalarTrap) {
    if (scalarTrap === 'muls_prev2') throw new XenosTranslateError(`alu[${instructionIndex}]: muls_prev2 is not supported yet`);
    if (scalarTrap === 'maxas' || scalarTrap === 'maxasf') throw new XenosTranslateError(`alu[${instructionIndex}]: ${scalarTrap} (a0 write) is not supported yet`);
    if (scalarTrap.startsWith('kills_')) throw new XenosTranslateError(`alu[${instructionIndex}]: ${scalarTrap} is pixel-shader-only (kills do not exist in a vertex stage)`);
    if (scalarTrap.startsWith('opcode_')) throw new XenosTranslateError(`alu[${instructionIndex}]: scalar opcode ${scalarOpcode} (${scalarTrap}) is invalid`);
    throw new XenosTranslateError(`alu[${instructionIndex}]: ${scalarTrap} (transcendental math with clamped IEEE semantics) is not supported yet`);
  }
}

/** Xenos "default nop" (vector max with no writes + scalar retain_prev with no writes): emits nothing. */
function isAluNop(instr) {
  const vectorNop = instr.vectorOpName === 'max' && instr.vectorOpResultWriteMask === 0 &&
    !instr.vectorClamp && instr.operands[0].storageSource === 'register' && instr.operands[0].index === 0 &&
    instr.operands.every((o) => o.storageSource === 'register' && o.index === 0 && o.swizzle === 0x0000 && !o.isNegated && !o.isAbsoluteValue) &&
    instr.vectorDest === 0 && !instr.vectorDestRelative;
  const scalarNop = instr.scalarOpName === 'retain_prev' && instr.scalarOpResultWriteMask === 0 &&
    !instr.scalarClamp && instr.scalarDest === 0 && !instr.scalarDestRelative;
  return vectorNop && scalarNop;
}

// ---------------------------------------------------------------- expressions
const LANE_CHARS = ['x', 'y', 'z', 'w'];

/** Base storage expression for an operand (no swizzle/modifiers). */
function operandBaseExpr(op) {
  return op.storageSource === 'constant' ? `fc[${op.index}]` : `r[${op.index}]`;
}

/** Source-lane expression for lane (0–3) with swizzle + abs + neg applied. */
function laneExpr(op, lane) {
  const comp = swizzledComponent(op.swizzle, lane);
  let expr = `${operandBaseExpr(op)}.${LANE_CHARS[comp]}`;
  if (op.isAbsoluteValue) expr = `abs(${expr})`;
  if (op.isNegated) expr = `(-(${expr}))`;
  return expr;
}

/** Component expression for component index (sc scalar: 3 = W, 0 = X). */
function componentExpr(op, compIndex) {
  return laneExpr(op, compIndex);
}

/** Identity-swizzle/no-modifier test for emitting the base vector directly. */
function isIdentityOperand(op) {
  return op.swizzle === 0x0000 && !op.isAbsoluteValue && !op.isNegated;
}

/** Whole-vector value for an operand (vec4f expression). */
function vectorExpr(op) {
  if (isIdentityOperand(op)) return operandBaseExpr(op);
  return `vec4f(${[0, 1, 2, 3].map((l) => laneExpr(op, l)).join(', ')})`;
}

/** Whole-vector value for a scalar broadcast context. */
function vectorValue(op) {
  return vectorExpr(op);
}

const CMP_OP = Object.freeze({ eq: '==', ne: '!=', gt: '>', ge: '>=' });
const KILL_VECTOR_CMP = Object.freeze({ kill_eq: '==', kill_gt: '>', kill_ge: '>=', kill_ne: '!=' });

/**
 * WGSL condition under which a kill instruction demotes this invocation
 * (`discard`), or null for non-kill instructions. xenia semantics:
 *   - vector kills compare all 4 lanes with the vector opcode and OR them
 *     (OpAny over bool4; kKillNe is FUnordNotEqual, which WGSL `!=` matches);
 *   - scalar kills compare lane W ('a') of the scalar source against 0.0
 *     (1.0 for kills_one), ordered except kills_ne/FUnordNotEqual.
 */
function killConditionFor(instr) {
  const o = instr.operands;
  const vcmp = instr.vectorOpName && KILL_VECTOR_CMP[instr.vectorOpName];
  if (vcmp) {
    const lanes = [0, 1, 2, 3].map((l) => `(${laneExpr(o[0], l)} ${vcmp} ${laneExpr(o[1], l)})`);
    return `any(vec4<bool>(${lanes.join(', ')}))`;
  }
  const a = componentExpr(o[2], 3); // W lane of the scalar source
  switch (instr.scalarOpName) {
    case 'kills_eq': return `(${a} == 0.0)`;
    case 'kills_gt': return `(${a} > 0.0)`;
    case 'kills_ge': return `(${a} >= 0.0)`;
    case 'kills_ne': return `(${a} != 0.0)`;
    case 'kills_one': return `(${a} == 1.0)`;
    default: return null;
  }
}
const SCALAR_CMP = Object.freeze({
  seqs: '==', sgts: '>', sges: '>=', snes: '!=',
  setp_eq: '==', setp_ne: '!=', setp_gt: '>', setp_ge: '>=',
});
const VECTOR_PUSH_CMP = Object.freeze({
  setp_eq_push: 'eq', setp_ne_push: 'ne', setp_gt_push: 'gt', setp_ge_push: 'ge',
});

/**
 * Compute the vector-ALU result lanes.
 * @returns {{exprs: string[], defined: boolean[]}} — lane exprs valid where `defined[i]`
 */
function computeVectorLanes(instr, suffix) {
  const mask = instr.vectorOpResultWriteMask;
  const exprs = [null, null, null, null];
  const defined = [false, false, false, false];
  const writeLane = (lane, expr) => {
    exprs[lane] = instr.vectorClamp ? `clamp(${expr}, 0.0, 1.0)` : expr;
    defined[lane] = true;
  };
  const o = instr.operands;
  const perLane = (fn) => {
    for (let lane = 0; lane < 4; lane++) {
      if (!(mask & (1 << lane))) continue;
      writeLane(lane, fn(lane));
    }
  };
  const broadcast = (expr) => perLane(() => expr);
  const lanePair = (lane, i) => laneExpr(o[i], lane);

  switch (instr.vectorOpName) {
    case 'add': perLane((l) => `(${lanePair(l, 0)} + ${lanePair(l, 1)})`); break;
    case 'mul': perLane((l) => `(${lanePair(l, 0)} * ${lanePair(l, 1)})`); break;
    case 'max': perLane((l) => `select(${lanePair(l, 1)}, ${lanePair(l, 0)}, (${lanePair(l, 0)} >= ${lanePair(l, 1)}))`); break;
    case 'min': perLane((l) => `select(${lanePair(l, 1)}, ${lanePair(l, 0)}, (${lanePair(l, 0)} < ${lanePair(l, 1)}))`); break;
    case 'seq': perLane((l) => `select(0.0, 1.0, (${lanePair(l, 0)} == ${lanePair(l, 1)}))`); break;
    case 'sgt': perLane((l) => `select(0.0, 1.0, (${lanePair(l, 0)} > ${lanePair(l, 1)}))`); break;
    case 'sge': perLane((l) => `select(0.0, 1.0, (${lanePair(l, 0)} >= ${lanePair(l, 1)}))`); break;
    case 'sne': perLane((l) => `select(0.0, 1.0, (${lanePair(l, 0)} != ${lanePair(l, 1)}))`); break;
    case 'frc': perLane((l) => `(${lanePair(l, 0)} - floor(${lanePair(l, 0)}))`); break;
    case 'trunc': perLane((l) => `trunc(${lanePair(l, 0)})`); break;
    case 'floor': perLane((l) => `floor(${lanePair(l, 0)})`); break;
    case 'mad': perLane((l) => `fma(${lanePair(l, 0)}, ${lanePair(l, 1)}, ${lanePair(l, 2)})`); break;
    case 'cndeq': perLane((l) => `select(${lanePair(l, 2)}, ${lanePair(l, 1)}, (${lanePair(l, 0)} == 0.0))`); break;
    case 'cndge': perLane((l) => `select(${lanePair(l, 2)}, ${lanePair(l, 1)}, (${lanePair(l, 0)} >= 0.0))`); break;
    case 'cndgt': perLane((l) => `select(${lanePair(l, 2)}, ${lanePair(l, 1)}, (${lanePair(l, 0)} > 0.0))`); break;
    case 'dp4':
    case 'dp3':
    case 'dp2add': {
      const count = instr.vectorOpName === 'dp4' ? 4 : instr.vectorOpName === 'dp3' ? 3 : 2;
      const acc = [];
      for (let i = 0; i < count; i++) acc.push(`(${lanePair(i, 0)} * ${lanePair(i, 1)})`);
      let dot = acc.reduceRight((rhs, lhs) => `(${lhs} + ${rhs})`);
      if (instr.vectorOpName === 'dp2add') {
        dot = `(${dot} + ${componentExpr(o[2], 0)})`;
      }
      broadcast(dot);
      break;
    }
    case 'max4': {
      const seen = [];
      for (let lane = 0; lane < 4; lane++) {
        const comp = swizzledComponent(o[0].swizzle, lane);
        if (!seen.includes(comp)) seen.push(comp);
      }
      let expr = laneExpr(o[0], seen[0]);
      for (let i = 1; i < seen.length; i++) {
        const next = laneExpr(o[0], seen[i]);
        expr = `select(${expr}, ${next}, (${next} >= ${expr}))`;
      }
      broadcast(expr);
      break;
    }
    case 'kill_eq':
    case 'kill_gt':
    case 'kill_ge':
    case 'kill_ne':
      // Kill instructions demote the invocation; the vector result is 0.0.
      broadcast('0.0');
      break;
    case 'dst': {
      const ySrc = `(${lanePair(1, 0)} * ${lanePair(1, 1)})`;
      for (let lane = 0; lane < 4; lane++) {
        if (!(mask & (1 << lane))) continue;
        switch (lane) {
          case 0: writeLane(0, '1.0'); break;
          case 1: writeLane(1, ySrc); break;
          case 2: writeLane(2, laneExpr(o[0], 2)); break;
          case 3: writeLane(3, laneExpr(o[1], 3)); break;
        }
      }
      break;
    }
    case 'setp_eq_push':
    case 'setp_ne_push':
    case 'setp_gt_push':
    case 'setp_ge_push':
      // Value: (x == 0 && y op 0) ? 0 : x + 1 — the full expression is computed
      // inline; the predicate half is handled by the scalar/predicate path.
      broadcast(`(select(${lanePair(0, 0)}, -1.0, (${lanePair(0, 0)} == 0.0) && (${lanePair(0, 1)} ${CMP_OP[VECTOR_PUSH_CMP[instr.vectorOpName]]} 0.0)) + 1.0)`);
      break;
    default:
      throw new XenosTranslateError(`alu[${instr.instructionIndex}]: unsupported vector op ${instr.vectorOpName ?? instr.vectorOpcode}`);
  }
  return { exprs, defined };
}

/** Predicate register write for vector push ops (null for other ops). */
function vectorPushPredicate(instr) {
  const pushOp = VECTOR_PUSH_CMP[instr.vectorOpName];
  if (!pushOp) return null;
  const o = instr.operands;
  const op = CMP_OP[pushOp];
  return `((${componentExpr(o[0], 3)} == 0.0) && (${componentExpr(o[1], 3)} ${op} 0.0))`;
}

/**
 * Compute the scalar-ALU result.
 * @returns {{valueExpr: string, storePs: boolean, predicateExpr: string|null, isNop: boolean}}
 */
function computeScalarResult(instr) {
  const o = instr.operands;
  const src = o[2]; // scalar instructions always read src3
  const mask = instr.scalarOpResultWriteMask;
  const a = () => componentExpr(src, 3); // W lane ('a')
  const b = () => componentExpr(src, 0); // X lane ('b')
  let valueExpr = null;
  let storePs = true;
  let predicateExpr = null;

  const set = (expr) => { valueExpr = expr; };
  switch (instr.scalarOpName) {
    case 'kills_eq':
    case 'kills_gt':
    case 'kills_ge':
    case 'kills_ne':
    case 'kills_one':
      // Kill instructions demote the invocation; the scalar result is 0.0.
      return { valueExpr: '0.0', storePs: true, predicateExpr: null, isNop: false };
    case 'retain_prev':
      // Value is ps; ps is not modified.
      return { valueExpr: 'ps', storePs: false, predicateExpr: null, isNop: mask === 0b0000 && !instr.scalarClamp };
    case 'adds': set(`(${a()} + ${b()})`); break;
    case 'adds_prev': set(`(${a()} + ps)`); break;
    case 'subs': set(`(${a()} - ${b()})`); break;
    case 'subs_prev': set(`(${a()} - ps)`); break;
    case 'muls': set(`(${a()} * ${b()})`); break;
    case 'muls_prev': set(`(${a()} * ps)`); break;
    case 'maxs': set(`select(${b()}, ${a()}, (${a()} >= ${b()}))`); break;
    case 'mins': set(`select(${b()}, ${a()}, (${a()} < ${b()}))`); break;
    case 'seqs':
    case 'sgts':
    case 'sges':
    case 'snes':
      set(`select(0.0, 1.0, (${a()} ${SCALAR_CMP[instr.scalarOpName]} 0.0))`);
      break;
    case 'frcs': set(`(${a()} - floor(${a()}))`); break;
    case 'truncs': set(`trunc(${a()})`); break;
    case 'floors': set(`floor(${a()})`); break;
    case 'setp_eq':
    case 'setp_ne':
    case 'setp_gt':
    case 'setp_ge': {
      const pred = `(${a()} ${SCALAR_CMP[instr.scalarOpName]} 0.0)`;
      predicateExpr = pred;
      set(`select(1.0, 0.0, ${pred})`);
      break;
    }
    case 'setp_inv': {
      const av = a();
      const pred = `(${av} == 1.0)`;
      predicateExpr = pred;
      set(`select(select(${av}, 1.0, (${av} == 0.0)), 0.0, ${pred})`);
      break;
    }
    case 'setp_pop': {
      const a1 = `(${a()} - 1.0)`;
      const pred = `(${a1} <= 0.0)`;
      predicateExpr = pred;
      set(`select(${a1}, 0.0, ${pred})`);
      break;
    }
    case 'setp_clr':
      predicateExpr = 'false';
      set('3.402823466e38'); // FLT_MAX
      break;
    case 'setp_rstr': {
      const av = a();
      const pred = `(${av} == 0.0)`;
      predicateExpr = pred;
      set(`select(${av}, 0.0, ${pred})`);
      break;
    }
    default:
      throw new XenosTranslateError(`alu[${instr.instructionIndex}]: unsupported scalar op ${instr.scalarOpName ?? instr.scalarOpcode}`);
  }
  if (instr.scalarClamp) valueExpr = `clamp(${valueExpr}, 0.0, 1.0)`;
  return { valueExpr, storePs, predicateExpr, isNop: false };
}

// ---------------------------------------------------------------- emitter
function makeEmitter() {
  const lines = [];
  let indent = 0;
  return {
    emit(line) {
      lines.push(line === '' ? '' : '  '.repeat(indent) + line);
    },
    indentIn() { indent += 1; },
    indentOut() { indent -= 1; },
    toString() { return lines.join('\n') + '\n'; },
  };
}

const POSITION_TARGET = { kind: 'position' };

/** Pre-pass over decoded blocks: opcode traps, count registers, detect feature use. */
function collectStats(dwords, decoded, shaderType) {
  let maxReg = 0;
  let usesFloatConstants = false;
  let usesBoolConstants = false;
  let predicateUsed = false;
  let psUsed = false;
  let writesPosition = false;
  let exportsColor0 = false;
  let exportsDepth = false;
  for (const block of decoded.blocks) {
    if (block.condition) {
      if (block.condition.kind === 'bool') usesBoolConstants = true;
      if (block.condition.kind === 'pred') predicateUsed = true;
    }
    for (const ins of block.instructions) {
      if (ins.kind === 'vfetch') {
        const fetch = decodeFetchForAttribute(dwords, ins.instructionIndex, shaderType);
        if (fetch.writtenLanes !== 0) maxReg = Math.max(maxReg, fetch.destReg + 1);
        continue;
      }
      const alu = ins.decoded;
      assertOpcodeSupported(alu, shaderType);
      if (isAluNop(alu)) continue;
      for (const op of alu.operands) {
        if (op.storageSource === 'constant') {
          usesFloatConstants = true;
        } else {
          // Temp sources can read GPRs the shader never writes (e.g. a fetch whose
          // lanes are all kKeep) — those still need to be allocated/zero-initialized.
          maxReg = Math.max(maxReg, op.index + 1);
        }
      }
      if (alu.isPredicated) predicateUsed = true;
      if ((alu.vectorOpName && alu.vectorOpName.includes('push'))) predicateUsed = true;
      if ((alu.scalarOpName && alu.scalarOpName.startsWith('setp'))) predicateUsed = true;
      const scalarNop = alu.scalarOpName === 'retain_prev' && alu.scalarOpResultWriteMask === 0 && !alu.scalarClamp;
      if (!scalarNop) psUsed = true;
      if (alu.isExport) {
        if (alu.vectorDestTarget.kind === 'position') writesPosition = true;
        if (alu.vectorDestTarget.kind === 'color0') exportsColor0 = true;
        if (alu.vectorDestTarget.kind === 'oDepth') exportsDepth = true;
      } else {
        if (alu.vectorOpResultWriteMask !== 0) maxReg = Math.max(maxReg, alu.vectorDest + 1);
        if (alu.scalarOpResultWriteMask !== 0) maxReg = Math.max(maxReg, alu.scalarDest + 1);
      }
    }
  }
  // registerCount counts guest GPRs r0..rN-1 that the shader may touch.
  return { registerCount: maxReg, usesFloatConstants, usesBoolConstants, predicateUsed, psUsed, writesPosition, exportsColor0, exportsDepth };
}

const EXPORT_TARGET_MEMBER = Object.freeze({ position: 'pos', interpolator: (i) => `v${i}`, color0: () => 'color0' });

function emitExportTarget(alu) {
  const t = alu.vectorDestTarget;
  if (t.kind === 'position') return 'out.pos';
  if (t.kind === 'interpolator') return `out.v${t.index}`;
  if (t.kind === 'color0') return 'out.color0';
  throw new XenosTranslateError(`export: unknown target kind ${t.kind}`);
}

/** Emit the combined export (vector + scalar halves + constant lanes). */
function emitExportInstruction(em, alu) {
  const t = alu.vectorDestTarget;
  const vec = computeVectorLanes(alu, '');
  const scalar = computeScalarResult(alu);
  const push = vectorPushPredicate(alu);
  const vecMask = alu.vectorOpResultWriteMask, scalMask = alu.scalarOpResultWriteMask;
  const c0 = alu.constant0WriteMask, c1 = alu.constant1WriteMask;

  if (t.kind === 'oDepth') {
    // oDepth is single-component (X only) and clamped to [0, 1] (xenia marks the
    // export is_clamped). Lane reconciliation is the same dual-issue rule as
    // vector lanes, applied to lane X.
    let lane;
    if (c1 & 1) lane = '1.0';
    else if (c0 & 1) lane = '0.0';
    else if (vecMask & 1) lane = vec.exprs[0];
    else if (scalMask & 1) lane = scalar.valueExpr;
    else lane = '0.0';
    em.emit(`out.depth = clamp(${lane}, 0.0, 1.0);`);
    if (push) em.emit(`p0 = ${push};`);
    emitScalarTail(em, alu, scalar, /*isExport*/ true);
    return;
  }

  const target = emitExportTarget(alu);

  if (vecMask === 0b1111 && scalMask === 0 && c0 === 0 && c1 === 0) {
    em.emit(`${target} = vec4f(${[0, 1, 2, 3].map((l) => vec.exprs[l]).join(', ')});`);
  } else {
    for (let lane = 0; lane < 4; lane++) {
      if (c1 & (1 << lane)) { em.emit(`${target}.${LANE_CHARS[lane]} = 1.0;`); continue; }
      if (c0 & (1 << lane)) { em.emit(`${target}.${LANE_CHARS[lane]} = 0.0;`); continue; }
      if (vecMask & (1 << lane)) { em.emit(`${target}.${LANE_CHARS[lane]} = ${vec.exprs[lane]};`); continue; }
      if (scalMask & (1 << lane)) { em.emit(`${target}.${LANE_CHARS[lane]} = ${scalar.valueExpr};`); }
    }
  }
  if (push) em.emit(`p0 = ${push};`);
  emitScalarTail(em, alu, scalar, /*isExport*/ true);
}

/** Emit the combined register-target store (vector mask lanes of r[d]). */
function emitVectorRegisterStore(em, alu) {
  const mask = alu.vectorOpResultWriteMask;
  const vec = computeVectorLanes(alu, '');
  const push = vectorPushPredicate(alu);
  if (mask !== 0) {
    const d = alu.vectorDest;
    if (mask === 0b1111) {
      em.emit(`r[${d}] = vec4f(${[0, 1, 2, 3].map((l) => vec.exprs[l]).join(', ')});`);
    } else {
      for (let lane = 0; lane < 4; lane++) {
        if (mask & (1 << lane)) em.emit(`r[${d}].${LANE_CHARS[lane]} = ${vec.exprs[lane]};`);
      }
    }
  }
  if (push) em.emit(`p0 = ${push};`);
}

/** ps update, predicate write, and (non-export) scalar dest store. */
function emitScalarTail(em, alu, scalar, isExport) {
  if (scalar.isNop) return;
  if (scalar.predicateExpr) em.emit(`p0 = ${scalar.predicateExpr};`);
  if (scalar.storePs) em.emit(`ps = ${scalar.valueExpr};`);
  if (isExport) return;
  const mask = alu.scalarOpResultWriteMask;
  if (mask === 0) return;
  const d = alu.scalarDest;
  if (mask === 0b1111) {
    em.emit(`r[${d}] = vec4f(${scalar.valueExpr});`);
  } else {
    for (let lane = 0; lane < 4; lane++) {
      if (mask & (1 << lane)) em.emit(`r[${d}].${LANE_CHARS[lane]} = ${scalar.valueExpr};`);
    }
  }
}

function emitAluInstruction(em, alu, shaderType) {
  if (isAluNop(alu)) return;
  assertOpcodeSupported(alu, shaderType); // traps before any output for this instruction
  em.emit(`// alu[${alu.instructionIndex}] ${alu.vectorOpName ?? 'nop'}/${alu.scalarOpName ?? 'nop'}`);
  if (alu.isPredicated) {
    em.emit(`if (${alu.predicateCondition ? 'p0' : '!p0'}) {`);
    em.indentIn();
  }
  const kill = killConditionFor(alu);
  if (kill) em.emit(`if (${kill}) { discard; }`);
  if (alu.isExport) {
    emitExportInstruction(em, alu);
  } else {
    emitVectorRegisterStore(em, alu);
    const scalar = computeScalarResult(alu);
    emitScalarTail(em, alu, scalar, /*isExport*/ false);
  }
  if (alu.isPredicated) {
    em.indentOut();
    em.emit('}');
  }
}

function emitBlock(em, block, shaderType) {
  const condition = block.condition;
  if (condition) {
    let condText;
    if (condition.kind === 'bool') condText = `bc(${condition.index}u)`;
    else condText = 'p0';
    if (!condition.condition) condText = `!${condText}`;
    em.emit(`if (${condText}) {`);
    em.indentIn();
  }
  for (const ins of block.instructions) {
    if (ins.kind === 'vfetch') {
      em.emit(`// vfetch[${ins.instructionIndex}] — attribute load happens at the pipeline boundary`);
      continue;
    }
    emitAluInstruction(em, ins.decoded, shaderType);
  }
  if (condition) {
    em.indentOut();
    em.emit('}');
  }
}

// ---------------------------------------------------------------- module assembly
function dwordsOf(ucodeWords) {
  return ucodeWords instanceof Uint32Array ? ucodeWords : Uint32Array.from(ucodeWords ?? [], (w) => w >>> 0);
}

function range(count) {
  const out = [];
  for (let i = 0; i < count; i++) out.push(i);
  return out;
}

/** Lane range of interpolator GPRs a pixel shader may read (r0..r15 ∩ static bound). */
function pixelInterpolatorLanes(stats) {
  return range(Math.min(16, stats.registerCount));
}

/**
 * Emit one shader module (vertex or pixel) from an already-decoded program.
 * @param {'vertex'|'pixel'} shaderType
 * @param {Uint32Array} dwords
 * @param {object} decoded from decodeShader
 * @param {object} stats from collectStats
 * @param {{bindings?: {float: number, bool: number, alphaTest?: number}|null, linkedInterpolators?: number[]|null, alphaTest?: {compare: number}|null}} options
 *   bindings — global (program-wide) binding indices to use instead of the
 *   per-stage self-computation (needed so VS and PS agree on @binding);
 *   alphaTest — pixel only: emit a discard-by-alpha (xenia's alpha test,
 *   RB_COLORCONTROL + RB_ALPHA_REF) comparing the oC0 alpha against a
 *   constant in `bindings.alphaTest`; the compare is a Xenos CompareFunction
 *   number (0..6 — kAlways never reaches the translator, the decode nulls it);
 *   linkedInterpolators — pixel only: the interpolator lanes (GPRs) wired to
 *   vertex outputs; others read as zero (xenia zero-initializes registers).
 */
function assembleStage(shaderType, dwords, decoded, stats, options) {
  const bindings = options?.bindings ?? null;
  const { registerCount, usesFloatConstants, usesBoolConstants, predicateUsed, psUsed, writesPosition } = stats;
  const inputLocations = decoded.inputLocations;
  const interpolatorLocations = decoded.interpolatorLocations;
  const linked = shaderType === 'pixel' ? (options?.linkedInterpolators ?? pixelInterpolatorLanes(stats)) : null;

  // Interface.
  let inStruct = null;
  let outStruct = null;
  if (shaderType === 'vertex') {
    const vsInMembers = inputLocations.map((n) => `  @location(${n}) a${n}: vec4f,`);
    inStruct = vsInMembers.length > 0 ? ['struct VsIn {', ...vsInMembers, '};'].join('\n') : null;
    const vsOutLines = ['struct VsOut {', '  @builtin(position) pos: vec4f,'];
    for (const n of interpolatorLocations) vsOutLines.push(`  @location(${n}) v${n}: vec4f,`);
    vsOutLines.push('};');
    outStruct = vsOutLines.join('\n');
  } else {
    const psInMembers = linked.map((n) => `  @location(${n}) i${n}: vec4f,`);
    inStruct = psInMembers.length > 0 ? ['struct PsIn {', ...psInMembers, '};'].join('\n') : null;
    outStruct = stats.exportsDepth
      ? 'struct PsOut {\n  @location(0) color0: vec4f,\n  @builtin(frag_depth) depth: f32,\n};'
      : 'struct PsOut {\n  @location(0) color0: vec4f\n};';
  }

  // The alpha test is meaningful only when the pixel shader exports oC0
  // (xenia applies it exactly then) and the compare is not kAlways.
  const alphaTest = shaderType === 'pixel' && options?.alphaTest && stats.exportsColor0 && options.alphaTest.compare >= 0 && options.alphaTest.compare < 7
    ? options.alphaTest
    : null;
  const em = makeEmitter();
  em.emit(shaderType === 'vertex'
    ? '// Generated by wgsl_translate.js — Xenos vertex shader ucode → WGSL'
    : '// Generated by wgsl_translate.js — Xenos pixel shader ucode → WGSL');
  em.emit('// Registers: rN = Xenos vector GPRs (zero-initialized); ps = previous scalar');
  em.emit('// result; p0 = the (single) hardware predicate register.');
  let floatBinding = -1;
  let boolBinding = -1;
  let alphaTestBinding = -1;
  if (bindings) {
    floatBinding = bindings.float;
    boolBinding = bindings.bool;
    alphaTestBinding = bindings.alphaTest ?? -1;
  } else {
    if (usesFloatConstants) floatBinding = 0;
    boolBinding = usesBoolConstants ? (floatBinding >= 0 ? 1 : 0) : -1;
    alphaTestBinding = alphaTest ? (boolBinding >= 0 ? boolBinding + 1 : (floatBinding >= 0 ? 1 : 0)) : -1;
  }
  if (alphaTest && alphaTestBinding < 0) alphaTestBinding = (usesBoolConstants && boolBinding >= 0 ? boolBinding + 1 : (usesFloatConstants && floatBinding >= 0 ? 1 : 0));
  if (usesFloatConstants && floatBinding >= 0) em.emit(`@group(0) @binding(${floatBinding}) var<uniform> fc: array<vec4f, 256>;`);
  if (usesBoolConstants && boolBinding >= 0) {
    em.emit(`@group(0) @binding(${boolBinding}) var<uniform> bools: array<u32, 8>;`);
    em.emit('fn bc(n: u32) -> bool { return ((bools[n >> 5u] >> (n & 31u)) & 1u) != 0u; }');
  }
  if (alphaTest) {
    em.emit(`@group(0) @binding(${alphaTestBinding}) var<uniform> alphaTest: vec4f;`);
    // xenia CompareFunction (same 3-bit numbering as zfunc); plain WGSL float
    // operators match xenia's NaN behavior (only kNotEqual is true for NaN).
    const expr = [
      'false',
      'x < alphaTest.x', 'x == alphaTest.x', 'x <= alphaTest.x',
      'x > alphaTest.x', 'x != alphaTest.x', 'x >= alphaTest.x',
    ][alphaTest.compare];
    em.emit(`fn alphaTestPass(x: f32) -> bool { return ${expr}; }`);
  }
  if (inStruct) em.emit(inStruct);
  em.emit(outStruct);
  if (shaderType === 'vertex') {
    em.emit('@vertex');
    em.emit(inStruct ? 'fn vs_main(in: VsIn) -> VsOut {' : 'fn vs_main() -> VsOut {');
  } else {
    em.emit('@fragment');
    em.emit(inStruct ? 'fn fs_main(in: PsIn) -> PsOut {' : 'fn fs_main() -> PsOut {');
  }
  em.indentIn();
  if (registerCount > 0) {
    em.emit(`var r: array<vec4f, ${registerCount}>;`);
    em.emit(`for (var i = 0u; i < ${registerCount}u; i = i + 1u) { r[i] = vec4f(0.0); }`);
  }
  if (psUsed) em.emit('var ps: f32 = 0.0;');
  if (predicateUsed) em.emit('var p0: bool = false;');
  if (shaderType === 'vertex') {
    for (const n of inputLocations) em.emit(`r[${n}] = in.a${n};`);
    em.emit('var out: VsOut;');
    em.emit('out.pos = vec4f(0.0);');
    for (const n of interpolatorLocations) em.emit(`out.v${n} = vec4f(0.0);`);
  } else {
    for (const n of linked) em.emit(`r[${n}] = in.i${n};`);
    em.emit('var out: PsOut;');
    em.emit('out.color0 = vec4f(0.0);');
    if (stats.exportsDepth) em.emit('out.depth = 1.0;');
  }
  for (const block of decoded.blocks) {
    emitBlock(em, block, shaderType);
  }
  if (alphaTest) {
    em.emit('if (!alphaTestPass(out.color0.w)) { discard; }');
  }
  em.emit('return out;');
  em.indentOut();
  em.emit('}');

  return {
    type: shaderType,
    wgsl: em.toString(),
    inputLocations: shaderType === 'vertex' ? inputLocations : linked,
    interpolatorLocations,
    writesPosition: shaderType === 'vertex' ? writesPosition : false,
    exportsColor0: shaderType === 'pixel' ? stats.exportsColor0 : false,
    exportsDepth: shaderType === 'pixel' ? stats.exportsDepth : false,
    tempRegisterCount: registerCount,
    usesFloatConstants,
    usesBoolConstants,
    predicateUsed,
    alphaTest: alphaTest ? { compare: alphaTest.compare, binding: alphaTestBinding } : null,
  };
}

function translateVertexShader(ucodeWords) {
  const dwords = dwordsOf(ucodeWords);
  const decoded = decodeShader(dwords, 'vertex');
  const stats = collectStats(dwords, decoded, 'vertex');
  return assembleStage('vertex', dwords, decoded, stats, {});
}

/**
 * Translate a pixel shader. Standalone by default: declares exactly the
 * interpolator lanes the ucode statically references (r0..r15 ∩ bound), at
 * @location(<lane>). Pass { bindings, linkedInterpolators } to make it agree
 * with a vertex shader — see translateShaderProgram().
 */
function translatePixelShader(ucodeWords, options) {
  const dwords = dwordsOf(ucodeWords);
  const decoded = decodeShader(dwords, 'pixel');
  const stats = collectStats(dwords, decoded, 'pixel');
  const bindings = options?.bindings ?? null;
  const psLanes = pixelInterpolatorLanes(stats);
  const linked = options?.linkedInterpolators ?? psLanes;
  return assembleStage('pixel', dwords, decoded, stats, { bindings, linkedInterpolators: linked, alphaTest: options?.alphaTest ?? null });
}

/**
 * Translate a vertex shader and (optionally) the pixel shader it feeds as one
 * consistent program. The pair shares one set of constant bindings (`fc` @0,
 * `bools` @(usesFloat ? 1 : 0) — only the stages that read constants declare
 * them), and the fragment stage declares exactly the interpolator locations
 * the vertex stage exports that the pixel shader can statically read — WGSL/
 * WebGPU forbid a fragment input without a matching vertex output, so lanes
 * the VS does not export read as zeros here (xenia zero-initializes registers
 * the same way).
 *
 * `options.alphaTest` ({ compare: <Xenos CompareFunction 0..6> }) enables the
 * alpha test (xenia's RB_COLORCONTROL alpha_test_enable + RB_ALPHA_REF): the
 * fragment stage discards fragments whose oC0 alpha fails the comparison.
 * Only applied when the pixel shader statically exports oC0 (xenia's
 * condition); kAlways (7) is rejected — the test always passes then. The
 * reference value is NOT baked into the shader — the backend binds the
 * IEEE-754 RB_ALPHA_REF bits as a small uniform at bindings.alphaTest.
 */
function translateShaderProgram(vsUcodeWords, psUcodeWords, options) {
  const vd = dwordsOf(vsUcodeWords);
  const decodedV = decodeShader(vd, 'vertex');
  const statsV = collectStats(vd, decodedV, 'vertex');
  const pd = psUcodeWords ? dwordsOf(psUcodeWords) : null;
  const decodedP = pd ? decodeShader(pd, 'pixel') : null;
  const statsP = pd ? collectStats(pd, decodedP, 'pixel') : null;
  const mergedFloat = statsV.usesFloatConstants || (statsP?.usesFloatConstants ?? false);
  const mergedBool = statsV.usesBoolConstants || (statsP?.usesBoolConstants ?? false);
  let alphaTest = null;
  if (options?.alphaTest && statsP?.exportsColor0) {
    const compare = options.alphaTest.compare >>> 0;
    if (compare > 7) throw new XenosTranslateError(`alpha test compare ${compare} is not a Xenos CompareFunction (0..7)`);
    if (compare !== 7) alphaTest = { compare };
  }
  const alphaTestBinding = alphaTest ? (mergedBool ? (mergedFloat ? 2 : 1) : (mergedFloat ? 1 : 0)) : -1;
  const bindings = { float: mergedFloat ? 0 : -1, bool: mergedBool ? (mergedFloat ? 1 : 0) : -1, alphaTest: alphaTestBinding };
  const vertex = assembleStage('vertex', vd, decodedV, statsV, { bindings });
  let fragment = null;
  if (pd) {
    const vsExports = new Set(decodedV.interpolatorLocations);
    const linked = pixelInterpolatorLanes(statsP).filter((i) => vsExports.has(i));
    fragment = assembleStage('pixel', pd, decodedP, statsP, { bindings, linkedInterpolators: linked, alphaTest });
  }
  return {
    type: 'program',
    vertex,
    fragment,
    usesFloatConstants: mergedFloat,
    usesBoolConstants: mergedBool,
    interpolatorLocations: vertex.interpolatorLocations,
    exportsColor0: fragment?.exportsColor0 ?? false,
    exportsDepth: fragment?.exportsDepth ?? false,
    alphaTest: fragment?.alphaTest ?? null,
  };
}

const api = {
  XenosTranslateError,
  translateVertexShader,
  translatePixelShader,
  translateShaderProgram,
};
if (typeof module !== 'undefined' && module.exports) module.exports = api;
global.XenonGpuTranslate = api;

})(typeof self !== 'undefined' ? self : globalThis);
