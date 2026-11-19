/**
 * render_state.js — Xenos render-target / blend / depth-stencil register decode.
 *
 * The Xenos ROP is configured by a handful of RB_* registers (xenia's
 * reg::RB_* unions). This module turns the raw 32-bit register values the
 * command processor snapshots at draw time into exactly what a WebGPU
 * pipeline and render pass need: a per-target write mask, an optional blend
 * state, the blend constant, the depth/stencil state, and the single stencil
 * reference (WebGPU has one reference per pass — see the BF trap below).
 *
 * Decode contracts come from xenia (BSD-3-Clause):
 *   - RB_COLOR_INFO (+ RB_COLOR[1-3]_INFO): color_format nibble at +16
 *     (xenos::ColorRenderTargetFormat), color_exp_bias at +20, color_base
 *     tiles at +0/+11.
 *   - RB_DEPTH_INFO: depth_format 1 bit at +16 (xenos::DepthRenderTargetFormat:
 *     0 = kD24S8, 1 = kD24FS8 "20e4 float").
 *   - RB_COLOR_MASK: 4 bits per render target (RGBA write enables).
 *   - RB_BLENDCONTROL[0-3]: color_srcblend +0:5, color_comb_fcn +5:3,
 *     color_destblend +8:5, alpha_srcblend +16:5, alpha_comb_fcn +21:3,
 *     alpha_destblend +24:5 (5-bit fields; xenia indexes 32-entry tables that
 *     map every unlisted value to kZero).
 *   - RB_DEPTHCONTROL: stencil_enable +0, z_enable +1, z_write_enable +2,
 *     zfunc +4:3, backface_enable +7, stencilfunc +8:3, stencilfail +11:3,
 *     stencilzpass +14:3, stencilzfail +17:3, and the back-face variants at
 *     +20/+23/+26/+29.
 *   - RB_STENCILREFMASK / _BF: stencilref +0:8, stencilmask +8:8,
 *     stencilwritemask +16:8.
 *   - RB_COLORCONTROL: alpha_func +0:3, alpha_test_enable +3,
 *     alpha_to_mask_enable +4.
 *   - RB_BLEND_RED/GREEN/BLUE/ALPHA: f32 blend constants (raw IEEE-754 dwords
 *     as carried by the host-order register file — the command processor
 *     already applied the BE→host swap when parsing PM4 packets).
 *
 * Honest-support policy (docs/XENOS_GPU.md step 6):
 *   - Only k_8_8_8_8 is representable on the scanout target today, so any
 *     other color format the pixel shader could actually be writing to traps
 *     loudly. Gamma-encoded targets (k_8_8_8_8_GAMMA) additionally need the
 *     PWL gamma encode, which lives with the EDRAM render-target cache
 *     (step 7) — trapping beats rendering 2.2x-dark output.
 *   - kD24S8 depth maps to WebGPU depth24plus-stencil8. kD24FS8 (20e4 float
 *     depth) is not representable and traps.
 *   - Alpha test (RB_COLORCONTROL.ALPHA_TEST) has no WGSL representation yet
 *     and traps loudly rather than silently passing rejected pixels.
 *   - WebGPU exposes one stencil reference per render pass; the back-face
 *     reference (RB_STENCILREFMASK_BF) cannot differ from the front-face one
 *     when back-face stencil is in effect.
 *   - Anything with an unrecognized enumeration still decodes to a named,
 *     loud XenosRenderStateError — never to a silent guess.
 */
(function (global) {
  'use strict';

  class XenosRenderStateError extends Error {}

  // Register indices (xenia XE_GPU_REG_RB_*; context registers live at 0x2000+).
  const RB_COLOR_INFO = 0x2001, RB_DEPTH_INFO = 0x2002;
  const RB_COLOR1_INFO = 0x2003, RB_COLOR2_INFO = 0x2004, RB_COLOR3_INFO = 0x2005;
  const RB_COLOR_MASK = 0x2104;
  const RB_BLEND_RED = 0x2105, RB_BLEND_GREEN = 0x2106, RB_BLEND_BLUE = 0x2107, RB_BLEND_ALPHA = 0x2108;
  const RB_STENCILREFMASK_BF = 0x210c, RB_STENCILREFMASK = 0x210d;
  const RB_DEPTHCONTROL = 0x2200, RB_BLENDCONTROL0 = 0x2201, RB_COLORCONTROL = 0x2202;
  const RB_BLENDCONTROL1 = 0x2209, RB_BLENDCONTROL2 = 0x220a, RB_BLENDCONTROL3 = 0x220b;

  // xenos::ColorRenderTargetFormat name table. Undefined nibbles are reported
  // as '0x<nibble> (unrecognized)'.
  const COLOR_FORMAT_NAMES = {
    0: 'k_8_8_8_8',
    1: 'k_8_8_8_8_GAMMA',
    2: 'k_2_10_10_10',
    3: 'k_2_10_10_10_FLOAT',
    4: 'k_16_16',
    5: 'k_16_16_16_16',
    6: 'k_16_16_FLOAT',
    7: 'k_16_16_16_16_FLOAT',
    10: 'k_2_10_10_10_AS_10_10_10_10',
    12: 'k_2_10_10_10_FLOAT_AS_16_16_16_16',
    14: 'k_32_FLOAT',
    15: 'k_32_32_FLOAT',
  };
  // Components that exist in each format (xenia GetColorRenderTargetFormatComponentCount).
  const COLOR_FORMAT_COMPONENTS = {
    0: 4, 1: 4, 2: 4, 3: 4, 4: 2, 5: 4, 6: 2, 7: 4, 10: 4, 12: 4, 14: 1, 15: 2,
  };

  // xenos::BlendFactor → GPUBlendFactor (32 slots; unlisted values are zero,
  // like xenia's static PipelineBlendFactor tables).
  const BLEND_FACTORS = new Array(32).fill('zero');
  BLEND_FACTORS[0] = 'zero'; BLEND_FACTORS[1] = 'one';
  BLEND_FACTORS[4] = 'src'; BLEND_FACTORS[5] = 'one-minus-src';
  BLEND_FACTORS[6] = 'src-alpha'; BLEND_FACTORS[7] = 'one-minus-src-alpha';
  BLEND_FACTORS[8] = 'dst'; BLEND_FACTORS[9] = 'one-minus-dst';
  BLEND_FACTORS[10] = 'dst-alpha'; BLEND_FACTORS[11] = 'one-minus-dst-alpha';
  BLEND_FACTORS[12] = 'constant'; BLEND_FACTORS[13] = 'one-minus-constant';
  BLEND_FACTORS[14] = 'constant-alpha'; BLEND_FACTORS[15] = 'one-minus-constant-alpha';
  BLEND_FACTORS[16] = 'src-alpha-saturated';
  // Same, but the "color" factors are reinterpreted as alpha (xenia's kBlendFactorAlphaMap).
  const BLEND_FACTORS_ALPHA = new Array(32).fill('zero');
  BLEND_FACTORS_ALPHA[0] = 'zero'; BLEND_FACTORS_ALPHA[1] = 'one';
  BLEND_FACTORS_ALPHA[4] = 'src-alpha'; BLEND_FACTORS_ALPHA[5] = 'one-minus-src-alpha';
  BLEND_FACTORS_ALPHA[6] = 'src-alpha'; BLEND_FACTORS_ALPHA[7] = 'one-minus-src-alpha';
  BLEND_FACTORS_ALPHA[8] = 'dst-alpha'; BLEND_FACTORS_ALPHA[9] = 'one-minus-dst-alpha';
  BLEND_FACTORS_ALPHA[10] = 'dst-alpha'; BLEND_FACTORS_ALPHA[11] = 'one-minus-dst-alpha';
  BLEND_FACTORS_ALPHA[12] = 'constant'; BLEND_FACTORS_ALPHA[13] = 'one-minus-constant';
  BLEND_FACTORS_ALPHA[14] = 'constant-alpha'; BLEND_FACTORS_ALPHA[15] = 'one-minus-constant-alpha';
  BLEND_FACTORS_ALPHA[16] = 'src-alpha-saturated';
  // xenos::BlendOp (3 bits; unlisted values fall back to 'add', like xenia).
  const BLEND_OPS = ['add', 'subtract', 'min', 'max', 'reverse-subtract', 'add', 'add', 'add'];
  // xenos::CompareFunction (uint3) — numbering matches GPUCompareFunction.
  const COMPARE = ['never', 'less', 'equal', 'less-equal', 'greater', 'not-equal', 'greater-equal', 'always'];
  // xenos::StencilOp (uint3) — numbering matches GPUStencilOperation.
  const STENCIL_OPS = ['keep', 'zero', 'replace', 'increment-clamp', 'decrement-clamp', 'invert', 'increment-wrap', 'decrement-wrap'];

  const bf = (v) => BLEND_FACTORS[v & 0x1F];
  const bfAlpha = (v) => BLEND_FACTORS_ALPHA[v & 0x1F];
  const bOp = (v) => BLEND_OPS[v & 0x7];
  const cmp = (v) => COMPARE[v & 0x7];
  const sOp = (v) => STENCIL_OPS[v & 0x7];

  /** Decode RB_COLOR_INFO-style registers. */
  function decodeColorInfo(value) {
    const v = value >>> 0;
    const format = (v >>> 16) & 0xF;
    return {
      value: v,
      colorBase: v & 0xFFF, // bits 0..10 color_base + bit 11 color_base_bit_11 (tiles)
      format,
      formatName: COLOR_FORMAT_NAMES[format] ?? `0x${format.toString(16)} (unrecognized)`,
      componentCount: COLOR_FORMAT_COMPONENTS[format] ?? 4,
      expBias: (v >>> 20) & 0x3F,
    };
  }

  /** Decode RB_DEPTH_INFO. */
  function decodeDepthInfo(value) {
    const v = value >>> 0;
    const format = (v >>> 16) & 0x1;
    return { value: v, depthBase: v & 0xFFF, format, formatName: format === 0 ? 'kD24S8' : 'kD24FS8' };
  }

  /** Decode one RB_BLENDCONTROL register into WebGPU blend components. */
  function decodeBlendControl(value) {
    const v = value >>> 0;
    return {
      value: v,
      color: {
        srcFactor: bf(v),
        dstFactor: bf(v >>> 8),
        operation: bOp(v >>> 5),
      },
      alpha: {
        srcFactor: bfAlpha(v >>> 16),
        dstFactor: bfAlpha(v >>> 24),
        operation: bOp(v >>> 21),
      },
    };
  }

  /** xenia's blend-enable condition: disabled iff both equations are one*src + zero*dst (add). */
  function blendEnabled(decoded) {
    return !(
      decoded.color.srcFactor === 'one' && decoded.color.dstFactor === 'zero' && decoded.color.operation === 'add' &&
      decoded.alpha.srcFactor === 'one' && decoded.alpha.dstFactor === 'zero' && decoded.alpha.operation === 'add'
    );
  }

  /**
   * Normalize a guest RB_COLOR_MASK nibble for a target with `componentCount`
   * components: clear the bits of components the format does not have, then set
   * them again so host drivers don't slow-path (xenia's GetNormalizedColorMask).
   */
  function normalizeWriteMask(maskNibble, componentCount) {
    const compMask = (1 << Math.min(componentCount, 4)) - 1;
    return ((maskNibble & compMask) | (0xF & ~compMask)) >>> 0;
  }

  const dwordToF32 = (v) => new Float32Array(new Uint32Array([v >>> 0]).buffer)[0];

  // The draw path decodes the ROP snapshot on every draw; identical raw values
  // decode to identical state, so memoize the decode (bounded — ROP state
  // changes rarely and the snapshot is tiny). Deterministic in (raw, flags).
  const rtDecodeCache = new Map();
  const RT_DECODE_CACHE_LIMIT = 64;

  /**
   * Build the WebGPU render-target state for one draw.
   * @param {object} raw — raw register snapshot (see module docs); all fields required.
   * @param {object} opts
   * @param {boolean} opts.psExportsColor0 — the translated pixel shader writes oC0
   *   (only then is the RT0 format required to be representable).
   * @param {boolean} opts.psExportsDepth — the translated pixel shader writes oDepth.
   * @returns render state for the backend: { key, writeMask, blend?, blendConstant,
   *   depthStencil: { format: 'depth24plus-stencil8', depthWriteEnabled, depthCompare,
   *     stencilFront?, stencilBack?, stencilReadMask?, stencilWriteMask? },
   *   stencilRef, colorFormatName, depthFormatName }
   * @throws XenosRenderStateError — with a named reason, for anything not representable.
   */
  function buildRenderTargetState(raw, opts = {}) {
    if (!raw || typeof raw !== 'object') {
      throw new XenosRenderStateError(
        'render-target register snapshot missing (params.renderTargets) — the command processor ' +
        'snapshots RB_COLOR_INFO/RB_DEPTH_INFO/RB_COLOR_MASK/RB_BLENDCONTROL[0-3]/RB_DEPTHCONTROL/RB_COLORCONTROL/' +
        'RB_STENCILREFMASK[_BF]/RB_BLEND_* at draw time; refusing to draw with fabricated render state');
    }
    const { psExportsColor0 = true, psExportsDepth = false } = opts;
    const cacheKey = [
      raw.colorInfo0 >>> 0, raw.depthInfo >>> 0, raw.colorMask >>> 0, raw.depthControl >>> 0,
      raw.blendControl0 >>> 0, raw.stencilRefMask >>> 0, raw.stencilRefMaskBf >>> 0, raw.colorControl >>> 0,
      raw.blendRed >>> 0, raw.blendGreen >>> 0, raw.blendBlue >>> 0, raw.blendAlpha >>> 0,
      (psExportsColor0 ? 1 : 0) | ((psExportsDepth ? 1 : 0) << 1),
    ].join(',');
    const cached = rtDecodeCache.get(cacheKey);
    if (cached) return cached;

    // --- RB_COLORCONTROL: alpha test is not representable yet ---------------
    const colorControl = raw.colorControl >>> 0;
    if ((colorControl & 0x8) !== 0) {
      const func = cmp(colorControl);
      throw new XenosRenderStateError(
        `alpha test is not implemented (RB_COLORCONTROL.ALPHA_TEST: alpha_func=${func}, alpha ref from RB_ALPHA_REF) — ` +
        'no WGSL representation exists yet; refusing to render rejected pixels as visible');
    }

    // --- RB_COLOR_INFO RT0: format must be representable when written ------
    const colorInfo0 = decodeColorInfo(raw.colorInfo0);
    if (psExportsColor0) {
      if (colorInfo0.format === 1) {
        throw new XenosRenderStateError(
          'render target 0 has format k_8_8_8_8_GAMMA — gamma-encoded output needs the PWL gamma conversion ' +
          '(xenia converts in the pixel shader; it lives with the EDRAM render-target cache work, step 7). ' +
          'Refusing to render linear values into a gamma-encoded target');
      }
      if (colorInfo0.format !== 0) {
        throw new XenosRenderStateError(
          `render target 0 has format ${colorInfo0.formatName} — the only format representable on the ` +
          'scanout target today is k_8_8_8_8 (8-bit unorm); other formats need the EDRAM render-target cache (step 7). Refusing to draw');
      }
    }

    // --- RB_DEPTH_INFO: D24S8 maps to depth24plus-stencil8; D24FS8 traps ----
    const depthInfo = decodeDepthInfo(raw.depthInfo);
    const depthControl = raw.depthControl >>> 0;
    const zEnable = (depthControl & 0x2) !== 0;
    const zWriteEnable = (depthControl & 0x4) !== 0;
    const stencilEnable = (depthControl & 0x1) !== 0;
    const backfaceEnable = (depthControl & 0x80) !== 0;
    if (depthInfo.format === 1 && (zEnable || stencilEnable || psExportsDepth)) {
      throw new XenosRenderStateError(
        'depth render target has format kD24FS8 (20e4 float depth) — not representable as depth24plus-stencil8; ' +
        'the float24 depth path is part of the EDRAM render-target cache work (step 7). Refusing to draw');
    }

    // --- stencil reference (one per pass in WebGPU) -------------------------
    const stencilRefMask = raw.stencilRefMask >>> 0;
    const stencilRefMaskBf = raw.stencilRefMaskBf >>> 0;
    const ref = stencilRefMask & 0xFF;
    const refBf = stencilRefMaskBf & 0xFF;
    if (stencilEnable && backfaceEnable && refBf !== ref) {
      throw new XenosRenderStateError(
        `stencil reference differs between front (0x${ref.toString(16).padStart(2, '0')}, RB_STENCILREFMASK) and back faces ` +
        `(0x${refBf.toString(16).padStart(2, '0')}, RB_STENCILREFMASK_BF) — WebGPU has one stencil reference per render pass, ` +
        'so the back faces would test against the wrong value. Refusing to draw');
    }

    // --- write mask ---------------------------------------------------------
    const colorMask = raw.colorMask >>> 0;
    const writeMask = normalizeWriteMask((colorMask >>> 0) & 0xF, colorInfo0.componentCount);

    // --- blend --------------------------------------------------------------
    const blend0 = decodeBlendControl(raw.blendControl0);
    let blend = null;
    if (blendEnabled(blend0)) {
      blend = { color: { ...blend0.color }, alpha: { ...blend0.alpha } };
    }
    const blendConstant = [
      dwordToF32(raw.blendRed), dwordToF32(raw.blendGreen), dwordToF32(raw.blendBlue), dwordToF32(raw.blendAlpha),
    ];

    // --- depth/stencil state ------------------------------------------------
    const depthStencil = {
      format: 'depth24plus-stencil8',
      depthWriteEnabled: zEnable && zWriteEnable,
      depthCompare: zEnable ? cmp(depthControl >>> 4) : 'always',
    };
    if (stencilEnable) {
      const front = {
        compare: cmp(depthControl >>> 8),
        failOp: sOp(depthControl >>> 11),
        depthFailOp: sOp(depthControl >>> 17),
        passOp: sOp(depthControl >>> 14),
      };
      const back = backfaceEnable ? {
        compare: cmp(depthControl >>> 20),
        failOp: sOp(depthControl >>> 23),
        depthFailOp: sOp(depthControl >>> 29),
        passOp: sOp(depthControl >>> 26),
      } : { ...front };
      depthStencil.stencilFront = front;
      depthStencil.stencilBack = back;
      depthStencil.stencilReadMask = (stencilRefMask >>> 8) & 0xFF;
      depthStencil.stencilWriteMask = (stencilRefMask >>> 16) & 0xFF;
    }

    // Cache key: raw values fully determine the decode.
    const key = [
      raw.colorInfo0 >>> 0, raw.depthInfo >>> 0, colorMask, raw.depthControl >>> 0,
      raw.blendControl0 >>> 0, raw.stencilRefMask >>> 0, raw.stencilRefMaskBf >>> 0, raw.colorControl >>> 0,
      raw.blendRed >>> 0, raw.blendGreen >>> 0, raw.blendBlue >>> 0, raw.blendAlpha >>> 0,
      psExportsColor0 ? 1 : 0, psExportsDepth ? 1 : 0,
    ].join(',');

    const state = {
      key,
      writeMask,
      blend,
      blendConstant,
      depthStencil,
      stencilRef: ref,
      colorFormatName: colorInfo0.formatName,
      depthFormatName: depthInfo.formatName,
    };
    if (rtDecodeCache.size >= RT_DECODE_CACHE_LIMIT) rtDecodeCache.clear();
    rtDecodeCache.set(cacheKey, state);
    return state;
  }

  const api = {
    XenosRenderStateError,
    RB_COLOR_INFO, RB_DEPTH_INFO,
    RB_COLOR1_INFO, RB_COLOR2_INFO, RB_COLOR3_INFO,
    RB_COLOR_MASK,
    RB_BLEND_RED, RB_BLEND_GREEN, RB_BLEND_BLUE, RB_BLEND_ALPHA,
    RB_STENCILREFMASK_BF, RB_STENCILREFMASK,
    RB_DEPTHCONTROL, RB_BLENDCONTROL0, RB_COLORCONTROL,
    RB_BLENDCONTROL1, RB_BLENDCONTROL2, RB_BLENDCONTROL3,
    decodeColorInfo, decodeDepthInfo, decodeBlendControl, blendEnabled, normalizeWriteMask,
    buildRenderTargetState,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonGpuRenderState = api;

})(typeof self !== 'undefined' ? self : globalThis);
