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
 *   - RB_SURFACE_INFO: surface_pitch +0:14, msaa_samples +16:2, hiz_pitch +18:14.
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
 *   - k_8_8_8_8 and k_8_8_8_8_GAMMA are representable on the scanout target;
 *     other color formats the pixel shader could actually be writing to trap
 *     loudly (they need the EDRAM render-target cache, step 7). k_8_8_8_8_GAMMA
 *     is honored by gamma-encoding oC0's RGB in the translated PS (xenia's RTV
 *     path); blending then happens on gamma-encoded values (same caveat as xenia).
 *   - kD24S8 depth maps to WebGPU depth24plus-stencil8; kD24FS8 (20e4 float
 *     depth) maps to depth32float — WebGPU's float depth format, which has no
 *     stencil, so a guest that wants stencil with a float depth traps. The
 *     precision differs (float32 vs the guest's float24); the compare/write
 *     semantics and the z remap are the guest's.
 *   - RB_SURFACE_INFO.msaa_samples selects the sample count of the frame's
 *     attachments: k1X renders single-sampled, k4X renders into a multisampled
 *     color and depth attachment whose resolve (a box filter over the samples)
 *     produces the single-sample target — the same box filter xenia's EDRAM
 *     resolve applies for "all samples". WebGPU's 4x sample positions are the
 *     standard D3D ones (the spec mandates them), which are the positions
 *     xenia's D3D12 host uses too. k2X TRAPS: WebGPU's sample count must be 1
 *     or 4 (the spec's sample-count validation), and xenia's host supports 2x.
 *   - Alpha test (RB_COLORCONTROL.ALPHA_TEST): xenia implements it in the
 *     pixel shader (OpDiscard), so it is honored exactly when the translated
 *     guest PS statically exports oC0 — discard against RB_ALPHA_REF (IEEE-754
 *     uniform). kAlways disables it (xenia skips it). The fixed bring-up FS
 *     never applies it (never the guest's shader).
 *   - WebGPU exposes one stencil reference and one read/write mask pair per
 *     render pass; the back-face reference/masks (RB_STENCILREFMASK_BF) cannot
 *     differ from the front-face ones when back-face stencil is in effect.
 *   - Rasterizer state (PA_SU_SC_MODE_CNTL + PA_SU_POLY_OFFSET_*): decoded for
 *     triangle-list-class (polygonal) primitives exactly like xenia's D3D12 pipeline
 *     cache — cull_front/cull_back select cullMode (both set = the draw is dropped,
 *     rasterization produces no fragments; our translated shaders cannot memexport),
 *     the `face` bit selects frontFace (0 = CCW, 1 = CW), and polygon offset becomes
 *     depthBias/depthBiasSlopeScale (UNORM24 factor for kD24S8, float24-ULP factor ×8
 *     for kD24FS8, slope in 1/16-subpixel units). Points/lines are never culled
 *     (xenia: filled front faces only) and use the para offset register pair.
 *     kDualMode with a non-triangle polygon type on a non-culled face is wireframe —
 *     WebGPU has no wireframe fill mode, so it traps loudly (4541096E-style
 *     poly_mode=2 + kTriangles does NOT trap, exactly like xenia).
 *   - Viewport (PA_CL_VTE_CNTL/PA_CL_CLIP_CNTL/the PA_CL_VPORT registers/PA_SU_VTX_CNTL,
 *     window offset from the scissor snapshot): WebGPU has no programmable
 *     viewport, so the guest viewport is applied in the translated vertex shader
 *     (xenia's clipping-disabled formula for every guest draw):
 *     ndc_scale[i] = scale[i] * (2/extent[i]);
 *     ndc_offset[i] = (offset[i] + window/half-pixel offsets) * (2/extent[i]) - 1.
 *     Z passes through when PA_CL_CLIP_CNTL.clip_disable or dx_clip_space_def is set,
 *     otherwise the GL-style -W..W → 0..W remap with zmin/zmax = saturate(oz),
 *     saturate(oz+sz) (xenia's exact z branch, minus the D3D12 reverse-Z swap —
 *     WebGPU's depth-compare is arbitrary). Guest pixels map 1:1 onto the canvas
 *     (extent = canvas size, assumed == guest RT extent); xenia's integer-extent
 *     host-viewport crop (a host-side EDRAM optimization) is NOT reproduced —
 *     out-of-viewport geometry rasterizes as on hardware. User clip planes
 *     (PA_CL_CLIP_CNTL.ucp_ena != 0) TRAP — WebGPU has no clip planes.
 *   - RB_COLORCONTROL.alpha_to_mask_enable (bit 4) is inert at 1x MSAA on
 *     Xenos-class hardware, so it needs no handling here; multisampled guests
 *     are refused by the MSAA trap instead.
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
  const PA_SU_SC_MODE_CNTL = 0x2205;
  const PA_SU_POLY_OFFSET_FRONT_SCALE = 0x2380, PA_SU_POLY_OFFSET_FRONT_OFFSET = 0x2381;
  const PA_SU_POLY_OFFSET_BACK_SCALE = 0x2382, PA_SU_POLY_OFFSET_BACK_OFFSET = 0x2383;

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
  // Guest color format → the WebGPU color target, following xenia's own host
  // mapping (RenderTargetCache::GetColorDXGIFormat + GetBaseColorFormat):
  //   k_8_8_8_8 / k_8_8_8_8_GAMMA → R8G8B8A8_UNORM (8-bit unorm RGBA)
  //   k_2_10_10_10                  → R10G10B10A2_UNORM
  //   k_2_10_10_10_FLOAT            → R16G16B16A16_FLOAT
  //   k_16_16                       → R16G16_SNORM
  //   k_16_16_16_16                 → R16G16B16A16_SNORM
  //   k_16_16_FLOAT                 → R16G16_FLOAT
  //   k_16_16_16_16_FLOAT           → R16G16B16A16_FLOAT
  //   k_2_10_10_10_AS_10_10_10_10   → base k_2_10_10_10 (the extra alpha
  //                                   planes are MRT — oC1..oC3, which the
  //                                   translator traps on)
  //   k_2_10_10_10_FLOAT_AS_16_16_16_16 → base k_2_10_10_10_FLOAT
  //   k_32_FLOAT                    → R32_FLOAT
  //   k_32_32_FLOAT                 → R32G32_FLOAT
  // `webgpu: null` means the canvas format itself (the canvas is 8-bit unorm
  // RGBA, so k_8_8_8_8 and k_8_8_8_8_GAMMA render straight into it).
  // `channels` is the WebGPU channel mask (R=1, G=2, B=4, A=8) the format has —
  // the guest's write mask is restricted to it (the absent channels do not
  // exist on the hardware either).
  // `blendable` follows the WebGPU texture-format table: r32float/rg32float
  // are only blendable with the optional "float32-blendable" feature, which
  // this port does not request (xenia's D3D12 host cannot blend them either).
  // `tier1` marks the 16-bit snorm formats — WebGPU exposes them only with
  // the optional "texture-formats-tier1" feature.
  const COLOR_FORMAT_TARGET = {
    0: { webgpu: null, channels: 0xF, blendable: true },
    1: { webgpu: null, channels: 0xF, blendable: true },
    2: { webgpu: 'rgb10a2unorm', channels: 0xF, blendable: true },
    3: { webgpu: 'rgba16float', channels: 0xF, blendable: true },
    4: { webgpu: 'rg16snorm', channels: 0x3, blendable: true, tier1: true },
    5: { webgpu: 'rgba16snorm', channels: 0xF, blendable: true, tier1: true },
    6: { webgpu: 'rg16float', channels: 0x3, blendable: true },
    7: { webgpu: 'rgba16float', channels: 0xF, blendable: true },
    10: { webgpu: 'rgb10a2unorm', channels: 0xF, blendable: true },
    12: { webgpu: 'rgba16float', channels: 0xF, blendable: true },
    14: { webgpu: 'r32float', channels: 0x1, blendable: false },
    15: { webgpu: 'rg32float', channels: 0x3, blendable: false },
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
      // 6-bit signed exponent (xenia's `int32_t color_exp_bias : 6`) — sign-extend.
      expBias: (() => { const b = (v >>> 20) & 0x3F; return b & 0x20 ? b - 0x40 : b; })(),
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

  // zero-allocation dword→f32 bitcast (single-threaded command processor — the
  // scratch is never held across a call boundary)
  const _dwordScratch = new ArrayBuffer(4);
  const _dwordScratchU32 = new Uint32Array(_dwordScratch);
  const _dwordScratchF32 = new Float32Array(_dwordScratch);
  const dwordToF32 = (v) => { _dwordScratchU32[0] = v >>> 0; return _dwordScratchF32[0]; };

  // xenia xenos::kPolygonOffsetScaleSubpixelUnit: the guest slope-scale registers
  // are in 1/16 subpixel units; WebGPU's depthBiasSlopeScale is per pixel.
  const K_POLY_OFFSET_SCALE_SUBPIXEL_UNIT = 1.0 / 16.0;
  // xenia draw_util.h: UNORM24 offsets in units of the 24-bit UNORM resolution;
  // float24 offsets in float24 ULPs (2^24, scaled down by the 3-mantissa-bit gap).
  const K_D3D10_POLYGON_OFFSET_FACTOR_UNORM24 = (1 << 24) - 1; // 16777215
  const K_D3D10_POLYGON_OFFSET_FACTOR_FLOAT24 = Math.pow(2, 24) * (1.0 / 8.0); // 2^21

  /**
   * Rasterizer state (PA_SU_SC_MODE_CNTL + PA_SU_POLY_OFFSET_*).
   * Mirrors xenia's D3D12 pipeline cache (d3d12/pipeline_cache.cc, "Rasterizer state").
   * @param {object} raw — raw register snapshot (paSuScModeCntl, paSuPolyOffset*);
   *   missing fields decode to 0 (all-disabled), like every other register here.
   * @param {object} ctx
   * @param {boolean} ctx.primitivePolygonal — xenia IsPrimitivePolygonal: triangle-list/
   *   fan/strip (+ fan-with-W-flags), quad-list/strip, polygon (and patch types on the
   *   tessellation path, which this port does not support). Points/lines: no culling.
   * @param {boolean} ctx.isFloat24Depth — RB_DEPTH_INFO format bit (kD24FS8), selecting the
   *   float24-ULP offset factor (xenia GetD3D10IntegerPolygonOffset).
   */
  function decodeRasterState(raw, { primitivePolygonal = false, isFloat24Depth = false } = {}) {
    const mode = raw.paSuScModeCntl >>> 0;
    const cullFront = (mode & 0x1) !== 0;
    const cullBack = (mode & 0x2) !== 0;
    const faceCw = (mode & 0x4) !== 0; // 0 - front is CCW, 1 - front is CW (xenia)
    const polyMode = (mode >>> 3) & 0x3; // xenos::PolygonModeEnable: 1 = kDualMode
    const frontPtype = (mode >>> 5) & 0x7, backPtype = (mode >>> 8) & 0x7; // xenos::PolygonType: 2 = kTriangles
    const frontOffsetEna = (mode & 0x800) !== 0;
    const backOffsetEna = (mode & 0x1000) !== 0;
    const paraOffsetEna = (mode & 0x2000) !== 0;

    let cullMode = 'none';
    let bothCull = false;
    if (primitivePolygonal) {
      if (cullFront && cullBack) {
        // xenia: rasterization is disabled for both-faces-culled draws (the draw
        // produces no fragments). Translated shaders cannot export to memory, so
        // dropping the draw is exactly equivalent.
        bothCull = true;
      } else if (cullFront) {
        cullMode = 'front';
      } else if (cullBack) {
        cullMode = 'back';
      }
      // Wireframe (kDualMode): an unrepresentable fill mode in WebGPU. Only a
      // non-culled face matters — xenia ignores the culled side entirely.
      if (polyMode === 1) {
        const frontWire = !cullFront && frontPtype !== 2;
        const backWire = !cullBack && backPtype !== 2;
        if (frontWire || backWire) {
          throw new XenosRenderStateError(
            `PA_SU_SC_MODE_CNTL poly_mode=kDualMode with a non-triangle ${frontWire ? 'front' : 'back'} polygon ` +
            `(ptype ${frontWire ? frontPtype : backPtype}, points/lines) — WebGPU has no wireframe fill mode; ` +
            'refusing to draw (4541096E-style poly_mode=2 + kTriangles is NOT affected, matching xenia)');
        }
      }
    }
    const frontFace = faceCw ? 'cw' : 'ccw';

    // Polygon offset (xenia GetPreferredFacePolygonOffset): front wins for
    // polygonal primitives (unless disabled or culled); back only when it
    // decodes to a non-zero scale/offset; para registers for points/lines.
    let scale = 0.0, offset = 0.0;
    if (primitivePolygonal) {
      if (frontOffsetEna && !cullFront) {
        scale = dwordToF32(raw.paSuPolyOffsetFrontScale);
        offset = dwordToF32(raw.paSuPolyOffsetFrontOffset);
      }
      if (backOffsetEna && !cullBack && !scale && !offset) {
        scale = dwordToF32(raw.paSuPolyOffsetBackScale);
        offset = dwordToF32(raw.paSuPolyOffsetBackOffset);
      }
    } else if (paraOffsetEna) {
      scale = dwordToF32(raw.paSuPolyOffsetFrontScale);
      offset = dwordToF32(raw.paSuPolyOffsetFrontOffset);
    }
    // xenia GetD3D10IntegerPolygonOffset (ceil-abs, sign restore).
    const factor = isFloat24Depth ? K_D3D10_POLYGON_OFFSET_FACTOR_FLOAT24 : K_D3D10_POLYGON_OFFSET_FACTOR_UNORM24;
    let bias = Math.ceil(Math.abs(offset) * factor);
    if (isFloat24Depth) bias *= 8; // float24 ULP -> float32 ULPs (xenia: bias <<= 3)
    const depthBias = offset < 0 ? -bias : bias;
    const depthBiasSlopeScale = scale * K_POLY_OFFSET_SCALE_SUBPIXEL_UNIT;

    return { cullMode, frontFace, depthBias, depthBiasSlopeScale, depthBiasClamp: 0, bothCull };
  }

  const PA_CL_CLIP_CNTL = 0x2204, PA_CL_VTE_CNTL = 0x2206, PA_SU_VTX_CNTL = 0x2302;
  const PA_CL_VPORT_XSCALE = 0x210F, PA_CL_VPORT_XOFFSET = 0x2110, PA_CL_VPORT_YSCALE = 0x2111, PA_CL_VPORT_YOFFSET = 0x2112;
  const PA_CL_VPORT_ZSCALE = 0x2113, PA_CL_VPORT_ZOFFSET = 0x2114;

  const sat01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
  const s15f = (v) => { v &= 0x7fff; return v & 0x4000 ? v - 0x8000 : v; };

  /**
   * Viewport decode (PA_CL_VTE_CNTL + PA_CL_CLIP_CNTL + PA_CL_VPORT_* + PA_SU_VTX_CNTL,
   * window offset from the scissor snapshot). WebGPU has no programmable viewport —
   * the returned coefficients are applied in the translated vertex shader
   * (in clip space, pre-w-divide), xenia's clipping-disabled formula for both XY axes:
   *   n' = n * scale * (2/E) + (offset + window/half-pixel offsets) * (2/E) - 1
   * @param {object} raw — snapshot fields paClVteCntl/paClClipCntl/paClVportXSCALE..OFFSET / paSuVtxCntl/paSuScModeCntl;
   *   missing fields decode to 0 = all enables off = scale 1 / offset 0 (xenia's zeroed-register behavior,
   *   which maps NDC across the whole render target — the pre-viewport bring-up behavior)
   * @param {object} [scissor] — scissor snapshot carrying windowOffset (PA_SC_WINDOW_OFFSET)
   * @param {object} surface — { width, height } canvas size (1:1 with the guest render target)
   * @returns {{ scale: number[], offset: number[] }} — [x,y,z] clip-space coefficients
   */
  function decodeViewportState(raw, scissor, surface) {
    const clip = (raw?.paClClipCntl ?? 0) >>> 0;
    const ucp = clip & 0x3F;
    if (ucp !== 0) {
      throw new XenosRenderStateError(
        `user clip planes are enabled (PA_CL_CLIP_CNTL.ucp_ena=0x${ucp.toString(16)}) — WebGPU has no clip planes and this port does not emulate them; ` +
        'refusing to draw (trap, never a silent missing clip)');
    }
    const vte = (raw?.paClVteCntl ?? 0) >>> 0;
    const mode = (raw?.paSuScModeCntl ?? 0) >>> 0;
    const sx = (vte & 0x1) ? dwordToF32(raw.paClVportXscale) : 1.0;
    const ox = (vte & 0x2) ? dwordToF32(raw.paClVportXoffset) : 0.0;
    const sy = (vte & 0x4) ? dwordToF32(raw.paClVportYscale) : 1.0;
    const oy = (vte & 0x8) ? dwordToF32(raw.paClVportYoffset) : 0.0;
    const sz = (vte & 0x10) ? dwordToF32(raw.paClVportZscale) : 1.0;
    const oz = (vte & 0x20) ? dwordToF32(raw.paClVportZoffset) : 0.0;

    // offset_add: PA_SC_WINDOW_OFFSET (only when vtx_window_offset_enable, bit 16) + the
    // half-pixel offset for kD3DZero pixel centers (xenia half_pixel_offset cvar, default on).
    let addX = 0, addY = 0;
    if ((mode & 0x10000) !== 0) {
      const w = (scissor?.windowOffset ?? 0) >>> 0;
      addX += s15f(w);
      addY += s15f(w >>> 16);
    }
    const pixCenter = (raw?.paSuVtxCntl ?? 0) & 0x1; // xenos::PixelCenter: 0 = kD3DZero, 1 = kOGLHalf
    if (pixCenter === 0) { addX += 0.5; addY += 0.5; }

    // XY: guest pixel space → WebGPU fixed mapping composite. Extent = canvas
    // size (1:1 with the guest RT); axes are independent.
    const ex = Math.max(1, surface?.width ?? 1), ey = Math.max(1, surface?.height ?? 1);
    const kx = 2 / ex, ky = 2 / ey;
    const ndcX = sx * kx, ndcY = sy * ky;
    const offX = (ox + addX) * kx - 1, offY = (oy + addY) * ky - 1;
    // round to f32 — the coefficients ride a Float32 uniform; keep the JS contract identical to the wire

    // Z: clip_disable or dx_clip_space_def → guest z passes through scale/offset
    // (xenia: host viewport depth range = [saturate(oz−sz), saturate(oz+sz)], VS
    // pass-through). Otherwise the guest clip space is -W..W: the VS remaps to 0..W
    // (×0.5 + 0.5W) and xenia's host depth range [saturate(oz−sz), saturate(oz+sz)]
    // (from host_clip_offset_z = offset_z − scale_z) is folded in as well, giving
    // z' = 0.5(zmax−zmin)·z_gl + 0.5(zmin+zmax). No reverse-z swap — WebGPU allows
    // any depthCompare direction (xenia's swap exists only to satisfy D3D12).
    let zScale, zOffset;
    if (((clip & 0x10000) !== 0) || ((clip & 0x80000) !== 0)) {
      zScale = sz;
      zOffset = oz;
    } else {
      const zmin = sat01(oz - sz), zmax = sat01(oz + sz);
      zScale = 0.5 * (zmax - zmin);
      zOffset = 0.5 * (zmin + zmax);
    }
    return {
      scale: [Math.fround(ndcX), Math.fround(ndcY), Math.fround(zScale)],
      offset: [Math.fround(offX), Math.fround(offY), Math.fround(zOffset)],
    };
  }

  // The draw path decodes the ROP snapshot on every draw; identical raw values
  // decode to identical state, so memoize the decode (bounded — ROP state
  // changes rarely and the snapshot is tiny). Deterministic in (raw, flags).
  const rtDecodeCache = new Map();
  const RT_DECODE_CACHE_LIMIT = 64;

  // Two-level memo for the decode hot path: a 32-bit FNV-1a fingerprint of every
  // field that determines the decode (same fields the string cacheKey covers),
  // bucketed and verified element-by-element on hit. This avoids building a long
  // string on every draw; collisions just fall through to the string cache
  // (the verification is exact — a fingerprint match never skips it).
  const RT_FP_CACHE_LIMIT = 64;
  const rtFpCache = new Map(); // fingerprint -> [{ raw, flags, state }] — verified against raw on hit
  const fpMix = (h, x) => Math.imul((h ^ (x >>> 0)) >>> 0, 16777619) >>> 0;
  // Fingerprint over exactly the fields the string cacheKey covers. Allocation-free.
  function fingerprintRaw(raw, flags) {
    let h = 2166136261 >>> 0;
    h = fpMix(h, raw.surfaceInfo === undefined ? -1 : raw.surfaceInfo);
    h = fpMix(h, raw.colorInfo0); h = fpMix(h, raw.depthInfo); h = fpMix(h, raw.colorMask);
    h = fpMix(h, raw.depthControl); h = fpMix(h, raw.blendControl0); h = fpMix(h, raw.stencilRefMask);
    h = fpMix(h, raw.stencilRefMaskBf); h = fpMix(h, raw.colorControl);
    h = fpMix(h, raw.blendRed); h = fpMix(h, raw.blendGreen); h = fpMix(h, raw.blendBlue); h = fpMix(h, raw.blendAlpha);
    h = fpMix(h, raw.alphaRef === undefined ? -1 : raw.alphaRef); h = fpMix(h, flags);
    h = fpMix(h, raw.paSuScModeCntl);
    h = fpMix(h, raw.paSuPolyOffsetFrontScale); h = fpMix(h, raw.paSuPolyOffsetFrontOffset);
    h = fpMix(h, raw.paSuPolyOffsetBackScale); h = fpMix(h, raw.paSuPolyOffsetBackOffset);
    return h >>> 0;
  }
  // Exact field-by-field verification against the stored snapshot of normalized values.
  // `a` is the stored snapshot (already >>> 0 / -1); `b` is the incoming raw snapshot — normalize as we compare.
  function sameFields(a, b) {
    const si = b.surfaceInfo === undefined ? -1 : b.surfaceInfo >>> 0;
    const ar = b.alphaRef === undefined ? -1 : b.alphaRef >>> 0;
    return a.surfaceInfo === si && a.colorInfo0 === (b.colorInfo0 >>> 0) && a.depthInfo === (b.depthInfo >>> 0)
      && a.colorMask === (b.colorMask >>> 0) && a.depthControl === (b.depthControl >>> 0)
      && a.blendControl0 === (b.blendControl0 >>> 0) && a.stencilRefMask === (b.stencilRefMask >>> 0)
      && a.stencilRefMaskBf === (b.stencilRefMaskBf >>> 0) && a.colorControl === (b.colorControl >>> 0)
      && a.blendRed === (b.blendRed >>> 0) && a.blendGreen === (b.blendGreen >>> 0)
      && a.blendBlue === (b.blendBlue >>> 0) && a.blendAlpha === (b.blendAlpha >>> 0)
      && a.alphaRef === ar
      && a.paSuScModeCntl === (b.paSuScModeCntl >>> 0)
      && a.paSuPolyOffsetFrontScale === (b.paSuPolyOffsetFrontScale >>> 0)
      && a.paSuPolyOffsetFrontOffset === (b.paSuPolyOffsetFrontOffset >>> 0)
      && a.paSuPolyOffsetBackScale === (b.paSuPolyOffsetBackScale >>> 0)
      && a.paSuPolyOffsetBackOffset === (b.paSuPolyOffsetBackOffset >>> 0);
  }
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
    const { psExportsColor0 = true, psExportsDepth = false, primitivePolygonal = false } = opts;
    // --- RB_SURFACE_INFO: the guest's MSAA sample count --------------------
    // msaa_samples at bits +16:2 — xenia's RB_SURFACE_INFO (xenos::MsaaSamples:
    // k1X=0, k2X=1, k4X=2; the field has no k8X, so 3 is invalid).
    // The frame renders into a multisampled attachment of the same size as
    // the target, and WebGPU's resolve (a box filter over the samples) writes
    // the single-sample target — xenia's EDRAM resolve for "all samples" is a
    // box filter over the sample planes too.
    const surfaceInfo = raw.surfaceInfo;
    if (surfaceInfo === undefined) {
      throw new XenosRenderStateError(
        'render-target snapshot carries no RB_SURFACE_INFO (params.renderTargets.surfaceInfo) — ' +
        'the command processor snapshots it at draw time; refusing to draw without the MSAA decision');
    }
    const msaaSamples = (surfaceInfo >>> 16) & 0x3;
    if (msaaSamples === 3) {
      throw new XenosRenderStateError(
        "RB_SURFACE_INFO.msaa_samples=3 — xenia's MsaaSamples has only k1X/k2X/k4X, so this is not a " +
        'guest MSAA mode. Refusing to draw');
    }
    if (msaaSamples === 1) {
      // WebGPU's multisample count must be 1 or 4 (the spec's sample-count
      // validation) — there is no 2x. xenia's D3D12 host supports k2X.
      throw new XenosRenderStateError(
        'guest render target is configured for 2x MSAA (RB_SURFACE_INFO.msaa_samples=1) — WebGPU only ' +
        'supports a sample count of 1 or 4, so 2x coverage cannot be reproduced. Refusing to draw');
    }
    const sampleCount = msaaSamples === 0 ? 1 : 4;
    const flags = (psExportsColor0 ? 1 : 0) | ((psExportsDepth ? 1 : 0) << 1) | ((primitivePolygonal ? 1 : 0) << 2);
    const fp = fingerprintRaw(raw, flags);
    const bucket = rtFpCache.get(fp);
    if (bucket) {
      for (let i = 0; i < bucket.length; i++) {
        const c = bucket[i];
        if (c.flags === flags && sameFields(c.raw, raw)) return c.state;
      }
    }
    const cacheKey = [
      raw.surfaceInfo === undefined ? -1 : (raw.surfaceInfo >>> 0), raw.colorInfo0 >>> 0, raw.depthInfo >>> 0, raw.colorMask >>> 0, raw.depthControl >>> 0,
      raw.blendControl0 >>> 0, raw.stencilRefMask >>> 0, raw.stencilRefMaskBf >>> 0, raw.colorControl >>> 0,
      raw.blendRed >>> 0, raw.blendGreen >>> 0, raw.blendBlue >>> 0, raw.blendAlpha >>> 0,
      raw.alphaRef === undefined ? -1 : (raw.alphaRef >>> 0), flags,
      raw.paSuScModeCntl >>> 0,
      raw.paSuPolyOffsetFrontScale >>> 0, raw.paSuPolyOffsetFrontOffset >>> 0,
      raw.paSuPolyOffsetBackScale >>> 0, raw.paSuPolyOffsetBackOffset >>> 0,
    ].join(',');
    const cached = rtDecodeCache.get(cacheKey);
    if (cached) return cached;

    // --- alpha test (RB_COLORCONTROL) — applied by the translated guest PS ---
    // xenia implements the alpha test in the pixel shader itself (OpDiscard),
    // comparing the oC0 alpha against RB_ALPHA_REF (a bitcast float). The
    // reference is not pipeline state: the backend uploads the raw dword as a
    // tiny uniform the translated PS declares. Applies only when the PS
    // exports oC0 (xenia's condition); kAlways passes trivially (xenia skips).
    const colorControl = raw.colorControl >>> 0;
    const alphaTestEnable = (colorControl & 0x8) !== 0;
    const alphaFunc = colorControl & 0x7;
    // RB_COLORCONTROL.alpha_to_mask_enable (bit 4): on Xenos/r5xx-class hardware A2C has an
    // effect only with multisampling; at 1x MSAA it is inert (reference: the same rule on the
    // host xenia/D3D path). Multisampled guests trap on RB_SURFACE_INFO below, so A2C can
    // never be silently dropped here — it is either a hardware no-op or the draw is refused.
    const alphaRef = raw.alphaRef;
    let alphaTest = null;
    if (alphaTestEnable && alphaFunc !== 7) {
      if (alphaRef === undefined) {
        throw new XenosRenderStateError(
          'RB_COLORCONTROL has alpha_test_enable set but the render-target snapshot carries no alpha reference ' +
          '(params.renderTargets.alphaRef) — refusing to draw with a fabricated reference');
      }
      alphaTest = { compare: alphaFunc, referenceDword: alphaRef >>> 0 };
    }

    // --- RB_COLOR_INFO RT0: the format decides the color attachment -------
    // The guest's color format maps to a WebGPU color target exactly like
    // xenia's host mapping (COLOR_FORMAT_TARGET above). k_8_8_8_8 and
    // k_8_8_8_8_GAMMA are 8-bit unorm RGBA — the canvas format itself, so
    // they render straight into the canvas; every other representable format
    // renders into an offscreen texture with exactly that format (so the
    // guest's writes are quantized and ranged like the hardware's), which the
    // backend presents to the canvas with a blit at swap().
    const colorInfo0 = decodeColorInfo(raw.colorInfo0);
    const colorTarget = COLOR_FORMAT_TARGET[colorInfo0.format] ?? null;
    if (psExportsColor0) {
      if (!colorTarget) {
        throw new XenosRenderStateError(
          `render target 0 has format ${colorInfo0.formatName} — no WebGPU color target represents it ` +
          '(the 16-bit snorm formats need WebGPU\'s optional "texture-formats-tier1" feature, and the ' +
          'remaining formats are not guest color formats at all). Refusing to draw');
      }
      if (colorTarget.tier1) {
        throw new XenosRenderStateError(
          `render target 0 has format ${colorInfo0.formatName} — WebGPU's ${colorTarget.webgpu} is an optional ` +
          'format ("texture-formats-tier1") that this port does not request. Refusing to draw');
      }
    }
    // The color attachment format: null = the canvas format (8-bit unorm RGBA).
    const colorFormat = (colorTarget && !colorTarget.tier1) ? colorTarget.webgpu : null;
    // The gamma encode is a no-op unless the PS statically exports oC0 (the
    // translator enforces that itself); the flag records the RT's format.
    const convertColor0ToGamma = colorInfo0.format === 1;

    // --- RB_DEPTH_INFO: D24S8 maps to depth24plus-stencil8; D24FS8 → depth32float ----
    // xenia's D3D12 host uses a float depth format for the 20e4 depth target
    // (the EDRAM clear/load/store shaders for kD24FS8 are float ones); WebGPU's
    // float depth format (depth32float) has no stencil, so a guest that wants
    // stencil with a float depth traps loudly.
    const depthInfo = decodeDepthInfo(raw.depthInfo);
    const depthControl = raw.depthControl >>> 0;
    const zEnable = (depthControl & 0x2) !== 0;
    const zWriteEnable = (depthControl & 0x4) !== 0;
    const stencilEnable = (depthControl & 0x1) !== 0;
    const backfaceEnable = (depthControl & 0x80) !== 0;
    const floatDepth = depthInfo.format === 1;
    if (floatDepth && stencilEnable) {
      throw new XenosRenderStateError(
        'depth render target has format kD24FS8 (20e4 float depth) with stencil enabled — WebGPU has no float ' +
        'depth format with a stencil plane (depth32float has none). Refusing to draw');
    }

    // --- stencil reference + masks (one pair per pass in WebGPU) -------------
    const stencilRefMask = raw.stencilRefMask >>> 0;
    const stencilRefMaskBf = raw.stencilRefMaskBf >>> 0;
    const ref = stencilRefMask & 0xFF;
    const refBf = stencilRefMaskBf & 0xFF;
    if (stencilEnable && backfaceEnable) {
      if (refBf !== ref) {
        throw new XenosRenderStateError(
          `stencil reference differs between front (0x${ref.toString(16).padStart(2, '0')}, RB_STENCILREFMASK) and back faces ` +
          `(0x${refBf.toString(16).padStart(2, '0')}, RB_STENCILREFMASK_BF) — WebGPU has one stencil reference per render pass, ` +
          'so the back faces would test against the wrong value. Refusing to draw');
      }
      const readMaskF = (stencilRefMask >>> 8) & 0xFF, readMaskB = (stencilRefMaskBf >>> 8) & 0xFF;
      const writeMaskF = (stencilRefMask >>> 16) & 0xFF, writeMaskB = (stencilRefMaskBf >>> 16) & 0xFF;
      if (readMaskB !== readMaskF || writeMaskB !== writeMaskF) {
        throw new XenosRenderStateError(
          `stencil read/write mask differs between front (stencilmask 0x${readMaskF.toString(16).padStart(2, '0')}, ` +
          `writemask 0x${writeMaskF.toString(16).padStart(2, '0')}, RB_STENCILREFMASK) and back faces ` +
          `(stencilmask 0x${readMaskB.toString(16).padStart(2, '0')}, writemask 0x${writeMaskB.toString(16).padStart(2, '0')}, RB_STENCILREFMASK_BF) — ` +
          'WebGPU has one stencil read/write mask pair per render pass. Refusing to draw');
      }
    }

    // --- write mask ---------------------------------------------------------
    const colorMask = raw.colorMask >>> 0;
    // The guest's mask, restricted to the channels the format actually has
    // (the absent channels do not exist on the hardware either — xenia hands
    // the raw nibble to D3D12, which ignores the bits the format lacks).
    const writeMask = (normalizeWriteMask((colorMask >>> 0) & 0xF, colorInfo0.componentCount) &
      (colorTarget ? colorTarget.channels : 0xF)) >>> 0;

    // --- blend --------------------------------------------------------------
    const blend0 = decodeBlendControl(raw.blendControl0);
    let blend = null;
    if (blendEnabled(blend0)) {
      if (colorTarget && !colorTarget.blendable) {
        // WebGPU: r32float/rg32float are only blendable with the optional
        // "float32-blendable" feature. xenia's D3D12 host cannot blend these
        // formats either (D3D12 requires a blendable format).
        throw new XenosRenderStateError(
          `render target 0 has format ${colorInfo0.formatName} — WebGPU cannot blend a ${colorTarget.webgpu} ` +
          'target (it needs the optional "float32-blendable" feature, which this port does not request). Refusing to draw');
      }
      blend = { color: { ...blend0.color }, alpha: { ...blend0.alpha } };
    }
    const blendConstant = [
      dwordToF32(raw.blendRed), dwordToF32(raw.blendGreen), dwordToF32(raw.blendBlue), dwordToF32(raw.blendAlpha),
    ];

    // --- rasterizer state (cull/winding/depth bias) --------------------------
    const raster = decodeRasterState(raw, { primitivePolygonal, isFloat24Depth: depthInfo.format === 1 });

    // --- depth/stencil state ------------------------------------------------
    const depthStencil = {
      // The attachment format follows the guest's depth format.
      format: floatDepth ? 'depth32float' : 'depth24plus-stencil8',
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

    // Cache key: raw values fully determine the decode. RB_SURFACE_INFO is
    // part of it — it carries the MSAA sample count, which the pipeline
    // descriptor depends on (the backend keys its pipeline cache on this).
    const key = [
      raw.surfaceInfo === undefined ? -1 : (raw.surfaceInfo >>> 0),
      raw.colorInfo0 >>> 0, raw.depthInfo >>> 0, colorMask, raw.depthControl >>> 0,
      raw.blendControl0 >>> 0, raw.stencilRefMask >>> 0, raw.stencilRefMaskBf >>> 0, raw.colorControl >>> 0,
      raw.blendRed >>> 0, raw.blendGreen >>> 0, raw.blendBlue >>> 0, raw.blendAlpha >>> 0,
      psExportsColor0 ? 1 : 0, psExportsDepth ? 1 : 0,
      raw.paSuScModeCntl >>> 0,
      raw.paSuPolyOffsetFrontScale >>> 0, raw.paSuPolyOffsetFrontOffset >>> 0,
      raw.paSuPolyOffsetBackScale >>> 0, raw.paSuPolyOffsetBackOffset >>> 0,
      primitivePolygonal ? 1 : 0,
    ].join(',');

    const state = {
      key,
      writeMask,
      blend,
      blendConstant,
      depthStencil,
      stencilRef: ref,
      alphaTest,
      raster,
      surfacePitch: surfaceInfo & 0x3FFF,
      convertColor0ToGamma,
      colorFormatName: colorInfo0.formatName,
      // The WebGPU color target format for this draw (null = the canvas
      // format — the guest's format is 8-bit unorm RGBA).
      colorFormat,
      // The guest's MSAA sample count (1, 2 or 4) — xenia's MsaaSamples.
      sampleCount,
      depthFormatName: depthInfo.formatName,
    };
    if (rtDecodeCache.size >= RT_DECODE_CACHE_LIMIT) rtDecodeCache.clear();
    rtDecodeCache.set(cacheKey, state);
    if (rtFpCache.size >= RT_FP_CACHE_LIMIT) rtFpCache.clear();
    // snapshot the 19 decoded fields once (not per draw) for verification on later hits
    const snap = {
      surfaceInfo: raw.surfaceInfo === undefined ? -1 : raw.surfaceInfo >>> 0, colorInfo0: raw.colorInfo0 >>> 0, depthInfo: raw.depthInfo >>> 0,
      colorMask: raw.colorMask, depthControl: raw.depthControl, blendControl0: raw.blendControl0,
      stencilRefMask: raw.stencilRefMask, stencilRefMaskBf: raw.stencilRefMaskBf, colorControl: raw.colorControl,
      blendRed: raw.blendRed, blendGreen: raw.blendGreen, blendBlue: raw.blendBlue, blendAlpha: raw.blendAlpha,
      alphaRef: raw.alphaRef === undefined ? -1 : (raw.alphaRef >>> 0),
      paSuScModeCntl: raw.paSuScModeCntl >>> 0,
      paSuPolyOffsetFrontScale: raw.paSuPolyOffsetFrontScale >>> 0, paSuPolyOffsetFrontOffset: raw.paSuPolyOffsetFrontOffset >>> 0,
      paSuPolyOffsetBackScale: raw.paSuPolyOffsetBackScale >>> 0, paSuPolyOffsetBackOffset: raw.paSuPolyOffsetBackOffset >>> 0,
    };
    const bucket2 = rtFpCache.get(fp);
    if (bucket2) bucket2.push({ raw: snap, flags, state });
    else rtFpCache.set(fp, [{ raw: snap, flags, state }]);
    return state;
  }

  // --- PA_SC_* scissor decode (xenia draw_util.cc GetScissor) ----------------
  // WebGPU scissor call takes (x, y, width, height) clamped to the attachment,
  // so the decode combines WINDOW scissor + WINDOW offset + SCREEN scissor,
  // collapses degenerate/inverted rects, and finally clamps to the surface
  // (canvas). The RB_SURFACE_INFO surface_pitch clamp xenia applies is an
  // EDRAM-tile-padding safety net — meaningless for the canvas target.
  const PA_SC_WINDOW_SCISSOR_TL = 0x2081, PA_SC_WINDOW_SCISSOR_BR = 0x2082;
  const PA_SC_WINDOW_OFFSET = 0x2080;
  const PA_SC_SCREEN_SCISSOR_TL = 0x200e, PA_SC_SCREEN_SCISSOR_BR = 0x200f;
  const u14 = (v) => v & 0x3fff;
  const s15 = (v) => { v &= 0x7fff; return v & 0x4000 ? v - 0x8000 : v; };

  function buildScissorState(raw, surface) {
    if (!raw || typeof raw !== 'object') {
      throw new XenosRenderStateError(
        'scissor register snapshot missing (params.scissor) — the command processor snapshots ' +
        'PA_SC_WINDOW_SCISSOR_TL/BR, PA_SC_WINDOW_OFFSET and PA_SC_SCREEN_SCISSOR_TL/BR at draw time; ' +
        'refusing to draw with fabricated raster state');
    }
    const width = surface?.width ?? 0, height = surface?.height ?? 0;
    if (!(width > 0) || !(height > 0)) {
      throw new XenosRenderStateError(`scissor surface has no extent (${width}x${height})`);
    }
    const tl = raw.windowScissorTl >>> 0;
    const br = raw.windowScissorBr >>> 0;
    const off = raw.windowOffset >>> 0;
    let tlX = u14(tl), tlY = u14(tl >>> 16);
    let brX = u14(br), brY = u14(br >>> 16);
    if ((tl & 0x80000000) === 0) {
      // window_offset_disable (bit 31) clear: WINDOW offset applies.
      const wx = s15(off), wy = s15(off >>> 16);
      tlX += wx; tlY += wy; brX += wx; brY += wy;
    }
    const stl = raw.screenScissorTl >>> 0, sbr = raw.screenScissorBr >>> 0;
    tlX = Math.max(tlX, s15(stl)); tlY = Math.max(tlY, s15(stl >>> 16));
    brX = Math.min(brX, s15(sbr)); brY = Math.min(brY, s15(sbr >>> 16));
    // Collapse negatives and inverted rects into a zero-sized one (xenia).
    tlX = Math.max(tlX, 0); tlY = Math.max(tlY, 0);
    brX = Math.max(brX, tlX); brY = Math.max(brY, tlY);
    // Clamp to the surface — WebGPU validation requires the scissor inside the attachment.
    const cx = Math.min(tlX, width), cy = Math.min(tlY, height);
    const cxBr = Math.min(brX, width), cyBr = Math.min(brY, height);
    return { x: cx, y: cy, width: Math.max(cxBr - cx, 0), height: Math.max(cyBr - cy, 0) };
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
    /** The alpha-test decision from raw RB_COLORCONTROL alone: { compare } or null (xenia's rules: bit 3 enable, bits 0-2 CompareFunction, kAlways disables). */
    decodeAlphaTest(colorControl) {
      const c = colorControl >>> 0;
      const enabled = (c & 0x8) !== 0;
      const func = c & 0x7;
      return enabled && func !== 7 ? { compare: func } : null;
    },
    buildRenderTargetState,
    decodeViewportState,
    decodeRasterState,
    PA_SU_SC_MODE_CNTL,
    PA_CL_CLIP_CNTL, PA_CL_VTE_CNTL, PA_SU_VTX_CNTL,
    PA_CL_VPORT_XSCALE, PA_CL_VPORT_XOFFSET, PA_CL_VPORT_YSCALE, PA_CL_VPORT_YOFFSET,
    PA_CL_VPORT_ZSCALE, PA_CL_VPORT_ZOFFSET,
    PA_SU_POLY_OFFSET_FRONT_SCALE, PA_SU_POLY_OFFSET_FRONT_OFFSET,
    PA_SU_POLY_OFFSET_BACK_SCALE, PA_SU_POLY_OFFSET_BACK_OFFSET,
    PA_SC_WINDOW_SCISSOR_TL, PA_SC_WINDOW_SCISSOR_BR, PA_SC_WINDOW_OFFSET,
    PA_SC_SCREEN_SCISSOR_TL, PA_SC_SCREEN_SCISSOR_BR,
    buildScissorState,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonGpuRenderState = api;

})(typeof self !== 'undefined' ? self : globalThis);
