/**
 * Render-target register decode tests — public/js/gpu/render_state.js.
 *
 * Pure decode of the RB_* register snapshot into WebGPU pipeline/pass state,
 * mirroring xenia's semantics (BSD-3-Clause reference in the module header).
 * No device, no mocks of Dawn: these assert the *state we ask WebGPU for*;
 * the pipeline-creation path is asserted in test_gpu_translated_draw.mjs with
 * a recording fake, and the WGSL itself is Dawn-validated elsewhere.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../../public/js/gpu/render_state.js';

const RS = globalThis.XenonGpuRenderState;

const ZERO_RT = {
  surfaceInfo: 0,
  colorInfo0: 0, colorInfo1: 0, colorInfo2: 0, colorInfo3: 0,
  depthInfo: 0,
  colorMask: 0x0000FFFF,
  blendRed: 0, blendGreen: 0, blendBlue: 0, blendAlpha: 0,
  stencilRefMask: 0, stencilRefMaskBf: 0,
  depthControl: 0,
  blendControl0: 0x00010001, blendControl1: 0x00010001, blendControl2: 0x00010001, blendControl3: 0x00010001,
  colorControl: 0,
  alphaRef: 0,
};
const rt = (overrides = {}) => ({ ...ZERO_RT, ...overrides });
const rtState = (overrides = {}, opts = {}) => RS.buildRenderTargetState(rt(overrides), { psExportsColor0: true, psExportsDepth: false, ...opts });

// ------------------------------------------------------------ color info
test('RB_COLOR_INFO decodes base, format nibble, bias', () => {
  const info = RS.decodeColorInfo((0x123 & 0x7FF) | (0x2 << 11 & 0x800) | (5 << 16) | (3 << 20));
  assert.equal(info.format, 5);
  assert.equal(info.formatName, 'k_16_16_16_16');
  assert.equal(info.componentCount, 4);
  assert.equal(info.expBias, 3);
  assert.equal(info.colorBase, 0x123);
});

test('unknown color-format nibbles decode to a named, unrecognized value (and trap when written)', () => {
  for (const bad of [9, 11, 13]) {
    const info = RS.decodeColorInfo(bad << 16);
    assert.match(info.formatName, /unrecognized/);
    assert.throws(() => rtState({ colorInfo0: bad << 16 }), (e) => e instanceof RS.XenosRenderStateError && /unrecognized|k_/.test(e.message));
  }
});

test('every guest color format maps to a WebGPU color target exactly like xenia maps it', () => {
  // xenia: RenderTargetCache::GetColorDXGIFormat + GetBaseColorFormat.
  // null = the canvas format itself (8-bit unorm RGBA).
  const cases = [
    [0, 'k_8_8_8_8', null],
    [1, 'k_8_8_8_8_GAMMA', null],
    [2, 'k_2_10_10_10', 'rgb10a2unorm'],
    [3, 'k_2_10_10_10_FLOAT', 'rgba16float'],
    [6, 'k_16_16_FLOAT', 'rg16float'],
    [7, 'k_16_16_16_16_FLOAT', 'rgba16float'],
    [10, 'k_2_10_10_10_AS_10_10_10_10', 'rgb10a2unorm'], // xenia's base format
    [12, 'k_2_10_10_10_FLOAT_AS_16_16_16_16', 'rgba16float'], // xenia's base format
    [14, 'k_32_FLOAT', 'r32float'],
    [15, 'k_32_32_FLOAT', 'rg32float'],
  ];
  for (const [format, name, webgpu] of cases) {
    const state = rtState({ colorInfo0: format << 16 });
    assert.equal(state.colorFormatName, name, name);
    assert.equal(state.colorFormat, webgpu, name);
  }
});

test('the 16-bit snorm color formats trap: WebGPU exposes them only with an optional feature', () => {
  for (const [format, name] of [[4, 'k_16_16'], [5, 'k_16_16_16_16']]) {
    assert.throws(
      () => rtState({ colorInfo0: format << 16 }),
      (e) => e instanceof RS.XenosRenderStateError && e.message.includes(name) && /texture-formats-tier1/.test(e.message),
      name);
  }
});

test('an unknown color format traps with its xenia name', () => {
  assert.throws(
    () => rtState({ colorInfo0: 9 << 16 }),
    (e) => e instanceof RS.XenosRenderStateError && /unrecognized/.test(e.message));
});

test('blending a format WebGPU cannot blend traps (r32float/rg32float need an optional feature)', () => {
  // blendControl0 = 0x00010001 is the "enabled" pattern the harness uses; a
  // real blend needs a non-trivial factor.
  assert.throws(
    () => rtState({ colorInfo0: 14 << 16, blendControl0: 0x00040001 }),
    (e) => e instanceof RS.XenosRenderStateError && /float32-blendable/.test(e.message));
  // the same format without blending is fine
  const state = rtState({ colorInfo0: 14 << 16 });
  assert.equal(state.colorFormat, 'r32float');
});

test('the write mask is restricted to the channels the format has', () => {
  // k_32_FLOAT has one channel: only RED can be written.
  assert.equal(rtState({ colorInfo0: 14 << 16, colorMask: 0x0000000F }).writeMask, 0x1);
  // k_16_16_FLOAT has two: RED|GREEN.
  assert.equal(rtState({ colorInfo0: 6 << 16, colorMask: 0x0000000F }).writeMask, 0x3);
  // four-channel formats keep the guest's mask as-is.
  assert.equal(rtState({ colorInfo0: 0, colorMask: 0x00000005 }).writeMask, 0x5);
});

test('k_8_8_8_8_GAMMA decodes (the PWL gamma encode happens in the translated PS, not here)', () => {
  const state = rtState({ colorInfo0: 1 << 16 });
  assert.equal(state.convertColor0ToGamma, true);
  assert.equal(state.colorFormatName, 'k_8_8_8_8_GAMMA');
  // and with the PS not exporting oC0 — the attachment is never written; the flag still records the format
  const state2 = RS.buildRenderTargetState(rt({ colorInfo0: 1 << 16 }), { psExportsColor0: false, psExportsDepth: false });
  assert.equal(state2.convertColor0ToGamma, true);
});

test('k_8_8_8_8 (0) does not gamma-encode', () => {
  assert.equal(rtState({ colorInfo0: 0 }).convertColor0ToGamma, false);
});

test('format traps are skipped when the pixel shader does not export oC0 (no MSAA garbage-writes)', () => {
  // GAMMA RT0, but the PS exports no color — the attachment is never written, so it is not validated.
  const state = RS.buildRenderTargetState(rt({ colorInfo0: 1 << 16 }), { psExportsColor0: false, psExportsDepth: false });
  assert.equal(state.colorFormatName, 'k_8_8_8_8_GAMMA');
});

// ------------------------------------------------------------ write mask
test('RB_COLOR_MASK nibble 0 becomes the RT write mask; others are ignored here', () => {
  assert.equal(rtState({ colorMask: 0x0000000F }).writeMask, 0xF);
  assert.equal(rtState({ colorMask: 0x00000035 }).writeMask, 0x5, 'only RT0 (bits 0-3) is the canvas target');
  assert.equal(rtState({ colorMask: 0x00000F05 }).writeMask, 0x5, 'lower nibble is RT0; higher nibbles do not leak in');
});

test('normalizeWriteMask pads non-existent components to 1 (xenia slow-path avoidance)', () => {
  assert.equal(RS.normalizeWriteMask(0b0011, 2), 0b1111);
  assert.equal(RS.normalizeWriteMask(0b0001, 1), 0b1111);
  assert.equal(RS.normalizeWriteMask(0b0111, 3), 0b1111);
  assert.equal(RS.normalizeWriteMask(0b1111, 4), 0b1111);
});

// ------------------------------------------------------------ blend decode
test('blend factors map exactly like xenia (color slot)', () => {
  const expected = {
    0: 'zero', 1: 'one', 2: 'zero', 3: 'zero', 4: 'src', 5: 'one-minus-src',
    6: 'src-alpha', 7: 'one-minus-src-alpha', 8: 'dst', 9: 'one-minus-dst',
    10: 'dst-alpha', 11: 'one-minus-dst-alpha',
    12: 'constant', 13: 'one-minus-constant', 14: 'constant-alpha', 15: 'one-minus-constant-alpha',
    16: 'src-alpha-saturated', 17: 'zero', 18: 'zero', 19: 'zero', 20: 'zero', 21: 'zero',
    22: 'zero', 23: 'zero', 24: 'zero', 25: 'zero', 26: 'zero', 27: 'zero', 28: 'zero',
    29: 'zero', 30: 'zero', 31: 'zero',
  };
  const expectedAlpha = { ...expected };
  // alpha slot: color factors reinterpreted as alpha variants (xenia's kBlendFactorAlphaMap)
  Object.assign(expectedAlpha, { 4: 'src-alpha', 5: 'one-minus-src-alpha', 8: 'dst-alpha', 9: 'one-minus-dst-alpha' });
  for (const [raw, name] of Object.entries(expected)) {
    // encode raw in both src positions: color slot + alpha slot
    const v = ((+raw & 0x1F) | ((+raw & 0x1F) << 16)) >>> 0;
    const decoded = RS.decodeBlendControl(v);
    assert.equal(decoded.color.srcFactor, name, `color factor ${raw}`);
    assert.equal(decoded.alpha.srcFactor, expectedAlpha[raw], `alpha factor ${raw}`);
  }
});

test('alpha slot reinterprets the color blend factors as alpha variants (xenia kBlendFactorAlphaMap)', () => {
  const cases = { 4: 'src-alpha', 5: 'one-minus-src-alpha', 6: 'src-alpha', 7: 'one-minus-src-alpha', 8: 'dst-alpha', 9: 'one-minus-dst-alpha', 10: 'dst-alpha', 11: 'one-minus-dst-alpha', 12: 'constant', 13: 'one-minus-constant', 14: 'constant-alpha', 15: 'one-minus-constant-alpha', 16: 'src-alpha-saturated' };
  for (const [raw, name] of Object.entries(cases)) {
    const decoded = RS.decodeBlendControl(((+raw & 0x1F) << 16) >>> 0, );
    assert.equal(decoded.alpha.srcFactor, name, `alpha slot factor ${raw}`);
  }
});

test('blend ops map like xenia (add/subtract/min/max/reverse-subtract, others add)', () => {
  const expected = ['add', 'subtract', 'min', 'max', 'reverse-subtract', 'add', 'add', 'add'];
  for (let op = 0; op < 8; op++) {
    assert.equal(RS.decodeBlendControl(op << 5).color.operation, expected[op], `color op ${op}`);
    assert.equal(RS.decodeBlendControl(op << 21).alpha.operation, expected[op], `alpha op ${op}`);
  }
});

test('blend is omitted for the one/zero/add identity (xenia enable condition)', () => {
  assert.equal(rtState().blend, null, 'src=1/dst=0/add + alpha same → blending disabled');
});

test('blend is present when any slot differs from the identity', () => {
  // Classic alpha blend: src-alpha / one-minus-src-alpha, both color and alpha.
  const v = (6 | (0 << 5) | (7 << 8) | (6 << 16) | (0 << 21) | (7 << 24)) >>> 0;
  const s = rtState({ blendControl0: v });
  assert.deepEqual(s.blend, {
    color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  });
  // Differing only in alpha still enables (xenia checks the alpha equation too)
  const alphaOnly = (1 | (0 << 5) | (0 << 8) | (6 << 16) | (0 << 21) | (7 << 24)) >>> 0;
  assert.notEqual(rtState({ blendControl0: alphaOnly }).blend, null);
  // Differing only in the color operation enables
  const opOnly = (1 | (2 << 5) | (0 << 8) | (1 << 16)) >>> 0; // min op, identity factors
  assert.notEqual(rtState({ blendControl0: opOnly }).blend, null);
});

test('blend constants decode as IEEE-754 floats from raw dwords', () => {
  const s = rtState({ blendRed: 0x3F800000, blendGreen: 0x40000000, blendBlue: 0xBF800000, blendAlpha: 0x41200000 });
  assert.deepEqual(s.blendConstant, [1, 2, -1, 10]);
});

test('writeMask/blend key changes are reflected in the cache key', () => {
  const a = rtState({ colorMask: 0xF });
  const b = rtState({ colorMask: 0x5 });
  assert.notEqual(a.key, b.key);
  assert.equal(rtState({ colorMask: 0xF }).key, a.key, 'same raw values → same key');
});

// ------------------------------------------------------------ depth/stencil
test('zero depth control: depth test always-passes, no writes, no stencil', () => {
  const s = rtState();
  assert.equal(s.depthStencil.format, 'depth24plus-stencil8');
  assert.equal(s.depthStencil.depthCompare, 'always');
  assert.equal(s.depthStencil.depthWriteEnabled, false);
  assert.equal(s.depthStencil.stencilFront, undefined);
  assert.equal(s.depthStencil.stencilReadMask, undefined);
});

test('z_enable + zfunc + z_write map to WebGPU compare (numbering matches)', () => {
  const names = ['never', 'less', 'equal', 'less-equal', 'greater', 'not-equal', 'greater-equal', 'always'];
  for (let func = 0; func < 8; func++) {
    const s = rtState({ depthControl: 0x2 | (func << 4) | 0x4 });
    assert.equal(s.depthStencil.depthCompare, names[func], `zfunc ${func}`);
    assert.equal(s.depthStencil.depthWriteEnabled, true);
  }
});

test('stencil disabled keeps depth-only state', () => {
  const s = rtState({ depthControl: 0x2 | (0x1 << 4) }); // z only
  assert.equal(s.depthStencil.stencilFront, undefined);
});

test('stencil enable maps func/fail/zpass/zfail for both faces (BF off → same as front)', () => {
  // stencil on, z off; func=kLess(1) at +8, fail=kReplace(2) at +11, zpass=kIncrementClamp(3) at +14, zfail=kInvert(5) at +17
  const dc = 0x1 | (1 << 8) | (2 << 11) | (3 << 14) | (5 << 17);
  const s = rtState({ depthControl: dc, stencilRefMask: 0x0000427F }); // ref 0x7F, mask 0x42, writemask 0x00
  assert.deepEqual(s.depthStencil.stencilFront, { compare: 'less', failOp: 'replace', depthFailOp: 'invert', passOp: 'increment-clamp' });
  assert.deepEqual(s.depthStencil.stencilBack, s.depthStencil.stencilFront);
  assert.equal(s.depthStencil.stencilReadMask, 0x42);
  assert.equal(s.depthStencil.stencilWriteMask, 0x00);
  assert.equal(s.stencilRef, 0x7F);
});

test('back-face stencil uses the BF fields when backface_enable is set', () => {
  const dc = 0x1 | 0x80 /* backface_enable */
    | (1 << 8) | (2 << 11) | (3 << 14) | (4 << 17)      // front: less/replace/incr-clamp/decr-clamp
    | (7 << 20) | (6 << 23) | (5 << 26) | (7 << 29);    // back: always/increment-wrap/invert/decrement-wrap
  const s = rtState({ depthControl: dc, stencilRefMask: 0x11, stencilRefMaskBf: 0x11 });
  assert.deepEqual(s.depthStencil.stencilFront, { compare: 'less', failOp: 'replace', depthFailOp: 'decrement-clamp', passOp: 'increment-clamp' });
  assert.deepEqual(s.depthStencil.stencilBack, { compare: 'always', failOp: 'increment-wrap', depthFailOp: 'decrement-wrap', passOp: 'invert' });
});

test('back-face stencil reference may not differ from the front one (WebGPU has one reference per pass)', () => {
  const dc = 0x1 | 0x80;
  assert.throws(() => rtState({ depthControl: dc, stencilRefMask: 0x01, stencilRefMaskBf: 0x02 }), /RB_STENCILREFMASK_BF/);
});

test('equal BF reference is fine', () => {
  const dc = 0x1 | 0x80;
  const s = rtState({ depthControl: dc, stencilRefMask: 0x42, stencilRefMaskBf: 0x42 });
  assert.equal(s.stencilRef, 0x42);
});

// ------------------------------------------------------------ traps
test('kD24FS8 depth traps when depth/stencil is active', () => {
  const fs8 = 1 << 16;
  assert.throws(() => rtState({ depthInfo: fs8, depthControl: 0x2 }), /kD24FS8/);
  assert.throws(() => rtState({ depthInfo: fs8, depthControl: 0x1 }), /kD24FS8/);
  assert.doesNotThrow(() => rtState({ depthInfo: fs8, depthControl: 0 }), 'inactive depth state is not decoded');
});

test('kD24FS8 depth traps when the PS exports oDepth', () => {
  assert.throws(() => rtState({ depthInfo: 1 << 16 }, { psExportsDepth: true }), /kD24FS8/);
});

test('alpha test decodes (applied by the translated PS, not pipeline state)', () => {
  const st = rtState({ colorControl: 0x8 | 0x1, alphaRef: 0x3F000000 }); // enable | kLess
  assert.deepEqual(st.alphaTest, { compare: 1, referenceDword: 0x3F000000 });
  // kAlways (7): always passes — decode treats the test as disabled (xenia skips it)
  assert.equal(rtState({ colorControl: 0x8 | 0x7, alphaRef: 0 }).alphaTest, null);
  // disabled (bit 3 clear): no alpha test even with function bits set
  assert.equal(rtState({ colorControl: 0x6 }).alphaTest, null);
  // every Xenos CompareFunction decodes 0..6 and alphaTest only appears with bit 3
  for (let compare = 0; compare <= 7; compare++) {
    const st = rtState({ colorControl: 0x8 | compare, alphaRef: 0x3F800000 });
    if (compare === 7) assert.equal(st.alphaTest, null);
    else assert.equal(st.alphaTest.compare, compare);
  }
  // alpha-to-coverage on its own is not alpha test — it is a no-op at 1x MSAA and does not trap
  assert.doesNotThrow(() => rtState({ colorControl: 0x10, alphaRef: 0 }));
  // enabled without a reference in the snapshot traps (no fabricated reference)
  assert.throws(() => rtState({ colorControl: 0x8 | 0x1, alphaRef: undefined }), /alphaRef/);
});

test('a missing snapshot traps with a named reason (no fabricated ROP state)', () => {
  assert.throws(() => RS.buildRenderTargetState(undefined, {}), /params.renderTargets/);
  assert.throws(() => RS.buildRenderTargetState(null, {}), /params.renderTargets/);
});

test('the error class is stable and carries the message', () => {
  try {
    rtState({ colorInfo0: 5 << 16 }); // k_16_16_16_16 — the optional-format trap
    assert.fail('should have thrown');
  } catch (e) {
    assert.ok(e instanceof RS.XenosRenderStateError);
    assert.match(e.message, /k_16_16_16_16/);
  }
});

// ------------------------------------------------------------ scissor (PA_SC_*)
const ZERO_SCISSOR = {
  windowScissorTl: 0,
  windowScissorBr: 0x00400040, // 64x64 (u14 fields)
  windowOffset: 0,
  screenScissorTl: 0,
  screenScissorBr: 0x20002000, // reset default: 8192x8192
};
const sc = (overrides = {}) => ({ ...ZERO_SCISSOR, ...overrides });
const scState = (overrides = {}, width = 64, height = 64) => RS.buildScissorState(sc(overrides), { width, height });

test('scissor decodes a full-surface rect', () => {
  assert.deepEqual(scState(), { x: 0, y: 0, width: 64, height: 64 });
});

test('window scissor br is exclusive (extent = br - tl)', () => {
  assert.deepEqual(
    scState({ windowScissorTl: (10 << 16) | 8, windowScissorBr: (40 << 16) | 32 }),
    { x: 8, y: 10, width: 24, height: 30 },
  );
});

test('window offset is added when the TL disable bit is clear; signed 15-bit offsets', () => {
  // offset (+3, +5): rect shifts
  assert.deepEqual(
    scState({ windowScissorTl: 0, windowScissorBr: 0x00400040, windowOffset: (5 << 16) | 3 }),
    { x: 3, y: 5, width: 61, height: 59 },
  );
  // negative offsets collapse at 0 (xenia: max(tl, 0), br = max(br, tl))
  const r = scState({ windowScissorTl: 0, windowScissorBr: 0x00400040, windowOffset: ((-10 & 0x7fff) << 16) | (-4 & 0x7fff) });
  assert.equal(r.x, 0);
  assert.equal(r.y, 0);
  // disable bit (31) skips the offset entirely
  assert.deepEqual(
    scState({ windowScissorTl: 0x80000000, windowScissorBr: 0x00400040, windowOffset: 0x7FFF7FFF }),
    { x: 0, y: 0, width: 64, height: 64 },
  );
});

test('screen scissor clamps the window rect from outside (tl max, br min)', () => {
  // screen rect (16,8)-(48,48): window rect clipped to it
  const r = scState({ screenScissorTl: (8 << 16) | 16, screenScissorBr: (48 << 16) | 48 });
  assert.deepEqual(r, { x: 16, y: 8, width: 32, height: 40 });
  // screen rect strictly inside a larger window rect keeps window bounds where screen is wider
  const r2 = scState({ windowScissorTl: (20 << 16) | 20, windowScissorBr: (30 << 16) | 30, screenScissorTl: 0, screenScissorBr: 0x20002000 });
  assert.deepEqual(r2, { x: 20, y: 20, width: 10, height: 10 });
});

test('inverted rects collapse to a zero-sized scissor (xenia)', () => {
  assert.deepEqual(
    scState({ windowScissorTl: 0x00400040, windowScissorBr: 0 }),
    { x: 64, y: 64, width: 0, height: 0 },
  );
});

test('rect is clamped to the surface (WebGPU requires the scissor inside the attachment)', () => {
  const r = scState({ windowScissorTl: 0, windowScissorBr: 0x1FFF1FFF }, 64, 64); // way past the canvas
  assert.deepEqual(r, { x: 0, y: 0, width: 64, height: 64 });
  const r2 = scState({ windowScissorTl: (100 << 16) | 100, windowScissorBr: 0x00400040 }, 64, 64);
  assert.deepEqual(r2, { x: 64, y: 64, width: 0, height: 0 });
});

test('window_offset_disable is only bit 31 (other TL bits are the u14 coords)', () => {
  const r = scState({ windowScissorTl: 0x80000005 | (7 << 16), windowScissorBr: 0x00400040, windowOffset: (100 << 16) | 100 });
  assert.equal(r.x, 5);
  assert.equal(r.y, 7);
});

test('a missing snapshot traps with a named reason (no fabricated raster state)', () => {
  assert.throws(() => RS.buildScissorState(undefined, { width: 64, height: 64 }), /params\.scissor/);
  assert.throws(() => RS.buildScissorState(null, { width: 64, height: 64 }), /params\.scissor/);
  assert.throws(() => RS.buildScissorState({}, { width: 0, height: 64 }), /no extent/);
});

// ------------------------------------------------------------ MSAA (RB_SURFACE_INFO)
test('RB_SURFACE_INFO msaa_samples 2x/4x trap loudly naming MSAA (canvas/depth are single-sampled)', () => {
  for (const [samples, label] of [[1, '2x'], [2, '4x']]) {
    assert.throws(() => rtState({ surfaceInfo: samples << 16 }), new RegExp(label + ' MSAA'));
  }
  // 3 (k3X, "this value is not supported on hardware" but must still not pass) — also traps
  assert.throws(() => rtState({ surfaceInfo: 3 << 16 }), /MSAA/);
});

test('RB_SURFACE_INFO k1X renders (default reset state) and exposes the surface pitch', () => {
  const state = rtState({ surfaceInfo: (0 << 16) | 0x0123 });
  assert.equal(state.surfacePitch, 0x0123);
});

test('render-target snapshot without RB_SURFACE_INFO refuses to draw (MSAA decision required)', () => {
  const noSurface = { ...rt() };
  delete noSurface.surfaceInfo;
  assert.throws(() => RS.buildRenderTargetState(noSurface, { psExportsColor0: true }), /RB_SURFACE_INFO/);
});

test('decodeColorInfo decodes color_exp_bias as a SIGNED 6-bit field (xenia int32_t :6)', () => {
  const bias = (v) => RS.decodeColorInfo(v << 20).expBias;
  assert.equal(bias(3), 3);
  assert.equal(bias(-2 & 0x3F), -2, '0b111110 decodes as -2, not 62');
  assert.equal(bias(0x20), -32, '0b100000 is -32');
  assert.equal(bias(0x1F), 31);
});

// ------------------------------------------- back-face stencil masks / alpha-to-coverage
test('BF stencil read/write mask mismatch traps naming both masks (WebGPU shares one pair)', () => {
  const front = 0x4255007F; // ref 0x7F, stencilmask 0x55, writemask 0x42
  assert.throws(() => rtState({ stencilRefMask: front, stencilRefMaskBf: (0x66 << 8) | 0x7F, depthControl: 0x1 | 0x80 }), /stencil read\/write mask differs/);
  assert.throws(() => rtState({ stencilRefMask: front, stencilRefMaskBf: (0x42 << 16) | 0x7F & ((0x77 << 8) | 0) | 0, depthControl: 0x1 | 0x80 }), /stencil read\/write mask differs|front/);
  // matching ref and masks — renders
  assert.doesNotThrow(() => rtState({ stencilRefMask: front, stencilRefMaskBf: front, depthControl: 0x1 | 0x80 }));
  // mismatch only matters with stencil + backface enabled
  assert.doesNotThrow(() => rtState({ stencilRefMask: front, stencilRefMaskBf: 0x007F, depthControl: 0x2 }));
});

// ------------------------------------------------------------ rasterizer state (PA_SU_SC_MODE_CNTL / PA_SU_POLY_OFFSET_*)
// xenia d3d12 pipeline_cache.cc "Rasterizer state" semantics.
const F32 = (x) => new Float32Array(new Float32Array([x]).buffer)[0];
const bits = (x) => new Uint32Array(new Float32Array([x]).buffer)[0];

test('rasterizer state defaults (registers zeroed): no culling, CCW front faces, no depth bias, no drop', () => {
  const s = rtState({});
  assert.deepEqual(s.raster, { cullMode: 'none', frontFace: 'ccw', depthBias: 0, depthBiasSlopeScale: 0, depthBiasClamp: 0, bothCull: false });
});

test('rasterizer state: face bit (bit 2) selects CW front faces', () => {
  const s = rtState({ paSuScModeCntl: 0x4 }, { primitivePolygonal: true });
  assert.equal(s.raster.frontFace, 'cw');
  const s2 = rtState({ paSuScModeCntl: 0x0 }, { primitivePolygonal: true });
  assert.equal(s2.raster.frontFace, 'ccw');
});

test('rasterizer state: cull_back / cull_front select the WebGPU cull mode (polygonal only)', () => {
  assert.equal(rtState({ paSuScModeCntl: 0x2 }, { primitivePolygonal: true }).raster.cullMode, 'back');
  assert.equal(rtState({ paSuScModeCntl: 0x1 }, { primitivePolygonal: true }).raster.cullMode, 'front');
  // points/lines: never culled (xenia: filled front faces only)
  assert.equal(rtState({ paSuScModeCntl: 0x2 }, { primitivePolygonal: false }).raster.cullMode, 'none');
  assert.equal(rtState({ paSuScModeCntl: 0x1 }, { primitivePolygonal: false }).raster.cullMode, 'none');
});

test('rasterizer state: both faces culled drops the draw (no fragments), does not throw', () => {
  const s = rtState({ paSuScModeCntl: 0x3 }, { primitivePolygonal: true });
  assert.equal(s.raster.bothCull, true);
  // same registers, non-polygonal primitive: no culling at all
  const s2 = rtState({ paSuScModeCntl: 0x3 }, { primitivePolygonal: false });
  assert.equal(s2.raster.bothCull, false);
  assert.equal(s2.raster.cullMode, 'none');
});

test('rasterizer state: kDualMode with a non-triangle polygon on a non-culled face traps (WebGPU has no wireframe)', () => {
  // poly_mode=1 (kDualMode) at bits 3-4, front ptype = kLines (1) at bits 5-7
  const mode = (1 << 3) | (1 << 5);
  assert.throws(() => rtState({ paSuScModeCntl: mode }, { primitivePolygonal: true }),
    (e) => e instanceof RS.XenosRenderStateError && /kDualMode/.test(e.message));
  // same but the front face is culled and the back face is triangles — no wireframe to emit
  const culled = (1 << 3) | (1 << 5) | (2 << 8) | 0x1;
  assert.doesNotThrow(() => rtState({ paSuScModeCntl: culled }, { primitivePolygonal: true }));
  // front = triangles (2), back = kPoints (0) at bits 8-10 → the back face traps
  const backWire = (1 << 3) | (2 << 5);
  assert.throws(() => rtState({ paSuScModeCntl: backWire }, { primitivePolygonal: true }),
    (e) => e instanceof RS.XenosRenderStateError && /back/.test(e.message));
  // kDualMode with both faces kTriangles (2): renders filled triangles, no trap (4541096E lesson)
  const dualTri = (1 << 3) | (2 << 5) | (2 << 8);
  assert.doesNotThrow(() => rtState({ paSuScModeCntl: dualTri }, { primitivePolygonal: true }));
  // poly_mode != kDualMode ignores the ptype fields entirely (4541096E uses poly_mode=2 + ptype 0)
  const reserved = (2 << 3) | (0 << 5) | (0 << 8);
  assert.doesNotThrow(() => rtState({ paSuScModeCntl: reserved }, { primitivePolygonal: true }));
});

test('rasterizer state: polygon offset (front registers) maps to depthBias in UNORM24 units', () => {
  // front offset = 0.5f, front scale = 2.0f; offset enable bit 11; polygonal
  const mode = 0x800;
  const s = rtState({
    paSuScModeCntl: mode,
    paSuPolyOffsetFrontScale: bits(2.0),
    paSuPolyOffsetFrontOffset: bits(0.5),
  }, { primitivePolygonal: true });
  // xenia GetD3D10IntegerPolygonOffset(kD24S8, 0.5) = ceil(0.5 * 16777215) = 8388608
  assert.equal(s.raster.depthBias, 8388608);
  assert.equal(s.raster.depthBiasSlopeScale, 2.0 / 16.0);
  assert.equal(s.raster.depthBiasClamp, 0);
});

test('rasterizer state: negative offset gives a negative depthBias', () => {
  const s = rtState({
    paSuScModeCntl: 0x800,
    paSuPolyOffsetFrontScale: bits(1.0),
    paSuPolyOffsetFrontOffset: bits(-2.0),
  }, { primitivePolygonal: true });
  assert.equal(s.raster.depthBias, -Math.ceil(2.0 * 16777215));
});

test('xenia polygon offset: back registers are used only when front is disabled/culled and decodes to zero', () => {
  // back enabled, front disabled → back registers
  const backOnly = rtState({
    paSuScModeCntl: 0x1000, // back offset enable only
    paSuPolyOffsetBackScale: bits(4.0),
    paSuPolyOffsetBackOffset: bits(1.0),
  }, { primitivePolygonal: true });
  assert.equal(backOnly.raster.depthBiasSlopeScale, 4.0 / 16.0);
  assert.equal(backOnly.raster.depthBias, Math.ceil(1.0 * 16777215));
  // both enabled, front decodes non-zero → front wins
  const frontWins = rtState({
    paSuScModeCntl: 0x800 | 0x1000,
    paSuPolyOffsetFrontScale: bits(1.0),
    paSuPolyOffsetFrontOffset: bits(1.0),
    paSuPolyOffsetBackScale: bits(8.0),
    paSuPolyOffsetBackOffset: bits(1.0),
  }, { primitivePolygonal: true });
  assert.equal(frontWins.raster.depthBiasSlopeScale, 1.0 / 16.0);
  // both enabled, front decodes to exactly 0/0 → back fills in (xenia !scale && !offset rule)
  const backFills = rtState({
    paSuScModeCntl: 0x800 | 0x1000,
    paSuPolyOffsetFrontScale: bits(0.0),
    paSuPolyOffsetFrontOffset: bits(0.0),
    paSuPolyOffsetBackScale: bits(8.0),
    paSuPolyOffsetBackOffset: bits(0.0),
  }, { primitivePolygonal: true });
  assert.equal(backFills.raster.depthBiasSlopeScale, 8.0 / 16.0);
  // both enabled but front faces culled → back registers (xenia also requires the back enable bit)
  const frontCulled = rtState({
    paSuScModeCntl: 0x800 | 0x1000 | 0x1, // front+back offset enable + cull front
    paSuPolyOffsetFrontScale: bits(1.0),
    paSuPolyOffsetFrontOffset: bits(1.0),
    paSuPolyOffsetBackScale: bits(8.0),
    paSuPolyOffsetBackOffset: bits(0.0),
  }, { primitivePolygonal: true });
  assert.equal(frontCulled.raster.depthBiasSlopeScale, 8.0 / 16.0);
});

test('rasterizer state: para polygon offset (points/lines) uses the FRONT register pair', () => {
  const s = rtState({
    paSuScModeCntl: 0x2000, // para offset enable
    paSuPolyOffsetFrontScale: bits(2.0),
    paSuPolyOffsetFrontOffset: bits(1.0),
  }, { primitivePolygonal: false });
  assert.equal(s.raster.depthBiasSlopeScale, 2.0 / 16.0);
  assert.equal(s.raster.depthBias, Math.ceil(1.0 * 16777215));
});

test('rasterizer state: kD24FS8 depth uses the float24-ULP offset factor (xenia GetD3D10IntegerPolygonOffset)', () => {
  const s = rtState({
    depthInfo: 1 << 16, // kD24FS8 (but no depth writes/tests → the D24FS8 trap does not fire)
    paSuScModeCntl: 0x800,
    paSuPolyOffsetFrontScale: bits(0.0),
    paSuPolyOffsetFrontOffset: bits(1.0),
  }, { primitivePolygonal: true });
  // ceil(1.0 * 2^24 * (1/8)) = ceil(2^21) = 2097152, then float24→float32 ULP: <<= 3
  assert.equal(s.raster.depthBias, 2097152 * 8);
});

test('rasterizer state: decode result is part of the render-target state cache key', () => {
  const a = rtState({ paSuScModeCntl: 0x2 }, { primitivePolygonal: true });
  const b = rtState({ paSuScModeCntl: 0x1 }, { primitivePolygonal: true });
  const c = rtState({ paSuScModeCntl: 0x2 }, { primitivePolygonal: false });
  assert.notEqual(a.key, b.key);
  assert.notEqual(a.key, c.key);
  assert.equal(a.key, rtState({ paSuScModeCntl: 0x2 }, { primitivePolygonal: true }).key);
});

// ------------------------------------------------------------ viewport decode (PA_CL_VTE_CNTL / PA_CL_CLIP_CNTL / PA_CL_VPORT_* / PA_SU_VTX_CNTL)
const VIEWPORT_ZERO = {
  paClVteCntl: 0, paClClipCntl: 0, paSuVtxCntl: 0, paSuScModeCntl: 0,
  paClVportXscale: 0, paClVportXoffset: 0, paClVportYscale: 0, paClVportYoffset: 0,
  paClVportZscale: 0, paClVportZoffset: 0,
};
const vp = (overrides = {}, scissor = {}, surface = { width: 640, height: 480 }) =>
  RS.decodeViewportState({ ...VIEWPORT_ZERO, ...overrides }, scissor, surface);
const f32 = (x) => new Float32Array(new Float32Array([x]).buffer)[0];

test('viewport: zeroed registers map NDC across the whole canvas (kD3DZero half-pixel on by default, GL z remap)', () => {
  const v = vp();
  const kx = 2 / 640, ky = 2 / 480;
  assert.equal(v.scale[0], f32(kx));
  assert.equal(v.scale[1], f32(ky));
  // pix_center is kD3DZero (bit 0 clear) → +0.5 pixel offsets (xenia half_pixel_offset default)
  assert.equal(v.offset[0], f32(0.5 * kx - 1));
  assert.equal(v.offset[1], f32(0.5 * ky - 1));
  // dx_clip_space_def clear → GL-style z remap, zmin/zmax = saturate(0)/saturate(1)
  assert.equal(v.scale[2], f32(0.5));
  assert.equal(v.offset[2], f32(0.5));
});

test('viewport: kOGLHalf pixel centers (PA_SU_VTX_CNTL.pix_center=1) disable the half-pixel offset', () => {
  const v = vp({ paSuVtxCntl: 1 });
  const kx = 2 / 640, ky = 2 / 480;
  assert.equal(v.offset[0], f32(-1));
  assert.equal(v.offset[1], f32(-1));
  assert.equal(v.scale[0], f32(kx));
  assert.equal(v.scale[1], f32(ky));
});

test('viewport: dx_clip_space_def passes the guest z through scale/offset with a full-range host depth', () => {
  const v = vp({ paClClipCntl: 0x80000, paClVteCntl: 0x30, paClVportZscale: bits(0.5), paClVportZoffset: bits(0.25) });
  assert.equal(v.scale[2], f32(0.5));
  assert.equal(v.offset[2], f32(0.25));
});

test('viewport: clip_disable also passes the guest z through (screen-space draws)', () => {
  const v = vp({ paClClipCntl: 0x10000, paClVteCntl: 0x30, paClVportZscale: bits(0.7), paClVportZoffset: bits(0.3) });
  assert.equal(v.scale[2], f32(0.7));
  assert.equal(v.offset[2], f32(0.3));
});

test('viewport: GL-style z (dx=0) folds the −W..W → 0..W remap and [saturate(oz−sz), saturate(oz+sz)] host range', () => {
  // oz=0.75, sz=0.5 → zmin=saturate(0.25)=0.25, zmax=saturate(1.25)=1.0 → scale 0.375, offset 0.625
  const v = vp({ paClVteCntl: 0x30, paClVportZscale: bits(0.5), paClVportZoffset: bits(0.75) });
  assert.equal(v.scale[2], f32(0.375));
  assert.equal(v.offset[2], f32(0.625));
  // canonical GL range: oz=0, sz=1 (identity in guest clip) → full host range
  const v0 = vp({ paClVteCntl: 0x30, paClVportZscale: bits(1.0), paClVportZoffset: bits(0.0) });
  assert.equal(v0.scale[2], f32(0.5));
  assert.equal(v0.offset[2], f32(0.5));
  // guest asking for the 0..1 end state: ZSCALE=0.5/ZOFFSET=0.5 → identity passthrough semantics
  const v1 = vp({ paClVteCntl: 0x30, paClVportZscale: bits(0.5), paClVportZoffset: bits(0.5) });
  assert.equal(v1.scale[2], f32(0.5));
  assert.equal(v1.offset[2], f32(0.5));
  // oz=-1, sz=4 → zmin=0, zmax=1 → the full range
  const v2 = vp({ paClVteCntl: 0x30, paClVportZscale: bits(4.0), paClVportZoffset: bits(-1.0) });
  assert.equal(v2.scale[2], f32(0.5));
  assert.equal(v2.offset[2], f32(0.5));
});

test('viewport: x scale/offset enable bits (and signs) produce the composed NDC coefficients', () => {
  const v = vp({ paSuVtxCntl: 1, paClVteCntl: 0x3, paClVportXscale: bits(320.0), paClVportXoffset: bits(160.0) });
  const kx = 2 / 640;
  assert.equal(v.scale[0], f32(320.0 * kx));
  assert.equal(v.offset[0], f32(160.0 * kx - 1));
  // negative y scale (guest flip) is preserved verbatim
  const v2 = vp({ paSuVtxCntl: 1, paClVteCntl: 0x4, paClVportYscale: bits(-1.0) });
  assert.equal(v2.scale[1], f32(-(2 / 480)));
});

test('viewport: PA_SC_WINDOW_OFFSET applies only when vtx_window_offset_enable is set', () => {
  // window offset x=16, y=-2 → packed s15 pair 0x7FFE0010
  const scissor = { windowOffset: 0x7FFE0010 };
  const on = vp({ paSuVtxCntl: 1, paSuScModeCntl: 0x10000, paClVteCntl: 0xF }, scissor);
  const kx = 2 / 640, ky = 2 / 480;
  assert.equal(on.offset[0], f32(16 * kx - 1));
  assert.equal(on.offset[1], f32(-2 * ky - 1));
  // same mode bit clear → the window offset is ignored
  const off = vp({ paSuVtxCntl: 1, paSuScModeCntl: 0x0, paClVteCntl: 0xF }, scissor);
  assert.equal(off.offset[0], f32(-1));
  assert.equal(off.offset[1], f32(-1));
});

test('viewport: user clip planes TRAP loudly (WebGPU has no clip planes)', () => {
  assert.throws(() => vp({ paClClipCntl: 0x00000001 }), (e) => e instanceof RS.XenosRenderStateError && /ucp_ena/.test(e.message));
  assert.throws(() => vp({ paClClipCntl: 0x0000003F }), (e) => e instanceof RS.XenosRenderStateError && /ucp_ena/.test(e.message));
  // ucp bits unrelated to the low 6 do not trap
  assert.doesNotThrow(() => vp({ paClClipCntl: 0x10000 }));
});
