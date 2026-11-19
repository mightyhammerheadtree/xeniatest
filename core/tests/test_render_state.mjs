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

test('every other color format traps with its xenia name when the PS writes oC0', () => {
  const cases = [
    [2, 'k_2_10_10_10'], [3, 'k_2_10_10_10_FLOAT'], [4, 'k_16_16'], [5, 'k_16_16_16_16'],
    [6, 'k_16_16_FLOAT'], [7, 'k_16_16_16_16_FLOAT'], [10, 'k_2_10_10_10_AS_10_10_10_10'],
    [12, 'k_2_10_10_10_FLOAT_AS_16_16_16_16'], [14, 'k_32_FLOAT'], [15, 'k_32_32_FLOAT'],
  ];
  for (const [format, name] of cases) {
    assert.throws(() => rtState({ colorInfo0: format << 16 }), (e) => e instanceof RS.XenosRenderStateError && e.message.includes(name));
  }
});

test('k_8_8_8_8_GAMMA traps naming the PWL gamma conversion (not silently rendered linear)', () => {
  assert.throws(() => rtState({ colorInfo0: 1 << 16 }), /8_8_8_8_GAMMA/);
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
    rtState({ colorInfo0: 7 << 16 });
    assert.fail('should have thrown');
  } catch (e) {
    assert.ok(e instanceof RS.XenosRenderStateError);
    assert.match(e.message, /k_16_16_16_16_FLOAT/);
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
