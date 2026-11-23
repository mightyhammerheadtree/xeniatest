/**
 * WGSL translator tests — Xenos ucode → WGSL (vertex shaders, v1 subset).
 *
 * Every generated shader that is expected to compile is validated with Dawn
 * (the actual WebGPU compiler) through core/tests/helpers/dawn_validate.mjs —
 * not just parsed. Traps assert on the error class AND message; nothing is
 * skipped and no test relies on a placeholder.
 *
 * ucode fixtures are hand-assembled from xenia's ucode.h bit layouts (same
 * conventions as test_vertex_fetch.mjs).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../../public/js/gpu/vertex_fetch.js';
import '../../public/js/gpu/wgsl_translate.js';
import { assertValidWgsl, checkProgramLinks } from './helpers/dawn_validate.mjs';

const T = globalThis.XenonGpuTranslate;

// ---------------------------------------------------------------- ucode helpers
function execCf(address, count, sequence, opcode = 2 /* kExecEnd */, extraW1 = 0) {
  return {
    dword0: (address & 0xFFF) | ((count & 0x7) << 12) | ((sequence & 0xFFF) << 16),
    dword1: ((opcode & 0xF) << 12) | (extraW1 & 0x0FFF),
  };
}
/** cexec/cexece (bool-conditional exec). */
function condExecCf(address, count, sequence, boolAddress, condition, end = false) {
  return execCf(address, count, sequence, end ? 4 /* kCondExecEnd */ : 3 /* kCondExec */,
    ((boolAddress & 0xFF) << 2) | ((condition & 1) << 10));
}
/** cexec_pred (predicated exec). */
function predExecCf(address, count, sequence, condition, end = false) {
  return execCf(address, count, sequence, end ? 6 /* kCondExecPredEnd */ : 5 /* kCondExecPred */,
    (condition & 1) << 10);
}
const NOP_CF = { dword0: 0, dword1: 0 };
function cfPair(a, b) {
  return [
    a.dword0 >>> 0,
    (a.dword1 & 0xFFFF) | ((b.dword0 & 0xFFFF) << 16),
    ((b.dword0 >>> 16) & 0xFFFF) | ((b.dword1 & 0xFFFF) << 16),
  ];
}
function vfetchWords({
  dstReg = 1, srcReg = 0, constIndex = 0, constIndexSel = 0, dstSwiz = 0x688,
  format = 57, stride = 1, offset = 0, isPredicated = 0,
} = {}) {
  const w0 = ((srcReg & 0x3F) << 5) | ((dstReg & 0x3F) << 12) | (1 << 19) | ((constIndex & 0x1F) << 20) | ((constIndexSel & 0x3) << 25);
  const w1 = (dstSwiz & 0xFFF) | (format << 16) | ((isPredicated & 1) << 31);
  const w2 = (stride & 0xFF) | ((offset & 0x7FFFFF) << 8);
  return [w0 >>> 0, w1 >>> 0, w2 >>> 0];
}
/** Texture fetch instruction (opcode 1) — VS tfetch. */
function tfetchWords({ dstReg = 5 } = {}) {
  const w0 = ((dstReg & 0x3F) << 12) | (1 << 19) | 1; // opcode 1 (texture)
  return [w0 >>> 0, 0x03800000, 0x00000000];
}
const temp = (reg) => ({ sel: 1, reg: reg & 0x3F });
const tempAbs = (reg) => ({ sel: 1, reg: (reg & 0x3F) | 0x80 }); // absolute-value temp
const tempRel = (reg) => ({ sel: 1, reg: (reg & 0x3F) | 0x40 }); // +aL relative temp
const konst = (index, { abs = false } = {}) => ({ sel: 0, reg: index & 0xFF, absConst: abs ? 1 : 0 });

function aluWords({
  vectorDest = 0, scalarDest = 0, isExport = 0, vectorWriteMask = 0xF,
  scalarWriteMask = 0, vectorClamp = 0, scalarClamp = 0, scalarOpcode = 50 /* retain_prev */,
  vectorOpcode = 2 /* max */, predicated = 0, predicateCondition = 0,
  vectorDestRelative = 0, scalarDestRelative = 0, absConstants = 0, constRelAbs = 0,
  src = [temp(0), temp(1), temp(1)],
} = {}) {
  const swz = src.map((s) => s.swizzle ?? 0);
  const neg = src.map((s) => s.negate ?? 0);
  let w0 = ((vectorDest & 0x3F)
    | ((vectorDestRelative & 1) << 6)
    | ((absConstants & 1) << 7)
    | ((scalarDest & 0x3F) << 8)
    | ((scalarDestRelative & 1) << 14)
    | ((isExport & 1) << 15)
    | ((vectorWriteMask & 0xF) << 16)
    | ((scalarWriteMask & 0xF) << 20)
    | ((vectorClamp & 1) << 24)
    | ((scalarClamp & 1) << 25)
    | ((scalarOpcode & 0x3F) << 26)) >>> 0;
  if (src.some((s) => s.absConst)) w0 = (w0 | (1 << 7)) >>> 0;
  const w1 = ((swz[2] & 0xFF)
    | ((swz[1] & 0xFF) << 8)
    | ((swz[0] & 0xFF) << 16)
    | ((neg[2] & 1) << 24)
    | ((neg[1] & 1) << 25)
    | ((neg[0] & 1) << 26)
    | ((predicateCondition & 1) << 27)
    | ((predicated & 1) << 28)
    | ((constRelAbs & 1) << 30)) >>> 0;
  const w2 = ((src[2].reg & 0xFF)
    | ((src[1].reg & 0xFF) << 8)
    | ((src[0].reg & 0xFF) << 16)
    | ((vectorOpcode & 0x1F) << 24)
    | ((src[2].sel & 1) << 29)
    | ((src[1].sel & 1) << 30)
    | ((src[0].sel & 1) << 31)) >>> 0;
  return [w0 >>> 0, w1 >>> 0, w2 >>> 0];
}
const seq = (...bits) => bits.reduce((s, b, i) => s | (b << (2 * i)), 0);
const SEQ_ALU = 0b00;
const SEQ_FETCH = 0b01;
const shader = (...words) => new Uint32Array(words.flat());

/** A minimal shader: fetch r1, export oPos = add(r1, r1). */
const minimalShader = () => shader(
  cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
  vfetchWords({ dstReg: 1, format: 57 }),
  aluWords({ isExport: 1, vectorDest: 62 /* oPos */, vectorOpcode: 0 /* add */, src: [temp(1), temp(1), temp(1)] }),
);

// ---------------------------------------------------------------- happy path
test('translates a minimal vfetch → oPos shader to valid WGSL', async () => {
  const out = T.translateVertexShader(minimalShader());
  assert.equal(out.type, 'vertex');
  assert.deepEqual(out.inputLocations, [1]);
  assert.equal(out.writesPosition, true);
  assert.equal(out.tempRegisterCount, 2); // r0..r1
  assert.equal(out.usesFloatConstants, false);
  assert.equal(out.usesBoolConstants, false);
  assert.equal(out.predicateUsed, false);
  assert.match(out.wgsl, /@location\(1\) a1: vec4f/);
  assert.match(out.wgsl, /@builtin\(position\) pos: vec4f/);
  assert.match(out.wgsl, /r\[1\] = in\.a1;/);
  assert.match(out.wgsl, /out\.pos = vec4f\(\(r\[1\]\.x \+ r\[1\]\.x\), \(r\[1\]\.y \+ r\[1\]\.y\), \(r\[1\]\.z \+ r\[1\]\.z\), \(r\[1\]\.w \+ r\[1\]\.w\)\);/);
  await assertValidWgsl(out.wgsl);
});

test('attributes whose dest lanes are all kKeep are not bound as inputs', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 3, dstSwiz: 0xFFF /* all kKeep */ }),
    aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 0, src: [temp(3), temp(3), temp(3)] }),
  ));
  assert.deepEqual(out.inputLocations, []);
  assert.doesNotMatch(out.wgsl, /@location\(3\) a3/);
  await assertValidWgsl(out.wgsl);
});

test('constant floats bind fc only when used', async () => {
  const plain = T.translateVertexShader(minimalShader());
  assert.doesNotMatch(plain.wgsl, /var<uniform> fc/);
  const withConst = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, scalarOpcode: 50, src: [temp(1), konst(0), konst(1)] }),
  ));
  assert.match(withConst.wgsl, /@group\(0\) @binding\(0\) var<uniform> fc: array<vec4f, 256>;/);
  assert.equal(withConst.usesFloatConstants, true);
  assert.match(withConst.wgsl, /abs\(fc\[0\]\)|fc\[0\]/);
  await assertValidWgsl(withConst.wgsl);
});

test('interpolator exports become @location(N) outputs', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 3, seq(SEQ_FETCH, SEQ_ALU, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ isExport: 1, vectorDest: 7, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
    aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.deepEqual(out.interpolatorLocations, [7]);
  assert.match(out.wgsl, /@location\(7\) v7: vec4f/);
  assert.match(out.wgsl, /out\.v7 = vec4f\(/);
  await assertValidWgsl(out.wgsl);
});

test('partial export lanes are written individually; untouched lanes keep the zero init', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ isExport: 1, vectorDest: 7, vectorOpcode: 0, vectorWriteMask: 0b0011, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /out\.v7\.x = /);
  assert.match(out.wgsl, /out\.v7\.y = /);
  assert.doesNotMatch(out.wgsl, /out\.v7\.z = /);
  assert.doesNotMatch(out.wgsl, /out\.v7\.w = /);
  // Only the zero-init full-vector store may exist.
  assert.equal((out.wgsl.match(/out\.v7 = /g) ?? []).length, 1);
  assert.match(out.wgsl, /out\.v7 = vec4f\(0\.0\);/);
  await assertValidWgsl(out.wgsl);
});

test('constant 0/1 export lanes (dual-issue vector+scalar masks) emit literal writes', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({
      isExport: 1, vectorDest: 5, vectorOpcode: 0, vectorWriteMask: 0b0011,
      scalarOpcode: 50 /* retain_prev */, scalarWriteMask: 0b0110, scalarDestRelative: 1,
      src: [temp(1), temp(1), temp(1)],
    }),
  ));
  // Lane x: vector only → vector expr; lane y: vector+scalar → constant 1.0;
  // lane z: scalar only → ps (retain_prev); lane w: neither, scalar_dest_rel → constant 0.0.
  assert.match(out.wgsl, /out\.v5\.x = /);
  assert.match(out.wgsl, /out\.v5\.y = 1\.0;/);
  assert.match(out.wgsl, /out\.v5\.z = ps;/);
  assert.match(out.wgsl, /out\.v5\.w = 0\.0;/);
  // No full-vector v5 store other than the zero-init.
  assert.equal((out.wgsl.match(/out\.v5 = /g) ?? []).length, 1);
  await assertValidWgsl(out.wgsl);
});

test('dual-issue lanes write 1.0 even without scalar_dest_rel; untouched lanes are not written', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({
      isExport: 1, vectorDest: 5, vectorOpcode: 0, vectorWriteMask: 0b0011,
      scalarOpcode: 50, scalarWriteMask: 0b0010, scalarDestRelative: 0,
      src: [temp(1), temp(1), temp(1)],
    }),
  ));
  assert.match(out.wgsl, /out\.v5\.y = 1\.0;/);
  assert.doesNotMatch(out.wgsl, /out\.v5\.w = /);
  await assertValidWgsl(out.wgsl);
});

test('position export can never corrupt when never written — member stays zero-initialized', () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 1, scalarOpcode: 50, scalarWriteMask: 0, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.equal(out.writesPosition, false);
  assert.match(out.wgsl, /out\.pos = vec4f\(0\.0\);/);
});

// ---------------------------------------------------------------- vector operators
const vectorOpCases = [
  [0, 'add', /r\[3\] = vec4f\(\(r\[1\]\.x \+ r\[2\]\.x\), \(r\[1\]\.y \+ r\[2\]\.y\)/],
  [1, 'mul', /r\[3\] = vec4f\(\(r\[1\]\.x \* r\[2\]\.x\), \(r\[1\]\.y \* r\[2\]\.y\)/],
  [2, 'max', /select\(r\[2\]\.x, r\[1\]\.x, \(r\[1\]\.x >= r\[2\]\.x\)\)/],
  [3, 'min', /select\(r\[2\]\.x, r\[1\]\.x, \(r\[1\]\.x < r\[2\]\.x\)\)/],
  [4, 'seq', /select\(0\.0, 1\.0, \(r\[1\]\.x == r\[2\]\.x\)\)/],
  [5, 'sgt', /select\(0\.0, 1\.0, \(r\[1\]\.x > r\[2\]\.x\)\)/],
  [6, 'sge', /select\(0\.0, 1\.0, \(r\[1\]\.x >= r\[2\]\.x\)\)/],
  [7, 'sne', /select\(0\.0, 1\.0, \(r\[1\]\.x != r\[2\]\.x\)\)/],
  [8, 'frc', /\(r\[1\]\.x - floor\(r\[1\]\.x\)\)/],
  [9, 'trunc', /trunc\(r\[1\]\.x\)/],
  [10, 'floor', /floor\(r\[1\]\.x\)/],
  [11, 'mad', /fma\(r\[1\]\.x, r\[2\]\.x, r\[1\]\.x\)/],
  [12, 'cndeq', /select\(r\[1\]\.x, r\[2\]\.x, \(r\[1\]\.x == 0\.0\)\)/],
  [13, 'cndge', /select\(r\[1\]\.x, r\[2\]\.x, \(r\[1\]\.x >= 0\.0\)\)/],
  [14, 'cndgt', /select\(r\[1\]\.x, r\[2\]\.x, \(r\[1\]\.x > 0\.0\)\)/],
  [19, 'max4', /select\(/],
  [28, 'dst', /vec4f\(1\.0, /],
];
for (const [opcode, name, pattern] of vectorOpCases) {
  test(`vector op ${name} (${opcode}) translates and passes Dawn`, async () => {
    const src = name === 'frc' || name === 'trunc' || name === 'floor' || name === 'max4'
      ? [temp(1), temp(2), temp(1)]
      : name === 'dst'
        ? [temp(1), temp(2), temp(1)]
        : [temp(1), temp(2), temp(1)];
    const out = T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ vectorDest: 3, vectorOpcode: opcode, src }),
    ));
    assert.match(out.wgsl, pattern, `expected ${pattern} in:\n${out.wgsl}`);
    await assertValidWgsl(out.wgsl);
  });
}

test('dp4 / dp3 / dp2add emit ordered dot-product chains', async () => {
  for (const [opcode, frag] of [[15, '(r[1].x * r[2].x)'], [16, '(r[1].z * r[2].z)'], [17, ' + (r[1].y * r[2].y)']]) {
    const out = T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ vectorDest: 3, vectorOpcode: opcode, src: [temp(1), temp(2), temp(1)] }),
    ));
    assert.match(out.wgsl, new RegExp(frag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
    await assertValidWgsl(out.wgsl);
  }
  // dp2add adds src2.x to the 2-dot product
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 17, src: [temp(1), temp(2), konst(4)] }),
  ));
  assert.match(out.wgsl, /\+ fc\[4\]\.x/);
  await assertValidWgsl(out.wgsl);
});

test('setp_eq_push (vector) writes p0 and a push counter, passes Dawn', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 20 /* setp_eq_push */, src: [temp(1), temp(2), temp(1)] }),
  ));
  assert.equal(out.predicateUsed, true);
  assert.match(out.wgsl, /p0 = \(\(r\[1\]\.w == 0\.0\) && \(r\[2\]\.w == 0\.0\)\);/);
  assert.match(out.wgsl, /select\(r\[1\]\.x, -1\.0, \(r\[1\]\.x == 0\.0\) && \(r\[2\]\.x == 0\.0\)\) \+ 1\.0/);
  await assertValidWgsl(out.wgsl);
});

test('clamped vector results clamp each written lane', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, vectorClamp: 1, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /r\[3\] = vec4f\(clamp\(/); // full mask → single store with per-lane clamps
  await assertValidWgsl(out.wgsl);
});

test('temp registers are allocated across written GPRs only', () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 5, format: 57 }),
    aluWords({ vectorDest: 2, vectorOpcode: 0, src: [temp(5), temp(5), temp(5)] }),
  ));
  assert.equal(out.tempRegisterCount, 6); // r0..r5
  assert.match(out.wgsl, /for \(var i = 0u; i < 6u; i = i \+ 1u\)/);
});

// ---------------------------------------------------------------- scalar operators
const scalarOpCases = [
  [14, 'exp', /exp2\(r\[1\]\.w\)/],
  [15, 'logc', /xenosLogc\(r\[1\]\.w\)/],
  [16, 'log', /log2\(r\[1\]\.w\)/],
  [17, 'rcpc', /xenosRcpc\(r\[1\]\.w\)/],
  [18, 'rcpf', /xenosRcpf\(r\[1\]\.w\)/],
  [19, 'rcp', /\(1\.0 \/ r\[1\]\.w\)/],
  [20, 'rsqc', /xenosRsqc\(r\[1\]\.w\)/],
  [21, 'rsqf', /xenosRsqf\(r\[1\]\.w\)/],
  [22, 'rsq', /inverseSqrt\(r\[1\]\.w\)/],
  [40, 'sqrt', /sqrt\(r\[1\]\.w\)/],
  [48, 'sin', /sin\(r\[1\]\.w\)/],
  [49, 'cos', /cos\(r\[1\]\.w\)/],
  [0, 'adds', /\(r\[1\]\.w \+ r\[1\]\.x\)/],
  [1, 'adds_prev', /\(r\[1\]\.w \+ ps\)/],
  [2, 'muls', /\(r\[1\]\.w \* r\[1\]\.x\)/],
  [3, 'muls_prev', /\(r\[1\]\.w \* ps\)/],
  [5, 'maxs', /select\(r\[1\]\.x, r\[1\]\.w, \(r\[1\]\.w >= r\[1\]\.x\)\)/],
  [6, 'mins', /select\(r\[1\]\.x, r\[1\]\.w, \(r\[1\]\.w < r\[1\]\.x\)\)/],
  [7, 'seqs', /select\(0\.0, 1\.0, \(r\[1\]\.w == 0\.0\)\)/],
  [8, 'sgts', /select\(0\.0, 1\.0, \(r\[1\]\.w > 0\.0\)\)/],
  [9, 'sges', /select\(0\.0, 1\.0, \(r\[1\]\.w >= 0\.0\)\)/],
  [10, 'snes', /select\(0\.0, 1\.0, \(r\[1\]\.w != 0\.0\)\)/],
  [11, 'frcs', /\(r\[1\]\.w - floor\(r\[1\]\.w\)\)/],
  [12, 'truncs', /trunc\(r\[1\]\.w\)/],
  [13, 'floors', /floor\(r\[1\]\.w\)/],
  [25, 'subs', /\(r\[1\]\.w - r\[1\]\.x\)/],
  [26, 'subs_prev', /\(r\[1\]\.w - ps\)/],
];
for (const [opcode, name, pattern] of scalarOpCases) {
  test(`scalar op ${name} (${opcode}) translates and passes Dawn`, async () => {
    const out = T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ vectorOpcode: 2, scalarOpcode: opcode, scalarWriteMask: 0xF, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
    ));
    assert.match(out.wgsl, pattern, `expected ${pattern} in:\n${out.wgsl}`);
    await assertValidWgsl(out.wgsl);
  });
}

test('scalar full-mask broadcast writes vec4f to the dest GPR', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 0, scalarWriteMask: 0xF, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /r\[4\] = vec4f\(\(r\[1\]\.w \+ r\[1\]\.x\)\);/);
  await assertValidWgsl(out.wgsl);
});

test('scalar partial mask writes only the written lanes', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 0, scalarWriteMask: 0b0100, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /r\[4\]\.z = /);
  assert.doesNotMatch(out.wgsl, /r\[4\]\.x = /);
  assert.doesNotMatch(out.wgsl, /r\[4\]\.w = /);
  await assertValidWgsl(out.wgsl);
});

test('scalar ops update ps; a later retain_prev can read it', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 3, seq(SEQ_FETCH, SEQ_ALU, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 0 /* adds */, scalarWriteMask: 0, src: [temp(1), temp(1), temp(1)] }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 50 /* retain_prev */, scalarWriteMask: 0xF, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /ps = \(r\[1\]\.w \+ r\[1\]\.x\);/);
  assert.match(out.wgsl, /r\[4\] = vec4f\(ps\);/);
  await assertValidWgsl(out.wgsl);
});

test('setp_eq writes ps=0/1 inverse predicate and p0', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 27 /* setp_eq */, scalarWriteMask: 0, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.equal(out.predicateUsed, true);
  assert.match(out.wgsl, /p0 = \(r\[1\]\.w == 0\.0\);/);
  assert.match(out.wgsl, /ps = select\(1\.0, 0\.0, \(r\[1\]\.w == 0\.0\)\);/);
  await assertValidWgsl(out.wgsl);
});

test('setp_clr forces FLT_MAX and p0=false; retain_prev preserves ps', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 33 /* setp_clr */, scalarWriteMask: 0, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /p0 = false;/);
  assert.match(out.wgsl, /ps = 3\.402823466e38;/);
  await assertValidWgsl(out.wgsl);
});

test('setp_inv / setp_pop / setp_rstr translate', async () => {
  for (const [opcode, name, pattern] of [
    [31, 'setp_inv', /p0 = \(r\[1\]\.w == 1\.0\);/],
    [32, 'setp_pop', /p0 = \(\(r\[1\]\.w - 1\.0\) <= 0\.0\);/],
    [34, 'setp_rstr', /p0 = \(r\[1\]\.w == 0\.0\);/],
  ]) {
    const out = T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ vectorOpcode: 2, scalarOpcode: opcode, scalarWriteMask: 0, src: [temp(1), temp(1), temp(1)] }),
    ));
    assert.match(out.wgsl, pattern, `${name}:\n${out.wgsl}`);
    await assertValidWgsl(out.wgsl);
  }
});

test('scalar clamp wraps the stored value', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 0, scalarWriteMask: 0xF, scalarDest: 4, scalarClamp: 1, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /r\[4\] = vec4f\(clamp\(/);
  await assertValidWgsl(out.wgsl);
});

test('ps/p0 are omitted when no scalar instruction exists', () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, scalarOpcode: 50, scalarWriteMask: 0, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.doesNotMatch(out.wgsl, /var ps/);
  assert.doesNotMatch(out.wgsl, /var p0/);
});

// ---------------------------------------------------------------- control flow & predication
test('predicated instructions are wrapped in if (p0) / if (!p0)', async () => {
  const on = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, predicated: 1, predicateCondition: 1, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(on.wgsl, /if \(p0\) \{/);
  assert.equal(on.predicateUsed, true);
  await assertValidWgsl(on.wgsl);

  const off = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, predicated: 1, predicateCondition: 0, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(off.wgsl, /if \(!p0\) \{/);
  await assertValidWgsl(off.wgsl);
});

test('cexec/cexece wrap the block in if (bc(n)) / if (!bc(n))', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(condExecCf(1, 2, seq(SEQ_FETCH, SEQ_ALU), 5, 1), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, scalarOpcode: 50, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.equal(out.usesBoolConstants, true);
  assert.match(out.wgsl, /var<uniform> bools: array<u32, 8>;/);
  assert.match(out.wgsl, /fn bc\(n: u32\) -> bool/);
  assert.match(out.wgsl, /if \(bc\(5u\)\) \{/);
  await assertValidWgsl(out.wgsl);

  const inverted = T.translateVertexShader(shader(
    cfPair(condExecCf(1, 2, seq(SEQ_FETCH, SEQ_ALU), 5, 0, true), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, scalarOpcode: 50, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(inverted.wgsl, /if \(!bc\(5u\)\) \{/);
  await assertValidWgsl(inverted.wgsl);
});

test('cexec_pred wraps the block in if (p0)', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(predExecCf(1, 2, seq(SEQ_FETCH, SEQ_ALU), 1), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, scalarOpcode: 50, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /if \(p0\) \{/);
  assert.equal(out.predicateUsed, true);
  await assertValidWgsl(out.wgsl);
});

test('predicated instruction inside a cexec block nests correctly', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(condExecCf(1, 2, seq(SEQ_FETCH, SEQ_ALU), 3, 1), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, predicated: 1, predicateCondition: 1, src: [temp(1), temp(1), temp(1)] }),
  ));
  const bcIndex = out.wgsl.indexOf('if (bc(3u)) {');
  const p0Index = out.wgsl.indexOf('if (p0) {');
  assert.ok(bcIndex >= 0 && p0Index > bcIndex, 'expected p0 block inside bc block');
  await assertValidWgsl(out.wgsl);
});

// ---------------------------------------------------------------- modifiers, swizzles, clamps
test('absolute-value constant sources emit abs(fc[n])', () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 11 /* mad: all three operands used */, absConstants: 1, src: [temp(1), konst(0), konst(1)] }),
  ));
  assert.match(out.wgsl, /abs\(fc\[0\]\./);
  assert.match(out.wgsl, /abs\(fc\[1\]\./);
});

test('absolute-value temp sources emit abs(r[n])', () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, src: [tempAbs(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /abs\(r\[1\]\.x\)/);
});

test('negated sources emit (-(x))', () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, src: [{ ...temp(1), negate: 1 }, { ...konst(0), negate: 1 }, temp(1)] }),
  ));
  assert.match(out.wgsl, /\(-\(r\[1\]\.x\)\)/);
  assert.match(out.wgsl, /\(-\(fc\[0\]\.x\)\)/);
});

test('source swizzles are component-relative (lane j reads lane (swz_j + j) & 3)', () => {
  // Replicate W across all lanes: Xenos swizzles are per-lane OFFSETS added to the
  // lane index (mod 4), so fields are (3,2,1,0) → 0b00011011 (0x1B).
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, src: [{ ...temp(1), swizzle: 0x1B }, temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /\(r\[1\]\.w \+ r\[1\]\.w\)/); // lane 3: swizzled w + identity w
  assert.match(out.wgsl, /\(r\[1\]\.w \+ r\[1\]\.y\)/); // lane 1: swizzled w + r[1].y
});

// ---------------------------------------------------------------- traps
test('trap: loops are rejected loudly', () => {
  assert.throws(
    () => T.translateVertexShader(shader(
      cfPair({ dword0: 0, dword1: 7 << 12 }, NOP_CF),
    )),
    (e) => e.constructor.name === 'XenosTranslateError' && /loop_start/.test(e.message));
});

test('trap: cond_jmp is rejected loudly', () => {
  assert.throws(() => T.translateVertexShader(shader(cfPair({ dword0: 0, dword1: 11 << 12 }, NOP_CF))), /cond_jmp/);
});

test('trap: cond_call and return are rejected loudly', () => {
  assert.throws(() => T.translateVertexShader(shader(cfPair({ dword0: 0, dword1: 9 << 12 }, NOP_CF))), /cond_call/);
  assert.throws(() => T.translateVertexShader(shader(cfPair({ dword0: 0, dword1: 10 << 12 }, NOP_CF))), /return/);
});

test('trap: texture fetch in a vertex shader is rejected loudly', () => {
  assert.throws(
    () => T.translateVertexShader(shader(cfPair(execCf(1, 1, seq(SEQ_FETCH)), NOP_CF), tfetchWords({}))),
    /texture fetch/,
  );
});

test('trap: cube and maxa vector ops are rejected loudly', () => {
  for (const [opcode, name] of [[18, 'cube'], [29, 'maxa']]) {
    assert.throws(
      () => T.translateVertexShader(shader(
        cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
        vfetchWords({ dstReg: 1, format: 57 }),
        aluWords({ vectorDest: 3, vectorOpcode: opcode, src: [temp(1), temp(1), temp(1)] }),
      )),
      new RegExp(name),
    );
  }
});

test('trap: maxas/maxasf scalar ops (a0 write) are rejected loudly', () => {
  for (const opcode of [23, 24]) {
    assert.throws(
      () => T.translateVertexShader(shader(
        cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
        vfetchWords({ dstReg: 1, format: 57 }),
        aluWords({ scalarOpcode: opcode, src: [temp(1), temp(1), temp(1)] }),
      )),
      /a0 write/,
    );
  }
});

test('trap: still-unsupported scalar ops (muls_prev2, maxas/maxasf, mulsc/addsc/subsc) are rejected loudly', () => {
  for (const opcode of [4, 23, 24, 42, 43, 44, 45, 46, 47]) {
    assert.throws(
      () => T.translateVertexShader(shader(
        cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
        vfetchWords({ dstReg: 1, format: 57 }),
        aluWords({ scalarOpcode: opcode, src: [temp(1), temp(1), temp(1)] }),
      )),
      /not supported yet|invalid/,
    );
  }
});

test('trap: mulsc/addsc/subsc scalar ops are rejected loudly', () => {
  for (const opcode of [42, 43, 44, 45, 46, 47]) {
    assert.throws(
      () => T.translateVertexShader(shader(
        cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
        vfetchWords({ dstReg: 1, format: 57 }),
        aluWords({ scalarOpcode: opcode, src: [temp(1), temp(1), temp(1)] }),
      )),
      /mulsc|addsc|subsc/,
    );
  }
});

test('trap: muls_prev2 and pixel kill ops are rejected loudly', () => {
  assert.throws(() => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ scalarOpcode: 4, src: [temp(1), temp(1), temp(1)] }),
  )), /muls_prev2/);
  for (const opcode of [35, 36, 37, 38, 39]) {
    assert.throws(() => T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ scalarOpcode: opcode, src: [temp(1), temp(1), temp(1)] }),
    )), /pixel-shader-only/);
  }
  for (const opcode of [24, 25, 26, 27]) {
    assert.throws(() => T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ vectorOpcode: opcode, src: [temp(1), temp(1), temp(1)] }),
    )), /pixel-shader-only/);
  }
});

test('trap: unassigned vector opcodes 30/31 and scalar opcodes 41/51-63 are rejected loudly', () => {
  for (const opcode of [30, 31]) {
    assert.throws(() => T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ vectorOpcode: opcode, src: [temp(1), temp(1), temp(1)] }),
    )), /invalid/);
  }
  for (const opcode of [41, 51, 52, 63]) {
    assert.throws(() => T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ scalarOpcode: opcode, src: [temp(1), temp(1), temp(1)] }),
    )), /invalid/);
  }
});

test('trap: vector dest GPR 30/31 are rejected loudly', () => {
  for (const vectorDest of [30, 31]) {
    assert.throws(() => T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ vectorDest, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
    )), /vector dest GPR/);
  }
});

test('trap: scalar dest GPR >= 32 is rejected loudly', () => {
  assert.throws(() => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ scalarOpcode: 0, scalarDest: 40, scalarWriteMask: 1, src: [temp(1), temp(1), temp(1)] }),
  )), /scalar dest GPR/);
});

test('trap: register-relative dests and +aL operands are rejected loudly', () => {
  assert.throws(() => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDestRelative: 1, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  )), /\+aL register-relative vector destination/);
  assert.throws(() => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ scalarDestRelative: 1, scalarWriteMask: 1, src: [temp(1), temp(1), temp(1)] }),
  )), /\+aL register-relative scalar destination/);
  assert.throws(() => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 0, src: [tempRel(1), temp(1), temp(1)] }),
  )), /\+aL register-relative temporary/);
});

test('trap: aL-relative constant addressing is rejected loudly', () => {
  assert.throws(() => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 0, constRelAbs: 1, src: [temp(1), konst(0), temp(1)] }),
  )), /aL-relative constant addressing/);
});

test('trap: shader length not a multiple of 3 dwords is rejected loudly', () => {
  assert.throws(() => T.translateVertexShader(new Uint32Array(4)), /not a multiple of 3/);
});

test('trap: exec CF pointing past the blob traps loudly', () => {
  const ucode = shader(cfPair(execCf(7, 1, seq(SEQ_FETCH)), NOP_CF), [0, 0, 0]);
  assert.throws(() => T.translateVertexShader(ucode), /out of shader/);
});

test('trap: memory exports and kill-vertex exports are rejected loudly', () => {
  for (const vectorDest of [32, 33, 37]) {
    assert.throws(() => T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ isExport: 1, vectorDest, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
    )), /memory export/);
  }
  assert.throws(() => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ isExport: 1, vectorDest: 63, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  )), /point-size\/edge-flag\/kill-vertex/);
  // Unsupported export registers trap too (61 = PS depth, others unassigned).
  for (const vectorDest of [17, 31, 61]) {
    assert.throws(() => T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ isExport: 1, vectorDest, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
    )), /unsupported export register|memory export/);
  }
});

test('trap: vfetch attribute dest GPR >= 16 is rejected loudly (WebGPU attribute location limit)', () => {
  assert.throws(() => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 16, format: 57 }),
    aluWords({ vectorDest: 1, vectorOpcode: 0, src: [temp(16), temp(16), temp(16)] }),
  )), /vertex-attribute location/);
});

// ---------------------------------------------------------------- regression & end-to-end
test('trailing instruction words that alias an exec CF do not corrupt the CF bound', () => {
  // Instruction 2's words look like an EXEC-family CF pair with address 0.
  // With a live-shrinking cfPairBound they must not be re-read as control flow.
  const alu = aluWords({ vectorOpcode: 2 /* max */, src: [{ ...temp(1), swizzle: 0x10 }, temp(1), temp(1)] });
  alu[0] = 0; // vectorDest 0, mask 0 — w0 = 0 → a naive walker would see CF addr 0, count 0
  const ucode = shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    alu,
  );
  const out = T.translateVertexShader(ucode); // must not throw / silently translate nothing
  assert.deepEqual(out.inputLocations, [1]);
});

test('non-writing vector/scalar NOP instruction emits nothing but keeps neighbors', () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2 /* max */, vectorWriteMask: 0, scalarOpcode: 50 /* retain_prev */, scalarWriteMask: 0, src: [temp(0), temp(0), temp(0)] }),
  ));
  assert.doesNotMatch(out.wgsl, /alu\[2\]/);
  assert.doesNotMatch(out.wgsl, /var ps/);
  assert.doesNotMatch(out.wgsl, /var p0/);
  assert.deepEqual(out.inputLocations, [1]);
});

test('predicated vfetch still binds its attribute', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 2, format: 57, isPredicated: 1 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, src: [temp(2), temp(2), temp(2)] }),
  ));
  assert.deepEqual(out.inputLocations, [2]);
  await assertValidWgsl(out.wgsl);
});

test('a larger end-to-end shader passes Dawn validation', async () => {
  // fetch r1, mad r3 = r1*fc0 + fc1, predicated dp4 → export oPos, cexec block exports interpolator
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 3, seq(SEQ_FETCH, SEQ_ALU, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, scalarDest: 4, vectorOpcode: 11 /* mad */, scalarOpcode: 0 /* adds */, scalarWriteMask: 0xF, src: [temp(1), konst(0), konst(1)] }),
    aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 15 /* dp4 */, scalarOpcode: 50, src: [temp(3), konst(2), temp(1)] }),
  ));
  assert.equal(out.writesPosition, true);
  assert.equal(out.usesFloatConstants, true);
  assert.match(out.wgsl, /out\.pos = vec4f\(/);
  await assertValidWgsl(out.wgsl);

  // And a cexec-wrapped export variant.
  const withCexec = T.translateVertexShader(shader(
    cfPair(condExecCf(1, 3, seq(SEQ_FETCH, SEQ_ALU, SEQ_ALU), 7, 1), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 0, scalarOpcode: 50, src: [temp(1), temp(1), temp(1)] }),
    aluWords({ isExport: 1, vectorDest: 6, vectorOpcode: 0, scalarOpcode: 50, src: [temp(3), temp(1), temp(1)] }),
  ));
  assert.match(withCexec.wgsl, /if \(bc\(7u\)\) \{/);
  assert.match(withCexec.wgsl, /out\.v6 = /);
  await assertValidWgsl(withCexec.wgsl);
});

// ================================================================ pixel shaders
// Fixtures: oC0 = vectorDest 0, oDepth = 61, MRT targets 1–3 = vectorDest 1–3.

/** PS that exports oC0 = fc[0] + fc[1]. */
const PS_CONST = () => shader(
  cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
  aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [konst(0), konst(1), konst(1)] }),
);
/** PS that exports oC0 = r[7] + r[7] (static bound 8). */
const PS_READ_R7 = () => shader(
  cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
  aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(7), temp(7), temp(7)] }),
);
/** VS exporting interpolators 7 and 62(position). */
const VS_EXPORT_7 = () => shader(
  cfPair(execCf(1, 3, seq(SEQ_FETCH, SEQ_ALU, SEQ_ALU)), NOP_CF),
  vfetchWords({ dstReg: 1, format: 57 }),
  aluWords({ isExport: 1, vectorDest: 7, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
);

test('pixel shader: oC0 from constants translates and validates', async () => {
  const out = T.translatePixelShader(PS_CONST());
  assert.equal(out.type, 'pixel');
  assert.match(out.wgsl, /struct PsOut \{\n  @location\(0\) color0: vec4f\n\};/);
  assert.match(out.wgsl, /function|fn fs_main\(\) -> PsOut/);
  assert.doesNotMatch(out.wgsl, /struct PsIn\b/, 'no interpolators referenced → no PsIn');
  assert.equal(out.usesFloatConstants, true);
  assert.deepEqual(out.inputLocations, []);
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: interpolator lanes are linked to the VS exports, others read as zero', async () => {
  const program = T.translateShaderProgram(VS_EXPORT_7(), PS_READ_R7());
  assert.deepEqual(program.interpolatorLocations, [7], 'VS exports interpolator 7');
  assert.deepEqual(program.fragment.inputLocations, [7], 'the FS wires exactly lane 7');
  assert.match(program.fragment.wgsl, /@location\(7\) i7: vec4f/);
  assert.doesNotMatch(program.fragment.wgsl, /@location\(0\) i0/);
  assert.match(program.fragment.wgsl, /r\[7\] = in\.i7/);
  assert.match(program.fragment.wgsl, /var r: array<vec4f, 8>/, 'static bound 8 covers lanes 0–7');
  await assertValidWgsl(program.vertex.wgsl);
  await assertValidWgsl(program.fragment.wgsl);
});

test('pixel shader: lanes the VS does not export are not declared (link-safe by construction)', async () => {
  // PS reads r[0] (static bound 1), VS exports lane 7 → linked = ∅; r[0] stays zero-initialized.
  const PS_READ_R0 = shader(
    cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(0), temp(0), temp(0)] }),
  );
  const program = T.translateShaderProgram(VS_EXPORT_7(), PS_READ_R0);
  assert.deepEqual(program.fragment.inputLocations, []);
  assert.doesNotMatch(program.fragment.wgsl, /struct PsIn\b/);
  await assertValidWgsl(program.fragment.wgsl);
});

test('pixel shader: Dawn pipeline accepts the translated pair (interface rule enforced)', async () => {
  const program = T.translateShaderProgram(VS_EXPORT_7(), PS_READ_R7());
  const vsBuf = [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] }];
  assert.equal(await checkProgramLinks(program.vertex.wgsl, program.fragment.wgsl, vsBuf), null);
});

test('pixel shader: kill_eq (vector opcode 24) emits discard with an any-lane compare', async () => {
  const out = T.translatePixelShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_ALU, SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 3, vectorOpcode: 24, vectorWriteMask: 0xF, src: [temp(1), konst(0), temp(1)] }),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /if \(any\(vec4<bool>\(\(r\[1\]\.x == fc\[0\]\.x\), \(r\[1\]\.y == fc\[0\]\.y\), \(r\[1\]\.z == fc\[0\]\.z\), \(r\[1\]\.w == fc\[0\]\.w\)\)\)\) \{ discard; \}/);
  assert.match(out.wgsl, /r\[3\] = vec4f\(0\.0, 0\.0, 0\.0, 0\.0\);/, 'kill writes 0.0 to its dest with the mask');
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: kill_ne (opcode 27) uses unordered semantics (!=)', async () => {
  const out = T.translatePixelShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_ALU, SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 3, vectorOpcode: 27, vectorWriteMask: 0xF, src: [temp(1), temp(2), temp(1)] }),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /any\(vec4<bool>\(\(r\[1\]\.x != r\[2\]\.x\)/);
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: scalar kills (kills_eq/gt/ge/ne/one) compare the W lane of the scalar source', async () => {
  for (const [op, needle] of [[35, '== 0.0'], [36, '> 0.0'], [37, '>= 0.0'], [38, '!= 0.0'], [39, '== 1.0']]) {
    const out = T.translatePixelShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_ALU, SEQ_ALU)), NOP_CF),
      aluWords({ scalarOpcode: op, vectorOpcode: 2, vectorWriteMask: 0, src: [temp(2), temp(2), temp(1)] }),
      aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
    ));
    assert.match(out.wgsl, new RegExp(`if \\(\\(r\\[1\\]\\.w ${needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\)\\) \\{ discard; \\}`), `opcode ${op}`);
    assert.match(out.wgsl, /ps = 0\.0;/, 'kill writes 0.0 to ps');
  }
});

test('pixel shader: a predicated kill only discards under p0', async () => {
  const out = T.translatePixelShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_ALU, SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 3, vectorOpcode: 24, vectorWriteMask: 0xF, predicated: 1, predicateCondition: 1, src: [temp(1), temp(2), temp(1)] }),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  ));
  const killIdx = out.wgsl.indexOf('discard');
  const predIdx = out.wgsl.lastIndexOf('if (p0) {', killIdx);
  assert.ok(predIdx !== -1 && predIdx < killIdx, 'the discard is nested inside the predicate');
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: kills in a VERTEX shader still trap loudly', () => {
  assert.throws(() => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorDest: 3, vectorOpcode: 24, src: [temp(1), temp(2), temp(1)] }),
  )), /pixel-shader-only/);
  assert.throws(() => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ scalarOpcode: 35, vectorWriteMask: 0, src: [temp(1), temp(1), temp(1)] }),
  )), /pixel-shader-only/);
});

test('pixel shader: MRT color exports (oC1..oC3) trap with a named reason', () => {
  for (const dest of [1, 2, 3]) {
    assert.throws(() => T.translatePixelShader(shader(
      cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
      aluWords({ isExport: 1, vectorDest: dest, vectorOpcode: 0, src: [konst(0), konst(0), konst(0)] }),
    )), /MRT/);
  }
});

test('pixel shader: oDepth export emits a clamped @builtin(frag_depth) output', async () => {
  const out = T.translatePixelShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_ALU, SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 1, vectorOpcode: 0, src: [konst(0), konst(0), konst(0)] }),           // r1 = fc0+fc0
    aluWords({ isExport: 1, vectorDest: 61, vectorWriteMask: 1, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }), // oDepth = r1.x
  ));
  assert.equal(out.exportsDepth, true);
  assert.match(out.wgsl, /@builtin\(frag_depth\) depth: f32,/);
  assert.match(out.wgsl, /out\.depth = clamp\(\(r\[1\]\.x \+ r\[1\]\.x\), 0\.0, 1\.0\);/);
  assert.match(out.wgsl, /out\.depth = 1\.0;/, 'unwritten paths still define frag_depth');
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: oDepth in a VS+PS pair links through Dawn with depth-stencil state', async () => {
  const PS_DEPTH = shader(
    cfPair(execCf(1, 2, seq(SEQ_ALU, SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 1, vectorOpcode: 0, src: [konst(0), konst(0), konst(0)] }),
    aluWords({ isExport: 1, vectorDest: 61, vectorWriteMask: 1, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  );
  const program = T.translateShaderProgram(VS_EXPORT_7(), PS_DEPTH);
  assert.equal(program.exportsDepth, true);
  assert.match(program.fragment.wgsl, /@builtin\(frag_depth\)/);
  const vsBuf = [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] }];
  assert.equal(await checkProgramLinks(program.vertex.wgsl, program.fragment.wgsl, vsBuf), null);
});

test('pixel shader: a program with no oDepth export declares no frag_depth (normal depth path)', () => {
  const program = T.translateShaderProgram(VS_EXPORT_7(), PS_CONST());
  assert.equal(program.exportsDepth, false);
  assert.doesNotMatch(program.fragment.wgsl, /frag_depth/);
});

test('pixel shader: texture fetch traps naming the limitation (tfetch is step 8)', () => {
  assert.throws(() => T.translatePixelShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    tfetchWords({ dstReg: 5 }),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(5), temp(5), temp(5)] }),
  )), /texture fetch[\s\S]*step 8/);
});

test('pixel shader: a vertex fetch inside a pixel shader traps loudly', () => {
  assert.throws(() => T.translatePixelShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  )), /vertex fetch inside a pixel shader/);
});

test('pixel shader: unsupported opcodes trap exactly as in the VS (cube, transcendentals)', () => {
  assert.throws(() => T.translatePixelShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_ALU, SEQ_ALU)), NOP_CF),
    aluWords({ vectorOpcode: 18, src: [temp(1), temp(2), temp(1)] }),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  )), /cube/);
  assert.throws(() => T.translatePixelShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_ALU, SEQ_ALU)), NOP_CF),
    aluWords({ scalarOpcode: 23 /* maxas */, vectorWriteMask: 0, src: [temp(1), temp(1), temp(1)] }),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  )), /maxas/);
});

test('translateShaderProgram: no pixel shader → fragment is null, bindings match the plain VS', () => {
  const bare = T.translateShaderProgram(VS_EXPORT_7(), null);
  assert.equal(bare.fragment, null);
  const plain = T.translateVertexShader(VS_EXPORT_7());
  assert.equal(bare.vertex.wgsl, plain.wgsl, 'VS-only program output equals the standalone VS translation');
});

test('translateShaderProgram: constants merged across stages share the fc/bool bindings', async () => {
  // VS uses bools only; PS uses fc only.
  const VS_BOOL = shader(
    cfPair(condExecCf(1, 2, seq(SEQ_FETCH, SEQ_ALU), 3, 1), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ isExport: 1, vectorDest: 7, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  );
  const PS_FC = shader(
    cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [konst(0), konst(0), konst(0)] }),
  );
  const program = T.translateShaderProgram(VS_BOOL, PS_FC);
  assert.equal(program.usesFloatConstants, true);
  assert.equal(program.usesBoolConstants, true);
  // VS: no fc of its own → only bools, declared at binding 1 (fc is merged-binding 0)
  assert.match(program.vertex.wgsl, /@group\(0\) @binding\(1\) var<uniform> bools/);
  assert.doesNotMatch(program.vertex.wgsl, /var<uniform> fc/);
  // PS: fc only → binding 0
  assert.match(program.fragment.wgsl, /@group\(0\) @binding\(0\) var<uniform> fc/);
  assert.doesNotMatch(program.fragment.wgsl, /var<uniform> bools/);
  await assertValidWgsl(program.vertex.wgsl);
  await assertValidWgsl(program.fragment.wgsl);
  // Both declare the same bools/fc layout? No — bools only in VS; the merged layout
  // is fc@0 (PS) + bools@1 (VS): one bind group covering both is what the backend builds.

  // VS uses fc+bools; PS uses fc only → PS must NOT declare bools at binding 0 or 1.
  const VS_FC_BOOL = shader(
    cfPair(condExecCf(1, 2, seq(SEQ_FETCH, SEQ_ALU), 3, 1), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ isExport: 1, vectorDest: 7, vectorOpcode: 0, src: [temp(1), konst(0), temp(1)] }),
  );
  const program2 = T.translateShaderProgram(VS_FC_BOOL, PS_FC);
  assert.match(program2.vertex.wgsl, /@group\(0\) @binding\(0\) var<uniform> fc/);
  assert.match(program2.vertex.wgsl, /@group\(0\) @binding\(1\) var<uniform> bools/);
  assert.match(program2.fragment.wgsl, /@group\(0\) @binding\(0\) var<uniform> fc/);
  assert.doesNotMatch(program2.fragment.wgsl, /var<uniform> bools/, 'PS does not use bools — must not declare them');
  await assertValidWgsl(program2.vertex.wgsl);
  await assertValidWgsl(program2.fragment.wgsl);
});

test('pixel shader: an empty (0-dword) blob translates to a degenerate-but-valid black shader', async () => {
  const out = T.translatePixelShader(new Uint32Array(0));
  assert.match(out.wgsl, /out\.color0 = vec4f\(0\.0\);/);
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: malformed blob (exec pointing at slot 0) traps instead of shading nothing', () => {
  assert.throws(() => T.translatePixelShader(shader(
    cfPair(execCf(0, 1, seq(SEQ_ALU)), NOP_CF),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [konst(0), konst(0), konst(0)] }),
  )), /no executable control flow/);
});

test('checkProgramLinks reports the exact diagnostic when the FS declares an unlinked input', async () => {
  // Negative control for the link-safe-by-construction design: the helper must be
  // able to SEE an interface break, not just rubber-stamp pairs.
  const program = T.translateShaderProgram(VS_EXPORT_7(), PS_READ_R7());
  const badFs = program.fragment.wgsl.replace('struct PsIn {', 'struct PsIn {\n  @location(9) i9: vec4f,');
  assert.notEqual(badFs, program.fragment.wgsl);
  const err = await checkProgramLinks(program.vertex.wgsl, badFs, [{
    arrayStride: 12, stepMode: 'vertex',
    attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }],
  }]);
  assert.match(err, /fragment input at location 9|corresponding vertex output/i);
});

// ================================================================ alpha test
// RB_COLORCONTROL.alpha_test_enable + alpha_func (CompareFunction, same 3-bit
// numbering as zfunc). The reference is NOT baked into the shader — it arrives
// as a small uniform (IEEE-754 bits at bindings.alphaTest) the backend binds.

test('pixel shader: alpha test emits the discard + uniform, and Dawn accepts it', async () => {
  const out = T.translatePixelShader(PS_CONST(), { alphaTest: { compare: 3 } }); // kLessEqual, PS also reads fc
  assert.equal(out.type, 'pixel');
  assert.match(out.wgsl, /@group\(0\) @binding\(0\) var<uniform> fc/);
  assert.match(out.wgsl, /@group\(0\) @binding\(1\) var<uniform> alphaTest: vec4f;/, 'fc occupies 0 — the alpha test takes the next free binding');
  assert.match(out.wgsl, /fn alphaTestPass\(x: f32\) -> bool \{ return x <= alphaTest\.x; \}/);
  assert.match(out.wgsl, /if \(!alphaTestPass\(out\.color0\.w\)\) \{ discard; \}\n  return out;/);
  assert.deepEqual(out.alphaTest, { compare: 3, binding: 1 });
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: alpha test reuses the merged layout — fc + bools + alphaTest get distinct bindings', async () => {
  // VS uses fc+bools; PS uses fc; alpha test for the pair → alphaTest takes binding 2.
  const VS_FC_BOOL = shader(
    cfPair(condExecCf(1, 2, seq(SEQ_FETCH, SEQ_ALU), 3, 1), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ isExport: 1, vectorDest: 7, vectorOpcode: 0, src: [temp(1), konst(0), temp(1)] }),
  );
  const program = T.translateShaderProgram(VS_FC_BOOL, PS_CONST(), { alphaTest: { compare: 1 } }); // kLess
  assert.match(program.fragment.wgsl, /@group\(0\) @binding\(0\) var<uniform> fc/);
  assert.match(program.fragment.wgsl, /@group\(0\) @binding\(2\) var<uniform> alphaTest/);
  assert.doesNotMatch(program.fragment.wgsl, /var<uniform> bools/);
  assert.equal(program.alphaTest.compare, 1);
  assert.equal(program.alphaTest.binding, 2);
  await assertValidWgsl(program.fragment.wgsl);
  const ok = await checkProgramLinks(program.vertex.wgsl, program.fragment.wgsl,
    [{ arrayStride: 12, stepMode: 'vertex', attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] }]);
  assert.equal(ok, null);
});

test('pixel shader: alpha test without fc/bools binds alphaTest at 0; bools present → at 1', async () => {
  const p1 = T.translateShaderProgram(VS_EXPORT_7(), PS_READ_R7(), { alphaTest: { compare: 6 } }); // kGreaterEqual
  assert.match(p1.fragment.wgsl, /@group\(0\) @binding\(0\) var<uniform> alphaTest/);
  assert.equal(p1.alphaTest.binding, 0);
  await assertValidWgsl(p1.fragment.wgsl);

  const PS_BOOL = shader( // PS uses bools only (fresh binding 0) + alphaTest (binding 1)
    cfPair(condExecCf(1, 1, seq(SEQ_ALU), 5, 1), NOP_CF),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(1), temp(1), temp(1)] }),
  );
  const p2 = T.translateShaderProgram(VS_EXPORT_7(), PS_BOOL, { alphaTest: { compare: 5 } }); // kNotEqual
  assert.match(p2.fragment.wgsl, /@group\(0\) @binding\(0\) var<uniform> bools/);
  assert.match(p2.fragment.wgsl, /@group\(0\) @binding\(1\) var<uniform> alphaTest/);
  assert.equal(p2.alphaTest.binding, 1);
  await assertValidWgsl(p2.fragment.wgsl);
});

test('pixel shader: alpha test ignored when the PS does not statically export oC0 (xenia applies it only then)', () => {
  const PS_DEAD = shader(
    cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 3, vectorOpcode: 2, vectorWriteMask: 0xF, src: [temp(1), temp(1), temp(1)] }), // dead write, no exports
  );
  const out = T.translatePixelShader(PS_DEAD, { alphaTest: { compare: 2 } });
  assert.equal(out.alphaTest, null);
  assert.doesNotMatch(out.wgsl, /alphaTest|discard/);
});

test('pixel shader: kAlways (7) is treated as no test at the program level', () => {
  const program = T.translateShaderProgram(VS_EXPORT_7(), PS_READ_R7(), { alphaTest: { compare: 7 } });
  assert.equal(program.alphaTest, null);
  assert.doesNotMatch(program.fragment.wgsl, /alphaTest|discard/);
});

test('pixel shader: compare 8 is rejected loudly', () => {
  assert.throws(() => T.translateShaderProgram(VS_EXPORT_7(), PS_CONST(), { alphaTest: { compare: 8 } }), /CompareFunction/);
});

test('pixel shader: every Xenos CompareFunction maps to the operator with matching NaN semantics', async () => {
  const expected = [
    [0, 'return false;'],
    [1, 'return x < alphaTest.x;'],
    [2, 'return x == alphaTest.x;'],
    [3, 'return x <= alphaTest.x;'],
    [4, 'return x > alphaTest.x;'],
    [5, 'return x != alphaTest.x;'],
    [6, 'return x >= alphaTest.x;'],
  ];
  for (const [compare, expr] of expected) {
    const out = T.translatePixelShader(PS_CONST(), { alphaTest: { compare } });
    assert.ok(out.wgsl.includes(`fn alphaTestPass(x: f32) -> bool { ${expr} }`), `compare ${compare}`);
    await assertValidWgsl(out.wgsl);
  }
});

// ================================================================ gamma (PWL) encode
// Xbox 360 RT format k_8_8_8_8_GAMMA: xenia gamma-encodes oC0's RGB lanes in
// the pixel shader (PreSaturatedLinearToPWLGamma) — saturate, 3-piece
// piecewise, trunc(x*scale)/255 + offset. Alpha untouched. Pure ALU.

test('pixel shader: convertColor0ToGamma emits the PWL helper + rewritten export, Dawn-valid', async () => {
  const out = T.translatePixelShader(PS_CONST(), { convertColor0ToGamma: true });
  assert.match(out.wgsl, /fn xenosGamma\(x: f32\) -> f32/);
  assert.match(out.wgsl, /if \(x >= \(128\.0 \/ 1023\.0\)\)/);
  assert.match(out.wgsl, /trunc\(x \* scale\) \* \(1\.0 \/ 255\.0\) \+ offset/);
  assert.match(out.wgsl, /out\.color0 = vec4f\(xenosGamma\(clamp\(out\.color0\.x, 0\.0, 1\.0\)\), xenosGamma\(clamp\(out\.color0\.y, 0\.0, 1\.0\)\), xenosGamma\(clamp\(out\.color0\.z, 0\.0, 1\.0\)\), out\.color0\.w\);\n  return out;/);
  assert.equal(out.convertColor0ToGamma, true);
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: gamma encode leaves the alpha lane alone and still exports oC0', async () => {
  const out = T.translatePixelShader(PS_CONST(), { convertColor0ToGamma: true });
  // alpha (w lane) is passed through, not encoded
  assert.match(out.wgsl, /, out\.color0\.w\);/);
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: no gamma encode without the option; no encode without an oC0 export', async () => {
  const plain = T.translatePixelShader(PS_CONST());
  assert.equal(plain.convertColor0ToGamma, false);
  assert.doesNotMatch(plain.wgsl, /xenosGamma/);
  // dead-write PS: never exports oC0 → the option is ignored (nothing to encode)
  const DEAD = shader(
    cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 3, vectorOpcode: 2, vectorWriteMask: 0xF, src: [temp(1), temp(1), temp(1)] }),
  );
  const dead = T.translatePixelShader(DEAD, { convertColor0ToGamma: true });
  assert.equal(dead.convertColor0ToGamma, false);
  assert.doesNotMatch(dead.wgsl, /xenosGamma/);
});

test('pixel shader: gamma encode composes with the alpha test (discard before encode)', async () => {
  const out = T.translatePixelShader(PS_CONST(), { convertColor0ToGamma: true, alphaTest: { compare: 1 } });
  const discardIdx = out.wgsl.indexOf('discard');
  const gammaIdx = out.wgsl.indexOf('out.color0 = vec4f(xenosGamma');
  assert.ok(discardIdx >= 0 && gammaIdx > discardIdx, 'the alpha-test discard is emitted before the gamma encode');
  await assertValidWgsl(out.wgsl);
});

test('translateShaderProgram: convertColor0ToGamma is part of the fragment output, VS untouched', async () => {
  const program = T.translateShaderProgram(VS_EXPORT_7(), PS_READ_R7(), { convertColor0ToGamma: true });
  assert.equal(program.convertColor0ToGamma, true);
  assert.equal(program.fragment.convertColor0ToGamma, true);
  assert.doesNotMatch(program.vertex.wgsl, /xenosGamma/);
  await assertValidWgsl(program.vertex.wgsl);
  await assertValidWgsl(program.fragment.wgsl);
});

// The 3 PWL segments, checked against xenia's formula at representative points:
//   x < 64/1023            -> trunc(x * 1023)/255
//   64/1023..128/1023      -> trunc(x * 1023/2)/255 + 32/255
//   128/1023..512/1023     -> trunc(x * 1023/4)/255 + 64/255
//   >= 512/1023            -> trunc(x * 1023/8)/255 + 128/255
test('pixel shader: PWL segment math (validated as WGSL, values checked by hand)', () => {
  // reference encode, mirrored from xenia's PreSaturatedLinearToPWLGamma
  const pwl = (x) => {
    let scale, offset;
    if (x >= 128 / 1023) {
      if (x >= 512 / 1023) { scale = 1023 / 8; offset = 128 / 255; } else { scale = 1023 / 4; offset = 64 / 255; }
    } else {
      if (x >= 64 / 1023) { scale = 1023 / 2; offset = 32 / 255; } else { scale = 1023; offset = 0; }
    }
    return (Math.trunc(x * scale) * (1 / 255)) + offset;
  };
  // boundary sanity: 0 → 0, 1 → 1 (255 untouched lanes), mid-segment exact truncations
  assert.equal(pwl(0), 0);
  assert.equal(pwl(1), 1);
  assert.equal(pwl(64 / 1023), Math.trunc((64 / 1023) * (1023 / 2)) / 255 + 32 / 255);
  assert.equal(pwl(512 / 1023), Math.trunc((512 / 1023) * (1023 / 8)) / 255 + 128 / 255);
  assert.equal(pwl(0.5), Math.trunc(0.5 * (1023 / 4)) / 255 + 64 / 255);
});

// ================================================================ color_exp_bias (RB_COLOR_INFO bits 20-25, signed 6-bit)
test('pixel shader: colorExpBias emits a full-register 2^bias multiply, after the discard, Dawn-valid', async () => {
  const out = T.translatePixelShader(PS_CONST(), { colorExpBias: 3 });
  assert.match(out.wgsl, /out\.color0 = out\.color0 \* vec4f\(8\.0000000000e\+0\);/);
  assert.equal(out.colorExpBias, 3);
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: negative exponent produces 1/2^|bias|', async () => {
  const out = T.translatePixelShader(PS_CONST(), { colorExpBias: -2 });
  assert.match(out.wgsl, /out\.color0 = out\.color0 \* vec4f\(2\.5000000000e\-1\);/);
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: bias 0 (default) emits nothing; |bias| is clamped to [-32, 31]', async () => {
  const plain = T.translatePixelShader(PS_CONST());
  assert.match(plain.wgsl, /vec4f\(/); // sanity that the module built
  assert.doesNotMatch(plain.wgsl, /out\.color0 = out\.color0 \* vec4f\(/);
  assert.equal(plain.colorExpBias, 0);
  const clamped = T.translatePixelShader(PS_CONST(), { colorExpBias: 77 });
  assert.equal(clamped.colorExpBias, 31);
  await assertValidWgsl(clamped.wgsl);
});

test('pixel shader: bias runs AFTER the alpha-test discard (xenia order) and BEFORE gamma', async () => {
  const out = T.translatePixelShader(PS_CONST(), { colorExpBias: 2, alphaTest: { compare: 2 }, convertColor0ToGamma: true });
  const discardIdx = out.wgsl.indexOf('discard');
  const biasIdx = out.wgsl.indexOf('out.color0 = out.color0 * vec4f(');
  const gammaIdx = out.wgsl.indexOf('vec4f(xenosGamma');
  assert.ok(discardIdx >= 0 && biasIdx > discardIdx, 'bias multiply after discard');
  assert.ok(gammaIdx > biasIdx, 'gamma encode after bias multiply');
  // alpha is scaled too (xenia OpMul covers all lanes — the alpha test already used the unbiased alpha)
  await assertValidWgsl(out.wgsl);
});

test('pixel shader: PS without an oC0 export ignores colorExpBias entirely', async () => {
  const DEAD = shader(
    cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
    aluWords({ vectorDest: 3, vectorOpcode: 2, vectorWriteMask: 0xF, src: [temp(1), temp(1), temp(1)] }),
  );
  const out = T.translatePixelShader(DEAD, { colorExpBias: 4 });
  assert.equal(out.colorExpBias, 0);
  assert.doesNotMatch(out.wgsl, /vec4f\(1\.6000000000e\+1\)/);
});

test('translateShaderProgram: colorExpBias surfaces in the program result', async () => {
  const program = T.translateShaderProgram(VS_EXPORT_7(), PS_READ_R7(), { colorExpBias: -5 });
  assert.equal(program.colorExpBias, -5);
  assert.equal(program.fragment.colorExpBias, -5);
  await assertValidWgsl(program.vertex.wgsl);
  await assertValidWgsl(program.fragment.wgsl);
});

// ================================================================ transcendentals (xenia SPIR-V semantics)
test('clamped transcendentals emit the xenia helpers with the +/-INF clamps and Dawn-validate', async () => {
  const cases = [
    [15, /fn xenosLogc\(a: f32\) -> f32/],
    [17, /fn xenosRcpc\(a: f32\) -> f32/],
    [18, /fn xenosRcpf\(a: f32\) -> f32/],
    [20, /fn xenosRsqc\(a: f32\) -> f32/],
    [21, /fn xenosRsqf\(a: f32\) -> f32/],
  ];
  for (const [opcode, pattern] of cases) {
    const out = T.translateVertexShader(shader(
      cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
      vfetchWords({ dstReg: 1, format: 57 }),
      aluWords({ vectorOpcode: 2, scalarOpcode: opcode, scalarWriteMask: 0xF, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
    ));
    assert.match(out.wgsl, pattern, `helper missing for opcode ${opcode}`);
    await assertValidWgsl(out.wgsl);
  }
  // no helper emitted when unused
  const plain = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 0, scalarWriteMask: 0xF, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.doesNotMatch(plain.wgsl, /fn xenosLogc/);
});

test('transcendental helpers implement the documented INF/FLT_MAX clamps (source-level check)', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 15, scalarWriteMask: 0xF, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
  ));
  const w = out.wgsl;
  assert.match(w, /let t = log2\(a\);/);
  assert.match(w, /let inf = abs\(a\) \/ 0\.0;/);
  assert.match(w, /if \(t == -inf\) \{ return -3\.402823466e\+38; \}/);
  const build = (op) => T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: op, scalarWriteMask: 0xF, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
  )).wgsl;
  assert.match(build(17), /let t = 1\.0 \/ a;/);
  assert.match(build(17), /if \(t == -inf\) \{ return -3\.402823466e\+38; \}/);
  assert.match(build(17), /if \(t == inf\) \{ return 3\.402823466e\+38; \}/);
  assert.match(build(18), /if \(t == inf\) \{ return 0\.0; \}/);
  assert.match(build(18), /if \(t == -inf\) \{ return -0\.0; \}/);
  assert.match(build(20), /let t = inverseSqrt\(a\);/);
  assert.match(build(21), /if \(t == inf\) \{ return 0\.0; \}/);
  assert.match(build(21), /if \(t == -inf\) \{ return -0\.0; \}/);
});

test('scalar transcendental with clamp writes clamped via the standard scalar-clamp path', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 14, scalarClamp: 1, scalarWriteMask: 0xF, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
  ));
  assert.match(out.wgsl, /clamp\(exp2\(r\[1\]\.w\), 0\.0, 1\.0\)/);
  await assertValidWgsl(out.wgsl);
});

test('transcendentals compose in exp/log back-to-back chains and pass Dawn', async () => {
  // r4.w = log2(pow2(r1.w)) via rcp/rsq/sin/cos pairs — two transcendentals in one program.
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 3, seq(SEQ_FETCH, SEQ_ALU, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 14, scalarWriteMask: 0xF, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
    aluWords({ vectorOpcode: 2, scalarOpcode: 16, scalarWriteMask: 0xF, scalarDest: 5, src: [temp(4), temp(4), temp(4)] }),
  ));
  assert.match(out.wgsl, /r\[4\] = vec4f\(exp2\(r\[1\]\.w\)\);/);
  assert.match(out.wgsl, /r\[5\] = vec4f\(log2\(r\[4\]\.w\)\);/);
  await assertValidWgsl(out.wgsl);
});

test('transcendental in a pixel shader (sin) runs in the fragment stage', async () => {
  const out = T.translatePixelShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_ALU, SEQ_ALU)), NOP_CF),
    aluWords({ vectorOpcode: 2, scalarOpcode: 48, scalarWriteMask: 0xF, scalarDest: 4, src: [temp(1), temp(1), temp(1)] }),
    aluWords({ isExport: 1, vectorDest: 0, vectorOpcode: 0, src: [temp(4), temp(4), temp(4)] }),
  ));
  assert.match(out.wgsl, /sin\(r\[1\]\.w\)/);
  await assertValidWgsl(out.wgsl);
});

// ------------------------------------------------------------ guest viewport emission
test('translated vertex shader emits the guest-viewport uniform (VS-only binding, applied after ucode)', async () => {
  const out = T.translateVertexShader(shader(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, format: 57 }),
    aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 15 /* dp4 */, scalarOpcode: 50, src: [temp(1), temp(1), temp(1)] }),
  ));
  // no fc/bools in this shader → viewport at binding 0
  assert.match(out.wgsl, /@group\(0\) @binding\(0\) var<uniform> viewport: array<vec4f, 2>;/);
  assert.equal(out.viewportBinding, 0);
  assert.match(out.wgsl, /out\.pos = vec4f\(out\.pos\.x \* viewport\[0\]\.x \+ viewport\[1\]\.x \* out\.pos\.w/);
  assert.ok(out.wgsl.lastIndexOf('viewport[0].x') < out.wgsl.indexOf('return out'));
  await assertValidWgsl(out.wgsl);
});

test('viewport binding skips the fc/bools slots (and alphaTest on the shared program path)', () => {
  const program = T.translateShaderProgram(
    shader(cfPair(execCf(1, 1, seq(SEQ_ALU)), NOP_CF),
      aluWords({ isExport: 1, vectorDest: 62, vectorOpcode: 0, src: [temp(1), konst(0), temp(1)] })),
    null,
    { alphaTest: undefined },
  );
  // fc used (one konst read) → fc @0, viewport @1
  assert.equal(program.viewportBinding, 1);
  assert.match(program.vertex.wgsl, /@group\(0\) @binding\(1\) var<uniform> viewport/);
  assert.match(program.vertex.wgsl, /@group\(0\) @binding\(0\) var<uniform> fc/);
});
