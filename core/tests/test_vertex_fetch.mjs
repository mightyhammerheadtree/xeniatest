/**
 * Vertex fetch tests — fetch constants (xe_gpu_vertex_fetch_t), the ucode
 * stream walker, format mapping and GPU vertex buffer descriptor building.
 * All microcode is hand-assembled bit-by-bit from xenia's ucode.h layout.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../../public/js/gpu/vertex_fetch.js';

const VF = globalThis.XenonGpuVertexFetch;

// ---------------------------------------------------------------- ucode helpers
function execCf(address, count, sequence, opcode = 2 /* kExecEnd */) {
  return { dword0: (address & 0xFFF) | ((count & 0x7) << 12) | ((sequence & 0xFFF) << 16), dword1: (opcode & 0xF) << 12 };
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
  format = 57, stride = 1, offset = 0, expAdjust = 0, isSigned = 0,
  numFormatAll = 0, isMini = 0, isPredicated = 0, srcRegAm = 0, dstRegAm = 0,
} = {}) {
  const w0 = ((srcReg & 0x3F) << 5) | ((srcRegAm & 1) << 11) | ((dstReg & 0x3F) << 12) |
    ((dstRegAm & 1) << 18) | (1 << 19 /* must_be_one */) | ((constIndex & 0x1F) << 20) |
    ((constIndexSel & 0x3) << 25);
  const w1 = (dstSwiz & 0xFFF) | ((isSigned & 1) << 12) | ((numFormatAll & 1) << 13) |
    (format << 16) | ((expAdjust & 0x3F) << 24) | ((isMini & 1) << 30) | ((isPredicated & 1) << 31);
  const w2 = (stride & 0xFF) | ((offset & 0x7FFFFF) << 8);
  return [w0 >>> 0, w1 >>> 0, w2 >>> 0];
}
const ALU_ZERO = [0, 0, 0];
const exec = (...pairs) => new Uint32Array(pairs.flat());
const SEQ_ALU = 0b00, SEQ_FETCH = 0b01;
const seq = (...bits) => bits.reduce((s, b, i) => s | (b << (2 * i)), 0);

// ---------------------------------------------------------------- fetch constants
test('decodeVertexFetchConstants decodes type, address, endian and size', () => {
  const regs = new Uint32Array(192);
  regs[0 * 2] = 0x00100003; regs[0 * 2 + 1] = (0x0100 << 2) | 0;   // vertex, base 0x00100000, endian none, size 0x100 words
  regs[1 * 2] = 0x00200003; regs[1 * 2 + 1] = (0x0200 << 2) | 2;   // vertex, base 0x00200000, endian 8in32, size 0x200 words
  regs[5 * 2] = 0x00000001; regs[5 * 2 + 1] = 0;                   // invalidVertex
  regs[7 * 2] = 0x00400002; regs[7 * 2 + 1] = (0x0010 << 2) | 1;   // texture, base 0x00400000, endian 8in16, size 0x10 words
  const constants = VF.decodeVertexFetchConstants((i) => regs[i]);
  assert.equal(constants.length, 96);
  assert.deepEqual(constants[0], { index: 0, type: 3, typeName: 'vertex', baseAddress: 0x00100000, endian: 0, endianName: 'none', sizeWords: 0x100 });
  assert.deepEqual(constants[1], { index: 1, type: 3, typeName: 'vertex', baseAddress: 0x00200000, endian: 2, endianName: '8in32', sizeWords: 0x200 });
  assert.equal(constants[5].typeName, 'invalidVertex');
  assert.equal(constants[5].baseAddress, 0);
  assert.equal(constants[7].typeName, 'texture');
  assert.equal(constants[7].endianName, '8in16');
});

test('the fetch constant address counts dwords — the byte address is address << 2 (xenia)', () => {
  // xenia: xe_gpu_vertex_fetch_t.address is "address in dwords" and the guest
  // address is "vfetch_constant.address << 2" (xenia
  // d3d12_command_processor.cc). 0x00040001 dwords = 0x00100004 bytes.
  const regs = new Uint32Array(192);
  regs[0] = ((0x00040001 << 2) | 3) >>> 0; // type=vertex, address 0x00040001 dwords
  const constants = VF.decodeVertexFetchConstants((i) => regs[i]);
  assert.equal(constants[0].baseAddress, 0x00100004);
});

test('decodeVertexFetchConstants requires readDword', () => {
  assert.throws(() => VF.decodeVertexFetchConstants(null), TypeError);
});

// ---------------------------------------------------------------- ucode walking
test('analyzeVertexShader binds a vfetch to its fetch constant with stride and offset', () => {
  const ucode = exec(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_ALU)), NOP_CF),
    vfetchWords({ dstReg: 1, constIndex: 2, constIndexSel: 1, format: 57, stride: 5, offset: 0 }),
    ALU_ZERO,
  );
  const analysis = VF.analyzeVertexShader(ucode);
  assert.equal(analysis.controlFlowPairBound, 1);
  assert.equal(analysis.vertexFetchInstructionCount, 1);
  assert.equal(analysis.bindings.length, 1);
  const binding = analysis.bindings[0];
  assert.equal(binding.fetchConstant, 7); // 2 * 3 + 1
  assert.equal(binding.strideWords, 5);
  assert.equal(binding.attributes.length, 1);
  const attr = binding.attributes[0];
  assert.equal(attr.destRegister, 1);
  assert.equal(attr.byteOffset, 0);
  assert.equal(attr.format, VF.VERTEX_FORMAT.k_32_32_32_FLOAT);
  assert.equal(attr.writtenMask, 0b1111);
});

test('attributes whose dest lanes are all kKeep are not emitted', () => {
  const ucode = exec(
    cfPair(execCf(1, 1, seq(SEQ_FETCH)), NOP_CF),
    vfetchWords({ dstReg: 3, dstSwiz: 0xFFF /* all kKeep */, format: 36, stride: 1 }),
  );
  const analysis = VF.analyzeVertexShader(ucode);
  assert.equal(analysis.vertexFetchInstructionCount, 1);
  assert.equal(analysis.bindings.length, 0, 'no bindings when nothing is written');
});

test('vfetch_mini inherits its stride from the preceding vfetch_full', () => {
  const full = vfetchWords({ dstReg: 1, constIndex: 0, constIndexSel: 0, format: 36, stride: 3, offset: 0 });
  const mini = vfetchWords({ dstReg: 2, constIndex: 0, constIndexSel: 0, format: 36, stride: 0, offset: 1, isMini: 1 });
  const ucode = exec(
    cfPair(execCf(1, 2, seq(SEQ_FETCH, SEQ_FETCH)), NOP_CF),
    full,
    mini,
  );
  const analysis = VF.analyzeVertexShader(ucode);
  assert.equal(analysis.bindings.length, 1);
  const binding = analysis.bindings[0];
  assert.equal(binding.strideWords, 3, 'binding stride comes from the full fetch');
  assert.equal(binding.attributes.length, 2);
  assert.equal(binding.attributes[1].isMiniFetch, true);
  assert.equal(binding.attributes[1].byteOffset, 4);
});

test('mini fetch with no preceding full fetch traps instead of fabricating a stride', () => {
  const mini = vfetchWords({ dstReg: 2, constIndex: 0, constIndexSel: 0, format: 36, stride: 0, offset: 0, isMini: 1 });
  const ucode = exec(cfPair(execCf(1, 1, seq(SEQ_FETCH)), NOP_CF), mini);
  assert.throws(() => VF.analyzeVertexShader(ucode), /mini/i);
});

test('trailing instruction words that alias an exec CF do not corrupt the CF bound', () => {
  // The vfetch's w1 low nibble is 0x3 = kCondExec: read as a CF pair the
  // instruction looks like an EXEC-family CF with address 0. Pass 1 must shrink
  // the CF bound live (like xenia / the WGSL translator) and never treat
  // instruction words as control flow — this exact aliasing silently dropped
  // the fetch before the fix.
  const aliased = vfetchWords({ dstReg: 1, constIndex: 0, constIndexSel: 1, format: 6 /* k_8_8_8_8 */, stride: 1, isSigned: 1, numFormatAll: 1 });
  assert.equal((aliased[1] >>> 12) & 0xF, 0x3, 'fixture must alias kCondExec');
  const ucode = exec(
    cfPair(execCf(1, 1, seq(SEQ_FETCH)), NOP_CF),
    aliased,
  );
  const analysis = VF.analyzeVertexShader(ucode);
  assert.equal(analysis.controlFlowPairBound, 1);
  assert.equal(analysis.vertexFetchInstructionCount, 1);
  assert.equal(analysis.bindings.length, 1);
  assert.equal(analysis.bindings[0].attributes[0].format, VF.VERTEX_FORMAT.k_8_8_8_8);
});

test('exec CF pointing past the blob traps loudly', () => {
  const ucode = exec(cfPair(execCf(7, 1, seq(SEQ_FETCH)), NOP_CF), ALU_ZERO);
  assert.throws(() => VF.analyzeVertexShader(ucode), /out of shader/);
});

test('ucode length not a multiple of 3 dwords traps', () => {
  assert.throws(() => VF.analyzeVertexShader(new Uint32Array(7)), /multiple of 3/);
});

// ---------------------------------------------------------------- format mapping
test('mapVertexFormat maps the representable Xenos formats', () => {
  const m = (format, opts = {}) => VF.mapVertexFormat({ format, isSigned: 0, isNormalized: 0, expAdjust: 0, ...opts });
  assert.equal(m(VF.VERTEX_FORMAT.k_32_FLOAT), 'float32');
  assert.equal(m(VF.VERTEX_FORMAT.k_32_32_FLOAT), 'float32x2');
  assert.equal(m(VF.VERTEX_FORMAT.k_32_32_32_FLOAT), 'float32x3');
  assert.equal(m(VF.VERTEX_FORMAT.k_32_32_32_32_FLOAT), 'float32x4');
  assert.equal(m(VF.VERTEX_FORMAT.k_16_16_FLOAT), 'float16x2');
  assert.equal(m(VF.VERTEX_FORMAT.k_16_16_16_16_FLOAT), 'float16x4');
  assert.equal(m(VF.VERTEX_FORMAT.k_32), 'uint32');
  assert.equal(m(VF.VERTEX_FORMAT.k_32, { isSigned: 1 }), 'sint32');
  assert.equal(m(VF.VERTEX_FORMAT.k_32_32, {}), 'uint32x2');
  assert.equal(m(VF.VERTEX_FORMAT.k_32_32_32_32, { isSigned: 1 }), 'sint32x4');
  assert.equal(m(VF.VERTEX_FORMAT.k_16_16, { isNormalized: 1, isSigned: 1 }), 'snorm16x2');
  assert.equal(m(VF.VERTEX_FORMAT.k_16_16, { isNormalized: 1 }), 'unorm16x2');
  assert.equal(m(VF.VERTEX_FORMAT.k_16_16_16_16, {}), 'uint16x4');
  assert.equal(m(VF.VERTEX_FORMAT.k_16_16_16_16, { isNormalized: 1, isSigned: 1 }), 'snorm16x4');
  assert.equal(m(VF.VERTEX_FORMAT.k_8_8_8_8, { isNormalized: 1 }), 'unorm8x4');
  assert.equal(m(VF.VERTEX_FORMAT.k_8_8_8_8, { isSigned: 1 }), 'sint8x4');
  assert.equal(m(VF.VERTEX_FORMAT.k_2_10_10_10, { isNormalized: 1 }), 'unorm10-10-10-2');
});

test('mapVertexFormat traps on formats WebGPU cannot represent', () => {
  const m = (format, opts = {}) => VF.mapVertexFormat({ format, isSigned: 0, isNormalized: 0, expAdjust: 0, ...opts });
  assert.throws(() => m(VF.VERTEX_FORMAT.k_10_11_11), /11/);
  assert.throws(() => m(VF.VERTEX_FORMAT.k_11_11_10), /11/);
  assert.throws(() => m(VF.VERTEX_FORMAT.k_32, { isNormalized: 1 }), /unorm|snorm/i);
  assert.throws(() => m(VF.VERTEX_FORMAT.k_2_10_10_10, { isSigned: 1, isNormalized: 1 }), /2-10-10-10/i);
  assert.throws(() => m(VF.VERTEX_FORMAT.k_32, { expAdjust: 3 }), /exp_adjust/);
  assert.throws(() => m(0), /kUndefined/);
});

// ---------------------------------------------------------------- GPU descriptors
function singleBindingAnalysis(attrOverrides, bindingOverrides = {}) {
  const base = {
    bindingIndex: 0,
    fetchConstant: 0,
    strideWords: 4,
    attributes: [{
      instructionIndex: 1, destRegister: 4, destRegisterRelative: 0, writtenMask: 0xF,
      usedSourceComponents: 0b0111, offsetWords: 0, byteOffset: 0,
      format: VF.VERTEX_FORMAT.k_32_32_32_FLOAT, formatName: 'k_32_32_32_FLOAT',
      componentCount: 3, isSigned: 0, isNormalized: false, signedRFMode: 0,
      expAdjust: 0, isIndexRounded: 0, isPredicated: 0, isMiniFetch: 0,
      ...attrOverrides,
    }],
    ...bindingOverrides,
  };
  return { bindings: [base] };
}

test('buildGpuVertexBufferDescriptors emits GPUVertexBufferLayout-shaped entries', () => {
  const descriptors = VF.buildGpuVertexBufferDescriptors(singleBindingAnalysis());
  assert.equal(descriptors.length, 1);
  assert.deepEqual(descriptors[0], {
    bindingIndex: 0,
    fetchConstant: 0,
    arrayStride: 16,
    stepMode: 'vertex',
    attributes: [{ shaderLocation: 4, offset: 0, format: 'float32x3' }],
  });
});

test('attribute that does not fit the stride traps', () => {
  assert.throws(
    () => VF.buildGpuVertexBufferDescriptors(singleBindingAnalysis({ offsetWords: 2, byteOffset: 8 }, {})),
    /stride/,
  );
});

test('negative attribute offset traps (not expressible in WebGPU)', () => {
  assert.throws(
    () => VF.buildGpuVertexBufferDescriptors(singleBindingAnalysis({ offsetWords: -1, byteOffset: -4 })),
    /negative/,
  );
});

test('stride-0 binding (same-location fetch) traps', () => {
  assert.throws(
    () => VF.buildGpuVertexBufferDescriptors(singleBindingAnalysis({}, { strideWords: 0 })),
    /stride 0/,
  );
});

test('duplicate GPR across attributes traps (shaderLocation collision)', () => {
  const a = singleBindingAnalysis().bindings[0].attributes[0];
  const analysis = {
    bindings: [
      { bindingIndex: 0, fetchConstant: 0, strideWords: 4, attributes: [a] },
      { bindingIndex: 1, fetchConstant: 1, strideWords: 4, attributes: [{ ...a, instructionIndex: 2 }] },
    ],
  };
  assert.throws(() => VF.buildGpuVertexBufferDescriptors(analysis), /collision/);
});
