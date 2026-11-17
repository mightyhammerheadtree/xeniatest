/**
 * wasm_emitter.js — minimal WebAssembly binary module builder.
 * Just enough of the format to emit the recompiled guests: types, imports,
 * memory, mutable globals, and functions with attached local declarations.
 * No data section needed (static dispatch is by BrTable).
 */

const TYPE_I32 = 0x7f, TYPE_I64 = 0x7e, TYPE_F32 = 0x7d, TYPE_F64 = 0x7c;

export const Const = {
  i32: (v) => [TYPE_I32, 0x41, ...sleb(BigInt(v))],
  i64: (v) => [TYPE_I64, 0x42, ...sleb(BigInt(v))],
  f64: (v) => { const b = new Uint8Array(8); new DataView(b.buffer).setFloat64(0, v, true); return [TYPE_F64, 0x44, ...b]; },
};

export function sleb(v) { // signed LEB128
  const out = [];
  let more = true;
  while (more) {
    let byte = Number(v & 0x7fn);
    v >>= 7n;
    if ((v === 0n && (byte & 0x40) === 0) || (v === -1n && (byte & 0x40) !== 0)) more = false;
    else byte |= 0x80;
    out.push(byte);
  }
  return out;
}

export function uleb(v) {
  let n = BigInt(v) < 0n ? BigInt(v) : v; // callers pass non-negative
  let x = typeof v === 'bigint' ? v : BigInt(v);
  const out = [];
  do {
    let byte = Number(x & 0x7fn);
    x >>= 7n;
    if (x > 0n) byte |= 0x80;
    out.push(byte);
  } while (x > 0n);
  return out;
}

export class WasmBuilder {
  constructor() {
    this.types = [];
    this.imports = [];
    this.functions = [];   // type index per function
    this.funcLocals = [];  // local decls per function
    this.funcBodies = [];
    this.memory = null;    // { min, max }
    this.globals = [];
    this.exports = [];
  }

  addType(params, results) {
    this.types.push({ params, results });
    return this.types.length - 1;
  }

  addImport(module, field, typeIdx) {
    this.imports.push({ module, field, kind: 0, typeIdx });
    return this.imports.length - 1;
  }

  addMemory(minPages, maxPages) {
    this.memory = { min: minPages, max: maxPages };
    // memory index is implicit; export kind 2 indexes memories
    return 0;
  }

  addGlobal(valType, mutable, initOps) {
    this.globals.push({ valType, mutable, init: [...initOps, 0x0b] });
    return this.globals.length - 1;
  }

  addFunction(typeIdx, localDefs, bodyOps) {
    // localDefs: array of [count, valType]
    this.functions.push(typeIdx);
    this.funcLocals.push(localDefs);
    this.funcBodies.push([...bodyOps, 0x0b]);
    // function index space = imports then functions
    return this.imports.length + this.functions.length - 1;
  }

  addExport(name, kind, index) {
    this.exports.push({ name, kind, index });
  }

  build() {
    const bytes = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
    const section = (id, payload) => {
      bytes.push(id, ...uleb(payload.length), ...payload);
    };

    // type section
    const typePayload = [...uleb(this.types.length)];
    for (const t of this.types) {
      typePayload.push(0x60, ...uleb(t.params.length), ...t.params, ...uleb(t.results.length), ...t.results);
    }
    section(1, typePayload);

    // import section
    const importPayload = [...uleb(this.imports.length)];
    for (const imp of this.imports) {
      const m = str(imp.module), f = str(imp.field);
      importPayload.push(...uleb(m.length), ...m, ...uleb(f.length), ...f, imp.kind, ...uleb(imp.typeIdx));
    }
    section(2, importPayload);

    // function section
    const funcPayload = [...uleb(this.functions.length)];
    for (const t of this.functions) funcPayload.push(...uleb(t));
    section(3, funcPayload);

    // memory section
    if (this.memory) {
      const flags = this.memory.max !== undefined ? 0x01 : 0x00;
      const memPayload = [0x01, flags, ...uleb(this.memory.min)];
      if (flags) memPayload.push(...uleb(this.memory.max));
      section(5, memPayload);
    }

    // global section
    const globalPayload = [...uleb(this.globals.length)];
    for (const g of this.globals) {
      globalPayload.push(g.valType, g.mutable ? 1 : 0, ...g.init);
    }
    section(6, globalPayload);

    // export section
    const exportPayload = [...uleb(this.exports.length)];
    for (const e of this.exports) {
      const nm = str(e.name);
      exportPayload.push(...uleb(nm.length), ...nm, e.kind, ...uleb(e.index));
    }
    section(7, exportPayload);

    // code section
    const codePayload = [...uleb(this.functions.length)];
    for (let i = 0; i < this.functions.length; i++) {
      const body = [];
      const locals = this.funcLocals[i];
      body.push(...uleb(locals.length));
      for (const [count, type] of locals) body.push(...uleb(count), type);
      body.push(...this.funcBodies[i]);
      codePayload.push(...uleb(body.length), ...body);
    }
    section(10, codePayload);

    return new Uint8Array(bytes);
  }
}

function str(s) {
  const enc = new TextEncoder().encode(s);
  return Array.from(enc);
}

/** Instruction byte helpers (raw opcode sequences). */
export const Ops = {
  unreachable: 0x00, nop: 0x01,
  block: (bt = 0x40) => [0x02, bt], loop: (bt = 0x40) => [0x03, bt], if_: (bt = 0x40) => [0x04, bt],
  else: 0x05, end: 0x0b,
  br: (l) => [0x0c, ...uleb(l)], br_if: (l) => [0x0d, ...uleb(l)],
  br_table: (labels, def) => [0x0e, ...uleb(labels.length), ...labels.flatMap(uleb), ...uleb(def)],
  return: 0x0f,
  call: (f) => [0x10, ...uleb(f)],
  local_get: (l) => [0x20, ...uleb(l)], local_set: (l) => [0x21, ...uleb(l)], local_tee: (l) => [0x22, ...uleb(l)],
  global_get: (g) => [0x23, ...uleb(g)], global_set: (g) => [0x24, ...uleb(g)],
  i32_load: (a = 0, o = 0) => [0x28, ...uleb(a), ...uleb(o)],
  i64_load: (a = 0, o = 0) => [0x29, ...uleb(a), ...uleb(o)],
  i32_load8_u: (a = 0, o = 0) => [0x2d, ...uleb(a), ...uleb(o)],
  i32_load16_u: (a = 0, o = 0) => [0x2f, ...uleb(a), ...uleb(o)],
  i32_load16_s: (a = 0, o = 0) => [0x2e, ...uleb(a), ...uleb(o)],
  i64_load32_u: (a = 0, o = 0) => [0x35, ...uleb(a), ...uleb(o)],
  f64_load: (a = 0, o = 0) => [0x2b, ...uleb(a), ...uleb(o)],
  i32_store: (a = 0, o = 0) => [0x36, ...uleb(a), ...uleb(o)],
  i64_store: (a = 0, o = 0) => [0x37, ...uleb(a), ...uleb(o)],
  f64_store: (a = 0, o = 0) => [0x39, ...uleb(a), ...uleb(o)],
  i32_store8: (a = 0, o = 0) => [0x3a, ...uleb(a), ...uleb(o)],
  i32_store16: (a = 0, o = 0) => [0x3b, ...uleb(a), ...uleb(o)],
  // i32/i64 constants are reduced mod 2^32 / 2^64 and reinterpreted as signed
  // before encoding — raw uint32 bit patterns (e.g. 0xFF00FF00) would otherwise
  // produce varints that overflow the signed range ("extra bits in varint").
  i32_const: (v) => { let x = BigInt(v) & 0xffffffffn; if (x >= 0x80000000n) x -= 0x100000000n; return [0x41, ...sleb(x)]; },
  i64_const: (v) => { let x = BigInt(v) & 0xffffffffffffffffn; if (x >= 0x8000000000000000n) x -= 0x10000000000000000n; return [0x42, ...sleb(x)]; },
  f64_const: (v) => Const.f64(v).slice(1),
  drop: 0x1a, select: 0x1b,
  i32_eq: 0x46, i32_ne: 0x47, i32_lt_s: 0x48, i32_gt_s: 0x4a, i32_le_s: 0x4c, i32_ge_s: 0x4e,
  i32_lt_u: 0x49, i32_gt_u: 0x4b, i32_le_u: 0x4d, i32_ge_u: 0x4f,
  i32_eqz: 0x45,
  i64_eq: 0x51, i64_ne: 0x52, i64_lt_s: 0x53, i64_gt_s: 0x55, i64_le_s: 0x57, i64_ge_s: 0x59,
  i64_lt_u: 0x54, i64_gt_u: 0x56, i64_le_u: 0x58, i64_ge_u: 0x5a,
  i64_eqz: 0x50,
  f64_eq: 0x61, f64_lt: 0x63, f64_gt: 0x64, f64_le: 0x65, f64_ge: 0x66,
  i32_clz: 0x67, i32_ctz: 0x68, i32_popcnt: 0x69,
  i32_add: 0x6a, i32_sub: 0x6b, i32_mul: 0x6c, i32_div_s: 0x6d, i32_div_u: 0x6e,
  i32_and: 0x71, i32_or: 0x72, i32_xor: 0x73, i32_shl: 0x74, i32_shr_s: 0x75, i32_shr_u: 0x76, i32_rotl: 0x77, i32_rotr: 0x78,
  i64_add: 0x7c, i64_sub: 0x7d, i64_mul: 0x7e, i64_div_s: 0x7f, i64_div_u: 0x80,
  i64_and: 0x83, i64_or: 0x84, i64_xor: 0x85, i64_shl: 0x86, i64_shr_s: 0x87, i64_shr_u: 0x88, i64_rotl: 0x89, i64_rotr: 0x8a,
  i32_wrap_i64: 0xa7, i64_extend_i32_s: 0xac, i64_extend_i32_u: 0xad,
  i32_extend8_s: 0xc0, i32_extend16_s: 0xc1, i64_extend8_s: 0xc2, i64_extend16_s: 0xc3, i64_extend32_s: 0xc4,
  f64_add: 0xa0, f64_sub: 0xa1, f64_mul: 0xa2, f64_div: 0xa3,
  f64_abs: 0x99, f64_neg: 0x9a,
  f64_promote_f32: 0xbb, f32_demote_f64: 0xb6,
  f32_reinterpret_i32: 0xbe, f64_reinterpret_i64: 0xbf, i64_reinterpret_f64: 0xbd, i32_reinterpret_f32: 0xbc,
  f64_ne: 0x62, f64_add_: 0xa0,
};

export function packOps(...chunks) {
  const out = [];
  for (const c of chunks) {
    if (c === undefined) continue;
    if (Array.isArray(c)) out.push(...c);
    else out.push(c);
  }
  return out;
}
