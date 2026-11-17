/**
 * recompiler.js — PPC64 (Xbox 360) → WebAssembly static recompiler.
 *
 * Reads a guest image (flat bytes + base VMA + entry VMA), discovers
 * fragments by following control flow, lifts each to a wasm function, and
 * links them with shared mutable global state (mirrors the C lifter in
 * tools/xenorecomp). Branches:
 *
 *   b/bl   → block within the fragment (jump) or a wasm call to the
 *            fragment containing the target (every direct-branch target is
 *            forced to be a fragment start, so calls are always direct)
 *   bc*    → in-fragment block selection via local $blk + br_table
 *   blr    → wasm `return` (guest LR semantics fall out of direct calls)
 *   bctr   → if a jump table was detected statically for this address, a
 *            compare chain on CTR calling the case fragments (labels are
 *            seeded as fragments); otherwise the imported host icall
 *            (runtime resolution; loud if unresolved)
 *
 * MMIO console window [0x00F00000, 0x00F01000) and the exit door at
 * 0x00F00FFC are routed to host imports (xenorecomp.mmio_store / mmio_load),
 * identical in behavior to the guest's architectural semantics.
 */
import { decodeInstruction } from './ppc_decode.js';
import { WasmBuilder, Ops, sleb, uleb } from './wasm_emitter.js';
import { loadXex2, XexLoadError } from './xex_loader.js';
import { detectSwitchTables } from './switch_tables.js';
import { detectSaveRestoreStubs, emitVmxStubOps } from './save_restore.js';
import './hle.js'; // UMD side effect: sets globalThis.XenonHLE (the HLE host)
import './hle_tables.js'; // UMD side effect: sets globalThis.XenonHLETables (ordinal → name)

export const GUEST_MEM_MASK = 0x03ffffff; // 64 MiB flat arena
export const GUEST_MEM_SIZE = 64 * 1024 * 1024;
export const MMIO_BASE = 0x00f00000;
export const MMIO_END = 0x00f01000;

const HLE_CTX_ADDR = 0x00F11000; // scratch area: r0..r31 as 32 × u64 BE (read by the HLE host)

const G = {
  R0: 0,           // r0..r31 → globals 0..31
  R3: 3,           // r3 — holds HLE/indirect-call return values
  F0: 32,          // f0..f31 → globals 32..63
  CR: 64,          // i32
  XER: 65,         // i32
  LR: 66,          // i64
  CTR: 67,         // i64
};
const L_BLK = 0, L_T64 = 1, L_T64B = 2, L_T32 = 3, L_T32B = 4;

const IMAGE_BASE_DEFAULT = 0x10000;

export class RecompError extends Error {}

/** Load/segment helpers ------------------------------------------------ */

/** Build the flat guest arena image from parsed file info. */
export function buildImage(kind, parsed, bytes, opts = {}) {
  const arena = new Uint8Array(GUEST_MEM_SIZE);
  let entry = 0;
  let base = IMAGE_BASE_DEFAULT;
  const segments = [];
  let hleCalls; // Map<addr, {addr, type, ordinal, moduleId, moduleName, name}> — XEX2 imports only
  let functionStarts = []; // seed function starts (XEX2 .pdata — XenonRecomp-style discovery); empty for bin/elf
  let peSections = null;   // XEX2 PE section table (null when unparseable or not XEX2)
  let peError = null;      // XEX2 PE parse breadcrumb, if any
  let switchTables = new Map(); // masked bctr addr → switch table descriptor (XEX2 code sections)
  let vmxStubs = new Map();   // masked VMX save/restore entry addr → synthesized-stub descriptor
  if (kind === 'bin') {
    const n = Math.min(bytes.length, GUEST_MEM_SIZE);
    arena.set(bytes.subarray(0, n), IMAGE_BASE_DEFAULT);
    segments.push({ addr: IMAGE_BASE_DEFAULT, data: bytes.subarray(0, n).slice() });
    entry = IMAGE_BASE_DEFAULT;
  } else if (kind === 'elf') {
    for (const ph of parsed.programHeaders) {
      if (ph.type !== 1) continue; // PT_LOAD
      const off = Number(ph.offset), vaddr = Number(ph.vaddr) & GUEST_MEM_MASK;
      const filesz = Number(ph.filesz), memsz = Number(ph.memsz);
      const n = Math.min(filesz, Math.max(0, GUEST_MEM_SIZE - vaddr));
      arena.set(bytes.subarray(off, off + n), vaddr);
      if (memsz > filesz) arena.fill(0, vaddr + n, Math.min(vaddr + memsz, GUEST_MEM_SIZE));
      segments.push({ addr: vaddr, data: bytes.subarray(off, off + n).slice() });
    }
    entry = Number(parsed.entry) & GUEST_MEM_MASK;
    base = entry;
  } else if (kind === 'xex2') {
    // XEX2: retail images are decrypted with the public retail XEX2 key
    // (the same constant xenia ships) — no console, no NAND, no CPU key.
    // Devkit images fall back automatically (xenia's own fallback). BASIC
    // decompression is supported; LZX/delta fail loudly. See docs/HLE.md.
    let loaded;
    try {
      loaded = loadXex2(bytes, opts);
    } catch (e) {
      if (e instanceof XexLoadError) throw new RecompError(e.message);
      throw e;
    }
    const loadBase = loaded.base & GUEST_MEM_MASK;
    if (loaded.image.length > GUEST_MEM_SIZE - loadBase) {
      throw new RecompError('XEX2 image is larger than the 64 MiB guest arena');
    }
    arena.set(loaded.image, loadBase);
    segments.push({ addr: loadBase, data: loaded.image.slice() });
    entry = loaded.entry & GUEST_MEM_MASK;
    base = entry;
    hleCalls = new Map(loaded.imports.filter((i) => i.type === 1).map((i) => [i.addr, i]));
    peSections = loaded.peSections ?? null;
    peError = loaded.peError ?? null;
    // Seeds: masked guest addresses, HLE import thunks excluded (those resolve
    // to host calls, never to recompiled code). discoverFragments re-filters.
    functionStarts = (loaded.pdataFunctions ?? [])
      .map((f) => f.addr & GUEST_MEM_MASK)
      .filter((a) => !hleCalls.has(a));
    // Jump-table detection over the code sections (XenonRecomp-style). The
    // table labels are intra-function destinations no direct branch reaches;
    // they are seeded as fragments too so the emitted compare chain can call
    // them. Any table that fails validation is dropped — its `bctr` keeps the
    // runtime `icall` behavior (loud on unknown targets).
    if (peSections) {
      const ranges = peSections
        .filter((s) => s.isCode)
        .map((s) => {
          const start = (loadBase + (s.virtualAddress & GUEST_MEM_MASK)) & GUEST_MEM_MASK;
          const size = (s.virtualSize || s.sizeOfRawData) >>> 0;
          return [start, Math.min(start + size, GUEST_MEM_SIZE)];
        });
      if (ranges.length) {
        const detected = detectSwitchTables(arena, { ranges, mask: GUEST_MEM_MASK });
        switchTables = detected.tables;
        const labels = new Set(functionStarts);
        for (const table of switchTables.values()) {
          for (const label of table.labels) {
            if (!hleCalls.has(label)) labels.add(label & GUEST_MEM_MASK);
          }
        }
        functionStarts = [...labels].sort((a, b) => a - b);
        // Save/restore stub detection (XenonRecomp-style): FPR/GPR entries are
        // liftable — seed them. VMX entries are seeded too but synthesized
        // (zero-filled save slots / no-op restore) since there is no VMX model.
        const saved = detectSaveRestoreStubs(arena, { ranges, mask: GUEST_MEM_MASK });
        vmxStubs = saved.vmxStubs;
        if (saved.seedAddresses.length) {
          for (const a of saved.seedAddresses) {
            if (!hleCalls.has(a)) labels.add(a);
          }
          functionStarts = [...labels].sort((a, b) => a - b);
        }
      }
    }
  } else {
    throw new RecompError('unsupported image kind: ' + kind);
  }
  return { arena, entry, base, segments, hleCalls, functionStarts, peSections, peError, switchTables, vmxStubs };
}

/** Discovery -------------------------------------------------------------- */

function readWordBE(arena, addr) {
  const a = addr & GUEST_MEM_MASK;
  return ((arena[a] << 24) | (arena[a + 1] << 16) | (arena[a + 2] << 8) | arena[a + 3]) >>> 0;
}

function isBranchTargetCandidate(insn) {
  return insn.isBranch && insn.branchTarget != null;
}

/** Does control continue at addr+4? (b is a hard stop; bl/bc* fall through.) */
function branchFallsThrough(insn) {
  const n = insn.name;
  if (n === 'b' || n === 'bctr' || n === 'blr' || n === 'blrl') return false;
  if (n === 'bl' || n === 'bctrl') return true;
  return insn.isBranch; // conditional bc*
}

/**
 * Discovery, phase A: collect entry + known function starts + every
 * direct-branch target as a fragment entry. `seedStarts` carries function
 * starts discovered outside the control flow — for XEX2 images these come
 * from the PE `.pdata` (IMAGE_CE_RUNTIME_FUNCTION), like XenonRecomp's
 * primary discovery seed. Phase B (finalizeFragments) cuts fragments at the
 * next known start, so every direct branch becomes a wasm call or block jump
 * with no exponential re-scanning.
 */
export function discoverFragments(arena, entry, hleCalls, seedStarts, vmxStubs) {
  const starts = new Set([entry]);
  if (Array.isArray(seedStarts)) {
    for (const s of seedStarts) {
      const a = (Number(s) >>> 0) & GUEST_MEM_MASK;
      if (a + 4 <= GUEST_MEM_SIZE && !(hleCalls && hleCalls.has(a))) starts.add(a);
    }
  }
  const queue = [...starts];
  const inImage = (a) => a >= 0 && a + 4 <= GUEST_MEM_SIZE;
  while (queue.length) {
    const f = queue.pop();
    let addr = f, steps = 0;
    while (steps++ < (1 << 16)) {
      if (!inImage(addr)) break;
      const word = readWordBE(arena, addr);
      if (word === 0) break;
      const insn = decodeInstruction(word, addr);
      if (insn.branchTarget != null) {
        const t = insn.branchTarget & GUEST_MEM_MASK;
        if (inImage(t) && !(hleCalls && hleCalls.has(t)) && !starts.has(t)) {
          starts.add(t);
          queue.push(t);
        }
      }
      if (insn.isBranch || insn.ends) {
        if (branchFallsThrough(insn)) {
          const f = addr + 4;
          if (inImage(f) && !(hleCalls && hleCalls.has(f)) && !starts.has(f)) {
            starts.add(f);
            queue.push(f);
          }
        }
        break;
      }
      addr += 4;
    }
  }
  return finalizeFragments(arena, entry, starts, vmxStubs);
}

function finalizeFragments(arena, entry, startsSet, vmxStubs) {
  const starts = [...startsSet].sort((a, b) => a - b);
  const startSet = new Set(starts);
  const fragments = new Map();
  for (const start of starts) {
    const insns = [];
    let addr = start, steps = 0;
    let cutTo = null;
    // Synthesized VMX stub: no guest bytes are decoded; the body is emitted
    // from the descriptor (see emitVmxStubOps).
    const vmxStub = vmxStubs ? vmxStubs.get(start) : null;
    if (vmxStub) { fragments.set(start, { start, insns: [], cutTo: null, vmxStub }); continue; }
    while (steps++ < (1 << 16)) {
      if (addr + 4 > GUEST_MEM_SIZE) break;
      if (addr !== start && startSet.has(addr)) { cutTo = addr; break; } // cut: falls into another fragment
      const word = readWordBE(arena, addr);
      if (word === 0 && addr > start) break; // zero padding — trap (data/padding)
      const insn = decodeInstruction(word, addr);
      insns.push(insn);
      if (insn.isBranch || insn.ends) {
        const f = addr + 4;
        if (startSet.has(f)) cutTo = f; // fallthrough lands in another fragment
        break;
      }
      addr += 4;
    }
    fragments.set(start, { start, insns, cutTo });
  }
  return fragments;
}

// ---------- WASM emission -------------------------------------------------

const O = Ops;
const B = (fn) => Array.from({ length: fn }, (_, i) => i);

const T_VOID = 0x40;
const T_FUNC_NONE = 0, T_I32_VOID = 1, T_I64_VOID = 2, T_I64I32_VOID = 3, T_I64_I64 = 4;

class Lifter {
  constructor(fragments, opts, hleCalls) {
    this.fragments = fragments;               // Map start -> {start, insns}
    this.starts = new Set(fragments.keys());
    this.hleCalls = hleCalls ?? null;        // Map<addr, import descriptor> — XEX2 imports only
    this.switchTables = opts?.switchTables ?? null; // Map<bctr addr, descriptor> — detected jump tables
    this.b = new WasmBuilder();
    // Types
    this.tyNone = this.b.addType([], []);
    this.tyI32 = this.b.addType([0x7f], []);
    this.tyI64 = this.b.addType([0x7e], []);
    this.tyI64I32 = this.b.addType([0x7e, 0x7f], []);
    this.tyI64I64I32 = this.b.addType([0x7e, 0x7e, 0x7f], []);
    // Imports (indices 0..4 are imports)
    this.imPUTC = this.b.addImport('xenorecomp', 'putc', this.tyI32);
    this.imEXIT = this.b.addImport('xenorecomp', 'exit', this.tyI32);
    this.imUNIMPL = this.b.addImport('xenorecomp', 'unimpl', this.tyI64I32);
    this.imICALL = this.b.addImport('xenorecomp', 'icall', this.tyI64);
    this.imMMIO_STORE = this.b.addImport('xenorecomp', 'mmio_store', this.tyI64I64I32);
    this.imMMIO_LOAD = this.b.addImport('xenorecomp', 'mmio_load', this.b.addType([0x7e, 0x7f], [0x7e]));
    this.imSC = this.b.addImport('xenorecomp', 'sc', this.tyI64);
    // Local helper (function index 7, right after the imports): writes r0..r31
    // big-endian into HLE_CTX_ADDR for the HLE host. Body is spliced in at build time.
    this.storeCtxIndex = this.b.addFunction(this.tyNone, [[1, 0x7e], [1, 0x7f]], []);
    // Memory: 64 MiB flat arena, fixed (no growth, no OOM).
    this.b.addMemory(1024, 1024);
    // Guest state globals
    for (let i = 0; i < 32; i++) this.b.addGlobal(0x7e, true, O.i64_const(0n));
    for (let i = 0; i < 32; i++) this.b.addGlobal(0x7c, true, O.f64_const(0.0));
    this.b.addGlobal(0x7f, true, O.i32_const(0)); // CR
    this.b.addGlobal(0x7f, true, O.i32_const(0)); // XER
    this.b.addGlobal(0x7e, true, O.i64_const(0n)); // LR
    this.b.addGlobal(0x7e, true, O.i64_const(0n)); // CTR
    this.stats = { fragments: fragments.size, insns: 0, unimpl: 0, blocks: 0, switchTables: this.switchTables?.size ?? 0, vmxStubs: 0 };
  }

  // -- operand helpers --
  raOrZero(ops, regPos = 1) {
    const reg = ops[regPos];
    return reg === 0 ? O.i64_const(0n) : O.global_get(reg);
  }
  /** Effective address (i64) on the stack. */
  ea(ops, reg = 'RA', off = 'SIMM') {
    const out = [...this.lifter.raOrZero(ops, this._regIndex(reg))];
    out.push(...O.i64_const(BigInt(ops[off] ?? 0)));
    out.push(O.i64_add);
    return out;
  }
  _regIndex() { return 1; } // helper kept for readability; EA uses [RT, RA, SIMM] order
  /** mask i64 addr → i32 memory index */
  addrMem() {
    return [O.i64_const(BigInt(GUEST_MEM_MASK)), O.i64_and, O.i32_wrap_i64];
  }
  /** set CR field (0..7) from crf bits i32 on stack (lt<<3|gt<<2|eq<<1|so) */
  crStoreFieldBits(field) {
    const shift = 28 - field * 4;
    const mask = (0x0f << shift) >>> 0;
    return [
      O.i32_const(shift), O.i32_shl,
      O.global_get(G.CR), O.i32_const((~mask) >>> 0), O.i32_and, O.i32_or, O.global_set(G.CR),
    ];
  }
  /** three-way compare → CR field */
  crSetCompare(field, aOps, bOps, isUnsigned, is64) {
    const lt = is64 ? (isUnsigned ? O.i64_lt_u : O.i64_lt_s) : (isUnsigned ? O.i32_lt_u : O.i32_lt_s);
    const gt = is64 ? (isUnsigned ? O.i64_gt_u : O.i64_gt_s) : (isUnsigned ? O.i32_gt_u : O.i32_gt_s);
    const eq = is64 ? O.i64_eq : O.i32_eq;
    return [
      ...aOps, ...bOps, lt, O.i32_const(3), O.i32_shl,
      ...aOps, ...bOps, gt, O.i32_const(2), O.i32_shl, O.i32_or,
      ...aOps, ...bOps, eq, O.i32_const(1), O.i32_shl, O.i32_or,
      ...this.crStoreFieldBits(field),
    ];
  }
  /** CR0 from an i32 result value (already in local L_T32B) */
  cr0FromI32Local() {
    return [
      O.local_get(L_T32B), O.i32_const(0), O.i32_lt_s, O.i32_const(3), O.i32_shl,
      O.local_get(L_T32B), O.i32_const(0), O.i32_gt_s, O.i32_const(2), O.i32_shl, O.i32_or,
      O.local_get(L_T32B), O.i32_eqz, O.i32_const(1), O.i32_shl, O.i32_or,
      ...this.crStoreFieldBits(0),
    ];
  }
  /** CR0 from an i64 result value (already in local L_T64B) */
  cr0FromI64Local() {
    return [
      O.local_get(L_T64B), O.i64_const(0n), O.i64_lt_s, O.i32_const(3), O.i32_shl,
      O.local_get(L_T64B), O.i64_const(0n), O.i64_gt_s, O.i32_const(2), O.i32_shl, O.i32_or,
      O.local_get(L_T64B), O.i64_eqz, O.i32_const(1), O.i32_shl, O.i32_or,
      ...this.crStoreFieldBits(0),
    ];
  }
}

/**
 * Build the body of the local `store_ctx` helper: copies all 32 GPRs
 * (wasm globals 0..31) into memory at HLE_CTX_ADDR as big-endian u64s.
 * Emitted once per module and called before every indirect call so the
 * HLE host can read the guest context (xenia's ppc_context save).
 * Locals: 0 = t64 (i64), 1 = t32 (i32).
 */
function buildStoreCtxOps() {
  const O2 = Ops;
  const bswapToL1 = () => [ // consume stack i32 → bswapped value in local 1
    O2.local_set(1),
    O2.local_get(1), O2.i32_const(8), O2.i32_shr_u, O2.i32_const(0x00ff00ff), O2.i32_and,
    O2.local_get(1), O2.i32_const(8), O2.i32_shl, O2.i32_const(0xff00ff00), O2.i32_and, O2.i32_or, O2.local_set(1),
    O2.local_get(1), O2.i32_const(16), O2.i32_shr_u,
    O2.local_get(1), O2.i32_const(16), O2.i32_shl, O2.i32_or, O2.local_set(1),
  ];
  const ops = [];
  for (let i = 0; i < 32; i++) {
    const addr = HLE_CTX_ADDR + i * 8;
    // high 32 bits, big-endian at addr
    ops.push(...O2.global_get(i), ...O2.local_tee(0), ...O2.i64_const(32n), O2.i64_shr_u, O2.i32_wrap_i64, ...bswapToL1(),
      ...O2.i32_const(addr), ...O2.local_get(1), ...O2.i32_store(2, 0));
    // low 32 bits, big-endian at addr+4
    ops.push(O2.local_get(0), O2.i32_wrap_i64, ...bswapToL1(),
      ...O2.i32_const(addr + 4), ...O2.local_get(1), ...O2.i32_store(2, 0));
  }
  return flattenOps(ops);
}

// ---------- fragments / basic blocks ----------

class FragmentCompiler {
  /** @param {Lifter} lifter @param {{start:number,insns:object[]}} fragment @param {number} fragCutTo next fragment's start (or null) @param {Map<number,number>} fnIndex wasm func idx per fragment start */
  constructor(lifter, fragment, fragCutTo, fnIndex) {
    this.lifter = lifter;
    this.fragment = fragment;
    this.cutTo = fragCutTo;
    this.fnIndex = fnIndex;
    this.blocks = this.blockSplit();
    this.blocksByAddr = new Map(this.blocks.map((b, i) => [b.start, i]));
  }

  blockSplit() {
    const targets = new Set();
    for (const insn of this.fragment.insns) if (insn.branchTarget != null) targets.add(insn.branchTarget & GUEST_MEM_MASK);
    const blocks = [];
    let cur = null;
    for (const insn of this.fragment.insns) {
      if (!cur) cur = { start: insn.address, insns: [] };
      else if (targets.has(insn.address)) { blocks.push(cur); cur = { start: insn.address, insns: [] }; }
      cur.insns.push(insn);
      if (insn.isBranch || insn.ends) { blocks.push(cur); cur = null; }
    }
    if (cur) blocks.push(cur);
    return blocks;
  }

  fnIdxOf(addr) { return this.fnIndex.get(addr & GUEST_MEM_MASK) ?? -1; }

  isInFragment(addr) { return this.blocksByAddr.has(addr & GUEST_MEM_MASK); }

  emit(b) {
    const blocks = this.blocks, n = blocks.length;
    const ops = [];
    ops.push(...O.i32_const(0), ...O.local_set(L_BLK)); // entry = block 0
    ops.push(...O.loop(T_VOID));
    for (let k = 0; k < n; k++) ops.push(...O.block(T_VOID));
    const labels = [];
    for (let k = 0; k < n; k++) labels.push(n - 1 - k);
    ops.push(...O.local_get(L_BLK));
    ops.push(...O.br_table(labels, Math.max(0, n - 1)));
    for (let k = n - 1; k >= 0; k--) {
      ops.push(O.end); // /block k
      ops.push(...this.emitBlock(blocks[k], k));
      ops.push(...O.br(k)); // $dispatch — depth k (enclosed in k blocks)
    }
    ops.push(O.end); // /loop
    return flattenOps(ops);
  }

  setBlkAndDispatch(blockIdx, enclosing) {
    return [...O.i32_const(blockIdx), ...O.local_set(L_BLK), ...O.br(enclosing)];
  }

  /** Jump (tail call) to another fragment: call + return. */
  tailCall(addr) {
    const a = addr & GUEST_MEM_MASK;
    if (this.lifter.hleCalls?.has(a)) {
      // Tail-call into an import thunk: snapshot the context and go through icall.
      return [...O.call(this.lifter.storeCtxIndex), ...O.i64_const(BigInt(a)), ...O.call(this.lifter.imICALL), O.return];
    }
    const idx = this.fnIdxOf(a);
    if (idx >= 0) return [...O.call(idx), O.return];
    // Unknown fragment: trap loudly via unimpl.
    return [...O.i64_const(BigInt(a)), ...O.i32_const(-1), ...O.call(this.lifter.imUNIMPL), O.unreachable];
  }

  /** Call another fragment inline (for bl). */
  callFragment(addr) {
    const a = addr & GUEST_MEM_MASK;
    if (this.lifter.hleCalls?.has(a)) {
      // bl to an import thunk: snapshot the context and go through icall.
      return [...O.call(this.lifter.storeCtxIndex), ...O.i64_const(BigInt(a)), ...O.call(this.lifter.imICALL)];
    }
    const idx = this.fnIdxOf(a);
    if (idx >= 0) return [...O.call(idx)];
    return [...O.i64_const(BigInt(a)), ...O.i32_const(-2), ...O.call(this.lifter.imUNIMPL)];
  }

  emitBranchCommon(insn, enclosing) {
    const bi = insn.BI, bo = insn.BO;
    const out = [];
    // cond bit: t = (cr >> (31 - bi)) & 1
    out.push(...O.global_get(G.CR), ...O.i32_const(31 - bi), O.i32_shr_u, ...O.i32_const(1), O.i32_and, ...O.local_set(L_T32));
    // cond_ok
    if (bo & 0x10) { out.push(...O.i32_const(1)); }
    else if (bo & 0x08) { out.push(...O.local_get(L_T32)); }
    else { out.push(...O.local_get(L_T32), ...O.i32_const(1), O.i32_xor); }
    out.push(...O.local_set(L_T32B));
    // ctr_ok
    if (!(bo & 0x04)) {
      out.push(...O.global_get(G.CTR), ...O.i64_const(1n), O.i64_sub, ...O.local_tee(L_T64), ...O.global_set(G.CTR));
      out.push(...O.local_get(L_T64), O.i32_wrap_i64);
      if (bo & 0x02) { out.push(O.i32_eqz); } // branch if CTR==0
      else { out.push(O.i32_eqz, O.i32_const(1), O.i32_xor); } // branch if CTR!=0
      out.push(...O.local_set(L_T32));
    } else {
      out.push(...O.i32_const(1), ...O.local_set(L_T32));
    }
    // br = cond_ok & ctr_ok
    out.push(...O.local_get(L_T32B), ...O.local_get(L_T32), O.i32_and, ...O.local_set(L_T32));
    return out;
  }

  emitCondBranch(insn, enclosing, nextBlockIdx) {
    const out = this.emitBranchCommon(insn, enclosing);
    const target = insn.branchTarget & GUEST_MEM_MASK;
    const tIdx = this.blocksByAddr.get(target);
    const tIsHere = tIdx !== undefined;
    // if (cond && ctr) { take } else { fall through }
    out.push(...O.local_get(L_T32));
    out.push(...O.if_(T_VOID));
    if (insn.LK) out.push(...O.i64_const(BigInt(insn.address + 4)), ...O.global_set(G.LR));
    if (tIsHere) out.push(...this.setBlkAndDispatch(tIdx, enclosing + 1));
    else out.push(...this.tailCall(target));
    out.push(O.else);
    // fallthrough
    if (nextBlockIdx !== null) out.push(...this.setBlkAndDispatch(nextBlockIdx, enclosing + 1));
    else out.push(...this.emitFallthrough(enclosing + 1));
    out.push(O.end);
    return out;
  }

  emitFallthrough(enclosing) {
    if (this.cutTo != null) return [...this.callFragment(this.cutTo), O.return];
    return [O.unreachable];
  }

  emitBlock(block, blockIdx) {
    const isLast = blockIdx === this.blocks.length - 1;
    const nextBlockIdx = isLast ? null : blockIdx + 1;
    const ops = [];
    if (globalThis.__RECOMP_DEBUG__) {
      for (const insn of block.insns) console.error('DBG insn', insn.name, 'RT=', insn.RT, 'RA=', insn.RA, 'SIMM=', insn.SIMM);
    }
    this.lifter.stats.blocks++;
    for (const insn of block.insns) {
      this.lifter.stats.insns++;
      const insnOps = this.emitInsn(insn, blockIdx, nextBlockIdx);
      if (globalThis.__RECOMP_DEBUG__) console.error('DBG ops', insn.name, JSON.stringify(insnOps));
      ops.push(...insnOps);
    }
    // If the block didn't end with a control transfer, fall through to the next fragment (cut) or stop.
    const last = block.insns[block.insns.length - 1];
    const endsWithControl = last && (last.isBranch || last.ends);
    if (!endsWithControl) {
      if (!isLast) ops.push(...this.setBlkAndDispatch(blockIdx + 1, blockIdx));
      else if (this.cutTo != null) ops.push(...this.callFragment(this.cutTo), O.return);
      else ops.push(O.unreachable);
    }
    return flattenOps(ops);
  }
}

// Deep-flatten nested byte arrays (defensive: any missed spread collapses here) and validate.
function flattenOps(ops) {
  const out = [];
  const walk = (a) => {
    for (const x of a) {
      if (Array.isArray(x)) walk(x);
      else if (typeof x === 'number' && x >= 0 && x <= 255) out.push(x);
      else throw new Error('recompiler emitted invalid byte: ' + String(x));
    }
  };
  walk(ops);
  return out;
}

// ---------- memory helpers (appended to FragmentCompiler) ----------

Object.assign(FragmentCompiler.prototype, {
  eaD(insn) {
    const out = insn.RA === 0 ? [O.i64_const(0n)] : [...O.global_get(insn.RA)];
    out.push(...O.i64_const(BigInt(insn.SIMM ?? 0)), O.i64_add);
    return out;
  },
  eaX(insn) {
    const out = insn.RA === 0 ? [O.i64_const(0n)] : [...O.global_get(insn.RA)];
    out.push(...O.global_get(insn.RB), O.i64_add);
    return out;
  },
  bswap32Seq() { // consumes nothing; transforms L_T32 → L_T32 swapped
    return [
      O.local_get(L_T32), O.i32_const(8), O.i32_shr_u, O.i32_const(0x00ff00ff), O.i32_and,
      O.local_get(L_T32), O.i32_const(8), O.i32_shl, O.i32_const(0xff00ff00), O.i32_and, O.i32_or,
      O.local_set(L_T32),
      O.local_get(L_T32), O.i32_const(16), O.i32_shr_u,
      O.local_get(L_T32), O.i32_const(16), O.i32_shl, O.i32_or,
      O.local_set(L_T32),
    ];
  },
  bswap64Seq() { // L_T64 → L_T64 swapped
    return [
      O.local_get(L_T64), O.i32_wrap_i64, O.local_set(L_T32),
      ...this.bswap32Seq(),
      O.local_get(L_T32), O.i64_extend_i32_u, O.i64_const(32n), O.i64_shl,
      O.local_set(L_T64B),
      O.local_get(L_T64), O.i64_const(32n), O.i64_shr_u, O.i32_wrap_i64, O.local_set(L_T32),
      ...this.bswap32Seq(),
      O.local_get(L_T32), O.i64_extend_i32_u, O.local_set(L_T64),
      O.local_get(L_T64B), O.local_get(L_T64), O.i64_or, O.local_set(L_T64),
    ];
  },
  bswap16Seq() { // L_T32 → L_T32 swapped (low 16 bits)
    return [
      O.local_get(L_T32), O.i32_const(8), O.i32_shl,
      O.local_get(L_T32), O.i32_const(8), O.i32_shr_u, O.i32_or,
      O.i32_const(0xffff), O.i32_and, O.local_set(L_T32),
    ];
  },
  mmioAddrCheck() { // stack i32 (1 if addr in MMIO window) — uses L_T64
    return [
      O.local_get(L_T64), O.i64_const(BigInt(MMIO_BASE)), O.i64_ge_u,
      O.local_get(L_T64), O.i64_const(BigInt(MMIO_END)), O.i64_lt_u, O.i32_and,
    ];
  },
  emitLoad(insn, signed, size, dest) {
    const L = this.lifter;
    const ea = (insn.RB !== undefined ? this.eaX(insn) : this.eaD(insn));
    const out = [...ea, ...O.local_set(L_T64)];
    // locked-prefix: update RA for -u forms AFTER ea is captured
    if (this._writebackRA(insn)) {
      out.push(...O.local_get(L_T64), ...O.global_set(insn.RA));
    }
    out.push(...this.mmioAddrCheck(), ...O.if_(0x7e));
    out.push(...O.local_get(L_T64), ...O.i32_const(size), ...O.call(L.imMMIO_LOAD));
    out.push(O.else);
    out.push(...O.local_get(L_T64), ...L.addrMem());
    if (size === 1) out.push(...O.i32_load8_u(0, 0), O.i64_extend_i32_u);
    else if (size === 2) out.push(...O.i32_load16_u(0, 0), ...O.local_set(L_T32), ...this.bswap16Seq(), O.local_get(L_T32), O.i64_extend_i32_u);
    else if (size === 4) out.push(...O.i32_load(0, 0), ...O.local_set(L_T32), ...this.bswap32Seq(), O.local_get(L_T32), O.i64_extend_i32_u);
    else out.push(...O.i64_load(0, 0), ...O.local_set(L_T64), ...this.bswap64Seq(), O.local_get(L_T64)); // value must land in L_T64 for bswap64Seq — bswap64(ea) is never a load result
    out.push(O.end);
    // single, uniform extension
    if (size === 2 && signed === 's16') out.push(O.i32_wrap_i64, O.i32_extend16_s, O.i64_extend_i32_s);
    else if (size === 2) out.push(O.i32_wrap_i64, O.i32_const(0xffff), O.i32_and, O.i64_extend_i32_u);
    else if (size === 4 && signed === 's32') out.push(O.i32_wrap_i64, O.i64_extend_i32_s);
    else if (size === 4) out.push(O.i32_wrap_i64, O.i64_extend_i32_u);
    out.push(...O.global_set(dest));
    return out;
  },
  _writebackRA(insn) {
    return this._isUpdateForm(insn.name) && insn.RA !== 0;
  },
  _isUpdateForm(name) {
    return name.length > 2 && name.endsWith('u') && (name.startsWith('l') || name.startsWith('st'));
  },
  emitStore(insn, size, srcReg) {
    const L = this.lifter;
    const ea = (insn.RB !== undefined ? this.eaX(insn) : this.eaD(insn));
    const out = [...ea, ...O.local_set(L_T64)];
    if (this._writebackRA(insn)) out.push(...O.local_get(L_T64), ...O.global_set(insn.RA));
    // value into L_T64B
    out.push(...O.global_get(srcReg), ...O.local_set(L_T64B));
    out.push(...this.mmioAddrCheck());
    out.push(...O.if_(T_VOID));
    out.push(...O.local_get(L_T64), ...O.local_get(L_T64B), ...O.i32_const(size), ...O.call(L.imMMIO_STORE));
    out.push(O.else);
    out.push(...O.local_get(L_T64), ...L.addrMem());
    if (size === 1) {
      out.push(...O.local_get(L_T64B), O.i32_wrap_i64, ...O.i32_store8(0, 0));
    } else if (size === 2) {
      out.push(...O.local_get(L_T64B), O.i32_wrap_i64, O.local_set(L_T32), ...this.bswap16Seq(), O.local_get(L_T32), ...O.i32_store16(0, 0));
    } else if (size === 4) {
      out.push(...O.local_get(L_T64B), O.i32_wrap_i64, O.local_set(L_T32), ...this.bswap32Seq(), O.local_get(L_T32), ...O.i32_store(0, 0));
    } else {
      out.push(...O.local_get(L_T64B), O.local_set(L_T64), ...this.bswap64Seq(), O.local_get(L_T64), ...O.i64_store(0, 0));
    }
    out.push(O.end);
    return out;
  },
});

// ---------- instruction handlers ----------

Object.assign(FragmentCompiler.prototype, {
  emitInsn(insn, blockIdx, nextBlockIdx) {
    const L = this.lifter;
    const e = (ops) => ops;
    const RA = insn.RA ?? 0, RB = insn.RB ?? 0, RT = insn.RT ?? 0;
    const SIMM = insn.SIMM ?? 0, UIMM = insn.UIMM ?? 0;
    const sext = (v) => BigInt(v);
    const name = insn.name;

    // ===== control flow =====
    if (name === 'b' || name === 'bl') {
      const t = insn.branchTarget & GUEST_MEM_MASK;
      const inFrag = this.blocksByAddr.has(t);
      const ops = [];
      if (name === 'bl') ops.push(...O.i64_const(BigInt(insn.address + 4)), ...O.global_set(G.LR));
      if (name === 'b' && inFrag) ops.push(...this.setBlkAndDispatch(this.blocksByAddr.get(t), blockIdx));
      else if (name === 'b') ops.push(...this.tailCall(t));
      else {
        // bl — call the callee, then fall into the continuation fragment
        // (every bl is a fragment end; cutTo is addr+4 or null at image end).
        ops.push(...this.callFragment(t));
        if (this.cutTo != null) ops.push(...this.callFragment(this.cutTo), O.return);
        else ops.push(O.return);
      }
      return ops;
    }
    if (insn.isBranch && insn.branchTarget != null) { // conditional bc*
      return this.emitCondBranch(insn, blockIdx, nextBlockIdx);
    }
    if (name === 'blr' || name === 'bclr' || name === 'blrl') {
      return [...this.emitBranchCommon(insn, blockIdx), ...O.local_get(L_T32), ...O.if_(T_VOID), O.return, O.end];
    }
    if (name === 'bctr' || name === 'bctrl') {
      const ops = this.emitBranchCommon(insn, blockIdx);
      ops.push(...O.local_get(L_T32), ...O.if_(T_VOID));
      const LK = insn.LK ? 1 : 0;
      if (LK) ops.push(...O.i64_const(BigInt(insn.address + 4)), ...O.global_set(G.LR));
      // Jump tables (detected statically): compare CTR against the table's
      // labels; on a match, tail-call that case fragment — a `bctr` never
      // falls through. No match → fall back to the runtime icall path (loud
      // on unknown targets), preserving today's behavior exactly.
      const table = !LK && this.lifter.switchTables ? this.lifter.switchTables.get(insn.address & GUEST_MEM_MASK) : null;
      if (table) {
        for (const label of table.labels) {
          ops.push(...O.global_get(G.CTR), ...O.i64_const(BigInt(GUEST_MEM_MASK)), O.i64_and,
                   ...O.i64_const(BigInt(label)), O.i64_eq, O.if_(T_VOID));
          ops.push(...this.tailCall(label));
          ops.push(O.end);
        }
      }
      ops.push(...O.call(L.storeCtxIndex), ...O.global_get(G.CTR), ...O.i64_const(BigInt(GUEST_MEM_MASK)), O.i64_and, ...O.call(L.imICALL));
      if (!insn.LK) ops.push(O.return);
      else if (this.cutTo != null) ops.push(...this.callFragment(this.cutTo), O.return);
      else ops.push(O.return);
      ops.push(O.end);
      // bctrl with the branch NOT taken falls through too
      if (insn.LK) {
        if (nextBlockIdx !== null) ops.push(...this.setBlkAndDispatch(nextBlockIdx, blockIdx + 1));
        else if (this.cutTo != null) ops.push(...this.callFragment(this.cutTo), O.return);
      }
      return ops;
    }
    if (name === 'rfi' || name === 'rfid') return [O.return];
    if (name === 'sc') {
      return [...O.global_get(G.LR), ...O.call(L.imSC)];
    }
    if (name === 'sync' || name === 'lwsync' || name === 'isync' || name === 'eieio' || name === 'nop') return [];

    // ===== D-form immediates / ALU =====
    if (name === 'li' || name === 'addi') {
      return [...this.lifter.raOrZero([RT, RA, SIMM], 1), ...O.i64_const(sext(SIMM)), O.i64_add, ...O.global_set(RT)];
    }
    if (name === 'lis') {
      const v = (SIMM << 16) | 0; // i32 signed
      return [...O.i32_const(v), O.i64_extend_i32_s, ...O.global_set(RT)];
    }
    if (name === 'addis') {
      return [...this.lifter.raOrZero([RT, RA, SIMM], 1), ...O.i64_const(BigInt(SIMM) << 16n), O.i64_add, ...O.global_set(RT)];
    }
    if (name === 'addic' || name === 'addic.') {
      return this.emitAddi32(insn, name.endsWith('.'));
    }
    if (name === 'subfic') return this.emitSubfic(insn);
    if (name === 'mulli') {
      return [...this.lifter.raOrZero([RT, RA, SIMM], 1), ...O.i64_const(sext(SIMM)), O.i64_mul, ...O.global_set(RT)];
    }

    // ===== compares =====
    if (name === 'cmpi' || name === 'cmpli') {
      const is64 = insn.L === 1;
      const isUnsigned = name === 'cmpli';
      const a64 = [...O.global_get(RA)];
      if (is64) {
        const bOps = [O.i64_const(BigInt(UIMM)) /* cmpi 32-bit imm sign-extended; cmpli zero-extended */];
        if (!isUnsigned) bOps[0] = O.i64_const(BigInt(SIMM));
        return this.compareOps(insn.CRFD ?? 0, a64, bOps, isUnsigned, true);
      }
      const a32 = [...a64, O.i32_wrap_i64];
      const b32 = [isUnsigned ? O.i32_const(UIMM) : O.i32_const(SIMM)];
      return this.compareOps(insn.CRFD ?? 0, a32, b32, isUnsigned, false);
    }
    if (name === 'cmp' || name === 'cmpl') {
      const is64 = insn.L === 1;
      const isUnsigned = name === 'cmpl';
      if (is64) return this.compareOps(insn.CRFD ?? 0, [...O.global_get(RA)], [...O.global_get(RB)], isUnsigned, true);
      return this.compareOps(insn.CRFD ?? 0, [...O.global_get(RA), O.i32_wrap_i64], [...O.global_get(RB), O.i32_wrap_i64], isUnsigned, false);
    }

    // ===== logical =====
    if (name === 'or' || name === 'and' || name === 'xor' || name === 'nor' || name === 'andc' || name === 'eqv') {
      return this.emitLogic(insn, name);
    }
    if (name === 'ori' || name === 'oris' || name === 'xori' || name === 'xoris') {
      const shift = (name.endsWith('is')) ? 16 : 0;
      const imm = BigInt(UIMM << shift);
      const op = name.startsWith('or') ? O.i64_or : O.i64_xor;
      return [...O.global_get(RA), ...O.i64_const(imm), op, ...O.global_set(RT)];
    }
    if (name === 'andi.' || name === 'andis.') {
      const shift = name === 'andis.' ? 16 : 0;
      const imm = (UIMM << shift) >>> 0;
      const out = [...O.global_get(RA), ...O.i64_const(BigInt(imm)), O.i64_and, ...O.global_set(RT)];
      // CR0 from low 32 bits
      out.push(...O.global_get(RT), O.i32_wrap_i64, ...O.local_set(L_T32B), ...this.lifter.cr0FromI32Local());
      return out;
    }
    // r0-r0 mr alias handled by 'or' generic path

    // ===== shifts / extends =====
    if (name === 'extsb') return [...O.global_get(RA), O.i64_extend8_s, ...O.global_set(RT)];
    if (name === 'extsh') return [...O.global_get(RA), O.i64_extend16_s, ...O.global_set(RT)];
    if (name === 'extsw') return [...O.global_get(RA), O.i64_extend32_s, ...O.global_set(RT)];
    if (name === 'slwi' || name === 'slw' || name === 'sld') {
      const sh = name === 'slwi' ? [O.i32_const(insn.SH)] : null;
      return this.emitShift(insn, name);
    }
    if (name === 'srwi' || name === 'srw') return this.emitShift(insn, name);
    if (name === 'sraw' || name === 'srawi') return this.emitShift(insn, name);
    if (name === 'sradi') return this.emitShift(insn, name);

    // ===== rotate =====
    if (name === 'rlwinm') return this.emitRlwinm(insn);
    if (name === 'rlwimi') return this.emitRlwimi(insn);
    if (name === 'rlwnm') return this.emitRlwnm(insn);
    if (name === 'rldicl' || name === 'rldicr') return this.emitRldic(insn, name);

    // ===== X-form arithmetic =====
    if (name === 'add') return this.emitAdd64(insn, false, false);
    if (name === 'addc') return this.emitAdd64(insn, true, false);
    if (name === 'adde') return this.emitAdd64(insn, true, true);
    if (name === 'addze') return this.emitAdd64(insn, true, 'ze');
    if (name === 'addme') return this.emitAdd64(insn, true, 'me');
    if (name === 'subf' || name === 'subfc' || name === 'subfe' || name === 'subfze' || name === 'subfme') {
      return this.emitSubf(insn, name);
    }
    if (name === 'mullw') {
      const out = [...O.global_get(RA), O.i32_wrap_i64, ...O.global_get(RB), O.i32_wrap_i64, O.i32_mul, O.i64_extend_i32_s, ...O.global_set(RT)];
      out.push(...this.cr0FromRT(insn, false));
      return out;
    }
    if (name === 'mulld') {
      const out = [...O.global_get(RA), ...O.global_get(RB), O.i64_mul, ...O.global_set(RT)];
      out.push(...this.cr0FromRT(insn, true));
      return out;
    }
    if (name === 'mulhw' || name === 'mulhwu' || name === 'mulhd' || name === 'mulhdu') {
      return this.emitMulh(insn, name);
    }
    if (name === 'divw' || name === 'divwu' || name === 'divd' || name === 'divdu') return this.emitDiv(insn, name);

    // ===== SPRs =====
    if (name === 'mtspr') {
      const spr = insn.SPR;
      if (spr === 8) return [...O.global_get(RT), O.global_set(G.LR)];
      if (spr === 9) return [...O.global_get(RT), O.global_set(G.CTR)];
      if (spr === 1) return [...O.global_get(RT), O.i32_wrap_i64, O.global_set(G.XER)];
      return this.unimplInsn(insn);
    }
    if (name === 'mfspr') {
      const spr = insn.SPR;
      if (spr === 8) return [...O.global_get(G.LR), O.global_set(RT)];
      if (spr === 9) return [...O.global_get(G.CTR), O.global_set(RT)];
      if (spr === 1) return [...O.global_get(G.XER), O.i64_extend_i32_u, O.global_set(RT)];
      return this.unimplInsn(insn);
    }

    // ===== loads =====
    if (name === 'lbz') return this.emitLoad(insn, 'none', 1, RT);
    if (name === 'lhz') return this.emitLoad(insn, 'none', 2, RT);
    if (name === 'lha') return this.emitLoad(insn, 's16', 2, RT);
    if (name === 'lwz' || name === 'lwzu') return this.emitLoad(insn, 'none', 4, RT);
    if (name === 'lwa') return this.emitLoad(insn, 's32', 4, RT);
    if (name === 'ld' || name === 'ldu') return this.emitLoad(insn, 'none', 8, RT);
    if (name === 'lwzx' || name === 'lbzx' || name === 'lhzx' || name === 'lwax' || name === 'ldx') {
      const size = { lwzx: 4, lbzx: 1, lhzx: 2, lwax: 4, ldx: 8 }[name];
      const signed = name === 'lwax' ? 's32' : 'none';
      return this.emitLoad(insn, signed, size, RT);
    }
    // ===== stores =====
    if (name === 'stb') return this.emitStore(insn, 1, RT);
    if (name === 'sth') return this.emitStore(insn, 2, RT);
    if (name === 'stw' || name === 'stwu') return this.emitStore(insn, 4, RT);
    if (name === 'std' || name === 'stdu') return this.emitStore(insn, 8, RT);
    if (name === 'stbx' || name === 'sthx' || name === 'stwx' || name === 'stdx') {
      const size = { stbx: 1, sthx: 2, stwx: 4, stdx: 8 }[name];
      return this.emitStore(insn, size, RT);
    }

    // ===== FP =====
    if (name === 'lfd' || name === 'lfs') return this.emitFpLoad(insn, name === 'lfd' ? 8 : 4);
    if (name === 'stfd' || name === 'stfs') return this.emitFpStore(insn, name === 'stfd' ? 8 : 4);
    if (name === 'fmr') return [...O.global_get(32 + insn.FRA), ...O.global_set(32 + insn.FRT)];
    if (name === 'fneg') return [...O.global_get(32 + insn.FRA), O.f64_neg, ...O.global_set(32 + insn.FRT)];
    if (name === 'fabs') return [...O.global_get(32 + insn.FRA), O.f64_abs, ...O.global_set(32 + insn.FRT)];
    if (name === 'frsp') return [...O.global_get(32 + insn.FRA), O.f32_demote_f64, O.f64_promote_f32, ...O.global_set(32 + insn.FRT)];
    if (name === 'fadd' || name === 'fadds') return this.emitFpBin(insn, O.f64_add);
    if (name === 'fsub' || name === 'fsubs') return this.emitFpBin(insn, O.f64_sub);
    if (name === 'fmul' || name === 'fmuls') return this.emitFpBin(insn, O.f64_mul);
    if (name === 'fdiv' || name === 'fdivs') return this.emitFpBin(insn, O.f64_div);
    if (name === 'fmadd' || name === 'fmadds') {
      return [...O.global_get(32 + insn.FRA), ...O.global_get(32 + (insn.FRC ?? 0)), O.f64_mul,
              ...O.global_get(32 + insn.FRB), O.f64_add, ...O.global_set(32 + insn.FRT)];
    }
    if (name === 'fcmpu' || name === 'fcmpo') return this.emitFpCmp(insn);

    return this.unimplInsn(insn);
  },

  unimplInsn(insn) {
    this.lifter.stats.unimpl++;
    return [...O.i64_const(BigInt(insn.address)), ...O.i32_const(insn.word | 0), ...O.call(this.lifter.imUNIMPL), O.unreachable];
  },

  compareOps(field, aOps, bOps, isUnsigned, is64) {
    return this.lifter.crSetCompare(field, aOps, bOps, isUnsigned, is64);
  },
});

// ---------- remaining instruction helpers ----------

Object.assign(FragmentCompiler.prototype, {
  /**
   * Unified 64-bit add-with-carry chain matching PowerISA semantics:
   *   t1 = a + b (wraps); c1 = t1 <u a; sum = t1 + cin; c2 = sum <u t1; ca = c1|c2
   * @param aOps operand A (i64 on stack producer) @param bOps operand B producer
   * @param cinOps 'one' | 'xer' | 'zero' @param setCA whether to write XER.CA
   */
  emitAddCarry(rt, aOps, bOps, cin, setCA, withRc) {
    const L = this.lifter;
    const cinOps = cin === 'one'
      ? [O.i64_const(1n)]
      : cin === 'xer'
        ? [...O.global_get(G.XER), ...O.i32_const(29), O.i32_shr_u, ...O.i32_const(1), O.i32_and, O.i64_extend_i32_u]
        : [O.i64_const(0n)];
    const out = [
      ...aOps, ...bOps, O.i64_add, ...O.local_set(L_T64B),   // t1
      ...O.local_get(L_T64B), ...aOps, O.i64_lt_u, ...O.local_set(L_T32),  // c1 = t1 <u a
      ...O.local_get(L_T64B), ...cinOps, O.i64_add, ...O.local_set(L_T64),          // sum
      ...O.local_get(L_T64), ...O.local_get(L_T64B), O.i64_lt_u, ...O.local_set(L_T32B), // c2
    ];
    if (setCA) {
      out.push(
        ...O.global_get(G.XER), O.i32_const(~0x20000000 >>> 0), O.i32_and,
        ...O.local_get(L_T32), ...O.local_get(L_T32B), O.i32_or,
        O.i32_const(29), O.i32_shl, O.i32_or, ...O.global_set(G.XER),
      );
    }
    out.push(...O.local_get(L_T64), ...O.global_set(rt));
    if (withRc) out.push(...this.lifter.cr0FromI64Local());
    return out;
  },

  /** X-form add family: add (no CA), addc, adde (cin=XER), addze (b=0), addme (b=-1) */
  emitAdd64(insn, setCA, mode) {
    const RT = insn.RT, RA = insn.RA, RB = insn.RB;
    let aOps, bOps, cin;
    if (mode === 'ze') { aOps = RA === 0 ? [O.i64_const(0n)] : [O.global_get(RA)]; bOps = [O.i64_const(0n)]; cin = 'xer'; }
    else if (mode === 'me') { aOps = RA === 0 ? [O.i64_const(0n)] : [O.global_get(RA)]; bOps = [O.i64_const(-1n)]; cin = 'xer'; }
    else { aOps = RA === 0 ? [O.i64_const(0n)] : [O.global_get(RA)]; bOps = [O.global_get(RB)]; cin = mode === true ? 'xer' : 'zero'; }
    const withRc = !!insn.Rc;
    return this.emitAddCarry(RT, aOps, bOps, cin, setCA, withRc);
  },

  /** subf/subfc/subfe/subfme/subfze: r = b + ~a + (subf|subfc ? 1 : XER.CA) */
  emitSubf(insn, name) {
    const RT = insn.RT, RA = insn.RA, RB = insn.RB;
    const isBase = name === 'subf' || name === 'subfc';
    const setCA = name !== 'subf'; // subf (plain) does not write CA
    const aOps = RA === 0 ? [O.i64_const(0n)] : [O.global_get(RA)];
    const aInv = [...aOps, ...O.i64_const(-1n), O.i64_xor];
    let bOps, cin;
    if (name === 'subfme') { bOps = [O.i64_const(-1n)]; cin = 'xer'; }
    else if (name === 'subfze') { bOps = [O.i64_const(0n)]; cin = 'xer'; }
    else { bOps = [O.global_get(RB)]; cin = isBase ? 'one' : 'xer'; }
    return this.emitAddCarry(RT, bOps, aInv, cin, setCA, !!insn.Rc);
  },

  emitAddi32(insn, withRc) {
    const RT = insn.RT, RA = insn.RA;
    const SIMM = BigInt(insn.SIMM) & 0xffffffffn;
    const raOps = RA === 0 ? [O.i64_const(0n)] : [O.global_get(RA)];
    const out = [
      ...raOps, O.i32_wrap_i64, O.i64_extend_i32_u,
      ...O.i64_const(SIMM), O.i64_add, O.i32_wrap_i64, ...O.local_set(L_T32B),
      O.local_get(L_T32B), O.i64_extend_i32_s, ...O.global_set(RT),
      // CA = sum > 0xFFFFFFFF (recompute u64 sum)
      ...raOps, O.i32_wrap_i64, O.i64_extend_i32_u,
      ...O.i64_const(SIMM), O.i64_add, ...O.local_set(L_T64),
      ...O.global_get(G.XER), O.i32_const(~0x20000000 >>> 0), O.i32_and,
      ...O.local_get(L_T64), O.i64_const(0xffffffffn), O.i64_gt_u,
      O.i32_const(29), O.i32_shl, O.i32_or, ...O.global_set(G.XER),
    ];
    if (withRc) out.push(...this.lifter.cr0FromI32Local());
    return out;
  },

  emitSubfic(insn) {
    const RT = insn.RT, RA = insn.RA;
    const SIMM = BigInt(insn.SIMM) & 0xffffffffn;
    const raOps = RA === 0 ? [O.i64_const(0n)] : [O.global_get(RA)];
    const out = [
      ...O.i64_const(SIMM),
      ...raOps, O.i32_wrap_i64, O.i64_extend_i32_u, O.i64_const(0xffffffffn), O.i64_xor, O.i64_add,
      ...O.i64_const(1n), O.i64_add, ...O.local_set(L_T64),
      O.local_get(L_T64), O.i64_extend_i32_s, ...O.global_set(RT),
      ...O.global_get(G.XER), O.i32_const(~0x20000000 >>> 0), O.i32_and,
      ...O.local_get(L_T64), O.i64_const(0xffffffffn), O.i64_gt_u,
      O.i32_const(29), O.i32_shl, O.i32_or, ...O.global_set(G.XER),
    ];
    return out;
  },

  /** CR0 from the RT register just written (after global_set(RT)). */
  cr0FromRT(insn, is64) {
    if (!insn.Rc) return [];
    if (is64) return [...O.global_get(insn.RT), ...O.local_set(L_T64B), ...this.lifter.cr0FromI64Local()];
    return [...O.global_get(insn.RT), O.i32_wrap_i64, ...O.local_set(L_T32B), ...this.lifter.cr0FromI32Local()];
  },

  emitLogic(insn, name) {
    const RT = insn.RT, RA = insn.RA, RB = insn.RB;
    const out = [...O.global_get(RA)];
    switch (name) {
      case 'and': out.push(...O.global_get(RB), O.i64_and); break;
      case 'or': out.push(...O.global_get(RB), O.i64_or); break;
      case 'xor': out.push(...O.global_get(RB), O.i64_xor); break;
      case 'nor': out.push(...O.global_get(RB), O.i64_or, ...O.i64_const(-1n), O.i64_xor); break;
      case 'andc': out.push(...O.global_get(RB), ...O.i64_const(-1n), O.i64_xor, O.i64_and); break;
      case 'eqv': out.push(...O.global_get(RB), O.i64_xor, ...O.i64_const(-1n), O.i64_xor); break;
    }
    out.push(...O.global_set(RT));
    if (insn.Rc) { out.push(...O.global_get(RT), ...O.local_set(L_T64B), ...this.lifter.cr0FromI64Local()); }
    return out;
  },

  emitShift(insn, name) {
    const RT = insn.RT, RA = insn.RA, RB = insn.RB;
    let out;
    switch (name) {
      case 'slwi': // M-form: dest is the rA field (decoder RA), source the rS field
        out = [...O.global_get(insn.RS ?? RA), O.i32_wrap_i64, ...O.i32_const(insn.SH), O.i32_shl, O.i64_extend_i32_u, ...O.global_set(RA)]; break;
      case 'slw':
        out = [...O.global_get(RA), O.i32_wrap_i64, ...O.global_get(RB), O.i32_wrap_i64, ...O.i32_const(31), O.i32_and, O.i32_shl, O.i64_extend_i32_u, ...O.global_set(RT)]; break;
      case 'sld':
        out = [...O.global_get(RA), ...O.global_get(RB), ...O.i64_const(63n), O.i64_and, O.i64_shl, ...O.global_set(RT)]; break;
      case 'srwi': // M-form: dest is the rA field, source the rS field
        out = [...O.global_get(insn.RS ?? RA), O.i32_wrap_i64, ...O.i32_const(insn.SH), O.i32_shr_u, O.i64_extend_i32_u, ...O.global_set(RA)]; break;
      case 'srw':
        out = [...O.global_get(RA), O.i32_wrap_i64, ...O.global_get(RB), O.i32_wrap_i64, ...O.i32_const(31), O.i32_and, O.i32_shr_u, O.i64_extend_i32_u, ...O.global_set(RT)]; break;
      case 'sraw':
        out = [...O.global_get(RA), O.i32_wrap_i64, ...O.global_get(RB), O.i32_wrap_i64, ...O.i32_const(31), O.i32_and, O.i32_shr_s, O.i64_extend_i32_s, ...O.global_set(RT)]; break;
      case 'srawi':
        out = [...O.global_get(RA), O.i32_wrap_i64, ...O.i32_const(insn.RB & 31), O.i32_shr_s, O.i64_extend_i32_s, ...O.global_set(RT)]; break;
      case 'sradi': { // MD-form: dest is the rA field, source the rS field
        const sh = ((insn.word & 1) ? 32 : 0) | insn.RB; // bit 0 = sh32
        out = [...O.global_get(insn.RS ?? RT), ...O.i64_const(BigInt(sh)), O.i64_shr_s, ...O.global_set(RA)]; break;
      }
      default: out = [];
    }
    const is64 = name === 'sld' || name === 'sradi';
    const mForm = name === 'slwi' || name === 'srwi' || name === 'sradi';
    out = [...out, ...this.cr0FromRT(mForm ? { ...insn, RT: insn.RA } : insn, is64)];
    return out;
  },

  emitRlwinm(insn) {
    const RA = insn.RA, RS = insn.RS ?? insn.RT; // M-form: rA = rotate(rS) & mask
    const SH = insn.SH, MB = insn.MB, ME = insn.ME;
    let mask;
    if (MB <= ME) mask = ((((1 << (ME - MB + 1)) - 1) << MB) >>> 0);
    else mask = (~(((1 << (MB - ME - 1)) - 1) << (ME + 1))) >>> 0;
    const out = [
      ...O.global_get(RS), O.i32_wrap_i64, ...O.i32_const(SH), O.i32_rotl,
      ...O.i32_const(mask), O.i32_and, O.i64_extend_i32_u, ...O.global_set(RA),
    ];
    if (insn.Rc) out.push(...O.global_get(RA), O.i32_wrap_i64, ...O.local_set(L_T32B), ...this.lifter.cr0FromI32Local());
    return out;
  },

  emitRlwimi(insn) {
    const RA = insn.RA, RS = insn.RS ?? insn.RT; // M-form: rA = (rA & ~mask) | (rotate(rS) & mask)
    const SH = insn.SH, MB = insn.MB, ME = insn.ME;
    let mask;
    if (MB <= ME) mask = ((((1 << (ME - MB + 1)) - 1) << MB) >>> 0);
    else mask = (~(((1 << (MB - ME - 1)) - 1) << (ME + 1))) >>> 0;
    const keep = (~mask) >>> 0;
    const out = [
      ...O.global_get(RA), O.i32_wrap_i64, ...O.i32_const(keep), O.i32_and,
      ...O.global_get(RS), O.i32_wrap_i64, ...O.i32_const(SH), O.i32_rotl, ...O.i32_const(mask), O.i32_and,
      O.i32_or, O.i64_extend_i32_u, ...O.global_set(RA),
    ];
    if (insn.Rc) out.push(...O.global_get(RA), O.i32_wrap_i64, ...O.local_set(L_T32B), ...this.lifter.cr0FromI32Local());
    return out;
  },

  emitRlwnm(insn) {
    const RA = insn.RA, RS = insn.RS ?? insn.RT, RB = insn.RB; // M-form: rA = rotate(rS, rB) & mask
    const MB = insn.MB, ME = insn.ME;
    let mask;
    if (MB <= ME) mask = ((((1 << (ME - MB + 1)) - 1) << MB) >>> 0);
    else mask = (~(((1 << (MB - ME - 1)) - 1) << (ME + 1))) >>> 0;
    const out = [
      ...O.global_get(RS), O.i32_wrap_i64,
      ...O.global_get(RB), O.i32_wrap_i64, ...O.i32_const(31), O.i32_and, O.i32_rotl,
      ...O.i32_const(mask), O.i32_and, O.i64_extend_i32_u, ...O.global_set(RA),
    ];
    out.push(...this.cr0FromRT({ ...insn, RT: RA }, false));
    return out;
  },

  emitRldic(insn, name) {
    const RA = insn.RA, RS = insn.RS ?? insn.RT; // MD-form: rA = rotate(rS) & mask
    const SH = BigInt(insn.SH);
    const MB = BigInt(insn.MB);
    let mask;
    if (name === 'rldicl') mask = MB === 0n ? 0xffffffffffffffffn : ((~0n) << (64n - MB)) & 0xffffffffffffffffn;
    else mask = ((1n << (MB + 1n)) - 1n) & 0xffffffffffffffffn; // rldicr
    const out = [
      ...O.global_get(RS), ...O.i64_const(SH), O.i64_rotl,
      ...O.i64_const(mask), O.i64_and, ...O.global_set(RA),
    ];
    out.push(...this.cr0FromRT({ ...insn, RT: RA }, true));
    return out;
  },

  emitMulh(insn, name) {
    const RT = insn.RT, RA = insn.RA, RB = insn.RB;
    if (name === 'mulhd' || name === 'mulhdu') {
      // 128-bit product high half — needs splitting; not supported yet (loud at compile time).
      this.lifter.stats.unimpl++;
      return [...O.i64_const(BigInt(insn.address)), ...O.i32_const(-3), ...O.call(this.lifter.imUNIMPL), O.unreachable];
    }
    const signed = name === 'mulhw';
    const ext = signed ? O.i64_extend_i32_s : O.i64_extend_i32_u;
    const out = [
      ...O.global_get(RA), O.i32_wrap_i64, ext,
      ...O.global_get(RB), O.i32_wrap_i64, ext, O.i64_mul,
      ...O.i64_const(32n), O.i64_shr_s, O.i32_wrap_i64,
      ...(signed ? [O.i64_extend_i32_s] : [O.i64_extend_i32_u]),
      ...O.global_set(RT),
    ];
    out.push(...this.cr0FromRT(insn, false));
    return out;
  },

  emitDiv(insn, name) {
    const RT = insn.RT, RA = insn.RA, RB = insn.RB;
    const is64 = name === 'divd' || name === 'divdu';
    const signed = name === 'divw' || name === 'divd';
    const wrap = (ops) => is64 ? ops : [...ops, O.i32_wrap_i64];
    const out = [];
    // if (b == 0) r = 0 else if (signed && a == MIN && b == -1) r = a else r = a / b
    const blk = is64 ? 0x7e : 0x7f;
    out.push(...wrap([...O.global_get(RB)]), ...(is64 ? [O.i64_eqz] : [O.i32_eqz]), ...O.if_(blk));
    out.push(...(is64 ? [O.i64_const(0n)] : [O.i32_const(0)]));
    out.push(O.else);
    if (signed) {
      // overflow case
      const min = is64 ? O.i64_const(-(1n << 63n)) : O.i32_const(-2147483648);
      const one = is64 ? O.i64_const(-1n) : O.i32_const(-1);
      out.push(...wrap([...O.global_get(RA)]), ...min, is64 ? O.i64_eq : O.i32_eq);
      out.push(...wrap([...O.global_get(RB)]), ...one, is64 ? O.i64_eq : O.i32_eq);
      out.push(O.i32_and, ...O.if_(blk));
      out.push(...wrap([...O.global_get(RA)]));
      out.push(O.else);
      out.push(...wrap([...O.global_get(RA)]), ...wrap([...O.global_get(RB)]), ...(is64 ? (signed ? [O.i64_div_s] : [O.i64_div_u]) : (signed ? [O.i32_div_s] : [O.i32_div_u])));
      out.push(O.end);
    } else {
      out.push(...wrap([...O.global_get(RA)]), ...wrap([...O.global_get(RB)]), ...(is64 ? [O.i64_div_u] : [O.i32_div_u]));
    }
    out.push(O.end);
    // promote to i64 GPR
    if (!is64) out.push(...(signed ? [O.i64_extend_i32_s] : [O.i64_extend_i32_u]));
    out.push(...O.global_set(RT));
    out.push(...this.cr0FromRT(insn, is64));
    return out;
  },

  emitFpBin(insn, op) {
    return [...O.global_get(32 + insn.FRA), ...O.global_get(32 + insn.FRB), op, ...O.global_set(32 + insn.FRT)];
  },

  emitFpCmp(insn) {
    // FL-form: bits 21-25 = BF<<2 | 00 — the decoder exposes them as FRT.
    const BF = ((insn.FRT ?? 0) >>> 2) & 7;
    const FRA = insn.FRA, FRB = insn.FRB;
    return [
      ...O.global_get(32 + FRA), ...O.global_get(32 + FRB), O.f64_lt, O.i32_const(3), O.i32_shl,
      ...O.global_get(32 + FRA), ...O.global_get(32 + FRB), O.f64_gt, O.i32_const(2), O.i32_shl, O.i32_or,
      ...O.global_get(32 + FRA), ...O.global_get(32 + FRB), O.f64_eq, O.i32_const(1), O.i32_shl, O.i32_or,
      // so = unordered
      ...O.global_get(32 + FRA), ...O.global_get(32 + FRA), O.f64_ne,
      ...O.global_get(32 + FRB), ...O.global_get(32 + FRB), O.f64_ne, O.i32_or, O.i32_or,
      ...this.crStoreFieldBits(BF),
    ];
  },

  emitFpLoad(insn, size) {
    const FRET = insn.FRT ?? insn.RT;
    const ea = (insn.RB !== undefined ? this.eaX(insn) : this.eaD(insn));
    const out = [...ea, ...O.local_set(L_T64)];
    out.push(...this.mmioAddrCheck(), ...O.if_(0x7e));
    out.push(...O.local_get(L_T64), ...O.i32_const(size), ...O.call(this.lifter.imMMIO_LOAD));
    out.push(O.else);
    out.push(...O.local_get(L_T64), ...this.lifter.addrMem());
    if (size === 8) {
      out.push(...O.i64_load(0, 0), ...O.local_set(L_T64), ...this.bswap64Seq(), O.local_get(L_T64)); // value in L_T64 for bswap64Seq
    } else {
      out.push(...O.i32_load(0, 0), ...O.local_set(L_T32), ...this.bswap32Seq(), O.local_get(L_T32), O.i64_extend_i32_u);
    }
    out.push(O.end);
    if (size === 8) out.push(O.f64_reinterpret_i64);
    else { out.push(O.i32_wrap_i64, O.f32_reinterpret_i32, O.f64_promote_f32); }
    out.push(...O.global_set(32 + FRET));
    return out;
  },

  emitFpStore(insn, size) {
    const FRS = insn.FRT ?? insn.RT;
    const ea = (insn.RB !== undefined ? this.eaX(insn) : this.eaD(insn));
    const out = [...ea, ...O.local_set(L_T64)];
    if (size === 8) {
      out.push(...O.global_get(32 + FRS), O.i64_reinterpret_f64, ...O.local_set(L_T64B));
    } else {
      out.push(...O.global_get(32 + FRS), O.f32_demote_f64, O.i32_reinterpret_f32, ...O.local_set(L_T32), ...this.bswap32Seq(), O.local_get(L_T32), O.i64_extend_i32_u, ...O.local_set(L_T64B));
    }
    out.push(...this.mmioAddrCheck(), ...O.if_(T_VOID));
    out.push(...O.local_get(L_T64), ...O.local_get(L_T64B), ...O.i32_const(size), ...O.call(this.lifter.imMMIO_STORE));
    out.push(O.else);
    out.push(...O.local_get(L_T64), ...this.lifter.addrMem());
    if (size === 8) {
      out.push(...O.local_get(L_T64B), ...O.local_set(L_T64), ...this.bswap64Seq(), O.local_get(L_T64), ...O.i64_store(0, 0));
    } else {
      out.push(...O.local_get(L_T64B), O.i32_wrap_i64, ...O.i32_store(0, 0));
    }
    out.push(O.end);
    return out;
  },
});

// ---------- module assembly & host harness ----------

export function recompileImage(arena, entry, opts = {}) {
  const vmxStubs = opts.vmxStubs ?? null;
  const lifter = new Lifter(discoverFragments(arena, entry, opts.hleCalls, opts.functionStarts, vmxStubs), { switchTables: opts.switchTables ?? null }, opts.hleCalls);
  const L = lifter;
  // Pre-assign wasm function indices: imports 0..6, store_ctx at 7, then one per fragment (sorted).
  const sorted = [...L.fragments.keys()].sort((a, b) => a - b);
  const fnIndex = new Map();
  const localBase = L.b.imports.length + 1; // store_ctx helper occupies the first local slot
  sorted.forEach((addr, i) => fnIndex.set(addr, localBase + i));

  // Cut addresses: next fragment's start (or null) for fallthrough calls.
  const cutOf = new Map();
  for (const addr of sorted) cutOf.set(addr, L.fragments.get(addr).cutTo ?? null);

  // Two passes: create all function entries, then bodies.
  const funcIdx = new Map();
  for (const addr of sorted) {
    funcIdx.set(addr, L.b.addFunction(L.tyNone, [[1, 0x7f] /* blk */, [1, 0x7e] /* t64 */, [1, 0x7e] /* t64b */, [1, 0x7f] /* t32 */, [1, 0x7f] /* t32b */], []));
  }
  const codeBodies = [];
  for (const addr of sorted) {
    const frag = L.fragments.get(addr);
    L.stats.fragments = sorted.length;
    if (frag.vmxStub) {
      L.stats.vmxStubs++;
      codeBodies.push({ addr, ops: emitVmxStubOps(L, frag.vmxStub, O) });
      continue;
    }
    const fc = new FragmentCompiler(L, frag, cutOf.get(addr), fnIndex);
    codeBodies.push({ addr, ops: fc.emit(L.b) });
  }
  // Splice bodies in creation order (functions were created in sorted order),
  // with the store_ctx helper body first (it was registered as the first local function).
  codeBodies.sort((a, b) => fnIndex.get(a.addr) - fnIndex.get(b.addr));
  L.b.funcBodies = [[...buildStoreCtxOps(), O.end]];
  for (const { ops } of codeBodies) L.b.funcBodies.push([...flattenOps(ops), O.end]);

  // Exports: memory + one fn per fragment + the r3 GPR (the HLE host writes
  // return values back into it — JS cannot otherwise reach wasm globals).
  L.b.addExport('memory', 2, 0);
  L.b.addExport('r3', 3, 3); // export kind 3 = global, index 3 = r3
  for (const addr of sorted) {
    L.b.addExport('fn_' + addr.toString(16), 0, fnIndex.get(addr));
  }

  const wasm = L.b.build();
  return { wasm, entry, stats: L.stats, functionStarts: sorted };
}

export class RecompExit extends Error {
  constructor(code) { super('guest exited with code ' + code); this.code = code; }
}
export class RecompUnimpl extends Error {
  constructor(addr, word) { super(`unimplemented instruction 0x${word.toString(16).padStart(8, '0')} at 0x${addr.toString(16)}`); this.addr = addr; }
}

export function makeImports({ onConsoleByte, onSc, onLog } = {}) {
  let consoleBuf = '';
  const createHleHost = () => globalThis.XenonHLE.createHleHost({
    onConsoleByte: onConsoleByte ?? ((ch) => {
      consoleBuf += String.fromCharCode(typeof ch === 'string' ? ch.charCodeAt(0) : ch);
    }),
    onLog,
  });
  const imports = {
    xenorecomp: {
      putc(ch) {
        if (onConsoleByte) onConsoleByte(ch);
        else consoleBuf += String.fromCharCode(ch);
      },
      exit(code) { throw new RecompExit(code | 0); },
      unimpl(addr, word) { throw new RecompUnimpl(Number(BigInt(addr) & 0x03ffffffn), word | 0); },
      icall(addr) {
        const target = Number(BigInt(addr) & 0x03ffffffn);
        const hle = imports.__hleCalls && imports.__hleCalls.get(target);
        if (hle) return imports.xenorecomp.hcall(hle.moduleId, hle.ordinal);
        const fn = imports.__exports && imports.__exports['fn_' + target.toString(16)];
        if (!fn) throw new RecompUnimpl(target, -2);
        fn();
        return imports.__exports && imports.__exports.r3 ? imports.__exports.r3.value : 0n;
      },
      hcall(moduleId, ordinal) {
        const host = (imports.__hleHost ??= createHleHost());
        const result = host.call(String(moduleId), Number(ordinal) & 0xffffffff);
        // propagate the return value into the guest's r3 (wasm globals are not
        // otherwise reachable from JS — the module exports r3)
        const exp = imports.__exports;
        if (exp && exp.r3) exp.r3.value = result;
        return result;
      },
      mmio_store(addr, value, size) {
        // Echo big-endian bytes before the exit door; the exit door halts the guest.
        const a = Number(BigInt(addr) & 0x03ffffffn);
        const sizeBytes = size | 0;
        const v = BigInt(value);
        const bytes = [];
        for (let i = 0; i < sizeBytes; i++) bytes.push(Number((v >> BigInt((sizeBytes - 1 - i) * 8)) & 0xffn));
        let echoLen = sizeBytes;
        if (a <= 0x00f00ffc && 0x00f00ffc < a + sizeBytes) echoLen = 0x00f00ffc - a;
        for (let i = 0; i < echoLen; i++) imports.xenorecomp.putc(bytes[i]);
        if (a <= 0x00f00ffc && 0x00f00ffc < a + sizeBytes) throw new RecompExit(0);
      },
      mmio_load(addr, size) { return 0n; },
      sc(addr) { if (onSc) onSc(Number(BigInt(addr) & 0x03ffffffn)); },
    },
  };
  imports.__consoleBuf = () => consoleBuf;
  imports.__attachMemory = (memory) => {
    // attach before any call happens; the HLE host snapshots GPRs per call
    const host = (imports.__hleHost ??= createHleHost());
    host.attachMemory(memory);
  };
  imports.__setHleCalls = (m) => { imports.__hleCalls = m; };
  return imports;
}

/**
 * @param {{arena: Uint8Array, entry: number, hleCalls?: Map, functionStarts?: number[], switchTables?: Map, vmxStubs?: Map}} built
 */
export async function runImage({ arena, entry, hleCalls, functionStarts, switchTables, vmxStubs }, { onConsole, onSc, onLog } = {}) {
  entry = entry & GUEST_MEM_MASK;
  const { wasm, stats } = recompileImage(arena, entry, { hleCalls, functionStarts, switchTables, vmxStubs });
  const imports = makeImports({ onConsoleByte: onConsole ? (b) => onConsole(String.fromCharCode(typeof b === 'string' ? b.charCodeAt(0) : b)) : undefined, onSc, onLog });
  if (hleCalls) imports.__setHleCalls(hleCalls);
  const { instance } = await WebAssembly.instantiate(wasm, imports);
  imports.__exports = instance.exports;
  imports.__attachMemory(instance.exports.memory);
  // Load the arena into wasm memory
  new Uint8Array(instance.exports.memory.buffer).set(arena);
  const t0 = performance.now();
  let exitCode = null;
  try {
    instance.exports['fn_' + entry.toString(16)]();
    exitCode = 0;
  } catch (e) {
    if (e instanceof RecompExit) exitCode = e.code;
    else throw e;
  }
  const elapsed = performance.now() - t0;
  return { exitCode, stats, elapsedMs: elapsed, exports: instance.exports, memory: instance.exports.memory };
}
