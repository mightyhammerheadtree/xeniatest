/**
 * ppc_decode.js — PowerPC64 (big-endian) decoder for the web static
 * recompiler. Dependency-free. X-form/A-form opcode matching uses xenia's
 * masked base-opcode constants (BSD-3, xenia-canary), which this repo already
 * mirrors in its WASM interpreter.
 *
 * Returns { name, address, word, ...fields }. Unknown instructions get
 * { name: 'op<primary>' } and are rejected loudly by the lifter.
 */

const X31 = [
  [0x54000000, 0xfc0007fe, 'rlwinm'],
  [0x5c000000, 0xfc0007fe, 'rlwnm'],
  [0x78000000, 0xfc00007e, 'rldicl'],
  [0x78000004, 0xfc00007e, 'rldicr'],
  [0x78000008, 0xfc00007e, 'rldic'],
  [0x7800000c, 0xfc00007e, 'rldimi'],
  [0x78000010, 0xfc00007e, 'rldcl'],
  [0x78000012, 0xfc00007e, 'rldcr'],
  [0x7c000000, 0xfc000bfe, 'cmp'],
  [0x7c0004ac, 0xffffffff, 'sync'],
  [0x7c2004ac, 0xffffffff, 'lwsync'],
  [0x7c0006ac, 0xffffffff, 'eieio'],
  [0x7c000010, 0xfc0007fe, 'subfc'],
  [0x7c000014, 0xfc0007fe, 'addc'],
  [0x7c000012, 0xfc0007fe, 'mulhd'],
  [0x7c000016, 0xfc0007fe, 'mulhwu'],
  [0x7c000028, 0xfc0007fe, 'lwarx'],
  [0x7c00002a, 0xfc0007fe, 'ldx'],
  [0x7c00002e, 0xfc0007fe, 'lwzx'],
  [0x7c000030, 0xfc0007fe, 'slw'],
  [0x7c000036, 0xfc0007fe, 'sld'],
  [0x7c000038, 0xfc0007fe, 'and'],
  [0x7c000040, 0xfc000bfe, 'cmpl'],
  [0x7c000050, 0xfc0007fe, 'subf'],
  [0x7c000078, 0xfc0007fe, 'andc'],
  [0x7c000092, 0xfc0007fe, 'mulhd'],
  [0x7c000096, 0xfc0007fe, 'mulhw'],
  [0x7c0000ae, 0xfc0007fe, 'lbzx'],
  [0x7c0000f8, 0xfc0007fe, 'nor'],
  [0x7c000110, 0xfc0007fe, 'subfe'],
  [0x7c000114, 0xfc0007fe, 'adde'],
  [0x7c00012a, 0xfc0007fe, 'stdx'],
  [0x7c00012e, 0xfc0007fe, 'stwx'],
  [0x7c000190, 0xfc0007fe, 'subfze'],
  [0x7c000194, 0xfc0007fe, 'addze'],
  [0x7c0001ae, 0xfc0007fe, 'stbx'],
  [0x7c0001d0, 0xfc0007fe, 'subfme'],
  [0x7c0001d2, 0xfc0007fe, 'mulld'],
  [0x7c0001d4, 0xfc0007fe, 'addme'],
  [0x7c0001d6, 0xfc0007fe, 'mullw'],
  [0x7c000214, 0xfc0007fe, 'add'],
  [0x7c00022e, 0xfc0007fe, 'lhzx'],
  [0x7c000238, 0xfc0007fe, 'eqv'],
  [0x7c000278, 0xfc0007fe, 'xor'],
  [0x7c0002a6, 0xfc0007fe, 'mfspr'],
  [0x7c0002aa, 0xfc0007fe, 'lwax'],
  [0x7c00032e, 0xfc0007fe, 'sthx'],
  [0x7c000378, 0xfc0007fe, 'or'],
  [0x7c000392, 0xfc0007fe, 'divdu'],
  [0x7c000396, 0xfc0007fe, 'divwu'],
  [0x7c0003a6, 0xfc0007fe, 'mtspr'],
  [0x7c0003d2, 0xfc0007fe, 'divd'],
  [0x7c0003d6, 0xfc0007fe, 'divw'],
  [0x7c000430, 0xfc0007fe, 'srw'],
  [0x7c000630, 0xfc0007fe, 'sraw'],
  [0x7c000670, 0xfc0007fe, 'srawi'],
  [0x7c000674, 0xfc0007fe, 'sradi'],
  [0x7c000734, 0xfc0007fe, 'extsh'],
  [0x7c000774, 0xfc0007fe, 'extsb'],
  [0x7c0007b4, 0xfc0007fe, 'extsw'],
];

const FP = [
  [0xec000024, 0xfc0007fe, 'fdivs'],
  [0xec000028, 0xfc0007fe, 'fsubs'],
  [0xec00002a, 0xfc0007fe, 'fadds'],
  [0xec000032, 0xfc0007fe, 'fmuls'],
  [0xec000038, 0xfc0007fe, 'fmsubs'],
  [0xec00003a, 0xfc0007fe, 'fmadds'],
  [0xec00003c, 0xfc0007fe, 'fnmsubs'],
  [0xec00003e, 0xfc0007fe, 'fnmadds'],
  [0xfc000000, 0xfc0007fe, 'fcmpu'],
  [0xfc000018, 0xfc0007fe, 'frsp'],
  [0xfc000024, 0xfc0007fe, 'fdiv'],
  [0xfc000028, 0xfc0007fe, 'fsub'],
  [0xfc00002a, 0xfc0007fe, 'fadd'],
  [0xfc000032, 0xfc0007fe, 'fmul'],
  [0xfc000038, 0xfc0007fe, 'fmsub'],
  [0xfc00003a, 0xfc0007fe, 'fmadd'],
  [0xfc00003c, 0xfc0007fe, 'fnmsub'],
  [0xfc00003e, 0xfc0007fe, 'fnmadd'],
  [0xfc000040, 0xfc0007fe, 'fcmpo'],
  [0xfc000050, 0xfc0007fe, 'fneg'],
  [0xfc000090, 0xfc0007fe, 'fmr'],
  [0xfc000210, 0xfc0007fe, 'fabs'],
];

const X31_MAP = new Map(X31.map(([b, m, n]) => [(b & m) >>> 0, n]));
const X31_MASKS = X31.map(([b, m]) => [b >>> 0, m >>> 0]);
const FP_MAP = new Map(FP.map(([b, m, n]) => [(b & m) >>> 0, n]));
const FP_MASKS = FP.map(([b, m]) => [b >>> 0, m >>> 0]);

function matchMasked(word, masks, map) {
  for (const [base, mask] of masks) {
    if (((word & mask) >>> 0) === (base >>> 0)) return map.get((base >>> 0));
  }
  return null;
}

const ext = (v, bits) => {
  const m = 1 << (bits - 1);
  return (v ^ m) - m;
};

function condName(BO, BI) {
  const l = BI & 3;
  const always = (BO & 0x10) !== 0;
  const expectTrue = (BO & 0x08) !== 0;
  if (always) {
    if ((BO & 0x04) !== 0) return 'b';
    return (BO & 0x02) !== 0 ? 'bdz' : 'bdnz';
  }
  const nm = ['lt', 'gt', 'eq', 'so'][l];
  if (expectTrue) return nm === 'eq' ? 'beq' : nm === 'lt' ? 'blt' : nm === 'gt' ? 'bgt' : 'bso';
  return nm === 'eq' ? 'bne' : nm === 'lt' ? 'bge' : nm === 'gt' ? 'ble' : 'bns';
}

function dsOffset(word) {
  let v = (word >>> 2) & 0x3fff;
  if (v & 0x2000) v |= ~0x3fff;
  return v * 4;
}

function sprOf(word) {
  const f = (word >>> 11) & 0x3ff;
  return ((f & 0x1f) << 5) | (f >> 5);
}

/**
 * Decode one instruction word.
 * @param {number} word 32-bit big-endian instruction
 * @param {number} address guest address of the instruction
 */
export function decodeInstruction(word, address) {
  const op = (word >>> 26) & 0x3f;
  const a = (name, fields) => ({ name, address, word, ...fields });
  const RT = (word >>> 21) & 31, RA = (word >>> 16) & 31, RB = (word >>> 11) & 31;
  const SIMM = ext(word & 0xffff, 16);
  const UIMM = word & 0xffff;
  const Rc = word & 1, OE = (word >>> 10) & 1;

  switch (op) {
    case 7: return a('mulli', { RT, RA, SIMM });
    case 8: return a('subfic', { RT, RA, SIMM });
    case 10: return a('cmpli', { CRFD: (word >>> 23) & 7, L: (word >>> 21) & 1, RA, UIMM });
    case 11: return a('cmpi', { CRFD: (word >>> 23) & 7, L: (word >>> 21) & 1, RA, SIMM });
    case 12: return a('addic', { RT, RA, SIMM });
    case 13: return a('addic.', { RT, RA, SIMM });
    case 14: {
      if (RA === 0) return a(RT === 0 && SIMM === 0 ? 'nop' : 'li', { RT, RA, SIMM });
      return a('addi', { RT, RA, SIMM });
    }
    case 15: return a(RA === 0 ? 'lis' : 'addis', { RT, RA, SIMM });
    case 16: {
      const BO = (word >>> 21) & 31, BI = (word >>> 16) & 31;
      const BD = ext((word >>> 2) & 0x3fff, 14) << 2;
      const AA = (word >>> 1) & 1, LK = word & 1;
      return a(condName(BO, BI), { BO, BI, BD, AA, LK, isBranch: true,
        branchTarget: (AA ? BD : address + BD) >>> 0 });
    }
    case 17: return a('sc', { LEV: (word >>> 5) & 0x7f, ends: true });
    case 18: {
      const LI = ext((word >>> 2) & 0x00ffffff, 24) << 2;
      const AA = (word >>> 1) & 1, LK = word & 1;
      return a(LK ? 'bl' : 'b', { LI, AA, LK, isBranch: true, isCall: !!LK, ends: !LK,
        branchTarget: (AA ? LI : address + LI) >>> 0 });
    }
    case 19: {
      const BO = (word >>> 21) & 31, BI = (word >>> 16) & 31;
      const XO = (word >>> 1) & 0x3ff, LK = word & 1;
      if (XO === 50) return a('rfi', { BO, BI, XO, LK, ends: true });
      if (XO === 18) return a('rfid', { BO, BI, XO, LK, ends: true });
      if (XO === 150) return a('isync', { BO, BI, XO, LK });
      if (XO === 16 || XO === 528) {
        return a(XO === 16 ? (LK ? 'blrl' : 'blr') : (LK ? 'bctrl' : 'bctr'), { BO, BI, XO, LK, isBranch: true, isIndirect: true });
      }
      return a('xl_' + XO.toString(16), { BO, BI, XO, LK });
    }
    case 20: { // rlwimi
      const SH = (word >>> 11) & 31, MB = (word >>> 6) & 31, ME = (word >>> 1) & 31;
      return a('rlwimi', { RO: RT, RA, RS: RT, SH, MB, ME, Rc });
    }
    case 21: { // rlwinm (+ slwi/srwi aliases)
      const SH = (word >>> 11) & 31, MB = (word >>> 6) & 31, ME = (word >>> 1) & 31;
      let name = 'rlwinm';
      if (MB === 0 && ME === (31 - SH)) name = 'slwi';
      else if (MB === SH && ME === 31) name = 'srwi';
      return a(name, { RT, RA, RS: RT, SH, MB, ME, Rc });
    }
    case 22: return a('rlwnm', { RT, RA, RB, MB: (word >>> 6) & 31, ME: (word >>> 1) & 31, Rc });
    case 24: return a('ori', { RT, RA, UIMM });
    case 25: return a('oris', { RT, RA, UIMM });
    case 26: return a('xori', { RT, RA, UIMM });
    case 27: return a('xoris', { RT, RA, UIMM });
    case 28: return a('andi.', { RT, RA, UIMM });
    case 29: return a('andis.', { RT, RA, UIMM });
    case 30: { // MD / MDS forms: rldicl, rldicr, rldic, rldimi, rldcl, rldcr
      const MD_MAP = new Map([
        [0x78000000, 'rldicl'], [0x78000004, 'rldicr'], [0x78000008, 'rldic'],
        [0x7800000c, 'rldimi'], [0x78000010, 'rldcl'], [0x78000012, 'rldcr'],
      ]);
      const name = MD_MAP.get((word & 0x7c00001e) >>> 0) ?? ('rld_' + ((word >>> 1) & 0x1f).toString(16));
      // SH is 6 bits: SH[5] at bit 1 (MD-form only), SH[4:0] at bits 11..15.
      const SH = (((word >>> 1) & 1) << 5) | ((word >>> 11) & 31);
      const MB = (((word >>> 5) & 1) << 5) | ((word >>> 6) & 31);
      return a(name, { RT, RA, RS: RT, RB, SH, MB, Rc });
    }
    case 31: {
      const name = matchMasked(word, X31_MASKS, X31_MAP);
      const base = { RT, RA, RB, XO: (word >>> 1) & 0x3ff, OE, Rc, CRFD: (word >>> 23) & 7 };
      if (name === 'cmp' || name === 'cmpl') base.CRFD = (word >>> 23) & 7;
      if (name === 'cmp' || name === 'cmpl') base.L = OE;
      if (name === 'mfspr' || name === 'mtspr') base.SPR = sprOf(word);
      return a(name ?? ('x31_' + ((word >>> 1) & 0x3ff).toString(16)), base);
    }
    case 32: return a('lwz', { RT, RA, SIMM });
    case 33: return a('lwzu', { RT, RA, SIMM });
    case 34: return a('lbz', { RT, RA, SIMM });
    case 35: return a('lbzu', { RT, RA, SIMM });
    case 36: return a('stw', { RT, RA, SIMM });
    case 38: return a('stb', { RT, RA, SIMM });
    case 40: return a('lhz', { RT, RA, SIMM });
    case 42: return a('lha', { RT, RA, SIMM });
    case 44: return a('sth', { RT, RA, SIMM });
    case 46: return a('sthu', { RT, RA, SIMM });
    case 48: return a('lfs', { RT, RA, SIMM });
    case 50: return a('lfd', { RT, RA, SIMM });
    case 52: return a('stfs', { RT, RA, SIMM });
    case 54: return a('stfd', { RT, RA, SIMM });
    case 58: {
      const XO = word & 3;
      if (XO === 2) return a('lwa', { RT, RA, SIMM: dsOffset(word) });
      if (XO === 0) return a('ld', { RT, RA, SIMM: dsOffset(word) });
      if (XO === 1) return a('ldu', { RT, RA, SIMM: dsOffset(word) });
      return a('ds58_' + XO, { RT, RA, SIMM: dsOffset(word) });
    }
    case 62: {
      const XO = word & 3;
      if (XO === 0) return a('std', { RT, RA, SIMM: dsOffset(word) });
      if (XO === 1) return a('stdu', { RT, RA, SIMM: dsOffset(word) });
      return a('ds62_' + XO, { RT, RA, SIMM: dsOffset(word) });
    }
    case 59: case 63: {
      const name = matchMasked(word, FP_MASKS, FP_MAP);
      return a(name ?? ('f_' + ((word >>> 1) & 0x3ff).toString(16)),
        { FRT: RT, FRA: RA, FRB: RB, FRC: (word >>> 6) & 31, XO: (word >>> 1) & 0x3ff, Rc, single: op === 59 });
    }
    default: return a('op' + op, { op });
  }
}

export const TERMINATORS = new Set(['b', 'blr', 'blrl', 'bclr', 'rfi', 'rfid', 'sc']);
export const BRANCH_OPS = new Set(['b', 'bl', 'beq', 'bne', 'blt', 'bgt', 'bge', 'ble', 'bso', 'bns', 'bdz', 'bdnz', 'bc', 'blr', 'bctr']);
