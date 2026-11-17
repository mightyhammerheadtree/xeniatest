/**
 * switch_tables.js — static detection of jump (switch) tables in guest code.
 *
 * A jump table shows up as: bounds check (`cmplwi` + `bgt`/`ble`), load of the
 * table base (`lis`/`addi`), an indexed load (`lwzx`/`lbzx`/`lhzx`), then
 * `mtctr` + `bctr`. Different Xbox 360 compiler vintages emit one of four
 * shapes (XenonRecomp calls them switch tables and detects them in
 * XenonAnalyse/main.cpp; the recompiler turns them into real switch cases).
 * Ported from hedge-dev/XenonRecomp (MIT, © hedge-dev 2025),
 * XenonAnalyse/main.cpp (pattern scan + ScanTable backward scan + ReadTable)
 * and XenonRecomp/recompiler.cpp (labels become jump destinations).
 *
 * Detection is deliberately conservative: a candidate pattern that fails the
 * backward bounds-check scan, whose labels point outside the code section, or
 * whose table entries are misaligned/zero, is dropped — the indirect `bctr`
 * then keeps today's behavior (runtime `icall`, which traps loudly if the
 * target is unknown). Nothing is silently miscompiled.
 *
 * Type semantics (all addresses are 32-bit guest VAs, masked to the arena):
 *   ABSOLUTE   lis base; addi; rlwinm idx<<2; lwzx; mtctr; bctr
 *              table[i] = absolute VA of case i
 *   COMPUTED   lis offs; addi; lbzx offs; rlwinm offs<<SHIFT; lis base; addi;
 *              add; mtctr
 *              table[i] = base + (byte << SHIFT)  (SHIFT = rlwinm SH)
 *   BYTEOFFSET lis offs; addi; lbzx offs; lis base; addi; add; mtctr
 *              table[i] = base + byte
 *   SHORTOFFSET lis idx; addi; rlwinm idx<<1; lhzx; lis base; addi; add; mtctr
 *              table[i] = base + u16be(offsets)   (rlwinm scales the *index*)
 *
 * The bounds-check backward scan finds the enclosing `crN` compare and its
 * branch, giving the case count (imm + 1) and the default label. The compare
 * immediate sizes the table; the compare register is reported for diagnostics
 * but the dispatch emission compares CTR directly against the labels.
 */

import { decodeInstruction } from './ppc_decode.js';

export const SWITCH_ABSOLUTE = 0;
export const SWITCH_COMPUTED = 1;
export const SWITCH_BYTEOFFSET = 2;
export const SWITCH_SHORTOFFSET = 3;

export const SWITCH_TYPE_NAMES = ['absolute', 'computed', 'byteoffset', 'shortoffset'];

const MAX_CASES = 512;      // sanity cap on imm+1 for the bounds check
const BACKSCAN = 32;        // instructions to look behind the pattern

function isLis(i) { return i.name === 'lis'; }
function isAddi(i) { return i.name === 'addi'; }
function isRlwinm(i) { return i.name === 'rlwinm' || i.name === 'slwi' || i.name === 'srwi'; }
function isMtCtr(i) { return i.name === 'mtspr' && i.SPR === 9; }
const isBctr = (i) => i.name === 'bctr';

const PATTERNS = [
  { type: SWITCH_ABSOLUTE, match: [isLis, isAddi, isRlwinm, (i) => i.name === 'lwzx', isMtCtr, isBctr], bctrAt: 5 },
  { type: SWITCH_COMPUTED, match: [isLis, isAddi, (i) => i.name === 'lbzx', isRlwinm, isLis, isAddi, (i) => i.name === 'add', isMtCtr], bctrAt: 8 },
  { type: SWITCH_BYTEOFFSET, match: [isLis, isAddi, (i) => i.name === 'lbzx', isLis, isAddi, (i) => i.name === 'add', isMtCtr], bctrAt: 7 },
  { type: SWITCH_SHORTOFFSET, match: [isLis, isAddi, isRlwinm, (i) => i.name === 'lhzx', isLis, isAddi, (i) => i.name === 'add', isMtCtr], bctrAt: 8 },
];

function readWordBE(arena, addr) {
  const a = addr >>> 0;
  if (a + 4 > arena.length) return 0;
  return ((arena[a] << 24) | (arena[a + 1] << 16) | (arena[a + 2] << 8) | arena[a + 3]) >>> 0;
}

function u32be(arena, addr, mask) {
  const a = addr & mask;
  return ((arena[a] << 24) | (arena[a + 1] << 16) | (arena[a + 2] << 8) | arena[a + 3]) >>> 0;
}
function u16be(arena, addr, mask) {
  const a = addr & mask;
  return ((arena[a] << 8) | arena[a + 1]) >>> 0;
}

/** Decode the instruction at guest address `addr` (masked arena offset). */
function dec(arena, addr, mask) {
  const a = addr & mask;
  if (a + 4 > arena.length) return null;
  return decodeInstruction(readWordBE(arena, a), a);
}

/** Addresses are 32-bit guest VAs; wrap and mask like the rest of the loader. */
const add32 = (a, b) => (a + b) >>> 0;

/**
 * Backward scan for the bounds check that precedes a switch: a cr-conditional
 * branch (default path), then a cmpi/cmpli on the same CR field giving the
 * case count. Returns { indexReg, caseCount, defaultLabel } or null.
 */
function scanBoundsCheck(arena, matchAddr, mask) {
  let cr = -1;
  let defaultLabel = 0;
  let indexReg = null;
  let caseCount = 0;
  for (let i = 1; i <= BACKSCAN; i++) {
    const addr = matchAddr - i * 4;
    if (addr < 0) break;
    const word = readWordBE(arena, addr & mask);
    if (word === 0) continue;
    const insn = decodeInstruction(word, addr & mask);
    const prim = (word >>> 26) & 0x3f;
    if (cr === -1) {
      // Plain cr-conditional branches only: no CTR decrement (bit 0x04 set = no decrement), no ignore-CR, no LK.
      if (prim === 16 && insn.isBranch && (insn.BO & 0x04) !== 0 && (insn.BO & 0x10) === 0 && !insn.LK) {
        cr = insn.BI >>> 2;
        defaultLabel = insn.branchTarget >>> 0;
        continue;
      }
    } else if ((insn.name === 'cmpli' || insn.name === 'cmpi') && insn.CRFD === cr) {
      const imm = insn.name === 'cmpli' ? insn.UIMM : insn.SIMM;
      if (!(imm >= 0) || imm + 1 > MAX_CASES) return null;
      indexReg = insn.RA;
      caseCount = imm + 1;
      return { indexReg, caseCount, defaultLabel: defaultLabel & mask };
    }
  }
  return null;
}

function inRanges(addr, ranges) {
  for (const [start, end] of ranges) {
    if (addr >= start && addr < end) return true;
  }
  return false;
}

/**
 * Detect jump tables in code ranges.
 * @param {Uint8Array} arena  guest arena (masked flat image)
 * @param {object} opts
 * @param {Array<[number, number]>} opts.ranges  [start, end) code ranges, masked
 * @param {number} opts.mask  arena mask (GUEST_MEM_MASK)
 * @returns {{ tables: Map<number, object>, dropped: number }}
 *   tables keyed by the masked guest address of the `bctr` instruction.
 */
export function detectSwitchTables(arena, { ranges, mask }) {
  const tables = new Map();
  let dropped = 0;
  for (const [rangeStart, rangeEnd] of ranges) {
    for (let addr = rangeStart; addr + 4 <= rangeEnd; addr += 4) {
      for (const pat of PATTERNS) {
        // Fast path: first instruction must decode as lis.
        const w0 = readWordBE(arena, addr);
        if (((w0 >>> 26) & 0x3f) !== 15) continue;
        let ok = true;
        for (let k = 0; k < pat.match.length; k++) {
          const insn = dec(arena, addr + k * 4, mask);
          if (!insn || !pat.match[k](insn)) { ok = false; break; }
        }
        if (!ok) continue;
        // For computed/offset patterns the mtctr is the last matched
        // instruction; the bctr must directly follow it.
        let bctrAddr = addr + pat.bctrAt * 4;
        if (pat.type !== SWITCH_ABSOLUTE) {
          const next = dec(arena, bctrAddr, mask);
          if (!next || next.name !== 'bctr') { dropped++; break; }
        }
        const found = readTable(arena, addr, pat, mask, ranges);
        if (found) {
          if (tables.has(found.bctr)) break; // first match wins
          tables.set(found.bctr, found);
        } else {
          dropped++;
        }
        break; // this position matched a pattern; move on
      }
    }
  }
  return { tables, dropped };
}

function readTable(arena, matchAddr, pat, mask, ranges) {
  const lis0 = dec(arena, matchAddr, mask);
  const addi0 = dec(arena, matchAddr + 4, mask);
  const pOffset = add32((lis0.SIMM << 16) >>> 0, addi0.SIMM) & mask;
  const bounds = scanBoundsCheck(arena, matchAddr, mask);
  if (!bounds) return null;
  const { indexReg, caseCount, defaultLabel } = bounds;
  const labels = [];
  const inCode = (a) => inRanges(a, ranges);
  const bctrAddr = (matchAddr + pat.bctrAt * 4) & mask;

  const wordAt = (a) => readWordBE(arena, a & mask);
  for (let i = 0; i < caseCount; i++) {
    let label;
    if (pat.type === SWITCH_ABSOLUTE) {
      label = u32be(arena, pOffset + i * 4, mask);
    } else if (pat.type === SWITCH_COMPUTED) {
      const shiftInsn = dec(arena, matchAddr + 3 * 4, mask);   // rlwinm
      const lis1 = dec(arena, matchAddr + 4 * 4, mask);
      const addi1 = dec(arena, matchAddr + 5 * 4, mask);
      const base = add32((lis1.SIMM << 16) >>> 0, addi1.SIMM);
      const shift = shiftInsn.SH & 31;
      label = add32(base, (arena[(pOffset + i) & mask] ?? 0) << shift);
    } else if (pat.type === SWITCH_BYTEOFFSET) {
      const lis1 = dec(arena, matchAddr + 3 * 4, mask);
      const addi1 = dec(arena, matchAddr + 4 * 4, mask);
      const base = add32((lis1.SIMM << 16) >>> 0, addi1.SIMM);
      label = add32(base, arena[(pOffset + i) & mask] ?? 0);
    } else { // SWITCH_SHORTOFFSET
      const lis1 = dec(arena, matchAddr + 4 * 4, mask);
      const addi1 = dec(arena, matchAddr + 5 * 4, mask);
      const base = add32((lis1.SIMM << 16) >>> 0, addi1.SIMM);
      label = add32(base, u16be(arena, pOffset + i * 2, mask));
    }
    label &= mask;
    // Validate: 4-aligned, inside a code range, and not zero padding.
    if ((label & 3) !== 0 || !inCode(label) || wordAt(label) === 0) return null;
    labels.push(label);
  }
  return {
    type: pat.type,
    typeName: SWITCH_TYPE_NAMES[pat.type],
    base: matchAddr & mask,
    bctr: bctrAddr,
    tableBase: pOffset,
    indexReg,
    defaultLabel,
    labels,
  };
}
