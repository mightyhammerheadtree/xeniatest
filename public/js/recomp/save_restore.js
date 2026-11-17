/**
 * save_restore.js — static detection of the Xbox 360 compiler's shared
 * register save/restore stubs, plus synthesis of the VMX64 stubs.
 *
 * The Xenon compiler emits one shared stub per non-volatile register class,
 * entered from many prologues/epilogues; `__savegprlr_N` means "save rN..r31".
 * Entries point into the middle of the stub body and are only reached through
 * branches, so they are easy to miss in static discovery (XenonRecomp lists
 * the stubs in its TOML config for exactly this reason). Detecting them by
 * pattern and seeding the entries is additive and conservative: a candidate
 * whose uniform sequence does not run to the end of its register class or
 * whose expected `blr` terminator is missing is dropped, and the stub then
 * behaves exactly as before.
 *
 *   FPR/GPR stubs (`__savefpr_N`, `__restfpr_N`, `__savegprlr_N`,
 *   `__restgprlr_N`) — sequences of `std`/`ld` (DS-form) or `stfd`/`lfd`
 *   (D-form) on r12 with the register field ascending by 1 and the
 *   displacement ascending by 8, ending at r31/f31 followed by `blr`.
 *   These contain only instructions the lifter already supports, so seeding
 *   the entries is enough — the stub bodies are recompiled normally.
 *
 *   VMX stubs (`__savevmx_N`, `__restvmx_N` for v14..v31, and the "vmx64"
 *   pair for v64..v127) — self-contained pairs `li r11, slot; stvx|lvx vI,
 *   r11, r12` (vmx14: opcode 31, lvx XO 103 / stvx XO 231) or the 128-bit
 *   forms `lvx128|stvx128 vI, r11, r12` (vmx64: opcode 4, XO 195 / XO 451,
 *   vector register split across the VDS128 field, see insert_vds128 in
 *   hedge-dev's ppc-dis.c). There is no VMX model in the lifter yet, so a
 *   plain lift would emit a loud runtime trap in *every prologue* of a real
 *   game. Instead the VMX entries are seeded and SYNTHESIZED:
 *     save    → zero-fill this entry's [r12+slot, +16) VMX save slots
 *     restore → return immediately (VMX register state is not modeled)
 *   The save area is the callee's reserved frame region (the same bytes the
 *   real stub writes), so defining it as zeros is guest-observable only
 *   through the VMX unit — and every real VMX instruction still traps
 *   loudly as before. Short of M3 (VMX support) nothing more is possible;
 *   wrong bits are worse than defined zeros.
 *
 * All addresses are 32-bit guest VAs masked like the rest of the recompiler.
 */

export const FPR_STUB = 'fpr';
export const GPR_STUB = 'gpr';
export const VMX_STUB = 'vmx';

const BLR = 0x4E800020;

function readWordBE(arena, addr) {
  const a = addr >>> 0;
  if (a + 4 > arena.length) return 0;
  return ((arena[a] << 24) | (arena[a + 1] << 16) | (arena[a + 2] << 8) | arena[a + 3]) >>> 0;
}

const sign16 = (w) => (((w & 0xffff) << 16) >> 16);
// DS-form displacement: bits 16..29 hold the offset, low 2 bits are the XO.
const dsOffset = (w) => (((w & 0xfffc) << 16) >> 16);

function inRanges(addr, ranges) {
  for (const [start, end] of ranges) {
    if (addr >= start && addr < end) return true;
  }
  return false;
}

/**
 * Classify an FPR/GPR save/restore instruction.
 * @returns {{fam:string, save:boolean, reg:number, ra:number, imm:number}|null}
 */
function classifyFpInt(w) {
  const op = w >>> 26;
  if (op === 50 || op === 54) { // lfd / stfd, D-form
    return { fam: FPR_STUB, save: op === 54, reg: (w >>> 21) & 31, ra: (w >>> 16) & 31, imm: sign16(w) };
  }
  if (op === 58 || op === 62) { // ld / ldu / std / stdu, DS-form
    const xo = w & 0x3;
    if (xo !== 0) return null; // only ld (xo=0) / std (xo=0)
    return { fam: GPR_STUB, save: op === 62, reg: (w >>> 21) & 31, ra: (w >>> 16) & 31, imm: dsOffset(w) };
  }
  return null;
}

const VX128_MASK = 0x7f3;
const LVX128 = 195, STVX128 = 451;  // opcode 4 XO values
const LVX = 103, STVX = 231;        // opcode 31 XO values
const vds128 = (w) => ((w << 3) & 0x60) | ((w >>> 21) & 0x1f);
const xformXo = (w) => (w >>> 1) & 0x3ff;

/**
 * Classify the second instruction of a VMX save/restore pair.
 * @returns {{save:boolean, reg:number}|null}
 */
function classifyVmxPair(w, firstReg) {
  const op = w >>> 26;
  const ra = (w >>> 16) & 31, rb = (w >>> 11) & 31;
  if (ra !== 11 || rb !== 12) return null;
  if (firstReg === 14) {
    if (op !== 31) return null;
    const xo = xformXo(w);
    if (xo !== LVX && xo !== STVX) return null;
    const reg = (w >>> 21) & 31;
    return { save: xo === STVX, reg };
  }
  // vmx64: lvx128 / stvx128
  if (op !== 4) return null;
  const xo = w & VX128_MASK;
  if (xo !== LVX128 && xo !== STVX128) return null;
  return { save: xo === STVX128, reg: vds128(w) };
}

/**
 * Detect save/restore stubs in code ranges.
 * @param {Uint8Array} arena
 * @param {{ranges: Array<[number, number]>, mask: number}} opts
 * @returns {{
 *   stubs: Array<{fam:string, save:boolean, base:number, firstReg:number, lastReg:number, entries:number[]}>,
 *   vmx: Array<{fam:string, save:boolean, base:number, firstReg:number, lastReg:number, stride:number, simmBase:number, entries:number[]}>,
 *   seedAddresses: number[],
 *   vmxStubs: Map<number, {save:boolean, firstReg:number, lastReg:number, simmBase:number, famFirst:number, famLast:number}>,
 *   dropped: number,
 * }}
 */
export function detectSaveRestoreStubs(arena, { ranges, mask }) {
  const stubs = [];
  const vmx = [];
  const dropped = { count: 0 };
  const entries = new Set();
  const vmxStubs = new Map();

  for (const [rangeStart, rangeEnd] of ranges) {
    let addr = rangeStart;
    while (addr + 4 <= rangeEnd) {
      // -- FPR / GPR runs --
      const c = classifyFpInt(readWordBE(arena, addr));
      if (c && c.ra === 12 && c.reg >= 14 && c.imm < 0 && (c.imm & 7) === 0) {
        const reg0 = c.reg, imm0 = c.imm, fam = c.fam, save = c.save;
        let k = 0;
        for (;; k++) {
          const a2 = addr + k * 4;
          if (a2 + 4 > rangeEnd) break;
          const c2 = classifyFpInt(readWordBE(arena, a2));
          if (!c2 || c2.fam !== fam || c2.save !== save) break;
          if (c2.ra !== 12 || c2.reg !== reg0 + k || c2.imm !== imm0 + 8 * k) break;
          if (reg0 + k === 31) { k++; break; }
        }
        const endReg = reg0 + k - 1;
        if (endReg === 31 && readWordBE(arena, addr + k * 4) === BLR) {
          const list = [];
          for (let i = 0; i < k; i++) { list.push((addr + i * 4) & mask); entries.add((addr + i * 4) & mask); }
          stubs.push({ fam, save, base: addr & mask, firstReg: reg0, lastReg: 31, entries: list });
          addr += k * 4;
          continue;
        }
        dropped.count++;
      }

      // -- VMX pair sequences --
      const w0 = readWordBE(arena, addr);
      if (((w0 >>> 26) & 0x3f) === 14 && ((w0 >>> 16) & 0x1f) === 0) { // addi/li to r11
        const imm = sign16(w0);
        const fam64 = imm === -0x400, fam14 = imm === -0x120;
        if (fam64 || fam14) {
          const firstReg = fam14 ? 14 : 64;
          const lastReg = fam14 ? 31 : 127;
          const nRegs = lastReg - firstReg + 1;
          let ok = true;
          let save = null;
          for (let k = 0; k < nRegs; k++) {
            const a2 = addr + k * 8;
            if (a2 + 8 > rangeEnd) { ok = false; break; }
            const li = readWordBE(arena, a2);
            const op = (li >>> 26) & 0x3f;
            if (op !== 14 || ((li >>> 21) & 31) !== 11 || ((li >>> 16) & 31) !== 0) { ok = false; break; }
            if (sign16(li) !== (fam14 ? -0x120 : -0x400) + k * 0x10) { ok = false; break; }
            const pair = classifyVmxPair(readWordBE(arena, a2 + 4), firstReg);
            if (!pair || pair.reg !== firstReg + k) { ok = false; break; }
            if (save === null) save = pair.save;
            else if (pair.save !== save) { ok = false; break; }
          }
          if (ok && readWordBE(arena, addr + nRegs * 8) === BLR) {
            // full VMX stub: every pair boundary is an entry point
            const list = [];
            for (let k = 0; k < nRegs; k++) {
              const e = (addr + k * 8) & mask;
              list.push(e);
              entries.add(e);
              vmxStubs.set(e, {
                save: save === true,
                firstReg: firstReg + k,
                lastReg,
                simmBase: fam14 ? -0x120 : -0x400,
                famFirst: firstReg,
              });
            }
            vmx.push({
              fam: fam14 ? 'vmx14' : 'vmx64',
              save: save === true,
              base: addr & mask,
              firstReg,
              lastReg,
              stride: 8,
              entries: list,
            });
            addr += nRegs * 8;
            continue;
          }
          dropped.count++;
        }
      }
      addr += 4;
    }
  }

  const seedAddresses = [...entries].sort((a, b) => a - b);
  return { stubs, vmx, seedAddresses, vmxStubs, dropped: dropped.count };
}

/**
 * Emission for a synthesized VMX save/restore fragment.
 * Save: zero-fill the [r12 + slot, +16) save slots for this entry's register
 * range. Restore: no-op (VMX register file is not modeled — see module docs).
 * Big-endian zeros are byte-order agnostic, so no bswap is needed.
 */
export function emitVmxStubOps(lifter, desc, O) {
  const out = [];
  if (desc.save) {
    for (let reg = desc.firstReg; reg <= desc.lastReg; reg++) {
      const simm = desc.simmBase + (reg - desc.famFirst) * 0x10;
      // ea = r12 + simm (i64)
      out.push(...O.global_get(12), ...O.i64_const(BigInt(simm)), O.i64_add, ...O.local_set(1)); // L_T64 = 1
      for (const off of [0, 8]) {
        out.push(...O.local_get(1), ...lifter.addrMem(), ...O.i64_const(0n), ...O.i64_store(0, off));
      }
    }
  }
  return out;
}
