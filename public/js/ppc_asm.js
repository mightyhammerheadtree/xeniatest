/**
 * ppc_asm.js — a tiny two-pass assembler for a big-endian 64-bit PowerPC
 * (Xenon "PPC64") subset, sufficient to author built-in demo programs and
 * unit-test payloads. Not a full toolchain: no relocations, no macros, no
 * floating point, no vector unit.
 *
 * Supported syntax:
 *   label:                      (own line, or "label: instr ...")
 *   addi r3, r4, 5              register numbers are r0..r31
 *   li r3, 0x1234 / lis r3, -1
 *   lis r3, sym@ha ; addi r3, r3, sym@l
 *   b label / bl label / beq label / bne / blt / bgt / bge / ble / bdnz / blr
 *   .org 0x10000                set location counter (gap filled with zeros)
 *   .byte 1, 2, 'A'
 *   .ascii "text\n"
 *   .asciiz "text\n"            NUL terminated
 *   .word 0xDEADBEEF            4 bytes, big-endian
 *   .dword 0x1122334455667788   8 bytes, big-endian
 *   .space 24                   reserve zeroed bytes
 *   # comment     (also // and ;)
 *
 * Assembled bytes are exactly what a big-endian Xenon would fetch.
 */

const REG_NAMES = { lr: 8, ctr: 9, xer: 1, cr: 0 };

function parseRegister(token) {
  const m = /^r([0-9]|[12][0-9]|3[01])$/.exec(token);
  if (m) return Number(m[1]);
  if (token in REG_NAMES) return REG_NAMES[token];
  throw new Error(`bad register: '${token}' (expected r0..r31, lr, ctr)`);
}

function parseFpRegister(token) {
  const m = /^f([0-9]|[12][0-9]|3[01])$/.exec(token);
  if (m) return Number(m[1]);
  throw new Error(`bad FP register: '${token}' (expected f0..f31)`);
}

function parseImmediate(token, symbols) {
  if (/^-?0x[0-9a-fA-F]+$/.test(token)) return parseInt(token, 16);
  if (/^-?\d+$/.test(token)) return parseInt(token, 10);
  const sym = /^([A-Za-z_.$][\w.$]*)(@ha|@h|@l)?$/.exec(token);
  if (!sym) throw new Error(`bad immediate: '${token}'`);
  if (!(sym[1] in symbols)) throw new Error(`unknown symbol '${sym[1]}'`);
  const value = symbols[sym[1]];
  switch (sym[2] || '@l') {
    case '@h': return (value >>> 16) & 0xffff;
    case '@ha': return ((value + 0x8000) >>> 16) & 0xffff;
    case '@l': return value & 0xffff;
    default: throw new Error(`bad relocation '${sym[2]}'`);
  }
}

function parseBranchTarget(token, symbols) {
  if (!(token in symbols)) throw new Error(`unknown label '${token}'`);
  return symbols[token];
}

const swapSpr = (spr) => ((spr & 0x1f) << 5) | ((spr >> 5) & 0x1f);

function buildEncoders() {
  const enc = {};
  // li/lis are aliases for addi/addis with r0.
  enc.li = ([rt, imm]) => (rt << 21) | (parseImm16(imm) & 0xffff) | (14 << 26);
  enc.lis = ([rt, imm]) => (rt << 21) | (parseImm16(imm) & 0xffff) | (15 << 26);
  enc.addi = ([rt, ra, imm]) => (14 << 26) | (rt << 21) | (ra << 16) | (imm & 0xffff);
  enc.addis = ([rt, ra, imm]) => (15 << 26) | (rt << 21) | (ra << 16) | (imm & 0xffff);
  enc.addic = ([rt, ra, imm]) => (12 << 26) | (rt << 21) | (ra << 16) | (imm & 0xffff);
  enc.subfic = ([rt, ra, imm]) => (8 << 26) | (rt << 21) | (ra << 16) | (imm & 0xffff);
  enc.ori = ([ra, rs, imm]) => (24 << 26) | (rs << 21) | (ra << 16) | (imm & 0xffff);
  enc.oris = ([ra, rs, imm]) => (25 << 26) | (rs << 21) | (ra << 16) | (imm & 0xffff);
  enc.xori = ([ra, rs, imm]) => (26 << 26) | (rs << 21) | (ra << 16) | (imm & 0xffff);
  enc.xoris = ([ra, rs, imm]) => (27 << 26) | (rs << 21) | (ra << 16) | (imm & 0xffff);
  enc.andi = ([ra, rs, imm]) => (28 << 26) | (rs << 21) | (ra << 16) | (imm & 0xffff);
  enc.andis = ([ra, rs, imm]) => (29 << 26) | (rs << 21) | (ra << 16) | (imm & 0xffff);
  // Arithmetic X-form: destination RT at field0.
  const x = (xo) => ([rs, ra, rb = 0]) =>
    (31 << 26) | (rs << 21) | (ra << 16) | (rb << 11) | (xo << 1);
  enc.add = x(266); enc.subf = x(40); enc.mullw = x(235);
  enc.divw = x(491); enc.divwu = x(459);
  // Logical X-form: destination RA at field1; sources RS (field0) and RB.
  const xlogic = (xo) => ([ra, rs, rb = 0]) =>
    (31 << 26) | (rs << 21) | (ra << 16) | (rb << 11) | (xo << 1);
  enc.xor = xlogic(316); enc.and = xlogic(28); enc.or = xlogic(444); enc.nor = xlogic(124);
  enc.neg = ([rt, ra]) => (31 << 26) | (rt << 21) | (ra << 16) | (104 << 1);
  enc.mr = ([rd, rs]) => (31 << 26) | (rd << 21) | (rs << 16) | (rs << 11) | (444 << 1);
  enc.nop = () => (24 << 26); // ori r0, r0, 0
  enc.mulli = ([rt, ra, imm]) => (7 << 26) | (rt << 21) | (ra << 16) | (imm & 0xffff);
  // Loads / stores.
  enc.lbz = ([rt, ra, d]) => (34 << 26) | (rt << 21) | (ra << 16) | (d & 0xffff);
  enc.lhz = ([rt, ra, d]) => (40 << 26) | (rt << 21) | (ra << 16) | (d & 0xffff);
  enc.lwz = ([rt, ra, d]) => (32 << 26) | (rt << 21) | (ra << 16) | (d & 0xffff);
  enc.stb = ([rs, ra, d]) => (38 << 26) | (rs << 21) | (ra << 16) | (d & 0xffff);
  enc.sth = ([rs, ra, d]) => (44 << 26) | (rs << 21) | (ra << 16) | (d & 0xffff);
  enc.stw = ([rs, ra, d]) => (36 << 26) | (rs << 21) | (ra << 16) | (d & 0xffff);
  // DS-form (ld/lwa/std): displacement is a signed 16-bit ds<<2.
  const ds = (op, xo) => ([rt, ra, d]) =>
    (op << 26) | (rt << 21) | (ra << 16) | (((d >> 2) & 0x3fff) << 2) | xo;
  enc.ld = ds(58, 0);
  enc.lwa = ds(58, 2);
  enc.std = ([rs, ra, d]) => (62 << 26) | (rs << 21) | (ra << 16) | (((d >> 2) & 0x3fff) << 2);
  // FP loads/stores, D-form (fpr = {fp:n} sentinel — see resolveInstructionOperands).
  enc.lfd = ([frt, ra, d]) => (50 << 26) | (frt.fp << 21) | (ra << 16) | (d & 0xffff);
  enc.stfd = ([frs, ra, d]) => (54 << 26) | (frs.fp << 21) | (ra << 16) | (d & 0xffff);
  // Compares. cr field defaults to 0.
  enc.cmpw = ([ra, rb, crf = 0]) => (31 << 26) | ((crf << 2) << 21) | (ra << 16) | (rb << 11);
  enc.cmplw = ([ra, rb, crf = 0]) => (31 << 26) | ((crf << 2) << 21) | (ra << 16) | (rb << 11) | (32 << 1);
  enc.cmpi = ([ra, simm, crf = 0]) => (11 << 26) | ((crf << 2) << 21) | (ra << 16) | (simm & 0xffff);
  enc.cmpwi = ([ra, simm, crf = 0]) => enc.cmpi([ra, simm, crf]);
  enc.cmplwi = ([ra, uimm, crf = 0]) => (10 << 26) | ((crf << 2) << 21) | (ra << 16) | (uimm & 0xffff);
  enc.cmpdi = ([ra, simm, crf = 0]) => (11 << 26) | (((crf << 2) | 1) << 21) | (ra << 16) | (simm & 0xffff);
  enc.cmpldi = ([ra, uimm, crf = 0]) => (10 << 26) | (((crf << 2) | 1) << 21) | (ra << 16) | (uimm & 0xffff);
  // SPRs: mtspr takes the SPR number, not the swapped encoding.
  enc.mtspr = ([spr, rt]) => (31 << 26) | (rt << 21) | (swapSpr(spr) << 11) | (467 << 1);
  enc.mfspr = ([rt, spr]) => (31 << 26) | (rt << 21) | (swapSpr(spr) << 11) | (339 << 1);
  enc.mtlr = ([rt]) => enc.mtspr([REG_NAMES.lr, rt]);
  enc.mtctr = ([rt]) => enc.mtspr([REG_NAMES.ctr, rt]);
  enc.mfctr = ([rt]) => enc.mfspr([rt, REG_NAMES.ctr]);
  enc.mflr = ([rt]) => enc.mfspr([rt, REG_NAMES.lr]);
  enc.mtxer = ([rt]) => enc.mtspr([REG_NAMES.xer, rt]);
  // Calls. blr = bclr with BO=0b10100, BI=0, LK=0.
  enc.blr = () => (19 << 26) | (20 << 21) | (16 << 1);
  enc.blrl = () => (19 << 26) | (20 << 21) | (16 << 1) | 1;
  // bcctr with BO=0b10100, BI=0, XO=528.
  enc.bctr = () => (19 << 26) | (20 << 21) | (528 << 1);
  enc.bctrl = () => (19 << 26) | (20 << 21) | (528 << 1) | 1;
  // 32-bit rotate/mask: MB,ME are 5-bit fields.
  enc.rlwinm = ([ra, rs, sh, mb, me]) => (21 << 26) | (rs << 21) | (ra << 16) | ((sh & 31) << 11) | ((mb & 31) << 6) | ((me & 31) << 1);
  enc.slwi = ([ra, rs, sh]) => enc.rlwinm([ra, rs, sh, 0, 31 - sh]);
  enc.srwi = ([ra, rs, sh]) => enc.rlwinm([ra, rs, sh, sh, 31]);
  // Indexed loads X-form (XO in bits 11..15, destination RT at field0).
  enc.lwzx = ([rt, ra, rb]) => (31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (23 << 1);
  enc.lbzx = ([rt, ra, rb]) => (31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (87 << 1);
  enc.lhzx = ([rt, ra, rb]) => (31 << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (279 << 1);
  // sc with LEV=0.
  enc.sc = () => (17 << 26) | 0x02;
  return enc;
}

// Branch condition table: mnemonic => {bo, bi}
const BCOND = {
  beq: { bo: 12, bi: 2 }, bne: { bo: 4, bi: 2 },
  blt: { bo: 12, bi: 0 }, bgt: { bo: 12, bi: 1 },
  bge: { bo: 4, bi: 0 }, ble: { bo: 4, bi: 1 },
  bdnz: { bo: 18, bi: 0 }, bdz: { bo: 16, bi: 0 },
};

function parseImm16(value) {
  // Already resolved to a number by the operand pass. May be negative
  // (sign-extended 16-bit immediates); clamp via modulo instead of |0
  // so large unsigned values survive.
  return Number(value) & 0xffff;
}

export function assemble(source) {
  const lines = source.split('\n');
  const items = []; // collected work items
  // Absolute ("linked") addresses are used for labels and branches. The first
  // .org sets the image base; the returned byte array starts at that base so
  // no huge zero-filled prefix is materialized.
  let addr = 0;
  let origin = null;
  const symbols = {};
  for (let lineno = 0; lineno < lines.length; lineno++) {
    const rawLine = lines[lineno];
    const line = rawLine
      .replace(/(\/\/|#|;).*$/, '')
      .trim();
    if (!line) continue;

    // Label(s) at start of line, e.g. "foo:" or "foo: bar".
    let rest = line;
    const labelNames = [];
    const labelRe = /^([A-Za-z_.$][\w.$]*):/;
    let m = labelRe.exec(rest);
    while (m) {
      if (labelNames.includes(m[1])) throw new Error(`line ${lineno + 1}: duplicate label '${m[1]}'`);
      labelNames.push(m[1]);
      rest = rest.slice(m[0].length).trim();
      m = labelRe.exec(rest);
    }

    // Instruction fetch is 4-byte aligned: pad the location counter before
    // emitting any instruction (or .word/.dword) so a preceding odd-length
    // string can't misalign the code that follows it.
    const isInstr = rest && !rest.startsWith('.');
    const dir0 = isInstr ? null : (rest.slice(1).split(/\s+/)[0] || '').toLowerCase();
    const alignUp = (a, m) => Math.ceil(a / m) * m; // double-safe: guest bases exceed 2^31
    if (isInstr || dir0 === 'word') addr = alignUp(addr, 4);
    else if (dir0 === 'dword') addr = alignUp(addr, 8);
    else if (labelNames.length) addr = alignUp(addr, 4);

    for (const name of labelNames) {
      if (name in symbols) throw new Error(`line ${lineno + 1}: duplicate label '${name}'`);
      symbols[name] = addr;
    }

    if (!rest) continue;

    if (rest.startsWith('.')) {
      const parts = rest.slice(1).split(/\s+/);
      const dir = parts[0].toLowerCase();
      const argStr = parts.slice(1).join(' ');
      if (dir === 'org') { const a = parseImmediate(argStr.trim(), symbols); if (origin === null) origin = a; addr = a; continue; }
      if (dir === 'space') { const n = parseImmediate(argStr.trim(), symbols) || 0; items.push({ type: 'data', dir, args: argStr, addr, size: n, lineno }); addr += n; continue; }
      if (dir === 'byte') { const n = splitOperands(argStr).length; items.push({ type: 'data', dir, args: argStr, addr, size: n, lineno }); addr += n; continue; }
      if (dir === 'word') { const n = splitOperands(argStr).length * 4; items.push({ type: 'data', dir, args: argStr, addr, size: n, lineno }); addr += n; continue; }
      if (dir === 'dword') { const n = splitOperands(argStr).length * 8; items.push({ type: 'data', dir, args: argStr, addr, size: n, lineno }); addr += n; continue; }
      if (dir === 'ascii' || dir === 'asciiz' || dir === 'string') {
        const bytes = stringDirectiveBytes(argStr, lineno, dir !== 'ascii');
        items.push({ type: 'data', dir: 'bytes', args: bytes, addr, size: bytes.length, lineno });
        addr += bytes.length; continue;
      }
      throw new Error(`line ${lineno + 1}: unknown directive '.${dir}'`);
    }

    const mnemo = rest.split(/\s+/)[0].toLowerCase();
    items.push({ type: 'instr', mnemo, args: rest.slice(rest.indexOf(mnemo) + mnemo.length).trim(), addr, size: 4, lineno });
    addr += 4;
  }

  // ---- Pass 2: resolve operands and emit.
  const base = origin ?? 0;
  const out = new Uint8Array(addr - base);
  const view = new DataView(out.buffer);
  const encoders = buildEncoders();

  for (const item of items) {
    if (item.type === 'data') {
      writeData(out, view, item, symbols, linenoOf(item), base);
      continue;
    }
    const ops = resolveInstructionOperands(item, symbols);
    const pc = item.addr;
    let word;
    if (item.mnemo in BCOND) {
      const { bo, bi } = BCOND[item.mnemo];
      const [targetTok, lkTok] = splitOperands(item.args);
      const target = parseBranchTarget(targetTok, symbols);
      const bd = ((target - pc) >> 2) & 0x3fff;
      const disp = target - pc;
      if (disp < -0x8000 || disp >= 0x8000) {
        throw new Error(`line ${item.lineno + 1}: branch out of range (${disp} bytes)`);
      }
      const lk = lkTok === 'lk' ? 1 : 0;
      word = (16 << 26) | (bo << 21) | (bi << 16) | (bd << 2) | lk;
    } else if (item.mnemo === 'b' || item.mnemo === 'bl') {
      const [targetTok] = splitOperands(item.args);
      const target = parseBranchTarget(targetTok, symbols);
      const disp = target - pc;
      if (disp < -0x2000000 || disp >= 0x2000000) {
        throw new Error(`line ${item.lineno + 1}: branch out of range (${disp} bytes)`);
      }
      const li = (disp >> 2) & 0xffffff;
      word = (18 << 26) | (li << 2) | (item.mnemo === 'bl' ? 1 : 0);
    } else if (item.mnemo === 'bc') {
      // bc BO, BI, target
      const toks = splitOperands(item.args);
      if (toks.length < 3) throw new Error(`line ${item.lineno + 1}: bc needs BO, BI, target`);
      const bo = Number(toks[0]), bi = Number(toks[1]);
      const target = parseBranchTarget(toks[2], symbols);
      const disp = target - pc;
      if (disp < -0x8000 || disp >= 0x8000) {
        throw new Error(`line ${item.lineno + 1}: branch out of range (${disp} bytes)`);
      }
      const bd = ((disp >> 2) & 0x3fff) << 2;
      word = (16 << 26) | (bo << 21) | (bi << 16) | bd;
    } else if (item.mnemo in encoders) {
      const resolved = ops;
      word = encoders[item.mnemo](resolved) >>> 0;
    } else {
      throw new Error(`line ${item.lineno + 1}: unsupported instruction '${item.mnemo}'`);
    }
    view.setUint32(pc - base, word >>> 0, false);
  }

  return { bytes: out, base, size: out.length };
}

function linenoOf(item) { return item.lineno + 1; }

function splitOperands(argStr) {
  const parts = [];
  let depth = 0, cur = '', inQuote = false;
  for (const ch of argStr) {
    if (ch === '"') { inQuote = !inQuote; cur += ch; continue; }
    if (!inQuote) {
      if (ch === '(') depth++;
      else if (ch === ')') depth--;
      else if (ch === ',' && depth === 0) { parts.push(cur.trim()); cur = ''; continue; }
    }
    cur += ch;
  }
  if (cur.trim()) parts.push(cur.trim());
  return parts.map((p) => p.trim());
}

function stringDirectiveBytes(argStr, lineno, nul) {
  const q = /^"(.*)"$/.exec(argStr.trim());
  if (!q) throw new Error(`line ${lineno + 1}: string directive needs a quoted argument`);
  const bytes = [];
  const body = q[1];
  for (let i = 0; i < body.length; i++) {
    const ch = body[i];
    if (ch === '\\' && i + 1 < body.length) {
      const esc = body[++i];
      switch (esc) {
        case 'n': bytes.push(10); break;
        case 't': bytes.push(9); break;
        case 'r': bytes.push(13); break;
        case '0': bytes.push(0); break;
        case '\\': bytes.push(92); break;
        case '"': bytes.push(34); break;
        default: throw new Error(`line ${lineno + 1}: bad escape '\\${esc}'`);
      }
    } else {
      const code = body.codePointAt(i);
      if (code > 0x7f) throw new Error(`line ${lineno + 1}: non-ASCII byte in string literal`);
      bytes.push(code);
    }
  }
  if (nul) bytes.push(0);
  return bytes;
}

function resolveInstructionOperands(item, symbols) {
  const toks = splitOperands(item.args);
  if (toks.length === 0 || (toks.length === 1 && (item.mnemo === 'nop' || item.mnemo === 'blr' || item.mnemo === 'sc'))) {
    return [];
  }
  const out = [];
  for (let i = 0; i < toks.length; i++) {
    const tok = toks[i];
    const mem = /^([^()]*)\(([^()]+)\)$/.exec(tok);
    if (mem) {
      const ra = parseRegister(mem[2].trim());
      const offTok = mem[1].trim();
      const offset = offTok === '' ? 0 : parseImmediate(offTok, symbols);
      out.push(ra, offset);
      continue;
    }
    const isRegister = /^([r][0-9]|[r][12][0-9]|[r]3[01]|lr|ctr|xer)$/.test(tok);
    if (isRegister) { out.push(parseRegister(tok)); continue; }
    const isFpRegister = /^f([0-9]|[12][0-9]|3[01])$/.test(tok);
    if (isFpRegister) { out.push({ fp: parseFpRegister(tok) }); continue; }
    out.push(parseImmediate(tok, symbols));
  }
  return out;
}

function writeData(out, view, item, symbols, lineno, base) {
  const { dir, args } = item;
  const addr = item.addr - base;
  if (dir === 'bytes') {
    (args || []).forEach((b, i) => { out[addr + i] = b; });
    return;
  }
  if (dir === 'space') return; // already zeroed
  const toks = splitOperands(args);
  if (dir === 'byte') {
    toks.forEach((t, i) => { out[addr + i] = valueOf(t, symbols, lineno) & 0xff; });
  } else if (dir === 'word') {
    toks.forEach((t, i) => view.setUint32(addr + i * 4, valueOf(t, symbols, lineno) >>> 0, false));
  } else if (dir === 'dword') {
    toks.forEach((t, i) => {
      const txt = String(t).trim();
      let v;
      if (/^-?0x[0-9a-fA-F]+$/.test(txt) || /^-?\d+$/.test(txt)) v = BigInt(txt);
      else if (txt in symbols) v = BigInt(symbols[txt] >>> 0);
      else throw new Error(`line ${lineno}: cannot resolve '${txt}'`);
      view.setBigUint64(addr + i * 8, v & 0xffffffffffffffffn, false);
    });
  }
}

function valueOf(tok, symbols, lineno) {
  const t = String(tok).trim();
  if (/^-?0x[0-9a-fA-F]+$/.test(t)) return parseInt(t, 16);
  if (/^-?\d+$/.test(t)) return parseInt(t, 10);
  const m = /^'(.)'$/.exec(t);
  if (m) return m[1].charCodeAt(0);
  if (t in symbols) return symbols[t];
  throw new Error(`line ${lineno}: cannot resolve '${t}'`);
}
