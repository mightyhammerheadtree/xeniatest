"""PPC64 (big-endian) → C lifter for Xbox 360 code.

Emits C against templates/web-runtime/recomp_types.h. Register semantics
mirror this repo's validated WASM interpreter (core/src/xenon_cpu.cc).

An instruction with no translation becomes `recomp_unimpl(...)` — it never
fails to compile; reaching it at run time is reported through the runtime.
Everything else lifts mechanically.
"""
from __future__ import annotations

from .disasm import (
    Disassembler,
    Instruction,
    parse_branch_target,
    parse_int,
    parse_mem_operand,
    parse_register,
    parse_simm16,
    split_operands,
)
from .functions import RecompiledFunction

TERMINATOR_RETURN = {"blr", "bclr"}
FRAGMENT_END = {"b", "blr", "bclr", "rfi", "rfid", "sc"}


def cr_bit_expr(bi: int) -> str:
    return f"((g_cr >> (31 - {bi})) & 1u)"


def set_cr_expr(field: int, lt: str, gt: str, eq: str, so: str) -> list[str]:
    shift = 28 - field * 4
    mask = 0xF << shift
    lines = [
        f"{{ uint32_t _crf = (({lt}) ? 8u : 0u) | (({gt}) ? 4u : 0u) | (({eq}) ? 2u : 0u) | (({so}) ? 1u : 0u);",
        f"  g_cr = (g_cr & ~0x{mask:08x}u) | ((_crf << {shift}) & 0x{mask:08x}u); }}",
    ]
    return lines


def cr_set_signed(field: int, a: str, b: str) -> list[str]:
    return set_cr_expr(field, f"({a}) < ({b})", f"({a}) > ({b})", f"({a}) == ({b})", "0")


def cr_set_unsigned(field: int, a: str, b: str) -> list[str]:
    return set_cr_expr(field, f"({a}) < ({b})", f"({a}) > ({b})", f"({a}) == ({b})", "0")


def ca_set_expr(carry: str) -> list[str]:
    return [f"g_xer = (g_xer & ~0x20000000u) | (({carry}) ? 0x20000000u : 0u);"]


CA_READ = "((g_xer & 0x20000000u) ? 1u : 0u)"


class LifterError(Exception):
    pass


class Lifter:
    def __init__(self, fragment_addresses: set[int]):
        self.fragment_addresses = fragment_addresses  # branch targets within the current fragment

    # ---- branch helpers --------------------------------------------------
    def emit_branch(self, insn: Instruction, func: RecompiledFunction) -> list[str]:
        mnem = insn.mnemonic
        target = insn.branch_target

        if mnem in ("beq", "bne", "blt", "bgt", "bge", "ble", "bdnz", "bdz", "bc", "bdnza", "beqa", "bnea") or (
            mnem.startswith("b") and target is not None and mnem not in ("b", "bl", "bclr", "bctr", "bctrl")
        ):
            bo, bi = self._decode_bc(insn)
            if bo is None:
                return self._unimpl(insn)
            lines = self._emit_bc_guard(bo, bi)
            if target in self.fragment_addresses:
                lines.append(f"  if (_br) goto L_{target:x};")
            else:
                lines.append(f"  if (_br) {{ recomp_icall(0x{target:x}ull); return; }}")
            lines.append("}")
            return lines

        if mnem == "b":
            assert target is not None
            if target in self.fragment_addresses:
                return [f"goto L_{target:x};"]
            return [f"recomp_call_tail(0x{target:x}ull);"]

        if mnem == "bl":
            assert target is not None
            # bl: LR = return address, then a plain C call to the lifted target.
            return [f"g_lr = 0x{insn.address + 4:x}ull;", f"fn_{target:x}();"]

        return self._unimpl(insn)

    def _emit_bc_guard(self, bo: int, bi: int) -> list[str]:
        lines = ["{"]
        lines.append("  uint64_t _ctr_ok = 1;")
        if not (bo & 0x04):
            lines.append("  g_ctr = g_ctr - 1;")
            if bo & 0x02:
                lines.append("  _ctr_ok = (g_ctr == 0) ? 1 : 0;")
            else:
                lines.append("  _ctr_ok = (g_ctr != 0) ? 1 : 0;")
        lines.append("  uint64_t _cond_ok = 1;")
        if not (bo & 0x10):
            bit = cr_bit_expr(bi)
            if bo & 0x08:
                lines.append(f"  _cond_ok = {bit};")
            else:
                lines.append(f"  _cond_ok = ({bit} ^ 1u);")
        lines.append("  uint64_t _br = _ctr_ok && _cond_ok;")
        return lines

    def _decode_bc(self, insn: Instruction):
        """Recover BO/BI by looking at the original word (authoritative)."""
        word = insn.word
        opcode = (word >> 26) & 0x3F
        if opcode == 16:  # bc
            bo = (word >> 21) & 0x1F
            bi = (word >> 16) & 0x1F
            return bo, bi
        # capstone simplified mnemonics — recover from word directly is safe for all bc forms
        if opcode == 19:  # bcctr (bclr family)
            bo = (word >> 21) & 0x1F
            bi = (word >> 16) & 0x1F
            return bo, bi
        return None, None

    def emit_bcctr(self, insn: Instruction) -> list[str]:
        """bclr/bctr family: conditional return or indirect tail call."""
        word = insn.word
        bo = (word >> 21) & 0x1F
        bi = (word >> 16) & 0x1F
        is_ctr_target = (word & 2) == 0 and ((word >> 1) & 1) == 0  # XO bits: 16<<1=bclr, 528<<1=bctr
        xo = (word >> 1) & 0x3FF
        is_link = (word & 1) != 0
        lines = self._emit_bc_guard(bo, bi)
        if is_link:
            lines.append(f"  if (_br) {{ g_lr = 0x{insn.address + 4:x}ull; }}")
        if xo == 16:
            # target = LR -> conditional return
            lines.append("  if (_br) { return; }")
        elif xo == 528:
            # target = CTR
            lines.append("  if (_br) { recomp_icall(g_ctr); return; }")
        else:
            lines.append(f'  if (_br) {{ recomp_unimpl("bcctr xo{xo}", 0x{insn.address:x}u); return; }}')
        lines.append("}")
        return lines

    # ---- misc ------------------------------------------------------------
    def _unimpl(self, insn: Instruction) -> list[str]:
        text = f"{insn.mnemonic} {insn.op_str}".replace('"', "'")
        return [f'recomp_unimpl("{text}", 0x{insn.address:x}u); /* 0x{insn.word:08x} */']

    # ---- main entry ------------------------------------------------------
    def lift(self, func: RecompiledFunction) -> list[str]:
        lines: list[str] = []
        lines.append(f"/* ---- function at 0x{func.start:x} ({len(func.instructions)} instructions) ---- */")
        lines.append(f"void fn_{func.start:x}(void) {{")

        targets = {i.branch_target for i in func.instructions if i.branch_target is not None}
        self.fragment_addresses = {t for t in targets if func.start <= t < func.end}
        emitted_labels: set[int] = set()

        for insn in func.instructions:
            if insn.address in self.fragment_addresses and insn.address not in emitted_labels:
                lines.append(f"L_{insn.address:x}:")
                emitted_labels.add(insn.address)
            lines.append(f"  /* 0x{insn.address:x}: {insn.mnemonic} {insn.op_str} */")
            lines.extend(self.lift_instruction(insn, func))
        lines.append("}")
        return lines

    def lift_instruction(self, insn: Instruction, func: RecompiledFunction) -> list[str]:
        m = insn.mnemonic
        ops = split_operands(insn.op_str)

        # --- branch forms first (need raw word) ---
        if (insn.branch_kind == "b" or insn.branch_kind == "bl") and m in ("b", "bl"):
            return self.emit_branch(insn, func)
        if m in ("b", "bl"):
            return self.emit_branch(insn, func)
        if m in ("beq", "bne", "blt", "bgt", "bge", "ble", "bdnz", "bdz", "bc", "bdnza") or (
            m.startswith("b") and insn.branch_target is not None and m not in ("bctr", "bctrl", "bclr", "blr", "bctrl")
            and (insn.word >> 26) & 0x3F == 16
        ):
            return self.emit_branch(insn, func)
        if m in ("bclr", "bctr", "bctrl", "blr", "beqlr", "bnelr", "bltlr", "bgtlr", "bgelr", "blelr", "bdnzlr", "bdzlr"):
            return self.emit_bcctr(insn)

        # --- integer D-form arithmetic ---
        if m in ("addi", "addis", "addic", "addic.", "subfic", "li", "lis"):
            if m == "li":
                rt = parse_register(ops[0]); imm = parse_simm16(ops[1])
                return [f"{rt} = (int64_t)({imm});"]
            if m == "lis":
                rt = parse_register(ops[0]); imm = parse_int(ops[1]) & 0xFFFF
                val = (imm << 16) if imm < 0x8000 else (imm - 0x10000) << 16
                return [f"{rt} = (int64_t)(int32_t)({(imm << 16) & 0xFFFFFFFF:#x}u);"]
            rt, ra, imm_tok = ops[0], ops[1], ops[2]
            imm = parse_simm16(imm_tok)
            if m == "addi":
                return [f"{parse_register(rt)} = (uint64_t)((int64_t){parse_register(ra)} + (int64_t){imm});"]
            if m == "addis":
                return [f"{parse_register(rt)} = (uint64_t)((int64_t){parse_register(ra)} + ((int64_t){imm} << 16));"]
            if m in ("addic", "addic."):
                r = parse_register(rt)
                return [
                    f"{{ uint64_t _ra = (uint64_t)(uint32_t){parse_register(ra)};",
                    f"  uint64_t _sum = _ra + (uint64_t)(uint32_t)(int64_t)({imm});",
                    f"  int32_t _r32 = (int32_t)(uint32_t)_sum;",
                    f"  {r} = (uint64_t)(int64_t)_r32;",
                    f"  g_xer = (g_xer & ~0x20000000u) | ((_sum > 0xFFFFFFFFull) ? 0x20000000u : 0u); }}",
                ]
            if m == "subfic":
                r = parse_register(rt)
                return [
                    f"{{ uint64_t _ra = (uint64_t)(uint32_t){parse_register(ra)};",
                    f"  uint64_t _r = (uint64_t)(uint32_t)(int64_t)({imm}) + ~_ra + 1u;",
                    f"  {r} = (uint64_t)(int64_t)(int32_t)(uint32_t)_r;",
                    f"  g_xer = (g_xer & ~0x20000000u) | ((_r > 0xFFFFFFFFull) ? 0x20000000u : 0u); }}",
                ]

        # --- X-form integer ---
        if m in ("add", "subf", "neg", "addex"):
            rt = parse_register(ops[0])
            if m == "neg":
                ra = parse_register(ops[1])
                return [f"{rt} = (uint64_t)(-(int64_t){ra});"]
            ra, rb = parse_register(ops[1]), parse_register(ops[2])
            if m == "add":
                return [f"{rt} = {ra} + {rb};"]
            if m == "subf":
                return [f"{rt} = {rb} - {ra};"]
            if m == "addex":
                return [
                    f"{{ uint64_t _a = {ra}, _b = {rb}; uint64_t _cin = (g_xer & 0x20000000u) ? 1 : 0;",
                    f"  {rt} = _a + _b + _cin;",
                    f"  g_xer = (g_xer & ~0x20000000u) | (((_b > ~_a) || (_b == ~_a && _cin)) ? 0x20000000u : 0u); }}",
                ]

        # --- logical ops ---
        if m in ("and", "or", "xor", "nor", "andc", "orc"):
            ra, rs, rb = parse_register(ops[0]), parse_register(ops[1]), parse_register(ops[2])
            expr = {"and": f"{ra} = {rs} & {rb};", "or": f"{ra} = {rs} | {rb};",
                    "xor": f"{ra} = {rs} ^ {rb};", "nor": f"{ra} = ~({rs} | {rb});",
                    "andc": f"{ra} = {rs} & ~{rb};", "orc": f"{ra} = {rs} | ~{rb};"}[m]
            return [expr]
        if m in ("mr", "mr."):
            ra, rs = parse_register(ops[0]), parse_register(ops[1])
            return [f"{ra} = {rs};"]
        if m in ("andi", "andis", "ori", "oris", "xori", "xoris", "andi.", "andis."):
            ra, rs = parse_register(ops[0]), parse_register(ops[1])
            imm = parse_int(ops[2]) & 0xFFFF
            shift = 16 if m.endswith("is") or m in ("andis", "andis.", "oris", "xoris") else 0
            mask = imm << shift
            op = {"and": "&", "ori": "|", "xori": "^", "andi": "&"}[m[:3].rstrip("is") if False else m.rstrip("is.").replace("andi", "and").replace("oris", "ori").replace("xoris", "xori")]
            op = {"andi": "&", "andis": "&", "andi.": "&", "andis.": "&", "ori": "|", "oris": "|", "xori": "^", "xoris": "^"}[m]
            lines = [f"{ra} = {rs} {op} 0x{mask:x}ull;"]
            if m.endswith("."):
                res = f"{ra}"
                lines.extend(cr_set_unsigned(0, f"(uint32_t){res}", "0"))
            return lines

        if m in ("extsb", "extsh", "extsw"):
            rt, ra = parse_register(ops[0]), parse_register(ops[1])
            bits = {"extsb": 8, "extsh": 16, "extsw": 32}[m]
            return [f"{rt} = (uint64_t)(int64_t)(int{bits}_t){ra};"]

        # --- multiply / divide ---
        if m == "mulli":
            rt, ra = parse_register(ops[0]), parse_register(ops[1])
            imm = parse_simm16(ops[2])
            return [f"{rt} = (uint64_t)((int64_t){ra} * (int64_t){imm});"]
        if m == "mullw":
            rt, ra, rb = parse_register(ops[0]), parse_register(ops[1]), parse_register(ops[2])
            return [f"{rt} = (uint64_t)(int64_t)(int32_t)((uint32_t){ra} * (uint32_t){rb});"]
        if m in ("divw", "divwu"):
            rt, ra, rb = parse_register(ops[0]), parse_register(ops[1]), parse_register(ops[2])
            divzero = f"((uint32_t){rb} == 0u)"
            if m == "divwu":
                return [f"{rt} = {divzero} ? 0ull : (uint64_t)(uint32_t)((uint32_t){ra} / (uint32_t){rb});"]
            return [f"{rt} = {divzero} ? 0ull : (uint64_t)(int64_t)(int32_t)((int32_t){ra} / (int32_t){rb});"]

        # --- compares ---
        if m in ("cmpi", "cmpwi", "cmplwi", "cmpdi", "cmpldi", "cmplw", "cmpw", "cmpl", "cmp"):
            return self._lift_compare(m, ops)

        # --- loads / stores ---
        if m in ("lbz", "lhz", "lwz", "lwa", "ld", "stb", "sth", "stw", "std"):
            return self._lift_memory(m, ops)

        # --- mtspr / mfspr / cr ---
        if m in ("mtspr", "mfspr"):
            return self._lift_spr(m, ops)
        if m == "mtlr":
            return [f"g_lr = {parse_register(ops[0])};"]
        if m == "mflr":
            return [f"{parse_register(ops[0])} = g_lr;"]
        if m == "mtctr":
            return [f"g_ctr = {parse_register(ops[0])};"]
        if m == "mfctr":
            return [f"{parse_register(ops[0])} = g_ctr;"]

        # --- floating point subset ---
        if m in ("lfd", "lfs", "stfd", "stfs", "fmr", "fadd", "fsub", "fmul", "fdiv", "frsp", "fabs", "fneg", "fmadd", "fcmpu"):
            return self._lift_fp(m, ops)

        # --- system ---
        if m == "sc":
            lev = parse_int(ops[0]) if ops else 0
            return [f"recomp_sc({lev});"]
        if m == "nop" or insn.word == 0x60000000:
            return []

        return self._unimpl(insn)

    # ---- compare ---------------------------------------------------------
    def _lift_compare(self, m: str, ops: list[str]) -> list[str]:
        unsigned = m.startswith("cmpl")
        is_imm = m in ("cmpi", "cmpwi", "cmplwi", "cmpdi", "cmpldi")
        # optional crfN prefix
        crf = 0
        body = list(ops)
        if body and body[0].lower().startswith("crf"):
            try:
                crf = int(body[0][3:])
            except ValueError:
                crf = 0
            body = body[1:]
        # optional L bit (0/1) for cmpi/cmp/cmpl families
        is_64 = m in ("cmpdi", "cmpldi", "cmp", "cmpl")
        if body and body[0] in ("0", "1") and len(body) > 2:
            if is_imm:
                is_64 = body[0] == "1"
                body = body[1:]
        if is_imm:
            ra = parse_register(body[0])
            imm = parse_int(body[1])
            if m == "cmplwi":
                imm &= 0xFFFF
                return cr_set_unsigned(crf, ra, f"0x{imm:x}ull")
            if m == "cmpwi":
                if imm >= 0x8000:
                    imm -= 0x10000
                return cr_set_signed(crf, f"(int64_t){ra}", str(imm))
            if m == "cmpi":
                simm = parse_simm16(body[1])
                if is_64:
                    return cr_set_signed(crf, f"(int64_t){ra}", str(simm))
                return cr_set_signed(crf, f"(int32_t){ra}", str(simm))
            if m in ("cmpdi", "cmpldi"):
                if unsigned:
                    return cr_set_unsigned(crf, ra, f"0x{imm & 0xFFFFFFFFFFFFFFFF:x}ull")
                return cr_set_signed(crf, f"(int64_t){ra}", str(imm))
            raise LifterError(f"bad immediate compare {m}")
        else:
            ra, rb = parse_register(body[0]), parse_register(body[1])
            if m == "cmpw":
                return cr_set_signed(crf, f"(int32_t){ra}", f"(int32_t){rb}")
            if m == "cmplw":
                return cr_set_unsigned(crf, f"(uint32_t){ra}", f"(uint32_t){rb}")
            if m == "cmp":
                return cr_set_signed(crf, f"(int64_t){ra}", f"(int64_t){rb}")
            if m == "cmpl":
                return cr_set_unsigned(crf, ra, rb)
            raise LifterError(f"bad register compare {m}")

    # ---- memory ----------------------------------------------------------
    def _lift_memory(self, m: str, ops: list[str]) -> list[str]:
        rt = parse_register(ops[0])
        offset, base = parse_mem_operand(ops[1])
        b = parse_register(base)
        addr = f"({b} + (int64_t){offset})" if offset >= 0 else f"({b} - (uint64_t){-offset})"
        if m == "lbz":
            return [f"{rt} = (uint64_t)recomp_load8({addr});"]
        if m == "lhz":
            return [f"{rt} = (uint64_t)recomp_load16({addr});"]
        if m == "lwz":
            return [f"{rt} = (uint64_t)recomp_load32({addr});"]
        if m == "lwa":
            return [f"{rt} = (uint64_t)(int64_t)(int32_t)recomp_load32({addr});"]
        if m == "ld":
            return [f"{rt} = recomp_load64({addr});"]
        if m == "stb":
            return [f"recomp_store8({addr}, (uint8_t){rt});"]
        if m == "sth":
            return [f"recomp_store16({addr}, (uint16_t){rt});"]
        if m == "stw":
            return [f"recomp_store32({addr}, (uint32_t){rt});"]
        if m == "std":
            return [f"recomp_store64({addr}, {rt});"]
        raise LifterError(m)

    # ---- SPRs -------------------------------------------------------------
    def _lift_spr(self, m: str, ops: list[str]) -> list[str]:
        if m == "mtspr":
            spr_tok, rt = ops[0], parse_register(ops[1])
            spr = self._parse_spr(spr_tok)
            if spr == 8:
                return [f"g_lr = {rt};"]
            if spr == 9:
                return [f"g_ctr = {rt};"]
            if spr == 1:
                return [f"g_xer = (uint32_t){rt};"]
            return [f'recomp_unimpl("mtspr {spr}", 0); /* mtspr {spr_tok}, {rt} */']
        else:
            rt, spr_tok = parse_register(ops[0]), ops[1]
            spr = self._parse_spr(spr_tok)
            if spr == 8:
                return [f"{rt} = g_lr;"]
            if spr == 9:
                return [f"{rt} = g_ctr;"]
            if spr == 1:
                return [f"{rt} = (uint64_t)g_xer;"]
            return [f'recomp_unimpl("mfspr {spr}", 0); /* mfspr {rt}, {spr_tok} */']

    def _parse_spr(self, token: str) -> int:
        t = token.strip().lower()
        names = {"lr": 8, "ctr": 9, "xer": 1, "fpscr": 0x100 + 0x00}
        if t in names:
            return names[t]
        return parse_int(t)

    # ---- floating point ---------------------------------------------------
    def _lift_fp(self, m: str, ops: list[str]) -> list[str]:
        if m in ("lfd", "lfs"):
            fd = parse_register(ops[0])
            offset, base = parse_mem_operand(ops[1])
            b = parse_register(base)
            addr = f"({b} + (int64_t){offset})" if offset >= 0 else f"({b} - (uint64_t){-offset})"
            op = "recomp_loadf64" if m == "lfd" else "recomp_loadf32_as_f64"
            return [f"{fd} = {op}({addr});"]
        if m in ("stfd", "stfs"):
            fd = parse_register(ops[0])
            offset, base = parse_mem_operand(ops[1])
            b = parse_register(base)
            addr = f"({b} + (int64_t){offset})" if offset >= 0 else f"({b} - (uint64_t){-offset})"
            op = "recomp_storef64" if m == "stfd" else "recomp_storef32_from_f64"
            return [f"{op}({addr}, {fd});"]
        if m == "fmr":
            return [f"{parse_register(ops[0])} = {parse_register(ops[1])};"]
        if m == "fabs":
            return [f"{parse_register(ops[0])} = __builtin_fabs({parse_register(ops[1])});"]
        if m == "fneg":
            return [f"{parse_register(ops[0])} = -{parse_register(ops[1])};"]
        if m == "frsp":
            return [f"{parse_register(ops[0])} = (double)(float){parse_register(ops[1])};"]
        if m in ("fadd", "fsub", "fmul", "fdiv"):
            fd, fa, fb = parse_register(ops[0]), parse_register(ops[1]), parse_register(ops[2])
            expr = {"fadd": f"{fa} + {fb}", "fsub": f"{fa} - {fb}", "fmul": f"{fa} * {fb}", "fdiv": f"{fa} / {fb}"}[m]
            return [f"{fd} = {expr};"]
        if m == "fmadd":
            fd, fa, fc, fe = parse_register(ops[0]), parse_register(ops[1]), parse_register(ops[2]), parse_register(ops[3])
            return [f"{fd} = {fa} * {fc} + {fe};"]
        if m == "fcmpu":
            # crf-prefixed compare of doubles; set CR field
            body = list(ops)
            crf = 0
            if body and body[0].lower().startswith("crf"):
                try:
                    crf = int(body[0][3:])
                except ValueError:
                    crf = 0
                body = body[1:]
            fa, fb = parse_register(body[0]), parse_register(body[1])
            return (
                cr_set_signed(crf, fa, fb)
            )
        return self._unimpl_fp(m, ops)

    def _unimpl_fp(self, m: str, ops: list[str]) -> list[str]:
        text = f"{m} {', '.join(ops)}".replace('"', "'")
        return [f'recomp_unimpl("{text}", 0);']


def lift_program(functions) -> list[str]:
    all_lines: list[str] = []
    # forward declarations so fragments may tail-call each other
    all_lines.append("/* forward declarations */")
    for start in sorted(functions):
        all_lines.append(f"void fn_{start:x}(void);")
    all_lines.append("")
    # indirect-call dispatch table (switch over known functions)
    all_lines.append("/* indirect call dispatch (recomp_icall) */")
    all_lines.append("int recomp_icall(uint64_t target) {")
    all_lines.append("  switch (target) {")
    for start in sorted(functions):
        all_lines.append(f"    case 0x{start:x}ull: fn_{start:x}(); return 1;")
    all_lines.append("    default: return 0;")
    all_lines.append("  }")
    all_lines.append("}")
    all_lines.append("")
    all_lines.append("/* tail call used by unconditional cross-function branches */")
    all_lines.append("void recomp_call_tail(uint64_t target) {")
    all_lines.append("  if (!recomp_icall(target)) {")
    all_lines.append('    recomp_unimpl("tailcall", (uint32_t)target);')
    all_lines.append("  }")
    all_lines.append("}")
    all_lines.append("")
    all_lines.append(f"/* entry point 0x{sorted(functions)[0]:x} */")
    for start in sorted(functions):
        lifter = Lifter(set())
        all_lines.extend(lifter.lift(functions[start]))
        all_lines.append("")
    return all_lines
