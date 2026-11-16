"""Disassembly stage: capstone PPC64 big-endian wrapper with operand helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from capstone import Cs, CS_ARCH_PPC, CS_MODE_64, CS_MODE_BIG_ENDIAN


@dataclass
class Instruction:
    address: int
    word: int
    mnemonic: str
    op_str: str
    branch_target: int | None = None     # absolute target for b/bl/bc when it fits
    branch_kind: str | None = None       # 'b' | 'bl' | 'bc' | 'bcctr-ct' (bclr) | None
    ends_function: bool = False


class Disassembler:
    def __init__(self):
        self.md = Cs(CS_ARCH_PPC, CS_MODE_64 + CS_MODE_BIG_ENDIAN)

    def decode_word(self, word: int, address: int) -> Instruction | None:
        code = word.to_bytes(4, "big")
        for insn in self.md.disasm(code, address):
            return self._wrap(insn, word)
        return None

    def _wrap(self, insn, word: int) -> Instruction:
        mnem = insn.mnemonic.lower()
        target = parse_branch_target(insn.op_str)
        if mnem in ("b", "bl"):
            kind = mnem
        elif mnem.startswith("b") and target is not None and (
            mnem.startswith("beq") or mnem.startswith("bne") or mnem.startswith("blt")
            or mnem.startswith("bgt") or mnem.startswith("bge") or mnem.startswith("ble")
            or mnem.startswith("bdnz") or mnem.startswith("bdz") or mnem == "bc"
        ):
            kind = "bc"
        elif mnem in ("bclr", "blrl", "beqlr", "bnelr", "bltlr", "bgtlr", "bgelr", "blelr", "blr"):
            kind = "bcctr"
            target = None
        elif mnem in ("bctr", "bctrl"):
            kind = "bcctr"
            target = None
        else:
            kind = None
        ends = mnem in ("b", "blr", "bclr", "rfi", "rfid", "sc")
        return Instruction(
            address=insn.address,
            word=word,
            mnemonic=mnem,
            op_str=insn.op_str,
            branch_target=target,
            branch_kind=kind,
            ends_function=ends,
        )


_TARGET_RE = re.compile(r"0x[0-9a-fA-F]+")
_MEM_RE = re.compile(r"^([+-]?(?:0x[0-9a-fA-F]+|\d+)?)\s*\(\s*(r\d+|[a-z]+)\s*\)$")


def parse_branch_target(op_str: str):
    """Return the absolute target of a direct branch, or None."""
    m = _TARGET_RE.search(op_str)
    if not m:
        return None
    try:
        return int(m.group(0), 16)
    except ValueError:
        return None


def parse_mem_operand(token: str):
    """Parse '16(r12)' / '0(r3)' → (offset:int, base:str). Raises ValueError otherwise."""
    m = _MEM_RE.match(token.strip())
    if not m:
        raise ValueError(f"bad memory operand: {token!r}")
    off_s, base = m.groups()
    off_s = (off_s or "0").strip()
    if off_s.startswith(("+", "-")) or off_s.isdigit() is False:
        offset = int(off_s, 0)
    else:
        offset = int(off_s, 0) if off_s.startswith("0x") else int(off_s)
    return offset, base


def split_operands(op_str: str):
    """Split at top-level commas (parens/quoted strings aware)."""
    parts, depth, cur, in_quote = [], 0, "", False
    for ch in op_str:
        if ch == '"':
            in_quote = not in_quote
            cur += ch
            continue
        if not in_quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append(cur.strip())
                cur = ""
                continue
        cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_register(token: str):
    t = token.strip().lower()
    if re.fullmatch(r"r(3[01]|[12]\d|\d)", t):
        return t
    if t in ("lr", "ctr", "xer", "cr"):
        return t
    raise ValueError(f"bad register: {token!r}")


def parse_int(token: str) -> int:
    t = token.strip().lower().replace("u", "")
    neg = t.startswith("-")
    body = t[1:] if neg else t
    if body.startswith("0x"):
        value = int(body, 16)
    else:
        value = int(body, 10)
    return -value if neg else value


def parse_simm16(token: str) -> int:
    v = parse_int(token)
    if v >= 0x8000:
        v -= 0x10000
    return v
