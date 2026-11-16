"""Function discovery: breadth-first from the entry point.

Unlike a linear sweep, this never disassembles embedded data (strings,
jump tables) — only reachable code. Branch instructions are classified:

* direct `b` / `bl` targets            → new function entries
* conditional `bc*` targets            → intra-function block labels
  (if the target ends up outside the current fragment, the lifter emits
  a tail call instead)
* unconditional `b` ends a fragment; the target is a new function
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .disasm import Disassembler, Instruction

TERMINATOR_MNEMONICS = {"b", "blr", "bclr", "rfi", "rfid", "sc"}


@dataclass
class RecompiledFunction:
    start: int
    instructions: list[Instruction] = field(default_factory=list)

    @property
    def end(self) -> int:
        if not self.instructions:
            return self.start
        return self.instructions[-1].address + 4


def read_word_be(blob: bytes, offset: int) -> int:
    return int.from_bytes(blob[offset:offset + 4], "big")


def discover(blob: bytes, base: int, entry: int, *, text_end: int | None = None) -> dict[int, RecompiledFunction]:
    """Discover reachable functions. `blob` holds the image; base is the guest VA of blob[0].

    Returns a dict mapping function start VA → RecompiledFunction.
    """
    dis = Disassembler()
    image_size = len(blob)
    if text_end is None:
        text_end = base + image_size

    def file_off(va: int) -> int:
        return va - base

    def word_at(va: int) -> int | None:
        off = file_off(va)
        if off < 0 or off + 4 > image_size:
            return None
        return read_word_be(blob, off)

    functions: dict[int, RecompiledFunction] = {}
    worklist = [entry]
    while worklist:
        start = worklist.pop()
        if start in functions:
            continue
        func = RecompiledFunction(start=start)
        addr = start
        # Collect at most a sane number of blocks to bound pathological data.
        for _ in range(1 << 20):
            word = word_at(addr)
            if word is None:
                break
            insn = dis.decode_word(word, addr)
            if insn is None:
                break
            # Guard against decoding into the trailing data region: stop at
            # 4-alignment gaps past text_end.
            if addr + 4 > text_end:
                break
            func.instructions.append(insn)
            # Direct branches seed new functions.
            if insn.branch_target is not None and insn.branch_kind in ("b", "bl", "bc"):
                t = insn.branch_target
                if base <= t < text_end:
                    if t not in functions:
                        worklist.append(t)
            if insn.ends_function:
                addr += 4
                break
            addr += 4
        functions[start] = func
    return functions
