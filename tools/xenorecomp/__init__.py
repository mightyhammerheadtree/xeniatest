"""
xenorecomp — Xbox 360 static recompilation to the web.

A pipeline modeled on xboxrecomp (parse → disasm → lift → emit), retargeted
from "original Xbox x86 → C → Windows .exe" to "Xbox 360 PPC (PowerPC64,
big-endian) → C → WebAssembly". The generated project is a static binary with
the kernel minimally HLE'd for the guest's used subset, compiled to WASM and
loaded by the browser host in this repository.

xenia-canary (via this repo's WASM core) remains the interpreter/validator;
xenorecomp is the native path.
"""

__version__ = "0.1.0"
