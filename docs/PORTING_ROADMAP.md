# Porting roadmap — from PPC subset to booted software

This is the honest plan for growing the **static recompiler** from "recompiles straight-line homebrew" to "boots an Xbox 360 homebrew title", and eventually retail games with your own dumped NAND. The pivot from a C++ interpreter to a PPC→WASM static recompiler is the architecture; everything below builds on it.

Reality constraints:

- **Retail boot is out of scope without your console.** Every Xbox 360 title is encrypted; xenia-canary needs your NAND flash dump and CPU key. The web build will never ship those, and neither should you.
- **The Xenos GPU is the hard part.** It is a unified-shader Xenon-era design; the practical path is a WebGPU compute reimplementation seeded from xenia's shader translator.
- **Static recompilation changes the calculus.** No interpreter fallback exists in shipping builds; unknown instructions must stay loud (trap with a PC), and the trap surface is the porting backlog.

## Milestones

### M0 — Recompiler core (done)
- Entry-point discovery → fragment lifter (64-bit integer ALU, rotate/mask/shift, load/store, branches incl. CTR loops and `bl`/`blr`) → validated WebAssembly emission.
- 64 MiB flat guest arena with the 0x03ffffff mask, MMIO console at 0x00F00000, exit door at 0x00F00FFC.
- Built-in samples (Hello / Fibonacci / Countdown) recompile and run in-page and under `node --test`.
- Project export: `guest.wasm` + segments + `file://` runner + manifest + kernel docs as one `.zip`.

### M1 — Coverage (next)
- FP arithmetic (fadd/fsub/fmul/fdiv/fmadd/fmsub/fnmadd/fnmsub), fcmp wired and runtime-tested against xenia semantics.
- Mulhd/mulhdu, `ldx`/`stwx` family audit, `endsWithControl` fall-through fix, more rotate/mask forms.
- Count traps by opcode across a corpus of homebrew; each closed trap class is a commit.

### M2 — Kernel HLE (xboxkrnl)
- Syscall/`sc` dispatch and XEX import-by-ordinal resolution; reimplement incrementally: thread create/suspend/resume, heap, TLS, synchronization objects, file I/O against browser storage.
- Milestone gate: a signed homebrew executable reaches its entry point and returns from `XeCrypt`/`Rtl*` stubs cleanly.

### M3 — AltiVec/VMX
- VMX decode + lift for the common forms (integer add/sub/mul, logical, permutes); FP VMX (`vaddfp` family) as the second wave.

### M4 — Xenos → WebGPU
- Shader translation: port xenia's shader peepholes to WGSL (or generate WGSL from the same HIR).
- Command processor emulation (ring buffer, fetch/decode), render target formats, tiling.
- Presenter: WebGPU canvas.

### M5 — Audio & input
- XMA decoding (ffmpeg→wasm drop-in or a clean decoder).
- Gamepad → XInput mapping.

### M6 — Threading
- Multi-core guest support via wasm threads + Atomics, or continuation-based interleaving on one thread.

### M7 — Homebrew milestone
- Boot a known-good free homebrew title on the web build; performance target: desktop-class recompiler output, no sustained memory growth over 30 minutes.

## Explicit non-goals

- Shipping, hosting, or linking to games, BIOS/NAND images, or Microsoft keys.
- Online Xbox Live functionality, ever.
- DRM circumvention. Inspecting your own dumps is fine; bypassing content protection is not, and this project does neither implicitly nor explicitly.
