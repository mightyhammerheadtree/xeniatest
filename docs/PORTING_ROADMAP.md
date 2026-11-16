# Porting roadmap — Xenia Canary → the browser

This is the honest plan for turning the web port from "a great PPC core with a pretty shell" into something that can actually boot an Xbox 360 homebrew title, and eventually retail games with your own dumped NAND.

Reality constraints:

- **Retail boot is out of scope without your console.** Every Xbox 360 title is encrypted; xenia-canary needs your NAND flash dump and CPU key. The web port will never ship those, and neither should you.
- **The Xenos GPU is the hard part.** It is a unified-shader Xenon-era design; the practical path is a HLSL/GLSL-free reimplementation targeting WebGPU compute, seeded from xenia's shader translator.
- **xenia is a JITted, threaded emulator.** The web core starts as an interpreter, then becomes a baseline JIT on wasm threads (SharedArrayBuffer + Atomics), which is why the server already sets `Cross-Origin-Opener-Policy`/`Cross-Origin-Embedder-Policy`.

## Milestones

### M0 — PPC core sandbox (done)
- xenia-canary opcode table + decoder ported to C99, compiled to WASM.
- Interpreter for the 64-bit integer/load-store/branch subset; `sc` exit door; MMIO console.
- 64 MiB fixed guest RAM; fixed WASM memory; zero OOMs by construction.
- node:test suites execute assembled programs and assert architectural state.

### M1 — Guest address model (next)
- Reservation of a 2 GiB guest address window (sparse, page-committed) instead of the flat 64 MiB arena.
- 64-bit virtual → guest-physical translation blocks (BAT/page-table emulation enough for homebrew).
- ELF/XEX loaders writing into the translated window (kernel-side, in C, beside the core).

### M2 — Kernel HLE (xboxkrnl)
- Syscall table from xenia (`xboxkrnl_*.cc`), reimplemented incrementally: thread create/suspend/resume, heap, TLS, synchronization objects (mutants/events/semaphores), file I/O against OPFS (the library *is* the filesystem).
- XEX import-by-ordinal resolution and static library initialization (xenia's `user_module.cc` model).
- Milestone gate: boot a signed homebrew executable to its entry point and return from `XeCrypt`/`Rtl*` stubs cleanly.

### M3 — AltiVec/VMX + FP
- VMX execution (the interpreter already traps and reports these; see `status == 2`).
- Correct FP including the NaN/infinit Y semantics — validate against xenia's own test ROMs.

### M4 — Xenos → WebGPU
- Shader translation: port xenia's shader peepholes to WGSL (or generate WGSL from the same HIR).
- Command processor emulation (ring buffer, fetch/decode), render target formats, tilers.
- Presenter: WebGPU canvas, 1280×720@30 target, vsync via rAF alignment.

### M5 — Audio & input
- XMA decoding (port xenia's XMA via ffmpeg→wasm drop-in, or a clean decoder).
- Gamepad → XInput mapping (the runner already detects pads).

### M6 — Threading & performance
- Baseline JIT: single-pass PPC→WASM translation for hot blocks (B3+ style), with the interpreter as fallback and validator.
- wasm threads + Atomics for the 3 Xenon cores' worth of threads the kernel actually spawns.
- Memory: keep the guest in a fixed max-size linear memory; never materialize textures/ISOs outside it.

### M7 — Homebrew milestone
- Boot [Xenia's own test programs](https://github.com/xenia-canary/xenia-canary) and a known-good free homebrew title (e.g. cube field / Flappy Xbox) on the web build.
- Performance target: ≥ 60% of desktop interpreter speed on a mid-range laptop; no sustained memory growth over 30 minutes.

## Explicit non-goals

- Shipping, hosting, or linking to games, BIOS/NAND images, or Microsoft keys.
- Online Xbox Live functionality, ever.
- DRM circumvention. Inspecting your own dumps is fine; bypassing content protection is not, and this project does neither implicitly nor explicitly.
