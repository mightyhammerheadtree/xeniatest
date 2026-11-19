# Porting roadmap — from PPC subset to booted software

This is the honest plan for growing the **static recompiler** from "recompiles straight-line homebrew" to "boots an Xbox 360 homebrew title", and eventually retail games with your own dumped NAND. The pivot from a C++ interpreter to a PPC→WASM static recompiler is the architecture; everything below builds on it.

Reality constraints:

- **Retail keys are not the blocker.** Retail XEX2 images decrypt with the
  public retail container key (the same constant xenia ships); devkit images
  fall back to the devkit key automatically. BASIC decompression is in,
  LZX/DELTA fail loudly with an actionable message. What stands between a
  retail executable and running is the Xenos GPU and the HLE surface —
  both are porting work, not legal/key issues. (See [HLE.md](HLE.md) and
  [LEGAL.md](LEGAL.md).)
- **The Xenos GPU is the hard part.** It is a unified-shader Xenon-era design; the practical path is a WebGPU compute reimplementation seeded from xenia's shader translator.
- **Static recompilation changes the calculus.** No interpreter fallback exists in shipping builds; unknown instructions must stay loud (trap with a PC), and the trap surface is the porting backlog.

## Milestones

### M0 — Recompiler core (done)
- Entry-point discovery → fragment lifter (64-bit integer ALU, rotate/mask/shift, load/store, branches incl. CTR loops and `bl`/`blr`) → validated WebAssembly emission.
- XEX2 `.pdata`-seeded discovery (XenonRecomp-style): the decrypted image's PE section table is parsed, `.pdata` (`IMAGE_CE_RUNTIME_FUNCTION`, big-endian) contributes every `BeginAddress` as a function start, so functions reached only indirectly (function pointers, `bctr`) are still recompiled and exported. Seeds never include HLE import thunks; a missing/unparseable `.pdata` degrades to the previous control-flow-only discovery with a `peError` breadcrumb instead of failing the image.
- 64 MiB flat guest arena with the 0x03ffffff mask, MMIO console at 0x00F00000, exit door at 0x00F00FFC.
- Built-in samples (Hello / Fibonacci / Countdown) recompile and run in-page and under `node --test`.
- Project export: `guest.wasm` + segments + `file://` runner + manifest + HLE host (`hle.js`/`hle_tables.js`) + `docs/HLE.md` as one `.zip`; `node run.js` in the archive prints DbgPrint output end-to-end.
- Jump-table (switch) detection (XenonRecomp-style): the four XenonAnalyse shapes are matched in the code sections; labels seed discovery and a detected `bctr` dispatches via a CTR compare chain into the case fragments. Tables failing validation are dropped to the runtime-`icall` path (loud on unknown targets). Along the way this surfaced and fixed a lifter bug where M-form/MD-form instructions (`rlwinm`, `slwi`, `rlwimi`, `rlwnm`, `rld*`, `sradi`) stored their result into the rS field instead of the rA field — any cross-register rotate silently miscomputed.
- Save/restore stub detection + VMX64 stub synthesis (XenonRecomp-style): uniform `std`/`ld` (DS-form) and `stfd`/`lfd` (D-form) sequences ending at r31/f31 + `blr`, and the self-contained `li r11, slot; stvx|lvx(128) vN, r11, r12` pair sequences (vmx14: opcode 31 XO 103/231; vmx64: opcode 4 XO 195/451, VDS128 split register field) are matched in the code sections; every mid-entry is seeded. FPR/GPR stubs lift natively; VMX stubs are synthesized (save = zero-filled save slots, restore = no-op) since the lifter has no VMX model — real VMX compute instructions still trap loudly. Two more lifter silent-misrun bugs surfaced and were fixed: `ld`/`lfd` never returned memory contents (the `bswap64Seq` reads its value from the EA local; the load deposited into the other local, so both returned `bswap64(ea)`).

### M1 — Coverage (next)
- FP arithmetic (fadd/fsub/fmul/fdiv/fmadd/fmsub/fnmadd/fnmsub), fcmp wired and runtime-tested against xenia semantics.
- Mulhd/mulhdu, `ldx`/`stwx` family audit, `endsWithControl` fall-through fix, more rotate/mask forms.
- Count traps by opcode across a corpus of homebrew; each closed trap class is a commit.

### M2 — Kernel HLE (xboxkrnl) — **in progress**
- ✅ XEX2 `IMPORT_LIBRARIES` parsing; import thunks resolved to `{ module, ordinal, name }` via xenia's export tables (2918 known exports) at image load.
- ✅ Retail-key decrypt + BASIC decompress; devkit fallback; loud LZX/DELTA rejection.
- ✅ HLE host (`hle.js`): full PPC context snapshot (r0..r31, big-endian) before every import call; args r3..r10 then stack; return in `r3` (exported wasm global, host-writes).
- ✅ Implemented (xboxkrnl): DbgPrint (variadic formatter), KeBugCheck(Ex), KeQuerySystemTime/PerformanceFrequency/InterruptTime, NtYieldExecution, RtlCompareMemory(Ulong), RtlFillMemoryUlong.
- ✅ Exported-zip parity: the same host wiring runs in-page, in node `run.js`, and in the `file://` runner (synthetic retail/devkit XEX2 in CI).
- Next increments: thread create/suspend/resume, heap (RtlAllocateHeap family), TLS, synchronization objects, file I/O against browser storage; `sc` dispatch once a guest needs it.
- Milestone gate: a signed homebrew executable reaches its entry point and returns from `XeCrypt`/`Rtl*` stubs cleanly.

### M3 — AltiVec/VMX
- VMX decode + lift for the common forms (integer add/sub/mul, logical, permutes); FP VMX (`vaddfp` family) as the second wave.

### M4 — Xenos → WebGPU — **in progress**
- ✅ Register file (reset defaults, write log) + generated register table from xenia (3435 registers).
- ✅ PM4 command processor: packet decode, SET_CONSTANT windows, REG_RMW (incl. register operands), MEM_WRITE / REG_TO_MEM with xenia GpuSwap32 endianness, EVENT_WRITE family (counter + WRITEBACK window mapping), DRAW_INDX decode (kDMA/kAutoIndex; kImmediate traps as in xenia), loud traps for everything else.
- ✅ WebGPU plumbing: adapter/device/canvas configuration, depth attachment, index-buffer upload (BE→swap→LE), present; bring-up triangle behind explicit `{ placeholderShader: true }` — never silently rendered.
- ✅ Ring buffer: host-side ring (`VdInitializeRingBuffer` / writeback / CP_RB_WPTR doorbell, wraparound + overrun trap) wired to the HLE Vd* exports and an in-page `XenonGpuHost`; end-to-end tested.
- ✅ Vertex fetch decode: `FETCH_*` constants, ucode walker, `GPUVertexFormat` mapping, `GPUVertexBufferLayout` descriptors — formats WebGPU can't represent trap.
- ✅ Vertex buffer binding on the fetch bring-up path (behind `{ placeholderShader: true }`, per draw, no caching): CP snapshots 192 fetch constants → `prepareVertexFetch()` (decode → analyze → descriptors) → `uploadVertexBuffers()` (GpuSwap32 BE→LE, VERTEX|COPY_DST) → fetch placeholder pipeline + `setVertexBuffer` + indexed-draw support.
- ✅ Performance pass on the bring-up paths (content-addressed vertex/index buffer cache with LRU + byte cap; contiguous-run ring drain — ~3–4× faster on micro-benchmarks; allocation-free register writes by default). See `tools/bench_gpu_hot_paths.mjs`.
- ✅ Vertex-shader ALU → WGSL translator (`wgsl_translate.js`, v1 subset): xenia-faithful decode of the ucode control-flow walker + ALU (vector/scalar dual-issue lanes), `fc[]`/bool constant bindings, instruction predication (`p0`) and `cexec` CF predication (`bc(n)`), select-based `max`/`min`, manual `dp4/dp3/dp2add`, `setp_*` → `p0`/`ps`, dual-issue export lane reconciliation (both-halves → `1.0`, neither-half + scalar-dest-relative → `0.0`), interpolator/position exports. Anything outside the subset (loops, calls/jumps, tfetch, memory exports, register-relative `+aL` addressing, transcendentals, unassigned opcodes) traps loudly. Every generated shader is compiled by **Dawn** in tests (`test_wgsl_translate.mjs`); the translator is not yet wired into `draw()` — the placeholder gate remains.
- ✅ GPU MMIO aperture in the recompiler (`[0x00F02000, 0x00F03000)`, byte offset = register × 4; CP_RB_WPTR drains the ring, CP_RB_RPTR reads back the read pointer, everything else traps) — recompiled guests can ring the doorbell end-to-end. Still needed for real titles: remap the retail GPU BAR onto the aperture (HLE), shader HIR → WGSL (replaces the placeholder gate); render targets/tiling; textures. See [XENOS_GPU.md](XENOS_GPU.md) for the bring-up order.

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
