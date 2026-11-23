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
- ✅ PM4 command processor: packet decode, SET_CONSTANT windows, REG_RMW (incl. register operands), MEM_WRITE / REG_TO_MEM with xenia GpuSwap32 endianness, EVENT_WRITE family (the frame counter is bumped by the swap, as in xenia; the WRITEBACK window mapping; the screen-extent writeback), WAIT_REG_MEM with xenia's comparison rules (a condition that would make xenia block refuses loudly here), DRAW_INDX decode (kDMA/kAutoIndex; kImmediate traps as in xenia), loud traps for everything else.
- ✅ WebGPU plumbing: adapter/device/canvas configuration, depth attachment, index-buffer upload (BE→swap→LE), present; bring-up triangle behind explicit `{ placeholderShader: true }` — never silently rendered.
- ✅ Ring buffer: host-side ring (`VdInitializeRingBuffer` / writeback / CP_RB_WPTR doorbell, wraparound + overrun trap) wired to the HLE Vd* exports and an in-page `XenonGpuHost`; end-to-end tested.
- ✅ Vertex fetch decode: `FETCH_*` constants, ucode walker, `GPUVertexFormat` mapping, `GPUVertexBufferLayout` descriptors — formats WebGPU can't represent trap.
- ✅ Vertex buffer binding on the fetch bring-up path (behind `{ placeholderShader: true }`, per draw, no caching): CP snapshots 192 fetch constants → `prepareVertexFetch()` (decode → analyze → descriptors) → `uploadVertexBuffers()` (GpuSwap32 BE→LE, VERTEX|COPY_DST) → fetch placeholder pipeline + `setVertexBuffer` + indexed-draw support.
- ✅ Performance pass on the bring-up paths (content-addressed vertex/index buffer cache with LRU + byte cap; contiguous-run ring drain — ~3–4× faster on micro-benchmarks; allocation-free register writes by default). See `tools/bench_gpu_hot_paths.mjs`.
- ✅ Vertex-shader ALU → WGSL translator (`wgsl_translate.js`, v1 subset): xenia-faithful decode of the ucode control-flow walker + ALU (vector/scalar dual-issue lanes), `fc[]`/bool constant bindings, instruction predication (`p0`) and `cexec` CF predication (`bc(n)`), select-based `max`/`min`, manual `dp4/dp3/dp2add`, `setp_*` → `p0`/`ps`, dual-issue export lane reconciliation (both-halves → `1.0`, neither-half + scalar-dest-relative → `0.0`), interpolator/position exports. Anything outside the subset (loops, calls/jumps, tfetch, memory exports, register-relative `+aL` addressing, transcendentals, unassigned opcodes) traps loudly. Pixel shaders decode with the same machinery: `oC0` export, interpolator inputs linked to the lanes the VS exports, kills → `discard`. Every generated shader is compiled by **Dawn** in tests (`test_wgsl_translate.mjs`), and translated VS+PS pairs are link-checked with an actual null-backend pipeline; the **whole program is wired into `draw()`** — both stages run translated (pipeline/bind-group cached per shader pair+layout+topology, ALU/bool constants shared and content-hashed into persistent uniform buffers), with the fixed bring-up FS only when no pixel shader is loaded. The `oDepth` export works (`@builtin(frag_depth)`, [0,1]-clamped). The `PA_SC_*` scissor rect is honored on the canvas target (`setScissorRect`, xenia `GetScissor` semantics, clamped to the attachment — a guest scissor on a larger guest RT would clip wrong, which is part of step 7). `RB_COLOR_MASK` RT0 writes, `RB_BLENDCONTROL0` blend (+ IEEE-754 constants via `setBlendConstant`), `RB_DEPTHCONTROL` depth/stencil on a `depth24plus-stencil8` attachment (the stencil reference/mask pair is selected exactly like xenia's, the per-face ops only for two-faced primitives, and `PA_CL_CLIP_CNTL.clip_disable` → `depthClipEnabled`), the alpha test (discard in the translated guest PS against an IEEE-754 `RB_ALPHA_REF` uniform, only when the PS exports oC0 — xenia's rule), and `k_8_8_8_8_GAMMA` RT0 (oC0 RGB gamma-encoded in the translated PS with xenia's exact PWL piecewise — alpha untouched, blending on gamma values like xenia's RTV path) are honored exactly (cache-keyed with the pipeline; the reference value flows through its own content-hashed uniform, never the WGSL); what is not representable — `oC1–3` MRT, color formats other than k_8_8_8_8/k_8_8_8_8_GAMMA, kD24FS8 float depth *with stencil* (WebGPU has no float depth format with a stencil plane; without stencil it renders into `depth32float`), 2x guest MSAA — traps loudly with a named reason (EDRAM render-target cache, step 7, covers MRT/clears/EDRAM-depth).
- ✅ Guest **viewport/VTE** + **rasterizer state** + **per-frame draw batching** + **host-side primitive conversion**: WebGPU has no programmable clip-space viewport, so the guest viewport (`PA_CL_VTE_CNTL` / `PA_CL_CLIP_CNTL` / `PA_CL_VPORT_*`) is decoded host-side and applied in the translated vertex shader (pre-w-divide, xenia's clipping-disabled formula; the GL-style z remap with xenia's exact saturate bounds; user clip planes trap loudly). The rasterizer (`PA_SU_SC_MODE_CNTL` + `PA_SU_POLY_OFFSET_*`) decodes cull mode (gated on xenia's `IsPrimitivePolygonal`), front-face winding, and polygon offset into WebGPU's `cullMode`/`frontFace`/`depthBias*` with xenia's exact offset factors; both faces culled drops the draw, and the kDualMode wireframe mode traps loudly. All draws between two `swap()`s now batch into **one** encoder + render pass (per-draw dynamic state inside the open pass; a trap drops the pending frame without submitting). The primitive types WebGPU has no topology for — **kTriangleFan, kLineLoop, kQuadList** — are converted host-side exactly like xenia's primitive processor (`TriangleFanToList` / `LineLoopToStrip` / `QuadListToTriangleList`, same index counts and ordering), with the guest endian swap applied before the expansion; non-indexed draws of those types get a synthesized, memoized sequential index buffer. The types xenia's primitive processor itself refuses (`kTriangleWithWFlags`, `kQuadStrip`, `kPolygon`, `kRectangleList` without VS expansion) still trap loudly.
- ✅ Guest **color render-target formats**: `RB_COLOR_INFO`'s format maps to a WebGPU color target exactly like xenia maps it to a host format (`GetColorDXGIFormat` + `GetBaseColorFormat`) — `k_2_10_10_10` → `rgb10a2unorm`, the float formats → `rgba16float`/`rg16float`/`r32float`/`rg32float`, `k_8_8_8_8`/`k_8_8_8_8_GAMMA` → the canvas itself. A format the canvas cannot carry renders into an offscreen texture of exactly that format (so the quantization and range match the hardware's) and is blitted to the canvas at `swap()`; the guest's write mask is restricted to the channels the format has. Loudly traps: the 16-bit snorm formats (WebGPU's optional `texture-formats-tier1`) and blending a float32 target (WebGPU's optional `float32-blendable`).
- ✅ Guest **MSAA** (`RB_SURFACE_INFO.msaa_samples`): k4X renders into a 4× multisampled color + depth attachment and resolves into the single-sample target (WebGPU's box-filter resolve — the same filter xenia's EDRAM resolve applies; WebGPU's 4× sample positions are the standard D3D ones xenia's host uses too). k2X traps loudly: WebGPU's multisample count must be 1 or 4.
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
