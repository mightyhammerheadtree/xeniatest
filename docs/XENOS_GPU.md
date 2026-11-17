# The Xenos GPU on WebGPU

The Xbox 360's Xenos GPU is the single biggest blocker between "recompiled homebrew" and "boots a title". This document tracks the port: what exists, what is honest about not working yet, and the order in which pieces come up. The architecture is xenia's — a register file driven by a PM4 command processor that emits draws against a unified shader core — re-hosted on WebGPU instead of D3D12/Vulkan.

xenia-canary (BSD-3-Clause) is the reference for every semantic decision in this directory. The relevant upstream files live in `gpu/` upstream: `command_processor.cc`, `register_file.cc/h`, `xenos.h` (register table + `GpuSwap32`), `pm4_command_processor_implement.h`, and the `shader_translator.*` family. Where this port deviates, it is called out in a comment next to the code.

## Layout

```
public/js/gpu/xenos_registers.js   GENERATED register table (3435 registers, reset defaults) — do not edit;
                                   regenerate with: node tools/gen_gpu_registers.mjs [path-to-register_table.inc]
public/js/gpu/register_file.js     20,483-entry dword register file, reset defaults, write log (xenia RegisterFile)
public/js/gpu/pm4.js               PM4 packet make/parse (type 0/1/2/3), opcodes, enums, gpuSwap32 (xenia GpuSwap32)
public/js/gpu/command_processor.js XenosCommandProcessor: executes a guest command buffer against the register file,
                                   decodes DRAW_INDX / SET_CONSTANT / REG_RMW / MEM_WRITE / REG_TO_MEM /
                                   EVENT_WRITE family; throws XenosGpuError naming the packet on anything it
                                   cannot do — never silently drops a packet
public/js/gpu/webgpu_backend.js    Browser backend: lazy attach() (adapter → device → canvas(webgpu) configure),
                                   depth24plus attachment, index-buffer upload from guest memory (BE→swap→LE),
                                   render-pass encoding, placeholder bring-up triangle behind explicit opt-in
core/tests/test_gpu_pm4.mjs        Command-processor semantics, hand-assembling real BE packets
core/tests/test_gpu_webgpu.mjs     Backend honesty paths (no device in node: attach error, memWrite plumbing,
                                   loud traps) — no mocks of the pipeline itself
tools/gen_gpu_registers.mjs        xenos register_table.inc → xenos_registers.js
```

## What works today (measured, not aspirational)

- **PM4 packet decode** for the full type-0/1/2/3 surface xenia implements on the CPU, including bit-exact `GpuSwap32` endianness handling (kNone/k8in16/k8in32/k16in32) for `MEM_WRITE` / `REG_TO_MEM` / index data.
- **Register file** with xenia's reset defaults and an ordered write log; reads of out-of-range indices throw instead of masking.
- **DRAW_INDX decode** exactly as xenia's `ExecutePacketType3Draw`: VGT_DRAW_INITIATOR (0x21FC) fields, `IsMajorModeExplicit`, kDMA index fetch (`VGT_DMA_BASE` 29-bit GPU bus mask, swap mode from `VGT_DMA_SIZE`) with immediate/auto-index/immediate-source handling; kImmediate traps (xenia does not support it either).
- **SET_CONSTANT** for ALU / fetch / bool / loop / REGISTERS constants, matching xenia's base addresses (ALU 0x4000, fetch 0x4800, bool 0x4900, loop 0x4908, context registers at 0x2000).
- **EVENT_WRITE / EVENT_WRITE_SHD**: initiator low 6 bits → `VGT_EVENT_INITIATOR`, `0x80000000` bit selects GPU-counter-write semantics (counter increments per event), the WRITEBACK_START/SIZE window (0x7F000000+off) translation for out-of-bus writeback addresses, and the 16/32/64-bit counter/payload sizes with GpuSwap endianness. `EVENT_WRITE_CFL/EXT/ZPD` are forwarded to the backend (`eventWrite`), not swallowed.
- **REG_RMW** including register-operand and_mask/or_mask bits (0x80000000/0x40000000), with the 13-bit register mask xenia applies.
- **WebGPU plumbing**: adapter/device request, canvas configuration, depth texture, index-buffer upload with BE swap, present on `swap()`. The triangle placeholder exists so the pipeline can be visually verified once a browser is attached — it is *not* the guest shader and is only reachable with `{ placeholderShader: true }` passed explicitly.
- **Loud failure everywhere else.** Unknown opcodes, unsupported addressing modes, draws before `attach()`: all throw `XenosGpuError`/`XenosWebGpuError` with the packet/PC/register named. Nothing is silently miscompiled — same discipline as the PPC recompiler.

## What does not work yet (the honest list)

1. **Guest shader translation.** Xenos microcode → WGSL does not exist yet. This is the bulk of M4. xenia's path is `shader_translator.cc` + the DXBC/SPIRV backends; the plan for this port is a WGSL backend off the same HIR, or a port of the peephole translators once the HIR shape is pinned down.
2. **Draws without `placeholderShader: true` refuse to render.** That refusal is deliberate. When the translator lands, the gate becomes "both shaders translated successfully" instead of the opt-in.
3. **Render targets, tiling, clears.** `RB_COLOR*`/`SCISSOR` registers are tracked in the file but no RT setup, tiling-mode handling, or MSAA resolve exists.
4. **Vertex fetch / shader fetch constants.** `FETCH_*` and `BOOL_*`/`LOOP_*` are decoded into the register file but not yet consumed by any fetcher.
5. **Ring buffer consumption.** The CP executes a caller-provided byte range (`executeCommandBuffer`); wiring a real `CP_RB_BASE/RPTR/WPTR` ring fed from guest memory + MMIO doorbell is not done.
6. **Textures, samplers, query (VIZ_QUERY), predication side effects** beyond register tracking.
7. **Browser-only.** Everything above runs in a real browser; node is used only for the PM4/CP/backend-honesty tests (no `navigator.gpu`). There is no headless WebGPU in CI yet.

## Bring-up order

1. ✅ Register file + PM4 decode with xenia-verified semantics (this file's `core/tests/test_gpu_pm4.mjs`).
2. ✅ WebGPU device/canvas/depth/index-upload plumbing (`webgpu_backend.js`), placeholder triangle behind explicit opt-in.
3. ⬜ Ring buffer: `CP_RB_*` from the guest, `executeCommandBuffer` fed per doorbell, overflow/kickoff handling.
4. ⬜ Vertex fetch: `FETCH_*` constants → vertex attribute layout → `GPUVertexBufferLayout`.
5. ⬜ Shader HIR + WGSL codegen (start: ALU-only vertex shaders, flat + no textures) → replace the placeholder gate in `draw()`.
6. ⬜ Pixel shaders + render target formats (`RB_COLOR_INFO`), blend, depth/stencil.
7. ⬜ Tiling-mode handling and clears (`RB_COLOR_CLEAR`), scissor.
8. ⬜ Textures + samplers (`VTXFETCH`, `SET_SAMPLER`), addressing modes.
9. ⬜ Queries (VIZ_QUERY), predication, EVENT_WRITE consumers (e.g. XMA fences).
10. ⬜ Boot a real homebrew title end-to-end; performance pass (index-buffer reuse, pipeline cache).

Each step lands as its own commit with tests; `npm test` must stay green.

## Testing

```
npm run test:gpu     # core/tests/test_gpu_pm4.mjs + test_gpu_webgpu.mjs
```

- `test_gpu_pm4.mjs` assembles real big-endian PM4 streams and executes them through the same modules the browser loads. It covers packet round-trips, constant windows, endianness rules, EVENT_WRITE counter/writeback mapping, DRAW_INDX decode, and predication.
- `test_gpu_webgpu.mjs` covers what can run without a GPU: constructor validation, the `attach()` error path in node, `memWrite` routing into guest memory, and loud traps (draw before attach). It does not mock a device.
- Anything touching an actual device (placeholder triangle, canvas present) is verified manually in Chrome/Edge until a headless WebGPU implementation is available; that checklist is this document, updated as bring-up advances.

## Updating the register table

`xenos_registers.js` is generated from xenia's `register_table.inc`:

```
node tools/gen_gpu_registers.mjs            # reads .xenia-ref/gpu/register_table.inc if present
node tools/gen_gpu_registers.mjs path/to/register_table.inc
```

`.xenia-ref/` is git-ignored (reference-only). Do not hand-edit the generated file; regenerate it and commit the diff.
