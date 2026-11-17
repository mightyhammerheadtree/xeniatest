# How the emulators and static recompilers handle the Xbox 360 kernel

Research collected while pivoting this repo to a static recompilation tool. The short version: **every serious Xbox 360 recompiler runs games on an HLE (high-level emulation) kernel reimplementation, not on the retail kernel**. The retail kernel (`xboxkrnl.exe`) and XAM (`xam.xex`) ship as PowerPC code too, but nobody statically recompiles them for production titles today — they reimplement the ~200–400 exports the title actually calls.

## 1. Xenia / xenia-canary (HLE, dynamic)

- Pure HLE. **No BIOS, firmware, NAND, or kernel file is loaded at any point** — Microsoft never shipped one for homebrew-compatible use, and the RetroBIOS summary of the xenia docs states this plainly ("Xenia doesn't require any Xbox 360 system files").
- Kernel modules loaded at startup: `xboxkrnl::XboxkrnlModule`, `xam::XamModule`, `xbdm::XbdmModule` (see `src/xenia/emulator.cc`, `LOAD_KERNEL_MODULE`).
- The kernel object model tracks objects (threads, events, mutants, files, …) in `kernel::KernelState`; exports are resolved by an export resolver against xboxkrnl/xam/xbdm tables.
- System calls transition guest → host; host native C++ executes and returns.
- References: [xenia architecture overview](https://www.mintlify.com/emoose/xenia/architecture/overview), [xenia emulator.cc](https://github.com/xenia-project/xenia/blob/master/src/xenia/emulator.cc), [Xenia Canary — RetroBIOS](https://abdess.github.io/retrobios/emulators/xenia/).

## 2. XenonRecomp (PPC → C++, static, hedge-dev)

- Tool only — **"This project does not provide a runtime implementation."** Every recompiled game ships its own runtime (UnleashedRecomp being the canonical example).
- PPC → C++: CPU state passed as an argument struct; 32-bit pointers; control flow compiled to native calls.
- **Indirect calls** (virtual calls, function pointers): a "perfect hash table" at runtime maps guest function addresses → recompiled function addresses; function addresses are placed after the valid XEX region in the base memory pointer (exposed as macros in `ppc_config.h`). Earlier approach used an 8 GB virtual allocation; abandoned.
- **Register-restore/save stubs**: 360 binaries contain specialized register restore/save helper functions (like switch-fallthrough shared epilogues). The recompiler needs their start addresses in a TOML config (`toml` project config) to handle them correctly — they look like switch statements and must be recompiled as fallthrough, not calls.
- UnleashedRecomp proves the model playable start-to-finish; CoD3Recomp implemented 179/179 imports for the game it targets; Forza 360 → 51k functions; all with per-project kernel/XAM layers.
- References: [github.com/hedge-dev/XenonRecomp](https://github.com/hedge-dev/XenonRecomp), [recompendium tool entry](https://recompendium.com/en/tool/xenonrecomp/).

## 3. ReXGlue SDK (xenia as link-time library, static)

- "Xenia's codebase wholesale and replaced its JIT backend with AOT mechanics." The kernel/memory/threading object model *originates from xenia* but with the JIT swapped for a **FunctionDispatcher** (guest→host dispatch table populated once at startup) and ExportResolver.
- Runtime derived from xenia: kernel objects, memory layout, threading primitives; graphics backends (D3D12/Vulkan) are Xenia's Xenos renderers compiled in. Kernel HLE, not LLE.
- v0.8.0+ CLI: `rexglue init --project-name X --xex-path game.xex` → TOML project → `rexglue codegen` → CMake link against `rex::kernel`, `rex::graphics`, …
- Explicitly **not a turnkey solution**: needs manual function-boundary specification, kernel import stubbing, per-title iteration.
- References: [ReXGlue SDK](https://github.com/rexglue/rexglue-sdk), [interview with the author](https://readonlymemo.com/rexglue-xbox-360-recompilation-interview/), [360tools playbook](https://github.com/sp00nznet/360tools).

## 4. xboxrecomp (the repo this project combined with — original Xbox, x86, not 360)

- Retargets the same pipeline idea to the **original Xbox** (x86, NV2A), producing a Windows executable via MSVC.
- Kernel exports are ordinal-driven: ~371 ordinals routed to C bridge functions (memory, file I/O, threading, sync, crypto, HAL). Unrouted imports used to fall through to silent return-0 — v0.9.0 made every one of them either a real bridge, a documented stub, or a loud failure. **Lesson adopted here: a missing kernel import must fail loudly, never silently succeed.**
- Pipeline: XBE parser → disasm → function ID → ABI analysis → lifter → C; runtime templates under `templates/runtime/`.

## 5. Consequences for this project — what the tool actually does now

1. **No kernel files, ever.** We HLE the kernel exactly like xenia: guest
   import-by-ordinal calls become host calls. The recompiler resolves the
   XEX2 `IMPORT_LIBRARIES` table (`0x000103FF`) at image load, maps every
   import thunk address to `{ module, ordinal, name }` via xenia's own
   export tables (2918 known exports across xboxkrnl/xam/xbdm,
   `public/js/recomp/hle_tables.js`), and compiles calls to those thunks
   into snapshot-and-call sequences — no "if you get a kernel" path, and
   no dump of any kind is needed or used.
2. **Retail images need no console and no keys.** The retail XEX2
   *container* key is a public constant shipped in xenia's source (16
   bytes, `20 B1 85 A5 9D 28 FD C3 40 58 3F BB 08 96 BF 91`) — it is the
   XEX container format's class constant, like a known compression header,
   not a secret. The per-image session key it unwraps is what actually
   encrypts the image body (AES-128-CBC, zero IV); BASIC decompression of
   the body is implemented, LZX/DELTA fail loudly. The devkit key (16
   zero bytes) is probed as a fallback, again exactly like xenia. The CPU
   key / keyvault is never involved: retail content encryption is a
   separate, untouched layer. See [HLE.md](HLE.md) and [LEGAL.md](LEGAL.md).
3. **Loud, named traps for everything not yet implemented.** Following
   xboxrecomp v0.9.0 and xenia's `UnimplementedExport`: each unimplemented
   export aborts with `unimplemented export: module!name (ordinal 0xNNN)`
   — never a silent 0. xenia's tables give us the names for all 2918
   exports, so the trap tells you exactly which export is missing.
   Implemented so far (xboxkrnl): DbgPrint (full variadic formatter),
   KeBugCheck(Ex), KeQuerySystemTime/PerformanceFrequency/InterruptTime,
   NtYieldExecution, RtlCompareMemory(Ulong), RtlFillMemoryUlong — see
   [HLE.md](HLE.md) for the ABI and the porting backlog.
4. **What ships today:**
   - console byte output (`0x00F00000` MMIO, homebrew convention; also what
     `DbgPrint` writes to),
   - exit door (`0x00F00FFC` store halts),
   - the HLE host (`public/js/recomp/hle.js`) + tables, wired through both
     the in-page runner and the exported project zip (node `run.js` and the
     `file://` page) — see
     [HLE.md](HLE.md) for the call ABI (r0..r31 big-endian snapshot at
     `0x00F11000`, args r3..r10 then stack, return in `r3`);
   - synthetic retail/devkit XEX2 images exercise the whole path in CI
     (decrypt → lift → run → DbgPrint output).
5. **LLE note, for completeness.** xboxkrnl.exe is PPC too and *could* be
   fed through the same recompiler as guest code (module-agnostic pipeline),
   which is the theoretical "load a real kernel" path. Nothing here
   requires it and nothing will ship it — xenia's HLE tables get us the
   same coverage with none of the IP entanglement.
6. **What we will NOT do**: fetch, ship, or link to Microsoft kernel/XAM
   binaries, game images, CPU keys, or vault blobs. Bring-your-own-dump
   only — and in the HLE model there is not even a dump to bring.

References: [xenia](https://github.com/xenia-canary/xenia-canary) · [XenonRecomp](https://github.com/hedge-dev/XenonRecomp) · [ReXGlue SDK](https://github.com/rexglue/rexglue-sdk) · [xboxrecomp](https://github.com/sp00nznet/xboxrecomp) · [xbox360-ports (list of real recomp projects)](https://github.com/CrownParkComputing/xbox360-ports)
