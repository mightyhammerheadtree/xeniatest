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

## 5. Consequences for this project

1. **No kernel files required to start.** We HLE the kernel: guest import-by-ordinal calls become direct calls to runtime functions (here: JS/WASM imports). The recompiler resolves the XEX2 import table (`0x000103FF`) and binding names/ordinals to stubs.
2. **Loud, complete stubs.** Following xboxrecomp v0.9.0: each unimplemented import aborts with the ordinal/name rather than returning 0. The runner prints how many imports the title references so you can see coverage before boot.
3. **Path to a real kernel, if desired.** The retail `xboxkrnl.exe` is PPC too — it *could* be fed through the same recompiler as guest code (LLE-style) and linked beside the game's recompiled image instead of HLE stubs. That is the "if you get a kernel" path: our pipeline is module-agnostic (any PPC image at any base address), but today the tool targets a homebrew/game image + HLE layer, matching what the whole ecosystem actually does.
4. **Kernel skeleton we ship** (this repo):
   - console byte output (`0x00F00000` MMIO, matching the homebrew convention),
   - exit door (`0x00F00FFC` store halts),
   - `recomp_unimpl` trap with address + ordinal, reported in the UI,
   - XEX2 import table enumeration displayed before running ("title wants N imports"),
   - extension point: a `kernel.conf` (JSON) mapping `{module, ordinal} → named stub`, reserved for your kernel work.
5. **What we will NOT do**: fetch, ship, or link to Microsoft kernel/XAM binaries, game images, or keys. Bring-your-own-dump only; this is law and also project policy.

References: [xenia](https://github.com/xenia-canary/xenia-canary) · [XenonRecomp](https://github.com/hedge-dev/XenonRecomp) · [ReXGlue SDK](https://github.com/rexglue/rexglue-sdk) · [xboxrecomp](https://github.com/sp00nznet/xboxrecomp) · [xbox360-ports (list of real recomp projects)](https://github.com/CrownParkComputing/xbox360-ports)
