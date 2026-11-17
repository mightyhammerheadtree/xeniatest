# Xenon Recompiler — Xbox 360 → WebAssembly, in your browser

A **static recompiler** for Xbox 360 (Xenon PPC64, big-endian) executables that runs entirely client-side. It lifts PPC machine code to WebAssembly at click-time, executes the result in the page, and can export the recompiled program as a standalone project `.zip` (double-clickable runner, no server required).

**No accounts, no telemetry, no uploads — your binary never leaves the machine.**

> Derivative work of [xenia-canary](https://github.com/xenia-canary/xenia-canary) (decoder semantics), not an emulator. You bring the binary; the recompiler does the rest.

## What works today (tested)

- **Static recompilation, end to end** — discovery of reachable code from the entry point, fragment lifting (integer ALU, rotate/mask/shift, 32/64-bit loads/stores, branches incl. CTR loops & link register, `bl`/`blr` calls, `sc`, FP fcmpo/fcmpu decode), wasm emission, instantiation, execution.
- **Built-in samples** — Hello, Fibonacci (`fib(25) = 75025`) and Countdown are real hand-assembled PPC binaries recompiled in-page; they print through the guest MMIO console and halt via the exit door.
- **In-page run** — drop a `.xex`, `.elf` or raw `.bin`; it recompiles and runs with one click, output shown in the on-page console.
- **Project export** — one click downloads a `.zip` containing the `guest.wasm`, the image segments, a `file://`-double-clickable runner, a Node runner (`run.js`), `manifest.json`, the pure-HLE kernel (`hle.js` + `hle_tables.js`, byte-exact copies) and `docs/HLE.md`. Works offline, nothing to install.
- **XEX2 inspection & loading** — header, optional headers, security info, page descriptors, import libraries. **Retail executables decrypt in the browser with the public retail container key** (the same constant xenia ships) — no console, NAND or CPU key involved; devkit images fall back automatically; BASIC-compressed images decompress on load (`NORMAL`/LZX and `DELTA` are rejected with an actionable message).
- **ELF (PPC) inspection** — ELF32/ELF64, program headers, sections; ELF homebrew recompiles end to end.
- **HLE (high-level emulation) kernel** — xboxkrnl / xam / xbdm exports are reimplemented as host functions, exactly xenia's model. Import calls become direct host calls with the full guest PPC context (r0..r31 snapshot, big-endian); the module exports `r3` so return values flow back. Implemented starter set (DbgPrint with a variadic formatter, KeBugCheck(Ex), KeQuerySystemTime/PerformanceFrequency/InterruptTime, NtYieldExecution, RtlCompareMemory(Ulong), RtlFillMemoryUlong); everything else traps loudly naming the module, export and ordinal — see [docs/HLE.md](docs/HLE.md).

## What does not work (yet)

Xbox 360 games **cannot boot yet** — not because of keys (retail images decrypt fine) but because of surface: the Xenos GPU, AltiVec/VMX, and the rest of the kernel HLE surface (file I/O, threads, sync objects) are porting work in progress. The recompiler covers a real subset of the ISA (integer + branch + load/store + control); anything it does not know is emitted as a trap with the offending PC, never silently miscompiled. Retail imports resolve against xenia's export tables (2918 known exports), so unimplemented calls fail with the exact `module!name` and ordinal. See [docs/HLE.md](docs/HLE.md) and [docs/PORTING_ROADMAP.md](docs/PORTING_ROADMAP.md) for the plan.

## Quick start

```bash
npm ci          # express + compression + jsdom (dev)
npm start       # serves on http://0.0.0.0:3000 (PORT env var overrides)
```

Then open the printed URL, pick a sample (or drop your own `.xex`/`.elf`/`.bin`), hit **Recompile & Run**.

## Development

```bash
npm test        # recompiler end-to-end, ELF loader, web app (jsdom), HLE/XEX2 suites — 25 tests
```

Layout:

```
public/js/recomp/recompiler.js   discovery + fragment lifter + wasm glue (entry: buildImage/recompileImage/runImage)
public/js/recomp/wasm_emitter.js minimal, validated WebAssembly module builder
public/js/recomp/ppc_decode.js   PPC instruction decoder used by the lifter
public/js/recomp/aes.js          AES-128 (ECB/CBC, caller-managed chain) for XEX2 containers
public/js/recomp/xex_loader.js   XEX2 load: retail/devkit decrypt (public keys), BASIC decompress, import-thunk resolution
public/js/recomp/hle.js          the HLE kernel host (DbgPrint & co.; unimplemented exports trap loudly)
public/js/recomp/hle_tables.js   2918 xboxkrnl/xam/xbdm export ordinals → names (from xenia)
public/js/recomp/hle_bundle.js   generated embed of hle.js/hle_tables.js/docs/HLE.md for the zip (npm run build:hle-bundle)
public/js/recomp/zip.js          dependency-free ZIP writer (store method)
public/js/recomp/templates.js    the project scaffold that ships inside the exported .zip
public/js/inspect.js             XEX2 / ELF parsers (shared by UI and tests)
public/js/recompiler-ui.js       browser wiring for the three-step page
public/js/ppc_asm.js             two-pass PPC assembler for the built-in samples
server/server.js                 static server with crossOriginIsolated + CSP + caching
docs/HLE.md                      the HLE kernel: model, ABI, implemented exports, porting backlog
docs/KERNELS.md                  research: how the emulators/recompilers handle the kernel (background)
docs/PORTING_ROADMAP.md          milestones from this PPC subset to booted homebrew/retail
tools/generate_hle_tables.py     regenerates hle_tables.js from the xenia export tables
tools/xenorecomp/                Python reference pieces (disassembler, lifter sketches)
```

## Testing philosophy

Tests assemble real big-endian Xenon PPC, lift it to WASM **exactly as the browser does**, execute the result and assert on guest-visible output (console bytes, exit code, stats) and on the structure of the exported project zip. If the assembler, decoder, lifter or zip writer breaks, tests fail — no mocks of the pipeline itself.

## License

BSD 3-Clause. This is a derivative work of xenia-canary (Copyright 2015 Ben Vanik and the xenia contributors). Upstream notices are preserved in [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No game data or Microsoft system files are distributed, ever.
