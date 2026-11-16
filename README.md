# Xenon Recompiler — Xbox 360 → WebAssembly, in your browser

A **static recompiler** for Xbox 360 (Xenon PPC64, big-endian) executables that runs entirely client-side. It lifts PPC machine code to WebAssembly at click-time, executes the result in the page, and can export the recompiled program as a standalone project `.zip` (double-clickable runner, no server required).

**No accounts, no telemetry, no uploads — your binary never leaves the machine.**

> Derivative work of [xenia-canary](https://github.com/xenia-canary/xenia-canary) (decoder semantics), not an emulator. You bring the binary; the recompiler does the rest.

## What works today (tested)

- **Static recompilation, end to end** — discovery of reachable code from the entry point, fragment lifting (integer ALU, rotate/mask/shift, 32/64-bit loads/stores, branches incl. CTR loops & link register, `bl`/`blr` calls, `sc`, FP fcmpo/fcmpu decode), wasm emission, instantiation, execution.
- **Built-in samples** — Hello, Fibonacci (`fib(25) = 75025`) and Countdown are real hand-assembled PPC binaries recompiled in-page; they print through the guest MMIO console and halt via the exit door.
- **In-page run** — drop a `.xex`, `.elf` or raw `.bin`; it recompiles and runs with one click, output shown in the on-page console.
- **Project export** — one click downloads a `.zip` containing the `guest.wasm`, the image segments, a `file://`-double-clickable runner, a Node runner (`run.js`), `manifest.json`, README and kernel notes. Works offline, nothing to install.
- **XEX2 inspection** — header, optional headers, security info, page descriptors, import libraries. Retail Xbox 360 executables load and run where the decrypted base image is supplied; encrypted titles need your own console NAND (see [docs/KERNELS.md](docs/KERNELS.md)).
- **ELF (PPC) inspection** — ELF32/ELF64, program headers, sections; ELF homebrew recompiles end to end.

## What does not work (yet)

Xbox 360 games **cannot boot**. That requires the kernel (HLE surface for the imported library set), the Xenos GPU, AltiVec/VMX, and — for retail executables — your own console NAND to decrypt them. The recompiler covers a real subset of the ISA (integer + branch + load/store + control); anything it does not know is emitted as a trap with the offending PC, never silently miscompiled. See [docs/KERNELS.md](docs/KERNELS.md) for the kernel situation and `docs/` for the porting roadmap.

## Quick start

```bash
npm ci          # express + compression + jsdom (dev)
npm start       # serves on http://0.0.0.0:3000 (PORT env var overrides)
```

Then open the printed URL, pick a sample (or drop your own `.xex`/`.elf`/`.bin`), hit **Recompile & Run**.

## Development

```bash
npm test        # recompiler end-to-end, ELF loader, web app (jsdom) suites
```

Layout:

```
public/js/recomp/recompiler.js   discovery + fragment lifter + wasm glue (entry: buildImage/recompileImage/runImage)
public/js/recomp/wasm_emitter.js minimal, validated WebAssembly module builder
public/js/recomp/ppc_decode.js   PPC instruction decoder used by the lifter
public/js/recomp/zip.js          dependency-free ZIP writer (store method)
public/js/recomp/templates.js    the project scaffold that ships inside the exported .zip
public/js/inspect.js             XEX2 / ELF parsers (shared by UI and tests)
public/js/recompiler-ui.js       browser wiring for the three-step page
public/js/ppc_asm.js             two-pass PPC assembler for the built-in samples
server/server.js                 static server with crossOriginIsolated + CSP + caching
docs/KERNELS.md                  what the Xbox 360 kernel is, why you supply your own
tools/xenorecomp/                Python reference pieces (disassembler, lifter sketches)
```

## Testing philosophy

Tests assemble real big-endian Xenon PPC, lift it to WASM **exactly as the browser does**, execute the result and assert on guest-visible output (console bytes, exit code, stats) and on the structure of the exported project zip. If the assembler, decoder, lifter or zip writer breaks, tests fail — no mocks of the pipeline itself.

## License

BSD 3-Clause. This is a derivative work of xenia-canary (Copyright 2015 Ben Vanik and the xenia contributors). Upstream notices are preserved in [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No game data or Microsoft system files are distributed, ever.
