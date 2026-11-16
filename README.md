# Xenia Canary — Web

A browser port of [xenia-canary](https://github.com/xenia-canary/xenia-canary), the Xbox 360 emulator, focused on doing the PPC core *right* as WebAssembly and being honest about the rest.

**Runs entirely client-side. No accounts, no telemetry, no uploads.**

## What works today (tested)

- **Xenon PPC (PowerPC64, big-endian) decoder** — the full xenia-canary opcode table, ported to C and compiled to WASM (`public/core/xeniatest-core.wasm`).
- **Interpreter** — 64-bit integer ALU, rotate/mask/shift, 32/64-bit loads & stores, branches (CTR loops, link register), `sc`, supervisor-call console path. Not a JIT yet — see roadmap.
- **Disassembler** — xenia's opcode descriptors, exact same output as the desktop project.
- **XEX2 inspection** — header, optional headers, security info, page descriptors, import libraries.
- **ELF inspection** — ELF32/ELF64, any endianness, PPC-focused; program headers and sections.
- **Library** — your dumps stored in the browser's Origin Private File System (OPFS), streamed in chunks, SHA-256 or size-based fingerprinting.
- **Runner** — built-in Xenon assembly programs (an honest smoke test: a real two-pass PPC assembler in `public/js/ppc_asm.js`), raw low-memory PPC binaries, and ELF64 images whose LOAD segments fit the guest window.
- **Bounded memory** — one 64 MiB guest RAM, one fixed WASM linear memory (88,080,384 B), zero 64-bit copies in JS. The core physically cannot grow memory, so it cannot OOM your tab.

## What does not work (yet)

Xbox 360 games **cannot boot**. That requires the kernel (HLE), the Xenos GPU, AltiVec, and your console's own NAND to decrypt retail executables. See [docs/PORTING_ROADMAP.md](docs/PORTING_ROADMAP.md) for the honest milestone plan and [docs/LEGAL.md](docs/LEGAL.md) for what is and isn't legal to do.

The interpreter is also not cycle-accurate and the floating-point path is partial; AltiVec/VMX instructions trap by default (status 2) and can be configured to skip.

## Quick start

```bash
npm install     # express + compression, that's all
npm start       # serves on http://0.0.0.0:3000 (PORT env var overrides)
```

Open the printed URL. The app needs **crossOriginIsolated** headers (COOP/COEP) — already set by the bundled server — to enable SharedArrayBuffer for the future threaded core.

## Development

```bash
npm test               # node --test core/tests/*.mjs (wasm tests, needs the built core)
npm run build:core     # rebuild the WASM core (requires zig; see core/README)
```

Layout:

```
core/src/xenia/        ported xenia-canary opcode tables & decoders (BSD-3, see THIRD_PARTY_NOTICES.md)
core/src/xenon_cpu.*   freestanding C ABI + interpreter written for this project
core/tests/            node:test suites that execute hand-assembled PPC and assert architectural state
public/js/ppc_asm.js   two-pass PPC assembly subset used by the built-in programs
public/js/storage.js   OPFS-backed dump library (streaming import, no 256 MiB buffers)
public/js/inspect.js   XEX2 / ELF parsers
public/js/runner.js    frame-budgeted core driver (keeps the tab responsive)
server/server.js       static server with crossOriginIsolated + CSP + caching
```

## Testing philosophy

The WASM core is tested *as an architectural sandbox*: tests hand-assemble real big-endian PPC instructions, run them in the sandbox, and assert on guest architectural state (PC, GPRs, XER carry, console bytes, halt status). The built-in browser programs are tested the same way — if the assembler, the decoder, or the interpreter breaks, tests fail.

## License

BSD 3-Clause. This is a derivative work of xenia-canary (Copyright 2015 Ben Vanik and the xenia contributors). Upstream notices are preserved in [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No game data or Microsoft system files are distributed, ever.
