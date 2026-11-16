# Core build notes

`public/core/xeniatest-core.wasm` is committed so the app runs out of the box.
Rebuild it only when you change `core/src/`.

## Requirements

- Python 3
- [ziglang](https://pypi.org/project/ziglang/) 0.17.x (`pip install ziglang`), used as the C/C++ compiler
- No system LLVM/clang/cmake needed

## Build

```bash
export ZIG=/path/to/zig   # e.g. "$(python3 -c 'import ziglang, os; print(os.path.join(os.path.dirname(ziglang.__file__), "zig"))')"
python3 core/build_core.py
python3 core/build_core.py --debug   # unoptimized + asserts
```

The script emits `public/core/xeniatest-core.wasm` with:

- fixed 64 MiB guest RAM (`-DXENIATEST_GUEST_RAM_MB=64`),
- WASM linear memory `--initial-memory == --max-memory == 88,080,384 bytes`
  (guest RAM + ~20 MiB of static overhead), so the memory can never grow,
- `--strip-all`, `--no-entry`, exported memory, and an explicit export list.

## Layout

- `core/src/xenia/` — ported xenia-canary sources (opcode tables, decoder,
  disassembler). Regenerate from a fresh xenia-canary checkout with
  `core/port_xenia_decoder.sh /path/to/xenia-canary`.
- `core/src/xenon_cpu.{h,cc}` — the C ABI and the interpreter. Written for
  this project; xenia-canary served as the semantic reference.
- `core/tests/` — node:test suites. They assemble big-endian PPC programs in
  JS, execute them in the sandbox, and assert architectural state.

## Why wasm32-wasi-musl instead of freestanding?

The ported xenia tables assume an environment with basic C/C++ runtime pieces
(new/delete, mem*). `wasm32-wasi-musl` provides them; the only WASI imports
that remain in the module (`args_get`, `args_sizes_get`, `proc_exit`) are
never called — the host (tests or `public/js/core.js`) stubs them anyway.
