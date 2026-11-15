#!/usr/bin/env python3
"""
Builds the xeniatest WASM core (public/core/xeniatest-core.wasm).

Toolchain: zig (used as a portable clang/lld driver targeting
wasm32-wasi-musl). zig provides its own libc/headers so the core
compiles with zero host dependencies, but we still target freestanding with
-nostdlib: the core has no libc needs and a deterministic static footprint.

Locating zig, in order:
  1. $ZIG (path to the zig binary)
  2. `zig` on PATH
  3. the `ziglang` pip package (python3 -m pip install ziglang)

Usage: python3 core/build_core.py [--ram-mb N] [--debug]
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "core", "src")
OUT_DIR = os.path.join(ROOT, "public", "core")
OUT_FILE = os.path.join(OUT_DIR, "xeniatest-core.wasm")

SOURCES = [
    "xenon_cpu.cc",
    os.path.join("xenia", "ppc_opcode_table_gen.cc"),
    os.path.join("xenia", "ppc_opcode_disasm_gen.cc"),
    os.path.join("xenia", "ppc_opcode_disasm.cc"),
    os.path.join("xenia", "ppc_opcode_info.cc"),
]


def find_zig():
    if os.environ.get("ZIG"):
        return os.environ["ZIG"]
    on_path = shutil.which("zig")
    if on_path:
        return on_path
    try:
        import ziglang  # type: ignore

        candidate = os.path.join(os.path.dirname(ziglang.__file__), "zig")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    sys.exit(
        "error: could not find zig. Install it via `pip install ziglang`, "
        "download it from https://ziglang.org, or set the ZIG environment variable."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ram-mb", type=int, default=64, help="guest RAM size in MiB (default 64)")
    parser.add_argument("--debug", action="store_true", help="build with debug asserts (traps on internal assert)")
    args = parser.parse_args()

    zig = find_zig()
    version = subprocess.run([zig, "version"], capture_output=True, text=True, check=True).stdout.strip()

    ram_bytes = args.ram_mb * 1024 * 1024
    page = 65536
    # Static data (guest RAM + core state) plus 1 MiB slack for the stack and data segments.
    initial_pages = math.ceil((ram_bytes + 20 * 1024 * 1024) / page)

    os.makedirs(OUT_DIR, exist_ok=True)

    cmd = [
        zig,
        "c++",
        "-target", "wasm32-wasi-musl",
        "-O2",
        "-DNDEBUG",
        f"-DXENIATEST_GUEST_RAM_MB={args.ram_mb}",
        "-fno-exceptions",
        "-fno-rtti",
        "-fno-threadsafe-statics",
        "-Wl,--no-entry",
        "-Wl,--strip-all",
        "-Wl,--export-memory",
        f"-Wl,--initial-memory={initial_pages * page}",
        f"-Wl,--max-memory={initial_pages * page}",
        "-o", OUT_FILE,
        *[os.path.join(SRC_DIR, src) for src in SOURCES],
    ]
    if args.debug:
        cmd.append("-DXENIATEST_DEBUG_ASSERTS=1")

    print(f"zig {version}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    size = os.path.getsize(OUT_FILE)
    print(f"built {OUT_FILE} ({size} bytes, guest RAM {args.ram_mb} MiB, memory {initial_pages * page} bytes)")


if __name__ == "__main__":
    main()
