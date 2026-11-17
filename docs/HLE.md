# HLE — how the recompiler replaces the Xbox 360 OS

**Status:** implemented and tested (synthetic retail/devkit XEX2 images run
end-to-end in Node and in the browser with zero firmware). The implemented
export surface is a starter set; see [Porting backlog](#porting-backlog).

The Xbox 360 OS surface — `xboxkrnl.exe`, `xam.xex`, `xbdm.xex` — is
emulated, not extracted. Every export a guest image imports is resolved, at
image-load time, against xenia's export tables (`public/js/recomp/hle_tables.js`,
2918 known exports across the three modules), and every import call becomes a
direct host call with the full guest PPC context. No BIOS, NAND dump, CPU
key, keyvault file or Microsoft binary is loaded, embedded, downloaded or
distributed at any point. This mirrors xenia's pure-HLE architecture.

## What your XEX2 needs to run here

Loaded by `public/js/recomp/xex_loader.js`:

- **Container format:** XEX2, any retail image as encrypted on disc.
  Decryption uses the **public retail container key** — the same constant
  xenia ships — to unwrap the per-image session key in the security info,
  then AES-128-CBC (zero IV) for the body. Development-kit images fall back
  to the devkit key automatically.
- **Compression:** `NONE` and `BASIC` are supported.
  - `NORMAL` (LZX) and `DELTA` images are **rejected with an actionable
    message** — recompress them to BASIC/none first (e.g. with xextool).
    LZX is deliberately out of scope for the browser build.
- **CPU key / keyvault:** never needed. Retail images are not encrypted
  with the CPU key; the session key is what the retail container key wraps.
- **PE sniff:** an image that does not decrypt into something starting with
  the PE signature (`MZ`) is rejected loudly — that is how the loader tells
  retail-wrapped from devkit-wrapped images (and how corrupted/foreign
  containers are caught).

The image bytes themselves are your responsibility — always your own dumps.
Nothing about them is uploaded; everything happens client-side.

## How an import call becomes a host call

1. `xex_loader.js` parses `IMPORT_LIBRARIES` and records
   `{ addr, type, ordinal, moduleId, moduleName, name }` for every import
   thunk (`type == 1`). Variable imports (`type == 0`) are poked with the
   unresolved-import sentinel, matching xenia.
2. At recompile time (`recompiler.js`), every direct `bl`/`b`/`bctrl` to a
   known thunk address is compiled to:
   - a call to `store_ctx`, which snapshots **r0..r31** into the scratch
     area at `0x00F11000` as 32 × u64 **big-endian** (plus the exit door
     semantics of xenia's context save), then
   - a call to the imported `xenorecomp.icall` import with the target
     address.
3. The host (`public/js/recomp/hle.js`) reads the snapshot, resolves
   `moduleId!ordinal` against xenia's tables, and dispatches to the
   implementation. Return value is delivered in **r3** (the wasm module
   exports the `r3` global; the host writes it), matching xenia's
   `SHIM_SET_RETURN` semantics.
4. Anything not implemented traps with `unimplemented export:
   xboxkrnl!<name> (ordinal 0xNNN)` — the same philosophy as xenia's
   `UnimplementedExport`: never silently pretend.

### Argument/return ABI (matches xenia's shim ABI)

- Arguments: `r3..r10` for the first 8, then the guest stack at
  `r1 + 0x54 + (n-8)*8` (xenia's `shim` convention).
- Return: `r3` only. Mutation of other registers is **not** propagated
  back (same as xenia's `SHIM_SET_RETURN`).
- Out-parameters (pointers) are written through the shared linear memory
  as big-endian u32/u64.
- Guest memory is **big-endian**; addresses are masked to the flat
  64 MiB guest arena (`0x03ffffff`).

## Implemented exports (xboxkrnl)

| ordinal | export | notes |
| --- | --- | --- |
| 0x0003 | `DbgPrint` | variadic formatter: `%d %i %u %x %X %s %c %p %ld %lu %lx %llx %llX %llu %%` — output goes to the MMIO console / `onConsole` |
| 0x0052 | `KeBugCheck` | throws `HleBugCheck` (traps loudly, by design) |
| 0x0053 | `KeBugCheckEx` | same, with parameters |
| 0x0082 | `KeQueryInterruptTime` | 100 ns ticks, monotonic (performance.now) |
| 0x0083 | `KeQueryPerformanceFrequency` | fixed 50,000,000 Hz |
| 0x0084 | `KeQuerySystemTime` | FILETIME (100 ns since 1601-01-01 UTC) |
| 0x0101 | `NtYieldExecution` | `STATUS_SUCCESS` |
| 0x011A | `RtlCompareMemory` | byte-wise compare, returns count compared (MSDN contract) |
| 0x011B | `RtlCompareMemoryUlong` | same, ULONG granularity |
| 0x0126 | `RtlFillMemoryUlong` | length in bytes, rounded down to u32s |

Everything else in the 2918 known exports traps with its name and ordinal.

## Porting backlog

Adding an export is mechanical:

1. Look up the ordinal in `public/js/recomp/hle_tables.js`
   (`XenonHLETables.XENIA_EXPORTS[module].get(ordinal)`).
2. Add an implementation to `IMPLS` in `public/js/recomp/hle.js`.
3. Run `npm test` (the synthetic-XEX2 suite exercises the full
   decrypt → recompile → HLE dispatch path).

xenia's `kernel/` tree is the reference implementation (BSD-3-Clause);
pulling in another subsystem is a port, not a guess — read xenia's source
for the contract before implementing.

## Why not a kernel dump?

Because there is nothing to dump. The 360 kernel image is encrypted and we
have no console here to produce one; more importantly, xenia proved that the
entire surface can be reimplemented as host functions. Where this build
still traps loudly, that is the backlog above — not a missing file.

`sc` (the raw syscall instruction) still traps: recompiled guests execute
syscalls through import thunks under this HLE model, so reaching `sc` means
the image used a mechanism this build does not model yet (and it will tell
you).

See also: [KERNELS.md](KERNELS.md) (why no kernel dump is required or
wanted), [ROADMAP.md](ROADMAP.md), [../public/docs/KERNELS.md](../public/docs/KERNELS.md).
