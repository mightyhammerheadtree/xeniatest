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
| 0x00CC | `NtAllocateVirtualMemory` | managed guest heap (see below); honours requested bases, 64 KiB alignment, `X_MEM_COMMIT/RESERVE`, zeroing |
| 0x00DC | `NtFreeVirtualMemory` | `X_MEM_RELEASE` removes the region; `X_MEM_DECOMMIT` zeroes it in place; double-free/unknown base rejected |
| 0x00E1 | `NtProtectVirtualMemory` | RW ↔ RO changes on managed regions; **EXECUTE bits rejected** (`X_STATUS_INVALID_PAGE_PROTECTION`, like xenia) |
| 0x0101 | `NtYieldExecution` | `STATUS_SUCCESS` |
| 0x011A | `RtlCompareMemory` | byte-wise compare, returns count of equal bytes (xenia contract — **not** stop-at-first-mismatch) |
| 0x011B | `RtlCompareMemoryUlong` | `(Source, Length, Pattern)` — BE u32s compared against the pattern until the first mismatch; returns **bytes** compared |
| 0x0125 | `RtlEnterCriticalSection` | single-threaded semantics (recursion, lock count); deadlocks trap loudly instead of hanging |
| 0x0126 | `RtlFillMemoryUlong` | length in bytes, rounded down to u32s |
| 0x012C | `RtlInitAnsiString` | writes `{ u16 length, u16 maximum_length, u32 pointer }` (BE) |
| 0x012D | `RtlInitUnicodeString` | same layout, length in bytes; returns the destination address |
| 0x012E | `RtlInitializeCriticalSection` | 28-byte `X_RTL_CRITICAL_SECTION`; type = EventSynchronizationObject, lock_count = -1 |
| 0x012F | `RtlInitializeCriticalSectionAndSpinCount` | spin count packed as `(spin + 255) >> 8`, capped at 255 (xenia) |
| 0x0130 | `RtlLeaveCriticalSection` | recursion-aware; unlocking a section not owned by the current thread traps loudly |
| 0x0141 | `RtlTryEnterCriticalSection` | returns 1 on acquire/recurse, 0 when locked elsewhere |
| 0x0195 | `XexGetModuleHandle` | `NULL` name → synthetic main-module handle (`STATUS_SUCCESS`); name lookup → `X_STATUS_OBJECT_NAME_NOT_FOUND` (not implemented, refuses to fabricate a module) |

### The managed guest heap

`NtAllocateVirtualMemory` carves pages out of a dedicated region of the flat
64 MiB guest arena managed by the HLE host (default `[0x00100000,
0x00F00000)` — below the MMIO exit door at `0x00F00FFC` and above everything
the recompiler emits). The region can be overridden per host with
`attachMemory(memory, { heapBase, heapEnd })`. Allocation is bump-pointer
with a free list (coalescing on release); pages are 4 KiB (64 KiB with
`X_MEM_LARGE_PAGES`, aligned to 64 KiB regardless). `X_MEM_RESET` and
`X_MEM_NOZERO` are reserved/handled as documented; anything outside the
managed region is rejected (`X_STATUS_INVALID_PARAMETER`) rather than
corrupting image memory.

Critical sections are implemented with **single-threaded host semantics**:
the recompiled guest runs on one WASM thread, so `Enter`/`Leave`/`TryEnter`
model recursion and lock counts exactly, and a genuine contention or
unbalanced unlock throws an `HleError` instead of deadlocking silently.

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
