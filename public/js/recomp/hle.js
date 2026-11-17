/**
 * hle.js — HLE (high-level emulation) host for the Xenon recompiler.
 *
 * Xenia is a pure HLE emulator: the entire Xbox 360 OS surface (xboxkrnl,
 * xam.xex, xbdm.xex) is reimplemented as host functions, so no BIOS,
 * firmware, NAND dump, CPU key or keyvault file is loaded at any point.
 * This module does the same for the static recompiler: every import the
 * guest executable declares is resolved (at image load time) against the
 * xenia export tables in hle_tables.js, and import calls become direct
 * host calls with the full guest PPC context (r0..r31), exactly like
 * xenia's `sc 2` syscall-rewrite model — unimplemented exports trap loudly
 * with the module!name and ordinal instead of silently misrunning
 * (xenia's UnimplementedExport philosophy).
 *
 * Implementation status: a starter set of exports is implemented
 * (DbgPrint, KeQuerySystemTime/InterruptTime/PerformanceFrequency,
 * NtYieldExecution, RtlCompareMemory(Ulong), RtlFillMemoryUlong,
 * KeBugCheck(Ex), NtAllocateVirtualMemory/NtFreeVirtualMemory/
 * NtProtectVirtualMemory over a managed guest heap,
 * RtlInitializeCriticalSection(AndSpinCount)/RtlEnterCriticalSection/
 * RtlLeaveCriticalSection/RtlTryEnterCriticalSection,
 * RtlInitAnsiString/RtlInitUnicodeString, XexGetModuleHandle).
 * Everything else traps naming the missing export. Growing the surface
 * is the porting backlog — see docs/HLE.md.
 *
 * Guest heap model: NtAllocateVirtualMemory carves pages from a managed
 * region of the flat 64 MiB arena. The region defaults to
 * [0x00100000, 0x00F00000) — below the MMIO exit door at 0x00F00FFC and
 * above nothing the recompiler emits — and can be overridden via
 * attachMemory(memory, { heapBase, heapEnd }). Allocations are tracked in
 * host-side metadata; NtFreeVirtualMemory RELEASE removes them, DECOMMIT
 * zeroes them in place. Out-of-region addresses, double frees and
 * execute-bit page protection are rejected the way xenia rejects them.
 *
 * ABI notes: arguments come from r3..r10, then from the guest stack at
 * r1 + 0x54 (xenia's shim ABI). The guest memory is big-endian; out
 * parameters are written to it through the shared linear-memory view.
 * Return value is the r3 wasm global (exported as `r3`) — the host writes
 * it so JS callers (icall paths discovered at runtime) observe the value
 * too. Mutation of argument registers is not propagated back (matches
 * xenia's SHIM_SET_RETURN semantics, which only writes r3).
 *
 * The recompiler snapshots r0..r31 big-endian into the scratch area at
 * HLE_CTX_ADDR immediately before every indirect call, which gives this
 * host its view of the guest registers.
 *
 * This file is deliberately dependency-free and UMD-wrapped so the exact
 * same source can ship inside the exported project zip as a classic
 * script (window.XenonHLE) and be required from node tests.
 */
(function (global) {
  'use strict';

  const HLE_CTX_ADDR = 0x00F11000; // scratch area: r0..r31 as 32 × u64 BE (256 bytes)
  const GUEST_MASK = 0x03ffffff;
  const MODULE_IDS = ['xboxkrnl', 'xam', 'xbdm'];

  // NTSTATUS values (match Windows NT exactly; see xenia xbox.h).
  const X_STATUS_SUCCESS = 0x00000000;
  const X_STATUS_UNSUCCESSFUL = 0xc0000001;
  const X_STATUS_NOT_IMPLEMENTED = 0xc0000002;
  const X_STATUS_INVALID_HANDLE = 0xc0000008;
  const X_STATUS_INVALID_PARAMETER = 0xc000000d;
  const X_STATUS_NO_MEMORY = 0xc0000017;
  const X_STATUS_ALREADY_COMMITTED = 0xc0000021;
  const X_STATUS_ACCESS_DENIED = 0xc0000022;
  const X_STATUS_OBJECT_NAME_NOT_FOUND = 0xc0000034;
  const X_STATUS_INVALID_PAGE_PROTECTION = 0xc0000045;
  const X_STATUS_MEMORY_NOT_ALLOCATED = 0xc00000a0;

  // X_MEM_* allocation types / X_PAGE_* protect bits (360 XDK values,
  // matching xenia's xbox.h).
  const X_MEM_COMMIT = 0x00001000;
  const X_MEM_RESERVE = 0x00002000;
  const X_MEM_DECOMMIT = 0x00004000;
  const X_MEM_RELEASE = 0x00008000;
  const X_MEM_RESET = 0x00080000;
  const X_MEM_TOP_DOWN = 0x00100000;
  const X_MEM_NOZERO = 0x00800000;
  const X_MEM_LARGE_PAGES = 0x20000000;
  const X_PAGE_EXECUTE = 0x00000010;
  const X_PAGE_EXECUTE_READ = 0x00000020;
  const X_PAGE_EXECUTE_READWRITE = 0x00000040;
  const X_PAGE_EXECUTE_WRITECOPY = 0x00000080;
  const X_PAGE_EXECUTE_MASK = X_PAGE_EXECUTE | X_PAGE_EXECUTE_READ | X_PAGE_EXECUTE_READWRITE | X_PAGE_EXECUTE_WRITECOPY;

  const X_OBJECT_TYPE_EVENT_SYNCHRONIZATION = 0x01; // X_OBJECT_TYPES::EventSynchronizationObject (Auto Reset)
  const DEFAULT_HEAP_BASE = 0x00100000; // 1 MiB into the arena
  const DEFAULT_HEAP_END = 0x00f00000;  // below the MMIO exit door at 0x00F00FFC
  const MAIN_MODULE_HANDLE = 0x00010000;  // synthetic handle of the (only) guest module
  const CURRENT_THREAD_ID = 0x00beed01;   // synthetic id of the single guest thread (main)

  class HleError extends Error {}
  class HleUnimpl extends HleError {
    constructor(module, ordinal, name) {
      super(`unimplemented export: ${module}!${name} (ordinal 0x${ordinal.toString(16).toUpperCase().padStart(3, '0')})`);
      this.name2 = name;
      this.module = module;
      this.ordinal = ordinal;
    }
  }
  class HleBugCheck extends HleError {
    constructor(code, p1, p2, p3) {
      super(`KeBugCheck 0x${(code >>> 0).toString(16).toUpperCase().padStart(8, '0')} (0x${p1.toString(16)}, 0x${p2.toString(16)}, 0x${p3.toString(16)})`);
      this.code = code >>> 0;
      this.params = [p1, p2, p3];
    }
  }

  function exportName(moduleId, ordinal) {
    const tables = globalThis.XenonHLETables;
    const table = tables && tables.XENIA_EXPORTS && tables.XENIA_EXPORTS[moduleId];
    return (table && table.get(ordinal)) ?? null;
  }

  /** Guest-memory helpers bound to one WebAssembly.Memory. */
  function makeCtx(memory) {
    const bytes = new Uint8Array(memory.buffer);
    const view = new DataView(memory.buffer);
    const regAddr = (i) => HLE_CTX_ADDR + i * 8;
    const maskAddr = (a) => Number(BigInt(a) & 0x03ffffffn); // BigInt-safe: sign-extended 64-bit guest addresses exceed 2^53
    const check = (a, len) => {
      if (a + len > bytes.length) throw new HleError(`guest memory access out of bounds at 0x${(a >>> 0).toString(16)} (${len} bytes)`);
    };
    const ctx = {
      memory,
      readReg(i) {
        const a = regAddr(i);
        return (BigInt(view.getUint32(a, false)) << 32n) | BigInt(view.getUint32(a + 4, false));
      },
      /** xenia shim ABI: arg n in r[3+n] for n <= 7, else on the stack at r1 + 0x54 + (n-8)*8. */
      arg(n) {
        if (n < 8) return ctx.readReg(3 + n);
        const sp = ctx.readReg(1);
        const a = maskAddr(sp + BigInt(0x54 + (n - 8) * 8));
        return ctx.readU64BE(a);
      },
      mask(addr) { return maskAddr(addr); },
      readInt32BE(addr) {
        const a = maskAddr(addr);
        check(a, 4);
        return view.getInt32(a, false);
      },
      readU8(addr) {
        const a = maskAddr(addr);
        check(a, 1);
        return bytes[a];
      },
      readU16BE(addr) {
        const a = maskAddr(addr);
        check(a, 2);
        return view.getUint16(a, false);
      },
      readU32BE(addr) {
        const a = maskAddr(addr);
        check(a, 4);
        return view.getUint32(a, false) >>> 0;
      },
      readU64BE(addr) {
        const a = maskAddr(addr);
        check(a, 8);
        return (BigInt(view.getUint32(a, false)) << 32n) | BigInt(view.getUint32(a + 4, false));
      },
      readInt64BE(addr) { return BigInt.asIntN(64, ctx.readU64BE(addr)); },
      readCString(addr, maxLen = 4096) {
        const a0 = maskAddr(addr);
        let out = '';
        for (let i = 0; i < maxLen; i++) {
          const a = (a0 + i) & GUEST_MASK;
          check(a, 1);
          const b = bytes[a];
          if (b === 0) break;
          out += String.fromCharCode(b);
        }
        return out;
      },
      writeU32BE(addr, value) {
        const a = maskAddr(addr);
        check(a, 4);
        view.setUint32(a, value >>> 0, false);
      },
      writeU64BE(addr, value) {
        const a = maskAddr(addr);
        check(a, 8);
        const v = BigInt(value) & 0xffffffffffffffffn;
        view.setUint32(a, Number((v >> 32n) & 0xffffffffn) >>> 0, false);
        view.setUint32(a + 4, Number(v & 0xffffffffn) >>> 0, false);
      },
      writeU8(addr, value) {
        const a = maskAddr(addr);
        check(a, 1);
        bytes[a] = value & 0xff;
      },
      writeU16BE(addr, value) {
        const a = maskAddr(addr);
        check(a, 2);
        view.setUint16(a, value & 0xffff, false);
      },
      writeInt32BE(addr, value) {
        const a = maskAddr(addr);
        check(a, 4);
        view.setInt32(a, value | 0, false);
      },
      fill(addr, value, count) {
        const a = maskAddr(addr);
        check(a, count);
        bytes.fill(value & 0xff, a, a + count);
      },
      fillU32BE(addr, value, countBytes) {
        const a = maskAddr(addr);
        const n = Math.floor(countBytes / 4) * 4;
        check(a, n);
        for (let i = 0; i < n; i += 4) view.setUint32(a + i, value >>> 0, false);
      },
    };
    return ctx;
  }

  /** Variadic formatter for DbgPrint — %d %i %u %x %X %s %c %p %ld %lu %lx %llx %llX %llu %%. */
  function formatDbgPrint(fmt, ctx) {
    let out = '';
    let argIndex = 1; // arg 0 is the format string itself
    const next = () => ctx.arg(argIndex++);
    for (let i = 0; i < fmt.length; i++) {
      const ch = fmt[i];
      if (ch !== '%') { out += ch; continue; }
      i++;
      if (fmt[i] === '%') { out += '%'; continue; }
      let longs = 0;
      while (fmt[i] === 'l' || fmt[i] === 'L' || fmt[i] === 'z') {
        if (fmt[i] === 'l' || fmt[i] === 'L') longs++;
        i++;
      }
      if (fmt[i] === 'h') i++; // ignore short length modifiers
      const spec = fmt[i];
      if (spec === undefined) break;
      const wide = longs >= 2; // %lld, %llu, %llx — read 8 bytes anyway on this ABI
      switch (spec) {
        case 'd': case 'i': {
          const v = next();
          const s = wide ? BigInt.asIntN(64, v).toString() : BigInt.asIntN(32, v).toString();
          out += s;
          break;
        }
        case 'u': {
          const v = next();
          out += (wide ? v & 0xffffffffffffffffn : v & 0xffffffffn).toString();
          break;
        }
        case 'x': case 'X': {
          const v = next();
          const s = (wide ? v & 0xffffffffffffffffn : v & 0xffffffffn).toString(16);
          out += spec === 'X' ? s.toUpperCase() : s;
          break;
        }
        case 'p': out += '0x' + next().toString(16); break;
        case 'c': out += String.fromCharCode(Number(next() & 0xffn)); break;
        case 's': {
          const p = next();
          out += p === 0n ? '(null)' : ctx.readCString(p);
          break;
        }
        default: out += '%' + spec; // unknown specifier: emit verbatim
      }
    }
    return out;
  }

  /**
   * Managed guest heap — bump allocator with a free list over a fixed
   * region of the flat arena. Host-side metadata only; the guest sees a
   * zeroed range of addresses like any NtAllocateVirtualMemory result.
   */
  function makeGuestHeap(heapBase, heapEnd) {
    const base = heapBase >>> 0;
    const end = heapEnd >>> 0;
    if (!(base < end) || base >= 0x04000000 || end > 0x04000000) {
      throw new HleError(`guest heap region [0x${base.toString(16)}, 0x${end.toString(16)}) is outside the 64 MiB arena`);
    }
    const PAGE = 0x1000;
    const LARGE_PAGE = 0x10000;
    const alignUp = (v, a) => (v + a - 1) & ~(a - 1);
    let next = base;
    const allocs = new Map(); // base -> { size, committed, protect }
    const freeList = [];      // { base, size } sorted-ish, coalesced on insert
    const addFree = (b, s) => {
      if (s <= 0) return;
      freeList.push({ base: b, size: s });
      freeList.sort((x, y) => x.base - y.base);
      // coalesce neighbours
      for (let i = freeList.length - 1; i > 0; i--) {
        const cur = freeList[i], prev = freeList[i - 1];
        if (prev.base + prev.size === cur.base) { prev.size += cur.size; freeList.splice(i, 1); }
      }
    };
    const takeFree = (need, alignment, fixedBase /* 0 = any */) => {
      if (fixedBase) {
        // carve [fixedBase, fixedBase+need) out of the free list or the bump frontier
        if (fixedBase < base || fixedBase + need > end) return 0;
        // overlap check against live allocations
        for (const [b, a] of allocs) if (fixedBase < b + a.size && b < fixedBase + need) return 0;
        // consume from free list if it fully covers; else from bump frontier; else fail
        for (let i = 0; i < freeList.length; i++) {
          const seg = freeList[i];
          if (seg.base <= fixedBase && fixedBase + need <= seg.base + seg.size) {
            const before = fixedBase - seg.base, after = seg.base + seg.size - (fixedBase + need);
            freeList.splice(i, 1);
            if (before) freeList.push({ base: seg.base, size: before });
            if (after) freeList.push({ base: fixedBase + need, size: after });
            return fixedBase;
          }
        }
        if (fixedBase >= next && fixedBase + need <= end) {
          if (fixedBase > next) addFree(next, fixedBase - next);
          next = fixedBase + need;
          return fixedBase;
        }
        return 0;
      }
      for (let i = 0; i < freeList.length; i++) {
        const seg = freeList[i];
        const segBase = alignUp(seg.base, alignment);
        const segEnd = segBase + need;
        if (segEnd <= seg.base + seg.size) {
          const before = segBase - seg.base, after = seg.base + seg.size - segEnd;
          freeList.splice(i, 1);
          if (before) freeList.push({ base: seg.base, size: before });
          if (after) freeList.push({ base: segEnd, size: after });
          return segBase;
        }
      }
      const addr = alignUp(next, alignment);
      if (addr + need > end) return 0;
      next = addr + need;
      return addr;
    };
    return {
      base, end, PAGE, LARGE_PAGE,
      /** Allocate `size` bytes rounded to page granularity at 64 KB (or requested) alignment. Returns 0 on failure. */
      alloc(size, { largePages = false, fixedBase = 0, commit = true, protect = 0x04 /* X_PAGE_READWRITE */ } = {}) {
        // xenia: specified bases use the heap's (4K) page size; LARGE_PAGES only applies to unspecified bases
        const pageSize = fixedBase ? PAGE : (largePages ? LARGE_PAGE : PAGE);
        let adj = size >>> 0;
        if ((size | 0) < 0) adj = (-size | 0) >>> 0; // games sometimes pass negative sizes (xenia handles this)
        const need = alignUp(adj, pageSize);
        if (need === 0) return { address: 0, size: 0 };
        const alignment = fixedBase ? 1 : 0x10000; // xenia aligns unspecified bases to 64 KB
        const at = fixedBase ? (fixedBase & ~(pageSize - 1)) : 0;
        const address = takeFree(need, alignment, at);
        if (!address) return { address: 0, size: 0 };
        const rec = { size: need, committed: commit, protect, protectWasSetExplicitly: false };
        allocs.set(address, rec);
        return { address, size: need, fresh: true };
      },
      free(addr) {
        const rec = allocs.get(addr >>> 0);
        if (!rec) return null;
        allocs.delete(addr >>> 0);
        addFree(addr >>> 0, rec.size);
        return rec;
      },
      decommit(addr, size) {
        const rec = allocs.get(addr >>> 0);
        if (!rec) return null;
        rec.committed = false;
        return rec;
      },
      /** Find the allocation containing [addr, addr+size). */
      findContaining(addr, size) {
        const a = addr >>> 0, b = a + (size >>> 0);
        for (const [base0, rec] of allocs) {
          if (base0 <= a && b <= base0 + rec.size) return { base: base0, rec };
        }
        return null;
      },
      lookup(addr) { return allocs.get(addr >>> 0) ?? null; },
    };
  }

  const CURRENT_THREAD = CURRENT_THREAD_ID;

  function createHleHost({ onConsoleByte, onLog } = {}) {
    let memory = null;
    let guestHeap = null;
    const log = typeof onLog === 'function' ? onLog : () => {};
    const emitByte = typeof onConsoleByte === 'function'
      ? onConsoleByte
      : (b) => { if (typeof process !== 'undefined' && process.stdout) process.stdout.write(String.fromCharCode(b)); };
    const emitString = (s) => { for (let i = 0; i < s.length; i++) emitByte(s.charCodeAt(i)); };

    function attachMemory(wasmMemory, opts = {}) {
      memory = wasmMemory;
      const heapBase = (opts.heapBase ?? DEFAULT_HEAP_BASE) >>> 0;
      const heapEnd = (opts.heapEnd ?? DEFAULT_HEAP_END) >>> 0;
      guestHeap = makeGuestHeap(heapBase, heapEnd);
    }

    function ctxNow() {
      if (!memory) throw new HleError('HLE host called before attachMemory()');
      return makeCtx(memory);
    }

    const IMPLS = {
      xboxkrnl: new Map([
        [0x00000003, (ctx) => { // DbgPrint
          const fmtAddr = ctx.arg(0);
          const fmt = ctx.readCString(fmtAddr);
          const text = formatDbgPrint(fmt, ctx);
          emitString(text);
          log('DbgPrint: ' + text.replace(/\n$/, ''));
          return 0n;
        }],
        [0x00000052, (ctx) => { // KeBugCheck
          throw new HleBugCheck(ctx.arg(0) & 0xffffffffn, 0n, 0n, 0n);
        }],
        [0x00000053, (ctx) => { // KeBugCheckEx
          throw new HleBugCheck(ctx.arg(0) & 0xffffffffn, ctx.arg(1), ctx.arg(2), ctx.arg(3));
        }],
        [0x00000082, (ctx) => { // KeQueryInterruptTime
          const out = ctx.arg(0);
          const ticks = BigInt(Math.floor((typeof performance !== 'undefined' ? performance.now() : Date.now()) * 10000));
          if (out !== 0n) ctx.writeU64BE(out, ticks);
          return 0n;
        }],
        [0x00000083, (ctx) => { // KeQueryPerformanceFrequency
          const out = ctx.arg(0);
          if (out !== 0n) ctx.writeU64BE(out, 50000000n);
          return 0n;
        }],
        [0x00000084, (ctx) => { // KeQuerySystemTime — FILETIME (100 ns since 1601-01-01 UTC)
          const out = ctx.arg(0);
          const filetime = (BigInt(Date.now()) + 11644473600000n) * 10000n;
          if (out !== 0n) ctx.writeU64BE(out, filetime);
          return 0n;
        }],
        [0x00000101, () => 0n], // NtYieldExecution — STATUS_SUCCESS
        [0x0000011a, (ctx) => { // RtlCompareMemory — returns the number of matching bytes (xenia/MSDN contract)
          const s1 = ctx.arg(0), s2 = ctx.arg(1), length = ctx.arg(2) & 0xffffffffn;
          let count = 0n;
          for (let i = 0n; i < length; i++) {
            if (ctx.readU8(s1 + i) === ctx.readU8(s2 + i)) count++;
          }
          return count;
        }],
        [0x0000011b, (ctx) => { // RtlCompareMemoryUlong(Source, Length, Pattern) — bytes compared (ULONG granularity) until first mismatch
          const source = ctx.arg(0), length = ctx.arg(1) & 0xffffffffn, pattern = Number(ctx.arg(2) & 0xffffffffn);
          let count = 0n;
          for (let i = 0n; i + 4n <= length; i += 4n) {
            if (ctx.readU32BE(source + i) !== pattern) break;
            count += 4n;
          }
          return count;
        }],
        [0x00000126, (ctx) => { // RtlFillMemoryUlong — length in bytes, rounded down to multiple of 4
          const dst = ctx.arg(0), length = ctx.arg(1) & 0xffffffffn, value = ctx.arg(2);
          ctx.fillU32BE(dst, value & 0xffffffffn, Number(length));
          return 0n;
        }],
        [0x000000cc, (ctx) => { // NtAllocateVirtualMemory(base_addr_ptr, region_size_ptr, alloc_type, protect, debug_memory)
          const basePtr = ctx.arg(0), sizePtr = ctx.arg(1), allocType = Number(ctx.arg(2) & 0xffffffffn), protect = Number(ctx.arg(3) & 0xffffffffn);
          if (basePtr === 0n || sizePtr === 0n) return BigInt(X_STATUS_INVALID_PARAMETER);
          if (!(allocType & (X_MEM_COMMIT | X_MEM_RESET | X_MEM_RESERVE))) return BigInt(X_STATUS_INVALID_PARAMETER);
          if (allocType & X_MEM_RESET && (allocType & ~X_MEM_RESET)) return BigInt(X_STATUS_INVALID_PARAMETER);
          if (allocType & X_MEM_RESET) {
            // xenia asserts here; we trap loudly instead of silently corrupting
            throw new HleError('NtAllocateVirtualMemory: X_MEM_RESET is not implemented (matches xenia)');
          }
          if (protect & X_PAGE_EXECUTE_MASK) log('NtAllocateVirtualMemory: game requested an EXECUTE page protection (allowed, like xenia)');
          if (!guestHeap) throw new HleError('HLE host: guest heap not initialised (attachMemory was not called)');
          const baseAddr = ctx.readU32BE(basePtr) >>> 0;
          const rawSize = ctx.readU32BE(sizePtr) >>> 0;
          if (rawSize === 0) return BigInt(X_STATUS_INVALID_PARAMETER);
          if (baseAddr !== 0 && (baseAddr < guestHeap.base || baseAddr >= guestHeap.end)) {
            // address lives outside the guest-virtual heap (xenia: LookupHeap failure → INVALID_PARAMETER)
            return BigInt(X_STATUS_INVALID_PARAMETER);
          }
          const largePages = (allocType & X_MEM_LARGE_PAGES) !== 0;
          const commit = (allocType & X_MEM_COMMIT) !== 0;
          const result = guestHeap.alloc(rawSize, { largePages, fixedBase: baseAddr, commit, protect });
          if (result.address === 0) return BigInt(X_STATUS_NO_MEMORY);
          // zero on commit unless MEM_NOZERO (fresh arena pages are already zero; re-zero to be sure for reused frees)
          if (commit && !(allocType & X_MEM_NOZERO)) ctx.fill(result.address, 0, result.size);
          ctx.writeU32BE(basePtr, result.address);
          ctx.writeU32BE(sizePtr, result.size >>> 0);
          return BigInt(X_STATUS_SUCCESS);
        }],
        [0x000000dc, (ctx) => { // NtFreeVirtualMemory(base_addr_ptr, region_size_ptr, free_type, debug_memory)
          const basePtr = ctx.arg(0), sizePtr = ctx.arg(1), freeType = Number(ctx.arg(2) & 0xffffffffn);
          if (basePtr === 0n) return BigInt(X_STATUS_MEMORY_NOT_ALLOCATED);
          if (!guestHeap) throw new HleError('HLE host: guest heap not initialised');
          const baseAddr = ctx.readU32BE(basePtr) >>> 0;
          if (baseAddr === 0) return BigInt(X_STATUS_MEMORY_NOT_ALLOCATED);
          if (freeType === X_MEM_DECOMMIT) {
            const sizeVal = sizePtr !== 0n ? ctx.readU32BE(sizePtr) >>> 0 : 0;
            const rec = guestHeap.findContaining(baseAddr, sizeVal || 4);
            if (!rec) return BigInt(X_STATUS_INVALID_PARAMETER);
            ctx.fill(rec.base, 0, rec.rec.size);
            guestHeap.decommit(rec.base, rec.rec.size);
            if (sizePtr !== 0n) ctx.writeU32BE(sizePtr, rec.rec.size >>> 0);
            return BigInt(X_STATUS_SUCCESS);
          }
          const rec = guestHeap.lookup(baseAddr);
          if (!rec) return BigInt(X_STATUS_INVALID_PARAMETER);
          guestHeap.free(baseAddr);
          if (sizePtr !== 0n) ctx.writeU32BE(sizePtr, rec.size >>> 0);
          return BigInt(X_STATUS_SUCCESS);
        }],
        [0x000000e1, (ctx) => { // NtProtectVirtualMemory(base_addr_ptr, region_size_ptr, protect, old_protect, debug_memory)
          const basePtr = ctx.arg(0), sizePtr = ctx.arg(1), protect = Number(ctx.arg(2) & 0xffffffffn), oldProtectPtr = ctx.arg(3);
          if (basePtr === 0n || sizePtr === 0n) return BigInt(X_STATUS_INVALID_PARAMETER);
          const sizeVal = ctx.readU32BE(sizePtr) >>> 0;
          if (sizeVal === 0) return BigInt(X_STATUS_INVALID_PARAMETER);
          if (protect & X_PAGE_EXECUTE_MASK) {
            log('NtProtectVirtualMemory: game requested an EXECUTE page protection — rejected (matches xenia)');
            return BigInt(X_STATUS_INVALID_PAGE_PROTECTION);
          }
          if (!guestHeap) throw new HleError('HLE host: guest heap not initialised');
          const baseAddr = ctx.readU32BE(basePtr) >>> 0;
          const containing = guestHeap.findContaining(baseAddr, sizeVal);
          if (!containing) return BigInt(X_STATUS_ACCESS_DENIED);
          const prev = containing.rec.protect >>> 0;
          containing.rec.protect = protect >>> 0;
          if (oldProtectPtr !== 0n) ctx.writeU32BE(oldProtectPtr, prev);
          return BigInt(X_STATUS_SUCCESS);
        }],
        [0x00000125, (ctx) => { // RtlEnterCriticalSection — single-threaded host semantics
          const a = ctx.mask(ctx.arg(0));
          if (ctx.arg(0) === 0n) { log('RtlEnterCriticalSection: null critical section'); return 0n; }
          const owner = ctx.readU32BE(a + 0x18) >>> 0;
          if (owner === CURRENT_THREAD) {
            ctx.writeInt32BE(a + 0x10, ctx.readInt32BE(a + 0x10) + 1);
            ctx.writeInt32BE(a + 0x14, ctx.readInt32BE(a + 0x14) + 1);
            return 0n;
          }
          if (ctx.readInt32BE(a + 0x10) === -1) {
            ctx.writeInt32BE(a + 0x10, 0);
            ctx.writeU32BE(a + 0x18, CURRENT_THREAD);
            ctx.writeInt32BE(a + 0x14, 1);
            return 0n;
          }
          throw new HleError(`RtlEnterCriticalSection: deadlock — lock 0x${a.toString(16)} is owned by thread 0x${owner.toString(16)} (single-threaded HLE host)`);
        }],
        [0x0000012e, (ctx) => { // RtlInitializeCriticalSection
          const a = ctx.mask(ctx.arg(0));
          if (ctx.arg(0) === 0n) { log('RtlInitializeCriticalSection: null critical section'); return 0n; }
          ctx.fill(a, 0, 28);
          ctx.writeU8(a + 0x00, X_OBJECT_TYPE_EVENT_SYNCHRONIZATION);
          ctx.writeU8(a + 0x01, 0x00); // absolute (spin count / 256)
          ctx.writeInt32BE(a + 0x10, -1);
          ctx.writeInt32BE(a + 0x14, 0);
          ctx.writeU32BE(a + 0x18, 0);
          return 0n;
        }],
        [0x0000012f, (ctx) => { // RtlInitializeCriticalSectionAndSpinCount
          const a = ctx.mask(ctx.arg(0));
          if (ctx.arg(0) === 0n) { log('RtlInitializeCriticalSectionAndSpinCount: null critical section'); return BigInt(X_STATUS_INVALID_PARAMETER); }
          const spin = Number(ctx.arg(1) & 0xffffffffn);
          const absolute = Math.min(255, (spin + 255) >> 8);
          ctx.fill(a, 0, 28);
          ctx.writeU8(a + 0x00, X_OBJECT_TYPE_EVENT_SYNCHRONIZATION);
          ctx.writeU8(a + 0x01, absolute);
          ctx.writeInt32BE(a + 0x10, -1);
          ctx.writeInt32BE(a + 0x14, 0);
          ctx.writeU32BE(a + 0x18, 0);
          return BigInt(X_STATUS_SUCCESS);
        }],
        [0x00000130, (ctx) => { // RtlLeaveCriticalSection — single-threaded host semantics
          const a = ctx.mask(ctx.arg(0));
          if (ctx.arg(0) === 0n) { log('RtlLeaveCriticalSection: null critical section'); return 0n; }
          const owner = ctx.readU32BE(a + 0x18) >>> 0;
          if (owner !== CURRENT_THREAD) {
            throw new HleError(`RtlLeaveCriticalSection: unbalanced unlock of 0x${a.toString(16)} (owned by 0x${owner.toString(16)}, current 0x${CURRENT_THREAD.toString(16)})`);
          }
          const rec = ctx.readInt32BE(a + 0x14) - 1;
          ctx.writeInt32BE(a + 0x14, rec);
          if (rec !== 0) {
            ctx.writeInt32BE(a + 0x10, ctx.readInt32BE(a + 0x10) - 1);
            return 0n;
          }
          ctx.writeU32BE(a + 0x18, 0);
          const old = ctx.readInt32BE(a + 0x10);
          ctx.writeInt32BE(a + 0x10, old - 1);
          // rec == 0 and old != -1 would wake a waiter; no waiters exist in the single-threaded host
          return 0n;
        }],
        [0x00000141, (ctx) => { // RtlTryEnterCriticalSection — returns 1 if entered, 0 otherwise
          const a = ctx.mask(ctx.arg(0));
          if (ctx.arg(0) === 0n) return 1n; // xenia pretends success on null
          const owner = ctx.readU32BE(a + 0x18) >>> 0;
          if (owner === CURRENT_THREAD) {
            ctx.writeInt32BE(a + 0x10, ctx.readInt32BE(a + 0x10) + 1);
            ctx.writeInt32BE(a + 0x14, ctx.readInt32BE(a + 0x14) + 1);
            return 1n;
          }
          if (ctx.readInt32BE(a + 0x10) === -1) {
            ctx.writeInt32BE(a + 0x10, 0);
            ctx.writeU32BE(a + 0x18, CURRENT_THREAD);
            ctx.writeInt32BE(a + 0x14, 1);
            return 1n;
          }
          return 0n;
        }],
        [0x0000012c, (ctx) => { // RtlInitAnsiString — guest ANSI_STRING { u16 length, u16 max, u32 ptr }
          const dest = ctx.arg(0), source = ctx.arg(1);
          if (dest === 0n) return 0n;
          let length = 0;
          if (source !== 0n) {
            const a0 = ctx.mask(source);
            while (length < 0xffff && ctx.readU8(a0 + length) !== 0) length++;
          }
          ctx.writeU16BE(dest, length);
          ctx.writeU16BE(dest + 2n, source !== 0n ? (length + 1) & 0xffff : 0);
          ctx.writeU32BE(dest + 4n, source !== 0n ? ctx.mask(source) : 0);
          return 0n;
        }],
        [0x0000012d, (ctx) => { // RtlInitUnicodeString — guest UNICODE_STRING, length in bytes
          const dest = ctx.arg(0), source = ctx.arg(1);
          if (dest === 0n) return 0n;
          let units = 0;
          if (source !== 0n) {
            const a0 = ctx.mask(source);
            while (units < 0xffff && ctx.readU16BE(a0 + units * 2) !== 0) units++;
          }
          ctx.writeU16BE(dest, Math.min(0xfffe, units * 2));
          ctx.writeU16BE(dest + 2n, source !== 0n ? Math.min(0xffff, (units + 1) * 2) : 0);
          ctx.writeU32BE(dest + 4n, source !== 0n ? ctx.mask(source) : 0);
          return BigInt(ctx.mask(dest));
        }],
        [0x00000195, (ctx) => { // XexGetModuleHandle(name, module_handle_ptr)
          const namePtr = ctx.arg(0), outPtr = ctx.arg(1);
          if (namePtr === 0n) {
            if (outPtr !== 0n) ctx.writeU32BE(outPtr, MAIN_MODULE_HANDLE);
            return BigInt(X_STATUS_SUCCESS);
          }
          // Name lookup is not implemented — do not fabricate a handle for a module we cannot identify
          log('XexGetModuleHandle: name lookup is not implemented; returning OBJECT_NAME_NOT_FOUND');
          if (outPtr !== 0n) ctx.writeU32BE(outPtr, 0);
          return BigInt(X_STATUS_OBJECT_NAME_NOT_FOUND);
        }],
      ]),
      xam: new Map(),
      xbdm: new Map(),
    };

    function call(moduleId, ordinal) {
      const module = String(moduleId);
      const ord = Number(ordinal) & 0xffff;
      const impl = IMPLS[module] && IMPLS[module].get(ord);
      const name = exportName(module, ord) ?? `ordinal_${ord.toString(16).toUpperCase().padStart(3, '0')}`;
      if (!impl) throw new HleUnimpl(module, ord, name);
      const ctx = ctxNow();
      return BigInt(impl(ctx) ?? 0n);
    }

    return { attachMemory, call };
  }

  const api = {
    createHleHost,
    HleError,
    HleUnimpl,
    HleBugCheck,
    HLE_CTX_ADDR,
    MODULE_IDS,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonHLE = api;
})(typeof self !== 'undefined' ? self : globalThis);
