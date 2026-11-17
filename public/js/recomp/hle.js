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
 * KeBugCheck(Ex)). Everything else traps naming the missing export.
 * Growing the surface is the porting backlog — see docs/HLE.md.
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

  function createHleHost({ onConsoleByte, onLog } = {}) {
    let memory = null;
    const log = typeof onLog === 'function' ? onLog : () => {};
    const emitByte = typeof onConsoleByte === 'function'
      ? onConsoleByte
      : (b) => { if (typeof process !== 'undefined' && process.stdout) process.stdout.write(String.fromCharCode(b)); };
    const emitString = (s) => { for (let i = 0; i < s.length; i++) emitByte(s.charCodeAt(i)); };

    function attachMemory(wasmMemory) {
      memory = wasmMemory;
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
        [0x0000011a, (ctx) => { // RtlCompareMemory — byte-wise, count until first mismatch (MSDN contract)
          const s1 = ctx.arg(0), s2 = ctx.arg(1), length = ctx.arg(2);
          let count = 0n;
          const limit = length & 0xffffffffn;
          for (let i = 0n; i < limit; i++) {
            if (ctx.readU8(s1 + i) !== ctx.readU8(s2 + i)) break;
            count++;
          }
          return count;
        }],
        [0x0000011b, (ctx) => { // RtlCompareMemoryUlong — length is bytes, rounded down to ULONGs
          const s1 = ctx.arg(0), s2 = ctx.arg(1), length = ctx.arg(2) & 0xffffffffn;
          let count = 0n;
          for (let i = 0n; i + 4n <= length; i += 4n) {
            if (ctx.readU32BE(s1 + i) !== ctx.readU32BE(s2 + i)) break;
            count += 4n;
          }
          return count;
        }],
        [0x00000126, (ctx) => { // RtlFillMemoryUlong — length in bytes, rounded down to multiple of 4
          const dst = ctx.arg(0), length = ctx.arg(1) & 0xffffffffn, value = ctx.arg(2);
          ctx.fillU32BE(dst, value & 0xffffffffn, Number(length));
          return 0n;
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
