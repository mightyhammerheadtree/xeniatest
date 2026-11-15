/**
 * runner.js — drives the WASM core from the browser: loads programs into
 * guest RAM, runs them in frame-budgeted slices (so the tab stays
 * responsive), drains the MMIO console, and reports architectural state.
 *
 * Memory discipline: one fixed-size guest RAM (64 MiB, compiled into the
 * core) + one console ring. No copies of the program persist beyond the
 * guest image, so running never holds two payloads at once.
 */
import { Core, CoreStatus, CoreStatusText, formatHex } from './core.js';
import { parseElf, classifyMagic } from './inspect.js';
import { assemble } from './ppc_asm.js';

const INSTRUCTIONS_PER_FRAME_DEFAULT = 65536;

export class Runner {
  constructor() {
    this.core = null;
    this.running = false;
    this.rafHandle = 0;
    this.budgetPerFrame = INSTRUCTIONS_PER_FRAME_DEFAULT;
    this.trapOnUnimplemented = true;
    this.loadedName = null;
    this.loadedBase = 0x10000;
    this.loadedBytes = 0;
    this.entry = 0x10000;
    this.listeners = new Set();
    this.mipsWindow = [];
    this.lastMips = 0;
  }

  async init() {
    if (this.core) return this.core;
    this.core = await Core.create('core/xeniatest-core.wasm');
    this.core.setTrapOnUnimplemented(true);
    this.core.reset(this.entry);
    return this.core;
  }

  onChange(cb) { this.listeners.add(cb); return () => this.listeners.delete(cb); }

  emit(kind, payload) {
    for (const cb of this.listeners) {
      try { cb(kind, payload); } catch { /* listener errors must not kill the loop */ }
    }
  }

  setBudget(n) { this.budgetPerFrame = Math.max(1024, n | 0); }
  setTrapOnUnimplemented(on) {
    this.trapOnUnimplemented = !!on;
    if (this.core) this.core.setTrapOnUnimplemented(this.trapOnUnimplemented);
  }

  /** Assemble and load a built-in (or user-assembled) source program. */
  loadSource(source, entry = 0x10000, name = 'program') {
    const image = assemble(source);
    this.loadImage(image.bytes, image.base, entry, name);
    return image;
  }

  loadImage(bytes, base, entry, name, { resetLog = true } = {}) {
    if (!this.core) throw new Error('core not initialized');
    this.stop();
    this.core.reset(entry);
    this.core.setTrapOnUnimplemented(this.trapOnUnimplemented);
    this.core.writeBytes(base, bytes);
    this.entry = entry;
    this.loadedName = name;
    this.loadedBase = base;
    this.loadedBytes = bytes.length;
    this.lastImage = { bytes, base, entry, name };
    this.emit('loaded', { name, base, entry, bytes: bytes.length });
    this.emitState();
    if (resetLog) this.emit('console', `; loaded ${name}: ${bytes.length} bytes at ${formatHex(BigInt(base), 8)}, entry ${formatHex(BigInt(entry), 8)}\n`);
  }

  /** Load a file from disk (raw binary or ELF). Returns a report object. */
  async loadFile(file) {
    if (!this.core) throw new Error('core not initialized');
    const head = new Uint8Array(await file.slice(0, 4096).arrayBuffer());
    const kind = classifyMagic(head);
    if (kind === 'elf') {
      // ELF: check that all LOAD segments fit in the guest's flat window.
      const bytes = new Uint8Array(await file.arrayBuffer());
      const elf = parseElf(bytes);
      const ram = this.core.ramSize;
      let maxAddr = 0;
      let minAddr = Number.MAX_SAFE_INTEGER;
      for (const ph of elf.programHeaders) {
        const pType = Number(ph.type);
        if (pType !== 1) continue; // PT_LOAD
        const start = Number(ph.vaddr);
        const end = start + Number(ph.memsz);
        if (end > maxAddr) maxAddr = end;
        if (start < minAddr) minAddr = start;
      }
      if (elf.programHeaders.length === 0) {
        return { ok: false, kind, reason: 'ELF has no program headers; nothing to load.', elf };
      }
      const entry = Number(elf.entry);
      if (maxAddr > ram || entry >= ram || minAddr < 0) {
        return {
          ok: false,
          kind,
          elf,
          reason: `This ELF targets the Xbox 360 address space (entry ${formatHex(elf.entry)}, image up to ${formatHex(BigInt(maxAddr))}), which the 64 MiB flat guest window cannot represent. The core currently executes low-memory PPC binaries only — run a built-in program, or use the Inspector to study this file.`,
        };
      }
      // Copy each PT_LOAD into the guest. Cache a flat image spanning the
      // loaded range so re-running after a halt works without the file.
      this.stop();
      this.core.reset(entry);
      this.core.setTrapOnUnimplemented(this.trapOnUnimplemented);
      let minLoaded = Number.MAX_SAFE_INTEGER, maxLoaded = 0, total = 0;
      const pending = [];
      for (const ph of elf.programHeaders) {
        if (Number(ph.type) !== 1) continue;
        const start = Number(ph.vaddr);
        const filesz = Number(ph.filesz);
        const memsz = Number(ph.memsz);
        const offset = Number(ph.offset ?? 0);
        if (filesz > 0) {
          const slice = bytes.subarray(offset, offset + filesz);
          this.core.writeBytes(start, slice);
          pending.push({ start, slice });
        }
        minLoaded = Math.min(minLoaded, start);
        maxLoaded = Math.max(maxLoaded, start + memsz);
        total += memsz;
      }
      this.entry = entry;
      this.loadedName = file.name;
      this.loadedBase = Number.isFinite(minLoaded) ? minLoaded : 0;
      this.loadedBytes = total;
      // Flat image covering [minLoaded, maxLoaded), plus sparse segments.
      const span = maxLoaded - (minLoaded === Number.MAX_SAFE_INTEGER ? 0 : minLoaded);
      const flatBase = minLoaded === Number.MAX_SAFE_INTEGER ? 0 : minLoaded;
      const flat = new Uint8Array(span);
      for (const { start, slice } of pending) flat.set(slice, start - flatBase);
      this.lastImage = { bytes: flat, base: flatBase, entry, name: file.name };
      this.emit('loaded', { name: file.name, base: this.loadedBase, entry, bytes: total });
      this.emit('console', `; loaded ELF ${file.name}: entry ${formatHex(elf.entry)}, ${elf.programHeaders.length} program headers, ${total} bytes mapped\n`);
      this.emitState();
      return { ok: true, kind, elf, entry };
    }
    // Raw binary: linked for 0x10000 by convention.
    const bytes = new Uint8Array(await file.arrayBuffer());
    this.loadImage(bytes, 0x10000, 0x10000, file.name);
    return { ok: true, kind: 'raw', entry: 0x10000 };
  }

  get instructionsExecuted() { return this.core ? this.core.instructionsExecuted() : 0n; }

  status() { return this.core ? this.core.status() : CoreStatus.Syscall; }

  isHalted() { return this.status() !== CoreStatus.Ok; }

  emitState() {
    if (!this.core) return;
    const x = this.core;
    let mips = 0;
    const now = performance.now();
    this.mipsWindow = this.mipsWindow.filter((p) => now - p.t < 1000);
    const recent = this.mipsWindow.reduce((acc, p) => acc + p.n, 0);
    mips = recent / 1e6;
    this.lastMips = mips;
    this.emit('state', {
      pc: x.pc(),
      lr: x.lr(),
      ctr: x.ctr(),
      xer: x.xer(),
      cr: x.cr(),
      instructions: x.instructionsExecuted(),
      unimplemented: x.unimplementedCount(),
      lastUnimplemented: x.lastUnimplemented(),
      lastFaultAddress: x.lastFaultAddress(),
      status: x.status(),
      statusText: CoreStatusText[x.status()] ?? 'unknown',
      mips,
      loadedName: this.loadedName,
      loadedBase: this.loadedBase,
      loadedBytes: this.loadedBytes,
      entry: this.entry,
    });
  }

  drainConsole() {
    if (!this.core) return;
    const text = this.core.consoleText();
    if (text) this.emit('console', text);
  }

  /** Run for at most `budget` instructions or until halt. */
  runSlice(budget) {
    if (!this.core) return { halted: true, status: CoreStatus.Syscall };
    const status = this.core.run(budget);
    this.mipsWindow.push({ t: performance.now(), n: budget });
    this.drainConsole();
    this.emitState();
    if (status !== CoreStatus.Ok) {
      this.stop();
      this.emit('halted', {
        status,
        statusText: CoreStatusText[status] ?? 'unknown',
        pc: this.core.pc(),
        lastUnimplemented: this.core.lastUnimplemented(),
        lastFaultAddress: this.core.lastFaultAddress(),
        instructions: this.core.instructionsExecuted(),
      });
      return { halted: true, status };
    }
    return { halted: false, status };
  }

  /** Start the frame loop. */
  run() {
    if (!this.core || this.running) return;
    if (this.isHalted()) {
      // RAM is wiped by reset, so reload the cached image before re-running.
      if (!this.lastImage) {
        this.emit('console', '; nothing loaded — pick a program first\n');
        return;
      }
      const { bytes, base, entry, name } = this.lastImage;
      this.core.reset(entry);
      this.core.setTrapOnUnimplemented(this.trapOnUnimplemented);
      this.core.writeBytes(base, bytes);
      this.emit('console', `; reloaded ${name}\n`);
    }
    this.running = true;
    this.emit('running', { running: true });
    const tick = () => {
      if (!this.running) return;
      const t0 = performance.now();
      let remaining = this.budgetPerFrame;
      while (remaining > 0 && this.running) {
        const slice = Math.min(remaining, 16384);
        const { halted } = this.runSlice(slice);
        remaining -= slice;
        if (halted) return;
        // Yield if the frame ran long (cooperative scheduling, keeps UI alive).
        if (performance.now() - t0 > 12) break;
      }
      this.rafHandle = requestAnimationFrame(tick);
    };
    this.rafHandle = requestAnimationFrame(tick);
  }

  pause() {
    this.running = false;
    if (this.rafHandle) cancelAnimationFrame(this.rafHandle);
    this.rafHandle = 0;
    this.emit('running', { running: false });
    this.emitState();
  }

  /** Execute exactly one instruction (from the stopped state). */
  step() {
    if (!this.core) return;
    this.pause();
    const status = this.core.step();
    this.drainConsole();
    this.emitState();
    if (status !== CoreStatus.Ok) {
      this.emit('halted', {
        status,
        statusText: CoreStatusText[status] ?? 'unknown',
        pc: this.core.pc(),
        lastUnimplemented: this.core.lastUnimplemented(),
        lastFaultAddress: this.core.lastFaultAddress(),
        instructions: this.core.instructionsExecuted(),
      });
    }
    return status;
  }

  stop() { this.pause(); }

  /** Full reset of the guest: RAM cleared, console cleared. */
  reset() {
    this.pause();
    if (this.core) {
      this.core.reset(this.entry);
      this.core.setTrapOnUnimplemented(this.trapOnUnimplemented);
      this.drainConsole();
      this.emitState();
      this.emit('console', '; guest state cleared (RAM zeroed)\n');
    }
  }

  /** Reset and reload the currently loaded program, if any are cached. */
  async reload(loader) {
    if (!this.core) return;
    this.pause();
    if (loader) {
      await loader();
    } else {
      this.emit('need-reload', { name: this.loadedName });
    }
  }

  coreSnapshot() {
    if (!this.core) return null;
    const x = this.core;
    return {
      pc: formatHex(x.pc()),
      lr: formatHex(x.lr()),
      ctr: formatHex(x.ctr()),
      xer: formatHex(x.xer()),
      cr: formatHex(BigInt(x.cr()), 8),
      instructions: x.instructionsExecuted().toString(),
      unimplemented: x.unimplementedCount().toString(),
      lastUnimplemented: formatHex(BigInt(x.lastUnimplemented()), 8),
      lastFaultAddress: formatHex(x.lastFaultAddress()),
      status: x.status(),
      statusText: CoreStatusText[x.status()] ?? 'unknown',
      mips: this.lastMips,
      loadedName: this.loadedName,
      loadedBase: formatHex(BigInt(this.loadedBase), 8),
      loadedBytes: this.loadedBytes,
      entry: formatHex(BigInt(this.entry), 8),
    };
  }
}
