/**
 * gpu_host.js — wires the Xenos GPU subsystem to the HLE and the page.
 *
 * hle.js (deliberately dependency-free, shipping in the exported zip) needs a
 * way to reach the GPU without importing it: it reads `globalThis.XenonGpuHost`
 * and calls initializeRingBuffer / enableReadPointerWriteBack /
 * notifyWritePointer / mmioStore / mmioLoad. This module is that host for the in-page frontend.
 * (The exported zip does not ship the GPU modules yet, so Vd* exports trap
 * there — loudly, with the reason — until it does.)
 *
 * GPU register aperture: recompiled guests reach the GPU through a second
 * MMIO window at [0x00F02000, 0x00F03000) (byte offset = register × 4), wired
 * in the recompiler as xenorecomp.gpu_mmio_* → here. Only CP_RB_WPTR (0x01C5,
 * the doorbell) and CP_RB_RPTR (0x01C4, read pointer) are modeled; every other
 * register traps loudly, by design.
 *
 * Usage (page):
 *   const host = new XenonGpuHost({ memory });   // guest WebAssembly.Memory
 *   globalThis.XenonGpuHost = host;              // HLE hook appears
 *   await host.attachWebGPU(canvas);             // optional; without a canvas
 *                                                // the ring still executes and
 *                                                // traps at draw() time like the
 *                                                // backend does today
 *
 * Everything guest-driven stays big-endian; the ring lives in guest physical
 * memory and is executed through the same XenosCommandProcessor the tests use.
 */
(function (global) {
  'use strict';

  function loadDeps() {
    const pm4g = globalThis.XenonPM4;
    const reg = globalThis.XenonGpuRegisterFile;
    const cp = globalThis.XenonGpuCommandProcessor;
    const ring = globalThis.XenonGpuRingBuffer;
    const webgpu = globalThis.XenonWebGpuBackend;
    if (!pm4g || !reg || !cp || !ring || !webgpu) {
      throw new Error('gpu_host.js: load xenos_registers.js, register_file.js, pm4.js, command_processor.js, ring_buffer.js, vertex_fetch.js, wgsl_translate.js and webgpu_backend.js first');
    }
    return { pm4g, reg, cp, ring, webgpu };
  }

  function makeGuestViews(memory) {
    const bytes = new Uint8Array(memory.buffer);
    const GUEST_MASK = 0x03ffffff; // flat 64 MiB arena, as the recompiler emits it
    return {
      readGuestMemory(addr) {
        const a = Number(BigInt(addr) & 0x03ffffffn) >>> 0;
        return new DataView(memory.buffer, a, memory.buffer.byteLength - a); // full-length view — callers bulk-read, no per-dword allocation
      },
      writeGuestMemory(addr, arr) {
        const a = Number(BigInt(addr) & 0x03ffffffn) >>> 0;
        bytes.set(arr, a);
      },
    };
  }

  /** GPU register aperture — byte offset = register × 4 (must match recompiler.js GPU_MMIO_BASE/END). */
  const GPU_MMIO_BASE = 0x00F02000;
  const GPU_MMIO_END = 0x00F03000;

  function registerName(reg) {
    const r = globalThis.XenonGpuRegisters;
    const e = r && r.XENOS_GPU_REGISTER_BY_INDEX && r.XENOS_GPU_REGISTER_BY_INDEX.get(reg);
    return e && e.name ? e.name : null;
  }

  class XenonGpuHost {
    /**
     * @param {object} opts
     * @param {WebAssembly.Memory} opts.memory — the shared guest arena
     * @param {(level: string, msg: string) => void} [opts.onLog]
     * @param {boolean} [opts.placeholderShader] — pass-through to the backend (see webgpu_backend.js)
     */
    constructor({ memory, onLog = null, placeholderShader = false } = {}) {
      if (!memory || !memory.buffer) throw new TypeError('XenonGpuHost: a WebAssembly.Memory (the guest arena) is required');
      const { cp, ring, webgpu } = loadDeps();
      this.log = typeof onLog === 'function' ? onLog : () => {};
      this.memory = memory;
      const views = makeGuestViews(memory);
      this.views = views;
      this.backend = new webgpu.XenosWebGpuBackend({
        readGuestMemory: views.readGuestMemory,
        writeGuestMemory: views.writeGuestMemory,
        onLog: this.log,
        placeholderShader,
      });
      this.commandProcessor = new cp.XenosCommandProcessor({
        backend: this.backend,
        readGuestMemory: views.readGuestMemory,
        onLog: this.log,
      });
      this.ringBuffer = new ring.XenosRingBuffer({
        commandProcessor: this.commandProcessor,
        readGuestMemory: views.readGuestMemory,
        writeGuestMemory: views.writeGuestMemory,
        onLog: this.log,
      });
    }

    /** Create the WebGPU device and configure `canvas`. Headless (no canvas) attaches for state tracking; draw() still traps as the backend documents. */
    async attachWebGPU(canvas) {
      if (canvas) this.backend.canvas = canvas;
      return this.backend.attach();
    }

    /**
     * Guest GPU MMIO write — recompiler aperture [0x00F02000, 0x00F03000),
     * byte offset = register × 4. Model only what the ring needs:
     *   CP_RB_RPTR  (0x01C4) — reads come via mmioLoad; writes trap
     *   CP_RB_WPTR  (0x01C5) — the doorbell (ring drain, synchronous on this thread)
     * Everything else traps with the register named (see docs/XENOS_GPU.md).
     */
    mmioStore(addr, value, size) {
      const a = Number(BigInt(addr) & 0x03ffffffn) >>> 0;
      if ((a & 3) !== 0) throw new Error(`GPU MMIO: unaligned write at 0x${a.toString(16)} — GPU registers are dword-aligned`);
      if (size !== 4) throw new Error(`GPU MMIO: ${size}-byte write at 0x${a.toString(16)} — only 4-byte (dword) register writes are modeled`);
      const reg = (a - GPU_MMIO_BASE) >>> 2;
      const v32 = Number(BigInt(value) & 0xffffffffn) >>> 0;
      if (reg === 0x01C5) return this.notifyWritePointer(v32);
      const name = registerName(reg);
      throw new Error(`GPU MMIO: write to register 0x${reg.toString(16).padStart(4, '0')}${name ? ` (${name})` : ''} is not modeled — only CP_RB_WPTR (0x01C5) is`);
    }

    /** Guest GPU MMIO read — only CP_RB_RPTR (0x01C4) answers (the ring's current read pointer). */
    mmioLoad(addr, size) {
      const a = Number(BigInt(addr) & 0x03ffffffn) >>> 0;
      if ((a & 3) !== 0) throw new Error(`GPU MMIO: unaligned read at 0x${a.toString(16)} — GPU registers are dword-aligned`);
      if (size !== 4) throw new Error(`GPU MMIO: ${size}-byte read at 0x${a.toString(16)} — only 4-byte (dword) register reads are modeled`);
      const reg = (a - GPU_MMIO_BASE) >>> 2;
      if (reg === 0x01C4) {
        if (!this.ringBuffer.isInitialized()) throw new Error('GPU MMIO: CP_RB_RPTR read before VdInitializeRingBuffer');
        return BigInt(this.ringBuffer.getReadIndex() >>> 0);
      }
      const name = registerName(reg);
      throw new Error(`GPU MMIO: read of register 0x${reg.toString(16).padStart(4, '0')}${name ? ` (${name})` : ''} is not modeled — only CP_RB_RPTR (0x01C4) is`);
    }

    initializeRingBuffer(ptr, sizeLog2) { return this.ringBuffer.initializeRingBuffer(ptr, sizeLog2); }
    enableReadPointerWriteBack(ptr, blockSizeLog2) { return this.ringBuffer.enableReadPointerWriteBack(ptr, blockSizeLog2); }

    /** Guest wrote CP_RB_WPTR — execute the newly submitted packets. */
    notifyWritePointer(value) { return this.ringBuffer.updateWritePointer(value); }

    /** Snapshot for the page/diagnostics. */
    getState() {
      return {
        ring: this.ringBuffer.getState(),
        cp: this.commandProcessor.stats,
        gpuReady: this.backend.ready,
        backend: this.backend.uploads,
      };
    }
  }

  const api = { XenonGpuHost, GPU_MMIO_BASE, GPU_MMIO_END };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonGpuHostFactory = XenonGpuHost;
})(typeof self !== 'undefined' ? self : globalThis);
