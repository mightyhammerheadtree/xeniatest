/**
 * gpu_host.js — wires the Xenos GPU subsystem to the HLE and the page.
 *
 * hle.js (deliberately dependency-free, shipping in the exported zip) needs a
 * way to reach the GPU without importing it: it reads `globalThis.XenonGpuHost`
 * and calls initializeRingBuffer / enableReadPointerWriteBack /
 * notifyWritePointer. This module is that host for the in-page frontend.
 * (The exported zip does not ship the GPU modules yet, so Vd* exports trap
 * there — loudly, with the reason — until it does.)
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
      throw new Error('gpu_host.js: load xenos_registers.js, register_file.js, pm4.js, command_processor.js, ring_buffer.js and webgpu_backend.js first');
    }
    return { pm4g, reg, cp, ring, webgpu };
  }

  function makeGuestViews(memory) {
    const bytes = new Uint8Array(memory.buffer);
    const GUEST_MASK = 0x03ffffff; // flat 64 MiB arena, as the recompiler emits it
    return {
      readGuestMemory(addr) {
        const a = Number(BigInt(addr) & 0x03ffffffn) >>> 0;
        return new DataView(memory.buffer, a, Math.min(4, memory.buffer.byteLength - a));
      },
      writeGuestMemory(addr, arr) {
        const a = Number(BigInt(addr) & 0x03ffffffn) >>> 0;
        bytes.set(arr, a);
      },
    };
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

  const api = { XenonGpuHost };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonGpuHostFactory = XenonGpuHost;
})(typeof self !== 'undefined' ? self : globalThis);
