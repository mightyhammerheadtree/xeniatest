/**
 * register_file.js — the Xenos GPU register file: a flat u32 array indexed
 * by the Xenos dword register index (0x0000–0x5002), seeded with the
 * hardware reset defaults, with write history for backend consumption.
 *
 * Semantics mirror xenia's RegisterFile (gpu/register_file.cc): 0x5003
 * u32s; reset defaults per the AMD R6xx/R7xx reference and Mesa r600g;
 * register metadata (name/type) comes from xenos_registers.js.
 *
 * Big-endian note: the guest writes ring words big-endian; consumers of
 * this file see host-order u32s (we decode on read). Raw BE bytes are
 * never stored here.
 */
(function (global) {
  'use strict';

  // Caller must load xenos_registers.js first (classic script or ESM import);
  // it sets globalThis.XenonGpuRegisters.
  function registers() {
    const g = globalThis.XenonGpuRegisters;
    if (!g) throw new Error('xenos_registers.js must be loaded before register_file.js');
    return g;
  }

  const XE_GPU_REG_VGT_DRAW_INITIATOR = 0x21fc;
  const XE_GPU_REG_VGT_DMA_BASE = 0x21fa;
  const XE_GPU_REG_VGT_DMA_SIZE = 0x21fb;
  const XE_GPU_REG_VGT_EVENT_INITIATOR = 0x21f9;
  const XE_GPU_REG_SQ_VS_PROGRAM = 0x21f7;
  const XE_GPU_REG_SQ_PS_PROGRAM = 0x21f6;

  class XenosRegisterFile {
    constructor() {
      const g = registers();
      this.values = new Uint32Array(g.XENOS_GPU_REGISTER_COUNT); // 0x5003
      // Apply hardware reset defaults by name (fail loudly if the table moved)
      for (const [name, value] of Object.entries(g.XENOS_GPU_RESET_DEFAULTS)) {
        const info = g.XENOS_GPU_REGISTER_BY_NAME.get(name);
        if (!info) throw new Error(`reset default references unknown register ${name}`);
        this.values[info.index] = value >>> 0;
      }
      this.resetDefaults = new Map(Object.entries(g.XENOS_GPU_RESET_DEFAULTS).map(([n, v]) => [registers().XENOS_GPU_REGISTER_BY_NAME.get(n).index, v >>> 0]));
      this.writeLog = null; // null = not recording (hot path is allocation-free); enableWriteLog() turns it on
      this.writeCount = 0;
    }

    /** Case-insensitive name → index, or undefined. */
    indexOf(name) {
      const info = registers().XENOS_GPU_REGISTER_BY_NAME.get(String(name).toUpperCase());
      return info ? info.index : undefined;
    }

    info(index) { return registers().XENOS_GPU_REGISTER_BY_INDEX.get(index & 0xffff) ?? null; }
    nameOf(index) { return this.info(index)?.name ?? `Unknown_Reg_${(index & 0xffff).toString(16).toUpperCase().padStart(4, '0')}`; }
    typeOf(index) { return this.info(index)?.type ?? 0; } // 0 dword, 1 float

    /** Raw u32 read (host order). Out-of-range indices throw — never silently mask. */
    read(index) {
      const idx = index >>> 0;
      if (idx >= this.values.length) throw new RangeError(`register index 0x${idx.toString(16)} out of the 0x${this.values.length.toString(16)}-entry file`);
      return this.values[idx] >>> 0;
    }
    readFloat(index) {
      const buf = new ArrayBuffer(4);
      new Uint32Array(buf)[0] = this.read(index);
      return new Float32Array(buf)[0];
    }

    /** Write with logging. Out-of-range indices are dropped with a logged warning — never silently aliased. */
    write(index, value) {
      const idx = index & 0xffff;
      if (idx >= this.values.length) {
        throw new RangeError(`register index 0x${idx.toString(16)} out of the 0x${this.values.length.toString(16)}-entry file`);
      }
      this.values[idx] = value >>> 0;
      this.writeCount++;
      if (this.writeLog !== null) this.writeLog.push({ index: idx, value: value >>> 0 });
    }

    /** Start recording register writes (allocates the log). Recording stays on until takeWriteLog()/resetWriteLog() clear it or enableWriteLog(false). */
    enableWriteLog(on = true) {
      if (on) { if (this.writeLog === null) this.writeLog = []; return true; }
      this.writeLog = null;
      return false;
    }

    /** Drain and return the recorded writes since the last drain (empty array when recording was never enabled). */
    takeWriteLog() {
      const log = this.writeLog ?? [];
      if (this.writeLog !== null) this.writeLog = [];
      return log;
    }

    /** Restore the hardware reset defaults (test aid). */
    reset() {
      this.values.fill(0);
      for (const [index, value] of this.resetDefaults) this.values[index] = value;
      this.writeLog = [];
      this.writeCount = 0;
    }
  }

  const api = {
    XenosRegisterFile,
    XE_GPU_REG_VGT_DRAW_INITIATOR,
    XE_GPU_REG_VGT_DMA_BASE,
    XE_GPU_REG_VGT_DMA_SIZE,
    XE_GPU_REG_VGT_EVENT_INITIATOR,
    XE_GPU_REG_SQ_VS_PROGRAM,
    XE_GPU_REG_SQ_PS_PROGRAM,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonGpuRegisterFile = api;
})(typeof self !== 'undefined' ? self : globalThis);
