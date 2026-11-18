/**
 * ring_buffer.js — the Xenos GPU command ring buffer, host-side.
 *
 * xenia model (BSD-3-Clause), gpu/command_processor.cc + graphics_system.cc +
 * kernel/xboxkrnl/xboxkrnl_video.cc:
 *
 *   VdInitializeRingBuffer(ptr, size_log2)  → primary_buffer_ptr_ = ptr;
 *                                              primary_buffer_size_ = 1 << (size_log2 + 3);
 *                                              memset(ring, 0); read_index = 0
 *   VdEnableRingBufferRPtrWriteBack(ptr, block_log2)
 *                                            → read_ptr_writeback_ptr_ = ptr;
 *                                              read_ptr_update_freq_ = 1 << (block_log2 - 2)
 *   guest writes CP_RB_WPTR (register 0x01C5) via GPU MMIO
 *                                            → UpdateWritePointer(value): the "doorbell"
 *   CP worker loop                           → ExecutePrimaryBuffer(read, write):
 *                                              executes packets from the ring with
 *                                              wraparound until read == write,
 *                                              then read_index = write and BE-writes
 *                                              read_index at read_ptr_writeback_ptr_.
 *
 * Indices are dword counts (not bytes) and grow monotonically — they are NOT
 * masked to the ring size; only the ring offset is taken modulo the size.
 *
 * Differences from xenia, deliberate and documented:
 *   - executeCommandBuffer() runs the whole available range synchronously on
 *     the calling thread (there is no CP worker thread yet; the browser would
 *     run one, or the HLE doorbell would drive it inline).
 *   - write_index - read_index > ring dword capacity is a loud XenosGpuError,
 *     not silent garbage execution (xenia relies on guest discipline).
 *   - CP_RB_RPTR writeback happens after EVERY drain (xenia defers this with a
 *     "TODO(benvanik): use reader_->Read_update_freq_" and does the same).
 *   - Ring zeroing at initialize: if writeGuestMemory is wired the ring is
 *     zeroed exactly like xenia; if it is not wired we trap (never pretend
 *     the memset happened).
 *
 * The flat 64 MiB JS arena cannot address the real GPU BAR (0x7FC8_xxxx), so
 * the doorbell is driven by the HLE exports (VdInitializeRingBuffer /
 * VdEnableRingBufferRPtrWriteBack), by gpu_host.js, and — for guest code —
 * through the GPU MMIO aperture at [0x00F02000, 0x00F03000) wired in the
 * recompiler (byte offset = register × 4; CP_RB_WPTR = register 0x01C5).
 * See docs/XENOS_GPU.md ("GPU MMIO aperture").
 */
(function (global) {
  'use strict';

  function pm4() {
    const g = globalThis.XenonPM4;
    if (!g) throw new Error('pm4.js must be loaded before ring_buffer.js');
    return g;
  }

  class XenosRingError extends Error {}

  /** GPU bus is 29 bits wide (xenia CpuToGpu). */
  const GPU_BUS_MASK = 0x1FFFFFFF;

  class XenosRingBuffer {
    /**
     * @param {object} opts
     * @param {object} opts.commandProcessor — XenosCommandProcessor (command_processor.js)
     * @param {(addr: number) => DataView} opts.readGuestMemory — BE view at guest physical `addr`
     * @param {(addr: number, bytes: Uint8Array) => void} opts.writeGuestMemory — apply writes to guest physical memory
     * @param {(level: string, msg: string) => void} [opts.onLog]
     */
    constructor({ commandProcessor, readGuestMemory, writeGuestMemory = null, onLog = null } = {}) {
      if (!commandProcessor || typeof commandProcessor.executeCommandBuffer !== 'function') {
        throw new TypeError('XenosRingBuffer requires a commandProcessor with executeCommandBuffer()');
      }
      if (typeof readGuestMemory !== 'function') throw new TypeError('XenosRingBuffer requires readGuestMemory(addr) => DataView');
      this.commandProcessor = commandProcessor;
      this.readGuestMemory = readGuestMemory;
      this.writeGuestMemory = typeof writeGuestMemory === 'function' ? writeGuestMemory : null;
      this.log = typeof onLog === 'function' ? onLog : () => {};

      /** @type {number|null} guest physical base of the ring (low 32 bits, xenia truncates to uint32 too) */
      this.base = null;
      /** @type {number} ring size in bytes (power of two) */
      this.sizeBytes = 0;
      /** @type {number} absolute dword index of the next packet to execute */
      this.readIndex = 0;
      /** @type {number|null} guest physical address for the read-pointer writeback, or null */
      this.readPtrWritebackAddr = null;
      /** @type {number|null} dwords between writeback updates (recorded; xenia has the same TODO) */
      this.readPtrUpdateFrequency = null;
    }

    isInitialized() { return this.base !== null; }
    getReadIndex() { return this.readIndex >>> 0; }
    getSizeBytes() { return this.sizeBytes; }

    /**
     * VdInitializeRingBuffer — ptr is a guest physical address (low 32 bits,
     * like xenia's uint32 truncation); sizeLog2 is log2 of the size in
     * quadwords (xenia: primary_buffer_size_ = 1 << (size_log2 + 3)).
     */
    initializeRingBuffer(ptr, sizeLog2) {
      const base = Number(BigInt(ptr) & 0xffffffffn) >>> 0;
      const log2 = Number(BigInt(sizeLog2) & 0xffffffffn) >>> 0;
      if (log2 > 28) {
        throw new XenosRingError(`VdInitializeRingBuffer: size_log2 ${log2} too large (max 28 → 2 GiB ring)`);
      }
      const sizeBytes = 1 << (log2 + 3);
      if (this.writeGuestMemory) {
        // xenia memsets the ring so stale bytes decode as harmless NOPs
        try {
          this.writeGuestMemory(base, new Uint8Array(sizeBytes));
        } catch (e) {
          this.log('warn', `VdInitializeRingBuffer: could not zero the ring at 0x${base.toString(16)} (${e.message}); continuing without the memset (xenia zeroes)`);
        }
      } else {
        throw new XenosRingError('VdInitializeRingBuffer: writeGuestMemory is not wired — refusing to continue with an unzeroed ring (xenia zeroes it)');
      }
      this.base = base;
      this.sizeBytes = sizeBytes;
      this.readIndex = 0;
      return { base, sizeBytes };
    }

    /** VdEnableRingBufferRPtrWriteBack — ptr is a guest physical address; blockSizeLog2 as xenia (log2 of quadwords per update). */
    enableReadPointerWriteBack(ptr, blockSizeLog2) {
      const addr = Number(BigInt(ptr) & 0xffffffffn) >>> 0;
      const log2 = Number(BigInt(blockSizeLog2) & 0xffffffffn) >>> 0;
      this.readPtrWritebackAddr = addr;
      this.readPtrUpdateFrequency = (1 << log2) >>> 2; // quadwords → dwords (xenia: 1 << block_size_log2 >> 2)
      return { address: addr, updateFrequencyDwords: this.readPtrUpdateFrequency };
    }

    /**
     * The doorbell — guest wrote CP_RB_WPTR (register 0x01C5). Executes every
     * packet up to `value` (absolute dword index), handling ring wraparound,
     * then advances the read pointer and performs the writeback if enabled.
     *
     * @param {number} value — new write pointer (absolute dword index)
     * @returns {{ executed: boolean, reason?: string, words: number, ringWraps: number, stats?: object }}
     */
    updateWritePointer(value) {
      if (!this.isInitialized()) {
        throw new XenosRingError('GPU doorbell (CP_RB_WPTR) written before VdInitializeRingBuffer — wire VdInitializeRingBuffer first');
      }
      const writeIndex = Number(BigInt(value) & 0xffffffffn) >>> 0;
      // xenia spins on the uninitialized sentinel; we report instead of hanging
      if (writeIndex === 0xBAADF00D) {
        return { executed: false, reason: 'uninitialized write pointer sentinel (0xBAADF00D)', words: 0, ringWraps: 0 };
      }
      if (writeIndex === this.readIndex) {
        return { executed: false, reason: 'ring empty', words: 0, ringWraps: 0 };
      }
      const sizeDwords = this.sizeBytes >>> 2;
      const deltaDwords = (writeIndex - this.readIndex) >>> 0;
      if (deltaDwords > sizeDwords) {
        throw new XenosRingError(`GPU doorbell: write pointer 0x${writeIndex.toString(16)} is more than one ring beyond the read pointer 0x${this.readIndex.toString(16)} (${deltaDwords} dwords > ring capacity ${sizeDwords}) — guest overwrote the ring before the GPU consumed it`);
      }

      // Gather the [readIndex, writeIndex) range, wrapping at the ring end.
      // Contiguous runs are read with one readGuestMemory() call each (≤2 per
      // drain) when the returned view is long enough — implementations handing
      // back short (e.g. 4-byte) views fall back to per-dword reads.
      const words = new Uint32Array(deltaDwords);
      let pos = ((this.readIndex >>> 0) * 4) % this.sizeBytes;
      let ringWraps = 0;
      const baseLo = (this.base & ~GPU_BUS_MASK) >>> 0;
      let i = 0;
      let remaining = deltaDwords;
      while (remaining > 0) {
        const run = Math.min(remaining, (this.sizeBytes - pos) >>> 2);
        const addr = (baseLo | ((this.base + pos) & GPU_BUS_MASK)) >>> 0;
        const view = this.readGuestMemory(addr);
        if (!view) throw new XenosRingError(`GPU ring read out of bounds at 0x${addr.toString(16)}`);
        if ((view.byteLength >>> 2) >= run) {
          for (let j = 0; j < run; j++) words[i++] = view.getUint32(j * 4, false) >>> 0;
        } else {
          for (let j = 0; j < run; j++) {
            const a = (baseLo | ((this.base + pos + j * 4) & GPU_BUS_MASK)) >>> 0;
            const v = this.readGuestMemory(a);
            if (!v) throw new XenosRingError(`GPU ring read out of bounds at 0x${a.toString(16)}`);
            words[i++] = v.getUint32(0, false) >>> 0;
          }
        }
        remaining -= run;
        pos = 0;
        if (remaining > 0) ringWraps++;
      }

      const stats = this.commandProcessor.executeCommandBuffer(words, { hostOrderWords: true });
      this.readIndex = writeIndex;

      if (this.readPtrWritebackAddr !== null) {
        const wptrBytes = new Uint8Array(4);
        new DataView(wptrBytes.buffer).setUint32(0, this.readIndex >>> 0, false);
        if (!this.writeGuestMemory) {
          throw new XenosRingError('GPU ring: read-pointer writeback requested but writeGuestMemory is not wired');
        }
        // xenia masks only low bits when translating; keep the caller's region bits
        this.writeGuestMemory(this.readPtrWritebackAddr, wptrBytes);
      }

      return { executed: true, words: deltaDwords, ringWraps, stats };
    }

    /** Snapshot for diagnostics/tests. */
    getState() {
      return {
        initialized: this.isInitialized(),
        base: this.base,
        sizeBytes: this.sizeBytes,
        readIndex: this.getReadIndex(),
        readPtrWritebackAddr: this.readPtrWritebackAddr,
        readPtrUpdateFrequency: this.readPtrUpdateFrequency,
      };
    }
  }

  const api = { XenosRingBuffer, XenosRingError, GPU_BUS_MASK };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonGpuRingBuffer = api;
})(typeof self !== 'undefined' ? self : globalThis);
