/**
 * command_processor.js — the Xenos GPU PM4 command processor, in the browser,
 * in pure JS. Mirrors the decode semantics of xenia's
 * COMMAND_PROCESSOR (gpu/pm4_command_processor_implement.h).
 *
 * The guest submits command buffers via a ring buffer in guest physical
 * memory; the recompiled or HLE'd code kicks the GPU by writing CP_RB_WPTR.
 * This class consumes one command buffer at a time and decodes packets:
 *
 *   type 0  — write a register range from the ring (or repeatedly, one_reg)
 *   type 1  — write two registers
 *   type 2  — no-op
 *   type 3  — opcode dispatch (PM4_*); see IMPLEMENTED below
 *
 * Register writes go to a XenosRegisterFile. Side effects that need a GPU —
 * draws, memory writes, event writes, shader loads — go to a backend object
 * implementing the IXenosGpuBackend interface (webgpu_backend.js in the
 * browser; test doubles in node). Every opcode xenia implements but we have
 * not wired yet, and every unknown opcode, throws XenosGpuError naming the
 * opcode instead of silently skipping — the same honesty rule as the HLE.
 *
 * Command buffer words are big-endian (guest memory); the caller hands us
 * decoded u32s from a BE source (see readCommandBuffer).
 */
(function (global) {
  'use strict';

  function pm4() {
    const g = globalThis.XenonPM4;
    if (!g) throw new Error('pm4.js must be loaded before command_processor.js');
    return g;
  }
  function regFile() {
    const g = globalThis.XenonGpuRegisterFile;
    if (!g) throw new Error('register_file.js must be loaded before command_processor.js');
    return g;
  }

  class XenosGpuError extends Error {}

  // SHADER_CONSTANT_000_X = 0x4000 (ALU), FETCH = 0x4800, BOOL = 0x4900, LOOP = 0x4908 — dword indices into the register file
  const SHADER_CONSTANT_ALU_BASE = 0x4000;
  const SHADER_CONSTANT_FETCH_BASE = 0x4800;
  const SHADER_CONSTANT_BOOL_BASE = 0x4900;
  const SHADER_CONSTANT_LOOP_BASE = 0x4908;
  const REGISTER_CONSTANT_BASE = 0x2000; // xenia: WriteREGISTERSRange* uses base + 0x2000 (context registers)

  const CONSTANT_TYPE = Object.freeze({ ALU: 0, FETCH: 1, BOOL: 2, LOOP: 3, REGISTERS: 4 });

  /**
   * @typedef {object} IXenosGpuBackend
   * Methods receive host-order values. All are optional except draw() —
   * missing methods trap loudly with the opcode that needed them.
   *
   *   memWrite(address /* guest physical, 0x3FFFFFFF-masked *\/, bytes /* Uint8Array *\/)
   *   draw({ primType, numIndices, isIndexed, indexGuestBase, indexCount, indexFormat /* 0=int16 1=int32 *\/, indexEndianness /* swap_mode 0..3 *\/, multiPrimIbEnable /* PA_SU_SC_MODE_CNTL.multi_prim_ib_ena *\/, multiPrimResetIndex /* VGT_MULTI_PRIM_IB_RESET_INDX *\/, vizQueryCondition, majorModeExplicit, vertexFetchConstants /* 192 dwords *\/, aluConstants /* 1024 dwords *\/, boolConstants /* 8 dwords *\/, renderTargets /* raw RB_* render-target register snapshot for the backend to decode (write mask, blend, depth/stencil, alpha test, PA_SU_* rasterizer state) *\/, scissor /* raw PA_SC_* scissor register snapshot *\/, depthOnly /* RB_MODECONTROL.edram_mode = kDepth: depth only, no color output *\/ })
   *   setShaders({ vertex, pixel })           // loaded via IM_LOAD*, shader objects are opaque to the CP
   *   eventWrite({ eventType, address /* may be 0 for none *\/, data /* may be null *\/, timestamp })
   *   interrupt(mask)                          // PM4_INTERRUPT — guest-visible interrupt request
   *   swap()                                   // PM4_XE_SWAP — present/flip notification (xenia-specific)
   *   vizQuery(state /* 'start'|'end' *\/)     // PM4_VIZ_QUERY begin/end
   *   invalidateState(mask)                    // PM4_INVALIDATE_STATE
   *   setShaderBases(vsBase, psBase)           // PM4_SET_SHADER_BASES
   *   log(level, message)                      // optional diagnostics sink
   */

  function backendOrTrap(backend, method, opcodeName) {
    const fn = backend && backend[method];
    if (typeof fn !== 'function') {
      throw new XenosGpuError(`${opcodeName}: backend does not implement ${method}() — wire up a WebGPU backend (webgpu_backend.js) or a test double`);
    }
    return fn.bind(backend);
  }

  class XenosCommandProcessor {
    /**
     * @param {object} opts
     * @param {object} [opts.backend] — IXenosGpuBackend implementation (optional; required only for packets with side effects)
     * @param {(guestPhysical: number) => DataView} [opts.readGuestMemory] — translate+view guest physical memory (BE). Required for LOAD_* packets (constants, IM_LOAD from pointer).
     * @param {(level: string, msg: string) => void} [opts.onLog]
     */
    constructor({ backend = null, readGuestMemory = null, onLog = null } = {}) {
      this.backend = backend;
      this.readGuestMemory = readGuestMemory;
      const logFn = typeof onLog === 'function' ? onLog : () => {};
      this.log = logFn;
      this.registers = new (regFile().XenosRegisterFile)();
      /** @type {bigint} 64-bit bin mask/select for predicated (tiled) rendering */
      this.binMask = 0n;
      this.binSelect = 0n;
      /** Bin base offset programmed by PM4_SET_BIN_BASE_OFFSET (tiled rendering; recorded only). */
      this.binBaseOffset = 0;
      /** Shaders loaded via IM_LOAD*; opaque descriptors owned by the backend. */
      this.vertexShader = null;
      this.pixelShader = null;
      /** Viz query bookkeeping: number of open surveys (depth may exceed 1 across nested flushes). */
      /** GPU event counter written back by EVENT_WRITE* with the counter bit (initiator & 0x80000000). */
      this.counter = 0;
      /** WRITEBACK_START/SIZE register indices (guest physical window for event writebacks). */
      this.writebackStartIndex = 0x0A04;
      this.writebackSizeIndex = 0x0A05;
      /** Stats for tests and diagnostics. */
      this.stats = { packets: 0, draws: 0, constantBytes: 0, memWriteBytes: 0, registerWrites: 0, indirectBuffers: 0, unimplementedTraps: 0 };
    }

    /**
     * Execute a command buffer.
     * @param {Uint8Array | Uint32Array | Array<number> | number[]} words — big-endian guest words OR pre-decoded host-order u32 array
     * @param {object} [opts]
     * @param {boolean} [opts.hostOrderWords] — when true, `words` are already decoded u32s (default false for Uint8Array, true otherwise)
     */
    executeCommandBuffer(words, { hostOrderWords } = {}) {
      let u32s;
      if (words instanceof Uint8Array) {
        if (words.length % 4 !== 0) throw new XenosGpuError('command buffer length is not a multiple of 4 bytes');
        const view = new DataView(words.buffer, words.byteOffset, words.byteLength);
        u32s = new Uint32Array(words.length / 4);
        for (let i = 0; i < u32s.length; i++) u32s[i] = view.getUint32(i * 4, false) >>> 0;
      } else if (words instanceof Uint32Array || Array.isArray(words)) {
        u32s = words instanceof Uint32Array ? words : Uint32Array.from(words, (w) => w >>> 0);
        hostOrderWords = true; // arrays are decoded values
      } else {
        throw new TypeError('command buffer must be Uint8Array, Uint32Array or number[]');
      }
      this._executeRange(u32s, 0, u32s.length, /* indirectDepth */ 0);
      return this.stats;
    }

    /** Parse a BE ring snapshot and execute only up to `usedBytes` (test/ring-buffer helper). */
    executeRing(memory /* Uint8Array of the ring */, usedBytes) {
      if (!(memory instanceof Uint8Array)) throw new TypeError('ring must be a Uint8Array');
      const used = Math.min(usedBytes >>> 0, memory.length - (memory.length % 4));
      return this.executeCommandBuffer(memory.subarray(0, used));
    }

    _executeRange(u32s, start, end, indirectDepth) {
      let offset = start >>> 0;
      const stop = end >>> 0;
      while (offset < stop) {
        const header = u32s[offset++] >>> 0;
        if (header === 0 || header === 0xbadf00d) { offset += 0; this.stats.packets++; continue; } // bad packet: xenia treats as padding
        if (header === 0xcdcdcdcd) { this.log('warn', 'PM4 packet 0xCDCDCDCD (uninitialized memory) — executing anyway, like xenia'); }
        const parsed = pm4().parsePacketHeader(header);
        this.stats.packets++;
        if (parsed.type === 0) {
          if (offset + parsed.count > stop) throw new XenosGpuError(`type-0 packet at word ${offset - 1}: truncated (${parsed.count} dwords needed, ${stop - offset} available)`);
          const index = parsed.baseIndex;
          if (parsed.writeOneReg) {
            // xenia's WriteOneRegisterFromRing reads a fresh value per write, all to the same target index
            for (let i = 0; i < parsed.count; i++) this._writeRegister(index, u32s[offset++]);
          } else {
            for (let i = 0; i < parsed.count; i++) this._writeRegister(index + i, u32s[offset++]);
          }
          continue;
        }
        if (parsed.type === 1) {
          if (offset + 2 > stop) throw new XenosGpuError(`type-1 packet at word ${offset - 1}: truncated`);
          this._writeRegister(parsed.reg1, u32s[offset++]);
          this._writeRegister(parsed.reg2, u32s[offset++]);
          continue;
        }
        if (parsed.type === 2) { continue; } // no-op
        // type 3
        if (offset + parsed.count > stop) throw new XenosGpuError(`${parsed.opcodeName} at word ${offset - 1}: truncated (${parsed.count} dwords needed, ${stop - offset} available)`);
        const payloadStart = offset;
        if (parsed.predicate) {
          // xenia: a predicated type-3 packet is skipped when the bin
          // predicate fails — and a predicated swap is skipped unconditionally
          // ("we also skip predicated swaps, as they are never valid").
          const anyPass = (this.binSelect & this.binMask) !== 0n;
          if (!anyPass || parsed.opcode === pm4().PM4_OPCODE.PM4_XE_SWAP) {
            offset += parsed.count; // skipped by the bin predicate
            continue;
          }
        }
        offset = this._executeType3(parsed, u32s, payloadStart, payloadStart + parsed.count, indirectDepth);
        if (offset !== payloadStart + parsed.count) {
          throw new XenosGpuError(`${parsed.opcodeName}: packet consumed ${offset - payloadStart} dwords, expected ${parsed.count}`);
        }
      }
    }

    _writeRegister(index, value) {
      this.registers.write(index, value);
      this.stats.registerWrites++;
    }

    _readGuestU32s(address, count) {
      if (!this.readGuestMemory) {
        throw new XenosGpuError('LOAD_* packets require readGuestMemory() to be wired to guest physical memory');
      }
      const view = this.readGuestMemory(address >>> 0);
      if (!view) throw new XenosGpuError(`guest physical address 0x${(address >>> 0).toString(16)} is not mapped`);
      const out = new Uint32Array(count);
      for (let i = 0; i < count; i++) out[i] = view.getUint32(i * 4, false) >>> 0;
      return out;
    }

    _executeType3(parsed, u32s, start, end, indirectDepth) {
      const op = pm4().PM4_OPCODE;
      const opcode = parsed.opcode;
      const count = parsed.count;
      switch (opcode) {
        case op.PM4_CONTEXT_UPDATE: {
          // xenia: PM4_CONTEXT_UPDATE asserts a single dword payload of 0 and
          // continues (it is a no-op); anything else asserts in xenia too.
          if (count !== 1) throw new XenosGpuError('PM4_CONTEXT_UPDATE: expected 1 payload dword');
          const value = u32s[start] >>> 0;
          if (value !== 0) {
            throw new XenosGpuError(
              `PM4_CONTEXT_UPDATE: value 0x${value.toString(16).toUpperCase()} is not 0 — xenia asserts on any ` +
              'other context-update value. Refusing to continue');
          }
          return end;
        }
        case op.PM4_NOP:
          return end;

        // --- register writes -------------------------------------------------
        case op.PM4_SET_CONSTANT2: // INCR_UPDATE_STATE: direct register range write
        case op.PM4_SET_SHADER_CONSTANTS: { // INCR_UPDT_CONST: same encoding
          const index = u32s[start] & 0xffff;
          for (let i = 0; i < count - 1; i++) this._writeRegister(index + i, u32s[start + 1 + i]);
          this.stats.constantBytes += (count - 1) * 4;
          return end;
        }
        case op.PM4_SET_CONSTANT: {
          const offsetType = u32s[start];
          const index = offsetType & 0x7ff;
          const type = (offsetType >>> 16) & 0xff;
          const data = u32s.subarray(start + 1, end);
          this._loadConstantType(type, index, data, parsed.opcodeName);
          return end;
        }
        case op.PM4_LOAD_ALU_CONSTANT: {
          const address = u32s[start] & 0x3fffffff;
          const offsetType = u32s[start + 1];
          const index = offsetType & 0x7ff;
          const type = (offsetType >>> 16) & 0xff;
          const sizeDwords = u32s[start + 2] & 0xfff;
          const data = this._readGuestU32s(address, sizeDwords);
          this._loadConstantType(type, index, data, parsed.opcodeName);
          return end;
        }

        // --- instruction memory (shaders) ------------------------------------
        case op.PM4_IM_LOAD: {
          const addrType = u32s[start];
          const shaderType = addrType & 0x3;
          const address = addrType & ~0x3;
          const startSize = u32s[start + 1];
          const sizeDwords = startSize & 0xffff;
          const code = this._readGuestU32s(address & 0x3fffffff, sizeDwords);
          this._loadShader(shaderType, code, parsed.opcodeName);
          return end;
        }
        case op.PM4_IM_LOAD_IMMEDIATE: {
          const shaderType = u32s[start];
          const startSize = u32s[start + 1];
          const sizeDwords = startSize & 0xffff;
          const code = u32s.subarray(start + 2, start + 2 + sizeDwords);
          this._loadShader(shaderType, Uint32Array.from(code), parsed.opcodeName);
          return end;
        }
        case op.PM4_SET_SHADER_BASES: {
          const value = u32s[start];
          const vsBase = value & 0x7fffffff;
          const psBase = u32s[start + 1] & 0x7fffffff;
          if (this.backend && typeof this.backend.setShaderBases === 'function') this.backend.setShaderBases(vsBase, psBase);
          else this.log('debug', `SET_SHADER_BASES vs=0x${vsBase.toString(16)} ps=0x${psBase.toString(16)} (no backend sink)`);
          return end;
        }
        case op.PM4_SET_STATE: {
          // xenia does not implement this either (its switch has no case) — we record the payload and move on;
          // state-block decode (shader DMA sub-blocks) is part of the Xenos work backlog.
          this.log('info', `SET_STATE: recorded ${count} dword payload (shader state blocks not decoded yet)`);
          return end;
        }
        case op.PM4_ME_INIT:
          this.log('debug', `ME_INIT: ${count} dword micro-engine init (recorded)`);
          return end;

        // --- memory writes ---------------------------------------------------
        case op.PM4_MEM_WRITE: {
          const addrWord = u32s[start];
          const endianness = addrWord & 0x3;
          let address = (addrWord & ~0x3) >>> 0;
          const data = u32s.subarray(start + 1, end);
          const bytes = new Uint8Array(data.length * 4);
          const view = new DataView(bytes.buffer);
          for (let i = 0; i < data.length; i++) view.setUint32(i * 4, pm4().gpuSwap32(data[i], endianness), false);
          backendOrTrap(this.backend, 'memWrite', parsed.opcodeName)(address >>> 0, bytes);
          this.stats.memWriteBytes += bytes.length;
          return end;
        }
        case op.PM4_REG_TO_MEM: {
          const regAddr = u32s[start];
          const memAddrWord = u32s[start + 1];
          const endianness = memAddrWord & 0x3;
          const memAddr = memAddrWord & ~0x3;
          const value = pm4().gpuSwap32(this.registers.read(regAddr >>> 0), endianness);
          const bytes = new Uint8Array(4);
          new DataView(bytes.buffer).setUint32(0, value >>> 0, false);
          backendOrTrap(this.backend, 'memWrite', parsed.opcodeName)(memAddr >>> 0, bytes);
          this.stats.memWriteBytes += 4;
          return end;
        }
        case op.PM4_REG_RMW: {
          const rmwInfo = u32s[start];
          const andMask = u32s[start + 1];
          const orMask = u32s[start + 2];
          const index = rmwInfo & 0x1fff; // xenia masks the RMW target at 13 bits
          let value = this.registers.read(index);
          value = (rmwInfo & 0x80000000) ? (value & this.registers.read(andMask & 0x1fff)) : (value & andMask);
          value = (rmwInfo & 0x40000000) ? (value | this.registers.read(orMask & 0x1fff)) : (value | orMask);
          this._writeRegister(index, value >>> 0);
          return end;
        }

        // --- draws -----------------------------------------------------------
        case op.PM4_DRAW_INDX: {
          const vizQueryCondition = u32s[start];
          return this._draw(u32s, start + 1, end, parsed.opcodeName, vizQueryCondition, true);
        }
        case op.PM4_DRAW_INDX_2:
          return this._draw(u32s, start, end, parsed.opcodeName, 0, false);
        case op.PM4_DRAW_INDX_BIN:
        case op.PM4_DRAW_INDX_2_BIN:
          // Tiled (binned) rendering. The binned draw packets carry the bin
          // base and the bin size *ahead of* the index-buffer DMA base/size
          // (the R6xx/R7xx tiled layout the Xenos inherits):
          //   PM4_DRAW_INDX_BIN   : viz query, draw indicator, bin base, bin
          //                          size, dma base, dma size
          //   PM4_DRAW_INDX_2_BIN : draw indicator, bin base, bin size,
          //                          dma base, dma size
          // xenia implements neither: its switch has no case for them, so the
          // packet falls into the default branch — "Unimplemented GPU OPCODE"
          // and an assert (in a release build the assert is compiled out and
          // the packet is skipped). This port cannot honor a bin draw either
          // (there is no bin/EDRAM tiling here), and decoding one as an
          // ordinary draw would read the bin base and bin size as the
          // index-buffer DMA base and size — a silent miscompile. Refuse.
          throw new XenosGpuError(
            `${parsed.opcodeName}: tiled rendering (a binned draw) is not implemented in this port — the packet ` +
            'carries a bin base and bin size that an ordinary draw cannot honor, and xenia does not implement ' +
            'the opcode either (its switch logs "Unimplemented GPU OPCODE" and asserts). Refusing rather than ' +
            'rasterizing the bin draw as a normal draw');

        // --- events / synchronization ----------------------------------------
        case op.PM4_EVENT_WRITE:
        case op.PM4_EVENT_WRITE_SHD: {
          this._eventWriteToMemory(u32s, start, end, parsed.opcodeName);
          return end;
        }
        case op.PM4_EVENT_WRITE_EXT: {
          // xenia writes the screen extent (six 16-bit values) to guest memory.
          this._eventWriteScreenExtent(u32s, start, end, parsed.opcodeName);
          return end;
        }
        case op.PM4_EVENT_WRITE_CFL:
        case op.PM4_EVENT_WRITE_ZPD: {
          // cache-flush / z-pass events: the initiator is recorded and the event
          // is forwarded to the backend — xenia implements neither writeback
          // (the z-pass counter is a TODO there), so nothing is written to guest
          // memory.
          const initiator = u32s[start];
          if (this.backend && typeof this.backend.eventWrite === 'function') this.backend.eventWrite({ eventType: initiator & 0x3f, address: 0, data: null, opcode: parsed.opcodeName });
          else this.log('info', `${parsed.opcodeName} initiator 0x${initiator.toString(16)} (no backend sink)`);
          this._writeRegister(regFile().XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3f);
          return end;
        }
        case op.PM4_VIZ_QUERY:
          // Occlusion queries cannot be produced here (the query path — querySet, resolve,
          // guest-visible PA_SC_VIZ_QUERY_STATUS results — is step 9). Silently swallowing the
          // packet would leave the guest polling a status that never lands, so refuse loudly.
          throw new XenosGpuError(`${parsed.opcodeName}: occlusion queries are not supported (step 9) — results cannot be produced; refusing to silently no-op (the draw path still honors PA_SC_VIZ_QUERY ena+kill by dropping hint draws)`);
        case op.PM4_INTERRUPT: {
          const mask = u32s[start];
          if (this.backend && typeof this.backend.interrupt === 'function') this.backend.interrupt(mask);
          else this.log('info', `INTERRUPT mask 0x${mask.toString(16)} (no backend sink)`);
          return end;
        }
        case op.PM4_INVALIDATE_STATE: {
          const mask = u32s[start];
          if (this.backend && typeof this.backend.invalidateState === 'function') this.backend.invalidateState(mask);
          else this.log('debug', `INVALIDATE_STATE mask 0x${mask.toString(16)} (no backend sink)`);
          return end;
        }
        case op.PM4_XE_SWAP: {
          // xenia: the swap packet starts with the 'SWAP' magic (asserted there)
          // and bumps the GPU frame counter that EVENT_WRITE_SHD writes back.
          if (end - start >= 1) {
            const magic = u32s[start] >>> 0;
            if (magic !== 0x53574150 /* 'SWAP' */) {
              this.log('warn', `XE_SWAP: unexpected magic 0x${magic.toString(16).toUpperCase()} (xenia asserts 'SWAP' here)`);
            }
          }
          if (this.backend && typeof this.backend.swap === 'function') this.backend.swap();
          else this.log('info', 'XE_SWAP (present) — no backend sink');
          this.counter = (this.counter + 1) >>> 0;
          return end;
        }

        // --- control flow ----------------------------------------------------
        case op.PM4_INDIRECT_BUFFER:
        case op.PM4_INDIRECT_BUFFER_PFD: {
          if (indirectDepth >= 4) throw new XenosGpuError(`${parsed.opcodeName}: indirect buffer nesting too deep`);
          const address = u32s[start] & 0x3fffffff;
          const count = u32s[start + 1] >>> 0; // dwords
          const data = this._readGuestU32s(address, count);
          this.stats.indirectBuffers++;
          this._executeRange(data, 0, data.length, indirectDepth + 1);
          return end;
        }
        case op.PM4_COND_EXEC: {
          // A conditional block of packets. xenia does not implement this
          // opcode either (its switch has no case for it — the default branch
          // logs "Unimplemented GPU OPCODE" and skips the packet, which would
          // run the conditional block unconditionally). This port refuses
          // loudly instead of running a block the guest meant to be
          // conditional.
          throw new XenosGpuError(
            'PM4_COND_EXEC: conditional execution is not implemented — xenia does not implement the opcode either ' +
            '(its command processor logs "Unimplemented GPU OPCODE" and skips the packet, which would run the ' +
            'conditional block unconditionally). Refusing rather than running a block the guest meant to be conditional');
        }
        case op.PM4_COND_WRITE:
          return this._condWrite(u32s, start, end, parsed.opcodeName);
        case op.PM4_WAIT_REG_MEM:
          return this._waitRegMem(u32s, start, end, parsed.opcodeName);
        case op.PM4_WAIT_REG_EQ:
        case op.PM4_WAIT_REG_GTE:
        case op.PM4_WAIT_FOR_IDLE:
        case op.PM4_WAIT_UNTIL_READ:
        case op.PM4_WAIT_IB_PFD_COMPLETE:
          // xenia does not implement these opcodes either — its command
          // processor logs "Unimplemented GPU OPCODE" and skips the packet.
          return end;

        // --- binning (predication) ------------------------------------------
        case op.PM4_SET_BIN_MASK_LO:
          this.binMask = (this.binMask & 0xffffffff00000000n) | BigInt(u32s[start] >>> 0);
          return end;
        case op.PM4_SET_BIN_MASK_HI:
          this.binMask = (this.binMask & 0x00000000ffffffffn) | (BigInt(u32s[start] >>> 0) << 32n);
          return end;
        case op.PM4_SET_BIN_SELECT_LO:
          this.binSelect = (this.binSelect & 0xffffffff00000000n) | BigInt(u32s[start] >>> 0);
          return end;
        case op.PM4_SET_BIN_SELECT_HI:
          this.binSelect = (this.binSelect & 0x00000000ffffffffn) | (BigInt(u32s[start] >>> 0) << 32n);
          return end;
        case op.PM4_SET_BIN_MASK: {
          if (count !== 2) throw new XenosGpuError('PM4_SET_BIN_MASK: expected 2 payload dwords');
          this.binMask = (BigInt(u32s[start] >>> 0) << 32n) | BigInt(u32s[start + 1] >>> 0);
          return end;
        }
        case op.PM4_SET_BIN_SELECT: {
          if (count !== 2) throw new XenosGpuError('PM4_SET_BIN_SELECT: expected 2 payload dwords');
          this.binSelect = (BigInt(u32s[start] >>> 0) << 32n) | BigInt(u32s[start + 1] >>> 0);
          return end;
        }
        case op.PM4_SET_BIN_BASE_OFFSET:
          // The offset added to the bin base of a DRAW_INDX_BIN packet.
          // xenia does not implement this opcode either (its switch has no
          // case for it), so it is only meaningful for the binned draws —
          // which this port refuses. It is recorded for diagnostics and the
          // packet is skipped.
          this.binBaseOffset = u32s[start] >>> 0;
          this.log('debug', `SET_BIN_BASE_OFFSET 0x${this.binBaseOffset.toString(16)} (recorded; tiled rendering is not implemented)`);
          return end;

        default:
          this.stats.unimplementedTraps++;
          throw new XenosGpuError(`${parsed.opcodeName}: unimplemented GPU opcode 0x${opcode.toString(16).toUpperCase().padStart(2, '0')} (count ${count}) — see docs/XENOS_GPU.md`);
      }
    }

    /**
     * xenia's PM4_WAIT_REG_MEM: wait until a register or a memory location
     * satisfies a comparison.
     *
     * xenia polls in a loop and blocks until the condition matches (sleeping
     * wait/256 ms between polls when wait >= 0x100). This port cannot block —
     * the command ring is consumed inline — so the comparison is evaluated
     * once with xenia's exact rules; if it does not match, xenia would sleep
     * and re-poll forever, and refusing loudly is the only honest answer.
     */
    _waitRegMem(u32s, start, end, opcodeName) {
      if (end - start < 5) {
        throw new XenosGpuError(`${opcodeName}: expected 5 payload dwords (wait_info, address, reference, mask, wait)`);
      }
      const waitInfo = u32s[start] >>> 0;
      const addrWord = u32s[start + 1] >>> 0;
      const ref = u32s[start + 2] >>> 0;
      const mask = u32s[start + 3] >>> 0;
      const value = this._readWaitValue(waitInfo, addrWord, opcodeName);
      const source = (waitInfo & 0x10)
        ? `memory 0x${((addrWord & ~0x3) >>> 0).toString(16)}`
        : `register 0x${(addrWord >>> 0).toString(16)}`;
      const matched = this._compareWaitValue(waitInfo, value, ref, mask);
      if (!matched) {
        throw new XenosGpuError(
          `${opcodeName}: the wait condition is not met — ${source} holds 0x${value.toString(16).toUpperCase()} ` +
          `(masked 0x${((value & mask) >>> 0).toString(16).toUpperCase()}), the reference is 0x${ref.toString(16).toUpperCase()} ` +
          `(compare 0x${(waitInfo & 0x7).toString(16)}). xenia polls in a loop and blocks until it matches; this port ` +
          'cannot block (the command ring is consumed inline), so it refuses instead of pretending the wait succeeded');
      }
      return end;
    }

    /**
     * xenia's wait comparison (WAIT_REG_MEM and COND_WRITE share it):
     * wait_info bits 0:2 select never/less/less-equal/equal/not-equal/
     * greater-equal/greater/always, and the reference is compared against the
     * *masked* value.
     */
    _compareWaitValue(waitInfo, value, ref, mask) {
      const masked = (value & mask) >>> 0;
      switch (waitInfo & 0x7) {
        case 0x0: return false; // never
        case 0x1: return masked < ref; // less than
        case 0x2: return masked <= ref; // less than or equal
        case 0x3: return masked === ref; // equal
        case 0x4: return masked !== ref; // not equal
        case 0x5: return masked >= ref; // greater than or equal
        case 0x6: return masked > ref; // greater than
        case 0x7: return true; // always
        default: return false;
      }
    }

    /**
     * Read the value xenia's WAIT_REG_MEM/COND_WRITE poll: a register, or a
     * memory location (wait_info bit 4) whose address's low 2 bits carry the
     * GpuSwap endianness.
     */
    _readWaitValue(waitInfo, addrWord, opcodeName) {
      if (waitInfo & 0x10) {
        const endianness = addrWord & 0x3;
        const address = (addrWord & ~0x3) >>> 0;
        if (!this.readGuestMemory) {
          throw new XenosGpuError(`${opcodeName}: reading memory needs readGuestMemory() to be wired to guest physical memory`);
        }
        const view = this.readGuestMemory(address);
        if (!view || view.byteLength < 4) {
          throw new XenosGpuError(`${opcodeName}: memory location 0x${address.toString(16)} is not readable`);
        }
        return pm4().gpuSwap32(view.getUint32(0, false), endianness);
      }
      const index = addrWord >>> 0;
      // xenia: assert_true(poll_reg_addr < RegisterFile::kRegisterCount = 0x5003).
      if (index >= 0x5003) {
        throw new XenosGpuError(`${opcodeName}: register index 0x${index.toString(16)} is outside the register file`);
      }
      return this.registers.read(index) >>> 0;
    }

    /**
     * xenia's PM4_COND_WRITE: a conditional write to a register or to memory.
     * The value is polled (a register, or a memory location when wait_info
     * bit 4 is set) and compared with xenia's rules; when it matches, the
     * payload is written to the destination (memory when wait_info bit 8 is
     * set, a register otherwise).
     */
    _condWrite(u32s, start, end, opcodeName) {
      if (end - start < 6) {
        throw new XenosGpuError(`${opcodeName}: expected 6 payload dwords (wait_info, address, reference, mask, destination, data)`);
      }
      const waitInfo = u32s[start] >>> 0;
      const pollAddr = u32s[start + 1] >>> 0;
      const ref = u32s[start + 2] >>> 0;
      const mask = u32s[start + 3] >>> 0;
      const destAddrWord = u32s[start + 4] >>> 0;
      const data = u32s[start + 5] >>> 0;
      const value = this._readWaitValue(waitInfo, pollAddr, opcodeName);
      if (!this._compareWaitValue(waitInfo, value, ref, mask)) return end;
      if (waitInfo & 0x100) {
        // Memory destination; the address's low 2 bits carry the endianness
        // (xenia: GpuSwap(write_data, endianness) then a plain store).
        const endianness = destAddrWord & 0x3;
        const address = (destAddrWord & ~0x3) >>> 0;
        const bytes = new Uint8Array(4);
        new DataView(bytes.buffer).setUint32(0, pm4().gpuSwap32(data >>> 0, endianness), false);
        backendOrTrap(this.backend, 'memWrite', opcodeName)(address >>> 0, bytes);
        this.stats.memWriteBytes += 4;
      } else {
        // Register destination. xenia's WriteRegister drops an out-of-range
        // index with a warning instead of writing it.
        const index = destAddrWord >>> 0;
        if (index >= 0x5003) {
          this.log('warn', `${opcodeName}: register index 0x${index.toString(16)} is outside the register file — the write is dropped`);
        } else {
          this._writeRegister(index, data >>> 0);
        }
      }
      return end;
    }

    _eventWriteToMemory(u32s, start, end, opcodeName) {
      const initiator = u32s[start];
      const eventType = initiator & 0x3f;
      this._writeRegister(regFile().XE_GPU_REG_VGT_EVENT_INITIATOR, eventType);
      if (end - start === 1) {
        // flag-only event — no writeback target
        if (this.backend && typeof this.backend.eventWrite === 'function') this.backend.eventWrite({ eventType, address: 0, data: null, opcode: opcodeName });
        else this.log('debug', `${opcodeName} flag-only event 0x${eventType.toString(16)} (no backend sink)`);
        return;
      }
      const addrWord = u32s[start + 1];
      const endianness = addrWord & 0x3;
      let address = (addrWord & ~0x3) >>> 0;
      const value = (initiator & 0x80000000) ? this.counter : (start + 2 < end ? u32s[start + 2] : 0) >>> 0;
      // xenia: addresses above the 29-bit bus in the WRITEBACK_START window map to 0x7F000000 + (addr - WRITEBACK_START)
      if (address > 0x1fffffff) {
        const wbBase = this.registers.read(this.writebackStartIndex);
        const wbSize = this.registers.read(this.writebackSizeIndex);
        const off = (address - wbBase) >>> 0;
        if (wbBase !== 0 && off < wbSize) address = (0x7f000000 + off) >>> 0;
      }
      const bytes = new Uint8Array(4);
      new DataView(bytes.buffer).setUint32(0, pm4().gpuSwap32(value >>> 0, endianness), false);
      backendOrTrap(this.backend, 'memWrite', opcodeName)(address >>> 0, bytes);
      this.stats.memWriteBytes += 4;
    }

    /**
     * xenia's PM4_EVENT_WRITE_EXT: a screen-extent writeback.
     * xenia writes six 16-bit values — {min x, max x, min y, max y, min z,
     * max z} = {0, 2560>>3, 0, 2560>>3, 0, 1} — to the guest address, with the
     * bytes of each half word swapped when the address carries the k8in16
     * endianness (xenia asserts that it does).
     */
    _eventWriteScreenExtent(u32s, start, end, opcodeName) {
      const initiator = u32s[start] >>> 0;
      this._writeRegister(regFile().XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3f);
      if (this.backend && typeof this.backend.eventWrite === 'function') {
        this.backend.eventWrite({ eventType: initiator & 0x3f, address: 0, data: null, opcode: opcodeName });
      }
      if (end - start < 2) {
        // xenia reads the address unconditionally (the packet always has one).
        this.log('debug', `${opcodeName}: no writeback address in the packet`);
        return;
      }
      const addrWord = u32s[start + 1] >>> 0;
      const endianness = addrWord & 0x3;
      let address = (addrWord & ~0x3) >>> 0;
      if (address > 0x1fffffff) {
        const wbBase = this.registers.read(this.writebackStartIndex);
        const wbSize = this.registers.read(this.writebackSizeIndex);
        const off = (address - wbBase) >>> 0;
        if (wbBase !== 0 && off < wbSize) address = (0x7f000000 + off) >>> 0;
      }
      // xenia: 0 >> 3, 2560 >> 3, 0 >> 3, 2560 >> 3, 0, 1
      const extents = [0, 2560 >> 3, 0, 2560 >> 3, 0, 1];
      const bytes = new Uint8Array(extents.length * 2);
      for (let i = 0; i < extents.length; i++) {
        const v = extents[i] & 0xffff;
        // xenia's GpuSwap(uint16): kNone leaves it, k8in16 swaps the bytes in
        // the half word; the other endiannesses are unhandled in xenia (the
        // value is written unchanged there).
        if (endianness === 1) {
          bytes[i * 2] = (v >>> 8) & 0xff;
          bytes[i * 2 + 1] = v & 0xff;
        } else {
          bytes[i * 2] = v & 0xff;
          bytes[i * 2 + 1] = (v >>> 8) & 0xff;
        }
      }
      backendOrTrap(this.backend, 'memWrite', opcodeName)(address >>> 0, bytes);
      this.stats.memWriteBytes += bytes.length;
    }

    _loadConstantType(type, index, data, opcodeName) {
      const base = { [CONSTANT_TYPE.ALU]: SHADER_CONSTANT_ALU_BASE, [CONSTANT_TYPE.FETCH]: SHADER_CONSTANT_FETCH_BASE, [CONSTANT_TYPE.BOOL]: SHADER_CONSTANT_BOOL_BASE, [CONSTANT_TYPE.LOOP]: SHADER_CONSTANT_LOOP_BASE }[type];
      if (base === undefined) {
        if (type === CONSTANT_TYPE.REGISTERS) {
          for (let i = 0; i < data.length; i++) this._writeRegister(REGISTER_CONSTANT_BASE + index + i, data[i]);
          this.stats.constantBytes += data.length * 4;
          return;
        }
        throw new XenosGpuError(`${opcodeName}: unsupported constant type ${type} (ALU=0, FETCH=1, BOOL=2, LOOP=3, REGISTERS=4 supported)`);
      }
      for (let i = 0; i < data.length; i++) this._writeRegister(base + index + i, data[i]);
      this.stats.constantBytes += data.length * 4;
    }

    _loadShader(shaderType, codeDwords, opcodeName) {
      // xenos::ShaderType: kVertex = 0, kPixel = 1 (see xenos.h)
      const kind = shaderType === 0 ? 'vertex' : shaderType === 1 ? 'pixel' : null;
      if (!kind) throw new XenosGpuError(`${opcodeName}: unknown shader type ${shaderType}`);
      const shader = { kind, code: Uint32Array.from(codeDwords), dwords: codeDwords.length };
      if (kind === 'vertex') this.vertexShader = shader; else this.pixelShader = shader;
      if (this.backend && typeof this.backend.setShaders === 'function') {
        this.backend.setShaders({ vertex: this.vertexShader, pixel: this.pixelShader });
      } else {
        this.log('info', `${opcodeName}: ${kind} shader loaded (${shader.dwords} dwords, no backend sink)`);
      }
    }

    _draw(u32s, start, end, opcodeName, vizQueryCondition, indexed) {
      const initiatorWord = u32s[start];
      const di = pm4().parseVgtDrawInitiator(initiatorWord);
      this._writeRegister(regFile().XE_GPU_REG_VGT_DRAW_INITIATOR, initiatorWord);

      let indexGuestBase = 0;
      let indexFormat = di.indexSize; // 0 = int16, 1 = int32
      let indexEndianness = 0;
      let indexCount = di.numIndices;
      let indexLengthWords = 0;

      const source = di.sourceSelect;
      let consumed = start + 1;
      if (source === pm4().SOURCE_SELECT.kDMA) {
        if (consumed + 2 > end) throw new XenosGpuError(`${opcodeName}: indexed draw packet too small for VGT_DMA_BASE/SIZE`);
        // xenia's primitive processor: "The base should already be aligned
        // according to the R6xx documentation, but aligning here too for
        // safety" — the base is aligned down to the index element size and
        // kept as the full 32-bit guest physical address.
        indexGuestBase = (u32s[consumed] & ~((di.indexSize === 1 ? 4 : 2) - 1)) >>> 0;
        const dmaSizeWord = u32s[consumed + 1];
        const dmaSize = pm4().parseVgtDmaSize(dmaSizeWord);
        indexEndianness = dmaSize.swapMode;
        indexLengthWords = dmaSize.numWords;
        // xenia's primitive processor clamps the drawn vertex count to the
        // index buffer's declared size: "N vertices attempted to be drawn with
        // an index buffer only containing M. Should be fetching zero indices
        // instead of overflowing ones, but this is a rare situation, so not
        // handled yet." (xenia primitive_processor.cc) — the draw fetches
        // min(num_indices, num_words) indices.
        if (di.numIndices > indexLengthWords) {
          this.log('warn', `${opcodeName}: ${di.numIndices} indices requested but the index buffer only declares ${indexLengthWords} — the draw is clamped to the declared range (xenia clamps it the same way)`);
          indexCount = indexLengthWords;
        }
        this._writeRegister(regFile().XE_GPU_REG_VGT_DMA_BASE, indexGuestBase);
        this._writeRegister(regFile().XE_GPU_REG_VGT_DMA_SIZE, dmaSizeWord);
        consumed += 2;
      } else if (source === pm4().SOURCE_SELECT.kAutoIndex) {
        // no index DMA
      } else if (source === pm4().SOURCE_SELECT.kImmediate) {
        throw new XenosGpuError(`${opcodeName}: immediate vertex indices are not supported (xenia does not support them either)`);
      } else {
        throw new XenosGpuError(`${opcodeName}: invalid source_select ${source}`);
      }

      const majorModeExplicit = di.majorMode !== pm4().MAJOR_MODE.kImplicit || di.primType >= 0x10;
      // Vertex fetch constants live in the register file (SHADER_CONSTANT_FETCH_BASE .. +192 dwords).
      // Hand them to the backend so it can analyze the fetch shader and upload/bind vertex buffers —
      // the draw path is the only consumer and this is the cheapest correct place to snapshot them.
      const vertexFetchConstants = new Uint32Array(192);
      for (let i = 0; i < 192; i++) vertexFetchConstants[i] = this.registers.read(SHADER_CONSTANT_FETCH_BASE + i) >>> 0;
      // ALU constants (256 vec4f = SHADER_CONSTANT_ALU_BASE .. +1024 dwords) and bool constants
      // (8 dwords = SHADER_CONSTANT_BOOL_BASE .. +8) are what the translated vertex shader's
      // fc[]/bools uniform bindings read — snapshot them the same way (the backend content-
      // hashes and only re-uploads when a draw actually changed them).
      const aluConstants = new Uint32Array(1024);
      for (let i = 0; i < 1024; i++) aluConstants[i] = this.registers.read(SHADER_CONSTANT_ALU_BASE + i) >>> 0;
      const boolConstants = new Uint32Array(8);
      for (let i = 0; i < 8; i++) boolConstants[i] = this.registers.read(SHADER_CONSTANT_BOOL_BASE + i) >>> 0;
      // Render-target (RB_*) registers — raw values; the backend decodes them
      // into the WebGPU pipeline/render-pass state (blend, write mask, depth/
      // stencil) and traps loudly on anything it cannot represent. xenia
      // register indices (see xenia gpu/registers.h).
      const RB_SURFACE_INFO = 0x2000;
      const RB_COLOR_INFO = 0x2001, RB_DEPTH_INFO = 0x2002;
      const RB_COLOR1_INFO = 0x2003, RB_COLOR2_INFO = 0x2004, RB_COLOR3_INFO = 0x2005;
      const RB_COLOR_MASK = 0x2104, RB_ALPHA_REF = 0x210e;
      const RB_MODECONTROL = 0x2208;
      const RB_BLEND_RED = 0x2105, RB_BLEND_GREEN = 0x2106, RB_BLEND_BLUE = 0x2107, RB_BLEND_ALPHA = 0x2108;
      const RB_STENCILREFMASK_BF = 0x210c, RB_STENCILREFMASK = 0x210d;
      const RB_DEPTHCONTROL = 0x2200, RB_BLENDCONTROL0 = 0x2201, RB_COLORCONTROL = 0x2202;
      const RB_BLENDCONTROL1 = 0x2209, RB_BLENDCONTROL2 = 0x220a, RB_BLENDCONTROL3 = 0x220b;
      const PA_SC_VIZ_QUERY = 0x2293;
      const VGT_MULTI_PRIM_IB_RESET_INDX = 0x2103;
      const PA_SU_SC_MODE_CNTL = 0x2205, PA_SU_VTX_CNTL = 0x2302;
      const PA_CL_CLIP_CNTL = 0x2204, PA_CL_VTE_CNTL = 0x2206;
      const PA_CL_VPORT_XSCALE = 0x210F, PA_CL_VPORT_XOFFSET = 0x2110, PA_CL_VPORT_YSCALE = 0x2111, PA_CL_VPORT_YOFFSET = 0x2112;
      const PA_CL_VPORT_ZSCALE = 0x2113, PA_CL_VPORT_ZOFFSET = 0x2114;
      const PA_SU_POLY_OFFSET_FRONT_SCALE = 0x2380, PA_SU_POLY_OFFSET_FRONT_OFFSET = 0x2381;
      const PA_SU_POLY_OFFSET_BACK_SCALE = 0x2382, PA_SU_POLY_OFFSET_BACK_OFFSET = 0x2383;
      const PA_SC_WINDOW_SCISSOR_TL = 0x2081, PA_SC_WINDOW_SCISSOR_BR = 0x2082, PA_SC_WINDOW_OFFSET = 0x2080;
      const PA_SC_SCREEN_SCISSOR_TL = 0x200e, PA_SC_SCREEN_SCISSOR_BR = 0x200f;
      const renderTargets = {
        surfaceInfo: this.registers.read(RB_SURFACE_INFO) >>> 0,
        rbModeControl: this.registers.read(RB_MODECONTROL) >>> 0,
        colorInfo0: this.registers.read(RB_COLOR_INFO) >>> 0,
        depthInfo: this.registers.read(RB_DEPTH_INFO) >>> 0,
        colorInfo1: this.registers.read(RB_COLOR1_INFO) >>> 0,
        colorInfo2: this.registers.read(RB_COLOR2_INFO) >>> 0,
        colorInfo3: this.registers.read(RB_COLOR3_INFO) >>> 0,
        colorMask: this.registers.read(RB_COLOR_MASK) >>> 0,
        blendRed: this.registers.read(RB_BLEND_RED) >>> 0,
        blendGreen: this.registers.read(RB_BLEND_GREEN) >>> 0,
        blendBlue: this.registers.read(RB_BLEND_BLUE) >>> 0,
        blendAlpha: this.registers.read(RB_BLEND_ALPHA) >>> 0,
        stencilRefMaskBf: this.registers.read(RB_STENCILREFMASK_BF) >>> 0,
        stencilRefMask: this.registers.read(RB_STENCILREFMASK) >>> 0,
        depthControl: this.registers.read(RB_DEPTHCONTROL) >>> 0,
        blendControl0: this.registers.read(RB_BLENDCONTROL0) >>> 0,
        colorControl: this.registers.read(RB_COLORCONTROL) >>> 0,
        alphaRef: this.registers.read(RB_ALPHA_REF) >>> 0,
        blendControl1: this.registers.read(RB_BLENDCONTROL1) >>> 0,
        blendControl2: this.registers.read(RB_BLENDCONTROL2) >>> 0,
        blendControl3: this.registers.read(RB_BLENDCONTROL3) >>> 0,
        paSuScModeCntl: this.registers.read(PA_SU_SC_MODE_CNTL) >>> 0,
        paSuVtxCntl: this.registers.read(PA_SU_VTX_CNTL) >>> 0,
        paClClipCntl: this.registers.read(PA_CL_CLIP_CNTL) >>> 0,
        paClVteCntl: this.registers.read(PA_CL_VTE_CNTL) >>> 0,
        paClVportXscale: this.registers.read(PA_CL_VPORT_XSCALE) >>> 0,
        paClVportXoffset: this.registers.read(PA_CL_VPORT_XOFFSET) >>> 0,
        paClVportYscale: this.registers.read(PA_CL_VPORT_YSCALE) >>> 0,
        paClVportYoffset: this.registers.read(PA_CL_VPORT_YOFFSET) >>> 0,
        paClVportZscale: this.registers.read(PA_CL_VPORT_ZSCALE) >>> 0,
        paClVportZoffset: this.registers.read(PA_CL_VPORT_ZOFFSET) >>> 0,
        paSuPolyOffsetFrontScale: this.registers.read(PA_SU_POLY_OFFSET_FRONT_SCALE) >>> 0,
        paSuPolyOffsetFrontOffset: this.registers.read(PA_SU_POLY_OFFSET_FRONT_OFFSET) >>> 0,
        paSuPolyOffsetBackScale: this.registers.read(PA_SU_POLY_OFFSET_BACK_SCALE) >>> 0,
        paSuPolyOffsetBackOffset: this.registers.read(PA_SU_POLY_OFFSET_BACK_OFFSET) >>> 0,
      };
      // xenia's IssueDraw gate (D3D12CommandProcessor::IssueDraw):
      //   RB_MODECONTROL.edram_mode (bits 0:2) — xenos::ModeControl:
      //     kIgnore (0)    → the draw is ignored entirely (no fragments),
      //     kCopy (6)     → the draw is a render-target copy,
      //     kDepth (5)    → depth-only: no pixel shader, no color output,
      //     kColorDepth (4)→ an ordinary color+depth draw.
      //   RB_SURFACE_INFO.surface_pitch == 0 → "Doesn't actually draw".
      const edramMode = (this.registers.read(RB_MODECONTROL) >>> 0) & 0x7;
      if (edramMode === 0 /* kIgnore */) {
        this.log('debug', `${opcodeName}: draw ignored — RB_MODECONTROL.edram_mode = kIgnore (xenia returns before rasterizing)`);
        return end;
      }
      if (edramMode === 6 /* kCopy */) {
        throw new XenosGpuError(
          `${opcodeName}: the guest issued a render-target copy (RB_MODECONTROL.edram_mode = kCopy) — the copy of the ` +
          'render target into guest memory (RB_COPY_*) is not implemented in this port (the EDRAM render-target cache, ' +
          'step 7). Refusing to draw instead of silently dropping the copy');
      }
      if (((this.registers.read(RB_SURFACE_INFO) >>> 0) & 0x3fff) === 0) {
        this.log('debug', `${opcodeName}: draw skipped — RB_SURFACE_INFO.surface_pitch is 0 (xenia: "doesn't actually draw")`);
        return end;
      }
      // xenia: if PA_SC_VIZ_QUERY has viz_query_ena (bit 0) AND kill_pix_post_hi_z (bit 7),
      // the draw is an occlusion-query hint and must NOT touch the render target (r6xx/r7xx
      // VIZ_QUERY). IssueDraw is not called then; other bit combinations draw normally.
      const vizQuery = this.registers.read(PA_SC_VIZ_QUERY) >>> 0;
      if ((vizQuery & 0x81) === 0x81) {
        this.log('debug', `${opcodeName}: draw skipped — PA_SC_VIZ_QUERY enables the query test with kill_pix_post_hi_z (xenia drops such draws; occlusion-vector hint, not an RT write)`);
        return end;
      }
      const scissor = {
        windowScissorTl: this.registers.read(PA_SC_WINDOW_SCISSOR_TL) >>> 0,
        windowScissorBr: this.registers.read(PA_SC_WINDOW_SCISSOR_BR) >>> 0,
        windowOffset: this.registers.read(PA_SC_WINDOW_OFFSET) >>> 0,
        screenScissorTl: this.registers.read(PA_SC_SCREEN_SCISSOR_TL) >>> 0,
        screenScissorBr: this.registers.read(PA_SC_SCREEN_SCISSOR_BR) >>> 0,
      };
      const draw = backendOrTrap(this.backend, 'draw', opcodeName);
      draw({
        primType: di.primType,
        primTypeName: di.primTypeName,
        numIndices: indexCount,
        isIndexed: source === pm4().SOURCE_SELECT.kDMA,
        indexGuestBase,
        indexFormat,
        indexEndianness,
        indexLengthWords,
        // xenia's primitive reset index (multi-primitive index buffers):
        // PA_SU_SC_MODE_CNTL.multi_prim_ib_ena enables it, and the reset value
        // itself comes from VGT_MULTI_PRIM_IB_RESET_INDX.
        multiPrimIbEnable: ((this.registers.read(PA_SU_SC_MODE_CNTL) >>> 21) & 1) !== 0,
        multiPrimResetIndex: this.registers.read(VGT_MULTI_PRIM_IB_RESET_INDX) >>> 0,
        vizQueryCondition,
        majorModeExplicit,
        vertexFetchConstants,
        aluConstants,
        boolConstants,
        renderTargets,
        scissor,
        // xenia's kDepth mode: no pixel shader, no color output — depth only.
        depthOnly: edramMode === 5,
      });
      this.stats.draws++;
      return end;
    }
  }

  const api = { XenosCommandProcessor, XenosGpuError, CONSTANT_TYPE, SHADER_CONSTANT_ALU_BASE, SHADER_CONSTANT_FETCH_BASE, SHADER_CONSTANT_BOOL_BASE, SHADER_CONSTANT_LOOP_BASE, REGISTER_CONSTANT_BASE };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonGpuCommandProcessor = api;
})(typeof self !== 'undefined' ? self : globalThis);
