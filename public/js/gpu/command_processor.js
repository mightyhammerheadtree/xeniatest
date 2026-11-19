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
   *   draw({ primType, numIndices, isIndexed, indexGuestBase, indexCount, indexFormat /* 0=int16 1=int32 *\/, indexEndianness /* swap_mode 0..3 *\/, vizQueryCondition, majorModeExplicit, vertexFetchConstants /* 192 dwords *\/, aluConstants /* 1024 dwords *\/, boolConstants /* 8 dwords *\/ })
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
      /** Shaders loaded via IM_LOAD*; opaque descriptors owned by the backend. */
      this.vertexShader = null;
      this.pixelShader = null;
      /** Viz query bookkeeping: number of open surveys (depth may exceed 1 across nested flushes). */
      this.vizQueryOpen = false;
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
          const anyPass = (this.binSelect & this.binMask) !== 0n;
          if (!anyPass && parsed.opcode !== pm4().PM4_OPCODE.PM4_XE_SWAP) {
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
        case op.PM4_DRAW_INDX_BIN: {
          const vizQueryCondition = u32s[start];
          return this._draw(u32s, start + 1, end, parsed.opcodeName, vizQueryCondition, true);
        }
        case op.PM4_DRAW_INDX_2_BIN:
          return this._draw(u32s, start, end, parsed.opcodeName, 0, false);

        // --- events / synchronization ----------------------------------------
        case op.PM4_EVENT_WRITE:
        case op.PM4_EVENT_WRITE_SHD: {
          this._eventWriteToMemory(u32s, start, end, parsed.opcodeName);
          return end;
        }
        case op.PM4_EVENT_WRITE_CFL:
        case op.PM4_EVENT_WRITE_EXT:
        case op.PM4_EVENT_WRITE_ZPD: {
          // cache-flush / screen-extent / z-pass events: initiator recorded; memory writeback formats differ per event — forwarded to the backend, never silently dropped
          const initiator = u32s[start];
          if (this.backend && typeof this.backend.eventWrite === 'function') this.backend.eventWrite({ eventType: initiator & 0x3f, address: 0, data: null, opcode: parsed.opcodeName });
          else this.log('info', `${parsed.opcodeName} initiator 0x${initiator.toString(16)} (no backend sink)`);
          this._writeRegister(regFile().XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3f);
          return end;
        }
        case op.PM4_VIZ_QUERY: {
          const value = u32s[start];
          const begin = (value & 0x1) === 0; // 0 = begin query, 1 = end query (r6xx VIZ_QUERY semantics)
          this.vizQueryOpen = begin;
          if (this.backend && typeof this.backend.vizQuery === 'function') this.backend.vizQuery(begin ? 'start' : 'end');
          return end;
        }
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
          if (this.backend && typeof this.backend.swap === 'function') this.backend.swap();
          else this.log('info', 'XE_SWAP (present) — no backend sink');
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
          // conditional execution based on a register/memory compare — not yet modeled; trap loudly
          throw new XenosGpuError('PM4_COND_EXEC: conditional execution is not implemented (xenia models it; port the compare logic before enabling)');
        }
        case op.PM4_COND_WRITE: {
          throw new XenosGpuError('PM4_COND_WRITE: conditional write is not implemented');
        }
        case op.PM4_WAIT_REG_MEM:
        case op.PM4_WAIT_REG_EQ:
        case op.PM4_WAIT_REG_GTE:
        case op.PM4_WAIT_FOR_IDLE:
        case op.PM4_WAIT_UNTIL_READ:
        case op.PM4_WAIT_IB_PFD_COMPLETE:
          // Genuinely synchronous in our model — the ring is consumed inline, so waits succeed trivially.
          // WAIT_REG_MEM with a memory location polls a value; xenia blocks until it matches.
          if (opcode === op.PM4_WAIT_REG_MEM) {
            throw new XenosGpuError('PM4_WAIT_REG_MEM: polling memory/register is not implemented; waits complete immediately only for WAIT_FOR_IDLE/WAIT_IB_PFD_COMPLETE');
          }
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
          // bin base offset for tiled rendering — records into registers like xenia would via RB path; no sink yet
          this.log('debug', `SET_BIN_BASE_OFFSET 0x${u32s[start].toString(16)} (no tiled-rendering sink yet)`);
          return end;

        default:
          this.stats.unimplementedTraps++;
          throw new XenosGpuError(`${parsed.opcodeName}: unimplemented GPU opcode 0x${opcode.toString(16).toUpperCase().padStart(2, '0')} (count ${count}) — see docs/XENOS_GPU.md`);
      }
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
      this.counter = (this.counter + 1) >>> 0;
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
        indexGuestBase = u32s[consumed] & 0x1fffffff; // xenia CpuToGpu: GPU bus is 29 bits
        const dmaSizeWord = u32s[consumed + 1];
        const dmaSize = pm4().parseVgtDmaSize(dmaSizeWord);
        indexEndianness = dmaSize.swapMode;
        indexLengthWords = dmaSize.numWords;
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
      const draw = backendOrTrap(this.backend, 'draw', opcodeName);
      draw({
        primType: di.primType,
        primTypeName: di.primTypeName,
        numIndices: di.numIndices,
        isIndexed: source === pm4().SOURCE_SELECT.kDMA,
        indexGuestBase,
        indexFormat,
        indexEndianness,
        indexLengthWords,
        vizQueryCondition,
        majorModeExplicit,
        vertexFetchConstants,
        aluConstants,
        boolConstants,
      });
      this.stats.draws++;
      return end;
    }
  }

  const api = { XenosCommandProcessor, XenosGpuError, CONSTANT_TYPE, SHADER_CONSTANT_ALU_BASE, SHADER_CONSTANT_FETCH_BASE, SHADER_CONSTANT_BOOL_BASE, SHADER_CONSTANT_LOOP_BASE, REGISTER_CONSTANT_BASE };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.XenonGpuCommandProcessor = api;
})(typeof self !== 'undefined' ? self : globalThis);
