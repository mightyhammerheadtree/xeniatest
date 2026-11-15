/**
 * core.js — loads the Xenia Canary WebAssembly core and exposes a typed
 * wrapper around its C ABI. The core has zero dependencies: one linear
 * memory, no imports that actually run (the WASI stubs are never executed),
 * no growth beyond its compiled-in maximum, so it can never OOM the tab.
 */

const WASI_STUBS = {
  wasi_snapshot_preview1: {
    args_get: () => 0,
    args_sizes_get: () => 0,
    proc_exit: (code) => {
      throw new Error(`guest called proc_exit(${code}) — the exit door was not used`);
    },
  },
};

export const CoreStatus = Object.freeze({
  Ok: 0,
  Syscall: 1,        // sc executed, or the MMIO exit door was written
  Unimplemented: 2,  // trap hit on an unimplemented instruction (opt-in)
  MemoryFault: 3,    // guest accessed out-of-bounds memory
});

export const CoreStatusText = {
  0: 'running',
  1: 'halted (exit door / sc)',
  2: 'trapped on unimplemented instruction',
  3: 'memory fault',
};

/**
 * Fetch and instantiate the core module. The fetch streams (no full-body
 * buffering in JS) and the compiled module is cached by the browser.
 */
export async function loadCore(url = 'core/xeniatest-core.wasm') {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch core: HTTP ${response.status}`);
  const backup = response.clone();
  try {
    const { instance } = await WebAssembly.instantiateStreaming(response, WASI_STUBS);
    return instance.exports;
  } catch {
    // Fallback for servers that don't set application/wasm.
    const bytes = await backup.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, WASI_STUBS);
    return instance.exports;
  }
}

const hex = (v, digits = 16) => '0x' + (BigInt(v) & ((1n << 64n) - 1n)).toString(16).toUpperCase().padStart(digits, '0');
const hex32 = (v) => '0x' + (Number(v) >>> 0).toString(16).toUpperCase().padStart(8, '0');

export class Core {
  constructor(exports) {
    if (!exports) throw new Error('core exports missing');
    this.x = exports;
    this.mem = new DataView(exports.memory.buffer);
    this.ramBase = exports.xeniatest_guest_ram_pointer();
    this.ramSize = exports.xeniatest_guest_ram_size();
  }

  static async create(url) {
    return new Core(await loadCore(url));
  }

  reset(entry) {
    this.x.xeniatest_reset(BigInt(entry));
  }

  setTrapOnUnimplemented(enabled) {
    this.x.xeniatest_set_trap_on_unimplemented(enabled ? 1 : 0);
  }

  /** Load bytes at a guest-physical address. */
  writeBytes(address, bytes) {
    if (address + bytes.length > this.ramSize) {
      throw new Error(`payload exceeds guest RAM (${bytes.length} B at ${hex32(address)})`);
    }
    new Uint8Array(this.x.memory.buffer, this.ramBase + address, bytes.length).set(bytes);
  }

  /** Write a NUL-terminated string into guest RAM. Used by the loader. */
  writeString(address, text) {
    const bytes = new TextEncoder().encode(text + '\0');
    this.writeBytes(address, bytes);
    return bytes.length;
  }

  run(maxInstructions) {
    return this.x.xeniatest_run(maxInstructions >>> 0);
  }

  step() {
    return this.x.xeniatest_step();
  }

  status() { return this.x.xeniatest_status(); }
  scLevel() { return this.x.xeniatest_sc_level(); }
  pc() { return this.x.xeniatest_pc(); }
  instructionsExecuted() { return this.x.xeniatest_instructions_executed(); }
  unimplementedCount() { return this.x.xeniatest_unimplemented_count(); }
  lastUnimplemented() { return this.x.xeniatest_last_unimplemented() >>> 0; }
  lastFaultAddress() { return this.x.xeniatest_last_fault_address(); }

  gpr(index) { return this.x.xeniatest_gpr(index); }
  gprLow32(index) { return this.x.xeniatest_gpr(index) & 0xffffffffn; }
  cr() { return this.x.xeniatest_cr(); }
  lr() { return this.x.xeniatest_lr(); }
  ctr() { return this.x.xeniatest_ctr(); }
  xer() { return this.x.xeniatest_xer(); }

  /** Drain the guest console ring buffer and return it as text. */
  consoleText() {
    const len = this.x.xeniatest_console_length();
    if (len === 0) return '';
    const start = this.x.xeniatest_console_data();
    const bytes = new Uint8Array(this.x.memory.buffer, start, len);
    const text = new TextDecoder().decode(bytes);
    this.x.xeniatest_console_consume(len);
    return text;
  }

  /** One disassembly line for a big-endian instruction word at `address`. */
  disassembleLine(address, code) {
    this.x.xeniatest_disasm(address >>> 0, code >>> 0);
    return readCString(this.x, this.x.xeniatest_disasm_line());
  }

  opcodeName(code) {
    const ptr = this.x.xeniatest_opcode_name(code >>> 0);
    return ptr ? readCString(this.x, ptr) : '';
  }

  opcodeGroup(code) {
    const ptr = this.x.xeniatest_opcode_group(code >>> 0);
    return ptr ? readCString(this.x, ptr) : 'unknown';
  }

  formatPc() { return hex(this.pc()); }
}

export const formatHex = hex;
export const formatHex32 = hex32;

function readCString(exports, ptr) {
  const mem = new Uint8Array(exports.memory.buffer);
  let end = ptr;
  while (mem[end] !== 0 && end < mem.length) end++;
  return new TextDecoder().decode(mem.subarray(ptr, end));
}
