/**
 * Xenia Canary - Web core: Xenon PPC interpreter + decoder, C ABI.
 *
 * Xenia: Xbox 360 Emulator Research Project
 * Original xenia-canary Copyright 2013-2015 Ben Vanik and contributors.
 * Ported/reimplemented for the web by the xeniatest project.
 * Released under the BSD license - see LICENSE in the repository root.
 *
 * This is the public ABI between the JavaScript front-end and the
 * freestanding WASM core. Nothing here allocates: all state lives in the
 * guest RAM array and in fixed static storage.
 *
 * Addressing model
 * ----------------
 * Guest addresses are flat, 0-based offsets into guest RAM
 * (XENIATEST_GUEST_RAM_SIZE bytes). The PPC front-end maps the full guest
 * address space linearly onto this window, matching how xenia maps the
 * Xenon's 512MB physical space. Virtual->physical translation and the
 * memory controller are NOT modeled yet (see docs/PORTING_ROADMAP.md).
 *
 * MMIO
 * ----
 * [0x00F00000, 0x00F01000) is reserved as the "xeniatest debug MMIO"
 * window. Byte stores there are appended to the console ring buffer;
 * word stores print their big-endian byte order. Reads return 0.
 */

#ifndef XENIATEST_CORE_H_
#define XENIATEST_CORE_H_

#include <cstdint>

#define XENIATEST_MMIO_BASE 0x00F00000u
#define XENIATEST_MMIO_SIZE 0x00001000u
#define XENIATEST_MMIO_CONSOLE 0x00F00000u
#define XENIATEST_MMIO_EXIT 0x00F00FFCu

// Run/status codes returned by xeniatest_run() and xeniatest_step().
enum XeniatestStatus {
  kStatusOk = 0,         // hit the instruction budget
  kStatusSyscall = 1,    // guest executed `sc` (level via xeniatest_sc_level())
  kStatusUnimplemented = 2,  // unimplemented instruction (trap-on-unimplemented)
  kStatusMemoryFault = 3,    // out-of-range guest access
  kStatusTrap = 4,           // guest executed tw/twi and the condition fired
  kStatusBadAddress = 5,     // fetch from non-executable/overflowing address
};

#ifdef __cplusplus
extern "C" {
#endif

// --- Lifecycle ---------------------------------------------------------------
void xeniatest_reset(uint64_t entry_pc);

// --- Execution -----------------------------------------------------------------
// Runs up to max_instructions; returns an XeniatestStatus code.
uint32_t xeniatest_run(uint32_t max_instructions);
// Executes exactly one instruction; returns status (0 = ok, budget irrelevant).
uint32_t xeniatest_step(void);

uint32_t xeniatest_status(void);        // last status code from run/step
uint32_t xeniatest_sc_level(void);      // LEV of the `sc` that halted the core
uint64_t xeniatest_pc(void);
uint64_t xeniatest_instructions_executed(void);
uint64_t xeniatest_unimplemented_count(void);
uint32_t xeniatest_last_unimplemented(void);  // raw 32-bit encoding
uint64_t xeniatest_last_fault_address(void);

void xeniatest_set_trap_on_unimplemented(uint32_t enabled);
uint32_t xeniatest_trap_on_unimplemented(void);

// --- Registers (for the UI / tests) --------------------------------------------
uint64_t xeniatest_gpr(uint32_t index);
void xeniatest_set_gpr(uint32_t index, uint64_t value);
uint64_t xeniatest_fpr_bits(uint32_t index);
void xeniatest_set_fpr_bits(uint32_t index, uint64_t value);
uint32_t xeniatest_cr(void);
void xeniatest_set_cr(uint32_t value);
uint64_t xeniatest_lr(void);
uint64_t xeniatest_ctr(void);
uint64_t xeniatest_xer(void);

// --- Guest RAM -----------------------------------------------------------------
uint32_t xeniatest_guest_ram_size(void);  // bytes
// Start address of guest RAM inside wasm linear memory. Guest address G is
// stored at (xeniatest_guest_ram_pointer() + G) in the exported memory.
uint32_t xeniatest_guest_ram_pointer(void);

// --- Console ring buffer ---------------------------------------------------------
const uint8_t* xeniatest_console_data(void);  // start of pending bytes
uint32_t xeniatest_console_length(void);      // pending bytes
void xeniatest_console_consume(uint32_t n);   // drop first n pending bytes

// --- Disassembler (xenia-canary derived) -------------------------------------------
// Writes a disassembly line into an internal static buffer (no allocation).
// Returns the line length. Read the bytes via xeniatest_disasm_line().
uint32_t xeniatest_disasm(uint32_t address, uint32_t code);
const char* xeniatest_disasm_line(void);
// Points at a static NUL-terminated name for the instruction, or "" if the
// encoding is not a known Xenon PPC instruction.
const char* xeniatest_opcode_name(uint32_t code);
// Name of the primary opcode group ("integer", "branch", "memory",
// "floating-point", "vmx", "control", "unknown").
const char* xeniatest_opcode_group(uint32_t code);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XENIATEST_CORE_H_
