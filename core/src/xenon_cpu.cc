/**
 * Xenia Canary - Web core: Xenon PPC64 interpreter.
 *
 * Xenia: Xbox 360 Emulator Research Project
 * Original xenia-canary Copyright 2013-2015 Ben Vanik and contributors.
 * New implementation for the web target by the xeniatest project (xenia has
 * no interpreter; it recompiles PPC -> host code). Instruction semantics
 * follow the PowerISA and mirror xenia-canary's emitters where relevant.
 * Released under the BSD license - see LICENSE in the repository root.
 *
 * Properties:
 *   - No heap allocation after startup: all state lives in static storage.
 *   - Guest RAM is a fixed-size static array (default 64 MiB, tunable at
 *     build time with -DXENIATEST_GUEST_RAM_MB). Big-endian, bounds-checked.
 *   - The instruction decoder and disassembler come from xenia-canary
 *     (see core/src/xenia/), so every instruction that decodes in xenia
 *     decodes here too.
 *
 * Conventions / caveats
 * ---------------------
 *   - 32-bit integer results are sign-extended into the full 64-bit GPR,
 *     matching PowerISA pseudocode for 64-bit execution of 32-bit ops.
 *   - D-form/DS-form load/store with RA=0 uses a literal zero base.
 *     Compare instructions always use register r0 as-is.
 *   - FP Rc variants record FPSCR[0:3] (the CC bits) into CR1.
 *   - sc halts the core (kernel HLE is not implemented yet - see
 *     docs/PORTING_ROADMAP.md). Writes to the last byte of the MMIO
 *     window [0x00F00FFC] also halt, which the built-in test programs use.
 */

#include "xenon_cpu.h"

#include <cstdint>
#include <cstddef>
#include <cstring>

#include "compat.h"
#include "xenia/ppc_opcode.h"
#include "xenia/ppc_opcode_info.h"
#include "xenia/ppc_decode_data.h"
#include "xenia/ppc_opcode_disasm.h"
#include "xenia/ppc_opcode_lookup_gen.cc"

namespace {

using xe::cpu::ppc::PPCDecodeData;
using xe::cpu::ppc::PPCOpcode;
using xe::cpu::ppc::XEMASK;

#ifndef XENIATEST_GUEST_RAM_MB
#define XENIATEST_GUEST_RAM_MB 64
#endif

constexpr uint64_t kGuestRamSize = (uint64_t)XENIATEST_GUEST_RAM_MB * 1024ull * 1024ull;

// Static storage: no malloc, no heap, ever.
uint8_t g_guest_ram[kGuestRamSize];

struct CpuState {
  uint64_t gpr[32];
  uint64_t fpr_bits[32];
  uint64_t lr;
  uint64_t ctr;
  uint32_t cr;
  uint64_t xer;   // uses architected 32-bit positions (SO=0x80000000, ...)
  uint64_t msr;
  uint64_t fpscr;
  uint64_t pc;
  uint64_t instructions_executed;
  uint64_t unimplemented_count;
  uint64_t last_unimplemented;
  uint64_t last_fault_address;
  uint32_t sc_level;
  uint32_t status;
  uint64_t reservation;  // lwarx/ldarx reservation; kNoReservation when invalid
  uint64_t tbr;          // time base (counts instructions)
  uint32_t trap_on_unimplemented;
};

constexpr uint64_t kNoReservation = ~0ull;
constexpr uint64_t kXerSO = 0x80000000ull;
constexpr uint64_t kXerOV = 0x40000000ull;
constexpr uint64_t kXerCA = 0x20000000ull;

CpuState s_cpu;

// --- Console ring buffer (fixed) ---------------------------------------------
constexpr size_t kConsoleCapacity = 4096;
char g_console[kConsoleCapacity];
size_t s_console_len = 0;

void console_write_bytes(const uint8_t* bytes, size_t len) {
  for (size_t i = 0; i < len; i++) {
    if (s_console_len == kConsoleCapacity) {
      __builtin_memmove(g_console, g_console + 1, kConsoleCapacity - 1);
      s_console_len = kConsoleCapacity - 1;
    }
    g_console[s_console_len++] = (char)bytes[i];
  }
}

// --- Memory access (bounds-checked, big-endian) --------------------------------
inline bool ram_in_bounds(uint64_t addr, uint64_t size) {
  return addr <= kGuestRamSize && size <= kGuestRamSize - addr;
}

bool mem_fault(uint64_t addr) {
  s_cpu.last_fault_address = addr;
  s_cpu.status = kStatusMemoryFault;
  return false;
}

inline uint8_t load_u8(uint64_t addr) { return g_guest_ram[addr]; }
inline uint16_t load_u16(uint64_t addr) {
  return (uint16_t)((uint16_t(g_guest_ram[addr]) << 8) | g_guest_ram[addr + 1]);
}
inline uint32_t load_u32(uint64_t addr) {
  return (uint32_t(g_guest_ram[addr]) << 24) | (uint32_t(g_guest_ram[addr + 1]) << 16) |
         (uint32_t(g_guest_ram[addr + 2]) << 8) | uint32_t(g_guest_ram[addr + 3]);
}
inline uint64_t load_u64(uint64_t addr) {
  return (uint64_t(load_u32(addr)) << 32) | load_u32(addr + 4);
}
inline void store_u8(uint64_t addr, uint8_t v) { g_guest_ram[addr] = v; }
inline void store_u16(uint64_t addr, uint16_t v) {
  g_guest_ram[addr] = (uint8_t)(v >> 8);
  g_guest_ram[addr + 1] = (uint8_t)v;
}
inline void store_u32(uint64_t addr, uint32_t v) {
  g_guest_ram[addr] = (uint8_t)(v >> 24);
  g_guest_ram[addr + 1] = (uint8_t)(v >> 16);
  g_guest_ram[addr + 2] = (uint8_t)(v >> 8);
  g_guest_ram[addr + 3] = (uint8_t)v;
}
inline void store_u64(uint64_t addr, uint64_t v) {
  store_u32(addr, (uint32_t)(v >> 32));
  store_u32(addr + 4, (uint32_t)v);
}

// Store result: 0 = ok, 1 = fault, 2 = core halt requested (MMIO exit door).
int mmio_write(uint64_t addr, uint64_t value, uint32_t size_bytes) {
  if (addr >= XENIATEST_MMIO_BASE && addr < XENIATEST_MMIO_BASE + XENIATEST_MMIO_SIZE) {
    uint8_t buf[8];
    for (uint32_t i = 0; i < size_bytes; i++) {
      buf[i] = (uint8_t)(value >> ((size_bytes - 1 - i) * 8));
    }
    // Bytes at/above the exit door are control, not console text: don't echo them.
    uint32_t echo_len = size_bytes;
    if (addr <= XENIATEST_MMIO_EXIT && XENIATEST_MMIO_EXIT < addr + size_bytes) {
      echo_len = (uint32_t)(XENIATEST_MMIO_EXIT - addr);
    }
    if (echo_len > 0) {
      console_write_bytes(buf, echo_len);
    }
    if (addr <= XENIATEST_MMIO_EXIT && XENIATEST_MMIO_EXIT < addr + size_bytes) {
      s_cpu.status = kStatusSyscall;
      s_cpu.sc_level = 0xFFFFFFFFu;
      return 2;
    }
    return 0;
  }
  return -1;  // not MMIO
}

// Returns 0 = ok, 1 = fault, 2 = halt requested.
int store_mem(uint64_t addr, uint64_t value, uint32_t size_bytes) {
  int mmio = mmio_write(addr, value, size_bytes);
  if (mmio >= 0) return mmio;
  if (!ram_in_bounds(addr, size_bytes)) return mem_fault(addr) ? 1 : 1;
  switch (size_bytes) {
    case 1: store_u8(addr, (uint8_t)value); return 0;
    case 2: store_u16(addr, (uint16_t)value); return 0;
    case 4: store_u32(addr, (uint32_t)value); return 0;
    case 8: store_u64(addr, value); return 0;
    default: return mem_fault(addr) ? 1 : 1;
  }
}

bool load_mem(uint64_t addr, uint64_t* out, uint32_t size_bytes, bool sign_extend) {
  if (addr >= XENIATEST_MMIO_BASE && addr < XENIATEST_MMIO_BASE + XENIATEST_MMIO_SIZE) {
    *out = 0;
    return true;  // MMIO reads currently return zero.
  }
  if (!ram_in_bounds(addr, size_bytes)) return mem_fault(addr);
  uint64_t v = 0;
  switch (size_bytes) {
    case 1: v = load_u8(addr); break;
    case 2: v = load_u16(addr); break;
    case 4: v = load_u32(addr); break;
    case 8: v = load_u64(addr); break;
    default: return mem_fault(addr);
  }
  if (sign_extend) {
    switch (size_bytes) {
      case 1: v = (uint64_t)(int64_t)(int8_t)v; break;
      case 2: v = (uint64_t)(int64_t)(int16_t)v; break;
      case 4: v = (uint64_t)(int64_t)(int32_t)v; break;
      default: break;
    }
  }
  *out = v;
  return true;
}

// --- CR helpers -----------------------------------------------------------------
inline void cr_set_field(uint32_t field, bool lt, bool gt, bool eq, bool so) {
  uint32_t shift = 28 - 4 * (field & 7);
  uint32_t mask = 0xFu << shift;
  uint32_t value = ((lt ? 8u : 0u) | (gt ? 4u : 0u) | (eq ? 2u : 0u) | (so ? 1u : 0u)) << shift;
  s_cpu.cr = (s_cpu.cr & ~mask) | value;
}
inline void cr0_from_i64(int64_t v) { cr_set_field(0, v < 0, v > 0, v == 0, false); }
inline void cr0_from_i32(int32_t v) { cr_set_field(0, v < 0, v > 0, v == 0, false); }

inline bool cr_get_bit(uint32_t bi) { return (s_cpu.cr >> (31 - (bi & 31))) & 1; }
inline void cr_set_bit(uint32_t bt, bool v) {
  uint32_t shift = 31 - (bt & 31);
  s_cpu.cr = (s_cpu.cr & ~(1u << shift)) | ((v ? 1u : 0u) << shift);
}
inline uint32_t cr_field(uint32_t crf) { return (s_cpu.cr >> (28 - 4 * (crf & 7))) & 0xF; }
// FP Rc variants record FPSCR[0:3] into CR1.
inline void fp_record_cr1(bool rc) {
  if (!rc) return;
  uint32_t field = (uint32_t)((s_cpu.fpscr >> 28) & 0xF);
  s_cpu.cr = (s_cpu.cr & ~(0xFu << 24)) | (field << 24);
}

// --- XER helpers ------------------------------------------------------------------
inline uint64_t xer_ca() { return s_cpu.xer & kXerCA ? 1 : 0; }
inline void xer_set_ca(bool ca) {
  if (ca) s_cpu.xer |= kXerCA;
  else s_cpu.xer &= ~kXerCA;
}
inline void xer_record_overflow(bool ov) {
  if (ov) s_cpu.xer |= (kXerOV | kXerSO);
  else s_cpu.xer &= ~kXerOV;
}

// --- Register / operand helpers -----------------------------------------------------
inline uint64_t gpr(uint32_t i) { return s_cpu.gpr[i & 31]; }
// D-form / address offset rule: RA field of 0 means literal zero.
inline uint64_t ra_or_zero(uint32_t ra) { return ra == 0 ? 0 : s_cpu.gpr[ra]; }

constexpr uint64_t kAdd64(uint64_t a, uint64_t b, uint64_t cin, uint64_t* carry) {
  // Uses builtin 128-bit arithmetic; folds to native add + adc on wasm.
  unsigned __int128 sum = (unsigned __int128)a + b + cin;
  *carry = (uint64_t)(sum >> 64);
  return (uint64_t)sum;
}

inline uint64_t rotl64(uint64_t v, uint32_t n) {
  n &= 63;
  if (n == 0) return v;
  return (v << n) | (v >> (64 - n));
}

double bits_to_double(uint64_t bits) {
  double d;
  __builtin_memcpy(&d, &bits, sizeof(d));
  return d;
}
uint64_t double_to_bits(double d) {
  uint64_t bits;
  __builtin_memcpy(&bits, &d, sizeof(bits));
  return bits;
}

// --- Branch helpers -------------------------------------------------------------------
bool branch_condition(uint32_t bo, uint32_t bi) {
  bool ctr_ok;
  if ((bo & 0x04) == 0) {
    s_cpu.ctr -= 1;
    ctr_ok = (bo & 0x02) ? (s_cpu.ctr != 0) : (s_cpu.ctr == 0);
  } else {
    ctr_ok = true;
  }
  bool cr_ok;
  if (bo & 0x10) {
    cr_ok = true;
  } else {
    uint32_t expected = (bo & 0x08) ? 1u : 0u;
    cr_ok = (((s_cpu.cr >> (31 - (bi & 31))) & 1) == expected);
  }
  return ctr_ok && cr_ok;
}

// --- Interpreter ------------------------------------------------------------------------
static XeniatestStatus execute_one() {
  s_cpu.reservation = kNoReservation;
  const uint64_t pc = s_cpu.pc;
  if (pc + 4 > kGuestRamSize) {
    s_cpu.last_fault_address = pc;
    s_cpu.status = kStatusBadAddress;
    return kStatusBadAddress;
  }
  const uint32_t code = load_u32(pc);
  PPCDecodeData d;
  d.address = (uint32_t)pc;
  d.code = code;
  const PPCOpcode opcode = xe::cpu::ppc::LookupOpcode(code);

  const uint64_t next_pc = pc + 4;
  bool pc_set = false;
  auto set_pc = [&](uint64_t target) {
    s_cpu.pc = target;
    pc_set = true;
  };
  auto finish = [&]() -> XeniatestStatus {
    if (!pc_set) s_cpu.pc = next_pc;
    s_cpu.instructions_executed++;
    s_cpu.tbr++;
    s_cpu.status = kStatusOk;
    return kStatusOk;
  };
  auto unimplemented = [&]() -> XeniatestStatus {
    s_cpu.unimplemented_count++;
    s_cpu.last_unimplemented = code;
    if (s_cpu.trap_on_unimplemented) {
      s_cpu.status = kStatusUnimplemented;
      return kStatusUnimplemented;
    }
    s_cpu.pc = next_pc;
    s_cpu.instructions_executed++;
    s_cpu.tbr++;
    s_cpu.status = kStatusOk;
    return kStatusOk;
  };
  switch (opcode) {
    // ---------------- Integer arithmetic ----------------
    case PPCOpcode::addi: {
      uint64_t ra = ra_or_zero(d.D.RA());
      s_cpu.gpr[d.D.RT()] = (uint64_t)((int64_t)ra + (int64_t)d.D.SIMM());
      break;
    }
    case PPCOpcode::addis: {
      uint64_t ra = ra_or_zero(d.D.RA());
      s_cpu.gpr[d.D.RT()] = (uint64_t)((int64_t)ra + ((int64_t)d.D.SIMM() << 16));
      break;
    }
    case PPCOpcode::addic:
    case PPCOpcode::addicx: {
      uint64_t ra = ra_or_zero(d.D.RA());
      // 32-bit carry lives at bit 32, not bit 64.
      uint64_t sum = (uint64_t)(uint32_t)ra + (uint32_t)(int64_t)d.D.SIMM();
      int32_t r32 = (int32_t)(uint32_t)sum;
      s_cpu.gpr[d.D.RT()] = (uint64_t)(int64_t)r32;
      xer_set_ca(sum > 0xFFFFFFFFull);
      if (opcode == PPCOpcode::addicx) cr0_from_i32(r32);
      break;
    }
    case PPCOpcode::addx:
    case PPCOpcode::addcx:
    case PPCOpcode::addex: {
      uint64_t a = gpr(d.XO.RA());
      uint64_t b = gpr(d.XO.RB());
      uint64_t cin = (opcode == PPCOpcode::addex) ? xer_ca() : 0;
      uint64_t carry = 0;
      uint64_t r = kAdd64(a, b, cin, &carry);
      if (opcode != PPCOpcode::addx) xer_set_ca(carry != 0);
      s_cpu.gpr[d.XO.RT()] = r;
      if (d.XO.OE()) {
        bool ov = (~(a ^ b) & (a ^ r) & 0x8000000000000000ull) != 0;
        xer_record_overflow(ov);
      }
      if (d.XO.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::addmex: {
      uint64_t a = gpr(d.XO.RA());
      uint64_t cin = xer_ca();
      uint64_t carry = 0;
      uint64_t r = kAdd64(a, ~0ull, cin, &carry);
      xer_set_ca(carry != 0);
      s_cpu.gpr[d.XO.RT()] = r;
      if (d.XO.OE()) {
        // ~0 == -1: overflow only for INT64_MIN with no incoming carry.
        xer_record_overflow(cin == 0 && a == 0x8000000000000000ull);
      }
      if (d.XO.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::addzex: {
      uint64_t a = gpr(d.XO.RA());
      uint64_t cin = xer_ca();
      uint64_t r = a + cin;
      xer_set_ca(r < a);
      s_cpu.gpr[d.XO.RT()] = r;
      if (d.XO.OE()) xer_record_overflow(cin == 1 && a == 0x7FFFFFFFFFFFFFFFll);
      if (d.XO.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::subficx: {
      uint64_t ra = ra_or_zero(d.D.RA());
      // subfic: rD = SIMM - rA (32-bit, CA = SIMM >= rA unsigned)
      uint64_t r = (uint64_t)(uint32_t)(int64_t)d.D.SIMM() + ~(uint64_t)(uint32_t)ra + 1;
      s_cpu.gpr[d.D.RT()] = (uint64_t)(int64_t)(int32_t)(uint32_t)r;
      xer_set_ca(r > 0xFFFFFFFFull);
      break;
    }
    case PPCOpcode::subfx:
    case PPCOpcode::subfcx:
    case PPCOpcode::subfex:
    case PPCOpcode::subfmex:
    case PPCOpcode::subfzex: {
      uint64_t a = gpr(d.XO.RA());
      uint64_t b;
      uint64_t cin;
      if (opcode == PPCOpcode::subfex) { b = gpr(d.XO.RB()); cin = xer_ca(); }
      else if (opcode == PPCOpcode::subfmex) { b = ~0ull; cin = xer_ca(); }
      else if (opcode == PPCOpcode::subfzex) { b = 0ull; cin = xer_ca(); }
      else { b = gpr(d.XO.RB()); cin = 0; }
      uint64_t carry = 0;
      // r = b + ~a + cin (two's complement subtraction)
      uint64_t r = kAdd64(b, ~a, (opcode == PPCOpcode::subfx || opcode == PPCOpcode::subfcx) ? 1 : cin, &carry);
      if (opcode == PPCOpcode::subfcx || opcode == PPCOpcode::subfex ||
          opcode == PPCOpcode::subfmex || opcode == PPCOpcode::subfzex) {
        xer_set_ca(carry != 0);
      }
      s_cpu.gpr[d.XO.RT()] = r;
      if (d.XO.OE()) {
        bool ov = ((a ^ b) & (b ^ r) & 0x8000000000000000ull) != 0;
        xer_record_overflow(ov);
      }
      if (d.XO.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::negx: {
      uint64_t a = gpr(d.XO.RA());
      s_cpu.gpr[d.XO.RT()] = 0ull - a;
      if (d.XO.OE()) xer_record_overflow(a == 0x8000000000000000ull);
      if (d.XO.Rc()) cr0_from_i64((int64_t)(0ull - a));
      break;
    }
    case PPCOpcode::mulli: {
      int64_t a = (int64_t)gpr(d.D.RA());
      s_cpu.gpr[d.D.RT()] = (uint64_t)(a * (int64_t)d.D.SIMM());
      break;
    }
    case PPCOpcode::mullwx: {
      int32_t a = (int32_t)gpr(d.XO.RA());
      int32_t b = (int32_t)gpr(d.XO.RB());
      uint32_t product = (uint32_t)((uint64_t)(uint32_t)a * (uint32_t)b);
      s_cpu.gpr[d.XO.RT()] = (uint64_t)(int64_t)(int32_t)product;
      if (d.XO.OE()) xer_record_overflow(false);
      if (d.XO.Rc()) cr0_from_i32((int32_t)product);
      break;
    }
    case PPCOpcode::mulldx: {
      uint64_t r = gpr(d.XO.RA()) * gpr(d.XO.RB());
      s_cpu.gpr[d.XO.RT()] = r;
      if (d.XO.OE()) xer_record_overflow(false);
      if (d.XO.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::mulhwx: {
      int64_t a = (int32_t)gpr(d.XO.RA());
      int64_t b = (int32_t)gpr(d.XO.RB());
      int64_t p = a * b;
      s_cpu.gpr[d.XO.RT()] = (uint64_t)(int64_t)(int32_t)((uint64_t)p >> 32);
      if (d.XO.Rc()) cr0_from_i32((int32_t)((uint64_t)p >> 32));
      break;
    }
    case PPCOpcode::mulhwux: {
      uint32_t a = (uint32_t)gpr(d.XO.RA());
      uint32_t b = (uint32_t)gpr(d.XO.RB());
      s_cpu.gpr[d.XO.RT()] = (uint64_t)(((uint64_t)a * b) >> 32);
      if (d.XO.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.XO.RT()]);
      break;
    }
    case PPCOpcode::mulhdx: {
      __int128 a = (__int128)(int64_t)gpr(d.XO.RA());
      __int128 b = (__int128)(int64_t)gpr(d.XO.RB());
      s_cpu.gpr[d.XO.RT()] = (uint64_t)((a * b) >> 64);
      if (d.XO.Rc()) cr0_from_i64((int64_t)((a * b) >> 64));
      break;
    }
    case PPCOpcode::mulhdux: {
      unsigned __int128 a = (unsigned __int128)gpr(d.XO.RA());
      unsigned __int128 b = (unsigned __int128)gpr(d.XO.RB());
      s_cpu.gpr[d.XO.RT()] = (uint64_t)((a * b) >> 64);
      if (d.XO.Rc()) cr0_from_i64((int64_t)((a * b) >> 64));
      break;
    }
    case PPCOpcode::divwx:
    case PPCOpcode::divdx: {
      bool is64 = opcode == PPCOpcode::divdx;
      int64_t a = is64 ? (int64_t)gpr(d.XO.RA()) : (int32_t)gpr(d.XO.RA());
      int64_t b = is64 ? (int64_t)gpr(d.XO.RB()) : (int32_t)gpr(d.XO.RB());
      uint64_t r;
      if (b == 0) {
        r = ~0ull;
      } else if (is64) {
        r = (uint64_t)(a / b);
      } else {
        r = (uint64_t)(int64_t)(int32_t)((int32_t)a / (int32_t)b);
      }
      s_cpu.gpr[d.XO.RT()] = r;
      if (d.XO.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::divwux:
    case PPCOpcode::divdux: {
      bool is64 = opcode == PPCOpcode::divdux;
      uint64_t a = is64 ? gpr(d.XO.RA()) : (uint32_t)gpr(d.XO.RA());
      uint64_t b = is64 ? gpr(d.XO.RB()) : (uint32_t)gpr(d.XO.RB());
      if (b == 0) {
        s_cpu.gpr[d.XO.RT()] = ~0ull;
      } else if (is64) {
        s_cpu.gpr[d.XO.RT()] = a / b;
      } else {
        s_cpu.gpr[d.XO.RT()] = (uint32_t)a / (uint32_t)b;
      }
      if (d.XO.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.XO.RT()]);
      break;
    }

    // ---------------- Logical ----------------
    // Logical X-form: `and rA, rS, rB` - RA is the DESTINATION, RT is rS (source).
    case PPCOpcode::andx:    s_cpu.gpr[d.X.RA()] = gpr(d.X.RT()) & gpr(d.X.RB()); goto record_x;
    case PPCOpcode::andcx:   s_cpu.gpr[d.X.RA()] = gpr(d.X.RT()) & ~gpr(d.X.RB()); goto record_x;
    case PPCOpcode::orx:     s_cpu.gpr[d.X.RA()] = gpr(d.X.RT()) | gpr(d.X.RB()); goto record_x;
    case PPCOpcode::orcx:    s_cpu.gpr[d.X.RA()] = gpr(d.X.RT()) | ~gpr(d.X.RB()); goto record_x;
    case PPCOpcode::norx:    s_cpu.gpr[d.X.RA()] = ~(gpr(d.X.RT()) | gpr(d.X.RB())); goto record_x;
    case PPCOpcode::nandx:   s_cpu.gpr[d.X.RA()] = ~(gpr(d.X.RT()) & gpr(d.X.RB())); goto record_x;
    case PPCOpcode::eqvx:    s_cpu.gpr[d.X.RA()] = ~(gpr(d.X.RT()) ^ gpr(d.X.RB())); goto record_x;
    case PPCOpcode::xorx:    s_cpu.gpr[d.X.RA()] = gpr(d.X.RT()) ^ gpr(d.X.RB()); goto record_x;
    record_x:
      if (d.X.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.X.RA()]);
      break;
    case PPCOpcode::andix:   s_cpu.gpr[d.D.RA()] = gpr(d.D.RT()) & (uint64_t)d.D.UIMM(); cr0_from_i64((int64_t)s_cpu.gpr[d.D.RA()]); break;  // andi. always records
    case PPCOpcode::andisx:  s_cpu.gpr[d.D.RA()] = gpr(d.D.RT()) & ((uint64_t)d.D.UIMM() << 16); cr0_from_i64((int64_t)s_cpu.gpr[d.D.RA()]); break;  // andis. always records
    case PPCOpcode::ori:     s_cpu.gpr[d.D.RA()] = gpr(d.D.RT()) | (uint64_t)d.D.UIMM(); break;
    case PPCOpcode::oris:    s_cpu.gpr[d.D.RA()] = gpr(d.D.RT()) | ((uint64_t)d.D.UIMM() << 16); break;
    case PPCOpcode::xori:    s_cpu.gpr[d.D.RA()] = gpr(d.D.RT()) ^ (uint64_t)d.D.UIMM(); break;
    case PPCOpcode::xoris:   s_cpu.gpr[d.D.RA()] = gpr(d.D.RT()) ^ ((uint64_t)d.D.UIMM() << 16); break;

    // ---------------- Shifts & rotates ----------------
    case PPCOpcode::slwx: {
      uint32_t n = (uint32_t)gpr(d.X.RB()) & 0x3F;
      s_cpu.gpr[d.X.RA()] = (n & 0x20) ? 0 : (uint64_t)(uint32_t)gpr(d.X.RT()) << (n & 0x1F);
      if (d.X.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.X.RA()]);
      break;
    }
    case PPCOpcode::srwx: {
      uint32_t n = (uint32_t)gpr(d.X.RB()) & 0x3F;
      s_cpu.gpr[d.X.RA()] = (n & 0x20) ? 0 : (uint64_t)((uint32_t)gpr(d.X.RT()) >> (n & 0x1F));
      if (d.X.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.X.RA()]);
      break;
    }
    case PPCOpcode::srawx: {
      uint32_t n = (uint32_t)gpr(d.X.RB()) & 0x3F;
      int32_t v = (int32_t)gpr(d.X.RT());
      int32_t r = v >> (n & 0x1F);
      xer_set_ca(n >= 32 || (n != 0 && v < 0 && (r << (n & 0x1F)) != v));
      s_cpu.gpr[d.X.RA()] = (uint64_t)(int64_t)r;
      if (d.X.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::srawix: {
      uint32_t n = d.XS.SH() & 0x1F;
      int32_t v = (int32_t)gpr(d.XS.RT());
      int32_t r = v >> n;
      xer_set_ca(n != 0 && v < 0 && (r << n) != v);
      s_cpu.gpr[d.XS.RA()] = (uint64_t)(int64_t)r;
      if (d.XS.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::sldx: {
      uint32_t n = (uint32_t)gpr(d.X.RB()) & 0x7F;
      s_cpu.gpr[d.X.RA()] = (n & 0x40) ? 0 : (gpr(d.X.RT()) << n);
      if (d.X.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.X.RA()]);
      break;
    }
    case PPCOpcode::srdx: {
      uint32_t n = (uint32_t)gpr(d.X.RB()) & 0x7F;
      s_cpu.gpr[d.X.RA()] = (n & 0x40) ? 0 : (gpr(d.X.RT()) >> n);
      if (d.X.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.X.RA()]);
      break;
    }
    case PPCOpcode::sradix: {
      uint32_t n = d.XS.SH();
      int64_t v = (int64_t)gpr(d.XS.RT());
      s_cpu.gpr[d.XS.RA()] = (uint64_t)(v >> n);
      xer_set_ca(n != 0 && v < 0 && ((v >> n) << n) != v);
      if (d.XS.Rc()) cr0_from_i64((int64_t)(v >> n));
      break;
    }
    case PPCOpcode::rlwinmx: {
      uint32_t sh = d.M.SH();
      uint64_t m = XEMASK(d.M.MB() + 32, d.M.ME() + 32);
      uint64_t r = rotl64(gpr(d.M.RT()), sh) & m;
      s_cpu.gpr[d.M.RA()] = r;
      if (d.M.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::rlwimix: {
      uint32_t sh = d.M.SH();
      uint64_t m = XEMASK(d.M.MB() + 32, d.M.ME() + 32);
      uint64_t r = (rotl64(gpr(d.M.RT()), sh) & m) | (gpr(d.M.RA()) & ~m);
      s_cpu.gpr[d.M.RA()] = r;
      if (d.M.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::rlwnmx: {
      uint32_t sh = (uint32_t)gpr(d.M.RB()) & 0x1F;
      uint64_t m = XEMASK(d.M.MB() + 32, d.M.ME() + 32);
      uint64_t r = rotl64(gpr(d.M.RT()), sh) & m;
      s_cpu.gpr[d.M.RA()] = r;
      if (d.M.Rc()) cr0_from_i64((int64_t)r);
      break;
    }
    case PPCOpcode::rldiclx: {
      uint32_t sh = d.MD.SH(), mb = d.MD.MB();
      uint64_t m = XEMASK(mb, 63);
      uint64_t v = sh ? rotl64(gpr(d.MD.RT()), sh) : gpr(d.MD.RT());
      if (m != ~0ull) v &= m;
      s_cpu.gpr[d.MD.RA()] = v;
      if (d.MD.Rc()) cr0_from_i64((int64_t)v);
      break;
    }
    case PPCOpcode::rldicrx: {
      uint32_t sh = d.MD.SH(), me = d.MD.MB();  // ME/MB share the field for this format
      uint64_t m = XEMASK(0, me);
      uint64_t v = sh ? rotl64(gpr(d.MD.RT()), sh) : gpr(d.MD.RT());
      if (m != ~0ull) v &= m;
      s_cpu.gpr[d.MD.RA()] = v;
      if (d.MD.Rc()) cr0_from_i64((int64_t)v);
      break;
    }
    case PPCOpcode::rldicx: {
      uint32_t sh = d.MD.SH(), mb = d.MD.MB();
      uint64_t m = XEMASK(mb, ~sh);
      uint64_t v = sh ? rotl64(gpr(d.MD.RT()), sh) : gpr(d.MD.RT());
      s_cpu.gpr[d.MD.RA()] = v & m;
      if (d.MD.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.MD.RA()]);
      break;
    }
    case PPCOpcode::rldimix: {
      uint32_t sh = d.MD.SH(), mb = d.MD.MB();
      uint64_t m = XEMASK(mb, ~sh);
      uint64_t v = sh ? rotl64(gpr(d.MD.RT()), sh) : gpr(d.MD.RT());
      s_cpu.gpr[d.MD.RA()] = (m == ~0ull) ? v : ((v & m) | (gpr(d.MD.RA()) & ~m));
      if (d.MD.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.MD.RA()]);
      break;
    }
    case PPCOpcode::rldclx: {
      uint32_t sh = (uint32_t)gpr(d.MDS.RB()) & 0x3F;
      uint64_t m = XEMASK(d.MDS.MB(), 63);
      uint64_t v = rotl64(gpr(d.MDS.RT()), sh) & m;
      s_cpu.gpr[d.MDS.RA()] = v;
      if (d.MDS.Rc()) cr0_from_i64((int64_t)v);
      break;
    }
    case PPCOpcode::rldcrx: {
      uint32_t sh = (uint32_t)gpr(d.MDS.RB()) & 0x3F;
      uint64_t m = XEMASK(0, d.MDS.ME());
      uint64_t v = rotl64(gpr(d.MDS.RT()), sh) & m;
      s_cpu.gpr[d.MDS.RA()] = v;
      if (d.MDS.Rc()) cr0_from_i64((int64_t)v);
      break;
    }

    case PPCOpcode::extsbx: s_cpu.gpr[d.X.RA()] = (uint64_t)(int64_t)(int8_t)gpr(d.X.RT()); if (d.X.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.X.RA()]); break;
    case PPCOpcode::extshx: s_cpu.gpr[d.X.RA()] = (uint64_t)(int64_t)(int16_t)gpr(d.X.RT()); if (d.X.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.X.RA()]); break;
    case PPCOpcode::extswx: s_cpu.gpr[d.X.RA()] = (uint64_t)(int64_t)(int32_t)gpr(d.X.RT()); if (d.X.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.X.RA()]); break;
    case PPCOpcode::cntlzwx: {
      uint32_t v = (uint32_t)gpr(d.X.RT());
      s_cpu.gpr[d.X.RA()] = v == 0 ? 32 : (uint64_t)__builtin_clz(v);
      if (d.X.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.X.RA()]);
      break;
    }
    case PPCOpcode::cntlzdx: {
      uint64_t v = gpr(d.X.RT());
      s_cpu.gpr[d.X.RA()] = v == 0 ? 64 : (uint64_t)__builtin_clzll(v);
      if (d.X.Rc()) cr0_from_i64((int64_t)s_cpu.gpr[d.X.RA()]);
      break;
    }

    // ---------------- Compare ----------------
    case PPCOpcode::cmp:
    case PPCOpcode::cmpl: {
      uint32_t crf = d.X.CRFD();
      uint64_t a = gpr(d.X.RA());
      uint64_t b = gpr(d.X.RB());
      if (opcode == PPCOpcode::cmp) {
        cr_set_field(crf, (int64_t)a < (int64_t)b, (int64_t)a > (int64_t)b, a == b, false);
      } else {
        cr_set_field(crf, a < b, a > b, a == b, false);
      }
      break;
    }
    case PPCOpcode::cmpi:
    case PPCOpcode::cmpli: {
      uint32_t crf = d.D.CRFD();
      uint64_t a = gpr(d.D.RA());  // compares use the register r0, not literal 0
      if (opcode == PPCOpcode::cmpi) {
        int64_t sa = (int64_t)a, sb = (int64_t)d.D.SIMM();
        cr_set_field(crf, sa < sb, sa > sb, sa == sb, false);
      } else {
        uint64_t sb = (uint64_t)(uint16_t)d.D.SIMM();
        cr_set_field(crf, a < sb, a > sb, a == sb, false);
      }
      break;
    }

    // ---------------- Condition register ----------------
    case PPCOpcode::mcrf: {
      uint32_t dst = d.XL.CRBD() >> 2, src = d.XL.CRBA() >> 2;
      uint32_t val = cr_field(src);
      uint32_t shift = 28 - 4 * dst;
      s_cpu.cr = (s_cpu.cr & ~(0xFu << shift)) | (val << shift);
      break;
    }
    case PPCOpcode::mfcr:
      s_cpu.gpr[d.X.RT()] = s_cpu.cr;
      break;
    case PPCOpcode::mtcrf: {
      uint32_t crm = (code >> 12) & 0xFF;  // CRM field
      uint64_t rs = gpr(d.X.RT());
      for (uint32_t i = 0; i < 8; i++) {
        if (crm & (0x80u >> i)) {
          uint32_t shift = 28 - 4 * i;
          uint32_t field = (uint32_t)((rs >> shift) & 0xF);
          s_cpu.cr = (s_cpu.cr & ~(0xFu << shift)) | (field << shift);
        }
      }
      break;
    }
    case PPCOpcode::crand:  cr_set_bit(d.XL.CRBD(), cr_get_bit(d.XL.CRBA()) && cr_get_bit(d.XL.CRBB())); break;
    case PPCOpcode::crandc: cr_set_bit(d.XL.CRBD(), cr_get_bit(d.XL.CRBA()) && !cr_get_bit(d.XL.CRBB())); break;
    case PPCOpcode::creqv:  cr_set_bit(d.XL.CRBD(), !(cr_get_bit(d.XL.CRBA()) ^ cr_get_bit(d.XL.CRBB()))); break;
    case PPCOpcode::crnand: cr_set_bit(d.XL.CRBD(), !(cr_get_bit(d.XL.CRBA()) && cr_get_bit(d.XL.CRBB()))); break;
    case PPCOpcode::crnor:  cr_set_bit(d.XL.CRBD(), !(cr_get_bit(d.XL.CRBA()) || cr_get_bit(d.XL.CRBB()))); break;
    case PPCOpcode::cror:   cr_set_bit(d.XL.CRBD(), cr_get_bit(d.XL.CRBA()) || cr_get_bit(d.XL.CRBB())); break;
    case PPCOpcode::crorc:  cr_set_bit(d.XL.CRBD(), cr_get_bit(d.XL.CRBA()) || !cr_get_bit(d.XL.CRBB())); break;
    case PPCOpcode::crxor:  cr_set_bit(d.XL.CRBD(), cr_get_bit(d.XL.CRBA()) ^ cr_get_bit(d.XL.CRBB())); break;

    // ---------------- SPR / MSR ----------------
    case PPCOpcode::mfspr: {
      uint32_t spr = d.XFX.TBR();
      uint64_t v = 0;
      switch (spr) {
        case 1: v = s_cpu.xer; break;
        case 8: v = s_cpu.lr; break;
        case 9: v = s_cpu.ctr; break;
        default: v = 0; break;
      }
      s_cpu.gpr[d.XFX.RT()] = v;
      break;
    }
    case PPCOpcode::mtspr: {
      uint32_t spr = d.XFX.TBR();
      uint64_t v = gpr(d.XFX.RT());
      switch (spr) {
        case 1: s_cpu.xer = v & 0x00000000FFFFFFFFull; break;
        case 8: s_cpu.lr = v; break;
        case 9: s_cpu.ctr = v; break;
        default: break;
      }
      break;
    }
    case PPCOpcode::mftb: {
      uint32_t tbr = d.XFX.TBR();
      s_cpu.gpr[d.XFX.RT()] = (tbr == 268) ? s_cpu.tbr : (s_cpu.tbr >> 32);
      break;
    }
    case PPCOpcode::mfmsr:
      s_cpu.gpr[d.X.RT()] = s_cpu.msr;
      break;
    case PPCOpcode::mtmsr:
    case PPCOpcode::mtmsrd:
      s_cpu.msr = gpr(d.X.RT());
      break;

    // ---------------- Branching ----------------
    case PPCOpcode::bx: {
      if (d.I.LK()) s_cpu.lr = next_pc;
      set_pc(d.I.ADDR());
      break;
    }
    case PPCOpcode::bcx: {
      uint64_t target = (uint64_t)(int64_t)d.B.ADDR();
      if (d.B.LK()) s_cpu.lr = next_pc;
      if (branch_condition(d.B.BO(), d.B.BI())) {
        set_pc(target);
      }
      break;
    }
    case PPCOpcode::bclrx: {
      uint64_t target = s_cpu.lr;
      if (d.XL.LK()) s_cpu.lr = next_pc;
      if (branch_condition(d.XL.BO(), d.XL.BI())) {
        set_pc(target);
      }
      break;
    }
    case PPCOpcode::bcctrx: {
      uint64_t target = s_cpu.ctr;
      if (d.XL.LK()) s_cpu.lr = next_pc;
      if (branch_condition(d.XL.BO(), d.XL.BI())) {
        set_pc(target & ~3ull);
      }
      break;
    }

    // ---------------- Traps / syscalls ----------------
    case PPCOpcode::twi:
    case PPCOpcode::tw: {
      uint32_t to = d.D.TO();
      int64_t a = (int64_t)gpr(d.D.RA());
      int64_t b = (opcode == PPCOpcode::twi) ? (int64_t)d.D.SIMM() : (int64_t)gpr(d.X.RB());
      uint32_t cond = 0;
      if (a < b) cond |= 0x10;
      if (a > b) cond |= 0x08;
      if (a == b) cond |= 0x04;
      uint64_t ua = (uint64_t)a, ub = (uint64_t)b;
      if (ua < ub) cond |= 0x02;
      if (ua > ub) cond |= 0x01;
      if ((to & cond) != 0) {
        s_cpu.status = kStatusTrap;
        return kStatusTrap;
      }
      break;
    }
    case PPCOpcode::sc: {
      s_cpu.sc_level = (code >> 5) & 0x7F;
      s_cpu.status = kStatusSyscall;
      return kStatusSyscall;
    }

    // ---------------- Cache / sync ----------------
    case PPCOpcode::dcbz:
    case PPCOpcode::dcbz128: {
      uint64_t addr = ra_or_zero(d.DCBZ.RA()) + gpr(d.X.RB());
      uint64_t line = 128;  // Xenon cache line is 128 bytes
      uint64_t aligned = addr & ~(line - 1);
      if (!ram_in_bounds(aligned, line)) return mem_fault(aligned) ? kStatusMemoryFault : kStatusMemoryFault;
      __builtin_memset(g_guest_ram + aligned, 0, (size_t)line);
      break;
    }
    case PPCOpcode::dcbf: case PPCOpcode::dcbi: case PPCOpcode::dcbst:
    case PPCOpcode::dcbt: case PPCOpcode::dcbtst: case PPCOpcode::icbi:
    case PPCOpcode::isync: case PPCOpcode::sync: case PPCOpcode::eieio:
      break;  // cache ops are no-ops in the emulator

    // ---------------- Loads (D-form, sign-extend flag per opcode) ----------------
    case PPCOpcode::lbz: case PPCOpcode::lhz: case PPCOpcode::lha: case PPCOpcode::lwz: {
      uint32_t size; bool sex;
      switch (opcode) {
        case PPCOpcode::lbz: size = 1; sex = false; break;
        case PPCOpcode::lhz: size = 2; sex = false; break;
        case PPCOpcode::lha: size = 2; sex = true; break;
        default:             size = 4; sex = false; break;  // lwz
      }
      uint64_t addr = ra_or_zero(d.D.RA()) + (uint64_t)(int64_t)d.D.d();
      uint64_t v;
      if (!load_mem(addr, &v, size, sex)) return kStatusMemoryFault;
      s_cpu.gpr[d.D.RT()] = v;
      break;
    }
    case PPCOpcode::lwa: {
      // lwa is DS-form: displacement is (DS << 2), sign-extended 16 bits.
      uint64_t addr = ra_or_zero(d.DS.RA()) + (uint64_t)(int64_t)d.DS.ds();
      uint64_t v;
      if (!load_mem(addr, &v, 4, true)) return kStatusMemoryFault;
      s_cpu.gpr[d.DS.RT()] = v;
      break;
    }
    case PPCOpcode::ld: {
      uint64_t addr = ra_or_zero(d.DS.RA()) + (uint64_t)(int64_t)d.DS.ds();
      uint64_t v;
      if (!load_mem(addr, &v, 8, false)) return kStatusMemoryFault;
      s_cpu.gpr[d.DS.RT()] = v;
      break;
    }
    case PPCOpcode::ldu: {
      uint64_t addr = gpr(d.DS.RA()) + (uint64_t)(int64_t)d.DS.ds();  // ldu requires RA != 0
      uint64_t v;
      if (!load_mem(addr, &v, 8, false)) return kStatusMemoryFault;
      s_cpu.gpr[d.DS.RT()] = v;
      s_cpu.gpr[d.DS.RA()] = addr;
      break;
    }
    case PPCOpcode::lbzu: case PPCOpcode::lhzu: case PPCOpcode::lhau: case PPCOpcode::lwzu: {
      uint32_t size; bool sex;
      switch (opcode) {
        case PPCOpcode::lbzu: size = 1; sex = false; break;
        case PPCOpcode::lhzu: size = 2; sex = false; break;
        case PPCOpcode::lhau: size = 2; sex = true; break;
        default:              size = 4; sex = false; break;
      }
      uint64_t addr = gpr(d.D.RA()) + (uint64_t)(int64_t)d.D.SIMM();
      uint64_t v;
      if (!load_mem(addr, &v, size, sex)) return kStatusMemoryFault;
      s_cpu.gpr[d.D.RT()] = v;
      s_cpu.gpr[d.D.RA()] = addr;
      break;
    }
    case PPCOpcode::lbzx: case PPCOpcode::lhzx: case PPCOpcode::lhax:
    case PPCOpcode::lwzx: case PPCOpcode::ldx: case PPCOpcode::lwaux: {
      uint32_t size; bool sex;
      switch (opcode) {
        case PPCOpcode::lbzx:   size = 1; sex = false; break;
        case PPCOpcode::lhzx:   size = 2; sex = false; break;
        case PPCOpcode::lhax:   size = 2; sex = true; break;
        case PPCOpcode::lwzx:   size = 4; sex = false; break;
        case PPCOpcode::lwaux:  size = 4; sex = true; break;
        default:                size = 8; sex = false; break;
      }
      uint64_t addr = ra_or_zero(d.X.RA()) + gpr(d.X.RB());
      uint64_t v;
      if (!load_mem(addr, &v, size, sex)) return kStatusMemoryFault;
      s_cpu.gpr[d.X.RT()] = v;
      break;
    }
    case PPCOpcode::lbzux: case PPCOpcode::lhzux: case PPCOpcode::lhaux:
    case PPCOpcode::lwzux: case PPCOpcode::ldux: {
      uint32_t size; bool sex;
      switch (opcode) {
        case PPCOpcode::lbzux: size = 1; sex = false; break;
        case PPCOpcode::lhzux: size = 2; sex = false; break;
        case PPCOpcode::lhaux: size = 2; sex = true; break;
        case PPCOpcode::lwzux: size = 4; sex = false; break;
        default:               size = 8; sex = false; break;
      }
      uint64_t addr = gpr(d.X.RA()) + gpr(d.X.RB());
      uint64_t v;
      if (!load_mem(addr, &v, size, sex)) return kStatusMemoryFault;
      s_cpu.gpr[d.X.RT()] = v;
      s_cpu.gpr[d.X.RA()] = addr;
      break;
    }
    case PPCOpcode::ldarx: {
      uint64_t addr = ra_or_zero(d.X.RA()) + gpr(d.X.RB());
      uint64_t v;
      if (!load_mem(addr, &v, 8, false)) return kStatusMemoryFault;
      s_cpu.gpr[d.X.RT()] = v;
      s_cpu.reservation = addr;
      break;
    }
    case PPCOpcode::lmw: {
      uint64_t addr = ra_or_zero(d.D.RA()) + (uint64_t)(int64_t)d.D.d();
      for (uint32_t i = d.D.RT(); i < 32; i++) {
        uint64_t v;
        if (!load_mem(addr, &v, 4, false)) return kStatusMemoryFault;
        s_cpu.gpr[i] = v;
        addr += 4;
      }
      break;
    }

    // ---------------- Stores ----------------
    case PPCOpcode::stb: case PPCOpcode::sth: case PPCOpcode::stw: {
      uint32_t size = opcode == PPCOpcode::stb ? 1 : opcode == PPCOpcode::sth ? 2 : 4;
      uint64_t addr = ra_or_zero(d.D.RA()) + (uint64_t)(int64_t)d.D.d();
      int r = store_mem(addr, gpr(d.D.RT()), size);
      if (r == 1) return kStatusMemoryFault;
      if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      break;
    }
    case PPCOpcode::std: {
      uint64_t addr = ra_or_zero(d.DS.RA()) + (uint64_t)(int64_t)d.DS.ds();
      int r = store_mem(addr, gpr(d.DS.RT()), 8);
      if (r == 1) return kStatusMemoryFault;
      if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      break;
    }
    case PPCOpcode::stdu: {
      uint64_t addr = gpr(d.DS.RA()) + (uint64_t)(int64_t)d.DS.ds();
      int r = store_mem(addr, gpr(d.DS.RT()), 8);
      if (r == 1) return kStatusMemoryFault;
      if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      s_cpu.gpr[d.DS.RA()] = addr;
      break;
    }
    case PPCOpcode::stbu: case PPCOpcode::sthu: case PPCOpcode::stwu: {
      uint32_t size = opcode == PPCOpcode::stbu ? 1 : opcode == PPCOpcode::sthu ? 2 : 4;
      uint64_t addr = gpr(d.D.RA()) + (uint64_t)(int64_t)d.D.SIMM();
      int r = store_mem(addr, gpr(d.D.RT()), size);
      if (r == 1) return kStatusMemoryFault;
      if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      s_cpu.gpr[d.D.RA()] = addr;
      break;
    }
    case PPCOpcode::stbx: case PPCOpcode::sthx: case PPCOpcode::stwx: case PPCOpcode::stdx: {
      uint32_t size = opcode == PPCOpcode::stbx ? 1 : opcode == PPCOpcode::sthx ? 2 : opcode == PPCOpcode::stwx ? 4 : 8;
      uint64_t addr = ra_or_zero(d.X.RA()) + gpr(d.X.RB());
      int r = store_mem(addr, gpr(d.X.RT()), size);
      if (r == 1) return kStatusMemoryFault;
      if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      break;
    }
    case PPCOpcode::stbux: case PPCOpcode::sthux: case PPCOpcode::stwux: case PPCOpcode::stdux: {
      uint32_t size = opcode == PPCOpcode::stbux ? 1 : opcode == PPCOpcode::sthux ? 2 : opcode == PPCOpcode::stwux ? 4 : 8;
      uint64_t addr = gpr(d.X.RA()) + gpr(d.X.RB());
      int r = store_mem(addr, gpr(d.X.RT()), size);
      if (r == 1) return kStatusMemoryFault;
      if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      s_cpu.gpr[d.X.RA()] = addr;
      break;
    }
    case PPCOpcode::stdcx: {
      uint64_t addr = ra_or_zero(d.X.RA()) + gpr(d.X.RB());
      bool reserved = s_cpu.reservation == addr;
      int r = store_mem(addr, gpr(d.X.RT()), 8);
      if (r == 1) return kStatusMemoryFault;
      if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      cr_set_field(0, false, false, reserved, false);
      break;
    }
    case PPCOpcode::stmw: {
      uint64_t addr = ra_or_zero(d.D.RA()) + (uint64_t)(int64_t)d.D.d();
      for (uint32_t i = d.D.RT(); i < 32; i++) {
        int r = store_mem(addr, s_cpu.gpr[i], 4);
        if (r == 1) return kStatusMemoryFault;
        if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
        addr += 4;
      }
      break;
    }

    // ---------------- FP load/store ----------------
    case PPCOpcode::lfs: case PPCOpcode::lfd: {
      uint64_t addr = ra_or_zero(d.D.RA()) + (uint64_t)(int64_t)d.D.d();
      if (opcode == PPCOpcode::lfs) {
        uint64_t v;
        if (!load_mem(addr, &v, 4, false)) return kStatusMemoryFault;
        union { uint32_t u; float f; } cvt;
        cvt.u = (uint32_t)v;
        s_cpu.fpr_bits[d.D.RT()] = double_to_bits((double)cvt.f);
      } else {
        uint64_t v;
        if (!load_mem(addr, &v, 8, false)) return kStatusMemoryFault;
        s_cpu.fpr_bits[d.D.RT()] = v;
      }
      break;
    }
    case PPCOpcode::lfsx: case PPCOpcode::lfdx: {
      uint64_t addr = ra_or_zero(d.X.RA()) + gpr(d.X.RB());
      if (opcode == PPCOpcode::lfsx) {
        uint64_t v;
        if (!load_mem(addr, &v, 4, false)) return kStatusMemoryFault;
        union { uint32_t u; float f; } cvt;
        cvt.u = (uint32_t)v;
        s_cpu.fpr_bits[d.X.RT()] = double_to_bits((double)cvt.f);
      } else {
        uint64_t v;
        if (!load_mem(addr, &v, 8, false)) return kStatusMemoryFault;
        s_cpu.fpr_bits[d.X.RT()] = v;
      }
      break;
    }
    case PPCOpcode::stfs: case PPCOpcode::stfd: {
      uint64_t addr = ra_or_zero(d.D.RA()) + (uint64_t)(int64_t)d.D.d();
      double v = bits_to_double(s_cpu.fpr_bits[d.D.RT()]);
      if (opcode == PPCOpcode::stfs) {
        union { float f; uint32_t u; } cvt;
        cvt.f = (float)v;
        int r = store_mem(addr, (uint64_t)cvt.u, 4);
        if (r == 1) return kStatusMemoryFault;
        if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      } else {
        int r = store_mem(addr, double_to_bits(v), 8);
        if (r == 1) return kStatusMemoryFault;
        if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      }
      break;
    }
    case PPCOpcode::stfsx: case PPCOpcode::stfdx: {
      uint64_t addr = ra_or_zero(d.X.RA()) + gpr(d.X.RB());
      double v = bits_to_double(s_cpu.fpr_bits[d.X.RT()]);
      if (opcode == PPCOpcode::stfsx) {
        union { float f; uint32_t u; } cvt;
        cvt.f = (float)v;
        int r = store_mem(addr, (uint64_t)cvt.u, 4);
        if (r == 1) return kStatusMemoryFault;
        if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      } else {
        int r = store_mem(addr, double_to_bits(v), 8);
        if (r == 1) return kStatusMemoryFault;
        if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      }
      break;
    }
    case PPCOpcode::stfiwx: {
      uint64_t addr = ra_or_zero(d.X.RA()) + gpr(d.X.RB());
      int r = store_mem(addr, s_cpu.fpr_bits[d.X.RT()] & 0xFFFFFFFFull, 4);
      if (r == 1) return kStatusMemoryFault;
      if (r == 2) { s_cpu.pc = next_pc; s_cpu.instructions_executed++; return kStatusSyscall; }
      break;
    }

    // ---------------- FP arithmetic ----------------
    case PPCOpcode::fmrx: {
      s_cpu.fpr_bits[d.X.RT()] = s_cpu.fpr_bits[d.X.RB()];
      fp_record_cr1(d.X.Rc());
      break;
    }
    case PPCOpcode::fabsx: s_cpu.fpr_bits[d.X.RT()] = s_cpu.fpr_bits[d.X.RB()] & ~(1ull << 63); fp_record_cr1(d.X.Rc()); break;
    case PPCOpcode::fnegx: s_cpu.fpr_bits[d.X.RT()] = s_cpu.fpr_bits[d.X.RB()] ^ (1ull << 63); fp_record_cr1(d.X.Rc()); break;
    case PPCOpcode::fnabsx: s_cpu.fpr_bits[d.X.RT()] = s_cpu.fpr_bits[d.X.RB()] | (1ull << 63); fp_record_cr1(d.X.Rc()); break;
    case PPCOpcode::faddx: case PPCOpcode::fsubx: case PPCOpcode::fmulx: case PPCOpcode::fdivx: {
      double a = bits_to_double(s_cpu.fpr_bits[d.A.FA()]);
      double b = bits_to_double(s_cpu.fpr_bits[d.A.FB()]);
      double r = opcode == PPCOpcode::faddx ? a + b
               : opcode == PPCOpcode::fsubx ? a - b
               : opcode == PPCOpcode::fmulx ? a * b
               : (b == 0.0 ? (a < 0.0 ? -__builtin_huge_val() : __builtin_huge_val()) : a / b);
      s_cpu.fpr_bits[d.A.FT()] = double_to_bits(r);
      fp_record_cr1(d.A.Rc());
      break;
    }
    case PPCOpcode::fmaddx: case PPCOpcode::fmaddsx:
    case PPCOpcode::fmsubx: case PPCOpcode::fmsubsx:
    case PPCOpcode::fnmaddx: case PPCOpcode::fnmaddsx:
    case PPCOpcode::fnmsubx: case PPCOpcode::fnmsubsx: {
      double a = bits_to_double(s_cpu.fpr_bits[d.A.FA()]);
      double b = bits_to_double(s_cpu.fpr_bits[d.A.FB()]);
      double c = bits_to_double(s_cpu.fpr_bits[d.A.FC()]);
      double r;
      bool negative = opcode == PPCOpcode::fnmaddx || opcode == PPCOpcode::fnmaddsx ||
                      opcode == PPCOpcode::fnmsubx || opcode == PPCOpcode::fnmsubsx;
      bool subtract = opcode == PPCOpcode::fmsubx || opcode == PPCOpcode::fmsubsx ||
                      opcode == PPCOpcode::fnmsubx || opcode == PPCOpcode::fnmsubsx;
      r = a * b + (subtract ? -c : c);
      if (negative) r = -r;
      if (opcode == PPCOpcode::fmaddsx || opcode == PPCOpcode::fmsubsx ||
          opcode == PPCOpcode::fnmaddsx || opcode == PPCOpcode::fnmsubsx) {
        r = (double)(float)r;
      }
      s_cpu.fpr_bits[d.A.FT()] = double_to_bits(r);
      fp_record_cr1(d.A.Rc());
      break;
    }
    case PPCOpcode::faddsx: case PPCOpcode::fsubsx: case PPCOpcode::fmulsx: case PPCOpcode::fdivsx: {
      float a = (float)bits_to_double(s_cpu.fpr_bits[d.A.FA()]);
      float b = (float)bits_to_double(s_cpu.fpr_bits[d.A.FB()]);
      float r = opcode == PPCOpcode::faddsx ? a + b
              : opcode == PPCOpcode::fsubsx ? a - b
              : opcode == PPCOpcode::fmulsx ? a * b
              : (b == 0.0f ? (a < 0.0f ? -3.4e38f : 3.4e38f) : a / b);
      s_cpu.fpr_bits[d.A.FT()] = double_to_bits((double)r);
      fp_record_cr1(d.A.Rc());
      break;
    }
    case PPCOpcode::fselx: {
      double a = bits_to_double(s_cpu.fpr_bits[d.A.FA()]);
      double b = bits_to_double(s_cpu.fpr_bits[d.A.FB()]);
      double c = bits_to_double(s_cpu.fpr_bits[d.A.FC()]);
      bool nan = a != a;
      s_cpu.fpr_bits[d.A.FT()] = double_to_bits((a >= 0.0 || nan) ? b : c);
      fp_record_cr1(d.A.Rc());
      break;
    }
    case PPCOpcode::fsqrtx: {
      double a = bits_to_double(s_cpu.fpr_bits[d.A.FB()]);
      s_cpu.fpr_bits[d.A.FT()] = double_to_bits(a >= 0.0 ? __builtin_sqrt(a) : a);
      fp_record_cr1(d.A.Rc());
      break;
    }
    case PPCOpcode::fsqrtsx: {
      double a = (double)(float)bits_to_double(s_cpu.fpr_bits[d.A.FB()]);
      float r = a >= 0.0 ? (float)__builtin_sqrt(a) : (float)a;
      s_cpu.fpr_bits[d.A.FT()] = double_to_bits((double)r);
      fp_record_cr1(d.A.Rc());
      break;
    }
    case PPCOpcode::frspx: {
      float f = (float)bits_to_double(s_cpu.fpr_bits[d.X.RT()]);
      s_cpu.fpr_bits[d.X.RT()] = double_to_bits((double)f);
      break;
    }
    case PPCOpcode::frsqrtex: {
      double b = bits_to_double(s_cpu.fpr_bits[d.A.FB()]);
      s_cpu.fpr_bits[d.A.FT()] = double_to_bits(b > 0.0 ? 1.0 / __builtin_sqrt(b) : b);
      fp_record_cr1(d.A.Rc());
      break;
    }
    case PPCOpcode::fresx: {
      double b = bits_to_double(s_cpu.fpr_bits[d.A.FB()]);
      s_cpu.fpr_bits[d.A.FT()] = double_to_bits(b != 0.0 ? 1.0 / b : b);
      fp_record_cr1(d.A.Rc());
      break;
    }
    case PPCOpcode::fctiwx:
    case PPCOpcode::fctiwzx: {
      double b = bits_to_double(s_cpu.fpr_bits[d.X.RB()]);
      int32_t r = 0;
      if (b == b) r = b >= 0.0 ? (int32_t)b : -(int32_t)(-b);  // truncated
      s_cpu.fpr_bits[d.X.RT()] = (uint64_t)(uint32_t)r;
      break;
    }
    case PPCOpcode::fctidx:
    case PPCOpcode::fctidzx: {
      double b = bits_to_double(s_cpu.fpr_bits[d.X.RB()]);
      int64_t r = 0;
      if (b == b) r = b >= 0.0 ? (int64_t)b : -(int64_t)(-b);
      s_cpu.fpr_bits[d.X.RT()] = (uint64_t)r;
      break;
    }
    case PPCOpcode::fcmpo:
    case PPCOpcode::fcmpu: {
      double a = bits_to_double(s_cpu.fpr_bits[d.X.RA()]);
      double b = bits_to_double(s_cpu.fpr_bits[d.X.RB()]);
      uint32_t crf = d.X.CRFD();
      uint32_t field;
      if (a != a || b != b) field = 0x1;   // UN (SO bit)
      else if (a < b) field = 0x8;         // LT
      else if (a > b) field = 0x4;         // GT
      else field = 0x2;                    // EQ
      uint32_t shift = 28 - 4 * crf;
      s_cpu.cr = (s_cpu.cr & ~(0xFu << shift)) | (field << shift);
      s_cpu.fpscr = (s_cpu.fpscr & ~0xF0000000ull) | ((uint64_t)field << 28);
      break;
    }
    case PPCOpcode::mffsx:
      s_cpu.fpr_bits[d.X.RT()] = s_cpu.fpscr;
      break;
    case PPCOpcode::mcrfs: {
      uint32_t dst = d.X.CRFD();
      uint32_t src = d.X.CRFS();
      uint32_t field = (uint32_t)((s_cpu.fpscr >> (28 - 4 * src)) & 0xF);
      uint32_t shift = 28 - 4 * dst;
      s_cpu.cr = (s_cpu.cr & ~(0xFu << shift)) | (field << shift);
      break;
    }

    default:
      return unimplemented();
  }

  return finish();
}

}  // namespace

extern "C" {

__attribute__((export_name("xeniatest_reset")))
void xeniatest_reset(uint64_t entry_pc) {
  __builtin_memset(&s_cpu, 0, sizeof(s_cpu));
  __builtin_memset(g_guest_ram, 0, kGuestRamSize);
  s_cpu.msr = 0x8000000000000000ull;  // SF: 64-bit mode
  s_cpu.reservation = ~0ull;
  s_cpu.status = kStatusOk;
  s_cpu.trap_on_unimplemented = 1;
  s_cpu.pc = entry_pc;
  s_console_len = 0;
}

__attribute__((export_name("xeniatest_run")))
uint32_t xeniatest_run(uint32_t max_instructions) {
  for (uint32_t i = 0; i < max_instructions; i++) {
    uint32_t status = execute_one();
    if (status != kStatusOk) {
      return status;
    }
  }
  return kStatusOk;
}

__attribute__((export_name("xeniatest_step")))
uint32_t xeniatest_step(void) { return execute_one(); }

__attribute__((export_name("xeniatest_status")))
uint32_t xeniatest_status(void) { return s_cpu.status; }
__attribute__((export_name("xeniatest_sc_level")))
uint32_t xeniatest_sc_level(void) { return s_cpu.sc_level; }
__attribute__((export_name("xeniatest_pc")))
uint64_t xeniatest_pc(void) { return s_cpu.pc; }
__attribute__((export_name("xeniatest_instructions_executed")))
uint64_t xeniatest_instructions_executed(void) { return s_cpu.instructions_executed; }
__attribute__((export_name("xeniatest_unimplemented_count")))
uint64_t xeniatest_unimplemented_count(void) { return s_cpu.unimplemented_count; }
__attribute__((export_name("xeniatest_last_unimplemented")))
uint32_t xeniatest_last_unimplemented(void) { return (uint32_t)s_cpu.last_unimplemented; }
__attribute__((export_name("xeniatest_last_fault_address")))
uint64_t xeniatest_last_fault_address(void) { return s_cpu.last_fault_address; }

__attribute__((export_name("xeniatest_set_trap_on_unimplemented")))
void xeniatest_set_trap_on_unimplemented(uint32_t enabled) { s_cpu.trap_on_unimplemented = enabled ? 1 : 0; }
__attribute__((export_name("xeniatest_trap_on_unimplemented")))
uint32_t xeniatest_trap_on_unimplemented(void) { return s_cpu.trap_on_unimplemented; }

__attribute__((export_name("xeniatest_gpr")))
uint64_t xeniatest_gpr(uint32_t index) { return s_cpu.gpr[index & 31]; }
__attribute__((export_name("xeniatest_set_gpr")))
void xeniatest_set_gpr(uint32_t index, uint64_t value) { s_cpu.gpr[index & 31] = value; }
__attribute__((export_name("xeniatest_fpr_bits")))
uint64_t xeniatest_fpr_bits(uint32_t index) { return s_cpu.fpr_bits[index & 31]; }
__attribute__((export_name("xeniatest_set_fpr_bits")))
void xeniatest_set_fpr_bits(uint32_t index, uint64_t value) { s_cpu.fpr_bits[index & 31] = value; }
__attribute__((export_name("xeniatest_cr")))
uint32_t xeniatest_cr(void) { return s_cpu.cr; }
__attribute__((export_name("xeniatest_set_cr")))
void xeniatest_set_cr(uint32_t value) { s_cpu.cr = value; }
__attribute__((export_name("xeniatest_lr")))
uint64_t xeniatest_lr(void) { return s_cpu.lr; }
__attribute__((export_name("xeniatest_ctr")))
uint64_t xeniatest_ctr(void) { return s_cpu.ctr; }
__attribute__((export_name("xeniatest_xer")))
uint64_t xeniatest_xer(void) { return s_cpu.xer; }

__attribute__((export_name("xeniatest_guest_ram_size")))
uint32_t xeniatest_guest_ram_size(void) { return (uint32_t)kGuestRamSize; }
__attribute__((export_name("xeniatest_guest_ram_pointer")))
uint32_t xeniatest_guest_ram_pointer(void) { return (uint32_t)(uintptr_t)g_guest_ram; }

__attribute__((export_name("xeniatest_console_data")))
const uint8_t* xeniatest_console_data(void) { return (const uint8_t*)g_console; }
__attribute__((export_name("xeniatest_console_length")))
uint32_t xeniatest_console_length(void) { return (uint32_t)s_console_len; }
__attribute__((export_name("xeniatest_console_consume")))
void xeniatest_console_consume(uint32_t n) {
  if (n >= s_console_len) {
    s_console_len = 0;
    return;
  }
  __builtin_memmove(g_console, g_console + n, s_console_len - n);
  s_console_len -= n;
}

// --- Disassembler ----------------------------------------------------------------
StringBuffer s_disasm_line;
char s_opcode_group[24];

__attribute__((export_name("xeniatest_disasm")))
uint32_t xeniatest_disasm(uint32_t address, uint32_t code) {
  s_disasm_line.clear();
  xe::cpu::ppc::DisasmPPC(address, code, &s_disasm_line);
  return (uint32_t)s_disasm_line.length();
}
__attribute__((export_name("xeniatest_disasm_line")))
const char* xeniatest_disasm_line(void) { return s_disasm_line.c_str(); }

__attribute__((export_name("xeniatest_opcode_name")))
const char* xeniatest_opcode_name(uint32_t code) {
  auto opcode = xe::cpu::ppc::LookupOpcode(code);
  if (opcode == PPCOpcode::kInvalid) return "";
  return xe::cpu::ppc::GetOpcodeDisasmInfo(opcode).name;
}

__attribute__((export_name("xeniatest_opcode_group")))
const char* xeniatest_opcode_group(uint32_t code) {
  auto opcode = xe::cpu::ppc::LookupOpcode(code);
  const char* name = "unknown";
  if (opcode != PPCOpcode::kInvalid) {
    switch (xe::cpu::ppc::GetOpcodeInfo(opcode).group) {
      case xe::cpu::ppc::PPCOpcodeGroup::kB: name = "branch"; break;
      case xe::cpu::ppc::PPCOpcodeGroup::kC: name = "control"; break;
      case xe::cpu::ppc::PPCOpcodeGroup::kM: name = "memory"; break;
      case xe::cpu::ppc::PPCOpcodeGroup::kI: name = "integer"; break;
      case xe::cpu::ppc::PPCOpcodeGroup::kF: name = "floating-point"; break;
      case xe::cpu::ppc::PPCOpcodeGroup::kV: name = "vmx"; break;
    }
  }
  size_t len = xeniatest_strlen(name) + 1;
  if (len > sizeof(s_opcode_group)) len = sizeof(s_opcode_group);
  __builtin_memcpy(s_opcode_group, name, len);
  return s_opcode_group;
}

}  // extern "C"
