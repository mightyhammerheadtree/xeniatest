/**
 * Xenia Canary - Web port compatibility shim.
 *
 * Xenia: Xbox 360 Emulator Research Project
 * Original Copyright 2015 Ben Vanik. All rights reserved.
 * Original source released under the BSD license (see LICENSE).
 *
 * This header replaces the parts of xenia's base library that the ported
 * PPC decoder/disassembler depends on (xenia/base/assert.h,
 * xenia/base/string_buffer.h, xenia/base/math.h) with a minimal,
 * freestanding implementation so the code builds for wasm32-freestanding
 * with no libc, no heap, and no imports from the host.
 */

#ifndef XENIATEST_COMPAT_H_
#define XENIATEST_COMPAT_H_

#include <cstdint>
#include <cstddef>

// Freestanding string helpers (memcpy/memset/strlen are compiler builtins and
// lower to inline wasm; the rest we implement by hand so there is no libc).
static inline size_t xeniatest_strlen(const char* s) {
  const char* p = s;
  while (*p) p++;
  return (size_t)(p - s);
}
static inline const char* xeniatest_strchr(const char* s, char c) {
  if (!s) return nullptr;
  for (const char* p = s;; p++) {
    if (*p == c) return p;
    if (*p == '\0') return nullptr;
  }
}
static inline bool xeniatest_strstr(const char* haystack, const char* needle) {
  if (!haystack || !needle) return false;
  if (*needle == '\0') return true;
  for (const char* h = haystack; *h; h++) {
    const char* x = h;
    const char* n = needle;
    while (*x && *n && *x == *n) { x++; n++; }
    if (*n == '\0') return true;
  }
  return false;
}

// --- xenia/base/assert.h replacements ---------------------------------------
// The ported code only uses assert_always()/assert_null() as guards in paths
// that are unreachable for well-formed input. In the WASM core they are no-ops
// unless XENIATEST_DEBUG_ASSERTS is defined.
#if defined(XENIATEST_DEBUG_ASSERTS)
#define assert_always(...) __builtin_trap()
#define assert_null(...) __builtin_trap()
#else
#define assert_always(...) ((void)0)
#define assert_null(...) ((void)0)
#endif

#ifndef XE_MAYBE_UNUSED
#define XE_MAYBE_UNUSED __attribute__((unused))
#endif

// --- StringBuffer -----------------------------------------------------------
// Fixed-capacity drop-in for xenia/base/string_buffer.h implementing exactly
// what the ported disassembler uses. Disassembly lines for real PPC code are
// far shorter than the cap, and overflow is clamped instead of corrupting.
class StringBuffer {
 public:
  static constexpr size_t kCapacity = 512;

  StringBuffer() : len_(0) { buf_[0] = '\0'; }

  size_t length() const { return len_; }
  const char* c_str() const { return buf_; }
  void clear() { len_ = 0; buf_[0] = '\0'; }

  void Append(const char* str) {
    if (!str) return;
    AppendBytes(str, xeniatest_strlen(str));
  }
  void Append(char c) { AppendBytes(&c, 1); }
  void AppendBytes(const uint8_t* bytes, size_t len) { AppendBytes(reinterpret_cast<const char*>(bytes), len); }
  void AppendBytes(const char* bytes, size_t len) {
    if (len > kCapacity - len_) len = kCapacity - len_;
    if (len == 0) return;
    __builtin_memcpy(buf_ + len_, bytes, len);
    len_ += len;
    buf_[len_] = '\0';
  }

  // Supports the placeholder vocabulary used by the generated disassembler:
  //   {}      decimal (or raw string for string arguments)
  //   {:X}    uppercase hexadecimal
  //   {:<N}   decimal/string left-aligned, space padded to width N
  template <typename... Args>
  void AppendFormat(const char* format, const Args&... args) {
    FormatAppend(format, args...);
  }

 private:
  char buf_[kCapacity];
  size_t len_;

  void AppendN(char c, size_t count) {
    while (count-- > 0) Append(c);
  }

  void AppendPadded(const char* body, size_t body_len, int width) {
    AppendBytes(body, body_len);
    if ((int)body_len < width) AppendN(' ', (size_t)(width - (int)body_len));
  }

  // Integer rendering helpers (no libc).
  static void UintToDec(uint64_t value, char* out, size_t* out_len) {
    char tmp[24];
    size_t n = 0;
    if (value == 0) tmp[n++] = '0';
    while (value != 0) {
      tmp[n++] = (char)('0' + (value % 10));
      value /= 10;
    }
    for (size_t i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
    *out_len = n;
  }
  static void UintToHex(uint64_t value, char* out, size_t* out_len) {
    char tmp[24];
    size_t n = 0;
    if (value == 0) tmp[n++] = '0';
    while (value != 0) {
      uint32_t digit = (uint32_t)(value & 0xF);
      tmp[n++] = (char)(digit < 10 ? '0' + digit : 'A' + digit - 10);
      value >>= 4;
    }
    for (size_t i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
    *out_len = n;
  }
  static int LeftAlignWidth(const char* spec) {
    if (spec[0] != '<') return -1;
    int width = 0;
    for (const char* p = spec + 1; *p >= '0' && *p <= '9'; p++) width = width * 10 + (*p - '0');
    return width;
  }

  void AppendFormattedUint(const char* spec, uint64_t value) {
    char body[24];
    size_t body_len = 0;
    if (xeniatest_strstr(spec, "X")) {
      UintToHex(value, body, &body_len);
    } else {
      UintToDec(value, body, &body_len);
    }
    int width = LeftAlignWidth(spec);
    if (width >= 0) {
      AppendPadded(body, body_len, width);
      return;
    }
    AppendBytes(body, body_len);
  }

  // One overload per argument category the disassembler passes.
  void AppendValue(const char* spec, uint64_t v) { AppendFormattedUint(spec, v); }
  void AppendValue(const char* spec, int64_t v) {
    if (xeniatest_strstr(spec, "X")) {
      AppendFormattedUint(spec, (uint64_t)v);
      return;
    }
    if (v < 0) {
      Append('-');
      AppendFormattedUint(spec, (uint64_t)(-v));
    } else {
      AppendFormattedUint(spec, (uint64_t)v);
    }
  }
  void AppendValue(const char* spec, int32_t v) { AppendValue(spec, (int64_t)v); }
  void AppendValue(const char* spec, int16_t v) { AppendValue(spec, (int64_t)v); }
  void AppendValue(const char* spec, uint32_t v) { AppendValue(spec, (uint64_t)v); }
  void AppendValue(const char* spec, bool v) { Append(v ? "true" : "false"); }
  void AppendValue(const char* spec, const char* v) {
    const char* str = v ? v : "(null)";
    int width = LeftAlignWidth(spec);
    size_t len = xeniatest_strlen(str);
    if (width >= 0) {
      AppendPadded(str, len, width);
      return;
    }
    AppendBytes(str, len);
  }

  void FormatAppend(const char* fmt) { Append(fmt); }

  template <typename T, typename... Rest>
  void FormatAppend(const char* fmt, const T& arg, const Rest&... rest) {
    const char* placeholder = xeniatest_strchr(fmt, '{');
    if (!placeholder) {
      Append(fmt);
      return;
    }
    AppendBytes(fmt, (size_t)(placeholder - fmt));
    const char* end = xeniatest_strchr(placeholder, '}');
    if (!end) {
      Append(placeholder);
      return;
    }
    char spec[16];
    size_t spec_len = (size_t)(end - placeholder - 1);
    if (spec_len >= sizeof(spec)) spec_len = sizeof(spec) - 1;
    __builtin_memcpy(spec, placeholder + 1, spec_len);
    spec[spec_len] = '\0';
    AppendValue(spec, arg);
    FormatAppend(end + 1, rest...);
  }
};

#endif  // XENIATEST_COMPAT_H_
