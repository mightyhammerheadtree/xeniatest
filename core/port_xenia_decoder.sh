#!/usr/bin/env bash
#
# Ports the PPC instruction decoder and disassembler from a xenia-canary
# checkout into core/src/xenia/, keeping the original copyright headers and
# license notices (BSD - see LICENSE and THIRD_PARTY_NOTICES.md).
#
# Usage: ./core/port_xenia_decoder.sh /path/to/xenia-canary
set -euo pipefail

SRC_ROOT="${1:?usage: port_xenia_decoder.sh /path/to/xenia-canary}"
SRC="$SRC_ROOT/src/xenia/cpu/ppc"
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src/xenia"

if [ ! -f "$SRC/ppc_opcode.h" ]; then
  echo "error: $SRC/ppc_opcode.h not found - is $SRC_ROOT a xenia-canary checkout?" >&2
  exit 1
fi

mkdir -p "$DST"

copy() {  # copy <file> [sed-expr...]
  local file="$1"; shift
  cp "$SRC/$file" "$DST/$file"
  if [ "$#" -gt 0 ]; then
    local args=()
    for expr in "$@"; do args+=(-E -e "$expr"); done
    sed -i "${args[@]}" "$DST/$file"
  fi
}

COMMON_INCLUDE_SED=(
  's#"xenia/base/assert.h"#"../compat.h"#g'
  's#"xenia/base/string_buffer.h"#"../compat.h"#g'
  's#"xenia/base/math.h"#"../compat.h"#g'
  's#"xenia/cpu/ppc/#"#g'
)

# 1. ppc_opcode.h - pure enum, verbatim.
copy ppc_opcode.h

# 2. ppc_instr.h - decode data bit-field unions; drop <string>/<vector>.
copy ppc_instr.h "${COMMON_INCLUDE_SED[@]}"
python3 - "$DST/ppc_instr.h" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
src = src.replace("#include <string>\n#include <vector>\n", "")
open(path, "w").write(src)
EOF

# 3. ppc_decode_data.h - include list only.
copy ppc_decode_data.h "${COMMON_INCLUDE_SED[@]}"

# 4. ppc_opcode_info.h - keep disasm-related declarations, drop the JIT emitter
#    plumbing (xenia's JIT uses HIR; the web core is an interpreter).
copy ppc_opcode_info.h "${COMMON_INCLUDE_SED[@]}"
python3 - "$DST/ppc_opcode_info.h" <<'EOF'
import re, sys
path = sys.argv[1]
src = open(path).read()
src = src.replace("""struct InstrData;
class PPCHIRBuilder;

""", "")
src = src.replace("""typedef int (*InstrEmitFn)(PPCHIRBuilder& f, const InstrData& i);

""", "")
src = src.replace("""struct PPCOpcodeInfo {
  PPCOpcodeGroup group;
  PPCOpcodeType type;
  InstrEmitFn emit;
};""", """// Ported for the web core: the JIT emitter pointer was dropped. The
// interpreter dispatches on the opcode enum directly.
struct PPCOpcodeInfo {
  PPCOpcodeGroup group;
  PPCOpcodeType type;
};""")
src = src.replace("""void RegisterOpcodeEmitter(PPCOpcode opcode, InstrEmitFn fn);
""", "")
open(path, "w").write(src)
EOF

# 5. ppc_opcode_table_gen.cc - generated opcode table; emitter pointer removed.
copy ppc_opcode_table_gen.cc "${COMMON_INCLUDE_SED[@]}"
python3 - "$DST/ppc_opcode_table_gen.cc" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
src = src.replace("""#define INSTRUCTION(opcode, mnem, form, group, type) \\
    {PPCOpcodeGroup::group, PPCOpcodeType::type, nullptr}""",
"""#define INSTRUCTION(opcode, mnem, form, group, type) \\
    {PPCOpcodeGroup::group, PPCOpcodeType::type}""")
# Drop RegisterOpcodeEmitter definition (assert_null is unused elsewhere).
src = src.replace("""void RegisterOpcodeEmitter(PPCOpcode opcode, InstrEmitFn fn) {
  assert_null(ppc_opcode_table[static_cast<int>(opcode)].emit);
  ppc_opcode_table[static_cast<int>(opcode)].emit = fn;
}
""", "")
open(path, "w").write(src)
EOF

# 6. ppc_opcode_lookup_gen.cc - generated decoder; include list only.
copy ppc_opcode_lookup_gen.cc "${COMMON_INCLUDE_SED[@]}"

# 7. ppc_opcode_disasm.h - include list only.
copy ppc_opcode_disasm.h "${COMMON_INCLUDE_SED[@]}"

# 8. ppc_opcode_disasm.cc - include list only.
copy ppc_opcode_disasm.cc "${COMMON_INCLUDE_SED[@]}"

# 9. ppc_opcode_disasm_gen.cc - generated disassembler; include list only.
copy ppc_opcode_disasm_gen.cc "${COMMON_INCLUDE_SED[@]}"
python3 - "$DST/ppc_opcode_disasm_gen.cc" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
src = src.replace("""void RegisterOpcodeDisasm(PPCOpcode opcode, InstrDisasmFn fn) {
  assert_null(ppc_opcode_disasm_table[static_cast<int>(opcode)].disasm);
  ppc_opcode_disasm_table[static_cast<int>(opcode)].disasm = fn;
}""", """void RegisterOpcodeDisasm(PPCOpcode opcode, InstrDisasmFn fn) {
  (void)opcode;
  (void)fn;
}""")
open(path, "w").write(src)
EOF

# 10. ppc_opcode_info.cc - DisasmPPC entry point; include list only.
copy ppc_opcode_info.cc "${COMMON_INCLUDE_SED[@]}"

echo "Ported xenia-canary PPC decoder/disassembler into $DST"
