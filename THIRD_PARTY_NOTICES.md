Xenon Recompiler — third-party notices
=======================================

This project is a derivative work of xenia-canary (https://github.com/xenia-canary/xenia-canary),
which is licensed under the BSD 3-Clause license. The full license text is reproduced in LICENSE.

Copyright (c) 2015, Ben Vanik. All rights reserved.

Files derived from xenia-canary (see core/src/xenia/ and core/src/compat.h):
  - core/src/xenia/ppc_opcode.h, ppc_opcode_info.h, ppc_opcode_info.cc
  - core/src/xenia/ppc_decode_data.h, ppc_instr.h
  - core/src/xenia/ppc_opcode_table_gen.cc, ppc_opcode_lookup_gen.cc, ppc_opcode_disasm_gen.cc (auto-generated tables)
  - core/src/xenia/ppc_opcode_disasm.{h,cc}, ppc_opcode_lookup_gen.cc
  - Decoder and disassembler semantics follow xenia-canary's cpu/ppc implementation.
  - core/src/xenon_cpu.cc is a new implementation written for this project, using xenia-canary as a reference.

The xenia-canary project is the work of Ben Vanik and the xenia contributors.

"Xbox 360", "Xenon", "Xenos" and "Xbox" are trademarks of Microsoft Corporation.
This project distributes no Microsoft code, no game data, and no system files.
