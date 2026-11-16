/**
 * programs.js — built-in demo payloads for the web runner.
 *
 * Each program is real Xenon PPC64 (big-endian) assembly assembled at load
 * time by js/ppc_asm.js and recompiled to WebAssembly on click. They are
 * the web equivalent of xenia-canary's homebrew test harnesses: they prove
 * the assembler, decoder, lifter and MMIO console path end to end in a browser.
 */

const PREAMBLE = `
# ---- Runtime helpers shared by every built-in program ---------------------
# Exit door: any store covering 0x00F00FFC halts the core.
exit_door:
    li    r9, 0
    lis   r10, 0x00f0
    ori   r10, r10, 0x0ffc
    stb   r9, 0(r10)
    b     exit_door        # in case the store faults, spin forever

print_str:                  # r3 = pointer to NUL-terminated string
    lis   r6, 0x00f0
ps_loop:
    lbz   r5, 0(r3)
    cmpwi r5, 0
    beq   ps_done
    stb   r5, 0(r6)
    addi  r3, r3, 1
    b     ps_loop
ps_done:
    blr

print_u64:                  # r3 = value (prints decimal, no leading zeros)
    lis   r10, digits@ha
    addi  r10, r10, digits@l
    addi  r10, r10, 21
    li    r9, 0
    stb   r9, 0(r10)         # NUL terminator at buffer end
    li    r12, 10
div_loop:
    divwu r13, r3, r12
    mullw r14, r13, r12
    subf  r15, r14, r3       # r15 = r3 - (r13 * 10) = digit
    addi  r15, r15, 48       # '0'
    addi  r10, r10, -1
    stb   r15, 0(r10)
    mr    r3, r13
    cmpwi r3, 0
    bne   div_loop
    mr    r3, r10
    b     print_str          # tail call: prints from r10 (NUL terminated)

newline:
    lis   r3, nl_str@ha
    addi  r3, r3, nl_str@l
    b     print_str          # tail call

nl_str: .asciiz  "\\n"
digits: .space 24
`;

const HELLO_SOURCE = `
.org 0x10000
_start:
    lis   r3, msg@ha
    addi  r3, r3, msg@l
    bl    print_str
    bl    newline
    b     exit_door
msg: .asciiz "Hello from Xenon PPC, assembled and executed inside your browser!"
` + PREAMBLE;

const FIB_SOURCE = `
.org 0x10000
_start:
    lis   r3, msg@ha
    addi  r3, r3, msg@l
    bl    print_str
    li    r3, 0
    li    r4, 1
    li    r7, 25             # fib(25)
fib_loop:
    add   r5, r3, r4
    mr    r3, r4
    mr    r4, r5
    addi  r7, r7, -1
    cmpwi r7, 0
    bne   fib_loop
    bl    print_u64          # r3 = fib(25) = 75025
    bl    newline
    b     exit_door
msg: .asciiz "fib(25) = "
` + PREAMBLE;

const COUNTDOWN_SOURCE = `
.org 0x10000
_start:
    lis   r3, msg@ha
    addi  r3, r3, msg@l
    bl    print_str
    li    r7, 10
cd_loop:
    mr    r3, r7
    bl    print_u64
    bl    newline
    addi  r7, r7, -1
    cmpwi r7, 0
    bne   cd_loop
    lis   r3, bye@ha
    addi  r3, r3, bye@l
    bl    print_str
    b     exit_door
msg: .asciiz "Launch sequence initiated. Counting down:\\n"
bye: .asciiz "Liftoff!\\n"
` + PREAMBLE;

export const BUILTIN_PROGRAMS = [
  {
    id: 'builtin:hello',
    name: 'Hello from Xenon PPC',
    description: 'The Xenon PPC64 porting-layer smoke test: prints a string through the guest MMIO console, then halts via the exit door.',
    kind: 'builtin',
    source: HELLO_SOURCE,
    entry: 0x10000,
  },
  {
    id: 'builtin:fib',
    name: 'Fibonacci(25) — integer ALU',
    description: 'Iterative Fibonacci computed in registers. Exercises add, mr, addi, cmpwi, bne, divwu, mullw and the decimal digit printer.',
    kind: 'builtin',
    source: FIB_SOURCE,
    entry: 0x10000,
  },
  {
    id: 'builtin:countdown',
    name: 'Launch countdown — loops & calls',
    description: 'Prints 10..1 with a bl/blr subroutine call and a real loop. Validates branch condition handling (cmpwi + bne) on the web core.',
    kind: 'builtin',
    source: COUNTDOWN_SOURCE,
    entry: 0x10000,
  },
];
