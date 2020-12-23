.section .text.loader
.globl _start
.globl _stack_start # is this really needed?
.globl _stop
_start:
# setup machine trap vector
la t0, _exec_handler
csrw mtvec, t0
# stack pointer
la sp, _stack_start

la t0, main
la ra, _stop
csrw mepc, t0
mret # jumps in user mode since mstatus.mpp is 0

# this should be unreachable, so halt
j _stop

_exec_handler:
csrrw sp, mscratch, zero
# save every register on to the stack
addi sp, sp, 30*8
sd ra,  (8* 0)(sp)
sd gp,  (8* 1)(sp)
sd tp,  (8* 2)(sp)
sd t0,  (8* 3)(sp)
sd t1,  (8* 4)(sp)
sd t2,  (8* 5)(sp)
sd s0,  (8* 6)(sp)
sd s1,  (8* 7)(sp)
sd a0,  (8* 8)(sp)
sd a1,  (8* 9)(sp)
sd a2,  (8*10)(sp)
sd a3,  (8*11)(sp)
sd a4,  (8*12)(sp)
sd a5,  (8*13)(sp)
sd a6,  (8*14)(sp)
sd a7,  (8*15)(sp)
sd s2,  (8*16)(sp)
sd s3,  (8*17)(sp)
sd s4,  (8*18)(sp)
sd s5,  (8*19)(sp)
sd s6,  (8*20)(sp)
sd s7,  (8*21)(sp)
sd s8,  (8*22)(sp)
sd s9,  (8*23)(sp)
sd s10, (8*24)(sp)
sd s11, (8*25)(sp)
sd t3,  (8*26)(sp)
sd t4,  (8*27)(sp)
sd t5,  (8*28)(sp)
sd t6,  (8*29)(sp)

csrr a0, mcause
addi a1, sp, 8*8 # pointer to original a0
la t0, _handle_exception
jalr t0

# restore the stack
ld ra,  (8* 0)(sp)
ld gp,  (8* 1)(sp)
ld tp,  (8* 2)(sp)
ld t0,  (8* 3)(sp)
ld t1,  (8* 4)(sp)
ld t2,  (8* 5)(sp)
ld s0,  (8* 6)(sp)
ld s1,  (8* 7)(sp)
ld a0,  (8* 8)(sp)
ld a1,  (8* 9)(sp)
ld a2,  (8*10)(sp)
ld a3,  (8*11)(sp)
ld a4,  (8*12)(sp)
ld a5,  (8*13)(sp)
ld a6,  (8*14)(sp)
ld a7,  (8*15)(sp)
ld s2,  (8*16)(sp)
ld s3,  (8*17)(sp)
ld s4,  (8*18)(sp)
ld s5,  (8*19)(sp)
ld s6,  (8*20)(sp)
ld s7,  (8*21)(sp)
ld s8,  (8*22)(sp)
ld s9,  (8*23)(sp)
ld s10, (8*24)(sp)
ld s11, (8*25)(sp)
ld t3,  (8*26)(sp)
ld t4,  (8*27)(sp)
ld t5,  (8*28)(sp)
ld t6,  (8*29)(sp)
addi sp, sp, 30*8

# TODO: return to whatever privilege mode was used to enter
mret

_stop:
csrwi 0x7c0, 1
