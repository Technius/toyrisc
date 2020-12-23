.text
.globl _start
_start:
# setup machine trap vector
la t0, mtvec
csrw mtvec, t0

addi t0, zero, 1
ecall

mtvec:
# halt with custom csr
addi t1, zero, 1337
csrwi 0x730, 1
