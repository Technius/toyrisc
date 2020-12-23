.globl _start
.text
_start:
csrwi 0x7C0, 1
li t1, 1337
