#include "runtime.h"

static int64_t sys_halt(uint64_t status) {
    __asm__("csrwi 0x7c0, 1");
    return 0;
}

int64_t syscall(uint64_t code, void *args[7]) {
    __asm__("ecall");
    int64_t retval;
    __asm__("addi %0, a0, 0" : "=r"(retval));
    return retval;
}

int64_t _handle_exception(int64_t cause, void *args[8]) {
    if (cause < 0) {
        // interrupt
    } else {
        // trap
        switch (cause) {
        case 11:
            // environment call from M-mode
            switch ((uint64_t)args[0]) {
            case SYS_HALT: return sys_halt((uint64_t)args[1]);
            }
            break;
        default: /* nop */;
        }
    }

    return -1;
}
