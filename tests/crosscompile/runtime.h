#ifndef RUNTIME_H
#define RUNTIME_H

#include <stdint.h>

#define SYS_HALT 0

int64_t _handle_exception(int64_t cause, void *args[8]);

int64_t syscall(uint64_t code, void *args[7]);

#endif
