#include <stdint.h>

int main(void) {
    const char message[] = "hello world\n";
    char *dev = (char*) 0x10000000;
    for (uint64_t i = 0; message[i] != 0; i++) {
        *dev = message[i];
    }
    return 5;
}
