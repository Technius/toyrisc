TARGET = riscv64-unknown-linux-gnu
CC = $(TARGET)-gcc
OBJDUMP = $(TARGET)-objdump
OBJCOPY = $(TARGET)-objcopy

CFLAGS += -nostdlib -ffreestanding -march=rv64id # -Ttext=0

OBJECTS = foo.o halt.o hello.o loader.o
BINARIES = hello syscall try_kernel

BINDIR = bin

# all: $(BINARIES:%:$(BINDIR)/%)

# riscv64-unknown-linux-gnu-objcopy -O binary -j .text bin/hello bin/hello.bin
# riscv64-unknown-linux-gnu-objdump -d -M 'no-aliases,numeric' bin/hello

$(BINDIR)/hello.bin: $(BINDIR)/hello
	$(OBJCOPY) -O binary -j .text.loader -j .rodata -j .text -j .data $< $@

$(BINDIR)/hello: $(BINDIR)/loader.o $(BINDIR)/runtime.o hello.c
	$(CC) $(CFLAGS) -T loader.lds $^ -o $@

# How to compile with musl???
$(BINDIR)/try_kernel: try_kernel.c
	$(CC) $^ -o $@

$(BINDIR)/syscall: $(BINDIR)/syscall.o
	$(CC) $(CFLAGS) $^ -o $@

$(BINDIR)/loader.o: loader.S
	$(CC) $(CFLAGS) -c $^ -o $@

$(BINDIR)/halt.o: halt.S
	$(CC) $(CFLAGS) -c $^ -o $@

$(BINDIR)/runtime.o: runtime.c runtime.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BINDIR)/%.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

.PHONY: clean
clean:
	@rm -f $(BINDIR)/*.o
	@rm -f $(BINDIR)/$(BINARIES) $(BINDIR)/*.o $(BINDIR)/*.bin
