MEMORY {
    bootrom (rx) : ORIGIN = 0x0, LENGTH = 4K
    ram (rwx) : ORIGIN = 0x08000000, LENGTH = 32K
}

STACK_SIZE = 0x4000; /* 4 kB */

ENTRY(_start);

SECTIONS {
    .text.loader : {
        _start = .;
        *(.text.loader)
    } > bootrom

    .rodata : {
        *(.rodata*)
    } > bootrom

    /* TODO: provide an elf loader in the simulator so it can relocate the following sections */

    .text : {
        _stext = .;
        *(.text)
        _etext = .;
    } > bootrom /* ram */

    .data : {
        *(.data*)
        _edata = .;
    } /* > ram AT */ > bootrom

    .stack (NOLOAD) : {
        _stack_start = .;
        . = ALIGN(8);
        . = . + STACK_SIZE;
        . = ALIGN(8);
    } > ram
}
