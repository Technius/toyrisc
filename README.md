# ToyRISC

This is a toy RISC-V simulator.

# Notes

* Only way to change privilege mode is through `ECALL` and `MRET`/`SRET`.
* `MRET`/`SRET` will change the privilege mode to the one in `MSTATUS.MPP`
* Traps switch machine to M-mode unless `mdeleg` is set.
* "privilege mode stack"?
* https://cdn2.hubspot.net/hubfs/3020607/An%20Introduction%20to%20the%20RISC-V%20Architecture.pdf
