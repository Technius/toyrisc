use num_enum::{IntoPrimitive, TryFromPrimitive};
use std::ops::{Index, IndexMut};

#[derive(Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Clone, Copy)]
#[repr(u16)]
#[allow(non_camel_case_types)]
pub enum Csr {
    // These U CSRs are the N extension--are they interesting enough to implement?

    // pub const USTATUS: usize = 0x000;
    // pub const UTVEC: usize = 0x005;
    // pub const USCRATCH: usize = 0x040;
    // pub const UEPC: usize = 0x041;
    // pub const UCAUSE: usize = 0x042;
    // pub const UTVAL: usize = 0x043;

    SSTATUS = 0x100,
    STVEC = 0x105,
    SSCRATCH = 0x140,
    SEPC = 0x141,
    SCAUSE = 0x142,
    STVAL = 0x143,
    SATP = 0x180,

    MSTATUS = 0x300,
    MIE = 0x304,
    MTVEC = 0x305,
    MSCRATCH = 0x340,
    MEPC = 0x341,
    MCAUSE = 0x342,
    MTVAL = 0x343,

    /// Write a 1 to this CSR to halt the machine
    CUSTOM_HALT = 0x7C0,
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum Access {
    ReadWrite,
    ReadOnly,
}

impl From<u16> for Access {
    fn from(a: u16) -> Self {
        match a {
            0b00 | 0b01 | 0b10 => Access::ReadWrite,
            0b11 => Access::ReadOnly,
            _ => panic!("invalid access level"),
        }
    }
}

impl Csr {
    pub fn access(&self) -> Access {
        Access::from((*self as u16) >> 10)
    }
}

pub const CSR_COUNT: usize = 4096;

pub struct CsrFile {
    registers: [u64; CSR_COUNT]
}

impl CsrFile {
    pub fn new() -> Self {
        CsrFile {
            registers: [0; CSR_COUNT]
        }
    }
}

impl Index<Csr> for CsrFile {
    type Output = u64;
    fn index(&self, csr: Csr) -> &Self::Output {
        &self.registers[csr as usize]
    }
}

impl IndexMut<Csr> for CsrFile {
    fn index_mut(&mut self, csr: Csr) -> &mut Self::Output {
        &mut self.registers[csr as usize]
    }
}

pub mod mask {
    use std::ops::Range;

    pub const MSTATUS_MPP: Range<usize> = 11..13;
}
