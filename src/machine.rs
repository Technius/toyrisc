use crate::rv32i::*;
use crate::vmem::SparseMem;
use crate::mmio::MmioDevice;
use crate::csr::{Csr, CsrFile};
use std::convert::{TryInto, TryFrom};
use bitvec::bitvec;
use bitvec::fields::BitField;
use num_enum::TryFromPrimitive;

pub const X_REGISTER_COUNT: usize = 32;
pub const MEMORY_SIZE: usize = 1024 * 1024; // 1000 kb
pub const PAGE_SIZE: usize = 4096;

/// Address to load boot ROM from
pub const BOOT_ADDR: usize = 0x0;

/// Program counter register
const PC: usize = 31;

pub struct Machine {
    x_register: [u64; X_REGISTER_COUNT],
    halt: bool,
    pub memory: SparseMem,
    csr_space: CsrFile,
    mode: Priv,
    mmio_devices: Vec<MmioDevice>,
}

/// Privilege modes
#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone, Copy, TryFromPrimitive)]
#[repr(u8)]
pub enum Priv {
    U = 0,
    S = 1,
    M = 3,
}

type EResult<T> = Result<T, ()>;

fn check_register(register: u8) -> EResult<usize> {
    register.try_into().map_err(|_| ())
}

fn as_signed(x: u64) -> i64 {
    unsafe { std::mem::transmute(x) }
}

fn as_unsigned(x: i64) -> u64 {
    unsafe { std::mem::transmute(x) }
}

fn sext12(x: u64) -> u64 {
    return sext_at(x, 11);
}

/// Sign extend the `bit` bit
fn sext_at(x: u64, bit: usize) -> u64 {
    let shamt = 64 - bit - 1;
    return as_unsigned((as_signed(x as u64) << shamt) >> shamt);
}

impl Machine {
    pub fn new() -> Self {
        Machine {
            x_register: [0; X_REGISTER_COUNT],
            halt: false,
            memory: SparseMem::new(PAGE_SIZE),
            csr_space: CsrFile::new(),
            mode: Priv::M,
            mmio_devices: Vec::new(),
        }
    }

    pub fn run(&mut self) {
        for ref mut device in &mut self.mmio_devices {
            (*device.handler).mmio_reset();
        }

        self.x_register[PC] = BOOT_ADDR as u64;
        let mut instr_count = 0;
        while !self.halt {
            let pc = self.x_register[PC];
            let instr_word = self.load_bytes(pc, 0, 4) as u32;
            if let Some(decoded_instr) = DInstr::decode(instr_word) {
                eprintln!("Executing {:08x}: {:?}", pc, decoded_instr);
                self.execute(&decoded_instr);
            } else {
                println!("Failed to decode at pc = {:08x}: {:08x}", pc, instr_word);
                self.halt = true;
            }

            if pc > 0x1000 {
                println!("--DEBUG-- PC is large, halting");
                self.halt = true;
            }

            if self.csr_space[Csr::CUSTOM_HALT] == 0x1 {
                println!("Halt set in CSR");
                self.halt = true;
            }

            instr_count += 1;

            if instr_count > 300 {
                println!("--DEBUG-- too many instructions, halting");
                self.halt = true;
            }
        }
    }

    pub fn execute(&mut self, instr: &DInstr) {
        let op = &instr.op;
        let status = match instr.contents {
            c if op.opcode() == Opcode::System as u8 => self.execute_system(op, c),
            Instr::R(instr) => self.execute_r(op, instr),
            Instr::I(instr) => self.execute_i(op, instr),
            Instr::S(instr) => self.execute_s(op, instr),
            Instr::U(instr) => self.execute_u(op, instr),
            Instr::B(instr) => self.execute_b(op, instr),
        };

        if status.is_err() {
            self.halt = true;
        }
    }

    fn set_reg(&mut self, reg: u8, value: u64) -> EResult<()> {
        if reg > 0 {
            self.x_register[check_register(reg)?] = value;
        }
        Ok(())
    }

    fn execute_r(&mut self, op: &Op, instr: instr::R) -> EResult<()> {
        let v1 = self.x_register[check_register(instr.rs1)?];
        let v2 = self.x_register[check_register(instr.rs2)?];
        let result = match op {
            Op::Add => as_unsigned(as_signed(v1) + as_signed(v2)),
            Op::Sub => as_unsigned(as_signed(v1) - as_signed(v2)),
            _ => unreachable!(),
        };
        self.set_reg(instr.rd, result)?;
        self.x_register[PC] += 4;
        Ok(())
    }

    pub fn load_bytes(&mut self, base: u64, offs: u64, bytes: usize) -> u64 {
        let base_addr = as_unsigned(as_signed(base) + as_signed(sext12(offs)));
        let mut val = 0;
        if let Some(ref mut mmio) = self.get_mmio_device(base_addr as u64) {
            let offset = base_addr - mmio.addr_range.start;
            val = (*mmio.handler).mmio_read(offset, bytes as u64);
        } else {
            for i in 0..bytes {
                // little endian!
                val = ((self.memory[(base_addr as usize) + i] as u64) << (8 * i)) | val;
            }
        }
        val
    }

    pub fn store_bytes(&mut self, base: u64, offs: u64, bytes: usize, value: u64) {
        let base_addr = as_unsigned(as_signed(base) + as_signed(offs));
        if let Some(ref mut mmio) = self.get_mmio_device(base_addr as u64) {
            let offset = base_addr - mmio.addr_range.start;
            (*mmio.handler).mmio_write(offset, bytes as u64, value);
        } else {
            for i in 0..bytes {
                self.memory[(base_addr as usize) + i] = ((value >> (i * 8)) & 0xFF) as u8;
            }
        }
    }

    /// Writes N words to memory, starting at address `addr` and going to `addr
    /// + N * 4` in ascending order. Bytes within a word are written in
    /// little-endian order.
    pub fn store_words(&mut self, addr: usize, words: &[u64]) {
        for offs in 0..words.len() {
            let value = words[offs];
            for i in 0..4 {
                self.memory[addr + (offs * 4) + i] = ((value >> (i * 8)) & 0xFF) as u8;
            }
        }
    }

    fn execute_i(&mut self, op: &Op, instr: instr::I) -> EResult<()> {
        let v1 = self.x_register[check_register(instr.rs1)?];
        let imm = instr.imm as u64;
        let result = match op {
            Op::Addi => Some(as_unsigned(as_signed(v1) + as_signed(sext12(imm)))),
            Op::Slti => Some((as_signed(v1) < as_signed(sext12(imm))) as u64),
            Op::Sltiu => Some((v1 < (sext12(imm))) as u64),
            Op::Xori => Some(v1 ^ sext12(imm)),
            Op::Andi => Some(v1 & sext12(imm)),
            Op::Ori => Some(v1 | sext12(instr.imm as u64)),
            Op::Slli => Some(v1 << (imm & 0x1F)),
            Op::Srli => Some(v1 >> (imm & 0x1F)),
            Op::Srai => Some(as_unsigned(as_signed(v1) >> (imm & 0x1F))),
            Op::Lb => Some(sext_at(self.load_bytes(v1, imm, 1), 7)),
            Op::Lh => Some(sext_at(self.load_bytes(v1, imm, 2), 15)),
            Op::Lw => Some(sext_at(self.load_bytes(v1, imm, 4), 31)),
            Op::Lbu => Some(self.load_bytes(v1, imm, 1)),
            Op::Lhu => Some(self.load_bytes(v1, imm, 2)),
            Op::Lwu => Some(self.load_bytes(v1, imm, 4)),
            Op::Ld => Some(self.load_bytes(v1, imm, 8)),
            Op::Jalr => {
                let offs = sext12(instr.imm as u64);
                self.set_reg(instr.rd, self.x_register[PC] + 4)?;
                self.x_register[PC] = as_unsigned(as_signed(v1) + as_signed(offs));
                return Ok(());
            },
            _ => unreachable!(),
        };
        if let Some(result) = result {
            self.set_reg(instr.rd, result)?;
        }
        self.x_register[PC] += 4;
        Ok(())
    }

    fn execute_s(&mut self, op: &Op, instr: instr::S) -> EResult<()> {
        let base = self.x_register[check_register(instr.rs1)?];
        let offs = sext12(instr.imm as u64);
        let v = self.x_register[check_register(instr.rs2)?];

        let bytes: usize = match op {
            Op::Sb => 1,
            Op::Sh => 2,
            Op::Sw => 4,
            Op::Sd => 8,
            _ => unreachable!(),
        };

        self.store_bytes(base, offs, bytes, v);
        self.x_register[PC] += 4;
        Ok(())
    }

    fn execute_u(&mut self, op: &Op, instr: instr::U) -> EResult<()> {
        let result = match op {
            Op::Lui => sext_at(((instr.imm & 0xFFFFF) as u64) << 12, 31),
            Op::Auipc => {
                let offs = sext_at(((instr.imm & 0xFFFFF) as u64) << 12, 31);
                as_unsigned(as_signed(self.x_register[PC]) + as_signed(offs))
            },
            Op::Jal => {
                let imm = instr.imm as u64;
                let imm = ((imm & 0x80000) << 1)
                    | (((imm >> 9) & 0x3FF) << 1)
                    | (((imm >> 8) & 1) << 11)
                    | ((imm & 0xFF) << 12);
                let offs = sext_at(imm, 21);
                self.set_reg(instr.rd, self.x_register[PC] + 4)?;
                self.x_register[PC] = as_unsigned(as_signed(self.x_register[PC]) + as_signed(offs));
                return Ok(())
            },
            _ => unreachable!(),
        };
        self.set_reg(instr.rd, result)?;
        self.x_register[PC] += 4;
        Ok(())
    }

    fn execute_b(&mut self, op: &Op, instr: instr::B) -> EResult<()> {
        let v1 = self.x_register[check_register(instr.rs1)?];
        let v2 = self.x_register[check_register(instr.rs2)?];
        let branch = match op {
            Op::Beq => v1 == v2,
            Op::Bne => v1 != v2,
            Op::Blt => as_signed(v1) < as_signed(v2),
            Op::Bge => as_signed(v1) >= as_signed(v2),
            Op::Bltu => v1 < v2,
            Op::Bgeu => v1 >= v2,
            _ => unreachable!(),
        };
        if branch {
            let offset = sext_at(instr.imm as u64, 12);
            self.x_register[PC] = as_unsigned(as_signed(self.x_register[PC]) + as_signed(offset));
        } else {
            self.x_register[PC] += 4;
        }
        Ok(())
    }

    fn get_csr(&self, csr: u16) -> EResult<u64> {
        if let Ok(csr) = Csr::try_from(csr) {
            Ok(self.csr_space[csr])
        } else {
            Err(())
        }
    }

    fn set_csr(&mut self, csr: u16, value: u64) -> EResult<()> {
        if let Ok(csr) = Csr::try_from(csr) {
            self.csr_space[csr] = value;
        }
        Ok(())
    }

    fn execute_system(&mut self, op: &Op, instr: Instr) -> EResult<()> {
        match op {
            Op::Ecall => {
                let handler_addr = self.csr_space[Csr::MTVEC];
                // TODO: vectored mode

                // TODO: this is wrong; should get the ones corresponding to the
                // entered privilege level
                let (epc, cause, cause_val, tval) = match self.mode {
                    Priv::M => (Csr::MEPC, Csr::MCAUSE, 11, Csr::MTVAL),
                    Priv::S => (Csr::SEPC, Csr::SCAUSE, 9, Csr::STVAL),
                    Priv::U => unimplemented!(), // (Csr::UEPC, Csr::UCAUSE, 8, Csr::UTVAL),
                };
                // TODO: check if correct bits are set for S-mode
                self.csr_space[epc] = self.x_register[PC];
                self.csr_space[cause] = cause_val;
                self.csr_space[tval] = 0;

                // u32 is a workaround for myrrlyn/bitvec#58
                let mut status = bitvec![bitvec::order::Msb0, u32; 0; 64];
                status[crate::csr::mask::MSTATUS_MPP].store(self.mode as u8);
                self.csr_space[Csr::MSTATUS] = status.load_be();
                self.mode = Priv::M;

                self.x_register[PC] = handler_addr;
            }
            Op::Mret => {
                self.x_register[PC] = self.csr_space[Csr::MEPC];
                // TODO: write a test for this
                let mut status = bitvec![bitvec::order::Msb0, u32; 0; 64];
                status.store_be(self.csr_space[Csr::MSTATUS]);
                if let Ok(mode) = Priv::try_from(status[crate::csr::mask::MSTATUS_MPP].load::<u8>()) {
                    self.mode = mode;
                } else {
                    panic!("Invalid mode");
                }
            }
            Op::Csrrw | Op::Csrrwi => {
                let instr = instr.as_i().unwrap();
                let csr_w = if op == &Op::Csrrw {
                    self.x_register[check_register(instr.rs1)?]
                } else {
                    instr.rs1 as u64
                };
                if instr.rd > 0 {
                    self.x_register[instr.rd as usize] = self.get_csr(instr.imm)?;
                }
                self.set_csr(instr.imm, csr_w)?;
                self.x_register[PC] += 4;
            }
            Op::Csrrs | Op::Csrrc | Op::Csrrsi | Op::Csrrci => {
                let instr = instr.as_i().unwrap();
                let mut csr_value = self.get_csr(instr.imm)?;
                self.x_register[instr.rd as usize] = csr_value;
                if instr.rs1 > 0 {
                    let mask = if op == &Op::Csrrs || op == &Op::Csrrc {
                        self.x_register[check_register(instr.rs1)?]
                    } else {
                        instr.rs1 as u64
                    };
                    if op == &Op::Csrrs || op == &Op::Csrrsi {
                        csr_value |= mask;
                    } else {
                        csr_value &= !mask;
                    }
                    self.set_csr(instr.imm, csr_value)?;
                }
                self.x_register[PC] += 4;
            }
            _ => return Err(()),
        }
        Ok(())
    }

    pub fn get_registers(&self) -> &[u64] {
        &self.x_register
    }

    pub fn is_halted(&self) -> bool {
        self.halt
    }

    pub fn register_mmio(&mut self, device: MmioDevice) {
        let no_overlap = self.mmio_devices.iter().all(|ref d| {
            (device.addr_range.end <= d.addr_range.start) || (d.addr_range.end <= d.addr_range.start)
        });

        if no_overlap {
            self.mmio_devices.push(device);
        } else {
            panic!("mmio device address range overlaps with that of existing device");
        }
    }

    fn get_mmio_device(&mut self, addr: u64) -> Option<&mut MmioDevice> {
        self.mmio_devices.iter_mut().find(|ref d| d.addr_range.contains(&addr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_halt() {
        let mut machine = Machine::new();
        machine.store_words(
            0,
            vec![
                0x7c00d073, // csrwi 0x7c0, 1
                0x53900313, // addi t1, zero, 1337
            ]
            .as_slice(),
        );
        machine.run();
        assert_eq!(1, machine.csr_space[Csr::CUSTOM_HALT]);
        assert_eq!(0x4, machine.x_register[PC]);
        assert_eq!(0, machine.x_register[6]);
    }

    #[test]
    fn syscall_and_halt() {
        let mut machine = Machine::new();
        machine.store_words(
            0,
            vec![
                0x00000297, 0x01428293, 0x30529073, 0x00100293, 0x00000073, 0x53900313, 0x7300d073,
            ]
            .as_slice(),
        );
        machine.run();
        assert_eq!(0x14, machine.csr_space[Csr::MTVEC]);
        assert_eq!(1337, machine.x_register[6]);
        assert_eq!(0x1C, machine.x_register[PC]);
    }

    #[test]
    fn from_file() {
        println!("{}", std::env::current_dir().unwrap().to_str().unwrap());
        let program = {
            use std::io::Read;
            let f = std::fs::File::open("tests/crosscompile/bin/hello.bin").unwrap();
            let mut p = Vec::new();
            let mut buf_reader = std::io::BufReader::new(f);
            buf_reader.read_to_end(&mut p).unwrap();
            p
        };

        let mut machine = Machine::new();
        for i in 0..program.len() {
            machine.memory[i] = program[i];
        }
        machine.run();
        assert_eq!(5, machine.x_register[10]);
    }
}
