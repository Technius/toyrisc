use num_enum::TryFromPrimitive;
use std::convert::TryFrom;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
#[repr(u8)]
pub enum Op {
    Lui, Auipc,
    // Arithmetic I-type instructions
    Addi, Slti, Sltiu, Xori, Ori, Andi,
    Slli, Srli, Srai,
    // Arithmetic R-type instructions
    Add, Sub,
    // Control transfer
    Jal, Jalr,
    Beq, Bne, Blt, Bge, Bltu, Bgeu,
    // Memory
    Lb, Lh, Lw, Lbu, Lhu, Lwu, Ld,
    Sb, Sh, Sw, Sd,
    // Zicsr
    Csrrw, Csrrs, Csrrc, Csrrwi, Csrrsi, Csrrci,
    // Privileged instructions
    Ecall,
    Mret,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, TryFromPrimitive)]
#[repr(u8)]
pub enum Opcode {
    Lui = 0b110111,
    Auipc = 0b0010111,
    OpImm = 0b0010011,
    Op = 0b0110011,
    Jal = 0b1101111,
    Jalr = 0b1100111,
    Branch = 0b1100011,
    Load = 0b0000011,
    Store = 0b0100011,
    System = 0b1110011,
}

impl Op {
    pub fn opcode(&self) -> u8 {
        use Op::*;
        let oc = match self {
            Lui => Opcode::Lui,
            Auipc => Opcode::Auipc,
            Addi | Slti | Sltiu | Xori | Ori | Andi | Slli | Srli | Srai => Opcode::OpImm,
            Add | Sub => Opcode::Op,
            Jal => Opcode::Jal,
            Jalr => Opcode::Jalr,
            Beq | Bne | Blt | Bge | Bltu | Bgeu => Opcode::Branch,
            Lb | Lh | Lw | Lbu | Lhu | Lwu | Ld => Opcode::Load,
            Sb | Sh | Sw | Sd => Opcode::Store,
            Csrrw | Csrrs | Csrrc | Csrrwi | Csrrsi | Csrrci | Ecall | Mret => Opcode::System,
        };
        oc as u8
    }

    pub fn funct(&self) -> Option<u8> {
        use Op::*;
        match self {
            Addi => Some(0b000),
            Slti => Some(0b010),
            Sltiu => Some(0b011),
            Xori => Some(0b100),
            Ori => Some(0b110),
            Andi => Some(0b111),
            Slli => Some(0b001),
            Srli | Srai => Some(0b101),
            Add => Some(0b000),
            Sub => Some(0b000),
            Beq => Some(0b000),
            Bne => Some(0b001),
            Blt => Some(0b100),
            Bge => Some(0b101),
            Bltu => Some(0b110),
            Bgeu => Some(0b111),
            Lb => Some(0b000),
            Lh => Some(0b001),
            Lw => Some(0b010),
            Lbu => Some(0b100),
            Lhu => Some(0b101),
            Lwu => Some(0b110),
            Ld => Some(0b011),
            Sb => Some(0b000),
            Sh => Some(0b001),
            Sw => Some(0b010),
            Sd => Some(0b011),
            Ecall => Some(0),
            Mret => Some(0b0011000),
            Csrrw => Some(0b001),
            Csrrs => Some(0b010),
            Csrrc => Some(0b011),
            Csrrwi => Some(0b101),
            Csrrsi => Some(0b110),
            Csrrci => Some(0b111),
            _ => None,
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum InstrType {
    R,
    I,
    S,
    B,
    U,
    J,
}

pub mod instr {
    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
    pub struct R {
        pub funct7: u8,
        pub rs2: u8,
        pub rs1: u8,
        pub rd: u8,
    }

    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
    pub struct I {
        pub imm: u16,
        pub rs1: u8,
        pub rd: u8,
    }

    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
    pub struct S {
        pub imm: u16,
        pub rs2: u8,
        pub rs1: u8,
    }

    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
    pub struct U {
        pub imm: u32,
        pub rd: u8,
    }

    #[derive(PartialEq, Eq, Debug, Clone, Copy)]
    pub struct B {
        pub imm: u32,
        pub rs2: u8,
        pub rs1: u8,
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Instr {
    R(instr::R),
    I(instr::I),
    S(instr::S),
    U(instr::U),
    B(instr::B),
}

impl Instr {
    pub fn as_r_mut(&mut self) -> Option<&mut instr::R> {
        match self {
            Instr::R(i) => Some(i),
            _ => None,
        }
    }

    pub fn as_i_mut(&mut self) -> Option<&mut instr::I> {
        match self {
            Instr::I(i) => Some(i),
            _ => None,
        }
    }

    pub fn as_s_mut(&mut self) -> Option<&mut instr::S> {
        match self {
            Instr::S(i) => Some(i),
            _ => None,
        }
    }

    pub fn as_u_mut(&mut self) -> Option<&mut instr::U> {
        match self {
            Instr::U(i) => Some(i),
            _ => None,
        }
    }

    pub fn as_r(&self) -> Option<&instr::R> {
        match self {
            Instr::R(i) => Some(i),
            _ => None,
        }
    }

    pub fn as_i(&self) -> Option<&instr::I> {
        match self {
            Instr::I(i) => Some(i),
            _ => None,
        }
    }

    pub fn as_s(&self) -> Option<&instr::S> {
        match self {
            Instr::S(i) => Some(i),
            _ => None,
        }
    }

    pub fn as_u(&self) -> Option<&instr::U> {
        match self {
            Instr::U(i) => Some(i),
            _ => None,
        }
    }
}

/// Decoded instruction
#[derive(PartialEq, Eq, Debug)]
pub struct DInstr {
    pub op: Op,
    pub contents: Instr,
}

impl DInstr {
    pub fn encode(&self) -> u32 {
        let mut bits = self.op.opcode() as u32;
        match self.contents {
            Instr::R(instr) => {
                bits |= (instr.rd as u32) << 7;
                bits |= (self.op.funct().unwrap() as u32) << 12;
                bits |= (instr.rs1 as u32) << 15;
                bits |= (instr.rs2 as u32) << 20;
                bits |= (instr.funct7 as u32) << 25;
            }
            Instr::I(instr) => {
                bits |= (instr.rd as u32) << 7;
                bits |= (self.op.funct().unwrap() as u32) << 12;
                bits |= (instr.rs1 as u32) << 15;
                bits |= (instr.imm as u32) << 20;
            }
            Instr::S(instr) => {
                bits |= ((instr.imm & 0x1F) as u32) << 7;
                bits |= (self.op.funct().unwrap() as u32) << 12;
                bits |= (instr.rs1 as u32) << 15;
                bits |= (instr.rs2 as u32) << 20;
                bits |= (((instr.imm >> 5) & 0x7F) as u32) << 25;
            }
            Instr::U(instr) => {
                bits |= (instr.rd as u32) << 7;
                bits |= (instr.imm as u32) << 12;
            }
            Instr::B(instr) => {
                bits |= ((instr.imm >> 11) & 1) << 7;
                bits |= ((instr.imm >> 1) & 0xF) << 8;
                bits |= (self.op.funct().unwrap() as u32) << 12;
                bits |= (instr.rs1 as u32) << 15;
                bits |= (instr.rs2 as u32) << 20;
                bits |= ((instr.imm >> 5) & 0x3F) << 25;
                bits |= ((instr.imm >> 12) & 1) << 31;
            }
        }
        bits
    }

    pub fn decode(instr: u32) -> Option<DInstr> {
        let opcode = (instr & 0x7F) as u8; // [6,0]
        let rd = ((instr >> 7) & 0x1F) as u8; // [11,7]
        let funct3 = ((instr >> 12) & 0x7) as u8; // [14,12]
        let rs1 = ((instr >> 15) & 0x1F) as u8; // [19,15]
        let rs2 = ((instr >> 20) & 0x1F) as u8; // [24,20]
        let funct7 = (instr >> 25) as u8; // [31,25]
        let imm12 = (instr >> 20) as u16; // [31,20]
        let opcode = Opcode::try_from(opcode).ok()?;
        let (op, itype) = match opcode {
            Opcode::Lui => (Op::Lui, InstrType::U),
            Opcode::Auipc => (Op::Auipc, InstrType::U),
            Opcode::OpImm => {
                let op = match funct3 {
                    0x0 => Op::Addi,
                    0x2 => Op::Slti,
                    0x3 => Op::Sltiu,
                    0x4 => Op::Xori,
                    0x6 => Op::Ori,
                    0x7 => Op::Andi,
                    0x1 => Op::Slli,
                    0x5 => {
                        match funct7 {
                            0x0 => Op::Srli,
                            0x2F => Op::Srai,
                            _ => return None,
                        }
                    }
                    _ => unreachable!(),
                };
                (op, InstrType::I)
            }
            Opcode::Op => {
                // TODO: check funct
                let op = match funct3 {
                    0 => {
                        match funct7 {
                            0 => Op::Add,
                            _ => unimplemented!()
                        }
                    },
                    _ => unimplemented!()
                };
                (op, InstrType::R)
            },
            Opcode::Store => {
                let op = match funct3 {
                    0x0 => Op::Sb,
                    0x1 => Op::Sh,
                    0x2 => Op::Sw,
                    0x3 => Op::Sd,
                    _ => return None
                };
                (op, InstrType::S)
            },
            Opcode::Load => {
                let op = match funct3 {
                    0x0 => Op::Lb,
                    0x1 => Op::Lh,
                    0x2 => Op::Lw,
                    0x4 => Op::Lbu,
                    0x5 => Op::Lhu,
                    0x6 => Op::Lwu,
                    0x3 => Op::Ld,
                    _ => unreachable!()
                };
                (op, InstrType::I)
            },
            Opcode::System => {
                match funct3 {
                    0x1 => (Op::Csrrw, InstrType::I),
                    0x2 => (Op::Csrrs, InstrType::I),
                    0x4 => (Op::Csrrc, InstrType::I),
                    0x5 => (Op::Csrrwi, InstrType::I),
                    0x6 => (Op::Csrrsi, InstrType::I),
                    0x7 => (Op::Csrrci, InstrType::I),
                    0x0 if rd == 0 && rs1 == 0 => {
                        match funct7 {
                            0x0 if rs2 == 0 => (Op::Ecall, InstrType::I),
                            0b0011000 => (Op::Mret, InstrType::R),
                            _ => return None
                        }
                    }
                    _ => unreachable!()
                }
            },
            Opcode::Jal => (Op::Jal, InstrType::U),
            Opcode::Jalr => (Op::Jalr, InstrType::I),
            Opcode::Branch => {
                let op = match funct3 {
                    0x0 => Op::Beq,
                    0x1 => Op::Bne,
                    0x4 => Op::Blt,
                    0x5 => Op::Bge,
                    0x6 => Op::Bltu,
                    0x7 => Op::Bgeu,
                    _ => return None,
                };
                (op, InstrType::B)
            },
        };
        let contents = match itype {
            InstrType::R => Instr::R(instr::R { funct7, rs1, rs2, rd }),
            InstrType::I => {
                Instr::I(instr::I { imm: imm12, rs1, rd })
            }
            InstrType::S => {
                let imm = ((((instr >> 25) & 0x7F) << 5) | ((instr >> 7) & 0x1F)) as u16;
                Instr::S(instr::S { imm, rs1, rs2 })
            }
            InstrType::U => {
                let imm = instr >> 12;
                Instr::U(instr::U { imm, rd })
            }
            InstrType::B => {
                let imm = ((instr >> 31) << 12)
                    | (((instr >> 7) & 0x1) << 11)
                    | (((instr >> 25) & 0x3F) << 5)
                    | (((instr >> 8) & 0xF) << 1);
                Instr::B(instr::B { imm, rs1, rs2 })
            }
            InstrType::J => unreachable!(),
        };
        Some(DInstr { op, contents })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enc_dec_i() {
        use Op::*;
        let contents = Instr::I(instr::I {
            rd: 5,
            rs1: 1,
            imm: 43,
        });
        let ops = vec![
            Addi, Slti, Sltiu, Xori, Ori, Andi, Slli, /* Srli, Srai, */
        ];
        for op in &ops {
            let instr = DInstr { op: *op, contents };
            let enc = instr.encode();
            println!("Testing {:?}; encoding = 0x{:08x}", op, enc);
            let dec_opt = DInstr::decode(enc);
            assert!(dec_opt.is_some());
            assert_eq!(dec_opt.unwrap(), instr);
        }
    }

    #[test]
    fn decode_add() {
        let instr = DInstr {
            op: Op::Addi,
            contents: Instr::I(instr::I {
                rd: 0,
                rs1: 1,
                imm: 4,
            }),
        };
        let raw = 0x408013;
        let dec_opt = DInstr::decode(raw);
        assert!(dec_opt.is_some());
        assert_eq!(instr, dec_opt.unwrap());
        assert_eq!(instr.encode(), raw);
    }

    #[test]
    fn decode_sw() {
        let instr = DInstr {
            op: Op::Sw,
            contents: Instr::S(instr::S {
                rs2: 15,    /* a5 */
                rs1: 8,     /* s0 */
                imm: 0xfe4, /* -28 */
            }),
        };
        let raw = 0xfef42223; // sw a5, -28(s0)
        let dec_opt = DInstr::decode(raw);
        assert!(dec_opt.is_some());
        assert_eq!(instr, dec_opt.unwrap());
        assert_eq!(instr.encode(), raw);
    }

    #[test]
    fn decode_jal() {
        let instr = DInstr {
            op: Op::Jal,
            contents: Instr::U(instr::U {
                rd: 0,
                imm: 0x11c00,
            }),
        };
        let raw = 0x11c0006f; // jal zero, 0x13c
        let dec_opt = DInstr::decode(raw);
        assert!(dec_opt.is_some());
        assert_eq!(instr, dec_opt.unwrap());
        assert_eq!(instr.encode(), raw);
    }

    #[test]
    fn decode_bne() {
        let instr = DInstr {
            op: Op::Bne,
            contents: Instr::B(instr::B {
                rs1: 15,
                rs2: 0,
                imm: 0x1fcc,
            }),
        };
        let raw = 0xfc0796e3; // bne a5, zero, pc-0x34
        let dec_opt = DInstr::decode(raw);
        assert!(dec_opt.is_some());
        assert_eq!(instr, dec_opt.unwrap());
        assert_eq!(instr.encode(), raw);
    }
}
