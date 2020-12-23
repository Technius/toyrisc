use std::ops::Range;

pub struct MmioDevice {
    pub addr_range: Range<u64>,
    pub(crate) handler: Box<dyn MmioHandler>,
}

impl MmioDevice {
    pub fn new(addr_range: Range<u64>, handler: Box<dyn MmioHandler>) -> Self {
        MmioDevice {
            addr_range,
            handler,
        }
    }
}

pub trait MmioHandler {
    /// Machine is reset
    fn mmio_reset(&mut self) {}

    /// Read `len` (up to 8) bytes from the MMIO device
    fn mmio_read(&mut self, offset: u64, len: u64) -> u64;

    /// Write `len` (up to 8) bytes of `value` to the MMIO device
    fn mmio_write(&mut self, offset: u64, len: u64, value: u64);
}

pub struct ConsoleDevice {}
impl MmioHandler for ConsoleDevice {
    fn mmio_read(&mut self, _offset: u64, _len: u64) -> u64 { 0 }

    fn mmio_write(&mut self, offset: u64, len: u64, value: u64) {
        if offset == 0 {
            let bytes = value.to_le_bytes();
            for i in 0..len {
                print!("{}", bytes[i as usize] as char);
            }
        }
    }
}
