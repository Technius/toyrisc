use toyrisc::machine::Machine;
use toyrisc::mmio::{MmioDevice, ConsoleDevice};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let filename = if args.len() > 1 {
        args[1].clone()
    } else {
        "tests/crosscompile/bin/hello.bin".to_string()
    };
    println!("Loading raw instructions from {}", filename);

    let program = {
        use std::io::Read;
        let f = std::fs::File::open(filename).unwrap();
        let mut p = Vec::new();
        let mut buf_reader = std::io::BufReader::new(f);
        buf_reader.read_to_end(&mut p).unwrap();
        p
    };

    // assert!(program.len() % 4 == 0);

    let mut machine = Machine::new();
    machine.register_mmio(MmioDevice::new(0x10000000..0x10001000, Box::new(ConsoleDevice {})));
    for i in 0..program.len() {
        machine.memory[i] = program[i];
    }
    machine.run();
}
