use std::path::Path;
use std::io::Read;
use risc_v_simulator::abstractions::non_determinism::QuasiUARTSource;

use crate::ir::*;
use crate::vm::*;

#[cfg(test)]
mod baseline_no_snapshotting;

#[track_caller]
pub fn read_binary(path: &Path) -> (Vec<u8>, Vec<u32>) {
    let mut file = std::fs::File::open(path).expect("must open provided file");
    let mut buffer = vec![];
    file.read_to_end(&mut buffer).expect("must read the file");
    assert_eq!(buffer.len() % core::mem::size_of::<u32>(), 0);
    let mut binary = Vec::with_capacity(buffer.len() / core::mem::size_of::<u32>());
    for el in buffer.as_chunks::<4>().0 {
        binary.push(u32::from_le_bytes(*el));
    }

    (buffer, binary)
}

// #[inline(never)]
#[inline(always)]
pub fn run_baseline_bench(
    state: &mut State<()>,
    num_snapshots: usize,
    ram: &mut BenchmarkingRAM,
    tape: &SimpleTape,
    period: usize,
    non_determinism: &mut QuasiUARTSource,
) {
    VM::run_basic_unrolled::<
        (),
        _,
        _,
    >(
        state,
        num_snapshots,
        ram,
        &mut (),
        tape,
        period,
        non_determinism,
    );
}

