use common_constants::{INITIAL_TIMESTAMP, TIMESTAMP_STEP};
use risc_v_simulator::abstractions::non_determinism::QuasiUARTSource;

use super::*;

#[test]
fn baseline_forward_benchmark() {
    let (binary, binary_u32) = read_binary(Path::new("blobs/app.bin"));
    let (text, text_u32) = read_binary(Path::new("blobs/app.text"));

    let mut file = std::fs::File::open("blobs/23620012_witness").expect("should open file");
    let mut witness = vec![];
    file.read_to_end(&mut witness)
        .expect("must read witness from file");
    let witness = hex::decode(core::str::from_utf8(&witness).unwrap()).unwrap();
    assert_eq!(witness.len() % 4, 0);
    let witness: Vec<_> = witness
        .as_chunks::<4>()
        .0
        .iter()
        .map(|el| u32::from_be_bytes(*el))
        .collect();

    let instructions: Vec<Instruction> = text_u32
        .iter()
        .map(|el| decode::<FullUnsignedMachineDecoderConfig>(*el))
        .collect();
    let tape = SimpleTape::new(&instructions);
    let period = 1 << 20;
    let num_snapshots = 1000;

    for _ in 0..16 {
        let mut ram = BenchmarkingRAM::from_rom_content(&binary_u32, 1 << 30);
        let mut non_determinism = QuasiUARTSource::new_with_reads(witness.clone());
        let mut state = State::initial_with_counters(());

        let now = std::time::Instant::now();
        run_baseline_bench(
            &mut state,
            num_snapshots,
            &mut ram,
            &tape,
            period,
            &mut non_determinism,
        );
        let elapsed = now.elapsed();

        let exact_cycles = (state.timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP;

        println!(
            "Performance is {} MHz ({} cycles)",
            (exact_cycles as f64) / (elapsed.as_micros() as f64),
            exact_cycles
        );
    }
}