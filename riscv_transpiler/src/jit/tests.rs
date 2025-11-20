use risc_v_simulator::abstractions::non_determinism::QuasiUARTSource;

use super::*;
use crate::vm::test::*;
use std::path::Path;

#[test]
fn test_jit_simple_fibonacci() {
    let path = std::env::current_dir().unwrap();
    println!("The current directory is {}", path.display());

    // let (_, binary) = read_binary(&Path::new("riscv_transpiler/examples/fibonacci/app.bin"));
    // let (_, text) = read_binary(&Path::new("riscv_transpiler/examples/fibonacci/app.text"));

    // let (_, binary) = read_binary(&Path::new("examples/fibonacci/app.bin"));
    // let (_, text) = read_binary(&Path::new("examples/fibonacci/app.text"));

    let (_, binary) = read_binary(&Path::new("examples/keccak_f1600/app.bin"));
    let (_, text) = read_binary(&Path::new("examples/keccak_f1600/app.text"));

    JittedCode::<_>::run_alternative_simulator(&text, &mut (), &binary, None);
}

#[test]
fn test_few_instr() {
    use std::collections::HashMap;

    // let source = [
    //     "addi x1, x0, 1234",
    //     "addi x2, x0, 4",
    //     "sw x1, 4(x2)"
    // ];

    // let source = [
    //     "addi x1, x0, 1234",
    //     "addi x2, x0, 4",
    //     "sh x1, 2(x2)"
    // ];

    let source = [
        "addi x1, x0, 1234",
        "addi x2, x0, 4",
        "sb x1, 0(x2)",
        "addi x4, x0, 8",
        "lb x3, -4(x4)",
    ];

    let mut empty_hash: HashMap<String, u32> = HashMap::new();
    let mut text = vec![];
    for el in source.into_iter() {
        let encoding = lib_rv32_asm::assemble_ir(el, &mut empty_hash, 0)
            .unwrap()
            .unwrap();
        text.push(encoding);
    }
    text.push(0x0000006f);

    JittedCode::<_>::run_alternative_simulator(&text, &mut (), &[], None);
}

#[test]
fn test_jit_full_block() {
    let path = std::env::current_dir().unwrap();
    println!("The current directory is {}", path.display());

    let (_, binary) = read_binary(&Path::new("examples/zksync_os/app.bin"));
    let (_, text) = read_binary(&Path::new("examples/zksync_os/app.text"));

    let (witness, _) = read_binary(&Path::new("examples/zksync_os/23620012_witness"));
    let witness = hex::decode(core::str::from_utf8(&witness).unwrap()).unwrap();
    let witness: Vec<_> = witness
        .as_chunks::<4>()
        .0
        .iter()
        .map(|el| u32::from_be_bytes(*el))
        .collect();
    let mut source = QuasiUARTSource::new_with_reads(witness);

    let (state, _) = JittedCode::<_>::run_alternative_simulator(&text, &mut source, &binary, None);

    println!("PC = 0x{:08x}", state.pc);
    dbg!(state.registers);
}

fn run_reference_for_num_cycles(
    binary: &[u32],
    text: &[u32],
    mut source: impl NonDeterminismCSRSource,
    bound: u32,
) -> (
    State<DelegationsAndFamiliesCounters>,
    RamWithRomRegion<{ common_constants::rom::ROM_SECOND_WORD_BITS }>,
) {
    use crate::ir::*;

    let instructions: Vec<Instruction> =
        preprocess_bytecode::<FullUnsignedMachineDecoderConfig>(text);
    let tape = SimpleTape::new(&instructions);
    let mut ram =
        RamWithRomRegion::<{ common_constants::rom::ROM_SECOND_WORD_BITS }>::from_rom_content(
            &binary,
            1 << 30,
        );

    let mut state = State::initial_with_counters(DelegationsAndFamiliesCounters::default());
    // let mut snapshotter = SimpleSnapshotter::new_with_cycle_limit(1 << 31, period, state);

    VM::run_basic_unrolled::<_, _, _>(
        &mut state,
        &mut ram,
        // &mut snapshotter,
        &mut (),
        &tape,
        bound as usize,
        &mut source,
    );

    (state, ram)
}

#[test]
fn test_reference_block_exec() {
    use crate::ir::*;

    let (_, binary) = read_binary(&Path::new("examples/zksync_os/app.bin"));
    let (_, text) = read_binary(&Path::new("examples/zksync_os/app.text"));

    let (witness, _) = read_binary(&Path::new("examples/zksync_os/23620012_witness"));
    let witness = hex::decode(core::str::from_utf8(&witness).unwrap()).unwrap();
    let witness: Vec<_> = witness
        .as_chunks::<4>()
        .0
        .iter()
        .map(|el| u32::from_be_bytes(*el))
        .collect();
    let mut source = QuasiUARTSource::new_with_reads(witness);

    let instructions: Vec<Instruction> =
        preprocess_bytecode::<FullUnsignedMachineDecoderConfig>(&text);
    let tape = SimpleTape::new(&instructions);
    let mut ram =
        RamWithRomRegion::<{ common_constants::rom::ROM_SECOND_WORD_BITS }>::from_rom_content(
            &binary,
            1 << 30,
        );

    let cycles_bound = 1 << 31;

    let mut state = State::initial_with_counters(DelegationsAndFamiliesCounters::default());
    let mut snapshotter = SimpleSnapshotter::<_, { common_constants::rom::ROM_SECOND_WORD_BITS }>::new_with_cycle_limit(cycles_bound, state);

    let now = std::time::Instant::now();
    VM::run_basic_unrolled::<_, _, _>(
        &mut state,
        &mut ram,
        &mut snapshotter,
        &tape,
        cycles_bound,
        &mut source,
    );
    let elapsed = now.elapsed();

    println!("PC = 0x{:08x}", state.pc);
    dbg!(state.registers.map(|el| el.value));
}

#[test]
fn run_and_compare() {
    let (_, binary) = read_binary(&Path::new("examples/zksync_os/app.bin"));
    let (_, text) = read_binary(&Path::new("examples/zksync_os/app.text"));

    let (witness, _) = read_binary(&Path::new("examples/zksync_os/23620012_witness"));
    let witness = hex::decode(core::str::from_utf8(&witness).unwrap()).unwrap();
    let witness: Vec<_> = witness
        .as_chunks::<4>()
        .0
        .iter()
        .map(|el| u32::from_be_bytes(*el))
        .collect();
    let mut source = QuasiUARTSource::new_with_reads(witness);

    let step = 1 << 22;
    let initial_step = step;
    let upper_bound = (1 << 30) - 8;

    let mut num_steps = initial_step;
    while num_steps < upper_bound {
        let (jit_state, jit_memory) = JittedCode::run_alternative_simulator(
            &text,
            &mut source.clone(),
            &binary,
            Some(num_steps),
        );

        // NOTE: as JITted simulator skips many cycles in case of precompiles, we need to make it more precise for reference one
        let reference_cycles = (jit_state.timestamp - INITIAL_TIMESTAMP) / TIMESTAMP_STEP;

        println!(
            "JITted simulator ran for {} cycles instead of {}",
            reference_cycles, num_steps
        );

        let (reference_state, reference_rom) =
            run_reference_for_num_cycles(&binary, &text, source.clone(), num_steps as u32);

        assert_eq!(
            reference_state.timestamp, jit_state.timestamp,
            "TIMESTAMP diverged after {} steps",
            num_steps
        );
        if reference_state.pc != jit_state.pc {
            panic!(
                "PC diverged after {} steps: expected 0x{:08x}, got 0x{:08x}",
                num_steps, reference_state.pc, jit_state.pc,
            );
        }

        assert_eq!(
            reference_state.counters.add_sub_family as u32,
            jit_state.counters[CounterType::AddSubLui as u8 as usize]
        );
        assert_eq!(
            reference_state.counters.slt_branch_family as u32,
            jit_state.counters[CounterType::BranchSlt as u8 as usize]
        );
        assert_eq!(
            reference_state.counters.binary_shift_csr_family as u32,
            jit_state.counters[CounterType::ShiftBinaryCsr as u8 as usize]
        );
        assert_eq!(
            reference_state.counters.mul_div_family as u32,
            jit_state.counters[CounterType::MulDiv as u8 as usize]
        );
        assert_eq!(
            reference_state.counters.word_size_mem_family as u32,
            jit_state.counters[CounterType::MemWord as u8 as usize]
        );
        assert_eq!(
            reference_state.counters.subword_size_mem_family as u32,
            jit_state.counters[CounterType::MemSubword as u8 as usize]
        );
        assert_eq!(
            reference_state.counters.blake_calls as u32,
            jit_state.counters[CounterType::BlakeDelegation as u8 as usize]
        );
        assert_eq!(
            reference_state.counters.bigint_calls as u32,
            jit_state.counters[CounterType::BigintDelegation as u8 as usize]
        );
        assert_eq!(
            reference_state.counters.keccak_calls as u32,
            jit_state.counters[CounterType::KeccakDelegation as u8 as usize]
        );

        let mut equal_state = true;
        for (reg_idx, ((reference, jit_value), jit_ts)) in reference_state
            .registers
            .iter()
            .zip(jit_state.registers.iter())
            .zip(jit_state.register_timestamps.iter())
            .enumerate()
        {
            if reference.value != *jit_value {
                println!(
                    "VALUE diverged for x{} after {} steps:\nreference\n{}\njitted\n{}",
                    reg_idx, num_steps, reference.value, jit_value
                );
                equal_state = false;
            }
            if reference.timestamp != *jit_ts {
                println!(
                    "TIMESTAMP diverged for x{} after {} steps:\nreference\n{}\njitted\n{}",
                    reg_idx, num_steps, reference.timestamp, jit_ts
                );
                equal_state = false;
            }
        }

        assert_eq!(reference_rom.backing.len(), jit_memory.memory.len());
        for (word_idx, ((reference_value, jit_value), jit_ts)) in reference_rom
            .backing
            .iter()
            .zip(jit_memory.memory.iter())
            .zip(jit_memory.timestamps.iter())
            .enumerate()
        {
            assert_eq!(
                reference_value.value, *jit_value,
                "VALUE diverged for word {} after {} steps",
                word_idx, num_steps
            );
            assert_eq!(
                reference_value.timestamp, *jit_ts,
                "TIMESTAMP diverged for word {} after {} steps",
                word_idx, num_steps
            );
        }

        if equal_state == false {
            panic!("State diverged");
        }

        println!("Passed for {} cycles", num_steps);

        num_steps += step;
    }
}
