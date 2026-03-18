use core::mem::offset_of;

use super::*;
#[cfg(test)]
use prover::prover_stages::Proof;
#[cfg(feature = "legacy_tests")]
use verifier_common::proof_flattener::*;
use verifier_common::prover::nd_source_std::*;
use verifier_common::{
    cs::one_row_compiler::CompiledCircuitArtifact, DefaultLeafInclusionVerifier,
};

#[allow(dead_code)]
fn serialize_to_file<T: serde::Serialize>(el: &T, filename: &str) {
    let mut dst = std::fs::File::create(filename).unwrap();
    serde_json::to_writer_pretty(&mut dst, el).unwrap();
}

fn deserialize_from_file<T: serde::de::DeserializeOwned>(filename: &str) -> T {
    let src = std::fs::File::open(filename).unwrap();
    serde_json::from_reader(src).unwrap()
}

#[cfg(test)]
use test_utils::skip_if_ci;

#[cfg(test)]
#[ignore = "manual unified/delegation verifier fixture test"]
#[test]
fn test_unified_cycle_or_delegation() {
    skip_if_ci!();
    // create an oracle to feed into verifier and look at the transcript values

    // let proof: Proof = deserialize_from_file("../../zksync-airbender/prover/delegation_proof");
    // let proof: Proof = deserialize_from_file("../../zksync-airbender/prover/blake2s_delegator_proof");
    let proof: Proof =
        deserialize_from_file("../../zksync-airbender/prover/keccak_delegator_proof");

    // let compiled_circuit: CompiledCircuitArtifact<Mersenne31Field> =
    //     deserialize_from_file("../../zksync-airbender/prover/full_machine_layout.json");
    // let compiled_circuit: CompiledCircuitArtifact<Mersenne31Field> =
    // deserialize_from_file("../../zksync-airbender/prover/blake2s_delegator_layout");
    let compiled_circuit: CompiledCircuitArtifact<Mersenne31Field> =
        deserialize_from_file("../prover/keccak_delegation_circuit_layout.json");

    // now form flattened iterator
    use verifier_common::proof_flattener::*;

    let mut oracle_data = vec![];
    oracle_data.extend(flatten_proof_for_skeleton(
        &proof,
        compiled_circuit
            .memory_layout
            .shuffle_ram_inits_and_teardowns
            .len(),
    ));
    for query in proof.queries.iter() {
        oracle_data.extend(flatten_query(query));
    }

    // Spawn a new thread as it's large stack in debug builds
    let result = std::thread::Builder::new()
        .name("verifier thread".to_string())
        .stack_size(1 << 27)
        .spawn(move || {
            let it = oracle_data.into_iter();

            set_iterator(it);

            #[allow(invalid_value)]
            unsafe {
                verify_with_configuration::<ThreadLocalBasedSource, DefaultLeafInclusionVerifier>(
                    &mut MaybeUninit::uninit().assume_init(),
                    &mut ProofPublicInputs::uninit(),
                )
            };
        })
        .map(|t| t.join());

    match result {
        Ok(..) => {}
        Err(err) => {
            panic!("Verifier thread fails with {}", err);
        }
    }
}

#[test]
fn test_unrolled_circuit() {
    // create an oracle to feed into verifier and look at the transcript values

    // let name = "add_sub_lui_auipc_mop";
    // let name = "jump_branch_slt";
    let name = "shift_binop_csrrw";
    // let name = "mul_div_unsigned";
    // let name = "word_only_load_store";
    // let name = "subword_only_load_store";
    // let name = "inits_and_teardowns";

    let proof: prover::prover_stages::unrolled_prover::UnrolledModeProof =
        deserialize_from_file(&format!("../prover/{}_unrolled_proof.json", name));
    let compiled_circuit: CompiledCircuitArtifact<Mersenne31Field> =
        deserialize_from_file(&format!("../cs/{}_preprocessed_layout.json", name));

    dbg!(&proof.public_inputs);
    dbg!(&proof.aux_boundary_values);
    dbg!(&proof.delegation_argument_accumulator);

    // now form flattened iterator
    use verifier_common::proof_flattener::*;

    let mut oracle_data = vec![];
    oracle_data.extend(flatten_unrolled_circuits_proof_for_skeleton(
        &proof,
        &compiled_circuit,
    ));
    for query in proof.queries.iter() {
        oracle_data.extend(flatten_query(query));
    }

    // Spawn a new thread as it's large stack in debug builds
    let result = std::thread::Builder::new()
        .name("verifier thread".to_string())
        .stack_size(1 << 27)
        .spawn(move || {
            let it = oracle_data.into_iter();

            set_iterator(it);

            #[allow(invalid_value)]
            unsafe {
                verify_with_configuration::<ThreadLocalBasedSource, DefaultLeafInclusionVerifier>(
                    &mut MaybeUninit::uninit().assume_init(),
                    &mut ProofPublicInputs::uninit(),
                )
            };
        })
        .map(|t| t.join());

    match result {
        Ok(..) => {}
        Err(err) => {
            panic!("Verifier thread fails with {}", err);
        }
    }
}

#[cfg(feature = "legacy_tests")]
use risc_v_simulator::abstractions::non_determinism::QuasiUARTSourceState;
#[cfg(feature = "legacy_tests")]
use risc_v_simulator::cycle::IWithoutByteAccessIsaConfigWithDelegation;
#[cfg(feature = "legacy_tests")]
use std::collections::VecDeque;
#[cfg(feature = "legacy_tests")]
struct VectorBasedNonDeterminismSource(VecDeque<u32>, QuasiUARTSourceState);

#[cfg(feature = "legacy_tests")]
impl
    risc_v_simulator::abstractions::non_determinism::NonDeterminismCSRSource<
        risc_v_simulator::abstractions::memory::VectorMemoryImpl,
    > for VectorBasedNonDeterminismSource
{
    fn read(&mut self) -> u32 {
        self.0.pop_front().unwrap()
    }
    fn write_with_memory_access(
        &mut self,
        _memory: &risc_v_simulator::abstractions::memory::VectorMemoryImpl,
        value: u32,
    ) {
        self.1.process_write(value);
    }
}

#[cfg(feature = "legacy_tests")]
#[ignore = "legacy fixture format drifts; run manually when fixtures are refreshed"]
#[test]
// TODO(legacy-cleanup): determine whether the legacy code path exercised here can be removed.
fn test_full_machine_verifier_out_of_simulator() {
    let proof: Proof = deserialize_from_file("../prover/delegation_proof");
    let compiled_circuit: CompiledCircuitArtifact<Mersenne31Field> =
        deserialize_from_file("../prover/full_machine_layout.json");

    let mut oracle_data: Vec<u32> = vec![];

    oracle_data.extend(flatten_proof_for_skeleton(
        &proof,
        compiled_circuit
            .memory_layout
            .shuffle_ram_inits_and_teardowns
            .len(),
    ));
    for query in proof.queries.iter() {
        oracle_data.extend(flatten_query(query));
    }

    // we have a problem with a stack size in debug, so let's cheat
    std::thread::Builder::new()
        .stack_size(1 << 27)
        .spawn(move || {
            let it = oracle_data.into_iter();

            set_iterator(it);

            #[allow(invalid_value)]
            let mut proof_output: ProofOutput<
                TREE_CAP_SIZE,
                NUM_COSETS,
                NUM_DELEGATION_CHALLENGES,
                NUM_AUX_BOUNDARY_VALUES,
                NUM_MACHINE_STATE_PERMUTATION_CHALLENGES,
            > = unsafe { MaybeUninit::uninit().assume_init() };
            let mut state_variables = ProofPublicInputs::uninit();

            unsafe { verify(&mut proof_output, &mut state_variables) };

            dbg!(proof_output, state_variables);
        })
        .unwrap()
        .join()
        .unwrap();
}

#[cfg(feature = "legacy_tests")]
#[ignore = "legacy fixture format drifts; run manually when fixtures are refreshed"]
#[test]
// TODO(legacy-cleanup): determine whether the legacy code path exercised here can be removed.
fn test_reduced_machine_verifier_out_of_simulator() {
    let proof: Proof = deserialize_from_file("../prover/reduced_machine_proof");
    let compiled_circuit: CompiledCircuitArtifact<Mersenne31Field> =
        deserialize_from_file("../prover/reduced_machine_layout");

    let mut oracle_data: Vec<u32> = vec![];

    oracle_data.extend(flatten_proof_for_skeleton(
        &proof,
        compiled_circuit
            .memory_layout
            .shuffle_ram_inits_and_teardowns
            .len(),
    ));
    for query in proof.queries.iter() {
        oracle_data.extend(flatten_query(query));
    }

    // we have a problem with a stack size in debug, so let's cheat
    std::thread::Builder::new()
        .stack_size(1 << 27)
        .spawn(move || {
            let it = oracle_data.into_iter();

            set_iterator(it);

            #[allow(invalid_value)]
            let mut proof_output: ProofOutput<
                TREE_CAP_SIZE,
                NUM_COSETS,
                NUM_DELEGATION_CHALLENGES,
                NUM_AUX_BOUNDARY_VALUES,
                NUM_MACHINE_STATE_PERMUTATION_CHALLENGES,
            > = unsafe { MaybeUninit::uninit().assume_init() };
            let mut state_variables = ProofPublicInputs::uninit();

            unsafe { verify(&mut proof_output, &mut state_variables) };

            dbg!(proof_output, state_variables);
        })
        .unwrap()
        .join()
        .unwrap();
}

// #[ignore = "Requires ZKsyncOS app bin"]
#[cfg(feature = "legacy_tests")]
#[ignore = "manual simulator integration test"]
#[test]
// TODO(legacy-cleanup): determine whether the legacy code path exercised here can be removed.
fn test_verifier_in_simulator() {
    let proof: Proof = deserialize_from_file("../../zksync-airbender/prover/delegation_proof");
    let compiled_circuit: CompiledCircuitArtifact<Mersenne31Field> =
        deserialize_from_file("../../zksync-airbender/prover/full_machine_layout.json");

    // let proof: Proof = deserialize_from_file("../../zksync-airbender/prover/proof");
    // let compiled_circuit: CompiledCircuitArtifact<Mersenne31Field> =
    //     deserialize_from_file("../../zksync-airbender/prover/layout");

    let mut oracle_data: Vec<u32> = vec![];
    {
        oracle_data.extend(flatten_proof_for_skeleton(
            &proof,
            compiled_circuit
                .memory_layout
                .shuffle_ram_inits_and_teardowns
                .len(),
        ));
        for query in proof.queries.iter() {
            oracle_data.extend(flatten_query(query));
        }

        let path = "../tools/verifier/tester.bin";
        let path_sym = "../tools/verifier/tester.elf";

        use risc_v_simulator::runner::run_simple_with_entry_point_and_non_determimism_source_for_config;
        use risc_v_simulator::sim::*;

        let mut config = SimulatorConfig::simple(path);
        config.cycles = 1 << 23;
        config.entry_point = 0;
        config.diagnostics = Some({
            let mut d = DiagnosticsConfig::new(std::path::PathBuf::from(path_sym));

            d.profiler_config = {
                let mut p =
                    ProfilerConfig::new(std::env::current_dir().unwrap().join("flamegraph.svg"));

                p.frequency_recip = 1;
                p.reverse_graph = false;

                Some(p)
            };

            d
        });

        let inner = VecDeque::<u32>::from(oracle_data);
        let oracle = VectorBasedNonDeterminismSource(inner, QuasiUARTSourceState::Ready);
        let output = run_simple_with_entry_point_and_non_determimism_source_for_config::<
            _,
            IWithoutByteAccessIsaConfigWithDelegation,
            // IMIsaConfigWithAllDelegations,
        >(config, oracle);
        dbg!(output.state);
    }
}

#[test]
fn test_query_values_offsets() {
    // Create a dummy QueryValuesInstance to test pointer arithmetic
    let dummy = MaybeUninit::<QueryValuesInstance>::uninit();
    let base_ptr = dummy.as_ptr().cast::<u32>();

    for (i, &offset_increment) in BASE_CIRCUIT_QUERY_VALUES_OFFSETS.iter().enumerate() {
        let current_ptr = unsafe { base_ptr.add(offset_increment) };

        match i {
            0 => {
                // After first offset, we should be at setup_leaf
                let expected_ptr = unsafe {
                    base_ptr.add(
                        offset_of!(QueryValuesInstance, setup_leaf) / core::mem::size_of::<u32>(),
                    )
                };
                assert_eq!(current_ptr, expected_ptr, "setup_leaf pointer mismatch");
            }
            idx if idx == LEAF_SIZE_SETUP => {
                // After setup_leaf elements, we should be at witness_leaf
                let expected_ptr = unsafe {
                    base_ptr.add(
                        offset_of!(QueryValuesInstance, witness_leaf) / core::mem::size_of::<u32>(),
                    )
                };
                assert_eq!(current_ptr, expected_ptr, "witness_leaf pointer mismatch");
            }
            idx if idx == LEAF_SIZE_SETUP + LEAF_SIZE_WITNESS_TREE => {
                // After witness_leaf elements, we should be at memory_leaf
                let expected_ptr = unsafe {
                    base_ptr.add(
                        offset_of!(QueryValuesInstance, memory_leaf) / core::mem::size_of::<u32>(),
                    )
                };
                assert_eq!(current_ptr, expected_ptr, "memory_leaf pointer mismatch");
            }
            idx if idx == LEAF_SIZE_SETUP + LEAF_SIZE_WITNESS_TREE + LEAF_SIZE_MEMORY_TREE => {
                // After memory_leaf elements, we should be at stage_2_leaf
                let expected_ptr = unsafe {
                    base_ptr.add(
                        offset_of!(QueryValuesInstance, stage_2_leaf) / core::mem::size_of::<u32>(),
                    )
                };
                assert_eq!(current_ptr, expected_ptr, "stage_2_leaf pointer mismatch");
            }
            idx if idx
                == LEAF_SIZE_SETUP
                    + LEAF_SIZE_WITNESS_TREE
                    + LEAF_SIZE_MEMORY_TREE
                    + LEAF_SIZE_STAGE_2 =>
            {
                // After stage_2_leaf elements, we should be at quotient_leaf
                let expected_ptr = unsafe {
                    base_ptr.add(
                        offset_of!(QueryValuesInstance, quotient_leaf)
                            / core::mem::size_of::<u32>(),
                    )
                };
                assert_eq!(current_ptr, expected_ptr, "quotient_leaf pointer mismatch");
            }
            idx if idx
                == LEAF_SIZE_SETUP
                    + LEAF_SIZE_WITNESS_TREE
                    + LEAF_SIZE_MEMORY_TREE
                    + LEAF_SIZE_STAGE_2
                    + LEAF_SIZE_QUOTIENT =>
            {
                // After quotient_leaf elements, we should be at fri_oracles_leafs
                let expected_ptr = unsafe {
                    base_ptr.add(
                        offset_of!(QueryValuesInstance, fri_oracles_leafs)
                            / core::mem::size_of::<u32>(),
                    )
                };
                assert_eq!(
                    current_ptr, expected_ptr,
                    "fri_oracles_leafs pointer mismatch"
                );
            }
            _ => {}
        }
    }
}

#[cfg(feature = "gkr_verify")]
#[path = "generated/gkr_verifier.rs"]
mod generated_gkr_inlined;

#[test]
#[cfg(feature = "gkr_verify")]
fn test_gkr_sumcheck_verify_inlined() {
    use field::baby_bear::base::BabyBearField;
    use field::baby_bear::ext4::BabyBearExt4;
    use prover::gkr::prover::GKRProof;
    use prover::merkle_trees::DefaultTreeConstructor;
    use verifier_common::cs::gkr_compiler::GKRCircuitArtifact;
    use verifier_common::gkr::flatten::flatten_gkr_proof_for_nds;
    use verifier_common::prover::nd_source_std::*;

    // to generate the proof and circuit run the gkr_run_basic_unrolled_test test
    let proof: GKRProof<BabyBearField, BabyBearExt4, DefaultTreeConstructor> =
        deserialize_from_file("../prover/add_sub_lui_auipc_mop_gkr_proof.json");
    let compiled_circuit: GKRCircuitArtifact<BabyBearField> =
        deserialize_from_file("../prover/add_sub_lui_auipc_mop_gkr_circuit.json");

    let oracle_data = flatten_gkr_proof_for_nds::<
        BabyBearField,
        BabyBearExt4,
        DefaultTreeConstructor,
    >(&proof, &compiled_circuit);

    let result = std::thread::Builder::new()
        .name("gkr verifier".to_string())
        .stack_size(1 << 27)
        .spawn(move || {
            set_iterator(oracle_data.into_iter());
            let result = generated_gkr_inlined::verify_gkr_sumcheck::<
                ThreadLocalBasedSource,
            >();
            match result {
                Ok(output) => {
                    println!("Verification complete");
                    println!(
                        "  base_layer_claims: {} entries, evaluation_point_len: {}, grand_product_accumulator: {:?}",
                        output.base_layer_claims.len(), output.evaluation_point_len,
                        output.grand_product_accumulator
                    );
                    assert!(
                        !output.additional_base_layer_openings.is_empty(),
                        "expected additional base layer openings for layer 0"
                    );
                    println!(
                        "  additional_base_layer_openings: {} addresses",
                        output.additional_base_layer_openings.len()
                    );
                }
                Err(e) => panic!("GKR verification failed: {:?}", e),
            }
        })
        .map(|t| t.join());

    match result {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            std::panic::resume_unwind(e);
        }
        Err(err) => {
            panic!("Failed to spawn verifier thread: {}", err);
        }
    }
}

#[test]
#[cfg(feature = "gkr_verify")]
fn test_gkr_sumcheck_verify_inlined_rejects_corrupted_proof() {
    use field::baby_bear::base::BabyBearField;
    use field::baby_bear::ext4::BabyBearExt4;
    use prover::gkr::prover::GKRProof;
    use prover::merkle_trees::DefaultTreeConstructor;
    use verifier_common::cs::gkr_compiler::GKRCircuitArtifact;
    use verifier_common::gkr::flatten::flatten_gkr_proof_for_nds;
    use verifier_common::prover::nd_source_std::*;

    // to generate the proof and circuit run the gkr_run_basic_unrolled_test test
    let proof: GKRProof<BabyBearField, BabyBearExt4, DefaultTreeConstructor> =
        deserialize_from_file("../prover/add_sub_lui_auipc_mop_gkr_proof.json");
    let compiled_circuit: GKRCircuitArtifact<BabyBearField> =
        deserialize_from_file("../prover/add_sub_lui_auipc_mop_gkr_circuit.json");

    let mut oracle_data = flatten_gkr_proof_for_nds::<
        BabyBearField,
        BabyBearExt4,
        DefaultTreeConstructor,
    >(&proof, &compiled_circuit);

    // Corrupt a word in the sumcheck coefficient region (past the transcript preamble and evaluations)
    let corrupt_idx = oracle_data.len() / 2;
    oracle_data[corrupt_idx] ^= 1;

    let result = std::thread::Builder::new()
        .name("gkr verifier corrupted".to_string())
        .stack_size(1 << 27)
        .spawn(move || {
            set_iterator(oracle_data.into_iter());
            generated_gkr_inlined::verify_gkr_sumcheck::<ThreadLocalBasedSource>()
        })
        .expect("failed to spawn thread")
        .join()
        .expect("verifier thread panicked");

    assert!(
        result.is_err(),
        "verifier should reject corrupted proof data"
    );
}

#[test]
#[cfg(feature = "gkr_verify")]
#[ignore = "requires RISC-V binary from tools/gkr_verifier"]
fn test_gkr_verifier_in_transpiler() {
    use field::baby_bear::base::BabyBearField;
    use field::baby_bear::ext4::BabyBearExt4;
    use prover::gkr::prover::GKRProof;
    use prover::merkle_trees::DefaultTreeConstructor;
    use risc_v_simulator::abstractions::non_determinism::QuasiUARTSource;
    use riscv_transpiler::ir::*;
    use riscv_transpiler::vm::*;
    use verifier_common::cs::gkr_compiler::GKRCircuitArtifact;
    use verifier_common::gkr::flatten::flatten_gkr_proof_for_nds;

    // to generate the proof and circuit run the gkr_run_basic_unrolled_test test
    let proof: GKRProof<BabyBearField, BabyBearExt4, DefaultTreeConstructor> =
        deserialize_from_file("../prover/add_sub_lui_auipc_mop_gkr_proof.json");
    let compiled_circuit: GKRCircuitArtifact<BabyBearField> =
        deserialize_from_file("../prover/add_sub_lui_auipc_mop_gkr_circuit.json");

    let oracle_data = flatten_gkr_proof_for_nds::<
        BabyBearField,
        BabyBearExt4,
        DefaultTreeConstructor,
    >(&proof, &compiled_circuit);

    println!("Oracle data length: {} u32 words", oracle_data.len());

    let binary_bytes = std::fs::read("../tools/gkr_verifier/app.bin")
        .expect("Missing app.bin — run `cd tools/gkr_verifier && ./dump_bin.sh` first");
    assert!(binary_bytes.len() % 4 == 0);
    let binary: Vec<u32> = binary_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let text_bytes = std::fs::read("../tools/gkr_verifier/app.text")
        .expect("Missing app.text — run `cd tools/gkr_verifier && ./dump_bin.sh` first");
    assert!(text_bytes.len() % 4 == 0);
    let text_section: Vec<u32> = text_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let instructions: Vec<Instruction> =
        preprocess_bytecode::<ReducedMachineDecoderConfig>(&text_section);
    let tape = SimpleTape::new(&instructions);
    let mut ram =
        RamWithRomRegion::<{ common_constants::rom::ROM_SECOND_WORD_BITS }>::from_rom_content(
            &binary,
            1 << 30,
        );

    let cycles_bound = 1 << 24;
    let mut state = State::initial_with_counters(DelegationsAndFamiliesCounters::default());
    let mut snapshotter = SimpleSnapshotter::<
        DelegationsAndFamiliesCounters,
        { common_constants::rom::ROM_SECOND_WORD_BITS },
    >::new_with_cycle_limit(cycles_bound, state);
    let mut non_determinism = QuasiUARTSource::new_with_reads(oracle_data);

    let symbols_path = std::path::PathBuf::from("../tools/gkr_verifier/app.elf");
    let output_path = std::env::current_dir().unwrap().join("gkr_flamegraph.svg");
    let mut fg_config =
        riscv_transpiler::vm::FlamegraphConfig::new(symbols_path, output_path.clone());
    fg_config.frequency_recip = 1; // sample every cycle for accuracy
    let mut profiler = riscv_transpiler::vm::VmFlamegraphProfiler::new(fg_config).unwrap();

    let is_program_finished =
        VM::<DelegationsAndFamiliesCounters>::run_basic_unrolled_with_flamegraph::<_, _, _>(
            &mut state,
            &mut ram,
            &mut snapshotter,
            &tape,
            cycles_bound,
            &mut non_determinism,
            &mut profiler,
        )
        .expect("flamegraph profiler IO error");

    assert!(
        is_program_finished,
        "GKR verifier program did not finish (PC stuck or cycle bound reached)"
    );

    let exact_cycles =
        (state.timestamp - common_constants::INITIAL_TIMESTAMP) / common_constants::TIMESTAMP_STEP;
    println!("GKR verifier finished in {} cycles", exact_cycles);

    println!("  PC = 0x{:08x}", state.pc);
    for (i, reg) in state.registers[10..18].iter().enumerate() {
        println!("  a{} = 0x{:08x} ({})", i, reg.value, reg.value);
    }

    let a0 = state.registers[10].value;
    if a0 == 0xDEAD {
        let error_code = state.registers[11].value;
        let layer = state.registers[12].value;
        let round = state.registers[13].value;
        match error_code {
            1 => panic!(
                "GKR SumcheckRoundFailed at layer={}, round={}",
                layer, round
            ),
            2 => panic!("GKR FinalStepCheckFailed at layer={}", layer),
            _ => panic!("GKR unknown error code={}", error_code),
        }
    }
    assert_eq!(
        a0, 1,
        "GKR verifier failed: a0 = {} (expected 1 for success)",
        a0
    );

    println!("GKR verifier completed successfully in transpiler");
    println!("Flamegraph written to {}", output_path.display());
}
