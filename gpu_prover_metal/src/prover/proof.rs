//! Proof orchestration for Metal prover.
//! Ported from gpu_prover/src/prover/proof.rs.
//!
//! On Metal, proof generation is synchronous (no async event-based job tracking).
//! The CUDA version returns a ProofJob that can be polled; here we compute directly.
//!
//! The prove() function orchestrates stages 1-5, PoW, and queries, then assembles
//! the final proof. Stage bodies are wired through as their implementations mature.

use super::callbacks::Callbacks;
use super::context::ProverContext;
use super::pow::PowOutput;
use super::queries::QueriesOutput;
use super::setup::SetupPrecomputations;
use super::stage_1::StageOneOutput;
use super::stage_2::StageTwoOutput;
use super::stage_3::StageThreeOutput;
use super::stage_4::StageFourOutput;
use super::stage_5::StageFiveOutput;
use super::trace_holder::{flatten_tree_caps, get_tree_caps, TreesCacheMode};
use super::{BF, E2, E4};
use crate::metal_runtime::MetalResult;
use cs::one_row_compiler::CompiledCircuitArtifact;
use fft::LdePrecomputations;
use field::Mersenne31Field;
use itertools::Itertools;

fn canonicalize_e4(val: E4) -> E4 {
    E4 {
        c0: E2 {
            c0: Mersenne31Field(val.c0.c0.to_reduced_u32()),
            c1: Mersenne31Field(val.c0.c1.to_reduced_u32()),
        },
        c1: E2 {
            c0: Mersenne31Field(val.c1.c0.to_reduced_u32()),
            c1: Mersenne31Field(val.c1.c1.to_reduced_u32()),
        },
    }
}
use prover::definitions::{ExternalValues, Transcript, OPTIMAL_FOLDING_PROPERTIES};
use prover::prover_stages::cached_data::ProverCachedData;
use prover::prover_stages::Proof;
use prover::transcript::Seed;
use std::sync::Arc;

pub struct ProofJob<'a> {
    proof: Option<Proof>,
    _lifetime: std::marker::PhantomData<&'a ()>,
}

impl<'a> ProofJob<'a> {
    pub fn is_finished(&self) -> bool {
        true
    }

    pub fn finish(mut self) -> MetalResult<(Proof, f32)> {
        let proof = self.proof.take().unwrap();
        Ok((proof, 0.0))
    }
}

/// Main prove orchestration: runs stages 1-5, PoW, queries, and assembles the proof.
pub fn prove<'a>(
    circuit: Arc<CompiledCircuitArtifact<BF>>,
    external_values: ExternalValues,
    setup: &mut SetupPrecomputations,
    tracing_data_transfer: super::tracing_data::TracingDataTransfer<'a, impl fft::GoodAllocator>,
    lde_precomputations: &LdePrecomputations<impl fft::GoodAllocator>,
    circuit_sequence: usize,
    delegation_processing_type: Option<u16>,
    lde_factor: usize,
    num_queries: usize,
    pow_bits: u32,
    external_pow_nonce: Option<u64>,
    recompute_cosets: bool,
    trees_cache_mode: TreesCacheMode,
    context: &ProverContext,
) -> MetalResult<ProofJob<'a>> {
    #[cfg(feature = "log_gpu_stages_timings")]
    crate::metal_runtime::profiler::init();

    let trace_len = circuit.trace_len;
    assert!(trace_len.is_power_of_two());
    let log_domain_size = trace_len.trailing_zeros();
    let optimal_folding = OPTIMAL_FOLDING_PROPERTIES[log_domain_size as usize];
    assert!(circuit_sequence <= u16::MAX as usize);
    let delegation_processing_type = delegation_processing_type.unwrap_or_default();
    let cached_data_values = ProverCachedData::new(
        &circuit,
        &external_values,
        trace_len,
        circuit_sequence,
        delegation_processing_type,
    );
    assert!(lde_factor.is_power_of_two());
    let log_lde_factor = lde_factor.trailing_zeros();
    let log_tree_cap_size = optimal_folding.total_caps_size_log2 as u32;
    let mut callbacks = Callbacks::new();

    // Setup
    log::info!("prove: starting setup extension");
    crate::cpu_span!("setup_extension");
    setup.ensure_is_extended(context)?;
    log::info!("prove: setup extended");

    // Stage 1: allocate trace holders
    crate::cpu_span!("stage1_alloc");
    log::info!("prove: allocating stage 1 trace holders");
    let mut stage_1_output = StageOneOutput::allocate_trace_holders(
        &circuit,
        log_lde_factor,
        log_tree_cap_size,
        recompute_cosets,
        trees_cache_mode,
        context,
    )?;

    // Stage 2: allocate trace evaluations
    let mut stage_2_output = StageTwoOutput::allocate_trace_evaluations(
        &circuit,
        log_lde_factor,
        log_tree_cap_size,
        recompute_cosets,
        trees_cache_mode,
        context,
    )?;

    // Stage 1: witness generation
    crate::cpu_span!("stage1_witness");
    log::info!("prove: generating witness (stage 1)");
    let t = std::time::Instant::now();
    stage_1_output.generate_witness(
        &circuit,
        setup,
        tracing_data_transfer,
        circuit_sequence,
        context,
    )?;
    log::info!("prove: stage 1 witness done in {:.1}s", t.elapsed().as_secs_f64());

    // Stage 1: commit witness
    crate::cpu_span!("stage1_commit");
    let t = std::time::Instant::now();
    stage_1_output.commit_witness(&circuit, &mut callbacks, context)?;
    let mem_caps = stage_1_output.memory_holder.get_tree_caps_accessors();
    let wit_caps = stage_1_output.witness_holder.get_tree_caps_accessors();
    log::info!("prove: stage 1 commit done in {:.1}s (memory_caps={}, witness_caps={})",
        t.elapsed().as_secs_f64(), mem_caps.len(), wit_caps.len());

    // Seed initialization
    crate::cpu_span!("seed_init");
    let _g_seed = crate::cpu_scoped!("seed_init_detail");
    let mut seed = initialize_seed(
        &circuit,
        external_values.clone(),
        circuit_sequence,
        delegation_processing_type,
        setup,
        &stage_1_output,
    );

    drop(_g_seed);

    // Stage 2: argument polynomial computation
    crate::cpu_span!("stage2");
    let t = std::time::Instant::now();
    stage_2_output.generate(
        &mut seed,
        &circuit,
        &cached_data_values,
        setup,
        &mut stage_1_output,
        &mut callbacks,
        context,
    )?;

    log::info!("prove: stage 2 done in {:.1}s", t.elapsed().as_secs_f64());

    // Stage 3: constraint quotient computation
    crate::cpu_span!("stage3");
    let t = std::time::Instant::now();
    let mut stage_3_output = StageThreeOutput::new(
        &mut seed,
        &circuit,
        &cached_data_values,
        lde_precomputations,
        external_values.clone(),
        setup,
        &mut stage_1_output,
        &mut stage_2_output,
        log_lde_factor,
        log_tree_cap_size,
        trees_cache_mode,
        &mut callbacks,
        context,
    )?;

    log::info!("prove: stage 3 done in {:.1}s", t.elapsed().as_secs_f64());

    // Stage 4: DEEP quotient + FRI polynomial
    crate::cpu_span!("stage4");
    let t = std::time::Instant::now();
    let mut stage_4_output = StageFourOutput::new(
        &mut seed,
        &circuit,
        &cached_data_values,
        setup,
        &mut stage_1_output,
        &mut stage_2_output,
        &mut stage_3_output,
        log_lde_factor,
        log_tree_cap_size,
        &optimal_folding,
        &mut callbacks,
        context,
    )?;

    log::info!("prove: stage 4 done in {:.1}s", t.elapsed().as_secs_f64());

    // Stage 5: FRI folding
    crate::cpu_span!("stage5");
    let t = std::time::Instant::now();
    let stage_5_output = StageFiveOutput::new(
        &mut seed,
        &mut stage_4_output,
        log_domain_size,
        log_lde_factor,
        &optimal_folding,
        num_queries,
        lde_precomputations,
        &mut callbacks,
        context,
    )?;

    log::info!("prove: stage 5 done in {:.1}s", t.elapsed().as_secs_f64());

    // PoW
    crate::cpu_span!("pow");
    log::info!("prove: computing PoW");
    let t = std::time::Instant::now();
    let pow_output = PowOutput::new(&mut seed, pow_bits, external_pow_nonce, context)?;
    log::info!("prove: PoW done in {:.1}s", t.elapsed().as_secs_f64());

    // Queries
    crate::cpu_span!("queries");
    log::info!("prove: computing queries (heavy GPU phase)");
    let t = std::time::Instant::now();
    let queries_output = QueriesOutput::new(
        &mut seed,
        setup,
        &mut stage_1_output,
        &mut stage_2_output,
        &mut stage_3_output,
        &mut stage_4_output,
        &stage_5_output,
        log_domain_size,
        log_lde_factor,
        num_queries,
        &optimal_folding,
        context,
    )?;

    log::info!("prove: queries done in {:.1}s", t.elapsed().as_secs_f64());

    // Assemble proof
    crate::cpu_span!("proof_assembly");
    log::info!("prove: assembling proof");
    let proof = create_proof(
        external_values,
        circuit_sequence,
        delegation_processing_type,
        setup,
        stage_1_output,
        stage_2_output,
        stage_3_output,
        stage_4_output,
        stage_5_output,
        pow_output,
        queries_output,
    );

    #[cfg(feature = "log_gpu_stages_timings")]
    {
        crate::metal_runtime::profiler::cpu_mark_end();
        crate::metal_runtime::profiler::finish_and_report();
    }

    Ok(ProofJob {
        proof: Some(proof),
        _lifetime: std::marker::PhantomData,
    })
}

fn initialize_seed(
    circuit: &Arc<CompiledCircuitArtifact<Mersenne31Field>>,
    external_values: ExternalValues,
    circuit_sequence: usize,
    delegation_processing_type: u16,
    setup: &SetupPrecomputations,
    stage_1_output: &StageOneOutput,
) -> Seed {
    let public_inputs = stage_1_output
        .public_inputs
        .as_ref()
        .expect("public_inputs not set");
    let public_inputs_accessor = public_inputs.get_accessor();
    let public_inputs_slice = unsafe { public_inputs_accessor.get() };
    let setup_tree_caps = setup
        .trees_and_caps
        .caps
        .iter()
        .flat_map(|c| &c.cap)
        .copied()
        .flatten()
        .collect_vec();

    let witness_tree_caps = stage_1_output
        .witness_holder
        .get_tree_caps_accessors();
    let memory_tree_caps = stage_1_output
        .memory_holder
        .get_tree_caps_accessors();

    let mut input = vec![];
    input.push(circuit_sequence as u32);
    input.push(delegation_processing_type as u32);
    input.extend(public_inputs_slice.iter().map(BF::to_reduced_u32));
    input.extend(setup_tree_caps);
    input.extend_from_slice(&external_values.challenges.memory_argument.flatten());
    if let Some(delegation_argument_challenges) =
        external_values.challenges.delegation_argument.as_ref()
    {
        input.extend_from_slice(&delegation_argument_challenges.flatten());
    }
    if circuit
        .memory_layout
        .shuffle_ram_inits_and_teardowns
        .is_some()
    {
        input.extend_from_slice(&external_values.aux_boundary_values.flatten());
    }
    input.extend(flatten_tree_caps(&witness_tree_caps));
    input.extend(flatten_tree_caps(&memory_tree_caps));

    Transcript::commit_initial(&input)
}

fn create_proof(
    external_values: ExternalValues,
    circuit_sequence: usize,
    delegation_processing_type: u16,
    setup: &SetupPrecomputations,
    stage_1_output: StageOneOutput,
    stage_2_output: StageTwoOutput,
    stage_3_output: StageThreeOutput,
    stage_4_output: StageFourOutput,
    stage_5_output: StageFiveOutput,
    pow_output: PowOutput,
    queries_output: QueriesOutput,
) -> Proof {
    let public_inputs = match stage_1_output.public_inputs.as_ref() {
        Some(pi) => unsafe { pi.get_accessor().get().to_vec() },
        None => vec![],
    };

    let witness_tree_caps =
        get_tree_caps(&stage_1_output.witness_holder.get_tree_caps_accessors());
    let memory_tree_caps =
        get_tree_caps(&stage_1_output.memory_holder.get_tree_caps_accessors());
    let setup_tree_caps = setup.trees_and_caps.caps.as_ref().clone();
    let stage_2_tree_caps =
        get_tree_caps(&stage_2_output.trace_holder.get_tree_caps_accessors());

    let memory_grand_product_accumulator = stage_2_output.grand_product_accumulator;
    let delegation_argument_accumulator = stage_2_output.delegation_argument_accumulator;

    let quotient_tree_caps =
        get_tree_caps(&stage_3_output.trace_holder.get_tree_caps_accessors());
    let evaluations_at_random_points: Vec<_> =
        unsafe { stage_4_output.values_at_z.get_accessor().get() }
            .iter()
            .map(|v| canonicalize_e4(*v))
            .collect();
    let deep_poly_caps =
        get_tree_caps(&stage_4_output.trace_holder.get_tree_caps_accessors());

    let intermediate_fri_oracle_caps = stage_5_output
        .fri_oracles
        .iter()
        .filter(|s| !s.tree_caps.is_empty())
        .map(|s| s.get_tree_caps_accessors())
        .map(|a| get_tree_caps(&a))
        .collect_vec();

    let last_fri_step_plain_leaf_values: Vec<Vec<_>> = stage_5_output
        .last_fri_step_plain_leaf_values
        .iter()
        .map(|h| unsafe { h.get_accessor().get() }.iter().map(|v| canonicalize_e4(*v)).collect())
        .collect();

    let final_monomial_form: Vec<_> =
        unsafe { stage_5_output.final_monomials.get_accessor().get() }
            .iter()
            .map(|v| canonicalize_e4(*v))
            .collect();

    let queries = queries_output.into_query_sets();
    let pow_nonce = pow_output.nonce;
    let circuit_sequence = circuit_sequence as u16;
    let delegation_type = delegation_processing_type;

    Proof {
        external_values,
        public_inputs,
        witness_tree_caps,
        memory_tree_caps,
        setup_tree_caps,
        stage_2_tree_caps,
        memory_grand_product_accumulator,
        delegation_argument_accumulator,
        quotient_tree_caps,
        evaluations_at_random_points,
        deep_poly_caps,
        intermediate_fri_oracle_caps,
        last_fri_step_plain_leaf_values,
        final_monomial_form,
        queries,
        pow_nonce,
        circuit_sequence,
        delegation_type,
    }
}
