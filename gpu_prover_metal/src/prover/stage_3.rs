//! Stage 3: constraint quotient computation.
//! Ported from gpu_prover/src/prover/stage_3.rs.

use super::arg_utils::{ConstantsTimesChallenges, LookupChallenges};
use super::callbacks::Callbacks;
use super::context::ProverContext;
use super::setup::SetupPrecomputations;
use super::stage_1::StageOneOutput;
use super::stage_2::StageTwoOutput;
use super::stage_3_kernels::*;
use super::trace_holder::{TraceHolder, TreesCacheMode};
use super::{BF, E4};
use crate::prover::precomputations::PRECOMPUTATIONS;
use blake2s_u32::BLAKE2S_DIGEST_SIZE_U32_WORDS;
use cs::one_row_compiler::CompiledCircuitArtifact;
use fft::{materialize_powers_serial_starting_with_one, GoodAllocator, LdePrecomputations};
use field::FieldExtension;
use prover::definitions::ExternalValues;
use prover::prover_stages::cached_data::ProverCachedData;
use prover::prover_stages::stage3::AlphaPowersLayout;
use prover::prover_stages::Transcript;
use prover::transcript::Seed;
use std::alloc::Global;
use std::sync::Arc;

use crate::metal_runtime::MetalResult;

pub(crate) struct StageThreeOutput {
    pub(crate) trace_holder: TraceHolder<BF>,
}

impl StageThreeOutput {
    pub fn new(
        seed: &mut Seed,
        circuit: &Arc<CompiledCircuitArtifact<BF>>,
        cached_data: &ProverCachedData,
        lde_precomputations: &LdePrecomputations<impl GoodAllocator>,
        external_values: ExternalValues,
        setup: &mut SetupPrecomputations,
        stage_1_output: &mut StageOneOutput,
        stage_2_output: &mut StageTwoOutput,
        log_lde_factor: u32,
        log_tree_cap_size: u32,
        trees_cache_mode: TreesCacheMode,
        callbacks: &mut Callbacks,
        context: &ProverContext,
    ) -> MetalResult<Self> {
        const COSET_INDEX: usize = 1;
        let trace_len = circuit.trace_len;
        assert!(trace_len.is_power_of_two());
        let log_domain_size = trace_len.trailing_zeros();
        let mut trace_holder = TraceHolder::new(
            log_domain_size,
            log_lde_factor,
            0,
            log_tree_cap_size,
            4,
            true,
            false,
            false,
            trees_cache_mode,
            context,
        )?;

        let alpha_powers_layout =
            AlphaPowersLayout::new(&circuit, cached_data.num_stage_3_quotient_terms);
        let alpha_powers_count = alpha_powers_layout.precomputation_size;
        let tau = lde_precomputations.domain_bound_precomputations[COSET_INDEX]
            .as_ref()
            .unwrap()
            .coset_offset;

        // Draw challenges from transcript
        let _g = crate::cpu_scoped!("s3_challenges_and_powers");
        let mut transcript_challenges =
            [0u32; (2usize * 4).next_multiple_of(BLAKE2S_DIGEST_SIZE_U32_WORDS)];
        Transcript::draw_randomness(seed, &mut transcript_challenges);
        let mut it = transcript_challenges.as_chunks::<4>().0.iter();
        let mut get_challenge =
            || E4::from_coeffs_in_base(&it.next().unwrap().map(BF::from_nonreduced_u32));
        let alpha = get_challenge();
        let beta = get_challenge();

        let mut alpha_powers =
            materialize_powers_serial_starting_with_one::<_, Global>(alpha, alpha_powers_count);
        alpha_powers.reverse();
        let beta_powers =
            materialize_powers_serial_starting_with_one::<_, Global>(beta, BETA_POWERS_COUNT);
        drop(_g);

        // Get stage 2 data
        let grand_product_accumulator = stage_2_output.grand_product_accumulator;
        let sum_over_delegation_poly = stage_2_output
            .delegation_argument_accumulator
            .unwrap_or_default();

        let lookup_challenges_ref: &LookupChallenges = &stage_2_output
            .lookup_challenges
            .as_ref()
            .unwrap()
            .0;
        let public_inputs_ref: &[BF] = &stage_1_output.public_inputs.as_ref().unwrap().0;

        let omega_index = log_domain_size as usize;
        let omega = PRECOMPUTATIONS.omegas[omega_index];
        let omega_inv = PRECOMPUTATIONS.omegas_inv[omega_index];

        let _g = crate::cpu_scoped!("s3_metadata");
        let mut helpers = Vec::with_capacity(MAX_HELPER_VALUES);
        let mut constants_times_challenges = ConstantsTimesChallenges::default();

        let metadata = Metadata::new(
            &alpha_powers,
            &beta_powers,
            tau,
            omega,
            omega_inv,
            lookup_challenges_ref,
            cached_data,
            &circuit,
            &external_values,
            public_inputs_ref,
            grand_product_accumulator,
            sum_over_delegation_poly,
            log_domain_size,
            &mut helpers,
            &mut constants_times_challenges,
        );

        drop(_g);
        // Upload data to GPU buffers
        let _g = crate::cpu_scoped!("s3_buffer_upload");
        let d_alpha_powers = context.alloc_from_slice(&alpha_powers)?;
        let d_beta_powers = context.alloc_from_slice(&beta_powers)?;
        let d_helpers = context.alloc_from_slice(&helpers)?;
        let d_constants_times_challenges =
            context.alloc_from_slice(std::slice::from_ref(&constants_times_challenges))?;

        drop(_g);
        // Get coset evaluations for all traces
        let _g = crate::cpu_scoped!("s3_coset_evals");
        let d_setup_cols = setup
            .trace_holder
            .get_coset_evaluations(COSET_INDEX, context)?;
        let d_witness_cols = stage_1_output
            .witness_holder
            .get_coset_evaluations(COSET_INDEX, context)?;
        let d_memory_cols = stage_1_output
            .memory_holder
            .get_coset_evaluations(COSET_INDEX, context)?;
        let d_stage_2_cols = stage_2_output
            .trace_holder
            .get_coset_evaluations(COSET_INDEX, context)?;

        let d_quotient = trace_holder.get_uninit_coset_evaluations_mut(COSET_INDEX);
        drop(_g);

        let cmd_buf = context.new_command_buffer()?;
        {
        let _g = crate::cpu_scoped!("s3_quotient_dispatch");
        compute_stage_3_composition_quotient_on_coset_into(
            &cmd_buf,
            cached_data,
            &circuit,
            metadata,
            d_setup_cols,
            d_witness_cols,
            d_memory_cols,
            d_stage_2_cols,
            &d_alpha_powers,
            &d_beta_powers,
            &d_helpers,
            &d_constants_times_challenges,
            d_quotient,
            log_domain_size,
            context,
        )?;
        }
        {
        let _g = crate::cpu_scoped!("s3_extend_commit");
        let batched = trace_holder.extend_and_commit_into(COSET_INDEX, &cmd_buf, context)?;
        cmd_buf.commit_and_wait();
        if batched {
            trace_holder.transfer_existing_tree_caps();
        } else {
            trace_holder.extend_and_commit(COSET_INDEX, context)?;
        }
        }

        let update_seed_fn = trace_holder.get_update_seed_fn(seed);
        callbacks.schedule(update_seed_fn);

        Ok(Self { trace_holder })
    }
}
