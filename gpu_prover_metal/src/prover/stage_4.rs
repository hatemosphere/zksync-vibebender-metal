//! Stage 4: DEEP quotient and FRI polynomial computation.
//! Ported from gpu_prover/src/prover/stage_4.rs.

use super::callbacks::Callbacks;
use super::context::{HostAllocation, ProverContext};
use super::setup::SetupPrecomputations;
use super::stage_1::StageOneOutput;
use super::stage_2::StageTwoOutput;
use super::stage_3::StageThreeOutput;
use super::stage_4_kernels::{
    compute_deep_denom_at_z_on_main_domain, compute_deep_quotient_on_main_domain,
    get_e4_scratch_count_for_deep_quotiening, get_metadata, ChallengesTimesEvals,
    NonWitnessChallengesAtZOmega,
};
use super::trace_holder::{TraceHolder, TreesCacheMode};
use super::{BF, E2, E4};
use crate::barycentric::{precompute_common_factor, precompute_lagrange_coeffs};
use crate::metal_runtime::MetalResult;
use crate::ops_complex::PowersLayerDesc;
use crate::prover::precomputations::PRECOMPUTATIONS;
use blake2s_u32::BLAKE2S_DIGEST_SIZE_U32_WORDS;
use cs::one_row_compiler::CompiledCircuitArtifact;
use field::{Field, FieldExtension};
use itertools::Itertools;
use prover::definitions::FoldingDescription;
use prover::prover_stages::cached_data::ProverCachedData;
use prover::prover_stages::Transcript;
use prover::transcript::Seed;
use std::sync::Arc;

pub(crate) struct StageFourOutput {
    pub(crate) trace_holder: TraceHolder<E4>,
    pub(crate) values_at_z: HostAllocation<[E4]>,
}

impl StageFourOutput {
    pub fn new(
        seed: &mut Seed,
        circuit: &Arc<CompiledCircuitArtifact<BF>>,
        cached_data: &ProverCachedData,
        setup: &mut SetupPrecomputations,
        stage_1_output: &mut StageOneOutput,
        stage_2_output: &mut StageTwoOutput,
        stage_3_output: &mut StageThreeOutput,
        log_lde_factor: u32,
        log_tree_cap_size: u32,
        folding_description: &FoldingDescription,
        callbacks: &mut Callbacks,
        context: &ProverContext,
    ) -> MetalResult<Self> {
        const COSET_INDEX: usize = 0;
        let trace_len = circuit.trace_len;
        assert!(trace_len.is_power_of_two());
        let log_domain_size = trace_len.trailing_zeros();
        let log_fold_by = folding_description.folding_sequence[0] as u32;
        let mut trace_holder = TraceHolder::new(
            log_domain_size,
            log_lde_factor,
            log_fold_by,
            log_tree_cap_size,
            1,
            false,
            true,
            false,
            TreesCacheMode::CacheFull,
            context,
        )?;

        let num_evals_at_z = circuit.num_openings_at_z();
        let num_evals_at_z_omega = circuit.num_openings_at_z_omega();
        let num_evals = num_evals_at_z + num_evals_at_z_omega;

        // Draw z challenge from transcript (synchronous on Metal)
        let _g = crate::cpu_scoped!("s4_challenge_and_alloc");
        let mut transcript_challenges =
            [0u32; (1usize * 4).next_multiple_of(BLAKE2S_DIGEST_SIZE_U32_WORDS)];
        Transcript::draw_randomness(seed, &mut transcript_challenges);
        let z_coeffs = transcript_challenges
            .as_chunks::<4>()
            .0
            .iter()
            .next()
            .unwrap()
            .map(BF::from_nonreduced_u32);
        let z = E4::from_coeffs_in_base(&z_coeffs);

        let coset = E2::ONE;
        let decompression_factor = E2::ONE;

        // Allocate GPU buffers
        let d_z = context.alloc_from_slice(std::slice::from_ref(&z))?;
        let d_common_factor = context.alloc::<E4>(1)?;
        let d_lagrange_coeffs = context.alloc::<E4>(trace_len)?;
        drop(_g);

        // Get coset evaluations for all traces (main domain, coset 0)
        let _g = crate::cpu_scoped!("s4_coset_evals");
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
        let d_composition_col = stage_3_output
            .trace_holder
            .get_coset_evaluations(COSET_INDEX, context)?;

        drop(_g);
        // Precompute Lagrange coefficients
        let cmd_buf = context.new_command_buffer()?;
        let device = context.device();
        let device_ctx = context.device_context();

        // w_inv_step must be omega^(-grid_size) where grid_size = block_dim * grid_dim
        // for the kernel's strided iteration over elements.
        let bary_block_dim = (32 * 4) as u32; // SIMD_GROUP_SIZE * 4
        let bary_inv_batch = 4u32;
        let bary_grid_dim =
            (trace_len as u32 + bary_inv_batch * bary_block_dim - 1) / (bary_inv_batch * bary_block_dim);
        let bary_grid_size = bary_block_dim * bary_grid_dim;
        let w = PRECOMPUTATIONS.omegas[log_domain_size as usize];
        let w_inv = w.inverse().expect("inverse of omega must exist");
        let w_inv_step = w_inv.pow(bary_grid_size);

        let powers_fine_desc = PowersLayerDesc {
            mask: (1u32 << device_ctx.fine_log_count) - 1,
            log_count: device_ctx.fine_log_count,
        };
        let powers_coarser_desc = PowersLayerDesc {
            mask: (1u32 << device_ctx.coarser_log_count) - 1,
            log_count: device_ctx.coarser_log_count,
        };
        let powers_coarsest_desc = PowersLayerDesc {
            mask: (1u32 << device_ctx.coarsest_log_count) - 1,
            log_count: device_ctx.coarsest_log_count,
        };

        precompute_common_factor(
            device,
            &cmd_buf,
            &d_z,
            &d_common_factor,
            coset,
            decompression_factor,
            trace_len as u32,
        )?;

        precompute_lagrange_coeffs(
            device,
            &cmd_buf,
            &d_z,
            &d_common_factor,
            w_inv_step,
            coset,
            &d_lagrange_coeffs,
            log_domain_size,
            &device_ctx.powers_of_w_fine,
            &powers_fine_desc,
            &device_ctx.powers_of_w_coarser,
            &powers_coarser_desc,
            &device_ctx.powers_of_w_coarsest,
            &powers_coarsest_desc,
        )?;

        // GPU-side barycentric evaluation matching CUDA batch_barycentric_eval.
        let mut d_evals = context.alloc::<E4>(num_evals)?;

        // No sync needed — Metal guarantees Lagrange precompute completes before
        // barycentric eval dispatches start (sequential within command buffer).
        {
            let n = trace_len as u32;
            let num_setup_cols = circuit.setup_layout.total_width;
            let num_witness_cols = circuit.witness_layout.total_width;
            let num_memory_cols = circuit.memory_layout.total_width;
            let stage_2_num_bf = circuit.stage_2_layout.num_base_field_polys();
            let stage_2_num_e4 = circuit.stage_2_layout.num_ext4_field_polys();
            let stage_2_e4_offset = circuit.stage_2_layout.ext4_polys_offset;

            // Identify columns needing z*omega evaluation
            let mut witness_z_omega_cols = vec![];
            for (_src, dst) in circuit.state_linkage_constraints.iter() {
                let cs::one_row_compiler::ColumnAddress::WitnessSubtree(idx) = *dst else { panic!() };
                witness_z_omega_cols.push(idx);
            }
            let mut memory_z_omega_cols = vec![];
            if let Some(shuffle) = circuit.memory_layout.shuffle_ram_inits_and_teardowns {
                let start = shuffle.lazy_init_addresses_columns.start();
                memory_z_omega_cols.push(start);
                memory_z_omega_cols.push(start + 1);
            }
            let offset_for_grand_product_poly = circuit
                .stage_2_layout
                .intermediate_polys_for_memory_argument
                .get_range(cached_data.offset_for_grand_product_accumulation_poly)
                .start;

            let num_partial_blocks = ((n + crate::barycentric::BARY_THREADS_PER_GROUP - 1)
                / crate::barycentric::BARY_THREADS_PER_GROUP) as usize;
            // Allocate partial buffer large enough for batched multi-column evals
            let max_batch_cols = num_setup_cols.max(num_witness_cols).max(num_memory_cols).max(stage_2_num_bf);
            let mut d_partial = context.alloc::<E4>(num_partial_blocks * max_batch_cols)?;
            let mut eval_idx = 0usize;

            // ALL barycentric evals (BF + E4 + z*omega) in ONE command buffer.
            let d_e4_temp = context.alloc::<E4>(trace_len)?;
            let d_shifted_lagrange = context.alloc::<E4>(trace_len)?;
            {

                // Setup BF columns - batched
                if num_setup_cols > 0 {
                    crate::barycentric::eval_bf_columns_batch(
                        device, &cmd_buf, &d_lagrange_coeffs, d_setup_cols,
                        0, num_setup_cols as u32, n,
                        &mut d_evals, eval_idx, &mut d_partial,
                    )?;
                    eval_idx += num_setup_cols;
                }
                // Witness BF columns - batched
                if num_witness_cols > 0 {
                    crate::barycentric::eval_bf_columns_batch(
                        device, &cmd_buf, &d_lagrange_coeffs, d_witness_cols,
                        0, num_witness_cols as u32, n,
                        &mut d_evals, eval_idx, &mut d_partial,
                    )?;
                    eval_idx += num_witness_cols;
                }
                // Memory BF columns - batched
                if num_memory_cols > 0 {
                    crate::barycentric::eval_bf_columns_batch(
                        device, &cmd_buf, &d_lagrange_coeffs, d_memory_cols,
                        0, num_memory_cols as u32, n,
                        &mut d_evals, eval_idx, &mut d_partial,
                    )?;
                    eval_idx += num_memory_cols;
                }
                // Stage 2 BF columns - batched
                if stage_2_num_bf > 0 {
                    crate::barycentric::eval_bf_columns_batch(
                        device, &cmd_buf, &d_lagrange_coeffs, d_stage_2_cols,
                        0, stage_2_num_bf as u32, n,
                        &mut d_evals, eval_idx, &mut d_partial,
                    )?;
                    eval_idx += stage_2_num_bf;
                }

                // Stage 2 E4 columns
                for col_e4 in 0..stage_2_num_e4 {
                    let bf_col = stage_2_e4_offset + col_e4 * 4;
                    crate::barycentric::eval_e4_column_at_z_batched(
                        device, &cmd_buf, &d_lagrange_coeffs, d_stage_2_cols,
                        bf_col, n, &mut d_evals, eval_idx, &mut d_partial, &d_e4_temp,
                    )?;
                    eval_idx += 1;
                }
                // Composition E4 column
                crate::barycentric::eval_e4_column_at_z_batched(
                    device, &cmd_buf, &d_lagrange_coeffs, d_composition_col,
                    0, n, &mut d_evals, eval_idx, &mut d_partial, &d_e4_temp,
                )?;
                eval_idx += 1;
                assert_eq!(eval_idx, num_evals_at_z);

                // z*omega evaluations: rotate + BF evals + grand product E4
                crate::ops_simple::rotate_right_e4(
                    device, &cmd_buf, &d_lagrange_coeffs, &d_shifted_lagrange,
                )?;
                for &col in witness_z_omega_cols.iter() {
                    crate::barycentric::eval_bf_column_at_z_batched(
                        device, &cmd_buf, &d_shifted_lagrange, d_witness_cols,
                        col * trace_len, n, &mut d_evals, eval_idx, &mut d_partial,
                    )?;
                    eval_idx += 1;
                }
                for &col in memory_z_omega_cols.iter() {
                    crate::barycentric::eval_bf_column_at_z_batched(
                        device, &cmd_buf, &d_shifted_lagrange, d_memory_cols,
                        col * trace_len, n, &mut d_evals, eval_idx, &mut d_partial,
                    )?;
                    eval_idx += 1;
                }
                // Grand product E4 at z*omega
                crate::barycentric::eval_e4_column_at_z_batched(
                    device, &cmd_buf, &d_shifted_lagrange, d_stage_2_cols,
                    offset_for_grand_product_poly, n, &mut d_evals, eval_idx, &mut d_partial, &d_e4_temp,
                )?;
                eval_idx += 1;

                cmd_buf.commit_and_wait();
            }
            assert_eq!(eval_idx, num_evals);

        }

        // Read back evals
        let _g = crate::cpu_scoped!("s4_evals_readback");
        let mut values_at_z = unsafe { HostAllocation::<[E4]>::new_uninit_slice(num_evals) };
        unsafe { d_evals.copy_to_slice(&mut *values_at_z) };
        drop(_g);

        // Draw alpha challenge from transcript (after committing eval results)
        let _g = crate::cpu_scoped!("s4_transcript_commit");
        let transcript_input = values_at_z
            .iter()
            .map(|el| el.into_coeffs_in_base())
            .flatten()
            .map(|el: BF| el.to_reduced_u32())
            .collect_vec();
        Transcript::commit_with_seed(seed, &transcript_input);
        let mut transcript_challenges =
            [0u32; (1usize * 4).next_multiple_of(BLAKE2S_DIGEST_SIZE_U32_WORDS)];
        Transcript::draw_randomness(seed, &mut transcript_challenges);
        let alpha_coeffs = transcript_challenges
            .as_chunks::<4>()
            .0
            .iter()
            .next()
            .unwrap()
            .map(BF::from_nonreduced_u32);
        let alpha = E4::from_coeffs_in_base(&alpha_coeffs);

        drop(_g);
        // Deep denom + quotient: dispatch denom, do CPU metadata work in parallel, then quotient.
        // Merged into ONE command buffer (saves 1 sync point).
        let d_denom_at_z = context.alloc::<E4>(trace_len)?;
        let cmd_buf = context.new_command_buffer()?;
        compute_deep_denom_at_z_on_main_domain(
            device,
            &cmd_buf,
            &d_denom_at_z,
            &d_z,
            log_domain_size,
            false,
            device_ctx,
        )?;
        // CPU metadata computation overlaps with GPU denom computation
        let _g = crate::cpu_scoped!("s4_deep_metadata");
        let omega_inv = PRECOMPUTATIONS.omegas_inv[log_domain_size as usize];
        let e4_scratch_elems = get_e4_scratch_count_for_deep_quotiening();
        let mut h_e4_scratch = vec![E4::ZERO; e4_scratch_elems];
        let mut h_challenges_times_evals = ChallengesTimesEvals::default();
        let mut h_non_witness_challenges_at_z_omega = NonWitnessChallengesAtZOmega::default();

        let metadata = get_metadata(
            &*values_at_z,
            alpha,
            omega_inv,
            cached_data,
            circuit,
            &mut h_e4_scratch,
            &mut h_challenges_times_evals,
            &mut h_non_witness_challenges_at_z_omega,
        );

        drop(_g);
        let _g = crate::cpu_scoped!("s4_deep_quotient_dispatch");
        let d_e4_scratch = context.alloc_from_slice(&h_e4_scratch)?;
        let d_challenges_times_evals =
            context.alloc_from_slice(std::slice::from_ref(&h_challenges_times_evals))?;
        let d_non_witness_challenges_at_z_omega =
            context.alloc_from_slice(std::slice::from_ref(&h_non_witness_challenges_at_z_omega))?;
        let d_witness_cols_to_challenges_at_z_omega_map =
            context.alloc_from_slice(std::slice::from_ref(
                &metadata.witness_cols_to_challenges_at_z_omega_map,
            ))?;

        let d_quotient_bf = context.alloc::<BF>(4 * trace_len)?;
        let stride = trace_len as u32;
        // Deep quotient dispatched on SAME cmd_buf — Metal ensures denom completes first
        compute_deep_quotient_on_main_domain(
            device,
            &cmd_buf,
            metadata,
            d_setup_cols,
            stride,
            d_witness_cols,
            stride,
            d_memory_cols,
            stride,
            d_stage_2_cols,
            stride,
            d_composition_col,
            stride,
            &d_denom_at_z,
            &d_e4_scratch,
            &d_challenges_times_evals,
            &d_non_witness_challenges_at_z_omega,
            &d_witness_cols_to_challenges_at_z_omega_map,
            &d_quotient_bf,
            stride,
            cached_data,
            circuit,
            log_domain_size,
            false,
        )?;
        // NTT extension dispatched on SAME cmd_buf as deep quotient — Metal sequential
        // encoding ensures quotient data is available before NTT reads it.
        assert_eq!(log_lde_factor, 1);
        let d_quotient_bf_coset1 = context.alloc::<BF>(4 * trace_len)?;
        crate::ntt::natural_trace_main_evals_to_bitrev_Z(
            device,
            &cmd_buf,
            &d_quotient_bf,
            &d_quotient_bf_coset1,
            trace_len as u32,
            log_domain_size,
            4, // num_bf_cols
            context.ntt_twiddles(),
        )?;
        {
            let const_dst =
                unsafe { &*(&d_quotient_bf_coset1 as *const crate::metal_runtime::MetalBuffer<BF>) };
            crate::ntt::bitrev_Z_to_natural_trace_coset_evals(
                device,
                &cmd_buf,
                const_dst,
                &d_quotient_bf_coset1,
                trace_len as u32,
                log_domain_size,
                4,
                context.ntt_twiddles(),
            )?;
        }
        cmd_buf.commit_and_wait();
        drop(_g);

        // For each coset: transpose BF column-major → E4 row-major, bit-reverse, build tree
        let coset_bf_bufs = [&d_quotient_bf, &d_quotient_bf_coset1];
        let log_fold_by = folding_description.folding_sequence[0] as u32;
        let log_coset_tree_cap_size = log_tree_cap_size - log_lde_factor;
        let layers_count = log_domain_size + 1 - log_fold_by - log_coset_tree_cap_size;
        let num_leaves = 1usize << (log_domain_size - log_fold_by);
        let tree_len = num_leaves * 2;

        let mut tree_caps = super::trace_holder::allocate_tree_caps(log_lde_factor, log_tree_cap_size);

        let _g = crate::cpu_scoped!("s4_coset_trees");
        // Batch BOTH cosets into a single command buffer (saves 1 GPU sync point)
        {
            let (coset0, coset1) = match &mut trace_holder.cosets {
                super::trace_holder::CosetsHolder::Full(evals) => {
                    let (a, b) = evals.split_at_mut(1);
                    (&mut a[0], &mut b[0])
                }
                _ => unreachable!(),
            };
            let (tree0, tree1) = match &mut trace_holder.trees {
                super::trace_holder::TreesHolder::Full(trees) => {
                    let (a, b) = trees.split_at_mut(1);
                    (&mut a[0], &mut b[0])
                }
                super::trace_holder::TreesHolder::Partial(trees) => {
                    let (a, b) = trees.split_at_mut(1);
                    (&mut a[0], &mut b[0])
                }
                _ => unreachable!(),
            };
            assert_eq!(tree0.len(), tree_len);
            assert_eq!(tree1.len(), tree_len);

            let cmd_buf = context.new_command_buffer()?;
            let e4_stride = trace_len as u32;
            let digest_size = std::mem::size_of::<crate::blake2s::Digest>();

            // Encode both cosets into the same command buffer
            macro_rules! encode_coset {
                ($coset_idx:expr, $d_coset_e4:expr, $tree:expr) => {{
                    crate::ops_simple::transpose_bf4_to_e4(
                        device, &cmd_buf,
                        coset_bf_bufs[$coset_idx].raw(),
                        $d_coset_e4.raw(),
                        trace_len as u32,
                    )?;
                    crate::ops_complex::bit_reverse_naive_e4(
                        device, &cmd_buf,
                        $d_coset_e4, e4_stride,
                        $d_coset_e4, e4_stride,
                        log_domain_size, 1,
                    )?;
                    let bf_len = $d_coset_e4.len() * 4;
                    crate::blake2s::build_merkle_tree_leaves_raw(
                        device, &cmd_buf, $d_coset_e4.raw(), bf_len,
                        $tree, num_leaves, log_fold_by + 2,
                    )?;
                    if layers_count > 0 {
                        crate::blake2s::build_merkle_tree_nodes_with_offset(
                            device, &cmd_buf, $tree, 0, num_leaves,
                            $tree, num_leaves * digest_size, layers_count - 1,
                        )?;
                    }
                }};
            }
            encode_coset!(0, coset0, tree0);
            encode_coset!(1, coset1, tree1);
            cmd_buf.commit_and_wait();

            super::trace_holder::transfer_tree_cap(
                tree0, &mut tree_caps[0], log_lde_factor, log_tree_cap_size,
            );
            super::trace_holder::transfer_tree_cap(
                tree1, &mut tree_caps[1], log_lde_factor, log_tree_cap_size,
            );
        }
        drop(_g);
        trace_holder.tree_caps = Some(tree_caps);

        let update_seed_fn = trace_holder.get_update_seed_fn(seed);
        callbacks.schedule(update_seed_fn);

        Ok(Self {
            trace_holder,
            values_at_z,
        })
    }
}
