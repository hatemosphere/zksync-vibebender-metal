//! Stage 5: FRI folding and final monomial computation.
//! Ported from gpu_prover/src/prover/stage_5.rs.

use super::callbacks::Callbacks;
use super::context::{HostAllocation, ProverContext, UnsafeAccessor};
use super::stage_4::StageFourOutput;
use super::trace_holder::{allocate_tree_caps, flatten_tree_caps, transfer_tree_cap, CosetsHolder};
use super::{BF, E2, E4};
use crate::blake2s::{self, Digest};
use crate::metal_runtime::{MetalBuffer, MetalResult};
use crate::ops_complex::{self, PowersLayerDesc};
use crate::prover::precomputations::PRECOMPUTATIONS;
use blake2s_u32::BLAKE2S_DIGEST_SIZE_U32_WORDS;
use fft::{
    bitreverse_enumeration_inplace, partial_ifft_natural_to_natural, GoodAllocator,
    LdePrecomputations,
};
use field::{Field, FieldExtension, Mersenne31Field};
use itertools::Itertools;
use prover::definitions::{FoldingDescription, Transcript};
use prover::transcript::Seed;
use std::iter;

pub(crate) struct FRIStep {
    pub ldes: Vec<MetalBuffer<E4>>,
    pub trees: Vec<MetalBuffer<Digest>>,
    pub tree_caps: Vec<HostAllocation<[Digest]>>,
}

impl FRIStep {
    pub fn get_tree_caps_accessors(&self) -> Vec<UnsafeAccessor<[Digest]>> {
        self.tree_caps
            .iter()
            .map(|h| <HostAllocation<[Digest]>>::get_accessor(h))
            .collect_vec()
    }
}

pub(crate) struct StageFiveOutput {
    pub(crate) fri_oracles: Vec<FRIStep>,
    pub(crate) last_fri_step_plain_leaf_values: Vec<HostAllocation<[E4]>>,
    pub(crate) final_monomials: HostAllocation<[E4]>,
}

impl StageFiveOutput {
    pub fn new(
        seed: &mut Seed,
        stage_4_output: &mut StageFourOutput,
        log_domain_size: u32,
        log_lde_factor: u32,
        folding_description: &FoldingDescription,
        num_queries: usize,
        lde_precomputations: &LdePrecomputations<impl GoodAllocator>,
        _callbacks: &mut Callbacks<'_>,
        context: &ProverContext,
    ) -> MetalResult<Self> {
        assert_eq!(log_domain_size, stage_4_output.trace_holder.log_domain_size);
        let log_tree_cap_size = folding_description.total_caps_size_log2 as u32;
        let lde_factor = 1usize << log_lde_factor;
        let mut log_current_domain_size = log_domain_size;
        let oracles_count = folding_description.folding_sequence.len() - 1;
        let mut fri_oracles: Vec<FRIStep> = vec![];
        let mut last_fri_step_plain_leaf_values: Vec<HostAllocation<[E4]>> = Default::default();

        // Get taus from LDE precomputations
        let mut taus = lde_precomputations.domain_bound_precomputations[0]
            .as_ref()
            .unwrap()
            .taus
            .clone();

        for (i, &current_log_fold) in folding_description
            .folding_sequence
            .iter()
            .take(oracles_count)
            .enumerate()
        {
            let folding_degree_log2 = current_log_fold as u32;
            let log_folded_domain_size = log_current_domain_size - folding_degree_log2;
            let next_log_fold = folding_description.folding_sequence[i + 1] as u32;
            let log_num_leafs = log_folded_domain_size - next_log_fold;

            let mut ldes = Vec::with_capacity(lde_factor);
            for _ in 0..lde_factor {
                ldes.push(context.alloc::<E4>(1 << log_folded_domain_size)?);
            }

            let folding_inputs = if i == 0 {
                match &stage_4_output.trace_holder.cosets {
                    CosetsHolder::Full(evaluations) => evaluations,
                    CosetsHolder::Single { .. } => unreachable!(),
                }
            } else {
                &fri_oracles[i - 1].ldes
            };

            // Compute folding challenges
            let challenges_len = lde_factor * current_log_fold;
            let mut challenges = vec![E4::default(); challenges_len];
            Self::set_folding_challenges(
                seed,
                &mut taus,
                &mut challenges,
                current_log_fold,
            );

            // Upload challenges to device
            let d_challenges = context.alloc_from_slice(&challenges)?;

            // Batch all coset folds into ONE command buffer
            {
                let cmd_buf = context.new_command_buffer()?;
                for (coset_idx, (folding_input, folding_output)) in folding_inputs
                    .iter()
                    .zip(ldes.iter_mut())
                    .enumerate()
                {
                    let challenge_offset = coset_idx * current_log_fold;
                    Self::fold_coset_dispatch(
                        &cmd_buf,
                        folding_degree_log2,
                        &d_challenges,
                        challenge_offset,
                        folding_input,
                        folding_output,
                        context,
                    )?;
                }
                cmd_buf.commit_and_wait();
            }

            let expose_all_leafs = if i == oracles_count - 1 {
                let log_bound = num_queries.next_power_of_two().trailing_zeros();
                log_num_leafs + 1 - log_lde_factor <= log_bound
            } else {
                false
            };

            let (trees, tree_caps) = if expose_all_leafs {
                // Copy LDE values to host for transcript commitment
                let mut leaf_values = vec![];
                for d_coset in ldes.iter() {
                    let len = d_coset.len();
                    let mut h_coset = unsafe { HostAllocation::<[E4]>::new_uninit_slice(len) };
                    let accessor = h_coset.get_mut_accessor();
                    unsafe {
                        d_coset.copy_to_slice(accessor.get_mut());
                    }
                    leaf_values.push(h_coset);
                }
                last_fri_step_plain_leaf_values = leaf_values;

                // Commit leaf values to transcript
                let mut transcript_input = vec![];
                for values in last_fri_step_plain_leaf_values.iter() {
                    let accessor = values.get_accessor();
                    let it = unsafe { accessor.get() }
                        .iter()
                        .flat_map(|x| x.into_coeffs_in_base().map(|y: BF| y.to_reduced_u32()));
                    transcript_input.extend(it);
                }
                Transcript::commit_with_seed(seed, &transcript_input);
                (vec![], vec![])
            } else {
                // Build ALL coset merkle trees in ONE command buffer
                let mut trees = Vec::with_capacity(lde_factor);
                for _ in 0..lde_factor {
                    trees.push(context.alloc::<Digest>(1 << (log_num_leafs + 1))?);
                }
                let mut tree_caps = allocate_tree_caps(log_lde_factor, log_tree_cap_size);

                assert!(log_tree_cap_size >= log_lde_factor);
                let log_coset_cap_size = log_tree_cap_size - log_lde_factor;

                let device = context.device();
                let cmd_buf = context.new_command_buffer()?;
                let num_leaves = 1usize << log_num_leafs;

                for (lde, tree) in ldes.iter().zip_eq(trees.iter_mut()) {
                    let log_tree_len = log_num_leafs + 1;
                    let layers_count = log_num_leafs + 1 - log_coset_cap_size;
                    assert_eq!(tree.len(), 1 << log_tree_len);

                    let bf_len = lde.len() * 4;
                    blake2s::build_merkle_tree_leaves_raw(
                        device, &cmd_buf, lde.raw(), bf_len,
                        tree, num_leaves, next_log_fold + 2,
                    )?;
                    if layers_count > 0 {
                        let digest_size = std::mem::size_of::<blake2s::Digest>();
                        blake2s::build_merkle_tree_nodes_with_offset(
                            device, &cmd_buf, tree, 0, num_leaves,
                            tree, num_leaves * digest_size, layers_count - 1,
                        )?;
                    }
                }
                cmd_buf.commit_and_wait();

                // CPU: transfer all tree caps after GPU completes
                for (tree, caps) in trees.iter().zip_eq(tree_caps.iter_mut()) {
                    transfer_tree_cap(tree, caps, log_lde_factor, log_tree_cap_size);
                }

                // Commit tree caps to transcript
                let tree_caps_accessors = tree_caps
                    .iter()
                    .map(|h| <HostAllocation<[Digest]>>::get_accessor(h))
                    .collect_vec();
                let input = flatten_tree_caps(&tree_caps_accessors).collect_vec();
                Transcript::commit_with_seed(seed, &input);

                (trees, tree_caps)
            };

            let oracle = FRIStep {
                ldes,
                trees,
                tree_caps,
            };
            fri_oracles.push(oracle);
            log_current_domain_size = log_folded_domain_size;
        }

        assert_eq!(
            log_current_domain_size as usize,
            folding_description.final_monomial_degree_log2
                + folding_description.folding_sequence.last().unwrap()
        );

        // Final monomials
        let final_monomials = {
            let log_folding_degree = *folding_description.folding_sequence.last().unwrap() as u32;
            let challenges_len = log_folding_degree as usize;
            let mut h_challenges = vec![E4::default(); challenges_len];
            Self::set_folding_challenges(
                seed,
                &mut taus[..1],
                &mut h_challenges,
                log_folding_degree as usize,
            );

            let d_challenges = context.alloc_from_slice(&h_challenges)?;
            let log_folded_domain_size = log_current_domain_size - log_folding_degree;
            let folded_domain_size = 1 << log_folded_domain_size;
            let mut d_folded_domain = context.alloc::<E4>(folded_domain_size)?;

            Self::fold_coset(
                log_folding_degree,
                &d_challenges,
                0,
                &fri_oracles.last().unwrap().ldes[0],
                &mut d_folded_domain,
                context,
            )?;

            // Copy folded domain to host
            let mut h_folded_domain = vec![E4::default(); folded_domain_size];
            unsafe { d_folded_domain.copy_to_slice(&mut h_folded_domain) };

            log_current_domain_size -= log_folding_degree;
            let domain_size = 1 << log_current_domain_size;

            // Compute monomials via interpolation
            let mut c0 = h_folded_domain.iter().map(|el| el.c0).collect_vec();
            let mut c1 = h_folded_domain.iter().map(|el| el.c1).collect_vec();
            assert_eq!(c0.len(), domain_size);
            assert_eq!(c1.len(), domain_size);
            bitreverse_enumeration_inplace(&mut c0);
            bitreverse_enumeration_inplace(&mut c1);
            Self::interpolate(&mut c0);
            Self::interpolate(&mut c1);
            let coeffs = c0
                .into_iter()
                .zip(c1.into_iter())
                .map(|(c0, c1)| E4 { c0, c1 })
                .collect_vec();

            let mut monomials = unsafe { HostAllocation::<[E4]>::new_uninit_slice(domain_size) };
            let monomials_accessor = monomials.get_mut_accessor();
            unsafe { monomials_accessor.get_mut().copy_from_slice(&coeffs) };

            // Commit monomials to transcript
            let mut transcript_input = vec![];
            let accessor = monomials.get_accessor();
            let it = unsafe { accessor.get() }
                .iter()
                .flat_map(|x| x.into_coeffs_in_base().map(|y: BF| y.to_reduced_u32()));
            transcript_input.extend(it);
            Transcript::commit_with_seed(seed, &transcript_input);

            monomials
        };

        assert_eq!(
            log_current_domain_size as usize,
            folding_description.final_monomial_degree_log2
        );

        Ok(Self {
            fri_oracles,
            last_fri_step_plain_leaf_values,
            final_monomials,
        })
    }

    fn draw_challenge(seed: &mut Seed) -> E4 {
        let mut transcript_challenges =
            [0u32; 4usize.next_multiple_of(BLAKE2S_DIGEST_SIZE_U32_WORDS)];
        Transcript::draw_randomness(seed, &mut transcript_challenges);
        let coeffs = transcript_challenges
            .as_chunks::<4>()
            .0
            .iter()
            .next()
            .unwrap()
            .map(BF::from_nonreduced_u32);
        E4::from_coeffs_in_base(&coeffs)
    }

    fn set_folding_challenges(
        seed: &mut Seed,
        taus: &mut [E2],
        challenges: &mut [E4],
        log_degree: usize,
    ) {
        assert_eq!(challenges.len(), taus.len() * log_degree);
        let mut challenge = Self::draw_challenge(seed);
        let challenge_powers = iter::once(challenge)
            .chain((1..log_degree).map(|_| {
                challenge.square();
                challenge
            }))
            .collect_vec();
        for (tau, chunk) in taus.iter_mut().zip(challenges.chunks_mut(log_degree)) {
            let mut tau_inv = tau.inverse().unwrap();
            for (challenge, mut power) in chunk.iter_mut().zip(challenge_powers.iter().copied()) {
                power.mul_assign_by_base(&tau_inv);
                *challenge = power;
                tau_inv.square();
                tau.square();
            }
        }
    }

    fn fold_coset_dispatch(
        cmd_buf: &crate::metal_runtime::command_queue::MetalCommandBuffer,
        log_degree: u32,
        d_challenges: &MetalBuffer<E4>,
        challenge_offset: usize,
        input: &MetalBuffer<E4>,
        output: &mut MetalBuffer<E4>,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let log_degree = log_degree as usize;
        let domain_size = input.len();
        assert!(domain_size.is_power_of_two());
        let log_domain_size = domain_size.trailing_zeros();

        let device = context.device();
        let dc = context.device_context();

        let fine_desc = PowersLayerDesc {
            mask: (1u32 << dc.fine_log_count) - 1,
            log_count: dc.fine_log_count,
        };
        let coarser_desc = PowersLayerDesc {
            mask: (1u32 << dc.coarser_log_count) - 1,
            log_count: dc.coarser_log_count,
        };
        let coarsest_desc = PowersLayerDesc {
            mask: (1u32 << dc.coarsest_log_count) - 1,
            log_count: dc.coarsest_log_count,
        };

        let mut temp_alloc: Option<MetalBuffer<E4>> = None;
        let mut output_ref = Some(output);

        for i in 0..log_degree {
            let log_current_domain_size = log_domain_size - i as u32;
            let log_next_domain_size = log_current_domain_size - 1;

            let src = if let Some(temp) = temp_alloc.as_ref() {
                temp
            } else {
                input
            };

            let is_last = i == log_degree - 1;

            let challenge_idx = challenge_offset + i;
            let challenge_val = unsafe { d_challenges.as_slice() }[challenge_idx];
            let d_single_challenge = context.alloc_from_slice(&[challenge_val])?;

            if is_last {
                let dst = output_ref.take().unwrap();
                ops_complex::fold(
                    device, cmd_buf, &d_single_challenge, src, dst, 0,
                    log_next_domain_size,
                    &dc.powers_of_w_fine, &fine_desc,
                    &dc.powers_of_w_coarser, &coarser_desc,
                    &dc.powers_of_w_coarsest, &coarsest_desc,
                )?;
            } else {
                let new_temp = context.alloc::<E4>(1 << log_next_domain_size)?;
                ops_complex::fold(
                    device, cmd_buf, &d_single_challenge, src, &new_temp, 0,
                    log_next_domain_size,
                    &dc.powers_of_w_fine, &fine_desc,
                    &dc.powers_of_w_coarser, &coarser_desc,
                    &dc.powers_of_w_coarsest, &coarsest_desc,
                )?;
                temp_alloc = Some(new_temp);
            }
        }

        Ok(())
    }

    fn fold_coset(
        log_degree: u32,
        d_challenges: &MetalBuffer<E4>,
        challenge_offset: usize,
        input: &MetalBuffer<E4>,
        output: &mut MetalBuffer<E4>,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let cmd_buf = context.new_command_buffer()?;
        Self::fold_coset_dispatch(
            &cmd_buf, log_degree, d_challenges, challenge_offset,
            input, output, context,
        )?;
        cmd_buf.commit_and_wait();
        Ok(())
    }

    fn interpolate(c0: &mut [E2]) {
        let twiddles = &PRECOMPUTATIONS.inverse_twiddles[..c0.len() / 2];
        partial_ifft_natural_to_natural(c0, E2::ONE, twiddles);
        if c0.len() > 1 {
            let n_inv = Mersenne31Field(c0.len() as u32).inverse().unwrap();
            let mut i = 0;
            let work_size = c0.len();
            while i < work_size {
                c0[i].mul_assign_by_base(&n_inv);
                i += 1;
            }
        }
    }
}
