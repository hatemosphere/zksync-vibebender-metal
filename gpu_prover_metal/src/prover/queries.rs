//! Query construction for FRI verification.
//! Ported from gpu_prover/src/prover/queries.rs.
//!
//! On Metal with unified memory, we simplify the CUDA version:
//! - No async device-to-host copies (data is already accessible)
//! - Direct host-side gather from Metal buffers

use super::context::ProverContext;
use super::setup::SetupPrecomputations;
use super::stage_1::StageOneOutput;
use super::stage_2::StageTwoOutput;
use super::stage_3::StageThreeOutput;
use super::stage_4::StageFourOutput;
use super::stage_5::StageFiveOutput;
use super::trace_holder::{CosetsHolder, TreesHolder};
use super::BF;
use crate::metal_runtime::MetalCommandBuffer;
use field::Mersenne31Field;
use objc2::runtime::ProtocolObject;
use crate::blake2s::Digest;
use crate::metal_runtime::{MetalBuffer, MetalResult};
use blake2s_u32::BLAKE2S_DIGEST_SIZE_U32_WORDS;
use itertools::Itertools;
use rayon::prelude::*;
use prover::definitions::{FoldingDescription, Transcript};
use prover::prover_stages::query_producer::{assemble_query_index, BitSource};
use prover::prover_stages::stage5::Query;
use prover::prover_stages::QuerySet;
use prover::transcript::Seed;

struct LeafsAndDigests {
    leafs: Vec<BF>,
    digests: Vec<Digest>,
    columns_count: usize,
}

pub(crate) struct QueriesOutput {
    leafs_and_digest_sets: Vec<LeafsAndDigestsSet>,
    query_indexes: Vec<u32>,
    log_domain_size: u32,
    folding_sequence: Vec<u32>,
}

struct LeafsAndDigestsSet {
    witness: LeafsAndDigests,
    memory: LeafsAndDigests,
    setup: LeafsAndDigests,
    stage_2: LeafsAndDigests,
    quotient: LeafsAndDigests,
    initial_fri: LeafsAndDigests,
    intermediate_fri: Vec<LeafsAndDigests>,
}

impl QueriesOutput {
    pub fn empty() -> Self {
        Self {
            leafs_and_digest_sets: vec![],
            query_indexes: vec![],
            log_domain_size: 0,
            folding_sequence: vec![],
        }
    }

    pub fn new(
        seed: &mut Seed,
        setup: &mut SetupPrecomputations,
        stage_1_output: &mut StageOneOutput,
        stage_2_output: &mut StageTwoOutput,
        stage_3_output: &mut StageThreeOutput,
        stage_4_output: &mut StageFourOutput,
        stage_5_output: &StageFiveOutput,
        log_domain_size: u32,
        log_lde_factor: u32,
        num_queries: usize,
        folding_description: &FoldingDescription,
        context: &ProverContext,
    ) -> MetalResult<Self> {
        let _g = crate::cpu_scoped!("queries_index_gen");
        let tree_index_bits = log_domain_size;
        let tree_index_mask = (1u32 << tree_index_bits) - 1;
        let coset_index_bits = log_lde_factor;
        let lde_factor = 1usize << log_lde_factor;
        let log_tree_cap_size = folding_description.total_caps_size_log2 as u32;
        let log_coset_tree_cap_size = log_tree_cap_size - log_lde_factor;
        let query_index_bits = tree_index_bits + coset_index_bits;
        let num_required_bits = query_index_bits * num_queries as u32;
        let num_required_words = num_required_bits.next_multiple_of(u32::BITS) / u32::BITS;
        let num_required_words_padded =
            (num_required_words as usize + 1).next_multiple_of(BLAKE2S_DIGEST_SIZE_U32_WORDS);

        // Generate query indexes
        let mut query_indexes = vec![0u32; num_queries];
        let mut tree_indexes = vec![0u32; num_queries];
        {
            let mut source = vec![0u32; num_required_words_padded];
            Transcript::draw_randomness(seed, &mut source);
            let mut bit_source = BitSource::new(source[1..].to_vec());
            for i in 0..num_queries {
                let query_index =
                    assemble_query_index(query_index_bits as usize, &mut bit_source) as u32;
                let tree_index = query_index & tree_index_mask;
                query_indexes[i] = query_index;
                tree_indexes[i] = tree_index;
            }
        }
        drop(_g);

        // Cache column counts before mutable borrows in the loop
        let witness_cols = stage_1_output.witness_holder.columns_count;
        let memory_cols = stage_1_output.memory_holder.columns_count;
        let setup_cols_count = setup.trace_holder.columns_count;
        let stage_2_cols_count = stage_2_output.trace_holder.columns_count;
        let stage_3_cols_count = stage_3_output.trace_holder.columns_count;

        let mut leafs_and_digest_sets = Vec::with_capacity(lde_factor);
        for coset_idx in 0..lde_factor {
            let _g = crate::cpu_scoped!("queries_coset_gather");
            let mut current_tree_indexes = tree_indexes.clone();
            let mut log_dom = log_domain_size;
            let mut layers_count = log_domain_size - log_coset_tree_cap_size;

            // Batch all 5 trace gathers into ONE command buffer (saves 4 sync points per coset)
            let (witness_evals, witness_tree) = stage_1_output
                .witness_holder
                .get_coset_evaluations_and_tree(coset_idx, context)?;
            let (memory_evals, memory_tree) = stage_1_output
                .memory_holder
                .get_coset_evaluations_and_tree(coset_idx, context)?;
            let setup_evals = setup
                .trace_holder
                .get_coset_evaluations(coset_idx, context)?;
            let setup_tree = &setup.trees_and_caps.trees[coset_idx];
            let (stage_2_evals, stage_2_tree) = stage_2_output
                .trace_holder
                .get_coset_evaluations_and_tree(coset_idx, context)?;
            let (stage_3_evals, stage_3_tree) = stage_3_output
                .trace_holder
                .get_coset_evaluations_and_tree(coset_idx, context)?;

            let d_indexes = context.alloc_from_slice(&current_tree_indexes)?;
            let domain_size = 1u32 << log_dom;
            let queries_count = current_tree_indexes.len() as u32;
            let cmd_buf = context.new_command_buffer()?;

            let d_witness = Self::dispatch_gather(
                &cmd_buf, &d_indexes, witness_evals.raw(),
                domain_size, witness_cols as u32, queries_count,
                0, true, log_dom, context)?;
            let d_memory = Self::dispatch_gather(
                &cmd_buf, &d_indexes, memory_evals.raw(),
                domain_size, memory_cols as u32, queries_count,
                0, true, log_dom, context)?;
            let d_setup = Self::dispatch_gather(
                &cmd_buf, &d_indexes, setup_evals.raw(),
                domain_size, setup_cols_count as u32, queries_count,
                0, true, log_dom, context)?;
            let d_stage2 = Self::dispatch_gather(
                &cmd_buf, &d_indexes, stage_2_evals.raw(),
                domain_size, stage_2_cols_count as u32, queries_count,
                0, true, log_dom, context)?;
            let d_stage3 = Self::dispatch_gather(
                &cmd_buf, &d_indexes, stage_3_evals.raw(),
                domain_size, stage_3_cols_count as u32, queries_count,
                0, true, log_dom, context)?;
            cmd_buf.commit_and_wait();

            let witness_tree_slice = unsafe { witness_tree.as_buffer().as_slice() };
            let witness = Self::finish_gather(&d_witness, &current_tree_indexes, witness_tree_slice, layers_count, witness_cols);
            let memory_tree_slice = unsafe { memory_tree.as_buffer().as_slice() };
            let memory = Self::finish_gather(&d_memory, &current_tree_indexes, memory_tree_slice, layers_count, memory_cols);
            let setup_ld = Self::finish_gather(&d_setup, &current_tree_indexes, setup_tree, layers_count, setup_cols_count);
            let stage_2_tree_slice = unsafe { stage_2_tree.as_buffer().as_slice() };
            let stage_2 = Self::finish_gather(&d_stage2, &current_tree_indexes, stage_2_tree_slice, layers_count, stage_2_cols_count);
            let stage_3_tree_slice = unsafe { stage_3_tree.as_buffer().as_slice() };
            let quotient = Self::finish_gather(&d_stage3, &current_tree_indexes, stage_3_tree_slice, layers_count, stage_3_cols_count);

            // FRI folding
            let folding_sequence = folding_description.folding_sequence;
            let initial_log_fold = folding_sequence[0] as u32;
            current_tree_indexes
                .iter_mut()
                .for_each(|x| *x >>= initial_log_fold);
            layers_count -= initial_log_fold;

            let stage_4_evals = match &stage_4_output.trace_holder.cosets {
                CosetsHolder::Full(evaluations) => &evaluations[coset_idx],
                CosetsHolder::Single { .. } => unreachable!(),
            };
            let stage_4_tree = match &stage_4_output.trace_holder.trees {
                TreesHolder::Full(trees) => &trees[coset_idx],
                TreesHolder::Partial(trees) => &trees[coset_idx],
                TreesHolder::None => unreachable!(),
            };

            // E4 buffer transmuted to BF for leaf gathering
            let initial_fri = Self::gather_leafs_and_digests_e4(
                &current_tree_indexes,
                false,
                stage_4_evals,
                stage_4_tree,
                log_dom + 2,
                initial_log_fold + 2,
                layers_count,
                context,
            )?;
            log_dom -= initial_log_fold;

            let mut intermediate_fri = vec![];
            for (i, intermediate_oracle) in stage_5_output.fri_oracles.iter().enumerate() {
                if intermediate_oracle.trees.is_empty() {
                    continue;
                }
                let log_fold = folding_sequence[i + 1] as u32;
                layers_count -= log_fold;
                current_tree_indexes
                    .iter_mut()
                    .for_each(|x| *x >>= log_fold);
                let queries = Self::gather_leafs_and_digests_e4(
                    &current_tree_indexes,
                    false,
                    &intermediate_oracle.ldes[coset_idx],
                    &intermediate_oracle.trees[coset_idx],
                    log_dom + 2,
                    log_fold + 2,
                    layers_count,
                    context,
                )?;
                log_dom -= log_fold;
                intermediate_fri.push(queries);
            }

            let set = LeafsAndDigestsSet {
                witness,
                memory,
                setup: setup_ld,
                stage_2,
                quotient,
                initial_fri,
                intermediate_fri,
            };
            leafs_and_digest_sets.push(set);
        }

        let folding_sequence = folding_description
            .folding_sequence
            .iter()
            .map(|&x| x as u32)
            .collect_vec();

        Ok(Self {
            leafs_and_digest_sets,
            query_indexes,
            log_domain_size,
            folding_sequence,
        })
    }

    /// Dispatch GPU gather kernel into existing command buffer (no sync).
    /// Returns the GPU leafs buffer for later readback.
    fn dispatch_gather(
        cmd_buf: &MetalCommandBuffer,
        d_query_indexes: &MetalBuffer<u32>,
        values: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
        domain_size: u32,
        columns_count: u32,
        queries_count: u32,
        log_rows_per_index: u32,
        bit_reverse: bool,
        log_domain_size: u32,
        context: &ProverContext,
    ) -> MetalResult<MetalBuffer<BF>> {
        let rows_per_index = 1u32 << log_rows_per_index;
        let total = columns_count * queries_count * rows_per_index;
        let d_leafs: MetalBuffer<BF> = context.alloc(total as usize)?;
        crate::blake2s::gather_query_leafs(
            context.device(), cmd_buf,
            values, d_query_indexes, &d_leafs,
            domain_size, columns_count, queries_count,
            log_rows_per_index, bit_reverse, log_domain_size,
        )?;
        Ok(d_leafs)
    }

    /// Read back gathered leafs from GPU buffer + gather merkle paths on CPU.
    fn finish_gather(
        d_leafs: &MetalBuffer<BF>,
        tree_indexes: &[u32],
        tree_slice: &[Digest],
        layers_count: u32,
        columns_count: usize,
    ) -> LeafsAndDigests {
        let total = d_leafs.len();
        let mut leafs = vec![Mersenne31Field(0); total];
        unsafe { d_leafs.copy_to_slice(&mut leafs) };
        let digests = gather_merkle_paths_from_slice(tree_indexes, tree_slice, layers_count);
        LeafsAndDigests { leafs, digests, columns_count }
    }

    /// Legacy single-call gather (used for host_tree variant).
    fn gather_leafs_and_digests(
        tree_indexes: &[u32],
        bit_reverse_leaf_indexing: bool,
        values: &MetalBuffer<BF>,
        tree: &MetalBuffer<Digest>,
        log_domain_size: u32,
        log_rows_per_index: u32,
        layers_count: u32,
        actual_columns_count: usize,
        context: &ProverContext,
    ) -> MetalResult<LeafsAndDigests> {
        let queries_count = tree_indexes.len();
        let domain_size = 1usize << log_domain_size;
        let d_query_indexes = context.alloc_from_slice(tree_indexes)?;
        let cmd_buf = context.new_command_buffer()?;
        let d_leafs = Self::dispatch_gather(
            &cmd_buf, &d_query_indexes, values.raw(),
            domain_size as u32, actual_columns_count as u32, queries_count as u32,
            log_rows_per_index, bit_reverse_leaf_indexing, log_domain_size, context,
        )?;
        cmd_buf.commit_and_wait();
        let tree_slice = unsafe { tree.as_slice() };
        Ok(Self::finish_gather(&d_leafs, tree_indexes, tree_slice, layers_count, actual_columns_count))
    }

    /// Gather from E4 buffers (transmuted to BF for leaf content).
    /// Uses GPU gather kernel — E4 buffer treated as raw BF via transmute.
    fn gather_leafs_and_digests_e4(
        tree_indexes: &[u32],
        bit_reverse_leaf_indexing: bool,
        values: &MetalBuffer<super::E4>,
        tree: &MetalBuffer<Digest>,
        log_domain_size: u32,
        log_rows_per_index: u32,
        layers_count: u32,
        context: &ProverContext,
    ) -> MetalResult<LeafsAndDigests> {
        let bf_len = values.len() * 4;
        let queries_count = tree_indexes.len();
        let domain_size = 1usize << log_domain_size;
        let columns_count = bf_len / domain_size;
        let rows_per_index = 1usize << log_rows_per_index;
        let total_leafs = queries_count * rows_per_index * columns_count;

        // GPU gather: E4 buffer raw-pointer is valid as BF since E4=[BF;4]
        let d_query_indexes = context.alloc_from_slice(tree_indexes)?;
        let d_leafs: MetalBuffer<BF> = context.alloc(total_leafs)?;
        let device = context.device();
        let cmd_buf = context.new_command_buffer()?;
        crate::blake2s::gather_query_leafs(
            device, &cmd_buf,
            values.raw(), &d_query_indexes, &d_leafs,
            domain_size as u32, columns_count as u32, queries_count as u32,
            log_rows_per_index, bit_reverse_leaf_indexing, log_domain_size,
        )?;
        cmd_buf.commit_and_wait();
        let mut leafs = vec![Mersenne31Field(0); total_leafs];
        unsafe { d_leafs.copy_to_slice(&mut leafs) };

        let tree_slice = unsafe { tree.as_slice() };
        let digests = gather_merkle_paths_from_slice(tree_indexes, tree_slice, layers_count);

        Ok(LeafsAndDigests { leafs, digests, columns_count })
    }

    /// Gather from a host-side tree (Box<[Digest]>).
    fn gather_leafs_and_digests_host_tree(
        tree_indexes: &[u32],
        bit_reverse_leaf_indexing: bool,
        values: &MetalBuffer<BF>,
        tree: &Box<[Digest]>,
        log_domain_size: u32,
        log_rows_per_index: u32,
        layers_count: u32,
        actual_columns_count: usize,
        context: &ProverContext,
    ) -> MetalResult<LeafsAndDigests> {
        let queries_count = tree_indexes.len();
        let domain_size = 1usize << log_domain_size;
        let columns_count = actual_columns_count;
        let rows_per_index = 1usize << log_rows_per_index;
        let total_leafs = queries_count * rows_per_index * columns_count;

        // GPU gather for leaf values (values buffer is on GPU)
        let d_query_indexes = context.alloc_from_slice(tree_indexes)?;
        let d_leafs: MetalBuffer<BF> = context.alloc(total_leafs)?;
        let device = context.device();
        let cmd_buf = context.new_command_buffer()?;
        crate::blake2s::gather_query_leafs(
            device, &cmd_buf,
            values.raw(), &d_query_indexes, &d_leafs,
            domain_size as u32, columns_count as u32, queries_count as u32,
            log_rows_per_index, bit_reverse_leaf_indexing, log_domain_size,
        )?;
        cmd_buf.commit_and_wait();
        let mut leafs = vec![Mersenne31Field(0); total_leafs];
        unsafe { d_leafs.copy_to_slice(&mut leafs) };

        // Merkle paths from host tree (already on CPU)
        let digests = gather_merkle_paths_from_slice(tree_indexes, tree, layers_count);

        Ok(LeafsAndDigests { leafs, digests, columns_count })
    }

    pub fn into_query_sets(self) -> Vec<QuerySet> {
        let _g = crate::cpu_scoped!("queries_assembly");
        self.produce_query_sets()
    }

    pub fn produce_query_sets(&self) -> Vec<QuerySet> {
        let query_indexes = &self.query_indexes;
        let tree_index_bits = self.log_domain_size;
        let tree_index_mask = (1u32 << tree_index_bits) - 1;
        let tree_indexes = query_indexes
            .iter()
            .map(|&x| x & tree_index_mask)
            .collect_vec();

        let mut witness_queries_by_coset = vec![];
        let mut memory_queries_by_coset = vec![];
        let mut setup_queries_by_coset = vec![];
        let mut stage_2_queries_by_coset = vec![];
        let mut quotient_queries_by_coset = vec![];
        let mut initial_fri_queries_by_coset = vec![];
        let mut intermediate_fri_queries_by_coset = vec![];

        for set in self.leafs_and_digest_sets.iter() {
            let mut ti = tree_indexes.clone();
            let witness = produce_queries(query_indexes, &ti, &set.witness, 0);
            witness_queries_by_coset.push(witness);
            let memory = produce_queries(query_indexes, &ti, &set.memory, 0);
            memory_queries_by_coset.push(memory);
            let setup = produce_queries(query_indexes, &ti, &set.setup, 0);
            setup_queries_by_coset.push(setup);
            let stage_2 = produce_queries(query_indexes, &ti, &set.stage_2, 0);
            stage_2_queries_by_coset.push(stage_2);
            let quotient = produce_queries(query_indexes, &ti, &set.quotient, 0);
            quotient_queries_by_coset.push(quotient);

            let initial_log_fold = self.folding_sequence[0];
            ti.iter_mut().for_each(|x| *x >>= initial_log_fold);
            let initial_fri = produce_queries(
                query_indexes,
                &ti,
                &set.initial_fri,
                initial_log_fold + 2,
            );
            initial_fri_queries_by_coset.push(initial_fri);

            let intermediate_fri = set
                .intermediate_fri
                .iter()
                .zip(self.folding_sequence.iter().skip(1))
                .map(|(ld, &fold)| {
                    ti.iter_mut().for_each(|x| *x >>= fold);
                    produce_queries(query_indexes, &ti, ld, fold as u32 + 2)
                })
                .collect_vec();
            intermediate_fri_queries_by_coset.push(intermediate_fri);
        }

        query_indexes
            .par_iter()
            .enumerate()
            .map(|(i, &query_index)| {
                let coset_index = query_index as usize >> tree_index_bits;
                QuerySet {
                    witness_query: witness_queries_by_coset[coset_index][i].clone(),
                    memory_query: memory_queries_by_coset[coset_index][i].clone(),
                    setup_query: setup_queries_by_coset[coset_index][i].clone(),
                    stage_2_query: stage_2_queries_by_coset[coset_index][i].clone(),
                    quotient_query: quotient_queries_by_coset[coset_index][i].clone(),
                    initial_fri_query: initial_fri_queries_by_coset[coset_index][i].clone(),
                    intermediate_fri_queries: intermediate_fri_queries_by_coset[coset_index]
                        .iter()
                        .map(|queries| queries[i].clone())
                        .collect_vec(),
                }
            })
            .collect()
    }
}

fn produce_queries(
    query_indexes: &[u32],
    tree_indexes: &[u32],
    ld: &LeafsAndDigests,
    log_rows_per_index: u32,
) -> Vec<Query> {
    let queries_count = query_indexes.len();
    let leafs = &ld.leafs;
    let digests = &ld.digests;
    let values_per_column_count = queries_count << log_rows_per_index;

    let columns_count = ld.columns_count;
    let layers_count = if queries_count > 0 {
        digests.len() / queries_count
    } else {
        0
    };

    query_indexes
        .par_iter()
        .enumerate()
        .map(|(i, &query_index)| {
            let tree_index = tree_indexes[i];
            let mut leaf_content = Vec::with_capacity(columns_count << log_rows_per_index);
            let leaf_offset = i << log_rows_per_index;
            for col in 0..columns_count {
                for row in 0..1u32 << log_rows_per_index {
                    leaf_content.push(leafs[leaf_offset + values_per_column_count * col + row as usize]);
                }
            }
            let mut merkle_proof = Vec::with_capacity(layers_count);
            for layer in 0..layers_count {
                merkle_proof.push(digests[i + layer * queries_count]);
            }
            Query {
                query_index,
                tree_index,
                leaf_content,
                merkle_proof,
            }
        })
        .collect()
}

fn gather_merkle_paths_from_slice(
    indexes: &[u32],
    tree: &[Digest],
    layers_count: u32,
) -> Vec<Digest> {
    let queries_count = indexes.len();
    let total_len = tree.len();
    assert!(total_len.is_power_of_two());
    let leaves_count = total_len / 2;

    let mut digests = Vec::with_capacity(queries_count * layers_count as usize);
    for layer in 0..layers_count {
        let _layer_size = leaves_count >> layer;
        let _layer_offset = if layer == 0 { 0 } else { leaves_count * 2 - (leaves_count >> (layer - 1)) * 2 };
        // More standard approach: tree[0..leaves_count] = leaves,
        // tree[leaves_count..2*leaves_count] has internal nodes in level order
        // Actually merkle tree layout: leaves at [0..N), parents at [N..2N)
        // with node i's children at 2*(i-N) and 2*(i-N)+1 within the leaf range
        // Standard layout used here: tree[i]'s sibling is tree[i^1]
        // tree[i]'s parent is tree[leaves_count + i/2]
        for &idx in indexes.iter() {
            let node_idx = idx as usize >> layer;
            // Walk up `layer` levels from the leaf
            let mut current_level_offset = 0usize;
            let mut current_level_size = leaves_count;
            for _l in 0..layer {
                current_level_offset += current_level_size;
                current_level_size >>= 1;
            }
            // The sibling at this layer
            let sibling_idx = current_level_offset + (node_idx ^ 1);
            if sibling_idx < total_len {
                digests.push(tree[sibling_idx]);
            } else {
                digests.push([0u32; 8]);
            }
        }
    }
    digests
}

fn bit_reverse(val: usize, num_bits: usize) -> usize {
    let mut result = 0;
    let mut v = val;
    for _ in 0..num_bits {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}
