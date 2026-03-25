//! Trace buffer management for Metal.
//! Ported from gpu_prover/src/prover/trace_holder.rs.
//!
//! The CUDA version uses DeviceAllocation<T> from a sub-allocator.
//! On Metal, we use MetalBuffer<T> directly (unified memory).

use super::context::{HostAllocation, ProverContext, UnsafeAccessor};
use super::{BF, E4};
use crate::blake2s::{self, Digest};
use crate::metal_runtime::{MetalBuffer, MetalResult};
use itertools::Itertools;
use prover::merkle_trees::MerkleTreeCapVarLength;
use prover::prover_stages::Transcript;
use prover::transcript::Seed;

#[derive(Copy, Clone)]
pub enum TreesCacheMode {
    CacheNone,
    CachePartial,
    CacheFull,
}

pub(crate) enum CosetsHolder<T> {
    Full(Vec<MetalBuffer<T>>),
    Single {
        current_coset_index: usize,
        evaluations: MetalBuffer<T>,
    },
}

pub(crate) enum TreesHolder {
    Full(Vec<MetalBuffer<Digest>>),
    Partial(Vec<MetalBuffer<Digest>>),
    None,
}

pub(crate) enum TreeReference<'a> {
    Borrowed(&'a MetalBuffer<Digest>),
    Owned(MetalBuffer<Digest>),
}

impl<'a> TreeReference<'a> {
    pub fn as_buffer(&self) -> &MetalBuffer<Digest> {
        match self {
            TreeReference::Borrowed(b) => b,
            TreeReference::Owned(o) => o,
        }
    }
}

pub(crate) trait TraceHolderImpl {
    fn ensure_coset_computed(
        &mut self,
        coset_index: usize,
        context: &ProverContext,
    ) -> MetalResult<()>;
}

pub(crate) struct TraceHolder<T> {
    pub(crate) log_domain_size: u32,
    pub(crate) log_lde_factor: u32,
    pub(crate) log_rows_per_leaf: u32,
    pub(crate) log_tree_cap_size: u32,
    pub(crate) columns_count: usize,
    pub(crate) padded_to_even: bool,
    pub(crate) compressed_coset: bool,
    pub(crate) cosets: CosetsHolder<T>,
    pub(crate) trees: TreesHolder,
    pub(crate) tree_caps: Option<Vec<HostAllocation<[Digest]>>>,
}

impl TraceHolder<BF> {
    pub(crate) fn make_evaluations_sum_to_zero(
        &mut self,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let evaluations = match &mut self.cosets {
            CosetsHolder::Full(evaluations) => &mut evaluations[0],
            CosetsHolder::Single {
                current_coset_index,
                evaluations,
            } => {
                assert_eq!(*current_coset_index, 0);
                evaluations
            }
        };
        make_evaluations_sum_to_zero(
            evaluations,
            self.log_domain_size,
            self.columns_count,
            self.padded_to_even,
            context,
        )
    }

    pub(crate) fn extend(
        &mut self,
        source_coset_index: usize,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let log_domain_size = self.log_domain_size;
        let log_lde_factor = self.log_lde_factor;
        let compressed_coset = self.compressed_coset;
        assert_eq!(log_lde_factor, 1);
        match &mut self.cosets {
            CosetsHolder::Full(evaluations) => {
                let (src, dst) = split_evaluations_pair(evaluations, source_coset_index);
                compute_coset_evaluations(
                    src,
                    dst,
                    source_coset_index,
                    log_domain_size,
                    log_lde_factor,
                    compressed_coset,
                    context,
                )?;
            }
            CosetsHolder::Single {
                current_coset_index,
                evaluations,
            } => {
                assert_eq!(source_coset_index, *current_coset_index);
                *current_coset_index = 1 - source_coset_index;
                switch_coset_evaluations_in_place(
                    evaluations,
                    source_coset_index,
                    log_domain_size,
                    log_lde_factor,
                    compressed_coset,
                    context,
                )?;
            }
        }
        Ok(())
    }

    pub(crate) fn make_evaluations_sum_to_zero_and_extend(
        &mut self,
        context: &ProverContext,
    ) -> MetalResult<()> {
        self.make_evaluations_sum_to_zero(context)?;
        self.extend(0, context)
    }

    fn commit_and_transfer_tree_caps(
        &mut self,
        coset_index: usize,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let log_domain_size = self.log_domain_size;
        let log_lde_factor = self.log_lde_factor;
        let log_rows_per_leaf = self.log_rows_per_leaf;
        let log_tree_cap_size = self.log_tree_cap_size;
        let columns_count = self.columns_count;
        let mut tree = match &mut self.trees {
            TreesHolder::Full(trees) => trees.remove(coset_index),
            TreesHolder::Partial(trees) => trees.remove(coset_index),
            TreesHolder::None => allocate_tree(log_domain_size, log_rows_per_leaf, context)?,
        };
        let evaluations = self.get_coset_evaluations_raw(coset_index);
        commit_trace(
            evaluations,
            &mut tree,
            log_domain_size,
            log_lde_factor,
            log_rows_per_leaf,
            log_tree_cap_size,
            columns_count,
            context,
        )?;
        let caps = &mut self.tree_caps.as_mut().unwrap()[coset_index];
        transfer_tree_cap(&tree, caps, log_lde_factor, log_tree_cap_size);
        match &mut self.trees {
            TreesHolder::Full(trees) => trees.insert(coset_index, tree),
            TreesHolder::Partial(trees) => trees.insert(coset_index, tree),
            TreesHolder::None => drop(tree),
        };
        Ok(())
    }

    pub(crate) fn extend_and_commit(
        &mut self,
        source_coset_index: usize,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let tree_caps = allocate_tree_caps(self.log_lde_factor, self.log_tree_cap_size);
        assert!(self.tree_caps.replace(tree_caps).is_none());
        self.commit_and_transfer_tree_caps(source_coset_index, context)?;
        self.extend(source_coset_index, context)?;
        self.commit_and_transfer_tree_caps(1 - source_coset_index, context)?;
        Ok(())
    }

    pub(crate) fn make_evaluations_sum_to_zero_extend_and_commit(
        &mut self,
        context: &ProverContext,
    ) -> MetalResult<()> {
        self.make_evaluations_sum_to_zero(context)?;
        self.extend_and_commit(0, context)?;
        Ok(())
    }

    pub(crate) fn get_coset_evaluations_and_tree(
        &mut self,
        coset_index: usize,
        context: &ProverContext,
    ) -> MetalResult<(&MetalBuffer<BF>, TreeReference<'_>)> {
        self.ensure_coset_computed(coset_index, context)?;
        let evaluations = match &self.cosets {
            CosetsHolder::Full(evaluations) => &evaluations[coset_index],
            CosetsHolder::Single {
                evaluations,
                current_coset_index,
            } => {
                assert_eq!(*current_coset_index, coset_index);
                evaluations
            }
        };
        let tree = match &self.trees {
            TreesHolder::Full(trees) => TreeReference::Borrowed(&trees[coset_index]),
            TreesHolder::Partial(trees) => TreeReference::Borrowed(&trees[coset_index]),
            TreesHolder::None => {
                let mut tree =
                    allocate_tree(self.log_domain_size, self.log_rows_per_leaf, context)?;
                commit_trace(
                    evaluations,
                    &mut tree,
                    self.log_domain_size,
                    self.log_lde_factor,
                    self.log_rows_per_leaf,
                    self.log_tree_cap_size,
                    self.columns_count,
                    context,
                )?;
                TreeReference::Owned(tree)
            }
        };
        Ok((evaluations, tree))
    }
}

impl TraceHolderImpl for TraceHolder<BF> {
    fn ensure_coset_computed(
        &mut self,
        coset_index: usize,
        context: &ProverContext,
    ) -> MetalResult<()> {
        assert!(coset_index < (1 << self.log_lde_factor));
        match &mut self.cosets {
            CosetsHolder::Full(evaluations) => {
                assert!(evaluations.len() > coset_index);
                Ok(())
            }
            CosetsHolder::Single {
                current_coset_index,
                evaluations,
            } => {
                if *current_coset_index == coset_index {
                    return Ok(());
                }
                switch_coset_evaluations_in_place(
                    evaluations,
                    *current_coset_index,
                    self.log_domain_size,
                    self.log_lde_factor,
                    self.compressed_coset,
                    context,
                )?;
                *current_coset_index = coset_index;
                Ok(())
            }
        }
    }
}

impl<T> TraceHolder<T> {
    pub fn empty() -> Self {
        Self {
            log_domain_size: 0,
            log_lde_factor: 0,
            log_rows_per_leaf: 0,
            log_tree_cap_size: 0,
            columns_count: 0,
            padded_to_even: false,
            compressed_coset: false,
            cosets: CosetsHolder::Full(vec![]),
            trees: TreesHolder::None,
            tree_caps: None,
        }
    }

    pub(crate) fn new(
        log_domain_size: u32,
        log_lde_factor: u32,
        log_rows_per_leaf: u32,
        log_tree_cap_size: u32,
        columns_count: usize,
        pad_to_even: bool,
        compressed_coset: bool,
        recompute_cosets: bool,
        trees_cache_mode: TreesCacheMode,
        context: &ProverContext,
    ) -> MetalResult<Self> {
        assert_eq!(log_lde_factor, 1);
        let padded_to_even = pad_to_even && columns_count.next_multiple_of(2) != columns_count;
        let instances_count = 1 << log_lde_factor;
        let cosets = match recompute_cosets {
            true => CosetsHolder::Single {
                current_coset_index: 0,
                evaluations: allocate_coset(log_domain_size, columns_count, pad_to_even, context)?,
            },
            false => CosetsHolder::Full(allocate_cosets(
                instances_count,
                log_domain_size,
                columns_count,
                pad_to_even,
                context,
            )?),
        };
        let trees = match trees_cache_mode {
            TreesCacheMode::CacheNone => TreesHolder::None,
            TreesCacheMode::CachePartial => TreesHolder::Partial(allocate_trees(
                instances_count,
                log_domain_size,
                log_rows_per_leaf,
                context,
            )?),
            TreesCacheMode::CacheFull => TreesHolder::Full(allocate_trees(
                instances_count,
                log_domain_size,
                log_rows_per_leaf,
                context,
            )?),
        };
        Ok(Self {
            log_domain_size,
            log_lde_factor,
            log_rows_per_leaf,
            log_tree_cap_size,
            columns_count,
            padded_to_even,
            compressed_coset,
            cosets,
            trees,
            tree_caps: None,
        })
    }

    pub(crate) fn allocate_only_evaluation(
        log_domain_size: u32,
        log_lde_factor: u32,
        log_rows_per_leaf: u32,
        log_tree_cap_size: u32,
        columns_count: usize,
        pad_to_even: bool,
        compressed_coset: bool,
        recompute_cosets: bool,
        trees_cache_mode: TreesCacheMode,
        context: &ProverContext,
    ) -> MetalResult<Self> {
        let padded_to_even = pad_to_even && columns_count.next_multiple_of(2) != columns_count;
        let evaluations = allocate_coset(log_domain_size, columns_count, pad_to_even, context)?;
        let cosets = match recompute_cosets {
            true => CosetsHolder::Single {
                current_coset_index: 0,
                evaluations,
            },
            false => CosetsHolder::Full(vec![evaluations]),
        };
        let trees = match trees_cache_mode {
            TreesCacheMode::CacheNone => TreesHolder::Full(vec![]),
            TreesCacheMode::CachePartial => TreesHolder::Partial(vec![]),
            TreesCacheMode::CacheFull => TreesHolder::None,
        };
        Ok(Self {
            log_domain_size,
            log_lde_factor,
            log_rows_per_leaf,
            log_tree_cap_size,
            columns_count,
            padded_to_even,
            compressed_coset,
            cosets,
            trees,
            tree_caps: None,
        })
    }

    pub(crate) fn allocate_to_full(&mut self, context: &ProverContext) -> MetalResult<()> {
        let instances_count = 1 << self.log_lde_factor;
        match &mut self.cosets {
            CosetsHolder::Full(evaluations) => {
                assert_eq!(evaluations.len(), 1);
                let additional_evaluations = allocate_cosets(
                    instances_count - 1,
                    self.log_domain_size,
                    self.columns_count,
                    self.padded_to_even,
                    context,
                )?;
                evaluations.extend(additional_evaluations);
            }
            CosetsHolder::Single { .. } => {}
        }
        match &mut self.trees {
            TreesHolder::Full(trees) => {
                assert!(trees.is_empty());
                let new_trees = allocate_trees(
                    instances_count,
                    self.log_domain_size,
                    self.log_rows_per_leaf,
                    context,
                )?;
                trees.extend(new_trees);
            }
            TreesHolder::Partial(trees) => {
                assert!(trees.is_empty());
                let new_trees = allocate_trees(
                    instances_count,
                    self.log_domain_size,
                    self.log_rows_per_leaf,
                    context,
                )?;
                trees.extend(new_trees);
            }
            TreesHolder::None => {}
        }
        Ok(())
    }

    pub(crate) fn get_tree_caps_accessors(&self) -> Vec<UnsafeAccessor<[Digest]>> {
        self.tree_caps
            .as_ref()
            .unwrap()
            .iter()
            .map(|h| <HostAllocation<[Digest]>>::get_accessor(h))
            .collect_vec()
    }

    pub(crate) fn get_update_seed_fn(&self, seed: &mut Seed) -> impl FnOnce() + Send {
        let tree_caps_accessors = self.get_tree_caps_accessors();
        let seed_accessor = super::context::UnsafeMutAccessor::new(seed);
        move || unsafe {
            let input = flatten_tree_caps(&tree_caps_accessors).collect_vec();
            Transcript::commit_with_seed(seed_accessor.get_mut(), &input);
        }
    }

    /// Get the raw evaluations buffer for a coset without checking computation state.
    fn get_coset_evaluations_raw(&self, coset_index: usize) -> &MetalBuffer<T> {
        match &self.cosets {
            CosetsHolder::Full(evaluations) => &evaluations[coset_index],
            CosetsHolder::Single {
                evaluations,
                current_coset_index,
            } => {
                assert_eq!(*current_coset_index, coset_index);
                evaluations
            }
        }
    }
}

impl TraceHolder<E4> {
    pub(crate) fn extend(
        &mut self,
        source_coset_index: usize,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let log_domain_size = self.log_domain_size;
        let log_lde_factor = self.log_lde_factor;
        let compressed_coset = self.compressed_coset;
        assert_eq!(log_lde_factor, 1);
        match &mut self.cosets {
            CosetsHolder::Full(evaluations) => {
                assert_eq!(evaluations.len(), 2);
                let (src_idx, dst_idx) = if source_coset_index == 0 {
                    (0, 1)
                } else {
                    (1, 0)
                };
                // Transmute E4 buffers to BF (4x elements) for NTT
                let src_bf: MetalBuffer<BF> =
                    unsafe { evaluations[src_idx].transmute_view() };
                let mut dst_bf: MetalBuffer<BF> =
                    unsafe { evaluations[dst_idx].transmute_view() };
                compute_coset_evaluations(
                    &src_bf,
                    &mut dst_bf,
                    source_coset_index,
                    log_domain_size,
                    log_lde_factor,
                    compressed_coset,
                    context,
                )?;
            }
            CosetsHolder::Single { .. } => {
                panic!("E4 extend requires Full cosets mode");
            }
        }
        Ok(())
    }

    fn commit_and_transfer_tree_caps(
        &mut self,
        coset_index: usize,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let log_domain_size = self.log_domain_size;
        let log_lde_factor = self.log_lde_factor;
        let log_rows_per_leaf = self.log_rows_per_leaf;
        let log_tree_cap_size = self.log_tree_cap_size;
        let mut tree = match &mut self.trees {
            TreesHolder::Full(trees) => trees.remove(coset_index),
            TreesHolder::Partial(trees) => trees.remove(coset_index),
            TreesHolder::None => allocate_tree(log_domain_size, log_rows_per_leaf, context)?,
        };
        let evaluations = self.get_coset_evaluations_raw(coset_index);
        commit_trace_e4(
            evaluations,
            &mut tree,
            log_domain_size,
            log_lde_factor,
            log_rows_per_leaf,
            log_tree_cap_size,
            context,
        )?;
        let caps = &mut self.tree_caps.as_mut().unwrap()[coset_index];
        transfer_tree_cap(&tree, caps, log_lde_factor, log_tree_cap_size);
        match &mut self.trees {
            TreesHolder::Full(trees) => trees.insert(coset_index, tree),
            TreesHolder::Partial(trees) => trees.insert(coset_index, tree),
            TreesHolder::None => drop(tree),
        };
        Ok(())
    }

    pub(crate) fn extend_and_commit(
        &mut self,
        source_coset_index: usize,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let tree_caps = allocate_tree_caps(self.log_lde_factor, self.log_tree_cap_size);
        assert!(self.tree_caps.replace(tree_caps).is_none());
        self.commit_and_transfer_tree_caps(source_coset_index, context)?;
        self.extend(source_coset_index, context)?;
        self.commit_and_transfer_tree_caps(1 - source_coset_index, context)?;
        Ok(())
    }
}

impl TraceHolderImpl for TraceHolder<E4> {
    fn ensure_coset_computed(
        &mut self,
        coset_index: usize,
        _context: &ProverContext,
    ) -> MetalResult<()> {
        assert!(coset_index < (1 << self.log_lde_factor));
        match &mut self.cosets {
            CosetsHolder::Full(evaluations) => {
                assert!(evaluations.len() > coset_index);
                Ok(())
            }
            CosetsHolder::Single {
                current_coset_index,
                ..
            } => {
                assert_eq!(
                    *current_coset_index, coset_index,
                    "E4 TraceHolder Single mode does not support coset switching"
                );
                Ok(())
            }
        }
    }
}

impl<T> TraceHolder<T>
where
    TraceHolder<T>: TraceHolderImpl,
{
    pub(crate) fn get_coset_evaluations(
        &mut self,
        coset_index: usize,
        context: &ProverContext,
    ) -> MetalResult<&MetalBuffer<T>> {
        self.ensure_coset_computed(coset_index, context)?;
        let evaluations = match &self.cosets {
            CosetsHolder::Full(evaluations) => &evaluations[coset_index],
            CosetsHolder::Single {
                evaluations,
                current_coset_index,
            } => {
                assert_eq!(*current_coset_index, coset_index);
                evaluations
            }
        };
        Ok(evaluations)
    }

    pub(crate) fn get_uninit_coset_evaluations_mut(
        &mut self,
        coset_index: usize,
    ) -> &mut MetalBuffer<T> {
        match &mut self.cosets {
            CosetsHolder::Full(evaluations) => &mut evaluations[coset_index],
            CosetsHolder::Single {
                evaluations,
                current_coset_index,
            } => {
                *current_coset_index = coset_index;
                evaluations
            }
        }
    }

    pub(crate) fn get_evaluations(
        &mut self,
        context: &ProverContext,
    ) -> MetalResult<&MetalBuffer<T>> {
        self.get_coset_evaluations(0, context)
    }

    pub(crate) fn get_uninit_evaluations_mut(&mut self) -> &mut MetalBuffer<T> {
        self.get_uninit_coset_evaluations_mut(0)
    }
}

// ---- Allocation helpers ----

pub(crate) fn allocate_coset<T>(
    log_domain_size: u32,
    columns_count: usize,
    pad_to_even: bool,
    context: &ProverContext,
) -> MetalResult<MetalBuffer<T>> {
    let columns_count = if pad_to_even {
        columns_count.next_multiple_of(2)
    } else {
        columns_count
    };
    let size = columns_count << log_domain_size;
    context.alloc(size)
}

fn allocate_cosets<T>(
    instances_count: usize,
    log_domain_size: u32,
    columns_count: usize,
    pad_to_even: bool,
    context: &ProverContext,
) -> MetalResult<Vec<MetalBuffer<T>>> {
    let mut result = Vec::with_capacity(instances_count);
    for _ in 0..instances_count {
        result.push(allocate_coset(
            log_domain_size,
            columns_count,
            pad_to_even,
            context,
        )?);
    }
    Ok(result)
}

fn allocate_tree(
    log_domain_size: u32,
    log_rows_per_leaf: u32,
    context: &ProverContext,
) -> MetalResult<MetalBuffer<Digest>> {
    let size = 1 << (log_domain_size + 1 - log_rows_per_leaf);
    context.alloc(size)
}

fn allocate_trees(
    instances_count: usize,
    log_domain_size: u32,
    log_rows_per_leaf: u32,
    context: &ProverContext,
) -> MetalResult<Vec<MetalBuffer<Digest>>> {
    let mut result = Vec::with_capacity(instances_count);
    for _ in 0..instances_count {
        result.push(allocate_tree(log_domain_size, log_rows_per_leaf, context)?);
    }
    Ok(result)
}

pub(crate) fn allocate_tree_caps(
    log_lde_factor: u32,
    log_tree_cap_size: u32,
) -> Vec<HostAllocation<[Digest]>> {
    let lde_factor = 1 << log_lde_factor;
    let log_coset_tree_cap_size = log_tree_cap_size - log_lde_factor;
    let coset_tree_cap_size = 1 << log_coset_tree_cap_size;
    let mut result = Vec::with_capacity(lde_factor);
    for _ in 0..lde_factor {
        let tree_cap = unsafe { HostAllocation::new_uninit_slice(coset_tree_cap_size) };
        result.push(tree_cap);
    }
    result
}

// ---- NTT / coset evaluation helpers ----

fn make_evaluations_sum_to_zero(
    evaluations: &mut MetalBuffer<BF>,
    log_domain_size: u32,
    columns_count: usize,
    padded_to_even: bool,
    context: &ProverContext,
) -> MetalResult<()> {
    let domain_size = 1usize << log_domain_size;
    let device = context.device();

    // GPU: segmented reduce (sum first n-1 rows per column) + negate & scatter to last row
    let num_cols = columns_count as u32;
    let stride = domain_size as u32;
    let d_col_sums: MetalBuffer<BF> = context.alloc(columns_count)?;

    let cmd_buf = context.new_command_buffer()?;
    crate::ops_cub::device_reduce::segmented_reduce_bf(
        device, &cmd_buf,
        crate::ops_cub::device_reduce::ReduceOperation::Sum,
        evaluations,
        &d_col_sums,
        stride,
        num_cols,
        stride - 1, // sum first n-1 rows (exclude last)
    )?;
    let config = crate::metal_runtime::dispatch::MetalLaunchConfig::basic_1d(
        (num_cols + 255) / 256, 256,
    );
    crate::metal_runtime::dispatch::dispatch_kernel(
        device, &cmd_buf,
        "ab_negate_and_scatter_to_last_row_kernel",
        &config,
        |encoder| {
            crate::metal_runtime::dispatch::set_buffer(encoder, 0, d_col_sums.raw(), 0);
            crate::metal_runtime::dispatch::set_buffer(encoder, 1, evaluations.raw(), 0);
            unsafe {
                crate::metal_runtime::dispatch::set_bytes(encoder, 2, &num_cols);
                crate::metal_runtime::dispatch::set_bytes(encoder, 3, &stride);
            }
        },
    )?;

    // Zero padding columns if needed (GPU memset with byte offset)
    if padded_to_even && columns_count % 2 != 0 {
        let count = domain_size as u32;
        let pad_offset = columns_count * domain_size * std::mem::size_of::<BF>();
        let config = crate::metal_runtime::dispatch::MetalLaunchConfig::basic_1d(
            (count + 255) / 256, 256,
        );
        crate::metal_runtime::dispatch::dispatch_kernel(
            device, &cmd_buf, "ab_memset_zero_u32_kernel", &config,
            |encoder| {
                crate::metal_runtime::dispatch::set_buffer(
                    encoder, 0, evaluations.raw(), pad_offset,
                );
                unsafe {
                    crate::metal_runtime::dispatch::set_bytes(encoder, 1, &count);
                }
            },
        )?;
    }

    cmd_buf.commit_and_wait();
    Ok(())
}

/// Compute coset evaluations from main domain evaluations.
/// This is the core LDE (Low-Degree Extension) operation.
///
/// Mirrors the CUDA version's compute_coset_evaluations logic:
/// - source_coset_index == 0: N2B(main) -> B2N(coset) to produce coset evals
/// - source_coset_index == 1: N2B(coset/compressed) -> B2N(main) to produce main evals
pub(crate) fn compute_coset_evaluations(
    src: &MetalBuffer<BF>,
    dst: &mut MetalBuffer<BF>,
    source_coset_index: usize,
    log_domain_size: u32,
    log_lde_factor: u32,
    compressed_coset: bool,
    context: &ProverContext,
) -> MetalResult<()> {
    assert_eq!(log_lde_factor, 1);
    let len = src.len();
    assert_eq!(len, dst.len());
    let domain_size = 1usize << log_domain_size;
    assert_eq!(len & ((domain_size << 1) - 1), 0);
    let num_bf_cols = (len >> log_domain_size) as u32;
    let log_n = log_domain_size;
    let stride = domain_size as u32;
    let device = context.device();
    let cmd_buf = context.new_command_buffer()?;
    let twiddles = context.ntt_twiddles();

    if source_coset_index == 0 {
        // Main evals -> coset evals:
        // N2B (main domain, no coset unscaling) then B2N (coset=1)
        crate::ntt::natural_trace_main_evals_to_bitrev_Z(
            device, &cmd_buf, src, dst, stride, log_n, num_bf_cols, twiddles,
        )?;
        // In-place B2N from bitrev Z to natural coset evals
        let const_dst = unsafe { &*(dst as *const MetalBuffer<BF>) };
        crate::ntt::bitrev_Z_to_natural_trace_coset_evals(
            device, &cmd_buf, const_dst, dst, stride, log_n, num_bf_cols, twiddles,
        )?;
    } else {
        assert_eq!(source_coset_index, 1);
        if compressed_coset {
            // Compressed coset evals -> bitrev Z -> main evals
            crate::ntt::natural_compressed_coset_evals_to_bitrev_Z(
                device, &cmd_buf, src, dst, stride, log_n, num_bf_cols, twiddles,
            )?;
        } else {
            // Composition coset evals -> bitrev Z -> main evals
            crate::ntt::natural_composition_coset_evals_to_bitrev_Z(
                device, &cmd_buf, src, dst, stride, log_n, num_bf_cols, twiddles,
            )?;
        }
        let const_dst = unsafe { &*(dst as *const MetalBuffer<BF>) };
        crate::ntt::bitrev_Z_to_natural_composition_main_evals(
            device, &cmd_buf, const_dst, dst, stride, log_n, num_bf_cols, twiddles,
        )?;
    }
    cmd_buf.commit_and_wait();
    Ok(())
}

fn switch_coset_evaluations_in_place(
    evals: &mut MetalBuffer<BF>,
    source_coset_index: usize,
    log_domain_size: u32,
    log_lde_factor: u32,
    compressed_coset: bool,
    context: &ProverContext,
) -> MetalResult<()> {
    // For in-place switching, we create an aliasing immutable reference via unsafe,
    // mirroring the CUDA version's approach with raw pointers.
    let const_evals = unsafe { &*(evals as *const MetalBuffer<BF>) };
    compute_coset_evaluations(
        const_evals,
        evals,
        source_coset_index,
        log_domain_size,
        log_lde_factor,
        compressed_coset,
        context,
    )
}

pub(crate) fn commit_trace(
    lde: &MetalBuffer<BF>,
    tree: &mut MetalBuffer<Digest>,
    log_domain_size: u32,
    log_lde_factor: u32,
    log_rows_per_leaf: u32,
    log_tree_cap_size: u32,
    _columns_count: usize,
    context: &ProverContext,
) -> MetalResult<()> {
    assert!(log_tree_cap_size >= log_lde_factor);
    let tree_len = 1usize << (log_domain_size + 1 - log_rows_per_leaf);
    assert_eq!(tree.len(), tree_len);
    let log_coset_tree_cap_size = log_tree_cap_size - log_lde_factor;
    let layers_count = log_domain_size + 1 - log_rows_per_leaf - log_coset_tree_cap_size;
    let device = context.device();
    let cmd_buf = context.new_command_buffer()?;

    // Tree layout: [leaves_count digests | internal nodes]
    // CUDA does: build_merkle_tree_leaves(values, leaves, ...) then
    //            build_merkle_tree_nodes(leaves, nodes, layers_count - 1, ...)
    // We replicate this using buffer offsets since MetalBuffer can't be split.
    let domain_size = 1usize << log_domain_size;
    let leaves_count = domain_size >> log_rows_per_leaf;
    let digest_size = std::mem::size_of::<blake2s::Digest>();
    let nodes_byte_offset = leaves_count * digest_size;

    // Use actual columns_count (not padded buffer size) so leaf hashes match CPU
    // for traces with odd column counts (e.g., delegation memory 243 cols).
    let bf_len_for_hash = _columns_count * domain_size;
    blake2s::build_merkle_tree_leaves_raw(
        device,
        &cmd_buf,
        lde.raw(),
        bf_len_for_hash,
        tree,
        leaves_count,
        log_rows_per_leaf,
    )?;

    // Bit-reverse leaf hashes before building nodes (matching CUDA/CPU prover).
    // CRITICAL: use a SEPARATE temp buffer for output to avoid GPU race conditions.
    // In-place bit-reversal (src == dst) causes data corruption because parallel threads
    // read from and write to the same buffer simultaneously.
    {
        let log_leaves = (leaves_count as u32).trailing_zeros();
        let count = leaves_count as u32;
        let cols = 1u32;
        let temp_buf: crate::metal_runtime::MetalBuffer<Digest> = context.alloc(leaves_count)?;
        let config = crate::metal_runtime::dispatch::MetalLaunchConfig::basic_2d(
            ((count + 127) / 128, 1), (128, 1));
        crate::metal_runtime::dispatch::dispatch_kernel(
            device, &cmd_buf, "ab_bit_reverse_naive_dg_kernel", &config,
            |encoder| {
                crate::metal_runtime::dispatch::set_buffer(encoder, 0, tree.raw(), 0); // src = tree leaves
                unsafe {
                    crate::metal_runtime::dispatch::set_bytes(encoder, 1, &count);
                }
                crate::metal_runtime::dispatch::set_buffer(encoder, 2, temp_buf.raw(), 0); // dst = temp
                unsafe {
                    crate::metal_runtime::dispatch::set_bytes(encoder, 3, &count);
                    crate::metal_runtime::dispatch::set_bytes(encoder, 4, &log_leaves);
                    crate::metal_runtime::dispatch::set_bytes(encoder, 5, &cols);
                }
            },
        )?;
        // GPU copy bit-reversed leaves back from temp to tree (no CPU sync needed)
        let copy_bytes = leaves_count * digest_size;
        crate::ops_simple::memcpy_gpu(device, &cmd_buf, temp_buf.raw(), tree.raw(), copy_bytes)?;
    }

    // Build merkle tree nodes from bit-reversed leaves (same cmd_buf, no sync)
    // Metal guarantees sequential dispatch within a command buffer.
    if layers_count > 0 {
        blake2s::build_merkle_tree_nodes_with_offset(
            device,
            &cmd_buf,
            tree,
            0,
            leaves_count,
            tree,
            nodes_byte_offset,
            layers_count - 1,
        )?;
    }
    cmd_buf.commit_and_wait();
    Ok(())
}

/// Commit an E4 trace buffer to a Merkle tree.
/// The E4 buffer is passed as raw MTLBuffer with explicit BF column count,
/// since E4 = [BF; 4] and the blake2s kernel operates on BF data.
fn commit_trace_e4(
    lde: &MetalBuffer<E4>,
    tree: &mut MetalBuffer<Digest>,
    log_domain_size: u32,
    log_lde_factor: u32,
    log_rows_per_leaf: u32,
    log_tree_cap_size: u32,
    context: &ProverContext,
) -> MetalResult<()> {
    assert!(log_tree_cap_size >= log_lde_factor);
    let tree_len = 1usize << (log_domain_size + 1 - log_rows_per_leaf);
    assert_eq!(tree.len(), tree_len);
    let log_coset_tree_cap_size = log_tree_cap_size - log_lde_factor;
    let layers_count = log_domain_size + 1 - log_rows_per_leaf - log_coset_tree_cap_size;
    let device = context.device();
    let cmd_buf = context.new_command_buffer()?;

    let bf_len = lde.len() * 4;
    let num_leaves = tree_len / 2;
    let digest_size = std::mem::size_of::<blake2s::Digest>();
    blake2s::build_merkle_tree_leaves_raw(
        device,
        &cmd_buf,
        lde.raw(),
        bf_len,
        tree,
        num_leaves,
        log_rows_per_leaf,
    )?;

    // E4 trees: no bit-reversal of leaf hashes here.
    // The caller is responsible for bit-reversing E4 DATA before calling this
    // (done in stage_4 and stage_5, matching CUDA's bit_reverse_in_place).
    if layers_count > 0 {
        blake2s::build_merkle_tree_nodes_with_offset(
            device,
            &cmd_buf,
            tree,
            0,
            num_leaves,
            tree,
            num_leaves * digest_size,
            layers_count - 1,
        )?;
    }

    cmd_buf.commit_and_wait();
    Ok(())
}

pub(crate) fn transfer_tree_cap(
    tree: &MetalBuffer<Digest>,
    cap: &mut HostAllocation<[Digest]>,
    log_lde_factor: u32,
    log_tree_cap_size: u32,
) {
    let log_subtree_cap_size = log_tree_cap_size - log_lde_factor;
    let (offset, count) = blake2s::merkle_tree_cap(tree, log_subtree_cap_size);
    // On Metal with unified memory, direct copy from buffer
    let accessor = cap.get_mut_accessor();
    let cap_slice = unsafe { accessor.get_mut() };
    assert_eq!(cap_slice.len(), count);
    let tree_slice = unsafe { tree.as_slice() };
    cap_slice.copy_from_slice(&tree_slice[offset..offset + count]);
}

pub(crate) fn flatten_tree_caps(
    accessors: &[UnsafeAccessor<[Digest]>],
) -> impl Iterator<Item = u32> + use<'_> {
    accessors
        .iter()
        .flat_map(|accessor| unsafe { accessor.get().to_vec() })
        .flatten()
}

pub(crate) fn get_tree_caps(accessors: &[UnsafeAccessor<[Digest]>]) -> Vec<MerkleTreeCapVarLength> {
    accessors
        .iter()
        .map(|accessor| unsafe { accessor.get().to_vec() })
        .map(|cap| MerkleTreeCapVarLength { cap })
        .collect_vec()
}

pub(crate) fn split_evaluations_pair(
    evaluations: &mut [MetalBuffer<BF>],
    coset_index: usize,
) -> (&MetalBuffer<BF>, &mut MetalBuffer<BF>) {
    assert_eq!(evaluations.len(), 2);
    let (src, dst) = evaluations.split_at_mut(1);
    if coset_index == 0 {
        (&src[0], &mut dst[0])
    } else {
        (&dst[0], &mut src[0])
    }
}
