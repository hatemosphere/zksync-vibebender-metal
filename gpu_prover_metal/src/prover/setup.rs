//! Setup precomputations for Metal.
//! Ported from gpu_prover/src/prover/setup.rs.

use super::context::ProverContext;
use super::trace_holder::{get_tree_caps, TraceHolder, TreesCacheMode, TreesHolder};
use super::transfer::Transfer;
use super::BF;
use crate::blake2s::Digest;
use crate::metal_runtime::MetalResult;
use cs::one_row_compiler::CompiledCircuitArtifact;
use fft::GoodAllocator;
use prover::merkle_trees::MerkleTreeCapVarLength;
use std::sync::Arc;

#[derive(Clone)]
pub struct SetupTreesAndCaps {
    pub trees: Vec<Arc<Box<[Digest]>>>,
    pub caps: Arc<Vec<MerkleTreeCapVarLength>>,
}

pub struct SetupPrecomputations<'a> {
    pub(crate) trace_holder: TraceHolder<BF>,
    pub(crate) transfer: Transfer<'a>,
    pub(crate) trees_and_caps: SetupTreesAndCaps,
    pub(crate) is_extended: bool,
}

impl<'a> SetupPrecomputations<'a> {
    pub fn new(
        circuit: &CompiledCircuitArtifact<BF>,
        log_lde_factor: u32,
        log_tree_cap_size: u32,
        recompute_cosets: bool,
        trees_and_caps: SetupTreesAndCaps,
        context: &ProverContext,
    ) -> MetalResult<Self> {
        let trace_len = circuit.trace_len;
        assert!(trace_len.is_power_of_two());
        let log_domain_size = trace_len.trailing_zeros();
        let columns_count = circuit.setup_layout.total_width;
        let trace_holder = TraceHolder::new(
            log_domain_size,
            log_lde_factor,
            0,
            log_tree_cap_size,
            columns_count,
            true,
            true,
            recompute_cosets,
            TreesCacheMode::CacheNone,
            context,
        )?;
        let transfer = Transfer::new();
        transfer.record_allocated(context)?;
        Ok(Self {
            trace_holder,
            transfer,
            trees_and_caps,
            is_extended: false,
        })
    }

    pub fn schedule_transfer(
        &mut self,
        trace: Arc<Vec<BF, impl GoodAllocator + 'a>>,
        context: &ProverContext,
    ) -> MetalResult<()> {
        let dst = self.trace_holder.get_uninit_evaluations_mut();
        self.transfer.schedule(trace, dst, context)?;
        self.transfer.record_transferred(context)
    }

    pub fn ensure_is_extended(&mut self, context: &ProverContext) -> MetalResult<()> {
        if self.is_extended {
            return Ok(());
        }
        self.extend(context)
    }

    fn extend(&mut self, context: &ProverContext) -> MetalResult<()> {
        {
            let _g = crate::cpu_scoped!("setup_transfer");
            self.transfer.ensure_transferred(context)?;
        }
        {
            let _g = crate::cpu_scoped!("setup_extend");
            self.trace_holder
                .make_evaluations_sum_to_zero_and_extend(context)?;
        }
        self.is_extended = true;
        Ok(())
    }

    pub fn get_trees_and_caps(
        circuit: &CompiledCircuitArtifact<BF>,
        log_lde_factor: u32,
        log_tree_cap_size: u32,
        trace: Arc<Vec<BF, impl GoodAllocator>>,
        context: &ProverContext,
    ) -> MetalResult<SetupTreesAndCaps> {
        let trace_len = circuit.trace_len;
        assert!(trace_len.is_power_of_two());
        let log_domain_size = trace_len.trailing_zeros();
        let columns_count = circuit.setup_layout.total_width;
        let mut trace_holder = TraceHolder::new(
            log_domain_size,
            log_lde_factor,
            0,
            log_tree_cap_size,
            columns_count,
            true,
            true,
            false,
            TreesCacheMode::CacheFull,
            context,
        )?;
        let mut transfer = Transfer::new();
        transfer.record_allocated(context)?;
        let dst = trace_holder.get_uninit_evaluations_mut();
        transfer.schedule(trace, dst, context)?;
        transfer.record_transferred(context)?;
        transfer.ensure_transferred(context)?;
        trace_holder.make_evaluations_sum_to_zero_extend_and_commit(context)?;

        let caps = get_tree_caps(&trace_holder.get_tree_caps_accessors());

        // On Metal with unified memory, tree data is directly accessible.
        // Copy tree buffers to host Boxes (matching CUDA interface).
        let d_trees = match &trace_holder.trees {
            TreesHolder::Full(trees) => trees,
            _ => unreachable!(),
        };
        let lde_factor = 1usize << log_lde_factor;
        let tree_len = 1usize << (log_domain_size + 1);

        let mut trees = Vec::with_capacity(lde_factor);
        for coset_index in 0..lde_factor {
            let d_tree = &d_trees[coset_index];
            assert_eq!(d_tree.len(), tree_len);
            let tree_slice = unsafe { d_tree.as_slice() };
            let mut host_tree = unsafe { Box::new_uninit_slice(tree_len).assume_init() };
            host_tree.copy_from_slice(tree_slice);
            trees.push(Arc::new(host_tree));
        }

        let caps = Arc::new(caps);
        Ok(SetupTreesAndCaps { trees, caps })
    }
}
