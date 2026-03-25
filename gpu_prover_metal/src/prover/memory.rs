//! Memory commitment for Metal.
//! Computes the memory trace from tracing data and commits it to a Merkle tree.
//! Ported from gpu_prover/src/prover/memory.rs.

use super::trace_holder::{get_tree_caps, TraceHolder, TreesCacheMode};
use super::BF;
use crate::metal_runtime::MetalResult;
use crate::prover::context::ProverContext;
use crate::prover::tracing_data::{TracingDataDevice, TracingDataTransfer};
use cs::one_row_compiler::CompiledCircuitArtifact;
use fft::GoodAllocator;
use prover::merkle_trees::MerkleTreeCapVarLength;

pub struct MemoryCommitmentResult {
    pub tree_caps: Vec<MerkleTreeCapVarLength>,
}

impl MemoryCommitmentResult {
    /// Consume into tree caps.
    /// Returns (tree_caps, elapsed_ms). On Metal, elapsed is always 0.0.
    pub fn finish(self) -> MetalResult<(Vec<MerkleTreeCapVarLength>, f64)> {
        Ok((self.tree_caps, 0.0))
    }
}

pub fn commit_memory<'a>(
    tracing_data_transfer: TracingDataTransfer<'a, impl GoodAllocator>,
    circuit: &CompiledCircuitArtifact<BF>,
    log_lde_factor: u32,
    log_tree_cap_size: u32,
    context: &ProverContext,
) -> MetalResult<MemoryCommitmentResult> {
    let trace_len = circuit.trace_len;
    assert!(trace_len.is_power_of_two());
    let log_domain_size = trace_len.trailing_zeros();
    let memory_subtree = &circuit.memory_layout;
    let memory_columns_count = memory_subtree.total_width;

    let mut memory_holder = TraceHolder::new(
        log_domain_size,
        log_lde_factor,
        0,
        log_tree_cap_size,
        memory_columns_count,
        true,
        true,
        false,
        TreesCacheMode::CacheFull,
        context,
    )?;

    let TracingDataTransfer {
        circuit_type: _,
        data_host: _,
        data_device,
        _lifetime: _,
    } = tracing_data_transfer;

    let device = context.device();
    let cmd_buf = context.new_command_buffer()?;
    let stride = trace_len as u32;

    // Fill memory buffer from tracing data (matching CUDA's commit_memory)
    let evaluations = memory_holder.get_uninit_evaluations_mut();
    // Zero the buffer on GPU (Metal doesn't zero allocations)
    crate::ops_simple::memset_zero(device, &cmd_buf, evaluations.raw(), evaluations.byte_len())?;

    match data_device {
        TracingDataDevice::Main {
            setup_and_teardown,
            trace,
        } => {
            let count = trace.cycle_data.len() as u32;
            crate::witness::memory_main::generate_memory_values_main(
                device,
                &cmd_buf,
                memory_subtree,
                &setup_and_teardown,
                &trace,
                evaluations,
                stride,
                count,
            )?;
        }
        TracingDataDevice::Delegation(trace) => {
            let count = trace.num_requests as u32;
            crate::witness::memory_delegation::generate_memory_values_delegation(
                device,
                &cmd_buf,
                memory_subtree,
                &trace,
                evaluations,
                stride,
                count,
                context,
            )?;
        }
    };
    cmd_buf.commit_and_wait();

    memory_holder.make_evaluations_sum_to_zero_extend_and_commit(context)?;

    let tree_caps = get_tree_caps(&memory_holder.get_tree_caps_accessors());

    Ok(MemoryCommitmentResult { tree_caps })
}
