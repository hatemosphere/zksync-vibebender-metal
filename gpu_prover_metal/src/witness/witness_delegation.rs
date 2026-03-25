use super::trace_delegation::DelegationTraceDevice;
use super::BF;
use crate::circuit_type::DelegationCircuitType;
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use crate::prover::context::ProverContext;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

const WARP_SIZE: u32 = 32;
const MAX_INDIRECT_ACCESS_REGISTERS: usize = 2;
const MAX_INDIRECT_ACCESS_WORDS: usize = 24;

/// Dispatch delegation witness kernel.
/// Uses DELEGATION_KERNEL macro which expects individual buffer parameters
/// (not a serialized struct, which has device-pointer issues on Metal).
///
/// Buffer layout:
///   0  = num_requests (u32)
///   1  = num_register_accesses (u32)
///   2  = num_indirect_reads (u32)
///   3  = num_indirect_writes (u32)
///   4  = base_register_index (u32)
///   5  = delegation_type (u16)
///   6  = indirect_accesses_properties (buffer)
///   7  = write_timestamp (buffer)
///   8  = register_accesses (buffer)
///   9  = indirect_reads (buffer)
///   10 = indirect_writes (buffer)
///   11 = generic_lookup_tables (buffer)
///   12 = memory (buffer)
///   13 = witness (buffer)
///   14 = lookup_mappings (buffer)
///   15 = stride (u32)
///   16 = count (u32)
pub fn generate_witness_values_delegation(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    circuit_type: DelegationCircuitType,
    trace: &DelegationTraceDevice,
    generic_lookup_tables: &MetalBuffer<BF>,
    generic_lookup_tables_offset: usize,
    memory: &MetalBuffer<BF>,
    witness: &MetalBuffer<BF>,
    lookup_mapping: &MetalBuffer<u32>,
    stride: u32,
    count: u32,
    context: &ProverContext,
) -> MetalResult<()> {
    let kernel_name = match circuit_type {
        DelegationCircuitType::BigIntWithControl => {
            "ab_generate_bigint_with_control_witness_kernel"
        }
        DelegationCircuitType::Blake2WithCompression => {
            "ab_generate_blake2_with_compression_witness_kernel"
        }
    };

    // Encode indirect_accesses_properties into a flat buffer
    let mut props = vec![0u32; MAX_INDIRECT_ACCESS_REGISTERS * MAX_INDIRECT_ACCESS_WORDS];
    for (i, access) in trace.indirect_accesses_properties.iter().enumerate() {
        if i >= MAX_INDIRECT_ACCESS_REGISTERS { break; }
        for (j, loc) in access.iter().enumerate() {
            if j >= MAX_INDIRECT_ACCESS_WORDS { break; }
            props[i * MAX_INDIRECT_ACCESS_WORDS + j] =
                ((loc.use_writes as u32) << 31) | (loc.index as u32);
        }
    }
    let d_props = context.alloc_from_slice(&props)?;

    let block_dim = WARP_SIZE * 4;
    let grid_dim = (count + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(device, cmd_buf, kernel_name, &config, |encoder| {
        let mut idx = 0u32;
        let actual_num_requests = trace.write_timestamp.len() as u32;
        unsafe { set_bytes(encoder, idx, &actual_num_requests); }
        idx += 1;
        unsafe { set_bytes(encoder, idx, &(trace.num_register_accesses_per_delegation as u32)); }
        idx += 1;
        unsafe { set_bytes(encoder, idx, &(trace.num_indirect_reads_per_delegation as u32)); }
        idx += 1;
        unsafe { set_bytes(encoder, idx, &(trace.num_indirect_writes_per_delegation as u32)); }
        idx += 1;
        unsafe { set_bytes(encoder, idx, &trace.base_register_index); }
        idx += 1;
        unsafe { set_bytes(encoder, idx, &trace.delegation_type); }
        idx += 1;
        set_buffer(encoder, idx, d_props.raw(), 0);
        idx += 1;
        set_buffer(encoder, idx, trace.write_timestamp.raw(), 0);
        idx += 1;
        set_buffer(encoder, idx, trace.register_accesses.raw(), 0);
        idx += 1;
        set_buffer(encoder, idx, trace.indirect_reads.raw(), 0);
        idx += 1;
        set_buffer(encoder, idx, trace.indirect_writes.raw(), 0);
        idx += 1;
        set_buffer(encoder, idx, generic_lookup_tables.raw(), generic_lookup_tables_offset);
        idx += 1;
        set_buffer(encoder, idx, memory.raw(), 0);
        idx += 1;
        set_buffer(encoder, idx, witness.raw(), 0);
        idx += 1;
        set_buffer(encoder, idx, lookup_mapping.raw(), 0);
        idx += 1;
        unsafe { set_bytes(encoder, idx, &stride); }
        idx += 1;
        unsafe { set_bytes(encoder, idx, &count); }
    })
}
