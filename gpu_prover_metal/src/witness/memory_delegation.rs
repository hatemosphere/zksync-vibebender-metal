use super::layout::DelegationProcessingLayout;
use super::ram_access::{
    RegisterAndIndirectAccessDescription, RegisterAndIndirectAccessTimestampComparisonAuxVars,
};
use super::trace_delegation::DelegationTraceDevice;
use super::BF;
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use crate::prover::context::ProverContext;
use cs::definitions::MemorySubtree;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

/// Pad a Metal buffer to `target_len` elements, zero-filling the extra entries.
fn pad_buffer<T: Copy + Default>(
    src: &MetalBuffer<T>,
    target_len: usize,
    context: &ProverContext,
) -> MetalResult<MetalBuffer<T>> {
    if src.len() >= target_len {
        // Buffer is already large enough — create a view/copy
        let mut dst = context.alloc::<T>(target_len)?;
        unsafe {
            let src_slice = src.as_slice();
            let dst_slice = dst.as_mut_slice();
            dst_slice[..target_len.min(src_slice.len())].copy_from_slice(&src_slice[..target_len.min(src_slice.len())]);
        }
        return Ok(dst);
    }
    let mut dst = context.alloc::<T>(target_len)?;
    unsafe {
        // Zero the entire buffer
        std::ptr::write_bytes(dst.as_mut_ptr() as *mut u8, 0, target_len * std::mem::size_of::<T>());
        // Copy actual data
        let src_slice = src.as_slice();
        let dst_slice = dst.as_mut_slice();
        dst_slice[..src_slice.len()].copy_from_slice(src_slice);
    }
    Ok(dst)
}

const WARP_SIZE: u32 = 32;
const MAX_REGISTER_AND_INDIRECT_ACCESSES_COUNT: usize = 4;
const MAX_INDIRECT_ACCESS_REGISTERS: usize = 2;
const MAX_INDIRECT_ACCESS_WORDS: usize = 24;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct DelegationMemorySubtree {
    delegation_processor_layout: DelegationProcessingLayout,
    register_and_indirect_accesses_count: u32,
    register_and_indirect_accesses:
        [RegisterAndIndirectAccessDescription; MAX_REGISTER_AND_INDIRECT_ACCESSES_COUNT],
}

impl From<&MemorySubtree> for DelegationMemorySubtree {
    fn from(value: &MemorySubtree) -> Self {
        assert!(value.shuffle_ram_inits_and_teardowns.is_none());
        assert!(value.shuffle_ram_access_sets.is_empty());
        assert!(value.delegation_request_layout.is_none());
        assert_eq!(value.batched_ram_accesses.len(), 0);
        let delegation_processor_layout = value.delegation_processor_layout.unwrap().into();
        let register_and_indirect_accesses_count =
            value.register_and_indirect_accesses.len() as u32;
        assert!(
            register_and_indirect_accesses_count <= MAX_REGISTER_AND_INDIRECT_ACCESSES_COUNT as u32
        );
        let mut register_and_indirect_accesses = [RegisterAndIndirectAccessDescription::default();
            MAX_REGISTER_AND_INDIRECT_ACCESSES_COUNT];
        for (i, value) in value.register_and_indirect_accesses.iter().enumerate() {
            register_and_indirect_accesses[i] = value.clone().into();
        }
        Self {
            delegation_processor_layout,
            register_and_indirect_accesses_count,
            register_and_indirect_accesses,
        }
    }
}

fn encode_indirect_accesses_properties(trace: &DelegationTraceDevice) -> Vec<u32> {
    let mut props = vec![0u32; MAX_INDIRECT_ACCESS_REGISTERS * MAX_INDIRECT_ACCESS_WORDS];
    for (i, access) in trace.indirect_accesses_properties.iter().enumerate() {
        if i >= MAX_INDIRECT_ACCESS_REGISTERS { break; }
        for (j, loc) in access.iter().enumerate() {
            if j >= MAX_INDIRECT_ACCESS_WORDS { break; }
            props[i * MAX_INDIRECT_ACCESS_WORDS + j] =
                ((loc.use_writes as u32) << 31) | (loc.index as u32);
        }
    }
    props
}

/// Kernel expects (from memory_delegation.metal):
///   buffer(0) = subtree (constant)
///   buffer(1) = num_requests
///   buffer(2) = num_register_accesses
///   buffer(3) = num_indirect_reads
///   buffer(4) = num_indirect_writes
///   buffer(5) = base_register_index
///   buffer(6) = delegation_type
///   buffer(7) = indirect_accesses_properties
///   buffer(8) = write_timestamp_data
///   buffer(9) = register_accesses_data
///   buffer(10) = indirect_reads_data
///   buffer(11) = indirect_writes_data
///   buffer(12) = memory_ptr
///   buffer(13) = stride
///   buffer(14) = count
pub(crate) fn generate_memory_values_delegation(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    subtree: &MemorySubtree,
    trace: &DelegationTraceDevice,
    memory: &MetalBuffer<BF>,
    stride: u32,
    count: u32,
    context: &ProverContext,
) -> MetalResult<()> {
    let subtree: DelegationMemorySubtree = subtree.into();
    let props = encode_indirect_accesses_properties(trace);
    let d_props = context.alloc_from_slice(&props)?;

    // subtree exceeds Metal's 4KB setBytes limit — must use device buffer
    let d_subtree = context.alloc_from_slice(unsafe {
        std::slice::from_raw_parts(
            &subtree as *const DelegationMemorySubtree as *const u8,
            std::mem::size_of::<DelegationMemorySubtree>(),
        )
    })?;

    let block_dim = WARP_SIZE * 4;
    let grid_dim = (count + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_generate_memory_values_delegation_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            let actual_num_requests = trace.write_timestamp.len() as u32;
            set_buffer(encoder, idx, d_subtree.raw(), 0);
            idx += 1;
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
            set_buffer(encoder, idx, memory.raw(), 0);
            idx += 1;
            unsafe { set_bytes(encoder, idx, &stride); }
            idx += 1;
            unsafe { set_bytes(encoder, idx, &count); }
        },
    )
}

/// Kernel expects (from memory_delegation.metal):
///   buffer(0) = subtree_const
///   buffer(1) = aux_vars_const
///   buffer(2) = num_requests
///   buffer(3) = num_register_accesses
///   buffer(4) = num_indirect_reads
///   buffer(5) = num_indirect_writes
///   buffer(6) = base_register_index
///   buffer(7) = delegation_type
///   buffer(8) = indirect_accesses_properties
///   buffer(9) = write_timestamp_data
///   buffer(10) = register_accesses_data
///   buffer(11) = indirect_reads_data
///   buffer(12) = indirect_writes_data
///   buffer(13) = memory_ptr
///   buffer(14) = witness_ptr
///   buffer(15) = stride
///   buffer(16) = count
pub(crate) fn generate_memory_and_witness_values_delegation(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    subtree: &MemorySubtree,
    aux_vars: &cs::definitions::RegisterAndIndirectAccessTimestampComparisonAuxVars,
    trace: &DelegationTraceDevice,
    memory: &MetalBuffer<BF>,
    witness: &MetalBuffer<BF>,
    stride: u32,
    count: u32,
    context: &ProverContext,
) -> MetalResult<()> {
    let subtree: DelegationMemorySubtree = subtree.into();
    let aux_vars: RegisterAndIndirectAccessTimestampComparisonAuxVars = aux_vars.into();
    let props = encode_indirect_accesses_properties(trace);
    let d_props = context.alloc_from_slice(&props)?;

    // Dump first 64 bytes of subtree to compare with Metal kernel's view
    let subtree_bytes = unsafe {
        std::slice::from_raw_parts(
            &subtree as *const DelegationMemorySubtree as *const u8,
            std::cmp::min(64, std::mem::size_of::<DelegationMemorySubtree>()),
        )
    };
    let hex: Vec<String> = subtree_bytes.iter().map(|b| format!("{:02x}", b)).collect();
    log::info!("delegation subtree first 64 bytes: {}", hex.join(" "));
    log::info!("delegation: reg_and_indirect_count={}, num_requests={}, stride={}, count={}",
        subtree.register_and_indirect_accesses_count, trace.num_requests, stride, count);

    // subtree exceeds Metal's 4KB setBytes limit — must use device buffer
    let d_subtree = context.alloc_from_slice(unsafe {
        std::slice::from_raw_parts(
            &subtree as *const DelegationMemorySubtree as *const u8,
            std::mem::size_of::<DelegationMemorySubtree>(),
        )
    })?;

    let block_dim = WARP_SIZE * 4;
    let grid_dim = (count + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_generate_memory_and_witness_values_delegation_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            let actual_num_requests = trace.write_timestamp.len() as u32;
            set_buffer(encoder, idx, d_subtree.raw(), 0);
            idx += 1;
            unsafe { set_bytes(encoder, idx, &aux_vars); }
            idx += 1;
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
            set_buffer(encoder, idx, memory.raw(), 0);
            idx += 1;
            set_buffer(encoder, idx, witness.raw(), 0);
            idx += 1;
            unsafe { set_bytes(encoder, idx, &stride); }
            idx += 1;
            unsafe { set_bytes(encoder, idx, &count); }
        },
    )
}
