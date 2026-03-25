//! Metal dispatch wrappers for stage 2 kernels.
//! Ports gpu_prover/src/prover/stage_2_kernels.rs from CUDA to Metal.

use super::arg_utils::*;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes};
use crate::metal_runtime::{MetalBuffer, MetalCommandBuffer, MetalLaunchConfig, MetalResult};
use field::{Mersenne31Field, Mersenne31Quartic};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

type BF = Mersenne31Field;
type E4 = Mersenne31Quartic;

const WARP_SIZE: u32 = 32;

pub fn launch_range_check_aggregated_entry_invs(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    challenges: &MetalBuffer<LookupChallenges>,
    witness_cols: &MetalBuffer<BF>,
    witness_stride: u32,
    setup_cols: &MetalBuffer<BF>,
    setup_stride: u32,
    stage_2_e4_cols: &MetalBuffer<BF>,
    stage_2_e4_byte_offset: usize,
    stage_2_e4_stride: u32,
    aggregated_entry_invs: &MetalBuffer<E4>,
    start_col_in_setup: u32,
    multiplicities_src_cols_start: u32,
    multiplicities_dst_cols_start: u32,
    num_multiplicities_cols: u32,
    num_table_rows_tail: u32,
    log_n: u32,
) -> MetalResult<()> {
    let n = 1u32 << log_n;
    let block_dim = WARP_SIZE * 4;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_range_check_aggregated_entry_invs_and_multiplicities_arg_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, challenges.raw(), 0); idx += 1;
            set_buffer(encoder, idx, witness_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &witness_stride); } idx += 1;
            set_buffer(encoder, idx, setup_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &setup_stride); } idx += 1;
            set_buffer(encoder, idx, stage_2_e4_cols.raw(), stage_2_e4_byte_offset); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_e4_stride); } idx += 1;
            set_buffer(encoder, idx, aggregated_entry_invs.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &start_col_in_setup); } idx += 1;
            unsafe { set_bytes(encoder, idx, &multiplicities_src_cols_start); } idx += 1;
            unsafe { set_bytes(encoder, idx, &multiplicities_dst_cols_start); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_multiplicities_cols); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_table_rows_tail); } idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); }
        },
    )
}

pub fn launch_generic_aggregated_entry_invs(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    challenges: &MetalBuffer<LookupChallenges>,
    witness_cols: &MetalBuffer<BF>,
    witness_stride: u32,
    setup_cols: &MetalBuffer<BF>,
    setup_stride: u32,
    stage_2_e4_cols: &MetalBuffer<BF>,
    stage_2_e4_byte_offset: usize,
    stage_2_e4_stride: u32,
    aggregated_entry_invs: &MetalBuffer<E4>,
    start_col_in_setup: u32,
    multiplicities_src_cols_start: u32,
    multiplicities_dst_cols_start: u32,
    num_multiplicities_cols: u32,
    num_table_rows_tail: u32,
    log_n: u32,
) -> MetalResult<()> {
    let n = 1u32 << log_n;
    let block_dim = WARP_SIZE * 4;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_generic_aggregated_entry_invs_and_multiplicities_arg_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, challenges.raw(), 0); idx += 1;
            set_buffer(encoder, idx, witness_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &witness_stride); } idx += 1;
            set_buffer(encoder, idx, setup_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &setup_stride); } idx += 1;
            set_buffer(encoder, idx, stage_2_e4_cols.raw(), stage_2_e4_byte_offset); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_e4_stride); } idx += 1;
            set_buffer(encoder, idx, aggregated_entry_invs.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &start_col_in_setup); } idx += 1;
            unsafe { set_bytes(encoder, idx, &multiplicities_src_cols_start); } idx += 1;
            unsafe { set_bytes(encoder, idx, &multiplicities_dst_cols_start); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_multiplicities_cols); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_table_rows_tail); } idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); }
        },
    )
}

pub fn launch_delegation_aux_poly(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    delegation_challenges: &MetalBuffer<DelegationChallenges>,
    request_metadata: &MetalBuffer<DelegationRequestMetadata>,
    processing_metadata: &MetalBuffer<DelegationProcessingMetadata>,
    memory_cols: &MetalBuffer<BF>,
    memory_stride: u32,
    setup_cols: &MetalBuffer<BF>,
    setup_stride: u32,
    stage_2_e4_cols: &MetalBuffer<BF>,
    stage_2_e4_byte_offset: usize,
    stage_2_e4_stride: u32,
    delegation_aux_poly_col: u32,
    handle_delegation_requests: bool,
    log_n: u32,
) -> MetalResult<()> {
    let n = 1u32 << log_n;
    let block_dim = 128;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);
    let handle_flag: u32 = if handle_delegation_requests { 1 } else { 0 };

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_delegation_aux_poly_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, delegation_challenges.raw(), 0); idx += 1;
            set_buffer(encoder, idx, request_metadata.raw(), 0); idx += 1;
            set_buffer(encoder, idx, processing_metadata.raw(), 0); idx += 1;
            set_buffer(encoder, idx, memory_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_stride); } idx += 1;
            set_buffer(encoder, idx, setup_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &setup_stride); } idx += 1;
            set_buffer(encoder, idx, stage_2_e4_cols.raw(), stage_2_e4_byte_offset); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_e4_stride); } idx += 1;
            unsafe { set_bytes(encoder, idx, &delegation_aux_poly_col); } idx += 1;
            unsafe { set_bytes(encoder, idx, &handle_flag); } idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); }
        },
    )
}

pub fn launch_lookup_args(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    range_check_16_layout: &MetalBuffer<RangeCheck16ArgsLayout>,
    expressions: &MetalBuffer<FlattenedLookupExpressionsLayout>,
    expressions_for_shuffle_ram: &MetalBuffer<FlattenedLookupExpressionsForShuffleRamLayout>,
    lazy_init_teardown_layout: &MetalBuffer<LazyInitTeardownLayout>,
    setup_cols: &MetalBuffer<BF>,
    setup_stride: u32,
    witness_cols: &MetalBuffer<BF>,
    witness_stride: u32,
    memory_cols: &MetalBuffer<BF>,
    memory_stride: u32,
    aggregated_entry_invs_for_range_check_16: &MetalBuffer<E4>,
    aggregated_entry_invs_for_timestamp_range_checks: &MetalBuffer<E4>,
    aggregated_entry_invs_for_generic_lookups: &MetalBuffer<E4>,
    generic_args_start: u32,
    num_generic_args: u32,
    generic_lookups_args_to_table_entries_map: &MetalBuffer<u32>,
    generic_lookups_map_stride: u32,
    stage_2_bf_cols: &MetalBuffer<BF>,
    stage_2_bf_stride: u32,
    stage_2_e4_cols: &MetalBuffer<BF>,
    stage_2_e4_byte_offset: usize,
    stage_2_e4_stride: u32,
    memory_timestamp_high_from_circuit_idx: BF,
    num_stage_2_bf_cols: u32,
    num_stage_2_e4_cols: u32,
    log_n: u32,
) -> MetalResult<()> {
    let n = 1u32 << log_n;
    let block_dim = 128;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_lookup_args_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, range_check_16_layout.raw(), 0); idx += 1;
            set_buffer(encoder, idx, expressions.raw(), 0); idx += 1;
            set_buffer(encoder, idx, expressions_for_shuffle_ram.raw(), 0); idx += 1;
            set_buffer(encoder, idx, lazy_init_teardown_layout.raw(), 0); idx += 1;
            set_buffer(encoder, idx, setup_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &setup_stride); } idx += 1;
            set_buffer(encoder, idx, witness_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &witness_stride); } idx += 1;
            set_buffer(encoder, idx, memory_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_stride); } idx += 1;
            set_buffer(encoder, idx, aggregated_entry_invs_for_range_check_16.raw(), 0); idx += 1;
            set_buffer(encoder, idx, aggregated_entry_invs_for_timestamp_range_checks.raw(), 0); idx += 1;
            set_buffer(encoder, idx, aggregated_entry_invs_for_generic_lookups.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &generic_args_start); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_generic_args); } idx += 1;
            set_buffer(encoder, idx, generic_lookups_args_to_table_entries_map.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &generic_lookups_map_stride); } idx += 1;
            set_buffer(encoder, idx, stage_2_bf_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_bf_stride); } idx += 1;
            set_buffer(encoder, idx, stage_2_e4_cols.raw(), stage_2_e4_byte_offset); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_e4_stride); } idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_timestamp_high_from_circuit_idx); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_stage_2_bf_cols); } idx += 1;
            unsafe { set_bytes(encoder, idx, &num_stage_2_e4_cols); } idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); }
        },
    )
}

pub fn launch_shuffle_ram_memory_args(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    memory_challenges: &MetalBuffer<MemoryChallenges>,
    shuffle_ram_accesses: &MetalBuffer<ShuffleRamAccesses>,
    setup_cols: &MetalBuffer<BF>,
    setup_stride: u32,
    memory_cols: &MetalBuffer<BF>,
    memory_stride: u32,
    stage_2_e4_cols: &MetalBuffer<BF>,
    stage_2_e4_byte_offset: usize,
    stage_2_e4_stride: u32,
    lazy_init_teardown_layout: &MetalBuffer<LazyInitTeardownLayout>,
    memory_timestamp_high_from_circuit_idx: BF,
    memory_args_start: u32,
    log_n: u32,
) -> MetalResult<()> {
    let n = 1u32 << log_n;
    let block_dim = 128;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_shuffle_ram_memory_args_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, memory_challenges.raw(), 0); idx += 1;
            set_buffer(encoder, idx, shuffle_ram_accesses.raw(), 0); idx += 1;
            set_buffer(encoder, idx, setup_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &setup_stride); } idx += 1;
            set_buffer(encoder, idx, memory_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_stride); } idx += 1;
            set_buffer(encoder, idx, stage_2_e4_cols.raw(), stage_2_e4_byte_offset); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_e4_stride); } idx += 1;
            set_buffer(encoder, idx, lazy_init_teardown_layout.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_timestamp_high_from_circuit_idx); } idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_args_start); } idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); }
        },
    )
}

pub fn launch_batched_ram_memory_args(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    memory_challenges: &MetalBuffer<MemoryChallenges>,
    batched_ram_accesses: &MetalBuffer<BatchedRamAccesses>,
    memory_cols: &MetalBuffer<BF>,
    memory_stride: u32,
    stage_2_e4_cols: &MetalBuffer<BF>,
    stage_2_e4_byte_offset: usize,
    stage_2_e4_stride: u32,
    memory_args_start: u32,
    log_n: u32,
) -> MetalResult<()> {
    let n = 1u32 << log_n;
    let block_dim = 128;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_batched_ram_memory_args_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, memory_challenges.raw(), 0); idx += 1;
            set_buffer(encoder, idx, batched_ram_accesses.raw(), 0); idx += 1;
            set_buffer(encoder, idx, memory_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_stride); } idx += 1;
            set_buffer(encoder, idx, stage_2_e4_cols.raw(), stage_2_e4_byte_offset); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_e4_stride); } idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_args_start); } idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); }
        },
    )
}

pub fn launch_register_and_indirect_memory_args(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    memory_challenges: &MetalBuffer<MemoryChallenges>,
    register_and_indirect_accesses: &MetalBuffer<RegisterAndIndirectAccesses>,
    memory_cols: &MetalBuffer<BF>,
    memory_stride: u32,
    stage_2_e4_cols: &MetalBuffer<BF>,
    stage_2_e4_byte_offset: usize,
    stage_2_e4_stride: u32,
    memory_args_start: u32,
    log_n: u32,
) -> MetalResult<()> {
    let n = 1u32 << log_n;
    let block_dim = 128;
    let grid_dim = (n + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_register_and_indirect_memory_args_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, memory_challenges.raw(), 0); idx += 1;
            set_buffer(encoder, idx, register_and_indirect_accesses.raw(), 0); idx += 1;
            set_buffer(encoder, idx, memory_cols.raw(), 0); idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_stride); } idx += 1;
            set_buffer(encoder, idx, stage_2_e4_cols.raw(), stage_2_e4_byte_offset); idx += 1;
            unsafe { set_bytes(encoder, idx, &stage_2_e4_stride); } idx += 1;
            unsafe { set_bytes(encoder, idx, &memory_args_start); } idx += 1;
            unsafe { set_bytes(encoder, idx, &log_n); }
        },
    )
}
