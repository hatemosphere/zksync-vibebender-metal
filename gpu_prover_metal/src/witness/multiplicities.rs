use super::BF;
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use crate::ops_cub::device_radix_sort;
use crate::ops_cub::device_run_length_encode;
use crate::ops_simple::set_by_val_u32;
use crate::prover::context::ProverContext;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

const WARP_SIZE: u32 = 32;

pub(crate) fn generate_generic_lookup_multiplicities(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    lookup_mapping: &MetalBuffer<u32>,
    lookup_mapping_stride: u32,
    lookup_mapping_cols: u32,
    multiplicities: &MetalBuffer<BF>,
    multiplicities_offset: usize,
    multiplicities_size: u32,
    multiplicities_stride: u32,
    context: &ProverContext,
) -> MetalResult<()> {
    let stride = lookup_mapping_stride as usize;
    assert!(stride.is_power_of_two());

    // Set the last row of lookup_mapping to 0xffffffff (sentinel)
    set_by_val_u32(
        device,
        cmd_buf,
        0xffffffff,
        lookup_mapping,
        lookup_mapping_stride,
        1, // one row (the last row)
        lookup_mapping_cols,
    )?;

    let lookup_mapping_size = lookup_mapping.len() as u32;

    // Allocate sorted buffer
    let sorted_lookup_mapping: MetalBuffer<u32> =
        context.alloc(lookup_mapping_size as usize)?;

    let lookup_mapping_bits_count = (multiplicities_size as usize)
        .next_power_of_two()
        .trailing_zeros() as u32;

    // Allocate radix sort temp storage
    let sort_temp_elems = device_radix_sort::get_sort_temp_storage_elems(lookup_mapping_size);
    let sort_temp: MetalBuffer<u32> = context.alloc(sort_temp_elems as usize)?;

    // Sort keys
    device_radix_sort::sort_keys(
        device,
        cmd_buf,
        false,
        lookup_mapping,
        &sorted_lookup_mapping,
        &sort_temp,
        lookup_mapping_size,
        0,
        lookup_mapping_bits_count,
    )?;

    // Allocate RLE output buffers
    let unique_lookup_mapping: MetalBuffer<u32> =
        context.alloc(multiplicities_size as usize)?;
    let counts: MetalBuffer<u32> = context.alloc(multiplicities_size as usize)?;
    let num_runs: MetalBuffer<u32> = context.alloc(1)?;

    // Allocate RLE temp storage
    let rle_temp_elems =
        device_run_length_encode::get_encode_temp_storage_elems(lookup_mapping_size);
    let rle_temp: MetalBuffer<u32> = context.alloc(rle_temp_elems as usize)?;

    // Run-length encode
    device_run_length_encode::encode_u32(
        device,
        cmd_buf,
        &sorted_lookup_mapping,
        &unique_lookup_mapping,
        &counts,
        &num_runs,
        &rle_temp,
        lookup_mapping_size,
    )?;

    // Dispatch multiplicities kernel
    let count = multiplicities_size;
    let block_dim = WARP_SIZE * 4;
    let grid_dim = (count + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_generate_multiplicities_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            set_buffer(encoder, idx, unique_lookup_mapping.raw(), 0);
            idx += 1;
            set_buffer(encoder, idx, counts.raw(), 0);
            idx += 1;
            set_buffer(encoder, idx, num_runs.raw(), 0);
            idx += 1;
            set_buffer(encoder, idx, multiplicities.raw(), multiplicities_offset);
            idx += 1;
            unsafe {
                set_bytes(encoder, idx, &multiplicities_stride);
            }
            idx += 1;
            unsafe {
                set_bytes(encoder, idx, &count);
            }
        },
    )
}
