use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use crate::ops_cub::device_scan;
use crate::utils::div_ceil;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

const RLE_THREADS_PER_GROUP: u32 = 256;

/// Returns the number of u32 elements needed for RLE temp storage.
/// Need space for: flags array (num_items) + positions array (num_items) + scan temp.
pub fn get_encode_temp_storage_elems(num_items: u32) -> u32 {
    let scan_temp = device_scan::get_scan_temp_storage_elems(num_items);
    // flags + positions + scan_temp
    num_items + num_items + scan_temp
}

/// Run-length encode u32 values.
///
/// `d_temp` must have `get_encode_temp_storage_elems(num_items)` u32 elements.
/// After execution:
/// - `d_unique_out` contains the unique run values
/// - `d_counts_out` contains the run lengths
/// - `d_num_runs_out` contains the total number of runs (single element)
#[allow(clippy::too_many_arguments)]
pub fn encode_u32(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    d_in: &MetalBuffer<u32>,
    d_unique_out: &MetalBuffer<u32>,
    d_counts_out: &MetalBuffer<u32>,
    d_num_runs_out: &MetalBuffer<u32>,
    d_temp: &MetalBuffer<u32>,
    num_items: u32,
) -> MetalResult<()> {
    // Partition temp storage:
    // [0..num_items) = flags
    // [num_items..2*num_items) = positions (exclusive prefix sum of flags)
    // [2*num_items..) = scan temp
    let flags_offset = 0usize;
    let positions_offset = num_items as usize * std::mem::size_of::<u32>();
    let scan_temp_offset = 2 * num_items as usize * std::mem::size_of::<u32>();

    let threadgroups = div_ceil(num_items as usize, RLE_THREADS_PER_GROUP as usize) as u32;

    // Step 1: Mark boundaries
    let config1 = MetalLaunchConfig::basic_1d(threadgroups, RLE_THREADS_PER_GROUP);
    dispatch_kernel(
        device,
        cmd_buf,
        "ab_rle_mark_boundaries_kernel",
        &config1,
        |encoder| {
            set_buffer(encoder, 0, d_in.raw(), 0);
            set_buffer(encoder, 1, d_temp.raw(), flags_offset);
            unsafe {
                set_bytes(encoder, 2, &num_items);
            }
        },
    )?;

    // Step 2: Exclusive prefix sum of flags -> positions
    // We use the scan kernels with the temp buffer partitioned appropriately.
    let _scan_temp_elems = device_scan::get_scan_temp_storage_elems(num_items);
    let num_blocks = div_ceil(num_items as usize, 256) as u32;

    // Pass 1 of scan
    let config_s1 = MetalLaunchConfig::basic_1d(num_blocks, 256);
    let inclusive_u32: u32 = 0; // exclusive scan
    dispatch_kernel(
        device,
        cmd_buf,
        "ab_scan_pass1_add_u32_kernel",
        &config_s1,
        |encoder| {
            set_buffer(encoder, 0, d_temp.raw(), flags_offset);
            set_buffer(encoder, 1, d_temp.raw(), positions_offset);
            set_buffer(encoder, 2, d_temp.raw(), scan_temp_offset);
            unsafe {
                set_bytes(encoder, 3, &num_items);
                set_bytes(encoder, 4, &inclusive_u32);
            }
        },
    )?;

    if num_blocks > 1 {
        // Pass 2 of scan
        let config_s2 = MetalLaunchConfig::basic_1d(1, 256);
        dispatch_kernel(
            device,
            cmd_buf,
            "ab_scan_pass2_add_u32_kernel",
            &config_s2,
            |encoder| {
                set_buffer(encoder, 0, d_temp.raw(), scan_temp_offset);
                unsafe {
                    set_bytes(encoder, 1, &num_blocks);
                }
            },
        )?;

        // Pass 3 of scan
        let config_s3 = MetalLaunchConfig::basic_1d(num_blocks, 256);
        dispatch_kernel(
            device,
            cmd_buf,
            "ab_scan_pass3_add_u32_kernel",
            &config_s3,
            |encoder| {
                set_buffer(encoder, 1, d_temp.raw(), scan_temp_offset);
                set_buffer(encoder, 0, d_temp.raw(), positions_offset);
                unsafe {
                    set_bytes(encoder, 2, &num_items);
                }
            },
        )?;
    }

    // Step 3: Scatter unique values and run lengths
    let config3 = MetalLaunchConfig::basic_1d(threadgroups, RLE_THREADS_PER_GROUP);
    dispatch_kernel(
        device,
        cmd_buf,
        "ab_rle_scatter_kernel",
        &config3,
        |encoder| {
            set_buffer(encoder, 0, d_in.raw(), 0);
            set_buffer(encoder, 1, d_temp.raw(), positions_offset);
            set_buffer(encoder, 2, d_unique_out.raw(), 0);
            set_buffer(encoder, 3, d_counts_out.raw(), 0);
            set_buffer(encoder, 4, d_num_runs_out.raw(), 0);
            unsafe {
                set_bytes(encoder, 5, &num_items);
            }
        },
    )?;

    Ok(())
}
