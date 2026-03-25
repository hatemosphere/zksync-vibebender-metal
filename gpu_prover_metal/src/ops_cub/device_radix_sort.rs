use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use crate::utils::div_ceil;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

const RADIX_SORT_THREADS: u32 = 256;
const RADIX_BITS: u32 = 4;
const RADIX_SIZE: u32 = 1 << RADIX_BITS;

/// Returns the number of u32 elements needed for histogram temp storage.
pub fn get_sort_temp_storage_elems(num_items: u32) -> u32 {
    let num_blocks = div_ceil(num_items as usize, RADIX_SORT_THREADS as usize) as u32;
    RADIX_SIZE * num_blocks
}

/// Sort u32 keys using multi-pass 4-bit radix sort with ping-pong buffering.
///
/// `d_keys_in` and `d_keys_out` are used as a ping-pong pair.
/// After completion, the sorted result is in `d_keys_out`.
/// `d_histograms` must have `get_sort_temp_storage_elems(num_items)` elements.
#[allow(clippy::too_many_arguments)]
pub fn sort_keys(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    _descending: bool,
    d_keys_in: &MetalBuffer<u32>,
    d_keys_out: &MetalBuffer<u32>,
    d_histograms: &MetalBuffer<u32>,
    num_items: u32,
    begin_bit: u32,
    end_bit: u32,
) -> MetalResult<()> {
    let num_blocks = div_ceil(num_items as usize, RADIX_SORT_THREADS as usize) as u32;
    let total_bins = RADIX_SIZE * num_blocks;
    let num_passes = div_ceil((end_bit - begin_bit) as usize, RADIX_BITS as usize);

    // Determine ping-pong so the final result lands in d_keys_out.
    // Pass i reads from src and writes to dst, alternating each pass.
    // Pass 0: A→B, Pass 1: B→A, ...
    // Last pass = num_passes-1. If num_passes is odd, last pass is even (writes to B).
    // If num_passes is even, last pass is odd (writes to A).
    // We need the final write to go to d_keys_out.
    let (buf_a, buf_b) = if num_passes % 2 == 1 {
        // Odd passes: last pass (even) writes to B. Set B=d_keys_out.
        (d_keys_in.raw(), d_keys_out.raw())
    } else {
        // Even passes: last pass (odd) writes to A. Set A=d_keys_out.
        // Copy input data to d_keys_out so pass 0 reads from it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                d_keys_in.as_ptr(),
                d_keys_out.as_ptr() as *mut u32,
                num_items as usize,
            );
        }
        (d_keys_out.raw(), d_keys_in.raw())
    };

    let mut bit = begin_bit;
    let mut pass = 0u32;
    while bit < end_bit {
        let bit_offset = bit;
        let (src, dst) = if pass % 2 == 0 {
            (buf_a, buf_b)
        } else {
            (buf_b, buf_a)
        };

        // Pass 1: histogram
        let config1 = MetalLaunchConfig::basic_1d(num_blocks, RADIX_SORT_THREADS);
        dispatch_kernel(device, cmd_buf, "ab_radix_sort_histogram_kernel", &config1, |encoder| {
            set_buffer(encoder, 0, src, 0);
            set_buffer(encoder, 1, d_histograms.raw(), 0);
            unsafe {
                set_bytes(encoder, 2, &num_items);
                set_bytes(encoder, 3, &bit_offset);
                set_bytes(encoder, 4, &num_blocks);
            }
        })?;

        // Pass 2: prefix sum of histograms
        let scan_threads = total_bins.min(RADIX_SORT_THREADS);
        let config2 = MetalLaunchConfig::basic_1d(1, scan_threads);
        dispatch_kernel(device, cmd_buf, "ab_radix_sort_scan_histograms_kernel", &config2, |encoder| {
            set_buffer(encoder, 0, d_histograms.raw(), 0);
            unsafe { set_bytes(encoder, 1, &total_bins); }
        })?;

        // Pass 3: scatter
        let config3 = MetalLaunchConfig::basic_1d(num_blocks, RADIX_SORT_THREADS);
        dispatch_kernel(device, cmd_buf, "ab_radix_sort_scatter_keys_kernel", &config3, |encoder| {
            set_buffer(encoder, 0, src, 0);
            set_buffer(encoder, 1, dst, 0);
            set_buffer(encoder, 2, d_histograms.raw(), 0);
            unsafe {
                set_bytes(encoder, 3, &num_items);
                set_bytes(encoder, 4, &bit_offset);
                set_bytes(encoder, 5, &num_blocks);
            }
        })?;

        bit += RADIX_BITS;
        pass += 1;
    }

    Ok(())
}

/// Sort u32 key-value pairs using multi-pass 4-bit radix sort with ping-pong.
#[allow(clippy::too_many_arguments)]
pub fn sort_pairs(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    _descending: bool,
    d_keys_in: &MetalBuffer<u32>,
    d_keys_out: &MetalBuffer<u32>,
    d_values_in: &MetalBuffer<u32>,
    d_values_out: &MetalBuffer<u32>,
    d_histograms: &MetalBuffer<u32>,
    num_items: u32,
    begin_bit: u32,
    end_bit: u32,
) -> MetalResult<()> {
    let num_blocks = div_ceil(num_items as usize, RADIX_SORT_THREADS as usize) as u32;
    let total_bins = RADIX_SIZE * num_blocks;
    let num_passes = div_ceil((end_bit - begin_bit) as usize, RADIX_BITS as usize);

    let (keys_a, keys_b, vals_a, vals_b) = if num_passes % 2 == 1 {
        (d_keys_in.raw(), d_keys_out.raw(), d_values_in.raw(), d_values_out.raw())
    } else {
        unsafe {
            std::ptr::copy_nonoverlapping(d_keys_in.as_ptr(), d_keys_out.as_ptr() as *mut u32, num_items as usize);
            std::ptr::copy_nonoverlapping(d_values_in.as_ptr(), d_values_out.as_ptr() as *mut u32, num_items as usize);
        }
        (d_keys_out.raw(), d_keys_in.raw(), d_values_out.raw(), d_values_in.raw())
    };

    let mut bit = begin_bit;
    let mut pass = 0u32;
    while bit < end_bit {
        let bit_offset = bit;
        let (k_src, k_dst, v_src, v_dst) = if pass % 2 == 0 {
            (keys_a, keys_b, vals_a, vals_b)
        } else {
            (keys_b, keys_a, vals_b, vals_a)
        };

        let config1 = MetalLaunchConfig::basic_1d(num_blocks, RADIX_SORT_THREADS);
        dispatch_kernel(device, cmd_buf, "ab_radix_sort_histogram_kernel", &config1, |encoder| {
            set_buffer(encoder, 0, k_src, 0);
            set_buffer(encoder, 1, d_histograms.raw(), 0);
            unsafe {
                set_bytes(encoder, 2, &num_items);
                set_bytes(encoder, 3, &bit_offset);
                set_bytes(encoder, 4, &num_blocks);
            }
        })?;

        let scan_threads = total_bins.min(RADIX_SORT_THREADS);
        let config2 = MetalLaunchConfig::basic_1d(1, scan_threads);
        dispatch_kernel(device, cmd_buf, "ab_radix_sort_scan_histograms_kernel", &config2, |encoder| {
            set_buffer(encoder, 0, d_histograms.raw(), 0);
            unsafe { set_bytes(encoder, 1, &total_bins); }
        })?;

        let config3 = MetalLaunchConfig::basic_1d(num_blocks, RADIX_SORT_THREADS);
        dispatch_kernel(device, cmd_buf, "ab_radix_sort_scatter_pairs_kernel", &config3, |encoder| {
            set_buffer(encoder, 0, k_src, 0);
            set_buffer(encoder, 1, k_dst, 0);
            set_buffer(encoder, 2, v_src, 0);
            set_buffer(encoder, 3, v_dst, 0);
            set_buffer(encoder, 4, d_histograms.raw(), 0);
            unsafe {
                set_bytes(encoder, 5, &num_items);
                set_bytes(encoder, 6, &bit_offset);
                set_bytes(encoder, 7, &num_blocks);
            }
        })?;

        bit += RADIX_BITS;
        pass += 1;
    }

    Ok(())
}
