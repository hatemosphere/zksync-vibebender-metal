#![allow(non_snake_case)]

pub mod utils;

#[cfg(test)]
pub mod tests;

use crate::field::BaseField;
use crate::metal_runtime::buffer::MetalBuffer;
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

type BF = BaseField;

const COMPLEX_COLS_PER_SYNC_BLOCK: u32 = 1;
const COMPLEX_COLS_PER_WARP_BLOCK: u32 = 4;

/// NTT twiddle factor data, held on Metal buffers.
/// Replaces the CUDA __constant__ memory symbols.
pub struct NttTwiddleData {
    // powers_data_2_layer for B2N (forward NTT twiddles)
    pub b2n_fine: MetalBuffer<[u32; 2]>,
    pub b2n_fine_mask: u32,
    pub b2n_fine_log_count: u32,
    pub b2n_coarse: MetalBuffer<[u32; 2]>,
    pub b2n_coarse_mask: u32,
    pub b2n_coarse_log_count: u32,
    // powers_data_2_layer for N2B (inverse NTT twiddles)
    pub n2b_fine: MetalBuffer<[u32; 2]>,
    pub n2b_fine_mask: u32,
    pub n2b_fine_log_count: u32,
    pub n2b_coarse: MetalBuffer<[u32; 2]>,
    pub n2b_coarse_mask: u32,
    pub n2b_coarse_log_count: u32,
    // powers_data_3_layer for LDE scaling (powers of w)
    pub pow_fine: MetalBuffer<[u32; 2]>,
    pub pow_fine_mask: u32,
    pub pow_fine_log_count: u32,
    pub pow_coarser: MetalBuffer<[u32; 2]>,
    pub pow_coarser_mask: u32,
    pub pow_coarser_log_count: u32,
    pub pow_coarsest: MetalBuffer<[u32; 2]>,
    pub pow_coarsest_mask: u32,
    // Inverse sizes: inv_sizes[log_n] = 1/(2^log_n) as ext2_field
    pub inv_sizes: MetalBuffer<[u32; 2]>,
}

impl NttTwiddleData {
    /// Construct NttTwiddleData from DeviceContext by reinterpreting E2 buffers
    /// as `[u32; 2]` buffers. E2 (Mersenne31Complex) has the same memory layout.
    pub fn from_device_context(
        dc: &crate::device_context::DeviceContext,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> MetalResult<Self> {
        fn reinterpret_e2_buffer(
            src: &MetalBuffer<field::Mersenne31Complex>,
            device: &ProtocolObject<dyn MTLDevice>,
        ) -> MetalResult<MetalBuffer<[u32; 2]>> {
            let len = src.len();
            let src_slice = unsafe {
                std::slice::from_raw_parts(src.as_ptr() as *const [u32; 2], len)
            };
            MetalBuffer::from_slice(device, src_slice)
        }

        fn reinterpret_bf_to_u32x2_buffer(
            src: &MetalBuffer<field::Mersenne31Field>,
            device: &ProtocolObject<dyn MTLDevice>,
        ) -> MetalResult<MetalBuffer<[u32; 2]>> {
            // inv_sizes is stored as BF values; pack pairs into [u32; 2]
            // Each BF is one u32, but the kernel expects ext2 format [real, 0]
            let len = src.len();
            let src_slice = unsafe {
                std::slice::from_raw_parts(src.as_ptr() as *const u32, len)
            };
            let mut pairs = Vec::with_capacity(len);
            for &v in src_slice {
                pairs.push([v, 0u32]);
            }
            MetalBuffer::from_slice(device, &pairs)
        }

        let b2n_fine_log_count = dc.ntt_fine_log_count;
        let b2n_coarse_log_count = dc.ntt_coarse_log_count;
        let n2b_fine_log_count = dc.ntt_fine_log_count;
        let n2b_coarse_log_count = dc.ntt_coarse_log_count;
        let pow_fine_log_count = dc.fine_log_count;
        let pow_coarser_log_count = dc.coarser_log_count;

        Ok(Self {
            b2n_fine: reinterpret_e2_buffer(&dc.powers_of_w_fine_bitrev_for_ntt, device)?,
            b2n_fine_mask: (1u32 << b2n_fine_log_count) - 1,
            b2n_fine_log_count,
            b2n_coarse: reinterpret_e2_buffer(&dc.powers_of_w_coarse_bitrev_for_ntt, device)?,
            b2n_coarse_mask: (1u32 << b2n_coarse_log_count) - 1,
            b2n_coarse_log_count,
            n2b_fine: reinterpret_e2_buffer(&dc.powers_of_w_inv_fine_bitrev_for_ntt, device)?,
            n2b_fine_mask: (1u32 << n2b_fine_log_count) - 1,
            n2b_fine_log_count,
            n2b_coarse: reinterpret_e2_buffer(&dc.powers_of_w_inv_coarse_bitrev_for_ntt, device)?,
            n2b_coarse_mask: (1u32 << n2b_coarse_log_count) - 1,
            n2b_coarse_log_count,
            pow_fine: reinterpret_e2_buffer(&dc.powers_of_w_fine, device)?,
            pow_fine_mask: (1u32 << pow_fine_log_count) - 1,
            pow_fine_log_count,
            pow_coarser: reinterpret_e2_buffer(&dc.powers_of_w_coarser, device)?,
            pow_coarser_mask: (1u32 << pow_coarser_log_count) - 1,
            pow_coarser_log_count,
            pow_coarsest: reinterpret_e2_buffer(&dc.powers_of_w_coarsest, device)?,
            pow_coarsest_mask: (1u32 << dc.coarsest_log_count) - 1,
            inv_sizes: reinterpret_bf_to_u32x2_buffer(&dc.inv_sizes, device)?,
        })
    }
}

// ============================================================================
// B2N (bitrev Z -> natural evals) dispatch
// ============================================================================

/// Dispatch one-stage B2N kernel (fallback for log_n < 16).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_b2n_one_stage(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    start_stage: u32,
    log_n: u32,
    blocks_per_ntt: u32,
    log_extension_degree: u32,
    coset_idx: u32,
    num_z_cols: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    let threads: u32 = 128;
    let total_blocks = blocks_per_ntt * num_z_cols;
    let config = MetalLaunchConfig::basic_1d(total_blocks, threads);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_bitrev_Z_to_natural_coset_evals_one_stage",
        &config,
        |encoder| {
            set_buffer(encoder, 0, input.raw(), 0);
            set_buffer(encoder, 1, output.raw(), 0);
            set_buffer(encoder, 2, twiddles.b2n_fine.raw(), 0);
            set_buffer(encoder, 3, twiddles.b2n_coarse.raw(), 0);
            set_buffer(encoder, 4, twiddles.pow_fine.raw(), 0);
            set_buffer(encoder, 5, twiddles.pow_coarser.raw(), 0);
            set_buffer(encoder, 6, twiddles.pow_coarsest.raw(), 0);
            unsafe {
                set_bytes(encoder, 7, &stride);
                set_bytes(encoder, 8, &start_stage);
                set_bytes(encoder, 9, &log_n);
                set_bytes(encoder, 10, &blocks_per_ntt);
                set_bytes(encoder, 11, &log_extension_degree);
                set_bytes(encoder, 12, &coset_idx);
                set_bytes(encoder, 13, &twiddles.b2n_fine_mask);
                set_bytes(encoder, 14, &twiddles.b2n_fine_log_count);
                set_bytes(encoder, 15, &twiddles.b2n_coarse_mask);
                set_bytes(encoder, 16, &twiddles.b2n_coarse_log_count);
                set_bytes(encoder, 17, &twiddles.pow_fine_mask);
                set_bytes(encoder, 18, &twiddles.pow_fine_log_count);
                set_bytes(encoder, 19, &twiddles.pow_coarser_mask);
                set_bytes(encoder, 20, &twiddles.pow_coarser_log_count);
                set_bytes(encoder, 21, &twiddles.pow_coarsest_mask);
            }
        },
    )
}

/// Dispatch a multi-stage B2N kernel.
#[allow(clippy::too_many_arguments)]
fn dispatch_b2n_multi_stage(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    kernel_name: &str,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    start_stage: u32,
    stages_this_launch: u32,
    log_n: u32,
    num_z_cols: u32,
    log_extension_degree: u32,
    coset_idx: u32,
    grid_offset: u32,
    grid_dim_x: u32,
    grid_dim_y: u32,
    block_dim_x: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    let config = MetalLaunchConfig::basic_2d((grid_dim_x, grid_dim_y), (block_dim_x, 1));

    dispatch_kernel(device, cmd_buf, kernel_name, &config, |encoder| {
        set_buffer(encoder, 0, input.raw(), 0);
        set_buffer(encoder, 1, output.raw(), 0);
        set_buffer(encoder, 2, twiddles.b2n_fine.raw(), 0);
        set_buffer(encoder, 3, twiddles.b2n_coarse.raw(), 0);
        set_buffer(encoder, 4, twiddles.pow_fine.raw(), 0);
        set_buffer(encoder, 5, twiddles.pow_coarser.raw(), 0);
        set_buffer(encoder, 6, twiddles.pow_coarsest.raw(), 0);
        unsafe {
            set_bytes(encoder, 7, &stride);
            set_bytes(encoder, 8, &start_stage);
            set_bytes(encoder, 9, &stages_this_launch);
            set_bytes(encoder, 10, &log_n);
            set_bytes(encoder, 11, &num_z_cols);
            set_bytes(encoder, 12, &log_extension_degree);
            set_bytes(encoder, 13, &coset_idx);
            set_bytes(encoder, 14, &grid_offset);
            set_bytes(encoder, 15, &twiddles.b2n_fine_mask);
            set_bytes(encoder, 16, &twiddles.b2n_fine_log_count);
            set_bytes(encoder, 17, &twiddles.b2n_coarse_mask);
            set_bytes(encoder, 18, &twiddles.b2n_coarse_log_count);
            set_bytes(encoder, 19, &twiddles.pow_fine_mask);
            set_bytes(encoder, 20, &twiddles.pow_fine_log_count);
            set_bytes(encoder, 21, &twiddles.pow_coarser_mask);
            set_bytes(encoder, 22, &twiddles.pow_coarser_log_count);
            set_bytes(encoder, 23, &twiddles.pow_coarsest_mask);
        }
    })
}

/// Run B2N NTT for small sizes (log_n < 16) using repeated one-stage kernels.
#[allow(clippy::too_many_arguments)]
pub fn bitrev_Z_to_natural_evals_small(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    log_n: u32,
    num_bf_cols: u32,
    log_extension_degree: u32,
    coset_idx: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    assert!(log_n >= 1);
    assert_eq!(num_bf_cols % 2, 0);
    let n = 1u32 << log_n;
    let num_z_cols = num_bf_cols / 2;
    let threads = 128u32;
    let blocks_per_ntt = (n / 2 + threads - 1) / threads;

    dispatch_b2n_one_stage(
        device,
        cmd_buf,
        input,
        output,
        stride,
        0,
        log_n,
        blocks_per_ntt,
        log_extension_degree,
        coset_idx,
        num_z_cols,
        twiddles,
    )?;

    for stage in 1..log_n {
        dispatch_b2n_one_stage(
            device,
            cmd_buf,
            output,
            output,
            stride,
            stage,
            log_n,
            blocks_per_ntt,
            log_extension_degree,
            coset_idx,
            num_z_cols,
            twiddles,
        )?;
    }

    Ok(())
}

/// Run B2N NTT using multi-stage kernels (log_n >= 16).
#[allow(clippy::too_many_arguments)]
pub fn bitrev_Z_to_natural_evals(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    log_n: u32,
    num_bf_cols: u32,
    log_extension_degree: u32,
    coset_idx: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    assert!(log_n >= 1);
    assert_eq!(num_bf_cols % 2, 0);
    let n = 1u32 << log_n;
    let num_z_cols = num_bf_cols / 2;

    if log_n < 16 {
        return bitrev_Z_to_natural_evals_small(
            device,
            cmd_buf,
            input,
            output,
            stride,
            log_n,
            num_bf_cols,
            log_extension_degree,
            coset_idx,
            twiddles,
        );
    }

    use utils::B2N_LAUNCH::*;
    let plan = &utils::STAGE_PLANS_B2N[(log_n - 16) as usize];
    let mut stage = 0u32;

    for kernel in plan.iter() {
        let start_stage = stage;
        if let Some((kern, stages_this_launch, vals_per_block)) = kernel {
            let stages_this_launch = *stages_this_launch as u32;
            let vals_per_block = *vals_per_block as u32;
            stage += stages_this_launch;

            let (kernel_name, grid_dim_x, block_dim_x, num_chunks): (&str, u32, u32, u32) =
                match kern {
                    INITIAL_7_WARP => (
                        "ab_bitrev_Z_to_natural_coset_evals_initial_7_stages_warp",
                        n / vals_per_block,
                        128,
                        num_z_cols.div_ceil(COMPLEX_COLS_PER_WARP_BLOCK),
                    ),
                    INITIAL_8_WARP => (
                        "ab_bitrev_Z_to_natural_coset_evals_initial_8_stages_warp",
                        n / vals_per_block,
                        128,
                        num_z_cols.div_ceil(COMPLEX_COLS_PER_WARP_BLOCK),
                    ),
                    INITIAL_9_TO_12_BLOCK => (
                        "ab_bitrev_Z_to_natural_coset_evals_initial_9_to_12_stages_block",
                        n / vals_per_block,
                        512,
                        num_z_cols.div_ceil(COMPLEX_COLS_PER_SYNC_BLOCK),
                    ),
                    NONINITIAL_7_OR_8_BLOCK => (
                        "ab_bitrev_Z_to_natural_coset_evals_noninitial_7_or_8_stages_block",
                        n / vals_per_block,
                        512,
                        num_z_cols.div_ceil(COMPLEX_COLS_PER_SYNC_BLOCK),
                    ),
                };

            let input_buf = if start_stage == 0 { input } else { output };

            dispatch_b2n_multi_stage(
                device,
                cmd_buf,
                kernel_name,
                input_buf,
                output,
                stride,
                start_stage,
                stages_this_launch,
                log_n,
                num_z_cols,
                log_extension_degree,
                coset_idx,
                0,
                grid_dim_x,
                num_chunks,
                block_dim_x,
                twiddles,
            )?;
        }
    }
    assert_eq!(stage, log_n);
    Ok(())
}

/// Convenience: B2N for trace coset evals (log_extension_degree=1, coset_idx=1).
#[allow(clippy::too_many_arguments)]
pub fn bitrev_Z_to_natural_trace_coset_evals(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    log_n: u32,
    num_bf_cols: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    bitrev_Z_to_natural_evals(
        device,
        cmd_buf,
        input,
        output,
        stride,
        log_n,
        num_bf_cols,
        1,
        1,
        twiddles,
    )
}

/// Convenience: B2N for composition main domain (log_extension_degree=1, coset_idx=0).
#[allow(clippy::too_many_arguments)]
pub fn bitrev_Z_to_natural_composition_main_evals(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    log_n: u32,
    num_bf_cols: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    bitrev_Z_to_natural_evals(
        device,
        cmd_buf,
        input,
        output,
        stride,
        log_n,
        num_bf_cols,
        1,
        0,
        twiddles,
    )
}

// ============================================================================
// N2B (natural evals -> bitrev Z) dispatch
// ============================================================================

/// Dispatch one-stage N2B kernel (fallback for log_n < 16).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_n2b_one_stage(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    start_stage: u32,
    log_n: u32,
    blocks_per_ntt: u32,
    evals_are_coset: u32,
    evals_are_compressed: u32,
    num_z_cols: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    let threads: u32 = 128;
    let total_blocks = blocks_per_ntt * num_z_cols;
    let config = MetalLaunchConfig::basic_1d(total_blocks, threads);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_evals_to_Z_one_stage",
        &config,
        |encoder| {
            set_buffer(encoder, 0, input.raw(), 0);
            set_buffer(encoder, 1, output.raw(), 0);
            set_buffer(encoder, 2, twiddles.n2b_fine.raw(), 0);
            set_buffer(encoder, 3, twiddles.n2b_coarse.raw(), 0);
            set_buffer(encoder, 4, twiddles.pow_fine.raw(), 0);
            set_buffer(encoder, 5, twiddles.pow_coarser.raw(), 0);
            set_buffer(encoder, 6, twiddles.pow_coarsest.raw(), 0);
            unsafe {
                set_bytes(encoder, 7, &stride);
                set_bytes(encoder, 8, &start_stage);
                set_bytes(encoder, 9, &log_n);
                set_bytes(encoder, 10, &blocks_per_ntt);
                set_bytes(encoder, 11, &evals_are_coset);
                set_bytes(encoder, 12, &evals_are_compressed);
                set_bytes(encoder, 13, &twiddles.n2b_fine_mask);
                set_bytes(encoder, 14, &twiddles.n2b_fine_log_count);
                set_bytes(encoder, 15, &twiddles.n2b_coarse_mask);
                set_bytes(encoder, 16, &twiddles.n2b_coarse_log_count);
                set_bytes(encoder, 17, &twiddles.pow_fine_mask);
                set_bytes(encoder, 18, &twiddles.pow_fine_log_count);
                set_bytes(encoder, 19, &twiddles.pow_coarser_mask);
                set_bytes(encoder, 20, &twiddles.pow_coarser_log_count);
                set_bytes(encoder, 21, &twiddles.pow_coarsest_mask);
            }
            set_buffer(encoder, 22, twiddles.inv_sizes.raw(), 0);
        },
    )
}

/// Dispatch a multi-stage N2B kernel.
/// `evals_are_coset`: 0 = main domain, 1 = coset, 2 = compressed coset
#[allow(clippy::too_many_arguments)]
fn dispatch_n2b_multi_stage(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    kernel_name: &str,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    start_stage: u32,
    stages_this_launch: u32,
    log_n: u32,
    num_z_cols: u32,
    grid_offset: u32,
    evals_are_coset: u32,
    grid_dim_x: u32,
    grid_dim_y: u32,
    block_dim_x: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    let config = MetalLaunchConfig::basic_2d((grid_dim_x, grid_dim_y), (block_dim_x, 1));

    dispatch_kernel(device, cmd_buf, kernel_name, &config, |encoder| {
        set_buffer(encoder, 0, input.raw(), 0);
        set_buffer(encoder, 1, output.raw(), 0);
        set_buffer(encoder, 2, twiddles.n2b_fine.raw(), 0);
        set_buffer(encoder, 3, twiddles.n2b_coarse.raw(), 0);
        set_buffer(encoder, 4, twiddles.pow_fine.raw(), 0);
        set_buffer(encoder, 5, twiddles.pow_coarser.raw(), 0);
        set_buffer(encoder, 6, twiddles.pow_coarsest.raw(), 0);
        unsafe {
            set_bytes(encoder, 7, &stride);
            set_bytes(encoder, 8, &start_stage);
            set_bytes(encoder, 9, &stages_this_launch);
            set_bytes(encoder, 10, &log_n);
            set_bytes(encoder, 11, &num_z_cols);
            set_bytes(encoder, 12, &grid_offset);
            set_bytes(encoder, 13, &evals_are_coset);
            set_bytes(encoder, 14, &twiddles.n2b_fine_mask);
            set_bytes(encoder, 15, &twiddles.n2b_fine_log_count);
            set_bytes(encoder, 16, &twiddles.n2b_coarse_mask);
            set_bytes(encoder, 17, &twiddles.n2b_coarse_log_count);
            set_bytes(encoder, 18, &twiddles.pow_fine_mask);
            set_bytes(encoder, 19, &twiddles.pow_fine_log_count);
            set_bytes(encoder, 20, &twiddles.pow_coarser_mask);
            set_bytes(encoder, 21, &twiddles.pow_coarser_log_count);
            set_bytes(encoder, 22, &twiddles.pow_coarsest_mask);
        }
        set_buffer(encoder, 23, twiddles.inv_sizes.raw(), 0);
    })
}

/// Run N2B NTT for small sizes (log_n < 16) using repeated one-stage kernels.
#[allow(clippy::too_many_arguments)]
pub fn natural_evals_to_bitrev_Z_small(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    log_n: u32,
    num_bf_cols: u32,
    evals_are_coset: bool,
    evals_are_compressed: bool,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    assert!(log_n >= 1);
    assert_eq!(num_bf_cols % 2, 0);
    let n = 1u32 << log_n;
    let num_z_cols = num_bf_cols / 2;
    let threads = 128u32;
    let blocks_per_ntt = (n / 2 + threads - 1) / threads;
    let coset_u32 = evals_are_coset as u32;
    let compressed_u32 = evals_are_compressed as u32;

    dispatch_n2b_one_stage(
        device,
        cmd_buf,
        input,
        output,
        stride,
        0,
        log_n,
        blocks_per_ntt,
        coset_u32,
        compressed_u32,
        num_z_cols,
        twiddles,
    )?;

    for stage in 1..log_n {
        dispatch_n2b_one_stage(
            device,
            cmd_buf,
            output,
            output,
            stride,
            stage,
            log_n,
            blocks_per_ntt,
            coset_u32,
            compressed_u32,
            num_z_cols,
            twiddles,
        )?;
    }

    Ok(())
}

/// Run N2B NTT using multi-stage kernels.
#[allow(clippy::too_many_arguments)]
pub fn natural_evals_to_bitrev_Z(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    log_n: u32,
    num_bf_cols: u32,
    evals_are_coset: bool,
    evals_are_compressed: bool,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    assert!(log_n >= 1);
    assert_eq!(num_bf_cols % 2, 0);
    if !evals_are_coset {
        assert!(!evals_are_compressed);
    }
    let n = 1u32 << log_n;
    let num_z_cols = num_bf_cols / 2;

    if log_n < 16 {
        return natural_evals_to_bitrev_Z_small(
            device,
            cmd_buf,
            input,
            output,
            stride,
            log_n,
            num_bf_cols,
            evals_are_coset,
            evals_are_compressed,
            twiddles,
        );
    }

    // evals_are_coset encoding for the kernel: 0 = main, 1 = coset, 2 = compressed
    let evals_are_coset_flag: u32 = if evals_are_coset {
        if evals_are_compressed {
            2
        } else {
            1
        }
    } else {
        0
    };

    use utils::N2B_LAUNCH::*;
    let plan = &utils::STAGE_PLANS_N2B[(log_n - 16) as usize];
    let mut stage = 0u32;

    for kernel in plan.iter() {
        let start_stage = stage;
        if let Some((kern, stages_this_launch, vals_per_block)) = kernel {
            let stages_this_launch = *stages_this_launch as u32;
            let vals_per_block = *vals_per_block as u32;
            stage += stages_this_launch;

            let (kernel_name, grid_dim_x, block_dim_x, num_chunks): (&str, u32, u32, u32) =
                match kern {
                    FINAL_7_WARP => (
                        "ab_evals_to_Z_final_7_stages_warp",
                        n / vals_per_block,
                        128,
                        num_z_cols.div_ceil(COMPLEX_COLS_PER_WARP_BLOCK),
                    ),
                    FINAL_8_WARP => (
                        "ab_evals_to_Z_final_8_stages_warp",
                        n / vals_per_block,
                        128,
                        num_z_cols.div_ceil(COMPLEX_COLS_PER_WARP_BLOCK),
                    ),
                    FINAL_9_TO_12_BLOCK => (
                        "ab_evals_to_Z_final_9_to_12_stages_block",
                        n / vals_per_block,
                        512,
                        num_z_cols.div_ceil(COMPLEX_COLS_PER_SYNC_BLOCK),
                    ),
                    NONFINAL_7_OR_8_BLOCK => (
                        "ab_evals_to_Z_nonfinal_7_or_8_stages_block",
                        n / vals_per_block,
                        512,
                        num_z_cols.div_ceil(COMPLEX_COLS_PER_SYNC_BLOCK),
                    ),
                };

            let input_buf = if start_stage == 0 { input } else { output };

            // Final kernels get the coset flag; nonfinal always get 0
            let coset_flag = match kern {
                NONFINAL_7_OR_8_BLOCK => 0u32,
                _ => evals_are_coset_flag,
            };

            dispatch_n2b_multi_stage(
                device,
                cmd_buf,
                kernel_name,
                input_buf,
                output,
                stride,
                start_stage,
                stages_this_launch,
                log_n,
                num_z_cols,
                0,
                coset_flag,
                grid_dim_x,
                num_chunks,
                block_dim_x,
                twiddles,
            )?;
        }
    }
    assert_eq!(stage, log_n);
    Ok(())
}

/// Convenience: N2B for main domain evals (no coset unscaling).
#[allow(clippy::too_many_arguments)]
pub fn natural_trace_main_evals_to_bitrev_Z(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    log_n: u32,
    num_bf_cols: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    natural_evals_to_bitrev_Z(
        device,
        cmd_buf,
        input,
        output,
        stride,
        log_n,
        num_bf_cols,
        false,
        false,
        twiddles,
    )
}

/// Convenience: N2B for coset evals (with coset unscaling, not compressed).
#[allow(clippy::too_many_arguments)]
pub fn natural_composition_coset_evals_to_bitrev_Z(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    log_n: u32,
    num_bf_cols: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    natural_evals_to_bitrev_Z(
        device,
        cmd_buf,
        input,
        output,
        stride,
        log_n,
        num_bf_cols,
        true,
        false,
        twiddles,
    )
}

/// Convenience: N2B for compressed coset evals.
#[allow(clippy::too_many_arguments)]
pub fn natural_compressed_coset_evals_to_bitrev_Z(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input: &MetalBuffer<BF>,
    output: &MetalBuffer<BF>,
    stride: u32,
    log_n: u32,
    num_bf_cols: u32,
    twiddles: &NttTwiddleData,
) -> MetalResult<()> {
    natural_evals_to_bitrev_Z(
        device,
        cmd_buf,
        input,
        output,
        stride,
        log_n,
        num_bf_cols,
        true,
        true,
        twiddles,
    )
}
