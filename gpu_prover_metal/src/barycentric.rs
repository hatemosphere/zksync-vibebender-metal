use crate::field::{BaseField, Ext2Field, Ext4Field};
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use crate::ops_complex::PowersLayerDesc;
use crate::prover::context::ProverContext;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

type BF = BaseField;
type E2 = Ext2Field;
type E4 = Ext4Field;

const SIMD_GROUP_SIZE: usize = 32;
const INV_BATCH_E4: u32 = 4;
pub const BARY_THREADS_PER_GROUP: u32 = 256;

/// Precompute the common factor for barycentric evaluation.
pub fn precompute_common_factor(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    z: &MetalBuffer<E4>,
    common_factor: &MetalBuffer<E4>,
    coset: E2,
    decompression_factor: E2,
    count: u32,
) -> MetalResult<()> {
    let config = MetalLaunchConfig::basic_1d(1, 1);
    dispatch_kernel(device, cmd_buf, "ab_barycentric_precompute_common_factor_kernel", &config, |encoder| {
        set_buffer(encoder, 0, z.raw(), 0);
        set_buffer(encoder, 1, common_factor.raw(), 0);
        unsafe {
            set_bytes(encoder, 2, &coset);
            set_bytes(encoder, 3, &decompression_factor);
            set_bytes(encoder, 4, &count);
        }
    })
}

/// Precompute Lagrange coefficients for barycentric evaluation.
pub fn precompute_lagrange_coeffs(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    z: &MetalBuffer<E4>,
    common_factor: &MetalBuffer<E4>,
    w_inv_step: E2,
    coset: E2,
    lagrange_coeffs: &MetalBuffer<E4>,
    log_count: u32,
    powers_fine: &MetalBuffer<E2>,
    powers_fine_desc: &PowersLayerDesc,
    powers_coarser: &MetalBuffer<E2>,
    powers_coarser_desc: &PowersLayerDesc,
    powers_coarsest: &MetalBuffer<E2>,
    powers_coarsest_desc: &PowersLayerDesc,
) -> MetalResult<()> {
    let count = 1u32 << log_count;
    let block_dim = (SIMD_GROUP_SIZE * 4) as u32;
    let grid_dim = (count + INV_BATCH_E4 * block_dim - 1) / (INV_BATCH_E4 * block_dim);
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);
    dispatch_kernel(device, cmd_buf, "ab_barycentric_precompute_lagrange_coeffs_kernel", &config, |encoder| {
        set_buffer(encoder, 0, z.raw(), 0);
        set_buffer(encoder, 1, common_factor.raw(), 0);
        unsafe {
            set_bytes(encoder, 2, &w_inv_step);
            set_bytes(encoder, 3, &coset);
        }
        set_buffer(encoder, 4, lagrange_coeffs.raw(), 0);
        unsafe { set_bytes(encoder, 5, &log_count); }
        set_buffer(encoder, 6, powers_fine.raw(), 0);
        unsafe {
            set_bytes(encoder, 7, &powers_fine_desc.mask);
            set_bytes(encoder, 8, &powers_fine_desc.log_count);
        }
        set_buffer(encoder, 9, powers_coarser.raw(), 0);
        unsafe {
            set_bytes(encoder, 10, &powers_coarser_desc.mask);
            set_bytes(encoder, 11, &powers_coarser_desc.log_count);
        }
        set_buffer(encoder, 12, powers_coarsest.raw(), 0);
        unsafe {
            set_bytes(encoder, 13, &powers_coarsest_desc.mask);
            set_bytes(encoder, 14, &powers_coarsest_desc.log_count);
        }
    })
}

/// Evaluate a BF column at z into an existing command buffer (no sync).
pub fn eval_bf_column_at_z_batched(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    lagrange_coeffs: &MetalBuffer<E4>,
    bf_col: &MetalBuffer<BF>,
    col_offset: usize,
    count: u32,
    result: &mut MetalBuffer<E4>,
    result_offset: usize,
    partial_buf: &mut MetalBuffer<E4>,
) -> MetalResult<()> {
    let num_blocks = (count + BARY_THREADS_PER_GROUP - 1) / BARY_THREADS_PER_GROUP;
    let config = MetalLaunchConfig::basic_1d(num_blocks, BARY_THREADS_PER_GROUP);
    let col_stride = count;
    dispatch_kernel(device, cmd_buf, "ab_barycentric_eval_bf_col_partial_reduce", &config, |encoder| {
        set_buffer(encoder, 0, lagrange_coeffs.raw(), 0);
        set_buffer(encoder, 1, bf_col.raw(), col_offset * std::mem::size_of::<BF>());
        set_buffer(encoder, 2, partial_buf.raw(), 0);
        unsafe {
            set_bytes(encoder, 3, &count);
            set_bytes(encoder, 4, &col_stride);
        }
    })?;
    let config2 = MetalLaunchConfig::basic_1d(1, BARY_THREADS_PER_GROUP);
    dispatch_kernel(device, cmd_buf, "ab_barycentric_eval_final_reduce", &config2, |encoder| {
        set_buffer(encoder, 0, partial_buf.raw(), 0);
        set_buffer(encoder, 1, result.raw(), result_offset * std::mem::size_of::<E4>());
        unsafe { set_bytes(encoder, 2, &num_blocks); }
    })
}

/// Evaluate a BF column at z: result = sum_i(lagrange[i] * bf_col[i])
pub fn eval_bf_column_at_z(
    device: &ProtocolObject<dyn MTLDevice>,
    lagrange_coeffs: &MetalBuffer<E4>,
    bf_col: &MetalBuffer<BF>,
    col_offset: usize,
    count: u32,
    result: &mut MetalBuffer<E4>,
    result_offset: usize,
    partial_buf: &mut MetalBuffer<E4>,
    context: &ProverContext,
) -> MetalResult<()> {
    let num_blocks = (count + BARY_THREADS_PER_GROUP - 1) / BARY_THREADS_PER_GROUP;
    let config = MetalLaunchConfig::basic_1d(num_blocks, BARY_THREADS_PER_GROUP);
    let cmd_buf = context.new_command_buffer()?;
    let count_val = count;
    let col_stride = count;
    dispatch_kernel(device, &cmd_buf, "ab_barycentric_eval_bf_col_partial_reduce", &config, |encoder| {
        set_buffer(encoder, 0, lagrange_coeffs.raw(), 0);
        set_buffer(encoder, 1, bf_col.raw(), col_offset * std::mem::size_of::<BF>());
        set_buffer(encoder, 2, partial_buf.raw(), 0);
        unsafe {
            set_bytes(encoder, 3, &count_val);
            set_bytes(encoder, 4, &col_stride);
        }
    })?;
    let config2 = MetalLaunchConfig::basic_1d(1, BARY_THREADS_PER_GROUP);
    dispatch_kernel(device, &cmd_buf, "ab_barycentric_eval_final_reduce", &config2, |encoder| {
        set_buffer(encoder, 0, partial_buf.raw(), 0);
        set_buffer(encoder, 1, result.raw(), result_offset * std::mem::size_of::<E4>());
        unsafe { set_bytes(encoder, 2, &num_blocks); }
    })?;
    cmd_buf.commit_and_wait();
    Ok(())
}

/// Evaluate an E4 column at z into an existing command buffer (no sync).
/// Reuses a pre-allocated E4 temp buffer for the BF4→E4 transpose.
pub fn eval_e4_column_at_z_batched(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    lagrange_coeffs: &MetalBuffer<E4>,
    bf_data: &MetalBuffer<BF>,
    col_offset: usize,
    count: u32,
    result: &mut MetalBuffer<E4>,
    result_offset: usize,
    partial_buf: &mut MetalBuffer<E4>,
    e4_temp: &MetalBuffer<E4>,
) -> MetalResult<()> {
    crate::ops_simple::transpose_bf4_to_e4_strided(
        device, cmd_buf,
        bf_data.raw(), e4_temp.raw(),
        count, col_offset as u32, count,
    )?;
    let num_blocks = (count + BARY_THREADS_PER_GROUP - 1) / BARY_THREADS_PER_GROUP;
    let config = MetalLaunchConfig::basic_1d(num_blocks, BARY_THREADS_PER_GROUP);
    dispatch_kernel(device, cmd_buf, "ab_barycentric_eval_e4_col_partial_reduce", &config, |encoder| {
        set_buffer(encoder, 0, lagrange_coeffs.raw(), 0);
        set_buffer(encoder, 1, e4_temp.raw(), 0);
        set_buffer(encoder, 2, partial_buf.raw(), 0);
        unsafe { set_bytes(encoder, 3, &count); }
    })?;
    let config2 = MetalLaunchConfig::basic_1d(1, BARY_THREADS_PER_GROUP);
    dispatch_kernel(device, cmd_buf, "ab_barycentric_eval_final_reduce", &config2, |encoder| {
        set_buffer(encoder, 0, partial_buf.raw(), 0);
        set_buffer(encoder, 1, result.raw(), result_offset * std::mem::size_of::<E4>());
        unsafe { set_bytes(encoder, 2, &num_blocks); }
    })
}

/// Evaluate an E4 column at z: result = sum_i(lagrange[i] * e4_col[i])
/// The E4 column is stored as 4 BF columns in column-major layout.
/// GPU transpose replaces the CPU loop that was the main stage 4 bottleneck.
pub fn eval_e4_column_at_z(
    device: &ProtocolObject<dyn MTLDevice>,
    lagrange_coeffs: &MetalBuffer<E4>,
    bf_data: &MetalBuffer<BF>,
    col_offset: usize, // BF column offset for first of 4 columns
    count: u32,        // trace_len
    result: &mut MetalBuffer<E4>,
    result_offset: usize,
    partial_buf: &mut MetalBuffer<E4>,
    context: &ProverContext,
) -> MetalResult<()> {
    let n = count as usize;
    // GPU transpose: 4 column-major BF cols → contiguous E4 buffer
    let d_e4_col: MetalBuffer<E4> = context.alloc(n)?;
    let cmd_buf = context.new_command_buffer()?;
    crate::ops_simple::transpose_bf4_to_e4_strided(
        device, &cmd_buf,
        bf_data.raw(), d_e4_col.raw(),
        count, col_offset as u32, count,
    )?;

    let num_blocks = (count + BARY_THREADS_PER_GROUP - 1) / BARY_THREADS_PER_GROUP;
    let config = MetalLaunchConfig::basic_1d(num_blocks, BARY_THREADS_PER_GROUP);
    dispatch_kernel(device, &cmd_buf, "ab_barycentric_eval_e4_col_partial_reduce", &config, |encoder| {
        set_buffer(encoder, 0, lagrange_coeffs.raw(), 0);
        set_buffer(encoder, 1, d_e4_col.raw(), 0);
        set_buffer(encoder, 2, partial_buf.raw(), 0);
        unsafe { set_bytes(encoder, 3, &count); }
    })?;
    let config2 = MetalLaunchConfig::basic_1d(1, BARY_THREADS_PER_GROUP);
    dispatch_kernel(device, &cmd_buf, "ab_barycentric_eval_final_reduce", &config2, |encoder| {
        set_buffer(encoder, 0, partial_buf.raw(), 0);
        set_buffer(encoder, 1, result.raw(), result_offset * std::mem::size_of::<E4>());
        unsafe { set_bytes(encoder, 2, &num_blocks); }
    })?;
    cmd_buf.commit_and_wait();
    Ok(())
}
