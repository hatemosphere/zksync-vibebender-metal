use crate::field::{BaseField, Ext2Field, Ext4Field};
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLDevice};

type BF = BaseField;
type E2 = Ext2Field;
type E4 = Ext4Field;

/// 2D tiled transpose: column-major → leaf-major with coalesced memory access.
/// Threadgroups cooperatively load 16-column tiles through shared memory.
pub fn transpose_cols_to_leaves_tiled(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    src: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    dst: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    num_leaves: u32,
    cols_count: u32,
    rows_per_leaf: u32,
    col_stride: u32,
) -> MetalResult<()> {
    let threads_per_group = 256u32;
    let threadgroups = (num_leaves + threads_per_group - 1) / threads_per_group;
    let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);
    dispatch_kernel(device, cmd_buf, "ab_transpose_cols_to_leaves_tiled_kernel", &config, |encoder| {
        set_buffer(encoder, 0, src, 0);
        set_buffer(encoder, 1, dst, 0);
        unsafe {
            set_bytes(encoder, 2, &num_leaves);
            set_bytes(encoder, 3, &cols_count);
            set_bytes(encoder, 4, &rows_per_leaf);
            set_bytes(encoder, 5, &col_stride);
        }
    })
}

/// Hash pre-transposed leaf-major data with blake2s (sequential reads).
pub fn blake2s_leaves_sequential(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    values: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    results: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    elems_per_leaf: u32,
    count: u32,
) -> MetalResult<()> {
    let threads_per_group = 128u32;
    let threadgroups = (count + threads_per_group - 1) / threads_per_group;
    let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);
    dispatch_kernel(device, cmd_buf, "ab_blake2s_leaves_sequential_kernel", &config, |encoder| {
        set_buffer(encoder, 0, values, 0);
        set_buffer(encoder, 1, results, 0);
        unsafe {
            set_bytes(encoder, 2, &elems_per_leaf);
            set_bytes(encoder, 3, &count);
        }
    })
}

/// GPU zero-fill: replaces CPU write_bytes for GPU buffers.
/// Operates on u32 granularity — byte_len must be a multiple of 4.
pub fn memset_zero(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    byte_len: usize,
) -> MetalResult<()> {
    let _ = device;
    if byte_len == 0 {
        return Ok(());
    }
    let encoder = cmd_buf.raw().blitCommandEncoder().ok_or_else(|| {
        crate::metal_runtime::error::MetalError::ResourceCreationFailed(
            "Failed to create blit command encoder".into(),
        )
    })?;
    encoder.fillBuffer_range_value(
        buffer,
        objc2_foundation::NSRange::new(0, byte_len),
        0,
    );
    encoder.endEncoding();
    cmd_buf.maybe_auto_commit();
    Ok(())
}

/// GPU zero-fill for a subrange of a buffer.
/// `byte_offset` and `byte_len` must be multiples of 4.
pub fn memset_zero_offset(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    byte_offset: usize,
    byte_len: usize,
) -> MetalResult<()> {
    let _ = device;
    if byte_len == 0 {
        return Ok(());
    }
    let encoder = cmd_buf.raw().blitCommandEncoder().ok_or_else(|| {
        crate::metal_runtime::error::MetalError::ResourceCreationFailed(
            "Failed to create blit command encoder".into(),
        )
    })?;
    encoder.fillBuffer_range_value(
        buffer,
        objc2_foundation::NSRange::new(byte_offset, byte_len),
        0,
    );
    encoder.endEncoding();
    cmd_buf.maybe_auto_commit();
    Ok(())
}

/// GPU buffer copy: replaces CPU memcpy on unified memory.
/// byte_len must be a multiple of 4.
pub fn memcpy_gpu(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    src: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    dst: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    byte_len: usize,
) -> MetalResult<()> {
    let _ = device;
    if byte_len == 0 {
        return Ok(());
    }
    let encoder = cmd_buf.raw().blitCommandEncoder().ok_or_else(|| {
        crate::metal_runtime::error::MetalError::ResourceCreationFailed(
            "Failed to create blit command encoder".into(),
        )
    })?;
    unsafe {
        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            src,
            0,
            dst,
            0,
            byte_len,
        );
    }
    encoder.endEncoding();
    cmd_buf.maybe_auto_commit();
    Ok(())
}

/// GPU transpose: 4 column-major BF columns → row-major E4.
/// src has 4*n BF elements (4 columns of n rows each, column-major).
/// dst has n E4 elements (row-major interleaved).
pub fn transpose_bf4_to_e4(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    src: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    dst: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    n: u32,
) -> MetalResult<()> {
    let threads_per_group = 256u32;
    let threadgroups = (n + threads_per_group - 1) / threads_per_group;
    let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);
    dispatch_kernel(device, cmd_buf, "ab_transpose_bf4_to_e4_kernel", &config, |encoder| {
        set_buffer(encoder, 0, src, 0);
        set_buffer(encoder, 1, dst, 0);
        unsafe { set_bytes(encoder, 2, &n); }
    })
}

/// GPU transpose: strided column-major 4 BF cols → contiguous row-major E4.
/// Reads from src at columns [base_col..base_col+4] with given stride.
pub fn transpose_bf4_to_e4_strided(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    src: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    dst: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    n: u32,
    base_col: u32,
    stride: u32,
) -> MetalResult<()> {
    let threads_per_group = 256u32;
    let threadgroups = (n + threads_per_group - 1) / threads_per_group;
    let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);
    dispatch_kernel(device, cmd_buf, "ab_transpose_bf4_to_e4_strided_kernel", &config, |encoder| {
        set_buffer(encoder, 0, src, 0);
        set_buffer(encoder, 1, dst, 0);
        unsafe {
            set_bytes(encoder, 2, &n);
            set_bytes(encoder, 3, &base_col);
            set_bytes(encoder, 4, &stride);
        }
    })
}

/// GPU transpose: contiguous row-major E4 → strided column-major 4 BF cols.
/// Writes to dst at columns [base_col..base_col+4] with given stride.
pub fn transpose_e4_to_bf4(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    src: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    dst: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    n: u32,
    base_col: u32,
    stride: u32,
) -> MetalResult<()> {
    let threads_per_group = 256u32;
    let threadgroups = (n + threads_per_group - 1) / threads_per_group;
    let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);
    dispatch_kernel(device, cmd_buf, "ab_transpose_e4_to_bf4_kernel", &config, |encoder| {
        set_buffer(encoder, 0, src, 0);
        set_buffer(encoder, 1, dst, 0);
        unsafe {
            set_bytes(encoder, 2, &n);
            set_bytes(encoder, 3, &base_col);
            set_bytes(encoder, 4, &stride);
        }
    })
}

/// GPU rotate-right by 1: dst[i] = src[(i-1) mod count].
/// Used for shifted lagrange coefficient computation.
pub fn rotate_right_e4(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    src: &MetalBuffer<crate::field::Ext4Field>,
    dst: &MetalBuffer<crate::field::Ext4Field>,
) -> MetalResult<()> {
    let count = src.len() as u32;
    let threads_per_group = 256u32;
    let threadgroups = (count + threads_per_group - 1) / threads_per_group;
    let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);
    dispatch_kernel(device, cmd_buf, "ab_rotate_right_e4_kernel", &config, |encoder| {
        set_buffer(encoder, 0, src.raw(), 0);
        set_buffer(encoder, 1, dst.raw(), 0);
        unsafe { set_bytes(encoder, 2, &count); }
    })
}

/// PtrAndStride representation matching the MSL kernel layout.
/// The MSL kernels take separate buffer + stride arguments.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct MatrixDesc {
    pub stride: u32,
    pub rows: u32,
    pub cols: u32,
}

fn get_launch_config_2d(rows: u32, cols: u32) -> MetalLaunchConfig {
    let threads_per_group = 128u32;
    let threadgroups_x = (rows + threads_per_group - 1) / threads_per_group;
    MetalLaunchConfig::basic_2d(
        (threadgroups_x, cols),
        (threads_per_group, 1),
    )
}

/// Fill a matrix with a constant value.
pub fn set_by_val_bf(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    value: BF,
    result: &MetalBuffer<BF>,
    stride: u32,
    rows: u32,
    cols: u32,
) -> MetalResult<()> {
    let config = get_launch_config_2d(rows, cols);
    dispatch_kernel(device, cmd_buf, "ab_set_by_val_bf_kernel", &config, |encoder| {
        unsafe {
            set_bytes(encoder, 0, &value);
        }
        set_buffer(encoder, 1, result.raw(), 0);
        unsafe {
            set_bytes(encoder, 2, &stride);
            set_bytes(encoder, 3, &rows);
        }
    })
}

pub fn set_by_val_e2(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    value: E2,
    result: &MetalBuffer<E2>,
    stride: u32,
    rows: u32,
    cols: u32,
) -> MetalResult<()> {
    let config = get_launch_config_2d(rows, cols);
    dispatch_kernel(device, cmd_buf, "ab_set_by_val_e2_kernel", &config, |encoder| {
        unsafe {
            set_bytes(encoder, 0, &value);
        }
        set_buffer(encoder, 1, result.raw(), 0);
        unsafe {
            set_bytes(encoder, 2, &stride);
            set_bytes(encoder, 3, &rows);
        }
    })
}

pub fn set_by_val_e4(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    value: E4,
    result: &MetalBuffer<E4>,
    stride: u32,
    rows: u32,
    cols: u32,
) -> MetalResult<()> {
    let config = get_launch_config_2d(rows, cols);
    dispatch_kernel(device, cmd_buf, "ab_set_by_val_e4_kernel", &config, |encoder| {
        unsafe {
            set_bytes(encoder, 0, &value);
        }
        set_buffer(encoder, 1, result.raw(), 0);
        unsafe {
            set_bytes(encoder, 2, &stride);
            set_bytes(encoder, 3, &rows);
        }
    })
}

pub fn set_by_val_u32(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    value: u32,
    result: &MetalBuffer<u32>,
    stride: u32,
    rows: u32,
    cols: u32,
) -> MetalResult<()> {
    let config = get_launch_config_2d(rows, cols);
    dispatch_kernel(device, cmd_buf, "ab_set_by_val_u32_kernel", &config, |encoder| {
        unsafe {
            set_bytes(encoder, 0, &value);
        }
        set_buffer(encoder, 1, result.raw(), 0);
        unsafe {
            set_bytes(encoder, 2, &stride);
            set_bytes(encoder, 3, &rows);
        }
    })
}

pub fn set_by_val_u64(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    value: u64,
    result: &MetalBuffer<u64>,
    stride: u32,
    rows: u32,
    cols: u32,
) -> MetalResult<()> {
    let config = get_launch_config_2d(rows, cols);
    dispatch_kernel(device, cmd_buf, "ab_set_by_val_u64_kernel", &config, |encoder| {
        unsafe {
            set_bytes(encoder, 0, &value);
        }
        set_buffer(encoder, 1, result.raw(), 0);
        unsafe {
            set_bytes(encoder, 2, &stride);
            set_bytes(encoder, 3, &rows);
        }
    })
}

/// Copy one matrix to another.
macro_rules! set_by_ref_fn {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        pub fn $fn_name(
            device: &ProtocolObject<dyn MTLDevice>,
            cmd_buf: &MetalCommandBuffer,
            src: &MetalBuffer<$ty>,
            src_stride: u32,
            dst: &MetalBuffer<$ty>,
            dst_stride: u32,
            rows: u32,
            cols: u32,
        ) -> MetalResult<()> {
            let config = get_launch_config_2d(rows, cols);
            dispatch_kernel(device, cmd_buf, $kernel_name, &config, |encoder| {
                set_buffer(encoder, 0, src.raw(), 0);
                unsafe { set_bytes(encoder, 1, &src_stride); }
                set_buffer(encoder, 2, dst.raw(), 0);
                unsafe {
                    set_bytes(encoder, 3, &dst_stride);
                    set_bytes(encoder, 4, &rows);
                }
            })
        }
    };
}

set_by_ref_fn!(set_by_ref_bf, "ab_set_by_ref_base_field_kernel", BF);
set_by_ref_fn!(set_by_ref_e2, "ab_set_by_ref_ext2_field_kernel", E2);
set_by_ref_fn!(set_by_ref_e4, "ab_set_by_ref_ext4_field_kernel", E4);
set_by_ref_fn!(set_by_ref_u32, "ab_set_by_ref_u32_kernel", u32);
set_by_ref_fn!(set_by_ref_u64, "ab_set_by_ref_u64_kernel", u64);

/// Unary operations: dbl, inv, neg, sqr.
macro_rules! unary_op_fn {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        pub fn $fn_name(
            device: &ProtocolObject<dyn MTLDevice>,
            cmd_buf: &MetalCommandBuffer,
            src: &MetalBuffer<$ty>,
            src_stride: u32,
            dst: &MetalBuffer<$ty>,
            dst_stride: u32,
            rows: u32,
            cols: u32,
        ) -> MetalResult<()> {
            let config = get_launch_config_2d(rows, cols);
            dispatch_kernel(device, cmd_buf, $kernel_name, &config, |encoder| {
                set_buffer(encoder, 0, src.raw(), 0);
                unsafe { set_bytes(encoder, 1, &src_stride); }
                set_buffer(encoder, 2, dst.raw(), 0);
                unsafe {
                    set_bytes(encoder, 3, &dst_stride);
                    set_bytes(encoder, 4, &rows);
                }
            })
        }
    };
}

unary_op_fn!(dbl_bf, "ab_dbl_base_field_kernel", BF);
unary_op_fn!(dbl_e2, "ab_dbl_ext2_field_kernel", E2);
unary_op_fn!(dbl_e4, "ab_dbl_ext4_field_kernel", E4);
unary_op_fn!(inv_bf, "ab_inv_base_field_kernel", BF);
unary_op_fn!(inv_e2, "ab_inv_ext2_field_kernel", E2);
unary_op_fn!(inv_e4, "ab_inv_ext4_field_kernel", E4);
unary_op_fn!(neg_bf, "ab_neg_base_field_kernel", BF);
unary_op_fn!(neg_e2, "ab_neg_ext2_field_kernel", E2);
unary_op_fn!(neg_e4, "ab_neg_ext4_field_kernel", E4);
unary_op_fn!(sqr_bf, "ab_sqr_base_field_kernel", BF);
unary_op_fn!(sqr_e2, "ab_sqr_ext2_field_kernel", E2);
unary_op_fn!(sqr_e4, "ab_sqr_ext4_field_kernel", E4);

/// Parametrized operations: pow, shl, shr.
macro_rules! parametrized_op_fn {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        pub fn $fn_name(
            device: &ProtocolObject<dyn MTLDevice>,
            cmd_buf: &MetalCommandBuffer,
            src: &MetalBuffer<$ty>,
            src_stride: u32,
            dst: &MetalBuffer<$ty>,
            dst_stride: u32,
            rows: u32,
            cols: u32,
            param: u32,
        ) -> MetalResult<()> {
            let config = get_launch_config_2d(rows, cols);
            dispatch_kernel(device, cmd_buf, $kernel_name, &config, |encoder| {
                set_buffer(encoder, 0, src.raw(), 0);
                unsafe { set_bytes(encoder, 1, &src_stride); }
                set_buffer(encoder, 2, dst.raw(), 0);
                unsafe {
                    set_bytes(encoder, 3, &dst_stride);
                    set_bytes(encoder, 4, &rows);
                    set_bytes(encoder, 5, &param);
                }
            })
        }
    };
}

parametrized_op_fn!(pow_bf, "ab_pow_base_field_kernel", BF);
parametrized_op_fn!(pow_e2, "ab_pow_ext2_field_kernel", E2);
parametrized_op_fn!(pow_e4, "ab_pow_ext4_field_kernel", E4);
parametrized_op_fn!(shl_bf, "ab_shl_base_field_kernel", BF);
parametrized_op_fn!(shl_e2, "ab_shl_ext2_field_kernel", E2);
parametrized_op_fn!(shl_e4, "ab_shl_ext4_field_kernel", E4);
parametrized_op_fn!(shr_bf, "ab_shr_base_field_kernel", BF);
parametrized_op_fn!(shr_e2, "ab_shr_ext2_field_kernel", E2);
parametrized_op_fn!(shr_e4, "ab_shr_ext4_field_kernel", E4);

/// Binary operations: add, mul, sub.
/// Note: The kernel name encodes the MSL field type names (base_field, ext2_field, ext4_field).
macro_rules! binary_op_fn {
    ($fn_name:ident, $kernel_name:expr, $t0:ty, $t1:ty, $tr:ty) => {
        pub fn $fn_name(
            device: &ProtocolObject<dyn MTLDevice>,
            cmd_buf: &MetalCommandBuffer,
            src0: &MetalBuffer<$t0>,
            src0_stride: u32,
            src1: &MetalBuffer<$t1>,
            src1_stride: u32,
            dst: &MetalBuffer<$tr>,
            dst_stride: u32,
            rows: u32,
            cols: u32,
        ) -> MetalResult<()> {
            let config = get_launch_config_2d(rows, cols);
            dispatch_kernel(device, cmd_buf, $kernel_name, &config, |encoder| {
                set_buffer(encoder, 0, src0.raw(), 0);
                unsafe { set_bytes(encoder, 1, &src0_stride); }
                set_buffer(encoder, 2, src1.raw(), 0);
                unsafe { set_bytes(encoder, 3, &src1_stride); }
                set_buffer(encoder, 4, dst.raw(), 0);
                unsafe {
                    set_bytes(encoder, 5, &dst_stride);
                    set_bytes(encoder, 6, &rows);
                }
            })
        }
    };
}

// add
binary_op_fn!(add_bf_bf, "ab_add_base_field_base_field_kernel", BF, BF, BF);
binary_op_fn!(add_bf_e2, "ab_add_base_field_ext2_field_kernel", BF, E2, E2);
binary_op_fn!(add_e2_bf, "ab_add_ext2_field_base_field_kernel", E2, BF, E2);
binary_op_fn!(add_e2_e2, "ab_add_ext2_field_ext2_field_kernel", E2, E2, E2);
binary_op_fn!(add_bf_e4, "ab_add_base_field_ext4_field_kernel", BF, E4, E4);
binary_op_fn!(add_e2_e4, "ab_add_ext2_field_ext4_field_kernel", E2, E4, E4);
binary_op_fn!(add_e4_bf, "ab_add_ext4_field_base_field_kernel", E4, BF, E4);
binary_op_fn!(add_e4_e2, "ab_add_ext4_field_ext2_field_kernel", E4, E2, E4);
binary_op_fn!(add_e4_e4, "ab_add_ext4_field_ext4_field_kernel", E4, E4, E4);

// mul
binary_op_fn!(mul_bf_bf, "ab_mul_base_field_base_field_kernel", BF, BF, BF);
binary_op_fn!(mul_bf_e2, "ab_mul_base_field_ext2_field_kernel", BF, E2, E2);
binary_op_fn!(mul_e2_bf, "ab_mul_ext2_field_base_field_kernel", E2, BF, E2);
binary_op_fn!(mul_e2_e2, "ab_mul_ext2_field_ext2_field_kernel", E2, E2, E2);
binary_op_fn!(mul_bf_e4, "ab_mul_base_field_ext4_field_kernel", BF, E4, E4);
binary_op_fn!(mul_e2_e4, "ab_mul_ext2_field_ext4_field_kernel", E2, E4, E4);
binary_op_fn!(mul_e4_bf, "ab_mul_ext4_field_base_field_kernel", E4, BF, E4);
binary_op_fn!(mul_e4_e2, "ab_mul_ext4_field_ext2_field_kernel", E4, E2, E4);
binary_op_fn!(mul_e4_e4, "ab_mul_ext4_field_ext4_field_kernel", E4, E4, E4);

// sub
binary_op_fn!(sub_bf_bf, "ab_sub_base_field_base_field_kernel", BF, BF, BF);
binary_op_fn!(sub_bf_e2, "ab_sub_base_field_ext2_field_kernel", BF, E2, E2);
binary_op_fn!(sub_e2_bf, "ab_sub_ext2_field_base_field_kernel", E2, BF, E2);
binary_op_fn!(sub_e2_e2, "ab_sub_ext2_field_ext2_field_kernel", E2, E2, E2);
binary_op_fn!(sub_bf_e4, "ab_sub_base_field_ext4_field_kernel", BF, E4, E4);
binary_op_fn!(sub_e2_e4, "ab_sub_ext2_field_ext4_field_kernel", E2, E4, E4);
binary_op_fn!(sub_e4_bf, "ab_sub_ext4_field_base_field_kernel", E4, BF, E4);
binary_op_fn!(sub_e4_e2, "ab_sub_ext4_field_ext2_field_kernel", E4, E2, E4);
binary_op_fn!(sub_e4_e4, "ab_sub_ext4_field_ext4_field_kernel", E4, E4, E4);
