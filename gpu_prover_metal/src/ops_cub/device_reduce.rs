use crate::field::{BaseField, Ext2Field, Ext4Field};
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use crate::utils::div_ceil;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

type BF = BaseField;
type E2 = Ext2Field;
type E4 = Ext4Field;

const REDUCE_THREADS_PER_GROUP: u32 = 256;

#[derive(Copy, Clone)]
pub enum ReduceOperation {
    Sum,
    Product,
}

fn kernel_suffix(operation: ReduceOperation, type_name: &str) -> String {
    let op = match operation {
        ReduceOperation::Sum => "add",
        ReduceOperation::Product => "mul",
    };
    format!("{}_{}", op, type_name)
}

/// Returns the number of partial-result elements needed for temp storage.
pub fn get_reduce_temp_storage_elems(num_items: u32) -> u32 {
    div_ceil(num_items as usize, REDUCE_THREADS_PER_GROUP as usize) as u32
}

/// Trait to abstract over the raw MTLBuffer for different element types.
trait RawBuffer {
    fn raw_buf(&self) -> &ProtocolObject<dyn objc2_metal::MTLBuffer>;
}

impl<T> RawBuffer for MetalBuffer<T> {
    fn raw_buf(&self) -> &ProtocolObject<dyn objc2_metal::MTLBuffer> {
        self.raw()
    }
}

/// 2-pass reduce: pass1 produces per-block partials, pass2 reduces those.
fn reduce_impl(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    suffix: &str,
    d_in: &dyn RawBuffer,
    d_out: &dyn RawBuffer,
    d_partials: &dyn RawBuffer,
    num_items: u32,
) -> MetalResult<()> {
    let num_blocks = div_ceil(num_items as usize, REDUCE_THREADS_PER_GROUP as usize) as u32;

    // Pass 1
    let pass1_name = format!("ab_reduce_pass1_{}_kernel", suffix);
    let config1 = MetalLaunchConfig::basic_1d(num_blocks, REDUCE_THREADS_PER_GROUP);
    dispatch_kernel(device, cmd_buf, &pass1_name, &config1, |encoder| {
        set_buffer(encoder, 0, d_in.raw_buf(), 0);
        set_buffer(encoder, 1, d_partials.raw_buf(), 0);
        unsafe {
            set_bytes(encoder, 2, &num_items);
        }
    })?;

    // Pass 2
    let pass2_name = format!("ab_reduce_pass2_{}_kernel", suffix);
    let config2 = MetalLaunchConfig::basic_1d(1, REDUCE_THREADS_PER_GROUP);
    dispatch_kernel(device, cmd_buf, &pass2_name, &config2, |encoder| {
        set_buffer(encoder, 0, d_partials.raw_buf(), 0);
        set_buffer(encoder, 1, d_out.raw_buf(), 0);
        unsafe {
            set_bytes(encoder, 2, &num_blocks);
        }
    })?;

    Ok(())
}

/// Segmented (batch) reduce: one threadgroup per segment.
fn segmented_reduce_impl(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    suffix: &str,
    d_in: &dyn RawBuffer,
    d_out: &dyn RawBuffer,
    stride: u32,
    num_segments: u32,
    num_items_per_segment: u32,
) -> MetalResult<()> {
    let kernel_name = format!("ab_segmented_reduce_{}_kernel", suffix);
    let config = MetalLaunchConfig::basic_1d(num_segments, REDUCE_THREADS_PER_GROUP);
    dispatch_kernel(device, cmd_buf, &kernel_name, &config, |encoder| {
        set_buffer(encoder, 0, d_in.raw_buf(), 0);
        set_buffer(encoder, 1, d_out.raw_buf(), 0);
        unsafe {
            set_bytes(encoder, 2, &stride);
            set_bytes(encoder, 3, &num_segments);
            set_bytes(encoder, 4, &num_items_per_segment);
        }
    })
}

// --- Public API for each field type ---

macro_rules! reduce_fn {
    ($fn_name:ident, $seg_fn_name:ident, $type_suffix:expr, $ty:ty) => {
        pub fn $fn_name(
            device: &ProtocolObject<dyn MTLDevice>,
            cmd_buf: &MetalCommandBuffer,
            operation: ReduceOperation,
            d_in: &MetalBuffer<$ty>,
            d_out: &MetalBuffer<$ty>,
            d_partials: &MetalBuffer<$ty>,
            num_items: u32,
        ) -> MetalResult<()> {
            let suffix = kernel_suffix(operation, $type_suffix);
            reduce_impl(device, cmd_buf, &suffix, d_in, d_out, d_partials, num_items)
        }

        pub fn $seg_fn_name(
            device: &ProtocolObject<dyn MTLDevice>,
            cmd_buf: &MetalCommandBuffer,
            operation: ReduceOperation,
            d_in: &MetalBuffer<$ty>,
            d_out: &MetalBuffer<$ty>,
            stride: u32,
            num_segments: u32,
            num_items_per_segment: u32,
        ) -> MetalResult<()> {
            let suffix = kernel_suffix(operation, $type_suffix);
            segmented_reduce_impl(device, cmd_buf, &suffix, d_in, d_out, stride, num_segments, num_items_per_segment)
        }
    };
}

reduce_fn!(reduce_bf, segmented_reduce_bf, "bf", BF);
reduce_fn!(reduce_e2, segmented_reduce_e2, "e2", E2);
reduce_fn!(reduce_e4, segmented_reduce_e4, "e4", E4);
