use crate::field::{BaseField, Ext4Field};
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use crate::utils::div_ceil;
use field::Field;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer as _, MTLDevice};

type BF = BaseField;
type E4 = Ext4Field;

const SCAN_THREADS_PER_GROUP: u32 = 256;

#[derive(Copy, Clone)]
pub enum ScanOperation {
    Sum,
    Product,
}

fn kernel_suffix(operation: ScanOperation, type_name: &str) -> String {
    let op = match operation {
        ScanOperation::Sum => "add",
        ScanOperation::Product => "mul",
    };
    format!("{}_{}", op, type_name)
}

/// Returns the number of bytes needed for scan temp storage.
/// Temp storage = block_totals array of num_blocks elements.
pub fn get_scan_temp_storage_elems(num_items: u32) -> u32 {
    div_ceil(num_items as usize, SCAN_THREADS_PER_GROUP as usize) as u32
}

/// Perform a 3-pass parallel prefix scan.
///
/// `temp_storage` must have at least `get_scan_temp_storage_elems(num_items)` elements.
#[allow(clippy::too_many_arguments)]
fn scan_impl(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    suffix: &str,
    d_in: &dyn RawBuffer,
    d_out: &dyn RawBuffer,
    d_block_totals: &dyn RawBuffer,
    num_items: u32,
    inclusive: bool,
) -> MetalResult<()> {
    let num_blocks = div_ceil(num_items as usize, SCAN_THREADS_PER_GROUP as usize) as u32;
    let inclusive_u32: u32 = if inclusive { 1 } else { 0 };

    // Pass 1: per-threadgroup scan
    let pass1_name = format!("ab_scan_pass1_{}_kernel", suffix);
    let config1 = MetalLaunchConfig::basic_1d(num_blocks, SCAN_THREADS_PER_GROUP);
    dispatch_kernel(device, cmd_buf, &pass1_name, &config1, |encoder| {
        set_buffer(encoder, 0, d_in.raw_buf(), 0);
        set_buffer(encoder, 1, d_out.raw_buf(), 0);
        set_buffer(encoder, 2, d_block_totals.raw_buf(), 0);
        unsafe {
            set_bytes(encoder, 3, &num_items);
            set_bytes(encoder, 4, &inclusive_u32);
        }
    })?;

    if num_blocks > 1 {
        if num_blocks <= SCAN_THREADS_PER_GROUP {
            // Pass 2: scan block totals in a single threadgroup
            let pass2_name = format!("ab_scan_pass2_{}_kernel", suffix);
            let config2 = MetalLaunchConfig::basic_1d(1, SCAN_THREADS_PER_GROUP);
            dispatch_kernel(device, cmd_buf, &pass2_name, &config2, |encoder| {
                set_buffer(encoder, 0, d_block_totals.raw_buf(), 0);
                unsafe {
                    set_bytes(encoder, 1, &num_blocks);
                }
            })?;
        } else {
            // Recursive GPU scan of block totals (avoids CPU sync point).
            // Allocate a small temp buffer for the recursive level.
            let num_meta_blocks = div_ceil(num_blocks as usize, SCAN_THREADS_PER_GROUP as usize) as u32;
            let elem_size = d_block_totals.raw_buf().length() as usize / std::cmp::max(num_blocks as usize, 1);
            let meta_buf_size = (num_meta_blocks as usize * elem_size) as u64;
            let meta_buf = device
                .newBufferWithLength_options(meta_buf_size as usize, objc2_metal::MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate recursive scan buffer");
            let meta_wrapper = RawMTLBuffer(&*meta_buf);
            scan_impl(
                device, cmd_buf, suffix,
                d_block_totals, d_block_totals, &meta_wrapper,
                num_blocks, false,
            )?;
        }

        // Pass 3: add block prefixes
        let pass3_name = format!("ab_scan_pass3_{}_kernel", suffix);
        let config3 = MetalLaunchConfig::basic_1d(num_blocks, SCAN_THREADS_PER_GROUP);
        dispatch_kernel(device, cmd_buf, &pass3_name, &config3, |encoder| {
            set_buffer(encoder, 0, d_out.raw_buf(), 0);
            set_buffer(encoder, 1, d_block_totals.raw_buf(), 0);
            unsafe {
                set_bytes(encoder, 2, &num_items);
            }
        })?;
    }

    Ok(())
}

/// CPU fallback for EXCLUSIVE prefix scan of block totals when they exceed GPU single-threadgroup capacity.
/// Must match pass2 kernel behavior: block_totals[i] = op(blocks 0..i-1), identity for i=0.
/// Works on unified memory (StorageModeShared) — no copy needed.
fn scan_block_totals_on_cpu(d_block_totals: &dyn RawBuffer, count: usize, suffix: &str) {
    let ptr = d_block_totals.raw_buf().contents().as_ptr();
    match suffix {
        "mul_e4" => {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut E4, count) };
            let mut running = E4::ONE;
            for v in slice.iter_mut() {
                let cur = *v;
                *v = running;
                running.mul_assign(&cur);
            }
        }
        "add_e4" => {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut E4, count) };
            let mut running = E4::ZERO;
            for v in slice.iter_mut() {
                let cur = *v;
                *v = running;
                running.add_assign(&cur);
            }
        }
        "add_bf" => {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut BF, count) };
            let mut running = BF::ZERO;
            for v in slice.iter_mut() {
                let cur = *v;
                *v = running;
                running.add_assign(&cur);
            }
        }
        "add_u32" => {
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, count) };
            let mut running = 0u32;
            for v in slice.iter_mut() {
                let cur = *v;
                *v = running;
                running = running.wrapping_add(cur);
            }
        }
        _ => panic!("unsupported scan suffix for CPU fallback: {}", suffix),
    }
}

/// Wrapper for a raw MTLBuffer reference (used in recursive scan).
struct RawMTLBuffer<'a>(&'a ProtocolObject<dyn objc2_metal::MTLBuffer>);

impl RawBuffer for RawMTLBuffer<'_> {
    fn raw_buf(&self) -> &ProtocolObject<dyn objc2_metal::MTLBuffer> {
        self.0
    }
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

/// Scan for base_field elements.
#[allow(clippy::too_many_arguments)]
pub fn scan_bf(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    operation: ScanOperation,
    inclusive: bool,
    d_in: &MetalBuffer<BF>,
    d_out: &MetalBuffer<BF>,
    d_temp: &MetalBuffer<BF>,
    num_items: u32,
) -> MetalResult<()> {
    let suffix = kernel_suffix(operation, "bf");
    scan_impl(device, cmd_buf, &suffix, d_in, d_out, d_temp, num_items, inclusive)
}

/// Scan for ext4_field elements.
#[allow(clippy::too_many_arguments)]
pub fn scan_e4(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    operation: ScanOperation,
    inclusive: bool,
    d_in: &MetalBuffer<E4>,
    d_out: &MetalBuffer<E4>,
    d_temp: &MetalBuffer<E4>,
    num_items: u32,
) -> MetalResult<()> {
    let suffix = kernel_suffix(operation, "e4");
    scan_impl(device, cmd_buf, &suffix, d_in, d_out, d_temp, num_items, inclusive)
}

/// Scan for u32 elements (sum only).
pub fn scan_u32(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    inclusive: bool,
    d_in: &MetalBuffer<u32>,
    d_out: &MetalBuffer<u32>,
    d_temp: &MetalBuffer<u32>,
    num_items: u32,
) -> MetalResult<()> {
    scan_impl(device, cmd_buf, "add_u32", d_in, d_out, d_temp, num_items, inclusive)
}
