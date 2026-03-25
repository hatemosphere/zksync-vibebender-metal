use crate::field::{BaseField, Ext2Field, Ext4Field};
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use crate::utils::get_grid_threadgroup_dims;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

type BF = BaseField;
type E2 = Ext2Field;
type E4 = Ext4Field;

const SIMD_GROUP_SIZE: usize = 32;

fn get_launch_dims(count: u32) -> (u32, u32) {
    get_grid_threadgroup_dims(SIMD_GROUP_SIZE * 4, count)
}

/// Compute powers: result[i] = base^(i + offset), optionally with bit-reversed indices.
macro_rules! get_powers_by_val_fn {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        pub fn $fn_name(
            device: &ProtocolObject<dyn MTLDevice>,
            cmd_buf: &MetalCommandBuffer,
            base: $ty,
            offset: u32,
            bit_reverse: bool,
            result: &MetalBuffer<$ty>,
            count: u32,
        ) -> MetalResult<()> {
            let (threadgroups, threads_per_group) = get_launch_dims(count);
            let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);
            let bit_reverse_u32: u32 = if bit_reverse { 1 } else { 0 };
            dispatch_kernel(device, cmd_buf, $kernel_name, &config, |encoder| {
                unsafe {
                    set_bytes(encoder, 0, &base);
                    set_bytes(encoder, 1, &offset);
                    set_bytes(encoder, 2, &bit_reverse_u32);
                }
                set_buffer(encoder, 3, result.raw(), 0);
                unsafe { set_bytes(encoder, 4, &count); }
            })
        }
    };
}

get_powers_by_val_fn!(get_powers_by_val_bf, "ab_get_powers_by_val_bf_kernel", BF);
get_powers_by_val_fn!(get_powers_by_val_e2, "ab_get_powers_by_val_e2_kernel", E2);
get_powers_by_val_fn!(get_powers_by_val_e4, "ab_get_powers_by_val_e4_kernel", E4);

macro_rules! get_powers_by_ref_fn {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        pub fn $fn_name(
            device: &ProtocolObject<dyn MTLDevice>,
            cmd_buf: &MetalCommandBuffer,
            base: &MetalBuffer<$ty>,
            offset: u32,
            bit_reverse: bool,
            result: &MetalBuffer<$ty>,
            count: u32,
        ) -> MetalResult<()> {
            let (threadgroups, threads_per_group) = get_launch_dims(count);
            let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);
            let bit_reverse_u32: u32 = if bit_reverse { 1 } else { 0 };
            dispatch_kernel(device, cmd_buf, $kernel_name, &config, |encoder| {
                set_buffer(encoder, 0, base.raw(), 0);
                unsafe {
                    set_bytes(encoder, 1, &offset);
                    set_bytes(encoder, 2, &bit_reverse_u32);
                }
                set_buffer(encoder, 3, result.raw(), 0);
                unsafe { set_bytes(encoder, 4, &count); }
            })
        }
    };
}

get_powers_by_ref_fn!(get_powers_by_ref_bf, "ab_get_powers_by_ref_bf_kernel", BF);
get_powers_by_ref_fn!(get_powers_by_ref_e2, "ab_get_powers_by_ref_e2_kernel", E2);
get_powers_by_ref_fn!(get_powers_by_ref_e4, "ab_get_powers_by_ref_e4_kernel", E4);

/// Batch inversion using Montgomery's trick.
macro_rules! batch_inv_fn {
    ($fn_name:ident, $kernel_name:expr, $ty:ty, $batch_size:expr) => {
        pub fn $fn_name(
            device: &ProtocolObject<dyn MTLDevice>,
            cmd_buf: &MetalCommandBuffer,
            src: &MetalBuffer<$ty>,
            dst: &MetalBuffer<$ty>,
            count: u32,
        ) -> MetalResult<()> {
            let block_dim = (SIMD_GROUP_SIZE * 4) as u32;
            let batch_size: u32 = $batch_size;
            let grid_dim = (count + batch_size * block_dim - 1) / (batch_size * block_dim);
            let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);
            dispatch_kernel(device, cmd_buf, $kernel_name, &config, |encoder| {
                set_buffer(encoder, 0, src.raw(), 0);
                set_buffer(encoder, 1, dst.raw(), 0);
                unsafe { set_bytes(encoder, 2, &count); }
            })
        }
    };
}

batch_inv_fn!(batch_inv_bf, "ab_batch_inv_bf_kernel", BF, 20);
batch_inv_fn!(batch_inv_e2, "ab_batch_inv_e2_kernel", E2, 5);
batch_inv_fn!(batch_inv_e4, "ab_batch_inv_e4_kernel", E4, 3);

/// Naive bit-reversal permutation.
fn get_launch_config_2d(rows: u32, cols: u32) -> MetalLaunchConfig {
    let threads_per_group = 128u32;
    let threadgroups_x = (rows + threads_per_group - 1) / threads_per_group;
    MetalLaunchConfig::basic_2d((threadgroups_x, cols), (threads_per_group, 1))
}

macro_rules! bit_reverse_naive_fn {
    ($fn_name:ident, $kernel_name:expr, $ty:ty) => {
        pub fn $fn_name(
            device: &ProtocolObject<dyn MTLDevice>,
            cmd_buf: &MetalCommandBuffer,
            src: &MetalBuffer<$ty>,
            src_stride: u32,
            dst: &MetalBuffer<$ty>,
            dst_stride: u32,
            log_count: u32,
            cols: u32,
        ) -> MetalResult<()> {
            let rows = 1u32 << log_count;
            let config = get_launch_config_2d(rows, cols);
            dispatch_kernel(device, cmd_buf, $kernel_name, &config, |encoder| {
                set_buffer(encoder, 0, src.raw(), 0);
                unsafe { set_bytes(encoder, 1, &src_stride); }
                set_buffer(encoder, 2, dst.raw(), 0);
                unsafe {
                    set_bytes(encoder, 3, &dst_stride);
                    set_bytes(encoder, 4, &log_count);
                }
            })
        }
    };
}

bit_reverse_naive_fn!(bit_reverse_naive_bf, "ab_bit_reverse_naive_bf_kernel", BF);
bit_reverse_naive_fn!(bit_reverse_naive_e2, "ab_bit_reverse_naive_e2_kernel", E2);
bit_reverse_naive_fn!(bit_reverse_naive_e4, "ab_bit_reverse_naive_e4_kernel", E4);

/// Digest type matching the MSL dg struct (8 x base_field = 32 bytes).
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Digest {
    pub values: [BF; 8],
}

bit_reverse_naive_fn!(bit_reverse_naive_dg, "ab_bit_reverse_naive_dg_kernel", Digest);

/// Powers data for the 3-layer twiddle factor lookup.
/// These must be passed as buffer arguments to the fold kernel.
#[repr(C)]
pub struct PowersLayerDesc {
    pub mask: u32,
    pub log_count: u32,
}

/// Fold kernel: used in FRI protocol.
pub fn fold(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    challenge: &MetalBuffer<E4>,
    src: &MetalBuffer<E4>,
    dst: &MetalBuffer<E4>,
    root_offset: u32,
    log_count: u32,
    powers_fine: &MetalBuffer<E2>,
    powers_fine_desc: &PowersLayerDesc,
    powers_coarser: &MetalBuffer<E2>,
    powers_coarser_desc: &PowersLayerDesc,
    powers_coarsest: &MetalBuffer<E2>,
    powers_coarsest_desc: &PowersLayerDesc,
) -> MetalResult<()> {
    let count = 1u32 << log_count;
    let (threadgroups, threads_per_group) = get_launch_dims(count);
    let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);
    dispatch_kernel(device, cmd_buf, "ab_fold_kernel", &config, |encoder| {
        set_buffer(encoder, 0, challenge.raw(), 0);
        set_buffer(encoder, 1, src.raw(), 0);
        set_buffer(encoder, 2, dst.raw(), 0);
        unsafe {
            set_bytes(encoder, 3, &root_offset);
            set_bytes(encoder, 4, &log_count);
        }
        set_buffer(encoder, 5, powers_fine.raw(), 0);
        unsafe {
            set_bytes(encoder, 6, &powers_fine_desc.mask);
            set_bytes(encoder, 7, &powers_fine_desc.log_count);
        }
        set_buffer(encoder, 8, powers_coarser.raw(), 0);
        unsafe {
            set_bytes(encoder, 9, &powers_coarser_desc.mask);
            set_bytes(encoder, 10, &powers_coarser_desc.log_count);
        }
        set_buffer(encoder, 11, powers_coarsest.raw(), 0);
        unsafe {
            set_bytes(encoder, 12, &powers_coarsest_desc.mask);
        }
    })
}

#[cfg(test)]
mod tests {
    use field::Field;
    use rand::Rng;

    use super::*;
    use crate::metal_runtime::buffer::MetalBuffer;
    use crate::metal_runtime::command_queue::MetalCommandQueue;
    use crate::metal_runtime::device::system_default_device;
    use crate::metal_runtime::pipeline::init_shader_library;

    /// Batch inverse: verify that a[i] * inv(a[i]) == 1 for all elements.
    #[test]
    fn test_batch_inv_bf() {
        const N: usize = 1 << 14;
        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let mut rng = rand::rng();
        // Use nonzero random elements
        let h_in: Vec<BF> = (0..N)
            .map(|_| {
                loop {
                    let v = BF::from_nonreduced_u32(rng.random());
                    if v != BF::ZERO {
                        return v;
                    }
                }
            })
            .collect();

        let d_in = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_out = MetalBuffer::<BF>::alloc(device, N).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        batch_inv_bf(device, &cmd_buf, &d_in, &d_out, N as u32).unwrap();
        cmd_buf.commit_and_wait();

        let mut h_out = vec![BF::ZERO; N];
        unsafe {
            d_out.copy_to_slice(&mut h_out);
        }

        for i in 0..N {
            let mut product = h_in[i];
            product.mul_assign(&h_out[i]);
            assert_eq!(
                product,
                BF::ONE,
                "batch_inv_bf: a[{}] * inv(a[{}]) != 1",
                i,
                i
            );
        }
    }

    /// Bit-reverse permutation: applying it twice is the identity.
    #[test]
    fn test_bit_reverse_bf() {
        const LOG_N: u32 = 12;
        const N: usize = 1 << LOG_N;
        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let mut rng = rand::rng();
        let h_in: Vec<BF> = (0..N)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect();

        let d_src = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_mid = MetalBuffer::<BF>::alloc(device, N).unwrap();
        let d_dst = MetalBuffer::<BF>::alloc(device, N).unwrap();

        // First bit-reverse
        let cmd_buf = queue.new_command_buffer().unwrap();
        bit_reverse_naive_bf(
            device, &cmd_buf, &d_src, N as u32, &d_mid, N as u32, LOG_N, 1,
        )
        .unwrap();
        cmd_buf.commit_and_wait();

        // Verify it's actually permuted (not identity for random data)
        let mut h_mid = vec![BF::ZERO; N];
        unsafe {
            d_mid.copy_to_slice(&mut h_mid);
        }
        // For N > 2, a random array should differ from its bit-reversal
        assert_ne!(h_in, h_mid, "bit-reverse should permute the data");

        // Second bit-reverse: should recover original
        let cmd_buf2 = queue.new_command_buffer().unwrap();
        bit_reverse_naive_bf(
            device, &cmd_buf2, &d_mid, N as u32, &d_dst, N as u32, LOG_N, 1,
        )
        .unwrap();
        cmd_buf2.commit_and_wait();

        let mut h_out = vec![BF::ZERO; N];
        unsafe {
            d_dst.copy_to_slice(&mut h_out);
        }
        assert_eq!(h_in, h_out, "double bit-reverse should be identity");
    }

    /// Bit-reverse permutation: verify against CPU reference.
    #[test]
    fn test_bit_reverse_bf_cpu_reference() {
        const LOG_N: u32 = 10;
        const N: usize = 1 << LOG_N;
        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let mut rng = rand::rng();
        let h_in: Vec<BF> = (0..N)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect();

        let d_src = MetalBuffer::from_slice(device, &h_in).unwrap();
        let d_dst = MetalBuffer::<BF>::alloc(device, N).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        bit_reverse_naive_bf(
            device, &cmd_buf, &d_src, N as u32, &d_dst, N as u32, LOG_N, 1,
        )
        .unwrap();
        cmd_buf.commit_and_wait();

        let mut h_out = vec![BF::ZERO; N];
        unsafe {
            d_dst.copy_to_slice(&mut h_out);
        }

        // CPU reference bit-reverse
        for i in 0..N {
            let rev = (i as u32).reverse_bits() >> (32 - LOG_N);
            assert_eq!(
                h_in[i], h_out[rev as usize],
                "bit_reverse mismatch at index {}",
                i
            );
        }
    }
}
