use super::command_queue::MetalCommandBuffer;
use super::error::{MetalError, MetalResult};
use super::pipeline;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLDevice, MTLSize,
};

/// Grid and threadgroup dimensions for kernel dispatch,
/// analogous to `CudaLaunchConfig` with `Dim3` grid/block dims.
#[derive(Debug, Clone, Copy)]
pub struct MetalLaunchConfig {
    /// Total number of threadgroups in each dimension (analogous to CUDA grid_dim).
    pub grid_dim: MTLSize,
    /// Threads per threadgroup in each dimension (analogous to CUDA block_dim).
    pub threadgroup_dim: MTLSize,
}

impl MetalLaunchConfig {
    /// Create a 1D launch configuration.
    /// Analogous to `CudaLaunchConfig::basic(grid_x, block_x, stream)`.
    pub fn basic_1d(threadgroups: u32, threads_per_group: u32) -> Self {
        Self {
            grid_dim: MTLSize {
                width: threadgroups as usize,
                height: 1,
                depth: 1,
            },
            threadgroup_dim: MTLSize {
                width: threads_per_group as usize,
                height: 1,
                depth: 1,
            },
        }
    }

    /// Create a 2D launch configuration.
    pub fn basic_2d(grid: (u32, u32), threadgroup: (u32, u32)) -> Self {
        Self {
            grid_dim: MTLSize {
                width: grid.0 as usize,
                height: grid.1 as usize,
                depth: 1,
            },
            threadgroup_dim: MTLSize {
                width: threadgroup.0 as usize,
                height: threadgroup.1 as usize,
                depth: 1,
            },
        }
    }

    /// Create from CUDA-style grid/block dims (for easier porting).
    /// `grid_dim` = number of blocks, `block_dim` = threads per block.
    pub fn from_cuda_dims(grid_dim: impl Into<Dim3>, block_dim: impl Into<Dim3>) -> Self {
        let grid: Dim3 = grid_dim.into();
        let block: Dim3 = block_dim.into();
        Self {
            grid_dim: MTLSize {
                width: grid.x as usize,
                height: grid.y as usize,
                depth: grid.z as usize,
            },
            threadgroup_dim: MTLSize {
                width: block.x as usize,
                height: block.y as usize,
                depth: block.z as usize,
            },
        }
    }
}

/// 3D dimension type, mirrors `era_cudart::execution::Dim3`.
#[derive(Debug, Copy, Clone)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

impl Default for Dim3 {
    fn default() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Self { x, y, z: 1 }
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self { x, y, z }
    }
}

/// Dispatch a GPU kernel on a serial compute encoder.
///
/// Each call creates a new compute encoder, dispatches one kernel, and ends
/// the encoder. Encoders within a command buffer execute sequentially.
///
/// The `cmd_buf` parameter provides access to the underlying command queue.
pub fn dispatch_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    function_name: &str,
    config: &MetalLaunchConfig,
    encode_args: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> MetalResult<()> {
    dispatch_kernel_impl(device, cmd_buf, function_name, config, encode_args, false)
}

/// Dispatch a GPU kernel on a concurrent compute encoder.
///
/// Dispatches using `MTLDispatchType::Concurrent`, allowing the GPU to
/// execute this kernel in parallel with other concurrent dispatches in the
/// same command buffer. Use for independent kernels that don't read each
/// other's outputs (e.g., two independent Merkle tree builds).
///
/// IMPORTANT: A memory barrier (serial dispatch or `commit_and_wait`) is
/// needed BEFORE any kernel that reads outputs from concurrent dispatches.
pub fn dispatch_kernel_concurrent(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    function_name: &str,
    config: &MetalLaunchConfig,
    encode_args: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> MetalResult<()> {
    dispatch_kernel_impl(device, cmd_buf, function_name, config, encode_args, true)
}

fn dispatch_kernel_impl(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    function_name: &str,
    config: &MetalLaunchConfig,
    encode_args: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    concurrent: bool,
) -> MetalResult<()> {
    let pipeline = pipeline::get_pipeline(device, function_name)?;

    let encoder = if concurrent {
        cmd_buf
            .raw()
            .computeCommandEncoderWithDispatchType(objc2_metal::MTLDispatchType::Concurrent)
    } else {
        cmd_buf.raw().computeCommandEncoder()
    }
    .ok_or_else(|| {
        MetalError::ResourceCreationFailed("Failed to create compute command encoder".into())
    })?;

    // Label encoder with kernel name for Metal System Trace / Instruments visibility
    let label = objc2_foundation::NSString::from_str(function_name);
    encoder.setLabel(Some(&label));

    encoder.setComputePipelineState(&pipeline);
    encode_args(&encoder);

    encoder.dispatchThreadgroups_threadsPerThreadgroup(config.grid_dim, config.threadgroup_dim);
    encoder.endEncoding();

    #[cfg(feature = "log_gpu_stages_timings")]
    super::profiler::record_dispatch(function_name);

    // Check if we should auto-commit this batch to stay under the GPU watchdog.
    cmd_buf.maybe_auto_commit();

    Ok(())
}

/// Encode a buffer argument at the given index.
/// Convenience function for the common pattern of setting buffer arguments.
#[inline]
pub fn set_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: u32,
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
) {
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buffer), offset, index as usize);
    }
}

/// Encode a value argument (small data like u32, structs) at the given index.
///
/// # Safety
/// The value must be a plain-old-data type that matches the MSL kernel argument layout.
#[inline]
pub unsafe fn set_bytes<T>(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    index: u32,
    value: &T,
) {
    let ptr = value as *const T as *const std::ffi::c_void;
    let len = std::mem::size_of::<T>();
    unsafe {
        encoder.setBytes_length_atIndex(
            std::ptr::NonNull::new(ptr as *mut _).unwrap(),
            len,
            index as usize,
        );
    }
}
