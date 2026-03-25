use crate::field::BaseField;
use crate::metal_runtime::dispatch::{self, MetalLaunchConfig};
use crate::metal_runtime::{MetalBuffer, MetalCommandBuffer, MetalResult};
use crate::utils::{get_grid_threadgroup_dims, SIMD_GROUP_SIZE};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDevice};

type BF = BaseField;

pub const RATE: usize = 8;
pub const CAPACITY: usize = 8;
pub const WIDTH: usize = RATE + CAPACITY;

pub type Digest = [BF; CAPACITY];

/// Size of the threadgroup memory needed for bar_lookup table: 256 + 128 = 384 bytes.
const BAR_LOOKUP_SIZE: usize = (1 << 8) + (1 << 7);

pub fn launch_leaves_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    values: &MetalBuffer<BF>,
    results: &MetalBuffer<Digest>,
    log_rows_per_hash: u32,
) -> MetalResult<()> {
    let values_len = values.len();
    let count = results.len();
    assert_eq!(values_len % (count << log_rows_per_hash as usize), 0);
    let cols_count = (values_len / (count << log_rows_per_hash as usize)) as u32;
    let count_u32 = count as u32;
    let threads_per_group = SIMD_GROUP_SIZE * 4;
    let (threadgroups, tpg) = get_grid_threadgroup_dims(threads_per_group, count_u32);
    let config = MetalLaunchConfig::basic_1d(threadgroups, tpg);
    dispatch_with_threadgroup_memory(
        device,
        cmd_buf,
        "ab_monolith_leaves_kernel",
        &config,
        BAR_LOOKUP_SIZE,
        |encoder| {
            dispatch::set_buffer(encoder, 0, values.raw(), 0);
            dispatch::set_buffer(encoder, 1, results.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 2, &log_rows_per_hash);
                dispatch::set_bytes(encoder, 3, &cols_count);
                dispatch::set_bytes(encoder, 4, &count_u32);
            }
        },
    )
}

pub fn build_merkle_tree_leaves(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    values: &MetalBuffer<BF>,
    results: &MetalBuffer<Digest>,
    log_rows_per_hash: u32,
) -> MetalResult<()> {
    let values_len = values.len();
    let leaves_count = results.len();
    assert_eq!(values_len % leaves_count, 0);
    launch_leaves_kernel(device, cmd_buf, values, results, log_rows_per_hash)
}

pub fn launch_nodes_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    values: &MetalBuffer<Digest>,
    results: &MetalBuffer<Digest>,
) -> MetalResult<()> {
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(values_len, results_len * 2);
    let count = results_len as u32;
    let threads_per_group = SIMD_GROUP_SIZE * 4;
    let (threadgroups, tpg) = get_grid_threadgroup_dims(threads_per_group, count);
    let config = MetalLaunchConfig::basic_1d(threadgroups, tpg);
    dispatch_with_threadgroup_memory(
        device,
        cmd_buf,
        "ab_monolith_nodes_kernel",
        &config,
        BAR_LOOKUP_SIZE,
        |encoder| {
            dispatch::set_buffer(encoder, 0, values.raw(), 0);
            dispatch::set_buffer(encoder, 1, results.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 2, &count);
            }
        },
    )
}

pub fn gather_rows(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    indexes: &MetalBuffer<u32>,
    log_rows_per_index: u32,
    values: &MetalBuffer<BF>,
    values_stride: u32,
    results: &MetalBuffer<BF>,
    results_stride: u32,
    cols_count: u32,
) -> MetalResult<()> {
    let indexes_count = indexes.len() as u32;
    let rows_per_index = 1u32 << log_rows_per_index;
    let simd_size = SIMD_GROUP_SIZE as u32;
    let indexes_per_group = if log_rows_per_index < (crate::utils::LOG_SIMD_GROUP_SIZE as u32) {
        simd_size >> log_rows_per_index
    } else {
        1u32
    };
    let threadgroups_x = (indexes_count + indexes_per_group - 1) / indexes_per_group;
    let threadgroups_y = cols_count;
    let config = MetalLaunchConfig::basic_2d(
        (threadgroups_x, threadgroups_y),
        (rows_per_index, indexes_per_group),
    );
    dispatch::dispatch_kernel(
        device,
        cmd_buf,
        "ab_monolith_gather_rows_kernel",
        &config,
        |encoder| {
            dispatch::set_buffer(encoder, 0, indexes.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 1, &indexes_count);
            }
            dispatch::set_buffer(encoder, 2, values.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 3, &values_stride);
            }
            dispatch::set_buffer(encoder, 4, results.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 5, &results_stride);
            }
        },
    )
}

pub fn gather_merkle_paths(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    indexes: &MetalBuffer<u32>,
    values: &MetalBuffer<Digest>,
    results: &MetalBuffer<Digest>,
    layers_count: u32,
) -> MetalResult<()> {
    let indexes_count = indexes.len() as u32;
    let values_count = values.len();
    assert!(values_count.is_power_of_two());
    let log_values_count = values_count.trailing_zeros();
    assert_ne!(log_values_count, 0);
    let log_leaves_count = log_values_count - 1;
    assert!(layers_count < log_leaves_count);
    assert_eq!(
        indexes.len() * layers_count as usize,
        results.len()
    );
    let simd_size = SIMD_GROUP_SIZE as u32;
    assert_eq!(simd_size % CAPACITY as u32, 0);
    let indexes_per_group = simd_size / CAPACITY as u32;
    let threadgroups_x = (indexes_count + indexes_per_group - 1) / indexes_per_group;
    let config = MetalLaunchConfig::basic_2d(
        (threadgroups_x, layers_count),
        (CAPACITY as u32, indexes_per_group),
    );
    dispatch::dispatch_kernel(
        device,
        cmd_buf,
        "ab_monolith_gather_merkle_paths_kernel",
        &config,
        |encoder| {
            dispatch::set_buffer(encoder, 0, indexes.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 1, &indexes_count);
            }
            dispatch::set_buffer(encoder, 2, values.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 3, &log_leaves_count);
            }
            dispatch::set_buffer(encoder, 4, results.raw(), 0);
        },
    )
}

pub fn merkle_tree_cap(values: &MetalBuffer<Digest>, cap_size: usize) -> (usize, usize) {
    assert_ne!(cap_size, 0);
    assert!(cap_size.is_power_of_two());
    let log_cap_size = cap_size.trailing_zeros();
    let values_len = values.len();
    assert_ne!(values_len, 0);
    assert!(values_len.is_power_of_two());
    let log_values_len = values_len.trailing_zeros();
    assert!(log_values_len > log_cap_size);
    let offset = values_len - (1 << (log_cap_size + 1));
    (offset, cap_size)
}

/// Dispatch a kernel with threadgroup memory allocation.
/// Similar to `dispatch::dispatch_kernel` but also sets threadgroup memory length.
fn dispatch_with_threadgroup_memory(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    function_name: &str,
    config: &MetalLaunchConfig,
    threadgroup_memory_bytes: usize,
    encode_args: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
) -> MetalResult<()> {
    use crate::metal_runtime::error::MetalError;
    use crate::metal_runtime::pipeline;

    let pipeline = pipeline::get_pipeline(device, function_name)?;

    let encoder = cmd_buf
        .raw()
        .computeCommandEncoder()
        .ok_or_else(|| {
            MetalError::ResourceCreationFailed(
                "Failed to create compute command encoder".into(),
            )
        })?;

    encoder.setComputePipelineState(&pipeline);
    encode_args(&encoder);

    // Set threadgroup memory for index 0
    unsafe {
        encoder.setThreadgroupMemoryLength_atIndex(threadgroup_memory_bytes, 0);
    }

    encoder.dispatchThreadgroups_threadsPerThreadgroup(config.grid_dim, config.threadgroup_dim);
    encoder.endEncoding();

    Ok(())
}
