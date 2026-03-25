use crate::metal_runtime::dispatch::MetalLaunchConfig;

/// SIMD group size on Apple GPUs (equivalent to CUDA warp size).
pub const SIMD_GROUP_SIZE: usize = 32;
pub const LOG_SIMD_GROUP_SIZE: usize = 5;

/// Compute grid and threadgroup dimensions for a 1D kernel launch.
///
/// Analogous to `get_grid_block_dims_for_threads_count` in `gpu_prover/src/utils.rs`.
pub fn get_grid_threadgroup_dims(threads_per_group: usize, total_threads: u32) -> (u32, u32) {
    let threads_per_group = threads_per_group as u32;
    let threadgroups = (total_threads + threads_per_group - 1) / threads_per_group;
    (threadgroups, threads_per_group)
}

/// Create a 1D MetalLaunchConfig from thread count, analogous to CudaLaunchConfig::basic.
pub fn launch_config_1d(threads_per_group: usize, total_threads: u32) -> MetalLaunchConfig {
    let (groups, tpg) = get_grid_threadgroup_dims(threads_per_group, total_threads);
    MetalLaunchConfig::basic_1d(groups, tpg)
}

/// Ceiling division.
#[inline]
pub fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}
