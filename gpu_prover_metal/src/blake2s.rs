use crate::field::BaseField;
use crate::metal_runtime::dispatch::{self, MetalLaunchConfig};
use crate::metal_runtime::{MetalBuffer, MetalCommandBuffer, MetalResult};
use crate::utils::{get_grid_threadgroup_dims, SIMD_GROUP_SIZE};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice};

type BF = BaseField;

pub const STATE_SIZE: usize = 8;
pub const BLOCK_SIZE: usize = 16;

pub type Digest = [u32; STATE_SIZE];

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
    dispatch::dispatch_kernel(
        device,
        cmd_buf,
        "ab_blake2s_leaves_kernel",
        &config,
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

/// Build merkle tree leaves from a buffer with an explicit BF element count.
/// Used when the source buffer is a transmuted E4 buffer (E4 = 4 BF).
/// `leaf_count` is the number of leaves to hash (may differ from results.len()
/// when results is a full tree buffer with both leaves and nodes).
pub fn build_merkle_tree_leaves_raw(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    values_raw: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    values_bf_len: usize,
    results: &MetalBuffer<Digest>,
    leaf_count: usize,
    log_rows_per_hash: u32,
) -> MetalResult<()> {
    let count = leaf_count;
    assert_eq!(values_bf_len % (count << log_rows_per_hash as usize), 0);
    let cols_count = (values_bf_len / (count << log_rows_per_hash as usize)) as u32;
    let count_u32 = count as u32;
    let threads_per_group = SIMD_GROUP_SIZE * 4;
    let (threadgroups, tpg) = get_grid_threadgroup_dims(threads_per_group, count_u32);
    let config = MetalLaunchConfig::basic_1d(threadgroups, tpg);
    dispatch::dispatch_kernel(
        device,
        cmd_buf,
        "ab_blake2s_leaves_kernel",
        &config,
        |encoder| {
            dispatch::set_buffer(encoder, 0, values_raw, 0);
            dispatch::set_buffer(encoder, 1, results.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 2, &log_rows_per_hash);
                dispatch::set_bytes(encoder, 3, &cols_count);
                dispatch::set_bytes(encoder, 4, &count_u32);
            }
        },
    )
}

/// Tiled leaf hashing with shared-memory staging for coalesced memory access.
/// Threadgroup cooperatively loads column data into 32KB shared memory, then
/// each thread hashes its leaf from on-chip memory. Single dispatch, no state
/// buffer needed.
pub fn build_merkle_tree_leaves_tiled(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    values_raw: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    values_bf_len: usize,
    results: &MetalBuffer<Digest>,
    leaf_count: usize,
    log_rows_per_hash: u32,
) -> MetalResult<()> {
    const TILE_LEAVES: u32 = 256;
    const SHARED_MEM_BYTES: u32 = 32768; // 32KB

    let count = leaf_count as u32;
    assert_eq!(values_bf_len % (leaf_count << log_rows_per_hash as usize), 0);
    let total_cols = (values_bf_len / (leaf_count << log_rows_per_hash as usize)) as u32;

    // Small column counts: fall back to original kernel
    if total_cols <= 16 {
        return build_merkle_tree_leaves_raw(
            device, cmd_buf, values_raw, values_bf_len,
            results, leaf_count, log_rows_per_hash,
        );
    }

    let rows_per_leaf = 1u32 << log_rows_per_hash;
    // cols_per_tile = 32KB / (TILE_LEAVES * rows_per_leaf * 4 bytes)
    let cols_per_tile = SHARED_MEM_BYTES / (TILE_LEAVES * rows_per_leaf * 4);
    assert!(cols_per_tile >= 1, "rows_per_leaf too large for 32KB shared memory");

    let threadgroups = (count + TILE_LEAVES - 1) / TILE_LEAVES;
    let shared_bytes = (cols_per_tile * rows_per_leaf * TILE_LEAVES * 4) as usize;

    let config = MetalLaunchConfig::basic_1d(threadgroups, TILE_LEAVES);

    dispatch::dispatch_kernel(
        device, cmd_buf,
        "ab_blake2s_leaves_tiled_kernel",
        &config,
        |encoder| {
            dispatch::set_buffer(encoder, 0, values_raw, 0);
            dispatch::set_buffer(encoder, 1, results.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 2, &log_rows_per_hash);
                dispatch::set_bytes(encoder, 3, &total_cols);
                dispatch::set_bytes(encoder, 4, &count);
                dispatch::set_bytes(encoder, 5, &cols_per_tile);
                encoder.setThreadgroupMemoryLength_atIndex(shared_bytes, 0);
            }
        },
    )
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
    dispatch::dispatch_kernel(
        device,
        cmd_buf,
        "ab_blake2s_nodes_kernel",
        &config,
        |encoder| {
            dispatch::set_buffer(encoder, 0, values.raw(), 0);
            dispatch::set_buffer(encoder, 1, results.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 2, &count);
            }
        },
    )
}

/// Build merkle tree nodes with explicit byte offsets for input and output.
/// input_buffer[input_byte_offset..] contains `input_count` digests.
/// output_buffer[output_byte_offset..] receives the computed nodes.
pub fn build_merkle_tree_nodes_with_offset(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    input_buffer: &MetalBuffer<Digest>,
    input_byte_offset: usize,
    input_count: usize,
    output_buffer: &MetalBuffer<Digest>,
    output_byte_offset: usize,
    layers_count: u32,
) -> MetalResult<()> {
    if layers_count == 0 {
        return Ok(());
    }
    let digest_size = std::mem::size_of::<Digest>();
    let mut cur_input_offset = input_byte_offset;
    let mut cur_input_count = input_count;
    let mut cur_output_offset = output_byte_offset;

    for _layer in 0..layers_count {
        let output_count = cur_input_count / 2;
        let count = output_count as u32;
        let threads_per_group = SIMD_GROUP_SIZE * 4;
        let (threadgroups, tpg) = get_grid_threadgroup_dims(threads_per_group, count);
        let config = MetalLaunchConfig::basic_1d(threadgroups, tpg);

        dispatch::dispatch_kernel(
            device,
            cmd_buf,
            "ab_blake2s_nodes_kernel",
            &config,
            |encoder| {
                dispatch::set_buffer(encoder, 0, input_buffer.raw(), cur_input_offset);
                dispatch::set_buffer(encoder, 1, output_buffer.raw(), cur_output_offset);
                unsafe {
                    dispatch::set_bytes(encoder, 2, &count);
                }
            },
        )?;

        cur_input_offset = cur_output_offset;
        cur_input_count = output_count;
        cur_output_offset += output_count * digest_size;
    }
    Ok(())
}

pub fn build_merkle_tree_nodes(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    values: &MetalBuffer<Digest>,
    results: &MetalBuffer<Digest>,
    layers_count: u32,
) -> MetalResult<()> {
    if layers_count == 0 {
        return Ok(());
    }
    // Mirror CUDA's recursive approach iteratively:
    // Layer 0: input is `values` (len N), output is results[0..N/2]
    // Layer 1: input is results[0..N/2], output is results[N/2..N/2+N/4]
    // ...
    // We use buffer byte offsets to address sub-regions.
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(values_len, results_len);
    assert!(values_len.is_power_of_two());

    let digest_size = std::mem::size_of::<Digest>();
    let mut input_offset: usize = 0;
    let mut input_count = values_len; // number of input digests for this layer
    let mut output_offset: usize = 0;

    for layer in 0..layers_count {
        let output_count = input_count / 2;
        let count = output_count as u32;
        let threads_per_group = SIMD_GROUP_SIZE * 4;
        let (threadgroups, tpg) = get_grid_threadgroup_dims(threads_per_group, count);
        let config = MetalLaunchConfig::basic_1d(threadgroups, tpg);

        // For layer 0, input comes from `values`; subsequent layers from `results`.
        let input_buffer = if layer == 0 { values.raw() } else { results.raw() };
        let input_byte_offset = if layer == 0 { 0 } else { input_offset * digest_size };
        let output_byte_offset = output_offset * digest_size;

        dispatch::dispatch_kernel(
            device,
            cmd_buf,
            "ab_blake2s_nodes_kernel",
            &config,
            |encoder| {
                dispatch::set_buffer(encoder, 0, input_buffer, input_byte_offset);
                dispatch::set_buffer(encoder, 1, results.raw(), output_byte_offset);
                unsafe {
                    dispatch::set_bytes(encoder, 2, &count);
                }
            },
        )?;

        input_offset = output_offset;
        input_count = output_count;
        output_offset += output_count;
    }
    Ok(())
}

pub fn gather_rows(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    indexes: &MetalBuffer<u32>,
    bit_reverse_indexes: bool,
    log_rows_per_index: u32,
    values: &MetalBuffer<BF>,
    values_stride: u32,
    results: &MetalBuffer<BF>,
    results_stride: u32,
    cols_count: u32,
) -> MetalResult<()> {
    let indexes_count = indexes.len() as u32;
    let rows_per_index = 1u32 << log_rows_per_index;
    let indexes_per_group = if log_rows_per_index < (LOG_SIMD_GROUP_SIZE as u32) {
        (SIMD_GROUP_SIZE as u32) >> log_rows_per_index
    } else {
        1u32
    };
    let threadgroups_x = (indexes_count + indexes_per_group - 1) / indexes_per_group;
    let threadgroups_y = cols_count;
    let bit_reverse_u32: u32 = if bit_reverse_indexes { 1 } else { 0 };
    let log_rows_count = values_stride.trailing_zeros(); // approximate
    let config = MetalLaunchConfig::basic_2d(
        (threadgroups_x, threadgroups_y),
        (rows_per_index, indexes_per_group),
    );
    dispatch::dispatch_kernel(
        device,
        cmd_buf,
        "ab_blake2s_gather_rows_kernel",
        &config,
        |encoder| {
            dispatch::set_buffer(encoder, 0, indexes.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 1, &indexes_count);
                dispatch::set_bytes(encoder, 2, &bit_reverse_u32);
                dispatch::set_bytes(encoder, 3, &log_rows_count);
            }
            dispatch::set_buffer(encoder, 4, values.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 5, &values_stride);
            }
            dispatch::set_buffer(encoder, 6, results.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 7, &results_stride);
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
    assert_eq!(simd_size % STATE_SIZE as u32, 0);
    let indexes_per_group = simd_size / STATE_SIZE as u32;
    let threadgroups_x = (indexes_count + indexes_per_group - 1) / indexes_per_group;
    let config = MetalLaunchConfig::basic_2d(
        (threadgroups_x, layers_count),
        (STATE_SIZE as u32, indexes_per_group),
    );
    dispatch::dispatch_kernel(
        device,
        cmd_buf,
        "ab_blake2s_gather_merkle_paths_kernel",
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

pub fn blake2s_pow(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    seed: &MetalBuffer<u32>,
    bits_count: u32,
    start_nonce: u64,
    max_nonce: u64,
    result_lo: &MetalBuffer<u32>,
    result_hi: &MetalBuffer<u32>,
    num_groups: u32,
) -> MetalResult<()> {
    assert_eq!(seed.len(), STATE_SIZE);
    // Use a reasonable number of threads for PoW search
    let threads_per_group = (SIMD_GROUP_SIZE * 4) as u32;
    let config = MetalLaunchConfig::basic_1d(num_groups, threads_per_group);
    dispatch::dispatch_kernel(
        device,
        cmd_buf,
        "ab_blake2s_pow_kernel",
        &config,
        |encoder| {
            dispatch::set_buffer(encoder, 0, seed.raw(), 0);
            unsafe {
                dispatch::set_bytes(encoder, 1, &bits_count);
                dispatch::set_bytes(encoder, 2, &start_nonce);
                dispatch::set_bytes(encoder, 3, &max_nonce);
            }
            dispatch::set_buffer(encoder, 4, result_lo.raw(), 0);
            dispatch::set_buffer(encoder, 5, result_hi.raw(), 0);
        },
    )
}

pub fn merkle_tree_cap(values: &MetalBuffer<Digest>, log_tree_cap_size: u32) -> (usize, usize) {
    let values_len = values.len();
    assert_ne!(values_len, 0);
    assert!(values_len.is_power_of_two());
    let log_values_len = values_len.trailing_zeros();
    assert!(log_values_len > log_tree_cap_size);
    let offset = values_len - (1 << (log_tree_cap_size + 1));
    let count = 1 << log_tree_cap_size;
    (offset, count)
}

/// GPU query leaf gathering: replaces CPU triple-nested loop.
/// Gathers scattered leaf values from column-major GPU buffer into contiguous output.
/// Output layout: leafs[col * queries_count * rows_per_index + query * rows_per_index + row]
pub fn gather_query_leafs(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    values: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    query_indexes: &MetalBuffer<u32>,
    leafs: &MetalBuffer<BF>,
    domain_size: u32,
    columns_count: u32,
    queries_count: u32,
    log_rows_per_index: u32,
    bit_reverse: bool,
    log_domain_size: u32,
) -> MetalResult<()> {
    let rows_per_index = 1u32 << log_rows_per_index;
    let total = columns_count * queries_count * rows_per_index;
    if total == 0 {
        return Ok(());
    }
    let threads_per_group = 256u32;
    let threadgroups = (total + threads_per_group - 1) / threads_per_group;
    let config = MetalLaunchConfig::basic_1d(threadgroups, threads_per_group);
    let bit_reverse_flag: u32 = if bit_reverse { 1 } else { 0 };
    dispatch::dispatch_kernel(device, cmd_buf, "ab_gather_query_leafs_kernel", &config, |encoder| {
        dispatch::set_buffer(encoder, 0, values, 0);
        dispatch::set_buffer(encoder, 1, query_indexes.raw(), 0);
        dispatch::set_buffer(encoder, 2, leafs.raw(), 0);
        unsafe {
            dispatch::set_bytes(encoder, 3, &domain_size);
            dispatch::set_bytes(encoder, 4, &columns_count);
            dispatch::set_bytes(encoder, 5, &queries_count);
            dispatch::set_bytes(encoder, 6, &log_rows_per_index);
            dispatch::set_bytes(encoder, 7, &bit_reverse_flag);
            dispatch::set_bytes(encoder, 8, &log_domain_size);
        }
    })
}

use crate::utils::LOG_SIMD_GROUP_SIZE;

#[cfg(test)]
mod tests {
    use blake2s_u32::Blake2sState;
    use field::Field;
    use itertools::Itertools;
    use objc2_metal::MTLBuffer;
    use rand::Rng;

    use super::*;
    use crate::metal_runtime::{
        init_shader_library, system_default_device, MetalBuffer, MetalCommandQueue,
    };

    const USE_REDUCED_BLAKE2_ROUNDS: bool = true;

    fn div_ceil(a: usize, b: usize) -> usize {
        (a + b - 1) / b
    }

    fn verify_leaves(values: &[BF], results: &[Digest], log_rows_per_hash: u32) {
        let count = results.len();
        let values_len = values.len();
        assert_eq!(values_len % (count << log_rows_per_hash), 0);
        let cols_count = values_len / (count << log_rows_per_hash);
        let rows_count = 1 << log_rows_per_hash;
        for i in 0..count {
            let mut input = vec![];
            for col in 0..cols_count {
                let offset = (i << log_rows_per_hash) + col * rows_count * count;
                input.extend_from_slice(&values[offset..offset + rows_count]);
            }
            let blocks_count = div_ceil(input.len(), BLOCK_SIZE);
            let mut state = Blake2sState::new();
            let mut expected = Digest::default();
            for (i, chunk) in input.iter().chunks(BLOCK_SIZE).into_iter().enumerate() {
                let chunk = chunk.cloned().collect_vec();
                let block_len = chunk.len();
                let mut block = [0; BLOCK_SIZE];
                let chunk = chunk
                    .into_iter()
                    .map(|x| x.0)
                    .chain(std::iter::repeat(0))
                    .take(BLOCK_SIZE)
                    .collect_vec();
                block.copy_from_slice(&chunk);
                if i == blocks_count - 1 {
                    state.absorb_final_block::<USE_REDUCED_BLAKE2_ROUNDS>(
                        &block,
                        block_len,
                        &mut expected,
                    );
                } else {
                    state.absorb::<USE_REDUCED_BLAKE2_ROUNDS>(&block);
                }
            }
            let actual = results[i];
            assert_eq!(expected, actual);
        }
    }

    fn verify_nodes(values: &[Digest], results: &[Digest]) {
        let results_len = results.len();
        let values_len = values.len();
        assert_eq!(values_len, results_len * 2);
        values
            .chunks_exact(2)
            .zip(results)
            .for_each(|(input, &actual)| {
                let state = input
                    .iter()
                    .flat_map(|&x| x.into_iter())
                    .collect_vec()
                    .try_into()
                    .unwrap();
                let mut expected = Digest::default();
                Blake2sState::compress_two_to_one::<USE_REDUCED_BLAKE2_ROUNDS>(
                    &state,
                    &mut expected,
                );
                assert_eq!(expected, actual);
            });
    }

    fn random_digest() -> Digest {
        let mut rng = rand::rng();
        let mut result = Digest::default();
        result.fill_with(|| rng.random());
        result
    }

    #[test]
    fn test_blake2s_leaves() {
        const LOG_N: usize = 10;
        const N: usize = 1 << LOG_N;
        const VALUES_PER_ROW: usize = 125;
        const LOG_ROWS_PER_HASH: u32 = 1;
        let mut values_host = vec![BF::ZERO; (N * VALUES_PER_ROW) << LOG_ROWS_PER_HASH];
        let mut rng = rand::rng();
        values_host.fill_with(|| BF::from_nonreduced_u32(rng.random()));

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let values_buf = MetalBuffer::from_slice(device, &values_host).unwrap();
        let results_buf = MetalBuffer::<Digest>::alloc(device, N).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        launch_leaves_kernel(device, &cmd_buf, &values_buf, &results_buf, LOG_ROWS_PER_HASH)
            .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut results_host = vec![Digest::default(); N];
        unsafe {
            results_buf.copy_to_slice(&mut results_host);
        }
        verify_leaves(&values_host, &results_host, LOG_ROWS_PER_HASH);
    }

    #[test]
    fn test_blake2s_nodes() {
        const LOG_N: usize = 10;
        const N: usize = 1 << LOG_N;
        let mut values_host = vec![Digest::default(); N * 2];
        values_host.fill_with(random_digest);

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let values_buf = MetalBuffer::from_slice(device, &values_host).unwrap();
        let results_buf = MetalBuffer::<Digest>::alloc(device, N).unwrap();

        let cmd_buf = queue.new_command_buffer().unwrap();
        launch_nodes_kernel(device, &cmd_buf, &values_buf, &results_buf).unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut results_host = vec![Digest::default(); N];
        unsafe {
            results_buf.copy_to_slice(&mut results_host);
        }
        verify_nodes(&values_host, &results_host);
    }

    fn verify_tree(values: &[Digest], results: &[Digest], layers_count: u32) {
        assert_eq!(values.len(), results.len());
        if layers_count == 0 {
            assert!(results.iter().all(|x| x.iter().all(|&x| x == 0)));
        } else {
            let (nodes, nodes_remaining) = results.split_at(results.len() >> 1);
            verify_nodes(values, nodes);
            verify_tree(nodes, nodes_remaining, layers_count - 1);
        }
    }

    fn run_merkle_tree_test(log_n: usize) {
        const VALUES_PER_ROW: usize = 125;
        const LOG_ROWS_PER_HASH: u32 = 1;
        let n = 1 << log_n;
        let node_layers_count = log_n as u32;
        let mut values_host = vec![BF::ZERO; (n * VALUES_PER_ROW) << LOG_ROWS_PER_HASH];
        let mut rng = rand::rng();
        values_host.fill_with(|| BF::from_nonreduced_u32(rng.random()));

        let device = system_default_device().unwrap();
        init_shader_library(device).unwrap();
        let queue = MetalCommandQueue::new(device).unwrap();

        let values_buf = MetalBuffer::from_slice(device, &values_host).unwrap();
        let leaves_buf = MetalBuffer::<Digest>::alloc(device, n).unwrap();
        let nodes_buf = MetalBuffer::<Digest>::alloc(device, n).unwrap();
        // Zero-init nodes
        unsafe {
            let slice = std::slice::from_raw_parts_mut(
                nodes_buf.raw().contents().cast::<u8>().as_ptr(),
                nodes_buf.byte_len(),
            );
            slice.fill(0);
        }

        let cmd_buf = queue.new_command_buffer().unwrap();
        build_merkle_tree_leaves(device, &cmd_buf, &values_buf, &leaves_buf, LOG_ROWS_PER_HASH)
            .unwrap();
        build_merkle_tree_nodes(device, &cmd_buf, &leaves_buf, &nodes_buf, node_layers_count)
            .unwrap();
        cmd_buf.commit_and_wait();
        cmd_buf.status().unwrap();

        let mut leaves_host = vec![Digest::default(); n];
        let mut nodes_host = vec![Digest::default(); n];
        unsafe {
            leaves_buf.copy_to_slice(&mut leaves_host);
            nodes_buf.copy_to_slice(&mut nodes_host);
        }
        verify_leaves(&values_host, &leaves_host, LOG_ROWS_PER_HASH);
        verify_tree(&leaves_host, &nodes_host, node_layers_count);
    }

    #[test]
    fn test_merkle_tree() {
        run_merkle_tree_test(8);
    }
}
