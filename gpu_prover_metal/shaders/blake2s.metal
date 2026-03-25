#include "common.metal"
#include "field.metal"

using namespace airbender::field;

namespace airbender {
namespace blake2s {

typedef uint32_t u32;
typedef uint64_t u64;
typedef base_field bf;

#define ROTR32(x, y) (((x) >> (y)) ^ ((x) << (32 - (y))))

#define G(a, b, c, d, x, y)                                                   \
  v[a] = v[a] + v[b] + (x);                                                   \
  v[d] = ROTR32(v[d] ^ v[a], 16);                                             \
  v[c] = v[c] + v[d];                                                         \
  v[b] = ROTR32(v[b] ^ v[c], 12);                                             \
  v[a] = v[a] + v[b] + (y);                                                   \
  v[d] = ROTR32(v[d] ^ v[a], 8);                                              \
  v[c] = v[c] + v[d];                                                         \
  v[b] = ROTR32(v[b] ^ v[c], 7);

constant constexpr bool USE_REDUCED_ROUNDS = true;
constant constexpr unsigned FULL_ROUNDS = 10;
constant constexpr unsigned REDUCED_ROUNDS = 7;
constant constexpr unsigned ROUNDS = USE_REDUCED_ROUNDS ? REDUCED_ROUNDS : FULL_ROUNDS;
constant constexpr unsigned STATE_SIZE = 8;
constant constexpr unsigned BLOCK_SIZE = 16;
constant constexpr u32 IV_0_TWIST = 0x01010000 ^ 32;

constant constexpr u32 IV[STATE_SIZE] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

constant constexpr unsigned SIGMAS[10][BLOCK_SIZE] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}
};

DEVICE_FORCEINLINE void initialize(thread u32 state[STATE_SIZE]) {
  #pragma unroll
  for (unsigned i = 0; i < STATE_SIZE; i++)
    state[i] = IV[i];
  state[0] ^= IV_0_TWIST;
}

template <bool IS_FINAL_BLOCK>
DEVICE_FORCEINLINE void compress(thread u32 state[STATE_SIZE], thread u32 &t,
                                 const thread u32 m[BLOCK_SIZE],
                                 const unsigned block_size) {
  u32 v[BLOCK_SIZE];
  #pragma unroll
  for (unsigned i = 0; i < STATE_SIZE; i++) {
    v[i] = state[i];
    v[i + STATE_SIZE] = IV[i];
  }
  t += (IS_FINAL_BLOCK ? block_size : BLOCK_SIZE) * sizeof(u32);
  v[12] ^= t;
  if (IS_FINAL_BLOCK)
    v[14] ^= 0xffffffff;
  #pragma unroll
  for (unsigned i = 0; i < ROUNDS; i++) {
    constant unsigned *s = SIGMAS[i];
    G(0, 4, 8, 12, m[s[0]], m[s[1]])
    G(1, 5, 9, 13, m[s[2]], m[s[3]])
    G(2, 6, 10, 14, m[s[4]], m[s[5]])
    G(3, 7, 11, 15, m[s[6]], m[s[7]])
    G(0, 5, 10, 15, m[s[8]], m[s[9]])
    G(1, 6, 11, 12, m[s[10]], m[s[11]])
    G(2, 7, 8, 13, m[s[12]], m[s[13]])
    G(3, 4, 9, 14, m[s[14]], m[s[15]])
  }
  #pragma unroll
  for (unsigned i = 0; i < STATE_SIZE; ++i)
    state[i] ^= v[i] ^ v[i + STATE_SIZE];
}

} // namespace blake2s
} // namespace airbender

using namespace airbender::blake2s;

kernel void ab_blake2s_leaves_kernel(
    const device bf *values [[buffer(0)]],
    device u32 *results [[buffer(1)]],
    constant uint &log_rows_count [[buffer(2)]],
    constant uint &cols_count [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count)
    return;
  const device bf *my_values = values + (gid << log_rows_count);
  device u32 *my_results = results + gid * STATE_SIZE;
  const unsigned row_mask = (1u << log_rows_count) - 1;
  u32 state[STATE_SIZE];
  u32 block[BLOCK_SIZE];
  initialize(state);
  u32 t = 0;
  const unsigned values_count = cols_count << log_rows_count;
  unsigned offset = 0;
  while (offset < values_count) {
    const unsigned remaining = values_count - offset;
    const bool is_final_block = remaining <= BLOCK_SIZE;
    for (unsigned i = 0; i < BLOCK_SIZE; i++, offset++) {
      const unsigned row = offset & row_mask;
      const unsigned col = offset >> log_rows_count;
      block[i] = col < cols_count
                     ? bf::into_canonical_u32(
                           my_values[row + (col * count << log_rows_count)])
                     : 0;
    }
    if (is_final_block)
      compress<true>(state, t, block, remaining);
    else
      compress<false>(state, t, block, BLOCK_SIZE);
  }
  #pragma unroll
  for (unsigned i = 0; i < STATE_SIZE; i++)
    my_results[i] = state[i];
}

/// Tiled leaf hashing: threadgroup cooperatively loads column data into shared
/// memory with coalesced reads, then each thread hashes its leaf from shared memory.
///
/// Original kernel: each thread reads from ALL columns with domain_size stride
///   → 3% bandwidth utilization (16MB stride defeats cache).
/// Tiled kernel: threadgroup reads one column at a time, 256 consecutive values
///   → perfectly coalesced, full cache line utilization.
///
/// Shared memory: TILE_LEAVES × cols_per_tile u32s. For 256 threads:
///   log_rows=0: 256 leaves × 32 cols = 32KB
///   log_rows=1: 256 leaves × 16 cols = 32KB (×2 rows per leaf)
///   log_rows=2: 256 leaves × 8 cols = 32KB  (×4 rows per leaf)
constant constexpr unsigned TILE_LEAVES = 256;

kernel void ab_blake2s_leaves_tiled_kernel(
    const device bf *values [[buffer(0)]],
    device u32 *results [[buffer(1)]],
    constant uint &log_rows_count [[buffer(2)]],
    constant uint &cols_count [[buffer(3)]],
    constant uint &count [[buffer(4)]],      // leaf_count
    constant uint &cols_per_tile [[buffer(5)]],
    threadgroup u32 *shared_data [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]) {

  const unsigned rows_per_leaf = 1u << log_rows_count;
  const unsigned domain_size = count * rows_per_leaf;
  const unsigned base_leaf = gid * TILE_LEAVES;
  const unsigned my_leaf = base_leaf + tid;
  const unsigned total_values = cols_count * rows_per_leaf;

  // Initialize Blake2s state
  u32 state[STATE_SIZE];
  u32 block[BLOCK_SIZE];
  initialize(state);
  u32 t = 0;
  unsigned block_idx = 0;
  unsigned values_consumed = 0;

  // Process columns in tiles
  for (unsigned col_base = 0; col_base < cols_count; col_base += cols_per_tile) {
    const unsigned cols_this_tile = min(cols_per_tile, cols_count - col_base);
    const unsigned tile_elems = cols_this_tile * rows_per_leaf;

    // Cooperative load: all threads read coalesced from device memory into shared
    for (unsigned c = 0; c < cols_this_tile; c++) {
      const unsigned col = col_base + c;
      for (unsigned r = 0; r < rows_per_leaf; r++) {
        const unsigned shared_idx = (c * rows_per_leaf + r) * TILE_LEAVES + tid;
        const unsigned device_idx = (base_leaf + tid) * rows_per_leaf + r + col * domain_size;
        // All 256 threads read consecutive addresses within one column → coalesced
        shared_data[shared_idx] = (my_leaf < count)
            ? bf::into_canonical_u32(values[device_idx])
            : 0;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each thread hashes its leaf's data from shared memory (zero device reads)
    if (my_leaf < count) {
      for (unsigned c = 0; c < cols_this_tile; c++) {
        for (unsigned r = 0; r < rows_per_leaf; r++) {
          block[block_idx++] = shared_data[(c * rows_per_leaf + r) * TILE_LEAVES + tid];
          values_consumed++;
          if (block_idx == BLOCK_SIZE) {
            if (values_consumed == total_values) {
              compress<true>(state, t, block, BLOCK_SIZE);
            } else {
              compress<false>(state, t, block, BLOCK_SIZE);
            }
            block_idx = 0;
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Partial final block
  if (my_leaf < count && block_idx > 0) {
    for (unsigned i = block_idx; i < BLOCK_SIZE; i++)
      block[i] = 0;
    compress<true>(state, t, block, block_idx);
  }

  // Write final digest
  if (my_leaf < count) {
    device u32 *my_results = results + my_leaf * STATE_SIZE;
    #pragma unroll
    for (unsigned i = 0; i < STATE_SIZE; i++)
      my_results[i] = state[i];
  }
}

kernel void ab_blake2s_nodes_kernel(
    const device u32 *values [[buffer(0)]],
    device u32 *results [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count)
    return;
  const device u32 *my_values = values + gid * BLOCK_SIZE;
  device u32 *my_results = results + gid * STATE_SIZE;
  u32 state[STATE_SIZE];
  u32 block[BLOCK_SIZE];
  initialize(state);
  u32 t = 0;
  #pragma unroll
  for (unsigned i = 0; i < BLOCK_SIZE; i++)
    block[i] = my_values[i];
  compress<true>(state, t, block, BLOCK_SIZE);
  #pragma unroll
  for (unsigned i = 0; i < STATE_SIZE; i++)
    my_results[i] = state[i];
}

kernel void ab_blake2s_gather_rows_kernel(
    const device uint *indexes [[buffer(0)]],
    constant uint &indexes_count [[buffer(1)]],
    constant uint &bit_reverse_indexes_u32 [[buffer(2)]],
    constant uint &log_rows_count [[buffer(3)]],
    const device bf *values [[buffer(4)]],
    constant uint &values_stride [[buffer(5)]],
    device bf *results [[buffer(6)]],
    constant uint &results_stride [[buffer(7)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]]) {
  const unsigned idx = tid.y + gid.x * tg_size.y;
  if (idx >= indexes_count)
    return;
  const unsigned i = indexes[idx];
  const bool bit_rev = (bit_reverse_indexes_u32 != 0);
  const unsigned index = bit_rev ? (reverse_bits(i) >> (32 - log_rows_count)) : i;
  const unsigned src_row = index * tg_size.x + tid.x;
  const unsigned dst_row = idx * tg_size.x + tid.x;
  const unsigned col = gid.y;
  const bf value = values[src_row + col * values_stride];
  const bf result = {bf::into_canonical_u32(value)};
  results[dst_row + col * results_stride] = result;
}

kernel void ab_blake2s_gather_merkle_paths_kernel(
    const device uint *indexes [[buffer(0)]],
    constant uint &indexes_count [[buffer(1)]],
    const device u32 *values [[buffer(2)]],
    constant uint &log_leaves_count [[buffer(3)]],
    device u32 *results [[buffer(4)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]]) {
  const unsigned idx = tid.y + gid.x * tg_size.y;
  if (idx >= indexes_count)
    return;
  const unsigned leaf_index = indexes[idx];
  const unsigned layer_index = gid.y;
  const unsigned layer_offset =
      ((1u << (log_leaves_count + 1)) -
       (1u << (log_leaves_count + 1 - layer_index))) *
      STATE_SIZE;
  const unsigned hash_offset = ((leaf_index >> layer_index) ^ 1) * STATE_SIZE;
  const unsigned element_offset = tid.x;
  const unsigned src_index = layer_offset + hash_offset + element_offset;
  const unsigned dst_index = layer_index * indexes_count * STATE_SIZE +
                             idx * STATE_SIZE + element_offset;
  results[dst_index] = values[src_index];
}

// PoW kernel uses two u32 atomics to simulate a u64 CAS.
// result[0] = low 32 bits, result[1] = high 32 bits.
// We initialize both to 0xFFFFFFFF (representing u64::MAX).
// Debug kernel: full PoW pipeline dump
kernel void ab_blake2s_pow_debug_kernel(
    const device u64 *seed [[buffer(0)]],
    constant u32 &bits_count [[buffer(1)]],
    constant u64 &start_nonce [[buffer(2)]],
    constant u64 &max_nonce [[buffer(3)]],
    device uint *result [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid != 0) return;
  // Dump received parameters
  result[0] = bits_count;
  result[1] = static_cast<u32>(start_nonce);
  result[2] = static_cast<u32>(start_nonce >> 32);
  result[3] = static_cast<u32>(max_nonce);
  result[4] = static_cast<u32>(max_nonce >> 32);
  // Compute mask
  const uint32_t digest_mask = 0xffffffff << (32 - bits_count);
  result[5] = digest_mask;
  // Compute hash for nonce=start_nonce
  u32 m_u32[BLOCK_SIZE] = {};
  thread u64 *m_u64 = reinterpret_cast<thread u64 *>(m_u32);
  #pragma unroll
  for (unsigned i = 0; i < 4; i++)
    m_u64[i] = seed[i];
  m_u64[STATE_SIZE / 2] = start_nonce;
  u32 state[STATE_SIZE];
  initialize(state);
  u32 t = 0;
  compress<true>(state, t, m_u32, STATE_SIZE + 2);
  result[6] = state[0];
  result[7] = state[0] & digest_mask;
  result[8] = static_cast<u32>(!(state[0] & digest_mask));
}

kernel void ab_blake2s_pow_kernel(
    const device u64 *seed [[buffer(0)]],
    constant u32 &bits_count [[buffer(1)]],
    constant u64 &start_nonce [[buffer(2)]],
    constant u64 &max_nonce [[buffer(3)]],
    device uint *result_lo [[buffer(4)]],
    device uint *result_hi [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
  // This file is compiled at -O0 due to a Metal compiler bug that
  // miscompiles compress() inside a for loop at -O1 or higher.
  const uint32_t digest_mask = (bits_count >= 32) ? 0xffffffff : (0xffffffff << (32 - bits_count));
  u32 m_u32[BLOCK_SIZE] = {};
  thread u64 *m_u64 = reinterpret_cast<thread u64 *>(m_u32);
  #pragma unroll
  for (unsigned i = 0; i < 4; i++)
    m_u64[i] = seed[i];
  const unsigned stride = grid_size;
  for (uint64_t nonce = start_nonce + gid; nonce < max_nonce; nonce += stride) {
    m_u64[STATE_SIZE / 2] = nonce;
    u32 state[STATE_SIZE];
    initialize(state);
    u32 t = 0;
    compress<true>(state, t, m_u32, STATE_SIZE + 2);
    if (!(state[0] & digest_mask)) {
      result_lo[0] = static_cast<u32>(nonce);
      result_hi[0] = static_cast<u32>(nonce >> 32);
      return;
    }
    if (result_lo[0] != 0xFFFFFFFF)
      return;
  }
}

// --- blake2s_leaves_sequential: hash pre-transposed leaf-major data ---
// Input is leaf-major: each leaf's data is contiguous in memory.
// This is ~10x faster than the strided column-major kernel due to coalesced reads.

kernel void ab_blake2s_leaves_sequential_kernel(
    const device u32 *values [[buffer(0)]],
    device u32 *results [[buffer(1)]],
    constant uint &elems_per_leaf [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  const device u32 *my_values = values + gid * elems_per_leaf;
  device u32 *my_results = results + gid * STATE_SIZE;
  u32 state[STATE_SIZE];
  u32 block[BLOCK_SIZE];
  initialize(state);
  u32 t = 0;
  uint offset = 0;
  while (offset < elems_per_leaf) {
    const uint remaining = elems_per_leaf - offset;
    const bool is_final_block = remaining <= BLOCK_SIZE;
    #pragma unroll
    for (uint i = 0; i < BLOCK_SIZE; i++, offset++) {
      // Canonicalize: ORDER (2^31-1) → 0
      u32 raw = (offset < elems_per_leaf) ? my_values[offset] : 0;
      block[i] = (raw == bf::ORDER) ? 0u : raw;
    }
    if (is_final_block)
      compress<true>(state, t, block, remaining);
    else
      compress<false>(state, t, block, BLOCK_SIZE);
  }
  #pragma unroll
  for (uint i = 0; i < STATE_SIZE; i++)
    my_results[i] = state[i];
}

// --- GPU query leaf gathering ---
// Replaces the CPU triple-nested loop: for col, for query, for row.
// Each thread gathers one (col, query, row) element from column-major GPU buffer.
// Output layout: leafs[col * queries_count * rows_per_index + query * rows_per_index + row]

kernel void ab_gather_query_leafs_kernel(
    const device uint *values [[buffer(0)]],
    const device uint *query_indexes [[buffer(1)]],
    device uint *leafs [[buffer(2)]],
    constant uint &domain_size [[buffer(3)]],
    constant uint &columns_count [[buffer(4)]],
    constant uint &queries_count [[buffer(5)]],
    constant uint &log_rows_per_index [[buffer(6)]],
    constant uint &bit_reverse_flag [[buffer(7)]],
    constant uint &log_domain_size [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint rows_per_index = 1u << log_rows_per_index;
    const uint total = columns_count * queries_count * rows_per_index;
    if (gid >= total) return;

    const uint row = gid % rows_per_index;
    const uint query = (gid / rows_per_index) % queries_count;
    const uint col = gid / (rows_per_index * queries_count);

    const uint tree_idx = query_indexes[query];
    const uint base_idx = bit_reverse_flag
        ? (reverse_bits(tree_idx) >> (32 - log_domain_size))
        : tree_idx;
    const uint src_idx = col * domain_size + (base_idx << log_rows_per_index) + row;

    const uint val = values[src_idx];
    leafs[gid] = (val == bf::ORDER) ? 0u : val;
}
