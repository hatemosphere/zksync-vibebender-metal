#include "field.metal"

using namespace airbender::field;

namespace airbender {
namespace monolith {

typedef base_field bf;

constant constexpr unsigned CAPACITY = 8;
constant constexpr unsigned RATE = 8;
constant constexpr unsigned WIDTH = CAPACITY + RATE;
constant constexpr unsigned NUM_ROUNDS = 6;
constant constexpr unsigned NUM_FULL_ROUNDS = NUM_ROUNDS - 1;
constant constexpr unsigned NUM_BARS = 8;

constant constexpr uint32_t ROUND_CONSTANTS[NUM_FULL_ROUNDS][WIDTH] = {
    {1821280327, 1805192324, 127749067, 534494027, 504066389, 661859220,
     1964605566, 11087311, 1178584041, 412585466, 2078905810, 549234502,
     1181028407, 363220519, 1649192353, 895839514},
    {939676630, 132824540, 1081150345, 1901266162, 1248854474, 722216947,
     711899879, 991065584, 872971327, 1747874412, 889258434, 857014393,
     1145792277, 329607215, 1069482641, 1809464251},
    {1792923486, 1071073386, 2086334655, 615259270, 1680936759, 2069228098,
     679754665, 598972355, 1448263353, 2102254560, 1676515281, 1529495635,
     981915006, 436108429, 1959227325, 1710180674},
    {814766386, 746021429, 758709057, 1777861169, 1875425297, 1630916709,
     180204592, 1301124329, 307222363, 297236795, 866482358, 1784330946,
     1841790988, 1855089478, 2122902104, 1522878966},
    {1132611924, 1823267038, 539457094, 934064219, 561891167, 1325624939,
     1683493283, 1582152536, 851185378, 1187215684, 1520269176, 801897118,
     741765053, 1300119213, 1960664069, 1633755961},
};

constant constexpr unsigned MDS_MATRIX_INDEXES[WIDTH + 1] = {
    0, 4, 10, 12, 7, 14, 8, 13, 11, 1, 6, 15, 2, 3, 5, 9, 15};

constant constexpr unsigned MDS_MATRIX_SHIFTS[WIDTH + 1] = {
    4, 2, 4, 2, 4, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0};

DEVICE_FORCEINLINE uint8_t rotl_8(const uint8_t x, const unsigned shift) {
  return x << shift | x >> (8 - shift);
}

DEVICE_FORCEINLINE uint8_t rotl_7(const uint8_t x, const unsigned shift) {
  return (x << shift | x >> (7 - shift)) & 0x7F;
}

DEVICE_FORCEINLINE uint8_t s_box_8(const uint8_t x) {
  return rotl_8(x ^ (~rotl_8(x, 1) & rotl_8(x, 2) & rotl_8(x, 3)), 1);
}

DEVICE_FORCEINLINE uint8_t s_box_7(const uint8_t x) {
  return rotl_7(x ^ (~rotl_7(x, 1) & rotl_7(x, 2)), 1);
}

DEVICE_FORCEINLINE void initialize_lookup(threadgroup uint8_t *bar_lookup,
                                          const unsigned tid,
                                          const unsigned block_size) {
  for (unsigned i = tid; i < 1u << 8; i += block_size)
    bar_lookup[i] = s_box_8(i);
  for (unsigned i = tid; i < 1u << 7; i += block_size)
    bar_lookup[(1u << 8) + i] = s_box_7(i);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

DEVICE_FORCEINLINE uint32_t bar(uint32_t limb,
                                threadgroup const uint8_t *bar_lookup) {
  uint32_t result;
  thread uint8_t *rb = reinterpret_cast<thread uint8_t *>(&result);
  thread const uint8_t *lb = reinterpret_cast<thread const uint8_t *>(&limb);
  rb[0] = bar_lookup[lb[0]];
  rb[1] = bar_lookup[lb[1]];
  rb[2] = bar_lookup[lb[2]];
  rb[3] = bar_lookup[(1u << 8) + lb[3]];
  return result;
}

DEVICE_FORCEINLINE void bars_st(thread bf state[WIDTH],
                                threadgroup const uint8_t *bar_lookup) {
  #pragma unroll
  for (unsigned i = 0; i < NUM_BARS; i++)
    state[i] = {bar(state[i].limb, bar_lookup)};
}

DEVICE_FORCEINLINE void bricks_st(thread bf state[WIDTH]) {
  #pragma unroll
  for (unsigned i = WIDTH - 1; i > 0; i--)
    state[i] = bf::add(state[i], bf::sqr(state[i - 1]));
}

template <unsigned ROUND>
DEVICE_FORCEINLINE void concrete_shl_st(thread bf state[WIDTH]) {
  bf result[WIDTH];
  #pragma unroll
  for (unsigned row = 0; row < WIDTH; row++) {
    uint64_t acc = 0;
    #pragma unroll
    for (unsigned i = 0; i < WIDTH + 1; i++) {
      const unsigned index = MDS_MATRIX_INDEXES[i];
      const unsigned col = (index + row) % WIDTH;
      const uint32_t value = state[col].limb;
      acc = i ? acc + value : value;
      if (MDS_MATRIX_SHIFTS[i])
        acc <<= MDS_MATRIX_SHIFTS[i];
    }
    acc <<= 2; // multiply by 4 for compatibility with the FFT variant
    if (ROUND != 0 && ROUND < NUM_ROUNDS)
      acc += ROUND_CONSTANTS[ROUND - 1][row];
    result[row] = bf::from_u62_max_minus_one(acc);
  }
  #pragma unroll
  for (unsigned i = 0; i < WIDTH; i++)
    state[i] = result[i];
}

template <unsigned ROUND>
DEVICE_FORCEINLINE void concrete_st(thread bf state[WIDTH]) {
  concrete_shl_st<ROUND>(state);
}

template <unsigned ROUND>
DEVICE_FORCEINLINE void round_st(thread bf state[WIDTH],
                                 threadgroup const uint8_t *bar_lookup) {
  if (ROUND != 0) {
    bars_st(state, bar_lookup);
    bricks_st(state);
  }
  concrete_st<ROUND>(state);
}

DEVICE_FORCEINLINE void permutation_st(thread bf state[WIDTH],
                                       threadgroup const uint8_t *bar_lookup) {
  round_st<0>(state, bar_lookup);
  round_st<1>(state, bar_lookup);
  round_st<2>(state, bar_lookup);
  round_st<3>(state, bar_lookup);
  round_st<4>(state, bar_lookup);
  round_st<5>(state, bar_lookup);
  round_st<6>(state, bar_lookup);
}

} // namespace monolith
} // namespace airbender

using namespace airbender::monolith;

kernel void ab_monolith_leaves_kernel(
    const device bf *values [[buffer(0)]],
    device bf *results [[buffer(1)]],
    constant uint &log_rows_count [[buffer(2)]],
    constant uint &cols_count [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup uint8_t *tg_mem [[threadgroup(0)]]) {
  threadgroup uint8_t *bar_lookup = tg_mem;
  initialize_lookup(bar_lookup, lid, tg_size);
  if (gid >= count)
    return;
  const device bf *my_values = values + (gid << log_rows_count);
  device bf *my_results = results + gid * CAPACITY;
  const unsigned row_mask = (1u << log_rows_count) - 1;
  bf state[WIDTH];
  unsigned offset = 0;
  #pragma unroll
  for (unsigned i = 0; i < WIDTH; i++, offset++) {
    const unsigned row = offset & row_mask;
    const unsigned col = offset >> log_rows_count;
    state[i] = col < cols_count
                   ? my_values[row + ((col * count) << log_rows_count)]
                   : bf::zero();
  }
  permutation_st(state, bar_lookup);
  while (offset < (cols_count << log_rows_count)) {
    #pragma unroll
    for (unsigned i = 0; i < RATE; i++, offset++) {
      const unsigned row = offset & row_mask;
      const unsigned col = offset >> log_rows_count;
      state[i + CAPACITY] =
          col < cols_count
              ? my_values[row + ((col * count) << log_rows_count)]
              : bf::zero();
    }
    permutation_st(state, bar_lookup);
  }
  #pragma unroll
  for (unsigned i = 0; i < CAPACITY; i++)
    my_results[i] = state[i];
}

kernel void ab_monolith_nodes_kernel(
    const device bf *values [[buffer(0)]],
    device bf *results [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup uint8_t *tg_mem [[threadgroup(0)]]) {
  threadgroup uint8_t *bar_lookup = tg_mem;
  initialize_lookup(bar_lookup, lid, tg_size);
  if (gid >= count)
    return;
  const device bf *my_values = values + gid * WIDTH;
  device bf *my_results = results + gid * CAPACITY;
  bf state[WIDTH];
  #pragma unroll
  for (unsigned i = 0; i < WIDTH; i++)
    state[i] = my_values[i];
  permutation_st(state, bar_lookup);
  #pragma unroll
  for (unsigned i = 0; i < CAPACITY; i++)
    my_results[i] = state[i];
}

kernel void ab_monolith_gather_rows_kernel(
    const device uint *indexes [[buffer(0)]],
    constant uint &indexes_count [[buffer(1)]],
    const device bf *values [[buffer(2)]],
    constant uint &values_stride [[buffer(3)]],
    device bf *results [[buffer(4)]],
    constant uint &results_stride [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]]) {
  const unsigned idx = tid.y + gid.x * tg_size.y;
  if (idx >= indexes_count)
    return;
  const unsigned index = indexes[idx];
  const unsigned src_row = index * tg_size.x + tid.x;
  const unsigned dst_row = idx * tg_size.x + tid.x;
  const unsigned col = gid.y;
  results[dst_row + col * results_stride] = values[src_row + col * values_stride];
}

kernel void ab_monolith_gather_merkle_paths_kernel(
    const device uint *indexes [[buffer(0)]],
    constant uint &indexes_count [[buffer(1)]],
    const device bf *values [[buffer(2)]],
    constant uint &log_leaves_count [[buffer(3)]],
    device bf *results [[buffer(4)]],
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
      CAPACITY;
  const unsigned hash_offset = ((leaf_index >> layer_index) ^ 1) * CAPACITY;
  const unsigned element_offset = tid.x;
  const unsigned src_index = layer_offset + hash_offset + element_offset;
  const unsigned dst_index = layer_index * indexes_count * CAPACITY +
                             idx * CAPACITY + element_offset;
  results[dst_index] = values[src_index];
}
