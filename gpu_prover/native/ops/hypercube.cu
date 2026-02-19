#include "../field.cuh"
#include "../memory.cuh"

using namespace ::airbender::field;
using namespace ::airbender::memory;

namespace airbender::ops::hypercube {

DEVICE_FORCEINLINE unsigned swizzle_shared_lane(const unsigned idx) {
  // Mix warp-id bits into lane bits to reduce systematic bank aliasing.
  return idx ^ (idx >> 5);
}

DEVICE_FORCEINLINE unsigned transpose64_conflict_free_idx(const unsigned row6, const unsigned col6) {
  // 64x64 bijection tailored for the initial-12 kernel transpose access
  // patterns. Low 5 bits are selected to give distinct banks per lane for both
  // (lo,hi) and (hi,lo) warp patterns used by this kernel.
  const unsigned row4 = row6 >> 2;
  const unsigned col4 = col6 >> 2;
  const unsigned row2 = row6 & 3u;
  const unsigned col2 = col6 & 3u;

  // bank[3:0] = row4 ^ col4, bank[4] = row4[0]
  const unsigned bank = ((row4 ^ col4) & 15u) | ((row4 & 1u) << 4);
  // major keeps mapping bijective together with bank bits.
  const unsigned major = ((row4 >> 1) << 4) | (row2 << 2) | col2;
  return (major << 5) | bank;
}

DEVICE_FORCEINLINE unsigned select_u32(const unsigned a, const unsigned b, const unsigned mask) {
  return (a & ~mask) | (b & mask);
}

template <ld_modifier MODIFIER>
DEVICE_FORCEINLINE uint4 load_u4_mod(const bf *__restrict__ ptr, const unsigned row_base) {
  return load<uint4, MODIFIER>(reinterpret_cast<const uint4 *>(ptr + row_base));
}

template <st_modifier MODIFIER>
DEVICE_FORCEINLINE void store_u4_mod(bf *__restrict__ ptr, const unsigned row_base, const uint4 packed) {
  store<uint4, MODIFIER>(reinterpret_cast<uint4 *>(ptr + row_base), packed);
}

template <ld_modifier MODIFIER>
DEVICE_FORCEINLINE uint4 load_u4(const bf *__restrict__ ptr, const unsigned row_base) {
  return load<uint4, MODIFIER>(reinterpret_cast<const uint4 *>(ptr + row_base));
}

DEVICE_FORCEINLINE uint4 load_u4_select(const bf *__restrict__ ptr, const unsigned row_base, const bool use_cg) {
  if (use_cg) {
    return load_u4<ld_modifier::cg>(ptr, row_base);
  }
  return load_u4<ld_modifier::cs>(ptr, row_base);
}

DEVICE_FORCEINLINE void apply_6_rounds_pair_major(bf (&vals)[4], const unsigned lane16, const unsigned lane32) {
  // Stage 0 (bit 0)
  vals[1] = bf::sub(vals[1], vals[0]);
  vals[3] = bf::sub(vals[3], vals[2]);

  // Stage 1 (bit 1)
  vals[2] = bf::sub(vals[2], vals[0]);
  vals[3] = bf::sub(vals[3], vals[1]);

#pragma unroll
  for (unsigned bit = 0; bit < 4; bit++) {
    const unsigned lane_bit = (lane16 >> bit) & 1u;
    const unsigned lane_mask = 0u - lane_bit;
    const unsigned peer_lane32 = (lane32 & 16u) | (lane16 ^ (1u << bit));
#pragma unroll
    for (unsigned e = 0; e < 4; e++) {
      const unsigned self = vals[e].limb;
      const unsigned peer = __shfl_sync(0xFFFFFFFFu, self, peer_lane32, 32);
      const unsigned lo = select_u32(self, peer, lane_mask);
      const unsigned hi_pre = select_u32(peer, self, lane_mask);
      const unsigned hi = bf::sub(bf(hi_pre), bf(lo)).limb;
      vals[e] = bf(select_u32(lo, hi, lane_mask));
    }
  }
}

template <unsigned THREADS = 256>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_initial12_log24(const bf *__restrict__ src,
                                                                            bf *__restrict__ dst,
                                                                            const bool use_cg_loads) {
  static_assert(THREADS == 256);
  constexpr unsigned SUB_SIZE = 1u << 12;
  constexpr unsigned TILE_ROWS = 64;
  constexpr unsigned GROUPS = 4;
  constexpr unsigned ELEMS = 4;

  if (blockIdx.y != 0u) {
    return;
  }

  __shared__ bf smem[SUB_SIZE];

  const unsigned tid = threadIdx.x;
  const unsigned subgroup = tid >> 4;
  const unsigned lane16 = tid & 15u;
  const unsigned lane32 = tid & 31u;
  const unsigned subproblem = blockIdx.x;
  const unsigned block_base = subproblem << 12;
  bf regs[GROUPS][ELEMS];

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned hi6 = subgroup * GROUPS + g;
    const unsigned lo_base = lane16 * ELEMS;
    const unsigned row_base = block_base + (hi6 * TILE_ROWS) + lo_base;
    const uint4 packed = use_cg_loads ? load_u4_mod<ld_modifier::cg>(src, row_base) : load_u4_mod<ld_modifier::cs>(src, row_base);
    regs[g][0] = bf(packed.x);
    regs[g][1] = bf(packed.y);
    regs[g][2] = bf(packed.z);
    regs[g][3] = bf(packed.w);
  }

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    apply_6_rounds_pair_major(regs[g], lane16, lane32);
  }

  // Midpoint transpose: convert low-dimension work into high-dimension work.
#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned hi6 = subgroup * GROUPS + g;
#pragma unroll
    for (unsigned e = 0; e < ELEMS; e++) {
      const unsigned lo6 = lane16 * ELEMS + e;
      smem[transpose64_conflict_free_idx(lo6, hi6)] = regs[g][e];
    }
  }

  __syncthreads();

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned hi6 = subgroup * GROUPS + g;
#pragma unroll
    for (unsigned e = 0; e < ELEMS; e++) {
      const unsigned lo6 = lane16 * ELEMS + e;
      regs[g][e] = smem[transpose64_conflict_free_idx(hi6, lo6)];
    }
  }

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    apply_6_rounds_pair_major(regs[g], lane16, lane32);
  }

  // Transpose back to canonical (hi-major) layout before coalesced stores.
#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned hi6 = subgroup * GROUPS + g;
#pragma unroll
    for (unsigned e = 0; e < ELEMS; e++) {
      const unsigned lo6 = lane16 * ELEMS + e;
      smem[transpose64_conflict_free_idx(lo6, hi6)] = regs[g][e];
    }
  }

  __syncthreads();

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned hi6 = subgroup * GROUPS + g;
#pragma unroll
    for (unsigned e = 0; e < ELEMS; e++) {
      const unsigned lo6 = lane16 * ELEMS + e;
      regs[g][e] = smem[transpose64_conflict_free_idx(hi6, lo6)];
    }
  }

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned hi6 = subgroup * GROUPS + g;
    const unsigned lo_base = lane16 * ELEMS;
    const unsigned row_base = block_base + (hi6 * TILE_ROWS) + lo_base;
    const uint4 packed = uint4{
        regs[g][0].limb,
        regs[g][1].limb,
        regs[g][2].limb,
        regs[g][3].limb,
    };
    store_u4_mod<st_modifier::cg>(dst, row_base, packed);
  }
}

template <unsigned ROUNDS, unsigned THREADS = 256>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_initial(const bf *__restrict__ src,
                                                                    bf *__restrict__ dst, const bool use_cg_loads) {
  constexpr unsigned SUB_SIZE = 1u << ROUNDS;
  constexpr unsigned GROUPS = SUB_SIZE >> 2;
  constexpr unsigned MAX_THREADS = THREADS;
  static_assert(MAX_THREADS == 256);
  static_assert((SUB_SIZE % 4) == 0);
  if (blockIdx.y != 0u) {
    return;
  }

  __shared__ bf smem[SUB_SIZE];

  const unsigned tid = threadIdx.x;
  const unsigned subproblem = blockIdx.x;
  const unsigned base = subproblem << ROUNDS;

  for (unsigned group = tid; group < GROUPS; group += MAX_THREADS) {
    const unsigned local_base = group << 2;
    const unsigned row_base = base + local_base;
    const uint4 packed = load_u4_select(src, row_base, use_cg_loads);
    smem[swizzle_shared_lane(local_base + 0)] = bf(packed.x);
    smem[swizzle_shared_lane(local_base + 1)] = bf(packed.y);
    smem[swizzle_shared_lane(local_base + 2)] = bf(packed.z);
    smem[swizzle_shared_lane(local_base + 3)] = bf(packed.w);
  }

  __syncthreads();

#pragma unroll
  for (unsigned stage = 0; stage < ROUNDS; stage++) {
    const unsigned bit = 1u << stage;
    for (unsigned idx = tid; idx < SUB_SIZE; idx += MAX_THREADS) {
      if ((idx & bit) == 0u) {
        continue;
      }
      const unsigned self_idx = swizzle_shared_lane(idx);
      const unsigned partner_idx = swizzle_shared_lane(idx ^ bit);
      smem[self_idx] = bf::sub(smem[self_idx], smem[partner_idx]);
    }
    if (stage + 1u < ROUNDS) {
      __syncthreads();
    }
  }
  __syncthreads();

  for (unsigned group = tid; group < GROUPS; group += MAX_THREADS) {
    const unsigned local_base = group << 2;
    const unsigned row_base = base + local_base;
    const uint4 packed = uint4{
        smem[swizzle_shared_lane(local_base + 0)].limb,
        smem[swizzle_shared_lane(local_base + 1)].limb,
        smem[swizzle_shared_lane(local_base + 2)].limb,
        smem[swizzle_shared_lane(local_base + 3)].limb,
    };
    store<uint4, st_modifier::cg>(reinterpret_cast<uint4 *>(dst + row_base), packed);
  }
}

template <unsigned ROUNDS, unsigned THREADS = 256>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial(const bf *__restrict__ src,
                                                                       bf *__restrict__ dst, const bool use_cg_loads,
                                                                       const unsigned start_stage) {
  constexpr unsigned SUB_SIZE = 1u << ROUNDS;
  constexpr unsigned MAX_THREADS = THREADS;
  if (blockIdx.y != 0u) {
    return;
  }

  // Noninitial kernels process "k" dimension in tiles:
  // each CTA owns LOW_TILE contiguous low indices and walks all k iters.
  constexpr unsigned K_WARPS = 1u << (ROUNDS - 5);
  constexpr unsigned LOW_TILE = MAX_THREADS / K_WARPS;
  static_assert((LOW_TILE % 4u) == 0u);

  constexpr unsigned LOW_GROUPS = LOW_TILE / 4;
  constexpr unsigned GROUPS_PER_K = LOW_GROUPS;
  constexpr unsigned LOW_MASK = LOW_TILE - 1u;
  constexpr unsigned LOW_LOG = (LOW_TILE == 128u) ? 7u : ((LOW_TILE == 64u) ? 6u : 5u);
  constexpr unsigned LOW_GROUP_LOG = (LOW_GROUPS == 32u) ? 5u : ((LOW_GROUPS == 16u) ? 4u : 3u);
  constexpr unsigned K_VEC_STRIDE = MAX_THREADS >> LOW_GROUP_LOG;
  constexpr unsigned K_STAGE_STRIDE = MAX_THREADS >> LOW_LOG;
  static_assert((LOW_TILE & LOW_MASK) == 0u);
  static_assert((1u << LOW_LOG) == LOW_TILE);
  static_assert((1u << LOW_GROUP_LOG) == LOW_GROUPS);
  static_assert(K_VEC_STRIDE > 0u);
  static_assert(K_STAGE_STRIDE > 0u);

  __shared__ bf smem[LOW_TILE * SUB_SIZE];

  const unsigned tid = threadIdx.x;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride / LOW_TILE;
  const unsigned tile = blockIdx.x;
  const unsigned high = tile / low_tiles;
  const unsigned low_base = (tile - high * low_tiles) * LOW_TILE;
  const unsigned block_base = high << (start_stage + ROUNDS);

  const unsigned low_group0 = tid & (LOW_GROUPS - 1u);
  const unsigned k_vec0 = tid >> LOW_GROUP_LOG;
  for (unsigned k = k_vec0; k < SUB_SIZE; k += K_VEC_STRIDE) {
    const unsigned low_group = low_group0;
    const unsigned low = low_base + (low_group << 2);
    const unsigned row_base = block_base + low + (k << start_stage);
    const uint4 packed = load_u4_select(src, row_base, use_cg_loads);
    const unsigned low_local = low_group << 2;
    const unsigned k_swizzle = k & LOW_MASK;
    const unsigned smem_base = k * LOW_TILE;
    smem[smem_base + ((low_local + 0) ^ k_swizzle)] = bf(packed.x);
    smem[smem_base + ((low_local + 1) ^ k_swizzle)] = bf(packed.y);
    smem[smem_base + ((low_local + 2) ^ k_swizzle)] = bf(packed.z);
    smem[smem_base + ((low_local + 3) ^ k_swizzle)] = bf(packed.w);
  }

  __syncthreads();

  const unsigned low_off = tid & LOW_MASK;
  const unsigned k_stage0 = tid >> LOW_LOG;
#pragma unroll
  for (unsigned stage = 0; stage < ROUNDS; stage++) {
    const unsigned bit = 1u << stage;
    for (unsigned k = k_stage0; k < SUB_SIZE; k += K_STAGE_STRIDE) {
      if ((k & bit) == 0u) {
        continue;
      }
      const unsigned partner = k ^ bit;
      const unsigned self_idx = k * LOW_TILE + (low_off ^ (k & LOW_MASK));
      const unsigned partner_idx = partner * LOW_TILE + (low_off ^ (partner & LOW_MASK));
      smem[self_idx] = bf::sub(smem[self_idx], smem[partner_idx]);
    }
    if (stage + 1u < ROUNDS) {
      __syncthreads();
    }
  }
  __syncthreads();

  for (unsigned k = k_vec0; k < SUB_SIZE; k += K_VEC_STRIDE) {
    const unsigned low_group = low_group0;
    const unsigned low = low_base + (low_group << 2);
    const unsigned row_base = block_base + low + (k << start_stage);
    const unsigned low_local = low_group << 2;
    const unsigned k_swizzle = k & LOW_MASK;
    const unsigned smem_base = k * LOW_TILE;
    const uint4 packed = uint4{
        smem[smem_base + ((low_local + 0) ^ k_swizzle)].limb,
        smem[smem_base + ((low_local + 1) ^ k_swizzle)].limb,
        smem[smem_base + ((low_local + 2) ^ k_swizzle)].limb,
        smem[smem_base + ((low_local + 3) ^ k_swizzle)].limb,
    };
    store<uint4, st_modifier::cg>(reinterpret_cast<uint4 *>(dst + row_base), packed);
  }
}

#define H2M_INITIAL_KERNEL(rounds)                                                                                                                        \
  EXTERN __launch_bounds__(256, 2) __global__ void ab_h2m_bitrev_bf_initial_##rounds##_kernel(                                                          \
      const bf *__restrict__ src, bf *__restrict__ dst, const unsigned use_cg_loads, const unsigned start_stage, const unsigned log_rows) {               \
    (void)start_stage;                                                                                                                                    \
    (void)log_rows;                                                                                                                                       \
    hypercube_evals_into_coeffs_bitrev_initial<rounds, 256>(src, dst, use_cg_loads != 0u);                                                              \
  }

#define H2M_NONINITIAL_KERNEL(rounds)                                                                                                                     \
  EXTERN __launch_bounds__(256, 2) __global__ void ab_h2m_bitrev_bf_noninitial_##rounds##_kernel(                                                       \
      const bf *__restrict__ src, bf *__restrict__ dst, const unsigned use_cg_loads, const unsigned start_stage, const unsigned log_rows) {               \
    (void)log_rows;                                                                                                                                       \
    hypercube_evals_into_coeffs_bitrev_noninitial<rounds, 256>(src, dst, use_cg_loads != 0u, start_stage);                                              \
  }

#define H2M_NONINITIAL_128_KERNEL(rounds)                                                                                                                 \
  EXTERN __launch_bounds__(128, 2) __global__ void ab_h2m_bitrev_bf_noninitial_##rounds##_128_kernel(                                                   \
      const bf *__restrict__ src, bf *__restrict__ dst, const unsigned use_cg_loads, const unsigned start_stage, const unsigned log_rows) {               \
    (void)log_rows;                                                                                                                                       \
    hypercube_evals_into_coeffs_bitrev_noninitial<rounds, 128>(src, dst, use_cg_loads != 0u, start_stage);                                              \
  }

// Initial kernels (start at stride 1)
H2M_INITIAL_KERNEL(8);
H2M_INITIAL_KERNEL(9);
H2M_INITIAL_KERNEL(10);
H2M_INITIAL_KERNEL(11);

EXTERN __launch_bounds__(256) __global__ void ab_h2m_bitrev_bf_initial_12_kernel(const bf *__restrict__ src,
                                                                                       bf *__restrict__ dst,
                                                                                       const unsigned use_cg_loads,
                                                                                       const unsigned start_stage,
                                                                                       const unsigned log_rows) {
  (void)start_stage;
  (void)log_rows;
  hypercube_evals_into_coeffs_bitrev_initial12_log24<256>(src, dst, use_cg_loads != 0u);
}

// Noninitial kernels (start at big stride)
H2M_NONINITIAL_KERNEL(6);
H2M_NONINITIAL_KERNEL(7);
H2M_NONINITIAL_KERNEL(8);
H2M_NONINITIAL_128_KERNEL(7);

} // namespace airbender::ops::hypercube
