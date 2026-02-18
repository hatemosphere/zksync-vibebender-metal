#include "../field.cuh"
#include "../memory.cuh"

using namespace ::airbender::field;
using namespace ::airbender::memory;

namespace airbender::ops::hypercube {

template <unsigned ROUNDS> DEVICE_FORCEINLINE unsigned transpose_old_to_new(const unsigned old_tid) {
  if constexpr (ROUNDS <= 5) {
    return old_tid;
  } else {
    constexpr unsigned CROSS = ROUNDS - 5;
    constexpr unsigned WARPS = 1u << CROSS;
    const unsigned lane = old_tid & 31u;
    const unsigned warp = old_tid >> 5;
    return lane * WARPS + warp;
  }
}

template <unsigned ROUNDS> DEVICE_FORCEINLINE unsigned transpose_new_to_old(const unsigned new_tid) {
  if constexpr (ROUNDS <= 5) {
    return new_tid;
  } else {
    constexpr unsigned CROSS = ROUNDS - 5;
    constexpr unsigned WARPS = 1u << CROSS;
    // Inverse of new_tid = old_lane * WARPS + old_warp.
    const unsigned old_warp = new_tid & (WARPS - 1u);
    const unsigned old_lane = new_tid >> CROSS;
    return (old_warp << 5) | old_lane;
  }
}

template <unsigned ROUNDS, bool WARP_SHARED_BACKEND, bool INITIAL>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_fused(const matrix_getter<bf, ld_modifier::cs> src,
                                                                  const matrix_setter<bf, st_modifier::cs> dst,
                                                                  const unsigned start_stage, const unsigned log_rows) {
  constexpr unsigned SUB_SIZE = 1u << ROUNDS;
  constexpr unsigned LOCAL_ROUNDS = ROUNDS < 5 ? ROUNDS : 5;
  constexpr unsigned MAX_THREADS = 256;

  __shared__ bf smem[MAX_THREADS];

  const unsigned tid = threadIdx.x;
  const unsigned stage0 = INITIAL ? 0u : start_stage;

  const unsigned subproblem = blockIdx.x;
  const unsigned col = blockIdx.y;

  const unsigned stride = 1u << stage0;
  const unsigned low = INITIAL ? 0u : (subproblem & (stride - 1u));
  const unsigned high = INITIAL ? subproblem : (subproblem >> stage0);
  const unsigned base = (high << (stage0 + ROUNDS)) + low;

  const unsigned n = 1u << log_rows;
  const bool active = tid < SUB_SIZE;

  bf value = bf::ZERO();
  if (active) {
    const unsigned row = base + (tid << stage0);
    if (row < n) {
      value = src.get(row, col);
    }
  }

  if constexpr (WARP_SHARED_BACKEND) {
#pragma unroll
    for (unsigned stage = 0; stage < LOCAL_ROUNDS; stage++) {
      if (active)
        smem[tid] = value;
      __syncwarp();
      if (active && ((tid >> stage) & 1u)) {
        const unsigned partner = tid ^ (1u << stage);
        value = bf::sub(value, smem[partner]);
      }
      __syncwarp();
    }
    // Warp-private local rounds reuse the shared-memory buffer.
    // Before cross-warp remap (which writes a global permutation into `smem`),
    // all warps must finish local phases to avoid inter-warp races.
    if constexpr (ROUNDS > 5) {
      __syncthreads();
    }
  } else {
#pragma unroll
    for (unsigned stage = 0; stage < LOCAL_ROUNDS; stage++) {
      const unsigned partner = __shfl_xor_sync(0xffffffffu, value.limb, 1u << stage);
      if (active && ((tid >> stage) & 1u)) {
        value = bf::sub(value, bf(partner));
      }
    }
  }

  if constexpr (ROUNDS > 5) {
    constexpr unsigned CROSS_ROUNDS = ROUNDS - 5;

    if (active) {
      const unsigned remapped_tid = transpose_old_to_new<ROUNDS>(tid);
      smem[remapped_tid] = value;
    }

    __syncthreads();

    if (active) {
      value = smem[tid];
#pragma unroll
      for (unsigned stage = 0; stage < CROSS_ROUNDS; stage++) {
        const unsigned partner = __shfl_xor_sync(0xffffffffu, value.limb, 1u << stage);
        if ((tid >> stage) & 1u) {
          value = bf::sub(value, bf(partner));
        }
      }

      const unsigned old_tid = transpose_new_to_old<ROUNDS>(tid);
      const unsigned row = base + (old_tid << stage0);
      if (row < n) {
        dst.set(row, col, value);
      }
    }
  } else {
    if (active) {
      const unsigned row = base + (tid << stage0);
      if (row < n) {
        dst.set(row, col, value);
      }
    }
  }
}

#define H2M_KERNEL(family, rounds, mode, initial_flag, warp_shared_flag)                                                                                \
  EXTERN __launch_bounds__(256, 2) __global__ void ab_h2m_bitrev_bf_##family##_##rounds##_##mode##_kernel(                                              \
      const matrix_getter<bf, ld_modifier::cs> src, const matrix_setter<bf, st_modifier::cs> dst, const unsigned start_stage, const unsigned log_rows) { \
    hypercube_evals_into_coeffs_bitrev_fused<rounds, warp_shared_flag, initial_flag>(src, dst, start_stage, log_rows);                                   \
  }

// Initial kernels (start at stride 1)
H2M_KERNEL(initial, 8, shuffle, true, false);
H2M_KERNEL(initial, 9, shuffle, true, false);
H2M_KERNEL(initial, 10, shuffle, true, false);
H2M_KERNEL(initial, 11, shuffle, true, false);
H2M_KERNEL(initial, 12, shuffle, true, false);

H2M_KERNEL(initial, 8, warp_shared, true, true);
H2M_KERNEL(initial, 9, warp_shared, true, true);
H2M_KERNEL(initial, 10, warp_shared, true, true);
H2M_KERNEL(initial, 11, warp_shared, true, true);
H2M_KERNEL(initial, 12, warp_shared, true, true);

// Noninitial kernels (start at big stride)
H2M_KERNEL(noninitial, 6, shuffle, false, false);
H2M_KERNEL(noninitial, 7, shuffle, false, false);
H2M_KERNEL(noninitial, 8, shuffle, false, false);

H2M_KERNEL(noninitial, 6, warp_shared, false, true);
H2M_KERNEL(noninitial, 7, warp_shared, false, true);
H2M_KERNEL(noninitial, 8, warp_shared, false, true);

} // namespace airbender::ops::hypercube
