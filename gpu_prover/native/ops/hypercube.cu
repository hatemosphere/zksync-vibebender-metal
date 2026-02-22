#include "../field.cuh"
#include "../memory.cuh"

using namespace ::airbender::field;
using namespace ::airbender::memory;

namespace airbender::ops::hypercube {

DEVICE_FORCEINLINE unsigned select_u32(const unsigned a, const unsigned b, const unsigned mask) {
  // Branchless select used in the shuffle-based butterfly path.
  // mask is expected to be either 0x00000000 (pick a) or 0xFFFFFFFF (pick b).
  return (a & ~mask) | (b & mask);
}

DEVICE_FORCEINLINE unsigned initial12_smem_u4_idx(const unsigned row6, const unsigned col4) {
  // Initial12 shared layout helper.
  //
  // Logical tile shape:
  // - 64 rows (row6 in [0,63])
  // - 16 uint4 columns (col4 in [0,15])
  //
  // Physical index:
  // - row-major base: row6 * 16 + col4
  // - swizzled column: col4 ^ (row6 >> 2)
  //
  // Why this swizzle:
  // - During the transpose boundary we access the same data with two different
  //   (row, col) traversals. A plain row-major layout creates concentrated bank
  //   pressure in one of those traversals.
  // - The row-dependent XOR spreads accesses more uniformly across banks while
  //   preserving a bijection (no collisions, no data loss).
  return (row6 << 4) | (col4 ^ (row6 >> 2));
}

DEVICE_FORCEINLINE unsigned initial11_smem_idx(const unsigned k, const unsigned p) {
  // Initial11 shared layout helper for a 64 x 32 scalar tile.
  //
  // Physical index:
  // - row-major base: k * 32 + p
  // - swizzled partition: p ^ (k >> 2)
  //
  // Why this swizzle:
  // - The initial/final remap touches k as lane16*4 + e (so k>>2 tracks lane16).
  // - The second-half warp pass touches fixed k and p=lane32.
  // - Using k>>2 in the xor keeps both patterns bank-friendly while preserving
  //   a bijection in each row.
  return (k << 5) | (p ^ (k >> 2));
}

DEVICE_FORCEINLINE unsigned noninitial6_smem_idx(const unsigned k, const unsigned p) {
  // Noninitial6 shared layout helper.
  //
  // Logical tile shape:
  // - k dimension: 64 rows (round domain)
  // - p dimension: 32 partitions
  //
  // Physical index:
  // - row-major base: k * 32 + p
  // - swizzled partition: p ^ (k & 31)
  //
  // Why this swizzle:
  // - Warp compute repeatedly loads/stores values at (k, p) and (k+32, p) with
  //   lane-aligned k. The swizzle keeps those lane-correlated accesses bank-friendly.
  return (k << 5) | (p ^ (k & 31u));
}

template <ld_modifier MODIFIER>
DEVICE_FORCEINLINE uint4 load_u4_mod(const bf *__restrict__ ptr, const unsigned row_base) {
  // BF is 32-bit; uint4 gives one 16-byte vector transaction per thread.
  return load<uint4, MODIFIER>(reinterpret_cast<const uint4 *>(ptr + row_base));
}

template <st_modifier MODIFIER>
DEVICE_FORCEINLINE void store_u4_mod(bf *__restrict__ ptr, const unsigned row_base, const uint4 packed) {
  store<uint4, MODIFIER>(reinterpret_cast<uint4 *>(ptr + row_base), packed);
}

DEVICE_FORCEINLINE void apply_pair_major_high_update_from_peers(
    bf (&vals)[4], const unsigned long long peer01, const unsigned long long peer23) {
  // vals is a local 4-element register group in pair-major order:
  // [lo0, hi0, lo1, hi1].
  //
  // peer01/peer23 carry the neighbor lane values packed as two 32-bit limbs.
  // This helper performs the "high -= low" update for both pairs using peer lows.
  const unsigned peer0 = static_cast<unsigned>(peer01 & 0xFFFFFFFFull);
  const unsigned peer1 = static_cast<unsigned>(peer01 >> 32);
  const unsigned peer2 = static_cast<unsigned>(peer23 & 0xFFFFFFFFull);
  const unsigned peer3 = static_cast<unsigned>(peer23 >> 32);
  vals[0] = bf::sub(vals[0], bf(peer0));
  vals[1] = bf::sub(vals[1], bf(peer1));
  vals[2] = bf::sub(vals[2], bf(peer2));
  vals[3] = bf::sub(vals[3], bf(peer3));
}

DEVICE_FORCEINLINE void apply_6_rounds_pair_major_4groups(bf (&vals)[4][4], const unsigned lane16) {
  // Apply six rounds over 4 independent register groups.
  //
  // Contract:
  // - vals[g] is [lo0, hi0, lo1, hi1] for group g.
  // - Operation per round is the GKR transform butterfly update on the high side:
  //     hi = hi - lo
  //
  // Rounds 0..1:
  // - Entirely lane-local; each thread has both operands in registers.
  //
  // Rounds 2..5:
  // - Operands span lanes inside one 16-thread subgroup.
  // - We exchange packed values with __shfl_xor_sync(width=16).
  // - Only lanes in the "high" half of each xor pair perform arithmetic.
  //   (low lanes keep their value unchanged for that stage.)
#pragma unroll
  for (unsigned g = 0; g < 4; g++) {
    vals[g][1] = bf::sub(vals[g][1], vals[g][0]);
    vals[g][3] = bf::sub(vals[g][3], vals[g][2]);
    vals[g][2] = bf::sub(vals[g][2], vals[g][0]);
    vals[g][3] = bf::sub(vals[g][3], vals[g][1]);
  }

  // Rounds 2..5 use lane-xor exchange inside 16-lane subgroups.
#pragma unroll
  for (unsigned bit = 0; bit < 4; bit++) {
    const unsigned lane_bit = 1u << bit;
    const bool high_lane = (lane16 & lane_bit) != 0u;

    const unsigned long long pair01_g0 =
        (static_cast<unsigned long long>(vals[0][1].limb) << 32) | static_cast<unsigned long long>(vals[0][0].limb);
    const unsigned long long pair23_g0 =
        (static_cast<unsigned long long>(vals[0][3].limb) << 32) | static_cast<unsigned long long>(vals[0][2].limb);
    const unsigned long long peer01_g0 = __shfl_xor_sync(0xFFFFFFFFu, pair01_g0, lane_bit, 16);
    const unsigned long long peer23_g0 = __shfl_xor_sync(0xFFFFFFFFu, pair23_g0, lane_bit, 16);

    const unsigned long long pair01_g1 =
        (static_cast<unsigned long long>(vals[1][1].limb) << 32) | static_cast<unsigned long long>(vals[1][0].limb);
    const unsigned long long pair23_g1 =
        (static_cast<unsigned long long>(vals[1][3].limb) << 32) | static_cast<unsigned long long>(vals[1][2].limb);
    const unsigned long long peer01_g1 = __shfl_xor_sync(0xFFFFFFFFu, pair01_g1, lane_bit, 16);
    const unsigned long long peer23_g1 = __shfl_xor_sync(0xFFFFFFFFu, pair23_g1, lane_bit, 16);

    // Only high lanes update: hi <- hi - lo (with peer-provided lows).
    if (high_lane) {
      apply_pair_major_high_update_from_peers(vals[0], peer01_g0, peer23_g0);
    }

    const unsigned long long pair01_g2 =
        (static_cast<unsigned long long>(vals[2][1].limb) << 32) | static_cast<unsigned long long>(vals[2][0].limb);
    const unsigned long long pair23_g2 =
        (static_cast<unsigned long long>(vals[2][3].limb) << 32) | static_cast<unsigned long long>(vals[2][2].limb);
    const unsigned long long peer01_g2 = __shfl_xor_sync(0xFFFFFFFFu, pair01_g2, lane_bit, 16);
    const unsigned long long peer23_g2 = __shfl_xor_sync(0xFFFFFFFFu, pair23_g2, lane_bit, 16);

    if (high_lane) {
      apply_pair_major_high_update_from_peers(vals[1], peer01_g1, peer23_g1);
    }

    const unsigned long long pair01_g3 =
        (static_cast<unsigned long long>(vals[3][1].limb) << 32) | static_cast<unsigned long long>(vals[3][0].limb);
    const unsigned long long pair23_g3 =
        (static_cast<unsigned long long>(vals[3][3].limb) << 32) | static_cast<unsigned long long>(vals[3][2].limb);
    const unsigned long long peer01_g3 = __shfl_xor_sync(0xFFFFFFFFu, pair01_g3, lane_bit, 16);
    const unsigned long long peer23_g3 = __shfl_xor_sync(0xFFFFFFFFu, pair23_g3, lane_bit, 16);

    if (high_lane) {
      apply_pair_major_high_update_from_peers(vals[2], peer01_g2, peer23_g2);
      apply_pair_major_high_update_from_peers(vals[3], peer01_g3, peer23_g3);
    }
  }
}

DEVICE_FORCEINLINE void apply_6_rounds_pair_major_2groups(bf (&vals)[2][4], const unsigned lane16) {
  // Same round structure as apply_6_rounds_pair_major_4groups, specialized to
  // two register groups to avoid redundant packing/shuffle traffic.
#pragma unroll
  for (unsigned g = 0; g < 2; g++) {
    vals[g][1] = bf::sub(vals[g][1], vals[g][0]);
    vals[g][3] = bf::sub(vals[g][3], vals[g][2]);
    vals[g][2] = bf::sub(vals[g][2], vals[g][0]);
    vals[g][3] = bf::sub(vals[g][3], vals[g][1]);
  }

#pragma unroll
  for (unsigned bit = 0; bit < 4; bit++) {
    const unsigned lane_bit = 1u << bit;
    const bool high_lane = (lane16 & lane_bit) != 0u;

    const unsigned long long pair01_g0 =
        (static_cast<unsigned long long>(vals[0][1].limb) << 32) | static_cast<unsigned long long>(vals[0][0].limb);
    const unsigned long long pair23_g0 =
        (static_cast<unsigned long long>(vals[0][3].limb) << 32) | static_cast<unsigned long long>(vals[0][2].limb);
    const unsigned long long peer01_g0 = __shfl_xor_sync(0xFFFFFFFFu, pair01_g0, lane_bit, 16);
    const unsigned long long peer23_g0 = __shfl_xor_sync(0xFFFFFFFFu, pair23_g0, lane_bit, 16);

    const unsigned long long pair01_g1 =
        (static_cast<unsigned long long>(vals[1][1].limb) << 32) | static_cast<unsigned long long>(vals[1][0].limb);
    const unsigned long long pair23_g1 =
        (static_cast<unsigned long long>(vals[1][3].limb) << 32) | static_cast<unsigned long long>(vals[1][2].limb);
    const unsigned long long peer01_g1 = __shfl_xor_sync(0xFFFFFFFFu, pair01_g1, lane_bit, 16);
    const unsigned long long peer23_g1 = __shfl_xor_sync(0xFFFFFFFFu, pair23_g1, lane_bit, 16);

    if (high_lane) {
      apply_pair_major_high_update_from_peers(vals[0], peer01_g0, peer23_g0);
      apply_pair_major_high_update_from_peers(vals[1], peer01_g1, peer23_g1);
    }
  }
}

DEVICE_FORCEINLINE void apply_5_rounds_warp64_pair_branchless_quad(
    bf &a0,
    bf &a1,
    bf &b0,
    bf &b1,
    bf &c0,
    bf &c1,
    bf &d0,
    bf &d1,
    const unsigned lane32) {
  // Apply rounds 0..4 for noninitial6 on four independent 2-value pairs.
  //
  // This variant is fully branchless in arithmetic selection:
  // - For each xor partner, form (lo, hi_pre) using lane mask selects.
  // - Compute hi = hi_pre - lo.
  // - Select output as lo for low lanes, hi for high lanes.
  //
  // Processing four pairs in one loop increases ILP and helps hide shuffle latency.
#pragma unroll
  for (unsigned bit = 0; bit < 5; bit++) {
    const unsigned mask = 1u << bit;
    const unsigned lane_bit = (lane32 >> bit) & 1u;
    const unsigned lane_mask = 0u - lane_bit;

    const unsigned a0_self = a0.limb;
    const unsigned a1_self = a1.limb;
    const unsigned b0_self = b0.limb;
    const unsigned b1_self = b1.limb;
    const unsigned c0_self = c0.limb;
    const unsigned c1_self = c1.limb;
    const unsigned d0_self = d0.limb;
    const unsigned d1_self = d1.limb;

    const unsigned a0_peer = __shfl_xor_sync(0xFFFFFFFFu, a0_self, mask, 32);
    const unsigned a1_peer = __shfl_xor_sync(0xFFFFFFFFu, a1_self, mask, 32);
    const unsigned b0_peer = __shfl_xor_sync(0xFFFFFFFFu, b0_self, mask, 32);
    const unsigned b1_peer = __shfl_xor_sync(0xFFFFFFFFu, b1_self, mask, 32);
    const unsigned c0_peer = __shfl_xor_sync(0xFFFFFFFFu, c0_self, mask, 32);
    const unsigned c1_peer = __shfl_xor_sync(0xFFFFFFFFu, c1_self, mask, 32);
    const unsigned d0_peer = __shfl_xor_sync(0xFFFFFFFFu, d0_self, mask, 32);
    const unsigned d1_peer = __shfl_xor_sync(0xFFFFFFFFu, d1_self, mask, 32);

    const unsigned a0_lo = select_u32(a0_self, a0_peer, lane_mask);
    const unsigned a0_hi_pre = select_u32(a0_peer, a0_self, lane_mask);
    const unsigned a0_hi = bf::sub(bf(a0_hi_pre), bf(a0_lo)).limb;
    a0 = bf(select_u32(a0_lo, a0_hi, lane_mask));

    const unsigned a1_lo = select_u32(a1_self, a1_peer, lane_mask);
    const unsigned a1_hi_pre = select_u32(a1_peer, a1_self, lane_mask);
    const unsigned a1_hi = bf::sub(bf(a1_hi_pre), bf(a1_lo)).limb;
    a1 = bf(select_u32(a1_lo, a1_hi, lane_mask));

    const unsigned b0_lo = select_u32(b0_self, b0_peer, lane_mask);
    const unsigned b0_hi_pre = select_u32(b0_peer, b0_self, lane_mask);
    const unsigned b0_hi = bf::sub(bf(b0_hi_pre), bf(b0_lo)).limb;
    b0 = bf(select_u32(b0_lo, b0_hi, lane_mask));

    const unsigned b1_lo = select_u32(b1_self, b1_peer, lane_mask);
    const unsigned b1_hi_pre = select_u32(b1_peer, b1_self, lane_mask);
    const unsigned b1_hi = bf::sub(bf(b1_hi_pre), bf(b1_lo)).limb;
    b1 = bf(select_u32(b1_lo, b1_hi, lane_mask));

    const unsigned c0_lo = select_u32(c0_self, c0_peer, lane_mask);
    const unsigned c0_hi_pre = select_u32(c0_peer, c0_self, lane_mask);
    const unsigned c0_hi = bf::sub(bf(c0_hi_pre), bf(c0_lo)).limb;
    c0 = bf(select_u32(c0_lo, c0_hi, lane_mask));

    const unsigned c1_lo = select_u32(c1_self, c1_peer, lane_mask);
    const unsigned c1_hi_pre = select_u32(c1_peer, c1_self, lane_mask);
    const unsigned c1_hi = bf::sub(bf(c1_hi_pre), bf(c1_lo)).limb;
    c1 = bf(select_u32(c1_lo, c1_hi, lane_mask));

    const unsigned d0_lo = select_u32(d0_self, d0_peer, lane_mask);
    const unsigned d0_hi_pre = select_u32(d0_peer, d0_self, lane_mask);
    const unsigned d0_hi = bf::sub(bf(d0_hi_pre), bf(d0_lo)).limb;
    d0 = bf(select_u32(d0_lo, d0_hi, lane_mask));

    const unsigned d1_lo = select_u32(d1_self, d1_peer, lane_mask);
    const unsigned d1_hi_pre = select_u32(d1_peer, d1_self, lane_mask);
    const unsigned d1_hi = bf::sub(bf(d1_hi_pre), bf(d1_lo)).limb;
    d1 = bf(select_u32(d1_lo, d1_hi, lane_mask));
  }
}

DEVICE_FORCEINLINE void apply_5_rounds_warp64_pair_branchless_quad_packed(
    bf &a0,
    bf &a1,
    bf &b0,
    bf &b1,
    bf &c0,
    bf &c1,
    bf &d0,
    bf &d1,
    const unsigned lane32) {
  // Packed variant for initial11: pair (x0, x1) into one 64-bit shuffle.
#pragma unroll
  for (unsigned bit = 0; bit < 5; bit++) {
    const unsigned mask = 1u << bit;
    const unsigned lane_bit = (lane32 >> bit) & 1u;
    const unsigned lane_mask = 0u - lane_bit;

    const unsigned a0_self = a0.limb;
    const unsigned a1_self = a1.limb;
    const unsigned b0_self = b0.limb;
    const unsigned b1_self = b1.limb;
    const unsigned c0_self = c0.limb;
    const unsigned c1_self = c1.limb;
    const unsigned d0_self = d0.limb;
    const unsigned d1_self = d1.limb;

    const unsigned long long a_pair =
        (static_cast<unsigned long long>(a1_self) << 32) | static_cast<unsigned long long>(a0_self);
    const unsigned long long b_pair =
        (static_cast<unsigned long long>(b1_self) << 32) | static_cast<unsigned long long>(b0_self);
    const unsigned long long c_pair =
        (static_cast<unsigned long long>(c1_self) << 32) | static_cast<unsigned long long>(c0_self);
    const unsigned long long d_pair =
        (static_cast<unsigned long long>(d1_self) << 32) | static_cast<unsigned long long>(d0_self);

    const unsigned long long a_peer_pair = __shfl_xor_sync(0xFFFFFFFFu, a_pair, mask, 32);
    const unsigned long long b_peer_pair = __shfl_xor_sync(0xFFFFFFFFu, b_pair, mask, 32);
    const unsigned long long c_peer_pair = __shfl_xor_sync(0xFFFFFFFFu, c_pair, mask, 32);
    const unsigned long long d_peer_pair = __shfl_xor_sync(0xFFFFFFFFu, d_pair, mask, 32);

    const unsigned a0_peer = static_cast<unsigned>(a_peer_pair & 0xFFFFFFFFull);
    const unsigned a1_peer = static_cast<unsigned>(a_peer_pair >> 32);
    const unsigned b0_peer = static_cast<unsigned>(b_peer_pair & 0xFFFFFFFFull);
    const unsigned b1_peer = static_cast<unsigned>(b_peer_pair >> 32);
    const unsigned c0_peer = static_cast<unsigned>(c_peer_pair & 0xFFFFFFFFull);
    const unsigned c1_peer = static_cast<unsigned>(c_peer_pair >> 32);
    const unsigned d0_peer = static_cast<unsigned>(d_peer_pair & 0xFFFFFFFFull);
    const unsigned d1_peer = static_cast<unsigned>(d_peer_pair >> 32);

    const unsigned a0_lo = select_u32(a0_self, a0_peer, lane_mask);
    const unsigned a0_hi_pre = select_u32(a0_peer, a0_self, lane_mask);
    const unsigned a0_hi = bf::sub(bf(a0_hi_pre), bf(a0_lo)).limb;
    a0 = bf(select_u32(a0_lo, a0_hi, lane_mask));

    const unsigned a1_lo = select_u32(a1_self, a1_peer, lane_mask);
    const unsigned a1_hi_pre = select_u32(a1_peer, a1_self, lane_mask);
    const unsigned a1_hi = bf::sub(bf(a1_hi_pre), bf(a1_lo)).limb;
    a1 = bf(select_u32(a1_lo, a1_hi, lane_mask));

    const unsigned b0_lo = select_u32(b0_self, b0_peer, lane_mask);
    const unsigned b0_hi_pre = select_u32(b0_peer, b0_self, lane_mask);
    const unsigned b0_hi = bf::sub(bf(b0_hi_pre), bf(b0_lo)).limb;
    b0 = bf(select_u32(b0_lo, b0_hi, lane_mask));

    const unsigned b1_lo = select_u32(b1_self, b1_peer, lane_mask);
    const unsigned b1_hi_pre = select_u32(b1_peer, b1_self, lane_mask);
    const unsigned b1_hi = bf::sub(bf(b1_hi_pre), bf(b1_lo)).limb;
    b1 = bf(select_u32(b1_lo, b1_hi, lane_mask));

    const unsigned c0_lo = select_u32(c0_self, c0_peer, lane_mask);
    const unsigned c0_hi_pre = select_u32(c0_peer, c0_self, lane_mask);
    const unsigned c0_hi = bf::sub(bf(c0_hi_pre), bf(c0_lo)).limb;
    c0 = bf(select_u32(c0_lo, c0_hi, lane_mask));

    const unsigned c1_lo = select_u32(c1_self, c1_peer, lane_mask);
    const unsigned c1_hi_pre = select_u32(c1_peer, c1_self, lane_mask);
    const unsigned c1_hi = bf::sub(bf(c1_hi_pre), bf(c1_lo)).limb;
    c1 = bf(select_u32(c1_lo, c1_hi, lane_mask));

    const unsigned d0_lo = select_u32(d0_self, d0_peer, lane_mask);
    const unsigned d0_hi_pre = select_u32(d0_peer, d0_self, lane_mask);
    const unsigned d0_hi = bf::sub(bf(d0_hi_pre), bf(d0_lo)).limb;
    d0 = bf(select_u32(d0_lo, d0_hi, lane_mask));

    const unsigned d1_lo = select_u32(d1_self, d1_peer, lane_mask);
    const unsigned d1_hi_pre = select_u32(d1_peer, d1_self, lane_mask);
    const unsigned d1_hi = bf::sub(bf(d1_hi_pre), bf(d1_lo)).limb;
    d1 = bf(select_u32(d1_lo, d1_hi, lane_mask));
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER, bool IN_PLACE>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_initial12(const bf *__restrict__ src, bf *__restrict__ dst) {
  // Initial12 kernel dataflow (handles one 2^12 chunk per CTA):
  //
  // Phase A: Global load (vectorized)
  // - 256 threads = 16 subgroups of 16 lanes.
  // - Each subgroup owns 256 values arranged as 4 stripes of length 64.
  // - Each lane loads 4 contiguous BF values per stripe (uint4).
  //
  // Phase B: First six rounds
  // - Run 6 rounds in subgroup-local form using apply_6_rounds_pair_major_4groups.
  //
  // Phase C: Shared transpose boundary
  // - Spill to shared tile A with swizzled addressing.
  // - CTA barrier.
  // - Reload in transposed view so the next 6 rounds are again subgroup-local.
  //
  // Phase D: Second six rounds
  // - Same compute helper as phase B.
  //
  // Phase E: Transpose back + store
  // - Spill to shared tile B (same swizzle), CTA barrier, reload canonical order.
  // - Store vectorized uint4 back to global memory.
  //
  // IN_PLACE controls whether stage 0 reads from src (out-of-place) or dst
  // (in-place path where caller passes src==dst).
  constexpr unsigned SUB_SIZE = 1u << 12;
  constexpr unsigned TILE_ROWS = 64;
  constexpr unsigned GROUPS = 4;
  constexpr unsigned ELEMS = 4;

  __shared__ uint4 smem_u4_a[SUB_SIZE >> 2];
  __shared__ uint4 smem_u4_b[SUB_SIZE >> 2];

  const unsigned tid = threadIdx.x;
  // Thread mapping:
  // - subgroup in [0,15]: selects one 256-value logical partition.
  // - lane16 in [0,15]: per-partition lane used for subgroup-local shuffles.
  const unsigned subgroup = tid >> 4;
  const unsigned lane16 = tid & 15u;
  const unsigned block_base = blockIdx.x << 12;

  const bf *__restrict__ load_ptr = IN_PLACE ? reinterpret_cast<const bf *>(dst) : src;
  bf regs[GROUPS][ELEMS];

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    // Each subgroup owns 4 x 64 values.
    // hi6 chooses one of the four 64-value rows in that subgroup.
    // lane16 * 4 gives contiguous 16-byte per-thread vector load.
    const unsigned hi6 = subgroup * GROUPS + g;
    const unsigned row_base = block_base + (hi6 * TILE_ROWS) + (lane16 * ELEMS);
    const uint4 packed = load_u4_mod<LOAD_MODIFIER>(load_ptr, row_base);
    regs[g][0] = bf(packed.x);
    regs[g][1] = bf(packed.y);
    regs[g][2] = bf(packed.z);
    regs[g][3] = bf(packed.w);
  }

  apply_6_rounds_pair_major_4groups(regs, lane16);

  // Spill first-half results in row-major logical view into swizzled shared tile A.
#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned row6 = subgroup * GROUPS + g;
    smem_u4_a[initial12_smem_u4_idx(row6, lane16)] = uint4{
        regs[g][0].limb,
        regs[g][1].limb,
        regs[g][2].limb,
        regs[g][3].limb,
    };
  }

  __syncthreads();

  // Reload in transposed view.
  // This is the key remap: rounds 6..11 now become subgroup-local again.
#pragma unroll
  for (unsigned e = 0; e < ELEMS; e++) {
    const unsigned row6 = lane16 * ELEMS + e;
    const uint4 packed = smem_u4_a[initial12_smem_u4_idx(row6, subgroup)];
    regs[0][e] = bf(packed.x);
    regs[1][e] = bf(packed.y);
    regs[2][e] = bf(packed.z);
    regs[3][e] = bf(packed.w);
  }

  apply_6_rounds_pair_major_4groups(regs, lane16);

  // Write second-half results to alternate tile B (same swizzled scheme).
#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned row6 = subgroup * GROUPS + g;
    smem_u4_b[initial12_smem_u4_idx(row6, lane16)] = uint4{
        regs[g][0].limb,
        regs[g][1].limb,
        regs[g][2].limb,
        regs[g][3].limb,
    };
  }

  __syncthreads();

  // Restore canonical row-major order before final global stores.
#pragma unroll
  for (unsigned e = 0; e < ELEMS; e++) {
    const unsigned row6 = lane16 * ELEMS + e;
    const uint4 packed = smem_u4_b[initial12_smem_u4_idx(row6, subgroup)];
    regs[0][e] = bf(packed.x);
    regs[1][e] = bf(packed.y);
    regs[2][e] = bf(packed.z);
    regs[3][e] = bf(packed.w);
  }

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned hi6 = subgroup * GROUPS + g;
    const unsigned row_base = block_base + (hi6 * TILE_ROWS) + (lane16 * ELEMS);
    const uint4 packed = uint4{
        regs[g][0].limb,
        regs[g][1].limb,
        regs[g][2].limb,
        regs[g][3].limb,
    };
    store_u4_mod<STORE_MODIFIER>(dst, row_base, packed);
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER, bool IN_PLACE>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_initial11(const bf *__restrict__ src, bf *__restrict__ dst) {
  // Initial11 kernel dataflow (handles one 2^11 chunk per CTA):
  //
  // Phase A: Global load (vectorized)
  // - 256 threads = 16 subgroups of 16 lanes.
  // - Each subgroup owns 2 rows of shape [k=64], each loaded as uint4 vectors.
  //
  // Phase B: First six rounds over k
  // - Run 6 rounds in 16-lane subgroup-local form.
  //
  // Phase C: Shared remap
  // - Spill to shared as (k, p) with row-dependent swizzle.
  //
  // Phase D: Last five rounds over p
  // - One warp processes 8 k-rows.
  // - For each fixed k row, apply 5 warp-local rounds across p=lane32.
  //
  // Phase E: Shared -> global store
  // - Reload canonical (p, k) ordering and store 2x uint4 per thread.
  //
  // IN_PLACE controls whether stage 0 reads from src (out-of-place) or dst
  // (in-place path where caller passes src==dst).
  constexpr unsigned SUB_SIZE = 1u << 11;
  constexpr unsigned K = 64;
  constexpr unsigned GROUPS = 2;
  constexpr unsigned ELEMS = 4;

  __shared__ bf smem[SUB_SIZE];

  const unsigned tid = threadIdx.x;
  const unsigned subgroup = tid >> 4;
  const unsigned lane16 = tid & 15u;
  const unsigned warp = tid >> 5;
  const unsigned lane32 = tid & 31u;
  const unsigned block_base = blockIdx.x << 11;
  const bf *__restrict__ load_ptr = IN_PLACE ? reinterpret_cast<const bf *>(dst) : src;

  // Map each 16-lane subgroup to two p rows. The odd subgroup in a warp is
  // shifted by +16 so both subgroup halves hit disjoint banks on spill/gather.
  const unsigned subgroup_warp = subgroup >> 1;
  const unsigned subgroup_parity = subgroup & 1u;
  const unsigned p_row0 = subgroup_warp + (subgroup_parity << 4);
  const unsigned k_local_base = lane16 * ELEMS;

  bf regs[GROUPS][ELEMS];

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned p = p_row0 + (g << 3);
    const unsigned row_base = block_base + (p * K) + k_local_base;
    const uint4 packed = load_u4_mod<LOAD_MODIFIER>(load_ptr, row_base);
    regs[g][0] = bf(packed.x);
    regs[g][1] = bf(packed.y);
    regs[g][2] = bf(packed.z);
    regs[g][3] = bf(packed.w);
  }

  apply_6_rounds_pair_major_2groups(regs, lane16);

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned p = p_row0 + (g << 3);
    smem[initial11_smem_idx(k_local_base + 0u, p)] = regs[g][0];
    smem[initial11_smem_idx(k_local_base + 1u, p)] = regs[g][1];
    smem[initial11_smem_idx(k_local_base + 2u, p)] = regs[g][2];
    smem[initial11_smem_idx(k_local_base + 3u, p)] = regs[g][3];
  }

  __syncthreads();

  // Each warp processes 8 k-rows; for each row, lane32 spans all 32 p values.
  const unsigned k_base = warp << 3;
  bf v00 = smem[initial11_smem_idx(k_base + 0u, lane32)];
  bf v01 = smem[initial11_smem_idx(k_base + 1u, lane32)];
  bf v10 = smem[initial11_smem_idx(k_base + 2u, lane32)];
  bf v11 = smem[initial11_smem_idx(k_base + 3u, lane32)];
  bf v20 = smem[initial11_smem_idx(k_base + 4u, lane32)];
  bf v21 = smem[initial11_smem_idx(k_base + 5u, lane32)];
  bf v30 = smem[initial11_smem_idx(k_base + 6u, lane32)];
  bf v31 = smem[initial11_smem_idx(k_base + 7u, lane32)];

  apply_5_rounds_warp64_pair_branchless_quad_packed(v00, v01, v10, v11, v20, v21, v30, v31, lane32);

  smem[initial11_smem_idx(k_base + 0u, lane32)] = v00;
  smem[initial11_smem_idx(k_base + 1u, lane32)] = v01;
  smem[initial11_smem_idx(k_base + 2u, lane32)] = v10;
  smem[initial11_smem_idx(k_base + 3u, lane32)] = v11;
  smem[initial11_smem_idx(k_base + 4u, lane32)] = v20;
  smem[initial11_smem_idx(k_base + 5u, lane32)] = v21;
  smem[initial11_smem_idx(k_base + 6u, lane32)] = v30;
  smem[initial11_smem_idx(k_base + 7u, lane32)] = v31;

  __syncthreads();

#pragma unroll
  for (unsigned g = 0; g < GROUPS; g++) {
    const unsigned p = p_row0 + (g << 3);
    const uint4 packed = uint4{
        smem[initial11_smem_idx(k_local_base + 0u, p)].limb,
        smem[initial11_smem_idx(k_local_base + 1u, p)].limb,
        smem[initial11_smem_idx(k_local_base + 2u, p)].limb,
        smem[initial11_smem_idx(k_local_base + 3u, p)].limb,
    };
    const unsigned row_base = block_base + (p * K) + k_local_base;
    store_u4_mod<STORE_MODIFIER>(dst, row_base, packed);
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial6(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Noninitial6 kernel dataflow (one 6-round stage):
  //
  // Goal:
  // - Preserve fully vectorized/coalesced global IO despite large stage stride.
  // - Perform all six rounds with warp-local math after one shared remap.
  //
  // Shared tile model:
  // - 64 (k) x 32 (partition) = 2048 BF values.
  // - k is the 6-round dimension.
  // - partition is the low-index lane among 32 independent butterflies.
  //
  // Phase A: Global load -> shared
  // - Each thread loads two uint4 vectors (32 bytes total per thread).
  // - Layout conversion writes into swizzled shared coordinates (k, p).
  //
  // Phase B: Warp-local rounds
  // - One warp handles 4 partitions.
  // - For each partition, load (k, p) and (k+32, p) registers.
  // - Rounds 0..4: branchless shuffle-xor butterfly helper.
  // - Round 5: local hi -= lo in registers.
  // - Store updated values back to shared.
  //
  // Phase C: Shared -> global store
  // - Gather shared values back into uint4 and store with selected cache policy.
  constexpr unsigned ROUNDS = 6;
  constexpr unsigned K = 1u << ROUNDS; // 64
  constexpr unsigned P = 32;
  constexpr unsigned LOW_TILE_LOG = 5u;
  constexpr unsigned VEC_ITERS = 2;

  __shared__ bf smem[K * P]; // 64 * 32 = 2048 BF values (8KB)

  const unsigned tid = threadIdx.x;
  // start_stage controls global stride between consecutive k values.
  const unsigned stride = 1u << start_stage;
  // blockIdx.x is decoded into:
  // - high: selects which 2^(start_stage+6) super-chunk this CTA belongs to.
  // - low_tile_id: selects one 32-value low-index window within stride.
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned tile = blockIdx.x;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + ROUNDS);

#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    // Vectorized load mapping:
    // - vec indexes one uint4 among the 512 vectors in this CTA tile.
    // - k chooses row in the 64-round domain.
    // - p0 chooses base partition in groups of 4 contiguous values.
    const unsigned vec = tid + (iter << 8);
    const unsigned k = vec >> 3;
    const unsigned p4 = vec & 7u;
    const unsigned p0 = p4 << 2;
    const unsigned row_base = block_base + (k << start_stage) + low_base + p0;
    const uint4 packed = load_u4_mod<LOAD_MODIFIER>(src, row_base);

    smem[noninitial6_smem_idx(k, p0 + 0u)] = bf(packed.x);
    smem[noninitial6_smem_idx(k, p0 + 1u)] = bf(packed.y);
    smem[noninitial6_smem_idx(k, p0 + 2u)] = bf(packed.z);
    smem[noninitial6_smem_idx(k, p0 + 3u)] = bf(packed.w);
  }

  __syncthreads();

  // Warp-local compute:
  // - each warp handles partitions {warp, warp+8, warp+16, warp+24}
  // - each lane handles one k in [0,31] and its partner k+32
  // - together this covers all 64 k-values for those 4 partitions.
  const unsigned warp = tid >> 5;
  const unsigned lane = tid & 31u;
  const unsigned p0 = warp + 0u;
  const unsigned p1 = warp + 8u;
  const unsigned p2 = warp + 16u;
  const unsigned p3 = warp + 24u;

  const unsigned idx00 = noninitial6_smem_idx(lane + 0u, p0);
  const unsigned idx01 = noninitial6_smem_idx(lane + 32u, p0);
  const unsigned idx10 = noninitial6_smem_idx(lane + 0u, p1);
  const unsigned idx11 = noninitial6_smem_idx(lane + 32u, p1);
  const unsigned idx20 = noninitial6_smem_idx(lane + 0u, p2);
  const unsigned idx21 = noninitial6_smem_idx(lane + 32u, p2);
  const unsigned idx30 = noninitial6_smem_idx(lane + 0u, p3);
  const unsigned idx31 = noninitial6_smem_idx(lane + 32u, p3);

  bf v00 = smem[idx00];
  bf v01 = smem[idx01];
  bf v10 = smem[idx10];
  bf v11 = smem[idx11];
  bf v20 = smem[idx20];
  bf v21 = smem[idx21];
  bf v30 = smem[idx30];
  bf v31 = smem[idx31];

  // Rounds 0..4 via branchless XOR-shuffle butterfly.
  apply_5_rounds_warp64_pair_branchless_quad(v00, v01, v10, v11, v20, v21, v30, v31, lane);
  // Round 5 is local high-minus-low inside each pair.
  v01 = bf::sub(v01, v00);
  v11 = bf::sub(v11, v10);
  v21 = bf::sub(v21, v20);
  v31 = bf::sub(v31, v30);

  smem[idx00] = v00;
  smem[idx01] = v01;
  smem[idx10] = v10;
  smem[idx11] = v11;
  smem[idx20] = v20;
  smem[idx21] = v21;
  smem[idx30] = v30;
  smem[idx31] = v31;

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    const unsigned vec = tid + (iter << 8);
    const unsigned k = vec >> 3;
    const unsigned p4 = vec & 7u;
    const unsigned p0 = p4 << 2;
    const unsigned row_base = block_base + (k << start_stage) + low_base + p0;

    const uint4 packed = uint4{
        smem[noninitial6_smem_idx(k, p0 + 0u)].limb,
        smem[noninitial6_smem_idx(k, p0 + 1u)].limb,
        smem[noninitial6_smem_idx(k, p0 + 2u)].limb,
        smem[noninitial6_smem_idx(k, p0 + 3u)].limb,
    };
    store_u4_mod<STORE_MODIFIER>(dst, row_base, packed);
  }
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_initial12_out_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // Out-of-place initial12 policy: ld.cs + st.wt.
  hypercube_evals_into_coeffs_bitrev_initial12<ld_modifier::cs, st_modifier::wt, false>(src, dst);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_initial12_in_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // In-place initial12 policy: ld.cg + st.wt.
  (void)src;
  hypercube_evals_into_coeffs_bitrev_initial12<ld_modifier::cg, st_modifier::wt, true>(src, dst);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_initial11_out_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // Out-of-place initial11 policy: ld.cs + st.wt.
  hypercube_evals_into_coeffs_bitrev_initial11<ld_modifier::cs, st_modifier::wt, false>(src, dst);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_initial11_in_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // In-place initial11 policy: ld.cg + st.wt.
  (void)src;
  hypercube_evals_into_coeffs_bitrev_initial11<ld_modifier::cg, st_modifier::wt, true>(src, dst);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage2_out_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Stage2 out-of-place policy: ld.cs + st.wt.
  hypercube_evals_into_coeffs_bitrev_noninitial6<ld_modifier::cs, st_modifier::wt>(src, dst, start_stage);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage2_in_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Stage2 in-place policy: ld.ca + st.wt.
  hypercube_evals_into_coeffs_bitrev_noninitial6<ld_modifier::ca, st_modifier::wt>(src, dst, start_stage);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage3_out_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Stage #3 kernel (out-of-place): ld.cs + st.cs.
  hypercube_evals_into_coeffs_bitrev_noninitial6<ld_modifier::cs, st_modifier::cs>(src, dst, start_stage);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage3_in_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Stage #3 kernel (in-place): ld.ca + st.cs.
  hypercube_evals_into_coeffs_bitrev_noninitial6<ld_modifier::ca, st_modifier::cs>(src, dst, start_stage);
}

} // namespace airbender::ops::hypercube
