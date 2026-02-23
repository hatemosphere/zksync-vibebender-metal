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

DEVICE_FORCEINLINE unsigned initial14_smem_idx(const unsigned k, const unsigned p) {
  // Initial14 shared layout helper for a 128 x 128 scalar tile.
  //
  // Layout decomposition:
  // - p = (p_hi2 << 2) | p_lo2
  // - p_hi2 in [0,31] is XOR-swizzled with (k >> 2)
  //
  // Why this layout:
  // - Row pass writes p in contiguous 4-value vectors (p_lo2 local, p_hi2 = lane32).
  // - Column pass reads fixed p while lane32 spans k in strides of 4.
  // - Using (k >> 2) keeps both row and column passes bank-friendly.
  const unsigned p_lo2 = p & 3u;
  const unsigned p_hi5 = p >> 2;
  return (k << 7) | (p_lo2 << 5) | (p_hi5 ^ (k >> 2));
}

DEVICE_FORCEINLINE unsigned initial13_smem_idx(const unsigned k, const unsigned p) {
  // Initial13 shared layout helper for a 64 x 128 scalar tile.
  //
  // Layout decomposition:
  // - p = (p_hi5 << 2) | p_lo2
  // - p_hi5 in [0,31], p_lo2 in [0,3]
  //
  // Bank swizzle:
  // - base term: p_hi5 ^ (k >> 2)
  // - extra term: ((p >> 5) << 4) toggles bank-half across 32-step p stripes
  //
  // Why this layout:
  // - Row pass keeps lane32 access as a bijection over banks.
  // - k-pass uses two 16-lane subgroups per warp over p values separated by 32;
  //   the extra bank-half toggle prevents systematic cross-subgroup collisions.
  const unsigned p_lo2 = p & 3u;
  const unsigned p_hi5 = p >> 2;
  const unsigned bank_swizzle = (p >> 5) << 4;
  return (k << 7) | (p_lo2 << 5) | ((p_hi5 ^ (k >> 2) ^ bank_swizzle) & 31u);
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

DEVICE_FORCEINLINE unsigned noninitial5_smem_idx(const unsigned k, const unsigned p) {
  // Noninitial5 shared layout helper.
  //
  // Logical tile shape:
  // - k dimension: 32 rows (round domain)
  // - p dimension: 32 partitions
  //
  // Physical index:
  // - row-major base: k * 32 + p
  // - swizzled partition: p ^ (k & 31)
  //
  // Why this swizzle:
  // - Warp compute touches fixed partition p while lane32 spans k in [0,31].
  // - The xor keeps lane-correlated accesses bank-friendly and bijective.
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

DEVICE_FORCEINLINE void apply_7_rounds_pair_major_warp32_quad(bf (&vals)[4], const unsigned lane32) {
  // Apply 7 rounds for a 128-point domain represented as:
  // - lane32: low 5 bits
  // - local quad index in vals[0..3]: high 2 bits
  //
  // Round order:
  // - rounds 0..1 are lane-local over vals[0..3]
  // - rounds 2..6 are lane-xor shuffles across 32-lane warps
  vals[1] = bf::sub(vals[1], vals[0]);
  vals[3] = bf::sub(vals[3], vals[2]);
  vals[2] = bf::sub(vals[2], vals[0]);
  vals[3] = bf::sub(vals[3], vals[1]);

#pragma unroll
  for (unsigned bit = 0; bit < 5; bit++) {
    const unsigned lane_bit = 1u << bit;
    const bool high_lane = (lane32 & lane_bit) != 0u;

    const unsigned long long pair01 =
        (static_cast<unsigned long long>(vals[1].limb) << 32) | static_cast<unsigned long long>(vals[0].limb);
    const unsigned long long pair23 =
        (static_cast<unsigned long long>(vals[3].limb) << 32) | static_cast<unsigned long long>(vals[2].limb);
    const unsigned long long peer01 = __shfl_xor_sync(0xFFFFFFFFu, pair01, lane_bit, 32);
    const unsigned long long peer23 = __shfl_xor_sync(0xFFFFFFFFu, pair23, lane_bit, 32);

    if (high_lane) {
      apply_pair_major_high_update_from_peers(vals, peer01, peer23);
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

DEVICE_FORCEINLINE void apply_5_rounds_warp32_branchless_quad(
    bf &a,
    bf &b,
    bf &c,
    bf &d,
    const unsigned lane32) {
  // Apply rounds 0..4 for noninitial5 on four independent scalar values.
  //
  // This is branchless and follows the same low/high lane selection contract as
  // the pair-major helpers:
  // - low lane keeps lo
  // - high lane gets hi = hi_pre - lo
#pragma unroll
  for (unsigned bit = 0; bit < 5; bit++) {
    const unsigned mask = 1u << bit;
    const unsigned lane_bit = (lane32 >> bit) & 1u;
    const unsigned lane_mask = 0u - lane_bit;

    const unsigned a_self = a.limb;
    const unsigned b_self = b.limb;
    const unsigned c_self = c.limb;
    const unsigned d_self = d.limb;

    const unsigned a_peer = __shfl_xor_sync(0xFFFFFFFFu, a_self, mask, 32);
    const unsigned b_peer = __shfl_xor_sync(0xFFFFFFFFu, b_self, mask, 32);
    const unsigned c_peer = __shfl_xor_sync(0xFFFFFFFFu, c_self, mask, 32);
    const unsigned d_peer = __shfl_xor_sync(0xFFFFFFFFu, d_self, mask, 32);

    const unsigned a_lo = select_u32(a_self, a_peer, lane_mask);
    const unsigned a_hi_pre = select_u32(a_peer, a_self, lane_mask);
    const unsigned a_hi = bf::sub(bf(a_hi_pre), bf(a_lo)).limb;
    a = bf(select_u32(a_lo, a_hi, lane_mask));

    const unsigned b_lo = select_u32(b_self, b_peer, lane_mask);
    const unsigned b_hi_pre = select_u32(b_peer, b_self, lane_mask);
    const unsigned b_hi = bf::sub(bf(b_hi_pre), bf(b_lo)).limb;
    b = bf(select_u32(b_lo, b_hi, lane_mask));

    const unsigned c_lo = select_u32(c_self, c_peer, lane_mask);
    const unsigned c_hi_pre = select_u32(c_peer, c_self, lane_mask);
    const unsigned c_hi = bf::sub(bf(c_hi_pre), bf(c_lo)).limb;
    c = bf(select_u32(c_lo, c_hi, lane_mask));

    const unsigned d_lo = select_u32(d_self, d_peer, lane_mask);
    const unsigned d_hi_pre = select_u32(d_peer, d_self, lane_mask);
    const unsigned d_hi = bf::sub(bf(d_hi_pre), bf(d_lo)).limb;
    d = bf(select_u32(d_lo, d_hi, lane_mask));
  }
}

DEVICE_FORCEINLINE void apply_5_rounds_warp32_branchless_scalar(bf &v, const unsigned lane32) {
  // Apply rounds 0..4 for one scalar value distributed over 32 lanes.
  //
  // Branchless low/high lane select:
  // - low lane keeps lo
  // - high lane gets hi = hi_pre - lo
#pragma unroll
  for (unsigned bit = 0; bit < 5; bit++) {
    const unsigned mask = 1u << bit;
    const unsigned lane_bit = (lane32 >> bit) & 1u;
    const unsigned lane_mask = 0u - lane_bit;

    const unsigned self = v.limb;
    const unsigned peer = __shfl_xor_sync(0xFFFFFFFFu, self, mask, 32);
    const unsigned lo = select_u32(self, peer, lane_mask);
    const unsigned hi_pre = select_u32(peer, self, lane_mask);
    const unsigned hi = bf::sub(bf(hi_pre), bf(lo)).limb;
    v = bf(select_u32(lo, hi, lane_mask));
  }
}

DEVICE_FORCEINLINE void apply_4_rounds_warp16_branchless_scalar(bf &v, const unsigned lane16) {
  // Apply 4 rounds on a 16-point domain carried by one 16-lane subgroup.
  //
  // Branchless low/high lane select:
  // - low lane keeps lo
  // - high lane gets hi = hi_pre - lo
#pragma unroll
  for (unsigned bit = 0; bit < 4; bit++) {
    const unsigned mask = 1u << bit;
    const unsigned lane_bit = (lane16 >> bit) & 1u;
    const unsigned lane_mask = 0u - lane_bit;

    const unsigned self = v.limb;
    const unsigned peer = __shfl_xor_sync(0xFFFFFFFFu, self, mask, 16);
    const unsigned lo = select_u32(self, peer, lane_mask);
    const unsigned hi_pre = select_u32(peer, self, lane_mask);
    const unsigned hi = bf::sub(bf(hi_pre), bf(lo)).limb;
    v = bf(select_u32(lo, hi, lane_mask));
  }
}

DEVICE_FORCEINLINE unsigned noninitial10_half_p8_smem_idx(const unsigned k, const unsigned p_local) {
  // Noninitial10 shared layout helper for one 512 x 8 half-tile.
  //
  // k decomposition:
  // - k = (kh << 5) | kl
  // - kh in [0,15], kl in [0,31]
  //
  // Layout decomposition:
  // - idx = (kh << 8) | (p_local << 5) | (kl ^ kh)
  //
  // Why this layout:
  // - Pass #1 (fixed kh,p_local; lane32 spans kl): (kl ^ kh) permutes 0..31,
  //   so the warp issues conflict-free shared accesses.
  // - Pass #2 (fixed kl,p_local; lane16 spans kh): bank index is (kl ^ kh),
  //   so each 16-lane subgroup is conflict-free.
  const unsigned kh = k >> 5;
  const unsigned kl = k & 31u;
  return (kh << 8) | (p_local << 5) | (kl ^ kh);
}

DEVICE_FORCEINLINE unsigned noninitial8_half_p8_smem_idx(const unsigned k, const unsigned p_local) {
  // Noninitial8 shared layout helper for one 128 x 8 half-tile.
  //
  // k decomposition:
  // - k = (kh << 5) | kl
  // - kh in [0,3], kl in [0,31]
  //
  // Bank swizzle:
  // - base term: kl ^ kh
  // - extra term: ((p_local >> 2) << 4) separates p groups {0..3} and {4..7}
  //
  // Why this layout:
  // - Keeps warp32 compute passes bank-friendly.
  // - Avoids paired-thread conflicts during load/store of p_local0 in {0,4}.
  const unsigned kh = k >> 5;
  const unsigned kl = k & 31u;
  const unsigned bank_swizzle = (p_local >> 2) << 4;
  return (kh << 8) | (p_local << 5) | ((kl ^ kh ^ bank_swizzle) & 31u);
}

DEVICE_FORCEINLINE unsigned noninitial11_full_p8_smem_idx(const unsigned k, const unsigned p_local) {
  // Noninitial11 shared layout helper for one full 2048 x 8 tile.
  //
  // k decomposition:
  // - k = (kh << 5) | kl
  // - kh in [0,63], kl in [0,31]
  //
  // Layout decomposition:
  // - idx = (kh << 8) | (p_local << 5) | (kl ^ (kh & 31))
  //
  // Why this layout:
  // - First 5 rounds fix (kh, p_local) and lane32 spans kl.
  // - Last 5 rounds fix (kl, p_local) and lane32 spans kh half.
  // - The xor swizzle keeps both passes bank-friendly.
  const unsigned kh = k >> 5;
  const unsigned kl = k & 31u;
  return (kh << 8) | (p_local << 5) | (kl ^ (kh & 31u));
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER, bool IN_PLACE, unsigned BLOCK_THREADS>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_initial14(const bf *__restrict__ src, bf *__restrict__ dst) {
  // Initial14 kernel dataflow (one 2^14 chunk per CTA):
  //
  // - Tile is interpreted as [k=128][p=128].
  // - Pass 1: for each k row, run 7 rounds over p.
  // - CTA barrier.
  // - Pass 2: for each p column, run 7 rounds over k.
  // - CTA barrier.
  // - Store canonical row-major order.
  //
  // Global IO is fully vectorized/coalesced via uint4 loads/stores.
  // Shared layout uses initial14_smem_idx() to keep row/column passes bank-friendly.
  constexpr unsigned SUB_SIZE = 1u << 14;
  constexpr unsigned K = 128;
  constexpr unsigned P = 128;
  constexpr unsigned WARP_COUNT = BLOCK_THREADS >> 5;
  constexpr unsigned ITERS = K / WARP_COUNT;
  static_assert(BLOCK_THREADS % 32u == 0u, "BLOCK_THREADS must be warp-multiple");
  static_assert(K % WARP_COUNT == 0u, "warp count must divide K");

  extern __shared__ bf smem[];

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane32 = tid & 31u;
  const unsigned block_base = blockIdx.x << 14;
  const bf *__restrict__ load_ptr = IN_PLACE ? reinterpret_cast<const bf *>(dst) : src;
  if (blockDim.x != BLOCK_THREADS) {
    return;
  }

#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned k = warp + (iter * WARP_COUNT);
    const unsigned p_base = lane32 << 2;
    const unsigned row_base = block_base + (k * P) + p_base;
    const uint4 packed = load_u4_mod<LOAD_MODIFIER>(load_ptr, row_base);

    bf vals[4] = {
        bf(packed.x),
        bf(packed.y),
        bf(packed.z),
        bf(packed.w),
    };
    apply_7_rounds_pair_major_warp32_quad(vals, lane32);

    smem[initial14_smem_idx(k, p_base + 0u)] = vals[0];
    smem[initial14_smem_idx(k, p_base + 1u)] = vals[1];
    smem[initial14_smem_idx(k, p_base + 2u)] = vals[2];
    smem[initial14_smem_idx(k, p_base + 3u)] = vals[3];
  }

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned p = warp + (iter * WARP_COUNT);
    const unsigned k_base = lane32 << 2;
    bf vals[4] = {
        smem[initial14_smem_idx(k_base + 0u, p)],
        smem[initial14_smem_idx(k_base + 1u, p)],
        smem[initial14_smem_idx(k_base + 2u, p)],
        smem[initial14_smem_idx(k_base + 3u, p)],
    };
    apply_7_rounds_pair_major_warp32_quad(vals, lane32);

    smem[initial14_smem_idx(k_base + 0u, p)] = vals[0];
    smem[initial14_smem_idx(k_base + 1u, p)] = vals[1];
    smem[initial14_smem_idx(k_base + 2u, p)] = vals[2];
    smem[initial14_smem_idx(k_base + 3u, p)] = vals[3];
  }

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned k = warp + (iter * WARP_COUNT);
    const unsigned p_base = lane32 << 2;
    const uint4 packed = uint4{
        smem[initial14_smem_idx(k, p_base + 0u)].limb,
        smem[initial14_smem_idx(k, p_base + 1u)].limb,
        smem[initial14_smem_idx(k, p_base + 2u)].limb,
        smem[initial14_smem_idx(k, p_base + 3u)].limb,
    };
    const unsigned row_base = block_base + (k * P) + p_base;
    store_u4_mod<STORE_MODIFIER>(dst, row_base, packed);
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER, bool IN_PLACE, unsigned BLOCK_THREADS>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_initial15(const bf *__restrict__ src, bf *__restrict__ dst) {
  // Initial15 kernel dataflow (one 2^15 chunk per CTA) with 64KB shared footprint:
  //
  // - Split chunk into low/high 2^14 halves.
  // - Compute low_out  = T14(low_raw) via initial14-style 7+7 passes.
  // - Compute high_out = T14(high_raw - low_raw) via the same 7+7 passes.
  //
  // This realizes the final 15th round without a full 128KB tile:
  //   T15(low, high) = (T14(low), T14(high) - T14(low))
  //                  = (T14(low), T14(high - low)).
  //
  // Global IO remains vectorized/coalesced (uint4). Shared access pattern
  // reuses initial14_smem_idx(), preserving bank-friendly row/column passes.
  constexpr unsigned SUB_SIZE = 1u << 15;
  constexpr unsigned HALF_SUB_SIZE = 1u << 14;
  constexpr unsigned K = 128;
  constexpr unsigned P = 128;
  constexpr unsigned WARP_COUNT = BLOCK_THREADS >> 5;
  constexpr unsigned ITERS = K / WARP_COUNT;
  static_assert(BLOCK_THREADS % 32u == 0u, "BLOCK_THREADS must be warp-multiple");
  static_assert(K % WARP_COUNT == 0u, "warp count must divide K");

  extern __shared__ bf smem[];

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane32 = tid & 31u;
  const unsigned block_base = blockIdx.x << 15;
  const unsigned low_half_base = block_base;
  const unsigned high_half_base = block_base + HALF_SUB_SIZE;
  const bf *__restrict__ load_ptr = IN_PLACE ? reinterpret_cast<const bf *>(dst) : src;
  if (blockDim.x != BLOCK_THREADS) {
    return;
  }

  uint4 low_raw_packed[ITERS];

  // Pass A: low half input -> first 7 rounds over p.
#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned k = warp + (iter * WARP_COUNT);
    const unsigned p_base = lane32 << 2;
    const unsigned row_base = low_half_base + (k * P) + p_base;
    const uint4 packed = load_u4_mod<LOAD_MODIFIER>(load_ptr, row_base);
    low_raw_packed[iter] = packed;

    bf vals[4] = {
        bf(packed.x),
        bf(packed.y),
        bf(packed.z),
        bf(packed.w),
    };
    apply_7_rounds_pair_major_warp32_quad(vals, lane32);

    smem[initial14_smem_idx(k, p_base + 0u)] = vals[0];
    smem[initial14_smem_idx(k, p_base + 1u)] = vals[1];
    smem[initial14_smem_idx(k, p_base + 2u)] = vals[2];
    smem[initial14_smem_idx(k, p_base + 3u)] = vals[3];
  }

  __syncthreads();

  // Pass B: low half second 7 rounds over k.
#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned p = warp + (iter * WARP_COUNT);
    const unsigned k_base = lane32 << 2;
    bf vals[4] = {
        smem[initial14_smem_idx(k_base + 0u, p)],
        smem[initial14_smem_idx(k_base + 1u, p)],
        smem[initial14_smem_idx(k_base + 2u, p)],
        smem[initial14_smem_idx(k_base + 3u, p)],
    };
    apply_7_rounds_pair_major_warp32_quad(vals, lane32);

    smem[initial14_smem_idx(k_base + 0u, p)] = vals[0];
    smem[initial14_smem_idx(k_base + 1u, p)] = vals[1];
    smem[initial14_smem_idx(k_base + 2u, p)] = vals[2];
    smem[initial14_smem_idx(k_base + 3u, p)] = vals[3];
  }

  __syncthreads();

  // Write low_out.
#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned k = warp + (iter * WARP_COUNT);
    const unsigned p_base = lane32 << 2;
    const unsigned row_base = low_half_base + (k * P) + p_base;
    const uint4 packed = uint4{
        smem[initial14_smem_idx(k, p_base + 0u)].limb,
        smem[initial14_smem_idx(k, p_base + 1u)].limb,
        smem[initial14_smem_idx(k, p_base + 2u)].limb,
        smem[initial14_smem_idx(k, p_base + 3u)].limb,
    };
    store_u4_mod<STORE_MODIFIER>(dst, row_base, packed);
  }

  // Ensure all low-half shared reads are complete before reusing smem.
  __syncthreads();

  // Pass C: diff input (high_raw - low_raw) -> first 7 rounds over p.
#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned k = warp + (iter * WARP_COUNT);
    const unsigned p_base = lane32 << 2;
    const unsigned row_base = high_half_base + (k * P) + p_base;
    const uint4 high_packed = load_u4_mod<LOAD_MODIFIER>(load_ptr, row_base);
    const uint4 diff_packed = uint4{
        bf::sub(bf(high_packed.x), bf(low_raw_packed[iter].x)).limb,
        bf::sub(bf(high_packed.y), bf(low_raw_packed[iter].y)).limb,
        bf::sub(bf(high_packed.z), bf(low_raw_packed[iter].z)).limb,
        bf::sub(bf(high_packed.w), bf(low_raw_packed[iter].w)).limb,
    };

    bf vals[4] = {
        bf(diff_packed.x),
        bf(diff_packed.y),
        bf(diff_packed.z),
        bf(diff_packed.w),
    };
    apply_7_rounds_pair_major_warp32_quad(vals, lane32);

    smem[initial14_smem_idx(k, p_base + 0u)] = vals[0];
    smem[initial14_smem_idx(k, p_base + 1u)] = vals[1];
    smem[initial14_smem_idx(k, p_base + 2u)] = vals[2];
    smem[initial14_smem_idx(k, p_base + 3u)] = vals[3];
  }

  __syncthreads();

  // Pass D: diff half second 7 rounds over k.
#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned p = warp + (iter * WARP_COUNT);
    const unsigned k_base = lane32 << 2;
    bf vals[4] = {
        smem[initial14_smem_idx(k_base + 0u, p)],
        smem[initial14_smem_idx(k_base + 1u, p)],
        smem[initial14_smem_idx(k_base + 2u, p)],
        smem[initial14_smem_idx(k_base + 3u, p)],
    };
    apply_7_rounds_pair_major_warp32_quad(vals, lane32);

    smem[initial14_smem_idx(k_base + 0u, p)] = vals[0];
    smem[initial14_smem_idx(k_base + 1u, p)] = vals[1];
    smem[initial14_smem_idx(k_base + 2u, p)] = vals[2];
    smem[initial14_smem_idx(k_base + 3u, p)] = vals[3];
  }

  __syncthreads();

  // Write high_out = T14(high_raw - low_raw).
#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned k = warp + (iter * WARP_COUNT);
    const unsigned p_base = lane32 << 2;
    const unsigned row_base = high_half_base + (k * P) + p_base;
    const uint4 packed = uint4{
        smem[initial14_smem_idx(k, p_base + 0u)].limb,
        smem[initial14_smem_idx(k, p_base + 1u)].limb,
        smem[initial14_smem_idx(k, p_base + 2u)].limb,
        smem[initial14_smem_idx(k, p_base + 3u)].limb,
    };
    store_u4_mod<STORE_MODIFIER>(dst, row_base, packed);
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER, bool IN_PLACE>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_initial13(const bf *__restrict__ src, bf *__restrict__ dst) {
  // Initial13 kernel dataflow (one 2^13 chunk per CTA):
  //
  // - Logical tile is [k=64][p=128].
  // - First 7 rounds are computed warp-local across p (lane32 domain).
  // - Results are remapped in shared with initial13_smem_idx swizzle.
  // - Last 6 rounds are computed subgroup-local across k (lane16 domain).
  //
  // Global IO remains fully vectorized/coalesced (uint4 loads/stores).
  constexpr unsigned SUB_SIZE = 1u << 13;
  constexpr unsigned K = 64;
  constexpr unsigned P = 128;
  constexpr unsigned BLOCK_THREADS = 1024;
  constexpr unsigned WARP_COUNT = BLOCK_THREADS >> 5;
  constexpr unsigned ITERS = K / WARP_COUNT;
  constexpr unsigned SUBGROUP_COUNT = BLOCK_THREADS >> 4;
  constexpr unsigned HALF_P = P >> 1;
  constexpr unsigned P_ITERS = HALF_P / SUBGROUP_COUNT;
  static_assert(K % WARP_COUNT == 0u, "K must be divisible by warp count");
  static_assert(HALF_P % SUBGROUP_COUNT == 0u, "HALF_P must be divisible by subgroup count");

  __shared__ bf smem[SUB_SIZE];

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane32 = tid & 31u;
  const unsigned subgroup = tid >> 4;
  const unsigned lane16 = tid & 15u;
  const unsigned block_base = blockIdx.x << 13;
  const bf *__restrict__ load_ptr = IN_PLACE ? reinterpret_cast<const bf *>(dst) : src;
  if (blockDim.x != BLOCK_THREADS) {
    return;
  }

  // First 7 rounds over p (size 128), row-major load/store.
#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned k = warp + (iter * WARP_COUNT);
    const unsigned p_base = lane32 << 2;
    const unsigned row_base = block_base + (k * P) + p_base;
    const uint4 packed = load_u4_mod<LOAD_MODIFIER>(load_ptr, row_base);
    bf vals[4] = {
        bf(packed.x),
        bf(packed.y),
        bf(packed.z),
        bf(packed.w),
    };
    apply_7_rounds_pair_major_warp32_quad(vals, lane32);

    smem[initial13_smem_idx(k, p_base + 0u)] = vals[0];
    smem[initial13_smem_idx(k, p_base + 1u)] = vals[1];
    smem[initial13_smem_idx(k, p_base + 2u)] = vals[2];
    smem[initial13_smem_idx(k, p_base + 3u)] = vals[3];
  }

  __syncthreads();

  // Last 6 rounds over k (size 64), two p-columns per subgroup via lane16 helper.
  const unsigned subgroup_warp = subgroup >> 1;
  const unsigned subgroup_parity = subgroup & 1u;
  const unsigned subgroup_swizzled = subgroup_warp + (subgroup_parity * (SUBGROUP_COUNT >> 1));
  const unsigned k_local_base = lane16 << 2;

#pragma unroll
  for (unsigned iter = 0; iter < P_ITERS; iter++) {
    const unsigned p_base = iter * SUBGROUP_COUNT;
    const unsigned p0 = p_base + subgroup_swizzled;
    const unsigned p1 = p0 + HALF_P;

    bf regs[2][4] = {
        {
            smem[initial13_smem_idx(k_local_base + 0u, p0)],
            smem[initial13_smem_idx(k_local_base + 1u, p0)],
            smem[initial13_smem_idx(k_local_base + 2u, p0)],
            smem[initial13_smem_idx(k_local_base + 3u, p0)],
        },
        {
            smem[initial13_smem_idx(k_local_base + 0u, p1)],
            smem[initial13_smem_idx(k_local_base + 1u, p1)],
            smem[initial13_smem_idx(k_local_base + 2u, p1)],
            smem[initial13_smem_idx(k_local_base + 3u, p1)],
        },
    };

    apply_6_rounds_pair_major_2groups(regs, lane16);

    smem[initial13_smem_idx(k_local_base + 0u, p0)] = regs[0][0];
    smem[initial13_smem_idx(k_local_base + 1u, p0)] = regs[0][1];
    smem[initial13_smem_idx(k_local_base + 2u, p0)] = regs[0][2];
    smem[initial13_smem_idx(k_local_base + 3u, p0)] = regs[0][3];

    smem[initial13_smem_idx(k_local_base + 0u, p1)] = regs[1][0];
    smem[initial13_smem_idx(k_local_base + 1u, p1)] = regs[1][1];
    smem[initial13_smem_idx(k_local_base + 2u, p1)] = regs[1][2];
    smem[initial13_smem_idx(k_local_base + 3u, p1)] = regs[1][3];
  }

  __syncthreads();

  // Restore row-major and store.
#pragma unroll
  for (unsigned iter = 0; iter < ITERS; iter++) {
    const unsigned k = warp + (iter * WARP_COUNT);
    const unsigned p_base = lane32 << 2;
    const unsigned row_base = block_base + (k * P) + p_base;
    const uint4 packed = uint4{
        smem[initial13_smem_idx(k, p_base + 0u)].limb,
        smem[initial13_smem_idx(k, p_base + 1u)].limb,
        smem[initial13_smem_idx(k, p_base + 2u)].limb,
        smem[initial13_smem_idx(k, p_base + 3u)].limb,
    };
    store_u4_mod<STORE_MODIFIER>(dst, row_base, packed);
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
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial5_tile(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage,
    const unsigned tile);

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial5_impl(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  hypercube_evals_into_coeffs_bitrev_noninitial5_tile<LOAD_MODIFIER, STORE_MODIFIER>(
      src, dst, start_stage, blockIdx.x);
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial5_tile(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage,
    const unsigned tile) {
  // Noninitial5 kernel dataflow (one 5-round stage):
  //
  // Goal:
  // - Preserve fully vectorized/coalesced global IO despite stage stride.
  // - Perform all five rounds with warp-local math after one shared remap.
  //
  // Shared tile model:
  // - 32 (k) x 32 (partition) = 1024 BF values.
  // - k is the 5-round dimension.
  // - partition is the low-index lane among 32 independent butterflies.
  //
  // Phase A: Global load -> shared
  // - 256 threads, one uint4 load per thread.
  // - Layout conversion writes into swizzled shared coordinates (k, p).
  //
  // Phase B: Warp-local rounds
  // - One warp handles 4 partitions.
  // - Each lane owns one k in [0,31].
  // - Run 5 branchless shuffle-xor rounds.
  //
  // Phase C: Shared -> global store
  // - Gather shared values back into uint4 and store with selected cache policy.
  constexpr unsigned ROUNDS = 5;
  constexpr unsigned K = 1u << ROUNDS; // 32
  constexpr unsigned P = 32;
  constexpr unsigned LOW_TILE_LOG = 5u;

  __shared__ bf smem[K * P]; // 32 * 32 = 1024 BF values (4KB)

  const unsigned tid = threadIdx.x;
  const unsigned stride = 1u << start_stage;
  // blockIdx.x is decoded into:
  // - high: selects which 2^(start_stage+5) super-chunk this CTA belongs to.
  // - low_tile_id: selects one 32-value low-index window within stride.
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + ROUNDS);

  // Vectorized load mapping:
  // - vec indexes one uint4 among the 256 vectors in this CTA tile.
  // - k chooses row in the 32-round domain.
  // - p0 chooses base partition in groups of 4 contiguous values.
  const unsigned vec = tid;
  const unsigned k = vec >> 3;
  const unsigned p4 = vec & 7u;
  const unsigned p0 = p4 << 2;
  const unsigned row_base = block_base + (k << start_stage) + low_base + p0;
  const uint4 packed = load_u4_mod<LOAD_MODIFIER>(src, row_base);

  smem[noninitial5_smem_idx(k, p0 + 0u)] = bf(packed.x);
  smem[noninitial5_smem_idx(k, p0 + 1u)] = bf(packed.y);
  smem[noninitial5_smem_idx(k, p0 + 2u)] = bf(packed.z);
  smem[noninitial5_smem_idx(k, p0 + 3u)] = bf(packed.w);

  __syncthreads();

  // Warp-local compute:
  // - each warp handles partitions {warp, warp+8, warp+16, warp+24}
  // - each lane handles one k in [0,31]
  const unsigned warp = tid >> 5;
  const unsigned lane = tid & 31u;
  const unsigned pp0 = warp + 0u;
  const unsigned pp1 = warp + 8u;
  const unsigned pp2 = warp + 16u;
  const unsigned pp3 = warp + 24u;

  const unsigned idx0 = noninitial5_smem_idx(lane, pp0);
  const unsigned idx1 = noninitial5_smem_idx(lane, pp1);
  const unsigned idx2 = noninitial5_smem_idx(lane, pp2);
  const unsigned idx3 = noninitial5_smem_idx(lane, pp3);

  bf v0 = smem[idx0];
  bf v1 = smem[idx1];
  bf v2 = smem[idx2];
  bf v3 = smem[idx3];

  apply_5_rounds_warp32_branchless_quad(v0, v1, v2, v3, lane);

  smem[idx0] = v0;
  smem[idx1] = v1;
  smem[idx2] = v2;
  smem[idx3] = v3;

  __syncthreads();

  const uint4 out = uint4{
      smem[noninitial5_smem_idx(k, p0 + 0u)].limb,
      smem[noninitial5_smem_idx(k, p0 + 1u)].limb,
      smem[noninitial5_smem_idx(k, p0 + 2u)].limb,
      smem[noninitial5_smem_idx(k, p0 + 3u)].limb,
  };
  store_u4_mod<STORE_MODIFIER>(dst, row_base, out);
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial5_impl_x2(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Two-tile noninitial5 path specialized for fixed start_stage=11.
  (void)start_stage;
  constexpr unsigned ROUNDS = 5;
  constexpr unsigned START_STAGE = 11;
  constexpr unsigned K = 1u << ROUNDS; // 32
  constexpr unsigned LOW_TILE_LOG = 5u;
  constexpr unsigned LOW_TILES = 1u << (START_STAGE - LOW_TILE_LOG); // 64
  constexpr unsigned LOW_TILES_MASK = LOW_TILES - 1u;

  __shared__ bf smem[K * 32u]; // 32 * 32 = 1024 BF values (4KB)

  const unsigned tid = threadIdx.x;
  const unsigned vec = tid;
  const unsigned k = vec >> 3;
  const unsigned p4 = vec & 7u;
  const unsigned p0 = p4 << 2;

  const unsigned warp = tid >> 5;
  const unsigned lane = tid & 31u;
  const unsigned pp0 = warp + 0u;
  const unsigned pp1 = warp + 8u;
  const unsigned pp2 = warp + 16u;
  const unsigned pp3 = warp + 24u;
  const unsigned idx0 = noninitial5_smem_idx(lane, pp0);
  const unsigned idx1 = noninitial5_smem_idx(lane, pp1);
  const unsigned idx2 = noninitial5_smem_idx(lane, pp2);
  const unsigned idx3 = noninitial5_smem_idx(lane, pp3);

  // blockIdx.x maps to two consecutive low tiles within one high chunk.
  const unsigned tile0 = blockIdx.x << 1;
  const unsigned high = tile0 >> (START_STAGE - LOW_TILE_LOG);
  const unsigned low0 = tile0 & LOW_TILES_MASK;
  const unsigned block_base = high << (START_STAGE + ROUNDS);
  const unsigned row_base0 = block_base + (k << START_STAGE) + (low0 << LOW_TILE_LOG) + p0;
  const unsigned row_base1 = row_base0 + (1u << LOW_TILE_LOG);

  const uint4 packed0 = load_u4_mod<LOAD_MODIFIER>(src, row_base0);
  const uint4 packed1 = load_u4_mod<LOAD_MODIFIER>(src, row_base1);

  // Tile #0
  smem[noninitial5_smem_idx(k, p0 + 0u)] = bf(packed0.x);
  smem[noninitial5_smem_idx(k, p0 + 1u)] = bf(packed0.y);
  smem[noninitial5_smem_idx(k, p0 + 2u)] = bf(packed0.z);
  smem[noninitial5_smem_idx(k, p0 + 3u)] = bf(packed0.w);

  __syncthreads();

  bf v0 = smem[idx0];
  bf v1 = smem[idx1];
  bf v2 = smem[idx2];
  bf v3 = smem[idx3];

  apply_5_rounds_warp32_branchless_quad(v0, v1, v2, v3, lane);

  smem[idx0] = v0;
  smem[idx1] = v1;
  smem[idx2] = v2;
  smem[idx3] = v3;

  __syncthreads();

  const uint4 out0 = uint4{
      smem[noninitial5_smem_idx(k, p0 + 0u)].limb,
      smem[noninitial5_smem_idx(k, p0 + 1u)].limb,
      smem[noninitial5_smem_idx(k, p0 + 2u)].limb,
      smem[noninitial5_smem_idx(k, p0 + 3u)].limb,
  };
  store_u4_mod<STORE_MODIFIER>(dst, row_base0, out0);

  // Tile #1
  smem[noninitial5_smem_idx(k, p0 + 0u)] = bf(packed1.x);
  smem[noninitial5_smem_idx(k, p0 + 1u)] = bf(packed1.y);
  smem[noninitial5_smem_idx(k, p0 + 2u)] = bf(packed1.z);
  smem[noninitial5_smem_idx(k, p0 + 3u)] = bf(packed1.w);

  __syncthreads();

  v0 = smem[idx0];
  v1 = smem[idx1];
  v2 = smem[idx2];
  v3 = smem[idx3];

  apply_5_rounds_warp32_branchless_quad(v0, v1, v2, v3, lane);

  smem[idx0] = v0;
  smem[idx1] = v1;
  smem[idx2] = v2;
  smem[idx3] = v3;

  __syncthreads();

  const uint4 out1 = uint4{
      smem[noninitial5_smem_idx(k, p0 + 0u)].limb,
      smem[noninitial5_smem_idx(k, p0 + 1u)].limb,
      smem[noninitial5_smem_idx(k, p0 + 2u)].limb,
      smem[noninitial5_smem_idx(k, p0 + 3u)].limb,
  };
  store_u4_mod<STORE_MODIFIER>(dst, row_base1, out1);
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial5(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Known schedules currently use start_stage in {11, 16} for noninitial5.
  switch (start_stage) {
    case 11u:
      hypercube_evals_into_coeffs_bitrev_noninitial5_impl<LOAD_MODIFIER, STORE_MODIFIER>(src, dst, 11u);
      return;
    case 16u:
      hypercube_evals_into_coeffs_bitrev_noninitial5_impl<LOAD_MODIFIER, STORE_MODIFIER>(src, dst, 16u);
      return;
    default:
      hypercube_evals_into_coeffs_bitrev_noninitial5_impl<LOAD_MODIFIER, STORE_MODIFIER>(src, dst, start_stage);
      return;
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial7_impl(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  constexpr unsigned ROUNDS = 7;
  constexpr unsigned K = 1u << ROUNDS; // 128
  constexpr unsigned P = 32;
  constexpr unsigned LOW_TILE_LOG = 5u;
  constexpr unsigned TOTAL_VECS = (K * P) >> 2; // 1024 uint4 vectors

  __shared__ bf smem[K * P]; // 128 * 32 = 4096 BF values (16KB)

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane = tid & 31u;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned tile = blockIdx.x;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + ROUNDS);
  if (blockDim.x != 256u) {
    return;
  }

#pragma unroll
  for (unsigned iter = 0; iter < 4; iter++) {
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

#pragma unroll
  for (unsigned iter = 0; iter < 4; iter++) {
    const unsigned p = warp + (iter << 3);
    const unsigned idx0 = noninitial6_smem_idx(lane + 0u, p);
    const unsigned idx1 = noninitial6_smem_idx(lane + 32u, p);
    const unsigned idx2 = noninitial6_smem_idx(lane + 64u, p);
    const unsigned idx3 = noninitial6_smem_idx(lane + 96u, p);

    bf vals[4] = {
        smem[idx0],
        smem[idx1],
        smem[idx2],
        smem[idx3],
    };
    apply_7_rounds_pair_major_warp32_quad(vals, lane);

    smem[idx0] = vals[0];
    smem[idx1] = vals[1];
    smem[idx2] = vals[2];
    smem[idx3] = vals[3];
  }

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < 4; iter++) {
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

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial7(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Known schedules currently use fixed noninitial7 start_stage=14.
  switch (start_stage) {
    case 14u:
      hypercube_evals_into_coeffs_bitrev_noninitial7_impl<LOAD_MODIFIER, STORE_MODIFIER>(src, dst, 14u);
      return;
    default:
      hypercube_evals_into_coeffs_bitrev_noninitial7_impl<LOAD_MODIFIER, STORE_MODIFIER>(
          src, dst, start_stage);
      return;
  }
}

template <ld_modifier LOAD_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial8_half_p8_compute(
    const bf *__restrict__ src,
    const unsigned start_stage,
    const unsigned block_base,
    const unsigned low_base,
    const unsigned k_half_base,
    const unsigned p_group_base,
    bf *__restrict__ smem) {
  // One 7-round transform on a 128 x 8 half-tile.
  // - vectorized load/store uses one uint4 per thread
  // - compute uses one warp per p_local and 4-way k packing.

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane = tid & 31u;
  const unsigned k_local = tid >> 1;       // [0,127]
  const unsigned p4 = tid & 1u;            // [0,1]
  const unsigned p_local0 = p4 << 2;       // {0,4}
  const unsigned p_global0 = p_group_base + p_local0;
  const unsigned k = k_local + k_half_base;
  const unsigned row_base = block_base + (k << start_stage) + low_base + p_global0;
  const uint4 packed = load_u4_mod<LOAD_MODIFIER>(src, row_base);

  smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 0u)] = bf(packed.x);
  smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 1u)] = bf(packed.y);
  smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 2u)] = bf(packed.z);
  smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 3u)] = bf(packed.w);

  __syncthreads();

  const unsigned p_local = warp; // [0,7]
  const unsigned idx0 = noninitial8_half_p8_smem_idx(lane + 0u, p_local);
  const unsigned idx1 = noninitial8_half_p8_smem_idx(lane + 32u, p_local);
  const unsigned idx2 = noninitial8_half_p8_smem_idx(lane + 64u, p_local);
  const unsigned idx3 = noninitial8_half_p8_smem_idx(lane + 96u, p_local);

  bf vals[4] = {
      smem[idx0],
      smem[idx1],
      smem[idx2],
      smem[idx3],
  };
  apply_7_rounds_pair_major_warp32_quad(vals, lane);

  smem[idx0] = vals[0];
  smem[idx1] = vals[1];
  smem[idx2] = vals[2];
  smem[idx3] = vals[3];

  __syncthreads();
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial8_impl(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Noninitial8 split-half x4 path for start_stage=13 (2-launch log21 [13,8]):
  // - split logical [256 x 32] tile into four CTAs over p groups (width 8)
  // - low half (k=0..127): run 7 rounds, cache outputs in registers
  // - high half (k=128..255): run 7 rounds in shared
  // - final merge round: high -= low
  //
  // Shared footprint: one 128 x 8 half-tile (4KB).
  constexpr unsigned HALF_K = 128;
  constexpr unsigned P_GROUP = 8;
  constexpr unsigned LOW_TILE_LOG = 5u;
  __shared__ bf smem[HALF_K * P_GROUP]; // 128 * 8 = 1024 BF values (4KB)

  if (blockDim.x != 256u) {
    return;
  }

  const unsigned tid = threadIdx.x;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned tile_x4 = blockIdx.x;
  const unsigned tile = tile_x4 >> 2;
  const unsigned p_group = tile_x4 & 3u;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + 8u);
  const unsigned half_stride = HALF_K << start_stage;
  const unsigned p_group_base = p_group * P_GROUP;

  const unsigned k_local = tid >> 1;  // [0,127]
  const unsigned p4 = tid & 1u;       // [0,1]
  const unsigned p_local0 = p4 << 2;  // {0,4}
  uint4 low_cache;

  hypercube_evals_into_coeffs_bitrev_noninitial8_half_p8_compute<LOAD_MODIFIER>(
      src, start_stage, block_base, low_base, 0u, p_group_base, smem);

  low_cache = uint4{
      smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 0u)].limb,
      smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 1u)].limb,
      smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 2u)].limb,
      smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 3u)].limb,
  };

  __syncthreads();

  hypercube_evals_into_coeffs_bitrev_noninitial8_half_p8_compute<LOAD_MODIFIER>(
      src, start_stage, block_base, low_base, HALF_K, p_group_base, smem);

  const unsigned p_global0 = p_group_base + p_local0;
  const unsigned low_row_base = block_base + (k_local << start_stage) + low_base + p_global0;
  const unsigned high_row_base = low_row_base + half_stride;
  const uint4 high_packed = uint4{
      smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 0u)].limb,
      smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 1u)].limb,
      smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 2u)].limb,
      smem[noninitial8_half_p8_smem_idx(k_local, p_local0 + 3u)].limb,
  };
  const uint4 merged = uint4{
      bf::sub(bf(high_packed.x), bf(low_cache.x)).limb,
      bf::sub(bf(high_packed.y), bf(low_cache.y)).limb,
      bf::sub(bf(high_packed.z), bf(low_cache.z)).limb,
      bf::sub(bf(high_packed.w), bf(low_cache.w)).limb,
  };
  store_u4_mod<STORE_MODIFIER>(dst, low_row_base, low_cache);
  store_u4_mod<STORE_MODIFIER>(dst, high_row_base, merged);
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial8(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  switch (start_stage) {
    case 13u:
      hypercube_evals_into_coeffs_bitrev_noninitial8_impl<LOAD_MODIFIER, STORE_MODIFIER>(
          src, dst, 13u);
      return;
    default:
      hypercube_evals_into_coeffs_bitrev_noninitial8_impl<LOAD_MODIFIER, STORE_MODIFIER>(
          src, dst, start_stage);
      return;
  }
}

template <ld_modifier SRC_LOAD_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial10_half_p8_compute(
    const bf *__restrict__ src,
    const unsigned start_stage,
    const unsigned tile,
    const unsigned k_half_base,
    const unsigned p_group_base,
    bf *__restrict__ smem) {
  // One 9-round transform on a 512 x 8 half-tile.
  constexpr unsigned HALF_K = 512;
  constexpr unsigned P_GROUP = 8;
  constexpr unsigned VEC_ITERS = (HALF_K * P_GROUP) >> 10; // 4, for 256 threads
  constexpr unsigned LOW_TILE_LOG = 5u;

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane32 = tid & 31u;
  const unsigned subgroup = tid >> 4;
  const unsigned lane16 = tid & 15u;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + 10u);

#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    const unsigned vec = tid + (iter << 8); // [0,1023]
    const unsigned k_local = vec >> 1;      // [0,511]
    const unsigned p4 = vec & 1u;           // [0,1]
    const unsigned p_local0 = p4 << 2;      // {0,4}
    const unsigned p_global0 = p_group_base + p_local0;
    const unsigned k = k_local + k_half_base;
    const unsigned row_base = block_base + (k << start_stage) + low_base + p_global0;
    const uint4 packed = load_u4_mod<SRC_LOAD_MODIFIER>(src, row_base);

    smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 0u)] = bf(packed.x);
    smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 1u)] = bf(packed.y);
    smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 2u)] = bf(packed.z);
    smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 3u)] = bf(packed.w);
  }

  __syncthreads();

  // First 5 rounds over kl (32-point subdomain), one kh slice per iteration.
#pragma unroll
  for (unsigned iter = 0; iter < 16; iter++) {
    const unsigned kh = iter;
    const unsigned p_local = warp; // 8 warps cover p_local in [0,7]
    const unsigned k = (kh << 5) | lane32;

    bf v = smem[noninitial10_half_p8_smem_idx(k, p_local)];
    apply_5_rounds_warp32_branchless_scalar(v, lane32);
    smem[noninitial10_half_p8_smem_idx(k, p_local)] = v;
  }

  __syncthreads();

  // Last 4 rounds over kh (16-point subdomain), one (kl,p_local) per subgroup.
#pragma unroll
  for (unsigned iter = 0; iter < 16; iter++) {
    const unsigned combo = subgroup + (iter << 4); // [0,255]
    const unsigned kl = combo >> 3;                // [0,31]
    const unsigned p_local = combo & 7u;           // [0,7]
    const unsigned k = (lane16 << 5) | kl;

    bf v = smem[noninitial10_half_p8_smem_idx(k, p_local)];
    apply_4_rounds_warp16_branchless_scalar(v, lane16);
    smem[noninitial10_half_p8_smem_idx(k, p_local)] = v;
  }

  __syncthreads();
}

template <ld_modifier SRC_LOAD_MODIFIER, ld_modifier MERGE_LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8_impl(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Noninitial10 split-half path for start_stage=14:
  // - low half (k=0..511): run 9 rounds, keep outputs in registers
  // - high half (k=512..1023): run 9 rounds in shared
  // - merged high and low outputs are written without any low-half global reload
  //
  // p-dimension is processed in 4 groups of 8.
  //
  // Shared footprint: one 512 x 8 half-tile (16KB).
  constexpr unsigned HALF_K = 512;
  constexpr unsigned P_GROUP = 8;
  constexpr unsigned P_GROUPS = 4;
  constexpr unsigned LOW_TILE_LOG = 5u;
  __shared__ bf smem[HALF_K * P_GROUP]; // 16KB

  if (blockDim.x != 256u) {
    return;
  }

  const unsigned tid = threadIdx.x;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned tile = blockIdx.x;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + 10u);
  const unsigned half_stride = HALF_K << start_stage;

  uint4 low_cache[4];

#pragma unroll
  for (unsigned group = 0; group < P_GROUPS; group++) {
    const unsigned p_group_base = group * P_GROUP;
    hypercube_evals_into_coeffs_bitrev_noninitial10_half_p8_compute<SRC_LOAD_MODIFIER>(
        src, start_stage, tile, 0u, p_group_base, smem);

#pragma unroll
    for (unsigned iter = 0; iter < 4; iter++) {
      const unsigned vec = tid + (iter << 8); // [0,1023]
      const unsigned k_local = vec >> 1;      // [0,511]
      const unsigned p4 = vec & 1u;           // [0,1]
      const unsigned p_local0 = p4 << 2;      // {0,4}
      low_cache[iter] = uint4{
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 0u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 1u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 2u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 3u)].limb,
      };
    }

    // Ensure all threads finish low-cache gather before smem is overwritten.
    __syncthreads();

    hypercube_evals_into_coeffs_bitrev_noninitial10_half_p8_compute<SRC_LOAD_MODIFIER>(
        src, start_stage, tile, HALF_K, p_group_base, smem);

#pragma unroll
    for (unsigned iter = 0; iter < 4; iter++) {
      const unsigned vec = tid + (iter << 8); // [0,1023]
      const unsigned k_local = vec >> 1;      // [0,511]
      const unsigned p4 = vec & 1u;           // [0,1]
      const unsigned p_local0 = p4 << 2;      // {0,4}
      const unsigned p_global0 = p_group_base + p_local0;
      const unsigned low_row_base = block_base + (k_local << start_stage) + low_base + p_global0;
      const unsigned high_row_base = low_row_base + half_stride;
      const uint4 high_packed = uint4{
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 0u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 1u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 2u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 3u)].limb,
      };
      const uint4 merged = uint4{
          bf::sub(bf(high_packed.x), bf(low_cache[iter].x)).limb,
          bf::sub(bf(high_packed.y), bf(low_cache[iter].y)).limb,
          bf::sub(bf(high_packed.z), bf(low_cache[iter].z)).limb,
          bf::sub(bf(high_packed.w), bf(low_cache[iter].w)).limb,
      };
      store_u4_mod<STORE_MODIFIER>(dst, low_row_base, low_cache[iter]);
      store_u4_mod<STORE_MODIFIER>(dst, high_row_base, merged);
    }

    if (group + 1u < P_GROUPS) {
      __syncthreads();
    }
  }
}

template <ld_modifier SRC_LOAD_MODIFIER, ld_modifier MERGE_LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Known 2-launch log24 schedule uses fixed start_stage=14.
  switch (start_stage) {
    case 14u:
      hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8_impl<
          SRC_LOAD_MODIFIER,
          MERGE_LOAD_MODIFIER,
          STORE_MODIFIER>(src, dst, 14u);
      return;
    default:
      hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8_impl<
          SRC_LOAD_MODIFIER,
          MERGE_LOAD_MODIFIER,
          STORE_MODIFIER>(src, dst, start_stage);
      return;
  }
}

template <ld_modifier SRC_LOAD_MODIFIER, ld_modifier MERGE_LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8_x2_impl(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Variant: split each original tile into two CTAs over p-dimension halves.
  // - CTA A handles p groups {0,1} (p in [0,15])
  // - CTA B handles p groups {2,3} (p in [16,31])
  // This doubles grid size while keeping all arithmetic unchanged.
  constexpr unsigned HALF_K = 512;
  constexpr unsigned P_GROUP = 8;
  constexpr unsigned P_GROUPS_PER_BLOCK = 2;
  constexpr unsigned VEC_ITERS = 4;
  constexpr unsigned LOW_TILE_LOG = 5u;
  __shared__ bf smem[HALF_K * P_GROUP]; // 16KB

  if (blockDim.x != 256u) {
    return;
  }

  const unsigned tid = threadIdx.x;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned tile_x2 = blockIdx.x;
  const unsigned tile = tile_x2 >> 1;
  const unsigned p_half = tile_x2 & 1u;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + 10u);
  const unsigned half_stride = HALF_K << start_stage;
  const unsigned p_group_start = p_half * P_GROUPS_PER_BLOCK;

  uint4 low_cache[VEC_ITERS];

#pragma unroll
  for (unsigned group_local = 0; group_local < P_GROUPS_PER_BLOCK; group_local++) {
    const unsigned group = p_group_start + group_local;
    const unsigned p_group_base = group * P_GROUP;
    hypercube_evals_into_coeffs_bitrev_noninitial10_half_p8_compute<SRC_LOAD_MODIFIER>(
        src, start_stage, tile, 0u, p_group_base, smem);

#pragma unroll
    for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
      const unsigned vec = tid + (iter << 8); // [0,1023]
      const unsigned k_local = vec >> 1;      // [0,511]
      const unsigned p4 = vec & 1u;           // [0,1]
      const unsigned p_local0 = p4 << 2;      // {0,4}
      low_cache[iter] = uint4{
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 0u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 1u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 2u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 3u)].limb,
      };
    }

    __syncthreads();

    hypercube_evals_into_coeffs_bitrev_noninitial10_half_p8_compute<SRC_LOAD_MODIFIER>(
        src, start_stage, tile, HALF_K, p_group_base, smem);

#pragma unroll
    for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
      const unsigned vec = tid + (iter << 8); // [0,1023]
      const unsigned k_local = vec >> 1;      // [0,511]
      const unsigned p4 = vec & 1u;           // [0,1]
      const unsigned p_local0 = p4 << 2;      // {0,4}
      const unsigned p_global0 = p_group_base + p_local0;
      const unsigned low_row_base = block_base + (k_local << start_stage) + low_base + p_global0;
      const unsigned high_row_base = low_row_base + half_stride;
      const uint4 high_packed = uint4{
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 0u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 1u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 2u)].limb,
          smem[noninitial10_half_p8_smem_idx(k_local, p_local0 + 3u)].limb,
      };
      const uint4 merged = uint4{
          bf::sub(bf(high_packed.x), bf(low_cache[iter].x)).limb,
          bf::sub(bf(high_packed.y), bf(low_cache[iter].y)).limb,
          bf::sub(bf(high_packed.z), bf(low_cache[iter].z)).limb,
          bf::sub(bf(high_packed.w), bf(low_cache[iter].w)).limb,
      };
      store_u4_mod<STORE_MODIFIER>(dst, low_row_base, low_cache[iter]);
      store_u4_mod<STORE_MODIFIER>(dst, high_row_base, merged);
    }

    if (group_local + 1u < P_GROUPS_PER_BLOCK) {
      __syncthreads();
    }
  }
}

template <ld_modifier SRC_LOAD_MODIFIER, ld_modifier MERGE_LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8_x2(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  switch (start_stage) {
    case 14u:
      hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8_x2_impl<
          SRC_LOAD_MODIFIER,
          MERGE_LOAD_MODIFIER,
          STORE_MODIFIER>(src, dst, 14u);
      return;
    default:
      hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8_x2_impl<
          SRC_LOAD_MODIFIER,
          MERGE_LOAD_MODIFIER,
          STORE_MODIFIER>(src, dst, start_stage);
      return;
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial11_x4_impl(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Noninitial11 final stage for 2-launch log24 [13,11], x4 p-group split:
  // - logical tile is [k=2048][p=32]
  // - each CTA handles one p-group of 8 columns (grid x4 multiplier)
  // - one full 11-round transform per (p,k) tile is executed in shared memory
  // - global IO remains one vectorized load and one vectorized store per element.
  constexpr unsigned K = 2048;
  constexpr unsigned P_GROUP = 8;
  constexpr unsigned VEC_ITERS = 4; // (2048 * 8) / (1024 * 4)
  constexpr unsigned LOW_TILE_LOG = 5u;
  extern __shared__ bf smem[]; // 2048 * 8 = 16384 BF values (64KB)

  if (blockDim.x != 1024u) {
    return;
  }

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane32 = tid & 31u;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned tile_x4 = blockIdx.x;
  const unsigned tile = tile_x4 >> 2;
  const unsigned p_group = tile_x4 & 3u;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + 11u);
  const unsigned p_group_base = p_group * P_GROUP;

  // Load canonical row-major vectors from global memory into the swizzled shared tile.
#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    const unsigned vec = tid + (iter << 10); // [0,4095]
    const unsigned k = vec >> 1;             // [0,2047]
    const unsigned p4 = vec & 1u;            // [0,1]
    const unsigned p_local0 = p4 << 2;       // {0,4}
    const unsigned p_global0 = p_group_base + p_local0;
    const unsigned row_base = block_base + (k << start_stage) + low_base + p_global0;
    const uint4 packed = load_u4_mod<LOAD_MODIFIER>(src, row_base);

    smem[noninitial11_full_p8_smem_idx(k, p_local0 + 0u)] = bf(packed.x);
    smem[noninitial11_full_p8_smem_idx(k, p_local0 + 1u)] = bf(packed.y);
    smem[noninitial11_full_p8_smem_idx(k, p_local0 + 2u)] = bf(packed.z);
    smem[noninitial11_full_p8_smem_idx(k, p_local0 + 3u)] = bf(packed.w);
  }

  __syncthreads();

  // First 5 rounds over kl: one (kh, p_local) combo per warp and iteration.
#pragma unroll
  for (unsigned iter = 0; iter < 16; iter++) {
    const unsigned combo = warp + (iter << 5); // [0,511]
    const unsigned kh = combo >> 3;            // [0,63]
    const unsigned p_local = combo & 7u;       // [0,7]
    const unsigned k = (kh << 5) | lane32;
    bf v = smem[noninitial11_full_p8_smem_idx(k, p_local)];
    apply_5_rounds_warp32_branchless_scalar(v, lane32);
    smem[noninitial11_full_p8_smem_idx(k, p_local)] = v;
  }

  // Synchronize between p-major and k-major traversal phases.
  __syncthreads();

  // Last 6 rounds over kh are implemented as:
  // - 5 rounds on kh in [0,31]
  // - 5 rounds on kh in [32,63]
  // - final merge round high -= low.
#pragma unroll
  for (unsigned iter = 0; iter < 8; iter++) {
    const unsigned combo = warp + (iter << 5); // [0,255]
    const unsigned kl = combo >> 3;            // [0,31]
    const unsigned p_local = combo & 7u;       // [0,7]
    const unsigned k_low = (lane32 << 5) | kl;
    bf v = smem[noninitial11_full_p8_smem_idx(k_low, p_local)];
    apply_5_rounds_warp32_branchless_scalar(v, lane32);
    smem[noninitial11_full_p8_smem_idx(k_low, p_local)] = v;
  }

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < 8; iter++) {
    const unsigned combo = warp + (iter << 5); // [0,255]
    const unsigned kl = combo >> 3;            // [0,31]
    const unsigned p_local = combo & 7u;       // [0,7]
    const unsigned k_high = ((lane32 + 32u) << 5) | kl;
    bf v = smem[noninitial11_full_p8_smem_idx(k_high, p_local)];
    apply_5_rounds_warp32_branchless_scalar(v, lane32);
    smem[noninitial11_full_p8_smem_idx(k_high, p_local)] = v;
  }

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < 8; iter++) {
    const unsigned combo = warp + (iter << 5); // [0,255]
    const unsigned kl = combo >> 3;            // [0,31]
    const unsigned p_local = combo & 7u;       // [0,7]
    const unsigned k_low = (lane32 << 5) | kl;
    const unsigned k_high = ((lane32 + 32u) << 5) | kl;
    const bf low = smem[noninitial11_full_p8_smem_idx(k_low, p_local)];
    const bf high = smem[noninitial11_full_p8_smem_idx(k_high, p_local)];
    smem[noninitial11_full_p8_smem_idx(k_high, p_local)] = bf::sub(high, low);
  }

  __syncthreads();

  // Write canonical row-major vectors back to global memory.
#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    const unsigned vec = tid + (iter << 10); // [0,4095]
    const unsigned k = vec >> 1;             // [0,2047]
    const unsigned p4 = vec & 1u;            // [0,1]
    const unsigned p_local0 = p4 << 2;       // {0,4}
    const unsigned p_global0 = p_group_base + p_local0;
    const unsigned row_base = block_base + (k << start_stage) + low_base + p_global0;
    const uint4 packed = uint4{
        smem[noninitial11_full_p8_smem_idx(k, p_local0 + 0u)].limb,
        smem[noninitial11_full_p8_smem_idx(k, p_local0 + 1u)].limb,
        smem[noninitial11_full_p8_smem_idx(k, p_local0 + 2u)].limb,
        smem[noninitial11_full_p8_smem_idx(k, p_local0 + 3u)].limb,
    };
    store_u4_mod<STORE_MODIFIER>(dst, row_base, packed);
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial11_x4(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  switch (start_stage) {
    case 13u:
      hypercube_evals_into_coeffs_bitrev_noninitial11_x4_impl<LOAD_MODIFIER, STORE_MODIFIER>(
          src, dst, 13u);
      return;
    default:
      hypercube_evals_into_coeffs_bitrev_noninitial11_x4_impl<LOAD_MODIFIER, STORE_MODIFIER>(
          src, dst, start_stage);
      return;
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial9_impl(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Noninitial9 kernel for start_stage=15 (2-launch log24 [15,9]):
  //
  // - Logical tile is [k=512][p=32] = 2^14 values.
  // - p is processed as four independent 512x8 groups.
  // - Per group:
  //   1) vectorized global load -> swizzled shared (512x8)
  //   2) first 5 rounds across kl (warp32)
  //   3) last 4 rounds across kh (warp16 subgroup)
  //   4) shared -> vectorized global store
  //
  // Shared footprint remains 16KB, one load and one store per element.
  constexpr unsigned K = 512;
  constexpr unsigned P_GROUP = 8;
  constexpr unsigned P_GROUPS = 4;
  constexpr unsigned VEC_ITERS = 4;
  constexpr unsigned LOW_TILE_LOG = 5u;
  __shared__ bf smem[K * P_GROUP]; // 512 * 8 = 4096 BF values (16KB)

  if (blockDim.x != 256u) {
    return;
  }

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane32 = tid & 31u;
  const unsigned subgroup = tid >> 4;
  const unsigned lane16 = tid & 15u;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned tile = blockIdx.x;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + 9u);

#pragma unroll
  for (unsigned group = 0; group < P_GROUPS; group++) {
    const unsigned p_group_base = group * P_GROUP;

#pragma unroll
    for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
      const unsigned vec = tid + (iter << 8); // [0,1023]
      const unsigned k = vec >> 1;            // [0,511]
      const unsigned p4 = vec & 1u;           // [0,1]
      const unsigned p_local0 = p4 << 2;      // {0,4}
      const unsigned p_global0 = p_group_base + p_local0;
      const unsigned row_base = block_base + (k << start_stage) + low_base + p_global0;
      const uint4 packed = load_u4_mod<LOAD_MODIFIER>(src, row_base);

      smem[noninitial10_half_p8_smem_idx(k, p_local0 + 0u)] = bf(packed.x);
      smem[noninitial10_half_p8_smem_idx(k, p_local0 + 1u)] = bf(packed.y);
      smem[noninitial10_half_p8_smem_idx(k, p_local0 + 2u)] = bf(packed.z);
      smem[noninitial10_half_p8_smem_idx(k, p_local0 + 3u)] = bf(packed.w);
    }

    __syncthreads();

    // First 5 rounds over kl (32-point subdomain), one kh slice per iteration.
#pragma unroll
    for (unsigned iter = 0; iter < 16; iter++) {
      const unsigned kh = iter;
      const unsigned p_local = warp; // 8 warps cover p_local in [0,7]
      const unsigned k = (kh << 5) | lane32;

      bf v = smem[noninitial10_half_p8_smem_idx(k, p_local)];
      apply_5_rounds_warp32_branchless_scalar(v, lane32);
      smem[noninitial10_half_p8_smem_idx(k, p_local)] = v;
    }

    __syncthreads();

    // Last 4 rounds over kh (16-point subdomain), one (kl,p_local) per subgroup.
#pragma unroll
    for (unsigned iter = 0; iter < 16; iter++) {
      const unsigned combo = subgroup + (iter << 4); // [0,255]
      const unsigned kl = combo >> 3;                // [0,31]
      const unsigned p_local = combo & 7u;           // [0,7]
      const unsigned k = (lane16 << 5) | kl;

      bf v = smem[noninitial10_half_p8_smem_idx(k, p_local)];
      apply_4_rounds_warp16_branchless_scalar(v, lane16);
      smem[noninitial10_half_p8_smem_idx(k, p_local)] = v;
    }

    __syncthreads();

#pragma unroll
    for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
      const unsigned vec = tid + (iter << 8); // [0,1023]
      const unsigned k = vec >> 1;            // [0,511]
      const unsigned p4 = vec & 1u;           // [0,1]
      const unsigned p_local0 = p4 << 2;      // {0,4}
      const unsigned p_global0 = p_group_base + p_local0;
      const unsigned row_base = block_base + (k << start_stage) + low_base + p_global0;
      const uint4 packed = uint4{
          smem[noninitial10_half_p8_smem_idx(k, p_local0 + 0u)].limb,
          smem[noninitial10_half_p8_smem_idx(k, p_local0 + 1u)].limb,
          smem[noninitial10_half_p8_smem_idx(k, p_local0 + 2u)].limb,
          smem[noninitial10_half_p8_smem_idx(k, p_local0 + 3u)].limb,
      };
      store_u4_mod<STORE_MODIFIER>(dst, row_base, packed);
    }

    if (group + 1u < P_GROUPS) {
      __syncthreads();
    }
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial9(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  switch (start_stage) {
    case 15u:
      hypercube_evals_into_coeffs_bitrev_noninitial9_impl<LOAD_MODIFIER, STORE_MODIFIER>(
          src, dst, 15u);
      return;
    default:
      hypercube_evals_into_coeffs_bitrev_noninitial9_impl<LOAD_MODIFIER, STORE_MODIFIER>(
          src, dst, start_stage);
      return;
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial9_x2_impl(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Experimental x4 split variant behind the x2 entrypoint name:
  // split each logical [512 x 32] tile into four CTAs over p groups.
  // Each CTA handles exactly one p group of width 8.
  constexpr unsigned K = 512;
  constexpr unsigned P_GROUP = 8;
  constexpr unsigned VEC_ITERS = 4;
  constexpr unsigned LOW_TILE_LOG = 5u;
  __shared__ bf smem[K * P_GROUP]; // 16KB

  if (blockDim.x != 256u) {
    return;
  }

  const unsigned tid = threadIdx.x;
  const unsigned warp = tid >> 5;
  const unsigned lane32 = tid & 31u;
  const unsigned subgroup = tid >> 4;
  const unsigned lane16 = tid & 15u;
  const unsigned stride = 1u << start_stage;
  const unsigned low_tiles = stride >> LOW_TILE_LOG;
  const unsigned tile_x4 = blockIdx.x;
  const unsigned tile = tile_x4 >> 2;
  const unsigned p_group = tile_x4 & 3u;
  const unsigned high = tile / low_tiles;
  const unsigned low_tile_id = tile - high * low_tiles;
  const unsigned low_base = low_tile_id << LOW_TILE_LOG;
  const unsigned block_base = high << (start_stage + 9u);
  const unsigned p_group_base = p_group * P_GROUP;

#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    const unsigned vec = tid + (iter << 8); // [0,1023]
    const unsigned k = vec >> 1;            // [0,511]
    const unsigned p4 = vec & 1u;           // [0,1]
    const unsigned p_local0 = p4 << 2;      // {0,4}
    const unsigned p_global0 = p_group_base + p_local0;
    const unsigned row_base = block_base + (k << start_stage) + low_base + p_global0;
    const uint4 packed = load_u4_mod<LOAD_MODIFIER>(src, row_base);

    smem[noninitial10_half_p8_smem_idx(k, p_local0 + 0u)] = bf(packed.x);
    smem[noninitial10_half_p8_smem_idx(k, p_local0 + 1u)] = bf(packed.y);
    smem[noninitial10_half_p8_smem_idx(k, p_local0 + 2u)] = bf(packed.z);
    smem[noninitial10_half_p8_smem_idx(k, p_local0 + 3u)] = bf(packed.w);
  }

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < 16; iter++) {
    const unsigned kh = iter;
    const unsigned p_local = warp;
    const unsigned k = (kh << 5) | lane32;

    bf v = smem[noninitial10_half_p8_smem_idx(k, p_local)];
    apply_5_rounds_warp32_branchless_scalar(v, lane32);
    smem[noninitial10_half_p8_smem_idx(k, p_local)] = v;
  }

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < 16; iter++) {
    const unsigned combo = subgroup + (iter << 4);
    const unsigned kl = combo >> 3;
    const unsigned p_local = combo & 7u;
    const unsigned k = (lane16 << 5) | kl;

    bf v = smem[noninitial10_half_p8_smem_idx(k, p_local)];
    apply_4_rounds_warp16_branchless_scalar(v, lane16);
    smem[noninitial10_half_p8_smem_idx(k, p_local)] = v;
  }

  __syncthreads();

#pragma unroll
  for (unsigned iter = 0; iter < VEC_ITERS; iter++) {
    const unsigned vec = tid + (iter << 8); // [0,1023]
    const unsigned k = vec >> 1;            // [0,511]
    const unsigned p4 = vec & 1u;           // [0,1]
    const unsigned p_local0 = p4 << 2;      // {0,4}
    const unsigned p_global0 = p_group_base + p_local0;
    const unsigned row_base = block_base + (k << start_stage) + low_base + p_global0;
    const uint4 packed = uint4{
        smem[noninitial10_half_p8_smem_idx(k, p_local0 + 0u)].limb,
        smem[noninitial10_half_p8_smem_idx(k, p_local0 + 1u)].limb,
        smem[noninitial10_half_p8_smem_idx(k, p_local0 + 2u)].limb,
        smem[noninitial10_half_p8_smem_idx(k, p_local0 + 3u)].limb,
    };
    store_u4_mod<STORE_MODIFIER>(dst, row_base, packed);
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial9_x2(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  switch (start_stage) {
    case 15u:
      hypercube_evals_into_coeffs_bitrev_noninitial9_x2_impl<LOAD_MODIFIER, STORE_MODIFIER>(
          src, dst, 15u);
      return;
    default:
      hypercube_evals_into_coeffs_bitrev_noninitial9_x2_impl<LOAD_MODIFIER, STORE_MODIFIER>(
          src, dst, start_stage);
      return;
  }
}

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial6_impl(
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

template <ld_modifier LOAD_MODIFIER, st_modifier STORE_MODIFIER>
DEVICE_FORCEINLINE void hypercube_evals_into_coeffs_bitrev_noninitial6(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Known schedules use start_stage in {11, 12, 16, 17, 18}. Dispatching these
  // literals allows full constant-folding inside the noninitial body.
  switch (start_stage) {
    case 11u:
      hypercube_evals_into_coeffs_bitrev_noninitial6_impl<LOAD_MODIFIER, STORE_MODIFIER>(src, dst, 11u);
      return;
    case 12u:
      hypercube_evals_into_coeffs_bitrev_noninitial6_impl<LOAD_MODIFIER, STORE_MODIFIER>(src, dst, 12u);
      return;
    case 16u:
      hypercube_evals_into_coeffs_bitrev_noninitial6_impl<LOAD_MODIFIER, STORE_MODIFIER>(src, dst, 16u);
      return;
    case 17u:
      hypercube_evals_into_coeffs_bitrev_noninitial6_impl<LOAD_MODIFIER, STORE_MODIFIER>(src, dst, 17u);
      return;
    case 18u:
      hypercube_evals_into_coeffs_bitrev_noninitial6_impl<LOAD_MODIFIER, STORE_MODIFIER>(src, dst, 18u);
      return;
    default:
      hypercube_evals_into_coeffs_bitrev_noninitial6_impl<LOAD_MODIFIER, STORE_MODIFIER>(src, dst, start_stage);
      return;
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

EXTERN __launch_bounds__(1024, 1) __global__ void ab_h2m_bitrev_bf_initial14_out_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // Out-of-place initial14 policy: ld.cs + st.wt.
  hypercube_evals_into_coeffs_bitrev_initial14<ld_modifier::cs, st_modifier::wt, false, 1024u>(src, dst);
}

EXTERN __launch_bounds__(1024, 1) __global__ void ab_h2m_bitrev_bf_initial14_in_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // In-place initial14 policy: ld.cg + st.wt.
  (void)src;
  hypercube_evals_into_coeffs_bitrev_initial14<ld_modifier::cg, st_modifier::wt, true, 1024u>(src, dst);
}

EXTERN __launch_bounds__(512, 1) __global__ void ab_h2m_bitrev_bf_initial14_out_512_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // Out-of-place initial14 (512-thread CTA) for log24 tuning.
  hypercube_evals_into_coeffs_bitrev_initial14<ld_modifier::cs, st_modifier::wt, false, 512u>(src, dst);
}

EXTERN __launch_bounds__(512, 1) __global__ void ab_h2m_bitrev_bf_initial14_in_512_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // In-place initial14 (512-thread CTA) for log24 tuning.
  (void)src;
  hypercube_evals_into_coeffs_bitrev_initial14<ld_modifier::cg, st_modifier::wt, true, 512u>(src, dst);
}

EXTERN __launch_bounds__(1024, 1) __global__ void ab_h2m_bitrev_bf_initial15_out_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // Out-of-place initial15 policy: ld.cs + st.wt.
  hypercube_evals_into_coeffs_bitrev_initial15<ld_modifier::cs, st_modifier::wt, false, 1024u>(
      src, dst);
}

EXTERN __launch_bounds__(1024, 1) __global__ void ab_h2m_bitrev_bf_initial15_in_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // In-place initial15 policy: ld.cg + st.wt.
  (void)src;
  hypercube_evals_into_coeffs_bitrev_initial15<ld_modifier::cg, st_modifier::wt, true, 1024u>(
      src, dst);
}

EXTERN __launch_bounds__(512, 1) __global__ void ab_h2m_bitrev_bf_initial15_out_512_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // Out-of-place initial15 (512-thread CTA) for log24 tuning.
  hypercube_evals_into_coeffs_bitrev_initial15<ld_modifier::cs, st_modifier::wt, false, 512u>(
      src, dst);
}

EXTERN __launch_bounds__(512, 1) __global__ void ab_h2m_bitrev_bf_initial15_in_512_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // In-place initial15 (512-thread CTA) for log24 tuning.
  (void)src;
  hypercube_evals_into_coeffs_bitrev_initial15<ld_modifier::cg, st_modifier::wt, true, 512u>(
      src, dst);
}

EXTERN __launch_bounds__(1024, 1) __global__ void ab_h2m_bitrev_bf_initial13_out_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // Out-of-place initial13 policy: ld.cs + st.wt.
  hypercube_evals_into_coeffs_bitrev_initial13<ld_modifier::cs, st_modifier::wt, false>(src, dst);
}

EXTERN __launch_bounds__(1024, 1) __global__ void ab_h2m_bitrev_bf_initial13_in_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst) {
  // In-place initial13 policy: ld.cg + st.wt.
  (void)src;
  hypercube_evals_into_coeffs_bitrev_initial13<ld_modifier::cg, st_modifier::wt, true>(src, dst);
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

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial5_stage2_out_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Stage2 out-of-place policy: ld.cs + st.wt.
  hypercube_evals_into_coeffs_bitrev_noninitial5<ld_modifier::cs, st_modifier::wt>(src, dst, start_stage);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial5_stage2_in_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Stage2 in-place policy: ld.ca + st.wt.
  hypercube_evals_into_coeffs_bitrev_noninitial5<ld_modifier::ca, st_modifier::wt>(src, dst, start_stage);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial5_stage3_out_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Stage #3 kernel (out-of-place): ld.cs + st.cs.
  hypercube_evals_into_coeffs_bitrev_noninitial5<ld_modifier::cs, st_modifier::cs>(src, dst, start_stage);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial5_stage3_in_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Stage #3 kernel (in-place): ld.ca + st.cs.
  hypercube_evals_into_coeffs_bitrev_noninitial5<ld_modifier::ca, st_modifier::cs>(src, dst, start_stage);
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

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage2_out_start11_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage2 (out-of-place) for log23 schedule: ld.cs + st.wt.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial6_impl<ld_modifier::cs, st_modifier::wt>(src, dst, 11u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage2_out_start12_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage2 (out-of-place) for log24 schedule: ld.cs + st.wt.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial6_impl<ld_modifier::cs, st_modifier::wt>(src, dst, 12u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage2_in_start11_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage2 (in-place) for log23 schedule: ld.ca + st.wt.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial6_impl<ld_modifier::ca, st_modifier::wt>(src, dst, 11u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage2_in_start12_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage2 (in-place) for log24 schedule: ld.ca + st.wt.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial6_impl<ld_modifier::ca, st_modifier::wt>(src, dst, 12u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial5_stage2_out_start11_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage2 (out-of-place) for log22/log21 schedules: ld.cs + st.wt.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial5_impl<ld_modifier::cs, st_modifier::wt>(src, dst, 11u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial5_stage2_in_start11_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage2 (in-place) for log22/log21 schedules: ld.ca + st.wt.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial5_impl<ld_modifier::ca, st_modifier::wt>(src, dst, 11u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial5_stage2_out_start11_x2_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage2 (out-of-place) two-tile CTA path for log22 schedule.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial5_impl_x2<ld_modifier::cs, st_modifier::wt>(src, dst, 11u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial5_stage2_in_start11_x2_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage2 (in-place) two-tile CTA path for log22 schedule.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial5_impl_x2<ld_modifier::ca, st_modifier::wt>(src, dst, 11u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage3_out_start16_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage3 (out-of-place) for log22 schedule: ld.cs + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial6_impl<ld_modifier::cs, st_modifier::cs>(src, dst, 16u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage3_in_start16_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage3 (in-place) for log22 schedule: ld.ca + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial6_impl<ld_modifier::ca, st_modifier::cs>(src, dst, 16u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage3_out_start17_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage3 (out-of-place) for log23 schedule: ld.cs + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial6_impl<ld_modifier::cs, st_modifier::cs>(src, dst, 17u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage3_out_start18_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage3 (out-of-place) for log24 schedule: ld.cs + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial6_impl<ld_modifier::cs, st_modifier::cs>(src, dst, 18u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage3_in_start17_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage3 (in-place) for log23 schedule: ld.ca + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial6_impl<ld_modifier::ca, st_modifier::cs>(src, dst, 17u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial6_stage3_in_start18_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage3 (in-place) for log24 schedule: ld.ca + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial6_impl<ld_modifier::ca, st_modifier::cs>(src, dst, 18u);
}

EXTERN __launch_bounds__(512, 2) __global__ void ab_h2m_bitrev_bf_noninitial7_stage3_out_start14_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial7 (out-of-place) for 2-launch schedule: ld.cs + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial7<ld_modifier::cs, st_modifier::cs>(src, dst, 14u);
}

EXTERN __launch_bounds__(512, 2) __global__ void ab_h2m_bitrev_bf_noninitial7_stage3_in_start14_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial7 (in-place) for 2-launch schedule: ld.ca + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial7<ld_modifier::ca, st_modifier::cs>(src, dst, 14u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial8_stage3_out_start13_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial8 (out-of-place) for 2-launch log21 [13,8].
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial8<ld_modifier::ca, st_modifier::cs>(src, dst, 13u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial8_stage3_in_start13_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial8 (in-place) for 2-launch log21 [13,8].
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial8<ld_modifier::ca, st_modifier::cs>(src, dst, 13u);
}

EXTERN __launch_bounds__(1024, 1) __global__ void ab_h2m_bitrev_bf_noninitial11_stage3_out_start13_x4_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial11 x4-grid (out-of-place) for log24 [13,11].
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial11_x4<ld_modifier::cg, st_modifier::cs>(
      src, dst, 13u);
}

EXTERN __launch_bounds__(1024, 1) __global__ void ab_h2m_bitrev_bf_noninitial11_stage3_in_start13_x4_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial11 x4-grid (in-place) for log24 [13,11].
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial11_x4<ld_modifier::ca, st_modifier::cs>(
      src, dst, 13u);
}

EXTERN __launch_bounds__(256, 4) __global__ void ab_h2m_bitrev_bf_noninitial9_stage3_out_start15_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial9 (out-of-place) for 2-launch log24 [15,9].
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial9_impl<ld_modifier::cs, st_modifier::cs>(src, dst, 15u);
}

EXTERN __launch_bounds__(256, 4) __global__ void ab_h2m_bitrev_bf_noninitial9_stage3_in_start15_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial9 (in-place) for 2-launch log24 [15,9].
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial9_impl<ld_modifier::ca, st_modifier::cs>(src, dst, 15u);
}

EXTERN __launch_bounds__(256, 12) __global__ void ab_h2m_bitrev_bf_noninitial9_stage3_out_start15_x2_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial9 x2-grid (out-of-place) for log24 [15,9].
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial9_x2_impl<ld_modifier::cg, st_modifier::cs>(
      src, dst, 15u);
}

EXTERN __launch_bounds__(256, 12) __global__ void ab_h2m_bitrev_bf_noninitial9_stage3_in_start15_x2_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial9 x2-grid (in-place) for log24 [15,9].
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial9_x2_impl<ld_modifier::ca, st_modifier::cs>(
      src, dst, 15u);
}

EXTERN __launch_bounds__(256, 3) __global__ void ab_h2m_bitrev_bf_noninitial10_stage3_out_start14_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial10 (out-of-place) for 2-launch log24.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8<
      ld_modifier::cg,
      ld_modifier::cg,
      st_modifier::cs>(src, dst, 14u);
}

EXTERN __launch_bounds__(256, 3) __global__ void ab_h2m_bitrev_bf_noninitial10_stage3_in_start14_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial10 (in-place) for 2-launch log24: ld.ca + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8<
      ld_modifier::ca,
      ld_modifier::ca,
      st_modifier::cs>(src, dst, 14u);
}

EXTERN __launch_bounds__(256, 3) __global__ void ab_h2m_bitrev_bf_noninitial10_stage3_out_start14_x2_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial10 split over x2 grid (out-of-place) for log24 exploration.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8_x2<
      ld_modifier::cg,
      ld_modifier::cg,
      st_modifier::cs>(src, dst, 14u);
}

EXTERN __launch_bounds__(256, 3) __global__ void ab_h2m_bitrev_bf_noninitial10_stage3_in_start14_x2_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Final-stage noninitial10 split over x2 grid (in-place) for log24 exploration.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial10_split_p8_x2<
      ld_modifier::ca,
      ld_modifier::ca,
      st_modifier::cs>(src, dst, 14u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial5_stage3_out_start16_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage3 (out-of-place) for log21 schedule: ld.cs + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial5_impl<ld_modifier::cs, st_modifier::cs>(src, dst, 16u);
}

EXTERN __launch_bounds__(256, 6) __global__ void ab_h2m_bitrev_bf_noninitial5_stage3_in_start16_kernel(
    const bf *__restrict__ src,
    bf *__restrict__ dst,
    const unsigned start_stage) {
  // Fixed-start stage3 (in-place) for log21 schedule: ld.ca + st.cs.
  (void)start_stage;
  hypercube_evals_into_coeffs_bitrev_noninitial5_impl<ld_modifier::ca, st_modifier::cs>(src, dst, 16u);
}

} // namespace airbender::ops::hypercube
