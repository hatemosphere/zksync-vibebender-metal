#pragma once

#include "../field.metal"
#include "../memory.metal"

// NTT common utilities — port of gpu_prover/native/ntt/ntt.cuh
// Key differences from CUDA:
// - Twiddle data passed as function args (no __constant__ globals)
// - __shfl_xor_sync → simd_shuffle_xor
// - __syncwarp → simdgroup_barrier
// - __brev → reverse_bits
// - __clz → clz

namespace airbender {
namespace ntt {

using bf = field::base_field;
using e2f = field::ext2_field;

static constant constexpr uint CIRCLE_GROUP_LOG_ORDER = 31;

// --- Twiddle data structures (passed as buffer args instead of __constant__) ---

struct powers_layer_data {
    const device e2f* values;
    uint mask;
    uint log_count;
};

struct powers_data_2_layer {
    powers_layer_data fine;
    powers_layer_data coarse;
};

// --- Bit reversal ---

DEVICE_FORCEINLINE uint bitrev(const uint idx, const uint log_n) {
    return reverse_bits(idx) >> (32 - log_n);
}

// --- DIT/DIF butterfly operations ---

DEVICE_FORCEINLINE void exchg_dit(thread e2f& a, thread e2f& b, const e2f twiddle) {
    b = e2f::mul(b, twiddle);
    const e2f a_tmp = a;
    a = e2f::add(a_tmp, b);
    b = e2f::sub(a_tmp, b);
}

DEVICE_FORCEINLINE void exchg_dif(thread e2f& a, thread e2f& b, const e2f twiddle) {
    const e2f a_tmp = a;
    a = e2f::add(a_tmp, b);
    b = e2f::sub(a_tmp, b);
    b = e2f::mul(b, twiddle);
}

// --- Twiddle factor loading ---
// In CUDA, twiddles are in __constant__ memory. In Metal, they're passed as buffer args.

DEVICE_FORCEINLINE e2f get_twiddle(const thread powers_data_2_layer& data, const uint i) {
    uint fine_idx = (i >> data.coarse.log_count) & data.fine.mask;
    uint coarse_idx = i & data.coarse.mask;
    e2f coarse = data.coarse.values[coarse_idx];
    if (fine_idx == 0)
        return coarse;
    e2f fine = data.fine.values[fine_idx];
    return e2f::mul(fine, coarse);
}

// --- SIMD shuffle for ext2_field ---
// Replaces CUDA's shfl_xor_e2f which used __shfl_xor_sync

DEVICE_FORCEINLINE void shfl_xor_e2f(thread e2f* vals, const uint i, const uint lane_id, const uint lane_mask) {
    e2f tmp;
    if (lane_id & lane_mask)
        tmp = vals[2 * i];
    else
        tmp = vals[2 * i + 1];
    tmp.coefficients[0].limb = simd_shuffle_xor(tmp.coefficients[0].limb, static_cast<ushort>(lane_mask));
    tmp.coefficients[1].limb = simd_shuffle_xor(tmp.coefficients[1].limb, static_cast<ushort>(lane_mask));
    if (lane_id & lane_mask)
        vals[2 * i] = tmp;
    else
        vals[2 * i + 1] = tmp;
}

// --- LDE scale and shift ---

DEVICE_FORCEINLINE e2f get_power_of_w(const device e2f* fine_values, uint fine_mask, uint fine_log_count,
                                       const device e2f* coarser_values, uint coarser_mask, uint coarser_log_count,
                                       const device e2f* coarsest_values, uint coarsest_mask,
                                       const uint index, const bool inverse) {
    const uint idx = inverse ? (1u << CIRCLE_GROUP_LOG_ORDER) - index : index;

    const uint coarsest_idx = (idx >> (fine_log_count + coarser_log_count)) & coarsest_mask;
    e2f val = coarsest_values[coarsest_idx];

    const uint coarser_idx = (idx >> fine_log_count) & coarser_mask;
    if (coarser_idx != 0)
        val = e2f::mul(val, coarser_values[coarser_idx]);

    const uint fine_idx = idx & fine_mask;
    if (fine_idx != 0)
        val = e2f::mul(val, fine_values[fine_idx]);

    return val;
}

DEVICE_FORCEINLINE e2f lde_scale_and_shift(const e2f Zk, const uint k,
                                            const uint log_extension_degree, const uint coset_idx, const uint log_n,
                                            const device e2f* fine_values, uint fine_mask, uint fine_log_count,
                                            const device e2f* coarser_values, uint coarser_mask, uint coarser_log_count,
                                            const device e2f* coarsest_values, uint coarsest_mask) {
    if (coset_idx == 0)
        return Zk;
    const uint tau_power_of_w = coset_idx << (CIRCLE_GROUP_LOG_ORDER - log_n - log_extension_degree);
    const uint H_over_two = 1u << (log_n - 1);
    const uint power_of_w = k >= H_over_two
        ? tau_power_of_w * (k - H_over_two)
        : (1u << CIRCLE_GROUP_LOG_ORDER) - tau_power_of_w * (H_over_two - k);
    return e2f::mul(Zk, get_power_of_w(fine_values, fine_mask, fine_log_count,
                                         coarser_values, coarser_mask, coarser_log_count,
                                         coarsest_values, coarsest_mask,
                                         power_of_w, false));
}

// --- Inverse LDE scale (for N2B coset unscaling) ---

DEVICE_FORCEINLINE e2f lde_scale_inverse(const e2f Zk, const uint k,
                                          const uint log_extension_degree, const uint coset_idx, const uint log_n,
                                          const device e2f* fine_values, uint fine_mask, uint fine_log_count,
                                          const device e2f* coarser_values, uint coarser_mask, uint coarser_log_count,
                                          const device e2f* coarsest_values, uint coarsest_mask) {
    if (coset_idx == 0)
        return Zk;
    const uint tau_power_of_w = coset_idx << (CIRCLE_GROUP_LOG_ORDER - log_n - log_extension_degree);
    const uint H_over_two = 1u << (log_n - 1);
    const uint power_of_w = k >= H_over_two
        ? tau_power_of_w * (k - H_over_two)
        : (1u << CIRCLE_GROUP_LOG_ORDER) - tau_power_of_w * (H_over_two - k);
    return e2f::mul(Zk, get_power_of_w(fine_values, fine_mask, fine_log_count,
                                         coarser_values, coarser_mask, coarser_log_count,
                                         coarsest_values, coarsest_mask,
                                         power_of_w, true));
}

// lde_scale (without shift) for N2B coset unscaling (non-compressed)
DEVICE_FORCEINLINE e2f lde_scale(const e2f Zk, const uint k,
                                  const uint log_extension_degree, const uint coset_idx, const uint log_n,
                                  const device e2f* fine_values, uint fine_mask, uint fine_log_count,
                                  const device e2f* coarser_values, uint coarser_mask, uint coarser_log_count,
                                  const device e2f* coarsest_values, uint coarsest_mask) {
    if (coset_idx == 0)
        return Zk;
    const uint tau_power_of_w = coset_idx << (CIRCLE_GROUP_LOG_ORDER - log_n - log_extension_degree);
    const uint power_of_w = k * tau_power_of_w;
    return e2f::mul(Zk, get_power_of_w(fine_values, fine_mask, fine_log_count,
                                         coarser_values, coarser_mask, coarser_log_count,
                                         coarsest_values, coarsest_mask,
                                         power_of_w, true));
}

// --- Twiddle loading helpers for multi-stage kernels ---

// COL_PAIRS_PER_BLOCK: 1 Z pair per block (matching CUDA's approach).
// Higher values cause shared memory corruption between data transpose and twiddle cache.
static constant constexpr uint COL_PAIRS_PER_BLOCK = 1;

// Column pairs processed per warp-level block (no shared memory conflicts).
static constant constexpr uint COL_PAIRS_PER_WARP_BLOCK = 4;

// Load initial twiddles for warp-level stages (6 stages of warp shuffles + LOG_VALS_PER_THREAD-1 register stages)
// This loads twiddles cooperatively across the warp, forward (DIF) direction.
template <uint VALS_PER_WARP, uint LOG_VALS_PER_THREAD, bool inverse>
DEVICE_FORCEINLINE void load_initial_twiddles_warp(threadgroup e2f* twiddle_cache,
                                                    const uint lane_id,
                                                    const uint gmem_offset,
                                                    const thread powers_data_2_layer& twiddle_data) {
    threadgroup e2f* twiddles_this_stage = twiddle_cache;
    uint num_twiddles_this_stage = VALS_PER_WARP >> 1;
    uint exchg_region_offset = gmem_offset >> 1;

    #pragma unroll
    for (uint stage = 0; stage < LOG_VALS_PER_THREAD; stage++) {
        for (uint i = lane_id; i < num_twiddles_this_stage; i += 32) {
            twiddles_this_stage[i] = get_twiddle(twiddle_data, i + exchg_region_offset);
        }
        twiddles_this_stage += num_twiddles_this_stage;
        num_twiddles_this_stage >>= 1;
        exchg_region_offset >>= 1;
    }

    // loads final 31 twiddles with minimal divergence
    const uint lz = clz(lane_id);
    const uint stage_offset = 5 - (32 - lz);
    const uint mask = (1u << (32 - lz)) - 1;
    if (lane_id > 0) {
        exchg_region_offset >>= stage_offset;
        twiddles_this_stage[lane_id ^ 31] = get_twiddle(twiddle_data, (lane_id ^ mask) + exchg_region_offset);
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// Load noninitial twiddles for warp-level stages
template <uint LOG_VALS_PER_THREAD, bool inverse>
DEVICE_FORCEINLINE void load_noninitial_twiddles_warp(threadgroup e2f* twiddle_cache,
                                                       const uint lane_id,
                                                       const uint warp_id,
                                                       const uint block_exchg_region_offset,
                                                       const thread powers_data_2_layer& twiddle_data) {
    constexpr uint NUM_INTRAWARP_STAGES = LOG_VALS_PER_THREAD + 1;
    uint num_twiddles_first_stage = 1u << LOG_VALS_PER_THREAD;
    uint exchg_region_offset = block_exchg_region_offset + warp_id * num_twiddles_first_stage;

    if (lane_id > 0 && lane_id < 2 * num_twiddles_first_stage) {
        const uint lz = clz(lane_id);
        const uint stage_offset = NUM_INTRAWARP_STAGES - (32 - lz);
        const uint mask = (1u << (32 - lz)) - 1;
        exchg_region_offset >>= stage_offset;
        twiddle_cache[lane_id ^ (2 * num_twiddles_first_stage - 1)] = get_twiddle(twiddle_data, (lane_id ^ mask) + exchg_region_offset);
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// --- Vectorized e2 matrix load/store helpers ---
// In CUDA these use vectorized_e2_matrix_getter/setter.
// In Metal, we pass raw bf* pointers with a stride parameter.
// Each "column pair" occupies two bf columns (c0, c1) separated by stride.

DEVICE_FORCEINLINE e2f load_e2(const device bf* ptr, uint row, uint stride) {
    e2f val;
    val.coefficients[0] = ptr[row];
    val.coefficients[1] = ptr[row + stride];
    return val;
}

DEVICE_FORCEINLINE void store_e2(device bf* ptr, uint row, uint stride, const e2f val) {
    ptr[row] = val.coefficients[0];
    ptr[row + stride] = val.coefficients[1];
}

} // namespace ntt
} // namespace airbender
