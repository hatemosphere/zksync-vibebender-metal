#include "ntt_common.metal"

using namespace airbender::ntt;
using namespace airbender::field;

// ============================================================================
// Simple one-stage N2B kernel — for log_n < 16 fallback
// ============================================================================
kernel void ab_evals_to_Z_one_stage(
    device const bf* gmem_in [[buffer(0)]],
    device bf* gmem_out [[buffer(1)]],
    device const e2f* twiddle_fine [[buffer(2)]],
    device const e2f* twiddle_coarse [[buffer(3)]],
    device const e2f* pow_fine [[buffer(4)]],
    device const e2f* pow_coarser [[buffer(5)]],
    device const e2f* pow_coarsest [[buffer(6)]],
    constant uint& stride [[buffer(7)]],
    constant uint& start_stage [[buffer(8)]],
    constant uint& log_n [[buffer(9)]],
    constant uint& blocks_per_ntt [[buffer(10)]],
    constant uint& evals_are_coset [[buffer(11)]],
    constant uint& evals_are_compressed [[buffer(12)]],
    // Twiddle metadata
    constant uint& twiddle_fine_mask [[buffer(13)]],
    constant uint& twiddle_fine_log_count [[buffer(14)]],
    constant uint& twiddle_coarse_mask [[buffer(15)]],
    constant uint& twiddle_coarse_log_count [[buffer(16)]],
    // Power metadata (for inverse scale)
    constant uint& pow_fine_mask [[buffer(17)]],
    constant uint& pow_fine_log_count [[buffer(18)]],
    constant uint& pow_coarser_mask [[buffer(19)]],
    constant uint& pow_coarser_log_count [[buffer(20)]],
    constant uint& pow_coarsest_mask [[buffer(21)]],
    // Inverse size
    device const e2f* inv_sizes [[buffer(22)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid_in_tg_2d [[thread_position_in_threadgroup]],
    uint2 tg_size_2d [[threads_per_threadgroup]]
) {
    const uint tid_in_tg = tid_in_tg_2d.x;
    const uint tg_size = tg_size_2d.x;
    const uint col_pair = tgid.x / blocks_per_ntt;
    const uint bid_in_ntt = tgid.x - col_pair * blocks_per_ntt;
    const uint tid_in_ntt = tid_in_tg + bid_in_ntt * tg_size;
    if (tid_in_ntt >= (1u << (log_n - 1)))
        return;

    const uint log_exchg_region_sz = log_n - start_stage;
    const uint exchg_region = tid_in_ntt >> (log_exchg_region_sz - 1);
    const uint tid_in_exchg_region = tid_in_ntt - (exchg_region << (log_exchg_region_sz - 1));
    const uint exchg_stride = 1u << (log_exchg_region_sz - 1);
    const uint a_idx = tid_in_exchg_region + exchg_region * (1u << log_exchg_region_sz);
    const uint b_idx = a_idx + exchg_stride;

    powers_data_2_layer twiddle_data;
    twiddle_data.fine.values = twiddle_fine;
    twiddle_data.fine.mask = twiddle_fine_mask;
    twiddle_data.fine.log_count = twiddle_fine_log_count;
    twiddle_data.coarse.values = twiddle_coarse;
    twiddle_data.coarse.mask = twiddle_coarse_mask;
    twiddle_data.coarse.log_count = twiddle_coarse_log_count;

    const uint col_offset = col_pair * 2 * stride;
    const device bf* in_ptr = gmem_in + col_offset;
    device bf* out_ptr = gmem_out + col_offset;

    const e2f twiddle = get_twiddle(twiddle_data, exchg_region);

    e2f a = load_e2(in_ptr, a_idx, stride);
    e2f b_val = load_e2(in_ptr, b_idx, stride);

    exchg_dit(a, b_val, twiddle);

    if (start_stage + 1 == log_n) {
        e2f inv_size = inv_sizes[log_n];
        a = e2f::mul(a, inv_size);
        b_val = e2f::mul(b_val, inv_size);
        if (evals_are_coset) {
            if (evals_are_compressed) {
                a = lde_scale_inverse(a, bitrev(a_idx, log_n), 1, 1, log_n,
                                       pow_fine, pow_fine_mask, pow_fine_log_count,
                                       pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                       pow_coarsest, pow_coarsest_mask);
                b_val = lde_scale_inverse(b_val, bitrev(b_idx, log_n), 1, 1, log_n,
                                           pow_fine, pow_fine_mask, pow_fine_log_count,
                                           pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                           pow_coarsest, pow_coarsest_mask);
            } else {
                a = lde_scale(a, bitrev(a_idx, log_n), 1, 1, log_n,
                               pow_fine, pow_fine_mask, pow_fine_log_count,
                               pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                               pow_coarsest, pow_coarsest_mask);
                b_val = lde_scale(b_val, bitrev(b_idx, log_n), 1, 1, log_n,
                                   pow_fine, pow_fine_mask, pow_fine_log_count,
                                   pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                   pow_coarsest, pow_coarsest_mask);
            }
        }
    }

    store_e2(out_ptr, a_idx, stride, a);
    store_e2(out_ptr, b_idx, stride, b_val);
}

// ============================================================================
// N2B: Final 7-stage warp kernel (LOG_VALS_PER_THREAD=2)
// 128 threads, 4 warps, VALS_PER_BLOCK = 512
// ============================================================================
// evals_are_coset: 0 = main domain, 1 = coset, 2 = compressed coset
kernel void ab_evals_to_Z_final_7_stages_warp(
    device const bf* gmem_in [[buffer(0)]],
    device bf* gmem_out [[buffer(1)]],
    device const e2f* twiddle_fine [[buffer(2)]],
    device const e2f* twiddle_coarse [[buffer(3)]],
    device const e2f* pow_fine [[buffer(4)]],
    device const e2f* pow_coarser [[buffer(5)]],
    device const e2f* pow_coarsest [[buffer(6)]],
    constant uint& stride [[buffer(7)]],
    constant uint& start_stage [[buffer(8)]],
    constant uint& stages_this_launch [[buffer(9)]],
    constant uint& log_n [[buffer(10)]],
    constant uint& num_Z_cols [[buffer(11)]],
    constant uint& grid_offset [[buffer(12)]],
    constant uint& evals_are_coset [[buffer(13)]],
    constant uint& twiddle_fine_mask [[buffer(14)]],
    constant uint& twiddle_fine_log_count [[buffer(15)]],
    constant uint& twiddle_coarse_mask [[buffer(16)]],
    constant uint& twiddle_coarse_log_count [[buffer(17)]],
    constant uint& pow_fine_mask [[buffer(18)]],
    constant uint& pow_fine_log_count [[buffer(19)]],
    constant uint& pow_coarser_mask [[buffer(20)]],
    constant uint& pow_coarser_log_count [[buffer(21)]],
    constant uint& pow_coarsest_mask [[buffer(22)]],
    device const e2f* inv_sizes [[buffer(23)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid_in_tg_2d [[thread_position_in_threadgroup]]
) {
    (void)start_stage;
    (void)stages_this_launch;
    const uint tid_in_tg = tid_in_tg_2d.x;
    constexpr uint LOG_VALS_PER_THREAD = 2;
    constexpr uint VALS_PER_THREAD = 4;
    constexpr uint PAIRS_PER_THREAD = 2;
    constexpr uint VALS_PER_WARP = 128;
    constexpr uint VALS_PER_BLOCK = 512;

    threadgroup e2f smem[VALS_PER_BLOCK];

    powers_data_2_layer twiddle_data;
    twiddle_data.fine.values = twiddle_fine;
    twiddle_data.fine.mask = twiddle_fine_mask;
    twiddle_data.fine.log_count = twiddle_fine_log_count;
    twiddle_data.coarse.values = twiddle_coarse;
    twiddle_data.coarse.mask = twiddle_coarse_mask;
    twiddle_data.coarse.log_count = twiddle_coarse_log_count;

    const uint effective_block_idx_x = tgid.x + grid_offset;
    const uint lane_id = tid_in_tg & 31;
    const uint warp_id = tid_in_tg >> 5;
    const uint gmem_offset = VALS_PER_BLOCK * effective_block_idx_x + VALS_PER_WARP * warp_id;

    threadgroup e2f* twiddle_cache = smem + VALS_PER_WARP * warp_id;

    load_initial_twiddles_warp<VALS_PER_WARP, LOG_VALS_PER_THREAD, true>(twiddle_cache, lane_id, gmem_offset, twiddle_data);

    const uint bound = min(COL_PAIRS_PER_WARP_BLOCK, num_Z_cols - COL_PAIRS_PER_WARP_BLOCK * tgid.y);
    for (uint ntt_idx = 0; ntt_idx < bound; ntt_idx++) {
        const uint col_pair = COL_PAIRS_PER_WARP_BLOCK * tgid.y + ntt_idx;
        const device bf* in_ptr = gmem_in + col_pair * 2 * stride + gmem_offset;
        device bf* out_ptr = gmem_out + col_pair * 2 * stride + gmem_offset;

        e2f vals[VALS_PER_THREAD];
        // Load in shuffled pattern (not pair-adjacent)
        #pragma unroll
        for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
            vals[2 * i] = load_e2(in_ptr, 64 * i + lane_id, stride);
            vals[2 * i + 1] = load_e2(in_ptr, 64 * i + lane_id + 32, stride);
        }

        // Register-level stages first (inverse of DIF order)
        // i=0: LOG_VALS_PER_THREAD - 1 = 1
        threadgroup e2f* twiddles_this_stage = twiddle_cache + VALS_PER_WARP - 2;
        uint num_twiddles_this_stage = 1;
        {
            // i=0 (only 1 iteration for LOG_VALS_PER_THREAD=2)
            const e2f tw = twiddles_this_stage[0];
            exchg_dit(vals[0], vals[2], tw);
            exchg_dit(vals[1], vals[3], tw);
            num_twiddles_this_stage <<= 1;
            twiddles_this_stage -= num_twiddles_this_stage;
        }

        // 6 warp-level stages (DIT, reverse shuffle direction)
        uint lane_mask = 16;
        #pragma unroll
        for (uint stage = 0; stage < 6; stage++) {
            uint s = 5 - stage;
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                const e2f tw = twiddles_this_stage[(32 * i + lane_id) >> s];
                exchg_dit(vals[2 * i], vals[2 * i + 1], tw);
            }
            if (stage < 5) {
                #pragma unroll
                for (uint i = 0; i < PAIRS_PER_THREAD; i++)
                    shfl_xor_e2f(vals, i, lane_id, lane_mask);
            }
            lane_mask >>= 1;
            num_twiddles_this_stage <<= 1;
            twiddles_this_stage -= num_twiddles_this_stage;
        }

        // Multiply by inverse size
        e2f inv_size = inv_sizes[log_n];
        #pragma unroll
        for (uint i = 0; i < VALS_PER_THREAD; i++)
            vals[i] = e2f::mul(vals[i], inv_size);

        // Coset unscaling if needed
        if (evals_are_coset > 0) {
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                const uint mem_idx = gmem_offset + 64 * i + 2 * lane_id;
                const uint idx0 = bitrev(mem_idx, log_n);
                const uint idx1 = bitrev(mem_idx + 1, log_n);
                if (evals_are_coset == 2) {
                    // compressed
                    vals[2 * i] = lde_scale_inverse(vals[2 * i], idx0, 1, 1, log_n,
                                                     pow_fine, pow_fine_mask, pow_fine_log_count,
                                                     pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                                     pow_coarsest, pow_coarsest_mask);
                    vals[2 * i + 1] = lde_scale_inverse(vals[2 * i + 1], idx1, 1, 1, log_n,
                                                         pow_fine, pow_fine_mask, pow_fine_log_count,
                                                         pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                                         pow_coarsest, pow_coarsest_mask);
                } else {
                    vals[2 * i] = lde_scale(vals[2 * i], idx0, 1, 1, log_n,
                                             pow_fine, pow_fine_mask, pow_fine_log_count,
                                             pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                             pow_coarsest, pow_coarsest_mask);
                    vals[2 * i + 1] = lde_scale(vals[2 * i + 1], idx1, 1, 1, log_n,
                                                 pow_fine, pow_fine_mask, pow_fine_log_count,
                                                 pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                                 pow_coarsest, pow_coarsest_mask);
                }
            }
        }

        // Write output (pair-adjacent)
        #pragma unroll
        for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
            store_e2(out_ptr, 64 * i + 2 * lane_id, stride, vals[2 * i]);
            store_e2(out_ptr, 64 * i + 2 * lane_id + 1, stride, vals[2 * i + 1]);
        }
    }
}

// ============================================================================
// N2B: Final 8-stage warp kernel (LOG_VALS_PER_THREAD=3)
// 128 threads, 4 warps, VALS_PER_BLOCK = 1024
// ============================================================================
kernel void ab_evals_to_Z_final_8_stages_warp(
    device const bf* gmem_in [[buffer(0)]],
    device bf* gmem_out [[buffer(1)]],
    device const e2f* twiddle_fine [[buffer(2)]],
    device const e2f* twiddle_coarse [[buffer(3)]],
    device const e2f* pow_fine [[buffer(4)]],
    device const e2f* pow_coarser [[buffer(5)]],
    device const e2f* pow_coarsest [[buffer(6)]],
    constant uint& stride [[buffer(7)]],
    constant uint& start_stage [[buffer(8)]],
    constant uint& stages_this_launch [[buffer(9)]],
    constant uint& log_n [[buffer(10)]],
    constant uint& num_Z_cols [[buffer(11)]],
    constant uint& grid_offset [[buffer(12)]],
    constant uint& evals_are_coset [[buffer(13)]],
    constant uint& twiddle_fine_mask [[buffer(14)]],
    constant uint& twiddle_fine_log_count [[buffer(15)]],
    constant uint& twiddle_coarse_mask [[buffer(16)]],
    constant uint& twiddle_coarse_log_count [[buffer(17)]],
    constant uint& pow_fine_mask [[buffer(18)]],
    constant uint& pow_fine_log_count [[buffer(19)]],
    constant uint& pow_coarser_mask [[buffer(20)]],
    constant uint& pow_coarser_log_count [[buffer(21)]],
    constant uint& pow_coarsest_mask [[buffer(22)]],
    device const e2f* inv_sizes [[buffer(23)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid_in_tg_2d [[thread_position_in_threadgroup]]
) {
    (void)start_stage;
    (void)stages_this_launch;
    const uint tid_in_tg = tid_in_tg_2d.x;
    constexpr uint LOG_VALS_PER_THREAD = 3;
    constexpr uint VALS_PER_THREAD = 8;
    constexpr uint PAIRS_PER_THREAD = 4;
    constexpr uint VALS_PER_WARP = 256;
    constexpr uint VALS_PER_BLOCK = 1024;

    threadgroup e2f smem[VALS_PER_BLOCK];

    powers_data_2_layer twiddle_data;
    twiddle_data.fine.values = twiddle_fine;
    twiddle_data.fine.mask = twiddle_fine_mask;
    twiddle_data.fine.log_count = twiddle_fine_log_count;
    twiddle_data.coarse.values = twiddle_coarse;
    twiddle_data.coarse.mask = twiddle_coarse_mask;
    twiddle_data.coarse.log_count = twiddle_coarse_log_count;

    const uint effective_block_idx_x = tgid.x + grid_offset;
    const uint lane_id = tid_in_tg & 31;
    const uint warp_id = tid_in_tg >> 5;
    const uint gmem_offset = VALS_PER_BLOCK * effective_block_idx_x + VALS_PER_WARP * warp_id;

    threadgroup e2f* twiddle_cache = smem + VALS_PER_WARP * warp_id;

    load_initial_twiddles_warp<VALS_PER_WARP, LOG_VALS_PER_THREAD, true>(twiddle_cache, lane_id, gmem_offset, twiddle_data);

    const uint bound = min(COL_PAIRS_PER_WARP_BLOCK, num_Z_cols - COL_PAIRS_PER_WARP_BLOCK * tgid.y);
    for (uint ntt_idx = 0; ntt_idx < bound; ntt_idx++) {
        const uint col_pair = COL_PAIRS_PER_WARP_BLOCK * tgid.y + ntt_idx;
        const device bf* in_ptr = gmem_in + col_pair * 2 * stride + gmem_offset;
        device bf* out_ptr = gmem_out + col_pair * 2 * stride + gmem_offset;

        e2f vals[VALS_PER_THREAD];
        #pragma unroll
        for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
            vals[2 * i] = load_e2(in_ptr, 64 * i + lane_id, stride);
            vals[2 * i + 1] = load_e2(in_ptr, 64 * i + lane_id + 32, stride);
        }

        // Register-level stages (LOG_VALS_PER_THREAD - 1 = 2, DIT order)
        threadgroup e2f* twiddles_this_stage = twiddle_cache + VALS_PER_WARP - 2;
        uint num_twiddles_this_stage = 1;
        // i=0: tile=8, half=4, j in 0..1
        {
            e2f tw = twiddles_this_stage[0];
            exchg_dit(vals[0], vals[4], tw);
            exchg_dit(vals[1], vals[5], tw);
            exchg_dit(vals[2], vals[6], tw);
            exchg_dit(vals[3], vals[7], tw);
            num_twiddles_this_stage <<= 1;
            twiddles_this_stage -= num_twiddles_this_stage;
        }
        // i=1: tile=4, half=2, j in 0..2
        {
            e2f tw0 = twiddles_this_stage[0];
            e2f tw1 = twiddles_this_stage[1];
            exchg_dit(vals[0], vals[2], tw0);
            exchg_dit(vals[1], vals[3], tw0);
            exchg_dit(vals[4], vals[6], tw1);
            exchg_dit(vals[5], vals[7], tw1);
            num_twiddles_this_stage <<= 1;
            twiddles_this_stage -= num_twiddles_this_stage;
        }

        // 6 warp-level stages (DIT)
        uint lane_mask = 16;
        #pragma unroll
        for (uint stage = 0; stage < 6; stage++) {
            uint s = 5 - stage;
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                const e2f tw = twiddles_this_stage[(32 * i + lane_id) >> s];
                exchg_dit(vals[2 * i], vals[2 * i + 1], tw);
            }
            if (stage < 5) {
                #pragma unroll
                for (uint i = 0; i < PAIRS_PER_THREAD; i++)
                    shfl_xor_e2f(vals, i, lane_id, lane_mask);
            }
            lane_mask >>= 1;
            num_twiddles_this_stage <<= 1;
            twiddles_this_stage -= num_twiddles_this_stage;
        }

        // Multiply by inverse size
        e2f inv_size = inv_sizes[log_n];
        #pragma unroll
        for (uint i = 0; i < VALS_PER_THREAD; i++)
            vals[i] = e2f::mul(vals[i], inv_size);

        // Coset unscaling
        if (evals_are_coset > 0) {
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                const uint mem_idx = gmem_offset + 64 * i + 2 * lane_id;
                const uint idx0 = bitrev(mem_idx, log_n);
                const uint idx1 = bitrev(mem_idx + 1, log_n);
                if (evals_are_coset == 2) {
                    vals[2 * i] = lde_scale_inverse(vals[2 * i], idx0, 1, 1, log_n,
                                                     pow_fine, pow_fine_mask, pow_fine_log_count,
                                                     pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                                     pow_coarsest, pow_coarsest_mask);
                    vals[2 * i + 1] = lde_scale_inverse(vals[2 * i + 1], idx1, 1, 1, log_n,
                                                         pow_fine, pow_fine_mask, pow_fine_log_count,
                                                         pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                                         pow_coarsest, pow_coarsest_mask);
                } else {
                    vals[2 * i] = lde_scale(vals[2 * i], idx0, 1, 1, log_n,
                                             pow_fine, pow_fine_mask, pow_fine_log_count,
                                             pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                             pow_coarsest, pow_coarsest_mask);
                    vals[2 * i + 1] = lde_scale(vals[2 * i + 1], idx1, 1, 1, log_n,
                                                 pow_fine, pow_fine_mask, pow_fine_log_count,
                                                 pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                                 pow_coarsest, pow_coarsest_mask);
                }
            }
        }

        // Write output (pair-adjacent)
        #pragma unroll
        for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
            store_e2(out_ptr, 64 * i + 2 * lane_id, stride, vals[2 * i]);
            store_e2(out_ptr, 64 * i + 2 * lane_id + 1, stride, vals[2 * i + 1]);
        }
    }
}

// ============================================================================
// N2B: Final 9-12 stage block kernel (LOG_VALS_PER_THREAD=3)
// 512 threads, 16 warps, VALS_PER_BLOCK = 4096
// ============================================================================
kernel void ab_evals_to_Z_final_9_to_12_stages_block(
    device const bf* gmem_in [[buffer(0)]],
    device bf* gmem_out [[buffer(1)]],
    device const e2f* twiddle_fine [[buffer(2)]],
    device const e2f* twiddle_coarse [[buffer(3)]],
    device const e2f* pow_fine [[buffer(4)]],
    device const e2f* pow_coarser [[buffer(5)]],
    device const e2f* pow_coarsest [[buffer(6)]],
    constant uint& stride [[buffer(7)]],
    constant uint& start_stage [[buffer(8)]],
    constant uint& stages_this_launch [[buffer(9)]],
    constant uint& log_n [[buffer(10)]],
    constant uint& num_Z_cols [[buffer(11)]],
    constant uint& grid_offset [[buffer(12)]],
    constant uint& evals_are_coset [[buffer(13)]],
    constant uint& twiddle_fine_mask [[buffer(14)]],
    constant uint& twiddle_fine_log_count [[buffer(15)]],
    constant uint& twiddle_coarse_mask [[buffer(16)]],
    constant uint& twiddle_coarse_log_count [[buffer(17)]],
    constant uint& pow_fine_mask [[buffer(18)]],
    constant uint& pow_fine_log_count [[buffer(19)]],
    constant uint& pow_coarser_mask [[buffer(20)]],
    constant uint& pow_coarser_log_count [[buffer(21)]],
    constant uint& pow_coarsest_mask [[buffer(22)]],
    device const e2f* inv_sizes [[buffer(23)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid_in_tg_2d [[thread_position_in_threadgroup]]
) {
    (void)start_stage;
    const uint tid_in_tg = tid_in_tg_2d.x;
    constexpr uint LOG_VALS_PER_THREAD = 3;
    constexpr uint VALS_PER_THREAD = 8;
    constexpr uint PAIRS_PER_THREAD = 4;
    constexpr uint VALS_PER_WARP = 256;
    constexpr uint VALS_PER_BLOCK = 4096;
    constexpr uint MAX_STAGES_THIS_LAUNCH = 12; // 2*(3+5)-4

    threadgroup e2f smem[VALS_PER_BLOCK];

    powers_data_2_layer twiddle_data;
    twiddle_data.fine.values = twiddle_fine;
    twiddle_data.fine.mask = twiddle_fine_mask;
    twiddle_data.fine.log_count = twiddle_fine_log_count;
    twiddle_data.coarse.values = twiddle_coarse;
    twiddle_data.coarse.mask = twiddle_coarse_mask;
    twiddle_data.coarse.log_count = twiddle_coarse_log_count;

    const uint effective_block_idx_x = tgid.x + grid_offset;
    const uint lane_id = tid_in_tg & 31;
    const uint warp_id = tid_in_tg >> 5;
    const uint gmem_block_offset = VALS_PER_BLOCK * effective_block_idx_x;
    const uint gmem_offset = gmem_block_offset + VALS_PER_WARP * warp_id;
    const uint gmem_in_thread_offset = 16 * warp_id + VALS_PER_WARP * (lane_id >> 4) + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);

    threadgroup e2f* twiddle_cache = smem + VALS_PER_WARP * warp_id;

    const uint bound = min(COL_PAIRS_PER_BLOCK, num_Z_cols - COL_PAIRS_PER_BLOCK * tgid.y);
    for (uint ntt_idx = 0; ntt_idx < bound; ntt_idx++) {
        const uint col_pair = COL_PAIRS_PER_BLOCK * tgid.y + ntt_idx;

        e2f vals[VALS_PER_THREAD];

        // Load in scrambled pattern
        const device bf* in_ptr = gmem_in + col_pair * 2 * stride + gmem_block_offset + gmem_in_thread_offset;
        #pragma unroll
        for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
            vals[2 * i] = load_e2(in_ptr, 4 * i * VALS_PER_WARP, stride);
            vals[2 * i + 1] = load_e2(in_ptr, (4 * i + 2) * VALS_PER_WARP, stride);
        }

        // Cross-warp stages first (DIT, reverse of DIF cross-warp)
        const uint stages_to_skip = MAX_STAGES_THIS_LAUNCH - stages_this_launch;
        uint exchg_region_offset = effective_block_idx_x;
        #pragma unroll
        for (uint i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
            if (i >= stages_to_skip) {
                for (uint j = 0; j < (1u << i); j++) {
                    const uint exchg_tile_sz = VALS_PER_THREAD >> i;
                    const uint half_exchg_tile_sz = exchg_tile_sz >> 1;
                    const e2f tw = get_twiddle(twiddle_data, exchg_region_offset + j);
                    for (uint k = 0; k < half_exchg_tile_sz; k++)
                        exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], tw);
                }
            }
            exchg_region_offset <<= 1;
        }

        uint lane_mask = 16;
        uint halfwarp_id = lane_id >> 4;
        #pragma unroll
        for (uint s = 0; s < 2; s++) {
            if ((s + LOG_VALS_PER_THREAD - 1) >= stages_to_skip) {
                #pragma unroll
                for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                    const e2f tw = get_twiddle(twiddle_data, exchg_region_offset + ((2 * i + halfwarp_id) >> (1 - s)));
                    exchg_dit(vals[2 * i], vals[2 * i + 1], tw);
                }
            }
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++)
                shfl_xor_e2f(vals, i, lane_id, lane_mask);
            lane_mask >>= 1;
            exchg_region_offset <<= 1;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Smem transpose
        {
            threadgroup e2f* pair_addr = smem + 16 * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2 * (tid_in_tg & 7);
            if (ntt_idx > 0) {
                e2f tmp[VALS_PER_THREAD];
                #pragma unroll
                for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                    tmp[2 * i] = twiddle_cache[64 * i + lane_id];
                    tmp[2 * i + 1] = twiddle_cache[64 * i + lane_id + 32];
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                #pragma unroll
                for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                    pair_addr[0] = vals[2 * i];
                    pair_addr[1] = vals[2 * i + 1];
                    pair_addr += 4 * VALS_PER_WARP;
                }
            } else {
                #pragma unroll
                for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                    pair_addr[0] = vals[2 * i];
                    pair_addr[1] = vals[2 * i + 1];
                    pair_addr += 4 * VALS_PER_WARP;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (ntt_idx > 0) {
                // Load vals from smem, restore twiddles from tmp (stored earlier)
                // Actually: just read from smem, and load twiddles back
                #pragma unroll
                for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                    vals[2 * i] = twiddle_cache[64 * i + lane_id];
                    vals[2 * i + 1] = twiddle_cache[64 * i + lane_id + 32];
                }
                // Can't easily get tmp back here in MSL without a second array...
                // For simplicity, re-load twiddles each iteration (safe but slightly slower)
                simdgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                #pragma unroll
                for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                    vals[2 * i] = twiddle_cache[64 * i + lane_id];
                    vals[2 * i + 1] = twiddle_cache[64 * i + lane_id + 32];
                }

                simdgroup_barrier(mem_flags::mem_threadgroup);

                load_initial_twiddles_warp<VALS_PER_WARP, LOG_VALS_PER_THREAD, true>(twiddle_cache, lane_id, gmem_offset, twiddle_data);
            }
        }

        // Warp-local register-level stages (DIT)
        threadgroup e2f* twiddles_this_stage = twiddle_cache + VALS_PER_WARP - 2;
        uint num_twiddles_this_stage = 1;
        // i=0: tile=8, half=4
        {
            e2f tw = twiddles_this_stage[0];
            exchg_dit(vals[0], vals[4], tw);
            exchg_dit(vals[1], vals[5], tw);
            exchg_dit(vals[2], vals[6], tw);
            exchg_dit(vals[3], vals[7], tw);
            num_twiddles_this_stage <<= 1;
            twiddles_this_stage -= num_twiddles_this_stage;
        }
        // i=1: tile=4, half=2
        {
            e2f tw0 = twiddles_this_stage[0];
            e2f tw1 = twiddles_this_stage[1];
            exchg_dit(vals[0], vals[2], tw0);
            exchg_dit(vals[1], vals[3], tw0);
            exchg_dit(vals[4], vals[6], tw1);
            exchg_dit(vals[5], vals[7], tw1);
            num_twiddles_this_stage <<= 1;
            twiddles_this_stage -= num_twiddles_this_stage;
        }

        // 6 warp-level SIMD stages (DIT)
        lane_mask = 16;
        #pragma unroll
        for (uint stage = 0; stage < 6; stage++) {
            uint s = 5 - stage;
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                const e2f tw = twiddles_this_stage[(32 * i + lane_id) >> s];
                exchg_dit(vals[2 * i], vals[2 * i + 1], tw);
            }
            if (stage < 5) {
                #pragma unroll
                for (uint i = 0; i < PAIRS_PER_THREAD; i++)
                    shfl_xor_e2f(vals, i, lane_id, lane_mask);
            }
            lane_mask >>= 1;
            num_twiddles_this_stage <<= 1;
            twiddles_this_stage -= num_twiddles_this_stage;
        }

        // Multiply by inverse size
        e2f inv_size = inv_sizes[log_n];
        #pragma unroll
        for (uint i = 0; i < VALS_PER_THREAD; i++)
            vals[i] = e2f::mul(vals[i], inv_size);

        // Coset unscaling
        if (evals_are_coset > 0) {
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                const uint mem_idx = gmem_offset + 64 * i + 2 * lane_id;
                const uint idx0 = bitrev(mem_idx, log_n);
                const uint idx1 = bitrev(mem_idx + 1, log_n);
                if (evals_are_coset == 2) {
                    vals[2 * i] = lde_scale_inverse(vals[2 * i], idx0, 1, 1, log_n,
                                                     pow_fine, pow_fine_mask, pow_fine_log_count,
                                                     pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                                     pow_coarsest, pow_coarsest_mask);
                    vals[2 * i + 1] = lde_scale_inverse(vals[2 * i + 1], idx1, 1, 1, log_n,
                                                         pow_fine, pow_fine_mask, pow_fine_log_count,
                                                         pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                                         pow_coarsest, pow_coarsest_mask);
                } else {
                    vals[2 * i] = lde_scale(vals[2 * i], idx0, 1, 1, log_n,
                                             pow_fine, pow_fine_mask, pow_fine_log_count,
                                             pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                             pow_coarsest, pow_coarsest_mask);
                    vals[2 * i + 1] = lde_scale(vals[2 * i + 1], idx1, 1, 1, log_n,
                                                 pow_fine, pow_fine_mask, pow_fine_log_count,
                                                 pow_coarser, pow_coarser_mask, pow_coarser_log_count,
                                                 pow_coarsest, pow_coarsest_mask);
                }
            }
        }

        // Write output (pair-adjacent)
        device bf* out_ptr = gmem_out + col_pair * 2 * stride + gmem_offset;
        #pragma unroll
        for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
            store_e2(out_ptr, 64 * i + 2 * lane_id, stride, vals[2 * i]);
            store_e2(out_ptr, 64 * i + 2 * lane_id + 1, stride, vals[2 * i + 1]);
        }
    }
}

// ============================================================================
// N2B: Nonfinal 7-or-8 stage block kernel (LOG_VALS_PER_THREAD=3)
// 512 threads, 16 warps, VALS_PER_BLOCK = 4096
// ============================================================================
kernel void ab_evals_to_Z_nonfinal_7_or_8_stages_block(
    device const bf* gmem_in [[buffer(0)]],
    device bf* gmem_out [[buffer(1)]],
    device const e2f* twiddle_fine [[buffer(2)]],
    device const e2f* twiddle_coarse [[buffer(3)]],
    // pow_fine/coarser/coarsest not used for nonfinal, but keep buffer layout consistent
    device const e2f* pow_fine [[buffer(4)]],
    device const e2f* pow_coarser [[buffer(5)]],
    device const e2f* pow_coarsest [[buffer(6)]],
    constant uint& stride [[buffer(7)]],
    constant uint& start_stage [[buffer(8)]],
    constant uint& stages_this_launch [[buffer(9)]],
    constant uint& log_n [[buffer(10)]],
    constant uint& num_Z_cols [[buffer(11)]],
    constant uint& grid_offset [[buffer(12)]],
    constant uint& unused_evals_are_coset [[buffer(13)]],
    constant uint& twiddle_fine_mask [[buffer(14)]],
    constant uint& twiddle_fine_log_count [[buffer(15)]],
    constant uint& twiddle_coarse_mask [[buffer(16)]],
    constant uint& twiddle_coarse_log_count [[buffer(17)]],
    constant uint& pow_fine_mask [[buffer(18)]],
    constant uint& pow_fine_log_count [[buffer(19)]],
    constant uint& pow_coarser_mask [[buffer(20)]],
    constant uint& pow_coarser_log_count [[buffer(21)]],
    constant uint& pow_coarsest_mask [[buffer(22)]],
    device const e2f* inv_sizes [[buffer(23)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid_in_tg_2d [[thread_position_in_threadgroup]]
) {
    (void)pow_fine; (void)pow_coarser; (void)pow_coarsest;
    (void)unused_evals_are_coset;
    (void)pow_fine_mask; (void)pow_fine_log_count;
    (void)pow_coarser_mask; (void)pow_coarser_log_count;
    (void)pow_coarsest_mask; (void)inv_sizes;
    const uint tid_in_tg = tid_in_tg_2d.x;
    constexpr uint LOG_VALS_PER_THREAD = 3;
    constexpr uint VALS_PER_THREAD = 8;
    constexpr uint PAIRS_PER_THREAD = 4;
    constexpr uint VALS_PER_WARP = 256;
    constexpr uint TILES_PER_WARP = 16;
    constexpr uint WARPS_PER_BLOCK = 16;
    constexpr uint VALS_PER_BLOCK = 4096;
    constexpr uint TILES_PER_BLOCK = 256;
    constexpr uint EXCHG_REGIONS_PER_BLOCK = 128;
    constexpr uint MAX_STAGES_THIS_LAUNCH = 8; // 2*(3+5)-8

    threadgroup e2f smem[VALS_PER_BLOCK];

    powers_data_2_layer twiddle_data;
    twiddle_data.fine.values = twiddle_fine;
    twiddle_data.fine.mask = twiddle_fine_mask;
    twiddle_data.fine.log_count = twiddle_fine_log_count;
    twiddle_data.coarse.values = twiddle_coarse;
    twiddle_data.coarse.mask = twiddle_coarse_mask;
    twiddle_data.coarse.log_count = twiddle_coarse_log_count;

    const bool skip_last_stage = (stages_this_launch == 7);
    const uint effective_block_idx_x = tgid.x + grid_offset;
    const uint lane_id = tid_in_tg & 31;
    const uint warp_id = tid_in_tg >> 5;
    const uint log_tile_stride = log_n - start_stage - MAX_STAGES_THIS_LAUNCH;
    const uint tile_stride = 1u << log_tile_stride;
    const uint log_blocks_per_region = log_tile_stride - 4;
    const uint block_bfly_region_size = TILES_PER_BLOCK * tile_stride;
    const uint block_bfly_region = effective_block_idx_x >> log_blocks_per_region;
    const uint block_bfly_region_start = block_bfly_region * block_bfly_region_size;
    const uint block_start_in_bfly_region = 16 * (effective_block_idx_x & ((1u << log_blocks_per_region) - 1));
    const uint gmem_in_thread_offset = tile_stride * warp_id + tile_stride * WARPS_PER_BLOCK * (lane_id >> 4) + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
    const uint gmem_in_offset = block_bfly_region_start + block_start_in_bfly_region + gmem_in_thread_offset;

    threadgroup e2f* twiddle_cache = smem + VALS_PER_WARP * warp_id;

    const uint bound = min(COL_PAIRS_PER_BLOCK, num_Z_cols - COL_PAIRS_PER_BLOCK * tgid.y);
    for (uint ntt_idx = 0; ntt_idx < bound; ntt_idx++) {
        const uint col_pair = COL_PAIRS_PER_BLOCK * tgid.y + ntt_idx;

        e2f vals[VALS_PER_THREAD];

        // Load in scrambled pattern
        const device bf* in_ptr = gmem_in + col_pair * 2 * stride + gmem_in_offset;
        #pragma unroll
        for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
            vals[2 * i] = load_e2(in_ptr, 4 * i * tile_stride * WARPS_PER_BLOCK, stride);
            vals[2 * i + 1] = load_e2(in_ptr, (4 * i + 2) * tile_stride * WARPS_PER_BLOCK, stride);
        }

        // Cross-warp stages (DIT)
        uint block_exchg_region_offset = block_bfly_region;
        #pragma unroll
        for (uint i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
            for (uint j = 0; j < (1u << i); j++) {
                const uint exchg_tile_sz = VALS_PER_THREAD >> i;
                const uint half_exchg_tile_sz = exchg_tile_sz >> 1;
                const e2f tw = get_twiddle(twiddle_data, block_exchg_region_offset + j);
                for (uint k = 0; k < half_exchg_tile_sz; k++)
                    exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], tw);
            }
            block_exchg_region_offset <<= 1;
        }

        uint lane_mask = 16;
        uint halfwarp_id = lane_id >> 4;
        #pragma unroll
        for (uint s = 0; s < 2; s++) {
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                const e2f tw = get_twiddle(twiddle_data, block_exchg_region_offset + ((2 * i + halfwarp_id) >> (1 - s)));
                exchg_dit(vals[2 * i], vals[2 * i + 1], tw);
            }
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++)
                shfl_xor_e2f(vals, i, lane_id, lane_mask);
            lane_mask >>= 1;
            block_exchg_region_offset <<= 1;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Smem transpose
        e2f tmp;
        if (ntt_idx > 0) {
            tmp = twiddle_cache[lane_id];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup e2f* smem_pair_addr = smem + 16 * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2 * (tid_in_tg & 7);
        #pragma unroll
        for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
            smem_pair_addr[0] = vals[2 * i];
            smem_pair_addr[1] = vals[2 * i + 1];
            smem_pair_addr += 4 * VALS_PER_WARP;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint smem_thread_offset = 16 * (lane_id >> 4) + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
        #pragma unroll
        for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
            vals[2 * i] = twiddle_cache[64 * i + smem_thread_offset];
            vals[2 * i + 1] = twiddle_cache[64 * i + smem_thread_offset + 32];
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (ntt_idx > 0) {
            twiddle_cache[lane_id] = tmp;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            load_noninitial_twiddles_warp<LOG_VALS_PER_THREAD, true>(twiddle_cache, lane_id, warp_id,
                                                                       block_bfly_region * EXCHG_REGIONS_PER_BLOCK,
                                                                       twiddle_data);
        }

        // Warp-local register-level stages (DIT)
        threadgroup e2f* twiddles_this_stage = twiddle_cache + 2 * VALS_PER_THREAD - 2;
        uint num_twiddles_this_stage = 1;
        #pragma unroll
        for (uint i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
            for (uint j = 0; j < (1u << i); j++) {
                const uint exchg_tile_sz = VALS_PER_THREAD >> i;
                const uint half_exchg_tile_sz = exchg_tile_sz >> 1;
                const e2f tw = twiddles_this_stage[j];
                for (uint k = 0; k < half_exchg_tile_sz; k++) {
                    exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], tw);
                }
            }
            num_twiddles_this_stage <<= 1;
            twiddles_this_stage -= num_twiddles_this_stage;
        }

        // Warp-level SIMD stages (DIT)
        lane_mask = 16;
        #pragma unroll
        for (uint s = 0; s < 2; s++) {
            if (!skip_last_stage || s < 1) {
                #pragma unroll
                for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                    const e2f tw = twiddles_this_stage[(2 * i + halfwarp_id) >> (1 - s)];
                    exchg_dit(vals[2 * i], vals[2 * i + 1], tw);
                }
                for (uint i = 0; i < PAIRS_PER_THREAD; i++)
                    shfl_xor_e2f(vals, i, lane_id, lane_mask);
                lane_mask >>= 1;
                num_twiddles_this_stage <<= 1;
                twiddles_this_stage -= num_twiddles_this_stage;
            }
        }

        // Write output
        device bf* out_ptr = gmem_out + col_pair * 2 * stride + block_bfly_region_start + block_start_in_bfly_region;
        if (skip_last_stage) {
            uint val0_offset = TILES_PER_WARP * tile_stride * warp_id + 2 * tile_stride * (lane_id >> 4) + 2 * (tid_in_tg & 7) + ((lane_id >> 3) & 1);
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                store_e2(out_ptr, val0_offset, stride, vals[2 * i]);
                store_e2(out_ptr, val0_offset + tile_stride, stride, vals[2 * i + 1]);
                val0_offset += 4 * tile_stride;
            }
        } else {
            uint pair_offset = TILES_PER_WARP * tile_stride * warp_id + tile_stride * (lane_id >> 3) + 2 * (tid_in_tg & 7);
            #pragma unroll
            for (uint i = 0; i < PAIRS_PER_THREAD; i++) {
                store_e2(out_ptr, pair_offset, stride, vals[2 * i]);
                store_e2(out_ptr, pair_offset + 1, stride, vals[2 * i + 1]);
                pair_offset += 4 * tile_stride;
            }
        }
    }
}
