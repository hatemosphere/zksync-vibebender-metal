#pragma once

#include "context.metal"
#include "field.metal"
#include "memory.metal"

using namespace airbender::field;
using namespace airbender::memory;

// --- Precompute common_factor for barycentric evaluation ---
// common_factor = coset * (z^N - coset^N) / (N * coset^N)

kernel void ab_barycentric_precompute_common_factor_kernel(
    const device ext4_field* z_ref [[buffer(0)]],
    device ext4_field* common_factor_ref [[buffer(1)]],
    constant ext2_field& coset [[buffer(2)]],
    constant ext2_field& decompression_factor [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;

    const ext4_field z = *z_ref;
    const ext2_field cosetN = ext2_field::pow(coset, count);
    const ext4_field zN = ext4_field::pow(z, count);
    const ext4_field num = ext4_field::mul(ext4_field::sub(zN, cosetN), coset);
    const ext2_field denom = ext2_field::mul(base_field{count}, cosetN);
    const ext2_field denom_inv = ext2_field::inv(denom);
    const ext2_field denom_inv_with_decompression = ext2_field::mul(denom_inv, decompression_factor);
    const ext4_field common_factor = ext4_field::mul(num, denom_inv_with_decompression);
    *common_factor_ref = common_factor;
}

// --- Precompute Lagrange coefficients ---

// Forward declare the batch_inv_registers from ops_complex
template <typename T, int INV_BATCH, bool batch_is_full>
DEVICE_FORCEINLINE void barycentric_batch_inv_registers(const thread T* inputs, thread T* fwd_scan_and_outputs, int runtime_batch_size) {
    T running_prod = T::one();
    #pragma unroll
    for (int i = 0; i < INV_BATCH; i++)
        if (batch_is_full || i < runtime_batch_size) {
            fwd_scan_and_outputs[i] = running_prod;
            running_prod = T::mul(running_prod, inputs[i]);
        }

    T inv = T::inv(running_prod);

    #pragma unroll
    for (int i = INV_BATCH - 1; i >= 0; i--) {
        if (batch_is_full || i < runtime_batch_size) {
            const auto input = inputs[i];
            fwd_scan_and_outputs[i] = T::mul(fwd_scan_and_outputs[i], inv);
            if (i > 0)
                inv = T::mul(inv, input);
        }
    }
}

kernel void ab_barycentric_precompute_lagrange_coeffs_kernel(
    const device ext4_field* z_ref [[buffer(0)]],
    const device ext4_field* common_factor_ref [[buffer(1)]],
    constant ext2_field& w_inv_step [[buffer(2)]],
    constant ext2_field& coset [[buffer(3)]],
    device ext4_field* lagrange_coeffs [[buffer(4)]],
    constant uint& log_count [[buffer(5)]],
    const device ext2_field* powers_fine [[buffer(6)]],
    constant uint& powers_fine_mask [[buffer(7)]],
    constant uint& powers_fine_log_count [[buffer(8)]],
    const device ext2_field* powers_coarser [[buffer(9)]],
    constant uint& powers_coarser_mask [[buffer(10)]],
    constant uint& powers_coarser_log_count [[buffer(11)]],
    const device ext2_field* powers_coarsest [[buffer(12)]],
    constant uint& powers_coarsest_mask [[buffer(13)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    constexpr int INV_BATCH = 4;

    const auto common_factor = *common_factor_ref;
    const uint count = 1u << log_count;
    if (gid >= count) return;

    ext4_field per_elem_factor_invs[INV_BATCH];

    const ext4_field z = *z_ref;
    uint runtime_batch_size = 0;
    const uint shift = CIRCLE_GROUP_LOG_ORDER - log_count;

    // get_power_of_w(gid << shift, true) - inline
    uint w_index = gid << shift;
    uint w_idx = (1u << CIRCLE_GROUP_LOG_ORDER) - w_index;

    uint coarsest_idx_w = (w_idx >> (powers_fine_log_count + powers_coarser_log_count)) & powers_coarsest_mask;
    ext2_field w_inv = powers_coarsest[coarsest_idx_w];
    uint coarser_idx_w = (w_idx >> powers_fine_log_count) & powers_coarser_mask;
    if (coarser_idx_w != 0)
        w_inv = ext2_field::mul(w_inv, powers_coarser[coarser_idx_w]);
    uint fine_idx_w = w_idx & powers_fine_mask;
    if (fine_idx_w != 0)
        w_inv = ext2_field::mul(w_inv, powers_fine[fine_idx_w]);

    for (uint i = 0, g = gid; i < INV_BATCH; i++, g += grid_size)
        if (g < count) {
            per_elem_factor_invs[i] = ext4_field::sub(ext4_field::mul(z, w_inv), coset);
            if (g + grid_size < count)
                w_inv = ext2_field::mul(w_inv, w_inv_step);
            runtime_batch_size++;
        }

    ext4_field per_elem_factors[INV_BATCH];

    if (runtime_batch_size < INV_BATCH) {
        barycentric_batch_inv_registers<ext4_field, INV_BATCH, false>(per_elem_factor_invs, per_elem_factors, runtime_batch_size);
    } else {
        barycentric_batch_inv_registers<ext4_field, INV_BATCH, true>(per_elem_factor_invs, per_elem_factors, runtime_batch_size);
    }

    for (uint i = 0, g = gid; i < INV_BATCH; i++, g += grid_size)
        if (g < count)
            lagrange_coeffs[g] = ext4_field::mul(per_elem_factors[i], common_factor);
}

// --- Barycentric evaluation: per-column partial reduction ---
// Each threadgroup reduces a chunk of rows for ONE column.
// Output: partial_sums[blockIdx] = sum of lagrange[i] * col_val[i] for rows in this chunk.
// A second pass sums the partial results.

// BF column variant: eval = sum(lagrange_e4[i] * bf_col[i])
kernel void ab_barycentric_eval_bf_col_partial_reduce(
    const device ext4_field* lagrange_coeffs [[buffer(0)]],
    const device base_field* column_data [[buffer(1)]],
    device ext4_field* partial_sums [[buffer(2)]],
    constant uint& count [[buffer(3)]],         // number of rows (trace_len)
    constant uint& col_stride [[buffer(4)]],    // stride between rows in column (= trace_len for col-major)
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint block_id [[threadgroup_position_in_grid]]
) {
    threadgroup ext4_field shared_sums[256];
    ext4_field local_sum = ext4_field::zero();

    for (uint i = gid; i < count; i += tg_size * ((count + tg_size - 1) / tg_size)) {
        if (i < count) {
            const auto coeff = lagrange_coeffs[i];
            const auto val = column_data[i]; // col-major: column starts at column_data, stride=1
            local_sum = ext4_field::add(local_sum, ext4_field::mul(coeff, val));
        }
    }

    shared_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] = ext4_field::add(shared_sums[tid], shared_sums[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sums[block_id] = shared_sums[0];
    }
}

// E4 column variant: eval = sum(lagrange_e4[i] * e4_col[i])
kernel void ab_barycentric_eval_e4_col_partial_reduce(
    const device ext4_field* lagrange_coeffs [[buffer(0)]],
    const device ext4_field* column_data [[buffer(1)]],
    device ext4_field* partial_sums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint block_id [[threadgroup_position_in_grid]]
) {
    threadgroup ext4_field shared_sums[256];
    ext4_field local_sum = ext4_field::zero();

    for (uint i = gid; i < count; i += tg_size * ((count + tg_size - 1) / tg_size)) {
        if (i < count) {
            const auto coeff = lagrange_coeffs[i];
            const auto val = column_data[i];
            local_sum = ext4_field::add(local_sum, ext4_field::mul(coeff, val));
        }
    }

    shared_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] = ext4_field::add(shared_sums[tid], shared_sums[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sums[block_id] = shared_sums[0];
    }
}

// Multi-column BF partial reduce: evaluate multiple BF columns in one dispatch.
// grid_dim = (num_blocks, num_cols), each Y slice processes one column.
kernel void ab_barycentric_eval_bf_multi_col_partial_reduce(
    const device ext4_field* lagrange_coeffs [[buffer(0)]],
    const device base_field* column_data [[buffer(1)]],
    device ext4_field* partial_sums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& col_stride [[buffer(4)]],
    constant uint& num_blocks_per_col [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint tg_size = 256;
    const uint col_idx = tgid.y;
    const uint block_id = tgid.x;
    const uint gid = block_id * tg_size + tid;
    const device base_field* col_ptr = column_data + col_idx * col_stride;
    device ext4_field* out = partial_sums + col_idx * num_blocks_per_col;

    threadgroup ext4_field shared_sums[256];
    ext4_field local_sum = ext4_field::zero();

    for (uint i = gid; i < count; i += tg_size * ((count + tg_size - 1) / tg_size)) {
        if (i < count) {
            const auto coeff = lagrange_coeffs[i];
            const auto val = col_ptr[i];
            local_sum = ext4_field::add(local_sum, ext4_field::mul(coeff, val));
        }
    }

    shared_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] = ext4_field::add(shared_sums[tid], shared_sums[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[block_id] = shared_sums[0];
    }
}

// Multi-column final reduce: sum partial results for multiple columns in one dispatch.
// grid_dim = (1, num_cols)
kernel void ab_barycentric_eval_multi_col_final_reduce(
    const device ext4_field* partial_sums [[buffer(0)]],
    device ext4_field* results [[buffer(1)]],
    constant uint& num_blocks [[buffer(2)]],
    constant uint& result_offset [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint col_idx = tgid.y;
    const device ext4_field* col_partials = partial_sums + col_idx * num_blocks;

    threadgroup ext4_field shared_sums[256];
    ext4_field local_sum = ext4_field::zero();

    for (uint i = tid; i < num_blocks; i += 256) {
        local_sum = ext4_field::add(local_sum, col_partials[i]);
    }

    shared_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] = ext4_field::add(shared_sums[tid], shared_sums[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        results[result_offset + col_idx] = shared_sums[0];
    }
}

// Final reduction: sum partial_sums[0..num_blocks] into result
kernel void ab_barycentric_eval_final_reduce(
    const device ext4_field* partial_sums [[buffer(0)]],
    device ext4_field* result [[buffer(1)]],
    constant uint& num_blocks [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup ext4_field shared_sums[256];
    ext4_field local_sum = ext4_field::zero();

    for (uint i = tid; i < num_blocks; i += 256) {
        local_sum = ext4_field::add(local_sum, partial_sums[i]);
    }

    shared_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] = ext4_field::add(shared_sums[tid], shared_sums[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        *result = shared_sums[0];
    }
}
