#pragma once

#include "context.metal"
#include "field.metal"
#include "memory.metal"

using namespace airbender::field;
using namespace airbender::memory;

// --- get_powers: compute base^(i + offset) for each thread ---

template <typename F>
DEVICE_FORCEINLINE void get_powers_impl(
    const F base, const uint offset, const bool bit_reverse,
    device F* result, const uint count, const uint gid
) {
    if (gid >= count) return;
    const uint power = (bit_reverse ? reverse_bits(gid) : gid) + offset;
    const F value = F::pow(base, power);
    result[gid] = value;
}

kernel void ab_get_powers_by_val_bf_kernel(
    constant base_field& base [[buffer(0)]],
    constant uint& offset [[buffer(1)]],
    constant uint& bit_reverse [[buffer(2)]],
    device base_field* result [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    get_powers_impl<base_field>(base, offset, bit_reverse != 0, result, count, gid);
}

kernel void ab_get_powers_by_val_e2_kernel(
    constant ext2_field& base [[buffer(0)]],
    constant uint& offset [[buffer(1)]],
    constant uint& bit_reverse [[buffer(2)]],
    device ext2_field* result [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    get_powers_impl<ext2_field>(base, offset, bit_reverse != 0, result, count, gid);
}

kernel void ab_get_powers_by_val_e4_kernel(
    constant ext4_field& base [[buffer(0)]],
    constant uint& offset [[buffer(1)]],
    constant uint& bit_reverse [[buffer(2)]],
    device ext4_field* result [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    get_powers_impl<ext4_field>(base, offset, bit_reverse != 0, result, count, gid);
}

kernel void ab_get_powers_by_ref_bf_kernel(
    const device base_field* base [[buffer(0)]],
    constant uint& offset [[buffer(1)]],
    constant uint& bit_reverse [[buffer(2)]],
    device base_field* result [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    get_powers_impl<base_field>(*base, offset, bit_reverse != 0, result, count, gid);
}

kernel void ab_get_powers_by_ref_e2_kernel(
    const device ext2_field* base [[buffer(0)]],
    constant uint& offset [[buffer(1)]],
    constant uint& bit_reverse [[buffer(2)]],
    device ext2_field* result [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    get_powers_impl<ext2_field>(*base, offset, bit_reverse != 0, result, count, gid);
}

kernel void ab_get_powers_by_ref_e4_kernel(
    const device ext4_field* base [[buffer(0)]],
    constant uint& offset [[buffer(1)]],
    constant uint& bit_reverse [[buffer(2)]],
    device ext4_field* result [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    get_powers_impl<ext4_field>(*base, offset, bit_reverse != 0, result, count, gid);
}

// --- batch_inv: batched field inversion using Montgomery's trick ---

template <typename T, int INV_BATCH, bool batch_is_full>
DEVICE_FORCEINLINE void batch_inv_registers(const thread T* inputs, thread T* fwd_scan_and_outputs, int runtime_batch_size) {
    T running_prod = T::one();
    for (int i = 0; i < INV_BATCH; i++)
        if (batch_is_full || i < runtime_batch_size) {
            fwd_scan_and_outputs[i] = running_prod;
            running_prod = T::mul(running_prod, inputs[i]);
        }

    T inv = T::inv(running_prod);

    for (int i = INV_BATCH - 1; i >= 0; i--) {
        if (batch_is_full || i < runtime_batch_size) {
            const auto input = inputs[i];
            fwd_scan_and_outputs[i] = T::mul(fwd_scan_and_outputs[i], inv);
            if (i > 0)
                inv = T::mul(inv, input);
        }
    }
}

template <typename T, int INV_BATCH>
DEVICE_FORCEINLINE void batch_inv_impl(
    const device T* src, device T* dst, const uint count,
    const uint gid, const uint grid_size
) {
    if (gid >= count) return;

    T inputs[INV_BATCH];
    T outputs[INV_BATCH];

    int runtime_batch_size = 0;
    uint g = gid;
    for (int i = 0; i < INV_BATCH; i++, g += grid_size)
        if (g < count) {
            inputs[i] = src[g];
            runtime_batch_size++;
        }

    if (runtime_batch_size < INV_BATCH) {
        batch_inv_registers<T, INV_BATCH, false>(inputs, outputs, runtime_batch_size);
    } else {
        batch_inv_registers<T, INV_BATCH, true>(inputs, outputs, runtime_batch_size);
    }

    g -= grid_size;
    for (int i = INV_BATCH - 1; i >= 0; --i, g -= grid_size)
        if (i < runtime_batch_size)
            dst[g] = outputs[i];
}

kernel void ab_batch_inv_bf_kernel(
    const device base_field* src [[buffer(0)]],
    device base_field* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    batch_inv_impl<base_field, 20>(src, dst, count, gid, grid_size);
}

kernel void ab_batch_inv_e2_kernel(
    const device ext2_field* src [[buffer(0)]],
    device ext2_field* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    batch_inv_impl<ext2_field, 5>(src, dst, count, gid, grid_size);
}

kernel void ab_batch_inv_e4_kernel(
    const device ext4_field* src [[buffer(0)]],
    device ext4_field* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    batch_inv_impl<ext4_field, 3>(src, dst, count, gid, grid_size);
}

// --- bit_reverse_naive: simple bit-reversal permutation ---

template <class T>
DEVICE_FORCEINLINE void bit_reverse_naive_impl(
    const device T* src, const uint src_stride,
    device T* dst, const uint dst_stride,
    const uint log_count, const uint row, const uint col
) {
    if (row >= 1u << log_count) return;
    const uint l_index = row;
    const uint r_index = reverse_bits(l_index) >> (32 - log_count);
    if (l_index > r_index) return;
    const T l_value = src[l_index + col * src_stride];
    const T r_value = src[r_index + col * src_stride];
    dst[l_index + col * dst_stride] = r_value;
    dst[r_index + col * dst_stride] = l_value;
}

kernel void ab_bit_reverse_naive_bf_kernel(
    const device base_field* src [[buffer(0)]],
    constant uint& src_stride [[buffer(1)]],
    device base_field* dst [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& log_count [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    bit_reverse_naive_impl<base_field>(src, src_stride, dst, dst_stride, log_count, tid.x, tid.y);
}

kernel void ab_bit_reverse_naive_e2_kernel(
    const device ext2_field* src [[buffer(0)]],
    constant uint& src_stride [[buffer(1)]],
    device ext2_field* dst [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& log_count [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    bit_reverse_naive_impl<ext2_field>(src, src_stride, dst, dst_stride, log_count, tid.x, tid.y);
}

kernel void ab_bit_reverse_naive_e4_kernel(
    const device ext4_field* src [[buffer(0)]],
    constant uint& src_stride [[buffer(1)]],
    device ext4_field* dst [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& log_count [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    bit_reverse_naive_impl<ext4_field>(src, src_stride, dst, dst_stride, log_count, tid.x, tid.y);
}

// Digest = 8 x base_field
struct dg {
    base_field values[8];
};

kernel void ab_bit_reverse_naive_dg_kernel(
    const device dg* src [[buffer(0)]],
    constant uint& src_stride [[buffer(1)]],
    device dg* dst [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& log_count [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    bit_reverse_naive_impl<dg>(src, src_stride, dst, dst_stride, log_count, tid.x, tid.y);
}

// --- fold kernel ---

kernel void ab_fold_kernel(
    const device ext4_field* challenge [[buffer(0)]],
    const device ext4_field* src [[buffer(1)]],
    device ext4_field* dst [[buffer(2)]],
    constant uint& root_offset [[buffer(3)]],
    constant uint& log_count [[buffer(4)]],
    const device ext2_field* powers_fine [[buffer(5)]],
    constant uint& powers_fine_mask [[buffer(6)]],
    constant uint& powers_fine_log_count [[buffer(7)]],
    const device ext2_field* powers_coarser [[buffer(8)]],
    constant uint& powers_coarser_mask [[buffer(9)]],
    constant uint& powers_coarser_log_count [[buffer(10)]],
    const device ext2_field* powers_coarsest [[buffer(11)]],
    constant uint& powers_coarsest_mask [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 1u << log_count) return;

    const ext4_field even = src[2 * gid];
    const ext4_field odd = src[2 * gid + 1];
    const ext4_field sum = ext4_field::add(even, odd);
    ext4_field diff = ext4_field::sub(even, odd);

    const uint root_index = reverse_bits(gid + root_offset) >> (32 - CIRCLE_GROUP_LOG_ORDER + 1);

    // Inline get_power_of_w with inverse=true
    const uint idx = (1u << CIRCLE_GROUP_LOG_ORDER) - root_index;

    powers_layer_data fine_data;
    fine_data.values = powers_fine;
    fine_data.mask = powers_fine_mask;
    fine_data.log_count = powers_fine_log_count;

    powers_layer_data coarser_data;
    coarser_data.values = powers_coarser;
    coarser_data.mask = powers_coarser_mask;
    coarser_data.log_count = powers_coarser_log_count;

    powers_layer_data coarsest_data;
    coarsest_data.values = powers_coarsest;
    coarsest_data.mask = powers_coarsest_mask;
    coarsest_data.log_count = 0; // not used for indexing

    const uint coarsest_idx = (idx >> (fine_data.log_count + coarser_data.log_count)) & powers_coarsest_mask;
    ext2_field root = powers_coarsest[coarsest_idx];

    const uint coarser_idx = (idx >> fine_data.log_count) & powers_coarser_mask;
    if (coarser_idx != 0)
        root = ext2_field::mul(root, powers_coarser[coarser_idx]);

    const uint fine_idx = idx & powers_fine_mask;
    if (fine_idx != 0)
        root = ext2_field::mul(root, powers_fine[fine_idx]);

    diff = ext4_field::mul(diff, root);
    diff = ext4_field::mul(diff, *challenge);

    const ext4_field result = ext4_field::add(sum, diff);
    dst[gid] = result;
}
