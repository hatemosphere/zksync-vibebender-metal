#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Scatter-add kernel for generic lookup multiplicities.
// Each thread reads a mapping index, computes destination, and atomically increments.
kernel void ab_scatter_add_multiplicities_kernel(
    const device uint *mapping [[buffer(0)]],
    device atomic_uint *witness [[buffer(1)]],
    constant uint &stride [[buffer(2)]],          // = trace_len
    constant uint &mult_cols_start [[buffer(3)]],
    constant uint &num_lookup_cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint stride_minus_1 = stride - 1;
    const uint total = num_lookup_cols * stride_minus_1;
    if (gid >= total)
        return;

    const uint lookup_col = gid / stride_minus_1;
    const uint row = gid % stride_minus_1;
    const uint abs_idx = mapping[lookup_col * stride + row];

    // Skip sentinel values (0xFFFFFFFF marks unused entries)
    if (abs_idx == 0xFFFFFFFF)
        return;

    const uint mult_col = abs_idx / stride_minus_1;
    const uint mult_row = abs_idx % stride_minus_1;
    const uint dst_idx = (mult_cols_start + mult_col) * stride + mult_row;
    atomic_fetch_add_explicit(&witness[dst_idx], 1u, memory_order_relaxed);
}
