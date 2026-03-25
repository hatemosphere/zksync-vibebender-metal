#include <metal_stdlib>
using namespace metal;

// Multi-pass 4-bit radix sort for u32 keys (and key-value pairs).
//
// Each pass:
// 1. Per-threadgroup histogram of a 4-bit digit
// 2. Prefix sum of histograms
// 3. Per-threadgroup scatter using offsets

constant constexpr uint RADIX_SORT_THREADS = 256;
constant constexpr uint RADIX_BITS = 4;
constant constexpr uint RADIX_SIZE = 1u << RADIX_BITS; // 16

// --- Pass 1: Compute per-block histograms ---

kernel void ab_radix_sort_histogram_kernel(
    const device uint* d_keys [[buffer(0)]],
    device uint* d_histograms [[buffer(1)]],
    constant uint& num_items [[buffer(2)]],
    constant uint& bit_offset [[buffer(3)]],
    constant uint& num_blocks [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup uint local_hist[RADIX_SIZE];

    // Initialize local histogram
    if (lid < RADIX_SIZE) {
        local_hist[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count digits for this block
    const uint block_start = tgid * RADIX_SORT_THREADS;
    const uint gid = block_start + lid;
    if (gid < num_items) {
        const uint key = d_keys[gid];
        const uint digit = (key >> bit_offset) & (RADIX_SIZE - 1);
        // Atomic add to shared memory histogram
        atomic_fetch_add_explicit(
            reinterpret_cast<threadgroup atomic_uint*>(&local_hist[digit]),
            1u,
            memory_order_relaxed
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write local histogram to global memory
    // Layout: d_histograms[digit * num_blocks + tgid]
    if (lid < RADIX_SIZE) {
        d_histograms[lid * num_blocks + tgid] = local_hist[lid];
    }
}

// --- Pass 2: Prefix sum of histograms ---
// Scans over RADIX_SIZE * num_blocks elements.
// For simplicity, uses a single threadgroup (works if num_blocks * RADIX_SIZE <= 65536).

kernel void ab_radix_sort_scan_histograms_kernel(
    device uint* d_histograms [[buffer(0)]],
    constant uint& total_bins [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    // Serial prefix sum within a single threadgroup
    // Each thread handles a strided portion
    threadgroup uint shared[4096]; // max total_bins we support

    // Load
    for (uint i = lid; i < total_bins; i += tpg) {
        shared[i] = d_histograms[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Single-thread prefix sum (only thread 0)
    if (lid == 0) {
        uint running = 0;
        for (uint i = 0; i < total_bins; i++) {
            uint val = shared[i];
            shared[i] = running;
            running += val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store back
    for (uint i = lid; i < total_bins; i += tpg) {
        d_histograms[i] = shared[i];
    }
}

// --- Pass 3: Scatter keys using prefix offsets ---
// Uses stable per-thread ranking (not atomics) to preserve relative order.

kernel void ab_radix_sort_scatter_keys_kernel(
    const device uint* d_keys_in [[buffer(0)]],
    device uint* d_keys_out [[buffer(1)]],
    const device uint* d_histograms [[buffer(2)]],
    constant uint& num_items [[buffer(3)]],
    constant uint& bit_offset [[buffer(4)]],
    constant uint& num_blocks [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup uint local_offsets[RADIX_SIZE];
    threadgroup uint local_digits[RADIX_SORT_THREADS];

    // Load this block's offsets from global prefix sum
    if (lid < RADIX_SIZE) {
        local_offsets[lid] = d_histograms[lid * num_blocks + tgid];
    }

    const uint gid = tgid * RADIX_SORT_THREADS + lid;
    const bool valid = gid < num_items;
    uint key = 0;
    uint digit = 0;
    if (valid) {
        key = d_keys_in[gid];
        digit = (key >> bit_offset) & (RADIX_SIZE - 1);
    }
    // Store digit so all threads can see each other's digits
    local_digits[lid] = valid ? digit : RADIX_SIZE; // sentinel for invalid
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        // Count how many preceding threads in this block have the same digit
        uint rank = 0;
        for (uint i = 0; i < lid; i++) {
            if (local_digits[i] == digit)
                rank++;
        }
        const uint dest = local_offsets[digit] + rank;
        d_keys_out[dest] = key;
    }
}

// --- Scatter key-value pairs ---

kernel void ab_radix_sort_scatter_pairs_kernel(
    const device uint* d_keys_in [[buffer(0)]],
    device uint* d_keys_out [[buffer(1)]],
    const device uint* d_values_in [[buffer(2)]],
    device uint* d_values_out [[buffer(3)]],
    const device uint* d_histograms [[buffer(4)]],
    constant uint& num_items [[buffer(5)]],
    constant uint& bit_offset [[buffer(6)]],
    constant uint& num_blocks [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup uint local_offsets[RADIX_SIZE];
    threadgroup uint local_digits[RADIX_SORT_THREADS];

    if (lid < RADIX_SIZE) {
        local_offsets[lid] = d_histograms[lid * num_blocks + tgid];
    }

    const uint gid = tgid * RADIX_SORT_THREADS + lid;
    const bool valid = gid < num_items;
    uint key = 0;
    uint value = 0;
    uint digit = 0;
    if (valid) {
        key = d_keys_in[gid];
        value = d_values_in[gid];
        digit = (key >> bit_offset) & (RADIX_SIZE - 1);
    }
    local_digits[lid] = valid ? digit : RADIX_SIZE;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
        uint rank = 0;
        for (uint i = 0; i < lid; i++) {
            if (local_digits[i] == digit)
                rank++;
        }
        const uint dest = local_offsets[digit] + rank;
        d_keys_out[dest] = key;
        d_values_out[dest] = value;
    }
}
