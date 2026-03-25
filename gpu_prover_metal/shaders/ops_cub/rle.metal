#include <metal_stdlib>
using namespace metal;

// Run-length encoding for u32 values.
//
// Uses 3 kernels:
// 1. Mark boundaries where consecutive values differ
// 2. Exclusive prefix sum of markers (done via scan kernels)
// 3. Scatter run starts and lengths

// --- Kernel 1: Mark run boundaries ---
// d_flags[i] = 1 if d_in[i] != d_in[i-1] (or i == 0), else 0

kernel void ab_rle_mark_boundaries_kernel(
    const device uint* d_in [[buffer(0)]],
    device uint* d_flags [[buffer(1)]],
    constant uint& num_items [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_items) return;

    uint flag;
    if (gid == 0) {
        flag = 1;
    } else {
        flag = (d_in[gid] != d_in[gid - 1]) ? 1u : 0u;
    }
    d_flags[gid] = flag;
}

// --- Kernel 2: Scatter unique values and run lengths ---
// d_positions contains the exclusive prefix sum of d_flags.
// Each boundary at position i writes:
//   d_unique_out[d_positions[i]] = d_in[i]
//   d_counts_out[d_positions[i]] = run length
// Run length is computed as: next_boundary_position - current_position
// where next_boundary_position is found from d_positions of the next boundary.

kernel void ab_rle_scatter_kernel(
    const device uint* d_in [[buffer(0)]],
    const device uint* d_positions [[buffer(1)]],
    device uint* d_unique_out [[buffer(2)]],
    device uint* d_counts_out [[buffer(3)]],
    device uint* d_num_runs_out [[buffer(4)]],
    constant uint& num_items [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_items) return;

    // Check if this is a boundary
    bool is_boundary;
    if (gid == 0) {
        is_boundary = true;
    } else {
        is_boundary = (d_in[gid] != d_in[gid - 1]);
    }

    if (is_boundary) {
        const uint out_idx = d_positions[gid];
        d_unique_out[out_idx] = d_in[gid];

        // Find run length: scan forward to find next boundary or end
        uint end = gid + 1;
        while (end < num_items && d_in[end] == d_in[gid]) {
            end++;
        }
        d_counts_out[out_idx] = end - gid;
    }

    // Last thread writes total number of runs
    if (gid == num_items - 1) {
        d_num_runs_out[0] = d_positions[gid] + 1;
    }
}
