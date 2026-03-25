#include <metal_stdlib>
using namespace metal;

/// Trivial identity copy kernel for testing the Metal runtime infrastructure.
/// Copies `count` uint32 values from `input` to `output`.
kernel void identity_copy(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        output[tid] = input[tid];
    }
}
