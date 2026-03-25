#include "field.metal"

using namespace airbender::field;

// Test kernel: performs field operations and writes results.
// Input: array of u32 values representing field elements
// Output: array of u32 results from various operations
//
// For each pair (input[2*i], input[2*i+1]):
//   output[6*i+0] = add(a, b)
//   output[6*i+1] = sub(a, b)
//   output[6*i+2] = mul(a, b)
//   output[6*i+3] = sqr(a)
//   output[6*i+4] = inv(a) (if a != 0, else 0)
//   output[6*i+5] = neg(a)
kernel void field_test(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& pair_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= pair_count) return;

    const base_field a = {input[2 * tid]};
    const base_field b = {input[2 * tid + 1]};

    output[6 * tid + 0] = base_field::into_canonical_u32(base_field::add(a, b));
    output[6 * tid + 1] = base_field::into_canonical_u32(base_field::sub(a, b));
    output[6 * tid + 2] = base_field::into_canonical_u32(base_field::mul(a, b));
    output[6 * tid + 3] = base_field::into_canonical_u32(base_field::sqr(a));
    output[6 * tid + 4] = (a.limb == 0) ? 0u : base_field::into_canonical_u32(base_field::inv(a));
    output[6 * tid + 5] = base_field::into_canonical_u32(base_field::neg(a));
}

// Test kernel for ext2 field operations
kernel void ext2_field_test(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    // Read 4 u32s as two ext2 elements: a = (input[4i], input[4i+1]), b = (input[4i+2], input[4i+3])
    const base_field a0 = {input[4 * tid + 0]};
    const base_field a1 = {input[4 * tid + 1]};
    const base_field b0 = {input[4 * tid + 2]};
    const base_field b1 = {input[4 * tid + 3]};
    const ext2_field a = {a0, a1};
    const ext2_field b = {b0, b1};

    // add, mul, sqr results (each 2 u32s)
    const ext2_field sum = ext2_field::add(a, b);
    const ext2_field prod = ext2_field::mul(a, b);
    const ext2_field sq = ext2_field::sqr(a);

    // Output: 6 u32s per element
    output[6 * tid + 0] = base_field::into_canonical_u32(sum.coefficients[0]);
    output[6 * tid + 1] = base_field::into_canonical_u32(sum.coefficients[1]);
    output[6 * tid + 2] = base_field::into_canonical_u32(prod.coefficients[0]);
    output[6 * tid + 3] = base_field::into_canonical_u32(prod.coefficients[1]);
    output[6 * tid + 4] = base_field::into_canonical_u32(sq.coefficients[0]);
    output[6 * tid + 5] = base_field::into_canonical_u32(sq.coefficients[1]);
}
