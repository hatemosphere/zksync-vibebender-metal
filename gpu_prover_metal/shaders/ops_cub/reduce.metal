#include "../field.metal"
#include <metal_simdgroup>

using namespace airbender::field;

// Parallel tree reduction for bf, e2, e4.
// Uses SIMD group shuffles for the bottom 32-element reduction (no barriers),
// then threadgroup memory for the cross-SIMD reduction (3 barriers for 256 threads).

constant constexpr uint REDUCE_THREADS_PER_GROUP = 256;
constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint NUM_SIMD_GROUPS = REDUCE_THREADS_PER_GROUP / SIMD_SIZE;

// --- SIMD shuffle reduce helpers ---
// Reduce 32 values within a SIMD group using shuffle_xor (5 steps, no barriers).

#define DEFINE_SIMD_REDUCE(name, field_t, op_combine) \
DEVICE_FORCEINLINE field_t simd_reduce_##name(field_t val)

// base_field: 1 component
DEFINE_SIMD_REDUCE(add_bf, base_field, base_field::add) {
    #pragma unroll
    for (ushort off = 1; off < SIMD_SIZE; off <<= 1) {
        base_field other = {simd_shuffle_xor(val.limb, off)};
        val = base_field::add(val, other);
    }
    return val;
}
DEFINE_SIMD_REDUCE(mul_bf, base_field, base_field::mul) {
    #pragma unroll
    for (ushort off = 1; off < SIMD_SIZE; off <<= 1) {
        base_field other = {simd_shuffle_xor(val.limb, off)};
        val = base_field::mul(val, other);
    }
    return val;
}

// ext2_field: 2 components
DEFINE_SIMD_REDUCE(add_e2, ext2_field, ext2_field::add) {
    #pragma unroll
    for (ushort off = 1; off < SIMD_SIZE; off <<= 1) {
        ext2_field other;
        other.coefficients[0].limb = simd_shuffle_xor(val.coefficients[0].limb, off);
        other.coefficients[1].limb = simd_shuffle_xor(val.coefficients[1].limb, off);
        val = ext2_field::add(val, other);
    }
    return val;
}
DEFINE_SIMD_REDUCE(mul_e2, ext2_field, ext2_field::mul) {
    #pragma unroll
    for (ushort off = 1; off < SIMD_SIZE; off <<= 1) {
        ext2_field other;
        other.coefficients[0].limb = simd_shuffle_xor(val.coefficients[0].limb, off);
        other.coefficients[1].limb = simd_shuffle_xor(val.coefficients[1].limb, off);
        val = ext2_field::mul(val, other);
    }
    return val;
}

// ext4_field: 4 components
DEFINE_SIMD_REDUCE(add_e4, ext4_field, ext4_field::add) {
    #pragma unroll
    for (ushort off = 1; off < SIMD_SIZE; off <<= 1) {
        ext4_field other;
        other.coefficients[0].coefficients[0].limb = simd_shuffle_xor(val.coefficients[0].coefficients[0].limb, off);
        other.coefficients[0].coefficients[1].limb = simd_shuffle_xor(val.coefficients[0].coefficients[1].limb, off);
        other.coefficients[1].coefficients[0].limb = simd_shuffle_xor(val.coefficients[1].coefficients[0].limb, off);
        other.coefficients[1].coefficients[1].limb = simd_shuffle_xor(val.coefficients[1].coefficients[1].limb, off);
        val = ext4_field::add(val, other);
    }
    return val;
}
DEFINE_SIMD_REDUCE(mul_e4, ext4_field, ext4_field::mul) {
    #pragma unroll
    for (ushort off = 1; off < SIMD_SIZE; off <<= 1) {
        ext4_field other;
        other.coefficients[0].coefficients[0].limb = simd_shuffle_xor(val.coefficients[0].coefficients[0].limb, off);
        other.coefficients[0].coefficients[1].limb = simd_shuffle_xor(val.coefficients[0].coefficients[1].limb, off);
        other.coefficients[1].coefficients[0].limb = simd_shuffle_xor(val.coefficients[1].coefficients[0].limb, off);
        other.coefficients[1].coefficients[1].limb = simd_shuffle_xor(val.coefficients[1].coefficients[1].limb, off);
        val = ext4_field::mul(val, other);
    }
    return val;
}

// --- Kernel macros using 2-level SIMD + threadgroup reduction ---

#define REDUCE_PASS1_KERNEL(name, field_t, op_body_combine, op_body_identity) \
kernel void ab_reduce_pass1_##name##_kernel( \
    const device field_t* d_in [[buffer(0)]], \
    device field_t* d_partials [[buffer(1)]], \
    constant uint& num_items [[buffer(2)]], \
    uint tgid [[threadgroup_position_in_grid]], \
    uint lid [[thread_position_in_threadgroup]] \
) { \
    const uint block_start = tgid * REDUCE_THREADS_PER_GROUP; \
    field_t acc = (block_start + lid < num_items) ? d_in[block_start + lid] : (op_body_identity); \
    acc = simd_reduce_##name(acc); \
    threadgroup field_t shared[NUM_SIMD_GROUPS]; \
    const uint simd_id = lid / SIMD_SIZE; \
    const uint lane_id = lid % SIMD_SIZE; \
    if (lane_id == 0) shared[simd_id] = acc; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lid < NUM_SIMD_GROUPS) { \
        field_t a = shared[lid]; \
        for (uint s = NUM_SIMD_GROUPS / 2; s > 0; s >>= 1) { \
            if (lid < s) { \
                field_t b = shared[lid + s]; \
                a = op_body_combine; \
                shared[lid] = a; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    if (lid == 0) d_partials[tgid] = shared[0]; \
}

#define REDUCE_PASS2_KERNEL(name, field_t, op_body_combine, op_body_identity) \
kernel void ab_reduce_pass2_##name##_kernel( \
    device field_t* d_partials [[buffer(0)]], \
    device field_t* d_out [[buffer(1)]], \
    constant uint& num_partials [[buffer(2)]], \
    uint lid [[thread_position_in_threadgroup]] \
) { \
    field_t val = (lid < num_partials) ? d_partials[lid] : (op_body_identity); \
    val = simd_reduce_##name(val); \
    threadgroup field_t shared[NUM_SIMD_GROUPS]; \
    const uint simd_id = lid / SIMD_SIZE; \
    const uint lane_id = lid % SIMD_SIZE; \
    if (lane_id == 0) shared[simd_id] = val; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lid < NUM_SIMD_GROUPS) { \
        field_t a = shared[lid]; \
        for (uint s = NUM_SIMD_GROUPS / 2; s > 0; s >>= 1) { \
            if (lid < s) { \
                field_t b = shared[lid + s]; \
                a = op_body_combine; \
                shared[lid] = a; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    if (lid == 0) d_out[0] = shared[0]; \
}

#define SEGMENTED_REDUCE_KERNEL(name, field_t, op_body_combine, op_body_identity) \
kernel void ab_segmented_reduce_##name##_kernel( \
    const device field_t* d_in [[buffer(0)]], \
    device field_t* d_out [[buffer(1)]], \
    constant uint& stride [[buffer(2)]], \
    constant uint& num_segments [[buffer(3)]], \
    constant uint& num_items [[buffer(4)]], \
    uint tgid [[threadgroup_position_in_grid]], \
    uint lid [[thread_position_in_threadgroup]] \
) { \
    if (tgid >= num_segments) return; \
    const device field_t* segment = d_in + tgid * stride; \
    field_t acc = (op_body_identity); \
    for (uint i = lid; i < num_items; i += REDUCE_THREADS_PER_GROUP) { \
        field_t b = segment[i]; \
        field_t a = acc; \
        acc = op_body_combine; \
    } \
    acc = simd_reduce_##name(acc); \
    threadgroup field_t shared[NUM_SIMD_GROUPS]; \
    const uint simd_id = lid / SIMD_SIZE; \
    const uint lane_id = lid % SIMD_SIZE; \
    if (lane_id == 0) shared[simd_id] = acc; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (lid < NUM_SIMD_GROUPS) { \
        field_t a = shared[lid]; \
        for (uint s = NUM_SIMD_GROUPS / 2; s > 0; s >>= 1) { \
            if (lid < s) { \
                field_t b = shared[lid + s]; \
                a = op_body_combine; \
                shared[lid] = a; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    if (lid == 0) d_out[tgid] = shared[0]; \
}

REDUCE_PASS1_KERNEL(add_bf, base_field, base_field::add(a, b), base_field::zero())
REDUCE_PASS2_KERNEL(add_bf, base_field, base_field::add(a, b), base_field::zero())
SEGMENTED_REDUCE_KERNEL(add_bf, base_field, base_field::add(a, b), base_field::zero())

REDUCE_PASS1_KERNEL(mul_bf, base_field, base_field::mul(a, b), base_field::one())
REDUCE_PASS2_KERNEL(mul_bf, base_field, base_field::mul(a, b), base_field::one())
SEGMENTED_REDUCE_KERNEL(mul_bf, base_field, base_field::mul(a, b), base_field::one())

REDUCE_PASS1_KERNEL(add_e2, ext2_field, ext2_field::add(a, b), ext2_field::zero())
REDUCE_PASS2_KERNEL(add_e2, ext2_field, ext2_field::add(a, b), ext2_field::zero())
SEGMENTED_REDUCE_KERNEL(add_e2, ext2_field, ext2_field::add(a, b), ext2_field::zero())

REDUCE_PASS1_KERNEL(mul_e2, ext2_field, ext2_field::mul(a, b), ext2_field::one())
REDUCE_PASS2_KERNEL(mul_e2, ext2_field, ext2_field::mul(a, b), ext2_field::one())
SEGMENTED_REDUCE_KERNEL(mul_e2, ext2_field, ext2_field::mul(a, b), ext2_field::one())

REDUCE_PASS1_KERNEL(add_e4, ext4_field, ext4_field::add(a, b), ext4_field::zero())
REDUCE_PASS2_KERNEL(add_e4, ext4_field, ext4_field::add(a, b), ext4_field::zero())
SEGMENTED_REDUCE_KERNEL(add_e4, ext4_field, ext4_field::add(a, b), ext4_field::zero())

REDUCE_PASS1_KERNEL(mul_e4, ext4_field, ext4_field::mul(a, b), ext4_field::one())
REDUCE_PASS2_KERNEL(mul_e4, ext4_field, ext4_field::mul(a, b), ext4_field::one())
SEGMENTED_REDUCE_KERNEL(mul_e4, ext4_field, ext4_field::mul(a, b), ext4_field::one())
