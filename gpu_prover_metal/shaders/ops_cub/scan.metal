#include "../field.metal"

using namespace airbender::field;

// 3-pass parallel prefix scan (inclusive/exclusive, add/mul) for bf, e4, u32.

constant constexpr uint SCAN_THREADS_PER_GROUP = 256;

// Threadgroup inclusive scan using shared memory passed as parameter.
template <typename T, typename OpFn>
DEVICE_FORCEINLINE T threadgroup_inclusive_scan(
    threadgroup T* shared,
    T val,
    uint lid,
    uint threads_in_group,
    OpFn op
) {
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < threads_in_group; stride <<= 1) {
        T other = (lid >= stride) ? shared[lid - stride] : op.identity();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[lid] = op(shared[lid], other);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return shared[lid];
}

// Pass 3 has no threadgroup memory, so template is fine
template <typename T, typename OpFn>
DEVICE_FORCEINLINE void scan_pass3_impl(
    device T* d_out,
    const device T* d_block_prefixes,
    const uint num_items,
    const uint tgid,
    const uint lid,
    OpFn op
) {
    const uint gid = tgid * SCAN_THREADS_PER_GROUP + lid;
    if (gid < num_items && tgid > 0) {
        T prefix = d_block_prefixes[tgid];
        d_out[gid] = op(prefix, d_out[gid]);
    }
}

// --- Macro-based kernel instantiation with threadgroup memory in kernel scope ---

#define SCAN_PASS1_KERNEL(name, field_t, op_body_combine, op_body_identity) \
kernel void ab_scan_pass1_##name##_kernel( \
    const device field_t* d_in [[buffer(0)]], \
    device field_t* d_out [[buffer(1)]], \
    device field_t* d_block_totals [[buffer(2)]], \
    constant uint& num_items [[buffer(3)]], \
    constant uint& inclusive [[buffer(4)]], \
    uint tgid [[threadgroup_position_in_grid]], \
    uint lid [[thread_position_in_threadgroup]] \
) { \
    threadgroup field_t shared[SCAN_THREADS_PER_GROUP]; \
    struct Op { \
        DEVICE_FORCEINLINE field_t operator()(field_t a, field_t b) const { return op_body_combine; } \
        DEVICE_FORCEINLINE field_t identity() const { return op_body_identity; } \
    }; \
    Op op; \
    const uint gid = tgid * SCAN_THREADS_PER_GROUP + lid; \
    field_t val = (gid < num_items) ? d_in[gid] : op.identity(); \
    field_t scanned = threadgroup_inclusive_scan(shared, val, lid, SCAN_THREADS_PER_GROUP, op); \
    if (lid == SCAN_THREADS_PER_GROUP - 1) { \
        d_block_totals[tgid] = scanned; \
    } \
    if (gid < num_items) { \
        if (inclusive != 0) { \
            d_out[gid] = scanned; \
        } else { \
            field_t exclusive_val = (lid > 0) ? shared[lid - 1] : op.identity(); \
            d_out[gid] = exclusive_val; \
        } \
    } \
}

#define SCAN_PASS2_KERNEL(name, field_t, op_body_combine, op_body_identity) \
kernel void ab_scan_pass2_##name##_kernel( \
    device field_t* d_block_totals [[buffer(0)]], \
    constant uint& num_blocks [[buffer(1)]], \
    uint lid [[thread_position_in_threadgroup]] \
) { \
    threadgroup field_t shared[SCAN_THREADS_PER_GROUP]; \
    struct Op { \
        DEVICE_FORCEINLINE field_t operator()(field_t a, field_t b) const { return op_body_combine; } \
        DEVICE_FORCEINLINE field_t identity() const { return op_body_identity; } \
    }; \
    Op op; \
    field_t val = (lid < num_blocks) ? d_block_totals[lid] : op.identity(); \
    threadgroup_inclusive_scan(shared, val, lid, SCAN_THREADS_PER_GROUP, op); \
    if (lid < num_blocks) { \
        field_t exclusive_val = (lid > 0) ? shared[lid - 1] : op.identity(); \
        d_block_totals[lid] = exclusive_val; \
    } \
}

#define SCAN_PASS3_KERNEL(name, field_t, op_body_combine, op_body_identity) \
kernel void ab_scan_pass3_##name##_kernel( \
    device field_t* d_out [[buffer(0)]], \
    const device field_t* d_block_prefixes [[buffer(1)]], \
    constant uint& num_items [[buffer(2)]], \
    uint tgid [[threadgroup_position_in_grid]], \
    uint lid [[thread_position_in_threadgroup]] \
) { \
    struct Op { \
        DEVICE_FORCEINLINE field_t operator()(field_t a, field_t b) const { return op_body_combine; } \
        DEVICE_FORCEINLINE field_t identity() const { return op_body_identity; } \
    }; \
    Op op; \
    scan_pass3_impl(d_out, d_block_prefixes, num_items, tgid, lid, op); \
}

// add_bf
SCAN_PASS1_KERNEL(add_bf, base_field, base_field::add(a, b), base_field::zero())
SCAN_PASS2_KERNEL(add_bf, base_field, base_field::add(a, b), base_field::zero())
SCAN_PASS3_KERNEL(add_bf, base_field, base_field::add(a, b), base_field::zero())

// mul_bf
SCAN_PASS1_KERNEL(mul_bf, base_field, base_field::mul(a, b), base_field::one())
SCAN_PASS2_KERNEL(mul_bf, base_field, base_field::mul(a, b), base_field::one())
SCAN_PASS3_KERNEL(mul_bf, base_field, base_field::mul(a, b), base_field::one())

// add_e4
SCAN_PASS1_KERNEL(add_e4, ext4_field, ext4_field::add(a, b), ext4_field::zero())
SCAN_PASS2_KERNEL(add_e4, ext4_field, ext4_field::add(a, b), ext4_field::zero())
SCAN_PASS3_KERNEL(add_e4, ext4_field, ext4_field::add(a, b), ext4_field::zero())

// mul_e4
SCAN_PASS1_KERNEL(mul_e4, ext4_field, ext4_field::mul(a, b), ext4_field::one())
SCAN_PASS2_KERNEL(mul_e4, ext4_field, ext4_field::mul(a, b), ext4_field::one())
SCAN_PASS3_KERNEL(mul_e4, ext4_field, ext4_field::mul(a, b), ext4_field::one())

// add_u32
SCAN_PASS1_KERNEL(add_u32, uint, a + b, 0u)
SCAN_PASS2_KERNEL(add_u32, uint, a + b, 0u)
SCAN_PASS3_KERNEL(add_u32, uint, a + b, 0u)
