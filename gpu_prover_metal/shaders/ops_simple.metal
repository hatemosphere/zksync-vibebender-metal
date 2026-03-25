#include "field.metal"
#include "memory.metal"

using namespace airbender::field;
using namespace airbender::memory;

// --- transpose_cols_to_leaves_tiled: 2D tiled transpose for coalesced memory access ---
// Threadgroup cooperatively loads TILE_COLS columns at a time through shared memory.
// Reading: all threads read from consecutive positions within same column (coalesced).
// Writing: each thread writes its leaf's data sequentially.
// TILE_COLS=16 matches blake2s BLOCK_SIZE so each tile produces one hash block per leaf.

constant constexpr uint TILE_COLS = 16;
constant constexpr uint TRANSPOSE_THREADS = 256;

kernel void ab_transpose_cols_to_leaves_tiled_kernel(
    const device uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    constant uint& num_leaves [[buffer(2)]],
    constant uint& cols_count [[buffer(3)]],
    constant uint& rows_per_leaf [[buffer(4)]],
    constant uint& col_stride [[buffer(5)]],     // = num_leaves * rows_per_leaf (elements between columns)
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    // Each threadgroup processes TRANSPOSE_THREADS leaves
    const uint leaf_base = tgid * TRANSPOSE_THREADS;
    const uint leaf = leaf_base + tid;
    if (leaf >= num_leaves) return;

    const uint elems_per_leaf = cols_count * rows_per_leaf;

    // Shared memory: TILE_COLS columns × TRANSPOSE_THREADS leaves × rows_per_leaf
    // For TILE_COLS=16, TRANSPOSE_THREADS=256, rows_per_leaf=2: 16*256*2 = 8192 elements = 32KB
    threadgroup uint tile[TILE_COLS * TRANSPOSE_THREADS * 2]; // max rows_per_leaf=2

    // Process columns in tiles of TILE_COLS
    for (uint col_base = 0; col_base < cols_count; col_base += TILE_COLS) {
        const uint tile_cols = min(TILE_COLS, cols_count - col_base);

        // Cooperative load: each thread loads its rows for each column in the tile
        // Within each column, threads 0..255 access positions [leaf_base*rpl .. (leaf_base+255)*rpl+rpl-1]
        // These are CONSECUTIVE in source memory → coalesced reads
        #pragma unroll
        for (uint tc = 0; tc < TILE_COLS; tc++) {
            if (tc < tile_cols) {
                const uint src_col_base = (col_base + tc) * col_stride + leaf * rows_per_leaf;
                const uint tile_offset = tc * TRANSPOSE_THREADS * rows_per_leaf + tid * rows_per_leaf;
                #pragma unroll
                for (uint r = 0; r < 2; r++) { // rows_per_leaf is 1 or 2
                    if (r < rows_per_leaf) {
                        uint val = src[src_col_base + r];
                        // Canonicalize: ORDER → 0
                        tile[tile_offset + r] = (val == 0x7FFFFFFFu) ? 0u : val;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread writes its leaf's tile data to the destination
        const uint dst_offset = leaf * elems_per_leaf + col_base * rows_per_leaf;
        #pragma unroll
        for (uint tc = 0; tc < TILE_COLS; tc++) {
            if (tc < tile_cols) {
                const uint tile_offset = tc * TRANSPOSE_THREADS * rows_per_leaf + tid * rows_per_leaf;
                #pragma unroll
                for (uint r = 0; r < 2; r++) {
                    if (r < rows_per_leaf) {
                        dst[dst_offset + tc * rows_per_leaf + r] = tile[tile_offset + r];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// --- negate_and_scatter: negate col_sums[i] and write to dst[i * stride + stride - 1] ---
// Used for stage 2 "sum to zero" correction: sets last row = -sum(other rows)

kernel void ab_negate_and_scatter_to_last_row_kernel(
    const device base_field* col_sums [[buffer(0)]],
    device base_field* dst [[buffer(1)]],
    constant uint& num_cols [[buffer(2)]],
    constant uint& stride [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_cols) return;
    dst[gid * stride + stride - 1] = base_field::neg(col_sums[gid]);
}

// --- memset_zero: zero-fill a buffer on GPU (replaces CPU write_bytes) ---

kernel void ab_memset_zero_u32_kernel(
    device uint* buffer [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count)
        buffer[gid] = 0;
}

// --- transpose_bf4_to_e4: convert 4 column-major BF cols to row-major E4 ---
// src layout: col0[0..N], col1[0..N], col2[0..N], col3[0..N]
// dst layout: {col0[0],col1[0],col2[0],col3[0]}, {col0[1],col1[1],col2[1],col3[1]}, ...

kernel void ab_transpose_bf4_to_e4_kernel(
    const device base_field* src [[buffer(0)]],
    device base_field* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid * 4 + 0] = src[0 * n + gid];
    dst[gid * 4 + 1] = src[1 * n + gid];
    dst[gid * 4 + 2] = src[2 * n + gid];
    dst[gid * 4 + 3] = src[3 * n + gid];
}

// --- transpose_e4_to_bf4: convert row-major E4 to 4 column-major BF cols (inverse of above) ---

kernel void ab_transpose_e4_to_bf4_kernel(
    const device base_field* src [[buffer(0)]],
    device base_field* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& base_col [[buffer(3)]],
    constant uint& stride [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[(base_col + 0) * stride + gid] = src[gid * 4 + 0];
    dst[(base_col + 1) * stride + gid] = src[gid * 4 + 1];
    dst[(base_col + 2) * stride + gid] = src[gid * 4 + 2];
    dst[(base_col + 3) * stride + gid] = src[gid * 4 + 3];
}

// --- transpose_bf4_to_e4_strided: same as bf4_to_e4 but reads from strided column-major layout ---

kernel void ab_transpose_bf4_to_e4_strided_kernel(
    const device base_field* src [[buffer(0)]],
    device base_field* dst [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& base_col [[buffer(3)]],
    constant uint& stride [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid * 4 + 0] = src[(base_col + 0) * stride + gid];
    dst[gid * 4 + 1] = src[(base_col + 1) * stride + gid];
    dst[gid * 4 + 2] = src[(base_col + 2) * stride + gid];
    dst[gid * 4 + 3] = src[(base_col + 3) * stride + gid];
}

// --- rotate_right_e4: dst[i] = src[(i-1) mod count] (used for shifted lagrange coeffs) ---

kernel void ab_rotate_right_e4_kernel(
    const device ext4_field* src [[buffer(0)]],
    device ext4_field* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    uint src_idx = (gid == 0) ? (count - 1) : (gid - 1);
    dst[gid] = src[src_idx];
}

// --- memcpy_u32: GPU-side buffer copy (replaces CPU memcpy on unified memory) ---

kernel void ab_memcpy_u32_kernel(
    const device uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count)
        dst[gid] = src[gid];
}

// --- set_by_val kernels: fill a matrix column with a constant value ---

kernel void ab_set_by_val_bf_kernel(
    constant base_field& value [[buffer(0)]],
    device base_field* result [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint row = tid.x;
    if (row >= rows) return;
    const uint col = tid.y;
    result[row + col * stride] = value;
}

kernel void ab_set_by_val_e2_kernel(
    constant ext2_field& value [[buffer(0)]],
    device ext2_field* result [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint row = tid.x;
    if (row >= rows) return;
    const uint col = tid.y;
    result[row + col * stride] = value;
}

kernel void ab_set_by_val_e4_kernel(
    constant ext4_field& value [[buffer(0)]],
    device ext4_field* result [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint row = tid.x;
    if (row >= rows) return;
    const uint col = tid.y;
    result[row + col * stride] = value;
}

kernel void ab_set_by_val_u32_kernel(
    constant uint& value [[buffer(0)]],
    device uint* result [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint row = tid.x;
    if (row >= rows) return;
    const uint col = tid.y;
    result[row + col * stride] = value;
}

kernel void ab_set_by_val_u64_kernel(
    constant ulong& value [[buffer(0)]],
    device ulong* result [[buffer(1)]],
    constant uint& stride [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint row = tid.x;
    if (row >= rows) return;
    const uint col = tid.y;
    result[row + col * stride] = value;
}

// --- set_by_ref kernels: copy one matrix to another ---

#define SET_BY_REF_KERNEL(field_t) \
kernel void ab_set_by_ref_##field_t##_kernel( \
    const device field_t* src [[buffer(0)]], \
    constant uint& src_stride [[buffer(1)]], \
    device field_t* dst [[buffer(2)]], \
    constant uint& dst_stride [[buffer(3)]], \
    constant uint& rows [[buffer(4)]], \
    uint2 tid [[thread_position_in_grid]] \
) { \
    const uint row = tid.x; \
    if (row >= rows) return; \
    const uint col = tid.y; \
    dst[row + col * dst_stride] = src[row + col * src_stride]; \
}

SET_BY_REF_KERNEL(base_field)
SET_BY_REF_KERNEL(ext2_field)
SET_BY_REF_KERNEL(ext4_field)

kernel void ab_set_by_ref_u32_kernel(
    const device uint* src [[buffer(0)]],
    constant uint& src_stride [[buffer(1)]],
    device uint* dst [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint row = tid.x;
    if (row >= rows) return;
    const uint col = tid.y;
    dst[row + col * dst_stride] = src[row + col * src_stride];
}

kernel void ab_set_by_ref_u64_kernel(
    const device ulong* src [[buffer(0)]],
    constant uint& src_stride [[buffer(1)]],
    device ulong* dst [[buffer(2)]],
    constant uint& dst_stride [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint row = tid.x;
    if (row >= rows) return;
    const uint col = tid.y;
    dst[row + col * dst_stride] = src[row + col * src_stride];
}

// --- Unary ops: dbl, inv, neg, sqr ---

#define UNARY_KERNEL(op, field_t) \
kernel void ab_##op##_##field_t##_kernel( \
    const device field_t* src [[buffer(0)]], \
    constant uint& src_stride [[buffer(1)]], \
    device field_t* dst [[buffer(2)]], \
    constant uint& dst_stride [[buffer(3)]], \
    constant uint& rows [[buffer(4)]], \
    uint2 tid [[thread_position_in_grid]] \
) { \
    const uint row = tid.x; \
    if (row >= rows) return; \
    const uint col = tid.y; \
    const field_t val = src[row + col * src_stride]; \
    dst[row + col * dst_stride] = field_t::op(val); \
}

UNARY_KERNEL(dbl, base_field)
UNARY_KERNEL(dbl, ext2_field)
UNARY_KERNEL(dbl, ext4_field)
UNARY_KERNEL(inv, base_field)
UNARY_KERNEL(inv, ext2_field)
UNARY_KERNEL(inv, ext4_field)
UNARY_KERNEL(neg, base_field)
UNARY_KERNEL(neg, ext2_field)
UNARY_KERNEL(neg, ext4_field)
UNARY_KERNEL(sqr, base_field)
UNARY_KERNEL(sqr, ext2_field)
UNARY_KERNEL(sqr, ext4_field)

// --- Parametrized ops: pow, shl, shr ---

#define PARAMETRIZED_KERNEL(op, field_t) \
kernel void ab_##op##_##field_t##_kernel( \
    const device field_t* src [[buffer(0)]], \
    constant uint& src_stride [[buffer(1)]], \
    device field_t* dst [[buffer(2)]], \
    constant uint& dst_stride [[buffer(3)]], \
    constant uint& rows [[buffer(4)]], \
    constant uint& param [[buffer(5)]], \
    uint2 tid [[thread_position_in_grid]] \
) { \
    const uint row = tid.x; \
    if (row >= rows) return; \
    const uint col = tid.y; \
    const field_t val = src[row + col * src_stride]; \
    dst[row + col * dst_stride] = field_t::op(val, param); \
}

PARAMETRIZED_KERNEL(pow, base_field)
PARAMETRIZED_KERNEL(pow, ext2_field)
PARAMETRIZED_KERNEL(pow, ext4_field)
PARAMETRIZED_KERNEL(shl, base_field)
PARAMETRIZED_KERNEL(shl, ext2_field)
PARAMETRIZED_KERNEL(shl, ext4_field)
PARAMETRIZED_KERNEL(shr, base_field)
PARAMETRIZED_KERNEL(shr, ext2_field)
PARAMETRIZED_KERNEL(shr, ext4_field)

// --- Binary ops: add, mul, sub ---

#define BINARY_KERNEL(op, field_t0, field_t1, result_t) \
kernel void ab_##op##_##field_t0##_##field_t1##_kernel( \
    const device field_t0* src0 [[buffer(0)]], \
    constant uint& src0_stride [[buffer(1)]], \
    const device field_t1* src1 [[buffer(2)]], \
    constant uint& src1_stride [[buffer(3)]], \
    device result_t* dst [[buffer(4)]], \
    constant uint& dst_stride [[buffer(5)]], \
    constant uint& rows [[buffer(6)]], \
    uint2 tid [[thread_position_in_grid]] \
) { \
    const uint row = tid.x; \
    if (row >= rows) return; \
    const uint col = tid.y; \
    const field_t0 a = src0[row + col * src0_stride]; \
    const field_t1 b = src1[row + col * src1_stride]; \
    dst[row + col * dst_stride] = result_t::op(a, b); \
}

BINARY_KERNEL(add, base_field, base_field, base_field)
BINARY_KERNEL(add, base_field, ext2_field, ext2_field)
BINARY_KERNEL(add, ext2_field, base_field, ext2_field)
BINARY_KERNEL(add, ext2_field, ext2_field, ext2_field)
BINARY_KERNEL(add, base_field, ext4_field, ext4_field)
BINARY_KERNEL(add, ext2_field, ext4_field, ext4_field)
BINARY_KERNEL(add, ext4_field, base_field, ext4_field)
BINARY_KERNEL(add, ext4_field, ext2_field, ext4_field)
BINARY_KERNEL(add, ext4_field, ext4_field, ext4_field)

BINARY_KERNEL(mul, base_field, base_field, base_field)
BINARY_KERNEL(mul, base_field, ext2_field, ext2_field)
BINARY_KERNEL(mul, ext2_field, base_field, ext2_field)
BINARY_KERNEL(mul, ext2_field, ext2_field, ext2_field)
BINARY_KERNEL(mul, base_field, ext4_field, ext4_field)
BINARY_KERNEL(mul, ext2_field, ext4_field, ext4_field)
BINARY_KERNEL(mul, ext4_field, base_field, ext4_field)
BINARY_KERNEL(mul, ext4_field, ext2_field, ext4_field)
BINARY_KERNEL(mul, ext4_field, ext4_field, ext4_field)

BINARY_KERNEL(sub, base_field, base_field, base_field)
BINARY_KERNEL(sub, base_field, ext2_field, ext2_field)
BINARY_KERNEL(sub, ext2_field, base_field, ext2_field)
BINARY_KERNEL(sub, ext2_field, ext2_field, ext2_field)
BINARY_KERNEL(sub, base_field, ext4_field, ext4_field)
BINARY_KERNEL(sub, ext2_field, ext4_field, ext4_field)
BINARY_KERNEL(sub, ext4_field, base_field, ext4_field)
BINARY_KERNEL(sub, ext4_field, ext2_field, ext4_field)
BINARY_KERNEL(sub, ext4_field, ext4_field, ext4_field)

// --- Ternary ops: mul_add (result = mul(x,y) + z), mul_sub (result = mul(x,y) - z) ---

// Free-standing mul_mixed overloads: dispatch to correct static mul for mixed type combinations.
DEVICE_FORCEINLINE base_field mul_mixed(const base_field a, const base_field b) { return base_field::mul(a, b); }
DEVICE_FORCEINLINE ext2_field mul_mixed(const base_field a, const ext2_field b) { return ext2_field::mul(a, b); }
DEVICE_FORCEINLINE ext2_field mul_mixed(const ext2_field a, const base_field b) { return ext2_field::mul(a, b); }
DEVICE_FORCEINLINE ext2_field mul_mixed(const ext2_field a, const ext2_field b) { return ext2_field::mul(a, b); }
DEVICE_FORCEINLINE ext4_field mul_mixed(const base_field a, const ext4_field b) { return ext4_field::mul(a, b); }
DEVICE_FORCEINLINE ext4_field mul_mixed(const ext4_field a, const base_field b) { return ext4_field::mul(a, b); }
DEVICE_FORCEINLINE ext4_field mul_mixed(const ext2_field a, const ext4_field b) { return ext4_field::mul(a, b); }
DEVICE_FORCEINLINE ext4_field mul_mixed(const ext4_field a, const ext2_field b) { return ext4_field::mul(a, b); }
DEVICE_FORCEINLINE ext4_field mul_mixed(const ext4_field a, const ext4_field b) { return ext4_field::mul(a, b); }

#define TERNARY_OP_KERNEL(op_name, addorsub, t0, t1, t2, rt) \
kernel void ab_##op_name##_##t0##_##t1##_##t2##_kernel( \
    const device t0* src0 [[buffer(0)]], \
    constant uint& src0_stride [[buffer(1)]], \
    const device t1* src1 [[buffer(2)]], \
    constant uint& src1_stride [[buffer(3)]], \
    const device t2* src2 [[buffer(4)]], \
    constant uint& src2_stride [[buffer(5)]], \
    device rt* dst [[buffer(6)]], \
    constant uint& dst_stride [[buffer(7)]], \
    constant uint& rows [[buffer(8)]], \
    uint2 tid [[thread_position_in_grid]] \
) { \
    const uint row = tid.x; \
    if (row >= rows) return; \
    const uint col = tid.y; \
    const t0 a = src0[row + col * src0_stride]; \
    const t1 b = src1[row + col * src1_stride]; \
    const t2 c = src2[row + col * src2_stride]; \
    dst[row + col * dst_stride] = rt::addorsub(mul_mixed(a, b), c); \
}

TERNARY_OP_KERNEL(mul_add, add, base_field, base_field, base_field, base_field)
TERNARY_OP_KERNEL(mul_add, add, base_field, base_field, ext2_field, ext2_field)
TERNARY_OP_KERNEL(mul_add, add, base_field, base_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, base_field, ext2_field, base_field, ext2_field)
TERNARY_OP_KERNEL(mul_add, add, base_field, ext2_field, ext2_field, ext2_field)
TERNARY_OP_KERNEL(mul_add, add, base_field, ext2_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, base_field, ext4_field, base_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, base_field, ext4_field, ext2_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, base_field, ext4_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext2_field, base_field, base_field, ext2_field)
TERNARY_OP_KERNEL(mul_add, add, ext2_field, base_field, ext2_field, ext2_field)
TERNARY_OP_KERNEL(mul_add, add, ext2_field, base_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext2_field, ext2_field, base_field, ext2_field)
TERNARY_OP_KERNEL(mul_add, add, ext2_field, ext2_field, ext2_field, ext2_field)
TERNARY_OP_KERNEL(mul_add, add, ext2_field, ext2_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext2_field, ext4_field, base_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext2_field, ext4_field, ext2_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext2_field, ext4_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext4_field, base_field, base_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext4_field, base_field, ext2_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext4_field, base_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext4_field, ext2_field, base_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext4_field, ext2_field, ext2_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext4_field, ext2_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext4_field, ext4_field, base_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext4_field, ext4_field, ext2_field, ext4_field)
TERNARY_OP_KERNEL(mul_add, add, ext4_field, ext4_field, ext4_field, ext4_field)

TERNARY_OP_KERNEL(mul_sub, sub, base_field, base_field, base_field, base_field)
TERNARY_OP_KERNEL(mul_sub, sub, base_field, base_field, ext2_field, ext2_field)
TERNARY_OP_KERNEL(mul_sub, sub, base_field, base_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, base_field, ext2_field, base_field, ext2_field)
TERNARY_OP_KERNEL(mul_sub, sub, base_field, ext2_field, ext2_field, ext2_field)
TERNARY_OP_KERNEL(mul_sub, sub, base_field, ext2_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, base_field, ext4_field, base_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, base_field, ext4_field, ext2_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, base_field, ext4_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext2_field, base_field, base_field, ext2_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext2_field, base_field, ext2_field, ext2_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext2_field, base_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext2_field, ext2_field, base_field, ext2_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext2_field, ext2_field, ext2_field, ext2_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext2_field, ext2_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext2_field, ext4_field, base_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext2_field, ext4_field, ext2_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext2_field, ext4_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext4_field, base_field, base_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext4_field, base_field, ext2_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext4_field, base_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext4_field, ext2_field, base_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext4_field, ext2_field, ext2_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext4_field, ext2_field, ext4_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext4_field, ext4_field, base_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext4_field, ext4_field, ext2_field, ext4_field)
TERNARY_OP_KERNEL(mul_sub, sub, ext4_field, ext4_field, ext4_field, ext4_field)
