#include "../arg_utils.metal"
#include <metal_atomic>

using namespace airbender::arg_utils;
using namespace airbender::field;
using namespace airbender::memory;

using bf = base_field;

kernel void ab_generate_multiplicities_kernel(
    const device uint *unique_indexes [[buffer(0)]],
    const device uint *counts [[buffer(1)]],
    const device uint *num_runs [[buffer(2)]],
    device bf *multiplicities_ptr [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count)
    return;
  if (gid >= num_runs[0] - 1)
    return;
  const uint stride_minus_1 = stride - 1;
  const uint index = unique_indexes[gid];
  const uint row = index % stride_minus_1;
  const uint col = index / stride_minus_1;
  const bf value = bf::from_u32(counts[gid]);
  // matrix_setter equivalent: ptr[row + col * stride]
  multiplicities_ptr[row + col * stride] = value;
}

kernel void ab_range_check_multiplicities_kernel(
    const device bf *setup_ptr [[buffer(0)]],
    const device bf *witness_ptr [[buffer(1)]],
    const device bf *memory_ptr [[buffer(2)]],
    const device RangeCheckArgsLayout &rc16_layout [[buffer(3)]],
    const device FlattenedLookupExpressionsLayout &expressions [[buffer(4)]],
    const device FlattenedLookupExpressionsForShuffleRamLayout &expressions_for_shuffle_ram [[buffer(5)]],
    device atomic_uint *rc16_histogram [[buffer(6)]],
    device atomic_uint *ts_histogram [[buffer(7)]],
    constant uint &setup_stride [[buffer(8)]],
    constant uint &witness_stride [[buffer(9)]],
    constant uint &memory_stride [[buffer(10)]],
    constant bf &memory_timestamp_high [[buffer(11)]],
    constant uint &process_shuffle_ram_init_flag [[buffer(12)]],
    constant uint &lazy_init_address_start [[buffer(13)]],
    constant uint &trace_len [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= trace_len - 1)
        return;

    matrix_getter<bf> setup_cols = {setup_ptr, setup_stride};
    matrix_getter<bf> witness_cols = {witness_ptr, witness_stride};
    matrix_getter<bf> memory_cols = {memory_ptr, memory_stride};

    setup_cols.add_row(gid);
    witness_cols.add_row(gid);
    memory_cols.add_row(gid);

    // Trivial RC16 columns: direct witness column reads
    for (uint i = 0; i < 2 * rc16_layout.num_dst_cols; i++) {
        const uint src = i + rc16_layout.src_cols_start;
        const uint val = bf::into_canonical_u32(witness_cols.get_at_col(src));
        atomic_fetch_add_explicit(&rc16_histogram[val], 1u, memory_order_relaxed);
    }

    // Non-trivial lookup expressions (RC16 then timestamp)
    {
        uint expression_idx = 0;
        uint flat_term_idx = 0;

        // RC16 expression pairs
        for (uint i = 0; i < expressions.num_range_check_16_expression_pairs; i++) {
            bf a_and_b[2];
            eval_a_and_b<true>(a_and_b, expressions, expression_idx, flat_term_idx,
                               witness_cols, memory_cols, expressions.range_check_16_constant_terms_are_zero);
            atomic_fetch_add_explicit(&rc16_histogram[bf::into_canonical_u32(a_and_b[0])], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&rc16_histogram[bf::into_canonical_u32(a_and_b[1])], 1u, memory_order_relaxed);
        }

        // Timestamp expression pairs
        for (uint i = 0; i < expressions.num_timestamp_expression_pairs; i++) {
            bf a_and_b[2];
            eval_a_and_b<true>(a_and_b, expressions, expression_idx, flat_term_idx,
                               witness_cols, memory_cols, expressions.timestamp_constant_terms_are_zero);
            atomic_fetch_add_explicit(&ts_histogram[bf::into_canonical_u32(a_and_b[0])], 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&ts_histogram[bf::into_canonical_u32(a_and_b[1])], 1u, memory_order_relaxed);
        }
    }

    // Shuffle RAM expressions (use setup+witness+memory, subtract timestamp_high from b)
    for (uint i = 0, expression_idx = 0, flat_term_idx = 0; i < expressions_for_shuffle_ram.num_expression_pairs; i++) {
        bf a_and_b[2];
        eval_a_and_b<true>(a_and_b, expressions_for_shuffle_ram, expression_idx, flat_term_idx,
                           setup_cols, witness_cols, memory_cols);
        a_and_b[1] = bf::sub(a_and_b[1], memory_timestamp_high);
        atomic_fetch_add_explicit(&ts_histogram[bf::into_canonical_u32(a_and_b[0])], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&ts_histogram[bf::into_canonical_u32(a_and_b[1])], 1u, memory_order_relaxed);
    }

    // Lazy init address columns (two 16-bit halves into RC16 histogram)
    if (process_shuffle_ram_init_flag) {
        const uint val0 = bf::into_canonical_u32(memory_cols.get_at_col(lazy_init_address_start));
        const uint val1 = bf::into_canonical_u32(memory_cols.get_at_col(lazy_init_address_start + 1));
        atomic_fetch_add_explicit(&rc16_histogram[val0], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&rc16_histogram[val1], 1u, memory_order_relaxed);
    }
}
