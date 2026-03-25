#include "arg_utils.metal"
#include "context.metal"
#include "vectorized.metal"

// Metal port of gpu_prover/native/stage3.cu
// Stage 3: constraint quotient computation kernels.

// Inline batch_inv_registers from ops_complex.metal (can't include it due to duplicate kernel symbols)
template <typename T, int INV_BATCH, bool batch_is_full>
DEVICE_FORCEINLINE void stage3_batch_inv_registers(const thread T* inputs, thread T* fwd_scan_and_outputs, int runtime_batch_size) {
    T running_prod = T::one();
    #pragma unroll
    for (int i = 0; i < INV_BATCH; i++)
        if (batch_is_full || i < runtime_batch_size) {
            fwd_scan_and_outputs[i] = running_prod;
            running_prod = T::mul(running_prod, inputs[i]);
        }
    T inv = T::inv(running_prod);
    #pragma unroll
    for (int i = INV_BATCH - 1; i >= 0; i--) {
        if (batch_is_full || i < runtime_batch_size) {
            const auto input = inputs[i];
            fwd_scan_and_outputs[i] = T::mul(fwd_scan_and_outputs[i], inv);
            if (i > 0)
                inv = T::mul(inv, input);
        }
    }
}

using namespace airbender::arg_utils;
using namespace airbender::field;
using namespace airbender::memory;
using namespace airbender::vectorized;

using bf = base_field;
using e2 = ext2_field;
using e4 = ext4_field;

// These values must match stage_3_kernels.rs
constant constexpr uint MAX_NON_BOOLEAN_CONSTRAINTS = 192;
constant constexpr uint MAX_TERMS = 1824;
constant constexpr uint MAX_EXPLICIT_COEFFS = 632;
constant constexpr uint MAX_FLAT_COL_IDXS = 3520;
constant constexpr uint8_t COEFF_IS_ONE = 0x00;
constant constexpr uint8_t COEFF_IS_MINUS_ONE = 0x01;

struct FlattenedGenericConstraintsMetadata {
    uint8_t coeffs_info[MAX_TERMS];
    bf explicit_coeffs[MAX_EXPLICIT_COEFFS];
    uint16_t col_idxs[MAX_FLAT_COL_IDXS];
    uchar2 num_linear_and_quadratic_terms_per_constraint[MAX_NON_BOOLEAN_CONSTRAINTS];
    e2 decompression_factor;
    e2 decompression_factor_squared;
    e2 every_row_zerofier;
    e2 omega_inv;
    uint current_flat_col_idx;
    uint current_flat_term_idx;
    uint num_boolean_constraints;
    uint num_non_boolean_quadratic_constraints;
    uint num_non_boolean_constraints;
};

DEVICE_FORCEINLINE void maybe_apply_coeff(const device FlattenedGenericConstraintsMetadata &metadata,
                                           const uint coeff_idx, thread uint &explicit_coeff_idx, thread bf &val) {
    switch (metadata.coeffs_info[coeff_idx]) {
    case COEFF_IS_ONE:
        break;
    case COEFF_IS_MINUS_ONE:
        val = bf::neg(val);
        break;
    default:
        val = bf::mul(val, metadata.explicit_coeffs[explicit_coeff_idx++]);
    }
}

kernel void ab_generic_constraints_kernel(
    const device FlattenedGenericConstraintsMetadata &metadata [[buffer(0)]],
    const device bf *witness_ptr [[buffer(1)]],
    constant uint &witness_stride [[buffer(2)]],
    const device bf *memory_ptr [[buffer(3)]],
    constant uint &memory_stride [[buffer(4)]],
    const device e4 *alphas_ptr [[buffer(5)]],
    device bf *quotient_ptr [[buffer(6)]],
    constant uint &quotient_stride [[buffer(7)]],
    constant uint &log_n [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint n = 1u << log_n;
    if (gid >= n)
        return;

    matrix_getter<bf> witness_cols = {witness_ptr, witness_stride};
    matrix_getter<bf> memory_cols = {memory_ptr, memory_stride};
    vector_getter<e4> alphas = {alphas_ptr};
    vectorized_e4_matrix_setter quotient = {{quotient_ptr, quotient_stride}};

    witness_cols.add_row(gid);
    memory_cols.add_row(gid);
    quotient.add_row(gid);

    e4 acc_linear = e4::zero();
    e4 acc_quadratic = e4::zero();

    // Boolean constraints
    for (uint constraint = 0; constraint < metadata.num_boolean_constraints; constraint++) {
        const bf val_neg = bf::neg(witness_cols.get_at_col(metadata.col_idxs[constraint]));
        const bf val_squared = bf::mul(val_neg, val_neg);
        const e4 alpha_power = alphas.get(constraint);
        acc_quadratic = e4::add(acc_quadratic, e4::mul(alpha_power, val_squared));
        acc_linear = e4::add(acc_linear, e4::mul(alpha_power, val_neg));
    }

    uint alpha_idx = metadata.num_boolean_constraints;
    uint flat_term_idx = 0;
    uint flat_col_idx = metadata.num_boolean_constraints;
    uint explicit_coeff_idx = 0;

    // Non-boolean quadratic constraints
    for (uint constraint = 0; constraint < metadata.num_non_boolean_quadratic_constraints; constraint++) {
        const uchar2 num_lt = metadata.num_linear_and_quadratic_terms_per_constraint[constraint];
        const uint num_quadratic_terms = num_lt.x;
        const uint num_linear_terms = num_lt.y;

        bf quadratic_contribution = bf::zero();
        uint lim = flat_term_idx + num_quadratic_terms;
        for (; flat_term_idx < lim; flat_term_idx++) {
            const bf val0 = get_witness_or_memory(metadata.col_idxs[flat_col_idx++], witness_cols, memory_cols);
            const bf val1 = get_witness_or_memory(metadata.col_idxs[flat_col_idx++], witness_cols, memory_cols);
            bf val = bf::mul(val0, val1);
            maybe_apply_coeff(metadata, flat_term_idx, explicit_coeff_idx, val);
            quadratic_contribution = bf::add(quadratic_contribution, val);
        }
        const e4 alpha_power = alphas.get(alpha_idx++);
        acc_quadratic = e4::add(acc_quadratic, e4::mul(alpha_power, quadratic_contribution));

        if (num_linear_terms > 0) {
            bf linear_contribution = bf::zero();
            lim = flat_term_idx + num_linear_terms;
            for (; flat_term_idx < lim; flat_term_idx++) {
                bf val = get_witness_or_memory(metadata.col_idxs[flat_col_idx++], witness_cols, memory_cols);
                maybe_apply_coeff(metadata, flat_term_idx, explicit_coeff_idx, val);
                linear_contribution = bf::add(linear_contribution, val);
            }
            acc_linear = e4::add(acc_linear, e4::mul(alpha_power, linear_contribution));
        }
    }

    // Linear constraints
    for (uint constraint = metadata.num_non_boolean_quadratic_constraints; constraint < metadata.num_non_boolean_constraints; constraint++) {
        const uchar2 num_lt = metadata.num_linear_and_quadratic_terms_per_constraint[constraint];
        const uint num_linear_terms = num_lt.y;

        bf linear_contribution = bf::zero();
        const uint lim = flat_term_idx + num_linear_terms;
        for (; flat_term_idx < lim; flat_term_idx++) {
            bf val = get_witness_or_memory(metadata.col_idxs[flat_col_idx++], witness_cols, memory_cols);
            maybe_apply_coeff(metadata, flat_term_idx, explicit_coeff_idx, val);
            linear_contribution = bf::add(linear_contribution, val);
        }
        const e4 alpha_power = alphas.get(alpha_idx++);
        acc_linear = e4::add(acc_linear, e4::mul(alpha_power, linear_contribution));
    }

    acc_quadratic = e4::mul(acc_quadratic, metadata.decompression_factor_squared);
    acc_linear = e4::mul(acc_linear, metadata.decompression_factor);
    e4 acc = e4::add(acc_quadratic, acc_linear);
    quotient.set(acc);
}

// Lookup args construction structures
constant constexpr uint LOOKUP_VAL_IS_COL_FLAG = 255;

constant constexpr uint DELEGATED_MAX_WIDTH_3_LOOKUPS = 224;
constant constexpr uint DELEGATED_MAX_WIDTH_3_LOOKUP_VALS = 640;
constant constexpr uint DELEGATED_MAX_WIDTH_3_LOOKUP_COEFFS = 1408;
constant constexpr uint DELEGATED_MAX_WIDTH_3_LOOKUP_COLS = 1888;

struct DelegatedWidth3LookupsLayout {
    uint coeffs[DELEGATED_MAX_WIDTH_3_LOOKUP_COEFFS];
    uint16_t col_idxs[DELEGATED_MAX_WIDTH_3_LOOKUP_COLS];
    uint8_t num_terms_per_expression[DELEGATED_MAX_WIDTH_3_LOOKUP_VALS];
    bool table_id_is_col[DELEGATED_MAX_WIDTH_3_LOOKUPS];
    uint16_t e4_arg_cols[DELEGATED_MAX_WIDTH_3_LOOKUPS];
    uint helpers_offset;
    uint num_helpers_used;
    uint num_lookups;
    uint e4_arg_cols_start;
};

constant constexpr uint NON_DELEGATED_MAX_WIDTH_3_LOOKUPS = 24;
constant constexpr uint NON_DELEGATED_MAX_WIDTH_3_LOOKUP_VALS = 72;
constant constexpr uint NON_DELEGATED_MAX_WIDTH_3_LOOKUP_COEFFS = 32;
constant constexpr uint NON_DELEGATED_MAX_WIDTH_3_LOOKUP_COLS = 96;

struct NonDelegatedWidth3LookupsLayout {
    uint coeffs[NON_DELEGATED_MAX_WIDTH_3_LOOKUP_COEFFS];
    uint16_t col_idxs[NON_DELEGATED_MAX_WIDTH_3_LOOKUP_COLS];
    uint8_t num_terms_per_expression[NON_DELEGATED_MAX_WIDTH_3_LOOKUP_VALS];
    bool table_id_is_col[NON_DELEGATED_MAX_WIDTH_3_LOOKUPS];
    uint16_t e4_arg_cols[NON_DELEGATED_MAX_WIDTH_3_LOOKUPS];
    uint helpers_offset;
    uint num_helpers_used;
    uint num_lookups;
    uint e4_arg_cols_start;
};

template <typename T>
DEVICE_FORCEINLINE void enforce_width_3_lookup_args_construction(
    const device T &layout,
    const thread matrix_getter<bf> &witness_cols,
    const thread matrix_getter<bf> &memory_cols,
    const thread vectorized_e4_matrix_getter &stage_2_e4_cols,
    thread vector_getter<e4> &helpers,
    thread e4 &acc_quadratic
) {
    uint col_idx = 0;
    uint val_idx = 0;
    uint coeff_idx = 0;
    for (uint term_idx = 0; term_idx < layout.num_lookups; term_idx++) {
        e4 acc = helpers.get(0); helpers.ptr++;
        if (layout.table_id_is_col[term_idx]) {
            const bf id = witness_cols.get_at_col(layout.col_idxs[col_idx++]);
            acc = e4::add(acc, e4::mul(helpers.get(0), id)); helpers.ptr++;
        }
        for (uint j = 0; j < NUM_LOOKUP_ARGUMENT_KEY_PARTS - 1; j++) {
            const uint num_expr_terms = layout.num_terms_per_expression[val_idx++];
            if (num_expr_terms == LOOKUP_VAL_IS_COL_FLAG) {
                const bf val = get_witness_or_memory(layout.col_idxs[col_idx++], witness_cols, memory_cols);
                acc = e4::add(acc, e4::mul(helpers.get(0), val)); helpers.ptr++;
            } else {
                bf val = bf::zero();
                const uint lim = col_idx + num_expr_terms;
                for (; col_idx < lim; col_idx++) {
                    bf next = get_witness_or_memory(layout.col_idxs[col_idx], witness_cols, memory_cols);
                    apply_coeff(layout.coeffs[coeff_idx++], next);
                    val = bf::add(val, next);
                }
                if (num_expr_terms > 0) {
                    acc = e4::add(acc, e4::mul(helpers.get(0), val)); helpers.ptr++;
                }
            }
        }
        const e4 e4_arg = stage_2_e4_cols.get_at_col(layout.e4_arg_cols[term_idx]);
        acc = e4::mul(acc, e4_arg);
        acc_quadratic = e4::add(acc_quadratic, acc);
    }
}

kernel void ab_delegated_width_3_lookups_kernel(
    const device DelegatedWidth3LookupsLayout &layout [[buffer(0)]],
    const device bf *witness_ptr [[buffer(1)]],
    constant uint &witness_stride [[buffer(2)]],
    const device bf *memory_ptr [[buffer(3)]],
    constant uint &memory_stride [[buffer(4)]],
    const device bf *stage_2_e4_ptr [[buffer(5)]],
    constant uint &stage_2_e4_stride [[buffer(6)]],
    const device e4 *helpers_ptr [[buffer(7)]],
    device bf *quotient_ptr [[buffer(8)]],
    constant uint &quotient_stride [[buffer(9)]],
    constant e2 &decompression_factor_squared [[buffer(10)]],
    constant uint &log_n [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint n = 1u << log_n;
    if (gid >= n)
        return;

    matrix_getter<bf> witness_cols = {witness_ptr, witness_stride};
    matrix_getter<bf> memory_cols = {memory_ptr, memory_stride};
    vectorized_e4_matrix_getter stage_2_e4_cols = {{stage_2_e4_ptr, stage_2_e4_stride}};
    vector_getter<e4> helpers = {helpers_ptr};
    vectorized_e4_matrix_getter_setter quotient = {{quotient_ptr, quotient_stride}};

    witness_cols.add_row(gid);
    memory_cols.add_row(gid);
    stage_2_e4_cols.add_row(gid);
    quotient.add_row(gid);
    helpers.ptr += layout.helpers_offset;

    e4 acc_quadratic = e4::zero();
    enforce_width_3_lookup_args_construction(layout, witness_cols, memory_cols, stage_2_e4_cols, helpers, acc_quadratic);

    acc_quadratic = e4::mul(acc_quadratic, decompression_factor_squared);
    const e4 current_quotient = quotient.get();
    acc_quadratic = e4::add(acc_quadratic, current_quotient);
    quotient.set(acc_quadratic);
}

// Hardcoded constraints kernel: full port of stage3.cu ab_hardcoded_constraints_kernel.

struct MultiplicitiesLayout {
    uint src_cols_start;
    uint dst_cols_start;
    uint setup_cols_start;
    uint num_dst_cols;
};

constant constexpr uint MAX_STATE_LINKAGE_CONSTRAINTS = 2;

struct StateLinkageConstraints {
    uint srcs[MAX_STATE_LINKAGE_CONSTRAINTS];
    uint dsts[MAX_STATE_LINKAGE_CONSTRAINTS];
    uint num_constraints;
};

constant constexpr bf SHIFT_16 = bf{1u << 16};

constant constexpr uint MAX_BOUNDARY_CONSTRAINTS_FIRST_ROW = 8;
constant constexpr uint MAX_BOUNDARY_CONSTRAINTS_ONE_BEFORE_LAST_ROW = 8;

struct BoundaryConstraints {
    uint first_row_cols[MAX_BOUNDARY_CONSTRAINTS_FIRST_ROW];
    uint one_before_last_row_cols[MAX_BOUNDARY_CONSTRAINTS_ONE_BEFORE_LAST_ROW];
    uint num_first_row;
    uint num_one_before_last_row;
};

struct ConstantsTimesChallenges {
    e4 first_row;
    e4 one_before_last_row;
    e4 sum;
};

struct HardcodedConstraintsParams {
    uint setup_stride;
    uint witness_stride;
    uint memory_stride;
    uint stage_2_bf_stride;
    uint stage_2_e4_stride;
    uint quotient_stride;
    uint process_delegations_flag;
    uint handle_delegation_requests_flag;
    uint process_batch_ram_access_flag;
    uint process_registers_flag;
    uint delegation_aux_poly_col;
    uint memory_args_start;
    uint memory_grand_product_col;
    uint log_n;
    bf memory_timestamp_high_from_circuit_idx;
    uint _pad0; // align e2 fields to match Rust layout (Rust E2 has align(8))
    e2 decompression_factor;
    e2 decompression_factor_squared;
    e2 every_row_zerofier;
    e2 omega_inv;
    e2 omega_inv_squared;
    uint twiddle_fine_mask;
    uint twiddle_fine_log_count;
    uint twiddle_coarser_mask;
    uint twiddle_coarser_log_count;
    uint twiddle_coarsest_mask;
    uint twiddle_coarsest_log_count;
};

// Helpers used in the hardcoded constraints kernel
DEVICE_FORCEINLINE void enforce_val_zero_if_pred_zero(const bf predicate, const bf val,
                                                       thread vector_getter<e4> &alphas,
                                                       thread e4 &acc_quadratic, thread e4 &acc_linear) {
    const e4 alpha_power = alphas.get(0); alphas.ptr++;
    const bf prod = bf::mul(predicate, val);
    acc_quadratic = e4::add(acc_quadratic, e4::mul(alpha_power, prod));
    acc_linear = e4::add(acc_linear, e4::mul(alpha_power, bf::neg(val)));
}

DEVICE_FORCEINLINE void enforce_width_1_bf_arg_construction(
    const bf a, const bf b, const bf bf_arg,
    thread vector_getter<e4> &alphas, thread vector_getter<e4> &helpers,
    thread e4 &acc_linear, thread e4 &acc_quadratic
) {
    (void)helpers;
    const e4 alpha = alphas.get(0); alphas.ptr++;
    const bf prod = bf::mul(a, b);
    acc_quadratic = e4::add(acc_quadratic, e4::mul(alpha, prod));
    acc_linear = e4::add(acc_linear, e4::mul(alpha, bf::neg(bf_arg)));
}

DEVICE_FORCEINLINE void enforce_width_1_e4_arg_construction(
    const bf a, const bf b, const bf bf_arg, const uint e4_arg_idx,
    const thread vectorized_e4_matrix_getter &stage_2_e4_cols,
    thread vector_getter<e4> &alphas, thread vector_getter<e4> &helpers,
    thread e4 &acc_linear, thread e4 &acc_quadratic
) {
    const e4 alpha = alphas.get(0); alphas.ptr++;
    const bf sum = bf::add(a, b);
    acc_linear = e4::add(acc_linear, e4::mul(alpha, bf::neg(sum)));
    const e4 alpha_times_gamma = helpers.get(0); helpers.ptr++;
    const e4 alpha_times_gamma_squared_adjusted = helpers.get(0); helpers.ptr++;
    const e4 bf_arg_term = e4::mul(alpha, bf_arg);
    const e4 gamma_terms = e4::add(alpha_times_gamma_squared_adjusted, e4::mul(alpha_times_gamma, sum));
    const e4 denoms_prod = e4::add(bf_arg_term, gamma_terms);
    const e4 e4_arg = stage_2_e4_cols.get_at_col(e4_arg_idx);
    const e4 quadratic_term = e4::mul(e4_arg, denoms_prod);
    acc_quadratic = e4::add(acc_quadratic, quadratic_term);
}

template <typename T>
DEVICE_FORCEINLINE void enforce_range_check_expressions_with_constant_terms(
    const device T &expressions, thread uint &i, thread uint &expression_idx,
    thread uint &flat_term_idx, const thread matrix_getter<bf> &witness_cols,
    const thread matrix_getter<bf> &memory_cols, const thread matrix_getter<bf> &stage_2_bf_cols,
    const thread vectorized_e4_matrix_getter &stage_2_e4_cols,
    const uint expression_pair_bound, thread vector_getter<e4> &alphas,
    thread vector_getter<e4> &helpers, thread e4 &acc_linear, thread e4 &acc_quadratic
) {
    for (; i < expression_pair_bound; i++) {
        bf a_and_b[2];
        eval_a_and_b<false>(a_and_b, expressions, expression_idx, flat_term_idx, witness_cols, memory_cols, false);
        const bf a = a_and_b[0];
        const bf b = a_and_b[1];
        const bf bf_arg = stage_2_bf_cols.get_at_col(expressions.bf_dst_cols[i]);
        const e4 alpha = alphas.get(0); alphas.ptr++;
        const bf prod = bf::mul(a, b);
        acc_quadratic = e4::add(acc_quadratic, e4::mul(alpha, prod));
        const bf a_constant_term = expressions.constant_terms[expression_idx - 2];
        const bf b_constant_term = expressions.constant_terms[expression_idx - 1];
        const bf linear_contribution_from_a_b_constants = bf::add(bf::mul(a, b_constant_term), bf::mul(b, a_constant_term));
        acc_linear = e4::add(acc_linear, e4::mul(alpha, bf::sub(linear_contribution_from_a_b_constants, bf_arg)));
        enforce_width_1_e4_arg_construction(a, b, bf_arg, expressions.e4_dst_cols[i], stage_2_e4_cols, alphas, helpers, acc_linear, acc_quadratic);
    }
}

template <uint ENTRY_WIDTH>
DEVICE_FORCEINLINE void enforce_lookup_multiplicities(
    const device MultiplicitiesLayout &layout, const thread matrix_getter<bf> &setup_cols,
    const thread matrix_getter<bf> &witness_cols, const thread vectorized_e4_matrix_getter &stage_2_e4_cols,
    thread vector_getter<e4> &alphas, thread vector_getter<e4> &helpers,
    thread e4 &acc_linear, thread e4 &acc_quadratic
) {
    for (uint i = 0; i < layout.num_dst_cols; i++) {
        const e4 alpha = alphas.get(0); alphas.ptr++;
        const bf m = witness_cols.get_at_col(layout.src_cols_start + i);
        acc_linear = e4::add(acc_linear, e4::mul(alpha, bf::neg(m)));
        e4 denom = helpers.get(0); helpers.ptr++;
        const uint setup_col_start = layout.setup_cols_start + i * ENTRY_WIDTH;
        denom = e4::add(denom, e4::mul(alpha, setup_cols.get_at_col(setup_col_start)));
        if (ENTRY_WIDTH > 1) {
            for (uint j = 1; j < ENTRY_WIDTH; j++) {
                const e4 adjusted_linearization_challenge = helpers.get(0); helpers.ptr++;
                const bf val = setup_cols.get_at_col(setup_col_start + j);
                denom = e4::add(denom, e4::mul(adjusted_linearization_challenge, val));
            }
        }
        const e4 e4_arg = stage_2_e4_cols.get_at_col(layout.dst_cols_start + i);
        denom = e4::mul(denom, e4_arg);
        acc_quadratic = e4::add(acc_quadratic, denom);
    }
}

kernel void ab_hardcoded_constraints_kernel(
    const device bf *setup_ptr [[buffer(0)]],
    const device bf *witness_ptr [[buffer(1)]],
    const device bf *memory_ptr [[buffer(2)]],
    const device bf *stage_2_bf_ptr [[buffer(3)]],
    const device bf *stage_2_e4_ptr [[buffer(4)]],
    const device DelegationChallenges &delegation_challenges [[buffer(5)]],
    const device DelegationProcessingMetadata &delegation_processing_metadata [[buffer(6)]],
    const device DelegationRequestMetadata &delegation_request_metadata [[buffer(7)]],
    const device LazyInitTeardownLayout &lazy_init_teardown_layout [[buffer(8)]],
    const device ShuffleRamAccesses &shuffle_ram_accesses [[buffer(9)]],
    const device BatchedRamAccesses &batched_ram_accesses [[buffer(10)]],
    const device RegisterAndIndirectAccesses &register_accesses [[buffer(11)]],
    const device RangeCheckArgsLayout &range_check_16_layout [[buffer(12)]],
    const device FlattenedLookupExpressionsLayout &expressions [[buffer(13)]],
    const device FlattenedLookupExpressionsForShuffleRamLayout &expressions_for_shuffle_ram [[buffer(14)]],
    const device NonDelegatedWidth3LookupsLayout &width_3_lookups_layout [[buffer(15)]],
    const device MultiplicitiesLayout &range_check_16_multiplicities_layout [[buffer(16)]],
    const device MultiplicitiesLayout &timestamp_range_check_multiplicities_layout [[buffer(17)]],
    const device MultiplicitiesLayout &generic_lookup_multiplicities_layout [[buffer(18)]],
    const device StateLinkageConstraints &state_linkage_constraints [[buffer(19)]],
    const device BoundaryConstraints &boundary_constraints [[buffer(20)]],
    const device e4 *alphas_ptr [[buffer(21)]],
    const device e4 *alphas_every_row_except_last_two_ptr [[buffer(22)]],
    const device e4 *betas_ptr [[buffer(23)]],
    const device e4 *helpers_ptr [[buffer(24)]],
    const device ConstantsTimesChallenges &constants_times_challenges [[buffer(25)]],
    device bf *quotient_ptr [[buffer(26)]],
    const device e2 *twiddle_fine_values [[buffer(27)]],
    const device e2 *twiddle_coarser_values [[buffer(28)]],
    const device e2 *twiddle_coarsest_values [[buffer(29)]],
    constant HardcodedConstraintsParams &params [[buffer(30)]],
    uint gid [[thread_position_in_grid]]
) {
    (void)delegation_challenges;
    const uint n = 1u << params.log_n;
    if (gid >= n)
        return;

    const bool process_delegations = params.process_delegations_flag != 0;
    const bool handle_delegation_requests = params.handle_delegation_requests_flag != 0;
    const bool process_batch_ram_access = params.process_batch_ram_access_flag != 0;
    const bool process_registers_and_indirect_access = params.process_registers_flag != 0;

    matrix_getter<bf> setup_cols = {setup_ptr, params.setup_stride};
    matrix_getter<bf> witness_cols = {witness_ptr, params.witness_stride};
    matrix_getter<bf> memory_cols = {memory_ptr, params.memory_stride};
    matrix_getter<bf> stage_2_bf_cols = {stage_2_bf_ptr, params.stage_2_bf_stride};
    vectorized_e4_matrix_getter stage_2_e4_cols = {{stage_2_e4_ptr, params.stage_2_e4_stride}};
    vectorized_e4_matrix_getter_setter quotient = {{quotient_ptr, params.quotient_stride}};
    vector_getter<e4> alphas = {alphas_ptr};
    vector_getter<e4> alphas_every_row_except_last_two = {alphas_every_row_except_last_two_ptr};
    vector_getter<e4> betas = {betas_ptr};
    vector_getter<e4> helpers = {helpers_ptr};

    setup_cols.add_row(gid);
    witness_cols.add_row(gid);
    memory_cols.add_row(gid);
    stage_2_bf_cols.add_row(gid);
    stage_2_e4_cols.add_row(gid);
    quotient.add_row(gid);

    e4 acc_linear = e4::zero();
    e4 acc_quadratic = e4::zero();

    // ========================================================================
    // 1. Delegation processing constraints
    // ========================================================================
    if (process_delegations) {
        const bf predicate = memory_cols.get_at_col(delegation_processing_metadata.multiplicity_col);
        const bf vals[4] = {
            predicate,
            memory_cols.get_at_col(delegation_processing_metadata.abi_mem_offset_high_col),
            memory_cols.get_at_col(delegation_processing_metadata.write_timestamp_col),
            memory_cols.get_at_col(delegation_processing_metadata.write_timestamp_col + 1)
        };
        for (uint i = 0; i < 4; i++)
            enforce_val_zero_if_pred_zero(predicate, vals[i], alphas, acc_quadratic, acc_linear);

        if (process_batch_ram_access) {
            for (uint i = 0; i < batched_ram_accesses.num_accesses; i++) {
                const device auto &access = batched_ram_accesses.accesses[i];
                enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_timestamp_col), alphas, acc_quadratic, acc_linear);
                enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_timestamp_col + 1), alphas, acc_quadratic, acc_linear);
                enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_value_col), alphas, acc_quadratic, acc_linear);
                enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_value_col + 1), alphas, acc_quadratic, acc_linear);
                if (access.is_write) {
                    enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.maybe_write_value_col), alphas, acc_quadratic, acc_linear);
                    enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.maybe_write_value_col + 1), alphas, acc_quadratic, acc_linear);
                }
            }
        }

        if (process_registers_and_indirect_access) {
            uint flat_indirect_idx = 0;
            for (uint i = 0; i < register_accesses.num_register_accesses; i++) {
                {
                    const device auto &access = register_accesses.register_accesses[i];
                    enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_timestamp_col), alphas, acc_quadratic, acc_linear);
                    enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_timestamp_col + 1), alphas, acc_quadratic, acc_linear);
                    enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_value_col), alphas, acc_quadratic, acc_linear);
                    enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_value_col + 1), alphas, acc_quadratic, acc_linear);
                    if (access.is_write) {
                        enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.maybe_write_value_col), alphas, acc_quadratic, acc_linear);
                        enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.maybe_write_value_col + 1), alphas, acc_quadratic, acc_linear);
                    }
                }
                const uint num_indirect_accesses = register_accesses.indirect_accesses_per_register_access[i];
                for (uint j = 0; j < num_indirect_accesses; j++, flat_indirect_idx++) {
                    const device auto &access = register_accesses.indirect_accesses[flat_indirect_idx];
                    enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_timestamp_col), alphas, acc_quadratic, acc_linear);
                    enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_timestamp_col + 1), alphas, acc_quadratic, acc_linear);
                    enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_value_col), alphas, acc_quadratic, acc_linear);
                    enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.read_value_col + 1), alphas, acc_quadratic, acc_linear);
                    if (access.is_write) {
                        enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.maybe_write_value_col), alphas, acc_quadratic, acc_linear);
                        enforce_val_zero_if_pred_zero(predicate, memory_cols.get_at_col(access.maybe_write_value_col + 1), alphas, acc_quadratic, acc_linear);
                    }
                    if (j > 0 && access.address_derivation_carry_bit_num_elements > 0) {
                        const bf carry_bit = memory_cols.get_at_col(access.address_derivation_carry_bit_col);
                        enforce_val_zero_if_pred_zero(carry_bit, carry_bit, alphas, acc_quadratic, acc_linear);
                    }
                }
            }
        }
    }

    // ========================================================================
    // 2. Range check 16 and timestamp range check args
    // ========================================================================
    {
        for (uint i = 0; i < range_check_16_layout.num_dst_cols; i++) {
            const uint src = 2 * i + range_check_16_layout.src_cols_start;
            const bf a = witness_cols.get_at_col(src);
            const bf b = witness_cols.get_at_col(src + 1);
            const bf bf_arg = stage_2_bf_cols.get_at_col(range_check_16_layout.bf_args_start + i);
            enforce_width_1_bf_arg_construction(a, b, bf_arg, alphas, helpers, acc_linear, acc_quadratic);
            enforce_width_1_e4_arg_construction(a, b, bf_arg, range_check_16_layout.e4_args_start + i, stage_2_e4_cols, alphas, helpers, acc_linear, acc_quadratic);
        }

        uint i = 0;
        uint expression_idx = 0;
        uint flat_term_idx = 0;

        if (expressions.range_check_16_constant_terms_are_zero) {
            for (; i < expressions.num_range_check_16_expression_pairs; i++) {
                bf a_and_b[2];
                eval_a_and_b<false>(a_and_b, expressions, expression_idx, flat_term_idx, witness_cols, memory_cols, true);
                const bf bf_arg = stage_2_bf_cols.get_at_col(expressions.bf_dst_cols[i]);
                enforce_width_1_bf_arg_construction(a_and_b[0], a_and_b[1], bf_arg, alphas, helpers, acc_linear, acc_quadratic);
                enforce_width_1_e4_arg_construction(a_and_b[0], a_and_b[1], bf_arg, expressions.e4_dst_cols[i], stage_2_e4_cols, alphas, helpers, acc_linear, acc_quadratic);
            }
        } else {
            enforce_range_check_expressions_with_constant_terms(expressions, i, expression_idx, flat_term_idx, witness_cols, memory_cols, stage_2_bf_cols,
                                                                stage_2_e4_cols, expressions.num_range_check_16_expression_pairs, alphas, helpers, acc_linear, acc_quadratic);
        }

        if (lazy_init_teardown_layout.process_shuffle_ram_init) {
            const bf a = memory_cols.get_at_col(lazy_init_teardown_layout.init_address_start);
            const bf b = memory_cols.get_at_col(lazy_init_teardown_layout.init_address_start + 1);
            const bf bf_arg = stage_2_bf_cols.get_at_col(lazy_init_teardown_layout.bf_arg_col);
            enforce_width_1_bf_arg_construction(a, b, bf_arg, alphas, helpers, acc_linear, acc_quadratic);
            enforce_width_1_e4_arg_construction(a, b, bf_arg, lazy_init_teardown_layout.e4_arg_col, stage_2_e4_cols, alphas, helpers, acc_linear, acc_quadratic);
        }

        if (expressions.timestamp_constant_terms_are_zero) {
            const uint expression_pair_bound = i + expressions.num_timestamp_expression_pairs;
            for (; i < expression_pair_bound; i++) {
                bf a_and_b[2];
                eval_a_and_b<false>(a_and_b, expressions, expression_idx, flat_term_idx, witness_cols, memory_cols, true);
                const bf bf_arg = stage_2_bf_cols.get_at_col(expressions.bf_dst_cols[i]);
                enforce_width_1_bf_arg_construction(a_and_b[0], a_and_b[1], bf_arg, alphas, helpers, acc_linear, acc_quadratic);
                enforce_width_1_e4_arg_construction(a_and_b[0], a_and_b[1], bf_arg, expressions.e4_dst_cols[i], stage_2_e4_cols, alphas, helpers, acc_linear, acc_quadratic);
            }
        } else {
            const uint expression_pair_bound = i + expressions.num_timestamp_expression_pairs;
            enforce_range_check_expressions_with_constant_terms(expressions, i, expression_idx, flat_term_idx, witness_cols, memory_cols, stage_2_bf_cols,
                                                                stage_2_e4_cols, expression_pair_bound, alphas, helpers, acc_linear, acc_quadratic);
        }

        // Shuffle RAM expressions
        for (uint sr_i = 0, sr_expression_idx = 0, sr_flat_term_idx = 0; sr_i < expressions_for_shuffle_ram.num_expression_pairs; sr_i++) {
            bf a_and_b[2];
            eval_a_and_b<false>(a_and_b, expressions_for_shuffle_ram, sr_expression_idx, sr_flat_term_idx, setup_cols, witness_cols, memory_cols);
            const bf a = a_and_b[0];
            const bf b = a_and_b[1];
            const bf bf_arg = stage_2_bf_cols.get_at_col(expressions_for_shuffle_ram.bf_dst_cols[sr_i]);
            const e4 alpha = alphas.get(0); alphas.ptr++;
            const bf prod = bf::mul(a, b);
            acc_quadratic = e4::add(acc_quadratic, e4::mul(alpha, prod));
            const bf a_constant_term = expressions_for_shuffle_ram.constant_terms[sr_expression_idx - 2];
            const bf b_constant_term = expressions_for_shuffle_ram.constant_terms[sr_expression_idx - 1];
            const bf b_constant_term_adjusted = bf::sub(b_constant_term, params.memory_timestamp_high_from_circuit_idx);
            const bf linear_contribution_from_a_b_constants = bf::add(bf::mul(a, b_constant_term_adjusted), bf::mul(b, a_constant_term));
            acc_linear = e4::add(acc_linear, e4::mul(alpha, bf::sub(linear_contribution_from_a_b_constants, bf_arg)));
            enforce_width_1_e4_arg_construction(a, b, bf_arg, expressions_for_shuffle_ram.e4_dst_cols[sr_i], stage_2_e4_cols, alphas, helpers, acc_linear, acc_quadratic);
        }
    }

    // ========================================================================
    // 3. Width 3 lookup args
    // ========================================================================
    if (process_delegations) {
        alphas.ptr += width_3_lookups_layout.num_lookups;
        helpers.ptr += width_3_lookups_layout.num_helpers_used;
    } else {
        enforce_width_3_lookup_args_construction(width_3_lookups_layout, witness_cols, memory_cols, stage_2_e4_cols, helpers, acc_quadratic);
        alphas.ptr += width_3_lookups_layout.num_lookups;
    }

    // ========================================================================
    // 4. Lookup multiplicities
    // ========================================================================
    enforce_lookup_multiplicities<1>(range_check_16_multiplicities_layout, setup_cols, witness_cols, stage_2_e4_cols, alphas, helpers, acc_linear, acc_quadratic);
    enforce_lookup_multiplicities<1>(timestamp_range_check_multiplicities_layout, setup_cols, witness_cols, stage_2_e4_cols, alphas, helpers, acc_linear, acc_quadratic);
    enforce_lookup_multiplicities<NUM_LOOKUP_ARGUMENT_KEY_PARTS>(generic_lookup_multiplicities_layout, setup_cols, witness_cols, stage_2_e4_cols, alphas, helpers, acc_linear, acc_quadratic);

    // ========================================================================
    // 5. Delegation requests
    // ========================================================================
    if (handle_delegation_requests) {
        const bf m = memory_cols.get_at_col(delegation_request_metadata.multiplicity_col);
        const e4 alpha = alphas.get(0); alphas.ptr++;
        acc_linear = e4::add(acc_linear, e4::mul(alpha, bf::neg(m)));
        e4 denom = helpers.get(0); helpers.ptr++;
        denom = e4::add(denom, e4::mul(alpha, memory_cols.get_at_col(delegation_request_metadata.delegation_type_col)));
        denom = e4::add(denom, e4::mul(helpers.get(0), memory_cols.get_at_col(delegation_request_metadata.abi_mem_offset_high_col))); helpers.ptr++;
        denom = e4::add(denom, e4::mul(helpers.get(0), setup_cols.get_at_col(delegation_request_metadata.timestamp_setup_col))); helpers.ptr++;
        denom = e4::add(denom, e4::mul(helpers.get(0), setup_cols.get_at_col(delegation_request_metadata.timestamp_setup_col + 1))); helpers.ptr++;
        const e4 e4_arg = stage_2_e4_cols.get_at_col(params.delegation_aux_poly_col);
        acc_quadratic = e4::add(acc_quadratic, e4::mul(e4_arg, denom));
    }

    // ========================================================================
    // 6. Delegation processing memory args
    // ========================================================================
    if (process_delegations) {
        const bf m = memory_cols.get_at_col(delegation_processing_metadata.multiplicity_col);
        const e4 alpha = alphas.get(0); alphas.ptr++;
        acc_linear = e4::add(acc_linear, e4::mul(alpha, bf::neg(m)));
        e4 denom = helpers.get(0); helpers.ptr++;
        denom = e4::add(denom, e4::mul(helpers.get(0), memory_cols.get_at_col(delegation_processing_metadata.abi_mem_offset_high_col))); helpers.ptr++;
        denom = e4::add(denom, e4::mul(helpers.get(0), memory_cols.get_at_col(delegation_processing_metadata.write_timestamp_col))); helpers.ptr++;
        denom = e4::add(denom, e4::mul(helpers.get(0), memory_cols.get_at_col(delegation_processing_metadata.write_timestamp_col + 1))); helpers.ptr++;
        const e4 e4_arg = stage_2_e4_cols.get_at_col(params.delegation_aux_poly_col);
        acc_quadratic = e4::add(acc_quadratic, e4::mul(e4_arg, denom));
    }

    // ========================================================================
    // 7. Lazy init/teardown (shuffle RAM)
    // ========================================================================
    if (lazy_init_teardown_layout.process_shuffle_ram_init) {
        e4 e4_arg_prev = e4::zero();
        {
            const bf address_low = memory_cols.get_at_col(lazy_init_teardown_layout.init_address_start);
            const bf address_high = memory_cols.get_at_col(lazy_init_teardown_layout.init_address_start + 1);
            const bf value_low = memory_cols.get_at_col(lazy_init_teardown_layout.teardown_value_start);
            const bf value_high = memory_cols.get_at_col(lazy_init_teardown_layout.teardown_value_start + 1);
            const bf timestamp_low = memory_cols.get_at_col(lazy_init_teardown_layout.teardown_timestamp_start);
            const bf timestamp_high = memory_cols.get_at_col(lazy_init_teardown_layout.teardown_timestamp_start + 1);
            const bf final_borrow = witness_cols.get_at_col(lazy_init_teardown_layout.init_address_final_borrow);

            enforce_val_zero_if_pred_zero(final_borrow, address_low, alphas, acc_quadratic, acc_linear);
            enforce_val_zero_if_pred_zero(final_borrow, address_high, alphas, acc_quadratic, acc_linear);
            enforce_val_zero_if_pred_zero(final_borrow, value_low, alphas, acc_quadratic, acc_linear);
            enforce_val_zero_if_pred_zero(final_borrow, value_high, alphas, acc_quadratic, acc_linear);
            enforce_val_zero_if_pred_zero(final_borrow, timestamp_low, alphas, acc_quadratic, acc_linear);
            enforce_val_zero_if_pred_zero(final_borrow, timestamp_high, alphas, acc_quadratic, acc_linear);

            e4 numerator = e4::mul(helpers.get(0), address_low); helpers.ptr++;
            numerator = e4::add(numerator, e4::mul(helpers.get(0), address_high)); helpers.ptr++;
            acc_linear = e4::sub(acc_linear, numerator);

            e4 denom = numerator;
            denom = e4::add(denom, e4::mul(helpers.get(0), value_low)); helpers.ptr++;
            denom = e4::add(denom, e4::mul(helpers.get(0), value_high)); helpers.ptr++;
            denom = e4::add(denom, e4::mul(helpers.get(0), timestamp_low)); helpers.ptr++;
            denom = e4::add(denom, e4::mul(helpers.get(0), timestamp_high)); helpers.ptr++;

            const e4 alpha_times_gamma_adjusted = helpers.get(0); helpers.ptr++;
            denom = e4::add(denom, alpha_times_gamma_adjusted);

            const e4 e4_arg = stage_2_e4_cols.get_at_col(params.memory_args_start);
            acc_quadratic = e4::add(acc_quadratic, e4::mul(e4_arg, denom));
            e4_arg_prev = e4_arg;

            alphas.ptr++; // advance alpha explicitly
        }

        const bf write_timestamp_in_setup_low = setup_cols.get_at_col(shuffle_ram_accesses.write_timestamp_in_setup_start);
        const bf write_timestamp_in_setup_high = setup_cols.get_at_col(shuffle_ram_accesses.write_timestamp_in_setup_start + 1);
        for (uint i = 0; i < shuffle_ram_accesses.num_accesses; i++) {
            const device auto &access = shuffle_ram_accesses.accesses[i];

            const bf address_low = memory_cols.get_at_col(access.address_start);
            e4 numerator = e4::mul(helpers.get(0), address_low); helpers.ptr++;

            if (access.is_register_only) {
                alphas.ptr++; // constant bf::one() already accounted for in numerator constant helper
            } else {
                const bf address_high = memory_cols.get_at_col(access.address_start + 1);
                numerator = e4::add(numerator, e4::mul(helpers.get(0), address_high)); helpers.ptr++;
                numerator = e4::add(numerator, e4::mul(alphas.get(0), memory_cols.get_at_col(access.maybe_is_register_start))); alphas.ptr++;
            }

            e4 denom = e4::zero();

            const e4 value_low_helper = helpers.get(0); helpers.ptr++;
            const e4 value_high_helper = helpers.get(0); helpers.ptr++;
            if (access.is_write) {
                denom = numerator;

                const bf read_value_low = memory_cols.get_at_col(access.read_value_start);
                denom = e4::add(denom, e4::mul(value_low_helper, read_value_low));
                const bf read_value_high = memory_cols.get_at_col(access.read_value_start + 1);
                denom = e4::add(denom, e4::mul(value_high_helper, read_value_high));

                const bf write_value_low = memory_cols.get_at_col(access.maybe_write_value_start);
                numerator = e4::add(numerator, e4::mul(value_low_helper, write_value_low));
                const bf write_value_high = memory_cols.get_at_col(access.maybe_write_value_start + 1);
                numerator = e4::add(numerator, e4::mul(value_high_helper, write_value_high));
            } else {
                const bf value_low = memory_cols.get_at_col(access.read_value_start);
                numerator = e4::add(numerator, e4::mul(value_low_helper, value_low));
                const bf value_high = memory_cols.get_at_col(access.read_value_start + 1);
                numerator = e4::add(numerator, e4::mul(value_high_helper, value_high));

                denom = numerator;
            }

            const e4 timestamp_low_helper = helpers.get(0); helpers.ptr++;
            const e4 timestamp_high_helper = helpers.get(0); helpers.ptr++;

            const bf read_timestamp_low = memory_cols.get_at_col(access.read_timestamp_start);
            denom = e4::add(denom, e4::mul(timestamp_low_helper, read_timestamp_low));
            const bf read_timestamp_high = memory_cols.get_at_col(access.read_timestamp_start + 1);
            denom = e4::add(denom, e4::mul(timestamp_high_helper, read_timestamp_high));

            numerator = e4::add(numerator, e4::mul(timestamp_low_helper, write_timestamp_in_setup_low));
            numerator = e4::add(numerator, e4::mul(timestamp_high_helper, write_timestamp_in_setup_high));

            // adjusted constant contributions
            denom = e4::add(denom, helpers.get(0)); helpers.ptr++;
            numerator = e4::add(numerator, helpers.get(0)); helpers.ptr++;

            const e4 e4_arg = stage_2_e4_cols.get_at_col(params.memory_args_start + 1 + i);
            acc_quadratic = e4::add(acc_quadratic, e4::mul(e4_arg, denom));

            acc_quadratic = e4::sub(acc_quadratic, e4::mul(e4_arg_prev, numerator));
            e4_arg_prev = e4_arg;
        }
    }

    // ========================================================================
    // 8. Batch RAM access
    // ========================================================================
    if (process_batch_ram_access) {
        const bf address_high = memory_cols.get_at_col(batched_ram_accesses.abi_mem_offset_high_col);
        const bf write_timestamp_low = memory_cols.get_at_col(batched_ram_accesses.write_timestamp_col);
        const bf write_timestamp_high = memory_cols.get_at_col(batched_ram_accesses.write_timestamp_col + 1);
        for (uint i = 0; i < batched_ram_accesses.num_accesses; i++) {
            const device auto &access = batched_ram_accesses.accesses[i];

            e4 numerator = e4::mul(helpers.get(0), address_high); helpers.ptr++;

            e4 denom = e4::zero();

            const e4 value_low_helper = helpers.get(0); helpers.ptr++;
            const e4 value_high_helper = helpers.get(0); helpers.ptr++;
            if (access.is_write) {
                denom = numerator;

                const bf read_value_low = memory_cols.get_at_col(access.read_value_col);
                denom = e4::add(denom, e4::mul(value_low_helper, read_value_low));
                const bf read_value_high = memory_cols.get_at_col(access.read_value_col + 1);
                denom = e4::add(denom, e4::mul(value_high_helper, read_value_high));

                const bf wv_low = memory_cols.get_at_col(access.maybe_write_value_col);
                numerator = e4::add(numerator, e4::mul(value_low_helper, wv_low));
                const bf wv_high = memory_cols.get_at_col(access.maybe_write_value_col + 1);
                numerator = e4::add(numerator, e4::mul(value_high_helper, wv_high));
            } else {
                const bf value_low = memory_cols.get_at_col(access.read_value_col);
                numerator = e4::add(numerator, e4::mul(value_low_helper, value_low));
                const bf value_high = memory_cols.get_at_col(access.read_value_col + 1);
                numerator = e4::add(numerator, e4::mul(value_high_helper, value_high));

                denom = numerator;
            }

            const e4 timestamp_low_helper = helpers.get(0); helpers.ptr++;
            const e4 timestamp_high_helper = helpers.get(0); helpers.ptr++;

            numerator = e4::add(numerator, e4::mul(timestamp_low_helper, write_timestamp_low));
            numerator = e4::add(numerator, e4::mul(timestamp_high_helper, write_timestamp_high));

            const bf read_timestamp_low = memory_cols.get_at_col(access.read_timestamp_col);
            denom = e4::add(denom, e4::mul(timestamp_low_helper, read_timestamp_low));
            const bf read_timestamp_high = memory_cols.get_at_col(access.read_timestamp_col + 1);
            denom = e4::add(denom, e4::mul(timestamp_high_helper, read_timestamp_high));

            // adjusted constant contributions
            const e4 const_adj = helpers.get(0); helpers.ptr++;
            denom = e4::add(denom, const_adj);
            const e4 e4_arg = stage_2_e4_cols.get_at_col(params.memory_args_start + i);
            acc_quadratic = e4::add(acc_quadratic, e4::mul(e4_arg, denom));

            // flush result
            if (i == 0) {
                acc_linear = e4::sub(acc_linear, numerator);
            } else {
                numerator = e4::add(numerator, const_adj);
                const e4 e4_arg_prev = stage_2_e4_cols.get_at_col(params.memory_args_start + i - 1);
                acc_quadratic = e4::sub(acc_quadratic, e4::mul(e4_arg_prev, numerator));
            }
        }

        alphas.ptr += batched_ram_accesses.num_accesses;
    }

    // ========================================================================
    // 9. Register and indirect access
    // ========================================================================
    if (process_registers_and_indirect_access) {
        const bf write_timestamp_low = memory_cols.get_at_col(register_accesses.write_timestamp_col);
        const bf write_timestamp_high = memory_cols.get_at_col(register_accesses.write_timestamp_col + 1);
        uint flat_indirect_idx = 0;
        e4 e4_arg_prev = e4::zero();
        for (uint i = 0; i < register_accesses.num_register_accesses; i++) {
            bf base_low;
            bf base_high;
            {
                const device auto &access = register_accesses.register_accesses[i];
                e4 numerator = e4::zero();
                e4 denom = e4::zero();

                const e4 value_low_helper = helpers.get(0); helpers.ptr++;
                const e4 value_high_helper = helpers.get(0); helpers.ptr++;
                if (access.is_write) {
                    const bf read_value_low = memory_cols.get_at_col(access.read_value_col);
                    denom = e4::mul(value_low_helper, read_value_low);
                    const bf read_value_high = memory_cols.get_at_col(access.read_value_col + 1);
                    denom = e4::add(denom, e4::mul(value_high_helper, read_value_high));

                    base_low = bf::into_canonical(read_value_low);
                    base_high = bf::into_canonical(read_value_high);

                    const bf wv_low = memory_cols.get_at_col(access.maybe_write_value_col);
                    numerator = e4::mul(value_low_helper, wv_low);
                    const bf wv_high = memory_cols.get_at_col(access.maybe_write_value_col + 1);
                    numerator = e4::add(numerator, e4::mul(value_high_helper, wv_high));
                } else {
                    const bf value_low = memory_cols.get_at_col(access.read_value_col);
                    numerator = e4::mul(value_low_helper, value_low);
                    const bf value_high = memory_cols.get_at_col(access.read_value_col + 1);
                    numerator = e4::add(numerator, e4::mul(value_high_helper, value_high));

                    base_low = bf::into_canonical(value_low);
                    base_high = bf::into_canonical(value_high);

                    denom = numerator;
                }

                const e4 timestamp_low_helper = helpers.get(0); helpers.ptr++;
                const e4 timestamp_high_helper = helpers.get(0); helpers.ptr++;

                numerator = e4::add(numerator, e4::mul(timestamp_low_helper, write_timestamp_low));
                numerator = e4::add(numerator, e4::mul(timestamp_high_helper, write_timestamp_high));

                const bf read_timestamp_low = memory_cols.get_at_col(access.read_timestamp_col);
                denom = e4::add(denom, e4::mul(timestamp_low_helper, read_timestamp_low));
                const bf read_timestamp_high = memory_cols.get_at_col(access.read_timestamp_col + 1);
                denom = e4::add(denom, e4::mul(timestamp_high_helper, read_timestamp_high));

                // adjusted constant contributions
                const e4 const_adj = helpers.get(0); helpers.ptr++;
                denom = e4::add(denom, const_adj);
                const e4 e4_arg = stage_2_e4_cols.get_at_col(params.memory_args_start + i + flat_indirect_idx);
                acc_quadratic = e4::add(acc_quadratic, e4::mul(e4_arg, denom));

                // flush result
                if (i == 0) {
                    acc_linear = e4::sub(acc_linear, numerator);
                    e4_arg_prev = e4_arg;
                } else {
                    numerator = e4::add(numerator, const_adj);
                    acc_quadratic = e4::sub(acc_quadratic, e4::mul(e4_arg_prev, numerator));
                    e4_arg_prev = e4_arg;
                }
            }

            const uint start = flat_indirect_idx;
            const uint end = flat_indirect_idx + register_accesses.indirect_accesses_per_register_access[i];
            for (; flat_indirect_idx < end; flat_indirect_idx++) {
                const device auto &access = register_accesses.indirect_accesses[flat_indirect_idx];
                e4 numerator = e4::zero();
                e4 denom = e4::zero();

                const e4 address_low_helper = helpers.get(0); helpers.ptr++;
                const e4 address_high_helper = helpers.get(0); helpers.ptr++;
                if (flat_indirect_idx == start || access.address_derivation_carry_bit_num_elements == 0) {
                    numerator = e4::mul(address_low_helper, base_low);
                    numerator = e4::add(numerator, e4::mul(address_high_helper, base_high));
                } else {
                    const bf carry_bit = memory_cols.get_at_col(access.address_derivation_carry_bit_col);
                    numerator = e4::mul(address_low_helper, bf::sub(base_low, bf::mul(carry_bit, SHIFT_16)));
                    numerator = e4::add(numerator, e4::mul(address_high_helper, bf::add(base_high, carry_bit)));
                }

                const e4 value_low_helper = helpers.get(0); helpers.ptr++;
                const e4 value_high_helper = helpers.get(0); helpers.ptr++;
                if (access.is_write) {
                    denom = numerator;

                    const bf read_value_low = memory_cols.get_at_col(access.read_value_col);
                    denom = e4::add(denom, e4::mul(value_low_helper, read_value_low));
                    const bf read_value_high = memory_cols.get_at_col(access.read_value_col + 1);
                    denom = e4::add(denom, e4::mul(value_high_helper, read_value_high));

                    const bf wv_low = memory_cols.get_at_col(access.maybe_write_value_col);
                    numerator = e4::add(numerator, e4::mul(value_low_helper, wv_low));
                    const bf wv_high = memory_cols.get_at_col(access.maybe_write_value_col + 1);
                    numerator = e4::add(numerator, e4::mul(value_high_helper, wv_high));
                } else {
                    const bf value_low = memory_cols.get_at_col(access.read_value_col);
                    numerator = e4::add(numerator, e4::mul(value_low_helper, value_low));
                    const bf value_high = memory_cols.get_at_col(access.read_value_col + 1);
                    numerator = e4::add(numerator, e4::mul(value_high_helper, value_high));

                    denom = numerator;
                }

                const e4 timestamp_low_helper = helpers.get(0); helpers.ptr++;
                const e4 timestamp_high_helper = helpers.get(0); helpers.ptr++;

                numerator = e4::add(numerator, e4::mul(timestamp_low_helper, write_timestamp_low));
                numerator = e4::add(numerator, e4::mul(timestamp_high_helper, write_timestamp_high));

                const bf read_timestamp_low = memory_cols.get_at_col(access.read_timestamp_col);
                denom = e4::add(denom, e4::mul(timestamp_low_helper, read_timestamp_low));
                const bf read_timestamp_high = memory_cols.get_at_col(access.read_timestamp_col + 1);
                denom = e4::add(denom, e4::mul(timestamp_high_helper, read_timestamp_high));

                // adjusted constant contributions
                const e4 const_adj = helpers.get(0); helpers.ptr++;
                denom = e4::add(denom, const_adj);
                const e4 e4_arg = stage_2_e4_cols.get_at_col(params.memory_args_start + flat_indirect_idx + i + 1);
                acc_quadratic = e4::add(acc_quadratic, e4::mul(e4_arg, denom));

                // flush result
                numerator = e4::add(numerator, const_adj);
                acc_quadratic = e4::sub(acc_quadratic, e4::mul(e4_arg_prev, numerator));
                e4_arg_prev = e4_arg;
            }
        }

        alphas.ptr += register_accesses.num_register_accesses + flat_indirect_idx;
    }

    // ========================================================================
    // 10. Memory grand product consistency
    // ========================================================================
    {
        const e4 memory_arg_entry = stage_2_e4_cols.get_at_col(params.memory_grand_product_col - 1);
        const e4 grand_product_entry = stage_2_e4_cols.get_at_col(params.memory_grand_product_col);
        e4 grand_product_entry_next = e4::zero();
        if (gid == n - 1) {
            stage_2_e4_cols.sub_row(gid);
            grand_product_entry_next = stage_2_e4_cols.get_at_col(params.memory_grand_product_col);
            stage_2_e4_cols.add_row(gid);
        } else {
            stage_2_e4_cols.add_row(1);
            grand_product_entry_next = stage_2_e4_cols.get_at_col(params.memory_grand_product_col);
            stage_2_e4_cols.sub_row(1);
        }
        const e4 alpha = alphas.get(0); alphas.ptr++;
        acc_linear = e4::add(acc_linear, e4::mul(alpha, grand_product_entry_next));
        const e4 prod = e4::mul(memory_arg_entry, grand_product_entry);
        acc_quadratic = e4::sub(acc_quadratic, e4::mul(alpha, prod));
    }

    // ========================================================================
    // Finalize "every row except last" contributions
    // ========================================================================
    acc_quadratic = e4::mul(acc_quadratic, params.decompression_factor_squared);
    acc_linear = e4::mul(acc_linear, params.decompression_factor);
    e4 acc = e4::add(acc_quadratic, acc_linear);
    const e4 current_quotient = quotient.get();
    acc = e4::add(acc, current_quotient);
    acc = e4::add(acc, constants_times_challenges.sum);

    // Compute x from twiddle factors
    powers_data_3_layer powers;
    powers.fine = {twiddle_fine_values, params.twiddle_fine_mask, params.twiddle_fine_log_count};
    powers.coarser = {twiddle_coarser_values, params.twiddle_coarser_mask, params.twiddle_coarser_log_count};
    powers.coarsest = {twiddle_coarsest_values, params.twiddle_coarsest_mask, params.twiddle_coarsest_log_count};
    const uint shift = 1u << (CIRCLE_GROUP_LOG_ORDER - params.log_n - 1);
    const e2 x = get_power_of_w(powers, shift * (2 * gid + 1), false);

    const e2 num = e2::sub(x, params.omega_inv);
    e2 multiplier = e2::mul(num, params.every_row_zerofier);
    acc = e4::mul(acc, multiplier);
    acc = e4::mul(acc, betas.get(5));

    // ========================================================================
    // 11. Every row except last two contributions
    // ========================================================================
    if (state_linkage_constraints.num_constraints > 0 || lazy_init_teardown_layout.process_shuffle_ram_init) {
        e4 acc_linear_erelto = e4::zero();

        {
            matrix_getter<bf> witness_cols_next_row = witness_cols;
            if (gid < n - 1)
                witness_cols_next_row.add_row(1);
            else
                witness_cols_next_row.sub_row(gid);

            for (uint i = 0; i < state_linkage_constraints.num_constraints; i++) {
                const e4 alpha = alphas_every_row_except_last_two.get(0); alphas_every_row_except_last_two.ptr++;
                const bf src_val = witness_cols.get_at_col(state_linkage_constraints.srcs[i]);
                const bf dst_val = witness_cols_next_row.get_at_col(state_linkage_constraints.dsts[i]);
                acc_linear_erelto = e4::add(acc_linear_erelto, e4::mul(alpha, bf::sub(src_val, dst_val)));
            }
        }

        if (lazy_init_teardown_layout.process_shuffle_ram_init) {
            matrix_getter<bf> memory_cols_next_row = memory_cols;
            if (gid < n - 1)
                memory_cols_next_row.add_row(1);
            else
                memory_cols_next_row.sub_row(gid);

            const bf intermediate_borrow = witness_cols.get_at_col(lazy_init_teardown_layout.init_address_intermediate_borrow);
            {
                const bf this_low = memory_cols.get_at_col(lazy_init_teardown_layout.init_address_start);
                const bf next_low = memory_cols_next_row.get_at_col(lazy_init_teardown_layout.init_address_start);
                const bf aux_low = witness_cols.get_at_col(lazy_init_teardown_layout.init_address_aux_low);
                bf tmp = bf::mul(SHIFT_16, intermediate_borrow);
                tmp = bf::add(tmp, this_low);
                tmp = bf::sub(tmp, next_low);
                tmp = bf::sub(tmp, aux_low);
                const e4 alpha = alphas_every_row_except_last_two.get(0); alphas_every_row_except_last_two.ptr++;
                acc_linear_erelto = e4::add(acc_linear_erelto, e4::mul(alpha, tmp));
            }
            {
                const bf final_borrow = witness_cols.get_at_col(lazy_init_teardown_layout.init_address_final_borrow);
                const bf this_high = memory_cols.get_at_col(lazy_init_teardown_layout.init_address_start + 1);
                const bf next_high = memory_cols_next_row.get_at_col(lazy_init_teardown_layout.init_address_start + 1);
                const bf aux_high = witness_cols.get_at_col(lazy_init_teardown_layout.init_address_aux_high);
                bf tmp = bf::mul(SHIFT_16, final_borrow);
                tmp = bf::add(tmp, this_high);
                tmp = bf::sub(tmp, intermediate_borrow);
                tmp = bf::sub(tmp, next_high);
                tmp = bf::sub(tmp, aux_high);
                const e4 alpha = alphas_every_row_except_last_two.get(0); alphas_every_row_except_last_two.ptr++;
                acc_linear_erelto = e4::add(acc_linear_erelto, e4::mul(alpha, tmp));
            }
        }

        // Finalize "every row except last two" contributions
        acc_linear_erelto = e4::mul(acc_linear_erelto, params.decompression_factor);
        multiplier = e2::mul(multiplier, e2::sub(x, params.omega_inv_squared));
        acc_linear_erelto = e4::mul(acc_linear_erelto, multiplier);
        acc = e4::add(acc, e4::mul(betas.get(4), acc_linear_erelto));
    }

    // ========================================================================
    // Batch inversion for denominators
    // ========================================================================
    const e2 denoms[4] = {x, e2::sub(x, bf::one()), e2::sub(x, params.omega_inv_squared), e2::sub(x, params.omega_inv)};
    e2 denom_invs[4] = {};
    stage3_batch_inv_registers<e2, 4, true>(denoms, denom_invs, 4);

    // ========================================================================
    // 12. First row boundary constraints
    // ========================================================================
    {
        e4 acc_linear_fr = e4::mul(helpers.get(0), stage_2_e4_cols.get_at_col(params.memory_grand_product_col)); helpers.ptr++;
        uint i = 0;
        if (lazy_init_teardown_layout.process_shuffle_ram_init)
            for (; i < 6; i++) {
                acc_linear_fr = e4::add(acc_linear_fr, e4::mul(helpers.get(0), memory_cols.get_at_col(boundary_constraints.first_row_cols[i]))); helpers.ptr++;
            }
        for (; i < boundary_constraints.num_first_row; i++) {
            acc_linear_fr = e4::add(acc_linear_fr, e4::mul(helpers.get(0), witness_cols.get_at_col(boundary_constraints.first_row_cols[i]))); helpers.ptr++;
        }
        acc_linear_fr = e4::add(acc_linear_fr, constants_times_challenges.first_row);
        acc_linear_fr = e4::mul(acc_linear_fr, denom_invs[1]);
        acc = e4::add(acc, acc_linear_fr);
    }

    // ========================================================================
    // 13. One-before-last row boundary constraints
    // ========================================================================
    if (boundary_constraints.num_one_before_last_row > 0) {
        e4 acc_linear_oblr = e4::zero();
        uint i = 0;
        if (lazy_init_teardown_layout.process_shuffle_ram_init) {
            acc_linear_oblr = e4::mul(helpers.get(0), memory_cols.get_at_col(boundary_constraints.one_before_last_row_cols[0])); helpers.ptr++;
            i++;
            for (; i < 6; i++) {
                acc_linear_oblr = e4::add(acc_linear_oblr, e4::mul(helpers.get(0), memory_cols.get_at_col(boundary_constraints.one_before_last_row_cols[i]))); helpers.ptr++;
            }
        } else {
            acc_linear_oblr = e4::mul(helpers.get(0), witness_cols.get_at_col(boundary_constraints.one_before_last_row_cols[0])); helpers.ptr++;
            i++;
        }
        for (; i < boundary_constraints.num_one_before_last_row; i++) {
            acc_linear_oblr = e4::add(acc_linear_oblr, e4::mul(helpers.get(0), witness_cols.get_at_col(boundary_constraints.one_before_last_row_cols[i]))); helpers.ptr++;
        }
        acc_linear_oblr = e4::add(acc_linear_oblr, constants_times_challenges.one_before_last_row);
        acc_linear_oblr = e4::mul(acc_linear_oblr, denom_invs[2]);
        acc = e4::add(acc, acc_linear_oblr);
    }

    // ========================================================================
    // 14. Last row constraints (grand product accumulator)
    // ========================================================================
    {
        e4 acc_linear_lr = e4::mul(helpers.get(0), stage_2_e4_cols.get_at_col(params.memory_grand_product_col)); helpers.ptr++;
        acc_linear_lr = e4::add(acc_linear_lr, helpers.get(0)); helpers.ptr++;
        acc_linear_lr = e4::mul(acc_linear_lr, denom_invs[3]);
        acc = e4::add(acc, acc_linear_lr);
    }

    // ========================================================================
    // 15. Last row and x=0 constraints
    // ========================================================================
    {
        e4 acc_linear_lrx0 = e4::neg(stage_2_e4_cols.get_at_col(range_check_16_multiplicities_layout.dst_cols_start));
        // validate col sums for range check 16 lookup e4 args
        {
            const uint num_range_check_16_e4_args = range_check_16_layout.num_dst_cols + expressions.num_range_check_16_expression_pairs;
            for (uint i = 0; i < num_range_check_16_e4_args; i++)
                acc_linear_lrx0 = e4::add(acc_linear_lrx0, stage_2_e4_cols.get_at_col(range_check_16_layout.e4_args_start + i));
            if (lazy_init_teardown_layout.process_shuffle_ram_init)
                acc_linear_lrx0 = e4::add(acc_linear_lrx0, stage_2_e4_cols.get_at_col(lazy_init_teardown_layout.e4_arg_col));
            acc_linear_lrx0 = e4::mul(acc_linear_lrx0, helpers.get(0)); helpers.ptr++;
        }
        // validate col sums for timestamp range check e4 args
        if (timestamp_range_check_multiplicities_layout.num_dst_cols > 0) {
            e4 acc_timestamp = e4::neg(stage_2_e4_cols.get_at_col(timestamp_range_check_multiplicities_layout.dst_cols_start));
            const uint num_timestamp_e4_args = expressions.num_timestamp_expression_pairs + expressions_for_shuffle_ram.num_expression_pairs;
            const uint start_e4_col = (expressions.num_timestamp_expression_pairs > 0)
                ? expressions.e4_dst_cols[expressions.num_range_check_16_expression_pairs]
                : expressions_for_shuffle_ram.e4_dst_cols[0];
            for (uint i = 0; i < num_timestamp_e4_args; i++)
                acc_timestamp = e4::add(acc_timestamp, stage_2_e4_cols.get_at_col(start_e4_col + i));
            acc_timestamp = e4::mul(acc_timestamp, helpers.get(0)); helpers.ptr++;
            acc_linear_lrx0 = e4::add(acc_linear_lrx0, acc_timestamp);
        }
        // validate col sums for generic lookup e4 args
        {
            e4 acc_generic = e4::neg(stage_2_e4_cols.get_at_col(generic_lookup_multiplicities_layout.dst_cols_start));
            for (uint i = 1; i < generic_lookup_multiplicities_layout.num_dst_cols; i++)
                acc_generic = e4::sub(acc_generic, stage_2_e4_cols.get_at_col(generic_lookup_multiplicities_layout.dst_cols_start + i));
            for (uint i = 0; i < width_3_lookups_layout.num_lookups; i++)
                acc_generic = e4::add(acc_generic, stage_2_e4_cols.get_at_col(width_3_lookups_layout.e4_arg_cols_start + i));
            acc_generic = e4::mul(acc_generic, helpers.get(0)); helpers.ptr++;
            acc_linear_lrx0 = e4::add(acc_linear_lrx0, acc_generic);
        }
        if (handle_delegation_requests || process_delegations) {
            const e4 interpolant = e4::mul(helpers.get(0), x); helpers.ptr++;
            const e4 e4_arg = stage_2_e4_cols.get_at_col(params.delegation_aux_poly_col);
            const e4 diff = e4::sub(e4_arg, interpolant);
            const e4 term = e4::mul(diff, helpers.get(0)); helpers.ptr++;
            acc_linear_lrx0 = e4::add(acc_linear_lrx0, term);
        }
        const e2 denom_inv = e2::mul(denom_invs[0], denom_invs[3]);
        acc_linear_lrx0 = e4::mul(acc_linear_lrx0, denom_inv);
        acc = e4::add(acc, acc_linear_lrx0);
    }

    quotient.set(acc);
}
