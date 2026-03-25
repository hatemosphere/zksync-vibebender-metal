#include "arg_utils.metal"
#include "vectorized.metal"

// Metal port of gpu_prover/native/stage2.cu
// Stage 2 quotient computation kernels: lookup arguments, memory arguments, etc.

using namespace airbender::arg_utils;
using namespace airbender::field;
using namespace airbender::memory;
using namespace airbender::vectorized;

using bf = base_field;
using e2 = ext2_field;
using e4 = ext4_field;

// Populates entry-invs and b-cols for range check 16 lookups (ENTRY_WIDTH = 1).
template <uint ENTRY_WIDTH>
DEVICE_FORCEINLINE void
aggregated_entry_invs_and_multiplicities_arg_impl(
    const device LookupChallenges &challenges,
    matrix_getter<bf> witness_cols,
    matrix_getter<bf> setup_cols,
    vectorized_e4_matrix_setter stage_2_e4_cols,
    vector_setter<e4> aggregated_entry_invs,
    const uint start_col_in_setup,
    const uint multiplicities_src_cols_start,
    const uint multiplicities_dst_cols_start,
    const uint num_multiplicities_cols,
    const uint num_table_rows_tail,
    const uint log_n,
    const uint gid) {
    const uint n = 1u << log_n;
    if (gid >= n - 1)
        return;

    stage_2_e4_cols.add_row(gid);
    stage_2_e4_cols.add_col(multiplicities_dst_cols_start);
    witness_cols.add_row(gid);
    witness_cols.add_col(multiplicities_src_cols_start);
    aggregated_entry_invs.ptr += gid;

    if (ENTRY_WIDTH > 1) {
        setup_cols.add_row(gid);
        setup_cols.add_col(start_col_in_setup);
    }

    const auto linearization_challenges = challenges.linearization_challenges;
    const auto gamma = challenges.gamma;
    for (uint i = 0; i < num_multiplicities_cols; i++) {
        if (i == num_multiplicities_cols - 1 && gid >= num_table_rows_tail) {
            stage_2_e4_cols.set(e4::zero());
            return;
        }

        bf val;
        if (ENTRY_WIDTH == 1) {
            val = bf{gid};
        } else {
            val = setup_cols.get();
            setup_cols.add_col(1);
        }
        e4 denom = e4::add(gamma, val);
        if (ENTRY_WIDTH > 1) {
            for (uint j = 1; j < ENTRY_WIDTH; j++) {
                const auto v = setup_cols.get();
                setup_cols.add_col(1);
                denom = e4::add(denom, e4::mul(linearization_challenges[j - 1], v));
            }
        }

        const e4 denom_inv = e4::inv(denom);

        const auto multiplicity = witness_cols.get();
        stage_2_e4_cols.set(e4::mul(denom_inv, multiplicity));
        aggregated_entry_invs.set(denom_inv);

        witness_cols.add_col(1);
        aggregated_entry_invs.ptr += n - 1;
        stage_2_e4_cols.add_col(1);
    }
}

kernel void ab_range_check_aggregated_entry_invs_and_multiplicities_arg_kernel(
    const device LookupChallenges &challenges [[buffer(0)]],
    const device bf *witness_ptr [[buffer(1)]],
    constant uint &witness_stride [[buffer(2)]],
    const device bf *setup_ptr [[buffer(3)]],
    constant uint &setup_stride [[buffer(4)]],
    device bf *stage_2_e4_ptr [[buffer(5)]],
    constant uint &stage_2_e4_stride [[buffer(6)]],
    device e4 *aggregated_entry_invs [[buffer(7)]],
    constant uint &start_col_in_setup [[buffer(8)]],
    constant uint &multiplicities_src_cols_start [[buffer(9)]],
    constant uint &multiplicities_dst_cols_start [[buffer(10)]],
    constant uint &num_multiplicities_cols [[buffer(11)]],
    constant uint &num_table_rows_tail [[buffer(12)]],
    constant uint &log_n [[buffer(13)]],
    uint gid [[thread_position_in_grid]]
) {
    matrix_getter<bf> witness_cols = {witness_ptr, witness_stride};
    matrix_getter<bf> setup_cols = {setup_ptr, setup_stride};
    vectorized_e4_matrix_setter stage_2_e4_cols = {{stage_2_e4_ptr, stage_2_e4_stride}};
    vector_setter<e4> entry_invs = {aggregated_entry_invs};
    aggregated_entry_invs_and_multiplicities_arg_impl<1>(
        challenges, witness_cols, setup_cols, stage_2_e4_cols, entry_invs,
        start_col_in_setup, multiplicities_src_cols_start, multiplicities_dst_cols_start,
        num_multiplicities_cols, num_table_rows_tail, log_n, gid);
}

kernel void ab_generic_aggregated_entry_invs_and_multiplicities_arg_kernel(
    const device LookupChallenges &challenges [[buffer(0)]],
    const device bf *witness_ptr [[buffer(1)]],
    constant uint &witness_stride [[buffer(2)]],
    const device bf *setup_ptr [[buffer(3)]],
    constant uint &setup_stride [[buffer(4)]],
    device bf *stage_2_e4_ptr [[buffer(5)]],
    constant uint &stage_2_e4_stride [[buffer(6)]],
    device e4 *aggregated_entry_invs [[buffer(7)]],
    constant uint &start_col_in_setup [[buffer(8)]],
    constant uint &multiplicities_src_cols_start [[buffer(9)]],
    constant uint &multiplicities_dst_cols_start [[buffer(10)]],
    constant uint &num_multiplicities_cols [[buffer(11)]],
    constant uint &num_table_rows_tail [[buffer(12)]],
    constant uint &log_n [[buffer(13)]],
    uint gid [[thread_position_in_grid]]
) {
    matrix_getter<bf> witness_cols = {witness_ptr, witness_stride};
    matrix_getter<bf> setup_cols = {setup_ptr, setup_stride};
    vectorized_e4_matrix_setter stage_2_e4_cols = {{stage_2_e4_ptr, stage_2_e4_stride}};
    vector_setter<e4> entry_invs = {aggregated_entry_invs};
    aggregated_entry_invs_and_multiplicities_arg_impl<NUM_LOOKUP_ARGUMENT_KEY_PARTS>(
        challenges, witness_cols, setup_cols, stage_2_e4_cols, entry_invs,
        start_col_in_setup, multiplicities_src_cols_start, multiplicities_dst_cols_start,
        num_multiplicities_cols, num_table_rows_tail, log_n, gid);
}

kernel void ab_delegation_aux_poly_kernel(
    const device DelegationChallenges &challenges [[buffer(0)]],
    const device DelegationRequestMetadata &request_metadata [[buffer(1)]],
    const device DelegationProcessingMetadata &processing_metadata [[buffer(2)]],
    const device bf *memory_ptr [[buffer(3)]],
    constant uint &memory_stride [[buffer(4)]],
    const device bf *setup_ptr [[buffer(5)]],
    constant uint &setup_stride [[buffer(6)]],
    device bf *stage_2_e4_ptr [[buffer(7)]],
    constant uint &stage_2_e4_stride [[buffer(8)]],
    constant uint &delegation_aux_poly_col [[buffer(9)]],
    constant uint &handle_delegation_requests [[buffer(10)]],
    constant uint &log_n [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint n = 1u << log_n;
    if (gid >= n - 1)
        return;

    matrix_getter<bf> memory_cols = {memory_ptr, memory_stride};
    matrix_getter<bf> setup_cols = {setup_ptr, setup_stride};
    vectorized_e4_matrix_setter stage_2_e4_cols = {{stage_2_e4_ptr, stage_2_e4_stride}};

    stage_2_e4_cols.add_row(gid);
    memory_cols.add_row(gid);
    setup_cols.add_row(gid);

    if (handle_delegation_requests) {
        const bf num = memory_cols.get_at_col(request_metadata.multiplicity_col);

        bf timestamp_low = setup_cols.get_at_col(request_metadata.timestamp_setup_col);
        timestamp_low = bf::add(timestamp_low, request_metadata.in_cycle_write_idx);

        bf timestamp_high = setup_cols.get_at_col(request_metadata.timestamp_setup_col + 1);
        timestamp_high = bf::add(timestamp_high, request_metadata.memory_timestamp_high_from_circuit_idx);

        e4 denom = challenges.gamma;
        denom = e4::add(denom, memory_cols.get_at_col(request_metadata.delegation_type_col));
        denom = e4::add(denom, e4::mul(challenges.linearization_challenges[0], memory_cols.get_at_col(request_metadata.abi_mem_offset_high_col)));
        denom = e4::add(denom, e4::mul(challenges.linearization_challenges[1], timestamp_low));
        denom = e4::add(denom, e4::mul(challenges.linearization_challenges[2], timestamp_high));

        const e4 denom_inv = e4::inv(denom);
        stage_2_e4_cols.set_at_col(delegation_aux_poly_col, e4::mul(num, denom_inv));
    } else {
        const bf num = memory_cols.get_at_col(processing_metadata.multiplicity_col);

        e4 denom = challenges.gamma;
        denom = e4::add(denom, processing_metadata.delegation_type);
        denom = e4::add(denom, e4::mul(challenges.linearization_challenges[0], memory_cols.get_at_col(processing_metadata.abi_mem_offset_high_col)));
        denom = e4::add(denom, e4::mul(challenges.linearization_challenges[1], memory_cols.get_at_col(processing_metadata.write_timestamp_col)));
        denom = e4::add(denom, e4::mul(challenges.linearization_challenges[2], memory_cols.get_at_col(processing_metadata.write_timestamp_col + 1)));

        const e4 denom_inv = e4::inv(denom);
        stage_2_e4_cols.set_at_col(delegation_aux_poly_col, e4::mul(num, denom_inv));
    }
}

kernel void ab_lookup_args_kernel(
    const device RangeCheckArgsLayout &range_check_16_layout [[buffer(0)]],
    const device FlattenedLookupExpressionsLayout &expressions [[buffer(1)]],
    const device FlattenedLookupExpressionsForShuffleRamLayout &expressions_for_shuffle_ram [[buffer(2)]],
    const device LazyInitTeardownLayout &lazy_init_teardown_layout [[buffer(3)]],
    const device bf *setup_ptr [[buffer(4)]],
    constant uint &setup_stride [[buffer(5)]],
    const device bf *witness_ptr [[buffer(6)]],
    constant uint &witness_stride [[buffer(7)]],
    const device bf *memory_ptr [[buffer(8)]],
    constant uint &memory_stride [[buffer(9)]],
    const device e4 *aggregated_entry_invs_for_range_check_16 [[buffer(10)]],
    const device e4 *aggregated_entry_invs_for_timestamp_range_checks [[buffer(11)]],
    const device e4 *aggregated_entry_invs_for_generic_lookups [[buffer(12)]],
    constant uint &generic_args_start [[buffer(13)]],
    constant uint &num_generic_args [[buffer(14)]],
    const device uint *generic_map_ptr [[buffer(15)]],
    constant uint &generic_map_stride [[buffer(16)]],
    device bf *stage_2_bf_ptr [[buffer(17)]],
    constant uint &stage_2_bf_stride [[buffer(18)]],
    device bf *stage_2_e4_ptr [[buffer(19)]],
    constant uint &stage_2_e4_stride [[buffer(20)]],
    constant bf &memory_timestamp_high_from_circuit_idx [[buffer(21)]],
    constant uint &num_stage_2_bf_cols [[buffer(22)]],
    constant uint &num_stage_2_e4_cols [[buffer(23)]],
    constant uint &log_n [[buffer(24)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint n = 1u << log_n;
    if (gid >= n)
        return;

    matrix_setter<bf> stage_2_bf_cols = {stage_2_bf_ptr, stage_2_bf_stride};
    vectorized_e4_matrix_setter stage_2_e4_cols = {{stage_2_e4_ptr, stage_2_e4_stride}};

    stage_2_bf_cols.add_row(gid);
    stage_2_e4_cols.add_row(gid);

    // Zero last row
    if (gid == n - 1) {
        for (uint i = 0; i < num_stage_2_bf_cols; i++)
            stage_2_bf_cols.set_at_col(i, bf::zero());
        for (uint i = 0; i < num_stage_2_e4_cols; i++)
            stage_2_e4_cols.set_at_col(i, e4::zero());
        return;
    }

    matrix_getter<bf> setup_cols = {setup_ptr, setup_stride};
    matrix_getter<bf> witness_cols = {witness_ptr, witness_stride};
    matrix_getter<bf> memory_cols = {memory_ptr, memory_stride};
    matrix_getter<uint> generic_lookups_map = {generic_map_ptr, generic_map_stride};
    vector_getter<e4> rc16_invs = {aggregated_entry_invs_for_range_check_16};
    vector_getter<e4> ts_invs = {aggregated_entry_invs_for_timestamp_range_checks};
    vector_getter<e4> generic_invs = {aggregated_entry_invs_for_generic_lookups};

    setup_cols.add_row(gid);
    witness_cols.add_row(gid);
    memory_cols.add_row(gid);
    generic_lookups_map.add_row(gid);

    // Range check 16 args
    for (uint i = 0; i < range_check_16_layout.num_dst_cols; i++) {
        const uint src = 2 * i + range_check_16_layout.src_cols_start;
        const bf val0 = bf::into_canonical(witness_cols.get_at_col(src));
        const bf val1 = bf::into_canonical(witness_cols.get_at_col(src + 1));
        const auto entry0 = rc16_invs.get(val0.limb);
        const auto entry1 = rc16_invs.get(val1.limb);
        const auto bf_arg = bf::mul(val0, val1);
        const auto e4_arg = e4::add(entry0, entry1);
        stage_2_bf_cols.set_at_col(range_check_16_layout.bf_args_start + i, bf_arg);
        stage_2_e4_cols.set_at_col(range_check_16_layout.e4_args_start + i, e4_arg);
    }

    // Lookup expressions
    {
        uint i = 0, expression_idx = 0, flat_term_idx = 0;
        for (; i < expressions.num_range_check_16_expression_pairs; i++) {
            bf a_and_b[2];
            eval_a_and_b<true>(a_and_b, expressions, expression_idx, flat_term_idx, witness_cols, memory_cols, expressions.range_check_16_constant_terms_are_zero);
            a_and_b[0] = bf::into_canonical(a_and_b[0]);
            a_and_b[1] = bf::into_canonical(a_and_b[1]);
            const e4 entry_a = rc16_invs.get(a_and_b[0].limb);
            const e4 entry_b = rc16_invs.get(a_and_b[1].limb);
            const bf bf_arg = bf::mul(a_and_b[0], a_and_b[1]);
            const e4 e4_arg = e4::add(entry_a, entry_b);
            stage_2_bf_cols.set_at_col(expressions.bf_dst_cols[i], bf_arg);
            stage_2_e4_cols.set_at_col(expressions.e4_dst_cols[i], e4_arg);
        }

        for (; i < expressions.num_range_check_16_expression_pairs + expressions.num_timestamp_expression_pairs; i++) {
            bf a_and_b[2];
            eval_a_and_b<true>(a_and_b, expressions, expression_idx, flat_term_idx, witness_cols, memory_cols, expressions.timestamp_constant_terms_are_zero);
            a_and_b[0] = bf::into_canonical(a_and_b[0]);
            a_and_b[1] = bf::into_canonical(a_and_b[1]);
            const e4 entry_a = ts_invs.get(a_and_b[0].limb);
            const e4 entry_b = ts_invs.get(a_and_b[1].limb);
            const bf bf_arg = bf::mul(a_and_b[0], a_and_b[1]);
            const e4 e4_arg = e4::add(entry_a, entry_b);
            stage_2_bf_cols.set_at_col(expressions.bf_dst_cols[i], bf_arg);
            stage_2_e4_cols.set_at_col(expressions.e4_dst_cols[i], e4_arg);
        }
    }

    // Lookup expressions for shuffle ram
    for (uint i = 0, expression_idx = 0, flat_term_idx = 0; i < expressions_for_shuffle_ram.num_expression_pairs; i++) {
        bf a_and_b[2];
        eval_a_and_b<true>(a_and_b, expressions_for_shuffle_ram, expression_idx, flat_term_idx, setup_cols, witness_cols, memory_cols);
        a_and_b[1] = bf::sub(a_and_b[1], memory_timestamp_high_from_circuit_idx);
        a_and_b[0] = bf::into_canonical(a_and_b[0]);
        a_and_b[1] = bf::into_canonical(a_and_b[1]);
        const e4 entry_a = ts_invs.get(a_and_b[0].limb);
        const e4 entry_b = ts_invs.get(a_and_b[1].limb);
        const bf bf_arg = bf::mul(a_and_b[0], a_and_b[1]);
        const e4 e4_arg = e4::add(entry_a, entry_b);
        stage_2_bf_cols.set_at_col(expressions_for_shuffle_ram.bf_dst_cols[i], bf_arg);
        stage_2_e4_cols.set_at_col(expressions_for_shuffle_ram.e4_dst_cols[i], e4_arg);
    }

    // 32-bit lazy init address cols
    if (lazy_init_teardown_layout.process_shuffle_ram_init) {
        const bf val0 = bf::into_canonical(memory_cols.get_at_col(lazy_init_teardown_layout.init_address_start));
        const bf val1 = bf::into_canonical(memory_cols.get_at_col(lazy_init_teardown_layout.init_address_start + 1));
        const auto entry0 = rc16_invs.get(val0.limb);
        const auto entry1 = rc16_invs.get(val1.limb);
        const auto bf_arg = bf::mul(val0, val1);
        const auto e4_arg = e4::add(entry0, entry1);
        stage_2_bf_cols.set_at_col(lazy_init_teardown_layout.bf_arg_col, bf_arg);
        stage_2_e4_cols.set_at_col(lazy_init_teardown_layout.e4_arg_col, e4_arg);
    }

    // Width-3 generic args with fixed table ids
    for (uint i = 0; i < num_generic_args; i++) {
        const uint absolute_row_index = generic_lookups_map.get_at_col(i);
        const e4 aggregated_entry_inv = generic_invs.get(absolute_row_index);
        stage_2_e4_cols.set_at_col(generic_args_start + i, aggregated_entry_inv);
    }
}

kernel void ab_shuffle_ram_memory_args_kernel(
    const device MemoryChallenges &challenges [[buffer(0)]],
    const device ShuffleRamAccesses &shuffle_ram_accesses [[buffer(1)]],
    const device bf *setup_ptr [[buffer(2)]],
    constant uint &setup_stride [[buffer(3)]],
    const device bf *memory_ptr [[buffer(4)]],
    constant uint &memory_stride [[buffer(5)]],
    device bf *stage_2_e4_ptr [[buffer(6)]],
    constant uint &stage_2_e4_stride [[buffer(7)]],
    const device LazyInitTeardownLayout &lazy_init_teardown_layout [[buffer(8)]],
    constant bf &memory_timestamp_high_from_circuit_idx [[buffer(9)]],
    constant uint &memory_args_start [[buffer(10)]],
    constant uint &log_n [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint n = 1u << log_n;
    if (gid >= n - 1)
        return;

    vectorized_e4_matrix_setter stage_2_e4_cols = {{stage_2_e4_ptr, stage_2_e4_stride}};
    matrix_getter<bf> setup_cols = {setup_ptr, setup_stride};
    matrix_getter<bf> memory_cols = {memory_ptr, memory_stride};

    stage_2_e4_cols.add_row(gid);
    setup_cols.add_row(gid);
    memory_cols.add_row(gid);

    // Shuffle ram init
    e4 numerator = challenges.gamma;
    const bf address_low = memory_cols.get_at_col(lazy_init_teardown_layout.init_address_start);
    numerator = e4::add(numerator, e4::mul(challenges.address_low_challenge, address_low));
    const bf address_high = memory_cols.get_at_col(lazy_init_teardown_layout.init_address_start + 1);
    numerator = e4::add(numerator, e4::mul(challenges.address_high_challenge, address_high));

    e4 denom = numerator;
    const bf value_low = memory_cols.get_at_col(lazy_init_teardown_layout.teardown_value_start);
    denom = e4::add(denom, e4::mul(challenges.value_low_challenge, value_low));
    const bf value_high = memory_cols.get_at_col(lazy_init_teardown_layout.teardown_value_start + 1);
    denom = e4::add(denom, e4::mul(challenges.value_high_challenge, value_high));
    const bf timestamp_low = memory_cols.get_at_col(lazy_init_teardown_layout.teardown_timestamp_start);
    denom = e4::add(denom, e4::mul(challenges.timestamp_low_challenge, timestamp_low));
    const bf timestamp_high = memory_cols.get_at_col(lazy_init_teardown_layout.teardown_timestamp_start + 1);
    denom = e4::add(denom, e4::mul(challenges.timestamp_high_challenge, timestamp_high));

    e4 num_over_denom_acc = numerator;
    e4 denom_inv = e4::inv(denom);
    num_over_denom_acc = e4::mul(num_over_denom_acc, denom_inv);
    stage_2_e4_cols.set_at_col(memory_args_start, num_over_denom_acc);

    // Shuffle ram accesses
    const bf write_timestamp_in_setup_low = setup_cols.get_at_col(shuffle_ram_accesses.write_timestamp_in_setup_start);
    const bf write_timestamp_in_setup_high = setup_cols.get_at_col(shuffle_ram_accesses.write_timestamp_in_setup_start + 1);
    for (uint i = 0; i < shuffle_ram_accesses.num_accesses; i++) {
        const auto access = shuffle_ram_accesses.accesses[i];

        e4 num_i = challenges.gamma;
        const bf addr_low = memory_cols.get_at_col(access.address_start);
        num_i = e4::add(num_i, e4::mul(challenges.address_low_challenge, addr_low));

        if (access.is_register_only) {
            num_i = e4::add(num_i, bf::one());
        } else {
            const bf addr_high = memory_cols.get_at_col(access.address_start + 1);
            num_i = e4::add(num_i, e4::mul(challenges.address_high_challenge, addr_high));
            num_i = e4::add(num_i, memory_cols.get_at_col(access.maybe_is_register_start));
        }

        e4 denom_i;

        if (access.is_write) {
            denom_i = num_i;
            const bf rv_low = memory_cols.get_at_col(access.read_value_start);
            denom_i = e4::add(denom_i, e4::mul(challenges.value_low_challenge, rv_low));
            const bf rv_high = memory_cols.get_at_col(access.read_value_start + 1);
            denom_i = e4::add(denom_i, e4::mul(challenges.value_high_challenge, rv_high));

            const bf wv_low = memory_cols.get_at_col(access.maybe_write_value_start);
            num_i = e4::add(num_i, e4::mul(challenges.value_low_challenge, wv_low));
            const bf wv_high = memory_cols.get_at_col(access.maybe_write_value_start + 1);
            num_i = e4::add(num_i, e4::mul(challenges.value_high_challenge, wv_high));
        } else {
            const bf v_low = memory_cols.get_at_col(access.read_value_start);
            num_i = e4::add(num_i, e4::mul(challenges.value_low_challenge, v_low));
            const bf v_high = memory_cols.get_at_col(access.read_value_start + 1);
            num_i = e4::add(num_i, e4::mul(challenges.value_high_challenge, v_high));
            denom_i = num_i;
        }

        const bf rt_low = memory_cols.get_at_col(access.read_timestamp_start);
        denom_i = e4::add(denom_i, e4::mul(challenges.timestamp_low_challenge, rt_low));
        const bf rt_high = memory_cols.get_at_col(access.read_timestamp_start + 1);
        denom_i = e4::add(denom_i, e4::mul(challenges.timestamp_high_challenge, rt_high));

        const bf access_index = bf{i};
        const bf wt_low = bf::add(write_timestamp_in_setup_low, access_index);
        num_i = e4::add(num_i, e4::mul(challenges.timestamp_low_challenge, wt_low));
        const bf wt_high = bf::add(write_timestamp_in_setup_high, memory_timestamp_high_from_circuit_idx);
        num_i = e4::add(num_i, e4::mul(challenges.timestamp_high_challenge, wt_high));

        num_over_denom_acc = e4::mul(num_over_denom_acc, num_i);
        e4 di = e4::inv(denom_i);
        num_over_denom_acc = e4::mul(num_over_denom_acc, di);
        stage_2_e4_cols.set_at_col(memory_args_start + 1 + i, num_over_denom_acc);
    }
}

kernel void ab_batched_ram_memory_args_kernel(
    const device MemoryChallenges &challenges [[buffer(0)]],
    const device BatchedRamAccesses &batched_ram_accesses [[buffer(1)]],
    const device bf *memory_ptr [[buffer(2)]],
    constant uint &memory_stride [[buffer(3)]],
    device bf *stage_2_e4_ptr [[buffer(4)]],
    constant uint &stage_2_e4_stride [[buffer(5)]],
    constant uint &memory_args_start [[buffer(6)]],
    constant uint &log_n [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint n = 1u << log_n;
    if (gid >= n - 1)
        return;

    vectorized_e4_matrix_setter stage_2_e4_cols = {{stage_2_e4_ptr, stage_2_e4_stride}};
    matrix_getter<bf> memory_cols = {memory_ptr, memory_stride};

    stage_2_e4_cols.add_row(gid);
    memory_cols.add_row(gid);

    const bf addr_high = memory_cols.get_at_col(batched_ram_accesses.abi_mem_offset_high_col);
    const e4 addr_high_contribution = e4::mul(addr_high, challenges.address_high_challenge);
    const bf wt_low = memory_cols.get_at_col(batched_ram_accesses.write_timestamp_col);
    const e4 wt_low_contribution = e4::mul(wt_low, challenges.timestamp_low_challenge);
    const bf wt_high = memory_cols.get_at_col(batched_ram_accesses.write_timestamp_col + 1);
    const e4 wt_high_contribution = e4::mul(wt_high, challenges.timestamp_high_challenge);
    const e4 wt_contribution = e4::add(wt_low_contribution, wt_high_contribution);

    e4 num_over_denom_acc;
    for (uint i = 0; i < batched_ram_accesses.num_accesses; i++) {
        const auto access = batched_ram_accesses.accesses[i];
        e4 numerator = e4::add(access.gamma_plus_address_low_contribution, addr_high_contribution);

        e4 denom;
        if (access.is_write) {
            denom = numerator;
            const bf rv_low = memory_cols.get_at_col(access.read_value_col);
            denom = e4::add(denom, e4::mul(challenges.value_low_challenge, rv_low));
            const bf rv_high = memory_cols.get_at_col(access.read_value_col + 1);
            denom = e4::add(denom, e4::mul(challenges.value_high_challenge, rv_high));

            const bf wv_low = memory_cols.get_at_col(access.maybe_write_value_col);
            numerator = e4::add(numerator, e4::mul(challenges.value_low_challenge, wv_low));
            const bf wv_high = memory_cols.get_at_col(access.maybe_write_value_col + 1);
            numerator = e4::add(numerator, e4::mul(challenges.value_high_challenge, wv_high));
        } else {
            const bf v_low = memory_cols.get_at_col(access.read_value_col);
            numerator = e4::add(numerator, e4::mul(challenges.value_low_challenge, v_low));
            const bf v_high = memory_cols.get_at_col(access.read_value_col + 1);
            numerator = e4::add(numerator, e4::mul(challenges.value_high_challenge, v_high));
            denom = numerator;
        }

        numerator = e4::add(numerator, wt_contribution);
        const bf rt_low = memory_cols.get_at_col(access.read_timestamp_col);
        denom = e4::add(denom, e4::mul(challenges.timestamp_low_challenge, rt_low));
        const bf rt_high = memory_cols.get_at_col(access.read_timestamp_col + 1);
        denom = e4::add(denom, e4::mul(challenges.timestamp_high_challenge, rt_high));

        if (i == 0)
            num_over_denom_acc = numerator;
        else
            num_over_denom_acc = e4::mul(num_over_denom_acc, numerator);
        e4 di = e4::inv(denom);
        num_over_denom_acc = e4::mul(num_over_denom_acc, di);
        stage_2_e4_cols.set_at_col(memory_args_start + i, num_over_denom_acc);
    }
}

kernel void ab_register_and_indirect_memory_args_kernel(
    const device MemoryChallenges &challenges [[buffer(0)]],
    const device RegisterAndIndirectAccesses &accesses [[buffer(1)]],
    const device bf *memory_ptr [[buffer(2)]],
    constant uint &memory_stride [[buffer(3)]],
    device bf *stage_2_e4_ptr [[buffer(4)]],
    constant uint &stage_2_e4_stride [[buffer(5)]],
    constant uint &memory_args_start [[buffer(6)]],
    constant uint &log_n [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint n = 1u << log_n;
    if (gid >= n - 1)
        return;

    vectorized_e4_matrix_setter stage_2_e4_cols = {{stage_2_e4_ptr, stage_2_e4_stride}};
    matrix_getter<bf> memory_cols = {memory_ptr, memory_stride};

    stage_2_e4_cols.add_row(gid);
    stage_2_e4_cols.add_col(memory_args_start);
    memory_cols.add_row(gid);

    const bf wt_low = memory_cols.get_at_col(accesses.write_timestamp_col);
    const e4 wt_low_contribution = e4::mul(wt_low, challenges.timestamp_low_challenge);
    const bf wt_high = memory_cols.get_at_col(accesses.write_timestamp_col + 1);
    const e4 wt_high_contribution = e4::mul(wt_high, challenges.timestamp_high_challenge);
    const e4 wt_contribution = e4::add(wt_low_contribution, wt_high_contribution);

    e4 num_over_denom_acc;
    uint flat_indirect_idx = 0;

    for (uint i = 0; i < accesses.num_register_accesses; i++) {
        uint base_low;
        uint base_high;
        {
            const auto reg_access = accesses.register_accesses[i];
            e4 numerator = reg_access.gamma_plus_one_plus_address_low_contribution;
            e4 denom;

            if (reg_access.is_write) {
                denom = numerator;
                const bf rv_low = memory_cols.get_at_col(reg_access.read_value_col);
                denom = e4::add(denom, e4::mul(challenges.value_low_challenge, rv_low));
                base_low = bf::into_canonical(rv_low).limb;
                const bf rv_high = memory_cols.get_at_col(reg_access.read_value_col + 1);
                denom = e4::add(denom, e4::mul(challenges.value_high_challenge, rv_high));
                base_high = bf::into_canonical(rv_high).limb;

                const bf wv_low = memory_cols.get_at_col(reg_access.maybe_write_value_col);
                numerator = e4::add(numerator, e4::mul(challenges.value_low_challenge, wv_low));
                const bf wv_high = memory_cols.get_at_col(reg_access.maybe_write_value_col + 1);
                numerator = e4::add(numerator, e4::mul(challenges.value_high_challenge, wv_high));
            } else {
                const bf v_low = memory_cols.get_at_col(reg_access.read_value_col);
                numerator = e4::add(numerator, e4::mul(challenges.value_low_challenge, v_low));
                base_low = bf::into_canonical(v_low).limb;
                const bf v_high = memory_cols.get_at_col(reg_access.read_value_col + 1);
                numerator = e4::add(numerator, e4::mul(challenges.value_high_challenge, v_high));
                base_high = bf::into_canonical(v_high).limb;
                denom = numerator;
            }

            numerator = e4::add(numerator, wt_contribution);
            const bf rt_low = memory_cols.get_at_col(reg_access.read_timestamp_col);
            denom = e4::add(denom, e4::mul(challenges.timestamp_low_challenge, rt_low));
            const bf rt_high = memory_cols.get_at_col(reg_access.read_timestamp_col + 1);
            denom = e4::add(denom, e4::mul(challenges.timestamp_high_challenge, rt_high));

            if (i == 0)
                num_over_denom_acc = numerator;
            else
                num_over_denom_acc = e4::mul(num_over_denom_acc, numerator);
            e4 di = e4::inv(denom);
            num_over_denom_acc = e4::mul(num_over_denom_acc, di);
            stage_2_e4_cols.set(num_over_denom_acc);
            stage_2_e4_cols.add_col(1);
        }

        const uint lim = flat_indirect_idx + accesses.indirect_accesses_per_register_access[i];
        for (; flat_indirect_idx < lim; flat_indirect_idx++) {
            const auto indirect_access = accesses.indirect_accesses[flat_indirect_idx];

            const uint address = base_low + indirect_access.offset;
            const uint of = address >> 16;
            const bf addr_low = bf{address & 0x0000ffff};
            const bf addr_high = bf{base_high + of};

            e4 numerator = challenges.gamma;
            numerator = e4::add(numerator, e4::mul(challenges.address_low_challenge, addr_low));
            numerator = e4::add(numerator, e4::mul(challenges.address_high_challenge, addr_high));

            e4 denom;
            if (indirect_access.is_write) {
                denom = numerator;
                const bf rv_low = memory_cols.get_at_col(indirect_access.read_value_col);
                denom = e4::add(denom, e4::mul(challenges.value_low_challenge, rv_low));
                const bf rv_high = memory_cols.get_at_col(indirect_access.read_value_col + 1);
                denom = e4::add(denom, e4::mul(challenges.value_high_challenge, rv_high));

                const bf wv_low = memory_cols.get_at_col(indirect_access.maybe_write_value_col);
                numerator = e4::add(numerator, e4::mul(challenges.value_low_challenge, wv_low));
                const bf wv_high = memory_cols.get_at_col(indirect_access.maybe_write_value_col + 1);
                numerator = e4::add(numerator, e4::mul(challenges.value_high_challenge, wv_high));
            } else {
                const bf v_low = memory_cols.get_at_col(indirect_access.read_value_col);
                numerator = e4::add(numerator, e4::mul(challenges.value_low_challenge, v_low));
                const bf v_high = memory_cols.get_at_col(indirect_access.read_value_col + 1);
                numerator = e4::add(numerator, e4::mul(challenges.value_high_challenge, v_high));
                denom = numerator;
            }

            numerator = e4::add(numerator, wt_contribution);
            const bf rt_low = memory_cols.get_at_col(indirect_access.read_timestamp_col);
            denom = e4::add(denom, e4::mul(challenges.timestamp_low_challenge, rt_low));
            const bf rt_high = memory_cols.get_at_col(indirect_access.read_timestamp_col + 1);
            denom = e4::add(denom, e4::mul(challenges.timestamp_high_challenge, rt_high));

            num_over_denom_acc = e4::mul(num_over_denom_acc, numerator);
            e4 di = e4::inv(denom);
            num_over_denom_acc = e4::mul(num_over_denom_acc, di);
            stage_2_e4_cols.set(num_over_denom_acc);
            stage_2_e4_cols.add_col(1);
        }
    }
}
