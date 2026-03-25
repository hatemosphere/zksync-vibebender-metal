#pragma once

#include "field.metal"
#include "memory.metal"

// Metal port of gpu_prover/native/arg_utils.cuh
// Provides shared structures and utilities for stage 2 and stage 3 kernels.

namespace airbender {
namespace arg_utils {

using bf = field::base_field;
using e2 = field::ext2_field;
using e4 = field::ext4_field;

constant constexpr uint NUM_DELEGATION_ARGUMENT_KEY_PARTS = 4;

struct DelegationChallenges {
    e4 linearization_challenges[NUM_DELEGATION_ARGUMENT_KEY_PARTS - 1];
    e4 gamma;
};

struct DelegationRequestMetadata {
    uint multiplicity_col;
    uint timestamp_setup_col;
    bf memory_timestamp_high_from_circuit_idx;
    uint delegation_type_col;
    uint abi_mem_offset_high_col;
    bf in_cycle_write_idx;
};

struct DelegationProcessingMetadata {
    uint multiplicity_col;
    bf delegation_type;
    uint abi_mem_offset_high_col;
    uint write_timestamp_col;
};

constant constexpr uint NUM_LOOKUP_ARGUMENT_KEY_PARTS = 4;

struct LookupChallenges {
    e4 linearization_challenges[NUM_LOOKUP_ARGUMENT_KEY_PARTS - 1];
    e4 gamma;
};

struct RangeCheckArgsLayout {
    uint num_dst_cols;
    uint src_cols_start;
    uint bf_args_start;
    uint e4_args_start;
};

struct MemoryChallenges {
    e4 address_low_challenge;
    e4 address_high_challenge;
    e4 timestamp_low_challenge;
    e4 timestamp_high_challenge;
    e4 value_low_challenge;
    e4 value_high_challenge;
    e4 gamma;
};

constant constexpr uint MAX_EXPRESSION_PAIRS = 84;
constant constexpr uint MAX_EXPRESSIONS = 2 * MAX_EXPRESSION_PAIRS;
constant constexpr uint MAX_TERMS_PER_EXPRESSION = 4;
constant constexpr uint MAX_EXPRESSION_TERMS = MAX_TERMS_PER_EXPRESSION * MAX_EXPRESSIONS;

struct FlattenedLookupExpressionsLayout {
    uint coeffs[MAX_EXPRESSION_TERMS];
    uint16_t col_idxs[MAX_EXPRESSION_TERMS];
    bf constant_terms[MAX_EXPRESSIONS];
    uint8_t num_terms_per_expression[MAX_EXPRESSIONS];
    uint8_t bf_dst_cols[MAX_EXPRESSION_PAIRS];
    uint8_t e4_dst_cols[MAX_EXPRESSION_PAIRS];
    uint num_range_check_16_expression_pairs;
    uint num_timestamp_expression_pairs;
    bool range_check_16_constant_terms_are_zero;
    bool timestamp_constant_terms_are_zero;
};

constant constexpr uint MAX_EXPRESSION_PAIRS_FOR_SHUFFLE_RAM = 4;
constant constexpr uint MAX_EXPRESSIONS_FOR_SHUFFLE_RAM = 2 * MAX_EXPRESSION_PAIRS_FOR_SHUFFLE_RAM;
constant constexpr uint MAX_EXPRESSION_TERMS_FOR_SHUFFLE_RAM = MAX_TERMS_PER_EXPRESSION * MAX_EXPRESSIONS_FOR_SHUFFLE_RAM;

struct FlattenedLookupExpressionsForShuffleRamLayout {
    uint coeffs[MAX_EXPRESSION_TERMS_FOR_SHUFFLE_RAM];
    uint16_t col_idxs[MAX_EXPRESSION_TERMS_FOR_SHUFFLE_RAM];
    bf constant_terms[MAX_EXPRESSIONS_FOR_SHUFFLE_RAM];
    uint8_t num_terms_per_expression[MAX_EXPRESSIONS_FOR_SHUFFLE_RAM];
    uint8_t bf_dst_cols[MAX_EXPRESSION_PAIRS_FOR_SHUFFLE_RAM];
    uint8_t e4_dst_cols[MAX_EXPRESSION_PAIRS_FOR_SHUFFLE_RAM];
    uint num_expression_pairs;
};

// Column type flags encoded in the top 2 bits of u16 col indexes
constant constexpr uint COL_TYPE_MASK = 3u << 14;
constant constexpr uint COL_IDX_MASK = (1u << 14) - 1;
constant constexpr uint COL_TYPE_WITNESS = 0;
constant constexpr uint COL_TYPE_MEMORY = 1u << 14;
constant constexpr uint COL_TYPE_SETUP = 1u << 15;

template <typename T, typename U>
DEVICE_FORCEINLINE bf get_witness_or_memory(const uint col_idx, const thread T &witness_cols, const thread U &memory_cols) {
    return (col_idx & COL_TYPE_MEMORY) ? memory_cols.get_at_col(col_idx & COL_IDX_MASK) : witness_cols.get_at_col(col_idx);
}

template <typename T, typename U, typename V>
DEVICE_FORCEINLINE bf get_witness_memory_or_setup(const uint col_idx, const thread T &witness_cols, const thread U &memory_cols, const thread V &setup_cols) {
    const uint col_type = col_idx & COL_TYPE_MASK;
    bf val;
    switch (col_type) {
    case COL_TYPE_WITNESS:
        val = witness_cols.get_at_col(col_idx & COL_IDX_MASK);
        break;
    case COL_TYPE_MEMORY:
        val = memory_cols.get_at_col(col_idx & COL_IDX_MASK);
        break;
    case COL_TYPE_SETUP:
        val = setup_cols.get_at_col(col_idx & COL_IDX_MASK);
        break;
    default:
        break;
    }
    return val;
}

DEVICE_FORCEINLINE void apply_coeff(const uint coeff, thread bf &val) {
    switch (coeff) {
    case 1:
        break;
    case bf::MINUS_ONE:
        val = bf::neg(val);
        break;
    default:
        val = bf::mul(val, bf{coeff});
    }
}

template <bool APPLY_CONSTANT_TERMS, typename T>
DEVICE_FORCEINLINE void eval_a_and_b(thread bf a_and_b[2], const device FlattenedLookupExpressionsLayout &expressions,
                                      thread uint &expression_idx, thread uint &flat_term_idx,
                                      const thread T &witness_cols, const thread T &memory_cols, const bool constant_terms_are_zero) {
    for (int j = 0; j < 2; j++, expression_idx++) {
        const uint lim = flat_term_idx + expressions.num_terms_per_expression[expression_idx];
        a_and_b[j] = get_witness_or_memory(expressions.col_idxs[flat_term_idx], witness_cols, memory_cols);
        apply_coeff(expressions.coeffs[flat_term_idx], a_and_b[j]);
        flat_term_idx++;
        for (; flat_term_idx < lim; flat_term_idx++) {
            bf val = get_witness_or_memory(expressions.col_idxs[flat_term_idx], witness_cols, memory_cols);
            apply_coeff(expressions.coeffs[flat_term_idx], val);
            a_and_b[j] = bf::add(a_and_b[j], val);
        }
        if (APPLY_CONSTANT_TERMS && !constant_terms_are_zero) {
            a_and_b[j] = bf::add(a_and_b[j], expressions.constant_terms[expression_idx]);
        }
    }
}

template <bool APPLY_CONSTANT_TERMS, typename T, typename U>
DEVICE_FORCEINLINE void eval_a_and_b(thread bf a_and_b[2], const device FlattenedLookupExpressionsForShuffleRamLayout &expressions,
                                      thread uint &expression_idx, thread uint &flat_term_idx,
                                      const thread T &setup_cols, const thread U &witness_cols, const thread U &memory_cols) {
    for (int j = 0; j < 2; j++, expression_idx++) {
        const uint lim = flat_term_idx + expressions.num_terms_per_expression[expression_idx];
        a_and_b[j] = get_witness_memory_or_setup(expressions.col_idxs[flat_term_idx], witness_cols, memory_cols, setup_cols);
        apply_coeff(expressions.coeffs[flat_term_idx], a_and_b[j]);
        flat_term_idx++;
        for (; flat_term_idx < lim; flat_term_idx++) {
            const uint col = expressions.col_idxs[flat_term_idx];
            bf val = get_witness_memory_or_setup(col, witness_cols, memory_cols, setup_cols);
            apply_coeff(expressions.coeffs[flat_term_idx], val);
            a_and_b[j] = bf::add(a_and_b[j], val);
        }
        if (APPLY_CONSTANT_TERMS) {
            a_and_b[j] = bf::add(a_and_b[j], expressions.constant_terms[expression_idx]);
        }
    }
}

struct LazyInitTeardownLayout {
    uint init_address_start;
    uint teardown_value_start;
    uint teardown_timestamp_start;
    uint init_address_aux_low;
    uint init_address_aux_high;
    uint init_address_intermediate_borrow;
    uint init_address_final_borrow;
    uint bf_arg_col;
    uint e4_arg_col;
    bool process_shuffle_ram_init;
};

constant constexpr uint MAX_SHUFFLE_RAM_ACCESSES = 3;

struct ShuffleRamAccess {
    uint address_start;
    uint read_timestamp_start;
    uint read_value_start;
    uint maybe_write_value_start;
    uint maybe_is_register_start;
    bool is_write;
    bool is_register_only;
};

struct ShuffleRamAccesses {
    ShuffleRamAccess accesses[MAX_SHUFFLE_RAM_ACCESSES];
    uint num_accesses;
    uint write_timestamp_in_setup_start;
};

constant constexpr uint MAX_BATCHED_RAM_ACCESSES = 36;

struct BatchedRamAccess {
    e4 gamma_plus_address_low_contribution;
    uint read_timestamp_col;
    uint read_value_col;
    uint maybe_write_value_col;
    bool is_write;
};

struct BatchedRamAccesses {
    BatchedRamAccess accesses[MAX_BATCHED_RAM_ACCESSES];
    uint num_accesses;
    uint write_timestamp_col;
    uint abi_mem_offset_high_col;
};

struct RegisterAccess {
    e4 gamma_plus_one_plus_address_low_contribution;
    uint read_timestamp_col;
    uint read_value_col;
    uint maybe_write_value_col;
    bool is_write;
};

struct IndirectAccess {
    uint offset;
    uint read_timestamp_col;
    uint read_value_col;
    uint maybe_write_value_col;
    uint address_derivation_carry_bit_col;
    uint address_derivation_carry_bit_num_elements;
    bool is_write;
};

constant constexpr uint MAX_REGISTER_ACCESSES = 4;
constant constexpr uint MAX_INDIRECT_ACCESSES = 40;

struct RegisterAndIndirectAccesses {
    RegisterAccess register_accesses[MAX_REGISTER_ACCESSES];
    IndirectAccess indirect_accesses[MAX_INDIRECT_ACCESSES];
    uint indirect_accesses_per_register_access[MAX_REGISTER_ACCESSES];
    uint num_register_accesses;
    uint write_timestamp_col;
};

} // namespace arg_utils
} // namespace airbender
