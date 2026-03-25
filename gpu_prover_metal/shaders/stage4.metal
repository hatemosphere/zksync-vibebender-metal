#include "context.metal"
#include "ops_complex.metal"
#include "vectorized.metal"

// Metal port of gpu_prover/native/stage4.cu
// Stage 4: DEEP quotient and FRI folding kernels.

using namespace airbender::field;
using namespace airbender::memory;
using namespace airbender::vectorized;

using bf = base_field;
using e2 = ext2_field;
using e4 = ext4_field;

constant constexpr uint MAX_WITNESS_COLS = 672;
constant constexpr uint DOES_NOT_NEED_Z_OMEGA = 0xFFFFFFFF;
constant constexpr uint MAX_NON_WITNESS_TERMS_AT_Z_OMEGA = 3;

template <typename T, int INV_BATCH>
DEVICE_FORCEINLINE void batch_inv_registers_stage4(const thread T* inputs, thread T* fwd_scan_and_outputs, int runtime_batch_size) {
    T running_prod = T::one();
    #pragma unroll
    for (int i = 0; i < INV_BATCH; i++)
        if (i < runtime_batch_size) {
            fwd_scan_and_outputs[i] = running_prod;
            running_prod = T::mul(running_prod, inputs[i]);
        }

    T inv = T::inv(running_prod);

    #pragma unroll
    for (int i = INV_BATCH - 1; i >= 0; i--) {
        if (i < runtime_batch_size) {
            const auto input = inputs[i];
            fwd_scan_and_outputs[i] = T::mul(fwd_scan_and_outputs[i], inv);
            if (i > 0)
                inv = T::mul(inv, input);
        }
    }
}

kernel void ab_deep_denom_at_z_kernel(
    device e4 *denom_at_z [[buffer(0)]],
    constant uint &denom_stride [[buffer(1)]],
    const device e4 *z_ref [[buffer(2)]],
    constant uint &log_n [[buffer(3)]],
    constant uint &bit_reversed_flag [[buffer(4)]],
    // Powers data for get_power_of_w
    const device e2 *powers_fine [[buffer(5)]],
    constant uint &powers_fine_mask [[buffer(6)]],
    constant uint &powers_fine_log_count [[buffer(7)]],
    const device e2 *powers_coarser [[buffer(8)]],
    constant uint &powers_coarser_mask [[buffer(9)]],
    constant uint &powers_coarser_log_count [[buffer(10)]],
    const device e2 *powers_coarsest [[buffer(11)]],
    constant uint &powers_coarsest_mask [[buffer(12)]],
    uint gid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]
) {
    (void)denom_stride;
    constexpr int INV_BATCH = 4;

    const uint n = 1u << log_n;
    if (gid >= n)
        return;

    const bool bit_reversed = bit_reversed_flag != 0;

    // Build powers_data for get_power_of_w
    powers_layer_data fine_data;
    fine_data.values = powers_fine;
    fine_data.mask = powers_fine_mask;
    fine_data.log_count = powers_fine_log_count;

    powers_layer_data coarser_data;
    coarser_data.values = powers_coarser;
    coarser_data.mask = powers_coarser_mask;
    coarser_data.log_count = powers_coarser_log_count;

    powers_layer_data coarsest_data;
    coarsest_data.values = powers_coarsest;
    coarsest_data.mask = powers_coarsest_mask;
    coarsest_data.log_count = 0;

    powers_data_3_layer powers = {fine_data, coarser_data, coarsest_data};

    e4 per_elem_factor_invs[INV_BATCH];

    const e4 z = *z_ref;
    int runtime_batch_size = 0;
    const uint log_shift = CIRCLE_GROUP_LOG_ORDER - log_n;

    #pragma unroll
    for (int i = 0; i < INV_BATCH; i++) {
        uint g = gid + i * grid_size;
        if (g < n) {
            const uint k = (bit_reversed ? reverse_bits(g) >> (32 - log_n) : g) << log_shift;
            const auto x = get_power_of_w(powers, k, false);
            per_elem_factor_invs[i] = e4::sub(x, z);
            runtime_batch_size++;
        }
    }

    e4 per_elem_factors[INV_BATCH];
    batch_inv_registers_stage4<e4, INV_BATCH>(per_elem_factor_invs, per_elem_factors, runtime_batch_size);

    #pragma unroll
    for (int i = 0; i < INV_BATCH; i++) {
        uint g = gid + i * grid_size;
        if (g < n)
            denom_at_z[g] = per_elem_factors[i];
    }
}

struct ColIdxsToChallengeIdxsMap {
    uint map[MAX_WITNESS_COLS];
};

struct NonWitnessChallengesAtZOmega {
    e4 challenges[MAX_NON_WITNESS_TERMS_AT_Z_OMEGA];
};

struct ChallengesTimesEvals {
    e4 at_z_sum_neg;
    e4 at_z_omega_sum_neg;
};

kernel void ab_deep_quotient_kernel(
    const device bf *setup_ptr [[buffer(0)]],
    constant uint &setup_stride [[buffer(1)]],
    const device bf *witness_ptr [[buffer(2)]],
    constant uint &witness_stride [[buffer(3)]],
    const device bf *memory_ptr [[buffer(4)]],
    constant uint &memory_stride [[buffer(5)]],
    const device bf *stage_2_bf_ptr [[buffer(6)]],
    constant uint &stage_2_bf_stride [[buffer(7)]],
    const device bf *stage_2_e4_ptr [[buffer(8)]],
    constant uint &stage_2_e4_stride [[buffer(9)]],
    const device bf *composition_ptr [[buffer(10)]],
    constant uint &composition_stride [[buffer(11)]],
    const device e4 *denom_at_z_ptr [[buffer(12)]],
    const device e4 *witness_challenges_at_z_ptr [[buffer(13)]],
    const device e4 *witness_challenges_at_z_omega_ptr [[buffer(14)]],
    const device ColIdxsToChallengeIdxsMap &witness_cols_map [[buffer(15)]],
    const device e4 *non_witness_challenges_at_z_ptr [[buffer(16)]],
    const device NonWitnessChallengesAtZOmega &non_witness_challenges_at_z_omega [[buffer(17)]],
    const device ChallengesTimesEvals &challenges_times_evals [[buffer(18)]],
    device bf *quotient_ptr [[buffer(19)]],
    constant uint &quotient_stride [[buffer(20)]],
    constant uint &num_setup_cols [[buffer(21)]],
    constant uint &num_witness_cols [[buffer(22)]],
    constant uint &num_memory_cols [[buffer(23)]],
    constant uint &num_stage_2_bf_cols [[buffer(24)]],
    constant uint &num_stage_2_e4_cols [[buffer(25)]],
    constant uint &process_shuffle_ram_init_flag [[buffer(26)]],
    constant uint &memory_lazy_init_addresses_cols_start [[buffer(27)]],
    constant uint &log_n [[buffer(28)]],
    constant uint &bit_reversed_flag [[buffer(29)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint n = 1u << log_n;
    if (gid >= n)
        return;

    const bool bit_reversed = bit_reversed_flag != 0;
    const bool process_shuffle_ram_init = process_shuffle_ram_init_flag != 0;

    matrix_getter<bf> setup_cols = {setup_ptr, setup_stride};
    matrix_getter<bf> witness_cols = {witness_ptr, witness_stride};
    matrix_getter<bf> memory_cols = {memory_ptr, memory_stride};
    matrix_getter<bf> stage_2_bf_cols = {stage_2_bf_ptr, stage_2_bf_stride};
    vectorized_e4_matrix_getter stage_2_e4_cols = {{stage_2_e4_ptr, stage_2_e4_stride}};
    vectorized_e4_matrix_getter composition_col = {{composition_ptr, composition_stride}};
    vector_getter<e4> denom_at_z = {denom_at_z_ptr};
    vector_getter<e4> witness_challenges_at_z = {witness_challenges_at_z_ptr};
    vector_getter<e4> witness_challenges_at_z_omega = {witness_challenges_at_z_omega_ptr};
    vector_getter<e4> non_witness_challenges_at_z = {non_witness_challenges_at_z_ptr};
    vectorized_e4_matrix_setter quotient = {{quotient_ptr, quotient_stride}};

    setup_cols.add_row(gid);
    witness_cols.add_row(gid);
    memory_cols.add_row(gid);
    stage_2_bf_cols.add_row(gid);
    stage_2_e4_cols.add_row(gid);
    composition_col.add_row(gid);
    quotient.add_row(gid);

    e4 acc_z = e4::zero();
    e4 acc_z_omega = e4::zero();

    // Witness terms at z and z * omega
    for (uint i = 0; i < num_witness_cols; i++) {
        const bf val = witness_cols.get_at_col(i);
        const e4 challenge = witness_challenges_at_z.get(i);
        acc_z = e4::add(acc_z, e4::mul(challenge, val));
        const uint maybe_idx = witness_cols_map.map[i];
        if (maybe_idx != DOES_NOT_NEED_Z_OMEGA) {
            const e4 ch = witness_challenges_at_z_omega.get(maybe_idx);
            acc_z_omega = e4::add(acc_z_omega, e4::mul(ch, val));
        }
    }

    // Non-witness terms
    uint flat_idx = 0;

    // setup terms at z
    for (uint i = 0; i < num_setup_cols; i++) {
        const bf val = setup_cols.get_at_col(i);
        const e4 challenge = non_witness_challenges_at_z.get(flat_idx++);
        acc_z = e4::add(acc_z, e4::mul(challenge, val));
    }

    // memory terms at z and z * omega
    for (uint i = 0; i < num_memory_cols; i++) {
        const bf val = memory_cols.get_at_col(i);
        const e4 challenge = non_witness_challenges_at_z.get(flat_idx++);
        acc_z = e4::add(acc_z, e4::mul(challenge, val));
        if (process_shuffle_ram_init && i >= memory_lazy_init_addresses_cols_start && i < memory_lazy_init_addresses_cols_start + 2) {
            const e4 ch = non_witness_challenges_at_z_omega.challenges[i - memory_lazy_init_addresses_cols_start];
            acc_z_omega = e4::add(acc_z_omega, e4::mul(ch, val));
        }
    }

    // stage 2 bf terms at z
    for (uint i = 0; i < num_stage_2_bf_cols; i++) {
        const bf val = stage_2_bf_cols.get_at_col(i);
        const e4 challenge = non_witness_challenges_at_z.get(flat_idx++);
        acc_z = e4::add(acc_z, e4::mul(challenge, val));
    }

    // stage 2 e4 terms at z and z * omega
    for (uint i = 0; i < num_stage_2_e4_cols - 1; i++) {
        const e4 val = stage_2_e4_cols.get_at_col(i);
        const e4 challenge = non_witness_challenges_at_z.get(flat_idx++);
        acc_z = e4::add(acc_z, e4::mul(challenge, val));
    }
    { // peel last iteration for grand product at z * omega
        const e4 val = stage_2_e4_cols.get_at_col(num_stage_2_e4_cols - 1);
        const e4 challenge = non_witness_challenges_at_z.get(flat_idx++);
        acc_z = e4::add(acc_z, e4::mul(challenge, val));
        const e4 challenge_at_z_omega =
            process_shuffle_ram_init ? non_witness_challenges_at_z_omega.challenges[2] : non_witness_challenges_at_z_omega.challenges[0];
        acc_z_omega = e4::add(acc_z_omega, e4::mul(challenge_at_z_omega, val));
    }

    // composition term at z
    const e4 comp_val = composition_col.get();
    const e4 comp_challenge = non_witness_challenges_at_z.get(flat_idx);
    acc_z = e4::add(acc_z, e4::mul(comp_challenge, comp_val));

    const e4 denom_z = denom_at_z.get(gid);
    const uint raw_row = bit_reversed ? reverse_bits(gid) >> (32 - log_n) : gid;
    const uint row_shift = n - 1;
    const uint raw_shifted_row = (raw_row + row_shift >= n) ? raw_row + row_shift - n : raw_row + row_shift;
    const uint shifted_row = bit_reversed ? reverse_bits(raw_shifted_row) >> (32 - log_n) : raw_shifted_row;
    const e4 denom_z_omega = denom_at_z.get(shifted_row);

    acc_z = e4::add(acc_z, challenges_times_evals.at_z_sum_neg);
    acc_z_omega = e4::add(acc_z_omega, challenges_times_evals.at_z_omega_sum_neg);
    acc_z = e4::mul(acc_z, denom_z);
    acc_z_omega = e4::mul(acc_z_omega, denom_z_omega);

    quotient.set(e4::add(acc_z, acc_z_omega));
}
