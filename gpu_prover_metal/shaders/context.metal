#pragma once

#include "field.metal"

// Device context structures for twiddle factors and power tables.
// Replaces gpu_prover/native/context.cuh.
//
// Key difference from CUDA: No __device__ __constant__ global variables.
// Instead, these structs are passed as kernel buffer arguments.

namespace airbender {
namespace field {

static constant constexpr uint OMEGA_LOG_ORDER = 25;
static constant constexpr uint CIRCLE_GROUP_LOG_ORDER = 31;

struct powers_layer_data {
    const device ext2_field* values;
    uint mask;
    uint log_count;
};

struct powers_data_2_layer {
    powers_layer_data fine;
    powers_layer_data coarse;
};

struct powers_data_3_layer {
    powers_layer_data fine;
    powers_layer_data coarser;
    powers_layer_data coarsest;
};

// Get a power of omega from the 3-layer lookup table.
// In CUDA this reads from __constant__ memory; in Metal the data
// is passed via kernel buffer arguments.
DEVICE_FORCEINLINE ext2_field get_power(const thread powers_data_3_layer& data,
                                         const uint index,
                                         const bool inverse) {
    const uint idx = inverse ? (1u << CIRCLE_GROUP_LOG_ORDER) - index : index;

    const uint coarsest_idx = (idx >> (data.fine.log_count + data.coarser.log_count)) & data.coarsest.mask;
    ext2_field val = data.coarsest.values[coarsest_idx];

    const uint coarser_idx = (idx >> data.fine.log_count) & data.coarser.mask;
    if (coarser_idx != 0)
        val = ext2_field::mul(val, data.coarser.values[coarser_idx]);

    const uint fine_idx = idx & data.fine.mask;
    if (fine_idx != 0)
        val = ext2_field::mul(val, data.fine.values[fine_idx]);

    return val;
}

DEVICE_FORCEINLINE ext2_field get_power_of_w(const thread powers_data_3_layer& data,
                                              const uint index,
                                              const bool inverse) {
    return get_power(data, index, inverse);
}

} // namespace field
} // namespace airbender
