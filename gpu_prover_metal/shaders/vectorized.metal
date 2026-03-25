#pragma once

#include "field.metal"
#include "memory.metal"

// Metal port of gpu_prover/native/vectorized.cuh
// Provides vectorized matrix accessors that read/write extension field elements
// stored as interleaved base field columns.

namespace airbender {
namespace vectorized {

using namespace field;
using namespace memory;

// Vectorized getter: reads WIDTH base_field columns as an extension field element
template <uint WIDTH>
struct vectorized_matrix_getter {
    matrix_getter<base_field> internal;
    DEVICE_FORCEINLINE void add_row(const uint offset) { internal.add_row(offset); }
    DEVICE_FORCEINLINE void sub_row(const uint offset) { internal.sub_row(offset); }
    DEVICE_FORCEINLINE void add_col(const uint offset) { internal.add_col(WIDTH * offset); }

    DEVICE_FORCEINLINE ext4_field get() const {
        base_field coeffs[4];
        coeffs[0] = internal.get();
        for (uint i = 1; i < WIDTH; i++)
            coeffs[i] = internal.get_at_col(i);
        // fill remaining with zero if WIDTH < 4
        for (uint i = WIDTH; i < 4; i++)
            coeffs[i] = base_field::zero();
        return {ext2_field{coeffs[0], coeffs[1]}, ext2_field{coeffs[2], coeffs[3]}};
    }

    DEVICE_FORCEINLINE ext4_field get_at_col(const uint col) const {
        base_field coeffs[4];
        const uint bf_col = WIDTH * col;
        coeffs[0] = internal.get_at_col(bf_col);
        for (uint i = 1; i < WIDTH; i++)
            coeffs[i] = internal.get_at_col(bf_col + i);
        for (uint i = WIDTH; i < 4; i++)
            coeffs[i] = base_field::zero();
        return {ext2_field{coeffs[0], coeffs[1]}, ext2_field{coeffs[2], coeffs[3]}};
    }

    DEVICE_FORCEINLINE matrix_getter<base_field> copy() const {
        return internal;
    }
};

// Vectorized setter
template <uint WIDTH>
struct vectorized_matrix_setter {
    matrix_setter<base_field> internal;
    DEVICE_FORCEINLINE void add_row(const uint offset) { internal.add_row(offset); }
    DEVICE_FORCEINLINE void sub_row(const uint offset) { internal.sub_row(offset); }
    DEVICE_FORCEINLINE void add_col(const uint offset) { internal.add_col(WIDTH * offset); }

    DEVICE_FORCEINLINE void set(const thread ext4_field &value) const {
        internal.set(value.coefficients[0].coefficients[0]);
        if (WIDTH > 1) internal.set_at_col(1, value.coefficients[0].coefficients[1]);
        if (WIDTH > 2) internal.set_at_col(2, value.coefficients[1].coefficients[0]);
        if (WIDTH > 3) internal.set_at_col(3, value.coefficients[1].coefficients[1]);
    }

    DEVICE_FORCEINLINE void set_at_col(const uint col, const thread ext4_field &value) const {
        const uint bf_col = WIDTH * col;
        internal.set_at_col(bf_col, value.coefficients[0].coefficients[0]);
        if (WIDTH > 1) internal.set_at_col(bf_col + 1, value.coefficients[0].coefficients[1]);
        if (WIDTH > 2) internal.set_at_col(bf_col + 2, value.coefficients[1].coefficients[0]);
        if (WIDTH > 3) internal.set_at_col(bf_col + 3, value.coefficients[1].coefficients[1]);
    }
};

// Vectorized getter+setter
template <uint WIDTH>
struct vectorized_matrix_getter_setter {
    matrix_getter_setter<base_field> internal;
    DEVICE_FORCEINLINE void add_row(const uint offset) { internal.add_row(offset); }
    DEVICE_FORCEINLINE void sub_row(const uint offset) { internal.sub_row(offset); }
    DEVICE_FORCEINLINE void add_col(const uint offset) { internal.add_col(WIDTH * offset); }

    DEVICE_FORCEINLINE ext4_field get() const {
        base_field coeffs[4];
        coeffs[0] = internal.get();
        for (uint i = 1; i < WIDTH; i++)
            coeffs[i] = internal.get_at_col(i);
        for (uint i = WIDTH; i < 4; i++)
            coeffs[i] = base_field::zero();
        return {ext2_field{coeffs[0], coeffs[1]}, ext2_field{coeffs[2], coeffs[3]}};
    }

    DEVICE_FORCEINLINE ext4_field get_at_col(const uint col) const {
        base_field coeffs[4];
        const uint bf_col = WIDTH * col;
        coeffs[0] = internal.get_at_col(bf_col);
        for (uint i = 1; i < WIDTH; i++)
            coeffs[i] = internal.get_at_col(bf_col + i);
        for (uint i = WIDTH; i < 4; i++)
            coeffs[i] = base_field::zero();
        return {ext2_field{coeffs[0], coeffs[1]}, ext2_field{coeffs[2], coeffs[3]}};
    }

    DEVICE_FORCEINLINE void set(const thread ext4_field &value) const {
        internal.set(value.coefficients[0].coefficients[0]);
        if (WIDTH > 1) internal.set_at_col(1, value.coefficients[0].coefficients[1]);
        if (WIDTH > 2) internal.set_at_col(2, value.coefficients[1].coefficients[0]);
        if (WIDTH > 3) internal.set_at_col(3, value.coefficients[1].coefficients[1]);
    }
};

// Convenience aliases matching CUDA naming
using vectorized_e4_matrix_getter = vectorized_matrix_getter<4>;
using vectorized_e4_matrix_setter = vectorized_matrix_setter<4>;
using vectorized_e4_matrix_getter_setter = vectorized_matrix_getter_setter<4>;

} // namespace vectorized
} // namespace airbender
