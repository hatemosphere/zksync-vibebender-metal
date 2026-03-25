#pragma once

#include "common.metal"

// Metal memory access utilities.
// Replaces gpu_prover/native/memory.cuh.
//
// Key differences from CUDA:
// - No cache load/store modifiers (Metal has no instruction-level cache hints)
// - Warp shuffles → simd_shuffle_xor (Metal SIMD group intrinsics)
// - __syncwarp → simdgroup_barrier
// - __brev → reverse_bits
// - All accessor structs simplified: no ld_modifier/st_modifier template params

namespace airbender {
namespace memory {

// --- Simple load/store (no cache modifiers in Metal) ---

template <typename T>
DEVICE_FORCEINLINE T load(const device T* address) {
    return *address;
}

template <typename T>
DEVICE_FORCEINLINE void store(device T* address, const thread T& value) {
    *address = value;
}

// Strided load: loads sizeof(T)/sizeof(U) elements at stride intervals
template <typename T, typename U, uint STRIDE>
DEVICE_FORCEINLINE T ld_strided(const device T* address, const uint offset) {
    constexpr uint count = sizeof(T) / sizeof(U);
    T result;
    const device U* pa = reinterpret_cast<const device U*>(address) + offset;
    thread U* pr = reinterpret_cast<thread U*>(&result);
    for (uint i = 0; i < count; i++) {
        pr[i] = pa[i * STRIDE];
    }
    return result;
}

// Strided store
template <typename T, typename U, uint STRIDE>
DEVICE_FORCEINLINE void st_strided(device T* address, const thread T& value, const uint offset) {
    constexpr uint count = sizeof(T) / sizeof(U);
    device U* pa = reinterpret_cast<device U*>(address) + offset;
    const thread U* pv = reinterpret_cast<const thread U*>(&value);
    for (uint i = 0; i < count; i++) {
        pa[i * STRIDE] = pv[i];
    }
}

// Convenience aliases matching CUDA naming (all resolve to plain load/store)
template <typename T>
DEVICE_FORCEINLINE T load_g(const device T* address) { return *address; }

template <typename T>
DEVICE_FORCEINLINE T load_cg(const device T* address) { return *address; }

template <typename T>
DEVICE_FORCEINLINE T load_ca(const device T* address) { return *address; }

template <typename T>
DEVICE_FORCEINLINE T load_cs(const device T* address) { return *address; }

template <typename T>
DEVICE_FORCEINLINE void store_cs(device T* address, const thread T& value) { *address = value; }

template <typename T>
DEVICE_FORCEINLINE void store_cg(device T* address, const thread T& value) { *address = value; }

// --- SIMD shuffle operations (replacing CUDA warp shuffles) ---
// Metal supports simd_shuffle_xor since Metal 2.0 on Apple GPUs (SIMD width = 32).

DEVICE_FORCEINLINE uint shfl_xor(uint var, ushort lane_mask) {
    return simd_shuffle_xor(var, lane_mask);
}

DEVICE_FORCEINLINE uint2 shfl_xor(uint2 var, ushort lane_mask) {
    return uint2(simd_shuffle_xor(var.x, lane_mask),
                 simd_shuffle_xor(var.y, lane_mask));
}

DEVICE_FORCEINLINE uint4 shfl_xor(uint4 var, ushort lane_mask) {
    return uint4(simd_shuffle_xor(var.x, lane_mask),
                 simd_shuffle_xor(var.y, lane_mask),
                 simd_shuffle_xor(var.z, lane_mask),
                 simd_shuffle_xor(var.w, lane_mask));
}

DEVICE_FORCEINLINE uint shfl(uint var, ushort src_lane) {
    return simd_shuffle(var, src_lane);
}

// --- Swap and transpose utilities ---

template <typename T>
DEVICE_FORCEINLINE void swap(thread T& a, thread T& b) {
    T temp = a;
    a = b;
    b = temp;
}

template <uint STRIDE>
DEVICE_FORCEINLINE uint swap_index(const uint index) {
    const uint i1 = index % STRIDE;
    const uint i2 = index / STRIDE;
    const uint i3 = i2 * STRIDE * 2;
    return i3 + i1;
}

// Transpose tile using SIMD shuffles (replaces CUDA warp shuffle transpose)
template <typename T, uint STRIDE_ROW, uint COUNT_ROW, uint STRIDE_COL = STRIDE_ROW, uint COUNT_COL = COUNT_ROW>
DEVICE_FORCEINLINE void transpose_tile(thread T* u, const uint lane_id) {
    const bool swap_rows = !(lane_id & STRIDE_ROW);
    if (swap_rows) {
        for (uint i = 0; i < COUNT_ROW; i++) {
            const uint index = swap_index<STRIDE_ROW>(i);
            swap(u[index], u[index + STRIDE_ROW]);
        }
    }
    for (uint i = 0; i < COUNT_COL; i++) {
        const uint index = swap_index<STRIDE_COL>(i);
        u[index] = simd_shuffle_xor(u[index], static_cast<ushort>(STRIDE_COL));
    }
    if (swap_rows) {
        for (uint i = 0; i < COUNT_ROW; i++) {
            const uint index = swap_index<STRIDE_ROW>(i);
            swap(u[index], u[index + STRIDE_ROW]);
        }
    }
}

// --- Vector accessor (simplified, no cache modifier template params) ---

template <typename T>
struct vector_getter {
    const device T* ptr;

    DEVICE_FORCEINLINE T get() const { return *ptr; }
    DEVICE_FORCEINLINE T get(const uint i) const { return ptr[i]; }
};

template <typename T>
struct vector_setter {
    device T* ptr;

    DEVICE_FORCEINLINE void set(const thread T& value) const { *ptr = value; }
    DEVICE_FORCEINLINE void set(const uint i, const thread T& value) const { ptr[i] = value; }
};

// --- Matrix accessor (simplified, no cache modifier template params) ---

template <typename T>
struct matrix_getter {
    const device T* ptr;
    uint stride;

    DEVICE_FORCEINLINE T get() const { return *ptr; }
    DEVICE_FORCEINLINE T get_at_row(const uint row) const { return ptr[row]; }
    DEVICE_FORCEINLINE T get_at_col(const uint col) const { return ptr[col * stride]; }
    DEVICE_FORCEINLINE T get(const uint row, const uint col) const { return ptr[row + col * stride]; }

    DEVICE_FORCEINLINE void add_row(const uint offset) { ptr += offset; }
    DEVICE_FORCEINLINE void sub_row(const uint offset) { ptr -= offset; }
    DEVICE_FORCEINLINE void add_col(const uint offset) { ptr += offset * stride; }
    DEVICE_FORCEINLINE void sub_col(const uint offset) { ptr -= offset * stride; }
};

template <typename T>
struct matrix_setter {
    device T* ptr;
    uint stride;

    DEVICE_FORCEINLINE void set(const thread T& value) const { *ptr = value; }
    DEVICE_FORCEINLINE void set_at_row(const uint row, const thread T& value) const { ptr[row] = value; }
    DEVICE_FORCEINLINE void set_at_col(const uint col, const thread T& value) const { ptr[col * stride] = value; }
    DEVICE_FORCEINLINE void set(const uint row, const uint col, const thread T& value) const {
        ptr[row + col * stride] = value;
    }

    DEVICE_FORCEINLINE void add_row(const uint offset) { ptr += offset; }
    DEVICE_FORCEINLINE void sub_row(const uint offset) { ptr -= offset; }
    DEVICE_FORCEINLINE void add_col(const uint offset) { ptr += offset * stride; }
    DEVICE_FORCEINLINE void sub_col(const uint offset) { ptr -= offset * stride; }
};

template <typename T>
struct matrix_getter_setter {
    device T* ptr;
    uint stride;

    DEVICE_FORCEINLINE T get() const { return *ptr; }
    DEVICE_FORCEINLINE T get_at_row(const uint row) const { return ptr[row]; }
    DEVICE_FORCEINLINE T get_at_col(const uint col) const { return ptr[col * stride]; }
    DEVICE_FORCEINLINE T get(const uint row, const uint col) const { return ptr[row + col * stride]; }

    DEVICE_FORCEINLINE void set(const thread T& value) const { *ptr = value; }
    DEVICE_FORCEINLINE void set_at_row(const uint row, const thread T& value) const { ptr[row] = value; }
    DEVICE_FORCEINLINE void set_at_col(const uint col, const thread T& value) const { ptr[col * stride] = value; }
    DEVICE_FORCEINLINE void set(const uint row, const uint col, const thread T& value) const {
        ptr[row + col * stride] = value;
    }

    DEVICE_FORCEINLINE void add_row(const uint offset) { ptr += offset; }
    DEVICE_FORCEINLINE void sub_row(const uint offset) { ptr -= offset; }
    DEVICE_FORCEINLINE void add_col(const uint offset) { ptr += offset * stride; }
    DEVICE_FORCEINLINE void sub_col(const uint offset) { ptr -= offset * stride; }
};

} // namespace memory
} // namespace airbender
