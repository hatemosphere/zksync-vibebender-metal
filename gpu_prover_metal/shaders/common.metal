#pragma once

#include <metal_stdlib>
using namespace metal;

// Metal equivalents of CUDA qualifiers — all become plain inline in MSL.
// These are kept as macros for mechanical porting of kernel code.
#define DEVICE_FORCEINLINE inline __attribute__((always_inline))
#define HOST_DEVICE_FORCEINLINE inline __attribute__((always_inline))
#define EXTERN

// CUDA-style warp constants mapped to Metal SIMD group.
// Apple GPUs have SIMD group width = 32.
constant constexpr uint WARP_SIZE = 32;
constant constexpr uint LOG_WARP_SIZE = 5;
