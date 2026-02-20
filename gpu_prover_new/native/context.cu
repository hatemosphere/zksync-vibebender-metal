#include "context.cuh"

using namespace ::airbender::field;

__device__ __constant__ powers_data_3_layer ab_powers_data_w;
__device__ __constant__ base_field ab_inv_sizes[TWO_ADICITY + 1];
// Use cmem twiddles for stages where warps access them uniformly
__device__ __constant__ base_field ab_fwd_cmem_twiddles_coarse[1 << CMEM_COARSE_LOG_COUNT];
__device__ __constant__ base_field ab_inv_cmem_twiddles_coarse[1 << CMEM_COARSE_LOG_COUNT];
__device__ __constant__ base_field ab_fwd_cmem_twiddles_fine[1 << CMEM_FINE_LOG_COUNT];
__device__ __constant__ base_field ab_inv_cmem_twiddles_fine[1 << CMEM_FINE_LOG_COUNT];
__device__ __constant__ base_field ab_fwd_cmem_twiddles_finest_10[1 << 10];
__device__ __constant__ base_field ab_inv_cmem_twiddles_finest_10[1 << 10];
__device__ __constant__ base_field ab_fwd_cmem_twiddles_finest_11[1 << 11];
__device__ __constant__ base_field ab_inv_cmem_twiddles_finest_11[1 << 11];
__device__ __constant__ const base_field *ab_fwd_gmem_twiddles_coarse;
__device__ __constant__ const base_field *ab_inv_gmem_twiddles_coarse;
