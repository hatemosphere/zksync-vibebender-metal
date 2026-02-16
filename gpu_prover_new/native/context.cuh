#pragma once
#include "field.cuh"
#include "memory.cuh"

using namespace ::airbender::field;
using namespace ::airbender::memory;

namespace airbender::field {

// TODO:
// Decide max order we need based on trace column length and constraint degree
static constexpr int TWO_ADICITY = 27;
static constexpr int CMEM_COARSE_LOG_COUNT = 10;
static constexpr int CMEM_COARSE_MASK = (1 << CMEM_COARSE_LOG_COUNT) - 1;
static constexpr int CMEM_FINE_LOG_COUNT = 8;
static constexpr int CMEM_FINE_MASK = (1 << CMEM_FINE_LOG_COUNT) - 1;
static constexpr int CMEM_FINEST_LOG_COUNT = 10;
static constexpr int CMEM_FINEST_MASK = (1 << CMEM_FINEST_LOG_COUNT) - 1;
static constexpr int GMEM_COARSE_LOG_COUNT = 13;
static constexpr int GMEM_COARSE_MASK = (1 << GMEM_COARSE_LOG_COUNT) - 1;

struct powers_layer_data {
  const base_field *values;
  unsigned mask;
  unsigned log_count;
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

} // namespace airbender::field

EXTERN __device__ __constant__ powers_data_3_layer ab_powers_data_w;
EXTERN __device__ __constant__ base_field ab_inv_sizes[TWO_ADICITY + 1];
// Use cmem twiddles for stages where warps access them uniformly
EXTERN __device__ __constant__ base_field ab_fwd_cmem_twiddles_coarse[1 << CMEM_COARSE_LOG_COUNT];
EXTERN __device__ __constant__ base_field ab_inv_cmem_twiddles_coarse[1 << CMEM_COARSE_LOG_COUNT];
EXTERN __device__ __constant__ base_field ab_fwd_cmem_twiddles_fine[1 << CMEM_FINE_LOG_COUNT];
EXTERN __device__ __constant__ base_field ab_inv_cmem_twiddles_fine[1 << CMEM_FINE_LOG_COUNT];
EXTERN __device__ __constant__ base_field ab_fwd_cmem_twiddles_finest[1 << CMEM_FINEST_LOG_COUNT];
EXTERN __device__ __constant__ base_field ab_inv_cmem_twiddles_finest[1 << CMEM_FINEST_LOG_COUNT];
EXTERN __device__ __constant__ const base_field *ab_fwd_gmem_twiddles_coarse;
EXTERN __device__ __constant__ const base_field *ab_inv_gmem_twiddles_coarse;

namespace airbender::field {

// DEVICE_FORCEINLINE base_field get_power(const powers_data_3_layer &data, const unsigned index, const bool inverse) {
//   const unsigned idx = inverse ? (1u << CIRCLE_GROUP_LOG_ORDER) - index : index;
// 
//   const unsigned coarsest_idx = (idx >> (data.fine.log_count + data.coarser.log_count)) & data.coarsest.mask;
//   base_field val = load_ca(data.coarsest.values + coarsest_idx);
// 
//   const unsigned coarser_idx = (idx >> data.fine.log_count) & data.coarser.mask;
//   if (coarser_idx != 0)
//     val = base_field::mul(val, load_ca(data.coarser.values + coarser_idx));
// 
//   const unsigned fine_idx = idx & data.fine.mask;
//   if (fine_idx != 0)
//     val = base_field::mul(val, load_ca(data.fine.values + fine_idx));
// 
//   return val;
// }

// DEVICE_FORCEINLINE base_field get_power_of_w(const unsigned index, const bool inverse) { return get_power(ab_powers_data_w, index, inverse); }

} // namespace airbender::field
