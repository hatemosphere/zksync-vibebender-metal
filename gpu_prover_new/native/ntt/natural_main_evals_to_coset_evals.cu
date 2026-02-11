#include "../context.cuh"
#include "../field.cuh"
#include "../memory.cuh"
#include <cuda_pipeline.h>

using namespace ::airbender::field;
using namespace ::airbender::memory;

namespace airbender::ntt {

using bf = base_field;

// This is a little tricky:
// it assumes "i" NEEDS to be bitreved and accounts for that by assuming "fine" and "coarse"
// arrays are already bitreved.
template <bool inverse> DEVICE_FORCEINLINE bf get_twiddle(const int i) {
  const powers_data_2_layer &data = inverse ? ab_powers_data_w_inv_bitrev_for_ntt : ab_powers_data_w_bitrev_for_ntt;
  int fine_idx = (i >> data.coarse.log_count) & data.fine.mask;
  int coarse_idx = i & data.coarse.mask;
  auto coarse = memory::load_ca(data.coarse.values + coarse_idx);
  if (fine_idx == 0)
    return coarse;
  auto fine = memory::load_ca(data.fine.values + fine_idx);
  return bf::mul(fine, coarse);
}

DEVICE_FORCEINLINE void exchg_dit_0(bf &a, bf &b) {
  const auto a_tmp = a;
  a = bf::add(a_tmp, b);
  b = bf::sub(a_tmp, b);
}

DEVICE_FORCEINLINE void exchg_dit(bf &a, bf &b, const bf &twiddle) {
  b = bf::mul(b, twiddle);
  const auto a_tmp = a;
  a = bf::add(a_tmp, b);
  b = bf::sub(a_tmp, b);
}

DEVICE_FORCEINLINE void exchg_dif_0(bf &a, bf &b) {
  const auto a_tmp = a;
  a = bf::add(a_tmp, b);
  b = bf::sub(a_tmp, b);
}

DEVICE_FORCEINLINE void exchg_dif(bf &a, bf &b, const bf &twiddle) {
  const auto a_tmp = a;
  a = bf::add(a_tmp, b);
  b = bf::sub(a_tmp, b);
  b = bf::mul(b, twiddle);
}

// bank-conflict-free swizzling pattern from https://www.nvidia.com/en-us/on-demand/session/gtc24-s62400/ slide 92
DEVICE_FORCEINLINE int xy_to_linear(const int x, const int y) {
  return y * 32 + (y ^ x);
}

template <int STRIDE, int REGION_SIZE, int NUM_REGIONS>
DEVICE_FORCEINLINE void reg_exchg_inv(bf *vals, const int exchg_region_offset, const bf *twiddles) {
#pragma unroll
  for (int region{0}; region < NUM_REGIONS; region++) {
    const bf twiddle = twiddles[exchg_region_offset + region];
    const int region_offset = region * REGION_SIZE;
#pragma unroll
    for (int lane_in_region{0}; lane_in_region < STRIDE; lane_in_region++) {
      const int i = region_offset + lane_in_region;
      exchg_dit(vals[i], vals[i + STRIDE], twiddle);
    }
  }
}

template <int STRIDE, int REGION_SIZE, int NUM_REGIONS>
DEVICE_FORCEINLINE void reg_exchg_fwd(bf *vals, const int exchg_region_offset, const bf *twiddles) {
#pragma unroll
  for (int region{0}; region < NUM_REGIONS; region++) {
    const bf twiddle = twiddles[exchg_region_offset + region];
    const int region_offset = region * REGION_SIZE;
#pragma unroll
    for (int lane_in_region{0}; lane_in_region < STRIDE; lane_in_region++) {
      const int i = region_offset + lane_in_region;
      exchg_dif(vals[i], vals[i + STRIDE], twiddle);
    }
  }
}

template <int GROUP>
DEVICE_FORCEINLINE void exchg_pipeline_group(bf *vals, const bf twiddle) {
  exchg_dit_0(vals[GROUP], vals[GROUP + 16]);
  exchg_dit_0(vals[GROUP + 8], vals[GROUP + 24]);
  exchg_dit_0(vals[GROUP], vals[GROUP + 8]);
  exchg_dit(vals[GROUP + 16], vals[GROUP + 24], twiddle);
}

template <int GROUP, int IL_GMEM_STRIDE, int PL_GROUP_SIZE, int PL_STRIDE>
DEVICE_FORCEINLINE void prefetch_pipeline_group(bf *vals, const bf_matrix_getter<ld_modifier::cg> &gmem_in, const int thread_il_gmem_start) {
#pragma unroll
  for (int i{0}, row{thread_il_gmem_start + GROUP * IL_GMEM_STRIDE}; i < PL_GROUP_SIZE; i++, row += IL_GMEM_STRIDE * PL_STRIDE)
    vals[GROUP + i * PL_STRIDE] = gmem_in.get_at_row(row);
}

EXTERN __launch_bounds__(512, 1) __global__
    void ab_main_to_monomials_first_10_stages_register_pipeline_kernel(bf_matrix_getter<ld_modifier::cg> gmem_in,
                                                                       bf_matrix_setter<st_modifier::cg> gmem_out,
                                                                       const int log_n,
                                                                       const int num_ntts) {
  constexpr int VALS_PER_THREAD = 32;
  constexpr int LOG_DATA_TILE_SIZE = 4;
  constexpr int TILE_SIZE = 1 << LOG_DATA_TILE_SIZE;
  constexpr int LOG_DATA_TILES_PER_BLOCK = 10;
  constexpr int THREAD_TILES_PER_BLOCK = 32;
  constexpr int TILE_GMEM_STRIDE = 1 << (24 - 10);
  constexpr int IL_GMEM_STRIDE = TILE_GMEM_STRIDE * THREAD_TILES_PER_BLOCK;

  constexpr int PL_GROUP_SIZE = 4;
  // constexpr int NUM_PL_GROUPS = 8;
  constexpr int PL_STRIDE = 8;
  // constexpr int PL_DEPTH = 2;

  // TODO: make some of these kernel arguments
  const int lane_in_tile = threadIdx.x & 15;
  const int tile_id = threadIdx.x >> LOG_DATA_TILE_SIZE;
  const int tile_gmem_stride = 1 << (log_n - LOG_DATA_TILES_PER_BLOCK);
  const int gmem_block_offset = blockIdx.x << LOG_DATA_TILE_SIZE;
  gmem_in.add_row(gmem_block_offset);
  gmem_out.add_row(gmem_block_offset);

  gmem_in.add_col(blockIdx.y);
  gmem_out.add_col(blockIdx.y);

  extern __shared__ bf smem_block[]; // 16384 * 4 bytes

  const bf *twiddles = ab_inv_twiddles_first_10_stages;

  bf vals[VALS_PER_THREAD];

  // "ct" = consecutive tile layout
  // "it" = interleaved tile layout
  const int thread_il_gmem_start = lane_in_tile + tile_id * TILE_GMEM_STRIDE;
  const int thread_ct_gmem_start = lane_in_tile + tile_id * IL_GMEM_STRIDE;
  const int thread_il_smem_start = lane_in_tile + tile_id * TILE_SIZE;
  const int thread_ct_smem_start = lane_in_tile + tile_id * TILE_SIZE * THREAD_TILES_PER_BLOCK;

// #pragma unroll 1
//   for (int ntt_idx{0}; ntt_idx < num_ntts; ntt_idx++) {
    const bf twiddle = twiddles[1];
    prefetch_pipeline_group<0, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<0>(vals, twiddle);
    prefetch_pipeline_group<1, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<1>(vals, twiddle);
    prefetch_pipeline_group<2, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<2>(vals, twiddle);
    prefetch_pipeline_group<3, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<3>(vals, twiddle);
    prefetch_pipeline_group<4, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<4>(vals, twiddle);
    prefetch_pipeline_group<5, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<5>(vals, twiddle);
    prefetch_pipeline_group<6, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<6>(vals, twiddle);
    prefetch_pipeline_group<7, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<7>(vals, twiddle);

    // reg_exchg_inv<16, 32, 1>(vals, 0, twiddles);
    // reg_exchg_inv<8, 16, 2>(vals, 0, twiddles);
    reg_exchg_inv<4, 8, 4>(vals, 0, twiddles);
    reg_exchg_inv<2, 4, 8>(vals, 0, twiddles);
    reg_exchg_inv<1, 2, 16>(vals, 0, twiddles);

#pragma unroll
    for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
      smem_block[addr] = vals[i]; // write interleaved smem tiles

    __syncthreads();

#pragma unroll
    for (int i{0}, addr{thread_ct_smem_start}; i < 32; i++, addr += TILE_SIZE)
      vals[i] = smem_block[addr]; // read consecutive smem tiles

    int tile_exchg_region_offset = tile_id;
    reg_exchg_inv<16, 32, 1>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
    reg_exchg_inv<8, 16, 2>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
    reg_exchg_inv<4, 8, 4>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
    reg_exchg_inv<2, 4, 8>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
    reg_exchg_inv<1, 2, 16>(vals, tile_exchg_region_offset, twiddles);

#pragma unroll
    for (int i{0}, row{thread_ct_gmem_start}; i < 32; i++, row += tile_gmem_stride)
      gmem_out.set_at_row(row, vals[i]); // write consecutive gmem tiles
//     gmem_out.inc_col();
//   }
}

EXTERN __launch_bounds__(512, 1) __global__
    void ab_main_to_monomials_first_10_stages_pipeline_tile_8_kernel(bf_matrix_getter<ld_modifier::cg> gmem_in,
                                                                       bf_matrix_setter<st_modifier::cg> gmem_out,
                                                                       const int log_n,
                                                                       const int num_ntts) {
  constexpr int VALS_PER_THREAD = 32;
  constexpr int LOG_DATA_TILE_SIZE = 3;
  constexpr int TILE_SIZE = 1 << LOG_DATA_TILE_SIZE;
  constexpr int LOG_DATA_TILES_PER_BLOCK = 10;
  constexpr int THREAD_TILES_PER_BLOCK = 32;
  constexpr int TILE_GMEM_STRIDE = 1 << (24 - 10);
  constexpr int IL_GMEM_STRIDE = TILE_GMEM_STRIDE * THREAD_TILES_PER_BLOCK;

  constexpr int PL_GROUP_SIZE = 4;
  // constexpr int NUM_PL_GROUPS = 8;
  constexpr int PL_STRIDE = 8;
  // constexpr int PL_DEPTH = 2;

  // TODO: make some of these kernel arguments
  const int lane_in_tile = threadIdx.x & 7;
  const int tile_id = threadIdx.x >> LOG_DATA_TILE_SIZE;
  const int tile_gmem_stride = 1 << (log_n - LOG_DATA_TILES_PER_BLOCK);
  const int gmem_block_offset = blockIdx.x << LOG_DATA_TILE_SIZE;
  gmem_in.add_row(gmem_block_offset);
  gmem_out.add_row(gmem_block_offset);

  extern __shared__ bf smem_block[]; // 8192 * 4 bytes

  const bf *twiddles = ab_inv_twiddles_first_10_stages;

  bf vals[VALS_PER_THREAD];

  // "ct" = consecutive tile layout
  // "it" = interleaved tile layout
  const int thread_il_gmem_start = lane_in_tile + tile_id * TILE_GMEM_STRIDE;
  const int thread_ct_gmem_start = lane_in_tile + tile_id * IL_GMEM_STRIDE;
  const int thread_il_smem_start = lane_in_tile + tile_id * TILE_SIZE;
  const int thread_ct_smem_start = lane_in_tile + tile_id * TILE_SIZE * THREAD_TILES_PER_BLOCK;

// #pragma unroll 1
//   for (int ntt_idx{0}; ntt_idx < num_ntts; ntt_idx++) {
    const bf twiddle = twiddles[1];
    prefetch_pipeline_group<0, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<0>(vals, twiddle);
    prefetch_pipeline_group<1, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<1>(vals, twiddle);
    prefetch_pipeline_group<2, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<2>(vals, twiddle);
    prefetch_pipeline_group<3, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<3>(vals, twiddle);
    prefetch_pipeline_group<4, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<4>(vals, twiddle);
    prefetch_pipeline_group<5, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<5>(vals, twiddle);
    prefetch_pipeline_group<6, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<6>(vals, twiddle);
    prefetch_pipeline_group<7, IL_GMEM_STRIDE, PL_GROUP_SIZE, PL_STRIDE>(vals, gmem_in, thread_il_gmem_start);
    exchg_pipeline_group<7>(vals, twiddle);

    // reg_exchg_inv<16, 32, 1>(vals, 0, twiddles);
    // reg_exchg_inv<8, 16, 2>(vals, 0, twiddles);
    reg_exchg_inv<4, 8, 4>(vals, 0, twiddles);
    reg_exchg_inv<2, 4, 8>(vals, 0, twiddles);
    reg_exchg_inv<1, 2, 16>(vals, 0, twiddles);

#pragma unroll
    for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
      smem_block[addr] = vals[i]; // write interleaved smem tiles

    __syncthreads();

#pragma unroll
    for (int i{0}, addr{thread_ct_smem_start}; i < 32; i++, addr += TILE_SIZE)
      vals[i] = smem_block[addr]; // read consecutive smem tiles

    int tile_exchg_region_offset = tile_id;
    reg_exchg_inv<16, 32, 1>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
    reg_exchg_inv<8, 16, 2>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
    reg_exchg_inv<4, 8, 4>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
    reg_exchg_inv<2, 4, 8>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
    reg_exchg_inv<1, 2, 16>(vals, tile_exchg_region_offset, twiddles);

#pragma unroll
    for (int i{0}, row{thread_ct_gmem_start}; i < 32; i++, row += tile_gmem_stride)
      gmem_out.set_at_row(row, vals[i]); // write consecutive gmem tiles
//     gmem_out.inc_col();
//   }
}

EXTERN __launch_bounds__(512, 1) __global__
    void ab_main_to_monomials_first_10_stages_coalesced_kernel(bf_matrix_getter<ld_modifier::cg> gmem_in,
                                                               bf_matrix_setter<st_modifier::cg> gmem_out,
                                                               const int log_n,
                                                               const int num_ntts) {
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  const int TILE_SIZE = 32;
  const int LOG_DATA_TILES_PER_BLOCK = 10;

  const int gmem_block_offset = blockIdx.x << 5;
  const int tile_gmem_stride = 1 << (log_n - LOG_DATA_TILES_PER_BLOCK);
  gmem_in.add_row(gmem_block_offset);
  gmem_out.add_row(gmem_block_offset);
  const bf *gmem_in_ptr = gmem_in.ptr;
  bf *gmem_out_ptr = gmem_out.ptr;

  uint4 vals[8];

  {
    const int vectorized_memcpy_thread_il_gmem_start = 4 * (lane_id & 7) + (32 * (lane_id >> 3) + 2 * warp_id) * tile_gmem_stride;
    // const int vectorized_memcpy_thread_il_smem_start = 4 * (lane_id & 7) + (32 * (lane_id >> 3) + 2 * warp_id) * TILE_SIZE;
    // const int vectorized_memcpy_thread_ct_smem_start = 4 * lane_id + warp_id * TILE_SIZE * THREAD_TILES_PER_BLOCK;

    for (int i{0}, row{vectorized_memcpy_thread_il_gmem_start}; i < 8; i++, row += 4 * 32 * tile_gmem_stride)
      vals[i] = *reinterpret_cast<const uint4 *>(gmem_in_ptr + row);

    for (int i{0}, row{vectorized_memcpy_thread_il_gmem_start}; i < 8; i++, row += 4 * 32 * tile_gmem_stride)
      *reinterpret_cast<uint4 *>(gmem_out_ptr + row) = vals[i];
  }

  {
    const int vectorized_memcpy_thread_il_gmem_start = 4 * (lane_id & 7) + (32 * (lane_id >> 3) + 2 * warp_id + 1) * tile_gmem_stride;
    const int vectorized_memcpy_thread_il_smem_start = 4 * (lane_id & 7) + (32 * (lane_id >> 3) + 2 * warp_id + 1) * TILE_SIZE;
    // const int vectorized_memcpy_thread_ct_smem_start = 4 * lane_id + warp_id * TILE_SIZE * THREAD_TILES_PER_BLOCK;

    for (int i{0}, row{vectorized_memcpy_thread_il_gmem_start}; i < 8; i++, row += 4 * 32 * tile_gmem_stride)
      vals[i] = *reinterpret_cast<const uint4 *>(gmem_in_ptr + row);

    for (int i{0}, row{vectorized_memcpy_thread_il_gmem_start}; i < 8; i++, row += 4 * 32 * tile_gmem_stride)
      *reinterpret_cast<uint4 *>(gmem_out_ptr + row) = vals[i];
  }
}

EXTERN __launch_bounds__(256, 2) __global__
    void ab_main_to_monomials_first_10_stages_tile_8_kernel(bf_matrix_getter<ld_modifier::cg> gmem_in,
                                                            bf_matrix_setter<st_modifier::cg> gmem_out,
                                                            const int log_n,
                                                            const int num_ntts) {
  constexpr int VALS_PER_THREAD = 32;
  constexpr int LOG_DATA_TILE_SIZE = 3;
  constexpr int TILE_SIZE = 1 << LOG_DATA_TILE_SIZE;
  constexpr int LOG_DATA_TILES_PER_BLOCK = 10;
  constexpr int THREAD_TILES_PER_BLOCK = 32;

  // TODO: make some of these kernel arguments
  const int lane_in_tile = threadIdx.x & 7;
  const int tile_id = threadIdx.x >> LOG_DATA_TILE_SIZE;
  const int tile_gmem_stride = 1 << (log_n - LOG_DATA_TILES_PER_BLOCK);
  const int gmem_block_offset = blockIdx.x << LOG_DATA_TILE_SIZE;
  gmem_in.add_row(gmem_block_offset);
  gmem_out.add_row(gmem_block_offset);

  extern __shared__ bf smem_block[]; // 8192 * 4 bytes

  const bf *twiddles = ab_inv_twiddles_first_10_stages;

  bf vals[VALS_PER_THREAD];

  // "ct" = consecutive tile layout
  // "it" = interleaved tile layout
  const int thread_il_gmem_start = lane_in_tile + tile_id * tile_gmem_stride;
  const int thread_ct_gmem_start = lane_in_tile + tile_id * tile_gmem_stride * THREAD_TILES_PER_BLOCK;
  const int thread_il_smem_start = lane_in_tile + tile_id * TILE_SIZE;
  const int thread_ct_smem_start = lane_in_tile + tile_id * TILE_SIZE * THREAD_TILES_PER_BLOCK;

  const int pipeline_memcpy_thread_il_gmem_start = 4 * (lane_in_tile & 3) + (THREAD_TILES_PER_BLOCK * (lane_in_tile >> 2) + tile_id) * tile_gmem_stride;
  const int pipeline_memcpy_thread_il_smem_start = 4 * (lane_in_tile & 3) + (THREAD_TILES_PER_BLOCK * (lane_in_tile >> 2) + tile_id) * TILE_SIZE;
  const int pipeline_memcpy_thread_ct_smem_start = 4 * lane_in_tile + tile_id * TILE_SIZE * THREAD_TILES_PER_BLOCK;

#pragma unroll 1
  for (int ntt_idx{0}; ntt_idx < num_ntts; ntt_idx++) {
    if (ntt_idx > 0) {
      __pipeline_wait_prior(0);
      __syncwarp(); // necessary because warp prefetches values cooperatively
    }

    if (ntt_idx & 1) {
#pragma unroll
      for (int i{0}, addr{thread_ct_smem_start}; i < 32; i++, addr += TILE_SIZE)
        vals[i] = smem_block[addr]; // read consecutive smem tiles

      reg_exchg_inv<16, 32, 1>(vals, 0, twiddles);
      reg_exchg_inv<8, 16, 2>(vals, 0, twiddles);
      reg_exchg_inv<4, 8, 4>(vals, 0, twiddles);
      reg_exchg_inv<2, 4, 8>(vals, 0, twiddles);
      reg_exchg_inv<1, 2, 16>(vals, 0, twiddles);

#pragma unroll
      for (int i{0}, addr{thread_ct_smem_start}; i < 32; i++, addr += TILE_SIZE)
        smem_block[addr] = vals[i]; // write consecutive smem tiles

      __syncthreads();

#pragma unroll
      for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
        vals[i] = smem_block[addr]; // read interleaved smem tiles

      __syncwarp(); // necessary because warp prefetches values cooperatively

      if (ntt_idx < num_ntts - 1) {
        gmem_in.inc_col();
        const bf *gmem_in_ptr = gmem_in.ptr;
#pragma unroll 2 // helps avoid register spilling, should be fine because prefetches are fire and forget
        for (int i{0}, addr{pipeline_memcpy_thread_il_smem_start}, row{pipeline_memcpy_thread_il_gmem_start};
             i < 8;
             i++, addr += 4 * THREAD_TILES_PER_BLOCK * TILE_SIZE, row += 4 * THREAD_TILES_PER_BLOCK * tile_gmem_stride)
          __pipeline_memcpy_async(smem_block + addr,  gmem_in_ptr + row, 4 * sizeof(bf)); // interleaved gmem tiles to interleaved smem tiles
        __pipeline_commit();
      }

      int tile_exchg_region_offset = tile_id;
      reg_exchg_inv<16, 32, 1>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<8, 16, 2>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<4, 8, 4>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<2, 4, 8>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<1, 2, 16>(vals, tile_exchg_region_offset, twiddles);
    } else {
      if (ntt_idx == 0) {
#pragma unroll
        for (int i{0}, row{thread_il_gmem_start}; i < 32; i++, row += THREAD_TILES_PER_BLOCK * tile_gmem_stride)
          vals[i] = gmem_in.get_at_row(row); // read initial set of interleaved tiles
      } else {
#pragma unroll
         for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
           vals[i] = smem_block[addr]; // read interleaved smem tiles
      }

      reg_exchg_inv<16, 32, 1>(vals, 0, twiddles);
      reg_exchg_inv<8, 16, 2>(vals, 0, twiddles);
      reg_exchg_inv<4, 8, 4>(vals, 0, twiddles);
      reg_exchg_inv<2, 4, 8>(vals, 0, twiddles);
      reg_exchg_inv<1, 2, 16>(vals, 0, twiddles);

#pragma unroll
      for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
        smem_block[addr] = vals[i]; // write interleaved smem tiles

      __syncthreads();

#pragma unroll
      for (int i{0}, addr{thread_ct_smem_start}; i < 32; i++, addr += TILE_SIZE)
        vals[i] = smem_block[addr]; // read consecutive smem tiles

      // __syncthreads();
      __syncwarp(); // necessary because warp prefetches values cooperatively

      if (ntt_idx < num_ntts - 1) {
        gmem_in.inc_col();
        const bf *gmem_in_ptr = gmem_in.ptr;
#pragma unroll 2 // helps avoid register spilling, should be fine because prefetches are fire and forget
        for (int i{0}, addr{pipeline_memcpy_thread_ct_smem_start}, row{pipeline_memcpy_thread_il_gmem_start};
             i < 8;
             i++, addr += 4 * TILE_SIZE, row += 4 * THREAD_TILES_PER_BLOCK * tile_gmem_stride)
          __pipeline_memcpy_async(smem_block + addr,  gmem_in_ptr + row, 4 * sizeof(bf)); // interleaved gmem tiles to consecutive smem tiles
        __pipeline_commit();
      }

      int tile_exchg_region_offset = tile_id;
      reg_exchg_inv<16, 32, 1>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<8, 16, 2>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<4, 8, 4>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<2, 4, 8>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<1, 2, 16>(vals, tile_exchg_region_offset, twiddles);
    }

#pragma unroll
    for (int i{0}, row{thread_ct_gmem_start}; i < 32; i++, row += tile_gmem_stride)
      gmem_out.set_at_row(row, vals[i]); // write consecutive gmem tiles
    gmem_out.inc_col();

    // gmem_in.inc_col();
  }
}

EXTERN __launch_bounds__(640, 1) __global__
    void ab_main_to_monomials_first_10_stages_pc_kernel(bf_matrix_getter<ld_modifier::cg> gmem_in,
                                                        bf_matrix_setter<st_modifier::cg> gmem_out,
                                                        const int log_n,
                                                        const int num_ntts) {
  constexpr int VALS_PER_THREAD = 32;
  constexpr int LOG_DATA_TILE_SIZE = 4;
  constexpr int TILE_SIZE = 1 << LOG_DATA_TILE_SIZE;
  constexpr int LOG_DATA_TILES_PER_BLOCK = 10;
  constexpr int THREAD_TILES_PER_BLOCK = 32;

  // TODO: make some of these kernel arguments
  const int lane_in_tile = threadIdx.x & 15;
  const int tile_id = threadIdx.x >> LOG_DATA_TILE_SIZE;
  const int TILE_GMEM_STRIDE = 1 << (24 - LOG_DATA_TILES_PER_BLOCK);
  const int IL_GMEM_STRIDE = TILE_GMEM_STRIDE * THREAD_TILES_PER_BLOCK;
  const int gmem_block_offset = blockIdx.x << LOG_DATA_TILE_SIZE;
  gmem_in.add_row(gmem_block_offset);
  gmem_out.add_row(gmem_block_offset);

  extern __shared__ uint8_t dynamic_smem[]; // 16384 * 4 bytes
  bf *smem_block = reinterpret_cast<bf *>(dynamic_smem);

  __mbarrier_t *bars = reinterpret_cast<__mbarrier_t *>(dynamic_smem + 16384 * sizeof(bf));
  if (threadIdx.x < 2)
    __mbarrier_init(bars + threadIdx.x, blockDim.x);
  unsigned ptx_bar_all{0}, ptx_bar_count_all{blockDim.x};
  asm volatile("barrier.cta.sync %0, %1;" : "+r"(ptx_bar_all) : "r"(ptx_bar_count_all): "memory");

  // "ct" = consecutive tile layout
  // "it" = interleaved tile layout
  const int thread_il_gmem_start = lane_in_tile + tile_id * TILE_GMEM_STRIDE;
  const int thread_ct_gmem_start = lane_in_tile + tile_id * IL_GMEM_STRIDE;
  const int thread_il_smem_start = lane_in_tile + tile_id * TILE_SIZE;
  const int thread_ct_smem_start = lane_in_tile + tile_id * TILE_SIZE * THREAD_TILES_PER_BLOCK;

  const bool is_producer = threadIdx.x >= 512;

  if (is_producer) {
    const int producer_lane = threadIdx.x - 512;
    constexpr int NUM_PRODUCER_LANES = 128;
    constexpr int PRODUCER_TILES_PER_ITER = NUM_PRODUCER_LANES >> 2;
    constexpr int NUM_PRODUCER_ITERS = 32;
    const int producer_start_tile = producer_lane >> 2;
    const int producer_gmem_start = 4 * (producer_lane & 3) + producer_start_tile * TILE_GMEM_STRIDE;
    const int producer_smem_start = 4 * (producer_lane & 3) + producer_start_tile * TILE_SIZE;
    const int producer_gmem_stride = PRODUCER_TILES_PER_ITER * TILE_GMEM_STRIDE;
    const int producer_smem_stride = PRODUCER_TILES_PER_ITER * TILE_SIZE;

#pragma unroll 1
    for (int ntt_idx{0}; ntt_idx < num_ntts - 1; ntt_idx++) {
        __mbarrier_token_t token = __mbarrier_arrive(&bars[0]);
        // while(!__mbarrier_try_wait(&bars[0], token, 1000)) {}
        while(!__mbarrier_test_wait(&bars[0], token)) {}

        gmem_in.inc_col();
        const bf *gmem_in_ptr = gmem_in.ptr;
#pragma unroll 4 // helps avoid register spilling, should be fine because prefetches are fire and forget
        for (int i{0}, addr{producer_smem_start}, row{producer_gmem_start};
             i < NUM_PRODUCER_ITERS;
             i++, addr += producer_smem_stride, row += producer_gmem_stride)
          __pipeline_memcpy_async(smem_block + addr,  gmem_in_ptr + row, 4 * sizeof(bf)); // interleaved gmem tiles to interleaved smem tiles
        __pipeline_arrive_on(&bars[1]);
        (void)__mbarrier_arrive(&bars[1]);
        // while(!__mbarrier_try_wait(&bars[1], token, 1000)) {}
        // while(!__mbarrier_test_wait(&bars[1], token)) {}
    }
  } else {
    bf vals[VALS_PER_THREAD];
    const bf *twiddles = ab_inv_twiddles_first_10_stages;
    unsigned ptx_bar{1}, ptx_bar_count{512};

#pragma unroll 1
    for (int ntt_idx{0}; ntt_idx < num_ntts; ntt_idx++) {
      if (ntt_idx == 0) {
#pragma unroll
        for (int i{0}, row{thread_il_gmem_start}; i < 32; i++, row += IL_GMEM_STRIDE)
          vals[i] = gmem_in.get_at_row(row); // read initial set of interleaved tiles
      } else {
        __mbarrier_token_t token = __mbarrier_arrive(&bars[1]);
        // while(!__mbarrier_try_wait(&bars[1], token, 1000)) {}
        while(!__mbarrier_test_wait(&bars[1], token)) {}
#pragma unroll
        for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
          vals[i] = smem_block[addr]; // read interleaved smem tiles
      }

      reg_exchg_inv<16, 32, 1>(vals, 0, twiddles);
      reg_exchg_inv<8, 16, 2>(vals, 0, twiddles);
      reg_exchg_inv<4, 8, 4>(vals, 0, twiddles);
      reg_exchg_inv<2, 4, 8>(vals, 0, twiddles);
      reg_exchg_inv<1, 2, 16>(vals, 0, twiddles);

#pragma unroll
      for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
        smem_block[addr] = vals[i]; // write interleaved smem tiles

      asm volatile("barrier.cta.sync %0, %1;" : "+r"(ptx_bar) : "r"(ptx_bar_count): "memory");

#pragma unroll
      for (int i{0}, addr{thread_ct_smem_start}; i < 32; i++, addr += TILE_SIZE)
        vals[i] = smem_block[addr]; // read consecutive smem tiles

      if (ntt_idx < num_ntts - 1)
        (void)__mbarrier_arrive(&bars[0]);

      int tile_exchg_region_offset{tile_id};
      reg_exchg_inv<16, 32, 1>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<8, 16, 2>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<4, 8, 4>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<2, 4, 8>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<1, 2, 16>(vals, tile_exchg_region_offset, twiddles);

#pragma unroll
      for (int i{0}, row{thread_ct_gmem_start}; i < 32; i++, row += TILE_GMEM_STRIDE)
        gmem_out.set_at_row(row, vals[i]); // write consecutive gmem tiles

      gmem_out.inc_col();
    }
  }
}

EXTERN __launch_bounds__(512, 1) __global__
    void ab_main_to_monomials_first_10_stages_kernel(bf_matrix_getter<ld_modifier::cg> gmem_in,
                                                      bf_matrix_setter<st_modifier::cg> gmem_out,
                                                      const int log_n,
                                                      const int num_ntts) {
  constexpr int VALS_PER_THREAD = 32;
  constexpr int LOG_DATA_TILE_SIZE = 4;
  constexpr int TILE_SIZE = 1 << LOG_DATA_TILE_SIZE;
  constexpr int LOG_DATA_TILES_PER_BLOCK = 10;
  constexpr int THREAD_TILES_PER_BLOCK = 32;

  // TODO: make some of these kernel arguments
  const int lane_in_tile = threadIdx.x & 15;
  const int tile_id = threadIdx.x >> LOG_DATA_TILE_SIZE;
  const int TILE_GMEM_STRIDE = 1 << (24 - LOG_DATA_TILES_PER_BLOCK);
  const int IL_GMEM_STRIDE = TILE_GMEM_STRIDE * THREAD_TILES_PER_BLOCK;
  const int gmem_block_offset = blockIdx.x << LOG_DATA_TILE_SIZE;
  gmem_in.add_row(gmem_block_offset);
  gmem_out.add_row(gmem_block_offset);

  extern __shared__ bf smem_block[]; // 16384 * 4 bytes

  const bf *twiddles = ab_inv_twiddles_first_10_stages;

  bf vals[VALS_PER_THREAD];

  // "ct" = consecutive tile layout
  // "it" = interleaved tile layout
  const int thread_il_gmem_start = lane_in_tile + tile_id * TILE_GMEM_STRIDE;
  const int thread_ct_gmem_start = lane_in_tile + tile_id * TILE_GMEM_STRIDE * THREAD_TILES_PER_BLOCK;
  const int thread_il_smem_start = lane_in_tile + tile_id * TILE_SIZE;
  const int thread_ct_smem_start = lane_in_tile + tile_id * TILE_SIZE * THREAD_TILES_PER_BLOCK;

  const int pipeline_memcpy_thread_il_gmem_start = 4 * (lane_in_tile & 3) + (THREAD_TILES_PER_BLOCK * (lane_in_tile >> 2) + tile_id) * TILE_GMEM_STRIDE;
  const int pipeline_memcpy_thread_il_smem_start = 4 * (lane_in_tile & 3) + (THREAD_TILES_PER_BLOCK * (lane_in_tile >> 2) + tile_id) * TILE_SIZE;
  const int pipeline_memcpy_thread_ct_smem_start = 4 * lane_in_tile + tile_id * TILE_SIZE * THREAD_TILES_PER_BLOCK;

#pragma unroll 1
  for (int ntt_idx{0}; ntt_idx < num_ntts; ntt_idx++) {
    if (ntt_idx > 0) {
      __pipeline_wait_prior(0);
      __syncwarp(); // necessary because warp prefetches values cooperatively
    }

    if (ntt_idx & 1) {
#pragma unroll
      for (int i{0}, addr{thread_ct_smem_start}; i < 32; i++, addr += TILE_SIZE)
        vals[i] = smem_block[addr]; // read consecutive smem tiles

      reg_exchg_inv<16, 32, 1>(vals, 0, twiddles);
      reg_exchg_inv<8, 16, 2>(vals, 0, twiddles);
      reg_exchg_inv<4, 8, 4>(vals, 0, twiddles);
      reg_exchg_inv<2, 4, 8>(vals, 0, twiddles);
      reg_exchg_inv<1, 2, 16>(vals, 0, twiddles);

#pragma unroll
      for (int i{0}, addr{thread_ct_smem_start}; i < 32; i++, addr += TILE_SIZE)
        smem_block[addr] = vals[i]; // write consecutive smem tiles

      __syncthreads();

#pragma unroll
      for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
        vals[i] = smem_block[addr]; // read interleaved smem tiles

      __syncwarp(); // necessary because warp prefetches values cooperatively

      if (ntt_idx < num_ntts - 1) {
        gmem_in.inc_col();
        const bf *gmem_in_ptr = gmem_in.ptr;
#pragma unroll 4 // helps avoid register spilling, should be fine because prefetches are fire and forget
        for (int i{0}, addr{pipeline_memcpy_thread_il_smem_start}, row{pipeline_memcpy_thread_il_gmem_start};
             i < 8;
             i++, addr += 4 * THREAD_TILES_PER_BLOCK * TILE_SIZE, row += 4 * IL_GMEM_STRIDE)
          __pipeline_memcpy_async(smem_block + addr,  gmem_in_ptr + row, 4 * sizeof(bf)); // interleaved gmem tiles to interleaved smem tiles
        __pipeline_commit();
      }

      int tile_exchg_region_offset{tile_id};
      reg_exchg_inv<16, 32, 1>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<8, 16, 2>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<4, 8, 4>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<2, 4, 8>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<1, 2, 16>(vals, tile_exchg_region_offset, twiddles);
    } else {
      if (ntt_idx == 0) {
// #pragma unroll
//         for (int i{0}, row{thread_il_gmem_start}; i < 32; i++, row += THREAD_TILES_PER_BLOCK * TILE_GMEM_STRIDE)
//           vals[i] = gmem_in.get_at_row(row); // read initial set of interleaved tiles
        const bf *gmem_in_ptr = gmem_in.ptr;
#pragma unroll 4 // helps avoid register spilling, should be fine because prefetches are fire and forget
        for (int i{0}, addr{pipeline_memcpy_thread_il_smem_start}, row{pipeline_memcpy_thread_il_gmem_start};
             i < 8;
             i++, addr += 4 * THREAD_TILES_PER_BLOCK * TILE_SIZE, row += 4 * IL_GMEM_STRIDE)
          __pipeline_memcpy_async(smem_block + addr,  gmem_in_ptr + row, 4 * sizeof(bf)); // interleaved gmem tiles to interleaved smem tiles
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncwarp(); // necessary because warp prefetches values cooperatively
#pragma unroll
        for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
          vals[i] = smem_block[addr]; // read interleaved smem tiles
      } else {
#pragma unroll
         for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
           vals[i] = smem_block[addr]; // read interleaved smem tiles
      }

      reg_exchg_inv<16, 32, 1>(vals, 0, twiddles);
      reg_exchg_inv<8, 16, 2>(vals, 0, twiddles);
      reg_exchg_inv<4, 8, 4>(vals, 0, twiddles);
      reg_exchg_inv<2, 4, 8>(vals, 0, twiddles);
      reg_exchg_inv<1, 2, 16>(vals, 0, twiddles);

#pragma unroll
      for (int i{0}, addr{thread_il_smem_start}; i < 32; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
        smem_block[addr] = vals[i]; // write interleaved smem tiles

      __syncthreads();

#pragma unroll
      for (int i{0}, addr{thread_ct_smem_start}; i < 32; i++, addr += TILE_SIZE)
        vals[i] = smem_block[addr]; // read consecutive smem tiles

      __syncwarp(); // necessary because warp prefetches values cooperatively

      if (ntt_idx < num_ntts - 1) {
        gmem_in.inc_col();
        const bf *gmem_in_ptr = gmem_in.ptr;
#pragma unroll 4 // helps avoid register spilling, should be fine because prefetches are fire and forget
        for (int i{0}, addr{pipeline_memcpy_thread_ct_smem_start}, row{pipeline_memcpy_thread_il_gmem_start};
             i < 8;
             i++, addr += 4 * TILE_SIZE, row += 4 * IL_GMEM_STRIDE)
          __pipeline_memcpy_async(smem_block + addr,  gmem_in_ptr + row, 4 * sizeof(bf)); // interleaved gmem tiles to consecutive smem tiles
        __pipeline_commit();
      }

      int tile_exchg_region_offset{tile_id};
      reg_exchg_inv<16, 32, 1>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<8, 16, 2>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<4, 8, 4>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<2, 4, 8>(vals, tile_exchg_region_offset, twiddles); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<1, 2, 16>(vals, tile_exchg_region_offset, twiddles);
    }

#pragma unroll
    for (int i{0}, row{thread_ct_gmem_start}; i < 32; i++, row += TILE_GMEM_STRIDE)
      gmem_out.set_at_row(row, vals[i]); // write consecutive gmem tiles
    gmem_out.inc_col();
  }
}

EXTERN __launch_bounds__(512, 1) __global__
    void ab_main_to_coset_middle_28_stages_megakernel(bf_matrix_getter<ld_modifier::cs> gmem_in,
                                                      // bf_matrix_setter<st_modifier::cs> gmem_monomials_out,
                                                      bf_matrix_setter<st_modifier::cs> gmem_out,
                                                      // const bool materialize_monomials,
                                                      const int num_ntts) {
  constexpr int WARP_SIZE = 32;
  constexpr int VALS_PER_THREAD = 32;
  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int VALS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE * VALS_PER_THREAD; // 16384

  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int tile_stride = VALS_PER_BLOCK >> 4;
  const int gmem_block_offset = blockIdx.x * VALS_PER_BLOCK;
  const int thread_start = 64 * warp_id + lane_id;
  const int pipeline_memcpy_thread_start = 64 * warp_id + (lane_id >> 1) * tile_stride + 4 * (lane_id & 15);
  gmem_in.add_row(gmem_block_offset);
  // gmem_monomials_out.add_row(gmem_block_offset + warp_id * 1024 + lane_id * 32);
  gmem_out.add_row(gmem_block_offset);

  extern __shared__ bf smem_block[]; // 16384 * 4 bytes
  bf *smem_warp = smem_block + warp_id * 1024;

  bf *twiddles = smem_block;

  bf vals[VALS_PER_THREAD];

#pragma unroll 1
  for (int ntt_idx = 0; ntt_idx < num_ntts; ntt_idx++) {
    if (ntt_idx == 0) {
#pragma unroll
      for (int i{0}, row{thread_start}; i < 32; i += 2, row += tile_stride) {
        vals[i] = gmem_in.get_at_row(row);
        vals[i + 1] = gmem_in.get_at_row(row + 32);
      }
    } else {
      __pipeline_wait_prior(0);
      __syncwarp(); // necessary because warp prefetches values cooperatively
#pragma unroll
      for (int i{0}, row{thread_start}; i < 32; i += 2, row += tile_stride) {
        vals[i] = smem_block[row];
        vals[i + 1] = smem_block[row + 32];
      }
    }
  
    int block_exchg_region_offset = blockIdx.x;
    reg_exchg_inv<16, 32, 1>(vals, block_exchg_region_offset, twiddles); block_exchg_region_offset <<= 1;
    reg_exchg_inv<8, 16, 2>(vals, block_exchg_region_offset, twiddles); block_exchg_region_offset <<= 1;
    reg_exchg_inv<4, 8, 4>(vals, block_exchg_region_offset, twiddles); block_exchg_region_offset <<= 1;
    reg_exchg_inv<2, 4, 8>(vals, block_exchg_region_offset, twiddles);
  
#pragma unroll
    for (int i{0}, row{thread_start}; i < 32; i += 2, row += tile_stride) {
      smem_block[row] = vals[i] = smem_block[row];
      smem_block[row + 32] = vals[i + 1];
    }
  
    __syncthreads(); // all-to-all, so ptx barriers are unlikely to help
  
#pragma unroll
    for (int i{0}, row{lane_id}; i < 32; i++, row += 32)
      vals[i] = smem_warp[row];
  
    int warp_exchg_region_offset = block_exchg_region_offset + warp_id;
    reg_exchg_inv<16, 32, 1>(vals, warp_exchg_region_offset, twiddles); warp_exchg_region_offset <<= 1;
    reg_exchg_inv<8, 16, 2>(vals, warp_exchg_region_offset, twiddles); warp_exchg_region_offset <<= 1;
    reg_exchg_inv<4, 8, 4>(vals, warp_exchg_region_offset, twiddles); warp_exchg_region_offset <<= 1;
    reg_exchg_inv<2, 4, 8>(vals, warp_exchg_region_offset, twiddles); warp_exchg_region_offset <<= 1;
    reg_exchg_inv<1, 2, 16>(vals, warp_exchg_region_offset, twiddles);
  
    __syncwarp();
#pragma unroll
    for (int y = 0; y < 32; y++)
      smem_warp[xy_to_linear(lane_id, y)] = vals[y];
    __syncwarp();
#pragma unroll
    for (int x = 0; x < 32; x++)
      vals[x] = smem_warp[xy_to_linear(x, lane_id)];
  
    int thread_exchg_region_offset = warp_exchg_region_offset + lane_id;
    reg_exchg_inv<16, 32, 1>(vals, thread_exchg_region_offset, twiddles); thread_exchg_region_offset <<= 1;
    reg_exchg_inv<8, 16, 2>(vals, thread_exchg_region_offset, twiddles); thread_exchg_region_offset <<= 1;
    reg_exchg_inv<4, 8, 4>(vals, thread_exchg_region_offset, twiddles); thread_exchg_region_offset <<= 1;
    reg_exchg_inv<2, 4, 8>(vals, thread_exchg_region_offset, twiddles); thread_exchg_region_offset <<= 1;
    reg_exchg_inv<1, 2, 16>(vals, thread_exchg_region_offset, twiddles);
  
  //   if (materialize_monomials) {
  //     // uncoalesced, but vectorized and should fire off quickly
  //     uint4 *gmem_monomials_out_ptr = reinterpret_cast<uint4 *>(gmem_monomials_out.ptr);
  // #pragma unroll
  //     for (int i{0}; i < 32; i += 4, gmem_monomials_out_ptr++)
  //       *gmem_monomials_out_ptr = {vals[i].limb, vals[i + 1].limb, vals[i + 2].limb, vals[i + 3].limb};
  //   }

    // apply coset prefactors here, once decided
  
    reg_exchg_fwd<1, 2, 16>(vals, thread_exchg_region_offset, twiddles); thread_exchg_region_offset >>= 1;
    reg_exchg_fwd<2, 4, 8>(vals, thread_exchg_region_offset, twiddles); thread_exchg_region_offset >>= 1;
    reg_exchg_fwd<4, 8, 4>(vals, thread_exchg_region_offset, twiddles); thread_exchg_region_offset >>= 1;
    reg_exchg_fwd<8, 16, 2>(vals, thread_exchg_region_offset, twiddles); thread_exchg_region_offset >>= 1;
    reg_exchg_fwd<16, 32, 1>(vals, thread_exchg_region_offset, twiddles);
  
#pragma unroll
    for (int x = 0; x < 32; x++)
      smem_warp[xy_to_linear(x, lane_id)] = vals[x];
    __syncwarp();
#pragma unroll
    for (int y = 0; y < 32; y++)
      vals[y] = smem_warp[xy_to_linear(lane_id, y)];
    __syncwarp();
  
    reg_exchg_fwd<1, 2, 16>(vals, warp_exchg_region_offset, twiddles); warp_exchg_region_offset >>= 1;
    reg_exchg_fwd<2, 4, 8>(vals, warp_exchg_region_offset, twiddles); warp_exchg_region_offset >>= 1;
    reg_exchg_fwd<4, 8, 4>(vals, warp_exchg_region_offset, twiddles); warp_exchg_region_offset >>= 1;
    reg_exchg_fwd<8, 16, 2>(vals, warp_exchg_region_offset, twiddles); warp_exchg_region_offset >>= 1;
    reg_exchg_fwd<16, 32, 1>(vals, warp_exchg_region_offset, twiddles);
  
#pragma unroll
    for (int i{0}, row{lane_id}; i < 32; i++, row += 32)
      smem_warp[row] = vals[i];
  
    __syncthreads(); // all-to-all, so ptx barriers are unlikely to help
  
#pragma unroll
    for (int i{0}, row{thread_start}; i < 32; i += 2, row += tile_stride) {
      vals[i] = smem_block[row];
      vals[i + 1] = smem_block[row + 32];
    }

    __syncwarp(); // necessary because warp prefetches values cooperatively
  
    if (ntt_idx < num_ntts - 1) {
      gmem_in.inc_col();
      const bf *gmem_in_ptr = gmem_in.ptr;
      // 16-byte LDGSTS cooperatively within warp
#pragma unroll
      for (int i{0}, row{pipeline_memcpy_thread_start}; i < 8; i++, row += 2 * tile_stride)
        __pipeline_memcpy_async(smem_block + row, gmem_in_ptr + row, 4 * sizeof(bf));
      __pipeline_commit();
    }
  
    reg_exchg_fwd<2, 4, 8>(vals, block_exchg_region_offset, twiddles); block_exchg_region_offset >>= 1;
    reg_exchg_fwd<4, 8, 4>(vals, block_exchg_region_offset, twiddles); block_exchg_region_offset >>= 1;
    reg_exchg_fwd<8, 16, 2>(vals, block_exchg_region_offset, twiddles); block_exchg_region_offset >>= 1;
    reg_exchg_fwd<16, 32, 1>(vals, block_exchg_region_offset, twiddles);
  
#pragma unroll
    for (int i{0}, row{thread_start}; i < 32; i += 2, row += tile_stride) {
      gmem_out.set_at_row(row, vals[i]);
      gmem_out.set_at_row(row + 32, vals[i + 1]);
    }

    // gmem_monomials_out.inc_col();
    gmem_out.inc_col();
  }
}

// barrier stuff for reference in case i need it
// using barrier_t = cuda::barrier<cuda::thread_scope_block>;
// __shared__ barrier_t bar;
// int parity = 0;
// if (block.thread_rank() == 0)
//   init(&bar, blockDim.x);
// ...
// In the megakernel pattern, when we move to the next NTT the current warp has already "reserved"
// the smem it's about to write, so no sync is needed.
// if (ntt > 0) {
//   while (!cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bar), parity)) {}
//   parity ^= 1;
// }
// ...
// here's where we could put a barrier to protect the above reads against the next ntt's fetches.
// But in the megakernel pattern, the upcoming fetches touch memory in each thread in exactly the same
// pattern as the above reads, so no barrier is needed.
// (void)cuda::ptx::mbarrier_arrive(cuda::device::barrier_native_handle(bar));

} // namespace airbender::ntt
