#include "ntt.cuh"

namespace airbender::ntt {

EXTERN __launch_bounds__(512, 2) __global__
    void ab_main_to_monomials_nonfinal_8_stages_kernel(bf_matrix_getter<ld_modifier::cg> gmem_in,
                                                       bf_matrix_setter<st_modifier::cg> gmem_out,
                                                       const int log_n,
                                                       const int start_stage) {
  constexpr int VALS_PER_THREAD = 16;
  constexpr int LOG_DATA_TILE_SIZE = 5;
  constexpr int TILE_SIZE = 1 << LOG_DATA_TILE_SIZE;
  constexpr int LOG_DATA_TILES_PER_BLOCK = 8;
  constexpr int THREAD_TILES_PER_BLOCK = 16;

  const int lane_in_tile = threadIdx.x & 31;
  const int tile_id = threadIdx.x >> LOG_DATA_TILE_SIZE;

  const int exchg_region_size = 1 << (log_n - start_stage);
  const int tile_gmem_stride = exchg_region_size >> LOG_DATA_TILES_PER_BLOCK;
  const int interleaved_gmem_stride = tile_gmem_stride * THREAD_TILES_PER_BLOCK;

  // Reversed block indexing for the middle kernel, to help L2 hits
  const int alternating_block_idx_x = (start_stage == 0) ? blockIdx.x : (gridDim.x - 1 - blockIdx.x);
  const int alternating_block_idx_y = (start_stage == 0) ? blockIdx.y : (gridDim.y - 1 - blockIdx.y);
  const int gmem_block_offset = alternating_block_idx_y * exchg_region_size + (alternating_block_idx_x << LOG_DATA_TILE_SIZE);
  gmem_in.add_row(gmem_block_offset);
  gmem_out.add_row(gmem_block_offset);

  // __shared__ bf smem_block[49152];
  __shared__ bf smem_block[8192];

  bf vals[VALS_PER_THREAD];

  // "ct" = consecutive tile layout
  // "it" = interleaved tile layout
  const int thread_il_gmem_start = lane_in_tile + tile_id * tile_gmem_stride;
  const int thread_ct_gmem_start = lane_in_tile + tile_id * interleaved_gmem_stride;
  const int thread_il_smem_start = lane_in_tile + tile_id * TILE_SIZE;
  const int thread_ct_smem_start = lane_in_tile + tile_id * TILE_SIZE * THREAD_TILES_PER_BLOCK;

#pragma unroll
  for (int i{0}, addr{thread_il_gmem_start}; i < VALS_PER_THREAD; i++, addr += interleaved_gmem_stride)
    vals[i] = gmem_in.get_at_row(addr);

  int block_exchg_region_offset = alternating_block_idx_y;
  if (start_stage == 0) {
    reg_exchg_inv<8, 16, 1>(vals, block_exchg_region_offset); block_exchg_region_offset <<= 1;
    reg_exchg_inv<4, 8, 2>(vals, block_exchg_region_offset); block_exchg_region_offset <<= 1;
    reg_exchg_inv<2, 4, 4>(vals, block_exchg_region_offset); block_exchg_region_offset <<= 1;
    reg_exchg_inv<1, 2, 8>(vals, block_exchg_region_offset); block_exchg_region_offset <<= 1;
  } else {
    reg_exchg_cmem_twiddles_inv<8, 16, 1>(vals, block_exchg_region_offset); block_exchg_region_offset <<= 1;
    reg_exchg_cmem_twiddles_inv<4, 8, 2>(vals, block_exchg_region_offset); block_exchg_region_offset <<= 1;
    reg_exchg_cmem_twiddles_inv<2, 4, 4>(vals, block_exchg_region_offset); block_exchg_region_offset <<= 1;
    reg_exchg_cmem_twiddles_inv<1, 2, 8>(vals, block_exchg_region_offset); block_exchg_region_offset <<= 1;
  }

#pragma unroll
    for (int i{0}, addr{thread_il_smem_start}; i < VALS_PER_THREAD; i++, addr += TILE_SIZE * THREAD_TILES_PER_BLOCK)
      smem_block[addr] = vals[i]; // write interleaved smem tiles

    __syncthreads();

#pragma unroll
    for (int i{0}, addr{thread_ct_smem_start}; i < VALS_PER_THREAD; i++, addr += TILE_SIZE)
      vals[i] = smem_block[addr]; // read consecutive smem tiles

    int tile_exchg_region_offset = block_exchg_region_offset + tile_id;
    if (start_stage == 0) {
      reg_exchg_inv<8, 16, 1>(vals, tile_exchg_region_offset); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<4, 8, 2>(vals, tile_exchg_region_offset); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<2, 4, 4>(vals, tile_exchg_region_offset); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<1, 2, 8>(vals, tile_exchg_region_offset);
    } else {
      reg_exchg_cmem_twiddles_inv<8, 16, 1>(vals, tile_exchg_region_offset); tile_exchg_region_offset <<= 1;
      reg_exchg_cmem_twiddles_inv<4, 8, 2>(vals, tile_exchg_region_offset); tile_exchg_region_offset <<= 1;
      reg_exchg_cmem_twiddles_inv<2, 4, 4>(vals, tile_exchg_region_offset); tile_exchg_region_offset <<= 1;
      reg_exchg_cmem_twiddles_inv<1, 2, 8>(vals, tile_exchg_region_offset);
    }

#pragma unroll
    for (int i{0}, row{thread_ct_gmem_start}; i < VALS_PER_THREAD; i++, row += tile_gmem_stride)
      gmem_out.set_at_row(row, vals[i]); // write consecutive gmem tiles
}

EXTERN __launch_bounds__(512, 2) __global__
    void ab_main_to_monomials_final_8_stages_kernel(bf_matrix_getter<ld_modifier::cg> gmem_in,
                                                    bf_matrix_setter<st_modifier::cg> gmem_out,
                                                    const int log_n) {
  constexpr int WARP_SIZE = 32;
  constexpr int VALS_PER_THREAD = 8;
  constexpr int VALS_PER_WARP = WARP_SIZE * VALS_PER_THREAD;
  constexpr int WARPS_PER_BLOCK = 16;
  constexpr int VALS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE * VALS_PER_THREAD; // 8192

  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int thread_start = lane_id + warp_id * VALS_PER_WARP;
  const int pipeline_memcpy_start = 4 * threadIdx.x;
  const int pipeline_memcpy_stride = 4 * blockDim.x;
  const int gmem_block_offset = blockIdx.x * VALS_PER_BLOCK;
  gmem_in.add_row(gmem_block_offset);
  // gmem_out.add_row(gmem_block_offset + warp_id * VALS_PER_WARP);
  gmem_out.add_row(gmem_block_offset);

  __shared__ bf smem_block[8192]; // 4096 vals, 4096 coarse twiddles
  bf *smem_warp = smem_block + warp_id * VALS_PER_WARP;
  bf *smem_twiddles = smem_block + VALS_PER_BLOCK;
  constexpr bf *cmem_twiddles = ab_inv_cmem_twiddles_finest_11;

  bf vals[VALS_PER_THREAD];

  // Cooperatively fetch fine gmem twiddle powers used by last 5 stages.
  // The gmem layout is already swizzled, so it's a linear copy and we can vectorize :)
  // The cooperative twiddle fetch is actually the only reason this kernel needs a __syncthreads().
  // Unlike the 2-pass kernel, there's no compute overlap here, but gmem->smem is preferable to gmem->register->smem.
#pragma unroll
  for (int i{0}, addr{pipeline_memcpy_start}; i < 2; i++, addr += pipeline_memcpy_stride)
      __pipeline_memcpy_async(smem_twiddles + addr, ab_inv_gmem_twiddles_coarse + addr, 4 * sizeof(bf));
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

#pragma unroll
  for (int i{0}, row{thread_start}; i < VALS_PER_THREAD; i++, row += WARP_SIZE)
    vals[i] = gmem_in.get_at_row(row);

  int warp_exchg_region_offset = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  reg_exchg_cmem_smem_twiddles_inv<EightStages, 4, 8, 1, cmem_twiddles>(vals, warp_exchg_region_offset, smem_twiddles); warp_exchg_region_offset <<= 1;
  reg_exchg_cmem_smem_twiddles_inv<EightStages, 2, 4, 2, cmem_twiddles>(vals, warp_exchg_region_offset, smem_twiddles); warp_exchg_region_offset <<= 1;
  reg_exchg_cmem_smem_twiddles_inv<EightStages, 1, 2, 4, cmem_twiddles>(vals, warp_exchg_region_offset, smem_twiddles); warp_exchg_region_offset <<= 1;

//   constexpr unsigned mask = 0xffffffff;
//   constexpr int laneMask = 16;
// #pragma unroll
//   for (int i{0}; i < 8; i += 2) {
//     if (lane_id & 16)
//       vals[i].limb = __shfl_xor_sync(mask, vals[i].limb, laneMask);
//     else
//       vals[i + 1].limb = __shfl_xor_sync(mask, vals[i + 1].limb, laneMask);
//   }
  reg_exchg_cmem_smem_twiddles_inv<EightStages, 1, 2, 4, cmem_twiddles>(vals, warp_exchg_region_offset, smem_twiddles); warp_exchg_region_offset <<= 1;
  reg_exchg_cmem_smem_twiddles_inv<EightStages, 1, 2, 4, cmem_twiddles>(vals, warp_exchg_region_offset, smem_twiddles); warp_exchg_region_offset <<= 1;
  reg_exchg_cmem_smem_twiddles_inv<EightStages, 1, 2, 4, cmem_twiddles>(vals, warp_exchg_region_offset, smem_twiddles); warp_exchg_region_offset <<= 1;
  reg_exchg_cmem_smem_twiddles_inv<EightStages, 1, 2, 4, cmem_twiddles>(vals, warp_exchg_region_offset, smem_twiddles); warp_exchg_region_offset <<= 1;
  reg_exchg_cmem_smem_twiddles_inv<EightStages, 1, 2, 4, cmem_twiddles>(vals, warp_exchg_region_offset, smem_twiddles); warp_exchg_region_offset <<= 1;

#pragma unroll
  for (int i{0}, row{thread_start}; i < VALS_PER_THREAD; i++, row += WARP_SIZE)
    gmem_out.set_at_row(row, vals[i]);

//   // TODO: consider shfl pattern here instead
//   __syncwarp();
// #pragma unroll
//   for (int y = 0; y < VALS_PER_THREAD; y++)
//     smem_warp[xy_to_swizzled(lane_id, y)] = vals[y];
//   __syncwarp();
// #pragma unroll
//   for (int x = 0; x < VALS_PER_THREAD; x++)
//     vals[x] = smem_warp[xy_to_swizzled(x, lane_id)];
// 
//   // int thread_exchg_region_offset = warp_exchg_region_offset + lane_id;
//   // reg_exchg_cmem_smem_twiddles_inv<EightStages, 8, 16, 1, cmem_twiddles>(vals, thread_exchg_region_offset, smem_twiddles); thread_exchg_region_offset <<= 1;
//   // reg_exchg_cmem_smem_twiddles_inv<EightStages, 4, 8, 2, cmem_twiddles>(vals, thread_exchg_region_offset, smem_twiddles); thread_exchg_region_offset <<= 1;
//   // reg_exchg_cmem_smem_twiddles_inv<EightStages, 2, 4, 4, cmem_twiddles>(vals, thread_exchg_region_offset, smem_twiddles); thread_exchg_region_offset <<= 1;
//   // reg_exchg_cmem_smem_twiddles_inv<EightStages, 1, 2, 8, cmem_twiddles>(vals, thread_exchg_region_offset, smem_twiddles);
// 
//   // uncoalesced, but vectorized and should fire off quickly
//   uint4 *gmem_monomials_out_ptr = reinterpret_cast<uint4 *>(gmem_out.ptr + VALS_PER_THREAD * lane_id);
// #pragma unroll
//   for (int i{0}; i < VALS_PER_THREAD; i += 4, gmem_monomials_out_ptr++)
//     *gmem_monomials_out_ptr = {vals[i].limb, vals[i + 1].limb, vals[i + 2].limb, vals[i + 3].limb};
}

} // namespace airbender::ntt
