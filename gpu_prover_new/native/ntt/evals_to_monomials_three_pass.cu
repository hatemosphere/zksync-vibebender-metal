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

} // namespace airbender::ntt
