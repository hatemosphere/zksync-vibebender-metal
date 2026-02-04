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
// template <bool inverse> DEVICE_FORCEINLINE bf get_twiddle_bf(const unsigned i) {
//   const powers_data_2_layer_bf &data = inverse ? ab_powers_data_w_inv_bitrev_for_ntt_bf : ab_powers_data_w_bitrev_for_ntt_bf;
//   unsigned fine_idx = (i >> data.coarse.log_count) & data.fine.mask;
//   unsigned coarse_idx = i & data.coarse.mask;
//   auto coarse = memory::load_ca(data.coarse.values + coarse_idx);
//   if (fine_idx == 0)
//     return coarse;
//   auto fine = memory::load_ca(data.fine.values + fine_idx);
//   return bf::mul(fine, coarse);
// }

DEVICE_FORCEINLINE void exchg_dit(bf &a, bf &b, const bf &twiddle) {
  b = bf::mul(b, twiddle);
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
DEVICE_FORCEINLINE void reg_exchg_inv(bf *vals, const int exchg_region_offset, const bf *twiddle_proxy) {
#pragma unroll
  for (int region{0}; region < NUM_REGIONS; region++) {
    const bf twiddle = twiddle_proxy[exchg_region_offset + region];
    const int region_offset = region * REGION_SIZE;
#pragma unroll
    for (int lane_in_region{0}; lane_in_region < STRIDE; lane_in_region++) {
      const int i = region_offset + lane_in_region;
      exchg_dif(vals[i], vals[i + STRIDE], twiddle);
    }
  }
}

template <int STRIDE, int REGION_SIZE, int NUM_REGIONS>
DEVICE_FORCEINLINE void reg_exchg_fwd(bf *vals, const int exchg_region_offset, const bf *twiddles_proxy) {
#pragma unroll
  for (int region{0}; region < NUM_REGIONS; region++) {
    const bf twiddle = twiddles_proxy[exchg_region_offset + region];
    const int region_offset = region * REGION_SIZE;
#pragma unroll
    for (int lane_in_region{0}; lane_in_region < STRIDE; lane_in_region++) {
      const int i = region_offset + lane_in_region;
      exchg_dit(vals[i], vals[i + STRIDE], twiddle);
    }
  }
}

EXTERN __launch_bounds__(512, 1) __global__
    void ab_main_to_monomials_first_10_stages_kernel(bf_matrix_getter<ld_modifier::cs> gmem_in,
                                                      // bf_matrix_setter<st_modifier::cs> gmem_monomials_out,
                                                      bf_matrix_setter<st_modifier::cs> gmem_out,
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
  const int tile_gmem_stride = 1 << (log_n - LOG_DATA_TILES_PER_BLOCK);
  const int gmem_block_offset = blockIdx.x << LOG_DATA_TILE_SIZE;
  gmem_in.add_row(gmem_block_offset);
  gmem_out.add_row(gmem_block_offset);

  extern __shared__ bf smem_block[]; // 16384 * 4 bytes

  const bf *twiddles_proxy = smem_block;

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

      reg_exchg_inv<16, 32, 1>(vals, 0, twiddles_proxy);
      reg_exchg_inv<8, 16, 2>(vals, 0, twiddles_proxy);
      reg_exchg_inv<4, 8, 4>(vals, 0, twiddles_proxy);
      reg_exchg_inv<2, 4, 8>(vals, 0, twiddles_proxy);
      reg_exchg_inv<1, 2, 16>(vals, 0, twiddles_proxy);

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
             i++, addr += 4 * THREAD_TILES_PER_BLOCK * TILE_SIZE, row += 4 * THREAD_TILES_PER_BLOCK * tile_gmem_stride)
          __pipeline_memcpy_async(smem_block + addr,  gmem_in_ptr + row, 4 * sizeof(bf)); // interleaved gmem tiles to interleaved smem tiles
        __pipeline_commit();
      }

      int tile_exchg_region_offset = tile_id;
      reg_exchg_inv<16, 32, 1>(vals, tile_exchg_region_offset, twiddles_proxy); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<8, 16, 2>(vals, tile_exchg_region_offset, twiddles_proxy); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<4, 8, 4>(vals, tile_exchg_region_offset, twiddles_proxy); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<2, 4, 8>(vals, tile_exchg_region_offset, twiddles_proxy); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<1, 2, 16>(vals, tile_exchg_region_offset, twiddles_proxy);
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

      reg_exchg_inv<16, 32, 1>(vals, 0, twiddles_proxy);
      reg_exchg_inv<8, 16, 2>(vals, 0, twiddles_proxy);
      reg_exchg_inv<4, 8, 4>(vals, 0, twiddles_proxy);
      reg_exchg_inv<2, 4, 8>(vals, 0, twiddles_proxy);
      reg_exchg_inv<1, 2, 16>(vals, 0, twiddles_proxy);

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
             i++, addr += 4 * TILE_SIZE, row += 4 * THREAD_TILES_PER_BLOCK * tile_gmem_stride)
          __pipeline_memcpy_async(smem_block + addr,  gmem_in_ptr + row, 4 * sizeof(bf)); // interleaved gmem tiles to consecutive smem tiles
        __pipeline_commit();
      }

      int tile_exchg_region_offset = tile_id;
      reg_exchg_inv<16, 32, 1>(vals, tile_exchg_region_offset, twiddles_proxy); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<8, 16, 2>(vals, tile_exchg_region_offset, twiddles_proxy); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<4, 8, 4>(vals, tile_exchg_region_offset, twiddles_proxy); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<2, 4, 8>(vals, tile_exchg_region_offset, twiddles_proxy); tile_exchg_region_offset <<= 1;
      reg_exchg_inv<1, 2, 16>(vals, tile_exchg_region_offset, twiddles_proxy);
    }

#pragma unroll
    for (int i{0}, row{thread_ct_gmem_start}; i < 32; i++, row += tile_gmem_stride)
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

  bf *twiddles_proxy = smem_block;

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
    reg_exchg_inv<16, 32, 1>(vals, block_exchg_region_offset, twiddles_proxy); block_exchg_region_offset <<= 1;
    reg_exchg_inv<8, 16, 2>(vals, block_exchg_region_offset, twiddles_proxy); block_exchg_region_offset <<= 1;
    reg_exchg_inv<4, 8, 4>(vals, block_exchg_region_offset, twiddles_proxy); block_exchg_region_offset <<= 1;
    reg_exchg_inv<2, 4, 8>(vals, block_exchg_region_offset, twiddles_proxy);
  
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
    reg_exchg_inv<16, 32, 1>(vals, warp_exchg_region_offset, twiddles_proxy); warp_exchg_region_offset <<= 1;
    reg_exchg_inv<8, 16, 2>(vals, warp_exchg_region_offset, twiddles_proxy); warp_exchg_region_offset <<= 1;
    reg_exchg_inv<4, 8, 4>(vals, warp_exchg_region_offset, twiddles_proxy); warp_exchg_region_offset <<= 1;
    reg_exchg_inv<2, 4, 8>(vals, warp_exchg_region_offset, twiddles_proxy); warp_exchg_region_offset <<= 1;
    reg_exchg_inv<1, 2, 16>(vals, warp_exchg_region_offset, twiddles_proxy);
  
    __syncwarp();
#pragma unroll
    for (int y = 0; y < 32; y++)
      smem_warp[xy_to_linear(lane_id, y)] = vals[y];
    __syncwarp();
#pragma unroll
    for (int x = 0; x < 32; x++)
      vals[x] = smem_warp[xy_to_linear(x, lane_id)];
  
    int thread_exchg_region_offset = warp_exchg_region_offset + lane_id;
    reg_exchg_inv<16, 32, 1>(vals, thread_exchg_region_offset, twiddles_proxy); thread_exchg_region_offset <<= 1;
    reg_exchg_inv<8, 16, 2>(vals, thread_exchg_region_offset, twiddles_proxy); thread_exchg_region_offset <<= 1;
    reg_exchg_inv<4, 8, 4>(vals, thread_exchg_region_offset, twiddles_proxy); thread_exchg_region_offset <<= 1;
    reg_exchg_inv<2, 4, 8>(vals, thread_exchg_region_offset, twiddles_proxy); thread_exchg_region_offset <<= 1;
    reg_exchg_inv<1, 2, 16>(vals, thread_exchg_region_offset, twiddles_proxy);
  
  //   if (materialize_monomials) {
  //     // uncoalesced, but vectorized and should fire off quickly
  //     uint4 *gmem_monomials_out_ptr = reinterpret_cast<uint4 *>(gmem_monomials_out.ptr);
  // #pragma unroll
  //     for (unsigned i{0}; i < 32; i += 4, gmem_monomials_out_ptr++)
  //       *gmem_monomials_out_ptr = {vals[i].limb, vals[i + 1].limb, vals[i + 2].limb, vals[i + 3].limb};
  //   }

    // apply coset prefactors here, once decided
  
    reg_exchg_fwd<1, 2, 16>(vals, thread_exchg_region_offset, twiddles_proxy); thread_exchg_region_offset >>= 1;
    reg_exchg_fwd<2, 4, 8>(vals, thread_exchg_region_offset, twiddles_proxy); thread_exchg_region_offset >>= 1;
    reg_exchg_fwd<4, 8, 4>(vals, thread_exchg_region_offset, twiddles_proxy); thread_exchg_region_offset >>= 1;
    reg_exchg_fwd<8, 16, 2>(vals, thread_exchg_region_offset, twiddles_proxy); thread_exchg_region_offset >>= 1;
    reg_exchg_fwd<16, 32, 1>(vals, thread_exchg_region_offset, twiddles_proxy);
  
#pragma unroll
    for (int x = 0; x < 32; x++)
      smem_warp[xy_to_linear(x, lane_id)] = vals[x];
    __syncwarp();
#pragma unroll
    for (int y = 0; y < 32; y++)
      vals[y] = smem_warp[xy_to_linear(lane_id, y)];
    __syncwarp();
  
    reg_exchg_fwd<1, 2, 16>(vals, warp_exchg_region_offset, twiddles_proxy); warp_exchg_region_offset >>= 1;
    reg_exchg_fwd<2, 4, 8>(vals, warp_exchg_region_offset, twiddles_proxy); warp_exchg_region_offset >>= 1;
    reg_exchg_fwd<4, 8, 4>(vals, warp_exchg_region_offset, twiddles_proxy); warp_exchg_region_offset >>= 1;
    reg_exchg_fwd<8, 16, 2>(vals, warp_exchg_region_offset, twiddles_proxy); warp_exchg_region_offset >>= 1;
    reg_exchg_fwd<16, 32, 1>(vals, warp_exchg_region_offset, twiddles_proxy);
  
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
  
    reg_exchg_fwd<2, 4, 8>(vals, block_exchg_region_offset, twiddles_proxy); block_exchg_region_offset >>= 1;
    reg_exchg_fwd<4, 8, 4>(vals, block_exchg_region_offset, twiddles_proxy); block_exchg_region_offset >>= 1;
    reg_exchg_fwd<8, 16, 2>(vals, block_exchg_region_offset, twiddles_proxy); block_exchg_region_offset >>= 1;
    reg_exchg_fwd<16, 32, 1>(vals, block_exchg_region_offset, twiddles_proxy);
  
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
