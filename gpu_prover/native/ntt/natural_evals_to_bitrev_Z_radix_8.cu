#include "ntt.cuh"

namespace airbender::ntt {

DEVICE_FORCEINLINE void size_8_fwd_dit(e2f *x) {
  constexpr e2f W_1_8 = e2f{bf{32768}, bf{2147450879}};
  // constexpr e2f W_1_4 = e2f{bf{0}, bf{2147483646}};
  constexpr e2f W_3_8 = e2f{bf{2147450879}, bf{2147450879}};

  // first stage
#pragma unroll
  for (unsigned i{0}; i < 4; i++) {
      const e2f tmp = x[i];
      x[i] = e2f::add(tmp, x[i + 4]);
      x[i + 4] = e2f::sub(tmp, x[i + 4]);
  }

  // second stage
#pragma unroll
  for (unsigned i{0}; i < 2; i++) {
      const e2f tmp = x[i];
      x[i] = e2f::add(tmp, x[i + 2]);
      x[i + 2] = e2f::sub(tmp, x[i + 2]);
  }
  // x[4] = x[4] + W_1_4 * (x[6].real + i * x[6].imag)
  //      = x[4] + (-i) * (x[6].real + i * x[6].imag)
  //      = x[4] + (x[6].imag - i * x[6].real)
  // x[6] = x[4] - W_1_4 * (x[6].real + i * x[6].imag)
  //      = x[4] - (-i) * (x[6].real + i * x[6].imag)
  //      = x[4] + (-x[6].imag + i * x[6].real)
#pragma unroll
  for (unsigned i{4}; i < 6; i++) {
      const e2f tmp0 = x[i];
      x[i][0] = bf::add(x[i][0], x[i + 2][1]);
      x[i][1] = bf::sub(x[i][1], x[i + 2][0]);
      const bf tmp1 = x[i + 2][0];
      x[i + 2][0] = bf::sub(tmp0[0], x[i + 2][1]);
      x[i + 2][1] = bf::add(tmp0[1], tmp1);
  }

  // third stage
  {
    // x[3] = W_1_4 * x[3]
    //      = -i * (x[3].real + i * x[3].imag)
    //      = x[3].imag - i * x[3].real) 
    const bf tmp = x[3][0];
    x[3][0] = x[3][1];
    x[3][1] = bf::neg(tmp);
  }
  x[5] = e2f::mul(W_1_8, x[5]); // don't bother optimizing, marginal gains
  x[7] = e2f::mul(W_3_8, x[7]); // don't bother optimizing, marginal gains
#pragma unroll
  for (unsigned i{0}; i < 8; i += 2) {
      const e2f tmp = x[i];
      x[i] = e2f::add(tmp, x[i + 1]);
      x[i + 1] = e2f::sub(tmp, x[i + 1]);
  }

  // undo bitrev
  const e2f tmp0 = x[1];
  x[1] = x[4];
  x[4] = tmp0;
  const e2f tmp1 = x[3];
  x[3] = x[6];
  x[6] = tmp1;
}

DEVICE_FORCEINLINE void size_8_inv_dit(e2f *x) {
  constexpr e2f W_1_8_INV = e2f{bf{32768}, bf{32768}};
  // constexpr e2f W_1_4_INV = e2f{bf{0}, bf{1}};
  constexpr e2f W_3_8_INV = e2f{bf{2147450879}, bf{32768}};

  // first stage
#pragma unroll
  for (unsigned i{0}; i < 4; i++) {
      const e2f tmp = x[i];
      x[i] = e2f::add(tmp, x[i + 4]);
      x[i + 4] = e2f::sub(tmp, x[i + 4]);
  }

  // second stage
#pragma unroll
  for (unsigned i{0}; i < 2; i++) {
      const e2f tmp = x[i];
      x[i] = e2f::add(tmp, x[i + 2]);
      x[i + 2] = e2f::sub(tmp, x[i + 2]);
  }
  // x[4] = x[4] + W_1_4_INV * (x[6].real + i * x[6].imag)
  //      = x[4] + i * (x[6].real + i * x[6].imag)
  //      = x[4] + (-x[6].imag + i * x[6].real)
  // x[6] = x[4] - W_1_4_INV * (x[6].real + i * x[6].imag)
  //      = x[4] - i * (x[6].real + i * x[6].imag)
  //      = x[4] + (x[6].imag - i * x[6].real)
#pragma unroll
  for (unsigned i{4}; i < 6; i++) {
      const e2f tmp0 = x[i];
      x[i][0] = bf::sub(x[i][0], x[i + 2][1]);
      x[i][1] = bf::add(x[i][1], x[i + 2][0]);
      const bf tmp1 = x[i + 2][0];
      x[i + 2][0] = bf::add(tmp0[0], x[i + 2][1]);
      x[i + 2][1] = bf::sub(tmp0[1], tmp1);
  }

  // third stage
  {
    // x[3] = W_1_4_INV * x[3]
    //      = i * (x[3].real + i * x[3].imag)
    //      = -x[3].imag + i * x[3].real) 
    const bf tmp = x[3][0];
    x[3][0] = bf::neg(x[3][1]);
    x[3][1] = tmp;
  }
  x[5] = e2f::mul(W_1_8_INV, x[5]); // don't bother optimizing, marginal gains
  x[7] = e2f::mul(W_3_8_INV, x[7]); // don't bother optimizing, marginal gains
#pragma unroll
  for (unsigned i{0}; i < 8; i += 2) {
      const e2f tmp = x[i];
      x[i] = e2f::add(tmp, x[i + 1]);
      x[i + 1] = e2f::sub(tmp, x[i + 1]);
  }

  // undo bitrev
  const e2f tmp0 = x[1];
  x[1] = x[4];
  x[4] = tmp0;
  const e2f tmp1 = x[3];
  x[3] = x[6];
  x[6] = tmp1;
}

template <unsigned LOG_RADIX>
DEVICE_FORCEINLINE unsigned bitrev_by_radix(const unsigned idx, const unsigned bit_chunks) {
  constexpr unsigned RADIX_MASK = (1 << LOG_RADIX) - 1;
  unsigned out{0}, tmp_idx{idx};
  for (unsigned i{0}; i < bit_chunks; i++) {
    out <<= LOG_RADIX;
    out |= tmp_idx & RADIX_MASK;
    tmp_idx >>= LOG_RADIX;
  }
  return out;
}

EXTERN __launch_bounds__(128, 8) __global__
    void ab_bit_reverse_by_radix_8(vectorized_e2_matrix_getter<ld_modifier::cg> src, vectorized_e2_matrix_setter<st_modifier::cg> dst,
                                 const unsigned bit_chunks, const unsigned log_n) {
  const unsigned n = 1 << log_n;
  const unsigned l_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (l_index >= n)
    return;
  const unsigned r_index = bitrev_by_radix<3>(l_index, bit_chunks);
  if (l_index > r_index)
    return;
  const e2f l_value = src.get_at_row(l_index);
  const e2f r_value = src.get_at_row(r_index);
  dst.set_at_row(l_index, r_value);
  dst.set_at_row(r_index, l_value);
}
                                                     

template <unsigned LOG_RADIX>
DEVICE_FORCEINLINE void apply_twiddles_same_region(e2f *vals0, e2f *vals1, const unsigned exchg_region, const unsigned twiddle_stride,
                                                   const unsigned idx_bit_chunks) {
  constexpr unsigned RADIX = 1 << LOG_RADIX;
  if (exchg_region > 0) {
    const unsigned v = bitrev_by_radix<LOG_RADIX>(exchg_region, idx_bit_chunks);
#pragma unroll
    for (unsigned i{1}; i < RADIX; i++) {
      const auto twiddle = get_twiddle_with_direct_index<true>(v * i * twiddle_stride);
      vals0[i] = e2f::mul(vals0[i], twiddle);
      vals1[i] = e2f::mul(vals1[i], twiddle);
    }  
  }
}

template <unsigned LOG_RADIX>
DEVICE_FORCEINLINE void apply_twiddles_distinct_regions(e2f *vals0, e2f *vals1, const unsigned exchg_region_0, const unsigned exchg_region_1,
                                                        const unsigned twiddle_stride, const unsigned idx_bit_chunks) {
  constexpr unsigned RADIX = 1 << LOG_RADIX;
  if (exchg_region_0 > 0) {
    const unsigned v = bitrev_by_radix<LOG_RADIX>(exchg_region_0, idx_bit_chunks);
#pragma unroll
    for (unsigned i{1}; i < RADIX; i++) {
      const auto twiddle = get_twiddle_with_direct_index<true>(v * i * twiddle_stride);
      vals0[i] = e2f::mul(vals0[i], twiddle);
    }  
  }
  // exchg_region_1 should never be 0
  const unsigned v = bitrev_by_radix<LOG_RADIX>(exchg_region_1, idx_bit_chunks);
#pragma unroll
  for (unsigned i{1}; i < RADIX; i++) {
    const auto twiddle = get_twiddle_with_direct_index<true>(v * i * twiddle_stride);
    vals1[i] = e2f::mul(vals1[i], twiddle);
  }  
}

EXTERN __launch_bounds__(256, 3) __global__
    void ab_radix_8_main_domain_evals_to_Z_final_12_stages_block(vectorized_e2_matrix_getter<ld_modifier::cg> gmem_in,
                                                                 vectorized_e2_matrix_setter<st_modifier::cg> gmem_out, const unsigned start_stage,
                                                                 unsigned exchg_region_bit_chunks, const unsigned log_n, const unsigned grid_offset) {
  constexpr unsigned WARP_SIZE = 32u;
  constexpr unsigned LOG_RADIX = 3u;
  constexpr unsigned RADIX = 1 << LOG_RADIX;
  constexpr unsigned VALS_PER_BLOCK = 4096;

  __shared__ e2f static_smem[VALS_PER_BLOCK];
  e2f *smem = static_smem;

  const unsigned effective_block_idx_x = blockIdx.x + grid_offset;
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned lane_id{threadIdx.x & 31};
  const unsigned gmem_block_offset = VALS_PER_BLOCK * effective_block_idx_x;
  gmem_in.add_row(gmem_block_offset);
  gmem_out.add_row(gmem_block_offset);

  e2f vals0[RADIX];
  e2f vals1[RADIX];

  unsigned twiddle_stride = 1 << (OMEGA_LOG_ORDER - LOG_RADIX * (start_stage + 1));

  // First three stages
  {
#pragma unroll
    for (unsigned i{0}, addr{threadIdx.x}; i < RADIX; i++, addr += 2 * blockDim.x) {
      vals0[i] = gmem_in.get_at_row(addr);
      vals1[i] = gmem_in.get_at_row(addr + blockDim.x);
    }

    if (start_stage > 0)
      apply_twiddles_same_region<LOG_RADIX>(vals0, vals1, effective_block_idx_x, twiddle_stride, exchg_region_bit_chunks);

    size_8_inv_dit(vals0);
    size_8_inv_dit(vals1);

#pragma unroll
    for (unsigned i{0}, addr{threadIdx.x}; i < RADIX; i++, addr += 2 * blockDim.x) {
      smem[addr] = vals0[i];
      smem[addr + blockDim.x] = vals1[i];
      // gmem_out.set_at_row(addr, vals0[i]);
      // gmem_out.set_at_row(addr + blockDim.x, vals1[i]);
    }

    __syncthreads();
  }

  // The remaining stages will be processed within each warp.
  const unsigned warp_offset = warp_id * 512;
  smem += warp_offset;
  gmem_out.add_row(warp_offset);
  unsigned warp_exchg_region_offset = effective_block_idx_x * RADIX + warp_id;

  // Second three stages
  {
#pragma unroll
    for (unsigned i{0}, addr{lane_id}; i < RADIX; i++, addr += 2 * WARP_SIZE) {
      vals0[i] = smem[addr];
      vals1[i] = smem[addr + WARP_SIZE];
    }

    twiddle_stride >>= LOG_RADIX;
    apply_twiddles_same_region<LOG_RADIX>(vals0, vals1, warp_exchg_region_offset, twiddle_stride, ++exchg_region_bit_chunks);

    size_8_inv_dit(vals0);
    size_8_inv_dit(vals1);

#pragma unroll
    for (unsigned i{0}, addr{lane_id}; i < RADIX; i++, addr += 2 * WARP_SIZE) {
      smem[addr] = vals0[i];
      smem[addr + WARP_SIZE] = vals1[i];
    }

    __syncwarp();
  }

  // Third three stages
  {
    const unsigned tile_id{lane_id >> 3};
    const unsigned thread_offset = (lane_id & 7) + tile_id * 128;
    // I could swizzle access order across tiles to avoid bank conflicts, but swizzling logic
    // would probably add more instructions than the bank conflict replays. Seems marginal.
#pragma unroll
    for (unsigned i{0}, addr{thread_offset}; i < RADIX; i++, addr += RADIX) {
      vals0[i] = smem[addr];
      vals1[i] = smem[addr + 64];
    }

    warp_exchg_region_offset *= RADIX;
    const unsigned exchg_region_0 = warp_exchg_region_offset + tile_id * 2; 
    const unsigned exchg_region_1 = exchg_region_0 + 1;
    twiddle_stride >>= LOG_RADIX;
    apply_twiddles_distinct_regions<LOG_RADIX>(vals0, vals1, exchg_region_0, exchg_region_1, twiddle_stride, ++exchg_region_bit_chunks);

    size_8_inv_dit(vals0);
    size_8_inv_dit(vals1);

#pragma unroll
    for (unsigned i{0}, addr{thread_offset}; i < RADIX; i++, addr += RADIX) {
      smem[addr] = vals0[i];
      smem[addr + 64] = vals1[i];
      // gmem_out.set_at_row(addr, vals0[i]);
      // gmem_out.set_at_row(addr + 64, vals1[i]);
    }

    __syncwarp();
  }

  // Fourth three stages
  {
    const unsigned thread_offset = lane_id * 8;
#pragma unroll
    for (unsigned i{0}; i < RADIX; i++) {
      vals0[i] = smem[i + thread_offset];
      vals1[i] = smem[i + thread_offset + 256];
    }

    warp_exchg_region_offset *= RADIX;
    const unsigned exchg_region_0 = warp_exchg_region_offset + lane_id; 
    const unsigned exchg_region_1 = exchg_region_0 + 32;
    twiddle_stride >>= LOG_RADIX;
    apply_twiddles_distinct_regions<LOG_RADIX>(vals0, vals1, exchg_region_0, exchg_region_1, twiddle_stride, ++exchg_region_bit_chunks);

    size_8_inv_dit(vals0);
    size_8_inv_dit(vals1);

#pragma unroll
    for (unsigned i{0}; i < RADIX; i++) {
      vals0[i] = e2f::mul(vals0[i], ab_inv_sizes[log_n]);
      vals1[i] = e2f::mul(vals1[i], ab_inv_sizes[log_n]);
    }

    gmem_out.set_four_adjacent(thread_offset, vals0[0], vals0[1], vals0[2], vals0[3]);
    gmem_out.set_four_adjacent(thread_offset + 4, vals0[4], vals0[5], vals0[6], vals0[7]);
    gmem_out.set_four_adjacent(thread_offset + 256, vals1[0], vals1[1], vals1[2], vals1[3]);
    gmem_out.set_four_adjacent(thread_offset + 260, vals1[4], vals1[5], vals1[6], vals1[7]);
  }
}

} // namespace airbender::ntt
