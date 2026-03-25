use crate::metal_runtime::{MetalBuffer, MetalResult};
use field::{Field, Mersenne31Complex, Mersenne31Field};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

pub const OMEGA_LOG_ORDER: u32 = 25;
pub const CIRCLE_GROUP_LOG_ORDER: u32 = 31;
pub const FINEST_LOG_COUNT: u32 = CIRCLE_GROUP_LOG_ORDER - OMEGA_LOG_ORDER;

type BF = Mersenne31Field;
type E2 = Mersenne31Complex;

/// Device context for twiddle factor tables.
/// In CUDA these live in __constant__ memory; in Metal they are passed as buffer arguments.
pub struct DeviceContext {
    // 3-layer powers of w (for coset evaluation)
    pub powers_of_w_fine: MetalBuffer<E2>,
    pub powers_of_w_coarser: MetalBuffer<E2>,
    pub powers_of_w_coarsest: MetalBuffer<E2>,
    pub fine_log_count: u32,
    pub coarser_log_count: u32,
    pub coarsest_log_count: u32,
    // 2-layer NTT twiddles (forward)
    pub powers_of_w_fine_bitrev_for_ntt: MetalBuffer<E2>,
    pub powers_of_w_coarse_bitrev_for_ntt: MetalBuffer<E2>,
    pub ntt_fine_log_count: u32,
    pub ntt_coarse_log_count: u32,
    // 2-layer NTT twiddles (inverse)
    pub powers_of_w_inv_fine_bitrev_for_ntt: MetalBuffer<E2>,
    pub powers_of_w_inv_coarse_bitrev_for_ntt: MetalBuffer<E2>,
    // Inverse domain sizes: inv_sizes[i] = (1/2)^i
    pub inv_sizes: MetalBuffer<BF>,
}

/// Compute powers [1, g, g^2, ..., g^(n-1)] of a generator.
fn compute_powers(gen: E2, count: usize) -> Vec<E2> {
    let mut result = vec![E2::ONE; count];
    for i in 1..count {
        let mut prev = result[i - 1];
        Field::mul_assign(&mut prev, &gen);
        result[i] = prev;
    }
    result
}

impl DeviceContext {
    /// Create device context. Parameter `powers_of_w_coarsest_log_count` matches
    /// the CUDA DeviceContext::create parameter of the same name.
    pub fn create(
        device: &ProtocolObject<dyn MTLDevice>,
        powers_of_w_coarsest_log_count: u32,
    ) -> MetalResult<Self> {
        assert!(powers_of_w_coarsest_log_count <= OMEGA_LOG_ORDER);

        let fine_log_count = FINEST_LOG_COUNT;
        let coarser_log_count = OMEGA_LOG_ORDER - powers_of_w_coarsest_log_count;
        let coarsest_log_count = powers_of_w_coarsest_log_count;

        // === 3-layer powers of w (NOT bitreversed, matching CUDA) ===
        let fine_count = 1usize << fine_log_count;
        let fine_gen = fft::field_utils::domain_generator_for_size::<E2>(
            1u64 << CIRCLE_GROUP_LOG_ORDER,
        );
        let fine_values = compute_powers(fine_gen, fine_count);
        let powers_of_w_fine = MetalBuffer::from_slice(device, &fine_values)?;

        let coarser_count = 1usize << coarser_log_count;
        let coarser_gen =
            fft::field_utils::domain_generator_for_size::<E2>(1u64 << OMEGA_LOG_ORDER);
        let coarser_values = compute_powers(coarser_gen, coarser_count);
        let powers_of_w_coarser = MetalBuffer::from_slice(device, &coarser_values)?;

        let coarsest_count = 1usize << coarsest_log_count;
        let coarsest_gen =
            fft::field_utils::domain_generator_for_size::<E2>(coarsest_count as u64);
        let coarsest_values = compute_powers(coarsest_gen, coarsest_count);
        let powers_of_w_coarsest = MetalBuffer::from_slice(device, &coarsest_values)?;

        // === 2-layer NTT twiddles (forward, bitreversed) ===
        // Accounts for twiddle arrays only covering half the range
        let ntt_fine_log_count = coarser_log_count - 1;
        let ntt_coarse_log_count = coarsest_log_count;
        let ntt_fine_count = 1usize << ntt_fine_log_count;
        let ntt_coarse_count = 1usize << ntt_coarse_log_count;

        let ntt_fine_gen =
            fft::field_utils::domain_generator_for_size::<E2>(1u64 << OMEGA_LOG_ORDER);
        let mut ntt_fine_values = compute_powers(ntt_fine_gen, ntt_fine_count);
        fft::bitreverse_enumeration_inplace(&mut ntt_fine_values);
        let powers_of_w_fine_bitrev_for_ntt =
            MetalBuffer::from_slice(device, &ntt_fine_values)?;

        let ntt_coarse_gen = fft::field_utils::domain_generator_for_size::<E2>(
            (ntt_coarse_count * 2) as u64,
        );
        let mut ntt_coarse_values = compute_powers(ntt_coarse_gen, ntt_coarse_count);
        fft::bitreverse_enumeration_inplace(&mut ntt_coarse_values);
        let powers_of_w_coarse_bitrev_for_ntt =
            MetalBuffer::from_slice(device, &ntt_coarse_values)?;

        // === 2-layer NTT twiddles (inverse, bitreversed) ===
        let ntt_inv_fine_gen = ntt_fine_gen.inverse().expect("must exist");
        let mut ntt_inv_fine_values = compute_powers(ntt_inv_fine_gen, ntt_fine_count);
        fft::bitreverse_enumeration_inplace(&mut ntt_inv_fine_values);
        let powers_of_w_inv_fine_bitrev_for_ntt =
            MetalBuffer::from_slice(device, &ntt_inv_fine_values)?;

        let ntt_inv_coarse_gen = ntt_coarse_gen.inverse().expect("must exist");
        let mut ntt_inv_coarse_values = compute_powers(ntt_inv_coarse_gen, ntt_coarse_count);
        fft::bitreverse_enumeration_inplace(&mut ntt_inv_coarse_values);
        let powers_of_w_inv_coarse_bitrev_for_ntt =
            MetalBuffer::from_slice(device, &ntt_inv_coarse_values)?;

        // === Inverse sizes: inv_sizes[i] = (1/2)^i ===
        let two_inv = BF::new(2).inverse().expect("must exist");
        let mut inv_sizes_host = vec![BF::ONE; (OMEGA_LOG_ORDER + 1) as usize];
        fft::field_utils::distribute_powers_serial(&mut inv_sizes_host, BF::ONE, two_inv);
        let inv_sizes = MetalBuffer::from_slice(device, &inv_sizes_host)?;

        Ok(Self {
            powers_of_w_fine,
            powers_of_w_coarser,
            powers_of_w_coarsest,
            fine_log_count,
            coarser_log_count,
            coarsest_log_count,
            powers_of_w_fine_bitrev_for_ntt,
            powers_of_w_coarse_bitrev_for_ntt,
            ntt_fine_log_count,
            ntt_coarse_log_count,
            powers_of_w_inv_fine_bitrev_for_ntt,
            powers_of_w_inv_coarse_bitrev_for_ntt,
            inv_sizes,
        })
    }
}
