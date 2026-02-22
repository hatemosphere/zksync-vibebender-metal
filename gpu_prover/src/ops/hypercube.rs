use era_cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use era_cudart::result::CudaResult;
use era_cudart::slice::DeviceSlice;
use era_cudart::stream::CudaStream;
use era_cudart::{cuda_kernel_declaration, cuda_kernel_signature_arguments_and_function};

use crate::field::BF;

const BLOCK_THREADS: u32 = 256;
const SUPPORTED_LOG_ROWS: u32 = 24;
const NONINITIAL_STAGE2_START: u32 = 12;
const NONINITIAL_STAGE3_START: u32 = 18;

cuda_kernel_signature_arguments_and_function!(
    HypercubeBitrevInitial,
    src: *const BF,
    dst: *mut BF,
);

cuda_kernel_signature_arguments_and_function!(
    HypercubeBitrevNonInitial,
    src: *const BF,
    dst: *mut BF,
    start_stage: u32,
);

macro_rules! declare_h2m_initial_kernel {
    ($name:ident) => {
        cuda_kernel_declaration!(
            $name(
                src: *const BF,
                dst: *mut BF,
            )
        );
    };
}

macro_rules! declare_h2m_noninitial_kernel {
    ($name:ident) => {
        cuda_kernel_declaration!(
            $name(
                src: *const BF,
                dst: *mut BF,
                start_stage: u32,
            )
        );
    };
}

declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial12_out_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial12_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_out_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_out_kernel);

fn validate_len(rows: usize) {
    assert!(rows.is_power_of_two());
    assert_eq!(
        rows.trailing_zeros(),
        SUPPORTED_LOG_ROWS,
        "only log24 (2^24 rows) is supported",
    );
}

fn launch_chain(
    launch0_kernel: HypercubeBitrevInitialSignature,
    launch1_kernel: HypercubeBitrevNonInitialSignature,
    launch2_kernel: HypercubeBitrevNonInitialSignature,
    launch0_src: *const BF,
    launch_dst: *mut BF,
    rows: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    // Locked log24 policy:
    // out-of-place:  #1 ld.cs/st.wt, #2 ld.cs/st.wt, #3 ld.cs/st.cs
    // in-place:      #1 ld.cg/st.wt, #2 ld.ca/st.wt, #3 ld.ca/st.cs
    let grid_initial_12 = (rows >> 12) as u32;
    let grid_noninitial_6 = (rows >> 11) as u32;

    let config0 = CudaLaunchConfig::basic(
        Dim3 {
            x: grid_initial_12,
            y: 1,
            z: 1,
        },
        BLOCK_THREADS,
        stream,
    );
    let args0 = HypercubeBitrevInitialArguments::new(launch0_src, launch_dst);
    HypercubeBitrevInitialFunction(launch0_kernel).launch(&config0, &args0)?;

    let launch1_src = launch_dst as *const BF;

    let config1 = CudaLaunchConfig::basic(
        Dim3 {
            x: grid_noninitial_6,
            y: 1,
            z: 1,
        },
        BLOCK_THREADS,
        stream,
    );
    let args1 =
        HypercubeBitrevNonInitialArguments::new(launch1_src, launch_dst, NONINITIAL_STAGE2_START);
    HypercubeBitrevNonInitialFunction(launch1_kernel).launch(&config1, &args1)?;

    let args2 =
        HypercubeBitrevNonInitialArguments::new(launch1_src, launch_dst, NONINITIAL_STAGE3_START);
    HypercubeBitrevNonInitialFunction(launch2_kernel).launch(&config1, &args2)?;

    Ok(())
}

pub fn hypercube_evals_into_coeffs_bitrev_bf(
    src: &DeviceSlice<BF>,
    dst: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    let rows = src.len();
    assert_eq!(dst.len(), rows);
    validate_len(rows);

    launch_chain(
        ab_h2m_bitrev_bf_initial12_out_kernel,
        ab_h2m_bitrev_bf_noninitial6_stage2_out_kernel,
        ab_h2m_bitrev_bf_noninitial6_stage3_out_kernel,
        src.as_ptr(),
        dst.as_mut_ptr(),
        rows,
        stream,
    )
}

pub fn hypercube_evals_into_coeffs_bitrev_bf_in_place(
    values: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    validate_len(values.len());
    let dst = values.as_mut_ptr();

    launch_chain(
        ab_h2m_bitrev_bf_initial12_in_kernel,
        ab_h2m_bitrev_bf_noninitial6_stage2_in_kernel,
        ab_h2m_bitrev_bf_noninitial6_stage3_in_kernel,
        dst as *const BF,
        dst,
        values.len(),
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use era_cudart::memory::{memory_copy_async, DeviceAllocation};
    use field::Field;
    use prover::gkr::whir::hypercube_to_monomial::multivariate_hypercube_evals_into_coeffs;
    use rand::{rng, Rng};

    fn bitreverse_permute(values: &[BF]) -> Vec<BF> {
        assert!(values.len().is_power_of_two());
        let log_rows = values.len().trailing_zeros();
        let mut out = vec![BF::ZERO; values.len()];
        for (i, value) in values.iter().copied().enumerate() {
            let j = (i as u32).reverse_bits() >> (u32::BITS - log_rows);
            out[j as usize] = value;
        }
        out
    }

    fn reference_h2m_bitrev(input_bitrev: &[BF]) -> Vec<BF> {
        let mut canonical = bitreverse_permute(input_bitrev);
        let size_log2 = canonical.len().trailing_zeros();
        multivariate_hypercube_evals_into_coeffs(&mut canonical, size_log2);
        bitreverse_permute(&canonical)
    }

    fn run_out_of_place_case(rows: usize) {
        let mut rng = rng();
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();
        let h_expected = reference_h2m_bitrev(&h_input);

        let stream = CudaStream::default();
        let mut d_src = DeviceAllocation::alloc(rows).unwrap();
        let mut d_dst = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_src, &h_input, &stream).unwrap();

        hypercube_evals_into_coeffs_bitrev_bf(&d_src, &mut d_dst, &stream).unwrap();
        let mut h_actual = vec![BF::ZERO; rows];
        memory_copy_async(&mut h_actual, &d_dst, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(h_actual, h_expected);
    }

    fn run_in_place_case(rows: usize) {
        let mut rng = rng();
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();
        let h_expected = reference_h2m_bitrev(&h_input);

        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_values, &h_input, &stream).unwrap();

        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
        let mut h_actual = vec![BF::ZERO; rows];
        memory_copy_async(&mut h_actual, &d_values, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(h_actual, h_expected);
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_out_of_place_log24() {
        run_out_of_place_case(1usize << SUPPORTED_LOG_ROWS);
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_in_place_log24() {
        run_in_place_case(1usize << SUPPORTED_LOG_ROWS);
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_single_invocation_log24_col1() {
        // Profiling-only entrypoint: launches one kernel chain without correctness checks.
        let rows = 1usize << SUPPORTED_LOG_ROWS;
        let mut rng = rng();
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        let mut d_src = DeviceAllocation::alloc(rows).unwrap();
        let mut d_dst = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_src, &h_input, &stream).unwrap();

        hypercube_evals_into_coeffs_bitrev_bf(&d_src, &mut d_dst, &stream).unwrap();
        stream.synchronize().unwrap();
    }

    #[test]
    #[should_panic]
    fn unsupported_log_rows_panics() {
        let len = 1usize << 23;
        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(len).unwrap();
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
    }

    #[test]
    #[should_panic]
    fn non_power_of_two_panics() {
        let len = (1usize << SUPPORTED_LOG_ROWS) - 1;
        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(len).unwrap();
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
    }
}
