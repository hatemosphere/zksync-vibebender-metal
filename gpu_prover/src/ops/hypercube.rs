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
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_kernel);

fn validate_len(rows: usize) {
    // Current public path is intentionally locked to the tuned log24 schedule.
    assert!(rows.is_power_of_two());
    assert_eq!(
        rows.trailing_zeros(),
        SUPPORTED_LOG_ROWS,
        "only log24 (2^24 rows) is supported",
    );
}

fn launch_chain(
    launch0_kernel: HypercubeBitrevInitialSignature,
    launch0_src: *const BF,
    launch_dst: *mut BF,
    rows: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    // Launch geometry depends only on rows and per-kernel rounds:
    // initial12 owns 2^12 points per block; noninitial6 owns 2^11 points per block.
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
    let args1 = HypercubeBitrevNonInitialArguments::new(
        launch1_src,
        launch_dst,
        NONINITIAL_STAGE2_START,
    );
    HypercubeBitrevNonInitialFunction(ab_h2m_bitrev_bf_noninitial6_stage2_kernel)
        .launch(&config1, &args1)?;

    let args2 = HypercubeBitrevNonInitialArguments::new(
        launch1_src,
        launch_dst,
        NONINITIAL_STAGE3_START,
    );
    HypercubeBitrevNonInitialFunction(ab_h2m_bitrev_bf_noninitial6_stage3_kernel)
        .launch(&config1, &args2)?;

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
    use field::{Field, Rand};
    use prover::gkr::whir::hypercube_to_monomial::multivariate_hypercube_evals_into_coeffs;
    use rand::rng;

    fn bitreverse_permute(values: &[BF]) -> Vec<BF> {
        assert!(values.len().is_power_of_two());
        let log_rows = values.len().trailing_zeros();
        let mut out = vec![BF::ZERO; values.len()];
        for (i, value) in values.iter().copied().enumerate() {
            let j = i.reverse_bits() >> (usize::BITS - log_rows);
            out[j] = value;
        }
        out
    }

    fn cpu_reference_bitrev(src: &[BF], rows: usize) -> Vec<BF> {
        let mut natural = bitreverse_permute(src);
        multivariate_hypercube_evals_into_coeffs(&mut natural, rows.trailing_zeros());
        bitreverse_permute(&natural)
    }

    fn run_case(in_place: bool) {
        let len = 1usize << SUPPORTED_LOG_ROWS;
        let mut rng = rng();
        let h_src: Vec<BF> = (0..len).map(|_| BF::random_element(&mut rng)).collect();
        let expected = cpu_reference_bitrev(&h_src, len);

        let stream = CudaStream::default();
        let mut h_dst = vec![BF::ZERO; len];

        if in_place {
            let mut d_values = DeviceAllocation::alloc(len).unwrap();
            memory_copy_async(&mut d_values, &h_src, &stream).unwrap();
            hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
            memory_copy_async(&mut h_dst, &d_values, &stream).unwrap();
        } else {
            let mut d_src = DeviceAllocation::alloc(len).unwrap();
            let mut d_dst = DeviceAllocation::alloc(len).unwrap();
            memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
            hypercube_evals_into_coeffs_bitrev_bf(&d_src, &mut d_dst, &stream).unwrap();
            memory_copy_async(&mut h_dst, &d_dst, &stream).unwrap();
        }

        stream.synchronize().unwrap();
        assert_eq!(h_dst, expected);
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_out_of_place_log24() {
        run_case(false);
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_in_place_log24() {
        run_case(true);
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_single_invocation_log24_col1() {
        let len = 1usize << SUPPORTED_LOG_ROWS;
        let stream = CudaStream::default();
        let h_src = vec![BF::ZERO; len];
        let mut d_src = DeviceAllocation::alloc(len).unwrap();
        let mut d_dst = DeviceAllocation::alloc(len).unwrap();

        memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
        stream.synchronize().unwrap();

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
