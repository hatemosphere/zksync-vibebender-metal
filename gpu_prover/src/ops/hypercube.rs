use era_cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use era_cudart::result::CudaResult;
use era_cudart::slice::DeviceSlice;
use era_cudart::stream::CudaStream;
use era_cudart::{cuda_kernel_declaration, cuda_kernel_signature_arguments_and_function};

use crate::field::BF;

const BLOCK_THREADS: u32 = 256;
const MIN_SUPPORTED_LOG_ROWS: u32 = 23;
const MAX_SUPPORTED_LOG_ROWS: u32 = 24;
const NONINITIAL_GRID_LOG_ROWS: u32 = 11;

const LOG23_INITIAL_ROUNDS: u32 = 11;
const LOG23_NONINITIAL_STAGE2_START: u32 = LOG23_INITIAL_ROUNDS;
const LOG23_NONINITIAL_STAGE3_START: u32 = LOG23_NONINITIAL_STAGE2_START + 6;

const LOG24_INITIAL_ROUNDS: u32 = 12;
const LOG24_NONINITIAL_STAGE2_START: u32 = LOG24_INITIAL_ROUNDS;
const LOG24_NONINITIAL_STAGE3_START: u32 = LOG24_NONINITIAL_STAGE2_START + 6;

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
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial11_out_kernel);
declare_h2m_initial_kernel!(ab_h2m_bitrev_bf_initial11_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_out_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage2_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_in_kernel);
declare_h2m_noninitial_kernel!(ab_h2m_bitrev_bf_noninitial6_stage3_out_kernel);

fn validate_len(rows: usize) -> u32 {
    assert!(rows.is_power_of_two());
    let log_rows = rows.trailing_zeros();
    assert!(
        (MIN_SUPPORTED_LOG_ROWS..=MAX_SUPPORTED_LOG_ROWS).contains(&log_rows),
        "only log23/log24 (2^23/2^24 rows) are supported",
    );
    log_rows
}

fn launch_chain(
    launch0_kernel: HypercubeBitrevInitialSignature,
    launch1_kernel: HypercubeBitrevNonInitialSignature,
    launch2_kernel: HypercubeBitrevNonInitialSignature,
    initial_rounds: u32,
    noninitial_stage2_start: u32,
    noninitial_stage3_start: u32,
    launch0_src: *const BF,
    launch_dst: *mut BF,
    rows: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    // Locked cache policy by stage role:
    // - log24 schedule: [12, 6, 6]
    // - log23 schedule: [11, 6, 6]
    // out-of-place:  #1 ld.cs/st.wt, #2 ld.cs/st.wt, #3 ld.cs/st.cs
    // in-place:      #1 ld.cg/st.wt, #2 ld.ca/st.wt, #3 ld.ca/st.cs
    let grid_initial = (rows >> initial_rounds) as u32;
    let grid_noninitial_6 = (rows >> NONINITIAL_GRID_LOG_ROWS) as u32;

    let config0 = CudaLaunchConfig::basic(
        Dim3 {
            x: grid_initial,
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
        HypercubeBitrevNonInitialArguments::new(launch1_src, launch_dst, noninitial_stage2_start);
    HypercubeBitrevNonInitialFunction(launch1_kernel).launch(&config1, &args1)?;

    let args2 =
        HypercubeBitrevNonInitialArguments::new(launch1_src, launch_dst, noninitial_stage3_start);
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
    let log_rows = validate_len(rows);
    let (launch0_kernel, initial_rounds, stage2_start, stage3_start): (
        HypercubeBitrevInitialSignature,
        u32,
        u32,
        u32,
    ) = match log_rows {
        24 => (
            ab_h2m_bitrev_bf_initial12_out_kernel,
            LOG24_INITIAL_ROUNDS,
            LOG24_NONINITIAL_STAGE2_START,
            LOG24_NONINITIAL_STAGE3_START,
        ),
        23 => (
            ab_h2m_bitrev_bf_initial11_out_kernel,
            LOG23_INITIAL_ROUNDS,
            LOG23_NONINITIAL_STAGE2_START,
            LOG23_NONINITIAL_STAGE3_START,
        ),
        _ => unreachable!("validate_len enforces supported log rows"),
    };

    launch_chain(
        launch0_kernel,
        ab_h2m_bitrev_bf_noninitial6_stage2_out_kernel,
        ab_h2m_bitrev_bf_noninitial6_stage3_out_kernel,
        initial_rounds,
        stage2_start,
        stage3_start,
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
    let log_rows = validate_len(values.len());
    let (launch0_kernel, initial_rounds, stage2_start, stage3_start): (
        HypercubeBitrevInitialSignature,
        u32,
        u32,
        u32,
    ) = match log_rows {
        24 => (
            ab_h2m_bitrev_bf_initial12_in_kernel,
            LOG24_INITIAL_ROUNDS,
            LOG24_NONINITIAL_STAGE2_START,
            LOG24_NONINITIAL_STAGE3_START,
        ),
        23 => (
            ab_h2m_bitrev_bf_initial11_in_kernel,
            LOG23_INITIAL_ROUNDS,
            LOG23_NONINITIAL_STAGE2_START,
            LOG23_NONINITIAL_STAGE3_START,
        ),
        _ => unreachable!("validate_len enforces supported log rows"),
    };
    let dst = values.as_mut_ptr();

    launch_chain(
        launch0_kernel,
        ab_h2m_bitrev_bf_noninitial6_stage2_in_kernel,
        ab_h2m_bitrev_bf_noninitial6_stage3_in_kernel,
        initial_rounds,
        stage2_start,
        stage3_start,
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
    use rand::{rng, rngs::StdRng, Rng, SeedableRng};
    use std::time::Instant;

    const PROFILE_MULTI_WARMUP_ITERS: usize = 20;
    const PROFILE_MULTI_MEASURE_ITERS: usize = 100;

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

    fn percentile_from_sorted(sorted_samples_us: &[f64], percentile: f64) -> f64 {
        assert!(!sorted_samples_us.is_empty());
        let clamped = percentile.clamp(0.0, 1.0);
        let idx = ((sorted_samples_us.len() - 1) as f64 * clamped).round() as usize;
        sorted_samples_us[idx]
    }

    fn run_profile_out_of_place_multi_invocation(
        rows: usize,
        warmup_iters: usize,
        measure_iters: usize,
    ) {
        assert!(measure_iters > 0);
        let mut rng = StdRng::seed_from_u64(0xA1B2_C3D4_55AA_77EEu64 ^ rows as u64);
        let h_input = (0..rows)
            .map(|_| BF::from_nonreduced_u32(rng.random()))
            .collect::<Vec<_>>();

        let stream = CudaStream::default();
        let mut d_src = DeviceAllocation::alloc(rows).unwrap();
        let mut d_dst = DeviceAllocation::alloc(rows).unwrap();
        memory_copy_async(&mut d_src, &h_input, &stream).unwrap();
        stream.synchronize().unwrap();

        for _ in 0..warmup_iters {
            hypercube_evals_into_coeffs_bitrev_bf(&d_src, &mut d_dst, &stream).unwrap();
        }
        stream.synchronize().unwrap();

        let mut samples_us = Vec::with_capacity(measure_iters);
        for _ in 0..measure_iters {
            let start = Instant::now();
            hypercube_evals_into_coeffs_bitrev_bf(&d_src, &mut d_dst, &stream).unwrap();
            stream.synchronize().unwrap();
            samples_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }

        let mut sorted = samples_us.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = samples_us.iter().sum::<f64>() / samples_us.len() as f64;
        let variance = samples_us
            .iter()
            .map(|sample| {
                let delta = sample - mean;
                delta * delta
            })
            .sum::<f64>()
            / samples_us.len() as f64;
        let stddev = variance.sqrt();
        let cv_pct = if mean > 0.0 { (stddev / mean) * 100.0 } else { 0.0 };

        println!(
            "profile_h2m_chain rows={} log_rows={} warmup={} iters={} mean_us={:.3} median_us={:.3} p90_us={:.3} p95_us={:.3} min_us={:.3} max_us={:.3} stddev_us={:.3} cv_pct={:.2}",
            rows,
            rows.trailing_zeros(),
            warmup_iters,
            measure_iters,
            mean,
            percentile_from_sorted(&sorted, 0.50),
            percentile_from_sorted(&sorted, 0.90),
            percentile_from_sorted(&sorted, 0.95),
            sorted[0],
            sorted[sorted.len() - 1],
            stddev,
            cv_pct,
        );
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_out_of_place_log24() {
        run_out_of_place_case(1usize << MAX_SUPPORTED_LOG_ROWS);
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_in_place_log24() {
        run_in_place_case(1usize << MAX_SUPPORTED_LOG_ROWS);
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_out_of_place_log23() {
        run_out_of_place_case(1usize << MIN_SUPPORTED_LOG_ROWS);
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_in_place_log23() {
        run_in_place_case(1usize << MIN_SUPPORTED_LOG_ROWS);
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_single_invocation_log24_col1() {
        // Profiling-only entrypoint: launches one kernel chain without correctness checks.
        let rows = 1usize << MAX_SUPPORTED_LOG_ROWS;
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
    #[ignore]
    fn profile_hypercube_bitrev_bf_single_invocation_log23_col1() {
        // Profiling-only entrypoint: launches one kernel chain without correctness checks.
        let rows = 1usize << MIN_SUPPORTED_LOG_ROWS;
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
    #[ignore]
    fn profile_hypercube_bitrev_bf_multi_invocation_log24_col1() {
        run_profile_out_of_place_multi_invocation(
            1usize << MAX_SUPPORTED_LOG_ROWS,
            PROFILE_MULTI_WARMUP_ITERS,
            PROFILE_MULTI_MEASURE_ITERS,
        );
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_multi_invocation_log23_col1() {
        run_profile_out_of_place_multi_invocation(
            1usize << MIN_SUPPORTED_LOG_ROWS,
            PROFILE_MULTI_WARMUP_ITERS,
            PROFILE_MULTI_MEASURE_ITERS,
        );
    }

    #[test]
    #[should_panic]
    fn unsupported_log_rows_panics() {
        let len = 1usize << 22;
        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(len).unwrap();
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
    }

    #[test]
    #[should_panic]
    fn non_power_of_two_panics() {
        let len = (1usize << MAX_SUPPORTED_LOG_ROWS) - 1;
        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(len).unwrap();
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
    }
}
