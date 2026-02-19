use era_cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use era_cudart::result::CudaResult;
use era_cudart::slice::DeviceSlice;
use era_cudart::stream::CudaStream;
use era_cudart::{cuda_kernel_declaration, cuda_kernel_signature_arguments_and_function};

use crate::field::BF;

const MIN_LOG_ROWS: u32 = 20;
const MAX_LOG_ROWS: u32 = 24;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum KernelFamily {
    Initial,
    NonInitial,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct LaunchSpec {
    family: KernelFamily,
    rounds: u32,
}

const fn spec(family: KernelFamily, rounds: u32) -> LaunchSpec {
    LaunchSpec { family, rounds }
}

// These are the pre-tuned defaults used by the public API.
const DEFAULT_SCHEDULES: [[LaunchSpec; 3]; 5] = [
    [
        spec(KernelFamily::Initial, 8),
        spec(KernelFamily::NonInitial, 6),
        spec(KernelFamily::NonInitial, 6),
    ],
    [
        spec(KernelFamily::Initial, 9),
        spec(KernelFamily::NonInitial, 6),
        spec(KernelFamily::NonInitial, 6),
    ],
    [
        spec(KernelFamily::Initial, 8),
        spec(KernelFamily::NonInitial, 8),
        spec(KernelFamily::NonInitial, 6),
    ],
    [
        spec(KernelFamily::Initial, 10),
        spec(KernelFamily::NonInitial, 7),
        spec(KernelFamily::NonInitial, 6),
    ],
    [
        spec(KernelFamily::Initial, 12),
        spec(KernelFamily::NonInitial, 6),
        spec(KernelFamily::NonInitial, 6),
    ],
];

cuda_kernel_signature_arguments_and_function!(
    HypercubeBitrevBf,
    src: *const BF,
    dst: *mut BF,
    use_cg_loads: u32,
    start_stage: u32,
    log_rows: u32,
);

macro_rules! declare_h2m_kernel {
    ($name:ident) => {
        cuda_kernel_declaration!(
            $name(
                src: *const BF,
                dst: *mut BF,
                use_cg_loads: u32,
                start_stage: u32,
                log_rows: u32,
            )
        );
    };
}

declare_h2m_kernel!(ab_h2m_bitrev_bf_initial_8_kernel);
declare_h2m_kernel!(ab_h2m_bitrev_bf_initial_9_kernel);
declare_h2m_kernel!(ab_h2m_bitrev_bf_initial_10_kernel);
declare_h2m_kernel!(ab_h2m_bitrev_bf_initial_11_kernel);
declare_h2m_kernel!(ab_h2m_bitrev_bf_initial_12_kernel);
declare_h2m_kernel!(ab_h2m_bitrev_bf_noninitial_6_kernel);
declare_h2m_kernel!(ab_h2m_bitrev_bf_noninitial_7_kernel);
declare_h2m_kernel!(ab_h2m_bitrev_bf_noninitial_8_kernel);
declare_h2m_kernel!(ab_h2m_bitrev_bf_noninitial_7_128_kernel);

fn default_schedule(log_rows: u32) -> [LaunchSpec; 3] {
    DEFAULT_SCHEDULES[(log_rows - MIN_LOG_ROWS) as usize]
}

fn block_threads_for_spec(spec: LaunchSpec) -> u32 {
    if spec.family == KernelFamily::NonInitial && spec.rounds == 7 {
        128
    } else if spec.family == KernelFamily::NonInitial {
        256
    } else if spec.rounds <= 7 {
        128
    } else {
        256
    }
}

fn noninitial_tile_subproblems(spec: LaunchSpec) -> usize {
    match (spec.rounds, block_threads_for_spec(spec)) {
        (6, 256) => 128,
        (7, 256) => 64,
        (7, 128) => 32,
        (8, 256) => 32,
        _ => panic!("unsupported noninitial spec: {spec:?}"),
    }
}

fn resolve_kernel(spec: LaunchSpec) -> HypercubeBitrevBfSignature {
    match (spec.family, spec.rounds) {
        (KernelFamily::Initial, 8) => ab_h2m_bitrev_bf_initial_8_kernel,
        (KernelFamily::Initial, 9) => ab_h2m_bitrev_bf_initial_9_kernel,
        (KernelFamily::Initial, 10) => ab_h2m_bitrev_bf_initial_10_kernel,
        (KernelFamily::Initial, 11) => ab_h2m_bitrev_bf_initial_11_kernel,
        (KernelFamily::Initial, 12) => ab_h2m_bitrev_bf_initial_12_kernel,
        (KernelFamily::NonInitial, 6) => ab_h2m_bitrev_bf_noninitial_6_kernel,
        (KernelFamily::NonInitial, 7) => {
            if block_threads_for_spec(spec) == 128 {
                ab_h2m_bitrev_bf_noninitial_7_128_kernel
            } else {
                ab_h2m_bitrev_bf_noninitial_7_kernel
            }
        }
        (KernelFamily::NonInitial, 8) => ab_h2m_bitrev_bf_noninitial_8_kernel,
        _ => panic!("unsupported launch spec: {spec:?}"),
    }
}

fn launch_with_schedule(
    src: &DeviceSlice<BF>,
    dst: &mut DeviceSlice<BF>,
    schedule: &[LaunchSpec; 3],
    stream: &CudaStream,
) -> CudaResult<()> {
    let rows = src.len();
    assert_eq!(dst.len(), rows);
    assert!(rows.is_power_of_two());
    let log_rows = rows.trailing_zeros();
    assert!((MIN_LOG_ROWS..=MAX_LOG_ROWS).contains(&log_rows));

    let mut start_stage = 0u32;
    let mut launch_src = src.as_ptr();
    let launch_dst = dst.as_mut_ptr();
    let dst_as_src = launch_dst as *const BF;
    for (idx, spec) in schedule.iter().copied().enumerate() {
        if idx == 0 {
            assert_eq!(spec.family, KernelFamily::Initial);
            assert!((8..=12).contains(&spec.rounds));
        } else {
            assert_eq!(spec.family, KernelFamily::NonInitial);
            assert!((6..=8).contains(&spec.rounds));
        }

        assert!(start_stage + spec.rounds <= log_rows);

        let subproblems = rows >> (spec.rounds as usize);
        let grid_x = if spec.family == KernelFamily::Initial {
            subproblems
        } else {
            let tile = noninitial_tile_subproblems(spec);
            assert_eq!(subproblems % tile, 0);
            subproblems / tile
        };
        let grid_dim = Dim3 {
            x: grid_x as u32,
            y: 1,
            z: 1,
        };
        let block_dim = block_threads_for_spec(spec);
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let use_cg_loads = if idx == 2 { 1 } else { 0 };
        let args = HypercubeBitrevBfArguments::new(
            launch_src,
            launch_dst,
            use_cg_loads,
            start_stage,
            log_rows,
        );

        HypercubeBitrevBfFunction(resolve_kernel(spec)).launch(&config, &args)?;
        launch_src = dst_as_src;
        start_stage += spec.rounds;
    }

    assert_eq!(start_stage, log_rows);
    Ok(())
}

pub fn hypercube_evals_into_coeffs_bitrev_bf(
    src: &DeviceSlice<BF>,
    dst: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    let rows = src.len();
    assert_eq!(dst.len(), rows);
    assert!(rows.is_power_of_two());
    let log_rows = rows.trailing_zeros();
    assert!((MIN_LOG_ROWS..=MAX_LOG_ROWS).contains(&log_rows));
    launch_with_schedule(src, dst, &default_schedule(log_rows), stream)
}

pub fn hypercube_evals_into_coeffs_bitrev_bf_in_place(
    values: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    let rows = values.len();
    assert!(rows.is_power_of_two());
    let log_rows = rows.trailing_zeros();
    assert!((MIN_LOG_ROWS..=MAX_LOG_ROWS).contains(&log_rows));

    let schedule = default_schedule(log_rows);
    let src = values.as_ptr();
    let dst = values.as_mut_ptr();

    let mut start_stage = 0u32;
    for (idx, spec) in schedule.iter().copied().enumerate() {
        if idx == 0 {
            assert_eq!(spec.family, KernelFamily::Initial);
            assert!((8..=12).contains(&spec.rounds));
        } else {
            assert_eq!(spec.family, KernelFamily::NonInitial);
        }

        let subproblems = rows >> (spec.rounds as usize);
        let grid_x = if spec.family == KernelFamily::Initial {
            subproblems
        } else {
            let tile = noninitial_tile_subproblems(spec);
            assert_eq!(subproblems % tile, 0);
            subproblems / tile
        };
        let grid_dim = Dim3 {
            x: grid_x as u32,
            y: 1,
            z: 1,
        };
        let block_dim = block_threads_for_spec(spec);
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let use_cg_loads = if idx == 2 { 1 } else { 0 };
        let args = HypercubeBitrevBfArguments::new(src, dst, use_cg_loads, start_stage, log_rows);
        HypercubeBitrevBfFunction(resolve_kernel(spec)).launch(&config, &args)?;
        start_stage += spec.rounds;
    }

    assert_eq!(start_stage, log_rows);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use era_cudart::memory::{memory_copy_async, DeviceAllocation};
    use field::{Field, Rand};
    use prover::gkr::whir::hypercube_to_monomial::multivariate_hypercube_evals_into_coeffs;
    use rand::rng;
    use std::mem::size_of;
    use std::time::Instant;

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

    fn run_case(log_rows: u32, in_place: bool) {
        let rows = 1usize << log_rows;
        let len = rows;
        let mut rng = rng();
        let h_src: Vec<BF> = (0..len).map(|_| BF::random_element(&mut rng)).collect();
        let expected = cpu_reference_bitrev(&h_src, rows);

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
    fn hypercube_bitrev_bf_out_of_place_log20() {
        run_case(20, false);
    }

    #[test]
    fn hypercube_bitrev_bf_in_place_log20() {
        run_case(20, true);
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_out_of_place_log20_to_24() {
        for log_rows in 20..=24 {
            run_case(log_rows, false);
        }
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_in_place_log20_to_24() {
        for log_rows in 20..=24 {
            run_case(log_rows, true);
        }
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_single_invocation_log24_col1() {
        let log_rows = 24u32;
        let rows = 1usize << log_rows;
        let len = rows;

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
    fn schedule_table_is_well_formed() {
        for (i, schedule) in DEFAULT_SCHEDULES.iter().enumerate() {
            let log_rows = MIN_LOG_ROWS + i as u32;
            assert_eq!(schedule[0].family, KernelFamily::Initial);
            assert_eq!(schedule[1].family, KernelFamily::NonInitial);
            assert_eq!(schedule[2].family, KernelFamily::NonInitial);
            let rounds_sum: u32 = schedule.iter().map(|spec| spec.rounds).sum();
            assert_eq!(rounds_sum, log_rows);
        }
    }

    #[test]
    fn schedule_log24_is_12_6_6() {
        let schedule = default_schedule(24);
        assert_eq!(schedule[0], spec(KernelFamily::Initial, 12));
        assert_eq!(schedule[1], spec(KernelFamily::NonInitial, 6));
        assert_eq!(schedule[2], spec(KernelFamily::NonInitial, 6));
    }

    #[test]
    #[should_panic]
    fn unsupported_log_rows_panics() {
        let rows = 1usize << 19;
        let len = rows;
        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(len).unwrap();
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
    }

    #[test]
    #[should_panic]
    fn non_power_of_two_panics() {
        let len = (1usize << 20) - 1;
        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(len).unwrap();
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut d_values, &stream).unwrap();
    }

    #[test]
    #[ignore]
    fn tune_schedules_log20_to_24() {
        let split_candidates: &[(u32, &[[u32; 3]])] = &[
            (20, &[[8, 6, 6]]),
            (21, &[[8, 7, 6], [9, 6, 6]]),
            (22, &[[8, 7, 7], [8, 8, 6], [9, 7, 6], [10, 6, 6]]),
            (23, &[[8, 8, 7], [9, 7, 7], [10, 7, 6], [11, 6, 6]]),
            (24, &[[8, 8, 8], [9, 8, 7], [10, 7, 7], [11, 7, 6], [12, 6, 6]]),
        ];

        let stream = CudaStream::default();
        let repeats = 8usize;
        for (log_rows, splits) in split_candidates.iter().copied() {
            let rows = 1usize << log_rows;
            let len = rows;
            let mut rng = rng();
            let h_src: Vec<BF> = (0..len).map(|_| BF::random_element(&mut rng)).collect();
            let mut d_src = DeviceAllocation::alloc(len).unwrap();
            let mut d_dst = DeviceAllocation::alloc(len).unwrap();
            memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
            stream.synchronize().unwrap();

            let mut best_gbps = 0.0f64;
            let mut best_label = String::new();

            for split in splits {
                if split[0] < 8 || split[0] > 12 {
                    continue;
                }
                if split[1] < 6 || split[1] > 8 || split[2] < 6 || split[2] > 8 {
                    continue;
                }

                let schedule = [
                    spec(KernelFamily::Initial, split[0]),
                    spec(KernelFamily::NonInitial, split[1]),
                    spec(KernelFamily::NonInitial, split[2]),
                ];
                let now = Instant::now();
                for repeat_idx in 0..repeats {
                    if let Err(err) = launch_with_schedule(&d_src, &mut d_dst, &schedule, &stream) {
                        panic!(
                            "launch failure: log_rows={log_rows}, split={split:?}, repeat={repeat_idx}, error={err:?}"
                        );
                    }
                    if let Err(err) = stream.synchronize() {
                        panic!(
                            "sync failure: log_rows={log_rows}, split={split:?}, repeat={repeat_idx}, error={err:?}"
                        );
                    }
                }
                let elapsed = now.elapsed().as_secs_f64();

                let bytes_per_run = (rows * size_of::<BF>() * 2 * 3) as f64;
                let gbps = (bytes_per_run * repeats as f64) / elapsed / 1e9;
                if gbps > best_gbps {
                    best_gbps = gbps;
                    best_label = format!("split={split:?}");
                }
            }

            println!(
                "tuning log_rows={log_rows}: best {best_label}, {:.2} GB/s",
                best_gbps
            );
        }
    }
}
