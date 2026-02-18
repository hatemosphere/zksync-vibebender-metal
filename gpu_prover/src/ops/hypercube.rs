use era_cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use era_cudart::paste::paste;
use era_cudart::result::CudaResult;
use era_cudart::stream::CudaStream;
use era_cudart::{cuda_kernel_declaration, cuda_kernel_signature_arguments_and_function};

use crate::device_structures::{
    DeviceMatrixChunkImpl, DeviceMatrixChunkMutImpl, MutPtrAndStride, PtrAndStride,
};
use crate::field::BF;

const MIN_LOG_ROWS: u32 = 20;
const MAX_LOG_ROWS: u32 = 24;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum KernelFamily {
    Initial,
    NonInitial,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum IntraWarpBackend {
    Shuffle,
    WarpShared,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct LaunchSpec {
    family: KernelFamily,
    rounds: u32,
    backend: IntraWarpBackend,
}

const fn spec(family: KernelFamily, rounds: u32, backend: IntraWarpBackend) -> LaunchSpec {
    LaunchSpec {
        family,
        rounds,
        backend,
    }
}

// These are the pre-tuned defaults used by the public API.
const DEFAULT_SCHEDULES: [[LaunchSpec; 3]; 5] = [
    [
        spec(KernelFamily::Initial, 8, IntraWarpBackend::WarpShared),
        spec(KernelFamily::NonInitial, 6, IntraWarpBackend::Shuffle),
        spec(KernelFamily::NonInitial, 6, IntraWarpBackend::Shuffle),
    ],
    [
        spec(KernelFamily::Initial, 8, IntraWarpBackend::WarpShared),
        spec(KernelFamily::NonInitial, 7, IntraWarpBackend::Shuffle),
        spec(KernelFamily::NonInitial, 6, IntraWarpBackend::Shuffle),
    ],
    [
        spec(KernelFamily::Initial, 8, IntraWarpBackend::WarpShared),
        spec(KernelFamily::NonInitial, 7, IntraWarpBackend::Shuffle),
        spec(KernelFamily::NonInitial, 7, IntraWarpBackend::Shuffle),
    ],
    [
        spec(KernelFamily::Initial, 8, IntraWarpBackend::WarpShared),
        spec(KernelFamily::NonInitial, 8, IntraWarpBackend::Shuffle),
        spec(KernelFamily::NonInitial, 7, IntraWarpBackend::WarpShared),
    ],
    [
        spec(KernelFamily::Initial, 8, IntraWarpBackend::WarpShared),
        spec(KernelFamily::NonInitial, 8, IntraWarpBackend::Shuffle),
        spec(KernelFamily::NonInitial, 8, IntraWarpBackend::WarpShared),
    ],
];

cuda_kernel_signature_arguments_and_function!(
    HypercubeBitrevBf,
    src: PtrAndStride<BF>,
    dst: MutPtrAndStride<BF>,
    start_stage: u32,
    log_rows: u32,
);

macro_rules! declare_h2m_kernel {
    ($family:ident, $rounds:literal, $backend:ident) => {
        paste! {
            cuda_kernel_declaration!(
                [<ab_h2m_bitrev_bf_ $family _ $rounds _ $backend _kernel>](
                    src: PtrAndStride<BF>,
                    dst: MutPtrAndStride<BF>,
                    start_stage: u32,
                    log_rows: u32,
                )
            );
        }
    };
}

declare_h2m_kernel!(initial, 8, shuffle);
declare_h2m_kernel!(initial, 9, shuffle);
declare_h2m_kernel!(initial, 10, shuffle);
declare_h2m_kernel!(initial, 11, shuffle);
declare_h2m_kernel!(initial, 12, shuffle);
declare_h2m_kernel!(initial, 8, warp_shared);
declare_h2m_kernel!(initial, 9, warp_shared);
declare_h2m_kernel!(initial, 10, warp_shared);
declare_h2m_kernel!(initial, 11, warp_shared);
declare_h2m_kernel!(initial, 12, warp_shared);
declare_h2m_kernel!(noninitial, 6, shuffle);
declare_h2m_kernel!(noninitial, 7, shuffle);
declare_h2m_kernel!(noninitial, 8, shuffle);
declare_h2m_kernel!(noninitial, 6, warp_shared);
declare_h2m_kernel!(noninitial, 7, warp_shared);
declare_h2m_kernel!(noninitial, 8, warp_shared);

fn default_schedule(log_rows: u32) -> [LaunchSpec; 3] {
    DEFAULT_SCHEDULES[(log_rows - MIN_LOG_ROWS) as usize]
}

fn block_threads_for_spec(spec: LaunchSpec) -> u32 {
    if spec.rounds <= 7 {
        128
    } else {
        256
    }
}

fn resolve_kernel(spec: LaunchSpec) -> HypercubeBitrevBfSignature {
    match (spec.family, spec.rounds, spec.backend) {
        (KernelFamily::Initial, 8, IntraWarpBackend::Shuffle) => {
            ab_h2m_bitrev_bf_initial_8_shuffle_kernel
        }
        (KernelFamily::Initial, 9, IntraWarpBackend::Shuffle) => {
            ab_h2m_bitrev_bf_initial_9_shuffle_kernel
        }
        (KernelFamily::Initial, 10, IntraWarpBackend::Shuffle) => {
            ab_h2m_bitrev_bf_initial_10_shuffle_kernel
        }
        (KernelFamily::Initial, 11, IntraWarpBackend::Shuffle) => {
            ab_h2m_bitrev_bf_initial_11_shuffle_kernel
        }
        (KernelFamily::Initial, 12, IntraWarpBackend::Shuffle) => {
            ab_h2m_bitrev_bf_initial_12_shuffle_kernel
        }
        (KernelFamily::Initial, 8, IntraWarpBackend::WarpShared) => {
            ab_h2m_bitrev_bf_initial_8_warp_shared_kernel
        }
        (KernelFamily::Initial, 9, IntraWarpBackend::WarpShared) => {
            ab_h2m_bitrev_bf_initial_9_warp_shared_kernel
        }
        (KernelFamily::Initial, 10, IntraWarpBackend::WarpShared) => {
            ab_h2m_bitrev_bf_initial_10_warp_shared_kernel
        }
        (KernelFamily::Initial, 11, IntraWarpBackend::WarpShared) => {
            ab_h2m_bitrev_bf_initial_11_warp_shared_kernel
        }
        (KernelFamily::Initial, 12, IntraWarpBackend::WarpShared) => {
            ab_h2m_bitrev_bf_initial_12_warp_shared_kernel
        }
        (KernelFamily::NonInitial, 6, IntraWarpBackend::Shuffle) => {
            ab_h2m_bitrev_bf_noninitial_6_shuffle_kernel
        }
        (KernelFamily::NonInitial, 7, IntraWarpBackend::Shuffle) => {
            ab_h2m_bitrev_bf_noninitial_7_shuffle_kernel
        }
        (KernelFamily::NonInitial, 8, IntraWarpBackend::Shuffle) => {
            ab_h2m_bitrev_bf_noninitial_8_shuffle_kernel
        }
        (KernelFamily::NonInitial, 6, IntraWarpBackend::WarpShared) => {
            ab_h2m_bitrev_bf_noninitial_6_warp_shared_kernel
        }
        (KernelFamily::NonInitial, 7, IntraWarpBackend::WarpShared) => {
            ab_h2m_bitrev_bf_noninitial_7_warp_shared_kernel
        }
        (KernelFamily::NonInitial, 8, IntraWarpBackend::WarpShared) => {
            ab_h2m_bitrev_bf_noninitial_8_warp_shared_kernel
        }
        _ => panic!("unsupported launch spec: {spec:?}"),
    }
}

fn launch_with_schedule(
    src: &(impl DeviceMatrixChunkImpl<BF> + ?Sized),
    dst: &mut (impl DeviceMatrixChunkMutImpl<BF> + ?Sized),
    schedule: &[LaunchSpec; 3],
    stream: &CudaStream,
) -> CudaResult<()> {
    let rows = src.rows();
    let cols = src.cols();
    assert_eq!(dst.rows(), rows);
    assert_eq!(dst.cols(), cols);
    if cols == 0 {
        return Ok(());
    }

    assert!(rows.is_power_of_two());
    let log_rows = rows.trailing_zeros();
    assert!((MIN_LOG_ROWS..=MAX_LOG_ROWS).contains(&log_rows));

    let mut start_stage = 0u32;
    let mut launch_src = src.as_ptr_and_stride();
    let launch_dst = dst.as_mut_ptr_and_stride();
    let dst_as_src = PtrAndStride::new(launch_dst.ptr as *const BF, launch_dst.stride);
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
        let grid_dim = Dim3 {
            x: subproblems as u32,
            y: cols as u32,
            z: 1,
        };
        let block_dim = block_threads_for_spec(spec);
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let args = HypercubeBitrevBfArguments::new(
            launch_src,
            launch_dst,
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
    src: &(impl DeviceMatrixChunkImpl<BF> + ?Sized),
    dst: &mut (impl DeviceMatrixChunkMutImpl<BF> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    let rows = src.rows();
    assert!(rows.is_power_of_two());
    let log_rows = rows.trailing_zeros();
    assert!((MIN_LOG_ROWS..=MAX_LOG_ROWS).contains(&log_rows));
    launch_with_schedule(src, dst, &default_schedule(log_rows), stream)
}

pub fn hypercube_evals_into_coeffs_bitrev_bf_in_place(
    values: &mut (impl DeviceMatrixChunkMutImpl<BF> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    let rows = values.rows();
    assert!(rows.is_power_of_two());
    let log_rows = rows.trailing_zeros();
    assert!((MIN_LOG_ROWS..=MAX_LOG_ROWS).contains(&log_rows));

    let schedule = default_schedule(log_rows);
    let src = values.as_ptr_and_stride();
    let dst = values.as_mut_ptr_and_stride();

    let cols = values.cols();
    if cols == 0 {
        return Ok(());
    }

    let mut start_stage = 0u32;
    for (idx, spec) in schedule.iter().copied().enumerate() {
        if idx == 0 {
            assert_eq!(spec.family, KernelFamily::Initial);
        } else {
            assert_eq!(spec.family, KernelFamily::NonInitial);
        }

        let subproblems = rows >> (spec.rounds as usize);
        let grid_dim = Dim3 {
            x: subproblems as u32,
            y: cols as u32,
            z: 1,
        };
        let block_dim = block_threads_for_spec(spec);
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let args = HypercubeBitrevBfArguments::new(src, dst, start_stage, log_rows);
        HypercubeBitrevBfFunction(resolve_kernel(spec)).launch(&config, &args)?;
        start_stage += spec.rounds;
    }

    assert_eq!(start_stage, log_rows);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_structures::{DeviceMatrix, DeviceMatrixMut};
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

    fn cpu_reference_bitrev(src: &[BF], rows: usize, cols: usize) -> Vec<BF> {
        let mut out = vec![BF::ZERO; src.len()];
        for col in 0..cols {
            let offset = col * rows;
            let mut natural = bitreverse_permute(&src[offset..offset + rows]);
            multivariate_hypercube_evals_into_coeffs(&mut natural, rows.trailing_zeros());
            let expected = bitreverse_permute(&natural);
            out[offset..offset + rows].copy_from_slice(&expected);
        }
        out
    }

    fn run_case(log_rows: u32, cols: usize, in_place: bool) {
        let rows = 1usize << log_rows;
        let len = rows * cols;
        let mut rng = rng();
        let h_src: Vec<BF> = (0..len).map(|_| BF::random_element(&mut rng)).collect();
        let expected = cpu_reference_bitrev(&h_src, rows, cols);

        let stream = CudaStream::default();
        let mut h_dst = vec![BF::ZERO; len];

        if in_place {
            let mut d_values = DeviceAllocation::alloc(len).unwrap();
            memory_copy_async(&mut d_values, &h_src, &stream).unwrap();
            let mut matrix = DeviceMatrixMut::new(&mut d_values, rows);
            hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut matrix, &stream).unwrap();
            memory_copy_async(&mut h_dst, &d_values, &stream).unwrap();
        } else {
            let mut d_src = DeviceAllocation::alloc(len).unwrap();
            let mut d_dst = DeviceAllocation::alloc(len).unwrap();
            memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
            let src_matrix = DeviceMatrix::new(&d_src, rows);
            let mut dst_matrix = DeviceMatrixMut::new(&mut d_dst, rows);
            hypercube_evals_into_coeffs_bitrev_bf(&src_matrix, &mut dst_matrix, &stream).unwrap();
            memory_copy_async(&mut h_dst, &d_dst, &stream).unwrap();
        }

        stream.synchronize().unwrap();
        assert_eq!(h_dst, expected);
    }

    #[test]
    fn hypercube_bitrev_bf_out_of_place_log20() {
        run_case(20, 2, false);
    }

    #[test]
    fn hypercube_bitrev_bf_in_place_log20() {
        run_case(20, 2, true);
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_out_of_place_log20_to_24() {
        for log_rows in 20..=24 {
            run_case(log_rows, 1, false);
        }
    }

    #[test]
    #[ignore]
    fn hypercube_bitrev_bf_in_place_log20_to_24() {
        for log_rows in 20..=24 {
            run_case(log_rows, 1, true);
        }
    }

    #[test]
    #[ignore]
    fn profile_hypercube_bitrev_bf_single_invocation_log24_col1() {
        let log_rows = 24u32;
        let rows = 1usize << log_rows;
        let cols = 1usize;
        let len = rows * cols;

        let stream = CudaStream::default();
        let h_src = vec![BF::ZERO; len];
        let mut d_src = DeviceAllocation::alloc(len).unwrap();
        let mut d_dst = DeviceAllocation::alloc(len).unwrap();

        memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
        stream.synchronize().unwrap();

        let src_matrix = DeviceMatrix::new(&d_src, rows);
        let mut dst_matrix = DeviceMatrixMut::new(&mut d_dst, rows);
        hypercube_evals_into_coeffs_bitrev_bf(&src_matrix, &mut dst_matrix, &stream).unwrap();
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
    #[should_panic]
    fn unsupported_log_rows_panics() {
        let rows = 1usize << 19;
        let cols = 1usize;
        let len = rows * cols;
        let stream = CudaStream::default();
        let mut d_values = DeviceAllocation::alloc(len).unwrap();
        let mut matrix = DeviceMatrixMut::new(&mut d_values, rows);
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut matrix, &stream).unwrap();
    }

    #[test]
    #[ignore]
    fn tune_schedules_log20_to_24() {
        let backend_sets = [
            [
                IntraWarpBackend::Shuffle,
                IntraWarpBackend::Shuffle,
                IntraWarpBackend::Shuffle,
            ],
            [
                IntraWarpBackend::WarpShared,
                IntraWarpBackend::Shuffle,
                IntraWarpBackend::Shuffle,
            ],
            [
                IntraWarpBackend::Shuffle,
                IntraWarpBackend::Shuffle,
                IntraWarpBackend::WarpShared,
            ],
            [
                IntraWarpBackend::WarpShared,
                IntraWarpBackend::WarpShared,
                IntraWarpBackend::WarpShared,
            ],
        ];

        let split_candidates: &[(u32, &[[u32; 3]])] = &[
            (20, &[[8, 6, 6]]),
            (21, &[[8, 7, 6], [9, 6, 6]]),
            (22, &[[8, 7, 7], [8, 8, 6], [9, 7, 6], [10, 6, 6]]),
            (23, &[[8, 8, 7], [9, 7, 7], [10, 7, 6], [11, 6, 6]]),
            (24, &[[8, 8, 8], [9, 8, 7], [10, 7, 7], [11, 7, 6], [12, 6, 6]]),
        ];

        let stream = CudaStream::default();
        let repeats = 8usize;
        let cols_variants = [1usize, 8usize];

        for (log_rows, splits) in split_candidates.iter().copied() {
            let rows = 1usize << log_rows;
            for cols in cols_variants {
                let len = rows * cols;
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

                    for backends in backend_sets {
                        let schedule = [
                            spec(KernelFamily::Initial, split[0], backends[0]),
                            spec(KernelFamily::NonInitial, split[1], backends[1]),
                            spec(KernelFamily::NonInitial, split[2], backends[2]),
                        ];
                        let src_matrix = DeviceMatrix::new(&d_src, rows);
                        let mut dst_matrix = DeviceMatrixMut::new(&mut d_dst, rows);

                        let now = Instant::now();
                        for _ in 0..repeats {
                            launch_with_schedule(&src_matrix, &mut dst_matrix, &schedule, &stream)
                                .unwrap();
                        }
                        stream.synchronize().unwrap();
                        let elapsed = now.elapsed().as_secs_f64();

                        let bytes_per_run = (rows * cols * size_of::<BF>() * 2 * 3) as f64;
                        let gbps = (bytes_per_run * repeats as f64) / elapsed / 1e9;
                        if gbps > best_gbps {
                            best_gbps = gbps;
                            best_label = format!("split={split:?}, backends={backends:?}");
                        }
                    }
                }

                println!(
                    "tuning log_rows={log_rows}, cols={cols}: best {best_label}, {:.2} GB/s",
                    best_gbps
                );
            }
        }
    }
}
