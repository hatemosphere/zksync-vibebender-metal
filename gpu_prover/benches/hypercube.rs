#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Bencher, Criterion, Throughput};
use era_criterion_cuda::CudaMeasurement;
use era_cudart::memory::{memory_copy_async, DeviceAllocation};
use era_cudart::result::CudaResult;
use era_cudart::stream::CudaStream;
use field::Field;
use gpu_prover::field::BF;
use gpu_prover::{
    hypercube_evals_into_coeffs_bitrev_bf, hypercube_evals_into_coeffs_bitrev_bf_in_place,
    DeviceMatrix, DeviceMatrixMut,
};

struct HypercubeBitrevBenchCase {
    rows: usize,
    cols: usize,
    d_src: DeviceAllocation<BF>,
    d_dst: DeviceAllocation<BF>,
}

impl HypercubeBitrevBenchCase {
    fn new(log_rows: u32, cols: usize, stream: &CudaStream) -> CudaResult<Self> {
        assert!((20..=24).contains(&log_rows));
        assert!(cols > 0);

        let rows = 1usize << log_rows;
        let len = rows * cols;

        let mut d_src = DeviceAllocation::alloc(len)?;
        let d_dst = DeviceAllocation::alloc(len)?;

        // Fill once to avoid benchmarking uninitialized memory reads.
        let h_src = vec![BF::ZERO; len];
        memory_copy_async(&mut d_src, &h_src, stream)?;
        stream.synchronize()?;

        Ok(Self {
            rows,
            cols,
            d_src,
            d_dst,
        })
    }

    fn run_out_of_place(&mut self, stream: &CudaStream) -> CudaResult<()> {
        let src_matrix = DeviceMatrix::new(&self.d_src, self.rows);
        let mut dst_matrix = DeviceMatrixMut::new(&mut self.d_dst, self.rows);
        hypercube_evals_into_coeffs_bitrev_bf(&src_matrix, &mut dst_matrix, stream)
    }

    fn run_in_place(&mut self, stream: &CudaStream) -> CudaResult<()> {
        let mut src_matrix = DeviceMatrixMut::new(&mut self.d_src, self.rows);
        hypercube_evals_into_coeffs_bitrev_bf_in_place(&mut src_matrix, stream)
    }

    fn bytes_per_transform(&self) -> u64 {
        // Approximate traffic: read + write per launch, with exactly 3 launches.
        (self.rows as u64) * (self.cols as u64) * (std::mem::size_of::<BF>() as u64) * 2 * 3
    }
}

const LOG_ROWS_CASES: [u32; 5] = [20, 21, 22, 23, 24];
const COL_CASES: [usize; 2] = [1, 2];

fn benchmark_out_of_place(c: &mut Criterion<CudaMeasurement>) {
    let stream = CudaStream::default();
    let mut group = c.benchmark_group("hypercube_bitrev_bf_out_of_place");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for log_rows in LOG_ROWS_CASES {
        for cols in COL_CASES {
            let mut bench_case = HypercubeBitrevBenchCase::new(log_rows, cols, &stream).unwrap();
            group.throughput(Throughput::Bytes(bench_case.bytes_per_transform()));
            group.bench_with_input(
                BenchmarkId::new("transform", format!("log_rows={log_rows},cols={cols}")),
                &(log_rows, cols),
                |b: &mut Bencher<CudaMeasurement>, _| {
                    b.iter(|| {
                        bench_case.run_out_of_place(&stream).unwrap();
                        stream.synchronize().unwrap();
                    })
                },
            );
        }
    }

    group.finish();
}

fn benchmark_in_place(c: &mut Criterion<CudaMeasurement>) {
    let stream = CudaStream::default();
    let mut group = c.benchmark_group("hypercube_bitrev_bf_in_place");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(4));

    for log_rows in LOG_ROWS_CASES {
        for cols in COL_CASES {
            let mut bench_case = HypercubeBitrevBenchCase::new(log_rows, cols, &stream).unwrap();
            group.throughput(Throughput::Bytes(bench_case.bytes_per_transform()));
            group.bench_with_input(
                BenchmarkId::new("transform", format!("log_rows={log_rows},cols={cols}")),
                &(log_rows, cols),
                |b: &mut Bencher<CudaMeasurement>, _| {
                    b.iter(|| {
                        bench_case.run_in_place(&stream).unwrap();
                        stream.synchronize().unwrap();
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    name = bench;
    config = Criterion::default().with_measurement::<CudaMeasurement>(CudaMeasurement {});
    targets = benchmark_out_of_place, benchmark_in_place
);
criterion_main!(bench);
