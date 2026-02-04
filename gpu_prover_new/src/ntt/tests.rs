#![allow(non_snake_case)]

use std::alloc::Global;
use std::ops::Range;

use era_cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use era_cudart::slice::DeviceSlice;
use era_cudart::stream::CudaStream;
use fft::field_utils::domain_generator_for_size;
use fft::utils::bitreverse_enumeration_inplace;
use fft::{fft_natural_to_bitreversed, ifft_natural_to_natural, precompute_twiddles_for_fft};
use field::Field;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serial_test::serial;
use worker::Worker;

// use crate::device_context::DeviceContext;
use crate::device_structures::{
    DeviceMatrixChunk, DeviceMatrixChunkImpl, DeviceMatrixChunkMut, DeviceMatrixChunkMutImpl,
};
use crate::field::BaseField;
use crate::ntt::main_to_coset;

type BF = BaseField;

fn run_main_to_coset(
    log_n_range: Range<usize>,
    num_bf_cols: usize,
) {
    // let ctx = DeviceContext::create(12).unwrap();
    let n_max = 1 << (log_n_range.end - 1);
    let worker = Worker::new();
    let twiddles = precompute_twiddles_for_fft::<BF, Global, true>(n_max, &worker);

    let mut rng = rand::rng();
    const OFFSET: usize = 0;
    let max_stride: usize = n_max + OFFSET;
    let max_memory_size = (max_stride * num_bf_cols) as usize;
    // Using parallel rng generation, as in the benches, does not reduce runtime noticeably
    let mut inputs_orig_host =
        HostAllocation::<BF>::alloc(max_memory_size, CudaHostAllocFlags::DEFAULT).unwrap();
    inputs_orig_host.fill_with(|| BF::from_nonreduced_u32(rng.random()));
    let mut inputs_host =
        HostAllocation::<BF>::alloc(max_memory_size, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut outputs_host =
        HostAllocation::<BF>::alloc(max_memory_size, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut inplace_host =
        HostAllocation::<BF>::alloc(max_memory_size, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut inputs_device = DeviceAllocation::<BF>::alloc(max_memory_size).unwrap();
    let mut outputs_device = DeviceAllocation::<BF>::alloc(max_memory_size).unwrap();
    let mut inplace_device = DeviceAllocation::<BF>::alloc(max_memory_size).unwrap();
    let stream = CudaStream::default();
    for log_n in log_n_range {
        let n = (1 << log_n) as usize;
        let stride = n + OFFSET;
        let memory_size = stride * num_bf_cols;

        (&mut inputs_host[0..memory_size]).copy_from_slice(&inputs_orig_host[0..memory_size]);

        // out of place
        memory_copy_async(
            &mut inputs_device[0..memory_size],
            &inputs_host[0..memory_size],
            &stream,
        )
        .unwrap();
        let inputs_device_matrix =
            DeviceMatrixChunk::new(&inputs_device[0..memory_size], stride, OFFSET, n);
        let mut outputs_device_matrix =
            DeviceMatrixChunkMut::new(&mut outputs_device[0..memory_size], stride, OFFSET, n);
        main_to_coset(
            &inputs_device_matrix,
            &mut outputs_device_matrix,
            log_n,
            &stream,
        )
        .unwrap();
        memory_copy_async(
            &mut outputs_host[0..memory_size],
            &outputs_device[0..memory_size],
            &stream,
        )
        .unwrap();

        // in place
        memory_copy_async(
            &mut inplace_device[0..memory_size],
            &inputs_host[0..memory_size],
            &stream,
        )
        .unwrap();
        let inplace_output_view = &mut inplace_device[0..memory_size];
        let inplace_input_view = unsafe {
            DeviceSlice::from_raw_parts(inplace_output_view.as_ptr(), inplace_output_view.len())
        };
        let inplace_input_view_matrix =
            DeviceMatrixChunk::new(&inplace_input_view[0..memory_size], stride, OFFSET, n);
        let mut inplace_output_view_matrix =
            DeviceMatrixChunkMut::new(&mut inplace_output_view[0..memory_size], stride, OFFSET, n);
        main_to_coset(
            &inplace_input_view_matrix,
            &mut inplace_output_view_matrix,
            log_n,
            &stream,
        )
        .unwrap();
        memory_copy_async(
            &mut inplace_host[0..memory_size],
            inplace_output_view,
            &stream,
        )
        .unwrap();

        stream.synchronize().unwrap();

        for col in 0..num_bf_cols {
            let start = (col * stride + OFFSET) as usize;
            let range = start..start + n;
            bitreverse_enumeration_inplace(&mut outputs_host[range.clone()]);
            bitreverse_enumeration_inplace(&mut inplace_host[range.clone()]);
        }

        // Check forward variants against CPU forward results

        for ntt in 0..num_bf_cols {
            let start = ntt * stride + OFFSET as usize;
            let xs_range = start..start + n;
            let twiddles = &twiddles[..(n >> 1)];
            let gpu_results_out_of_place = &outputs_host[xs_range.clone()];
            let gpu_results_in_place = &inplace_host[xs_range.clone()];
            let mut cpu_refs: Vec<BF> = (&inputs_host[xs_range.clone()]).to_vec();
            ifft_natural_to_natural::<BF, BF, BF>(&mut cpu_refs, BF::ONE, twiddles);
            for k in 0..n {
                assert_eq!(
                    gpu_results_out_of_place[k],
                    cpu_refs[k],
                    "out of place 2^{} ntt {} k {}",
                    log_n,
                    ntt,
                    k,
                );
                assert_eq!(
                    gpu_results_in_place[k],
                    cpu_refs[k],
                    "in place 2^{} ntt {} k {}",
                    log_n,
                    ntt,
                    k,
                );
            }
        }
    }
    // ctx.destroy().unwrap();
}

#[test]
#[serial]
fn test_main_to_coset() {
    run_main_to_coset(24..25, 1);
}
