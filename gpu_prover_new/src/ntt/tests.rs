#![allow(non_snake_case)]

use std::alloc::Global;
use std::ops::Range;

use era_cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use era_cudart::slice::DeviceSlice;
use era_cudart::stream::CudaStream;
use fft::field_utils::domain_generator_for_size;
use fft::utils::bitreverse_enumeration_inplace;
use fft::{
    ifft_natural_to_natural, precompute_twiddles_for_fft, serial_ct_ntt_bitreversed_to_natural,
};
use field::Field;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serial_test::serial;
use worker::Worker;

use crate::device_context::DeviceContext;
use crate::device_structures::{
    DeviceMatrixChunk, DeviceMatrixChunkImpl, DeviceMatrixChunkMut, DeviceMatrixChunkMutImpl,
};
use crate::field::BaseField;
use crate::ntt::{main_to_monomials_2_pass, main_to_monomials_3_pass, monomials_to_evals_3_pass};
use crate::ops::complex::bit_reverse_in_place;

type BF = BaseField;

#[derive(PartialEq)]
enum Passes {
    Two,
    Three,
}

#[derive(PartialEq)]
enum InOrOutOfPlace {
    In,
    Out,
}

fn transpose_monomials(vals: &mut [BF]) {
    for chunk in vals.chunks_mut(1024) {
        for row in 0..32 {
            for col in 0..row {
                let a = chunk[row * 32 + col];
                let b = chunk[col * 32 + row];
                chunk[row * 32 + col] = b;
                chunk[col * 32 + row] = a;
            }
        }
    }
}

fn run_main_to_monomials(
    log_n_range: Range<usize>,
    num_bf_cols: usize,
    passes_variant: Passes,
    in_or_out_of_place: InOrOutOfPlace,
    transposed_monomials: bool,
) {
    let ctx = DeviceContext::create(12).unwrap();
    let n_max = 1 << (log_n_range.end - 1);
    let worker = Worker::new();
    let stream = CudaStream::default();
    let twiddles = precompute_twiddles_for_fft::<BF, Global, true>(n_max, &worker);

    let mut rng = rand::rng();
    const OFFSET: usize = 0;
    let max_stride: usize = n_max + OFFSET;
    let max_memory_size = (max_stride * num_bf_cols) as usize;
    let flush_l2_size = 1 << 26;
    // Using parallel rng generation, as in the benches, does not reduce runtime noticeably
    let mut inputs_orig_host =
        HostAllocation::<BF>::alloc(max_memory_size, CudaHostAllocFlags::DEFAULT).unwrap();
    inputs_orig_host.fill_with(|| BF::from_nonreduced_u32(rng.random()));
    let mut inputs_host =
        HostAllocation::<BF>::alloc(max_memory_size, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut outputs_host =
        HostAllocation::<BF>::alloc(max_memory_size, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut flush_l2_host =
        HostAllocation::<BF>::alloc(flush_l2_size, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut inputs_device = DeviceAllocation::<BF>::alloc(max_memory_size).unwrap();
    let mut outputs_device = DeviceAllocation::<BF>::alloc(max_memory_size).unwrap();
    let mut flush_l2_device = DeviceAllocation::<BF>::alloc(1 << 26).unwrap();
    let mut flush_l2 = || {
        memory_copy_async(&mut flush_l2_device[..], &flush_l2_host[..], &stream).unwrap();
    };
    for log_n in log_n_range {
        let n = (1 << log_n) as usize;
        let stride = n + OFFSET;
        let memory_size = stride * num_bf_cols;

        (&mut inputs_host[0..memory_size]).copy_from_slice(&inputs_orig_host[0..memory_size]);

        match in_or_out_of_place {
            InOrOutOfPlace::Out => {
                memory_copy_async(
                    &mut inputs_device[0..memory_size],
                    &inputs_host[0..memory_size],
                    &stream,
                )
                .unwrap();
                flush_l2();
                let inputs_device_matrix =
                    DeviceMatrixChunk::new(&inputs_device[0..memory_size], stride, OFFSET, n);
                let mut outputs_device_matrix = DeviceMatrixChunkMut::new(
                    &mut outputs_device[0..memory_size],
                    stride,
                    OFFSET,
                    n,
                );
                match passes_variant {
                    Passes::Two => main_to_monomials_2_pass(
                        &inputs_device_matrix,
                        &mut outputs_device_matrix,
                        log_n,
                        transposed_monomials,
                        &stream,
                    )
                    .unwrap(),
                    Passes::Three => main_to_monomials_3_pass(
                        &inputs_device_matrix,
                        &mut outputs_device_matrix,
                        log_n,
                        transposed_monomials,
                        &stream,
                    )
                    .unwrap(),
                };
                memory_copy_async(
                    &mut outputs_host[0..memory_size],
                    &outputs_device[0..memory_size],
                    &stream,
                )
                .unwrap();
            }
            InOrOutOfPlace::In => {
                memory_copy_async(
                    &mut inputs_device[0..memory_size],
                    &inputs_host[0..memory_size],
                    &stream,
                )
                .unwrap();
                flush_l2();
                let inplace_output_view = &mut inputs_device[0..memory_size];
                let inplace_input_view = unsafe {
                    DeviceSlice::from_raw_parts(
                        inplace_output_view.as_ptr(),
                        inplace_output_view.len(),
                    )
                };
                let inplace_input_view_matrix =
                    DeviceMatrixChunk::new(&inplace_input_view[0..memory_size], stride, OFFSET, n);
                let mut inplace_output_view_matrix = DeviceMatrixChunkMut::new(
                    &mut inplace_output_view[0..memory_size],
                    stride,
                    OFFSET,
                    n,
                );
                match passes_variant {
                    Passes::Two => main_to_monomials_2_pass(
                        &inplace_input_view_matrix,
                        &mut inplace_output_view_matrix,
                        log_n,
                        transposed_monomials,
                        &stream,
                    )
                    .unwrap(),
                    Passes::Three => main_to_monomials_3_pass(
                        &inplace_input_view_matrix,
                        &mut inplace_output_view_matrix,
                        log_n,
                        transposed_monomials,
                        &stream,
                    )
                    .unwrap(),
                };
                memory_copy_async(
                    &mut outputs_host[0..memory_size],
                    inplace_output_view,
                    &stream,
                )
                .unwrap();
            }
        }

        stream.synchronize().unwrap();

        for col in 0..num_bf_cols {
            let start = (col * stride + OFFSET) as usize;
            let range = start..start + n;
            if transposed_monomials {
                transpose_monomials(&mut outputs_host[range.clone()]);
            }
            bitreverse_enumeration_inplace(&mut outputs_host[range]);
        }

        for ntt in 0..num_bf_cols {
            let start = ntt * stride + OFFSET as usize;
            let xs_range = start..start + n;
            let twiddles = &twiddles[..(n >> 1)];
            let gpu_results = &outputs_host[xs_range.clone()];
            let mut cpu_refs: Vec<BF> = (&inputs_host[xs_range.clone()]).to_vec();
            ifft_natural_to_natural::<BF, BF, BF>(&mut cpu_refs, BF::ONE, twiddles);
            for k in 0..n {
                assert_eq!(
                    gpu_results[k], cpu_refs[k],
                    "2^{} ntt {} k {}",
                    log_n, ntt, k
                );
            }
        }
    }
    ctx.destroy().unwrap();
}

fn run_monomials_to_evals(
    log_n_range: Range<usize>,
    num_bf_cols: usize,
    passes_variant: Passes,
    in_or_out_of_place: InOrOutOfPlace,
    transposed_monomials: bool,
) {
    let ctx = DeviceContext::create(12).unwrap();
    let n_max = 1 << (log_n_range.end - 1);
    let worker = Worker::new();
    let stream = CudaStream::default();
    let twiddles = precompute_twiddles_for_fft::<BF, Global, false>(n_max, &worker);

    let mut rng = rand::rng();
    const OFFSET: usize = 0;
    let max_stride: usize = n_max + OFFSET;
    let max_memory_size = (max_stride * num_bf_cols) as usize;
    let flush_l2_size = 1 << 26;
    // Using parallel rng generation, as in the benches, does not reduce runtime noticeably
    let mut inputs_orig_host =
        HostAllocation::<BF>::alloc(max_memory_size, CudaHostAllocFlags::DEFAULT).unwrap();
    inputs_orig_host.fill_with(|| BF::from_nonreduced_u32(rng.random()));
    let mut inputs_host =
        HostAllocation::<BF>::alloc(max_memory_size, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut outputs_host =
        HostAllocation::<BF>::alloc(max_memory_size, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut flush_l2_host =
        HostAllocation::<BF>::alloc(flush_l2_size, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut inputs_device = DeviceAllocation::<BF>::alloc(max_memory_size).unwrap();
    let mut outputs_device = DeviceAllocation::<BF>::alloc(max_memory_size).unwrap();
    let mut flush_l2_device = DeviceAllocation::<BF>::alloc(1 << 26).unwrap();
    let mut flush_l2 = || {
        memory_copy_async(&mut flush_l2_device[..], &flush_l2_host[..], &stream).unwrap();
    };
    for log_n in log_n_range {
        let n = (1 << log_n) as usize;
        let stride = n + OFFSET;
        let memory_size = stride * num_bf_cols;

        (&mut inputs_host[0..memory_size]).copy_from_slice(&inputs_orig_host[0..memory_size]);

        for col in 0..num_bf_cols {
            let start = (col * stride + OFFSET) as usize;
            let range = start..start + n;
            if transposed_monomials {
                transpose_monomials(&mut inputs_host[range]);
            }
        }
        memory_copy_async(
            &mut inputs_device[0..memory_size],
            &inputs_host[0..memory_size],
            &stream,
        )
        .unwrap();
        flush_l2();

        match in_or_out_of_place {
            InOrOutOfPlace::Out => {
                let inputs_device_matrix =
                    DeviceMatrixChunk::new(&inputs_device[0..memory_size], stride, OFFSET, n);
                let mut outputs_device_matrix = DeviceMatrixChunkMut::new(
                    &mut outputs_device[0..memory_size],
                    stride,
                    OFFSET,
                    n,
                );
                match passes_variant {
                    Passes::Three => monomials_to_evals_3_pass(
                        &inputs_device_matrix,
                        &mut outputs_device_matrix,
                        log_n,
                        transposed_monomials,
                        &stream,
                    )
                    .unwrap(),
                    _ => {}
                };
                memory_copy_async(
                    &mut outputs_host[0..memory_size],
                    &outputs_device[0..memory_size],
                    &stream,
                )
                .unwrap();
            }
            InOrOutOfPlace::In => {
                let inplace_output_view = &mut inputs_device[0..memory_size];
                let inplace_input_view = unsafe {
                    DeviceSlice::from_raw_parts(
                        inplace_output_view.as_ptr(),
                        inplace_output_view.len(),
                    )
                };
                let inplace_input_view_matrix =
                    DeviceMatrixChunk::new(&inplace_input_view[0..memory_size], stride, OFFSET, n);
                let mut inplace_output_view_matrix = DeviceMatrixChunkMut::new(
                    &mut inplace_output_view[0..memory_size],
                    stride,
                    OFFSET,
                    n,
                );
                match passes_variant {
                    Passes::Three => monomials_to_evals_3_pass(
                        &inplace_input_view_matrix,
                        &mut inplace_output_view_matrix,
                        log_n,
                        transposed_monomials,
                        &stream,
                    )
                    .unwrap(),
                    _ => {}
                };
                memory_copy_async(
                    &mut outputs_host[0..memory_size],
                    inplace_output_view,
                    &stream,
                )
                .unwrap();
            }
        }

        stream.synchronize().unwrap();

        for ntt in 0..num_bf_cols {
            let start = ntt * stride + OFFSET as usize;
            let xs_range = start..start + n;
            let twiddles = &twiddles[..(n >> 1)];
            let gpu_results = &outputs_host[xs_range.clone()];
            let mut cpu_refs: Vec<BF> = (&inputs_orig_host[xs_range.clone()]).to_vec();
            serial_ct_ntt_bitreversed_to_natural(&mut cpu_refs, log_n as u32, twiddles);
            // bitreverse_enumeration_inplace(&mut cpu_refs);
            for k in 0..n {
                assert_eq!(
                    gpu_results[k], cpu_refs[k],
                    "2^{} ntt {} k {}",
                    log_n, ntt, k
                );
            }
        }
    }
    ctx.destroy().unwrap();
}

#[test]
#[serial]
fn test_main_to_monomials_2_pass_out_of_place() {
    run_main_to_monomials(23..25, 8, Passes::Two, InOrOutOfPlace::Out, false);
}

#[test]
#[serial]
fn test_main_to_monomials_2_pass_in_place() {
    run_main_to_monomials(23..25, 8, Passes::Two, InOrOutOfPlace::In, false);
}

#[test]
#[serial]
fn test_main_to_monomials_2_pass_transposed_monomials_out_of_place() {
    run_main_to_monomials(23..25, 8, Passes::Two, InOrOutOfPlace::Out, true);
}

#[test]
#[serial]
fn test_main_to_monomials_2_pass_transposed_monomials_in_place() {
    run_main_to_monomials(23..25, 8, Passes::Two, InOrOutOfPlace::In, true);
}

#[test]
#[serial]
fn test_main_to_monomials_3_pass_out_of_place() {
    run_main_to_monomials(21..25, 8, Passes::Three, InOrOutOfPlace::Out, false);
}

#[test]
#[serial]
fn test_main_to_monomials_3_pass_in_place() {
    run_main_to_monomials(21..25, 8, Passes::Three, InOrOutOfPlace::In, false);
}

#[test]
#[serial]
fn test_main_to_monomials_3_pass_transposed_monomials_out_of_place() {
    run_main_to_monomials(21..25, 8, Passes::Three, InOrOutOfPlace::Out, true);
}

#[test]
#[serial]
fn test_main_to_monomials_3_pass_transposed_monomials_in_place() {
    run_main_to_monomials(21..25, 8, Passes::Three, InOrOutOfPlace::In, true);
}

#[test]
#[serial]
fn test_monomials_to_evals_3_pass_out_of_place() {
    run_monomials_to_evals(21..25, 8, Passes::Three, InOrOutOfPlace::Out, false);
}

#[test]
#[serial]
fn test_monomials_to_evals_3_pass_in_place() {
    run_monomials_to_evals(21..25, 8, Passes::Three, InOrOutOfPlace::In, false);
}

#[test]
#[serial]
fn test_monomials_to_evals_3_pass_transposed_monomials_out_of_place() {
    run_monomials_to_evals(21..25, 8, Passes::Three, InOrOutOfPlace::Out, true);
}

#[test]
#[serial]
fn test_monomials_to_evals_3_pass_transposed_monomials_in_place() {
    run_monomials_to_evals(21..25, 8, Passes::Three, InOrOutOfPlace::In, true);
}
