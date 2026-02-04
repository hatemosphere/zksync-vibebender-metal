#![allow(non_snake_case)]

#[cfg(test)]
pub mod tests;

use era_cudart::cuda_kernel;
use era_cudart::error::get_last_error;
use era_cudart::event::{CudaEvent, CudaEventCreateFlags};
use era_cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use era_cudart::result::{CudaResult, CudaResultWrap};
use era_cudart::slice::DeviceSlice;
use era_cudart::stream::{CudaStream, CudaStreamWaitEventFlags};

use crate::device_structures::{
    DeviceMatrixChunk, DeviceMatrixChunkImpl, DeviceMatrixChunkMut, DeviceMatrixChunkMutImpl,
    MutPtrAndStride, PtrAndStride,
};
use crate::field::{BaseField, Ext2Field};
use crate::utils::GetChunksCount;

use itertools::Itertools;
use std::mem::size_of;

type BF = BaseField;

cuda_kernel!(
    MainToMonomialsFirst10Stages,
    main_to_monomials_first_10_stages,
    inputs_matrix: PtrAndStride<BF>,
    outputs_matrix: MutPtrAndStride<BF>,
    log_n: i32,
    num_ntts: i32,
);

main_to_monomials_first_10_stages!(ab_main_to_monomials_first_10_stages_kernel);

cuda_kernel!(
    MainToCosetMiddle28Stages,
    main_to_coset_middle_28_stages,
    inputs_matrix: PtrAndStride<BF>,
    outputs_matrix: MutPtrAndStride<BF>,
    num_ntts: i32,
);

main_to_coset_middle_28_stages!(ab_main_to_coset_middle_28_stages_megakernel);

pub fn main_to_coset(
    inputs_matrix: &(impl DeviceMatrixChunkImpl<BF> + ?Sized),
    outputs_matrix: &mut (impl DeviceMatrixChunkMutImpl<BF> + ?Sized),
    log_n: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    let n = 1 << log_n;
    assert_eq!(inputs_matrix.rows(), n);
    assert_eq!(outputs_matrix.rows(), n);
    // __pipeline_memcpy_asyncs in the kernel require 16 byte alignment
    assert_eq!(inputs_matrix.slice().as_ptr() as usize % 16, 0);
    assert_eq!(outputs_matrix.slice().as_ptr() as usize % 16, 0);
    assert_eq!((inputs_matrix.stride() * size_of::<BF>()) % 16, 0);
    assert_eq!((outputs_matrix.offset() * size_of::<BF>()) % 16, 0);
    assert_eq!((inputs_matrix.stride() * size_of::<BF>()) % 16, 0);
    assert_eq!((outputs_matrix.offset() * size_of::<BF>()) % 16, 0);
    let num_ntts = outputs_matrix.cols();
    let inputs_matrix = inputs_matrix.as_ptr_and_stride();
    let outputs_matrix_const = outputs_matrix.as_ptr_and_stride();
    let outputs_matrix_mut = outputs_matrix.as_mut_ptr_and_stride();
    let threads = 512;
    let blocks = n.get_chunks_count(16384);
    let config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    let args = MainToMonomialsFirst10StagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    MainToMonomialsFirst10StagesFunction(ab_main_to_monomials_first_10_stages_kernel)
        .launch(&config, &args)?;
    let threads = 512;
    let blocks = n.get_chunks_count(16384);
    let config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    let args = MainToCosetMiddle28StagesArguments::new(
        outputs_matrix_const,
        outputs_matrix_mut,
        num_ntts as i32,
    );
    MainToCosetMiddle28StagesFunction(ab_main_to_coset_middle_28_stages_megakernel)
        .launch(&config, &args)
}
