#![allow(non_snake_case)]

#[cfg(test)]
pub mod tests;

use era_cudart::cuda_kernel;
use era_cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use era_cudart::result::{CudaResult, CudaResultWrap};
use era_cudart::stream::CudaStream;
use era_cudart_sys::{CudaFuncAttribute, cudaFuncSetAttribute};

use crate::device_structures::{
    DeviceMatrixChunkImpl, DeviceMatrixChunkMutImpl,
    MutPtrAndStride, PtrAndStride,
};
use crate::field::BaseField;
use crate::utils::GetChunksCount;

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
main_to_monomials_first_10_stages!(ab_main_to_monomials_first_10_stages_tile_8_kernel);
main_to_monomials_first_10_stages!(ab_main_to_monomials_first_10_stages_coalesced_kernel);
main_to_monomials_first_10_stages!(ab_main_to_monomials_first_10_stages_register_pipeline_kernel);
main_to_monomials_first_10_stages!(ab_main_to_monomials_first_10_stages_pipeline_tile_8_kernel);
main_to_monomials_first_10_stages!(ab_main_to_monomials_first_10_stages_pc_kernel);

cuda_kernel!(
    MainToCosetMiddle28Stages,
    main_to_coset_middle_28_stages,
    inputs_matrix: PtrAndStride<BF>,
    outputs_matrix: MutPtrAndStride<BF>,
    num_ntts: i32,
);

main_to_coset_middle_28_stages!(ab_main_to_coset_middle_28_stages_megakernel);

pub fn main_to_coset_pc(
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
    let BF_VALS_PER_BLOCK = 16384;
    let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>() + 16;
    let consumer_threads = 512;
    let producer_threads = 128;
    let threads = consumer_threads + producer_threads;
    let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    config.dynamic_smem_bytes = smem_bytes;
    let args = MainToMonomialsFirst10StagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    let function = MainToMonomialsFirst10StagesFunction(ab_main_to_monomials_first_10_stages_pc_kernel);
    let func_ptr = function.as_ptr();
    unsafe {
        cudaFuncSetAttribute(
            func_ptr,
            CudaFuncAttribute::MaxDynamicSharedMemorySize,
            smem_bytes as i32
        ).wrap()?;
    }
    function.launch(&config, &args)
    // let BF_VALS_PER_BLOCK = 16384;
    // let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>();
    // let threads = 512;
    // let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    // let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    // config.dynamic_smem_bytes = smem_bytes;
    // let args = MainToCosetMiddle28StagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddle28StagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
    // let func_ptr = function.as_ptr();
    // unsafe {
    //     cudaFuncSetAttribute(
    //         func_ptr,
    //         CudaFuncAttribute::MaxDynamicSharedMemorySize,
    //         smem_bytes as i32
    //     ).wrap()?;
    // }
    // function.launch(&config, &args)
}

pub fn main_to_coset_register_pipeline(
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
    let BF_VALS_PER_BLOCK = 16384;
    let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>();
    let threads = 512;
    let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    let mut grid_dim: Dim3 = (blocks as u32).into();
    grid_dim.y = num_ntts as u32;
    let mut config = CudaLaunchConfig::basic(grid_dim, threads as u32, stream);
    config.dynamic_smem_bytes = smem_bytes;
    let args = MainToMonomialsFirst10StagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    let function = MainToMonomialsFirst10StagesFunction(
        ab_main_to_monomials_first_10_stages_register_pipeline_kernel,
    );
    let func_ptr = function.as_ptr();
    unsafe {
        cudaFuncSetAttribute(
            func_ptr,
            CudaFuncAttribute::MaxDynamicSharedMemorySize,
            smem_bytes as i32
        ).wrap()?;
    }
    function.launch(&config, &args)
    // let BF_VALS_PER_BLOCK = 16384;
    // let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>();
    // let threads = 512;
    // let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    // let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    // config.dynamic_smem_bytes = smem_bytes;
    // let args = MainToCosetMiddle28StagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddle28StagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
    // let func_ptr = function.as_ptr();
    // unsafe {
    //     cudaFuncSetAttribute(
    //         func_ptr,
    //         CudaFuncAttribute::MaxDynamicSharedMemorySize,
    //         smem_bytes as i32
    //     ).wrap()?;
    // }
    // function.launch(&config, &args)
}

pub fn main_to_coset_pipeline_tile_8(
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
    // let BF_VALS_PER_BLOCK = 16384;
    let BF_VALS_PER_BLOCK = 8192;
    let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>();
    // let threads = 512;
    let threads = 256;
    let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    config.dynamic_smem_bytes = smem_bytes;
    let args = MainToMonomialsFirst10StagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    // let function = MainToMonomialsFirst10StagesFunction(ab_main_to_monomials_first_10_stages_kernel);
    let function = MainToMonomialsFirst10StagesFunction(ab_main_to_monomials_first_10_stages_pipeline_tile_8_kernel);
    let func_ptr = function.as_ptr();
    unsafe {
        cudaFuncSetAttribute(
            func_ptr,
            CudaFuncAttribute::MaxDynamicSharedMemorySize,
            smem_bytes as i32
        ).wrap()?;
    }
    function.launch(&config, &args)
    // let BF_VALS_PER_BLOCK = 16384;
    // let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>();
    // let threads = 512;
    // let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    // let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    // config.dynamic_smem_bytes = smem_bytes;
    // let args = MainToCosetMiddle28StagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddle28StagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
    // let func_ptr = function.as_ptr();
    // unsafe {
    //     cudaFuncSetAttribute(
    //         func_ptr,
    //         CudaFuncAttribute::MaxDynamicSharedMemorySize,
    //         smem_bytes as i32
    //     ).wrap()?;
    // }
    // function.launch(&config, &args)
}

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
    let BF_VALS_PER_BLOCK = 16384;
    let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>();
    let threads = 512;
    let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    config.dynamic_smem_bytes = smem_bytes;
    let args = MainToMonomialsFirst10StagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    let function = MainToMonomialsFirst10StagesFunction(ab_main_to_monomials_first_10_stages_kernel);
    let func_ptr = function.as_ptr();
    unsafe {
        cudaFuncSetAttribute(
            func_ptr,
            CudaFuncAttribute::MaxDynamicSharedMemorySize,
            smem_bytes as i32
        ).wrap()?;
    }
    function.launch(&config, &args)
    // let BF_VALS_PER_BLOCK = 16384;
    // let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>();
    // let threads = 512;
    // let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    // let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    // config.dynamic_smem_bytes = smem_bytes;
    // let args = MainToCosetMiddle28StagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddle28StagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
    // let func_ptr = function.as_ptr();
    // unsafe {
    //     cudaFuncSetAttribute(
    //         func_ptr,
    //         CudaFuncAttribute::MaxDynamicSharedMemorySize,
    //         smem_bytes as i32
    //     ).wrap()?;
    // }
    // function.launch(&config, &args)
}

pub fn main_to_coset_tile_8(
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
    // let BF_VALS_PER_BLOCK = 16384;
    let BF_VALS_PER_BLOCK = 8192;
    let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>();
    // let threads = 512;
    let threads = 256;
    let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    config.dynamic_smem_bytes = smem_bytes;
    let args = MainToMonomialsFirst10StagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    // let function = MainToMonomialsFirst10StagesFunction(ab_main_to_monomials_first_10_stages_kernel);
    let function = MainToMonomialsFirst10StagesFunction(ab_main_to_monomials_first_10_stages_tile_8_kernel);
    let func_ptr = function.as_ptr();
    unsafe {
        cudaFuncSetAttribute(
            func_ptr,
            CudaFuncAttribute::MaxDynamicSharedMemorySize,
            smem_bytes as i32
        ).wrap()?;
    }
    function.launch(&config, &args)
    // let BF_VALS_PER_BLOCK = 16384;
    // let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>();
    // let threads = 512;
    // let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    // let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    // config.dynamic_smem_bytes = smem_bytes;
    // let args = MainToCosetMiddle28StagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddle28StagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
    // let func_ptr = function.as_ptr();
    // unsafe {
    //     cudaFuncSetAttribute(
    //         func_ptr,
    //         CudaFuncAttribute::MaxDynamicSharedMemorySize,
    //         smem_bytes as i32
    //     ).wrap()?;
    // }
    // function.launch(&config, &args)
}

pub fn main_to_coset_coalesced(
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
    let BF_VALS_PER_BLOCK = 32768;
    let smem_bytes = (BF_VALS_PER_BLOCK / 2) * size_of::<BF>();
    let threads = 512;
    let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    config.dynamic_smem_bytes = smem_bytes;
    let args = MainToMonomialsFirst10StagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    let function = MainToMonomialsFirst10StagesFunction(ab_main_to_monomials_first_10_stages_coalesced_kernel);
    let func_ptr = function.as_ptr();
    unsafe {
        cudaFuncSetAttribute(
            func_ptr,
            CudaFuncAttribute::MaxDynamicSharedMemorySize,
            smem_bytes as i32
        ).wrap()?;
    }
    function.launch(&config, &args)
    // let BF_VALS_PER_BLOCK = 16384;
    // let smem_bytes = BF_VALS_PER_BLOCK * size_of::<BF>();
    // let threads = 512;
    // let blocks = n.get_chunks_count(BF_VALS_PER_BLOCK);
    // let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    // config.dynamic_smem_bytes = smem_bytes;
    // let args = MainToCosetMiddle28StagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddle28StagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
    // let func_ptr = function.as_ptr();
    // unsafe {
    //     cudaFuncSetAttribute(
    //         func_ptr,
    //         CudaFuncAttribute::MaxDynamicSharedMemorySize,
    //         smem_bytes as i32
    //     ).wrap()?;
    // }
    // function.launch(&config, &args)
}
