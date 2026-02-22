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
    MainToMonomialsFirstStages,
    main_to_monomials_first_stages,
    inputs_matrix: PtrAndStride<BF>,
    outputs_matrix: MutPtrAndStride<BF>,
    log_n: i32,
    num_ntts_or_start_stage: i32,
);

// pretty good for 2-pass
main_to_monomials_first_stages!(ab_main_to_monomials_first_9_stages_register_pipeline_kernel);
main_to_monomials_first_stages!(ab_main_to_monomials_first_10_stages_register_pipeline_kernel);

// 3-pass
main_to_monomials_first_stages!(ab_main_to_monomials_nonfinal_8_stages_kernel);

// experiment graveyard
main_to_monomials_first_stages!(ab_main_to_monomials_first_10_stages_kernel);
main_to_monomials_first_stages!(ab_main_to_monomials_first_10_stages_tile_8_kernel);
main_to_monomials_first_stages!(ab_main_to_monomials_first_10_stages_coalesced_kernel);
main_to_monomials_first_stages!(ab_main_to_monomials_first_10_stages_pipeline_tile_8_kernel);
main_to_monomials_first_stages!(ab_main_to_monomials_first_10_stages_pc_kernel);

cuda_kernel!(
    MainToCosetMiddleStages,
    main_to_coset_middle_stages,
    inputs_matrix: PtrAndStride<BF>,
    outputs_matrix: MutPtrAndStride<BF>,
    num_ntts: i32,
);

// pretty good for 2-pass
main_to_coset_middle_stages!(ab_main_to_monomials_last_14_stages_kernel);

// 3-pass
main_to_coset_middle_stages!(ab_main_to_monomials_final_8_stages_kernel);

// experiment graveyard
main_to_coset_middle_stages!(ab_main_to_coset_middle_28_stages_megakernel);

pub fn main_to_monomials_3_pass(
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
    let bf_vals_per_block = 1 << 13; // 8192
    let threads = 512;
    let mut start_stage = 0;
    let num_exchg_regions = 1 << start_stage;
    let exchg_region_size = n >> start_stage;
    let blocks_per_exchg_region = exchg_region_size / bf_vals_per_block;
    assert_eq!(blocks_per_exchg_region * num_exchg_regions, n / bf_vals_per_block);
    let mut grid_dim: Dim3 = (blocks_per_exchg_region as u32).into();
    grid_dim.y = num_exchg_regions as u32;
    let mut config = CudaLaunchConfig::basic(grid_dim, threads as u32, stream);
    let args = MainToMonomialsFirstStagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        start_stage as i32,
    );
    MainToMonomialsFirstStagesFunction(ab_main_to_monomials_nonfinal_8_stages_kernel)
        .launch(&config, &args)?;
    start_stage += 8;
    let num_exchg_regions = 1 << start_stage;
    let exchg_region_size = n >> start_stage;
    let blocks_per_exchg_region = exchg_region_size / bf_vals_per_block;
    assert_eq!(blocks_per_exchg_region * num_exchg_regions, n / bf_vals_per_block);
    let mut grid_dim: Dim3 = (blocks_per_exchg_region as u32).into();
    grid_dim.y = num_exchg_regions as u32;
    let mut config = CudaLaunchConfig::basic(grid_dim, threads as u32, stream);
    let args = MainToMonomialsFirstStagesArguments::new(
        outputs_matrix_const,
        outputs_matrix_mut,
        log_n as i32,
        start_stage as i32,
    );
    MainToMonomialsFirstStagesFunction(ab_main_to_monomials_nonfinal_8_stages_kernel)
        .launch(&config, &args)?;
    start_stage += 8;
    let threads = 256;
    let bf_vals_per_block = 1 << 13; // 8192
    let blocks = n.get_chunks_count(bf_vals_per_block);
    let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    let args = MainToCosetMiddleStagesArguments::new(
        outputs_matrix_const,
        outputs_matrix_mut,
        log_n as i32,
    );
    MainToCosetMiddleStagesFunction(ab_main_to_monomials_final_8_stages_kernel)
        .launch(&config, &args)
}

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
    let args = MainToMonomialsFirstStagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    let function = MainToMonomialsFirstStagesFunction(ab_main_to_monomials_first_10_stages_pc_kernel);
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
    // let args = MainToCosetMiddleStagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddleStagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
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
    let bf_vals_per_block = 1 << 14; // 16384
    let smem_bytes = bf_vals_per_block * size_of::<BF>();
    let threads = 512;
    let blocks = n.get_chunks_count(bf_vals_per_block);
    let mut grid_dim: Dim3 = (blocks as u32).into();
    grid_dim.y = num_ntts as u32;
    let mut config = CudaLaunchConfig::basic(grid_dim, threads as u32, stream);
    config.dynamic_smem_bytes = smem_bytes;
    let args = MainToMonomialsFirstStagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    let function = match log_n {
        23 => MainToMonomialsFirstStagesFunction(
            ab_main_to_monomials_first_9_stages_register_pipeline_kernel,
        ),
        24 => MainToMonomialsFirstStagesFunction(
            ab_main_to_monomials_first_10_stages_register_pipeline_kernel,
        ),
        _ => unimplemented!(),
    };
    let func_ptr = function.as_ptr();
    unsafe {
        cudaFuncSetAttribute(
            func_ptr,
            CudaFuncAttribute::MaxDynamicSharedMemorySize,
            smem_bytes as i32
        ).wrap()?;
    }
    function.launch(&config, &args)?;
    let bf_vals_per_block = 1 << 14; // 16384 
    let smem_twiddles_per_block = 1 << 13; // 8192
    let smem_bytes = (bf_vals_per_block + smem_twiddles_per_block) * size_of::<BF>();
    let threads = 512;
    let blocks = n.get_chunks_count(bf_vals_per_block);
    let mut config = CudaLaunchConfig::basic(blocks as u32, threads as u32, stream);
    config.dynamic_smem_bytes = smem_bytes;
    let args = MainToCosetMiddleStagesArguments::new(
        outputs_matrix_const,
        outputs_matrix_mut,
        num_ntts as i32,
    );
    let function = MainToCosetMiddleStagesFunction(ab_main_to_monomials_last_14_stages_kernel);
    let func_ptr = function.as_ptr();
    unsafe {
        cudaFuncSetAttribute(
            func_ptr,
            CudaFuncAttribute::MaxDynamicSharedMemorySize,
            smem_bytes as i32
        ).wrap()?;
    }
    function.launch(&config, &args)
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
    let args = MainToMonomialsFirstStagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    // let function = MainToMonomialsFirstStagesFunction(ab_main_to_monomials_first_10_stages_kernel);
    let function = MainToMonomialsFirstStagesFunction(ab_main_to_monomials_first_10_stages_pipeline_tile_8_kernel);
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
    // let args = MainToCosetMiddleStagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddleStagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
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
    let args = MainToMonomialsFirstStagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    let function = MainToMonomialsFirstStagesFunction(ab_main_to_monomials_first_10_stages_kernel);
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
    // let args = MainToCosetMiddleStagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddleStagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
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
    let args = MainToMonomialsFirstStagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    // let function = MainToMonomialsFirstStagesFunction(ab_main_to_monomials_first_10_stages_kernel);
    let function = MainToMonomialsFirstStagesFunction(ab_main_to_monomials_first_10_stages_tile_8_kernel);
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
    // let args = MainToCosetMiddleStagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddleStagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
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
    let args = MainToMonomialsFirstStagesArguments::new(
        inputs_matrix,
        outputs_matrix_mut,
        log_n as i32,
        num_ntts as i32,
    );
    let function = MainToMonomialsFirstStagesFunction(ab_main_to_monomials_first_10_stages_coalesced_kernel);
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
    // let args = MainToCosetMiddleStagesArguments::new(
    //     outputs_matrix_const,
    //     outputs_matrix_mut,
    //     num_ntts as i32,
    // );
    // let function = MainToCosetMiddleStagesFunction(ab_main_to_coset_middle_28_stages_megakernel);
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
