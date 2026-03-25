#![allow(incomplete_features)]
#![allow(dead_code)]
#![feature(allocator_api)]
#![feature(generic_const_exprs)]
#![feature(btree_cursors)]
#![feature(pointer_is_aligned_to)]
#![feature(new_zeroed_alloc)]
#![feature(vec_push_within_capacity)]
#![feature(iter_array_chunks)]
#![feature(iter_advance_by)]
#![feature(sync_unsafe_cell)]
#![feature(once_cell_try)]

pub mod metal_runtime;
pub mod allocator;
pub mod barycentric;
pub mod blake2s;
pub mod circuit_type;
pub mod device_context;
pub mod device_structures;
pub mod execution;
pub mod field;
pub mod monolith;
pub mod ntt;
pub mod ops_complex;
pub mod ops_cub;
pub mod ops_simple;
pub mod prover;
pub mod utils;
pub mod witness;

/// CUDA compatibility shim — provides `cudart::result::CudaResult` as `MetalResult`.
/// This allows code that imports `gpu_prover::cudart::result::CudaResult` to compile
/// unchanged when using the Metal backend.
pub mod cudart {
    pub mod result {
        pub type CudaResult<T> = crate::metal_runtime::error::MetalResult<T>;
    }
}

#[cfg(test)]
mod tests;
