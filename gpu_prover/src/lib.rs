#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(btree_cursors)]
#![feature(generic_const_exprs)]
#![feature(pointer_is_aligned_to)]

pub(crate) mod allocator;
pub(crate) mod circuit_type;
pub(crate) mod device_context;
pub(crate) mod device_structures;
pub mod field;
pub(crate) mod machine_type;
pub(crate) mod ops;
pub(crate) mod prover;
pub(crate) mod utils;
pub(crate) mod witness;

pub use crate::device_structures::{DeviceMatrix, DeviceMatrixMut};
pub use crate::ops::hypercube::{
    hypercube_evals_into_coeffs_bitrev_bf, hypercube_evals_into_coeffs_bitrev_bf_in_place,
};

#[cfg(test)]
mod tests;
