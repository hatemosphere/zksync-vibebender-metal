#![feature(allocator_api)]
#![feature(pointer_is_aligned_to)]
#![feature(btree_cursors)]

pub(crate) mod allocator;
pub(crate) mod device_context;
pub(crate) mod device_structures;
pub mod field;
pub(crate) mod ntt;
pub(crate) mod ops;
pub(crate) mod prover;
pub(crate) mod utils;
