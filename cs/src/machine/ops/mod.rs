use super::*;

pub mod unrolled;

pub mod common_impls;

pub const RS1_LOAD_LOCAL_TIMESTAMP: usize = 0;
pub const RS2_LOAD_LOCAL_TIMESTAMP: usize = 1;
pub const RD_STORE_LOCAL_TIMESTAMP: usize = 2;

pub use self::common_impls::*;

use devices::diffs::NextPcValue;
