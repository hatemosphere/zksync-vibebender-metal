use super::*;

use crate::gkr::witness_gen::column_major_proxy::ColumnMajorWitnessProxy;
use cs::cs::oracle::Oracle;

// mod init_and_teardown;
mod memory;
// mod unified;
// mod witness;

// pub use self::init_and_teardown::{
//     evaluate_init_and_teardown_memory_witness, evaluate_init_and_teardown_witness,
// };
pub use self::memory::evaluate_gkr_memory_witness_for_executor_family;
// pub use self::unified::{
//     evaluate_memory_witness_for_unified_executor, evaluate_witness_for_unified_executor,
// };
// pub use self::witness::evaluate_witness_for_executor_family;
