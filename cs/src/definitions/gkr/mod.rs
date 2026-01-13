mod lookup;
mod ram_access;

pub use self::lookup::*;
pub use self::ram_access::*;

use crate::definitions::GKRAddress;
use crate::definitions::REGISTER_SIZE;
use common_constants::NUM_TIMESTAMP_COLUMNS_FOR_RAM;

#[derive(Clone, Copy, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub struct GKRMachineState {
    pub pc: [usize; REGISTER_SIZE],
    pub timestamp: [usize; NUM_TIMESTAMP_COLUMNS_FOR_RAM],
}

#[derive(Clone, Copy, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub struct MachineStatePermutationDescription {
    pub execute: usize,
    pub initial_state: GKRMachineState,
    pub final_state: GKRMachineState,
}

#[derive(Clone, Copy, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub struct DecoderPlacementDescription {
    // rs1 is always in memory
    pub rs1_index: usize,
    // can be memory or witness, as there can be some selection there
    pub rs2_index: GKRAddress,
    pub rd_index: GKRAddress,
    // can rarely happen to be in memory columns, and can also be empty if it's just linear expression
    pub circuit_family_extra_mask: Option<GKRAddress>,
    // the rest are either all in memory, or all in witness
    pub decoder_witness_is_in_memory: bool,
    pub rd_is_zero: usize,
    pub imm: [usize; REGISTER_SIZE],
    pub funct3: Option<usize>,
}

#[derive(Clone, Debug, Hash, serde::Serialize, serde::Deserialize)]
pub struct GKRMemoryLayout {
    pub shuffle_ram_access_sets: Vec<RamQuery>,
    pub machine_state: Option<MachineStatePermutationDescription>,
    pub decoder_input: Option<DecoderPlacementDescription>,
    pub register_and_indirect_accesses: Vec<()>,
    pub total_width: usize,
}
