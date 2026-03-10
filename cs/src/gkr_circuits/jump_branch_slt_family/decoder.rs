use super::*;
use crate::types::Boolean;

const JAL_BIT: usize = 0;
const JALR_BIT: usize = 1;
const SLT_BIT: usize = 2;
const BRANCH_BIT: usize = 3;
const RD_IS_ZERO_BIT: usize = 4;

const SLT_FUNCT3: u8 = 0b010;
const SLTU_FUNCT3: u8 = 0b011;

#[derive(Clone, Copy, Debug)]
pub struct JumpSltBranchDecoder;

#[derive(Clone, Copy, Debug)]
pub struct JumpSltBranchFamilyCircuitMask {
    inner: [Boolean; JUMP_SLT_BRANCH_FAMILY_NUM_BITS],
}

impl JumpSltBranchFamilyCircuitMask {
    // getters for our opcodes
    pub fn perform_jal(&self) -> Boolean {
        self.inner[JAL_BIT]
    }

    pub fn perform_jalr(&self) -> Boolean {
        self.inner[JALR_BIT]
    }

    pub fn perform_slt(&self) -> Boolean {
        self.inner[SLT_BIT]
    }

    pub fn perform_branch(&self) -> Boolean {
        self.inner[BRANCH_BIT]
    }

    pub fn rd_is_zero(&self) -> Boolean {
        self.inner[RD_IS_ZERO_BIT]
    }
}

impl OpcodeFamilyDecoder for JumpSltBranchDecoder {
    type BitmaskCircuitParser = JumpSltBranchFamilyCircuitMask;

    fn instruction_family_index(&self) -> u8 {
        common_constants::circuit_families::JUMP_BRANCH_SLT_CIRCUIT_FAMILY_IDX
    }

    fn define_decoder_subspace(
        &self,
        preprocessed_opcode: Instruction,
    ) -> Result<ExecutorFamilyDecoderData, ()> {
        let (mut rs1_index, mut rs2_index, mut rd_index) = (0, 0u16, 0);
        let mut imm = 0;
        let mut bitmask = 0u32;
        let mut funct3 = None;

        match preprocessed_opcode.name {
            InstructionName::Jal => {
                assert_eq!(preprocessed_opcode.rs1, 0);
                assert_eq!(preprocessed_opcode.rs2, 0);

                rd_index = preprocessed_opcode.rd;
                imm = preprocessed_opcode.imm;
                bitmask |= 1 << JAL_BIT;
                if preprocessed_opcode.rd == 0 {
                    bitmask |= 1 << RD_IS_ZERO_BIT;
                }
            }
            InstructionName::Jalr => {
                assert_ne!(preprocessed_opcode.rs1, 0);
                assert_eq!(preprocessed_opcode.rs2, 0);

                rs1_index = preprocessed_opcode.rs1;
                rd_index = preprocessed_opcode.rd;
                imm = preprocessed_opcode.imm;
                bitmask |= 1 << JALR_BIT;
                if preprocessed_opcode.rd == 0 {
                    bitmask |= 1 << RD_IS_ZERO_BIT;
                }
            }
            InstructionName::Branch => {
                assert_ne!(preprocessed_opcode.rs1, 0);
                assert_ne!(preprocessed_opcode.rs2, 0);

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                imm = preprocessed_opcode.imm;
                funct3 = Some(preprocessed_opcode.rd); // Stored as rd index
                bitmask |= 1 << BRANCH_BIT;
                bitmask |= 1 << RD_IS_ZERO_BIT;
            }
            InstructionName::Slt => {
                assert_ne!(preprocessed_opcode.rd, 0);
                if preprocessed_opcode.imm != 0 {
                    assert_eq!(preprocessed_opcode.rs2, 0);
                }

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;
                imm = preprocessed_opcode.imm;
                funct3 = Some(SLT_FUNCT3);
                bitmask |= 1 << SLT_BIT;
            }
            InstructionName::Sltu => {
                assert_ne!(preprocessed_opcode.rd, 0);
                if preprocessed_opcode.imm != 0 {
                    assert_eq!(preprocessed_opcode.rs2, 0);
                }

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;
                imm = preprocessed_opcode.imm;
                funct3 = Some(SLTU_FUNCT3);
                bitmask |= 1 << SLT_BIT;
            }
            _ => {
                return Err(());
            }
        }

        let decoded = ExecutorFamilyDecoderData {
            imm,
            rs1_index,
            rs2_index,
            rd_index,
            funct3,
            funct7: None,
            opcode_family_bits: bitmask,
        };

        Ok(decoded)
    }
}
