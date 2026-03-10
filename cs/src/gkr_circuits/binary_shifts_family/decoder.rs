use super::*;
use crate::types::Boolean;

const SLL_BIT: usize = 0;
const SRL_BIT: usize = 1;
const SRA_BIT: usize = 2;
const BINARY_OP_BIT: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct ShiftBinaryDecoder;

#[derive(Clone, Copy, Debug)]
pub struct ShiftBinaryFamilyCircuitMask {
    inner: [Boolean; SHIFT_BINARY_CSRRW_FAMILY_NUM_FLAGS],
}

impl ShiftBinaryFamilyCircuitMask {
    // getters for our opcodes
    pub fn perform_sll(&self) -> Boolean {
        self.inner[SLL_BIT]
    }

    pub fn perform_srl(&self) -> Boolean {
        self.inner[SRL_BIT]
    }

    pub fn perform_sra(&self) -> Boolean {
        self.inner[SRA_BIT]
    }

    pub fn perform_binary_op(&self) -> Boolean {
        self.inner[BINARY_OP_BIT]
    }
}

impl OpcodeFamilyDecoder for ShiftBinaryDecoder {
    type BitmaskCircuitParser = ShiftBinaryFamilyCircuitMask;

    fn instruction_family_index(&self) -> u8 {
        common_constants::circuit_families::SHIFT_BINARY_CSR_CIRCUIT_FAMILY_IDX
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
            InstructionName::And => {
                assert_ne!(preprocessed_opcode.rd, 0);
                if preprocessed_opcode.imm != 0 {
                    assert_eq!(preprocessed_opcode.rs2, 0);
                }

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;
                imm = preprocessed_opcode.imm;
                funct3 = Some(AND_TABLE_ID as u8);
                bitmask |= 1 << BINARY_OP_BIT;
            }
            InstructionName::Or => {
                assert_ne!(preprocessed_opcode.rd, 0);
                if preprocessed_opcode.imm != 0 {
                    assert_eq!(preprocessed_opcode.rs2, 0);
                }

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;
                imm = preprocessed_opcode.imm;
                funct3 = Some(OR_TABLE_ID as u8);
                bitmask |= 1 << BINARY_OP_BIT;
            }
            InstructionName::Or => {
                assert_ne!(preprocessed_opcode.rd, 0);
                if preprocessed_opcode.imm != 0 {
                    assert_eq!(preprocessed_opcode.rs2, 0);
                }

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;
                imm = preprocessed_opcode.imm;
                funct3 = Some(XOR_TABLE_ID as u8);
                bitmask |= 1 << BINARY_OP_BIT;
            }
            InstructionName::Sll => {
                assert_ne!(preprocessed_opcode.rd, 0);
                if preprocessed_opcode.imm != 0 {
                    assert_eq!(preprocessed_opcode.rs2, 0);
                }

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;
                imm = preprocessed_opcode.imm;
                funct3 = Some(0);
                bitmask |= 1 << SLL_BIT;
            }
            InstructionName::Srl => {
                assert_ne!(preprocessed_opcode.rd, 0);
                if preprocessed_opcode.imm != 0 {
                    assert_eq!(preprocessed_opcode.rs2, 0);
                }

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;
                imm = preprocessed_opcode.imm;
                funct3 = Some(0 as u8);
                bitmask |= 1 << SRL_BIT;
            }
            InstructionName::Sra => {
                assert_ne!(preprocessed_opcode.rd, 0);
                if preprocessed_opcode.imm != 0 {
                    assert_eq!(preprocessed_opcode.rs2, 0);
                }

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;
                imm = preprocessed_opcode.imm;
                funct3 = Some(0 as u8);
                bitmask |= 1 << SRA_BIT;
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
