use super::*;
use crate::types::Boolean;

const IS_DIVISION_BIT: usize = 0;
const RD_IS_ZERO_BIT: usize = 1;
const MUL_DIV_DIVU: usize = 2; // SIGNED
const UNSIGNED_MUL_DIVU: usize = 2; // UNSIGNED
const MUL_MULH_MULHSU_DIV_REM: usize = 3;
const MUL_MULH_DIV_REM: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct DivMulDecoder<const SUPPORT_SIGNED: bool>;

#[derive(Clone, Copy, Debug)]
pub struct DivMulFamilyCircuitMask<const SUPPORT_SIGNED: bool> {
    inner: [Boolean; MUL_DIV_FAMILY_NUM_FLAGS],
}

impl<const SUPPORT_SIGNED: bool> DivMulFamilyCircuitMask<SUPPORT_SIGNED> {
    // getters for our opcodes
    pub fn perform_division_group(&self) -> Boolean {
        self.inner[IS_DIVISION_BIT]
    }

    pub fn perform_rs1_signed(&self) -> Boolean {
        assert!(SUPPORT_SIGNED);
        self.inner[MUL_MULH_MULHSU_DIV_REM]
    }

    pub fn perform_rs2_signed(&self) -> Boolean {
        assert!(SUPPORT_SIGNED);
        self.inner[MUL_MULH_DIV_REM]
    }

    pub fn perform_mul_div_divu(&self) -> Boolean {
        assert!(SUPPORT_SIGNED);
        self.inner[MUL_DIV_DIVU]
    }

    pub fn perform_mul_divu(&self) -> Boolean {
        assert!(!SUPPORT_SIGNED);
        self.inner[UNSIGNED_MUL_DIVU]
    }

    pub fn rd_is_zero(&self) -> Boolean {
        self.inner[RD_IS_ZERO_BIT]
    }
}

impl<const SUPPORT_SIGNED: bool> OpcodeFamilyDecoder for DivMulDecoder<SUPPORT_SIGNED> {
    type BitmaskCircuitParser = DivMulFamilyCircuitMask<SUPPORT_SIGNED>;

    fn instruction_family_index(&self) -> u8 {
        common_constants::circuit_families::MUL_DIV_CIRCUIT_FAMILY_IDX
    }

    // const IS_DIVISION_BIT: usize = 0;
    // const MUL_MULH_MULHSU_DIV_REM: usize = 1;
    // const MUL_MULH_DIV_REM: usize = 2;
    // const MUL_DIV_DIVU: usize = 3;

    fn define_decoder_subspace(
        &self,
        preprocessed_opcode: Instruction,
    ) -> Result<ExecutorFamilyDecoderData, ()> {
        let (mut rs1_index, mut rs2_index, mut rd_index) = (0, 0u16, 0);
        let mut imm = 0;
        let mut bitmask = 0u32;

        match preprocessed_opcode.name {
            InstructionName::Mul => {
                assert_ne!(preprocessed_opcode.rd, 0);
                assert_eq!(preprocessed_opcode.imm, 0);

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;
                if SUPPORT_SIGNED {
                    bitmask |= 1 << MUL_MULH_MULHSU_DIV_REM;
                    bitmask |= 1 << MUL_MULH_DIV_REM;
                    bitmask |= 1 << MUL_DIV_DIVU;
                } else {
                    bitmask |= 1 << UNSIGNED_MUL_DIVU;
                }
            }
            InstructionName::Mulh if SUPPORT_SIGNED => {
                assert_ne!(preprocessed_opcode.rd, 0);
                assert_eq!(preprocessed_opcode.imm, 0);

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;

                bitmask |= 1 << MUL_MULH_MULHSU_DIV_REM;
                bitmask |= 1 << MUL_MULH_DIV_REM;
            }
            InstructionName::Mulhsu if SUPPORT_SIGNED => {
                assert_ne!(preprocessed_opcode.rd, 0);
                assert_eq!(preprocessed_opcode.imm, 0);

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;

                bitmask |= 1 << MUL_MULH_MULHSU_DIV_REM;
            }
            InstructionName::Mulhu => {
                assert_ne!(preprocessed_opcode.rd, 0);
                assert_eq!(preprocessed_opcode.imm, 0);

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;
                // We avoid setting any bits, as this is always the default/negative case for all bits
            }
            InstructionName::Div if SUPPORT_SIGNED => {
                assert_eq!(preprocessed_opcode.imm, 0);

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;

                bitmask |= 1 << IS_DIVISION_BIT;
                bitmask |= 1 << MUL_MULH_MULHSU_DIV_REM;
                bitmask |= 1 << MUL_MULH_DIV_REM;
                bitmask |= 1 << MUL_DIV_DIVU;

                if preprocessed_opcode.rd == 0 {
                    bitmask |= 1 << RD_IS_ZERO_BIT;
                }
            }
            InstructionName::Divu => {
                assert_eq!(preprocessed_opcode.imm, 0);

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;

                if SUPPORT_SIGNED {
                    bitmask |= 1 << IS_DIVISION_BIT;
                    bitmask |= 1 << MUL_DIV_DIVU;
                } else {
                    bitmask |= 1 << IS_DIVISION_BIT;
                    bitmask |= 1 << UNSIGNED_MUL_DIVU; // it's the same anyway
                }

                if preprocessed_opcode.rd == 0 {
                    bitmask |= 1 << RD_IS_ZERO_BIT;
                }
            }
            InstructionName::Rem if SUPPORT_SIGNED => {
                assert_eq!(preprocessed_opcode.imm, 0);

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;

                bitmask |= 1 << IS_DIVISION_BIT;
                bitmask |= 1 << MUL_MULH_MULHSU_DIV_REM;
                bitmask |= 1 << MUL_MULH_DIV_REM;

                if preprocessed_opcode.rd == 0 {
                    bitmask |= 1 << RD_IS_ZERO_BIT;
                }
            }
            InstructionName::Remu => {
                assert_eq!(preprocessed_opcode.imm, 0);

                rs1_index = preprocessed_opcode.rs1;
                rs2_index = preprocessed_opcode.rs2 as u16;
                rd_index = preprocessed_opcode.rd;

                // same as for MULHU, it's the default case
                // except that it belongs to division group
                bitmask |= 1 << IS_DIVISION_BIT;

                if preprocessed_opcode.rd == 0 {
                    bitmask |= 1 << RD_IS_ZERO_BIT;
                }
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
            funct3: None,
            funct7: None,
            opcode_family_bits: bitmask,
        };

        Ok(decoded)
    }
}
