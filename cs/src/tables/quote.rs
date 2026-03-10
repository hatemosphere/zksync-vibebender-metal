use crate::tables::TableType;

impl quote::ToTokens for TableType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        use quote::quote;
        let stream = match self {
            TableType::And => quote! { TableType::And },
            TableType::Xor => quote! { TableType::Xor },
            TableType::Or => quote! { TableType::Or },
            TableType::RangeCheck8x8 => quote! { TableType::RangeCheck8x8 },
            // TableType::RangeCheckLarge => quote! { TableType::RangeCheckLarge },
            TableType::PowersOf2 => quote! { TableType::PowersOf2 },
            TableType::OpTypeBitmask => quote! { TableType::OpTypeBitmask },
            TableType::InsnEncodingChecker => quote! { TableType::InsnEncodingChecker },
            TableType::CsrBitmask => quote! { TableType::CsrBitmask },
            TableType::ZeroEntry => quote! { TableType::ZeroEntry },
            TableType::AndNot => quote! { TableType::AndNot },
            TableType::QuickDecodeDecompositionCheck4x4x4 => {
                quote! { TableType::QuickDecodeDecompositionCheck4x4x4 }
            }
            TableType::QuickDecodeDecompositionCheck7x3x6 => {
                quote! { TableType::QuickDecodeDecompositionCheck7x3x6 }
            }
            TableType::MRetProcessLow => quote! { TableType::MRetProcessLow },
            TableType::MRetClearHigh => quote! { TableType::MRetClearHigh },
            TableType::TrapProcessLow => quote! { TableType::TrapProcessLow },
            TableType::U16GetSignAndHighByte => quote! { TableType::U16GetSignAndHighByte },
            TableType::JumpCleanupOffset => quote! { TableType::JumpCleanupOffset },
            TableType::MemoryOffsetGetBits => quote! { TableType::MemoryOffsetGetBits },
            TableType::MemoryLoadGetSigns => quote! { TableType::MemoryLoadGetSigns },
            TableType::SRASignFiller => quote! { TableType::SRASignFiller },
            TableType::ConditionalOpAllConditionsResolver => {
                quote! { TableType::ConditionalOpAllConditionsResolver }
            }
            TableType::RomAddressSpaceSeparator => quote! { TableType::RomAddressSpaceSeparator },
            TableType::RomRead => quote! { TableType::RomRead },
            TableType::SpecialCSRProperties => quote! { TableType::SpecialCSRProperties },
            TableType::Xor3 => quote! { TableType::Xor3 },
            TableType::Xor4 => quote! { TableType::Xor4 },
            TableType::Xor7 => quote! { TableType::Xor7 },
            TableType::Xor9 => quote! { TableType::Xor9 },
            TableType::Xor12 => quote! { TableType::Xor12 },
            TableType::U16SplitAsBytes => quote! { TableType::U16SplitAsBytes },
            TableType::RangeCheck9x9 => quote! { TableType::RangeCheck9x9 },
            TableType::RangeCheck10x10 => quote! { TableType::RangeCheck10x10 },
            TableType::RangeCheck11 => quote! { TableType::RangeCheck11 },
            TableType::RangeCheck12 => quote! { TableType::RangeCheck12 },
            TableType::RangeCheck13 => quote! { TableType::RangeCheck13 },
            TableType::ShiftImplementation => quote! { TableType::ShiftImplementation },
            TableType::U16SelectByteAndGetByteSign => {
                quote! { TableType::U16SelectByteAndGetByteSign }
            }
            TableType::ConditionalOpUnsignedConditionsResolver => {
                todo!()
            }
            TableType::StoreByteSourceContribution => {
                quote! { TableType::StoreByteSourceContribution }
            }
            TableType::StoreByteExistingContribution => {
                quote! { TableType::StoreByteExistingContribution }
            }
            TableType::ExtendLoadedValue => quote! { TableType::ExtendLoadedValue },
            TableType::TruncateShift => quote! { TableType::TruncateShift },
            TableType::AlignedRomRead => quote! { TableType::AlignedRomRead },
            TableType::ConditionalJmpBranchSlt => {
                quote! { TableType::ConditionalJmpBranchSlt }
            }
            TableType::SllWith16BitInputLow => {
                quote! { TableType::SllWith16BitInputLow }
            }
            TableType::SllWith16BitInputHigh => {
                quote! { TableType::SllWith16BitInputHigh }
            }
            TableType::SrlWith16BitInputLow => {
                quote! { TableType::SrlWith16BitInputLow }
            }
            TableType::SrlWith16BitInputHigh => {
                quote! { TableType::SrlWith16BitInputHigh }
            }
            TableType::Sra16BitInputSignFill => {
                quote! { TableType::Sra16BitInputSignFill }
            }
            TableType::RangeCheck16WithZeroPads => {
                quote! { TableType::RangeCheck16WithZeroPads }
            }
            TableType::TruncateShiftAmount => {
                quote! { TableType::TruncateShiftAmount }
            }
            TableType::MemStoreClearOriginalRamValueLimb => {
                quote! { TableType::MemStoreClearOriginalRamValueLimb }
            }
            TableType::MemStoreClearWrittenValueLimb => {
                quote! { TableType::MemStoreClearWrittenValueLimb }
            }
            TableType::MemoryGetOffsetAndMaskWithTrap => {
                quote! { TableType::MemoryGetOffsetAndMaskWithTrap }
            }
            TableType::MemoryLoadHalfwordOrByte => quote! { TableType::MemoryLoadHalfwordOrByte },
            TableType::KeccakPermutationIndices12 => quote!(TableType::KeccakPermutationIndices12),
            TableType::KeccakPermutationIndices34 => quote!(TableType::KeccakPermutationIndices34),
            TableType::KeccakPermutationIndices56 => quote!(TableType::KeccakPermutationIndices56),
            TableType::XorSpecialIota => quote!(TableType::XorSpecialIota),
            TableType::AndN => quote!(TableType::AndN),
            TableType::RotL => quote!(TableType::RotL),
            TableType::Decoder => quote!(TableType::Decoder),
            TableType::DynamicPlaceholder => {
                unimplemented!("should not appear in final circuits")
            }
        };

        tokens.extend(stream);
    }
}
