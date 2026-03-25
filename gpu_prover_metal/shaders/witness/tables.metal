#pragma once

#include "common.metal"

using namespace ::airbender::witness;

namespace airbender {
namespace witness {
namespace tables {

enum TableType : u16 {
  ZeroEntry = 0,
  OpTypeBitmask,
  PowersOf2,
  InsnEncodingChecker,
  Xor = 4,
  CsrBitmask,
  Or = 6,
  And = 7,
  RangeCheckSmall, // 8
  RangeCheckLarge,
  AndNot,
  QuickDecodeDecompositionCheck4x4x4,
  QuickDecodeDecompositionCheck7x3x6,
  MRetProcessLow,
  MRetClearHigh,
  TrapProcessLow,
  U16GetSignAndHighByte, // 16
  JumpCleanupOffset,
  MemoryOffsetGetBits,
  MemoryLoadGetSigns,
  SRASignFiller,
  ConditionalOpUnsignedConditionsResolver,
  ConditionalOpAllConditionsResolver,
  RomAddressSpaceSeparator,
  RomRead, // 24
  SpecialCSRProperties,
  Xor3,
  Xor4,
  Xor7,
  Xor9,
  Xor12,
  U16SplitAsBytes,
  RangeCheck9x9, // 32
  RangeCheck10x10,
  RangeCheck11,
  RangeCheck12,
  RangeCheck13,
  ShiftImplementation,
  U16SelectByteAndGetByteSign,
  ExtendLoadedValue,
  StoreByteSourceContribution,
  StoreByteExistingContribution,
  TruncateShift,
  ExtractLower5Bits,
  DynamicPlaceholder,
};

DEVICE_FORCEINLINE void set_u16_values_from_u32(const u32 value, thread bf *values) {
  values[0] = bf{value & 0xFFFF};
  values[1] = bf{value >> 16};
}

template <uint K> DEVICE_FORCEINLINE void keys_into_binary_keys(const thread bf *keys, thread u32 *binary_keys) {
  for (uint i = 0; i < K; i++)
    binary_keys[i] = bf::into_canonical_u32(keys[i]);
}

// Variadic shift index computation using arrays
template <uint N> DEVICE_FORCEINLINE u32 index_for_binary_keys_with_shifts(const thread u32 *keys, const thread u32 *shifts) {
  u32 result = shifts[0] ? keys[0] << shifts[0] : keys[0];
  for (uint i = 1; i < N; i++)
    result |= shifts[i] ? keys[i] << shifts[i] : keys[i];
  return result;
}

DEVICE_FORCEINLINE u32 index_for_single_key(const thread bf *keys) {
  return bf::into_canonical_u32(keys[0]);
}

template <uint N> DEVICE_FORCEINLINE void set_to_zero(thread bf *values) {
  for (uint i = 0; i < N; i++)
    values[i] = bf::zero();
}

template <uint K, uint V> struct TableDriver {
  const device bf *tables;
  uint stride;
  const constant u32 *offsets;

  DEVICE_FORCEINLINE u32 get_absolute_index(const TableType t, const u32 index) const {
    return offsets[t] + index;
  }

  DEVICE_FORCEINLINE void set_values_from_tables(const u32 absolute_index, thread bf *values) const {
    if (V == 0) return;
    const uint col_offset = absolute_index / (stride - 1) * (1 + K + V) + K;
    const uint row = absolute_index % (stride - 1);
    for (uint i = 0; i < V; i++) {
      const uint col = i + col_offset;
      const uint idx = col * stride + row;
      values[i] = tables[idx];
    }
  }

  DEVICE_FORCEINLINE u32 single_key_set_values_from_tables(const TableType t, const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    const u32 absolute_index = get_absolute_index(t, index);
    if (V != 0)
      set_values_from_tables(absolute_index, values);
    return absolute_index;
  }

  DEVICE_FORCEINLINE u32 op_type_bitmask(const thread bf *keys, thread bf *values) const {
    return single_key_set_values_from_tables(OpTypeBitmask, keys, values);
  }

  DEVICE_FORCEINLINE u32 powers_of_2(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      const u32 shifted = 1u << index;
      values[0] = bf{shifted & 0xFFFF};
      values[1] = bf{shifted >> 16};
    }
    return get_absolute_index(PowersOf2, index);
  }

  template <TableType T, uint WIDTH>
  DEVICE_FORCEINLINE u32 binary_op_xor(const thread bf *keys, thread bf *values) const {
    u32 binary_keys[2];
    keys_into_binary_keys<2>(keys, binary_keys);
    const u32 index = (binary_keys[0] << WIDTH) | binary_keys[1];
    if (V != 0)
      values[0] = bf{binary_keys[0] ^ binary_keys[1]};
    return get_absolute_index(T, index);
  }

  DEVICE_FORCEINLINE u32 or_(const thread bf *keys, thread bf *values) const {
    u32 binary_keys[2];
    keys_into_binary_keys<2>(keys, binary_keys);
    const u32 index = (binary_keys[0] << 8) | binary_keys[1];
    if (V != 0)
      values[0] = bf{binary_keys[0] | binary_keys[1]};
    return get_absolute_index(Or, index);
  }

  DEVICE_FORCEINLINE u32 and_(const thread bf *keys, thread bf *values) const {
    u32 binary_keys[2];
    keys_into_binary_keys<2>(keys, binary_keys);
    const u32 index = (binary_keys[0] << 8) | binary_keys[1];
    if (V != 0)
      values[0] = bf{binary_keys[0] & binary_keys[1]};
    return get_absolute_index(And, index);
  }

  DEVICE_FORCEINLINE u32 and_not(const thread bf *keys, thread bf *values) const {
    u32 binary_keys[2];
    keys_into_binary_keys<2>(keys, binary_keys);
    const u32 index = (binary_keys[0] << 8) | binary_keys[1];
    if (V != 0)
      values[0] = bf{binary_keys[0] & (!binary_keys[1] ? 1u : 0u)};
    return get_absolute_index(AndNot, index);
  }

  template <TableType T, uint SHIFT0, uint SHIFT1>
  DEVICE_FORCEINLINE u32 ranges_generic_2(const thread bf *keys) const {
    u32 binary_keys[2];
    keys_into_binary_keys<2>(keys, binary_keys);
    const u32 index = (binary_keys[0] << SHIFT0) | binary_keys[1];
    return get_absolute_index(T, index);
  }

  template <TableType T, uint SHIFT0, uint SHIFT1, uint SHIFT2>
  DEVICE_FORCEINLINE u32 ranges_generic_3(const thread bf *keys) const {
    u32 binary_keys[3];
    keys_into_binary_keys<3>(keys, binary_keys);
    const u32 index = (binary_keys[0] << SHIFT0) | (binary_keys[1] << SHIFT1) | binary_keys[2];
    return get_absolute_index(T, index);
  }

  template <TableType T>
  DEVICE_FORCEINLINE u32 ranges_generic_1(const thread bf *keys) const {
    const u32 index = index_for_single_key(keys);
    return get_absolute_index(T, index);
  }

  DEVICE_FORCEINLINE u32 u16_get_sign_and_high_byte(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      values[0] = bf{index >> 15};
      values[1] = bf{index >> 8};
    }
    return get_absolute_index(U16GetSignAndHighByte, index);
  }

  DEVICE_FORCEINLINE u32 jump_cleanup_offset(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      values[0] = bf{(index >> 1) & 0x1};
      values[1] = bf{index & ~0x3u};
    }
    return get_absolute_index(JumpCleanupOffset, index);
  }

  DEVICE_FORCEINLINE u32 memory_offset_get_bits(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      values[0] = bf{index & 0x1};
      values[1] = bf{(index >> 1) & 0x1};
    }
    return get_absolute_index(MemoryOffsetGetBits, index);
  }

  DEVICE_FORCEINLINE u32 memory_load_get_signs(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      values[0] = bf{(index >> 7) & 0x1};
      values[1] = bf{(index >> 15) & 0x1};
    }
    return get_absolute_index(MemoryLoadGetSigns, index);
  }

  DEVICE_FORCEINLINE u32 sra_sign_filler(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      const bool input_sign = (index & 1) != 0;
      const bool is_sra = ((index >> 1) & 1) != 0;
      const u32 shift_amount = index >> 2;
      if (!input_sign || !is_sra || shift_amount == 0) {
        values[0] = bf{0};
        values[1] = bf{0};
      } else {
        const uint mask = 0xFFFFFFFF << (32 - shift_amount);
        values[0] = bf{mask & 0xFFFF};
        values[1] = bf{mask >> 16};
      }
    }
    return get_absolute_index(SRASignFiller, index);
  }

  DEVICE_FORCEINLINE u32 conditional_op_all_conditions(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      const u32 funct3 = index & 0x7;
      const bool unsigned_lt_flag = (index & (1u << 3)) != 0;
      const bool eq_flag = (index & (1u << 4)) != 0;
      const bool src1_bit = (index & (1u << 5)) != 0;
      const bool src2_bit = (index & (1u << 6)) != 0;
      const bool operands_different_signs_flag = src1_bit ^ src2_bit;
      bool should_branch = false;
      bool should_store = false;
      switch (funct3) {
      case 0: should_branch = eq_flag; break;
      case 1: should_branch = !eq_flag; break;
      case 2: should_store = operands_different_signs_flag ^ unsigned_lt_flag; break;
      case 3: should_store = unsigned_lt_flag; break;
      case 4: should_branch = operands_different_signs_flag ^ unsigned_lt_flag; break;
      case 5: should_branch = !(operands_different_signs_flag ^ unsigned_lt_flag); break;
      case 6: should_branch = unsigned_lt_flag; break;
      case 7: should_branch = !unsigned_lt_flag; break;
      }
      values[0] = bf{static_cast<u32>(should_branch)};
      values[1] = bf{static_cast<u32>(should_store)};
    }
    return get_absolute_index(ConditionalOpAllConditionsResolver, index);
  }

  DEVICE_FORCEINLINE u32 rom_address_space_separator(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      constexpr uint ROM_ADDRESS_SPACE_SECOND_WORD_BITS = 5;
      values[0] = bf{static_cast<u32>((index >> ROM_ADDRESS_SPACE_SECOND_WORD_BITS) != 0)};
      values[1] = bf{index & ((1u << ROM_ADDRESS_SPACE_SECOND_WORD_BITS) - 1)};
    }
    return get_absolute_index(RomAddressSpaceSeparator, index);
  }

  DEVICE_FORCEINLINE u32 rom_read(const thread bf *keys, thread bf *values) const {
    const u32 index = bf::into_canonical_u32(keys[0]) >> 2;
    const u32 absolute_index = get_absolute_index(RomRead, index);
    if (V != 0)
      set_values_from_tables(absolute_index, values);
    return absolute_index;
  }

  DEVICE_FORCEINLINE u32 special_csr_properties(const thread bf *keys, thread bf *values) const {
    return single_key_set_values_from_tables(SpecialCSRProperties, keys, values);
  }

  DEVICE_FORCEINLINE u32 u16_split_as_bytes(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      values[0] = bf{index & 0xFF};
      values[1] = bf{index >> 8};
    }
    return get_absolute_index(U16SplitAsBytes, index);
  }

  DEVICE_FORCEINLINE u32 shift_implementation(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      const u32 input_word = index & 0xFFFF;
      const u32 shift_amount = (index >> 16) & 0x1F;
      const bool is_right_shift = (index >> (16 + 5)) != 0;
      if (is_right_shift) {
        const u32 input = input_word << 16;
        const u32 t = input >> shift_amount;
        values[0] = bf{t >> 16};
        values[1] = bf{t & 0xFFFF};
      } else {
        const u32 input = input_word;
        const u32 t = input << shift_amount;
        values[0] = bf{t & 0xFFFF};
        values[1] = bf{t >> 16};
      }
    }
    return get_absolute_index(ShiftImplementation, index);
  }

  DEVICE_FORCEINLINE u32 u16_select_byte_and_get_byte_sign(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      const bool selector_bit = (index >> 16) != 0;
      const u32 selected_byte = (selector_bit ? index >> 8 : index) & 0xFF;
      const bool sign_bit = (selected_byte & (1u << 7)) != 0;
      values[0] = bf{selected_byte};
      values[1] = bf{static_cast<u32>(sign_bit)};
    }
    return get_absolute_index(U16SelectByteAndGetByteSign, index);
  }

  DEVICE_FORCEINLINE u32 extend_loaded_value(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      const u32 word = index & 0xFFFF;
      const bool use_high_half = (index & 0x00010000) != 0;
      const u32 funct3 = index >> 17;
      const u32 selected_byte = use_high_half ? word >> 8 : word & 0xFF;
      u32 loaded_word = 0;
      switch (funct3) {
      case 0: loaded_word = (selected_byte & 0x80) != 0 ? selected_byte | 0xFFFFFF00 : selected_byte; break;
      case 4: loaded_word = selected_byte; break;
      case 1: loaded_word = (word & 0x8000) != 0 ? word | 0xFFFF0000 : word; break;
      case 5: loaded_word = word; break;
      default: loaded_word = 0;
      }
      values[0] = bf{loaded_word & 0xFFFF};
      values[1] = bf{loaded_word >> 16};
    }
    return get_absolute_index(ExtendLoadedValue, index);
  }

  DEVICE_FORCEINLINE u32 store_byte_source_contribution(const thread bf *keys, thread bf *values) const {
    u32 binary_keys[2];
    keys_into_binary_keys<2>(keys, binary_keys);
    const u32 index = (binary_keys[0] << 1) | binary_keys[1];
    if (V != 0) {
      const bool bit_0 = binary_keys[1] != 0;
      const u32 byte_val = binary_keys[0] & 0xFF;
      const u32 result = bit_0 ? byte_val << 8 : byte_val;
      values[0] = bf{result};
    }
    return get_absolute_index(StoreByteSourceContribution, index);
  }

  DEVICE_FORCEINLINE u32 store_byte_existing_contribution(const thread bf *keys, thread bf *values) const {
    u32 binary_keys[2];
    keys_into_binary_keys<2>(keys, binary_keys);
    const u32 index = (binary_keys[0] << 1) | binary_keys[1];
    if (V != 0) {
      const bool bit_0 = binary_keys[1] != 0;
      const u32 result = bit_0 ? binary_keys[0] & 0x00FF : binary_keys[0] & 0xFF00;
      values[0] = bf{result};
    }
    return get_absolute_index(StoreByteExistingContribution, index);
  }

  DEVICE_FORCEINLINE u32 truncate_shift(const thread bf *keys, thread bf *values) const {
    u32 binary_keys[2];
    keys_into_binary_keys<2>(keys, binary_keys);
    const u32 index = (binary_keys[0] << 1) | binary_keys[1];
    if (V != 0) {
      const bool is_right_shift = binary_keys[1] != 0;
      const u32 shift_amount = binary_keys[0] & 31;
      const u32 result = is_right_shift ? shift_amount : 32 - shift_amount;
      values[0] = bf{result};
    }
    return get_absolute_index(TruncateShift, index);
  }

  DEVICE_FORCEINLINE u32 extract_lower_5_bits(const thread bf *keys, thread bf *values) const {
    const u32 index = index_for_single_key(keys);
    if (V != 0) {
      values[0] = bf{index & 0x1F};
      values[1] = bf{0};
    }
    return get_absolute_index(ExtractLower5Bits, index);
  }

  DEVICE_FORCEINLINE u32 get_index_and_set_values(const TableType table_type, const thread bf *keys, thread bf *values) const {
    switch (table_type) {
    case ZeroEntry:
      set_to_zero<V>(values);
      return 0;
    case OpTypeBitmask:
      return op_type_bitmask(keys, values);
    case PowersOf2:
      return powers_of_2(keys, values);
    case Xor:
      return binary_op_xor<Xor, 8>(keys, values);
    case Or:
      return or_(keys, values);
    case And:
      return and_(keys, values);
    case RangeCheckSmall:
      return ranges_generic_2<RangeCheckSmall, 8, 0>(keys);
    case RangeCheckLarge:
      return ranges_generic_1<RangeCheckLarge>(keys);
    case AndNot:
      return and_not(keys, values);
    case QuickDecodeDecompositionCheck4x4x4:
      return ranges_generic_3<QuickDecodeDecompositionCheck4x4x4, 8, 4, 0>(keys);
    case QuickDecodeDecompositionCheck7x3x6:
      return ranges_generic_3<QuickDecodeDecompositionCheck7x3x6, 9, 6, 0>(keys);
    case U16GetSignAndHighByte:
      return u16_get_sign_and_high_byte(keys, values);
    case JumpCleanupOffset:
      return jump_cleanup_offset(keys, values);
    case MemoryOffsetGetBits:
      return memory_offset_get_bits(keys, values);
    case MemoryLoadGetSigns:
      return memory_load_get_signs(keys, values);
    case SRASignFiller:
      return sra_sign_filler(keys, values);
    case ConditionalOpAllConditionsResolver:
      return conditional_op_all_conditions(keys, values);
    case RomAddressSpaceSeparator:
      return rom_address_space_separator(keys, values);
    case RomRead:
      return rom_read(keys, values);
    case SpecialCSRProperties:
      return special_csr_properties(keys, values);
    case Xor3:
      return binary_op_xor<Xor3, 3>(keys, values);
    case Xor4:
      return binary_op_xor<Xor4, 4>(keys, values);
    case Xor7:
      return binary_op_xor<Xor7, 7>(keys, values);
    case Xor9:
      return binary_op_xor<Xor9, 9>(keys, values);
    case Xor12:
      return binary_op_xor<Xor12, 12>(keys, values);
    case U16SplitAsBytes:
      return u16_split_as_bytes(keys, values);
    case RangeCheck9x9:
      return ranges_generic_2<RangeCheck9x9, 9, 0>(keys);
    case RangeCheck10x10:
      return ranges_generic_2<RangeCheck10x10, 10, 0>(keys);
    case RangeCheck11:
      return ranges_generic_1<RangeCheck11>(keys);
    case RangeCheck12:
      return ranges_generic_1<RangeCheck12>(keys);
    case RangeCheck13:
      return ranges_generic_1<RangeCheck13>(keys);
    case ShiftImplementation:
      return shift_implementation(keys, values);
    case U16SelectByteAndGetByteSign:
      return u16_select_byte_and_get_byte_sign(keys, values);
    case ExtendLoadedValue:
      return extend_loaded_value(keys, values);
    case StoreByteSourceContribution:
      return store_byte_source_contribution(keys, values);
    case StoreByteExistingContribution:
      return store_byte_existing_contribution(keys, values);
    case TruncateShift:
      return truncate_shift(keys, values);
    case ExtractLower5Bits:
      return extract_lower_5_bits(keys, values);
    default:
      return 0; // No __trap() in Metal; return sentinel
    }
  }
};

} // namespace tables
} // namespace witness
} // namespace airbender
