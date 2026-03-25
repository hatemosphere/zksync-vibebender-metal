#pragma once

#include "../common.metal"
#include "../field.metal"

namespace airbender {
namespace witness {

using namespace ::airbender::field;

typedef base_field bf;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

DEVICE_FORCEINLINE u16 u16_low_from_u32(const u32 value) { return static_cast<u16>(value & 0xFFFF); }
DEVICE_FORCEINLINE u16 u16_high_from_u32(const u32 value) { return static_cast<u16>(value >> 16); }
DEVICE_FORCEINLINE u32 u16_pair_to_u32(const u16 low, const u16 high) { return static_cast<u32>(low) | (static_cast<u32>(high) << 16); }

DEVICE_FORCEINLINE u32 u32_low_from_u64(const u64 value) { return static_cast<u32>(value & 0xFFFFFFFF); }
DEVICE_FORCEINLINE u32 u32_high_from_u64(const u64 value) { return static_cast<u32>(value >> 32); }
DEVICE_FORCEINLINE u64 u32_pair_to_u64(const u32 low, const u32 high) { return static_cast<u64>(low) | (static_cast<u64>(high) << 32); }

// add_carry returns {result, carry} packed as (result_type, bool)
// For u8: result in low byte, carry in high byte of u16
DEVICE_FORCEINLINE u16 add_carry_u8(const u8 lhs, const u8 rhs) {
  const u16 sum = static_cast<u16>(lhs) + static_cast<u16>(rhs);
  return sum; // low byte = result, high byte = carry
}

DEVICE_FORCEINLINE u32 add_carry_u16(const u16 lhs, const u16 rhs) {
  const u32 sum = static_cast<u32>(lhs) + static_cast<u32>(rhs);
  return sum; // low 16 = result, bit 16 = carry
}

DEVICE_FORCEINLINE u64 add_carry_u32(const u32 lhs, const u32 rhs) {
  const u64 sum = static_cast<u64>(lhs) + static_cast<u64>(rhs);
  return sum; // low 32 = result, bit 32 = carry
}

// sub_borrow: returns {result, borrow} where borrow=1 means underflow
DEVICE_FORCEINLINE u16 sub_borrow_u8(const u8 lhs, const u8 rhs) {
  const u16 diff = (static_cast<u16>(1) << 8 | static_cast<u16>(lhs)) - static_cast<u16>(rhs);
  return diff ^ (1u << 8); // low byte = result, high byte = borrow
}

DEVICE_FORCEINLINE u32 sub_borrow_u16(const u16 lhs, const u16 rhs) {
  const u32 diff = (static_cast<u32>(1) << 16 | static_cast<u32>(lhs)) - static_cast<u32>(rhs);
  return diff ^ (1u << 16); // low 16 = result, bit 16 = borrow
}

DEVICE_FORCEINLINE u64 sub_borrow_u32(const u32 lhs, const u32 rhs) {
  const u64 diff = (static_cast<u64>(1) << 32 | static_cast<u64>(lhs)) - static_cast<u64>(rhs);
  return diff ^ (static_cast<u64>(1) << 32); // low 32 = result, bit 32 = borrow
}

// Overloaded add_carry/sub_borrow matching CUDA interface
// Returns uint2{result, carry/borrow}
DEVICE_FORCEINLINE uint2 add_carry(const u8 lhs, const u8 rhs) {
  const u16 r = add_carry_u8(lhs, rhs);
  return uint2(r & 0xFF, (r >> 8) & 0x1);
}

DEVICE_FORCEINLINE uint2 add_carry(const u16 lhs, const u16 rhs) {
  const u32 r = add_carry_u16(lhs, rhs);
  return uint2(r & 0xFFFF, (r >> 16) & 0x1);
}

DEVICE_FORCEINLINE uint2 add_carry(const u32 lhs, const u32 rhs) {
  const u64 r = add_carry_u32(lhs, rhs);
  return uint2(u32_low_from_u64(r), static_cast<u32>((r >> 32) & 0x1));
}

DEVICE_FORCEINLINE uint2 sub_borrow(const u8 lhs, const u8 rhs) {
  const u16 r = sub_borrow_u8(lhs, rhs);
  return uint2(r & 0xFF, (r >> 8) & 0x1);
}

DEVICE_FORCEINLINE uint2 sub_borrow(const u16 lhs, const u16 rhs) {
  const u32 r = sub_borrow_u16(lhs, rhs);
  return uint2(r & 0xFFFF, (r >> 16) & 0x1);
}

DEVICE_FORCEINLINE uint2 sub_borrow(const u32 lhs, const u32 rhs) {
  const u64 r = sub_borrow_u32(lhs, rhs);
  return uint2(u32_low_from_u64(r), static_cast<u32>((r >> 32) & 0x1));
}

} // namespace witness
} // namespace airbender
