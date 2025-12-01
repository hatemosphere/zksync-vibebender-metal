#pragma once

/*****
 * u32
 *****/

namespace airbender::ptx {

DEVICE_FORCEINLINE uint32_t mul_lo(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile ("mul.lo.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

DEVICE_FORCEINLINE uint32_t mul_hi(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(r): "r"(a), "r"(b));
  return r;
}

DEVICE_FORCEINLINE uint32_t mad_lo(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r): "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE_FORCEINLINE uint32_t mad_lo_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r): "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE_FORCEINLINE uint32_t mad_hi_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
  uint32_t result;
  asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

DEVICE_FORCEINLINE uint32_t madc_hi(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r): "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE_FORCEINLINE uint32_t madc_hi_cc(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r): "r"(a), "r"(b), "r"(c));
  return r;
}

DEVICE_FORCEINLINE uint32_t madc_lo_cc(const uint32_t x, const uint32_t y, const uint32_t z) {
  uint32_t result;
  asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(x), "r"(y), "r"(z));
  return result;
}

DEVICE_FORCEINLINE uint32_t addc(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("addc.u32 %0, %1, %2;" : "=r"(r): "r"(a), "r"(b));
  return r;
}

DEVICE_FORCEINLINE uint32_t add_cc(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r): "r"(a), "r"(b));
  return r;
}

DEVICE_FORCEINLINE uint32_t addc_cc(uint32_t a, uint32_t b) {
  uint32_t r;
  asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(r): "r"(a), "r"(b));
  return r;
}

DEVICE_FORCEINLINE uint64_t sub_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("sub.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

DEVICE_FORCEINLINE uint64_t subc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("subc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

DEVICE_FORCEINLINE uint64_t subc_cc(const uint64_t x, const uint64_t y) {
  uint64_t result;
  asm volatile("subc.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(x), "l"(y));
  return result;
}

/*****
 * u64
 *****/

DEVICE_FORCEINLINE uint64_t mul_lo(uint64_t a, uint64_t b) {
  uint64_t r;
  asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}

DEVICE_FORCEINLINE uint64_t mul_hi(uint64_t a, uint64_t b) {
  uint64_t r;
  asm volatile("mul.hi.u64 %0, %1, %2;" : "=l"(r): "l"(a), "l"(b));
  return r;
}

DEVICE_FORCEINLINE uint64_t mad_lo_cc(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t r;
  asm volatile("mad.lo.cc.u64 %0, %1, %2, %3;" : "=l"(r): "l"(a), "l"(b), "l"(c));
  return r;
}

DEVICE_FORCEINLINE uint64_t mad_hi_cc(const uint64_t x, const uint64_t y, const uint64_t z) {
  uint64_t result;
  asm volatile("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

DEVICE_FORCEINLINE uint64_t madc_hi(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t r;
  asm volatile("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r): "l"(a), "l"(b), "l"(c));
  return r;
}

DEVICE_FORCEINLINE uint64_t madc_hi_cc(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t r;
  asm volatile("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r): "l"(a), "l"(b), "l"(c));
  return r;
}

DEVICE_FORCEINLINE uint64_t madc_lo_cc(const uint64_t x, const uint64_t y, const uint64_t z) {
  uint64_t result;
  asm volatile("madc.lo.cc.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(x), "l"(y), "l"(z));
  return result;
}

DEVICE_FORCEINLINE uint64_t addc(uint64_t a, uint64_t b) {
  uint64_t r;
  asm volatile("addc.u64 %0, %1, %2;" : "=l"(r): "l"(a), "l"(b));
  return r;
}

} // namespace airbender::ptx
