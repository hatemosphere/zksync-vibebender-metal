#pragma once

#include "placeholder.metal"
#include "tables.metal"

using namespace ::airbender::witness::placeholder;
using namespace ::airbender::witness::tables;

namespace airbender {
namespace witness {
namespace generation {

using namespace field;

struct wrapped_f {
  typedef bf innerType;
  bf inner;

  static constexpr DEVICE_FORCEINLINE wrapped_f new_(const bf value) { return {bf::into_canonical(value)}; }
  static constexpr DEVICE_FORCEINLINE wrapped_f new_(const u32 value) { return {bf::into_canonical(bf{value})}; }

  template <typename T> static DEVICE_FORCEINLINE wrapped_f from(const thread T &value);

  static DEVICE_FORCEINLINE wrapped_f add(const thread wrapped_f &lhs, const thread wrapped_f &rhs) { return wrapped_f{bf::into_canonical(bf::add(lhs.inner, rhs.inner))}; }
  static DEVICE_FORCEINLINE wrapped_f sub(const thread wrapped_f &lhs, const thread wrapped_f &rhs) { return wrapped_f{bf::into_canonical(bf::sub(lhs.inner, rhs.inner))}; }
  static DEVICE_FORCEINLINE wrapped_f mul(const thread wrapped_f &lhs, const thread wrapped_f &rhs) { return wrapped_f{bf::into_canonical(bf::mul(lhs.inner, rhs.inner))}; }

  static DEVICE_FORCEINLINE wrapped_f mul_add(const thread wrapped_f &mul_0, const thread wrapped_f &mul_1, const thread wrapped_f &a) {
    return wrapped_f{bf::into_canonical(bf::add(bf::mul(mul_0.inner, mul_1.inner), a.inner))};
  }

  static DEVICE_FORCEINLINE wrapped_f inv(const thread wrapped_f &value) { return wrapped_f{bf::into_canonical(bf::inv(value.inner))}; }
};

struct wrapped_b {
  typedef bool innerType;
  bool inner;

  static constexpr DEVICE_FORCEINLINE wrapped_b new_(const bool value) { return {value}; }

  template <typename T> static DEVICE_FORCEINLINE wrapped_b from(const thread T &value) { return wrapped_b{static_cast<bool>(value.inner)}; }

  template <typename T> static DEVICE_FORCEINLINE wrapped_b from_integer_equality(const thread T &lhs, const thread T &rhs) { return wrapped_b{lhs.inner == rhs.inner}; }

  template <typename T> static DEVICE_FORCEINLINE wrapped_b from_integer_carry(const thread T &lhs, const thread T &rhs) { return wrapped_b{T::add_carry(lhs, rhs)}; }

  template <typename T> static DEVICE_FORCEINLINE wrapped_b from_integer_borrow(const thread T &lhs, const thread T &rhs) { return wrapped_b{T::sub_borrow(lhs, rhs)}; }

  static DEVICE_FORCEINLINE wrapped_b from_field_equality(const thread wrapped_f &lhs, const thread wrapped_f &rhs) {
    return wrapped_b{bf::into_canonical_u32(lhs.inner) == bf::into_canonical_u32(rhs.inner)};
  }

  static DEVICE_FORCEINLINE wrapped_b and_(const thread wrapped_b &lhs, const thread wrapped_b &rhs) { return wrapped_b{lhs.inner && rhs.inner}; }
  static DEVICE_FORCEINLINE wrapped_b or_(const thread wrapped_b &lhs, const thread wrapped_b &rhs) { return wrapped_b{lhs.inner || rhs.inner}; }

  template <typename T> static DEVICE_FORCEINLINE T select(const thread wrapped_b &selector, const thread T &if_true, const thread T &if_false) {
    return selector.inner ? if_true : if_false;
  }

  static DEVICE_FORCEINLINE wrapped_b negate(const thread wrapped_b &value) { return wrapped_b{!value.inner}; }
};

template <typename T> struct wrapped_integer {
  typedef T innerType;
  T inner;

  static constexpr DEVICE_FORCEINLINE wrapped_integer new_(const T value) { return {value}; }

  template <typename U> static DEVICE_FORCEINLINE wrapped_integer from(const thread U &value) { return wrapped_integer{static_cast<T>(value.inner)}; }

  static DEVICE_FORCEINLINE wrapped_integer add(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs) { return wrapped_integer{static_cast<T>(lhs.inner + rhs.inner)}; }

  static DEVICE_FORCEINLINE bool add_carry(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs) {
    return ::airbender::witness::add_carry(lhs.inner, rhs.inner).y;
  }

  static DEVICE_FORCEINLINE wrapped_integer sub(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs) { return wrapped_integer{static_cast<T>(lhs.inner - rhs.inner)}; }

  static DEVICE_FORCEINLINE bool sub_borrow(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs) {
    return ::airbender::witness::sub_borrow(lhs.inner, rhs.inner).y;
  }

  static DEVICE_FORCEINLINE wrapped_integer mul(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs) { return wrapped_integer{static_cast<T>(lhs.inner * rhs.inner)}; }

  static DEVICE_FORCEINLINE wrapped_integer mul_add(const thread wrapped_integer &mul_0, const thread wrapped_integer &mul_1, const thread wrapped_integer &a) {
    return wrapped_integer{static_cast<T>(mul_0.inner * mul_1.inner + a.inner)};
  }

  static DEVICE_FORCEINLINE wrapped_integer shl(const thread wrapped_integer &value, const thread uint &shift) { return wrapped_integer{static_cast<T>(value.inner << shift)}; }
  static DEVICE_FORCEINLINE wrapped_integer shr(const thread wrapped_integer &value, const thread uint &shift) { return wrapped_integer{static_cast<T>(value.inner >> shift)}; }

  static DEVICE_FORCEINLINE wrapped_integer lowest_bits(const thread wrapped_integer &value, const thread uint &count) {
    return wrapped_integer{static_cast<T>(value.inner & ((1u << count) - 1))};
  }

  static DEVICE_FORCEINLINE wrapped_integer mul_low(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs);
  static DEVICE_FORCEINLINE wrapped_integer mul_high(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs);

  static DEVICE_FORCEINLINE wrapped_integer div(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs) { return wrapped_integer{static_cast<T>(lhs.inner / rhs.inner)}; }
  static DEVICE_FORCEINLINE wrapped_integer rem(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs) { return wrapped_integer{static_cast<T>(lhs.inner % rhs.inner)}; }

  template <typename U> static DEVICE_FORCEINLINE U signed_mul_low(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs);
  template <typename U> static DEVICE_FORCEINLINE U signed_mul_high(const thread wrapped_integer &lhs, const thread wrapped_integer &rhs);
  template <typename U> static DEVICE_FORCEINLINE U mixed_mul_low(const thread wrapped_integer &lhs, const thread U &rhs);
  template <typename U> static DEVICE_FORCEINLINE U mixed_mul_high(const thread wrapped_integer &lhs, const thread U &rhs);
};

typedef wrapped_integer<u8> wrapped_u8;
typedef wrapped_integer<u16> wrapped_u16;
typedef wrapped_integer<u32> wrapped_u32;
typedef wrapped_integer<int32_t> wrapped_i32;

// Specializations for wrapped_f::from
template <> DEVICE_FORCEINLINE wrapped_f wrapped_f::from(const thread wrapped_f &value) { return value; }
template <> DEVICE_FORCEINLINE wrapped_f wrapped_f::from(const thread wrapped_b &value) { return wrapped_f{bf::into_canonical(bf::from_u32(static_cast<u32>(value.inner)))}; }
template <> DEVICE_FORCEINLINE wrapped_f wrapped_f::from(const thread wrapped_u8 &value) { return wrapped_f{bf::into_canonical(bf::from_u32(static_cast<u32>(value.inner)))}; }
template <> DEVICE_FORCEINLINE wrapped_f wrapped_f::from(const thread wrapped_u16 &value) { return wrapped_f{bf::into_canonical(bf::from_u32(static_cast<u32>(value.inner)))}; }
template <> DEVICE_FORCEINLINE wrapped_f wrapped_f::from(const thread wrapped_u32 &value) { return wrapped_f{bf::into_canonical(bf::from_u32(value.inner))}; }
template <> DEVICE_FORCEINLINE wrapped_f wrapped_f::from(const thread wrapped_i32 &value) { return wrapped_f{bf::into_canonical(bf::from_u32(static_cast<u32>(value.inner)))}; }

// Specializations for wrapped_b::from
template <> DEVICE_FORCEINLINE wrapped_b wrapped_b::from(const thread wrapped_f &value) {
  const u32 canonical = bf::into_canonical_u32(value.inner);
  return wrapped_b{static_cast<bool>(canonical)};
}

// Specializations for wrapped integer from wrapped_f
template <> template <> DEVICE_FORCEINLINE wrapped_u8 wrapped_u8::from(const thread wrapped_f &value) {
  const u32 canonical = bf::into_canonical_u32(value.inner);
  return wrapped_u8{static_cast<u8>(canonical)};
}

template <> template <> DEVICE_FORCEINLINE wrapped_u16 wrapped_u16::from(const thread wrapped_f &value) {
  const u32 canonical = bf::into_canonical_u32(value.inner);
  return wrapped_u16{static_cast<u16>(canonical)};
}

template <> template <> DEVICE_FORCEINLINE wrapped_u32 wrapped_u32::from(const thread wrapped_f &value) {
  const u32 canonical = bf::into_canonical_u32(value.inner);
  return wrapped_u32{canonical};
}

template <> template <> DEVICE_FORCEINLINE wrapped_i32 wrapped_i32::from(const thread wrapped_f &value) {
  const u32 canonical = bf::into_canonical_u32(value.inner);
  return wrapped_i32{static_cast<int32_t>(canonical)};
}

// mul_low specializations
template <> DEVICE_FORCEINLINE wrapped_u8 wrapped_u8::mul_low(const thread wrapped_u8 &lhs, const thread wrapped_u8 &rhs) {
  const u16 result = static_cast<u16>(lhs.inner) * static_cast<u16>(rhs.inner);
  return wrapped_u8{static_cast<u8>(result)};
}

template <> DEVICE_FORCEINLINE wrapped_u16 wrapped_u16::mul_low(const thread wrapped_u16 &lhs, const thread wrapped_u16 &rhs) {
  const u32 result = static_cast<u32>(lhs.inner) * static_cast<u32>(rhs.inner);
  return wrapped_u16{static_cast<u16>(result)};
}

template <> DEVICE_FORCEINLINE wrapped_u32 wrapped_u32::mul_low(const thread wrapped_u32 &lhs, const thread wrapped_u32 &rhs) {
  const u64 result = static_cast<u64>(lhs.inner) * static_cast<u64>(rhs.inner);
  return wrapped_u32{static_cast<u32>(result)};
}

template <> DEVICE_FORCEINLINE wrapped_i32 wrapped_i32::mul_low(const thread wrapped_i32 &lhs, const thread wrapped_i32 &rhs) {
  const int64_t result = static_cast<int64_t>(lhs.inner) * static_cast<int64_t>(rhs.inner);
  return wrapped_i32{static_cast<int32_t>(result)};
}

// mul_high specializations
template <> DEVICE_FORCEINLINE wrapped_u8 wrapped_u8::mul_high(const thread wrapped_u8 &lhs, const thread wrapped_u8 &rhs) {
  const u16 result = static_cast<u16>(lhs.inner) * static_cast<u16>(rhs.inner);
  return wrapped_u8{static_cast<u8>(result >> 8)};
}

template <> DEVICE_FORCEINLINE wrapped_u16 wrapped_u16::mul_high(const thread wrapped_u16 &lhs, const thread wrapped_u16 &rhs) {
  const u32 result = static_cast<u32>(lhs.inner) * static_cast<u32>(rhs.inner);
  return wrapped_u16{static_cast<u16>(result >> 16)};
}

template <> DEVICE_FORCEINLINE wrapped_u32 wrapped_u32::mul_high(const thread wrapped_u32 &lhs, const thread wrapped_u32 &rhs) {
  const u64 result = static_cast<u64>(lhs.inner) * static_cast<u64>(rhs.inner);
  return wrapped_u32{static_cast<u32>(result >> 32)};
}

template <> DEVICE_FORCEINLINE wrapped_i32 wrapped_i32::mul_high(const thread wrapped_i32 &lhs, const thread wrapped_i32 &rhs) {
  const int64_t result = static_cast<int64_t>(lhs.inner) * static_cast<int64_t>(rhs.inner);
  return wrapped_i32{static_cast<int32_t>(result >> 32)};
}

// signed_mul_low / signed_mul_high
template <> template <> DEVICE_FORCEINLINE wrapped_u32 wrapped_i32::signed_mul_low(const thread wrapped_i32 &lhs, const thread wrapped_i32 &rhs) {
  const int64_t result = static_cast<int64_t>(lhs.inner) * static_cast<int64_t>(rhs.inner);
  return wrapped_u32{static_cast<u32>(static_cast<u64>(result))};
}

template <> template <> DEVICE_FORCEINLINE wrapped_u32 wrapped_i32::signed_mul_high(const thread wrapped_i32 &lhs, const thread wrapped_i32 &rhs) {
  const int64_t result = static_cast<int64_t>(lhs.inner) * static_cast<int64_t>(rhs.inner);
  return wrapped_u32{static_cast<u32>(static_cast<u64>(result) >> 32)};
}

// mixed_mul_low / mixed_mul_high
template <> template <> DEVICE_FORCEINLINE wrapped_u32 wrapped_i32::mixed_mul_low(const thread wrapped_i32 &lhs, const thread wrapped_u32 &rhs) {
  const int64_t result = static_cast<int64_t>(lhs.inner) * static_cast<int64_t>(rhs.inner);
  return wrapped_u32{static_cast<u32>(static_cast<u64>(result))};
}

template <> template <> DEVICE_FORCEINLINE wrapped_u32 wrapped_i32::mixed_mul_high(const thread wrapped_i32 &lhs, const thread wrapped_u32 &rhs) {
  const int64_t result = static_cast<int64_t>(lhs.inner) * static_cast<int64_t>(rhs.inner);
  return wrapped_u32{static_cast<u32>(static_cast<u64>(result) >> 32)};
}

// WitnessProxy
template <class R> struct WitnessProxy {
  R oracle;
  const device wrapped_f *generic_lookup_tables;
  const device wrapped_f *memory;
  device wrapped_f *witness;
  device u32 *lookup_mappings;
  thread wrapped_f *scratch;
  uint stride;
  uint offset;

  template <typename T> DEVICE_FORCEINLINE T get_memory_place(const uint idx) const {
    const auto value = memory[idx * stride + offset];
    return T::from(value);
  }

  template <typename T> DEVICE_FORCEINLINE T get_witness_place(const uint idx) const {
    auto value = witness[idx * stride + offset];
    return T::from(value);
  }

  template <typename T> DEVICE_FORCEINLINE T get_scratch_place(const uint idx) const {
    auto value = scratch[idx];
    return T::from(value);
  }

  template <typename T> DEVICE_FORCEINLINE T get_oracle_value(const Placeholder placeholder) const {
    const auto value = oracle.template get_witness_from_placeholder<typename T::innerType>(placeholder, offset);
    return T{value};
  }

  template <typename T> DEVICE_FORCEINLINE void set_memory_place(const uint idx, const thread T &value) const {
    auto f = wrapped_f::from(value);
    memory[idx * stride + offset] = f;
  }

  template <typename T> DEVICE_FORCEINLINE void set_witness_place(const uint idx, const thread T &value) const {
    const auto f = wrapped_f::from(value);
    witness[idx * stride + offset] = f;
  }

  template <typename T> DEVICE_FORCEINLINE void set_scratch_place(const uint idx, const thread T &value) const {
    auto f = wrapped_f::from(value);
    scratch[idx] = f;
  }

  template <uint I, uint O>
  DEVICE_FORCEINLINE u32 get_lookup_index_and_value(const thread wrapped_f *inputs, const wrapped_u16 table_id, thread wrapped_f *outputs, const constant u32 *table_offsets) const {
    const device bf *tbls = reinterpret_cast<const device bf *>(generic_lookup_tables);
    const TableDriver<I, O> table_driver{tbls, stride, table_offsets};
    const auto table_type = static_cast<TableType>(table_id.inner);
    const thread bf *keys = reinterpret_cast<const thread bf *>(inputs);
    thread bf *values = reinterpret_cast<thread bf *>(outputs);
    const u32 index = table_driver.get_index_and_set_values(table_type, keys, values);
    return index;
  }

  template <uint I, uint O>
  DEVICE_FORCEINLINE void lookup(const thread wrapped_f *inputs, const wrapped_u16 table_id, thread wrapped_f *outputs, const uint lookup_mapping_idx,
                                 const constant u32 *table_offsets) const {
    const u32 index = get_lookup_index_and_value<I, O>(inputs, table_id, outputs, table_offsets);
    lookup_mappings[lookup_mapping_idx * stride + offset] = index;
  }

  template <uint N>
  DEVICE_FORCEINLINE void lookup_enforce(const thread wrapped_f *values, const wrapped_u16 table_id, const uint lookup_mapping_idx, const constant u32 *table_offsets) const {
    const u32 index = get_lookup_index_and_value<N, 0>(values, table_id, nullptr, table_offsets);
    lookup_mappings[lookup_mapping_idx * stride + offset] = index;
  }

  template <uint I, uint O>
  DEVICE_FORCEINLINE void maybe_lookup(const thread wrapped_f *inputs, const wrapped_u16 table_id, const wrapped_b mask, thread wrapped_f *outputs,
                                       const constant u32 *table_offsets) const {
    if (!mask.inner)
      return;
    get_lookup_index_and_value<I, O>(inputs, table_id, outputs, table_offsets);
  }
};

// DSL macros
#define VAR(N) var_##N
#define CONSTANT(T, N, VALUE) thread wrapped_##T VAR(N) = wrapped_##T::new_(VALUE);
#define GET_MEMORY_PLACE(T, N, IDX) const wrapped_##T VAR(N) = p.template get_memory_place<wrapped_##T>(IDX);
#define GET_WITNESS_PLACE(T, N, IDX) const wrapped_##T VAR(N) = p.template get_witness_place<wrapped_##T>(IDX);
#define GET_SCRATCH_PLACE(T, N, IDX) const wrapped_##T VAR(N) = p.template get_scratch_place<wrapped_##T>(IDX);
#define GET_ORACLE_VALUE(T, N, P) const wrapped_##T VAR(N) = p.template get_oracle_value<wrapped_##T>(P);
#define LOOKUP_TABLE_OFFSETS(...) static constant constexpr u32 lookup_table_offsets[] = {__VA_ARGS__};
#define LOOKUP_OUTPUTS(N, NO) wrapped_f VAR(N)[NO] = {};
#define LOOKUP(N, NI, NO, TID, LMI, ...)                                                                                                                       \
  LOOKUP_OUTPUTS(N, NO) {                                                                                                                                      \
    wrapped_f inputs[] = {__VA_ARGS__};                                                                                                                        \
    p.template lookup<NI, NO>(inputs, VAR(TID), VAR(N), LMI, lookup_table_offsets);                                                                            \
  }
#define LOOKUP_ENFORCE(NI, TID, LMI, ...)                                                                                                                      \
  {                                                                                                                                                            \
    wrapped_f inputs[] = {__VA_ARGS__};                                                                                                                        \
    p.template lookup_enforce<NI>(inputs, VAR(TID), LMI, lookup_table_offsets);                                                                                \
  }
#define MAYBE_LOOKUP(N, NI, NO, TID, M, ...)                                                                                                                   \
  LOOKUP_OUTPUTS(N, NO) {                                                                                                                                      \
    wrapped_f inputs[] = {__VA_ARGS__};                                                                                                                        \
    p.template maybe_lookup<NI, NO>(inputs, VAR(TID), VAR(M), VAR(N), lookup_table_offsets);                                                                   \
  }
#define ACCESS_LOOKUP(N, O, IDX) const wrapped_f VAR(N) = VAR(O)[IDX];
#define FROM(T, N, I) const wrapped_##T VAR(N) = wrapped_##T::from(VAR(I));
#define B_FROM_INTEGER_EQUALITY(N, LHS, RHS) const wrapped_b VAR(N) = wrapped_b::from_integer_equality(VAR(LHS), VAR(RHS));
#define B_FROM_INTEGER_CARRY(N, LHS, RHS) const wrapped_b VAR(N) = wrapped_b::from_integer_carry(VAR(LHS), VAR(RHS));
#define B_FROM_INTEGER_BORROW(N, LHS, RHS) const wrapped_b VAR(N) = wrapped_b::from_integer_borrow(VAR(LHS), VAR(RHS));
#define B_FROM_FIELD_EQUALITY(N, LHS, RHS) const wrapped_b VAR(N) = wrapped_b::from_field_equality(VAR(LHS), VAR(RHS));
#define AND(N, LHS, RHS) const wrapped_b VAR(N) = wrapped_b::and_(VAR(LHS), VAR(RHS));
#define OR(N, LHS, RHS) const wrapped_b VAR(N) = wrapped_b::or_(VAR(LHS), VAR(RHS));
#define SELECT(T, N, S, TRUE, FALSE) const wrapped_##T VAR(N) = wrapped_b::select(VAR(S), VAR(TRUE), VAR(FALSE));
#define SELECT_VAR(S, TRUE, FALSE) wrapped_b::select(VAR(S), VAR(TRUE), VAR(FALSE))
#define NEGATE(N, I) const wrapped_b VAR(N) = wrapped_b::negate(VAR(I));
#define ADD(T, N, LHS, RHS) const wrapped_##T VAR(N) = wrapped_##T::add(VAR(LHS), VAR(RHS));
#define SUB(T, N, LHS, RHS) const wrapped_##T VAR(N) = wrapped_##T::sub(VAR(LHS), VAR(RHS));
#define MUL(T, N, LHS, RHS) const wrapped_##T VAR(N) = wrapped_##T::mul(VAR(LHS), VAR(RHS));
#define MUL_ADD(T, N, M0, M1, A) const wrapped_##T VAR(N) = wrapped_##T::mul_add(VAR(M0), VAR(M1), VAR(A));
#define INV(T, N, I) const wrapped_##T VAR(N) = wrapped_##T::inv(VAR(I));
#define SHL(T, N, I, M) const wrapped_##T VAR(N) = wrapped_##T::shl(VAR(I), M);
#define SHR(T, N, I, M) const wrapped_##T VAR(N) = wrapped_##T::shr(VAR(I), M);
#define LOWEST_BITS(T, N, I, M) const wrapped_##T VAR(N) = wrapped_##T::lowest_bits(VAR(I), M);
#define MUL_LOW(T, N, LHS, RHS) const wrapped_##T VAR(N) = wrapped_##T::mul_low(VAR(LHS), VAR(RHS));
#define MUL_HIGH(T, N, LHS, RHS) const wrapped_##T VAR(N) = wrapped_##T::mul_high(VAR(LHS), VAR(RHS));
#define DIV(T, N, LHS, RHS) const wrapped_##T VAR(N) = wrapped_##T::div(VAR(LHS), VAR(RHS));
#define REM(T, N, LHS, RHS) const wrapped_##T VAR(N) = wrapped_##T::rem(VAR(LHS), VAR(RHS));
#define SIGNED_MUL_LOW(N, LHS, RHS) const wrapped_u32 VAR(N) = wrapped_i32::signed_mul_low<wrapped_u32>(VAR(LHS), VAR(RHS));
#define SIGNED_MUL_HIGH(N, LHS, RHS) const wrapped_u32 VAR(N) = wrapped_i32::signed_mul_high<wrapped_u32>(VAR(LHS), VAR(RHS));
#define MIXED_MUL_LOW(N, LHS, RHS) const wrapped_u32 VAR(N) = wrapped_i32::mixed_mul_low<wrapped_u32>(VAR(LHS), VAR(RHS));
#define MIXED_MUL_HIGH(N, LHS, RHS) const wrapped_u32 VAR(N) = wrapped_i32::mixed_mul_high<wrapped_u32>(VAR(LHS), VAR(RHS));
#define IF(S, T)                                                                                                                                               \
  if (VAR(S).inner) {                                                                                                                                          \
    T                                                                                                                                                          \
  }
#define SET_MEMORY_PLACE(IDX, V) p.set_memory_place(IDX, VAR(V));
#define SET_WITNESS_PLACE(IDX, V) p.set_witness_place(IDX, VAR(V));
#define SET_SCRATCH_PLACE(IDX, V) p.set_scratch_place(IDX, VAR(V));

#define FN_BEGIN(N) template <class R> DEVICE_FORCEINLINE void fn_##N(const WitnessProxy<R> p) {
#define FN_END }

#define FN_CALL(N) fn_##N(p);

// Circuit include path macros
#define INCLUDE_PREFIX ../../../../circuit_defs
#define INCLUDE_SUFFIX generated/witness_generation_fn.cuh
#define PATH_CAT(a, b, c) a/b/c
#define STRINGIFY(X) STRINGIFY2(X)
#define STRINGIFY2(X) #X
#define CIRCUIT_INCLUDE(NAME) STRINGIFY(PATH_CAT(INCLUDE_PREFIX, NAME, INCLUDE_SUFFIX))

#define KERNEL_NAME(NAME) ab_generate_##NAME##_witness_kernel
#define SCRATCH thread wrapped_f scratch[64] = {};

// Metal kernel macro — equivalent of CUDA's KERNEL(NAME, ORACLE)
// Creates a kernel entry point that invokes the generated witness functions.
//
// buffer(0) = the oracle's underlying data buffer (e.g., cycle_data for MainTrace).
// The ORACLE struct is constructed on-GPU from the buffer pointer.
// This avoids the CUDA pattern of passing a struct-with-device-pointers
// through constant memory (which doesn't work in Metal's address space model).
#define KERNEL(NAME, ORACLE)                                                                   \
kernel void KERNEL_NAME(NAME)(                                                                 \
    device const void* oracle_data [[buffer(0)]],                                              \
    device const wrapped_f* generic_lookup_tables [[buffer(1)]],                               \
    device const wrapped_f* memory_data [[buffer(2)]],                                         \
    device wrapped_f* witness [[buffer(3)]],                                                   \
    device uint* lookup_mappings [[buffer(4)]],                                                \
    constant uint& stride [[buffer(5)]],                                                       \
    constant uint& count [[buffer(6)]],                                                        \
    uint gid [[thread_position_in_grid]]                                                       \
) {                                                                                            \
    if (gid >= count) return;                                                                  \
    thread wrapped_f scratch[64] = {};                                                         \
    const ORACLE oracle = ORACLE::from_buffer(oracle_data);                                    \
    const WitnessProxy<ORACLE> p = {oracle, generic_lookup_tables, memory_data,                \
                                     witness, lookup_mappings, scratch, stride, gid};           \
    FN_CALL(generate)                                                                          \
}

// Delegation kernel macro — builds DelegationTrace from individual buffer parameters
// instead of struct copy (from_buffer has issues with device pointers on Metal).
// Buffer layout matches witness_delegation.rs dispatch order.
#define DELEGATION_KERNEL(NAME)                                                                    \
kernel void KERNEL_NAME(NAME)(                                                                     \
    constant uint& num_requests [[buffer(0)]],                                                     \
    constant uint& num_register_accesses [[buffer(1)]],                                            \
    constant uint& num_indirect_reads [[buffer(2)]],                                               \
    constant uint& num_indirect_writes [[buffer(3)]],                                              \
    constant uint& base_register_index [[buffer(4)]],                                              \
    constant u16& delegation_type_val [[buffer(5)]],                                               \
    constant uint* indirect_accesses_props [[buffer(6)]],                                          \
    const device TimestampData* write_ts_data [[buffer(7)]],                                       \
    const device RegisterOrIndirectReadWriteData* reg_acc_data [[buffer(8)]],                      \
    const device RegisterOrIndirectReadData* ind_read_data [[buffer(9)]],                          \
    const device RegisterOrIndirectReadWriteData* ind_write_data [[buffer(10)]],                   \
    device const wrapped_f* generic_lookup_tables [[buffer(11)]],                                  \
    device const wrapped_f* memory_data [[buffer(12)]],                                            \
    device wrapped_f* witness [[buffer(13)]],                                                      \
    device uint* lookup_mappings [[buffer(14)]],                                                   \
    constant uint& stride [[buffer(15)]],                                                          \
    constant uint& count [[buffer(16)]],                                                           \
    uint gid [[thread_position_in_grid]]                                                           \
) {                                                                                                \
    using namespace airbender::witness::trace::delegation;                                         \
    if (gid >= count) return;                                                                      \
    thread wrapped_f scratch[64] = {};                                                             \
    DelegationTrace oracle{};                                                                      \
    oracle.num_requests = num_requests;                                                            \
    oracle.num_register_accesses_per_delegation = num_register_accesses;                           \
    oracle.num_indirect_reads_per_delegation = num_indirect_reads;                                 \
    oracle.num_indirect_writes_per_delegation = num_indirect_writes;                               \
    oracle.base_register_index = base_register_index;                                              \
    oracle.delegation_type = delegation_type_val;                                                  \
    for (uint i = 0; i < MAX_INDIRECT_ACCESS_REGISTERS; i++)                                      \
        for (uint j = 0; j < MAX_INDIRECT_ACCESS_WORDS; j++)                                      \
            oracle.indirect_accesses_properties[i][j] =                                            \
                indirect_accesses_props[i * MAX_INDIRECT_ACCESS_WORDS + j];                        \
    oracle.write_timestamp = write_ts_data;                                                        \
    oracle.register_accesses = reg_acc_data;                                                       \
    oracle.indirect_reads = ind_read_data;                                                         \
    oracle.indirect_writes = ind_write_data;                                                       \
    const WitnessProxy<DelegationTrace> p = {oracle, generic_lookup_tables, memory_data,           \
                                              witness, lookup_mappings, scratch, stride, gid};      \
    FN_CALL(generate)                                                                              \
}

} // namespace generation
} // namespace witness
} // namespace airbender
