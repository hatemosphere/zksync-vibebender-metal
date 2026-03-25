#pragma once

#include "common.metal"
#include "placeholder.metal"
#include "trace.metal"

using namespace ::airbender::witness;
using namespace ::airbender::witness::placeholder;
using namespace ::airbender::witness::trace;

namespace airbender {
namespace witness {
namespace trace {
namespace delegation {

struct RegisterOrIndirectReadData {
  u32 read_value;
  TimestampData timestamp;
};

struct RegisterOrIndirectReadWriteData {
  u32 read_value;
  u32 write_value;
  TimestampData timestamp;
};

#define MAX_INDIRECT_ACCESS_REGISTERS 2
#define MAX_INDIRECT_ACCESS_WORDS 24
#define USE_WRITES_MASK (1u << 31)

struct DelegationTrace {
  u32 num_requests;
  u32 num_register_accesses_per_delegation;
  u32 num_indirect_reads_per_delegation;
  u32 num_indirect_writes_per_delegation;
  u32 base_register_index;
  u16 delegation_type;
  u32 indirect_accesses_properties[MAX_INDIRECT_ACCESS_REGISTERS][MAX_INDIRECT_ACCESS_WORDS];
  const device TimestampData *write_timestamp;
  const device RegisterOrIndirectReadWriteData *register_accesses;
  const device RegisterOrIndirectReadData *indirect_reads;
  const device RegisterOrIndirectReadWriteData *indirect_writes;

  /// Construct from raw buffer pointer. For DelegationTrace, the buffer
  /// must contain the full struct layout (serialized from Rust side).
  static DEVICE_FORCEINLINE DelegationTrace from_buffer(device const void* buf) {
    return *reinterpret_cast<device const DelegationTrace*>(buf);
  }

  template <typename T> DEVICE_FORCEINLINE T get_witness_from_placeholder(Placeholder, uint) const;
};

// u32 specialization
template <> DEVICE_FORCEINLINE u32 DelegationTrace::get_witness_from_placeholder<u32>(const Placeholder placeholder, const uint trace_row) const {
  if (trace_row >= num_requests)
    return 0;
  const auto dp = placeholder.payload.delegation_payload;
  const uint register_index = dp.register_index;
  const uint word_index = dp.word_index;
  const uint register_offset = register_index - base_register_index;
  switch (placeholder.tag) {
  case DelegationRegisterReadValue: {
    const uint offset = trace_row * num_register_accesses_per_delegation + register_offset;
    return register_accesses[offset].read_value;
  }
  case DelegationRegisterWriteValue: {
    const uint offset = trace_row * num_register_accesses_per_delegation + register_offset;
    return register_accesses[offset].write_value;
  }
  case DelegationIndirectReadValue: {
    const u32 access = indirect_accesses_properties[register_offset][word_index];
    const bool use_writes = access & USE_WRITES_MASK;
    const u32 index = access & ~USE_WRITES_MASK;
    const u32 t = use_writes ? num_indirect_writes_per_delegation : num_indirect_reads_per_delegation;
    const uint offset = trace_row * t + index;
    return use_writes ? indirect_writes[offset].read_value : indirect_reads[offset].read_value;
  }
  case DelegationIndirectWriteValue: {
    const u32 access = indirect_accesses_properties[register_offset][word_index];
    const u32 index = access & ~USE_WRITES_MASK;
    const uint offset = trace_row * num_indirect_writes_per_delegation + index;
    return indirect_writes[offset].write_value;
  }
  default:
    return 0;
  }
}

// u16 specialization
template <> DEVICE_FORCEINLINE u16 DelegationTrace::get_witness_from_placeholder<u16>(const Placeholder placeholder, const uint trace_row) const {
  if (trace_row >= num_requests)
    return 0;
  switch (placeholder.tag) {
  case DelegationABIOffset:
    return 0;
  case DelegationType:
    return delegation_type;
  default:
    return 0;
  }
}

// bool specialization
template <> DEVICE_FORCEINLINE bool DelegationTrace::get_witness_from_placeholder<bool>(const Placeholder placeholder, const uint trace_row) const {
  if (trace_row >= num_requests)
    return false;
  switch (placeholder.tag) {
  case ExecuteDelegation:
    return true;
  default:
    return false;
  }
}

// TimestampData specialization
template <> DEVICE_FORCEINLINE TimestampData DelegationTrace::get_witness_from_placeholder<TimestampData>(const Placeholder placeholder, const uint trace_row) const {
  if (trace_row >= num_requests)
    return {};
  const auto dp = placeholder.payload.delegation_payload;
  const uint register_index = dp.register_index;
  const uint word_index = dp.word_index;
  switch (placeholder.tag) {
  case DelegationWriteTimestamp:
    return write_timestamp[trace_row];
  case DelegationRegisterReadTimestamp: {
    const uint register_offset = register_index - base_register_index;
    const uint offset = trace_row * num_register_accesses_per_delegation + register_offset;
    return register_accesses[offset].timestamp;
  }
  case DelegationIndirectReadTimestamp: {
    const uint register_offset = register_index - base_register_index;
    const u32 access = indirect_accesses_properties[register_offset][word_index];
    const bool use_writes = access & USE_WRITES_MASK;
    const u32 index = access & ~USE_WRITES_MASK;
    const u32 t = use_writes ? num_indirect_writes_per_delegation : num_indirect_reads_per_delegation;
    const uint offset = trace_row * t + index;
    return use_writes ? indirect_writes[offset].timestamp : indirect_reads[offset].timestamp;
  }
  default:
    return {};
  }
}

} // namespace delegation
} // namespace trace
} // namespace witness
} // namespace airbender
