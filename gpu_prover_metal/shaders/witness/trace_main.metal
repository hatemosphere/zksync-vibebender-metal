#pragma once

#include "placeholder.metal"
#include "trace.metal"

using namespace ::airbender::witness::placeholder;
using namespace ::airbender::witness::trace;

namespace airbender {
namespace witness {
namespace trace {
namespace main {

struct SingleCycleTracingData {
  u32 pc;
  u32 rs1_read_value;
  TimestampData rs1_read_timestamp;
  u16 rs1_reg_idx;
  // 16
  u32 rs2_or_mem_word_read_value;
  RegIndexOrMemWordIndex rs2_or_mem_word_address;
  TimestampData rs2_or_mem_read_timestamp;
  u16 delegation_request;
  // 32
  u32 rd_or_mem_word_read_value;
  u32 rd_or_mem_word_write_value;
  RegIndexOrMemWordIndex rd_or_mem_word_address;
  TimestampData rd_or_mem_read_timestamp;
  // 52
  u32 non_determinism_read;
};

struct MainTrace {
  const device SingleCycleTracingData *cycle_data;

  /// Construct from raw buffer pointer (used by KERNEL macro).
  static DEVICE_FORCEINLINE MainTrace from_buffer(device const void* buf) {
    return MainTrace{reinterpret_cast<const device SingleCycleTracingData*>(buf)};
  }

  template <typename T> DEVICE_FORCEINLINE T get_witness_from_placeholder(Placeholder, uint) const;
};

// u32 specialization
template <> DEVICE_FORCEINLINE u32 MainTrace::get_witness_from_placeholder<u32>(const Placeholder placeholder, const uint trace_step) const {
  const device SingleCycleTracingData *data = &cycle_data[trace_step];
  switch (placeholder.tag) {
  case PcInit:
    return data->pc;
  case SecondRegMem:
    return data->rs2_or_mem_word_read_value;
  case WriteRdReadSetWitness:
    return data->rd_or_mem_word_read_value;
  case MemSlot: {
    RegIndexOrMemWordIndex rs2_v = data->rs2_or_mem_word_address;
    RegIndexOrMemWordIndex rd_v = data->rd_or_mem_word_address;
    const auto rs2_or_mem_address_is_register = rs2_v.is_register();
    const auto rd_or_mem_address_is_register = rd_v.is_register();
    if (!rs2_or_mem_address_is_register && rd_or_mem_address_is_register)
      return data->rs2_or_mem_word_read_value;
    if (rs2_or_mem_address_is_register && !rd_or_mem_address_is_register)
      return data->rd_or_mem_word_read_value;
    return 0;
  }
  case ShuffleRamAddress: {
    switch (placeholder.payload.u32_val) {
    case 0: return data->rs1_reg_idx;
    case 1: { RegIndexOrMemWordIndex v = data->rs2_or_mem_word_address; return v.as_u32_formal_address(); }
    case 2: { RegIndexOrMemWordIndex v = data->rd_or_mem_word_address; return v.as_u32_formal_address(); }
    default: return 0;
    }
  }
  case ShuffleRamReadValue: {
    switch (placeholder.payload.u32_val) {
    case 0: return data->rs1_read_value;
    case 1: return data->rs2_or_mem_word_read_value;
    case 2: return data->rd_or_mem_word_read_value;
    default: return 0;
    }
  }
  case ShuffleRamWriteValue: {
    switch (placeholder.payload.u32_val) {
    case 0: return data->rs1_read_value;
    case 1: return data->rs2_or_mem_word_read_value;
    case 2: return data->rd_or_mem_word_write_value;
    default: return 0;
    }
  }
  case ExternalOracle:
    return data->non_determinism_read;
  default:
    return 0;
  }
}

// u16 specialization
template <> DEVICE_FORCEINLINE u16 MainTrace::get_witness_from_placeholder<u16>(const Placeholder placeholder, const uint trace_step) const {
  const device SingleCycleTracingData *data = &cycle_data[trace_step];
  switch (placeholder.tag) {
  case DelegationABIOffset:
    return 0;
  case DelegationType:
    return data->delegation_request;
  case ShuffleRamAddress: {
    switch (placeholder.payload.u32_val) {
    case 0: return data->rs1_reg_idx;
    case 1: { RegIndexOrMemWordIndex v = data->rs2_or_mem_word_address; return static_cast<u16>(v.as_u32_formal_address()); }
    case 2: { RegIndexOrMemWordIndex v = data->rd_or_mem_word_address; return static_cast<u16>(v.as_u32_formal_address()); }
    default: return 0;
    }
  }
  case ExecuteDelegation:
    return data->delegation_request != 0;
  default:
    return 0;
  }
}

// bool specialization
template <> DEVICE_FORCEINLINE bool MainTrace::get_witness_from_placeholder<bool>(const Placeholder placeholder, const uint trace_step) const {
  const device SingleCycleTracingData *data = &cycle_data[trace_step];
  switch (placeholder.tag) {
  case ShuffleRamIsRegisterAccess:
    switch (placeholder.payload.u32_val) {
    case 0: return true;
    case 1: { RegIndexOrMemWordIndex v = data->rs2_or_mem_word_address; return v.is_register(); }
    case 2: { RegIndexOrMemWordIndex v = data->rd_or_mem_word_address; return v.is_register(); }
    default: return false;
    }
  case ExecuteDelegation:
    return data->delegation_request != 0;
  default:
    return false;
  }
}

// TimestampData specialization
template <> DEVICE_FORCEINLINE TimestampData MainTrace::get_witness_from_placeholder<TimestampData>(const Placeholder placeholder, const uint trace_step) const {
  const device SingleCycleTracingData *data = &cycle_data[trace_step];
  switch (placeholder.tag) {
  case ShuffleRamReadTimestamp:
    switch (placeholder.payload.u32_val) {
    case 0: return data->rs1_read_timestamp;
    case 1: return data->rs2_or_mem_read_timestamp;
    case 2: return data->rd_or_mem_read_timestamp;
    default: return {};
    }
  default:
    return {};
  }
}

struct LazyInitAndTeardown {
  u32 address;
  u32 teardown_value;
  TimestampData teardown_timestamp;
};

struct ShuffleRamSetupAndTeardown {
  const device LazyInitAndTeardown *lazy_init_data;
};

} // namespace main
} // namespace trace
} // namespace witness
} // namespace airbender
