#include "layout.metal"
#include "memory.metal"
#include "option.metal"
#include "trace_main.metal"

using namespace airbender::witness::layout;
using namespace airbender::witness::memory_witness;
using namespace airbender::witness::option;
using namespace airbender::witness::trace::main;
using namespace airbender::witness::column;
using namespace airbender::witness::trace;
using namespace airbender::witness::ram_access;
using namespace airbender::memory;
using namespace airbender::field;

#define MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT 4

struct MainMemorySubtree {
  ShuffleRamInitAndTeardownLayout shuffle_ram_inits_and_teardowns;
  uint shuffle_ram_access_sets_count;
  ShuffleRamQueryColumns shuffle_ram_access_sets[MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT];
  Option<DelegationRequestLayout> delegation_request_layout;
};

struct MemoryQueriesTimestampComparisonAuxVars {
  uint addresses_count;
  ColumnAddress addresses[MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT];
};

template <bool COMPUTE_WITNESS>
DEVICE_FORCEINLINE void process_lazy_inits_and_teardowns(const constant MainMemorySubtree &subtree, const thread ShuffleRamSetupAndTeardown &setup_and_teardown,
                                                         const thread ShuffleRamAuxComparisonSet &lazy_init_address_aux_vars,
                                                         const matrix_setter<bf> mem_setter, const matrix_setter<bf> wit_setter,
                                                         const uint count, const uint index) {
  const device LazyInitAndTeardown *lazy_inits_and_teardowns = setup_and_teardown.lazy_init_data;
  const auto addresses_columns = subtree.shuffle_ram_inits_and_teardowns.lazy_init_addresses_columns;
  const auto values_columns = subtree.shuffle_ram_inits_and_teardowns.lazy_teardown_values_columns;
  const auto timestamps_columns = subtree.shuffle_ram_inits_and_teardowns.lazy_teardown_timestamps_columns;
  const auto init_data = lazy_inits_and_teardowns[index];
  const u32 init_address = init_data.address;
  const u32 teardown_value = init_data.teardown_value;
  const TimestampData teardown_timestamp = init_data.teardown_timestamp;
  write_u32_value(addresses_columns, init_address, mem_setter);
  write_u32_value(values_columns, teardown_value, mem_setter);
  write_timestamp_value(timestamps_columns, teardown_timestamp, mem_setter);
  if (!COMPUTE_WITNESS)
    return;
  u16 low_value;
  u16 high_value;
  bool intermediate_borrow_value;
  bool final_borrow_value;
  if (index == count - 1) {
    low_value = 0;
    high_value = 0;
    intermediate_borrow_value = false;
    final_borrow_value = true;
  } else {
    const u32 next_row_lazy_init_address_value = lazy_inits_and_teardowns[index + 1].address;
    const u16 a_low = static_cast<u16>(init_address & 0xFFFF);
    const u16 a_high = static_cast<u16>(init_address >> 16);
    const u16 b_low = static_cast<u16>(next_row_lazy_init_address_value & 0xFFFF);
    const u16 b_high = static_cast<u16>(next_row_lazy_init_address_value >> 16);
    const auto r0 = airbender::witness::sub_borrow(a_low, b_low);
    const auto r1 = airbender::witness::sub_borrow(a_high, b_high);
    const auto r2 = airbender::witness::sub_borrow(static_cast<u16>(r1.x), static_cast<u16>(r0.y));
    low_value = static_cast<u16>(r0.x);
    high_value = static_cast<u16>(r2.x);
    intermediate_borrow_value = r0.y;
    final_borrow_value = r1.y || r2.y;
  }
  const auto low_address = lazy_init_address_aux_vars.aux_low_high[0];
  const auto high_address = lazy_init_address_aux_vars.aux_low_high[1];
  write_u16_value(low_address, low_value, wit_setter);
  write_u16_value(high_address, high_value, wit_setter);
  write_bool_value(lazy_init_address_aux_vars.intermediate_borrow, intermediate_borrow_value, wit_setter);
  write_bool_value(lazy_init_address_aux_vars.final_borrow, final_borrow_value, wit_setter);
}

template <bool COMPUTE_WITNESS>
DEVICE_FORCEINLINE void
process_shuffle_ram_access_sets(const constant MainMemorySubtree &subtree, const thread MemoryQueriesTimestampComparisonAuxVars &memory_queries_timestamp_comparison_aux_vars,
                                const thread MainTrace &trace, const TimestampScalar timestamp_high_from_circuit_sequence,
                                const matrix_setter<bf> mem_setter, const matrix_setter<bf> wit_setter, const uint index) {
  for (uint i = 0; i < MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT; ++i) {
    if (i == subtree.shuffle_ram_access_sets_count)
      break;
    const auto tag = subtree.shuffle_ram_access_sets[i].tag;
    const auto payload = subtree.shuffle_ram_access_sets[i].payload;
    ShuffleRamAddressEnum address = {};
    ColumnSet<NUM_TIMESTAMP_COLUMNS_FOR_RAM> read_timestamp_columns = {};
    ColumnSet<REGISTER_SIZE> read_value_columns = {};
    switch (tag) {
    case Readonly: {
      auto columns = payload.shuffle_ram_query_read_columns;
      address = columns.address;
      read_timestamp_columns = columns.read_timestamp;
      read_value_columns = columns.read_value;
      break;
    }
    case Write: {
      const auto columns = payload.shuffle_ram_query_write_columns;
      address = columns.address;
      read_timestamp_columns = columns.read_timestamp;
      read_value_columns = columns.read_value;
      break;
    }
    }
    Placeholder ph{};
    ph.payload.u32_val = i;
    switch (address.tag) {
    case RegisterOnly: {
      const auto register_index_col = address.payload.register_only_access_address.register_index;
      ph.tag = ShuffleRamAddress;
      const u16 value = trace.get_witness_from_placeholder<u16>(ph, index);
      write_u16_value(register_index_col, value, mem_setter);
      break;
    }
    case RegisterOrRam: {
      const auto is_register_columns = address.payload.register_or_ram_access_address.is_register;
      const auto address_columns = address.payload.register_or_ram_access_address.address;
      ph.tag = ShuffleRamIsRegisterAccess;
      const bool is_register_value = trace.get_witness_from_placeholder<bool>(ph, index);
      write_bool_value(is_register_columns, is_register_value, mem_setter);
      ph.tag = ShuffleRamAddress;
      const u32 address_value = trace.get_witness_from_placeholder<u32>(ph, index);
      write_u32_value(address_columns, address_value, mem_setter);
      break;
    }
    }
    ph.tag = ShuffleRamReadTimestamp;
    const TimestampData read_timestamp_value = trace.get_witness_from_placeholder<TimestampData>(ph, index);
    write_timestamp_value(read_timestamp_columns, read_timestamp_value, mem_setter);
    ph.tag = ShuffleRamReadValue;
    const u32 read_value_value = trace.get_witness_from_placeholder<u32>(ph, index);
    write_u32_value(read_value_columns, read_value_value, mem_setter);
    if (tag == Write) {
      const auto write_value_columns = payload.shuffle_ram_query_write_columns.write_value;
      ph.tag = ShuffleRamWriteValue;
      const u32 write_value_value = trace.get_witness_from_placeholder<u32>(ph, index);
      write_u32_value(write_value_columns, write_value_value, mem_setter);
    }
    if (!COMPUTE_WITNESS)
      continue;
    const TimestampScalar write_timestamp_base =
        timestamp_high_from_circuit_sequence + (static_cast<TimestampScalar>(index + 1) << TimestampData::NUM_EMPTY_BITS_FOR_RAM_TIMESTAMP);
    const ColumnAddress borrow_address = memory_queries_timestamp_comparison_aux_vars.addresses[i];
    const u32 read_timestamp_low = read_timestamp_value.get_low();
    const TimestampData write_timestamp = TimestampData::from_scalar(write_timestamp_base + i);
    const u32 write_timestamp_low = write_timestamp.get_low();
    const bool intermediate_borrow = TimestampData::sub_borrow(read_timestamp_low, write_timestamp_low).y;
    write_bool_value(borrow_address, intermediate_borrow, wit_setter);
  }
}

DEVICE_FORCEINLINE void process_delegation_requests(const constant MainMemorySubtree &subtree, const thread MainTrace &trace, const matrix_setter<bf> mem_setter,
                                                    const uint index) {
  const auto dr = subtree.delegation_request_layout.value;
  Placeholder ph{};
  ph.tag = ExecuteDelegation;
  const bool execute_delegation_value = trace.get_witness_from_placeholder<bool>(ph, index);
  write_bool_value(dr.multiplicity, execute_delegation_value, mem_setter);
  ph.tag = DelegationType;
  const u16 delegation_type_value = trace.get_witness_from_placeholder<u16>(ph, index);
  write_u16_value(dr.delegation_type, delegation_type_value, mem_setter);
  ph.tag = DelegationABIOffset;
  const u16 abi_mem_offset_high_value = trace.get_witness_from_placeholder<u16>(ph, index);
  write_u16_value(dr.abi_mem_offset_high, abi_mem_offset_high_value, mem_setter);
}

kernel void ab_generate_memory_values_main_kernel(
    constant MainMemorySubtree &subtree [[buffer(0)]],
    const device LazyInitAndTeardown *lazy_init_data [[buffer(1)]],
    const device SingleCycleTracingData *cycle_data [[buffer(2)]],
    device bf *memory_ptr [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count)
    return;
  matrix_setter<bf> mem_setter{memory_ptr, stride};
  mem_setter.add_row(gid);
  ShuffleRamSetupAndTeardown setup_and_teardown{lazy_init_data};
  MainTrace trace{cycle_data};
  ShuffleRamAuxComparisonSet empty_aux{};
  MemoryQueriesTimestampComparisonAuxVars empty_mqtcav{};
  process_lazy_inits_and_teardowns<false>(subtree, setup_and_teardown, empty_aux, mem_setter, mem_setter, count, gid);
  process_shuffle_ram_access_sets<false>(subtree, empty_mqtcav, trace, 0, mem_setter, mem_setter, gid);
  if (subtree.delegation_request_layout.tag == Some)
    process_delegation_requests(subtree, trace, mem_setter, gid);
}

kernel void ab_generate_memory_and_witness_values_main_kernel(
    constant MainMemorySubtree &subtree [[buffer(0)]],
    constant MemoryQueriesTimestampComparisonAuxVars &memory_queries_timestamp_comparison_aux_vars [[buffer(1)]],
    const device LazyInitAndTeardown *lazy_init_data [[buffer(2)]],
    constant ShuffleRamAuxComparisonSet &lazy_init_address_aux_vars [[buffer(3)]],
    const device SingleCycleTracingData *cycle_data [[buffer(4)]],
    constant TimestampScalar &timestamp_high_from_circuit_sequence [[buffer(5)]],
    device bf *memory_ptr [[buffer(6)]],
    device bf *witness_ptr [[buffer(7)]],
    constant uint &stride [[buffer(8)]],
    constant uint &count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count)
    return;
  matrix_setter<bf> mem_setter{memory_ptr, stride};
  matrix_setter<bf> wit_setter{witness_ptr, stride};
  mem_setter.add_row(gid);
  wit_setter.add_row(gid);
  ShuffleRamSetupAndTeardown setup_and_teardown{lazy_init_data};
  MainTrace trace{cycle_data};
  ShuffleRamAuxComparisonSet local_lazy_init_aux = lazy_init_address_aux_vars;
  process_lazy_inits_and_teardowns<true>(subtree, setup_and_teardown, local_lazy_init_aux, mem_setter, wit_setter, count, gid);
  MemoryQueriesTimestampComparisonAuxVars local_mqtcav = memory_queries_timestamp_comparison_aux_vars;
  process_shuffle_ram_access_sets<true>(subtree, local_mqtcav, trace, timestamp_high_from_circuit_sequence, mem_setter, wit_setter, gid);
  if (subtree.delegation_request_layout.tag == Some)
    process_delegation_requests(subtree, trace, mem_setter, gid);
}
