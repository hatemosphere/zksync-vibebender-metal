#include "layout.metal"
#include "memory.metal"
#include "trace_delegation.metal"

using namespace airbender::witness::layout;
using namespace airbender::witness::memory_witness;
using namespace airbender::witness::trace::delegation;
using namespace airbender::witness::column;
using namespace airbender::witness::trace;
using namespace airbender::witness::ram_access;
using namespace airbender::witness::placeholder;
using namespace airbender::memory;
using namespace airbender::field;

#define MAX_REGISTER_AND_INDIRECT_ACCESSES_COUNT_DELEG 4

struct DelegationMemorySubtree {
  DelegationProcessingLayout delegation_processor_layout;
  uint register_and_indirect_accesses_count;
  RegisterAndIndirectAccessDescription register_and_indirect_accesses[MAX_REGISTER_AND_INDIRECT_ACCESSES_COUNT_DELEG];
};

DEVICE_FORCEINLINE void process_delegation_requests_execution(const thread DelegationMemorySubtree &subtree, const thread DelegationTrace &trace,
                                                              const matrix_setter<bf> mem_setter, const uint index) {
  const auto multiplicity = subtree.delegation_processor_layout.multiplicity;
  const auto abi_mem_offset_high_column = subtree.delegation_processor_layout.abi_mem_offset_high;
  const auto write_timestamp_columns = subtree.delegation_processor_layout.write_timestamp;
  Placeholder ph{};
  ph.tag = ExecuteDelegation;
  const bool execute_delegation_value = trace.get_witness_from_placeholder<bool>(ph, index);
  write_bool_value(multiplicity, execute_delegation_value, mem_setter);
  ph.tag = DelegationABIOffset;
  const u16 abi_mem_offset_high_value = trace.get_witness_from_placeholder<u16>(ph, index);
  write_u16_value(abi_mem_offset_high_column, abi_mem_offset_high_value, mem_setter);
  ph.tag = DelegationWriteTimestamp;
  const TimestampData delegation_write_timestamp_value = trace.get_witness_from_placeholder<TimestampData>(ph, index);
  write_timestamp_value(write_timestamp_columns, delegation_write_timestamp_value, mem_setter);
}

template <bool COMPUTE_WITNESS>
DEVICE_FORCEINLINE void process_indirect_memory_accesses(const thread DelegationMemorySubtree &subtree,
                                                         const thread RegisterAndIndirectAccessTimestampComparisonAuxVars &aux_vars, const thread DelegationTrace &trace,
                                                         const matrix_setter<bf> mem_setter, const matrix_setter<bf> wit_setter,
                                                         const uint index) {
  Placeholder ph{};
  ph.tag = DelegationWriteTimestamp;
  const TimestampData write_timestamp = COMPUTE_WITNESS ? trace.get_witness_from_placeholder<TimestampData>(ph, index) : TimestampData{};
  for (uint i = 0; i < MAX_REGISTER_AND_INDIRECT_ACCESSES_COUNT_DELEG; ++i) {
    if (i == subtree.register_and_indirect_accesses_count)
      break;
    const auto register_tag = subtree.register_and_indirect_accesses[i].register_access.tag;
    const auto register_payload = subtree.register_and_indirect_accesses[i].register_access.payload;
    uint register_index = 0;
    ColumnSet<NUM_TIMESTAMP_COLUMNS_FOR_RAM> register_read_timestamp_columns = {};
    ColumnSet<REGISTER_SIZE> register_read_value_columns = {};
    switch (register_tag) {
    case RegisterReadAccess: {
      const auto acc = register_payload.register_access_columns_read_access;
      register_index = acc.register_index;
      register_read_timestamp_columns = acc.read_timestamp;
      register_read_value_columns = acc.read_value;
      break;
    }
    case RegisterWriteAccess: {
      const auto acc = register_payload.register_access_columns_write_access;
      register_index = acc.register_index;
      register_read_timestamp_columns = acc.read_timestamp;
      register_read_value_columns = acc.read_value;
      break;
    }
    }
    ph.tag = DelegationRegisterReadTimestamp;
    ph.payload.delegation_payload = {register_index, 0};
    const TimestampData register_read_timestamp_value = trace.get_witness_from_placeholder<TimestampData>(ph, index);
    write_timestamp_value(register_read_timestamp_columns, register_read_timestamp_value, mem_setter);
    ph.tag = DelegationRegisterReadValue;
    ph.payload.delegation_payload = {register_index, 0};
    const u32 register_read_value = trace.get_witness_from_placeholder<u32>(ph, index);
    write_u32_value(register_read_value_columns, register_read_value, mem_setter);
    if (register_tag == RegisterWriteAccess) {
      const auto register_write_access_columns = register_payload.register_access_columns_write_access.write_value;
      ph.tag = DelegationRegisterWriteValue;
      ph.payload.delegation_payload = {register_index, 0};
      const u32 register_write_value = trace.get_witness_from_placeholder<u32>(ph, index);
      write_u32_value(register_write_access_columns, register_write_value, mem_setter);
    }
    if (COMPUTE_WITNESS) {
      const auto borrow_address = aux_vars.aux_borrow_sets[i].borrow;
      const u32 read_timestamp_low = register_read_timestamp_value.get_low();
      const u32 write_timestamp_low = write_timestamp.get_low();
      const bool intermediate_borrow = TimestampData::sub_borrow(read_timestamp_low, write_timestamp_low).y;
      write_bool_value(borrow_address, intermediate_borrow, wit_setter);
    }
    const u32 base_address = register_read_value;
    const auto indirect_accesses_count = subtree.register_and_indirect_accesses[i].indirect_accesses_count;
    for (uint access_index = 0; access_index < MAX_INDIRECT_ACCESS_DESCRIPTION_INDIRECT_ACCESSES_COUNT; ++access_index) {
      if (access_index == indirect_accesses_count)
        break;
      const auto indirect_tag = subtree.register_and_indirect_accesses[i].indirect_accesses[access_index].tag;
      const auto indirect_payload = subtree.register_and_indirect_accesses[i].indirect_accesses[access_index].payload;
      ColumnSet<NUM_TIMESTAMP_COLUMNS_FOR_RAM> read_timestamp_columns = {};
      ColumnSet<REGISTER_SIZE> read_value_columns = {};
      ColumnSet<1> address_derivation_carry_bit_column = {};
      switch (indirect_tag) {
      case IndirectReadAccess: {
        const auto acc = indirect_payload.indirect_access_columns_read_access;
        read_timestamp_columns = acc.read_timestamp;
        read_value_columns = acc.read_value;
        address_derivation_carry_bit_column = acc.address_derivation_carry_bit;
        break;
      }
      case IndirectWriteAccess: {
        const auto acc = indirect_payload.indirect_access_columns_write_access;
        read_timestamp_columns = acc.read_timestamp;
        read_value_columns = acc.read_value;
        address_derivation_carry_bit_column = acc.address_derivation_carry_bit;
        break;
      }
      }
      ph.tag = DelegationIndirectReadTimestamp;
      ph.payload.delegation_payload = {register_index, access_index};
      const TimestampData read_timestamp_value = trace.get_witness_from_placeholder<TimestampData>(ph, index);
      write_timestamp_value(read_timestamp_columns, read_timestamp_value, mem_setter);
      ph.tag = DelegationIndirectReadValue;
      ph.payload.delegation_payload = {register_index, access_index};
      const u32 read_value_value = trace.get_witness_from_placeholder<u32>(ph, index);
      write_u32_value(read_value_columns, read_value_value, mem_setter);
      if (indirect_tag == IndirectWriteAccess) {
        ph.tag = DelegationIndirectWriteValue;
        ph.payload.delegation_payload = {register_index, access_index};
        const u32 write_value_value = trace.get_witness_from_placeholder<u32>(ph, index);
        const auto write_value_columns = indirect_payload.indirect_access_columns_write_access.write_value;
        write_u32_value(write_value_columns, write_value_value, mem_setter);
      }
      if (access_index != 0 && address_derivation_carry_bit_column.num_elements != 0) {
        const u32 derived_address = base_address + access_index * sizeof(u32);
        const bool carry_bit = (derived_address >> 16) != (base_address >> 16);
        write_u16_value(address_derivation_carry_bit_column, carry_bit, mem_setter);
      }
      if (!COMPUTE_WITNESS)
        continue;
      const auto borrow_addr = aux_vars.aux_borrow_sets[i].indirects[access_index];
      const u32 read_ts_low = read_timestamp_value.get_low();
      const u32 write_ts_low = write_timestamp.get_low();
      const bool indirect_borrow = TimestampData::sub_borrow(read_ts_low, write_ts_low).y;
      write_bool_value(borrow_addr, indirect_borrow, wit_setter);
    }
  }
}

kernel void ab_generate_memory_values_delegation_kernel(
    constant DelegationMemorySubtree &subtree_const [[buffer(0)]],
    constant uint &num_requests [[buffer(1)]],
    constant uint &num_register_accesses [[buffer(2)]],
    constant uint &num_indirect_reads [[buffer(3)]],
    constant uint &num_indirect_writes [[buffer(4)]],
    constant uint &base_register_index [[buffer(5)]],
    constant u16 &delegation_type [[buffer(6)]],
    constant uint *indirect_accesses_properties [[buffer(7)]],
    const device TimestampData *write_timestamp_data [[buffer(8)]],
    const device RegisterOrIndirectReadWriteData *register_accesses_data [[buffer(9)]],
    const device RegisterOrIndirectReadData *indirect_reads_data [[buffer(10)]],
    const device RegisterOrIndirectReadWriteData *indirect_writes_data [[buffer(11)]],
    device bf *memory_ptr [[buffer(12)]],
    constant uint &stride [[buffer(13)]],
    constant uint &count [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count)
    return;
  matrix_setter<bf> mem_setter{memory_ptr, stride};
  mem_setter.add_row(gid);
  DelegationTrace trace{};
  trace.num_requests = num_requests;
  trace.num_register_accesses_per_delegation = num_register_accesses;
  trace.num_indirect_reads_per_delegation = num_indirect_reads;
  trace.num_indirect_writes_per_delegation = num_indirect_writes;
  trace.base_register_index = base_register_index;
  trace.delegation_type = delegation_type;
  for (uint i = 0; i < MAX_INDIRECT_ACCESS_REGISTERS; i++)
    for (uint j = 0; j < MAX_INDIRECT_ACCESS_WORDS; j++)
      trace.indirect_accesses_properties[i][j] = indirect_accesses_properties[i * MAX_INDIRECT_ACCESS_WORDS + j];
  trace.write_timestamp = write_timestamp_data;
  trace.register_accesses = register_accesses_data;
  trace.indirect_reads = indirect_reads_data;
  trace.indirect_writes = indirect_writes_data;
  // Copy constant data to thread-local for helper function calls
  DelegationMemorySubtree subtree = subtree_const;
  RegisterAndIndirectAccessTimestampComparisonAuxVars empty_aux{};
  process_delegation_requests_execution(subtree, trace, mem_setter, gid);
  process_indirect_memory_accesses<false>(subtree, empty_aux, trace, mem_setter, mem_setter, gid);
}

kernel void ab_generate_memory_and_witness_values_delegation_kernel(
    constant DelegationMemorySubtree &subtree_const [[buffer(0)]],
    constant RegisterAndIndirectAccessTimestampComparisonAuxVars &aux_vars_const [[buffer(1)]],
    constant uint &num_requests [[buffer(2)]],
    constant uint &num_register_accesses [[buffer(3)]],
    constant uint &num_indirect_reads [[buffer(4)]],
    constant uint &num_indirect_writes [[buffer(5)]],
    constant uint &base_register_index [[buffer(6)]],
    constant u16 &delegation_type [[buffer(7)]],
    constant uint *indirect_accesses_properties [[buffer(8)]],
    const device TimestampData *write_timestamp_data [[buffer(9)]],
    const device RegisterOrIndirectReadWriteData *register_accesses_data [[buffer(10)]],
    const device RegisterOrIndirectReadData *indirect_reads_data [[buffer(11)]],
    const device RegisterOrIndirectReadWriteData *indirect_writes_data [[buffer(12)]],
    device bf *memory_ptr [[buffer(13)]],
    device bf *witness_ptr [[buffer(14)]],
    constant uint &stride [[buffer(15)]],
    constant uint &count [[buffer(16)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count)
    return;
  matrix_setter<bf> mem_setter{memory_ptr, stride};
  matrix_setter<bf> wit_setter{witness_ptr, stride};
  mem_setter.add_row(gid);
  wit_setter.add_row(gid);
  DelegationTrace trace{};
  trace.num_requests = num_requests;
  trace.num_register_accesses_per_delegation = num_register_accesses;
  trace.num_indirect_reads_per_delegation = num_indirect_reads;
  trace.num_indirect_writes_per_delegation = num_indirect_writes;
  trace.base_register_index = base_register_index;
  trace.delegation_type = delegation_type;
  for (uint i = 0; i < MAX_INDIRECT_ACCESS_REGISTERS; i++)
    for (uint j = 0; j < MAX_INDIRECT_ACCESS_WORDS; j++)
      trace.indirect_accesses_properties[i][j] = indirect_accesses_properties[i * MAX_INDIRECT_ACCESS_WORDS + j];
  trace.write_timestamp = write_timestamp_data;
  trace.register_accesses = register_accesses_data;
  trace.indirect_reads = indirect_reads_data;
  trace.indirect_writes = indirect_writes_data;
  // Copy constant data to thread-local for helper function calls
  DelegationMemorySubtree subtree = subtree_const;
  RegisterAndIndirectAccessTimestampComparisonAuxVars aux_vars = aux_vars_const;
  process_delegation_requests_execution(subtree, trace, mem_setter, gid);
  process_indirect_memory_accesses<true>(subtree, aux_vars, trace, mem_setter, wit_setter, gid);
}
