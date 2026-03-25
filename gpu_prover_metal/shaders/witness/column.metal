#pragma once

#include "common.metal"

namespace airbender {
namespace witness {
namespace column {

#define REGISTER_SIZE 2
#define NUM_TIMESTAMP_COLUMNS_FOR_RAM 2

template <uint WIDTH> struct ColumnSet {
  uint offset;
  uint num_elements;
};

enum ColumnAddressTag : uint {
  WitnessSubtree,
  MemorySubtree,
  SetupSubtree,
  OptimizedOut,
};

struct ColumnAddress {
  ColumnAddressTag tag;
  uint offset;
};

} // namespace column
} // namespace witness
} // namespace airbender
