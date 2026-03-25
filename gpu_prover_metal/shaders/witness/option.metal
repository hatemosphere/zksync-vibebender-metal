#pragma once

#include "common.metal"

namespace airbender {
namespace witness {
namespace option {

enum OptionTag : uint {
  None = 0,
  Some = 1,
};

template <typename T> struct Option {
  OptionTag tag;
  T value;
};

} // namespace option
} // namespace witness
} // namespace airbender
