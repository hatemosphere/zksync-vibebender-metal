#include "../trace_delegation.metal"
#include "../witness_generation.metal"

using namespace airbender::witness::generation;
using namespace airbender::witness::trace::delegation;
using airbender::witness::trace::delegation::DelegationTrace;

#include CIRCUIT_INCLUDE(blake2_with_compression)

DELEGATION_KERNEL(blake2_with_compression)
