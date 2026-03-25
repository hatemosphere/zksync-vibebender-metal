#include "../trace_delegation.metal"
#include "../witness_generation.metal"

using namespace airbender::witness::generation;
using namespace airbender::witness::trace::delegation;
using airbender::witness::trace::delegation::DelegationTrace;

#include CIRCUIT_INCLUDE(bigint_with_control)

DELEGATION_KERNEL(bigint_with_control)
