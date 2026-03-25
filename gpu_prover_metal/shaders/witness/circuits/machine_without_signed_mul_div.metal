#include "../trace_main.metal"
#include "../witness_generation.metal"

using namespace airbender::witness::generation;
using namespace airbender::witness::trace::main;
using airbender::witness::trace::main::MainTrace;

#include CIRCUIT_INCLUDE(machine_without_signed_mul_div)

KERNEL(machine_without_signed_mul_div, MainTrace)
