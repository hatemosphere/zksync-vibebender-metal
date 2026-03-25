#include "../trace_main.metal"
#include "../witness_generation.metal"

using namespace airbender::witness::generation;
using namespace airbender::witness::trace::main;
using airbender::witness::trace::main::MainTrace;

#include CIRCUIT_INCLUDE(risc_v_cycles)

KERNEL(risc_v_cycles, MainTrace)
