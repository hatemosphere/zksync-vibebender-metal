#include "../trace_main.metal"
#include "../witness_generation.metal"

using namespace airbender::witness::generation;
using namespace airbender::witness::trace::main;
using airbender::witness::trace::main::MainTrace;

#include CIRCUIT_INCLUDE(reduced_risc_v_machine)

KERNEL(reduced_risc_v_machine, MainTrace)
