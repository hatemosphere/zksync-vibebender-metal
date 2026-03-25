use super::trace_main::MainTraceDevice;
use super::BF;
use crate::circuit_type::MainCircuitType;
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

const WARP_SIZE: u32 = 32;

pub fn generate_witness_values_main(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    circuit_type: MainCircuitType,
    trace: &MainTraceDevice,
    generic_lookup_tables: &MetalBuffer<BF>,
    generic_lookup_tables_offset: usize,
    memory: &MetalBuffer<BF>,
    witness: &MetalBuffer<BF>,
    lookup_mapping: &MetalBuffer<u32>,
    stride: u32,
    count: u32,
) -> MetalResult<()> {
    let kernel_name = match circuit_type {
        MainCircuitType::FinalReducedRiscVMachine => {
            "ab_generate_final_reduced_risc_v_machine_witness_kernel"
        }
        MainCircuitType::MachineWithoutSignedMulDiv => {
            "ab_generate_machine_without_signed_mul_div_witness_kernel"
        }
        MainCircuitType::ReducedRiscVMachine => "ab_generate_reduced_risc_v_machine_witness_kernel",
        MainCircuitType::ReducedRiscVLog23Machine => {
            "ab_generate_reduced_risc_v_log_23_machine_witness_kernel"
        }
        MainCircuitType::RiscVCycles => "ab_generate_risc_v_cycles_witness_kernel",
    };

    let block_dim = WARP_SIZE * 4;
    let grid_dim = (count + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(device, cmd_buf, kernel_name, &config, |encoder| {
        // buffer(0) = oracle struct (MainTrace = {cycle_data_ptr})
        // The MSL kernel reads `device const MainTrace* oracle_ptr [[buffer(0)]]`
        // and dereferences it to get the cycle_data pointer.
        // We pass the cycle_data buffer directly and the MSL struct reads
        // the buffer's GPU address as a pointer.
        set_buffer(encoder, 0, trace.cycle_data.raw(), 0);
        set_buffer(encoder, 1, generic_lookup_tables.raw(), generic_lookup_tables_offset);
        set_buffer(encoder, 2, memory.raw(), 0);
        set_buffer(encoder, 3, witness.raw(), 0);
        set_buffer(encoder, 4, lookup_mapping.raw(), 0);
        unsafe {
            set_bytes(encoder, 5, &stride);
            set_bytes(encoder, 6, &count);
        }
    })
}
