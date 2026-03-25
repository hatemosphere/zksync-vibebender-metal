use super::column::ColumnAddress;
use super::layout::{DelegationRequestLayout, ShuffleRamInitAndTeardownLayout};
use super::ram_access::{ShuffleRamAuxComparisonSet, ShuffleRamQueryColumns};
use super::trace_main::{MainTraceDevice, ShuffleRamSetupAndTeardownDevice};
use super::BF;
use crate::metal_runtime::command_queue::MetalCommandBuffer;
use crate::metal_runtime::dispatch::{dispatch_kernel, set_buffer, set_bytes, MetalLaunchConfig};
use crate::metal_runtime::error::MetalResult;
use crate::metal_runtime::MetalBuffer;
use cs::definitions::{MemorySubtree, TimestampScalar};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

const WARP_SIZE: u32 = 32;
const MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct MainMemorySubtree {
    shuffle_ram_inits_and_teardowns: ShuffleRamInitAndTeardownLayout,
    shuffle_ram_access_sets_count: u32,
    shuffle_ram_access_sets: [ShuffleRamQueryColumns; MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT],
    delegation_request_layout: super::option::Option<DelegationRequestLayout>,
}

impl From<&MemorySubtree> for MainMemorySubtree {
    fn from(value: &MemorySubtree) -> Self {
        assert!(value.delegation_processor_layout.is_none());
        assert!(value.batched_ram_accesses.is_empty());
        assert!(value.register_and_indirect_accesses.is_empty());
        let shuffle_ram_inits_and_teardowns = value.shuffle_ram_inits_and_teardowns.unwrap().into();
        let shuffle_ram_access_sets_count = value.shuffle_ram_access_sets.len() as u32;
        assert!(shuffle_ram_access_sets_count <= MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT as u32);
        let mut shuffle_ram_access_sets =
            [ShuffleRamQueryColumns::default(); MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT];
        for (i, value) in value.shuffle_ram_access_sets.iter().enumerate() {
            shuffle_ram_access_sets[i] = value.clone().into();
        }
        let delegation_request_layout = value.delegation_request_layout.into();
        Self {
            shuffle_ram_inits_and_teardowns,
            shuffle_ram_access_sets_count,
            shuffle_ram_access_sets,
            delegation_request_layout,
        }
    }
}

#[repr(C)]
struct MemoryQueriesTimestampComparisonAuxVars {
    addresses_count: u32,
    addresses: [ColumnAddress; MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT],
}

impl From<&[cs::definitions::ColumnAddress]> for MemoryQueriesTimestampComparisonAuxVars {
    fn from(value: &[cs::definitions::ColumnAddress]) -> Self {
        let len = value.len();
        assert!(len <= MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT);
        let mut addresses = [ColumnAddress::default(); MAX_SHUFFLE_RAM_ACCESS_SETS_COUNT];
        for (i, &address) in value.iter().enumerate() {
            addresses[i] = address.into();
        }
        Self {
            addresses_count: len as u32,
            addresses,
        }
    }
}

pub(crate) fn generate_memory_values_main(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    subtree: &MemorySubtree,
    setup_and_teardown: &ShuffleRamSetupAndTeardownDevice,
    trace: &MainTraceDevice,
    memory: &MetalBuffer<BF>,
    memory_stride: u32,
    count: u32,
) -> MetalResult<()> {
    let subtree: MainMemorySubtree = subtree.into();

    let block_dim = WARP_SIZE * 4;
    let grid_dim = (count + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_generate_memory_values_main_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            unsafe {
                set_bytes(encoder, idx, &subtree);
            }
            idx += 1;
            set_buffer(encoder, idx, setup_and_teardown.lazy_init_data.raw(), 0);
            idx += 1;
            set_buffer(encoder, idx, trace.cycle_data.raw(), 0);
            idx += 1;
            set_buffer(encoder, idx, memory.raw(), 0);
            idx += 1;
            unsafe {
                set_bytes(encoder, idx, &memory_stride);
            }
            idx += 1;
            unsafe {
                set_bytes(encoder, idx, &count);
            }
        },
    )
}

pub(crate) fn generate_memory_and_witness_values_main(
    device: &ProtocolObject<dyn MTLDevice>,
    cmd_buf: &MetalCommandBuffer,
    subtree: &MemorySubtree,
    memory_queries_timestamp_comparison_aux_vars: &[cs::definitions::ColumnAddress],
    setup_and_teardown: &ShuffleRamSetupAndTeardownDevice,
    lazy_init_address_aux_vars: &cs::definitions::ShuffleRamAuxComparisonSet,
    trace: &MainTraceDevice,
    timestamp_high_from_circuit_sequence: TimestampScalar,
    memory: &MetalBuffer<BF>,
    witness: &MetalBuffer<BF>,
    stride: u32,
    count: u32,
) -> MetalResult<()> {
    let subtree: MainMemorySubtree = subtree.into();
    let aux_vars: MemoryQueriesTimestampComparisonAuxVars =
        memory_queries_timestamp_comparison_aux_vars.into();
    let lazy_init_aux: ShuffleRamAuxComparisonSet = lazy_init_address_aux_vars.into();

    let block_dim = WARP_SIZE * 4;
    let grid_dim = (count + block_dim - 1) / block_dim;
    let config = MetalLaunchConfig::basic_1d(grid_dim, block_dim);

    dispatch_kernel(
        device,
        cmd_buf,
        "ab_generate_memory_and_witness_values_main_kernel",
        &config,
        |encoder| {
            let mut idx = 0u32;
            unsafe {
                set_bytes(encoder, idx, &subtree);
            }
            idx += 1;
            unsafe {
                set_bytes(encoder, idx, &aux_vars);
            }
            idx += 1;
            set_buffer(encoder, idx, setup_and_teardown.lazy_init_data.raw(), 0);
            idx += 1;
            unsafe {
                set_bytes(encoder, idx, &lazy_init_aux);
            }
            idx += 1;
            set_buffer(encoder, idx, trace.cycle_data.raw(), 0);
            idx += 1;
            unsafe {
                set_bytes(encoder, idx, &timestamp_high_from_circuit_sequence);
            }
            idx += 1;
            set_buffer(encoder, idx, memory.raw(), 0);
            idx += 1;
            set_buffer(encoder, idx, witness.raw(), 0);
            idx += 1;
            unsafe {
                set_bytes(encoder, idx, &stride);
            }
            idx += 1;
            unsafe {
                set_bytes(encoder, idx, &count);
            }
        },
    )
}
