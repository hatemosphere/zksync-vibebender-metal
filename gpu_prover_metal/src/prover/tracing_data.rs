use super::context::ProverContext;
use crate::circuit_type::CircuitType;
use crate::metal_runtime::{MetalBuffer, MetalResult};
use crate::witness::trace_delegation::{DelegationTraceDevice, DelegationTraceHost};
use crate::witness::trace_main::{
    MainTraceDevice, MainTraceHost, ShuffleRamSetupAndTeardownDevice,
};
use fft::GoodAllocator;

pub enum TracingDataDevice {
    Main {
        setup_and_teardown: ShuffleRamSetupAndTeardownDevice,
        trace: MainTraceDevice,
    },
    Delegation(DelegationTraceDevice),
}

#[derive(Clone)]
pub enum TracingDataHost<A: GoodAllocator> {
    Main {
        setup_and_teardown: Option<crate::witness::trace_main::ShuffleRamSetupAndTeardownHost<A>>,
        trace: MainTraceHost<A>,
    },
    Delegation(DelegationTraceHost<A>),
}

pub struct TracingDataTransfer<'a, A: GoodAllocator> {
    pub circuit_type: CircuitType,
    pub data_host: TracingDataHost<A>,
    pub data_device: TracingDataDevice,
    pub _lifetime: std::marker::PhantomData<&'a ()>,
}

impl<'a, A: GoodAllocator + 'a> TracingDataTransfer<'a, A> {
    pub fn new(
        circuit_type: CircuitType,
        data_host: TracingDataHost<A>,
        context: &ProverContext,
    ) -> MetalResult<Self> {
        let device = context.device();

        let data_device = match &data_host {
            TracingDataHost::Main {
                setup_and_teardown: _,
                trace,
            } => {
                let len = trace.cycle_data.len();
                let lazy_init_data = MetalBuffer::alloc(device, len)?;
                let setup_and_teardown = ShuffleRamSetupAndTeardownDevice { lazy_init_data };
                let cycle_data = MetalBuffer::alloc(device, len)?;
                let trace = MainTraceDevice { cycle_data };
                TracingDataDevice::Main {
                    setup_and_teardown,
                    trace,
                }
            }
            TracingDataHost::Delegation(trace) => {
                let d_write_timestamp = MetalBuffer::alloc(device, trace.write_timestamp.len())?;
                let d_register_accesses =
                    MetalBuffer::alloc(device, trace.register_accesses.len())?;
                let d_indirect_reads = MetalBuffer::alloc(device, trace.indirect_reads.len())?;
                let d_indirect_writes = MetalBuffer::alloc(device, trace.indirect_writes.len())?;
                let trace_dev = DelegationTraceDevice {
                    num_requests: trace.num_requests,
                    num_register_accesses_per_delegation: trace
                        .num_register_accesses_per_delegation,
                    num_indirect_reads_per_delegation: trace.num_indirect_reads_per_delegation,
                    num_indirect_writes_per_delegation: trace.num_indirect_writes_per_delegation,
                    base_register_index: trace.base_register_index,
                    delegation_type: trace.delegation_type,
                    indirect_accesses_properties: trace.indirect_accesses_properties.clone(),
                    write_timestamp: d_write_timestamp,
                    register_accesses: d_register_accesses,
                    indirect_reads: d_indirect_reads,
                    indirect_writes: d_indirect_writes,
                };
                TracingDataDevice::Delegation(trace_dev)
            }
        };

        Ok(Self {
            circuit_type,
            data_host,
            data_device,
            _lifetime: std::marker::PhantomData,
        })
    }

    /// Transfer host data to device buffers.
    /// On Metal with unified memory, this is a direct memcpy.
    pub fn schedule_transfer(&mut self, _context: &ProverContext) -> MetalResult<()> {
        match (&self.data_host, &mut self.data_device) {
            (
                TracingDataHost::Main {
                    setup_and_teardown: h_sat,
                    trace: h_trace,
                },
                TracingDataDevice::Main {
                    setup_and_teardown: d_sat,
                    trace: d_trace,
                },
            ) => {
                if let Some(h_sat) = h_sat {
                    unsafe {
                        d_sat.lazy_init_data.copy_from_slice(&h_sat.lazy_init_data);
                    }
                } else {
                    // Zero-init
                    unsafe {
                        std::ptr::write_bytes(
                            d_sat.lazy_init_data.as_mut_ptr(),
                            0,
                            d_sat.lazy_init_data.len(),
                        );
                    }
                }
                unsafe {
                    d_trace.cycle_data.copy_from_slice(&h_trace.cycle_data);
                }
            }
            (
                TracingDataHost::Delegation(h),
                TracingDataDevice::Delegation(d),
            ) => unsafe {
                d.write_timestamp.copy_from_slice(&h.write_timestamp);
                d.register_accesses.copy_from_slice(&h.register_accesses);
                d.indirect_reads.copy_from_slice(&h.indirect_reads);
                d.indirect_writes.copy_from_slice(&h.indirect_writes);
            },
            _ => panic!("TracingDataTransfer: host/device type mismatch"),
        }
        Ok(())
    }
}
