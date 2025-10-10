use super::gpu_worker::GpuWorkResult;
use crate::circuit_type::CircuitType;
use crate::prover::tracing_data::TracingDataHost;
use crate::witness::trace_unrolled::ShuffleRamInitsAndTeardownsHost;
use cs::definitions::TimestampScalar;
use fft::GoodAllocator;
use std::collections::HashSet;
use trace_and_split::FinalRegisterValue;

pub struct InitsAndTeardownsData<A: GoodAllocator> {
    pub circuit_type: CircuitType,
    pub sequence_id: usize,
    pub inits_and_teardowns: Option<ShuffleRamInitsAndTeardownsHost<A>>,
}

pub struct TracingData<A: GoodAllocator> {
    pub circuit_type: CircuitType,
    pub sequence_id: usize,
    pub tracing_data: TracingDataHost<A>,
    pub participating_snapshot_indexes: HashSet<usize>,
}

pub struct SimulationResult {
    pub final_register_values: [FinalRegisterValue; 32],
    pub final_pc: u32,
    pub final_timestamp: TimestampScalar,
    pub cycles_used: usize,
}

pub enum WorkerResult<A: GoodAllocator> {
    InitsAndTeardownsData(InitsAndTeardownsData<A>),
    TracingData(TracingData<A>),
    SimulationResult(SimulationResult),
    SnapshotReplayed(usize),
    GpuWorkResult(GpuWorkResult<A>),
}
