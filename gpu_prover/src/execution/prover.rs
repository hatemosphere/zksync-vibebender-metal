use super::gpu_manager::{GpuManager, GpuWorkBatch};
use super::precomputations::{get_common_precomputations, CircuitPrecomputations};
use super::A;
use crate::circuit_type::{
    CircuitType, UnrolledCircuitType, UnrolledMemoryCircuitType, UnrolledNonMemoryCircuitType,
};
use crate::cudart::device::get_device_count;
use crate::cudart::memory::{CudaHostAllocFlags, HostAllocation};
use crate::execution::cpu_worker::{
    run_split_replayer, run_split_simulator, run_unified_replayer, run_unified_simulator,
    NonDeterminism,
};
use crate::execution::gpu_worker::{
    GpuWorkRequest, GpuWorkResult, MemoryCommitmentRequest, MemoryCommitmentResult, ProofRequest,
    ProofResult,
};
use crate::execution::messages::{
    InitsAndTeardownsData, SimulationResult, TracingData, WorkerResult,
};
use crate::machine_type::MachineType;
use crate::prover::context::ProverContextConfig;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use crossbeam_utils::sync::WaitGroup;
use cs::definitions::TimestampScalar;
use itertools::Itertools;
use log::{debug, info, trace};
use prover::definitions::{AuxArgumentsBoundaryValues, ExternalChallenges, ExternalValues};
use prover::merkle_trees::MerkleTreeCapVarLength;
use prover::prover_stages::unrolled_prover::UnrolledModeProof;
use prover::prover_stages::Proof;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use riscv_transpiler::ir::{
    decode, FullMachineDecoderConfig, FullUnsignedMachineDecoderConfig, ReducedMachineDecoderConfig,
};
use riscv_transpiler::vm::SimpleTape;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Instant;
use trace_and_split::{
    fs_transform_for_memory_and_delegation_arguments_for_unrolled_circuits, FinalRegisterValue,
};
use worker::Worker;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExecutionKind {
    Unrolled,
    Unified,
}

// const CPU_WORKERS_COUNT: usize = 6;
// const CYCLES_TRACING_WORKERS_COUNT: usize = CPU_WORKERS_COUNT - 2;
// const CACHE_DELEGATIONS: bool = false;

/// Represents an executable binary that can be proven by the prover
///
///  # Fields
/// * `key`: unique identifier for the binary, can be for example a &str or usize, anything that implements Clone, Debug, Eq, and Hash
/// * `circuit_type`: the type of the circuit this binary is for, one of the values from the `MainCircuitType` enumeration
/// * `bytecode`: the bytecode of the binary, can be a Vec<u32> or any other type that can be converted into Box<[u32]>
///
// #[derive(Clone)]
// pub struct ExecutableBinary<K: Clone + Debug + Eq + Hash, B: Into<Box<[u32]>>> {
//     pub key: K,
//     pub circuit_type: MainCircuitType,
//     pub bytecode: B,
// }

struct BinaryHolder {
    execution_kind: ExecutionKind,
    machine_type: MachineType,
    binary_image: Arc<Box<[u32]>>,
    instruction_tape: Arc<SimpleTape>,
    precomputations: HashMap<UnrolledCircuitType, CircuitPrecomputations>,
}

#[derive(Clone, Copy, Debug)]
pub struct ExecutionProverConfiguration {
    pub prover_context_config: ProverContextConfig,
    pub host_allocator_backing_allocation_size: usize,
    pub replay_worker_threads_count: usize,
    pub host_allocators_per_worker_count: usize,
    pub host_allocators_per_device_count: usize,
}

impl Default for ExecutionProverConfiguration {
    fn default() -> Self {
        Self {
            prover_context_config: Default::default(),
            host_allocator_backing_allocation_size: 1 << 26, // 64 MB
            replay_worker_threads_count: 4,
            host_allocators_per_worker_count: 16, // 1 GB
            host_allocators_per_device_count: 64, // 4 GB
        }
    }
}

pub struct CommitMemoryResult {
    pub final_register_values: [FinalRegisterValue; 32],
    pub final_pc: u32,
    pub final_timestamp: TimestampScalar,
    pub circuit_families_memory_caps: BTreeMap<u8, Vec<Vec<MerkleTreeCapVarLength>>>,
    pub inits_and_teardowns_memory_caps: Vec<Vec<MerkleTreeCapVarLength>>,
    pub delegation_circuits_memory_caps: BTreeMap<u32, Vec<Vec<MerkleTreeCapVarLength>>>,
}

pub struct ProveResult {
    pub register_final_values: [FinalRegisterValue; 32],
    pub final_pc: u32,
    pub final_timestamp: TimestampScalar,
    pub circuit_families_proofs: BTreeMap<u8, Vec<UnrolledModeProof>>,
    pub inits_and_teardowns_proofs: Vec<UnrolledModeProof>,
    pub delegation_proofs: BTreeMap<u32, Vec<Proof>>,
}

enum ExecutionProverResult {
    CommitMemory(CommitMemoryResult),
    Prove(ProveResult),
}

impl ExecutionProverResult {
    pub fn into_memory_commitment_result(self) -> CommitMemoryResult {
        match self {
            ExecutionProverResult::CommitMemory(result) => result,
            _ => panic!("expected CommitMemoryResult"),
        }
    }

    pub fn into_proof_result(self) -> ProveResult {
        match self {
            ExecutionProverResult::Prove(result) => result,
            _ => panic!("expected ProveResult"),
        }
    }
}

pub struct ExecutionProver {
    configuration: ExecutionProverConfiguration,
    gpu_manager: GpuManager,
    worker: Worker,
    binary_holders: BTreeMap<usize, BinaryHolder>,
    common_precomputations: BTreeMap<CircuitType, CircuitPrecomputations>,
    free_allocators_sender: Sender<A>,
    free_allocators_receiver: Receiver<A>,
}

impl ExecutionProver {
    pub fn new() -> Self {
        Self::with_configuration(ExecutionProverConfiguration::default())
    }
    ///  Creates a new instance of `ExecutionProver`.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent_batches`: maximum number of concurrent batches that the prover allocates host buffers for, this is a soft limit, the prover will work with more batches if needed, but it can stall certain operations for some time
    /// * `binaries`: a vector of executable binaries that the prover can work with, each binary must have a unique key
    ///
    /// returns: an instance of `ExecutionProver` that can be used to generate memory commitments and proofs for the provided binaries, it is supposed to be a Singleton instance
    ///
    pub fn with_configuration(configuration: ExecutionProverConfiguration) -> Self {
        let device_count = get_device_count().unwrap() as usize;
        assert_ne!(device_count, 0, "no CUDA capable devices found");
        let gpu_wait_group = WaitGroup::new();
        let gpu_manager =
            GpuManager::new(gpu_wait_group.clone(), configuration.prover_context_config);
        let worker = Worker::new_with_num_threads(16);
        info!(
            "PROVER thread pool with {} threads created",
            worker.num_cores
        );
        let binary_holders = BTreeMap::new();
        info!("PROVER generating common precomputations");
        let common_precomputations = get_common_precomputations(&worker);
        let host_allocators_count = configuration.replay_worker_threads_count
            * configuration.host_allocators_per_worker_count
            + device_count * configuration.host_allocators_per_device_count;
        let host_allocation_size = configuration.host_allocator_backing_allocation_size;
        let host_allocation_log_chunk_size = host_allocation_size.trailing_zeros();
        info!(
            "PROVER initializing {} host buffers with {} MB per buffer",
            host_allocators_count,
            host_allocation_size >> 20
        );
        let (free_allocators_sender, free_allocators_receiver) = unbounded();
        let free_allocators_sender_ref = &free_allocators_sender;
        (0..host_allocators_count).into_par_iter().for_each(|_| {
            let allocation =
                HostAllocation::alloc(host_allocation_size, CudaHostAllocFlags::DEFAULT).unwrap();
            let allocator = A::new([allocation], host_allocation_log_chunk_size);
            free_allocators_sender_ref.send(allocator).unwrap();
        });
        gpu_wait_group.wait();
        info!("PROVER initialized");
        Self {
            configuration,
            gpu_manager,
            worker,
            binary_holders,
            common_precomputations,
            free_allocators_sender,
            free_allocators_receiver,
        }
    }

    pub fn add_binary(
        &mut self,
        key: usize,
        execution_kind: ExecutionKind,
        machine_type: MachineType,
        mut binary_image: Vec<u32>,
        mut text_section: Vec<u32>,
    ) {
        setups::pad_bytecode_for_proving(&mut binary_image);
        let binary_image = Arc::new(binary_image.into_boxed_slice());
        let decode_fn = match machine_type {
            MachineType::Full => decode::<FullMachineDecoderConfig>,
            MachineType::FullUnsigned => decode::<FullUnsignedMachineDecoderConfig>,
            MachineType::Reduced => decode::<ReducedMachineDecoderConfig>,
        };
        let preprocessed_bytecode = text_section.iter().copied().map(decode_fn).collect_vec();
        setups::pad_bytecode_for_proving(&mut text_section);
        let instruction_tape = Arc::new(SimpleTape::new(&preprocessed_bytecode));
        let circuit_types = match execution_kind {
            ExecutionKind::Unrolled => {
                let memory =
                    UnrolledMemoryCircuitType::get_circuit_types_for_machine_type(machine_type)
                        .into_iter()
                        .copied()
                        .map(|ct| UnrolledCircuitType::Memory(ct));
                let non_memory =
                    UnrolledNonMemoryCircuitType::get_circuit_types_for_machine_type(machine_type)
                        .into_iter()
                        .copied()
                        .map(|ct| UnrolledCircuitType::NonMemory(ct));
                memory.chain(non_memory).collect_vec()
            }
            ExecutionKind::Unified => {
                assert_eq!(
                    machine_type,
                    MachineType::Reduced,
                    "Unified execution kind is only supported for Reduced machine type"
                );
                vec![UnrolledCircuitType::Unified]
            }
        };
        let precomputations = circuit_types.into_iter().map(|circuit_type| {
            debug!("PROVER producing precomputations for circuit {circuit_type:?} and binary with key {key:?}");
            let precomputations = CircuitPrecomputations::new(CircuitType::Unrolled(circuit_type), &binary_image, &text_section, &self.worker);
            (circuit_type, precomputations)
        }).collect();
        let holder = BinaryHolder {
            execution_kind,
            machine_type,
            binary_image,
            instruction_tape,
            precomputations,
        };
        info!("PROVER inserting binary with key {key:?}");
        assert!(self.binary_holders.insert(key, holder).is_none());
    }

    pub fn remove_binary(&mut self, key: usize) {
        info!("PROVER removing binary with key {key:?}");
        assert!(self.binary_holders.remove(&key).is_some());
    }

    fn get_result(
        &self,
        proving: bool,
        batch_id: u64,
        binary_key: usize,
        cycles_limit: usize,
        non_determinism_source: impl NonDeterminism + Send + Sync + 'static,
        external_challenges: Option<ExternalChallenges>,
    ) -> ExecutionProverResult {
        assert!(proving ^ external_challenges.is_none());
        let replayers_count = self.configuration.replay_worker_threads_count;
        let holder = &self.binary_holders[&binary_key];
        let (work_results_sender, work_results_receiver) = unbounded();
        let (gpu_work_requests_sender, gpu_work_requests_receiver) = unbounded();
        let mut gpu_work_requests_sender = Some(gpu_work_requests_sender);
        let (split_snapshot_sender, split_snapshot_receiver) = bounded(replayers_count);
        let (unified_snapshot_sender, unified_snapshot_receiver) = bounded(replayers_count);
        let gpu_work_batch = GpuWorkBatch {
            batch_id,
            receiver: gpu_work_requests_receiver,
            sender: work_results_sender.clone(),
        };
        trace!("BATCH[{batch_id}] PROVER sending work batch to GPU manager");
        self.gpu_manager.send_batch(gpu_work_batch);
        let execution_kind = holder.execution_kind;
        let machine_type = holder.machine_type;
        trace!("BATCH[{batch_id}] PROVER spawning SIMULATOR worker");
        {
            let free_allocators_receiver = self.free_allocators_receiver.clone();
            let binary_image = holder.binary_image.clone();
            let instruction_tape = holder.instruction_tape.clone();
            let work_results_sender = work_results_sender.clone();
            self.worker.pool.spawn(move || match execution_kind {
                ExecutionKind::Unrolled => run_split_simulator(
                    batch_id,
                    machine_type,
                    binary_image,
                    instruction_tape,
                    non_determinism_source,
                    cycles_limit,
                    split_snapshot_sender,
                    work_results_sender,
                    free_allocators_receiver,
                ),
                ExecutionKind::Unified => run_unified_simulator(
                    batch_id,
                    binary_image,
                    instruction_tape,
                    non_determinism_source,
                    cycles_limit,
                    unified_snapshot_sender,
                    work_results_sender,
                    free_allocators_receiver,
                ),
            });
        }
        trace!("BATCH[{batch_id}] PROVER spawning REPLAY workers");
        for worker_id in 0..replayers_count {
            let instruction_tape = holder.instruction_tape.clone();
            let split_snapshot_receiver = split_snapshot_receiver.clone();
            let unified_snapshot_receiver = unified_snapshot_receiver.clone();
            let work_results_sender = work_results_sender.clone();
            self.worker.pool.spawn(move || match execution_kind {
                ExecutionKind::Unrolled => run_split_replayer(
                    batch_id,
                    worker_id,
                    instruction_tape,
                    split_snapshot_receiver,
                    work_results_sender,
                ),
                ExecutionKind::Unified => run_unified_replayer(
                    batch_id,
                    worker_id,
                    instruction_tape,
                    unified_snapshot_receiver,
                    work_results_sender,
                ),
            });
        }
        drop(split_snapshot_receiver);
        drop(work_results_sender);
        let mut simulation_result = None;
        let mut pending_requests_count = 0;
        let mut uninitialized_tracing_data = BTreeMap::new();
        let mut uninitialized_tracing_data_key_by_snapshot_index = BTreeMap::new();
        let mut processed_snapshots = BTreeSet::<usize>::new();
        let mut unpaired_unified_inits_and_teardowns = BTreeMap::new();
        let mut unpaired_unified_tracing_data = BTreeMap::new();
        let mut circuit_families_memory_caps = BTreeMap::new();
        let mut inits_and_teardowns_memory_caps = BTreeMap::new();
        let mut delegation_circuits_memory_caps = BTreeMap::new();
        let mut circuit_families_proofs = BTreeMap::new();
        let mut inits_and_teardowns_proofs = BTreeMap::new();
        let mut delegation_circuits_proofs = BTreeMap::new();
        match execution_kind {
            ExecutionKind::Unrolled => {
                let non_memory =
                    UnrolledNonMemoryCircuitType::get_circuit_types_for_machine_type(machine_type)
                        .iter()
                        .map(|t| t.get_family_idx());
                let memory =
                    UnrolledMemoryCircuitType::get_circuit_types_for_machine_type(machine_type)
                        .iter()
                        .map(|t| t.get_family_idx());
                for family_idx in non_memory.chain(memory) {
                    circuit_families_memory_caps.insert(family_idx, BTreeMap::new());
                    circuit_families_proofs.insert(family_idx, BTreeMap::new());
                }
            }
            ExecutionKind::Unified => {
                let family_idx = UnrolledCircuitType::Unified.get_family_idx();
                circuit_families_memory_caps.insert(family_idx, BTreeMap::new());
                circuit_families_proofs.insert(family_idx, BTreeMap::new());
            }
        }
        let get_gpu_work_request = |inits_and_teardowns: Option<InitsAndTeardownsData<A>>,
                                    tracing_data: Option<TracingData<A>>|
         -> GpuWorkRequest<A> {
            let mut circuit_type_value = None;
            let mut sequence_id_vealue = None;
            let inits_and_teardowns = if let Some(inits_and_teardowns) = inits_and_teardowns {
                let InitsAndTeardownsData {
                    circuit_type,
                    sequence_id,
                    inits_and_teardowns,
                } = inits_and_teardowns;
                circuit_type_value = Some(circuit_type);
                sequence_id_vealue = Some(sequence_id);
                inits_and_teardowns
            } else {
                None
            };
            let tracing_data = if let Some(tracing_data) = tracing_data {
                let TracingData {
                    circuit_type,
                    sequence_id,
                    tracing_data,
                    ..
                } = tracing_data;
                assert_eq!(
                    circuit_type_value.get_or_insert(circuit_type),
                    &circuit_type
                );
                assert_eq!(sequence_id_vealue.get_or_insert(sequence_id), &sequence_id);
                Some(tracing_data)
            } else {
                None
            };
            let circuit_type = circuit_type_value.unwrap();
            let sequence_id = sequence_id_vealue.unwrap();
            let precomputations = match circuit_type {
                CircuitType::Delegation(_)
                | CircuitType::Unrolled(UnrolledCircuitType::InitsAndTeardowns) => {
                    self.common_precomputations[&circuit_type].clone()
                }
                CircuitType::Unrolled(circuit_type) => {
                    holder.precomputations[&circuit_type].clone()
                }
            };
            if proving {
                let request = ProofRequest {
                    batch_id,
                    circuit_type,
                    sequence_id,
                    precomputations,
                    inits_and_teardowns,
                    tracing_data,
                    external_challenges: external_challenges.clone().unwrap(),
                };
                GpuWorkRequest::Proof(request)
            } else {
                let request = MemoryCommitmentRequest {
                    batch_id,
                    circuit_type,
                    sequence_id,
                    precomputations,
                    inits_and_teardowns,
                    tracing_data,
                };
                GpuWorkRequest::MemoryCommitment(request)
            }
        };
        for work_result in work_results_receiver {
            let mut gpu_work_requests = VecDeque::new();
            match work_result {
                WorkerResult::InitsAndTeardownsData(data) => match data.circuit_type {
                    CircuitType::Unrolled(UnrolledCircuitType::InitsAndTeardowns) => {
                        let request = get_gpu_work_request(Some(data), None);
                        gpu_work_requests.push_back(request);
                    }
                    CircuitType::Unrolled(UnrolledCircuitType::Unified) => {
                        let sequence_id = data.sequence_id;
                        assert!(!unpaired_unified_inits_and_teardowns.contains_key(&sequence_id));
                        if let Some(tracing_data) =
                            unpaired_unified_tracing_data.remove(&sequence_id)
                        {
                            let request = get_gpu_work_request(Some(data), Some(tracing_data));
                            gpu_work_requests.push_back(request);
                        } else {
                            assert!(unpaired_unified_inits_and_teardowns
                                .insert(sequence_id, data)
                                .is_none());
                        }
                    }
                    _ => panic!("unexpected circuit type for inits and teardowns data"),
                },
                WorkerResult::TracingData(data) => {
                    if data
                        .participating_snapshot_indexes
                        .is_subset(&processed_snapshots)
                    {
                        match data.circuit_type {
                            CircuitType::Unrolled(UnrolledCircuitType::InitsAndTeardowns) => {
                                panic!(
                                "tracing data can not have the inits and teardowns circuit_type"
                            )
                            }
                            CircuitType::Unrolled(UnrolledCircuitType::Unified) => {
                                let sequence_id = data.sequence_id;
                                assert!(!unpaired_unified_tracing_data.contains_key(&sequence_id));
                                if let Some(inits_and_teardowns) =
                                    unpaired_unified_inits_and_teardowns.remove(&sequence_id)
                                {
                                    let request =
                                        get_gpu_work_request(Some(inits_and_teardowns), Some(data));
                                    gpu_work_requests.push_back(request);
                                } else {
                                    assert!(unpaired_unified_tracing_data
                                        .insert(sequence_id, data)
                                        .is_none());
                                }
                            }
                            _ => {
                                let request = get_gpu_work_request(None, Some(data));
                                gpu_work_requests.push_back(request);
                            }
                        }
                    } else {
                        let key = (data.circuit_type, data.sequence_id);
                        for snapshot_index in data.participating_snapshot_indexes.iter().copied() {
                            let entry = uninitialized_tracing_data_key_by_snapshot_index
                                .entry(snapshot_index)
                                .or_insert_with(|| BTreeSet::new());
                            assert!(!entry.contains(&key));
                            entry.insert(key);
                        }
                        assert!(uninitialized_tracing_data.insert(key, data).is_none());
                    }
                }
                WorkerResult::SimulationResult(result) => {
                    simulation_result = Some(result);
                }
                WorkerResult::SnapshotReplayed(sequence_id) => {
                    assert!(processed_snapshots.insert(sequence_id));
                    if let Some(keys) =
                        uninitialized_tracing_data_key_by_snapshot_index.get_mut(&sequence_id)
                    {
                        for key in keys.clone().into_iter() {
                            if uninitialized_tracing_data
                                .get(&key)
                                .unwrap()
                                .participating_snapshot_indexes
                                .is_subset(&processed_snapshots)
                            {
                                keys.remove(&key);
                                let data = uninitialized_tracing_data.remove(&key).unwrap();
                                match data.circuit_type {
                                    CircuitType::Unrolled(
                                        UnrolledCircuitType::InitsAndTeardowns,
                                    ) => {
                                        panic!(
                                            "tracing data can not have the inits and teardowns circuit_type"
                                        )
                                    }
                                    CircuitType::Unrolled(UnrolledCircuitType::Unified) => {
                                        let sequence_id = data.sequence_id;
                                        assert!(!unpaired_unified_tracing_data
                                            .contains_key(&sequence_id));
                                        if let Some(inits_and_teardowns) =
                                            unpaired_unified_inits_and_teardowns
                                                .remove(&sequence_id)
                                        {
                                            let request = get_gpu_work_request(
                                                Some(inits_and_teardowns),
                                                Some(data),
                                            );
                                            gpu_work_requests.push_back(request);
                                        } else {
                                            assert!(unpaired_unified_tracing_data
                                                .insert(sequence_id, data)
                                                .is_none());
                                        }
                                    }
                                    _ => {
                                        let request = get_gpu_work_request(None, Some(data));
                                        gpu_work_requests.push_back(request);
                                    }
                                }
                            }
                        }
                    }
                }
                WorkerResult::GpuWorkResult(result) => {
                    assert_ne!(pending_requests_count, 0);
                    pending_requests_count -= 1;
                    match result {
                        GpuWorkResult::MemoryCommitment(commitment) => {
                            assert!(!proving);
                            let MemoryCommitmentResult {
                                batch_id,
                                circuit_type,
                                sequence_id,
                                inits_and_teardowns,
                                tracing_data,
                                merkle_tree_caps,
                            } = commitment;
                            trace!("BATCH[{batch_id}] PROVER received memory commitment for circuit {circuit_type:?}[{sequence_id}]");
                            if let Some(inits_and_teardowns) = inits_and_teardowns {
                                for allocator in inits_and_teardowns.into_allocators() {
                                    self.free_allocators_sender.send(allocator).unwrap();
                                }
                            }
                            if let Some(tracing_data) = tracing_data {
                                for allocator in tracing_data.into_allocators() {
                                    self.free_allocators_sender.send(allocator).unwrap();
                                }
                            }
                            let caps = match circuit_type {
                                CircuitType::Delegation(circuit_type) => {
                                    &mut delegation_circuits_memory_caps
                                        .entry(circuit_type as u32)
                                        .or_insert_with(|| BTreeMap::new())
                                }
                                CircuitType::Unrolled(circuit_type) => match circuit_type {
                                    UnrolledCircuitType::InitsAndTeardowns => {
                                        &mut inits_and_teardowns_memory_caps
                                    }
                                    _ => &mut circuit_families_memory_caps
                                        .get_mut(&circuit_type.get_family_idx())
                                        .unwrap(),
                                },
                            };
                            assert!(caps.insert(sequence_id, merkle_tree_caps).is_none());
                        }
                        GpuWorkResult::Proof(proof) => {
                            assert!(proving);
                            let ProofResult {
                                batch_id,
                                circuit_type,
                                sequence_id,
                                inits_and_teardowns,
                                tracing_data,
                                proof,
                            } = proof;
                            trace!("BATCH[{batch_id}] PROVER received proof for circuit {circuit_type:?}[{sequence_id}]");
                            if let Some(inits_and_teardowns) = inits_and_teardowns {
                                for allocator in inits_and_teardowns.into_allocators() {
                                    self.free_allocators_sender.send(allocator).unwrap();
                                }
                            }
                            if let Some(tracing_data) = tracing_data {
                                for allocator in tracing_data.into_allocators() {
                                    self.free_allocators_sender.send(allocator).unwrap();
                                }
                            }
                            match circuit_type {
                                CircuitType::Delegation(circuit_type) => {
                                    assert!(delegation_circuits_proofs
                                        .entry(circuit_type as u32)
                                        .or_insert_with(|| BTreeMap::new())
                                        .insert(sequence_id, unrolled_proof_into_proof(proof))
                                        .is_none())
                                }
                                CircuitType::Unrolled(circuit_type) => match circuit_type {
                                    UnrolledCircuitType::InitsAndTeardowns => {
                                        assert!(inits_and_teardowns_proofs
                                            .insert(sequence_id, proof)
                                            .is_none())
                                    }
                                    _ => assert!(circuit_families_proofs
                                        .get_mut(&circuit_type.get_family_idx())
                                        .unwrap()
                                        .insert(sequence_id, proof)
                                        .is_none()),
                                },
                            };
                        }
                    }
                }
            }
            for request in gpu_work_requests {
                gpu_work_requests_sender
                    .as_ref()
                    .unwrap()
                    .send(request)
                    .unwrap();
                pending_requests_count += 1;
            }
            if simulation_result.is_some()
                && uninitialized_tracing_data.is_empty()
                && unpaired_unified_inits_and_teardowns.is_empty()
                && unpaired_unified_tracing_data.is_empty()
            {
                gpu_work_requests_sender = None;
            }
        }
        assert_eq!(pending_requests_count, 0);
        let SimulationResult {
            final_register_values,
            final_pc,
            final_timestamp,
        } = simulation_result.unwrap();
        if proving {
            let circuit_families_proofs = circuit_families_proofs
                .into_iter()
                .map(|(i, v)| (i, v.into_values().collect_vec()))
                .collect();
            let inits_and_teardowns_proofs = inits_and_teardowns_proofs
                .into_iter()
                .map(|(_, v)| v)
                .collect_vec();
            let delegation_circuits_proofs = delegation_circuits_proofs
                .into_iter()
                .map(|(i, v)| (i, v.into_values().collect_vec()))
                .collect();
            let result = ProveResult {
                register_final_values: final_register_values,
                final_pc,
                final_timestamp,
                circuit_families_proofs,
                inits_and_teardowns_proofs,
                delegation_proofs: delegation_circuits_proofs,
            };
            ExecutionProverResult::Prove(result)
        } else {
            let circuit_families_memory_caps = circuit_families_memory_caps
                .into_iter()
                .map(|(i, v)| (i, v.into_values().collect_vec()))
                .collect();
            let inits_and_teardowns_memory_caps = inits_and_teardowns_memory_caps
                .into_iter()
                .map(|(_, v)| v)
                .collect_vec();
            let delegation_circuits_memory_caps = delegation_circuits_memory_caps
                .into_iter()
                .map(|(i, v)| (i, v.into_values().collect_vec()))
                .collect();
            let result = CommitMemoryResult {
                final_register_values,
                final_pc,
                final_timestamp,
                circuit_families_memory_caps,
                inits_and_teardowns_memory_caps,
                delegation_circuits_memory_caps,
            };
            ExecutionProverResult::CommitMemory(result)
        }
    }

    fn commit_memory_inner(
        &self,
        batch_id: u64,
        binary_key: usize,
        cycle_limit: usize,
        non_determinism_source: impl NonDeterminism + Send + Sync + 'static,
    ) -> CommitMemoryResult {
        info!("BATCH[{batch_id}] PROVER producing memory commitments for binary with key {binary_key:?}");
        let timer = Instant::now();
        let result = self
            .get_result(
                false,
                batch_id,
                binary_key,
                cycle_limit,
                non_determinism_source,
                None,
            )
            .into_memory_commitment_result();
        let elapsed = timer.elapsed().as_secs_f64();
        info!("BATCH[{batch_id}] PROVER produced memory commitments for binary with key {binary_key:?} in {elapsed:.3}s");
        result
    }

    ///  Produces memory commitments.
    ///
    /// # Arguments
    ///
    /// * `batch_id`: a unique identifier for the batch of work, used to distinguish batches in a multithreaded scenario
    /// * `binary_key`: a key that identifies the binary to work with, this key must match one of the keys in the `binaries` map provided during the creation of the `ExecutionProver`
    /// * `num_instances_upper_bound`: maximum number of main circuit instances that the prover will try to trace, if the simulation does not end within this limit, it will fail
    /// * `non_determinism_source`: a value implementing the `NonDeterminism` trait that provides non-deterministic values for the simulation
    ///
    /// returns: a tuple containing:
    ///     - final register values for the main circuit,
    ///     - a vector of memory commitments for the chunks of the main circuit,
    ///     - a vector of memory commitments for the chunks of the delegation circuits, where each element is a tuple containing the delegation circuit type and a vector of memory commitments for that type
    ///
    pub fn commit_memory(
        &self,
        batch_id: u64,
        binary_key: usize,
        cycle_limit: usize,
        non_determinism_source: impl NonDeterminism + Send + Sync + 'static,
    ) -> CommitMemoryResult {
        self.commit_memory_inner(batch_id, binary_key, cycle_limit, non_determinism_source)
    }

    fn prove_inner(
        &self,
        batch_id: u64,
        binary_key: usize,
        cycle_limit: usize,
        non_determinism_source: impl NonDeterminism + Send + Sync + 'static,
        external_challenges: ExternalChallenges,
    ) -> ProveResult {
        info!("BATCH[{batch_id}] PROVER producing proofs for binary with key {binary_key:?}");
        let timer = Instant::now();
        let result = self
            .get_result(
                true,
                batch_id,
                binary_key,
                cycle_limit,
                non_determinism_source,
                Some(external_challenges),
            )
            .into_proof_result();
        let elapsed = timer.elapsed().as_secs_f64();
        info!("BATCH[{batch_id}] PROVER produced proofs for binary with key {binary_key:?} in {elapsed:.3}s");
        result
    }

    ///  Produces proofs.
    ///
    /// # Arguments
    ///
    /// * `batch_id`: a unique identifier for the batch of work, used to distinguish batches in a multithreaded scenario
    /// * `binary_key`: a key that identifies the binary to work with, this key must match one of the keys in the `binaries` map provided during the creation of the `ExecutionProver`
    /// * `num_instances_upper_bound`: maximum number of main circuit instances that the prover will try to trace, if the simulation does not end within this limit, it will fail
    /// * `non_determinism_source`: a value implementing the `NonDeterminism` trait that provides non-deterministic values for the simulation
    /// * `external_challenges`: an instance of `ExternalChallenges` that contains the challenges to be used in the proof generation
    ///
    /// returns: a tuple containing:
    ///     - final register values for the main circuit,
    ///     - a vector of proofs for the chunks of the main circuit,
    ///     - a vector of proofs for the chunks of the delegation circuits, where each element is a tuple containing the delegation circuit type and a vector of memory commitments for that type
    ///
    pub fn prove(
        &self,
        batch_id: u64,
        binary_key: usize,
        cycle_limit: usize,
        non_determinism_source: impl NonDeterminism + Send + Sync + 'static,
        external_challenges: ExternalChallenges,
    ) -> ProveResult {
        self.prove_inner(
            batch_id,
            binary_key,
            cycle_limit,
            non_determinism_source,
            external_challenges,
        )
    }

    ///  Commits to memory and produces proofs using challenge derived from the memory commitments.
    ///
    /// # Arguments
    ///
    /// * `batch_id`: a unique identifier for the batch of work, used to distinguish batches in a multithreaded scenario
    /// * `binary_key`: a key that identifies the binary to work with, this key must match one of the keys in the `binaries` map provided during the creation of the `ExecutionProver`
    /// * `num_instances_upper_bound`: maximum number of main circuit instances that the prover will try to trace, if the simulation does not end within this limit, it will fail
    /// * `non_determinism_source`: a value implementing the `NonDeterminism` trait that provides non-deterministic values for the simulation
    ///
    /// returns: a tuple containing:
    ///     - final register values for the main circuit,
    ///     - a vector of proofs for the chunks of the main circuit,
    ///     - a vector of proofs for the chunks of the delegation circuits, where each element is a tuple containing the delegation circuit type and a vector of memory commitments for that type
    ///
    pub fn commit_memory_and_prove(
        &self,
        batch_id: u64,
        binary_key: usize,
        cycle_limit: usize,
        non_determinism_source: impl NonDeterminism + Clone + Send + Sync + 'static,
    ) -> ProveResult {
        let timer = Instant::now();
        let memory_commitment_result = self.commit_memory_inner(
            batch_id,
            binary_key,
            cycle_limit,
            non_determinism_source.clone(),
        );
        let CommitMemoryResult {
            final_register_values,
            final_pc,
            final_timestamp,
            circuit_families_memory_caps,
            inits_and_teardowns_memory_caps,
            delegation_circuits_memory_caps,
        } = memory_commitment_result;
        let all_challenges_seed =
            fs_transform_for_memory_and_delegation_arguments_for_unrolled_circuits(
                &final_register_values,
                final_pc,
                final_timestamp,
                &circuit_families_memory_caps
                    .iter()
                    .map(|(i, v)| (*i as u32, v.clone()))
                    .collect_vec(),
                &inits_and_teardowns_memory_caps,
                &delegation_circuits_memory_caps
                    .iter()
                    .map(|(i, v)| (*i, v.clone()))
                    .collect_vec(),
            );
        let external_challenges =
            ExternalChallenges::draw_from_transcript_seed_with_state_permutation(
                all_challenges_seed,
            );
        let prove_result = self.prove_inner(
            batch_id,
            binary_key,
            cycle_limit,
            non_determinism_source,
            external_challenges,
        );
        assert_eq!(prove_result.register_final_values, final_register_values);
        assert_eq!(prove_result.final_pc, final_pc);
        assert_eq!(prove_result.final_timestamp, final_timestamp);
        fn assert_caps_mach(
            a: &Vec<Vec<MerkleTreeCapVarLength>>,
            b: &Vec<Vec<MerkleTreeCapVarLength>>,
        ) {
            for (a, b) in a.iter().zip(b.iter()) {
                assert_eq!(a, b);
            }
        }
        prove_result
            .circuit_families_proofs
            .iter()
            .zip(circuit_families_memory_caps.iter())
            .for_each(|(a, b)| {
                assert_eq!(a.0, b.0);
                assert_caps_mach(
                    &a.1.iter().map(|p| p.memory_tree_caps.clone()).collect_vec(),
                    &b.1,
                );
            });
        assert_caps_mach(
            &prove_result
                .inits_and_teardowns_proofs
                .iter()
                .map(|p| p.memory_tree_caps.clone())
                .collect_vec(),
            &inits_and_teardowns_memory_caps,
        );
        prove_result
            .delegation_proofs
            .iter()
            .zip(delegation_circuits_memory_caps.iter())
            .for_each(|(a, b)| {
                assert_eq!(a.0, b.0);
                assert_caps_mach(
                    &a.1.iter().map(|p| p.memory_tree_caps.clone()).collect_vec(),
                    &b.1,
                );
            });
        let elapsed = timer.elapsed().as_secs_f64();
        info!("BATCH[{batch_id}] PROVER committed to memory and produced proofs for binary with key {binary_key:?} in {elapsed:.3}s");
        prove_result
    }
}

// impl<'a, K: Debug + Eq + Hash> Drop for ExecutionProver<K> {
//     fn drop(&mut self) {
//         trace!("PROVER waiting for all threads to finish");
//         self.wait_group.take().unwrap().wait();
//         trace!("PROVER all threads finished");
//     }
// }

fn unrolled_proof_into_proof(proof: UnrolledModeProof) -> Proof {
    assert!(proof.aux_boundary_values.is_empty());
    Proof {
        external_values: ExternalValues {
            challenges: proof.external_challenges,
            aux_boundary_values: AuxArgumentsBoundaryValues::default(),
        },
        public_inputs: proof.public_inputs,
        witness_tree_caps: proof.witness_tree_caps,
        memory_tree_caps: proof.memory_tree_caps,
        setup_tree_caps: proof.setup_tree_caps,
        stage_2_tree_caps: proof.stage_2_tree_caps,
        memory_grand_product_accumulator: proof.permutation_grand_product_accumulator,
        delegation_argument_accumulator: proof.delegation_argument_accumulator,
        quotient_tree_caps: proof.quotient_tree_caps,
        evaluations_at_random_points: proof.evaluations_at_random_points,
        deep_poly_caps: proof.deep_poly_caps,
        intermediate_fri_oracle_caps: proof.intermediate_fri_oracle_caps,
        last_fri_step_plain_leaf_values: proof.last_fri_step_plain_leaf_values,
        final_monomial_form: proof.final_monomial_form,
        queries: proof.queries,
        pow_nonce: proof.pow_nonce,
        circuit_sequence: 0,
        delegation_type: proof.delegation_type,
    }
}

#[cfg(test)]
mod tests {
    use crate::execution::prover::{ExecutionKind, ExecutionProver, ExecutionProverConfiguration};
    use crate::machine_type::MachineType;
    use crate::tests::{init_logger, read_binary};
    use prover::risc_v_simulator::abstractions::non_determinism::QuasiUARTSource;
    use std::path::Path;

    #[test]
    fn test_execution_prover() {
        init_logger();
        let mut configuration = ExecutionProverConfiguration::default();
        configuration.replay_worker_threads_count = 8;
        let mut prover = ExecutionProver::with_configuration(configuration);
        let binary_image = read_binary(&Path::new("../examples/hashed_fibonacci/app.bin"));
        let text_section = read_binary(&Path::new("../examples/hashed_fibonacci/app.text"));
        prover.add_binary(
            0,
            ExecutionKind::Unrolled,
            MachineType::FullUnsigned,
            binary_image,
            text_section,
        );
        let non_determinism_source = QuasiUARTSource::new_with_reads(vec![3 << 25, 0]);
        let base_layer_result =
            prover.commit_memory_and_prove(0, 0, 1 << 36, non_determinism_source);
        let binary_image = read_binary(&Path::new("../tools/verifier/unrolled_base_layer.bin"));
        let text_section = read_binary(&Path::new("../tools/verifier/unrolled_base_layer.text"));
        prover.add_binary(
            1,
            ExecutionKind::Unrolled,
            MachineType::Reduced,
            binary_image,
            text_section,
        );

        drop(prover);
    }
}
