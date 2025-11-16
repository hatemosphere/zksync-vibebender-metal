use super::gpu_manager::{GpuManager, GpuWorkBatch};
use super::precomputations::{get_common_precomputations, CircuitPrecomputations};
use crate::allocator::host::ConcurrentStaticHostAllocator;
use crate::circuit_type::{
    CircuitType, UnrolledCircuitType, UnrolledMemoryCircuitType, UnrolledNonMemoryCircuitType,
};
use crate::cudart::device::get_device_count;
use crate::cudart::memory::{CudaHostAllocFlags, HostAllocation};
use crate::execution::cpu_worker::{run_split_replayer, run_split_simulator, NonDeterminism};
use crate::execution::gpu_worker;
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
use prover::definitions::ExternalChallenges;
use prover::merkle_trees::MerkleTreeCapVarLength;
use prover::prover_stages::unrolled_prover::UnrolledModeProof;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use riscv_transpiler::ir::{
    decode, FullMachineDecoderConfig, FullUnsignedMachineDecoderConfig, ReducedMachineDecoderConfig,
};
use riscv_transpiler::vm::SimpleTape;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Instant;
use trace_and_split::{
    fs_transform_for_memory_and_delegation_arguments_for_unrolled_circuits, FinalRegisterValue,
};
use worker::Worker;

type A = ConcurrentStaticHostAllocator;

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
    pub cycles_used: usize,
    pub circuit_families_memory_caps: Vec<(u32, Vec<Vec<MerkleTreeCapVarLength>>)>,
    pub inits_and_teardowns_memory_caps: Vec<Vec<MerkleTreeCapVarLength>>,
    pub delegation_circuits_memory_caps: Vec<(u32, Vec<Vec<MerkleTreeCapVarLength>>)>,
}

pub struct ProveResult {
    pub final_register_values: [FinalRegisterValue; 32],
    pub final_pc: u32,
    pub final_timestamp: TimestampScalar,
    pub cycles_used: usize,
    pub circuit_families_proofs: Vec<(u32, Vec<UnrolledModeProof>)>,
    pub inits_and_teardowns_proofs: Vec<UnrolledModeProof>,
    pub delegation_circuits_proofs: Vec<(u32, Vec<UnrolledModeProof>)>,
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

pub struct ExecutionProver<K: Debug + Eq + Hash> {
    configuration: ExecutionProverConfiguration,
    device_count: usize,
    gpu_manager: GpuManager,
    worker: Worker,
    binary_holders: HashMap<K, BinaryHolder>,
    common_precomputations: HashMap<CircuitType, CircuitPrecomputations>,
    free_allocators_sender: Sender<A>,
    free_allocators_receiver: Receiver<A>,
}

impl<K: Clone + Debug + Eq + Hash> ExecutionProver<K> {
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
        let binary_holders = HashMap::new();
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
            let allocator =
                ConcurrentStaticHostAllocator::new([allocation], host_allocation_log_chunk_size);
            free_allocators_sender_ref.send(allocator).unwrap();
        });
        gpu_wait_group.wait();
        info!("PROVER initialized");
        Self {
            configuration,
            device_count,
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
        key: K,
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

    pub fn remove_binary(&mut self, key: &K) {
        info!("PROVER removing binary with key {key:?}");
        assert!(self.binary_holders.remove(key).is_some());
    }

    fn get_result(
        &self,
        proving: bool,
        batch_id: u64,
        binary_key: &K,
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
        let (snapshot_sender, snapshot_receiver) = bounded(replayers_count);
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
                    snapshot_sender,
                    work_results_sender,
                    free_allocators_receiver,
                ),
                ExecutionKind::Unified => todo!(),
            });
        }
        trace!("BATCH[{batch_id}] PROVER spawning REPLAY workers");
        for worker_id in 0..replayers_count {
            let instruction_tape = holder.instruction_tape.clone();
            let snapshot_receiver = snapshot_receiver.clone();
            let work_results_sender = work_results_sender.clone();
            self.worker.pool.spawn(move || match execution_kind {
                ExecutionKind::Unrolled => run_split_replayer(
                    batch_id,
                    worker_id,
                    instruction_tape,
                    snapshot_receiver,
                    work_results_sender,
                ),
                ExecutionKind::Unified => todo!(),
            });
        }
        drop(snapshot_receiver);
        drop(work_results_sender);
        let mut simulation_result = None;
        let mut expected_requests_count = 0;
        let mut pending_requests_count = 0;
        let mut tracing_data: HashMap<(CircuitType, usize), TracingData<A>> = HashMap::new();
        let mut tracing_data_key_by_snapshot_index: HashMap<usize, Vec<(CircuitType, usize)>> =
            HashMap::new();
        let mut processed_snapshots = HashSet::<usize>::new();
        let mut gpu_work_requests = VecDeque::new();
        let mut circuit_families_memory_caps = HashMap::new();
        let mut inits_and_teardowns_memory_caps = HashMap::new();
        let mut delegation_circuits_memory_caps = HashMap::new();
        let mut circuit_families_proofs = HashMap::new();
        let mut inits_and_teardowns_proofs = HashMap::new();
        let mut delegation_circuits_proofs = HashMap::new();
        let get_gpu_work_request = |tracing_data: TracingData<A>| -> GpuWorkRequest<A> {
            let TracingData {
                circuit_type,
                sequence_id,
                tracing_data,
                ..
            } = tracing_data;
            let precomputations = match circuit_type {
                CircuitType::Delegation(_)
                | CircuitType::Unrolled(UnrolledCircuitType::InitsAndTeardowns) => {
                    self.common_precomputations[&circuit_type].clone()
                }
                CircuitType::Unrolled(circuit_type) => {
                    holder.precomputations[&circuit_type].clone()
                }
            };
            let inits_and_teardowns = match circuit_type {
                CircuitType::Unrolled(UnrolledCircuitType::Unified) => todo!(),
                _ => None,
            };
            let tracing_data = Some(tracing_data);
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
            match work_result {
                WorkerResult::InitsAndTeardownsData(data) => {
                    let InitsAndTeardownsData {
                        circuit_type,
                        sequence_id,
                        inits_and_teardowns,
                    } = data;
                    match circuit_type {
                        CircuitType::Unrolled(UnrolledCircuitType::InitsAndTeardowns) => {
                            assert!(inits_and_teardowns.is_some());
                            let precomputations =
                                self.common_precomputations[&circuit_type].clone();
                            let request = if proving {
                                let request = ProofRequest {
                                    batch_id,
                                    circuit_type,
                                    sequence_id,
                                    precomputations,
                                    inits_and_teardowns,
                                    tracing_data: None,
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
                                    tracing_data: None,
                                };
                                GpuWorkRequest::MemoryCommitment(request)
                            };
                            gpu_work_requests.push_back(request);
                            expected_requests_count += 1;
                        }
                        CircuitType::Unrolled(UnrolledCircuitType::Unified) => todo!(),
                        _ => panic!("unexpected circuit type for inits and teardowns data"),
                    }
                }
                WorkerResult::TracingData(data) => {
                    if data
                        .participating_snapshot_indexes
                        .is_subset(&processed_snapshots)
                    {
                        let request = get_gpu_work_request(data);
                        gpu_work_requests.push_back(request);
                    } else {
                        let key = (data.circuit_type, data.sequence_id);
                        for snapshot_index in data.participating_snapshot_indexes.iter().copied() {
                            let entry = tracing_data_key_by_snapshot_index
                                .entry(snapshot_index)
                                .or_insert_with(|| Vec::new());
                            assert!(!entry.contains(&key));
                            entry.push(key);
                        }
                        assert!(tracing_data.insert(key, data).is_none());
                    }
                    expected_requests_count += 1;
                }
                WorkerResult::SimulationResult(result) => {
                    simulation_result = Some(result);
                }
                WorkerResult::SnapshotReplayed(sequence_id) => {
                    assert!(processed_snapshots.insert(sequence_id));
                    if let Some(keys) = tracing_data_key_by_snapshot_index.get_mut(&sequence_id) {
                        for key in keys.clone().iter() {
                            if tracing_data
                                .get(key)
                                .unwrap()
                                .participating_snapshot_indexes
                                .is_subset(&processed_snapshots)
                            {
                                let (index, _) =
                                    keys.iter().copied().find_position(|k| k.eq(key)).unwrap();
                                keys.remove(index);
                                let data = tracing_data.remove(key).unwrap();
                                let request = get_gpu_work_request(data);
                                gpu_work_requests.push_back(request);
                            }
                        }
                    }
                }
                WorkerResult::GpuWorkResult(result) => match result {
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
                                    .or_insert_with(|| HashMap::new())
                            }
                            CircuitType::Unrolled(circuit_type) => match circuit_type {
                                UnrolledCircuitType::InitsAndTeardowns => {
                                    &mut inits_and_teardowns_memory_caps
                                }
                                _ => &mut circuit_families_memory_caps
                                    .entry(circuit_type.get_family_idx() as u32)
                                    .or_insert_with(|| HashMap::new()),
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
                        let proofs = match circuit_type {
                            CircuitType::Delegation(circuit_type) => {
                                &mut delegation_circuits_proofs
                                    .entry(circuit_type as u32)
                                    .or_insert_with(|| HashMap::new())
                            }
                            CircuitType::Unrolled(circuit_type) => match circuit_type {
                                UnrolledCircuitType::InitsAndTeardowns => {
                                    &mut inits_and_teardowns_proofs
                                }
                                _ => &mut circuit_families_proofs
                                    .entry(circuit_type.get_family_idx() as u32)
                                    .or_insert_with(|| HashMap::new()),
                            },
                        };
                        assert!(proofs.insert(sequence_id, proof).is_none());
                    }
                },
            }
            while let Some(request) = gpu_work_requests.pop_front() {
                gpu_work_requests_sender
                    .as_ref()
                    .unwrap()
                    .send(request)
                    .unwrap();
                pending_requests_count += 1;
            }
            if pending_requests_count == expected_requests_count && simulation_result.is_some() {
                gpu_work_requests_sender = None;
            }
        }
        let SimulationResult {
            final_register_values,
            final_pc,
            final_timestamp,
            cycles_used,
        } = simulation_result.unwrap();
        if proving {
            let count = circuit_families_proofs
                .values()
                .map(|v| v.len())
                .sum::<usize>()
                + inits_and_teardowns_proofs.len()
                + delegation_circuits_proofs
                    .values()
                    .map(|v| v.len())
                    .sum::<usize>();
            assert_eq!(expected_requests_count, count);
            let circuit_families_proofs = circuit_families_proofs
                .into_iter()
                .map(|(i, v)| {
                    (
                        i,
                        v.into_iter()
                            .sorted_by_key(|(i, _)| *i)
                            .map(|(_, v)| v)
                            .collect_vec(),
                    )
                })
                .sorted_by_key(|(i, _)| *i)
                .collect_vec();
            let inits_and_teardowns_proofs = inits_and_teardowns_proofs
                .into_iter()
                .sorted_by_key(|(i, _)| *i)
                .map(|(_, v)| v)
                .collect_vec();
            let delegation_circuits_proofs = delegation_circuits_proofs
                .into_iter()
                .map(|(i, v)| {
                    (
                        i,
                        v.into_iter()
                            .sorted_by_key(|(i, _)| *i)
                            .map(|(_, v)| v)
                            .collect_vec(),
                    )
                })
                .sorted_by_key(|(i, _)| *i)
                .collect_vec();
            let result = ProveResult {
                final_register_values,
                final_pc,
                final_timestamp,
                cycles_used,
                circuit_families_proofs,
                inits_and_teardowns_proofs,
                delegation_circuits_proofs,
            };
            ExecutionProverResult::Prove(result)
        } else {
            let count = circuit_families_memory_caps
                .values()
                .map(|v| v.len())
                .sum::<usize>()
                + inits_and_teardowns_memory_caps.len()
                + delegation_circuits_memory_caps
                    .values()
                    .map(|v| v.len())
                    .sum::<usize>();
            assert_eq!(expected_requests_count, count);
            let circuit_families_memory_caps = circuit_families_memory_caps
                .into_iter()
                .map(|(i, v)| {
                    (
                        i,
                        v.into_iter()
                            .sorted_by_key(|(i, _)| *i)
                            .map(|(_, v)| v)
                            .collect_vec(),
                    )
                })
                .sorted_by_key(|(i, _)| *i)
                .collect_vec();
            let inits_and_teardowns_memory_caps = inits_and_teardowns_memory_caps
                .into_iter()
                .sorted_by_key(|(i, _)| *i)
                .map(|(_, v)| v)
                .collect_vec();
            let delegation_circuits_memory_caps = delegation_circuits_memory_caps
                .into_iter()
                .map(|(i, v)| {
                    (
                        i,
                        v.into_iter()
                            .sorted_by_key(|(i, _)| *i)
                            .map(|(_, v)| v)
                            .collect_vec(),
                    )
                })
                .sorted_by_key(|(i, _)| *i)
                .collect_vec();
            let result = CommitMemoryResult {
                final_register_values,
                final_pc,
                final_timestamp,
                cycles_used,
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
        binary_key: &K,
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
        binary_key: &K,
        cycle_limit: usize,
        non_determinism_source: impl NonDeterminism + Send + Sync + 'static,
    ) -> CommitMemoryResult {
        self.commit_memory_inner(batch_id, binary_key, cycle_limit, non_determinism_source)
    }

    fn prove_inner(
        &self,
        batch_id: u64,
        binary_key: &K,
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
        binary_key: &K,
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
        binary_key: &K,
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
            cycles_used,
            circuit_families_memory_caps,
            inits_and_teardowns_memory_caps,
            delegation_circuits_memory_caps,
        } = memory_commitment_result;
        let all_challenges_seed =
            fs_transform_for_memory_and_delegation_arguments_for_unrolled_circuits(
                &final_register_values,
                final_pc,
                final_timestamp,
                &circuit_families_memory_caps,
                &inits_and_teardowns_memory_caps,
                &delegation_circuits_memory_caps,
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
        assert_eq!(prove_result.final_register_values, final_register_values);
        assert_eq!(prove_result.final_pc, final_pc);
        assert_eq!(prove_result.final_timestamp, final_timestamp);
        assert_eq!(prove_result.cycles_used, cycles_used);
        let assert_caps_mach =
            |a: &Vec<UnrolledModeProof>, b: &Vec<Vec<MerkleTreeCapVarLength>>| {
                for (proof, caps) in a.iter().zip(b.iter()) {
                    assert_eq!(&proof.memory_tree_caps, caps);
                }
            };
        let assert_ids_and_caps_mach = |tuple: (
            &(u32, Vec<UnrolledModeProof>),
            &(u32, Vec<Vec<MerkleTreeCapVarLength>>),
        )| {
            let a = tuple.0;
            let b = tuple.1;
            assert_eq!(a.0, b.0);
            assert_caps_mach(&a.1, &b.1);
        };
        prove_result
            .circuit_families_proofs
            .iter()
            .zip(circuit_families_memory_caps.iter())
            .for_each(assert_ids_and_caps_mach);
        assert_caps_mach(
            &prove_result.inits_and_teardowns_proofs,
            &inits_and_teardowns_memory_caps,
        );
        prove_result
            .delegation_circuits_proofs
            .iter()
            .zip(delegation_circuits_memory_caps.iter())
            .for_each(assert_ids_and_caps_mach);
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
        let mut prover = ExecutionProver::<usize>::with_configuration(configuration);
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
        let result = prover.commit_memory_and_prove(0, &0, 1 << 36, non_determinism_source);
        drop(prover);
    }
}
