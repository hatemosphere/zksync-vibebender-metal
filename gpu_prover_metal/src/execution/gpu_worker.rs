use super::messages::WorkerResult;
use super::precomputations::CircuitPrecomputations;
use crate::allocator::host::ConcurrentStaticHostAllocator;
use crate::circuit_type::{CircuitType, MainCircuitType};
use crate::metal_runtime::MetalResult;
use crate::prover::context::{ProverContext, ProverContextConfig};
use crate::prover::setup::SetupPrecomputations;
use crate::prover::trace_holder::TreesCacheMode;
use crate::prover::tracing_data::TracingDataTransfer;
use crossbeam_channel::{Receiver, Sender};
use fft::GoodAllocator;
use field::Mersenne31Field;
use log::{error, info, trace};
use prover::definitions::{
    AuxArgumentsBoundaryValues, ExternalChallenges, ExternalValues, OPTIMAL_FOLDING_PROPERTIES,
};
use prover::merkle_trees::MerkleTreeCapVarLength;
use prover::prover_stages::Proof;
use std::cell::RefCell;
use std::process::exit;
use std::rc::Rc;
use std::sync::Arc;

type BF = Mersenne31Field;

const NUM_QUERIES: usize = 53;
const POW_BITS: u32 = 28;

pub use crate::prover::tracing_data::TracingDataHost;

#[derive(Clone)]
pub struct SetupToCache {
    pub circuit_type: CircuitType,
    pub precomputations: CircuitPrecomputations,
}

pub struct MemoryCommitmentRequest<A: GoodAllocator> {
    pub batch_id: u64,
    pub circuit_type: CircuitType,
    pub circuit_sequence: usize,
    pub precomputations: CircuitPrecomputations,
    pub tracing_data: TracingDataHost<A>,
}

pub struct MemoryCommitmentResult<A: GoodAllocator> {
    pub batch_id: u64,
    pub circuit_type: CircuitType,
    pub circuit_sequence: usize,
    pub tracing_data: TracingDataHost<A>,
    pub merkle_tree_caps: Vec<MerkleTreeCapVarLength>,
}

pub struct ProofRequest<A: GoodAllocator> {
    pub batch_id: u64,
    pub circuit_type: CircuitType,
    pub circuit_sequence: usize,
    pub precomputations: CircuitPrecomputations,
    pub tracing_data: TracingDataHost<A>,
    pub external_challenges: ExternalChallenges,
}

pub struct ProofResult<A: GoodAllocator> {
    pub batch_id: u64,
    pub circuit_type: CircuitType,
    pub circuit_sequence: usize,
    pub tracing_data: TracingDataHost<A>,
    pub proof: Proof,
}

pub enum GpuWorkRequest<A: GoodAllocator> {
    MemoryCommitment(MemoryCommitmentRequest<A>),
    Proof(ProofRequest<A>),
}

impl<A: GoodAllocator> GpuWorkRequest<A> {
    pub fn batch_id(&self) -> u64 {
        match self {
            GpuWorkRequest::MemoryCommitment(request) => request.batch_id,
            GpuWorkRequest::Proof(request) => request.batch_id,
        }
    }
}

pub fn get_gpu_worker_func(
    device_id: i32,
    prover_context_config: ProverContextConfig,
    setups_to_cache: Vec<SetupToCache>,
    is_initialized: Sender<()>,
    requests: Receiver<Option<GpuWorkRequest<ConcurrentStaticHostAllocator>>>,
    results: Sender<Option<WorkerResult<ConcurrentStaticHostAllocator>>>,
) -> impl FnOnce() + Send + 'static {
    move || {
        let result = gpu_worker(
            device_id,
            prover_context_config,
            setups_to_cache,
            is_initialized,
            requests,
            results,
        );
        if let Err(e) = result {
            error!("GPU_WORKER[{device_id}] worker encountered an error: {e}");
            exit(1);
        }
    }
}

const fn get_tree_cap_size(log_domain_size: u32) -> u32 {
    OPTIMAL_FOLDING_PROPERTIES[log_domain_size as usize].total_caps_size_log2 as u32
}

fn get_trees_cache_mode(circuit_type: CircuitType, _context: &ProverContext) -> TreesCacheMode {
    // On Apple Silicon with unified memory, we can always cache
    // unless memory is very constrained
    match circuit_type {
        CircuitType::Main(main) => match main {
            MainCircuitType::ReducedRiscVLog23Machine => {
                // Conservative: don't cache for the largest circuit if memory is limited
                TreesCacheMode::CacheNone
            }
            _ => TreesCacheMode::CacheFull,
        },
        _ => TreesCacheMode::CacheFull,
    }
}

#[derive(Clone)]
struct SetupHolder<'a> {
    pub setup: Rc<RefCell<SetupPrecomputations<'a>>>,
    pub trace: Arc<Vec<BF, ConcurrentStaticHostAllocator>>,
}

fn gpu_worker(
    device_id: i32,
    prover_context_config: ProverContextConfig,
    setups_to_cache: Vec<SetupToCache>,
    is_initialized: Sender<()>,
    requests: Receiver<Option<GpuWorkRequest<ConcurrentStaticHostAllocator>>>,
    results: Sender<Option<WorkerResult<ConcurrentStaticHostAllocator>>>,
) -> MetalResult<()> {
    trace!("GPU_WORKER[{device_id}] started");

    // On Metal, there's typically only one device (Apple Silicon GPU)
    let context = ProverContext::new(&prover_context_config)?;
    let _device = context.device();
    let props = context.device_properties();
    info!(
        "GPU_WORKER[{device_id}] GPU: {} (unified memory)",
        props.name,
    );

    // Cache setups
    log::info!("GPU_WORKER[{device_id}] caching {} setups", setups_to_cache.len());
    for (setup_idx, precomputations) in setups_to_cache.iter().enumerate().filter_map(|(i, setup)| {
        if device_id == 0 || (i as i32 % 1) == device_id {
            Some((i, &setup.precomputations))
        } else {
            None
        }
    }) {
        let lde_factor = precomputations.lde_precomputations.lde_factor;
        assert!(lde_factor.is_power_of_two());
        let log_lde_factor = lde_factor.trailing_zeros();
        let domain_size = precomputations.lde_precomputations.domain_size;
        assert!(domain_size.is_power_of_two());
        let log_domain_size = domain_size.trailing_zeros();
        let log_tree_cap_size = get_tree_cap_size(log_domain_size);
        let circuit = &precomputations.compiled_circuit;
        let trace = precomputations.setup_trace.clone();
        let t = std::time::Instant::now();
        log::info!("GPU_WORKER[{device_id}] caching setup {setup_idx} (domain=2^{log_domain_size}, cols={})", circuit.setup_layout.total_width);
        let _ = precomputations.setup_trees_and_caps.get_or_try_init(|| {
            SetupPrecomputations::get_trees_and_caps(
                circuit,
                log_lde_factor,
                log_tree_cap_size,
                trace,
                &context,
            )
        })?;
        log::info!("GPU_WORKER[{device_id}] cached setup {setup_idx} in {:.1}s", t.elapsed().as_secs_f64());
    }

    let mut setups = vec![];
    for setup in setups_to_cache {
        let SetupToCache {
            circuit_type,
            precomputations,
        } = setup;
        let lde_factor = precomputations.lde_precomputations.lde_factor;
        assert!(lde_factor.is_power_of_two());
        let log_lde_factor = lde_factor.trailing_zeros();
        let domain_size = precomputations.lde_precomputations.domain_size;
        assert!(domain_size.is_power_of_two());
        let log_domain_size = domain_size.trailing_zeros();
        let log_tree_cap_size = get_tree_cap_size(log_domain_size);
        let circuit = &precomputations.compiled_circuit;
        let trees_and_caps = precomputations.setup_trees_and_caps.wait().clone();
        let mut setup = SetupPrecomputations::new(
            circuit,
            log_lde_factor,
            log_tree_cap_size,
            false,
            trees_and_caps,
            &context,
        )?;
        match circuit_type {
            CircuitType::Main(main) => trace!("GPU_WORKER[{device_id}] transferring setup trace for main circuit {main:?}"),
            CircuitType::Delegation(delegation) => trace!("GPU_WORKER[{device_id}] transferring setup trace for delegation circuit {delegation:?}"),
        }
        let trace = precomputations.setup_trace.clone();
        setup.schedule_transfer(trace.clone(), &context)?;
        match circuit_type {
            CircuitType::Main(main) => {
                trace!("GPU_WORKER[{device_id}] generating setup for main circuit {main:?}")
            }
            CircuitType::Delegation(delegation) => trace!(
                "GPU_WORKER[{device_id}] generating setup for delegation circuit {delegation:?}"
            ),
        }
        setup.ensure_is_extended(&context)?;
        // On Metal, command buffers complete synchronously when using commit_and_wait
        let setup = Rc::new(RefCell::new(setup));
        let holder = SetupHolder { setup, trace };
        setups.push(holder);
    }

    is_initialized.send(()).unwrap();
    drop(is_initialized);

    for request in requests {
        let result = if let Some(request) = request {
            let (batch_id, circuit_type, circuit_sequence) = match &request {
                GpuWorkRequest::MemoryCommitment(r) => (r.batch_id, r.circuit_type, r.circuit_sequence),
                GpuWorkRequest::Proof(r) => (r.batch_id, r.circuit_type, r.circuit_sequence),
            };
            match circuit_type {
                CircuitType::Main(main) => trace!(
                    "BATCH[{batch_id}] GPU_WORKER[{device_id}] processing {:?} for main circuit {main:?} chunk {circuit_sequence}",
                    match &request { GpuWorkRequest::MemoryCommitment(_) => "memory commitment", GpuWorkRequest::Proof(_) => "proof" }
                ),
                CircuitType::Delegation(delegation) => trace!(
                    "BATCH[{batch_id}] GPU_WORKER[{device_id}] processing {:?} for delegation circuit {delegation:?} chunk {circuit_sequence}",
                    match &request { GpuWorkRequest::MemoryCommitment(_) => "memory commitment", GpuWorkRequest::Proof(_) => "proof" }
                ),
            }

            match request {
                GpuWorkRequest::MemoryCommitment(request) => {
                    let MemoryCommitmentRequest {
                        batch_id,
                        circuit_type,
                        circuit_sequence,
                        precomputations,
                        tracing_data,
                    } = request;

                    let lde_factor = precomputations.lde_precomputations.lde_factor;
                    assert!(lde_factor.is_power_of_two());
                    let _log_lde_factor = lde_factor.trailing_zeros();
                    let domain_size = precomputations.lde_precomputations.domain_size;
                    assert!(domain_size.is_power_of_two());
                    let log_domain_size = domain_size.trailing_zeros();
                    let _log_tree_cap_size = get_tree_cap_size(log_domain_size);

                    let circuit = &precomputations.compiled_circuit;
                    let mut transfer = TracingDataTransfer::new(
                        circuit_type,
                        tracing_data.clone(),
                        &context,
                    )?;
                    transfer.schedule_transfer(&context)?;
                    let commitment = crate::prover::memory::commit_memory(
                        transfer,
                        circuit,
                        _log_lde_factor,
                        _log_tree_cap_size,
                        &context,
                    )?;
                    let (merkle_tree_caps, _elapsed) = commitment.finish()?;

                    let result = MemoryCommitmentResult {
                        batch_id,
                        circuit_type,
                        circuit_sequence,
                        tracing_data,
                        merkle_tree_caps,
                    };
                    Some(WorkerResult::MemoryCommitment(result))
                }
                GpuWorkRequest::Proof(request) => {
                    let ProofRequest {
                        batch_id,
                        circuit_type,
                        circuit_sequence,
                        precomputations,
                        tracing_data,
                        external_challenges,
                    } = request;

                    let lde_factor = precomputations.lde_precomputations.lde_factor;
                    let trees_cache_mode = get_trees_cache_mode(circuit_type, &context);

                    let setup_holder = setups
                        .iter()
                        .find(|s| Arc::ptr_eq(&s.trace, &precomputations.setup_trace))
                        .expect("setup not found for circuit");

                    let aux_boundary_values = match &tracing_data {
                        TracingDataHost::Main {
                            setup_and_teardown,
                            ..
                        } => {
                            if let Some(sat) = setup_and_teardown {
                                let cycles_per_circuit = precomputations.compiled_circuit.trace_len - 1;
                                crate::witness::trace_main::get_aux_arguments_boundary_values(
                                    &sat.lazy_init_data,
                                    cycles_per_circuit,
                                )
                            } else {
                                AuxArgumentsBoundaryValues::default()
                            }
                        }
                        TracingDataHost::Delegation(_) => AuxArgumentsBoundaryValues::default(),
                    };
                    let external_values = ExternalValues {
                        challenges: external_challenges,
                        aux_boundary_values,
                    };

                    let delegation_processing_type = match circuit_type {
                        CircuitType::Main(_) => None,
                        CircuitType::Delegation(d) => Some(d as u16),
                    };
                    let actual_circuit_sequence = match circuit_type {
                        CircuitType::Main(_) => circuit_sequence,
                        CircuitType::Delegation(_) => 0,
                    };

                    let mut tracing_data_transfer = crate::prover::tracing_data::TracingDataTransfer::new(
                        circuit_type, tracing_data.clone(), &context,
                    )?;
                    tracing_data_transfer.schedule_transfer(&context)?;
                    let proof_job = crate::prover::proof::prove(
                        precomputations.compiled_circuit.clone(),
                        external_values,
                        &mut setup_holder.setup.borrow_mut(),
                        tracing_data_transfer,
                        &precomputations.lde_precomputations,
                        actual_circuit_sequence,
                        delegation_processing_type,
                        lde_factor,
                        NUM_QUERIES,
                        POW_BITS,
                        None,
                        false,
                        trees_cache_mode,
                        &context,
                    )?;

                    let (proof, _time_ms) = proof_job.finish()?;

                    let result = ProofResult {
                        batch_id,
                        circuit_type,
                        circuit_sequence,
                        tracing_data,
                        proof,
                    };
                    Some(WorkerResult::Proof(result))
                }
            }
        } else {
            None
        };
        results.send(result).unwrap();
    }

    trace!("GPU_WORKER[{device_id}] finished");
    Ok(())
}
