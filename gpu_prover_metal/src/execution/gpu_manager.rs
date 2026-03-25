use super::gpu_worker::{get_gpu_worker_func, GpuWorkRequest, SetupToCache};
use super::messages::WorkerResult;
use crate::allocator::host::ConcurrentStaticHostAllocator;
use crate::prover::context::ProverContextConfig;
use crossbeam_channel::{bounded, Receiver, Sender};
use crossbeam_utils::sync::WaitGroup;
use fft::GoodAllocator;
use log::{error, info, trace};
use std::process::exit;
use std::thread;

pub struct GpuWorkBatch<A: GoodAllocator> {
    pub batch_id: u64,
    pub receiver: Receiver<GpuWorkRequest<A>>,
    pub sender: Sender<WorkerResult<A>>,
}

pub struct GpuManager {
    wait_group: Option<WaitGroup>,
    batches_sender: Option<Sender<GpuWorkBatch<ConcurrentStaticHostAllocator>>>,
}

impl GpuManager {
    pub fn new(
        setups_to_cache: Vec<SetupToCache>,
        initialized_wait_group: WaitGroup,
    ) -> Self {
        let prover_context_config = ProverContextConfig::default();
        let device_count = 1usize; // Metal: single device (Apple Silicon)
        info!("GPU_MANAGER found {device_count} Metal capable device(s)");

        let wait_group = WaitGroup::new();
        let (batches_sender, batches_receiver) = bounded::<GpuWorkBatch<ConcurrentStaticHostAllocator>>(1);

        // Spawn the manager thread
        let wg = wait_group.clone();
        thread::spawn(move || {
            run_manager(
                batches_receiver,
                setups_to_cache,
                prover_context_config,
                initialized_wait_group,
            );
            drop(wg);
        });

        Self {
            wait_group: Some(wait_group),
            batches_sender: Some(batches_sender),
        }
    }

    pub fn get_batches_sender(&self) -> &Sender<GpuWorkBatch<ConcurrentStaticHostAllocator>> {
        self.batches_sender.as_ref().unwrap()
    }

    pub fn send_batch(&self, batch: GpuWorkBatch<ConcurrentStaticHostAllocator>) {
        self.batches_sender.as_ref().unwrap().send(batch).unwrap();
    }
}

impl Drop for GpuManager {
    fn drop(&mut self) {
        // Close the channel to signal manager thread to exit
        self.batches_sender.take();
        // Wait for manager thread to finish
        if let Some(wg) = self.wait_group.take() {
            wg.wait();
        }
        trace!("GPU_MANAGER all workers finished");
    }
}

/// Simple sequential manager loop for Metal.
///
/// Unlike the CUDA version which uses a complex Select-based event loop
/// with double-buffered pipeline and async overlap, Metal's synchronous
/// execution model works best with sequential request processing:
///
/// 1. Receive a batch of work requests
/// 2. For each request in the batch:
///    a. Send request to worker
///    b. Wait for result
///    c. Send result back to batch sender
/// 3. Repeat for next batch
fn run_manager(
    batches_receiver: Receiver<GpuWorkBatch<ConcurrentStaticHostAllocator>>,
    setups_to_cache: Vec<SetupToCache>,
    prover_context_config: ProverContextConfig,
    initialized_wait_group: WaitGroup,
) {
    trace!("GPU_MANAGER spawning");

    // Create worker channel pair
    let (request_sender, request_receiver) = bounded::<Option<GpuWorkRequest<ConcurrentStaticHostAllocator>>>(1);
    let (result_sender, result_receiver) = bounded::<Option<WorkerResult<ConcurrentStaticHostAllocator>>>(1);

    // Wait for worker initialization
    let (init_sender, init_receiver) = bounded(0);

    let device_id = 0;
    let gpu_worker_func = get_gpu_worker_func(
        device_id,
        prover_context_config,
        setups_to_cache,
        init_sender,
        request_receiver,
        result_sender,
    );

    // Spawn the GPU worker thread
    thread::spawn(move || {
        if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(gpu_worker_func)) {
            error!("GPU_WORKER[0] panicked: {:?}", e);
            exit(1);
        }
    });

    // Wait for worker to initialize
    init_receiver.recv().unwrap();
    trace!("GPU_MANAGER all GPU workers initialized");
    // Signal that all setups are cached and ready
    drop(initialized_wait_group);

    // Process batches sequentially
    for batch in batches_receiver {
        let GpuWorkBatch {
            batch_id,
            receiver: work_requests,
            sender: results_sender,
        } = batch;
        trace!("BATCH[{batch_id}] GPU_MANAGER received new batch");

        // Process each work request in the batch
        for request in work_requests {
            let request_type = match &request {
                GpuWorkRequest::MemoryCommitment(_) => "memory commitment",
                GpuWorkRequest::Proof(_) => "proof",
            };
            trace!("BATCH[{batch_id}] GPU_MANAGER sending {request_type} request to worker");

            // Send request to worker
            request_sender.send(Some(request)).unwrap();

            // Wait for result
            let result = result_receiver.recv().unwrap();

            if let Some(result) = result {
                match &result {
                    WorkerResult::MemoryCommitment(_r) => {
                        trace!("BATCH[{batch_id}] GPU_MANAGER received memory commitment result");
                    }
                    WorkerResult::Proof(_r) => {
                        trace!("BATCH[{batch_id}] GPU_MANAGER received proof result");
                    }
                    _ => {}
                }
                results_sender.send(result).unwrap();
            }
        }

        trace!("BATCH[{batch_id}] GPU_MANAGER batch completed");
    }

    // Signal worker to exit
    drop(request_sender);
    trace!("GPU_MANAGER shutting down");
}
