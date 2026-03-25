use super::error::{MetalError, MetalResult};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLDevice};
use std::cell::RefCell;

/// Wrapper around `MTLCommandQueue`.
pub struct MetalCommandQueue {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

/// Maximum non-completed command buffers the queue will allow.
/// Limits GPU queue depth at the Metal API level, ensuring the display
/// compositor gets scheduling windows between compute submissions.
/// With synchronous flush (waitUntilCompleted), at most 1 buffer executes
/// at a time, but this provides a hard backstop.
const MAX_COMMAND_BUFFER_COUNT: usize = 2;

impl MetalCommandQueue {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> MetalResult<Self> {
        let queue = device
            .newCommandQueueWithMaxCommandBufferCount(MAX_COMMAND_BUFFER_COUNT)
            .ok_or_else(|| {
                MetalError::ResourceCreationFailed("Failed to create Metal command queue".into())
            })?;
        Ok(Self { queue })
    }

    pub fn new_command_buffer(&self) -> MetalResult<MetalCommandBuffer> {
        MetalCommandBuffer::new(&self.queue)
    }

    #[inline]
    pub fn raw(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }
}

/// Command buffer wrapper with auto-commit batching for GPU watchdog safety.
///
/// Batches kernel dispatches into command buffers of bounded size, using
/// synchronous commit+wait to ensure at most ONE buffer is executing on the
/// GPU at any time. This means:
/// - Each GPU submission is bounded to ~1-2s (well under the 5s watchdog)
/// - When the process is killed, the GPU is freed within ~2s
/// - The display compositor gets scheduling windows between submissions
///
/// Buffer creation is lazy to avoid exhausting the Metal command buffer pool.
pub struct MetalCommandBuffer {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// None after flush; created lazily on next dispatch.
    buffer: RefCell<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    /// Number of dispatches in current batch.
    dispatch_count: RefCell<u32>,
}

/// Maximum dispatches per command buffer before auto-commit (watchdog safety).
/// Each command buffer should complete within ~2s (well under 5s macOS GPU watchdog).
/// Higher = fewer commit boundaries = less overhead, but risk longer individual submissions.
const MAX_DISPATCHES_PER_BATCH: u32 = 64;

impl MetalCommandBuffer {
    fn new(queue: &ProtocolObject<dyn MTLCommandQueue>) -> MetalResult<Self> {
        let queue_retained = unsafe {
            Retained::retain(queue as *const _ as *mut ProtocolObject<dyn MTLCommandQueue>)
        }
        .ok_or_else(|| {
            MetalError::ResourceCreationFailed("Failed to retain command queue".into())
        })?;
        Ok(Self {
            queue: queue_retained,
            buffer: RefCell::new(None),
            dispatch_count: RefCell::new(0),
        })
    }

    /// Ensure there is a live command buffer, creating one lazily if needed.
    fn ensure_buffer(&self) {
        let mut buf = self.buffer.borrow_mut();
        if buf.is_none() {
            let new_buf = self
                .queue
                .commandBuffer()
                .expect("Failed to create Metal command buffer");
            *buf = Some(new_buf);
        }
    }

    /// Called after each kernel dispatch to check if we should auto-commit.
    /// Uses commit-without-wait: submits the buffer to the GPU but does NOT
    /// block the CPU. Metal queues are FIFO, so the next buffer will start
    /// after this one completes. This gives WindowServer scheduling windows
    /// between batches without the CPU sync overhead.
    pub(crate) fn maybe_auto_commit(&self) {
        let count = {
            let mut c = self.dispatch_count.borrow_mut();
            *c += 1;
            *c
        };

        if count >= MAX_DISPATCHES_PER_BATCH {
            self.commit_no_wait();
        }
    }

    /// Commit current buffer WITHOUT waiting — fire-and-forget.
    /// The GPU processes it asynchronously. Safe because Metal queues are FIFO:
    /// the next command buffer will start after this one finishes.
    fn commit_no_wait(&self) {
        autoreleasepool(|_| {
            let mut buf = self.buffer.borrow_mut();
            if let Some(b) = buf.take() {
                b.commit();
                // For profiling: wait and capture GPU timestamps even on auto-commit
                #[cfg(feature = "log_gpu_stages_timings")]
                {
                    b.waitUntilCompleted();
                    let gpu_start = b.GPUStartTime();
                    let gpu_end = b.GPUEndTime();
                    super::profiler::record_commit(gpu_start, gpu_end);
                }
            }
        });
        *self.dispatch_count.borrow_mut() = 0;
    }

    /// Commit current buffer and WAIT for completion.
    /// Only call this when the CPU needs to read GPU results.
    fn flush(&self) {
        autoreleasepool(|_| {
            let mut buf = self.buffer.borrow_mut();
            if let Some(b) = buf.take() {
                b.commit();
                b.waitUntilCompleted();
                #[cfg(feature = "log_gpu_stages_timings")]
                {
                    let gpu_start = b.GPUStartTime();
                    let gpu_end = b.GPUEndTime();
                    super::profiler::record_commit(gpu_start, gpu_end);
                }
            }
        });
        *self.dispatch_count.borrow_mut() = 0;
    }

    /// Commit and wait for all pending work. Use when CPU needs GPU results.
    pub fn commit_and_wait(&self) {
        if *self.dispatch_count.borrow() > 0 {
            self.flush();
        }
    }

    /// Commit without waiting.
    pub fn commit(&self) {
        if let Some(b) = self.buffer.borrow().as_ref() {
            b.commit();
        }
    }

    /// Wait for completion of current buffer.
    pub fn wait_until_completed(&self) {
        if let Some(b) = self.buffer.borrow().as_ref() {
            b.waitUntilCompleted();
        }
    }

    /// Get the underlying MTLCommandBuffer for encoding.
    /// Creates a buffer lazily if needed.
    #[inline]
    pub fn raw(&self) -> std::cell::Ref<'_, ProtocolObject<dyn MTLCommandBuffer>> {
        self.ensure_buffer();
        std::cell::Ref::map(self.buffer.borrow(), |b| b.as_deref().unwrap())
    }

    /// Check status.
    pub fn status(&self) -> MetalResult<()> {
        if let Some(b) = self.buffer.borrow().as_ref() {
            if let Some(err) = b.error() {
                return Err(MetalError::CommandBufferError(err.to_string()));
            }
        }
        Ok(())
    }
}
