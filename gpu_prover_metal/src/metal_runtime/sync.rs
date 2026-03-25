use super::command_queue::MetalCommandBuffer;
use super::error::{MetalError, MetalResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLDevice, MTLEvent};

/// Wrapper around `MTLEvent`, analogous to `CudaEvent`.
///
/// Used for synchronization between command buffers (the Metal equivalent of
/// inter-stream synchronization in CUDA).
///
/// # Usage Pattern (mirrors CUDA stream events)
///
/// ```ignore
/// // CUDA:
/// //   event.record(stream_a)?;
/// //   stream_b.wait_event(&event)?;
///
/// // Metal:
/// //   event.signal(&cmd_buf_a, value);
/// //   event.wait(&cmd_buf_b, value);
/// ```
pub struct MetalEvent {
    event: Retained<ProtocolObject<dyn MTLEvent>>,
    /// Monotonically increasing counter for signal/wait pairs.
    counter: std::sync::atomic::AtomicU64,
}

impl MetalEvent {
    /// Create a new event on the given device.
    /// Analogous to `CudaEvent::create()`.
    pub fn create(device: &ProtocolObject<dyn MTLDevice>) -> MetalResult<Self> {
        let event = device.newEvent().ok_or_else(|| {
            MetalError::ResourceCreationFailed("Failed to create Metal event".into())
        })?;
        Ok(Self {
            event,
            counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Signal this event from a command buffer at the next counter value.
    /// Analogous to `CudaEvent::record(&stream)`.
    ///
    /// Returns the signal value for use with `wait`.
    pub fn signal(&self, cmd_buf: &MetalCommandBuffer) -> u64 {
        let value = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        cmd_buf.raw().encodeSignalEvent_value(&self.event, value);
        value
    }

    /// Wait for this event to reach the given value before executing further
    /// commands in the given command buffer.
    /// Analogous to `CudaStream::wait_event(&event)`.
    pub fn wait(&self, cmd_buf: &MetalCommandBuffer, value: u64) {
        cmd_buf.raw().encodeWaitForEvent_value(&self.event, value);
    }

    /// Signal from one command buffer and wait on another — the common
    /// pattern for inter-stream synchronization.
    ///
    /// Equivalent to:
    /// ```ignore
    /// let val = event.signal(producer);
    /// event.wait(consumer, val);
    /// ```
    pub fn signal_and_wait(
        &self,
        producer: &MetalCommandBuffer,
        consumer: &MetalCommandBuffer,
    ) -> u64 {
        let value = self.signal(producer);
        self.wait(consumer, value);
        value
    }

    /// Get the current counter value.
    pub fn current_value(&self) -> u64 {
        self.counter.load(std::sync::atomic::Ordering::SeqCst)
    }
}
