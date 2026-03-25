use crate::device_context::DeviceContext;
use crate::metal_runtime::{
    system_default_device, init_shader_library, MetalBuffer, MetalCommandBuffer,
    MetalCommandQueue, MetalResult, MetalStream,
};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

pub struct DeviceProperties {
    pub max_threadgroup_memory_length: usize,
    pub max_threads_per_threadgroup: usize,
    pub name: String,
}

impl DeviceProperties {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let props = crate::metal_runtime::DeviceProperties::query(device);
        Self {
            max_threadgroup_memory_length: props.max_threadgroup_memory_length,
            max_threads_per_threadgroup: props.max_threads_per_threadgroup,
            name: props.name,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ProverContextConfig {
    pub powers_of_w_coarse_log_count: u32,
    /// Block size for device allocations (compatibility with CUDA version).
    /// On Metal, this is largely unused since we use unified memory.
    pub allocation_block_log_size: u32,
    pub device_slack_blocks_count: u32,
}

impl Default for ProverContextConfig {
    fn default() -> Self {
        Self {
            powers_of_w_coarse_log_count: 12,
            allocation_block_log_size: 22,    // 4 MB blocks (CUDA compat)
            device_slack_blocks_count: 2,
        }
    }
}

use std::sync::atomic::{AtomicBool, Ordering};
static HOST_ALLOCATOR_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Metal prover context, replacing CUDA's ProverContext.
///
/// Key differences from CUDA:
/// - No separate device/host allocators (Metal has unified memory)
/// - Command queue replaces CUDA streams
/// - MetalBuffer replaces DeviceAllocation
pub struct ProverContext {
    device_context: DeviceContext,
    ntt_twiddles: crate::ntt::NttTwiddleData,
    command_queue: MetalCommandQueue,
    device: &'static ProtocolObject<dyn MTLDevice>,
    device_properties: DeviceProperties,
}

impl ProverContext {
    /// Check if the global host allocator is initialized (CUDA compat).
    /// On Metal, always returns true (unified memory, no separate host allocator).
    pub fn is_global_host_allocator_initialized() -> bool {
        HOST_ALLOCATOR_INITIALIZED.load(Ordering::SeqCst)
    }

    /// Initialize the global host allocator (CUDA compat).
    /// On Metal, this is a no-op since unified memory is used.
    pub fn initialize_global_host_allocator(
        _num_blocks: usize,
        _block_size: usize,
        _block_log_size: u32,
    ) -> MetalResult<()> {
        HOST_ALLOCATOR_INITIALIZED.store(true, Ordering::SeqCst);
        Ok(())
    }

    pub fn new(config: &ProverContextConfig) -> MetalResult<Self> {
        let device = system_default_device()?;
        init_shader_library(device)?;

        let command_queue = MetalCommandQueue::new(device)?;
        let device_context =
            DeviceContext::create(device, config.powers_of_w_coarse_log_count)?;
        let ntt_twiddles =
            crate::ntt::NttTwiddleData::from_device_context(&device_context, device)?;
        let device_properties = DeviceProperties::new(device);

        log::info!(
            "Metal prover context initialized on device: {}",
            device_properties.name
        );

        Ok(Self {
            device_context,
            ntt_twiddles,
            command_queue,
            device,
            device_properties,
        })
    }

    pub fn device(&self) -> &'static ProtocolObject<dyn MTLDevice> {
        self.device
    }

    pub fn command_queue(&self) -> &MetalCommandQueue {
        &self.command_queue
    }

    pub fn device_context(&self) -> &DeviceContext {
        &self.device_context
    }

    pub fn device_properties(&self) -> &DeviceProperties {
        &self.device_properties
    }

    pub fn ntt_twiddles(&self) -> &crate::ntt::NttTwiddleData {
        &self.ntt_twiddles
    }

    /// Create a new command buffer (replaces CUDA stream operations).
    pub fn new_command_buffer(&self) -> MetalResult<MetalCommandBuffer> {
        self.command_queue.new_command_buffer()
    }

    /// Create a new MetalStream (exec stream equivalent).
    pub fn exec_stream(&self) -> MetalStream<'_> {
        MetalStream::new(&self.command_queue)
    }

    /// Allocate a MetalBuffer of `len` elements.
    pub fn alloc<T>(&self, len: usize) -> MetalResult<MetalBuffer<T>> {
        MetalBuffer::alloc(self.device, len)
    }

    /// Allocate a MetalBuffer from a host slice.
    pub fn alloc_from_slice<T>(&self, data: &[T]) -> MetalResult<MetalBuffer<T>> {
        MetalBuffer::from_slice(self.device, data)
    }
}

// ---- Unsafe accessor types for callback closures ----
// These mirror the CUDA version's UnsafeAccessor/UnsafeMutAccessor.
// They allow closures scheduled on Metal command buffer completion handlers
// to access data that outlives the borrow checker's reach.

#[repr(transparent)]
pub(crate) struct UnsafeAccessor<T: ?Sized>(*const T);

impl<T: ?Sized> UnsafeAccessor<T> {
    pub fn new(value: &T) -> Self {
        UnsafeAccessor(value as *const T)
    }

    pub unsafe fn get(&self) -> &T {
        &*self.0
    }
}

impl<T: ?Sized> Clone for UnsafeAccessor<T> {
    fn clone(&self) -> Self {
        UnsafeAccessor(self.0)
    }
}

impl<T: ?Sized> Copy for UnsafeAccessor<T> {}

unsafe impl<T: ?Sized> Send for UnsafeAccessor<T> {}
unsafe impl<T: ?Sized> Sync for UnsafeAccessor<T> {}

#[repr(transparent)]
pub(crate) struct UnsafeMutAccessor<T: ?Sized>(*mut T);

impl<T: ?Sized> UnsafeMutAccessor<T> {
    pub fn new(value: &mut T) -> Self {
        UnsafeMutAccessor(value as *mut T)
    }

    pub unsafe fn get(&self) -> &T {
        &*self.0
    }

    pub unsafe fn get_mut(&self) -> &mut T {
        &mut *(self.0)
    }

    pub unsafe fn set(&self, value: T)
    where
        T: Sized,
    {
        *(self.0) = value;
    }
}

impl<T: ?Sized> Clone for UnsafeMutAccessor<T> {
    fn clone(&self) -> Self {
        UnsafeMutAccessor(self.0)
    }
}

impl<T: ?Sized> Copy for UnsafeMutAccessor<T> {}

unsafe impl<T: ?Sized> Send for UnsafeMutAccessor<T> {}
unsafe impl<T: ?Sized> Sync for UnsafeMutAccessor<T> {}

/// Host-side allocation wrapper, analogous to the CUDA version's HostAllocation.
/// On Metal with unified memory, this is just a Box<T> — no pinned memory needed.
pub(crate) struct HostAllocation<T: ?Sized>(pub(crate) Box<T>);

impl<T: Sized> HostAllocation<T> {
    pub fn new(value: T) -> Self {
        Self(Box::new(value))
    }

    pub unsafe fn new_uninit() -> Self {
        Self(Box::new_uninit().assume_init())
    }

    pub fn get_accessor(&self) -> UnsafeAccessor<T> {
        UnsafeAccessor::new(&self.0)
    }

    pub fn get_mut_accessor(&mut self) -> UnsafeMutAccessor<T> {
        UnsafeMutAccessor::new(&mut self.0)
    }
}

impl<T: ?Sized> std::ops::Deref for HostAllocation<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: ?Sized> std::ops::DerefMut for HostAllocation<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> HostAllocation<[T]> {
    pub unsafe fn new_uninit_slice(len: usize) -> Self {
        Self(Box::new_uninit_slice(len).assume_init())
    }

    pub fn get_accessor(&self) -> UnsafeAccessor<[T]> {
        UnsafeAccessor::new(&self.0)
    }

    pub fn get_mut_accessor(&mut self) -> UnsafeMutAccessor<[T]> {
        UnsafeMutAccessor::new(&mut self.0)
    }
}
