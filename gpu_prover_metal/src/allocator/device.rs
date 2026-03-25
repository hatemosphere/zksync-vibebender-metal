use crate::allocator::{
    ConcurrentInnerStaticAllocatorWrapper, InnerStaticAllocatorWrapper,
    NonConcurrentInnerStaticAllocatorWrapper, StaticAllocation, StaticAllocationBackend,
    StaticAllocator,
};
use crate::metal_runtime::{MetalBuffer, MetalResult};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;

/// Metal device allocation backend.
/// Wraps a large MetalBuffer<u8> that sub-allocations are carved from.
/// On Apple Silicon with unified memory, this buffer is directly CPU-accessible.
pub struct StaticDeviceAllocationBackend {
    buffer: MetalBuffer<u8>,
}

impl StaticDeviceAllocationBackend {
    pub fn new(buffer: MetalBuffer<u8>) -> Self {
        Self { buffer }
    }

    pub fn alloc(
        device: &ProtocolObject<dyn MTLDevice>,
        len: usize,
    ) -> MetalResult<Self> {
        let buffer = MetalBuffer::alloc(device, len)?;
        Ok(Self { buffer })
    }
}

impl StaticAllocationBackend for StaticDeviceAllocationBackend {
    fn as_non_null(&mut self) -> NonNull<u8> {
        unsafe { NonNull::new_unchecked(self.buffer.as_mut_ptr()) }
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

trait InnerStaticDeviceAllocatorWrapper:
    InnerStaticAllocatorWrapper<StaticDeviceAllocationBackend>
{
}

type ConcurrentInnerStaticDeviceAllocatorWrapper =
    ConcurrentInnerStaticAllocatorWrapper<StaticDeviceAllocationBackend>;

impl InnerStaticDeviceAllocatorWrapper for ConcurrentInnerStaticDeviceAllocatorWrapper {}

type NonConcurrentInnerStaticDeviceAllocatorWrapper =
    NonConcurrentInnerStaticAllocatorWrapper<StaticDeviceAllocationBackend>;

impl InnerStaticDeviceAllocatorWrapper for NonConcurrentInnerStaticDeviceAllocatorWrapper {}

type StaticDeviceAllocator<W> = StaticAllocator<StaticDeviceAllocationBackend, W>;

type StaticDeviceAllocation<T, W> = StaticAllocation<T, StaticDeviceAllocationBackend, W>;

pub type ConcurrentStaticDeviceAllocator =
    StaticDeviceAllocator<ConcurrentInnerStaticDeviceAllocatorWrapper>;

pub type ConcurrentStaticDeviceAllocation<T> =
    StaticDeviceAllocation<T, ConcurrentInnerStaticDeviceAllocatorWrapper>;

pub type NonConcurrentStaticDeviceAllocator =
    StaticDeviceAllocator<NonConcurrentInnerStaticDeviceAllocatorWrapper>;

pub type NonConcurrentStaticDeviceAllocation<T> =
    StaticDeviceAllocation<T, NonConcurrentInnerStaticDeviceAllocatorWrapper>;

impl<T, W: InnerStaticDeviceAllocatorWrapper> Deref for StaticDeviceAllocation<T, W> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.data.ptr.as_ptr(), self.data.len) }
    }
}

impl<T, W: InnerStaticDeviceAllocatorWrapper> DerefMut for StaticDeviceAllocation<T, W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.data.ptr.as_ptr(), self.data.len) }
    }
}
