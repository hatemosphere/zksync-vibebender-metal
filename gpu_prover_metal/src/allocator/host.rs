use crate::allocator::allocation_data::StaticAllocationData;
use crate::allocator::tracker::AllocationPlacement;
use crate::allocator::{
    ConcurrentInnerStaticAllocatorWrapper, InnerStaticAllocatorWrapper,
    NonConcurrentInnerStaticAllocatorWrapper, StaticAllocation, StaticAllocationBackend,
    StaticAllocator,
};
use fft::GoodAllocator;
use log::error;
use std::alloc::{AllocError, Allocator, Layout};
use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;
use std::sync::OnceLock;

pub static STATIC_HOST_ALLOCATOR: OnceLock<ConcurrentStaticHostAllocator> = OnceLock::new();

/// Host allocation backend for Metal.
/// On Apple Silicon with unified memory, we use page-aligned heap allocations.
/// This replaces CUDA's HostAllocation<u8> (pinned memory) since Metal
/// doesn't need pinned memory -- all allocations are already unified.
pub struct HostAllocation {
    ptr: NonNull<u8>,
    len: usize,
}

impl HostAllocation {
    pub fn alloc(len: usize) -> Self {
        assert_ne!(len, 0);
        let layout = Layout::from_size_align(len, 4096).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).expect("host allocation failed");
        Self { ptr, len }
    }
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, 4096).unwrap();
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

impl Deref for HostAllocation {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for HostAllocation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

// SAFETY: HostAllocation owns its memory and can be safely sent/shared across threads.
unsafe impl Send for HostAllocation {}
unsafe impl Sync for HostAllocation {}

impl StaticAllocationBackend for HostAllocation {
    fn as_non_null(&mut self) -> NonNull<u8> {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

trait InnerStaticHostAllocatorWrapper: InnerStaticAllocatorWrapper<HostAllocation> {}

type ConcurrentInnerStaticHostAllocatorWrapper =
    ConcurrentInnerStaticAllocatorWrapper<HostAllocation>;

impl InnerStaticHostAllocatorWrapper for ConcurrentInnerStaticHostAllocatorWrapper {}

type NonConcurrentInnerStaticHostAllocatorWrapper =
    NonConcurrentInnerStaticAllocatorWrapper<HostAllocation>;

impl InnerStaticHostAllocatorWrapper for NonConcurrentInnerStaticHostAllocatorWrapper {}

type StaticHostAllocator<W> = StaticAllocator<HostAllocation, W>;

pub type ConcurrentStaticHostAllocator =
    StaticHostAllocator<ConcurrentInnerStaticHostAllocatorWrapper>;

impl ConcurrentStaticHostAllocator {
    pub fn initialize_global(
        backends: impl IntoIterator<Item = HostAllocation>,
        log_chunk_size: u32,
    ) {
        let allocator = ConcurrentStaticHostAllocator::new(backends, log_chunk_size);
        assert!(STATIC_HOST_ALLOCATOR.set(allocator).is_ok());
    }

    pub fn get_global() -> &'static ConcurrentStaticHostAllocator {
        STATIC_HOST_ALLOCATOR.get().unwrap()
    }

    pub fn is_initialized_global() -> bool {
        STATIC_HOST_ALLOCATOR.get().is_some()
    }
}

pub type NonConcurrentStaticHostAllocator =
    StaticHostAllocator<NonConcurrentInnerStaticHostAllocatorWrapper>;

impl<T, W: InnerStaticHostAllocatorWrapper> Deref for StaticAllocation<T, HostAllocation, W> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.data.ptr.as_ptr(), self.data.len) }
    }
}

impl<T, W: InnerStaticHostAllocatorWrapper> DerefMut
    for StaticAllocation<T, HostAllocation, W>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.data.ptr.as_ptr(), self.data.len) }
    }
}

unsafe impl<W: InnerStaticHostAllocatorWrapper> Allocator for StaticHostAllocator<W> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let len = layout.size();
        if let Ok(data) = self
            .inner
            .execute(|inner| inner.alloc(len, AllocationPlacement::BestFit))
        {
            let ptr = data.ptr;
            assert!(ptr.is_aligned_to(layout.align()));
            assert_eq!(data.len, len);
            let len = data.alloc_len;
            assert_eq!(data.len.next_multiple_of(1 << self.log_chunk_size), len);
            Ok(NonNull::slice_from_raw_parts(ptr, len))
        } else {
            error!("allocation of {len} bytes in StaticHostAllocator failed");
            Err(AllocError)
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let len = layout.size();
        let alloc_len = len.next_multiple_of(1 << self.log_chunk_size);
        let data = StaticAllocationData::new(ptr, len, alloc_len);
        self.inner.execute(|inner| inner.free(data));
    }
}

impl Default for ConcurrentStaticHostAllocator {
    fn default() -> Self {
        ConcurrentStaticHostAllocator::get_global().clone()
    }
}

impl Debug for ConcurrentStaticHostAllocator {
    fn fmt(&self, _: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl GoodAllocator for ConcurrentStaticHostAllocator {}
