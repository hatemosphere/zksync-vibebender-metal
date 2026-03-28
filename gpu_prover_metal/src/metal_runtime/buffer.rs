use super::error::{MetalError, MetalResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::marker::PhantomData;
use std::mem;
use std::ptr::NonNull;
use std::slice;

/// Typed wrapper around `MTLBuffer`, analogous to `DeviceAllocation<T>` from `era_cudart`.
///
/// On Apple Silicon, buffers use shared memory (unified CPU/GPU address space),
/// so no explicit host-to-device copies are needed for `StorageModeShared` buffers.
pub struct MetalBuffer<T> {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> MetalBuffer<T> {
    /// Allocate a new buffer with `len` elements, uninitialized.
    /// Analogous to `DeviceAllocation::alloc(len)`.
    pub fn alloc(
        device: &ProtocolObject<dyn MTLDevice>,
        len: usize,
    ) -> MetalResult<Self> {
        let byte_len = len * mem::size_of::<T>();
        if byte_len == 0 {
            return Self::alloc_bytes(device, mem::size_of::<T>().max(1), 0);
        }
        Self::alloc_bytes(device, byte_len, len)
    }

    fn alloc_bytes(
        device: &ProtocolObject<dyn MTLDevice>,
        byte_len: usize,
        len: usize,
    ) -> MetalResult<Self> {
        let buffer = device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                MetalError::ResourceCreationFailed(format!(
                    "Failed to allocate Metal buffer of {byte_len} bytes"
                ))
            })?;
        Ok(Self {
            buffer,
            len,
            _marker: PhantomData,
        })
    }

    /// Create a buffer initialized with data from a slice.
    pub fn from_slice(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[T],
    ) -> MetalResult<Self> {
        let byte_len = data.len() * mem::size_of::<T>();
        if byte_len == 0 {
            return Self::alloc(device, 0);
        }
        let ptr = std::ptr::NonNull::new(data.as_ptr() as *mut std::ffi::c_void).unwrap();
        let buffer = unsafe {
            device.newBufferWithBytes_length_options(
                ptr,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
        }
            .ok_or_else(|| {
                MetalError::ResourceCreationFailed(format!(
                    "Failed to create Metal buffer from slice of {byte_len} bytes"
                ))
            })?;
        Ok(Self {
            buffer,
            len: data.len(),
            _marker: PhantomData,
        })
    }

    /// Wrap an existing host slice without copying.
    ///
    /// # Safety
    /// The caller must ensure the backing allocation outlives the returned MetalBuffer.
    pub unsafe fn from_slice_no_copy(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[T],
    ) -> MetalResult<Self> {
        let byte_len = data.len() * mem::size_of::<T>();
        if byte_len == 0 {
            return Self::alloc(device, 0);
        }
        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void).unwrap();
        let buffer = device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                byte_len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .ok_or_else(|| {
                MetalError::ResourceCreationFailed(format!(
                    "Failed to create no-copy Metal buffer from slice of {byte_len} bytes"
                ))
            })?;
        Ok(Self {
            buffer,
            len: data.len(),
            _marker: PhantomData,
        })
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte length of the buffer.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * mem::size_of::<T>()
    }

    /// Raw pointer to the buffer contents (CPU-accessible for shared storage mode).
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.buffer.contents().cast::<T>().as_ptr()
    }

    /// Mutable raw pointer to the buffer contents.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.buffer.contents().cast::<T>().as_ptr()
    }

    /// View buffer contents as a slice. Safe because shared storage mode
    /// guarantees CPU accessibility. Caller must ensure no concurrent GPU writes.
    ///
    /// # Safety
    /// Caller must ensure the GPU is not concurrently writing to this buffer.
    pub unsafe fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        slice::from_raw_parts(self.as_ptr(), self.len)
    }

    /// View buffer contents as a mutable slice.
    ///
    /// # Safety
    /// Caller must ensure no concurrent GPU access to this buffer.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        slice::from_raw_parts_mut(self.as_mut_ptr(), self.len)
    }

    /// Copy data from a host slice into this buffer.
    ///
    /// # Safety
    /// Caller must ensure no concurrent GPU access to this buffer.
    pub unsafe fn copy_from_slice(&mut self, src: &[T]) {
        assert_eq!(src.len(), self.len, "Source slice length mismatch");
        if self.len == 0 {
            return;
        }
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.as_mut_ptr(), self.len);
    }

    /// Copy data from this buffer into a host slice.
    ///
    /// # Safety
    /// Caller must ensure the GPU has finished writing to this buffer.
    pub unsafe fn copy_to_slice(&self, dst: &mut [T]) {
        assert_eq!(dst.len(), self.len, "Destination slice length mismatch");
        if self.len == 0 {
            return;
        }
        std::ptr::copy_nonoverlapping(self.as_ptr(), dst.as_mut_ptr(), self.len);
    }

    /// Get a reference to the underlying `MTLBuffer` for passing to encoders.
    #[inline]
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Split this buffer view into two at `mid`, analogous to `DeviceSlice::split_at`.
    /// Returns (MetalBufferSlice for [0..mid], MetalBufferSlice for [mid..len]).
    pub fn split_at(&self, mid: usize) -> (MetalBufferSlice<'_, T>, MetalBufferSlice<'_, T>) {
        assert!(mid <= self.len);
        let left = MetalBufferSlice {
            buffer: &self.buffer,
            offset: 0,
            len: mid,
            _marker: PhantomData,
        };
        let right = MetalBufferSlice {
            buffer: &self.buffer,
            offset: mid * mem::size_of::<T>(),
            len: self.len - mid,
            _marker: PhantomData,
        };
        (left, right)
    }

    /// Reinterpret this buffer as a different element type.
    /// The new element count is computed from the byte length.
    ///
    /// # Safety
    /// - `size_of::<T>() * self.len` must be divisible by `size_of::<U>()`
    /// - The memory layout of `U` must be compatible with how the data will be used.
    pub unsafe fn transmute_view<U>(&self) -> MetalBuffer<U> {
        let byte_len = self.len * mem::size_of::<T>();
        let new_len = byte_len / mem::size_of::<U>();
        assert_eq!(
            byte_len % mem::size_of::<U>(),
            0,
            "Buffer byte length not divisible by target element size"
        );
        MetalBuffer {
            buffer: self.buffer.clone(),
            len: new_len,
            _marker: PhantomData,
        }
    }

    /// Get the full buffer as a slice reference.
    pub fn as_buffer_slice(&self) -> MetalBufferSlice<'_, T> {
        MetalBufferSlice {
            buffer: &self.buffer,
            offset: 0,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

// SAFETY: Metal buffers in shared storage mode are safe to send between threads.
// The Metal runtime handles synchronization internally.
unsafe impl<T: Send> Send for MetalBuffer<T> {}
unsafe impl<T: Sync> Sync for MetalBuffer<T> {}

/// A borrowed sub-region of a `MetalBuffer`, analogous to `&DeviceSlice<T>`.
///
/// Carries the underlying `MTLBuffer` reference plus a byte offset and element count,
/// enabling zero-copy sub-buffer views for kernel argument encoding.
pub struct MetalBufferSlice<'a, T> {
    buffer: &'a ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    len: usize,
    _marker: PhantomData<T>,
}

impl<'a, T> MetalBufferSlice<'a, T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte offset into the parent buffer.
    #[inline]
    pub fn byte_offset(&self) -> usize {
        self.offset
    }

    /// Reference to the underlying MTLBuffer (for encoder.setBuffer).
    #[inline]
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.buffer
    }

    pub fn split_at(self, mid: usize) -> (MetalBufferSlice<'a, T>, MetalBufferSlice<'a, T>) {
        assert!(mid <= self.len);
        let left = MetalBufferSlice {
            buffer: self.buffer,
            offset: self.offset,
            len: mid,
            _marker: PhantomData,
        };
        let right = MetalBufferSlice {
            buffer: self.buffer,
            offset: self.offset + mid * mem::size_of::<T>(),
            len: self.len - mid,
            _marker: PhantomData,
        };
        (left, right)
    }
}

impl<T> Clone for MetalBufferSlice<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for MetalBufferSlice<'_, T> {}
