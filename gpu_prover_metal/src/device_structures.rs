use crate::metal_runtime::MetalBuffer;
use std::marker::PhantomData;

/// Pointer-and-stride pair passed to Metal kernels as value arguments.
/// Replaces CUDA's PtrAndStride<T> / MutPtrAndStride<T>.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PtrAndStride<T> {
    pub ptr: *const T,
    pub stride: usize,
}

impl<T> PtrAndStride<T> {
    pub fn new(ptr: *const T, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MutPtrAndStride<T> {
    pub ptr: *mut T,
    pub stride: usize,
}

impl<T> MutPtrAndStride<T> {
    pub fn new(ptr: *mut T, stride: usize) -> Self {
        Self { ptr, stride }
    }
}

/// A matrix stored as a MetalBuffer with column-major layout.
pub struct DeviceMatrix<T> {
    pub buffer: MetalBuffer<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> DeviceMatrix<T> {
    pub fn new(buffer: MetalBuffer<T>, rows: usize, cols: usize) -> Self {
        assert_eq!(buffer.len(), rows * cols);
        Self { buffer, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn stride(&self) -> usize {
        self.rows
    }

    pub fn as_ptr_and_stride(&self) -> PtrAndStride<T> {
        PtrAndStride::new(self.buffer.as_ptr(), self.rows)
    }

    pub fn as_mut_ptr_and_stride(&mut self) -> MutPtrAndStride<T> {
        MutPtrAndStride::new(self.buffer.as_mut_ptr(), self.rows)
    }
}

/// A chunk (sub-view) of a DeviceMatrix.
pub struct DeviceMatrixChunk<'a, T> {
    _buffer: &'a MetalBuffer<T>,
    offset: usize,
    rows: usize,
    cols: usize,
    stride: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> DeviceMatrixChunk<'a, T> {
    pub fn new(
        buffer: &'a MetalBuffer<T>,
        offset: usize,
        rows: usize,
        cols: usize,
        stride: usize,
    ) -> Self {
        Self {
            _buffer: buffer,
            offset,
            rows,
            cols,
            stride,
            _marker: PhantomData,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    #[allow(dead_code)]
    pub fn offset(&self) -> usize {
        self.offset
    }
}
