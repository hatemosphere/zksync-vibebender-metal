//! Data transfer management for Metal.
//! On Metal with unified memory, transfers are trivial memcpy operations.
//! CUDA's event-based host-to-device synchronization is unnecessary since
//! Metal buffers use StorageModeShared (unified CPU/GPU address space).
//! Ported from gpu_prover/src/prover/transfer.rs.

use super::callbacks::Callbacks;
use super::context::ProverContext;
use crate::metal_runtime::{MetalBuffer, MetalResult};
use std::sync::Arc;

pub struct Transfer<'a> {
    pub(crate) callbacks: Callbacks<'a>,
}

impl<'a> Transfer<'a> {
    pub(crate) fn new() -> Self {
        Self {
            callbacks: Callbacks::new(),
        }
    }

    /// No-op on Metal: allocation is immediately visible to both CPU and GPU.
    pub(crate) fn record_allocated(&self, _context: &ProverContext) -> MetalResult<()> {
        Ok(())
    }

    /// No-op on Metal: no h2d stream synchronization needed.
    pub(crate) fn ensure_allocated(&self, _context: &ProverContext) -> MetalResult<()> {
        Ok(())
    }

    /// Copy data from a host slice (wrapped in Arc) into a MetalBuffer.
    /// On Metal with unified memory, this is a direct memcpy.
    /// The Arc is captured in a callback to keep the source alive until
    /// the GPU is done with any prior work.
    pub fn schedule<T: Copy>(
        &mut self,
        src: Arc<impl AsRef<[T]> + Send + Sync + 'a>,
        dst: &mut MetalBuffer<T>,
        _context: &ProverContext,
    ) -> MetalResult<()> {
        let src_slice = src.as_ref().as_ref();
        assert_eq!(src_slice.len(), dst.len());
        unsafe {
            dst.copy_from_slice(src_slice);
        }
        // Keep the Arc alive until callbacks are dropped, matching CUDA behavior
        let f = move || {
            let _ = src.clone();
        };
        self.callbacks.schedule(f);
        Ok(())
    }

    /// No-op on Metal: transfer completes synchronously via memcpy.
    pub(crate) fn record_transferred(&self, _context: &ProverContext) -> MetalResult<()> {
        Ok(())
    }

    /// No-op on Metal: no inter-stream synchronization needed.
    pub fn ensure_transferred(&self, _context: &ProverContext) -> MetalResult<()> {
        Ok(())
    }
}
