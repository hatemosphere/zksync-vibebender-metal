use super::buffer::MetalBuffer;
use super::command_queue::MetalCommandBuffer;
use super::error::MetalResult;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLDevice};

pub fn buffer_copy_raw(
    cmd_buf: &MetalCommandBuffer,
    src: &ProtocolObject<dyn MTLBuffer>,
    src_offset: usize,
    dst: &ProtocolObject<dyn MTLBuffer>,
    dst_offset: usize,
    byte_len: usize,
) -> MetalResult<()> {
    if byte_len == 0 {
        return Ok(());
    }
    let encoder = cmd_buf.raw().blitCommandEncoder().ok_or_else(|| {
        super::error::MetalError::ResourceCreationFailed(
            "Failed to create blit command encoder".into(),
        )
    })?;
    unsafe {
        encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            src,
            src_offset,
            dst,
            dst_offset,
            byte_len,
        );
    }
    encoder.endEncoding();
    Ok(())
}

/// Copy data between two Metal buffers using a blit encoder.
///
/// Analogous to `era_cudart::memory::memory_copy_async` for device-to-device copies.
/// On Apple Silicon with shared storage mode, this can also be done via direct
/// pointer copies, but the blit encoder approach is GPU-scheduled.
pub fn buffer_copy<T>(
    cmd_buf: &MetalCommandBuffer,
    src: &MetalBuffer<T>,
    dst: &mut MetalBuffer<T>,
) -> MetalResult<()> {
    assert_eq!(src.len(), dst.len(), "Buffer length mismatch in copy");
    let byte_len = src.byte_len();
    if byte_len == 0 {
        return Ok(());
    }
    buffer_copy_raw(cmd_buf, src.raw(), 0, dst.raw(), 0, byte_len)
}

/// Fill a Metal buffer with zeros using a blit encoder.
///
/// Analogous to `era_cudart::memory::memory_set_async(dst, 0, stream)`.
pub fn buffer_fill_zeros<T>(
    cmd_buf: &MetalCommandBuffer,
    dst: &mut MetalBuffer<T>,
) -> MetalResult<()> {
    let byte_len = dst.byte_len();
    if byte_len == 0 {
        return Ok(());
    }
    let encoder = cmd_buf.raw().blitCommandEncoder().ok_or_else(|| {
        super::error::MetalError::ResourceCreationFailed(
            "Failed to create blit command encoder".into(),
        )
    })?;
    encoder.fillBuffer_range_value(dst.raw(), objc2_foundation::NSRange::new(0, byte_len), 0);
    encoder.endEncoding();
    Ok(())
}

/// Synchronous copy from host slice to Metal buffer (direct memcpy).
///
/// Analogous to `era_cudart::memory::memory_copy` (host-to-device, synchronous).
/// Works because Apple Silicon uses unified memory with shared storage mode.
///
/// # Safety
/// Caller must ensure no concurrent GPU access to `dst`.
pub unsafe fn copy_host_to_buffer<T>(dst: &mut MetalBuffer<T>, src: &[T]) {
    dst.copy_from_slice(src);
}

/// Synchronous copy from Metal buffer to host slice (direct memcpy).
///
/// Analogous to `era_cudart::memory::memory_copy` (device-to-host, synchronous).
///
/// # Safety
/// Caller must ensure GPU has finished writing to `src`.
pub unsafe fn copy_buffer_to_host<T>(src: &MetalBuffer<T>, dst: &mut [T]) {
    src.copy_to_slice(dst);
}

/// Query available memory. On Apple Silicon, this returns the recommended
/// working set size as there's no separate "device memory".
///
/// Analogous to `era_cudart::memory::memory_get_info`.
pub fn memory_get_info(device: &ProtocolObject<dyn MTLDevice>) -> (usize, usize) {
    let recommended = device.recommendedMaxWorkingSetSize() as usize;
    let max_alloc = device.maxBufferLength() as usize;
    (recommended, max_alloc)
}
