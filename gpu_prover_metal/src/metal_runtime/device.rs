use super::error::{MetalError, MetalResult};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use std::sync::OnceLock;

// MTLCreateSystemDefaultDevice is a C function, not an ObjC method.
#[link(name = "Metal", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut ProtocolObject<dyn MTLDevice>;
}

/// Returns a reference to the system default Metal device.
///
/// The device is created once and cached for the lifetime of the process.
/// Analogous to `era_cudart::device::get_device()` + `set_device()`.
pub fn system_default_device() -> MetalResult<&'static ProtocolObject<dyn MTLDevice>> {
    static DEVICE: OnceLock<&'static ProtocolObject<dyn MTLDevice>> = OnceLock::new();
    DEVICE
        .get_or_try_init(|| {
            let ptr = unsafe { MTLCreateSystemDefaultDevice() };
            if ptr.is_null() {
                return Err(MetalError::DeviceNotFound);
            }
            // SAFETY: MTLCreateSystemDefaultDevice returns a retained object.
            // We leak it intentionally for 'static lifetime (one-time init).
            Ok(unsafe { &*ptr })
        })
        .copied()
}

/// Query device properties, analogous to `era_cudart::device::device_get_attribute`.
pub struct DeviceProperties {
    /// Maximum threadgroup memory length in bytes.
    pub max_threadgroup_memory_length: usize,
    /// Maximum threads per threadgroup.
    pub max_threads_per_threadgroup: usize,
    /// Maximum buffer length in bytes.
    pub max_buffer_length: usize,
    /// Device name.
    pub name: String,
    /// Whether the device supports unified memory (always true on Apple Silicon).
    pub has_unified_memory: bool,
}

impl DeviceProperties {
    pub fn query(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let max_tg_size = device.maxThreadsPerThreadgroup();
        Self {
            max_threadgroup_memory_length: device.maxThreadgroupMemoryLength(),
            max_threads_per_threadgroup: max_tg_size.width * max_tg_size.height * max_tg_size.depth,
            max_buffer_length: device.maxBufferLength(),
            name: device.name().to_string(),
            has_unified_memory: device.hasUnifiedMemory(),
        }
    }
}
