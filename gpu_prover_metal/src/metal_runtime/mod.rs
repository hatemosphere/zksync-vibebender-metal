pub mod buffer;
pub mod command_queue;
pub mod device;
pub mod dispatch;
pub mod error;
pub mod memory;
pub mod pipeline;
pub mod profiler;
pub mod sync;

// Re-export key types for convenience.
pub use buffer::{MetalBuffer, MetalBufferSlice};
pub use command_queue::{MetalCommandBuffer, MetalCommandQueue};
pub use device::{system_default_device, DeviceProperties};
pub use dispatch::{Dim3, MetalLaunchConfig};
pub use error::{MetalError, MetalResult};
pub use pipeline::{get_pipeline, init_shader_library};
pub use sync::MetalEvent;

/// A "stream" abstraction that mirrors CUDA's persistent stream model.
///
/// In CUDA, `CudaStream` is created once and used to enqueue many operations.
/// In Metal, the equivalent is creating fresh `MTLCommandBuffer` objects from
/// a shared `MTLCommandQueue`. This struct wraps that pattern.
///
/// Each `MetalStream` corresponds to one of the 3 CUDA streams in `ProverContext`:
/// `exec_stream`, `aux_stream`, `h2d_stream`.
pub struct MetalStream<'a> {
    queue: &'a MetalCommandQueue,
}

impl<'a> MetalStream<'a> {
    /// Create a new stream backed by the given command queue.
    /// Analogous to `CudaStream::create()`.
    pub fn new(queue: &'a MetalCommandQueue) -> Self {
        Self { queue }
    }

    /// Create a new command buffer for this stream.
    /// Each batch of work (kernel dispatches, copies) should get its own command buffer.
    pub fn new_command_buffer(&self) -> MetalResult<MetalCommandBuffer> {
        self.queue.new_command_buffer()
    }
}

/// Macro to declare a Metal kernel function with typed arguments.
///
/// This is the Metal equivalent of the `cuda_kernel!` macro from `era_cudart`.
/// It generates:
/// - An argument encoding function
/// - A dispatch helper function
///
/// # Usage
///
/// ```ignore
/// metal_kernel!(
///     leaves_kernel,           // Rust function name
///     "ab_blake2s_leaves_kernel",  // MSL function name in .metallib
///     values: MetalBuffer<BF>,     // buffer args
///     results: MetalBuffer<Digest>,
///     ;                             // separator between buffer and value args
///     log_rows_per_hash: u32,      // value (bytes) args
///     cols_count: u32,
///     count: u32,
/// );
/// ```
#[macro_export]
macro_rules! metal_kernel {
    (
        $fn_name:ident,
        $msl_name:expr,
        $( $buf_name:ident : & $buf_ty:ty ),* $(,)?
        ;
        $( $val_name:ident : $val_ty:ty ),* $(,)?
    ) => {
        #[allow(clippy::too_many_arguments)]
        pub fn $fn_name(
            device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
            cmd_buf: &$crate::metal_runtime::MetalCommandBuffer,
            config: &$crate::metal_runtime::MetalLaunchConfig,
            $( $buf_name: &$buf_ty, )*
            $( $val_name: $val_ty, )*
        ) -> $crate::metal_runtime::MetalResult<()> {
            $crate::metal_runtime::dispatch::dispatch_kernel(
                device,
                cmd_buf,
                $msl_name,
                config,
                |encoder| {
                    let mut _index: u32 = 0;
                    $(
                        $crate::metal_runtime::dispatch::set_buffer(
                            encoder,
                            _index,
                            $buf_name.raw(),
                            0,
                        );
                        _index += 1;
                    )*
                    $(
                        unsafe {
                            $crate::metal_runtime::dispatch::set_bytes(
                                encoder,
                                _index,
                                &$val_name,
                            );
                        }
                        _index += 1;
                    )*
                },
            )
        }
    };
}
