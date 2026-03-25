use super::error::{MetalError, MetalResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Global shader library, loaded once from the embedded metallib.
static SHADER_LIBRARY: OnceLock<Retained<ProtocolObject<dyn MTLLibrary>>> = OnceLock::new();

/// Cache of compiled pipeline states, keyed by function name.
static PIPELINE_CACHE: OnceLock<Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>> =
    OnceLock::new();

/// Initialize the shader library from the embedded metallib data.
///
/// Must be called once before any kernel dispatch. Typically called during
/// `ProverContext` initialization.
pub fn init_shader_library(device: &ProtocolObject<dyn MTLDevice>) -> MetalResult<()> {
    SHADER_LIBRARY.get_or_try_init(|| {
        #[cfg(not(no_metal))]
        {
            let metallib_bytes: &[u8] = include_bytes!(env!("METAL_LIB_PATH"));
            if metallib_bytes.is_empty() {
                return Err(MetalError::ResourceCreationFailed(
                    "Embedded metallib is empty — no shaders compiled".into(),
                ));
            }
            // Create a dispatch_data_t from the bytes
            let data = dispatch2::DispatchData::from_bytes(metallib_bytes);
            let library = device
                .newLibraryWithData_error(&data)
                .map_err(|e| {
                    MetalError::ResourceCreationFailed(format!(
                        "Failed to load Metal shader library: {e}"
                    ))
                })?;
            Ok(library)
        }
        #[cfg(no_metal)]
        {
            let _ = device;
            Err(MetalError::Other("Metal not available".into()))
        }
    })?;

    PIPELINE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    Ok(())
}

/// Get or create a compute pipeline state for the named kernel function.
///
/// Pipeline states are cached — repeated calls with the same name return
/// the same `MTLComputePipelineState`.
pub fn get_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    function_name: &str,
) -> MetalResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let cache = PIPELINE_CACHE
        .get()
        .ok_or_else(|| MetalError::Other("Shader library not initialized".into()))?;

    let mut cache = cache.lock().unwrap();

    if let Some(pipeline) = cache.get(function_name) {
        return Ok(pipeline.clone());
    }

    let library = SHADER_LIBRARY
        .get()
        .ok_or_else(|| MetalError::Other("Shader library not initialized".into()))?;

    let name = NSString::from_str(function_name);
    let function = library.newFunctionWithName(&name).ok_or_else(|| {
        MetalError::FunctionNotFound(function_name.to_string())
    })?;

    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| {
            MetalError::PipelineCreationFailed(format!(
                "Failed to create pipeline for '{function_name}': {e}"
            ))
        })?;

    cache.insert(function_name.to_string(), pipeline.clone());
    Ok(pipeline)
}

/// Query the max total threads per threadgroup for a given pipeline.
pub fn max_threads_per_threadgroup(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> usize {
    pipeline.maxTotalThreadsPerThreadgroup() as usize
}

/// Query the threadExecutionWidth (SIMD group size, typically 32 on Apple GPUs).
pub fn thread_execution_width(
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
) -> usize {
    pipeline.threadExecutionWidth() as usize
}
