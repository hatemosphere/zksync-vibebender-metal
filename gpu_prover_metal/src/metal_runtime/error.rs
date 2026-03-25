use std::fmt;

/// Metal GPU error type, analogous to `CudaError` from `era_cudart`.
#[derive(Debug, Clone)]
pub enum MetalError {
    /// Device not found or not available.
    DeviceNotFound,
    /// Failed to create a Metal resource (buffer, pipeline, etc.).
    ResourceCreationFailed(String),
    /// Shader function not found in the Metal library.
    FunctionNotFound(String),
    /// Pipeline state creation failed.
    PipelineCreationFailed(String),
    /// Command buffer execution error.
    CommandBufferError(String),
    /// Buffer size mismatch or out-of-bounds access.
    InvalidBufferSize {
        expected: usize,
        actual: usize,
    },
    /// General error with a message.
    Other(String),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::DeviceNotFound => write!(f, "Metal device not found"),
            MetalError::ResourceCreationFailed(msg) => {
                write!(f, "Metal resource creation failed: {msg}")
            }
            MetalError::FunctionNotFound(name) => {
                write!(f, "Metal function '{name}' not found in library")
            }
            MetalError::PipelineCreationFailed(msg) => {
                write!(f, "Metal pipeline creation failed: {msg}")
            }
            MetalError::CommandBufferError(msg) => {
                write!(f, "Metal command buffer error: {msg}")
            }
            MetalError::InvalidBufferSize { expected, actual } => {
                write!(
                    f,
                    "Metal buffer size mismatch: expected {expected}, got {actual}"
                )
            }
            MetalError::Other(msg) => write!(f, "Metal error: {msg}"),
        }
    }
}

impl std::error::Error for MetalError {}

/// Result type for Metal operations, analogous to `CudaResult<T>`.
pub type MetalResult<T> = Result<T, MetalError>;
