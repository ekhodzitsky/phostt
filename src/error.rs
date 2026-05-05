//! Error types for the phostt public API.
//!
//! [`PhosttError`] is the primary error type returned by [`Engine`](crate::inference::Engine)
//! methods. It provides structured error variants so consumers can match on specific
//! failure modes without downcasting.

use std::fmt;

/// Errors returned by phostt public API methods.
///
/// This enum covers the main failure categories:
/// - Model loading failures ([`ModelLoad`](PhosttError::ModelLoad))
/// - Runtime inference errors ([`Inference`](PhosttError::Inference))
/// - Invalid audio input ([`InvalidAudio`](PhosttError::InvalidAudio))
/// - Filesystem / I/O errors ([`Io`](PhosttError::Io))
///
/// # Matching on errors
///
/// ```ignore
/// use phostt::error::PhosttError;
///
/// match err {
///     PhosttError::ModelLoad(msg) => eprintln!("Model problem: {msg}"),
///     PhosttError::Inference(msg) => eprintln!("Inference failed: {msg}"),
///     PhosttError::InvalidAudio(msg) => eprintln!("Bad audio: {msg}"),
///     PhosttError::Io(e) => eprintln!("I/O error: {e}"),
///     _ => eprintln!("Other error"),
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum PhosttError {
    /// Model files not found or failed to load ONNX sessions.
    ModelLoad(String),
    /// ONNX inference failed during encode, decode, or join.
    Inference(String),
    /// Invalid audio input (unsupported format, excessive duration, corrupt data).
    InvalidAudio(String),
    /// Filesystem or I/O error.
    Io(std::io::Error),
}

impl fmt::Display for PhosttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PhosttError::ModelLoad(msg) => write!(f, "model load error: {msg}"),
            PhosttError::Inference(msg) => write!(f, "inference error: {msg}"),
            PhosttError::InvalidAudio(msg) => write!(f, "invalid audio: {msg}"),
            PhosttError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for PhosttError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PhosttError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PhosttError {
    fn from(e: std::io::Error) -> Self {
        PhosttError::Io(e)
    }
}

#[cfg(test)]
/// Backward-compatibility alias for code that references the historical
/// `GigasttError` name from the upstream fork.
#[deprecated(since = "0.4.0", note = "Use `PhosttError` instead")]
pub type GigasttError = PhosttError;

mod tests {
    #[allow(unused_imports)]
    use super::PhosttError;

    #[test]
    fn test_display_model_load() {
        let e = PhosttError::ModelLoad("encoder not found".into());
        assert_eq!(e.to_string(), "model load error: encoder not found");
    }

    #[test]
    fn test_display_inference() {
        let e = PhosttError::Inference("decoder failed".into());
        assert_eq!(e.to_string(), "inference error: decoder failed");
    }

    #[test]
    fn test_display_invalid_audio() {
        let e = PhosttError::InvalidAudio("too long".into());
        assert_eq!(e.to_string(), "invalid audio: too long");
    }

    #[test]
    fn test_display_io() {
        let e = PhosttError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
        assert!(e.to_string().contains("gone"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e: PhosttError = io_err.into();
        assert!(matches!(e, PhosttError::Io(_)));
    }

    #[test]
    fn test_error_source_io() {
        let e = PhosttError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "x"));
        assert!(std::error::Error::source(&e).is_some());
    }

    #[test]
    fn test_error_source_none_for_others() {
        let e = PhosttError::ModelLoad("x".into());
        assert!(std::error::Error::source(&e).is_none());
    }

    #[test]
    fn test_into_anyhow() {
        // Verify PhosttError works with ? in anyhow::Result contexts
        fn returns_anyhow() -> anyhow::Result<()> {
            Err(PhosttError::Inference("test".into()))?;
            Ok(())
        }
        assert!(returns_anyhow().is_err());
    }
}
