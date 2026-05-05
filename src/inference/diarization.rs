//! Speaker diarization adapter around the `polyvoice` crate.
//!
//! Re-exports polyvoice types and provides a convenience helper to load the
//! WeSpeaker ResNet34 ONNX embedding extractor used by phostt.

use std::path::Path;

pub use polyvoice::{
    DiarizationConfig, EmbeddingError, EmbeddingExtractor, OnlineDiarizer, OnnxEmbeddingExtractor,
    SampleRate, SpeakerId,
};

/// Dimension of speaker embedding vectors (WeSpeaker ResNet34).
pub const EMBEDDING_DIM: usize = 256;

/// Number of audio samples per analysis window (1.5 s at 16 kHz).
pub const SEGMENT_SAMPLES: usize = 24000;

/// Load the ONNX speaker embedding extractor from `model_dir/wespeaker_resnet34.onnx`.
///
/// Creates a lock-free session pool with `pool_size` sessions so that multiple
/// concurrent streaming connections can extract embeddings without mutex contention.
///
/// # Errors
///
/// Returns an error if the model file is missing or an ONNX session cannot be
/// created.
pub fn load_extractor(
    model_dir: &Path,
    pool_size: usize,
) -> anyhow::Result<OnnxEmbeddingExtractor> {
    let path = model_dir.join("wespeaker_resnet34.onnx");
    if !path.exists() {
        anyhow::bail!(
            "wespeaker_resnet34.onnx not found in {}",
            model_dir.display()
        );
    }
    OnnxEmbeddingExtractor::new(&path, EMBEDDING_DIM, SEGMENT_SAMPLES, pool_size)
        .map_err(|e| anyhow::anyhow!("{e:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dim_constant() {
        assert_eq!(EMBEDDING_DIM, 256);
    }

    #[test]
    fn test_segment_samples_constant() {
        // 1.5 s * 16000 Hz = 24000
        assert_eq!(SEGMENT_SAMPLES, 24000);
    }

    #[test]
    fn test_load_extractor_missing_file() {
        let result = load_extractor(Path::new("/nonexistent/path"), 1);
        assert!(result.is_err());
        let err = result.err().unwrap();
        let msg = format!("{err}");
        assert!(msg.contains("wespeaker_resnet34.onnx"));
    }
}
