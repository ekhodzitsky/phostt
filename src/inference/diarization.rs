//! Speaker diarization adapter around the `polyvoice` crate.
//!
//! Loads a shared WeSpeaker ResNet34 ONNX embedder (`ResNet34Adapter`) and
//! exposes a cheap `Clone` wrapper so each WebSocket session can own a
//! `StreamingPipeline` without duplicating the ONNX session pool.

use std::path::Path;
use std::sync::Arc;

use polyvoice::{Embedder, EmbedderError, ResNet34Adapter};

/// Dimension of speaker embedding vectors (WeSpeaker ResNet34).
pub const EMBEDDING_DIM: usize = 256;

/// Number of audio samples per analysis window (1.5 s at 16 kHz).
/// Matches `polyvoice::WindowConfig::default()` (`window_secs = 1.5`).
pub const SEGMENT_SAMPLES: usize = 24000;

/// Shared WeSpeaker ResNet34 embedder. `Clone` is an `Arc` bump so every
/// streaming session can hold its own `StreamingPipeline` against one pool.
#[derive(Clone)]
pub struct SharedEmbedder(Arc<ResNet34Adapter>);

impl Embedder for SharedEmbedder {
    fn dim(&self) -> usize {
        self.0.dim()
    }

    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        self.0.embed(audio)
    }

    fn embed_batch(&self, audios: &[&[f32]]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.0.embed_batch(audios)
    }
}

/// Load the ONNX speaker embedding extractor from `model_dir/wespeaker_resnet34.onnx`.
///
/// Creates a lock-free session pool with `pool_size` sessions so that multiple
/// concurrent streaming connections can extract embeddings without mutex contention.
///
/// # Errors
///
/// Returns an error if the model file is missing or an ONNX session cannot be
/// created.
pub fn load_extractor(model_dir: &Path, pool_size: usize) -> anyhow::Result<SharedEmbedder> {
    let path = model_dir.join("wespeaker_resnet34.onnx");
    if !path.exists() {
        anyhow::bail!(
            "wespeaker_resnet34.onnx not found in {}",
            model_dir.display()
        );
    }
    let adapter = ResNet34Adapter::new(&path, pool_size, polyvoice::onnx::ExecutionProvider::Cpu)
        .map_err(|e| anyhow::anyhow!("{e:#}"))?;
    Ok(SharedEmbedder(Arc::new(adapter)))
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
