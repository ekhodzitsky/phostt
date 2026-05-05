//! Runtime smoke tests for ONNX Execution Providers.
//!
//! These tests verify that the CUDA and CoreML EPs actually load and run
//! inference end-to-end. They are `#[ignore]` by default because they require
//! the corresponding hardware and the model bundle.
//!
//! Run manually:
//!   cargo test --test ep_smoke --features cuda -- --ignored
//!   cargo test --test ep_smoke --features coreml -- --ignored

mod common;

use phostt::inference::{Engine, audio};

/// Smoke test: load engine with CUDA EP and transcribe one WAV.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "Requires CUDA GPU and model download"]
fn test_cuda_ep_smoke() {
    let model_dir = common::model_dir();
    let engine = Engine::load(&model_dir).expect("Engine::load with CUDA feature should succeed");

    let wav_path = common::test_wav_path(0);
    let samples = audio::decode_audio_file(wav_path.to_str().unwrap())
        .expect("WAV decode should succeed");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut triplet = rt.block_on(async { engine.pool.checkout().await.unwrap() });
    let result = engine.transcribe_samples(&samples, &mut triplet)
        .expect("Inference with CUDA EP should succeed");

    assert!(
        !result.text.is_empty(),
        "CUDA EP should produce non-empty transcript, got empty"
    );
    eprintln!("CUDA EP smoke passed — transcript: {}", result.text);
}

/// Smoke test: load engine with CoreML EP and transcribe one WAV.
#[cfg(feature = "coreml")]
#[test]
#[ignore = "Requires CoreML (macOS) and model download"]
fn test_coreml_ep_smoke() {
    let model_dir = common::model_dir();
    let engine = Engine::load(&model_dir).expect("Engine::load with CoreML feature should succeed");

    let wav_path = common::test_wav_path(0);
    let samples = audio::decode_audio_file(wav_path.to_str().unwrap())
        .expect("WAV decode should succeed");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut triplet = rt.block_on(async { engine.pool.checkout().await.unwrap() });
    let result = engine.transcribe_samples(&samples, &mut triplet)
        .expect("Inference with CoreML EP should succeed");

    assert!(
        !result.text.is_empty(),
        "CoreML EP should produce non-empty transcript, got empty"
    );
    eprintln!("CoreML EP smoke passed — transcript: {}", result.text);
}
