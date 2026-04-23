//! End-to-end FFI streaming tests.
//!
//! Requires the ONNX model bundle (~850 MB).
//! Run with: `cargo test --test ffi_streaming -- --ignored --features ffi`

mod common;

use phostt::ffi::{
    phostt_engine_free, phostt_stream_flush, phostt_stream_free, phostt_stream_new,
    phostt_stream_process_chunk, phostt_string_free, PhosttEngine,
};
use std::ffi::CString;

/// Happy-path streaming: create a stream, feed PCM16 tone chunks, process and
/// flush, then free. Validates that JSON output is well-formed and that the
/// pool slot is returned after `stream_free`.
#[test]
#[ignore = "Requires model download"]
fn test_ffi_streaming_happy_path() {
    let model_dir = common::model_dir();
    let engine = phostt::inference::Engine::load(&model_dir).unwrap();
    let original_available = engine.pool.available();

    let engine_ptr = Box::into_raw(Box::new(PhosttEngine::new(engine)));

    // 1. Create stream — checks out a triplet.
    let stream_ptr = unsafe { phostt_stream_new(engine_ptr) };
    assert!(!stream_ptr.is_null(), "stream_new should succeed");

    let engine_ref = unsafe { (*engine_ptr).engine() };
    assert_eq!(
        engine_ref.pool.available(),
        original_available.saturating_sub(1),
        "pool slot should be checked out"
    );

    // 2. Feed a 1-second 440 Hz tone at 16 kHz.
    let pcm16 = common::generate_pcm16_tone(1.0, 16000, 440.0);
    let cstring = unsafe {
        phostt_stream_process_chunk(
            engine_ptr,
            stream_ptr,
            pcm16.as_ptr(),
            pcm16.len(),
            16000,
        )
    };
    assert!(!cstring.is_null(), "process_chunk should return a JSON string");

    let json_str = unsafe { CString::from_raw(cstring) }
        .to_string_lossy()
        .to_string();
    let json: serde_json::Value = serde_json::from_str(&json_str).expect("JSON should parse");
    assert!(json.is_array(), "process_chunk output should be a JSON array");

    // 3. Flush — should return a (possibly empty) array.
    let cstring = unsafe { phostt_stream_flush(engine_ptr, stream_ptr) };
    assert!(!cstring.is_null(), "flush should return a JSON string");

    let json_str = unsafe { CString::from_raw(cstring) }
        .to_string_lossy()
        .to_string();
    let json: serde_json::Value = serde_json::from_str(&json_str).expect("JSON should parse");
    assert!(json.is_array(), "flush output should be a JSON array");

    // 4. Free stream — triplet returns to pool.
    unsafe { phostt_stream_free(stream_ptr) };
    assert_eq!(
        engine_ref.pool.available(),
        original_available,
        "pool slot should be returned after stream_free"
    );

    // 5. Tear down engine.
    unsafe { phostt_engine_free(engine_ptr) };
}

/// Verify that `process_chunk` accepts 48 kHz input and resamples internally.
#[test]
#[ignore = "Requires model download"]
fn test_ffi_streaming_resample_48k() {
    let model_dir = common::model_dir();
    let engine = phostt::inference::Engine::load(&model_dir).unwrap();
    let engine_ptr = Box::into_raw(Box::new(PhosttEngine::new(engine)));

    let stream_ptr = unsafe { phostt_stream_new(engine_ptr) };
    assert!(!stream_ptr.is_null());

    let pcm16 = common::generate_pcm16_tone(0.5, 48000, 440.0);
    let cstring = unsafe {
        phostt_stream_process_chunk(
            engine_ptr,
            stream_ptr,
            pcm16.as_ptr(),
            pcm16.len(),
            48000,
        )
    };
    assert!(!cstring.is_null(), "process_chunk with 48 kHz should succeed");
    unsafe { phostt_string_free(cstring) };

    let cstring = unsafe { phostt_stream_flush(engine_ptr, stream_ptr) };
    assert!(!cstring.is_null());
    unsafe { phostt_string_free(cstring) };

    unsafe { phostt_stream_free(stream_ptr) };
    unsafe { phostt_engine_free(engine_ptr) };
}
