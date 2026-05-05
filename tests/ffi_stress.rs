//! Long-running FFI stress test for streaming.
//!
//! Verifies that repeated stream create → process → flush → free cycles do not
//! leak pool slots, crash, or degrade over time.
//!
//! Run with:
//!   cargo test --test ffi_stress -- --ignored --features ffi
//!
//! Environment variables:
//!   PHOSTT_FFI_SOAK_DURATION_SECS  — total soak time per worker (default: 60)
//!   PHOSTT_FFI_SOAK_PARALLEL       — concurrent workers (default: 2)

#![cfg(feature = "ffi")]

mod common;

use phostt::ffi::{
    PhosttEngine, phostt_engine_free, phostt_stream_flush, phostt_stream_free,
    phostt_stream_new, phostt_stream_process_chunk, phostt_string_free,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Generate a short 200 ms PCM16 tone chunk (16 kHz, 440 Hz).
fn tone_chunk() -> Vec<u8> {
    common::generate_pcm16_tone(0.2, 16000, 440.0)
}

/// Single-threaded soak loop. Each worker loads its own engine.
fn soak_worker(model_dir: String, duration: Duration, error_count: Arc<AtomicUsize>) {
    let engine = phostt::inference::Engine::load(&model_dir).unwrap();
    let original_available = engine.pool.available();
    let engine_ptr = Box::into_raw(Box::new(PhosttEngine::new(engine)));

    let chunk = tone_chunk();
    let deadline = Instant::now() + duration;
    let mut iterations = 0usize;

    while Instant::now() < deadline {
        iterations += 1;

        // 1. Create stream.
        let stream_ptr = unsafe { phostt_stream_new(engine_ptr) };
        if stream_ptr.is_null() {
            error_count.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        // 2. Feed 10 small chunks (~2 s of audio total).
        for _ in 0..10 {
            let cstring = unsafe {
                phostt_stream_process_chunk(
                    engine_ptr,
                    stream_ptr,
                    chunk.as_ptr(),
                    chunk.len(),
                    16000,
                )
            };
            if cstring.is_null() {
                error_count.fetch_add(1, Ordering::Relaxed);
                break;
            }
            unsafe { phostt_string_free(cstring) };
        }

        // 3. Flush.
        let cstring = unsafe { phostt_stream_flush(engine_ptr, stream_ptr) };
        if cstring.is_null() {
            error_count.fetch_add(1, Ordering::Relaxed);
        } else {
            unsafe { phostt_string_free(cstring) };
        }

        // 4. Free stream — triplet must return to pool.
        unsafe { phostt_stream_free(stream_ptr) };
    }

    // Verify pool recovered all slots.
    let engine_ref = unsafe { (*engine_ptr).engine() };
    let final_available = engine_ref.pool.available();
    assert_eq!(
        final_available, original_available,
        "Pool should recover all slots after stress (expected {original_available}, got {final_available})"
    );

    unsafe { phostt_engine_free(engine_ptr) };
    eprintln!("Soak worker finished {iterations} iterations");
}

#[test]
#[ignore = "Requires model download"]
fn test_ffi_streaming_stress() {
    let model_dir = common::model_dir();

    let duration_secs: u64 = std::env::var("PHOSTT_FFI_SOAK_DURATION_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let parallel: usize = std::env::var("PHOSTT_FFI_SOAK_PARALLEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let duration = Duration::from_secs(duration_secs);
    let error_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..parallel)
        .map(|_| {
            let dir = model_dir.clone();
            let err = Arc::clone(&error_count);
            thread::spawn(move || soak_worker(dir, duration, err))
        })
        .collect();

    for h in handles {
        h.join().expect("soak worker should not panic");
    }

    let elapsed = start.elapsed();
    let errors = error_count.load(Ordering::Relaxed);

    assert_eq!(
        errors, 0,
        "Stress test produced {errors} errors across all workers"
    );

    eprintln!(
        "FFI stress passed: {} workers × {} s, {} errors, elapsed {:?}",
        parallel, duration_secs, errors, elapsed
    );
}
