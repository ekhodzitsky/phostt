//! C-ABI FFI layer for Android / JNI integration.
//!
//! Exposes a minimal surface so that Kotlin (or any other JNI consumer) can:
//! 1. Load the inference engine (`phostt_engine_new`).
//! 2. Transcribe a WAV file (`phostt_transcribe_file`).
//! 3. Stream audio in real-time (`phostt_stream_new`, `phostt_stream_process_chunk`,
//!    `phostt_stream_flush`, `phostt_stream_free`).
//! 4. Free the returned C string (`phostt_string_free`).
//! 5. Tear down the engine (`phostt_engine_free`).
//!
//! All functions are `unsafe` by nature (raw pointers cross the FFI boundary) but
//! the implementation checks nulls and logs errors before returning sentinel values.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use crate::inference::audio;
use crate::inference::{Engine, OwnedReservation, SessionTriplet, StreamingState};

/// Opaque handle to the inference engine.
///
/// The Kotlin side sees this as a `Long` (pointer-sized integer).
pub struct PhosttEngine {
    engine: Engine,
}

impl PhosttEngine {
    /// Wrap a raw `Engine` into the FFI handle.
    pub fn new(engine: Engine) -> Self {
        Self { engine }
    }

    /// Borrow the inner inference engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }
}

/// Opaque handle to a streaming transcription session.
///
/// Holds a checked-out `SessionTriplet` and a `StreamingState`. The triplet is
/// returned to the pool when `phostt_stream_free` is called.
pub struct PhosttStream {
    state: StreamingState,
    triplet: SessionTriplet,
    reservation: OwnedReservation<SessionTriplet>,
}

/// Load the ONNX models from `model_dir` and create an inference engine.
///
/// # Safety
/// `model_dir` must be a valid, null-terminated UTF-8 string.
/// Returns a pointer to a `PhosttEngine` on success, or `NULL` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phostt_engine_new(model_dir: *const c_char) -> *mut PhosttEngine {
    if model_dir.is_null() {
        tracing::error!("phostt_engine_new: model_dir is null");
        eprintln!("phostt_engine_new: model_dir is null");
        return ptr::null_mut();
    }

    let dir_str = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("phostt_engine_new: model_dir is not valid UTF-8: {e}");
            eprintln!("phostt_engine_new: model_dir is not valid UTF-8: {e}");
            return ptr::null_mut();
        }
    };

    match Engine::load(dir_str) {
        Ok(engine) => {
            let handle = Box::new(PhosttEngine { engine });
            Box::into_raw(handle)
        }
        Err(e) => {
            tracing::error!("phostt_engine_new: failed to load engine: {e}");
            eprintln!("phostt_engine_new: failed to load engine: {e}");
            ptr::null_mut()
        }
    }
}

/// Transcribe an audio file and return the recognized text as a newly allocated C string.
///
/// # Safety
/// - `engine` must be a non-null pointer returned by `phostt_engine_new` and not yet freed.
/// - `wav_path` must be a valid, null-terminated UTF-8 string.
///
/// Returns a pointer to a NUL-terminated UTF-8 string on success, or `NULL` on failure.
/// The caller **must** free the returned string with `phostt_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phostt_transcribe_file(
    engine: *mut PhosttEngine,
    wav_path: *const c_char,
) -> *mut c_char {
    if engine.is_null() {
        tracing::error!("phostt_transcribe_file: engine is null");
        eprintln!("phostt_transcribe_file: engine is null");
        return ptr::null_mut();
    }
    if wav_path.is_null() {
        tracing::error!("phostt_transcribe_file: wav_path is null");
        eprintln!("phostt_transcribe_file: wav_path is null");
        return ptr::null_mut();
    }

    let path_str = match unsafe { CStr::from_ptr(wav_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("phostt_transcribe_file: wav_path is not valid UTF-8: {e}");
            eprintln!("phostt_transcribe_file: wav_path is not valid UTF-8: {e}");
            return ptr::null_mut();
        }
    };

    let engine_ref = unsafe { &(*engine).engine };

    // Checkout a session triplet synchronously. The PoolGuard auto-returns
    // the triplet to the pool when it goes out of scope (including on panic
    // unwind), so there is no leak on the happy or unhappy path.
    let mut guard = match engine_ref.pool.checkout_blocking() {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("phostt_transcribe_file: failed to checkout session from pool: {e}");
            eprintln!("phostt_transcribe_file: failed to checkout session from pool: {e}");
            return ptr::null_mut();
        }
    };

    let result = match engine_ref.transcribe_file(path_str, &mut guard) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("phostt_transcribe_file: transcription failed: {e}");
            eprintln!("phostt_transcribe_file: transcription failed: {e}");
            return ptr::null_mut();
        }
    };

    // `guard` drops here and returns the triplet to the pool automatically.

    match CString::new(result.text) {
        Ok(cstr) => cstr.into_raw(),
        Err(e) => {
            tracing::error!("phostt_transcribe_file: result contains interior NUL: {e}");
            eprintln!("phostt_transcribe_file: result contains interior NUL: {e}");
            ptr::null_mut()
        }
    }
}

/// Free a C string previously returned by `phostt_transcribe_file` or the
/// streaming functions.
///
/// # Safety
/// `s` must be a pointer returned by one of the transcription functions and not
/// yet freed, or `NULL` (in which case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phostt_string_free(s: *mut c_char) {
    if !s.is_null() {
        // Reclaim the CString so its Drop impl frees the allocation.
        let _ = unsafe { CString::from_raw(s) };
    }
}

/// Free an inference engine previously created by `phostt_engine_new`.
///
/// # Safety
/// `engine` must be a pointer returned by `phostt_engine_new` and not yet freed,
/// or `NULL` (in which case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phostt_engine_free(engine: *mut PhosttEngine) {
    if !engine.is_null() {
        // Reclaim the Box so its Drop impl frees the engine and its sessions.
        let _ = unsafe { Box::from_raw(engine) };
    }
}

// ---------------------------------------------------------------------------
// Streaming API
// ---------------------------------------------------------------------------

/// Create a new streaming session.
///
/// Checks out a `SessionTriplet` from the engine pool and creates a fresh
/// `StreamingState`. The triplet is held for the lifetime of the stream and
/// returned to the pool by `phostt_stream_free`.
///
/// # Safety
/// `engine` must be a valid pointer returned by `phostt_engine_new`.
/// Returns a pointer to a `PhosttStream` on success, or `NULL` on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phostt_stream_new(engine: *mut PhosttEngine) -> *mut PhosttStream {
    if engine.is_null() {
        tracing::error!("phostt_stream_new: engine is null");
        eprintln!("phostt_stream_new: engine is null");
        return ptr::null_mut();
    }

    let engine_ref = unsafe { &(*engine).engine };

    let guard = match engine_ref.pool.checkout_blocking() {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("phostt_stream_new: pool checkout failed: {e}");
            eprintln!("phostt_stream_new: pool checkout failed: {e}");
            return ptr::null_mut();
        }
    };

    let (triplet, reservation) = guard.into_owned();

    let state = match engine_ref.create_state(false) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("phostt_stream_new: failed to create streaming state: {e}");
            eprintln!("phostt_stream_new: failed to create streaming state: {e}");
            // Return triplet to pool before giving up.
            reservation.checkin(triplet);
            return ptr::null_mut();
        }
    };

    let stream = PhosttStream {
        state,
        triplet,
        reservation,
    };
    Box::into_raw(Box::new(stream))
}

/// Process a chunk of PCM16 audio and return any partial/final segments.
///
/// # Safety
/// - `engine` and `stream` must be valid pointers.
/// - `pcm16_bytes` must point to at least `len` valid bytes (little-endian mono PCM16).
///
/// Returns a newly allocated JSON array string on success, or `NULL` on failure.
/// The caller **must** free the returned string with `phostt_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phostt_stream_process_chunk(
    engine: *mut PhosttEngine,
    stream: *mut PhosttStream,
    pcm16_bytes: *const u8,
    len: usize,
    sample_rate: u32,
) -> *mut c_char {
    if engine.is_null() {
        tracing::error!("phostt_stream_process_chunk: engine is null");
        return ptr::null_mut();
    }
    if stream.is_null() {
        tracing::error!("phostt_stream_process_chunk: stream is null");
        return ptr::null_mut();
    }
    if pcm16_bytes.is_null() {
        tracing::error!("phostt_stream_process_chunk: pcm16_bytes is null");
        return ptr::null_mut();
    }

    let engine_ref = unsafe { &(*engine).engine };
    let stream_ref = unsafe { &mut (*stream) };

    // Convert PCM16 LE bytes → f32 samples.
    let bytes = unsafe { std::slice::from_raw_parts(pcm16_bytes, len) };
    let pcm16: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let mut samples_f32: Vec<f32> = pcm16.iter().map(|&s| s as f32 / 32768.0).collect();

    // Resample to 16 kHz if needed.
    if sample_rate != crate::inference::TARGET_SAMPLE_RATE {
        samples_f32 = match audio::resample(
            &samples_f32,
            sample_rate,
            crate::inference::TARGET_SAMPLE_RATE,
        ) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("phostt_stream_process_chunk: resample failed: {e}");
                return ptr::null_mut();
            }
        };
    }

    let segments = match engine_ref.process_chunk(
        &samples_f32,
        &mut stream_ref.state,
        &mut stream_ref.triplet,
    ) {
        Ok(segs) => segs,
        Err(e) => {
            tracing::error!("phostt_stream_process_chunk: inference failed: {e}");
            return ptr::null_mut();
        }
    };

    let json = serde_json::to_string(&segments).unwrap_or_else(|_| "[]".into());
    match CString::new(json) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Flush the streaming state and return the final segment(s).
///
/// # Safety
/// `engine` and `stream` must be valid pointers.
///
/// Returns a newly allocated JSON array string (possibly `[]`) on success,
/// or `NULL` on failure. The caller **must** free the returned string with
/// `phostt_string_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phostt_stream_flush(
    engine: *mut PhosttEngine,
    stream: *mut PhosttStream,
) -> *mut c_char {
    if engine.is_null() {
        tracing::error!("phostt_stream_flush: engine is null");
        return ptr::null_mut();
    }
    if stream.is_null() {
        tracing::error!("phostt_stream_flush: stream is null");
        return ptr::null_mut();
    }

    let engine_ref = unsafe { &(*engine).engine };
    let stream_ref = unsafe { &mut (*stream) };

    let segments: Vec<crate::inference::TranscriptSegment> = engine_ref
        .flush_state(&mut stream_ref.state, &mut stream_ref.triplet)
        .into_iter()
        .collect();

    let json = serde_json::to_string(&segments).unwrap_or_else(|_| "[]".into());
    match CString::new(json) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a streaming session and return its triplet to the pool.
///
/// # Safety
/// `stream` must be a pointer returned by `phostt_stream_new` and not yet freed,
/// or `NULL` (in which case this is a no-op).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn phostt_stream_free(stream: *mut PhosttStream) {
    if !stream.is_null() {
        let stream = unsafe { Box::from_raw(stream) };
        stream.reservation.checkin(stream.triplet);
        // `state` is dropped automatically when `stream` goes out of scope.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_new_null_engine() {
        let stream = unsafe { phostt_stream_new(ptr::null_mut()) };
        assert!(stream.is_null());
    }

    #[test]
    fn test_stream_new_pool_closed() {
        let engine = Engine::test_stub();
        engine.pool.close();
        let engine_ptr = Box::into_raw(Box::new(PhosttEngine { engine }));
        let stream = unsafe { phostt_stream_new(engine_ptr) };
        assert!(stream.is_null());
        unsafe { phostt_engine_free(engine_ptr) };
    }

    #[test]
    fn test_stream_process_chunk_null_args() {
        let engine = Engine::test_stub();
        let engine_ptr = Box::into_raw(Box::new(PhosttEngine { engine }));

        // null engine
        let r = unsafe {
            phostt_stream_process_chunk(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                0,
                crate::inference::TARGET_SAMPLE_RATE,
            )
        };
        assert!(r.is_null());

        // null stream
        let r = unsafe {
            phostt_stream_process_chunk(
                engine_ptr,
                ptr::null_mut(),
                ptr::null(),
                0,
                crate::inference::TARGET_SAMPLE_RATE,
            )
        };
        assert!(r.is_null());

        // null pcm16
        let dummy_stream = 0x1 as *mut PhosttStream;
        let r = unsafe {
            phostt_stream_process_chunk(
                engine_ptr,
                dummy_stream,
                ptr::null(),
                0,
                crate::inference::TARGET_SAMPLE_RATE,
            )
        };
        assert!(r.is_null());

        unsafe { phostt_engine_free(engine_ptr) };
    }

    #[test]
    fn test_stream_flush_null_args() {
        let engine = Engine::test_stub();
        let engine_ptr = Box::into_raw(Box::new(PhosttEngine { engine }));

        let r = unsafe { phostt_stream_flush(ptr::null_mut(), ptr::null_mut()) };
        assert!(r.is_null());

        let r = unsafe { phostt_stream_flush(engine_ptr, ptr::null_mut()) };
        assert!(r.is_null());

        unsafe { phostt_engine_free(engine_ptr) };
    }

    #[test]
    fn test_stream_free_null() {
        // Should be a no-op, not a crash.
        unsafe { phostt_stream_free(ptr::null_mut()) };
    }
}
