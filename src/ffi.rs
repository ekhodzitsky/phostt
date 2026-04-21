//! C-ABI FFI layer for Android / JNI integration.
//!
//! Exposes a minimal surface so that Kotlin (or any other JNI consumer) can:
//! 1. Load the inference engine (`phostt_engine_new`).
//! 2. Transcribe a WAV file (`phostt_transcribe_file`).
//! 3. Free the returned C string (`phostt_string_free`).
//! 4. Tear down the engine (`phostt_engine_free`).
//!
//! All functions are `unsafe` by nature (raw pointers cross the FFI boundary) but
//! the implementation checks nulls and logs errors before returning sentinel values.

use std::ffi::{c_char, CStr, CString};
use std::ptr;

use crate::inference::Engine;

/// Opaque handle to the inference engine.
///
/// The Kotlin side sees this as a `Long` (pointer-sized integer).
pub struct PhosttEngine {
    engine: Engine,
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

/// Free a C string previously returned by `phostt_transcribe_file`.
///
/// # Safety
/// `s` must be a pointer returned by `phostt_transcribe_file` and not yet freed,
/// or `NULL` (in which case this is a no-op).
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
