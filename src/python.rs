//! Python bindings for phostt via PyO3.
//!
//! Build with: maturin build --features python
//!
//! Provides a synchronous Python API over the Rust inference engine.

use pyo3::prelude::*;
use std::sync::Arc;

use crate::error::PhosttError;
use crate::inference::Engine;

/// Python wrapper around [`Engine`].
///
/// Thread-safe: the underlying engine lives in an `Arc` so multiple Python
/// threads can transcribe concurrently (up to the ONNX session pool size).
#[pyclass(name = "Engine")]
pub struct PyEngine {
    engine: Arc<Engine>,
}

#[pymethods]
impl PyEngine {
    /// Load the Zipformer-vi ONNX bundle from `model_dir`.
    ///
    /// The directory must contain `encoder.int8.onnx`, `decoder.onnx`,
    /// `joiner.int8.onnx`, `bpe.model`, and `tokens.txt`.
    /// If the models are missing, they can be downloaded first with the Rust
    /// CLI: `phostt download`.
    #[new]
    fn new(model_dir: &str) -> PyResult<Self> {
        let engine = Engine::load(model_dir).map_err(|e| match e {
            PhosttError::ModelLoad(msg) => PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(msg),
            PhosttError::Inference(msg) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg),
            PhosttError::InvalidAudio(msg) => PyErr::new::<pyo3::exceptions::PyValueError, _>(msg),
            PhosttError::Io(err) => PyErr::new::<pyo3::exceptions::PyOSError, _>(format!("{err}")),
        })?;
        Ok(Self {
            engine: Arc::new(engine),
        })
    }

    /// Transcribe an audio file (WAV, MP3, M4A, OGG, FLAC).
    ///
    /// Returns the transcribed text as a string.
    fn transcribe_file(&self, path: &str) -> PyResult<String> {
        let engine = self.engine.clone();
        let path = path.to_string();
        // Run checkout + inference on a dedicated thread so the GIL is released.
        let text = std::thread::spawn(move || {
            let mut guard = engine.pool.checkout_blocking()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Pool checkout failed: {e}")))?;
            let result = engine.transcribe_file(&path, &mut *guard)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Transcription failed: {e}")))?;
            Ok::<String, PyErr>(result.text)
        })
        .join()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Thread panicked: {e:?}")))??;
        Ok(text)
    }

    /// Transcribe audio from raw bytes in memory.
    ///
    /// `data` should contain a complete audio file (WAV, MP3, etc.) — not raw
    /// PCM samples.
    fn transcribe_bytes<'py>(&self, data: &[u8]) -> PyResult<String> {
        let engine = self.engine.clone();
        let data = data.to_vec();
        let text = std::thread::spawn(move || {
            let mut guard = engine.pool.checkout_blocking()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Pool checkout failed: {e}")))?;
            let result = engine.transcribe_bytes(&data, &mut *guard)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Transcription failed: {e}")))?;
            Ok::<String, PyErr>(result.text)
        })
        .join()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Thread panicked: {e:?}")))??;
        Ok(text)
    }
}

/// The phostt Python module.
#[pymodule]
fn phostt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    Ok(())
}
