//! # phostt
//!
//! Local speech-to-text powered by Zipformer-vi RNN-T — on-device Vietnamese speech
//! recognition via ONNX Runtime. No cloud APIs, no API keys, full privacy.
//!
//! ## Quick start
//!
//! ```ignore
//! use phostt::inference::Engine;
//!
//! let engine = Engine::load("~/.phostt/models")?;
//!
//! // File transcription
//! let text = engine.transcribe_file("audio.wav")?;
//!
//! // Streaming recognition
//! let mut state = engine.create_state(/* diarization_enabled: */ false);
//! let segments = engine.process_chunk(&audio_16khz, &mut state)?;
//! ```
//!
//! ## Modules
//!
//! - [`inference`] — ONNX inference engine, streaming state, audio utilities
//! - [`error`] — Typed error types ([`GigasttError`](error::GigasttError))
//! - [`protocol`] — WebSocket JSON message types
//! - [`server`] — WebSocket server entry point
//! - [`model`] — Model download and management

pub mod error;
pub mod inference;
pub mod inspect;
pub mod model;
pub mod protocol;
pub mod server;
