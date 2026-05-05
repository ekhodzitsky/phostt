//! ONNX Runtime inference engine for Zipformer-vi RNN-T.
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T streaming decode loop.

pub mod audio;
mod decode;
mod features;
mod tokenizer;

#[cfg(feature = "diarization")]
pub mod diarization;

// When both `coreml` and `cuda` are enabled, CUDA takes precedence.
// This allows `cargo check --all-features` to pass for CI hygiene.

use anyhow::Context;
#[cfg(any(feature = "coreml", feature = "cuda"))]
use ort::ep;
use ort::session::Session;
use ort::value::TensorRef;
use serde::Serialize;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::Arc;

use crate::error::GigasttError;

use features::MelSpectrogram;
use kaldi_native_fbank::fbank::FbankComputer;
use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};
use tokenizer::Tokenizer;

/// Target audio sample rate in Hz. All audio fed to the model must be
/// resampled to this rate.
pub const TARGET_SAMPLE_RATE: u32 = 16000;
/// Number of mel frequency bins used for spectrogram features.
/// Zipformer-vi expects 80-bin FBANK (kaldi-native-fbank default for ASR).
pub const N_MELS: usize = 80;
/// Frame window length in samples (25 ms × 16 kHz). The FBANK extractor pads
/// internally to the next power of two (512) before the FFT; callers that
/// only need to know "how many samples make one usable frame" use this.
pub const N_FFT: usize = 400;
/// Hop length between consecutive FBANK frames in samples (10 ms × 16 kHz).
pub const HOP_LENGTH: usize = 160;
/// Encoder output channel dimension. Zipformer-vi-30M emits 512-dim frames
/// after a 4× subsampling stage.
pub const ENCODER_OUT_DIM: usize = 512;
/// Decoder output dimension (matches encoder so the joiner is symmetric).
pub const DECODER_OUT_DIM: usize = 512;
/// Number of previously emitted tokens the stateless Zipformer decoder
/// reads on every step. Sherpa-onnx Zipformer transducers ship with a
/// context size of 2 (matches icefall defaults).
pub const CONTEXT_SIZE: usize = 2;

fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

pub(crate) fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Seconds per encoder frame (HOP_LENGTH * 4 / TARGET_SAMPLE_RATE = 0.04s).
const SECONDS_PER_FRAME: f64 = (HOP_LENGTH as f64) * 4.0 / (TARGET_SAMPLE_RATE as f64);

/// Default number of session triplets in the pool.
const DEFAULT_POOL_SIZE: usize = 4;

/// Tunable streaming overlap-buffer parameters.
///
/// The offline Zipformer-vi encoder is fed fixed-size windows; these knobs
/// control the latency / accuracy trade-off. Smaller windows reduce latency
/// but increase boundary artefacts; larger overlap improves continuity at the
/// cost of more compute.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Window size in mel frames (default 400 ≈ 4 s).
    pub window_frames: usize,
    /// Overlap between consecutive windows in mel frames (default 100 ≈ 1 s).
    pub overlap_frames: usize,
    /// Fuzzy-match threshold for the overlap merge (0.0 = exact, 1.0 = anything).
    /// Words with normalized Levenshtein similarity ≥ this value are treated
    /// as equal during boundary deduplication.
    pub fuzzy_match_threshold: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            window_frames: 400,
            overlap_frames: 100,
            fuzzy_match_threshold: 1.0, // exact match by default
        }
    }
}

impl StreamingConfig {
    /// Shift between window starts in mel frames.
    pub fn shift_frames(&self) -> usize {
        self.window_frames.saturating_sub(self.overlap_frames)
    }

    /// Shift between window starts in encoder frames (subsampling-by-4).
    pub fn shift_encoder_frames(&self) -> usize {
        self.shift_frames() / 4
    }

    /// Validate invariants. Returns an error message if config is invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.window_frames == 0 {
            return Err("streaming window must be > 0 frames".into());
        }
        if self.overlap_frames >= self.window_frames {
            return Err("streaming overlap must be smaller than window".into());
        }
        if !self.window_frames.is_multiple_of(4) {
            return Err("streaming window must be a multiple of 4 (encoder subsampling)".into());
        }
        if !self.overlap_frames.is_multiple_of(4) {
            return Err("streaming overlap must be a multiple of 4 (encoder subsampling)".into());
        }
        if !(0.0..=1.0).contains(&self.fuzzy_match_threshold) {
            return Err("fuzzy_match_threshold must be in [0.0, 1.0]".into());
        }
        Ok(())
    }
}

/// A set of ONNX sessions for one inference pipeline (encoder + decoder + joiner).
///
/// Moved out of the pool on checkout and returned on checkin.
/// Each triplet is independent and can run inference concurrently with others.
pub struct SessionTriplet {
    pub(crate) encoder: Session,
    pub(crate) decoder: Session,
    pub(crate) joiner: Session,
}

/// Errors returned by [`Pool::checkout`].
#[derive(Debug)]
pub enum PoolError {
    /// The pool was closed (graceful shutdown). All current and future
    /// waiters resolve to this variant; the caller should respond with a
    /// 503 / `pool_closed` to the client.
    Closed,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::Closed => write!(f, "session pool is closed"),
        }
    }
}

impl std::error::Error for PoolError {}

/// Pool of pre-loaded items of type `T` backed by an MPMC `async-channel`.
///
/// `SessionPool = Pool<SessionTriplet>` is the only public instantiation
/// outside this module. Generic `T` exists so the pool semantics can be
/// unit-tested without ONNX models.
///
/// Checkout = `recv` from the channel, checkin = `send` back via the
/// [`PoolGuard`] returned by [`checkout`](Self::checkout). The pool size acts
/// as the concurrency limit — no separate semaphore needed. FIFO ordering is
/// intrinsic to the underlying channel, and `close()` flips all current and
/// future waiters into [`PoolError::Closed`] so graceful shutdown can drain
/// without panicking.
pub struct Pool<T> {
    sender: async_channel::Sender<T>,
    receiver: async_channel::Receiver<T>,
    total: usize,
}

/// Public alias for the production pool: holds [`SessionTriplet`] instances.
pub type SessionPool = Pool<SessionTriplet>;

impl<T> Pool<T> {
    /// Create a pool pre-filled with the given items.
    pub fn new(items: Vec<T>) -> Self {
        let total = items.len();
        // Bounded channel with capacity == total: send is always immediate
        // (try_send never returns Full while we own the only sender for
        // checked-out items), and the channel's internal queue holds the
        // available pool inventory.
        let (sender, receiver) = async_channel::bounded(total.max(1));
        for item in items {
            sender
                .try_send(item)
                .expect("channel capacity matches item count");
        }
        Self {
            sender,
            receiver,
            total,
        }
    }

    /// Checkout an item from the pool. Awaits FIFO if none available.
    ///
    /// Returns [`PoolError::Closed`] if the pool was shut down via
    /// [`close`](Self::close) before an item became available.
    pub async fn checkout(&self) -> Result<PoolGuard<'_, T>, PoolError> {
        match self.receiver.recv().await {
            Ok(item) => Ok(PoolGuard {
                pool: self,
                item: Some(item),
            }),
            Err(_) => Err(PoolError::Closed),
        }
    }

    /// Checkout an item from the pool synchronously (blocks until one is
    /// available).  This is the FFI-friendly counterpart to [`checkout`](Self::checkout).
    ///
    /// Returns [`PoolError::Closed`] if the pool was shut down.
    pub fn checkout_blocking(&self) -> Result<PoolGuard<'_, T>, PoolError> {
        match self.receiver.recv_blocking() {
            Ok(item) => Ok(PoolGuard {
                pool: self,
                item: Some(item),
            }),
            Err(_) => Err(PoolError::Closed),
        }
    }

    /// Close the pool: all current and future [`checkout`](Self::checkout)
    /// callers resolve to [`PoolError::Closed`]. Used by graceful shutdown.
    /// Idempotent.
    pub fn close(&self) {
        self.sender.close();
        self.receiver.close();
    }

    /// Total number of items the pool was created with.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Number of currently available (not checked-out) items. O(1).
    pub fn available(&self) -> usize {
        self.receiver.len()
    }
}

/// RAII guard that auto-checks-in an item when dropped.
///
/// Returned by [`Pool::checkout`]. Deref to access the inner item.
/// On drop (including panic unwind) the item is returned to the pool;
/// if the pool was closed in the meantime the item is silently dropped.
pub struct PoolGuard<'a, T> {
    pool: &'a Pool<T>,
    item: Option<T>,
}

impl<T> PoolGuard<'_, T> {
    /// Strip the lifetime so the guard can be moved into a `'static`
    /// context (e.g. `tokio::task::spawn_blocking`). Returns the owned
    /// item together with an [`OwnedReservation`] that must receive the
    /// item back via [`OwnedReservation::checkin`] when the blocking task
    /// is done. Forgets the original guard so the inner Drop does not also
    /// try to check-in.
    pub fn into_owned(mut self) -> (T, OwnedReservation<T>) {
        let item = self
            .item
            .take()
            .expect("PoolGuard::into_owned called after drop");
        let reservation = OwnedReservation {
            sender: self.pool.sender.clone(),
        };
        (item, reservation)
    }
}

impl<T> Deref for PoolGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .expect("PoolGuard accessed after item taken")
    }
}

impl<T> DerefMut for PoolGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .expect("PoolGuard accessed after item taken")
    }
}

impl<T> Drop for PoolGuard<'_, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            // Best-effort checkin. `try_send` is non-blocking and the
            // channel capacity equals total items, so it can only fail
            // if the pool was closed — in which case dropping the item
            // is the right thing.
            let _ = self.pool.sender.try_send(item);
        }
    }
}

/// Owned counterpart to [`PoolGuard`] for `'static` contexts (e.g.
/// `spawn_blocking`). The item is returned to the pool automatically on Drop
/// via [`Self::checkin`], or if the guard is forgotten, via the Drop impl.
///
/// After a panic the guard is dropped during unwind, so the item is recovered
/// without requiring the caller to manually invoke `checkin`.
pub struct OwnedReservation<T> {
    sender: async_channel::Sender<T>,
}

impl<T> OwnedReservation<T> {
    /// Return the item to the pool from a synchronous (blocking) context.
    /// Silently drops the item if the pool has been closed.
    pub fn checkin(self, item: T) {
        let _ = self.sender.try_send(item);
    }

    /// Create an RAII guard that holds both the item and the reservation.
    /// On drop (including during panic unwind) the item is returned to the pool.
    pub fn guard(self, item: T) -> PoolItemGuard<T> {
        PoolItemGuard {
            reservation: self,
            item: Some(item),
        }
    }
}

/// RAII guard that couples an owned pool item with its reservation.
///
/// On drop the item is automatically checked back into the pool. This is the
/// recommended pattern for `spawn_blocking` tasks where a panic would otherwise
/// leak the pool slot.
pub struct PoolItemGuard<T> {
    reservation: OwnedReservation<T>,
    item: Option<T>,
}

impl<T> PoolItemGuard<T> {
    /// Mutable access to the inner item.
    pub fn item_mut(&mut self) -> &mut T {
        self.item
            .as_mut()
            .expect("PoolItemGuard item already taken")
    }

    /// Immutable access to the inner item.
    pub fn item(&self) -> &T {
        self.item
            .as_ref()
            .expect("PoolItemGuard item already taken")
    }

    /// Consume the guard and return the item, **without** checking it back in.
    /// The caller is responsible for returning the item via `checkin`.
    pub fn into_inner(mut self) -> T {
        self.item.take().expect("PoolItemGuard item already taken")
    }
}

impl<T> Deref for PoolItemGuard<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.item()
    }
}

impl<T> DerefMut for PoolItemGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.item_mut()
    }
}

impl<T> Drop for PoolItemGuard<T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            let _ = self.reservation.sender.try_send(item);
        }
    }
}

/// State passed to Zipformer's stateless decoder on every step.
///
/// Holds the rolling [`CONTEXT_SIZE`]-token window the decoder embedding
/// reads, plus a running blank counter used by streaming endpointing.
/// Initialised left-padded with `blank_id` so the very first decoder
/// invocation sees `[<blk>, <blk>]`.
#[non_exhaustive]
pub struct DecoderState {
    /// Last [`CONTEXT_SIZE`] non-blank token ids (left-padded with `blank_id`).
    pub tokens: Vec<i64>,
    /// Blank token id (cached so [`Self::push_token`] can reset state without
    /// re-reading it from the engine).
    pub blank_id: usize,
    /// Count of consecutive blank frames (used for endpointing).
    pub consecutive_blanks: usize,
}

impl StreamingState {
    /// Reset overlap-buffer state for the start of a new utterance.
    /// Called by the VAD path after a completed speech segment is emitted.
    pub fn reset_utterance_state(&mut self) {
        self.decoder = DecoderState::new(self.blank_id);
        self.accumulated_text = Arc::new(String::new());
        self.accumulated_words = Arc::new(Vec::new());
        self.feature_window.clear();
        self.prev_window_words.clear();
        self.total_frames = 0;
    }
}

impl DecoderState {
    /// Create a fresh decoder state with the context window left-padded with
    /// `blank_id` and zero blank streak.
    pub fn new(blank_id: usize) -> Self {
        Self {
            tokens: vec![blank_id as i64; CONTEXT_SIZE],
            blank_id,
            consecutive_blanks: 0,
        }
    }

    /// Slide a newly emitted non-blank token into the context window,
    /// dropping the oldest entry to keep the length at [`CONTEXT_SIZE`].
    pub fn push_token(&mut self, token: i64) {
        // VecDeque would be cleaner but the window is fixed at CONTEXT_SIZE,
        // so a `rotate_left + assign` keeps the tensor view contiguous.
        self.tokens.rotate_left(1);
        let last = self.tokens.last_mut().expect("CONTEXT_SIZE > 0");
        *last = token;
    }
}

/// A recognized word with timing and confidence metadata.
///
/// Produced by the RNN-T decoder during [`Engine::process_chunk`] or [`Engine::transcribe_file`].
/// Timestamps are in seconds relative to the start of the audio stream.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct WordInfo {
    /// The recognized word text (BPE tokens joined, `▁` stripped).
    pub word: String,
    /// Start time in seconds from the beginning of the audio stream.
    pub start: f64,
    /// End time in seconds from the beginning of the audio stream.
    pub end: f64,
    /// Softmax confidence score (0.0–1.0), averaged over constituent BPE tokens.
    pub confidence: f32,
    /// Speaker label from diarization (zero-based index). Omitted if diarization is disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<u32>,
}

/// Per-connection diarization state accumulating audio and speaker assignments.
#[cfg(feature = "diarization")]
pub struct DiarizationStreamState {
    /// Raw 16 kHz f32 samples accumulated since the last embedding extraction.
    pub audio_buffer: Vec<f32>,
    /// Online speaker cluster tracking centroids across the session.
    pub cluster: diarization::SpeakerCluster,
    /// Speaker ID assigned to the most recent segment.
    pub current_speaker: Option<u32>,
}

/// Per-connection streaming state that persists across audio chunks.
///
/// Created via [`Engine::create_state`]. Holds the decoder LSTM state, an audio
/// sample buffer for incomplete frames, and accumulated transcript text/words.
/// Pass this to [`Engine::process_chunk`] for each incoming audio chunk and
/// [`Engine::flush_state`] when the stream ends.
#[non_exhaustive]
pub struct StreamingState {
    /// Decoder state (persisted across chunks).
    pub decoder: DecoderState,
    /// Online FBANK feature extractor.
    pub online: OnlineFeature,
    /// Number of mel frames already consumed from `online`.
    pub frames_seen: usize,
    /// Accumulated transcript text across chunks (reset on endpointing).
    /// Arc allows O(1) clone when emitting partial segments.
    pub accumulated_text: Arc<String>,
    /// Accumulated words with timestamps (reset on endpointing).
    /// Arc allows O(1) clone when emitting partial segments.
    pub accumulated_words: Arc<Vec<WordInfo>>,
    /// Absolute encoder frame offset for the next window (after subsampling-by-4).
    pub total_frames: usize,
    /// Mel feature frames waiting to fill the next streaming window.
    pub feature_window: Vec<f32>,
    /// Words from the previous window, used for overlap merging.
    pub prev_window_words: Vec<WordInfo>,
    /// Streaming configuration (window / overlap sizes).
    pub config: StreamingConfig,
    /// Blank token id cached so the state can be reset between utterances
    /// without re-reading the tokenizer.
    pub blank_id: usize,
    /// Per-connection Silero VAD session (only when VAD is enabled).
    pub vad_session: Option<silero::Session>,
    /// Per-connection VAD stream state (recurrent memory + pending samples).
    pub vad_stream_state: Option<silero::StreamState>,
    /// Per-connection VAD speech segmenter.
    pub vad_segmenter: Option<silero::SpeechSegmenter>,
    /// Accumulated raw audio samples for VAD-based segmentation.
    pub vad_audio_buffer: Vec<f32>,
    /// Sample offset of `vad_audio_buffer[0]` relative to the start of the stream.
    pub vad_sample_offset: u64,
    /// Completed VAD utterances waiting for async offline ASR.
    /// Populated by `process_chunk_vad`, drained by the WebSocket handler.
    pub vad_pending_asr: Vec<Vec<f32>>,
    /// Diarization state (present only when diarization is enabled).
    #[cfg(feature = "diarization")]
    pub diarization_state: Option<DiarizationStreamState>,
}

/// ONNX Runtime inference engine for Zipformer-vi RNN-T.
///
/// Thread-safe: inference sessions live in a [`SessionPool`] so `Engine` can be
/// shared across connections via `Arc<Engine>`. The pool size acts as the
/// concurrency limit — no separate semaphore needed. Typical usage:
///
/// ```ignore
/// let engine = Engine::load("~/.phostt/models")?;
/// let mut guard = engine.pool.checkout().await?;
/// let text = engine.transcribe_file("audio.wav", &mut guard)?;
/// // guard is returned to the pool on drop
/// ```
///
/// For streaming recognition, use [`create_state`](Engine::create_state) +
/// [`process_chunk`](Engine::process_chunk) + [`flush_state`](Engine::flush_state).
pub struct Engine {
    /// Pool of ONNX session triplets for concurrent inference.
    pub pool: SessionPool,
    tokenizer: Tokenizer,
    mel: MelSpectrogram,
    /// Overlap-buffer streaming configuration.
    streaming_config: StreamingConfig,
    /// When true, use Silero VAD for speech segmentation instead of the
    /// fixed-size overlap-buffer. Each detected utterance is transcribed
    /// offline, eliminating boundary artefacts.
    vad_enabled: bool,
    /// Speaker encoder for diarization (None if model file is absent).
    #[cfg(feature = "diarization")]
    pub speaker_encoder: Option<diarization::SpeakerEncoder>,
}

impl Engine {
    /// Size of the BPE vocabulary the loaded tokenizer covers. Exposed so the
    /// REST `/v1/models` handler can report the real value instead of a
    /// hardcoded literal that would drift if the upstream model rev changes.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// Build a test-only engine with an empty pool. Used by server handler
    /// unit tests so they can exercise guard paths without loading ONNX models.
    #[cfg(test)]
    pub fn test_stub() -> Self {
        Self {
            pool: Pool::new(vec![]),
            tokenizer: Tokenizer::from_tokens(vec!["<blk>".into(), "a".into()], 0),
            mel: MelSpectrogram::new(),
            streaming_config: StreamingConfig::default(),
            vad_enabled: false,
            #[cfg(feature = "diarization")]
            speaker_encoder: None,
        }
    }

    /// Load ONNX models from the given directory and create an inference engine.
    ///
    /// Creates a pool of [`DEFAULT_POOL_SIZE`] session triplets for concurrent inference.
    /// Expects files: `encoder.int8.onnx`, `decoder.onnx`, `joiner.int8.onnx`,
    /// `bpe.model`, and `tokens.txt` — the layout published by sherpa-onnx in
    /// `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2`.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::ModelLoad`] if model files are missing or ONNX session creation fails.
    pub fn load(model_dir: &str) -> Result<Self, GigasttError> {
        Self::load_with_pool_size(model_dir, DEFAULT_POOL_SIZE)
    }

    /// Load ONNX models with a custom pool size (default streaming config).
    pub fn load_with_pool_size(model_dir: &str, pool_size: usize) -> Result<Self, GigasttError> {
        Self::load_with_pool_size_and_config(model_dir, pool_size, StreamingConfig::default())
    }

    /// Load ONNX models with a custom pool size and streaming configuration.
    pub fn load_with_pool_size_and_config(
        model_dir: &str,
        pool_size: usize,
        streaming_config: StreamingConfig,
    ) -> Result<Self, GigasttError> {
        Self::load_with_pool_size_and_config_and_vad(model_dir, pool_size, streaming_config, false)
    }

    /// Load ONNX models with custom pool size, streaming config, and VAD flag.
    pub fn load_with_pool_size_and_config_and_vad(
        model_dir: &str,
        pool_size: usize,
        streaming_config: StreamingConfig,
        vad_enabled: bool,
    ) -> Result<Self, GigasttError> {
        streaming_config
            .validate()
            .map_err(|e| GigasttError::ModelLoad(format!("invalid streaming config: {e}")))?;
        let dir = Path::new(model_dir);
        if !dir.join("encoder.int8.onnx").exists() {
            return Err(GigasttError::ModelLoad(format!(
                "encoder.int8.onnx not found in {model_dir}"
            )));
        }
        Self::load_inner(dir, model_dir, pool_size, streaming_config, vad_enabled)
            .map_err(|e| GigasttError::ModelLoad(format!("{e:#}")))
    }

    /// Load a single set of encoder/decoder/joiner ONNX sessions from disk.
    fn load_sessions(
        dir: &Path,
        prepacked: &ort::session::builder::PrepackedWeights,
    ) -> anyhow::Result<(Session, Session, Session)> {
        // Zipformer-vi ships pre-quantized — there is no FP32 fallback to choose.
        let encoder_path = dir.join("encoder.int8.onnx");

        #[cfg(all(feature = "coreml", not(feature = "cuda")))]
        let (encoder, decoder, joiner) = {
            // CoreML has its own cache (`coreml_cache/`) for compiled subgraphs.
            // We do NOT call `.with_optimized_model_path(...)` here: CoreML EP
            // replaces part of the graph with compiled nodes that cannot be
            // re-serialized as ONNX, and ORT errors out with
            // `Unable to serialize model as it contains compiled nodes`
            // on macOS 14+. The CoreML cache below is sufficient.
            let cache_dir = dir.join("coreml_cache");
            let coreml_ep = ep::CoreML::default()
                .with_compute_units(ep::coreml::ComputeUnits::CPUAndNeuralEngine)
                .with_specialization_strategy(ep::coreml::SpecializationStrategy::FastPrediction)
                .with_model_cache_dir(cache_dir.to_string_lossy())
                .build();

            let encoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([coreml_ep.clone()])
                .map_err(ort_err)?
                .commit_from_file(&encoder_path)
                .map_err(ort_err)?;
            let decoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([coreml_ep.clone()])
                .map_err(ort_err)?
                .commit_from_file(dir.join("decoder.onnx"))
                .map_err(ort_err)?;
            let joiner = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([coreml_ep])
                .map_err(ort_err)?
                .commit_from_file(dir.join("joiner.int8.onnx"))
                .map_err(ort_err)?;
            (encoder, decoder, joiner)
        };

        #[cfg(feature = "cuda")]
        let (encoder, decoder, joiner) = {
            // CUDA EP compiles subgraphs that cannot be re-serialized as ONNX,
            // so we do NOT call `.with_optimized_model_path(...)` here — same
            // reason as the CoreML block. ORT's CUDA EP keeps its own caches
            // internally.
            let cuda_ep = ep::CUDA::default().build();

            let encoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([cuda_ep.clone()])
                .map_err(ort_err)?
                .commit_from_file(&encoder_path)
                .map_err(ort_err)?;
            let decoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([cuda_ep.clone()])
                .map_err(ort_err)?
                .commit_from_file(dir.join("decoder.onnx"))
                .map_err(ort_err)?;
            let joiner = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_execution_providers([cuda_ep])
                .map_err(ort_err)?
                .commit_from_file(dir.join("joiner.int8.onnx"))
                .map_err(ort_err)?;
            (encoder, decoder, joiner)
        };

        #[cfg(not(any(feature = "coreml", feature = "cuda")))]
        let (encoder, decoder, joiner) = {
            let cache_dir = dir.join("optimized_cache");
            std::fs::create_dir_all(&cache_dir).ok();
            let encoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .with_optimized_model_path(cache_dir.join("encoder_optimized.onnx"))
                .map_err(ort_err)?
                .commit_from_file(&encoder_path)
                .map_err(ort_err)?;
            let decoder = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .commit_from_file(dir.join("decoder.onnx"))
                .map_err(ort_err)?;
            let joiner = Session::builder()
                .map_err(ort_err)?
                .with_prepacked_weights(prepacked)
                .map_err(ort_err)?
                .commit_from_file(dir.join("joiner.int8.onnx"))
                .map_err(ort_err)?;
            (encoder, decoder, joiner)
        };

        Ok((encoder, decoder, joiner))
    }

    fn load_inner(
        dir: &Path,
        model_dir: &str,
        pool_size: usize,
        streaming_config: StreamingConfig,
        vad_enabled: bool,
    ) -> anyhow::Result<Self> {
        tracing::info!(
            "Loading Zipformer-vi INT8 ONNX models from {model_dir} (pool_size={pool_size})..."
        );

        #[cfg(feature = "coreml")]
        tracing::info!("Using CoreML execution provider (Neural Engine + CPU)");
        #[cfg(feature = "cuda")]
        tracing::info!("Using CUDA execution provider (falls back to CPU if unavailable)");
        #[cfg(not(any(feature = "coreml", feature = "cuda")))]
        tracing::info!("Using CPU execution provider");

        // Shared prepacked weights container (Arc-based, thread-safe)
        let prepacked = ort::session::builder::PrepackedWeights::new();

        let triplets: Vec<SessionTriplet> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..pool_size)
                .map(|i| {
                    let pp = &prepacked;
                    s.spawn(move || {
                        tracing::info!(
                            "Loading session triplet {}/{pool_size} (shared weights)",
                            i + 1
                        );
                        let (encoder, decoder, joiner) = Self::load_sessions(dir, pp)?;
                        Ok(SessionTriplet {
                            encoder,
                            decoder,
                            joiner,
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| anyhow::anyhow!("Thread panicked during model loading"))?
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })?;

        let tokenizer = Tokenizer::load(&dir.join("tokens.txt"))?;
        let mel = MelSpectrogram::new();

        tracing::info!(
            "Models loaded (vocab_size={}, pool_size={pool_size})",
            tokenizer.vocab_size()
        );

        #[cfg(feature = "diarization")]
        let speaker_encoder = match diarization::SpeakerEncoder::load(dir) {
            Ok(enc) => {
                tracing::info!("Speaker encoder loaded (diarization available)");
                Some(enc)
            }
            Err(e) => {
                tracing::warn!("Speaker encoder not loaded, diarization unavailable: {e:#}");
                None
            }
        };

        Ok(Self {
            pool: SessionPool::new(triplets),
            tokenizer,
            mel,
            streaming_config,
            vad_enabled,
            #[cfg(feature = "diarization")]
            speaker_encoder,
        })
    }

    /// Return `true` if a speaker encoder is loaded and diarization is available.
    #[cfg(feature = "diarization")]
    pub fn has_speaker_encoder(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Create a fresh streaming state for a new connection.
    ///
    /// Pass `diarization_enabled = true` to activate speaker diarization for
    /// this session. Without the `diarization` feature or a loaded speaker
    /// encoder, the flag is silently ignored (a `warn!` is emitted when the
    /// caller asked for diarization but the build does not support it, so the
    /// contract mismatch is visible in logs).
    pub fn create_state(&self, diarization_enabled: bool) -> Result<StreamingState, GigasttError> {
        #[cfg(feature = "diarization")]
        let diarization_state = if diarization_enabled && self.speaker_encoder.is_some() {
            Some(DiarizationStreamState {
                audio_buffer: Vec::new(),
                cluster: diarization::SpeakerCluster::new(),
                current_speaker: None,
            })
        } else {
            None
        };

        #[cfg(not(feature = "diarization"))]
        if diarization_enabled {
            tracing::warn!(
                "diarization_enabled=true ignored: build lacks the `diarization` feature"
            );
        }

        let computer = FbankComputer::new(features::phostt_fbank_options())
            .map_err(|e| GigasttError::Inference(format!("FBANK init failed: {e}")))?;
        let online = OnlineFeature::new(FeatureComputer::Fbank(computer));

        let (vad_session, vad_stream_state, vad_segmenter) = if self.vad_enabled {
            let session = silero::Session::bundled()
                .map_err(|e| GigasttError::ModelLoad(format!("silero VAD load failed: {e}")))?;
            let stream = silero::StreamState::new(silero::SampleRate::Rate16k);
            let segmenter = silero::SpeechSegmenter::new(silero::SpeechOptions::default());
            (Some(session), Some(stream), Some(segmenter))
        } else {
            (None, None, None)
        };

        let blank_id = self.tokenizer.blank_id();

        Ok(StreamingState {
            decoder: DecoderState::new(blank_id),
            online,
            frames_seen: 0,
            accumulated_text: Arc::new(String::new()),
            accumulated_words: Arc::new(Vec::new()),
            total_frames: 0,
            feature_window: Vec::new(),
            prev_window_words: Vec::new(),
            config: self.streaming_config.clone(),
            blank_id,
            vad_session,
            vad_stream_state,
            vad_segmenter,
            vad_audio_buffer: Vec::new(),
            vad_sample_offset: 0,
            vad_pending_asr: Vec::new(),
            #[cfg(feature = "diarization")]
            diarization_state,
        })
    }

    /// Process a chunk of 16kHz f32 audio samples and return any new transcript segments.
    ///
    /// Returns [`TranscriptSegment`] with `is_final == false` during speech (Partial),
    /// and `is_final == true` on endpointing (~600ms silence detected).
    /// Streaming state (LSTM hidden/cell, leftover audio, accumulated text) is maintained in `state`.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::Inference`] if the ONNX runtime fails.
    pub fn process_chunk(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<TranscriptSegment>, GigasttError> {
        if samples.is_empty() {
            return Ok(vec![]);
        }

        // VAD path: segment speech with Silero VAD, transcribe each utterance offline.
        if state.vad_session.is_some() {
            return self.process_chunk_vad(samples, state, triplet);
        }

        self.process_chunk_overlap(samples, state, triplet)
    }

    /// Overlap-buffer streaming path (the original non-VAD logic).
    fn process_chunk_overlap(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<TranscriptSegment>, GigasttError> {
        // Keep a copy of the 16kHz samples for diarization before the buffer
        // logic potentially pads/realigns them. Skip allocation when diarization
        // is not active for this session.
        #[cfg(feature = "diarization")]
        let samples_16k_copy = if state.diarization_state.is_some() {
            Some(samples.to_vec())
        } else {
            None
        };

        state
            .online
            .accept_waveform(TARGET_SAMPLE_RATE as f32, samples);

        let ready = state.online.num_frames_ready();
        let new_frames = ready.saturating_sub(state.frames_seen);
        if new_frames == 0 {
            return Ok(vec![]);
        }

        let new_features =
            features::extract_online_frames(&state.online, state.frames_seen, new_frames);
        state.frames_seen = ready;
        state.feature_window.extend_from_slice(&new_features);

        let mut emitted_words: Vec<WordInfo> = Vec::new();
        let mut endpoint = false;

        while state.feature_window.len() / N_MELS >= state.config.window_frames {
            let num_frames = state.config.window_frames;
            let features = &state.feature_window[..num_frames * N_MELS];
            let frame_offset = state.total_frames;

            let (window_words, window_endpoint, _enc_len) = self
                .run_inference(
                    triplet,
                    features,
                    num_frames,
                    &mut state.decoder,
                    frame_offset,
                )
                .map_err(|e| GigasttError::Inference(format!("{e:#}")))?;

            let delta = Self::delta_words(
                &window_words,
                &state.prev_window_words,
                state.config.fuzzy_match_threshold,
            );
            emitted_words.extend(delta);
            state.prev_window_words = window_words;

            // Shift window, keeping overlap
            let shift = state.config.shift_frames() * N_MELS;
            state.feature_window.drain(..shift);
            state.total_frames += state.config.shift_encoder_frames();

            if window_endpoint {
                endpoint = true;
                break;
            }
        }

        // --- Diarization: accumulate audio, extract embeddings, assign speakers ---
        #[cfg(feature = "diarization")]
        if let (Some(dia), Some(copy), Some(enc)) = (
            state.diarization_state.as_mut(),
            samples_16k_copy.as_ref(),
            self.speaker_encoder.as_ref(),
        ) {
            dia.audio_buffer.extend_from_slice(copy);

            while dia.audio_buffer.len() >= diarization::SEGMENT_SAMPLES {
                let segment: Vec<f32> = dia
                    .audio_buffer
                    .drain(..diarization::SEGMENT_SAMPLES)
                    .collect();
                match enc.extract_embedding(&segment) {
                    Ok(embedding) => {
                        let speaker = dia.cluster.assign(&embedding);
                        dia.current_speaker = Some(speaker);
                    }
                    Err(e) => {
                        tracing::warn!("Embedding extraction failed: {e:#}");
                    }
                }
            }

            // Annotate all words in this chunk with current speaker
            if let Some(speaker_id) = dia.current_speaker {
                for w in &mut emitted_words {
                    w.speaker = Some(speaker_id);
                }
            }
        }

        if emitted_words.is_empty() && !endpoint {
            return Ok(vec![]);
        }

        // Accumulate new words — make_mut clones only if refcount > 1
        let acc_text = Arc::make_mut(&mut state.accumulated_text);
        let acc_words = Arc::make_mut(&mut state.accumulated_words);
        for w in &emitted_words {
            if !acc_text.is_empty() {
                acc_text.push(' ');
            }
            acc_text.push_str(&w.word);
        }
        acc_words.extend(emitted_words);

        let text = Arc::clone(&state.accumulated_text);
        let words = Arc::clone(&state.accumulated_words);
        let ts = now_timestamp();

        if endpoint {
            // Endpoint detected: emit Final and reset accumulation
            state.accumulated_text = Arc::new(String::new());
            state.accumulated_words = Arc::new(Vec::new());
            state.decoder.consecutive_blanks = 0;
            state.prev_window_words.clear();
            Ok(vec![TranscriptSegment {
                text,
                words,
                is_final: true,
                timestamp: ts,
            }])
        } else {
            // Ongoing speech: emit Partial
            Ok(vec![TranscriptSegment {
                text,
                words,
                is_final: false,
                timestamp: ts,
            }])
        }
    }

    /// VAD-based streaming: feed samples to Silero VAD, emit a Final segment
    /// for each completed speech utterance.
    fn process_chunk_vad(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<TranscriptSegment>, GigasttError> {
        state.vad_audio_buffer.extend_from_slice(samples);

        // --- Run VAD inference in a scoped block so segmenter borrow ends ---
        let (speech_segments, is_active) = {
            let session = state.vad_session.as_mut().unwrap();
            let stream = state.vad_stream_state.as_mut().unwrap();
            let segmenter = state.vad_segmenter.as_mut().unwrap();

            let mut segments: Vec<silero::SpeechSegment> = Vec::new();
            session
                .process_stream(stream, samples, |probability| {
                    if let Some(segment) = segmenter.push_probability(probability) {
                        segments.push(segment);
                    }
                })
                .map_err(|e| GigasttError::Inference(format!("VAD inference failed: {e}")))?;

            let active = segmenter.is_active();
            (segments, active)
        };

        let mut emitted_segments: Vec<TranscriptSegment> = Vec::new();
        let buffer_start = state.vad_sample_offset;

        // Queue completed utterances for async offline ASR so VAD can keep
        // listening without blocking on the encoder.
        for segment in &speech_segments {
            let buf_start = segment.start_sample().saturating_sub(buffer_start) as usize;
            let buf_end = segment.end_sample().saturating_sub(buffer_start) as usize;
            if buf_end > state.vad_audio_buffer.len() {
                tracing::warn!("VAD segment extends beyond audio buffer, skipping");
                continue;
            }
            let speech_samples = &state.vad_audio_buffer[buf_start..buf_end];
            if speech_samples.is_empty() {
                continue;
            }
            state.vad_pending_asr.push(speech_samples.to_vec());
            state.reset_utterance_state();
        }

        // Drain processed audio from the buffer.
        if let Some(last_seg) = speech_segments.last() {
            let remove_up_to = (last_seg.end_sample().saturating_sub(buffer_start)) as usize;
            if remove_up_to <= state.vad_audio_buffer.len() {
                state.vad_audio_buffer.drain(..remove_up_to);
                state.vad_sample_offset += remove_up_to as u64;
            }
        }

        // Emit Partial segments while speech is still active.
        if is_active {
            let partials = self.process_chunk_overlap(samples, state, triplet)?;
            emitted_segments.extend(partials);
        }

        Ok(emitted_segments)
    }

    /// VAD flush: process any trailing pending samples and close the final
    /// open speech segment.
    fn flush_state_vad(
        &self,
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Option<TranscriptSegment> {
        let session = state.vad_session.as_mut()?;
        let stream = state.vad_stream_state.as_mut()?;
        let segmenter = state.vad_segmenter.as_mut()?;

        // Flush pending VAD samples.
        if let Ok(Some(probability)) = session.flush_stream(stream)
            && let Some(segment) = segmenter.push_probability(probability)
        {
            let buffer_start = state.vad_sample_offset;
            let buf_start = segment.start_sample().saturating_sub(buffer_start) as usize;
            let buf_end = (segment.end_sample().saturating_sub(buffer_start) as usize)
                .min(state.vad_audio_buffer.len());
            if buf_start < buf_end {
                let speech_samples = &state.vad_audio_buffer[buf_start..buf_end];
                if let Ok(result) = self.transcribe_samples(speech_samples, triplet)
                    && !result.text.is_empty()
                {
                    state.reset_utterance_state();
                    return Some(TranscriptSegment {
                        text: Arc::new(result.text),
                        words: Arc::new(result.words),
                        is_final: true,
                        timestamp: now_timestamp(),
                    });
                }
            }
        }

        // Close any trailing open segment.
        if let Some(segment) = segmenter.finish() {
            let buffer_start = state.vad_sample_offset;
            let buf_start = segment.start_sample().saturating_sub(buffer_start) as usize;
            let buf_end = (segment.end_sample().saturating_sub(buffer_start) as usize)
                .min(state.vad_audio_buffer.len());
            if buf_start < buf_end {
                let speech_samples = &state.vad_audio_buffer[buf_start..buf_end];
                if let Ok(result) = self.transcribe_samples(speech_samples, triplet)
                    && !result.text.is_empty()
                {
                    state.reset_utterance_state();
                    return Some(TranscriptSegment {
                        text: Arc::new(result.text),
                        words: Arc::new(result.words),
                        is_final: true,
                        timestamp: now_timestamp(),
                    });
                }
            }
        }

        None
    }

    /// Flush accumulated text as a Final segment (called on Stop/Close).
    pub fn flush_state(
        &self,
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Option<TranscriptSegment> {
        // VAD path: flush pending samples and emit trailing segment.
        if state.vad_session.is_some() {
            return self.flush_state_vad(state, triplet);
        }

        state.online.input_finished();
        let ready = state.online.num_frames_ready();
        let new_frames = ready.saturating_sub(state.frames_seen);
        if new_frames > 0 {
            let new_features =
                features::extract_online_frames(&state.online, state.frames_seen, new_frames);
            state.feature_window.extend_from_slice(&new_features);
            state.frames_seen = ready;
        }

        if !state.feature_window.is_empty() {
            let num_frames = state.feature_window.len() / N_MELS;
            let features = &state.feature_window[..];
            let frame_offset = state.total_frames;
            let (window_words, _endpoint, _enc_len) = self
                .run_inference(
                    triplet,
                    features,
                    num_frames,
                    &mut state.decoder,
                    frame_offset,
                )
                .ok()?;
            let delta = Self::delta_words(
                &window_words,
                &state.prev_window_words,
                state.config.fuzzy_match_threshold,
            );
            let acc_text = Arc::make_mut(&mut state.accumulated_text);
            let acc_words = Arc::make_mut(&mut state.accumulated_words);
            for w in &delta {
                if !acc_text.is_empty() {
                    acc_text.push(' ');
                }
                acc_text.push_str(&w.word);
            }
            acc_words.extend(delta);
            state.prev_window_words = window_words;
            state.feature_window.clear();
            state.total_frames += num_frames / 4;
        }

        if state.accumulated_text.is_empty() {
            return None;
        }
        let seg = TranscriptSegment {
            text: Arc::clone(&state.accumulated_text),
            words: Arc::clone(&state.accumulated_words),
            is_final: true,
            timestamp: now_timestamp(),
        };
        Some(seg)
    }

    /// Transcribe an audio file to text (supports WAV, MP3, M4A/AAC, OGG, FLAC).
    ///
    /// Decodes the file to mono 16kHz, runs the full encoder+decoder pipeline,
    /// and returns the recognized text with word-level details and duration.
    ///
    /// # Errors
    ///
    /// Returns [`GigasttError::InvalidAudio`] if the file cannot be decoded, or
    /// [`GigasttError::Inference`] if the ONNX runtime fails.
    pub fn transcribe_file(
        &self,
        path: &str,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        let float_samples = audio::decode_audio_file(path)
            .map_err(|e| GigasttError::InvalidAudio(format!("{e:#}")))?;
        self.transcribe_samples(&float_samples, triplet)
    }

    /// Transcribe audio from raw bytes in memory (no temp file needed).
    ///
    /// Backwards-compatible shim: clones `data` into a [`bytes::Bytes`] and
    /// delegates to [`Engine::transcribe_bytes_shared`]. Prefer the shared
    /// variant on hot paths (REST/SSE) to avoid the extra copy.
    pub fn transcribe_bytes(
        &self,
        data: &[u8],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        self.transcribe_bytes_shared(bytes::Bytes::copy_from_slice(data), triplet)
    }

    /// Transcribe audio from a reference-counted [`bytes::Bytes`] buffer
    /// without cloning.
    ///
    /// Reuses the same decode/inference pipeline as [`Engine::transcribe_bytes`]
    /// but hands the buffer straight to symphonia via [`audio::decode_audio_bytes_shared`].
    /// This is the zero-copy entry point used by the REST upload handler so a
    /// 50 MiB `axum::body::Bytes` body stays as a single in-memory buffer
    /// instead of being cloned into a `Vec<u8>` before decode.
    pub fn transcribe_bytes_shared(
        &self,
        data: bytes::Bytes,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        let float_samples = audio::decode_audio_bytes_shared(data)
            .map_err(|e| GigasttError::InvalidAudio(format!("{e:#}")))?;
        self.transcribe_samples(&float_samples, triplet)
    }

    /// Run the full mel + encoder + RNN-T decode pipeline on an already-decoded
    /// 16 kHz f32 sample buffer. Shared tail of [`Engine::transcribe_file`] and
    /// [`Engine::transcribe_bytes_shared`].
    pub fn transcribe_samples(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, GigasttError> {
        let duration_s = float_samples.len() as f64 / (TARGET_SAMPLE_RATE as f64);

        let (features, num_frames) = self.mel.compute(float_samples);
        tracing::info!("Extracted {} mel frames", num_frames);

        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        let (words, _endpoint, _enc_len) = self
            .run_inference(triplet, &features, num_frames, &mut decoder_state, 0)
            .map_err(|e| GigasttError::Inference(format!("{e:#}")))?;
        let text: String = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(TranscribeResult {
            text,
            words,
            duration_s,
        })
    }

    fn run_inference(
        &self,
        triplet: &mut SessionTriplet,
        features: &[f32],
        num_frames: usize,
        decoder_state: &mut DecoderState,
        frame_offset: usize,
    ) -> anyhow::Result<(Vec<WordInfo>, bool, usize)> {
        // Zipformer encoder input: features [1, T, N_MELS] (frames-first)
        // + features_length [1] int64. Output: encoder_out [1, T', 512]
        // (Zipformer subsamples by 4) + encoder_out_lens [1] int64.
        let features_tensor =
            TensorRef::from_array_view(([1_usize, num_frames, N_MELS], features))?;
        let length_data = [num_frames as i64];
        let length_tensor = TensorRef::from_array_view(([1_usize], length_data.as_slice()))?;

        let enc_start = std::time::Instant::now();
        let encoder_outputs = triplet
            .encoder
            .run(ort::inputs![features_tensor, length_tensor])
            .context("Encoder inference failed")?;
        tracing::info!(
            elapsed_ms = enc_start.elapsed().as_millis() as u64,
            "encoder_inference"
        );

        let (_enc_shape, enc_data) = encoder_outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract encoder output")?;
        let (_len_shape, len_data) = encoder_outputs[1]
            .try_extract_tensor::<i64>()
            .context("Failed to extract encoder length")?;

        let enc_len = usize::try_from(len_data[0]).context("Negative encoder length")?;

        tracing::debug!("Encoder output: {} frames", enc_len);

        // Copy encoder data so we can release the encoder output borrow
        let enc_data_owned: Vec<f32> = enc_data.to_vec();
        drop(encoder_outputs);

        // RNN-T greedy decode
        let dec_start = std::time::Instant::now();
        let result = decode::greedy_decode(
            &mut triplet.decoder,
            &mut triplet.joiner,
            &enc_data_owned,
            enc_len,
            self.tokenizer.blank_id(),
            self.tokenizer.vocab_size(),
            decoder_state,
        )?;
        tracing::info!(
            elapsed_ms = dec_start.elapsed().as_millis() as u64,
            "greedy_decode"
        );

        // Convert token infos to words with timestamps
        let words = self.tokens_to_words(&result.tokens, frame_offset);

        let preview: String = words
            .iter()
            .take(10)
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let ellipsis = if words.len() > 10 { "..." } else { "" };
        tracing::info!(
            "Decoded {} tokens → \"{preview}{ellipsis}\"",
            result.tokens.len()
        );

        Ok((words, result.endpoint_detected, enc_len))
    }

    /// Convert decoded tokens into words with timestamps and confidence.
    fn tokens_to_words(&self, tokens: &[decode::TokenInfo], frame_offset: usize) -> Vec<WordInfo> {
        // Fast path for the no-speech frame case. The word-boundary loop
        // below would also return `Vec::new()` on an empty input, but
        // bailing early skips the allocation of the intermediate state.
        if tokens.is_empty() {
            return Vec::new();
        }

        // Group tokens by words (BPE ▁ marks word boundaries)
        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut word_start_frame: Option<usize> = None;
        let mut word_end_frame: usize = 0;
        let mut word_confidences: Vec<f32> = Vec::new();

        for token in tokens {
            let token_text = self.tokenizer.token_text(token.token_id);
            let is_word_boundary = token_text.starts_with('\u{2581}');

            if is_word_boundary && !current_word.is_empty() {
                // Emit previous word
                let avg_conf: f32 = if word_confidences.is_empty() {
                    1.0
                } else {
                    word_confidences.iter().sum::<f32>() / word_confidences.len() as f32
                };
                words.push(WordInfo {
                    word: current_word.clone(),
                    start: (word_start_frame.unwrap_or(0) + frame_offset) as f64
                        * SECONDS_PER_FRAME,
                    end: (word_end_frame + frame_offset) as f64 * SECONDS_PER_FRAME,
                    confidence: avg_conf,
                    speaker: None,
                });
                current_word.clear();
                word_confidences.clear();
                word_start_frame = None;
            }

            let clean = token_text.replace('\u{2581}', "");
            if !clean.is_empty() {
                current_word.push_str(&clean);
                if word_start_frame.is_none() {
                    word_start_frame = Some(token.frame_index);
                }
                word_end_frame = token.frame_index;
                word_confidences.push(token.confidence);
            }
        }

        // Emit last word
        if !current_word.is_empty() {
            let avg_conf: f32 = if word_confidences.is_empty() {
                1.0
            } else {
                word_confidences.iter().sum::<f32>() / word_confidences.len() as f32
            };
            words.push(WordInfo {
                word: current_word,
                start: (word_start_frame.unwrap_or(0) + frame_offset) as f64 * SECONDS_PER_FRAME,
                end: (word_end_frame + frame_offset) as f64 * SECONDS_PER_FRAME,
                confidence: avg_conf,
                speaker: None,
            });
        }

        words
    }

    /// Return the words from `new` that are not already present in `prev`,
    /// using a suffix/prefix overlap merge with optional fuzzy matching.
    fn delta_words(new: &[WordInfo], prev: &[WordInfo], fuzzy_threshold: f32) -> Vec<WordInfo> {
        if prev.is_empty() {
            return new.to_vec();
        }
        let mut best = 0;
        for start in 0..prev.len() {
            let mut matched = 0;
            for (a, b) in new.iter().zip(prev[start..].iter()) {
                if words_match(&a.word, &b.word, fuzzy_threshold) {
                    matched += 1;
                } else {
                    break;
                }
            }
            if matched > best {
                best = matched;
            }
        }
        new[best..].to_vec()
    }
}

/// Compute Levenshtein edit distance between two strings.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_len = a.chars().count();
    let b_len = b.chars().count();
    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }
    let mut prev = vec![0; b_len + 1];
    let mut curr = vec![0; b_len + 1];
    for (j, item) in prev.iter_mut().enumerate().take(b_len + 1) {
        *item = j;
    }
    for (i, ca) in a.chars().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_len]
}

/// Normalized similarity in [0.0, 1.0]. 1.0 = identical.
fn word_similarity(a: &str, b: &str) -> f32 {
    if a == b {
        return 1.0;
    }
    let dist = levenshtein(a, b);
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - (dist as f32 / max_len as f32)
}

/// Check whether two words match, respecting the fuzzy threshold.
fn words_match(a: &str, b: &str, threshold: f32) -> bool {
    if threshold >= 1.0 {
        return a == b;
    }
    word_similarity(a, b) >= threshold
}

/// Result of file transcription, including word-level details.
#[derive(Debug, Clone, Serialize)]
pub struct TranscribeResult {
    pub text: String,
    pub words: Vec<WordInfo>,
    pub duration_s: f64,
}

/// A transcript segment emitted by the inference engine.
///
/// Partial segments (`is_final == false`) represent interim results that may change.
/// Final segments (`is_final == true`) represent completed utterances after endpointing.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct TranscriptSegment {
    /// Recognized text for this segment.
    pub text: Arc<String>,
    /// Individual words with timing and confidence metadata.
    pub words: Arc<Vec<WordInfo>>,
    /// Whether this segment is final (utterance complete) or partial (interim).
    pub is_final: bool,
    /// Unix timestamp (seconds since epoch) when this segment was produced.
    pub timestamp: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_state_new_left_pads_context_with_blank() {
        let blank_id = 0;
        let state = DecoderState::new(blank_id);
        assert_eq!(state.tokens.len(), CONTEXT_SIZE);
        assert!(state.tokens.iter().all(|&t| t == blank_id as i64));
        assert_eq!(state.blank_id, blank_id);
        assert_eq!(state.consecutive_blanks, 0);
    }

    #[test]
    fn test_decoder_state_push_token_slides_window() {
        let mut state = DecoderState::new(0);
        // CONTEXT_SIZE == 2 → start [0, 0], push 7 → [0, 7], push 9 → [7, 9].
        state.push_token(7);
        assert_eq!(state.tokens.last().copied(), Some(7));
        state.push_token(9);
        assert_eq!(state.tokens, vec![7, 9]);
    }

    #[test]
    fn test_decoder_state_custom_blank_id_seeds_context() {
        let state = DecoderState::new(42);
        assert!(state.tokens.iter().all(|&t| t == 42));
    }

    // ---- Pool tests (B.7) ---------------------------------------------------
    //
    // These exercise `Pool<T>` with synthetic items so the contract is
    // observable without loading ONNX models. `SessionPool = Pool<SessionTriplet>`
    // is just an alias, so any property proven here also holds for the real
    // pool.

    #[tokio::test]
    async fn test_pool_guard_returns_triplet_on_normal_drop() {
        let pool = Pool::new(vec![1u32, 2, 3]);
        assert_eq!(pool.available(), 3);
        {
            let _guard = pool.checkout().await.expect("checkout");
            assert_eq!(pool.available(), 2);
        }
        // Dropping the guard returns the item.
        assert_eq!(pool.available(), 3);
    }

    #[tokio::test]
    async fn test_pool_guard_returns_triplet_on_panic_unwind() {
        // The guard's Drop impl runs during unwind, so a panic between
        // checkout and the natural end of scope still restores capacity.
        let pool = std::sync::Arc::new(Pool::new(vec![1u32]));
        assert_eq!(pool.available(), 1);

        let pool_clone = pool.clone();
        let result = tokio::spawn(async move {
            let _guard = pool_clone.checkout().await.expect("checkout");
            assert_eq!(pool_clone.available(), 0);
            panic!("synthetic inference panic");
        })
        .await;
        assert!(result.is_err(), "spawned task must report the panic");

        // Capacity is restored thanks to PoolGuard::drop running on unwind.
        assert_eq!(pool.available(), 1);
    }

    #[tokio::test]
    async fn test_pool_close_wakes_waiters_with_closed() {
        // A waiter blocked in `checkout` after the inventory is exhausted
        // must resolve to PoolError::Closed when `close()` fires. Map the
        // borrowed guard to the `()` success path so the spawn doesn't
        // need to carry the pool's lifetime.
        let pool = std::sync::Arc::new(Pool::<u32>::new(vec![]));
        let waiter = tokio::spawn({
            let pool = pool.clone();
            async move { pool.checkout().await.map(|_g| ()) }
        });

        // Give the waiter a moment to park on the channel.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        pool.close();

        let res = waiter.await.expect("join");
        assert!(matches!(res, Err(PoolError::Closed)));
    }

    #[tokio::test]
    async fn test_pool_fifo_under_contention() {
        // With a single-slot pool and three queued waiters, the order of
        // wake-ups must match the order in which `checkout` was called.
        // `async_channel` is internally FIFO; this test guards against
        // accidental Mutex<mpsc> regressions that lose that property.
        let pool = std::sync::Arc::new(Pool::new(vec![0u32]));

        let primary = pool.checkout().await.expect("primary checkout");
        assert_eq!(pool.available(), 0);

        let waker_log = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for id in 0u32..3 {
            let pool = pool.clone();
            let log = waker_log.clone();
            handles.push(tokio::spawn(async move {
                let g = pool.checkout().await.expect("checkout");
                log.lock().await.push(id);
                drop(g);
            }));
            // Stagger spawns so each waiter is parked before the next one
            // is registered with the channel.
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }

        // Release the only inventory slot so the queued waiters can run.
        drop(primary);
        for h in handles {
            h.await.expect("join");
        }

        let log = waker_log.lock().await.clone();
        assert_eq!(log, vec![0, 1, 2], "waiters must wake in FIFO order");
    }

    #[tokio::test]
    async fn test_into_owned_for_spawn_blocking() {
        // `into_owned` strips the lifetime so the item can be moved into a
        // blocking thread, then `OwnedReservation::checkin` returns it.
        let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
        let guard = pool.checkout().await.expect("checkout");
        let (item, reservation) = guard.into_owned();

        let item = tokio::task::spawn_blocking(move || {
            // Pretend we're running blocking inference.
            assert_eq!(item, "triplet");
            reservation.checkin(item.clone());
            item
        })
        .await
        .expect("join");

        // After the blocking task returns the item, the pool is full again.
        assert_eq!(pool.available(), 1);
        assert_eq!(item, "triplet");
    }

    #[tokio::test]
    async fn test_pool_close_is_idempotent() {
        // `pool.close()` is wired into the shutdown hook; calling it twice
        // (e.g. shutdown signal + Drop) must not panic.
        let pool = Pool::<u32>::new(vec![]);
        pool.close();
        pool.close();
    }

    #[tokio::test]
    async fn test_pool_triplet_survives_panic() {
        // Spec 001: If a spawn_blocking task panics while holding an owned
        // triplet, the pool slot MUST be recovered automatically via Drop.
        let pool = std::sync::Arc::new(Pool::new(vec![String::from("triplet")]));
        let guard = pool.checkout().await.expect("checkout");
        let (item, reservation) = guard.into_owned();

        // Use the new PoolItemGuard pattern: item + reservation are coupled.
        // Even if the closure panics, the guard's Drop returns the slot.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let mut guard = reservation.guard(item);
            guard.push_str(" mutated");
            panic!("simulated inference panic");
        }));
        assert!(result.is_err(), "panic should have occurred");

        // The guard's Drop automatically returns the (possibly mutated) item.
        assert_eq!(
            pool.available(),
            1,
            "pool slot must be recovered after panic"
        );

        // Verify the item was actually returned (not just forgotten).
        let g = pool.checkout().await.expect("checkout after panic");
        assert_eq!(g.as_str(), "triplet mutated");
    }

    #[tokio::test]
    async fn test_pool_all_slots_recovered_after_panic_storm() {
        // Spec 001: If every checkout panics simultaneously, every slot must
        // still be recovered.
        let pool = std::sync::Arc::new(Pool::new(vec![1u32, 2, 3, 4]));
        assert_eq!(pool.available(), 4);

        let mut handles = Vec::new();
        for _ in 0..4 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let guard = pool.checkout().await.expect("checkout");
                let (item, reservation) = guard.into_owned();
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                    let guard = reservation.guard(item);
                    let _ = *guard;
                    panic!("simulated inference panic");
                }));
            }));
        }

        for h in handles {
            let _ = h.await;
        }

        assert_eq!(
            pool.available(),
            4,
            "all pool slots must be recovered after panic storm"
        );
    }

    // ---- StreamingConfig --------------------------------------------------

    #[test]
    fn test_streaming_config_defaults() {
        let cfg = StreamingConfig::default();
        assert_eq!(cfg.window_frames, 400);
        assert_eq!(cfg.overlap_frames, 100);
        assert_eq!(cfg.fuzzy_match_threshold, 1.0);
        assert_eq!(cfg.shift_frames(), 300);
        assert_eq!(cfg.shift_encoder_frames(), 75);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_streaming_config_validation() {
        assert!(
            StreamingConfig {
                window_frames: 0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            StreamingConfig {
                window_frames: 100,
                overlap_frames: 100,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            StreamingConfig {
                window_frames: 100,
                overlap_frames: 50,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            StreamingConfig {
                window_frames: 200,
                overlap_frames: 40,
                fuzzy_match_threshold: 1.5,
            }
            .validate()
            .is_err()
        );
        assert!(
            StreamingConfig {
                window_frames: 200,
                overlap_frames: 40,
                fuzzy_match_threshold: 0.8,
            }
            .validate()
            .is_ok()
        );
    }

    // ---- Fuzzy word match --------------------------------------------------

    #[test]
    fn test_levenshtein_basic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("ab", "abc"), 1);
    }

    #[test]
    fn test_word_similarity() {
        assert!((word_similarity("hello", "hello") - 1.0).abs() < f32::EPSILON);
        assert!(word_similarity("hello", "hallo") > 0.7);
        assert!(word_similarity("hello", "world") < 0.5);
    }

    #[test]
    fn test_words_match_exact_threshold() {
        assert!(words_match("hello", "hello", 1.0));
        assert!(!words_match("hello", "hallo", 1.0));
    }

    #[test]
    fn test_words_match_fuzzy() {
        assert!(words_match("hello", "hallo", 0.7));
        assert!(!words_match("hello", "world", 0.7));
    }

    #[test]
    fn test_delta_words_exact() {
        let mk = |w: &str| WordInfo {
            word: w.into(),
            start: 0.0,
            end: 0.0,
            confidence: 1.0,
            speaker: None,
        };
        let prev = vec![mk("a"), mk("b"), mk("c")];
        let new = vec![mk("b"), mk("c"), mk("d")];
        let delta = Engine::delta_words(&new, &prev, 1.0);
        assert_eq!(delta.len(), 1);
        assert_eq!(delta[0].word, "d");
    }

    #[test]
    fn test_delta_words_fuzzy() {
        let mk = |w: &str| WordInfo {
            word: w.into(),
            start: 0.0,
            end: 0.0,
            confidence: 1.0,
            speaker: None,
        };
        // "helio" vs "hello" — one char diff, 80% similarity
        let prev = vec![mk("a"), mk("hello")];
        let new = vec![mk("helio"), mk("b")];
        let delta_exact = Engine::delta_words(&new, &prev, 1.0);
        // exact: no match for "helio" vs "hello", so best = 0
        assert_eq!(delta_exact.len(), 2);

        let delta_fuzzy = Engine::delta_words(&new, &prev, 0.7);
        // fuzzy: "helio" ~= "hello", matched = 1, best = 1
        assert_eq!(delta_fuzzy.len(), 1);
        assert_eq!(delta_fuzzy[0].word, "b");
    }

    #[tokio::test]
    async fn test_vad_streaming_produces_same_text_as_offline() {
        // Skip if model is not downloaded.
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        let model_dir = home.as_ref().map(|p| p.join(".phostt/models"));
        if model_dir.is_none()
            || !model_dir
                .as_ref()
                .unwrap()
                .join("encoder.int8.onnx")
                .exists()
        {
            eprintln!("Skipping test_vad_streaming: model not found");
            return;
        }
        let model_dir = model_dir.unwrap();
        let wav_path = model_dir.join("test_wavs").join("0.wav");
        if !wav_path.exists() {
            eprintln!("Skipping test_vad_streaming: test WAV not found");
            return;
        }
        let samples = audio::decode_audio_file(wav_path.to_str().unwrap()).unwrap();

        let engine = Engine::load_with_pool_size_and_config_and_vad(
            model_dir.to_str().unwrap(),
            1,
            StreamingConfig::default(),
            true,
        )
        .unwrap();
        let mut triplet = engine.pool.checkout().await.unwrap();

        let offline = engine.transcribe_samples(&samples, &mut triplet).unwrap();
        let offline_text = offline.text;

        let mut state = engine.create_state(false).unwrap();
        let chunk_size = samples.len() / 3;
        let chunks = vec![
            &samples[..chunk_size],
            &samples[chunk_size..2 * chunk_size],
            &samples[2 * chunk_size..],
        ];

        let mut vad_text = String::new();
        for chunk in chunks {
            let segs = engine
                .process_chunk(chunk, &mut state, &mut triplet)
                .unwrap();
            // Drain async ASR queue (completed VAD utterances) and accumulate
            // the offline transcripts as Final text.
            for audio in state.vad_pending_asr.drain(..) {
                let result = engine.transcribe_samples(&audio, &mut triplet).unwrap();
                if !result.text.is_empty() {
                    if !vad_text.is_empty() {
                        vad_text.push(' ');
                    }
                    vad_text.push_str(&result.text);
                }
            }
            // Partial segments from the overlap-buffer are ignored for the
            // final-text comparison.
            let _ = segs;
        }
        if let Some(flush) = engine.flush_state(&mut state, &mut triplet) {
            if !vad_text.is_empty() {
                vad_text.push(' ');
            }
            vad_text.push_str(&flush.text);
        }

        let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(
            normalize(&vad_text),
            normalize(&offline_text),
            "VAD streaming transcript should match offline transcript"
        );
    }

    #[tokio::test]
    async fn test_vad_hybrid_emits_partials() {
        // Skip if model is not downloaded.
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        let model_dir = home.as_ref().map(|p| p.join(".phostt/models"));
        if model_dir.is_none()
            || !model_dir
                .as_ref()
                .unwrap()
                .join("encoder.int8.onnx")
                .exists()
        {
            eprintln!("Skipping test_vad_hybrid_emits_partials: model not found");
            return;
        }
        let model_dir = model_dir.unwrap();
        let wav_path = model_dir.join("test_wavs").join("0.wav");
        if !wav_path.exists() {
            eprintln!("Skipping test_vad_hybrid_emits_partials: test WAV not found");
            return;
        }
        let samples = audio::decode_audio_file(wav_path.to_str().unwrap()).unwrap();

        // Use a small window so overlap-buffer produces partials even on short audio.
        let config = StreamingConfig {
            window_frames: 100,
            overlap_frames: 20,
            fuzzy_match_threshold: 1.0,
        };
        let engine = Engine::load_with_pool_size_and_config_and_vad(
            model_dir.to_str().unwrap(),
            1,
            config,
            true,
        )
        .unwrap();
        let mut triplet = engine.pool.checkout().await.unwrap();
        let mut state = engine.create_state(false).unwrap();

        // Feed audio in small chunks so VAD sees active speech before the segment completes.
        let chunk_size = samples.len() / 10;
        let mut partial_count = 0usize;
        let mut final_count = 0usize;
        for i in 0..10 {
            let end = ((i + 1) * chunk_size).min(samples.len());
            let chunk = &samples[i * chunk_size..end];
            let segs = engine
                .process_chunk(chunk, &mut state, &mut triplet)
                .unwrap();
            for seg in &segs {
                if seg.is_final {
                    final_count += 1;
                } else {
                    partial_count += 1;
                }
            }
            // Drain async ASR queue and count the resulting Final segments.
            for audio in state.vad_pending_asr.drain(..) {
                let result = engine.transcribe_samples(&audio, &mut triplet).unwrap();
                if !result.text.is_empty() {
                    final_count += 1;
                }
            }
        }
        if let Some(flush) = engine.flush_state(&mut state, &mut triplet)
            && flush.is_final
        {
            final_count += 1;
        }

        assert!(
            partial_count > 0,
            "VAD hybrid should emit Partial segments during active speech, got {partial_count}"
        );
        assert!(
            final_count > 0,
            "VAD hybrid should emit at least one Final segment, got {final_count}"
        );
    }

    #[tokio::test]
    async fn test_vad_hybrid_matches_offline() {
        // Skip if model is not downloaded.
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        let model_dir = home.as_ref().map(|p| p.join(".phostt/models"));
        if model_dir.is_none()
            || !model_dir
                .as_ref()
                .unwrap()
                .join("encoder.int8.onnx")
                .exists()
        {
            eprintln!("Skipping test_vad_hybrid_matches_offline: model not found");
            return;
        }
        let model_dir = model_dir.unwrap();
        let wav_path = model_dir.join("test_wavs").join("0.wav");
        if !wav_path.exists() {
            eprintln!("Skipping test_vad_hybrid_matches_offline: test WAV not found");
            return;
        }
        let samples = audio::decode_audio_file(wav_path.to_str().unwrap()).unwrap();

        let engine = Engine::load_with_pool_size_and_config_and_vad(
            model_dir.to_str().unwrap(),
            1,
            StreamingConfig::default(),
            true,
        )
        .unwrap();
        let mut triplet = engine.pool.checkout().await.unwrap();

        let offline = engine.transcribe_samples(&samples, &mut triplet).unwrap();
        let offline_text = offline.text;

        let mut state = engine.create_state(false).unwrap();
        let chunk_size = samples.len() / 4;
        let chunks = vec![
            &samples[..chunk_size],
            &samples[chunk_size..2 * chunk_size],
            &samples[2 * chunk_size..3 * chunk_size],
            &samples[3 * chunk_size..],
        ];

        let mut hybrid_text = String::new();
        for chunk in chunks {
            let _segs = engine
                .process_chunk(chunk, &mut state, &mut triplet)
                .unwrap();
            // Drain async ASR queue (completed VAD utterances).
            for audio in state.vad_pending_asr.drain(..) {
                let result = engine.transcribe_samples(&audio, &mut triplet).unwrap();
                if !result.text.is_empty() {
                    if !hybrid_text.is_empty() {
                        hybrid_text.push(' ');
                    }
                    hybrid_text.push_str(&result.text);
                }
            }
        }
        if let Some(flush) = engine.flush_state(&mut state, &mut triplet) {
            if !hybrid_text.is_empty() {
                hybrid_text.push(' ');
            }
            hybrid_text.push_str(&flush.text);
        }

        let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(
            normalize(&hybrid_text),
            normalize(&offline_text),
            "VAD hybrid final transcript should match offline transcript"
        );
    }

    #[tokio::test]
    async fn test_streaming_matches_offline() {
        // Skip if model is not downloaded.
        let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
        let model_dir = home.as_ref().map(|p| p.join(".phostt/models"));
        if model_dir.is_none()
            || !model_dir
                .as_ref()
                .unwrap()
                .join("encoder.int8.onnx")
                .exists()
        {
            eprintln!("Skipping test_streaming_matches_offline: model not found");
            return;
        }
        let model_dir = model_dir.unwrap();
        let engine = Engine::load(model_dir.to_str().unwrap()).unwrap();
        let wav_path = model_dir.join("test_wavs").join("0.wav");
        if !wav_path.exists() {
            eprintln!("Skipping test_streaming_matches_offline: test WAV not found");
            return;
        }
        let samples = audio::decode_audio_file(wav_path.to_str().unwrap()).unwrap();

        let mut triplet = engine.pool.checkout().await.unwrap();
        let offline = engine.transcribe_samples(&samples, &mut triplet).unwrap();
        let offline_text = offline.text;

        let mut state = engine.create_state(false).unwrap();
        let chunk_size = samples.len() / 3;
        let chunks = vec![
            &samples[..chunk_size],
            &samples[chunk_size..2 * chunk_size],
            &samples[2 * chunk_size..],
        ];

        let mut streaming_text = String::new();
        for chunk in chunks {
            let segs = engine
                .process_chunk(chunk, &mut state, &mut triplet)
                .unwrap();
            for seg in segs {
                if seg.is_final {
                    if !streaming_text.is_empty() {
                        streaming_text.push(' ');
                    }
                    streaming_text.push_str(&seg.text);
                }
            }
        }
        if let Some(flush) = engine.flush_state(&mut state, &mut triplet) {
            if !streaming_text.is_empty() {
                streaming_text.push(' ');
            }
            streaming_text.push_str(&flush.text);
        }

        let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(
            normalize(&streaming_text),
            normalize(&offline_text),
            "streaming transcript should match offline transcript"
        );
    }

    #[test]
    fn test_vad_pipeline_with_synthetic_audio() {
        // Model-free: manually drive the Silero VAD segmenter with synthetic
        // probability values to verify the segment-to-buffer queueing logic
        // that process_chunk_vad performs.
        let mut segmenter = silero::SpeechSegmenter::new(silero::SpeechOptions::default());

        // Simulate ~3 seconds of speech (Silero VAD processes 512-sample
        // windows at 16 kHz → ~32 ms per window → ~94 windows in 3 s).
        for _ in 0..100 {
            if segmenter.push_probability(0.95).is_some() {
                // segment completed mid-speech — unlikely with steady high prob
            }
        }

        // Simulate ~1 second of silence to force the segment to close.
        let mut segments: Vec<silero::SpeechSegment> = Vec::new();
        for _ in 0..35 {
            if let Some(seg) = segmenter.push_probability(0.02) {
                segments.push(seg);
            }
        }

        // Finish must emit any trailing open segment.
        if let Some(seg) = segmenter.finish() {
            segments.push(seg);
        }

        assert!(
            !segments.is_empty(),
            "VAD segmenter should emit at least one segment after speech+silence"
        );

        // Build a synthetic audio buffer whose length covers the segment.
        let sample_rate = 16000usize;
        let duration_sec = 5usize;
        let total_samples = sample_rate * duration_sec;
        let mut samples = Vec::with_capacity(total_samples);
        for i in 0..total_samples {
            let t = i as f32 / sample_rate as f32;
            samples.push((2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5);
        }

        // Simulate the queueing logic that process_chunk_vad performs.
        let vad_audio_buffer = samples.clone();
        let vad_sample_offset: u64 = 0;
        let mut vad_pending_asr: Vec<Vec<f32>> = Vec::new();

        for segment in &segments {
            let buf_start = segment.start_sample().saturating_sub(vad_sample_offset) as usize;
            let buf_end = segment.end_sample().saturating_sub(vad_sample_offset) as usize;
            if buf_end <= vad_audio_buffer.len() && buf_start < buf_end {
                let speech_samples = &vad_audio_buffer[buf_start..buf_end];
                vad_pending_asr.push(speech_samples.to_vec());
            }
        }

        assert!(
            !vad_pending_asr.is_empty(),
            "vad_pending_asr should be populated with queued utterances"
        );
        for utterance in &vad_pending_asr {
            assert!(!utterance.is_empty(), "queued utterance must not be empty");
        }
    }
}
