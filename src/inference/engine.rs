//! Core ONNX inference engine.
//!
//! Loads encoder, decoder, and joiner ONNX models and runs the RNN-T
//! offline and streaming decode loop.

use anyhow::Context;
#[cfg(any(feature = "coreml", feature = "cuda"))]
use ort::ep;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use crate::error::PhosttError;

use super::audio;
use super::decode;
#[cfg(feature = "diarization")]
use super::diarization;
use super::{
    DecoderState, N_MELS, SessionPool, SessionTriplet, StreamingConfig, TARGET_SAMPLE_RATE,
    TranscribeResult, WordInfo,
};
/// Seconds per encoder frame (HOP_LENGTH * 4 / TARGET_SAMPLE_RATE = 0.04s).
const SECONDS_PER_FRAME: f64 =
    (super::HOP_LENGTH as f64) * 4.0 / (super::TARGET_SAMPLE_RATE as f64);

/// Default number of session triplets in the pool.
const DEFAULT_POOL_SIZE: usize = 4;

/// Check whether two words match, respecting the fuzzy threshold.
pub fn words_match(a: &str, b: &str, threshold: f32) -> bool {
    if threshold >= 1.0 {
        return a == b;
    }
    word_similarity(a, b) >= threshold
}

/// Normalized similarity in [0.0, 1.0]. 1.0 = identical.
pub fn word_similarity(a: &str, b: &str) -> f32 {
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

/// Compute Levenshtein edit distance between two strings.
pub fn levenshtein(a: &str, b: &str) -> usize {
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

fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

use super::features::MelSpectrogram;
use super::tokenizer::Tokenizer;

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
/// For streaming recognition, use [`create_state`](super::streaming::Engine::create_state) +
/// [`process_chunk`](super::streaming::Engine::process_chunk) +
/// [`flush_state`](super::streaming::Engine::flush_state).
pub struct Engine {
    /// Pool of ONNX session triplets for concurrent inference.
    pub pool: super::SessionPool,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) mel: MelSpectrogram,
    /// Overlap-buffer streaming configuration.
    pub(crate) streaming_config: StreamingConfig,
    /// When true, use Silero VAD for speech segmentation instead of the
    /// fixed-size overlap-buffer. Each detected utterance is transcribed
    /// offline, eliminating boundary artefacts.
    pub(crate) vad_enabled: bool,
    /// Speaker encoder for diarization (None if model file is absent).
    #[cfg(feature = "diarization")]
    pub(crate) speaker_encoder: Option<super::diarization::SpeakerEncoder>,
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
            pool: super::Pool::new(vec![]),
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
    /// Returns [`PhosttError::ModelLoad`] if model files are missing or ONNX session creation fails.
    pub fn load(model_dir: &str) -> Result<Self, PhosttError> {
        Self::load_with_pool_size(model_dir, DEFAULT_POOL_SIZE)
    }

    /// Load ONNX models with a custom pool size (default streaming config).
    pub fn load_with_pool_size(model_dir: &str, pool_size: usize) -> Result<Self, PhosttError> {
        Self::load_with_pool_size_and_config(model_dir, pool_size, StreamingConfig::default())
    }

    /// Load ONNX models with a custom pool size and streaming configuration.
    pub fn load_with_pool_size_and_config(
        model_dir: &str,
        pool_size: usize,
        streaming_config: StreamingConfig,
    ) -> Result<Self, PhosttError> {
        Self::load_with_pool_size_and_config_and_vad(model_dir, pool_size, streaming_config, false)
    }

    /// Load ONNX models with custom pool size, streaming config, and VAD flag.
    pub fn load_with_pool_size_and_config_and_vad(
        model_dir: &str,
        pool_size: usize,
        streaming_config: StreamingConfig,
        vad_enabled: bool,
    ) -> Result<Self, PhosttError> {
        streaming_config
            .validate()
            .map_err(|e| PhosttError::ModelLoad(format!("invalid streaming config: {e}")))?;
        let dir = Path::new(model_dir);
        if !dir.join("encoder.int8.onnx").exists() {
            return Err(PhosttError::ModelLoad(format!(
                "encoder.int8.onnx not found in {model_dir}"
            )));
        }
        Self::load_inner(dir, model_dir, pool_size, streaming_config, vad_enabled)
            .map_err(|e| PhosttError::ModelLoad(format!("{e:#}")))
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
    pub fn transcribe_file(
        &self,
        path: &str,
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, PhosttError> {
        let float_samples = audio::decode_audio_file(path)
            .map_err(|e| PhosttError::InvalidAudio(format!("{e:#}")))?;
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
    ) -> Result<TranscribeResult, PhosttError> {
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
    ) -> Result<TranscribeResult, PhosttError> {
        let float_samples = audio::decode_audio_bytes_shared(data)
            .map_err(|e| PhosttError::InvalidAudio(format!("{e:#}")))?;
        self.transcribe_samples(&float_samples, triplet)
    }

    /// Run the full mel + encoder + RNN-T decode pipeline on an already-decoded
    /// 16 kHz f32 sample buffer. Shared tail of [`Engine::transcribe_file`] and
    /// [`Engine::transcribe_bytes_shared`].
    pub fn transcribe_samples(
        &self,
        float_samples: &[f32],
        triplet: &mut SessionTriplet,
    ) -> Result<TranscribeResult, PhosttError> {
        let duration_s = float_samples.len() as f64 / (TARGET_SAMPLE_RATE as f64);

        let (features, num_frames) = self.mel.compute(float_samples);
        tracing::info!("Extracted {} mel frames", num_frames);

        let mut decoder_state = DecoderState::new(self.tokenizer.blank_id());
        let (words, _endpoint, _enc_len) = self
            .run_inference(triplet, &features, num_frames, &mut decoder_state, 0)
            .map_err(|e| PhosttError::Inference(format!("{e:#}")))?;
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

    pub(crate) fn run_inference(
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
}

/// Return the words from `new` that are not already present in `prev`,
/// using a suffix/prefix overlap merge with optional fuzzy matching.
pub fn delta_words(new: &[WordInfo], prev: &[WordInfo], fuzzy_threshold: f32) -> Vec<WordInfo> {
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
