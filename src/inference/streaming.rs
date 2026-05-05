//! Streaming inference state and overlap-buffer/VAD pipeline.
//!
//! Holds per-connection [`StreamingState`], [`DecoderState`], and the
//! streaming half of [`Engine`](super::engine::Engine) (`process_chunk`,
//! `flush_state`, etc.).

use serde::Serialize;
use std::sync::Arc;

use crate::error::PhosttError;

use super::engine::Engine;
use super::features;
#[cfg(feature = "diarization")]
use super::diarization;
use super::{CONTEXT_SIZE, N_MELS, SessionTriplet, TARGET_SAMPLE_RATE};
use kaldi_native_fbank::fbank::FbankComputer;
use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};

pub fn now_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

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
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
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

impl Engine {
    pub fn create_state(&self, diarization_enabled: bool) -> Result<StreamingState, PhosttError> {
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
            .map_err(|e| PhosttError::Inference(format!("FBANK init failed: {e}")))?;
        let online = OnlineFeature::new(FeatureComputer::Fbank(computer));

        let (vad_session, vad_stream_state, vad_segmenter) = if self.vad_enabled {
            let session = silero::Session::bundled()
                .map_err(|e| PhosttError::ModelLoad(format!("silero VAD load failed: {e}")))?;
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
    /// Returns [`PhosttError::Inference`] if the ONNX runtime fails.
    pub fn process_chunk(
        &self,
        samples: &[f32],
        state: &mut StreamingState,
        triplet: &mut SessionTriplet,
    ) -> Result<Vec<TranscriptSegment>, PhosttError> {
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
    ) -> Result<Vec<TranscriptSegment>, PhosttError> {
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
                .map_err(|e| PhosttError::Inference(format!("{e:#}")))?;

            let delta = super::delta_words(
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
    ) -> Result<Vec<TranscriptSegment>, PhosttError> {
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
                .map_err(|e| PhosttError::Inference(format!("VAD inference failed: {e}")))?;

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
            let delta = super::delta_words(
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
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TranscriptSegment {
    /// Recognized text for this segment.
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub text: Arc<String>,
    /// Individual words with timing and confidence metadata.
    #[cfg_attr(feature = "openapi", schema(value_type = Vec<WordInfo>))]
    pub words: Arc<Vec<WordInfo>>,
    /// Whether this segment is final (utterance complete) or partial (interim).
    pub is_final: bool,
    /// Unix timestamp (seconds since epoch) when this segment was produced.
    pub timestamp: f64,
}
