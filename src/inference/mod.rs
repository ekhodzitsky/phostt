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

mod pool;
pub use pool::{
    OwnedReservation, Pool, PoolError, PoolGuard, PoolItemGuard, SessionPool, SessionTriplet,
};

mod engine;
pub use engine::{Engine, delta_words, levenshtein, word_similarity, words_match};

mod streaming;
#[cfg(feature = "diarization")]
pub use streaming::DiarizationStreamState;
pub use streaming::{
    DecoderState, StreamingConfig, StreamingState, TranscribeResult, TranscriptSegment, WordInfo,
    now_timestamp,
};

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
        let delta = delta_words(&new, &prev, 1.0);
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
        let delta_exact = delta_words(&new, &prev, 1.0);
        // exact: no match for "helio" vs "hello", so best = 0
        assert_eq!(delta_exact.len(), 2);

        let delta_fuzzy = delta_words(&new, &prev, 0.7);
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
