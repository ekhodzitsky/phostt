# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-05-05

### Added

- **Polyvoice diarization integration**. Replaced the internal `SpeakerEncoder` +
  `SpeakerCluster` stack with `polyvoice::OnlineDiarizer` and
  `polyvoice::OnnxEmbeddingExtractor` (lock-free pool via `crossbeamqueue`).
  The `diarization` feature now pulls `polyvoice` instead of in-tree ONNX
  inference code.
- **ARM Linux build check** in CI (`aarch64-unknown-linux-gnu`).
- **WER quality gate** (`cargo test --test wer -- --ignored`) runs in E2E CI
  against the bundled Vietnamese test WAVs.
- **Load test** (`cargo test --test load_test -- --ignored --test-threads=1`)
  runs on every push to `main`.
- **Peak RSS tracking** in benchmark suite (`scripts/benchmark.sh` and
  `.github/workflows/benchmark.yml`).

### Fixed

- **Security / resource safety**:
  - `rustls-webpki` RUSTSEC-2026-0104: updated to 0.103.13.
  - `flush_state`: replaced silent `.ok()?` with explicit `match` +
    `tracing::error!`.
  - `run_inference`: added `encoder_outputs.len() < 2` check to prevent OOB
    indexing on malformed encoder output.
  - `extract_encoder_frame`: `debug_assert!` → `assert!` so the bounds check
    is active in release builds.
  - `resample`: capped `max_chunk_size` to `TARGET_SAMPLE_RATE * 5` to avoid
    huge rubato allocations on adversarial input.
  - `decode_audio_inner`: capped `Vec::with_capacity` to `max_samples` to
    prevent malicious `n_frames_hint` OOM.
  - `session_deadline_instant`: `u32::MAX` → `u64::MAX / 4` to avoid
    `Instant` overflow on macOS.
- **FFI odd-byte PCM16 handling**: `PhosttStream` now carries `pending_byte`
  to preserve odd-length PCM16 byte streams across chunk calls, matching the
  WebSocket handler behaviour exactly.

### Changed

- `scripts/benchmark.sh` refactored to measure latency, RTF, and peak RSS in
  a single pass per backend using `/usr/bin/time -l` (macOS) / `-v` (Linux).
- README completely rewritten with badges, platform support matrix,
  architecture diagram, mobile/FFI section, and quality/WER benchmarks.

## [0.3.0] - 2026-05-05

### Added

- **Silero VAD simulated streaming** (`--vad`). When enabled, speech is
  segmented by voice activity instead of the fixed overlap-buffer. Each
  detected utterance is transcribed offline, eliminating boundary artefacts
  entirely. While speech is active, partial (interim) results are still
  emitted via the overlap-buffer so clients see live transcription progress.
  Uses the bundled `silero_vad.onnx` (~1 MB) via the `silero` crate.
- Configurable streaming overlap-buffer parameters:
  - `--streaming-window-ms` (default 4000) and `--streaming-overlap-ms` (default 1000)
    control the latency / accuracy trade-off of the overlap-buffer streaming.
  - `--streaming-fuzzy-threshold` (default 1.0) enables fuzzy word matching on
    window boundaries using normalized Levenshtein distance. Reduces duplicate
    words when the offline encoder produces slightly different transcripts in
    the overlap zone.
- `StreamingConfig` struct with validation (window / overlap must be multiples
  of 4 due to encoder subsampling, overlap must be smaller than window).
- In-tree Levenshtein distance implementation (`levenshtein`, `word_similarity`,
  `words_match`) with unit tests.

### Changed

- Upgraded default model to `sherpa-onnx-zipformer-vi-int8-2025-04-20` (~77 MB),
  trained on **70,000 hours** of Vietnamese speech (vs previous ~6,000h).
  Expected WER improvement: ~20–30% relative on VLSP/GigaSpeech2 benchmarks.
  The download/extract pipeline now normalizes epoch-specific filenames
  (e.g. `encoder-epoch-12-avg-8.int8.onnx` → `encoder.int8.onnx`) automatically.

## [0.2.7] - 2026-04-21

### Added

- **Spec 005** — Graceful shutdown unit tests. Extracted pure helpers
  `ws_shutdown_response()` and `compute_session_deadline()` from WebSocket
  handlers for testability. Added 5 fast unit tests covering drain clamping,
  session deadline overflow safety, shutdown response shape, and eviction
  loop cancellation. Added 1 lightweight async test (`< 100 ms`) that
  exercises the pre-checkout cancel path with a real TCP socket and
  `tokio_tungstenite` client but no ONNX model load. Total test count: 121.

## [0.2.6] - 2026-04-21

### Fixed

- **Spec 004** — Eliminated per-step `Vec<f32>` allocations in the greedy decode
  loop. `run_decoder` and `run_joiner_single` now write into reusable buffers
  (`&mut Vec<f32>`) instead of returning freshly allocated `Vec`s. Three buffers
  (`joiner_buf`, `decoder_buf_a`, `decoder_buf_b`) are created once per
  `greedy_decode` call; `mem::swap` rotates the decoder double-buffer on each
  non-blank token. For a typical 10-second utterance this removes 250+
  allocations from the hottest inference path.

## [0.2.5] - 2026-04-21

### Added

- **Spec 003** — Android streaming FFI. Four new C-ABI exports:
  - `phostt_stream_new(engine)` — checkout triplet + create `StreamingState`
  - `phostt_stream_process_chunk(engine, stream, pcm16_bytes, len, sample_rate)` —
    convert PCM16 → f32, resample if needed, run `process_chunk`, return JSON array
  - `phostt_stream_flush(engine, stream)` — return final segment(s) as JSON
  - `phostt_stream_free(stream)` — return triplet to pool
- Kotlin bridge (`PhosttBridge.kt`) updated with `streamNew`,
  `streamProcessChunk`, `streamFlush`, `streamFree`.
- `tests/ffi_streaming.rs` — `#[ignore]` integration tests for happy path and
  48 kHz resample path.
- `#[derive(Serialize)]` on `TranscriptSegment` (enables JSON serialization in
  FFI and future protocol boundaries).

### Changed

- `PhosttEngine` gained `pub fn new(engine: Engine)` constructor.

## [0.2.4] - 2026-04-21

### Added

- **Spec 002** — 13 fast REST handler unit tests (error paths + SSE mapping).
  No ONNX model required; runs in ~10 ms. Covers `empty_body`,
  `payload_too_large`, `pool_timeout`, `pool_closed` for both `/v1/transcribe`
  and `/v1/transcribe/stream`, plus metrics endpoint and SSE JSON mapping.

### Changed

- `tokio` feature set extended with `test-util` (enables
  `tokio::time::pause`/`advance` in unit tests).

## [0.2.3] - 2026-04-21

### Fixed

- **Spec 001** — Pool slot leak: `PoolItemGuard<T>` RAII guard automatically
  returns triplets to the pool during panic unwind. Previously a panic in
  `spawn_blocking` permanently lost the slot.
- **Spec 001** — Metrics poison: `RwLock::write().unwrap_or_else(|e| e.into_inner())`
  recovers from poisoned locks instead of panicking. A single handler crash no
  longer breaks all subsequent metric writes.

### Added

- `PoolItemGuard<T>` — couples an owned pool item with its reservation for
  automatic Drop recovery.
- `specs/001-pool-metrics-reliability.md` — spec-driven design document.

## [0.2.2] - 2026-04-21

### Fixed

- **High** — Zero-copy SSE streaming: `decode_audio_streaming` feeds decoded
  chunks directly into inference without a full `Vec<f32>` buffer. Clients
  receive the first partial immediately after the first decoded packet.

### Changed

- `transcribe_stream`: single `spawn_blocking` pass (decode + inference merged),
  eliminating the double-buffer memory peak.
- `resample` test tolerance widened (rubato sinc latency edge case).

### Added

- `test_decode_streaming_matches_batch`: verifies bit-exact equivalence between
  streaming and batch decode paths.

## [0.2.0] - 2026-04-21

### Added

- FFI layer for Android (`src/ffi.rs`): C-ABI exposing `phostt_engine_new`,
  `phostt_transcribe_file`, `phostt_string_free`, `phostt_engine_free`.
- Kotlin JNI bridge skeleton (`ffi/android/PhosttBridge.kt`).
- Android build docs (`ANDROID.md`) with NDK setup, model bundling strategies,
  and Kotlin integration guide.
- `cargo-ndk` linker config (`.cargo/config.toml`) for arm64, armv7, x86_64,
  i686 Android targets.
- Benchmark harness (`benches/latency.rs`) using Criterion for offline
  transcription latency measurement.
- WER regression test (`tests/wer.rs`) against Vietnamese test WAVs with
  reference transcripts.
- `Pool::checkout_blocking` synchronous pool checkout for FFI consumers.
- `Engine::transcribe_samples` made public (exposed to FFI).
- `ffi` Cargo feature pulling in `ort/nnapi` for Android NPU/DSP acceleration.
- Streaming wrapper fully wired: `OnlineFeature` with sliding 4 s window +
  1 s overlap, overlap-merge word deduplication, `input_finished()` flush on
  endpoint / stop / close. Unit test `test_streaming_matches_offline` confirms
  3-chunk streaming produces identical text to offline inference.
- E2E test suite fully switched to Vietnamese audio fixtures (`test_wavs/*.wav`).

### Changed

- `README.md`: updated status to "release candidate", added real Vietnamese
  smoke-test output (`RỒI CŨNG HỖ TRỢ...`), clarified latency note.
- `CLAUDE.md`: updated project status, added Streaming model section explaining
  the offline-encoder + overlap-buffer trade-off.

### Removed

- No-op `quantize` feature flag removed from `Cargo.toml` (Zipformer-vi ships
  pre-quantized; in-tree quantizer was dead weight).

## [0.2.1] - 2026-04-21

### Fixed

- **Critical** — Eliminated O(n²) latency degradation in streaming by replacing
  `String`/`Vec` clones with `Arc<String>`/`Arc<Vec<WordInfo>>` in
  `StreamingState` and `TranscriptSegment`.
- **Critical** — Replaced `.expect()` panics in `Engine::load_inner` and
  `Engine::create_state` with proper `Result` returns. Server no longer crashes
  on thread panic during model load or FBANK init failure.

### Security

- `extract_bundle`: reject symlink and hard-link entries in tar archives to
  prevent directory-traversal attacks.

### Changed

- `sha256_file`: use `BufReader` with 64KB chunks instead of loading the entire
  ~850MB model file into RAM.
- `stream_to_partial_then_finalize`: add 30s connect timeout and 10min total
  request timeout.

### Refactored

- `tokens_to_words`: removed duplicate empty-token guard.
- All `create_state` callers updated to handle `Result`.

## [0.1.0] - YYYY-MM-DD

### Added

- Initial scaffolding forked from [gigastt](https://github.com/ekhodzitsky/gigastt)
  v0.9.4 — HTTP/WebSocket/SSE server, REST API, rate limiting, Prometheus
  metrics, graceful shutdown, multi-arch Docker images, Homebrew formula.
- Project rebranded to **phostt** for on-device Vietnamese speech recognition.
- Crate metadata, environment variables (`PHOSTT_*`), config paths
  (`~/.phostt/`), CLI binary name (`phostt`) all renamed from `gigastt`.
- Replace GigaAM v3 e2e_rnnt model wiring with Zipformer-vi RNN-T
  (`hynt/Zipformer-30M-RNNT-6000h` via sherpa-onnx GitHub releases).
- Mel feature extraction: 64 → 80 mel bins, N_FFT 320 → 400 (Zipformer
  defaults).
- Tokenizer: replace GigaAM BPE (`vocab.txt`, 1025 tokens) with
  SentencePiece BPE (`bpe.model`).
- Decoder: drop LSTM (h, c) state in favor of stateless prev-token
  embedding used by Zipformer-Transducer.
- Streaming wrapper: overlap-buffer decode for offline Zipformer-vi
  encoder.
- New end-to-end + benchmark fixtures using Vietnamese audio.
