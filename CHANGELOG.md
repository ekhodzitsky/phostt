# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
