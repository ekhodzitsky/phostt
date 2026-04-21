# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**phostt** — local speech-to-text server for Vietnamese, powered by Zipformer-vi RNN-T. On-device inference via ONNX Runtime. No cloud, no API keys, full privacy.

- **Repository**: https://github.com/ekhodzitsky/phostt
- **License**: MIT
- **Crates.io**: `phostt` name verified available (not yet published)
- **Status**: 0.1.0 release candidate. Forked from [`gigastt`](https://github.com/ekhodzitsky/gigastt) v0.9.4 (Russian STT). The HTTP/WS/SSE/metrics/shutdown stack is production-grade and unchanged. The inference path (model fetch, 80-bin mel features, SentencePiece BPE tokenizer, stateless RNN-T decode, overlap-buffer streaming) is fully wired and tested against Vietnamese audio fixtures.

## Build & Test

```sh
cargo build                          # CPU-only debug build (default, any platform)
cargo build --features coreml        # macOS ARM64 (CoreML / Neural Engine)
cargo build --features cuda          # Linux x86_64 (CUDA 12+)
cargo build --release                # Release build (LTO, stripped)
cargo test                           # Unit tests (no model required)
cargo clippy                         # Lint (no expected warnings)
```

`--features coreml` is confirmed working on macOS ARM64 (Apple Silicon).

## Model

**Zipformer-30M-RNNT-6000h** packaged by `sherpa-onnx` (Apache 2.0):

- Source bundle: `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2`
  (~26 MB) from [k2-fsa/sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)
- Files inside: `encoder.int8.onnx` (26 MB), `decoder.onnx` (4.9 MB),
  `joiner.int8.onnx` (1.0 MB), `bpe.model` (262 KB), `tokens.txt`
- Sample rate: 16 kHz · Mel bins: 80 · Vocab: SentencePiece BPE
- Decode: RNN-T greedy (stateless decoder — no LSTM h/c)
- Training: ~6000 hours of Vietnamese speech (`hynt/Zipformer-30M-RNNT-6000h`)
- WER: ~12.3 on VLSP2020-T1, ~7.97 on VLSP2025-Pub, ~7.56 on GigaSpeech2

The bundle is downloaded on first run into `~/.phostt/models/`. SHA-256
verification + atomic rename, same hardening as the upstream gigastt fetcher.

## Architecture

```
src/
  lib.rs                  # Public module exports
  main.rs                 # CLI (clap): serve, download, transcribe, inspect
  model/mod.rs            # Model bundle download (tar.bz2 → .onnx + bpe.model + tokens.txt)
  inference/
    mod.rs                # Engine: ONNX session pool, StreamingState, DecoderState
    features.rs           # Mel spectrogram (80 bins, FFT=400, hop=160) via kaldi_native_fbank::OnlineFeature
    tokenizer.rs          # SentencePiece BPE (bpe.model)
    decode.rs             # RNN-T greedy decode (stateless decoder)
    audio.rs              # Audio loading, resampling, channel mixing
  error.rs                # Typed error types (PhosttError)
  inspect.rs              # ONNX session I/O metadata printer
  server/
    mod.rs                # axum router: HTTP + WebSocket on single port
    http.rs               # REST handlers: /health, /v1/models, /v1/transcribe, /v1/transcribe/stream
    rate_limit.rs         # In-tree per-IP token-bucket rate limiter
    metrics.rs            # In-tree Prometheus text encoder
  protocol/mod.rs         # JSON message types for the WS protocol
```

### Streaming
- Zipformer-vi-30M is published as an offline transducer; phostt wraps it in
  an overlap-buffer streaming layer (StreamingState) to expose the same
  WebSocket protocol as the upstream gigastt server.
- Server accepts configurable sample rates (8/16/24/44.1/48 kHz) via the
  `Configure` message; default 48 kHz is resampled to 16 kHz with rubato.

## Streaming model

Zipformer-vi-30M is an offline transducer. phostt wraps it in a sliding
overlap-buffer (`StreamingState`) to expose real-time WebSocket streaming.
The implementation uses a 4-second window with 1-second overlap: audio is
chunked, each chunk is featurized and encoded independently, and partial
results are merged across boundaries.

`kaldi_native_fbank::OnlineFeature` in `inference/features.rs` drives the
feature extraction, providing the same povey-window + preemphasis + Slaney
mel filterbank pipeline that sherpa-onnx uses upstream. The `OnlineFeature`
wrapper handles streaming increments, while the overlap-and-merge logic in
`StreamingState` works around the offline encoder constraint.

### Graceful shutdown
- `CancellationToken` + `TaskTracker` cascades through every WS / SSE handler.
- On SIGTERM each session flushes, emits a final frame, and closes with
  `Close(1001 Going Away)`.
- `--shutdown-drain-secs` (default 10) bounds the wait after `axum::serve`
  returns. `--max-session-secs` (default 3600) caps any single WS session.

## Development guidelines

### TDD workflow
1. Write failing test first
2. Implement minimal code to pass
3. Refactor, verify tests still pass
4. `cargo test && cargo clippy` before every commit

### API versioning & backward compatibility
- WebSocket protocol version: `PROTOCOL_VERSION = "1.0"` (in `protocol/mod.rs`)
- `ServerMessage::Ready` includes `version` field sent on connection
- Canonical WS path: `/v1/ws`. `/ws` remains as a deprecated alias.
- New fields are additive only; never remove or rename existing fields.

### Code style
- Rust 2024 edition
- `anyhow` for error handling, `tracing` for logging
- No `unwrap()` in production paths (use `?`, `context()`, or `unwrap_or_else`)
- Shared inference constants live in `inference/mod.rs`

### Audio format support
- File transcription: WAV, M4A/AAC, MP3, OGG/Vorbis, FLAC (via symphonia)
- WebSocket: raw PCM16 binary frames at configurable sample rate; resampled
  to 16 kHz server-side via rubato

### Security
- **Loopback bind by default.** `127.0.0.1` only; `--bind-all` /
  `PHOSTT_ALLOW_BIND_ANY=1` required for non-loopback.
- **Origin allowlist.** Cross-origin denied by default; `--allow-origin`
  to extend, `--cors-allow-any` for wildcard.
- **Runtime limits**: `--idle-timeout-secs` (300), `--ws-frame-max-bytes`
  (512 KiB), `--body-limit-bytes` (50 MiB), `--pool-size` (4),
  `--max-session-secs` (3600), `--shutdown-drain-secs` (10).
- **Per-IP rate limiting** (opt-in): `--rate-limit-per-minute N` +
  `--rate-limit-burst`.
- **SHA-256 verification + atomic rename** on every model file.
- **Internal errors sanitized** — no path or model leakage to clients.
- **Prometheus `/metrics`** (opt-in via `--metrics`).

