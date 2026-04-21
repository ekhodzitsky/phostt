# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial scaffolding forked from [gigastt](https://github.com/ekhodzitsky/gigastt)
  v0.9.4 — HTTP/WebSocket/SSE server, REST API, rate limiting, Prometheus
  metrics, graceful shutdown, multi-arch Docker images, Homebrew formula.
- Project rebranded to **phostt** for on-device Vietnamese speech recognition.
- Crate metadata, environment variables (`PHOSTT_*`), config paths
  (`~/.phostt/`), CLI binary name (`phostt`) all renamed from `gigastt`.

### Pending

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

## [0.1.0] - TBD

Initial pre-alpha release once the inference path lands.
