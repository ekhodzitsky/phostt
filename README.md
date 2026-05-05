<p align="center">
  <h1 align="center">phostt</h1>
  <p align="center"><strong>On-device Vietnamese speech recognition</strong></p>
  <p align="center">Local STT server powered by Zipformer-vi RNN-T — no cloud, no API keys, full privacy</p>
  <p align="center">
    <a href="https://crates.io/crates/phostt"><img src="https://img.shields.io/crates/v/phostt.svg" alt="Crates.io"></a>
    <a href="https://github.com/ekhodzitsky/phostt/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/ekhodzitsky/phostt/ci.yml?branch=main&label=ci" alt="CI"></a>
    <a href="https://github.com/ekhodzitsky/phostt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
    <a href="https://github.com/ekhodzitsky/phostt/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-Keep%20a%20Changelog-orange" alt="Changelog"></a>
  </p>
</p>

---

**phostt** turns any machine into a Vietnamese speech recognition server that
runs entirely on-device. Zipformer-vi RNN-T weights ship pre-quantized from
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx); the binary is one
command, the model is ~75 MB, everything runs locally.

```sh
cargo install phostt && phostt serve
# WebSocket: ws://127.0.0.1:9876/v1/ws
# REST API:  http://127.0.0.1:9876/v1/transcribe
```

Or build from source:

```sh
git clone https://github.com/ekhodzitsky/phostt
cd phostt
cargo run --release -- serve
```

## Why phostt?

| | phostt | PhoWhisper-large | Cloud APIs |
|---|:---:|:---:|:---:|
| **Architecture** | Zipformer + RNN-T | Whisper enc-dec | varies |
| **Model size (INT8)** | **~75 MB** | ~1.5 GB | server-side |
| **WER (GigaSpeech2-vi)** | ~7.7% | n/a | varies |
| **Latency (3.7 s audio)** | **~61 ms** | ~300 ms | network + queue |
| **Throughput** | **61× RTF** | ~3× RTF | varies |
| **Privacy** | 100% local | 100% local | data leaves device |
| **Cost** | free forever | free | $0.006/min+ |
| **Setup** | `cargo install` | Python + deps | API key + billing |
| **Streaming** | real-time WebSocket | batch only | varies |

The Zipformer-vi-30M weights ship via [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) (Apache 2.0,
trained on ~70,000 hours of Vietnamese speech, ICASSP-published WER on the
VLSP and GigaSpeech2 benchmarks).

## Features

- **Real-time streaming** — partial transcription via WebSocket as you speak
- **REST API + SSE** — file transcription with instant or streaming response
- **Hardware acceleration** — CoreML + Neural Engine (macOS), CUDA 12+ (Linux), CPU everywhere
- **Pre-quantized INT8** — encoder ships at ~75 MB INT8 from upstream
- **Multi-format audio** — WAV, M4A/AAC, MP3, OGG/Vorbis, FLAC
- **Auto-download** — model fetched from sherpa-onnx GitHub releases on first run
- **Speaker diarization** — optional `diarization` feature for multi-speaker sessions
- **Docker ready** — CPU and CUDA images with multi-stage builds
- **Android FFI** — C-ABI + Kotlin bridge for mobile integration
- **Hardened** — connection limits, frame caps, idle timeouts, sanitized errors, rate limiting

## Platform Support

| Platform | Target | Backend | Notes |
|---|---|---|---|
| macOS (Apple Silicon) | `aarch64-apple-darwin` | CoreML / CPU | Neural Engine + CPU fallback |
| macOS (Intel) | `x86_64-apple-darwin` | CPU | |
| Linux (x86_64) | `x86_64-unknown-linux-gnu` | CUDA 12+ / CPU | CUDA via `--features cuda` |
| Linux (ARM64) | `aarch64-unknown-linux-gnu` | CPU | Buildable, not CI-tested yet |
| Android | `aarch64-linux-android`, `armv7-linux-androideabi` | NNAPI / CPU | Via `cargo-ndk` + `ffi` feature |
| Windows | `x86_64-pc-windows-msvc` | CPU | Community-maintained |

> **iOS** is theoretically supported via CoreML (`--features coreml,ffi`), but not yet verified in CI.

## Quick Start

### Install

```sh
cargo install phostt
```

The first run downloads the ~75 MB Zipformer-vi ONNX bundle automatically into `~/.phostt/models/`.

### Serve

```sh
phostt serve
# Listening on ws://127.0.0.1:9876/v1/ws
# REST API at http://127.0.0.1:9876/v1/transcribe
```

### Smoke test

```sh
phostt transcribe ~/.phostt/models/test_wavs/0.wav
```

Expected output (from the bundled Vietnamese test fixture):

```
RỒI CŨNG HỖ TRỢ CHO LÂU LÂU CŨNG CHO GẠO CHO NÀY KIA
```

### Docker

```sh
# CPU (any platform)
docker build -t phostt .
docker run -p 9876:9876 phostt

# CUDA (Linux + NVIDIA Container Toolkit)
docker build -f Dockerfile.cuda -t phostt-cuda .
docker run --gpus all -p 9876:9876 phostt-cuda
```

## Benchmarks

Measured on Apple Silicon M2 Pro, release build, 3.74 s Vietnamese test audio:

| Backend | Mean Latency | Median | P95 | RTF | Peak RSS |
|---|---|---|---|---|---|
| **CPU** | **60 ms** | 60 ms | 61 ms | **62×** | **1.4 GB** |
| CoreML (Neural Engine) | 93 ms | 90 ms | 124 ms | 40× | 1.2 GB |

*RTF = real-time factor (audio seconds processed per wall-clock second).*
*For this 30M-param INT8 model, CPU is faster than CoreML on Apple Silicon.*

Auto-updated benchmark history: [`BENCHMARKS.md`](BENCHMARKS.md).

## Quality / WER

| Dataset | Samples | WER | Notes |
|---|---|---|---|
| **GigaSpeech2-vi (clean)** | — | **~7.7%** | Published upstream benchmark on clean Vietnamese speech |
| **FLEURS Vietnamese** | 857 | **103.08%** | Foreign names, numbers, and English terms transcribed phonetically into Vietnamese |

> The FLEURS baseline is **intentionally high** — the dataset contains many proper names, digits, and English loanwords that the model transcribes phonetically into Vietnamese. It is used for **regression tracking only** (threshold: 1.05), not as an absolute quality metric. For production-quality assessment, refer to the GigaSpeech2-vi benchmark above.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐
│   Client    │────▶│  axum HTTP  │────▶│   SessionPool       │
│  (WS/REST)  │     │   router    │     │  (async-channel)    │
└─────────────┘     └─────────────┘     └─────────────────────┘
                                                │
                    ┌───────────────────────────┘
                    ▼
           ┌────────────────┐
           │ SessionTriplet │──▶ Zipformer Encoder (ONNX)
           │ (enc/dec/join) │──▶ RNN-T Decoder (greedy)
           └────────────────┘──▶ Joiner
                    │
                    ▼
           ┌────────────────┐
           │ StreamingState │──▶ overlap-buffer / VAD
           │ (per-connection)│    → partial + final segments
           └────────────────┘
```

## Mobile / FFI

phostt exposes a C-ABI for Android integration:

```c
PhosttEngine* engine = phostt_engine_new("/path/to/models");
PhosttStream* stream = phostt_stream_new(engine);
char* json = phostt_stream_process_chunk(engine, stream, pcm16, len, 16000);
// ... free with phostt_string_free(json) ...
```

See [`ANDROID.md`](ANDROID.md) for NDK setup, Kotlin bridge (`ffi/android/PhosttBridge.kt`), and model bundling strategies.

## Roadmap

- [x] v0.3.0 — Silero VAD streaming, configurable overlap-buffer, auto-benchmark CI
- [x] v0.4.0 — Polyvoice diarization, security/resource hardening, benchmark RSS
- [ ] iOS build verification (CoreML + `ffi` feature)
- [ ] Quantized embedding extractor for faster diarization
- [ ] Offline batch re-clustering pass for improved speaker accuracy

See [`TODO.md`](TODO.md) for the full tracker.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Quick start for developers:

```sh
cargo build --release --features coreml   # or cuda
cargo test                                # 146 fast unit tests, no model needed
cargo clippy --all-targets -- -D warnings
cargo deny check
```

## Acknowledgements

phostt is a Vietnamese fork of [`gigastt`](https://github.com/ekhodzitsky/gigastt),
which provides the production-grade server scaffolding (HTTP/WS/SSE,
rate-limit, metrics, graceful shutdown). The Vietnamese inference uses the
Zipformer-Transducer weights packaged by the
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project.

## License

MIT — see [LICENSE](LICENSE).
