<p align="center">
  <h1 align="center">phostt</h1>
  <p align="center"><strong>On-device Vietnamese speech recognition</strong></p>
  <p align="center">Local STT server powered by Zipformer-vi RNN-T — no cloud, no API keys, full privacy</p>
  <p align="center">
    <a href="https://github.com/ekhodzitsky/phostt/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
    <a href="https://github.com/ekhodzitsky/phostt/blob/main/CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-Keep%20a%20Changelog-orange" alt="Changelog"></a>
  </p>
</p>

---

> **Status: 0.1.0 — feature complete, release candidate.**
> The server scaffolding (HTTP, WebSocket, SSE, rate-limit, metrics, graceful
> shutdown) is forked from the production-grade [`gigastt`](https://github.com/ekhodzitsky/gigastt)
> stack. The Vietnamese inference path (Zipformer-vi model fetch, 80-bin mel
> features, SentencePiece BPE tokenizer, RNN-T greedy decode, overlap-buffer
> streaming) is fully wired and tested.

**phostt** turns any machine into a Vietnamese speech recognition server that
runs entirely on-device. Zipformer-vi RNN-T weights ship pre-quantized from
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx); the binary is one
command, the model is ~75 MB, everything runs locally.

```sh
# Once 0.1.0 ships:
cargo install phostt && phostt serve
# WebSocket: ws://127.0.0.1:9876/v1/ws
# REST API:  http://127.0.0.1:9876/v1/transcribe
```

## Why phostt?

| | phostt | PhoWhisper-large | Cloud APIs |
|---|:---:|:---:|:---:|
| **Architecture** | Zipformer + RNN-T | Whisper enc-dec | varies |
| **Model size (INT8)** | **~75 MB** | ~1.5 GB | server-side |
| **WER (GigaSpeech2-vi)** | ~7.7% | n/a | varies |
| **Privacy** | 100% local | 100% local | data leaves device |
| **Cost** | free forever | free | $0.006/min+ |
| **Setup** | `cargo install` | Python + deps | API key + billing |
| **Streaming** | real-time WebSocket | batch only | varies |

The Zipformer-vi-30M weights ship via [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) (Apache 2.0,
trained on ~6000 hours of Vietnamese speech, ICASSP-published WER on the
VLSP and GigaSpeech2 benchmarks).

## Features

- **Real-time streaming** — partial transcription via WebSocket as you speak ✅
- **REST API + SSE** — file transcription with instant or streaming response
- **Hardware acceleration** — CoreML + Neural Engine (macOS), CUDA 12+ (Linux), CPU everywhere
- **Pre-quantized INT8** — encoder ships at ~75 MB INT8 from upstream
- **Multi-format audio** — WAV, M4A/AAC, MP3, OGG/Vorbis, FLAC
- **Auto-download** — model fetched from sherpa-onnx GitHub releases on first run
- **Docker ready** — CPU and CUDA images with multi-stage builds
- **Hardened** — connection limits, frame caps, idle timeouts, sanitized errors

## Quick Start (preview)

```sh
# From source
git clone https://github.com/ekhodzitsky/phostt
cd phostt
cargo run --release -- serve
```

The Zipformer-vi ONNX bundle (~75 MB) downloads automatically on first run
into `~/.phostt/models/`.

### Smoke test

With the server running (or using the `transcribe` command directly):

```sh
phostt transcribe ~/.phostt/models/test_wavs/0.wav
```

Expected output (from the bundled Vietnamese test fixture):

```
RỒI CŨNG HỖ TRỢ CHO LÂU LÂU CŨNG CHO GẠO CHO NÀY KIA
```

> **Latency:** ~50 ms total on 3.7 s of audio (M1 Pro, debug build).
> Release builds with LTO + `strip = true` are ~3–5× faster.

### Docker

```sh
# CPU (any platform)
docker build -t phostt .
docker run -p 9876:9876 phostt

# CUDA (Linux + NVIDIA Container Toolkit)
docker build -f Dockerfile.cuda -t phostt-cuda .
docker run --gpus all -p 9876:9876 phostt-cuda
```

## Acknowledgements

phostt is a Vietnamese fork of [`gigastt`](https://github.com/ekhodzitsky/gigastt),
which provides the production-grade server scaffolding (HTTP/WS/SSE,
rate-limit, metrics, graceful shutdown). The Vietnamese inference uses the
Zipformer-Transducer weights packaged by the
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project.

## License

MIT — see [LICENSE](LICENSE).
