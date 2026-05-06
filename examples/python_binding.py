#!/usr/bin/env python3
"""Demo: transcribe Vietnamese audio with phostt Python bindings.

Prerequisites:
    pip install phostt

The Zipformer-vi ONNX bundle (~75 MB) is downloaded automatically on first
run of the Rust CLI (`phostt download`) or placed manually at ~/.phostt/models.
"""

from pathlib import Path
from phostt import Engine

MODEL_DIR = Path.home() / ".phostt" / "models"
TEST_WAV = MODEL_DIR / "test_wavs" / "0.wav"


def main():
    if not (MODEL_DIR / "encoder.int8.onnx").exists():
        print("Model not found. Run first:")
        print("    cargo install phostt && phostt download")
        return

    engine = Engine(str(MODEL_DIR))

    if TEST_WAV.exists():
        print(f"Transcribing {TEST_WAV} ...")
        text = engine.transcribe_file(str(TEST_WAV))
        print("Result:", text)
    else:
        print(f"Test WAV not found: {TEST_WAV}")

    # In-memory transcription from bytes
    # with open("audio.mp3", "rb") as f:
    #     data = f.read()
    # text = engine.transcribe_bytes(data)
    # print(text)


if __name__ == "__main__":
    main()
