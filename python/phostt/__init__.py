"""On-device Vietnamese speech recognition — Python bindings for phostt.

Quick start::

    from phostt import Engine

    engine = Engine("~/.phostt/models")
    text = engine.transcribe_file("audio.wav")
    print(text)

The engine loads Zipformer-vi RNN-T ONNX models and runs inference locally.
No cloud APIs, no API keys, full privacy.
"""

from phostt.phostt import Engine as _Engine

__all__ = ["Engine"]
__version__ = "0.4.2"


class Engine:
    """Python wrapper around the Rust inference engine.

    Thread-safe: multiple Python threads can call ``transcribe_file``
    concurrently (limited by the ONNX session pool size, default 4).
    """

    def __init__(self, model_dir: str):
        """Load models from *model_dir*.

        The directory must contain ``encoder.int8.onnx``, ``decoder.onnx``,
        ``joiner.int8.onnx``, ``bpe.model``, and ``tokens.txt``.
        Models can be downloaded with the Rust CLI ``phostt download``.
        """
        self._engine = _Engine(model_dir)

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file (WAV, MP3, M4A, OGG, FLAC).

        Returns the transcribed text.
        """
        return self._engine.transcribe_file(path)

    def transcribe_bytes(self, data: bytes) -> str:
        """Transcribe audio from raw bytes in memory.

        *data* must contain a complete audio file, not raw PCM samples.
        """
        return self._engine.transcribe_bytes(data)

    def __repr__(self) -> str:
        return f"<phostt.Engine model_dir=...>"
