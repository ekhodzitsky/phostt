#!/usr/bin/env python3
"""Download the Google FLEURS Vietnamese test split and export to WAV + manifest.

Usage:
    python scripts/prepare_fleurs_benchmark.py

Output:
    ~/.phostt/benchmark/fleurs_vi/manifest.tsv
    ~/.phostt/benchmark/fleurs_vi/audio/*.wav

The manifest is a TSV with columns: id, wav_path, reference_text
"""

import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    import soundfile as sf
    import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install datasets soundfile tqdm")
    sys.exit(1)


def main():
    home = Path.home()
    out_dir = home / ".phostt" / "benchmark" / "fleurs_vi"
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FLEURS Vietnamese test split...")
    ds = load_dataset("google/fleurs", "vi_vn", split="test", trust_remote_code=True)

    manifest_path = out_dir / "manifest.tsv"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("id\twav_path\treference_text\n")
        for i, sample in enumerate(tqdm.tqdm(ds, desc="Exporting WAVs")):
            wav_name = f"{i:05d}.wav"
            wav_path = audio_dir / wav_name

            audio_array = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            sf.write(wav_path, audio_array, sr)

            ref = sample["transcription"].strip().replace("\t", " ")
            f.write(f"{i}\t{wav_path}\t{ref}\n")

    print(f"Done. {len(ds)} samples written to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
