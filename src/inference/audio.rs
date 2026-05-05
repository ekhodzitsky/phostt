//! Audio decoding, resampling, and buffer management utilities.

use anyhow::{Context, Result};
use bytes::Bytes;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::{HOP_LENGTH, N_FFT, TARGET_SAMPLE_RATE};

#[allow(dead_code)]
const MAX_BUFFER_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * 5; // 5 seconds at 16kHz
const MAX_DURATION_S: f64 = 600.0; // 10 minutes

/// A [`MediaSource`] that borrows its data from a reference-counted [`Bytes`]
/// buffer instead of cloning into a `Vec<u8>`.
///
/// Axum delivers REST upload bodies as `axum::body::Bytes`, which re-exports
/// `bytes::Bytes`. Before this type the decode path called `body.to_vec()` and
/// then wrapped the clone in `std::io::Cursor`, doubling the transient
/// memory footprint for every upload (a 50 MiB body briefly held 100 MiB in
/// RAM, plus another symphonia-side clone). `Bytes::clone` is a refcount bump,
/// so the shared variant decodes the original axum buffer in place.
///
/// The type is deliberately small and crate-private: it only needs to satisfy
/// `Read + Seek + Send + Sync` so symphonia's `MediaSourceStream` can drive it.
pub(crate) struct BytesMediaSource {
    data: Bytes,
    pos: u64,
}

impl BytesMediaSource {
    pub(crate) fn new(data: Bytes) -> Self {
        Self { data, pos: 0 }
    }
}

impl std::io::Read for BytesMediaSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let len = self.data.len() as u64;
        if self.pos >= len {
            return Ok(0);
        }
        let start = self.pos as usize;
        let available = self.data.len() - start;
        let n = available.min(buf.len());
        buf[..n].copy_from_slice(&self.data[start..start + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for BytesMediaSource {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let len = self.data.len() as u64;
        // `std::io::Seek` semantics: seeking past the end is allowed; the next
        // read returns 0. Seeking to a negative offset is an error.
        let new_pos: i128 = match pos {
            std::io::SeekFrom::Start(n) => n as i128,
            std::io::SeekFrom::End(off) => len as i128 + off as i128,
            std::io::SeekFrom::Current(off) => self.pos as i128 + off as i128,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before start of buffer",
            ));
        }
        self.pos = new_pos as u64;
        Ok(self.pos)
    }
}

impl MediaSource for BytesMediaSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.data.len() as u64)
    }
}

/// Decode any supported audio file to mono f32 samples at 16kHz.
///
/// Supports WAV, MP3, M4A/AAC, OGG/Vorbis, and FLAC via symphonia.
/// Multi-channel audio is mixed to mono. Files longer than 10 minutes are rejected.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, decoded, or exceeds the duration limit.
pub fn decode_audio_file(path: &str) -> Result<Vec<f32>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open audio file: {path}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }

    let source_label = format!(
        "format={}",
        std::path::Path::new(path)
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
    );

    decode_audio_inner(mss, hint, &source_label)
}

/// Decode audio from raw bytes in memory (no temp file needed).
///
/// Backwards-compatible shim: clones `data` into a [`Bytes`] and delegates
/// to [`decode_audio_bytes_shared`]. New call sites should pass a
/// `bytes::Bytes` (or `axum::body::Bytes`) directly to avoid the copy.
///
/// # Errors
///
/// Returns an error if the bytes cannot be decoded or the audio exceeds the duration limit.
pub fn decode_audio_bytes(data: &[u8]) -> Result<Vec<f32>> {
    decode_audio_bytes_shared(Bytes::copy_from_slice(data))
}

/// Decode audio from a shared [`Bytes`] buffer in place — no `to_vec()` clone.
///
/// Same logic as [`decode_audio_file`] but reads from a reference-counted
/// in-memory buffer. Supports WAV, MP3, M4A/AAC, OGG/Vorbis, and FLAC via
/// symphonia. Multi-channel audio is mixed to mono. The 10-minute duration
/// cap is enforced **incrementally** on each decoded packet: a malicious or
/// malformed upload is aborted before its decoded samples blow up RAM.
///
/// # Errors
///
/// Returns an error if the bytes cannot be decoded or the audio exceeds the
/// duration limit.
pub fn decode_audio_bytes_shared(data: Bytes) -> Result<Vec<f32>> {
    let source = BytesMediaSource::new(data);
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let hint = Hint::new();
    decode_audio_inner(mss, hint, "bytes")
}

/// Streaming audio decoder: decodes packet-by-packet and feeds each decoded
/// chunk (after mono mix and optional resample) into a callback.
///
/// The callback receives `&[f32]` slices at the target sample rate (16kHz).
/// It should copy the data if it needs to outlive the call. Returning `Err`
/// aborts the decode loop early. Returning `Ok(())` continues.
///
/// This is the zero-copy path for `/v1/transcribe/stream`: decoded samples
/// flow straight into the inference loop without a full `Vec<f32>` buffer.
pub fn decode_audio_streaming<F>(data: Bytes, mut on_chunk: F) -> Result<()>
where
    F: FnMut(&[f32]) -> Result<()>,
{
    let source = BytesMediaSource::new(data);
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

    let mut format = probed.format;
    let track = format.default_track().context("No audio track found")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("Unknown sample rate")?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    tracing::info!("Audio streaming: {sample_rate}Hz, {channels}ch");

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Unsupported audio codec")?;

    let max_samples: usize = (MAX_DURATION_S * sample_rate as f64) as usize;
    let mut total_decoded: usize = 0;

    // For non-16kHz sources we accumulate into a scratch buffer and resample
    // in chunks. For 16kHz we bypass resampling entirely.
    let mut resample_buf: Vec<f32> = Vec::new();
    let needs_resample = sample_rate != TARGET_SAMPLE_RATE;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(anyhow::anyhow!("Error reading packet: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).context("Decode error")?;
        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        // Mix to mono if multi-channel
        let mono_samples: Vec<f32> = if spec.channels.count() > 1 {
            let ch = spec.channels.count();
            (0..num_frames)
                .map(|frame| {
                    let sum: f32 = (0..ch).map(|c| samples[frame * ch + c]).sum();
                    sum / ch as f32
                })
                .collect()
        } else {
            samples.to_vec()
        };

        total_decoded += mono_samples.len();
        if total_decoded > max_samples {
            let observed_s = total_decoded as f64 / sample_rate as f64;
            anyhow::bail!(
                "Audio file too long ({observed_s:.0}s). Maximum supported: {MAX_DURATION_S:.0}s."
            );
        }

        if needs_resample {
            resample_buf.extend(mono_samples);
            // Resample in ~1-second chunks to avoid excessive buffering
            let chunk_samples = sample_rate as usize;
            while resample_buf.len() >= chunk_samples {
                let chunk = resample_buf.drain(..chunk_samples).collect::<Vec<_>>();
                let resampled = resample(&chunk, sample_rate, TARGET_SAMPLE_RATE)
                    .context("Resampling failed")?;
                on_chunk(&resampled)?;
            }
        } else {
            on_chunk(&mono_samples)?;
        }
    }

    // Flush any remaining resample buffer
    if needs_resample && !resample_buf.is_empty() {
        let resampled = resample(&resample_buf, sample_rate, TARGET_SAMPLE_RATE)
            .context("Resampling failed")?;
        on_chunk(&resampled)?;
    }

    tracing::info!("Streaming decode complete: {total_decoded} samples decoded");
    Ok(())
}

/// Shared decode logic: probe → format → decode → mono mix → duration check → resample.
fn decode_audio_inner(mss: MediaSourceStream, hint: Hint, source_label: &str) -> Result<Vec<f32>> {
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("Unsupported audio format")?;

    let mut format = probed.format;

    let track = format.default_track().context("No audio track found")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("Unknown sample rate")?;
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let n_frames_hint = track.codec_params.n_frames;

    tracing::info!("Audio ({source_label}): {sample_rate}Hz, {channels}ch");

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Unsupported audio codec")?;

    let max_samples: usize = (MAX_DURATION_S * sample_rate as f64) as usize;
    let mut all_samples: Vec<f32> = match n_frames_hint {
        Some(n) if n > 0 && n <= (MAX_DURATION_S as u64 + 1) * sample_rate as u64 => {
            // Cap capacity to avoid malicious n_frames_hint causing OOM.
            Vec::with_capacity((n as usize).min(max_samples))
        }
        _ => Vec::new(),
    };

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(anyhow::anyhow!("Error reading packet: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder.decode(&packet).context("Decode error")?;
        let spec = *decoded.spec();
        let num_frames = decoded.frames();

        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        if spec.channels.count() > 1 {
            let ch = spec.channels.count();
            for frame in 0..num_frames {
                let mut sum = 0.0_f32;
                for c in 0..ch {
                    sum += samples[frame * ch + c];
                }
                all_samples.push(sum / ch as f32);
            }
        } else {
            all_samples.extend_from_slice(samples);
        }

        if all_samples.len() > max_samples {
            let observed_s = all_samples.len() as f64 / sample_rate as f64;
            anyhow::bail!(
                "Audio file too long ({observed_s:.0}s). Maximum supported: {MAX_DURATION_S:.0}s.",
            );
        }
    }

    let duration_s = all_samples.len() as f64 / sample_rate as f64;
    tracing::info!(
        "Decoded {} samples at {}Hz ({:.1}s)",
        all_samples.len(),
        sample_rate,
        duration_s
    );

    if sample_rate != TARGET_SAMPLE_RATE {
        all_samples =
            resample(&all_samples, sample_rate, TARGET_SAMPLE_RATE).context("Resampling failed")?;
        tracing::info!("Resampled to 16kHz: {} samples", all_samples.len());
    }

    Ok(all_samples)
}

/// High-quality polyphase FIR resampler (rubato SincFixedIn).
///
/// Non-finite samples (NaN, infinity) are replaced with `0.0` before resampling.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return Ok(Vec::new());
    }
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let samples: Vec<f32> = samples
        .iter()
        .map(|&s| if s.is_finite() { s } else { 0.0 })
        .collect();

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_rate as f64 / from_rate as f64;
    // Process in 5-second chunks to avoid excessive internal allocation
    // when resampling very large buffers (e.g. 10-minute files).
    const CHUNK_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * 5;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        params,
        samples.len().min(CHUNK_SAMPLES),
        1, // mono
    )
    .map_err(|e| anyhow::anyhow!("Resampler init failed: {e}"))?;

    let waves_in = vec![samples];
    let mut waves_out = resampler
        .process(&waves_in, None)
        .map_err(|e| anyhow::anyhow!("Resampling failed: {e}"))?;
    Ok(waves_out.remove(0))
}

/// Prepare audio buffer for processing: merge new samples with leftover,
/// truncate if too long, split into usable samples and new leftover.
///
/// Returns `Some(usable_samples)` if enough data for at least one frame,
/// `None` if all data was buffered for the next call.
/// Updates `buffer` in-place with leftover samples.
#[allow(dead_code)]
pub(crate) fn prepare_audio_buffer(new_samples: &[f32], buffer: &mut Vec<f32>) -> Option<Vec<f32>> {
    let mut all_samples = std::mem::take(buffer);
    all_samples.extend_from_slice(new_samples);

    if all_samples.len() > MAX_BUFFER_SAMPLES {
        tracing::warn!("Audio buffer exceeded 5s limit, truncating");
        all_samples = all_samples[all_samples.len() - MAX_BUFFER_SAMPLES..].to_vec();
    }

    let hop_length = HOP_LENGTH;
    let n_fft = N_FFT;
    let usable = if all_samples.len() >= n_fft {
        let num_frames = (all_samples.len() - n_fft) / hop_length + 1;
        (num_frames - 1) * hop_length + n_fft
    } else {
        0
    };

    if usable == 0 {
        *buffer = all_samples;
        return None;
    }

    *buffer = all_samples[usable..].to_vec();
    Some(all_samples[..usable].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_silence_yields_finite_samples() {
        // 1 second of silence at 16kHz, mono, 16-bit PCM
        let num_samples = TARGET_SAMPLE_RATE as usize;
        let data_size: u32 = (num_samples * 2) as u32;
        let file_size: u32 = 44 + data_size;

        let mut wav = Vec::with_capacity(file_size as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(file_size - 8).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&TARGET_SAMPLE_RATE.to_le_bytes());
        wav.extend_from_slice(&(TARGET_SAMPLE_RATE * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for _ in 0..num_samples {
            wav.extend_from_slice(&0i16.to_le_bytes());
        }

        let samples = decode_audio_bytes(&wav).expect("Should decode silence WAV");
        assert_eq!(samples.len(), TARGET_SAMPLE_RATE as usize);
        for &s in &samples {
            assert!(s.is_finite(), "expected finite sample, got {s}");
            assert_eq!(s, 0.0, "silence should decode to 0.0");
        }
    }

    #[test]
    fn test_decode_duration_cap_blocks_long_files() {
        // 11 minutes of silence at 16kHz
        let num_samples: usize = (TARGET_SAMPLE_RATE as f64 * (MAX_DURATION_S + 60.0)) as usize;
        let data_size: u32 = (num_samples * 2) as u32;
        let file_size: u32 = 44 + data_size;

        let mut wav = Vec::with_capacity(file_size as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(file_size - 8).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&TARGET_SAMPLE_RATE.to_le_bytes());
        wav.extend_from_slice(&(TARGET_SAMPLE_RATE * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        wav.resize(file_size as usize, 0);

        let result = decode_audio_bytes(&wav);
        assert!(
            result.is_err(),
            "Should reject audio exceeding duration cap"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("too long"),
            "Error should mention duration: {err}"
        );
    }

    #[test]
    fn test_decode_shared_bytes_matches_decode_bytes() {
        let num_samples = TARGET_SAMPLE_RATE as usize;
        let data_size: u32 = (num_samples * 2) as u32;
        let file_size: u32 = 44 + data_size;

        let mut wav = Vec::with_capacity(file_size as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(file_size - 8).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&TARGET_SAMPLE_RATE.to_le_bytes());
        wav.extend_from_slice(&(TARGET_SAMPLE_RATE * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for i in 0..num_samples {
            let sample =
                ((440.0 * 2.0 * std::f64::consts::PI * i as f64 / TARGET_SAMPLE_RATE as f64).sin()
                    * 1000.0) as i16;
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        let via_bytes = decode_audio_bytes(&wav).unwrap();
        let via_shared = decode_audio_bytes_shared(Bytes::from(wav)).unwrap();
        assert_eq!(via_bytes.len(), via_shared.len());
        for (a, b) in via_bytes.iter().zip(via_shared.iter()) {
            assert!((a - b).abs() < 1e-5, "Samples differ: {a} vs {b}");
        }
    }

    #[test]
    fn test_resample_identity_same_rate() {
        let samples: Vec<f32> = (0..TARGET_SAMPLE_RATE as usize)
            .map(|i| i as f32 / TARGET_SAMPLE_RATE as f32)
            .collect();
        let result = resample(&samples, TARGET_SAMPLE_RATE, TARGET_SAMPLE_RATE).unwrap();
        assert_eq!(result.len(), samples.len());
        for (a, b) in samples.iter().zip(result.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Same-rate resample should be identity"
            );
        }
    }

    #[test]
    fn test_resample_48k_to_16k() {
        // 1 second of 440Hz sine at 48kHz
        let samples_48k: Vec<f32> = (0..48000)
            .map(|i| (440.0 * 2.0 * std::f64::consts::PI * i as f64 / 48000.0).sin() as f32)
            .collect();
        let result = resample(&samples_48k, 48000, TARGET_SAMPLE_RATE).unwrap();
        // Should be approximately 1 second at 16kHz
        assert!(
            (result.len() as i64 - TARGET_SAMPLE_RATE as i64).abs() < 100,
            "Expected ~{} samples, got {}",
            TARGET_SAMPLE_RATE,
            result.len()
        );
    }

    #[test]
    fn test_decode_streaming_matches_batch() {
        // 1 second of 440Hz sine at 16kHz
        let num_samples = TARGET_SAMPLE_RATE as usize;
        let data_size: u32 = (num_samples * 2) as u32;
        let file_size: u32 = 44 + data_size;

        let mut wav = Vec::with_capacity(file_size as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(file_size - 8).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&TARGET_SAMPLE_RATE.to_le_bytes());
        wav.extend_from_slice(&(TARGET_SAMPLE_RATE * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        for i in 0..num_samples {
            let sample =
                ((440.0 * 2.0 * std::f64::consts::PI * i as f64 / TARGET_SAMPLE_RATE as f64).sin()
                    * 1000.0) as i16;
            wav.extend_from_slice(&sample.to_le_bytes());
        }

        let batch = decode_audio_bytes(&wav).unwrap();
        let mut streaming = Vec::new();
        decode_audio_streaming(Bytes::from(wav), |chunk| {
            streaming.extend_from_slice(chunk);
            Ok(())
        })
        .unwrap();

        assert_eq!(
            batch.len(),
            streaming.len(),
            "Streaming and batch decode should produce the same number of samples"
        );
        for (a, b) in batch.iter().zip(streaming.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Samples differ between batch and streaming: {a} vs {b}"
            );
        }
    }
}
