//! Log-mel (FBANK) feature extraction for Zipformer-vi.
//!
//! Thin wrapper over `kaldi-native-fbank` so phostt's preprocessing matches
//! the `kaldifeat` pipeline used to train the upstream Zipformer-vi weights:
//! 80 mel bins, 25 ms / 10 ms povey-windowed frames, 0.97 preemphasis,
//! Slaney mel scale, log-FBANK without per-frame energy. Dither is forced
//! to 0 so transcription is bit-deterministic.
//!
//! Streaming preprocessing in [`super::StreamingState`] will switch to
//! [`kaldi_native_fbank::OnlineFeature`] in the next step; this offline
//! [`MelSpectrogram::compute`] keeps the legacy `(Vec<f32>, num_frames)`
//! signature so `Engine::transcribe_file` and the existing decode plumbing
//! continue to compile while the rest of the inference path is rewired.

use kaldi_native_fbank::fbank::{FbankComputer, FbankOptions};
use kaldi_native_fbank::online::{FeatureComputer, OnlineFeature};

const SAMPLE_RATE: f32 = 16000.0;

pub struct MelSpectrogram {
    opts: FbankOptions,
    n_mels: usize,
}

impl MelSpectrogram {
    pub fn new() -> Self {
        let opts = phostt_fbank_options();
        // Build a probe computer once so a typo in [`phostt_fbank_options`]
        // surfaces immediately at engine load instead of at the first frame.
        let probe = FbankComputer::new(opts.clone()).expect("FBANK options valid");
        let n_mels = probe.dim();
        Self { opts, n_mels }
    }

    /// Compute log-FBANK features for an offline buffer.
    /// Returns `[num_frames, n_mels]` flat in frames-first layout — the
    /// shape Zipformer's encoder ONNX expects for its `features` input.
    pub fn compute(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        if samples.is_empty() {
            return (vec![0.0; self.n_mels], 1);
        }
        let computer = FbankComputer::new(self.opts.clone()).expect("FBANK options valid");
        let mut online = OnlineFeature::new(FeatureComputer::Fbank(computer));
        online.accept_waveform(SAMPLE_RATE, samples);
        online.input_finished();

        let num_frames = online.num_frames_ready();
        if num_frames == 0 {
            // snip_edges=true drops anything shorter than one window.
            // Return a single all-zero frame so downstream tensor shapes
            // remain valid for the few callers that pass <25 ms buffers.
            return (vec![0.0; self.n_mels], 1);
        }
        let mut out = vec![0f32; num_frames * self.n_mels];
        for f in 0..num_frames {
            let frame = online
                .get_frame(f)
                .expect("frame index < num_frames_ready must be retrievable");
            // Frames-first: contiguous mel vector for frame `f` lives at
            // `out[f * n_mels .. (f + 1) * n_mels]`.
            out[f * self.n_mels..(f + 1) * self.n_mels].copy_from_slice(&frame[..self.n_mels]);
        }
        (out, num_frames)
    }
}

/// FBANK options used by Zipformer-vi. Kaldi-default everything except the
/// three knobs the model demands: 80 mel bins, no per-frame energy, no
/// dither (so identical input -> identical features in tests + production).
pub(super) fn phostt_fbank_options() -> FbankOptions {
    let mut opts = FbankOptions::default();
    opts.mel_opts.num_bins = 80;
    opts.use_energy = false;
    opts.frame_opts.dither = 0.0;
    opts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::N_MELS;

    #[test]
    fn test_default_dim_matches_const() {
        let mel = MelSpectrogram::new();
        assert_eq!(mel.n_mels, N_MELS, "n_mels must agree with the public constant");
    }

    #[test]
    fn test_silence_returns_finite_features() {
        let mel = MelSpectrogram::new();
        let silence = vec![0.0_f32; 16000];
        let (features, num_frames) = mel.compute(&silence);
        assert!(num_frames > 0, "silence must still produce frames");
        assert_eq!(features.len(), N_MELS * num_frames);
        for &v in &features {
            assert!(v.is_finite(), "expected finite mel value, got {v}");
        }
    }

    #[test]
    fn test_too_short_returns_single_zero_frame() {
        let mel = MelSpectrogram::new();
        let samples = vec![0.0_f32; 100];
        let (features, num_frames) = mel.compute(&samples);
        assert_eq!(num_frames, 1);
        assert_eq!(features.len(), N_MELS);
    }

    #[test]
    fn test_sine_wave_has_dynamic_range() {
        let mel = MelSpectrogram::new();
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let (features, num_frames) = mel.compute(&samples);
        assert!(num_frames > 0);
        let max = features.iter().copied().fold(f32::MIN, f32::max);
        let min = features.iter().copied().fold(f32::MAX, f32::min);
        assert!(
            max - min > 1.0,
            "expected wide log-mel range for a 440 Hz tone, got max={max} min={min}"
        );
    }

    #[test]
    fn test_one_second_yields_about_one_hundred_frames() {
        // snip_edges=true with 25 ms window, 10 ms shift on 1 s of audio:
        // (1000 ms - 25 ms) / 10 ms + 1 = 98 frames. Allow a small slack
        // because rounding-to-power-of-two padding may shift the boundary
        // by one frame on some inputs.
        let mel = MelSpectrogram::new();
        let samples = vec![0.0_f32; 16000];
        let (_, num_frames) = mel.compute(&samples);
        assert!(
            (96..=100).contains(&num_frames),
            "expected ~98 frames for 1 s of audio, got {num_frames}"
        );
    }
}
