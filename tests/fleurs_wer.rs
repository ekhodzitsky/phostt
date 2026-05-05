//! WER benchmark on the Google FLEURS Vietnamese test split.
//!
//! Requires the FLEURS dataset (~350 utterances, ~1–2 h of audio).
//! Prepare it once with:
//!   python scripts/prepare_fleurs_benchmark.py
//!
//! Run with:
//!   cargo test --test fleurs_wer -- --ignored
//!
//! The test asserts mean WER stays below a threshold (tune after the first
//! successful run on your hardware).

use phostt::inference::{Engine, audio};
use std::path::PathBuf;

/// Maximum acceptable mean WER. Tune this after the first successful run.
/// Expected ballpark for Zipformer-vi RNNT on FLEURS vi_vn: 0.10–0.18.
const MAX_MEAN_WER: f64 = 0.25;

fn levenshtein(a: &[&str], b: &[&str]) -> usize {
    let n = a.len();
    let m = b.len();
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1).min(prev[j] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

fn wer(reference: &str, hypothesis: &str) -> f64 {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();
    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein(&ref_words, &hyp_words) as f64 / ref_words.len() as f64
}

#[test]
#[ignore = "Requires FLEURS dataset (~350 WAVs) and model download"]
fn fleurs_vi_wer_benchmark() {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let manifest = match home.as_ref().map(|p| p.join(".phostt/benchmark/fleurs_vi/manifest.tsv")) {
        Some(ref p) if p.exists() => p.clone(),
        _ => {
            eprintln!("Skipping: FLEURS manifest not found. Run: python scripts/prepare_fleurs_benchmark.py");
            return;
        }
    };

    let model_dir = match home.as_ref().map(|p| p.join(".phostt/models")) {
        Some(ref d) if d.join("encoder.int8.onnx").exists() => d.clone(),
        _ => {
            eprintln!("Skipping: model not found at ~/.phostt/models");
            return;
        }
    };

    let engine = Engine::load(model_dir.to_str().unwrap()).expect("Engine load failed");
    let rt = tokio::runtime::Runtime::new().expect("Tokio runtime");

    let manifest_text = std::fs::read_to_string(&manifest).expect("read manifest");
    let mut lines = manifest_text.lines();
    let _header = lines.next().expect("manifest header");

    let mut total_wer = 0.0;
    let mut count = 0usize;
    let mut failures = 0usize;

    for line in lines {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 3 {
            continue;
        }
        let wav_path = cols[1];
        let reference = cols[2];

        let samples = match audio::decode_audio_file(wav_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("WARN: failed to decode {wav_path}: {e}");
                failures += 1;
                continue;
            }
        };

        let mut triplet = rt.block_on(async { engine.pool.checkout().await.unwrap() });
        let result = match engine.transcribe_samples(&samples, &mut triplet) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("WARN: inference failed for {wav_path}: {e}");
                failures += 1;
                continue;
            }
        };

        let sample_wer = wer(reference, &result.text);
        total_wer += sample_wer;
        count += 1;

        if sample_wer > 0.5 {
            eprintln!(
                "HIGH WER ({:.2}) for {}\n  REF: {}\n  HYP: {}",
                sample_wer, wav_path, reference, result.text
            );
        }
    }

    if count == 0 {
        panic!("No samples processed — check manifest and audio files");
    }

    let mean_wer = total_wer / count as f64;
    eprintln!(
        "FLEURS vi benchmark: {} samples, {} decode failures, mean WER = {:.4} ({:.2}%)",
        count, failures, mean_wer, mean_wer * 100.0
    );

    assert!(
        mean_wer <= MAX_MEAN_WER,
        "Mean WER {:.4} exceeds threshold {:.4}",
        mean_wer, MAX_MEAN_WER
    );
}
