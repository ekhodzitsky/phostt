//! Self-service benchmark report for README metrics.
//!
//! Run with:
//!   cargo test --test benchmark_report -- --ignored --nocapture
//!   cargo test --test benchmark_report --features coreml -- --ignored --nocapture
//!
//! Outputs a Markdown table row you can paste into README.md.

mod common;

use phostt::inference::{Engine, audio};
use std::time::Instant;

const WARMUP_RUNS: usize = 3;
const BENCHMARK_RUNS: usize = 20;

#[test]
#[ignore = "Requires model download"]
fn benchmark_latency_and_rtf() {
    let model_dir = common::model_dir();
    let wav_path = common::test_wav_path(0);
    let samples = audio::decode_audio_file(wav_path.to_str().unwrap()).expect("WAV decode failed");

    let audio_duration_sec = samples.len() as f64 / 16_000.0;
    let engine = Engine::load(&model_dir).expect("Engine load failed");
    let rt = tokio::runtime::Runtime::new().expect("Tokio runtime");
    let mut triplet = rt.block_on(async { engine.pool.checkout().await.unwrap() });

    // Warmup
    for _ in 0..WARMUP_RUNS {
        let _ = engine.transcribe_samples(&samples, &mut triplet).unwrap();
    }

    let mut latencies = Vec::with_capacity(BENCHMARK_RUNS);
    for _ in 0..BENCHMARK_RUNS {
        let start = Instant::now();
        let _ = engine.transcribe_samples(&samples, &mut triplet).unwrap();
        latencies.push(start.elapsed().as_secs_f64() * 1000.0); // ms
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let median = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
    let min = latencies[0];
    let max = latencies[latencies.len() - 1];
    let rtf = audio_duration_sec / (mean / 1000.0);

    eprintln!();
    eprintln!("## Benchmark Report");
    eprintln!();
    eprintln!("| Metric | Value |");
    eprintln!("|---|---|");
    eprintln!("| Audio duration | {:.2} s |", audio_duration_sec);
    eprintln!("| Mean latency | {:.2} ms |", mean);
    eprintln!("| Median latency | {:.2} ms |", median);
    eprintln!("| P95 latency | {:.2} ms |", p95);
    eprintln!("| P99 latency | {:.2} ms |", p99);
    eprintln!("| Min latency | {:.2} ms |", min);
    eprintln!("| Max latency | {:.2} ms |", max);
    eprintln!("| RTF (real-time factor) | {:.2}× |", rtf);
    eprintln!();
    eprintln!("Copy this row into README.md:");
    eprintln!();

    #[cfg(not(any(feature = "coreml", feature = "cuda")))]
    eprintln!(
        "| CPU (Apple Silicon / x86_64) | {:.2} ms | {:.2}× |",
        mean, rtf
    );
    #[cfg(feature = "coreml")]
    eprintln!("| CoreML (Neural Engine) | {:.2} ms | {:.2}× |", mean, rtf);
    #[cfg(feature = "cuda")]
    eprintln!("| CUDA | {:.2} ms | {:.2}× |", mean, rtf);
}
