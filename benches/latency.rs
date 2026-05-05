use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use phostt::inference::{Engine, audio};
use std::time::Duration;

fn latency_benchmark(c: &mut Criterion) {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    let model_dir = home.as_ref().map(|p| p.join(".phostt/models"));
    let model_dir = match model_dir {
        Some(ref d) if d.join("encoder.int8.onnx").exists() => d,
        _ => {
            eprintln!("Skipping latency benchmark: model not found at ~/.phostt/models");
            return;
        }
    };

    let wav_path = model_dir.join("test_wavs").join("0.wav");
    if !wav_path.exists() {
        eprintln!(
            "Skipping latency benchmark: test WAV not found at {}",
            wav_path.display()
        );
        return;
    }

    let engine = Engine::load(model_dir.to_str().unwrap()).expect("Engine load failed");
    let samples = audio::decode_audio_file(wav_path.to_str().unwrap()).expect("WAV decode failed");
    // samples at 16 kHz → milliseconds of audio
    let audio_duration_ms = (samples.len() as f64 / 16.0).round() as u64;

    let rt = tokio::runtime::Runtime::new().expect("Tokio runtime creation failed");
    let mut triplet = rt.block_on(async { engine.pool.checkout().await.unwrap() });

    let mut group = c.benchmark_group("latency");
    group.measurement_time(Duration::from_secs(30));
    // Throughput as milliseconds of audio per second of wall time.
    // e.g. 3500 elem/s reported by Criterion = 3.5× real-time.
    group.throughput(Throughput::Elements(audio_duration_ms));

    group.bench_function("transcribe_0", |b| {
        b.iter(|| {
            engine
                .transcribe_samples(black_box(&samples), &mut triplet)
                .unwrap()
        });
    });

    group.finish();
}

criterion_group!(benches, latency_benchmark);
criterion_main!(benches);
