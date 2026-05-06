use phostt::inference::{Engine, audio};

const REFERENCES: &[(&str, &str)] = &[
    (
        "0.wav",
        "RỒI CŨNG HỖ TRỢ CHO LÂU LÂU CŨNG CHO GẠO CHO NÀY KIA",
    ),
    ("1.wav", "NHỮNG NƠI ĐÃ KHỐNG CHẾ ĐƯỢC CĂN BỆNH"),
    ("2.wav", "ÂM LƯỢNG TV GIẢM"),
];

/// Levenshtein edit distance on word-tokenized strings.
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

/// Word Error Rate between reference and hypothesis.
fn wer(reference: &str, hypothesis: &str) -> f64 {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();
    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }
    let distance = levenshtein(&ref_words, &hyp_words);
    distance as f64 / ref_words.len() as f64
}

#[test]
#[ignore = "Requires model download"]
fn wer_regression() {
    let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
    let model_dir = match home.as_ref().map(|p| p.join(".phostt/models")) {
        Some(ref d) if d.join("encoder.int8.onnx").exists() => d.clone(),
        _ => {
            eprintln!("Skipping wer_regression: model not found at ~/.phostt/models");
            return;
        }
    };

    let engine = Engine::load(model_dir.to_str().unwrap()).expect("Engine load failed");
    let rt = tokio::runtime::Runtime::new().expect("Tokio runtime creation failed");

    for (filename, reference) in REFERENCES {
        let wav_path = model_dir.join("test_wavs").join(filename);
        if !wav_path.exists() {
            panic!("Test WAV not found: {}", wav_path.display());
        }

        let samples =
            audio::decode_audio_file(wav_path.to_str().unwrap()).expect("WAV decode failed");

        let mut triplet = rt.block_on(async { engine.pool.checkout().await.unwrap() });
        let result = engine.transcribe_samples(&samples, &mut triplet).unwrap();

        let computed_wer = wer(reference, &result.text);
        assert!(
            computed_wer <= 0.30,
            "WER for {filename} is {computed_wer} (threshold 0.30)\nReference: {reference}\nHypothesis: {}",
            result.text
        );
    }
}
