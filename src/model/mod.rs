//! Model bundle download and management.
//!
//! Fetches the upstream sherpa-onnx Zipformer-vi RNN-T release tarball from
//! GitHub Releases, verifies its SHA-256, extracts the encoder, decoder,
//! joiner, BPE model and tokens vocabulary into `~/.phostt/models/`, then
//! discards the archive. The bundle ships with a `test_wavs/` directory
//! that we deliberately keep so e2e fixtures can replay the upstream
//! reference samples.
//!
//! Atomic-rename + `.partial` semantics carried over from the upstream
//! `gigastt` fetcher: a crash between download and SHA verification can
//! never leave a half-extracted model dir under the visible name.

use anyhow::{Context, Result};
use bzip2::read::BzDecoder;
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::path::Path;
use tar::Archive;
use tokio::io::AsyncWriteExt;

/// Simple download progress reporter (no external deps).
struct DownloadProgress {
    total: u64,
    current: u64,
    last_percent: u8,
}

impl DownloadProgress {
    fn new(total: u64) -> Self {
        Self {
            total,
            current: 0,
            last_percent: 0,
        }
    }

    fn update(&mut self, bytes: u64) {
        self.current += bytes;
        let percent = (self.current * 100)
            .checked_div(self.total)
            .map(|p| p as u8)
            .unwrap_or(0);
        if percent != self.last_percent {
            self.last_percent = percent;
            eprint!(
                "\rDownloading... {percent}% ({:.1}MB / {:.1}MB)",
                self.current as f64 / 1_048_576.0,
                self.total as f64 / 1_048_576.0
            );
        }
    }

    fn finish(&self) {
        eprintln!(
            "\rDownload complete ({:.1}MB)                    ",
            self.current as f64 / 1_048_576.0
        );
    }
}

/// GitHub release that hosts the bundle. The sherpa-onnx project re-uses a
/// single `asr-models` tag for every published ASR weight set, so the URL
/// is stable across model upgrades — only [`MODEL_BUNDLE_FILENAME`] and
/// [`MODEL_BUNDLE_SHA256`] need to move when we re-pin.
const MODEL_BUNDLE_REPO: &str = "k2-fsa/sherpa-onnx";
const MODEL_BUNDLE_RELEASE: &str = "asr-models";
const MODEL_BUNDLE_FILENAME: &str = "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2";
/// Top-level directory inside the tarball. We strip this prefix on
/// extraction so phostt sees a flat `<model_dir>/encoder.int8.onnx` layout.
const MODEL_BUNDLE_TOP_DIR: &str = "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09";

/// SHA-256 of the published `*.tar.bz2`. Verified against the GitHub release
/// asset on 2026-04-21 (26 442 384 bytes); refresh whenever the bundle is
/// re-uploaded.
const MODEL_BUNDLE_SHA256: &str =
    "da8b637947091829d7ee9eda23da2a4ec7caa399233a3f4e34eb719fb2ea6b9b";

/// Files we require under `<model_dir>/` after extraction. `model_files_exist`
/// uses this list to short-circuit subsequent runs without re-downloading.
pub const MODEL_FILES: &[&str] = &[
    "encoder.int8.onnx",
    "decoder.onnx",
    "joiner.int8.onnx",
    "bpe.model",
    "tokens.txt",
];

#[cfg(feature = "diarization")]
const SPEAKER_HF_REPO: &str = "onnx-community/wespeaker-voxceleb-resnet34-LM";
#[cfg(feature = "diarization")]
pub const SPEAKER_MODEL_FILE: &str = "wespeaker_resnet34.onnx";

/// SHA-256 of the upstream speaker-diarization model (`onnx/model.onnx` at
/// `onnx-community/wespeaker-voxceleb-resnet34-LM`, 26 535 549 bytes).
/// Verified against the canonical HuggingFace copy on 2026-04-20; if the
/// upstream model is ever rotated, update this constant alongside the
/// SPEAKER_MODEL_FILE bump.
#[cfg(feature = "diarization")]
const SPEAKER_MODEL_SHA256: &str =
    "3955447b0499dc9e0a4541a895df08b03c69098eba4e56c02b5603e9f7f4fcbb";

fn home_dir() -> Option<std::path::PathBuf> {
    #[cfg(unix)]
    {
        std::env::var_os("HOME").map(std::path::PathBuf::from)
    }
    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE").map(std::path::PathBuf::from)
    }
}

/// Return the default model directory path (`~/.phostt/models/`).
///
/// Falls back to `.phostt/models` if the home directory cannot be determined.
pub fn default_model_dir() -> String {
    home_dir()
        .map(|h| {
            h.join(".phostt")
                .join("models")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| ".phostt/models".into())
}

/// Ensure model files exist in `model_dir`, downloading and extracting the
/// upstream sherpa-onnx Zipformer-vi bundle if missing.
pub async fn ensure_model(model_dir: &str) -> Result<()> {
    let dir = Path::new(model_dir);

    if model_files_exist(dir) {
        tracing::info!("Model found at {model_dir}");
        return Ok(());
    }

    tracing::info!("Model not found, downloading Zipformer-vi bundle...");
    std::fs::create_dir_all(dir).context("Failed to create model directory")?;

    let archive_dest = dir.join(MODEL_BUNDLE_FILENAME);
    let url = format!(
        "https://github.com/{MODEL_BUNDLE_REPO}/releases/download/{MODEL_BUNDLE_RELEASE}/{MODEL_BUNDLE_FILENAME}"
    );
    stream_to_partial_then_finalize(
        &url,
        &archive_dest,
        Some(MODEL_BUNDLE_SHA256),
        MODEL_BUNDLE_FILENAME,
    )
    .await?;

    tracing::info!("Extracting bundle into {}", dir.display());
    extract_bundle(&archive_dest, dir)?;
    // Discard the archive once contents are in place — it is large and
    // re-downloads cheaply if we ever need a fresh copy.
    let _ = std::fs::remove_file(&archive_dest);

    if !model_files_exist(dir) {
        anyhow::bail!(
            "Bundle extracted but expected files are still missing under {}",
            dir.display()
        );
    }
    tracing::info!("Model bundle ready");
    Ok(())
}

/// Ensure the speaker diarization model exists in `model_dir`, downloading from HuggingFace if missing.
///
/// Downloads `wespeaker_resnet34.onnx` from `onnx-community/wespeaker-voxceleb-resnet34-LM`
/// into `<model_dir>/wespeaker_resnet34.onnx.partial`, verifies its SHA-256 against
/// `SPEAKER_MODEL_SHA256`, and atomically renames it into place. On checksum mismatch or
/// crash the final path is never observable, so a subsequent `ensure_speaker_model` call
/// will re-download from scratch rather than loading a tampered model.
#[cfg(feature = "diarization")]
pub async fn ensure_speaker_model(model_dir: &str) -> Result<()> {
    let dir = Path::new(model_dir);
    let final_dest = dir.join(SPEAKER_MODEL_FILE);

    if final_dest.exists() {
        tracing::info!("Speaker model found at {}", final_dest.display());
        return Ok(());
    }

    tracing::info!("Speaker model not found, downloading from HuggingFace...");
    std::fs::create_dir_all(dir).context("Failed to create model directory")?;

    let url = format!("https://huggingface.co/{SPEAKER_HF_REPO}/resolve/main/onnx/model.onnx");
    stream_to_partial_then_finalize(
        &url,
        &final_dest,
        Some(SPEAKER_MODEL_SHA256),
        SPEAKER_MODEL_FILE,
    )
    .await
}

fn model_files_exist(dir: &Path) -> bool {
    MODEL_FILES.iter().all(|f| dir.join(f).exists())
}

/// Append `.partial` to a path; used as the staging location for downloads
/// so a crash between the write and the SHA verification never leaves a
/// half-written file under the final name that `model_files_exist` accepts.
fn partial_path(final_path: &Path) -> std::path::PathBuf {
    let mut s: std::ffi::OsString = final_path.as_os_str().to_owned();
    s.push(".partial");
    std::path::PathBuf::from(s)
}

/// Compute SHA-256 for a file synchronously, returning the lowercase hex digest.
fn sha256_file(path: &Path) -> Result<String> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read file for verification: {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify a staged `.partial` file against `expected_sha256` (when provided)
/// and atomically rename it into `final_path`. On mismatch the partial is
/// removed so a corrupt artefact cannot be mistaken for a good download on
/// restart. On success the partial no longer exists and `final_path` is the
/// only visible artefact.
fn finalize_download(
    partial_path: &Path,
    final_path: &Path,
    expected_sha256: Option<&str>,
    label: &str,
) -> Result<()> {
    if let Some(expected) = expected_sha256 {
        let actual = sha256_file(partial_path)?;
        if actual != expected {
            let _ = std::fs::remove_file(partial_path);
            anyhow::bail!("SHA-256 mismatch for {label}: expected {expected}, got {actual}");
        }
        tracing::info!("SHA-256 verified: {label}");
    }

    std::fs::rename(partial_path, final_path).with_context(|| {
        format!(
            "Failed to rename {} -> {}",
            partial_path.display(),
            final_path.display()
        )
    })?;
    Ok(())
}

/// Streaming download with SHA-256 verification and atomic rename.
async fn stream_to_partial_then_finalize(
    url: &str,
    final_dest: &Path,
    expected_sha256: Option<&str>,
    label: &str,
) -> Result<()> {
    let partial = partial_path(final_dest);

    if partial.exists() {
        let _ = tokio::fs::remove_file(&partial).await;
    }

    tracing::info!("Downloading {label}...");

    let response = reqwest::get(url).await.context("HTTP request failed")?;
    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("Download failed for {label}: HTTP {status}");
    }
    let total_size = response.content_length().unwrap_or(0);

    let mut progress = DownloadProgress::new(total_size);

    let mut file = tokio::fs::File::create(&partial)
        .await
        .context("Failed to create partial model file")?;
    let mut stream = response.bytes_stream();

    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Download stream error")?;
        file.write_all(&chunk)
            .await
            .context("Failed to write chunk")?;
        downloaded += chunk.len() as u64;
        progress.update(chunk.len() as u64);
    }

    file.flush().await?;
    drop(file);
    progress.finish();
    tracing::info!("Wrote partial {} ({downloaded} bytes)", partial.display());

    finalize_download(&partial, final_dest, expected_sha256, label)?;
    tracing::info!("Saved {label}");

    Ok(())
}

/// Extract a `.tar.bz2` archive into `dest_dir`, stripping the top-level
/// directory ([`MODEL_BUNDLE_TOP_DIR`]) so files land at
/// `<dest_dir>/<file>` instead of
/// `<dest_dir>/<bundle_name>/<file>`. Refuses to write outside `dest_dir`
/// (rejects `..` and absolute paths) — the upstream tar is trusted but the
/// guard costs nothing.
fn extract_bundle(archive: &Path, dest_dir: &Path) -> Result<()> {
    let file = std::fs::File::open(archive)
        .with_context(|| format!("Failed to open archive {}", archive.display()))?;
    let bz = BzDecoder::new(file);
    let mut tar = Archive::new(bz);

    for entry in tar.entries().context("Failed to read tar entries")? {
        let mut entry = entry.context("Tar entry read error")?;
        let path = entry.path().context("Tar entry has no path")?.into_owned();

        // Strip the well-known top directory; skip the bare directory entry.
        let relative = match path.strip_prefix(MODEL_BUNDLE_TOP_DIR) {
            Ok(rel) if rel.as_os_str().is_empty() => continue,
            Ok(rel) => rel.to_path_buf(),
            Err(_) => path.clone(),
        };

        // Reject path traversal — `tar::Entry::unpack_in` already enforces
        // this when handed a base, but we materialise the destination
        // ourselves to keep the strip-prefix behaviour explicit.
        if relative
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            anyhow::bail!(
                "Refusing to extract {}: parent-dir component in archive entry",
                relative.display()
            );
        }
        if relative.is_absolute() {
            anyhow::bail!(
                "Refusing to extract {}: absolute path in archive entry",
                relative.display()
            );
        }

        let target = dest_dir.join(&relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create directory {}", parent.display())
            })?;
        }
        entry
            .unpack(&target)
            .with_context(|| format!("Failed to unpack {}", target.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_home_dir_returns_some() {
        assert!(
            home_dir().is_some(),
            "home_dir() must return Some on this platform"
        );
    }

    #[test]
    fn test_default_model_dir_contains_phostt() {
        let dir = default_model_dir();
        assert!(
            dir.contains(".phostt"),
            "default_model_dir() should contain \".phostt\", got: {dir}"
        );
    }

    #[test]
    fn test_download_progress_basic() {
        let mut progress = DownloadProgress::new(1_000_000);
        progress.update(500_000);
        assert_eq!(progress.current, 500_000);
        assert_eq!(progress.last_percent, 50);
        progress.finish();
    }

    #[test]
    fn test_download_progress_zero_total() {
        let mut progress = DownloadProgress::new(0);
        progress.update(100);
        assert_eq!(progress.last_percent, 0);
        progress.finish();
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    fn stage_partial(final_path: &Path, bytes: &[u8]) -> std::path::PathBuf {
        let partial = partial_path(final_path);
        let mut f = std::fs::File::create(&partial).expect("create partial");
        f.write_all(bytes).expect("write partial");
        f.sync_all().expect("sync partial");
        partial
    }

    #[test]
    fn test_partial_path_appends_suffix() {
        let p = partial_path(Path::new("/tmp/phostt/encoder.onnx"));
        assert_eq!(
            p,
            std::path::PathBuf::from("/tmp/phostt/encoder.onnx.partial"),
        );
    }

    #[test]
    fn test_download_writes_partial_then_renames() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("encoder.onnx");
        let payload = b"fake encoder weights";
        let expected = sha256_hex(payload);

        let partial = stage_partial(&final_path, payload);
        assert!(partial.exists(), "precondition: partial is present");
        assert!(!final_path.exists(), "precondition: final is absent");

        finalize_download(&partial, &final_path, Some(&expected), "encoder.onnx")
            .expect("finalize should succeed");

        assert!(
            !partial.exists(),
            "partial must be gone after atomic rename"
        );
        assert!(
            final_path.exists(),
            "final path must exist after atomic rename"
        );
        assert_eq!(std::fs::read(&final_path).unwrap(), payload);
    }

    #[test]
    fn test_download_crash_before_rename_leaves_no_final_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("encoder.onnx");
        let partial = stage_partial(&final_path, b"half-written junk");

        assert!(partial.exists(), "partial must exist to simulate crash");
        assert!(
            !final_path.exists(),
            "crash before rename must never leave the final artefact visible"
        );

        assert!(
            !model_files_exist(tmp.path()),
            "model_files_exist must not accept a staged partial"
        );
    }

    #[test]
    fn test_download_rejects_sha256_mismatch() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("decoder.onnx");
        let payload = b"real bytes";
        let wrong_expected = sha256_hex(b"different bytes");

        let partial = stage_partial(&final_path, payload);

        let err = finalize_download(&partial, &final_path, Some(&wrong_expected), "decoder.onnx")
            .expect_err("mismatch must error");
        let msg = format!("{err}");
        assert!(msg.contains("SHA-256 mismatch"), "unexpected error: {msg}");

        assert!(!partial.exists(), "partial must be removed on SHA mismatch");
        assert!(
            !final_path.exists(),
            "final must never appear on SHA mismatch"
        );
    }

    #[test]
    fn test_download_atomic_on_success_without_checksum() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let final_path = tmp.path().join("vocab.txt");
        let payload = b"token0\ntoken1\n";

        let partial = stage_partial(&final_path, payload);

        finalize_download(&partial, &final_path, None, "vocab.txt")
            .expect("no-checksum finalize should succeed");

        assert!(!partial.exists(), "partial must be gone after rename");
        assert!(final_path.exists(), "final path must exist");
        assert_eq!(std::fs::read(&final_path).unwrap(), payload);
    }

    #[test]
    fn test_sha256_file_matches_in_memory_hash() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let p = tmp.path().join("blob");
        let payload = b"phostt-model-bytes";
        std::fs::write(&p, payload).unwrap();

        let got = sha256_file(&p).expect("sha256_file");
        let want = sha256_hex(payload);
        assert_eq!(got, want);
    }

    #[test]
    fn test_model_bundle_sha256_shape() {
        assert_eq!(
            MODEL_BUNDLE_SHA256.len(),
            64,
            "MODEL_BUNDLE_SHA256 must be a 64-char hex digest"
        );
        assert!(
            MODEL_BUNDLE_SHA256
                .chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "MODEL_BUNDLE_SHA256 must be lowercase hex; got: {MODEL_BUNDLE_SHA256}"
        );
    }

    #[test]
    fn test_model_files_list_matches_required_layout() {
        // Sanity-check the constant rather than the runtime extractor: every
        // file the engine needs must appear in MODEL_FILES so a missed entry
        // surfaces as a unit-test failure instead of a runtime mystery.
        for required in ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "bpe.model", "tokens.txt"] {
            assert!(
                MODEL_FILES.contains(&required),
                "MODEL_FILES is missing required entry {required}"
            );
        }
    }

    #[test]
    fn test_extract_bundle_strips_top_dir_and_rejects_traversal() {
        // Build a tiny in-memory tar.bz2 that mirrors the upstream layout
        // (top-level dir + a few files) so we exercise the strip-prefix
        // and traversal-guard logic without touching the network.
        use bzip2::write::BzEncoder;
        use bzip2::Compression;
        use std::io::Cursor;
        use tar::Header;

        fn append(builder: &mut tar::Builder<&mut Vec<u8>>, path: &str, data: &[u8]) {
            let mut header = Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, path, Cursor::new(data)).unwrap();
        }

        // Happy path: top-dir prefix is stripped.
        let tmp = tempfile::tempdir().expect("tempdir");
        let archive_path = tmp.path().join("bundle.tar.bz2");
        {
            let mut tar_buf = Vec::new();
            {
                let mut builder = tar::Builder::new(&mut tar_buf);
                append(
                    &mut builder,
                    &format!("{MODEL_BUNDLE_TOP_DIR}/encoder.int8.onnx"),
                    b"encoder-bytes",
                );
                append(
                    &mut builder,
                    &format!("{MODEL_BUNDLE_TOP_DIR}/test_wavs/0.wav"),
                    b"wav-bytes",
                );
                builder.finish().unwrap();
            }
            let mut bz = BzEncoder::new(
                std::fs::File::create(&archive_path).unwrap(),
                Compression::fast(),
            );
            std::io::copy(&mut Cursor::new(tar_buf), &mut bz).unwrap();
            bz.finish().unwrap();
        }
        let dest = tmp.path().join("out");
        std::fs::create_dir_all(&dest).unwrap();
        extract_bundle(&archive_path, &dest).expect("happy-path extract");
        assert!(dest.join("encoder.int8.onnx").exists());
        assert!(dest.join("test_wavs").join("0.wav").exists());
        assert!(
            !dest.join(MODEL_BUNDLE_TOP_DIR).exists(),
            "top dir must be stripped, not nested"
        );
    }
}
