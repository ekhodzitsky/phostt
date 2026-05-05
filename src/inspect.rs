//! ONNX session introspection for the `phostt inspect` debug subcommand.
//!
//! Opens the encoder, decoder and joiner ONNX files standalone (no
//! [`crate::inference::Engine`] / pool warm-up) and dumps their
//! `inputs() / outputs()` so an upstream model rotation that renames a
//! tensor or changes a dtype is caught immediately, before the next
//! `phostt transcribe` returns garbled audio.

use anyhow::{Context, Result};
use ort::session::Session;
use std::path::Path;

const ONNX_FILES: &[(&str, &str)] = &[
    ("encoder", "encoder.int8.onnx"),
    ("decoder", "decoder.onnx"),
    ("joiner", "joiner.int8.onnx"),
];

/// Open every ONNX session in `model_dir` and print its tensor I/O.
/// Output goes to stdout so a user can pipe it into a snapshot file
/// when re-pinning [`crate::model`] against a new upstream bundle.
pub fn inspect_models(model_dir: &Path) -> Result<()> {
    println!("phostt inspect — model_dir: {}", model_dir.display());

    for (label, filename) in ONNX_FILES {
        let path = model_dir.join(filename);
        anyhow::ensure!(
            path.exists(),
            "{label} model not found at {}",
            path.display()
        );
        let session = Session::builder()
            .map_err(anyhow::Error::msg)?
            .commit_from_file(&path)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("Failed to open {} session at {}", label, path.display()))?;

        println!();
        println!("== {label} ({filename}) ==");
        print_outlets("inputs", session.inputs());
        print_outlets("outputs", session.outputs());
    }

    Ok(())
}

fn print_outlets(kind: &str, outlets: &[ort::value::Outlet]) {
    if outlets.is_empty() {
        println!("  {kind}: <none>");
        return;
    }
    println!("  {kind}:");
    for outlet in outlets {
        println!(
            "    - name={:<40} dtype={:?}",
            outlet.name(),
            outlet.dtype()
        );
    }
}
