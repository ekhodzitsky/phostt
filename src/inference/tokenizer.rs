//! Subword tokenizer for Zipformer-vi RNN-T.
//!
//! Reads `tokens.txt` from the sherpa-onnx Zipformer-vi bundle. The file
//! lists one token per line in `"<token> <id>"` form, ids dense from 0:
//!
//! ```text
//! <blk> 0
//! <sos/eos> 1
//! <unk> 2
//! ▁HAI 3
//! ▁KHÔNG 4
//! ...
//! ```
//!
//! The leading [`▁`] (U+2581) marks a word boundary in SentencePiece BPE.
//! Decode treats it as a space and otherwise concatenates token text
//! verbatim — the upstream model emits already-cased Vietnamese text with
//! diacritics, so no extra normalization is needed.

use anyhow::{Context, Result};
use std::path::Path;

/// SentencePiece tokens that must never appear in transcribed output.
/// `<blk>` is the RNN-T blank emitted between real tokens; `<sos/eos>`
/// and `<unk>` are training-time markers we filter defensively even
/// though the upstream model is not expected to predict them.
fn is_special(token: &str) -> bool {
    matches!(token, "<blk>" | "<sos/eos>" | "<unk>" | "<s>" | "</s>")
}

#[derive(Debug)]
pub struct Tokenizer {
    tokens: Vec<String>,
    blank_id: usize,
}

impl Tokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read tokens file: {}", path.display()))?;

        let mut tokens = Vec::new();
        for line in content.lines() {
            if line.is_empty() {
                continue;
            }
            // Each line is `<token> <id>`; the id is appended as the last
            // whitespace-separated field. Split from the right so tokens that
            // happen to contain a space inside themselves still survive.
            let token = match line.rsplit_once(|c: char| c.is_whitespace()) {
                Some((tok, tail)) if tail.trim().parse::<usize>().is_ok() => tok.to_string(),
                _ => line.to_string(),
            };
            tokens.push(token);
        }

        let blank_id = tokens
            .iter()
            .position(|t| t == "<blk>")
            .context("tokens.txt is missing a <blk> entry")?;

        tracing::info!(
            "Loaded vocabulary: {} tokens, blank_id={}",
            tokens.len(),
            blank_id
        );

        Ok(Self { tokens, blank_id })
    }

    pub fn blank_id(&self) -> usize {
        self.blank_id
    }

    /// Get the raw subword text for `id`. Returns an empty slice for the
    /// blank, special markers, and out-of-range ids so callers can blindly
    /// concatenate `token_text(...)` without filtering noise themselves.
    pub fn token_text(&self, id: usize) -> &str {
        if id >= self.tokens.len() {
            return "";
        }
        let t = &self.tokens[id];
        if is_special(t) { "" } else { t }
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Build a tokenizer from an in-memory token list. Test-only: avoids
    /// file I/O and model dependencies.
    #[cfg(test)]
    pub fn from_tokens(tokens: Vec<String>, blank_id: usize) -> Self {
        Self { tokens, blank_id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_vocab(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "{content}").unwrap();
        f
    }

    /// First handful of lines from the upstream
    /// `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/tokens.txt`. Used to
    /// pin the parser against the actual on-disk format so a layout drift
    /// (tab vs space, missing blank, header line) fails locally instead
    /// of at first inference.
    const SHERPA_PREFIX: &str = "<blk> 0\n<sos/eos> 1\n<unk> 2\n▁HAI 3\n▁KHÔNG 4\n";

    #[test]
    fn test_load_sherpa_format_assigns_blank_zero() {
        let f = write_vocab(SHERPA_PREFIX);
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.vocab_size(), 5);
        assert_eq!(
            tok.blank_id(),
            0,
            "sherpa-onnx Zipformer-vi puts <blk> at id=0"
        );
    }

    #[test]
    fn test_special_tokens_filtered_from_text() {
        let f = write_vocab(SHERPA_PREFIX);
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.token_text(0), "");
        assert_eq!(tok.token_text(1), "");
        assert_eq!(tok.token_text(2), "");
        assert_eq!(tok.token_text(3), "▁HAI");
        assert_eq!(tok.token_text(4), "▁KHÔNG");
    }

    #[test]
    fn test_token_text_out_of_range_returns_empty() {
        let f = write_vocab(SHERPA_PREFIX);
        let tok = Tokenizer::load(f.path()).unwrap();
        assert_eq!(tok.token_text(99), "");
    }

    #[test]
    fn test_load_rejects_vocab_without_blank() {
        let f = write_vocab("a 0\nb 1\n");
        let err = Tokenizer::load(f.path()).expect_err("missing <blk> must error");
        assert!(
            format!("{err:#}").contains("<blk>"),
            "error must mention the missing <blk> entry, got: {err:#}"
        );
    }

    #[test]
    fn test_word_boundary_marker_is_preserved_for_caller() {
        // The decode loop in Engine::process_chunk inspects the leading
        // U+2581 to decide where to insert spaces, so token_text must
        // hand the marker through unmodified.
        let f = write_vocab(SHERPA_PREFIX);
        let tok = Tokenizer::load(f.path()).unwrap();
        assert!(tok.token_text(3).starts_with('\u{2581}'));
    }
}
