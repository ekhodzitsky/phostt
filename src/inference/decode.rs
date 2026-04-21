//! RNN-T greedy decoding for Zipformer-vi (stateless decoder + joiner).
//!
//! Mirrors the inner loop of `sherpa-onnx`'s
//! `OfflineTransducerGreedySearchDecoder` but operates on the single-batch
//! shape phostt uses end-to-end. The decoder is stateless: each step we
//! feed the rolling [`super::CONTEXT_SIZE`]-token window, get back the
//! decoder embedding, and reuse it for every blank frame until a non-blank
//! token arrives — that one cached call is the fast path.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::TensorRef;

use super::{CONTEXT_SIZE, DECODER_OUT_DIM, DecoderState, ENCODER_OUT_DIM};

const MAX_TOKENS_PER_STEP: usize = 10;
/// Number of consecutive blank encoder frames that flip the engine into
/// "speech ended" — same threshold the legacy GigaAM pipeline used.
pub(crate) const ENDPOINT_BLANK_THRESHOLD: usize = 15;

/// Token emitted by the decoder with metadata.
#[derive(Debug, Clone)]
pub(crate) struct TokenInfo {
    pub token_id: usize,
    pub frame_index: usize,
    pub confidence: f32,
}

/// Result of greedy decode: tokens + endpointing signal.
#[derive(Debug)]
pub(crate) struct DecodeResult {
    pub tokens: Vec<TokenInfo>,
    pub endpoint_detected: bool,
}

/// Copy frame `t` out of a frames-first encoder buffer
/// (`[T, ENCODER_OUT_DIM]` flat) into `enc_frame`.
pub(crate) fn extract_encoder_frame(
    encoded: &[f32],
    encoded_len: usize,
    t: usize,
    enc_frame: &mut [f32],
) {
    let dim = enc_frame.len();
    debug_assert!(t < encoded_len, "frame index out of range");
    let start = t * dim;
    enc_frame.copy_from_slice(&encoded[start..start + dim]);
}

/// Argmax over logits, returning the index of the largest value.
/// Returns `blank_id` for an empty slice so callers can treat the empty
/// case as "blank emitted, advance time".
pub(crate) fn argmax(logits: &[f32], blank_id: usize) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_i, a), (_j, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(blank_id)
}

/// Argmax with softmax confidence. The confidence is `exp(top - max) / Σ`,
/// equivalent to a numerically stable softmax max value.
pub(crate) fn argmax_with_confidence(logits: &[f32], blank_id: usize) -> (usize, f32) {
    if logits.is_empty() {
        return (blank_id, 0.0);
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let token = argmax(logits, blank_id);
    let confidence = (logits[token] - max_logit).exp() / sum_exp;
    (token, confidence)
}

/// Run the stateless decoder on the current context window.
/// Input: `decoder_input` `[1, CONTEXT_SIZE]` int64.
/// Output: `decoder_out` `[1, DECODER_OUT_DIM]` float32, returned owned.
fn run_decoder(decoder: &mut Session, context: &[i64]) -> Result<Vec<f32>> {
    debug_assert_eq!(context.len(), CONTEXT_SIZE);
    let input_tensor = TensorRef::from_array_view(([1_usize, CONTEXT_SIZE], context))?;
    let outputs = decoder
        .run(ort::inputs![input_tensor])
        .context("Decoder inference failed")?;
    let (_shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract decoder output")?;
    Ok(data.to_vec())
}

/// Run the joiner on a single (encoder_frame, decoder_frame) pair.
/// Inputs: encoder_out `[1, ENCODER_OUT_DIM]`, decoder_out `[1, DECODER_OUT_DIM]`.
/// Output: logits `[1, vocab_size]`, flattened to `Vec<f32>`.
fn run_joiner_single(
    joiner: &mut Session,
    enc_frame: &[f32],
    dec_data: &[f32],
) -> Result<Vec<f32>> {
    let enc_tensor = TensorRef::from_array_view(([1_usize, ENCODER_OUT_DIM], enc_frame))?;
    let dec_tensor = TensorRef::from_array_view(([1_usize, DECODER_OUT_DIM], dec_data))?;
    let outputs = joiner
        .run(ort::inputs![enc_tensor, dec_tensor])
        .context("Joiner inference failed")?;
    let (_shape, logits) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract joiner output")?;
    Ok(logits.to_vec())
}

/// RNN-T greedy decode over a frames-first encoder output buffer.
///
/// During blank runs the context window does not change, so we keep the
/// previous `decoder_out` around and skip the decoder call for every blank
/// frame — that is by far the dominant cost on long silences.
pub fn greedy_decode(
    decoder: &mut Session,
    joiner: &mut Session,
    encoded: &[f32], // [encoded_len, ENCODER_OUT_DIM] flat (frames-first)
    encoded_len: usize,
    blank_id: usize,
    state: &mut DecoderState,
) -> Result<DecodeResult> {
    anyhow::ensure!(
        encoded.len() >= ENCODER_OUT_DIM * encoded_len,
        "Encoder output size mismatch: got {}, expected >= {}",
        encoded.len(),
        ENCODER_OUT_DIM * encoded_len
    );

    let mut tokens = Vec::new();
    let mut endpoint_detected = false;
    let mut enc_frame = vec![0.0_f32; ENCODER_OUT_DIM];

    let mut decoder_calls: u32 = 0;
    let mut joiner_calls: u32 = 0;

    // Prime the decoder with the existing context. After the first call we
    // refresh `cached_decoder_out` only when we actually push a new token.
    let mut cached_decoder_out = run_decoder(decoder, &state.tokens)?;
    decoder_calls += 1;

    for t in 0..encoded_len {
        extract_encoder_frame(encoded, encoded_len, t, &mut enc_frame);
        let mut tokens_this_step = 0;

        loop {
            joiner_calls += 1;
            let logits = run_joiner_single(joiner, &enc_frame, &cached_decoder_out)?;
            let (token, confidence) = argmax_with_confidence(&logits, blank_id);

            if token == blank_id {
                state.consecutive_blanks += 1;
                if state.consecutive_blanks >= ENDPOINT_BLANK_THRESHOLD && !tokens.is_empty() {
                    endpoint_detected = true;
                }
                break;
            }

            if tokens_this_step >= MAX_TOKENS_PER_STEP {
                // Bail to avoid a runaway emission loop — let the next
                // frame deliver any further tokens. The cached decoder
                // output is stale (we already advanced the context this
                // inner loop), so do not reuse it.
                state.consecutive_blanks += 1;
                if state.consecutive_blanks >= ENDPOINT_BLANK_THRESHOLD && !tokens.is_empty() {
                    endpoint_detected = true;
                }
                break;
            }

            // Real token: slide the context window, refresh decoder out.
            state.consecutive_blanks = 0;
            state.push_token(token as i64);
            cached_decoder_out = run_decoder(decoder, &state.tokens)?;
            decoder_calls += 1;
            tokens.push(TokenInfo {
                token_id: token,
                frame_index: t,
                confidence,
            });
            tokens_this_step += 1;
        }
    }

    tracing::debug!(
        decoder_calls,
        joiner_calls,
        encoded_len,
        "decode_loop_stats"
    );
    Ok(DecodeResult {
        tokens,
        endpoint_detected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- extract_encoder_frame (frames-first layout) ---

    #[test]
    fn test_extract_encoder_frame_first() {
        // 3 frames, 2 dim. Frames-first: [f0d0, f0d1, f1d0, f1d1, f2d0, f2d1].
        let encoded = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut frame = vec![0.0; 2];
        extract_encoder_frame(&encoded, 3, 0, &mut frame);
        assert_eq!(frame, vec![1.0, 2.0]);
    }

    #[test]
    fn test_extract_encoder_frame_last() {
        let encoded = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut frame = vec![0.0; 2];
        extract_encoder_frame(&encoded, 3, 2, &mut frame);
        assert_eq!(frame, vec![5.0, 6.0]);
    }

    #[test]
    fn test_extract_encoder_frame_middle() {
        let encoded = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut frame = vec![0.0; 2];
        extract_encoder_frame(&encoded, 3, 1, &mut frame);
        assert_eq!(frame, vec![3.0, 4.0]);
    }

    // --- argmax / argmax_with_confidence ---

    #[test]
    fn test_argmax_clear_winner() {
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        assert_eq!(argmax(&logits, 999), 2);
    }

    #[test]
    fn test_argmax_tie_returns_last() {
        let logits = vec![1.0, 1.0, 0.5];
        assert_eq!(argmax(&logits, 999), 1);
    }

    #[test]
    fn test_argmax_negative_values() {
        let logits = vec![-3.0, -1.0, -2.0];
        assert_eq!(argmax(&logits, 999), 1);
    }

    #[test]
    fn test_argmax_empty_returns_blank() {
        let logits: Vec<f32> = vec![];
        assert_eq!(argmax(&logits, 1024), 1024);
    }

    #[test]
    fn test_argmax_blank_id_selected() {
        let logits = vec![0.1, 0.2, 0.9];
        assert_eq!(argmax(&logits, 2), 2);
    }

    #[test]
    fn test_confidence_picks_top_softmax_value() {
        // Logits where one entry dominates -> confidence close to 1.
        let logits = vec![0.0, 0.0, 100.0];
        let (id, conf) = argmax_with_confidence(&logits, 0);
        assert_eq!(id, 2);
        assert!(conf > 0.999, "expected near-1 confidence, got {conf}");
    }
}
