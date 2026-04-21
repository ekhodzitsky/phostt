# phostt TODO

Tracker for the remaining work to ship phostt 0.1.0. Carries over from
the gigastt → phostt fork performed on 2026-04-21. Everything below
assumes the current state of `main`:

- HEAD `e7acabe` (feat(decode): Zipformer stateless rewire)
- All 129 unit tests green, clippy clean.
- Runtime smoke test passes: `phostt download` + `phostt transcribe` on
  the three bundled `test_wavs/*.wav` returns correct Vietnamese text
  with diacritics.

## Stage 2.5 — Streaming wrapper ✅

**Goal:** WebSocket / SSE streaming pumps audio through `kaldi_native_fbank::OnlineFeature` instead of recomputing FBANK from scratch on every chunk.

- [x] `src/inference/mod.rs::StreamingState`: replace the `audio_buffer: Vec<f32>` field with an `OnlineFeature` instance built from `phostt_fbank_options()`; expose helper to drain only the *newly ready* frames between calls.
- [x] `Engine::process_chunk`: feed `samples` straight to `online.accept_waveform(16000.0, samples)`, pull `online.num_frames_ready() - already_seen` frames, send them to the encoder.
- [x] Match the encoder's frame-batching expectations: Zipformer offline encoder needs the *whole* utterance, so wrap the streaming view in a sliding offline buffer with overlap-and-merge (start with 4 s window + 1 s overlap; tune later).
- [x] On endpointing or `Stop`/`Close`, call `online.input_finished()` so the trailing partial frame is flushed before final decode.
- [x] Decide what `total_frames` means now (legacy: encoder frames; new: encoder frames *after subsampling-by-4*) so timestamps in `WordInfo` stay correct.
- [x] New unit tests: streaming-vs-offline numerical equivalence on a 3-chunk split of the same audio buffer (small ε allowed because the overlap-merge path may pad differently).

## Stage 2.6 — End-to-end fixtures ✅

**Goal:** the `tests/e2e_*.rs` and `tests/soak_test.rs` suites run against the Zipformer-vi bundle, not the legacy GigaAM artefacts.

- [x] `tests/common/mod.rs::model_dir()` already checks for `encoder.int8.onnx`; add a parallel guard that the three `test_wavs/*.wav` exist (used by the fresh fixtures below). — `test_wavs_dir()` already validates all three WAVs.
- [x] Replace any leftover `gigaam` / Russian assumptions in `tests/e2e_rest.rs`, `tests/e2e_ws.rs`, `tests/e2e_errors.rs`, `tests/e2e_shutdown.rs`, `tests/e2e_rate_limit.rs` with the matching Vietnamese transcripts captured during smoke testing. — No legacy refs remain (verified via grep).
- [x] Re-add a `tests/benchmark.rs` harness — but for Vietnamese tone-aware WER, not the deleted Russian-number-to-words helpers. — Implemented as `tests/wer.rs` with Levenshtein WER against bundled Vietnamese test WAVs.
- [x] `tests/soak_test.rs`: cycle WS sessions against `test_wavs/0.wav` instead of generated tones so the encoder actually exercises real frames.

## Stage 3 — Documentation & release polish ✅

- [x] README.md: add the actual `phostt transcribe` smoke-test output and a one-line recap of measured latency (debug build: ~50 ms total on 3.7 s of audio on M1).
- [x] CLAUDE.md: drop the "Known TODO" block once the corresponding stages above land, replace with a "Streaming model" section explaining the offline-encoder + overlap-buffer trade-off.
- [ ] Formula/phostt.rb: replace the placeholder `sha256 "0000…"` lines with whatever the first signed release publishes (the `homebrew.yml` workflow already does this, but verify on the first tag). — **Blocked until first release is built.**
- [x] Cargo.toml: cull the no-op `quantize` feature flag once we are confident no external user is pinning it. — Already removed.
- [x] CONTRIBUTING.md: confirm the release runbook still matches reality after the inference rewrite (specifically: VERSION bump path, CHANGELOG layout).

## Stage 4 — Runtime hardening (post-0.1.0, optional)

Lower priority, surfaced during the rewrite — captured here so they are
not lost.

- [x] ~~Inspect & document the encoder ONNX input/output names via `ort::Session::inputs()/outputs()` so a future upstream re-export with renamed tensors does not silently break us. Add a `phostt inspect-onnx` debug subcommand.~~ Landed as `phostt inspect`. Confirmed encoder=`(x [N, T, 80] f32, x_lens [N] i64) -> (encoder_out [N, T', 512] f32, encoder_out_lens [N] i64)`, decoder=`(y [N, 2] i64) -> (decoder_out [N, 512] f32)`, joiner=`(encoder_out [N, 512], decoder_out [N, 512]) -> (logit [N, 2000] f32)`.
- [ ] Verify `RuntimeLimits::shutdown_drain_secs` semantics with the new (longer-loop) decoder; current default of 10 s should still be enough but exercise it with a deliberately slow encoder pool.
- [x] ~~Decide the long-term fate of `src/quantize.rs` — Zipformer-vi ships pre-quantized so the in-tree quantizer is dead weight on the user hot path. Either delete (and drop `prost`/`prost-build` + `proto/onnx.proto` + the `protoc` build dependency) or move under a `--features quantize` cfg gate and document it as a developer-only utility.~~ Already resolved: quantize feature removed, code cleaned up.
- [ ] Re-enable a real WER benchmark on a public Vietnamese test set (VLSP, FLEURS) to track regressions across model bumps.
- [ ] Confirm the `--features cuda` / `--features coreml` builds still link and run with the new tensor shapes (only CPU EP was exercised during the smoke test).

## Stage 5 — Release engineering 🚧 IN PROGRESS

- [ ] First `v0.1.0` tag. The release workflow
  (`.github/workflows/release.yml`) and Homebrew formula are already
  wired up but unverified end-to-end on the new repo.
- [ ] First `cargo publish` to crates.io as `phostt`. Check the name
  is still available and reserve it before the public announcement.
- [x] Decide whether `gigaam` / `gigastt` keywords should be added to
  the crate metadata for discoverability ("forked from"), or kept
  out to avoid confusion. — Decision: keep `gigastt` in keywords for
  discoverability (`keywords = ["stt", "asr", "vietnamese", "zipformer", "gigastt"]`);
  README and docs already explain the fork lineage clearly enough to avoid
  confusion.
