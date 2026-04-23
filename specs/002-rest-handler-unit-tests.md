# Spec 002 — REST Handler Fast Unit Tests (Error Paths & SSE Mapping)

## Invariants

1. `empty_body` → 400 with `code: "empty_body"` for both `/v1/transcribe` and `/v1/transcribe/stream`.
2. `payload_too_large` → 413 with `code: "payload_too_large"` for both endpoints.
3. `pool_timeout` → 503 with `Retry-After` header and `retry_after_ms` body field for both endpoints.
4. `pool_closed` → 503 with `code: "pool_closed"` and no `retry_after_ms` for both endpoints.
5. `metrics_disabled` → 404 with `code: "metrics_disabled"`.
6. `metrics_enabled` → 200 with Prometheus text format.
7. SSE JSON mapping:
   - partial segment → `{"type": "partial", ...}`
   - final segment → `{"type": "final", ...}`
   - error → `{"type": "error", "code": "inference_error", ...}`

## Architecture

- `Tokenizer::from_tokens` — test-only constructor avoiding file I/O.
- `Engine::test_stub` — test-only engine with empty pool (no ONNX).
- `segment_to_json_value` — pure function extracted from SSE stream mapping.
- Tests call handlers directly with stub `AppState` (no axum router, no TCP).

## Tests

All 13 tests live in `src/server/http.rs` `#[cfg(test)] mod tests`:

- `test_transcribe_empty_body`
- `test_transcribe_payload_too_large`
- `test_transcribe_pool_timeout`
- `test_transcribe_pool_closed`
- `test_stream_empty_body`
- `test_stream_payload_too_large`
- `test_stream_pool_timeout`
- `test_stream_pool_closed`
- `test_metrics_disabled`
- `test_metrics_enabled`
- `test_sse_partial_event`
- `test_sse_final_event`
- `test_sse_error_event`

## Changes

- `Cargo.toml`: added `test-util` to tokio features (enables `tokio::time::pause` / `advance` in tests).
- `src/inference/tokenizer.rs`: added `#[cfg(test)] pub fn from_tokens`.
- `src/inference/mod.rs`: added `#[cfg(test)] pub fn test_stub` to `Engine`.
- `src/server/http.rs`:
  - Extracted `segment_to_event` + `segment_to_json_value`.
  - Added `#[derive(Debug)]` to `HealthResponse`, `ModelInfo`, `TranscribeResponse`.
  - Added 13 fast unit tests.
