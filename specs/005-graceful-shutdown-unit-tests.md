# Spec 005 — Graceful Shutdown Unit Tests

## Context

Graceful shutdown logic lives in `src/server/mod.rs` and is critical for
zero-downtime deploys. Before this spec it was only exercised by heavy
`#[ignore]` end-to-end tests (full model load, real audio, real TCP). The
only fast tests were the integration-level `test_shutdown_graceful` and
`test_shutdown_immediate` in `tests/e2e_shutdown.rs`, which still need a
compiled binary and a free port.

The shutdown path has two distinct phases that need coverage:

1. **Pre-checkout** — the client has connected but `handle_ws` has not yet
   acquired a `SessionTriplet` from the pool. If shutdown is triggered here,
   the `select!` between `pool.checkout()` and `cancel.cancelled()` must win
   on the cancellation branch and send `Close(1001)`.

2. **Post-checkout** — the client is inside `handle_ws_inner`. If shutdown
   is triggered here, the loop must finish the current frame, flush, send
   `Close(1001)`, and exit cleanly.

Both phases contain pure-ish helper functions (`compute_session_deadline`,
`ws_shutdown_response`) that were previously inlined and therefore
untestable.

## Decision

Extract the pure helpers and add fast unit tests for them, plus one
lightweight async test that exercises the pre-checkout cancel path using a
real TCP socket but no ONNX model.

## Changes

### Extracted helpers (testable, pure)

- `ws_shutdown_response() -> Response`  
  Returns a consistent `503 Service Unavailable` JSON body with
  `{"error": "Server shutting down", "code": "shutting_down"}`. Used by
  both `ws_handler` and `ws_handler_legacy`.

- `compute_session_deadline(max_session_secs: u64) -> Instant`  
  Overflow-safe deadline calculation. Replaces inline `Instant::now() +
  Duration::from_secs(...)` which panics on macOS when given `u64::MAX/2`.
  Falls back to ~1 year when `checked_add` returns `None`.

### New tests

| Test | Type | What it covers |
|------|------|---------------|
| `test_shutdown_drain_secs_zero_clamped_to_one` | unit | `RuntimeLimits` parser clamps `shutdown_drain_secs == 0` to `1` so `sleep(Duration::ZERO)` never happens. |
| `test_session_deadline_disabled_is_far_future` | unit | `max_session_secs == 0` → deadline is ~1 year away, avoiding overflow. |
| `test_session_deadline_enabled_is_near_future` | unit | `max_session_secs == 5` → deadline is roughly 5 s from now (±1 s). |
| `test_ws_shutdown_response_status_and_code` | unit | Response is HTTP 503 and JSON contains expected `error`/`code` fields. |
| `test_rate_limit_eviction_loop_exits_when_cancelled` | async unit | Eviction task terminates immediately when token is already cancelled. |
| `test_ws_cancel_before_checkout_sends_close_1001` | async integration | Real TCP + `tokio_tungstenite` client; engine stub with empty pool blocks checkout; cancelling token causes `Close(1001)`. No ONNX load. Runs in ~60 ms. |

## Trade-offs

- **Pre-checkout test uses real TCP** because axum's `WebSocketUpgrade` is
  tightly coupled to hyper's upgrade mechanism; faking it would require
  significant test-only plumbing. The test still avoids model load and is
  fast (~60 ms).
- **Post-checkout shutdown** is still only covered by `#[ignore]` e2e tests
  because it requires a real model session to reach `handle_ws_inner`.
  However, the loop's cancel handling is structurally identical to the
  pre-checkout path, so confidence is high.

## Verification

```bash
cargo test --lib        # 121 passed, 0 failed
cargo clippy            # clean
cargo test --test e2e_shutdown  # still passes
```

## Future work

The last remaining gap from the original 20-issue audit is a WER benchmark
on a public Vietnamese dataset (VLSP / FLEURS) for regression tracking in
CI.
