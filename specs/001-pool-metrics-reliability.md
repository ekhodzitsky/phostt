# Spec 001: Pool Slot Recovery + Metrics Poison Resilience

## Problem Statement

The inference engine uses a pool of ONNX session triplets (`Pool<SessionTriplet>`).
Triplets are checked out for each streaming/WebSocket/SSE session and must be
returned to the pool when the session ends.

### Invariant P1: Triplet Conservation
**Every checked-out triplet MUST be returned to the pool exactly once.**

Violation: If a `spawn_blocking` task panics after `into_owned()` but before
the triplet is returned, the slot is lost forever. The pool gradually drains
until no sessions can be served (denial of service).

### Invariant P2: Metrics Resilience
**A panic in one HTTP handler MUST NOT break metrics collection for all
subsequent handlers.**

Violation: `MetricsRegistry` uses `std::sync::RwLock` with `.expect("... lock
poisoned")`. If any handler panics while holding the write lock, all future
metric increments panic, cascading a single handler failure into a global
denial of metrics.

## Current State (as of v0.2.2)

### Pool slot leak
- `src/server/mod.rs:handle_ws_inner` — triplet is `into_owned()` before
  `spawn_blocking`. The `catch_unwind` inside the closure recovers the triplet,
  but if the panic happens in symphonia decode (outside the closure), or if
  `catch_unwind` itself fails to capture, the triplet is lost.
- The comment says "degrades gracefully" but the degradation is unbounded.

### Metrics poison
- `src/server/metrics.rs` — every `record_*` method does `.write().expect(...)`.
- No recovery path after poison.

## Desired Behavior

### Pool
1. **RAII guard**: A drop guard inside the blocking task ensures the triplet
   is returned even if `catch_unwind` fails or is bypassed.
2. **Leak detection**: Log a warning if the pool reaches zero available slots
   and stays there for >30s (indicates a leak).
3. **Poisoned session recovery**: If `handle_ws_inner` detects a poisoned
   `state_opt` or `triplet_opt`, it should reset both instead of reusing.

### Metrics
1. **Poison recovery**: Use `RwLock::write().unwrap_or_else(|e| e.into_inner())`
   to recover the lock value after a poisoned thread.
2. **No panic on poison**: Metrics must never panic, even after 100% of
   previous holders panicked.

## Acceptance Criteria

- [ ] Unit test: simulate a panic during inference, assert triplet is returned.
- [ ] Unit test: simulate a panic during metric write, assert subsequent
      metric writes succeed.
- [ ] Unit test: pool reaches capacity, all slots recovered after panic storm.
- [ ] No `expect()` or `unwrap()` in pool return path or metrics write path.
- [ ] All 124 existing tests still pass.

## Design Decisions

### Pool recovery strategy
- **Chosen**: RAII `Reservation` guard with `Drop` impl that checks in the
  triplet. This works even if `catch_unwind` is bypassed or the thread is
  killed.
- **Rejected**: Manual `checkin` at every exit point — too many edge cases
  (early return, break, panic).

### Metrics recovery strategy
- **Chosen**: `parking_lot::RwLock` (never poisons) OR manual poison recovery.
  `parking_lot` is a small dependency with better performance.
- **Alternative**: `std::sync::RwLock` + `into_inner()` — no new deps but
  more verbose and error-prone.

## Test Plan

1. `test_pool_triplet_survives_panic` — checkout, panic in closure, assert
   pool size restored.
2. `test_metrics_survive_poison` — poison the lock, assert next write works.
3. `test_metrics_multiple_poison_cycles` — poison 10 times, assert metric
   value is still correct.

## References

- Audit finding #5: Pool slot leak
- Audit finding #11: MetricsRegistry panics on poisoned RwLock
- `src/server/mod.rs:638-710` (handle_ws)
- `src/server/metrics.rs:112-200` (MetricsRegistry)
