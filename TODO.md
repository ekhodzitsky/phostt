# phostt TODO

Tracker for remaining work. Current state:

- HEAD `master` (post-v0.4.2)
- 139 unit tests green, clippy clean (`-D warnings`)
- Runtime smoke test passes: `phostt download` + `phostt transcribe` on bundled WAVs

## Completed (recent)

- [x] Architecture refactor: split `inference/mod.rs` into `pool.rs` + `streaming.rs` + `engine.rs`
- [x] Split `server/mod.rs` — WebSocket handlers extracted to `server/ws/mod.rs`
- [x] Rename `GigasttError` → `PhosttError` with backward-compat alias
- [x] Deep health probe (`/health` returns `degraded` when pool saturated)
- [x] WebSocket frame size limits (`ws_frame_max_bytes`)
- [x] Request ID tracing (`x-request-id`)
- [x] OpenAPI/Swagger UI (`openapi` feature)
- [x] Graceful shutdown unit tests (Spec 005)
- [x] Pool slot recovery + metrics poison resilience (Spec 001)
- [x] REST handler fast unit tests (Spec 002)
- [x] E2E tests in CI on PRs with model caching
- [x] Docker hygiene + `docker-compose.yml`
- [x] Docs CI → GitHub Pages
- [x] ~~Redis rate limiter~~ — **removed** (use case is edge/desktop/mobile, not multi-instance)
- [x] GitHub issue templates (bug report + feature request)
- [x] `SECURITY.md`
- [x] README badges (downloads, docs.rs, release), table of contents, `docker compose` example
- [x] Python bindings (PyO3 + maturin) + PyPI CI workflow

## Open

### Blocked on external
- [x] `PyPI publish` — published v0.4.2
- [x] GHCR Docker images — automatic push on master

### Blocked on release
- [x] `Formula/phostt.rb`: updated to v0.4.2 URLs — `sha256` filled by `homebrew.yml`

### Runtime verification (needs GPU hardware)
- [x] Smoke test harness for CUDA / CoreML EP (`tests/ep_smoke.rs`) — **CoreML verified on M2 Pro**
- [ ] Confirm `--features cuda` / `--features coreml` link and run with Zipformer-vi tensor shapes
- [ ] Verify `RuntimeLimits::shutdown_drain_secs` semantics with the slower decoder loop

### Quality tracking
- [x] FLEURS Vietnamese WER benchmark (`tests/fleurs_wer.rs` + `scripts/prepare_fleurs_benchmark.py`)
- [x] Tune `MAX_MEAN_WER` threshold — **baseline: 1.0308 (103.08%) on 857 samples**

### Mobile / FFI
- [x] FFI streaming stress test (`tests/ffi_stress.rs`) — **verified: 2 workers × 5s, 0 errors**
- [ ] Android streaming FFI stress test on physical device
- [ ] iOS build verification (CoreML feature)
