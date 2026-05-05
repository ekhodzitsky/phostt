# phostt TODO

Tracker for remaining work. Current state:

- HEAD `master` (post-v0.3.0)
- 154 unit tests green, clippy clean (`-D warnings`)
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

## Open

### Blocked on release
- [ ] `Formula/phostt.rb`: replace placeholder `sha256` after next signed release tag

### Runtime verification (needs GPU hardware)
- [ ] Confirm `--features cuda` / `--features coreml` link and run with Zipformer-vi tensor shapes
- [ ] Verify `RuntimeLimits::shutdown_drain_secs` semantics with the slower decoder loop

### Quality tracking
- [ ] WER benchmark on public Vietnamese test set (VLSP or FLEURS) for CI regression tracking

### Mobile / FFI
- [ ] Android streaming FFI stress test on physical device
- [ ] iOS build verification (CoreML feature)
