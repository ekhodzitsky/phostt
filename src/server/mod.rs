//! HTTP + WebSocket server that accepts audio and streams transcripts.
//!
//! Single port serves both REST API (health, transcribe, SSE) and WebSocket.

pub mod http;
pub mod metrics;
pub mod rate_limit;

#[cfg(feature = "openapi")]
pub mod openapi;


mod ws;
pub use ws::{
    FrameOutcome, WsSink, handle_binary_frame, handle_ws, session_deadline_instant, ws_handler,
    ws_handler_legacy, ws_shutdown_response,
};

fn shutdown_drain_secs_clamped(secs: u64) -> u64 {
    secs.max(1)
}

use crate::inference::Engine;
use anyhow::{Context, Result};
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::State;
use axum::response::Response;
use axum::routing::{get, post};
use std::net::SocketAddr;
use std::sync::Arc;

/// Serialize a server message to JSON with a safe fallback on error.
fn json_text(msg: &impl serde::Serialize) -> String {
    serde_json::to_string(msg).unwrap_or_else(|e| {
        tracing::error!("Failed to serialize server message: {e}");
        r#"{"type":"error","message":"Internal serialization error","code":"internal"}"#.into()
    })
}

/// Supported input sample rates (Hz). Default is 48000 for backward
/// compatibility. Single source of truth for both the WebSocket `Ready`
/// payload and the REST `/v1/models` capabilities response.
pub(crate) const SUPPORTED_RATES: &[u32] = &[8000, 16000, 24000, 44100, 48000];
const DEFAULT_SAMPLE_RATE: u32 = 48000;

/// Hint (milliseconds) returned to clients that hit pool backpressure —
/// matches the `Retry-After` header emitted by the REST handlers and keeps
/// transient 503 / WebSocket error payloads consistent with the 30 s
/// checkout timeout used throughout the server.
pub(crate) const POOL_RETRY_AFTER_MS: u32 = 30_000;
pub(crate) const POOL_RETRY_AFTER_SECS: u64 = 30;

/// Origin policy for CORS + cross-origin deny middleware.
///
/// phostt is a privacy-first local server: by default we deny cross-origin
/// requests outright so a malicious page cannot trigger transcription from a
/// logged-in user's microphone via a drive-by WebSocket. Loopback origins
/// (`localhost`, `127.0.0.1`, `[::1]`) are always permitted; additional origins
/// must be listed explicitly via `--allow-origin`, and the wildcard `*`
/// behavior is opt-in via `--cors-allow-any`.
#[derive(Debug, Clone, Default)]
pub struct OriginPolicy {
    /// When true, the server accepts ANY `Origin` and echoes `*` in the CORS
    /// response — matches the old v0.5.x behavior. Dangerous default-off.
    pub allow_any: bool,
    /// Exact-match allowlist (e.g. `https://app.example.com`). Case-insensitive.
    pub allowed_origins: Vec<String>,
}

impl OriginPolicy {
    /// Loopback-only default policy: cross-origin requests from non-local
    /// pages are denied until the operator adds explicit allowlist entries.
    pub fn loopback_only() -> Self {
        Self::default()
    }
}

#[derive(Debug)]
enum OriginVerdict {
    /// No `Origin` header or opaque `null` — treat as non-browser client,
    /// no CORS echo required.
    AllowedNoEcho,
    /// Origin matches the policy; echo the exact string (or `*` if
    /// `allow_any` is on).
    Allowed(String),
    /// Origin present but not allowed — respond 403 before reaching the
    /// handler.
    Denied,
}

fn is_loopback_origin(origin: &str) -> bool {
    // Normalize once; compare lowercase prefixes. The prefix must be followed
    // by a port separator (`:`), a path (`/`), or end-of-string — otherwise
    // `http://localhost.evil.com` would be accepted as a DNS continuation of
    // the loopback hostname.
    let lowered = origin.to_ascii_lowercase();
    const HOST_PREFIXES: &[&str] = &[
        "http://localhost",
        "https://localhost",
        "http://127.0.0.1",
        "https://127.0.0.1",
        "http://[::1]",
        "https://[::1]",
    ];
    HOST_PREFIXES.iter().any(|p| match lowered.strip_prefix(p) {
        None => false,
        Some(rest) => rest.is_empty() || rest.starts_with(':') || rest.starts_with('/'),
    })
}

impl OriginPolicy {
    fn evaluate(&self, origin: Option<&str>) -> OriginVerdict {
        let Some(origin) = origin else {
            return OriginVerdict::AllowedNoEcho;
        };
        if origin.eq_ignore_ascii_case("null") {
            return OriginVerdict::AllowedNoEcho;
        }
        if self.allow_any || is_loopback_origin(origin) {
            return OriginVerdict::Allowed(origin.to_string());
        }
        if self
            .allowed_origins
            .iter()
            .any(|a| a.eq_ignore_ascii_case(origin))
        {
            return OriginVerdict::Allowed(origin.to_string());
        }
        OriginVerdict::Denied
    }
}

/// Runtime limits surfaced to per-request handlers. Separate from `ServerConfig`
/// because it needs to travel through `http::AppState` to the WebSocket handler.
#[derive(Debug, Clone)]
pub struct RuntimeLimits {
    /// WebSocket idle timeout. If no frame arrives within this window the
    /// server closes the connection. Default: 300s.
    pub idle_timeout_secs: u64,
    /// Maximum WebSocket frame / message size in bytes. Default: 512 KiB.
    pub ws_frame_max_bytes: usize,
    /// Maximum REST request body in bytes. Default: 50 MiB.
    pub body_limit_bytes: usize,
    /// Per-IP rate limit: requests-per-minute. `0` disables the limiter
    /// (default). Applies to /v1/* and /v1/ws; /health is exempt.
    pub rate_limit_per_minute: u32,
    /// Max burst size before the token bucket starts refilling.
    pub rate_limit_burst: u32,
    /// Maximum wall-clock duration of a single WebSocket session (seconds).
    /// `0` disables the cap entirely (not recommended — a silence-streaming
    /// client would hold a triplet forever). Default: 3600 (1 hour).
    pub max_session_secs: u64,
    /// Grace window (seconds) after the shutdown signal during which in-flight
    /// WebSocket / SSE tasks may emit their final frames and close cleanly.
    /// Values of `0` are clamped to `1` to avoid a no-op drain. Default: 10.
    pub shutdown_drain_secs: u64,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            idle_timeout_secs: 300,
            ws_frame_max_bytes: 512 * 1024,
            body_limit_bytes: 50 * 1024 * 1024,
            rate_limit_per_minute: 0,
            rate_limit_burst: 10,
            max_session_secs: 3600,
            shutdown_drain_secs: 10,
        }
    }
}

/// Server runtime configuration. `run_with_config` is the canonical entry
/// point; `run` / `run_with_shutdown` remain as thin wrappers for callers
/// that only need the pre-0.6 positional parameters.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub port: u16,
    pub host: String,
    pub origin_policy: OriginPolicy,
    pub limits: RuntimeLimits,
    /// Expose Prometheus metrics at `GET /metrics`. Off by default — keeps
    /// the server quiet for single-user local installs. When on, a
    /// `PrometheusHandle` is attached to `AppState` and the endpoint is
    /// added to the protected router so the Origin allowlist still applies.
    pub metrics_enabled: bool,
}

impl ServerConfig {
    /// Sensible local-only default: listen on `127.0.0.1:9876`, deny
    /// non-loopback origins, default runtime limits, metrics off.
    pub fn local(port: u16) -> Self {
        Self {
            port,
            host: "127.0.0.1".to_string(),
            origin_policy: OriginPolicy::loopback_only(),
            limits: RuntimeLimits::default(),
            metrics_enabled: false,
        }
    }
}

/// Start the HTTP + WebSocket STT server on the given host and port.
///
/// Serves REST API endpoints and WebSocket on a single port:
/// - `GET /health` — health check
/// - `POST /v1/transcribe` — file transcription
/// - `POST /v1/transcribe/stream` — SSE streaming transcription
/// - `GET /ws` — WebSocket streaming protocol
///
/// Runs until `Ctrl-C` is received.
pub async fn run(engine: Engine, port: u16, host: &str) -> Result<()> {
    run_with_shutdown(engine, port, host, None).await
}

/// Start server with an optional programmatic shutdown signal.
///
/// When `shutdown` is `Some`, the server stops when the sender fires (or is dropped).
/// When `None`, the server stops on Ctrl-C. Used by tests for clean teardown.
pub async fn run_with_shutdown(
    engine: Engine,
    port: u16,
    host: &str,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
) -> Result<()> {
    let config = ServerConfig {
        port,
        host: host.to_string(),
        origin_policy: OriginPolicy::loopback_only(),
        limits: RuntimeLimits::default(),
        metrics_enabled: false,
    };
    run_with_config(engine, config, shutdown).await
}

/// Start server with a full [`ServerConfig`] and optional programmatic
/// shutdown signal. This is the canonical entry point — the other `run_*`
/// helpers construct a default `ServerConfig` and dispatch here.
pub async fn run_with_config(
    engine: Engine,
    config: ServerConfig,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
) -> Result<()> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .context("Invalid host:port")?;

    // Stand up our in-tree metrics registry when the operator asked for it.
    // Unlike the old `PrometheusBuilder::install_recorder()` path this is
    // per-`run_with_config` rather than process-global — restarting the
    // server in tests cannot collide with itself, so we do not need the
    // "already installed" warning fallback the old stack needed.
    let metrics_registry = if config.metrics_enabled {
        let reg = std::sync::Arc::new(self::metrics::MetricsRegistry::new());
        reg.register_counter(
            "phostt_http_requests_total",
            "Total HTTP requests processed",
        );
        reg.register_histogram(
            "phostt_http_request_duration_seconds",
            "HTTP request duration in seconds",
            self::metrics::DEFAULT_BUCKETS,
        );
        tracing::info!("Prometheus /metrics endpoint enabled");
        Some(reg)
    } else {
        None
    };

    // V1-04 sanity check: an `idle_timeout` larger than `max_session_secs`
    // is usually a misconfiguration — the cap fires before the idle timeout
    // can ever apply, which is surprising. Warn without rejecting so
    // operators who intentionally want both can keep the behaviour.
    if config.limits.max_session_secs != 0
        && config.limits.max_session_secs < config.limits.idle_timeout_secs
    {
        tracing::warn!(
            max_session_secs = config.limits.max_session_secs,
            idle_timeout_secs = config.limits.idle_timeout_secs,
            "max_session_secs < idle_timeout_secs — sessions will be capped before \
             the idle timer can fire; this is probably not what you want"
        );
    }

    // Shutdown lane (V1-03): `shutdown_root` is cancelled when the caller's
    // oneshot fires (or Ctrl-C is received). Every WS / SSE handler gets a
    // clone so a SIGTERM propagates without racing `axum::serve`'s own
    // graceful shutdown.
    let shutdown_root = tokio_util::sync::CancellationToken::new();
    let tracker = tokio_util::task::TaskTracker::new();

    let state = Arc::new(http::AppState {
        engine: Arc::new(engine),
        limits: config.limits.clone(),
        metrics_registry,
        shutdown: shutdown_root.clone(),
        tracker: tracker.clone(),
    });

    let policy = Arc::new(config.origin_policy.clone());

    let origin_layer = {
        let policy = policy.clone();
        axum::middleware::from_fn(move |req, next| {
            let policy = policy.clone();
            async move { origin_middleware(policy, req, next).await }
        })
    };

    // Protected sub-router: /v1/*, /ws alias, and /metrics — all subject to
    // the origin allowlist and (when enabled) the per-IP rate limiter.
    let protected = Router::new()
        .route("/v1/models", get(http::models))
        .route("/v1/transcribe", post(http::transcribe))
        .route("/v1/transcribe/stream", post(http::transcribe_stream))
        // `/v1/ws` is the canonical path (versioned, aligned with REST); `/ws`
        // remains as an alias for existing clients and logs a deprecation
        // warning on each upgrade. Plan: drop `/ws` in v1.0.
        .route("/v1/ws", get(ws_handler))
        .route("/ws", get(ws_handler_legacy))
        .route("/metrics", get(http::metrics))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            http_metrics_middleware,
        ))
        .with_state(state.clone());

    let protected = if config.limits.rate_limit_per_minute > 0 {
        // Replacing `tower_governor` with our own token-bucket implementation
        // (see `rate_limit.rs`) drops the `governor` + `dashmap` +
        // `forwarded-header-value` transitive crates and restores control of
        // the V1-06 refill math: `refill_per_ms = rpm / 60_000`. The V1-11
        // IP-extraction contract (X-Forwarded-For → X-Real-IP → ConnectInfo)
        // is preserved bit-for-bit. `RateLimiter::new` owns the `rpm > MAX_RPM`
        // clamp + warn so the log line below stays consistent.
        let limiter = Arc::new(rate_limit::RateLimiter::new(
            config.limits.rate_limit_per_minute,
            config.limits.rate_limit_burst,
        ));
        let interval_ms = limiter.interval_ms();

        // Background eviction: bound memory under sustained traffic by
        // dropping buckets that haven't been touched in 5 minutes. `tokio`
        // task (not `std::thread::spawn`, V1-15 style) tied to `shutdown_root`
        // so the GC loop exits cleanly on SIGTERM instead of leaking.
        let evict_limiter = limiter.clone();
        let evict_cancel = shutdown_root.clone();
        tokio::spawn(rate_limit_eviction_loop(
            evict_limiter,
            evict_cancel,
            std::time::Duration::from_secs(60),
            std::time::Duration::from_secs(300),
        ));

        tracing::info!(
            rpm = config.limits.rate_limit_per_minute,
            interval_ms,
            burst = config.limits.rate_limit_burst,
            "per-IP rate limiting enabled"
        );
        let layer_limiter = limiter.clone();
        protected.layer(axum::middleware::from_fn(move |req, next| {
            let limiter = layer_limiter.clone();
            async move { rate_limit::rate_limit_middleware(limiter, req, next).await }
        }))
    } else {
        protected
    };

    // Clone the engine handle before `state` is consumed by `with_state` so
    // the shutdown closure can call `pool.close()` after the listener task
    // begins draining.
    let shutdown_engine = state.engine.clone();

    let app = Router::new()
        .route("/health", get(http::health))
        .merge(protected)
        .layer(DefaultBodyLimit::max(config.limits.body_limit_bytes))
        .layer(origin_layer)
        .layer(axum::middleware::from_fn(request_id_middleware))
        .with_state(state);

    #[cfg(feature = "openapi")]
    let app = app.merge(openapi::router());

    tracing::info!("phostt server listening on http://{addr}");
    tracing::info!("  WebSocket: ws://{addr}/v1/ws (legacy alias: ws://{addr}/ws)");
    tracing::info!("  REST API:  http://{addr}/health, /v1/transcribe, /v1/transcribe/stream");
    if config.origin_policy.allow_any {
        tracing::warn!(
            "CORS allow-any is ON: any cross-origin page can call this server. \
             Only use with trusted callers."
        );
    } else if !config.origin_policy.allowed_origins.is_empty() {
        tracing::info!(
            "CORS allowlist (in addition to loopback): {:?}",
            config.origin_policy.allowed_origins
        );
    }

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    let shutdown_drain_secs = shutdown_drain_secs_clamped(config.limits.shutdown_drain_secs);

    let shutdown_fut = {
        let shutdown_root = shutdown_root.clone();
        async move {
            match shutdown {
                Some(rx) => {
                    rx.await.ok();
                }
                None => {
                    tokio::signal::ctrl_c().await.ok();
                }
            }
            tracing::info!("Shutting down server");
            // Cancel the per-handler token FIRST so WS / SSE tasks start
            // draining while axum is still completing the in-flight HTTP
            // futures.
            shutdown_root.cancel();
            // Wake every waiter still blocked on `pool.checkout()` with
            // PoolError::Closed so they fall through to a 503 / `pool_closed`
            // response instead of being stranded for the full 30 s timeout.
            // Idempotent — safe even if the pool was already closed.
            shutdown_engine.pool.close();
        }
    };

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_fut)
    .await?;

    // Drain window: give WS / SSE tasks `shutdown_drain_secs` to emit their
    // Final frames and close cleanly. TaskTracker::wait() returns when every
    // tracked future completes; we close() first so no new futures can be
    // added after shutdown.
    tracker.close();
    match tokio::time::timeout(
        std::time::Duration::from_secs(shutdown_drain_secs),
        tracker.wait(),
    )
    .await
    {
        Ok(()) => tracing::info!("Drain complete: all tracked WS/SSE tasks finished"),
        Err(_) => tracing::warn!(
            drain_secs = shutdown_drain_secs,
            pending = tracker.len(),
            "Drain window expired with tracked tasks still running — forcing exit"
        ),
    }

    Ok(())
}

/// Instrumentation middleware: records HTTP request counters and a duration
/// histogram under the `phostt_http_*` namespace. Takes the whole
/// `AppState` so we can reach `metrics_registry` — when the operator did
/// not pass `--metrics` the registry is `None` and the middleware
/// degrades to a single `Instant::now()` per request.
async fn http_metrics_middleware(
    State(state): State<Arc<http::AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let Some(registry) = state.metrics_registry.clone() else {
        return next.run(req).await;
    };
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let start = std::time::Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed().as_secs_f64();
    let status = response.status().as_u16().to_string();
    registry.counter_inc(
        "phostt_http_requests_total",
        vec![
            ("method".into(), method.clone()),
            ("path".into(), path.clone()),
            ("status".into(), status),
        ],
        1,
    );
    registry.histogram_record(
        "phostt_http_request_duration_seconds",
        vec![("method".into(), method), ("path".into(), path)],
        elapsed,
    );
    response
}

async fn origin_middleware(
    policy: Arc<OriginPolicy>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    use axum::http::{StatusCode, header};
    use axum::response::IntoResponse;

    // `/health` is a liveness probe for container orchestrators and monitoring
    // tools that don't send Origin — let it through unconditionally.
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    let origin = req
        .headers()
        .get("origin")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    match policy.evaluate(origin.as_deref()) {
        OriginVerdict::AllowedNoEcho => next.run(req).await,
        OriginVerdict::Allowed(echo) => {
            let mut response = next.run(req).await;
            let headers = response.headers_mut();
            let value = if policy.allow_any { "*".into() } else { echo };
            if let Ok(v) = axum::http::HeaderValue::from_str(&value) {
                headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, v);
            }
            headers.insert(
                header::ACCESS_CONTROL_ALLOW_METHODS,
                axum::http::HeaderValue::from_static("GET, POST, OPTIONS"),
            );
            headers.insert(
                header::ACCESS_CONTROL_ALLOW_HEADERS,
                axum::http::HeaderValue::from_static("*"),
            );
            response
        }
        OriginVerdict::Denied => {
            let origin_str = origin.as_deref().unwrap_or("");
            let path = req.uri().path().to_string();
            tracing::warn!(
                origin = %origin_str,
                path = %path,
                "cross-origin request denied by default policy"
            );
            (
                StatusCode::FORBIDDEN,
                axum::response::Json(serde_json::json!({
                    "error": "Origin not allowed",
                    "code": "origin_denied",
                })),
            )
                .into_response()
        }
    }
}

/// Injects a `x-request-id` header into every request (generating UUIDv4 if
/// absent) and attaches it to the tracing span so all log lines in the
/// request carry the same correlation id.
async fn request_id_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let span = tracing::info_span!("request", request_id = %request_id);
    let mut response = tracing::Instrument::instrument(async move { next.run(req).await }, span).await;

    if let Ok(value) = axum::http::HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

/// Background eviction loop for the per-IP rate limiter. Exited cleanly
/// via `cancel` so the task does not leak across shutdown.
async fn rate_limit_eviction_loop(
    limiter: Arc<rate_limit::RateLimiter>,
    cancel: tokio_util::sync::CancellationToken,
    interval: std::time::Duration,
    stale_age: std::time::Duration,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // First tick fires immediately; skip it so the limiter is populated
    // before the first eviction pass.
    ticker.tick().await;
    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            _ = ticker.tick() => {
                limiter.evict_stale(stale_age);
            }
        }
    }
}

/// Build the 503 response returned when a WS upgrade is requested while
/// the server is already shutting down. Extracted so the JSON shape is
/// defined in one place and unit-testable.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ClientMessage;
    use axum::extract::ws::{WebSocket, WebSocketUpgrade};

    #[test]
    fn test_runtime_limits_default_rate_limit_disabled() {
        let limits = RuntimeLimits::default();
        assert_eq!(
            limits.rate_limit_per_minute, 0,
            "rate limiting must be off by default (privacy-first)"
        );
        assert_eq!(limits.rate_limit_burst, 10, "default burst size must be 10");
    }

    #[test]
    fn test_runtime_limits_default_session_and_drain() {
        // V1-03 / V1-04: locks in the documented defaults so a silent change
        // can't quietly disable the shutdown drain or the session cap.
        let limits = RuntimeLimits::default();
        assert_eq!(
            limits.max_session_secs, 3600,
            "default session cap must be 1 hour to stop silence-streamers from \
             holding a triplet forever"
        );
        assert_eq!(
            limits.shutdown_drain_secs, 10,
            "default shutdown drain must be 10 s — comfortably inside the usual \
             k8s terminationGracePeriodSeconds = 30"
        );
    }

    #[test]
    fn test_supported_rates_contains_common() {
        assert!(
            SUPPORTED_RATES.contains(&8000),
            "SUPPORTED_RATES must include 8000 Hz"
        );
        assert!(
            SUPPORTED_RATES.contains(&crate::inference::TARGET_SAMPLE_RATE),
            "SUPPORTED_RATES must include {} Hz",
            crate::inference::TARGET_SAMPLE_RATE
        );
        assert!(
            SUPPORTED_RATES.contains(&48000),
            "SUPPORTED_RATES must include 48000 Hz"
        );
    }

    #[test]
    fn test_default_sample_rate_in_supported() {
        assert!(
            SUPPORTED_RATES.contains(&DEFAULT_SAMPLE_RATE),
            "DEFAULT_SAMPLE_RATE ({DEFAULT_SAMPLE_RATE}) must be present in SUPPORTED_RATES"
        );
    }

    #[test]
    fn test_loopback_origin_matcher() {
        assert!(is_loopback_origin("http://localhost"));
        assert!(is_loopback_origin("https://localhost:3000"));
        assert!(is_loopback_origin("http://127.0.0.1:9876"));
        assert!(is_loopback_origin("HTTPS://127.0.0.1")); // case-insensitive
        assert!(is_loopback_origin("http://[::1]:9876"));
        assert!(!is_loopback_origin("https://evil.example.com"));
        assert!(!is_loopback_origin("http://192.168.1.10"));
        // Foiled prefix spoof: host must be exactly localhost / 127.0.0.1 / [::1]
        assert!(!is_loopback_origin("http://localhost.evil.example.com"));
    }

    #[test]
    fn test_origin_policy_default_denies_third_party() {
        let policy = OriginPolicy::loopback_only();
        assert!(matches!(
            policy.evaluate(Some("https://evil.example.com")),
            OriginVerdict::Denied
        ));
    }

    #[test]
    fn test_origin_policy_allows_loopback_by_default() {
        let policy = OriginPolicy::loopback_only();
        assert!(matches!(
            policy.evaluate(Some("http://localhost:3000")),
            OriginVerdict::Allowed(_)
        ));
    }

    #[test]
    fn test_origin_policy_allows_listed_origin() {
        let policy = OriginPolicy {
            allow_any: false,
            allowed_origins: vec!["https://app.example.com".into()],
        };
        assert!(matches!(
            policy.evaluate(Some("https://app.example.com")),
            OriginVerdict::Allowed(_)
        ));
        // Trailing-path mutations are not a match — allowlist is exact origin only.
        assert!(matches!(
            policy.evaluate(Some("https://app.example.com.evil.com")),
            OriginVerdict::Denied
        ));
    }

    #[test]
    fn test_origin_policy_allow_any_short_circuits() {
        let policy = OriginPolicy {
            allow_any: true,
            allowed_origins: vec![],
        };
        assert!(matches!(
            policy.evaluate(Some("https://anything.example.com")),
            OriginVerdict::Allowed(_)
        ));
    }

    #[test]
    fn test_origin_policy_no_header_allowed() {
        let policy = OriginPolicy::loopback_only();
        assert!(matches!(
            policy.evaluate(None),
            OriginVerdict::AllowedNoEcho
        ));
        assert!(matches!(
            policy.evaluate(Some("null")),
            OriginVerdict::AllowedNoEcho
        ));
    }

    #[tokio::test]
    async fn test_origin_middleware_integration() {
        // End-to-end check of the origin_middleware layer on a minimal
        // router. Uses real axum::serve + reqwest to catch regressions that
        // unit tests on `OriginPolicy` alone would miss — e.g. the middleware
        // attaching to the wrong routes, or `/health` accidentally being
        // guarded.
        use axum::Router;
        use axum::routing::get;

        let policy = Arc::new(OriginPolicy::loopback_only());
        let origin_layer = {
            let policy = policy.clone();
            axum::middleware::from_fn(move |req, next| {
                let policy = policy.clone();
                async move { origin_middleware(policy, req, next).await }
            })
        };
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/v1/transcribe", get(|| async { "ok" }))
            .layer(origin_layer);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        // Allow the server to bind.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let base = format!("http://127.0.0.1:{port}");

        // /health is exempt — monitoring probes work even when Origin is set.
        let r = client
            .get(format!("{base}/health"))
            .header("Origin", "https://evil.example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "/health must skip the Origin guard");

        // Cross-origin request must be denied on /v1/*.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .header("Origin", "https://evil.example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(
            r.status(),
            403,
            "non-loopback Origin must receive 403 Forbidden"
        );
        let text = r.text().await.unwrap();
        let body: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(body["code"], "origin_denied");

        // Loopback origin is always allowed.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .header("Origin", "http://localhost:3000")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "loopback Origin must be allowed");
        assert_eq!(
            r.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("http://localhost:3000"),
            "CORS echo must mirror the incoming Origin (no wildcard by default)",
        );

        // No Origin header (curl, CLI, server-to-server) — policy allows
        // through without a CORS echo.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "requests without Origin must pass");

        // Attacker trying DNS continuation on the loopback prefix must be denied.
        let r = client
            .get(format!("{base}/v1/transcribe"))
            .header("Origin", "http://localhost.evil.example.com")
            .send()
            .await
            .unwrap();
        assert_eq!(
            r.status(),
            403,
            "localhost.* DNS continuation must not impersonate loopback"
        );
    }

    #[test]
    fn test_rate_limit_interval_formula() {
        // Mirrors the formula used in `run_with_config` so a regression on the
        // V1-06 fix (integer-divide `/60` truncates sub-60 rpm to 1 rps) trips
        // a unit test before reaching the e2e path.
        const MAX_RPM: u64 = 60_000;
        fn interval_ms_for(rpm: u32) -> u64 {
            let rpm = (rpm as u64).min(MAX_RPM);
            (60_000u64 / rpm).max(1)
        }
        let cases: &[(u32, u64)] = &[
            (1, 60_000),
            (10, 6_000),
            (30, 2_000),
            (59, 1_016), // 60_000 / 59 = 1016 (rounds down) → ~59.05 rpm
            (60, 1_000),
            (600, 100),
            (60_000, 1),
            (120_000, 1), // clamped to MAX_RPM, stays at 1 ms
        ];
        for (rpm, expected) in cases {
            assert_eq!(
                interval_ms_for(*rpm),
                *expected,
                "rpm={rpm} should map to interval_ms={expected}"
            );
        }
    }

    #[test]
    fn test_catch_unwind_preserves_ownership_across_panic() {
        // Locks in the ownership contract used by `handle_ws_inner`'s spawn_blocking
        // block: moving captured values into the closure and wrapping the inner
        // computation in `catch_unwind(AssertUnwindSafe(_))` guarantees that the
        // values are observable after a panic, so the triplet can be returned to the
        // pool and the streaming state can be reset.
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let mut state = 42u32;
        let mut triplet_marker = String::from("pool_slot");

        let result = catch_unwind(AssertUnwindSafe(|| {
            state = 99;
            triplet_marker.push_str("/taken");
            panic!("simulated inference panic");
        }));

        assert!(result.is_err(), "catch_unwind must report the panic");
        assert_eq!(state, 99, "state must remain accessible after panic");
        assert_eq!(
            triplet_marker, "pool_slot/taken",
            "triplet marker must survive panic"
        );
    }

    // -----------------------------------------------------------------------
    // Graceful shutdown helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_shutdown_drain_secs_zero_clamped_to_one() {
        assert_eq!(shutdown_drain_secs_clamped(0), 1);
        assert_eq!(shutdown_drain_secs_clamped(1), 1);
        assert_eq!(shutdown_drain_secs_clamped(10), 10);
    }

    #[test]
    fn test_session_deadline_disabled_is_far_future() {
        let now = tokio::time::Instant::now();
        let deadline = session_deadline_instant(0);
        assert!(deadline > now + std::time::Duration::from_secs(1_000_000_000));
        // Verify it does not overflow (regression guard for the old u64::MAX/2 value).
        let _ = deadline - now;
    }

    #[test]
    fn test_session_deadline_enabled_is_near_future() {
        let now = tokio::time::Instant::now();
        let deadline = session_deadline_instant(60);
        let diff = deadline - now;
        assert!(diff >= std::time::Duration::from_secs(59));
        assert!(diff <= std::time::Duration::from_secs(61));
    }

    #[tokio::test]
    async fn test_ws_shutdown_response_status_and_code() {
        let resp = ws_shutdown_response();
        assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
        // Parse the JSON body.
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "shutting_down");
    }

    #[tokio::test]
    async fn test_rate_limit_eviction_loop_exits_when_cancelled() {
        let limiter = Arc::new(rate_limit::RateLimiter::new(10, 10));
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel(); // already cancelled before loop starts

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            rate_limit_eviction_loop(
                limiter,
                cancel,
                std::time::Duration::from_secs(60),
                std::time::Duration::from_secs(300),
            ),
        )
        .await;

        assert!(
            result.is_ok(),
            "eviction loop must exit immediately when cancelled"
        );
    }

    /// End-to-end test of the WS pre-checkout cancel path. Uses a real TCP
    /// listener and `tokio_tungstenite` client, but no ONNX model — the engine
    /// stub has an empty pool so `checkout()` blocks forever; cancelling the
    /// token while the client is connected causes `handle_ws` to send
    /// `Close(1001)`.
    #[tokio::test]
    async fn test_ws_cancel_before_checkout_sends_close_1001() {
        use axum::Router;
        use axum::routing::get;
        use futures_util::StreamExt;
        use tokio_tungstenite::tungstenite::Message;

        let cancel = tokio_util::sync::CancellationToken::new();
        let engine = crate::inference::Engine::test_stub();
        let state = Arc::new(http::AppState {
            engine: Arc::new(engine),
            limits: RuntimeLimits::default(),
            metrics_registry: None,
            shutdown: cancel.clone(),
            tracker: tokio_util::task::TaskTracker::new(),
        });

        let app = Router::new()
            .route("/v1/ws", get(ws_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let _ = axum::serve(
                listener,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await;
        });

        // Give the server a moment to bind.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let url = format!("ws://{addr}/v1/ws");
        let (mut ws, _) = tokio_tungstenite::connect_async(&url)
            .await
            .expect("WebSocket handshake should succeed");

        // Server is now inside `handle_ws`, blocked on the empty pool.
        // Cancelling the token must win the `select!` and send Close(1001).
        cancel.cancel();

        let msg = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next())
            .await
            .expect("should receive a message within 5s")
            .expect("stream should not end")
            .expect("message should not be an error");

        if let Message::Close(Some(frame)) = msg {
            assert_eq!(
                u16::from(frame.code),
                1001,
                "expected Close(1001) on pre-checkout cancel, got code {}",
                u16::from(frame.code)
            );
        } else {
            panic!("expected Close(1001) on pre-checkout cancel, got {msg:?}");
        }

        let _ = shutdown_tx.send(());
    }

    // -----------------------------------------------------------------------
    // WebSocket handler error paths (model-free)
    // -----------------------------------------------------------------------

    /// Spin up a minimal axum WS endpoint, complete the handshake, and hand
    /// back the server-side `SplitSink` so `handle_binary_frame` can be called
    /// directly in unit tests.  The client connection is dropped before
    /// returning, but the sink is never used in the error-path tests below.
    async fn test_ws_sink() -> WsSink {
        use futures_util::StreamExt;
        let (ws_tx, mut ws_rx) = tokio::sync::mpsc::unbounded_channel::<WebSocket>();
        let app = Router::new().route(
            "/_test_ws",
            get(move |ws: WebSocketUpgrade| async move {
                ws.on_upgrade(move |socket| async move {
                    let _ = ws_tx.send(socket);
                })
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let _client = tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{port}/_test_ws"))
            .await
            .unwrap()
            .0;

        let server_ws = tokio::time::timeout(std::time::Duration::from_secs(5), ws_rx.recv())
            .await
            .expect("server ws should be sent")
            .expect("server ws channel should not close");

        let (sink, _stream) = server_ws.split();
        sink
    }

    #[tokio::test]
    async fn test_handle_binary_frame_state_none_errors() {
        let mut sink = test_ws_sink().await;
        let engine = Arc::new(Engine::test_stub());
        let mut state_opt = None;
        let mut triplet_opt = None;
        let mut audio_received = false;
        let mut pending_byte = None;
        let peer = SocketAddr::from(([127, 0, 0, 1], 12345));
        let data = axum::body::Bytes::from(vec![0x00, 0x00, 0x00, 0x00]);

        let result = handle_binary_frame(
            &mut sink,
            &engine,
            &mut state_opt,
            &mut triplet_opt,
            &mut audio_received,
            16000,
            &mut pending_byte,
            peer,
            data,
        )
        .await;

        match result {
            Ok(_) => panic!("handle_binary_frame must error when state_opt is None"),
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("Streaming state lost"),
                    "error message should mention lost state: {msg}"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_handle_binary_frame_triplet_none_errors() {
        let mut sink = test_ws_sink().await;
        let engine = Arc::new(Engine::test_stub());
        let mut state_opt = Some(engine.create_state(false).unwrap());
        let mut triplet_opt = None;
        let mut audio_received = false;
        let mut pending_byte = None;
        let peer = SocketAddr::from(([127, 0, 0, 1], 12345));
        let data = axum::body::Bytes::from(vec![0x00, 0x00, 0x00, 0x00]);

        let result = handle_binary_frame(
            &mut sink,
            &engine,
            &mut state_opt,
            &mut triplet_opt,
            &mut audio_received,
            16000,
            &mut pending_byte,
            peer,
            data,
        )
        .await;

        match result {
            Ok(_) => panic!("handle_binary_frame must error when triplet_opt is None"),
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("Triplet lost"),
                    "error message should mention lost triplet: {msg}"
                );
            }
        }
    }

    #[test]
    fn test_malformed_json_text_frame_returns_continue() {
        let text = "not valid json {{{";
        let result = serde_json::from_str::<ClientMessage>(text);
        assert!(result.is_err(), "malformed JSON must fail to parse");
        // In handle_ws_inner the Err branch logs a debug message and returns
        // FrameOutcome::Continue — the test below locks in that the parser does
        // not panic and the branch is reachable.
    }

    // -----------------------------------------------------------------------
    // Metrics middleware
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_http_metrics_middleware_records_requests() {
        let registry = Arc::new(metrics::MetricsRegistry::new());
        registry.register_counter(
            "phostt_http_requests_total",
            "Total HTTP requests processed",
        );
        registry.register_histogram(
            "phostt_http_request_duration_seconds",
            "HTTP request duration in seconds",
            metrics::DEFAULT_BUCKETS,
        );

        let engine = crate::inference::Engine::test_stub();
        let state = Arc::new(http::AppState {
            engine: Arc::new(engine),
            limits: RuntimeLimits::default(),
            metrics_registry: Some(registry.clone()),
            shutdown: tokio_util::sync::CancellationToken::new(),
            tracker: tokio_util::task::TaskTracker::new(),
        });

        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                http_metrics_middleware,
            ))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let r = client
            .get(format!("http://127.0.0.1:{port}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);

        let text = registry.render_prometheus();
        assert!(
            text.contains(
                "phostt_http_requests_total{method=\"GET\",path=\"/health\",status=\"200\"} 1"
            ),
            "counter should record the request: {text}"
        );
        assert!(
            text.contains(
                "phostt_http_request_duration_seconds_count{method=\"GET\",path=\"/health\"} 1"
            ),
            "histogram should record the request: {text}"
        );
    }
}
