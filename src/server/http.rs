//! HTTP handlers for REST API endpoints.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use serde::Serialize;
use std::sync::Arc;

use super::metrics::MetricsRegistry;
use super::{POOL_RETRY_AFTER_MS, POOL_RETRY_AFTER_SECS, RuntimeLimits};
use crate::inference::Engine;

/// Shared application state for all handlers. Carries runtime limits so the
/// WebSocket path can enforce configurable frame / idle bounds without
/// re-threading every CLI arg through each handler, plus an optional
/// in-tree `MetricsRegistry` backing the `/metrics` endpoint.
///
/// Also carries a shutdown `CancellationToken` and a `TaskTracker` used to
/// drain in-flight WebSocket / SSE tasks on SIGTERM (V1-03). `axum::serve`'s
/// built-in `with_graceful_shutdown` only tracks the HTTP router; upgraded
/// WebSocket handlers and `spawn_blocking` SSE tasks fall outside that lane
/// and must be drained explicitly.
pub struct AppState {
    pub engine: Arc<Engine>,
    pub limits: RuntimeLimits,
    pub metrics_registry: Option<Arc<MetricsRegistry>>,
    pub shutdown: tokio_util::sync::CancellationToken,
    pub tracker: tokio_util::task::TaskTracker,
}

/// GET /metrics — Prometheus text-format exposition. Returns 404 when the
/// server was started without `--metrics`.
pub async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    match &state.metrics_registry {
        Some(registry) => (
            StatusCode::OK,
            [(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            registry.render_prometheus(),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "metrics endpoint disabled",
                "code": "metrics_disabled",
            })),
        )
            .into_response(),
    }
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub version: String,
}

/// Model info response.
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub version: String,
    pub encoder: String,
    pub vocab_size: usize,
    pub sample_rate: u32,
    pub pool_size: usize,
    pub pool_available: usize,
    pub supported_formats: Vec<String>,
    pub supported_rates: Vec<u32>,
    /// Whether speaker diarization is available (feature-gated build + model loaded).
    /// Added in v0.7.0 so clients can probe capabilities via REST instead of
    /// opening a WebSocket just to read the `Ready` frame.
    pub diarization: bool,
}

/// Transcription response.
#[derive(Debug, Serialize)]
pub struct TranscribeResponse {
    pub text: String,
    pub words: Vec<crate::inference::WordInfo>,
    pub duration: f64,
}

/// Error response produced by the REST handlers. Using `Response` directly
/// (rather than a `(StatusCode, Json<_>)` tuple) lets timeout paths attach
/// a `Retry-After` header without changing the handler signatures.
type ApiError = Response;

fn api_error(status: StatusCode, msg: &str, code: &str) -> ApiError {
    (
        status,
        Json(serde_json::json!({"error": msg, "code": code})),
    )
        .into_response()
}

/// 503 response for pool-saturation backpressure: carries both the standard
/// `Retry-After` header (seconds, per RFC 9110 §10.2.3) and a machine-readable
/// `retry_after_ms` field in the JSON body so clients on either surface can
/// back off with the same hint.
fn api_timeout_error() -> ApiError {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, POOL_RETRY_AFTER_SECS.to_string())],
        Json(serde_json::json!({
            "error": "Server busy, try again later",
            "code": "timeout",
            "retry_after_ms": POOL_RETRY_AFTER_MS,
        })),
    )
        .into_response()
}

/// 503 response for the case where the pool was closed (graceful shutdown
/// in progress). Distinct from `timeout` so clients can decide whether to
/// retry: a closed pool is not coming back, so no `retry_after_ms` hint.
fn api_pool_closed_error() -> ApiError {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(serde_json::json!({
            "error": "Server is shutting down",
            "code": "pool_closed",
        })),
    )
        .into_response()
}

/// Checkout a session triplet from the engine pool with a 30-second timeout.
/// Returns the owned triplet and its reservation so it can be moved into
/// `spawn_blocking` and checked back in afterwards.
async fn checkout_triplet(
    engine: &std::sync::Arc<Engine>,
) -> Result<
    (
        crate::inference::SessionTriplet,
        crate::inference::OwnedReservation<crate::inference::SessionTriplet>,
    ),
    ApiError,
> {
    match tokio::time::timeout(std::time::Duration::from_secs(30), engine.pool.checkout()).await {
        Ok(Ok(guard)) => {
            let (triplet, reservation) = guard.into_owned();
            Ok((triplet, reservation))
        }
        Ok(Err(_pool_closed)) => Err(api_pool_closed_error()),
        Err(_timeout) => Err(api_timeout_error()),
    }
}

/// GET /health — health check for monitoring and Docker HEALTHCHECK.
pub async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let _ = &state.engine;
    Json(HealthResponse {
        status: "ok".into(),
        model: "zipformer-vi-rnnt".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

/// GET /v1/models — list loaded models and capabilities.
pub async fn models(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {
    let engine = &state.engine;
    #[cfg(feature = "diarization")]
    let diarization = engine.has_speaker_encoder();
    #[cfg(not(feature = "diarization"))]
    let diarization = false;
    Json(ModelInfo {
        id: "zipformer-vi-rnnt".into(),
        name: "Zipformer-vi RNN-T".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        encoder: "int8".into(),
        vocab_size: engine.vocab_size(),
        sample_rate: crate::inference::TARGET_SAMPLE_RATE,
        pool_size: engine.pool.total(),
        pool_available: engine.pool.available(),
        supported_formats: vec![
            "wav".into(),
            "mp3".into(),
            "m4a".into(),
            "ogg".into(),
            "flac".into(),
        ],
        supported_rates: super::SUPPORTED_RATES.to_vec(),
        diarization,
    })
}

/// POST /v1/transcribe — upload audio file, get full transcript.
///
/// Accepts raw audio body. Supported formats: WAV, MP3, M4A/AAC, OGG, FLAC.
/// Max body size enforced by the axum `DefaultBodyLimit` layer configured
/// from [`RuntimeLimits::body_limit_bytes`] (default 50 MiB).
pub async fn transcribe(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<TranscribeResponse>, ApiError> {
    if body.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        ));
    }

    // Defence-in-depth: `DefaultBodyLimit` already rejects oversized bodies
    // before they reach this handler, but a mis-ordered middleware stack or
    // a `Content-Length`-spoofing client could still deliver too many bytes.
    // The explicit 413 keeps the REST contract honest and gives clients a
    // machine-readable `payload_too_large` code alongside the spec-conformant
    // status. Cheap: `Bytes::len()` is a load, not a walk.
    if body.len() > state.limits.body_limit_bytes {
        return Err(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body exceeds the configured size limit",
            "payload_too_large",
        ));
    }

    let (triplet, reservation) = checkout_triplet(&state.engine).await?;

    let engine = state.engine.clone();

    let result = tokio::task::spawn_blocking(move || {
        let mut triplet = triplet;
        // catch_unwind ensures triplet is returned to pool even on panic
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // `body` is an `axum::body::Bytes` (re-export of `bytes::Bytes`):
            // `clone()` is a refcount bump, not a data copy, so the decode
            // path shares the original upload buffer.
            engine.transcribe_bytes_shared(body, &mut triplet)
        }));
        match r {
            Ok(inference_result) => (inference_result, triplet),
            Err(_) => {
                tracing::error!("Panic in REST transcribe — triplet recovered");
                (
                    Err(crate::error::PhosttError::Inference(
                        "Inference thread panicked".into(),
                    )),
                    triplet,
                )
            }
        }
    })
    .await;

    match result {
        Ok((Ok(result), triplet)) => {
            reservation.checkin(triplet);
            Ok(Json(TranscribeResponse {
                text: result.text,
                words: result.words,
                duration: result.duration_s,
            }))
        }
        Ok((Err(e), triplet)) => {
            reservation.checkin(triplet);
            tracing::error!("Transcription error: {e}");
            Err(api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Transcription failed. Check audio format.",
                "transcription_error",
            ))
        }
        Err(e) => {
            // spawn_blocking task itself failed (e.g., runtime shutdown).
            // Triplet is lost in this branch; reservation is dropped without
            // sending. The pool degrades by one slot.
            tracing::error!("spawn_blocking join error: {e}");
            Err(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error",
                "internal",
            ))
        }
    }
}

/// POST /v1/transcribe/stream — upload audio file, get SSE stream of partial/final results.
///
/// Real streaming: audio is processed chunk-by-chunk inside `spawn_blocking`,
/// and segments are sent to the SSE stream via an mpsc channel as they are produced.
pub async fn transcribe_stream(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, ApiError> {
    if body.is_empty() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "Empty request body",
            "empty_body",
        ));
    }

    // Defence-in-depth early reject; matches `/v1/transcribe` — see that
    // handler for the rationale.
    if body.len() > state.limits.body_limit_bytes {
        return Err(api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body exceeds the configured size limit",
            "payload_too_large",
        ));
    }

    let (triplet, reservation) = checkout_triplet(&state.engine).await?;

    // Create mpsc channel for streaming segments from spawn_blocking to SSE
    let (tx, rx) =
        tokio::sync::mpsc::channel::<Result<crate::inference::TranscriptSegment, String>>(16);

    let engine = state.engine.clone();
    // V1-03: the axum handler future has already returned by the time the
    // SSE stream starts flowing, so `with_graceful_shutdown` can't observe
    // this task. Clone the shutdown token and check it before every chunk
    // so SIGTERM during a long transcription drops cleanly.
    let cancel = state.shutdown.clone();
    let tracker = state.tracker.clone();
    tracker.spawn_blocking(move || {
        let mut triplet = triplet;

        // catch_unwind ensures triplet is returned to pool even on panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut stream_state = match engine.create_state(false) {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.blocking_send(Err(format!("{e}")));
                    return;
                }
            };
            let chunk_size = crate::inference::TARGET_SAMPLE_RATE as usize; // 1 second at 16kHz
            let mut chunk_buf: Vec<f32> = Vec::with_capacity(chunk_size);

            // Zero-copy streaming decode: symphonia feeds decoded samples
            // straight into the inference loop without a full Vec<f32> buffer.
            let decode_result = crate::inference::audio::decode_audio_streaming(body, |samples| {
                if cancel.is_cancelled() {
                    return Ok(());
                }
                chunk_buf.extend_from_slice(samples);
                while chunk_buf.len() >= chunk_size {
                    let chunk: Vec<f32> = chunk_buf.drain(..chunk_size).collect();
                    match engine.process_chunk(&chunk, &mut stream_state, &mut triplet) {
                        Ok(segs) => {
                            for seg in segs {
                                if tx.blocking_send(Ok(seg)).is_err() {
                                    // Receiver dropped (client disconnected)
                                    return Err(anyhow::anyhow!("receiver dropped"));
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.blocking_send(Err(format!("{e}")));
                            return Err(anyhow::anyhow!("inference failed"));
                        }
                    }
                }
                Ok(())
            });

            if let Err(e) = decode_result {
                tracing::error!("Streaming decode error: {e:#}");
                let _ = tx.blocking_send(Err(format!("{e}")));
                return;
            }

            // Process any trailing samples (< chunk_size)
            if !chunk_buf.is_empty() && !cancel.is_cancelled() {
                match engine.process_chunk(&chunk_buf, &mut stream_state, &mut triplet) {
                    Ok(segs) => {
                        for seg in segs {
                            if tx.blocking_send(Ok(seg)).is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("{e}")));
                        return;
                    }
                }
            }

            // Flush final segment — best-effort; skipped on cancel.
            if !cancel.is_cancelled()
                && let Some(seg) = engine.flush_state(&mut stream_state, &mut triplet)
            {
                let _ = tx.blocking_send(Ok(seg));
            }
        }));

        if result.is_err() {
            tracing::error!("Panic in SSE inference task — triplet recovered");
        }

        // Always return triplet to pool (even after panic). Sync `try_send`
        // is safe from a blocking thread; if the pool was closed in the
        // interim the triplet is silently dropped.
        reservation.checkin(triplet);
    });

    // Convert receiver to SSE stream
    let stream =
        tokio_stream::wrappers::ReceiverStream::new(rx).map(|result| Ok(segment_to_event(result)));

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// Map a transcript segment (or error) into an SSE event. Extracted so the
/// mapping logic can be unit-tested without spawning a stream or an engine.
fn segment_to_event(result: Result<crate::inference::TranscriptSegment, String>) -> Event {
    Event::default().data(segment_to_json_value(result).to_string())
}

/// Pure JSON mapping — the testable half of [`segment_to_event`].
fn segment_to_json_value(
    result: Result<crate::inference::TranscriptSegment, String>,
) -> serde_json::Value {
    match result {
        Ok(seg) => {
            if seg.is_final {
                serde_json::json!({"type": "final", "text": seg.text.as_ref(), "timestamp": seg.timestamp, "words": seg.words.as_ref()})
            } else {
                serde_json::json!({"type": "partial", "text": seg.text.as_ref(), "timestamp": seg.timestamp, "words": seg.words.as_ref()})
            }
        }
        Err(_) => {
            serde_json::json!({"type": "error", "message": "Transcription failed.", "code": "inference_error"})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{Engine, TranscriptSegment, WordInfo};
    use axum::body::to_bytes;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;
    use tokio_util::task::TaskTracker;

    fn test_state(limits: RuntimeLimits, metrics: Option<Arc<MetricsRegistry>>) -> Arc<AppState> {
        Arc::new(AppState {
            engine: Arc::new(Engine::test_stub()),
            limits,
            metrics_registry: metrics,
            shutdown: CancellationToken::new(),
            tracker: TaskTracker::new(),
        })
    }

    // -----------------------------------------------------------------------
    // 1. Empty body (both endpoints)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_transcribe_empty_body() {
        let state = test_state(RuntimeLimits::default(), None);
        let result = transcribe(State(state), Bytes::new()).await;
        assert!(result.is_err());
        let resp = result.unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "empty_body");
    }

    #[tokio::test]
    async fn test_stream_empty_body() {
        let state = test_state(RuntimeLimits::default(), None);
        let result = transcribe_stream(State(state), Bytes::new()).await;
        assert!(result.is_err());
        let resp = result.unwrap_err();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "empty_body");
    }

    // -----------------------------------------------------------------------
    // 2. Payload too large (both endpoints)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_transcribe_payload_too_large() {
        let limits = RuntimeLimits {
            body_limit_bytes: 10,
            ..RuntimeLimits::default()
        };
        let state = test_state(limits, None);
        let result = transcribe(State(state), Bytes::from(vec![0u8; 100])).await;
        assert!(result.is_err());
        let resp = result.unwrap_err();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "payload_too_large");
    }

    #[tokio::test]
    async fn test_stream_payload_too_large() {
        let limits = RuntimeLimits {
            body_limit_bytes: 10,
            ..RuntimeLimits::default()
        };
        let state = test_state(limits, None);
        let result = transcribe_stream(State(state), Bytes::from(vec![0u8; 100])).await;
        assert!(result.is_err());
        let resp = result.unwrap_err();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "payload_too_large");
    }

    // -----------------------------------------------------------------------
    // 3. Pool timeout (both endpoints) — empty pool, advance past 30 s
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_transcribe_pool_timeout() {
        tokio::time::pause();
        let state = test_state(RuntimeLimits::default(), None);
        let handle =
            tokio::spawn(async move { transcribe(State(state), Bytes::from(vec![1u8])).await });
        tokio::time::advance(std::time::Duration::from_secs(31)).await;
        let result = handle.await.unwrap();
        assert!(result.is_err());
        let resp = result.unwrap_err();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get(header::RETRY_AFTER).unwrap(),
            POOL_RETRY_AFTER_SECS.to_string().as_str()
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "timeout");
        assert_eq!(json["retry_after_ms"], POOL_RETRY_AFTER_MS);
    }

    #[tokio::test]
    async fn test_stream_pool_timeout() {
        tokio::time::pause();
        let state = test_state(RuntimeLimits::default(), None);
        let handle =
            tokio::spawn(
                async move { transcribe_stream(State(state), Bytes::from(vec![1u8])).await },
            );
        tokio::time::advance(std::time::Duration::from_secs(31)).await;
        let result = handle.await.unwrap();
        assert!(result.is_err());
        let resp = result.unwrap_err();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get(header::RETRY_AFTER).unwrap(),
            POOL_RETRY_AFTER_SECS.to_string().as_str()
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "timeout");
        assert_eq!(json["retry_after_ms"], POOL_RETRY_AFTER_MS);
    }

    // -----------------------------------------------------------------------
    // 4. Pool closed (both endpoints)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_transcribe_pool_closed() {
        let state = test_state(RuntimeLimits::default(), None);
        state.engine.pool.close();
        let result = transcribe(State(state), Bytes::from(vec![1u8])).await;
        assert!(result.is_err());
        let resp = result.unwrap_err();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "pool_closed");
        assert!(json.get("retry_after_ms").is_none());
    }

    #[tokio::test]
    async fn test_stream_pool_closed() {
        let state = test_state(RuntimeLimits::default(), None);
        state.engine.pool.close();
        let result = transcribe_stream(State(state), Bytes::from(vec![1u8])).await;
        assert!(result.is_err());
        let resp = result.unwrap_err();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "pool_closed");
        assert!(json.get("retry_after_ms").is_none());
    }

    // -----------------------------------------------------------------------
    // 5. Metrics endpoint
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_metrics_disabled() {
        let state = test_state(RuntimeLimits::default(), None);
        let resp = metrics(State(state)).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "metrics_disabled");
    }

    #[tokio::test]
    async fn test_metrics_enabled() {
        let registry = Arc::new(MetricsRegistry::new());
        registry.register_counter("requests_total", "Total requests");
        registry.counter_inc("requests_total", vec![], 1);
        let state = test_state(RuntimeLimits::default(), Some(registry));
        let resp = metrics(State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get(header::CONTENT_TYPE).unwrap();
        assert!(ct.to_str().unwrap().contains("text/plain"));
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("requests_total"));
    }

    // -----------------------------------------------------------------------
    // 6. SSE JSON mapping
    // -----------------------------------------------------------------------

    #[test]
    fn test_sse_partial_event() {
        let seg = TranscriptSegment {
            text: Arc::new("hello".into()),
            words: Arc::new(vec![]),
            is_final: false,
            timestamp: 1.5,
        };
        let json = segment_to_json_value(Ok(seg));
        assert_eq!(json["type"], "partial");
        assert_eq!(json["text"], "hello");
        assert_eq!(json["timestamp"], 1.5);
        assert!(json["words"].is_array());
    }

    #[test]
    fn test_sse_final_event() {
        let word = WordInfo {
            word: "world".into(),
            start: 0.0,
            end: 1.0,
            confidence: 0.95,
            speaker: None,
        };
        let seg = TranscriptSegment {
            text: Arc::new("world".into()),
            words: Arc::new(vec![word]),
            is_final: true,
            timestamp: 2.0,
        };
        let json = segment_to_json_value(Ok(seg));
        assert_eq!(json["type"], "final");
        assert_eq!(json["text"], "world");
        let words = json["words"].as_array().unwrap();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0]["word"], "world");
    }

    #[test]
    fn test_sse_error_event() {
        let json = segment_to_json_value(Err("boom".into()));
        assert_eq!(json["type"], "error");
        assert_eq!(json["code"], "inference_error");
    }
}
