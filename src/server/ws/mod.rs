//! WebSocket handlers for the phostt server.
//!
//! Extracted from `server/mod.rs` to keep the WebSocket lifecycle logic
//! (handshake, message loop, binary frame decode, configure, stop, flush)
//! in one place.

use crate::inference::{Engine, SessionTriplet};
use crate::protocol::{ClientMessage, ServerMessage};
use anyhow::Result;
use axum::extract::State;
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;

use super::http;
use super::{DEFAULT_SAMPLE_RATE, POOL_RETRY_AFTER_MS, RuntimeLimits, SUPPORTED_RATES, json_text};

pub fn ws_shutdown_response() -> Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    (
        StatusCode::SERVICE_UNAVAILABLE,
        axum::response::Json(serde_json::json!({
            "error": "Server shutting down",
            "code": "shutting_down",
        })),
    )
        .into_response()
}

/// Clamp `shutdown_drain_secs` to a minimum of 1 so the drain window
/// is never zero-length (which would immediately kill in-flight tasks).
/// fires in practice.
pub fn session_deadline_instant(max_session_secs: u64) -> tokio::time::Instant {
    tokio::time::Instant::now()
        + if max_session_secs == 0 {
            std::time::Duration::from_secs(u32::MAX as u64)
        } else {
            std::time::Duration::from_secs(max_session_secs)
        }
}

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<SocketAddr>,
    State(state): State<Arc<http::AppState>>,
) -> Response {
    // Origin allowlist is enforced by `origin_middleware` before the request
    // reaches this handler; anything that arrives here has already been cleared.
    //
    // V1-03: if shutdown has already been requested, refuse the upgrade
    // instead of handing the client a socket we're about to drain. Returning
    // a plain 503 with the `shutting_down` error code keeps the surface
    // consistent with the pool-saturation 503.
    if state.shutdown.is_cancelled() {
        tracing::warn!(peer = %peer, "Rejecting WS upgrade after shutdown");
        return ws_shutdown_response();
    }
    let max_bytes = state.limits.ws_frame_max_bytes;
    let state_cloned = state.clone();
    ws.max_message_size(max_bytes)
        .max_frame_size(max_bytes)
        .on_upgrade(move |socket| async move {
            // Track every upgraded handler on the shared TaskTracker so
            // `run_with_config` can wait for in-flight sessions to drain
            // before the process exits. `track_future` returns a wrapper
            // that decrements the tracker when dropped.
            state_cloned
                .tracker
                .clone()
                .track_future(handle_ws(socket, peer, state_cloned.clone()))
                .await
        })
}

/// Deprecated WebSocket endpoint at `/ws`. Identical behaviour to `/v1/ws`
/// but emits a warn-level log on every upgrade and adds RFC 8594 `Deprecation`
/// plus `Link: </v1/ws>; rel="successor-version"` headers on the upgrade
/// response so client libraries can surface the migration warning to users
/// before v1.0 drops the alias.
pub async fn ws_handler_legacy(
    ws: WebSocketUpgrade,
    axum::extract::ConnectInfo(peer): axum::extract::ConnectInfo<SocketAddr>,
    State(state): State<Arc<http::AppState>>,
) -> Response {
    tracing::warn!(
        peer = %peer,
        "WebSocket client connected to deprecated /ws path — switch to /v1/ws before v1.0"
    );
    if state.shutdown.is_cancelled() {
        return ws_shutdown_response();
    }
    let max_bytes = state.limits.ws_frame_max_bytes;
    let state_cloned = state.clone();
    let mut response = ws
        .max_message_size(max_bytes)
        .max_frame_size(max_bytes)
        .on_upgrade(move |socket| async move {
            state_cloned
                .tracker
                .clone()
                .track_future(handle_ws(socket, peer, state_cloned.clone()))
                .await
        });
    let headers = response.headers_mut();
    headers.insert("deprecation", axum::http::HeaderValue::from_static("true"));
    headers.insert(
        "link",
        axum::http::HeaderValue::from_static("</v1/ws>; rel=\"successor-version\""),
    );
    response
}

pub async fn handle_ws(socket: WebSocket, peer: SocketAddr, state: Arc<http::AppState>) {
    // `select!` the pool checkout against the shutdown token so SIGTERM
    // during pool saturation returns immediately instead of waiting the full
    // 30 s checkout window. `biased;` keeps cancel priority over progress.
    let guard = tokio::select! {
        biased;
        _ = state.shutdown.cancelled() => {
            tracing::info!(peer = %peer, "Shutdown requested before pool checkout");
            let (mut sink, _) = socket.split();
            let _ = sink.send(ws_close_message(1001, "server shutdown")).await;
            return;
        }
        res = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            state.engine.pool.checkout(),
        ) => match res {
            Ok(Ok(guard)) => guard,
            Ok(Err(_pool_closed)) => {
                tracing::info!("WebSocket pool closed for {peer} — server is shutting down");
                let (mut sink, _) = socket.split();
                let err = ServerMessage::Error {
                    message: "Server is shutting down".into(),
                    code: "pool_closed".into(),
                    retry_after_ms: None,
                };
                let _ = sink.send(WsMessage::Text(json_text(&err).into())).await;
                return;
            }
            Err(_) => {
                tracing::warn!("WebSocket pool checkout timeout for {peer}");
                let (mut sink, _) = socket.split();
                let err = ServerMessage::Error {
                    message: "Server busy, try again later".into(),
                    code: "timeout".into(),
                    retry_after_ms: Some(POOL_RETRY_AFTER_MS),
                };
                let _ = sink.send(WsMessage::Text(json_text(&err).into())).await;
                return;
            }
        }
    };

    // Strip the lifetime so the triplet can travel through `handle_ws_inner`,
    // which currently owns it directly. The reservation handles checkin on
    // the way back; if the inner loop loses the triplet to a spawn_blocking
    // panic, the reservation is dropped without sending and the pool
    // degrades gracefully — matches the pre-rewrite contract.
    let (triplet, reservation) = guard.into_owned();

    // Spec 001: background monitor that warns if the pool stays exhausted
    // for an extended period, indicating a possible slot leak.
    let pool_monitor_engine = state.engine.clone();
    let pool_monitor_cancel = state.shutdown.clone();
    tokio::spawn(async move {
        let mut last_nonzero = tokio::time::Instant::now();
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(5));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                _ = pool_monitor_cancel.cancelled() => break,
                _ = ticker.tick() => {
                    let available = pool_monitor_engine.pool.available();
                    let total = pool_monitor_engine.pool.total();
                    if available > 0 {
                        last_nonzero = tokio::time::Instant::now();
                    } else if total > 0 && last_nonzero.elapsed() > std::time::Duration::from_secs(30) {
                        tracing::warn!(
                            pool_total = total,
                            "Pool has been exhausted for >30s — possible slot leak"
                        );
                    }
                }
            }
        }
    });

    let (triplet_opt, result) = handle_ws_inner(
        socket,
        peer,
        &state.engine,
        &state.limits,
        triplet,
        state.shutdown.clone(),
    )
    .await;
    if let Err(e) = result {
        tracing::error!("WebSocket error from {peer}: {e}");
    }

    if let Some(triplet) = triplet_opt {
        reservation.checkin(triplet);
    }
    // If triplet_opt is None, the triplet was lost due to a spawn_blocking panic.
    // The pool degrades gracefully with fewer available slots.
}

/// Outcome returned by per-frame handlers. Keeps `handle_ws_inner` a thin
/// orchestration loop instead of a 250-line one-big-match.
pub enum FrameOutcome {
    /// Continue consuming frames.
    Continue,
    /// Clean break — client asked to stop (Stop message) or the socket closed.
    Break,
}

pub type WsSink = futures_util::stream::SplitSink<WebSocket, WsMessage>;

/// Build a WebSocket Close message with the given code and reason.
fn ws_close_message(code: u16, reason: &str) -> WsMessage {
    WsMessage::Close(Some(axum::extract::ws::CloseFrame {
        code,
        reason: reason.into(),
    }))
}

/// Decode a raw PCM16 binary frame into f32 samples, carrying an odd trailing
/// byte across frames so misaligned streams don't accumulate phase shift.
fn decode_pcm16_frame(data: &[u8], pending_byte: &mut Option<u8>, peer: SocketAddr) -> Vec<f32> {
    let carry_prev = pending_byte.take();
    let samples_f32: Vec<f32> = if carry_prev.is_some() || !data.len().is_multiple_of(2) {
        let mut combined = Vec::with_capacity(data.len() + 1);
        if let Some(prev) = carry_prev {
            combined.push(prev);
        }
        combined.extend_from_slice(data);
        if !combined.len().is_multiple_of(2) {
            tracing::warn!(
                "Odd-length PCM stream from {peer}: {} bytes incl. carry, deferring 1 byte",
                combined.len()
            );
            *pending_byte = combined.pop();
        }
        combined
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect()
    } else {
        data.chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
            .collect()
    };
    samples_f32
}

/// Send a serialized ServerMessage over the WebSocket sink. `?`-friendly so
/// handlers can delegate error propagation without duplicating the sink dance.
async fn send_server_message(sink: &mut WsSink, msg: &ServerMessage) -> Result<()> {
    sink.send(WsMessage::Text(json_text(msg).into()))
        .await
        .map_err(Into::into)
}

/// Handle a single PCM16 audio frame: resample if needed, run inference in a
/// `spawn_blocking` guarded by `catch_unwind`, and emit partial/final/error
/// payloads. Always returns the triplet to `triplet_opt` (or recovers a fresh
/// state after an inference panic) so the connection can keep serving.
#[allow(clippy::too_many_arguments)]
pub async fn handle_binary_frame(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
    triplet_opt: &mut Option<SessionTriplet>,
    audio_received: &mut bool,
    client_sample_rate: u32,
    pending_byte: &mut Option<u8>,
    peer: SocketAddr,
    data: axum::body::Bytes,
) -> Result<FrameOutcome> {
    if data.is_empty() {
        tracing::debug!("Empty binary frame from {peer}, skipping");
        return Ok(FrameOutcome::Continue);
    }
    *audio_received = true;

    let samples_f32 = decode_pcm16_frame(&data, pending_byte, peer);
    let samples_16k = if client_sample_rate == crate::inference::TARGET_SAMPLE_RATE {
        samples_f32
    } else {
        crate::inference::audio::resample(
            &samples_f32,
            client_sample_rate,
            crate::inference::TARGET_SAMPLE_RATE,
        )?
    };

    let state = state_opt
        .take()
        .ok_or_else(|| anyhow::anyhow!("Streaming state lost"))?;
    let triplet = triplet_opt.take().ok_or_else(|| {
        tracing::error!("Triplet unexpectedly missing for {peer}");
        anyhow::anyhow!("Triplet lost")
    })?;

    let eng = engine.clone();
    let join_result = tokio::task::spawn_blocking(move || {
        // Move ownership into the closure so state and triplet come back
        // unconditionally, including after a panic inside `process_chunk`.
        // Mirrors the pattern in src/server/http.rs.
        let mut state = state;
        let mut triplet = triplet;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eng.process_chunk(&samples_16k, &mut state, &mut triplet)
        }));
        (r, state, triplet)
    })
    .await;

    match join_result {
        Ok((Ok(Ok(segments)), state_back, triplet_back)) => {
            *state_opt = Some(state_back);
            *triplet_opt = Some(triplet_back);
            for seg in segments {
                let msg = if seg.is_final {
                    ServerMessage::Final {
                        text: seg.text.to_string(),
                        timestamp: seg.timestamp,
                        words: seg.words.to_vec(),
                    }
                } else {
                    ServerMessage::Partial {
                        text: seg.text.to_string(),
                        timestamp: seg.timestamp,
                        words: seg.words.to_vec(),
                    }
                };
                send_server_message(sink, &msg).await?;
            }
            Ok(FrameOutcome::Continue)
        }
        Ok((Ok(Err(e)), state_back, triplet_back)) => {
            *state_opt = Some(state_back);
            *triplet_opt = Some(triplet_back);
            tracing::error!("Inference error for {peer}: {e:#}");
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message: "Inference failed. Please check audio format.".into(),
                    code: "inference_error".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
            Ok(FrameOutcome::Continue)
        }
        Ok((Err(_panic), _state_back, triplet_back)) => {
            // Inference panicked: triplet is recovered, but the streaming
            // state (LSTM h/c buffers) may be mid-update and unsafe to reuse.
            // Drop it and install a fresh state so the session continues.
            tracing::error!(
                "Panic in WS inference for {peer} — triplet recovered, streaming state reset"
            );
            *triplet_opt = Some(triplet_back);
            match engine.create_state(false) {
                Ok(state) => *state_opt = Some(state),
                Err(e) => {
                    tracing::error!("Failed to create streaming state after panic: {e}");
                    // Session will error out on next frame
                }
            }
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message: "Inference failed unexpectedly. Session reset.".into(),
                    code: "inference_panic".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
            Ok(FrameOutcome::Continue)
        }
        Err(e) => {
            // spawn_blocking itself failed (runtime shutdown or cancellation).
            // Triplet is truly lost in this branch; bail out.
            tracing::error!("spawn_blocking join error for {peer}: {e}");
            Err(anyhow::anyhow!("Blocking task join failed"))
        }
    }
}

/// Handle `{"type":"configure",…}`. Rejects configure-after-first-audio,
/// validates sample rate against `SUPPORTED_RATES`, and (with diarization
/// feature) recreates the streaming state.
#[allow(clippy::too_many_arguments)]
async fn handle_configure_message(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
    client_sample_rate: &mut u32,
    audio_received: bool,
    sample_rate: Option<u32>,
    diarization: Option<bool>,
    peer: SocketAddr,
) -> Result<FrameOutcome> {
    if audio_received {
        send_server_message(
            sink,
            &ServerMessage::Error {
                message: "Configure must be sent before first audio frame".into(),
                code: "configure_too_late".into(),
                retry_after_ms: None,
            },
        )
        .await?;
        return Ok(FrameOutcome::Continue);
    }
    if let Some(rate) = sample_rate {
        if SUPPORTED_RATES.contains(&rate) {
            *client_sample_rate = rate;
            tracing::info!("Client {peer} configured sample rate: {rate}Hz");
        } else {
            send_server_message(
                sink,
                &ServerMessage::Error {
                    message: format!(
                        "Unsupported sample rate: {rate}Hz. Supported: {SUPPORTED_RATES:?}"
                    ),
                    code: "invalid_sample_rate".into(),
                    retry_after_ms: None,
                },
            )
            .await?;
        }
    }
    #[cfg(feature = "diarization")]
    if let Some(enable_dia) = diarization {
        tracing::info!("Client {peer} configured diarization: {enable_dia}");
        *state_opt = Some(
            engine
                .create_state(enable_dia)
                .map_err(|e| anyhow::anyhow!("State init failed: {e}"))?,
        );
    }
    #[cfg(not(feature = "diarization"))]
    {
        let _ = (engine, state_opt, diarization);
    }
    Ok(FrameOutcome::Continue)
}

/// Handle `{"type":"stop"}`. Flushes the streaming state, sends a final
/// segment (empty if there was nothing pending), and signals clean break.
async fn handle_stop_message(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
    triplet_opt: &mut Option<SessionTriplet>,
    peer: SocketAddr,
) -> Result<FrameOutcome> {
    tracing::info!("Stop received from {peer}, finalizing");
    let Some(mut state) = state_opt.take() else {
        return Ok(FrameOutcome::Break);
    };
    let Some(mut triplet) = triplet_opt.take() else {
        return Ok(FrameOutcome::Break);
    };
    let flush_seg = engine.flush_state(&mut state, &mut triplet);
    *triplet_opt = Some(triplet);
    let final_msg = if let Some(seg) = flush_seg {
        ServerMessage::Final {
            text: seg.text.to_string(),
            timestamp: seg.timestamp,
            words: seg.words.to_vec(),
        }
    } else {
        ServerMessage::Final {
            text: String::new(),
            timestamp: crate::inference::now_timestamp(),
            words: vec![],
        }
    };
    send_server_message(sink, &final_msg).await?;
    Ok(FrameOutcome::Break)
}

/// Flush any pending streaming state and emit a `Final` frame (even an empty
/// one) so e2e tests and clients can reliably assert that every session ends
/// with a Final before the Close. Used by the cancel and session-cap branches
/// of `handle_ws_inner`.
async fn flush_and_final(
    sink: &mut WsSink,
    engine: &Arc<Engine>,
    state_opt: &mut Option<crate::inference::StreamingState>,
    triplet_opt: &mut Option<SessionTriplet>,
) -> Result<()> {
    let Some(mut triplet) = triplet_opt.take() else {
        let final_msg = ServerMessage::Final {
            text: String::new(),
            timestamp: crate::inference::now_timestamp(),
            words: vec![],
        };
        return send_server_message(sink, &final_msg).await;
    };
    let flush_seg = state_opt
        .as_mut()
        .and_then(|state| engine.flush_state(state, &mut triplet));
    *triplet_opt = Some(triplet);
    let final_msg = match flush_seg {
        Some(seg) => ServerMessage::Final {
            text: seg.text.to_string(),
            timestamp: seg.timestamp,
            words: seg.words.to_vec(),
        },
        None => ServerMessage::Final {
            text: String::new(),
            timestamp: crate::inference::now_timestamp(),
            words: vec![],
        },
    };
    send_server_message(sink, &final_msg).await
}

/// Runs the WebSocket session loop. Always tries to return the triplet so the
/// caller can check it back into the pool. Returns `None` only if the triplet
/// was lost due to a thread panic inside `spawn_blocking`.
async fn handle_ws_inner(
    socket: WebSocket,
    peer: SocketAddr,
    engine: &Arc<Engine>,
    limits: &RuntimeLimits,
    triplet: SessionTriplet,
    cancel: tokio_util::sync::CancellationToken,
) -> (Option<SessionTriplet>, Result<()>) {
    let (mut sink, mut source) = socket.split();
    tracing::info!("Client connected: {peer}");

    #[cfg(feature = "diarization")]
    let diarization_available = engine.has_speaker_encoder();
    #[cfg(not(feature = "diarization"))]
    let diarization_available = false;

    let ready = ServerMessage::Ready {
        model: "zipformer-vi-rnnt".into(),
        sample_rate: DEFAULT_SAMPLE_RATE,
        version: crate::protocol::PROTOCOL_VERSION.into(),
        supported_rates: SUPPORTED_RATES.to_vec(),
        diarization: diarization_available,
    };
    if let Err(e) = send_server_message(&mut sink, &ready).await {
        return (Some(triplet), Err(e));
    }

    let mut state_opt = match engine.create_state(false) {
        Ok(state) => Some(state),
        Err(e) => {
            tracing::error!("State init failed: {e}");
            return (
                Some(triplet),
                Err(anyhow::anyhow!("State init failed: {e}")),
            );
        }
    };
    let mut triplet_opt = Some(triplet);
    let mut client_sample_rate: u32 = DEFAULT_SAMPLE_RATE;
    let mut audio_received = false;
    // V1-25: carries the trailing odd byte across PCM16 frames so clients
    // that split their streams on odd boundaries don't accumulate a
    // 1-sample phase shift in the decoded audio.
    let mut pending_byte: Option<u8> = None;

    let idle_timeout = std::time::Duration::from_secs(limits.idle_timeout_secs);

    // V1-04: wall-clock deadline independent of `idle_timeout`.
    let session_deadline = session_deadline_instant(limits.max_session_secs);

    // Async ASR pipeline: completed VAD utterances are off-loaded to a
    // background worker so the WebSocket loop can keep accepting audio
    // while offline ASR runs.
    let (asr_tx, mut asr_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<f32>>();
    let (final_tx, mut final_rx) =
        tokio::sync::mpsc::unbounded_channel::<crate::inference::TranscriptSegment>();

    let asr_engine = engine.clone();
    let _asr_handle = tokio::spawn(async move {
        while let Some(audio) = asr_rx.recv().await {
            let Ok(guard) = asr_engine.pool.checkout().await else {
                tracing::warn!("ASR worker: pool checkout failed, dropping utterance");
                continue;
            };
            let (mut triplet, reservation) = guard.into_owned();
            let eng = asr_engine.clone();
            let result = tokio::task::spawn_blocking(move || {
                let r = eng.transcribe_samples(&audio, &mut triplet);
                (r, triplet)
            })
            .await;
            match result {
                Ok((Ok(transcript), triplet_back)) if !transcript.text.is_empty() => {
                    reservation.checkin(triplet_back);
                    let seg = crate::inference::TranscriptSegment {
                        text: std::sync::Arc::new(transcript.text),
                        words: std::sync::Arc::new(transcript.words),
                        is_final: true,
                        timestamp: crate::inference::now_timestamp(),
                    };
                    let _ = final_tx.send(seg);
                }
                Ok((_, triplet_back)) => {
                    reservation.checkin(triplet_back);
                }
                Err(e) => {
                    tracing::warn!("ASR worker: spawn_blocking panicked: {e}");
                }
            }
        }
    });

    let result: Result<()> = loop {
        // Fast-path deadline / cancel check: if a client streams frames
        // continuously (e.g. 20 ms silence every 100 ms) the `source.next()`
        // arm is always ready when we re-enter `select!`, and with `biased;`
        // the runtime still polls cancel / sleep_until first — but only if
        // they have a registered waker. `sleep_until` registers its waker
        // correctly, yet a subtle race on fast CI runners can let the frame
        // arm fire before the timer's waker is installed. A cheap
        // pre-check here guarantees the deadline / cancel wins.
        if cancel.is_cancelled() {
            tracing::info!(peer = %peer, "Shutdown signalled — flushing WS session");
            let _ = flush_and_final(&mut sink, engine, &mut state_opt, &mut triplet_opt).await;
            let _ = sink.send(ws_close_message(1001, "server shutdown")).await;
            break Ok(());
        }
        if tokio::time::Instant::now() >= session_deadline {
            tracing::warn!(
                peer = %peer,
                max_session_secs = limits.max_session_secs,
                "Session cap reached — closing WS"
            );
            let _ = send_server_message(
                &mut sink,
                &ServerMessage::Error {
                    message: "Maximum session duration exceeded".into(),
                    code: "max_session_duration_exceeded".into(),
                    retry_after_ms: None,
                },
            )
            .await;
            let _ = flush_and_final(&mut sink, engine, &mut state_opt, &mut triplet_opt).await;
            let _ = sink
                .send(ws_close_message(1008, "max session duration"))
                .await;
            break Ok(());
        }

        tokio::select! {
            // `biased;` — cancel > deadline > frame. Guarantees that a
            // SIGTERM always wins a race against a pending frame, so the
            // drain path is deterministic.
            biased;

            _ = cancel.cancelled() => {
                tracing::info!(peer = %peer, "Shutdown signalled — flushing WS session");
                // Best-effort: the socket may already be dead if the peer
                // closed first, so every send is swallowed.
                let _ = flush_and_final(&mut sink, engine, &mut state_opt, &mut triplet_opt).await;
                let _ = sink.send(ws_close_message(1001, "server shutdown")).await;
                break Ok(());
            }

            _ = tokio::time::sleep_until(session_deadline) => {
                tracing::warn!(
                    peer = %peer,
                    max_session_secs = limits.max_session_secs,
                    "Session cap reached — closing WS"
                );
                let _ = send_server_message(
                    &mut sink,
                    &ServerMessage::Error {
                        message: "Maximum session duration exceeded".into(),
                        code: "max_session_duration_exceeded".into(),
                        retry_after_ms: None,
                    },
                )
                .await;
                let _ = flush_and_final(&mut sink, engine, &mut state_opt, &mut triplet_opt).await;
                let _ = sink.send(ws_close_message(1008, "max session duration")).await;
                break Ok(());
            }

            maybe_final = final_rx.recv() => {
                if let Some(seg) = maybe_final {
                    let msg = ServerMessage::Final {
                        text: seg.text.to_string(),
                        timestamp: seg.timestamp,
                        words: seg.words.to_vec(),
                    };
                    if let Err(e) = send_server_message(&mut sink, &msg).await {
                        return (triplet_opt, Err(e));
                    }
                }
            }

            maybe_msg = tokio::time::timeout(idle_timeout, source.next()) => {
                let msg = match maybe_msg {
                    Ok(Some(Ok(msg))) => msg,
                    Ok(Some(Err(e))) => break Err(e.into()),
                    Ok(None) => break Ok(()),
                    Err(_) => {
                        tracing::info!(
                            "Client {peer} idle timeout ({}s)",
                            limits.idle_timeout_secs
                        );
                        break Ok(());
                    }
                };

                let outcome = match msg {
                    WsMessage::Binary(data) => {
                        handle_binary_frame(
                            &mut sink,
                            engine,
                            &mut state_opt,
                            &mut triplet_opt,
                            &mut audio_received,
                            client_sample_rate,
                            &mut pending_byte,
                            peer,
                            data,
                        )
                        .await
                    }
                    WsMessage::Text(text) => match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(ClientMessage::Configure {
                            sample_rate,
                            diarization,
                        }) => {
                            handle_configure_message(
                                &mut sink,
                                engine,
                                &mut state_opt,
                                &mut client_sample_rate,
                                audio_received,
                                sample_rate,
                                diarization,
                                peer,
                            )
                            .await
                        }
                        Ok(ClientMessage::Stop) => {
                            handle_stop_message(&mut sink, engine, &mut state_opt, &mut triplet_opt, peer).await
                        }
                        Err(_) => {
                            tracing::debug!(
                                "Unrecognized text message from {peer}: {}",
                                &text[..text.len().min(100)]
                            );
                            Ok(FrameOutcome::Continue)
                        }
                    },
                    WsMessage::Close(_) => Ok(FrameOutcome::Break),
                    _ => Ok(FrameOutcome::Continue), // ignore ping/pong
                };

                match outcome {
                    Ok(FrameOutcome::Continue) => {
                        if let Some(ref mut state) = state_opt {
                            for audio in state.vad_pending_asr.drain(..) {
                                let _ = asr_tx.send(audio);
                            }
                        }
                        continue;
                    }
                    Ok(FrameOutcome::Break) => break Ok(()),
                    Err(e) => break Err(e),
                }
            }
        }
    };

    tracing::info!("Client disconnected: {peer}");
    (triplet_opt, result)
}
