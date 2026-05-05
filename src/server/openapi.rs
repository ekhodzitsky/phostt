//! OpenAPI specification (requires the `openapi` feature).

use axum::{Json, Router, routing::get};
use utoipa::OpenApi;

/// Generate the OpenAPI JSON document.
pub fn router() -> Router {
    #[derive(utoipa::OpenApi)]
    #[openapi(
        info(title = "phostt", version = env!("CARGO_PKG_VERSION")),
        paths(
            super::http::health,
            super::http::models,
            super::http::transcribe,
            super::http::transcribe_stream,
        ),
        components(schemas(
            super::http::HealthResponse,
            super::http::ModelInfo,
            super::http::TranscribeResponse,
            crate::inference::TranscriptSegment,
            crate::inference::WordInfo,
        ))
    )]
    struct ApiDoc;

    let doc = ApiDoc::openapi();
    let swagger = utoipa_swagger_ui::SwaggerUi::new("/docs").url("/openapi.json", doc.clone());

    Router::new()
        .route("/openapi.json", get(move || async { Json(doc) }))
        .merge(swagger)
}
