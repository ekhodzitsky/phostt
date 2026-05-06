# Multi-stage build for phostt
# Build: docker build -t phostt .
# Run:   docker run -p 9876:9876 phostt

# --- Builder stage ---
FROM rust:bookworm AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependency-compilation cache: copy manifests first and compile a dummy
# binary so `cargo build` downloads + builds every transitive crate.
# Subsequent edits to src/ only invalidate the final compilation layer,
# cutting incremental rebuild time from minutes to seconds.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src benches && \
    echo 'fn main() {}' > src/main.rs && \
    touch src/lib.rs && \
    touch benches/latency.rs && \
    cargo build --release && \
    rm -rf src benches target/release/deps/phostt-* target/release/phostt*

# Now bring in the actual source and build the real binary.
COPY src/ src/
COPY benches/ benches/

RUN cargo build --release && \
    strip target/release/phostt

# --- Model bake stage (runs only when PHOSTT_BAKE_MODEL=1) ---
FROM debian:bookworm-slim AS model-fetcher

ARG PHOSTT_BAKE_MODEL=0

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/phostt /usr/local/bin/phostt

RUN mkdir -p /models && \
    if [ "$PHOSTT_BAKE_MODEL" = "1" ]; then \
        phostt download --model-dir /models; \
    fi

# --- Runtime stage ---
FROM debian:bookworm-slim

ARG PHOSTT_BAKE_MODEL=0

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/phostt /usr/local/bin/phostt

RUN groupadd -r phostt && useradd -r -g phostt phostt && \
    mkdir -p /home/phostt/.phostt/models && chown -R phostt:phostt /home/phostt

# Copy baked model files (only present when PHOSTT_BAKE_MODEL=1)
COPY --from=model-fetcher --chown=phostt:phostt /models/. /home/phostt/.phostt/models/

USER phostt

ENV RUST_LOG=phostt=info

EXPOSE 9876

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:9876/health || exit 1

# Download model if not present, then start server.
# `--bind-all` acknowledges that container networking requires listening on
# 0.0.0.0; outside Docker the default `127.0.0.1` bind stays in effect.
ENTRYPOINT ["phostt"]
CMD ["serve", "--port", "9876", "--host", "0.0.0.0", "--bind-all"]
