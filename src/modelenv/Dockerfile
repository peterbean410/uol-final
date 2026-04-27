# syntax=docker/dockerfile:1.7
# Multi-stage build for modelenv-server (Rust + tonic gRPC)

FROM rust:1.83-slim-bookworm AS builder

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        pkg-config \
        libssl-dev \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY modelenv ./modelenv

WORKDIR /build/modelenv
RUN cargo build --release --bin modelenv-server \
 && strip target/release/modelenv-server

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        awscli \
        ca-certificates \
        libssl3 \
        tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 1000 modelenv \
 && useradd --system --uid 1000 --gid 1000 --no-create-home modelenv

COPY --from=builder /build/modelenv/target/release/modelenv-server /usr/local/bin/modelenv-server

USER modelenv:modelenv
EXPOSE 50051
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/modelenv-server"]
