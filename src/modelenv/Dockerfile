# syntax=docker/dockerfile:1.7
# Multi-stage build for modelenv-server (Rust + tonic gRPC).
#
# The builder pins itself to the *build* platform (BUILDPLATFORM) and
# cross-compiles to the requested TARGETARCH, instead of running the whole Rust
# toolchain under QEMU emulation. Emulated cargo builds of a tonic/h2 server are
# far too slow/fragile for CI multi-arch (amd64 + arm64 for the Grace-Blackwell
# spark GPU nodes) and were timing out the image chain. The crate has no
# native-TLS / openssl / ring deps (plaintext in-pod gRPC on localhost), so a
# plain GNU cross toolchain is all that's needed; no cross-compiled C libs.
FROM --platform=$BUILDPLATFORM rust:1.85-slim-bookworm AS builder

# docker's TARGETARCH (amd64 | arm64) for the platform currently being built.
ARG TARGETARCH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        pkg-config \
        ca-certificates \
        gcc-aarch64-linux-gnu \
        binutils-aarch64-linux-gnu \
 && rm -rf /var/lib/apt/lists/*

# Map docker's TARGETARCH -> Rust target triple, install that target's std, and
# point cargo at the GNU cross linker for aarch64. protoc and proc-macros run on
# the native build host (BUILDPLATFORM), so only the emitted object code is
# cross-targeted, no QEMU.
RUN case "${TARGETARCH}" in \
        amd64) echo "x86_64-unknown-linux-gnu"  > /tmp/rust-target ;; \
        arm64) echo "aarch64-unknown-linux-gnu" > /tmp/rust-target ;; \
        *) echo "unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
      esac \
 && rustup target add "$(cat /tmp/rust-target)"

ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

WORKDIR /build
COPY modelenv ./modelenv

WORKDIR /build/modelenv
RUN TARGET="$(cat /tmp/rust-target)" \
 && cargo build --locked --release --target "${TARGET}" --bin modelenv-server \
 && cp "target/${TARGET}/release/modelenv-server" /tmp/modelenv-server \
 && case "${TARGETARCH}" in \
        amd64) strip /tmp/modelenv-server ;; \
        arm64) aarch64-linux-gnu-strip /tmp/modelenv-server ;; \
      esac

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

COPY --from=builder /tmp/modelenv-server /usr/local/bin/modelenv-server

USER modelenv:modelenv
EXPOSE 50051
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/modelenv-server"]
