FROM --platform=$BUILDPLATFORM rust:1.85-slim-bookworm AS builder

ARG TARGETARCH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        pkg-config \
        ca-certificates \
        gcc-aarch64-linux-gnu \
        binutils-aarch64-linux-gnu \
 && rm -rf /var/lib/apt/lists/*

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
