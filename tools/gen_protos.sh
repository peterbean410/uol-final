#!/usr/bin/env bash
# Generate the Python gRPC stubs for the modelenv environment contract.
#
# modelenv's Rust side compiles proto/environment.proto at build time via
# tonic_build. The Python side (deepqnetwork's environment client, and dqnpf
# through it) needs the equivalent generated stubs, which are NOT committed;
# they are build output. Run this once after installing requirements:
#
#     ./tools/gen_protos.sh
#
# It writes environment_pb2.py and environment_pb2_grpc.py into src/, which is
# the directory the packages are imported from.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$ROOT/src/modelenv/proto/proto"
OUT="$ROOT/src"

python3 -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  "$PROTO_DIR/environment.proto"

echo "generated:"
ls -1 "$OUT"/environment_pb2*.py | sed "s|$ROOT/|  |"
