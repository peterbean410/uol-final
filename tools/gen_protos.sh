#!/usr/bin/env bash
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
