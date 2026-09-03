#!/usr/bin/env bash
# serve.sh — launch llama-server GPU-native on Strix Halo (gfx1151), tuned defaults.
# No sidecar: the server IS llama.cpp compiled against ROCm.
#   ./serve.sh /path/to/model.gguf [extra llama-server args...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# shellcheck source=env.sh
source "$HERE/env.sh"

BIN="$HERE/llama.cpp/build/bin/llama-server"
[ -x "$BIN" ] || { echo "[serve] $BIN not found — run ./build.sh first"; exit 1; }

MODEL="${1:-}"
[ -n "$MODEL" ] || { echo "usage: ./serve.sh /path/to/model.gguf [args...]"; exit 1; }
[ -f "$MODEL" ] || { echo "[serve] model not found: $MODEL"; exit 1; }
shift || true

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

# -ngl 999 : all layers on GPU
# -fa on   : flash attention
# -b 2048  : logical batch
# -ub 512  : physical micro-batch
# --no-mmap: fully resident weights (steadier throughput on unified memory)
exec "$BIN" \
  -m "$MODEL" \
  --host "$HOST" --port "$PORT" \
  -ngl 999 \
  -fa on \
  -b 2048 -ub 512 \
  --no-mmap \
  "$@"
