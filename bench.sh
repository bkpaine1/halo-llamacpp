#!/usr/bin/env bash
# bench.sh — real throughput benchmark on Strix Halo (gfx1151), GPU-native.
# Runs llama-bench over a model list, full GPU offload + flash-attn, no mmap.
# Results (jsonl + human table) land in bench/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# shellcheck source=env.sh
source "$HERE/env.sh"

BIN="$HERE/llama.cpp/build/bin/llama-bench"
[ -x "$BIN" ] || { echo "[bench] $BIN not found — run ./build.sh first"; exit 1; }

OUT="$HERE/bench"
mkdir -p "$OUT"
STAMP="$(date +%Y%m%d-%H%M%S)"

# ---------------------------------------------------------------------------
# MODELS: paths to .gguf files to benchmark.
# Set MODEL_DIR or edit this list for your installation.
MODEL_DIR="${MODEL_DIR:-/path/to/models}"
MODELS=(
  "$MODEL_DIR/Qwen3-4B-UD-Q4_K_XL.gguf"   # verified: pp512 1408 t/s, tg128 69 t/s
  # TODO add larger models to fill out the table:
  # "$MODEL_DIR/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
  # "$MODEL_DIR/Llama-3.3-70B-Instruct-Q4_K_M.gguf"
)
# ---------------------------------------------------------------------------

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "[bench] MODELS list is empty — edit bench.sh and add .gguf paths. (TODO)"
  exit 0
fi

for m in "${MODELS[@]}"; do
  [ -f "$m" ] || { echo "[bench] SKIP missing: $m"; continue; }
  name="$(basename "$m" .gguf)"
  jsonl="$OUT/${name}.${STAMP}.jsonl"
  echo "=== bench: $name ==="
  # -ngl 999 : offload all layers to GPU
  # -fa 1    : flash attention
  # --no-mmap: load weights fully into unified memory (better/steadier tg on Strix)
  "$BIN" -m "$m" -ngl 999 -fa 1 --no-mmap -o jsonl 2>>"$OUT/${name}.${STAMP}.err" \
    | tee "$jsonl"
  echo "[bench] wrote $jsonl"
done
