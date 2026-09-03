#!/usr/bin/env bash
# build.sh — idempotent max-throughput ROCm/HIP llama.cpp build for Strix Halo (gfx1151).
# No sidecar: this compiles llama.cpp itself against ROCm, GPU-native.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# shellcheck source=env.sh
source "$HERE/env.sh"

REPO_URL="https://github.com/ggml-org/llama.cpp.git"
SRC="$HERE/llama.cpp"
# Pinned commit this recipe was verified against. Override: PIN_COMMIT=<sha> ./build.sh
PIN_COMMIT="${PIN_COMMIT:-1425386fd996511e1f3295e7366c38289a92a271}"

# 1. Get / update the checkout at the pinned commit.
if [ ! -d "$SRC/.git" ]; then
  echo "[build] cloning llama.cpp -> $SRC"
  git clone "$REPO_URL" "$SRC"
fi
cd "$SRC"
git fetch --depth 1 origin "$PIN_COMMIT" 2>/dev/null || git fetch origin
git checkout -q "$PIN_COMMIT"
echo "[build] llama.cpp @ $(git rev-parse HEAD)"

# 2. Optional ccache (skipped automatically if not installed).
CCACHE_FLAGS=()
if command -v ccache >/dev/null 2>&1; then
  echo "[build] ccache found -> enabling"
  CCACHE_FLAGS=(-DGGML_CCACHE=ON
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi

# 3. Configure.
#   GGML_HIP                 -> build the HIP/ROCm GPU backend (not CPU-only)
#   GPU_TARGETS=gfx1151      -> compile kernels for THIS GPU only (Strix Halo)
#   GGML_HIP_ROCWMMA_FATTN   -> flash-attn via rocWMMA (headers present). If this
#                               ever fails to compile, rerun with FATTN=0 ./build.sh
#   GGML_HIP_NO_VMM          -> disable HIP virtual-memory mgmt; it is buggy on
#                               gfx1151 (llama.cpp #20856). Keep ON.
FATTN="${FATTN:-1}"
FATTN_FLAG=(-DGGML_HIP_ROCWMMA_FATTN=ON)
[ "$FATTN" = "0" ] && FATTN_FLAG=() && echo "[build] rocWMMA FATTN DISABLED (fallback build)"

cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1151 \
  "${FATTN_FLAG[@]}" \
  -DGGML_HIP_NO_VMM=ON \
  -DCMAKE_BUILD_TYPE=Release \
  "${CCACHE_FLAGS[@]}"

# 4. Build.
cmake --build build --config Release -j"$(nproc)"

echo "[build] DONE. Binaries in $SRC/build/bin/"
"$SRC/build/bin/llama-cli" --version 2>&1 | head -3 || true
