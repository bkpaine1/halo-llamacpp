# env.sh — source this before build/bench/serve on THIS box (AI-Strix, gfx1151).
# Usage:  source env.sh
#
# Strix Halo = AMD RYZEN AI MAX+ 395 / Radeon 8060S, gfx1151, 128GB unified.
# ROCm 7.2.4 at /opt/rocm. hipBLASLt gfx1151 kernels are native-present, so we
# DO NOT set HSA_OVERRIDE_GFX_VERSION (overriding would fall back to slower paths).

export ROCM_PATH=/opt/rocm
export HIPCXX="$(hipconfig -l)/clang"
export HIP_PATH="$(hipconfig -R)"

# THE key runtime lever: route rocBLAS calls through hipBLASLt, which HAS tuned
# gfx1151 kernels. This is the single biggest throughput win on Strix Halo.
export ROCBLAS_USE_HIPBLASLT=1

# Make the GPU unambiguous (single dGPU/APU visible as device 0).
export HIP_VISIBLE_DEVICES=0

echo "[env] ROCM_PATH=$ROCM_PATH  HIPCXX=$HIPCXX  ROCBLAS_USE_HIPBLASLT=$ROCBLAS_USE_HIPBLASLT"
