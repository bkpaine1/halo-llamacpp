# halo-llamacpp — max-throughput ROCm/HIP llama.cpp on Strix Halo

**No sidecar.** This is llama.cpp itself compiled GPU-native against ROCm/HIP for this
exact box — not an Ollama/vLLM wrapper, not a runtime shim. The binary *is* the engine.

## This box

| | |
|---|---|
| Host | `AI-Strix` |
| APU | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S |
| GPU arch | **gfx1151** (Strix Halo) |
| Memory | 128 GB unified |
| ROCm | 7.2.4 (`/opt/rocm`, hipconfig 7.2.53211) |
| Compiler | ROCm clang-22 (`/opt/rocm/lib/llvm/bin/clang`) |
| llama.cpp | pinned commit **`1425386fd996511e1f3295e7366c38289a92a271`** (ggml 0.17.0) |

## Quick start

```bash
source env.sh          # exports ROCM_PATH, HIPCXX/HIP_PATH, ROCBLAS_USE_HIPBLASLT=1
./build.sh             # idempotent: clone/pull @ pinned commit, configure, build
./bench.sh             # edit the MODELS list first (TODO)
./serve.sh /path/to/model.gguf
```

## Build flags — and why each one

```
-DGGML_HIP=ON                 # build the HIP/ROCm GPU backend (not CPU-only)
-DGPU_TARGETS=gfx1151         # compile kernels for THIS GPU only — smaller, faster build
-DGGML_HIP_ROCWMMA_FATTN=ON   # flash-attention via rocWMMA (headers present in /opt/rocm)
-DGGML_HIP_NO_VMM=ON          # disable HIP virtual-memory mgmt — it is BUGGY on gfx1151
                              #   (llama.cpp #20856). Leaving it on risks alloc corruption.
-DCMAKE_BUILD_TYPE=Release    # -O3
```

Deliberately **NOT** set:
- `HSA_OVERRIDE_GFX_VERSION` — the box has native hipBLASLt gfx1151 kernels in
  `/opt/rocm/lib/hipblaslt/library/`. Overriding the arch would push work onto a
  generic/slower codepath. Leave it unset.

### The key runtime lever

```bash
export ROCBLAS_USE_HIPBLASLT=1
```
Routes rocBLAS GEMMs through **hipBLASLt**, which ships tuned gfx1151 kernels on this
box. This is the single biggest throughput win on Strix Halo and is baked into `env.sh`.

### rocWMMA flash-attn note (A/B later)
There is a known rocWMMA-vs-plain-HIP flash-attn tradeoff on RDNA3.5. This build enables
rocWMMA FATTN. To build the plain-HIP fallback for comparison: `FATTN=0 ./build.sh`.

## Verify

```bash
build/bin/llama-cli --version
build/bin/llama-cli --list-devices      # must show a ROCm/HIP device = gfx1151
```

## Benchmarks — real numbers on this box

All measured **2026-07-23** on **Strix Halo gfx1151 / ROCm 7.2.4**, llama.cpp @ pinned
commit **`1425386`**, every run:

```bash
ROCBLAS_USE_HIPBLASLT=1  llama-bench  -m <model>  -ngl 999  -fa 1
```

`pp512` = prompt-processing throughput (512-token prefill), `tg128` = token-generation
throughput (128-token decode). Higher is better. Raw llama-bench output for each model is
in [`bench/`](bench/).

| Model | Params | Quant | File size | pp512 (t/s) | tg128 (t/s) |
|-------|-------:|-------|----------:|------------:|------------:|
| Qwen3-4B | 4.02 B (dense) | Q4_K_XL | 2.37 GiB | **1388.0 ± 26.2** | **69.7 ± 0.1** |
| Qwen3-30B-A3B-Instruct-2507 | 30.53 B / 3 B active (MoE) | Q4_K_M | 17.28 GiB | **843.3 ± 5.4** | **73.7 ± 0.2** |
| gpt-oss-20b | 20.91 B (MoE) | MXFP4 | 11.27 GiB | 961.2 ± 390.5 † | **73.0 ± 0.1** |
| Llama-3.3-70B-Instruct | 70 B (dense) | Q4_K_M | 42.5 GB | — ‡ | — ‡ |

† gpt-oss pp512 had huge run-to-run variance (±390) from GPU contention with the box's
live services — treat the *tg* number as the reliable one.

‡ **70B not completed — reported honestly, not faked.** The box was concurrently running the
live resident inference services (a `gemma-4-26B-A4B` MoE at 128k ctx + a vision sidecar) which
pin ~23 GB system RAM + ~6.5 GB VRAM, and swap was already full. The 42.5 GB dense model needs
~49 GB GTT *and* ~37 GB of mmap page-cache in the same unified 128 GB pool → system RAM hit
zero-free and the run swap-thrashed for **41 min at 99 % GPU / 91 % CPU without emitting a
single row**. It was killed rather than OOM the live services. On an idle box a 42.5 GB dense
70B is expected to be memory-bandwidth-bound at ~5-6 t/s tg; that idle number was **not**
measured here. See [`bench/llama-3.3-70b.md`](bench/llama-3.3-70b.md).

**Notable:** the 30B-A3B MoE *out-generates the 4B dense* (73.7 vs 69.7 tg) — only ~3 B params
are active per token, so decode is faster despite 7× the weights. Deep-context holds up too.

### Deep-context (the tuned-HIP story)

For the 30B-A3B MoE, throughput at a **4096-token** prefix depth (`-d 0,4096`):

| test | depth 0 | depth 4096 |
|------|--------:|-----------:|
| pp512 | 843.3 t/s | **209.2 t/s** |
| tg128 | 73.7 t/s | **47.5 t/s** |

Still ~47 t/s of generation with 4k of context already in the KV cache — this graceful
long-context falloff is where the tuned rocWMMA-FA + hipBLASLt HIP path earns its keep on
Strix Halo. See [`bench/qwen3-30b-a3b.md`](bench/qwen3-30b-a3b.md).

### The hipBLASLt lever

`ROCBLAS_USE_HIPBLASLT=1` (baked into `env.sh`) routes rocBLAS GEMMs through hipBLASLt's
tuned gfx1151 kernels. It is the single biggest **prompt-processing** win on this box — omit
it and pp throughput falls off a cliff. It is the one env var you must never drop.

### Tuned llama.cpp (direct) vs Ollama — same weights

Same `llama3.2:3b` GGUF blob run through both runtimes, both GPU-offloaded:

| runtime | tg (generation) |
|---------|----------------:|
| **tuned llama.cpp (direct, this repo)** | **85.1 t/s** |
| Ollama | 82.1 t/s |

The tuned direct build is ~4 % faster on generation for a tiny 3B where both are already
saturated; the tuned build's real lead is in prompt processing (pp512 = 1918 t/s direct on
that same blob), but Ollama's short benchmark prompt gives no comparable pp512 figure, so
only the fair *tg* comparison is shown. See
[`bench/ollama-headtohead-llama3.2-3b.md`](bench/ollama-headtohead-llama3.2-3b.md).

### Proof the offload is real (not CPU)

Every run prints:
```
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 65536 MiB):
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
```
`VMM: no` confirms `-DGGML_HIP_NO_VMM=ON` took effect.

## Repo layout

```
halo-llamacpp/
├── env.sh        # ROCm env + ROCBLAS_USE_HIPBLASLT=1 (source before anything)
├── build.sh      # idempotent clone/pull @ pinned commit + configure + build
├── bench.sh      # llama-bench over a model list -> bench/*.jsonl
├── serve.sh      # llama-server, tuned defaults (-ngl 999 -fa on -b 2048 -ub 512 --no-mmap)
├── README.md
├── .gitignore    # ignores llama.cpp/build/, *.gguf, bench/*.jsonl
└── llama.cpp/    # upstream checkout (build tree gitignored)
```
