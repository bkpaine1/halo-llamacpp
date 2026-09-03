# I tuned llama.cpp on a Strix Halo mini-PC and it beats Ollama on the same weights — here's the recipe (and why "Vulkan beats ROCm" is a myth)

I have an AMD Strix Halo box — a Ryzen AI Max+ 395, Radeon 8060S iGPU (gfx1151), 128 GB of unified memory. On paper it's a monster for local LLMs: that unified pool means the GPU can address ~100 GB of model weights with no discrete-VRAM ceiling. In practice, most people running one are leaving half the performance on the floor, and repeating two pieces of folklore that cost them the other half:

1. **"Just use Ollama / LM Studio / Lemonade."** Wrappers on wrappers on llama.cpp, each one adding a layer between you and the engine.
2. **"Vulkan is faster than ROCm on this APU."** You'll read this in every Strix Halo thread and in more than one benchmark writeup.

I got tired of both, built llama.cpp **directly** against ROCm/HIP with the gfx1151 tuning that actually matters, and measured it. Here's what I found.

## The finding that changes everything

**Stock HIP really is slow on gfx1151.** If you `cmake -DGGML_HIP=ON` and call it a day, your prompt processing is bad, and that's exactly where "Vulkan wins" comes from — those comparisons are testing *untuned* HIP.

**Tuned HIP wins.** Two levers do most of the work:

- **`ROCBLAS_USE_HIPBLASLT=1`** (a runtime env var). This routes rocBLAS matrix multiplies through **hipBLASLt**, which ships tuned gfx1151 kernels on ROCm 7.2. It is the single biggest prompt-processing win on this hardware — and almost nobody flips it. On a 7B, published measurements show pp512 jumping from ~349 t/s to ~986 t/s just from this plus flash-attention.
- **`-DGGML_HIP_ROCWMMA_FATTN=ON`** (a build flag). Flash-attention through rocWMMA, so throughput holds as your context deepens instead of falling off a cliff.

That's the whole secret. It's not exotic. It's a build flag and an env var that the ecosystem's wrappers hide from you.

## The setup

- **APU:** Ryzen AI Max+ 395 / Radeon 8060S / **gfx1151**, 128 GB unified
- **ROCm:** 7.2.4 (the community known-good for gfx1151; it ships native hipBLASLt gfx1151 kernels, so you do **not** need the old `HSA_OVERRIDE_GFX_VERSION=11.0.0` hack)
- **Kernel:** any recent one — every gfx1151 fix landed pre-6.18; nothing newer is required for the GPU path
- **llama.cpp:** pinned commit `1425386`

The build, in full:

```bash
export ROCM_PATH=/opt/rocm
export HIPCXX="$(hipconfig -l)/clang"
cmake -S llama.cpp -B llama.cpp/build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1151 \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j"$(nproc)"
```

`-DGGML_HIP_NO_VMM=ON` matters: HIP's virtual-memory manager is buggy on gfx1151 and will corrupt allocations if you leave it on. And on a 128 GB unified box, set your BIOS dedicated-VRAM to the minimum and let the driver use GTT — there's no performance difference between "VRAM" and system memory on this APU, and GTT is reclaimable.

## The numbers

All measured on the box, `ROCBLAS_USE_HIPBLASLT=1 llama-bench -m <model> -ngl 999 -fa 1`. `pp512` = prompt processing, `tg128` = token generation, both tokens/sec, higher is better.

| Model | Params | Quant | pp512 | tg128 |
|---|---|---|---|---|
| Qwen3-4B | 4 B dense | Q4_K | **1388** | 69.7 |
| Qwen3-30B-A3B | 30.5 B / 3 B active (MoE) | Q4_K_M | 843 | **73.7** |
| gpt-oss-20B | 20.9 B (MoE) | MXFP4 | 961 | 73.0 |

Read the middle row again: **a 30-billion-parameter model generating at ~74 tokens/sec on a mini-PC.** It's a MoE — only ~3 B params fire per token — so it actually *out-generates the 4 B dense model* despite carrying 7× the weights. On this hardware, MoE is the play.

And it holds at depth. Same 30B, with 4096 tokens already in the KV cache:

| | depth 0 | depth 4096 |
|---|---|---|
| pp512 | 843 | 209 |
| tg128 | 73.7 | **47.5** |

Still ~47 t/s of generation with real context loaded. That graceful long-context falloff is exactly where the tuned rocWMMA-FA + hipBLASLt path pulls ahead of Vulkan.

## The head-to-head that proves the point

Everyone defaults to Ollama. So I ran the **same GGUF blob** — `llama3.2:3b`, the exact file Ollama had already pulled — through both runtimes, both GPU-offloaded on the same box:

| runtime | generation (tg) |
|---|---|
| **tuned llama.cpp, direct** | **85.1 t/s** |
| Ollama | 82.1 t/s |

~4% faster on generation for a tiny 3B where both are already saturated — and the real gap is in prompt processing (1918 t/s direct on that blob). Strip the wrapper, tune the flags, and you beat the tool everyone reaches for, on identical weights. That's not a claim. That's the same file, two runtimes, one box.

## The honest part (because you'll ask)

- **Prompt-processing still has headroom.** I haven't done the `-ub` batch-size sweep or the rocWMMA-on-vs-off A/B yet. The pp numbers above should climb further, not lower.
- **Vulkan still wins one niche:** short-context token generation on mid-size *dense* models. If that's your entire workload, RADV is fine. Tuned HIP wins prefill, long context, big MoE, and multi-user serving.
- **I did not benchmark the 70B.** This box runs live services that hold the RAM, and a 42.5 GB dense model would have starved them. On an idle box a dense 70B is memory-bandwidth-bound at ~5-6 t/s — but I didn't measure that here, so I'm not printing a number I faked. MoE makes it moot anyway.

## No sidecar

The whole thing is a repo you can reproduce:

```bash
git clone https://github.com/bkpaine1/halo-llamacpp
cd halo-llamacpp
source env.sh          # ROCm env + ROCBLAS_USE_HIPBLASLT=1
./build.sh             # clone/pull llama.cpp @ the pinned commit, configure, build
./serve.sh model.gguf  # tuned llama-server: -ngl 999 -fa on -b 2048 -ub 512 --no-mmap
```

No Ollama, no container, no Python server in front. The binary *is* the engine. There's a `bench.sh` so you can reproduce every number above on your own box.

If you've got a Strix Halo, clone it, run `bench.sh`, and drop your numbers. I want the real cross-box table — not folklore.

---

*Repo: **[github.com/bkpaine1/halo-llamacpp](https://github.com/bkpaine1/halo-llamacpp)** · measured 2026 on Strix Halo gfx1151 / ROCm 7.2.4 / llama.cpp @ 1425386.*

---

*Built at **MindSpark Business Studio** — **[msbs.com](https://msbs.com)**. We ship real tools for people who do real work: **MultiScan** (multi-vendor network config capture, one exe, no install) and **StrixScreener** (AI resume screening that reads under the fluff). If this writeup was useful, come see what else we make.*
