# Llama-3.3-70B-Instruct Q4_K_M — NOT COMPLETED (honest note)

Model: Llama-3.3-70B-Instruct-Q4_K_M.gguf
Quant: Q4_K_M   File size: 42.5 GB (42520397472 bytes)
Command: ROCBLAS_USE_HIPBLASLT=1 llama-bench -m <70B> -ngl 999 -fa 1 -r 3 -o md

Result: NO NUMBER PRODUCED. Attempted twice on 2026-07-23.

- GPU init succeeded (gfx1151, VMM: no). Weights loaded to VRAM (climbed to 45-49 GB GTT).
- Then ran 41 minutes at GPU 99% / CPU 91% (STAT Rl) without emitting a single result row,
  and was killed to protect the box.

Root cause (measured, not guessed):
  This box was simultaneously running its own resident inference services:
    - a resident MoE server (gemma-4-26B-A4B, 128k ctx) on a local port
    - a vision sidecar on a local port
  Those pin ~23 GB of the 61 GB system partition and ~6.5 GB VRAM. Swap was already 8/8 GB full.
  The 42.5 GB dense model needs ~49 GB GTT AND keeps ~37 GB resident as mmap page-cache in the
  same unified 128 GB pool. System RAM hit 0 free -> swap thrash. That is why a model that
  should tg at the memory-bandwidth limit (~5-6 t/s) instead crawled for 40+ min at 99% GPU.

Conclusion: a 42.5 GB dense 70B does not co-reside with the running resident + vision services
in this box's unified memory without swapping. It was NOT benchmarked rather than kill those
services to free RAM. On an idle box it is expected to be bandwidth-bound (~5-6 t/s tg);
that idle-box number was not measured here.
