# Head-to-head: tuned llama.cpp (direct) vs Ollama runtime — SAME weights

Model: llama3.2:3b (Llama 3B, Q4_K_M), the exact same GGUF blob used by both runtimes:
  $OLLAMA_MODELS/blobs/sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff
Measured 2026-07-23 on Strix Halo gfx1151, ROCm 7.2.4, llama.cpp @ 1425386.
Both runtimes GPU-offloaded on the same box (live family services also running in both cases).

## Tuned llama.cpp direct (llama-bench, ROCBLAS_USE_HIPBLASLT=1 -ngl 999 -fa 1)
  pp512 : 1917.92 t/s
  tg128 :   85.07 t/s

## Ollama (ollama run llama3.2:3b --verbose, fixed prompt)
  prompt eval rate : 880.53 t/s   (only 38-token prompt — NOT comparable to pp512)
  eval rate (tg)   :  82.08 t/s   <-- the clean apples-to-apples generation number

## Takeaway
Generation (tg): tuned-direct 85.07 vs ollama 82.08 -> tuned build ~3.6% faster on a tiny
3B where both are already saturated. The prompt-processing gap is where the tuned hipBLASLt
path pulls ahead (pp512 = 1918 t/s direct), but Ollama's short 38-token prompt here does not
give a comparable pp512 figure, so only the tg comparison is reported as fair.
