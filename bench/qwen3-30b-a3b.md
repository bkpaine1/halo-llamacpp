| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCm       | 999 |   1 |           pp512 |        843.25 ± 5.39 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCm       | 999 |   1 |           tg128 |         73.74 ± 0.17 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCm       | 999 |   1 |   pp512 @ d4096 |        209.17 ± 0.17 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | ROCm       | 999 |   1 |   tg128 @ d4096 |         47.48 ± 0.08 |

build: 1425386 (1)
