| C | n | out tok/s | total tok/s | p50@TTFT, ms | mean@TTFT, ms | p95@TTFT, ms | p50@TPOT, ms | p50@ITL, ms | p99@ITL, ms | p50@E2EL, ms | std@E2EL, ms | p50 TTFT/E2EL, % | $/1M out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 87 | 434 | 86.1 | 86.1 | 89.6 | 11.21 | 11.22 | 11.72 | 2946 | 3.1 | 2.9 | 3.563 |
| 8 | 80 | 568 | 2839 | 484.9 | 433.6 | 538.4 | 12.28 | 12.05 | 13.31 | 3605 | 7.9 | 13.5 | 0.545 |
| 32 | 320 | 1346 | 6731 | 1334.2 | 1233.2 | 1968.4 | 18.55 | 15.91 | 41.01 | 6085 | 114.8 | 21.9 | 0.230 |

C=1: p50@TPOT = 11.21 ms; 1.39x spec floor 8.04 ms (72% of spec bandwidth); 1.20x measured-bandwidth floor 9.32 ms (83% of measured bandwidth)
