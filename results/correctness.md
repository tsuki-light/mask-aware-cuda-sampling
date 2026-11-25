# Correctness results (reported local measurements)

Shared FP64 noise, real-model lockstep: 5 prompts × 3 seeds × 64 steps; 960 sampling calls; 128,448/128,448 total tokens and 62,243/62,243 active tokens matched. This compares both samplers on the same evolving CUDA V3 trajectory, not independent native-RNG generations.

| Distribution | Samples per implementation | PyTorch p | V3 p |
|---|---:|---:|---:|
| Synthetic, 8 categories | 1048576 | 0.078624 | 0.219283 |
| Real, low confidence | 32768 | 0.233322 | 0.020057 |
| Real, high confidence | 32768 | 0.563601 | 0.204979 |
| Real, mask competition | 32768 | 0.477870 | 0.064813 |
| Real, low confidence retest | 262144 | 0.571536 | 0.435334 |

The initial low-confidence result rejected the null at unadjusted alpha=0.05; the larger independent-seed retest did not. No zero-probability tokens were sampled in the real-distribution tests. Failure to reject does not establish equivalence. The grouped chi-square approximation may be imperfect for rare categories.

GSM8K smoke test (10 examples, zero-shot): flexible exact match 10% vs 0%, strict 0% vs 0%. Inconclusive; do not claim downstream quality parity. Timing from this run is not a controlled performance measurement.
