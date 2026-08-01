# ASM-R 100M multiseed confirmation

This artifact consolidates the frozen continuous-validation results for ASM-R
(`J_NO_DIRECTION`) at eight training milestones and three independent seeds.

## Protocol

| Item | Value |
|---|---|
| Training corpus | Wikipedia byte-token memmap |
| Validation manifest | `data/benchmark_125m_wikipedia/validation/manifest.json` |
| Manifest SHA-256 | `4adabd5a6a64c30bda37ec23fd2db0341421995f6cdf516f46757e49c1948c07` |
| Validation targets | 4,834,787 per checkpoint |
| Seeds | 1, 2, 3 |
| Milestones | 1M, 2M, 5M, 10M, 20M, 30M, 50M, 100M tokens |
| Hardware | NVIDIA GeForce RTX 4090 |
| Precision | BF16 |

Every checkpoint was rescored over the same continuous validation sequence.
The result does not use the noisy sampled `best_val_ce` values from training.

## Result

| Tokens | Seed 1 CE | Seed 2 CE | Seed 3 CE | Mean CE | Population std |
|---:|---:|---:|---:|---:|---:|
| 1M | 2.231452 | 2.230333 | 2.235966 | 2.232584 | 0.002435 |
| 2M | 2.010348 | 2.009562 | 2.023646 | 2.014519 | 0.006462 |
| 5M | 1.750542 | 1.751412 | 1.750822 | 1.750925 | 0.000363 |
| 10M | 1.612410 | 1.611031 | 1.611041 | 1.611494 | 0.000648 |
| 20M | 1.513506 | 1.509661 | 1.511408 | 1.511525 | 0.001572 |
| 30M | 1.465237 | 1.465594 | 1.467070 | 1.465967 | 0.000794 |
| 50M | 1.413747 | 1.410157 | 1.410314 | 1.411406 | 0.001656 |
| 100M | 1.344849 | 1.343751 | 1.345016 | **1.344538** | **0.000561** |

At 100M, the corresponding perplexities are 3.8376, 3.8334, and 3.8382.
ASM-R completed all three runs without non-finite parameters.

ASM-F is not a valid 100M multiseed competitor in this generation. Its seed 2
and seed 3 runs diverged before 70M tokens, and both 100M checkpoints contain
only non-finite parameters. Seed 3 also produced non-finite validation CE from
the 30M checkpoint onward. A stabilized ASM-F rerun is classified as a
second-generation experiment.

Machine-readable values and source hashes are in [summary.json](summary.json).
The scientific interpretation is in
[report 037](../../report/037_Confirmacao_Multiseed_100M_e_Promocao_ASM_R_2026_08_01.md).
