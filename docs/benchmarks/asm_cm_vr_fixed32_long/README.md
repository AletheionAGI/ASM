# ASM-CM-VR long multiseed curriculum

## Status

The requested full-64, fixed-32, and exploratory adaptive-32 arms completed for seeds 17, 29, and 43. The fixed-32 promotion gate **failed**. This benchmark measures logical rank with dense storage and kernels; it does not demonstrate physical speedup or memory reduction.

## Protocol

- Curriculum lengths: 40, 80, 160, 320, 512, 1K, and 4K.
- Frozen held-out tests: 40, 512, 4K, and 32K.
- Seeds: 17, 29, and 43.
- Promotion requires fixed-32 accuracy of at least 80% at every test length, finite 32K streaming, and full/stream parity at most `1e-4` in every seed.
- Adaptive-32 is exploratory and cannot change the fixed-32 promotion decision.

## Main results

| Arm | MQAR-40 | MQAR-512 | MQAR-4K | MQAR-32K | Passed seeds |
|---|---:|---:|---:|---:|---:|
| full-64 | 99.84% | 67.22% | 67.19% | 66.67% | 2/3 |
| fixed-32 | 99.82% | 65.85% | 66.67% | 66.67% | 2/3 |
| adaptive-32 | 98.37% | 98.24% | 97.40% | 72.92% | 2/3 |

The full/fixed averages include the failed seed-29 outcome as zero at 32K. Among the two finite successful seeds, both full-64 and fixed-32 reached 100% at 32K.

Adaptive-32 varied from rank 16 to 64. Its mean held-out rank rose from `17.51` at MQAR-40 to `36.91` at 32K. The controller received gradients in all `7,725/7,725` checked updates. Its 32K accuracy was seed-dependent: 46.88%, 81.25%, and 90.63%.

## Recorded failures

- full-64 seed 29: 1.76% at 512, 1.56% at 4K, non-finite CE at 32K, and streaming failed at token 14,175 with `effective_coordinates must contain only finite values`.
- fixed-32 seed 29: 1.76% at 512, 1.56% at 4K, non-finite CE at 32K, and streaming failed at token 13,951 with the same error.
- Adaptive-32 remained finite and completed 32K streaming in all seeds, but failed the 80% 32K quality threshold in seed 17.

All finite full/stream parity errors were at most `4.77e-6`. Every arm retained `66,112` bytes. No-read and no-write accuracy stayed between 0.88% and 1.66%, confirming that the memory payload remained causal. These results do not establish safety, universal superiority, or physical efficiency.

## Reproduce or resume

```bash
.venv/bin/python scripts/run_asm_cm_vr_fixed32_long.py
```

The runner reuses completed `runs/asm_cm_vr_fixed32_long/<arm>/seed_<seed>/result.json` files and regenerates the summary, figures, and offline dashboard.

- Dashboard: `index.html`
- Raw aggregate: `summary.json`
- Frozen protocol: `manifest.json`
