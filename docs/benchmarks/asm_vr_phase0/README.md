# ASM-VR Phase 0 invariants

This artifact records the isolated Phase 0 validation of the variable-rank
state contract. It does not train a language model and does not claim runtime
speedup.

## Reproduce

```bash
.venv/bin/python scripts/eval_asm_vr_phase0.py --samples 2048 --seed 2026
```

## Acceptance results

| Check | Result |
|---|---:|
| Hard-projector idempotence error | 0.0 |
| Soft-filter non-idempotence | 2.5913430029491997 |
| Discarded-information recovery from effective state | 0.0001124092436157742 |
| Discarded-information recovery from declared external memory | 1.0 |
| Numerical rank of the `8→3→5→8` cycle Jacobian | 3 |
| Required Jacobian rank upper bound | 3 |
| Cycle rank deficit | 5 |
| Minimum dissipation eigenvalue | 0.0 |

All acceptance checks passed. The probe is a deterministic held-out ridge
linear probe. A near-zero score is evidence against a **linear** bypass in this
controlled Gaussian experiment; it is not a proof against every nonlinear
channel. The external-memory condition deliberately exposes the discarded
coordinates and serves as the positive control.

Source data: [`summary.json`](summary.json).
