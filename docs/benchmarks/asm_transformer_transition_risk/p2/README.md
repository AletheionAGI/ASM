# ATTR P2 — sealed predictive benchmark

P2 completed 30 training runs: six arms, five seeds (`29, 43, 71, 89, 107`), and 1,000 updates per arm. Test ID, shift, and OOD were generated only after every terminal checkpoint passed SHA-256 verification.

## Registered result

| Gate | Result | Evidence |
|---|---:|---|
| G0 integrity | PASS | 30/30 checkpoints; preseal and dataset seal verified |
| G1 next-state NLL | PASS | ASM-X Base − Transformer `-0.9958`, CI95 `[-1.7036, -0.4271]`; registered margin `+0.02` |
| G2 anticipation | FAIL | ΔAUPRC H8 `+0.0007`, CI95 `[-0.0340, +0.0214]`; required mean `≥0.03` and lower CI `>0` |
| G3 warning | NOT EVALUATED | P2 is predictive only |
| G4 intervention | NOT EVALUATED | P2 is predictive only |
| G5 robustness | FAIL CLOSED | ASM-X Base positive in 3/5 ID seeds; no registered critical-OOD subgroup floor |

The registered predictive sequence did **not** pass. No safety, causal-intervention, causal-understanding, or universal-superiority claim follows.

## Sealed test ID

| Arm | AUPRC H8 ↑ | Brier H8 ↓ | Next-state NLL ↓ |
|---|---:|---:|---:|
| ASM-X Base | 0.1505 | 0.1135 | 2.6036 |
| Tiny Transformer 220K | 0.1498 | 0.1137 | 3.5993 |
| ASM-CM | 0.1577 | 0.1165 | 2.3044 |
| ASM-VR-S full-64 | 0.1675 | 0.1129 | 2.2116 |
| ASM-VR-S fixed-32 | 0.1754 | 0.1122 | 2.0783 |
| ASM-R 240K protocol control | 0.1756 | 0.1121 | 2.2496 |

Supplementary arms do not change the registered ASM-X Base/Transformer gate. Fixed-32 versus full-64 ΔAUPRC was `+0.0080`, CI95 `[-0.0103, +0.0305]`; it does not support fixed-32 promotion.

H8 prevalence was 12.83% in ID, 13.84% in shift, and 56.86% in OOD. Do not interpret higher OOD AUPRC as cross-split superiority without accounting for this base-rate change.

## Artifacts

- [Offline dashboard](index.html)
- [Canonical summary](summary.json)
- [Immutable test preseal](test_spec_preseal.json)
- [Dataset/checkpoint seal](dataset_seal.json)
- [Test-open event](test_open_event.json)
- [Training implementation manifest](training_implementation_manifest.json)
- [All-arm metrics (SVG)](sealed_metrics.svg) / [PNG](sealed_metrics.png)
- [Per-seed ID AUPRC (SVG)](test_id_multiseed.svg) / [PNG](test_id_multiseed.png)
- [Registered deltas (SVG)](registered_pair_deltas.svg) / [PNG](registered_pair_deltas.png)
- Predictions: `runs/attr_p2/predictions/`
- Terminal checkpoints: `runs/attr_p2/checkpoints/`

The first evaluation attempt stopped after test opening and before writing any prediction because the runner still expected an object instead of the final split dictionary. The post-training orchestration-only patch and hashes are preserved in the implementation manifest. Checkpoints, backbones, and heads were unchanged.


## Why AUPRC H8 is not determined by next-state NLL

`sealed_metrics` reads the correct independent fields from `summary.json`. The common `NextStateHead` predicts a six-dimensional Gaussian for the next observation and is scored by NLL at every step. The separate `HazardHead` reads the same backbone representation but directly predicts the binary labels “unsafe entry within H”; AUPRC is computed from those logits. Hazard probabilities are not derived from next-state NLL or from a predicted trajectory rollout.

Consequently, lower NLL can help indirectly through a better shared representation, but the benchmark imposes no equation or monotonic constraint linking the metrics. A classifier can rank a few hazard cues correctly while reconstructing the full next-state density poorly; AUPRC also depends only on ranking, whereas NLL is sensitive to mean error and uncertainty scale across all state coordinates and mostly safe steps.

This explains how the Transformer can be competitive on H8 despite its worse NLL. “Competitive” is limited: on ID its AUPRC `0.1498` is only `1.168×` the H8 prevalence baseline `0.1283`; ASM-X Base is `1.173×`. Both direct hazard heads are weak. Across all six arms, lower ID NLL and higher AUPRC have descriptive Spearman correlation `-0.829`, but the registered pair is an important exception and the six-arm sample is too small and unmatched for causal inference.

This is a benchmark limitation, not a data-extraction error. P2 shows whether each representation supports direct hazard classification; it does not establish that better next-state forecasting *causes* better hazard anticipation.

The corrected `sealed_metrics` keeps the canonical P2 panels but labels AUPRC and Brier explicitly as outputs of the **direct hazard head**. Hazard-conditioned dynamics are intentionally separated into `hazard_conditioned_dynamics`:

- `Next-state NLL | H8=1`: dynamics error on steps from which unsafe entry occurs within eight steps;
- `Next-state NLL | H8=0`: dynamics error on the remaining steps.

On ID H8-positive steps, ASM-X Base NLL is `2.9290` and Transformer NLL is `4.1385`; on shift they are `6.5802/12.6509`, and on OOD `4.4692/9.1122`. The Transformer is therefore worse at next-state prediction even specifically along approaching-danger trajectories. Its competitive AUPRC comes from the separate direct classifier, not from demonstrated trajectory forecasting.

Conditioning by H8 is an evaluation decomposition only: the future label is not used as an input or risk score. A genuine trajectory-derived AUPRC cannot be computed from the current one-step head without leaking the realized next state. It requires a new multi-horizon state-prediction protocol and a fixed unsafe predicate.

See [canonical sealed metrics](sealed_metrics.svg), [hazard-conditioned dynamics](hazard_conditioned_dynamics.svg), and [dynamics versus anticipation](dynamics_vs_anticipation.svg).


## Post-hoc diagnostic: ASM-X + Native Risk Mass

This exploratory extension compares **ASM-X Base** with **ASM-X + Native Risk Mass**. Both have exactly `226,444` total/trainable parameters including the common heads and identical initial tensors. The sole configuration delta is `use_powerlaw_risk: false → true`. The same five seeds, train/validation episodes, common heads, objective, 1,000-update budget, and already-sealed P2 test splits were used.

The arm was selected after the original P2 test was observed. It is therefore diagnostic, not a seventh registered arm, and cannot revise G0–G5.

| Split | Base AUPRC H8 | + Native Risk Mass | Paired ΔAUPRC | CI95 |
|---|---:|---:|---:|---:|
| ID | 0.1504970 | 0.1504975 | +0.00000001 | [-0.00000560, +0.00000727] |
| shift | 0.1697434 | 0.1697435 | -0.00000010 | [-0.00000456, +0.00000250] |
| OOD | 0.6227413 | 0.6227408 | -0.00000124 | [-0.00001633, +0.00000481] |

Brier, validation-threshold recall/FPR, and next-state NLL were likewise operationally unchanged. All H1/H4/H8/H16 AUPRC confidence intervals include zero. The native risk parameters did receive gradients and changed from initialization in all five seeds, but under the frozen configuration their effect on the learned trajectory and predictions was negligible. This result does not show that risk-aware modeling is generally ineffective; it shows that simply enabling the existing Native Risk Mass with its current weight and common ATTR objective did not improve this benchmark.

Artifacts:

- [Diagnostic summary](risk_mass_extension_summary.json)
- [Pretrain manifest](risk_mass_extension_pretrain_manifest.json)
- [Checkpoint seal](risk_mass_extension_checkpoint_seal.json)
- [Metric grid](risk_mass_metrics.svg)
- [AUPRC by horizon](risk_mass_horizons.svg)
- [Paired deltas](risk_mass_deltas.svg)
- [Per-seed ID comparison](risk_mass_test_id_multiseed.svg)
