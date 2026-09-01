# ATTR P1 train-only pilot

Train/validation pilot only. Test remained sealed. These results support no safety, causal-understanding, or universal-superiority claim.

## Pilot snapshot

- Validation H8 event prevalence: 13.13%.
- ASM-X: AUPRC 0.2087, Brier 0.1128, recall @ FPR≤5% 5.61%.
- Transformer: AUPRC 0.2226, Brier 0.1099, recall @ FPR≤5% 12.15%.
- The Transformer led the registered main-pair single-seed validation pilot. No registered predictive or intervention gate was assessed.

### Supplementary arms

- ASM-CM: AUPRC 0.2376, Brier 0.1099, recall 11.21%; descriptive only.
- ASM-VR-S full: AUPRC 0.2242, Brier 0.1098, recall 10.28%; descriptive only.
- ASM-VR-S fixed-32: AUPRC 0.2342, Brier 0.1100, recall 11.21%; descriptive only.
- ASM-R 240K: AUPRC 0.2328, Brier 0.1097, recall 10.28%; descriptive only.

- Seed: 17; updates per arm: 1000.
- Same HazardWorld episodes, horizons and objective for all arms.
- Threshold selected on validation only.
- Feature and episode-split leakage audits passed.
- Controls: persistence, Markov and Kalman.
- Raw results: `summary.json`; offline dashboard: `index.html`.
