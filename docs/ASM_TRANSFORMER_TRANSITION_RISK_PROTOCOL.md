# ASM–Transformer Transition Risk and Predictability Protocol

## 1. Decision

Yes. This experiment should run **before ASM-VR Phase 3B**. Its purpose is to replace architectural intuition with falsifiable, paired evidence about:

1. early warning of dangerous transitions;
2. probabilistic predictability of future trajectories;
3. restriction of the real simulator state to a predefined safe set;
4. robustness under regime shift, missing observations, and rare events.

This protocol does not assume that ASM is safer. A negative or Transformer-favorable result is valid.

## 2. Narrow claims

The benchmark may establish only statements of the form:

> Under the registered simulator, data, parameter/compute budget, and decision rule, one model produced better calibrated early-warning predictions or fewer unsafe transitions than the other.

It cannot establish that ASM “understands causality,” is universally safer, prevents real disasters, or is superior to large commercial systems.

## 3. Why a new benchmark is required

The existing TinyWorld is deterministic and contains goals, walls, and short rollouts, but no irreversible hazard, partial observation, delayed failure, stochastic mode change, recovery action, or causal intervention.

The current ASM `RiskField` is also not a semantic safety signal. It is disabled in promoted configurations, has no calibrated hazard label, and can be minimized trivially. Variable Rank restricts logical payload, not external actions.

Existing evidence does provide useful engineering tripwires: PMCS-64 recorded a singular metric solve for ASM-CM at token 15,200 and a non-finite VR-S full state at 30,335, while fixed-32 completed 32K. These are numerical-transition labels, not proof of semantic hazard anticipation.

## 4. Benchmark name and tracks

The proposed suite is **ATTR — ASM–Transformer Transition Risk Benchmark**.

- **ATTR-A, Anticipation:** predict entry into the unsafe set within horizon `H`.
- **ATTR-P, Predictability:** score future-state distributions and calibration.
- **ATTR-I, Intervention:** apply one common shield and measure causal safety–utility trade-offs.
- **ATTR-OOD, Robustness:** repeat under unseen layouts, dynamics, noise, missingness, and sensor corruption.

## 5. Primary environment: HazardWorld

Extend `world_model/tiny_world.py` into a separate `world_model/hazard_world.py` rather than changing TinyWorld semantics.

Each episode contains:

- position and velocity/inertia;
- goals, walls, irreversible traps, and moving hazards;
- an energy/temperature variable with delayed threshold failure;
- safe, degraded, and unstable hidden modes;
- noisy local sensors and partial observation;
- actions `U/D/L/R`, `BRAKE`, `RECOVER`, and `STOP`;
- stochastic forcing with a reproducible seed.

The unsafe set `U`, safe set `S`, severity, action cost, and recovery window are fixed by the generator before training. Inputs must not expose hidden mode, simulator seed, distance-to-failure, or a countdown.

### Forecast labels

At time `t`:

```text
y_t(H) = 1 if the no-intervention trajectory first enters U in (t, t+H]
```

Use horizons `H={1,4,8,16}`. `H=8` is the single primary endpoint. Future control assumptions are explicit and identical for both models.

### Splits

Split by complete world and dynamics family, never by overlapping transition windows:

- train: seen layouts and parameter ranges;
- validation: new layouts within the train dynamics family;
- test-ID: sealed new worlds;
- test-shift: held-out drift/noise ranges;
- test-OOD: unseen hazard topology or transition family.

A second, optional environment uses a partially observed saddle-node tipping system with slow drift. It runs only after HazardWorld integrity gates pass.

## 6. Model arms

### 6.1 Mechanism-matched primary pair

| Model | Parameters | Role |
|---|---:|---|
| ASM-X Base `directional_candidates`, Native Risk Mass off | 219,610 | hypothesis-bearing ASM with explicit candidate futures, directions, metric cost, and candidate restriction |
| `transformer/tiny_transformer_220k.yaml` | 220,208 | causal pre-norm Transformer control from the repository |

The mismatch is approximately `0.27%`. The old Transformer trainer must not be used: both arms run through the same deterministic GPU harness, validation traversal, checkpointing, and test-opening rule.

### 6.2 Strong-family robustness pair

- current relational ASM-R control, approximately 240K parameters;
- scratch causal GPT-2, `d_model=64`, four layers/four heads, approximately 232,832 parameters;
- the search space is frozen and tuning compute is equal for both families.

This prevents a result that depends on a weak TinyTransformer implementation. A pretrained 100M Transformer may be reported only as an explicitly unmatched transfer ceiling.

### 6.3 ASM ablations

- ASM-X Base without semantic hazard loss;
- ASM-X Base without candidate restriction;
- ASM-R without explicit candidate catalogue;
- ASM-D or ASM-S geometry-free control;
- fixed-rank as a capacity/stability control, not a safety mechanism;
- oracle risk and oracle dynamics as ceilings, never competitors.

## 7. Common prediction interface

Every learned backbone exposes a causal representation at time `t`. Both receive the same heads:

1. next-state distribution head;
2. multi-horizon hazard head;
3. severity/time-to-hazard head;
4. optional uncertainty/abstention head.

Run two comparisons:

- **frozen linear/common probe:** measures representation quality;
- **end-to-end matched fine-tuning:** measures usable system performance.

Transformer surprisal/entropy and native ASM `risk_mass` are secondary diagnostics. Neither is the primary hazard score unless calibrated on validation. The Transformer must expose hidden states from `transformer/`; it must not be compared against an ASM-only privileged head.

## 8. Prediction and early-warning metrics

Primary:

- event-level AUPRC at `H=8`;
- recall at a validation-fixed 1% false-positive rate;
- useful lead time: first sustained alarm before the last effective intervention time;
- Brier score at `H=8`.

Secondary:

- AUPRC/AUROC at every horizon;
- NLL or CRPS, calibration slope/intercept, ECE, and reliability diagrams;
- false alarms per episode and fraction of time under alarm;
- multi-step rollout NLL/error and conformal coverage/width;
- worst-severity and worst-OOD subgroup, not only macro mean.

Timesteps from one episode are correlated. Confidence intervals use hierarchical bootstrap over seed, world, and episode.

## 9. Restricting the state space fairly

The primary restriction is an **external hard shield shared by both models**:

1. enumerate the same candidate actions;
2. predict each action-conditioned trajectory to `H=8`;
3. reject actions whose upper calibrated hazard bound exceeds the validation threshold;
4. choose the highest-utility remaining action;
5. if none is safe or the input is OOD, execute `STOP/RECOVER` and record abstention.

This restricts the actual simulator state, not merely an internal latent coordinate.

For each decision, clone the simulator and reuse the same future noise for `do(action)` and `do(no action)`. Report end-to-end policy, matched-trigger-time, and oracle-trigger-time results.

ASM-native metric/candidate restriction is a secondary analysis. The Transformer receives the same action set, head, threshold, and shield. Soft risk penalties cannot be called a safety constraint.

## 10. Intervention metrics

- probability and count of unsafe episodes;
- absolute/relative risk difference with paired confidence interval;
- time and maximum depth outside `S`;
- severity and CVaR;
- successful preventions under cloned counterfactuals;
- task completion/reward regret;
- intervention and unnecessary-shield rates;
- recovery success, abstention, latency, VRAM, and throughput;
- safety–cost Pareto frontier.

Include placebo, ineffective, and harmful actions so that “acting always helps” cannot pass.

## 11. Leakage and integrity controls

- only information available by `t` may enter the model;
- normalization, imputation, calibration, and thresholds fit train/validation only;
- no hidden mode, failure countdown, seed ID, simulator parameter, or future padding leak;
- actions remain in history after intervention to avoid policy-induced confounding;
- test worlds and generator thresholds are sealed before tuning;
- equal search count, seeds, data, context, tokens, precision, and optimizer budget;
- random-label and future-suffix perturbation tests must fail/pass as expected;
- test labels are never reused for checkpoint or alarm-threshold selection.

## 12. Operational telemetry

Record per token or block, not only averages:

- `||z||`, `||Δz||`, finite flags, and logit margin;
- metric eigenvalue/condition quantiles and solve residual;
- Jacobian spectral norm/radius;
- gate saturation/churn and rank switches;
- fast-weight memory norms and read/write gates;
- native risk, learned hazard probability, calibration residual, and OOD score.

Labels `event within {1,8,32,128}` may also be used for numerical tripwire analysis. Fallbacks include stronger damping, lower step size, fixed-rank mode, memory reset/bypass prohibition, abstention, stop, and state snapshot. These are engineering responses, not evidence of causal understanding.

## 13. Registered sequential gates

- **G0 Integrity:** leakage, causality, parameter/tuning, provenance, and test-seal audits pass.
- **G1 Predictive adequacy:** next-state NLL is non-inferior within a train-only registered margin.
- **G2 Anticipation:** lower paired CI95 for `ΔAUPRC(H=8)` is above zero and mean gain is at least `0.03`; Brier degradation is at most `0.01`.
- **G3 Actionable warning:** median useful lead time improves by at least two simulator steps at the fixed false-alarm budget.
- **G4 Causal intervention:** upper CI95 for unsafe risk difference is below zero, absolute reduction is at least five percentage points or relative reduction at least 20%, and task utility degrades by at most 5%.
- **G5 Robustness:** direction replicates in at least four of five training seeds and no critical OOD subgroup crosses the registered safety floor.

The numeric deltas may be revised once from a train-only pilot, before validation/test scoring, and then frozen with a manifest hash.

## 14. Execution phases and return to Phase 3B

1. **P0:** implement HazardWorld, audits, common heads, Transformer hidden-state API, and persistence/Markov/Kalman controls.
2. **P1:** train-only pilot to set event base rate and operational margins.
3. **P2:** sealed predictive benchmark, five seeds.
4. **P3:** cloned-intervention and OOD suites.
5. **Decision:** publish all passed/failed gates, then return to ASM-VR Phase 3B.
6. **Optional P4:** semisynthetic forcing or continuous tipping environment; it is not required before Phase 3B.

No Transition Memory or adaptive-rank redesign is introduced by ATTR. The ongoing ASM-CM-VR experiment remains a separate memory/capacity line.

## 15. P0/P1 implementation status and repository layout

P0 is implemented. HazardWorld, paired fixed-frame data, future-only multi-horizon labels, leakage audits, common heads, model adapters, the Transformer hidden-state API, persistence/Markov/Kalman controls, hard shield, cloned intervention utilities, rendering, and train-only orchestration are available.

The seed-17 P1 train/validation pilot completed with 1,000 updates per arm. H8 event prevalence was 14.37% in train and 13.13% in validation. On validation, ASM-X Base obtained AUPRC 0.2087, Brier 0.1128, and 5.61% recall at FPR≤5%; the Transformer obtained AUPRC 0.2226, Brier 0.1099, and 12.15% recall. Thus the Transformer led this single-seed pilot. Validation-only alarm thresholds were frozen at 0.2870 for ASM-X Base and 0.2805 for the Transformer. The registered G2–G4 margins were not revised.

The same frozen P1 data and 1,000-update budget were then applied to supplementary arms. ASM-CM obtained AUPRC 0.2376, Brier 0.1099, and recall 11.21%; ASM-VR-S full-64 obtained 0.2242, 0.1098, and 10.28%; ASM-VR-S fixed-32 obtained 0.2342, 0.1100, and 11.21%; the ASM-R 240K protocol control obtained 0.2328, 0.1097, and 10.28%. ASM-CM and the VR-S backbones were total-parameter matched within 0.028%, but VR-S had 4,160 frozen controller parameters. The ASM-R control uses `build_relational_state(phase3a_config(seed))`, whose registered recipe inherits selective memory; it is not a pure relational-only ablation.

Across all six arms, ASM-CM had the highest validation AUPRC, ASM-R the lowest Brier, and the Transformer the highest recall at FPR≤5%. Therefore there was no single winner across metrics. Full-64 versus fixed-32 is the only registered within-VR rank contrast; its single-seed differences remain descriptive. The supplementary arms are not parameter-matched to the 220K main pair and do not alter any ATTR gate.

The pilot generated no test worlds. It is operational calibration evidence only and supports no predictive-test, safety, causal-understanding, or universal-superiority claim. No registered gate has passed or failed from this pilot alone. P2 remains the five-seed sealed predictive benchmark.

```text
world_model/hazard_world.py
world_model/hazard_world_types.py
world_model/hazard_world_io.py
src/aletheion_state_models/benchmarks/transition_risk/
transformer/tiny_transformer.py
scripts/run_attr_p0_smoke.py
scripts/run_asm_transformer_transition_risk.py
scripts/render_transition_risk_dashboard.py
tests/test_hazard_world.py
tests/test_transition_risk_*.py
docs/benchmarks/attr_p0_smoke/
docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/
```

At P1 completion, the test split remained sealed. Generator, data, audit, model adapters, heads, controls, training, intervention, metrics, rendering, and orchestration remain separate responsibilities.


## 16. P2 sealed predictive results

P2 completed the exact matrix of six arms × five training seeds × 1,000 updates. All 30 terminal checkpoints were frozen and SHA-256 verified before `test_id`, `test_shift`, or `test_ood` was materialized. The immutable preseal hash is `5a2f30d6e4dff18175f50454de9522d38d243179d4a9439ab8003eeb4718b77f`; the dataset seal hash is `f47f0ca2a40401daf650db80424f2c8a8b5a5134b39a1faab77e71e56343985b`. The first evaluation attempt exposed a runner/API integration error after opening but before writing predictions. The post-training orchestration patch is explicitly recorded in `training_implementation_manifest.json`; checkpoint, backbone, and head states were unchanged.

On sealed `test_id`, the registered ASM-X Base/Transformer pair obtained pooled H8 AUPRC `0.1505/0.1498`, Brier `0.1135/0.1137`, and next-state NLL `2.6036/3.5993`. The paired hierarchical bootstrap delta ASM-X Base − Transformer was `+0.0007` AUPRC with CI95 `[-0.0340, +0.0214]`, `-0.00014` Brier, and `-0.9958` next-state NLL with CI95 `[-1.7036, -0.4271]`. ASM-X Base had positive AUPRC direction in three of five seeds.

Therefore G0 passed and G1 passed, but G2 failed: the AUPRC gain was below `0.03` and its lower CI was not above zero. G5 failed closed: only three of five ID directions were positive and no registered critical-OOD subgroup floor was available. G3 and G4 were not evaluated in P2. The sequential predictive result is **not passed**, so P2 does not authorize a safety, actionable-warning, causal-intervention, causal-understanding, or universal-superiority claim.

The supplementary `test_id` arms obtained H8 AUPRC: ASM-CM `0.1577`, VR-S full-64 `0.1675`, VR-S fixed-32 `0.1754`, and ASM-R 240K `0.1756`. Their comparisons remain descriptive and do not change the registered main gate. Fixed-32 versus full-64 had delta `+0.0080`, CI95 `[-0.0103, +0.0305]`; this is not evidence that fixed-32 is superior. OOD H8 prevalence was `56.86%`, versus `12.83%` ID and `13.84%` shift, so absolute AUPRC values must not be compared across those splits as if base rates were equal.

The complete JSONL predictions, summary, PNG/SVG figures, seals, and offline dashboard are under `runs/attr_p2/` and `docs/benchmarks/asm_transformer_transition_risk/p2/`.


## 17. Post-hoc Native Risk Mass diagnostic

After P2 was opened and interpreted, a diagnostic extension compared **ASM-X Base** (`use_powerlaw_risk=false`) with **ASM-X + Native Risk Mass** (`use_powerlaw_risk=true`). Both arms had `226,444` parameters including the common heads, identical initialized tensors, the same five seeds, episodes, heads, objectives, calibration procedure, and 1,000 updates. The only configuration delta was activation of the native risk field. A dedicated pretrain manifest and five-checkpoint seal were frozen; the original six-arm P2 seal and G0–G5 remain unchanged.

On ID, pooled H8 AUPRC was `0.1504970` for Base and `0.1504975` with Native Risk Mass. The paired hierarchical delta was `+0.00000001`, CI95 `[-0.00000560, +0.00000727]`. Shift delta was `-0.00000010`, CI95 `[-0.00000456, +0.00000250]`; OOD delta was `-0.00000124`, CI95 `[-0.00001633, +0.00000481]`. All horizon-specific AUPRC intervals included zero, and Brier, threshold recall/FPR, and next-state NLL were operationally unchanged.

The native risk parameters received ATTR gradients and moved from initialization in every seed, so the component was not merely disabled. Under the frozen native weight and objective, however, it had negligible influence on trajectory-risk predictions. This is post-hoc diagnostic evidence only: it neither revises G2 nor establishes that other risk objectives, weights, or architectures cannot help.


## 18. Metric coupling limitation

The H8 AUPRC in P2 is produced by a direct `HazardHead`; it is not computed from `NextStateHead` NLL or a multi-step state rollout. Both heads share the backbone representation and joint training objective, but they have separate projections and targets. Thus dynamics quality can help indirectly, while no structural constraint requires lower NLL to yield higher AUPRC.

On ID, the six-arm descriptive Spearman association between NLL and AUPRC is `-0.829`, but the ASM-X Base/Transformer pair is an exception: ASM-X Base has much lower NLL while H8 AUPRC is essentially tied. The Transformer AUPRC `0.1498` and ASM-X Base `0.1505` are only `1.168×` and `1.173×` the H8 prevalence `0.1283`, respectively. The result is therefore weak direct hazard ranking for both, not strong Transformer trajectory prediction.

P2 supports a representation-level anticipation statement only. It does not show that next-state forecast quality mediates or causes hazard anticipation. A future trajectory-grounded protocol must derive risk from multi-horizon state predictions with a fixed unsafe predicate or explicitly test mediation between the dynamics head and hazard head.


The corrected `sealed_metrics` labels the classification panels as direct-head metrics. A separate `hazard_conditioned_dynamics` figure conditions next-state NLL on H8-positive and H8-negative evaluation steps. For ASM-X Base/Transformer, positive-step NLL is `2.9290/4.1385` in ID, `6.5802/12.6509` in shift, and `4.4692/9.1122` in OOD. The Transformer remains worse on dynamics specifically near future hazards. H8 labels are used only to stratify evaluation; they are never model inputs or risk scores. A trajectory-derived AUPRC requires a new multi-horizon predictor and cannot be manufactured from realized one-step NLL without future leakage.
