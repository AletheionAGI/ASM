# ATTR-TG1 — Trajectory-Grounded Anticipation Protocol

## Question

Can ASM-X Base and a parameter-matched Tiny Transformer anticipate danger when the direct hazard classifier is removed and the only permitted path is:

`causal representation → predicted physical trajectory → fixed unsafe predicate → risk`?

This protocol is new and does not reinterpret ATTR P2. P2 test data and checkpoints are ineligible.

## Registered arms and seeds

- `asm_x_base`: ASM-X Base with `use_powerlaw_risk=false`;
- `transformer_base`: Tiny Transformer 220k;
- paired optimizer seeds: `29, 43, 71, 89, 107`;
- ASM representations are zero-padded from 64 to 72 without trainable parameters; both arms use the same 72-wide decoder initialized bit-for-bit identically.

## Data and actions

Train and validation use new fixed common corpora: 64×4 and 16×4 episodes. Fresh ID, shift and OOD tests use 32×4 episodes each and are not generated before all ten checkpoints and K256 validation records are sealed.

At each causal origin `t`, an H8 open-loop plan is committed using only the current action and a domain-separated hash. Its seven-step suffix never reads future observations and excludes STOP. A cloned simulator executes the same plan to produce privileged physical training targets. Future actions from the realized closed-loop behavior policy are never exposed at `t`.

## Physical trajectory decoder

The common autoregressive decoder predicts all H1…H8 states, not only H1/H4/H8. It predicts categorical distributions for three trap cells and, at each step, agent cell, moving-hazard cell, velocity, physical energy, low-energy counter, recovery counter, hidden mode, and safe termination.

The loss is physical categorical trajectory NLL only. No `HazardHead`, unsafe label, severity, time-to-hazard target, or direct risk loss is allowed.

## Risk construction

For each origin, K=256 free-running physical trajectories are sampled with arm-paired common random numbers. Risk at H is the fraction entering the fixed unsafe set by H:

- collision: predicted agent cell equals a predicted trap or moving-hazard cell;
- delayed failure: predicted low-energy counter reaches the public environment delay while recovery is zero;
- duplicated predicted traps, invalid values, or non-finite values fail closed as unsafe;
- positions after a predicted safe terminal are ignored.

True traps and realized future state are never inputs to scoring. Ground-truth unsafe events are held separately for metrics only.

## Training and sealing

Each arm/seed receives 1,000 AdamW updates, batch 4, learning rate `3e-4`, weight decay `0.01`, and terminal-checkpoint-only selection. Train/validation data, code, configs, seeds, action planner, decoder, predicate, metrics and gates are hashed before training. Test opening is one-shot and occurs only after 10/10 terminal checkpoints plus complete validation K256 records.

The leakage control is procedural, not cryptographic custody: test seeds are committed in the preseal but episodes are materialized only after the checkpoint seal.

## Metrics and registered decision

Primary ID/H8 metrics are trajectory-derived AUPRC and Brier. Joint and per-field trajectory NLL, event log loss, H1/H4, validation-only FPR≤5% thresholds and lead time are supporting metrics. Uncertainty uses paired hierarchical bootstrap seed→world→episode with 1,000 registered replicates.

`TG2_trajectory_anticipation_id` passes only if ASM minus Transformer satisfies all three:

- ΔAUPRC H8 ≥ `0.03`;
- lower IC95 ΔAUPRC H8 > `0`;
- upper IC95 ΔBrier H8 ≤ `0.01`.

Shift/OOD robustness is reported separately and requires non-negative lower AUPRC bounds and Brier upper bounds ≤0.01. Failed/non-finite/missing runs remain in the matrix and fail closed.

## Interpretation boundary

A positive result is evidence for useful `state → trajectory → predictability`. It is not evidence of causal intervention or safety. The last link requires a later cloned `do(action)` experiment with unsafe reduction and utility bounds.

## Outputs

Results are separate from P2 under `docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/`. The main comparison is `trajectory_grounded_anticipation.png/.svg`.
