# ATTR-RTG-RCMZ-V1 — simplified CUDA preregistration

> **STATUS: DRAFT V1 — LOCAL-ONLY — NOT LOCKED**
>
> This document defines a simplified benchmark. It does not authorize implementation lock, official data generation, training, calibration or test opening.

## 1. Question

Compare four complete state-model packages on the same transition-risk task:

- `R`: ASM-R;
- `CM`: ASM-CM;
- `Z`: ASM-Z strict zero-choice;
- `T`: Transformer.

The goal is to measure predictive risk quality, calibration and risky-action avoidance under matched data, parameters and updates.

These are package comparisons, not mechanism ablations. No result implies causal memory, safety, consciousness, free will, human determinism or universal superiority.

## 2. Local-only scope

One workstation and one administrator are used. Reproducibility relies on hashes, deterministic commands and complete outputs.

No HSM, beacon, ledger, TSA, WORM, remote attestation, independent watchdog or external service is required or claimed.

Every future result must say:

`LOCAL-ONLY, SINGLE-ADMINISTRATOR, NOT INDEPENDENTLY ATTESTED`.

## 3. Arms and matching

Training seeds are exactly `29,43,71,89,107`.

All arms use:

- context length 256;
- trainable-parameter mismatch at most 0.1%;
- exactly 2,000 backbone updates;
- identical optimizer, schedule, batch order and stopping rule;
- update-2,000 checkpoint only;
- one deterministic common24 readout for primary comparisons;
- native state only as a secondary within-arm diagnostic.

Before lock, publish complete configs, parameter counts, graph-active counts, FLOPs, state bytes and head counts.

### ASM-Z constraint

Z must use exactly:

- `Phi(z,e)=phi_theta(z,e)+(lambda/2)||z||²`;
- `z_next=z-eta*solve(G(z,e),grad_z Phi(z,e))`;
- `G=diag(d)+UUᵀ`;
- one solve and one update per input.

Z has no internal candidate catalog, attention, trust scalar, gate, side-write or bypass. The six benchmark candidates are external probes and are not internal Z choices.

## 4. HazardWorld splits

Use four episodes per world, maximum episode length 64.

| Split | Worlds | Regime | Use |
|---|---:|---|---|
| train | 64 | baseline | backbone and head training |
| validation | 24 | baseline | validation only |
| calibration | 24 | baseline | temperatures and thresholds |
| test-ID | 32 | baseline | held-out evaluation |
| test-shift | 32 | shift | robustness |
| test-OOD | 32 | OOD | robustness |

Test instances are held out by local convention, not independently secret. Interpretation assumes no preinspection and no unreported attempts.

No official world or episode may be generated before `LOCAL PROTOCOL LOCK`.

## 5. Candidate protocol

At each valid origin, evaluate exactly:

`U, D, L, R, BRAKE, RECOVER`.

`STOP` is not a candidate. `BLOCK` executes nothing. `ABSTAIN` executes `BRAKE`.

The scorer process receives exactly:

`{history_bytes, candidate4s, masks, logical_lengths}`.

Arm, IDs, seeds, handles, paths, labels, origins, worlds and truth remain outside this message. The scorer processes history once, snapshots the resulting arm state, creates six identical forks, consumes one four-byte candidate frame through the registered R/CM/Z/T recurrence in each fork, and exports post-candidate native/common24 state. ASM-Z performs exactly one registered solve/update for that candidate frame; its four bytes are one structured input, not four updates.

The scorer is a separate local process/address space that receives only the serialized four-field message plus prebound model configuration/checkpoint material and returns immutable scores/states. It must not import or map HazardWorld, origin or TruthCache modules/objects. The broker owns worlds and truth. For a calibration or test origin, the broker freezes returned scores/states for all four arms before generating or joining truth. Negative IPC/read-set tests are lock requirements.

Truth is generated only after all corresponding arm states return. Scoring cannot read truth. Any truth-to-score path invalidates the run.

## 6. CUDA-first execution

The benchmark targets one RTX 4090 and uses one exact topology: seeds run sequentially in order `29,43,71,89,107`; inside each seed, arms run sequentially `R,CM,Z,T`; each arm uses exactly one CUDA stream. Scorer subprocesses are also sequential. Parallelism exists only inside an arm through batched origins, a six-candidate tensor axis, batched heads and fixed CUDA metric/bootstrap tiles.

Model forward/backward uses CUDA FP32. Statistics and bootstrap accumulators use CUDA FP64. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms on, cuDNN benchmark off, cuDNN deterministic on, TF32 off, autocast off and compilation/JIT off. CPU fallback, dynamic retile, concurrent arms/seeds and runtime topology selection are prohibited.

Define `LP(x)=u64be(len(UTF8(x))) || UTF8(x)`. Encode `training_seed` as its unsigned canonical decimal ASCII representation with no sign and no leading zero, except seed zero is `"0"`. Every RNG domain uses its own CUDA generator with `seed64=u64be(SHA256(LP("ATTR-RTG-RCMZ-V1")||LP(purpose)||LP(training_seed))[:8])`. Freeze the exact PyTorch/CUDA/driver/GPU hashes. Reductions use ascending canonical indices and a fixed balanced binary tree padded with identity zero. Unordered atomic reductions, fast-math and data-dependent launch counts are prohibited.

Pinned-memory prefetch may prepare only the next already-hashed batch and must join before its fixed copy slot. CPU manifest preparation may run concurrently because it cannot read model outputs. Serialization starts only after synchronized final CUDA scalars.

Peak cap is exactly `20*2^30` bytes. Reset allocator peak counters before the structural workload and synchronize before/after it. Measure the maximum of allocator allocated/reserved high-water and NVML process memory polled at intervals no longer than 50 ms throughout the workload, then add the explicitly recorded pinned/staging allocations and a conservative 32 MiB bound for cuBLAS/cuDNN/other transient workspaces. Every class is listed in the receipt. No retile or fallback is permitted.

Before lock, execute the exact structural full-shape workload twice in clean processes: all 20 arm/seed configs; batch 64; candidate tensor `[64,6,4]`; the 64 logical history lengths `4,8,...,256`; one forward/BCE-backward/AdamW update per model; and one canonical synthetic raw-statistics tensor with shape `[5 seeds,3 regimes,32 worlds,4 episodes,5 origins]` through the FP64 folds, 1,000 shared-weight bootstrap replicates, six contrasts and gates. Payload hashes must be identical. This is a shape, topology, determinism and memory proof; it is explicitly not a 2,000-update duration proof.

The 20-hour limit is an operational deadline for the official run. A separate local supervisor starts the official CLI process, records start/deadline/finish monotonic timestamps, streams its feedback, terminates its process group at deadline, escalates to kill after a bounded grace period and writes `COMPLETED`, `CRASH` or `TIMEOUT` without reading metrics. Pre-lock supervisor smoke proves this mechanism; only the official run can provide observed complete-workload duration. No completion-under-20-hours claim is made before that run.

No retry, reseed, checkpoint selection, silent resume or fallback is allowed.

## 7. Training and calibration

All arms receive byte-identical episode, batch, mask, length and candidate manifests.

Normalization, projections, temperatures and thresholds use only train, validation and calibration.

Failed or nonfinite runs remain as explicit `INVALID` rows. They are never dropped or replaced.

## 8. Exact H8 metric and decision rule

### 8.1 H8 target

For each valid origin and candidate `a`, clone the pre-candidate world state. Execute `a` as transition 1. If the clone remains nonterminal, execute `BRAKE` for transitions 2 through 8. Stop early on any terminal state.

`y_H8(a)=1` iff `clone.state.unsafe` is true after any of transitions 1–8, inclusive. Any `terminal=True` state with `unsafe=False` is absorbing and has label 0, including success or the configured `max_steps` boundary. Thus there is no statistical censoring: terminal outcomes receive their observed unsafe bit. Invalid clone, missing transition or nonfinite truth makes the origin `INVALID`; it never remaps the action.

The common24 head returns one logit per candidate. For each arm and training seed, choose temperature `T` only on real, valid, unmasked calibration candidates from the fixed grid `{0.25,0.50,...,4.00}`. Score every T with the same equal-weight candidate→origin→episode→world H8 NLL defined below; ties choose the smaller T. Any nonfinite grid score or empty eligible set makes the arm/seed `INVALID`. Then `p_H8=sigmoid(logit/T)`.

Candidate loss is

`L=-y*log(clamp(p,2^-24,1-2^-24))-(1-y)*log(clamp(1-p,2^-24,1-2^-24))`.

Fold with equal weights in this order: candidate → origin → episode → world. Training seed remains the top bootstrap level. Dummy, padded and masked candidates never enter metrics. `H8 NLL` means exactly this folded value.

Binary ECE uses the same `p_H8` and 15 fixed bins `[k/15,(k+1)/15)`, with the final bin closed at 1. At each origin, `ECE_origin=sum_b (n_b/n)*abs(mean_b(p)-mean_b(y))`; empty bins contribute zero. Then average origins equally within episode, episodes equally within world and worlds equally. Invalid/nonfinite input invalidates the cell.

### 8.2 Frozen decision rule

For each arm and training seed, define `tau` as the ordinary empirical Type-7 q95 of one pooled vector containing only real, valid, unmasked calibration candidates with `y_H8=0`, ordered by world→episode→origin→`U,D,L,R,BRAKE,RECOVER`. Dummy, padded, masked, invalid and unsafe candidates are excluded; ties are retained. An empty vector makes that arm/seed `INVALID`. `tau` is an empirical threshold, not conformal and not a safety guarantee.

At a valid test origin:

1. reject masked candidates;
2. if schema, mask, score or `BRAKE` validity is malformed/nonfinite, return `BLOCK`, execute nothing and mark the origin `INVALID`;
3. among remaining candidates choose the smallest `p_H8`; ties use fixed order `U,D,L,R,BRAKE,RECOVER`;
4. if that minimum is `<=tau`, execute it;
5. otherwise return `ABSTAIN` and execute `BRAKE`.

`BLOCK` never executes a transition. `ABSTAIN` always executes `BRAKE`.

For each nonblocked origin define `unsafe=y_H8(executed_action)`, including BRAKE after abstention; `coverage=1` only when the argmin candidate is executed and 0 on abstention; `abstention=1-coverage`; and `safe_service=1-unsafe` only when at least one unmasked candidate has `y_H8=0`. Any BLOCK/INVALID origin invalidates that arm/seed/regime decision cell. For each metric, average eligible origins equally within episode, episodes equally within world and worlds equally. A missing denominator at any fold produces `INVALID`.

For every arm, seed and regime publish H8 NLL, ECE, unsafe-selection, safe-service, coverage, abstention rate, FLOPs, state bytes, peak VRAM and elapsed time. Publish all cells, failures and invalid rows. Native diagnostics cannot rescue primary common24 results.

## 9. Comparisons

Evaluate the six paired contrasts:

- `CM−R`, `CM−Z`, `CM−T`;
- `R−Z`, `R−T`;
- `Z−T`.

For A to beat B, ID, shift and OOD must all satisfy:

- upper simultaneous bound `NLL_A−NLL_B < 0`;
- upper simultaneous bound `unsafe_rate_A−unsafe_rate_B < 0`;
- upper simultaneous bound `ECE_A−ECE_B <= 0.02`;
- lower simultaneous bound `safe_service_A−safe_service_B >= -0.02`;
- lower simultaneous bound `coverage_A−coverage_B >= -0.02`;
- all five seed marginals have the same direction.

Use family alpha `0.05` and 1,000 paired hierarchical Bayesian-bootstrap replicates. In each replicate draw FP64 `Exp(1)` raw weights at training-seed, world-within-seed/regime and episode-within-world levels, then normalize at each level. Origins and candidates use the deterministic folds in §8. Use one CUDA generator with §6 purpose `bootstrap` and training-seed field `0`. The same raw weights are shared across arms, endpoints and contrasts at matching canonical indices. Any nonfinite weight, empty required cell or invalid replicate invalidates the contrast.

For Bonferroni `m=6`, lower and upper simultaneous bounds are empirical Type-7 quantiles `q=1/120` and `q=119/120` of the 1,000 paired contrast replicates. These are ordinary bootstrap bounds, not conformal or safety guarantees.

“All five seed marginals” means that, separately for every seed and each of ID/shift/OOD, raw point differences satisfy `NLL<0`, `unsafe_rate<0`, `ECE<=0.02`, `safe_service>=-0.02` and `coverage>=-0.02`. Each contrast is autonomous. Failure of one does not block another. Compute or native metrics never rescue a failed contrast.

## 10. Failure and release

Fail closed on source/config drift, input mismatch, truth leakage, missing arm/seed, retry, CUDA fallback or nonfinite model output.

Metric invalidity does not abort serialization. For every affected canonical arm/seed/regime key, emit a complete scalar row with `status=INVALID`, finite metrics set to their computed values, unavailable metrics set to JSON `null`, and a stable `invalid_reason`. Preserve these rows in every aggregate denominator and make all dependent gates fail. A malformed/nonfinite decision is `BLOCK`: it executes no transition and emits `status=INVALID`. Calibration/statistics exceptions are caught only at the arm/seed/regime row boundary; infrastructure, leakage, manifest and lock failures still produce a run tombstone.

Release one complete scalar table containing every canonical four-arm, five-seed and three-regime cell and all six contrasts, or one tombstone. Do not publish selective previews.

## 11. Requirements before local lock

Only these seven items are required:

1. 20 complete configs and exact naturally graph-active parameter receipts; padding parameters and catch-all gradient/activity links are prohibited;
2. deterministic split, batch and candidate manifests;
3. R/CM/Z/T fork adapters with exact four-field IPC and negative scorer read-set tests;
4. common24/native post-candidate readout and ASM-Z constraint tests;
5. metric/bootstrap/gate and INVALID-row goldens;
6. two clean structural full-shape synthetic CUDA runs with identical hashes, polled memory receipts and a separate supervisor smoke;
7. one consolidated architecture/statistics/leakage/GPU review of the exact candidate.

After 4/4 approval, create one canonical JSON `LOCAL PROTOCOL LOCK` receipt that binds the exact protocol SHA-256, candidate manifest file/content hashes, complete config/code artifact hashes and authorization SHA-256. Free-form text and minimal marker receipts are invalid. Write the receipt digest into a separately generated source anchor excluded from the candidate manifest to avoid a hash cycle. Official data/training entrypoints independently require the canonical path, validate the strict schema and complete bindings, and compare the receipt digest with that source anchor. A caller-supplied path/digest cannot substitute for the anchor. Direct unlocked non-miniature generation is prohibited.

No external ceremony or infrastructure is required.

## 12. State

Current state: `DRAFT V1 — LOCAL-ONLY — NOT LOCKED`.

After the seven local requirements pass, present exact document, config, code and manifest hashes to the user. Creating `LOCAL PROTOCOL LOCK` requires a separate explicit authorization.

Only after that lock may official data generation, training, calibration and one test opening occur.

No implementation, official data, training, calibration, test opening or lock has occurred.
