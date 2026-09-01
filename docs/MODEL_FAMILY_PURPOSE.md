# ASM Model Family: Practical Purpose and Selection Guide

## Scope

This guide answers **which ASM model to choose for a practical objective**. It complements [`MODEL_FAMILY.md`](MODEL_FAMILY.md), which defines the taxonomy, and the benchmark reports, which contain the evidence. “Promoted” always means promoted for a stated objective, not universally best.

## Quick choice

| Objective | Start with | Why | Main warning |
|---|---|---|---|
| Best language quality per parameter in the current PMCS-64 scale | **ASM-VR-S full** | Lowest matched test CE | This result is protocol- and scale-specific |
| Explicit durable associative memory | **ASM-CM** | Only matched arm to learn short MQAR in every seed | Long recall passed in only 1/3 matched seeds; validate each checkpoint |
| Logical capacity bottleneck | **ASM-VR-S fixed-32** | Half logical rank with language non-inferiority and the only completed 32K stream | Dense implementation gave no physical efficiency gain |
| Validated relational quality/token lineage | **ASM-R** | Stronger historical quality/token evidence at the larger protocol | Geometry costs and streaming behavior require separate checks |
| Simple selective state baseline | **ASM-S** | Concentrates budget in selective memory without relational geometry | It is not explicit associative memory |
| Compact execution of ASM-R weights | **ASM-C** | Bounded retained streaming cache | It failed the historical short-MQAR gate |
| Geometry or DRM research | **ASM-X, ASM-U, ASM-F, ASM-RS** | Exposes explicit geometric hypotheses | Experimental; do not treat theoretical fidelity as measured utility |
| Structural controls and ablations | **ASM-D, ASM-M** | Isolates direct state and minimal causal memory | Not promoted architectures |

## Practical selection matrix

| Variant | Practical purpose | Choose when | Main strength | Main limitation or non-claim | Status |
|---|---|---|---|---|---|
| **ASM-X** | Explicit DRM state-model research | You need directions, metric, movement, and memory exposed in one model | Closest mapping to the explicit DRM formulation | The geometric cost/benefit has not justified practical promotion | Experimental/taxonomic |
| **ASM-U** | Metric-subspace movement research | You need motion constrained to `span(V)` | Strong geometric coherence | Proposed second-generation candidate without promotion evidence | Proposed/experimental |
| **ASM-F** | Metric-normalized local-frame research | You are testing explicitly relational frames | Geometrically consistent composition | Earlier generation diverged in additional seeds | Experimental |
| **ASM-R** | Relational quality per token | You want the promoted relational backbone under its validated protocol | Contextual transition conditioned by a learned metric | Its historical evidence is not directly comparable with PMCS-64 | Promoted for relational quality/token |
| **ASM-C** | Compact streaming of ASM-R weights | You need bounded retained state with the same trained weights | Streaming engineering path with small retained cache | Historical short MQAR failed; bounded cache is not associative recall | Validated streaming mechanism, experimental model |
| **ASM-C2** | Fixed-slot addressable-memory research | You need explicit bounded K/V slots | Makes read/write storage inspectable | End-to-end MQAR, ablation, parity, and regression gates remain unapproved | Experimental |
| **ASM-C2-FW** | Fast-weight associative-memory research | You are testing matrix memory and delta-rule writes | Isolated probe demonstrated strong associative capacity | Phase-separated probes do not prove causal end-to-end control | Experimental candidate |
| **ASM-CM** | Durable associative memory with bounded retained state | The task requires explicit key/value retention over long delays | Learned short MQAR in 3/3 PMCS-64 seeds; one seed generalized through 32K | PMCS-64 long recall was only 1/3 seeds; streaming failed before 32K and parity failed | Promoted for the previously validated configuration; matched configuration not re-promoted |
| **ASM-CM-VR** | Rank-aware durable associative-memory research | You need explicit memory plus a strict logical state bottleneck | Fixed-32 passed structural and seed-17 Phase-1 gates; long full/fixed passed 2/3 seeds | Seed 29 became non-finite at 32K; adaptive also passed only 2/3; dense bytes unchanged | Experimental; long promotion gate failed |
| **ASM-RS** | Explicit relational + selective composition | You need the historical R+S recipe made explicit | Cleanly separates relational and selective components | More parameters and worse quality/cost than S in its matched gate | Validated, not promoted |
| **ASM-D** | Direct-state structural control | You need a geometry-free neural-state baseline | Simple and useful for causal attribution | Lower novelty and no promotion evidence | Control |
| **ASM-S** | Selective state and efficient architectural baseline | You want preserve/forget/write behavior without relational geometry | Simple allocation of capacity to selective memory | Not DRM and not explicit durable key/value memory | Validated option |
| **ASM-VR-S** | Logical state-capacity control on ASM-S | You need full/fixed rank comparisons or rank research | Full won PMCS-64 language; fixed-32 retained quality within `+0.03 nat` and completed 32K | Current paths are dense; adaptive controller missed the fixed-rank frontier | Full/fixed validated; adaptive experimental |
| **ASM-M** | Minimal causal-memory control | You need mixer + residual + narrow selective memory without rich transition geometry | Isolates a compact causal-memory recipe | Must be compared with gated RNNs and selective SSMs; not promoted | Control/candidate |
| **ASM-VR-RS** | Variable Rank on the explicit R+S composition | You need a relational-selective rank research control | Applies the same rank interface to ASM-RS | Published evidence covers only the full-rank control, which was not promoted | Defined/experimental |

`ASM-C2-FW-LM` is a technical lineage identifier whose public promoted name is ASM-CM. `ASM-VR-R` remains a benchmark lineage/control unless it is separately formalized as a public family member.

## What Variable Rank is useful for

Variable Rank projects the state onto a prefix of a shared coordinate frame. Rank therefore controls **how many state directions can carry information** at a block boundary.

### Useful now

1. **Capacity bottleneck experiments.** Full versus fixed rank tests whether all state dimensions are needed.
2. **Architecture comparison at equal logical capacity.** The same rank policy can be applied to different backbones without changing the no-bypass rule.
3. **Regularization and stability studies.** In PMCS-64, fixed-32 stayed within `+0.0283 nat` of full language CE and was the only arm to complete the seed-17 32K stream. This is evidence for that checkpoint, not a universal stability guarantee.
4. **Controller research.** Adaptive rank can eventually allocate more directions to difficult blocks and fewer to easy blocks, if it beats fixed policies and physical execution becomes compact.
5. **Diagnostics.** Rank sweeps expose the quality frontier and reveal whether a controller is better than a constant budget.

### Three policies

- **Full rank:** safest current quality baseline; projection is logically the identity.
- **Fixed rank:** clearest practical research tool; gives an explicit, reproducible bottleneck.
- **Adaptive rank:** open research mechanism. The current input-only controller varied and received gradients, but stayed `+0.0631 nat` outside the fixed frontier in ASM-VR-S Phase 3A.2.

### What Variable Rank does not provide yet

The published implementation still executes dense tensors. Lower rank therefore does **not** guarantee:

- fewer FLOPs;
- lower VRAM;
- faster training or decoding;
- fewer retained-state bytes;
- lower energy use.

PMCS-64 confirmed this boundary. Fixed-32 and full had the same observed peak training memory (`77.9 MiB`), and fixed-32 was not at least 5% faster. Do not translate “half logical rank” into “half compute.”

A physical benefit requires compact gather/scatter or low-rank kernels, grouped execution for examples with similar rank, a cost-aware controller, and measurements on the target hardware.

## PMCS-64: matched ASM-CM versus ASM-VR-S

The new suite uses 274,058 parameters for ASM-CM and 274,135 for each ASM-VR-S arm, a `0.028%` total-count mismatch. The VR arms have 4,160 frozen controller parameters and therefore 1.49% fewer trainable parameters. See [`benchmarks/asm_cm_vs_vr_s_pmcs64/README.md`](benchmarks/asm_cm_vs_vr_s_pmcs64/README.md).

### Main measured outcomes

- **Language:** VR-S full CE `2.5168`; fixed-32 `2.5451`; CM `2.5489`.
- **MQAR short control:** CM `99.95%`; VR-S full `3.96%`; fixed-32 `2.29%`.
- **MQAR 32K:** CM mean `33.33%`, caused by one approximately perfect seed and two failed seeds; both VR-S arms `0%`.
- **Streaming state at 4K:** CM `131,584` bytes versus `320` bytes for both VR-S arms.
- **Streaming throughput at 4K:** CM `162.3` tok/s; VR-S full `1,132.3`; fixed-32 `1,135.5`.
- **32K streaming:** CM failed at token 15,200; VR-S full failed at 30,335; fixed-32 completed.

These results delimit purpose rather than naming an overall winner:

- VR-S full is the stronger matched language model.
- CM has the explicit associative mechanism, but its long-memory and streaming robustness were seed/configuration dependent.
- Fixed-32 is the strongest logical-capacity/stability trade-off observed here, without a physical rank-speed claim.

## Status vocabulary

- **Promoted for an objective:** passed the named protocol for that objective.
- **Validated option:** implemented and measured, but not the default promoted choice.
- **Experimental:** research candidate with incomplete or failed promotion gates.
- **Control:** designed mainly to isolate a mechanism.
- **Proposed:** taxonomic/design definition without sufficient empirical evidence.

## Evidence boundaries

Always record corpus, token budget, seeds, optimizer, precision, hardware, parameter counts, test-opening rule, and artifact path. Do not compare historical scores as if they came from one matched run. Separate full-forward quality, retained-state streaming, associative recall, and physical cost.
