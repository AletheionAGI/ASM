# Aletheion State Models: Architecture Family and Naming Criteria

## 1. Purpose

DRM Language Emitter began as a partial computational translation of
Directional Relational Manifolds. Recent ablations produced architectures that
retain different parts of that proposal.

This document separates three levels of identity:

```text
Directional Relational Manifolds
        theory
          ↓
Aletheion State Models
    research program
          ↓
ASM-X, ASM-R, ASM-C, ASM-C2, ASM-C2-FW, ASM-S, ASM-F...
 architectural variants
```

The primary architecture will be selected through reproducible evidence rather
than the repository's historical name.

## 2. Three levels of identity

### 2.1 Theory: Directional Relational Manifolds

DRM remains the mathematical and philosophical theory concerned with local
states and relations, accessible directions, relational metrics, transport,
path dependence, rank transitions, effective dimension, and transformation of
the set of possible futures.

The current software does not realize the complete formal theory. In
particular, its SPD metric has fixed exact rank. Gate activity and numerical
rank must not be presented as the variable formal rank proposed by DRM.

### 2.2 Research program: Aletheion State Models

**Aletheion State Models**, abbreviated **ASM**, is the umbrella for causal
models that compress history into a persistent state and learn how that state
evolves.

ASM may contain geometric, directional, selective, or purely neural variants.
A DRM hypothesis can therefore be rejected without invalidating the entire
research program.

### 2.3 Promoted architecture

The promoted architecture is the variant that demonstrates the best combined
evidence across cross-entropy, data scaling, quality per GPU hour, throughput,
seed stability, memory, associative recall, long context, controllability, and
observability.

It may receive its own public name after the scaling-law study and additional
seed confirmation.

## 3. Proposed taxonomy

| Code | Name | Architectural core |
|---|---|---|
| ASM-X | Explicit DRM State Model | explicit directions + metric + movement + memory |
| ASM-U | Metric Subspace State Model | naturalized movement inside the subspace |
| ASM-F | Relational Frame State Model | directional frame normalized by the metric |
| ASM-R | Relational State Model | direct transition conditioned by the metric |
| ASM-C | Compact State Model | ASM-R weights with bounded streaming inference state |
| ASM-C2 | Compact Addressable State Model | ASM-C with bounded key/value memory slots |
| ASM-C2-FW | Compact Fast-Weight State Model | ASM-C with a bounded delta-rule associative matrix |
| ASM-CM | Aletheion Compact Memory Model | promoted public identity of ASM-C2-FW-LM |
| ASM-D | Direct State Model | direct neural transition without geometry |
| ASM-S | Selective State Model | capacity concentrated in selective memory |
| ASM-M | Causal Memory State Model | mixer, residual, and narrow selective memory |

The letters identify mechanisms, not quality rankings.

## 4. ASM-X — Explicit DRM State Model

`ASM-X` represents explicit DRM. **X** marks the explicit factorization of the
dynamics into directions, gates, coefficients, and geometry.

```text
token
  → causal state
  → directional field
  → gates and coefficients
  → directional movement
  → metric naturalization
  → selective memory and mixer
  → emitter
```

In compact notation:

$$
v_{\mathrm{raw}}=Vc,
\qquad
v=G^{-1}Vc.
$$

`ASM-X` is the taxonomic successor to J and the variant closest to the original
DRM Language Emitter identity.

The public name **DRM Language Emitter** should remain primary only if direction
and metric show a positive contribution at scale or on capabilities that
justify their cost.

## 5. ASM-U — Metric Subspace State Model

`ASM-U` retains the directional field while making the metric act inside the
possibility subspace.

Assuming directions are columns of $V$:

$$
\hat c=(V^\top GV+\lambda I)^{-1}c,
$$

$$
v=V\hat c,
$$

so that:

$$
v\in\mathrm{span}(V).
$$

Suggested public name:

> DRM Subspace Emitter

This would constitute a legitimate second-generation DRM architecture: geometry
organizes action without moving it outside the declared directional space.

## 6. ASM-F — Relational Frame State Model

`ASM-F` transforms directions into a local frame conditioned by the metric.
With directions represented as columns of $V$:

$$
V^\top GV+\lambda I=LL^\top,
$$

$$
Q=VL^{-\top},
$$

and, for small damping:

$$
Q^\top GQ\approx I.
$$

The code stores directions by rows and performs the equivalent transposed
operation. This convention must be stated to avoid the dimensionally incorrect
$L^{-1}V$ formula when $V$ is defined by columns.

Suggested public name:

> Relational Frame Emitter

Definition:

> A causal model whose transition possibilities are represented in a local
> frame normalized by learned geometry.

## 7. ASM-R — Relational State Model

`ASM-R` corresponds to `J_NO_DIRECTION`. It removes the explicit catalogue of
directions and produces movement through a contextual transition:

$$
v_{\mathrm{raw}}=T(z,x),
$$

$$
v=G^{-1}T(z,x).
$$

Movement remains directional in the sense that it is a vector, but its
directions are implicit in $T$. The relational metric still conditions the
update.

Suggested public name:

> Relational State Emitter

Technical subtitle:

> A metric-conditioned causal state model derived from DRM research.

`ASM-R` is the promoted quality-per-token architecture. Across three
independent 100M-token runs, its frozen-validation CE was `1.344538 ±
0.000561` (population standard deviation). It loses to the selective control
at 5M tokens, overtakes it by 30M, and maintains a reproducible trajectory to
100M.

### 7.1 ASM-C — Aletheion Compact State Model

`ASM-C` is the compact-streaming inference form of ASM-R. It reuses the same
trained parameters and relational transition, but retains only a token counter,
the completed state, and a bounded open block. Its emitter consumes only the
latest state instead of allocating activations proportional to the full prefix.

The name **Compact** describes the implemented mechanism without prematurely
claiming constant memory. ASM-C remains experimental until real-checkpoint BF16
parity, bounded cache and peak VRAM, stable 4K–32K throughput, and corrected
MQAR retention satisfy the criteria in
`report/044_Plano_Implementacao_ASM_C_Streaming_Constante_2026_08_01.md`.

The first completed 32K validation passed all three streaming-engineering
criteria, including a constant 6,144-byte retained cache and 99.6% throughput
retention from 4K to 32K. It failed the short MQAR control (32.25% versus the
80% gate), so compact execution is validated but associative retention is not.
ASM-C remains experimental.

### 7.2 ASM-C2 — Aletheion Compact Addressable State Model

ASM-C2 adds fixed-capacity, content-addressable key/value slots to ASM-C. It is
designed to preserve bounded streaming while supporting explicit retrieval.
It uses attention-like addressing over fixed slots, not token-prefix
self-attention. ASM-C2 is unpromoted until MQAR, ablation, streaming, parity,
and language-regression gates pass.

### 7.3 ASM-C2-FW — Aletheion Compact Fast-Weight State Model

ASM-C2-FW is the corrected associative-memory candidate. It removes learned
discrete slot agreement and uses one shared key projection for writing and
reading a fixed-size matrix. A gated delta rule writes the residual between a
candidate value and the value already predicted by that key. Read, write, and
retention gates are learned causally from the current state, current token, and
previous token.

The isolated, phase-controlled fast-weight probe reached 100% MQAR accuracy,
whereas dense learned-slot memory reached 12.18% after 10,000 steps. This
authorizes end-to-end testing, not promotion: the isolated probe was told when
to write and read, while ASM-C2-FW must learn that control from causal inputs.

The durable experimental form separates fast working memory from a slow
consolidated matrix and trains with delayed-MQAR curriculum plus language
replay. It is not a new promoted family code; it is the next validation stage
of ASM-C2-FW.

### 7.4 ASM-CM — promoted language-compatible compact memory

ASM-C2-FW-LM starts from a 100M-token ASM-R checkpoint and specializes the
compact fast-weight architecture with approximately 80% language replay and
20% delayed MQAR. It uses a lower learning rate for the language backbone, a
higher rate for fast-weight memory and gates, frozen ASM-R logit distillation,
and FP32 computation in the numerically sensitive memory recurrence.

The initial three-seed specialization suite passed every internal gate. Its
seed-1 frozen-language CE improved from `1.342003` to `1.326738`; MQAR reached
100% at 512, 4K, and 32K tokens; retained cache remained `0.1367 MiB`; and the
BF16 argmax mismatch rate was `0.195%`. Independent confirmation and frozen
post-FP32 revalidation then passed every gate across three distinct ASM-R
lineages. The promoted public name is **ASM-CM — Aletheion Compact Memory
Model**; `ASM-C2-FW-LM` remains the exact technical lineage identifier. Final
mean CE was `1.328496 ± 0.000687`; at 32K, cache remained `143,360 bytes`, peak
VRAM `363.66 MiB`, and mean throughput `80.68 tok/s`. Transformer CE remains
lower, so promotion is specifically for durable associative memory, bounded
state, and language compatibility.

## 8. ASM-D — Direct State Model

`ASM-D` is the structural direct control without geometry:

```text
token
  → state
  → direct contextual transition
  → selective memory
  → causal mixer
  → emitter
```

Suggested public name:

> Causal State Emitter

This promotion would be appropriate if direction, metric, and naturalization do
not justify their cost even before parameter redistribution.

## 9. ASM-S — Selective State Model

`ASM-S` corresponds to `J_DIRECT_CONTROL_MATCHED`. Geometry is removed and its
budget is redistributed to selective memory:

$$
m_t=f_t\odot m_{t-1}+w_t\odot c_t.
$$

The model preserves, forgets, writes, transforms, and emits. **State** is
therefore more accurate than only **Memory**.

Suggested public name:

> Selective State Emitter

This name becomes appropriate if the scaling law shows that allocating
capacity to memory is more useful than modeling geometry.

## 10. ASM-M — Causal Memory State Model

`ASM-M` corresponds to the narrow `SSM_CONTROL`: causal mixer, residual,
selective memory, and emitter without geometry or a rich contextual transition.

Suggested public name:

> Causal Memory Emitter

If it wins, its novelty must be compared carefully against selective SSMs,
gated RNNs, and other modern memory architectures.

## 11. Scaling-law decision

Continuous evaluation uses checkpoints at:

```text
1M, 2M, 5M, 10M, 20M, 30M, 50M, and 100M tokens
```

An exploratory curve for each architecture may be represented as:

$$
L(N)=L_\infty+AN^{-\alpha}.
$$

The decision must consider matched-checkpoint CE, observed crossovers, estimated
scaling exponent, CE per GPU hour, throughput, memory use, complementary
capabilities, and three-seed confirmation of the finalists.

## 12. Promotion and public naming matrix

| Confirmed result | Code | Recommended public name |
|---|---|---|
| J wins | ASM-X | DRM Language Emitter |
| Metric Subspace wins | ASM-U | DRM Subspace Emitter |
| Metric Orthonormal wins | ASM-F | Relational Frame Emitter |
| No Direction wins | ASM-R | Relational State Emitter |
| Direct Control wins | ASM-D | Causal State Emitter |
| Direct Matched wins | ASM-S | Selective State Emitter |
| SSM Control wins | ASM-M | Causal Memory Emitter |

## 13. Interpretation scenarios

If `ASM-X` starts behind and wins at scale, the initial cost of learning
geometry is part of its scaling behavior and DRM Language Emitter may remain.

If `ASM-U` or `ASM-F` wins, the failure was not direction itself but an
incompatible composition of direction and metric. This would be a correction
of the original implementation and a reformulated strengthening of DRM.

If `ASM-R` wins, DRM remains the theoretical origin but leaves the primary
model name. The project becomes Relational State Emitter.

If `ASM-D`, `ASM-S`, or `ASM-M` wins, the promoted architecture is no longer
DRM. The theory remains documented and `ASM-X` remains an experimental variant.

Different variants may also win under different objectives. The ASM family can
promote separate models for early efficiency, quality at scale, long context,
or controllability.

## 14. Recommended repository evolution

The repository should not be renamed before experimental confirmation. After
promotion, the internal organization may evolve toward:

```text
src/aletheion_state_models/
├── core/
│   ├── state_model.py
│   ├── transition.py
│   ├── memory.py
│   ├── mixer.py
│   └── emitter.py
├── geometry/
│   ├── metric.py
│   ├── directional_basis.py
│   └── naturalization.py
└── variants/
    ├── explicit_drm.py
    ├── metric_subspace.py
    ├── relational_state.py
    ├── direct_state.py
    └── selective_state.py
```

ASM-F generation 1 remains in the shared geometric implementation and
experimental matrix. It diverged before 70M tokens in both additional seeds,
so its stabilized factorization is classified as a second-generation research
line rather than a validated competitor to the promoted ASM-R.

Recommended neutral interfaces include `StateModel`, `StateTransition`,
`StateMemory`, `CausalMixer`, `GeometryOperator`, `DirectionalBasis`, and
`TokenEmitter`.

Legacy names should remain temporarily as deprecated aliases to avoid an
immediate API break.

## 15. Future documentation structure

After promotion, documentation should be separated into:

- `THEORY.md`: DRM theory independently of the winning model;
- `MODEL_FAMILY.md`: ASM and its variants;
- `EXPERIMENTAL_EVIDENCE.md`: results, seeds, compute, and scaling laws;
- `ARCHITECTURE.md`: only the promoted architecture;
- `HISTORY.md`: DRM origin and evidence-driven evolution.

## 16. Current status

The 100M multiseed confirmation is complete. ASM prevents the historical name
from pressuring the scientific decision, while `ASM-X` explicitly preserves
DRM inside the family.

The promoted architecture is:

> **ASM-CM — Aletheion Compact Memory Model**

with the public architecture name:

> **Relational State Emitter**

This promotion applies to quality per token under the current Wikipedia
byte-level protocol. ASM-S remains the efficiency-oriented option, and a
stabilized ASM-F remains a second-generation geometric experiment. See
[report 037](report/037_Confirmacao_Multiseed_100M_e_Promocao_ASM_R_2026_08_01.md).
