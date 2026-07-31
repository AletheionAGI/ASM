# Prior Art

This project explores DRM Language Emitter, an experimental non-Transformer language model based on Directional Relational Manifold dynamics.

## Related Areas

- Recurrent neural networks and state-space language models.
- Neural ordinary differential equations and learned dynamical systems.
- Energy-based and action-regularized learning.
- Natural gradient methods and metric-aware optimization.
- Riemannian and differential-geometric machine learning.
- Manifold learning and latent trajectory modeling.
- Non-attention autoregressive sequence models.

## DRM-Specific Position

DRM Language Emitter is intended as a research implementation of language generation through:

- active directional fields;
- variable effective dimension;
- learned relational metrics;
- metric action diagnostics;
- causal token emission from a latent trajectory.

The project does not claim that these ideas have no predecessors. It claims only that this repository implements a specific experimental architecture under the name DRM Language Emitter.

## Relationship to Mamba and Selective SSMs

The original DRM Language Emitter was conceived as a geometric latent-dynamics
architecture built around:

- a learned directional field;
- an SPD relational metric;
- metric-preconditioned nonlinear flow;
- latent-state trajectories;
- experimental geodesic, energy, risk, and fixed-point mechanisms.

According to the author's development chronology, this original DRM concept
was developed before the author became aware of Mamba. This statement describes
the provenance of this project; it is not a claim that DRM predates Mamba, that
all related mechanisms are unprecedented, or that independent conception alone
establishes legal or scientific priority.

### Earlier projects in the author's DRM line

DRM Language Emitter was not the author's first public implementation involving
Directional Relational Manifold ideas. Two earlier repositories document
intermediate stages of the research:

- [`gnai-creator/drm_transformer`](https://github.com/gnai-creator/drm_transformer)
  retained a decoder-only Transformer backbone while modifying attention with
  learned manifold coordinates, an SPD low-rank metric, geometry-aware
  distances, per-token dimensional gating, and geometric regularization.
- [`gnai-creator/aletheion-llm-v2`](https://github.com/gnai-creator/aletheion-llm-v2)
  retained a decoder-only Transformer language-model core and integrated an
  epistemic DRM subsystem. Its public architecture describes 5D manifold
  coordinates, an SPD metric tensor, a directional field, and geodesic
  diagnostics as part of per-token epistemic tomography.

These projects establish a visible development lineage from DRM-assisted
Transformer and epistemic-head experiments toward the present attempt to make
DRM dynamics themselves the causal language-model core:

```text
DRM geometry inside Transformer attention / epistemic heads
                            ↓
DRM Language Emitter as a non-attention latent-dynamics core
                            ↓
variant J: DRM core combined with selective memory
```

The earlier repositories should not be described as equivalent to the current
DRM Language Emitter. `drm_transformer` still uses attention, and
`aletheion-llm-v2` still uses Transformer blocks for its language-model
backbone. Their relevance here is evidence of continuity in the author's DRM
research direction, not evidence that they implemented Mamba, a selective SSM,
or the current emitter architecture.

Mamba starts from a different central mechanism: a selective state-space model
whose input-dependent transition parameters control which information is
preserved, written, and read. The original DRM instead asks how a latent state
should move under a learned local relational geometry.

The projects became conceptually closer only after experiments exposed weak
sample efficiency in the original DRM. Variant J then added an explicit
selective forget/write memory:

\[
m_t = f_t \odot m_{t-1} + w_t \odot c_t
\]

This addition was developed after studying Mamba and modern selective-SSM
literature and should be attributed accordingly. Variant J is therefore best
described as a hybrid:

```text
selective SSM-like memory
              +
original DRM geometric dynamics
```

It should not be described as an independently invented implementation of
Mamba. A precise historical summary is:

> DRM was conceived independently as a geometric latent-dynamics architecture.
> The author learned about Mamba later. After empirical limitations were found
> in the original DRM, selective-memory mechanisms informed by modern SSM
> literature were incorporated into variant J.

The decisive scientific question is whether DRM geometry contributes beyond
the selective-memory component. The repository therefore includes a
parameter-matched `SSM_CONTROL` that removes `DirectionField`,
`RelationalMetric`, `DRMFlow`, and `RiskField` while retaining selective memory,
the causal mixer, token-to-state residual, and emitter.

Initial seed-1 results favor complete J over this control, but they are
preliminary. A contribution from DRM geometry should be claimed only if the
advantage survives paired seeds, continuous deterministic rescoring, synthetic
associative-recall evaluation, compute accounting, and comparison with an
actual Mamba baseline.

## Non-Claims

This project does not claim:

- superiority over Transformers;
- formal proof of emergent geodesics;
- spontaneous toroidal topology;
- AGI, alignment, or safety guarantees;
- production readiness.

## Citation And Disclosure

If you compare this project against prior models or derivative work, cite this repository and clearly describe which components are reused, modified, or independently implemented.
