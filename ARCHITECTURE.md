# Architecture

ASM — Aletheion State Models is a causal state-model family built around a
latent state `z_t`. It does not use sequence attention. The repository now contains the
original recurrent DRM, blockwise approximations, selective-memory hybrids, and
geometry-free controls. These paths must be distinguished when reporting a
result.

The explicit DRM architecture is named **ASM-X**. Neutral family interfaces
live in `src/aletheion_state_models/`; `src/drm_language_emitter/` remains the
checkpoint-compatible implementation during migration. See
[docs/MODEL_FAMILY.md](docs/MODEL_FAMILY.md).

ASM-C — Aletheion Compact State Model — is an experimental inference form of
ASM-R. It emits from the latest state and retains a bounded open block instead
of the full prefix. It does not replace promoted ASM-R until the 32K validation
criteria pass. See [ASM-C architecture](docs/ARCHITECTURE_ASM_C.md).

ASM-C2 adds a fixed-capacity content-addressable memory to ASM-C. It remains an
unpromoted second-generation experiment; see
[ASM-C2 architecture](docs/ARCHITECTURE_ASM_C2.md).

ASM-C2-FW replaces the unsuccessful learned slot router with a bounded
fast-weight associative matrix. Its isolated storage control reached 100%
MQAR accuracy, but the complete learned-controller variant remains an
unpromoted candidate until the end-to-end gates pass.

## Promoted Architecture: ASM-R

ASM-R — Aletheion Relational State Model — is the promoted architecture for
quality per training token. It completed three independent 100M-token runs
with frozen-validation CE `1.344538 ± 0.000561` (population standard
deviation). Its transition path is:

```text
token -> causal state -> direct contextual transition
      -> relational metric naturalization
      -> causal mixer + token residual + selective memory
      -> emitter
```

ASM-R retains relational metric conditioning but removes the explicit
direction catalogue. ASM-F remains experimental after its generation-1 runs
diverged in both additional seeds before 70M tokens. See
[the confirmation report](docs/report/037_Confirmacao_Multiseed_100M_e_Promocao_ASM_R_2026_08_01.md).

## Current Architecture Families

| Family | Geometry | Selective memory | Purpose |
|---|---|---|---|
| Original DRM | direction field + metric + flow | no | reference implementation |
| Block-cumsum DRM | blockwise directional deltas | optional | scalable causal approximation |
| J | block-cumsum DRM | forget/write | explicit DRM reference (ASM-X) |
| J_NO_* | component removed or bypassed | forget/write | causal geometry ablations |
| SSM_CONTROL | none | widened forget/write | parameter-matched control |
| ASM-C | ASM-R relational transition | forget/write | compact streaming inference experiment |
| ASM-C2 | ASM-R relational transition | forget/write + fixed key/value slots | addressable streaming experiment |
| ASM-C2-FW | ASM-R relational transition | forget/write + fast-weight matrix | bounded associative streaming candidate |

Variant J is a hybrid. Its selective memory was added after the original DRM
showed weak sample efficiency; it must not be presented as part of the original
independent DRM conception.

## Latent State

The recurrent latent state is written as:

```math
z_t \in \mathcal{M}, \qquad z_t \approx \mathbf{z}_t \in \mathbb{R}^{d_{\text{state}}}
```

For the MVP, `z_t` is represented by a vector in `R^d_state`. This is a coordinate representation of the latent manifold, not a claim that the true geometric object is globally Euclidean.

`DRMStateInitializer` uses a learned initial state expanded to the batch. Prompt tokens then move the state through the DRM dynamics.

## DirectionField

`DirectionField(z)` returns:

- `V(z) [B, n_directions, d_state]`
- `gates a(z) [B, n_directions]`
- `dimD(z) = sum_i a_i(z)`

```math
D(z) = \{V_i(z)\}_{i=1}^{n_{\text{directions}}}, \qquad
a_i(z) \in [0, 1], \qquad
\mathrm{dim}_{\text{active}}(z) = \sum_i a_i(z)
```

The directions are not orthogonalized. Optional normalization keeps their scale controlled but does not impose an orthonormal frame. The gates define an effective local active dimension.

## RelationalMetric

The metric is:

```math
G(z) =
\mathrm{diag}(\mathrm{softplus}(d(z)) + \epsilon)
+ U(z)U(z)^\top
```

It is positive definite up to the `eps` floor and measures energy/coupling of velocities and directions:

```math
E_z(v) = v^\top G(z)v
```

`pairwise_coupling(z, V)` computes relational coupling between learned directions under `G(z)`.

```math
C_{ij}(z) = V_i(z)^\top G(z)V_j(z)
```

## DRMFlow

`DRMFlow` receives `z_t`, the current token embedding `e_t`, active directions, and gates. It emits coefficients:

```math
c_i(t) = c_i(z_t, e_t)
```

The raw velocity is a gated directional combination:

```math
\Delta z_t^{\text{raw}}
= \sum_i a_i(z_t)c_i(z_t, e_t)V_i(z_t)
```

Therefore the velocity belongs explicitly to the span of active directions.

In the default config, the raw directional velocity is naturalized by the learned metric:

```math
\Delta z_t = G(z_t)^{-1}\Delta z_t^{\text{raw}}
```

The implementation uses a damped Woodbury solve for the diagonal plus low-rank metric:

```math
\Delta z_t =
\left(G(z_t) + \lambda I\right)^{-1}\Delta z_t^{\text{raw}}
```

The naturalization strength is scheduled during training. This makes the metric part of the movement law while avoiding immediate over-conditioning.

The state update is:

```math
z_{t+1} = z_t + dt\,\Delta z_t
```

## Blockwise Causal Path

At the 125M scale, `directional_block_cumsum` divides the sequence into causal
blocks. Geometry is evaluated from the state at the beginning of each block,
token-conditioned local velocities are evaluated in parallel, and prefix
states are approximated by a cumulative sum:

```math
\tilde z_{b,t}
= z_b + \sum_{j \le t} dt\,\Delta z_{b,j}
```

Blocks remain sequential because the final state of block $b$ initializes
block $b+1$. Positions inside a block remain prefix-causal. An optional
depthwise causal convolutional mixer corrects the approximate states using only
left context.

This is an engineering approximation to the recurrent trajectory, not an exact
parallel solution of the nonlinear recurrence.

## Selective Memory in Variant J

J applies a content-dependent affine recurrence after the token residual:

```math
m_t = f_t \odot m_{t-1} + w_t \odot c_t
```

where forget, write, and candidate values depend on the previous causal state
and current token. The recurrence is evaluated with an associative affine scan
that avoids division by vanishing cumulative products.

The current J path is:

```text
blockwise directional flow
-> metric naturalization
-> causal local mixer
-> token-to-state residual
-> selective forget/write memory
-> language emitter
```

J does not instantiate `RiskField` in the current CE-only experiments.

## Geometry Ablations and Control

- `J_NO_METRIC` removes `RelationalMetric` and uses identity geometry.
- `J_NO_NATURALIZATION` retains the metric parameters and diagnostics but does
  not apply the metric inverse to the flow.
- `J_NO_DIRECTION` replaces the direction field and direction-constrained flow
  with a direct causal neural transition.
- `SSM_CONTROL` removes direction, metric, flow, and risk while retaining the
  mixer, token residual, selective memory, and emitter.

`J_NO_METRIC` and `J_NO_DIRECTION` are structural ablations and have fewer
parameters than J. `SSM_CONTROL` widens selective memory to match J's parameter
budget; it is not compute-matched and is not an implementation of Mamba.

## Action Loss

The action term is the mean metric energy of the rollout:

```math
\mathcal{L}_{\text{action}}
= \frac{1}{T}\sum_{t=1}^{T} \Delta z_t^\top G(z_t)\Delta z_t
```

This does not make the model an exact geodesic solver. It biases learned trajectories toward lower action under the current learned metric.

## Language Emitter

`LanguageEmitter(z)` is a small MLP with RMSNorm and GELU. It maps the current latent state to vocabulary logits.

```math
\ell_t = f_{\text{emit}}(z_t), \qquad
p(x_{t+1} \mid x_{\le t}) = \mathrm{softmax}(\ell_t)
```

For supervised language modeling, the primary loss is token cross entropy:

```math
\mathcal{L}_{\text{CE}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p(x_{t+1} \mid x_{\le t})
```

The training objective can combine token prediction with geometric
regularization:

```math
\mathcal{L}
= \mathcal{L}_{\text{CE}}
+ \lambda_{\text{action}}\mathcal{L}_{\text{action}}
+ \sum_k \lambda_k \mathcal{R}_k
```

The regularizers `R_k` include the active-fraction target, dimension variance,
metric conditioning/diversity terms, recurrence/stability proxies, and optional
risk/metric-floor penalties when enabled by config. Variant J and its current
component ablations set the geometric auxiliary weights to zero, so those runs
optimize next-token CE only.

## Generation

The current `generation.py` helper warms `z` with prompt tokens through the
original recurrent geometry. Then it repeatedly:

1. emits logits from `z`,
2. samples the next token,
3. embeds that token,
4. updates `z` through `DirectionField`, `RelationalMetric`, and `DRMFlow`.

There is no attention cache.

This helper does **not** yet reproduce block-cumsum, local-mixer, selective
memory, component-ablation, or SSM_CONTROL semantics. Generation from J-family
checkpoints must not be presented as faithful until the helper is unified with
the training forward path.

## Why It Is Not A Transformer

The project does not instantiate `nn.MultiheadAttention`, does not construct query/key/value projections, and does not run pairwise token attention. Sequence history is compressed into the trajectory state `z_t`.

## Geodesic Emergence

A geodesic in the full DRM sense would minimize an action functional over admissible curves whose velocities remain in `span(D(z))`. The MVP provides a training pressure and diagnostics for low-action trajectories. It does not solve the boundary-value geodesic problem exactly.

## Toroidal Topology

The optional toroidal utility represents circular coordinates as `(cos theta, sin theta)`. The code does not claim spontaneous toroidal convergence. Such a claim would require boundedness, recurrence, structural stability, and empirical diagnostics.
