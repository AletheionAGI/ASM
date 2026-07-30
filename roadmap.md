# DRM Formal Implementation Roadmap

Original date: 2026-07-11
Last aligned with repository: 2026-07-30

## Project Status on 2026-07-30

The project is currently in **Gate 0 — empirical validation and runtime
correctness**, before Phase 1 of the formal DRM roadmap.

The active milestone is the frozen independent 125M benchmark:

- document-disjoint Wikipedia train/validation manifests are prepared;
- the external PG-19 test split is frozen and has not been consulted by the
  training launcher;
- exact-block contamination audits passed with zero overlaps;
- the CUDA smoke test passed on one RTX 4090;
- three DRM and three GPT-2 runs at 150M tokens each are in progress;
- final PG-19 evaluation remains pending until best checkpoints are selected
  using validation CE.

Formal roadmap status:

| Gate/phase | Status | Repository evidence |
|---|---|---|
| Gate 0 — benchmark and runtime correctness | In progress | Frozen protocol, independent manifests, active six-run benchmark |
| Phase 1 — metric rank, kernel and strata | Not started | Current SPD metric has full mathematical rank |
| Phase 2 — anchor and admissible motion | Not started | Existing “anchor” variables are solver proposals, not `rho_p` |
| Phase 3 — stratumwise connection and transport | Not started | No connection or formal parallel transport |
| Phase 4 — transition maps and energy criteria | Not started | No explicit interstratum `J` maps |
| Phase 5 — hybrid holonomy and hysteresis | Not started | No ordered hybrid loop operators/invariants |
| Phase 6 — reduction diagnostics | Not started | No Riemannian/sub-Riemannian/Fisher identification |

Gate 0 is complete only after:

1. all six frozen runs finish and their best validation checkpoints are fixed;
2. each selected checkpoint is evaluated once on PG-19;
3. results and uncertainty across seeds are published;
4. generation is made faithful to the trained sequence mode/local mixer;
5. inert public configuration flags are implemented or rejected explicitly.

This roadmap describes the next formal DRM implementation layers needed to bring
`drm-language-emitter` closer to the mathematical article:

`DRM: Variedades Relacionais Direcionais - Metrica Relacional, Transporte, Curvatura e Fechamento Toroidal Condicional`.

The current code is already a practical neural DRM-inspired language model. It
implements directional latent dynamics, gates, a learned relational metric,
metric naturalization, state updates, and geometry diagnostics. It does not yet
implement the full formal structure of relational transport, covariant
connection, holonomy, toroidal closure, Fisher-Rao pullback, or an explicit
anchor map.

The recommended strategy is incremental:

1. Implement formal features first as diagnostics.
2. Validate that diagnostics are numerically stable.
3. Only then expose optional loss terms or training-time behavior.
4. Keep the current language-model benchmark path working at all times.

## Current Implementation Baseline

Implemented today:

- `DRMEmitterModel`: recurrent latent state model over token sequences.
- `DirectionField`: learns active directions `D(z)` and gates.
- `DRMFlow`: computes coefficients and velocity `dz` in the span of active directions.
- `RelationalMetric`: learns `G(z) = diag(z) + U(z)U(z)^T`.
- `RelationalMetric.naturalize`: applies Woodbury-style `G^{-1}` preconditioning.
- `StateUpdater`: updates `z <- z + dt * dz` with optional bounded state.
- Geometry diagnostics: active fractions, gate quantiles, action proxy, condition proxy, metric norms, recurrence and stability proxies.
- Causal block/superblock cumsum sequence paths and optional causal local mixer.
- Causal Anderson/fixed-point experimental paths and causality regression tests.
- Memory-mapped byte-token datasets with shard integrity verification.
- Safe weights-only checkpoint loading with payload/config validation.
- Independent train/validation/test preparation and frozen-test evaluation
  infrastructure.

Important current limitations:

- Directions live directly in latent state space; there is no explicit anchor `rho_p`.
- There is no relational transport operator `P^gamma_{s -> t}`.
- There is no formal covariant derivative `nabla^D`.
- Curvature and holonomy are not computed.
- `use_toroidal_state` exists but does not change the dynamics.
- The metric is numerically positive definite by construction, not semidefinite with an actual kernel.
- Consequently, its exact rank is always `d_state`; current `dimD` is the sum
  of directional gates and is **not** the paper's
  `d_DRM(p) = rank(g_p)`.
- Fisher-Rao geometry is mentioned as a future connection but not implemented.
- Autoregressive `generation.py` still advances with the basic recurrent
  transition and does not reproduce the block-cumsum/local-mixer forward path.
- `tie_embeddings` is accepted by `DRMConfig` but is not wired into the model.
- `geometry_report` uses same-token targets when none are supplied; its default
  CE must not be interpreted as next-token CE.

## Roadmap Summary

| Phase | Feature | Status | Priority | First integration mode |
|---:|---|---|---|---|
| 0 | Independent validation and runtime fidelity | In progress | Critical | Frozen benchmark + correctness fixes |
| 1 | Metric rank, soft kernel and rank strata | Not started | Critical | Evaluation-only, then optional PSD path |
| 2 | Explicit anchor `rho` and admissible motion | Not started | High | Identity/default diagnostic path |
| 3 | Stratumwise connection and metric transport | Not started | High | Evaluation-only |
| 4 | Interstratum transition maps and energy defects | Not started | High | Evaluation-only |
| 5 | Ordered hybrid holonomy and hysteresis invariants | Not started | High | Evaluation-only |
| 6 | Riemannian/sub-Riemannian/Fisher reductions | Not started | Medium | Diagnostic identification |

Conditional toroidal closure is an optional appendix track, not a core phase:
the paper explicitly states that boundedness and recurrence do not imply a
torus. It must not be inferred from `bounded_state`.

## Alignment with `docs/paper/drm_v6.tex`

The paper's dependency chain is authoritative for formal implementation:

```text
PSD metric
→ kernel and quotient effective fiber
→ constant-rank strata
→ kernel-compatible anchor
→ stratumwise connection/transport
→ explicit rank-transition maps
→ ordered hybrid holonomy/hysteresis
```

The neural implementation currently covers only an engineering analogue of
the ambient state, directional dynamics, a strictly positive-definite metric
and an action-like energy. It does not yet instantiate the formal tuple

```text
(M, E, g, rho, {S_alpha}, {nabla_alpha}, {J_e}).
```

Terminology must therefore remain precise:

- gate activity is not metric rank;
- low metric condition is not a kernel or quotient fiber;
- causal Anderson/local mixing is not relational parallel transport;
- recurrence proxies are not closed-loop holonomy;
- bounded latent states are not toroidal closure;
- Fisher pullback diagnostics would be an empirical extension, while the
  paper itself proves a conditional Fisher–Rao reduction.

The detailed phase sketches below predate `drm_v6.tex`. They remain useful
implementation notes, but their old numbering/order is superseded by the
paper-aligned summary above. In particular, the old “Phase 3 — Effective Rank”
must be implemented before the old transport/holonomy sketches.

---

# Phase 1 - Relational Transport Diagnostics

## Meaning

In the article, relational transport compares internal directions at different
states along a trajectory:

```text
P^gamma_{s -> t}: E_gamma(s)^act -> E_gamma(t)^act
```

In the current code, `DirectionField(z)` creates a fresh set of directions at
each state. There is no explicit rule for saying whether direction `i` at step
`t` is "the same" direction as direction `i` at step `t + 1`.

Transport will let us measure:

- whether direction fields preserve identity over time;
- whether gates are stable or chaotic;
- whether the geometry drifts smoothly;
- how much an active basis rotates along a sequence.

## Current State

Current direction field:

```python
directions, gates = model.direction_field(z)
```

No transport operator exists.

## Proposed Change

Add a new module:

```text
src/drm_language_emitter/transport.py
```

Initial implementation should be diagnostic-only. It should not change training.

The first transport approximation can align direction bases between adjacent
states by cosine similarity / Procrustes-style matching.

## Implementation Sketch

```python
# src/drm_language_emitter/transport.py
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RelationalTransport(nn.Module):
    """Diagnostic transport between local direction frames."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def pairwise_alignment(self, directions_a: torch.Tensor, directions_b: torch.Tensor) -> torch.Tensor:
        """Return cosine alignment matrix between two direction frames.

        directions_a: [batch, n_directions, d_state]
        directions_b: [batch, n_directions, d_state]
        returns: [batch, n_directions, n_directions]
        """
        a = F.normalize(directions_a, dim=-1, eps=self.eps)
        b = F.normalize(directions_b, dim=-1, eps=self.eps)
        return torch.bmm(a, b.transpose(1, 2))

    def soft_transport_matrix(self, directions_a: torch.Tensor, directions_b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """Soft assignment matrix transporting frame A into frame B."""
        scores = self.pairwise_alignment(directions_a, directions_b)
        return torch.softmax(scores / max(temperature, self.eps), dim=-1)

    def transport_gates(self, gates_a: torch.Tensor, transport: torch.Tensor) -> torch.Tensor:
        """Move gates from frame A into frame B coordinates."""
        return torch.bmm(gates_a.unsqueeze(1), transport).squeeze(1)

    def drift(self, directions_a: torch.Tensor, directions_b: torch.Tensor) -> torch.Tensor:
        """Frame drift after best soft alignment. Lower is more stable."""
        transport = self.soft_transport_matrix(directions_a, directions_b)
        transported = torch.bmm(transport, directions_b)
        return (F.normalize(directions_a, dim=-1) - F.normalize(transported, dim=-1)).pow(2).mean()
```

## Model Integration

Add to `DRMEmitterModel.__init__`:

```python
from .transport import RelationalTransport

self.transport = RelationalTransport()
```

During `collect_diagnostics=True`, keep previous and current direction frames:

```python
if collect_diagnostics and prev_directions is not None:
    transport = self.transport.soft_transport_matrix(prev_directions, directions)
    transported_prev_gates = self.transport.transport_gates(prev_gates, transport)
    transport_drift_values.append(self.transport.drift(prev_directions, directions))
    gate_transport_error_values.append((transported_prev_gates - gates).abs().mean(dim=-1))

prev_directions = directions.detach()
prev_gates = gates.detach()
```

Add diagnostics:

```python
"transport_drift": torch.stack(transport_drift_values).mean(),
"gate_transport_error": torch.stack(gate_transport_error_values).mean(),
```

## Tests

Add:

```text
tests/test_transport.py
```

Test cases:

- identical frames produce low drift;
- permuted frames can still align;
- transported gates preserve shape;
- diagnostics are finite.

## Success Criteria

- No training path changes.
- Geometry report includes `transport_drift` and `gate_transport_error`.
- Diagnostics remain finite on CPU and CUDA.

---

# Phase 2 - Curvature and Holonomy Diagnostics

## Meaning

Curvature measures whether relational transport depends on the path. Holonomy
measures what happens when a direction is transported around a loop and returns
to the starting state.

In simple terms:

```text
If the model state makes a loop, do its internal directions return unchanged?
```

This is highly aligned with the article because it gives measurable content to:

- relational curvature;
- finite loop holonomy;
- non-commutativity of directional transport.

## Current State

There is a low-action / geodesic-like diagnostic in:

```text
scripts/eval_geodesic_paths.py
```

But it explicitly says it is not a formal geodesic solver. There is no
holonomy or curvature diagnostic.

## Proposed Change

Add:

```text
src/drm_language_emitter/curvature.py
scripts/eval_holonomy.py
```

Start with finite-loop holonomy diagnostics, not infinitesimal curvature.
Finite loops are easier and more robust in neural systems.

## Implementation Sketch

```python
# src/drm_language_emitter/curvature.py
from __future__ import annotations

import torch

from .transport import RelationalTransport


class HolonomyDiagnostics:
    def __init__(self, transport: RelationalTransport):
        self.transport = transport

    def compose_transport(self, frames: list[torch.Tensor]) -> torch.Tensor:
        """Compose soft transport matrices around a trajectory of frames."""
        if len(frames) < 2:
            raise ValueError("at least two frames are required")
        batch, n, _ = frames[0].shape
        total = torch.eye(n, device=frames[0].device, dtype=frames[0].dtype).expand(batch, n, n).clone()
        for a, b in zip(frames[:-1], frames[1:]):
            step = self.transport.soft_transport_matrix(a, b)
            total = torch.bmm(total, step)
        return total

    def holonomy_error(self, frames: list[torch.Tensor]) -> torch.Tensor:
        """Return ||P_loop - I|| for a closed or approximately closed loop."""
        total = self.compose_transport(frames)
        n = total.shape[-1]
        identity = torch.eye(n, device=total.device, dtype=total.dtype).expand_as(total)
        return (total - identity).pow(2).mean()
```

## Evaluation Script Sketch

```python
# scripts/eval_holonomy.py
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.curvature import HolonomyDiagnostics
from drm_language_emitter.tokenizer import load_tokenizer
from drm_language_emitter.utils import save_json


@torch.no_grad()
def collect_frames(model, ids: torch.Tensor):
    z = model.initializer(ids.shape[0], ids.device)
    frames = []
    for t in range(ids.shape[1]):
        directions, gates = model.direction_field(z)
        frames.append(directions)
        e_t = model.token_embedding(ids[:, t])
        metric_diag, metric_u = model.metric(z)
        dz_raw, _ = model.flow(z, e_t, directions, gates)
        dz = model.metric.naturalize(dz_raw, metric_diag, metric_u)
        z = model.updater(z, dz)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="DRM relational loop DRM relational loop")
    parser.add_argument("--output", default="runs/holonomy/holonomy.json")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    tokenizer = load_tokenizer(args.tokenizer)
    ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    frames = collect_frames(model, ids)
    diag = HolonomyDiagnostics(model.transport)
    result = {"holonomy_error": diag.holonomy_error(frames).item()}
    save_json(args.output, result)
    print(f"saved={Path(args.output)}")


if __name__ == "__main__":
    main()
```

## Tests

Add:

```text
tests/test_curvature.py
```

Test cases:

- identical repeated frames have near-zero holonomy error;
- randomly changing frames have finite nonzero holonomy error;
- composed transport shape is `[batch, n_directions, n_directions]`.

## Success Criteria

- `eval_holonomy.py` produces `holonomy_error`.
- The diagnostic can be plotted over checkpoints.
- No training behavior changes.

---

# Phase 3 - Effective Rank and Soft Kernel

## Meaning

The article allows the relational metric to be positive semidefinite. Its
kernel represents collapsed, inactive, or invisible directions. The local
effective dimensionality is:

```text
dim_D(p) = rank(g_p)
```

Current code avoids singular metrics for numerical stability:

```python
diag = softplus(...) + eps
G = diag + U U^T
```

This is effectively positive definite. It is stable, but it does not produce a
true kernel.

## Current State

Current diagnostics approximate dimensionality mainly through gates:

- `dimD_mean`;
- `soft_active_fraction`;
- hard active fractions at gate thresholds.

This is useful, but it is not the same as metric rank.

## Proposed Change

Implement effective-rank diagnostics first:

- entropy effective rank of metric spectrum;
- participation ratio;
- soft kernel mass below a threshold;
- effective rank of gate vector.

Do not make the metric singular during training yet.

## Implementation Sketch

Add to `RelationalMetric`:

```python
def dense_matrix(self, metric_diag: torch.Tensor, metric_u: torch.Tensor) -> torch.Tensor:
    """Build dense G for diagnostics only. Avoid in hot training path."""
    g = torch.diag_embed(metric_diag)
    if metric_u.shape[-1] > 0:
        g = g + torch.bmm(metric_u, metric_u.transpose(1, 2))
    return g

@staticmethod
def effective_rank_from_eigenvalues(eigenvalues: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = eigenvalues.clamp_min(0)
    probs = values / values.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    return entropy.exp()

@staticmethod
def participation_rank(eigenvalues: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    values = eigenvalues.clamp_min(0)
    return values.sum(dim=-1).pow(2) / values.pow(2).sum(dim=-1).clamp_min(eps)

def rank_diagnostics(self, metric_diag: torch.Tensor, metric_u: torch.Tensor, threshold: float = 1e-4) -> dict[str, torch.Tensor]:
    g = self.dense_matrix(metric_diag, metric_u)
    eig = torch.linalg.eigvalsh(g)
    return {
        "metric_effective_rank": self.effective_rank_from_eigenvalues(eig).mean(),
        "metric_participation_rank": self.participation_rank(eig).mean(),
        "metric_kernel_mass": (eig < threshold).float().mean(),
        "metric_min_eigenvalue": eig.min(),
        "metric_max_eigenvalue": eig.max(),
    }
```

Because dense eigendecomposition is expensive, call this only during
`collect_diagnostics=True`, and optionally on a reduced subset of steps.

## Optional Later Training Regularizer

Once diagnostics are stable:

```python
rank_target_loss = (metric_effective_rank - target_rank).pow(2)
```

This should be optional and disabled by default.

## Tests

Add cases:

- identity matrix has effective rank close to `d_state`;
- rank-one low-rank matrix has low participation rank;
- diagnostics are finite for `metric_rank=0`.

## Success Criteria

- Report includes metric effective rank.
- This gives a stronger measurement of emergent dimension than gate count alone.
- No singular metric is introduced in training yet.

---

# Phase 4 - Approximate Fisher-Rao Pullback Metric

## Meaning

Fisher-Rao geometry measures distances between probability distributions. For
language modeling, this is attractive because the model emits a distribution:

```text
p(next token | z) = softmax(logits(z))
```

A Fisher-style pullback metric would connect latent geometry directly to
changes in output distribution.

Formal idea:

```text
G_F(z) = J(z)^T F_output(z) J(z)
```

where `J` is the Jacobian of logits/probabilities with respect to `z`.

## Current State

The current metric is learned from `z`:

```python
metric_diag, metric_u = model.metric(z)
```

It is not derived from the emitted token distribution.

## Proposed Change

Start with an approximate diagnostic:

- sample a small number of random directions in latent space;
- measure KL change in output distribution under small perturbations;
- estimate local sensitivity / Fisher-like energy.

Avoid full Jacobians initially.

## Implementation Sketch

Add:

```text
src/drm_language_emitter/fisher.py
```

```python
from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def fisher_directional_energy(model, z: torch.Tensor, directions: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Approximate Fisher energy by symmetric KL under small latent perturbations.

    z: [batch, d_state]
    directions: [batch, n_probe, d_state]
    returns: [batch, n_probe]
    """
    base_logits = model.emitter(z)
    base_logp = F.log_softmax(base_logits, dim=-1)
    base_p = base_logp.exp()
    energies = []
    for i in range(directions.shape[1]):
        v = directions[:, i]
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        plus_logp = F.log_softmax(model.emitter(z + eps * v), dim=-1)
        minus_logp = F.log_softmax(model.emitter(z - eps * v), dim=-1)
        kl_plus = (base_p * (base_logp - plus_logp)).sum(dim=-1)
        kl_minus = (base_p * (base_logp - minus_logp)).sum(dim=-1)
        energies.append((kl_plus + kl_minus) / (eps * eps))
    return torch.stack(energies, dim=1)
```

## Diagnostic Integration

Use active DRM directions as probe directions:

```python
fisher_energy = fisher_directional_energy(model, z, directions)
diagnostics["fisher_energy_mean"] = fisher_energy.mean()
diagnostics["fisher_energy_std"] = fisher_energy.std(unbiased=False)
```

## Later Optional Metric Coupling

After diagnostic validation, add an optional loss aligning learned metric energy
with Fisher directional energy:

```python
learned_energy = model.metric.metric_energy(z, directions[:, i], metric_diag, metric_u)
loss_fisher_alignment = (learned_energy - fisher_energy[:, i]).pow(2).mean()
```

This should be off by default.

## Tests

- Fisher energy is non-negative.
- Constant emitter has near-zero Fisher energy.
- Shape matches `[batch, n_probe]`.

## Success Criteria

- DRM can report whether learned directions correspond to output-distribution sensitivity.
- This directly connects the implementation to the Fisher-Rao reduction concept.

---

# Phase 5 - Conditional Toroidal State Dynamics

## Meaning

The article states that toroidal closure is conditional. Recurrence alone does
not imply a torus. A torus is appropriate when the system has independent
periodic phases.

The code currently has:

```python
use_toroidal_state: bool = False
```

and a helper:

```python
def toroidal_pairs(theta):
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
```

But `use_toroidal_state` does not change the model dynamics.

## Current State

Toroidal state is a placeholder.

## Proposed Change

Implement optional toroidal coordinates as a subset of the latent state:

- first `n_toroidal_dims` are phase variables;
- phase variables wrap modulo `2*pi`;
- emitter receives `sin/cos` representation or augmented toroidal features;
- diagnostics measure phase recurrence.

## Config Additions

```python
toroidal_dims: int = 0
toroidal_period: float = 6.283185307179586
```

## Implementation Sketch

Modify `StateUpdater`:

```python
import math

class StateUpdater(nn.Module):
    ...
    def forward(self, z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
        z_next = z + self.config.dt * dz
        if self.config.use_toroidal_state and self.config.toroidal_dims > 0:
            n = min(self.config.toroidal_dims, z_next.shape[-1])
            period = self.config.toroidal_period
            phase = torch.remainder(z_next[..., :n] + math.pi, period) - math.pi
            z_next = torch.cat([phase, z_next[..., n:]], dim=-1)
        if self.config.bounded_state:
            ...
        return z_next
```

Potential emitter feature augmentation:

```python
def toroidal_features(z, n):
    phase = z[..., :n]
    rest = z[..., n:]
    return torch.cat([torch.cos(phase), torch.sin(phase), rest], dim=-1)
```

This requires adapting emitter input dimension if enabled.

## Safer First Version

Do not change emitter dimensions. Only wrap phase coordinates in the state
update. This is less expressive but safer.

## Tests

- phase coordinates stay in `[-pi, pi]`;
- non-toroidal coordinates remain unchanged except normal updater behavior;
- `use_toroidal_state=False` preserves current behavior.

## Success Criteria

- `use_toroidal_state=True` has a real runtime effect.
- Toroidal diagnostics can measure phase recurrence.
- The default path remains unchanged.

---

# Phase 6 - Explicit Anchor Map `rho_p`

## Meaning

In the article, the anchor maps internal relational directions to observable
state-space velocities:

```text
rho_p: E_p^act -> T_pM
```

Current code assumes directions are already expressed in latent state
coordinates. That means the anchor is implicitly the identity map.

## Current State

Current flow:

```python
dz = einsum(active_coefficients, directions)
z_next = updater(z, dz)
```

This is equivalent to:

```text
rho_p = identity
```

or, more precisely, internal directions and observable latent velocities share
the same coordinate system.

## Proposed Change

Add an optional anchor module:

```text
src/drm_language_emitter/anchor.py
```

It should map internal direction vectors to latent velocity vectors.

## Implementation Sketch

```python
from __future__ import annotations

import torch
from torch import nn

from .config import DRMConfig


class AnchorMap(nn.Module):
    """Map relational directions into observable latent velocity directions."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        self.enabled = config.anchor_hidden_size > 0
        if self.enabled:
            h = config.anchor_hidden_size
            self.net = nn.Sequential(
                nn.Linear(config.d_state * 2, h),
                nn.GELU(),
                nn.Linear(h, config.d_state),
            )

    def forward(self, z: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """Apply anchor to each direction.

        z: [batch, d_state]
        directions: [batch, n_directions, d_state]
        """
        if not self.enabled:
            return directions
        batch, n, d = directions.shape
        z_expanded = z.unsqueeze(1).expand(batch, n, d)
        anchored = self.net(torch.cat([z_expanded, directions], dim=-1))
        return anchored
```

Config:

```python
anchor_hidden_size: int = 0
anchor_residual: bool = True
```

Integrate into `DRMEmitterModel`:

```python
self.anchor = AnchorMap(config)
```

Then in forward:

```python
directions = self.anchor(z, directions)
```

## Diagnostics

Measure anchor distortion:

```python
anchor_shift = (anchored_directions - directions).norm(dim=-1).mean()
anchor_cosine = F.cosine_similarity(anchored_directions, directions, dim=-1).mean()
```

## Tests

- disabled anchor returns directions unchanged;
- enabled anchor preserves shape;
- gradients flow through anchor.

## Success Criteria

- The code can represent non-identity anchors.
- The default identity-anchor behavior remains unchanged.
- The article statement about `rho_p` becomes directly implementable.

---

# Recommended Implementation Order

## Gate 0

Complete the independent 125M benchmark without changing its frozen training
path. In parallel, prepare (but do not inject into active runs) correctness
fixes for generation parity, inert config flags and default geometry-report
target semantics. Merge those fixes after the frozen run artifact is secured,
then add regression tests proving that generation uses the configured sequence
mode.

## Step 1

Implement metric-rank semantics:

- distinguish exact rank, numerical effective rank and gate activity;
- expose eigenvalue, participation-rank and kernel-mass diagnostics;
- identify candidate constant-rank strata without changing training.

Reason: the paper defines every later effective fiber and transition from
`Ker(g)` and `rank(g)`.

## Step 2

Implement the explicit anchor:

- identity anchor as the safe default;
- kernel-compatibility diagnostics;
- admissible-motion and observable-mobility ranks.

Reason: the anchor precedes connections and admissible curves in the formal
construction.

## Step 3

Implement stratumwise connection and transport:

- transport only inside identified constant-rank strata;
- verify metric compatibility and invertibility;
- label frame-alignment transport clearly as an approximation.

Reason: regular parallel transport is defined on the quotient metric bundle.

## Step 4

Implement transition identification:

- detect crossings between candidate rank strata;
- estimate explicit linear transition maps `J`;
- report rank defect, operator norm and `J^T G_+ J <= G_-` violations.

Reason: hybrid transport composes these maps with the already defined regular
transports.

## Step 5

Implement hybrid holonomy and hysteresis:

- ordered products of intrastatum transports and transition maps;
- rank, norm, energy, spectrum and order-memory defects;
- flat-stratum hysteresis tests matching the paper examples.

Reason: this is the paper's central global construction.

## Step 6

Validate reduction regimes:

- Riemannian and sub-Riemannian identification checks;
- Fisher–Rao reduction diagnostics when hypotheses are satisfied;
- toroidal closure only as a separate optional experiment requiring the
  appendix hypotheses.

Reason: these are conditional reductions, not substitutes for the core
rank-transition construction.

---

# What Will Change in the Project

## Short Term

The repository will gain stronger geometry diagnostics without changing the
default benchmark behavior.

Expected new outputs:

```text
transport_drift
gate_transport_error
holonomy_error
metric_effective_rank
metric_participation_rank
metric_kernel_mass
fisher_energy_mean
```

## Medium Term

The reports and dashboards can distinguish:

- validation quality;
- throughput;
- geometry stability;
- directional identity preservation;
- relational curvature;
- emergent metric dimensionality.

This will make DRM claims more defensible than CE/PPL alone.

## Long Term

The implementation can become a closer computational realization of the formal
DRM article:

- explicit relational transport;
- measurable curvature;
- optional toroidal closure;
- Fisher-Rao-compatible output geometry;
- explicit anchor maps.

At that point, the project can more safely claim:

```text
drm-language-emitter implements a neural realization of Directional Relational
Manifolds, including learned relational metric geometry, directional transport
diagnostics, emergent effective dimension, and optional toroidal phase dynamics.
```

Until then, the most accurate claim remains:

```text
drm-language-emitter is a neural language-modeling prototype inspired by
Directional Relational Manifolds.
```
