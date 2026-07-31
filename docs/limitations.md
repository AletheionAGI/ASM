# Limitations

This document describes limitations of the currently implemented repository,
not only limitations of the original DRM proposal. Architecture paths and
experimental protocols must be named explicitly when interpreting a result.

## Scientific status

DRM Language Emitter is an experimental causal language-model architecture. It
is not a production model, a validated replacement for Transformers, or a
demonstration of general superiority over GPT-2, Mamba, or other state-space
models.

The previously published 36M and 125M comparisons against GPT-2 are retracted
as comparative evidence. Their GPT-2 training path applied a target shift
before calling a causal-LM implementation that shifts internally, causing a
double-shift. Historical artifacts remain available for audit, but their
target-reached, time-to-quality, and DRM-versus-GPT-2 conclusions are invalid.

No completed corrected multiseed GPT-2 comparison currently establishes a DRM
advantage. A parameter-matched Mamba baseline has not yet been run.

## Current evidence is narrow

The strongest valid component result currently available is a 5M-token,
three-seed internal ablation:

- J validation CE mean: `1.760581`;
- SSM_CONTROL validation CE mean: `1.806518`;
- J won all three paired seeds;
- SSM_CONTROL trained approximately 2.5 times faster.

This supports a contribution from the implemented geometric system relative to
that internal selective-memory control. It does not establish:

- which geometric component causes the gain;
- that every DRM component is necessary;
- that J wins at 30M or 150M tokens;
- that J beats Mamba or a corrected GPT-2;
- that the CE gain compensates for the compute cost in deployment.

`J_NO_METRIC`, `J_NO_DIRECTION`, and `J_NO_NATURALIZATION` are intended to
decompose this result. The first two are structural ablations with fewer
parameters than J; they are not parameter-matched controls.

## Architecture and compute

The original recurrent path contains a strict temporal dependency and is slow
relative to optimized Transformer and SSM kernels.

The block-cumsum path improves parallelism by evaluating local deltas from the
state at the beginning of a causal block. It is an approximation to the
nonlinear recurrent trajectory, not an exact parallel solution. Increasing the
block size reduces geometry-update frequency and can weaken state-dependent
adaptation.

Variant J adds a selective forget/write recurrence. This materially improves
sample efficiency in current experiments, but also makes J a hybrid of the
original DRM geometry and an SSM-like memory mechanism. Claims about the
original DRM must not attribute J's selective memory to the independently
conceived geometric core.

The selective affine scan is implemented in PyTorch rather than a custom fused
kernel. It avoids the numerical failure of the earlier cumprod/division form,
but is not hardware-optimized like mature selective-scan implementations.

The relational metric, direction field, and flow are parameter-heavy. J is
approximately 2.5 times slower than SSM_CONTROL in the current 5M-token runs.
Parameter matching therefore does not imply compute matching.

## Generation mismatch

`generation.py` currently advances only the original recurrent geometry. It
does not reproduce:

- block-cumsum state construction;
- the causal local mixer;
- the token-to-state residual;
- selective memory;
- component-ablation paths;
- SSM_CONTROL.

Generation from J-family checkpoints is therefore not faithful to the training
forward path and must not be used as qualitative evidence until generation is
unified with model forward semantics.

## Geometry and mathematical interpretation

The implemented metric

$$
G(z)=\mathrm{diag}(\mathrm{softplus}(d(z))+\epsilon)+U(z)U(z)^\top
$$

is SPD because of its positive diagonal floor. Its exact matrix rank is
therefore `d_state`. `dimD`, gate activity, low-rank factor size, and numerical
or spectral effective rank are different quantities. None should be reported
as the formal `rank(g)` of the paper.

Metric naturalization is a damped first-order preconditioner. It is not a
complete variational geodesic solver. Low action does not prove that a
trajectory is geodesic, globally optimal, or unique.

The current implementation does not yet provide the paper's complete formal
kernel, effective fiber, stratification, transport, connection, holonomy,
anchor maps, or Fisher-Rao pullback construction.

The blockwise path may compute diagnostics from approximate rather than exact
recurrent trajectories. Those values are engineering diagnostics unless
validated for the selected `sequence_mode`.

## Numerical stability

The metric inverse uses damping and a Woodbury solve, but learned metrics can
still become poorly conditioned. Strong naturalization can amplify optimization
noise or suppress useful velocity components.

State clipping and `tanh` bounding improve numerical stability but alter the
unconstrained dynamics. Bounded coordinates do not prove recurrence,
compact-manifold structure, or toroidal topology.

The original recurrent update is Euler-like. Optional geodesic, candidate,
fixed-point, and Anderson mechanisms are experimental approximations rather
than certified differential-equation solvers.

## Data, tokenizer, and evaluation

The primary large experiments use a byte-level tokenizer. Byte-level modeling
guarantees coverage but produces longer dependency chains than subword
tokenization and can change CE/throughput trade-offs. CE values from different
tokenizers are not directly comparable.

Intermediate validation over a few sampled batches is noisy. Model selection
must use the same deterministic continuous validation sequence across variants.
Selecting the minimum `best_val_ce` across changing validation samples is
biased downward and is not accepted as the final comparison.

PG-19 is intended as a frozen external test set and must not be consulted
during architecture development. Until that one-time evaluation is completed,
the current evidence remains internal validation rather than external
generalization evidence.

MQAR is a synthetic associative-recall diagnostic. Success or failure on MQAR
does not by itself establish language-model quality.

## Product and safety

The repository does not include instruction tuning, RLHF, production serving,
security hardening, alignment evaluation, or a complete safety evaluation.
Base checkpoints are not chat models.

Risk, blindspot, and dubiety mechanisms are experimental scaffolds. J currently
does not instantiate RiskField, and these mechanisms must not be presented as
validated uncertainty, alignment, or safety systems.

## Topology

The optional toroidal representation maps circular coordinates to
`(cos(theta), sin(theta))`. It is a coordinate utility, not evidence of
spontaneous toroidal convergence. Such a claim would require dedicated
topological diagnostics, recurrence evidence, structural stability, controls,
and independent replication.
