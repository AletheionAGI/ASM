# Mathematical Notes

This document separates the mathematics implemented in the repository from the
broader formal DRM program. Equations describe the current code unless a
section is explicitly marked as proposed or future work.

## 1. State and token representation

Let

\[
z_t\in\mathbb{R}^{d_{\mathrm{state}}}
\]

denote the causal latent state and

\[
e_t=E(x_t)\in\mathbb{R}^{d_{\mathrm{token}}}
\]

the embedding of token \(x_t\). `DRMStateInitializer` learns \(z_0\), which is
expanded across the batch.

The vector \(z_t\) is treated as a coordinate representation of a latent
manifold. The implementation does not prove that a globally defined smooth
manifold with the full formal structure exists.

## 2. Directional field

`DirectionField` predicts \(n\) directions and soft activity gates:

\[
V_i(z)\in\mathbb{R}^{d_{\mathrm{state}}},\qquad
a_i(z)\in[0,1].
\]

The engineering effective dimension is

\[
\operatorname{dimD}(z)=\sum_{i=1}^{n}a_i(z).
\]

The directions may be non-orthogonal. Optional normalization controls their
Euclidean norm but does not construct an orthonormal frame.

`dimD` measures gate activity. It is not the exact rank of the implemented
metric and not yet the formal \(\operatorname{rank}(g)\) proposed by the paper.

## 3. Direction-constrained flow

`DRMFlow` predicts token- and state-dependent coefficients:

\[
c_i(z_t,e_t)\in[-1,1].
\]

The raw velocity is

\[
v_t^{\mathrm{raw}}
=\sum_{i=1}^{n}
a_i(z_t)c_i(z_t,e_t)V_i(z_t).
\]

Consequently,

\[
v_t^{\mathrm{raw}}\in
\operatorname{span}\{V_i(z_t):a_i(z_t)\ne0\}
\]

up to the soft-gate interpretation.

`J_NO_DIRECTION` does not use this decomposition. It replaces the directional
field and constrained flow with a direct causal neural transition
\(T(z_t,e_t)\).

## 4. Implemented relational metric

`RelationalMetric` predicts

\[
G(z)
=D(z)+U(z)U(z)^\top,
\]

where

\[
D(z)=\operatorname{diag}
\left(\operatorname{softplus}(d(z))+\epsilon\right).
\]

For nonzero \(\epsilon\),

\[
G(z)\succ0.
\]

The metric energy of velocity \(v\) is

\[
\mathcal{E}_z(v)
=v^\top G(z)v
=v^\top D(z)v+\lVert U(z)^\top v\rVert_2^2.
\]

The coupling between learned directions is

\[
C_{ij}(z)=V_i(z)^\top G(z)V_j(z).
\]

### Exact and numerical rank

Because \(D(z)\) has a strictly positive diagonal floor,

\[
\operatorname{rank}(G(z))=d_{\mathrm{state}}
\]

in exact arithmetic. `metric_rank` configures the width of the low-rank update
\(U\); it does not set the exact rank of \(G\).

A thresholded spectral or numerical effective rank may be useful as a
diagnostic, but it must be labeled as an approximation. It is not the formal
rank of a degenerate metric.

## 5. Metric naturalization

The code can precondition the raw velocity:

\[
\hat v_t=(G(z_t)+\lambda I)^{-1}v_t^{\mathrm{raw}},
\]

where \(\lambda\) is damping.

With naturalization strength \(s\in[0,1]\), the applied velocity is

\[
v_t=(1-s)v_t^{\mathrm{raw}}+s\hat v_t.
\]

The inverse is evaluated through the Woodbury identity for diagonal plus
low-rank structure:

\[
(D+UU^\top)^{-1}
=D^{-1}
-D^{-1}U(I+U^\top D^{-1}U)^{-1}U^\top D^{-1}.
\]

This is metric-aware first-order preconditioning. It is not equivalent to
solving the geodesic boundary-value problem.

`J_NO_NATURALIZATION` retains \(G(z)\) but sets \(s=0\).
`J_NO_METRIC` removes the metric module and uses \(v_t=v_t^{\mathrm{raw}}\).
Under the current CE-only protocol, these two paths should have identical
state trajectories when their shared initialization streams match, because the
metric affects CE only through naturalization.

## 6. Recurrent state update

The original recurrent path uses an Euler-like update:

\[
z_{t+1}=z_t+dt\,v_t.
\]

When `bounded_state` is enabled, the implementation additionally applies norm
clipping and a coordinate-wise `tanh` projection. This stabilizes optimization
but changes the unconstrained dynamics.

## 7. Block-cumsum approximation

For a causal block beginning at state \(z_b\), the blockwise path evaluates
token-conditioned local velocities from the block-start geometry and
approximates prefix states by

\[
\tilde z_{b,t}
=z_b+\sum_{j\le t}dt\,v_{b,j}.
\]

The final state of one block initializes the next:

\[
z_{b+1}=\tilde z_{b,L}.
\]

This preserves causal block ordering and prefix causality. It does not exactly
equal the nonlinear recurrent rollout because geometry is not recomputed after
every within-block state update.

A causal depthwise convolutional mixer can produce a residual correction

\[
\bar z_{b,1:L}
=\tilde z_{b,1:L}
+\eta_{\mathrm{mix}}F_{\mathrm{causal}}
(\tilde z,e,\Delta z,\operatorname{dimD},r).
\]

Only left padding is used, so future positions do not alter prefix outputs.

## 8. Token-to-state residual

The D-and-later ablations can add a direct lexical path:

\[
z_t'=\bar z_t+\eta_{\mathrm{tok}}W_{\mathrm{tok}}e_t.
\]

This prevents all token information from being forced exclusively through
directional coefficients and block geometry.

## 9. Selective memory in J

Variant J predicts forget, write, and candidate vectors from the previous
causal base state and current token:

\[
\begin{aligned}
f_t&=\sigma(W_f h_t+b_f),\\
w_t&=\sigma(W_w h_t+b_w),\\
c_t&=\tanh(W_c h_t+b_c).
\end{aligned}
\]

The memory recurrence is

\[
m_t=f_t\odot m_{t-1}+w_t\odot c_t.
\]

The state receives a scaled correction:

\[
z_t^{J}=z_t'+\eta_{\mathrm{mem}}m_t.
\]

Each step is an element-wise affine map

\[
T_t(m)=f_t\odot m+u_t,\qquad
u_t=w_t\odot c_t.
\]

Affine maps compose associatively:

\[
(f_2,u_2)\circ(f_1,u_1)
=(f_2\odot f_1,\;u_2+f_2\odot u_1).
\]

The implementation uses this associative composition for a parallel scan. It
does not use the numerically unstable identity that divides by a cumulative
product of forget gates.

This recurrence is SSM-like selective memory, but it is not an implementation
of Mamba's complete selective SSM parameterization or block architecture.

## 10. Language emission and CE

`LanguageEmitter` maps the causal state to vocabulary logits:

\[
\ell_t=f_{\mathrm{emit}}(z_t).
\]

The next-token distribution is

\[
p(x_{t+1}\mid x_{\le t})
=\operatorname{softmax}(\ell_t).
\]

The primary objective is

\[
\mathcal{L}_{\mathrm{CE}}
=-\frac1T\sum_{t=1}^{T}
\log p(x_{t+1}\mid x_{\le t}).
\]

The historical GPT-2 double-shift bug did not change this DRM equation; it made
the old cross-family benchmark invalid by training GPT-2 against the wrong
target offset.

## 11. Optional geometric objective

The discrete action proxy is

\[
\mathcal{A}(z_{0:T})
=\sum_t dt\,v_t^\top G(z_t)v_t.
\]

A general configured loss may be written as

\[
\mathcal{L}
=\mathcal{L}_{\mathrm{CE}}
+\lambda_{\mathrm{action}}\mathcal{L}_{\mathrm{action}}
+\sum_k\lambda_k\mathcal{R}_k.
\]

The regularizers can include gate activity, dimension variance, metric
conditioning/diversity, recurrence, stability, risk, and consistency proxies.

Variant J and its current component ablations set these geometric auxiliary
weights to zero. Their reported 5M-token results optimize CE only. Metric
parameters in `J_NO_NATURALIZATION` therefore have no CE gradient path.

## 12. Causality

For a causal model, if two inputs share a prefix through position \(t\), their
logits through \(t\) must match:

\[
x_{\le t}=x'_{\le t}
\Longrightarrow
\ell_{\le t}(x)=\ell_{\le t}(x').
\]

The recurrent, causal-convolution, selective-scan, and component-ablation paths
have automated prefix-causality tests. Causality does not imply that blockwise
states equal exact recurrent states.

## 13. SSM_CONTROL

The geometry-free control retains:

\[
\text{embedding}
\rightarrow\text{causal mixer}
\rightarrow\text{token residual}
\rightarrow\text{selective memory}
\rightarrow\text{emitter}.
\]

It removes direction, metric, flow, and risk. Its selective-memory hidden width
is increased to approximately match J's parameter count. This makes it a
parameter control, not a compute control and not a Mamba baseline.

## 14. Geodesic interpretation

A formal geodesic would be a stationary or minimizing curve of an action
functional under appropriate boundary conditions and admissibility
constraints. The current implementation provides:

- a learned SPD metric;
- direction-constrained velocities;
- an action proxy;
- optional local candidate and solver approximations.

It does not prove:

- global or local action optimality;
- satisfaction of the geodesic equation;
- uniqueness;
- completeness;
- convergence of the numerical path to a geodesic.

Low measured action must therefore be reported as a diagnostic, not as proof of
geodesic emergence.

## 15. Proposed formal extensions

The following belong to the roadmap or paper program and are not yet complete
in the runtime:

- a degenerate metric with a nontrivial formal kernel;
- effective fibers and rank-based strata;
- anchor maps and relational transport;
- connection and holonomy diagnostics;
- Fisher-Rao or decoder-distribution pullback metrics;
- formal variational boundary-value solvers;
- validated topological invariants.

Future metrics may be defined as pullbacks of a distribution-space metric:

\[
G_{\mathrm{pullback}}(z)
=J_f(z)^\top G_{\mathrm{distribution}}(f(z))J_f(z),
\]

where \(f\) maps latent states to emitted distributions. This would connect
latent geometry more directly to changes in language-model output.

## 16. Toroidal topology

The optional toroidal coordinate utility represents angles by

\[
\theta\mapsto(\cos\theta,\sin\theta).
\]

This guarantees a circular coordinate representation only when explicitly
used. It does not demonstrate spontaneous toroidal topology, recurrence, or
structural stability of learned trajectories.
