# DRM: Philosophy, Geometry, and Architectural Re-evaluation

## 1. What DRM is

Directional Relational Manifolds, or DRM, is a geometric proposal for
describing systems in which the number of effectively available possibilities
can change throughout the system's own evolution.

Classical geometry usually begins with a space of fixed dimension. An object
moves within that space, but the space is already given.

DRM begins from a different idea:

> Geometry does not need to exist only as an immutable stage. It may emerge
> from the active relations of the system itself.

At each state, some directions may be available while others may be inactive,
redundant, or collapsed. In the complete DRM theory, effective dimension would
not be determined only by the number of chosen coordinates, but by the rank of
the relational metric structure at that state.

In simple terms:

> A system may change not only its position, but also what it is capable of
> doing.

## 2. State, relations, directions, and movement

An idealized DRM description contains:

1. a state situated in a latent space;
2. relations that depend on that state;
3. locally available directions;
4. a metric determining coupling, distance, and cost;
5. a rule of movement;
6. a transformation of the state and potentially of geometry itself.

A simplified representation is:

```text
input
  → state
  → active relations and directions
  → raw movement
  → metric transformation
  → new state
  → output
```

If $z$ is the state, $V_{i}(z)$ are local directions, and $a_{i}(z,x)$ are
state- and input-dependent coefficients, raw movement may be written as:

$$
v_{\mathrm{raw}}(z,x)=\sum_{i} a_{i}(z,x)V_{i}(z)
$$

A relational metric $G(z)$ may transform or naturalize this movement:

$$
v(z,x)=G(z)^{-1}v_{\mathrm{raw}}(z,x)
$$

A local update would then be:

$$
z_{t+1}=z_{t}+\Delta t\,v(z_{t},x_{t})
$$

This formulation separates two questions:

- Directions ask: **where can or should the system move?**
- The metric asks: **what is the cost, length, or significance of each
  movement?**

Formally, a metric does not choose a direction by itself. It transforms a
vector or measures relations among vectors. A directional field and a metric
are therefore not the same mathematical object.

## 3. Relational dimension and accessible possibilities

Imagine a robotic joint that can initially move in several directions. During
its trajectory, it enters a region in which some possibilities disappear. It
later returns to its initial point.

The joint may physically occupy the same position again. This does not
necessarily mean that it has recovered every possibility it previously had.

In DRM:

> Returning to the same place does not necessarily mean returning to the same
> state.

This produces an important consequence: the history of a system may become
part of its geometry.

Two systems may display the same observable state in the present while having
arrived through different paths. They may consequently possess different
internal capabilities.

A path does not merely take a system somewhere. It may transform the set of
possible futures.

This property can be discussed through transport, holonomy, rank transitions,
and geometric hysteresis. A closed trajectory may return to its starting point
while still transforming what was transported along the path.

If effective dimension is reduced during the path, some previous information
or possibilities may not be recoverable.

## 4. Irreversibility without violating energy conservation

Irreversible processes are normally associated with dissipation, increasing
entropy, or information loss to the environment.

DRM adds another possibility:

> A process may be irreversible because the system crossed a region in which
> certain directions effectively ceased to exist.

Even if the number of directions later increases, what was lost need not be
reconstructed.

This does not imply a violation of energy conservation. Conserved energy is not
the same as preserved possibilities. A system may conserve energy while losing
access to particular states.

This is a conceptual hypothesis. To become a physical explanation, it must
produce predictions distinct from existing explanations and survive controls
for dissipation, noise, decoherence, conventional hysteresis, and drift.

## 5. Local, relational, and dynamic dimensionality

The dimension of reality may not need to be understood only as a fixed global
number. The universe may retain its familiar spatial structure while specific
systems exhibit different numbers of effectively accessible degrees of freedom
under different conditions.

One direction may emerge. Another may disappear. Two directions may synchronize
and begin to act as one. A constraint may make a movement impossible. A phase
transition may create new collective modes.

In this sense, dimensionality becomes local, relational, and dynamic.

Three notions must be distinguished:

- **coordinate dimension:** how many numbers represent the state;
- **effective or numerical dimension:** the approximate number of relevant
  modes;
- **formal metric rank:** the number of non-degenerate directions under the
  geometric structure.

They are not automatically equivalent.

## 6. Causality as a structure of possibilities

DRM also offers a different way to think about causality.

What can happen from a state depends on the directions active at that state.
Causality would not be only a rule specifying how the system must move. It
would also involve the structure determining which movements are possible.

Reality could therefore be described not only by states and laws, but by:

```text
states
+ relations
+ accessible possibilities
+ transformation rules
```

The law of evolution and the effective domain on which it acts could transform
together.

## 7. Order, transport, and path dependence

In many systems, applying operation A before operation B gives a different
result from applying B before A.

In DRM, the difference may be deeper. The order of operations may modify not
only the final state but also the set of available directions. The sequence of
events becomes physically fundamental.

This suggests a family of experimental tests. An experiment could compare two
cycles containing the same stages executed in different orders.

If the final result showed a path-dependent structural transformation —
especially a rank change or loss of accessibility that could not be explained
only by dissipation, noise, decoherence, or drift — it would provide relevant
evidence for further investigation.

Path dependence alone would not confirm DRM. Holonomy, hysteresis, and
noncommutativity already occur in established theories. DRM would need a new
quantitative prediction or a more economical and testable explanation.

## 8. Time: returning does not mean undoing

DRM does not prove that time travel is possible. It does, however, provide a
mathematical language for returns without perfect repetition.

A system could return to the same position, configuration, or coordinate
without restoring all previous relations. The trajectory may close externally
while remaining internally transformed.

In other words:

> Returning does not mean undoing.

From this perspective, an arrow of time could emerge not only from current
position but from the accumulated transformation of the structure of
possibilities.

## 9. Cognition and learning

A mind may not merely move through a fixed space of thoughts. It may rebuild
its own space of possibilities as it learns.

Learning may create new directions of reasoning. Trauma may make some paths
inaccessible. A discovery may connect previously separate regions. A belief
may reorganize the distance between ideas.

Experience does not merely change the contents of the mind:

> It may change the geometry through which the mind thinks.

This interpretation does not claim that human cognition is literally a DRM
implementation. It offers a formal metaphor and a modeling hypothesis:
learning may jointly modify representations, transitions, and accessible
degrees of freedom.

## 10. Translation into artificial intelligence

DRM Language Emitter translated part of this proposal into a causal language
architecture. Instead of using pairwise token attention as its central
mechanism, the model compresses the sequence into the trajectory of a latent
state.

In the original formulation:

```text
token
  → causal state
  → directional field and gates
  → constrained flow
  → relational metric and naturalization
  → state update
  → language emitter
```

This suggests a different view of artificial intelligence:

> Thinking may be modeled as movement through learned geometry.

In principle, this approach permits state memory, interventions on internal
directions, trajectory observability, and new forms of control.

Recent variants also contain a causal mixer, a direct lexical path, and
selective forget/write memory. These are engineering mechanisms added to
improve context, associative recall, and sample efficiency. They are not, by
themselves, part of geometric DRM theory.

## 11. The mathematical limitation of the current implementation

The theory considers rank transitions and variable effective dimension. The
metric currently implemented by DRM Language Emitter is:

$$
G(z)=\mathrm{diag}\!\left(\mathrm{softplus}(d(z))+\varepsilon\right)+U(z)U(z)^{\mathsf T}
$$

Because its diagonal has a strictly positive floor, $G(z)$ is SPD and its exact
rank is always $d_{\mathrm{state}}$.

Consequently:

- `dimD` measures gate activity;
- `metric_rank` is the width of the low-rank update $U$;
- a spectral or numerical rank would only be an approximate diagnostic;
- none of these is currently the variable formal rank proposed by the theory.

The software implements a computational inspiration from DRM, not a complete
realization of every mathematical or physical hypothesis.

## 12. Re-evaluating the directional field

The latest experiments decomposed the architecture into components. Under the
five-million-token protocol, with three paired seeds and rescoring over the
same continuous validation sequence, the variant without an explicit
directional field achieved lower and more stable CE than complete J.

It replaces:

$$
v_{\mathrm{raw}}=\sum_{i} a_{i}(z,x)V_{i}(z)
$$

with a direct neural transition:

$$
v_{\mathrm{raw}}=T(z,x)
$$

The variant still has movement. What disappears is the requirement that every
movement be factored through an explicit collection of directions.

The current result supports a limited and precise statement:

> Under this protocol and at this scale, the explicit directional field harmed
> CE.

It does not yet show that all geometry is useless, that DRM theory is false, or
that directions could never help under a different parameterization.

## 13. Were directions already contained in the metric?

Formally, no. A metric measures and transforms vectors; it does not independently
provide the initial force or intention of movement.

Computationally, however, a direct neural transition can learn directions
implicitly. For every pair $(z,x)$, $T(z,x)$ produces a vector in state space.
As state and input vary, the function defines an entire vector field.

Directions may therefore not have disappeared. They may have migrated from an
explicit and restrictive basis into an implicit representation inside the
transition function.

This distinction is central:

1. every dynamic has directions of change;
2. not every architecture must represent them as explicit modules.

The possible mistake in DRM Language Emitter was not imagining directional
movement. It may have been requiring a particular factorization before
allowing movement to occur.

## 14. Why explicit directions may be unnecessary

### 14.1 Subspace bottleneck

When movement is a combination of a limited set of directions, it is restricted
to their span:

$$
v_{\mathrm{raw}}\in\mathrm{span}\!\left\{V_{1},\ldots,V_{n}\right\}
$$

A direct transition may produce any vector in state space. A restriction
intended to provide structure may instead remove capacity.

### 14.2 Redundant decisions

The field, gates, coefficients, and metric all determine aspects of the same
movement. The optimizer must coordinate them, while a direct transition learns
$T(z,x)$ as a single function.

### 14.3 Non-identifiability

Different bases can represent the same final vector. Directions can be rescaled
or rotated while coefficients compensate. This creates many equivalent
parameterizations that the CE objective has no need to distinguish.

### 14.4 Conflict between field and metric

The field proposes a vector and the metric transforms it. One module may learn
a direction that the other compresses or redirects. A direct transition can
learn a vector already suited to the metric transformation.

### 14.5 Blockwise approximation

For efficiency, the blockwise implementation reuses parts of geometry within
causal blocks. A direction computed at the beginning of a block may become
inappropriate as the state evolves. This tests an operational approximation,
not every conceivable DRM dynamic.

### 14.6 An objective without geometric reward

CE rewards next-token prediction. It does not directly reward geodesics,
holonomy, interpretability, rank transitions, or directional consistency. A
geometrically interesting structure may not provide an advantage under this
objective.

## 15. Where DRM may be wrong

The possible philosophical confusion is:

> Every dynamic can be described through directions; therefore, directions
> must be explicit and restrictive objects in the architecture.

The conclusion does not follow from the premise. A free transition function
already defines a vector field.

A simpler reformulation would be:

```text
input
  → state
  → free contextual transition field
  → learned geometry
  → new state
  → output
```

This preserves the intuition of relational dynamics while making an explicit
catalog of directions optional.

DRM may also impose structure in the wrong place. Relations and constraints
might need to emerge from memory, training, or an output-space metric rather
than restricting every local state update.

## 16. The decisive experiment

The next control compares:

1. a direct transition with metric and naturalization;
2. the same direct transition without metric or naturalization;
3. a geometry-free version with an equivalent parameter budget;
4. the selective-memory, mixer, and emitter control.

Possible interpretations are:

- If the metric version wins, geometry still contributes without an explicit
  directional field.
- If pure metric removal ties or wins, current geometry is dispensable for CE
  at this scale.
- If only the parameter-matched control wins, parameters are more useful in
  memory than in the metric.
- If direct transition beats the memory control, it adds capacity, but that
  alone does not validate geometric DRM theory.

The results should determine both the architecture and its name. If complete
DRM adds cost without benefit, the project may be renamed to honestly represent
the system produced by the evidence.

## 17. Scientific resilience

A scientific theory should not be protected from tests capable of contradicting
it. The purpose of an experimental architecture is not to confirm its author's
intuition, but to make that intuition measurable.

Removing a favored component when it harms the system is not abandoning the
project. It separates the project's identity from a specific implementation.

The objective becomes:

> Maximize the potential of the system without forcing it to preserve DRM
> theory if the evidence supports a better formulation.

This permits three equally useful outcomes:

- identify which DRM components genuinely contribute;
- reformulate the theory while retaining only its productive ideas;
- discover a different architecture that emerged from DRM investigation.

## 18. Implications for reality

The deepest philosophical implication of DRM may be:

> Reality may not consist only of things and positions. It may also consist of
> the relations that determine what each thing can still become.

In the traditional view, a state evolves according to laws inside a previously
defined space. In DRM, evolution may transform simultaneously:

- the state;
- geometry;
- active relations;
- the set of possible futures.

Reality would then not be merely a place where events occur. It would be a
structure that rebuilds itself as it happens.

This is a philosophical and mathematical proposal under investigation, not an
experimental conclusion about nature. Its value will depend on its ability to
produce precise models, distinctive predictions, rigorous controls, and
reproducible results.

## 19. The order of metric and direction

The re-evaluation exposed a deeper hypothesis: the problem may not be the
existence of a directional field, but the mathematical order in which it is
combined with the metric.

The original implementation is approximately:

```text
input → state → direction → raw movement → metric → final movement
```

or:

$$
v_{\mathrm{raw}}=Vc,\qquad v=G^{-1}Vc
$$

If $G^{-1}$ mixes coordinates, final movement need not remain in the
directional subspace:

$$
G^{-1}Vc\notin\mathrm{span}(V)
$$

This creates a conceptual tension. The field declares which movements are
available, but subsequent naturalization may move the result outside that set.

A philosophically more coherent order would be:

```text
input
  → state
  → relations
  → metric
  → directions interpreted in that geometry
  → movement
  → output
```

In compact form:

> State → geometry → possibilities → action.

Merely applying $G^{-1}$ to each direction before adding them does not change
the result because the transformation is linear:

$$
G^{-1}\sum_{i} c_{i}V_{i}=\sum_{i} c_{i}G^{-1}V_{i}
$$

Order changes effectively only when the metric participates in constructing,
normalizing, selecting, or combining directions.

### 19.1 Naturalization inside the subspace

The metric induced in directional space is:

$$
C=V^{\mathsf T}GV
$$

If $q$ represents movement intent, a constrained composition is:

$$
c=\left(V^{\mathsf T}GV+\lambda I\right)^{-1}q
$$

$$
v=V\left(V^{\mathsf T}GV+\lambda I\right)^{-1}q
$$

Now:

$$
v\in\mathrm{span}(V).
$$

The metric changes how much to move along each direction without invalidating
the possibility space declared by the field.

### 19.2 Metric-orthonormal directions

Another possibility transforms directions into a basis $Q$ satisfying:

$$
Q^{\mathsf T}GQ\approx I
$$

Gates and coefficients then operate on directions normalized under relational
geometry rather than only under a Euclidean norm.

### 19.3 New experimental hypothesis

Two variants have been defined:

- `J_METRIC_SUBSPACE`: solves coefficients inside the induced metric
  $V^{\mathsf T}GV$;
- `J_METRIC_ORTHONORMAL_DIRECTION`: orthonormalizes directions under the metric
  before composing movement.

They will be compared against original J, `J_NO_DIRECTION`, and
`J_DIRECT_CONTROL_MATCHED`.

If a metric-first composition recovers the advantage, the directional field
was not necessarily useless; it was being combined incompatibly with its own
geometry. If it continues to lose, evidence that explicit directional
factorization is a bottleneck for this objective will become stronger.

## 20. Summary

DRM began with the idea that states evolve through relational directions inside
learned geometry. The current re-evaluation suggests that movement and geometry
may remain relevant while explicit directional factorization may be redundant
or harmful.

The future project may preserve complete DRM. It may preserve only its view of
states that transform possibilities. It may produce a new architecture.

In every case, the central question remains:

> Is the system learning a real and useful structure, or merely carrying an
> elegant theory that the objective does not require?

Answering that question honestly is an essential part of the project's own
philosophy.
