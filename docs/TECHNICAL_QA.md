# Technical FAQ and Benchmark Methodology

This document describes the currently implemented DRM Language Emitter architecture and experimental evidence. It distinguishes verified behavior, preliminary results, and open research questions.

## What is implemented?

The implemented DRM model is a causal language emitter built around recurrent
latent-state dynamics rather than Transformer attention. `DRMEmitterModel` is
assembled in `src/drm_language_emitter/model.py`; the blockwise implementation
is split across focused modules:

- `model_components.py`: state initializer and causal local mixer;
- `directional_forward.py`: cumsum forward, losses, and diagnostics;
- `directional_blocks.py`: block and superblock construction;
- `directional_solvers.py`: fixed-point, Anderson, and transition helpers;
- `geometric_steps.py`: state bounding, candidates, and geodesic refinement.

The implemented path is:

```text
input_ids
  -> TokenEmbedding
  -> initial state z_0
  -> DirectionField(z_t) -> directions, gates
  -> RelationalMetric(z_t) -> diag + U U^T
  -> DRMFlow(z_t, e_t, directions, gates) -> dz_raw
  -> metric.naturalize(dz_raw, diag, U)
  -> StateUpdater(z_t, dz) -> z_{t+1}
  -> LanguageEmitter(z_{t+1}) -> logits
```

The model is autoregressive: the emitter projects the updated state to next-token logits and the training objective includes next-token cross entropy.

The leading 125M path uses causal block64 velocity cumsum plus a two-layer
causal convolutional local mixer. Prefix-causality tests verify that changing
future tokens does not alter earlier states or logits.

## How is DRM different from a Transformer?

A Transformer contextualizes tokens primarily through attention projections and token-token attention. The DRM implementation updates a latent state through learned directions, gates, a relational metric, and a flow update.

```text
Transformer: embedding -> attention/QKV -> MLP -> logits
DRM:         embedding -> state -> direction -> relational metric -> flow -> emitter
```

This is not only a naming difference in the current implementation: the DRM core does not instantiate Transformer blocks or attention layers.

## Does the DRM core use attention or QKV?

No. The DRM core does not use self-attention, Q/K/V projections, `nn.MultiheadAttention`, or a KV cache.

The repository includes `tests/test_no_transformer.py`, which checks that `DRMEmitterModel` does not instantiate `nn.MultiheadAttention` and that the package source does not define common QKV projection names such as `q_proj`, `k_proj`, or `v_proj`.

Transformer and Hugging Face models exist in the repository only as baselines for comparison.

## What is the main model equation?

The operational update is:

```text
z_{t+1} = z_t + dt * dz_t

dz_raw_t = sum_i gates_i(z_t) * c_i(z_t, e_t) * direction_i(z_t)
dz_t = naturalize_G(dz_raw_t)

logits_t = Emitter(z_{t+1})
p(x_{t+1} | z_{t+1}) = softmax(logits_t)
```

Where:

- `z_t` is the latent state.
- `e_t` is the current token embedding.
- `direction_i(z_t)` is produced by `DirectionField`.
- `gates_i(z_t)` controls active directions.
- `G(z_t)` is the learned relational metric.
- `naturalize_G` applies a stable metric preconditioner.
- `Emitter` maps the updated state to vocabulary logits.

## What is the relational metric?

The implemented metric is:

```text
G(z) = diag(softplus(d(z)) + eps) + U(z)U(z)^T
```

It is implemented in `RelationalMetric`. The diagonal is strictly positive because of `softplus + metric_eps`, and `U U^T` is positive semidefinite. This provides an SPD metric form while avoiding a full dense `d_state x d_state` matrix.

The implementation also computes a diagnostic `condition_proxy`:

```text
low_rank_scale = sum(U^2)
upper = max(diag) + low_rank_scale
lower = max(min(diag), 1e-8)
condition_proxy = upper / lower
```

This is a numerical proxy, not an exact eigenspectrum calculation.

## Does the current metric implement the formal rank from the paper?

No. Because `softplus(d) + eps > 0`, the current neural metric is strictly
positive definite:

```text
Ker(G) = {0}
rank(G) = d_state
```

The current `dimD` diagnostic is the sum of directional gates. It measures
directional activity, not the paper's formal definition:

```text
d_DRM(p) = rank(g_p)
```

A tolerance-based numerical rank can be a useful spectral approximation, but
it must be labeled as such. A literal implementation of the paper requires a
PSD parameterization capable of producing a genuine kernel.

## What does "direction" mean?

Direction is a learned vector in latent state space. `DirectionField(z)` produces a set of directions and sigmoid gates. The flow computes movement inside the span of the active directions.

Direction should not be interpreted as ethics, intent, or alignment by itself. In the current code it is an operational update variable.

## What diagnostics are implemented?

The model reports operational diagnostics including:

- `action_mean`: mean metric energy of the movement, computed as `G_z(dz, dz)`;
- `dimD_mean`: mean sum of direction gates;
- `soft_active_fraction`: `dimD_mean / n_directions`;
- hard active fractions at gate thresholds;
- `condition_proxy`;
- `metric_U_norm_mean`;
- `recurrence_proxy`;
- `stability_proxy`;
- risk-field diagnostics when enabled.

These diagnostics are not psychological or semantic measurements. Their technical value must be tested through stability analysis, correlation, interventions, and ablations.

## What is the historical 36M/37M benchmark?

The historical public artifact is:

```text
docs/benchmarks/bench_36M/
```

It compares DRM, GPT-2-style, and OPT-style models with approximately matched parameter counts around 37M parameters. The internal model labels still contain `125m`, but the actual parameter counts in this run are about 37M.

| model label | family | parameters | seeds | tokens per seed |
|---|---:|---:|---:|---:|
| `drm_36M` | DRM | 37,253,702 | 1, 2, 3 | 2,048,000 |
| `gpt2_36M` | GPT-2 | 36,915,984 | 1, 2, 3 | 2,048,000 |
| `opt_36M` | OPT | 36,916,992 | 1, 2, 3 | 2,048,000 |

The corrected public name for this result is "36M/37M benchmark"; the `125m`
label should be treated as a legacy script/configuration label, not the real
size of the run. It is no longer the latest large benchmark.

## What dataset and tokenizer were used?

The benchmark uses:

| item | value |
|---|---:|
| dataset | `wikimedia/wikipedia` |
| config | `20231101.en` |
| split | `train` |
| loading | streaming |
| written sample | 50,000,002 characters |
| documents | 2,272 |
| minimum document length | 200 characters |
| prepared file | `data/wikipedia_en_20231101_sample.txt` |
| tokenizer | byte-level |
| effective vocabulary | 256 |

The train/validation split is made over the tokenized sequence: approximately 90% train and the final portion for validation, with sequence overlap needed to form language-model windows.

## Why use byte-level tokenization?

Byte-level tokenization reduces dependency on a learned vocabulary and guarantees coverage of arbitrary text with only 256 symbols. It also makes this initial comparison more uniform because DRM, GPT-2, and OPT baselines use the same effective vocabulary size.

The downside is that byte-level tokenization usually produces longer sequences than subword tokenizers. That can affect cost, effective context length, and linguistic quality. It is a controlled experimental choice, not a claim that byte-level tokenization is optimal for scale.

## How many tokens did each model see?

Each run saw:

```text
steps * grad_accum_steps * batch_size * seq_len
= 1000 * 1 * 4 * 512
= 2,048,000 tokens
```

With 3 seeds per family, each family processed 6,144,000 tokens across the aggregate, but each reported seed run corresponds to 2,048,000 tokens.

## What was the benchmark protocol?

| item | value |
|---|---:|
| steps | 1,000 |
| batch size | 4 |
| gradient accumulation | 1 |
| sequence length | 512 |
| learning rate | 3e-4 |
| optimizer | AdamW |
| gradient clipping | 1.0 |
| eval interval | 100 |
| eval batches | 1 |
| seeds | 1, 2, 3 |
| precision | PyTorch default; no AMP in the benchmark script |
| hardware | not recorded in the versioned artifacts |

The benchmark is parameter-matched and protocol-matched. It is not rigorously time-matched or compute-matched, because wall-clock time, FLOPs, and kernel efficiency were not normalized.

## What were the benchmark results?

Aggregate validation cross entropy:

| model | best val CE mean | std | final val CE mean | std |
|---|---:|---:|---:|---:|
| DRM | 2.3063 | 0.0391 | 2.3225 | 0.0468 |
| GPT-2 | 2.8914 | 0.0211 | 2.9252 | 0.0166 |
| OPT | 2.8927 | 0.0248 | 2.9729 | 0.0536 |

Aggregate throughput and memory:

| model | tokens/sec mean | max memory MB mean |
|---|---:|---:|
| DRM | 1,806.5 | 1,131.8 |
| GPT-2 | 61,382.2 | 2,139.2 |
| OPT | 67,926.2 | 1,568.8 |

Interpretation: DRM achieved lower validation CE in this preliminary 36M/37M setup, but trained much more slowly. This does not establish general superiority over Transformers.

## What are the main benchmark limitations?

Known limitations:

- only 3 seeds;
- small token budget by modern LM standards;
- only one validation batch per evaluation point;
- no hardware metadata in the versioned artifact;
- not compute-matched or time-matched;
- no Mamba/modern SSM baseline yet;
- no large-scale human generation evaluation;
- no formal statistical significance claim beyond reported means and standard deviations;
- possible sensitivity to split choice, corpus duplication, or benchmark setup.

The result should be read as evidence of feasibility and a promising controlled comparison, not as a final architecture claim.

## What are the exact DRM architectural parameters in the 36M benchmark?

The DRM run uses the internal config file `configs/drm_125m.yaml` and produced 37,253,702 trainable parameters. The public label for this benchmark result is `drm_36M`.

| parameter | value |
|---|---:|
| `vocab_size` | 256 |
| `d_token` | 768 |
| `d_state` | 768 |
| `n_directions` | 32 |
| `metric_rank` | 32 |
| `hidden_size` | 2048 |
| `n_flow_steps` | 1 |
| `dt` | 0.08 |
| `max_seq_len` | 512 |
| `dropout` | 0.0 |
| `bounded_state` | true |
| `state_clip_norm` | 8.0 |
| `direction_norm` | true |
| `direction_basis_size` | 128 |
| `metric_u_basis_size` | 128 |
| `geometry_update_interval` | 4 |
| `bptt_truncate_interval` | 64 |
| `emitter_layers` | 1 |
| `emitter_swiglu` | false |
| `emitter_residual` | false |
| `tie_embeddings` | false |

Auxiliary loss and regularization settings:

| setting | value |
|---|---:|
| `lambda_action` | 0.01 |
| `lambda_dim_sparsity` | 0.001 |
| `lambda_dim_entropy` | 0.001 |
| `lambda_dim_variance` | 0.01 |
| `target_dim_std` | 0.15 |
| `lambda_metric_reg` | 0.001 |
| `lambda_metric_diversity` | 0.001 |
| `lambda_active_fraction` | 0.01 |
| `target_active_fraction` | 0.65 |
| `lambda_condition` | 0.001 |
| `target_condition` | 100.0 |
| `lambda_metric_u_floor` | 0.001 |
| `metric_u_min_norm` | 0.05 |
| `lambda_metric_u_target` | 0.001 |
| `metric_u_target_norm` | 1.0 |
| `lambda_recurrence` | 0.0 |
| `lambda_stability` | 0.0 |
| `lambda_blindspot` | 0.0 |

## Does the experiment use weight tying?

No. In the internal benchmark config `configs/drm_125m.yaml`, `tie_embeddings: false`. The current implementation uses `TokenEmbedding.embedding` for input and a separate `LanguageEmitter.lm_head` for output.

This choice affects parameter count and regularization. It should be declared,
and future comparisons should either match or explicitly control this choice.
Enabling `tie_embeddings` currently has no effect because the flag is not yet
wired into model construction; it should be implemented or rejected
explicitly before use.

## Where do the 37M parameters come from?

The parameter count is measured by `count_parameters(model)` in `scripts/run_scale_lm_comparison.py`, summing all trainable PyTorch parameters.

Major sources include:

- byte-level embedding `256 x 768`;
- `DirectionField` trunk MLP, directional basis, and direction/gate heads;
- `RelationalMetric` trunk MLP, diagonal head, low-rank basis, and coefficient head;
- `DRMFlow` coefficient network over `d_state + d_token`;
- `LanguageEmitter` RMSNorm and projection network;
- learned initial state `z0`.

## How does this relate to RNNs, SSMs, Mamba, and Neural ODEs?

DRM is related to recurrent and state-space approaches because it maintains
and updates a state. The central distinction is:

> Mamba is fundamentally a selective linear state-space model; DRM is intended
> to be a nonlinear dynamics model guided by learned relational geometry.

Approximately, a Mamba transition can be described as:

\[
h_t = A_t h_{t-1} + B_t x_t,\qquad y_t = C_t h_t
\]

Input-dependent parameters decide what to preserve, write, and read while
retaining an SSM structure designed for an efficient selective scan.

The original DRM update is closer to:

\[
\Delta z_t =
g(z_t)^{-1}
\sum_k \alpha_k(z_t,x_t)\,v_k(z_t)
\]

Here, \(z_t\) is the latent state, \(v_k\) are learned directions,
\(\alpha_k\) are direction gates, and \(g(z_t)\) is a learned SPD metric that
preconditions or “naturalizes” the update. Geodesic steps, directional
candidates, energy, risk, and fixed-point mechanisms can additionally be
investigated.

In simplified terms, Mamba asks what information should be retained, forgotten,
and retrieved. DRM asks in which direction the state should move under the
local relational geometry.

| Aspect | Mamba | DRM |
|---|---|---|
| Core | Selective SSM | Directional field and relational metric |
| State transition | Selective linear recurrence | Nonlinear geometric dynamics |
| Selection | Token-dependent transition parameters | Gates select movement directions |
| SPD geometry | Not its central mechanism | Central to the DRM proposal |
| Geodesics/fixed points | Not part of the core | Investigated by DRM |
| Forget/write memory | Central | Added experimentally in variant J |
| Efficient execution | Hardware-aware selective scan | Depends on variant and solver |
| Maturity | Established architecture | Experimental architecture |

The difference proposed by the DRM implementation is the explicit combination
of:

- learned directional field;
- direction gates;
- learned relational metric;
- metric-preconditioned flow;
- causal language emitter.

### Why do variant J and Mamba look more similar?

Variant J adds a selective forget/write memory:

\[
m_t=f_t\odot m_{t-1}+w_t\odot c_t
\]

This mechanism is conceptually close to Mamba, although it is not a Mamba
implementation. J and J_DILATED are hybrids: they combine SSM-like selective
memory with DRM geometric dynamics. They were introduced because the original
DRM showed weak sample efficiency and possible associative-recall limitations.

If removing the metric, directions, and flow does not harm CE, that would
indicate that the selective memory—not DRM geometry—is providing most of the
useful capacity. The decisive experiment is therefore:

1. complete J;
2. J without the relational metric;
3. J without the directional field;
4. selective memory + mixer + emitter only;
5. complete J under the same parameter budget.

This ablation is necessary to determine empirically whether DRM geometry adds
capability beyond the Mamba-inspired mechanism. Mamba and other modern SSMs
also remain necessary external baselines.

DRM is also conceptually close to Neural ODEs because it describes state evolution, but the current implementation uses a discrete update:

```text
z_next = z + dt * dz
```

There is no adaptive ODE solver in the current core.

## Does DRM solve alignment or safety?

No. The implemented architecture exposes state, directions, metric, flow, and diagnostics, which may be useful for research into observability and control. It does not solve alignment or safety by itself.

Safety still requires external evaluation, adversarial testing, policy controls, sandboxing, monitoring, and deployment constraints.

## Do internal variables cause the result or just correlate with it?

Inspection of internal diagnostics alone does not prove causality. Causal evidence requires interventions and ablations, such as:

- zeroing gates;
- replacing the metric with identity;
- freezing the directional field;
- randomizing components;
- removing metric naturalization;
- changing `metric_rank`;
- measuring effects on loss, stability, trajectory, and generation.

If these interventions do not measurably change behavior, strong interpretations of direction or metric should be reduced.

## What is the published 125M benchmark?

The strongest completed and versioned benchmark uses three seeds per family
and 150M tokens per seed:

| model | parameters | best validation CE mean | std | tokens/sec |
|---|---:|---:|---:|---:|
| DRM block64 + causal local mixer | 127.27M | 1.3116 | 0.0019 | 10,678.7 |
| GPT-2 real | 126.08M | 1.7305 | 0.0259 | 41,224.4 |

Artifacts:

```text
docs/benchmarks/competition_125m_local_mixer_h256_l2_s02_150m/
```

This result is evidence under its specific protocol, not a general
superiority claim. It used a token-level validation split and did not preserve
the exact best DRM checkpoint weights.

## What changed in the independent 125M benchmark?

The frozen independent protocol adds:

- document-disjoint Wikipedia train and validation manifests;
- exact-document deduplication before tokenization;
- the official PG-19 test split as a frozen external source;
- zero-overlap contamination audits using 512-byte blocks and stride 256;
- recorded corpus hashes, hardware, software, and seeds;
- three DRM and three GPT-2 runs at 150M tokens each;
- best-checkpoint saving;
- one external evaluation after validation-only checkpoint selection.

Primary files:

```text
configs/independent_125m_protocol.json
scripts/run_independent_125m_smoke.sh
scripts/run_independent_125m_benchmark.sh
scripts/evaluate_frozen_test.py
docs/report/023_Proposta_Validacao_Independente_125M_2026_07_30.md
```

As of 2026-07-30, training is in progress and PG-19 has not been used for
model selection or reported test results.

There is a methodological caveat: intermediate validation uses only four
batches. The observed difference between best and final validation CE shows
that this estimate is noisy. Before PG-19 is accessed, candidate checkpoints
should be rescored on a larger deterministic validation traversal using
identical tokens for DRM and GPT-2. This protocol deviation must be disclosed.

## Does generation reproduce the trained local-mixer forward path?

Not yet. `generation.py` currently advances the state through the basic
recurrent transition and does not reproduce the block-cumsum/local-mixer
forward path used by the leading 125M model.

This does not invalidate training or CE evaluation, which call
`DRMEmitterModel.forward`. It does mean that chat or samples from those
checkpoints should not yet be treated as faithful inference from the trained
sequence engine. A shared prefill/decode API and parity tests are required.

## How can the historical 36M benchmark be reproduced?

The benchmark script is:

```powershell
.\scripts\run_wiki_en_125m_matched.ps1
```

It invokes `scripts/run_scale_lm_comparison.py` with:

```text
--models drm_125m gpt2_125m opt_125m
--dataset wikipedia-en
--steps 1000
--seeds 1 2 3
--batch-size 4
--grad-accum-steps 1
--seq-len 512
--lr 3e-4
--eval-interval 100
--eval-batches 1
--hf-vocab-size 256
```

Install optional Hugging Face dependencies before running the comparison:

```bash
pip install -e ".[hf]"
```

## How can the independent 125M benchmark be reproduced?

Verify CUDA, manifests, and both model forwards:

```bash
./scripts/run_independent_125m_smoke.sh
```

Run or resume all six training runs:

```bash
./scripts/run_independent_125m_benchmark.sh
```

Do not run the frozen PG-19 evaluation before checkpoints have been selected
using validation only.

## What should be improved next?

High-priority next steps:

- complete and publish the independent 125M validation;
- rescore checkpoint candidates on a deterministic validation traversal;
- unify generation with the block-cumsum/local-mixer forward path;
- implement or reject inert configuration flags;
- implement numerical-rank/kernel/strata diagnostics from the formal roadmap;
- add Mamba and modern SSM baselines;
- run compute-matched and time-matched comparisons;
- run ablations for direction, metric, gates, flow, and naturalization;
- measure inference separately from training;
- test longer context and retention;
- evaluate generation quality with standardized benchmarks and human review.

## What is the license?

The project metadata declares `AGPL-3.0-only`, and `LICENSE` contains the GNU Affero General Public License v3. The repository also includes `LICENCE-COMMERCIAL.md` for commercial licensing.

Short version: the public repository is AGPL-3.0-only; proprietary or commercial use should review the commercial license path.
