# ASM — Aletheion State Models

**An attention-free causal state-model research family, derived from DRM and selected by ablations and scaling evidence.**

![Aletheion State Models banner](assets/drm-language-emitter-banner.svg)

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-pure%20torch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](pyproject.toml)
[![Non Transformer](https://img.shields.io/badge/architecture-non--Transformer-14B8A6?style=for-the-badge)](ARCHITECTURE.md)
[![No Attention](https://img.shields.io/badge/attention-none-0F172A?style=for-the-badge)](tests/test_no_transformer.py)
[![Benchmarks](https://img.shields.io/badge/benchmarks-audited-F59E0B?style=for-the-badge)](docs/benchmarks/README.md)
[![License](https://img.shields.io/badge/license-AGPL--3.0-64748B?style=for-the-badge)](LICENSE)

ASM studies language generation as the evolution of a persistent causal state.
The family contains explicit DRM geometry, metric-conditioned direct
transitions, geometry-free controls, and selective-memory models under one
reproducible experimental framework.

This repository was formerly named **DRM Language Emitter**. DRM now identifies
the theory and the explicit geometric variant **ASM-X**, while the public model
family remains free to follow the architecture that survives ablations and
scaling-law evaluation.

This is a research scaffold, not a production model and not a claim of superiority over Transformers or general world models.

## Quick Links

- [Architecture](ARCHITECTURE.md)
- [ASM Model Family](docs/MODEL_FAMILY.md)
- [DRM Philosophy and Re-evaluation](docs/drm_philosophy.md)
- [Project History and Rename](HISTORY.md)
- [Formal DRM Implementation Roadmap](roadmap.md)
- [Mathematical Notes](docs/math.md)
- [Competition Protocol](docs/competition.md)
- [Tiny World-Model Competition](docs/world_model_competition.md)
- [Scale LM Comparison](docs/scale_lm_comparison.md)
- [Technical FAQ and Benchmark Methodology](docs/TECHNICAL_QA.md)
- [Benchmark Artifacts](docs/benchmarks/README.md)
- [Time-To-Quality Benchmark](docs/benchmarks/tta/README.md)
- [Model Card](MODEL_CARD.md)
- [Limitations](docs/limitations.md)
- [API Reference](docs/api.md)
- [Minimal Training Loop](docs/examples/minimal_training_loop.md)
- [Minimal Training Notebook](docs/notebooks/minimal_training_loop.ipynb)
- [Compliance Checklist](docs/compliance_checklist.md)
- [Third-Party Licenses And Data Provenance](docs/third_party_licenses.md)
- [Commercial License](LICENCE-COMMERCIAL.md)

## What Makes ASM Different

The current ASM variants do not use:

- Transformer blocks;
- self-attention;
- Q/K/V attention;
- `nn.MultiheadAttention`;
- KV cache.

Their central computation is a latent trajectory. ASM-X, the explicit DRM
variant, uses this causal path:

```text
token e_t
  -> latent state z_t in M
  -> active directions D(z_t)
  -> gates a_i(z_t), effective dimD(z_t)
  -> relational metric g_z = diag + U U^T
  -> velocity dz in span(D(z_t))
  -> z_{t+1}
  -> token logits
```

The current experimental high-quality path can also solve short causal trajectory blocks without a Python loop over every token. In `directional_block_cumsum`, local directional deltas are evaluated in parallel inside blocks, prefix states are recovered with `torch.cumsum`, and optional causal Anderson refinement computes prefix-only coefficients with cumulative Gram matrices plus batched small linear solves. This keeps autoregressive prefix causality while replacing the strict one-step-at-a-time loop with a blockwise solver.

The promoted architecture is **ASM-R**, represented experimentally by
`J_NO_DIRECTION`: a direct contextual transition, relational metric
naturalization, causal mixer, token-to-state residual, and selective
forget/write memory. It completed three independent 100M-token runs with mean
frozen-validation CE **1.344538** and population standard deviation
**0.000561**. ASM-S remains the faster efficiency-oriented variant.

The working hypothesis is that language generation can be modeled as motion through a relational state space, where geometry is measurable through action, condition, active dimension, recurrence, stability, and low-action path diagnostics.

## Install

```bash
pip install -e .
```

Optional dev tools:

```bash
pip install -e ".[dev]"
```

The project is CPU-runnable. CUDA is optional but recommended for the larger memmap benchmarks.

## Quickstart

Train a tiny legacy DRM/ASM-X model:

```bash
python scripts/train_tiny.py --config configs/tiny.yaml --text data/tiny.txt
```

Generate text:

```bash
python scripts/generate.py --checkpoint runs/tiny/drm_tiny.pt --prompt "DRM "
```

Run geometry diagnostics:

```bash
python scripts/eval_geometry.py --checkpoint runs/tiny/drm_tiny.pt
python scripts/eval_geodesic_paths.py --checkpoint runs/tiny/drm_tiny.pt
```

If `data/tiny.txt` is missing, the training script creates a tiny fallback corpus. The default tokenizer is byte-level, so mixed case, digits, punctuation, and prompts such as `DRM` are representable.

## Architecture Family

The family taxonomy is:

| Code | Architecture | Experimental variant |
|---|---|---|
| ASM-X | Explicit DRM State Model | J |
| ASM-U | Metric Subspace State Model | J_METRIC_SUBSPACE |
| ASM-F | Relational Frame State Model | J_METRIC_ORTHONORMAL_DIRECTION |
| ASM-R | Relational State Model | J_NO_DIRECTION |
| ASM-C | Compact State Model | ASM-R weights + compact streaming inference |
| ASM-C2 | Compact Addressable State Model | ASM-C + bounded key/value slots |
| ASM-C2-FW | Compact Fast-Weight State Model | ASM-C + bounded delta-rule associative matrix |
| ASM-D | Direct State Model | J_DIRECT_CONTROL |
| ASM-S | Selective State Model | J_DIRECT_CONTROL_MATCHED |
| ASM-M | Causal Memory State Model | SSM_CONTROL |

See [MODEL_FAMILY.md](docs/MODEL_FAMILY.md) for definitions and promotion
criteria.

The first ASM-C validation kept cache (`6,144 B`), peak CUDA allocation
(`387.53 MiB`), and throughput (`~503 tok/s`) effectively flat through 32K,
reaching `2.97x` the legacy ASM-R streaming throughput at 32K. Its short MQAR
control failed (`32.25%` versus the `80%` gate), so long-range associative
memory is not yet established. See the
[versioned ASM-C benchmark](docs/benchmarks/asm_c_streaming_32k/README.md).

### ASM-X recurrent path

Default recurrent path:

```text
input_ids
  |
TokenEmbedding
  |
for each time step:
  z_t
   |
DirectionField(z_t) -> directions V(z_t), gates a(z_t), dimD
   |
RelationalMetric(z_t) -> diag + U U^T
   |
DRMFlow(z_t, e_t, V, a) -> dz in active directional span
   |
metric action g_z(dz, dz)
   |
StateUpdater -> z_{t+1}
   |
LanguageEmitter(z_{t+1}) -> logits
```

Experimental blockwise path:

```text
input_ids
  |
TokenEmbedding
  |
split sequence into causal blocks
  |
evaluate local directional candidates in parallel from z_block
  |
prefix cumsum of local deltas -> approximate states
  |
optional causal Anderson refinement:
  residual_history -> prefix Gram cumsum -> batched solve -> prefix-only coefficients
  |
LanguageEmitter(states) -> logits
```

The model is autoregressive, but its memory is the evolving latent state rather than attention over a token sequence. The causal Anderson path is not self-attention: it mixes only solver histories for prefix-consistent trajectory refinement, and tests check that future tokens do not change prefix states/logits.

Current 125M local-mixer path:

```text
input_ids
  |
TokenEmbedding
  |
split sequence into causal block64 chunks
  |
parallel directional velocity candidates
  |
prefix cumsum of local deltas -> causal states
  |
causal local mixer over state/features
  |
token residual + selective forget/write memory (variant J)
  |
LanguageEmitter(states) -> logits
```

Read the full design in [ARCHITECTURE.md](ARCHITECTURE.md). The planned formal DRM implementation layers, including relational transport, holonomy diagnostics, effective rank, Fisher-Rao pullback, toroidal state dynamics, and explicit anchor maps, are tracked in [roadmap.md](roadmap.md).

## Main Components

- `src/aletheion_state_models/core/`: architecture-neutral state, transition, memory, mixer, and emitter interfaces.
- `src/aletheion_state_models/geometry/`: optional metric, directional basis, and naturalization operators.
- `src/aletheion_state_models/variants/`: named ASM-X, ASM-U, ASM-F, ASM-R, ASM-C, ASM-C2, ASM-C2-FW, ASM-D, ASM-S, and ASM-M constructors.
- `src/drm_language_emitter/`: checkpoint-compatible legacy implementation retained during migration.

- `src/drm_language_emitter/config.py`: validated `DRMConfig` schema.
- `src/drm_language_emitter/model.py`: model assembly and core recurrent forward path.
- `src/drm_language_emitter/model_components.py`: state initializer, causal mixer, direct control transition, and selective memory.
- `src/drm_language_emitter/selective_control.py`: geometry-free selective-memory control.
- `src/drm_language_emitter/mqar.py`: synthetic associative-recall data.
- `src/drm_language_emitter/direction_field.py`: active directional fields and gates.
- `src/drm_language_emitter/metric.py`: relational metric `diag + U U^T` and naturalization.
- `src/drm_language_emitter/dynamics.py`: directional flow and state update.
- `src/drm_language_emitter/directional_forward.py`: directional cumsum forward path, losses, and diagnostics.
- `src/drm_language_emitter/directional_blocks.py`: block and superblock trajectory construction.
- `src/drm_language_emitter/directional_solvers.py`: fixed-point, causal Anderson, and transition helpers.
- `src/drm_language_emitter/geometric_steps.py`: state bounding, geodesic refinement, and directional candidates.
- `src/drm_language_emitter/deer.py`: reusable trajectory solvers.
- `src/drm_language_emitter/emitter.py`: token embedding and language-emission head.
- `src/drm_language_emitter/generation.py`: autoregressive generation.
- `src/drm_language_emitter/data.py`: in-memory and memory-mapped language-model datasets.
- `src/drm_language_emitter/checkpoint.py`: validated weights-only checkpoint loading.
- `transformer/`: Transformer baselines.
- `world_model/`: symbolic seq2seq world-model baseline.

## Diagnostics

The code logs and exports:

- cross entropy and approximate perplexity;
- metric action;
- effective active dimension `dimD`;
- gate entropy and hard/soft active fractions;
- metric low-rank norm and condition proxy;
- recurrence and stability proxies;
- learned low-action path diagnostics;
- symbolic world-modeling metrics in the gridworld benchmark.

Important caveat: `scripts/eval_geodesic_paths.py` evaluates learned low-action trajectories. It is not an exact geodesic solver.

Another caveat: some diagnostics in experimental blockwise modes summarize approximate trajectory behavior rather than the exact recurrent path. Treat them as engineering diagnostics until each metric is explicitly validated for the selected `sequence_mode`.

## Benchmarks

Benchmark outputs that are small enough to keep are copied to `docs/benchmarks/`. Large run directories remain under `runs/` and are ignored by git.

### Retracted GPT-2 comparisons

The previously published 125M/150M-token and 36M time-to-quality comparisons
against GPT-2 are **deprecated and retracted as comparative evidence**.

The historical GPT-2 training path shifted next-token labels before passing
them to a Hugging Face causal-LM implementation that performs its own internal
shift. GPT-2 was therefore trained against $x_{t+2}$ instead of the intended
$x_{t+1}$. This double-shift made the GPT-2 CE artificially poor. The bug has
been corrected in `scripts/train_gpt2_memmap.py` and covered by regression
tests.

Consequences:

- the old claims that DRM beat GPT-2 at 36M or 125M are withdrawn;
- old target-reached and time-to-quality conclusions are invalid;
- the artifacts remain available only for audit/history;
- DRM-only curves may describe those DRM runs, but they cannot validate a
  DRM-versus-GPT-2 conclusion;
- no current README claim states that DRM outperforms GPT-2.

Historical, invalid-for-comparison artifacts:

```text
docs/benchmarks/competition_125m_local_mixer_h256_l2_s02_150m/
docs/benchmarks/tta/
```

### Current valid ASM component evidence

The current controlled 5M-token, three-seed result uses deterministic
continuous validation over 4,834,787 targets:

| Variant | Parameters | Validation CE mean | Std | Interpretation |
|---|---:|---:|---:|---|
| I | 127.01M | 1.878244 | 0.000647 | geometry baseline |
| J | 126.08M | **1.760581** | 0.003057 | geometry + selective memory |
| SSM_CONTROL | 126.08M | 1.806518 | 0.006191 | selective memory without geometry |

J beat SSM_CONTROL in all three seeds by 0.045937 CE on average, while the
control trained about 2.5x faster. This historical component result motivated
the decomposed ASM family; it is not a comparison with Mamba or GPT-2.

At 30M tokens and three paired seeds, ASM-R (`J_NO_DIRECTION`) achieved mean
validation CE `1.477576`, while parameter-matched ASM-S
(`J_DIRECT_CONTROL_MATCHED`) achieved `1.487258`. At 5M the order was reversed.
The continuous confirmation then took ASM-R to 100M across three seeds:

| Tokens | ASM-R validation CE mean | Population std |
|---:|---:|---:|
| 5M | 1.750925 | 0.000363 |
| 30M | 1.465967 | 0.000794 |
| 50M | 1.411406 | 0.001656 |
| 100M | **1.344538** | **0.000561** |

ASM-R is therefore promoted as the main quality-per-token architecture. ASM-F
generation 1 is not a valid 100M multiseed competitor: seeds 2 and 3 diverged
before 70M and produced fully non-finite 100M checkpoints. A numerically
stabilized ASM-F rerun is classified as a second-generation experiment.

See the [multiseed benchmark](docs/benchmarks/asm_r_confirmation_100m_multiseed/README.md)
and [report 037](docs/report/037_Confirmacao_Multiseed_100M_e_Promocao_ASM_R_2026_08_01.md).

See [report 027](docs/report/027_Contribuicao_Geometrica_J_vs_SSM_Control_e_Proximas_Ablacoes_2026_07_31.md).

### DRM vs Transformer

Versioned dashboard:

```text
docs/benchmarks/drm_transformer_full_1k_3k/dashboard.html
```

Run the sweep:

```bash
python scripts/sweep_drm_transformer.py --steps 1000 2000 3000 --seeds 1 2 3 --output-root runs/sweep_drm_transformer
python scripts/make_competition_dashboard.py --root runs/sweep_drm_transformer --title "DRM vs Transformer Sweep"
```

Current interpretation: DRM showed strong step-matched and parameter-matched results in the tiny regime, while Transformer throughput remains much higher. This does not establish broad superiority.

### DRM vs Transformer vs Tiny World Model

Versioned dashboard:

```text
docs/benchmarks/world_model_competition/dashboard.html
```

Run the benchmark:

```bash
python scripts/make_tiny_world_dataset.py --output-root data/tiny_world --seed 1 --grid-size 5 --num-train 20000 --num-val 2000 --max-rollout-len 8
python scripts/sweep_world_model_competition.py --steps 1000 2000 3000 --seeds 1 2 3 --dataset-root data/tiny_world --output-root runs/world_model_competition
python scripts/make_world_model_dashboard.py --root runs/world_model_competition --title "DRM vs Transformer vs Tiny Symbolic World Model"
```

Latest local result:

- 72 runs, 24 aggregate rows.
- Best next-state exact match: `drm_tiny @ 2000` with `0.0751`.
- Lowest invalid-state rate among top next-state rows: `transformer_tiny_220k @ 3000` with `0.0026`.
- Best supervised world-model CE among top rows: `world_model_tiny @ 3000` around `0.2497`, but exact-match metrics remained low.

Interpretation: DRM had the best next-state exact-match score in this tiny symbolic text-world benchmark, but absolute symbolic accuracy is still low. This is a diagnostic result, not evidence that DRM is broadly better than Transformers or general world models.

See [docs/report/002_world_model_competition_2026-06-18.md](docs/report/002_world_model_competition_2026-06-18.md).

## Useful Commands

Quick DRM vs Transformer comparison:

```bash
python scripts/compare_drm_transformer.py --steps 50 --batch-size 4 --output-root runs/quick_compare
```

Robustness:

```bash
python scripts/eval_robustness.py --drm-checkpoint runs/quick_compare/drm/drm_tiny.pt --drm-tokenizer runs/quick_compare/drm/tokenizer.json --transformer-checkpoint runs/quick_compare/transformer/tiny_transformer.pt
```

Bridge diagnostic:

```bash
python scripts/eval_bridge_task.py --checkpoint runs/quick_compare/drm/drm_tiny.pt --tokenizer runs/quick_compare/drm/tokenizer.json
```

Sequence stability:

```bash
python scripts/eval_sequence_stability.py --drm-checkpoint runs/quick_compare/drm/drm_tiny.pt --drm-tokenizer runs/quick_compare/drm/tokenizer.json --transformer-checkpoint runs/quick_compare/transformer/tiny_transformer.pt
```

DRM profile:

```bash
python scripts/profile_drm.py --checkpoint runs/quick_compare/drm/drm_tiny.pt
```

## Tests

Evaluate a promoted ASM-R checkpoint with one command:

```bash
./scripts/run_asm_r_post_promotion_suite.sh \
  --checkpoint runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt \
  --output-root runs/asm_r_post_promotion_quick \
  --device cuda \
  --quick
```

Read the [suite report](docs/report/038_Suite_Avaliacao_Pos_Promocao_ASM_R_2026_08_01.md)
before running `--full`.

```bash
python -m pytest -q
```

If `pytest` is not installed:

```bash
pip install -e ".[dev]"
```

CUDA tests are conditional. They run only when `torch.cuda.is_available()` is true.

## Repository Map

```text
configs/                 DRM and benchmark configs
docs/                    math, limitations, competition notes, benchmark artifacts
scripts/                 training, generation, evaluation, sweeps, dashboards
src/aletheion_state_models/ ASM family and neutral public interfaces
src/drm_language_emitter/ checkpoint-compatible legacy implementation
tests/                   smoke and invariant tests
transformer/             tiny Transformer baseline
world_model/             tiny symbolic world-model baseline
```

## Scientific Status

Allowed claims:

- ASM is a functional attention-free causal state-model research family.
- ASM-X preserves the explicit DRM architecture inside the family.
- ASM-R is the promoted quality-per-token architecture after three stable
  100M-token seeds with frozen-validation CE `1.344538 ± 0.000561`.
- ASM-R and ASM-S exhibit a measured ranking crossover between 5M and 30M tokens.
- ASM-F generation 1 diverged before 70M in both additional seeds; a stabilized
  ASM-F is a separate second-generation experiment.
- Its geometry is explicit, measurable, and trainable in small experiments.
- The repository includes controlled tiny comparisons against Transformer and a tiny symbolic world model.
- The repository includes an experimental causal blockwise trajectory solver using prefix cumsum and causal Anderson refinement.
- The old 36M and 125M GPT-2 comparisons are retracted because the GPT-2
  labels were double-shifted.
- In an internal 5M-token component ablation, J beat the parameter-matched
  SSM_CONTROL in all three seeds.

Not allowed:

- DRM is better than Transformers in general.
- DRM is better than world models in general.
- The retracted GPT-2 benchmarks demonstrate DRM superiority.
- The 125M base LM checkpoints are chat or instruction-tuned models.
- The model has proven emergent geodesics.
- The model has proven toroidal topology.
- The model is production-ready or safety-evaluated.

## Limitations

- The default recurrent path is slow compared with optimized Transformer kernels.
- The experimental blockwise causal Anderson path removes the strict token-by-token Python loop inside blocks, but its b8 solver became too slow at 125M scale.
- No corrected, completed multiseed GPT-2 comparison currently establishes a
  DRM advantage.
- Benchmarks are diagnostic and tied to the exact dataset, tokenizer, optimizer, hardware, and run scripts in this repository.
- Historical GPT-2 dashboards remain for audit but are invalid for comparative
  conclusions.
- The shared prefill/decode path has parity coverage; incremental decoding
  still has performance limitations documented in the inference reports.
- Low-action path evaluation is not a formal geodesic solver.
- Symbolic world-modeling exact match is still low.
- No RLHF, alignment evaluation, instruction tuning, or safety validation is included.
- Toroidal convergence is not guaranteed; it is only a possible diagnostic under boundedness, recurrence, and stability assumptions.

## Roadmap

- Add stronger trajectory integrators and variational path objectives.
- Improve constrained symbolic decoding for the world benchmark.
- Add time-matched CUDA comparisons.
- Add larger-token and larger-parameter time-to-quality continuations.
- Broaden ablations around metric, gates, and active dimension.
- Study pullback/Fisher-style metrics as future work.

## License

Copyright © 2026 Felipe Maya Muniz
SPDX-License-Identifier: AGPL-3.0-only

This project is dual-licensed:
- GNU AGPL v3.0 only; or
- a separate commercial license.

Commercial licensing and acquisition inquiries:
[LICENCE-COMMERCIAL.md](LICENCE-COMMERCIAL.md) or contact `felupe@truthagi.ai`.
