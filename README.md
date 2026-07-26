# DRM Language Emitter

**A geometry-first language model lab for building generative AI without attention, without Q/K/V, and without Transformer blocks.**

![DRM Language Emitter manifold banner](assets/drm-language-emitter-banner.svg)

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-pure%20torch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](pyproject.toml)
[![Non Transformer](https://img.shields.io/badge/architecture-non--Transformer-14B8A6?style=for-the-badge)](ARCHITECTURE.md)
[![No Attention](https://img.shields.io/badge/attention-none-0F172A?style=for-the-badge)](tests/test_no_transformer.py)
[![Benchmarks](https://img.shields.io/badge/benchmarks-reproducible-F59E0B?style=for-the-badge)](docs/benchmarks/README.md)
[![License](https://img.shields.io/badge/license-AGPL--3.0-64748B?style=for-the-badge)](LICENSE)

DRM Language Emitter turns language generation into controlled motion through a learned relational manifold: active directions choose where the model can move, a learned metric shapes how expensive that movement is, and an emitter decodes the resulting state into tokens.

This is a research scaffold, not a production model and not a claim of superiority over Transformers or general world models.

## Quick Links

- [Architecture](ARCHITECTURE.md)
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

## What Makes It Different

DRM Language Emitter does not use:

- Transformer blocks;
- self-attention;
- Q/K/V attention;
- `nn.MultiheadAttention`;
- KV cache.

Its central computation is a latent trajectory. The baseline path is a causal latent recurrence:

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

Train a tiny DRM model:

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

## Architecture

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

Read the full design in [ARCHITECTURE.md](ARCHITECTURE.md). The planned formal DRM implementation layers, including relational transport, holonomy diagnostics, effective rank, Fisher-Rao pullback, toroidal state dynamics, and explicit anchor maps, are tracked in [roadmap.md](roadmap.md).

## Main Components

- `src/drm_language_emitter/config.py`: `DRMConfig`
- `src/drm_language_emitter/direction_field.py`: active directional fields and gates
- `src/drm_language_emitter/metric.py`: relational metric `diag + U U^T`
- `src/drm_language_emitter/dynamics.py`: DRM flow and metric naturalization
- `src/drm_language_emitter/deer.py`: blockwise fixed-point and causal Anderson trajectory solvers
- `src/drm_language_emitter/model.py`: causal language emitter
- `transformer/`: tiny Transformer baseline
- `world_model/`: tiny symbolic seq2seq world-model baseline

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

### Time-To-Quality: DRM Causal Anderson vs GPT-2 36M

Versioned dashboard:

```text
docs/benchmarks/tta/dashboard.html
```

Latest local seed-1 result:

| Model | Tokens Seen | Best Val CE | Target Reached | Time To Target |
|---|---:|---:|---:|---:|
| DRM causal Anderson b8 | 20,004,864 | 1.8295 | yes | 2,947.9s |
| GPT-2 36M | 22,005,760 | 2.0715 | no | >701.1s |

The target was `best_val_ce_DRM + 0.01 = 1.8395`. GPT-2 was required to train beyond the DRM token floor before plateau stopping was accepted. This is a single-seed diagnostic result, not a final general claim; multi-seed and larger-scale confirmation are still required.

Run the controller:

```powershell
.\scripts\run_time_to_quality.ps1 -Seeds 1
```

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
src/drm_language_emitter/ DRM model package
tests/                   smoke and invariant tests
transformer/             tiny Transformer baseline
world_model/             tiny symbolic world-model baseline
```

## Scientific Status

Allowed claims:

- DRM Language Emitter is a functional non-Transformer language model prototype.
- Its geometry is explicit, measurable, and trainable in small experiments.
- The repository includes controlled tiny comparisons against Transformer and a tiny symbolic world model.
- The repository includes an experimental causal blockwise trajectory solver using prefix cumsum and causal Anderson refinement.
- In one seed-1 time-to-quality run, DRM causal Anderson b8 reached a CE target that GPT-2 36M did not reach before plateau under the tested token floor.

Not allowed:

- DRM is better than Transformers in general.
- DRM is better than world models in general.
- The seed-1 time-to-quality result is definitive across seeds, scales, or datasets.
- The model has proven emergent geodesics.
- The model has proven toroidal topology.
- The model is production-ready or safety-evaluated.

## Limitations

- The default recurrent path is slow compared with optimized Transformer kernels.
- The experimental blockwise causal Anderson path removes the strict token-by-token Python loop inside blocks, but its solver is still much slower than GPT-2 kernels in current benchmarks.
- Benchmarks are tiny and diagnostic.
- The strongest time-to-quality result currently versioned is single-seed and needs multi-seed confirmation.
- Low-action path evaluation is not a formal geodesic solver.
- Symbolic world-modeling exact match is still low.
- No large-scale benchmark, RLHF, alignment evaluation, or safety validation is included.
- Toroidal convergence is not guaranteed; it is only a possible diagnostic under boundedness, recurrence, and stability assumptions.

## Roadmap

- Add stronger trajectory integrators and variational path objectives.
- Improve constrained symbolic decoding for the world benchmark.
- Add time-matched CUDA comparisons.
- Add multi-seed time-to-quality comparisons and larger-token continuations.
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
