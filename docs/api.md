# API Reference

This page documents the stable public surface used by scripts and tests. The project is still a research scaffold, so internal module details may change between experiments.

## ASM family API

The neutral public namespace is `aletheion_state_models`:

```python
from aletheion_state_models import StateModel
from aletheion_state_models.variants import (
    build_compact_streaming,
    build_compact_addressable,
    build_compact_fast_weight,
    build_direct_state,
    build_explicit_drm,
    build_metric_subspace,
    build_relational_state,
    build_selective_state,
)
```

`build_compact_streaming(config)` constructs ASM-C, the compact inference form
of ASM-R. It enables bounded fixed-block streaming state while preserving the
ASM-R parameterization and checkpoint keys.

`build_compact_addressable(config, slots=32)` constructs ASM-C2 with a bounded
content-addressable memory. `read_enabled` and `write_enabled` expose causal
ablations without changing the surrounding ASM-C transition.

`build_compact_fast_weight(config)` constructs ASM-C2-FW with a bounded
delta-rule associative matrix and learned causal read, write, and retention
gates.

`StateModel` is currently an alias of `DRMEmitterModel`, not a subclass. This
preserves exact state-dict keys and checkpoint behavior during migration.

The `drm_language_emitter` namespace remains supported as the legacy,
checkpoint-compatible implementation. It is not being removed in version
`0.2.0`.

## Legacy-compatible configuration

`drm_language_emitter.config.DRMConfig`

Main fields:

- `vocab_size`, `d_token`, `d_state`, `n_directions`, `metric_rank`, `hidden_size`: model shape.
- `n_flow_steps`, `dt`, `bounded_state`: recurrent state update controls.
- `use_powerlaw_risk`, `risk_mass_max`, `risk_exponent_min`, `risk_exponent_max`, `risk_alpha_max`: blindspot/dubiety risk controls.
- `use_metric_naturalization`, `metric_naturalization_strength`, `metric_damping`: metric preconditioning controls.
- `use_torch_compile`: opt-in compilation of the DRM forward path with fallback to eager execution.
- `compact_streaming_inference`: enables the ASM-C inference state; it requires
  a supported fixed-block sequence mode and fails explicitly otherwise.
- `addressable_memory` and `addressable_memory_*`: enable and configure the
  ASM-C2 read/write memory. `addressable_memory_backend` selects `slots` or
  `fast_weight`. The current implementation supports one
  read/write head and requires compact fixed-block inference.
- `epistemic_memory_gating`: enables the experimental ASM-CM-E reliability
  gates on fast-weight reads and writes. `epistemic_gate_hidden_dim`,
  `epistemic_gate_num_layers`, `epistemic_gate_dropout`, and
  `epistemic_gate_initial_confidence` configure them. This option does not
  replace language cross-entropy or mix uncertain memories uniformly.

`DRMConfig.from_dict(data)` rejects unknown keys. This is intentional: experiment config typos should fail before training starts.

## Model

`drm_language_emitter.model.DRMEmitterModel`

```python
out = model(input_ids, targets=None, return_states=False, global_step=None)
```

Inputs:

- `input_ids`: `LongTensor` with shape `[batch, seq_len]`.
- `targets`: optional `LongTensor` with shape `[batch, seq_len]`.
- `return_states`: when true, includes latent states with shape `[batch, seq_len, d_state]`.
- `global_step`: optional integer used for metric naturalization warmup.

Output keys:

- `logits`: token logits `[batch, seq_len, vocab_size]`.
- `loss`: total scalar loss.
- `aux_losses`: component losses.
- `diagnostics`: scalar tensors for geometry, gates, action, metric condition, and risk.
- `states`: present only when `return_states=True`.

## Generation

`drm_language_emitter.generation.generate`

```python
tokens = generate(model, input_ids, max_new_tokens=32, temperature=0.9, top_k=20)
```

Generation replays the prompt into the latent state, samples from the emitter,
and advances the state with each generated token. It does not use attention or
a KV cache. In ASM-C mode, incremental inference retains the completed state,
bounded open block, selective-memory state, and position counter rather than
the complete token prefix.

## Tokenizers

- `ByteTokenizer`: fixed UTF-8 byte vocabulary of size 256.
- `CharTokenizer`: character vocabulary trained from supplied text.

Use `drm_language_emitter.tokenizer.load_tokenizer(path)` to reload saved tokenizer metadata.

## Core Modules

- `DirectionField`: maps latent state `z` to directions `[B, n_directions, d_state]` and gates `[B, n_directions]`.
- `RelationalMetric`: returns a positive diagonal metric and optional low-rank factor `U`.
- `DRMFlow`: computes velocity constrained to active directions.
- `StateUpdater`: applies the recurrent state update and bounded-state clipping.
- `RiskField`: optional bounded risk signal that thickens metric energy.
- `LanguageEmitter`: decodes latent state to token logits.
