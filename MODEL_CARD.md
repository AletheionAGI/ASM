# Model Card

## Model

ASM — Aletheion State Models, version `0.2.0`.

## Intended Use

Research on attention-free causal state models, including explicit DRM,
metric-conditioned, direct-transition, and selective-memory variants.

## Out-of-Scope Use

Do not use this model for production language generation, safety-critical tasks, medical/legal/financial advice, autonomous agents, or user-facing deployment.

## Architecture

The family is autoregressive and trained with next-token prediction. All
current variants use a persistent causal state. Geometry, explicit directions,
and selective-memory capacity vary by ASM code. ASM-X is the explicit DRM
variant; ASM-R and ASM-S are the current scaling-law finalists. No current
variant uses Transformer blocks or self-attention.

ASM-C is the experimental compact-streaming inference form of ASM-R. It reuses
ASM-R weights and is not yet promoted as a separate trained language model.
Its first 32K validation passed bounded-cache, peak-VRAM, and throughput gates,
but failed the short MQAR control; long-range associative memory is therefore
not established.

## Training Data

The default script uses a tiny local text file or fallback corpus. This is only a smoke test and has no meaningful coverage.

## Evaluation

Current evidence is experimental and architecture-internal. Tests verify
shape, finite losses, causality, metric positivity where applicable,
checkpointing, scaling-law instrumentation, and absence of attention modules.

## Safety

No RLHF, red-teaming, content filtering, alignment method, jailbreak evaluation, or safety benchmark has been performed.

## Limitations

The model is experimental, slow, minimally trained, and unvalidated. It should be read as architecture research code rather than a capable language model.
