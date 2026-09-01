# Benchmark Artifacts

This directory stores lightweight, versionable benchmark outputs.

Generated training directories under `runs/` remain local working artifacts and are ignored by git. The selected dashboards, summaries, CSVs, and SVG plots are copied here when a result should be preserved in the repository.

## Available Benchmarks

- `asm_cm_vr_fixed32_phase1/`
  - Strict rank-aware ASM-CM-VR full-64 versus fixed-32 seed-17 smoke.
  - Both reached 100% held-out MQAR-40; fixed-32 passed anti-bypass, causal
    read/write ablations, full/stream parity, and retained-state streaming to 4K.
  - Main entry: `asm_cm_vr_fixed32_phase1/README.md`

- `asm_cm_vr_fixed32_long/`
  - Full-64, fixed-32, and exploratory adaptive-32 curriculum through 4K with
    frozen MQAR/streaming extrapolation to 32K in seeds 17, 29, and 43.
  - Fixed-32 was not promoted: full/fixed passed 2/3 seeds and both recorded
    non-finite seed-29 failures; logical rank did not reduce retained bytes.
  - Main entry: `asm_cm_vr_fixed32_long/README.md`

- `asm_cm_vs_vr_s_pmcs64/`
  - PMCS-64 parameter-matched comparison of ASM-CM, ASM-VR-S full, and
    ASM-VR-S fixed-32 across language, mixed MQAR specialization, and retained-state streaming.
  - Separates total/trainable parameters, logical rank, state bytes, numerical
    failures, and physical throughput without converting logical rank into a speed claim.
  - Main entry: `asm_cm_vs_vr_s_pmcs64/README.md`

- `asm_vr_phase3a2/`
  - Parameter-matched ASM-VR-R versus ASM-VR-S across five rank policies and three seeds.
  - ASM-VR-S won all paired quality comparisons; adaptive controller still failed fixed frontier.
  - Main entry: `asm_vr_phase3a2/README.md`
- `asm_vr_phase3a2_adamm_confirm/`
  - New-seed AdamM confirmation of the promoted ASM-VR-S base.
  - Main entry: `asm_vr_phase3a2_adamm_confirm/README.md`
- `asm_vr_phase3a3_rs/`
  - Formal ASM-VR-RS full-rank comparison against R and S.
  - Main entry: `asm_vr_phase3a3_rs/README.md`

- `asm_vr_phase3a1_adamm/`
  - Matched AdamW versus AdamM optimizer ablation on fixed-32 and adaptive-32.
  - AdamM improved both arms by about 0.02 nat but did not rescue adaptive rank.
  - Includes PNG/SVG charts and offline dashboard.
  - Main entry: `asm_vr_phase3a1_adamm/README.md`

- `asm_vr_phase3a1/`
  - Stage A: factorial projected-scaffold ablation with separate PNG/SVG dashboard.
  - Stage B: full/fixed/adaptive rank matrix on the selected mixer+residual scaffold.
  - Quality was recovered; adaptive rank still failed the fixed-rank Pareto frontier.
  - Main entry: `asm_vr_phase3a1/README.md`

- `asm_vr_phase3a/`
  - Small-scale byte-language matrix: six variants, three seeds, ~2M tokens/run.
  - Includes paired quality/rank/cost results, PNG/SVG charts, and offline HTML dashboard.
  - Operational gates passed; adaptive rank was not Pareto-superior to fixed rank 16.
  - Main entry: `asm_vr_phase3a/README.md`

- `asm_vr_phase2/`
  - Multiseed variable-capacity copy benchmark for trainable hard rank.
  - Confirms rank/difficulty adaptation and quality near a fixed-rank control.
  - Dense execution only; no hardware speedup claim.
  - Main entry: `asm_vr_phase2/README.md`

- `asm_vr_phase1/`
  - Completed no-bypass gate integrated with the ASM-R direct-transition core.
  - Confirms paired logits/state equality, zero discarded Jacobian, compact
    cache, and full-forward versus streaming parity.
  - Main entry: `asm_vr_phase1/README.md`

- `asm_vr_phase0/`
  - Isolated invariants for the experimental ASM-VR state contract.
  - Verifies hard-projector idempotence, discarded-information probes, the
    external-memory positive control, and Jacobian rank for an `8→3→5→8` cycle.
  - This is not a trained language-model or efficiency result.
  - Main entry: `asm_vr_phase0/README.md`

- `asm_cm_post_fp32/`
  - Final frozen three-seed revalidation of promoted ASM-CM after the FP32
    recurrent-core correction; no checkpoint was retrained.
  - CE: `1.328496 ± 0.000687`; fixed 32K cache: `143,360 bytes`; mean 32K
    throughput: `80.68 tok/s`; mean peak VRAM: `363.66 MiB`.
  - Main entry: `asm_cm_post_fp32/README.md`

- `asm_c_streaming_32k/`
  - Compact ASM-C streaming validation against the legacy ASM-R inference path
    through 32K tokens, plus BF16 parity, MQAR, and a paired Transformer probe.
  - Streaming criteria passed; the short MQAR control failed.
  - Main entry: `asm_c_streaming_32k/README.md`

- `asm_r_confirmation_100m_multiseed/`
  - Promoted ASM-R result at 100M tokens with seeds 1, 2, and 3, using the
    same frozen continuous validation traversal at every milestone.
  - Final CE: `1.344538 ± 0.000561` (population standard deviation).
  - Records the invalid ASM-F generation-1 multiseed outcome separately.
  - Main entry: `asm_r_confirmation_100m_multiseed/README.md`

- `asm_scaling_law_100m_seed1/`
  - Exploratory seed-1 scaling-law comparison of ASM-R, ASM-X, ASM-F, and
    ASM-S from 1M through 100M training tokens.
  - Main entry: `asm_scaling_law_100m_seed1/README.md`

- `drm_transformer_full_1k_3k/`
  - Step-matched and parameter-matched DRM vs Tiny Transformer sweep.
  - Main entry: `drm_transformer_full_1k_3k/dashboard.html`

- `world_model_competition/`
  - DRM vs Tiny Transformer vs tiny symbolic `world_model/` benchmark.
  - Main entry: `world_model_competition/dashboard.html`

- `bench_36M/`
  - DRM vs GPT-2-style vs OPT-style parameter-matched language-model comparison around 37M parameters.
  - Main entry: `bench_36M/dashboard.html`

- `bench_125M/`
  - DRM vs GPT-2-style vs OPT-style parameter-matched language-model comparison around 125M parameters.
  - Main entry: `bench_125M/dashboard.html`

- `scale_lm_comparison/` (local run target)
  - DRM vs GPT-2-style vs OPT-style 125M/350M architecture comparison.
  - Generated by `scripts/run_scale_lm_comparison.py`.

- `attr_p0_smoke/`
  - CPU integration smoke for HazardWorld, common ASM/Transformer heads,
    leakage audits, classical controls, and offline rendering.
  - It is not a sealed benchmark and supports no safety or superiority claim.
  - Main entry: `attr_p0_smoke/index.html`

- `asm_transformer_transition_risk/p2/`
  - Completed five-seed sealed ATTR P2 benchmark with the six registered arms,
    plus a separately labeled post-hoc comparison of ASM-X Base against
    ASM-X + Native Risk Mass.
  - Native Risk Mass was operationally indistinguishable from Base under the
    frozen configuration and does not revise the registered P2 gates.
  - Main entry: `asm_transformer_transition_risk/p2/index.html`

- `asm_transformer_transition_risk/pilot_seed_17/`
  - ATTR P1 train/validation-only pilot with 1,000 updates per arm: registered
    ASM-X/Transformer pair plus supplementary ASM-CM, VR-S full/fixed, and ASM-R.
  - Different metrics had different leaders; test remained sealed, registered
    margins were unchanged, and no safety/predictive gate was claimed.
  - Main entry: `asm_transformer_transition_risk/pilot_seed_17/index.html`

## Reproducibility

The corresponding commands are documented in:

- `docs/competition.md`
- `docs/world_model_competition.md`
- `docs/scale_lm_comparison.md`
- `docs/report/001_drm_competition_protocol_2026-06-17.md`
- `docs/report/002_world_model_competition_2026-06-18.md`
