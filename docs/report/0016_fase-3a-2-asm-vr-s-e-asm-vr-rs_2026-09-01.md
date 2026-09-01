# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Executar o experimento completo usando ASM-S como base do ASM-VR, incluindo matriz R versus S, confirmação AdamM, e depois formalizar e testar ASM-VR-RS — Variable-Rank Relational Selective State Emitter.

## Summary

A Fase 3A.2 executou 30 runs AdamW parameter-matched R/S e promoveu ASM-VR-S por superioridade consistente de qualidade: -0.0485 nat full e -0.0503 nat fixed-32, com vitória em 15/15 pares. AdamM confirmou -0.0451 nat em 3/3 seeds novas. ASM-VR-RS full foi formalizado e executado em três seeds, mas ASM-VR-S continuou melhor com menos parâmetros, memória e maior throughput. O controller adaptativo permaneceu fora da fronteira fixa em ambas as bases.

## Modified files

- [docs/ARCHITECTURE_ASM_VR.md](../ARCHITECTURE_ASM_VR.md)
- [docs/MODEL_FAMILY.md](../MODEL_FAMILY.md)
- [docs/MODEL_FAMILY_ptbr.md](../MODEL_FAMILY_ptbr.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_vr_phase3a2/README.md](../benchmarks/asm_vr_phase3a2/README.md)
- [docs/benchmarks/asm_vr_phase3a2/adaptive_rank_ranges.png](../benchmarks/asm_vr_phase3a2/adaptive_rank_ranges.png)
- [docs/benchmarks/asm_vr_phase3a2/adaptive_rank_ranges.svg](../benchmarks/asm_vr_phase3a2/adaptive_rank_ranges.svg)
- [docs/benchmarks/asm_vr_phase3a2/index.html](../benchmarks/asm_vr_phase3a2/index.html)
- [docs/benchmarks/asm_vr_phase3a2/manifest.json](../benchmarks/asm_vr_phase3a2/manifest.json)
- [docs/benchmarks/asm_vr_phase3a2/observed_dense_cost.png](../benchmarks/asm_vr_phase3a2/observed_dense_cost.png)
- [docs/benchmarks/asm_vr_phase3a2/observed_dense_cost.svg](../benchmarks/asm_vr_phase3a2/observed_dense_cost.svg)
- [docs/benchmarks/asm_vr_phase3a2/paired_s_minus_r_deltas.png](../benchmarks/asm_vr_phase3a2/paired_s_minus_r_deltas.png)
- [docs/benchmarks/asm_vr_phase3a2/paired_s_minus_r_deltas.svg](../benchmarks/asm_vr_phase3a2/paired_s_minus_r_deltas.svg)
- [docs/benchmarks/asm_vr_phase3a2/quality_vs_observed_throughput.png](../benchmarks/asm_vr_phase3a2/quality_vs_observed_throughput.png)
- [docs/benchmarks/asm_vr_phase3a2/quality_vs_observed_throughput.svg](../benchmarks/asm_vr_phase3a2/quality_vs_observed_throughput.svg)
- [docs/benchmarks/asm_vr_phase3a2/quality_vs_rank_frontiers.png](../benchmarks/asm_vr_phase3a2/quality_vs_rank_frontiers.png)
- [docs/benchmarks/asm_vr_phase3a2/quality_vs_rank_frontiers.svg](../benchmarks/asm_vr_phase3a2/quality_vs_rank_frontiers.svg)
- [docs/benchmarks/asm_vr_phase3a2/summary.json](../benchmarks/asm_vr_phase3a2/summary.json)
- [docs/benchmarks/asm_vr_phase3a2/test_ce_heatmap.png](../benchmarks/asm_vr_phase3a2/test_ce_heatmap.png)
- [docs/benchmarks/asm_vr_phase3a2/test_ce_heatmap.svg](../benchmarks/asm_vr_phase3a2/test_ce_heatmap.svg)
- [docs/benchmarks/asm_vr_phase3a2/validation_ce_by_base_rank.png](../benchmarks/asm_vr_phase3a2/validation_ce_by_base_rank.png)
- [docs/benchmarks/asm_vr_phase3a2/validation_ce_by_base_rank.svg](../benchmarks/asm_vr_phase3a2/validation_ce_by_base_rank.svg)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/README.md](../benchmarks/asm_vr_phase3a2_adamm_confirm/README.md)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/adamm_confirmation_cost.png](../benchmarks/asm_vr_phase3a2_adamm_confirm/adamm_confirmation_cost.png)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/adamm_confirmation_cost.svg](../benchmarks/asm_vr_phase3a2_adamm_confirm/adamm_confirmation_cost.svg)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/index.html](../benchmarks/asm_vr_phase3a2_adamm_confirm/index.html)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/manifest.json](../benchmarks/asm_vr_phase3a2_adamm_confirm/manifest.json)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/paired_s_minus_r_adamm.png](../benchmarks/asm_vr_phase3a2_adamm_confirm/paired_s_minus_r_adamm.png)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/paired_s_minus_r_adamm.svg](../benchmarks/asm_vr_phase3a2_adamm_confirm/paired_s_minus_r_adamm.svg)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/quality_optimizer_context.png](../benchmarks/asm_vr_phase3a2_adamm_confirm/quality_optimizer_context.png)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/quality_optimizer_context.svg](../benchmarks/asm_vr_phase3a2_adamm_confirm/quality_optimizer_context.svg)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/summary.json](../benchmarks/asm_vr_phase3a2_adamm_confirm/summary.json)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/validation_ce_adamm_new_seeds.png](../benchmarks/asm_vr_phase3a2_adamm_confirm/validation_ce_adamm_new_seeds.png)
- [docs/benchmarks/asm_vr_phase3a2_adamm_confirm/validation_ce_adamm_new_seeds.svg](../benchmarks/asm_vr_phase3a2_adamm_confirm/validation_ce_adamm_new_seeds.svg)
- [docs/benchmarks/asm_vr_phase3a3_rs/README.md](../benchmarks/asm_vr_phase3a3_rs/README.md)
- [docs/benchmarks/asm_vr_phase3a3_rs/full_base_observed_cost.png](../benchmarks/asm_vr_phase3a3_rs/full_base_observed_cost.png)
- [docs/benchmarks/asm_vr_phase3a3_rs/full_base_observed_cost.svg](../benchmarks/asm_vr_phase3a3_rs/full_base_observed_cost.svg)
- [docs/benchmarks/asm_vr_phase3a3_rs/full_base_test_ce.png](../benchmarks/asm_vr_phase3a3_rs/full_base_test_ce.png)
- [docs/benchmarks/asm_vr_phase3a3_rs/full_base_test_ce.svg](../benchmarks/asm_vr_phase3a3_rs/full_base_test_ce.svg)
- [docs/benchmarks/asm_vr_phase3a3_rs/index.html](../benchmarks/asm_vr_phase3a3_rs/index.html)
- [docs/benchmarks/asm_vr_phase3a3_rs/manifest.json](../benchmarks/asm_vr_phase3a3_rs/manifest.json)
- [docs/benchmarks/asm_vr_phase3a3_rs/quality_vs_parameters.png](../benchmarks/asm_vr_phase3a3_rs/quality_vs_parameters.png)
- [docs/benchmarks/asm_vr_phase3a3_rs/quality_vs_parameters.svg](../benchmarks/asm_vr_phase3a3_rs/quality_vs_parameters.svg)
- [docs/benchmarks/asm_vr_phase3a3_rs/summary.json](../benchmarks/asm_vr_phase3a3_rs/summary.json)
- [docs/benchmarks/asm_vr_phase3a3_rs/validation_ce_full_bases.png](../benchmarks/asm_vr_phase3a3_rs/validation_ce_full_bases.png)
- [docs/benchmarks/asm_vr_phase3a3_rs/validation_ce_full_bases.svg](../benchmarks/asm_vr_phase3a3_rs/validation_ce_full_bases.svg)
- [src/drm_language_emitter/directional_blocks.py](../../src/drm_language_emitter/directional_blocks.py)
- [src/drm_language_emitter/directional_forward.py](../../src/drm_language_emitter/directional_forward.py)
- [src/aletheion_state_models/variants/__init__.py](../../src/aletheion_state_models/variants/__init__.py)
- [src/aletheion_state_models/variants/variable_rank.py](../../src/aletheion_state_models/variants/variable_rank.py)
- [src/aletheion_state_models/variants/selective_state.py](../../src/aletheion_state_models/variants/selective_state.py)
- [src/aletheion_state_models/variants/relational_selective_state.py](../../src/aletheion_state_models/variants/relational_selective_state.py)
- [src/aletheion_state_models/benchmarks/phase3a_training.py](../../src/aletheion_state_models/benchmarks/phase3a_training.py)
- [src/aletheion_state_models/benchmarks/phase3a2_variants.py](../../src/aletheion_state_models/benchmarks/phase3a2_variants.py)
- [src/aletheion_state_models/benchmarks/phase3a2_calibration.py](../../src/aletheion_state_models/benchmarks/phase3a2_calibration.py)
- [src/aletheion_state_models/benchmarks/phase3a2_summary.py](../../src/aletheion_state_models/benchmarks/phase3a2_summary.py)
- [src/aletheion_state_models/benchmarks/phase3a2_plots.py](../../src/aletheion_state_models/benchmarks/phase3a2_plots.py)
- [src/aletheion_state_models/benchmarks/phase3a2_adamm_plots.py](../../src/aletheion_state_models/benchmarks/phase3a2_adamm_plots.py)
- [src/aletheion_state_models/benchmarks/phase3a3_variants.py](../../src/aletheion_state_models/benchmarks/phase3a3_variants.py)
- [src/aletheion_state_models/benchmarks/phase3a3_summary.py](../../src/aletheion_state_models/benchmarks/phase3a3_summary.py)
- [src/aletheion_state_models/benchmarks/phase3a3_plots.py](../../src/aletheion_state_models/benchmarks/phase3a3_plots.py)
- [scripts/run_asm_vr_phase3a2.py](../../scripts/run_asm_vr_phase3a2.py)
- [scripts/finalize_asm_vr_phase3a2.py](../../scripts/finalize_asm_vr_phase3a2.py)
- [scripts/run_asm_vr_phase3a3_rs.py](../../scripts/run_asm_vr_phase3a3_rs.py)
- [scripts/run_asm_vr_phase3a2_adamm_confirm.py](../../scripts/run_asm_vr_phase3a2_adamm_confirm.py)
- [tests/test_asm_vr_phase3a1.py](../../tests/test_asm_vr_phase3a1.py)
- [tests/test_asm_vr_phase3a2.py](../../tests/test_asm_vr_phase3a2.py)
- [runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_17/best.pt](../../runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_17/best.pt)
- [runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_17/result.json](../../runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_17/result.json)
- [runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_29/best.pt](../../runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_29/best.pt)
- [runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_29/result.json](../../runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_29/result.json)
- [runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_43/best.pt](../../runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_43/best.pt)
- [runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_43/result.json](../../runs/asm_vr_phase3a2/vr_r_adaptive_32/seed_43/result.json)
- [runs/asm_vr_phase3a2/vr_r_fixed_16/seed_17/best.pt](../../runs/asm_vr_phase3a2/vr_r_fixed_16/seed_17/best.pt)
- [runs/asm_vr_phase3a2/vr_r_fixed_16/seed_17/result.json](../../runs/asm_vr_phase3a2/vr_r_fixed_16/seed_17/result.json)
- [runs/asm_vr_phase3a2/vr_r_fixed_16/seed_29/best.pt](../../runs/asm_vr_phase3a2/vr_r_fixed_16/seed_29/best.pt)
- [runs/asm_vr_phase3a2/vr_r_fixed_16/seed_29/result.json](../../runs/asm_vr_phase3a2/vr_r_fixed_16/seed_29/result.json)
- [runs/asm_vr_phase3a2/vr_r_fixed_16/seed_43/best.pt](../../runs/asm_vr_phase3a2/vr_r_fixed_16/seed_43/best.pt)
- [runs/asm_vr_phase3a2/vr_r_fixed_16/seed_43/result.json](../../runs/asm_vr_phase3a2/vr_r_fixed_16/seed_43/result.json)
- [runs/asm_vr_phase3a2/vr_r_fixed_32/seed_17/best.pt](../../runs/asm_vr_phase3a2/vr_r_fixed_32/seed_17/best.pt)
- [runs/asm_vr_phase3a2/vr_r_fixed_32/seed_17/result.json](../../runs/asm_vr_phase3a2/vr_r_fixed_32/seed_17/result.json)
- [runs/asm_vr_phase3a2/vr_r_fixed_32/seed_29/best.pt](../../runs/asm_vr_phase3a2/vr_r_fixed_32/seed_29/best.pt)
- [runs/asm_vr_phase3a2/vr_r_fixed_32/seed_29/result.json](../../runs/asm_vr_phase3a2/vr_r_fixed_32/seed_29/result.json)
- [runs/asm_vr_phase3a2/vr_r_fixed_32/seed_43/best.pt](../../runs/asm_vr_phase3a2/vr_r_fixed_32/seed_43/best.pt)
- [runs/asm_vr_phase3a2/vr_r_fixed_32/seed_43/result.json](../../runs/asm_vr_phase3a2/vr_r_fixed_32/seed_43/result.json)
- [runs/asm_vr_phase3a2/vr_r_fixed_48/seed_17/best.pt](../../runs/asm_vr_phase3a2/vr_r_fixed_48/seed_17/best.pt)
- [runs/asm_vr_phase3a2/vr_r_fixed_48/seed_17/result.json](../../runs/asm_vr_phase3a2/vr_r_fixed_48/seed_17/result.json)
- [runs/asm_vr_phase3a2/vr_r_fixed_48/seed_29/best.pt](../../runs/asm_vr_phase3a2/vr_r_fixed_48/seed_29/best.pt)
- [runs/asm_vr_phase3a2/vr_r_fixed_48/seed_29/result.json](../../runs/asm_vr_phase3a2/vr_r_fixed_48/seed_29/result.json)
- [runs/asm_vr_phase3a2/vr_r_fixed_48/seed_43/best.pt](../../runs/asm_vr_phase3a2/vr_r_fixed_48/seed_43/best.pt)
- [runs/asm_vr_phase3a2/vr_r_fixed_48/seed_43/result.json](../../runs/asm_vr_phase3a2/vr_r_fixed_48/seed_43/result.json)
- [runs/asm_vr_phase3a2/vr_r_full/seed_17/best.pt](../../runs/asm_vr_phase3a2/vr_r_full/seed_17/best.pt)
- [runs/asm_vr_phase3a2/vr_r_full/seed_17/result.json](../../runs/asm_vr_phase3a2/vr_r_full/seed_17/result.json)
- [runs/asm_vr_phase3a2/vr_r_full/seed_29/best.pt](../../runs/asm_vr_phase3a2/vr_r_full/seed_29/best.pt)
- [runs/asm_vr_phase3a2/vr_r_full/seed_29/result.json](../../runs/asm_vr_phase3a2/vr_r_full/seed_29/result.json)
- [runs/asm_vr_phase3a2/vr_r_full/seed_43/best.pt](../../runs/asm_vr_phase3a2/vr_r_full/seed_43/best.pt)
- [runs/asm_vr_phase3a2/vr_r_full/seed_43/result.json](../../runs/asm_vr_phase3a2/vr_r_full/seed_43/result.json)
- [runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_17/best.pt](../../runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_17/best.pt)
- [runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_17/result.json](../../runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_17/result.json)
- [runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_29/best.pt](../../runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_29/best.pt)
- [runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_29/result.json](../../runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_29/result.json)
- [runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_43/best.pt](../../runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_43/best.pt)
- [runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_43/result.json](../../runs/asm_vr_phase3a2/vr_s_adaptive_32/seed_43/result.json)
- [runs/asm_vr_phase3a2/vr_s_fixed_16/seed_17/best.pt](../../runs/asm_vr_phase3a2/vr_s_fixed_16/seed_17/best.pt)
- [runs/asm_vr_phase3a2/vr_s_fixed_16/seed_17/result.json](../../runs/asm_vr_phase3a2/vr_s_fixed_16/seed_17/result.json)
- [runs/asm_vr_phase3a2/vr_s_fixed_16/seed_29/best.pt](../../runs/asm_vr_phase3a2/vr_s_fixed_16/seed_29/best.pt)
- [runs/asm_vr_phase3a2/vr_s_fixed_16/seed_29/result.json](../../runs/asm_vr_phase3a2/vr_s_fixed_16/seed_29/result.json)
- [runs/asm_vr_phase3a2/vr_s_fixed_16/seed_43/best.pt](../../runs/asm_vr_phase3a2/vr_s_fixed_16/seed_43/best.pt)
- [runs/asm_vr_phase3a2/vr_s_fixed_16/seed_43/result.json](../../runs/asm_vr_phase3a2/vr_s_fixed_16/seed_43/result.json)
- [runs/asm_vr_phase3a2/vr_s_fixed_32/seed_17/best.pt](../../runs/asm_vr_phase3a2/vr_s_fixed_32/seed_17/best.pt)
- [runs/asm_vr_phase3a2/vr_s_fixed_32/seed_17/result.json](../../runs/asm_vr_phase3a2/vr_s_fixed_32/seed_17/result.json)
- [runs/asm_vr_phase3a2/vr_s_fixed_32/seed_29/best.pt](../../runs/asm_vr_phase3a2/vr_s_fixed_32/seed_29/best.pt)
- [runs/asm_vr_phase3a2/vr_s_fixed_32/seed_29/result.json](../../runs/asm_vr_phase3a2/vr_s_fixed_32/seed_29/result.json)
- [runs/asm_vr_phase3a2/vr_s_fixed_32/seed_43/best.pt](../../runs/asm_vr_phase3a2/vr_s_fixed_32/seed_43/best.pt)
- [runs/asm_vr_phase3a2/vr_s_fixed_32/seed_43/result.json](../../runs/asm_vr_phase3a2/vr_s_fixed_32/seed_43/result.json)
- [runs/asm_vr_phase3a2/vr_s_fixed_48/seed_17/best.pt](../../runs/asm_vr_phase3a2/vr_s_fixed_48/seed_17/best.pt)
- [runs/asm_vr_phase3a2/vr_s_fixed_48/seed_17/result.json](../../runs/asm_vr_phase3a2/vr_s_fixed_48/seed_17/result.json)
- [runs/asm_vr_phase3a2/vr_s_fixed_48/seed_29/best.pt](../../runs/asm_vr_phase3a2/vr_s_fixed_48/seed_29/best.pt)
- [runs/asm_vr_phase3a2/vr_s_fixed_48/seed_29/result.json](../../runs/asm_vr_phase3a2/vr_s_fixed_48/seed_29/result.json)
- [runs/asm_vr_phase3a2/vr_s_fixed_48/seed_43/best.pt](../../runs/asm_vr_phase3a2/vr_s_fixed_48/seed_43/best.pt)
- [runs/asm_vr_phase3a2/vr_s_fixed_48/seed_43/result.json](../../runs/asm_vr_phase3a2/vr_s_fixed_48/seed_43/result.json)
- [runs/asm_vr_phase3a2/vr_s_full/seed_17/best.pt](../../runs/asm_vr_phase3a2/vr_s_full/seed_17/best.pt)
- [runs/asm_vr_phase3a2/vr_s_full/seed_17/result.json](../../runs/asm_vr_phase3a2/vr_s_full/seed_17/result.json)
- [runs/asm_vr_phase3a2/vr_s_full/seed_29/best.pt](../../runs/asm_vr_phase3a2/vr_s_full/seed_29/best.pt)
- [runs/asm_vr_phase3a2/vr_s_full/seed_29/result.json](../../runs/asm_vr_phase3a2/vr_s_full/seed_29/result.json)
- [runs/asm_vr_phase3a2/vr_s_full/seed_43/best.pt](../../runs/asm_vr_phase3a2/vr_s_full/seed_43/best.pt)
- [runs/asm_vr_phase3a2/vr_s_full/seed_43/result.json](../../runs/asm_vr_phase3a2/vr_s_full/seed_43/result.json)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_107/best.pt](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_107/best.pt)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_107/result.json](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_107/result.json)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_71/best.pt](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_71/best.pt)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_71/result.json](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_71/result.json)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_89/best.pt](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_89/best.pt)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_89/result.json](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_r_full/seed_89/result.json)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_107/best.pt](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_107/best.pt)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_107/result.json](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_107/result.json)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_71/best.pt](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_71/best.pt)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_71/result.json](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_71/result.json)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_89/best.pt](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_89/best.pt)
- [runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_89/result.json](../../runs/asm_vr_phase3a2_adamm_confirm/adamm_vr_s_full/seed_89/result.json)
- [runs/asm_vr_phase3a3_rs/vr_rs_full/seed_17/best.pt](../../runs/asm_vr_phase3a3_rs/vr_rs_full/seed_17/best.pt)
- [runs/asm_vr_phase3a3_rs/vr_rs_full/seed_17/result.json](../../runs/asm_vr_phase3a3_rs/vr_rs_full/seed_17/result.json)
- [runs/asm_vr_phase3a3_rs/vr_rs_full/seed_29/best.pt](../../runs/asm_vr_phase3a3_rs/vr_rs_full/seed_29/best.pt)
- [runs/asm_vr_phase3a3_rs/vr_rs_full/seed_29/result.json](../../runs/asm_vr_phase3a3_rs/vr_rs_full/seed_29/result.json)
- [runs/asm_vr_phase3a3_rs/vr_rs_full/seed_43/best.pt](../../runs/asm_vr_phase3a3_rs/vr_rs_full/seed_43/best.pt)
- [runs/asm_vr_phase3a3_rs/vr_rs_full/seed_43/result.json](../../runs/asm_vr_phase3a3_rs/vr_rs_full/seed_43/result.json)
- [docs/report/0016_fase-3a-2-asm-vr-s-e-asm-vr-rs_2026-09-01.md](0016_fase-3a-2-asm-vr-s-e-asm-vr-rs_2026-09-01.md)

## Changes

- Adicionada projeção de local_delta antes do mixer, fechando o último caminho lógico fora do rank ativo.
- Adicionado bloco aberto causal de forma fixa no forward/prefill/decode, reduzindo streaming máximo para <=2.86e-6.
- Criada configuração canônica ASM-S e builder variable-rank seletivo sem métrica/naturalização.
- Pareado ASM-VR-R 223814 parâmetros com ASM-VR-S 223738 parâmetros usando selective hidden 308.
- Executada matriz AdamW 2 bases x5 políticas x3 seeds, com test selado até decisões validation-only.
- Promovido ASM-VR-S pelo caminho de ganho consistente >=0.02 nat, não pelo gate de throughput.
- Confirmado o efeito S-R sob AdamM em seeds novas 71/89/107.
- Formalizados ASM-RS e ASM-VR-RS e executado vr_rs_full em três seeds.
- Gerados 15 gráficos PNG/SVG e três dashboards HTML com relatórios separados.
- Atualizadas arquitetura e taxonomia bilingue da família ASM.

## Validation

- Phase 3A.2 AdamW — 30/30 runs completas e finitas; S venceu R em 15/15 comparações pareadas.
- ASM-VR-S full — test CE 2.5318 versus R full 2.5803; delta -0.0485 nat.
- ASM-VR-S fixed-32 — test CE 2.5605 versus R fixed-32 2.6108; delta -0.0503 nat.
- Confirmação AdamM — S venceu R em 3/3 seeds novas; delta médio -0.0451 nat.
- ASM-VR-RS full — test CE 2.5721; S full continuou vencedor com 8.5% menos parâmetros.
- Controller adaptive R/S — orçamento, variação e gradiente passaram; near-fixed32, Pareto e fronteira falharam.
- Streaming FP32 — máximos 2.86e-6 (3A.2), 1.91e-6 (RS) e 2.38e-6 (AdamM), todos <=1e-4.
- 15 PNGs validados, 15 SVGs parseados, links dos três dashboards verificados e gráficos principais inspecionados visualmente.
- .venv/bin/python -m pytest -q — 205 testes passaram; dois warnings preexistentes.
- compileall e git diff --check — passaram.
- solid_source_modularity — novos módulos conformes; quatro exceções coesas DRM 301-500 e quatro violações >500 preexistentes fora do escopo.
