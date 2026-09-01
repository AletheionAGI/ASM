# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Implementar a Fase 3A.1 antes da Fase 3B, reintegrando componentes úteis do ASM-R sob projeção hard, comparando full/fixed/adaptativo no mesmo scaffold e gerando relatórios e gráficos separados para 3A.1-A e 3A.1-B.

## Summary

A Fase 3A.1 foi implementada e executada em 39 runs de ~2.003M tokens. A etapa A avaliou o fatorial 2^3 e selecionou mixer+residual, recuperando 0.5853 nat sobre o scaffold estrito e ficando a 0.0082 nat do ASM-R histórico com custo observado menor. A etapa B calibrada atingiu rank 32.21, mas o adaptativo ficou 0.0676 nat pior que fixed-32 e falhou Pareto/fronteira fixa. Foram produzidos relatórios, PNG/SVG e dashboards independentes para A e B.

## Modified files

- [docs/ARCHITECTURE_ASM_VR.md](../ARCHITECTURE_ASM_VR.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_vr_phase3a1/README.md](../benchmarks/asm_vr_phase3a1/README.md)
- [docs/benchmarks/asm_vr_phase3a1/manifest.json](../benchmarks/asm_vr_phase3a1/manifest.json)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/README.md](../benchmarks/asm_vr_phase3a1/stage_a/README.md)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/factorial_effects.png](../benchmarks/asm_vr_phase3a1/stage_a/factorial_effects.png)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/factorial_effects.svg](../benchmarks/asm_vr_phase3a1/stage_a/factorial_effects.svg)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/index.html](../benchmarks/asm_vr_phase3a1/stage_a/index.html)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/observed_dense_cost.png](../benchmarks/asm_vr_phase3a1/stage_a/observed_dense_cost.png)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/observed_dense_cost.svg](../benchmarks/asm_vr_phase3a1/stage_a/observed_dense_cost.svg)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/scaffold_validation_ce.png](../benchmarks/asm_vr_phase3a1/stage_a/scaffold_validation_ce.png)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/scaffold_validation_ce.svg](../benchmarks/asm_vr_phase3a1/stage_a/scaffold_validation_ce.svg)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/summary.json](../benchmarks/asm_vr_phase3a1/stage_a/summary.json)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/validation_ce_by_tokens.png](../benchmarks/asm_vr_phase3a1/stage_a/validation_ce_by_tokens.png)
- [docs/benchmarks/asm_vr_phase3a1/stage_a/validation_ce_by_tokens.svg](../benchmarks/asm_vr_phase3a1/stage_a/validation_ce_by_tokens.svg)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/README.md](../benchmarks/asm_vr_phase3a1/stage_b/README.md)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/adaptive_rank_range.png](../benchmarks/asm_vr_phase3a1/stage_b/adaptive_rank_range.png)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/adaptive_rank_range.svg](../benchmarks/asm_vr_phase3a1/stage_b/adaptive_rank_range.svg)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/index.html](../benchmarks/asm_vr_phase3a1/stage_b/index.html)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/observed_dense_cost.png](../benchmarks/asm_vr_phase3a1/stage_b/observed_dense_cost.png)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/observed_dense_cost.svg](../benchmarks/asm_vr_phase3a1/stage_b/observed_dense_cost.svg)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/paired_adaptive_deltas.png](../benchmarks/asm_vr_phase3a1/stage_b/paired_adaptive_deltas.png)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/paired_adaptive_deltas.svg](../benchmarks/asm_vr_phase3a1/stage_b/paired_adaptive_deltas.svg)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/quality_vs_mean_rank.png](../benchmarks/asm_vr_phase3a1/stage_b/quality_vs_mean_rank.png)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/quality_vs_mean_rank.svg](../benchmarks/asm_vr_phase3a1/stage_b/quality_vs_mean_rank.svg)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/summary.json](../benchmarks/asm_vr_phase3a1/stage_b/summary.json)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/validation_ce_by_tokens.png](../benchmarks/asm_vr_phase3a1/stage_b/validation_ce_by_tokens.png)
- [docs/benchmarks/asm_vr_phase3a1/stage_b/validation_ce_by_tokens.svg](../benchmarks/asm_vr_phase3a1/stage_b/validation_ce_by_tokens.svg)
- [scripts/run_asm_vr_phase3a1.py](../../scripts/run_asm_vr_phase3a1.py)
- [scripts/finalize_asm_vr_phase3a1.py](../../scripts/finalize_asm_vr_phase3a1.py)
- [src/drm_language_emitter/config.py](../../src/drm_language_emitter/config.py)
- [src/drm_language_emitter/model.py](../../src/drm_language_emitter/model.py)
- [src/drm_language_emitter/directional_forward.py](../../src/drm_language_emitter/directional_forward.py)
- [src/drm_language_emitter/directional_blocks.py](../../src/drm_language_emitter/directional_blocks.py)
- [src/drm_language_emitter/inference.py](../../src/drm_language_emitter/inference.py)
- [src/aletheion_state_models/variants/__init__.py](../../src/aletheion_state_models/variants/__init__.py)
- [src/aletheion_state_models/variants/variable_rank.py](../../src/aletheion_state_models/variants/variable_rank.py)
- [src/aletheion_state_models/benchmarks/phase3a_training.py](../../src/aletheion_state_models/benchmarks/phase3a_training.py)
- [src/aletheion_state_models/benchmarks/phase3a1_variants.py](../../src/aletheion_state_models/benchmarks/phase3a1_variants.py)
- [src/aletheion_state_models/benchmarks/phase3a1_summary.py](../../src/aletheion_state_models/benchmarks/phase3a1_summary.py)
- [src/aletheion_state_models/benchmarks/phase3a1_plots.py](../../src/aletheion_state_models/benchmarks/phase3a1_plots.py)
- [tests/test_asm_vr_phase3a1.py](../../tests/test_asm_vr_phase3a1.py)
- [runs/asm_vr_phase3a1/stage_a/all_projected/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_a/all_projected/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_a/all_projected/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_a/all_projected/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_a/all_projected/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_a/all_projected/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_a/all_projected/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_a/all_projected/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_a/all_projected/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_a/all_projected/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_a/all_projected/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_a/all_projected/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_a/mixer/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_a/mixer/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_a/mixer/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_a/mixer/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_a/mixer/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_a/mixer/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_a/mixer/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_a/mixer/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_a/mixer/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_a/mixer/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_a/mixer/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_a/mixer/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_a/mixer_residual/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_a/mixer_selective/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_a/residual/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_a/residual/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_a/residual/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_a/residual/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_a/residual/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_a/residual/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_a/residual/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_a/residual/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_a/residual/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_a/residual/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_a/residual/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_a/residual/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_a/residual_selective/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_a/residual_selective/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_a/residual_selective/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_a/residual_selective/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_a/residual_selective/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_a/residual_selective/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_a/residual_selective/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_a/residual_selective/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_a/residual_selective/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_a/residual_selective/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_a/residual_selective/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_a/residual_selective/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_a/selective/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_a/selective/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_a/selective/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_a/selective/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_a/selective/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_a/selective/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_a/selective/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_a/selective/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_a/selective/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_a/selective/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_a/selective/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_a/selective/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_a/strict/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_a/strict/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_a/strict/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_a/strict/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_a/strict/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_a/strict/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_a/strict/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_a/strict/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_a/strict/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_a/strict/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_a/strict/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_a/strict/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_adaptive_32/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_16/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_32/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_fixed_48/seed_43/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_full/seed_17/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_full/seed_17/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_full/seed_17/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_full/seed_17/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_full/seed_29/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_full/seed_29/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_full/seed_29/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_full/seed_29/result.json)
- [runs/asm_vr_phase3a1/stage_b/selected_full/seed_43/best.pt](../../runs/asm_vr_phase3a1/stage_b/selected_full/seed_43/best.pt)
- [runs/asm_vr_phase3a1/stage_b/selected_full/seed_43/result.json](../../runs/asm_vr_phase3a1/stage_b/selected_full/seed_43/result.json)
- [docs/report/0012_fase-3a-1-scaffold-projetado_2026-09-01.md](0012_fase-3a-1-scaffold-projetado_2026-09-01.md)

## Changes

- Adicionado contrato explícito de projection mask sem estado global mutável e projeção após transição, mixer, residual e memória seletiva.
- Adicionado modo phase3a1_projected e builder opt-in, mantendo Phase 1/2 e legado inalterados.
- Adicionado padding causal de forma estável no decode para reduzir erro streaming projetado de ~6e-4 para no máximo 2.38e-5.
- Executado fatorial full-rank de oito scaffolds por três seeds e selecionado mixer+residual somente por validation.
- Executada matriz full/fixed16/fixed32/fixed48/adaptive por três seeds no scaffold congelado.
- Calibrado threshold 0.672 somente nas distribuições de score de validation.
- Gerados quatro gráficos PNG/SVG e dashboard para 3A.1-A e cinco gráficos PNG/SVG e dashboard para 3A.1-B.
- Documentados separadamente recuperação de qualidade e falha científica do controller adaptativo.

## Validation

- .venv/bin/python -m pytest -q — 200 testes passaram; dois warnings preexistentes.
- Testes anti-bypass — sentinelas entre componentes, Jacobiano no complemento, full-rank parity e cache streaming passaram.
- 3A.1-A — 24/24 runs finitas; matriz, streaming e recuperação de qualidade passaram.
- 3A.1-B — 15/15 runs finitas; orçamento, variação, gradiente e streaming passaram; qualidade próxima, Pareto e vantagem sobre fronteira falharam.
- Streaming FP32 — máximo 2.38e-5 na etapa A e 9.54e-6 na etapa B, ambos <=1e-4.
- Nove PNGs validados por assinatura, nove SVGs parseados e todos os links dos dashboards verificados.
- Inspeção visual de scaffold CE, efeitos fatoriais, qualidade-rank e faixa adaptativa — legíveis e consistentes com summaries.
- compileall e git diff --check — passaram.
- solid_source_modularity — novos módulos conformes; config/directional modules permanecem exceções coesas 301–500 e quatro violações >500 preexistentes fora do escopo.
