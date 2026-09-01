# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Implementar a Fase 3A do ASM-VR em linguagem, com gráficos comparativos das medições para facilitar análise humana; instalar matplotlib e fornecer comandos para execução e acompanhamento no terminal.

## Summary

A Fase 3A de pequena escala foi implementada e executada: split documental 90/5/5, seis variantes, três seeds e ~2M tokens/run. Todos os nove gates operacionais passaram após calibração hard em validação. Foram gerados seis gráficos em PNG/SVG e dashboard HTML. O adaptativo ficou próximo ao fixed-32, mas foi dominado pelo fixed-16 e não demonstrou vantagem Pareto nem speedup.

## Modified files

- [pyproject.toml](../../pyproject.toml)
- [docs/ARCHITECTURE_ASM_VR.md](../ARCHITECTURE_ASM_VR.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_vr_phase3a/README.md](../benchmarks/asm_vr_phase3a/README.md)
- [docs/benchmarks/asm_vr_phase3a/development_budget_0_00125.json](../benchmarks/asm_vr_phase3a/development_budget_0_00125.json)
- [docs/benchmarks/asm_vr_phase3a/development_budget_0_0025_threshold_0_5.json](../benchmarks/asm_vr_phase3a/development_budget_0_0025_threshold_0_5.json)
- [docs/benchmarks/asm_vr_phase3a/development_threshold_0_7_training.json](../benchmarks/asm_vr_phase3a/development_threshold_0_7_training.json)
- [docs/benchmarks/asm_vr_phase3a/final_test_ce_by_variant.png](../benchmarks/asm_vr_phase3a/final_test_ce_by_variant.png)
- [docs/benchmarks/asm_vr_phase3a/final_test_ce_by_variant.svg](../benchmarks/asm_vr_phase3a/final_test_ce_by_variant.svg)
- [docs/benchmarks/asm_vr_phase3a/index.html](../benchmarks/asm_vr_phase3a/index.html)
- [docs/benchmarks/asm_vr_phase3a/manifest.json](../benchmarks/asm_vr_phase3a/manifest.json)
- [docs/benchmarks/asm_vr_phase3a/observed_cost.png](../benchmarks/asm_vr_phase3a/observed_cost.png)
- [docs/benchmarks/asm_vr_phase3a/observed_cost.svg](../benchmarks/asm_vr_phase3a/observed_cost.svg)
- [docs/benchmarks/asm_vr_phase3a/paired_seed_deltas.png](../benchmarks/asm_vr_phase3a/paired_seed_deltas.png)
- [docs/benchmarks/asm_vr_phase3a/paired_seed_deltas.svg](../benchmarks/asm_vr_phase3a/paired_seed_deltas.svg)
- [docs/benchmarks/asm_vr_phase3a/quality_vs_mean_rank.png](../benchmarks/asm_vr_phase3a/quality_vs_mean_rank.png)
- [docs/benchmarks/asm_vr_phase3a/quality_vs_mean_rank.svg](../benchmarks/asm_vr_phase3a/quality_vs_mean_rank.svg)
- [docs/benchmarks/asm_vr_phase3a/rank_distribution.png](../benchmarks/asm_vr_phase3a/rank_distribution.png)
- [docs/benchmarks/asm_vr_phase3a/rank_distribution.svg](../benchmarks/asm_vr_phase3a/rank_distribution.svg)
- [docs/benchmarks/asm_vr_phase3a/summary.json](../benchmarks/asm_vr_phase3a/summary.json)
- [docs/benchmarks/asm_vr_phase3a/validation_ce_by_tokens.png](../benchmarks/asm_vr_phase3a/validation_ce_by_tokens.png)
- [docs/benchmarks/asm_vr_phase3a/validation_ce_by_tokens.svg](../benchmarks/asm_vr_phase3a/validation_ce_by_tokens.svg)
- [scripts/run_asm_vr_phase3a.py](../../scripts/run_asm_vr_phase3a.py)
- [scripts/calibrate_asm_vr_phase3a.py](../../scripts/calibrate_asm_vr_phase3a.py)
- [src/aletheion_state_models/benchmarks/__init__.py](../../src/aletheion_state_models/benchmarks/__init__.py)
- [src/aletheion_state_models/benchmarks/phase3a_checkpoint.py](../../src/aletheion_state_models/benchmarks/phase3a_checkpoint.py)
- [src/aletheion_state_models/benchmarks/phase3a_data.py](../../src/aletheion_state_models/benchmarks/phase3a_data.py)
- [src/aletheion_state_models/benchmarks/phase3a_variants.py](../../src/aletheion_state_models/benchmarks/phase3a_variants.py)
- [src/aletheion_state_models/benchmarks/phase3a_training.py](../../src/aletheion_state_models/benchmarks/phase3a_training.py)
- [src/aletheion_state_models/benchmarks/phase3a_summary.py](../../src/aletheion_state_models/benchmarks/phase3a_summary.py)
- [src/aletheion_state_models/benchmarks/phase3a_plots.py](../../src/aletheion_state_models/benchmarks/phase3a_plots.py)
- [tests/test_asm_vr_phase3a.py](../../tests/test_asm_vr_phase3a.py)
- [docs/report/0009_fase-3a-asm-vr-com-graficos_2026-09-01.md](0009_fase-3a-asm-vr-com-graficos_2026-09-01.md)

## Changes

- Instalado matplotlib 3.11.1 no ambiente ASM e adicionado extra opcional viz.
- Criados split SHA-256 por documento, fábrica de seis variantes, treino/eval/checkpoint, agregação e gates Phase 3A.
- Executadas 18 runs pareadas com seeds 17/29/43 e aproximadamente 2.003M tokens por run.
- Calibrado threshold hard global 0.8 somente na validação; streaming final medido em FP32.
- Gerados gráficos CE×tokens, CE final, qualidade×rank, distribuição de rank, custo observado e deltas pareados em PNG e SVG.
- Gerado dashboard HTML offline e documentação com interpretação negativa/Pareto e disclosures das matrizes de desenvolvimento.
- Fornecidos comandos de execução, monitoramento, inspeção de resultados e abertura do dashboard.

## Validation

- .venv/bin/python -m pytest -q — 192 testes passaram; dois warnings preexistentes.
- Matriz Phase 3A — 18/18 runs completas; todos os 9 gates finais passaram.
- Resultado adaptativo — test CE 3.1869, rank médio 30.63; delta pareado médio +0.0116 nat contra fixed-32.
- Streaming FP32 adaptativo — erro máximo entre 1.9e-06 e 2.4e-06.
- Seis PNGs validados por assinatura e seis SVGs parseados por ElementTree; links do dashboard válidos.
- .venv/bin/python -m compileall -q — passou; git diff --check — passou.
- solid_source_modularity — todos os novos módulos conformes; oito exceções coesas 301–500 e quatro violações >500 preexistentes fora do escopo.
