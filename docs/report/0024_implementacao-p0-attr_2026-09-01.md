# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Codar o ATTR em paralelo enquanto o currículo longo ASM-CM-VR continua treinando, sem disputar a GPU nem unsealar test.

## Summary

Implementado o P0 modular do ATTR: HazardWorld causal, dataset pareado, labels multi-horizonte, heads/adapters comuns, hidden states do Transformer, métricas, leakage audits, shield, intervenções clonadas, controles persistence/Markov/Kalman, treino train-only, runners, gráficos e dashboard offline. O smoke CPU de uma atualização passou como evidência apenas de integração; P1 foi codado mas não executado e test permaneceu selado.

## Modified files

- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/attr_p0_smoke/README.md](../benchmarks/attr_p0_smoke/README.md)
- [docs/benchmarks/attr_p0_smoke/index.html](../benchmarks/attr_p0_smoke/index.html)
- [docs/benchmarks/attr_p0_smoke/parameter_match.png](../benchmarks/attr_p0_smoke/parameter_match.png)
- [docs/benchmarks/attr_p0_smoke/parameter_match.svg](../benchmarks/attr_p0_smoke/parameter_match.svg)
- [docs/benchmarks/attr_p0_smoke/summary.json](../benchmarks/attr_p0_smoke/summary.json)
- [docs/benchmarks/attr_p0_smoke/validation_metrics.png](../benchmarks/attr_p0_smoke/validation_metrics.png)
- [docs/benchmarks/attr_p0_smoke/validation_metrics.svg](../benchmarks/attr_p0_smoke/validation_metrics.svg)
- [scripts/run_attr_p0_smoke.py](../../scripts/run_attr_p0_smoke.py)
- [scripts/run_asm_transformer_transition_risk.py](../../scripts/run_asm_transformer_transition_risk.py)
- [scripts/render_transition_risk_dashboard.py](../../scripts/render_transition_risk_dashboard.py)
- [src/aletheion_state_models/benchmarks/transition_risk/__init__.py](../../src/aletheion_state_models/benchmarks/transition_risk/__init__.py)
- [src/aletheion_state_models/benchmarks/transition_risk/types.py](../../src/aletheion_state_models/benchmarks/transition_risk/types.py)
- [src/aletheion_state_models/benchmarks/transition_risk/labels.py](../../src/aletheion_state_models/benchmarks/transition_risk/labels.py)
- [src/aletheion_state_models/benchmarks/transition_risk/metrics.py](../../src/aletheion_state_models/benchmarks/transition_risk/metrics.py)
- [src/aletheion_state_models/benchmarks/transition_risk/leakage.py](../../src/aletheion_state_models/benchmarks/transition_risk/leakage.py)
- [src/aletheion_state_models/benchmarks/transition_risk/shield.py](../../src/aletheion_state_models/benchmarks/transition_risk/shield.py)
- [src/aletheion_state_models/benchmarks/transition_risk/intervention.py](../../src/aletheion_state_models/benchmarks/transition_risk/intervention.py)
- [src/aletheion_state_models/benchmarks/transition_risk/model_adapters.py](../../src/aletheion_state_models/benchmarks/transition_risk/model_adapters.py)
- [src/aletheion_state_models/benchmarks/transition_risk/model_heads.py](../../src/aletheion_state_models/benchmarks/transition_risk/model_heads.py)
- [src/aletheion_state_models/benchmarks/transition_risk/dataset.py](../../src/aletheion_state_models/benchmarks/transition_risk/dataset.py)
- [src/aletheion_state_models/benchmarks/transition_risk/training.py](../../src/aletheion_state_models/benchmarks/transition_risk/training.py)
- [src/aletheion_state_models/benchmarks/transition_risk/baselines.py](../../src/aletheion_state_models/benchmarks/transition_risk/baselines.py)
- [src/aletheion_state_models/benchmarks/transition_risk/runner.py](../../src/aletheion_state_models/benchmarks/transition_risk/runner.py)
- [src/aletheion_state_models/benchmarks/transition_risk/render.py](../../src/aletheion_state_models/benchmarks/transition_risk/render.py)
- [transformer/tiny_transformer.py](../../transformer/tiny_transformer.py)
- [world_model/hazard_world.py](../../world_model/hazard_world.py)
- [world_model/hazard_world_types.py](../../world_model/hazard_world_types.py)
- [world_model/hazard_world_io.py](../../world_model/hazard_world_io.py)
- [tests/test_hazard_world.py](../../tests/test_hazard_world.py)
- [tests/test_transition_risk_baselines.py](../../tests/test_transition_risk_baselines.py)
- [tests/test_transition_risk_dataset.py](../../tests/test_transition_risk_dataset.py)
- [tests/test_transition_risk_intervention.py](../../tests/test_transition_risk_intervention.py)
- [tests/test_transition_risk_labels_metrics.py](../../tests/test_transition_risk_labels_metrics.py)
- [tests/test_transition_risk_leakage.py](../../tests/test_transition_risk_leakage.py)
- [tests/test_transition_risk_model_interface.py](../../tests/test_transition_risk_model_interface.py)
- [docs/report/0024_implementacao-p0-attr_2026-09-01.md](0024_implementacao-p0-attr_2026-09-01.md)

## Changes

- Implementado HazardWorld reproduzível com observação parcial, safe set, falhas atrasadas e common-random cloned interventions.
- Adicionadas interfaces pareadas ASM/Transformer, heads de próximo estado/hazard/severidade e API causal opt-in de hidden states.
- Adicionados dados em frames fixos, labels futuras H=1/4/8/16, auditorias fail-closed, métricas, hard shield e controles persistence/Markov/Kalman.
- Adicionados treino e runner P0/P1 train-only, sem gerar mundos de test; criado smoke CPU com mismatch total de parâmetros de 0,600%.
- Gerados PNG/SVG, dashboard HTML e JSON; documentos bilíngues agora registram P0 concluído e P1 como próximo passo.

## Validation

- .venv/bin/python -m pytest -q — 242 passed; 5 warnings conhecidos
- scripts/run_asm_transformer_transition_risk.py --phase p0-smoke --updates 1 --device cpu — concluído; feature/split audits passaram e test_worlds_generated=false
- .venv/bin/python -m compileall -q ... — passou
- git diff --check — passou
- Validação PNG/SVG/HTML/JSON — passou; links locais existem e imagens são legíveis
- Auditoria SOLID Python — 267 compliant, 8 exceções e 4 violações preexistentes; nenhum módulo ATTR excede 300 linhas
