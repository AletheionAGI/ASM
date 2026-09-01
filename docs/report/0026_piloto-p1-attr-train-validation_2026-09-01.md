# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Executar o ATTR P1 e fornecer uma estimativa de tempo, mantendo test selado.

## Summary

P1 seed 17 concluído na GPU em 304,3 s com 1.000 updates pareados por braço, 64 mundos train e 16 validation. Test não foi gerado. Transformer liderou o snapshot validation de uma seed em AUPRC, Brier e recall no orçamento FPR; thresholds validation-only e margens G2-G4 foram congelados sem revisão. Dashboard, prevalência e manifesto reproduzível foram publicados.

## Modified files

- [src/aletheion_state_models/benchmarks/transition_risk/training.py](../../src/aletheion_state_models/benchmarks/transition_risk/training.py)
- [src/aletheion_state_models/benchmarks/transition_risk/runner.py](../../src/aletheion_state_models/benchmarks/transition_risk/runner.py)
- [src/aletheion_state_models/benchmarks/transition_risk/render.py](../../src/aletheion_state_models/benchmarks/transition_risk/render.py)
- [src/aletheion_state_models/benchmarks/transition_risk/pilot.py](../../src/aletheion_state_models/benchmarks/transition_risk/pilot.py)
- [tests/test_transition_risk_pilot.py](../../tests/test_transition_risk_pilot.py)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/README.md](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/index.html](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/index.html)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/manifest.json](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/manifest.json)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/parameter_match.png](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/parameter_match.png)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/parameter_match.svg](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/parameter_match.svg)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/summary.json](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/validation_metrics.png](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/validation_metrics.png)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/validation_metrics.svg](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/validation_metrics.svg)
- [docs/report/0026_piloto-p1-attr-train-validation_2026-09-01.md](0026_piloto-p1-attr-train-validation_2026-09-01.md)

## Changes

- Adicionada telemetria de update e estimativa durante treino ATTR.
- Executado P1 train/validation seed 17 para ASM-X directional e Tiny Transformer 220K.
- Calculadas prevalências por horizonte e congelados thresholds validation-only em manifesto com hash da implementação.
- Corrigidos dashboard e README para distinguir P1 do smoke P0.
- Atualizados protocolos EN/PT-BR e índice de benchmarks com resultados limitados do piloto.
- Adicionados testes de prevalência e fail-closed quando test é gerado.

## Validation

- Processo 1409154 — exit code 0 em 304,3 s
- P1 — 1.000 updates por braço; feature leakage e episode-split audits passaram; test_worlds_generated=false
- .venv/bin/python -m pytest -q — 246 passed; 5 warnings conhecidos
- .venv/bin/python -m compileall -q src scripts tests — passou
- git diff --check — passou
- Validação de PNG/SVG/HTML/JSON/manifest e links offline — passou
- Auditoria SOLID Python — 270 compliant, 8 exceções e 4 violações preexistentes; nenhum arquivo P1 excede 300 linhas
