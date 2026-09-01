# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Executar o mesmo piloto ATTR P1 com ASM-CM, ASM-VR e ASM-R e incluir os resultados no gráfico.

## Summary

Executados quatro braços suplementares sobre os mesmos dados train/validation seed 17 e 1.000 updates: ASM-CM, ASM-VR-S full-64, ASM-VR-S fixed-32 e controle ASM-R 240K. O dashboard agora compara seis braços e separa parâmetros totais/treináveis e papéis registrados/suplementares. Test permaneceu selado. ASM-CM liderou AUPRC, ASM-R teve menor Brier e Transformer maior recall; não houve vencedor único nem alteração de gates.

## Modified files

- [src/aletheion_state_models/benchmarks/transition_risk/model_adapters.py](../../src/aletheion_state_models/benchmarks/transition_risk/model_adapters.py)
- [src/aletheion_state_models/benchmarks/transition_risk/training.py](../../src/aletheion_state_models/benchmarks/transition_risk/training.py)
- [src/aletheion_state_models/benchmarks/transition_risk/render.py](../../src/aletheion_state_models/benchmarks/transition_risk/render.py)
- [src/aletheion_state_models/benchmarks/transition_risk/pilot.py](../../src/aletheion_state_models/benchmarks/transition_risk/pilot.py)
- [src/aletheion_state_models/benchmarks/transition_risk/supplementary.py](../../src/aletheion_state_models/benchmarks/transition_risk/supplementary.py)
- [scripts/run_attr_p1_supplementary.py](../../scripts/run_attr_p1_supplementary.py)
- [tests/test_transition_risk_model_interface.py](../../tests/test_transition_risk_model_interface.py)
- [tests/test_transition_risk_supplementary.py](../../tests/test_transition_risk_supplementary.py)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/README.md](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/index.html](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/index.html)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/manifest.json](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/manifest.json)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/parameter_match.png](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/parameter_match.png)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/parameter_match.svg](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/parameter_match.svg)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/summary.json](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/supplementary_results.json](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/supplementary_results.json)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/validation_metrics.png](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/validation_metrics.png)
- [docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/validation_metrics.svg](../benchmarks/asm_transformer_transition_risk/pilot_seed_17/validation_metrics.svg)
- [docs/report/0027_bracos-suplementares-p1-attr_2026-09-01.md](0027_bracos-suplementares-p1-attr_2026-09-01.md)

## Changes

- Adicionado runner retomável para quatro braços suplementares no dataset P1 congelado.
- Estendido ASMModelAdapter para propagar global_step exigido pelo VR-S durante treino.
- Corrigida contagem de parâmetros totais versus realmente treináveis, incluindo controller VR congelado.
- Incluídos controles VR-S full-64 e fixed-32 para interpretar rank sem atribuição indevida.
- Atualizados gráficos para três painéis legíveis de AUPRC, Brier e recall, mais gráfico total/trainable.
- Atualizados manifesto, protocolos EN/PT-BR, README e dashboard com seis braços e papéis de comparação explícitos.
- Adicionados testes de parameter matching CM/VR, global_step e zero payload inativo fixed-32.

## Validation

- Processo 1458891 — exit code 0 em 116,3 s
- Quatro braços — 1.000 updates cada; mesmo seed/dados/horizontes/objetivo; test_worlds_generated=false
- .venv/bin/python -m pytest -q — 249 passed; 5 warnings conhecidos
- .venv/bin/python -m compileall -q src scripts tests — passou
- git diff --check — passou
- Validação de PNG/SVG/HTML/JSON/manifest, links e hash de implementação — passou
- Auditoria SOLID Python — 273 compliant, 8 exceções e 4 violações preexistentes; nenhum arquivo desta solicitação excede 300 linhas
