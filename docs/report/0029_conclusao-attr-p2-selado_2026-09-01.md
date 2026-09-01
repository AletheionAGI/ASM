# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Integrar as estatísticas P2 entregues pelo agente attr-p2-statistics ao benchmark ATTR P2, concluir a integração fail-closed do runner/seal, executar a matriz selada de cinco seeds e seis braços, avaliar ID/shift/OOD somente após congelar os 30 checkpoints, calcular bootstrap e gates, e publicar JSON, PNG/SVG, dashboard e documentação sem claims além da evidência.

## Summary

ATTR P2 concluído com 30/30 checkpoints verificados antes da abertura de test e 90 arquivos selados de prediction. G0 e G1 passaram; G2 falhou; G3/G4 não foram avaliados; G5 falhou fechado. O par ASM-X/Transformer ficou praticamente empatado em AUPRC H8 ID (+0,0007, IC95 [-0,0340; +0,0214]), enquanto ASM-X teve NLL menor. O predictive gate não passou e nenhum claim de safety ou causalidade foi feito.

## Modified files

- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md)
- [docs/benchmarks/asm_transformer_transition_risk/p2/README.md](../benchmarks/asm_transformer_transition_risk/p2/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/p2/dataset_seal.json](../benchmarks/asm_transformer_transition_risk/p2/dataset_seal.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/index.html](../benchmarks/asm_transformer_transition_risk/p2/index.html)
- [docs/benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.png](../benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.svg](../benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.png](../benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.svg](../benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/summary.json](../benchmarks/asm_transformer_transition_risk/p2/summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.png](../benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.svg](../benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/test_open_event.json](../benchmarks/asm_transformer_transition_risk/p2/test_open_event.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/test_spec_preseal.json](../benchmarks/asm_transformer_transition_risk/p2/test_spec_preseal.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/training_implementation_manifest.json](../benchmarks/asm_transformer_transition_risk/p2/training_implementation_manifest.json)
- [scripts/run_attr_p2.py](../../scripts/run_attr_p2.py)
- [src/aletheion_state_models/benchmarks/transition_risk/__init__.py](../../src/aletheion_state_models/benchmarks/transition_risk/__init__.py)
- [src/aletheion_state_models/benchmarks/transition_risk/dataset.py](../../src/aletheion_state_models/benchmarks/transition_risk/dataset.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_checkpoint.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_checkpoint.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_evaluation.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_evaluation.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_models.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_models.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_plots.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_plots.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_runner.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_runner.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_seal.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_seal.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_statistics.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_statistics.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_summary.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_summary.py)
- [tests/test_transition_risk_p2_checkpoint.py](../../tests/test_transition_risk_p2_checkpoint.py)
- [tests/test_transition_risk_p2_evaluation.py](../../tests/test_transition_risk_p2_evaluation.py)
- [tests/test_transition_risk_p2_runner.py](../../tests/test_transition_risk_p2_runner.py)
- [tests/test_transition_risk_p2_seal.py](../../tests/test_transition_risk_p2_seal.py)
- [tests/test_transition_risk_p2_statistics.py](../../tests/test_transition_risk_p2_statistics.py)
- [tests/test_transition_risk_p2_summary.py](../../tests/test_transition_risk_p2_summary.py)
- [runs/attr_p2](../../runs/attr_p2)
- [docs/report/0029_conclusao-attr-p2-selado_2026-09-01.md](0029_conclusao-attr-p2-selado_2026-09-01.md)

## Changes

- Integrei agregação por horizonte, bootstrap hierárquico pareado seed→world→episode, deltas AUPRC/Brier/NLL e gates fail-closed.
- Implementei checkpoints terminais atômicos, matriz exata de 30 braços-seeds, preseal imutável, seal SHA-256 e materialização de test somente após verificação completa.
- Treinei seis braços em seeds 29, 43, 71, 89 e 107 por 1.000 updates, preservando validation-only calibration e test fechado durante treino.
- Avaliei 32 mundos × quatro episódios em ID, shift e OOD para cada checkpoint e preservei records JSONL auditáveis.
- Registrei transparentemente a primeira falha de integração pós-abertura e o patch apenas de orquestração; nenhum prediction havia sido gravado e modelos/checkpoints não mudaram.
- Publiquei summary, dashboard offline, três gráficos em PNG/SVG e documentação bilíngue com todos os gates, inclusive falhas e não avaliados.

## Validation

- 30 checkpoints, 30 resultados e 30 predictions validation — hashes e metadados verificados; test_opened=false durante treino
- 90/90 prediction files de test — 128 episódios cada, JSON válido e sem NaN/Infinity
- python scripts/run_attr_p2.py --phase summarize --device cpu — concluído; G0/G1 pass, G2/G5 fail, G3/G4 not evaluated
- python -m pytest -q — 267 passed, cinco warnings conhecidos
- python -m compileall -q src world_model transformer scripts — passed
- git diff --check — passed
- solid-source-modularity audit — 288 compliant, 8 documented pre-existing exceptions, 4 pre-existing violations; all P2 authored modules <=300 lines
